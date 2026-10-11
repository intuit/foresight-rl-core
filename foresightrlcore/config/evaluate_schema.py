"""Configuration schema for the ``evaluate`` operation.

Follows :mod:`foresightrlcore.config.optimizer_schema` and reuses the same
:class:`JobConfig`, :class:`ExecutionConfig`, :class:`OutputConfig`,
:class:`ContainerConfig`, :class:`SecretsConfig`, :class:`LLMClientsConfig`,
:class:`VpcConfig` and :class:`VllmServingConfig` building blocks.

The design goal is that **adding a benchmark is a YAML entry**. A
:class:`BenchmarkSpec` therefore has exactly one required field (``name``);
everything else either comes from the benchmark catalog or has a default. A
config entry that names a catalog benchmark and overrides one field is written
as exactly that -- the two are merged, entry wins.

Two infra profiles are resolved from the candidate models, as in the optimizer:

* **API-only** (every candidate is ``bedrock``/``genos``/``echo``): CPU
  instance, lightweight image.
* **Local-serving** (any candidate is ``provider: local``): GPU instance so
  ``vllm serve`` has hardware.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from foresightrlcore.common.thinking import DEFAULT_THINKING_TAGS
from foresightrlcore.config.perf_schema import VllmServingConfig
from foresightrlcore.config.schema import (
    ContainerConfig,
    ExecutionConfig,
    JobConfig,
    LLMClientsConfig,
    OutputConfig,
    SecretsConfig,
    VpcConfig,
)
from foresightrlcore.evaluate.spec import ModelReportInfo

# Default instance types per infra profile (used when instance_type is unset).
DEFAULT_CPU_INSTANCE_TYPE = "ml.m5.2xlarge"
DEFAULT_GPU_INSTANCE_TYPE = "ml.g5.12xlarge"

#: Providers that can back an evaluate candidate model.
MODEL_PROVIDERS = ("bedrock", "genos", "local", "echo")

#: Providers whose candidates are served on this machine, and so need a GPU.
#: Stated as the exception rather than as its complement so that a provider
#: added later defaults to CPU sizing -- which is both the cheap answer and the
#: right one for anything reached over HTTP.
GPU_PROVIDERS = ("local",)


class BenchmarkSource(BaseModel):
    """Where a benchmark's code comes from when it is not in-package.

    Exactly one of ``git`` / ``s3`` / ``path`` must be set. This is what lets a
    benchmark that lives in its own repository run without being vendored.
    """

    model_config = ConfigDict(extra="forbid")

    git: Optional[str] = Field(
        default=None, description="Git URL to clone the benchmark from."
    )
    ref: Optional[str] = Field(
        default=None,
        description="Git branch, tag, or commit. Pin this: an unpinned "
        "benchmark makes results irreproducible.",
    )
    s3: Optional[str] = Field(
        default=None, description="s3:// prefix holding the benchmark tree."
    )
    path: Optional[str] = Field(
        default=None, description="Local directory holding the benchmark tree."
    )
    subdir: Optional[str] = Field(
        default=None,
        description="Subdirectory within the fetched tree to treat as the root.",
    )

    @model_validator(mode="after")
    def _exactly_one(self) -> "BenchmarkSource":
        set_fields = [f for f in ("git", "s3", "path") if getattr(self, f)]
        if len(set_fields) != 1:
            raise ValueError(
                "benchmark.source must set exactly one of git/s3/path, got: "
                f"{set_fields or 'none'}"
            )
        return self


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively layer ``overrides`` onto ``base``; mappings merge, rest wins."""
    merged = dict(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


class EvalDatasetSpec(BaseModel):
    """The data a benchmark reads. Loader-specific keys are allowed and
    forwarded verbatim, so a loader can take options the core never sees."""

    model_config = ConfigDict(extra="allow")

    loader: str = Field(
        default="csv",
        description="Registered dataset-loader name (csv | jsonl | parquet | "
        "scenario_dir | inline).",
    )
    path: Optional[str] = Field(
        default=None,
        description="Dataset path: local filesystem path or s3:// URI "
        "(downloaded before the run / inside the job).",
    )
    split: Optional[str] = Field(
        default=None,
        description="Which entry of `splits` to use. Defaults to the sole "
        "split when only one is defined.",
    )
    splits: Dict[str, str] = Field(
        default_factory=dict,
        description="Named split -> path mapping, e.g. {dev: s3://.../dev.csv, "
        "test: s3://.../test.csv}. `split` selects one.",
    )
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        description="Evaluate at most N records (after sampling). Useful for "
        "smoke runs; omit for a full benchmark.",
    )

    def resolved_path(self) -> Optional[str]:
        """Return the path for the selected split, or ``path``.

        Raises:
            ValueError: If ``split`` names an entry that does not exist, or if
                splits are defined but none was selected unambiguously.
        """
        if not self.splits:
            return self.path
        if self.split:
            if self.split not in self.splits:
                raise ValueError(
                    f"dataset.split={self.split!r} is not in splits "
                    f"{sorted(self.splits)}"
                )
            return self.splits[self.split]
        if len(self.splits) == 1:
            return next(iter(self.splits.values()))
        raise ValueError(
            f"dataset defines splits {sorted(self.splits)} but no `split` was "
            "selected"
        )

    def as_loader_spec(self) -> Dict[str, Any]:
        """Return the full mapping (known + extra keys) for the loader."""
        spec = self.model_dump()
        spec["path"] = self.resolved_path()
        return spec


class SampleConfig(BaseModel):
    """Deterministic sampling parameters."""

    model_config = ConfigDict(extra="forbid")

    size: int = Field(default=50, ge=1, description="Number of records to sample.")
    seed: int = Field(default=13, description="RNG seed for reproducible sampling.")
    stratify_by: str = Field(
        default="category",
        description="Record meta key to stratify the sample by (proportional "
        "allocation). Falls back to a single stratum when absent.",
    )


class BenchmarkSpec(BaseModel):
    """One benchmark. ``name`` is the only required field.

    A spec either defines a benchmark inline or names one from the catalog and
    overrides parts of it. Unknown keys are allowed and forwarded to the
    benchmark type via :meth:`type_params`, so a benchmark can accept
    parameters the core has never heard of.
    """

    model_config = ConfigDict(extra="allow")

    name: str = Field(
        description="Benchmark name: a catalog key, or a new name defined here."
    )
    type: Optional[str] = Field(
        default=None,
        description="Registered benchmark type (mcq | generative | "
        "external_process | plugin | lm_eval). Defaults to 'mcq' after the "
        "catalog merge.",
    )
    display_name: Optional[str] = Field(
        default=None, description="Human-readable name used in reports."
    )
    source: Optional[BenchmarkSource] = Field(
        default=None,
        description="External code to stage before running (git / s3 / path).",
    )
    dataset: Optional[EvalDatasetSpec] = Field(
        default=None, description="Dataset + split selection."
    )
    prompt_set: Optional[str] = Field(
        default=None, description="Registered prompt-set name (record-tier types)."
    )
    prompts_path: Optional[str] = Field(
        default=None,
        description="Explicit prompts YAML path (local or s3://) overriding the "
        "prompt set's default file.",
    )
    metrics: List[str] = Field(
        default_factory=list,
        description="Registered metric names. Defaults per benchmark type when "
        "left empty.",
    )
    command: List[str] = Field(
        default_factory=list,
        description="argv for harness-tier types. Entries may reference the "
        "contract env vars, e.g. '$EVALUATE_OUTPUT_FILE'.",
    )
    env: Dict[str, str] = Field(
        default_factory=dict,
        description="Extra environment for the benchmark process, layered over "
        "the endpoint contract vars. Use this to alias contract names onto "
        "whatever the benchmark already expects.",
    )
    isolate: str = Field(
        default="none",
        description="Isolation for `command`: 'venv' creates a dedicated "
        "interpreter and installs the benchmark's own requirements; 'none' "
        "runs in the job's interpreter.",
    )
    requirements: List[str] = Field(
        default_factory=list,
        description="pip requirements installed into the isolated venv "
        "(isolate=venv only).",
    )
    timeout_seconds: int = Field(
        default=3600,
        ge=1,
        description="Wall-clock limit for one benchmark run (one trial).",
    )
    trials: Optional[int] = Field(
        default=None,
        ge=1,
        description="Repeat count for this benchmark, overriding the run-level "
        "`trials`. Use >1 for benchmarks with stochastic scenarios.",
    )
    models: List[str] = Field(
        default_factory=list,
        description="Restrict this benchmark to these model names or model "
        "groups. Empty means every model in the run.",
    )
    sample: Optional[SampleConfig] = Field(
        default=None, description="Deterministic sub-sampling of the dataset."
    )
    failure_tolerance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of records allowed to error before the benchmark "
        "score is reported as invalid (-1) rather than computed on a partial "
        "population.",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Benchmark-type-specific parameters, forwarded verbatim.",
    )

    @field_validator("isolate")
    @classmethod
    def _validate_isolate(cls, v: str) -> str:
        allowed = {"none", "venv"}
        if v not in allowed:
            raise ValueError(f"isolate must be one of {sorted(allowed)}, got: {v!r}")
        return v

    def type_params(self) -> Dict[str, Any]:
        """Return ``params`` plus any unknown top-level keys.

        Extra keys are a convenience: a benchmark can be configured with
        ``number_of_choices: 4`` at the top level instead of nesting it under
        ``params``.
        """
        known = set(type(self).model_fields)
        extras = {k: v for k, v in self.model_dump().items() if k not in known}
        return {**extras, **self.params}

    def merged_over(self, base: "BenchmarkSpec") -> "BenchmarkSpec":
        """Return this spec layered over a catalog ``base`` entry.

        Only fields explicitly set here override the catalog; everything else
        is inherited. This is what makes "use the catalog benchmark but with
        `split: dev`" a two-line config entry.

        Nested blocks merge key by key, so overriding ``dataset.limit`` keeps
        the catalog's ``dataset.path``. Lists replace wholesale -- a config
        naming two metrics means those two, not those two plus the catalog's.
        """
        overrides = self.model_dump(exclude_unset=True)
        overrides.pop("name", None)
        data = _deep_merge(base.model_dump(), overrides)
        return BenchmarkSpec.model_validate(data)


class AgentRuntimeConfig(BaseModel):
    """How to drive an *agentic* candidate rather than call it directly.

    Agentic benchmarks do not send a prompt and read a completion -- they run an
    agent that uses tools in a workspace. The agent is a program; this block
    says which program and how to invoke it. The agent still reaches the model
    through the endpoint contract, so no provider knowledge leaks in.
    """

    model_config = ConfigDict(extra="allow")

    kind: str = Field(
        description="Agent runtime identifier, e.g. 'claude_code'. Benchmarks "
        "read this to decide how to launch the agent."
    )
    command: List[str] = Field(
        default_factory=list,
        description="argv used to invoke the agent CLI. Empty lets the "
        "benchmark pick its own default for `kind`.",
    )
    env: Dict[str, str] = Field(
        default_factory=dict,
        description="Extra environment for the agent process.",
    )
    max_turns: Optional[int] = Field(
        default=None, ge=1, description="Cap on agent turns per scenario."
    )


class LeaderboardInfo(BaseModel):
    """Descriptive metadata about a candidate, for tracking sinks.

    None of it affects scoring -- it exists so a published run can be read
    without opening the config that produced it.

    ``extra="allow"``: a sink that wants a tag this model does not name
    should not require a schema change here. Those extra keys are forwarded
    as tags.
    """

    model_config = ConfigDict(extra="allow")

    model_size: Optional[str] = Field(
        default=None, description="Parameter count as published, e.g. '8B'."
    )
    model_type: Optional[str] = Field(
        default=None, description="e.g. 'pretrained', 'instruct', 'finetuned'."
    )
    context_window: Optional[str] = Field(
        default=None, description="Context length as published, e.g. '128K'."
    )

    def as_tags(self) -> Dict[str, str]:
        """Return the set fields as string tags, dropping the unset ones."""
        return {
            key: str(value)
            for key, value in self.model_dump(exclude_none=True).items()
            if str(value) != ""
        }


class ThinkingConfig(BaseModel):
    """Declares that a candidate narrates before it answers.

    Set this and the runner removes the narration before any metric sees the
    response. Leave it unset and the response is scored verbatim, which is
    correct for a non-reasoning model and wrong for a reasoning one: the
    scratchpad is where a model records the answers it considered and
    discarded, so an answer extractor reading it finds the rejected one.

    Presence means enabled -- writing the block at all is the intent. The
    ``enabled`` field exists so a config can keep its tag list while switching
    the behaviour off.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True, description="Strip reasoning tags from this model's output."
    )
    tags: List[str] = Field(
        default_factory=lambda: list(DEFAULT_THINKING_TAGS),
        description="Tag names wrapping the reasoning, with or without angle "
        "brackets. Defaults to 'think' and 'reasoning'; override for a model "
        "that uses its own.",
    )

    @field_validator("tags")
    @classmethod
    def _non_empty(cls, v: List[str]) -> List[str]:
        if not [t for t in v if str(t).strip().strip("<>/").strip()]:
            raise ValueError("model.thinking.tags must name at least one tag")
        return v


class EvaluateModelConfig(BaseModel):
    """A candidate model + provider + token pricing (USD per 1K tokens)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Logical candidate name (used in the results table). For "
        "local serving this is also the HF id / S3 prefix of the model to "
        "serve, unless `base_model` gives one separately."
    )
    base_model: Optional[str] = Field(
        default=None,
        description="HF id / S3 prefix of the weights to serve, when `name` is "
        "a label rather than a path. Needed to compare candidates that share "
        "weights -- a base model against the same base plus a LoRA adapter -- "
        "since `name` has to stay unique to keep them apart in the results.",
    )
    display_name: Optional[str] = Field(
        default=None,
        description="Human-readable label for reports and leaderboards. "
        "Defaults to `name`. Set this whenever `name` is an S3 prefix, which "
        "is otherwise reproduced verbatim as a table column header.",
    )
    leaderboard: Optional[LeaderboardInfo] = Field(
        default=None,
        description="Optional model metadata forwarded to tracking reporters "
        "as tags. Omit it and the run still publishes, just without these tags.",
    )
    provider: str = Field(
        default="bedrock",
        description="Inference provider: 'bedrock' | 'local' | 'genos' "
        "(requires foresightrl-intuit) | 'echo' (offline, for dry runs).",
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Provider model id. Defaults to `name` when unset.",
    )
    price_in: float = Field(default=0.0, ge=0.0, description="USD per 1K input tokens.")
    price_out: float = Field(
        default=0.0, ge=0.0, description="USD per 1K output tokens."
    )
    max_tokens: int = Field(
        default=4096, ge=1, description="Max output tokens per call."
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling cutoff. Omitted from the request when "
        "unset, leaving the provider's own default in force.",
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Sample from the K most likely tokens. Not part of the "
        "OpenAI schema, so it travels as a provider extension: vLLM reads it "
        "from the request body, Bedrock forwards it in "
        "additionalModelRequestFields. A provider that does not support it "
        "ignores it rather than failing. -1 is vLLM's spelling of 'no limit' "
        "and is sent as-is, so use it only for a vLLM-served model.",
    )
    repetition_penalty: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Penalise tokens already present; >1 discourages repeats. "
        "Like `top_k`, not in the OpenAI schema, so it travels as a provider "
        "extension. It belongs beside the other two because temperature and "
        "top_p alone do not pin a distribution: the matching baselines were "
        "run at 1.1, and that is what stops a reasoning model looping until it "
        "exhausts its token budget.",
    )
    extra_body: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra JSON fields merged into every request body, for a "
        "provider option with no field of its own -- `chat_template_kwargs` "
        "carrying `reasoning_effort`, for instance. Applied last, so it can "
        "override any field above when a provider spells one differently.",
    )
    thinking: Optional[ThinkingConfig] = Field(
        default=None,
        description="Set for a reasoning model, so its scratchpad is removed "
        "before scoring. Omit for a model that answers directly.",
    )
    truncate_prompt_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Truncate each prompt to the last N tokens (provider=local "
        "only; ignored by API providers).",
    )
    region: Optional[str] = Field(
        default=None,
        description="AWS region (provider=bedrock). Falls back to "
        "AWS_DEFAULT_REGION.",
    )
    serving: Optional[VllmServingConfig] = Field(
        default=None,
        description="vLLM serving parameters (required for provider=local).",
    )
    weights_path: Optional[str] = Field(
        default=None,
        description="Local directory to serve the weights from. Populated at "
        "runtime when SageMaker has staged an S3 model; leave unset in YAML.",
    )
    adapter: Optional[str] = Field(
        default=None,
        description="LoRA adapter to serve on top of `name` (provider=local "
        "only): an HF id, an S3 prefix, or a local directory. `name` stays the "
        "base model. Evaluating a fine-tune this way costs no merge job, but "
        "not every architecture can serve an adapter -- vLLM's LoRA kernel "
        "rejects some layer layouts, in which case the adapter has to be "
        "merged into the base weights and served as an ordinary model.",
    )
    adapter_path: Optional[str] = Field(
        default=None,
        description="Local directory the adapter was staged to. Populated at "
        "runtime, like `weights_path`; leave unset in YAML.",
    )
    agent_runtime: Optional[AgentRuntimeConfig] = Field(
        default=None,
        description="Drive this candidate as an agent (agentic benchmarks) "
        "instead of calling it directly.",
    )

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        if v not in MODEL_PROVIDERS:
            raise ValueError(
                f"model.provider must be one of {sorted(MODEL_PROVIDERS)}, got: {v!r}"
            )
        return v

    @field_validator("top_k")
    @classmethod
    def _validate_top_k(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v != -1 and v < 1:
            raise ValueError(f"model.top_k must be >= 1, or -1 for no limit; got {v}")
        return v

    @model_validator(mode="after")
    def _defaults(self) -> "EvaluateModelConfig":
        if self.model_id is None:
            self.model_id = self.name
        if self.provider == "local" and self.serving is None:
            self.serving = VllmServingConfig()
        if self.adapter and self.provider != "local":
            raise ValueError(
                f"model {self.name!r}: adapter requires provider 'local' -- an "
                f"API provider serves whatever weights it was deployed with, "
                f"so there is nothing to attach an adapter to (got "
                f"provider={self.provider!r})."
            )
        if self.base_model and self.provider != "local":
            raise ValueError(
                f"model {self.name!r}: base_model requires provider 'local' -- "
                f"nothing serves it for provider={self.provider!r}, yet it "
                f"would still change the candidate's resume fingerprint."
            )
        # An HF id, S3 prefix or local directory all contain a "/"; a bare
        # name is a label, and vllm serve would only reject it on the GPU.
        if self.adapter and not self.base_model and "/" not in self.name:
            raise ValueError(
                f"model {self.name!r}: adapter is set but name is a label, not "
                f"weights, and base_model is unset -- set base_model to the HF "
                f"id or S3 prefix the adapter was trained on."
            )
        return self

    @property
    def resolved_model_id(self) -> str:
        return self.model_id or self.name

    @property
    def thinking_tags(self) -> Optional[List[str]]:
        """Tags to strip from this model's output, or ``None`` to leave it be."""
        if self.thinking is None or not self.thinking.enabled:
            return None
        return list(self.thinking.tags)

    @property
    def effective_temperature(self) -> Optional[float]:
        """Temperature to send, or ``None`` to leave it to the provider.

        ``temperature`` defaults to 0.0 rather than being unset, because a
        benchmark wants greedy decoding unless told otherwise. That default is
        sent on every request -- which is fine until a config also asks for
        ``top_p``, because **Anthropic models reject a request carrying both**
        (`temperature` and `top_p` cannot both be specified for this model).
        A config that mentioned only ``top_p`` would then fail every call, for
        a parameter it never set.

        So the implicit default yields to an explicit ``top_p``. Setting both
        deliberately still sends both, and a provider that dislikes the
        combination says so -- that is the operator's choice to make, and
        validation warns about it before the job starts.
        """
        if "temperature" in self.model_fields_set:
            return self.temperature
        if self.top_p is not None:
            return None
        return self.temperature

    @property
    def weights_source(self) -> str:
        """Where the weights come from, as configured: ``base_model`` or ``name``."""
        return self.base_model or self.name

    @property
    def serving_path(self) -> str:
        """What ``vllm serve`` is pointed at: staged weights, else as configured.

        Distinct from :attr:`name`, which stays the candidate's label so the
        leaderboard and the resume keys read the same whether the run happened
        on a laptop or on SageMaker.
        """
        return self.weights_path or self.weights_source

    @property
    def lora_adapter(self) -> Optional[str]:
        """What ``vllm serve`` loads as an adapter: staged copy, else as written.

        The same staged-versus-declared split as :attr:`serving_path`, for the
        same reason: the staged path is where this job happens to have put the
        files, while the declared value is part of what was measured.
        """
        return self.adapter_path or self.adapter

    @property
    def is_agentic(self) -> bool:
        return self.agent_runtime is not None

    @property
    def report_label(self) -> str:
        """The label reports render. Never used as a key -- see `name`."""
        return self.display_name or self.name

    def report_info(self) -> ModelReportInfo:
        """Presentation + metadata for this candidate, for the result document."""
        return ModelReportInfo(
            display_name=self.report_label,
            leaderboard=self.leaderboard.as_tags() if self.leaderboard else {},
        )


class ReporterSpec(BaseModel):
    """A reporter to run, with its parameters.

    Written either as a bare name (``json``) or as a mapping (``{type: aleta,
    params: {...}}``). The bare form stays valid because most reporters take no
    configuration; the mapping exists because some sinks need an address to
    publish to, and there was previously no way to give them one.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(description="Registered reporter name.")
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Keyword arguments for the reporter."
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_name(cls, value: Any) -> Any:
        return {"type": value} if isinstance(value, str) else value


class RetryConfig(BaseModel):
    """Resilient-call parameters for inference and harness invocations."""

    model_config = ConfigDict(extra="forbid")

    max_retries: int = Field(default=6, ge=1)
    base_delay: float = Field(
        default=2.0, ge=0.0, description="Base seconds for exponential backoff."
    )


class CheckpointConfig(BaseModel):
    """Incremental checkpoint / resume settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True)
    every_n_rows: int = Field(
        default=20,
        ge=1,
        description="Flush the checkpoint JSONL (and sync to S3 if "
        "applicable) every N scored records. This counts records, not "
        "trials -- it once counted trials despite the name, which meant a "
        "run with fewer trials than this number never synced until it "
        "finished, and an interrupted benchmark lost every record it had "
        "generated. A completed trial always flushes regardless. Lower is "
        "safer but costs more: each sync re-uploads the whole file, so the "
        "bytes grow with the square of the record count.",
    )


class ConfidenceConfig(BaseModel):
    """Multi-trial statistics.

    Benchmarks with stochastic scenarios are not measured by a single run, so
    ``trials > 1`` reports a mean with a **t-distribution** interval (the
    normal approximation is wrong at these sample sizes).
    """

    model_config = ConfigDict(extra="forbid")

    level: float = Field(
        default=0.95,
        gt=0.0,
        lt=1.0,
        description="Confidence level for the reported interval.",
    )
    retrial_flaky: bool = Field(
        default=False,
        description="After the planned trials, re-run only the sub-scores that "
        "disagreed across trials, instead of the whole benchmark.",
    )
    max_retrials: int = Field(
        default=2, ge=0, description="Cap on targeted re-trial rounds."
    )


class EvaluateRunConfig(BaseModel):
    """A single evaluate run: benchmarks x models, repeated over trials."""

    model_config = ConfigDict(extra="forbid")

    benchmarks: List[BenchmarkSpec] = Field(
        description="Benchmarks to run. One entry per benchmark."
    )
    models: List[EvaluateModelConfig] = Field(description="Candidate models.")
    model_groups: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Named model groups, e.g. {frontier: [claude, gpt]}. A "
        "benchmark's `models` list may name a group instead of each model.",
    )
    trials: int = Field(
        default=1,
        ge=1,
        description="Default repeat count per (benchmark, model). Per-benchmark "
        "`trials` overrides this.",
    )
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    concurrency: int = Field(
        default=4,
        ge=1,
        description="Thread-pool size for concurrent per-record evaluation "
        "(record-tier benchmarks).",
    )
    store_responses: bool = Field(
        default=False,
        description="Keep each model response on its result row. Off by "
        "default because storing every generation is expensive and, on a run "
        "that behaved, useless. Turn it on when a score needs explaining: a "
        "zero can mean the model was wrong, that it answered in a format the "
        "metric cannot read, or that reasoning-tag stripping took the answer "
        "with the narration, and nothing in the numbers tells those apart. "
        "Smoke configs should set it -- a gate that fails without saying why "
        "costs more than the storage does.",
    )
    store_responses_max_chars: int = Field(
        default=8000,
        ge=0,
        description="Truncate a stored response beyond this length. 0 keeps "
        "it whole. The default is generous enough for a reasoning model's "
        "scratchpad, which is the case worth seeing in full.",
    )
    retry: RetryConfig = Field(default_factory=RetryConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    reporters: List[ReporterSpec] = Field(
        default_factory=lambda: [ReporterSpec(type="table"), ReporterSpec(type="json")],
        description="Reporters to run over the results. Each entry is a "
        "registered reporter name, or a mapping with `type` and `params`.",
    )
    judge_clients: Optional[LLMClientsConfig] = Field(
        default=None,
        description="Provider-agnostic LLM client mapping used by judge metrics "
        "(required when a judge metric is configured).",
    )
    catalog_paths: List[str] = Field(
        default_factory=list,
        description="Extra benchmark-catalog YAML files to load, on top of the "
        "built-in catalog and any catalogs registered by extensions.",
    )

    role: Optional[str] = Field(
        default=None,
        description="SageMaker execution role ARN. Auto-detected if not set.",
    )

    # SageMaker sizing. instance_type has no hard default: it is resolved from
    # requires_gpu() when unset.
    instance_type: Optional[str] = Field(
        default=None,
        description="SageMaker instance type. When unset, resolved from whether "
        "any candidate serves locally (CPU for API-only, GPU otherwise).",
    )
    instance_count: int = Field(default=1, ge=1)
    volume_size_gb: int = Field(default=200, ge=30)
    training_plan_arn: Optional[str] = Field(
        default=None,
        description="SageMaker training-plan (reserved capacity) ARN.",
    )
    encrypt_inter_container_traffic: bool = Field(
        default=True,
        description="Encrypt inter-container traffic (TLS) between compute "
        "instances. Maps to SageMaker EnableInterContainerTrafficEncryption. "
        "Required by Intuit ACP policy IS031.1.",
    )
    vpc: Optional[VpcConfig] = Field(
        default=None,
        description="VPC config (subnets + security groups). Required to reach "
        "an internal network from the job.",
    )

    @field_validator("benchmarks", "models")
    @classmethod
    def _non_empty(cls, v: List[Any]) -> List[Any]:
        if not v:
            raise ValueError("must contain at least one entry")
        return v

    def model_names(self) -> List[str]:
        return [m.name for m in self.models]

    def benchmark_names(self) -> List[str]:
        return [b.name for b in self.benchmarks]

    def resolve_models_for(self, benchmark: BenchmarkSpec) -> List[EvaluateModelConfig]:
        """Return the candidate models ``benchmark`` should run against.

        Expands model-group names; an empty selector means every model.

        Raises:
            ValueError: If a selector matches neither a model nor a group.
        """
        if not benchmark.models:
            return list(self.models)

        wanted: List[str] = []
        for selector in benchmark.models:
            if selector in self.model_groups:
                wanted.extend(self.model_groups[selector])
            else:
                wanted.append(selector)

        by_name = {m.name: m for m in self.models}
        missing = [name for name in wanted if name not in by_name]
        if missing:
            raise ValueError(
                f"benchmark {benchmark.name!r} selects unknown model(s) "
                f"{missing}; known models: {sorted(by_name)}"
            )
        # Preserve declaration order and drop duplicates from overlapping groups.
        seen: set[str] = set()
        return [
            by_name[name]
            for name in wanted
            if not (name in seen or seen.add(name))  # type: ignore[func-returns-value]
        ]

    def trials_for(self, benchmark: BenchmarkSpec) -> int:
        return benchmark.trials if benchmark.trials is not None else self.trials


class EvaluateJobConfig(BaseModel):
    """Top-level configuration for the ``evaluate`` operation."""

    _source_path: str = ""

    version: int = Field(default=1)
    job: JobConfig
    evaluate: EvaluateRunConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    environment: Dict[str, str] = Field(default_factory=dict)
    container: ContainerConfig = Field(default_factory=ContainerConfig)
    secrets: Optional[SecretsConfig] = Field(default=None)

    @model_validator(mode="after")
    def _validate_root(self) -> "EvaluateJobConfig":
        valid_backends = {"aws_training_job", "local"}
        if self.execution.type not in valid_backends:
            raise ValueError(
                f"evaluate does not support execution.type "
                f"'{self.execution.type}'. Supported: {sorted(valid_backends)}"
            )
        return self

    @model_validator(mode="after")
    def _default_checkpoint_path(self) -> "EvaluateJobConfig":
        """Give checkpointing somewhere durable to live, so resume works.

        ``checkpoint.enabled`` defaults to true, and the runner honours it:
        it writes its checkpoint to the local directory all run long and reads
        that directory on startup. What makes those files outlive the job is a
        *different* field in a *different* section -- ``output.s3_path``'s
        neighbour ``checkpoint_s3_path`` -- which is what tells the launcher to
        attach a SageMaker ``CheckpointConfig``, and that is what downloads the
        previous run's rows into the container before the runner looks for
        them.

        Leave it unset and everything appears configured: no error, no warning,
        checkpoint files being written on schedule -- to container disk that is
        destroyed with the job. The cost is only discovered when a long run
        dies and there is nothing to resume from, which is precisely when it is
        most expensive. Deriving a default removes the trap; a config that
        wants its own location still sets the field and wins.

        Resuming by default is safe because of the fingerprint: a row is reused
        only if the benchmark spec and model config that produced it still hash
        the same, so an edited prompt or a swapped metric re-runs the trials it
        affects. To force a clean measurement regardless, set
        ``checkpoint.enabled: false``.

        Only for ``aws_training_job`` with an S3 output. A local run's output
        is a directory on disk that needs no syncing, and deriving an ``s3://``
        URI from a local path would be nonsense.
        """
        if (
            self.evaluate.checkpoint.enabled
            and not self.output.checkpoint_s3_path
            and self.execution.type == "aws_training_job"
            and self.output.s3_path
            and self.output.s3_path.startswith("s3://")
        ):
            base = self.output.s3_path.rstrip("/")
            self.output.checkpoint_s3_path = f"{base}/checkpoints/"
        return self

    def requires_gpu(self) -> bool:
        """True iff any candidate is served here rather than called over HTTP."""
        return any(m.provider in GPU_PROVIDERS for m in self.evaluate.models)

    def resolved_instance_type(self) -> str:
        """Resolve the SageMaker instance type (explicit value wins)."""
        if self.evaluate.instance_type:
            return self.evaluate.instance_type
        return (
            DEFAULT_GPU_INSTANCE_TYPE
            if self.requires_gpu()
            else DEFAULT_CPU_INSTANCE_TYPE
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvaluateJobConfig":
        """Load and validate config from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        instance = cls.model_validate(raw)
        instance._source_path = str(path)
        return instance

    def to_yaml(self, path: str | Path) -> None:
        """Serialize config to a YAML file, keeping only what was authored.

        ``exclude_unset`` is load-bearing, not tidiness. A benchmark entry
        naming a catalog benchmark is merged *over* the catalog, and lists
        replace rather than merge. Dumping defaults would turn every field the
        author left out into an explicit empty value, so a remote job would
        read ``metrics: []`` and score with the built-in default instead of the
        metrics the catalog specifies -- silently, and only when run remotely.
        """
        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(mode="json", exclude_none=True, exclude_unset=True),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    def merge_overrides(self, **kwargs: Any) -> "EvaluateJobConfig":
        """Return a new config with dotted-key CLI overrides merged in.

        Dumps only what was set so an override does not promote every default
        to an explicit value -- see :meth:`to_yaml` for why that matters once a
        catalog is involved.
        """
        data = self.model_dump(exclude_unset=True)

        shortcut_map = {
            "output_path": ("output", "s3_path"),
        }

        for key, value in kwargs.items():
            if value is None:
                continue
            if key in shortcut_map:
                section, field = shortcut_map[key]
                # setdefault, not indexing: dumping only what was set means a
                # section the author left at its defaults is simply absent.
                data.setdefault(section, {})[field] = value
            elif "." in key:
                parts = key.split(".")
                target = data
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = value

        return EvaluateJobConfig.model_validate(data)


def local_model_channels(
    models: Sequence["EvaluateModelConfig"],
) -> List[Tuple[str, "EvaluateModelConfig"]]:
    """Pair every S3-backed local candidate with its SageMaker input channel.

    A locally served model needs its weights on disk before ``vllm serve``
    starts, so the launcher mounts each one as an input channel and the
    container-side runner swaps the ``s3://`` URI for the mount path. Both
    sides call this function: if the naming lived in two places, weights would
    be mounted under one name and looked for under another, and the failure
    would only ever appear inside a GPU job.

    Channels are keyed on the prefix, not on the candidate, so two candidates
    sharing weights -- a base model and the same base plus an adapter -- mount
    one copy between them instead of downloading tens of gigabytes twice.

    Models served through an API provider are skipped -- they have no weights.
    """
    return _channels_by_uri(models, lambda m: m.weights_source, "model")


def _channels_by_uri(
    models: Sequence["EvaluateModelConfig"],
    uri_of: "Callable[[EvaluateModelConfig], Optional[str]]",
    prefix: str,
) -> List[Tuple[str, "EvaluateModelConfig"]]:
    """Assign one channel per distinct S3 URI, paired with every model using it."""
    assigned: Dict[str, str] = {}
    channels: List[Tuple[str, "EvaluateModelConfig"]] = []
    for model in models:
        uri = uri_of(model) or ""
        if model.provider != "local" or not uri.startswith("s3://"):
            continue
        if uri not in assigned:
            index = len(assigned)
            assigned[uri] = prefix if index == 0 else f"{prefix}{index}"
        channels.append((assigned[uri], model))
    return channels


def local_adapter_channels(
    models: Sequence["EvaluateModelConfig"],
) -> List[Tuple[str, "EvaluateModelConfig"]]:
    """Pair every S3-backed LoRA adapter with its SageMaker input channel.

    The adapter travels in its own channel rather than inside the weights
    channel because the two come from unrelated prefixes: the base model is
    published once, the adapter is written per training run. Numbered
    independently of the weight channels, so adding a base-only candidate never
    renames an adapter channel out from under a resumed run.
    """
    return _channels_by_uri(models, lambda m: m.adapter, "adapter")


def channel_mount_path(channel: str) -> str:
    """Where SageMaker puts ``channel``, honouring the env var it sets."""
    return os.environ.get(
        f"SM_CHANNEL_{channel.upper()}", f"/opt/ml/input/data/{channel}"
    )


__all__ = [
    "channel_mount_path",
    "local_adapter_channels",
    "local_model_channels",
    "DEFAULT_CPU_INSTANCE_TYPE",
    "DEFAULT_GPU_INSTANCE_TYPE",
    "GPU_PROVIDERS",
    "MODEL_PROVIDERS",
    "AgentRuntimeConfig",
    "BenchmarkSource",
    "BenchmarkSpec",
    "CheckpointConfig",
    "ConfidenceConfig",
    "EvalDatasetSpec",
    "EvaluateJobConfig",
    "EvaluateModelConfig",
    "EvaluateRunConfig",
    "LeaderboardInfo",
    "ReporterSpec",
    "RetryConfig",
    "SampleConfig",
]
