"""Configuration schema for the ``optimizer`` operation.

Lives next to :mod:`foresightrlcore.config.schema` and reuses the same
:class:`JobConfig`, :class:`ExecutionConfig`, :class:`OutputConfig`,
:class:`ContainerConfig`, :class:`SecretsConfig`, :class:`LLMClientsConfig`,
and :class:`VllmServingConfig` (from :mod:`perf_schema`) building blocks. Only
the optimizer-specific models are defined here.

Two infra profiles are resolved automatically from the candidate models:

* **API-only** (every candidate is ``genos``/``bedrock``): no GPU needed -
  defaults to a CPU instance and a lightweight CPU image.
* **Local-serving** (any candidate is ``provider: local``): a GPU instance is
  required to run ``vllm serve`` - defaults to a GPU instance + vLLM image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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

# Default instance types per infra profile (used when instance_type is unset).
DEFAULT_CPU_INSTANCE_TYPE = "ml.m5.2xlarge"
DEFAULT_GPU_INSTANCE_TYPE = "ml.g5.12xlarge"


class SampleConfig(BaseModel):
    """Deterministic sampling parameters."""

    model_config = ConfigDict(extra="forbid")

    size: int = Field(default=50, ge=1, description="Number of records to sample.")
    seed: int = Field(default=13, description="RNG seed for reproducible sampling.")
    stratify_by: str = Field(
        default="shard",
        description="Column/meta key to stratify the sample by (proportional "
        "allocation). Falls back to a single stratum if the column is absent.",
    )


class DatasetSpec(BaseModel):
    """One dataset entry. Provider-specific column-mapping keys are allowed
    via ``extra='allow'`` and forwarded verbatim to the dataset provider."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(description="Registered dataset provider name (registry key).")
    path: Optional[str] = Field(
        default=None,
        description="Dataset path. Local filesystem path or s3:// URI "
        "(downloaded before the run for the local backend / inside the job).",
    )
    filter: Dict[str, Any] = Field(
        default_factory=dict,
        description="Equality row filter, e.g. {data_complete: true}.",
    )

    def as_provider_spec(self) -> Dict[str, Any]:
        """Return the full mapping (known + extra keys) for the provider."""
        return self.model_dump()


class OptimizerModelConfig(BaseModel):
    """A candidate model + provider + token pricing (USD per 1K tokens)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Logical candidate name (used in the results table). For "
        "local serving this is the HF id / S3 prefix of the model to serve."
    )
    provider: str = Field(
        default="bedrock",
        description="Generation provider: 'bedrock' | 'local' | 'genos' (requires foresightrl-intuit).",
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Provider model id. Defaults to `name` when unset (genos "
        "model name, bedrock model id, or served-model alias for local).",
    )
    price_in: float = Field(default=0.0, ge=0.0, description="USD per 1K input tokens.")
    price_out: float = Field(
        default=0.0, ge=0.0, description="USD per 1K output tokens."
    )
    max_tokens: int = Field(
        default=16000, ge=1, description="Max output tokens per call."
    )
    truncate_prompt_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="If set, truncate each generation prompt to the last N "
        "tokens before generating. OFF by default. Honored only by "
        "provider=local (forwarded to vLLM as `truncate_prompt_tokens` in the "
        "request body); ignored by API providers (genos/bedrock), which have "
        "large context windows. Useful to keep small local models within their "
        "context window on validation/smoke runs.",
    )
    temperature: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for Bedrock or Genos candidates. "
        "Set to null to omit temperature for models that reject it.",
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

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        allowed = {"genos", "bedrock", "local"}
        if v not in allowed:
            raise ValueError(
                f"model.provider must be one of {sorted(allowed)}, got: {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _defaults(self) -> "OptimizerModelConfig":
        if self.model_id is None:
            self.model_id = self.name
        if self.provider == "local" and self.serving is None:
            self.serving = VllmServingConfig()
        return self

    @property
    def resolved_model_id(self) -> str:
        return self.model_id or self.name

    @property
    def serving_path(self) -> str:
        """Where a local server loads weights from.

        The optimizer does not stage weights to disk, so this is always the
        configured name; it exists so optimizer and evaluate models present the
        same surface to the shared generation and inference layers.
        """
        return self.name

    @property
    def lora_adapter(self) -> Optional[str]:
        """Always ``None``: the optimizer compares published models, not adapters.

        Present for the same reason as :attr:`serving_path` -- the shared
        inference layer reads one model surface, whichever operation built it.
        """
        return None


class JudgeSpec(BaseModel):
    """A judge selection with its aggregation weight and optional params.

    Judges with ``weight == 0`` act as deterministic gates: they do not move the
    weighted primary metric but their pass/fail is recorded (and gates must pass).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Registered judge name (registry key).")
    weight: float = Field(default=0.0, ge=0.0, description="Aggregation weight.")
    client: str = Field(
        default="default",
        description="Name of the `judge_clients` entry an LLM judge uses. "
        "Ignored by deterministic judges.",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Judge-specific params (e.g. thresholds, required sections).",
    )


class RetryConfig(BaseModel):
    """Resilient-call parameters for generation and LLM-judge calls."""

    model_config = ConfigDict(extra="forbid")

    max_retries: int = Field(default=10, ge=1)
    base_delay: float = Field(
        default=3.0, ge=0.0, description="Base seconds for exponential backoff."
    )


class CheckpointConfig(BaseModel):
    """Incremental checkpoint / resume settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True)
    every_n_rows: int = Field(
        default=20,
        ge=1,
        description="Flush the checkpoint JSONL (and sync to S3 if applicable) "
        "every N completed rows.",
    )


class OptimizerRunConfig(BaseModel):
    """A single optimizer run: the batch cross-product evaluation."""

    model_config = ConfigDict(extra="forbid")

    sample: SampleConfig = Field(default_factory=SampleConfig)
    datasets: List[DatasetSpec] = Field(description="One or more dataset entries.")
    prompt_set: str = Field(description="Registered prompt-set name.")
    prompts_path: Optional[str] = Field(
        default=None,
        description="Explicit prompts YAML path overriding the prompt_set's "
        "default file. May be a local path or an s3:// URI (downloaded at "
        "runtime, like the dataset). Required as an s3:// URI on "
        "aws_training_job unless the prompt_set's YAML ships inside an "
        "installed package: the container gets the packages, not the repo.",
    )
    models: List[OptimizerModelConfig] = Field(description="Candidate models.")
    modes: List[str] = Field(description="Registered generation-mode names.")
    judges: List[JudgeSpec] = Field(description="Judges to evaluate each output.")
    judge_clients: Optional[LLMClientsConfig] = Field(
        default=None,
        description="Provider-agnostic LLM client mapping used by LLM judges "
        "(required if any judge is an LLM judge).",
    )
    acceptance_threshold: float = Field(
        default=0.80,
        description="Primary-metric pass cutoff (strictly greater).",
    )
    concurrency: int = Field(
        default=4,
        ge=1,
        description="Thread-pool size for concurrent per-row evaluation.",
    )
    retry: RetryConfig = Field(default_factory=RetryConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    baseline_model: Optional[str] = Field(
        default=None,
        description="Metadata only - the reference output is read from the "
        "dataset's reference column, not regenerated.",
    )

    role: Optional[str] = Field(
        default=None,
        description="SageMaker execution role ARN. Auto-detected if not set.",
    )

    # SageMaker sizing (mirrors PerfTestRunConfig). instance_type has no hard
    # default: it is resolved from requires_gpu() when unset.
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
        "Required by Intuit ACP policy IS031.1 (AwsSagemakerTrainingjob must "
        "be encrypted in-transit). On by default; harmless for single-instance "
        "jobs and does not require a VPC.",
    )
    vpc: Optional[VpcConfig] = Field(
        default=None,
        description="VPC config (subnets + security groups). Required to reach "
        "the Intuit internal network from the job (e.g. to pip-install "
        "idps-sdk for the secrets: block, or call internal endpoints).",
    )

    @field_validator("datasets", "models", "modes", "judges")
    @classmethod
    def _non_empty(cls, v: List[Any]) -> List[Any]:
        if not v:
            raise ValueError("must contain at least one entry")
        return v

    def judge_names(self) -> List[str]:
        return [j.name for j in self.judges]


class OptimizerJobConfig(BaseModel):
    """Top-level configuration for the ``optimizer`` operation."""

    _source_path: str = ""

    version: int = Field(default=1)
    job: JobConfig
    optimizer: OptimizerRunConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    environment: Dict[str, str] = Field(default_factory=dict)
    container: ContainerConfig = Field(default_factory=ContainerConfig)
    secrets: Optional[SecretsConfig] = Field(default=None)

    @model_validator(mode="after")
    def _validate_root(self) -> "OptimizerJobConfig":
        valid_backends = {"aws_training_job", "local"}
        if self.execution.type not in valid_backends:
            raise ValueError(
                f"optimizer does not support execution.type "
                f"'{self.execution.type}'. Supported: {sorted(valid_backends)}"
            )
        return self

    @model_validator(mode="after")
    def _default_checkpoint_path(self) -> "OptimizerJobConfig":
        """Give checkpointing somewhere durable to live, so resume works.

        Same reasoning as the equivalent hook on ``EvaluateJobConfig``, and the
        same trap: ``checkpoint.enabled`` defaults to true and the runner
        honours it, but the rows only outlive the job if a checkpoint S3 prefix
        exists for the launcher to hang a SageMaker ``CheckpointConfig`` on.
        Without one, checkpoints are written to container disk and discarded,
        with no error and no warning to say so -- and an optimizer run is a
        long cross-product against paid endpoints, which is exactly the work
        worth not repeating.

        Reusing rows is safe here for the same reason: a checkpointed row is
        keyed on what produced it, so a changed prompt set or model re-runs.
        Set ``checkpoint.enabled: false`` to force a clean run.
        """
        if (
            self.optimizer.checkpoint.enabled
            and not self.output.checkpoint_s3_path
            and self.execution.type == "aws_training_job"
            and self.output.s3_path
            and self.output.s3_path.startswith("s3://")
        ):
            base = self.output.s3_path.rstrip("/")
            self.output.checkpoint_s3_path = f"{base}/checkpoints/"
        return self

    def requires_gpu(self) -> bool:
        """True iff any candidate model is served locally (needs a GPU)."""
        return any(m.provider == "local" for m in self.optimizer.models)

    def resolved_instance_type(self) -> str:
        """Resolve the SageMaker instance type (explicit value wins)."""
        if self.optimizer.instance_type:
            return self.optimizer.instance_type
        return (
            DEFAULT_GPU_INSTANCE_TYPE
            if self.requires_gpu()
            else DEFAULT_CPU_INSTANCE_TYPE
        )

    def uses_llm_judges(self) -> bool:
        """True iff any configured judge is an LLM judge (needs judge_clients)."""
        from foresightrlcore.optimizer.judges import is_llm_judge

        return any(is_llm_judge(j.name) for j in self.optimizer.judges)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OptimizerJobConfig":
        """Load and validate config from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        instance = cls.model_validate(raw)
        instance._source_path = str(path)
        return instance

    def to_yaml(self, path: str | Path) -> None:
        """Serialize config to a YAML file."""
        payload = self.model_dump(mode="json", exclude_none=True)
        # Null opts out of temperature; dropping it restores the 0.0 default
        # when the worker loads the serialized config.
        for model, serialized in zip(
            self.optimizer.models, payload["optimizer"]["models"]
        ):
            if model.temperature is None:
                serialized["temperature"] = None
        with open(path, "w") as f:
            yaml.dump(
                payload,
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    def merge_overrides(self, **kwargs: Any) -> "OptimizerJobConfig":
        """Return a new config with dotted-key CLI overrides merged in."""
        data = self.model_dump()

        shortcut_map = {
            "output_path": ("output", "s3_path"),
        }

        for key, value in kwargs.items():
            if value is None:
                continue
            if key in shortcut_map:
                section, field = shortcut_map[key]
                data[section][field] = value
            elif "." in key:
                parts = key.split(".")
                target = data
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = value

        return OptimizerJobConfig.model_validate(data)


__all__ = [
    "DEFAULT_CPU_INSTANCE_TYPE",
    "DEFAULT_GPU_INSTANCE_TYPE",
    "SampleConfig",
    "DatasetSpec",
    "OptimizerModelConfig",
    "JudgeSpec",
    "RetryConfig",
    "CheckpointConfig",
    "OptimizerRunConfig",
    "OptimizerJobConfig",
]
