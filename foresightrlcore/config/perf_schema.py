"""Configuration schema for the ``perf_test`` operation.

Lives next to :mod:`foresightrlcore.config.schema` and reuses the same
:class:`SpeculativeConfig`, :class:`JobConfig`, :class:`ModelConfig`,
:class:`OutputConfig`, :class:`ExecutionConfig`, :class:`ContainerConfig`,
and :class:`SecretsConfig` building blocks. Only the perf-test-specific
``serving`` / ``gatling`` models are defined here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from foresightrlcore.config.schema import (
    ContainerConfig,
    ExecutionConfig,
    JobConfig,
    ModelConfig,
    OutputConfig,
    SecretsConfig,
    SpeculativeConfig,
    VpcConfig,
)


class VllmServingConfig(BaseModel):
    """vLLM serve parameters for a perf-test run.

    Maps 1:1 onto ``vllm serve <model> --<flag> <value> ...`` invocations.
    Covers the production DJL serving properties used in
    ``foresight-rl-core`` deployments (max_num_seqs, served_model_name,
    trust_remote_code, vllm_env, speculative decoding, ...).
    """

    model_config = ConfigDict(extra="forbid")

    port: int = Field(
        default=8080,
        ge=1024,
        le=65535,
        description="vLLM server port. The vendored simulations hardcode "
        "http://localhost:8080 (org.foresightrl.perf.Constants), so leave this "
        "at 8080 unless the simulation class has been customized.",
    )
    tensor_parallel_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=0.9, ge=0.1, le=1.0)
    max_model_len: Optional[int] = Field(default=None, ge=256)
    max_num_seqs: int = Field(
        default=64,
        ge=1,
        description="Max concurrent sequences the engine will batch. Maps to "
        "the DJL `option.max_rolling_batch_size` and to "
        "`--max-num-seqs` on `vllm serve`.",
    )
    served_model_name: Optional[str] = Field(
        default=None,
        description="Alias the served model is exposed under in API responses "
        "(maps to `--served-model-name`). If unset, vLLM uses the model "
        "directory name.",
    )
    trust_remote_code: bool = Field(
        default=True,
        description="Pass `--trust-remote-code` to vllm serve. Required for "
        "custom model architectures (most Intuit checkpoints).",
    )
    enforce_eager: bool = Field(default=False)
    dtype: Optional[str] = Field(
        default=None,
        description="Override the model dtype (e.g. 'bfloat16', 'float16'). "
        "Leave unset to use the model's default.",
    )
    quantization: Optional[str] = Field(
        default=None,
        description="vLLM quantization method (e.g. 'awq', 'gptq'). Required "
        "when serving a quantized checkpoint.",
    )
    speculative: Optional[SpeculativeConfig] = Field(
        default=None,
        description="Optional speculative decoding config. Forwarded as a "
        "JSON object to `vllm serve --speculative-config`.",
    )
    enable_prefix_caching: bool = Field(
        default=False,
        description="Pass `--enable-prefix-caching` to vllm serve.",
    )
    max_lora_rank: Optional[int] = Field(
        default=None,
        ge=1,
        description="Largest LoRA rank the engine will accept "
        "(`--max-lora-rank`). Only emitted when an adapter is served. vLLM "
        "defaults this to 16 and refuses anything larger at load time, so an "
        "r=32 adapter needs it set explicitly.",
    )
    vllm_env: Dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables exported before launching vllm "
        "serve (e.g. `VLLM_USE_V1: '1'`, "
        "`VLLM_ATTENTION_BACKEND: 'FLASH_ATTN'`).",
    )
    extra_args: Dict[str, str] = Field(
        default_factory=dict,
        description="Escape hatch for arbitrary `--flag value` CLI args. Keys "
        "are flag names (without leading `--`); values are stringified. An "
        "empty value emits the flag on its own, which is how vllm's store_true "
        "options (`enable-lora`, `disable-log-requests`) are reachable.",
    )
    gpu_devices: Optional[str] = Field(
        default=None,
        description="Override CUDA_VISIBLE_DEVICES for the vllm serve "
        "subprocess (e.g. '0,1,2,3'). Defaults to all visible GPUs.",
    )
    startup_timeout: int = Field(
        default=1800,
        ge=60,
        le=7200,
        description="Seconds to wait for the vLLM server's /health endpoint to "
        "return 200 before failing the perf test.",
    )

    @field_validator("dtype")
    @classmethod
    def _validate_dtype(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {"auto", "half", "float16", "bfloat16", "float", "float32"}
        if v not in allowed:
            raise ValueError(f"Invalid dtype '{v}'. Must be one of: {sorted(allowed)}")
        return v


class GatlingConfig(BaseModel):
    """Gatling load-test parameters."""

    model_config = ConfigDict(extra="forbid")

    simulation_class: str = Field(
        default="org.foresightrl.perf.ThrottledTpsCsvSimulation",
        description="Fully-qualified Java class name of the Gatling Simulation "
        "to run. Resolved from the vendored project at "
        "`foresightrlcore/perf_test/gatling/src/gatling/java/...`, or from an "
        "extension package's `perf_test/gatling_java/` tree when that package "
        "is listed in `container.extra_packages` (the runner merges the two "
        "source trees before Gradle compiles). Pluggable: add a .java file and "
        "set this field, no Python changes needed.",
    )
    tps_levels: List[int] = Field(
        default_factory=lambda: [1, 2, 3, 4],
        description="List of target requests-per-second levels. The runner "
        "executes the simulation once per TPS value and aggregates results.",
    )
    duration_seconds: int = Field(
        default=600,
        ge=30,
        description="Steady-state duration of each Gatling run, in seconds.",
    )
    data_s3_path: str = Field(
        description="S3 prefix containing the user-supplied test data files "
        "(`prompts.csv`, `system-message.txt`, `chat-config.json`). Mounted "
        "as a SageMaker input channel and copied into the vendored Gatling "
        "resources directory at runtime. Required -- nothing is bundled by "
        "default to keep prompts and test data under user control.",
    )
    concurrent_users: Optional[int] = Field(
        default=None,
        ge=1,
        description="Explicit concurrent-users override for the closed "
        "injection model (`-DconcurrentUsers=`). When set, it is used verbatim "
        "for every TPS level. When unset, the runner derives a realistic pool "
        "per level via Little's Law: `tps * latency_budget_seconds` (see "
        "`latency_budget_seconds`).",
    )
    latency_budget_seconds: int = Field(
        default=30,
        ge=1,
        description="Per-request latency budget (SLO) in seconds, used to size "
        "the closed-model concurrent-user pool via Little's Law when "
        "`concurrent_users` is not set: `concurrentUsers = tps * "
        "latency_budget_seconds`. This makes the standing request pool match "
        "the concurrency the deployment is actually sized for, instead of the "
        "legacy `tps * 100` which over-provisioned the pool ~5x and produced "
        "client-timeout KOs even on a healthy server. Set this to the worst-"
        "case latency target (e.g. 30 for a 30s p95 SLO). Only used by the "
        "`closed` injection model (see `injection_model`).",
    )
    injection_model: str = Field(
        default="open",
        description="Gatling injection profile. 'open' (default, recommended): "
        "`constantUsersPerSec(tps)` -- new requests arrive at a fixed rate "
        "regardless of how fast the server responds. This is the textbook way "
        "to measure latency-at-a-target-arrival-rate: it never floods the "
        "server's admission queue beyond the steady-state arrival, so reported "
        "latency reflects real per-request cost rather than client-side queue "
        "wait. 'closed': `injectClosed` with a fixed standing pool of "
        "`concurrent_users` (or `tps * latency_budget_seconds`) virtual users "
        "throttled to `tps`. The closed model can over-inject relative to the "
        "server's `max_num_seqs` and report queue-wait as latency/timeouts, so "
        "prefer 'open' for SLO measurement. Passed verbatim to the simulation "
        "via `-DinjectionModel`.",
    )
    extra_jvm_args: Dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary additional `-D` JVM system properties. Keys "
        "without leading `-D` are stringified, e.g. "
        "`{'myFlag': 'true'}` becomes `-DmyFlag=true`.",
    )
    request_timeout_ms: int = Field(
        default=60_000,
        ge=1_000,
        description=(
            "Per-request HTTP timeout in milliseconds, applied via "
            "`-Dgatling.http.requestTimeout=...`. The default of 60_000 ms "
            "matches Gatling's stock default. Ramping TPS until requests "
            "start KO'ing is the measurement "
            "(saturation point under realistic SLOs). Raise this only when "
            "you specifically want to measure tail latencies past the stock "
            "SLO (e.g. for batch/offline workloads with no realtime budget)."
        ),
    )

    @field_validator("tps_levels")
    @classmethod
    def _validate_tps_levels(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("tps_levels must contain at least one value")
        for tps in v:
            if tps < 1:
                raise ValueError(f"All tps_levels must be >= 1, got: {tps}")
        return v

    @field_validator("injection_model")
    @classmethod
    def _validate_injection_model(cls, v: str) -> str:
        allowed = {"open", "closed"}
        if v not in allowed:
            raise ValueError(
                f"gatling.injection_model must be one of {sorted(allowed)}, "
                f"got: {v!r}"
            )
        return v

    @field_validator("data_s3_path")
    @classmethod
    def _validate_data_s3_path(cls, v: str) -> str:
        if not v.startswith("s3://"):
            raise ValueError(f"gatling.data_s3_path must be an s3:// URI, got: {v!r}")
        if not v.endswith("/"):
            v = v + "/"
        return v


class PerfTestRunConfig(BaseModel):
    """A single performance-test run -- one SageMaker training job."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Logical name for this run. Used as the SageMaker job "
        "name suffix and as the output subdirectory when multiple runs share "
        "the same `output.s3_path`."
    )
    role: Optional[str] = Field(
        default=None,
        description="SageMaker execution role ARN. Auto-detected if not set.",
    )
    instance_type: str = Field(
        default="ml.g5.xlarge",
        description="SageMaker instance type for the run.",
    )
    instance_count: int = Field(default=1, ge=1)
    volume_size_gb: int = Field(default=200, ge=30)
    training_plan_arn: Optional[str] = Field(
        default=None,
        description="SageMaker training-plan (capacity reservation) ARN. When "
        "set, injected as ResourceConfig.TrainingPlanArn on the CreateTrainingJob "
        "request so the perf-test job runs against reserved capacity (e.g. a p5en "
        "reservation) instead of on-demand.",
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
        description="VPC config (subnets + security groups). Without it the job "
        "egresses through SageMaker's service-managed NAT, an address pool "
        "shared across the region. Maven Central rate-limits that aggregate, so "
        "Gradle gets HTTP 429 resolving the Gatling plugin classpath and the "
        "load stage never runs. A VPC moves egress to the account's own NAT and "
        "makes an internal artifact mirror reachable. Mirrors the same field on "
        "the optimizer and training configs.",
    )
    serving: VllmServingConfig = Field(
        default_factory=lambda: VllmServingConfig(),
        description="vLLM serving parameters.",
    )
    gatling: GatlingConfig = Field(
        description="Gatling load-test parameters.",
    )
    run_gatling_test: bool = Field(
        default=True,
        description="If False, only start the vLLM server and exit "
        "successfully (smoke test mode). Useful for validating that the "
        "model loads before kicking off long Gatling runs.",
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not v:
            raise ValueError("run name must not be empty")
        if not all(c.isalnum() or c in "-_" for c in v):
            raise ValueError(
                f"Run name must contain only alphanumeric, '-', or '_' "
                f"characters, got: {v!r}"
            )
        return v


class PerfTestJobConfig(BaseModel):
    """Top-level configuration for the ``perf_test`` operation.

    Mirrors :class:`TrainingJobConfig` in spirit but tailored for
    performance testing: a single model, a single run, and a DJL-inference
    container instead of the pytorch-training DLC.
    """

    _source_path: str = ""

    version: int = Field(default=1)
    job: JobConfig
    model: ModelConfig
    perf_test: PerfTestRunConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    environment: Dict[str, str] = Field(default_factory=dict)
    container: ContainerConfig = Field(default_factory=ContainerConfig)
    secrets: Optional[SecretsConfig] = Field(default=None)

    @model_validator(mode="after")
    def _validate_root(self) -> "PerfTestJobConfig":
        valid_backends = {"aws_training_job", "local"}
        if self.execution.type not in valid_backends:
            raise ValueError(
                f"perf_test does not support execution.type "
                f"'{self.execution.type}'. Supported: {sorted(valid_backends)}"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PerfTestJobConfig":
        """Load and validate config from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        instance = cls.model_validate(raw)
        instance._source_path = str(path)
        return instance

    def to_yaml(self, path: str | Path) -> None:
        """Serialize config to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(
                self.model_dump(mode="json", exclude_none=True),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    def merge_overrides(self, **kwargs: Any) -> "PerfTestJobConfig":
        """Return a new config with CLI overrides merged in.

        Supports dotted paths (e.g. ``model.name=foo``) and a small set of
        shortcuts (``model_name``, ``output_path``).
        """
        data = self.model_dump()

        shortcut_map = {
            "model_name": ("model", "name"),
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

        return PerfTestJobConfig.model_validate(data)


__all__ = [
    "VllmServingConfig",
    "GatlingConfig",
    "PerfTestRunConfig",
    "PerfTestJobConfig",
]
