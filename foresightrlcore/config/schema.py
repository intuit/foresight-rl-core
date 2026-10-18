"""Training job configuration schema.

Canonical Pydantic models for all training job parameters. Supports
multiple methods (GRPO, SFT, DPO), execution backends (AWS, local),
and infrastructure configurations via a single YAML config file.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Topology(str, Enum):
    SINGLE_NODE = "single_node"
    MULTI_NODE = "multi_node"
    MULTI_NODE_SERVER = "multi_node_server"
    DEDICATED_VLLM = "dedicated_vllm"


class VllmMode(str, Enum):
    SERVER = "server"
    COLOCATE = "colocate"
    NONE = "none"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class JobConfig(BaseModel):
    name: str = Field(description="Base name for the training job")
    max_runtime_hours: int = Field(default=24, ge=1, le=432)
    max_wait_hours: Optional[int] = Field(
        default=None,
        ge=1,
        le=432,
        description="Max wall-clock time (hours) a managed spot training job has "
        "to complete, including time queued for spot capacity plus run time "
        "across interruptions. Maps to SageMaker MaxWaitTimeInSeconds. Only used "
        "when infra.use_spot_instances is true; must be >= max_runtime_hours. "
        "If unset while spot is enabled, defaults to max_runtime_hours.",
    )


class SpeculativeConfig(BaseModel):
    """vLLM speculative decoding config.

    Forwarded verbatim as a JSON object to ``vllm serve --speculative-config``.
    See vLLM docs for valid ``method`` values (e.g. ``eagle3``, ``medusa``,
    ``qwen3_next_mtp``). ``extra="allow"`` lets users pass new vLLM keys
    without a schema bump.
    """

    model_config = ConfigDict(extra="allow")

    method: str = Field(
        description="Speculative method, e.g. 'eagle3', 'qwen3_next_mtp'."
    )
    num_speculative_tokens: int = Field(ge=1, description="Tokens drafted per step.")
    model: Optional[str] = Field(
        default=None,
        description="Draft model path or HF ID. Omit for built-in MTP heads "
        "(e.g. qwen3_next_mtp doesn't need this).",
    )
    draft_tensor_parallel_size: Optional[int] = Field(default=None, ge=1)


class VllmConfig(BaseModel):
    host: str = Field(
        default="0.0.0.0",
        description="vLLM server host. For local/single-node this is 0.0.0.0. "
        "For multi-node topologies the entrypoint sets this to the vLLM node IP.",
    )
    port: int = Field(default=8002, ge=1024, le=65535)
    tensor_parallel_size: int = Field(default=2, ge=1)
    # For small models that fit on a single GPU (e.g. 8B in bf16 ≤ 16 GB),
    # data parallelism gives much higher throughput than tensor parallelism
    # because each DP replica handles its own requests with zero inter-GPU
    # comm during a forward pass. TRL 0.28+ vllm-serve supports DP via
    # VLLM_DP_RANK env vars and treats world_size = TP * DP + 1 for the
    # weight-sync NCCL group.
    data_parallel_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=0.9, ge=0.1, le=1.0)
    max_model_len: Optional[int] = Field(default=None, ge=256)
    enforce_eager: bool = Field(default=False)
    enable_prefix_caching: bool = Field(
        default=False,
        description="Pass --enable-prefix-caching to vllm serve. Helpful for "
        "GRPO where all rollouts of one prompt share the prompt prefix.",
    )
    disable_custom_all_reduce: bool = Field(
        default=False,
        description="When True, inject ``disable_custom_all_reduce=True`` into "
        "TRL's ``LLM(...)`` call via ``patch_trl_vllm.py``. Workaround for "
        "vLLM 0.20.x's broken custom all-reduce kernel with TP>1 on H100/H200. "
        "Server mode only — ignored (with a warning) when vllm_mode is "
        "'colocate' or 'none' (TRL's GRPOConfig has no equivalent kwarg).",
    )
    speculative: Optional[SpeculativeConfig] = Field(
        default=None,
        description="Optional speculative decoding config. Server mode only — "
        "ignored (with a warning) when vllm_mode is 'colocate' or 'none'.",
    )
    reasoning_parser: Optional[str] = Field(
        default=None,
        description="Reasoning parser identifier for thinking models "
        "(e.g. 'qwen3'). Required for ThinkingTokenBudgetLogitsProcessor to "
        "activate in vLLM >= 0.19. Server mode only — ignored (with a warning) "
        "when vllm_mode is 'colocate' or 'none'.",
    )
    reasoning_config: Optional[Dict[str, str]] = Field(
        default=None,
        description="Reasoning config dict with start/end delimiters, e.g. "
        '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}. '
        "Required alongside reasoning_parser for thinking models. Server mode "
        "only.",
    )
    compilation_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="vLLM CompilationConfig kwargs for CUDA graph control. "
        "Passed as-is to CompilationConfig(). Key fields: "
        "cudagraph_capture_sizes (list of batch sizes to capture), "
        "max_cudagraph_capture_size (cap on capture ladder), "
        "cudagraph_num_of_warmups (warmup runs before capture). Use to limit "
        "CUDA graph capture and avoid startup hangs on H100. Server mode only.",
    )
    startup_timeout: int = Field(
        default=1800,
        ge=60,
        le=7200,
        description="Seconds to wait for the vLLM server to become healthy. "
        "Large models (>7B) need longer for download + loading. "
        "Training nodes add a 300s buffer on top of this value.",
    )


class VpcConfig(BaseModel):
    subnets: List[str] = Field(default_factory=list)
    security_groups: List[str] = Field(default_factory=list)


class MultiNodeConfig(BaseModel):
    vllm_instance_type: str = Field(default="ml.g5.12xlarge")
    vllm_instance_count: int = Field(default=1, ge=1)
    training_instance_type: str = Field(default="ml.p4d.24xlarge")
    training_instance_count: int = Field(default=1, ge=1)


class InfraConfig(BaseModel):
    topology: Topology = Field(default=Topology.SINGLE_NODE)
    vllm_mode: VllmMode = Field(
        default=VllmMode.SERVER,
        description="'server' | 'colocate' | 'none' (torchrun, no vLLM).",
    )
    infra_profile: Optional[str] = Field(
        default=None,
        description="Named preset that expands to topology + vllm_mode defaults.",
    )
    role: Optional[str] = Field(
        default=None,
        description="SageMaker execution role ARN. Auto-detected if not set.",
    )
    instance_type: str = Field(default="ml.p4d.24xlarge")
    instance_count: int = Field(default=1, ge=1)
    vllm_instance_type: Optional[str] = Field(
        default=None,
        description="Opt-in SageMaker instance type for the dedicated_vllm "
        "vLLM group. When set, the job is submitted as a heterogeneous "
        "cluster (InstanceGroups): this type × vllm_instance_count for "
        "vLLM, and instance_type × (instance_count - vllm_instance_count) "
        "for training. When unset, all nodes use instance_type "
        "(homogeneous, the current default). dedicated_vllm + "
        "aws_training_job only.",
    )
    vllm_instance_count: int = Field(
        default=1,
        ge=1,
        description="Number of vLLM instances when vllm_instance_type is set. "
        "Must be < instance_count so at least one training node remains. "
        "Ignored when vllm_instance_type is unset.",
    )
    allow_mixed_gpu_families: bool = Field(
        default=False,
        description="Permit mixing GPU families or EFA capability between the "
        "vLLM and training instance types (e.g. g5 + p4d). TRL weight-sync "
        "uses NCCL across those nodes, so cross-family mixes are rejected "
        "by default. Set true only for explicit experiments.",
    )
    vllm_gpu_count: int = Field(default=2, ge=1)
    training_gpu_count: int = Field(default=6, ge=1)
    gpus_per_node: Optional[int] = Field(
        default=None,
        description="Override auto-detected GPU count per node.",
    )
    keep_alive_seconds: int = Field(default=0, ge=0)
    use_spot_instances: bool = Field(
        default=False,
        description="Enable SageMaker Managed Spot Training (EC2 Spot instances) "
        "for the training job, saving up to ~70-90% vs on-demand. Requires "
        "output.checkpoint_s3_path so runs can auto-resume across spot "
        "interruptions. Homogeneous topologies only (single_node, "
        "multi_node_server, homogeneous dedicated_vllm). Not supported "
        "with mixed-type dedicated_vllm (vllm_instance_type set) or "
        "the heterogeneous multi_node topology.",
    )
    volume_size_gb: int = Field(
        default=100, ge=1, description="EBS volume size in GB per instance"
    )
    training_plan_arn: Optional[str] = Field(default=None)
    nccl_transport: str = Field(default="efa")
    sagemaker_default_bucket: Optional[str] = Field(
        default=None,
        description="Override the SageMaker session default bucket "
        "(used for source code/config/output uploads). "
        "If unset, SageMaker uses 'sagemaker-<region>-<account-id>'.",
    )
    sagemaker_default_bucket_prefix: Optional[str] = Field(
        default=None,
        description="Optional key prefix under sagemaker_default_bucket.",
    )
    kms_key_id: Optional[str] = Field(
        default=None,
        description="KMS key ID or alias for SageMaker OutputDataConfig / "
        "checkpoint artifact encryption (e.g. 'alias/my-kms-key' or a full "
        "key ARN). When unset, S3 falls back to the destination bucket's "
        "own default encryption — which may be a key your training role "
        "isn't permitted to use (an explicit deny in an account's IAM "
        "permission boundary silently overrides any allow elsewhere).  "
        "Setting this makes the job request a specific key regardless of "
        "the bucket default. Wired into all four aws_training_job launch "
        "paths in aws_sagemaker_launcher.py — SDK v3 ModelTrainer's "
        "OutputDataConfig (the normal path), the boto3-direct "
        "CreateTrainingJob OutputDataConfig (reachable even for a plain "
        "sft_verl config whenever training_plan_arn is set, independent "
        "of vllm_mode), the SDK v2 Estimator's output_kms_key "
        "(reachable whenever the installed sagemaker package predates "
        "ModelTrainer), and the heterogeneous-InstanceGroups SDK v2 "
        "Estimator's output_kms_key (topology: multi_node, e.g. GRPO with "
        "a dedicated vLLM instance group). SageMaker's API has no "
        "separate CheckpointConfig KMS field; checkpoints inherit this "
        "same key on every path.",
    )
    encrypt_inter_container_traffic: bool = Field(
        default=True,
        description="Encrypt all traffic between ML compute instances in "
        "distributed training (TLS). Maps to SageMaker "
        "EnableInterContainerTrafficEncryption. Required by Intuit ACP policy "
        "IS031.1 (AwsSagemakerTrainingjob must be encrypted in-transit). "
        "Defaults to True; it only affects multi-instance jobs but is harmless "
        "— and keeps single-instance jobs compliant. Does not require a VPC.",
    )
    vllm: VllmConfig = Field(default_factory=VllmConfig)
    vpc: Optional[VpcConfig] = Field(default=None)
    multi_node: Optional[MultiNodeConfig] = Field(default=None)

    @model_validator(mode="after")
    def _apply_profile_and_validate(self) -> "InfraConfig":
        from foresightrlcore.config.profiles import resolve_infra_profile

        resolve_infra_profile(self)
        if self.topology == Topology.MULTI_NODE and self.multi_node is None:
            self.multi_node = MultiNodeConfig()
        return self

    def uses_mixed_instance_types(self) -> bool:
        """True when dedicated_vllm opted into SageMaker InstanceGroups."""
        return (
            self.topology == Topology.DEDICATED_VLLM
            and self.vllm_instance_type is not None
        )

    def mixed_training_instance_count(self) -> int:
        """Training-group size when ``vllm_instance_type`` is set."""
        return self.instance_count - self.vllm_instance_count


class LoraConfig_(BaseModel):
    """LoRA adapter configuration."""

    r: int = Field(default=32, ge=1)
    alpha: int = Field(default=64, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    exclude_modules: Union[List[str], str] = Field(
        default_factory=list,
        description="Module name patterns to exclude from LoRA adaptation "
        "(e.g. a router/gate path on MoE models). Empty list means no "
        "exclusions. A list entry matches only an exact trailing path "
        "segment (PEFT: key.endswith(f'.{entry}')) — to exclude an entire "
        "subtree (e.g. a whole vision tower nested under the text model), "
        "pass a single regex STRING instead (PEFT: re.fullmatch), e.g. "
        "'.*(vision_tower|vision_adapter).*'. Only forwarded by handlers "
        "that support it (currently: sft_verl, and the TRL/GRPO path in "
        "methods/base.py).",
    )
    target_parameters: List[str] = Field(
        default_factory=list,
        description="Dotted parameter-name patterns for LoRA-on-nn.Parameter "
        "targets — needed for MoE models whose expert FFN weights are fused "
        "3D nn.Parameter tensors rather than nn.Linear modules (e.g. Gemma 4, "
        "Mixtral, Qwen3-MoE, GPT-OSS, Llama 4), which target_modules matching "
        "cannot reach. Empty means none. Only forwarded by handlers that "
        "support it (currently: sft_verl, via verl's model.target_parameters "
        "-> peft.LoraConfig(target_parameters=...)). Confirm the exact "
        "parameter paths against the real checkpoint's named_parameters() "
        "before relying on this.",
    )


class ModelConfig(BaseModel):
    name: str = Field(
        description=(
            "HuggingFace model ID (e.g. 'Qwen/Qwen3.5-27B') or an S3 prefix "
            "URI (e.g. 's3://bucket/models/Qwen3.6-27B/'). When an S3 URI is "
            "given the launcher mounts it as a SageMaker 'model' input "
            "channel and the container loads from /opt/ml/input/data/model "
            "instead of HF Hub — eliminates rank-races on Hub downloads."
        )
    )
    profile: Optional[str] = Field(
        default=None,
        description="Named model profile (see "
        "foresightrlcore.config.model_profiles.MODEL_PROFILES) resolving "
        "LoRA targeting (target_modules/exclude_modules/target_parameters) "
        "and chat-template quirks (thinking_kwarg, trust_remote_code) for "
        "a specific model family/architecture. Only fills fields not "
        "explicitly set elsewhere in this config — set model.lora.* or "
        "model.thinking_kwarg/trust_remote_code directly to override.",
    )
    lora: LoraConfig_ = Field(default_factory=LoraConfig_)
    enable_thinking: bool = Field(
        default=True,
        description="Enable thinking mode in chat template (Qwen3 models)",
    )
    thinking_budget: Optional[int] = Field(
        default=None,
        ge=1,
        description="Max tokens for thinking-mode chat template "
        "(forwarded to TRL via chat_template_kwargs). Qwen3 models only.",
    )
    thinking_kwarg: Optional[str] = Field(
        default=None,
        description="apply_chat_template() kwarg name used to toggle "
        "thinking mode (e.g. 'enable_thinking'). Resolved from model.profile "
        "when unset; handlers fall back to 'enable_thinking' if still None "
        "after resolution.",
    )
    trust_remote_code: Optional[bool] = Field(
        default=None,
        description="Whether loading this model's tokenizer/config needs "
        "trust_remote_code=True. Resolved from model.profile when unset; "
        "handlers fall back to True if still None after resolution.",
    )
    attn_implementation: Optional[str] = Field(
        default=None,
        description="'flash_attention_2' | 'sdpa' | 'eager'. Resolved from "
        "model.profile when unset; verl's own default ('flash_attention_2') "
        "applies when still None after resolution. Only needed when a "
        "model's attention heads exceed flash-attn's hard 256 head_dim "
        "limit (see the gemma4_moe profile) — most models should leave "
        "this unset.",
    )

    @model_validator(mode="after")
    def _apply_model_profile(self) -> "ModelConfig":
        from foresightrlcore.config.model_profiles import resolve_model_profile

        resolve_model_profile(self)
        return self

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if v.startswith("s3://"):
            if not v.endswith("/"):
                raise ValueError(
                    f"S3 model.name must be a prefix ending with '/' "
                    f"(SageMaker requires a directory-style channel source); "
                    f"got: {v!r}"
                )
        return v

    @property
    def is_s3_uri(self) -> bool:
        """True when ``name`` is an s3:// prefix that should be mounted."""
        return self.name.startswith("s3://")


class GrpoConfig(BaseModel):
    num_generations: int = Field(default=8, ge=1)
    max_completion_length: int = Field(default=8192, ge=64)
    beta: float = Field(default=0.001, ge=0.0)
    epsilon: float = Field(default=0.2, ge=0.0)
    epsilon_high: float = Field(default=0.28, ge=0.0)
    loss_type: str = Field(default="dr_grpo")
    temperature: float = Field(default=0.6, ge=0.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=20, ge=0)
    scale_rewards: Union[str, bool] = Field(default=False)
    mask_truncated_completions: bool = Field(default=True)
    vllm_importance_sampling_correction: bool = Field(default=False)
    num_iterations: int = Field(default=1, ge=1)
    importance_sampling_level: Optional[str] = Field(default=None)
    reward_weights: Optional[List[float]] = Field(default=None)

    # Generation control
    min_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, ge=0.0)
    generation_kwargs: Optional[Dict[str, Any]] = Field(default=None)
    generation_batch_size: Optional[int] = Field(default=None, ge=1)
    num_generations_eval: Optional[int] = Field(default=None, ge=1)

    # Advanced loss and training
    delta: Optional[float] = Field(default=None)
    sync_ref_model: bool = Field(default=False)
    ref_model_mixup_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    ref_model_sync_steps: int = Field(default=512, ge=1)
    multi_objective_aggregation: str = Field(default="sum_then_normalize")
    top_entropy_quantile: float = Field(default=1.0, ge=0.0, le=1.0)

    # Importance sampling (extended)
    vllm_importance_sampling_mode: Optional[str] = Field(default=None)
    vllm_importance_sampling_cap: float = Field(default=3.0, ge=0.0)
    off_policy_mask_threshold: Optional[float] = Field(default=None, ge=0.0)
    use_bias_correction_kl: bool = Field(default=False)

    # Model control
    disable_dropout: bool = Field(default=False)
    cast_lm_head_to_fp32: bool = Field(default=False)

    # Data / distributed
    ds3_gather_for_generation: bool = Field(default=True)
    shuffle_dataset: bool = Field(default=True)

    # Agentic training
    max_tool_calling_iterations: Optional[int] = Field(default=None, ge=1)


class VerlSftConfig(BaseModel):
    """Hyperparameters for the ``sft_verl`` method (verl's ``fsdp``/``fsdp2``
    FSDP SFT trainer, launched via plain torchrun — no Ray).

    Lives under ``training.sft_verl`` in YAML. Field names mirror verl's own
    ``sft_trainer_engine.yaml`` data/engine/trainer sections so translation in
    :mod:`foresightrlcore.methods.sft_verl` is close to 1:1.
    """

    data_format: str = Field(
        default="flat",
        description="'flat': rows have separate prompt/response (+ optional "
        "system_prompt) string fields — synthesized into a messages list by "
        "foresightrlcore.methods.verl_sft_dataset.FlatPromptResponseSFTDataset. "
        "'messages': rows already have a full messages list (e.g. data "
        "produced for a different framework/model) — read as-is by verl's "
        "own native MultiTurnSFTDataset, no synthesis. 'packed': "
        "data.train_path/eval_path point at pre-packed, pre-tokenized "
        "Parquet produced offline by `rl-train tokenize-pack` (input_ids/"
        "loss_mask/seq_start_id columns) — no tokenization happens at "
        "train time at all; read by "
        "foresightrlcore.methods.verl_sft_dataset.PackedSFTDataset, which "
        "resets position_ids at each seq_start_id boundary so HF's own "
        "packed-sequence mask detection (position_ids-based, model-"
        "agnostic) keeps attention from crossing packed examples — this is "
        "how sequence packing is recovered for models like Gemma 4 that "
        "can't use verl's built-in flash-attn-only packing path (see "
        "model_profiles.py's gemma4_moe profile). prompt_key/response_key/"
        "system_prompt below are only used when data_format='flat'.",
    )
    prompt_key: str = Field(
        default="prompt",
        description="JSONL field holding the prompt/user turn text. "
        "Only used when data_format='flat'.",
    )
    response_key: str = Field(
        default="response",
        description="JSONL field holding the assistant response (thinking + "
        "final answer). Loss is computed over this entire field. "
        "Only used when data_format='flat'.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system-turn text prepended to every example. "
        "Only used when data_format='flat' — 'messages' data carries its "
        "own system turn (if any) inside each row's messages list.",
    )
    max_length: int = Field(
        default=18432,
        ge=1,
        description="Max tokenized sequence length (prompt + response). "
        "Rows exceeding this are handled per `truncation`.",
    )
    truncation: str = Field(
        default="error",
        description="'error' | 'left' | 'right'. 'error' fails loudly on "
        "over-length rows instead of silently truncating the response.",
    )
    use_dynamic_bsz: bool = Field(
        default=True,
        description="Pack variable-length sequences by token budget "
        "(max_token_len_per_gpu) instead of a fixed micro_batch_size_per_gpu.",
    )
    max_token_len_per_gpu: int = Field(default=18432, ge=1)
    micro_batch_size_per_gpu: int = Field(
        default=1,
        ge=1,
        description="Only used when use_dynamic_bsz is False.",
    )
    ulysses_sequence_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Ulysses sequence-parallel degree. Splits attention/"
        "activation memory for long sequences across GPUs. Requires "
        "use_remove_padding=True when > 1.",
    )
    use_remove_padding: bool = Field(
        default=True,
        description="Required (must be True) whenever "
        "ulysses_sequence_parallel_size > 1.",
    )
    train_batch_size: int = Field(default=256, ge=1, description="Global batch size.")
    dataloader_num_workers: Optional[int] = Field(
        default=None,
        ge=0,
        description="Forwarded to verl's data.num_workers — the SAME field "
        "backs both the train AND val StatefulDataLoader (verl/trainer/"
        "sft_trainer.py's _build_dataloader passes "
        "num_workers=self.config.data.num_workers to both; there is no "
        "separate val-specific knob). None leaves verl's own hydra default "
        "(8) in place. Every GPU-local-rank process forks this many worker "
        "subprocesses independently — on an 8-GPU-per-node instance that's "
        "8x8=64 new fork() calls per node, each against an already-large "
        "parent VSZ once a multi-billion-parameter model + FSDP + optimizer "
        "state has been constructed. Reproduced directly on a real "
        "SageMaker job (sft-verl-gemma4-26ba4b-sft-verl-smoke-packed-"
        "20260929183107, Gemma 4 26B-A4B): training itself completed and "
        "checkpointed fine (train_dataloader's workers fork successfully "
        "early, before the model/optimizer are built), but the FIRST "
        "construction of val_dataloader — which only happens once data."
        "eval_path is set and only at the last training step or a test_freq "
        "boundary — crashed on every rank with `OSError: [Errno 12] Cannot "
        "allocate memory` at os.fork(). Only reachable when data.eval_path "
        "is set; a config with no eval split never builds a val_dataloader "
        "at all and is unaffected regardless of this setting. Set to a low "
        "value (e.g. 2) for any sizable model that also configures "
        "data.eval_path.",
    )
    save_freq: int = Field(
        default=-1,
        description="Save a checkpoint every N steps. -1 disables periodic "
        "saves (final save still happens).",
    )
    resume_mode: str = Field(
        default="auto",
        description="'auto' | 'disable' | 'resume_path'. verl's own "
        "CheckpointHandler resumes from trainer.default_local_dir when "
        "'auto' and a checkpoint is present — used for SageMaker Managed "
        "Spot interruption recovery.",
    )
    save_lora_only: bool = Field(
        default=True,
        description="verl's checkpoint.save_lora_only: when the model has "
        "LoRA adapters (always true for this recipe — LoRA-only, no full "
        "fine-tune), save just the adapter weights per FSDP shard instead "
        "of the frozen 25B base + adapters. Confirmed against verl's real "
        "CheckpointConfig docstring: '~150 MiB vs ~54 GiB for a 27B model'. "
        "Auto-detected and merged back in on resume — no other config "
        "needed. Defaults True here (unlike verl's own False) since every "
        "current use of this method is LoRA, so there's no downside.",
    )
    merge_lora_only: bool = Field(
        default=False,
        description="Skip training entirely: locate the SFT checkpoint at "
        "output.checkpoint_s3_path, reconstruct its (adapter-only, under "
        "the default save_lora_only=True) state dict from its FSDP shards, "
        "merge the LoRA adapter into model.name's REAL pretrained base "
        "weights (not the checkpoint's — save_lora_only never saves the "
        "frozen base weights at all) via peft's own merge_and_unload(), "
        "and save a plain (non-PEFT) HF model to output.s3_path. Exists "
        "because verl "
        "0.9.1's own adapter-export tool (verl.model_merger) cannot "
        "correctly reconstruct target_parameters-based LoRA (confirmed "
        "directly: it infers adapter_config.json's target_modules purely "
        "from state-dict key names, and for our MoE experts that "
        "misattributes 'experts' as a target_modules entry while never "
        "recording target_parameters at all — reloading that config would "
        "try to wrap the fused-parameter experts module as a plain "
        "nn.Linear target). A full merge sidesteps the bug entirely: a "
        "merged model has no adapter/config structure to reconstruct "
        "incorrectly. Runs single-node, single-process (no torchrun "
        "distribution needed — verl's own model_merger CLI is a plain "
        "script for the same reason) despite reusing the torchrun-launched "
        "entrypoint; only rank 0 does real work.",
    )
    merge_checkpoint_step: Optional[int] = Field(
        default=None,
        description="Which global_step_N under output.checkpoint_s3_path to "
        "merge. None auto-detects the highest step present. Only used when "
        "merge_lora_only is true.",
    )
    use_liger: bool = Field(
        default=True,
        description="Apply liger-kernel's fused RMSNorm/GeGLU-or-SwiGLU-MLP "
        "kernels to the model's decoder layers via verl's own model.use_liger "
        "-> liger_kernel.transformers._apply_liger_kernel_to_instance "
        "(verl/workers/engine/fsdp/transformer_impl.py). Verified against "
        "liger-kernel 0.8.3's real MODEL_TYPE_TO_APPLY_LIGER_FN registry: it "
        "already has day-one 'gemma4'/'gemma4_text' entries (RMSNorm + GeGLU "
        "MLP fusion; MoE expert kernels are explicitly out of scope in "
        "liger's own gemma4 docstring, so this cannot interact with the "
        "target_parameters LoRA path on the experts). For any model_type "
        "liger doesn't recognize, _apply_liger_kernel_to_instance logs and "
        "no-ops rather than raising — safe to leave on as a default for "
        "future model profiles too. Note: verl hardcodes "
        "fused_linear_cross_entropy=False for every model (documented as "
        "conflicting with verl's own forward patching), so this does NOT "
        "get liger's fused-CE memory win for Gemma 4's 262K vocab logits — "
        "only the RMSNorm/MLP kernel fusion.",
    )
    fsdp_strategy: str = Field(
        default="fsdp2",
        description="'fsdp' | 'fsdp2'. verl's engine.strategy. Under 'fsdp' "
        "(FSDP1), verl wraps every LoRA parameter as its own FSDP unit "
        "(verl/utils/fsdp_utils.py's is_lora lambda policy) to keep it out "
        "of the frozen base layer's FlatParameter. FSDP1 units unshard by "
        "reusing one persistent buffer via in-place copy_ on every forward "
        "call — including gradient checkpointing's recompute forward during "
        "backward. PEFT's target_parameters MoE-LoRA path "
        "(peft.tuners.lora.layer.ParamWrapper.get_delta_factors) holds a "
        "reshape() view of the LoRA weight across that boundary, and the "
        "recompute's in-place overwrite invalidates it: `RuntimeError: "
        "Output 0 of ViewBackward0 is a view and its base ... has been "
        "modified inplace` on every rank. A known FSDP1 failure class (see "
        "https://github.com/pytorch/pytorch/issues/124017), not specific to "
        "this repo's code. 'fsdp2' (fully_shard/DTensor, also fully wired in "
        "verl 0.9.1) does not separately wrap LoRA leaves and allocates "
        "fresh unshard storage per call instead of reusing one buffer "
        "in-place, removing the mechanism that triggers this. Defaulted to "
        "'fsdp2' rather than kept at verl's own 'fsdp' default because the "
        "underlying lambda-wrap-every-LoRA-leaf behavior applies to any "
        "LoRA + gradient-checkpointing combination under FSDP1, not just "
        "target_parameters/MoE.",
    )


class HooksConfig(BaseModel):
    """Training hooks for customizing TRL behavior without modifying framework code.

    Supports three levels of customization:
      - callbacks: Extra TrainerCallback classes for runtime adjustments
      - pre_train / post_train: Callables invoked before/after trainer.train()
      - trainer_class: Override the TRL trainer class entirely
    """

    trainer_class: Optional[str] = Field(
        default=None,
        description="Dotted path to a custom Trainer subclass "
        "(e.g. 'my_pkg.trainers.IterativeGRPOTrainer')",
    )
    pre_train: Optional[str] = Field(
        default=None,
        description="Dotted path to a callable invoked before trainer.train(). "
        "Signature: fn(trainer, config) -> None",
    )
    post_train: Optional[str] = Field(
        default=None,
        description="Dotted path to a callable invoked after trainer.train(). "
        "Signature: fn(trainer, config) -> None",
    )
    callbacks: Optional[List[str]] = Field(
        default=None,
        description="List of dotted paths to TrainerCallback classes to instantiate",
    )


class TrainingConfig(BaseModel):
    epochs: int = Field(default=1, ge=1)
    per_device_batch_size: int = Field(default=4, ge=1)
    per_device_eval_batch_size: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Per-device batch size during evaluation. If unset, falls back to "
            "HF Trainer's default of 8 — which can OOM during eval on large "
            "models, and is usually NOT what you want when "
            "per_device_batch_size is smaller (e.g. 2). Set explicitly to "
            "keep training and eval batching consistent."
        ),
    )
    gradient_accumulation_steps: int = Field(default=2, ge=1)
    learning_rate: float = Field(default=5e-5, gt=0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    lr_scheduler: str = Field(default="cosine")
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=1.0)
    bf16: bool = Field(default=True)
    fp16: bool = Field(default=False)
    gradient_checkpointing: bool = Field(default=True)
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = Field(default=None)
    deepspeed: str = Field(default="ds_zero2_config.json")
    grpo: GrpoConfig = Field(default_factory=GrpoConfig)
    sft_verl: Optional[VerlSftConfig] = Field(
        default=None,
        description="Hyperparameters for the sft_verl method (verl FSDP SFT).",
    )
    logging_steps: int = Field(default=5, ge=1)
    save_steps: int = Field(default=50, ge=1)
    eval_steps: int = Field(default=100, ge=1)
    max_steps: Optional[int] = Field(
        default=None,
        description="Maximum number of optimizer steps. Mutually exclusive with "
        "data_points — validation fails if both are set.",
    )
    data_points: Optional[int] = Field(
        default=None,
        ge=1,
        description="Target number of training examples to process. "
        "Auto-derives max_steps from the infra GPU layout at validation time. "
        "Mutually exclusive with max_steps — validation fails if both are set.",
    )
    max_grad_norm: Optional[float] = Field(default=None, ge=0.0)
    log_completions: bool = Field(default=True)
    use_liger_kernel: bool = Field(
        default=False,
        description="Use Liger Kernel for fused operations (reduces memory, "
        "improves throughput). Requires the liger-kernel pip package to be "
        "installed in the container (already part of the qwen3_5 recipe).",
    )
    report_to: str = Field(default="tensorboard")
    resume_from_checkpoint: Optional[Union[bool, str]] = Field(
        default=None,
        description="Resume training from a previous checkpoint. Set True to "
        "auto-pick the latest `checkpoint-*` dir in output_dir (SageMaker "
        "syncs checkpoint_s3_path → /opt/ml/checkpoints at job start). "
        "Or pass an explicit path to a `checkpoint-N` dir.",
    )
    hooks: Optional[HooksConfig] = Field(
        default=None,
        description="Optional training hooks: custom trainer class, "
        "pre/post hooks, extra callbacks",
    )
    serverless: Optional[ServerlessHyperparametersConfig] = Field(
        default=None,
        description="Hyperparameters for serverless model customization",
    )


class DataConfig(BaseModel):
    train_path: Optional[str] = Field(
        default=None,
        description="Path to training dataset (S3 URI or local). "
        "Required for aws_training_job and local backends. "
        "Optional for aws_serverless when training_dataset_name is used.",
    )
    eval_path: Optional[str] = Field(
        default=None, description="Path to eval dataset (optional)"
    )
    eval_split_size: int = Field(default=100, ge=1)

    @field_validator("train_path")
    @classmethod
    def validate_train_path(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v:
            raise ValueError("train_path must not be empty when set")
        return v


class OutputConfig(BaseModel):
    s3_path: Optional[str] = Field(
        default=None,
        description="Output path (S3 URI or local directory). "
        "Required for aws_training_job and local backends. "
        "Optional for aws_serverless.",
    )
    checkpoint_s3_path: Optional[str] = Field(
        default=None,
        description="Mid-run checkpoint URI. Homogeneous SageMaker jobs "
        "two-way-sync /opt/ml/checkpoints via CheckpointConfig. Mixed "
        "dedicated_vllm omits that agent; training ranks aws s3 sync the "
        "same URI. Required when infra.use_spot_instances is true "
        "(homogeneous only).",
    )

    @field_validator("s3_path")
    @classmethod
    def validate_output_path(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v:
            raise ValueError("s3_path must not be empty when set")
        return v


class SecretEntry(BaseModel):
    """One secret to resolve at container runtime."""

    env_var: str = Field(
        description="Target environment variable name (e.g. GENOS_APP_SECRET)"
    )
    secret_path: str = Field(
        description="IDPS secret path/key to resolve (e.g. APP_SECRET)"
    )

    @field_validator("env_var")
    @classmethod
    def validate_env_var(cls, v: str) -> str:
        if not v.replace("_", "").isalnum() or v[0].isdigit():
            raise ValueError(
                f"Invalid env var name: '{v}'. Must be a valid shell identifier."
            )
        return v


class IDPSConfig(BaseModel):
    """IDPS connection configuration (requires foresightrl-intuit extension)."""

    policy_id: str = Field(description="IDPS policy ID")
    environment: str = Field(
        default="e2e",
        description="IDPS environment label (used by the Intuit IDPS provider).",
    )
    endpoint: str = Field(
        description="Explicit IDPS VKM endpoint hostname.",
    )


class LocalSecretsConfig(BaseModel):
    """Local secrets provider configuration — reads secrets from a .env file."""

    env_file: str = Field(
        description="Path to a .env file (KEY=VALUE per line) containing secrets"
    )


class SecretsConfig(BaseModel):
    """Secret resolution configuration. Secrets are resolved at container runtime,
    not at launch time. Only secret paths (not values) appear in the config."""

    provider: str = Field(
        default="local",
        description="Secret provider: local | idps (requires foresightrl-intuit)",
    )
    idps: Optional[IDPSConfig] = Field(
        default=None,
        description="IDPS provider configuration (required when provider=idps)",
    )
    local: Optional[LocalSecretsConfig] = Field(
        default=None,
        description="Local provider configuration (required when provider=local)",
    )
    entries: List[SecretEntry] = Field(
        default_factory=list,
        description="Secrets to resolve: each maps a secret path to a target env var",
    )


class ContainerConfig(BaseModel):
    recipe: Optional[str] = Field(
        default=None,
        description="Named container recipe (e.g. 'qwen3', 'qwen3_5'). "
        "Resolves framework_version and pip requirements automatically.",
    )
    framework_version: Optional[str] = Field(
        default=None,
        description="DLC framework version. Set automatically when a recipe "
        "is used; falls back to '2.8.0' when neither recipe nor explicit "
        "value is provided.",
    )
    framework: str = Field(
        default="pytorch",
        description="DLC framework family: 'pytorch' or 'huggingface'. "
        "Controls which AWS Deep Learning Container image is resolved.",
    )
    image_uri: Optional[str] = Field(
        default=None,
        description="Fully-qualified container image URI. When set, overrides "
        "framework + framework_version resolution entirely.",
    )
    requirements: Optional[List[str]] = Field(
        default=None,
        description="Inline pip requirements. Overrides recipe requirements "
        "when both are set.",
    )
    extra_requirements: List[str] = Field(
        default_factory=list,
        description="Pip requirements appended to whichever list applies "
        "(inline or recipe). Use this to add a package without restating the "
        "recipe. Lines are written verbatim, so a pip directive such as "
        "'--extra-index-url <index>' is a valid entry.",
    )
    extra_packages: List[str] = Field(
        default_factory=list,
        description="Importable package names of extension packages to stage "
        "into the job alongside foresightrlcore (e.g. 'foresightrl_intuit'). "
        "Each package's source tree is copied into the container source dir "
        "and imported at runtime so its plugins register.",
    )
    resolved_requirements: Optional[List[str]] = Field(
        default=None,
        exclude=True,
        description="Resolved pip requirements (from recipe). "
        "Internal field — not serialized to YAML.",
    )
    resolved_requirements_torchrun: Optional[List[str]] = Field(
        default=None,
        exclude=True,
        description="Resolved torchrun pip requirements (from recipe). "
        "Internal field — not serialized to YAML.",
    )
    perf_test_image_uri: Optional[str] = Field(
        default=None,
        description="Container image URI used by the perf_test operation. "
        "Resolved from the recipe (DJL inference base) when a recipe is set; "
        "may be overridden explicitly to pin a different perf-test image.",
    )
    resolved_requirements_perf_test: Optional[List[str]] = Field(
        default=None,
        exclude=True,
        description="Resolved perf-test pip requirements (from recipe). "
        "Internal field — not serialized to YAML.",
    )
    optimizer_image_uri: Optional[str] = Field(
        default=None,
        description="Container image URI used by the optimizer operation. "
        "Resolved from the recipe (DJL inference base, which ships vLLM for the "
        "local-serving path) when a recipe is set; may be overridden explicitly.",
    )
    resolved_requirements_optimizer: Optional[List[str]] = Field(
        default=None,
        exclude=True,
        description="Resolved optimizer pip requirements (from recipe). "
        "Internal field — not serialized to YAML.",
    )
    image_uri_from_recipe: bool = Field(
        default=False,
        exclude=True,
        description="True when image_uri was populated by the recipe rather "
        "than set explicitly by the user. Internal field — not serialized to "
        "YAML; lets validation distinguish a real recipe/image_uri conflict "
        "from the recipe simply supplying its own base image.",
    )
    evaluate_image_uri: Optional[str] = Field(
        default=None,
        description="Container image URI used by the evaluate operation. "
        "Resolved from the recipe (DJL inference base, which ships vLLM for the "
        "local-serving path) when a recipe is set; may be overridden explicitly.",
    )
    resolved_requirements_evaluate: Optional[List[str]] = Field(
        default=None,
        exclude=True,
        description="Resolved evaluate pip requirements (from recipe). "
        "Internal field — not serialized to YAML.",
    )

    def with_extras(self, reqs: Optional[List[str]]) -> Optional[List[str]]:
        """Append :attr:`extra_requirements` to an already-resolved list.

        Every launcher picks its base list differently (inline, or one of the
        per-operation recipe lists) but appends the same way, so the append
        lives here rather than being re-derived five times.

        ``None`` in means "no requirements file at all" and is preserved --
        unless extras were asked for, in which case they are the file.
        """
        if not self.extra_requirements:
            return reqs
        return [*(reqs or []), *self.extra_requirements]

    @field_validator("framework")
    @classmethod
    def validate_framework(cls, v: str) -> str:
        allowed = {"pytorch", "huggingface"}
        if v not in allowed:
            raise ValueError(
                f"Unknown framework '{v}'. Must be one of: {sorted(allowed)}"
            )
        return v

    @model_validator(mode="after")
    def _apply_recipe(self) -> "ContainerConfig":
        from foresightrlcore.config.recipes import resolve_container_recipe

        user_set_image_uri = self.image_uri is not None
        resolve_container_recipe(self)
        self.image_uri_from_recipe = (
            not user_set_image_uri and self.image_uri is not None
        )
        return self


class BedrockClientConfig(BaseModel):
    """Bedrock-backed LLM client settings.

    Auth is via the SageMaker execution role (or local AWS credentials) —
    no app secret is required. ``region`` falls back to
    ``AWS_DEFAULT_REGION`` when unset.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(
        description=(
            "Bedrock model ID, e.g. " "'us.anthropic.claude-sonnet-4-5-20250929-v1:0'."
        )
    )
    region: Optional[str] = Field(
        default=None,
        description="AWS region. Falls back to AWS_DEFAULT_REGION when unset.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0–2.0).",
    )


class GenosClientConfig(BaseModel):
    """Genos-backed LLM client settings (Intuit LLM Execution API)."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(
        description="Genos model ID, e.g. 'anthropic.claude-opus-4-5-20251101-v1-0'."
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0–2.0).",
    )


class LLMClientConfig(BaseModel):
    """A single provider-agnostic LLM client configuration.

    Embedded as a value inside :class:`LLMClientsConfig` (the
    ``task.config.llm_clients`` mapping). Selects one provider and
    holds its provider-specific settings.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["bedrock", "genos"] = Field(
        description="LLM provider: 'bedrock' (core) or 'genos' (requires foresightrl-intuit).",
    )
    bedrock: Optional[BedrockClientConfig] = Field(
        default=None,
        description="Bedrock client settings (required when provider='bedrock').",
    )
    genos: Optional[GenosClientConfig] = Field(
        default=None,
        description="Genos client settings (required when provider='genos').",
    )

    @model_validator(mode="after")
    def _check_provider_subkey(self) -> "LLMClientConfig":
        sub = getattr(self, self.provider)
        if sub is None:
            raise ValueError(
                f"llm_clients.<name>.provider={self.provider!r} but no "
                f"'{self.provider}:' subkey was provided"
            )
        return self


class LLMClientsConfig(RootModel[Dict[str, LLMClientConfig]]):
    """Mapping of task-defined client names to :class:`LLMClientConfig`.

    Single-client tasks use the name ``default`` by convention. Multi-client
    tasks use any names that make sense for them (e.g. ``reward_judge`` +
    ``safety_judge``). Providers can be mixed across clients within the
    same task.
    """

    @model_validator(mode="after")
    def _non_empty(self) -> "LLMClientsConfig":
        if not self.root:
            raise ValueError("llm_clients must contain at least one named client")
        return self

    def get(self, name: str) -> LLMClientConfig:
        """Return the client with the given ``name`` or raise a clear error."""
        if name not in self.root:
            raise ValueError(
                f"llm_clients.{name!r} is required but not configured. "
                f"Available clients: {sorted(self.root.keys())}"
            )
        return self.root[name]

    def __iter__(self) -> Any:
        """Iterate over configured :class:`LLMClientConfig` values."""
        return iter(self.root.values())


class TaskConfig(BaseModel):
    type: str = Field(
        description="Task module name (e.g. 'advice', 'matching', 'external')"
    )
    config: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# New Phase 1 additions
# ---------------------------------------------------------------------------


class MethodConfig(BaseModel):
    type: str = Field(
        default="grpo",
        description="Training method: grpo | sft_verl | sft | dpo | rlvr | rlaif",
    )


class ServerlessConfig(BaseModel):
    """Serverless model customization settings (SageMaker SDK v3).

    Lives under ``execution.serverless`` in YAML. Covers model selection,
    AI Registry resource references, and future-proofing fields for
    evaluation and deployment.
    """

    model_id: str = Field(
        description="SageMaker Hub model ID (e.g. 'huggingface-llm-qwen2-5-7b-instruct')"
    )
    training_type: str = Field(
        default="lora",
        description="Training type: 'lora' or 'full'",
    )
    accept_eula: bool = Field(
        default=False,
        description="Accept the model EULA (required for serverless training)",
    )

    # --- AI Registry resource references ---
    training_dataset_name: Optional[str] = Field(
        default=None,
        description="AI Registry dataset name for training",
    )
    training_dataset_source: Optional[str] = Field(
        default=None,
        description="S3 URI or local file path for auto-creating the training dataset "
        "when it does not exist in the registry",
    )
    validation_dataset_name: Optional[str] = Field(
        default=None,
        description="AI Registry dataset name for validation",
    )
    validation_dataset_source: Optional[str] = Field(
        default=None,
        description="S3 URI or local file path for auto-creating the validation dataset",
    )
    test_dataset_name: Optional[str] = Field(
        default=None,
        description="AI Registry dataset name for evaluation test set",
    )
    test_dataset_source: Optional[str] = Field(
        default=None,
        description="S3 URI or local file path for auto-creating the test dataset",
    )
    customization_technique: Optional[str] = Field(
        default=None,
        description="Dataset technique hint for DataSet.create (e.g. 'dpo'); "
        "if omitted, derived from method.type",
    )
    model_package_group: Optional[str] = Field(
        default=None,
        description="ModelPackageGroup name for auto model versioning",
    )
    model_package_group_description: Optional[str] = Field(
        default=None,
        description="Description for auto-creating the ModelPackageGroup",
    )
    reward_prompt_name: Optional[str] = Field(
        default=None,
        description="AI Registry evaluator name for the reward prompt (RLAIF)",
    )
    reward_prompt_source: Optional[str] = Field(
        default=None,
        description="S3 URI or local file path for auto-creating the reward prompt Evaluator",
    )
    reward_prompt_text: Optional[str] = Field(
        default=None,
        description="Inline reward prompt text; written to a temp file and registered "
        "as an Evaluator when the named evaluator does not exist",
    )

    # --- RLVR-specific ---
    custom_reward_function_name: Optional[str] = Field(
        default=None,
        description="AI Registry evaluator name for the custom reward function (RLVR). "
        "Resolved to an Evaluator ARN at launch time.",
    )
    custom_reward_function_source: Optional[str] = Field(
        default=None,
        description="Local file path or S3 URI for the Lambda source. When set, "
        "the framework calls Evaluator.create() at launch time if the named "
        "evaluator does not exist (same get-or-create pattern as "
        "training_dataset_source and reward_prompt_source). The SDK packages "
        "the source into a Lambda automatically.",
    )
    custom_reward_function_role: Optional[str] = Field(
        default=None,
        description="IAM role ARN used as the Lambda execution role when auto-"
        "creating the evaluator from custom_reward_function_source. Required "
        "when custom_reward_function_source is set. The role must allow "
        "lambda.amazonaws.com to assume and have the model-invocation "
        "permissions the reward function needs (e.g. bedrock:InvokeModel).",
    )
    custom_reward_function_extra_files: List[str] = Field(
        default_factory=list,
        description="Additional local file paths to bundle into the Lambda zip "
        "alongside custom_reward_function_source. Each file is placed at the "
        "zip root using its basename, so the Lambda can import it as a "
        "top-level module (e.g. 'foresightrlcore/tasks/advice_judge_logic.py' "
        "is importable as 'from advice_judge_logic import ...'). When this "
        "list is non-empty the framework bypasses the SageMaker SDK's "
        "single-file zip path and creates the Lambda directly via boto3, "
        "then registers it with Evaluator.create(source=<lambda_arn>).",
    )

    # --- RLAIF-specific ---
    reward_model_id: Optional[str] = Field(
        default=None,
        description="Reward model ID for RLAIF (e.g. 'openai.gpt-oss-120b-1:0')",
    )

    # --- MLflow integration ---
    mlflow_experiment_name: Optional[str] = Field(default=None)
    mlflow_run_name: Optional[str] = Field(default=None)
    mlflow_resource_arn: Optional[str] = Field(
        default=None,
        description="MLflow tracking server ARN. If unset, uses default MLflow experience.",
    )

    # --- Job configuration ---
    base_job_name: Optional[str] = Field(
        default=None,
        description="Custom prefix for the SageMaker job name. "
        "If unset, derived from model_id + method.",
    )
    kms_key_id: Optional[str] = Field(
        default=None,
        description="KMS key ID for encrypting training job outputs",
    )
    encrypt_inter_container_traffic: bool = Field(
        default=True,
        description="Encrypt traffic between the ML compute instances backing "
        "the serverless job (TLS). Maps to SageMaker "
        "EnableInterContainerTrafficEncryption. Required by Intuit ACP policy "
        "IS031.1 (AwsSagemakerTrainingjob must be encrypted in-transit). "
        "The SDK v3 serverless trainers do not expose this parameter, so the "
        "framework injects it into TrainingJob.create at submit time. Set to "
        "false to restore stock SDK behaviour if AWS rejects the parameter.",
    )
    networking: Optional[VpcConfig] = Field(
        default=None,
        description="VPC configuration (subnets + security groups) for job isolation",
    )
    max_runtime_seconds: Optional[int] = Field(
        default=None,
        ge=1,
        description="Max training runtime in seconds. Default is SageMaker service limit (24h).",
    )
    wait_timeout: Optional[int] = Field(
        default=None,
        ge=1,
        description="Max seconds to wait for job completion when wait=True",
    )
    tags: Optional[Dict[str, str]] = Field(
        default=None,
        description="AWS resource tags applied to the training job",
    )

    # --- Evaluation settings (future Phase 2) ---
    evaluator_model: Optional[str] = Field(
        default=None,
        description="Model ID for LLM-as-a-Judge (e.g. 'amazon.nova-pro-v1:0')",
    )
    builtin_metrics: Optional[List[str]] = Field(
        default=None,
        description="Built-in metric names for evaluation",
    )
    custom_metrics: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Custom metric definitions for evaluation",
    )
    benchmark: Optional[str] = Field(
        default=None,
        description="Benchmark name for BenchMarkEvaluator (e.g. 'MATH')",
    )

    # --- Deployment settings (future Phase 3) ---
    inference_instance_type: Optional[str] = Field(
        default=None,
        description="Instance type for deployment (e.g. 'ml.g5.2xlarge')",
    )
    inference_container_version: Optional[str] = Field(
        default=None,
        description="DJL LMI container version for deployment",
    )


class ServerlessHyperparametersConfig(BaseModel):
    """Hyperparameters passed through to the SageMaker serverless trainer.

    Lives under ``training.serverless`` in YAML.
    """

    global_batch_size: Optional[int] = Field(default=None, ge=1)
    max_epochs: Optional[int] = Field(default=None, ge=1)
    learning_rate: Optional[float] = Field(default=None, gt=0)
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional pass-through hyperparameters",
    )


class HyperPodEksConfig(BaseModel):
    """HyperPod EKS execution backend configuration."""

    cluster_name: str = Field(
        default="",
        description="SageMaker HyperPod cluster name",
    )
    eks_cluster_name: str = Field(
        default="",
        description="EKS cluster name (for kubeconfig context)",
    )
    namespace: str = Field(
        default="kubeflow",
        description="Kubernetes namespace for training jobs",
    )
    image_uri: str = Field(
        description="Container image URI (ECR) with training code and dependencies",
    )
    service_account: str = Field(
        default="default",
        description="Kubernetes service account for pods",
    )
    fsx_filesystem_id: str = Field(
        default="",
        description="FSx for Lustre filesystem ID for data/checkpoints",
    )
    fsx_mount_path: str = Field(
        default="/fsx",
        description="Mount path for FSx volume inside containers",
    )
    storage_class: str = Field(
        default="fsx-sc",
        description="Kubernetes storage class for FSx PVC",
    )
    pvc_name: str = Field(
        default="",
        description="Existing PVC name (if not using fsx_filesystem_id)",
    )
    node_selector: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional node selector labels",
    )
    tolerations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Pod tolerations for scheduling",
    )
    restart_policy: str = Field(
        default="OnFailure",
        description="Pod restart policy: OnFailure | Never",
    )
    max_restarts: int = Field(
        default=3,
        ge=0,
        description="Maximum job restart attempts",
    )
    ssm_relay_instance_id: Optional[str] = Field(
        default=None,
        description=(
            "EC2 instance ID used as the SSM relay to reach the private EKS API "
            "server. Must be an instance inside the EKS VPC with the SSM agent "
            "running and AmazonSSMManagedInstanceCore permissions (typically the "
            "bastion host). When set, the backend automatically opens an SSM "
            "port-forward tunnel (localhost:6443 → EKS API:443) before submitting "
            "the job and tears it down afterwards. If the tunnel is already open "
            "this is a no-op. Leave unset if kubectl is already configured to reach "
            "the cluster directly (e.g. when running from the bastion host itself)."
        ),
    )
    kubeconfig_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to a kubeconfig file for the HyperPod EKS cluster. Required "
            "alongside ssm_relay_instance_id. Relative paths are resolved "
            "against the working directory; ~ is expanded. The launcher exports "
            "KUBECONFIG=<this path> on every kubectl/SSM call — your home "
            "directory is not touched. See intuit/examples/kubeconfigs/README.md for "
            "the SSM-tunneled kubeconfig idiom and a working sample."
        ),
    )


class ExecutionConfig(BaseModel):
    """Execution backend selection + provider-specific config."""

    type: str = Field(
        default="aws_training_job",
        description="Execution backend: aws_training_job | aws_hyperpod_eks | local | ...",
    )
    local: Optional[Dict[str, Any]] = Field(default=None)
    aws: Optional[Dict[str, Any]] = Field(default=None)
    hyperpod_eks: Optional[HyperPodEksConfig] = Field(
        default=None,
        description="HyperPod EKS backend configuration",
    )
    serverless: Optional[ServerlessConfig] = Field(
        default=None,
        description="Serverless model customization config (SageMaker SDK v3)",
    )
    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class TrainingJobConfig(BaseModel):
    """Top-level configuration for a training job. Provider-neutral."""

    _source_path: str = ""
    version: int = Field(default=1)
    job: JobConfig
    infra: InfraConfig = Field(default_factory=InfraConfig)
    model: ModelConfig
    method: MethodConfig = Field(default_factory=MethodConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    environment: Dict[str, str] = Field(
        default_factory=dict,
        description="Backend-agnostic env vars injected into the training process "
        "(SageMaker container or local subprocess).",
    )
    container: ContainerConfig = Field(default_factory=ContainerConfig)
    secrets: Optional[SecretsConfig] = Field(
        default=None,
        description="Secret resolution config. Secrets are resolved at runtime.",
    )
    task: Optional[TaskConfig] = Field(
        default=None,
        description="Task module config. Required for GRPO/TRL methods that "
        "load a task plugin; not used by sft_verl.",
    )

    @model_validator(mode="after")
    def _default_use_remove_padding_for_sft_verl(self) -> "TrainingJobConfig":
        """Default ``training.sft_verl.use_remove_padding`` to ``False``

        when the config is otherwise incompatible with verl's packed
        (cu_seqlens-aware) attention path, so ``model.profile: gemma4_moe``
        alone — with no ``training.sft_verl`` overrides at all — is valid
        on its own. Runs here (on the top-level config, after both
        sub-models exist) rather than inside ``ModelConfig``'s own
        validator (where ``resolve_model_profile()`` runs) because that
        validator has no visibility into the sibling ``training`` section.

        Only fills the field in when the user didn't explicitly set it
        (``model_fields_set``, same "was this explicit" check
        ``resolve_model_profile`` already uses for ``lora``/
        ``thinking_kwarg``) — an explicit ``use_remove_padding: true``
        still surfaces as a real validation error downstream (e.g. against
        ``attn_implementation: sdpa``), exactly as intended.
        """
        sft_cfg = getattr(self.training, "sft_verl", None)
        if sft_cfg is None or "use_remove_padding" in sft_cfg.model_fields_set:
            return self

        attn_impl = self.model.attn_implementation
        incompatible_attn = attn_impl is not None and attn_impl != "flash_attention_2"
        if incompatible_attn or sft_cfg.data_format == "packed":
            sft_cfg.use_remove_padding = False
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingJobConfig":
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

    def merge_overrides(self, **kwargs: Any) -> "TrainingJobConfig":
        """Return a new config with CLI overrides merged in."""
        data = self.model_dump()

        shortcut_map = {
            "model_name": ("model", "name"),
            "instance_type": ("infra", "instance_type"),
            "train_data": ("data", "train_path"),
            "eval_data": ("data", "eval_path"),
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

        return TrainingJobConfig.model_validate(data)


# Backward-compatible alias
SageMakerTrainingConfig = TrainingJobConfig

__all__ = [
    "TrainingJobConfig",
    "SageMakerTrainingConfig",
    "Topology",
    "VllmMode",
    "JobConfig",
    "InfraConfig",
    "ModelConfig",
    "TrainingConfig",
    "GrpoConfig",
    "VerlSftConfig",
    "HooksConfig",
    "DataConfig",
    "OutputConfig",
    "ContainerConfig",
    "SecretEntry",
    "IDPSConfig",
    "LocalSecretsConfig",
    "SecretsConfig",
    "TaskConfig",
    "MethodConfig",
    "ExecutionConfig",
    "HyperPodEksConfig",
    "ServerlessConfig",
    "ServerlessHyperparametersConfig",
    "LoraConfig_",
]
