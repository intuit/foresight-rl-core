"""Capability matrix validation.

Checks that method × execution × infra combinations are valid
before submitting a job. Returns human-readable error messages.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foresightrlcore.config.schema import TrainingJobConfig

logger = logging.getLogger(__name__)


_SECRET_PATTERN_KEYWORDS = {"SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "API_KEY"}


_VALID_LOSS_TYPES = frozenset(
    {"grpo", "dr_grpo", "dapo", "bnpo", "cispo", "sapo", "luspo", "vespo"}
)
_VALID_SCALE_REWARDS = frozenset({"group", "batch", "none"})
_VALID_MULTI_OBJ_AGG = frozenset({"sum_then_normalize", "normalize_then_sum"})
_VALID_IS_MODES = frozenset(
    {"token_truncate", "token_mask", "sequence_truncate", "sequence_mask"}
)


_METHOD_CONFIG_FIELDS = frozenset({"grpo", "sft_verl"})
"""Training config field names that correspond to method-specific hyperparameters."""


_SERVERLESS_METHODS = frozenset({"dpo", "rlvr", "rlaif"})


def validate_config(config: "TrainingJobConfig") -> list[str]:
    """Run all validation checks. Returns list of error strings (empty = valid)."""
    errors: list[str] = []

    is_serverless = config.execution.type == "aws_serverless"

    # Resolve data_points → max_steps before all other checks so downstream
    # validators see the derived max_steps value.
    dp_errors = _resolve_data_points(config, is_serverless=is_serverless)
    if dp_errors:
        return dp_errors

    if is_serverless:
        errors.extend(_validate_serverless(config))
    else:
        errors.extend(_validate_infra(config))
        errors.extend(_validate_method_infra(config))
        errors.extend(_validate_grpo(config))
        errors.extend(_validate_sft_verl(config))
        errors.extend(_validate_data_required(config))

    errors.extend(_validate_mixed_dedicated_vllm(config))
    errors.extend(_validate_method_config_alignment(config))
    errors.extend(_validate_execution(config))
    errors.extend(_validate_container(config))
    errors.extend(_validate_secrets(config))
    return errors


def _resolve_data_points(
    config: "TrainingJobConfig", *, is_serverless: bool
) -> list[str]:
    """Derive max_steps from data_points + infra GPU layout.

    Mutates config.training.max_steps in-place when data_points is set so
    that the computed value is baked into the serialized YAML uploaded to
    SageMaker or written locally.

    Returns a non-empty error list if both max_steps and data_points are set
    (mutually exclusive). Returns an empty list otherwise.
    """
    import math

    from foresightrlcore.config.profiles import compute_training_processes

    training = config.training
    has_max_steps = training.max_steps is not None and training.max_steps > 0
    has_data_points = training.data_points is not None

    if not has_data_points:
        return []

    if config.method.type == "sft_verl":
        return [
            "training.data_points is not supported for method.type "
            "'sft_verl' — it is derived from TRL-shaped fields "
            "(per_device_batch_size * gradient_accumulation_steps) that "
            "sft_verl does not read, so the computed max_steps would be "
            "silently wrong for verl's own training.sft_verl.train_batch_size. "
            "Set training.max_steps directly instead."
        ]

    if has_max_steps:
        return [
            "training.max_steps and training.data_points are mutually exclusive. "
            "Remove one of them from the config."
        ]

    if is_serverless:
        logger.warning(
            "training.data_points is not supported for aws_serverless — "
            "the field will be ignored. Use training.serverless.max_epochs instead."
        )
        return []

    total_procs = compute_training_processes(config)
    effective_batch = (
        training.per_device_batch_size
        * total_procs
        * training.gradient_accumulation_steps
    )
    derived_steps = math.ceil(training.data_points / effective_batch)  # type: ignore[operator]
    training.max_steps = derived_steps

    logger.info(
        "Resolved data_points=%d → max_steps=%d "
        "(training_procs=%d, per_device_batch=%d, grad_accum=%d, effective_batch=%d)",
        training.data_points,
        derived_steps,
        total_procs,
        training.per_device_batch_size,
        training.gradient_accumulation_steps,
        effective_batch,
    )
    return []


def _validate_serverless(config: "TrainingJobConfig") -> list[str]:
    """Validate serverless-specific configuration."""
    errors: list[str] = []
    sl = config.execution.serverless

    if sl is None:
        errors.append(
            "execution.serverless section is required "
            "when execution.type is 'aws_serverless'."
        )
        return errors

    method = config.method.type
    if method not in _SERVERLESS_METHODS:
        errors.append(
            f"Serverless backend supports methods {sorted(_SERVERLESS_METHODS)}, "
            f"got '{method}'."
        )

    if not sl.accept_eula:
        errors.append(
            "execution.serverless.accept_eula must be true "
            "for serverless model customization."
        )

    valid_training_types = {"lora", "full"}
    if sl.training_type.lower() not in valid_training_types:
        errors.append(
            f"execution.serverless.training_type must be one of "
            f"{sorted(valid_training_types)}, got '{sl.training_type}'."
        )

    has_dataset_name = bool(sl.training_dataset_name)
    has_train_path = bool(config.data and config.data.train_path)
    if not has_dataset_name and not has_train_path:
        errors.append(
            "Either execution.serverless.training_dataset_name "
            "or data.train_path must be set."
        )

    if method == "rlaif":
        # RLAIF needs a judge model. AWS guidance: prefer setting
        # `judge_model_id` as a hyperparameter (under
        # `training.serverless.extra`) so the SDK doesn't validate it
        # against its allowlist; `reward_model_id` is the legacy path
        # and only works for allowlisted Bedrock models. Accept either.
        sl_hp = (
            getattr(config.training, "serverless", None) if config.training else None
        )
        extra = sl_hp.extra if sl_hp else {}
        has_judge = bool(sl.reward_model_id) or "judge_model_id" in extra
        if not has_judge:
            errors.append(
                "RLAIF requires a judge model: set either "
                "execution.serverless.reward_model_id (allowlisted Bedrock IDs) "
                "or training.serverless.extra.judge_model_id (any Bedrock ID, "
                "including inference profiles)."
            )
        if not sl.reward_prompt_name:
            errors.append(
                "execution.serverless.reward_prompt_name is required for RLAIF."
            )

    if sl.reward_prompt_text and sl.reward_prompt_source:
        logger.warning(
            "Both reward_prompt_text and reward_prompt_source are set; "
            "reward_prompt_text takes precedence."
        )

    if sl.custom_reward_function_source and not sl.custom_reward_function_role:
        errors.append(
            "execution.serverless.custom_reward_function_role is required "
            "when custom_reward_function_source is set (the Lambda needs an "
            "execution role)."
        )

    if sl.custom_reward_function_extra_files:
        if not sl.custom_reward_function_source:
            errors.append(
                "execution.serverless.custom_reward_function_extra_files is set "
                "but custom_reward_function_source is not. Extra files are "
                "bundled alongside the main source — set both, or neither."
            )
        for extra in sl.custom_reward_function_extra_files:
            if not os.path.isfile(extra):
                errors.append(
                    f"execution.serverless.custom_reward_function_extra_files "
                    f"references a path that does not exist or is not a file: "
                    f"{extra!r}"
                )

    if (
        sl.custom_reward_function_name
        and not sl.custom_reward_function_source
        and method == "rlvr"
    ):
        # Static check: when no `source` escape hatch is provided, the
        # name must resolve through the in-package registry (the live
        # AI Registry is checked at backend launch time, which we can't
        # reach from validation).
        try:
            from foresightrlcore.serverless import registry

            registry.discover_tasks()
            registered = sorted(registry.LAMBDAS)
        except ImportError:
            registered = []

        if sl.custom_reward_function_name not in registered:
            errors.append(
                "execution.serverless.custom_reward_function_name "
                f"{sl.custom_reward_function_name!r} is not in the in-package "
                f"registry and no custom_reward_function_source escape hatch "
                f"is set. Either register a LambdaSpec under that name in "
                f"foresightrlcore/tasks/<task>/__init__.py, or set "
                f"custom_reward_function_source to a .py path / S3 URI. "
                f"Registered: {registered!r}. (The live AI Registry "
                "evaluator is checked at launch time; if it exists this "
                "validation error can be ignored.)"
            )

    if (
        sl.custom_reward_function_source
        and sl.custom_reward_function_name
        and method == "rlvr"
    ):
        try:
            from foresightrlcore.serverless import registry

            registry.discover_tasks()
            if sl.custom_reward_function_name in registry.LAMBDAS:
                logger.warning(
                    "execution.serverless.custom_reward_function_source is set "
                    "AND %r is registered in foresightrlcore.serverless."
                    "registry.LAMBDAS — the explicit source will win (--source "
                    "beats registry).",
                    sl.custom_reward_function_name,
                )
        except ImportError:
            pass

    valid_techniques = {"dpo", "sft", "rlvr", "rlaif"}
    if (
        sl.customization_technique
        and sl.customization_technique.lower() not in valid_techniques
    ):
        errors.append(
            f"execution.serverless.customization_technique must be one of "
            f"{sorted(valid_techniques)}, got '{sl.customization_technique}'."
        )

    for ds_label in ("training", "validation", "test"):
        ds_source = getattr(sl, f"{ds_label}_dataset_source", None)
        ds_name = getattr(sl, f"{ds_label}_dataset_name", None)
        if ds_source and not ds_name:
            errors.append(
                f"execution.serverless.{ds_label}_dataset_name is required "
                f"when {ds_label}_dataset_source is set."
            )

    return errors


def _validate_data_required(config: "TrainingJobConfig") -> list[str]:
    """Ensure data.train_path and output.s3_path are set for non-serverless backends.

    sft_verl's merge_lora_only mode reads a checkpoint from
    output.checkpoint_s3_path and never touches training data at all — see
    VerlSftMethodHandler.validate() for the merge-specific checks (it
    requires output.checkpoint_s3_path instead).
    """
    errors: list[str] = []
    sft_cfg = getattr(config.training, "sft_verl", None)
    is_merge_only = (
        config.method.type == "sft_verl"
        and sft_cfg is not None
        and sft_cfg.merge_lora_only
    )
    if not is_merge_only and (not config.data or not config.data.train_path):
        errors.append("data.train_path is required for non-serverless backends.")
    if not config.output or not config.output.s3_path:
        errors.append("output.s3_path is required for non-serverless backends.")
    return errors


def _validate_mixed_dedicated_vllm(config: "TrainingJobConfig") -> list[str]:
    """Validate the opt-in mixed-instance dedicated_vllm path.

    ``vllm_instance_type`` is ignored unless topology is dedicated_vllm; setting
    it on any other topology (or backend) is an error so operators cannot
    silently launch a homogeneous job they thought was mixed.
    """
    from foresightrlcore.config.profiles import (
        gpu_family,
        gpus_for_instance_type,
        instance_has_efa,
    )

    errors: list[str] = []
    infra = config.infra
    if infra.vllm_instance_type is None:
        return errors

    if infra.topology.value != "dedicated_vllm":
        errors.append(
            "infra.vllm_instance_type is only valid with "
            "topology: dedicated_vllm "
            f"(got topology='{infra.topology.value}')."
        )
        return errors

    exec_type = config.execution.type
    if exec_type != "aws_training_job":
        errors.append(
            "Mixed-type dedicated_vllm (infra.vllm_instance_type) is only "
            f"supported on execution.type='aws_training_job' (got '{exec_type}')."
        )
        return errors

    train_count = infra.mixed_training_instance_count()
    if train_count < 1:
        errors.append(
            "dedicated_vllm with vllm_instance_type requires "
            "instance_count > vllm_instance_count so at least one training "
            f"node remains (got instance_count={infra.instance_count}, "
            f"vllm_instance_count={infra.vllm_instance_count})."
        )

    if infra.vllm_instance_count != 1:
        errors.append(
            "dedicated_vllm with vllm_instance_type currently supports "
            "vllm_instance_count=1 only (one dedicated vLLM node). "
            f"Got vllm_instance_count={infra.vllm_instance_count}."
        )

    vllm_gpus = gpus_for_instance_type(infra.vllm_instance_type)
    tp = infra.vllm.tensor_parallel_size
    if tp > vllm_gpus:
        errors.append(
            "infra.vllm.tensor_parallel_size "
            f"({tp}) exceeds GPUs on vllm_instance_type "
            f"'{infra.vllm_instance_type}' ({vllm_gpus})."
        )

    if infra.keep_alive_seconds > 0:
        errors.append(
            "infra.keep_alive_seconds is not supported with mixed-type "
            "dedicated_vllm (SageMaker warm pools are incompatible with "
            "InstanceGroups). Set keep_alive_seconds: 0."
        )

    vllm_family = gpu_family(infra.vllm_instance_type)
    train_family = gpu_family(infra.instance_type)
    vllm_efa = instance_has_efa(infra.vllm_instance_type)
    train_efa = instance_has_efa(infra.instance_type)
    family_mismatch = vllm_family != train_family
    efa_mismatch = vllm_efa != train_efa
    if (family_mismatch or efa_mismatch) and not infra.allow_mixed_gpu_families:
        reasons = []
        if family_mismatch:
            reasons.append(
                f"GPU family '{vllm_family}' (vLLM) vs '{train_family}' (training)"
            )
        if efa_mismatch:
            reasons.append(
                f"EFA {'present' if vllm_efa else 'absent'} on vLLM vs "
                f"{'present' if train_efa else 'absent'} on training"
            )
        errors.append(
            "Mixed-type dedicated_vllm mixes incompatible hardware "
            f"({'; '.join(reasons)}). TRL weight-sync uses NCCL between "
            "the vLLM and training nodes. Use matching families (e.g. "
            "g6e.12xlarge + g6e.48xlarge, or p4d + p4de), or set "
            "infra.allow_mixed_gpu_families: true to override."
        )
    elif family_mismatch:
        logger.warning(
            "infra.allow_mixed_gpu_families is true: vLLM instance %s "
            "(family %s) and training instance %s (family %s) share one "
            "container image. Confirm the image runs on both GPU architectures.",
            infra.vllm_instance_type,
            vllm_family,
            infra.instance_type,
            train_family,
        )

    vpc = infra.vpc
    has_vpc = vpc is not None and bool(vpc.subnets or vpc.security_groups)
    if has_vpc and (vllm_efa or train_efa):
        efa_types = [
            t
            for t, has_efa in (
                (infra.vllm_instance_type, vllm_efa),
                (infra.instance_type, train_efa),
            )
            if has_efa
        ]
        errors.append(
            "SageMaker heterogeneous clusters (InstanceGroups) only support "
            "non-EFA instance types when VPC is enabled. Use a non-EFA pair "
            "(e.g. ml.g5.2xlarge + ml.g5.4xlarge), or omit infra.vpc. "
            f"EFA instance types: {', '.join(efa_types)}."
        )

    return errors


def _validate_infra(config: "TrainingJobConfig") -> list[str]:
    errors: list[str] = []
    infra = config.infra

    if infra.topology.value == "dedicated_vllm" and infra.instance_count < 2:
        errors.append(
            "dedicated_vllm topology requires instance_count >= 2 "
            "(node 0 runs vLLM, node 1+ runs training)."
        )

    if infra.vllm.speculative is not None and infra.vllm_mode.value != "server":
        # Speculative decoding is wired only into the `trl vllm-serve` argv
        # (server mode). Colocate launches vLLM inside the TRL trainer process
        # and TRL's GRPOConfig has no vllm_speculative_config kwarg today.
        logger.warning(
            "infra.vllm.speculative is only honored in vllm_mode='server' "
            "(got '%s'); the field will be ignored.",
            infra.vllm_mode.value,
        )

    if infra.vllm.reasoning_parser is not None and infra.vllm_mode.value != "server":
        # Reasoning parser is injected into TRL's vllm-serve LLM() call via
        # patch_trl_vllm.py — server mode only. Colocate / none paths don't
        # invoke trl vllm-serve and TRL's GRPOConfig has no equivalent kwarg.
        logger.warning(
            "infra.vllm.reasoning_parser is only honored in vllm_mode='server' "
            "(got '%s'); the field will be ignored.",
            infra.vllm_mode.value,
        )

    if infra.vllm.compilation_config is not None and infra.vllm_mode.value != "server":
        # Same wiring story as reasoning_parser — only the server-mode path
        # patches TRL's vllm_serve.py to forward CompilationConfig to LLM().
        logger.warning(
            "infra.vllm.compilation_config is only honored in vllm_mode='server' "
            "(got '%s'); the field will be ignored.",
            infra.vllm_mode.value,
        )

    if infra.vllm.disable_custom_all_reduce and infra.vllm_mode.value != "server":
        # Same wiring story — only the server-mode path patches TRL's
        # vllm_serve.py to forward disable_custom_all_reduce=True to LLM().
        logger.warning(
            "infra.vllm.disable_custom_all_reduce is only honored in "
            "vllm_mode='server' (got '%s'); the field will be ignored.",
            infra.vllm_mode.value,
        )

    errors.extend(_validate_spot(config))

    return errors


def _validate_spot(config: "TrainingJobConfig") -> list[str]:
    """Validate SageMaker Managed Spot Training settings.

    Spot only applies to the aws_training_job backend. When enabled we require
    S3 checkpointing (so runs auto-resume across interruptions) and a wait
    window that is at least as long as the max runtime.
    """
    errors: list[str] = []
    infra = config.infra

    if not infra.use_spot_instances:
        return errors

    if config.execution.type != "aws_training_job":
        logger.warning(
            "infra.use_spot_instances is only honored by the aws_training_job "
            "backend (got execution.type='%s'); the flag will be ignored.",
            config.execution.type,
        )
        return errors

    checkpoint_path = (
        str(config.output.checkpoint_s3_path)
        if config.output.checkpoint_s3_path
        else ""
    )
    if not checkpoint_path.startswith("s3://"):
        errors.append(
            "infra.use_spot_instances requires output.checkpoint_s3_path to be "
            "set to an s3:// URI so training can auto-resume after a spot "
            f"interruption, got: {checkpoint_path!r}"
        )

    max_wait_hours = config.job.max_wait_hours
    if max_wait_hours is not None and max_wait_hours < config.job.max_runtime_hours:
        errors.append(
            "job.max_wait_hours must be >= job.max_runtime_hours when "
            f"infra.use_spot_instances is enabled (got max_wait_hours="
            f"{max_wait_hours}, max_runtime_hours={config.job.max_runtime_hours})."
        )

    if infra.topology.value == "multi_node":
        errors.append(
            "infra.use_spot_instances is not supported with the heterogeneous "
            "'multi_node' topology yet. Use single_node, multi_node_server, or "
            "homogeneous dedicated_vllm."
        )

    if infra.uses_mixed_instance_types():
        errors.append(
            "infra.use_spot_instances is not supported with mixed-type "
            "dedicated_vllm (infra.vllm_instance_type is set). Use a "
            "homogeneous dedicated_vllm cluster, or disable spot."
        )

    return errors


def _validate_method_infra(config: "TrainingJobConfig") -> list[str]:
    errors: list[str] = []
    method_type = config.method.type
    vllm_mode = config.infra.vllm_mode.value

    if method_type == "grpo" and vllm_mode == "none":
        logger.warning(
            "GRPO with vllm_mode=none uses native HF generation — "
            "may be slow for large models. Consider 'server' or 'colocate'."
        )

    if method_type in ("sft", "sft_verl", "dpo") and vllm_mode != "none":
        logger.warning(
            f"{method_type.upper()} does not need vLLM for generation. "
            f"Consider setting vllm_mode: none for simpler execution."
        )

    return errors


def _validate_method_config_alignment(config: "TrainingJobConfig") -> list[str]:
    """Ensure method.type matches the training hyperparameter block."""
    errors: list[str] = []
    method_type = config.method.type

    if method_type in _SERVERLESS_METHODS:
        if (
            config.execution.type == "aws_serverless"
            and config.training.serverless is None
        ):
            logger.info(
                "No training.serverless hyperparameters set for %s — "
                "using SageMaker defaults.",
                method_type,
            )
        return errors

    if method_type in _METHOD_CONFIG_FIELDS:
        method_cfg = getattr(config.training, method_type, None)
        if method_cfg is None:
            errors.append(
                f"method.type is '{method_type}' but training.{method_type} "
                f"section is missing."
            )

    for field_name in _METHOD_CONFIG_FIELDS:
        if field_name == method_type:
            continue
        other_cfg = getattr(config.training, field_name, None)
        if other_cfg is not None:
            logger.warning(
                "training.%s is present but method.type is '%s' — "
                "training.%s will be ignored.",
                field_name,
                method_type,
                field_name,
            )

    return errors


def _validate_grpo(config: "TrainingJobConfig") -> list[str]:
    """Validate GRPO-specific parameters when method is grpo."""
    errors: list[str] = []
    if config.method.type != "grpo":
        return errors

    grpo = config.training.grpo

    if grpo.loss_type not in _VALID_LOSS_TYPES:
        errors.append(
            f"Invalid training.grpo.loss_type '{grpo.loss_type}'. "
            f"Must be one of: {sorted(_VALID_LOSS_TYPES)}"
        )

    sr = grpo.scale_rewards
    if isinstance(sr, str) and sr not in _VALID_SCALE_REWARDS:
        errors.append(
            f"Invalid training.grpo.scale_rewards '{sr}'. "
            f"Must be one of: {sorted(_VALID_SCALE_REWARDS)} or a bool."
        )

    if grpo.multi_objective_aggregation not in _VALID_MULTI_OBJ_AGG:
        errors.append(
            f"Invalid training.grpo.multi_objective_aggregation "
            f"'{grpo.multi_objective_aggregation}'. "
            f"Must be one of: {sorted(_VALID_MULTI_OBJ_AGG)}"
        )

    if (
        grpo.vllm_importance_sampling_mode is not None
        and grpo.vllm_importance_sampling_mode not in _VALID_IS_MODES
    ):
        errors.append(
            f"Invalid training.grpo.vllm_importance_sampling_mode "
            f"'{grpo.vllm_importance_sampling_mode}'. "
            f"Must be one of: {sorted(_VALID_IS_MODES)}"
        )

    if grpo.sync_ref_model and grpo.beta <= 0:
        errors.append(
            "training.grpo.sync_ref_model requires beta > 0 "
            "(a reference model is only loaded when beta > 0)."
        )

    if grpo.loss_type == "luspo" and grpo.importance_sampling_level != "sequence":
        errors.append(
            "loss_type 'luspo' requires importance_sampling_level='sequence'."
        )

    if config.training.bf16 and config.training.fp16:
        errors.append("bf16 and fp16 cannot both be True.")

    return errors


def _validate_sft_verl(config: "TrainingJobConfig") -> list[str]:
    """Validate sft_verl-specific parameters when method is sft_verl.

    This is the single, canonical implementation of sft_verl validation
    — runs here, in validate_config() (called by both ``rl-train
    validate`` and ``rl-train train`` before a job is ever submitted),
    and is *delegated to* by ``VerlSftMethodHandler.validate()``
    (``methods/sft_verl.py``), which used to be a full independent
    ~150-line duplicate of these exact checks. Deliberately kept here
    rather than in ``methods/`` because this codebase's dependency
    direction is one-way (``methods/`` imports from ``config/``, never
    the reverse) — the handler importing this function respects that;
    the other direction would not have.

    The use_remove_padding/attn_implementation check in particular is a
    correctness check, not just a config-shape check: the wrong
    combination trains with silently corrupted gradients rather than
    crashing, so it must not be reachable only via a code path nothing
    actually calls. The train_batch_size/data-parallel-size check, the
    lora.dropout/target_parameters check, and the flat/messages train_path/
    eval_path collision check below are the same kind of thing from the
    other direction — they fail loudly at validate/train time instead of
    crashing (or silently truncating the global batch, or silently
    leaking eval into training) only after a real cluster is already
    provisioned and billed.
    """
    errors: list[str] = []
    if config.method.type != "sft_verl":
        return errors

    sft_cfg = getattr(config.training, "sft_verl", None)
    if sft_cfg is None:
        return errors

    if config.infra.vllm_mode.value != "none":
        errors.append(
            "sft_verl requires infra.vllm_mode: none "
            "(verl's SFT trainer does not use vLLM)."
        )

    # merge_lora_only never runs the dataset/attention/FSDP path at all (it
    # only reconstructs and merges an existing checkpoint on a single
    # process), so none of the checks below apply to it — mirrors
    # VerlSftMethodHandler.validate()'s early return for the same reason.
    if sft_cfg.merge_lora_only:
        if not config.output or not config.output.checkpoint_s3_path:
            errors.append(
                "training.sft_verl.merge_lora_only requires "
                "output.checkpoint_s3_path (the checkpoint to merge)."
            )
        return errors

    if not config.output or not config.output.checkpoint_s3_path:
        # See VerlSftMethodHandler.validate() for the full root-cause
        # explanation — verl's SFTTrainer only ever writes checkpoints to
        # config._local_output_dir; without output.checkpoint_s3_path,
        # aws_sagemaker_launcher.py never configures SageMaker's
        # CheckpointConfig S3 sync, so the LoRA checkpoint is destroyed
        # with the container when the job ends. A "successful" job with
        # no durable output anywhere is worse than a hard error here.
        errors.append(
            "output.checkpoint_s3_path is required for sft_verl training "
            "(verl only writes checkpoints there — without it, LoRA "
            "checkpoints are never synced to S3 and are lost when the "
            "job's instance terminates)."
        )

    if sft_cfg.ulysses_sequence_parallel_size > 1 and not sft_cfg.use_remove_padding:
        errors.append(
            "training.sft_verl.use_remove_padding must be true when "
            "ulysses_sequence_parallel_size > 1 "
            f"(got {sft_cfg.ulysses_sequence_parallel_size})."
        )

    attn_impl = config.model.attn_implementation
    if (
        attn_impl is not None
        and attn_impl != "flash_attention_2"
        and sft_cfg.use_remove_padding
    ):
        # CORRECTNESS, not just efficiency — see
        # VerlSftMethodHandler.validate() for the full root-cause
        # explanation (verl's remove-padding path relies entirely on
        # flash-attn's cu_seqlens-aware varlen kernel to stop attention
        # crossing packed sequence boundaries; any other backend would
        # silently let packed examples attend across each other, with
        # wrong gradients and no crash).
        errors.append(
            "training.sft_verl.use_remove_padding must be false when "
            f"model.attn_implementation={attn_impl!r} (only "
            "flash_attention_2 supports verl's cu_seqlens-aware packed "
            "attention; any other backend would silently let packed "
            "examples attend across each other's boundaries)."
        )

    if sft_cfg.data_format == "packed" and sft_cfg.use_remove_padding:
        # CORRECTNESS, independent of the attn_implementation check above.
        # data_format=packed rows already contain multiple documents
        # concatenated by the offline packer, with correctness resting
        # entirely on position_ids resets at each document boundary (see
        # PackedSFTDataset's module docstring). Under use_remove_padding,
        # verl computes cu_seqlens from the *row* boundaries of the
        # micro-batch (input_ids.offsets() in
        # verl/workers/engine/fsdp/transformer_impl.py) and passes them
        # straight through to transformers' _flash_attention_forward,
        # which — confirmed against transformers 5.10.4's
        # modeling_flash_attention_utils.py — skips its own
        # position_ids-based packed-sequence detection entirely whenever
        # cu_seq_lens_q/k are already supplied. So even with a
        # flash_attention_2-capable model, remove_padding=True would let
        # every document *within* a packed row attend across every other
        # document in that same row — wrong gradients with no crash.
        errors.append(
            "training.sft_verl.use_remove_padding must be false when "
            "training.sft_verl.data_format='packed' (verl's remove-padding "
            "path only isolates attention at the micro-batch ROW boundary, "
            "not between documents packed within the same row — position_ids "
            "resets, which is what actually separates packed documents, are "
            "never consulted once cu_seqlens is supplied directly)."
        )

    if sft_cfg.data_format not in ("flat", "messages", "packed"):
        errors.append(
            "training.sft_verl.data_format must be 'flat', 'messages', "
            f"or 'packed', got {sft_cfg.data_format!r}."
        )

    if (
        sft_cfg.data_format in ("flat", "messages")
        and config.data.eval_path
        and config.data.train_path
        and str(config.data.eval_path).rstrip("/")
        == str(config.data.train_path).rstrip("/")
    ):
        # CORRECTNESS, not just a config-shape check — only data_format=
        # "packed" has a training_*/validation_* filename split
        # (_glob_data_files' filename_prefix, produced by rl-train
        # tokenize-pack) that makes pointing train_path/eval_path at the
        # SAME prefix safe. For "flat"/"messages", _glob_data_files is
        # called with no filename_prefix at all (see run()), so identical
        # train_path/eval_path glob the exact same files for both splits
        # — eval silently measures training-set fit, not generalization,
        # with no crash and no warning otherwise. rstrip("/") so
        # "s3://b/d" and "s3://b/d/" — the same prefix, differing only by
        # a trailing slash neither DataConfig nor this check otherwise
        # normalizes — are still caught as the same path.
        errors.append(
            "data.train_path and data.eval_path must be different prefixes "
            f"when training.sft_verl.data_format={sft_cfg.data_format!r} "
            "(got the same path for both) — unlike data_format='packed', "
            "there is no training_*/validation_* filename split for this "
            "format, so identical paths would silently train and evaluate "
            "on the exact same files (eval leaks into training, with no "
            "crash)."
        )

    if sft_cfg.fsdp_strategy not in ("fsdp", "fsdp2"):
        errors.append(
            "training.sft_verl.fsdp_strategy must be 'fsdp' or 'fsdp2', "
            f"got {sft_cfg.fsdp_strategy!r}."
        )

    lora = getattr(config.model, "lora", None)
    if lora is not None and lora.target_parameters and lora.dropout != 0.0:
        # PEFT's ParamWrapper (peft.tuners.lora.layer.ParamWrapper.__init__,
        # dispatched for every model.lora.target_parameters entry — e.g.
        # Gemma 4 MoE's fused experts.gate_up_proj/down_proj, 3D
        # nn.Parameter tensors that target_modules matching can never reach
        # at all) unconditionally raises `ValueError: lora.ParamWrapper does
        # not work with lora_dropout != 0` at construction time — confirmed
        # directly against the real peft source. peft.LoraConfig.lora_dropout
        # is a single model-wide value, not settable per-target, so there is
        # no way to keep dropout on the target_modules-based adapters while
        # zeroing it only for the target_parameters ones; it must be 0 for
        # the whole config whenever ANY target_parameters entry is present.
        # Reproduced directly on a real SageMaker job
        # (sft-verl-gemma4-26ba4b-sft-verl-smoke-packed-20260929153449): the
        # earlier flash-attn fix (see _ensure_flash_attn_compatible) let the
        # job get all the way to LoRA injection this time, only to crash
        # there instead — a cluster already billed for ~13 minutes before
        # this was caught.
        errors.append(
            "model.lora.dropout must be 0.0 when model.lora.target_parameters "
            f"is non-empty (got dropout={lora.dropout}, target_parameters="
            f"{list(lora.target_parameters)}) — PEFT's ParamWrapper, used to "
            "adapt target_parameters (e.g. Gemma 4 MoE's fused expert "
            "weights), unconditionally rejects any nonzero lora_dropout, and "
            "LoraConfig.lora_dropout is one model-wide value shared across "
            "every LoRA target, not settable per-target."
        )

    if config.training.lr_scheduler not in ("cosine", "constant"):
        # See VerlSftMethodHandler.validate() for the full root-cause
        # explanation — verl's FSDPOptimizerConfig hard-asserts
        # lr_scheduler_type in ["constant", "cosine"] and nothing else;
        # run() silently maps anything else to "constant", so this must be
        # caught here rather than trained through quietly.
        errors.append(
            "training.lr_scheduler must be 'cosine' or 'constant' for "
            "sft_verl (verl's FSDP optimizer only supports these two "
            f"schedule types), got {config.training.lr_scheduler!r}."
        )

    if getattr(config, "infra", None) is not None and config.infra.instance_type:
        from foresightrlcore.config.recipes import resolve_gpu_family

        try:
            resolve_gpu_family(config.infra.instance_type)
        except ValueError as e:
            # Fail at `rl-train validate`/`rl-train train` time, not after
            # paying for GPU provisioning on a real job.
            errors.append(str(e))

        # verl's SFTTrainer computes train_batch_size_per_dp = train_batch_size
        # // dp_size (plain integer division, not an assert — verl/trainer/
        # sft_trainer.py's _build_dataloader) and hands that straight to a
        # real torch DataLoader(batch_size=...). dp_size here is world_size
        # (instance_count x GPUs/instance) divided by
        # ulysses_sequence_parallel_size. Reproduced directly against a real
        # SageMaker job: train_batch_size=4 on an 8-GPU single-node smoke
        # config floor-divided to 0, and DataLoader(batch_size=0) raised at
        # trainer setup — after the cluster was already billed. Checked here
        # (fail at validate/train time) rather than only in the example
        # configs' own comments, so this class of mistake can't recur
        # silently in a config nothing happens to review by eye.
        from foresightrlcore.config.profiles import gpus_for_instance_type

        world_size = config.infra.instance_count * gpus_for_instance_type(
            config.infra.instance_type
        )
        dp_size = world_size // sft_cfg.ulysses_sequence_parallel_size
        if dp_size <= 0:
            errors.append(
                "training.sft_verl.ulysses_sequence_parallel_size "
                f"({sft_cfg.ulysses_sequence_parallel_size}) exceeds the "
                f"total GPU count ({world_size} = infra.instance_count "
                f"{config.infra.instance_count} x "
                f"{gpus_for_instance_type(config.infra.instance_type)} "
                "GPUs/instance) — the resulting data-parallel size would be "
                "zero, which is not a valid FSDP topology."
            )
        elif sft_cfg.train_batch_size < dp_size:
            errors.append(
                f"training.sft_verl.train_batch_size ({sft_cfg.train_batch_size}) "
                f"must be >= the effective data-parallel size ({dp_size} = "
                f"{world_size} total GPUs / ulysses_sequence_parallel_size "
                f"{sft_cfg.ulysses_sequence_parallel_size}) — verl's "
                "train_batch_size // dp_size integer division would floor to "
                "0, crashing DataLoader construction at trainer setup."
            )
        elif sft_cfg.train_batch_size % dp_size != 0:
            errors.append(
                f"training.sft_verl.train_batch_size ({sft_cfg.train_batch_size}) "
                f"is not a multiple of the effective data-parallel size "
                f"({dp_size} = {world_size} total GPUs / "
                f"ulysses_sequence_parallel_size "
                f"{sft_cfg.ulysses_sequence_parallel_size}) — verl's "
                "train_batch_size // dp_size integer division would silently "
                f"truncate the global batch to "
                f"{(sft_cfg.train_batch_size // dp_size) * dp_size} instead "
                f"of {sft_cfg.train_batch_size}."
            )

    return errors


def _validate_execution(config: "TrainingJobConfig") -> list[str]:
    errors: list[str] = []
    exec_type = config.execution.type

    if exec_type == "aws_training_job":
        sft_cfg = getattr(config.training, "sft_verl", None)
        is_merge_only = (
            config.method.type == "sft_verl"
            and sft_cfg is not None
            and sft_cfg.merge_lora_only
        )
        train_path = str(config.data.train_path) if config.data.train_path else ""
        output_path = str(config.output.s3_path) if config.output.s3_path else ""
        if not is_merge_only and not train_path.startswith("s3://"):
            errors.append(
                f"AWS training jobs require s3:// paths for data.train_path, "
                f"got: {train_path!r}"
            )
        if not output_path.startswith("s3://"):
            errors.append(
                f"AWS training jobs require s3:// paths for output.s3_path, "
                f"got: {output_path!r}"
            )

    elif exec_type == "local":
        local_cfg = config.execution.local or {}
        if not local_cfg.get("gpus"):
            errors.append(
                "Local backend requires execution.local.gpus to specify GPU IDs."
            )

    elif exec_type == "aws_serverless":
        sl = config.execution.serverless
        if sl is not None and config.output.s3_path:
            output_path = str(config.output.s3_path)
            if not output_path.startswith("s3://"):
                errors.append(
                    f"Serverless output path must be an s3:// URI, "
                    f"got: {output_path!r}"
                )

    elif exec_type == "aws_hyperpod_eks":
        # The HyperPod EKS backend uses `aws s3 cp` to fetch training data and
        # the source-code tarball into worker pods, and uploads the source
        # tarball under `output.s3_path` before submitting the job. Both paths
        # must therefore be s3:// URIs — FSx-style "/fsx/..." paths look valid
        # but silently break the source upload (boto3 is called with bucket="")
        # and the in-pod `aws s3 cp` of the training data.
        eks_cfg = config.execution.hyperpod_eks
        if eks_cfg is None:
            errors.append(
                "execution.hyperpod_eks section is required "
                "when execution.type is 'aws_hyperpod_eks'."
            )
        elif not eks_cfg.image_uri:
            errors.append(
                "execution.hyperpod_eks.image_uri is required "
                "(ECR image with training code and dependencies)."
            )
        elif eks_cfg.ssm_relay_instance_id and not eks_cfg.kubeconfig_path:
            errors.append(
                "execution.hyperpod_eks.kubeconfig_path is required when "
                "ssm_relay_instance_id is set — the launcher needs a "
                "kubeconfig wired for the SSM tunnel "
                "(server: https://localhost:6443, tls-server-name: <real "
                "EKS endpoint>). See intuit/examples/kubeconfigs/README.md."
            )

        train_path = str(config.data.train_path) if config.data.train_path else ""
        output_path = str(config.output.s3_path) if config.output.s3_path else ""
        if train_path and not train_path.startswith("s3://"):
            errors.append(
                f"HyperPod EKS backend requires an s3:// URI for "
                f"data.train_path (workers fetch it via `aws s3 cp`), "
                f"got: {train_path!r}"
            )
        if output_path and not output_path.startswith("s3://"):
            errors.append(
                f"HyperPod EKS backend requires an s3:// URI for "
                f"output.s3_path (the launcher uploads the source tarball "
                f"there before submitting the job), got: {output_path!r}"
            )

        topology = config.infra.topology.value
        instance_count = config.infra.instance_count
        if topology == "dedicated_vllm" and instance_count < 2:
            errors.append(
                f"dedicated_vllm topology requires instance_count >= 2 "
                f"(node 0 runs vLLM, node 1+ runs training), got {instance_count}."
            )

        vllm_mode = config.infra.vllm_mode.value
        if vllm_mode == "server" and topology not in (
            "single_node",
            "dedicated_vllm",
            "multi_node_server",
        ):
            errors.append(
                f"HyperPod EKS: vllm_mode='server' is not supported with "
                f"topology='{topology}'. Use 'dedicated_vllm', 'single_node', "
                f"or 'multi_node_server'."
            )

    return errors


def _validate_container(config: "TrainingJobConfig") -> list[str]:
    """Warn about conflicting container overrides."""
    errors: list[str] = []
    c = config.container

    if c.recipe and c.image_uri and not c.image_uri_from_recipe:
        logger.warning(
            "container.recipe '%s' is set but container.image_uri is also set — "
            "image_uri takes precedence and the recipe's framework_version "
            "resolution is skipped for image selection.",
            c.recipe,
        )

    if c.recipe and c.requirements:
        logger.warning(
            "container.recipe '%s' is set but container.requirements is also "
            "set — inline requirements override the recipe's requirements.",
            c.recipe,
        )

    if c.recipe:
        from foresightrlcore.config.recipes import CONTAINER_RECIPES

        recipe = CONTAINER_RECIPES.get(c.recipe)
        if recipe and c.framework_version != recipe["framework_version"]:
            logger.warning(
                "container.framework_version '%s' differs from recipe '%s' "
                "default '%s'. Using the explicit value.",
                c.framework_version,
                c.recipe,
                recipe["framework_version"],
            )

    return errors


def _validate_secrets(config: "TrainingJobConfig") -> list[str]:
    errors: list[str] = []

    for source_name, env_dict in [
        ("environment", config.environment),
    ]:
        for key in env_dict:
            key_upper = key.upper()
            if any(kw in key_upper for kw in _SECRET_PATTERN_KEYWORDS):
                logger.warning(
                    "%s key '%s' looks like a secret. "
                    "Consider moving it to the 'secrets' section for secure "
                    "runtime resolution instead of plaintext env vars.",
                    source_name,
                    key,
                )

    if config.secrets is None:
        return errors

    secrets = config.secrets

    if secrets.entries and not secrets.provider:
        errors.append("secrets.provider is required when secrets.entries are defined.")

    if secrets.provider == "idps":
        if secrets.idps is None:
            errors.append(
                "secrets.idps config (policy_id, endpoint) is required "
                "when secrets.provider is 'idps'."
            )
        elif not secrets.idps.policy_id:
            errors.append("secrets.idps.policy_id must not be empty.")

    if secrets.provider == "local":
        if secrets.local is None:
            errors.append(
                "secrets.local config (env_file) is required "
                "when secrets.provider is 'local'."
            )
        elif not secrets.local.env_file:
            errors.append("secrets.local.env_file must not be empty.")

    env_vars_seen: set[str] = set()
    for entry in secrets.entries:
        if entry.env_var in env_vars_seen:
            errors.append(f"Duplicate secrets entry for env_var: {entry.env_var}")
        env_vars_seen.add(entry.env_var)

        if not entry.secret_path:
            errors.append(
                f"secrets.entries[env_var={entry.env_var}].secret_path must not be empty."
            )

    _warn_stale_genos_secret_when_no_genos_client(config)

    return errors


def _warn_stale_genos_secret_when_no_genos_client(
    config: "TrainingJobConfig",
) -> None:
    """Warn if a GENOS_APP_SECRET entry is configured but no client uses Genos.

    Iterates ``task.config.llm_clients`` values; if none have
    ``provider == 'genos'`` while ``secrets.entries`` still contains an
    entry for ``GENOS_APP_SECRET``, the secret is likely stale and can
    be removed.

    Generic — does not hardcode ``task.type``; any task using the
    ``llm_clients`` mapping gets the same check.
    """
    if config.secrets is None:
        return

    has_genos_secret = any(
        entry.env_var == "GENOS_APP_SECRET" for entry in config.secrets.entries
    )
    if not has_genos_secret:
        return

    task = config.task
    if task is None:
        return

    raw_clients = (
        task.config.get("llm_clients") if isinstance(task.config, dict) else None
    )
    if not isinstance(raw_clients, dict) or not raw_clients:
        return

    any_genos = False
    for client_cfg in raw_clients.values():
        if isinstance(client_cfg, dict) and client_cfg.get("provider") == "genos":
            any_genos = True
            break

    if not any_genos:
        logger.warning(
            "secrets.entries contains GENOS_APP_SECRET but no llm_clients entry "
            "uses provider='genos'. The Genos secret is likely stale and can be "
            "removed from the 'secrets' block."
        )
