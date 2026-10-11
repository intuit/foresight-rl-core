"""Base protocol and generic handler for training methods."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class MethodHandler(Protocol):
    """Interface that every training method must implement.

    A MethodHandler is responsible for:
      1. Validating that the config is compatible with this method
      2. Building the trainer arguments (TRL config, etc.)
      3. Running the training loop (inside the execution backend's process)
    """

    method_type: str

    def validate(self, config: Any) -> list[str]:
        """Validate method-specific config. Returns list of errors (empty = valid)."""
        ...

    def build_trainer_args(self, config: Any) -> dict[str, Any]:
        """Build method-specific trainer arguments from config."""
        ...

    def run(self, config: Any) -> None:
        """Execute the training loop. Called inside the runtime environment."""
        ...


class TrlMethodHandler:
    """Generic base handler for TRL-based training methods.

    Subclasses provide method-specific validation, trainer args, and trainer
    constructor kwargs.  The ``run()`` pipeline (task loading, datasets, LoRA,
    hooks, train/save) lives here so it never needs to be duplicated.

    Parameters
    ----------
    method_type:
        Identifier that matches ``method.type`` in the YAML config.
    default_trainer_path:
        Dotted import path to the default TRL Trainer class
        (e.g. ``"trl.GRPOTrainer"``).
    default_config_path:
        Dotted import path to the default TRL Config class
        (e.g. ``"trl.GRPOConfig"``).
    """

    method_type: str

    def __init__(
        self,
        method_type: str,
        default_trainer_path: str,
        default_config_path: str,
    ) -> None:
        self.method_type = method_type
        self._default_trainer_path = default_trainer_path
        self._default_config_path = default_config_path

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, config: Any) -> list[str]:
        """Validate shared training config. Subclasses extend via ``super()``."""
        errors: list[str] = []
        if getattr(config, "training", None) is None:
            errors.append("Training config section is missing.")
        return errors

    # ------------------------------------------------------------------
    # Trainer args
    # ------------------------------------------------------------------

    def build_trainer_args(self, config: Any) -> dict[str, Any]:
        """Build common TRL trainer-config kwargs.

        Subclasses call ``super().build_trainer_args(config)`` then add
        their method-specific fields.
        """
        training = config.training
        infra = config.infra

        vllm_mode = infra.vllm_mode.value
        vllm_cfg_dict = (
            infra.vllm.model_dump() if hasattr(infra.vllm, "model_dump") else {}
        )

        nccl_timeout_raw = int(os.environ.get("NCCL_TIMEOUT", "1800000"))
        nccl_timeout_sec = (
            nccl_timeout_raw if nccl_timeout_raw < 10000 else nccl_timeout_raw // 1000
        )

        if vllm_mode == "none":
            vllm_kwargs: dict[str, Any] = {"use_vllm": False}
        else:
            vllm_kwargs = {"use_vllm": True, "vllm_mode": vllm_mode}
            if vllm_mode == "server":
                vllm_kwargs["vllm_server_host"] = vllm_cfg_dict.get("host", "0.0.0.0")
                vllm_kwargs["vllm_server_port"] = vllm_cfg_dict.get("port", 8002)
            elif vllm_mode == "colocate":
                vllm_kwargs["vllm_gpu_memory_utilization"] = vllm_cfg_dict.get(
                    "gpu_memory_utilization", 0.9
                )
                vllm_kwargs["vllm_tensor_parallel_size"] = vllm_cfg_dict.get(
                    "tensor_parallel_size", 1
                )
                max_len = vllm_cfg_dict.get("max_model_len")
                if max_len:
                    vllm_kwargs["vllm_max_model_length"] = max_len

        output_dir = config.output.s3_path
        if hasattr(config, "_local_output_dir"):
            output_dir = config._local_output_dir

        # Pin TensorBoard event files to a deterministic local path under
        # output_dir. Without this, HF Trainer auto-generates a unique subdir
        # name per run (runs/<datetime>_<hostname>), and the post-training S3
        # sync ships everything under output_dir, but only if output_dir is
        # actually a local path. If output_dir is an s3:// URL (which happens
        # when _local_output_dir was not applied), TensorBoard silently writes
        # to a malformed path and events are lost. Logging the resolved paths
        # here so we can verify in container stdout on every run.
        logging_dir = os.path.join(str(output_dir), "runs")
        is_local_output = not str(output_dir).startswith("s3://")
        logger.info(
            f"Resolved output_dir={output_dir} (local_filesystem={is_local_output})"
        )
        logger.info(
            f"Resolved logging_dir={logging_dir} — TensorBoard events will write here"
        )
        if not is_local_output:
            logger.warning(
                "output_dir is an s3:// URL — TensorBoard events will NOT be "
                "captured. _apply_sagemaker_overrides() should have set "
                "_local_output_dir on the config; check runtime/scripts/train.py."
            )

        args: dict[str, Any] = {
            "output_dir": output_dir,
            "logging_dir": logging_dir,
            "logging_strategy": "steps",
            "ddp_timeout": nccl_timeout_sec,
            "num_train_epochs": training.epochs,
            "per_device_train_batch_size": training.per_device_batch_size,
            "gradient_accumulation_steps": training.gradient_accumulation_steps,
            "learning_rate": training.learning_rate,
            "weight_decay": training.weight_decay,
            "lr_scheduler_type": training.lr_scheduler,
            "warmup_ratio": training.warmup_ratio,
            "bf16": training.bf16,
            "fp16": training.fp16,
            "gradient_checkpointing": training.gradient_checkpointing,
            "deepspeed": training.deepspeed,
            "logging_steps": training.logging_steps,
            "save_steps": training.save_steps,
            "log_completions": training.log_completions,
            "use_liger_kernel": training.use_liger_kernel,
            "report_to": training.report_to,
            **vllm_kwargs,
        }

        if training.gradient_checkpointing_kwargs:
            args["gradient_checkpointing_kwargs"] = (
                training.gradient_checkpointing_kwargs
            )
        else:
            args["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

        if training.max_grad_norm is not None:
            args["max_grad_norm"] = training.max_grad_norm
        if training.max_steps is not None and training.max_steps > 0:
            args["max_steps"] = training.max_steps
        if training.per_device_eval_batch_size is not None:
            args["per_device_eval_batch_size"] = training.per_device_eval_batch_size

        return args

    # ------------------------------------------------------------------
    # Trainer constructor kwargs — override in subclasses
    # ------------------------------------------------------------------

    def _build_trainer_kwargs(
        self,
        *,
        config: Any,
        model_name: str,
        trl_config: Any,
        train_dataset: Any,
        eval_dataset: Any,
        peft_config: Any,
        all_callbacks: list,
        reward_fn: Any,
        task_callbacks: list,
    ) -> dict[str, Any]:
        """Return kwargs for the TRL Trainer constructor.

        Base implementation provides the common set.  Subclasses override to
        add method-specific arguments (e.g. ``reward_funcs`` for GRPO).
        """
        return {
            "model": model_name,
            "args": trl_config,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "peft_config": peft_config,
            "callbacks": all_callbacks,
        }

    # ------------------------------------------------------------------
    # Training pipeline
    # ------------------------------------------------------------------

    def run(self, config: Any) -> None:
        """Execute training. Called inside the runtime environment (GPU process)."""
        from peft import LoraConfig

        from foresightrlcore.hooks import (
            import_dotted_path,
            resolve_callbacks,
            resolve_pre_post_hooks,
            resolve_trainer_class,
        )
        from foresightrlcore.tasks import get_task_components

        default_trainer_cls = import_dotted_path(self._default_trainer_path)
        default_config_cls = import_dotted_path(self._default_config_path)

        model_name = config.model.name
        task_type = config.task.type
        task_config = dict(config.task.config)
        task_config.setdefault("enable_thinking", config.model.enable_thinking)

        logger.info(
            f"{self.method_type.upper()} training — "
            f"model={model_name}, task={task_type}"
        )

        reward_fn, task_callbacks, dataset_loader = get_task_components(
            task_type=task_type,
            task_config=task_config,
            model_name=model_name,
        )

        train_path = str(config.data.train_path)
        train_dataset = dataset_loader(train_path)

        eval_dataset = None
        eval_path = config.data.eval_path
        if eval_path and os.path.isdir(str(eval_path)):
            eval_dataset = dataset_loader(str(eval_path))
        elif len(train_dataset) > config.data.eval_split_size:
            split = train_dataset.train_test_split(
                test_size=config.data.eval_split_size, seed=42
            )
            train_dataset = split["train"]
            eval_dataset = split["test"]

        logger.info(f"Train: {len(train_dataset)} examples")
        if eval_dataset:
            logger.info(f"Eval: {len(eval_dataset)} examples")

        lora = config.model.lora
        # Empty target_modules → omit the kwarg so PEFT picks its own
        # architecture-aware defaults. Required for Gemma 4, whose
        # `Gemma4ClippableLinear` wrapper trips PEFT's allowlist when
        # explicit Qwen/Llama-style names like "q_proj" are passed.
        target_modules = list(lora.target_modules) or None
        peft_kwargs: dict[str, Any] = dict(
            r=lora.r,
            lora_alpha=lora.alpha,
            lora_dropout=lora.dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        # exclude_modules/target_parameters are additive — omitted entirely
        # (not just passed as empty) when unset, so existing configs on
        # older PEFT versions that predate target_parameters support see no
        # behavior change. Populated from model.profile (see
        # foresightrlcore.config.model_profiles) for MoE models whose expert
        # FFN weights are fused nn.Parameter tensors that target_modules
        # matching cannot reach — same profile data the sft_verl handler
        # uses (foresightrlcore/methods/sft_verl.py), so a model profile
        # benefits both TRL/GRPO and verl/SFT paths.
        if lora.exclude_modules:
            # A regex STRING excludes an entire subtree (PEFT's own
            # re.fullmatch branch) — a plain list only matches an exact
            # trailing path segment, which can't express "anywhere under
            # this prefix" (e.g. a nested vision tower). Only wrap actual
            # lists; PEFT accepts a bare string natively.
            peft_kwargs["exclude_modules"] = (
                list(lora.exclude_modules)
                if isinstance(lora.exclude_modules, list)
                else lora.exclude_modules
            )
        if lora.target_parameters:
            peft_kwargs["target_parameters"] = list(lora.target_parameters)
        peft_config = LoraConfig(**peft_kwargs)

        trainer_args = self.build_trainer_args(config)
        trl_config = default_config_cls(
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=config.training.eval_steps,
            **trainer_args,
        )

        vllm_cfg = config.infra.vllm
        if config.infra.vllm_mode.value == "colocate" and vllm_cfg.enforce_eager:
            _patch_vllm_enforce_eager(enforce_eager=True)

        hooks = getattr(config.training, "hooks", None)

        trainer_cls = resolve_trainer_class(hooks, default_trainer_cls)
        hook_callbacks = resolve_callbacks(hooks)
        pre_train_fn, post_train_fn = resolve_pre_post_hooks(hooks)

        all_callbacks = list(task_callbacks) + hook_callbacks
        if _self_manages_checkpoint_s3(config):
            all_callbacks.append(
                _make_mixed_checkpoint_callback(str(config.output.checkpoint_s3_path))
            )

        trainer_kwargs = self._build_trainer_kwargs(
            config=config,
            model_name=model_name,
            trl_config=trl_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            all_callbacks=all_callbacks,
            reward_fn=reward_fn,
            task_callbacks=task_callbacks,
        )
        trainer = trainer_cls(**trainer_kwargs)

        if pre_train_fn is not None:
            logger.info("Running pre_train hook")
            pre_train_fn(trainer, config)

        resume = config.training.resume_from_checkpoint
        if resume is None:
            # Auto-resume: after a SageMaker Managed Spot interruption, SageMaker
            # restores checkpoint_s3_path -> /opt/ml/checkpoints (= output_dir)
            # and restarts the same job. Mixed InstanceGroups skip that
            # agent (two-way sync races the vLLM node), so the training
            # ranks pull the prefix themselves first. If a checkpoint is
            # present, resume from it; otherwise start fresh. Safe for
            # on-demand retries too.
            if _self_manages_checkpoint_s3(config):
                _restore_mixed_checkpoints(
                    str(config.output.checkpoint_s3_path),
                    trainer_args["output_dir"],
                )
            detected = _find_latest_checkpoint(trainer_args["output_dir"])
            if detected is not None:
                logger.info(f"Auto-resuming from detected checkpoint: {detected}")
                resume = detected

        if resume is not None:
            if isinstance(resume, str) and resume.startswith("s3://"):
                resume = _download_resume_checkpoint(resume)
            logger.info(f"Resuming from checkpoint: {resume}")
            trainer.train(resume_from_checkpoint=resume)
        else:
            trainer.train()

        if post_train_fn is not None:
            logger.info("Running post_train hook")
            post_train_fn(trainer, config)

        # Save final model to _final_model_dir if set (= /opt/ml/model on
        # SageMaker), so it's tarred to output.s3_path. Intermediate
        # checkpoints live in /opt/ml/checkpoints (SageMaker CheckpointConfig
        # on homogeneous jobs; training-rank s3 sync on mixed). Falls back
        # to trainer's output_dir for local runs.
        final_dir = getattr(config, "_final_model_dir", trainer_args["output_dir"])
        logger.info(f"Saving final model to {final_dir}")
        trainer.save_model(final_dir)
        logger.info(f"{self.method_type.upper()} training complete.")


def _find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Return the path to the highest-step ``checkpoint-N`` dir under output_dir.

    Used for automatic resume after a SageMaker Managed Spot interruption:
    SageMaker restores ``checkpoint_s3_path`` -> ``/opt/ml/checkpoints``
    (= ``output_dir``) before restarting the job, so a fresh process can pick up
    where it left off. Returns ``None`` when ``output_dir`` is missing or holds
    no valid ``checkpoint-N`` directories, so a first run still starts clean.
    """
    import re

    if not output_dir or not os.path.isdir(output_dir):
        return None

    pattern = re.compile(r"^checkpoint-(\d+)$")
    best_step = -1
    best_path: Optional[str] = None
    for name in os.listdir(output_dir):
        match = pattern.match(name)
        if match is None:
            continue
        candidate = os.path.join(output_dir, name)
        if not os.path.isdir(candidate):
            continue
        step = int(match.group(1))
        if step > best_step:
            best_step = step
            best_path = candidate
    return best_path


def _download_resume_checkpoint(s3_uri: str) -> str:
    """Download an S3 checkpoint once per node, with all ranks blocking until ready.

    On multi-rank training, every rank calls this. Only LOCAL_RANK==0 performs
    the `aws s3 sync`; other ranks poll a sentinel file. This avoids 8x
    concurrent `aws s3 sync` processes on the same node (which previously
    caused AWS CLI crashes / SIGSEGVs when downloading ~1 GiB checkpoints).
    Uses a filesystem sentinel instead of dist.barrier() so it works
    regardless of whether the process group is initialized and which backend
    (NCCL/Gloo) is active.
    """
    import shutil
    import subprocess
    import time

    local_ckpt = "/tmp/resume_checkpoint"
    sentinel = "/tmp/resume_checkpoint.done"
    failure = "/tmp/resume_checkpoint.failed"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if shutil.which("aws") is None:
        raise RuntimeError(
            "AWS CLI ('aws') not found on PATH. Install it (https://aws.amazon.com/cli/) "
            "or pass a local directory path to resume_from_checkpoint instead of an S3 URI."
        )

    if local_rank == 0:
        # Clean stale markers from prior attempts in the same container.
        for p in (sentinel, failure):
            if os.path.exists(p):
                os.remove(p)
        try:
            logger.info(f"[rank0] Downloading checkpoint {s3_uri} -> {local_ckpt}")
            subprocess.run(
                ["aws", "s3", "sync", s3_uri, local_ckpt],
                check=True,
            )
            with open(sentinel, "w") as f:
                f.write("ok")
            logger.info(f"[rank0] Checkpoint download complete: {local_ckpt}")
        except Exception as e:
            with open(failure, "w") as f:
                f.write(str(e))
            raise
    else:
        timeout_s = int(os.environ.get("CHECKPOINT_SYNC_TIMEOUT", "1800"))
        start = time.time()
        logger.info(
            f"[rank{local_rank}] Waiting for rank0 to finish checkpoint download "
            f"(timeout={timeout_s}s)"
        )
        while True:
            if os.path.exists(sentinel):
                logger.info(f"[rank{local_rank}] Checkpoint ready at {local_ckpt}")
                break
            if os.path.exists(failure):
                raise RuntimeError(
                    f"rank0 failed to download checkpoint {s3_uri} (see /tmp/resume_checkpoint.failed)"
                )
            if time.time() - start > timeout_s:
                raise TimeoutError(
                    f"Timed out after {timeout_s}s waiting for rank0 "
                    f"to sync {s3_uri}"
                )
            time.sleep(2)
    return local_ckpt


def _self_manages_checkpoint_s3(config: Any) -> bool:
    """True when training ranks must aws s3 sync checkpoints themselves.

    SageMaker ``CheckpointConfig`` two-way-syncs one prefix from every
    instance. Mixed InstanceGroups include a vLLM node whose empty
    ``/opt/ml/checkpoints`` races DeepSpeed shards, so the launcher omits
    that agent. Homogeneous jobs keep the SageMaker sync.
    """
    output = getattr(config, "output", None)
    infra = getattr(config, "infra", None)
    path = getattr(output, "checkpoint_s3_path", None) if output else None
    if not path or infra is None:
        return False
    uses_mixed = getattr(infra, "uses_mixed_instance_types", None)
    return bool(callable(uses_mixed) and uses_mixed())


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _aws_s3_sync(src: str, dest: str) -> None:
    """Run ``aws s3 sync src dest`` (no ``--delete``).

    ``--delete`` would drop another training node's DeepSpeed shards when
    more than one training instance uploads to the same prefix.
    """
    import shutil
    import subprocess

    if shutil.which("aws") is None:
        raise RuntimeError(
            "AWS CLI ('aws') not found on PATH; cannot sync checkpoints "
            f"between {src!r} and {dest!r}."
        )
    logger.info("aws s3 sync %s -> %s", src, dest)
    result = subprocess.run(
        ["aws", "s3", "sync", src, dest],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"aws s3 sync failed ({src} -> {dest}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def _restore_mixed_checkpoints(s3_uri: str, local_dir: str) -> None:
    """Pull ``checkpoint_s3_path`` into ``local_dir`` once per node.

    LOCAL_RANK 0 runs ``aws s3 sync``; other ranks wait on a sentinel
    under ``local_dir`` (shared disk per instance). Matches the
    ``_download_resume_checkpoint`` pattern so multi-GPU nodes do not
    stampede S3.
    """
    import time

    os.makedirs(local_dir, exist_ok=True)
    sentinel = os.path.join(local_dir, ".foresight_ckpt_restore.done")
    failure = os.path.join(local_dir, ".foresight_ckpt_restore.failed")
    rank = _local_rank()

    if rank == 0:
        for marker in (sentinel, failure):
            if os.path.exists(marker):
                os.remove(marker)
        try:
            _aws_s3_sync(s3_uri, local_dir)
            with open(sentinel, "w") as handle:
                handle.write("ok")
            logger.info("Restored mixed checkpoints from %s -> %s", s3_uri, local_dir)
        except Exception as exc:
            with open(failure, "w") as handle:
                handle.write(str(exc))
            raise
        return

    timeout_s = int(os.environ.get("CHECKPOINT_SYNC_TIMEOUT", "1800"))
    start = time.time()
    logger.info(
        "[rank%s] Waiting for rank0 mixed-checkpoint restore (timeout=%ss)",
        rank,
        timeout_s,
    )
    while True:
        if os.path.exists(sentinel):
            return
        if os.path.exists(failure):
            raise RuntimeError(
                f"rank0 failed to restore checkpoints from {s3_uri} ({failure})"
            )
        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out after {timeout_s}s waiting for rank0 to restore "
                f"{s3_uri} into {local_dir}"
            )
        time.sleep(2)


def _upload_mixed_checkpoints(local_dir: str, s3_uri: str) -> None:
    """Push ``local_dir`` to ``checkpoint_s3_path`` from LOCAL_RANK 0 only."""
    if _local_rank() != 0:
        return
    if not local_dir or not os.path.isdir(local_dir):
        return
    _aws_s3_sync(local_dir, s3_uri)
    logger.info("Uploaded mixed checkpoints %s -> %s", local_dir, s3_uri)


def _make_mixed_checkpoint_callback(s3_uri: str) -> Any:
    """HF callback: upload after each save and at train end (local rank 0)."""
    from transformers import TrainerCallback

    class MixedS3CheckpointCallback(TrainerCallback):
        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            _upload_mixed_checkpoints(args.output_dir, s3_uri)

        def on_train_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> None:
            _upload_mixed_checkpoints(args.output_dir, s3_uri)

    return MixedS3CheckpointCallback()


def _patch_vllm_enforce_eager(enforce_eager: bool) -> None:
    """Inject enforce_eager into TRL's vLLM LLM constructor.

    TRL 0.28 does not expose enforce_eager in GRPOConfig. Without it, vLLM
    spends 3-6 minutes compiling CUDA graphs during colocate init.
    """
    try:
        import trl.generation.vllm_generation as _vllm_gen
        from vllm import LLM as _OriginalLLM

        class _LLMEnforceEager(_OriginalLLM):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                kwargs.setdefault("enforce_eager", enforce_eager)
                super().__init__(*args, **kwargs)

        _vllm_gen.LLM = _LLMEnforceEager
        logger.info(f"Patched vLLM LLM to enforce_eager={enforce_eager}")
    except Exception as exc:
        logger.warning(f"Could not patch vLLM enforce_eager: {exc}")
