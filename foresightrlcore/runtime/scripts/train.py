"""Unified training driver for SageMaker and local execution.

Reads configuration from a YAML file (--config), parses it into a
:class:`TrainingJobConfig`, resolves the method handler via the registry,
and runs training.

Usage (called by entrypoint.sh, not directly):
    python train.py --config /opt/ml/input/data/config/args.yaml
    python -m foresightrlcore.runtime.scripts.train --config args.yaml
"""

import argparse
import logging
import os
import sys

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")

SM_TRAIN_DIR = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
SM_EVAL_DIR = os.environ.get("SM_CHANNEL_EVAL", "/opt/ml/input/data/eval")
SM_CONFIG_DIR = os.environ.get("SM_CHANNEL_CONFIG", "/opt/ml/input/data/config")
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
SM_CHECKPOINT_DIR = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")
SM_INPUT_MODEL_DIR = os.environ.get("SM_CHANNEL_MODEL", "/opt/ml/input/data/model")

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_deepspeed_config(ds_value: str | None) -> str | None:
    """Return an absolute path to the DeepSpeed config JSON.

    On SageMaker the file lives in the container's working directory.
    For local runs the CWD is wherever ``rl-train`` was invoked, so a
    bare filename like ``ds_zero2_config.json`` won't be found.  Fall
    back to the copy shipped alongside this script.
    """
    if ds_value is None:
        return None
    if os.path.isabs(ds_value) or os.path.exists(ds_value):
        return ds_value
    packaged = os.path.join(_SCRIPTS_DIR, ds_value)
    if os.path.exists(packaged):
        logger.info(f"Resolved DeepSpeed config to packaged path: {packaged}")
        return packaged
    return ds_value


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Training Driver")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--vllm-host",
        type=str,
        default=None,
        help="Override vLLM server host (for multi-node topology)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config, searching in common locations if path is relative."""
    candidates = [
        config_path,
        os.path.join(SM_CONFIG_DIR, os.path.basename(config_path)),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"Config file not found. Tried: {candidates}")


def _resolve_method_type(config: dict) -> str:
    """Extract method type from config, defaulting to 'grpo' for backward compat."""
    return config.get("method", {}).get("type", "grpo")


# ------------------------------------------------------------------
# SageMaker environment helpers
# ------------------------------------------------------------------


def _apply_sagemaker_overrides(config: "TrainingJobConfig", args) -> "TrainingJobConfig":  # type: ignore[name-defined]
    """Patch a parsed config with SageMaker-specific paths and overrides.

    Mutates the config in-place and returns it for convenience.
    """
    config.data.train_path = SM_TRAIN_DIR

    if os.path.isdir(SM_EVAL_DIR) and os.listdir(SM_EVAL_DIR):
        config.data.eval_path = SM_EVAL_DIR

    # Pre-staged model: when YAML model.name is an s3:// URI the launcher
    # mounts it as a SageMaker 'model' channel (see launcher). At runtime
    # we swap the URI for the local mount so transformers loads from disk
    # and never touches HF Hub — this is the fix for the multi-rank Hub
    # cache race that loses random shards under from_pretrained().
    if os.path.isfile(os.path.join(SM_INPUT_MODEL_DIR, "config.json")):
        logger.info(
            f"Pre-staged model detected at {SM_INPUT_MODEL_DIR} — "
            f"overriding model.name (was: {config.model.name!r})"
        )
        config.model.name = SM_INPUT_MODEL_DIR

    os.makedirs(SM_MODEL_DIR, exist_ok=True)
    os.makedirs(SM_CHECKPOINT_DIR, exist_ok=True)
    # TRL writes intermediate `checkpoint-N/` dirs under output_dir during
    # training. Point that at /opt/ml/checkpoints so SageMaker streams them
    # to checkpoint_s3_path continuously instead of only at job end. Final
    # model still goes to /opt/ml/model via trainer.save_model() at the end.
    config._local_output_dir = SM_CHECKPOINT_DIR  # type: ignore[attr-defined]
    config._final_model_dir = SM_MODEL_DIR  # type: ignore[attr-defined]

    ds_resolved = _resolve_deepspeed_config(config.training.deepspeed)
    if ds_resolved is not None:
        config.training.deepspeed = ds_resolved

    if args.vllm_host:
        config.infra.vllm.host = args.vllm_host

    return config


def _maybe_nccl_warmup(raw_config: dict) -> None:
    """Run NCCL warmup for multi-node colocate setups."""
    infra_cfg = raw_config.get("infra", {})
    execution_type = raw_config.get("execution", {}).get("type", "")
    instance_count = infra_cfg.get("instance_count", 1)
    is_multi_node = instance_count > 1 and execution_type != "local"

    if not is_multi_node:
        return

    nccl_timeout_raw = int(os.environ.get("NCCL_TIMEOUT", "1800000"))
    nccl_timeout_sec = (
        nccl_timeout_raw if nccl_timeout_raw < 10000 else nccl_timeout_raw // 1000
    )

    from datetime import timedelta

    import torch
    import torch.distributed as dist
    import torch.distributed.distributed_c10d as _c10d

    _c10d.default_pg_timeout = timedelta(seconds=nccl_timeout_sec)
    logger.info(
        f"Set default process group timeout to {nccl_timeout_sec}s "
        f"(raw NCCL_TIMEOUT={nccl_timeout_raw})"
    )

    vllm_mode = infra_cfg.get("vllm_mode", "server")
    if vllm_mode == "colocate" and dist.is_initialized() and dist.get_world_size() > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        logger.info(
            f"Pre-warming NCCL connections "
            f"(world_size={dist.get_world_size()}, "
            f"rank={dist.get_rank()})..."
        )
        dist.barrier()
        dummy = torch.zeros(1, device=f"cuda:{local_rank}")
        dist.all_reduce(dummy)
        del dummy
        dist.barrier()
        logger.info("NCCL warmup complete — all cross-node connections established")


def _run_with_handler(raw_config: dict, args) -> None:
    """Parse config, resolve handler, and run training inside @record."""
    from torch.distributed.elastic.multiprocessing.errors import record

    @record
    def _inner():
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

        method_type = _resolve_method_type(raw_config)
        logger.info(f"Method: {method_type}")

        model_name = raw_config.get("model", {}).get("name", "unknown")
        task_type = raw_config.get("task", {}).get("type", "unknown")
        training_cfg = raw_config.get("training", {})
        logger.info(f"Model: {model_name}")
        logger.info(f"Task: {task_type}")
        logger.info(
            f"Training config: epochs={training_cfg.get('epochs', 1)}, "
            f"lr={training_cfg.get('learning_rate', 5e-5)}, "
            f"batch_size={training_cfg.get('per_device_batch_size', 4)}"
        )

        _maybe_nccl_warmup(raw_config)

        import foresightrlcore.methods.grpo  # noqa: F401 — ensure GRPO is registered
        import foresightrlcore.methods.sft_verl  # noqa: F401 — ensure sft_verl is registered
        from foresightrlcore.config.schema import TrainingJobConfig
        from foresightrlcore.methods import get_method

        handler_cls = get_method(method_type)
        handler = handler_cls()

        parsed_config = TrainingJobConfig.model_validate(raw_config)
        parsed_config = _apply_sagemaker_overrides(parsed_config, args)

        handler.run(parsed_config)

    _inner()


# ------------------------------------------------------------------
# Secrets resolution (unchanged)
# ------------------------------------------------------------------


def _resolve_secrets_if_configured(config_path: str) -> None:
    """Resolve secrets for torchrun mode (entrypoint.sh is not the dispatcher).

    Delegates to the co-located resolve_secrets.py, which writes a sourceable
    env file. IDPS installs and the internal package index live in the
    foresightrl-intuit extension, not in this module.
    """
    import subprocess

    env_file = "/tmp/resolved_secrets.env"

    if os.path.exists(env_file):
        logger.info("Secrets env file already exists at %s — loading.", env_file)
        _load_env_file(env_file)
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    resolve_script = os.path.join(script_dir, "resolve_secrets.py")

    logger.info("Resolving secrets via %s ...", resolve_script)
    subprocess.check_call(
        [
            sys.executable,
            resolve_script,
            "--config",
            config_path,
            "--env-file",
            env_file,
        ],
    )

    _load_env_file(env_file)


def _load_env_file(env_file: str) -> None:
    """Parse a sourceable env file and set vars in the current process."""
    import shlex

    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            key, _, value = line.partition("=")
            if key and value:
                os.environ[key] = shlex.split(value)[0]
                logger.info(
                    "Loaded secret -> %s (length: %d)", key, len(os.environ[key])
                )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main():
    args = parse_args()
    config = load_config(args.config)

    if config.get("secrets"):
        provider = config["secrets"].get("provider", "idps")
        if provider == "local":
            from foresightrlcore.auth import resolve_secrets

            resolve_secrets(args.config)
        else:
            _resolve_secrets_if_configured(args.config)

    _run_with_handler(config, args)


if __name__ == "__main__":
    main()
