"""foresightrlcore: Multi-method, multi-provider training library."""

__version__ = "0.1.0"


def train(config_path: str, wait: bool = False, **overrides) -> str:
    """Public Python API: load config, validate, and launch training.

    Args:
        config_path: Path to YAML config file.
        wait: Block until the job completes.
        **overrides: Dotted-key CLI overrides merged into config.

    Returns:
        Job identifier string.
    """
    from foresightrlcore.config.schema import TrainingJobConfig
    from foresightrlcore.operations.train import TrainOperation

    _ensure_backends_registered()
    _load_plugins()

    config = TrainingJobConfig.from_yaml(config_path)
    if overrides:
        config = config.merge_overrides(**overrides)

    op = TrainOperation()
    return op.run(config, wait=wait)


def _ensure_backends_registered() -> None:
    """Import backend modules so they auto-register via register_backend()."""
    import importlib

    for mod in [
        "foresightrlcore.execution.aws_training_job",
        "foresightrlcore.execution.aws_serverless",
        "foresightrlcore.execution.aws_hyperpod_eks",
        "foresightrlcore.execution.local",
        "foresightrlcore.methods.grpo",
        "foresightrlcore.operations.train",
        "foresightrlcore.operations.perf_test",
        "foresightrlcore.operations.optimizer",
        "foresightrlcore.operations.evaluate",
    ]:
        try:
            importlib.import_module(mod)
        except ImportError:
            pass


def _load_plugins() -> None:
    """Load optional extension plugins (LLM providers, auth, tasks)."""
    from foresightrlcore.plugins import load_plugins

    load_plugins()
