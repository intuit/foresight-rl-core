"""GenAI Optimizer operation for foresight-rl-core.

Batch-evaluates candidate LLMs on quality (deterministic + LLM-as-judge) and
cost (token pricing) over a sampled dataset, producing a results table and a
markdown report. Generic and pluggable (datasets / prompt sets / generation
modes / judges), with the BTS use case shipped fully configured.

Submodules
----------

* :mod:`foresightrlcore.optimizer.interfaces` -- dataclasses + protocols.
* :mod:`foresightrlcore.optimizer.registry` -- pluggable-axis registries.
* :mod:`foresightrlcore.optimizer.runner` -- container-side / in-process runner.
* :mod:`foresightrlcore.optimizer.generation` -- genos / bedrock / local vLLM.
* :mod:`foresightrlcore.optimizer.judges` -- deterministic + LLM judges.
* :mod:`foresightrlcore.optimizer.modes` -- generation modes.
* :mod:`foresightrlcore.optimizer.datasets` -- dataset providers.
* :mod:`foresightrlcore.optimizer.cli` -- ``rl-train optimizer`` /
  ``validate-optimizer`` subcommand dispatch.
"""

from foresightrlcore.optimizer import spec


def launch(config_path: str, wait: bool = False, **overrides) -> list[str]:
    """Public Python API: load an optimizer config, validate, and run/submit.

    Args:
        config_path: Path to YAML config file (``OptimizerJobConfig`` schema).
        wait: Block until the submitted job reaches a terminal state (AWS only).
        **overrides: Dotted-key CLI overrides merged into the config.

    Returns:
        Single-element list containing the backend-specific job identifier.
    """
    from foresightrlcore import _ensure_backends_registered
    from foresightrlcore.config.optimizer_schema import OptimizerJobConfig
    from foresightrlcore.operations.optimizer import OptimizerOperation

    _ensure_backends_registered()

    config = OptimizerJobConfig.from_yaml(config_path)
    if overrides:
        config = config.merge_overrides(**overrides)

    op = OptimizerOperation()
    return op.run(config, wait=wait)


__all__ = ["spec", "launch"]
