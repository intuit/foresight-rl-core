"""Evaluate operation: run benchmarks against models, produce a report.

A **substrate**, not a benchmark collection. The core owns three things --
inference endpoints, execution (workspaces, processes, retries, trials), and
reporting -- and nothing else. Real benchmarks are catalog YAML; benchmark
*kinds* are one class plus a decorator. See
``docs/evaluate-extension-contract.md`` for the stable surface.

Submodules
----------

* :mod:`foresightrlcore.evaluate.interfaces` -- the two benchmark tiers.
* :mod:`foresightrlcore.evaluate.registry` -- the four pluggable axes.
* :mod:`foresightrlcore.evaluate.catalog` -- benchmarks as data.
* :mod:`foresightrlcore.evaluate.runner` -- the cross-product orchestrator.
* :mod:`foresightrlcore.evaluate.workspace` -- source staging + trial
  workspaces.
* :mod:`foresightrlcore.evaluate.benchmarks` -- built-in benchmark types.
* :mod:`foresightrlcore.evaluate.datasets` -- built-in dataset loaders.
* :mod:`foresightrlcore.evaluate.metrics` -- built-in metrics + trial
  statistics.
* :mod:`foresightrlcore.evaluate.reporters` -- built-in output sinks.
* :mod:`foresightrlcore.evaluate.cli` -- ``rl-train evaluate`` /
  ``validate-evaluate`` / ``list-benchmarks``.
"""

from foresightrlcore.evaluate import spec


def launch(config_path: str, wait: bool = False, **overrides) -> list[str]:
    """Public Python API: load an evaluate config, validate, and run/submit.

    Args:
        config_path: Path to a YAML config in the ``EvaluateJobConfig`` schema.
        wait: Block until the submitted job reaches a terminal state (AWS only).
        **overrides: Dotted-key CLI overrides merged into the config.

    Returns:
        Single-element list containing the backend-specific job identifier.
    """
    from foresightrlcore import _ensure_backends_registered
    from foresightrlcore.config.evaluate_schema import EvaluateJobConfig
    from foresightrlcore.operations.evaluate import EvaluateOperation

    _ensure_backends_registered()

    config = EvaluateJobConfig.from_yaml(config_path)
    if overrides:
        config = config.merge_overrides(**overrides)

    return EvaluateOperation().run(config, wait=wait)


__all__ = ["spec", "launch"]
