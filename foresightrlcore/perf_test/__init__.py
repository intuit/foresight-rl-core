"""Performance-testing operation for foresight-rl-core.

Pairs a ``vllm serve`` process with a Gatling load-test driver and runs
a TPS sweep against the served model, producing a markdown summary of
throughput and percentile latencies per TPS level.

Submodules
----------

* :mod:`foresightrlcore.perf_test.spec` -- result schemas (per-TPS stats,
  run summary).
* :mod:`foresightrlcore.perf_test.runner` -- container-side orchestrator
  invoked by ``perf_entrypoint.sh``. Starts vLLM, copies user-supplied
  test data into the vendored Gatling resources, loops over TPS levels,
  and writes results.
* :mod:`foresightrlcore.perf_test.cli` -- ``rl-train perf-test`` and
  ``rl-train validate-perf`` subcommand dispatch.
* ``gatling/`` -- vendored Gatling Java project (built and executed at
  runtime by ``runner.py``).
"""

from foresightrlcore.perf_test import spec


def launch(config_path: str, wait: bool = False, **overrides) -> list[str]:
    """Public Python API: load a perf-test config, validate, and submit.

    Args:
        config_path: Path to YAML config file (``PerfTestJobConfig`` schema).
        wait: Block until the submitted job reaches a terminal state.
        **overrides: Dotted-key CLI overrides merged into the config.

    Returns:
        Single-element list containing the backend-specific job identifier.
    """
    from foresightrlcore import _ensure_backends_registered
    from foresightrlcore.config.perf_schema import PerfTestJobConfig
    from foresightrlcore.operations.perf_test import PerfTestOperation

    _ensure_backends_registered()

    config = PerfTestJobConfig.from_yaml(config_path)
    if overrides:
        config = config.merge_overrides(**overrides)

    op = PerfTestOperation()
    return op.run(config, wait=wait)


__all__ = ["spec", "launch"]
