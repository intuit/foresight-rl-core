"""Benchmark types. Importing this package registers the built-in ones.

Record tier (the runner drives the loop):

``mcq``
    Multiple choice.
``generative``
    Free text, scored by a metric (exact match, or an LLM judge).

Harness tier (the benchmark drives its own loop):

``scenario_suite``
    Structured scenarios graded check by check, in-process.
``external_process``
    Any command line, talking to the endpoint over HTTP.
``plugin``
    An importable Python object.
``lm_eval``
    An ``lm-evaluation-harness`` task.

Adding a kind is one class plus ``@BENCHMARK_TYPES.register("name")``.
"""

from foresightrlcore.evaluate.benchmarks import (  # noqa: F401
    external_process,
    generative,
    lm_eval_adapter,
    mcq,
    plugin,
    scenario_suite,
)

__all__ = [
    "external_process",
    "generative",
    "lm_eval_adapter",
    "mcq",
    "plugin",
    "scenario_suite",
]
