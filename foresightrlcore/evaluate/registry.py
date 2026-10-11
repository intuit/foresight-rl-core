"""The four pluggable axes of the evaluate operation.

Reuses :class:`foresightrlcore.common.registry.Registry`, so the API is the one
already used elsewhere in the package::

    from foresightrlcore.evaluate.registry import BENCHMARK_TYPES

    @BENCHMARK_TYPES.register("my_benchmark_kind")
    def _make(spec, params):
        return MyBenchmark(**params)

Adding a benchmark *kind* is one class plus one decorator. Adding a
*benchmark* needs neither -- it is a catalog YAML entry.
"""

from __future__ import annotations

from typing import Callable

from foresightrlcore.common.registry import Registry

#: name -> factory returning a RecordBenchmark or HarnessBenchmark.
BENCHMARK_TYPES: "Registry[Callable[..., object]]" = Registry("benchmark type")

#: name -> factory returning a DatasetLoader.
DATASET_LOADERS: "Registry[Callable[..., object]]" = Registry("dataset loader")

#: name -> factory returning a Metric.
METRICS: "Registry[Callable[..., object]]" = Registry("metric")

#: name -> factory returning a Reporter.
REPORTERS: "Registry[Callable[..., object]]" = Registry("reporter")


__all__ = ["BENCHMARK_TYPES", "DATASET_LOADERS", "METRICS", "REPORTERS"]
