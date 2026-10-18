"""Delegate a benchmark to an importable Python object (harness tier).

The in-process counterpart to ``external_process``: when a benchmark is already
Python and installed alongside the job, running it in a subprocess buys nothing
and costs a process launch plus JSON round-trip.

    benchmarks:
      - name: my_suite
        type: plugin
        params:
          target: mypkg.benchmarks:MySuite
          init: {strict: true}

``target`` is a dotted path with an optional ``:attr`` suffix. It may be:

* a class implementing ``run(ctx) -> HarnessOutcome`` -- instantiated with
  ``init``;
* a factory returning such an object;
* a plain callable taking ``ctx``.

A ``dict`` or a bare number returned instead of a ``HarnessOutcome`` is
coerced, so a small benchmark can just ``return {"score": 0.9}``.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Dict, Optional

from foresightrlcore.evaluate.interfaces import HarnessContext, HarnessOutcome
from foresightrlcore.evaluate.registry import BENCHMARK_TYPES

logger = logging.getLogger(__name__)


def import_target(target: str) -> Any:
    """Import ``pkg.module:attr`` or ``pkg.module.attr``.

    Raises:
        ValueError: If the path is empty or the attribute does not exist.
        ImportError: If the module cannot be imported.
    """
    if not target:
        raise ValueError("plugin benchmark requires a 'target' dotted path")

    if ":" in target:
        module_path, _, attr = target.partition(":")
    else:
        module_path, _, attr = target.rpartition(".")
        if not module_path:
            raise ValueError(
                f"plugin target {target!r} must be 'module:attr' or 'module.attr'"
            )

    module = importlib.import_module(module_path)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(
            f"plugin target {target!r}: module {module_path!r} has no attribute "
            f"{attr!r}"
        ) from exc


def coerce_outcome(value: Any) -> HarnessOutcome:
    """Coerce a benchmark's return value into a :class:`HarnessOutcome`."""
    if isinstance(value, HarnessOutcome):
        return value
    if isinstance(value, (int, float)):
        return HarnessOutcome(score=float(value), n_samples=1)
    if isinstance(value, dict):
        known = {f for f in HarnessOutcome.__dataclass_fields__}
        return HarnessOutcome(**{k: v for k, v in value.items() if k in known})
    raise TypeError(
        f"plugin benchmark returned {type(value).__name__}; expected a "
        f"HarnessOutcome, dict, or number"
    )


class PluginBenchmark:
    """Resolves ``target`` on first use and delegates ``run`` to it.

    Resolution is deferred so an unimportable target surfaces as a failed
    benchmark with a readable message, not an import error during registry
    setup that takes the whole run down.
    """

    def __init__(self, target: str, *, init: Optional[Dict[str, Any]] = None) -> None:
        self.target = target
        self.init = dict(init or {})
        self._runner: Optional[Callable[[HarnessContext], Any]] = None

    def _resolve(self) -> Callable[[HarnessContext], Any]:
        if self._runner is not None:
            return self._runner

        obj = import_target(self.target)
        if isinstance(obj, type):
            instance = obj(**self.init)
            runner = getattr(instance, "run", None)
            if not callable(runner):
                raise TypeError(
                    f"plugin target {self.target!r} has no callable run(ctx)"
                )
            self._runner = runner
        elif callable(obj):
            candidate = obj(**self.init) if self.init else obj
            runner = getattr(candidate, "run", candidate)
            if not callable(runner):
                raise TypeError(
                    f"plugin target {self.target!r} did not resolve to a callable"
                )
            self._runner = runner
        else:
            raise TypeError(f"plugin target {self.target!r} is not callable")

        return self._runner

    def run(self, ctx: HarnessContext) -> HarnessOutcome:
        try:
            runner = self._resolve()
        except Exception as exc:  # noqa: BLE001 - reported as a benchmark failure
            logger.warning("plugin benchmark %r failed to load: %s", self.target, exc)
            return HarnessOutcome(error=f"could not load plugin target: {exc}")

        try:
            return coerce_outcome(runner(ctx))
        except Exception as exc:  # noqa: BLE001 - reported as a benchmark failure
            logger.warning("plugin benchmark %r raised: %s", self.target, exc)
            return HarnessOutcome(error=f"{type(exc).__name__}: {exc}")


@BENCHMARK_TYPES.register("plugin")
def _make(spec: Any = None, **_: Any) -> PluginBenchmark:
    params: Dict[str, Any] = spec.type_params() if spec is not None else {}
    return PluginBenchmark(params.get("target", ""), init=params.get("init"))


__all__ = ["PluginBenchmark", "coerce_outcome", "import_target"]
