"""Run an ``lm-evaluation-harness`` task (harness tier).

The escape hatch: any task `lm-eval` already knows about runs here unchanged.
Because the core hands every benchmark an OpenAI-compatible endpoint, the
adapter uses lm-eval's ``local-chat-completions`` model type and points it at
that endpoint -- so a Bedrock or Genos candidate is evaluated by lm-eval
without lm-eval knowing either provider exists.

    benchmarks:
      - name: gsm8k
        type: lm_eval
        params:
          tasks: [gsm8k]
          num_fewshot: 5
          primary_metric: exact_match

``lm_eval`` is imported inside :meth:`run`, so it is never a package
dependency. Without it installed, the benchmark fails with an actionable
message and the rest of the run continues.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from foresightrlcore.evaluate.interfaces import HarnessContext, HarnessOutcome
from foresightrlcore.evaluate.registry import BENCHMARK_TYPES

logger = logging.getLogger(__name__)

#: Metric keys lm-eval commonly reports, most specific first. Used to pick the
#: primary score when the config does not name one.
METRIC_PREFERENCE = (
    "exact_match,none",
    "acc_norm,none",
    "acc,none",
    "exact_match",
    "acc_norm",
    "acc",
    "f1",
)


class LmEvalBenchmark:
    """Delegates to ``lm_eval.simple_evaluate`` against the run's endpoint.

    Args:
        tasks: lm-eval task names to run.
        num_fewshot: Few-shot examples per task.
        limit: Cap on documents per task (smoke runs).
        batch_size: Passed through to lm-eval.
        primary_metric: Which reported metric becomes ``score``. Defaults to
            the first of :data:`METRIC_PREFERENCE` present in the results.
        model_args: Extra kwargs merged into the model spec; ``base_url``,
            ``model``, and ``num_concurrent`` are supplied from the context.
        extra: Extra kwargs forwarded to ``simple_evaluate``.
    """

    def __init__(
        self,
        tasks: List[str],
        *,
        num_fewshot: Optional[int] = None,
        limit: Optional[int] = None,
        batch_size: Any = 1,
        primary_metric: Optional[str] = None,
        model_args: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not tasks:
            raise ValueError("lm_eval benchmark requires a non-empty 'tasks' list")
        self.tasks = list(tasks)
        self.num_fewshot = num_fewshot
        self.limit = limit
        self.batch_size = batch_size
        self.primary_metric = primary_metric
        self.model_args = dict(model_args or {})
        self.extra = dict(extra or {})

    def _model_args(self, ctx: HarnessContext) -> Dict[str, Any]:
        return {
            "base_url": f"{ctx.endpoint.base_url}/chat/completions",
            "model": ctx.endpoint.model_id,
            "num_concurrent": 1,
            "tokenized_requests": False,
            **self.model_args,
        }

    def run(self, ctx: HarnessContext) -> HarnessOutcome:
        try:
            import lm_eval  # type: ignore[import-not-found]
        except ImportError:
            return HarnessOutcome(
                error="benchmark type 'lm_eval' needs the lm-eval package; add "
                "'lm-eval' to the container requirements or use a recipe that "
                "ships it"
            )

        kwargs: Dict[str, Any] = {
            "model": "local-chat-completions",
            "model_args": self._model_args(ctx),
            "tasks": self.tasks,
            "batch_size": self.batch_size,
            **self.extra,
        }
        if self.num_fewshot is not None:
            kwargs["num_fewshot"] = self.num_fewshot
        if self.limit is not None:
            kwargs["limit"] = self.limit

        ctx.logger.info("lm_eval: running tasks %s", ", ".join(self.tasks))
        try:
            raw = lm_eval.simple_evaluate(**kwargs)
        except Exception as exc:  # noqa: BLE001 - reported as a benchmark failure
            logger.warning("lm_eval failed for %s: %s", ctx.benchmark, exc)
            return HarnessOutcome(error=f"lm_eval failed: {type(exc).__name__}: {exc}")

        return self.to_outcome(raw)

    def to_outcome(self, raw: Any) -> HarnessOutcome:
        """Normalise lm-eval's ``results`` mapping into a HarnessOutcome.

        Each task's chosen metric becomes a sub-score; the mean across tasks is
        the headline score, which for a single-task benchmark is just that
        task's metric.
        """
        results = (raw or {}).get("results") or {}
        if not results:
            return HarnessOutcome(error="lm_eval returned no results")

        sub_scores: Dict[str, float] = {}
        for task, metrics in results.items():
            value = _pick_metric(metrics, self.primary_metric)
            if value is not None:
                sub_scores[task] = value

        if not sub_scores:
            return HarnessOutcome(
                error="lm_eval results carried no numeric metric "
                f"(looked for {self.primary_metric or 'a known metric key'})",
                detail={"results": results},
            )

        n_samples = sum(
            int(count)
            for count in ((raw or {}).get("n-samples") or {}).values()
            if isinstance(count, int)
        )
        return HarnessOutcome(
            score=sum(sub_scores.values()) / len(sub_scores),
            n_samples=n_samples,
            sub_scores=sub_scores,
            detail={"results": results},
        )


def _pick_metric(metrics: Any, preferred: Optional[str]) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    keys: tuple = (preferred,) + METRIC_PREFERENCE if preferred else METRIC_PREFERENCE
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


@BENCHMARK_TYPES.register("lm_eval")
def _make(spec: Any = None, **_: Any) -> LmEvalBenchmark:
    params: Dict[str, Any] = spec.type_params() if spec is not None else {}
    tasks = params.get("tasks") or ([spec.name] if spec is not None else [])
    known = {
        "num_fewshot",
        "limit",
        "batch_size",
        "primary_metric",
        "model_args",
        "extra",
    }
    return LmEvalBenchmark(
        list(tasks), **{k: v for k, v in params.items() if k in known}
    )


__all__ = ["METRIC_PREFERENCE", "LmEvalBenchmark"]
