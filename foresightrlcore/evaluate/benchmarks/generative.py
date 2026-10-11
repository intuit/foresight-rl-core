"""Free-text benchmark type (record tier).

Ask an open question, score the answer with whatever metric the benchmark
configures -- ``exact_match`` for short factual answers, ``llm_judge`` for
answers where wording legitimately varies.

Parameters:

``instruction`` / ``system_prompt``
    Optional framing added around the question.
``template``
    Full override, formatted with ``{question}`` and ``{instruction}``.
``max_tokens``
    Per-benchmark output cap, overriding the model's default.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from foresightrlcore.evaluate.interfaces import (
    PARSE_FAILED_SUFFIX,
    EvalRecord,
    Metric,
    Prediction,
    PromptRequest,
    SampleScore,
)
from foresightrlcore.evaluate.metrics.text_match import ExactMatchMetric
from foresightrlcore.evaluate.registry import BENCHMARK_TYPES

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE = "{question}{instruction}"


class GenerativeBenchmark:
    """Sends an open question and scores the free-text answer."""

    def __init__(
        self,
        *,
        metrics: Optional[Sequence[Metric]] = None,
        instruction: str = "",
        system_prompt: Optional[str] = None,
        template: str = DEFAULT_TEMPLATE,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.metrics: List[Metric] = list(metrics or [ExactMatchMetric()])
        self.instruction = instruction
        self.system_prompt = system_prompt
        self.template = template
        self.max_tokens = max_tokens

    def build_prompt(self, record: EvalRecord) -> PromptRequest:
        instruction = f"\n\n{self.instruction}" if self.instruction else ""
        text = self.template.format(
            question=record.inputs.get("question", ""), instruction=instruction
        )
        request = PromptRequest.user(text, system=self.system_prompt)
        if self.max_tokens is not None:
            request.max_tokens = self.max_tokens
        return request

    def score(self, record: EvalRecord, prediction: Prediction) -> SampleScore:
        """Score with the first metric; extra metrics land in ``detail``.

        Each extra metric contributes its parse verdict alongside its value, so
        a run can report how many answers were readable at all and not only how
        many were right. See :meth:`McqBenchmark.score` for why that matters.
        """
        primary = self.metrics[0].score(record, prediction)
        for extra in self.metrics[1:]:
            verdict = extra.score(record, prediction)
            name = type(extra).__name__
            primary.detail[name] = verdict.value
            primary.detail[f"{name}{PARSE_FAILED_SUFFIX}"] = verdict.parse_failed
        return primary


@BENCHMARK_TYPES.register("generative")
def _make(
    spec: Any = None, metrics: Optional[Sequence[Metric]] = None, **_: Any
) -> GenerativeBenchmark:
    params: Dict[str, Any] = spec.type_params() if spec is not None else {}
    known = {"instruction", "system_prompt", "template", "max_tokens"}
    return GenerativeBenchmark(
        metrics=metrics, **{k: v for k, v in params.items() if k in known}
    )


__all__ = ["DEFAULT_TEMPLATE", "GenerativeBenchmark"]
