"""LLM-as-judge metric for free-text answers.

Reuses :func:`~foresightrlcore.clients.llm.resolve_llm_manager`, so the judge
model is configured exactly like any other client in the package and can be a
different provider from the model under test.

The judge is asked for a 1-5 score, matching the scale the optimizer's judges
already use, and that score is rescaled to ``[0, 1]`` so it aggregates with the
other metrics. A judge call that fails scores 0 and is recorded as a parse
failure rather than a wrong answer -- an unavailable judge is a harness
problem, not a model result.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from foresightrlcore.common.retry import call_with_retry
from foresightrlcore.evaluate.interfaces import EvalRecord, Prediction, SampleScore
from foresightrlcore.evaluate.registry import METRICS

logger = logging.getLogger(__name__)

DEFAULT_RUBRIC = (
    "Rate how well the candidate answer matches the reference answer for the "
    "question, on a 1-5 scale:\n"
    "5 = fully equivalent; 4 = correct with minor omissions; "
    "3 = partially correct; 2 = mostly incorrect; 1 = incorrect or irrelevant.\n"
    "Reply with the digit only."
)

_SCORE_PATTERN = re.compile(r"\b([1-5])\b")


class JudgeMetric:
    """Scores a free-text answer with an LLM judge.

    Args:
        manager: A resolved LLM client manager. Passing it in (rather than
            resolving here) keeps the metric constructible offline.
        rubric: Scoring instructions sent with every request.
        max_retries: Attempts per judge call.
        base_delay: Base seconds for exponential backoff.
    """

    def __init__(
        self,
        manager: Any = None,
        *,
        rubric: str = DEFAULT_RUBRIC,
        max_retries: int = 6,
        base_delay: float = 2.0,
    ) -> None:
        self._manager = manager
        self.rubric = rubric
        self._max_retries = max_retries
        self._base_delay = base_delay

    def _prompt(self, record: EvalRecord, prediction: Prediction) -> str:
        return (
            f"{self.rubric}\n\n"
            f"Question:\n{record.inputs.get('question', '')}\n\n"
            f"Reference answer:\n{record.gold}\n\n"
            f"Candidate answer:\n{prediction.text}\n"
        )

    def score(self, record: EvalRecord, prediction: Prediction) -> SampleScore:
        if prediction.failed:
            return SampleScore(detail={"error": prediction.error})
        if self._manager is None:
            return SampleScore(
                parse_failed=True,
                detail={"reason": "judge metric has no llm client configured"},
            )

        prompt = self._prompt(record, prediction)

        def _once() -> str:
            client = self._manager.get()
            message = client.bind(max_tokens=16).invoke(
                [{"role": "user", "content": prompt}]
            )
            content = getattr(message, "content", message)
            return content if isinstance(content, str) else str(content)

        try:
            raw = call_with_retry(
                _once,
                manager=self._manager,
                max_retries=self._max_retries,
                base_delay=self._base_delay,
                description="judge",
            )
        except Exception as exc:  # noqa: BLE001 - judge failure is a harness fault
            logger.warning("judge call failed for record %s: %s", record.id, exc)
            return SampleScore(
                parse_failed=True, detail={"reason": f"judge call failed: {exc}"}
            )

        rating = _parse_rating(raw)
        if rating is None:
            return SampleScore(
                parse_failed=True,
                detail={"reason": "judge did not return a 1-5 score", "raw": raw},
            )

        # 1-5 -> 0.0-1.0 so this aggregates with the binary metrics.
        value = (rating - 1) / 4.0
        return SampleScore(
            value=value,
            correct=rating >= 4,
            detail={"judge_rating": rating},
        )


def _parse_rating(raw: str) -> Optional[int]:
    match = _SCORE_PATTERN.search(raw or "")
    return int(match.group(1)) if match else None


@METRICS.register("llm_judge")
def _make(manager: Any = None, **params: Any) -> JudgeMetric:
    return JudgeMetric(manager, **params)


__all__ = ["DEFAULT_RUBRIC", "JudgeMetric"]
