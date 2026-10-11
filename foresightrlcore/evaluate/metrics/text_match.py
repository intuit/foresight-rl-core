"""String-comparison metrics.

Normalisation is deliberately conservative: case, surrounding whitespace,
trailing punctuation, and a leading article. Anything more aggressive starts
marking wrong answers correct, which is a worse failure than under-crediting.
"""

from __future__ import annotations

import re
import string
from typing import Any

from foresightrlcore.evaluate.interfaces import EvalRecord, Prediction, SampleScore
from foresightrlcore.evaluate.registry import METRICS

_ARTICLES = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def normalize(text: Any) -> str:
    """Lowercase, collapse whitespace, drop articles and edge punctuation."""
    value = _WHITESPACE.sub(" ", str(text or "")).strip().lower()
    value = value.strip(string.punctuation + " ")
    return _ARTICLES.sub("", value).strip()


class ExactMatchMetric:
    """Correct when the prediction equals the gold answer after normalisation."""

    def score(self, record: EvalRecord, prediction: Prediction) -> SampleScore:
        if prediction.failed:
            return SampleScore(detail={"error": prediction.error})
        got, want = normalize(prediction.text), normalize(record.gold)
        correct = bool(want) and got == want
        return SampleScore(
            value=1.0 if correct else 0.0,
            correct=correct,
            detail={"expected": want, "got": got},
        )


class SubstringMatchMetric:
    """Correct when the gold answer appears anywhere in the prediction.

    Right for free-text answers where the model explains itself; wrong for
    multiple choice, where an option can be quoted in passing while a different
    one is chosen.
    """

    def score(self, record: EvalRecord, prediction: Prediction) -> SampleScore:
        if prediction.failed:
            return SampleScore(detail={"error": prediction.error})
        got, want = normalize(prediction.text), normalize(record.gold)
        correct = bool(want) and want in got
        return SampleScore(
            value=1.0 if correct else 0.0,
            correct=correct,
            detail={"expected": want},
        )


@METRICS.register("exact_match")
def _make_exact() -> ExactMatchMetric:
    return ExactMatchMetric()


@METRICS.register("substring_match")
def _make_substring() -> SubstringMatchMetric:
    return SubstringMatchMetric()


__all__ = ["ExactMatchMetric", "SubstringMatchMetric", "normalize"]
