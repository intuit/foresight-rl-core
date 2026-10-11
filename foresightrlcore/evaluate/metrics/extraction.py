"""Metric for extraction tasks, where "nothing to extract" is a real answer.

``exact_match`` assumes every record has an answer worth giving. Extraction
tasks do not: the model is shown candidates and asked which, if any, apply, so
an empty answer is sometimes the only correct one. That makes each row exactly
one of TP / FP / FN / TN rather than simply right or wrong, which is what lets
the ``confusion`` reporter derive precision and recall.

Comparison is on the whole normalised answer, not per extracted item. Half of
a two-item answer is a false positive, not half credit — for a task that feeds
an automated action, a partially-right answer is as unusable as a wrong one.
"""

from __future__ import annotations

from typing import Any, Tuple

from foresightrlcore.evaluate.interfaces import EvalRecord, Prediction, SampleScore
from foresightrlcore.evaluate.registry import METRICS

#: The four outcomes, in the order reports conventionally list them.
OUTCOMES: Tuple[str, ...] = ("TP", "FP", "FN", "TN")


def as_answer(value: Any) -> str:
    """Render a gold or predicted answer as a comparable string.

    Missing values arrive spelled several ways depending on the reader — ``None``
    from JSON null, the string ``"nan"`` from a pandas float column that had a
    blank — and all of them mean the same thing here.
    """
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def classify(gold: str, prediction: str) -> str:
    """Return the confusion-matrix outcome for one row.

    Note the order: a wrong non-empty answer on a gold row is FP, not FN. FN is
    reserved for staying silent when there was something to find, so the two
    ways of failing a gold row stay distinguishable.
    """
    if gold and prediction == gold:
        return "TP"
    if prediction:
        return "FP"
    if gold:
        return "FN"
    return "TN"


class ExtractionMatchMetric:
    """Scores an extracted answer against gold, empty answers included.

    ``prediction.text`` is compared as given, so a benchmark that has to parse
    the answer out of a longer reply passes a :class:`Prediction` carrying the
    parsed answer rather than the raw completion.
    """

    def score(self, record: EvalRecord, prediction: Prediction) -> SampleScore:
        if prediction.failed:
            return SampleScore(detail={"error": prediction.error})

        gold = as_answer(record.gold)
        got = as_answer(prediction.text)
        outcome = classify(gold, got)
        return SampleScore(
            value=1.0 if outcome == "TP" else 0.0,
            correct=outcome in {"TP", "TN"},
            # An empty answer where gold exists is the model declining, not the
            # parser failing, so only a blank reply to a non-blank model output
            # counts as a parse failure.
            parse_failed=bool(prediction.text.strip()) and not got,
            detail={"outcome": outcome, "expected": gold, "got": got},
        )


@METRICS.register("extraction_match")
def _make() -> ExtractionMatchMetric:
    return ExtractionMatchMetric()


__all__ = ["OUTCOMES", "ExtractionMatchMetric", "as_answer", "classify"]
