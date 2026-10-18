"""Precision / recall reporting for extraction benchmarks, with slice breakdowns.

The runner's own aggregation answers "what fraction of rows were right", which
is the right summary when every row asks the same kind of question. Extraction
benchmarks are not like that: some rows have something to find and some do not,
so one accuracy number silently averages two different skills — picking the
right item, and declining when there is nothing to pick.

This reporter reads the per-row outcomes that ``extraction_match`` leaves in
``SampleScore.detail`` and derives:

* precision, recall and F1 over the whole run;
* the same, per value of a ``detail`` key (``group_key``) — for reading results
  by category;
* a breakdown by a second ``detail`` key (``segment_key``) where named segments
  are scored by abstention accuracy instead, because on those rows the only
  correct answer is the empty one and precision and recall are undefined.

Recall divides by the number of rows carrying a gold answer, not by ``TP + FN``.
The two differ: a gold row answered with the wrong value counts as FP, so it
leaves ``TP + FN`` while staying in the gold-row total. This is the stricter
reading, and the defensible one — a wrong answer on an answerable row should not
be excused.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from foresightrlcore.evaluate.metrics.aggregate import mean_with_ci
from foresightrlcore.evaluate.metrics.extraction import OUTCOMES
from foresightrlcore.evaluate.registry import REPORTERS
from foresightrlcore.evaluate.spec import EvaluateResult

logger = logging.getLogger(__name__)

FILENAME = "confusion.json"

#: Metrics an interval is reported for across trials. Ratios of counts, so each
#: has to be computed within a trial and averaged after -- pooling rows from
#: several trials first would produce one number for a measurement that was
#: taken several times, and no way to see how much it moved.
INTERVAL_METRICS = ("precision", "recall", "f1", "exact_accuracy")


def confusion_counts(outcomes: Iterable[str]) -> Dict[str, int]:
    """Tally outcomes, always reporting all four keys."""
    counter = Counter(outcomes)
    return {outcome: int(counter.get(outcome, 0)) for outcome in OUTCOMES}


def precision_recall(counts: Dict[str, int], num_gold: int) -> Dict[str, float]:
    """Derive precision, recall and F1. See the module docstring on ``num_gold``."""
    true_positives = counts.get("TP", 0)
    predicted = true_positives + counts.get("FP", 0)

    precision = true_positives / predicted if predicted else 0.0
    recall = true_positives / num_gold if num_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def abstention_accuracy(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Score rows whose only correct answer is the empty one.

    Precision and recall are undefined here — there is nothing to retrieve, so
    no answer can be right and every answer is wrong. What is left is one binary
    question per row, "did the model hold back", making plain accuracy the honest
    summary.
    """
    total = len(rows)
    abstained = sum(1 for row in rows if not _got(row))
    return {
        "num_records": total,
        "num_abstained": abstained,
        "num_over_matched": total - abstained,
        "accuracy": abstained / total if total else 0.0,
    }


def _got(row: Dict[str, Any]) -> str:
    return str(row.get("detail_got") or "").strip()


def _expected(row: Dict[str, Any]) -> str:
    return str(row.get("detail_expected") or "").strip()


def _slice_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    counts = confusion_counts(str(row.get("detail_outcome", "")) for row in rows)
    num_gold = sum(1 for row in rows if _expected(row))
    return {
        "num_records": len(rows),
        "num_gold": num_gold,
        "counts": counts,
        # For a slice where every row has gold, the share answered exactly right
        # reads more directly than precision does.
        "exact_accuracy": counts["TP"] / len(rows) if rows else 0.0,
        **precision_recall(counts, num_gold),
    }


def _group_by(
    rows: Sequence[Dict[str, Any]], key: str
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        buckets.setdefault(str(value), []).append(row)
    return sorted(buckets.items())


class ConfusionReporter:
    """Writes per-model confusion metrics with optional slice breakdowns.

    Args:
        group_key: ``detail`` key to break results down by, reported under
            ``by_group``. Scored the same way as the whole run.
        segment_key: ``detail`` key for a second breakdown, reported under
            ``by_segment``, where ``abstention_segments`` are scored differently.
        abstention_segments: Segment names whose only correct answer is empty.
        filename: Output file name.
        confidence_level: Interval level used when a run had several trials.
    """

    def __init__(
        self,
        *,
        group_key: str = "group",
        segment_key: str = "segment",
        abstention_segments: Sequence[str] = (),
        filename: str = FILENAME,
        confidence_level: float = 0.95,
    ) -> None:
        self.group_key = f"detail_{group_key}" if group_key else ""
        self.segment_key = f"detail_{segment_key}" if segment_key else ""
        self.abstention_segments = set(abstention_segments)
        self.filename = filename
        self.confidence_level = confidence_level

    def summarise(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Roll one model's sample rows into a metrics document."""
        scored = [row for row in rows if not row.get("error")]
        summary: Dict[str, Any] = {
            **_slice_metrics(scored),
            "num_errors": len(rows) - len(scored),
            "num_truncated": sum(
                1 for row in scored if row.get("finish_reason") == "length"
            ),
        }

        if self.group_key:
            groups = _group_by(scored, self.group_key)
            if groups:
                summary["by_group"] = {
                    name: _slice_metrics(bucket) for name, bucket in groups
                }

        if self.segment_key:
            segments = _group_by(scored, self.segment_key)
            if segments:
                summary["by_segment"] = {
                    name: (
                        {"expected": "no_match", **abstention_accuracy(bucket)}
                        if name in self.abstention_segments
                        else {"expected": "match", **_slice_metrics(bucket)}
                    )
                    for name, bucket in segments
                }

        return summary

    def summarise_trials(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarise one model's rows, keeping trials apart.

        A single-trial run reports exactly what :meth:`summarise` produces. With
        several trials each is summarised on its own rows and the headline
        metrics gain a mean and an interval, because F1 is a ratio of counts and
        so cannot be recovered by averaging per-record scores the way the
        runner's own aggregation does.
        """
        by_trial: Dict[int, List[Dict[str, Any]]] = {}
        for row in rows:
            by_trial.setdefault(int(row.get("trial", 0) or 0), []).append(row)

        if len(by_trial) == 1:
            return self.summarise(next(iter(by_trial.values())))

        trials = {
            str(trial): self.summarise(trial_rows)
            for trial, trial_rows in sorted(by_trial.items())
        }
        aggregate: Dict[str, Any] = {}
        for metric in INTERVAL_METRICS:
            values = [float(t[metric]) for t in trials.values() if metric in t]
            mean, stdev, low, high = mean_with_ci(values, self.confidence_level)
            aggregate[metric] = {
                "mean": mean,
                "stdev": stdev,
                "ci_low": low,
                "ci_high": high,
            }
        return {
            "trials": trials,
            "n_trials": len(trials),
            "ci_level": self.confidence_level,
            "aggregate": aggregate,
        }

    def report(self, result: EvaluateResult, output_dir: Path) -> None:
        if not result.samples:
            logger.info("confusion reporter: run produced no per-record samples")
            return

        document: Dict[str, Dict[str, Any]] = {}
        for row in result.samples:
            document.setdefault(str(row.get("benchmark", "")), {}).setdefault(
                str(row.get("model", "")), []
            ).append(row)

        summaries = {
            benchmark: {
                model: self.summarise_trials(rows)
                for model, rows in sorted(models.items())
            }
            for benchmark, models in sorted(document.items())
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / self.filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(summaries, handle, indent=2, default=str)
        logger.info("wrote %s", path)

        for benchmark, models in summaries.items():
            for model, summary in models.items():
                logger.info("\n%s", format_summary(summary, f"{benchmark} — {model}"))


def _format_trials(summary: Dict[str, Any], title: str) -> str:
    """Render a repeated measurement: the interval first, then each trial."""
    lines: List[str] = [
        "=" * 64,
        f"  {title}",
        "=" * 64,
        f"  {summary['n_trials']} trials, {summary['ci_level']:.0%} CI",
    ]
    for metric, stats in summary["aggregate"].items():
        lines.append(
            f"  {metric:<18}{float(stats['mean']):.3f}  "
            f"[{float(stats['ci_low']):.3f}, {float(stats['ci_high']):.3f}]"
        )
    for trial, trial_summary in summary["trials"].items():
        lines.append("-" * 64)
        lines.append(format_summary(trial_summary, f"{title} — trial {trial}"))
    return "\n".join(lines)


def format_summary(summary: Dict[str, Any], title: str) -> str:
    """Render :meth:`ConfusionReporter.summarise` output for the job log."""
    if "trials" in summary:
        return _format_trials(summary, title)

    counts = summary["counts"]
    lines: List[str] = [
        "=" * 64,
        f"  {title}",
        "=" * 64,
        f"  Records            {summary['num_records']}",
        f"  Rows with gold     {summary['num_gold']}  (recall denominator)",
        "  TP / FP / FN / TN  "
        f"{counts['TP']} / {counts['FP']} / {counts['FN']} / {counts['TN']}",
        f"  Precision          {float(summary['precision']):.3f}",
        f"  Recall             {float(summary['recall']):.3f}",
        f"  F1                 {float(summary['f1']):.3f}",
    ]

    if summary.get("num_truncated"):
        lines.append(
            f"  !! TRUNCATED       {summary['num_truncated']}  "
            "— hit max_tokens; scores above understate the model"
        )

    by_segment = summary.get("by_segment")
    if by_segment:
        lines.append("-" * 64)
        lines.append("  Expected answer differs per segment — scored accordingly")
        for name, segment in sorted(
            by_segment.items(), key=lambda item: item[1].get("expected", "")
        ):
            if segment.get("expected") == "no_match":
                lines.append(
                    f"  {name:<16}{segment['num_records']:>6}  "
                    f"abstained {segment['num_abstained']:>4}"
                    f"  accuracy {float(segment['accuracy']):.3f}"
                )
            else:
                lines.append(
                    f"  {name:<16}{segment['num_records']:>6}  "
                    f"exact {float(segment['exact_accuracy']):.3f}"
                    f"  P {float(segment['precision']):.3f}"
                    f"  R {float(segment['recall']):.3f}"
                )

    by_group = summary.get("by_group")
    if by_group:
        lines.append("-" * 64)
        lines.append(f"  {'Group':<12}{'N':>7}{'TP':>7}{'Precision':>12}{'Recall':>9}")
        for name, group in sorted(by_group.items()):
            lines.append(
                f"  {name:<12}{group['num_records']:>7}{group['counts']['TP']:>7}"
                f"{float(group['precision']):>12.3f}{float(group['recall']):>9.3f}"
            )

    lines.append("=" * 64)
    return "\n".join(lines)


@REPORTERS.register("confusion")
def _make(
    group_key: str = "group",
    segment_key: str = "segment",
    abstention_segments: Sequence[str] = (),
    filename: str = FILENAME,
) -> ConfusionReporter:
    return ConfusionReporter(
        group_key=group_key,
        segment_key=segment_key,
        abstention_segments=abstention_segments,
        filename=filename,
    )


__all__ = [
    "FILENAME",
    "ConfusionReporter",
    "abstention_accuracy",
    "confusion_counts",
    "format_summary",
    "precision_recall",
]
