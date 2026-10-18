"""Default reporter: parquet, a leaderboard CSV, and a markdown summary.

Three artifacts because three audiences read them:

``results.parquet``
    Every per-trial row, plus every per-record row when the run had any.
    This is what you load to ask a question nobody anticipated.
``benchmark_scores.csv``
    The benchmark x model matrix, one number per cell -- the shape a model
    card or leaderboard wants.
``summary.md``
    What a human reads first: scores with intervals, and what went wrong.

Parquet needs pandas; the CSV and markdown are stdlib, so the report still
lands on a machine without it.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from foresightrlcore.evaluate.registry import REPORTERS
from foresightrlcore.evaluate.spec import INVALID_SCORE, EvaluateResult

logger = logging.getLogger(__name__)

#: Printed in a cell whose benchmark could not be measured.
INVALID_CELL = "n/a"


class TableReporter:
    """Writes the parquet, leaderboard CSV, and markdown summary."""

    def report(self, result: EvaluateResult, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self._write_parquet(result, output_dir)
        self._write_scores_csv(result, output_dir)
        (output_dir / "summary.md").write_text(render_summary(result), encoding="utf-8")
        logger.info("wrote evaluate report to %s", output_dir)

    @staticmethod
    def _write_parquet(result: EvaluateResult, output_dir: Path) -> None:
        try:
            import pandas as pd
        except ImportError:
            logger.warning("pandas is unavailable; skipping parquet output")
            return

        try:
            pd.DataFrame(result.results).to_parquet(output_dir / "results.parquet")
            if result.samples:
                pd.DataFrame(result.samples).to_parquet(output_dir / "samples.parquet")
        except Exception as exc:  # noqa: BLE001 - the other artifacts still stand
            logger.warning("failed to write parquet output: %s", exc)

    @staticmethod
    def _write_scores_csv(result: EvaluateResult, output_dir: Path) -> None:
        models = result.models or sorted({str(a["model"]) for a in result.aggregates})
        cells = {(str(a["benchmark"]), str(a["model"])): a for a in result.aggregates}

        with (output_dir / "benchmark_scores.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["benchmark", *(result.label(m) for m in models)])
            for benchmark in result.benchmarks:
                row: List[Any] = [benchmark]
                for model in models:
                    agg = cells.get((benchmark, model))
                    row.append(
                        round(float(agg["mean"]), 4)
                        if agg and agg.get("valid")
                        else INVALID_CELL
                    )
                writer.writerow(row)


def _format_score(agg: Dict[str, Any]) -> str:
    """Render one cell: the mean, with an interval when there were trials."""
    if not agg.get("valid"):
        return INVALID_CELL
    mean = float(agg["mean"])
    if int(agg.get("trials", 1)) < 2:
        return f"{mean:.4f}"
    halfwidth = (float(agg["ci_high"]) - float(agg["ci_low"])) / 2.0
    return f"{mean:.4f} ± {halfwidth:.4f}"


def render_summary(result: EvaluateResult) -> str:
    """Render ``summary.md``."""
    lines: List[str] = [
        f"# Evaluate — {result.job_name}",
        "",
        f"- Started: {result.started_at}",
        f"- Finished: {result.finished_at or 'in progress'}",
        f"- Benchmarks: {len(result.benchmarks)}",
        f"- Models: {', '.join(result.label(m) for m in result.models)}",
        f"- Trials: {result.trials}",
        "",
    ]

    if not result.aggregates:
        lines.append("_No results produced._")
        return "\n".join(lines) + "\n"

    models = result.models or sorted({str(a["model"]) for a in result.aggregates})
    cells = {(str(a["benchmark"]), str(a["model"])): a for a in result.aggregates}

    lines.append("## Scores")
    lines.append("")
    lines.append("| benchmark | " + " | ".join(result.label(m) for m in models) + " |")
    lines.append("|---" * (len(models) + 1) + "|")
    for benchmark in result.benchmarks:
        row = [benchmark]
        for model in models:
            agg = cells.get((benchmark, model))
            row.append(_format_score(agg) if agg else INVALID_CELL)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    if result.trials > 1:
        lines.append(
            f"Intervals are {int(result.aggregates[0].get('ci_level', 0.95) * 100)}% "
            "confidence intervals from Student's t over the trials."
        )
        lines.append("")

    lines.extend(_cost_section(result.aggregates, result.label))
    lines.extend(_failure_section(result.results, result.label))
    return "\n".join(lines) + "\n"


def _cost_section(
    aggregates: Sequence[Dict[str, Any]], label: Callable[[str], str]
) -> List[str]:
    total = sum(float(a.get("cost_usd", 0.0)) for a in aggregates)
    if total <= 0:
        return []

    by_model: Dict[str, float] = {}
    for agg in aggregates:
        name = label(str(agg["model"]))
        by_model[name] = by_model.get(name, 0.0) + float(agg.get("cost_usd", 0.0))

    lines = ["## Cost", "", "| model | usd |", "|---|---:|"]
    lines.extend(
        f"| {model} | {cost:.4f} |" for model, cost in sorted(by_model.items())
    )
    lines.append(f"| **total** | **{total:.4f}** |")
    lines.append("")
    return lines


def _failure_section(
    results: Sequence[Dict[str, Any]], label: Callable[[str], str]
) -> List[str]:
    """List the trials that failed outright or were reported invalid.

    Kept in the summary rather than buried in the parquet: an invalid cell
    without an explanation is the fastest way to lose trust in a report.
    """
    failed = [
        row
        for row in results
        if row.get("error") or float(row.get("score", 0.0)) == INVALID_SCORE
    ]
    if not failed:
        return []

    lines = [
        "## Failures",
        "",
        "| benchmark | model | trial | reason |",
        "|---|---|---:|---|",
    ]
    for row in failed:
        reason = str(row.get("error") or "exceeded failure tolerance")
        lines.append(
            f"| {row.get('benchmark')} | {label(str(row.get('model', '')))} | "
            f"{row.get('trial')} | {reason[:160]} |"
        )
    lines.append("")
    return lines


@REPORTERS.register("table")
def _make() -> TableReporter:
    return TableReporter()


__all__ = ["INVALID_CELL", "TableReporter", "render_summary"]
