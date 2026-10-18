"""Weighted aggregation of judge scores (ported from the reference).

Behavior preserved exactly:

- ``primary = round(sum(w*score) / sum(weights), 2)`` over weighted judges;
  falls back to the mean of all scores when no judge is weighted.
- ``status = "pass" if primary > acceptance_threshold`` (strict ``>``).
- summary uses the **median** of positive primaries + ``describe()``-equivalent
  keys ``primary_mean/std/min/P25/P50/P75/max``.

Judges with weight 0 act as deterministic gates: they do not move the primary
metric but their pass/fail is still recorded (and gates must pass).
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Dict, List

from foresightrlcore.optimizer.interfaces import JudgeScore


def aggregate_judge_scores(
    scores: Dict[str, JudgeScore],
    weights: Dict[str, float],
    acceptance_threshold: float,
) -> Dict[str, Any]:
    """Aggregate per-judge scores into a primary metric + pass/fail status."""
    weighted = [(weights.get(n, 0.0), s.score) for n, s in scores.items()]
    total_w = sum(w for w, _ in weighted)
    if total_w > 0:
        primary = round(sum(w * s for w, s in weighted) / total_w, 2)
    else:
        all_scores = [s.score for s in scores.values()]
        primary = round(statistics.mean(all_scores), 2) if all_scores else 0.0

    gates_passed = all(
        s.passed for n, s in scores.items() if weights.get(n, 0.0) == 0.0
    )
    all_passed = all(s.passed for s in scores.values())
    status = "pass" if primary > acceptance_threshold else "fail"

    return {
        "primary_metric_value": primary,
        "evaluation_status": status,
        "gates_passed": gates_passed,
        "all_passed": all_passed,
    }


def calculate_primary_statistics(primary_scores: List[float]) -> Dict[str, float]:
    """Mean/std/min/quartiles/max of primary scores (pandas ``describe``)."""
    if not primary_scores:
        return {
            "primary_mean": 0.0,
            "primary_std": 0.0,
            "primary_min": 0.0,
            "primary_P25": 0.0,
            "primary_P50": 0.0,
            "primary_P75": 0.0,
            "primary_max": 0.0,
        }
    import pandas as pd

    stats = pd.Series(primary_scores).describe()
    out: Dict[str, float] = {}
    for k, v in stats.to_dict().items():
        transformed = re.sub(r"(\d+)%", r"P\1", k)
        out[f"primary_{transformed}"] = round(float(v), 3)
    return out


def calculate_summary_statistics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Comprehensive summary over result rows (ported subset)."""
    primary_scores = [
        r.get("primary_metric_value", 0)
        for r in rows
        if r.get("primary_metric_value", 0) > 0
    ]
    primary_stats = calculate_primary_statistics(primary_scores)
    primary_value = statistics.median(primary_scores) if primary_scores else 0.0

    successful = len([r for r in rows if r.get("evaluation_status") == "pass"])
    failed = len([r for r in rows if r.get("evaluation_status") == "fail"])
    errored = len([r for r in rows if r.get("evaluation_status") == "error"])
    total = len(rows)
    success_rate = (successful / total * 100) if total else 0.0
    error_rate = (errored / total * 100) if total else 0.0

    return {
        "total": total,
        "successful": successful,
        "failed": failed,
        "errored": errored,
        "success_rate": success_rate,
        "error_rate": error_rate,
        "primary_metric_value": primary_value,
        "primary_metric_statistics": primary_stats,
    }


__all__ = [
    "aggregate_judge_scores",
    "calculate_primary_statistics",
    "calculate_summary_statistics",
]
