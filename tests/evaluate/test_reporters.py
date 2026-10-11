"""Reporters: parquet + leaderboard CSV + markdown, and the JSON manifest."""

from __future__ import annotations

import csv
import json

import pytest

from foresightrlcore.evaluate.reporters.json_reporter import JsonReporter
from foresightrlcore.evaluate.reporters.table import (
    INVALID_CELL,
    TableReporter,
    render_summary,
)
from foresightrlcore.evaluate.spec import INVALID_SCORE, EvaluateResult


def _aggregate(benchmark, model, mean, **kwargs):
    base = {
        "benchmark": benchmark,
        "model": model,
        "trials": 1,
        "mean": mean,
        "stdev": 0.0,
        "ci_low": mean,
        "ci_high": mean,
        "ci_level": 0.95,
        "scores": [mean],
        "sub_scores": {},
        "n_samples": 10,
        "n_errors": 0,
        "cost_usd": 0.5,
        "valid": True,
    }
    base.update(kwargs)
    return base


def _result(**kwargs) -> EvaluateResult:
    payload = {
        "job_name": "demo",
        "benchmarks": ["b1", "b2"],
        "models": ["m1", "m2"],
        "trials": 1,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:10:00+00:00",
        "n_results": 4,
        "aggregates": [
            _aggregate("b1", "m1", 0.9),
            _aggregate("b1", "m2", 0.5),
            _aggregate("b2", "m1", 0.7),
            _aggregate("b2", "m2", 0.0, valid=False),
        ],
        "results": [
            {"benchmark": "b1", "model": "m1", "trial": 0, "score": 0.9, "error": ""},
            {
                "benchmark": "b2",
                "model": "m2",
                "trial": 0,
                "score": INVALID_SCORE,
                "error": "benchmark timed out after 60s",
            },
        ],
        "samples": [],
    }
    payload.update(kwargs)
    return EvaluateResult.model_validate(payload)


def test_the_leaderboard_csv_is_a_benchmark_by_model_matrix(tmp_path):
    TableReporter().report(_result(), tmp_path)

    with (tmp_path / "benchmark_scores.csv").open() as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == ["benchmark", "m1", "m2"]
    assert rows[1] == ["b1", "0.9", "0.5"]
    # An unmeasurable cell is marked, not silently zero.
    assert rows[2][2] == INVALID_CELL


def test_the_summary_shows_scores_and_explains_the_gaps():
    summary = render_summary(_result())

    assert "| b1 | 0.9000 | 0.5000 |" in summary
    assert "## Failures" in summary
    assert "benchmark timed out after 60s" in summary
    # A single trial has no interval to report.
    assert "±" not in summary


def test_multi_trial_scores_carry_their_interval():
    result = _result(
        trials=3,
        aggregates=[
            _aggregate("b1", "m1", 0.8, trials=3, ci_low=0.7, ci_high=0.9),
        ],
        results=[],
    )

    summary = render_summary(result)

    assert "0.8000 ± 0.1000" in summary
    assert "Student's t" in summary


def test_the_summary_totals_cost_per_model():
    summary = render_summary(_result())

    assert "## Cost" in summary
    assert "**2.0000**" in summary


def test_an_empty_run_produces_a_readable_summary():
    summary = render_summary(_result(aggregates=[], results=[]))
    assert "_No results produced._" in summary


def test_parquet_is_written_when_pandas_is_available(tmp_path):
    pytest.importorskip("pandas")
    TableReporter().report(
        _result(samples=[{"benchmark": "b1", "record_id": "r1", "score": 1.0}]),
        tmp_path,
    )

    assert (tmp_path / "results.parquet").exists()
    assert (tmp_path / "samples.parquet").exists()


def test_reports_render_the_display_name_not_the_key(tmp_path):
    """A locally served candidate is keyed on its S3 prefix. Reproducing that
    as a column header makes the leaderboard unreadable, so reports render the
    label while every key -- checkpoints, fingerprints -- stays the prefix."""
    uri = "s3://bucket/models/Meta-Llama-3.1-8B-Instruct/"
    result = _result(
        models=[uri],
        aggregates=[_aggregate("b1", uri, 0.9)],
        results=[{"benchmark": "b1", "model": uri, "trial": 0, "score": INVALID_SCORE}],
        model_info={uri: {"display_name": "Llama-3.1-8B"}},
    )

    TableReporter().report(result, tmp_path)
    with (tmp_path / "benchmark_scores.csv").open() as handle:
        header = next(csv.reader(handle))
    summary = render_summary(result)

    assert header == ["benchmark", "Llama-3.1-8B"]
    assert "Llama-3.1-8B" in summary
    assert uri not in summary
    # The identity is untouched: the parquet still carries the real key.
    assert result.aggregates[0]["model"] == uri


def test_a_model_without_a_display_name_falls_back_to_its_key():
    assert render_summary(_result()).count("m1") > 0


def test_the_json_manifest_round_trips_the_whole_run(tmp_path):
    JsonReporter().report(_result(), tmp_path)

    payload = json.loads((tmp_path / "evaluate_results.json").read_text())

    assert payload["job_name"] == "demo"
    assert len(payload["aggregates"]) == 4
    # Self-contained: the per-trial rows are here too, not only in the parquet.
    assert len(payload["results"]) == 2
