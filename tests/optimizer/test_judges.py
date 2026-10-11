"""Deterministic judges + weighted aggregation (ported reference behavior)."""

from __future__ import annotations

import pytest

# Absent from OSS checkouts — skip the module rather than fail collection.
pytest.importorskip("foresightrl_intuit")

from foresightrl_intuit.optimizer.judges.deterministic import (  # noqa: E402
    REQUIRED_SECTIONS,
    LengthJudge,
    NoRecommendationsJudge,
    NumericOverlapJudge,
    SectionsJudge,
)

from foresightrlcore.optimizer.interfaces import (
    DETERMINISTIC_FAIL_SCORE,
    DETERMINISTIC_PASS_SCORE,
    GenResult,
    JudgeScore,
    Record,
)
from foresightrlcore.optimizer.judges.aggregate import (
    aggregate_judge_scores,
    calculate_primary_statistics,
    calculate_summary_statistics,
)


def _record(evidence: str = "", reference: str = "") -> Record:
    return Record(
        realm_id="r1",
        epoch_date="2024-01-01",
        inputs={"evidence_bundle": evidence},
        reference=reference,
    )


def _gen(text: str) -> GenResult:
    return GenResult(text=text)


class TestSectionsJudge:
    def test_pass_when_all_sections_present(self):
        text = "\n".join(f"## {s}" for s in REQUIRED_SECTIONS)
        verdict = SectionsJudge().score(_record(), _gen(text))
        assert verdict.passed is True
        assert verdict.score == DETERMINISTIC_PASS_SCORE

    def test_fail_when_a_section_missing(self):
        text = "## " + REQUIRED_SECTIONS[0]
        verdict = SectionsJudge().score(_record(), _gen(text))
        assert verdict.passed is False
        assert verdict.score == DETERMINISTIC_FAIL_SCORE

    def test_custom_required_sections_param(self):
        judge = SectionsJudge({"required": ["Alpha", "Beta"]})
        assert judge.score(_record(), _gen("has Alpha and Beta")).passed is True
        assert judge.score(_record(), _gen("has Alpha only")).passed is False


class TestNoRecommendationsJudge:
    def test_pass_without_advisory_heading(self):
        assert NoRecommendationsJudge().score(_record(), _gen("# Overview")).passed

    def test_fail_on_recommendations_heading(self):
        verdict = NoRecommendationsJudge().score(
            _record(), _gen("## Recommendations\ndo x")
        )
        assert verdict.passed is False

    def test_fail_on_next_steps_heading(self):
        verdict = NoRecommendationsJudge().score(
            _record(), _gen("### Next Steps\n- foo")
        )
        assert verdict.passed is False


class TestNumericOverlapJudge:
    def test_pass_when_no_numbers_emitted(self):
        verdict = NumericOverlapJudge().score(_record("123 456"), _gen("no digits"))
        assert verdict.passed is True

    def test_pass_when_overlap_above_min(self):
        judge = NumericOverlapJudge({"min_overlap": 0.5})
        verdict = judge.score(_record("100 200 300"), _gen("100 and 200"))
        assert verdict.passed is True

    def test_fail_on_fabricated_numbers(self):
        judge = NumericOverlapJudge({"min_overlap": 0.5})
        verdict = judge.score(_record("100"), _gen("999 888 777"))
        assert verdict.passed is False

    def test_comma_normalization(self):
        verdict = NumericOverlapJudge().score(_record("1,000"), _gen("1000"))
        assert verdict.passed is True


class TestLengthJudge:
    def test_pass_within_band(self):
        verdict = LengthJudge().score(_record(reference="x" * 100), _gen("y" * 100))
        assert verdict.passed is True

    def test_fail_when_too_short(self):
        verdict = LengthJudge({"low": 0.5, "high": 2.0}).score(
            _record(reference="x" * 100), _gen("y" * 10)
        )
        assert verdict.passed is False

    def test_empty_reference_nonempty_candidate_passes(self):
        assert LengthJudge().score(_record(reference=""), _gen("y")).passed is True

    def test_empty_reference_empty_candidate_fails(self):
        assert LengthJudge().score(_record(reference=""), _gen("")).passed is False


class TestAggregate:
    def _scores(self, values):
        return {
            name: JudgeScore(name=name, score=sc, passed=passed)
            for name, (sc, passed) in values.items()
        }

    def test_weighted_primary_and_pass(self):
        scores = self._scores(
            {
                "correctness": (5, True),
                "hallucination": (3, True),
                "sections": (5, True),  # gate, weight 0
            }
        )
        weights = {"correctness": 1.0, "hallucination": 1.0, "sections": 0.0}
        agg = aggregate_judge_scores(scores, weights, acceptance_threshold=0.8)
        # (5+3)/2 = 4.0
        assert agg["primary_metric_value"] == 4.0
        assert agg["evaluation_status"] == "pass"
        assert agg["gates_passed"] is True
        assert agg["all_passed"] is True

    def test_gate_failure_recorded(self):
        scores = self._scores({"correctness": (5, True), "sections": (1, False)})
        weights = {"correctness": 1.0, "sections": 0.0}
        agg = aggregate_judge_scores(scores, weights, acceptance_threshold=0.8)
        assert agg["gates_passed"] is False
        assert agg["all_passed"] is False
        # gate weight 0 does not move the primary metric
        assert agg["primary_metric_value"] == 5.0

    def test_unweighted_falls_back_to_mean(self):
        scores = self._scores({"a": (2, True), "b": (4, True)})
        weights = {"a": 0.0, "b": 0.0}
        agg = aggregate_judge_scores(scores, weights, acceptance_threshold=0.8)
        assert agg["primary_metric_value"] == 3.0

    def test_status_uses_strict_greater_than(self):
        scores = self._scores({"a": (1, True)})
        weights = {"a": 1.0}
        agg = aggregate_judge_scores(scores, weights, acceptance_threshold=1.0)
        assert agg["evaluation_status"] == "fail"  # 1.0 > 1.0 is False


class TestSummaryStatistics:
    def test_primary_statistics_keys(self):
        stats = calculate_primary_statistics([1.0, 2.0, 3.0, 4.0])
        for key in (
            "primary_mean",
            "primary_std",
            "primary_min",
            "primary_P25",
            "primary_P50",
            "primary_P75",
            "primary_max",
        ):
            assert key in stats

    def test_empty_primary_statistics(self):
        stats = calculate_primary_statistics([])
        assert stats["primary_mean"] == 0.0

    def test_summary_success_rate_and_median(self):
        rows = [
            {"primary_metric_value": 4.0, "evaluation_status": "pass"},
            {"primary_metric_value": 2.0, "evaluation_status": "fail"},
            {"primary_metric_value": 5.0, "evaluation_status": "pass"},
        ]
        summary = calculate_summary_statistics(rows)
        assert summary["total"] == 3
        assert summary["successful"] == 2
        assert summary["failed"] == 1
        assert summary["success_rate"] == 2 / 3 * 100
        # median of positive primaries [4,2,5] -> 4
        assert summary["primary_metric_value"] == 4.0
