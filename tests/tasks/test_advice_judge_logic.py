"""Tests for foresightrl_intuit.tasks.advice_judge_logic — pure-Python judge pipeline.

This module is the single source of truth for the Advice task's reward
logic and is consumed by both the trainer reward function
(:mod:`foresightrl_intuit.tasks.advice`) and the SageMaker RLVR Lambda
(``serverless/advice/advice_rlvr_reward.py``). All tests here exercise
it directly so any drift in reward shape will surface as a single
failure here rather than in two independent test files.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pytest

# Absent from OSS checkouts — skip the module rather than fail collection.
pytest.importorskip("foresightrl_intuit")

from foresightrl_intuit.tasks.advice_judge_logic import (  # noqa: E402
    CRITERIA,
    CRITIC_PROMPT,
    CRITIC_SYSTEM_PROMPT,
    DIMENSION_WEIGHT_SUM,
    FORMAT_FAIL_REWARD,
    JUDGE_NEUTRAL_FALLBACK,
    THINK_LENGTH_HARD_CAP,
    THINK_LENGTH_SOFT_CAP,
    THINK_PENALTY_WEIGHT,
    JudgeResult,
    build_advice_section,
    compute_thinking_penalty,
    evaluate_completion,
    extract_thinking_length,
    parse_json_from_response,
    score_from_judge,
    strip_thinking_tags,
)

# ---------------------------------------------------------------------------
# Constants — frozen contract: any change here must be intentional
# ---------------------------------------------------------------------------


class TestConstants:
    def test_ten_criteria_in_canonical_order(self) -> None:
        assert CRITERIA == (
            "cites_numbers",
            "names_entity",
            "actionable",
            "factual",
            "insight_quality",
            "risk_aware",
            "quantified_impact",
            "grounded_estimate",
            "directionally_correct",
            "gp_material",
        )
        assert DIMENSION_WEIGHT_SUM == 10.0

    def test_thinking_caps_and_weight(self) -> None:
        assert THINK_LENGTH_SOFT_CAP == 15_000
        assert THINK_LENGTH_HARD_CAP == 25_000
        assert THINK_PENALTY_WEIGHT == 0.20

    def test_reward_anchors(self) -> None:
        assert FORMAT_FAIL_REWARD == -0.5
        assert JUDGE_NEUTRAL_FALLBACK == 0.5

    def test_prompts_have_expected_placeholders(self) -> None:
        assert "{state_excerpt}" in CRITIC_PROMPT
        assert "{advice_section}" in CRITIC_PROMPT
        assert "strict, calibrated evaluator" in CRITIC_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Thinking-tag helpers
# ---------------------------------------------------------------------------


class TestStripThinkingTags:
    def test_full_pair_removed(self) -> None:
        assert strip_thinking_tags("<think>noise</think>visible") == "visible"

    def test_lone_close_tag_strips_prefix(self) -> None:
        assert strip_thinking_tags("noise</think>actual content") == "actual content"

    def test_reasoning_tag_variant(self) -> None:
        assert strip_thinking_tags("<reasoning>r</reasoning>visible") == "visible"

    def test_no_tags_passthrough(self) -> None:
        assert strip_thinking_tags("plain text") == "plain text"

    def test_empty_string(self) -> None:
        assert strip_thinking_tags("") == ""


class TestExtractThinkingLength:
    def test_matched_pair(self) -> None:
        assert extract_thinking_length("a<think>12345</think>b") == 5

    def test_lone_close_tag_counts_prefix(self) -> None:
        assert extract_thinking_length("12345</think>tail") == 5

    def test_no_tags_returns_zero(self) -> None:
        assert extract_thinking_length("plain text") == 0

    def test_empty_pair_returns_zero(self) -> None:
        assert extract_thinking_length("<think></think>rest") == 0


class TestComputeThinkingPenalty:
    def test_no_penalty_below_soft_cap(self) -> None:
        content = "<think>" + "x" * 14_000 + "</think>"
        assert compute_thinking_penalty(content) == 0.0

    def test_no_penalty_at_soft_cap(self) -> None:
        content = "<think>" + "x" * 15_000 + "</think>"
        assert compute_thinking_penalty(content) == 0.0

    def test_linear_ramp_midway(self) -> None:
        # excess = 5000, range = 10000 -> penalty = 0.20 * 0.5 = 0.10
        content = "<think>" + "x" * 20_000 + "</think>"
        assert compute_thinking_penalty(content) == pytest.approx(0.10)

    def test_full_penalty_at_hard_cap(self) -> None:
        content = "<think>" + "x" * 25_000 + "</think>"
        assert compute_thinking_penalty(content) == pytest.approx(0.20)

    def test_capped_above_hard_cap(self) -> None:
        content = "<think>" + "x" * 50_000 + "</think>"
        assert compute_thinking_penalty(content) == pytest.approx(0.20)

    def test_custom_caps(self) -> None:
        # excess = 10, range = 10, weight 0.5 -> full penalty
        content = "<think>" + "x" * 20 + "</think>"
        penalty = compute_thinking_penalty(
            content, soft_cap=10, hard_cap=20, weight=0.5
        )
        assert penalty == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


class TestParseJsonFromResponse:
    def test_markdown_json_fence(self) -> None:
        assert parse_json_from_response('```json\n{"a": 1}\n```') == {"a": 1}

    def test_markdown_plain_fence(self) -> None:
        assert parse_json_from_response('```\n{"a": 1}\n```') == {"a": 1}

    def test_bare_object_in_surrounding_text(self) -> None:
        assert parse_json_from_response('lead {"a": 2} trail') == {"a": 2}

    def test_strips_thinking_before_parse(self) -> None:
        text = '<think>noise</think>{"a": 3}'
        assert parse_json_from_response(text) == {"a": 3}

    def test_returns_empty_on_unparseable(self) -> None:
        assert parse_json_from_response("not json at all") == {}

    def test_returns_empty_on_no_input(self) -> None:
        assert parse_json_from_response("") == {}


# ---------------------------------------------------------------------------
# Advice section builder
# ---------------------------------------------------------------------------


class TestBuildAdviceSection:
    def test_full_advice(self) -> None:
        section = build_advice_section(
            {
                "recommendation": "raise prices on Product A by 5%",
                "expected_impact": "$50k",
                "risks": "churn",
            }
        )
        assert "raise prices on Product A by 5%" in section
        assert "$50k" in section
        assert "churn" in section

    def test_missing_recommendation_placeholder(self) -> None:
        assert build_advice_section({}) == "ADVICE 1: [No recommendation]"

    def test_defaults_for_optional_fields(self) -> None:
        section = build_advice_section({"recommendation": "do x"})
        assert "Expected Impact: Not provided" in section
        assert "Risks: Not provided" in section


# ---------------------------------------------------------------------------
# Score-from-judge math
# ---------------------------------------------------------------------------


class TestScoreFromJudge:
    def _perfect_payload(self, *, safe: int = 1) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"safe": safe, "critique": "ok"}
        for crit in CRITERIA:
            payload[crit] = 1
        return payload

    def test_safe_gate_zeroes_score(self) -> None:
        parsed = self._perfect_payload(safe=0)
        score, breakdown = score_from_judge(parsed)
        assert score == 0.0
        assert breakdown == {"safe": 0, "total_score": 0, "critique": "ok"}

    def test_perfect_score(self) -> None:
        score, breakdown = score_from_judge(self._perfect_payload())
        assert score == 1.0
        assert breakdown["total_score"] == 10
        assert breakdown["safe"] == 1

    def test_partial_score(self) -> None:
        parsed: Dict[str, Any] = {"safe": 1, "critique": ""}
        for crit in CRITERIA:
            parsed[crit] = 0
        parsed["cites_numbers"] = 1
        parsed["actionable"] = 1
        parsed["gp_material"] = 1
        score, breakdown = score_from_judge(parsed)
        assert score == pytest.approx(0.3)
        assert breakdown["total_score"] == 3

    def test_clamps_invalid_values(self) -> None:
        parsed: Dict[str, Any] = {"safe": 1, "critique": ""}
        parsed["cites_numbers"] = 5  # > 1
        parsed["names_entity"] = -1  # < 0
        parsed["actionable"] = "1"  # string-int passthrough
        parsed["factual"] = "not-a-number"  # unparseable -> 0
        for crit in CRITERIA[4:]:
            parsed[crit] = 0
        score, breakdown = score_from_judge(parsed)
        # 1 (cites) + 0 (names, clamped from -1) + 1 (actionable) + 0 (factual) = 2 -> 0.2
        assert score == pytest.approx(0.2)
        assert breakdown["cites_numbers"] == 1
        assert breakdown["names_entity"] == 0
        assert breakdown["factual"] == 0


# ---------------------------------------------------------------------------
# evaluate_completion pipeline
# ---------------------------------------------------------------------------


def _stub_judge(score: float, breakdown: Dict[str, Any]) -> Any:
    """Return a JudgeCall stub that always returns the given (score, breakdown)."""

    def call(state: str, advice: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        return score, breakdown

    return call


class TestEvaluateCompletion:
    def test_format_fail_does_not_subtract_penalty(self) -> None:
        # 20k thinking chars -> would-be penalty 0.10; format-gate
        # returns FORMAT_FAIL_REWARD flat (matches grpo_advice.py).
        content = "<think>" + "x" * 20_000 + "</think>no recommendation here"
        called = []

        def fail_if_judged(state: str, advice: Dict[str, Any]) -> Any:
            called.append(True)
            return 1.0, {}

        result = evaluate_completion(content, "state", fail_if_judged)
        assert result.aggregate_reward == FORMAT_FAIL_REWARD
        assert result.format_fail is True
        assert result.judge_called is False
        assert result.think_length == 20_000
        assert result.think_penalty == pytest.approx(0.10)
        assert called == []

    def test_success_subtracts_penalty(self) -> None:
        content = (
            "<think>" + "x" * 20_000 + "</think>"
            '{"recommendation":"r","expected_impact":"i","risks":"k"}'
        )
        result = evaluate_completion(
            content,
            "state",
            _stub_judge(1.0, {"safe": 1, "total_score": 10, "critique": "good"}),
        )
        assert result.aggregate_reward == pytest.approx(0.9)
        assert result.judge_score == 1.0
        assert result.think_penalty == pytest.approx(0.10)
        assert result.format_fail is False
        assert result.judge_called is True

    def test_judge_passes_state_and_parsed_advice(self) -> None:
        captured: Dict[str, Any] = {}

        def capture(state: str, advice: Dict[str, Any]) -> Any:
            captured["state"] = state
            captured["advice"] = advice
            return 0.7, {"safe": 1, "total_score": 7, "critique": ""}

        content = (
            '{"recommendation":"raise prices","expected_impact":"$50k","risks":"churn"}'
        )
        evaluate_completion(content, "STATE BLOB", capture)
        assert captured["state"] == "STATE BLOB"
        assert captured["advice"]["recommendation"] == "raise prices"
        assert captured["advice"]["expected_impact"] == "$50k"

    def test_judge_transport_failure_propagates_neutral(self) -> None:
        # When judge_call returns the neutral fallback (e.g. exhausted retries),
        # evaluate_completion still applies the think-penalty subtraction.
        content = (
            "<think>" + "x" * 20_000 + "</think>"
            '{"recommendation":"r","expected_impact":"i","risks":"k"}'
        )
        result = evaluate_completion(
            content,
            "state",
            _stub_judge(JUDGE_NEUTRAL_FALLBACK, {"error": "judge_failed: ..."}),
        )
        # 0.5 (neutral) - 0.10 (penalty) = 0.40
        assert result.aggregate_reward == pytest.approx(0.40)
        assert "error" in result.breakdown
        assert result.format_fail is False

    def test_returns_judge_result_dataclass(self) -> None:
        result = evaluate_completion(
            '{"recommendation":"r","expected_impact":"i","risks":"k"}',
            "state",
            _stub_judge(0.5, {"safe": 1, "total_score": 5, "critique": ""}),
        )
        assert isinstance(result, JudgeResult)
        assert hasattr(result, "aggregate_reward")
        assert hasattr(result, "judge_score")
        assert hasattr(result, "think_length")
        assert hasattr(result, "think_penalty")
        assert hasattr(result, "format_fail")
        assert hasattr(result, "judge_called")
        assert hasattr(result, "breakdown")
