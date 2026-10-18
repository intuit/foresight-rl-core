"""LLM-as-judge scoring, result coercion, clamping, and error fallback."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from foresightrlcore.optimizer.interfaces import GenResult, Record
from foresightrlcore.optimizer.judges.llm_judges import (
    EvaluationResult,
    LlmJudge,
    _make_correctness,
    _make_hallucination,
)


def _record() -> Record:
    return Record(
        realm_id="r1",
        epoch_date="2024-01-01",
        inputs={"evidence_bundle": "revenue up 10%"},
        reference="reference state",
    )


def _gen(text: str = "candidate output") -> GenResult:
    return GenResult(text=text)


def _manager_returning(result):
    structured = MagicMock()
    structured.invoke.return_value = result
    client = MagicMock()
    client.with_structured_output.return_value = structured
    manager = MagicMock()
    manager.get.return_value = client
    return manager


class TestConstruction:
    def test_unknown_judge_name_raises(self):
        with pytest.raises(ValueError, match="unknown LLM judge"):
            LlmJudge("nonsense", MagicMock())

    def test_missing_manager_raises(self):
        with pytest.raises(ValueError, match="requires a client manager"):
            LlmJudge("correctness", None)


class TestScoring:
    def test_score_from_evaluation_result(self):
        manager = _manager_returning(EvaluationResult(score=5, reasoning="great"))
        judge = LlmJudge("correctness", manager, threshold=3.0)
        verdict = judge.score(_record(), _gen())
        assert verdict.name == "correctness"
        assert verdict.score == 5
        assert verdict.passed is True
        assert verdict.reasoning == "great"

    def test_score_clamped_to_range(self):
        manager = _manager_returning(EvaluationResult(score=9, reasoning="x"))
        verdict = LlmJudge("correctness", manager).score(_record(), _gen())
        assert verdict.score == 5

    def test_score_below_threshold_fails(self):
        manager = _manager_returning(EvaluationResult(score=2, reasoning="meh"))
        judge = LlmJudge("hallucination", manager, threshold=3.0)
        verdict = judge.score(_record(), _gen())
        assert verdict.passed is False

    def test_score_from_dict_result(self):
        manager = _manager_returning({"score": 4, "reasoning": "ok"})
        verdict = LlmJudge("correctness", manager).score(_record(), _gen())
        assert verdict.score == 4
        assert verdict.reasoning == "ok"

    def test_score_from_object_result(self):
        result = SimpleNamespace(score=3, reasoning="fine")
        manager = _manager_returning(result)
        verdict = LlmJudge("correctness", manager).score(_record(), _gen())
        assert verdict.score == 3
        assert verdict.reasoning == "fine"

    def test_score_error_fallback(self):
        manager = MagicMock()
        client = MagicMock()
        client.with_structured_output.side_effect = RuntimeError("boom")
        manager.get.return_value = client
        # max_retries=1 -> no backoff sleep before the fallback.
        judge = LlmJudge("correctness", manager, max_retries=1, base_delay=0.0)
        verdict = judge.score(_record(), _gen())
        assert verdict.score == 1
        assert verdict.passed is False
        assert "Evaluation failed" in verdict.reasoning


class TestFactories:
    def test_make_correctness_threshold(self):
        judge = _make_correctness({"threshold": 4.0}, manager=MagicMock())
        assert isinstance(judge, LlmJudge)
        assert judge.name == "correctness"
        assert judge._threshold == 4.0

    def test_make_hallucination_default_threshold(self):
        judge = _make_hallucination(manager=MagicMock())
        assert isinstance(judge, LlmJudge)
        assert judge.name == "hallucination"
        assert judge._threshold == 3.0
