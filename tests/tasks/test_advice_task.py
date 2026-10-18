"""Tests for foresightrl_intuit.tasks.advice — trainer-side advice reward module.

The reward-shape logic (prompts, criteria, helpers, scoring math,
``evaluate_completion`` pipeline) lives in
:mod:`foresightrl_intuit.tasks.advice_judge_logic` and is tested directly
by ``test_advice_judge_logic.py``. This file covers only the
**trainer-specific** layer:

* Public interface (``get_reward_function``, ``get_callbacks``,
  ``get_dataset_loader``) and mode dispatch.
* The LangChain transport closure (``_build_judge_call``) — structured
  output, retries, auth-error recovery.
* The reward function factory (``_build_reward_function``) — TRL
  contract (``list[float]`` output), criterion buffer feeding,
  predictions-log writing, thread-pool dispatch.
* The mock reward path.
* The criterion-logging callback.
"""

from __future__ import annotations

import collections
import json
import os
import sys
import tempfile
import threading
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Stub container-side dependencies not installed locally before importing
# foresightrl_intuit.tasks.advice (which transitively pulls them).
for _mod in ("transformers", "langchain_openai", "datasets", "peft"):
    sys.modules.setdefault(_mod, MagicMock())

# Absent from OSS checkouts — skip the module rather than fail collection.
pytest.importorskip("foresightrl_intuit")

from foresightrl_intuit.tasks.advice import (  # noqa: E402
    REWARD_MODE_LLM_JUDGE,
    REWARD_MODE_MOCK,
    CriterionLoggingCallback,
    QualityScoreItem,
    _build_judge_call,
    _build_mock_reward_function,
    _build_reward_function,
    get_callbacks,
    get_dataset_loader,
    get_reward_function,
)

# ---------------------------------------------------------------------------
# Helper: build a fake LangChain LLMClientManager whose structured_output
# returns a configured QualityScoreItem.
# ---------------------------------------------------------------------------


def _mgr_returning(item: QualityScoreItem) -> MagicMock:
    mock_mgr = MagicMock()
    mock_client = MagicMock()
    mock_mgr.get.return_value = mock_client
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = item
    mock_client.with_structured_output.return_value = mock_structured
    return mock_mgr


def _perfect_item() -> QualityScoreItem:
    return QualityScoreItem(
        id=1,
        safe=1,
        cites_numbers=1,
        names_entity=1,
        actionable=1,
        factual=1,
        insight_quality=1,
        risk_aware=1,
        quantified_impact=1,
        grounded_estimate=1,
        directionally_correct=1,
        gp_material=1,
        critique="great",
    )


def _zero_item() -> QualityScoreItem:
    return QualityScoreItem(
        id=1,
        safe=1,
        cites_numbers=0,
        names_entity=0,
        actionable=0,
        factual=0,
        insight_quality=0,
        risk_aware=0,
        quantified_impact=0,
        grounded_estimate=0,
        directionally_correct=0,
        gp_material=0,
        critique="all-zero",
    )


@pytest.fixture(autouse=True)
def _clear_module_globals() -> None:
    """Reset the criterion buffer + format-fail counter between tests."""
    import foresightrl_intuit.tasks.advice.trainer as mod

    with mod._criterion_buffer_lock:
        mod._criterion_buffer.clear()
        mod._format_fail_count = 0


# ===========================================================================
# Judge transport (LangChain closure)
# ===========================================================================


class TestBuildJudgeCall:
    def test_success_returns_score_from_judge(self) -> None:
        mgr = _mgr_returning(_perfect_item())
        judge_call = _build_judge_call(mgr)
        score, breakdown = judge_call(
            "state", {"recommendation": "raise prices", "expected_impact": "$50k"}
        )
        assert score == 1.0
        assert breakdown["safe"] == 1
        assert breakdown["total_score"] == 10

    def test_safe_zero_gates_score(self) -> None:
        item = _perfect_item()
        item.safe = 0
        mgr = _mgr_returning(item)
        judge_call = _build_judge_call(mgr)
        score, breakdown = judge_call("state", {"recommendation": "x"})
        assert score == 0.0
        assert breakdown["safe"] == 0

    def test_auth_error_triggers_force_refresh(self) -> None:
        from foresightrl_intuit.tasks.advice_judge_logic import JUDGE_NEUTRAL_FALLBACK

        mock_mgr = MagicMock()
        mock_client = MagicMock()
        mock_mgr.get.return_value = mock_client
        mock_structured = MagicMock()
        # Always fails with an auth-shaped error; after retries exhaust we
        # return the neutral fallback and force_refresh must have been called.
        mock_structured.invoke.side_effect = RuntimeError(
            "401 Unauthorized: ticket expired"
        )
        mock_client.with_structured_output.return_value = mock_structured

        judge_call = _build_judge_call(mock_mgr)
        with patch("foresightrl_intuit.tasks.advice.time.sleep"):
            score, breakdown = judge_call("state", {"recommendation": "x"})

        assert score == JUDGE_NEUTRAL_FALLBACK
        assert "error" in breakdown
        assert mock_mgr.force_refresh.called

    def test_retries_then_succeeds(self) -> None:
        mock_mgr = MagicMock()
        mock_client = MagicMock()
        mock_mgr.get.return_value = mock_client
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = [
            RuntimeError("transient blip"),
            _perfect_item(),
        ]
        mock_client.with_structured_output.return_value = mock_structured

        judge_call = _build_judge_call(mock_mgr)
        with patch("foresightrl_intuit.tasks.advice.time.sleep"):
            score, _ = judge_call("state", {"recommendation": "x"})
        assert score == 1.0
        assert mock_structured.invoke.call_count == 2


# ===========================================================================
# Reward function factory — TRL contract
# ===========================================================================


def _build_fn(
    mgr: MagicMock,
    *,
    soft_cap: int = 15_000,
    hard_cap: int = 30_000,
    penalty_weight: float = 0.20,
    predictions_log: str = "",
) -> Any:
    return _build_reward_function(
        critic_mgr=mgr,
        think_length_soft_cap=soft_cap,
        think_length_hard_cap=hard_cap,
        think_penalty_weight=penalty_weight,
        judge_parallel_workers=1,
        predictions_log=predictions_log,
    )


class TestBuildRewardFunction:
    def test_format_fail_flat_minus_half(self) -> None:
        fn = _build_fn(_mgr_returning(_perfect_item()))
        rewards = fn(completions=["no json here at all"], state=["some state"])
        assert rewards == [-0.5]

    def test_perfect_completion_unity_reward(self) -> None:
        fn = _build_fn(_mgr_returning(_perfect_item()))
        completion = (
            '{"recommendation":"cut costs","expected_impact":"$10K","risks":"none"}'
        )
        rewards = fn(completions=[completion], state=["business state data"])
        assert rewards[0] == pytest.approx(1.0)

    def test_thinking_penalty_subtracted(self) -> None:
        fn = _build_fn(
            _mgr_returning(_perfect_item()),
            soft_cap=10,
            hard_cap=20,
            penalty_weight=0.5,
        )
        long_thinking = "<think>" + "x" * 20 + "</think>"
        completion = long_thinking + '{"recommendation":"cut costs"}'
        rewards = fn(completions=[completion], state=["state"])
        # judge=1.0, penalty = 0.5 (full, at hard cap) -> reward 0.5
        assert rewards[0] == pytest.approx(0.5)

    def test_predictions_log_written(self) -> None:
        fn_mgr = _mgr_returning(_zero_item())
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            log_path = f.name
        try:
            fn = _build_fn(fn_mgr, predictions_log=log_path)
            fn(
                completions=['{"recommendation":"x"}'],
                state=["state"],
            )
            with open(log_path) as f:
                lines = f.readlines()
            assert len(lines) == 1
            rec = json.loads(lines[0])
            assert rec["reward"] == pytest.approx(0.0)
            assert rec["state"] == "state"
        finally:
            os.unlink(log_path)

    def test_thread_exception_returns_neutral(self) -> None:
        from foresightrl_intuit.tasks.advice_judge_logic import JUDGE_NEUTRAL_FALLBACK

        # judge.invoke always crashes; outer thread-pool catches and uses
        # the neutral fallback.
        mock_mgr = MagicMock()
        mock_client = MagicMock()
        mock_mgr.get.return_value = mock_client
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = RuntimeError("transport crash")
        mock_client.with_structured_output.return_value = mock_structured

        fn = _build_fn(mock_mgr)
        with patch("foresightrl_intuit.tasks.advice.time.sleep"):
            rewards = fn(
                completions=['{"recommendation":"x"}'],
                state=["state"],
            )
        # No think tags -> penalty 0; judge exhausted retries -> neutral 0.5
        assert rewards[0] == pytest.approx(JUDGE_NEUTRAL_FALLBACK)

    def test_dict_completion_format_supported(self) -> None:
        # TRL-style completions arrive as list[dict].
        fn = _build_fn(_mgr_returning(_perfect_item()))
        rewards = fn(
            completions=[[{"content": "no json"}]],
            state=["state"],
        )
        assert rewards[0] == -0.5

    def test_returns_one_float_per_completion(self) -> None:
        fn = _build_fn(_mgr_returning(_perfect_item()))
        completions = ['{"recommendation":"a"}'] * 5
        states = ["s"] * 5
        rewards = fn(completions=completions, state=states)
        assert len(rewards) == 5
        assert all(isinstance(r, float) for r in rewards)


# ===========================================================================
# Criterion logging callback — feeds TensorBoard via on_log()
# ===========================================================================


class TestCriterionLoggingCallback:
    def test_buffer_fed_on_success(self) -> None:
        import foresightrl_intuit.tasks.advice as mod

        fn = _build_fn(_mgr_returning(_perfect_item()))
        fn(completions=['{"recommendation":"r"}'], state=["s"])

        with mod._criterion_buffer_lock:
            # All ten criteria should have a 1.0 entry from the perfect item.
            assert len(mod._criterion_buffer) == 10
            for crit, scores in mod._criterion_buffer.items():
                assert scores == [1.0]

    def test_format_fail_count_incremented(self) -> None:
        import foresightrl_intuit.tasks.advice.trainer as mod

        fn = _build_fn(_mgr_returning(_perfect_item()))
        fn(completions=["no json"], state=["state"])
        with mod._criterion_buffer_lock:
            assert mod._format_fail_count == 1

    def test_callback_emits_averages_and_resets_buffer(self) -> None:
        import foresightrl_intuit.tasks.advice as mod

        # Pre-load the buffer with mixed scores.
        with mod._criterion_buffer_lock:
            mod._criterion_buffer["actionable"] = [1.0, 1.0, 0.0]
            mod._criterion_buffer["factual"] = [0.0, 1.0]

        cb = CriterionLoggingCallback()
        logs: Dict[str, float] = {}
        cb.on_log(args=None, state=None, control=None, logs=logs)

        assert logs["criteria/actionable"] == pytest.approx(2 / 3)
        assert logs["criteria/factual"] == pytest.approx(0.5)
        with mod._criterion_buffer_lock:
            assert mod._criterion_buffer == {}

    def test_callback_no_op_on_empty_buffer(self) -> None:
        cb = CriterionLoggingCallback()
        logs: Dict[str, float] = {}
        cb.on_log(args=None, state=None, control=None, logs=logs)
        assert logs == {}


# ===========================================================================
# Mock reward path
# ===========================================================================


class TestMockRewardFunction:
    def test_fixed_reward_no_sleep(self) -> None:
        fn = _build_mock_reward_function(
            sleep_seconds=0,
            constant_reward=0.75,
            log_message=None,
            predictions_log=None,
        )
        rewards = fn(completions=["a", "b", "c"], state=["s1", "s2", "s3"])
        assert rewards == [0.75, 0.75, 0.75]

    def test_predictions_log(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            log_path = f.name
        try:
            fn = _build_mock_reward_function(
                sleep_seconds=0,
                constant_reward=0.3,
                log_message="test_run",
                predictions_log=log_path,
            )
            fn(completions=["a", "b"], state=["s1", "s2"])
            with open(log_path) as f:
                lines = f.readlines()
            assert len(lines) == 2
            recs = [json.loads(line) for line in lines]
            assert all(r["reward"] == 0.3 for r in recs)
            assert all(r["mock_note"] == "test_run" for r in recs)
        finally:
            os.unlink(log_path)


# ===========================================================================
# Public interface
# ===========================================================================


class TestGetRewardFunction:
    def test_mock_mode(self) -> None:
        fn = get_reward_function({"reward_mode": "mock", "mock_sleep_seconds": 0})
        rewards = fn(completions=["a", "b"], state=["s1", "s2"])
        assert rewards == [0.5, 0.5]

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown task reward_mode"):
            get_reward_function({"reward_mode": "totally_invalid_xyz"})

    def test_critic_model_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="'critic_model_id' has been removed"):
            get_reward_function({"critic_model_id": "foo"})

    def test_llm_judge_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="'llm_judge' is no longer supported"):
            get_reward_function({"llm_judge": {"provider": "bedrock"}})

    def test_missing_llm_clients_in_llm_judge_mode(self) -> None:
        with pytest.raises(ValueError, match="task.config.llm_clients is required"):
            get_reward_function({})

    def test_dimension_weights_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        # The legacy override is silently ignored with a warning. Use the mock
        # path so we don't need real LLM clients configured.
        with caplog.at_level(
            "WARNING", logger="foresightrl_intuit.tasks.advice.trainer"
        ):
            fn = get_reward_function(
                {
                    "reward_mode": "mock",
                    "mock_sleep_seconds": 0,
                    "dimension_weights": {"factual": 2.0},
                }
            )
            # Mock path skips llm_clients resolution -> the warning would
            # only fire on the llm_judge path; verify the warning path
            # explicitly:
        # Switch to the LLM path with stubbed manager so the warning fires.
        with patch(
            "foresightrl_intuit.tasks.advice.trainer._resolve_default_llm_client",
            return_value=MagicMock(),
        ):
            caplog.clear()
            with caplog.at_level(
                "WARNING", logger="foresightrl_intuit.tasks.advice.trainer"
            ):
                get_reward_function(
                    {
                        "llm_clients": {
                            "default": {
                                "provider": "bedrock",
                                "bedrock": {
                                    "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                                    "region": "us-west-2",
                                },
                            }
                        },
                        "dimension_weights": {"factual": 2.0},
                    }
                )
            assert any("dimension_weights" in r.message for r in caplog.records)
        assert callable(fn)


class TestGetCallbacks:
    def test_llm_judge_returns_criterion_callback(self) -> None:
        cbs = get_callbacks({"reward_mode": REWARD_MODE_LLM_JUDGE})
        assert len(cbs) == 1
        assert isinstance(cbs[0], CriterionLoggingCallback)

    def test_mock_returns_empty(self) -> None:
        cbs = get_callbacks({"reward_mode": REWARD_MODE_MOCK})
        assert cbs == []


class TestGetDatasetLoader:
    def test_returns_callable(self) -> None:
        loader = get_dataset_loader({}, "test-model")
        assert callable(loader)
