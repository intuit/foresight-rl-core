"""Unit tests for ``foresightrl_intuit.tasks.advice.lambda_reward``.

Covers the AWS-specific glue layer of the SageMaker RLVR Lambda. The
reward-shape logic (prompts, thinking-penalty, JSON parsing, scoring
math, ``evaluate_completion`` pipeline) lives in
:mod:`foresightrl_intuit.tasks.advice.judge_logic` and is tested directly
by ``tests/tasks/test_advice_judge_logic.py``. This file therefore
focuses on:

* Bedrock ``converse`` transport (success + retry on transient
  ``ClientError`` + non-transient short-circuit + exhausted retries) —
  delegated to :class:`foresightrlcore.serverless.helpers.bedrock_judge.BedrockJudge`.
* SageMaker RLVR wire-format adapters (``extract_response`` and
  ``extract_state`` across verl/OpenAI/Nova payload shapes) — now
  provided by :mod:`foresightrlcore.serverless.helpers.sample_io`.
* The per-sample evaluation glue (``_evaluate_sample``) including the
  format-fail and thread-exception fallback paths.
* ``lambda_handler`` dispatch (empty batch, single-sample, batched,
  body-string, thread-exception swallow).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest import mock

import pytest
from botocore.exceptions import ClientError

# Absent from OSS checkouts — skip the module rather than fail collection.
pytest.importorskip("foresightrl_intuit")

from foresightrl_intuit.tasks.advice import lambda_reward as _MODULE  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_client() -> None:
    """Clear the cached Bedrock judge between tests."""
    _MODULE._reset_bedrock_client()


# ---------------------------------------------------------------------------
# Helpers — mock Bedrock responses + judge payloads
# ---------------------------------------------------------------------------


def _judge_payload(
    *, safe: int = 1, perfect: bool = False, extra: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"safe": safe, "critique": "stub"}
    for crit in _MODULE.CRITERIA:
        payload[crit] = 1 if perfect else 0
    if extra:
        payload.update(extra)
    return payload


def _converse_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "output": {
            "message": {"content": [{"text": json.dumps(payload)}]},
        }
    }


def _throttling_error() -> ClientError:
    return ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
        "Converse",
    )


def _validation_error() -> ClientError:
    return ClientError(
        {"Error": {"Code": "ValidationException", "Message": "no"}},
        "Converse",
    )


def _stub_judge_client(monkeypatch: pytest.MonkeyPatch, client: mock.MagicMock) -> None:
    """Wire ``client`` into the singleton :class:`BedrockJudge`.

    The lambda now uses :class:`BedrockJudge` which caches a boto3
    ``bedrock-runtime`` client on its first ``_get_client()`` call. We
    seed the cache by hand and patch out ``time.sleep`` so retry loops
    don't actually wait.
    """
    judge = _MODULE._get_judge()
    object.__setattr__(judge, "_client", client)
    from foresightrlcore.serverless.helpers import bedrock_judge as _BJ

    monkeypatch.setattr(_BJ.time, "sleep", lambda _seconds: None)


# ---------------------------------------------------------------------------
# Bedrock transport (_invoke_judge_bedrock)
# ---------------------------------------------------------------------------


class TestInvokeJudgeBedrock:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = mock.MagicMock()
        client.converse.return_value = _converse_response(_judge_payload(perfect=True))
        _stub_judge_client(monkeypatch, client)

        score, breakdown = _MODULE._invoke_judge_bedrock(
            "BUSINESS STATE: $1M revenue",
            {"recommendation": "Renegotiate vendor X"},
        )
        assert score == 1.0
        assert breakdown["safe"] == 1
        client.converse.assert_called_once()

    def test_retries_on_throttling_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = mock.MagicMock()
        client.converse.side_effect = [
            _throttling_error(),
            _converse_response(_judge_payload(perfect=True)),
        ]
        _stub_judge_client(monkeypatch, client)

        score, _ = _MODULE._invoke_judge_bedrock("state", {"recommendation": "do x"})
        assert score == 1.0
        assert client.converse.call_count == 2

    def test_neutral_fallback_on_exhausted_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = mock.MagicMock()
        client.converse.side_effect = [_throttling_error()] * (
            _MODULE.JUDGE_MAX_RETRIES
        )
        _stub_judge_client(monkeypatch, client)

        score, breakdown = _MODULE._invoke_judge_bedrock(
            "state", {"recommendation": "do x"}
        )
        assert score == _MODULE.JUDGE_NEUTRAL_FALLBACK
        assert "judge_failed" in breakdown["error"]

    def test_does_not_retry_on_non_transient_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = mock.MagicMock()
        client.converse.side_effect = _validation_error()
        _stub_judge_client(monkeypatch, client)

        score, breakdown = _MODULE._invoke_judge_bedrock(
            "state", {"recommendation": "do x"}
        )
        assert score == _MODULE.JUDGE_NEUTRAL_FALLBACK
        assert client.converse.call_count == 1
        assert "judge_failed" in breakdown["error"]

    def test_unparseable_judge_output_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The judge returns non-JSON; ``_invoke_judge_bedrock`` catches
        # the ValueError directly (it doesn't bubble up to ``BedrockJudge``'s
        # retry loop), so we return neutral fallback after one call.
        client = mock.MagicMock()
        bad_response = {
            "output": {"message": {"content": [{"text": "not json at all"}]}}
        }
        client.converse.return_value = bad_response
        _stub_judge_client(monkeypatch, client)

        score, breakdown = _MODULE._invoke_judge_bedrock(
            "state", {"recommendation": "x"}
        )
        assert score == _MODULE.JUDGE_NEUTRAL_FALLBACK
        assert "judge_failed" in breakdown["error"]
        assert client.converse.call_count == 1


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------


def _make_sample(
    assistant_content: str,
    *,
    state: str = "BUSINESS STATE: revenue $1M",
    sample_id: str = "s-1",
) -> Dict[str, Any]:
    return {
        "id": sample_id,
        "messages": [
            {"role": "system", "content": "you are an advisor"},
            {"role": "user", "content": "<business_state>...</business_state>"},
            {"role": "assistant", "content": assistant_content},
        ],
        "reference_answer": {"text": state},
    }


def _metric(metrics: List[Dict[str, Any]], name: str) -> float:
    for m in metrics:
        if m["name"] == name:
            return float(m["value"])
    raise AssertionError(f"metric {name!r} not found")


class TestEvaluateSample:
    def test_format_fail_flat_minus_half(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 18k thinking chars -> would-be penalty 0.06; format gate must
        # NOT subtract it, reward is flat -0.5.
        long_thinking = "<think>" + "x" * 18_000 + "</think>"
        sample = _make_sample(long_thinking + "no recommendation here")
        bedrock = mock.MagicMock()
        _stub_judge_client(monkeypatch, bedrock)

        result = _MODULE._evaluate_sample(0, sample)

        assert result["aggregate_reward_score"] == _MODULE.FORMAT_FAIL_REWARD
        assert _metric(result["metrics_list"], "format_fail") == 1.0
        assert _metric(result["metrics_list"], "judge_called") == 0.0
        assert _metric(result["metrics_list"], "think_length") == 18_000
        bedrock.converse.assert_not_called()

    def test_success_path_subtracts_penalty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = mock.MagicMock()
        client.converse.return_value = _converse_response(_judge_payload(perfect=True))
        _stub_judge_client(monkeypatch, client)

        # 20k thinking chars -> penalty = 0.10
        rollout = (
            "<think>" + "x" * 20_000 + "</think>"
            '{"recommendation":"raise prices on Product A by 5%",'
            '"expected_impact":"~$50k","risks":"churn"}'
        )
        result = _MODULE._evaluate_sample(0, _make_sample(rollout))

        assert result["aggregate_reward_score"] == pytest.approx(0.9)
        assert _metric(result["metrics_list"], "judge_score") == 1.0
        assert _metric(result["metrics_list"], "think_penalty") == pytest.approx(0.10)
        assert _metric(result["metrics_list"], "format_fail") == 0.0

    def test_safe_gate_zeroes_judge_score(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        unsafe_payload = _judge_payload(safe=0, perfect=True)
        client = mock.MagicMock()
        client.converse.return_value = _converse_response(unsafe_payload)
        _stub_judge_client(monkeypatch, client)

        rollout = '{"recommendation":"sell the building"}'
        result = _MODULE._evaluate_sample(0, _make_sample(rollout))

        assert result["aggregate_reward_score"] == 0.0
        assert _metric(result["metrics_list"], "judge_score") == 0.0
        assert _metric(result["metrics_list"], "safe") == 0.0

    def test_per_criterion_metrics_in_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = mock.MagicMock()
        client.converse.return_value = _converse_response(_judge_payload(perfect=True))
        _stub_judge_client(monkeypatch, client)

        rollout = '{"recommendation":"x","expected_impact":"$","risks":"r"}'
        result = _MODULE._evaluate_sample(0, _make_sample(rollout))
        names = {m["name"] for m in result["metrics_list"]}
        for crit in _MODULE.CRITERIA:
            assert f"criterion_{crit}" in names

    def test_extracts_state_from_reward_model_ground_truth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: Dict[str, Any] = {}

        def fake_judge(state: str, advice: Dict[str, Any]) -> Any:
            captured["state"] = state
            return 0.8, {"safe": 1, "total_score": 8, "critique": ""}

        monkeypatch.setattr(_MODULE, "_invoke_judge_bedrock", fake_judge)

        sample = {
            "id": "s",
            "messages": [
                {"role": "user", "content": "prompt"},
                {"role": "assistant", "content": '{"recommendation":"x"}'},
            ],
            "reward_model": {"ground_truth": "VERL-STYLE STATE", "style": "model"},
        }
        _MODULE._evaluate_sample(0, sample)
        assert captured["state"] == "VERL-STYLE STATE"


# ---------------------------------------------------------------------------
# Verl-style invocation shapes
# ---------------------------------------------------------------------------


class TestVerlStyleExtraction:
    """Real SageMaker payload shapes observed in the broken Opus 4.5 run."""

    def test_extract_response_from_top_level_response(self) -> None:
        assert (
            _MODULE.extract_response({"response": "the model output"})
            == "the model output"
        )

    def test_extract_response_from_top_level_completion(self) -> None:
        assert (
            _MODULE.extract_response({"completion": "the model output"})
            == "the model output"
        )

    def test_extract_response_from_top_level_solution_str(self) -> None:
        assert (
            _MODULE.extract_response({"solution_str": "the model output"})
            == "the model output"
        )

    def test_extract_response_from_prompt_appended_assistant(self) -> None:
        sample = {
            "prompt": [
                {"role": "system", "content": "you are an advisor"},
                {"role": "user", "content": "what should I do?"},
                {"role": "assistant", "content": "raise prices"},
            ]
        }
        assert _MODULE.extract_response(sample) == "raise prices"

    def test_extract_response_top_level_wins_over_messages(self) -> None:
        sample = {
            "response": "TOP",
            "messages": [{"role": "assistant", "content": "FROM_MESSAGES"}],
        }
        assert _MODULE.extract_response(sample) == "TOP"

    def test_extract_state_uses_reward_model_ground_truth(self) -> None:
        sample = {
            "prompt": [{"role": "user", "content": "ignore me"}],
            "reward_model": {"ground_truth": "VERL STATE", "style": "model"},
        }
        assert _MODULE.extract_state(sample) == "VERL STATE"

    def test_evaluate_sample_verl_payload_e2e(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: a sample shaped like the actual SageMaker RLVR payload
        must reach the judge and produce a non-default reward.
        """
        captured: Dict[str, Any] = {}

        def fake_judge(state: str, advice: Dict[str, Any]) -> Any:
            captured["state"] = state
            captured["advice"] = advice
            return 0.7, {"safe": 1, "total_score": 7, "critique": "ok"}

        monkeypatch.setattr(_MODULE, "_invoke_judge_bedrock", fake_judge)

        sample = {
            "data_source": "advice",
            "prompt": [
                {"role": "system", "content": "you are an advisor"},
                {
                    "role": "user",
                    "content": (
                        "<business_state>masked state</business_state> "
                        "What should we do?"
                    ),
                },
            ],
            "ability": "generation",
            "reward_model": {
                "ground_truth": "unmasked: $1M revenue, $200k COGS",
                "style": "model",
            },
            "response": (
                "<think>" + "x" * 4_000 + "</think>"
                '{"recommendation":"renegotiate top vendor",'
                '"expected_impact":"$30k","risks":"supply"}'
            ),
        }
        result = _MODULE._evaluate_sample(0, sample)

        assert captured["state"] == "unmasked: $1M revenue, $200k COGS"
        assert captured["advice"]["recommendation"] == "renegotiate top vendor"
        assert result["aggregate_reward_score"] == pytest.approx(0.7)
        assert _metric(result["metrics_list"], "format_fail") == 0.0
        assert _metric(result["metrics_list"], "judge_called") == 1.0
        assert _metric(result["metrics_list"], "think_length") == 4_000


# ---------------------------------------------------------------------------
# Lambda handler dispatch
# ---------------------------------------------------------------------------


class TestLambdaHandler:
    def test_empty_batch_returns_400(self) -> None:
        resp = _MODULE.lambda_handler({}, None)
        assert resp["statusCode"] == 400

    def test_batch_dispatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = mock.MagicMock()
        client.converse.return_value = _converse_response(_judge_payload(perfect=True))
        _stub_judge_client(monkeypatch, client)

        batch = [
            _make_sample(
                '{"recommendation":"raise prices","expected_impact":"x","risks":"y"}',
                sample_id=f"s-{i}",
            )
            for i in range(3)
        ]
        resp = _MODULE.lambda_handler({"batch": batch}, None)

        assert resp["statusCode"] == 200
        results = json.loads(resp["body"])
        assert len(results) == 3
        for i, r in enumerate(results):
            assert r["id"] == f"s-{i}"
            assert r["aggregate_reward_score"] == pytest.approx(1.0)

    def test_thread_exception_uses_neutral_minus_penalty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 20k think chars -> would-be penalty 0.10
        sample = _make_sample(
            "<think>" + "x" * 20_000 + "</think>"
            '{"recommendation":"r","expected_impact":"i","risks":"k"}'
        )

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("boom")

        monkeypatch.setattr(_MODULE, "_evaluate_sample", boom)
        resp = _MODULE.lambda_handler({"batch": [sample]}, None)

        assert resp["statusCode"] == 200
        results = json.loads(resp["body"])
        # neutral 0.5 - penalty 0.10 = 0.40
        assert results[0]["aggregate_reward_score"] == pytest.approx(0.4)
        assert _metric(results[0]["metrics_list"], "think_penalty") == pytest.approx(
            0.10
        )

    def test_handles_body_string_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = mock.MagicMock()
        client.converse.return_value = _converse_response(_judge_payload(perfect=True))
        _stub_judge_client(monkeypatch, client)

        sample = _make_sample('{"recommendation":"r"}')
        body = json.dumps({"batch": [sample]})
        resp = _MODULE.lambda_handler({"body": body}, None)

        assert resp["statusCode"] == 200
        results = json.loads(resp["body"])
        assert len(results) == 1

    def test_accepts_single_sample_dict_no_batch_wrapper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SageMaker has been observed invoking the Lambda with the sample
        as the raw event (no ``batch`` wrapper). The handler must still
        process it.
        """
        client = mock.MagicMock()
        client.converse.return_value = _converse_response(_judge_payload(perfect=True))
        _stub_judge_client(monkeypatch, client)

        sample = {
            "prompt": [{"role": "user", "content": "q"}],
            "reward_model": {"ground_truth": "state", "style": "model"},
            "response": '{"recommendation":"do x","expected_impact":"$","risks":"r"}',
        }
        resp = _MODULE.lambda_handler(sample, None)
        assert resp["statusCode"] == 200
        results = json.loads(resp["body"])
        assert len(results) == 1
        assert results[0]["aggregate_reward_score"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Sanity: the Lambda imports the shared module successfully
# (it's bundled via custom_reward_function_extra_files at deploy time;
# locally pytest resolves through the foresightrlcore package path).
# ---------------------------------------------------------------------------


class TestSharedModuleWiring:
    def test_constants_are_the_canonical_module_constants(self) -> None:
        from foresightrl_intuit.tasks.advice.judge_logic import (
            CRITERIA as CANONICAL_CRITERIA,
        )
        from foresightrl_intuit.tasks.advice.judge_logic import (
            FORMAT_FAIL_REWARD as CANONICAL_FORMAT_FAIL,
        )
        from foresightrl_intuit.tasks.advice.judge_logic import (
            JUDGE_NEUTRAL_FALLBACK as CANONICAL_NEUTRAL,
        )

        assert _MODULE.CRITERIA is CANONICAL_CRITERIA
        assert _MODULE.FORMAT_FAIL_REWARD == CANONICAL_FORMAT_FAIL
        assert _MODULE.JUDGE_NEUTRAL_FALLBACK == CANONICAL_NEUTRAL
