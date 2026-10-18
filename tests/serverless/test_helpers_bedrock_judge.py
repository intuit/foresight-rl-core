"""Tests for :class:`foresightrlcore.serverless.helpers.bedrock_judge.BedrockJudge`."""

from __future__ import annotations

from unittest import mock

import pytest
from botocore.exceptions import ClientError

from foresightrlcore.serverless.helpers import bedrock_judge as bj


def _client_with(side_effect):
    client = mock.MagicMock()
    if isinstance(side_effect, list):
        client.converse.side_effect = side_effect
    elif isinstance(side_effect, Exception):
        client.converse.side_effect = side_effect
    else:
        client.converse.return_value = side_effect
    return client


def _converse_ok(text: str = "ok"):
    return {"output": {"message": {"content": [{"text": text}]}}}


def _throttling():
    return ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "slow"}}, "Converse"
    )


def _validation():
    return ClientError(
        {"Error": {"Code": "ValidationException", "Message": "nope"}}, "Converse"
    )


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(bj.time, "sleep", lambda _seconds: None)


def _judge(client) -> bj.BedrockJudge:
    j = bj.BedrockJudge(model_id="m", region="us-west-2", max_retries=3)
    object.__setattr__(j, "_client", client)
    return j


class TestInvokeSuccess:
    def test_returns_text_on_success(self):
        j = _judge(_client_with(_converse_ok("hello")))
        assert j.invoke("sys", "user") == "hello"


class TestRetryPolicy:
    def test_retries_on_transient_then_succeeds(self):
        client = _client_with([_throttling(), _converse_ok("ok")])
        j = _judge(client)
        assert j.invoke("sys", "user") == "ok"
        assert client.converse.call_count == 2

    def test_no_retry_on_non_transient(self):
        client = _client_with(_validation())
        j = _judge(client)
        with pytest.raises(bj.JudgeError):
            j.invoke("sys", "user")
        assert client.converse.call_count == 1

    def test_exhausts_retries(self):
        client = _client_with([_throttling()] * 3)
        j = _judge(client)
        with pytest.raises(bj.JudgeError):
            j.invoke("sys", "user")
        assert client.converse.call_count == 3

    def test_empty_response_text_treated_as_failure(self):
        empty: dict = {"output": {"message": {"content": []}}}
        client = _client_with(empty)
        j = _judge(client)
        with pytest.raises(bj.JudgeError):
            j.invoke("sys", "user")


class TestExtractText:
    def test_concatenates_multiple_blocks(self):
        resp = {
            "output": {
                "message": {
                    "content": [
                        {"text": "hello "},
                        {"text": "world"},
                        {"foo": "ignore"},
                    ]
                }
            }
        }
        assert bj._extract_text_from_converse(resp) == "hello world"

    def test_empty_on_unexpected_shape(self):
        assert bj._extract_text_from_converse({}) == ""
        assert bj._extract_text_from_converse({"output": "wrong"}) == ""
