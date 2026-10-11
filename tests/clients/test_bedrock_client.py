"""Tests for the Bedrock LLM client shim and BedrockClientManager.

These tests mock ``boto3.client`` selectively for the cases that need
to exercise Converse-call behavior. We do NOT stub ``boto3`` at module
import time — boto3 is a hard dependency of the [aws] extra and is
always present in the test environment, and stubbing it in
``sys.modules`` would poison downstream tests that exercise the real
``sagemaker`` SDK.
"""

from __future__ import annotations

import os
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from foresightrlcore.clients.bedrock import (
    BedrockClientManager,
    _BedrockChat,
    _BedrockStructured,
    _extract_tool_args,
    _pydantic_to_tool_spec,
    _split_messages,
    get_bedrock_llm_client,
)

# ---------------------------------------------------------------------------
# Schemas used to exercise schema-agnostic behavior of the shim
# ---------------------------------------------------------------------------


class _SmallSchema(BaseModel):
    name: str = Field(description="A name")
    count: int = Field(description="A count")


class _NestedItem(BaseModel):
    label: str
    score: float


class _NestedSchema(BaseModel):
    items: List[_NestedItem]


# ---------------------------------------------------------------------------
# _split_messages
# ---------------------------------------------------------------------------


class TestSplitMessages:
    def test_system_messages_separated(self):
        system, conv = _split_messages(
            [
                {"role": "system", "content": "sys-1"},
                {"role": "user", "content": "u-1"},
                {"role": "assistant", "content": "a-1"},
            ]
        )
        assert system == [{"text": "sys-1"}]
        assert conv == [
            {"role": "user", "content": [{"text": "u-1"}]},
            {"role": "assistant", "content": [{"text": "a-1"}]},
        ]

    def test_no_system_messages(self):
        system, conv = _split_messages([{"role": "user", "content": "hi"}])
        assert system == []
        assert conv == [{"role": "user", "content": [{"text": "hi"}]}]

    def test_unknown_role_becomes_user(self):
        _system, conv = _split_messages([{"role": "tool", "content": "x"}])
        assert conv[0]["role"] == "user"

    def test_non_string_content_serialized(self):
        system, conv = _split_messages([{"role": "user", "content": {"k": "v"}}])
        assert system == []
        # JSON-serialized payload survives as the text part.
        assert '"k"' in conv[0]["content"][0]["text"]


# ---------------------------------------------------------------------------
# _pydantic_to_tool_spec — schema-agnostic
# ---------------------------------------------------------------------------


class TestPydanticToToolSpec:
    def test_flat_schema(self):
        spec = _pydantic_to_tool_spec(_SmallSchema)
        assert "toolSpec" in spec
        ts = spec["toolSpec"]
        assert ts["name"] == "emit_structured_response"
        schema = ts["inputSchema"]["json"]
        assert "name" in schema["properties"]
        assert "count" in schema["properties"]

    def test_nested_schema(self):
        spec = _pydantic_to_tool_spec(_NestedSchema)
        schema = spec["toolSpec"]["inputSchema"]["json"]
        # Nested schemas should reference items via $defs / definitions.
        assert "items" in schema["properties"]


# ---------------------------------------------------------------------------
# _extract_tool_args
# ---------------------------------------------------------------------------


class TestExtractToolArgs:
    def test_extracts_dict_input(self):
        response = {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "name": "emit_structured_response",
                                "input": {"name": "n", "count": 3},
                            }
                        }
                    ]
                }
            }
        }
        args = _extract_tool_args(response)
        assert args == {"name": "n", "count": 3}

    def test_skips_non_tool_parts(self):
        response = {
            "output": {
                "message": {
                    "content": [
                        {"text": "ignored chatter"},
                        {
                            "toolUse": {"input": {"name": "n", "count": 3}},
                        },
                    ]
                }
            }
        }
        assert _extract_tool_args(response) == {"name": "n", "count": 3}

    def test_string_input_is_json_decoded(self):
        response = {
            "output": {
                "message": {
                    "content": [{"toolUse": {"input": '{"name": "n", "count": 1}'}}]
                }
            }
        }
        assert _extract_tool_args(response) == {"name": "n", "count": 1}

    def test_no_tool_use_raises(self):
        response = {"output": {"message": {"content": [{"text": "no tool"}]}}}
        with pytest.raises(ValueError, match="toolUse"):
            _extract_tool_args(response)


# ---------------------------------------------------------------------------
# _BedrockStructured.invoke
# ---------------------------------------------------------------------------


def _converse_response_for(payload: dict) -> dict:
    return {"output": {"message": {"content": [{"toolUse": {"input": payload}}]}}}


class TestBedrockStructuredInvoke:
    def test_returns_validated_schema(self):
        mock_client = MagicMock()
        mock_client.converse.return_value = _converse_response_for(
            {"name": "alpha", "count": 7}
        )
        chat = _BedrockChat(client=mock_client, model_id="m", temperature=0.0)
        structured = chat.with_structured_output(_SmallSchema)
        result = structured.invoke(
            [
                {"role": "system", "content": "be precise"},
                {"role": "user", "content": "score this"},
            ]
        )
        assert isinstance(result, _SmallSchema)
        assert result.name == "alpha"
        assert result.count == 7

        # System block was sent separately, not in the conversation.
        call_kwargs = mock_client.converse.call_args.kwargs
        assert call_kwargs["system"] == [{"text": "be precise"}]
        assert call_kwargs["messages"][0]["role"] == "user"
        # toolChoice forces the structured tool.
        tool_choice = call_kwargs["toolConfig"]["toolChoice"]
        assert tool_choice == {"tool": {"name": "emit_structured_response"}}
        # Temperature flows through.
        assert call_kwargs["inferenceConfig"]["temperature"] == 0.0

    def test_works_with_nested_schema(self):
        mock_client = MagicMock()
        mock_client.converse.return_value = _converse_response_for(
            {"items": [{"label": "x", "score": 0.5}]}
        )
        chat = _BedrockChat(client=mock_client, model_id="m", temperature=0.2)
        structured = chat.with_structured_output(_NestedSchema)
        result = structured.invoke([{"role": "user", "content": "go"}])
        assert isinstance(result, _NestedSchema)
        assert result.items[0].label == "x"
        assert result.items[0].score == 0.5

    def test_no_system_messages_omits_system_kwarg(self):
        mock_client = MagicMock()
        mock_client.converse.return_value = _converse_response_for(
            {"name": "n", "count": 1}
        )
        chat = _BedrockChat(client=mock_client, model_id="m", temperature=0.0)
        chat.with_structured_output(_SmallSchema).invoke(
            [{"role": "user", "content": "hi"}]
        )
        call_kwargs = mock_client.converse.call_args.kwargs
        assert "system" not in call_kwargs


# ---------------------------------------------------------------------------
# get_bedrock_llm_client
# ---------------------------------------------------------------------------


class TestGetBedrockLlmClient:
    def test_uses_explicit_region(self, monkeypatch):
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        with patch("boto3.client") as boto_client:
            boto_client.return_value = MagicMock()
            chat = get_bedrock_llm_client("m", region="us-east-1")
        assert boto_client.call_args.args == ("bedrock-runtime",)
        assert boto_client.call_args.kwargs["region_name"] == "us-east-1"
        assert chat.model_id == "m"

    def test_falls_back_to_aws_default_region(self, monkeypatch):
        monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
        with patch("boto3.client") as boto_client:
            boto_client.return_value = MagicMock()
            get_bedrock_llm_client("m")
        assert boto_client.call_args.kwargs["region_name"] == "eu-west-1"

    def test_no_region_raises(self, monkeypatch):
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        # No `region=` kwarg and no env var → clear error.
        with patch("boto3.client"):
            with pytest.raises(RuntimeError, match="region"):
                get_bedrock_llm_client("m")

    def test_sets_generous_read_timeout_config(self, monkeypatch):
        """A botocore Config with a 120s read timeout must be passed so long
        ``converse`` generations don't hit the 60s default read timeout."""
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
        with patch("boto3.client") as boto_client:
            boto_client.return_value = MagicMock()
            get_bedrock_llm_client("m")
        boto_config = boto_client.call_args.kwargs["config"]
        assert boto_config.read_timeout == 120
        assert boto_config.connect_timeout == 10
        assert boto_config.retries == {"max_attempts": 3, "mode": "adaptive"}


# ---------------------------------------------------------------------------
# BedrockClientManager
# ---------------------------------------------------------------------------


class TestBedrockClientManager:
    def test_get_creates_once(self, monkeypatch):
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
        with patch("boto3.client") as boto_client:
            boto_client.return_value = MagicMock()
            mgr = BedrockClientManager(model_id="m")
            c1 = mgr.get()
            c2 = mgr.get()
        assert c1 is c2
        assert boto_client.call_count == 1

    def test_force_refresh_creates_new_client(self, monkeypatch):
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
        with patch("boto3.client") as boto_client:
            boto_client.side_effect = [MagicMock(), MagicMock()]
            mgr = BedrockClientManager(model_id="m")
            c1 = mgr.get()
            c2 = mgr.force_refresh()
        assert c1 is not c2
        assert boto_client.call_count == 2

    def test_explicit_region_overrides_env(self, monkeypatch):
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
        with patch("boto3.client") as boto_client:
            boto_client.return_value = MagicMock()
            BedrockClientManager(model_id="m", region="us-east-1").get()
        assert boto_client.call_count == 1
        assert boto_client.call_args.args == ("bedrock-runtime",)
        assert boto_client.call_args.kwargs["region_name"] == "us-east-1"

    def test_temperature_flows_into_client(self, monkeypatch):
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
        with patch("boto3.client") as boto_client:
            boto_client.return_value = MagicMock()
            chat = BedrockClientManager(model_id="m", temperature=0.7).get()
        assert chat.temperature == 0.7
