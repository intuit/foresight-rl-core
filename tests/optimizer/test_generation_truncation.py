"""Tests for the optional prompt-truncation knob (``truncate_prompt_tokens``).

The knob is OFF by default and, when set, must flow from the model config
through the factory into the local vLLM client and be forwarded to the
OpenAI-compatible endpoint as ``extra_body={"truncate_prompt_tokens": N}``.
API providers (genos/bedrock) ignore it.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Absent from OSS checkouts — skip the module rather than fail collection.
pytest.importorskip("foresightrl_intuit")

from foresightrl_intuit.optimizer.generation.genos_client import (  # noqa: E402
    LangChainChatGenerationClient,
)

from foresightrlcore.config.optimizer_schema import OptimizerModelConfig
from foresightrlcore.config.perf_schema import VllmServingConfig
from foresightrlcore.generation.factory import make_generation_client
from foresightrlcore.generation.local_vllm_client import (
    LocalVllmGenerationClient,
)


def _fake_manager(message):
    """Return (manager, client) where client.bind(...).invoke(...) -> message."""
    bound = MagicMock()
    bound.invoke.return_value = message
    client = MagicMock()
    client.bind.return_value = bound
    manager = MagicMock()
    manager.get.return_value = client
    return manager, client


def _msg(text: str = "hi"):
    return SimpleNamespace(
        content=text, usage_metadata={"input_tokens": 3, "output_tokens": 2}
    )


class TestTruncatePromptTokensConfig:
    def test_default_is_off(self):
        m = OptimizerModelConfig(name="x", provider="genos")
        assert m.truncate_prompt_tokens is None

    def test_accepts_positive_int(self):
        m = OptimizerModelConfig(
            name="Qwen/x", provider="local", truncate_prompt_tokens=6000
        )
        assert m.truncate_prompt_tokens == 6000

    def test_rejects_below_one(self):
        with pytest.raises(Exception):
            OptimizerModelConfig(
                name="Qwen/x", provider="local", truncate_prompt_tokens=0
            )


class TestLangChainExtraBody:
    def test_no_extra_body_binds_only_max_tokens(self):
        manager, client = _fake_manager(_msg())
        gen = LangChainChatGenerationClient(manager)
        gen.generate("prompt", tag="t", max_tokens=100)
        client.bind.assert_called_once_with(max_tokens=100)

    def test_extra_body_is_forwarded_to_bind(self):
        manager, client = _fake_manager(_msg())
        gen = LangChainChatGenerationClient(
            manager, extra_body={"truncate_prompt_tokens": 6000}
        )
        gen.generate("prompt", tag="t", max_tokens=100)
        client.bind.assert_called_once_with(
            max_tokens=100, extra_body={"truncate_prompt_tokens": 6000}
        )


class TestLocalVllmTruncation:
    def test_truncate_sets_delegate_extra_body(self):
        client = LocalVllmGenerationClient(
            "Qwen/Qwen3.5-0.8B", VllmServingConfig(), truncate_prompt_tokens=6000
        )
        assert client.truncate_prompt_tokens == 6000
        assert client._delegate._extra_body == {"truncate_prompt_tokens": 6000}

    def test_default_leaves_extra_body_none(self):
        client = LocalVllmGenerationClient("Qwen/Qwen3.5-0.8B", VllmServingConfig())
        assert client.truncate_prompt_tokens is None
        assert client._delegate._extra_body is None


class TestFactoryThreadsTruncation:
    def test_local_client_receives_truncate(self):
        model = OptimizerModelConfig(
            name="Qwen/Qwen3.5-0.8B", provider="local", truncate_prompt_tokens=6000
        )
        client = make_generation_client(model)
        assert isinstance(client, LocalVllmGenerationClient)
        assert client.truncate_prompt_tokens == 6000
        assert client._delegate._extra_body == {"truncate_prompt_tokens": 6000}

    def test_local_client_default_no_truncate(self):
        model = OptimizerModelConfig(name="Qwen/Qwen3.5-0.8B", provider="local")
        client = make_generation_client(model)
        assert isinstance(client, LocalVllmGenerationClient)
        assert client.truncate_prompt_tokens is None
        assert client._delegate._extra_body is None
