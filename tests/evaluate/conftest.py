"""Fixtures for the evaluate suite.

Everything here runs offline: the fake endpoint is a plain object and the fake
chat backend answers from a lookup table, so no test needs a model, a GPU, or
a network.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from foresightrlcore.inference.interfaces import ChatResult, InferenceEndpoint


@pytest.fixture(autouse=True)
def _registries_loaded() -> None:
    """Import the built-in axes once so every test sees a full registry."""
    from foresightrlcore.plugins import load_evaluate_plugins

    load_evaluate_plugins()


@pytest.fixture
def endpoint() -> InferenceEndpoint:
    return InferenceEndpoint(
        base_url="http://127.0.0.1:9/v1",
        api_key="test-key",
        model_id="test-model",
        model_name="test-model",
    )


class FakeChatClient:
    """Stands in for :class:`~foresightrlcore.inference.client.ChatClient`.

    Args:
        replies: Text to return, keyed by a substring of the prompt. The
            ``"*"`` key is the fallback.
        fail_on: Prompt substrings that should raise instead of answering.
        finish_reason: Reported on every reply. ``"length"`` simulates a
            generation cut off at the token budget.
    """

    def __init__(
        self,
        replies: Optional[Dict[str, str]] = None,
        *,
        fail_on: Optional[List[str]] = None,
        finish_reason: str = "stop",
    ) -> None:
        self.replies = replies or {"*": "A"}
        self.fail_on = fail_on or []
        self.finish_reason = finish_reason
        self.calls: List[List[Dict[str, Any]]] = []
        self.call_kwargs: List[Dict[str, Any]] = []

    def chat(self, messages: List[Dict[str, Any]], **kwargs: Any) -> ChatResult:
        self.calls.append(messages)
        self.call_kwargs.append(kwargs)
        prompt = " ".join(str(m.get("content", "")) for m in messages)

        for marker in self.fail_on:
            if marker in prompt:
                raise RuntimeError(f"endpoint refused: {marker}")

        for marker, reply in self.replies.items():
            if marker != "*" and marker in prompt:
                return ChatResult(
                    text=reply,
                    prompt_tokens=10,
                    completion_tokens=5,
                    finish_reason=self.finish_reason,
                )
        return ChatResult(
            text=self.replies.get("*", ""),
            prompt_tokens=10,
            completion_tokens=5,
            finish_reason=self.finish_reason,
        )
