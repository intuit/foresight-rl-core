"""Contracts for the inference substrate.

The substrate's whole job is to turn "a model, on some provider" into "an
OpenAI-compatible ``base_url``". Benchmarks -- in-process or an external
subprocess -- only ever see :class:`InferenceEndpoint`.

Stdlib only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

# A chat message in OpenAI wire form: ``{"role": ..., "content": ...}``.
ChatMessage = Dict[str, Any]


@dataclass(frozen=True)
class InferenceEndpoint:
    """A live OpenAI-compatible endpoint for exactly one model.

    Args:
        base_url: Endpoint root **including** the ``/v1`` suffix.
        api_key: Always non-empty; a placeholder when the endpoint needs no auth
            (clients routinely refuse to start without one).
        model_id: The id to send in the request body.
        model_name: Logical candidate name used in reports.
    """

    base_url: str
    api_key: str
    model_id: str
    model_name: str

    def env(self) -> Dict[str, str]:
        """Return the environment variables an external benchmark reads.

        ``ANTHROPIC_*`` mirrors ``OPENAI_*`` so agent CLIs that follow the
        Anthropic env convention hit the same endpoint without a shim. These
        names are part of the published extension contract -- see
        ``docs/evaluate-extension-contract.md``.
        """
        return {
            "OPENAI_BASE_URL": self.base_url,
            "OPENAI_API_KEY": self.api_key,
            "ANTHROPIC_BASE_URL": self.base_url,
            "ANTHROPIC_API_KEY": self.api_key,
            "EVALUATE_MODEL_ID": self.model_id,
            "EVALUATE_MODEL_NAME": self.model_name,
        }


@dataclass
class ChatResult:
    """One completion produced by a :class:`ChatBackend`."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)


@runtime_checkable
class ChatBackend(Protocol):
    """Turns a chat request into a completion.

    This is the seam the OpenAI-compatible proxy sits on: implement it for a
    provider and that provider gains an endpoint for free.
    """

    def chat(
        self,
        messages: List[ChatMessage],
        *,
        max_tokens: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        """Run one non-streaming chat completion.

        A backend is free to ignore a sampling parameter its provider does not
        expose; the proxy forwards whatever the caller sent and does not police
        which provider understands what.
        """
        ...


@runtime_checkable
class EndpointHandle(Protocol):
    """A started endpoint that must be stopped again.

    :func:`foresightrlcore.inference.endpoint_for` brackets these, so callers
    normally never touch ``start``/``stop`` directly.
    """

    def start(self) -> InferenceEndpoint:
        """Bring the endpoint up and return it once it is healthy."""
        ...

    def stop(self) -> None:
        """Tear the endpoint down. Must be safe to call more than once."""
        ...


__all__ = [
    "ChatBackend",
    "ChatMessage",
    "ChatResult",
    "EndpointHandle",
    "InferenceEndpoint",
]
