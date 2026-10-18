"""Chat backends the OpenAI-compatible proxy serves.

:class:`ManagerChatBackend` is the only backend the core needs: it dispatches
through :func:`~foresightrlcore.clients.llm.resolve_llm_manager`, so it speaks
to *whatever* providers are registered. Bedrock ships in the OSS core; Genos
arrives with ``foresightrl-intuit``. Registering a new LLM provider gives that
provider an inference endpoint with no change here.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from foresightrlcore.common.retry import call_with_retry
from foresightrlcore.generation.base import (
    text_from_langchain_message,
    usage_from_langchain_message,
)
from foresightrlcore.inference.interfaces import ChatMessage, ChatResult

logger = logging.getLogger(__name__)


def build_llm_client_config(
    provider: str,
    model_id: str,
    *,
    region: Optional[str] = None,
    temperature: Optional[float] = 0.0,
) -> Any:
    """Build an :class:`LLMClientConfig` for an API provider.

    Lets a caller name a provider and a model id in one line instead of nesting
    the provider's own sub-block, which is all an evaluate model config carries.

    Raises:
        ValueError: If ``provider`` is not an API provider with an endpoint.
    """
    from foresightrlcore.config.schema import (
        BedrockClientConfig,
        GenosClientConfig,
        LLMClientConfig,
    )

    # Endpoint client schemas still require a numeric default; nullable
    # optimizer temperatures are handled by the generation clients instead.
    if temperature is None:
        raise ValueError(
            "API endpoint client configuration requires a numeric temperature"
        )

    if provider == "bedrock":
        return LLMClientConfig(
            provider="bedrock",
            bedrock=BedrockClientConfig(
                model_id=model_id, region=region, temperature=temperature
            ),
        )
    if provider == "genos":
        return LLMClientConfig(
            provider="genos",
            genos=GenosClientConfig(model_id=model_id, temperature=temperature),
        )
    raise ValueError(
        f"provider {provider!r} has no API-backed chat client; "
        "expected 'bedrock' or 'genos'"
    )


class ManagerChatBackend:
    """A :class:`~foresightrlcore.inference.interfaces.ChatBackend` over an
    LLM client manager.

    The manager is resolved lazily on first use so constructing the backend
    never needs credentials -- which keeps config validation and unit tests
    offline.

    Args:
        client_config: An ``LLMClientConfig`` selecting provider + model.
        max_retries: Attempts per request (auth-refresh aware).
        base_delay: Base seconds for exponential backoff.
    """

    def __init__(
        self,
        client_config: Any,
        *,
        max_retries: int = 6,
        base_delay: float = 2.0,
    ) -> None:
        self._client_config = client_config
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._manager: Optional[Any] = None

    @property
    def manager(self) -> Any:
        if self._manager is None:
            from foresightrlcore.clients.llm import resolve_llm_manager

            self._manager = resolve_llm_manager(self._client_config)
        return self._manager

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
        def _once() -> ChatResult:
            client = self.manager.get()
            bind_kwargs: Dict[str, Any] = {"max_tokens": max_tokens}
            if temperature is not None:
                bind_kwargs["temperature"] = temperature
            if top_p is not None:
                bind_kwargs["top_p"] = top_p
            if stop:
                bind_kwargs["stop"] = stop
            # Neither top_k nor repetition_penalty is a parameter of either
            # client's signature -- Bedrock takes them in
            # additionalModelRequestFields and ChatOpenAI in extra_body -- so
            # they go through the extension channel both already have. A model
            # that does not support one ignores it.
            extension = dict(extra or {})
            if top_k is not None:
                extension.setdefault("top_k", top_k)
            if repetition_penalty is not None:
                extension.setdefault("repetition_penalty", repetition_penalty)
            if extension:
                bind_kwargs["extra_body"] = extension
            bound = client.bind(**bind_kwargs)

            t0 = time.monotonic()
            message = bound.invoke(messages)
            latency_ms = (time.monotonic() - t0) * 1000.0

            text = text_from_langchain_message(message)
            prompt_text = "\n".join(str(m.get("content", "")) for m in messages)
            prompt_tokens, completion_tokens = usage_from_langchain_message(
                message, prompt_text, text
            )
            return ChatResult(
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
            )

        return call_with_retry(
            _once,
            manager=self.manager,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            description="proxy.chat",
        )


class EchoChatBackend:
    """Deterministic offline backend used by tests and ``--dry-run``.

    Echoes the last user message with a stable prefix, so the proxy, the
    subprocess plumbing, and a benchmark's own parsing can all be exercised
    without a network or a GPU.
    """

    def __init__(self, model_name: str = "echo", reply: Optional[str] = None) -> None:
        self.model_name = model_name
        self.reply = reply

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
        # Sampling is accepted and ignored: echoing is deterministic, and the
        # signature has to match the protocol so this stays a usable stand-in.
        if self.reply is not None:
            text = self.reply
        else:
            last = ""
            for message in reversed(messages):
                if message.get("role") == "user":
                    last = str(message.get("content", ""))
                    break
            text = f"[echo:{self.model_name}] {last}"
        return ChatResult(
            text=text,
            prompt_tokens=sum(len(str(m.get("content", ""))) // 4 for m in messages),
            completion_tokens=max(1, len(text) // 4),
            latency_ms=0.1,
        )


__all__ = ["ManagerChatBackend", "EchoChatBackend", "build_llm_client_config"]
