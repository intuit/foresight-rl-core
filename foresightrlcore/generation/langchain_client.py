"""Shared LangChain ChatOpenAI generation client wrapper."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

from foresightrlcore.common.retry import call_with_retry
from foresightrlcore.generation.base import (
    text_from_langchain_message,
    usage_from_langchain_message,
)
from foresightrlcore.generation.interfaces import CallUsage

logger = logging.getLogger(__name__)


class LangChainChatGenerationClient:
    """A synchronous generation client over a ``ChatOpenAI``-returning manager."""

    def __init__(
        self,
        manager: Any,
        *,
        max_retries: int = 10,
        base_delay: float = 3.0,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._manager = manager
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._extra_body = extra_body

    def generate(
        self, prompt: str, *, tag: str, max_tokens: int
    ) -> Tuple[str, CallUsage]:
        def _once() -> Tuple[str, CallUsage]:
            client = self._manager.get()
            bind_kwargs: Dict[str, Any] = {"max_tokens": max_tokens}
            if self._extra_body:
                bind_kwargs["extra_body"] = self._extra_body
            bound = client.bind(**bind_kwargs)
            t0 = time.monotonic()
            message = bound.invoke([{"role": "user", "content": prompt}])
            latency_ms = (time.monotonic() - t0) * 1000.0
            text = text_from_langchain_message(message)
            prompt_tokens, completion_tokens = usage_from_langchain_message(
                message, prompt, text
            )
            return text, CallUsage(
                tag=tag,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
            )

        return call_with_retry(
            _once,
            manager=self._manager,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            description=f"generation[{tag}]",
        )

    def close(self) -> None:
        return None


__all__ = ["LangChainChatGenerationClient"]
