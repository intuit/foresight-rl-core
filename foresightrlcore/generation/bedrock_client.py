"""Bedrock generation client via ``bedrock-runtime.converse(...)``.

Reuses :class:`foresightrlcore.clients.bedrock.BedrockClientManager` for the
cached boto3 client + ``force_refresh`` on errors. Unlike the judge path (which
uses forced tool-use for structured output), plain generation calls ``converse``
with a maxTokens inference config and reads token usage from the response.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, Tuple

from foresightrlcore.common.retry import call_with_retry
from foresightrlcore.generation.base import approx_tokens
from foresightrlcore.generation.interfaces import CallUsage

logger = logging.getLogger(__name__)


def _text_from_converse(response: dict) -> str:
    from foresightrlcore.clients.bedrock import text_from_converse

    return text_from_converse(response)


def _usage_from_converse(response: dict, prompt: str, text: str) -> Tuple[int, int]:
    """Reported token counts, falling back to the ~4-chars/token estimate."""
    from foresightrlcore.clients.bedrock import token_usage_from_converse

    prompt_tokens, completion_tokens = token_usage_from_converse(response)
    if prompt_tokens == 0 and completion_tokens == 0:
        return approx_tokens(prompt), approx_tokens(text)
    return prompt_tokens, completion_tokens


class BedrockGenerationClient:
    """Synchronous generation client over Bedrock Converse."""

    def __init__(
        self,
        model_id: str,
        *,
        region: Optional[str] = None,
        temperature: Optional[float] = 0.0,
        max_retries: int = 10,
        base_delay: float = 3.0,
    ) -> None:
        from foresightrlcore.clients.bedrock import BedrockClientManager

        self._manager = BedrockClientManager(
            model_id=model_id,
            region=region,
            # Generation builds its own inferenceConfig below. The shared
            # manager's temperature is only used for structured judge calls.
            temperature=temperature if temperature is not None else 0.0,
        )
        self._temperature = temperature
        self._max_retries = max_retries
        self._base_delay = base_delay

    def generate(
        self, prompt: str, *, tag: str, max_tokens: int
    ) -> Tuple[str, CallUsage]:
        def _once() -> Tuple[str, CallUsage]:
            chat = self._manager.get()
            t0 = time.monotonic()
            inference_config: dict[str, Any] = {"maxTokens": max_tokens}
            if self._temperature is not None:
                inference_config["temperature"] = self._temperature
            response = chat.client.converse(
                modelId=chat.model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig=inference_config,
            )
            latency_ms = (time.monotonic() - t0) * 1000.0
            text = _text_from_converse(response)
            prompt_tokens, completion_tokens = _usage_from_converse(
                response, prompt, text
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
            description=f"bedrock-generation[{tag}]",
        )

    def close(self) -> None:
        return None


__all__ = ["BedrockGenerationClient"]
