"""Minimal OpenAI chat client over an :class:`InferenceEndpoint`.

Stdlib only -- ``urllib`` rather than the ``openai`` package, because the
evaluate container is deliberately thin and this needs exactly one endpoint.

Everything in the package that consumes an endpoint goes through here, so
in-package benchmarks exercise the very same HTTP path an external benchmark
uses. A provider-specific shortcut for in-process callers would mean the
contract is only tested from the outside.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from foresightrlcore.inference.interfaces import (
    ChatMessage,
    ChatResult,
    InferenceEndpoint,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 300


class EndpointError(RuntimeError):
    """The endpoint returned an error or an unusable body."""


class ChatClient:
    """Issues chat completions against one endpoint.

    Args:
        endpoint: The endpoint to call.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        endpoint: InferenceEndpoint,
        *,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout

    @property
    def url(self) -> str:
        return f"{self.endpoint.base_url.rstrip('/')}/chat/completions"

    def chat(
        self,
        messages: List[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        """Run one completion.

        Sampling parameters are sent only when set, so the provider's own
        defaults stay in force for anything the caller did not ask about.
        ``top_k`` and ``repetition_penalty`` are not in the OpenAI schema; they
        ride at the top level of the body, which is where vLLM reads them and
        where the proxy looks for them before handing them to an API provider as
        an extension.

        Raises:
            EndpointError: On a transport error, a non-2xx response, or a body
                that is not a usable chat completion. Callers wrap this in the
                shared retry helper.
        """
        payload: Dict[str, Any] = {
            "model": self.endpoint.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k
        if stop:
            payload["stop"] = stop
        if repetition_penalty is not None:
            payload["repetition_penalty"] = repetition_penalty
        # Last, so a caller can override any of the above when a provider needs
        # a field spelled differently.
        payload.update(extra or {})

        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.endpoint.api_key}",
            },
            method="POST",
        )

        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise EndpointError(f"HTTP {exc.code} from endpoint: {detail}") from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise EndpointError(f"could not reach endpoint: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise EndpointError(f"endpoint returned invalid JSON: {exc}") from exc

        latency_ms = (time.time() - started) * 1000.0
        return _to_result(body, latency_ms)


def _to_result(body: Dict[str, Any], latency_ms: float) -> ChatResult:
    if isinstance(body.get("error"), dict):
        raise EndpointError(str(body["error"].get("message", body["error"])))

    choices = body.get("choices") or []
    if not choices:
        raise EndpointError("endpoint response had no choices")

    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content")
    usage = body.get("usage") or {}

    return ChatResult(
        text=content if isinstance(content, str) else "",
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
        latency_ms=latency_ms,
        finish_reason=str(choice.get("finish_reason") or "stop"),
    )


__all__ = ["DEFAULT_TIMEOUT_SECONDS", "ChatClient", "EndpointError"]
