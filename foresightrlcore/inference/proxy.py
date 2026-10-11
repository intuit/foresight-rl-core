"""A stdlib OpenAI-compatible HTTP proxy in front of any :class:`ChatBackend`.

This is what lets a benchmark that only knows how to talk to an OpenAI endpoint
run against Bedrock, Genos, or anything else the package can reach: point it at
``base_url`` and it cannot tell the difference.

Deliberately built on ``http.server`` -- no FastAPI, no uvicorn, no new
dependency for a server that exists to serve one process on loopback.

Supported routes (with or without the ``/v1`` prefix):

* ``POST /v1/chat/completions`` -- non-streaming and SSE streaming
* ``POST /v1/completions`` -- legacy text completions
* ``GET  /v1/models``
* ``GET  /health``
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple, cast

from foresightrlcore.inference.interfaces import (
    ChatBackend,
    ChatMessage,
    ChatResult,
    InferenceEndpoint,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 4096

# Any non-empty value works; clients refuse to start without a key even when the
# endpoint ignores it.
PLACEHOLDER_API_KEY = "foresightrlcore-local"

# Request fields the proxy interprets itself. Everything else in the body is
# provider-specific and handed on untouched, so an option with no field of its
# own -- `chat_template_kwargs` carrying `reasoning_effort`, say -- still reaches
# the provider instead of being dropped here in silence.
CONSUMED_REQUEST_FIELDS = frozenset(
    {
        "model",
        "messages",
        "prompt",
        "stream",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "stop",
        "top_p",
        "top_k",
        "repetition_penalty",
    }
)


def _provider_passthrough(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Request fields the proxy does not interpret, to forward to the backend."""
    return {k: v for k, v in payload.items() if k not in CONSUMED_REQUEST_FIELDS}


def _normalise_messages(raw: Any) -> List[ChatMessage]:
    """Coerce an OpenAI ``messages`` array into plain role/content dicts.

    Content parts (``[{"type": "text", "text": ...}]``) are flattened to text,
    which is what every backend here consumes.
    """
    messages: List[ChatMessage] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user"))
        content = item.get("content", "")
        if isinstance(content, list):
            parts = [
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") in (None, "text")
            ]
            content = "".join(parts)
        messages.append({"role": role, "content": content})
    return messages


def _completion_body(
    result: ChatResult, model: str, *, completion_id: str, created: int
) -> Dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        },
    }


class _ProxyHandler(BaseHTTPRequestHandler):
    """Request handler; reads its backend off the owning server."""

    protocol_version = "HTTP/1.1"

    # -- plumbing ---------------------------------------------------------

    @property
    def _server(self) -> "_ProxyServer":
        return cast("_ProxyServer", self.server)

    def log_message(self, format: str, *args: Any) -> None:
        """Route access logs to the package logger instead of stderr."""
        logger.debug("proxy %s - %s", self.address_string(), format % args)

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str, err_type: str) -> None:
        self._send_json(
            status, {"error": {"message": message, "type": err_type, "code": status}}
        )

    def _read_json(self) -> Optional[Dict[str, Any]]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            length = 0
        if length <= 0:
            return {}
        try:
            return cast(Dict[str, Any], json.loads(self.rfile.read(length)))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

    @staticmethod
    def _route(path: str) -> str:
        """Strip a query string and an optional ``/v1`` prefix."""
        route = path.split("?", 1)[0].rstrip("/") or "/"
        if route.startswith("/v1/"):
            route = route[3:]
        elif route == "/v1":
            route = "/"
        return route

    # -- routes -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        route = self._route(self.path)
        if route == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if route == "/models":
            model = self._server.model_id
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": model,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "foresightrlcore",
                        }
                    ],
                },
            )
            return
        self._send_error_json(404, f"unknown route: {self.path}", "not_found")

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        route = self._route(self.path)
        if route not in ("/chat/completions", "/completions"):
            self._send_error_json(404, f"unknown route: {self.path}", "not_found")
            return

        payload = self._read_json()
        if payload is None:
            self._send_error_json(400, "request body is not valid JSON", "invalid_json")
            return

        if route == "/completions":
            prompt = payload.get("prompt", "")
            if isinstance(prompt, list):
                prompt = "".join(str(p) for p in prompt)
            messages: List[ChatMessage] = [{"role": "user", "content": str(prompt)}]
        else:
            messages = _normalise_messages(payload.get("messages"))

        if not messages:
            self._send_error_json(400, "no messages supplied", "invalid_request_error")
            return

        max_tokens = int(
            payload.get("max_completion_tokens")
            or payload.get("max_tokens")
            or self._server.default_max_tokens
        )
        temperature = payload.get("temperature")
        top_p = payload.get("top_p")
        top_k = payload.get("top_k")
        repetition_penalty = payload.get("repetition_penalty")
        stop = payload.get("stop")
        if isinstance(stop, str):
            stop = [stop]

        try:
            result = self._server.backend.chat(
                messages,
                max_tokens=max_tokens,
                temperature=(float(temperature) if temperature is not None else None),
                top_p=(float(top_p) if top_p is not None else None),
                top_k=(int(top_k) if top_k is not None else None),
                repetition_penalty=(
                    float(repetition_penalty)
                    if repetition_penalty is not None
                    else None
                ),
                stop=stop,
                extra=_provider_passthrough(payload),
            )
        except Exception as exc:  # noqa: BLE001 - any provider error becomes 502
            logger.warning("proxy upstream call failed: %s", exc)
            self._send_error_json(502, str(exc), "upstream_error")
            return

        model = str(payload.get("model") or self._server.model_id)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if payload.get("stream"):
            self._send_stream(result, model, completion_id, created)
            return

        self._send_json(
            200,
            _completion_body(
                result, model, completion_id=completion_id, created=created
            ),
        )

    def _send_stream(
        self, result: ChatResult, model: str, completion_id: str, created: int
    ) -> None:
        """Emit the completed response as a minimal SSE stream.

        The backends are synchronous, so there is nothing to stream
        incrementally -- but emitting the standard chunk sequence keeps clients
        that only speak ``stream=true`` working.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def chunk(delta: Dict[str, Any], finish: Optional[str]) -> bytes:
            body = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(body)}\n\n".encode("utf-8")

        self.wfile.write(chunk({"role": "assistant", "content": result.text}, None))
        self.wfile.write(chunk({}, result.finish_reason))
        self.wfile.write(b"data: [DONE]\n\n")
        self.close_connection = True


class _ProxyServer(ThreadingHTTPServer):
    """Threading HTTP server carrying the backend for its handlers."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: Tuple[str, int],
        backend: ChatBackend,
        model_id: str,
        default_max_tokens: int,
    ) -> None:
        super().__init__(address, _ProxyHandler)
        self.backend = backend
        self.model_id = model_id
        self.default_max_tokens = default_max_tokens


class OpenAICompatProxy:
    """Serves a :class:`ChatBackend` as an OpenAI-compatible endpoint.

    Args:
        backend: The chat backend to serve.
        model_id: Model id advertised by ``/models`` and used when a request
            omits ``model``.
        model_name: Logical candidate name carried on the endpoint.
        host: Bind address. Loopback by default -- the endpoint exists for
            processes on this machine and must not be exposed further.
        port: Bind port; ``0`` picks a free one.
        api_key: Value handed to clients as ``OPENAI_API_KEY``.
        default_max_tokens: Used when a request omits ``max_tokens``.
    """

    def __init__(
        self,
        backend: ChatBackend,
        *,
        model_id: str,
        model_name: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 0,
        api_key: str = PLACEHOLDER_API_KEY,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.backend = backend
        self.model_id = model_id
        self.model_name = model_name or model_id
        self.host = host
        self.port = port
        self.api_key = api_key
        self.default_max_tokens = default_max_tokens
        self._server: Optional[_ProxyServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> InferenceEndpoint:
        """Bind, serve in a background thread, and return the endpoint."""
        if self._server is not None:
            return self.endpoint

        self._server = _ProxyServer(
            (self.host, self.port),
            self.backend,
            self.model_id,
            self.default_max_tokens,
        )
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"openai-proxy-{self.port}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "inference proxy for %r listening on %s", self.model_name, self.base_url
        )
        return self.endpoint

    def stop(self) -> None:
        """Shut the server down. Safe to call more than once."""
        if self._server is None:
            return
        server, thread = self._server, self._thread
        self._server, self._thread = None, None
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join(timeout=5)
        logger.info("inference proxy for %r stopped", self.model_name)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def endpoint(self) -> InferenceEndpoint:
        return InferenceEndpoint(
            base_url=self.base_url,
            api_key=self.api_key,
            model_id=self.model_id,
            model_name=self.model_name,
        )

    def __enter__(self) -> InferenceEndpoint:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "PLACEHOLDER_API_KEY",
    "OpenAICompatProxy",
]
