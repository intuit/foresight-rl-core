"""The OpenAI-compatible proxy, end to end over real HTTP on loopback.

These tests start the actual server and speak to it with the actual client,
because the whole value of this layer is that an outside process cannot tell
the difference. Mocking the transport would test nothing.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import pytest

from foresightrlcore.inference.backends import EchoChatBackend
from foresightrlcore.inference.client import ChatClient, EndpointError
from foresightrlcore.inference.interfaces import ChatMessage, ChatResult
from foresightrlcore.inference.proxy import PLACEHOLDER_API_KEY, OpenAICompatProxy


class RecordingBackend:
    """Captures what the proxy passed through, and can be told to fail."""

    def __init__(self, text: str = "hello", error: Optional[Exception] = None) -> None:
        self.text = text
        self.error = error
        self.seen: List[Dict[str, Any]] = []

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
        self.seen.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "stop": stop,
                "extra": extra,
            }
        )
        if self.error is not None:
            raise self.error
        return ChatResult(text=self.text, prompt_tokens=7, completion_tokens=3)


@pytest.fixture
def proxy():
    backend = RecordingBackend()
    server = OpenAICompatProxy(backend, model_id="test-model", model_name="candidate")
    endpoint = server.start()
    try:
        yield server, backend, endpoint
    finally:
        server.stop()


def _post(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def _get(url: str) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read())


# ---------------------------------------------------------------------------
# The wire contract
# ---------------------------------------------------------------------------


def test_chat_completions_returns_the_openai_response_shape(proxy):
    _, _, endpoint = proxy

    body = _post(
        f"{endpoint.base_url}/chat/completions",
        {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
    assert body["usage"]["total_tokens"] == 10


def test_the_models_route_lists_the_served_model(proxy):
    _, _, endpoint = proxy

    body = _get(f"{endpoint.base_url}/models")

    assert [m["id"] for m in body["data"]] == ["test-model"]


def test_health_answers_before_any_request(proxy):
    _, _, endpoint = proxy
    assert _get(f"{endpoint.base_url}/health")["status"] == "ok"


def test_routes_work_with_and_without_the_v1_prefix(proxy):
    server, _, _ = proxy
    root = f"http://{server.host}:{server.port}"

    assert _get(f"{root}/health")["status"] == "ok"
    assert _get(f"{root}/v1/models")["data"]


def test_request_parameters_reach_the_backend(proxy):
    _, backend, endpoint = proxy

    _post(
        f"{endpoint.base_url}/chat/completions",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 32,
            "temperature": 0.7,
            "stop": "END",
        },
    )

    call = backend.seen[0]
    assert call["max_tokens"] == 32
    assert call["temperature"] == 0.7
    assert call["stop"] == ["END"]  # a bare string is normalised to a list


def test_the_sampling_parameters_survive_the_round_trip(proxy):
    """``top_k`` is not in the OpenAI schema, so the proxy must carry it itself."""
    _, backend, endpoint = proxy

    _post(
        f"{endpoint.base_url}/chat/completions",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 0.95,
            "top_k": 64,
        },
    )

    call = backend.seen[0]
    assert call["top_p"] == 0.95
    assert call["top_k"] == 64


def test_sampling_parameters_a_caller_omitted_are_passed_on_as_unset(proxy):
    _, backend, endpoint = proxy

    _post(
        f"{endpoint.base_url}/chat/completions",
        {"messages": [{"role": "user", "content": "hi"}]},
    )

    call = backend.seen[0]
    assert call["top_p"] is None and call["top_k"] is None


def test_content_parts_are_flattened_to_text(proxy):
    _, backend, endpoint = proxy

    _post(
        f"{endpoint.base_url}/chat/completions",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "part one "},
                        {"type": "text", "text": "part two"},
                    ],
                }
            ]
        },
    )

    assert backend.seen[0]["messages"][0]["content"] == "part one part two"


def test_the_legacy_completions_route_is_accepted(proxy):
    _, backend, endpoint = proxy

    body = _post(f"{endpoint.base_url}/completions", {"prompt": "once upon"})

    assert body["choices"][0]["message"]["content"] == "hello"
    assert backend.seen[0]["messages"][0]["content"] == "once upon"


def test_streaming_emits_the_standard_chunk_sequence(proxy):
    _, _, endpoint = proxy
    request = urllib.request.Request(
        f"{endpoint.base_url}/chat/completions",
        data=json.dumps(
            {"messages": [{"role": "user", "content": "hi"}], "stream": True}
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=10) as response:
        raw = response.read().decode()

    assert raw.startswith("data: ")
    assert raw.rstrip().endswith("data: [DONE]")
    first = json.loads(raw.split("\n\n")[0][len("data: ") :])
    assert first["choices"][0]["delta"]["content"] == "hello"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_an_upstream_failure_becomes_a_502(proxy):
    _, backend, endpoint = proxy
    backend.error = RuntimeError("provider exploded")

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(
            f"{endpoint.base_url}/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}]},
        )

    assert excinfo.value.code == 502
    assert "provider exploded" in excinfo.value.read().decode()


@pytest.mark.parametrize(
    "payload, status",
    [({"messages": []}, 400)],
)
def test_a_request_without_messages_is_rejected(proxy, payload, status):
    _, _, endpoint = proxy

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(f"{endpoint.base_url}/chat/completions", payload)

    assert excinfo.value.code == status


def test_an_unknown_route_is_a_404(proxy):
    _, _, endpoint = proxy

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(f"{endpoint.base_url}/nope")

    assert excinfo.value.code == 404


# ---------------------------------------------------------------------------
# Lifecycle and the endpoint object
# ---------------------------------------------------------------------------


def test_the_endpoint_carries_the_documented_contract_variables(proxy):
    _, _, endpoint = proxy
    env = endpoint.env()

    assert env["OPENAI_BASE_URL"] == endpoint.base_url
    assert env["OPENAI_API_KEY"] == PLACEHOLDER_API_KEY
    # Agent CLIs that follow the Anthropic convention hit the same endpoint.
    assert env["ANTHROPIC_BASE_URL"] == endpoint.base_url
    assert env["EVALUATE_MODEL_ID"] == "test-model"
    assert env["EVALUATE_MODEL_NAME"] == "candidate"


def test_starting_twice_reuses_the_running_server(proxy):
    server, _, endpoint = proxy
    assert server.start() == endpoint


def test_stopping_twice_is_safe(proxy):
    server, _, _ = proxy
    server.stop()
    server.stop()


def test_the_proxy_works_as_a_context_manager():
    with OpenAICompatProxy(EchoChatBackend("m"), model_id="m") as endpoint:
        assert _get(f"{endpoint.base_url}/health")["status"] == "ok"


# ---------------------------------------------------------------------------
# The client against the real server
# ---------------------------------------------------------------------------


def test_the_client_round_trips_through_the_proxy(proxy):
    _, _, endpoint = proxy

    result = ChatClient(endpoint).chat([{"role": "user", "content": "hi"}])

    assert result.text == "hello"
    assert result.prompt_tokens == 7
    assert result.latency_ms >= 0


def test_the_client_carries_sampling_parameters_the_whole_way(proxy):
    """Client, wire and proxy together -- the path a benchmark actually takes."""
    _, backend, endpoint = proxy

    ChatClient(endpoint).chat(
        [{"role": "user", "content": "hi"}], top_p=0.95, top_k=64, temperature=1.0
    )

    call = backend.seen[0]
    assert (call["top_p"], call["top_k"], call["temperature"]) == (0.95, 64, 1.0)


def test_the_client_reports_an_upstream_error(proxy):
    _, backend, endpoint = proxy
    backend.error = RuntimeError("provider exploded")

    with pytest.raises(EndpointError, match="502"):
        ChatClient(endpoint).chat([{"role": "user", "content": "hi"}])


def test_the_client_reports_an_unreachable_endpoint():
    from foresightrlcore.inference.interfaces import InferenceEndpoint

    endpoint = InferenceEndpoint(
        base_url="http://127.0.0.1:1/v1", api_key="k", model_id="m", model_name="m"
    )

    with pytest.raises(EndpointError, match="could not reach"):
        ChatClient(endpoint, timeout=2).chat([{"role": "user", "content": "hi"}])


def test_the_echo_backend_is_deterministic():
    result = EchoChatBackend("m").chat(
        [{"role": "user", "content": "ping"}], max_tokens=16
    )
    assert result.text == "[echo:m] ping"


class TestSamplingSurvivesTheProxy:
    """The proxy forwards sampling instead of dropping it.

    Before this, the proxy read only max_tokens / temperature / stop from the
    request body, so an API-provider candidate silently ignored top_p, top_k and
    repetition_penalty: the run looked fine and measured different sampling than
    the config asked for.
    """

    def test_sampling_reaches_the_backend(self, proxy):
        server, backend, endpoint = proxy

        ChatClient(endpoint).chat(
            [{"role": "user", "content": "hi"}],
            max_tokens=16,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.1,
        )

        seen = backend.seen[0]
        assert seen["temperature"] == 0.6
        assert seen["top_p"] == 0.95
        assert seen["top_k"] == 20
        assert seen["repetition_penalty"] == 1.1

    def test_unset_sampling_stays_unset(self, proxy):
        server, backend, endpoint = proxy

        ChatClient(endpoint).chat([{"role": "user", "content": "hi"}], max_tokens=16)

        seen = backend.seen[0]
        assert seen["top_p"] is None
        assert seen["top_k"] is None
        assert seen["repetition_penalty"] is None

    def test_unknown_fields_are_passed_through(self, proxy):
        """So an option with no field of its own still reaches the provider."""
        server, backend, endpoint = proxy

        ChatClient(endpoint).chat(
            [{"role": "user", "content": "hi"}],
            max_tokens=16,
            extra={"chat_template_kwargs": {"reasoning_effort": "xhigh"}},
        )

        assert backend.seen[0]["extra"] == {
            "chat_template_kwargs": {"reasoning_effort": "xhigh"}
        }

    def test_fields_the_proxy_interprets_are_not_also_passed_through(self, proxy):
        """Forwarding `stream` or `messages` as provider options would be a bug."""
        server, backend, endpoint = proxy

        ChatClient(endpoint).chat(
            [{"role": "user", "content": "hi"}], max_tokens=16, top_p=0.9
        )

        assert backend.seen[0]["extra"] == {}


def test_the_manager_backend_forwards_sampling_to_the_client():
    """top_k and repetition_penalty ride in extra_body; top_p binds directly."""
    from unittest import mock

    from foresightrlcore.inference.backends import (
        ManagerChatBackend,
        build_llm_client_config,
    )

    client = mock.Mock()
    client.bind.return_value.invoke.return_value = "ok"
    manager = mock.Mock()
    manager.get.return_value = client

    backend = ManagerChatBackend(build_llm_client_config("bedrock", "some.model"))
    backend._manager = manager

    backend.chat(
        [{"role": "user", "content": "hi"}],
        max_tokens=16,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.1,
    )

    bound = client.bind.call_args.kwargs
    assert bound["top_p"] == 0.95
    assert bound["extra_body"] == {"top_k": 20, "repetition_penalty": 1.1}


def test_caller_supplied_extra_body_survives_alongside_sampling():
    from unittest import mock

    from foresightrlcore.inference.backends import (
        ManagerChatBackend,
        build_llm_client_config,
    )

    client = mock.Mock()
    client.bind.return_value.invoke.return_value = "ok"
    manager = mock.Mock()
    manager.get.return_value = client

    backend = ManagerChatBackend(build_llm_client_config("bedrock", "some.model"))
    backend._manager = manager

    backend.chat(
        [{"role": "user", "content": "hi"}],
        max_tokens=16,
        top_k=20,
        extra={"chat_template_kwargs": {"reasoning_effort": "xhigh"}},
    )

    assert client.bind.call_args.kwargs["extra_body"] == {
        "chat_template_kwargs": {"reasoning_effort": "xhigh"},
        "top_k": 20,
    }
