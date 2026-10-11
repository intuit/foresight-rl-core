"""Endpoint selection per provider, the manager-backed backend, and local vLLM."""

from __future__ import annotations

import subprocess
from unittest import mock

import pytest

from foresightrlcore.config.perf_schema import VllmServingConfig
from foresightrlcore.inference import endpoint_for, make_endpoint_handle
from foresightrlcore.inference.backends import (
    ManagerChatBackend,
    build_llm_client_config,
)
from foresightrlcore.inference.local_vllm import LOCAL_API_KEY, VllmServerEndpoint
from foresightrlcore.inference.proxy import OpenAICompatProxy


class _Model:
    """The duck-typed ModelSpec the inference layer accepts."""

    def __init__(self, name, provider, **kwargs):
        self.name = name
        self.provider = provider
        self.temperature = kwargs.get("temperature", 0.0)
        self.max_tokens = kwargs.get("max_tokens", 512)
        self.region = kwargs.get("region")
        self.serving = kwargs.get("serving")
        self.truncate_prompt_tokens = None
        self._model_id = kwargs.get("model_id") or name
        self.weights_path = kwargs.get("weights_path")
        self.adapter = kwargs.get("adapter")

    @property
    def resolved_model_id(self):
        return self._model_id

    @property
    def serving_path(self):
        return self.weights_path or self.name

    @property
    def lora_adapter(self):
        return self.adapter


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["bedrock", "genos", "echo"])
def test_api_providers_are_served_by_the_proxy(provider):
    handle = make_endpoint_handle(_Model("m", provider, model_id="mid"))

    assert isinstance(handle, OpenAICompatProxy)
    assert handle.model_id == "mid"


def test_a_local_model_is_served_by_vllm():
    handle = make_endpoint_handle(
        _Model("Qwen/Qwen3-8B", "local", serving=VllmServingConfig(port=8123))
    )

    assert isinstance(handle, VllmServerEndpoint)
    assert handle.base_url == "http://127.0.0.1:8123/v1"


def test_an_unknown_provider_is_reported():
    with pytest.raises(ValueError, match="no inference endpoint for provider"):
        make_endpoint_handle(_Model("m", "carrier-pigeon"))


def test_endpoint_for_brackets_the_lifecycle():
    with endpoint_for(_Model("m", "echo")) as endpoint:
        base_url = endpoint.base_url
        assert base_url.endswith("/v1")

    # Rebinding the same port would fail if the server were still up.
    with endpoint_for(_Model("m", "echo")) as second:
        assert second.base_url.endswith("/v1")


def test_the_endpoint_is_torn_down_even_when_the_body_raises():
    stopped = []

    class _Handle:
        def start(self):
            from foresightrlcore.inference.interfaces import InferenceEndpoint

            return InferenceEndpoint("u", "k", "m", "m")

        def stop(self):
            stopped.append(True)

    with mock.patch(
        "foresightrlcore.inference.make_endpoint_handle", return_value=_Handle()
    ):
        with pytest.raises(RuntimeError):
            with endpoint_for(_Model("m", "echo")):
                raise RuntimeError("benchmark blew up")

    assert stopped == [True]


# ---------------------------------------------------------------------------
# Client config construction
# ---------------------------------------------------------------------------


def test_a_bedrock_client_config_is_built_from_a_flat_model_spec():
    config = build_llm_client_config("bedrock", "anthropic.claude", region="us-west-2")

    assert config.provider == "bedrock"
    assert config.bedrock.model_id == "anthropic.claude"
    assert config.bedrock.region == "us-west-2"


def test_a_genos_client_config_is_built_the_same_way():
    config = build_llm_client_config("genos", "gpt-4o", temperature=0.3)

    assert config.provider == "genos"
    assert config.genos.model_id == "gpt-4o"


def test_a_provider_with_no_api_client_is_reported():
    with pytest.raises(ValueError, match="no API-backed chat client"):
        build_llm_client_config("local", "m")


@pytest.mark.parametrize("provider", ["bedrock", "genos"])
def test_endpoint_client_config_still_requires_a_numeric_temperature(provider):
    with pytest.raises(ValueError, match="requires a numeric temperature"):
        make_endpoint_handle(_Model("m", provider, temperature=None))


# ---------------------------------------------------------------------------
# ManagerChatBackend
# ---------------------------------------------------------------------------


class _FakeManager:
    def __init__(self, content="answer"):
        self.content = content
        self.bound = None

    def get(self):
        backend = self

        class _Client:
            def bind(self, **kwargs):
                backend.bound = kwargs
                return self

            def invoke(self, messages):
                return type(
                    "Msg",
                    (),
                    {"content": backend.content, "response_metadata": {}},
                )()

        return _Client()

    def force_refresh(self):
        pass


def test_the_manager_backend_binds_the_request_parameters():
    backend = ManagerChatBackend(object())
    backend._manager = _FakeManager()

    result = backend.chat(
        [{"role": "user", "content": "hi"}],
        max_tokens=64,
        temperature=0.2,
        stop=["END"],
    )

    assert result.text == "answer"
    assert backend._manager.bound == {
        "max_tokens": 64,
        "temperature": 0.2,
        "stop": ["END"],
    }


def test_top_p_binds_directly_and_top_k_travels_as_a_provider_extension():
    """Neither client takes ``top_k`` by name; both accept an extension dict."""
    backend = ManagerChatBackend(object())
    backend._manager = _FakeManager()

    backend.chat(
        [{"role": "user", "content": "hi"}], max_tokens=8, top_p=0.95, top_k=64
    )

    assert backend._manager.bound == {
        "max_tokens": 8,
        "top_p": 0.95,
        "extra_body": {"top_k": 64},
    }


def test_an_explicit_extension_wins_over_the_top_k_shorthand():
    backend = ManagerChatBackend(object())
    backend._manager = _FakeManager()

    backend.chat(
        [{"role": "user", "content": "hi"}],
        max_tokens=8,
        top_k=64,
        extra={"top_k": 5, "other": 1},
    )

    assert backend._manager.bound == {
        "max_tokens": 8,
        "extra_body": {"top_k": 5, "other": 1},
    }


def test_unset_sampling_parameters_are_not_bound_at_all():
    backend = ManagerChatBackend(object())
    backend._manager = _FakeManager()

    backend.chat([{"role": "user", "content": "hi"}], max_tokens=8)

    assert backend._manager.bound == {"max_tokens": 8}


def test_the_manager_is_resolved_lazily():
    """Constructing a backend must not need credentials."""
    with mock.patch("foresightrlcore.clients.llm.resolve_llm_manager") as resolve:
        backend = ManagerChatBackend(object())
        assert resolve.call_count == 0
        backend.manager
        assert resolve.call_count == 1


# ---------------------------------------------------------------------------
# Local vLLM
# ---------------------------------------------------------------------------


def test_the_vllm_endpoint_advertises_the_served_model_name():
    handle = VllmServerEndpoint(
        "/models/qwen",
        VllmServingConfig(port=9000, served_model_name="qwen"),
        model_name="qwen-8b",
    )

    assert handle.endpoint.model_id == "qwen"
    assert handle.endpoint.model_name == "qwen-8b"
    assert handle.endpoint.api_key == LOCAL_API_KEY


def test_the_vllm_endpoint_waits_for_health_before_returning():
    serving = VllmServingConfig(port=9001, startup_timeout=60)
    handle = VllmServerEndpoint("/models/qwen", serving)
    proc = mock.MagicMock(pid=1234)
    proc.poll.return_value = None

    with (
        mock.patch("subprocess.Popen", return_value=proc),
        mock.patch("foresightrlcore.common.vllm_process.health_ok", return_value=True),
    ):
        endpoint = handle.start()

    assert endpoint.base_url == "http://127.0.0.1:9001/v1"


def test_a_server_that_dies_during_startup_is_reported():
    serving = VllmServingConfig(port=9002, startup_timeout=60)
    handle = VllmServerEndpoint("/models/qwen", serving)
    proc = mock.MagicMock(pid=1234)
    proc.poll.return_value = 1
    proc.returncode = 1

    with mock.patch("subprocess.Popen", return_value=proc):
        with pytest.raises(RuntimeError, match="exited with code 1"):
            handle.start()


def test_stopping_a_server_that_never_started_is_safe():
    VllmServerEndpoint("/models/qwen", VllmServingConfig()).stop()


def test_the_serve_command_carries_the_serving_parameters():
    from foresightrlcore.common.vllm_process import build_serve_command

    cmd = build_serve_command(
        "/models/qwen",
        VllmServingConfig(
            port=9003,
            tensor_parallel_size=4,
            max_model_len=8192,
            trust_remote_code=True,
            dtype="bfloat16",
        ),
    )

    assert cmd[:3] == ["vllm", "serve", "/models/qwen"]
    assert "--tensor-parallel-size" in cmd and "4" in cmd
    assert "--max-model-len" in cmd and "8192" in cmd
    assert "--trust-remote-code" in cmd
    assert "--dtype" in cmd and "bfloat16" in cmd


def test_serving_env_overrides_are_applied():
    from foresightrlcore.common.vllm_process import build_serve_env

    env = build_serve_env(
        VllmServingConfig(vllm_env={"VLLM_USE_V1": "0"}, gpu_devices="0,1")
    )

    assert env["VLLM_USE_V1"] == "0"
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_terminate_escalates_to_sigkill_when_the_process_ignores_sigterm():
    from foresightrlcore.common.vllm_process import terminate_process_group

    proc = mock.MagicMock(pid=4321)
    proc.poll.return_value = None
    proc.wait.side_effect = subprocess.TimeoutExpired("vllm", 1)

    with mock.patch("os.killpg") as killpg, mock.patch("os.getpgid", return_value=4321):
        terminate_process_group(proc, grace_seconds=1)

    signals = [call.args[1] for call in killpg.call_args_list]
    assert len(signals) == 2  # SIGTERM then SIGKILL


# ---------------------------------------------------------------------------
# The provider shims must satisfy the contract the proxy actually calls
#
# The proxy calls `manager.get().bind(**kwargs).invoke(messages)` for every
# provider. Genos inherits that from ChatOpenAI; Bedrock's in-house shim has to
# implement it. It did not, and no test noticed, because every test here used
# a fake backend -- so the first thing to touch real Bedrock was a live run.
# ---------------------------------------------------------------------------


class _FakeConverseClient:
    """Stands in for a boto3 bedrock-runtime client."""

    def __init__(self, text="hello"):
        self.text = text
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "output": {"message": {"content": [{"text": self.text}]}},
            "usage": {"inputTokens": 11, "outputTokens": 7},
        }


def _bedrock_chat(text="hello"):
    from foresightrlcore.clients.bedrock import _BedrockChat

    client = _FakeConverseClient(text)
    return _BedrockChat(client, "some.model", 0.0), client


def test_the_bedrock_shim_supports_bind_then_invoke():
    chat, client = _bedrock_chat("Paris")

    message = chat.bind(max_tokens=64, temperature=0.3).invoke(
        [{"role": "user", "content": "Capital of France?"}]
    )

    assert message.content == "Paris"
    assert message.usage_metadata == {"input_tokens": 11, "output_tokens": 7}
    inference = client.calls[0]["inferenceConfig"]
    assert inference["maxTokens"] == 64
    assert inference["temperature"] == 0.3


def test_bind_forwards_stop_sequences_and_extra_body():
    chat, client = _bedrock_chat()

    chat.bind(max_tokens=8, stop=["END"], extra_body={"top_k": 5}).invoke(
        [{"role": "user", "content": "hi"}]
    )

    assert client.calls[0]["inferenceConfig"]["stopSequences"] == ["END"]
    assert client.calls[0]["additionalModelRequestFields"] == {"top_k": 5}


def test_top_p_is_translated_to_the_converse_spelling():
    chat, client = _bedrock_chat()

    chat.bind(max_tokens=8, top_p=0.95).invoke([{"role": "user", "content": "hi"}])

    assert client.calls[0]["inferenceConfig"]["topP"] == 0.95


def test_a_bound_top_p_suppresses_the_clients_default_temperature():
    """Anthropic rejects both together, so the default must step aside.

    The client-level temperature is a default nobody asked for; the bound
    top_p is a request somebody made. Sending both fails the call outright.
    """
    chat, client = _bedrock_chat()

    chat.bind(max_tokens=16, top_p=0.95).invoke([{"role": "user", "content": "hi"}])

    inference = client.calls[0]["inferenceConfig"]
    assert inference["topP"] == 0.95
    assert "temperature" not in inference


def test_binding_both_sends_both_and_lets_the_provider_object():
    chat, client = _bedrock_chat()

    chat.bind(max_tokens=16, top_p=0.95, temperature=1.0).invoke(
        [{"role": "user", "content": "hi"}]
    )

    inference = client.calls[0]["inferenceConfig"]
    assert inference["topP"] == 0.95 and inference["temperature"] == 1.0


def test_an_unset_top_p_is_left_out_of_the_inference_config():
    chat, client = _bedrock_chat()

    chat.bind(max_tokens=8).invoke([{"role": "user", "content": "hi"}])

    assert "topP" not in client.calls[0]["inferenceConfig"]


def test_a_system_message_is_lifted_out_of_the_conversation():
    chat, client = _bedrock_chat()

    chat.bind(max_tokens=8).invoke(
        [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hi"},
        ]
    )

    call = client.calls[0]
    assert call["system"] == [{"text": "Be terse."}]
    assert [m["role"] for m in call["messages"]] == ["user"]


def test_invoke_works_without_binding_anything():
    chat, client = _bedrock_chat("ok")

    assert chat.invoke([{"role": "user", "content": "hi"}]).content == "ok"
    assert "maxTokens" not in client.calls[0]["inferenceConfig"]


def test_the_proxy_serves_the_real_bedrock_shim_end_to_end():
    """The exact path that failed live: proxy -> backend -> _BedrockChat."""
    from unittest import mock

    from foresightrlcore.inference.backends import (
        ManagerChatBackend,
        build_llm_client_config,
    )
    from foresightrlcore.inference.client import ChatClient
    from foresightrlcore.inference.proxy import OpenAICompatProxy

    chat, client = _bedrock_chat("Paris")
    manager = mock.Mock()
    manager.get.return_value = chat

    backend = ManagerChatBackend(build_llm_client_config("bedrock", "some.model"))
    backend._manager = manager

    with OpenAICompatProxy(backend, model_id="some.model") as endpoint:
        result = ChatClient(endpoint).chat(
            [{"role": "user", "content": "Capital of France?"}], max_tokens=32
        )

    assert result.text == "Paris"
    assert (result.prompt_tokens, result.completion_tokens) == (11, 7)


def test_a_top_p_request_reaches_bedrock_without_a_conflicting_temperature():
    """Reproduces the live SageMaker failure, along its full path.

    A candidate configuring only `top_p` produced
    ``ValidationException: temperature and top_p cannot both be specified``
    on every call. Every hop looked correct in isolation -- the temperature
    was added at the last one, from a client default.
    """
    from unittest import mock

    from foresightrlcore.inference.backends import (
        ManagerChatBackend,
        build_llm_client_config,
    )
    from foresightrlcore.inference.client import ChatClient
    from foresightrlcore.inference.proxy import OpenAICompatProxy

    chat, client = _bedrock_chat("Paris")
    manager = mock.Mock()
    manager.get.return_value = chat

    backend = ManagerChatBackend(build_llm_client_config("bedrock", "some.model"))
    backend._manager = manager

    with OpenAICompatProxy(backend, model_id="some.model") as endpoint:
        ChatClient(endpoint).chat(
            [{"role": "user", "content": "hi"}], max_tokens=32, top_p=0.95, top_k=64
        )

    inference = client.calls[0]["inferenceConfig"]
    assert inference["topP"] == 0.95
    assert "temperature" not in inference
    assert client.calls[0]["additionalModelRequestFields"] == {"top_k": 64}
