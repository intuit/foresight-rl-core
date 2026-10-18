"""Generation clients: base helpers, bedrock, genos, local vLLM, and factory.

Network / GPU / boto3 are all mocked; nothing here talks to a real endpoint.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from foresightrlcore.config.evaluate_schema import EvaluateModelConfig
from foresightrlcore.config.optimizer_schema import OptimizerModelConfig
from foresightrlcore.config.perf_schema import VllmServingConfig
from foresightrlcore.generation import base
from foresightrlcore.generation.factory import make_generation_client


class TestBaseHelpers:
    def test_approx_tokens_empty_is_zero(self):
        assert base.approx_tokens("") == 0

    def test_approx_tokens_minimum_one(self):
        assert base.approx_tokens("ab") == 1  # 2 // 4 == 0 -> max(1, 0)

    def test_approx_tokens_scales_with_length(self):
        assert base.approx_tokens("a" * 40) == 10

    def test_usage_prefers_metadata(self):
        msg = SimpleNamespace(usage_metadata={"input_tokens": 12, "output_tokens": 7})
        assert base.usage_from_langchain_message(msg, "p", "t") == (12, 7)

    def test_usage_falls_back_when_metadata_zero(self):
        msg = SimpleNamespace(usage_metadata={"input_tokens": 0, "output_tokens": 0})
        prompt, text = "x" * 40, "y" * 20
        assert base.usage_from_langchain_message(msg, prompt, text) == (10, 5)

    def test_usage_falls_back_when_metadata_missing(self):
        msg = SimpleNamespace()
        assert base.usage_from_langchain_message(msg, "abcd", "ef") == (1, 1)

    def test_text_from_string_content(self):
        assert base.text_from_langchain_message(SimpleNamespace(content="hello")) == (
            "hello"
        )

    def test_text_from_list_of_blocks(self):
        msg = SimpleNamespace(content=[{"text": "a"}, {"text": "b"}, "c"])
        assert base.text_from_langchain_message(msg) == "abc"

    def test_text_from_non_str_object(self):
        assert base.text_from_langchain_message(SimpleNamespace(content=123)) == "123"

    def test_text_from_bare_string(self):
        assert base.text_from_langchain_message("plain") == "plain"


class TestBedrockGenerationClient:
    def _client(self):
        from foresightrlcore.generation.bedrock_client import (
            BedrockGenerationClient,
        )

        gen = BedrockGenerationClient("model-x", region="us-west-2")
        return gen

    def test_text_from_converse(self):
        from foresightrlcore.generation.bedrock_client import (
            _text_from_converse,
        )

        response = {
            "output": {"message": {"content": [{"text": "foo"}, {"text": "bar"}]}}
        }
        assert _text_from_converse(response) == "foobar"

    def test_text_from_converse_empty(self):
        from foresightrlcore.generation.bedrock_client import (
            _text_from_converse,
        )

        assert _text_from_converse({}) == ""

    def test_usage_from_converse_reads_provider_tokens(self):
        from foresightrlcore.generation.bedrock_client import (
            _usage_from_converse,
        )

        response = {"usage": {"inputTokens": 30, "outputTokens": 9}}
        assert _usage_from_converse(response, "p", "t") == (30, 9)

    def test_usage_from_converse_falls_back(self):
        from foresightrlcore.generation.bedrock_client import (
            _usage_from_converse,
        )

        response = {"usage": {"inputTokens": 0, "outputTokens": 0}}
        assert _usage_from_converse(response, "x" * 40, "y" * 8) == (10, 2)

    def test_generate_returns_text_and_usage(self):
        gen = self._client()
        converse = MagicMock(
            return_value={
                "output": {"message": {"content": [{"text": "answer"}]}},
                "usage": {"inputTokens": 11, "outputTokens": 3},
            }
        )
        chat = SimpleNamespace(client=SimpleNamespace(converse=converse), model_id="m")
        gen._manager = SimpleNamespace(get=lambda: chat, force_refresh=lambda: chat)
        text, usage = gen.generate("prompt", tag="gen", max_tokens=100)
        assert text == "answer"
        assert usage.prompt_tokens == 11
        assert usage.completion_tokens == 3
        assert usage.tag == "gen"
        converse.assert_called_once()

    def test_close_is_noop(self):
        assert self._client().close() is None


@pytest.mark.requires_intuit
class TestGenosGenerationClient:
    def test_langchain_client_close_is_noop(self):
        from foresightrl_intuit.optimizer.generation.genos_client import (
            LangChainChatGenerationClient,
        )

        client = LangChainChatGenerationClient(MagicMock())
        assert client.close() is None

    def test_make_genos_generation_client(self):
        from foresightrl_intuit.optimizer.generation import genos_client

        fake_manager = MagicMock()
        with (
            patch(
                "foresightrl_intuit.clients.genos.load_genos_credentials_from_env",
                return_value={"app_secret": "x"},
            ) as load_creds,
            patch(
                "foresightrl_intuit.clients.genos.GenosClientManager",
                return_value=fake_manager,
            ) as ctor,
        ):
            client = genos_client.make_genos_generation_client("model-1")
        load_creds.assert_called_once()
        ctor.assert_called_once()
        assert client._manager is fake_manager


class TestLocalVllmClient:
    def _serving(self, **over):
        base_kwargs: dict[str, Any] = dict(port=9001, served_model_name="mymodel")
        base_kwargs.update(over)
        return VllmServingConfig(**base_kwargs)

    def _client(self, serving=None):
        from foresightrlcore.generation.local_vllm_client import (
            LocalVllmGenerationClient,
        )

        return LocalVllmGenerationClient(
            "Qwen/Qwen3.5-0.8B", serving or self._serving()
        )

    def test_build_command_basic(self):
        client = self._client()
        cmd = client._build_command()
        assert cmd[:3] == ["vllm", "serve", "Qwen/Qwen3.5-0.8B"]
        assert "--port" in cmd and "9001" in cmd
        assert "--served-model-name" in cmd and "mymodel" in cmd

    def test_build_command_optional_flags(self):
        serving = self._serving(
            max_model_len=4096,
            trust_remote_code=True,
            enforce_eager=True,
            dtype="bfloat16",
            quantization="fp8",
            enable_prefix_caching=True,
            extra_args={"max-num-batched-tokens": "8192"},
        )
        cmd = client_cmd = self._client(serving)._build_command()
        assert "--max-model-len" in cmd and "4096" in cmd
        assert "--trust-remote-code" in cmd
        assert "--enforce-eager" in cmd
        assert "--dtype" in cmd and "bfloat16" in cmd
        assert "--quantization" in cmd and "fp8" in cmd
        assert "--enable-prefix-caching" in cmd
        assert "--max-num-batched-tokens" in client_cmd

    def test_build_env_sets_gpu_devices(self):
        serving = self._serving(gpu_devices="0,1", vllm_env={"FOO": "bar"})
        env = self._client(serving)._build_env()
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert env["FOO"] == "bar"

    def test_start_success(self):
        client = self._client()
        proc = MagicMock()
        proc.poll.return_value = None
        with (
            patch(
                "foresightrlcore.generation.local_vllm_client.subprocess.Popen",
                return_value=proc,
            ),
            patch(
                "foresightrlcore.perf_test.runner._vllm_health_ok", return_value=True
            ),
        ):
            elapsed = client.start()
        assert elapsed >= 0.0
        assert client._proc is proc

    def test_start_raises_when_process_exits_early(self):
        client = self._client()
        proc = MagicMock()
        proc.poll.return_value = 1
        proc.returncode = 1
        with (
            patch(
                "foresightrlcore.generation.local_vllm_client.subprocess.Popen",
                return_value=proc,
            ),
            patch(
                "foresightrlcore.perf_test.runner._vllm_health_ok", return_value=False
            ),
        ):
            with pytest.raises(RuntimeError, match="exited with code"):
                client.start()

    def test_close_terminates_running_process(self):
        client = self._client()
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 4242
        client._proc = proc
        with (
            patch("foresightrlcore.generation.local_vllm_client.os.killpg") as killpg,
            patch(
                "foresightrlcore.generation.local_vllm_client.os.getpgid",
                return_value=4242,
            ),
        ):
            client.close()
        killpg.assert_called()
        assert client._proc is None

    def test_close_when_no_process(self):
        client = self._client()
        client._proc = None
        client.close()  # no raise

    def test_close_when_already_exited(self):
        client = self._client()
        proc = MagicMock()
        proc.poll.return_value = 0
        proc.returncode = 0
        client._proc = proc
        client.close()
        assert client._proc is None

    def test_local_chat_manager_creates_and_refreshes(self):
        from foresightrlcore.generation.local_vllm_client import (
            _LocalChatManager,
        )

        fake_chat_cls = MagicMock(return_value="chat-client")
        fake_module = SimpleNamespace(ChatOpenAI=fake_chat_cls)
        with patch.dict(sys.modules, {"langchain_openai": fake_module}):
            mgr = _LocalChatManager(port=9001, served_model_name="m")
            first = mgr.get()
            second = mgr.get()  # cached
            assert first == "chat-client"
            assert second == first
            fake_chat_cls.assert_called_once()
            refreshed = mgr.force_refresh()
            assert refreshed == "chat-client"
            assert fake_chat_cls.call_count == 2


class TestFactory:
    @pytest.mark.parametrize(
        "model_config, temperature",
        [
            (OptimizerModelConfig, None),
            (OptimizerModelConfig, 0.0),
            (OptimizerModelConfig, 0.7),
            (EvaluateModelConfig, 0.0),
            (EvaluateModelConfig, 0.7),
        ],
    )
    def test_bedrock_request_honors_candidate_temperature(
        self,
        model_config: type[OptimizerModelConfig] | type[EvaluateModelConfig],
        temperature: Optional[float],
    ):
        model = model_config.model_validate(
            {
                "name": "synthetic",
                "provider": "bedrock",
                "temperature": temperature,
                "region": "us-west-2",
            }
        )
        with patch("boto3.client") as boto_client:
            converse = boto_client.return_value.converse
            converse.return_value = {
                "output": {"message": {"content": [{"text": "answer"}]}}
            }
            client = make_generation_client(model, max_retries=1)
            text, _ = client.generate("synthetic input", tag="test", max_tokens=123)

        request = converse.call_args.kwargs
        expected: dict[str, Any] = {"maxTokens": 123}
        if temperature is not None:
            expected["temperature"] = temperature
        assert request["inferenceConfig"] == expected
        assert text == "answer"

    @pytest.mark.requires_intuit
    @pytest.mark.parametrize(
        "model_config, temperature",
        [
            (OptimizerModelConfig, None),
            (OptimizerModelConfig, 0.0),
            (OptimizerModelConfig, 0.7),
            (EvaluateModelConfig, 0.0),
            (EvaluateModelConfig, 0.7),
        ],
    )
    def test_genos_temperature_survives_expiry_and_forced_refresh(
        self,
        model_config: type[OptimizerModelConfig] | type[EvaluateModelConfig],
        temperature: Optional[float],
    ):
        model = model_config.model_validate(
            {"name": "synthetic", "provider": "genos", "temperature": temperature}
        )
        chat = MagicMock()
        chat.bind.return_value.invoke.return_value = SimpleNamespace(content="answer")
        with (
            patch(
                "foresightrl_intuit.clients.genos.load_genos_credentials_from_env",
                return_value={"genos_env": "prd"},
            ),
            patch(
                "foresightrl_intuit.clients.genos.get_genos_auth_header",
                return_value="synthetic-auth",
            ),
            patch(
                "foresightrl_intuit.clients.genos.time.time", return_value=1000
            ) as now,
            patch.dict(
                sys.modules,
                {
                    "langchain_openai": SimpleNamespace(
                        ChatOpenAI=MagicMock(return_value=chat)
                    )
                },
            ),
        ):
            constructor = sys.modules["langchain_openai"].ChatOpenAI
            client: Any = make_generation_client(model, max_retries=1)
            client.generate("synthetic input", tag="test", max_tokens=123)
            now.return_value = 2000
            client.generate("synthetic input", tag="test", max_tokens=123)
            client._manager.force_refresh()

        assert constructor.call_count == 3
        for call in constructor.call_args_list:
            assert call.kwargs["temperature"] == temperature
            assert call.kwargs["model"] == "synthetic"

    def test_unknown_provider_raises(self):
        model = OptimizerModelConfig(name="x", provider="genos")
        object.__setattr__(model, "provider", "openai")
        with pytest.raises(ValueError, match="unknown generation provider"):
            make_generation_client(model)

    @pytest.mark.requires_intuit
    def test_genos_provider(self):
        model = OptimizerModelConfig(name="gemini", provider="genos")
        with patch(
            "foresightrl_intuit.optimizer.generation.genos_client."
            "make_genos_generation_client",
            return_value="genos-client",
        ) as make:
            client: Any = make_generation_client(model)
        make.assert_called_once()
        assert client == "genos-client"

    def test_bedrock_provider(self):
        model = OptimizerModelConfig(
            name="claude", provider="bedrock", region="us-west-2"
        )
        with patch(
            "foresightrlcore.generation.bedrock_client." "BedrockGenerationClient",
            return_value="bedrock-client",
        ) as ctor:
            client: Any = make_generation_client(model)
        ctor.assert_called_once()
        assert client == "bedrock-client"
