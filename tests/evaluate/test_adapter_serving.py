"""Serving a LoRA adapter over a base model, end to end through the config.

The motivating case: an SFT run publishes an adapter, not a full checkpoint. The
alternative to serving it directly is a merge job plus a full-size copy of the
base weights per candidate, so this path is what makes evaluating a fine-tune
cheap. It is also architecture-dependent -- vLLM's LoRA kernel rejects some
layer layouts -- which is why nothing here asserts that serving *succeeds*, only
that the adapter is delivered and addressed correctly when it does.
"""

from __future__ import annotations

from unittest import mock

import pytest

from foresightrlcore.common.vllm_process import (
    DEFAULT_ADAPTER_NAME,
    build_serve_command,
)
from foresightrlcore.config.evaluate_schema import (
    EvaluateModelConfig,
    local_adapter_channels,
    local_model_channels,
)
from foresightrlcore.config.perf_schema import VllmServingConfig
from foresightrlcore.inference.local_vllm import VllmServerEndpoint

BASE = "s3://bucket/models/Base-27B/"
ADAPTER = "s3://bucket/sft_output/run-1/output/model/"


def _model(**kwargs) -> EvaluateModelConfig:
    kwargs.setdefault("name", BASE)
    kwargs.setdefault("provider", "local")
    return EvaluateModelConfig(**kwargs)


class TestConfigSurface:
    def test_an_adapter_is_optional(self):
        assert _model().lora_adapter is None

    def test_the_declared_adapter_is_used_until_one_is_staged(self):
        model = _model(adapter=ADAPTER)

        assert model.lora_adapter == ADAPTER

        model.adapter_path = "/opt/ml/input/data/adapter"
        assert model.lora_adapter == "/opt/ml/input/data/adapter"

    def test_an_api_provider_cannot_take_an_adapter(self):
        # Bedrock serves whatever it was deployed with; accepting the field
        # would silently evaluate the base model under the fine-tune's name.
        with pytest.raises(ValueError, match="requires provider 'local'"):
            EvaluateModelConfig(
                name="anthropic.claude", provider="bedrock", adapter=ADAPTER
            )

    def test_an_adapter_under_a_label_needs_a_base(self):
        # The LoRA example in docs/launching-evaluate.md minus base_model: the
        # label would reach vllm serve as the model and fail on the GPU.
        with pytest.raises(ValueError, match="adapter.*base_model"):
            _model(name="qwen3-8b-sft", adapter=ADAPTER)

    @pytest.mark.parametrize("name", [BASE, "Qwen/Qwen3-8B", "/opt/models/base"])
    def test_an_adapter_over_weights_named_directly_needs_no_base(self, name):
        assert _model(name=name, adapter=ADAPTER).serving_path == name

    @pytest.mark.parametrize("provider", ["bedrock", "genos", "echo"])
    def test_only_a_local_model_can_take_a_base(self, provider):
        with pytest.raises(ValueError, match="base_model requires provider 'local'"):
            EvaluateModelConfig(name="label", provider=provider, base_model=BASE)


class TestLabelAndWeightsAreSeparable:
    """`name` is the identity; the weights can be named separately."""

    def test_name_serves_as_the_path_when_no_base_is_given(self):
        assert _model().serving_path == BASE

    def test_a_named_base_is_what_gets_served(self):
        model = _model(name="qwen3-8-sft", base_model=BASE)

        assert model.serving_path == BASE
        assert model.name == "qwen3-8-sft"

    def test_staged_weights_still_win(self):
        model = _model(name="label", base_model=BASE)
        model.weights_path = "/opt/ml/input/data/model"

        assert model.serving_path == "/opt/ml/input/data/model"


class TestChannels:
    def test_an_s3_adapter_gets_its_own_channel(self):
        models = [_model(adapter=ADAPTER)]

        assert local_adapter_channels(models) == [("adapter", models[0])]
        assert local_model_channels(models) == [("model", models[0])]

    def test_adapters_are_numbered_independently_of_weights(self):
        # A base-only candidate takes a model channel but no adapter channel, so
        # the adapter numbering must not follow the model numbering -- otherwise
        # adding a baseline renames a channel a resumed run is looking for.
        plain = _model(name="s3://bucket/models/Other/")
        adapted = _model(adapter=ADAPTER)

        assert [c for c, _ in local_model_channels([plain, adapted])] == [
            "model",
            "model1",
        ]
        assert [c for c, _ in local_adapter_channels([plain, adapted])] == ["adapter"]

    def test_candidates_sharing_weights_share_one_channel(self):
        # The base-vs-adapter comparison serves the same 27B twice. Two channels
        # would have SageMaker download tens of gigabytes twice over.
        base = _model(name="base-label", base_model=BASE)
        adapted = _model(name="adapter-label", base_model=BASE, adapter=ADAPTER)

        channels = local_model_channels([base, adapted])

        assert [c for c, _ in channels] == ["model", "model"]

    def test_distinct_weights_still_get_distinct_channels(self):
        one = _model(name="a", base_model=BASE)
        two = _model(name="b", base_model="s3://bucket/models/Other/")

        assert [c for c, _ in local_model_channels([one, two])] == ["model", "model1"]

    def test_a_local_directory_adapter_needs_no_channel(self):
        assert local_adapter_channels([_model(adapter="/data/adapter")]) == []

    def test_an_api_model_never_gets_an_adapter_channel(self):
        assert local_adapter_channels([EvaluateModelConfig(name="m")]) == []


class TestServeCommand:
    def test_without_an_adapter_nothing_lora_appears(self):
        cmd = build_serve_command("/weights", VllmServingConfig())

        assert "--enable-lora" not in cmd
        assert "--lora-modules" not in cmd

    def test_an_adapter_is_enabled_and_named(self):
        cmd = build_serve_command(
            "/weights", VllmServingConfig(), lora_adapter="/adapter"
        )

        assert "--enable-lora" in cmd
        assert (
            cmd[cmd.index("--lora-modules") + 1] == f"{DEFAULT_ADAPTER_NAME}=/adapter"
        )

    def test_the_rank_is_passed_when_set(self):
        # vLLM defaults max_lora_rank to 16 and refuses anything larger at load
        # time, so an r=32 adapter is unservable without this.
        cmd = build_serve_command(
            "/weights",
            VllmServingConfig(max_lora_rank=32),
            lora_adapter="/adapter",
        )

        assert cmd[cmd.index("--max-lora-rank") + 1] == "32"

    def test_the_rank_is_not_passed_without_an_adapter(self):
        cmd = build_serve_command("/weights", VllmServingConfig(max_lora_rank=32))

        assert "--max-lora-rank" not in cmd


class TestEndpointAddressesTheAdapter:
    """The part that fails silently if wrong, so it is asserted directly."""

    def test_the_endpoint_model_id_is_the_adapter(self):
        # vLLM answers to both the base name and the adapter name. Addressing
        # the base returns unadapted completions, which score like a fine-tune
        # that did nothing -- indistinguishable from a real negative result.
        endpoint = VllmServerEndpoint(
            "/weights",
            VllmServingConfig(served_model_name="base"),
            model_name="Base-27B-sft",
            lora_adapter="/adapter",
        )

        assert endpoint.served_model_name == DEFAULT_ADAPTER_NAME

    def test_without_an_adapter_the_served_name_is_unchanged(self):
        endpoint = VllmServerEndpoint(
            "/weights", VllmServingConfig(served_model_name="base")
        )

        assert endpoint.served_model_name == "base"

    def test_the_adapter_reaches_the_serve_command(self):
        endpoint = VllmServerEndpoint(
            "/weights", VllmServingConfig(), lora_adapter="/adapter"
        )
        with (
            mock.patch(
                "foresightrlcore.inference.local_vllm.subprocess.Popen"
            ) as popen,
            mock.patch(
                "foresightrlcore.inference.local_vllm.wait_until_healthy",
                return_value=1.0,
            ),
        ):
            endpoint.start()

        assert "--enable-lora" in popen.call_args[0][0]


class TestStagedMounts:
    def test_a_staged_adapter_is_mounted_for_serving(self, tmp_path, monkeypatch):
        from foresightrlcore.evaluate.runner import mount_staged_model_weights

        weights = tmp_path / "model"
        adapter = tmp_path / "adapter"
        weights.mkdir()
        adapter.mkdir()
        monkeypatch.setenv("SM_CHANNEL_MODEL", str(weights))
        monkeypatch.setenv("SM_CHANNEL_ADAPTER", str(adapter))

        config = mock.MagicMock()
        model = _model(adapter=ADAPTER)
        config.evaluate.models = [model]
        mount_staged_model_weights(config)

        assert model.serving_path == str(weights)
        assert model.lora_adapter == str(adapter)

    def test_nothing_is_mounted_outside_a_job(self, tmp_path, monkeypatch):
        from foresightrlcore.evaluate.runner import mount_staged_model_weights

        monkeypatch.setenv("SM_CHANNEL_ADAPTER", str(tmp_path / "absent"))
        config = mock.MagicMock()
        model = _model(adapter=ADAPTER)
        config.evaluate.models = [model]
        mount_staged_model_weights(config)

        assert model.lora_adapter == ADAPTER


class TestFingerprint:
    """Swapping adapters must invalidate stored trials; staging must not."""

    def _fingerprint(self, model):
        from foresightrlcore.config.evaluate_schema import BenchmarkSpec
        from foresightrlcore.evaluate.runner import measurement_fingerprint

        return measurement_fingerprint(BenchmarkSpec(name="b", type="mcq"), model)

    def test_a_different_adapter_is_a_different_measurement(self):
        assert self._fingerprint(_model(adapter=ADAPTER)) != self._fingerprint(
            _model(adapter="s3://bucket/sft_output/run-2/output/model/")
        )

    def test_an_adapter_differs_from_the_bare_base(self):
        assert self._fingerprint(_model(adapter=ADAPTER)) != self._fingerprint(_model())

    def test_where_the_adapter_was_staged_is_not_part_of_the_measurement(self):
        staged = _model(adapter=ADAPTER)
        staged.adapter_path = "/opt/ml/input/data/adapter"

        assert self._fingerprint(staged) == self._fingerprint(_model(adapter=ADAPTER))


class TestLauncherChannels:
    def test_the_launcher_mounts_both_prefixes(self):
        from foresightrlcore.execution.evaluate_launcher import (
            EvaluateSageMakerLauncher,
        )

        config = mock.MagicMock()
        config.evaluate.models = [_model(adapter=ADAPTER)]
        launcher = mock.MagicMock(spec=EvaluateSageMakerLauncher)
        launcher.config = config

        channels = EvaluateSageMakerLauncher._build_model_inputs(launcher)

        assert channels == {"model": BASE, "adapter": ADAPTER}
