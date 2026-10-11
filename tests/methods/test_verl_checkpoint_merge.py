"""Tests for foresightrlcore.methods.verl_checkpoint_merge.

``find_latest_checkpoint_step`` is pure Python (no verl/torch/peft/
transformers dependency) and is tested fully here.

``reconstruct_full_state_dict``/``merge_lora_and_save``/``_build_lora_config``
need verl/peft/accelerate/transformers, which aren't installed in this
dev/test environment. When they genuinely are missing, the lazy-import
boundary itself is exercised (below). But their actual logic — the
FSDP-shard-reconstruction call chain, the LoraConfig kwargs mapping, the
strict-state-dict-load-then-merge-then-save sequence — is real, ordinary
Python control flow around those imports, so it's exercised directly by
injecting fake modules into ``sys.modules`` for the duration of each test
(the same class of technique used for verl in
:mod:`tests.tokenizer.test_cli`'s S3 mocking, just for import statements
instead of function calls). This is possible without changing any project
dependency and without a GPU: the actual merge math (meta-device LoRA
construction, ``peft``'s own ``merge_and_unload()`` on a
``target_parameters``-shaped fused-expert module) was separately verified
by hand against the real Gemma 4 architecture before this was wired in —
see the module docstring for what was checked and why.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from foresightrlcore.methods.verl_checkpoint_merge import (
    _build_lora_config,
    check_lora_train_meta_matches,
    find_latest_checkpoint_step,
    merge_lora_and_save,
    reconstruct_full_state_dict,
)


def _fake_lora(**overrides):
    base = dict(
        r=64, alpha=128, target_modules=[], exclude_modules=[], target_parameters=[]
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestFindLatestCheckpointStep:
    def test_explicit_step(self, tmp_path):
        (tmp_path / "global_step_10").mkdir()
        (tmp_path / "global_step_20").mkdir()
        result = find_latest_checkpoint_step(str(tmp_path), step=10)
        assert result == str(tmp_path / "global_step_10")

    def test_explicit_step_missing_raises(self, tmp_path):
        (tmp_path / "global_step_10").mkdir()
        with pytest.raises(FileNotFoundError, match="global_step_99"):
            find_latest_checkpoint_step(str(tmp_path), step=99)

    def test_auto_detects_highest_step(self, tmp_path):
        for n in (10, 30, 20):
            (tmp_path / f"global_step_{n}").mkdir()
        result = find_latest_checkpoint_step(str(tmp_path), step=None)
        assert result == str(tmp_path / "global_step_30")

    def test_ignores_non_checkpoint_directories(self, tmp_path):
        (tmp_path / "global_step_5").mkdir()
        (tmp_path / "some_other_dir").mkdir()
        (tmp_path / "global_step_not_a_number").mkdir()
        result = find_latest_checkpoint_step(str(tmp_path), step=None)
        assert result == str(tmp_path / "global_step_5")

    def test_raises_when_no_checkpoints_found(self, tmp_path):
        (tmp_path / "unrelated").mkdir()
        with pytest.raises(FileNotFoundError, match="global_step_"):
            find_latest_checkpoint_step(str(tmp_path), step=None)


class TestCheckLoraTrainMetaMatches:
    """Cross-checks the merge config's model.lora against verl's own

    lora_train_meta.json (written unconditionally by verl's
    FSDPCheckpointManager from the real trained peft_config.r/lora_alpha)
    — catches an alpha mismatch in particular, which no state-dict
    key/shape check can catch since alpha isn't itself a parameter.
    """

    def _write_meta(self, tmp_path, **fields):
        import json

        (tmp_path / "lora_train_meta.json").write_text(json.dumps(fields))

    def test_no_meta_file_is_a_noop(self, tmp_path):
        check_lora_train_meta_matches(str(tmp_path), _fake_lora(r=64, alpha=128))

    def test_matching_r_and_alpha_does_not_raise(self, tmp_path):
        self._write_meta(tmp_path, r=64, lora_alpha=128)
        check_lora_train_meta_matches(str(tmp_path), _fake_lora(r=64, alpha=128))

    def test_mismatched_alpha_raises(self, tmp_path):
        self._write_meta(tmp_path, r=64, lora_alpha=128)
        with pytest.raises(ValueError, match="lora_alpha: trained=128"):
            check_lora_train_meta_matches(str(tmp_path), _fake_lora(r=64, alpha=256))

    def test_mismatched_r_raises(self, tmp_path):
        self._write_meta(tmp_path, r=64, lora_alpha=128)
        with pytest.raises(ValueError, match="r: trained=64"):
            check_lora_train_meta_matches(str(tmp_path), _fake_lora(r=32, alpha=128))

    def test_missing_fields_in_meta_are_not_checked(self, tmp_path):
        self._write_meta(tmp_path, task_type="CAUSAL_LM")
        check_lora_train_meta_matches(str(tmp_path), _fake_lora(r=64, alpha=128))

    def test_malformed_json_does_not_crash(self, tmp_path):
        (tmp_path / "lora_train_meta.json").write_text("not json")
        check_lora_train_meta_matches(str(tmp_path), _fake_lora(r=64, alpha=128))

    def test_non_dict_json_does_not_crash(self, tmp_path):
        (tmp_path / "lora_train_meta.json").write_text("[1, 2, 3]")
        check_lora_train_meta_matches(str(tmp_path), _fake_lora(r=64, alpha=128))


class TestBuildLoraConfig:
    """_build_lora_config with a fake ``peft`` module injected into

    sys.modules — exercises the real kwargs-mapping logic without peft
    actually installed.
    """

    def _install_fake_peft(self, monkeypatch):
        captured: dict = {}
        fake_peft: Any = ModuleType("peft")

        def fake_lora_config(**kwargs):
            captured["kwargs"] = kwargs
            return SimpleNamespace(**kwargs)

        fake_peft.LoraConfig = fake_lora_config
        monkeypatch.setitem(sys.modules, "peft", fake_peft)
        return captured

    def test_defaults_to_all_linear_when_target_modules_empty(self, monkeypatch):
        captured = self._install_fake_peft(monkeypatch)
        _build_lora_config(_fake_lora())
        assert captured["kwargs"]["target_modules"] == "all-linear"
        assert captured["kwargs"]["r"] == 64
        assert captured["kwargs"]["lora_alpha"] == 128
        assert "exclude_modules" not in captured["kwargs"]
        assert "target_parameters" not in captured["kwargs"]

    def test_uses_explicit_target_modules_when_set(self, monkeypatch):
        captured = self._install_fake_peft(monkeypatch)
        _build_lora_config(_fake_lora(target_modules=["q_proj", "v_proj"]))
        assert captured["kwargs"]["target_modules"] == ["q_proj", "v_proj"]

    def test_includes_exclude_modules_when_set(self, monkeypatch):
        captured = self._install_fake_peft(monkeypatch)
        _build_lora_config(_fake_lora(exclude_modules=["router.proj"]))
        assert captured["kwargs"]["exclude_modules"] == ["router.proj"]

    def test_exclude_modules_regex_string_passed_through_unwrapped(self, monkeypatch):
        """A regex string must reach PEFT unchanged, not shredded into

        individual characters by an errant list(...) wrap — PEFT's list
        form only matches an exact trailing path segment and can't exclude
        an entire subtree (e.g. a nested vision tower); the string form
        (re.fullmatch) is required for that.
        """
        captured = self._install_fake_peft(monkeypatch)
        regex = ".*(vision_tower|vision_adapter).*"
        _build_lora_config(_fake_lora(exclude_modules=regex))
        assert captured["kwargs"]["exclude_modules"] == regex

    def test_includes_target_parameters_for_moe_experts(self, monkeypatch):
        captured = self._install_fake_peft(monkeypatch)
        _build_lora_config(
            _fake_lora(target_parameters=["experts.gate_up_proj", "experts.down_proj"])
        )
        assert captured["kwargs"]["target_parameters"] == [
            "experts.gate_up_proj",
            "experts.down_proj",
        ]


class TestReconstructFullStateDict:
    """reconstruct_full_state_dict with fake

    verl.model_merger.{base_model_merger,fsdp_model_merger} modules —
    exercises the real shard-reconstruction call chain wiring without verl
    installed.
    """

    def test_wires_the_shard_reconstruction_call_chain(self, monkeypatch, tmp_path):
        checkpoint_dir = str(tmp_path / "global_step_10")
        captured: dict = {}

        class FakeModelMergerConfig:
            def __init__(self, **kwargs):
                captured["config_kwargs"] = kwargs

        fake_merger = MagicMock()
        fake_merger._get_world_size.return_value = 16
        fake_merger._load_rank_zero_state_dict.return_value = {"rank0": "state"}
        fake_merger._extract_device_mesh_info.return_value = ("mesh", ["dp", "fsdp"])
        fake_merger._calculate_shard_configuration.return_value = (16, (2, 8))
        fake_merger._load_and_merge_state_dicts.return_value = {"full": "state_dict"}

        def fake_fsdp_model_merger(config):
            captured["merger_config"] = config
            return fake_merger

        base_mod: Any = ModuleType("verl.model_merger.base_model_merger")
        base_mod.ModelMergerConfig = FakeModelMergerConfig
        fsdp_mod: Any = ModuleType("verl.model_merger.fsdp_model_merger")
        fsdp_mod.FSDPModelMerger = fake_fsdp_model_merger

        monkeypatch.setitem(
            sys.modules, "verl.model_merger.base_model_merger", base_mod
        )
        monkeypatch.setitem(
            sys.modules, "verl.model_merger.fsdp_model_merger", fsdp_mod
        )

        result = reconstruct_full_state_dict(checkpoint_dir, trust_remote_code=False)

        assert result == {"full": "state_dict"}
        assert captured["config_kwargs"] == {
            "operation": "merge",
            "backend": "fsdp",
            "local_dir": checkpoint_dir,
            "hf_model_config_path": os.path.join(checkpoint_dir, "huggingface"),
            "trust_remote_code": False,
        }
        fake_merger._load_rank_zero_state_dict.assert_called_once_with(16)
        fake_merger._extract_device_mesh_info.assert_called_once_with(
            {"rank0": "state"}, 16
        )
        fake_merger._calculate_shard_configuration.assert_called_once_with(
            "mesh", ["dp", "fsdp"]
        )
        fake_merger._load_and_merge_state_dicts.assert_called_once_with(
            16, 16, (2, 8), ["dp", "fsdp"]
        )


class TestMergeLoraAndSave:
    """merge_lora_and_save with fake peft/transformers modules — exercises

    the real load-non-strict-then-merge-then-save sequence, including the
    "refuse to save a broken model" guard, without those packages
    installed. torch is a real dependency here already (used directly, not
    faked). No accelerate/init_empty_weights involved any more — the base
    model is loaded with real weights via from_pretrained, not built empty
    on the meta device (see the module docstring for why: the checkpoint's
    state dict is adapter-only under the default save_lora_only=True and
    never contains the frozen base weights at all).
    """

    def _install_fakes(
        self, monkeypatch, *, missing_keys=(), unexpected_keys=(), hf_config=None
    ):
        state: dict = {}

        fake_transformers: Any = ModuleType("transformers")
        fake_base_model = MagicMock()
        fake_tokenizer = MagicMock()
        fake_hf_config = hf_config if hf_config is not None else SimpleNamespace()

        fake_transformers.AutoConfig = SimpleNamespace(
            from_pretrained=MagicMock(return_value=fake_hf_config)
        )
        fake_transformers.AutoTokenizer = SimpleNamespace(
            from_pretrained=MagicMock(return_value=fake_tokenizer)
        )

        # get_hf_auto_model_class is the SAME dispatch verl's own FSDP
        # training path uses (see merge_lora_and_save's docstring) — faked
        # here to return a distinct mock class per hf_config, so tests can
        # assert the merge path actually consulted it rather than
        # hardcoding a single AutoModel class.
        fake_auto_model_class = SimpleNamespace(
            from_pretrained=MagicMock(return_value=fake_base_model)
        )
        fake_verl_utils_model: Any = ModuleType("verl.utils.model")
        fake_verl_utils_model.get_hf_auto_model_class = MagicMock(
            return_value=fake_auto_model_class
        )
        fake_verl_pkg: Any = ModuleType("verl")
        fake_verl_utils_pkg: Any = ModuleType("verl.utils")

        fake_peft: Any = ModuleType("peft")
        fake_lora_config_obj = SimpleNamespace(name="fake-lora-config")
        fake_peft.LoraConfig = MagicMock(return_value=fake_lora_config_obj)

        fake_peft_model = MagicMock()
        fake_peft_model.load_state_dict.return_value = SimpleNamespace(
            missing_keys=list(missing_keys), unexpected_keys=list(unexpected_keys)
        )
        fake_merged_model = MagicMock()
        fake_peft_model.merge_and_unload.return_value = fake_merged_model
        fake_peft.get_peft_model = MagicMock(return_value=fake_peft_model)

        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
        monkeypatch.setitem(sys.modules, "peft", fake_peft)
        monkeypatch.setitem(sys.modules, "verl", fake_verl_pkg)
        monkeypatch.setitem(sys.modules, "verl.utils", fake_verl_utils_pkg)
        monkeypatch.setitem(sys.modules, "verl.utils.model", fake_verl_utils_model)

        state.update(
            base_model=fake_base_model,
            tokenizer=fake_tokenizer,
            peft_model=fake_peft_model,
            merged_model=fake_merged_model,
            hf_config=fake_hf_config,
            auto_config_from_pretrained=fake_transformers.AutoConfig.from_pretrained,
            get_hf_auto_model_class=fake_verl_utils_model.get_hf_auto_model_class,
            auto_model_from_pretrained=fake_auto_model_class.from_pretrained,
        )
        return state

    def test_success_path_merges_and_saves_model_and_tokenizer(
        self, monkeypatch, tmp_path
    ):
        fakes = self._install_fakes(monkeypatch)
        state_dict = {"base_model.model.layers.0.self_attn.q_proj.lora_A.weight": "t"}
        output_dir = str(tmp_path / "merged")

        merge_lora_and_save(state_dict, "/models/gemma4", _fake_lora(), output_dir)

        fakes["auto_config_from_pretrained"].assert_called_once()
        assert (
            fakes["auto_config_from_pretrained"].call_args.args[0] == "/models/gemma4"
        )
        fakes["get_hf_auto_model_class"].assert_called_once_with(fakes["hf_config"])
        fakes["auto_model_from_pretrained"].assert_called_once()
        assert fakes["auto_model_from_pretrained"].call_args.args[0] == "/models/gemma4"
        fakes["peft_model"].load_state_dict.assert_called_once_with(
            state_dict, strict=False
        )
        fakes["peft_model"].merge_and_unload.assert_called_once_with()
        fakes["merged_model"].save_pretrained.assert_called_once_with(output_dir)
        fakes["tokenizer"].save_pretrained.assert_called_once_with(output_dir)

    def test_uses_get_hf_auto_model_class_dispatch_not_a_hardcoded_class(
        self, monkeypatch, tmp_path
    ):
        """Regression guard: a composite VLM config (e.g. MuseGlimmerConfig)

        is registered under AutoModelForImageTextToText, not
        AutoModelForCausalLM — hardcoding the latter raises
        'Unrecognized configuration class' immediately for such a model.
        Confirms the base model is loaded via whatever class
        get_hf_auto_model_class(hf_config) returns for THIS config, not a
        fixed transformers.AutoModelForCausalLM reference.
        """
        vlm_hf_config = SimpleNamespace(model_type="muse_glimmer")
        fakes = self._install_fakes(monkeypatch, hf_config=vlm_hf_config)
        output_dir = str(tmp_path / "merged")

        merge_lora_and_save({}, "/models/muse-glimmer", _fake_lora(), output_dir)

        fakes["get_hf_auto_model_class"].assert_called_once_with(vlm_hf_config)
        fakes["auto_model_from_pretrained"].assert_called_once()
        assert (
            fakes["auto_model_from_pretrained"].call_args.args[0]
            == "/models/muse-glimmer"
        )

    def test_missing_non_adapter_keys_are_fine(self, monkeypatch, tmp_path):
        """Missing keys for the model's OWN (frozen, already-loaded-for-real)

        base weights are expected and must not raise — that's the entire
        point of not treating the checkpoint's adapter-only state dict as
        complete.
        """
        fakes = self._install_fakes(
            monkeypatch,
            missing_keys=["base_model.model.layers.0.self_attn.q_proj.weight"],
        )
        output_dir = str(tmp_path / "merged")

        merge_lora_and_save({}, "/models/gemma4", _fake_lora(), output_dir)

        fakes["peft_model"].merge_and_unload.assert_called_once_with()
        fakes["merged_model"].save_pretrained.assert_called_once_with(output_dir)

    def test_raises_when_adapter_keys_are_missing(self, monkeypatch, tmp_path):
        fakes = self._install_fakes(
            monkeypatch,
            missing_keys=["base_model.model.layers.0.self_attn.q_proj.lora_A.weight"],
        )
        output_dir = str(tmp_path / "merged")

        with pytest.raises(RuntimeError, match="missing_adapter_keys=.*lora_A"):
            merge_lora_and_save({}, "/models/gemma4", _fake_lora(), output_dir)

        fakes["peft_model"].merge_and_unload.assert_not_called()
        fakes["merged_model"].save_pretrained.assert_not_called()

    def test_raises_when_state_dict_has_unexpected_keys(self, monkeypatch, tmp_path):
        fakes = self._install_fakes(monkeypatch, unexpected_keys=["extra.weight"])
        output_dir = str(tmp_path / "merged")

        with pytest.raises(RuntimeError, match="unexpected=.*extra.weight"):
            merge_lora_and_save({}, "/models/gemma4", _fake_lora(), output_dir)

        fakes["peft_model"].merge_and_unload.assert_not_called()
        fakes["merged_model"].save_pretrained.assert_not_called()


def test_module_imports_without_verl_or_peft_installed():
    import foresightrlcore.methods.verl_checkpoint_merge as mod

    assert callable(mod.reconstruct_full_state_dict)
    assert callable(mod.merge_lora_and_save)


def test_reconstruct_full_state_dict_fails_with_clear_import_error_when_verl_missing(
    tmp_path,
):
    try:
        import verl  # noqa: F401

        pytest.skip("verl is installed; lazy-import boundary not exercised")
    except ImportError:
        pass

    with pytest.raises(ImportError):
        reconstruct_full_state_dict(str(tmp_path))


def test_merge_lora_and_save_fails_with_clear_import_error_when_peft_missing(
    tmp_path,
):
    try:
        import peft  # noqa: F401
        import transformers  # noqa: F401

        pytest.skip(
            "peft/transformers are installed; lazy-import boundary not exercised"
        )
    except ImportError:
        pass

    class _FakeLora:
        r = 8
        alpha = 16
        target_modules: list = []
        exclude_modules: list = []
        target_parameters: list = []

    with pytest.raises(ImportError):
        merge_lora_and_save({}, str(tmp_path), _FakeLora(), str(tmp_path))


def test_build_lora_config_fails_with_clear_import_error_when_peft_missing():
    try:
        import peft  # noqa: F401

        pytest.skip("peft is installed; lazy-import boundary not exercised")
    except ImportError:
        pass

    class _FakeLora:
        r = 8
        alpha = 16
        target_modules: list = []
        exclude_modules: list = []
        target_parameters: list = []

    with pytest.raises(ImportError):
        _build_lora_config(_FakeLora())
