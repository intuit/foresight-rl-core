"""Tests for foresightrlcore.methods.verl_compat — verl-version-specific workarounds."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from foresightrlcore.methods.verl_compat import (
    patch_muse_glimmer_tie_word_embeddings,
    patch_verl_lora_dropout,
)


class TestPatchVerlLoraDropout:
    """patch_verl_lora_dropout works around verl 0.9.1's FSDP LoRA

    construction hardcoding a LoraConfig dict with no ``lora_dropout`` key
    (see the function's docstring). peft/verl aren't installed in this
    environment, so both are faked in sys.modules — the fakes reproduce the
    exact shape that matters: a real ``peft.LoraConfig``-like base class
    (so ``isinstance``/``super().__init__`` behave like the real thing) and
    a module binding ``LoraConfig`` as a bare name the same way verl's
    ``transformer_impl`` does, referenced by a function that mirrors verl's
    own dict-building call.
    """

    def _install_fake_peft_and_transformer_impl(self, monkeypatch):
        class FakeLoraConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.lora_dropout = kwargs.get("lora_dropout", 0.0)

        peft_mod: Any = ModuleType("peft")
        peft_mod.LoraConfig = FakeLoraConfig

        transformer_impl_mod: Any = ModuleType(
            "verl.workers.engine.fsdp.transformer_impl"
        )
        transformer_impl_mod.LoraConfig = FakeLoraConfig

        def build_lora_config(**kwargs):
            # Mirrors _build_lora_module's dict construction: no
            # lora_dropout key at all.
            return transformer_impl_mod.LoraConfig(**kwargs)

        transformer_impl_mod.build_lora_config = build_lora_config

        engine_pkg: Any = ModuleType("verl.workers.engine")
        fsdp_pkg: Any = ModuleType("verl.workers.engine.fsdp")

        monkeypatch.setitem(sys.modules, "peft", peft_mod)
        monkeypatch.setitem(sys.modules, "verl.workers.engine", engine_pkg)
        monkeypatch.setitem(sys.modules, "verl.workers.engine.fsdp", fsdp_pkg)
        monkeypatch.setitem(
            sys.modules,
            "verl.workers.engine.fsdp.transformer_impl",
            transformer_impl_mod,
        )
        return transformer_impl_mod, FakeLoraConfig

    def test_defaults_lora_dropout_when_caller_omits_it(self, monkeypatch):
        transformer_impl_mod, _ = self._install_fake_peft_and_transformer_impl(
            monkeypatch
        )

        before = transformer_impl_mod.build_lora_config(r=8, lora_alpha=16)
        assert before.lora_dropout == 0.0

        patch_verl_lora_dropout(0.05)

        after = transformer_impl_mod.build_lora_config(r=8, lora_alpha=16)
        assert after.lora_dropout == 0.05
        assert after.kwargs["r"] == 8

    def test_does_not_override_an_explicitly_passed_dropout(self, monkeypatch):
        transformer_impl_mod, _ = self._install_fake_peft_and_transformer_impl(
            monkeypatch
        )

        patch_verl_lora_dropout(0.05)

        result = transformer_impl_mod.build_lora_config(r=8, lora_dropout=0.2)
        assert result.lora_dropout == 0.2

    def test_raises_if_transformer_impl_lora_config_is_not_peft_lora_config(
        self, monkeypatch
    ):
        transformer_impl_mod, _ = self._install_fake_peft_and_transformer_impl(
            monkeypatch
        )

        class SomeOtherClass:
            pass

        transformer_impl_mod.LoraConfig = SomeOtherClass

        with pytest.raises(RuntimeError, match="Cannot patch verl's LoRA dropout"):
            patch_verl_lora_dropout(0.05)


class TestPatchMuseGlimmerTieWordEmbeddings:
    """patch_muse_glimmer_tie_word_embeddings works around

    MuseGlimmerConfig not exposing ``tie_word_embeddings`` at the top level
    (only on the nested ``text_config``) — see the function's docstring.
    transformers isn't installed in this dev/test environment, so a fake
    ``MuseGlimmerConfig`` is installed in sys.modules reproducing the exact
    shape that matters: no ``__getattr__`` of its own, a real
    ``AttributeError`` for any attribute genuinely missing from
    ``__dict__``, and a nested ``text_config.tie_word_embeddings``.
    """

    def _install_fake_muse_glimmer_config(self, monkeypatch):
        class FakeTextConfig:
            def __init__(self, tie_word_embeddings):
                self.tie_word_embeddings = tie_word_embeddings

        class FakeMuseGlimmerConfig:
            def __init__(self, tie_word_embeddings=False):
                self.model_type = "muse_glimmer"
                self.text_config = FakeTextConfig(tie_word_embeddings)

        module_name = "transformers.models.muse_glimmer.configuration_muse_glimmer"
        fake_module: Any = ModuleType(module_name)
        fake_module.MuseGlimmerConfig = FakeMuseGlimmerConfig
        monkeypatch.setitem(sys.modules, module_name, fake_module)
        return FakeMuseGlimmerConfig

    def test_top_level_attribute_missing_before_patch(self, monkeypatch):
        FakeMuseGlimmerConfig = self._install_fake_muse_glimmer_config(monkeypatch)
        cfg = FakeMuseGlimmerConfig(tie_word_embeddings=False)
        with pytest.raises(AttributeError):
            cfg.tie_word_embeddings

    def test_delegates_to_text_config_after_patch(self, monkeypatch):
        FakeMuseGlimmerConfig = self._install_fake_muse_glimmer_config(monkeypatch)
        patch_muse_glimmer_tie_word_embeddings()

        cfg = FakeMuseGlimmerConfig(tie_word_embeddings=False)
        assert cfg.tie_word_embeddings is False

        cfg2 = FakeMuseGlimmerConfig(tie_word_embeddings=True)
        assert cfg2.tie_word_embeddings is True

    def test_normal_attributes_still_resolve_after_patch(self, monkeypatch):
        FakeMuseGlimmerConfig = self._install_fake_muse_glimmer_config(monkeypatch)
        patch_muse_glimmer_tie_word_embeddings()

        cfg = FakeMuseGlimmerConfig()
        assert cfg.model_type == "muse_glimmer"

    def test_unrelated_missing_attribute_still_raises_after_patch(self, monkeypatch):
        FakeMuseGlimmerConfig = self._install_fake_muse_glimmer_config(monkeypatch)
        patch_muse_glimmer_tie_word_embeddings()

        cfg = FakeMuseGlimmerConfig()
        with pytest.raises(AttributeError):
            cfg.some_attribute_that_does_not_exist

    def test_raises_if_muse_glimmer_config_already_has_getattr(self, monkeypatch):
        FakeMuseGlimmerConfig = self._install_fake_muse_glimmer_config(monkeypatch)
        FakeMuseGlimmerConfig.__getattr__ = lambda self, name: None

        with pytest.raises(RuntimeError, match="already defines its own __getattr__"):
            patch_muse_glimmer_tie_word_embeddings()
