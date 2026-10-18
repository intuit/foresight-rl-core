"""Tests for foresightrlcore.config.model_profiles — named per-model LoRA/thinking presets."""

from __future__ import annotations

import pytest
import yaml

from foresightrlcore.config.model_profiles import MODEL_PROFILES, resolve_model_profile
from foresightrlcore.config.schema import LoraConfig_, ModelConfig, TrainingJobConfig


class TestModelProfileRegistry:
    def test_gemma4_moe_profile_exists(self):
        assert "gemma4_moe" in MODEL_PROFILES

    def test_gemma4_moe_profile_fields(self):
        profile = MODEL_PROFILES["gemma4_moe"]
        assert profile["target_modules"] == []
        assert profile["exclude_modules"] == ["router.proj"]
        assert profile["target_parameters"] == [
            "experts.gate_up_proj",
            "experts.down_proj",
        ]
        assert profile["thinking_kwarg"] == "enable_thinking"
        assert profile["trust_remote_code"] is True

    def test_muse_glimmer_profile_exists(self):
        assert "muse_glimmer" in MODEL_PROFILES

    def test_muse_glimmer_profile_fields(self):
        profile = MODEL_PROFILES["muse_glimmer"]
        assert profile["target_modules"] == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        # Must be a regex STRING, not a list — see the profile's own
        # comment for why a list can't express "anywhere under this
        # prefix" (needed to exclude the nested vision tower).
        assert profile["exclude_modules"] == (
            ".*(vision_tower|vision_adapter|vision_projection).*"
        )
        assert isinstance(profile["exclude_modules"], str)
        assert profile["trust_remote_code"] is False
        assert "target_parameters" not in profile
        assert "thinking_kwarg" not in profile
        assert "attn_implementation" not in profile


class TestResolveModelProfile:
    def test_no_profile_is_a_no_op(self):
        m = ModelConfig(name="some/model")
        resolve_model_profile(m)
        assert m.lora.target_parameters == []
        assert m.thinking_kwarg is None
        assert m.trust_remote_code is None

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="Unknown model profile"):
            ModelConfig(name="some/model", profile="nonexistent")

    def test_profile_fills_unset_lora_fields(self):
        m = ModelConfig(name="google/gemma-4-26B-A4B-it", profile="gemma4_moe")
        assert m.lora.target_modules == []
        assert m.lora.exclude_modules == ["router.proj"]
        assert m.lora.target_parameters == ["experts.gate_up_proj", "experts.down_proj"]
        assert m.thinking_kwarg == "enable_thinking"
        assert m.trust_remote_code is True

    def test_muse_glimmer_fills_unset_lora_fields_with_exclude_as_string(self):
        """The profile's exclude_modules is a regex string — must resolve

        through to model.lora.exclude_modules unchanged, not shredded into
        individual characters by an errant list(...) wrap (a list of
        1-char strings would still satisfy List[str] silently).
        """
        m = ModelConfig(name="meta-models/Muse-Glimmer-30B", profile="muse_glimmer")
        assert m.lora.target_modules == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        assert m.lora.exclude_modules == (
            ".*(vision_tower|vision_adapter|vision_projection).*"
        )
        assert isinstance(m.lora.exclude_modules, str)
        assert m.lora.target_parameters == []
        assert m.trust_remote_code is False
        assert m.thinking_kwarg is None

    def test_explicit_lora_overrides_profile(self):
        """Fields explicitly set in the lora: block win over the profile,
        per-field — this is the "switch models via YAML, override what you
        need" contract."""
        m = ModelConfig(
            name="google/gemma-4-26B-A4B-it",
            profile="gemma4_moe",
            lora=LoraConfig_(exclude_modules=["router.proj", "some_other.proj"]),
        )
        # explicitly set -> profile does not overwrite it
        assert m.lora.exclude_modules == ["router.proj", "some_other.proj"]
        # not explicitly set -> profile still fills it in
        assert m.lora.target_parameters == ["experts.gate_up_proj", "experts.down_proj"]

    def test_explicit_thinking_kwarg_overrides_profile(self):
        m = ModelConfig(
            name="google/gemma-4-26B-A4B-it",
            profile="gemma4_moe",
            thinking_kwarg="reasoning_effort",
        )
        assert m.thinking_kwarg == "reasoning_effort"

    def test_explicit_trust_remote_code_overrides_profile(self):
        m = ModelConfig(
            name="google/gemma-4-26B-A4B-it",
            profile="gemma4_moe",
            trust_remote_code=False,
        )
        assert m.trust_remote_code is False


class TestModelProfileViaYaml:
    def test_profile_resolves_through_full_config(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "google/gemma-4-26B-A4B-it", "profile": "gemma4_moe"},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        assert config.model.lora.target_parameters == [
            "experts.gate_up_proj",
            "experts.down_proj",
        ]
        assert config.model.thinking_kwarg == "enable_thinking"

    def test_unknown_profile_in_yaml_raises(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m", "profile": "nonexistent"},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        with pytest.raises(Exception, match="Unknown model profile"):
            TrainingJobConfig.from_yaml(tmp_yaml(content))
