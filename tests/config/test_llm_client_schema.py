"""Tests for the LLMClientConfig / LLMClientsConfig Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from foresightrlcore.config.schema import (
    BedrockClientConfig,
    GenosClientConfig,
    LLMClientConfig,
    LLMClientsConfig,
)


class TestBedrockClientConfig:
    def test_minimal_valid(self):
        cfg = BedrockClientConfig(model_id="us.anthropic.claude-sonnet-4-5-v1:0")
        assert cfg.model_id == "us.anthropic.claude-sonnet-4-5-v1:0"
        assert cfg.region is None
        assert cfg.temperature == 0.0

    def test_full_valid(self):
        cfg = BedrockClientConfig(
            model_id="m",
            region="us-west-2",
            temperature=0.5,
        )
        assert cfg.region == "us-west-2"
        assert cfg.temperature == 0.5

    def test_temperature_out_of_range(self):
        with pytest.raises(ValidationError):
            BedrockClientConfig(model_id="m", temperature=3.0)

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            BedrockClientConfig(model_id="m", unknown_field="x")  # type: ignore[call-arg]


class TestGenosClientConfig:
    def test_minimal_valid(self):
        cfg = GenosClientConfig(model_id="anthropic.claude-opus-4-5-v1-0")
        assert cfg.model_id == "anthropic.claude-opus-4-5-v1-0"
        assert cfg.temperature == 0.0

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            GenosClientConfig(model_id="m", region="us-west-2")  # type: ignore[call-arg]


class TestLLMClientConfig:
    def test_bedrock_valid(self):
        cfg = LLMClientConfig.model_validate(
            {
                "provider": "bedrock",
                "bedrock": {"model_id": "m"},
            }
        )
        assert cfg.provider == "bedrock"
        assert cfg.bedrock is not None
        assert cfg.bedrock.model_id == "m"
        assert cfg.genos is None

    def test_genos_valid(self):
        cfg = LLMClientConfig.model_validate(
            {
                "provider": "genos",
                "genos": {"model_id": "m"},
            }
        )
        assert cfg.provider == "genos"
        assert cfg.genos is not None
        assert cfg.bedrock is None

    def test_bedrock_provider_without_subkey(self):
        with pytest.raises(ValidationError, match="bedrock"):
            LLMClientConfig.model_validate({"provider": "bedrock"})

    def test_genos_provider_without_subkey(self):
        with pytest.raises(ValidationError, match="genos"):
            LLMClientConfig.model_validate({"provider": "genos"})

    def test_unknown_provider_rejected(self):
        with pytest.raises(ValidationError):
            LLMClientConfig.model_validate(
                {"provider": "openai", "genos": {"model_id": "m"}}
            )

    def test_extra_top_level_field_rejected(self):
        with pytest.raises(ValidationError):
            LLMClientConfig.model_validate(
                {
                    "provider": "bedrock",
                    "bedrock": {"model_id": "m"},
                    "extra_field": 1,
                }
            )


class TestLLMClientsConfig:
    def test_single_client_default(self):
        clients = LLMClientsConfig.model_validate(
            {
                "default": {
                    "provider": "bedrock",
                    "bedrock": {"model_id": "m", "region": "us-west-2"},
                }
            }
        )
        cfg = clients.get("default")
        assert cfg.provider == "bedrock"
        assert cfg.bedrock is not None
        assert cfg.bedrock.region == "us-west-2"

    def test_multi_client_mixed_providers(self):
        clients = LLMClientsConfig.model_validate(
            {
                "reward_judge": {
                    "provider": "bedrock",
                    "bedrock": {"model_id": "br-model", "region": "us-west-2"},
                },
                "safety_judge": {
                    "provider": "genos",
                    "genos": {"model_id": "ge-model"},
                },
            }
        )
        reward = clients.get("reward_judge")
        safety = clients.get("safety_judge")
        assert reward.provider == "bedrock"
        assert safety.provider == "genos"
        # Iterating yields LLMClientConfig values.
        providers = sorted(c.provider for c in clients)
        assert providers == ["bedrock", "genos"]

    def test_empty_mapping_rejected(self):
        with pytest.raises(ValidationError, match="at least one"):
            LLMClientsConfig.model_validate({})

    def test_missing_subkey_propagates(self):
        with pytest.raises(ValidationError, match="bedrock"):
            LLMClientsConfig.model_validate({"default": {"provider": "bedrock"}})

    def test_get_unknown_name_raises(self):
        clients = LLMClientsConfig.model_validate(
            {
                "default": {
                    "provider": "bedrock",
                    "bedrock": {"model_id": "m"},
                }
            }
        )
        with pytest.raises(ValueError, match="reward_judge"):
            clients.get("reward_judge")

    def test_get_present_name_returns_config(self):
        clients = LLMClientsConfig.model_validate(
            {
                "custom_name": {
                    "provider": "genos",
                    "genos": {"model_id": "m"},
                }
            }
        )
        cfg = clients.get("custom_name")
        assert cfg.provider == "genos"
