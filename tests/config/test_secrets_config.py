"""Tests for secrets config schema: SecretEntry, IDPSConfig, SecretsConfig."""

from __future__ import annotations

import pytest
import yaml

from foresightrlcore.config.schema import (
    IDPSConfig,
    LocalSecretsConfig,
    SecretEntry,
    SecretsConfig,
    TrainingJobConfig,
)


class TestSecretEntry:

    def test_valid_entry(self):
        entry = SecretEntry(env_var="GENOS_APP_SECRET", secret_path="APP_SECRET")
        assert entry.env_var == "GENOS_APP_SECRET"
        assert entry.secret_path == "APP_SECRET"

    def test_invalid_env_var_with_spaces(self):
        with pytest.raises(Exception, match="Invalid env var"):
            SecretEntry(env_var="BAD VAR", secret_path="path")

    def test_invalid_env_var_starts_with_digit(self):
        with pytest.raises(Exception, match="Invalid env var"):
            SecretEntry(env_var="1_BAD", secret_path="path")

    def test_underscore_env_var_allowed(self):
        entry = SecretEntry(env_var="_MY_SECRET", secret_path="MY_SECRET")
        assert entry.env_var == "_MY_SECRET"


class TestIDPSConfig:

    def test_requires_explicit_endpoint(self):
        cfg = IDPSConfig(
            policy_id="p-abc123",
            endpoint="vkm-e2e.example.com",
        )
        assert cfg.policy_id == "p-abc123"
        assert cfg.endpoint == "vkm-e2e.example.com"

    def test_missing_endpoint_raises(self):
        with pytest.raises(Exception):
            IDPSConfig(policy_id="p-abc123", environment="e2e")  # type: ignore[call-arg]

    def test_environment_label_preserved(self):
        cfg = IDPSConfig(
            policy_id="p-test",
            environment="prd",
            endpoint="vkm.example.com",
        )
        assert cfg.environment == "prd"


class TestSecretsConfig:

    def test_valid_full_config(self):
        cfg = SecretsConfig(
            provider="idps",
            idps=IDPSConfig(policy_id="p-abc", endpoint="vkm.example.com"),
            entries=[
                SecretEntry(env_var="GENOS_APP_SECRET", secret_path="APP_SECRET"),
            ],
        )
        assert cfg.provider == "idps"
        assert len(cfg.entries) == 1

    def test_default_provider_is_local(self):
        cfg = SecretsConfig()
        assert cfg.provider == "local"

    def test_empty_entries_allowed(self):
        cfg = SecretsConfig(
            provider="idps",
            idps=IDPSConfig(policy_id="p-x", endpoint="vkm.example.com"),
        )
        assert cfg.entries == []


class TestLocalSecretsConfig:

    def test_valid(self):
        cfg = LocalSecretsConfig(env_file="secrets/e2e.env")
        assert cfg.env_file == "secrets/e2e.env"


class TestSecretsConfigLocal:

    def test_local_provider(self):
        cfg = SecretsConfig(
            provider="local",
            local=LocalSecretsConfig(env_file="secrets/e2e.env"),
            entries=[
                SecretEntry(env_var="GENOS_APP_SECRET", secret_path="app_secret"),
            ],
        )
        assert cfg.provider == "local"
        assert cfg.local is not None
        assert cfg.local.env_file == "secrets/e2e.env"
        assert len(cfg.entries) == 1


class TestTrainingJobConfigWithSecrets:

    def test_config_without_secrets_still_works(self, minimal_config_yaml):
        config = TrainingJobConfig.from_yaml(minimal_config_yaml)
        assert config.secrets is None

    def test_config_with_secrets_section(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
                "secrets": {
                    "provider": "idps",
                    "idps": {
                        "policy_id": "p-test123",
                        "endpoint": "vkm.example.com",
                    },
                    "entries": [
                        {"env_var": "GENOS_APP_SECRET", "secret_path": "APP_SECRET"},
                        {
                            "env_var": "GENOS_EXPERIENCE_ID",
                            "secret_path": "EXPERIENCE_ID",
                        },
                    ],
                },
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        assert config.secrets is not None
        assert config.secrets.provider == "idps"
        assert config.secrets.idps.policy_id == "p-test123"  # type: ignore[union-attr]
        assert len(config.secrets.entries) == 2

    def test_secrets_roundtrip(self, tmp_yaml, tmp_path):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
                "secrets": {
                    "provider": "idps",
                    "idps": {
                        "policy_id": "p-roundtrip",
                        "endpoint": "vkm.example.com",
                    },
                    "entries": [
                        {"env_var": "MY_SECRET", "secret_path": "my/path"},
                    ],
                },
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        out = tmp_path / "roundtrip.yaml"
        config.to_yaml(out)
        config2 = TrainingJobConfig.from_yaml(out)
        assert config2.secrets.idps.policy_id == "p-roundtrip"  # type: ignore[union-attr]

    def test_config_with_local_secrets(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "data": {"train_path": "/data/train"},
                "output": {"s3_path": "/output"},
                "task": {"type": "advice"},
                "secrets": {
                    "provider": "local",
                    "local": {"env_file": "secrets/e2e.env"},
                    "entries": [
                        {"env_var": "GENOS_APP_SECRET", "secret_path": "app_secret"},
                    ],
                },
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        assert config.secrets.provider == "local"  # type: ignore[union-attr]
