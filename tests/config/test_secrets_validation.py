"""Tests for secrets validation in foresightrlcore.config.validation."""

from __future__ import annotations

import yaml

from foresightrlcore.config.schema import TrainingJobConfig
from foresightrlcore.config.validation import validate_config


class TestValidateSecrets:

    def test_no_secrets_section_passes(self, minimal_config_yaml):
        config = TrainingJobConfig.from_yaml(minimal_config_yaml)
        errors = validate_config(config)
        # Only check that no secrets-related errors are present
        secrets_errors = [e for e in errors if "secret" in e.lower()]
        assert secrets_errors == []

    def test_valid_idps_secrets_passes(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "execution": {"type": "aws_training_job"},
                "task": {"type": "advice"},
                "secrets": {
                    "provider": "idps",
                    "idps": {
                        "policy_id": "p-test123",
                        "endpoint": "vkm.example.com",
                    },
                    "entries": [
                        {"env_var": "GENOS_APP_SECRET", "secret_path": "APP_SECRET"},
                    ],
                },
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        errors = validate_config(config)
        assert errors == []

    def test_idps_provider_without_idps_config_errors(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "execution": {"type": "aws_training_job"},
                "task": {"type": "advice"},
                "secrets": {
                    "provider": "idps",
                    "entries": [
                        {"env_var": "MY_SECRET", "secret_path": "PATH"},
                    ],
                },
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        errors = validate_config(config)
        assert any("secrets.idps" in e for e in errors)

    def test_duplicate_env_var_errors(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "execution": {"type": "aws_training_job"},
                "task": {"type": "advice"},
                "secrets": {
                    "provider": "idps",
                    "idps": {
                        "policy_id": "p-dup",
                        "endpoint": "vkm.example.com",
                    },
                    "entries": [
                        {"env_var": "SAME_VAR", "secret_path": "path1"},
                        {"env_var": "SAME_VAR", "secret_path": "path2"},
                    ],
                },
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        errors = validate_config(config)
        assert any("Duplicate" in e for e in errors)

    def test_empty_secret_path_errors(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "execution": {"type": "aws_training_job"},
                "task": {"type": "advice"},
                "secrets": {
                    "provider": "idps",
                    "idps": {
                        "policy_id": "p-test",
                        "endpoint": "vkm.example.com",
                    },
                    "entries": [
                        {"env_var": "MY_VAR", "secret_path": ""},
                    ],
                },
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        errors = validate_config(config)
        assert any("secret_path" in e for e in errors)

    def test_secret_pattern_in_env_warns(self, tmp_yaml, caplog):
        """Environment keys matching secret patterns should trigger a warning."""
        import logging

        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "execution": {"type": "aws_training_job"},
                "task": {"type": "advice"},
                "environment": {
                    "MY_API_KEY": "should-not-be-here",
                    "GENOS_APP_SECRET": "should-not-be-here",
                    "GENOS_ENV": "e2e",
                },
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))

        validation_logger = logging.getLogger("foresightrlcore.config.validation")
        validation_logger.propagate = True
        with caplog.at_level(logging.WARNING):
            validate_config(config)
        assert any("MY_API_KEY" in r.message for r in caplog.records)
        assert any("GENOS_APP_SECRET" in r.message for r in caplog.records)

    def test_local_provider_without_local_config_errors(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "data": {"train_path": "/data/train"},
                "output": {"s3_path": "/output"},
                "execution": {"type": "local", "local": {"gpus": "0,1"}},
                "task": {"type": "advice"},
                "secrets": {
                    "provider": "local",
                    "entries": [
                        {"env_var": "MY_SECRET", "secret_path": "path"},
                    ],
                },
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        errors = validate_config(config)
        assert any("secrets.local" in e for e in errors)

    def test_valid_local_secrets_passes(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "data": {"train_path": "/data/train"},
                "output": {"s3_path": "/output"},
                "execution": {"type": "local", "local": {"gpus": "0,1"}},
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
        errors = validate_config(config)
        secrets_errors = [e for e in errors if "secret" in e.lower()]
        assert secrets_errors == []
