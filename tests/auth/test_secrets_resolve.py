"""Tests for foresightrlcore.auth.resolve_secrets — end-to-end secret resolution."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
import yaml

from foresightrlcore.auth import create_provider, resolve_secrets


class TestResolveSecrets:

    def _write_config(self, tmp_path, secrets_cfg):
        config = {
            "job": {"name": "test"},
            "model": {"name": "m"},
            "data": {"train_path": "s3://b/d"},
            "output": {"s3_path": "s3://b/o"},
            "task": {"type": "advice"},
        }
        if secrets_cfg is not None:
            config["secrets"] = secrets_cfg
        path = tmp_path / "args.yaml"
        path.write_text(yaml.dump(config))
        return str(path)

    def test_no_secrets_section_returns_zero(self, tmp_path):
        config_path = self._write_config(tmp_path, secrets_cfg=None)
        count = resolve_secrets(config_path)
        assert count == 0

    def test_empty_entries_returns_zero(self, tmp_path):
        config_path = self._write_config(
            tmp_path,
            secrets_cfg={
                "provider": "idps",
                "idps": {"policy_id": "p-test"},
                "entries": [],
            },
        )
        count = resolve_secrets(config_path)
        assert count == 0

    def test_resolves_and_sets_env_vars(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TEST_SECRET_A", raising=False)
        monkeypatch.delenv("TEST_SECRET_B", raising=False)

        config_path = self._write_config(
            tmp_path,
            secrets_cfg={
                "provider": "idps",
                "idps": {
                    "policy_id": "p-test",
                    "endpoint": "vkm-e2e.example.com",
                },
                "entries": [
                    {"env_var": "TEST_SECRET_A", "secret_path": "path/a"},
                    {"env_var": "TEST_SECRET_B", "secret_path": "path/b"},
                ],
            },
        )

        mock_provider = MagicMock()
        mock_provider.resolve.side_effect = ["resolved-a", "resolved-b"]

        with patch("foresightrlcore.auth.create_provider", return_value=mock_provider):
            count = resolve_secrets(config_path)

        assert count == 2
        assert os.environ.get("TEST_SECRET_A") == "resolved-a"
        assert os.environ.get("TEST_SECRET_B") == "resolved-b"

        # Clean up
        monkeypatch.delenv("TEST_SECRET_A")
        monkeypatch.delenv("TEST_SECRET_B")

    def test_resolution_failure_raises(self, tmp_path):
        config_path = self._write_config(
            tmp_path,
            secrets_cfg={
                "provider": "idps",
                "idps": {"policy_id": "p-test"},
                "entries": [
                    {"env_var": "FAIL_SECRET", "secret_path": "bad/path"},
                ],
            },
        )

        mock_provider = MagicMock()
        mock_provider.resolve.side_effect = RuntimeError("IDPS unreachable")

        with patch("foresightrlcore.auth.create_provider", return_value=mock_provider):
            with pytest.raises(RuntimeError, match="IDPS unreachable"):
                resolve_secrets(config_path)


class TestCreateProvider:

    def test_unknown_provider_raises(self):
        with pytest.raises(KeyError, match="Unknown secrets provider"):
            create_provider({"provider": "nonexistent"})

    def test_idps_provider_created(self):
        mock_provider = MagicMock()
        with patch(
            "foresightrlcore.auth._PROVIDER_FACTORIES",
            {"idps": lambda cfg: mock_provider},
        ):
            result = create_provider(
                {
                    "provider": "idps",
                    "idps": {
                        "policy_id": "p-test",
                        "endpoint": "vkm.test.com",
                    },
                }
            )
        assert result is mock_provider
