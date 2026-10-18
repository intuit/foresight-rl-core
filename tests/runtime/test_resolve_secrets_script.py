"""Tests for foresightrlcore.runtime.scripts.resolve_secrets."""

from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import patch

import pytest
import yaml

from foresightrlcore.runtime.scripts.resolve_secrets import (
    ENV_FILE_DEFAULT,
    _resolve_idps,
    _resolve_local,
    main,
)


class TestResolveIdps:
    """IDPS is not implemented in the core. The extension does the work."""

    def test_missing_extension_raises(self):
        secrets_cfg = {
            "idps": {"policy_id": "pol-123", "endpoint": "custom.endpoint.com"},
            "entries": [
                {"env_var": "MY_SECRET", "secret_path": "/secrets/my_secret"},
            ],
        }
        blocked = {
            "foresightrl_intuit": None,
            "foresightrl_intuit.auth": None,
            "foresightrl_intuit.auth.container_secrets": None,
        }
        with patch.dict(sys.modules, blocked):
            with pytest.raises(ImportError, match="foresightrl-intuit"):
                _resolve_idps(secrets_cfg, "/tmp/does-not-matter.env")

    def test_delegates_to_extension(self):
        pytest.importorskip("foresightrl_intuit.auth.container_secrets")
        secrets_cfg = {
            "idps": {"policy_id": "pol-123", "endpoint": "custom.endpoint.com"},
            "entries": [
                {"env_var": "MY_SECRET", "secret_path": "/secrets/my_secret"},
            ],
        }
        with tempfile.NamedTemporaryFile(suffix=".env", delete=False) as handle:
            env_file = handle.name
        try:
            with patch(
                "foresightrl_intuit.auth.container_secrets.resolve_idps_to_env_file",
                return_value=1,
            ) as delegated:
                count = _resolve_idps(secrets_cfg, env_file)
            assert count == 1
            delegated.assert_called_once_with(secrets_cfg, env_file)
        finally:
            os.unlink(env_file)


class TestResolveLocal:
    def test_resolve_from_env_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as src:
            src.write("app_secret=local-secret\n")
            src_path = src.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as out:
            out_path = out.name

        try:
            secrets_cfg = {
                "local": {"env_file": src_path},
                "entries": [
                    {"env_var": "GENOS_APP_SECRET", "secret_path": "app_secret"}
                ],
            }
            count = _resolve_local(secrets_cfg, out_path)
            assert count == 1
            assert os.environ.get("GENOS_APP_SECRET") == "local-secret"
        finally:
            os.unlink(src_path)
            os.unlink(out_path)
            os.environ.pop("GENOS_APP_SECRET", None)


class TestMain:
    def test_no_secrets_section(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"model": {"name": "test"}}, f)
            path = f.name

        try:
            with patch("sys.argv", ["resolve_secrets.py", "--config", path]):
                result = main()
            assert result == 0
        finally:
            os.unlink(path)

    def test_default_env_file(self):
        assert ENV_FILE_DEFAULT == "/tmp/resolved_secrets.env"
