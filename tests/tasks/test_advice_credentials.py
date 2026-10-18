"""Tests for Genos credential loading with GENOS_ENV support.

The credential loader was moved from ``tasks/advice.py::_load_credentials``
to the shared helper ``clients/genos.py::load_genos_credentials_from_env``
so any task (not just advice) can use the Genos provider without
copy-pasting env-var reading code.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# Stub container-side dependencies not installed locally.
for _mod in ("transformers", "langchain_openai", "datasets", "peft"):
    sys.modules.setdefault(_mod, MagicMock())

# Absent from OSS checkouts — skip the module rather than fail collection.
pytest.importorskip("foresightrl_intuit")

from foresightrl_intuit.clients.genos import (  # noqa: E402
    load_genos_credentials_from_env,
)


class TestLoadGenosCredentialsFromEnv:
    """Verify load_genos_credentials_from_env reads and defaults GENOS_ENV."""

    _ALL_GENOS_VARS = [
        "GENOS_APP_SECRET",
        "GENOS_INTUIT_ORIGINATING_ASSETALIAS",
        "GENOS_APP_ID",
        "GENOS_EXPERIENCE_ID",
        "GENOS_OFFLINE_JOB_ID",
        "GENOS_ENV",
    ]

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        """Remove any leftover GENOS_* vars so tests are isolated."""
        for var in self._ALL_GENOS_VARS:
            monkeypatch.delenv(var, raising=False)

    def _set_required_creds(self, monkeypatch):
        monkeypatch.setenv("GENOS_APP_SECRET", "test-secret")
        monkeypatch.setenv("GENOS_INTUIT_ORIGINATING_ASSETALIAS", "test-alias")
        monkeypatch.setenv("GENOS_EXPERIENCE_ID", "test-exp-id")
        monkeypatch.setenv("GENOS_OFFLINE_JOB_ID", "12345")

    def test_reads_genos_env_from_env_var(self, monkeypatch):
        self._set_required_creds(monkeypatch)
        monkeypatch.setenv("GENOS_ENV", "prf")
        creds = load_genos_credentials_from_env()
        assert creds["genos_env"] == "prf"

    def test_defaults_genos_env_to_e2e(self, monkeypatch):
        self._set_required_creds(monkeypatch)
        creds = load_genos_credentials_from_env()
        assert creds["genos_env"] == "e2e"

    def test_genos_env_prd(self, monkeypatch):
        self._set_required_creds(monkeypatch)
        monkeypatch.setenv("GENOS_ENV", "prd")
        creds = load_genos_credentials_from_env()
        assert creds["genos_env"] == "prd"

    def test_all_standard_creds_still_loaded(self, monkeypatch):
        self._set_required_creds(monkeypatch)
        monkeypatch.setenv("GENOS_ENV", "prf")
        creds = load_genos_credentials_from_env()
        assert creds["app_secret"] == "test-secret"
        assert creds["intuit_originating_assetalias"] == "test-alias"
        assert creds["experience_id"] == "test-exp-id"
        assert creds["offline_job_id"] == "12345"
        assert creds["genos_env"] == "prf"

    def test_missing_app_secret_raises(self, monkeypatch):
        monkeypatch.setenv("GENOS_ENV", "prf")
        with pytest.raises(RuntimeError, match="Genos credentials not found"):
            load_genos_credentials_from_env()
