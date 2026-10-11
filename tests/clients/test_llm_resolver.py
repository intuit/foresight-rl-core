"""Tests for resolve_llm_manager and load_genos_credentials_from_env."""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# langchain_openai is a container-side dependency not installed locally.
# Ensure a stub is present so any lazy imports inside the clients module
# resolve. (boto3 is in the [aws] extra and is always installed.)
if "langchain_openai" not in sys.modules:
    sys.modules["langchain_openai"] = MagicMock()

# Absent from OSS checkouts — skip the module rather than fail collection.
# The bedrock-only coverage here is duplicated in test_oss_standalone.py, which
# runs everywhere, so nothing OSS-relevant is lost when this module skips.
pytest.importorskip("foresightrl_intuit")

from foresightrl_intuit.clients.genos import (  # noqa: E402
    DEFAULT_GENOS_ENV,
    GenosClientManager,
    load_genos_credentials_from_env,
)

from foresightrlcore.clients import (  # noqa: E402
    BedrockClientManager,
    LLMClientManager,
    resolve_llm_manager,
)
from foresightrlcore.config.schema import LLMClientConfig  # noqa: E402

# ---------------------------------------------------------------------------
# load_genos_credentials_from_env
# ---------------------------------------------------------------------------


_ALL_GENOS_VARS = [
    "GENOS_APP_SECRET",
    "GENOS_INTUIT_ORIGINATING_ASSETALIAS",
    "GENOS_APP_ID",
    "GENOS_EXPERIENCE_ID",
    "GENOS_OFFLINE_JOB_ID",
    "GENOS_ENV",
]


@pytest.fixture(autouse=False)
def _clean_genos_env(monkeypatch):
    for var in _ALL_GENOS_VARS:
        monkeypatch.delenv(var, raising=False)


class TestLoadGenosCredentialsFromEnv:
    def test_loads_all_present(self, _clean_genos_env, monkeypatch):
        monkeypatch.setenv("GENOS_APP_SECRET", "sec")
        monkeypatch.setenv("GENOS_INTUIT_ORIGINATING_ASSETALIAS", "alias")
        monkeypatch.setenv("GENOS_APP_ID", "app-id")
        monkeypatch.setenv("GENOS_EXPERIENCE_ID", "exp")
        monkeypatch.setenv("GENOS_OFFLINE_JOB_ID", "job")
        monkeypatch.setenv("GENOS_ENV", "prf")
        creds = load_genos_credentials_from_env()
        assert creds["app_secret"] == "sec"
        assert creds["intuit_originating_assetalias"] == "alias"
        assert creds["app_id"] == "app-id"
        assert creds["experience_id"] == "exp"
        assert creds["offline_job_id"] == "job"
        assert creds["genos_env"] == "prf"

    def test_defaults_genos_env_when_missing(self, _clean_genos_env, monkeypatch):
        monkeypatch.setenv("GENOS_APP_SECRET", "sec")
        creds = load_genos_credentials_from_env()
        assert creds["genos_env"] == DEFAULT_GENOS_ENV

    def test_missing_app_secret_raises(self, _clean_genos_env, monkeypatch):
        monkeypatch.setenv("GENOS_EXPERIENCE_ID", "exp")
        with pytest.raises(RuntimeError, match="Genos credentials not found"):
            load_genos_credentials_from_env()


# ---------------------------------------------------------------------------
# resolve_llm_manager — bedrock branch
# ---------------------------------------------------------------------------


class TestResolveLlmManagerBedrock:
    def test_returns_bedrock_manager(self, monkeypatch):
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
        cfg = LLMClientConfig.model_validate(
            {
                "provider": "bedrock",
                "bedrock": {
                    "model_id": "br-model",
                    "region": "us-west-2",
                    "temperature": 0.3,
                },
            }
        )
        mgr: LLMClientManager = resolve_llm_manager(cfg)
        assert isinstance(mgr, BedrockClientManager)
        assert mgr._model_id == "br-model"
        assert mgr._region == "us-west-2"
        assert mgr._temperature == 0.3

    def test_does_not_load_genos_credentials(self, monkeypatch):
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
        cfg = LLMClientConfig.model_validate(
            {"provider": "bedrock", "bedrock": {"model_id": "m"}}
        )
        with patch(
            "foresightrl_intuit.clients.genos.load_genos_credentials_from_env"
        ) as mock_load:
            resolve_llm_manager(cfg)
        mock_load.assert_not_called()


# ---------------------------------------------------------------------------
# resolve_llm_manager — genos branch
# ---------------------------------------------------------------------------


class TestResolveLlmManagerGenos:
    def test_returns_genos_manager_and_loads_credentials(self):
        cfg = LLMClientConfig.model_validate(
            {"provider": "genos", "genos": {"model_id": "ge-model"}}
        )
        sample_creds = {"app_secret": "sec", "genos_env": "e2e"}
        with patch(
            "foresightrl_intuit.clients.genos.load_genos_credentials_from_env",
            return_value=sample_creds,
        ) as mock_load:
            mgr = resolve_llm_manager(cfg)
        mock_load.assert_called_once_with()
        assert isinstance(mgr, GenosClientManager)
        assert mgr._model_id == "ge-model"
        assert mgr._credentials == sample_creds

    def test_propagates_missing_credentials_error(self):
        cfg = LLMClientConfig.model_validate(
            {"provider": "genos", "genos": {"model_id": "ge-model"}}
        )
        with patch(
            "foresightrl_intuit.clients.genos.load_genos_credentials_from_env",
            side_effect=RuntimeError("Genos credentials not found"),
        ):
            with pytest.raises(RuntimeError, match="Genos credentials not found"):
                resolve_llm_manager(cfg)


# ---------------------------------------------------------------------------
# Sanity: env independence
# ---------------------------------------------------------------------------


class TestEnvironmentIndependence:
    def test_bedrock_path_works_with_no_genos_env(self, monkeypatch):
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
        for var in _ALL_GENOS_VARS:
            monkeypatch.delenv(var, raising=False)
        cfg = LLMClientConfig.model_validate(
            {"provider": "bedrock", "bedrock": {"model_id": "m"}}
        )
        mgr = resolve_llm_manager(cfg)
        assert isinstance(mgr, BedrockClientManager)
        # No GENOS_* vars present — nothing went looking for them.
        for var in _ALL_GENOS_VARS:
            assert os.environ.get(var) in (None, "")
