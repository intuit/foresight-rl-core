"""Tests for foresightrl_intuit.clients.genos — Genos auth, LLM client, and GenosClientManager."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# langchain_openai is a container-side dependency not installed locally.
# Ensure a stub is present so the lazy import inside get_llm_client resolves.
if "langchain_openai" not in sys.modules:
    sys.modules["langchain_openai"] = MagicMock()

# Absent from OSS checkouts — skip the module rather than fail collection.
pytest.importorskip("foresightrl_intuit")

from foresightrl_intuit.clients.genos import (  # noqa: E402
    DEFAULT_GENOS_ENV,
    GenosClientManager,
    genos_service_host,
    get_genos_auth_header,
    get_llm_client,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_identity_response() -> MagicMock:
    """Build a mock requests.post response for the identity GraphQL endpoint."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "data": {
            "identitySignInInternalApplicationWithPrivateAuth": {
                "authorizationHeader": "Intuit_IAM_Authentication ticket=abc123"
            }
        }
    }
    return resp


SAMPLE_CREDENTIALS = {
    "app_secret": "secret-value",
    "intuit_originating_assetalias": "Intuit.test.alias",
    "app_id": "test-app-id",
    "experience_id": "exp-123",
    "offline_job_id": "offline-job-1",
}


# ---------------------------------------------------------------------------
# get_genos_auth_header
# ---------------------------------------------------------------------------


class TestGetGenosAuthHeader:

    @patch("requests.post", return_value=_mock_identity_response())
    def test_default_env_uses_e2e_url(self, mock_post):
        get_genos_auth_header("app", "secret")
        url = mock_post.call_args[0][0]
        assert genos_service_host("identityinternal", "e2e") in url

    @pytest.mark.parametrize("env", ["prf", "qal"])
    @patch("requests.post", return_value=_mock_identity_response())
    def test_custom_env_in_identity_url(self, mock_post, env):
        get_genos_auth_header("app", "secret", genos_env=env)
        url = mock_post.call_args[0][0]
        assert genos_service_host("identityinternal", env) in url

    @pytest.mark.parametrize("env", ["prd", "prod"])
    @patch("requests.post", return_value=_mock_identity_response())
    def test_prd_uses_bare_identity_host(self, mock_post, env):
        get_genos_auth_header("app", "secret", genos_env=env)
        url = mock_post.call_args[0][0]
        assert url.startswith(
            "https://" + genos_service_host("identityinternal", env) + "/"
        )
        assert "-prd" not in url
        assert "-prod" not in url

    @patch("requests.post", return_value=_mock_identity_response())
    def test_returns_auth_header_with_suffix(self, mock_post):
        result = get_genos_auth_header("my_app", "my_secret", genos_env="e2e")
        assert "intuit_appid=my_app" in result
        assert "intuit_app_secret=my_secret" in result
        assert "abc123" in result

    @patch("requests.post")
    def test_raises_on_non_200(self, mock_post):
        resp = MagicMock()
        resp.status_code = 401
        resp.text = "Unauthorized"
        mock_post.return_value = resp
        with pytest.raises(Exception, match="Authentication failed.*401"):
            get_genos_auth_header("app", "secret")


# ---------------------------------------------------------------------------
# get_llm_client
# ---------------------------------------------------------------------------


def _get_chat_openai_mock():
    """Return the ChatOpenAI mock from the langchain_openai stub in sys.modules."""
    return sys.modules["langchain_openai"].ChatOpenAI


class TestGetLlmClient:

    @patch(
        "foresightrl_intuit.clients.genos.get_genos_auth_header",
        return_value="auth-header",
    )
    def test_default_genos_env_e2e(self, mock_auth):
        chat_mock = _get_chat_openai_mock()
        chat_mock.reset_mock()
        creds = {**SAMPLE_CREDENTIALS}
        get_llm_client("model-1", creds)
        _, kwargs = chat_mock.call_args
        assert genos_service_host("llmexecution", "e2e") in kwargs["base_url"]

    @pytest.mark.parametrize("env", ["prf", "qal"])
    @patch(
        "foresightrl_intuit.clients.genos.get_genos_auth_header",
        return_value="auth-header",
    )
    def test_custom_genos_env_in_base_url(self, mock_auth, env):
        chat_mock = _get_chat_openai_mock()
        chat_mock.reset_mock()
        creds = {**SAMPLE_CREDENTIALS, "genos_env": env}
        get_llm_client("model-1", creds)
        _, kwargs = chat_mock.call_args
        assert genos_service_host("llmexecution", env) in kwargs["base_url"]

    @pytest.mark.parametrize("env", ["prd", "prod"])
    @patch(
        "foresightrl_intuit.clients.genos.get_genos_auth_header",
        return_value="auth-header",
    )
    def test_prd_uses_bare_llm_host(self, mock_auth, env):
        chat_mock = _get_chat_openai_mock()
        chat_mock.reset_mock()
        creds = {**SAMPLE_CREDENTIALS, "genos_env": env}
        get_llm_client("model-1", creds)
        _, kwargs = chat_mock.call_args
        assert kwargs["base_url"].startswith(
            "https://" + genos_service_host("llmexecution", env) + "/v3/"
        )
        assert "llmexecution-prd" not in kwargs["base_url"]
        assert "llmexecution-prod" not in kwargs["base_url"]

    @patch(
        "foresightrl_intuit.clients.genos.get_genos_auth_header",
        return_value="auth-header",
    )
    def test_model_id_in_base_url(self, mock_auth):
        chat_mock = _get_chat_openai_mock()
        chat_mock.reset_mock()
        creds = {**SAMPLE_CREDENTIALS, "genos_env": "prf"}
        get_llm_client("anthropic.claude-opus-4-5-20251101-v1-0", creds)
        _, kwargs = chat_mock.call_args
        assert "anthropic.claude-opus-4-5-20251101-v1-0" in kwargs["base_url"]

    @patch(
        "foresightrl_intuit.clients.genos.get_genos_auth_header",
        return_value="auth-header",
    )
    def test_passes_genos_env_to_auth(self, mock_auth):
        chat_mock = _get_chat_openai_mock()
        chat_mock.reset_mock()
        creds = {**SAMPLE_CREDENTIALS, "genos_env": "prf"}
        get_llm_client("model-1", creds)
        _, kwargs = mock_auth.call_args
        assert kwargs["genos_env"] == "prf"


# ---------------------------------------------------------------------------
# GenosClientManager
# ---------------------------------------------------------------------------


class TestGenosClientManager:

    @patch("foresightrl_intuit.clients.genos.get_llm_client")
    def test_get_creates_client(self, mock_get_client):
        mock_get_client.return_value = MagicMock()
        mgr = GenosClientManager("model-1", SAMPLE_CREDENTIALS)
        client = mgr.get()
        assert client is not None
        mock_get_client.assert_called_once()

    @patch("foresightrl_intuit.clients.genos.get_llm_client")
    def test_get_caches_within_ttl(self, mock_get_client):
        mock_get_client.return_value = MagicMock()
        mgr = GenosClientManager("model-1", SAMPLE_CREDENTIALS)
        c1 = mgr.get()
        c2 = mgr.get()
        assert c1 is c2
        assert mock_get_client.call_count == 1

    @patch("foresightrl_intuit.clients.genos.get_llm_client")
    def test_force_refresh_creates_new_client(self, mock_get_client):
        first_client = MagicMock()
        second_client = MagicMock()
        mock_get_client.side_effect = [first_client, second_client]
        mgr = GenosClientManager("model-1", SAMPLE_CREDENTIALS)
        c1 = mgr.get()
        c2 = mgr.force_refresh()
        assert c1 is not c2
        assert mock_get_client.call_count == 2

    @patch("foresightrl_intuit.clients.genos.get_llm_client")
    @patch("foresightrl_intuit.clients.genos.time.time")
    def test_refreshes_after_ttl_expires(self, mock_time_fn, mock_get_client):
        first_client = MagicMock()
        second_client = MagicMock()
        mock_get_client.side_effect = [first_client, second_client]

        mock_time_fn.return_value = 0.0
        mgr = GenosClientManager("model-1", SAMPLE_CREDENTIALS)
        c1 = mgr.get()
        mock_time_fn.return_value = 1000.0
        c2 = mgr.get()  # time=1000 > TTL=900
        assert c1 is not c2
        assert mock_get_client.call_count == 2


# ---------------------------------------------------------------------------
# genos_service_host / DEFAULT_GENOS_ENV
# ---------------------------------------------------------------------------


class TestGenosServiceHost:

    @pytest.mark.parametrize("env", ["e2e", "prf", "qal"])
    def test_preprod_keeps_env_infix(self, env):
        host = genos_service_host("identityinternal", env)
        bare = genos_service_host("identityinternal", "prd")
        suffix = bare.split(".", 1)[1]
        assert host == f"identityinternal-{env}.{suffix}"
        assert genos_service_host("llmexecution", env) == f"llmexecution-{env}.{suffix}"

    @pytest.mark.parametrize("env", ["prd", "prod", "PRD", "  Prod  "])
    def test_prd_uses_bare_hosts(self, env):
        assert genos_service_host("identityinternal", env) == genos_service_host(
            "identityinternal", "prd"
        )
        assert genos_service_host("llmexecution", env) == genos_service_host(
            "llmexecution", "prd"
        )
        assert "-" not in genos_service_host("identityinternal", env).split(".", 1)[0]

    def test_empty_env_defaults_to_e2e(self):
        assert genos_service_host("llmexecution", "") == genos_service_host(
            "llmexecution", "e2e"
        )


class TestDefaults:

    def test_default_genos_env_is_e2e(self):
        assert DEFAULT_GENOS_ENV == "e2e"

    @patch(
        "foresightrl_intuit.clients.genos.get_genos_auth_header", return_value="auth"
    )
    def test_missing_genos_env_in_creds_falls_back_to_default(self, mock_auth):
        chat_mock = _get_chat_openai_mock()
        chat_mock.reset_mock()
        creds = {k: v for k, v in SAMPLE_CREDENTIALS.items() if k != "genos_env"}
        get_llm_client("model-1", creds)
        _, kwargs = mock_auth.call_args
        assert kwargs["genos_env"] == "e2e"
