"""Tests for foresightrl_intuit.auth.idps — IDPSProvider with mocked idps_sdk."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# The IDPS provider ships with the extension, not the OSS core (which offers
# the `local` provider). Absent from OSS checkouts — skip rather than fail.
pytest.importorskip("foresightrl_intuit")


class TestIDPSProvider:

    def _make_provider(self):
        from foresightrl_intuit.auth.idps import IDPSProvider

        return IDPSProvider(policy_id="p-test123", endpoint="vkm-e2e.example.com")

    def test_resolve_calls_idps_sdk(self):
        mock_secret = MagicMock()
        mock_secret.get_string_value.return_value = "secret-value-123"

        mock_client = MagicMock()
        mock_client.get_secret.return_value = mock_secret

        mock_factory = MagicMock()
        mock_factory.get_instance.return_value = mock_client

        mock_module = MagicMock()
        mock_module.IdpsClientFactory = mock_factory

        with patch.dict(
            sys.modules, {"idps_sdk": MagicMock(), "idps_sdk.idps_client": mock_module}
        ):
            provider = self._make_provider()
            result = provider.resolve("APP_SECRET")

        assert result == "secret-value-123"
        mock_factory.get_instance.assert_called_once_with(
            endpoint="vkm-e2e.example.com",
            policy_id="p-test123",
        )
        mock_client.get_secret.assert_called_once_with("APP_SECRET")

    def test_resolve_multiple_secrets_reuses_client(self):
        mock_secret1 = MagicMock()
        mock_secret1.get_string_value.return_value = "val1"
        mock_secret2 = MagicMock()
        mock_secret2.get_string_value.return_value = "val2"

        mock_client = MagicMock()
        mock_client.get_secret.side_effect = [mock_secret1, mock_secret2]

        mock_factory = MagicMock()
        mock_factory.get_instance.return_value = mock_client

        mock_module = MagicMock()
        mock_module.IdpsClientFactory = mock_factory

        with patch.dict(
            sys.modules, {"idps_sdk": MagicMock(), "idps_sdk.idps_client": mock_module}
        ):
            provider = self._make_provider()
            r1 = provider.resolve("SECRET_A")
            r2 = provider.resolve("SECRET_B")

        assert r1 == "val1"
        assert r2 == "val2"
        # Client factory called only once (lazy init + reuse)
        mock_factory.get_instance.assert_called_once()
        assert mock_client.get_secret.call_count == 2

    def test_missing_idps_sdk_raises_import_error(self):
        with patch.dict(sys.modules, {"idps_sdk": None, "idps_sdk.idps_client": None}):
            provider = self._make_provider()
            # Reset the cached client so it tries to import
            provider._client = None
            with pytest.raises(ImportError, match="idps-sdk"):
                provider.resolve("ANY_SECRET")
