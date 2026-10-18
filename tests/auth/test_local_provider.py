"""Tests for the local secrets provider (LocalFileProvider)."""

from __future__ import annotations

import pytest

from foresightrlcore.auth import create_provider, get_provider
from foresightrlcore.auth.local import LocalFileProvider


class TestLocalFileProvider:

    def test_reads_env_file(self, tmp_path):
        env_file = tmp_path / "secrets.env"
        env_file.write_text("app_secret=my-secret-value\nother_key=other-value\n")

        provider = LocalFileProvider(env_file=str(env_file))
        assert provider.resolve("app_secret") == "my-secret-value"
        assert provider.resolve("other_key") == "other-value"

    def test_skips_comments_and_blanks(self, tmp_path):
        env_file = tmp_path / "secrets.env"
        env_file.write_text("# This is a comment\n\napp_secret=value\n\n# end\n")

        provider = LocalFileProvider(env_file=str(env_file))
        assert provider.resolve("app_secret") == "value"

    def test_strips_whitespace(self, tmp_path):
        env_file = tmp_path / "secrets.env"
        env_file.write_text("  app_secret = my-value  \n")

        provider = LocalFileProvider(env_file=str(env_file))
        assert provider.resolve("app_secret") == "my-value"

    def test_missing_key_raises(self, tmp_path):
        env_file = tmp_path / "secrets.env"
        env_file.write_text("app_secret=value\n")

        provider = LocalFileProvider(env_file=str(env_file))
        with pytest.raises(KeyError, match="nonexistent"):
            provider.resolve("nonexistent")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LocalFileProvider(env_file=str(tmp_path / "does_not_exist.env"))

    def test_empty_value_allowed(self, tmp_path):
        env_file = tmp_path / "secrets.env"
        env_file.write_text("empty_key=\n")

        provider = LocalFileProvider(env_file=str(env_file))
        assert provider.resolve("empty_key") == ""

    def test_value_with_equals_sign(self, tmp_path):
        env_file = tmp_path / "secrets.env"
        env_file.write_text("key=val=ue=with=equals\n")

        provider = LocalFileProvider(env_file=str(env_file))
        assert provider.resolve("key") == "val=ue=with=equals"


class TestLocalProviderRegistration:

    def test_registered(self):
        cls = get_provider("local")
        assert cls is LocalFileProvider

    def test_create_provider_local(self, tmp_path):
        env_file = tmp_path / "test.env"
        env_file.write_text("app_secret=secret123\n")

        provider = create_provider(
            {
                "provider": "local",
                "local": {"env_file": str(env_file)},
                "entries": [{"env_var": "MY_SECRET", "secret_path": "app_secret"}],
            }
        )
        assert provider.resolve("app_secret") == "secret123"
