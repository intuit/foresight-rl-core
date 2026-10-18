"""Tests for foresightrlcore CLI."""

from __future__ import annotations

import pytest

from foresightrlcore.cli import main


class TestCli:
    """CLI subcommand tests."""

    def test_no_args_shows_help(self, capsys):
        result = main([])
        assert result == 0

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0

    def test_validate_missing_config(self):
        result = main(["validate", "--config", "/nonexistent/config.yaml"])
        assert result == 1

    def test_validate_valid_config(self, full_config_yaml):
        result = main(["validate", "--config", str(full_config_yaml)])
        assert result == 0

    def test_validate_shows_method_and_execution(self, full_config_yaml):
        result = main(["validate", "--config", str(full_config_yaml)])
        assert result == 0

    def test_validate_local_config(self, local_config_yaml):
        result = main(["validate", "--config", str(local_config_yaml)])
        assert result == 0


class TestSetupLocalGpu:
    """Tests for 'rl-train setup-local-gpu' subcommand."""

    def test_list_recipes(self):
        result = main(["setup-local-gpu", "--list"])
        assert result == 0

    def test_generate_requirements(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out = str(tmp_path / "reqs.txt")
        result = main(["setup-local-gpu", "--recipe", "qwen3_5", "--output", out])
        assert result == 0
        assert (tmp_path / "reqs.txt").exists()
        content = (tmp_path / "reqs.txt").read_text()
        assert "trl" in content

    def test_unknown_recipe_fails(self, tmp_path):
        out = str(tmp_path / "reqs.txt")
        result = main(["setup-local-gpu", "--recipe", "bogus", "--output", out])
        assert result == 1

    def test_recipe_and_list_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            main(["setup-local-gpu", "--recipe", "qwen3", "--list"])

    def test_no_args_fails(self):
        with pytest.raises(SystemExit):
            main(["setup-local-gpu"])
