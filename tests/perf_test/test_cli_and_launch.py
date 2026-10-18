"""Tests for ``foresightrlcore.perf_test.cli`` and the ``launch()`` API.

The CLI module is a thin argparse + dispatch shim; what we want to lock
down is:

* ``build_parser`` registers both subcommands on a real argparse parser
  with the expected required/optional flags.
* ``dispatch`` routes ``perf-test`` and ``validate-perf`` to the right
  internal handlers, and rejects unknown commands with a non-zero exit.
* ``_cmd_validate_perf`` reports validation errors, missing files, and
  prints a happy-path summary.
* ``_cmd_perf_test`` delegates to ``perf_test.launch`` and surfaces
  exceptions as a non-zero exit code.
* The public ``launch()`` function (in ``foresightrlcore.perf_test``)
  loads YAML, applies CLI-style overrides, and forwards to
  ``PerfTestOperation``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from foresightrlcore import perf_test as perf_test_pkg
from foresightrlcore.perf_test import cli as cli_mod


def _make_perf_yaml(path: Path, **overrides) -> Path:
    base = {
        "job": {"name": "perf-test"},
        "model": {"name": "s3://bucket/model/"},
        "perf_test": {
            "name": "run1",
            "serving": {"tensor_parallel_size": 1},
            "gatling": {"data_s3_path": "s3://bucket/data/"},
        },
        "execution": {"type": "aws_training_job"},
        "output": {"s3_path": "s3://bucket/out/"},
        "container": {"recipe": "qwen3_5"},
    }
    base.update(overrides)
    path.write_text(yaml.dump(base))
    return path


class TestBuildParser:
    def _make_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        cli_mod.build_parser(sub)
        return parser

    def test_perf_test_subcommand_registered(self):
        parser = self._make_parser()
        args = parser.parse_args(["perf-test", "--config", "x.yaml"])
        assert args.command == "perf-test"
        assert args.config == "x.yaml"
        assert args.wait is False

    def test_perf_test_wait_flag(self):
        parser = self._make_parser()
        args = parser.parse_args(["perf-test", "--config", "x.yaml", "--wait"])
        assert args.wait is True

    def test_perf_test_config_is_required(self):
        parser = self._make_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["perf-test"])

    def test_validate_perf_subcommand_registered(self):
        parser = self._make_parser()
        args = parser.parse_args(["validate-perf", "--config", "x.yaml"])
        assert args.command == "validate-perf"
        assert args.config == "x.yaml"


class TestDispatch:
    def test_dispatch_unknown_command_returns_1(self):
        args = argparse.Namespace(command="nope")
        assert cli_mod.dispatch(args) == 1

    def test_dispatch_routes_validate_perf(self, tmp_path):
        path = _make_perf_yaml(tmp_path / "c.yaml")
        args = argparse.Namespace(command="validate-perf", config=str(path))
        # Happy path returns 0 because the config validates.
        assert cli_mod.dispatch(args) == 0

    def test_dispatch_routes_perf_test(self, tmp_path):
        path = _make_perf_yaml(tmp_path / "c.yaml")
        args = argparse.Namespace(command="perf-test", config=str(path), wait=False)
        with patch.object(cli_mod, "_cmd_perf_test", return_value=0) as mock_handler:
            assert cli_mod.dispatch(args) == 0
        mock_handler.assert_called_once_with(args)


class TestValidatePerfCmd:
    def test_happy_path_logs_summary_and_returns_0(self, tmp_path, caplog):
        path = _make_perf_yaml(tmp_path / "c.yaml")
        args = argparse.Namespace(config=str(path))
        with caplog.at_level("INFO", logger="foresightrlcore.perf_test.cli"):
            assert cli_mod._cmd_validate_perf(args) == 0
        text = caplog.text
        assert "valid" in text.lower()
        assert "perf-test" in text  # job name appears
        assert "run1" in text  # run name appears

    def test_missing_file_returns_1(self, tmp_path, caplog):
        args = argparse.Namespace(config=str(tmp_path / "missing.yaml"))
        with caplog.at_level("ERROR", logger="foresightrlcore.perf_test.cli"):
            assert cli_mod._cmd_validate_perf(args) == 1
        assert "not found" in caplog.text.lower()

    def test_invalid_config_returns_1(self, tmp_path, caplog):
        # Output is required for aws_training_job; drop it to trigger errors.
        path = _make_perf_yaml(tmp_path / "c.yaml", output={})
        args = argparse.Namespace(config=str(path))
        with caplog.at_level("ERROR", logger="foresightrlcore.perf_test.cli"):
            assert cli_mod._cmd_validate_perf(args) == 1
        assert "validation failed" in caplog.text.lower()

    def test_malformed_yaml_returns_1(self, tmp_path, caplog):
        bad = tmp_path / "bad.yaml"
        bad.write_text("this: is: not: valid: yaml::::")
        args = argparse.Namespace(config=str(bad))
        with caplog.at_level("ERROR", logger="foresightrlcore.perf_test.cli"):
            # Either malformed YAML or a Pydantic validation error -- both
            # land on the broad-except branch and return 1.
            assert cli_mod._cmd_validate_perf(args) == 1


class TestPerfTestCmd:
    def test_happy_path_calls_launch_and_returns_0(self, tmp_path, caplog):
        path = _make_perf_yaml(tmp_path / "c.yaml")
        args = argparse.Namespace(config=str(path), wait=False)
        with patch.object(
            perf_test_pkg, "launch", return_value=["job-42"]
        ) as mock_launch:
            with caplog.at_level("INFO", logger="foresightrlcore.perf_test.cli"):
                assert cli_mod._cmd_perf_test(args) == 0
        mock_launch.assert_called_once_with(str(path), wait=False)
        assert "job-42" in caplog.text

    def test_launch_failure_returns_1(self, tmp_path, caplog):
        path = _make_perf_yaml(tmp_path / "c.yaml")
        args = argparse.Namespace(config=str(path), wait=False)
        with patch.object(perf_test_pkg, "launch", side_effect=RuntimeError("boom")):
            with caplog.at_level("ERROR", logger="foresightrlcore.perf_test.cli"):
                assert cli_mod._cmd_perf_test(args) == 1
        assert "boom" in caplog.text


class TestLaunchApi:
    def test_launch_invokes_operation_run(self, tmp_path):
        path = _make_perf_yaml(tmp_path / "c.yaml")

        with patch(
            "foresightrlcore.operations.perf_test.PerfTestOperation"
        ) as mock_op_cls:
            mock_op = MagicMock()
            mock_op.run.return_value = ["job-xyz"]
            mock_op_cls.return_value = mock_op
            result = perf_test_pkg.launch(str(path), wait=False)

        assert result == ["job-xyz"]
        mock_op_cls.assert_called_once()
        mock_op.run.assert_called_once()
        call_kwargs = mock_op.run.call_args.kwargs
        assert call_kwargs.get("wait") is False

    def test_launch_applies_dotted_overrides(self, tmp_path):
        path = _make_perf_yaml(tmp_path / "c.yaml")

        captured = {}

        def fake_run(config, wait=False):
            captured["model_name"] = config.model.name
            return ["job-1"]

        with patch(
            "foresightrlcore.operations.perf_test.PerfTestOperation"
        ) as mock_op_cls:
            mock_op_cls.return_value.run = fake_run
            perf_test_pkg.launch(
                str(path), wait=False, **{"model.name": "s3://other/model/"}
            )

        assert captured["model_name"] == "s3://other/model/"
