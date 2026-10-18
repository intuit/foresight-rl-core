"""OptimizerOperation dispatch + CLI subcommand wiring."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

import foresightrlcore.operations.optimizer  # noqa: F401  (registers operation)
from foresightrlcore.config.optimizer_schema import OptimizerJobConfig
from foresightrlcore.operations import get_operation
from foresightrlcore.operations.optimizer import OptimizerOperation
from foresightrlcore.optimizer import cli as optimizer_cli


def _config(backend: str = "aws_training_job", **over):
    d = {
        "job": {"name": "opt"},
        "optimizer": {
            "datasets": [{"name": "bts_parquet", "path": "s3://b/data.parquet"}],
            "prompt_set": "bts_v1",
            "prompts_path": "s3://b/prompts/bts_v1_prompts.yaml",
            "models": [{"name": "gemini", "provider": "genos"}],
            "modes": ["summary_state"],
            "judges": [{"name": "length", "weight": 0}],
        },
        "execution": {"type": backend},
        "output": {"s3_path": "s3://b/out/"},
        "container": {"recipe": "qwen3_5"},
    }
    d.update(over)
    return OptimizerJobConfig.model_validate(d)


class TestOperation:
    def test_registered(self):
        assert get_operation("optimizer") is OptimizerOperation

    def test_invalid_config_raises(self):
        config = _config(output={"s3_path": "/local/not-s3"})
        with pytest.raises(ValueError, match="validation failed"):
            OptimizerOperation().run(config)

    def test_aws_backend_dispatches_to_launcher(self):
        config = _config("aws_training_job")
        fake_launcher = MagicMock()
        fake_launcher.launch.return_value = "opt-job-123"
        with patch(
            "foresightrlcore.execution.optimizer_launcher.OptimizerSageMakerLauncher",
            return_value=fake_launcher,
        ) as ctor:
            result = OptimizerOperation().run(config, wait=True)
        ctor.assert_called_once()
        fake_launcher.launch.assert_called_once_with(wait=True)
        assert result == ["opt-job-123"]

    def test_local_backend_dispatches_to_runner(self):
        config = _config("local")
        with patch("foresightrlcore.optimizer.runner.run_optimizer") as run_optimizer:
            result = OptimizerOperation().run(config)
        run_optimizer.assert_called_once()
        assert result == ["opt"]


class TestCli:
    def _parse(self, argv):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        optimizer_cli.build_parser(subparsers)
        return parser.parse_args(argv)

    def test_optimizer_subcommand_registered(self):
        args = self._parse(["optimizer", "--config", "c.yaml", "--wait"])
        assert args.command == "optimizer"
        assert args.config == "c.yaml"
        assert args.wait is True

    def test_validate_optimizer_subcommand_registered(self):
        args = self._parse(["validate-optimizer", "--config", "c.yaml"])
        assert args.command == "validate-optimizer"
        assert args.config == "c.yaml"

    def test_dispatch_optimizer_calls_launch(self):
        args = argparse.Namespace(command="optimizer", config="c.yaml", wait=False)
        with patch(
            "foresightrlcore.optimizer.launch", return_value=["job-1"]
        ) as launch:
            rc = optimizer_cli.dispatch(args)
        launch.assert_called_once_with("c.yaml", wait=False)
        assert rc == 0

    def test_dispatch_validate_success(self, tmp_path):
        config = _config("aws_training_job")
        cfg_path = tmp_path / "opt.yaml"
        config.to_yaml(cfg_path)
        args = argparse.Namespace(command="validate-optimizer", config=str(cfg_path))
        rc = optimizer_cli.dispatch(args)
        assert rc == 0

    def test_dispatch_validate_failure(self, tmp_path):
        config = _config("aws_training_job", output={"s3_path": "/local/out"})
        cfg_path = tmp_path / "bad.yaml"
        config.to_yaml(cfg_path)
        args = argparse.Namespace(command="validate-optimizer", config=str(cfg_path))
        rc = optimizer_cli.dispatch(args)
        assert rc == 1
