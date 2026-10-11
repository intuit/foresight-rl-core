"""Tests for foresightrlcore.operations — train operation integration."""

from __future__ import annotations

import pytest

from foresightrlcore.config.schema import TrainingJobConfig
from foresightrlcore.operations.train import TrainOperation


class TestTrainOperation:
    """TrainOperation validation integration."""

    def test_invalid_config_raises(self, tmp_yaml):
        import yaml

        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "data": {"train_path": "/local/path"},
                "output": {"s3_path": "/local/output"},
                "execution": {"type": "aws_training_job"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))

        import foresightrlcore.execution.aws_training_job  # noqa: F401

        op = TrainOperation()
        with pytest.raises(ValueError, match="validation failed"):
            op.run(config)

    def test_valid_local_config_fails_at_launch(self, local_config_yaml):
        """A valid local config should pass validation but fail at launch
        (no GPUs available in test environment)."""
        config = TrainingJobConfig.from_yaml(local_config_yaml)

        import foresightrlcore.execution.local  # noqa: F401
        from foresightrlcore.config.validation import validate_config

        errors = validate_config(config)
        assert errors == []


class TestTrainOperationAttributes:
    """TrainOperation protocol conformance."""

    def test_name(self):
        op = TrainOperation()
        assert op.name == "train"

    def test_has_run_method(self):
        op = TrainOperation()
        assert callable(op.run)
