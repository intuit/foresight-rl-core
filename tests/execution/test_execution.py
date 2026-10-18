"""Tests for foresightrlcore.execution — backend protocol and validation."""

from __future__ import annotations

from unittest.mock import patch

from foresightrlcore.config.schema import TrainingJobConfig
from foresightrlcore.execution.local import LocalBackend


class TestLocalBackendValidation:
    """Local backend validate()."""

    def test_valid_local_config(self, local_config_yaml):
        config = TrainingJobConfig.from_yaml(local_config_yaml)
        backend = LocalBackend()
        with patch.object(LocalBackend, "_check_gpu_packages", return_value=[]):
            errors = backend.validate(config)
        assert errors == []

    def test_missing_gpus_error(self, tmp_yaml):
        import yaml

        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "data": {"train_path": "/data"},
                "output": {"s3_path": "/output"},
                "execution": {"type": "local"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        backend = LocalBackend()
        errors = backend.validate(config)
        assert any("gpus" in e for e in errors)

    def test_insufficient_gpus_for_server_mode(self, tmp_yaml):
        import yaml

        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "infra": {
                    "vllm_mode": "server",
                    "vllm_gpu_count": 2,
                    "training_gpu_count": 6,
                },
                "data": {"train_path": "/data"},
                "output": {"s3_path": "/output"},
                "execution": {
                    "type": "local",
                    "local": {"gpus": "0,1,2,3"},
                },
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        backend = LocalBackend()
        errors = backend.validate(config)
        assert any("Not enough GPUs" in e for e in errors)

    def test_enough_gpus_passes(self, tmp_yaml):
        import yaml

        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "infra": {
                    "vllm_mode": "server",
                    "vllm_gpu_count": 2,
                    "training_gpu_count": 6,
                },
                "data": {"train_path": "/data"},
                "output": {"s3_path": "/output"},
                "execution": {
                    "type": "local",
                    "local": {"gpus": "0,1,2,3,4,5,6,7"},
                },
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        backend = LocalBackend()
        with patch.object(LocalBackend, "_check_gpu_packages", return_value=[]):
            errors = backend.validate(config)
        assert errors == []

    def test_missing_gpu_packages_error(self, local_config_yaml):
        config = TrainingJobConfig.from_yaml(local_config_yaml)
        backend = LocalBackend()
        with patch.object(
            LocalBackend, "_check_gpu_packages", return_value=["trl", "vllm"]
        ):
            errors = backend.validate(config)
        assert any("Missing GPU packages" in e for e in errors)
        assert any("setup-local-gpu" in e for e in errors)


class TestAwsBackendValidation:
    """AWS backend validate()."""

    def test_valid_aws_config(self, full_config_yaml):
        from foresightrlcore.execution.aws_training_job import AwsTrainingJobBackend

        config = TrainingJobConfig.from_yaml(full_config_yaml)
        backend = AwsTrainingJobBackend()
        errors = backend.validate(config)
        # May fail on missing boto3, but should validate paths
        if not any("boto3" in e or "sagemaker" in e for e in errors):
            assert errors == []

    def test_aws_rejects_local_paths(self, tmp_yaml):
        import yaml

        from foresightrlcore.execution.aws_training_job import AwsTrainingJobBackend

        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "data": {"train_path": "/local/data"},
                "output": {"s3_path": "/local/output"},
                "execution": {"type": "aws_training_job"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        backend = AwsTrainingJobBackend()
        errors = backend.validate(config)
        # Either boto3/sagemaker is missing (early return), or we get path errors
        has_dep_error = any("not installed" in e for e in errors)
        has_path_error = any("s3://" in e for e in errors)
        assert has_dep_error or has_path_error

    def test_merge_lora_only_does_not_require_data_train_path(self, tmp_yaml):
        """merge_lora_only reads its checkpoint via output.checkpoint_s3_path

        (SageMaker's own CheckpointConfig sync), not an input channel, and
        legitimately has no data.train_path at all — this backend's own
        validate() must carry the same exemption as
        config/validation.py's _validate_data_required/_validate_execution,
        or a config that `rl-train validate` accepts still fails at submit
        time via TrainOperation -> backend.validate().
        """
        import yaml

        from foresightrlcore.execution.aws_training_job import AwsTrainingJobBackend

        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "method": {"type": "sft_verl"},
                "training": {"sft_verl": {"merge_lora_only": True}},
                "output": {
                    "checkpoint_s3_path": "s3://bucket/checkpoints/job",
                    "s3_path": "s3://bucket/output",
                },
                "execution": {"type": "aws_training_job"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        backend = AwsTrainingJobBackend()
        errors = backend.validate(config)
        has_dep_error = any("not installed" in e for e in errors)
        assert has_dep_error or errors == []

    def test_non_merge_sft_verl_still_requires_data_train_path(self, tmp_yaml):
        import yaml

        from foresightrlcore.execution.aws_training_job import AwsTrainingJobBackend

        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "method": {"type": "sft_verl"},
                "training": {"sft_verl": {}},
                "output": {"s3_path": "s3://bucket/output"},
                "execution": {"type": "aws_training_job"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        backend = AwsTrainingJobBackend()
        errors = backend.validate(config)
        has_dep_error = any("not installed" in e for e in errors)
        has_path_error = any("s3://" in e for e in errors)
        assert has_dep_error or has_path_error


class TestBackendProtocol:
    """Verify backends conform to the ExecutionBackend protocol."""

    def test_local_has_required_methods(self):
        backend = LocalBackend()
        assert hasattr(backend, "backend_type")
        assert hasattr(backend, "validate")
        assert hasattr(backend, "launch")
        assert hasattr(backend, "get_status")

    def test_aws_has_required_methods(self):
        from foresightrlcore.execution.aws_training_job import AwsTrainingJobBackend

        backend = AwsTrainingJobBackend()
        assert hasattr(backend, "backend_type")
        assert hasattr(backend, "validate")
        assert hasattr(backend, "launch")
        assert hasattr(backend, "get_status")
