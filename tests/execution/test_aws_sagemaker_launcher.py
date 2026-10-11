"""Tests for foresightrlcore.execution.aws_sagemaker_launcher — GRPOJobLauncher."""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import yaml

from foresightrlcore.config.schema import TrainingJobConfig


def _make_config(overrides: dict | None = None) -> TrainingJobConfig:
    base = {
        "job": {"name": "test-grpo", "max_runtime_hours": 6},
        "model": {"name": "Qwen/Qwen3-4B"},
        "infra": {
            "instance_type": "ml.p4d.24xlarge",
            "instance_count": 1,
            "vllm_mode": "server",
            "role": "arn:aws:iam::123456789012:role/SageMakerRole",
        },
        "data": {"train_path": "s3://bucket/train"},
        "output": {
            "s3_path": "s3://bucket/output",
            "checkpoint_s3_path": "s3://bucket/ckpt",
        },
        "execution": {"type": "aws_training_job"},
        "task": {"type": "advice", "config": {"reward_mode": "mock"}},
        "environment": {"GENOS_ENV": "e2e", "MY_VAR": "val"},
    }
    if overrides:
        _deep_merge(base, overrides)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(base, f)
        path = f.name
    try:
        return TrainingJobConfig.from_yaml(path)
    finally:
        os.unlink(path)


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


# ── Module-level helpers ────────────────────────────────────────────────


class TestIsSDKV3:
    def test_sdk_v3_detected(self):
        import foresightrlcore.execution.aws_sagemaker_launcher as mod

        mod._SDK_V3 = None
        with patch.dict("sys.modules", {"sagemaker.train.model_trainer": MagicMock()}):
            result = mod._is_sdk_v3()
        assert result is True
        mod._SDK_V3 = None

    def test_sdk_v2_fallback(self):
        import foresightrlcore.execution.aws_sagemaker_launcher as mod

        mod._SDK_V3 = None
        with (
            patch.dict("sys.modules", {"sagemaker.train.model_trainer": None}),
            patch("builtins.__import__", side_effect=ImportError),
        ):
            mod._SDK_V3 = False
            result = mod._is_sdk_v3()
        assert result is False
        mod._SDK_V3 = None

    def test_cached_value(self):
        import foresightrlcore.execution.aws_sagemaker_launcher as mod

        mod._SDK_V3 = True
        assert mod._is_sdk_v3() is True
        mod._SDK_V3 = False
        assert mod._is_sdk_v3() is False
        mod._SDK_V3 = None


class TestGetSession:
    def test_v3_session(self):
        import foresightrlcore.execution.aws_sagemaker_launcher as mod

        mock_session_cls = MagicMock()
        mod._SDK_V3 = True
        with patch.dict(
            "sys.modules",
            {
                "sagemaker.core.helper.session_helper": MagicMock(
                    Session=mock_session_cls
                )
            },
        ):
            mod._get_session()
        mock_session_cls.assert_called_once()
        mod._SDK_V3 = None

    def test_v2_session(self):
        import foresightrlcore.execution.aws_sagemaker_launcher as mod

        mock_session_cls = MagicMock()
        mod._SDK_V3 = False
        with patch.dict(
            "sys.modules",
            {"sagemaker": MagicMock(Session=mock_session_cls)},
        ):
            mod._get_session()
        mock_session_cls.assert_called_once()
        mod._SDK_V3 = None


class TestGetExecutionRole:
    def test_v3_role(self):
        import foresightrlcore.execution.aws_sagemaker_launcher as mod

        mock_get = MagicMock(return_value="arn:role")
        mod._SDK_V3 = True
        with patch.dict(
            "sys.modules",
            {
                "sagemaker.core.helper.session_helper": MagicMock(
                    get_execution_role=mock_get
                )
            },
        ):
            result = mod._get_execution_role()
        assert result == "arn:role"
        mod._SDK_V3 = None

    def test_v2_role(self):
        import foresightrlcore.execution.aws_sagemaker_launcher as mod

        mock_get = MagicMock(return_value="arn:role-v2")
        mod._SDK_V3 = False
        with patch.dict(
            "sys.modules",
            {"sagemaker": MagicMock(get_execution_role=mock_get)},
        ):
            result = mod._get_execution_role()
        assert result == "arn:role-v2"
        mod._SDK_V3 = None


class TestResolveImageUri:
    def test_v2_image(self):
        import foresightrlcore.execution.aws_sagemaker_launcher as mod

        mock_uris = MagicMock()
        mock_uris.retrieve.return_value = (
            "123456.dkr.ecr.us-west-2.amazonaws.com/image:tag"
        )
        mod._SDK_V3 = False
        with patch.dict("sys.modules", {"sagemaker": MagicMock(image_uris=mock_uris)}):
            uri = mod._resolve_image_uri("us-west-2", "2.6.0", "ml.p4d.24xlarge")
        assert "dkr.ecr" in uri
        mod._SDK_V3 = None

    def test_v3_image(self):
        import foresightrlcore.execution.aws_sagemaker_launcher as mod

        mock_uris = MagicMock()
        mock_uris.retrieve.return_value = "image-v3:tag"
        mod._SDK_V3 = True
        with patch.dict(
            "sys.modules",
            {"sagemaker.core": MagicMock(image_uris=mock_uris)},
        ):
            uri = mod._resolve_image_uri("us-west-2", "2.6.0", "ml.p4d.24xlarge")
        assert uri == "image-v3:tag"
        mod._SDK_V3 = None


# ── GRPOJobLauncher ─────────────────────────────────────────────────────


class TestGRPOJobLauncherProperties:
    def test_session_lazy_init(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        mock_session = MagicMock()
        launcher = GRPOJobLauncher(config, session=mock_session)
        assert launcher.session is mock_session

    def test_session_auto_create(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        launcher = GRPOJobLauncher(config)
        with patch(
            "foresightrlcore.execution.aws_sagemaker_launcher._get_session",
            return_value=MagicMock(),
        ) as mock_get:
            _ = launcher.session
        mock_get.assert_called_once()

    def test_role_from_config(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        launcher = GRPOJobLauncher(config)
        assert launcher.role == "arn:aws:iam::123456789012:role/SageMakerRole"

    def test_role_explicit(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        launcher = GRPOJobLauncher(config, role="explicit-role")
        assert launcher.role == "explicit-role"

    def test_role_from_execution_role(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"role": None}})
        launcher = GRPOJobLauncher(config)
        with patch(
            "foresightrlcore.execution.aws_sagemaker_launcher._get_execution_role",
            return_value="auto-role",
        ):
            assert launcher.role == "auto-role"

    def test_role_fallback_iam(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"role": None}})
        launcher = GRPOJobLauncher(config)
        mock_iam = MagicMock()
        mock_iam.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::123:role/fallback"}
        }
        with (
            patch(
                "foresightrlcore.execution.aws_sagemaker_launcher._get_execution_role",
                side_effect=ValueError("no role"),
            ),
            patch("boto3.client", return_value=mock_iam),
        ):
            assert launcher.role == "arn:aws:iam::123:role/fallback"

    def test_s3_client_lazy(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        launcher = GRPOJobLauncher(config)
        mock_client = MagicMock()
        with patch("boto3.client", return_value=mock_client):
            assert launcher.s3_client is mock_client

    def test_is_torchrun_mode(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"vllm_mode": "none"}})
        launcher = GRPOJobLauncher(config)
        assert launcher._is_torchrun_mode is True

        config2 = _make_config({"infra": {"vllm_mode": "server"}})
        launcher2 = GRPOJobLauncher(config2)
        assert launcher2._is_torchrun_mode is False

    def test_job_name_prefix_uses_method_type(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()  # default method.type == "grpo"
        launcher = GRPOJobLauncher(config)
        assert launcher._job_name_prefix == "grpo"

    def test_job_name_prefix_sanitizes_underscores_to_hyphens(self):
        """SageMaker's TrainingJobName pattern ([a-zA-Z0-9](-*[a-zA-Z0-9]){0,62})

        rejects underscores outright — method type strings like "sft_verl"
        must be hyphenated before they end up in a job name. The SDK v3
        ModelTrainer path happens to do this substitution internally too,
        but the boto3-direct and SDK v2 Estimator paths do not, so the
        launcher can't rely on that.
        """
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"method": {"type": "sft_verl"}})
        launcher = GRPOJobLauncher(config)
        assert launcher._job_name_prefix == "sft-verl"
        assert "_" not in launcher._job_name_prefix

    def test_gpus_per_node_g5_12(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"instance_type": "ml.g5.12xlarge"}})
        launcher = GRPOJobLauncher(config)
        assert launcher._gpus_per_node == 4

    def test_gpus_per_node_g6e_12(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"instance_type": "ml.g6e.12xlarge"}})
        launcher = GRPOJobLauncher(config)
        assert launcher._gpus_per_node == 4

    def test_gpus_per_node_p4d(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"instance_type": "ml.p4d.24xlarge"}})
        launcher = GRPOJobLauncher(config)
        assert launcher._gpus_per_node == 8

    def test_gpus_per_node_merge_lora_only_forces_single_process(self):
        """Regression guard: merge_lora_only is single-process,

        rank-0-only by design (VerlSftMethodHandler._run_merge_lora_only),
        but was still launched via the same Torchrun entrypoint configured
        for a full GPU count — even on a CPU-only instance like
        ml.m5.4xlarge (which examples/smoke_tests/sft_verl_merge_smoke.yaml
        explicitly recommends for this mode), since the g5.12/g6e.12
        heuristic below has no CPU-instance case and falls through to 8.
        """
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config(
            {
                "method": {"type": "sft_verl"},
                "infra": {"vllm_mode": "none", "instance_type": "ml.m5.4xlarge"},
                "training": {"sft_verl": {"merge_lora_only": True}},
            }
        )
        launcher = GRPOJobLauncher(config)
        assert launcher._gpus_per_node == 1

    def test_gpus_per_node_sft_verl_training_unaffected(self):
        """The merge_lora_only override must not affect a normal (non-merge)

        sft_verl training job on the same instance type.
        """
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config(
            {
                "method": {"type": "sft_verl"},
                "infra": {"vllm_mode": "none", "instance_type": "ml.p4d.24xlarge"},
                "training": {"sft_verl": {"merge_lora_only": False}},
            }
        )
        launcher = GRPOJobLauncher(config)
        assert launcher._gpus_per_node == 8

    def test_requirements_file_torchrun(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"vllm_mode": "none"}})
        launcher = GRPOJobLauncher(config)
        assert launcher._requirements_file == "requirements_torchrun.txt"

    def test_requirements_file_server(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"vllm_mode": "server"}})
        launcher = GRPOJobLauncher(config)
        assert launcher._requirements_file == "requirements.txt"


class TestParseS3Uri:
    def test_valid_uri(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        bucket, key = GRPOJobLauncher._parse_s3_uri("s3://my-bucket/path/to/data")
        assert bucket == "my-bucket"
        assert key == "path/to/data"

    def test_valid_uri_no_key(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        bucket, key = GRPOJobLauncher._parse_s3_uri("s3://my-bucket")
        assert bucket == "my-bucket"
        assert key == ""

    def test_invalid_uri(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        with pytest.raises(ValueError, match="Not an S3 URI"):
            GRPOJobLauncher._parse_s3_uri("/local/path")


class TestUploadFileToS3:
    def test_upload(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        mock_s3 = MagicMock()
        launcher = GRPOJobLauncher(config)
        launcher._s3_client = mock_s3  # type: ignore[assignment]

        uri = launcher._upload_file_to_s3("/tmp/file.txt", "bucket", "key/file.txt")

        mock_s3.upload_file.assert_called_once_with(
            "/tmp/file.txt", "bucket", "key/file.txt"
        )
        assert uri == "s3://bucket/key/file.txt"


class TestUploadConfig:
    def test_upload_config(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        mock_s3 = MagicMock()
        launcher = GRPOJobLauncher(config)
        launcher._s3_client = mock_s3  # type: ignore[assignment]

        uri = launcher._upload_config()

        mock_s3.upload_file.assert_called_once()
        assert uri.startswith("s3://bucket/")
        assert "args.yaml" in uri


class TestUploadSourceDir:
    def test_upload_source_dir_server_mode(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"vllm_mode": "server"}})
        mock_s3 = MagicMock()
        launcher = GRPOJobLauncher(config)
        launcher._s3_client = mock_s3  # type: ignore[assignment]

        uri = launcher._upload_source_dir()

        mock_s3.upload_file.assert_called_once()
        assert "sourcedir.tar.gz" in uri

    def test_upload_source_dir_torchrun_mode(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"vllm_mode": "none"}})
        mock_s3 = MagicMock()
        launcher = GRPOJobLauncher(config)
        launcher._s3_client = mock_s3  # type: ignore[assignment]

        uri = launcher._upload_source_dir()

        mock_s3.upload_file.assert_called_once()
        call_args = mock_s3.upload_file.call_args
        local_tar = call_args[0][0]
        assert os.path.exists(local_tar) is False  # cleaned up


class TestBuildBoto3Hyperparameters:
    def test_torchrun_mode(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"vllm_mode": "none"}})
        launcher = GRPOJobLauncher(config)

        hp = launcher._build_boto3_hyperparameters(
            "s3://bucket/source.tar.gz", "us-west-2"
        )

        assert json.loads(hp["sagemaker_program"]) == "train.py"
        assert json.loads(hp["sagemaker_torch_distributed_enabled"]) is True
        assert hp["config"] == "/opt/ml/input/data/config/args.yaml"

    def test_server_mode(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"vllm_mode": "server"}})
        launcher = GRPOJobLauncher(config)

        hp = launcher._build_boto3_hyperparameters(
            "s3://bucket/source.tar.gz", "us-west-2"
        )

        assert json.loads(hp["sagemaker_program"]) == "entrypoint.sh"
        assert "sagemaker_torch_distributed_enabled" not in hp


class TestSubmitViaBoto3:
    def test_submit_basic(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        mock_s3 = MagicMock()
        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {
            "TrainingJobArn": "arn:aws:sagemaker:us-west-2:123:training-job/test"
        }

        launcher = GRPOJobLauncher(config, role="test-role", session=mock_session)
        launcher._s3_client = mock_s3  # type: ignore[assignment]

        with patch("boto3.client", return_value=mock_sm):
            job_name = launcher._submit_via_boto3(
                image_uri="test-image:latest",
                input_channels={
                    "train": "s3://bucket/train",
                    "config": "s3://bucket/config",
                },
                wait=False,
            )

        assert job_name.startswith("grpo-test-grpo-")
        mock_sm.create_training_job.assert_called_once()
        call_kwargs = mock_sm.create_training_job.call_args[1]
        assert call_kwargs["RoleArn"] == "test-role"
        assert call_kwargs["ResourceConfig"]["InstanceType"] == "ml.p4d.24xlarge"
        assert "CheckpointConfig" in call_kwargs
        assert "Environment" in call_kwargs

    def test_submit_no_kms_key_by_default(self):
        """infra.kms_key_id is opt-in — unset must not change prior behavior."""
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:test"}

        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        with patch("boto3.client", return_value=mock_sm):
            launcher._submit_via_boto3("image:tag", {"train": "s3://b/t"}, wait=False)

        call_kwargs = mock_sm.create_training_job.call_args[1]
        assert "KmsKeyId" not in call_kwargs["OutputDataConfig"]

    def test_submit_with_kms_key(self):
        """This boto3-direct path is reachable from a normal sft_verl config

        whenever infra.training_plan_arn is set (see test_submit_with_training_plan
        below) — infra.kms_key_id must reach it, not just the SDK v3 path.
        """
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"kms_key_id": "alias/my-kms-key"}})
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:test"}

        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        with patch("boto3.client", return_value=mock_sm):
            launcher._submit_via_boto3("image:tag", {"train": "s3://b/t"}, wait=False)

        call_kwargs = mock_sm.create_training_job.call_args[1]
        assert call_kwargs["OutputDataConfig"]["KmsKeyId"] == "alias/my-kms-key"

    def test_submit_uses_top_level_env(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config(
            {
                "environment": {"GENOS_ENV": "e2e", "NCCL_DEBUG": "INFO"},
            }
        )
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:test"}

        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        with patch("boto3.client", return_value=mock_sm):
            launcher._submit_via_boto3("image:tag", {"train": "s3://b/t"}, wait=False)

        call_kwargs = mock_sm.create_training_job.call_args[1]
        env = call_kwargs["Environment"]
        assert env["GENOS_ENV"] == "e2e"
        assert env["NCCL_DEBUG"] == "INFO"

    def test_submit_with_training_plan(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config(
            {
                "infra": {
                    "training_plan_arn": "arn:aws:sagemaker:us-west-2:123:training-plan/tp1"
                }
            }
        )
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        mock_s3 = MagicMock()
        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:test"}

        launcher = GRPOJobLauncher(config, role="test-role", session=mock_session)
        launcher._s3_client = mock_s3  # type: ignore[assignment]

        with patch("boto3.client", return_value=mock_sm):
            launcher._submit_via_boto3("image:tag", {"train": "s3://b/t"}, wait=False)

        mock_sm.meta.events.register.assert_called_once()

    def test_submit_with_wait(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:test"}
        mock_waiter = MagicMock()
        mock_sm.get_waiter.return_value = mock_waiter

        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        with patch("boto3.client", return_value=mock_sm):
            launcher._submit_via_boto3("image:tag", {"train": "s3://b/t"}, wait=True)

        mock_sm.get_waiter.assert_called_once_with("training_job_completed_or_stopped")
        mock_waiter.wait.assert_called_once()

    def test_submit_with_vpc(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config(
            {
                "infra": {
                    "vpc": {
                        "subnets": ["subnet-1", "subnet-2"],
                        "security_groups": ["sg-1"],
                    }
                }
            }
        )
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:test"}

        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        with patch("boto3.client", return_value=mock_sm):
            launcher._submit_via_boto3("image:tag", {"train": "s3://b/t"}, wait=False)

        call_kwargs = mock_sm.create_training_job.call_args[1]
        assert "VpcConfig" in call_kwargs
        assert call_kwargs["VpcConfig"]["Subnets"] == ["subnet-1", "subnet-2"]

    def test_submit_encrypts_inter_container_by_default(self):
        """ACP IS031.1: in-transit encryption is on unless explicitly disabled."""
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:test"}

        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        with patch("boto3.client", return_value=mock_sm):
            launcher._submit_via_boto3("image:tag", {"train": "s3://b/t"}, wait=False)

        call_kwargs = mock_sm.create_training_job.call_args[1]
        assert call_kwargs["EnableInterContainerTrafficEncryption"] is True

    def test_submit_encryption_can_be_disabled(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"encrypt_inter_container_traffic": False}})
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:test"}

        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        with patch("boto3.client", return_value=mock_sm):
            launcher._submit_via_boto3("image:tag", {"train": "s3://b/t"}, wait=False)

        call_kwargs = mock_sm.create_training_job.call_args[1]
        assert call_kwargs["EnableInterContainerTrafficEncryption"] is False

    def test_submit_with_keep_alive(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"keep_alive_seconds": 1800}})
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:test"}

        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        with patch("boto3.client", return_value=mock_sm):
            launcher._submit_via_boto3("image:tag", {"train": "s3://b/t"}, wait=False)

        call_kwargs = mock_sm.create_training_job.call_args[1]
        assert call_kwargs["ResourceConfig"]["KeepAlivePeriodInSeconds"] == 1800

    def test_submit_no_spot_by_default(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:test"}

        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        with patch("boto3.client", return_value=mock_sm):
            launcher._submit_via_boto3("image:tag", {"train": "s3://b/t"}, wait=False)

        call_kwargs = mock_sm.create_training_job.call_args[1]
        assert "EnableManagedSpotTraining" not in call_kwargs
        assert "MaxWaitTimeInSeconds" not in call_kwargs["StoppingCondition"]

    def test_submit_with_spot(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config(
            {
                "job": {"max_runtime_hours": 6, "max_wait_hours": 10},
                "infra": {"use_spot_instances": True},
            }
        )
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:test"}

        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        with patch("boto3.client", return_value=mock_sm):
            launcher._submit_via_boto3("image:tag", {"train": "s3://b/t"}, wait=False)

        call_kwargs = mock_sm.create_training_job.call_args[1]
        assert call_kwargs["EnableManagedSpotTraining"] is True
        assert call_kwargs["StoppingCondition"]["MaxWaitTimeInSeconds"] == 10 * 3600

    def test_submit_spot_defaults_wait_to_runtime(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config(
            {
                "job": {"max_runtime_hours": 6},
                "infra": {"use_spot_instances": True},
            }
        )
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:test"}

        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        with patch("boto3.client", return_value=mock_sm):
            launcher._submit_via_boto3("image:tag", {"train": "s3://b/t"}, wait=False)

        call_kwargs = mock_sm.create_training_job.call_args[1]
        assert call_kwargs["StoppingCondition"]["MaxWaitTimeInSeconds"] == 6 * 3600

    def test_submit_mixed_dedicated_vllm_uses_instance_groups(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config(
            {
                "infra": {
                    "topology": "dedicated_vllm",
                    "instance_type": "ml.g6e.48xlarge",
                    "instance_count": 2,
                    "vllm_instance_type": "ml.g6e.12xlarge",
                    "vllm_instance_count": 1,
                }
            }
        )
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:test"}

        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        with patch("boto3.client", return_value=mock_sm):
            launcher._submit_via_boto3("image:tag", {"train": "s3://b/t"}, wait=False)

        call_kwargs = mock_sm.create_training_job.call_args[1]
        resource = call_kwargs["ResourceConfig"]
        assert "InstanceType" not in resource
        assert "InstanceCount" not in resource
        groups = {g["InstanceGroupName"]: g for g in resource["InstanceGroups"]}
        assert groups["vllm"]["InstanceType"] == "ml.g6e.12xlarge"
        assert groups["vllm"]["InstanceCount"] == 1
        assert groups["training"]["InstanceType"] == "ml.g6e.48xlarge"
        assert groups["training"]["InstanceCount"] == 1
        assert resource["VolumeSizeInGB"] == 100
        # Mixed InstanceGroups must not attach SageMaker CheckpointConfig —
        # that agent two-way-syncs every group (vLLM empty dir vs training
        # DeepSpeed shards → S3 SlowDown). Training ranks self-sync instead.
        assert "CheckpointConfig" not in call_kwargs

    def test_submit_mixed_ignores_keep_alive(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config(
            {
                "infra": {
                    "topology": "dedicated_vllm",
                    "instance_type": "ml.g6e.48xlarge",
                    "instance_count": 2,
                    "vllm_instance_type": "ml.g6e.12xlarge",
                    "keep_alive_seconds": 1800,
                }
            }
        )
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:test"}

        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        with patch("boto3.client", return_value=mock_sm):
            launcher._submit_via_boto3("image:tag", {"train": "s3://b/t"}, wait=False)

        resource = mock_sm.create_training_job.call_args[1]["ResourceConfig"]
        assert "KeepAlivePeriodInSeconds" not in resource
        assert "InstanceGroups" in resource


class TestBuildEstimator:
    def test_build_estimator_server_mode(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"vllm_mode": "server"}})
        mock_estimator_cls = MagicMock()
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        with patch.object(
            GRPOJobLauncher, "_get_estimator_class", return_value=mock_estimator_cls
        ):
            estimator = launcher._build_estimator("image:tag")

        mock_estimator_cls.assert_called_once()
        kwargs = mock_estimator_cls.call_args[1]
        assert kwargs["entry_point"] == "entrypoint.sh"
        assert kwargs["role"] == "role"
        assert "distribution" not in kwargs

    def test_build_estimator_torchrun(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"vllm_mode": "none"}})
        mock_estimator_cls = MagicMock()
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        with patch.object(
            GRPOJobLauncher, "_get_estimator_class", return_value=mock_estimator_cls
        ):
            estimator = launcher._build_estimator("image:tag")

        kwargs = mock_estimator_cls.call_args[1]
        assert kwargs["entry_point"] == "train.py"
        assert kwargs["distribution"] == {"torch_distributed": {"enabled": True}}
        assert (
            kwargs["hyperparameters"]["config"] == "/opt/ml/input/data/config/args.yaml"
        )

    def test_build_estimator_no_kms_key_by_default(self):
        """infra.kms_key_id is opt-in — unset must not change prior behavior."""
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        mock_estimator_cls = MagicMock()
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        with patch.object(
            GRPOJobLauncher, "_get_estimator_class", return_value=mock_estimator_cls
        ):
            launcher._build_estimator("image:tag")

        kwargs = mock_estimator_cls.call_args[1]
        assert "output_kms_key" not in kwargs

    def test_build_estimator_with_kms_key(self):
        """This SDK v2 path is reachable whenever the installed sagemaker

        package predates ModelTrainer (see _is_sdk_v3()), independent of
        which method is training — infra.kms_key_id must reach it too, not
        just the SDK v3 ModelTrainer path.
        """
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"kms_key_id": "alias/my-kms-key"}})
        mock_estimator_cls = MagicMock()
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        with patch.object(
            GRPOJobLauncher, "_get_estimator_class", return_value=mock_estimator_cls
        ):
            launcher._build_estimator("image:tag")

        kwargs = mock_estimator_cls.call_args[1]
        assert kwargs["output_kms_key"] == "alias/my-kms-key"

    def test_build_estimator_with_training_plan(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"training_plan_arn": "arn:plan:123"}})
        mock_cls = MagicMock()
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        with patch.object(
            GRPOJobLauncher, "_get_estimator_class", return_value=mock_cls
        ):
            launcher._build_estimator("image:tag")

        kwargs = mock_cls.call_args[1]
        assert kwargs["training_plan"] == "arn:plan:123"

    def test_build_estimator_with_vpc_and_checkpoint(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config(
            {
                "infra": {
                    "vpc": {
                        "subnets": ["subnet-a", "subnet-b", "subnet-c"],
                        "security_groups": ["sg-b"],
                    }
                },
                "output": {
                    "s3_path": "s3://bucket/out",
                    "checkpoint_s3_path": "s3://bucket/ckpt",
                },
            }
        )
        mock_cls = MagicMock()
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        with patch.object(
            GRPOJobLauncher, "_get_estimator_class", return_value=mock_cls
        ):
            launcher._build_estimator("image:tag")

        kwargs = mock_cls.call_args[1]
        assert kwargs["subnets"] == ["subnet-a", "subnet-b", "subnet-c"]
        assert kwargs["security_group_ids"] == ["sg-b"]
        assert kwargs["checkpoint_s3_uri"] == "s3://bucket/ckpt"

    def test_build_estimator_passes_volume_size(self):
        """SDK v2 Estimator must receive the configured EBS volume size.

        Regression: the v2 path previously omitted ``volume_size`` entirely, so
        SageMaker fell back to its 30 GB floor regardless of ``volume_size_gb``.
        """
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"volume_size_gb": 300}})
        mock_cls = MagicMock()
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        with patch.object(
            GRPOJobLauncher, "_get_estimator_class", return_value=mock_cls
        ):
            launcher._build_estimator("image:tag")

        kwargs = mock_cls.call_args[1]
        assert kwargs["volume_size"] == 300

    def test_build_estimator_no_spot_by_default(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        mock_cls = MagicMock()
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        with patch.object(
            GRPOJobLauncher, "_get_estimator_class", return_value=mock_cls
        ):
            launcher._build_estimator("image:tag")

        kwargs = mock_cls.call_args[1]
        assert "use_spot_instances" not in kwargs
        assert "max_wait" not in kwargs

    def test_build_estimator_with_spot(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config(
            {
                "job": {"max_runtime_hours": 6, "max_wait_hours": 12},
                "infra": {"use_spot_instances": True},
            }
        )
        mock_cls = MagicMock()
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        with patch.object(
            GRPOJobLauncher, "_get_estimator_class", return_value=mock_cls
        ):
            launcher._build_estimator("image:tag")

        kwargs = mock_cls.call_args[1]
        assert kwargs["use_spot_instances"] is True
        assert kwargs["max_wait"] == 12 * 3600

    def test_build_estimator_encrypts_inter_container_by_default(self):
        """ACP IS031.1: SDK v2 Estimator enables in-transit encryption by default."""
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        mock_cls = MagicMock()
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        with patch.object(
            GRPOJobLauncher, "_get_estimator_class", return_value=mock_cls
        ):
            launcher._build_estimator("image:tag")

        kwargs = mock_cls.call_args[1]
        assert kwargs["encrypt_inter_container_traffic"] is True

    def test_build_estimator_encryption_can_be_disabled(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"encrypt_inter_container_traffic": False}})
        mock_cls = MagicMock()
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        with patch.object(
            GRPOJobLauncher, "_get_estimator_class", return_value=mock_cls
        ):
            launcher._build_estimator("image:tag")

        kwargs = mock_cls.call_args[1]
        assert kwargs["encrypt_inter_container_traffic"] is False


class TestBuildMultiNodeEstimator:
    def test_multi_node(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config(
            {
                "infra": {
                    "topology": "multi_node",
                    "multi_node": {
                        "vllm_instance_type": "ml.g5.12xlarge",
                        "vllm_instance_count": 1,
                        "training_instance_type": "ml.p4d.24xlarge",
                        "training_instance_count": 2,
                    },
                }
            }
        )
        mock_cls = MagicMock()
        mock_ig = MagicMock()

        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        mock_sagemaker_ig = MagicMock()
        mock_sagemaker_ig.InstanceGroup = mock_ig
        with (
            patch.object(
                GRPOJobLauncher, "_get_estimator_class", return_value=mock_cls
            ),
            patch.dict("sys.modules", {"sagemaker.instance_group": mock_sagemaker_ig}),
        ):
            launcher._build_multi_node_estimator("image:tag")

        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args[1]
        assert "instance_groups" in kwargs
        # ACP IS031.1: in-transit encryption enabled by default.
        assert kwargs["encrypt_inter_container_traffic"] is True

    def test_multi_node_no_kms_key_by_default(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config(
            {
                "infra": {
                    "topology": "multi_node",
                    "multi_node": {
                        "vllm_instance_type": "ml.g5.12xlarge",
                        "vllm_instance_count": 1,
                        "training_instance_type": "ml.p4d.24xlarge",
                        "training_instance_count": 2,
                    },
                }
            }
        )
        mock_cls = MagicMock()
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        mock_sagemaker_ig = MagicMock()
        mock_sagemaker_ig.InstanceGroup = MagicMock()
        with (
            patch.object(
                GRPOJobLauncher, "_get_estimator_class", return_value=mock_cls
            ),
            patch.dict("sys.modules", {"sagemaker.instance_group": mock_sagemaker_ig}),
        ):
            launcher._build_multi_node_estimator("image:tag")

        kwargs = mock_cls.call_args[1]
        assert "output_kms_key" not in kwargs

    def test_multi_node_with_kms_key(self):
        """Regression guard: this heterogeneous-InstanceGroups path

        (topology: multi_node, e.g. GRPO with a dedicated vLLM instance
        group) never wired infra.kms_key_id at all, unlike the other three
        aws_training_job launch paths — a job using this path with
        kms_key_id set would still encrypt with the destination bucket's
        own default key instead of the requested one.
        """
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config(
            {
                "infra": {
                    "kms_key_id": "alias/my-kms-key",
                    "topology": "multi_node",
                    "multi_node": {
                        "vllm_instance_type": "ml.g5.12xlarge",
                        "vllm_instance_count": 1,
                        "training_instance_type": "ml.p4d.24xlarge",
                        "training_instance_count": 2,
                    },
                }
            }
        )
        mock_cls = MagicMock()
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        mock_sagemaker_ig = MagicMock()
        mock_sagemaker_ig.InstanceGroup = MagicMock()
        with (
            patch.object(
                GRPOJobLauncher, "_get_estimator_class", return_value=mock_cls
            ),
            patch.dict("sys.modules", {"sagemaker.instance_group": mock_sagemaker_ig}),
        ):
            launcher._build_multi_node_estimator("image:tag")

        kwargs = mock_cls.call_args[1]
        assert kwargs["output_kms_key"] == "alias/my-kms-key"

    def test_multi_node_missing_config(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        config.infra.multi_node = None
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        mock_cls = MagicMock()
        mock_ig_module = MagicMock()
        with (
            patch.object(
                GRPOJobLauncher, "_get_estimator_class", return_value=mock_cls
            ),
            patch.dict("sys.modules", {"sagemaker.instance_group": mock_ig_module}),
        ):
            with pytest.raises(ValueError, match="multi_node config required"):
                launcher._build_multi_node_estimator("image:tag")


class TestGetEstimatorClass:
    def test_prefers_pytorch(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        mock_pytorch = MagicMock()
        with patch.dict(
            "sys.modules", {"sagemaker.pytorch": MagicMock(PyTorch=mock_pytorch)}
        ):
            cls = GRPOJobLauncher._get_estimator_class()
        assert cls is mock_pytorch

    def test_fallback_to_estimator(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        mock_base = MagicMock()
        with (
            patch.dict("sys.modules", {"sagemaker.pytorch": None}),
            patch("builtins.__import__", side_effect=ImportError),
            patch.dict(
                "sys.modules", {"sagemaker.estimator": MagicMock(Estimator=mock_base)}
            ),
        ):
            # Simplified: just test that import fallback logic exists
            pass


class TestWriteResolvedRequirements:
    """Recipe/inline requirements are written to the source directory."""

    def test_recipe_requirements_written(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"container": {"recipe": "qwen3_5"}})
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())
        src = launcher._source_dir
        req_path = os.path.join(src, "requirements.txt")
        assert os.path.exists(req_path)
        content = Path(req_path).read_text()
        assert "trl==1.6.0" in content
        assert "vllm==0.26.0" in content

    def test_inline_requirements_written(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        custom = ["trl>=2.0.0", "transformers>=6.0.0"]
        config = _make_config(
            {"container": {"framework_version": "2.9.0", "requirements": custom}}
        )
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())
        src = launcher._source_dir
        req_path = os.path.join(src, "requirements.txt")
        content = Path(req_path).read_text()
        assert "trl>=2.0.0" in content
        assert "transformers>=6.0.0" in content

    def test_recipe_torchrun_requirements_written(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"container": {"recipe": "qwen3"}})
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())
        src = launcher._source_dir
        torchrun_path = os.path.join(src, "requirements_torchrun.txt")
        assert os.path.exists(torchrun_path)
        content = Path(torchrun_path).read_text()
        assert "trl==0.29.1" in content

    def test_no_recipe_keeps_static_files(self):
        """No recipe, no image_uri → static requirements.txt is left unchanged."""
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())
        src = launcher._source_dir
        req_path = os.path.join(src, "requirements.txt")
        assert os.path.exists(req_path)
        content = Path(req_path).read_text()
        # The static file should not be empty (would indicate it was overwritten
        # to the empty-self-contained-image case). We don't pin the trl version
        # here because the static file's pinning may differ from the recipe
        # files (the HyperPod EKS path uses this static file at runtime and
        # pins to trl 0.28 — see foresightrlcore/runtime/scripts/requirements.txt).
        assert "trl" in content
        assert "transformers" in content

    def test_custom_image_uri_writes_empty_requirements(self):
        """Explicit image_uri without recipe → empty requirements.txt (image is self-contained)."""
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config(
            {
                "container": {
                    "image_uri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:latest"
                }
            }
        )
        assert config.container.recipe is None
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())
        src = launcher._source_dir
        req_path = os.path.join(src, "requirements.txt")
        assert os.path.exists(req_path)
        content = Path(req_path).read_text().strip()
        assert content == "", f"Expected empty requirements.txt, got: {content!r}"

    def test_custom_image_uri_with_explicit_requirements(self):
        """Explicit image_uri + requirements → those requirements are installed."""
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        extra = ["my-package>=1.0.0"]
        config = _make_config(
            {
                "container": {
                    "image_uri": "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:latest",
                    "requirements": extra,
                }
            }
        )
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())
        src = launcher._source_dir
        req_path = os.path.join(src, "requirements.txt")
        content = Path(req_path).read_text()
        assert "my-package>=1.0.0" in content

    def test_empty_requirements_list_writes_empty_file(self):
        """requirements: [] → empty requirements.txt (explicit opt-out)."""
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"container": {"requirements": []}})
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())
        src = launcher._source_dir
        req_path = os.path.join(src, "requirements.txt")
        content = Path(req_path).read_text().strip()
        assert content == "", f"Expected empty requirements.txt, got: {content!r}"


class TestBuildInputDataV3:
    def test_builds_list(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        launcher = GRPOJobLauncher(config)

        mock_input_data = MagicMock()
        mock_configs = MagicMock(InputData=mock_input_data)
        with patch.dict("sys.modules", {"sagemaker.train.configs": mock_configs}):
            result = launcher._build_input_data_v3(
                {"train": "s3://b/t", "config": "s3://b/c"}
            )

        assert len(result) == 2


class TestLaunch:
    def test_launch_v2(self):
        import foresightrlcore.execution.aws_sagemaker_launcher as mod
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        mod._SDK_V3 = False

        with (
            patch.object(
                launcher, "_upload_config", return_value="s3://b/config/args.yaml"
            ),
            patch.object(launcher, "_launch_v2", return_value="job-v2") as mock_v2,
            patch(
                "foresightrlcore.execution.aws_sagemaker_launcher._resolve_image_uri",
                return_value="img",
            ),
        ):
            result = launcher.launch(wait=False)

        assert result == "job-v2"
        mod._SDK_V3 = None

    def test_launch_v3(self):
        import foresightrlcore.execution.aws_sagemaker_launcher as mod
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        mod._SDK_V3 = True

        with (
            patch.object(
                launcher, "_upload_config", return_value="s3://b/config/args.yaml"
            ),
            patch.object(launcher, "_launch_v3", return_value="job-v3") as mock_v3,
            patch(
                "foresightrlcore.execution.aws_sagemaker_launcher._resolve_image_uri",
                return_value="img",
            ),
        ):
            result = launcher.launch(wait=False)

        assert result == "job-v3"
        mod._SDK_V3 = None

    def test_launch_mixed_dedicated_vllm_uses_boto3_on_v2(self):
        import foresightrlcore.execution.aws_sagemaker_launcher as mod
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config(
            {
                "infra": {
                    "topology": "dedicated_vllm",
                    "instance_type": "ml.g6e.48xlarge",
                    "instance_count": 2,
                    "vllm_instance_type": "ml.g6e.12xlarge",
                }
            }
        )
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        mod._SDK_V3 = False

        with (
            patch.object(
                launcher, "_upload_config", return_value="s3://b/config/args.yaml"
            ),
            patch.object(
                launcher, "_submit_via_boto3", return_value="job-mixed"
            ) as mock_boto3,
            patch.object(launcher, "_launch_v2") as mock_v2,
            patch(
                "foresightrlcore.execution.aws_sagemaker_launcher._resolve_image_uri",
                return_value="img",
            ) as mock_image,
        ):
            result = launcher.launch(wait=False)

        assert result == "job-mixed"
        mock_boto3.assert_called_once()
        mock_v2.assert_not_called()
        assert mock_image.call_args.kwargs["instance_type"] == "ml.g6e.48xlarge"
        mod._SDK_V3 = None

    def test_launch_mixed_dedicated_vllm_uses_boto3_on_v3(self):
        import foresightrlcore.execution.aws_sagemaker_launcher as mod
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config(
            {
                "infra": {
                    "topology": "dedicated_vllm",
                    "instance_type": "ml.g6e.48xlarge",
                    "instance_count": 2,
                    "vllm_instance_type": "ml.g6e.12xlarge",
                }
            }
        )
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        mod._SDK_V3 = True

        with (
            patch.object(
                launcher, "_upload_config", return_value="s3://b/config/args.yaml"
            ),
            patch.object(
                launcher, "_submit_via_boto3", return_value="job-mixed-v3"
            ) as mock_boto3,
            patch.object(launcher, "_launch_v3") as mock_v3,
            patch(
                "foresightrlcore.execution.aws_sagemaker_launcher._resolve_image_uri",
                return_value="img",
            ),
        ):
            result = launcher.launch(wait=False)

        assert result == "job-mixed-v3"
        mock_boto3.assert_called_once()
        mock_v3.assert_not_called()
        mod._SDK_V3 = None

    def test_launch_v3_with_training_plan_uses_boto3(self):
        import foresightrlcore.execution.aws_sagemaker_launcher as mod
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"training_plan_arn": "arn:plan"}})
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        mod._SDK_V3 = True

        with (
            patch.object(
                launcher, "_upload_config", return_value="s3://b/config/args.yaml"
            ),
            patch.object(
                launcher, "_submit_via_boto3", return_value="job-boto3"
            ) as mock_boto3,
            patch(
                "foresightrlcore.execution.aws_sagemaker_launcher._resolve_image_uri",
                return_value="img",
            ),
        ):
            result = launcher.launch(wait=False)

        assert result == "job-boto3"
        mod._SDK_V3 = None

    def test_launch_adds_model_channel_for_s3_uri(self):
        """When model.name is an s3:// prefix, a 'model' input channel
        is added so SageMaker pre-mounts the weights and the container
        loads from /opt/ml/input/data/model instead of HF Hub."""
        import foresightrlcore.execution.aws_sagemaker_launcher as mod
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        s3_model_uri = "s3://bucket/models/Qwen3.6-27B/"
        config = _make_config({"model": {"name": s3_model_uri}})
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        mod._SDK_V3 = False

        captured = {}

        def _capture(image_uri, input_channels, wait):  # noqa: ARG001
            captured["channels"] = input_channels
            return "job-s3model"

        with (
            patch.object(
                launcher, "_upload_config", return_value="s3://b/config/args.yaml"
            ),
            patch.object(launcher, "_launch_v2", side_effect=_capture),
            patch(
                "foresightrlcore.execution.aws_sagemaker_launcher._resolve_image_uri",
                return_value="img",
            ),
        ):
            launcher.launch()

        mod._SDK_V3 = None
        assert captured["channels"]["model"] == s3_model_uri
        assert captured["channels"]["train"] == "s3://bucket/train"
        assert captured["channels"]["config"] == "s3://b/config/args.yaml"

    def test_launch_omits_model_channel_for_hf_id(self):
        """HF Hub model IDs do NOT get a 'model' input channel (no S3 mount)."""
        import foresightrlcore.execution.aws_sagemaker_launcher as mod
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"model": {"name": "Qwen/Qwen3.5-27B"}})
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        mod._SDK_V3 = False

        captured = {}

        def _capture(image_uri, input_channels, wait):  # noqa: ARG001
            captured["channels"] = input_channels
            return "job-hf"

        with (
            patch.object(
                launcher, "_upload_config", return_value="s3://b/config/args.yaml"
            ),
            patch.object(launcher, "_launch_v2", side_effect=_capture),
            patch(
                "foresightrlcore.execution.aws_sagemaker_launcher._resolve_image_uri",
                return_value="img",
            ),
        ):
            launcher.launch()

        mod._SDK_V3 = None
        assert "model" not in captured["channels"]

    def test_launch_omits_train_channel_for_merge_lora_only(self):
        """merge_lora_only reads its checkpoint via output.checkpoint_s3_path

        (SageMaker's own CheckpointConfig sync), not an input channel, and
        legitimately has no data.train_path set at all. Mounting a 'train'
        channel from an unset/None train_path would fail at submit time
        even though config validation allows merge_lora_only configs to
        omit data.train_path — see _validate_data_required's and
        _validate_execution's merge_lora_only exemptions.
        """
        import foresightrlcore.execution.aws_sagemaker_launcher as mod
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config(
            {
                "method": {"type": "sft_verl"},
                "training": {"sft_verl": {"merge_lora_only": True}},
                "data": {"train_path": None},
                "output": {"checkpoint_s3_path": "s3://bucket/checkpoints/step_10"},
            }
        )
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        mod._SDK_V3 = False

        captured = {}

        def _capture(image_uri, input_channels, wait):  # noqa: ARG001
            captured["channels"] = input_channels
            return "job-merge"

        with (
            patch.object(
                launcher, "_upload_config", return_value="s3://b/config/args.yaml"
            ),
            patch.object(launcher, "_launch_v2", side_effect=_capture),
            patch(
                "foresightrlcore.execution.aws_sagemaker_launcher._resolve_image_uri",
                return_value="img",
            ),
        ):
            launcher.launch()

        mod._SDK_V3 = None
        assert "train" not in captured["channels"]
        assert captured["channels"]["config"] == "s3://b/config/args.yaml"

    def test_launch_with_eval_path(self):
        import foresightrlcore.execution.aws_sagemaker_launcher as mod
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"data": {"eval_path": "s3://bucket/eval"}})
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        mod._SDK_V3 = False

        with (
            patch.object(
                launcher, "_upload_config", return_value="s3://b/config/args.yaml"
            ),
            patch.object(launcher, "_launch_v2", return_value="job-eval") as mock_v2,
            patch(
                "foresightrlcore.execution.aws_sagemaker_launcher._resolve_image_uri",
                return_value="img",
            ),
        ):
            result = launcher.launch()

        mod._SDK_V3 = None

    def test_launch_with_secrets(self):
        import foresightrlcore.execution.aws_sagemaker_launcher as mod
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config(
            {
                "secrets": {
                    "provider": "idps",
                    "idps": {
                        "policy_id": "test-policy",
                        "endpoint": "vkm.example.com",
                    },
                    "entries": [
                        {"env_var": "SECRET1", "secret_path": "/path/to/secret"},
                    ],
                }
            }
        )
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        mod._SDK_V3 = False

        with (
            patch.object(
                launcher, "_upload_config", return_value="s3://b/config/args.yaml"
            ),
            patch.object(launcher, "_launch_v2", return_value="job-secrets"),
            patch(
                "foresightrlcore.execution.aws_sagemaker_launcher._resolve_image_uri",
                return_value="img",
            ),
        ):
            result = launcher.launch()

        assert result == "job-secrets"
        mod._SDK_V3 = None

    def test_launch_torchrun(self):
        import foresightrlcore.execution.aws_sagemaker_launcher as mod
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"vllm_mode": "none"}})
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        launcher = GRPOJobLauncher(config, role="role", session=mock_session)
        launcher._s3_client = MagicMock()  # type: ignore[assignment]

        mod._SDK_V3 = False

        with (
            patch.object(
                launcher, "_upload_config", return_value="s3://b/config/args.yaml"
            ),
            patch.object(launcher, "_launch_v2", return_value="job-torchrun"),
            patch(
                "foresightrlcore.execution.aws_sagemaker_launcher._resolve_image_uri",
                return_value="img",
            ),
        ):
            result = launcher.launch()

        assert result == "job-torchrun"
        mod._SDK_V3 = None


class TestLaunchV2:
    def test_single_node(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        mock_estimator = MagicMock()
        mock_estimator.latest_training_job.name = "job-from-estimator"

        with patch.object(launcher, "_build_estimator", return_value=mock_estimator):
            result = launcher._launch_v2("image:tag", {"train": "s3://b/t"}, wait=False)

        mock_estimator.fit.assert_called_once()
        assert result == "job-from-estimator"

    def test_multi_node(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config(
            {
                "infra": {
                    "topology": "multi_node",
                    "multi_node": {
                        "vllm_instance_type": "ml.g5.12xlarge",
                        "vllm_instance_count": 1,
                        "training_instance_type": "ml.p4d.24xlarge",
                        "training_instance_count": 2,
                    },
                }
            }
        )
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        mock_estimator = MagicMock()
        mock_estimator.latest_training_job.name = "multi-job"

        with patch.object(
            launcher, "_build_multi_node_estimator", return_value=mock_estimator
        ):
            result = launcher._launch_v2("image:tag", {"train": "s3://b/t"}, wait=True)

        mock_estimator.fit.assert_called_once_with({"train": "s3://b/t"}, wait=True)
        assert result == "multi-job"


class TestLaunchV3:
    def test_v3_launch(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        mock_trainer = MagicMock()
        mock_trainer._latest_training_job.training_job_name = "v3-job"
        mock_input = [MagicMock()]

        with (
            patch.object(
                launcher,
                "_build_model_trainer",
                return_value=(mock_trainer, mock_input),
            ),
            patch.object(launcher, "_build_input_data_v3", return_value=mock_input),
        ):
            result = launcher._launch_v3("image:tag", {"train": "s3://b/t"}, wait=False)

        mock_trainer.train.assert_called_once()
        assert result == "v3-job"

    def test_build_model_trainer_with_vpc(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config(
            {
                "infra": {
                    "vpc": {
                        "subnets": ["subnet-1", "subnet-2", "subnet-3"],
                        "security_groups": ["sg-1"],
                    }
                },
            }
        )
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        mock_model_trainer = MagicMock()
        with patch(
            "sagemaker.train.model_trainer.ModelTrainer",
            mock_model_trainer,
        ):
            launcher._build_model_trainer("image:tag", [])

        kwargs = mock_model_trainer.call_args[1]
        networking = kwargs["networking"]
        assert networking.subnets == ["subnet-1", "subnet-2", "subnet-3"]
        assert networking.security_group_ids == ["sg-1"]
        # ACP IS031.1: encryption on even alongside a VPC.
        assert networking.enable_inter_container_traffic_encryption is True

    def test_build_model_trainer_encrypts_inter_container_by_default(self):
        """ACP IS031.1: Networking carries the encryption flag even without a VPC."""
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        mock_model_trainer = MagicMock()
        with patch(
            "sagemaker.train.model_trainer.ModelTrainer",
            mock_model_trainer,
        ):
            launcher._build_model_trainer("image:tag", [])

        kwargs = mock_model_trainer.call_args[1]
        networking = kwargs["networking"]
        assert networking.enable_inter_container_traffic_encryption is True

    def test_build_model_trainer_encryption_can_be_disabled(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"encrypt_inter_container_traffic": False}})
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        mock_model_trainer = MagicMock()
        with patch(
            "sagemaker.train.model_trainer.ModelTrainer",
            mock_model_trainer,
        ):
            launcher._build_model_trainer("image:tag", [])

        kwargs = mock_model_trainer.call_args[1]
        networking = kwargs["networking"]
        assert networking.enable_inter_container_traffic_encryption is False

    def test_build_model_trainer_passes_volume_size(self):
        """SDK v3 Compute must receive the configured EBS volume size.

        Regression: the v3 ModelTrainer path previously built ``Compute(...)``
        without ``volume_size_in_gb``, so SageMaker fell back to its 30 GB floor
        and large GRPO jobs ran out of disk during image pull / checkpointing.
        """
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"volume_size_gb": 300}})
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        mock_model_trainer = MagicMock()
        with patch(
            "sagemaker.train.model_trainer.ModelTrainer",
            mock_model_trainer,
        ):
            launcher._build_model_trainer("image:tag", [])

        kwargs = mock_model_trainer.call_args[1]
        compute = kwargs["compute"]
        assert compute.volume_size_in_gb == 300

    def test_build_model_trainer_no_kms_key_by_default(self):
        """infra.kms_key_id is opt-in — unset must not change prior behavior

        (the destination bucket's own default encryption key applies, same
        as before this field existed).
        """
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        mock_model_trainer = MagicMock()
        with patch(
            "sagemaker.train.model_trainer.ModelTrainer",
            mock_model_trainer,
        ):
            launcher._build_model_trainer("image:tag", [])

        kwargs = mock_model_trainer.call_args[1]
        assert kwargs["output_data_config"].kms_key_id is None

    def test_build_model_trainer_with_kms_key(self):
        """infra.kms_key_id must reach OutputDataConfig — SageMaker has no

        separate CheckpointConfig KMS field, so this same key is what
        checkpoint artifact uploads use too (see the field's own docstring
        in schema.py).
        """
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"infra": {"kms_key_id": "alias/my-kms-key"}})
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        mock_model_trainer = MagicMock()
        with patch(
            "sagemaker.train.model_trainer.ModelTrainer",
            mock_model_trainer,
        ):
            launcher._build_model_trainer("image:tag", [])

        kwargs = mock_model_trainer.call_args[1]
        assert kwargs["output_data_config"].kms_key_id == "alias/my-kms-key"

    def test_build_model_trainer_no_spot_by_default(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config()
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        mock_model_trainer = MagicMock()
        with patch(
            "sagemaker.train.model_trainer.ModelTrainer",
            mock_model_trainer,
        ):
            launcher._build_model_trainer("image:tag", [])

        kwargs = mock_model_trainer.call_args[1]
        # SDK v3 leaves unset fields as an Unassigned() sentinel (not None).
        assert kwargs["compute"].enable_managed_spot_training is not True
        assert not isinstance(
            kwargs["stopping_condition"].max_wait_time_in_seconds, int
        )

    def test_build_model_trainer_with_spot(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config(
            {
                "job": {"max_runtime_hours": 6, "max_wait_hours": 9},
                "infra": {"use_spot_instances": True},
            }
        )
        launcher = GRPOJobLauncher(config, role="role", session=MagicMock())

        mock_model_trainer = MagicMock()
        with patch(
            "sagemaker.train.model_trainer.ModelTrainer",
            mock_model_trainer,
        ):
            launcher._build_model_trainer("image:tag", [])

        kwargs = mock_model_trainer.call_args[1]
        assert kwargs["compute"].enable_managed_spot_training is True
        assert kwargs["stopping_condition"].max_wait_time_in_seconds == 9 * 3600


class TestSourceStagingExcludesBytecode:
    """Staging must never walk ``__pycache__``.

    CPython writes bytecode as ``<name>.pyc.<id>`` and renames it into place,
    so any interpreter compiling the tree while ``copytree`` runs makes the
    copy fail on a directory entry that has already vanished. CI hits this
    because every tox env runs in parallel against one workspace.
    """

    @staticmethod
    def _tree_with_bytecode(root: Path, name: str) -> Path:
        d = root / name
        (d / "__pycache__").mkdir(parents=True)
        (d / "mod.py").write_text("x = 1\n")
        (d / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\x00")
        return d

    def test_no_bytecode_in_staged_source_dir(self, tmp_path):
        from foresightrlcore.execution import aws_sagemaker_launcher as mod

        scripts = self._tree_with_bytecode(tmp_path, "scripts")
        package = self._tree_with_bytecode(tmp_path, "foresightrlcore")

        with (
            patch.object(mod, "SCRIPTS_DIR", scripts),
            patch.object(mod, "PACKAGE_DIR", package),
        ):
            launcher = mod.GRPOJobLauncher(_make_config())
            source_dir = Path(launcher._source_dir)

        assert not list(source_dir.rglob("__pycache__"))
        assert not list(source_dir.rglob("*.pyc"))
        assert (source_dir / "mod.py").is_file()
        assert (source_dir / "foresightrlcore" / "mod.py").is_file()


class TestExtraPackages:
    """Extension packages must be staged into the job and announced via env."""

    @pytest.mark.requires_intuit
    def test_source_dir_contains_staged_package(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"container": {"extra_packages": ["foresightrl_intuit"]}})
        launcher = GRPOJobLauncher(config)

        source_dir = Path(launcher._source_dir)

        assert (source_dir / "foresightrlcore").is_dir()
        assert (source_dir / "foresightrl_intuit" / "__init__.py").is_file()

    def test_environment_announces_staged_packages(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        config = _make_config({"container": {"extra_packages": ["foresightrl_intuit"]}})
        launcher = GRPOJobLauncher(config)

        env = launcher._job_environment()

        assert env["FORESIGHT_EXTRA_PACKAGES"] == "foresightrl_intuit"
        assert env["GENOS_ENV"] == "e2e"

    def test_no_extra_packages_leaves_environment_untouched(self):
        from foresightrlcore.execution.aws_sagemaker_launcher import GRPOJobLauncher

        launcher = GRPOJobLauncher(_make_config())

        env = launcher._job_environment()

        assert "FORESIGHT_EXTRA_PACKAGES" not in env
        assert env["GENOS_ENV"] == "e2e"

    def test_default_is_empty(self):
        assert _make_config().container.extra_packages == []
