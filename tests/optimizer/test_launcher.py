"""OptimizerSageMakerLauncher: image/instance/env resolution + job submission.

boto3, the SageMaker session, and the source-dir copy are all mocked; no AWS
call is made.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from foresightrlcore.config.optimizer_schema import OptimizerJobConfig
from foresightrlcore.execution.optimizer_launcher import (
    OptimizerSageMakerLauncher,
    _parent_prefix,
)


def _config(**over) -> OptimizerJobConfig:
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
        "execution": {"type": "aws_training_job"},
        "output": {"s3_path": "s3://b/out/"},
        "container": {"recipe": "qwen3_5"},
    }
    for k, v in over.items():
        d[k] = v
    return OptimizerJobConfig.model_validate(d)


def _launcher(config=None, **kw):
    session = MagicMock()
    session.boto_session.region_name = "us-west-2"
    return OptimizerSageMakerLauncher(
        config or _config(), role="arn:aws:iam::1:role/r", session=session, **kw
    )


class TestParentPrefix:
    def test_strips_object_to_prefix(self):
        assert _parent_prefix("s3://b/a/b/c.yaml") == "s3://b/a/b/"

    def test_bucket_only_unchanged(self):
        assert _parent_prefix("s3://bucket") == "s3://bucket"


class TestParseS3Uri:
    def test_valid(self):
        assert OptimizerSageMakerLauncher._parse_s3_uri("s3://b/a/c") == ("b", "a/c")

    def test_bucket_only(self):
        assert OptimizerSageMakerLauncher._parse_s3_uri("s3://b") == ("b", "")

    def test_non_s3_raises(self):
        with pytest.raises(ValueError, match="Not an S3 URI"):
            OptimizerSageMakerLauncher._parse_s3_uri("/local/path")


class TestJobName:
    def test_short_name(self):
        name = _launcher()._build_job_name("20240101000000")
        assert name == "optim-opt-20240101000000"

    def test_long_name_truncated_to_63(self):
        cfg = _config(job={"name": "x" * 80})
        name = _launcher(cfg)._build_job_name("20240101000000")
        assert len(name) == 63

    def test_run_output_path_has_trailing_slash(self):
        assert _launcher()._run_output_s3_path() == "s3://b/out/"


class TestResolveImageUri:
    def test_api_only_uses_optimizer_image_from_recipe(self):
        uri = _launcher()._resolve_image_uri()
        assert uri  # recipe qwen3_5 resolves optimizer_image_uri

    def test_api_only_explicit_image_uri(self):
        cfg = _config(container={"image_uri": "1.dkr.ecr/x:t"})
        assert _launcher(cfg)._resolve_image_uri() == "1.dkr.ecr/x:t"

    def test_api_only_without_any_image_raises(self):
        cfg = _config(container={})
        with pytest.raises(ValueError, match="optimizer requires a container image"):
            _launcher(cfg)._resolve_image_uri()

    def test_gpu_uses_perf_test_image(self):
        cfg = _config()
        cfg.optimizer.models[0].provider = "local"
        cfg.container.perf_test_image_uri = "1.dkr.ecr/vllm:t"
        assert _launcher(cfg)._resolve_image_uri() == "1.dkr.ecr/vllm:t"

    def test_gpu_without_vllm_image_raises(self):
        cfg = _config(container={})
        cfg.optimizer.models[0].provider = "local"
        with pytest.raises(ValueError, match="vLLM-capable GPU image"):
            _launcher(cfg)._resolve_image_uri()


class TestModelInput:
    def test_none_for_api_only(self):
        assert _launcher()._build_model_input() is None

    def test_s3_local_model_returns_uri(self):
        cfg = _config()
        cfg.optimizer.models[0].provider = "local"
        cfg.optimizer.models[0].name = "s3://bucket/models/qwen/"
        assert _launcher(cfg)._build_model_input() == "s3://bucket/models/qwen/"


class TestEnvironment:
    def test_env_sets_container_paths(self):
        env = _launcher()._build_environment("s3://b/out/source/sourcedir.tar.gz")
        assert env["OPTIMIZER_CONFIG"] == "/opt/ml/input/data/config/args.yaml"
        assert env["OPTIMIZER_OUTPUT_DIR"] == "/opt/ml/output/data"
        assert env["OPTIMIZER_CHECKPOINT_DIR"] == "/opt/ml/checkpoints"
        assert env["OPTIMIZER_SOURCE_S3"].endswith("sourcedir.tar.gz")


class TestWriteRequirements:
    def test_inline_requirements_win(self, tmp_path):
        cfg = _config()
        cfg.container.requirements = ["pandas==2.0", "pyarrow"]
        _launcher(cfg)._write_resolved_requirements(str(tmp_path))
        content = (tmp_path / "requirements.txt").read_text()
        assert "pandas==2.0" in content and "pyarrow" in content

    def test_resolved_optimizer_requirements_used(self, tmp_path):
        cfg = _config()
        cfg.container.requirements = None
        cfg.container.resolved_requirements_optimizer = ["vllm==0.20"]
        _launcher(cfg)._write_resolved_requirements(str(tmp_path))
        assert "vllm==0.20" in (tmp_path / "requirements.txt").read_text()

    def test_empty_when_nothing_configured(self, tmp_path):
        cfg = _config()
        cfg.container.requirements = None
        cfg.container.resolved_requirements_optimizer = None
        _launcher(cfg)._write_resolved_requirements(str(tmp_path))
        assert (tmp_path / "requirements.txt").exists()


class TestUpload:
    def test_upload_source_dir_tars_and_uploads(self, tmp_path):
        launcher = _launcher()
        src = tmp_path / "src"
        src.mkdir()
        (src / "entry.sh").write_text("echo hi")
        launcher._prepared_source_dir = str(src)
        launcher._s3_client = MagicMock()
        uri = launcher._upload_source_dir()
        assert uri == "s3://b/out/source/sourcedir.tar.gz"
        launcher._s3_client.upload_file.assert_called_once()

    def test_upload_config_uploads_yaml(self, tmp_path):
        launcher = _launcher()
        launcher._s3_client = MagicMock()
        uri = launcher._upload_config()
        assert uri == "s3://b/out/config/args.yaml"
        launcher._s3_client.upload_file.assert_called_once()

    def test_source_dir_writes_requirements(self, tmp_path):
        launcher = _launcher()
        # Avoid the heavy package copy; just verify requirements land in the dir.
        with patch("foresightrlcore.execution.optimizer_launcher.shutil.copytree"):
            src = launcher._source_dir
        import os

        assert os.path.exists(os.path.join(src, "requirements.txt"))


class TestLaunch:
    def test_launch_submits_training_job(self, tmp_path):
        cfg = _config(
            output={
                "s3_path": "s3://b/out/",
                "checkpoint_s3_path": "s3://b/ckpt/",
            },
        )
        cfg.optimizer.training_plan_arn = "arn:aws:sagemaker:plan/x"
        cfg.optimizer.vpc = None
        launcher = _launcher(cfg)

        src = tmp_path / "src"
        src.mkdir()
        (src / "entry.sh").write_text("echo hi")
        launcher._prepared_source_dir = str(src)

        sm_client = MagicMock()
        sm_client.create_training_job.return_value = {"TrainingJobArn": "arn:job"}
        s3_client = MagicMock()

        def fake_boto_client(name, **kw):
            return sm_client if name == "sagemaker" else s3_client

        with patch(
            "foresightrlcore.execution.optimizer_launcher.boto3.client",
            side_effect=fake_boto_client,
        ):
            job_name = launcher.launch(wait=False)

        assert job_name.startswith("optim-opt-")
        request = sm_client.create_training_job.call_args.kwargs
        assert request["RoleArn"] == "arn:aws:iam::1:role/r"
        assert "CheckpointConfig" in request
        assert request["CheckpointConfig"]["S3Uri"] == "s3://b/ckpt/"
        assert (
            request["ResourceConfig"]["TrainingPlanArn"] == "arn:aws:sagemaker:plan/x"
        )
        # ACP IS031.1: in-transit encryption on by default.
        assert request["EnableInterContainerTrafficEncryption"] is True
        # config channel is always present.
        channels = {c["ChannelName"] for c in request["InputDataConfig"]}
        assert "config" in channels

    def test_launch_encryption_can_be_disabled(self, tmp_path):
        cfg = _config()
        cfg.optimizer.vpc = None
        cfg.optimizer.encrypt_inter_container_traffic = False
        launcher = _launcher(cfg)

        src = tmp_path / "src"
        src.mkdir()
        (src / "e.sh").write_text("x")
        launcher._prepared_source_dir = str(src)

        sm_client = MagicMock()
        sm_client.create_training_job.return_value = {"TrainingJobArn": "arn:job"}
        s3_client = MagicMock()

        with patch(
            "foresightrlcore.execution.optimizer_launcher.boto3.client",
            side_effect=lambda name, **kw: (
                sm_client if name == "sagemaker" else s3_client
            ),
        ):
            launcher.launch(wait=False)

        request = sm_client.create_training_job.call_args.kwargs
        assert request["EnableInterContainerTrafficEncryption"] is False

    def test_launch_with_vpc_and_wait(self, tmp_path):
        cfg = _config()
        cfg.optimizer.vpc = MagicMock(subnets=["subnet-1"], security_groups=["sg-1"])
        launcher = _launcher(cfg)

        src = tmp_path / "src"
        src.mkdir()
        (src / "e.sh").write_text("x")
        launcher._prepared_source_dir = str(src)

        sm_client = MagicMock()
        sm_client.create_training_job.return_value = {"TrainingJobArn": "arn:job"}
        waiter = MagicMock()
        sm_client.get_waiter.return_value = waiter
        s3_client = MagicMock()

        with patch(
            "foresightrlcore.execution.optimizer_launcher.boto3.client",
            side_effect=lambda name, **kw: (
                sm_client if name == "sagemaker" else s3_client
            ),
        ):
            launcher.launch(wait=True)

        request = sm_client.create_training_job.call_args.kwargs
        assert request["VpcConfig"]["Subnets"] == ["subnet-1"]
        waiter.wait.assert_called_once()


class TestRoleResolution:
    def test_a_configured_role_beats_auto_detection(self):
        """A pinned role must not be overridden by the caller's identity.

        Auto-detection resolves to whoever launched the job, so the same
        config submitted from two shells can run as two different roles.
        """
        config = _config()
        config.optimizer.role = "arn:aws:iam::1:role/Pinned"
        launcher = OptimizerSageMakerLauncher(config, session=MagicMock())

        with patch(
            "sagemaker.core.helper.session_helper.get_execution_role",
            return_value="arn:aws:iam::1:role/Detected",
        ):
            assert launcher.role == "arn:aws:iam::1:role/Pinned"

    def test_an_unset_role_still_falls_back_to_auto_detection(self):
        config = _config()
        assert config.optimizer.role is None
        launcher = OptimizerSageMakerLauncher(config, session=MagicMock())

        with patch(
            "sagemaker.core.helper.session_helper.get_execution_role",
            return_value="arn:aws:iam::1:role/Detected",
        ):
            assert launcher.role == "arn:aws:iam::1:role/Detected"
