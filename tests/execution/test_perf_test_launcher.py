"""Tests for PerfTestSageMakerLauncher -- env-var mapping + job construction."""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from foresightrlcore.config.perf_schema import PerfTestJobConfig
from foresightrlcore.config.schema import VpcConfig
from foresightrlcore.execution import perf_test_launcher as launcher_mod
from foresightrlcore.execution.perf_test_launcher import (
    PerfTestSageMakerLauncher,
    _parent_prefix,
)


def _make_config(tmp_path) -> PerfTestJobConfig:
    base = {
        "job": {"name": "perf-test"},
        "model": {"name": "s3://bucket/model/"},
        "perf_test": {
            "name": "baseline",
            "instance_type": "ml.g5.2xlarge",
            "serving": {
                "tensor_parallel_size": 1,
                "vllm_env": {"VLLM_USE_V1": "1"},
            },
            "gatling": {"data_s3_path": "s3://bucket/data/"},
        },
        "execution": {"type": "aws_training_job"},
        "output": {"s3_path": "s3://bucket/out/"},
        "container": {"recipe": "qwen3_5"},
        "environment": {"FOO": "bar"},
    }
    p = tmp_path / "c.yaml"
    p.write_text(yaml.dump(base))
    return PerfTestJobConfig.from_yaml(p)


class TestParentPrefix:
    def test_parent_prefix(self):
        assert _parent_prefix("s3://bucket/a/b/c.yaml") == "s3://bucket/a/b/"

    def test_bucket_root(self):
        # No leading slash after bucket -- keep as-is.
        assert _parent_prefix("s3://bucket") == "s3://bucket"


class TestLauncherProperties:
    def test_resolve_image_uri_from_recipe(self, tmp_path):
        cfg = _make_config(tmp_path)
        launcher = PerfTestSageMakerLauncher(cfg)
        image = launcher._resolve_image_uri()
        assert "djl-inference" in image

    def test_build_model_input_from_s3(self, tmp_path):
        cfg = _make_config(tmp_path)
        launcher = PerfTestSageMakerLauncher(cfg)
        assert launcher._build_model_input() == "s3://bucket/model/"

    def test_build_model_input_hf_id_returns_none(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.model.name = "Qwen/Qwen3-4B"
        launcher = PerfTestSageMakerLauncher(cfg)
        assert launcher._build_model_input() is None

    def test_build_environment_forwards_vllm_env(self, tmp_path):
        cfg = _make_config(tmp_path)
        launcher = PerfTestSageMakerLauncher(cfg)
        env = launcher._build_environment("s3://bucket/foo/source/sourcedir.tar.gz")
        assert env["PERF_TEST_CONFIG"] == "/opt/ml/input/data/config/args.yaml"
        assert env["PERF_TEST_DATA_CHANNEL"] == "/opt/ml/input/data/perf_data"
        assert env["PERF_TEST_SOURCE_S3"] == "s3://bucket/foo/source/sourcedir.tar.gz"
        assert env["PERF_TEST_OUTPUT_DIR"] == "/opt/ml/output/data"
        # Forwarded from config.environment
        assert env["FOO"] == "bar"
        # vllm_env serialised to JSON
        assert json.loads(env["PERF_TEST_VLLM_ENV_JSON"]) == {"VLLM_USE_V1": "1"}

    def test_build_environment_no_model_channel_uses_model_name(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.model.name = "Qwen/Qwen3-4B"
        launcher = PerfTestSageMakerLauncher(cfg)
        env = launcher._build_environment("s3://bucket/foo/source/sourcedir.tar.gz")
        assert env["PERF_TEST_MODEL_PATH"] == "Qwen/Qwen3-4B"

    def test_build_environment_no_run_name_key(self, tmp_path):
        """PERF_TEST_RUN_NAME must no longer appear in the environment."""
        cfg = _make_config(tmp_path)
        launcher = PerfTestSageMakerLauncher(cfg)
        env = launcher._build_environment("s3://bucket/foo/source/sourcedir.tar.gz")
        assert "PERF_TEST_RUN_NAME" not in env


class TestJobNameBuilder:
    def test_build_job_name(self, tmp_path):
        cfg = _make_config(tmp_path)
        launcher = PerfTestSageMakerLauncher(cfg)
        name = launcher._build_job_name("20260101120000")
        assert name.startswith("perf-perf-test-baseline-")
        assert "_" not in name
        assert len(name) <= 63

    def test_build_job_name_truncates_long_names(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.job.name = "x" * 80
        launcher = PerfTestSageMakerLauncher(cfg)
        name = launcher._build_job_name("20260101120000")
        assert len(name) == 63


class TestRequirementsWriter:
    def test_writes_perf_test_reqs_when_recipe(self, tmp_path):
        cfg = _make_config(tmp_path)
        launcher = PerfTestSageMakerLauncher(cfg)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        launcher._write_resolved_requirements(str(scratch))
        content = (scratch / "requirements.txt").read_text()
        assert "py-markdown-table" in content
        assert "psutil" in content

    def test_inline_requirements_override_recipe(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.container.requirements = ["only-this==1.0"]
        cfg.container.resolved_requirements_perf_test = ["should-not-be-used==2.0"]
        launcher = PerfTestSageMakerLauncher(cfg)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        launcher._write_resolved_requirements(str(scratch))
        content = (scratch / "requirements.txt").read_text()
        assert "only-this==1.0" in content
        assert "should-not-be-used" not in content


class TestRunOutputPath:
    def test_output_subdir_uses_run_name(self, tmp_path):
        cfg = _make_config(tmp_path)
        launcher = PerfTestSageMakerLauncher(cfg)
        path = launcher._run_output_s3_path()
        assert path == "s3://bucket/out/baseline/"


# ---------------------------------------------------------------------------
# Coverage of the boto3- and SageMaker-touching paths
# ---------------------------------------------------------------------------


class TestS3UriHelpers:
    def test_parse_s3_uri_with_key(self):
        bucket, key = PerfTestSageMakerLauncher._parse_s3_uri("s3://buck/foo/bar.txt")
        assert bucket == "buck"
        assert key == "foo/bar.txt"

    def test_parse_s3_uri_bucket_only(self):
        bucket, key = PerfTestSageMakerLauncher._parse_s3_uri("s3://buck")
        assert bucket == "buck"
        assert key == ""

    def test_parse_s3_uri_rejects_non_s3(self):
        with pytest.raises(ValueError, match="Not an S3 URI"):
            PerfTestSageMakerLauncher._parse_s3_uri("/local/path")

    def test_upload_file_to_s3_returns_uri(self, tmp_path):
        cfg = _make_config(tmp_path)
        launcher = PerfTestSageMakerLauncher(cfg)
        launcher._s3_client = MagicMock()
        local = tmp_path / "x.txt"
        local.write_text("hi")

        uri = launcher._upload_file_to_s3(str(local), "bucket", "k/x.txt")

        assert uri == "s3://bucket/k/x.txt"
        launcher._s3_client.upload_file.assert_called_once_with(
            str(local), "bucket", "k/x.txt"
        )


class TestSessionRoleClient:
    """Cover the lazy-init session/role/s3_client properties.

    The launcher tries the sagemaker SDK v3 import path first and falls
    back to v2; we mock both so the test runs without the sagemaker SDK
    actually being importable.
    """

    def test_s3_client_initialised_lazily(self, tmp_path):
        cfg = _make_config(tmp_path)
        launcher = PerfTestSageMakerLauncher(cfg)
        sentinel = object()
        with patch.object(launcher_mod, "boto3") as mock_boto3:
            mock_boto3.client.return_value = sentinel
            assert launcher.s3_client is sentinel
            # Cached -- a second access does not re-create.
            assert launcher.s3_client is sentinel
            mock_boto3.client.assert_called_once_with("s3")

    def test_session_property_uses_injected_session(self, tmp_path):
        cfg = _make_config(tmp_path)
        sentinel_session = MagicMock()
        launcher = PerfTestSageMakerLauncher(cfg, session=sentinel_session)
        assert launcher.session is sentinel_session

    def test_role_property_uses_injected_role(self, tmp_path):
        cfg = _make_config(tmp_path)
        launcher = PerfTestSageMakerLauncher(cfg, role="arn:aws:iam::123:role/foo")
        assert launcher.role == "arn:aws:iam::123:role/foo"

    def test_a_configured_role_beats_auto_detection(self, tmp_path):
        """A pinned role must not be overridden by the caller's identity.

        Auto-detection resolves to whoever launched the job, so the same
        config submitted from two shells can run as two different roles.
        """
        cfg = _make_config(tmp_path)
        cfg.perf_test.role = "arn:aws:iam::123:role/Pinned"
        launcher = PerfTestSageMakerLauncher(cfg)

        with patch(
            "sagemaker.core.helper.session_helper.get_execution_role",
            return_value="arn:aws:iam::123:role/Detected",
        ):
            assert launcher.role == "arn:aws:iam::123:role/Pinned"

    def test_role_falls_back_to_iam_when_sagemaker_raises_valueerror(self, tmp_path):
        cfg = _make_config(tmp_path)
        launcher = PerfTestSageMakerLauncher(cfg)

        # Simulate sagemaker.get_execution_role failing (local dev box).
        fake_sm_module = MagicMock()
        fake_sm_module.get_execution_role.side_effect = ValueError("no role")

        mock_iam = MagicMock()
        mock_iam.get_role.return_value = {
            "Role": {"Arn": "arn:aws:iam::1:role/fallback"}
        }

        with (
            patch.dict(
                "sys.modules",
                {"sagemaker": fake_sm_module},
            ),
            patch.object(launcher_mod, "boto3") as mock_boto3,
        ):
            mock_boto3.client.return_value = mock_iam
            # Force the sagemaker.core import path to fail so we exercise
            # the v2 fallback (ImportError -> import sagemaker).
            with patch.dict(
                "sys.modules",
                {"sagemaker.core.helper.session_helper": None},
            ):
                role = launcher.role

        assert role == "arn:aws:iam::1:role/fallback"
        mock_iam.get_role.assert_called_once_with(RoleName="sagemaker_execution_role")


class TestResolveImageUri:
    def test_perf_test_image_uri_wins_over_image_uri(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.container.perf_test_image_uri = "1234.dkr.ecr.us/perf:latest"
        cfg.container.image_uri = "should-not-be-used:tag"
        launcher = PerfTestSageMakerLauncher(cfg)
        assert launcher._resolve_image_uri() == "1234.dkr.ecr.us/perf:latest"

    def test_falls_back_to_image_uri_when_no_recipe(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.container.recipe = None
        cfg.container.perf_test_image_uri = None
        cfg.container.image_uri = "custom-image:v1"
        launcher = PerfTestSageMakerLauncher(cfg)
        assert launcher._resolve_image_uri() == "custom-image:v1"

    def test_falls_back_to_image_uri_when_recipe_set_but_perf_missing(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.container.perf_test_image_uri = None
        cfg.container.image_uri = "manual-override:v2"
        launcher = PerfTestSageMakerLauncher(cfg)
        assert launcher._resolve_image_uri() == "manual-override:v2"

    def test_raises_when_no_image_or_recipe(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.container.recipe = None
        cfg.container.perf_test_image_uri = None
        cfg.container.image_uri = None
        launcher = PerfTestSageMakerLauncher(cfg)
        with pytest.raises(ValueError, match="perf_test_image_uri"):
            launcher._resolve_image_uri()


class TestSourceDirAndUploads:
    """Exercise the source-dir staging and S3 upload helpers."""

    def test_source_dir_is_cached_across_accesses(self, tmp_path):
        cfg = _make_config(tmp_path)
        launcher = PerfTestSageMakerLauncher(cfg)
        d1 = launcher._source_dir
        d2 = launcher._source_dir
        assert d1 == d2
        # The staged tree contains both the scripts (perf_entrypoint.sh) and
        # the foresightrlcore package copy.
        assert (Path(d1) / "foresightrlcore").exists()
        assert (Path(d1) / "requirements.txt").exists()

    def test_upload_source_dir_uploads_tarball_to_expected_key(self, tmp_path):
        cfg = _make_config(tmp_path)
        launcher = PerfTestSageMakerLauncher(cfg)
        launcher._s3_client = MagicMock()

        uri = launcher._upload_source_dir()

        launcher._s3_client.upload_file.assert_called_once()
        bucket, key = (
            launcher._s3_client.upload_file.call_args.args[1],
            launcher._s3_client.upload_file.call_args.args[2],
        )
        assert bucket == "bucket"
        assert key == "out/source/sourcedir.tar.gz"
        assert uri == "s3://bucket/out/source/sourcedir.tar.gz"

    def test_upload_source_dir_produces_valid_tarball(self, tmp_path):
        # Capture the tarball before it's deleted by checking its contents.
        cfg = _make_config(tmp_path)
        launcher = PerfTestSageMakerLauncher(cfg)

        captured = {}

        def fake_upload(local_path, bucket, key):
            captured["bucket"] = bucket
            captured["key"] = key
            captured["valid"] = tarfile.is_tarfile(local_path)
            with tarfile.open(local_path, "r:gz") as tar:
                captured["names"] = tar.getnames()

        launcher._s3_client = MagicMock()
        launcher._s3_client.upload_file.side_effect = fake_upload

        launcher._upload_source_dir()

        assert captured["valid"] is True
        # The tarball should contain the foresightrlcore package and the
        # perf_entrypoint script.
        joined = "\n".join(captured["names"])
        assert "foresightrlcore" in joined
        assert "perf_entrypoint" in joined

    def test_upload_config_uploads_yaml_to_expected_key(self, tmp_path):
        cfg = _make_config(tmp_path)
        launcher = PerfTestSageMakerLauncher(cfg)
        launcher._s3_client = MagicMock()

        uri = launcher._upload_config()

        assert uri == "s3://bucket/out/config/args.yaml"
        launcher._s3_client.upload_file.assert_called_once()


class TestLaunch:
    """End-to-end ``launch()``: mocks every boto3 / SageMaker boundary.

    The launcher submits a single ``CreateTrainingJob`` boto3 call. We mock
    boto3.client('sagemaker'), the SageMaker session (for the region), and
    the S3 upload paths to keep the test hermetic.
    """

    def _patched_launcher(self, cfg):
        launcher = PerfTestSageMakerLauncher(cfg, role="arn:aws:iam::1:role/x")
        # Inject a session with the boto_session.region_name attribute used
        # by _submit_run.
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        launcher._session = mock_session
        launcher._s3_client = MagicMock()
        return launcher

    def test_launch_submits_create_training_job(self, tmp_path):
        cfg = _make_config(tmp_path)
        launcher = self._patched_launcher(cfg)

        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:test"}

        with patch.object(launcher_mod, "boto3") as mock_boto3:
            # boto3.client('sagemaker', ...) returns the SM client; any
            # other call (e.g. iam fallback) gets its own mock.
            mock_boto3.client.return_value = mock_sm
            job_name = launcher.launch(wait=False)

        assert job_name.startswith("perf-perf-test-baseline-")
        mock_sm.create_training_job.assert_called_once()
        request = mock_sm.create_training_job.call_args.kwargs
        # Top-level shape sanity.
        assert request["RoleArn"] == "arn:aws:iam::1:role/x"
        assert request["StoppingCondition"]["MaxRuntimeInSeconds"] > 0
        assert request["ResourceConfig"]["InstanceType"] == "ml.g5.2xlarge"
        assert request["ResourceConfig"]["InstanceCount"] == 1
        # Input channels: config, perf_data, model
        channel_names = {c["ChannelName"] for c in request["InputDataConfig"]}
        assert channel_names == {"config", "perf_data", "model"}
        # The bootstrap downloads + extracts the source tarball.
        bootstrap = request["AlgorithmSpecification"]["ContainerArguments"][0]
        assert "aws s3 cp" in bootstrap
        assert "perf_entrypoint.sh" in bootstrap
        # Region environment var was forwarded.
        assert request["Environment"]["SAGEMAKER_REGION"] == "us-west-2"
        # ACP IS031.1: in-transit encryption on by default.
        assert request["EnableInterContainerTrafficEncryption"] is True
        # Did not call the waiter on wait=False.
        mock_sm.get_waiter.assert_not_called()

    def test_launch_encryption_can_be_disabled(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.perf_test.encrypt_inter_container_traffic = False
        launcher = self._patched_launcher(cfg)

        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:test"}

        with patch.object(launcher_mod, "boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_sm
            launcher.launch(wait=False)

        request = mock_sm.create_training_job.call_args.kwargs
        assert request["EnableInterContainerTrafficEncryption"] is False

    def test_launch_waits_when_wait_true(self, tmp_path):
        cfg = _make_config(tmp_path)
        launcher = self._patched_launcher(cfg)

        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:test"}
        mock_waiter = MagicMock()
        mock_sm.get_waiter.return_value = mock_waiter

        with patch.object(launcher_mod, "boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_sm
            job_name = launcher.launch(wait=True)

        mock_sm.get_waiter.assert_called_once_with("training_job_completed_or_stopped")
        mock_waiter.wait.assert_called_once_with(TrainingJobName=job_name)

    def test_launch_skips_model_channel_for_hf_id(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.model.name = "Qwen/Qwen3-4B"
        launcher = self._patched_launcher(cfg)

        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:test"}

        with patch.object(launcher_mod, "boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_sm
            launcher.launch(wait=False)

        request = mock_sm.create_training_job.call_args.kwargs
        channel_names = {c["ChannelName"] for c in request["InputDataConfig"]}
        # No 'model' channel for HF-hub model IDs.
        assert channel_names == {"config", "perf_data"}
        assert request["Environment"]["PERF_TEST_MODEL_PATH"] == "Qwen/Qwen3-4B"

    def test_launch_injects_training_plan_arn_when_set(self, tmp_path):
        cfg = _make_config(tmp_path)
        plan = "arn:aws:sagemaker:us-west-2:1:training-plan/gnrl-purpose-p5en-6mnth"
        cfg.perf_test.training_plan_arn = plan
        launcher = self._patched_launcher(cfg)

        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:test"}

        with patch.object(launcher_mod, "boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_sm
            launcher.launch(wait=False)

        request = mock_sm.create_training_job.call_args.kwargs
        assert request["ResourceConfig"]["TrainingPlanArn"] == plan

    def test_launch_omits_training_plan_arn_when_unset(self, tmp_path):
        cfg = _make_config(tmp_path)
        launcher = self._patched_launcher(cfg)

        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:test"}

        with patch.object(launcher_mod, "boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_sm
            launcher.launch(wait=False)

        request = mock_sm.create_training_job.call_args.kwargs
        assert "TrainingPlanArn" not in request["ResourceConfig"]

    def test_launch_injects_vpc_config_when_set(self, tmp_path):
        """Perf jobs must be able to egress through the account's own NAT.

        Without VpcConfig the job uses SageMaker's service-managed NAT, whose
        region-wide aggregate traffic Maven Central rate-limits — Gradle then
        gets HTTP 429 resolving the Gatling plugin classpath and the load stage
        never runs.
        """
        cfg = _make_config(tmp_path)
        cfg.perf_test.vpc = VpcConfig(
            subnets=["subnet-a", "subnet-b"],
            security_groups=["sg-1"],
        )
        launcher = self._patched_launcher(cfg)

        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:test"}

        with patch.object(launcher_mod, "boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_sm
            launcher.launch(wait=False)

        request = mock_sm.create_training_job.call_args.kwargs
        assert request["VpcConfig"] == {
            "Subnets": ["subnet-a", "subnet-b"],
            "SecurityGroupIds": ["sg-1"],
        }

    def test_launch_omits_vpc_config_when_unset(self, tmp_path):
        cfg = _make_config(tmp_path)
        launcher = self._patched_launcher(cfg)

        mock_sm = MagicMock()
        mock_sm.create_training_job.return_value = {"TrainingJobArn": "arn:test"}

        with patch.object(launcher_mod, "boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_sm
            launcher.launch(wait=False)

        request = mock_sm.create_training_job.call_args.kwargs
        assert "VpcConfig" not in request
