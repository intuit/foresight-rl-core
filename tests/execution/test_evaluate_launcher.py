"""The SageMaker launcher for evaluate: the CreateTrainingJob request shape."""

from __future__ import annotations

from typing import Any, Dict
from unittest import mock

import pytest

from foresightrlcore.config.evaluate_schema import EvaluateJobConfig
from foresightrlcore.execution.evaluate_launcher import EvaluateSageMakerLauncher


def _config(**overrides: Any) -> EvaluateJobConfig:
    payload: Dict[str, Any] = {
        "job": {"name": "eval_demo", "max_runtime_hours": 4},
        "execution": {"type": "aws_training_job"},
        "output": {"s3_path": "s3://bucket/eval/"},
        "container": {"evaluate_image_uri": "123.dkr.ecr.us-west-2.amazonaws.com/e:v1"},
        "evaluate": {
            "benchmarks": [{"name": "demo_mcq"}],
            "models": [{"name": "claude", "provider": "bedrock"}],
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = {**payload[key], **value}
        else:
            payload[key] = value
    return EvaluateJobConfig.model_validate(payload)


@pytest.fixture
def submitted():
    """Submit a job with all AWS calls faked; yield a request-capturing helper."""
    sm_client = mock.MagicMock()
    sm_client.create_training_job.return_value = {"TrainingJobArn": "arn:job"}

    def _submit(config, wait=False):
        launcher = EvaluateSageMakerLauncher(
            config, role="arn:aws:iam::1:role/r", session=mock.MagicMock()
        )
        launcher.session.boto_session.region_name = "us-west-2"
        launcher._s3_client = mock.MagicMock()

        with mock.patch("boto3.client", return_value=sm_client):
            job_name = launcher.launch(wait=wait)

        return job_name, sm_client.create_training_job.call_args.kwargs, sm_client

    return _submit


def test_the_job_name_is_prefixed_and_timestamped(submitted):
    job_name, request, _ = submitted(_config())

    assert job_name.startswith("eval-eval-demo-")
    assert request["TrainingJobName"] == job_name
    # SageMaker rejects underscores in job names.
    assert "_" not in job_name


def test_a_long_job_name_is_truncated_to_the_service_limit(submitted):
    job_name, _, _ = submitted(_config(job={"name": "x" * 90}))

    assert len(job_name) <= 63


def test_the_container_pulls_the_source_tarball_and_execs_the_entrypoint(submitted):
    _, request, _ = submitted(_config())

    bootstrap = request["AlgorithmSpecification"]["ContainerArguments"][0]

    assert "evaluate_entrypoint.sh" in bootstrap
    assert '"$EVALUATE_SOURCE_S3"' in bootstrap


def test_the_contract_env_vars_point_at_container_paths(submitted):
    _, request, _ = submitted(_config())

    env = request["Environment"]

    assert env["EVALUATE_CONFIG"] == "/opt/ml/input/data/config/args.yaml"
    assert env["EVALUATE_OUTPUT_DIR"] == "/opt/ml/output/data"
    assert env["EVALUATE_CHECKPOINT_DIR"] == "/opt/ml/checkpoints"
    assert env["EVALUATE_SOURCE_S3"].startswith("s3://bucket/eval/")
    assert env["SAGEMAKER_REGION"] == "us-west-2"


def test_user_environment_is_carried_through(submitted):
    _, request, _ = submitted(_config(environment={"GENOS_ENV": "prd"}))

    assert request["Environment"]["GENOS_ENV"] == "prd"


def test_an_extension_package_is_announced_to_the_job(submitted):
    with mock.patch("foresightrlcore.execution.evaluate_launcher.stage_extra_packages"):
        _, request, _ = submitted(
            _config(container={"extra_packages": ["foresightrl_intuit"]})
        )

    assert request["Environment"]["FORESIGHT_EXTRA_PACKAGES"] == "foresightrl_intuit"


# ---------------------------------------------------------------------------
# Infra sizing
# ---------------------------------------------------------------------------


def test_an_api_only_run_takes_a_cpu_instance_and_the_light_image(submitted):
    _, request, _ = submitted(_config())

    assert request["ResourceConfig"]["InstanceType"] == "ml.m5.2xlarge"
    assert (
        request["AlgorithmSpecification"]["TrainingImage"]
        == "123.dkr.ecr.us-west-2.amazonaws.com/e:v1"
    )


def test_a_locally_served_candidate_takes_a_gpu_instance_and_the_vllm_image(submitted):
    _, request, _ = submitted(
        _config(
            container={
                "perf_test_image_uri": "123.dkr.ecr.us-west-2.amazonaws.com/g:v1"
            },
            evaluate={
                "benchmarks": [{"name": "demo_mcq"}],
                "models": [{"name": "Qwen/Qwen3-8B", "provider": "local"}],
            },
        )
    )

    assert request["ResourceConfig"]["InstanceType"].startswith("ml.g")
    assert request["AlgorithmSpecification"]["TrainingImage"].endswith("/g:v1")


def test_local_serving_without_a_gpu_image_is_refused_before_submission():
    config = _config(
        container={"evaluate_image_uri": None},
        evaluate={
            "benchmarks": [{"name": "demo_mcq"}],
            "models": [{"name": "Qwen/Qwen3-8B", "provider": "local"}],
        },
    )
    launcher = EvaluateSageMakerLauncher(config, role="r", session=mock.MagicMock())

    with pytest.raises(ValueError, match="vLLM-capable GPU image"):
        launcher._resolve_image_uri()


def test_a_run_with_no_image_at_all_is_refused():
    launcher = EvaluateSageMakerLauncher(
        _config(container={"evaluate_image_uri": None}),
        role="r",
        session=mock.MagicMock(),
    )

    with pytest.raises(ValueError, match="requires a container image"):
        launcher._resolve_image_uri()


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------


def test_the_config_is_mounted_as_an_input_channel(submitted):
    _, request, _ = submitted(_config())

    channels = {c["ChannelName"] for c in request["InputDataConfig"]}
    assert channels == {"config"}


def test_s3_staged_weights_are_mounted_as_a_model_channel(submitted):
    _, request, _ = submitted(
        _config(
            container={"perf_test_image_uri": "img"},
            evaluate={
                "benchmarks": [{"name": "demo_mcq"}],
                "models": [{"name": "s3://bucket/models/qwen/", "provider": "local"}],
            },
        )
    )

    by_name = {c["ChannelName"]: c for c in request["InputDataConfig"]}
    assert by_name["model"]["DataSource"]["S3DataSource"]["S3Uri"] == (
        "s3://bucket/models/qwen/"
    )


# ---------------------------------------------------------------------------
# Optional blocks
# ---------------------------------------------------------------------------


def test_a_checkpoint_path_enables_resume_across_restarts(submitted):
    _, request, _ = submitted(
        _config(
            output={"s3_path": "s3://bucket/eval/", "checkpoint_s3_path": "s3://b/ck/"}
        )
    )

    assert request["CheckpointConfig"] == {
        "S3Uri": "s3://b/ck/",
        "LocalPath": "/opt/ml/checkpoints",
    }


def test_resume_is_wired_up_without_being_asked_for(submitted):
    """A config that says nothing about checkpoints still gets resume.

    The ``CheckpointConfig`` block is what makes resume real: it is what
    downloads a previous run's rows into the container before the runner looks
    for them. Without it the runner still writes checkpoints -- to container
    disk that dies with the job -- so an operator who never set a checkpoint
    path had a run that appeared to be checkpointing and could not resume.
    """
    _, request, _ = submitted(_config())

    assert request["CheckpointConfig"] == {
        "S3Uri": "s3://bucket/eval/checkpoints/",
        "LocalPath": "/opt/ml/checkpoints",
    }


def test_no_checkpoint_block_when_checkpointing_is_switched_off(submitted):
    """The opt-out for a deliberately clean, non-resuming measurement."""
    _, request, _ = submitted(
        _config(
            evaluate={
                "benchmarks": [{"name": "demo_mcq"}],
                "models": [{"name": "claude", "provider": "bedrock"}],
                "checkpoint": {"enabled": False},
            }
        )
    )

    assert "CheckpointConfig" not in request


def test_a_vpc_is_passed_through(submitted):
    _, request, _ = submitted(
        _config(
            evaluate={
                "benchmarks": [{"name": "demo_mcq"}],
                "models": [{"name": "claude"}],
                "vpc": {"subnets": ["subnet-1"], "security_groups": ["sg-1"]},
            }
        )
    )

    assert request["VpcConfig"] == {
        "Subnets": ["subnet-1"],
        "SecurityGroupIds": ["sg-1"],
    }


def test_a_training_plan_is_passed_through(submitted):
    _, request, _ = submitted(
        _config(
            evaluate={
                "benchmarks": [{"name": "demo_mcq"}],
                "models": [{"name": "claude"}],
                "training_plan_arn": "arn:aws:sagemaker:us-west-2:1:training-plan/p",
            }
        )
    )

    assert request["ResourceConfig"]["TrainingPlanArn"].endswith("training-plan/p")


def test_inter_container_encryption_is_on_by_default(submitted):
    """ACP IS031.1."""
    _, request, _ = submitted(_config())

    assert request["EnableInterContainerTrafficEncryption"] is True


def test_encryption_can_be_switched_off_explicitly(submitted):
    _, request, _ = submitted(
        _config(
            evaluate={
                "benchmarks": [{"name": "demo_mcq"}],
                "models": [{"name": "claude"}],
                "encrypt_inter_container_traffic": False,
            }
        )
    )

    assert request["EnableInterContainerTrafficEncryption"] is False


def test_the_runtime_limit_comes_from_the_job_block(submitted):
    _, request, _ = submitted(_config())

    assert request["StoppingCondition"]["MaxRuntimeInSeconds"] == 4 * 3600


def test_waiting_polls_the_service_waiter(submitted):
    _, _, sm_client = submitted(_config(), wait=True)

    sm_client.get_waiter.assert_called_once_with("training_job_completed_or_stopped")


# ---------------------------------------------------------------------------
# Source staging
# ---------------------------------------------------------------------------


def test_the_staged_source_carries_the_package_and_the_entrypoint():
    import os

    launcher = EvaluateSageMakerLauncher(_config(), role="r", session=mock.MagicMock())
    source_dir = launcher._source_dir

    assert os.path.exists(os.path.join(source_dir, "evaluate_entrypoint.sh"))
    assert os.path.exists(
        os.path.join(source_dir, "foresightrlcore", "evaluate", "runner.py")
    )
    # The runtime scripts are copied flat, not again under the package.
    assert not os.path.exists(os.path.join(source_dir, "foresightrlcore", "runtime"))


def test_recipe_requirements_are_written_for_the_container(tmp_path):
    launcher = EvaluateSageMakerLauncher(
        _config(container={"recipe": "qwen3_5"}), role="r", session=mock.MagicMock()
    )

    launcher._write_resolved_requirements(str(tmp_path))

    written = (tmp_path / "requirements.txt").read_text()
    assert written.strip()  # a recipe resolves to a real requirement set


def test_an_s3_uri_that_is_not_one_is_reported():
    with pytest.raises(ValueError, match="Not an S3 URI"):
        EvaluateSageMakerLauncher._parse_s3_uri("/local/path")


def test_a_configured_role_beats_auto_detection():
    """A pinned role must not be overridden by the caller's identity.

    Auto-detection resolves to whoever launched the job, so the same config
    submitted from two shells can run as two different roles -- which is how a
    gate whose IDPS policy admits one role only came to fail on a run where
    nothing about the config had changed.
    """
    config = _config(evaluate={"role": "arn:aws:iam::1:role/Pinned"})
    launcher = EvaluateSageMakerLauncher(config, session=mock.MagicMock())

    with mock.patch(
        "sagemaker.core.helper.session_helper.get_execution_role",
        return_value="arn:aws:iam::1:role/Detected",
    ):
        assert launcher.role == "arn:aws:iam::1:role/Pinned"


def test_an_unset_role_still_falls_back_to_auto_detection():
    config = _config()
    assert config.evaluate.role is None
    launcher = EvaluateSageMakerLauncher(config, session=mock.MagicMock())

    with mock.patch(
        "sagemaker.core.helper.session_helper.get_execution_role",
        return_value="arn:aws:iam::1:role/Detected",
    ):
        assert launcher.role == "arn:aws:iam::1:role/Detected"
