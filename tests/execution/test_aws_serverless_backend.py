"""Tests for the AWS Serverless Model Customization backend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import yaml

from foresightrlcore.config.schema import TrainingJobConfig
from foresightrlcore.config.validation import validate_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _serverless_config_dict(
    method: str = "rlvr",
    model_id: str = "huggingface-llm-qwen2-5-7b-instruct",
    accept_eula: bool = True,
    training_dataset_name: str = "rlvr-train",
    model_package_group: str = "huggingface-llm-qwen2-5-7b-instruct-rlvr",
    **extra_serverless,
) -> dict:
    """Build a minimal serverless config dict."""
    sl = {
        "model_id": model_id,
        "accept_eula": accept_eula,
        "training_dataset_name": training_dataset_name,
        "model_package_group": model_package_group,
        **extra_serverless,
    }
    return {
        "job": {"name": "test-serverless"},
        "model": {"name": model_id},
        "method": {"type": method},
        "execution": {"type": "aws_serverless", "serverless": sl},
    }


@pytest.fixture
def rlvr_config_dict():
    return _serverless_config_dict(method="rlvr")


@pytest.fixture
def dpo_config_dict():
    return _serverless_config_dict(
        method="dpo",
        model_id="meta-llama-3-2-1b-instruct",
        training_dataset_name="humanlike-dpo-train",
        model_package_group="humanlike-dpo-llama32-1b",
        training_type="lora",
    )


@pytest.fixture
def rlaif_config_dict():
    return _serverless_config_dict(
        method="rlaif",
        model_id="huggingface-llm-qwen2-5-7b-instruct",
        training_dataset_name="humanlike-rlaif-train",
        model_package_group="humanlike-rlaif-qwen25-7b",
        reward_model_id="openai.gpt-oss-120b-1:0",
        reward_prompt_name="humanlike-rlaif-reward-prompt",
    )


@pytest.fixture
def rlvr_config_yaml(tmp_yaml, rlvr_config_dict):
    return tmp_yaml(yaml.dump(rlvr_config_dict))


@pytest.fixture
def dpo_config_yaml(tmp_yaml, dpo_config_dict):
    return tmp_yaml(yaml.dump(dpo_config_dict))


@pytest.fixture
def rlaif_config_yaml(tmp_yaml, rlaif_config_dict):
    return tmp_yaml(yaml.dump(rlaif_config_dict))


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestServerlessConfigSchema:
    """Parsing and validation of serverless config YAML."""

    def test_parse_rlvr(self, rlvr_config_yaml):
        config = TrainingJobConfig.from_yaml(rlvr_config_yaml)
        assert config.execution.type == "aws_serverless"
        assert config.execution.serverless is not None
        sl = config.execution.serverless
        assert sl.model_id == "huggingface-llm-qwen2-5-7b-instruct"
        assert sl.accept_eula is True
        assert sl.training_dataset_name == "rlvr-train"
        assert config.method.type == "rlvr"

    def test_parse_dpo(self, dpo_config_yaml):
        config = TrainingJobConfig.from_yaml(dpo_config_yaml)
        assert config.method.type == "dpo"
        assert config.execution.serverless is not None
        assert config.execution.serverless.training_type == "lora"

    def test_parse_rlaif(self, rlaif_config_yaml):
        config = TrainingJobConfig.from_yaml(rlaif_config_yaml)
        assert config.method.type == "rlaif"
        sl = config.execution.serverless
        assert sl is not None
        assert sl.reward_model_id == "openai.gpt-oss-120b-1:0"
        assert sl.reward_prompt_name == "humanlike-rlaif-reward-prompt"

    def test_serverless_hyperparameters(self, tmp_yaml):
        d = _serverless_config_dict()
        d["training"] = {
            "serverless": {
                "global_batch_size": 64,
                "max_epochs": 3,
                "learning_rate": 1e-4,
                "extra": {"warmup_steps": 10},
            }
        }
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        hp = config.training.serverless
        assert hp is not None
        assert hp.global_batch_size == 64
        assert hp.max_epochs == 3
        assert hp.learning_rate == 1e-4
        assert hp.extra == {"warmup_steps": 10}

    def test_data_and_task_optional_for_serverless(self, tmp_yaml):
        """data.train_path, output.s3_path, and task are all optional for serverless."""
        d = _serverless_config_dict()
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        assert config.data.train_path is None
        assert config.output.s3_path is None
        assert config.task is None

    def test_future_proofing_fields(self, tmp_yaml):
        d = _serverless_config_dict(
            evaluator_model="amazon.nova-pro-v1:0",
            benchmark="MATH",
            inference_instance_type="ml.g5.2xlarge",
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        sl = config.execution.serverless
        assert sl is not None
        assert sl.evaluator_model == "amazon.nova-pro-v1:0"
        assert sl.benchmark == "MATH"
        assert sl.inference_instance_type == "ml.g5.2xlarge"

    def test_job_config_fields(self, tmp_yaml):
        """Test base_job_name, kms, vpc, stopping, mlflow_resource_arn, tags, wait_timeout."""
        d = _serverless_config_dict(
            base_job_name="my-custom-prefix",
            kms_key_id="arn:aws:kms:us-west-2:123:key/abc",
            networking={"subnets": ["subnet-1"], "security_groups": ["sg-1"]},
            max_runtime_seconds=7200,
            wait_timeout=3600,
            mlflow_resource_arn="arn:aws:sagemaker:us-west-2:123:mlflow/server",
            tags={"team": "ml", "env": "dev"},
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        sl = config.execution.serverless
        assert sl is not None
        assert sl.base_job_name == "my-custom-prefix"
        assert sl.kms_key_id == "arn:aws:kms:us-west-2:123:key/abc"
        assert sl.networking is not None
        assert sl.networking.subnets == ["subnet-1"]
        assert sl.networking.security_groups == ["sg-1"]
        assert sl.max_runtime_seconds == 7200
        assert sl.wait_timeout == 3600
        assert sl.mlflow_resource_arn == "arn:aws:sagemaker:us-west-2:123:mlflow/server"
        assert sl.tags == {"team": "ml", "env": "dev"}

    def test_rlvr_custom_reward_function(self, tmp_yaml):
        d = _serverless_config_dict(
            method="rlvr",
            custom_reward_function_name="my-evaluator",
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        sl = config.execution.serverless
        assert sl is not None
        assert sl.custom_reward_function_name == "my-evaluator"

    def test_training_type_full(self, tmp_yaml):
        d = _serverless_config_dict(training_type="full")
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        assert config.execution.serverless is not None
        assert config.execution.serverless.training_type == "full"

    def test_dataset_source_fields(self, tmp_yaml):
        d = _serverless_config_dict(
            training_dataset_source="s3://bucket/train.jsonl",
            validation_dataset_source="./local/val.jsonl",
            test_dataset_source="s3://bucket/test.jsonl",
            customization_technique="dpo",
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        sl = config.execution.serverless
        assert sl is not None
        assert sl.training_dataset_source == "s3://bucket/train.jsonl"
        assert sl.validation_dataset_source == "./local/val.jsonl"
        assert sl.test_dataset_source == "s3://bucket/test.jsonl"
        assert sl.customization_technique == "dpo"

    def test_reward_prompt_text_field(self, tmp_yaml):
        d = _serverless_config_dict(
            method="rlaif",
            reward_model_id="openai.gpt-oss-120b-1:0",
            reward_prompt_name="my-prompt",
            reward_prompt_text="Rate the response on a scale of 1-10.",
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        sl = config.execution.serverless
        assert sl is not None
        assert sl.reward_prompt_text == "Rate the response on a scale of 1-10."
        assert sl.reward_prompt_name == "my-prompt"

    def test_reward_prompt_source_field(self, tmp_yaml):
        d = _serverless_config_dict(
            method="rlaif",
            reward_model_id="openai.gpt-oss-120b-1:0",
            reward_prompt_name="my-prompt",
            reward_prompt_source="s3://bucket/prompt.txt",
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        sl = config.execution.serverless
        assert sl is not None
        assert sl.reward_prompt_source == "s3://bucket/prompt.txt"


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestServerlessValidation:
    """Config validation for serverless backend."""

    def test_valid_rlvr(self, rlvr_config_yaml):
        config = TrainingJobConfig.from_yaml(rlvr_config_yaml)
        errors = validate_config(config)
        assert errors == []

    def test_valid_dpo(self, dpo_config_yaml):
        config = TrainingJobConfig.from_yaml(dpo_config_yaml)
        errors = validate_config(config)
        assert errors == []

    def test_valid_rlaif(self, rlaif_config_yaml):
        config = TrainingJobConfig.from_yaml(rlaif_config_yaml)
        errors = validate_config(config)
        assert errors == []

    def test_missing_serverless_section(self, tmp_yaml):
        d = {
            "job": {"name": "test"},
            "model": {"name": "m"},
            "method": {"type": "rlvr"},
            "execution": {"type": "aws_serverless"},
        }
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        errors = validate_config(config)
        assert any("execution.serverless section is required" in e for e in errors)

    def test_unsupported_method(self, tmp_yaml):
        d = _serverless_config_dict(method="grpo")
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        errors = validate_config(config)
        assert any("Serverless backend supports methods" in e for e in errors)

    def test_eula_not_accepted(self, tmp_yaml):
        d = _serverless_config_dict(accept_eula=False)
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        errors = validate_config(config)
        assert any("accept_eula must be true" in e for e in errors)

    def test_no_dataset_source(self, tmp_yaml):
        d = _serverless_config_dict(training_dataset_name="")
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        errors = validate_config(config)
        assert any("training_dataset_name" in e for e in errors)

    def test_rlaif_missing_reward_model(self, tmp_yaml):
        d = _serverless_config_dict(
            method="rlaif",
            reward_prompt_name="prompt",
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        errors = validate_config(config)
        assert any("RLAIF requires a judge model" in e for e in errors)

    def test_rlaif_judge_model_id_via_extra(self, tmp_yaml):
        """RLAIF without reward_model_id is OK if judge_model_id is set
        in training.serverless.extra (per AWS guidance for non-allowlisted
        Bedrock judges like Claude inference profiles)."""
        d = _serverless_config_dict(
            method="rlaif",
            reward_prompt_name="prompt",
        )
        d["training"] = {
            "serverless": {
                "extra": {
                    "judge_model_id": "us.anthropic.claude-opus-4-5-20251101-v1:0",
                },
            }
        }
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        errors = validate_config(config)
        assert not any("RLAIF requires a judge model" in e for e in errors)

    def test_rlaif_missing_reward_prompt(self, tmp_yaml):
        d = _serverless_config_dict(
            method="rlaif",
            reward_model_id="openai.gpt-oss-120b-1:0",
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        errors = validate_config(config)
        assert any("reward_prompt_name is required" in e for e in errors)

    def test_infra_validation_skipped_for_serverless(self, tmp_yaml):
        """Serverless configs should not trigger infra/vllm/topology validation."""
        d = _serverless_config_dict()
        d["infra"] = {
            "topology": "dedicated_vllm",
            "instance_count": 1,
        }
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        errors = validate_config(config)
        assert not any("dedicated_vllm" in e for e in errors)

    def test_data_train_path_still_required_for_aws_training_job(self, tmp_yaml):
        """Non-serverless backends must still have data.train_path."""
        d = {
            "job": {"name": "test"},
            "model": {"name": "m"},
            "method": {"type": "grpo"},
            "execution": {"type": "aws_training_job"},
            "task": {"type": "advice"},
        }
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        errors = validate_config(config)
        assert any("data.train_path is required" in e for e in errors)

    def test_serverless_output_must_be_s3(self, tmp_yaml):
        d = _serverless_config_dict()
        d["output"] = {"s3_path": "/local/path"}
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        errors = validate_config(config)
        assert any("s3://" in e for e in errors)

    def test_invalid_training_type(self, tmp_yaml):
        d = _serverless_config_dict(training_type="qlora")
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        errors = validate_config(config)
        assert any("training_type must be one of" in e for e in errors)

    def test_invalid_customization_technique(self, tmp_yaml):
        d = _serverless_config_dict(customization_technique="invalid")
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        errors = validate_config(config)
        assert any("customization_technique must be one of" in e for e in errors)

    def test_dataset_source_without_name(self, tmp_yaml):
        d = _serverless_config_dict(
            training_dataset_name="",
            training_dataset_source="s3://bucket/data.jsonl",
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        errors = validate_config(config)
        assert any("training_dataset_name is required" in e for e in errors)


# ---------------------------------------------------------------------------
# Backend protocol tests
# ---------------------------------------------------------------------------


class TestAwsServerlessBackendProtocol:
    """Verify AwsServerlessBackend conforms to ExecutionBackend protocol."""

    def test_has_required_attributes(self):
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        backend = AwsServerlessBackend()
        assert backend.backend_type == "aws_serverless"
        assert hasattr(backend, "validate")
        assert hasattr(backend, "launch")
        assert hasattr(backend, "get_status")

    def test_registered_in_registry(self):
        import foresightrlcore  # noqa: F401 — triggers registration
        from foresightrlcore.execution import get_backend

        cls = get_backend("aws_serverless")
        assert hasattr(cls, "backend_type")
        assert cls.backend_type == "aws_serverless"


# ---------------------------------------------------------------------------
# Backend validate() tests (SDK mocked)
# ---------------------------------------------------------------------------


class TestAwsServerlessBackendValidate:
    """Test validate() method with mocked SDK."""

    def test_validate_passes_for_valid_rlvr(self, rlvr_config_yaml):
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        config = TrainingJobConfig.from_yaml(rlvr_config_yaml)
        backend = AwsServerlessBackend()
        errors = backend.validate(config)
        if not any("sagemaker" in e.lower() for e in errors):
            assert errors == []

    def test_validate_rejects_missing_serverless(self, tmp_yaml):
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        d = {
            "job": {"name": "test"},
            "model": {"name": "m"},
            "method": {"type": "rlvr"},
            "execution": {"type": "aws_serverless"},
        }
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        backend = AwsServerlessBackend()
        errors = backend.validate(config)
        has_dep_error = any("not installed" in e for e in errors)
        has_config_error = any("serverless" in e for e in errors)
        assert has_dep_error or has_config_error


# ---------------------------------------------------------------------------
# Backend launch() tests (fully mocked)
# ---------------------------------------------------------------------------


class TestAwsServerlessBackendLaunch:
    """Test launch() with fully mocked SageMaker SDK."""

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    @patch("foresightrlcore.execution.aws_serverless.AwsServerlessBackend._get_role")
    def test_launch_rlvr(self, mock_get_role, rlvr_config_yaml):
        mock_get_role.return_value = "arn:aws:iam::123:role/test"

        mock_session = MagicMock()
        mock_dataset = MagicMock()
        mock_mpg = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_training_job = MagicMock()
        mock_training_job.training_job_name = "rlvr-job-123"
        mock_trainer_instance.train.return_value = mock_training_job

        with (
            patch(
                "sagemaker.core.helper.session_helper.Session",
                return_value=mock_session,
            ),
            patch(
                "sagemaker.ai_registry.dataset.DataSet.get",
                return_value=mock_dataset,
            ),
            patch(
                "sagemaker.core.resources.ModelPackageGroup.get",
                return_value=mock_mpg,
            ),
            patch(
                "sagemaker.train.common.TrainingType",
            ) as mock_tt,
            patch(
                "sagemaker.train.rlvr_trainer.RLVRTrainer",
                return_value=mock_trainer_instance,
            ) as mock_trainer_cls,
        ):
            mock_tt.LORA = "LORA"
            mock_tt.FULL = "FULL"
            from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

            config = TrainingJobConfig.from_yaml(rlvr_config_yaml)
            backend = AwsServerlessBackend()
            job_name = backend.launch(config, wait=False)

            assert job_name == "rlvr-job-123"
            mock_trainer_instance.train.assert_called_once_with(wait=False)
            call_kwargs = mock_trainer_cls.call_args[1]
            assert call_kwargs["training_type"] == "LORA"
            assert call_kwargs["sagemaker_session"] is mock_session

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    @patch("foresightrlcore.execution.aws_serverless.AwsServerlessBackend._get_role")
    def test_launch_rlaif_resolves_reward_prompt(
        self, mock_get_role, rlaif_config_yaml
    ):
        mock_get_role.return_value = "arn:aws:iam::123:role/test"

        mock_session = MagicMock()
        mock_dataset = MagicMock()
        mock_mpg = MagicMock()
        mock_evaluator = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_training_job = MagicMock()
        mock_training_job.training_job_name = "rlaif-job-456"
        mock_trainer_instance.train.return_value = mock_training_job

        with (
            patch(
                "sagemaker.core.helper.session_helper.Session",
                return_value=mock_session,
            ),
            patch(
                "sagemaker.ai_registry.dataset.DataSet.get",
                return_value=mock_dataset,
            ),
            patch(
                "sagemaker.core.resources.ModelPackageGroup.get",
                return_value=mock_mpg,
            ),
            patch(
                "sagemaker.ai_registry.evaluator.Evaluator.get",
                return_value=mock_evaluator,
            ),
            patch(
                "sagemaker.train.common.TrainingType",
            ) as mock_tt,
            patch(
                "sagemaker.train.rlaif_trainer.RLAIFTrainer",
                return_value=mock_trainer_instance,
            ) as mock_trainer_cls,
        ):
            mock_tt.LORA = "LORA"
            mock_tt.FULL = "FULL"
            from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

            config = TrainingJobConfig.from_yaml(rlaif_config_yaml)
            backend = AwsServerlessBackend()
            job_name = backend.launch(config, wait=False)

            assert job_name == "rlaif-job-456"
            mock_trainer_instance.train.assert_called_once_with(wait=False)
            call_kwargs = mock_trainer_cls.call_args[1]
            assert call_kwargs["reward_model_id"] == "openai.gpt-oss-120b-1:0"
            assert call_kwargs["reward_prompt"] is mock_evaluator

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    @patch("foresightrlcore.execution.aws_serverless.AwsServerlessBackend._get_role")
    def test_launch_rlaif_omits_reward_model_id_when_unset(
        self, mock_get_role, tmp_yaml
    ):
        """When reward_model_id is not set, the SDK kwarg must be omitted
        so the SDK's allowlist validation is skipped (AWS guidance for
        non-allowlisted Bedrock judges)."""
        mock_get_role.return_value = "arn:aws:iam::123:role/test"

        d = _serverless_config_dict(
            method="rlaif",
            model_id="huggingface-llm-qwen2-5-7b-instruct",
            training_dataset_name="rlaif-train",
            model_package_group="rlaif-mpg",
            reward_prompt_name="rlaif-prompt",
        )
        d["training"] = {
            "serverless": {
                "extra": {
                    "judge_model_id": ("us.anthropic.claude-opus-4-5-20251101-v1:0"),
                },
            }
        }

        mock_session = MagicMock()
        mock_dataset = MagicMock()
        mock_mpg = MagicMock()
        mock_evaluator = MagicMock()
        mock_hp = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.hyperparameters = mock_hp
        mock_training_job = MagicMock()
        mock_training_job.training_job_name = "rlaif-no-rm-id"
        mock_trainer_instance.train.return_value = mock_training_job

        with (
            patch(
                "sagemaker.core.helper.session_helper.Session",
                return_value=mock_session,
            ),
            patch(
                "sagemaker.ai_registry.dataset.DataSet.get",
                return_value=mock_dataset,
            ),
            patch(
                "sagemaker.core.resources.ModelPackageGroup.get",
                return_value=mock_mpg,
            ),
            patch(
                "sagemaker.ai_registry.evaluator.Evaluator.get",
                return_value=mock_evaluator,
            ),
            patch("sagemaker.train.common.TrainingType") as mock_tt,
            patch(
                "sagemaker.train.rlaif_trainer.RLAIFTrainer",
                return_value=mock_trainer_instance,
            ) as mock_trainer_cls,
        ):
            mock_tt.LORA = "LORA"
            mock_tt.FULL = "FULL"
            from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

            config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
            backend = AwsServerlessBackend()
            backend.launch(config, wait=False)

            call_kwargs = mock_trainer_cls.call_args[1]
            assert "reward_model_id" not in call_kwargs
            assert call_kwargs["reward_prompt"] is mock_evaluator
            assert mock_hp.judge_model_id == (
                "us.anthropic.claude-opus-4-5-20251101-v1:0"
            )

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    @patch("foresightrlcore.execution.aws_serverless.AwsServerlessBackend._get_role")
    def test_launch_applies_hyperparameters(self, mock_get_role, tmp_yaml):
        mock_get_role.return_value = "arn:aws:iam::123:role/test"

        d = _serverless_config_dict()
        d["training"] = {
            "serverless": {
                "global_batch_size": 128,
                "max_epochs": 5,
            }
        }

        mock_session = MagicMock()
        mock_dataset = MagicMock()
        mock_mpg = MagicMock()
        mock_hp = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.hyperparameters = mock_hp
        mock_training_job = MagicMock()
        mock_training_job.training_job_name = "rlvr-hp-test"
        mock_trainer_instance.train.return_value = mock_training_job

        with (
            patch(
                "sagemaker.core.helper.session_helper.Session",
                return_value=mock_session,
            ),
            patch(
                "sagemaker.ai_registry.dataset.DataSet.get",
                return_value=mock_dataset,
            ),
            patch(
                "sagemaker.core.resources.ModelPackageGroup.get",
                return_value=mock_mpg,
            ),
            patch(
                "sagemaker.train.common.TrainingType",
            ) as mock_tt,
            patch(
                "sagemaker.train.rlvr_trainer.RLVRTrainer",
                return_value=mock_trainer_instance,
            ),
        ):
            mock_tt.LORA = "LORA"
            mock_tt.FULL = "FULL"
            from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

            config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
            backend = AwsServerlessBackend()
            backend.launch(config, wait=False)

            assert mock_hp.global_batch_size == 128
            assert mock_hp.max_epochs == 5

    def test_set_hp_widens_enum_when_value_outside(self):
        """`_set_hp` widens the SDK's `_specs[name]['enum']` in place when
        the requested value isn't in it, then assigns. This lets us bypass
        SageMaker SDK v3's client-side enum guard (which AWS server-side
        does not enforce). See `_set_hp` docstring for context."""

        class _FakeFineTuningOptions:
            """Minimal stand-in for sagemaker.train.common.FineTuningOptions.

            Mirrors its enum-validating __setattr__ closely enough to prove
            the helper works end-to-end."""

            def __init__(self, specs):
                object.__setattr__(self, "_specs", specs)

            def __setattr__(self, name, value):
                if name.startswith("_"):
                    object.__setattr__(self, name, value)
                    return
                spec = self._specs.get(name)  # type: ignore[attr-defined]
                if (
                    isinstance(spec, dict)
                    and "enum" in spec
                    and value not in spec["enum"]
                ):
                    raise ValueError(
                        f"{name} must be one of {spec['enum']}, got {value}"
                    )
                object.__setattr__(self, name, value)

        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        hp = _FakeFineTuningOptions(
            {
                "global_batch_size": {
                    "default": 128,
                    "enum": [128, 256, 512, 1024],
                    "type": "integer",
                },
                "max_epochs": {"default": 1, "min": 1, "max": 100, "type": "integer"},
            }
        )

        with pytest.raises(ValueError, match="must be one of"):
            hp.global_batch_size = 64

        AwsServerlessBackend._set_hp(hp, "global_batch_size", 64)
        assert hp.global_batch_size == 64  # type: ignore[attr-defined]
        assert 64 in hp._specs["global_batch_size"]["enum"]  # type: ignore[attr-defined]

        AwsServerlessBackend._set_hp(hp, "max_epochs", 5)
        assert hp.max_epochs == 5  # type: ignore[attr-defined]

        AwsServerlessBackend._set_hp(hp, "global_batch_size", 256)
        assert hp.global_batch_size == 256  # type: ignore[attr-defined]
        assert hp._specs["global_batch_size"]["enum"].count(64) == 1  # type: ignore[attr-defined]

    def test_set_hp_no_specs_attribute(self):
        """When the target object has no `_specs` (e.g. a MagicMock without
        one), `_set_hp` falls through to plain setattr."""
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        class _Plain:
            pass

        hp = _Plain()
        AwsServerlessBackend._set_hp(hp, "global_batch_size", 64)
        assert hp.global_batch_size == 64  # type: ignore[attr-defined]

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    @patch("foresightrlcore.execution.aws_serverless.AwsServerlessBackend._get_role")
    def test_launch_passes_job_config_fields(self, mock_get_role, tmp_yaml):
        """Verify base_job_name, kms, networking, stopping, tags are passed to trainer."""
        mock_get_role.return_value = "arn:aws:iam::123:role/test"

        d = _serverless_config_dict(
            base_job_name="my-prefix",
            kms_key_id="arn:aws:kms:us-west-2:123:key/abc",
            networking={"subnets": ["subnet-1"], "security_groups": ["sg-1"]},
            max_runtime_seconds=7200,
            mlflow_resource_arn="arn:aws:sagemaker:us-west-2:123:mlflow/s",
            tags={"team": "ml"},
        )

        mock_session = MagicMock()
        mock_dataset = MagicMock()
        mock_mpg = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_training_job = MagicMock()
        mock_training_job.training_job_name = "cfg-test-job"
        mock_trainer_instance.train.return_value = mock_training_job

        with (
            patch(
                "sagemaker.core.helper.session_helper.Session",
                return_value=mock_session,
            ),
            patch(
                "sagemaker.ai_registry.dataset.DataSet.get",
                return_value=mock_dataset,
            ),
            patch(
                "sagemaker.core.resources.ModelPackageGroup.get",
                return_value=mock_mpg,
            ),
            patch(
                "sagemaker.train.common.TrainingType",
            ) as mock_tt,
            patch(
                "sagemaker.train.rlvr_trainer.RLVRTrainer",
                return_value=mock_trainer_instance,
            ) as mock_trainer_cls,
            patch(
                "sagemaker.core.shapes.VpcConfig",
            ) as mock_vpc_cls,
            patch(
                "sagemaker.train.configs.StoppingCondition",
            ) as mock_stop_cls,
        ):
            mock_tt.LORA = "LORA"
            mock_tt.FULL = "FULL"
            from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

            config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
            backend = AwsServerlessBackend()
            backend.launch(config, wait=False)

            call_kwargs = mock_trainer_cls.call_args[1]
            assert call_kwargs["base_job_name"] == "my-prefix"
            assert call_kwargs["kms_key_id"] == "arn:aws:kms:us-west-2:123:key/abc"
            assert call_kwargs["mlflow_resource_arn"] == (
                "arn:aws:sagemaker:us-west-2:123:mlflow/s"
            )
            assert call_kwargs["networking"] is mock_vpc_cls.return_value
            assert call_kwargs["stopping_condition"] is mock_stop_cls.return_value
            assert call_kwargs["tags"] == [{"Key": "team", "Value": "ml"}]

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    @patch("foresightrlcore.execution.aws_serverless.AwsServerlessBackend._get_role")
    def test_launch_rlvr_with_custom_reward_function(self, mock_get_role, tmp_yaml):
        mock_get_role.return_value = "arn:aws:iam::123:role/test"

        d = _serverless_config_dict(
            method="rlvr",
            custom_reward_function_name="my-evaluator",
        )

        mock_session = MagicMock()
        mock_dataset = MagicMock()
        mock_mpg = MagicMock()
        mock_evaluator = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_training_job = MagicMock()
        mock_training_job.training_job_name = "rlvr-crf-test"
        mock_trainer_instance.train.return_value = mock_training_job

        with (
            patch(
                "sagemaker.core.helper.session_helper.Session",
                return_value=mock_session,
            ),
            patch(
                "sagemaker.ai_registry.dataset.DataSet.get",
                return_value=mock_dataset,
            ),
            patch(
                "sagemaker.core.resources.ModelPackageGroup.get",
                return_value=mock_mpg,
            ),
            patch(
                "sagemaker.ai_registry.evaluator.Evaluator.get",
                return_value=mock_evaluator,
            ),
            patch(
                "sagemaker.train.common.TrainingType",
            ) as mock_tt,
            patch(
                "sagemaker.train.rlvr_trainer.RLVRTrainer",
                return_value=mock_trainer_instance,
            ) as mock_trainer_cls,
        ):
            mock_tt.LORA = "LORA"
            mock_tt.FULL = "FULL"
            from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

            config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
            backend = AwsServerlessBackend()
            backend.launch(config, wait=False)

            call_kwargs = mock_trainer_cls.call_args[1]
            assert call_kwargs["custom_reward_function"] is mock_evaluator

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    @patch("foresightrlcore.execution.aws_serverless.AwsServerlessBackend._get_role")
    def test_launch_with_wait_timeout(self, mock_get_role, tmp_yaml):
        mock_get_role.return_value = "arn:aws:iam::123:role/test"

        d = _serverless_config_dict(wait_timeout=1800)

        mock_session = MagicMock()
        mock_dataset = MagicMock()
        mock_mpg = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_training_job = MagicMock()
        mock_training_job.training_job_name = "timeout-test"
        mock_trainer_instance.train.return_value = mock_training_job

        with (
            patch(
                "sagemaker.core.helper.session_helper.Session",
                return_value=mock_session,
            ),
            patch(
                "sagemaker.ai_registry.dataset.DataSet.get",
                return_value=mock_dataset,
            ),
            patch(
                "sagemaker.core.resources.ModelPackageGroup.get",
                return_value=mock_mpg,
            ),
            patch(
                "sagemaker.train.common.TrainingType",
            ) as mock_tt,
            patch(
                "sagemaker.train.rlvr_trainer.RLVRTrainer",
                return_value=mock_trainer_instance,
            ),
        ):
            mock_tt.LORA = "LORA"
            mock_tt.FULL = "FULL"
            from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

            config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
            backend = AwsServerlessBackend()
            backend.launch(config, wait=True)

            mock_trainer_instance.train.assert_called_once_with(
                wait=True, wait_timeout=1800
            )


# ---------------------------------------------------------------------------
# In-transit encryption (ACP IS031.1) tests
# ---------------------------------------------------------------------------


class _FakeTrainingJob:
    """Stand-in for the SDK resource the serverless trainers submit through."""

    calls: list[dict] = []

    @classmethod
    def create(
        cls,
        training_job_name: str,
        enable_inter_container_traffic_encryption: bool | None = None,
        **kwargs,
    ) -> object:
        cls.calls.append(
            {
                "enable_inter_container_traffic_encryption": (
                    enable_inter_container_traffic_encryption
                ),
            }
        )
        job = MagicMock()
        job.training_job_name = training_job_name
        return job


class TestServerlessLaunchEncryption:
    """launch() must set in-transit encryption on the submitted job."""

    @staticmethod
    def _launch(config_dict, tmp_yaml):
        _FakeTrainingJob.calls = []

        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.side_effect = lambda **_: _FakeTrainingJob.create(
            training_job_name="encrypted-job"
        )

        with (
            patch(
                "sagemaker.core.helper.session_helper.Session",
                return_value=MagicMock(),
            ),
            patch(
                "sagemaker.ai_registry.dataset.DataSet.get",
                return_value=MagicMock(),
            ),
            patch(
                "sagemaker.core.resources.ModelPackageGroup.get",
                return_value=MagicMock(),
            ),
            patch("sagemaker.core.resources.TrainingJob", _FakeTrainingJob),
            patch("sagemaker.train.common.TrainingType") as mock_tt,
            patch(
                "sagemaker.train.rlvr_trainer.RLVRTrainer",
                return_value=mock_trainer_instance,
            ),
        ):
            mock_tt.LORA = "LORA"
            mock_tt.FULL = "FULL"
            from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

            config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(config_dict)))
            AwsServerlessBackend().launch(config, wait=False)

        return _FakeTrainingJob.calls

    def test_defaults_to_enabled(self, tmp_yaml, rlvr_config_dict):
        assert (
            rlvr_config_dict["execution"]["serverless"].get(
                "encrypt_inter_container_traffic"
            )
            is None
        )

        calls = self._launch(rlvr_config_dict, tmp_yaml)

        assert len(calls) == 1
        assert calls[0]["enable_inter_container_traffic_encryption"] is True

    def test_can_be_disabled(self, tmp_yaml):
        config_dict = _serverless_config_dict(encrypt_inter_container_traffic=False)

        calls = self._launch(config_dict, tmp_yaml)

        assert len(calls) == 1
        assert calls[0]["enable_inter_container_traffic_encryption"] is None


# ---------------------------------------------------------------------------
# Backend get_status() tests
# ---------------------------------------------------------------------------


class TestAwsServerlessGetStatus:
    """Test get_status() with mocked boto3."""

    @patch("boto3.client")
    def test_get_status_success(self, mock_boto_client):
        mock_sm = MagicMock()
        mock_boto_client.return_value = mock_sm
        mock_sm.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
            "SecondaryStatus": "Completed",
        }

        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        backend = AwsServerlessBackend()
        status = backend.get_status("test-job-123")
        assert status["status"] == "Completed"
        assert status["job_id"] == "test-job-123"

    def test_get_status_handles_error(self):
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        with patch("boto3.client", side_effect=Exception("no creds")):
            backend = AwsServerlessBackend()
            status = backend.get_status("bad-job")
            assert status["status"] == "UNKNOWN"
            assert "no creds" in status["error"]


# ---------------------------------------------------------------------------
# Get-or-create resource tests
# ---------------------------------------------------------------------------


class TestGetOrCreateDataset:
    """Test _resolve_dataset with get-or-create logic."""

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    def test_get_existing_dataset(self):
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        mock_ds = MagicMock()
        with patch("sagemaker.ai_registry.dataset.DataSet.get", return_value=mock_ds):
            result = AwsServerlessBackend._resolve_dataset(
                "my-ds", "s3://bucket/data.jsonl", None, MagicMock()
            )
            assert result is mock_ds

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    def test_create_dataset_on_get_failure(self):
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        mock_created_ds = MagicMock()
        with (
            patch(
                "sagemaker.ai_registry.dataset.DataSet.get",
                side_effect=Exception("not found"),
            ),
            patch(
                "sagemaker.ai_registry.dataset.DataSet.create",
                return_value=mock_created_ds,
            ) as mock_create,
        ):
            session = MagicMock()
            result = AwsServerlessBackend._resolve_dataset(
                "new-ds", "s3://bucket/data.jsonl", None, session
            )
            assert result is mock_created_ds
            mock_create.assert_called_once_with(
                name="new-ds",
                source="s3://bucket/data.jsonl",
                customization_technique=None,
                wait=True,
                sagemaker_session=session,
            )

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    def test_get_failure_without_source_raises(self):
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        with patch(
            "sagemaker.ai_registry.dataset.DataSet.get",
            side_effect=Exception("not found"),
        ):
            with pytest.raises(Exception, match="not found"):
                AwsServerlessBackend._resolve_dataset("bad-ds", None, None, MagicMock())

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    def test_no_name_returns_none(self):
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        result = AwsServerlessBackend._resolve_dataset(
            None, "s3://bucket/data.jsonl", None, MagicMock()
        )
        assert result is None


class TestGetOrCreateRewardPrompt:
    """Test _resolve_reward_prompt with get-or-create logic."""

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    def test_get_existing_evaluator(self):
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        mock_eval = MagicMock()
        sl = MagicMock()
        sl.reward_prompt_name = "existing-prompt"
        sl.reward_prompt_source = None
        sl.reward_prompt_text = None

        with patch(
            "sagemaker.ai_registry.evaluator.Evaluator.get",
            return_value=mock_eval,
        ):
            result = AwsServerlessBackend._resolve_reward_prompt(sl, MagicMock())
            assert result is mock_eval

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    def test_create_from_source_on_get_failure(self):
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        mock_created = MagicMock()
        sl = MagicMock()
        sl.reward_prompt_name = "new-prompt"
        sl.reward_prompt_source = "s3://bucket/prompt.txt"
        sl.reward_prompt_text = None

        session = MagicMock()
        with (
            patch(
                "sagemaker.ai_registry.evaluator.Evaluator.get",
                side_effect=Exception("not found"),
            ),
            patch(
                "sagemaker.ai_registry.evaluator.Evaluator.create",
                return_value=mock_created,
            ) as mock_create,
            patch(
                "sagemaker.ai_registry.air_constants.REWARD_PROMPT",
                "RewardPrompt",
            ),
        ):
            result = AwsServerlessBackend._resolve_reward_prompt(sl, session)
            assert result is mock_created
            mock_create.assert_called_once_with(
                name="new-prompt",
                type="RewardPrompt",
                source="s3://bucket/prompt.txt",
                wait=True,
                sagemaker_session=session,
            )

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    def test_create_from_inline_text_on_get_failure(self):
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        mock_created = MagicMock()
        sl = MagicMock()
        sl.reward_prompt_name = "inline-prompt"
        sl.reward_prompt_source = None
        sl.reward_prompt_text = "Rate the response 1-10."

        session = MagicMock()
        with (
            patch(
                "sagemaker.ai_registry.evaluator.Evaluator.get",
                side_effect=Exception("not found"),
            ),
            patch(
                "sagemaker.ai_registry.evaluator.Evaluator.create",
                return_value=mock_created,
            ) as mock_create,
            patch(
                "sagemaker.ai_registry.air_constants.REWARD_PROMPT",
                "RewardPrompt",
            ),
        ):
            result = AwsServerlessBackend._resolve_reward_prompt(sl, session)
            assert result is mock_created
            call_args = mock_create.call_args
            assert call_args[1]["name"] == "inline-prompt"
            assert call_args[1]["type"] == "RewardPrompt"
            source_path = call_args[1]["source"]
            assert source_path.endswith(".txt")

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    def test_no_name_returns_none(self):
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        sl = MagicMock()
        sl.reward_prompt_name = None
        result = AwsServerlessBackend._resolve_reward_prompt(sl, MagicMock())
        assert result is None

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    def test_get_failure_no_source_raises(self):
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        sl = MagicMock()
        sl.reward_prompt_name = "missing"
        sl.reward_prompt_source = None
        sl.reward_prompt_text = None

        with patch(
            "sagemaker.ai_registry.evaluator.Evaluator.get",
            side_effect=Exception("not found"),
        ):
            with pytest.raises(Exception, match="not found"):
                AwsServerlessBackend._resolve_reward_prompt(sl, MagicMock())


class TestGetOrCreateCustomRewardFunction:
    """Test _resolve_evaluator with get-or-create logic for RLVR."""

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    def test_no_name_returns_none(self):
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        sl = MagicMock()
        sl.custom_reward_function_name = None
        result = AwsServerlessBackend._resolve_evaluator(sl, MagicMock())
        assert result is None

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    def test_get_existing_evaluator(self):
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        mock_eval = MagicMock()
        sl = MagicMock()
        sl.custom_reward_function_name = "existing-evaluator"
        sl.custom_reward_function_source = None
        sl.custom_reward_function_role = None

        with patch(
            "sagemaker.ai_registry.evaluator.Evaluator.get",
            return_value=mock_eval,
        ):
            result = AwsServerlessBackend._resolve_evaluator(sl, MagicMock())
            assert result is mock_eval

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    def test_create_from_source_on_get_failure(self):
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        mock_created = MagicMock()
        sl = MagicMock()
        sl.custom_reward_function_name = "new-evaluator"
        sl.custom_reward_function_source = "serverless/advice/advice_rlvr_reward.py"
        sl.custom_reward_function_role = "arn:aws:iam::123:role/lambda-role"
        sl.custom_reward_function_extra_files = []

        session = MagicMock()
        with (
            patch(
                "sagemaker.ai_registry.evaluator.Evaluator.get",
                side_effect=Exception("not found"),
            ),
            patch(
                "sagemaker.ai_registry.evaluator.Evaluator.create",
                return_value=mock_created,
            ) as mock_create,
            patch(
                "sagemaker.ai_registry.air_constants.REWARD_FUNCTION",
                "RewardFunction",
            ),
        ):
            result = AwsServerlessBackend._resolve_evaluator(sl, session)
            assert result is mock_created
            mock_create.assert_called_once_with(
                name="new-evaluator",
                type="RewardFunction",
                source="serverless/advice/advice_rlvr_reward.py",
                role="arn:aws:iam::123:role/lambda-role",
                wait=True,
                sagemaker_session=session,
            )

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    def test_get_failure_no_source_no_registry_entry_raises(self):
        """When the AI Registry lookup fails AND no ``source`` is set AND
        the name is not in the in-package registry, ``_resolve_evaluator``
        must fail with a clear message pointing to both escape hatches.
        """
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        sl = MagicMock()
        sl.custom_reward_function_name = "this-is-not-registered-anywhere"
        sl.custom_reward_function_source = None
        sl.custom_reward_function_role = None

        with patch(
            "sagemaker.ai_registry.evaluator.Evaluator.get",
            side_effect=Exception("not found"),
        ):
            with pytest.raises(ValueError, match="not found in AI Registry"):
                AwsServerlessBackend._resolve_evaluator(sl, MagicMock())


class TestCustomRewardFunctionValidation:
    """Test cross-field validation for the custom reward function fields."""

    def test_source_without_role_fails(self, tmp_yaml):
        d = _serverless_config_dict(
            method="rlvr",
            custom_reward_function_name="advice-rlvr",
            custom_reward_function_source="serverless/advice/advice_rlvr_reward.py",
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        errors = validate_config(config)
        assert any("custom_reward_function_role is required" in e for e in errors)

    def test_source_with_role_passes(self, tmp_yaml):
        d = _serverless_config_dict(
            method="rlvr",
            custom_reward_function_name="advice-rlvr",
            custom_reward_function_source="serverless/advice/advice_rlvr_reward.py",
            custom_reward_function_role="arn:aws:iam::123:role/lambda-role",
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        errors = validate_config(config)
        assert not any(
            "custom_reward_function" in e for e in errors
        ), f"Unexpected errors: {errors}"

    def test_extra_files_without_source_fails(self, tmp_yaml):
        d = _serverless_config_dict(
            method="rlvr",
            custom_reward_function_name="advice-rlvr",
            custom_reward_function_extra_files=[
                "intuit/foresightrl_intuit/tasks/advice_judge_logic.py",
            ],
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        errors = validate_config(config)
        assert any(
            "custom_reward_function_extra_files" in e
            and "custom_reward_function_source" in e
            for e in errors
        )

    def test_extra_files_nonexistent_path_fails(self, tmp_yaml):
        d = _serverless_config_dict(
            method="rlvr",
            custom_reward_function_name="advice-rlvr",
            custom_reward_function_source="serverless/advice/advice_rlvr_reward.py",
            custom_reward_function_role="arn:aws:iam::123:role/lambda-role",
            custom_reward_function_extra_files=["/nonexistent/path/to/file.py"],
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        errors = validate_config(config)
        assert any(
            "custom_reward_function_extra_files" in e and "does not exist" in e
            for e in errors
        )

    def test_extra_files_with_existing_paths_passes(self, tmp_yaml, tmp_path):
        # Any real file exercises the "path exists" branch. Pointing at a file
        # under intuit/ tied this OSS test to the optional extension directory,
        # so it failed in a checkout where intuit/ had been removed.
        extra_file = tmp_path / "judge_logic.py"
        extra_file.write_text("def score(*_a, **_kw):\n    return 1.0\n")
        d = _serverless_config_dict(
            method="rlvr",
            custom_reward_function_name="advice-rlvr",
            custom_reward_function_source="serverless/advice/advice_rlvr_reward.py",
            custom_reward_function_role="arn:aws:iam::123:role/lambda-role",
            custom_reward_function_extra_files=[str(extra_file)],
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(d)))
        errors = validate_config(config)
        assert not any(
            "custom_reward_function" in e for e in errors
        ), f"Unexpected errors: {errors}"


class TestExtraFilesVendoring:
    """Test the multi-file packaging path in ``_resolve_evaluator``.

    When ``custom_reward_function_extra_files`` is set, the framework
    builds a multi-file zip directly (because the SageMaker SDK only
    supports a single source file) and creates the Lambda via boto3,
    then registers it through ``Evaluator.create``'s Lambda-ARN fast
    path.
    """

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    def test_extras_route_through_lambda_arn_fast_path(self, tmp_path):
        """Extras present -> boto3 lambda.create_function -> Evaluator.create(source=arn)."""
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        main_source = tmp_path / "main.py"
        main_source.write_text("def lambda_handler(event, ctx):\n    return {}\n")
        extra = tmp_path / "extra.py"
        extra.write_text("X = 1\n")

        sl = MagicMock()
        sl.custom_reward_function_name = "advice-rlvr-with-extras"
        sl.custom_reward_function_source = str(main_source)
        sl.custom_reward_function_role = "arn:aws:iam::123:role/lambda-role"
        sl.custom_reward_function_extra_files = [str(extra)]

        mock_lambda_client = MagicMock()
        mock_lambda_client.create_function.return_value = {
            "FunctionArn": (
                "arn:aws:lambda:us-west-2:123:function:SageMaker-evaluator-advice-rlvr-with-extras-20260101_000000"
            ),
        }
        mock_created = MagicMock()
        session = MagicMock()

        with (
            patch(
                "sagemaker.ai_registry.evaluator.Evaluator.get",
                side_effect=Exception("not found"),
            ),
            patch(
                "sagemaker.ai_registry.evaluator.Evaluator.create",
                return_value=mock_created,
            ) as mock_create,
            patch(
                "sagemaker.ai_registry.air_constants.REWARD_FUNCTION",
                "RewardFunction",
            ),
            patch("boto3.client", return_value=mock_lambda_client),
        ):
            result = AwsServerlessBackend._resolve_evaluator(sl, session)

        assert result is mock_created
        # boto3 lambda create_function was called once with our zip
        mock_lambda_client.create_function.assert_called_once()
        kw = mock_lambda_client.create_function.call_args.kwargs
        # Runtime/Timeout/MemorySize must be set explicitly to the
        # package defaults (python3.12 / 900s / 1024MB) — falling
        # through to AWS's global defaults (3s / 128MB / python3.9)
        # silently breaks any Bedrock-judged reward function.
        assert kw["Runtime"] == "python3.12"
        assert kw["Timeout"] == 900
        assert kw["MemorySize"] == 1024
        assert kw["Handler"] == "lambda_function.lambda_handler"
        assert kw["Role"] == "arn:aws:iam::123:role/lambda-role"
        assert kw["FunctionName"].startswith(
            "SageMaker-evaluator-advice-rlvr-with-extras-"
        )

        # Evaluator.create called with source=<lambda ARN>, not the file path
        create_kwargs = mock_create.call_args.kwargs
        assert create_kwargs["source"].startswith("arn:aws:lambda:")
        assert create_kwargs["name"] == "advice-rlvr-with-extras"
        assert create_kwargs["type"] == "RewardFunction"

    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    def test_zip_contains_main_at_lambda_function_py_and_extras_by_basename(
        self, tmp_path
    ):
        """The zip must place the main source at the SDK-hardcoded handler
        path ``lambda_function.py`` and each extra at the zip root using
        its basename, so the main file can ``import <basename>`` it.
        """
        import io
        import zipfile

        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        main_source = tmp_path / "my_main.py"
        main_source.write_text("MAIN = True\n")
        extra_a = tmp_path / "advice_judge_logic.py"
        extra_a.write_text("A = 1\n")
        extra_b = tmp_path / "thinking_utils.py"
        extra_b.write_text("B = 2\n")

        captured_zip: dict = {}

        def capture_create_function(**kwargs):
            captured_zip["Code"] = kwargs["Code"]["ZipFile"]
            return {"FunctionArn": "arn:aws:lambda:us-west-2:123:function:fake"}

        mock_lambda_client = MagicMock()
        mock_lambda_client.create_function.side_effect = capture_create_function

        with patch("boto3.client", return_value=mock_lambda_client):
            arn = AwsServerlessBackend._create_lambda_with_extras(
                name="test-evaluator",
                main_source=str(main_source),
                extra_files=[str(extra_a), str(extra_b)],
                role="arn:aws:iam::123:role/lambda-role",
            )

        assert arn == "arn:aws:lambda:us-west-2:123:function:fake"
        zf = zipfile.ZipFile(io.BytesIO(captured_zip["Code"]))
        names = set(zf.namelist())
        # main was renamed to lambda_function.py at zip root
        assert "lambda_function.py" in names
        # extras placed at zip root with their basenames
        assert "advice_judge_logic.py" in names
        assert "thinking_utils.py" in names
        # original main basename should NOT be in the zip (it was renamed)
        assert "my_main.py" not in names
        # contents preserved
        assert zf.read("lambda_function.py") == b"MAIN = True\n"
        assert zf.read("advice_judge_logic.py") == b"A = 1\n"
        assert zf.read("thinking_utils.py") == b"B = 2\n"

    def test_explicit_runtime_timeout_memory_kwargs_are_forwarded(
        self, tmp_path
    ) -> None:
        """When the caller passes ``runtime`` / ``timeout_s`` /
        ``memory_mb`` to ``_create_lambda_with_extras``, those values
        must reach ``CreateFunction`` verbatim — not the package
        defaults."""
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        main_source = tmp_path / "main.py"
        main_source.write_text("HANDLER = True\n")
        extra = tmp_path / "extra.py"
        extra.write_text("X = 1\n")

        mock_lambda_client = MagicMock()
        mock_lambda_client.create_function.return_value = {
            "FunctionArn": "arn:aws:lambda:us-west-2:123:function:fake",
        }
        with patch("boto3.client", return_value=mock_lambda_client):
            AwsServerlessBackend._create_lambda_with_extras(
                name="explicit-config",
                main_source=str(main_source),
                extra_files=[str(extra)],
                role="arn:aws:iam::123:role/lambda-role",
                runtime="python3.13",
                timeout_s=120,
                memory_mb=2048,
            )
        kw = mock_lambda_client.create_function.call_args.kwargs
        assert kw["Runtime"] == "python3.13"
        assert kw["Timeout"] == 120
        assert kw["MemorySize"] == 2048

    def test_long_evaluator_name_triggers_function_name_truncation(
        self, tmp_path
    ) -> None:
        """A long evaluator name combined with the
        ``SageMaker-evaluator-`` prefix and the ``-YYYYMMDD_HHMMSS``
        timestamp suffix can exceed AWS's 64-char ``FunctionName``
        limit. The backend must truncate transparently so the create
        call succeeds. The bug this test pins down failed live when a
        smoke-source evaluator name was 67 characters and CreateFunction
        rejected it."""
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend
        from foresightrlcore.serverless import registry

        main_source = tmp_path / "main.py"
        main_source.write_text("HANDLER = True\n")
        extra = tmp_path / "extra.py"
        extra.write_text("X = 1\n")

        mock_lambda_client = MagicMock()
        mock_lambda_client.create_function.return_value = {
            "FunctionArn": "arn:aws:lambda:us-west-2:123:function:fake",
        }
        # Long enough that SageMaker-evaluator-<name>-<timestamp> exceeds 64.
        long_name = "example-reward-fn1-very-long-name"
        with patch("boto3.client", return_value=mock_lambda_client):
            AwsServerlessBackend._create_lambda_with_extras(
                name=long_name,
                main_source=str(main_source),
                extra_files=[str(extra)],
                role="arn:aws:iam::123:role/lambda-role",
            )
        kw = mock_lambda_client.create_function.call_args.kwargs
        assert len(kw["FunctionName"]) <= registry.MAX_LAMBDA_FUNCTION_NAME_LEN
        assert kw["FunctionName"].startswith("SageMaker-evaluator-")

    def test_source_with_known_name_inherits_registry_runtime_config(
        self, tmp_path
    ) -> None:
        """When ``custom_reward_function_source`` is set AND the name
        matches a registered ``LambdaSpec``, the source code wins
        (--source beats --registry) but the runtime config (runtime,
        timeout, memory, env vars) must be inherited from the spec.
        Otherwise a user "shipping the same Lambda with one file moved"
        via the source path would silently drop production's
        Bedrock-sized timeouts and revert to the package defaults.

        Set non-default LambdaSpec values to prove the values came
        from the spec rather than from the package defaults — both
        would otherwise be 900s / 1024MB and indistinguishable.
        """
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend
        from foresightrlcore.serverless import registry

        main_source = tmp_path / "main.py"
        main_source.write_text("HANDLER = True\n")
        extra = tmp_path / "extra.py"
        extra.write_text("X = 1\n")

        original_lambdas = dict(registry.LAMBDAS)
        try:
            registry.LAMBDAS.clear()
            registry.register_lambda(
                name="known-name",
                main_module="foresightrl_intuit.tasks.advice.lambda_reward",
                env_defaults={"JUDGE_MODEL_ID": "spec-judge"},
                runtime="python3.13",
                timeout_s=60,
                memory_mb=2048,
            )

            sl = MagicMock()
            sl.custom_reward_function_name = "known-name"
            sl.custom_reward_function_source = str(main_source)
            sl.custom_reward_function_extra_files = [str(extra)]
            sl.custom_reward_function_role = "arn:aws:iam::123:role/lambda-role"

            mock_lambda_client = MagicMock()
            mock_lambda_client.create_function.return_value = {
                "FunctionArn": "arn:aws:lambda:us-west-2:123:function:fake",
            }
            with (
                patch(
                    "sagemaker.ai_registry.evaluator.Evaluator.get",
                    side_effect=Exception("not found"),
                ),
                patch(
                    "sagemaker.ai_registry.evaluator.Evaluator.create",
                    return_value=MagicMock(),
                ),
                patch(
                    "sagemaker.ai_registry.air_constants.REWARD_FUNCTION",
                    "RewardFunction",
                ),
                patch("boto3.client", return_value=mock_lambda_client),
            ):
                AwsServerlessBackend._resolve_evaluator(sl, MagicMock())
        finally:
            registry.LAMBDAS.clear()
            registry.LAMBDAS.update(original_lambdas)

        kw = mock_lambda_client.create_function.call_args.kwargs
        assert kw["Runtime"] == "python3.13"
        assert kw["Timeout"] == 60
        assert kw["MemorySize"] == 2048
        env = kw.get("Environment", {}).get("Variables", {})
        assert env.get("JUDGE_MODEL_ID") == "spec-judge"

    def test_source_with_fresh_name_uses_package_defaults(self, tmp_path) -> None:
        """When ``custom_reward_function_source`` is set and the name
        is NOT in the registry (the smoke-source scenario), the
        backend must use the package-wide Bedrock-friendly defaults —
        not AWS's brittle global defaults (3s / 128MB / python3.9)."""
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend

        main_source = tmp_path / "main.py"
        main_source.write_text("HANDLER = True\n")
        extra = tmp_path / "extra.py"
        extra.write_text("X = 1\n")

        sl = MagicMock()
        sl.custom_reward_function_name = "this-name-is-not-registered"
        sl.custom_reward_function_source = str(main_source)
        sl.custom_reward_function_extra_files = [str(extra)]
        sl.custom_reward_function_role = "arn:aws:iam::123:role/lambda-role"

        mock_lambda_client = MagicMock()
        mock_lambda_client.create_function.return_value = {
            "FunctionArn": "arn:aws:lambda:us-west-2:123:function:fake",
        }
        with (
            patch(
                "sagemaker.ai_registry.evaluator.Evaluator.get",
                side_effect=Exception("not found"),
            ),
            patch(
                "sagemaker.ai_registry.evaluator.Evaluator.create",
                return_value=MagicMock(),
            ),
            patch(
                "sagemaker.ai_registry.air_constants.REWARD_FUNCTION",
                "RewardFunction",
            ),
            patch("boto3.client", return_value=mock_lambda_client),
        ):
            AwsServerlessBackend._resolve_evaluator(sl, MagicMock())
        kw = mock_lambda_client.create_function.call_args.kwargs
        assert kw["Runtime"] == "python3.12"
        assert kw["Timeout"] == 900
        assert kw["MemorySize"] == 1024


class TestRegistryFallback:
    """When ``source`` is not set, the backend falls back to the
    in-package registry to resolve ``custom_reward_function_name``.
    """

    @pytest.mark.requires_intuit
    @patch.dict("os.environ", {"AWS_DEFAULT_REGION": "us-west-2"})
    def test_registry_lookup_resolves_advice_lambda(self):
        """The advice lambda is registered at import time, so the
        backend can resolve it from name alone."""
        from foresightrlcore.execution.aws_serverless import AwsServerlessBackend
        from foresightrlcore.serverless import registry
        from foresightrlcore.tasks import load_task_module

        registry.discover_tasks()
        load_task_module("advice")
        reward = [
            name
            for name, spec in registry.LAMBDAS.items()
            if spec.main_module.endswith(".lambda_reward")
        ]
        assert reward, "the extension registered no reward lambda"
        lambda_name = reward[0]

        sl = MagicMock()
        sl.custom_reward_function_name = lambda_name
        sl.custom_reward_function_source = None
        sl.custom_reward_function_role = "arn:aws:iam::123:role/lambda-role"
        sl.custom_reward_function_extra_files = []

        mock_lambda_client = MagicMock()
        mock_lambda_client.create_function.return_value = {
            "FunctionArn": "arn:aws:lambda:us-west-2:123:function:fake",
        }
        mock_created = MagicMock()

        with (
            patch(
                "sagemaker.ai_registry.evaluator.Evaluator.get",
                side_effect=Exception("not found"),
            ),
            patch(
                "sagemaker.ai_registry.evaluator.Evaluator.create",
                return_value=mock_created,
            ),
            patch(
                "sagemaker.ai_registry.air_constants.REWARD_FUNCTION",
                "RewardFunction",
            ),
            patch("boto3.client", return_value=mock_lambda_client),
        ):
            result = AwsServerlessBackend._resolve_evaluator(sl, MagicMock())

        assert result is mock_created
        # The zipped function must have been built from registered files,
        # env_defaults forwarded, and runtime/timeout/memory match what
        # the advice task's LambdaSpec declares (sized for Bedrock).
        mock_lambda_client.create_function.assert_called_once()
        kw = mock_lambda_client.create_function.call_args.kwargs
        env = kw.get("Environment", {}).get("Variables", {})
        assert env.get("JUDGE_MODEL_ID")
        assert env.get("JUDGE_REGION") == "us-west-2"
        advice_spec = registry.LAMBDAS[lambda_name]
        assert kw["Runtime"] == advice_spec.runtime
        assert kw["Timeout"] == advice_spec.timeout_s
        assert kw["MemorySize"] == advice_spec.memory_mb
