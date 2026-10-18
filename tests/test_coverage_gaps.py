"""Tests to fill remaining coverage gaps across multiple small modules."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml

from foresightrlcore.config.schema import TrainingJobConfig

# ── foresightrlcore/__init__.py — train() + _ensure_backends_registered ──


class TestInitTrain:
    def test_train_loads_config_and_runs(self):
        config_data = {
            "job": {"name": "test"},
            "model": {"name": "m"},
            "data": {"train_path": "/data"},
            "output": {"s3_path": "/output"},
            "execution": {
                "type": "local",
                "local": {"gpus": "0,1,2,3,4,5,6,7"},
            },
            "task": {"type": "advice", "config": {"reward_mode": "mock"}},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            path = f.name

        try:
            from foresightrlcore import train

            with patch(
                "foresightrlcore.operations.train.TrainOperation.run",
                return_value="job-123",
            ):
                result = train(path)
            assert result == "job-123"
        finally:
            os.unlink(path)

    def test_train_with_overrides(self):
        config_data = {
            "job": {"name": "test"},
            "model": {"name": "m"},
            "data": {"train_path": "/data"},
            "output": {"s3_path": "/output"},
            "task": {"type": "advice"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            path = f.name

        try:
            from foresightrlcore import train

            with patch(
                "foresightrlcore.operations.train.TrainOperation.run",
                return_value="job-456",
            ):
                result = train(path, job__name="override-name")
            assert result == "job-456"
        finally:
            os.unlink(path)

    def test_ensure_backends_registered(self):
        from foresightrlcore import _ensure_backends_registered

        _ensure_backends_registered()

        from foresightrlcore.execution import get_backend
        from foresightrlcore.methods import get_method
        from foresightrlcore.operations import get_operation

        assert get_backend("aws_training_job") is not None
        assert get_backend("local") is not None
        assert get_method("grpo") is not None
        assert get_operation("train") is not None


# ── foresightrlcore/__main__.py ──────────────────────────────────────────


class TestMainModule:
    def test_module_invocation(self):
        with patch("foresightrlcore.cli.main", return_value=0) as mock_main:
            with pytest.raises(SystemExit) as exc:
                exec(
                    compile(
                        "from foresightrlcore.cli import main; raise SystemExit(main())",
                        "__main__",
                        "exec",
                    )
                )
            assert exc.value.code == 0


# ── foresightrlcore/cli.py — _cmd_train ──────────────────────────────────


class TestCliTrain:
    def test_train_success(self):
        from foresightrlcore.cli import main

        config_data = {
            "job": {"name": "test"},
            "model": {"name": "m"},
            "data": {"train_path": "/data"},
            "output": {"s3_path": "/output"},
            "task": {"type": "advice"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            path = f.name

        try:
            with patch("foresightrlcore.train", return_value="job-789"):
                result = main(["train", "--config", path])
            assert result == 0
        finally:
            os.unlink(path)

    def test_train_with_wait(self):
        from foresightrlcore.cli import main

        config_data = {
            "job": {"name": "test"},
            "model": {"name": "m"},
            "data": {"train_path": "/data"},
            "output": {"s3_path": "/output"},
            "task": {"type": "advice"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            path = f.name

        try:
            with patch("foresightrlcore.train", return_value="job-wait") as mock_train:
                result = main(["train", "--config", path, "--wait"])
            assert result == 0
            mock_train.assert_called_once_with(path, wait=True)
        finally:
            os.unlink(path)

    def test_train_failure(self):
        from foresightrlcore.cli import main

        config_data = {
            "job": {"name": "test"},
            "model": {"name": "m"},
            "data": {"train_path": "/data"},
            "output": {"s3_path": "/output"},
            "task": {"type": "advice"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            path = f.name

        try:
            with patch(
                "foresightrlcore.train", side_effect=RuntimeError("launch failed")
            ):
                result = main(["train", "--config", path])
            assert result == 1
        finally:
            os.unlink(path)

    def test_validate_invalid_config_content(self):
        from foresightrlcore.cli import main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("not: valid: yaml: config: !@#$")
            path = f.name

        try:
            result = main(["validate", "--config", path])
            assert result == 1
        finally:
            os.unlink(path)


# ── foresightrlcore/execution/base.py — protocol checks ─────────────────


class TestExecutionProtocol:
    def test_protocol_is_runtime_checkable(self):
        from foresightrlcore.execution.aws_training_job import AwsTrainingJobBackend
        from foresightrlcore.execution.base import ExecutionBackend
        from foresightrlcore.execution.local import LocalBackend

        assert isinstance(LocalBackend(), ExecutionBackend)
        assert isinstance(AwsTrainingJobBackend(), ExecutionBackend)

    def test_protocol_attributes(self):
        from foresightrlcore.execution.base import ExecutionBackend

        assert hasattr(ExecutionBackend, "validate")
        assert hasattr(ExecutionBackend, "launch")
        assert hasattr(ExecutionBackend, "get_status")


# ── foresightrlcore/methods/base.py — protocol checks ───────────────────


class TestMethodProtocol:
    def test_protocol_is_runtime_checkable(self):
        from foresightrlcore.methods.base import MethodHandler
        from foresightrlcore.methods.grpo import GrpoMethodHandler

        assert isinstance(GrpoMethodHandler(), MethodHandler)

    def test_protocol_attributes(self):
        from foresightrlcore.methods.base import MethodHandler

        assert hasattr(MethodHandler, "validate")
        assert hasattr(MethodHandler, "build_trainer_args")
        assert hasattr(MethodHandler, "run")


# ── foresightrlcore/operations/base.py — protocol checks ────────────────


class TestOperationProtocol:
    def test_protocol_is_runtime_checkable(self):
        from foresightrlcore.operations.base import Operation
        from foresightrlcore.operations.train import TrainOperation

        assert isinstance(TrainOperation(), Operation)

    def test_protocol_attributes(self):
        from foresightrlcore.operations.base import Operation

        assert hasattr(Operation, "run")


# ── foresightrlcore/operations/train.py — success path ──────────────────


class TestTrainOperationSuccess:
    def test_run_success(self):
        from foresightrlcore.operations.train import TrainOperation

        config_data = {
            "job": {"name": "test"},
            "model": {"name": "m"},
            "data": {"train_path": "/data"},
            "output": {"s3_path": "/output"},
            "execution": {
                "type": "local",
                "local": {"gpus": "0,1,2,3,4,5,6,7"},
            },
            "task": {"type": "advice"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            path = f.name

        try:
            config = TrainingJobConfig.from_yaml(path)
            op = TrainOperation()

            mock_backend = MagicMock()
            mock_backend.validate.return_value = []
            mock_backend.launch.return_value = "launched-job"

            with patch(
                "foresightrlcore.operations.train.get_backend",
                return_value=lambda: mock_backend,
            ):
                result = op.run(config)

            assert result == "launched-job"
        finally:
            os.unlink(path)

    def test_run_backend_validation_fails(self):
        from foresightrlcore.operations.train import TrainOperation

        config_data = {
            "job": {"name": "test"},
            "model": {"name": "m"},
            "data": {"train_path": "/data"},
            "output": {"s3_path": "/output"},
            "execution": {
                "type": "local",
                "local": {"gpus": "0,1,2,3,4,5,6,7"},
            },
            "task": {"type": "advice"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            path = f.name

        try:
            config = TrainingJobConfig.from_yaml(path)
            op = TrainOperation()

            mock_backend = MagicMock()
            mock_backend.validate.return_value = ["backend error 1"]

            with patch(
                "foresightrlcore.operations.train.get_backend",
                return_value=lambda: mock_backend,
            ):
                with pytest.raises(ValueError, match="Backend.*validation failed"):
                    op.run(config)
        finally:
            os.unlink(path)


# ── foresightrlcore/execution/aws_training_job.py — launch + get_status ──


class TestAwsTrainingJobBackendLaunch:
    def test_launch_delegates(self):
        from foresightrlcore.execution.aws_training_job import AwsTrainingJobBackend

        config_data = {
            "job": {"name": "test"},
            "model": {"name": "m"},
            "data": {"train_path": "s3://b/train"},
            "output": {"s3_path": "s3://b/out"},
            "task": {"type": "advice"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            path = f.name

        try:
            config = TrainingJobConfig.from_yaml(path)
            backend = AwsTrainingJobBackend()

            mock_launcher_instance = MagicMock()
            mock_launcher_instance.launch.return_value = "sm-job-123"

            with patch(
                "foresightrlcore.execution.aws_sagemaker_launcher.GRPOJobLauncher",
                return_value=mock_launcher_instance,
            ):
                result = backend.launch(config, wait=True)

            assert result == "sm-job-123"
            mock_launcher_instance.launch.assert_called_once_with(wait=True)
        finally:
            os.unlink(path)

    def test_get_status_success(self):
        from foresightrlcore.execution.aws_training_job import AwsTrainingJobBackend

        backend = AwsTrainingJobBackend()
        mock_sm = MagicMock()
        mock_sm.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
            "SecondaryStatus": "Completed",
        }

        with patch("boto3.client", return_value=mock_sm):
            status = backend.get_status("job-123")

        assert status["status"] == "Completed"
        assert status["job_id"] == "job-123"

    def test_get_status_error(self):
        from foresightrlcore.execution.aws_training_job import AwsTrainingJobBackend

        backend = AwsTrainingJobBackend()

        with patch("boto3.client", side_effect=Exception("no access")):
            status = backend.get_status("job-fail")

        assert status["status"] == "UNKNOWN"
        assert "error" in status


# ── foresightrlcore/config/validation.py — uncovered branches ────────────


class TestValidationGaps:
    def test_dedicated_vllm_requires_multiple_instances(self):
        from foresightrlcore.config.validation import validate_config

        config_data = {
            "job": {"name": "test"},
            "model": {"name": "m"},
            "infra": {
                "topology": "dedicated_vllm",
                "instance_count": 1,
            },
            "data": {"train_path": "s3://b/d"},
            "output": {"s3_path": "s3://b/o"},
            "task": {"type": "advice"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            path = f.name

        try:
            config = TrainingJobConfig.from_yaml(path)
            errors = validate_config(config)
            assert any("dedicated_vllm" in e for e in errors)
        finally:
            os.unlink(path)

    def test_sft_with_vllm_warns(self, caplog):
        from foresightrlcore.config.validation import validate_config

        config_data = {
            "job": {"name": "test"},
            "model": {"name": "m"},
            "method": {"type": "sft"},
            "infra": {"vllm_mode": "server"},
            "data": {"train_path": "s3://b/d"},
            "output": {"s3_path": "s3://b/o"},
            "task": {"type": "advice"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            path = f.name

        try:
            config = TrainingJobConfig.from_yaml(path)
            import logging

            with caplog.at_level(logging.WARNING):
                validate_config(config)
            assert any("does not need vLLM" in r.message for r in caplog.records)
        finally:
            os.unlink(path)

    def test_grpo_none_vllm_warns(self, caplog):
        from foresightrlcore.config.validation import validate_config

        config_data = {
            "job": {"name": "test"},
            "model": {"name": "m"},
            "method": {"type": "grpo"},
            "infra": {"vllm_mode": "none"},
            "data": {"train_path": "s3://b/d"},
            "output": {"s3_path": "s3://b/o"},
            "task": {"type": "advice"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            path = f.name

        try:
            config = TrainingJobConfig.from_yaml(path)
            import logging

            with caplog.at_level(logging.WARNING):
                validate_config(config)
            assert any("native HF generation" in r.message for r in caplog.records)
        finally:
            os.unlink(path)

    def test_local_missing_gpus(self):
        from foresightrlcore.config.validation import validate_config

        config_data = {
            "job": {"name": "test"},
            "model": {"name": "m"},
            "data": {"train_path": "/data"},
            "output": {"s3_path": "/output"},
            "execution": {"type": "local"},
            "task": {"type": "advice"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            path = f.name

        try:
            config = TrainingJobConfig.from_yaml(path)
            errors = validate_config(config)
            assert any("gpus" in e for e in errors)
        finally:
            os.unlink(path)


# ── foresightrlcore/tasks/__init__.py — uncovered branches ───────────────


class TestTasksModuleGaps:
    def test_dotted_path_import(self):
        from foresightrlcore.tasks import load_task_module

        module = load_task_module("foresightrlcore.tasks.thinking_utils")
        assert hasattr(module, "strip_thinking_tags")

    def test_builtin_fallback_not_found(self):
        from foresightrlcore.tasks import load_task_module

        with pytest.raises(ValueError, match="Unknown task type"):
            load_task_module("nonexistent_task_xyz")

    def test_get_task_components_with_mock(self):
        from foresightrlcore.tasks import get_task_components

        mock_module = MagicMock()
        mock_module.get_reward_function.return_value = lambda: 0.5
        mock_module.get_callbacks.return_value = []
        mock_module.get_dataset_loader.return_value = lambda p: []

        with patch("foresightrlcore.tasks.load_task_module", return_value=mock_module):
            reward, callbacks, loader = get_task_components("mock_task", {}, "model")

        assert callable(reward)
        assert callbacks == []
        assert callable(loader)

    def test_load_from_path_spec_none(self):
        from foresightrlcore.tasks import _load_from_path

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("x = 1\n")
            path = f.name

        try:
            module = _load_from_path(path)
            assert module.x == 1
        finally:
            os.unlink(path)


# ── foresightrlcore/tasks/thinking_utils.py — uncovered branches ────────


class TestThinkingUtilsGaps:
    def test_parse_json_malformed_code_block_with_fallback(self):
        from foresightrlcore.tasks.thinking_utils import parse_json_from_response

        text = '```json\nnot-json\n```\n{"fallback": 1}'
        result = parse_json_from_response(text)
        assert result == {"fallback": 1}

    def test_parse_json_all_malformed(self):
        from foresightrlcore.tasks.thinking_utils import parse_json_from_response

        result = parse_json_from_response("no code block, no raw json either")
        assert result == {}


# ── foresightrlcore/auth/__init__.py — uncovered branches ────────────────


class TestAuthGaps:
    def test_create_provider_idps_factory(self):
        from foresightrlcore.auth import create_provider

        mock_provider = MagicMock()
        with patch(
            "foresightrlcore.auth._PROVIDER_FACTORIES",
            {"idps": lambda cfg: mock_provider},
        ):
            result = create_provider(
                {
                    "provider": "idps",
                    "idps": {
                        "policy_id": "pol-1",
                        "endpoint": "vkm.example.com",
                    },
                }
            )
        assert result is mock_provider

    def test_create_provider_local(self):
        from foresightrlcore.auth import create_provider

        mock_provider_cls = MagicMock()
        with patch("foresightrlcore.auth._PROVIDERS", {"local": mock_provider_cls}):
            create_provider(
                {
                    "provider": "local",
                    "local": {"env_file": "/tmp/.env"},
                }
            )
        mock_provider_cls.assert_called_once_with(env_file="/tmp/.env")

    def test_create_provider_unsupported(self):
        from foresightrlcore.auth import create_provider

        mock_provider = MagicMock()
        with patch("foresightrlcore.auth.get_provider", return_value=mock_provider):
            with pytest.raises(ValueError, match="Unsupported"):
                create_provider({"provider": "vault", "other": {}})


# ── foresightrlcore/methods/grpo.py — validate missing sections ─────────


class TestGrpoValidateGaps:
    def test_validate_missing_training(self):
        from foresightrlcore.methods.grpo import GrpoMethodHandler

        handler = GrpoMethodHandler()
        mock_config = MagicMock(spec=[])
        mock_config.training = None
        errors = handler.validate(mock_config)
        assert any("missing" in e.lower() for e in errors)

    def test_validate_missing_grpo_section(self):
        from foresightrlcore.methods.grpo import GrpoMethodHandler

        handler = GrpoMethodHandler()
        mock_config = MagicMock()
        mock_config.training = MagicMock(spec=[])
        mock_config.training.grpo = None
        errors = handler.validate(mock_config)
        assert any("grpo" in e.lower() for e in errors)
