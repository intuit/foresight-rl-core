"""Tests for foresightrlcore.runtime.scripts.train — unified training driver."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml


class TestLoadConfig:
    def test_loads_absolute_path(self):
        from foresightrlcore.runtime.scripts.train import load_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"job": {"name": "test"}}, f)
            path = f.name
        try:
            config = load_config(path)
            assert config["job"]["name"] == "test"
        finally:
            os.unlink(path)

    def test_missing_config_raises(self):
        from foresightrlcore.runtime.scripts.train import load_config

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/config.yaml")

    def test_loads_from_sm_config_dir(self):
        from foresightrlcore.runtime.scripts.train import load_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = os.path.join(tmpdir, "args.yaml")
            with open(config_file, "w") as f:
                yaml.dump({"model": {"name": "test-model"}}, f)

            with patch("foresightrlcore.runtime.scripts.train.SM_CONFIG_DIR", tmpdir):
                config = load_config("args.yaml")
            assert config["model"]["name"] == "test-model"


class TestResolveMethodType:
    def test_explicit_grpo(self):
        from foresightrlcore.runtime.scripts.train import _resolve_method_type

        assert _resolve_method_type({"method": {"type": "grpo"}}) == "grpo"

    def test_explicit_sft(self):
        from foresightrlcore.runtime.scripts.train import _resolve_method_type

        assert _resolve_method_type({"method": {"type": "sft"}}) == "sft"

    def test_default_grpo(self):
        from foresightrlcore.runtime.scripts.train import _resolve_method_type

        assert _resolve_method_type({}) == "grpo"

    def test_missing_method_key(self):
        from foresightrlcore.runtime.scripts.train import _resolve_method_type

        assert _resolve_method_type({"other": "stuff"}) == "grpo"


class TestParseArgs:
    def test_parse_args(self):
        from foresightrlcore.runtime.scripts.train import parse_args

        with patch("sys.argv", ["train.py", "--config", "/path/to/config.yaml"]):
            args = parse_args()
        assert args.config == "/path/to/config.yaml"
        assert args.vllm_host is None

    def test_parse_args_with_vllm_host(self):
        from foresightrlcore.runtime.scripts.train import parse_args

        with patch(
            "sys.argv",
            ["train.py", "--config", "cfg.yaml", "--vllm-host", "10.0.0.1"],
        ):
            args = parse_args()
        assert args.vllm_host == "10.0.0.1"


class TestMain:
    def test_main_grpo(self):
        from foresightrlcore.runtime.scripts.train import main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "model": {"name": "test-model"},
                    "training": {"grpo": {}},
                    "task": {"type": "advice"},
                },
                f,
            )
            path = f.name

        try:
            with (
                patch("sys.argv", ["train.py", "--config", path]),
                patch(
                    "foresightrlcore.runtime.scripts.train._run_with_handler"
                ) as mock_run,
            ):
                main()
            mock_run.assert_called_once()
        finally:
            os.unlink(path)

    def test_main_sft_unregistered(self):
        """SFT handler is not registered — dispatches to _run_with_handler which
        will fail inside @record. We verify the dispatch reaches _run_with_handler."""
        from foresightrlcore.runtime.scripts.train import main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"method": {"type": "sft"}}, f)
            path = f.name

        try:
            with (
                patch("sys.argv", ["train.py", "--config", path]),
                patch(
                    "foresightrlcore.runtime.scripts.train._run_with_handler"
                ) as mock_run,
            ):
                main()
            mock_run.assert_called_once()
            raw_config = mock_run.call_args[0][0]
            assert raw_config["method"]["type"] == "sft"
        finally:
            os.unlink(path)

    def test_main_dpo_unregistered(self):
        """DPO handler is not registered — dispatches to _run_with_handler."""
        from foresightrlcore.runtime.scripts.train import main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"method": {"type": "dpo"}}, f)
            path = f.name

        try:
            with (
                patch("sys.argv", ["train.py", "--config", path]),
                patch(
                    "foresightrlcore.runtime.scripts.train._run_with_handler"
                ) as mock_run,
            ):
                main()
            mock_run.assert_called_once()
            raw_config = mock_run.call_args[0][0]
            assert raw_config["method"]["type"] == "dpo"
        finally:
            os.unlink(path)

    def test_main_unknown_method(self):
        """Unknown method type dispatches to _run_with_handler (KeyError at runtime)."""
        from foresightrlcore.runtime.scripts.train import main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"method": {"type": "unknown"}}, f)
            path = f.name

        try:
            with (
                patch("sys.argv", ["train.py", "--config", path]),
                patch(
                    "foresightrlcore.runtime.scripts.train._run_with_handler"
                ) as mock_run,
            ):
                main()
            mock_run.assert_called_once()
        finally:
            os.unlink(path)

    def test_main_with_secrets(self):
        from foresightrlcore.runtime.scripts.train import main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "method": {"type": "grpo"},
                    "secrets": {
                        "provider": "idps",
                        "entries": [{"env_var": "X", "secret_path": "/p"}],
                    },
                },
                f,
            )
            path = f.name

        try:
            with (
                patch("sys.argv", ["train.py", "--config", path]),
                patch(
                    "foresightrlcore.runtime.scripts.train._resolve_secrets_if_configured"
                ) as mock_rs,
                patch("foresightrlcore.runtime.scripts.train._run_with_handler"),
            ):
                main()
            mock_rs.assert_called_once_with(path)
        finally:
            os.unlink(path)


class TestLoadEnvFile:
    def test_loads_env_vars(self):
        from foresightrlcore.runtime.scripts.train import _load_env_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("export SECRET_KEY='my-secret-value'\n")
            f.write("OTHER_VAR=other_value\n")
            f.write("# comment line\n")
            f.write("\n")
            path = f.name

        try:
            _load_env_file(path)
            assert os.environ.get("SECRET_KEY") == "my-secret-value"
            assert os.environ.get("OTHER_VAR") == "other_value"
        finally:
            os.unlink(path)
            os.environ.pop("SECRET_KEY", None)
            os.environ.pop("OTHER_VAR", None)

    def test_empty_lines_skipped(self):
        from foresightrlcore.runtime.scripts.train import _load_env_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("\n\n# comment\n\n")
            path = f.name
        try:
            _load_env_file(path)
        finally:
            os.unlink(path)


class TestResolveSecretsIfConfigured:
    def test_skips_if_env_file_exists(self):
        from foresightrlcore.runtime.scripts.train import _resolve_secrets_if_configured

        env_file = "/tmp/resolved_secrets.env"
        with open(env_file, "w") as f:
            f.write("export TEST_SECRET='val'\n")

        try:
            with patch(
                "foresightrlcore.runtime.scripts.train._load_env_file"
            ) as mock_load:
                _resolve_secrets_if_configured("/some/config.yaml")
            mock_load.assert_called_once_with(env_file)
        finally:
            os.unlink(env_file)
            os.environ.pop("TEST_SECRET", None)

    def test_resolves_without_installing_a_secrets_client(self):
        from foresightrlcore.runtime.scripts.train import _resolve_secrets_if_configured

        env_file = "/tmp/resolved_secrets.env"
        if os.path.exists(env_file):
            os.unlink(env_file)

        with (
            patch("subprocess.check_call") as mock_call,
            patch("foresightrlcore.runtime.scripts.train._load_env_file"),
        ):
            _resolve_secrets_if_configured("/some/config.yaml")

        assert mock_call.call_count == 1
        resolve_call = mock_call.call_args_list[0]
        assert "resolve_secrets.py" in str(resolve_call)
        assert "pip" not in str(resolve_call)


class TestApplySagemakerOverridesModelChannel:
    """When SageMaker mounts a 'model' input channel, _apply_sagemaker_overrides
    swaps the YAML model.name (an s3:// URI) for the local mount path so
    transformers loads from disk instead of HF Hub."""

    def _make_args(self):
        ns = MagicMock()
        ns.vllm_host = None
        return ns

    def _make_min_config(self, model_name: str):
        from foresightrlcore.config.schema import TrainingJobConfig

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "job": {"name": "t"},
                    "model": {"name": model_name},
                    "infra": {
                        "instance_type": "ml.p4d.24xlarge",
                        "instance_count": 1,
                        "vllm_mode": "server",
                    },
                    "data": {"train_path": "s3://bucket/train"},
                    "output": {"s3_path": "s3://bucket/out"},
                    "execution": {"type": "aws_training_job"},
                    "task": {"type": "advice", "config": {"reward_mode": "mock"}},
                },
                f,
            )
            path = f.name
        try:
            return TrainingJobConfig.from_yaml(path)
        finally:
            os.unlink(path)

    def test_overrides_model_name_when_channel_mounted(self):
        from foresightrlcore.runtime.scripts import train as train_mod

        config = self._make_min_config("s3://bucket/models/Qwen3.6-27B/")
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                f.write("{}")
            with (
                patch.object(train_mod, "SM_INPUT_MODEL_DIR", tmpdir),
                patch.object(train_mod, "SM_TRAIN_DIR", "/tmp/train"),
                patch.object(train_mod, "SM_EVAL_DIR", "/tmp/eval"),
                patch.object(train_mod, "SM_MODEL_DIR", "/tmp/model"),
                patch.object(train_mod, "SM_CHECKPOINT_DIR", "/tmp/ckpt"),
            ):
                train_mod._apply_sagemaker_overrides(config, self._make_args())
            assert config.model.name == tmpdir

    def test_keeps_model_name_when_no_channel_mounted(self):
        from foresightrlcore.runtime.scripts import train as train_mod

        config = self._make_min_config("Qwen/Qwen3.5-27B")
        with tempfile.TemporaryDirectory() as empty_dir:
            with (
                patch.object(train_mod, "SM_INPUT_MODEL_DIR", empty_dir),
                patch.object(train_mod, "SM_TRAIN_DIR", "/tmp/train"),
                patch.object(train_mod, "SM_EVAL_DIR", "/tmp/eval"),
                patch.object(train_mod, "SM_MODEL_DIR", "/tmp/model"),
                patch.object(train_mod, "SM_CHECKPOINT_DIR", "/tmp/ckpt"),
            ):
                train_mod._apply_sagemaker_overrides(config, self._make_args())
            assert config.model.name == "Qwen/Qwen3.5-27B"


class TestPatchVllmEnforceEager:
    def test_patch_success(self):
        from foresightrlcore.methods.base import _patch_vllm_enforce_eager

        mock_vllm_gen = MagicMock()
        mock_original_llm = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "trl.generation.vllm_generation": mock_vllm_gen,
                "vllm": MagicMock(LLM=mock_original_llm),
            },
        ):
            _patch_vllm_enforce_eager(enforce_eager=True)

    def test_patch_failure_handled(self):
        from foresightrlcore.methods.base import _patch_vllm_enforce_eager

        with patch.dict("sys.modules", {"trl.generation.vllm_generation": None}):
            _patch_vllm_enforce_eager(enforce_eager=True)
