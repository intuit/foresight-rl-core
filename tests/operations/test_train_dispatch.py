"""Tests for the unified train.py dispatch and SageMaker override logic."""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

from foresightrlcore.config.schema import TrainingJobConfig


class TestApplySagemakerOverrides:
    """Tests for _apply_sagemaker_overrides path patching."""

    def _make_config(self, tmp_yaml) -> TrainingJobConfig:
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "data": {"train_path": "s3://bucket/train"},
                "output": {"s3_path": "s3://bucket/output"},
                "task": {"type": "advice"},
                "training": {"deepspeed": "ds_zero2_config.json"},
            }
        )
        return TrainingJobConfig.from_yaml(tmp_yaml(content))

    def test_overrides_train_path(self, tmp_yaml, tmp_path):
        from foresightrlcore.runtime.scripts.train import _apply_sagemaker_overrides

        config = self._make_config(tmp_yaml)
        args = SimpleNamespace(vllm_host=None)
        model_dir = str(tmp_path / "model")
        checkpoint_dir = str(tmp_path / "checkpoints")

        with (
            patch(
                "foresightrlcore.runtime.scripts.train.SM_TRAIN_DIR",
                "/opt/ml/data/train",
            ),
            patch("foresightrlcore.runtime.scripts.train.SM_MODEL_DIR", model_dir),
            patch(
                "foresightrlcore.runtime.scripts.train.SM_CHECKPOINT_DIR",
                checkpoint_dir,
            ),
            patch(
                "foresightrlcore.runtime.scripts.train.SM_EVAL_DIR",
                "/nonexistent/eval",
            ),
        ):
            _apply_sagemaker_overrides(config, args)

        assert config.data.train_path == "/opt/ml/data/train"

    def test_overrides_eval_path_when_dir_exists(self, tmp_yaml, tmp_path):
        from foresightrlcore.runtime.scripts.train import _apply_sagemaker_overrides

        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        (eval_dir / "data.jsonl").write_text("dummy")
        model_dir = str(tmp_path / "model")
        checkpoint_dir = str(tmp_path / "checkpoints")

        config = self._make_config(tmp_yaml)
        args = SimpleNamespace(vllm_host=None)

        with (
            patch("foresightrlcore.runtime.scripts.train.SM_EVAL_DIR", str(eval_dir)),
            patch("foresightrlcore.runtime.scripts.train.SM_MODEL_DIR", model_dir),
            patch(
                "foresightrlcore.runtime.scripts.train.SM_CHECKPOINT_DIR",
                checkpoint_dir,
            ),
        ):
            _apply_sagemaker_overrides(config, args)

        assert config.data.eval_path == str(eval_dir)

    def test_no_eval_override_if_dir_empty(self, tmp_yaml, tmp_path):
        from foresightrlcore.runtime.scripts.train import _apply_sagemaker_overrides

        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        model_dir = str(tmp_path / "model")
        checkpoint_dir = str(tmp_path / "checkpoints")

        config = self._make_config(tmp_yaml)
        args = SimpleNamespace(vllm_host=None)

        with (
            patch("foresightrlcore.runtime.scripts.train.SM_EVAL_DIR", str(eval_dir)),
            patch("foresightrlcore.runtime.scripts.train.SM_MODEL_DIR", model_dir),
            patch(
                "foresightrlcore.runtime.scripts.train.SM_CHECKPOINT_DIR",
                checkpoint_dir,
            ),
        ):
            _apply_sagemaker_overrides(config, args)

        assert config.data.eval_path is None

    def test_sets_local_output_dir(self, tmp_yaml, tmp_path):
        from foresightrlcore.runtime.scripts.train import _apply_sagemaker_overrides

        model_dir = str(tmp_path / "model")
        checkpoint_dir = str(tmp_path / "checkpoints")
        config = self._make_config(tmp_yaml)
        args = SimpleNamespace(vllm_host=None)

        with (
            patch("foresightrlcore.runtime.scripts.train.SM_MODEL_DIR", model_dir),
            patch(
                "foresightrlcore.runtime.scripts.train.SM_CHECKPOINT_DIR",
                checkpoint_dir,
            ),
            patch(
                "foresightrlcore.runtime.scripts.train.SM_EVAL_DIR",
                "/nonexistent/eval",
            ),
        ):
            _apply_sagemaker_overrides(config, args)

        # Intermediate checkpoints stream from /opt/ml/checkpoints; final
        # model is saved to /opt/ml/model at the end of training.
        assert config._local_output_dir == checkpoint_dir  # type: ignore[attr-defined]
        assert config._final_model_dir == model_dir  # type: ignore[attr-defined]

    def test_resolves_deepspeed_config(self, tmp_yaml, tmp_path):
        from foresightrlcore.runtime.scripts.train import _apply_sagemaker_overrides

        ds_file = tmp_path / "ds_zero2_config.json"
        ds_file.write_text("{}")
        model_dir = str(tmp_path / "model")
        checkpoint_dir = str(tmp_path / "checkpoints")

        config = self._make_config(tmp_yaml)
        args = SimpleNamespace(vllm_host=None)

        with (
            patch("foresightrlcore.runtime.scripts.train._SCRIPTS_DIR", str(tmp_path)),
            patch("foresightrlcore.runtime.scripts.train.SM_MODEL_DIR", model_dir),
            patch(
                "foresightrlcore.runtime.scripts.train.SM_CHECKPOINT_DIR",
                checkpoint_dir,
            ),
            patch(
                "foresightrlcore.runtime.scripts.train.SM_EVAL_DIR",
                "/nonexistent/eval",
            ),
        ):
            _apply_sagemaker_overrides(config, args)

        assert config.training.deepspeed == str(ds_file)


class TestMaybeNcclWarmup:
    """Tests for _maybe_nccl_warmup."""

    def test_noop_for_single_node(self):
        from foresightrlcore.runtime.scripts.train import _maybe_nccl_warmup

        _maybe_nccl_warmup({"infra": {"instance_count": 1}})

    def test_noop_for_local_execution(self):
        from foresightrlcore.runtime.scripts.train import _maybe_nccl_warmup

        _maybe_nccl_warmup(
            {
                "infra": {"instance_count": 2},
                "execution": {"type": "local"},
            }
        )


class TestRunWithHandler:
    """Tests for _run_with_handler dispatch."""

    def test_dispatches_to_grpo_handler(self, tmp_yaml):
        from foresightrlcore.runtime.scripts.train import _run_with_handler

        raw_config = {
            "job": {"name": "test"},
            "model": {"name": "m"},
            "method": {"type": "grpo"},
            "data": {"train_path": "s3://b/d"},
            "output": {"s3_path": "s3://b/o"},
            "task": {"type": "advice"},
        }
        args = SimpleNamespace(vllm_host=None)

        mock_handler = MagicMock()
        mock_handler_cls = MagicMock(return_value=mock_handler)

        with (
            patch(
                "foresightrlcore.runtime.scripts.train.SM_MODEL_DIR",
                str(tmp_yaml("").parent / "model"),
            ),
            patch(
                "foresightrlcore.runtime.scripts.train.SM_CHECKPOINT_DIR",
                str(tmp_yaml("").parent / "checkpoints"),
            ),
            patch(
                "foresightrlcore.runtime.scripts.train.SM_TRAIN_DIR",
                str(tmp_yaml("").parent),
            ),
            patch(
                "foresightrlcore.runtime.scripts.train.SM_EVAL_DIR",
                "/nonexistent/eval",
            ),
            patch("foresightrlcore.methods.get_method", return_value=mock_handler_cls),
            patch(
                "torch.distributed.elastic.multiprocessing.errors.record",
                side_effect=lambda fn: fn,
            ),
        ):
            _run_with_handler(raw_config, args)

        mock_handler.run.assert_called_once()

    def test_unknown_method_raises_key_error(self):
        from foresightrlcore.runtime.scripts.train import _run_with_handler

        raw_config = {"method": {"type": "nonexistent_xyz"}}
        args = SimpleNamespace(vllm_host=None)

        with (
            patch(
                "torch.distributed.elastic.multiprocessing.errors.record",
                side_effect=lambda fn: fn,
            ),
        ):
            try:
                _run_with_handler(raw_config, args)
                assert False, "Should have raised"
            except KeyError as exc:
                assert "nonexistent_xyz" in str(exc)


class TestBackwardCompat:
    """Existing configs without method key default to GRPO."""

    def test_no_method_key_defaults_to_grpo(self, tmp_yaml):
        from foresightrlcore.runtime.scripts.train import _run_with_handler

        raw_config = {
            "job": {"name": "test"},
            "model": {"name": "m"},
            "data": {"train_path": "s3://b/d"},
            "output": {"s3_path": "s3://b/o"},
            "task": {"type": "advice"},
        }
        args = SimpleNamespace(vllm_host=None)

        mock_handler = MagicMock()
        mock_handler_cls = MagicMock(return_value=mock_handler)

        with (
            patch(
                "foresightrlcore.runtime.scripts.train.SM_MODEL_DIR",
                str(tmp_yaml("").parent / "model"),
            ),
            patch(
                "foresightrlcore.runtime.scripts.train.SM_CHECKPOINT_DIR",
                str(tmp_yaml("").parent / "checkpoints"),
            ),
            patch(
                "foresightrlcore.runtime.scripts.train.SM_TRAIN_DIR",
                str(tmp_yaml("").parent),
            ),
            patch(
                "foresightrlcore.runtime.scripts.train.SM_EVAL_DIR",
                "/nonexistent/eval",
            ),
            patch("foresightrlcore.methods.get_method", return_value=mock_handler_cls),
            patch(
                "torch.distributed.elastic.multiprocessing.errors.record",
                side_effect=lambda fn: fn,
            ),
        ):
            _run_with_handler(raw_config, args)

        mock_handler_cls.assert_called_once()
        mock_handler.run.assert_called_once()


# ── _resolve_deepspeed_config edge cases ──────────────────────────────


class TestResolveDeepspeedConfig:
    """Cover lines 44, 46, 51 in train.py."""

    def test_returns_none_for_none(self):
        from foresightrlcore.runtime.scripts.train import _resolve_deepspeed_config

        assert _resolve_deepspeed_config(None) is None

    def test_returns_absolute_path_unchanged(self):
        from foresightrlcore.runtime.scripts.train import _resolve_deepspeed_config

        assert _resolve_deepspeed_config("/abs/path/ds.json") == "/abs/path/ds.json"

    def test_returns_existing_relative_path(self, tmp_path):
        from foresightrlcore.runtime.scripts.train import _resolve_deepspeed_config

        f = tmp_path / "ds.json"
        f.write_text("{}")
        assert _resolve_deepspeed_config(str(f)) == str(f)

    def test_falls_back_to_packaged(self, tmp_path):
        from foresightrlcore.runtime.scripts.train import _resolve_deepspeed_config

        packaged = tmp_path / "ds_custom.json"
        packaged.write_text("{}")

        with patch("foresightrlcore.runtime.scripts.train._SCRIPTS_DIR", str(tmp_path)):
            result = _resolve_deepspeed_config("ds_custom.json")

        assert result == str(packaged)

    def test_returns_original_when_not_found(self):
        from foresightrlcore.runtime.scripts.train import _resolve_deepspeed_config

        with patch(
            "foresightrlcore.runtime.scripts.train._SCRIPTS_DIR", "/nonexistent"
        ):
            result = _resolve_deepspeed_config("missing.json")

        assert result == "missing.json"


# ── _maybe_nccl_warmup multi-node path ───────────────────────────────


class TestMaybeNcclWarmupMultiNode:
    """Cover lines 127-158 in train.py for the multi-node code path."""

    def test_multi_node_sets_timeout(self):
        from foresightrlcore.runtime.scripts.train import _maybe_nccl_warmup

        raw = {
            "infra": {"instance_count": 2, "vllm_mode": "server"},
            "execution": {"type": "sagemaker"},
        }

        mock_c10d = MagicMock()
        mock_dist = MagicMock()
        mock_dist.is_initialized.return_value = False

        with (
            patch.dict(
                "sys.modules",
                {
                    "torch": MagicMock(),
                    "torch.distributed": mock_dist,
                    "torch.distributed.distributed_c10d": mock_c10d,
                },
            ),
            patch.dict(os.environ, {"NCCL_TIMEOUT": "1800000"}),
        ):
            _maybe_nccl_warmup(raw)

    def test_multi_node_colocate_with_warmup(self):
        from foresightrlcore.runtime.scripts.train import _maybe_nccl_warmup

        raw = {
            "infra": {"instance_count": 2, "vllm_mode": "colocate"},
            "execution": {"type": "sagemaker"},
        }

        mock_barrier = MagicMock()
        mock_all_reduce = MagicMock()

        with (
            patch.dict(os.environ, {"NCCL_TIMEOUT": "120", "LOCAL_RANK": "0"}),
            patch("torch.distributed.distributed_c10d.default_pg_timeout", create=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.barrier", mock_barrier),
            patch("torch.distributed.all_reduce", mock_all_reduce),
            patch("torch.cuda.set_device"),
            patch("torch.zeros", return_value=MagicMock()),
        ):
            _maybe_nccl_warmup(raw)

        assert mock_barrier.call_count == 2
        mock_all_reduce.assert_called_once()


# ── main() entry point ───────────────────────────────────────────────


class TestMainEntryPoint:
    """Cover lines 284-297 in train.py — secrets resolution branches."""

    def test_main_with_local_secrets(self, tmp_path):
        from foresightrlcore.runtime.scripts.train import main

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            yaml.dump(
                {
                    "job": {"name": "test"},
                    "model": {"name": "m"},
                    "data": {"train_path": "s3://b/d"},
                    "output": {"s3_path": "s3://b/o"},
                    "task": {"type": "advice"},
                    "secrets": {"provider": "local"},
                }
            )
        )

        with (
            patch("sys.argv", ["train.py", "--config", str(cfg_file)]),
            patch("foresightrlcore.auth.resolve_secrets") as mock_resolve,
            patch("foresightrlcore.runtime.scripts.train._run_with_handler"),
        ):
            main()

        mock_resolve.assert_called_once_with(str(cfg_file))

    def test_main_with_idps_secrets(self, tmp_path):
        from foresightrlcore.runtime.scripts.train import main

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            yaml.dump(
                {
                    "job": {"name": "test"},
                    "model": {"name": "m"},
                    "data": {"train_path": "s3://b/d"},
                    "output": {"s3_path": "s3://b/o"},
                    "task": {"type": "advice"},
                    "secrets": {"provider": "idps"},
                }
            )
        )

        with (
            patch("sys.argv", ["train.py", "--config", str(cfg_file)]),
            patch(
                "foresightrlcore.runtime.scripts.train._resolve_secrets_if_configured"
            ) as mock_resolve,
            patch("foresightrlcore.runtime.scripts.train._run_with_handler"),
        ):
            main()

        mock_resolve.assert_called_once_with(str(cfg_file))

    def test_main_no_secrets(self, tmp_path):
        from foresightrlcore.runtime.scripts.train import main

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            yaml.dump(
                {
                    "job": {"name": "test"},
                    "model": {"name": "m"},
                    "data": {"train_path": "s3://b/d"},
                    "output": {"s3_path": "s3://b/o"},
                    "task": {"type": "advice"},
                }
            )
        )

        with (
            patch("sys.argv", ["train.py", "--config", str(cfg_file)]),
            patch(
                "foresightrlcore.runtime.scripts.train._run_with_handler"
            ) as mock_run,
        ):
            main()

        mock_run.assert_called_once()
