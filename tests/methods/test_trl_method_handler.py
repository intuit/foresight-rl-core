"""Tests for TrlMethodHandler base class and GrpoMethodHandler subclass."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import yaml

from foresightrlcore.config.schema import TrainingJobConfig
from foresightrlcore.methods.base import (
    TrlMethodHandler,
    _find_latest_checkpoint,
    _patch_vllm_enforce_eager,
)
from foresightrlcore.methods.grpo import GrpoMethodHandler

# ── TrlMethodHandler base ─────────────────────────────────────────────


class TestTrlMethodHandler:
    """Unit tests for the generic TRL method handler base class."""

    def _make_handler(self) -> TrlMethodHandler:
        return TrlMethodHandler(
            method_type="grpo",
            default_trainer_path="trl.GRPOTrainer",
            default_config_path="trl.GRPOConfig",
        )

    def test_method_type_set(self):
        handler = self._make_handler()
        assert handler.method_type == "grpo"

    def test_validate_missing_training(self):
        handler = self._make_handler()
        obj = type("C", (), {})()
        errors = handler.validate(obj)
        assert any("Training config section is missing" in e for e in errors)

    def test_validate_ok(self, full_config_yaml):
        handler = self._make_handler()
        config = TrainingJobConfig.from_yaml(full_config_yaml)
        errors = handler.validate(config)
        assert errors == []

    def test_build_trainer_args_common_fields(self, full_config_yaml):
        handler = self._make_handler()
        config = TrainingJobConfig.from_yaml(full_config_yaml)
        args = handler.build_trainer_args(config)

        assert "output_dir" in args
        assert args["learning_rate"] == 5e-5
        assert args["num_train_epochs"] == 1
        assert args["per_device_train_batch_size"] == 4
        assert args["use_vllm"] is True
        assert "gradient_checkpointing_kwargs" in args
        assert args["use_liger_kernel"] is False

    def test_build_trainer_args_use_liger_kernel_opt_in(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "training": {"use_liger_kernel": True},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = self._make_handler()
        args = handler.build_trainer_args(config)
        assert args["use_liger_kernel"] is True

    def test_build_trainer_args_no_grpo_fields(self, full_config_yaml):
        """Base handler should NOT include GRPO-specific fields."""
        handler = self._make_handler()
        config = TrainingJobConfig.from_yaml(full_config_yaml)
        args = handler.build_trainer_args(config)

        assert "num_generations" not in args
        assert "beta" not in args
        assert "loss_type" not in args

    def test_build_trainer_args_vllm_none(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "infra": {"vllm_mode": "none"},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = self._make_handler()
        args = handler.build_trainer_args(config)
        assert args["use_vllm"] is False

    def test_build_trainer_args_vllm_colocate(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "infra": {
                    "vllm_mode": "colocate",
                    "vllm": {"gpu_memory_utilization": 0.7},
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = self._make_handler()
        args = handler.build_trainer_args(config)
        assert args["use_vllm"] is True
        assert args["vllm_mode"] == "colocate"
        assert args["vllm_gpu_memory_utilization"] == 0.7

    def test_build_trainer_args_max_grad_norm(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "training": {"max_grad_norm": 2.0, "max_steps": 50},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = self._make_handler()
        args = handler.build_trainer_args(config)
        assert args["max_grad_norm"] == 2.0
        assert args["max_steps"] == 50

    def test_build_trainer_args_gc_kwargs_custom(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "training": {
                    "gradient_checkpointing_kwargs": {"use_reentrant": True},
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = self._make_handler()
        args = handler.build_trainer_args(config)
        assert args["gradient_checkpointing_kwargs"] == {"use_reentrant": True}

    def test_build_trainer_kwargs_default(self):
        handler = self._make_handler()
        kwargs = handler._build_trainer_kwargs(
            config=None,
            model_name="m",
            trl_config="cfg",
            train_dataset="ds",
            eval_dataset=None,
            peft_config="pc",
            all_callbacks=[],
            reward_fn="rf",
            task_callbacks=[],
        )
        assert kwargs["model"] == "m"
        assert kwargs["args"] == "cfg"
        assert kwargs["train_dataset"] == "ds"
        assert "reward_funcs" not in kwargs


# ── GrpoMethodHandler subclass ────────────────────────────────────────


class TestGrpoMethodHandlerSubclass:
    """Verify GRPO handler inherits and extends the base properly."""

    def test_isinstance_of_base(self):
        handler = GrpoMethodHandler()
        assert isinstance(handler, TrlMethodHandler)

    def test_method_type_class_attr(self):
        assert GrpoMethodHandler.method_type == "grpo"

    def test_method_type_instance(self):
        assert GrpoMethodHandler().method_type == "grpo"

    def test_validate_includes_base_check(self):
        handler = GrpoMethodHandler()
        obj = type("C", (), {})()
        errors = handler.validate(obj)
        assert any("Training config section is missing" in e for e in errors)

    def test_build_trainer_args_has_grpo_fields(self, full_config_yaml):
        config = TrainingJobConfig.from_yaml(full_config_yaml)
        handler = GrpoMethodHandler()
        args = handler.build_trainer_args(config)

        assert "num_generations" in args
        assert args["loss_type"] == "dr_grpo"
        assert args["output_dir"] is not None
        assert args["learning_rate"] == 5e-5

    def test_build_trainer_kwargs_includes_reward_funcs(self):
        handler = GrpoMethodHandler()
        kwargs = handler._build_trainer_kwargs(
            config=None,
            model_name="m",
            trl_config="cfg",
            train_dataset="ds",
            eval_dataset=None,
            peft_config="pc",
            all_callbacks=[],
            reward_fn="my_reward",
            task_callbacks=[],
        )
        assert kwargs["reward_funcs"] == "my_reward"
        assert kwargs["model"] == "m"


# ── TRL_TRAINERS registry ─────────────────────────────────────────────


class TestTrlTrainersRegistry:
    """Tests for the TRL_TRAINERS dotted-path mapping."""

    def test_grpo_in_registry(self):
        from foresightrlcore.methods import TRL_TRAINERS

        assert "grpo" in TRL_TRAINERS
        trainer_path, config_path = TRL_TRAINERS["grpo"]
        assert trainer_path == "trl.GRPOTrainer"
        assert config_path == "trl.GRPOConfig"

    def test_dpo_in_registry(self):
        from foresightrlcore.methods import TRL_TRAINERS

        assert "dpo" in TRL_TRAINERS

    def test_kto_in_registry(self):
        from foresightrlcore.methods import TRL_TRAINERS

        assert "kto" in TRL_TRAINERS


# ── Method-config cross-validation ────────────────────────────────────


class TestMethodConfigCrossValidation:
    """Tests for _validate_method_config_alignment."""

    def test_grpo_method_with_grpo_config_passes(self, full_config_yaml):
        from foresightrlcore.config.validation import validate_config

        config = TrainingJobConfig.from_yaml(full_config_yaml)
        errors = validate_config(config)
        assert not any("section is missing" in e for e in errors)

    def test_method_config_mismatch_warns(self, tmp_yaml, caplog):
        import logging

        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "method": {"type": "sft"},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))

        from foresightrlcore.config.validation import validate_config

        with caplog.at_level(logging.WARNING):
            validate_config(config)

        assert any("training.grpo will be ignored" in r.message for r in caplog.records)


# ── TrlMethodHandler.run() ────────────────────────────────────────────


class TestTrlMethodHandlerRun:
    """Test the run() pipeline with mocked GPU dependencies."""

    def _make_config(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "test-model"},
                "infra": {"vllm_mode": "none"},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice", "config": {"reward_mode": "mock"}},
            }
        )
        return TrainingJobConfig.from_yaml(tmp_yaml(content))

    def test_run_pipeline(self, tmp_yaml):
        config = self._make_config(tmp_yaml)

        mock_trainer_instance = MagicMock()
        mock_trainer_cls = MagicMock(return_value=mock_trainer_instance)
        mock_config_cls = MagicMock()

        mock_train_ds = MagicMock()
        mock_train_ds.__len__ = MagicMock(return_value=10)
        mock_dataset_loader = MagicMock(return_value=mock_train_ds)
        mock_reward_fn = MagicMock()

        handler = TrlMethodHandler(
            method_type="grpo",
            default_trainer_path="collections.OrderedDict",
            default_config_path="collections.OrderedDict",
        )

        with (
            patch(
                "foresightrlcore.hooks.import_dotted_path",
                side_effect=[mock_trainer_cls, mock_config_cls],
            ),
            patch("peft.LoraConfig"),
            patch(
                "foresightrlcore.tasks.get_task_components",
                return_value=(mock_reward_fn, [], mock_dataset_loader),
            ),
            patch(
                "foresightrlcore.hooks.resolve_trainer_class",
                return_value=mock_trainer_cls,
            ),
            patch("foresightrlcore.hooks.resolve_callbacks", return_value=[]),
            patch(
                "foresightrlcore.hooks.resolve_pre_post_hooks",
                return_value=(None, None),
            ),
        ):
            handler.run(config)

        mock_trainer_cls.assert_called_once()
        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.save_model.assert_called_once()

    def test_run_with_eval_split(self, tmp_yaml):
        config = self._make_config(tmp_yaml)

        mock_trainer_instance = MagicMock()
        mock_trainer_cls = MagicMock(return_value=mock_trainer_instance)
        mock_config_cls = MagicMock()

        mock_split_train = MagicMock()
        mock_split_eval = MagicMock()
        mock_split_train.__len__ = MagicMock(return_value=900)
        mock_split_eval.__len__ = MagicMock(return_value=100)

        mock_train_ds = MagicMock()
        mock_train_ds.__len__ = MagicMock(return_value=1000)
        mock_train_ds.train_test_split.return_value = {
            "train": mock_split_train,
            "test": mock_split_eval,
        }
        mock_dataset_loader = MagicMock(return_value=mock_train_ds)

        handler = TrlMethodHandler(
            method_type="grpo",
            default_trainer_path="collections.OrderedDict",
            default_config_path="collections.OrderedDict",
        )

        with (
            patch(
                "foresightrlcore.hooks.import_dotted_path",
                side_effect=[mock_trainer_cls, mock_config_cls],
            ),
            patch("peft.LoraConfig"),
            patch(
                "foresightrlcore.tasks.get_task_components",
                return_value=(MagicMock(), [], mock_dataset_loader),
            ),
            patch(
                "foresightrlcore.hooks.resolve_trainer_class",
                return_value=mock_trainer_cls,
            ),
            patch("foresightrlcore.hooks.resolve_callbacks", return_value=[]),
            patch(
                "foresightrlcore.hooks.resolve_pre_post_hooks",
                return_value=(None, None),
            ),
        ):
            handler.run(config)

        mock_train_ds.train_test_split.assert_called_once()
        mock_config_cls.assert_called_once()
        call_kwargs = mock_config_cls.call_args
        assert call_kwargs[1]["eval_strategy"] == "steps"

    def test_run_with_pre_post_hooks(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "test-model"},
                "infra": {"vllm_mode": "none"},
                "training": {
                    "hooks": {
                        "pre_train": "os.path.join",
                        "post_train": "os.path.exists",
                    }
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))

        mock_trainer_instance = MagicMock()
        mock_trainer_cls = MagicMock(return_value=mock_trainer_instance)
        mock_config_cls = MagicMock()

        mock_train_ds = MagicMock()
        mock_train_ds.__len__ = MagicMock(return_value=10)

        mock_pre = MagicMock()
        mock_post = MagicMock()

        handler = TrlMethodHandler(
            method_type="grpo",
            default_trainer_path="collections.OrderedDict",
            default_config_path="collections.OrderedDict",
        )

        with (
            patch(
                "foresightrlcore.hooks.import_dotted_path",
                side_effect=[mock_trainer_cls, mock_config_cls],
            ),
            patch("peft.LoraConfig"),
            patch(
                "foresightrlcore.tasks.get_task_components",
                return_value=(MagicMock(), [], MagicMock(return_value=mock_train_ds)),
            ),
            patch(
                "foresightrlcore.hooks.resolve_trainer_class",
                return_value=mock_trainer_cls,
            ),
            patch("foresightrlcore.hooks.resolve_callbacks", return_value=[]),
            patch(
                "foresightrlcore.hooks.resolve_pre_post_hooks",
                return_value=(mock_pre, mock_post),
            ),
        ):
            handler.run(config)

        mock_pre.assert_called_once_with(mock_trainer_instance, config)
        mock_post.assert_called_once_with(mock_trainer_instance, config)

    def test_run_with_colocate_enforce_eager(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "test-model"},
                "infra": {
                    "vllm_mode": "colocate",
                    "vllm": {"enforce_eager": True},
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))

        mock_trainer_cls = MagicMock(return_value=MagicMock())
        mock_config_cls = MagicMock()
        mock_train_ds = MagicMock()
        mock_train_ds.__len__ = MagicMock(return_value=10)

        handler = TrlMethodHandler(
            method_type="grpo",
            default_trainer_path="collections.OrderedDict",
            default_config_path="collections.OrderedDict",
        )

        with (
            patch(
                "foresightrlcore.hooks.import_dotted_path",
                side_effect=[mock_trainer_cls, mock_config_cls],
            ),
            patch("peft.LoraConfig"),
            patch(
                "foresightrlcore.tasks.get_task_components",
                return_value=(MagicMock(), [], MagicMock(return_value=mock_train_ds)),
            ),
            patch(
                "foresightrlcore.hooks.resolve_trainer_class",
                return_value=mock_trainer_cls,
            ),
            patch("foresightrlcore.hooks.resolve_callbacks", return_value=[]),
            patch(
                "foresightrlcore.hooks.resolve_pre_post_hooks",
                return_value=(None, None),
            ),
            patch(
                "foresightrlcore.methods.base._patch_vllm_enforce_eager"
            ) as mock_patch_vllm,
        ):
            handler.run(config)

        mock_patch_vllm.assert_called_once_with(enforce_eager=True)

    def test_build_trainer_args_local_output_dir(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        config._local_output_dir = "/tmp/local_out"  # type: ignore[attr-defined]

        handler = TrlMethodHandler(
            method_type="grpo",
            default_trainer_path="trl.GRPOTrainer",
            default_config_path="trl.GRPOConfig",
        )
        args = handler.build_trainer_args(config)
        assert args["output_dir"] == "/tmp/local_out"

    def test_run_with_resume_from_checkpoint(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "test-model"},
                "infra": {"vllm_mode": "none"},
                "training": {"resume_from_checkpoint": True},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice", "config": {"reward_mode": "mock"}},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))

        mock_trainer_instance = MagicMock()
        mock_trainer_cls = MagicMock(return_value=mock_trainer_instance)
        mock_config_cls = MagicMock()

        mock_train_ds = MagicMock()
        mock_train_ds.__len__ = MagicMock(return_value=10)
        mock_dataset_loader = MagicMock(return_value=mock_train_ds)

        handler = TrlMethodHandler(
            method_type="grpo",
            default_trainer_path="collections.OrderedDict",
            default_config_path="collections.OrderedDict",
        )

        with (
            patch(
                "foresightrlcore.hooks.import_dotted_path",
                side_effect=[mock_trainer_cls, mock_config_cls],
            ),
            patch("peft.LoraConfig"),
            patch(
                "foresightrlcore.tasks.get_task_components",
                return_value=(MagicMock(), [], mock_dataset_loader),
            ),
            patch(
                "foresightrlcore.hooks.resolve_trainer_class",
                return_value=mock_trainer_cls,
            ),
            patch("foresightrlcore.hooks.resolve_callbacks", return_value=[]),
            patch(
                "foresightrlcore.hooks.resolve_pre_post_hooks",
                return_value=(None, None),
            ),
        ):
            handler.run(config)

        mock_trainer_instance.train.assert_called_once_with(resume_from_checkpoint=True)

    def test_run_with_resume_from_checkpoint_path(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "test-model"},
                "infra": {"vllm_mode": "none"},
                "training": {
                    "resume_from_checkpoint": "/opt/ml/checkpoints/checkpoint-100"
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice", "config": {"reward_mode": "mock"}},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))

        mock_trainer_instance = MagicMock()
        mock_trainer_cls = MagicMock(return_value=mock_trainer_instance)
        mock_config_cls = MagicMock()

        mock_train_ds = MagicMock()
        mock_train_ds.__len__ = MagicMock(return_value=10)
        mock_dataset_loader = MagicMock(return_value=mock_train_ds)

        handler = TrlMethodHandler(
            method_type="grpo",
            default_trainer_path="collections.OrderedDict",
            default_config_path="collections.OrderedDict",
        )

        with (
            patch(
                "foresightrlcore.hooks.import_dotted_path",
                side_effect=[mock_trainer_cls, mock_config_cls],
            ),
            patch("peft.LoraConfig"),
            patch(
                "foresightrlcore.tasks.get_task_components",
                return_value=(MagicMock(), [], mock_dataset_loader),
            ),
            patch(
                "foresightrlcore.hooks.resolve_trainer_class",
                return_value=mock_trainer_cls,
            ),
            patch("foresightrlcore.hooks.resolve_callbacks", return_value=[]),
            patch(
                "foresightrlcore.hooks.resolve_pre_post_hooks",
                return_value=(None, None),
            ),
        ):
            handler.run(config)

        mock_trainer_instance.train.assert_called_once_with(
            resume_from_checkpoint="/opt/ml/checkpoints/checkpoint-100"
        )

    def test_run_without_resume_calls_train_plain(self, tmp_yaml):
        config = self._make_config(tmp_yaml)

        mock_trainer_instance = MagicMock()
        mock_trainer_cls = MagicMock(return_value=mock_trainer_instance)
        mock_config_cls = MagicMock()

        mock_train_ds = MagicMock()
        mock_train_ds.__len__ = MagicMock(return_value=10)
        mock_dataset_loader = MagicMock(return_value=mock_train_ds)

        handler = TrlMethodHandler(
            method_type="grpo",
            default_trainer_path="collections.OrderedDict",
            default_config_path="collections.OrderedDict",
        )

        with (
            patch(
                "foresightrlcore.hooks.import_dotted_path",
                side_effect=[mock_trainer_cls, mock_config_cls],
            ),
            patch("peft.LoraConfig"),
            patch(
                "foresightrlcore.tasks.get_task_components",
                return_value=(MagicMock(), [], mock_dataset_loader),
            ),
            patch(
                "foresightrlcore.hooks.resolve_trainer_class",
                return_value=mock_trainer_cls,
            ),
            patch("foresightrlcore.hooks.resolve_callbacks", return_value=[]),
            patch(
                "foresightrlcore.hooks.resolve_pre_post_hooks",
                return_value=(None, None),
            ),
        ):
            handler.run(config)

        mock_trainer_instance.train.assert_called_once_with()

    def test_run_auto_resumes_from_detected_checkpoint(self, tmp_yaml):
        config = self._make_config(tmp_yaml)

        mock_trainer_instance = MagicMock()
        mock_trainer_cls = MagicMock(return_value=mock_trainer_instance)
        mock_config_cls = MagicMock()

        mock_train_ds = MagicMock()
        mock_train_ds.__len__ = MagicMock(return_value=10)
        mock_dataset_loader = MagicMock(return_value=mock_train_ds)

        handler = TrlMethodHandler(
            method_type="grpo",
            default_trainer_path="collections.OrderedDict",
            default_config_path="collections.OrderedDict",
        )

        with (
            patch(
                "foresightrlcore.hooks.import_dotted_path",
                side_effect=[mock_trainer_cls, mock_config_cls],
            ),
            patch("peft.LoraConfig"),
            patch(
                "foresightrlcore.tasks.get_task_components",
                return_value=(MagicMock(), [], mock_dataset_loader),
            ),
            patch(
                "foresightrlcore.hooks.resolve_trainer_class",
                return_value=mock_trainer_cls,
            ),
            patch("foresightrlcore.hooks.resolve_callbacks", return_value=[]),
            patch(
                "foresightrlcore.hooks.resolve_pre_post_hooks",
                return_value=(None, None),
            ),
            patch(
                "foresightrlcore.methods.base._find_latest_checkpoint",
                return_value="/opt/ml/checkpoints/checkpoint-200",
            ),
        ):
            handler.run(config)

        mock_trainer_instance.train.assert_called_once_with(
            resume_from_checkpoint="/opt/ml/checkpoints/checkpoint-200"
        )

    def test_run_mixed_restores_s3_checkpoints_and_registers_callback(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "test-model"},
                "infra": {
                    "topology": "dedicated_vllm",
                    "vllm_mode": "server",
                    "instance_type": "ml.g6.12xlarge",
                    "instance_count": 2,
                    "vllm_instance_type": "ml.g6.16xlarge",
                },
                "data": {"train_path": "s3://b/d"},
                "output": {
                    "s3_path": "s3://b/o",
                    "checkpoint_s3_path": "s3://b/ckpt",
                },
                "task": {"type": "advice", "config": {"reward_mode": "mock"}},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))

        mock_trainer_instance = MagicMock()
        mock_trainer_cls = MagicMock(return_value=mock_trainer_instance)
        mock_config_cls = MagicMock()
        mock_train_ds = MagicMock()
        mock_train_ds.__len__ = MagicMock(return_value=10)
        mock_cb = MagicMock()

        handler = TrlMethodHandler(
            method_type="grpo",
            default_trainer_path="collections.OrderedDict",
            default_config_path="collections.OrderedDict",
        )

        with (
            patch(
                "foresightrlcore.hooks.import_dotted_path",
                side_effect=[mock_trainer_cls, mock_config_cls],
            ),
            patch("peft.LoraConfig"),
            patch(
                "foresightrlcore.tasks.get_task_components",
                return_value=(
                    MagicMock(),
                    [],
                    MagicMock(return_value=mock_train_ds),
                ),
            ),
            patch(
                "foresightrlcore.hooks.resolve_trainer_class",
                return_value=mock_trainer_cls,
            ),
            patch("foresightrlcore.hooks.resolve_callbacks", return_value=[]),
            patch(
                "foresightrlcore.hooks.resolve_pre_post_hooks",
                return_value=(None, None),
            ),
            patch(
                "foresightrlcore.methods.base._make_mixed_checkpoint_callback",
                return_value=mock_cb,
            ) as make_cb,
            patch("foresightrlcore.methods.base._restore_mixed_checkpoints") as restore,
            patch(
                "foresightrlcore.methods.base._find_latest_checkpoint",
                return_value="/opt/ml/checkpoints/checkpoint-5",
            ),
        ):
            handler.run(config)

        make_cb.assert_called_once_with("s3://b/ckpt")
        restore.assert_called_once()
        assert restore.call_args[0][0] == "s3://b/ckpt"
        mock_trainer_instance.train.assert_called_once_with(
            resume_from_checkpoint="/opt/ml/checkpoints/checkpoint-5"
        )
        callbacks = mock_trainer_cls.call_args.kwargs.get(
            "callbacks"
        ) or mock_trainer_cls.call_args[1].get("callbacks")
        assert mock_cb in callbacks

    def test_build_trainer_args_colocate_max_model_len(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "infra": {
                    "vllm_mode": "colocate",
                    "vllm": {"max_model_len": 4096},
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = TrlMethodHandler(
            method_type="grpo",
            default_trainer_path="trl.GRPOTrainer",
            default_config_path="trl.GRPOConfig",
        )
        args = handler.build_trainer_args(config)
        assert args["vllm_max_model_length"] == 4096


# ── _patch_vllm_enforce_eager ─────────────────────────────────────────


class TestPatchVllmEnforceEagerBase:
    """Tests for the _patch_vllm_enforce_eager in methods/base.py."""

    def test_patch_replaces_llm_class(self):
        mock_vllm_gen = MagicMock()
        mock_original_llm = type("OrigLLM", (), {"__init__": lambda self: None})

        with patch.dict(
            "sys.modules",
            {
                "trl.generation.vllm_generation": mock_vllm_gen,
                "vllm": MagicMock(LLM=mock_original_llm),
            },
        ):
            _patch_vllm_enforce_eager(enforce_eager=True)

        assert hasattr(mock_vllm_gen, "LLM")

    def test_patch_handles_import_failure(self):
        with patch.dict("sys.modules", {"trl.generation.vllm_generation": None}):
            _patch_vllm_enforce_eager(enforce_eager=True)


class TestFindLatestCheckpoint:
    """Auto-resume checkpoint detection for Managed Spot Training restarts."""

    def test_missing_dir_returns_none(self, tmp_path):
        assert _find_latest_checkpoint(str(tmp_path / "does-not-exist")) is None

    def test_empty_string_returns_none(self):
        assert _find_latest_checkpoint("") is None

    def test_no_checkpoints_returns_none(self, tmp_path):
        (tmp_path / "runs").mkdir()
        (tmp_path / "config.json").write_text("{}")
        assert _find_latest_checkpoint(str(tmp_path)) is None

    def test_returns_highest_step_checkpoint(self, tmp_path):
        for step in (20, 200, 100):
            (tmp_path / f"checkpoint-{step}").mkdir()
        result = _find_latest_checkpoint(str(tmp_path))
        assert result == str(tmp_path / "checkpoint-200")

    def test_ignores_non_matching_and_files(self, tmp_path):
        (tmp_path / "checkpoint-50").mkdir()
        # A file (not a dir) that matches the pattern must be ignored.
        (tmp_path / "checkpoint-999").write_text("not a dir")
        # A dir that does not match the pattern must be ignored.
        (tmp_path / "checkpoint-final").mkdir()
        result = _find_latest_checkpoint(str(tmp_path))
        assert result == str(tmp_path / "checkpoint-50")
