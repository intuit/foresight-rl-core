"""Tests for foresightrlcore.methods — method handler protocol and GRPO handler."""

from __future__ import annotations

from foresightrlcore.config.schema import TrainingJobConfig
from foresightrlcore.methods.grpo import GrpoMethodHandler


class TestGrpoMethodHandler:
    """GRPO method handler unit tests."""

    def test_validate_valid_config(self, full_config_yaml):
        config = TrainingJobConfig.from_yaml(full_config_yaml)
        handler = GrpoMethodHandler()
        errors = handler.validate(config)
        assert errors == []

    def test_validate_low_num_generations(self, tmp_yaml):
        import yaml

        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "training": {"grpo": {"num_generations": 1}},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = GrpoMethodHandler()
        errors = handler.validate(config)
        assert any("num_generations" in e for e in errors)

    def test_build_trainer_args_structure(self, full_config_yaml):
        config = TrainingJobConfig.from_yaml(full_config_yaml)
        handler = GrpoMethodHandler()
        args = handler.build_trainer_args(config)

        assert "output_dir" in args
        assert "num_generations" in args
        assert args["num_generations"] == 8
        assert args["learning_rate"] == 5e-5
        assert args["use_vllm"] is True
        assert args["loss_type"] == "dr_grpo"

    def test_build_trainer_args_no_vllm(self, tmp_yaml):
        import yaml

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
        handler = GrpoMethodHandler()
        args = handler.build_trainer_args(config)

        assert args["use_vllm"] is False
        assert "vllm_server_host" not in args

    def test_build_trainer_args_colocate(self, tmp_yaml):
        import yaml

        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "infra": {
                    "vllm_mode": "colocate",
                    "vllm": {"gpu_memory_utilization": 0.8, "tensor_parallel_size": 1},
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = GrpoMethodHandler()
        args = handler.build_trainer_args(config)

        assert args["use_vllm"] is True
        assert args["vllm_mode"] == "colocate"
        assert args["vllm_gpu_memory_utilization"] == 0.8

    def test_method_type_attribute(self):
        handler = GrpoMethodHandler()
        assert handler.method_type == "grpo"

    def test_build_trainer_args_new_always_on_fields(self, full_config_yaml):
        config = TrainingJobConfig.from_yaml(full_config_yaml)
        handler = GrpoMethodHandler()
        args = handler.build_trainer_args(config)

        assert args["weight_decay"] == 0.0
        assert args["fp16"] is False
        assert args["repetition_penalty"] == 1.0
        assert args["sync_ref_model"] is False
        assert args["ref_model_mixup_alpha"] == 0.6
        assert args["ref_model_sync_steps"] == 512
        assert args["multi_objective_aggregation"] == "sum_then_normalize"
        assert args["top_entropy_quantile"] == 1.0
        assert args["vllm_importance_sampling_cap"] == 3.0
        assert args["use_bias_correction_kl"] is False
        assert args["disable_dropout"] is False
        assert args["cast_lm_head_to_fp32"] is False
        assert args["ds3_gather_for_generation"] is True
        assert args["shuffle_dataset"] is True

    def test_build_trainer_args_optional_excluded_when_none(self, full_config_yaml):
        config = TrainingJobConfig.from_yaml(full_config_yaml)
        handler = GrpoMethodHandler()
        args = handler.build_trainer_args(config)

        assert "min_p" not in args
        assert "generation_kwargs" not in args
        assert "generation_batch_size" not in args
        assert "num_generations_eval" not in args
        assert "delta" not in args
        assert "vllm_importance_sampling_mode" not in args
        assert "off_policy_mask_threshold" not in args
        assert "max_tool_calling_iterations" not in args
        assert "max_grad_norm" not in args
        assert "max_steps" not in args

    def test_build_trainer_args_optional_included_when_set(self, tmp_yaml):
        import yaml

        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "training": {
                    "max_grad_norm": 1.0,
                    "max_steps": 100,
                    "grpo": {
                        "min_p": 0.05,
                        "delta": 1.5,
                        "generation_batch_size": 32,
                        "num_generations_eval": 4,
                        "vllm_importance_sampling_mode": "sequence_mask",
                        "off_policy_mask_threshold": 0.5,
                        "max_tool_calling_iterations": 3,
                        "importance_sampling_level": "sequence",
                        "reward_weights": [0.5, 0.5],
                    },
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        handler = GrpoMethodHandler()
        args = handler.build_trainer_args(config)

        assert args["min_p"] == 0.05
        assert args["delta"] == 1.5
        assert args["generation_batch_size"] == 32
        assert args["num_generations_eval"] == 4
        assert args["vllm_importance_sampling_mode"] == "sequence_mask"
        assert args["off_policy_mask_threshold"] == 0.5
        assert args["max_tool_calling_iterations"] == 3
        assert args["importance_sampling_level"] == "sequence"
        assert args["reward_weights"] == [0.5, 0.5]
        assert args["max_grad_norm"] == 1.0
        assert args["max_steps"] == 100
