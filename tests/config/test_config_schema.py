"""Tests for foresightrlcore.config.schema — Phase 1 config schema."""

from __future__ import annotations

import pytest
import yaml

from foresightrlcore.config.schema import (
    ContainerConfig,
    DataConfig,
    GrpoConfig,
    HooksConfig,
    InfraConfig,
    LoraConfig_,
    ModelConfig,
    OutputConfig,
    SageMakerTrainingConfig,
    Topology,
    TrainingConfig,
    TrainingJobConfig,
    VllmMode,
)


class TestTrainingJobConfigLoading:
    """Config loading from YAML files."""

    def test_load_minimal_config(self, minimal_config_yaml):
        config = TrainingJobConfig.from_yaml(minimal_config_yaml)
        assert config.job.name == "test-job"
        assert config.model.name == "Qwen/Qwen3-4B"
        assert config.task is not None and config.task.type == "advice"

    def test_load_full_config(self, full_config_yaml):
        config = TrainingJobConfig.from_yaml(full_config_yaml)
        assert config.job.name == "full-test-job"
        assert config.method.type == "grpo"
        assert config.execution.type == "aws_training_job"
        assert config.training.grpo.loss_type == "dr_grpo"

    def test_load_local_config(self, local_config_yaml):
        config = TrainingJobConfig.from_yaml(local_config_yaml)
        assert config.execution.type == "local"
        assert config.execution.local["gpus"] == "0,1,2,3,4,5,6,7"  # type: ignore[index]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            TrainingJobConfig.from_yaml("/nonexistent/config.yaml")

    def test_invalid_yaml_raises(self, tmp_yaml):
        path = tmp_yaml("not: valid: yaml: [")
        with pytest.raises(Exception):
            TrainingJobConfig.from_yaml(path)


class TestNewPhase1Fields:
    """Phase 1 additions: method, execution, version."""

    def test_defaults_for_new_fields(self, minimal_config_yaml):
        config = TrainingJobConfig.from_yaml(minimal_config_yaml)
        assert config.version == 1
        assert config.method.type == "grpo"
        assert config.execution.type == "aws_training_job"

    def test_method_config_custom(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "test-model"},
                "method": {"type": "sft"},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        assert config.method.type == "sft"

    def test_execution_local_config(self, local_config_yaml):
        config = TrainingJobConfig.from_yaml(local_config_yaml)
        assert config.execution.type == "local"
        assert config.execution.local is not None
        assert "gpus" in config.execution.local


class TestBackwardCompatibility:
    """Ensure old configs still work with the new schema."""

    def test_sagemaker_alias(self, minimal_config_yaml):
        config = SageMakerTrainingConfig.from_yaml(minimal_config_yaml)
        assert config.job.name == "test-job"
        assert isinstance(config, TrainingJobConfig)

    def test_old_config_without_new_fields(self, minimal_config_yaml):
        config = TrainingJobConfig.from_yaml(minimal_config_yaml)
        assert config.method.type == "grpo"
        assert config.execution.type == "aws_training_job"

    def test_sagemaker_alias_is_same_class(self):
        from foresightrlcore.config.schema import SageMakerTrainingConfig

        assert SageMakerTrainingConfig is TrainingJobConfig


class TestMergeOverrides:
    """Test dotted-key CLI overrides."""

    def test_shortcut_overrides(self, minimal_config_yaml):
        config = TrainingJobConfig.from_yaml(minimal_config_yaml)
        updated = config.merge_overrides(model_name="new-model")
        assert updated.model.name == "new-model"
        assert config.model.name == "Qwen/Qwen3-4B"

    def test_dotted_key_overrides(self, minimal_config_yaml):
        config = TrainingJobConfig.from_yaml(minimal_config_yaml)
        updated = config.merge_overrides(**{"training.learning_rate": 1e-4})
        assert updated.training.learning_rate == 1e-4

    def test_none_values_ignored(self, minimal_config_yaml):
        config = TrainingJobConfig.from_yaml(minimal_config_yaml)
        updated = config.merge_overrides(model_name=None)
        assert updated.model.name == config.model.name


class TestSerializationRoundtrip:
    """Test YAML serialization and deserialization."""

    def test_roundtrip(self, full_config_yaml, tmp_path):
        config = TrainingJobConfig.from_yaml(full_config_yaml)
        out = tmp_path / "roundtrip.yaml"
        config.to_yaml(out)
        config2 = TrainingJobConfig.from_yaml(out)
        assert config2.job.name == config.job.name
        assert config2.method.type == config.method.type
        assert config2.execution.type == config.execution.type


class TestDataConfigValidation:
    """Data path validation."""

    def test_empty_train_path_rejected(self):
        with pytest.raises(Exception):
            DataConfig(train_path="")

    def test_local_path_allowed(self):
        dc = DataConfig(train_path="/local/data/train.jsonl")
        assert dc.train_path == "/local/data/train.jsonl"

    def test_s3_path_allowed(self):
        dc = DataConfig(train_path="s3://bucket/train")
        assert dc.train_path is not None and dc.train_path.startswith("s3://")


class TestOutputConfigValidation:
    """Output path validation."""

    def test_empty_output_rejected(self):
        with pytest.raises(Exception):
            OutputConfig(s3_path="")

    def test_local_output_allowed(self):
        oc = OutputConfig(s3_path="/local/output")
        assert oc.s3_path == "/local/output"


class TestSubConfigDefaults:
    """Verify important defaults match the original config.py."""

    def test_infra_defaults(self):
        infra = InfraConfig()
        assert infra.topology == Topology.SINGLE_NODE
        assert infra.vllm_mode == VllmMode.SERVER
        assert infra.instance_type == "ml.p4d.24xlarge"
        assert infra.vllm_gpu_count == 2
        assert infra.training_gpu_count == 6
        # ACP IS031.1: in-transit encryption is on by default.
        assert infra.encrypt_inter_container_traffic is True

    def test_grpo_defaults(self):
        grpo = GrpoConfig()
        assert grpo.num_generations == 8
        assert grpo.beta == 0.001
        assert grpo.loss_type == "dr_grpo"

    def test_lora_defaults(self):
        lora = LoraConfig_()
        assert lora.r == 32
        assert lora.alpha == 64
        assert "q_proj" in lora.target_modules


class TestContainerConfigDefaults:
    """ContainerConfig defaults and backward compatibility."""

    def test_default_framework_version_fallback(self):
        c = ContainerConfig()
        assert c.framework_version == "2.8.0"

    def test_default_framework(self):
        c = ContainerConfig()
        assert c.framework == "pytorch"

    def test_default_recipe_is_none(self):
        c = ContainerConfig()
        assert c.recipe is None

    def test_default_requirements_is_none(self):
        c = ContainerConfig()
        assert c.requirements is None

    def test_explicit_framework_version_preserved(self):
        c = ContainerConfig(framework_version="2.7.0")
        assert c.framework_version == "2.7.0"

    def test_old_config_without_recipe_works(self, minimal_config_yaml):
        config = TrainingJobConfig.from_yaml(minimal_config_yaml)
        assert config.container.framework_version == "2.8.0"
        assert config.container.recipe is None
        assert config.container.resolved_requirements is None


class TestGrpoConfigNewFields:
    """Verify new GRPO config fields have correct defaults and accept values."""

    def test_generation_control_defaults(self):
        grpo = GrpoConfig()
        assert grpo.min_p is None
        assert grpo.repetition_penalty == 1.0
        assert grpo.generation_kwargs is None
        assert grpo.generation_batch_size is None
        assert grpo.num_generations_eval is None

    def test_advanced_loss_defaults(self):
        grpo = GrpoConfig()
        assert grpo.delta is None
        assert grpo.sync_ref_model is False
        assert grpo.ref_model_mixup_alpha == 0.6
        assert grpo.ref_model_sync_steps == 512
        assert grpo.multi_objective_aggregation == "sum_then_normalize"
        assert grpo.top_entropy_quantile == 1.0

    def test_importance_sampling_defaults(self):
        grpo = GrpoConfig()
        assert grpo.vllm_importance_sampling_mode is None
        assert grpo.vllm_importance_sampling_cap == 3.0
        assert grpo.off_policy_mask_threshold is None
        assert grpo.use_bias_correction_kl is False

    def test_model_control_defaults(self):
        grpo = GrpoConfig()
        assert grpo.disable_dropout is False
        assert grpo.cast_lm_head_to_fp32 is False

    def test_data_distributed_defaults(self):
        grpo = GrpoConfig()
        assert grpo.ds3_gather_for_generation is True
        assert grpo.shuffle_dataset is True

    def test_agentic_defaults(self):
        grpo = GrpoConfig()
        assert grpo.max_tool_calling_iterations is None

    def test_scale_rewards_accepts_bool(self):
        grpo = GrpoConfig(scale_rewards=True)
        assert grpo.scale_rewards is True

    def test_scale_rewards_accepts_string(self):
        grpo = GrpoConfig(scale_rewards="batch")
        assert grpo.scale_rewards == "batch"

    def test_scale_rewards_accepts_none_string(self):
        grpo = GrpoConfig(scale_rewards="none")
        assert grpo.scale_rewards == "none"

    def test_new_fields_set_via_yaml(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "training": {
                    "grpo": {
                        "min_p": 0.05,
                        "repetition_penalty": 1.2,
                        "delta": 1.5,
                        "sync_ref_model": True,
                        "scale_rewards": "batch",
                        "disable_dropout": True,
                        "shuffle_dataset": False,
                        "max_tool_calling_iterations": 3,
                    }
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        grpo = config.training.grpo
        assert grpo.min_p == 0.05
        assert grpo.repetition_penalty == 1.2
        assert grpo.delta == 1.5
        assert grpo.sync_ref_model is True
        assert grpo.scale_rewards == "batch"
        assert grpo.disable_dropout is True
        assert grpo.shuffle_dataset is False
        assert grpo.max_tool_calling_iterations == 3

    def test_min_p_boundary_validation(self):
        with pytest.raises(Exception):
            GrpoConfig(min_p=1.5)
        with pytest.raises(Exception):
            GrpoConfig(min_p=-0.1)

    def test_top_entropy_quantile_boundary(self):
        grpo = GrpoConfig(top_entropy_quantile=0.2)
        assert grpo.top_entropy_quantile == 0.2
        with pytest.raises(Exception):
            GrpoConfig(top_entropy_quantile=1.5)

    def test_generation_batch_size_must_be_positive(self):
        with pytest.raises(Exception):
            GrpoConfig(generation_batch_size=0)


class TestTrainingConfigNewFields:
    """Verify new TrainingConfig fields."""

    def test_weight_decay_default(self):
        from foresightrlcore.config.schema import TrainingConfig

        tc = TrainingConfig()
        assert tc.weight_decay == 0.0

    def test_fp16_default(self):
        from foresightrlcore.config.schema import TrainingConfig

        tc = TrainingConfig()
        assert tc.fp16 is False

    def test_weight_decay_set_via_yaml(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "training": {"weight_decay": 0.01, "fp16": True, "bf16": False},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        assert config.training.weight_decay == 0.01
        assert config.training.fp16 is True
        assert config.training.bf16 is False


class TestFilm5307PortFields:
    """FILM-5307 port: use_liger_kernel + vLLM reasoning/compilation_config."""

    def test_use_liger_kernel_default_false(self):
        from foresightrlcore.config.schema import TrainingConfig

        tc = TrainingConfig()
        assert tc.use_liger_kernel is False

    def test_use_liger_kernel_set_via_yaml(self, tmp_yaml):
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
        assert config.training.use_liger_kernel is True

    def test_vllm_reasoning_and_compilation_default_none(self):
        from foresightrlcore.config.schema import VllmConfig

        v = VllmConfig()
        assert v.reasoning_parser is None
        assert v.reasoning_config is None
        assert v.compilation_config is None
        # Default is False — opt-in only, since the workaround patches TRL's
        # vllm_serve.py source on disk and we don't want it active by default.
        assert v.disable_custom_all_reduce is False

    def test_film5307_fields_roundtrip(self, tmp_yaml, tmp_path):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "training": {"use_liger_kernel": True},
                "infra": {
                    "vllm_mode": "server",
                    "vllm": {
                        "reasoning_parser": "qwen3",
                        "reasoning_config": {
                            "reasoning_start_str": "<think>",
                            "reasoning_end_str": "</think>",
                        },
                        "compilation_config": {
                            "cudagraph_capture_sizes": [1, 2, 4],
                            "cudagraph_num_of_warmups": 1,
                        },
                        "disable_custom_all_reduce": True,
                    },
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        out = tmp_path / "roundtrip_film5307.yaml"
        config.to_yaml(out)
        config2 = TrainingJobConfig.from_yaml(out)
        assert config2.training.use_liger_kernel is True
        assert config2.infra.vllm.reasoning_parser == "qwen3"
        assert config2.infra.vllm.reasoning_config == {
            "reasoning_start_str": "<think>",
            "reasoning_end_str": "</think>",
        }
        assert config2.infra.vllm.compilation_config == {
            "cudagraph_capture_sizes": [1, 2, 4],
            "cudagraph_num_of_warmups": 1,
        }
        assert config2.infra.vllm.disable_custom_all_reduce is True


class TestNewFieldsRoundtrip:
    """New fields survive YAML serialization roundtrip."""

    def test_new_grpo_fields_roundtrip(self, tmp_yaml, tmp_path):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "training": {
                    "weight_decay": 0.01,
                    "grpo": {
                        "min_p": 0.05,
                        "repetition_penalty": 1.1,
                        "scale_rewards": "batch",
                        "multi_objective_aggregation": "normalize_then_sum",
                        "top_entropy_quantile": 0.2,
                        "disable_dropout": True,
                    },
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        out = tmp_path / "roundtrip_new.yaml"
        config.to_yaml(out)
        config2 = TrainingJobConfig.from_yaml(out)
        assert config2.training.weight_decay == 0.01
        assert config2.training.grpo.min_p == 0.05
        assert config2.training.grpo.repetition_penalty == 1.1
        assert config2.training.grpo.scale_rewards == "batch"
        assert config2.training.grpo.multi_objective_aggregation == "normalize_then_sum"
        assert config2.training.grpo.top_entropy_quantile == 0.2
        assert config2.training.grpo.disable_dropout is True


class TestBackwardCompatNewFields:
    """Old configs without new fields still parse with correct defaults."""

    def test_old_config_gets_new_defaults(self, minimal_config_yaml):
        config = TrainingJobConfig.from_yaml(minimal_config_yaml)
        assert config.training.weight_decay == 0.0
        assert config.training.fp16 is False
        assert config.training.hooks is None
        assert config.training.grpo.min_p is None
        assert config.training.grpo.repetition_penalty == 1.0
        assert config.training.grpo.scale_rewards is False
        assert config.training.grpo.sync_ref_model is False
        assert config.training.grpo.disable_dropout is False
        assert config.training.grpo.shuffle_dataset is True


class TestHooksConfigFields:
    """HooksConfig field defaults and validation."""

    def test_hooks_defaults(self):
        tc = TrainingConfig()
        assert tc.hooks is None

    def test_hooks_config_defaults(self):
        hooks = HooksConfig()
        assert hooks.trainer_class is None
        assert hooks.pre_train is None
        assert hooks.post_train is None
        assert hooks.callbacks is None

    def test_hooks_set_via_yaml(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "training": {
                    "hooks": {
                        "trainer_class": "my_pkg.CustomTrainer",
                        "callbacks": ["my_pkg.Cb1"],
                    }
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        assert config.training.hooks is not None
        assert config.training.hooks.trainer_class == "my_pkg.CustomTrainer"
        assert config.training.hooks.callbacks == ["my_pkg.Cb1"]


class TestModelConfigS3Source:
    """ModelConfig accepts s3:// URIs as a pre-staged source for the model."""

    def test_hf_id_is_not_s3(self):
        m = ModelConfig(name="Qwen/Qwen3.5-27B")
        assert m.is_s3_uri is False
        assert m.name == "Qwen/Qwen3.5-27B"

    def test_local_path_is_not_s3(self):
        m = ModelConfig(name="/opt/ml/input/data/model")
        assert m.is_s3_uri is False

    def test_s3_prefix_with_trailing_slash_is_accepted(self):
        m = ModelConfig(name="s3://bucket/models/Qwen3.6-27B/")
        assert m.is_s3_uri is True
        assert m.name == "s3://bucket/models/Qwen3.6-27B/"

    def test_s3_uri_without_trailing_slash_is_rejected(self):
        with pytest.raises(ValueError, match="prefix ending with"):
            ModelConfig(name="s3://bucket/models/Qwen3.6-27B")


class TestMixedDedicatedVllmSchema:
    """Opt-in mixed instance types on dedicated_vllm."""

    def test_defaults_are_homogeneous(self):
        infra = InfraConfig()
        assert infra.vllm_instance_type is None
        assert infra.vllm_instance_count == 1
        assert infra.allow_mixed_gpu_families is False
        assert infra.uses_mixed_instance_types() is False

    def test_opt_in_via_yaml(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "infra": {
                    "topology": "dedicated_vllm",
                    "instance_type": "ml.g6e.48xlarge",
                    "instance_count": 2,
                    "vllm_instance_type": "ml.g6e.12xlarge",
                    "vllm_instance_count": 1,
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        assert config.infra.uses_mixed_instance_types() is True
        assert config.infra.vllm_instance_type == "ml.g6e.12xlarge"
        assert config.infra.mixed_training_instance_count() == 1

    def test_homogeneous_dedicated_vllm_does_not_opt_in(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "infra": {
                    "topology": "dedicated_vllm",
                    "instance_type": "ml.p4d.24xlarge",
                    "instance_count": 2,
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        assert config.infra.vllm_instance_type is None
        assert config.infra.uses_mixed_instance_types() is False
