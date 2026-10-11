"""Tests for foresightrlcore.config.validation — capability matrix checks."""

from __future__ import annotations

import yaml

from foresightrlcore.config.schema import TrainingJobConfig
from foresightrlcore.config.validation import validate_config


class TestValidateConfig:
    """End-to-end config validation."""

    def test_valid_aws_config(self, full_config_yaml):
        config = TrainingJobConfig.from_yaml(full_config_yaml)
        errors = validate_config(config)
        assert errors == []

    def test_aws_requires_s3_data_path(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "data": {"train_path": "/local/path"},
                "output": {"s3_path": "s3://b/o"},
                "execution": {"type": "aws_training_job"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        errors = validate_config(config)
        assert any("s3://" in e for e in errors)

    def test_aws_requires_s3_output_path(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "/local/output"},
                "execution": {"type": "aws_training_job"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        errors = validate_config(config)
        assert any("s3://" in e for e in errors)

    def test_local_requires_gpus(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "data": {"train_path": "/data/train"},
                "output": {"s3_path": "/output"},
                "execution": {"type": "local"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        errors = validate_config(config)
        assert any("gpus" in e for e in errors)

    def test_local_with_gpus_passes(self, local_config_yaml):
        config = TrainingJobConfig.from_yaml(local_config_yaml)
        errors = validate_config(config)
        assert errors == []


class TestSpotValidation:
    """SageMaker Managed Spot Training validation."""

    @staticmethod
    def _spot_yaml(tmp_yaml, *, infra=None, job=None, output=None):
        base_infra = {"instance_type": "ml.p4d.24xlarge", "use_spot_instances": True}
        base_infra.update(infra or {})
        base_job = {"name": "spot-test", "max_runtime_hours": 6}
        base_job.update(job or {})
        base_output = {
            "s3_path": "s3://b/o",
            "checkpoint_s3_path": "s3://b/ckpt",
        }
        base_output.update(output or {})
        content = yaml.dump(
            {
                "job": base_job,
                "model": {"name": "m"},
                "infra": base_infra,
                "data": {"train_path": "s3://b/d"},
                "output": base_output,
                "execution": {"type": "aws_training_job"},
                "task": {"type": "advice", "config": {"reward_mode": "mock"}},
            }
        )
        return TrainingJobConfig.from_yaml(tmp_yaml(content))

    def test_spot_with_checkpoint_passes(self, tmp_yaml):
        config = self._spot_yaml(tmp_yaml)
        errors = validate_config(config)
        assert errors == []

    def test_spot_requires_checkpoint_s3_path(self, tmp_yaml):
        config = self._spot_yaml(tmp_yaml, output={"checkpoint_s3_path": None})
        errors = validate_config(config)
        assert any("checkpoint_s3_path" in e for e in errors)

    def test_spot_requires_s3_checkpoint_uri(self, tmp_yaml):
        config = self._spot_yaml(tmp_yaml, output={"checkpoint_s3_path": "/local/ckpt"})
        errors = validate_config(config)
        assert any("checkpoint_s3_path" in e for e in errors)

    def test_spot_max_wait_below_runtime_errors(self, tmp_yaml):
        config = self._spot_yaml(
            tmp_yaml, job={"max_runtime_hours": 10, "max_wait_hours": 6}
        )
        errors = validate_config(config)
        assert any("max_wait_hours" in e for e in errors)

    def test_spot_max_wait_equal_runtime_passes(self, tmp_yaml):
        config = self._spot_yaml(
            tmp_yaml, job={"max_runtime_hours": 6, "max_wait_hours": 6}
        )
        errors = validate_config(config)
        assert errors == []

    def test_spot_rejects_multi_node_topology(self, tmp_yaml):
        config = self._spot_yaml(
            tmp_yaml,
            infra={
                "topology": "multi_node",
                "multi_node": {
                    "vllm_instance_type": "ml.g5.12xlarge",
                    "training_instance_type": "ml.p4d.24xlarge",
                },
            },
        )
        errors = validate_config(config)
        assert any("multi_node" in e for e in errors)

    def test_no_spot_without_checkpoint_passes(self, tmp_yaml):
        config = self._spot_yaml(
            tmp_yaml,
            infra={"use_spot_instances": False},
            output={"checkpoint_s3_path": None},
        )
        errors = validate_config(config)
        assert errors == []


class TestHyperPodEksValidation:
    """`_validate_execution` branch for execution.type == 'aws_hyperpod_eks'.

    Mirrors the structure of the aws_training_job tests above. The backend's
    runtime `validate()` covers process-level concerns (kubectl available,
    boto3 importable); the config-level branch checks paths, image_uri,
    topology/vllm_mode compatibility.
    """

    def _make_config_yaml(self, overrides: dict | None = None) -> dict:
        base = {
            "job": {"name": "test"},
            "model": {"name": "Qwen/Qwen3-8B"},
            "method": {"type": "grpo"},
            "training": {"grpo": {"num_generations": 4}},
            "infra": {
                "instance_type": "ml.p4d.24xlarge",
                "instance_count": 2,
                "topology": "dedicated_vllm",
                "vllm_mode": "server",
            },
            "data": {"train_path": "s3://bucket/train.jsonl"},
            "output": {"s3_path": "s3://bucket/output"},
            "execution": {
                "type": "aws_hyperpod_eks",
                "hyperpod_eks": {
                    "image_uri": "123.dkr.ecr.us-east-1.amazonaws.com/x:latest",
                    "namespace": "default",
                },
            },
            "task": {"type": "advice"},
        }
        if overrides:
            for k, v in overrides.items():
                existing = base.get(k)
                if isinstance(v, dict) and isinstance(existing, dict):
                    existing.update(v)
                else:
                    base[k] = v
        return base

    def _validate(self, tmp_yaml, overrides: dict | None = None) -> list[str]:
        cfg = self._make_config_yaml(overrides)
        return validate_config(TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(cfg))))

    def test_happy_path_passes(self, tmp_yaml):
        errors = self._validate(tmp_yaml)
        assert errors == []

    def test_missing_hyperpod_eks_section(self, tmp_yaml):
        # Drop the whole hyperpod_eks block by building a fresh config
        # without it (the merge helper preserves nested keys, so we can't
        # just override "execution").
        cfg = self._make_config_yaml()
        cfg["execution"] = {"type": "aws_hyperpod_eks"}
        errors = validate_config(TrainingJobConfig.from_yaml(tmp_yaml(yaml.dump(cfg))))
        assert any("hyperpod_eks section is required" in e for e in errors)

    def test_missing_image_uri(self, tmp_yaml):
        errors = self._validate(
            tmp_yaml,
            {
                "execution": {
                    "type": "aws_hyperpod_eks",
                    "hyperpod_eks": {"image_uri": ""},
                }
            },
        )
        assert any("image_uri is required" in e for e in errors)

    def test_fsx_train_path_rejected(self, tmp_yaml):
        errors = self._validate(
            tmp_yaml, {"data": {"train_path": "/fsx/data/train.jsonl"}}
        )
        assert any("s3:// URI for data.train_path" in e for e in errors)

    def test_fsx_output_path_rejected(self, tmp_yaml):
        """Regression: matches the FILM-5272 PR review #4 bug.

        The advice example YAML had this exact shape; without this check the
        bug would only surface inside boto3.upload_file with bucket=''.
        """
        errors = self._validate(
            tmp_yaml,
            {"output": {"s3_path": "/fsx/output/grpo-advice-hyperpod"}},
        )
        assert any("s3:// URI for output.s3_path" in e for e in errors)

    def test_dedicated_vllm_with_single_node_fails(self, tmp_yaml):
        errors = self._validate(tmp_yaml, {"infra": {"instance_count": 1}})
        assert any(
            "dedicated_vllm topology requires instance_count" in e for e in errors
        )

    def test_mixed_instance_types_rejected(self, tmp_yaml):
        errors = self._validate(
            tmp_yaml,
            {
                "infra": {
                    "vllm_instance_type": "ml.g6e.12xlarge",
                    "instance_type": "ml.g6e.48xlarge",
                }
            },
        )
        assert any("aws_training_job" in e for e in errors)

    def test_vllm_mode_server_with_invalid_topology_fails(self, tmp_yaml):
        errors = self._validate(
            tmp_yaml,
            {
                "infra": {
                    "topology": "multi_node",
                    "vllm_mode": "server",
                    "instance_count": 2,
                }
            },
        )
        assert any("vllm_mode='server' is not supported" in e for e in errors)

    def test_single_node_with_colocate_passes(self, tmp_yaml):
        # Smaller-than-default shape — single node, colocate vLLM
        errors = self._validate(
            tmp_yaml,
            {
                "infra": {
                    "topology": "single_node",
                    "vllm_mode": "colocate",
                    "instance_count": 1,
                }
            },
        )
        assert errors == []

    def test_ssm_relay_without_kubeconfig_rejected(self, tmp_yaml):
        # Setting ssm_relay_instance_id alone is a silent footgun: the
        # launcher would fall through to whatever KUBECONFIG / ~/.kube/config
        # happens to be set, which is almost certainly the wrong cluster.
        # Require an explicit kubeconfig_path here.
        errors = self._validate(
            tmp_yaml,
            {
                "execution": {
                    "type": "aws_hyperpod_eks",
                    "hyperpod_eks": {
                        "image_uri": "123.dkr.ecr.us-east-1.amazonaws.com/x:latest",
                        "namespace": "default",
                        "ssm_relay_instance_id": "i-abc",
                        # kubeconfig_path intentionally omitted
                    },
                }
            },
        )
        assert any("kubeconfig_path is required" in e for e in errors)

    def test_ssm_relay_with_kubeconfig_passes(self, tmp_yaml):
        errors = self._validate(
            tmp_yaml,
            {
                "execution": {
                    "type": "aws_hyperpod_eks",
                    "hyperpod_eks": {
                        "image_uri": "123.dkr.ecr.us-east-1.amazonaws.com/x:latest",
                        "namespace": "default",
                        "ssm_relay_instance_id": "i-abc",
                        "kubeconfig_path": "kubeconfigs/your-cluster.yaml",
                    },
                }
            },
        )
        assert errors == []


class TestGrpoValidation:
    """GRPO-specific validation rules."""

    def _make_config(
        self,
        tmp_yaml,
        grpo_overrides: dict | None = None,
        training_overrides: dict | None = None,
    ) -> TrainingJobConfig:
        grpo_cfg: dict = {"num_generations": 8}
        if grpo_overrides:
            grpo_cfg.update(grpo_overrides)
        training_cfg: dict = {"grpo": grpo_cfg}
        if training_overrides:
            training_cfg.update(training_overrides)
        cfg = {
            "job": {"name": "test"},
            "model": {"name": "m"},
            "method": {"type": "grpo"},
            "training": training_cfg,
            "data": {"train_path": "s3://b/d"},
            "output": {"s3_path": "s3://b/o"},
            "task": {"type": "advice"},
        }
        content = yaml.dump(cfg)
        return TrainingJobConfig.from_yaml(tmp_yaml(content))

    def test_valid_loss_type(self, tmp_yaml):
        for lt in [
            "grpo",
            "dr_grpo",
            "dapo",
            "bnpo",
            "cispo",
            "sapo",
            "luspo",
            "vespo",
        ]:
            overrides = {"loss_type": lt}
            if lt == "luspo":
                overrides["importance_sampling_level"] = "sequence"
            config = self._make_config(tmp_yaml, grpo_overrides=overrides)
            errors = validate_config(config)
            assert not any("loss_type" in e for e in errors), f"loss_type={lt} failed"

    def test_invalid_loss_type(self, tmp_yaml):
        config = self._make_config(tmp_yaml, grpo_overrides={"loss_type": "invalid"})
        errors = validate_config(config)
        assert any("loss_type" in e for e in errors)

    def test_scale_rewards_valid_strings(self, tmp_yaml):
        for sr in ["group", "batch", "none"]:
            config = self._make_config(tmp_yaml, grpo_overrides={"scale_rewards": sr})
            errors = validate_config(config)
            assert not any("scale_rewards" in e for e in errors), f"scale_rewards={sr}"

    def test_scale_rewards_invalid_string(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml, grpo_overrides={"scale_rewards": "invalid"}
        )
        errors = validate_config(config)
        assert any("scale_rewards" in e for e in errors)

    def test_scale_rewards_bool_passes(self, tmp_yaml):
        config = self._make_config(tmp_yaml, grpo_overrides={"scale_rewards": True})
        errors = validate_config(config)
        assert not any("scale_rewards" in e for e in errors)

    def test_invalid_multi_objective_aggregation(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml, grpo_overrides={"multi_objective_aggregation": "bad"}
        )
        errors = validate_config(config)
        assert any("multi_objective_aggregation" in e for e in errors)

    def test_valid_multi_objective_aggregation(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml,
            grpo_overrides={"multi_objective_aggregation": "normalize_then_sum"},
        )
        errors = validate_config(config)
        assert not any("multi_objective_aggregation" in e for e in errors)

    def test_invalid_is_mode(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml,
            grpo_overrides={"vllm_importance_sampling_mode": "bad_mode"},
        )
        errors = validate_config(config)
        assert any("vllm_importance_sampling_mode" in e for e in errors)

    def test_valid_is_modes(self, tmp_yaml):
        for mode in [
            "token_truncate",
            "token_mask",
            "sequence_truncate",
            "sequence_mask",
        ]:
            config = self._make_config(
                tmp_yaml, grpo_overrides={"vllm_importance_sampling_mode": mode}
            )
            errors = validate_config(config)
            assert not any("vllm_importance_sampling_mode" in e for e in errors)

    def test_is_mode_none_passes(self, tmp_yaml):
        config = self._make_config(tmp_yaml)
        errors = validate_config(config)
        assert not any("vllm_importance_sampling_mode" in e for e in errors)

    def test_sync_ref_model_requires_beta(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml, grpo_overrides={"sync_ref_model": True, "beta": 0.0}
        )
        errors = validate_config(config)
        assert any("sync_ref_model" in e for e in errors)

    def test_sync_ref_model_with_beta_passes(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml, grpo_overrides={"sync_ref_model": True, "beta": 0.001}
        )
        errors = validate_config(config)
        assert not any("sync_ref_model" in e for e in errors)

    def test_luspo_requires_sequence_level(self, tmp_yaml):
        config = self._make_config(tmp_yaml, grpo_overrides={"loss_type": "luspo"})
        errors = validate_config(config)
        assert any("luspo" in e for e in errors)

    def test_luspo_with_sequence_passes(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml,
            grpo_overrides={
                "loss_type": "luspo",
                "importance_sampling_level": "sequence",
            },
        )
        errors = validate_config(config)
        assert not any("luspo" in e for e in errors)

    def test_bf16_fp16_conflict(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml, training_overrides={"bf16": True, "fp16": True}
        )
        errors = validate_config(config)
        assert any("bf16" in e and "fp16" in e for e in errors)

    def test_non_grpo_skips_grpo_validation(self, tmp_yaml):
        cfg = {
            "job": {"name": "test"},
            "model": {"name": "m"},
            "method": {"type": "sft"},
            "training": {"grpo": {"loss_type": "totally_invalid"}},
            "data": {"train_path": "s3://b/d"},
            "output": {"s3_path": "s3://b/o"},
            "task": {"type": "advice"},
        }
        content = yaml.dump(cfg)
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        errors = validate_config(config)
        assert not any("loss_type" in e for e in errors)


class TestContainerValidation:
    """Container recipe conflict warnings."""

    def test_no_container_warnings_by_default(self, full_config_yaml):
        config = TrainingJobConfig.from_yaml(full_config_yaml)
        errors = validate_config(config)
        assert errors == []

    def test_recipe_with_image_uri_warns(self, tmp_yaml, caplog):
        import logging

        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "container": {
                    "recipe": "qwen3",
                    "image_uri": "123456.dkr.ecr.us-west-2.amazonaws.com/img:tag",
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        with caplog.at_level(logging.WARNING):
            validate_config(config)
        assert any("image_uri" in r.message for r in caplog.records)

    def test_recipe_with_inline_requirements_warns(self, tmp_yaml, caplog):
        import logging

        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "container": {
                    "recipe": "qwen3",
                    "requirements": ["trl>=2.0.0"],
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        with caplog.at_level(logging.WARNING):
            validate_config(config)
        assert any("requirements" in r.message for r in caplog.records)

    def test_recipe_with_overridden_framework_version_warns(self, tmp_yaml, caplog):
        import logging

        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "container": {
                    "recipe": "qwen3",
                    "framework_version": "2.7.0",
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        with caplog.at_level(logging.WARNING):
            validate_config(config)
        assert any("framework_version" in r.message for r in caplog.records)


class TestFilm5307VllmFieldWarnings:
    """FILM-5307 port: warn when reasoning_parser/compilation_config set with non-server vllm_mode."""

    def test_reasoning_parser_warns_when_not_server(self, tmp_yaml, caplog):
        import logging

        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "infra": {
                    "vllm_mode": "colocate",
                    "vllm": {
                        "reasoning_parser": "qwen3",
                        "reasoning_config": {
                            "reasoning_start_str": "<think>",
                            "reasoning_end_str": "</think>",
                        },
                    },
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        with caplog.at_level(logging.WARNING):
            errors = validate_config(config)
        assert errors == []
        assert any("reasoning_parser" in r.message for r in caplog.records)

    def test_compilation_config_warns_when_not_server(self, tmp_yaml, caplog):
        import logging

        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "infra": {
                    "vllm_mode": "none",
                    "vllm": {
                        "compilation_config": {"cudagraph_capture_sizes": [1, 2, 4]},
                    },
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        with caplog.at_level(logging.WARNING):
            errors = validate_config(config)
        assert errors == []
        assert any("compilation_config" in r.message for r in caplog.records)

    def test_no_warning_when_server_mode(self, tmp_yaml, caplog):
        import logging

        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "infra": {
                    "vllm_mode": "server",
                    "vllm": {
                        "reasoning_parser": "qwen3",
                        "reasoning_config": {
                            "reasoning_start_str": "<think>",
                            "reasoning_end_str": "</think>",
                        },
                        "compilation_config": {"cudagraph_capture_sizes": [1, 2, 4]},
                        "disable_custom_all_reduce": True,
                    },
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        with caplog.at_level(logging.WARNING):
            validate_config(config)
        assert not any(
            (
                "reasoning_parser" in r.message
                or "compilation_config" in r.message
                or "disable_custom_all_reduce" in r.message
            )
            for r in caplog.records
        )

    def test_disable_custom_all_reduce_warns_when_not_server(self, tmp_yaml, caplog):
        """The flag is only honored in server mode (patches TRL's vllm_serve)."""
        import logging

        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "infra": {
                    "vllm_mode": "colocate",
                    "vllm": {
                        "disable_custom_all_reduce": True,
                    },
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "task": {"type": "advice"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        with caplog.at_level(logging.WARNING):
            errors = validate_config(config)
        assert errors == []
        assert any("disable_custom_all_reduce" in r.message for r in caplog.records)


class TestMixedDedicatedVllmValidation:
    """Opt-in mixed instance types on dedicated_vllm."""

    @staticmethod
    def _yaml(tmp_yaml, *, infra=None, execution=None):
        base_infra = {
            "topology": "dedicated_vllm",
            "instance_type": "ml.g6e.48xlarge",
            "instance_count": 2,
            "vllm_instance_type": "ml.g6e.12xlarge",
            "vllm": {"tensor_parallel_size": 4},
        }
        base_infra.update(infra or {})
        base_execution = {"type": "aws_training_job"}
        if execution:
            base_execution.update(execution)
        content = yaml.dump(
            {
                "job": {"name": "mixed-test"},
                "model": {"name": "m"},
                "infra": base_infra,
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
                "execution": base_execution,
                "task": {"type": "advice", "config": {"reward_mode": "mock"}},
            }
        )
        return TrainingJobConfig.from_yaml(tmp_yaml(content))

    def test_same_family_passes(self, tmp_yaml):
        config = self._yaml(tmp_yaml)
        assert validate_config(config) == []

    def test_homogeneous_dedicated_vllm_still_passes(self, tmp_yaml):
        config = self._yaml(
            tmp_yaml,
            infra={
                "vllm_instance_type": None,
                "instance_type": "ml.p4d.24xlarge",
                "vllm": {"tensor_parallel_size": 8},
            },
        )
        assert config.infra.uses_mixed_instance_types() is False
        assert validate_config(config) == []

    def test_rejects_wrong_topology(self, tmp_yaml):
        config = self._yaml(tmp_yaml, infra={"topology": "single_node"})
        errors = validate_config(config)
        assert any("only valid with topology: dedicated_vllm" in e for e in errors)

    def test_rejects_local_backend(self, tmp_yaml):
        config = self._yaml(
            tmp_yaml,
            execution={"type": "local", "local": {"gpus": [0, 1]}},
        )
        # local also needs non-s3 paths; we only assert the mixed-type error
        errors = validate_config(config)
        assert any("aws_training_job" in e for e in errors)

    def test_rejects_family_mismatch(self, tmp_yaml):
        config = self._yaml(
            tmp_yaml,
            infra={
                "instance_type": "ml.p4d.24xlarge",
                "vllm_instance_type": "ml.g5.12xlarge",
            },
        )
        errors = validate_config(config)
        assert any("allow_mixed_gpu_families" in e for e in errors)
        assert any("g5" in e and "p4d" in e for e in errors)

    def test_allow_mixed_gpu_families_override(self, tmp_yaml):
        config = self._yaml(
            tmp_yaml,
            infra={
                "instance_type": "ml.p4d.24xlarge",
                "vllm_instance_type": "ml.g5.12xlarge",
                "allow_mixed_gpu_families": True,
            },
        )
        assert validate_config(config) == []

    def test_rejects_efa_mismatch_same_family(self, tmp_yaml):
        config = self._yaml(
            tmp_yaml,
            infra={
                "instance_type": "ml.g6e.48xlarge",
                "vllm_instance_type": "ml.g6e.xlarge",
                "vllm": {"tensor_parallel_size": 1},
            },
        )
        errors = validate_config(config)
        assert any("EFA" in e for e in errors)

    def test_rejects_tp_exceeding_vllm_gpus(self, tmp_yaml):
        config = self._yaml(
            tmp_yaml,
            infra={"vllm": {"tensor_parallel_size": 8}},
        )
        errors = validate_config(config)
        assert any("tensor_parallel_size" in e for e in errors)

    def test_rejects_vllm_instance_count_not_one(self, tmp_yaml):
        config = self._yaml(
            tmp_yaml,
            infra={"instance_count": 3, "vllm_instance_count": 2},
        )
        errors = validate_config(config)
        assert any("vllm_instance_count=1" in e for e in errors)

    def test_rejects_zero_training_nodes(self, tmp_yaml):
        config = self._yaml(
            tmp_yaml,
            infra={"instance_count": 1, "vllm_instance_count": 1},
        )
        errors = validate_config(config)
        assert any("at least one training" in e for e in errors)

    def test_rejects_keep_alive(self, tmp_yaml):
        config = self._yaml(tmp_yaml, infra={"keep_alive_seconds": 1800})
        errors = validate_config(config)
        assert any("keep_alive_seconds" in e for e in errors)

    def test_rejects_efa_types_when_vpc_enabled(self, tmp_yaml):
        config = self._yaml(
            tmp_yaml,
            infra={
                "vpc": {
                    "subnets": ["subnet-abc"],
                    "security_groups": ["sg-abc"],
                }
            },
        )
        errors = validate_config(config)
        assert any("non-EFA instance types when VPC is enabled" in e for e in errors)

    def test_g5_pair_passes_with_vpc(self, tmp_yaml):
        config = self._yaml(
            tmp_yaml,
            infra={
                "instance_type": "ml.g5.4xlarge",
                "vllm_instance_type": "ml.g5.2xlarge",
                "vllm": {"tensor_parallel_size": 1},
                "vpc": {
                    "subnets": ["subnet-abc"],
                    "security_groups": ["sg-abc"],
                },
            },
        )
        assert validate_config(config) == []

    def test_rejects_spot(self, tmp_yaml):
        content = yaml.dump(
            {
                "job": {"name": "mixed-spot"},
                "model": {"name": "m"},
                "infra": {
                    "topology": "dedicated_vllm",
                    "instance_type": "ml.g6e.48xlarge",
                    "instance_count": 2,
                    "vllm_instance_type": "ml.g6e.12xlarge",
                    "use_spot_instances": True,
                    "vllm": {"tensor_parallel_size": 4},
                },
                "data": {"train_path": "s3://b/d"},
                "output": {
                    "s3_path": "s3://b/o",
                    "checkpoint_s3_path": "s3://b/ckpt",
                },
                "execution": {"type": "aws_training_job"},
                "task": {"type": "advice", "config": {"reward_mode": "mock"}},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        errors = validate_config(config)
        assert any("use_spot_instances" in e for e in errors)


class TestSftVerlValidation:
    """sft_verl-specific validation rules (_validate_sft_verl + the

    sft_verl branch of _resolve_data_points). These checks run inside
    validate_config() — the same path both `rl-train validate` and
    `rl-train train` call — not just inside VerlSftMethodHandler.validate()
    (which nothing in the real submit path actually calls).
    """

    def _make_config(
        self,
        tmp_yaml,
        sft_verl_overrides: dict | None = None,
        model_overrides: dict | None = None,
        infra_overrides: dict | None = None,
        training_overrides: dict | None = None,
        data_overrides: dict | None = None,
        include_data: bool = True,
        include_checkpoint_s3_path: bool = True,
    ) -> TrainingJobConfig:
        sft_verl_cfg: dict = {}
        if sft_verl_overrides:
            sft_verl_cfg.update(sft_verl_overrides)
        training_cfg: dict = {"sft_verl": sft_verl_cfg}
        if training_overrides:
            training_cfg.update(training_overrides)
        model_cfg: dict = {"name": "m"}
        if model_overrides:
            model_cfg.update(model_overrides)
        infra_cfg: dict = {
            "vllm_mode": "none",
            "instance_type": "ml.p4de.24xlarge",
            "instance_count": 1,
        }
        if infra_overrides:
            infra_cfg.update(infra_overrides)
        output_cfg: dict = {"s3_path": "s3://b/o"}
        if include_checkpoint_s3_path:
            # verl only ever writes checkpoints to output.checkpoint_s3_path
            # (via SageMaker's CheckpointConfig sync) — required for a
            # normal training run, matching every shipped example config.
            output_cfg["checkpoint_s3_path"] = "s3://b/checkpoints"
        cfg: dict = {
            "job": {"name": "test"},
            "model": model_cfg,
            "method": {"type": "sft_verl"},
            "execution": {"type": "aws_training_job"},
            "infra": infra_cfg,
            "training": training_cfg,
            "output": output_cfg,
        }
        if include_data:
            data_cfg: dict = {"train_path": "s3://b/d"}
            if data_overrides:
                data_cfg.update(data_overrides)
            cfg["data"] = data_cfg
        content = yaml.dump(cfg)
        return TrainingJobConfig.from_yaml(tmp_yaml(content))

    def test_valid_config_passes(self, tmp_yaml):
        config = self._make_config(tmp_yaml)
        assert validate_config(config) == []

    def test_missing_sft_verl_section_is_a_noop_here(self, tmp_yaml):
        """_validate_sft_verl itself contributes no error when

        training.sft_verl is absent — _validate_method_config_alignment is
        the function responsible for flagging that as missing.
        """
        content = yaml.dump(
            {
                "job": {"name": "test"},
                "model": {"name": "m"},
                "method": {"type": "sft_verl"},
                "execution": {"type": "aws_training_job"},
                "infra": {
                    "vllm_mode": "none",
                    "instance_type": "ml.p4de.24xlarge",
                    "instance_count": 1,
                },
                "data": {"train_path": "s3://b/d"},
                "output": {"s3_path": "s3://b/o"},
            }
        )
        config = TrainingJobConfig.from_yaml(tmp_yaml(content))
        errors = validate_config(config)
        assert any("training.sft_verl" in e for e in errors)

    def test_requires_vllm_mode_none(self, tmp_yaml):
        config = self._make_config(tmp_yaml, infra_overrides={"vllm_mode": "server"})
        errors = validate_config(config)
        assert any("vllm_mode: none" in e for e in errors)

    def test_ulysses_requires_remove_padding(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml,
            sft_verl_overrides={
                "ulysses_sequence_parallel_size": 4,
                "use_remove_padding": False,
            },
        )
        errors = validate_config(config)
        assert any("ulysses_sequence_parallel_size" in e for e in errors)

    def test_target_parameters_with_nonzero_dropout_rejected(self, tmp_yaml):
        """PEFT's ParamWrapper (dispatched for every target_parameters

        entry) unconditionally rejects nonzero lora_dropout at construction
        time — reproduced directly on a real SageMaker job
        (sft-verl-gemma4-26ba4b-sft-verl-smoke-packed-20260929153449).
        """
        config = self._make_config(
            tmp_yaml,
            model_overrides={
                "lora": {
                    "target_parameters": ["experts.gate_up_proj"],
                    "dropout": 0.05,
                }
            },
        )
        errors = validate_config(config)
        assert any(
            "model.lora.dropout" in e and "target_parameters" in e for e in errors
        )

    def test_target_parameters_with_zero_dropout_is_valid(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml,
            model_overrides={
                "lora": {
                    "target_parameters": ["experts.gate_up_proj"],
                    "dropout": 0.0,
                }
            },
        )
        assert validate_config(config) == []

    def test_nonzero_dropout_without_target_parameters_is_valid(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml,
            model_overrides={"lora": {"dropout": 0.05}},
        )
        assert validate_config(config) == []

    def test_target_parameters_with_nonzero_dropout_exempt_for_merge_lora_only(
        self, tmp_yaml
    ):
        """merge_lora_only's own LoraConfig build

        (verl_checkpoint_merge.py's _build_lora_config) never passes
        lora_dropout at all, so PEFT's own LoraConfig default (0.0) always
        applies there regardless of model.lora.dropout — this combination
        must not be flagged for a merge job.
        """
        config = self._make_config(
            tmp_yaml,
            model_overrides={
                "lora": {
                    "target_parameters": ["experts.gate_up_proj"],
                    "dropout": 0.05,
                }
            },
            sft_verl_overrides={"merge_lora_only": True},
            include_data=False,
        )
        assert validate_config(config) == []

    def test_messages_format_same_train_eval_path_rejected(self, tmp_yaml):
        """Regression guard: unlike data_format='packed' (which has a

        training_*/validation_* filename split), 'flat'/'messages' glob
        train_path/eval_path with no disambiguation at all — identical
        paths would silently train and evaluate on the exact same files
        with no crash and no warning otherwise.
        """
        config = self._make_config(
            tmp_yaml,
            sft_verl_overrides={"data_format": "messages"},
            data_overrides={"train_path": "s3://b/d", "eval_path": "s3://b/d"},
        )
        errors = validate_config(config)
        assert any(
            "train_path" in e and "eval_path" in e and "same path" in e for e in errors
        )

    def test_messages_format_same_train_eval_path_trailing_slash_rejected(
        self, tmp_yaml
    ):
        """ "s3://b/d" and "s3://b/d/" are the same prefix — neither

        DataConfig nor this check's comparison should be defeated by a
        superficial trailing-slash difference.
        """
        config = self._make_config(
            tmp_yaml,
            sft_verl_overrides={"data_format": "messages"},
            data_overrides={"train_path": "s3://b/d", "eval_path": "s3://b/d/"},
        )
        errors = validate_config(config)
        assert any(
            "train_path" in e and "eval_path" in e and "same path" in e for e in errors
        )

    def test_flat_format_same_train_eval_path_rejected(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml,
            sft_verl_overrides={"data_format": "flat"},
            data_overrides={"train_path": "s3://b/d", "eval_path": "s3://b/d"},
        )
        errors = validate_config(config)
        assert any("train_path" in e and "eval_path" in e for e in errors)

    def test_messages_format_different_train_eval_path_is_valid(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml,
            sft_verl_overrides={"data_format": "messages"},
            data_overrides={"train_path": "s3://b/train/", "eval_path": "s3://b/eval/"},
        )
        assert validate_config(config) == []

    def test_messages_format_no_eval_path_is_valid(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml,
            sft_verl_overrides={"data_format": "messages"},
        )
        assert validate_config(config) == []

    def test_packed_format_same_train_eval_path_is_valid(self, tmp_yaml):
        """packed's training_*/validation_* filename split makes a shared

        prefix safe — confirms the new check doesn't overreach into the
        format it was explicitly scoped to exclude.
        """
        config = self._make_config(
            tmp_yaml,
            sft_verl_overrides={"data_format": "packed"},
            data_overrides={"train_path": "s3://b/d", "eval_path": "s3://b/d"},
        )
        assert validate_config(config) == []

    def test_ulysses_one_does_not_require_remove_padding(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml,
            sft_verl_overrides={
                "ulysses_sequence_parallel_size": 1,
                "use_remove_padding": False,
            },
        )
        errors = validate_config(config)
        assert not any("ulysses_sequence_parallel_size" in e for e in errors)

    def test_remove_padding_requires_flash_attention_2(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml,
            model_overrides={"attn_implementation": "sdpa"},
            sft_verl_overrides={"use_remove_padding": True},
        )
        errors = validate_config(config)
        assert any("attn_implementation" in e for e in errors)

    def test_remove_padding_allowed_with_flash_attention_2(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml,
            model_overrides={"attn_implementation": "flash_attention_2"},
            sft_verl_overrides={"use_remove_padding": True},
        )
        errors = validate_config(config)
        assert not any("attn_implementation" in e for e in errors)

    def test_remove_padding_allowed_when_attn_implementation_unset(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml,
            sft_verl_overrides={"use_remove_padding": True},
        )
        errors = validate_config(config)
        assert not any("attn_implementation" in e for e in errors)

    def test_packed_data_format_requires_no_remove_padding(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml,
            sft_verl_overrides={"data_format": "packed", "use_remove_padding": True},
        )
        errors = validate_config(config)
        assert any("data_format='packed'" in e for e in errors)

    def test_packed_data_format_with_remove_padding_false_is_valid(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml,
            sft_verl_overrides={"data_format": "packed", "use_remove_padding": False},
        )
        errors = validate_config(config)
        assert not any("data_format='packed'" in e for e in errors)

    def test_invalid_data_format(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml, sft_verl_overrides={"data_format": "bogus"}
        )
        errors = validate_config(config)
        assert any("data_format" in e for e in errors)

    def test_invalid_fsdp_strategy(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml, sft_verl_overrides={"fsdp_strategy": "bogus"}
        )
        errors = validate_config(config)
        assert any("fsdp_strategy" in e for e in errors)

    def test_valid_fsdp_strategies(self, tmp_yaml):
        for strategy in ("fsdp", "fsdp2"):
            config = self._make_config(
                tmp_yaml, sft_verl_overrides={"fsdp_strategy": strategy}
            )
            errors = validate_config(config)
            assert not any(
                "fsdp_strategy" in e for e in errors
            ), f"fsdp_strategy={strategy}"

    def test_invalid_lr_scheduler_rejected(self, tmp_yaml):
        # verl's FSDPOptimizerConfig hard-asserts lr_scheduler_type in
        # ["constant", "cosine"] — run() silently maps anything else to
        # "constant", so it must be caught here instead.
        config = self._make_config(
            tmp_yaml, training_overrides={"lr_scheduler": "linear"}
        )
        errors = validate_config(config)
        assert any("lr_scheduler" in e for e in errors)

    def test_valid_lr_schedulers_accepted(self, tmp_yaml):
        for scheduler in ("cosine", "constant"):
            config = self._make_config(
                tmp_yaml, training_overrides={"lr_scheduler": scheduler}
            )
            errors = validate_config(config)
            assert not any(
                "lr_scheduler" in e for e in errors
            ), f"lr_scheduler={scheduler}"

    def test_unrecognized_gpu_family_rejected(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml, infra_overrides={"instance_type": "ml.t2.micro"}
        )
        errors = validate_config(config)
        assert any("Unrecognized GPU family" in e for e in errors)

    def test_train_batch_size_smaller_than_world_size_rejected(self, tmp_yaml):
        """Reproduces a real SageMaker failure directly: verl's

        train_batch_size // dp_size (plain integer division) floored to 0
        on an 8-GPU single-node config, and DataLoader(batch_size=0) raised
        at trainer setup — after the cluster was already billed.
        """
        config = self._make_config(
            tmp_yaml,
            sft_verl_overrides={"train_batch_size": 4},
            infra_overrides={"instance_type": "ml.p4d.24xlarge", "instance_count": 1},
        )
        errors = validate_config(config)
        assert any("train_batch_size" in e and "must be >=" in e for e in errors)

    def test_train_batch_size_not_a_multiple_of_dp_size_rejected(self, tmp_yaml):
        """train_batch_size >= dp_size but not an exact multiple — verl's

        integer division would silently truncate the global batch rather
        than crash, which is exactly the kind of silent-correctness issue
        this codebase's other sft_verl checks fail loudly on instead.
        """
        config = self._make_config(
            tmp_yaml,
            sft_verl_overrides={"train_batch_size": 10},
            infra_overrides={"instance_type": "ml.p4d.24xlarge", "instance_count": 1},
        )
        errors = validate_config(config)
        assert any("train_batch_size" in e and "not a multiple" in e for e in errors)

    def test_train_batch_size_exact_multiple_of_dp_size_is_valid(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml,
            sft_verl_overrides={"train_batch_size": 16},
            infra_overrides={"instance_type": "ml.p4d.24xlarge", "instance_count": 1},
        )
        assert validate_config(config) == []

    def test_ulysses_size_exceeding_world_size_rejected(self, tmp_yaml):
        """ulysses_sequence_parallel_size > total GPU count would floor

        dp_size itself to 0 (world_size // ulysses_sequence_parallel_size),
        an invalid FSDP topology regardless of train_batch_size — caught
        before the train_batch_size/dp_size comparison even runs.
        """
        config = self._make_config(
            tmp_yaml,
            sft_verl_overrides={
                "ulysses_sequence_parallel_size": 16,
                "use_remove_padding": True,
            },
            infra_overrides={"instance_type": "ml.p4d.24xlarge", "instance_count": 1},
        )
        errors = validate_config(config)
        assert any(
            "ulysses_sequence_parallel_size" in e and "exceeds" in e for e in errors
        )

    def test_train_batch_size_check_accounts_for_multi_node_world_size(self, tmp_yaml):
        """world_size = instance_count x GPUs/instance — a batch size too

        small for a single node can still be valid once instance_count
        scales dp_size up (8 GPUs/node x 2 nodes = 16).
        """
        config = self._make_config(
            tmp_yaml,
            sft_verl_overrides={"train_batch_size": 16},
            infra_overrides={"instance_type": "ml.p4d.24xlarge", "instance_count": 2},
        )
        assert validate_config(config) == []

    def test_merge_lora_only_requires_checkpoint_s3_path(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml,
            sft_verl_overrides={"merge_lora_only": True},
            include_data=False,
            include_checkpoint_s3_path=False,
        )
        errors = validate_config(config)
        assert any("checkpoint_s3_path" in e for e in errors)

    def test_merge_lora_only_with_checkpoint_s3_path_is_valid(self, tmp_yaml):
        config = self._make_config(
            tmp_yaml,
            sft_verl_overrides={"merge_lora_only": True},
            include_data=False,
            include_checkpoint_s3_path=False,
        )
        # merge_lora_only reads from output.checkpoint_s3_path and never
        # touches training data at all, so data.train_path is intentionally
        # absent here — see _validate_data_required's merge_lora_only
        # exemption.
        config.output.checkpoint_s3_path = "s3://b/checkpoints/step_10"
        errors = validate_config(config)
        assert errors == []

    def test_merge_lora_only_skips_attention_correctness_checks(self, tmp_yaml):
        """merge_lora_only never touches the dataset/attention path at all,

        so a config that would otherwise trip the packed/use_remove_padding
        or attn_implementation checks must not error when merge_lora_only
        is set.
        """
        config = self._make_config(
            tmp_yaml,
            model_overrides={"attn_implementation": "sdpa"},
            sft_verl_overrides={
                "merge_lora_only": True,
                "data_format": "packed",
                "use_remove_padding": True,
            },
            include_data=False,
        )
        config.output.checkpoint_s3_path = "s3://b/checkpoints/step_10"
        errors = validate_config(config)
        assert errors == []

    def test_missing_checkpoint_s3_path_rejected_for_training(self, tmp_yaml):
        """verl only ever writes checkpoints to output.checkpoint_s3_path

        (via SageMaker's CheckpointConfig sync) — without it, a training
        job "succeeds" with zero durable output anywhere.
        """
        config = self._make_config(tmp_yaml, include_checkpoint_s3_path=False)
        errors = validate_config(config)
        assert any("output.checkpoint_s3_path" in e for e in errors)

    def test_data_points_is_a_hard_error_for_sft_verl(self, tmp_yaml):
        config = self._make_config(tmp_yaml, training_overrides={"data_points": 1000})
        errors = validate_config(config)
        assert any("training.data_points is not supported" in e for e in errors)
