"""OptimizerJobConfig schema behavior + validate_optimizer_config rules."""

from __future__ import annotations

import textwrap

import pytest

# Registry side-effect import so is_llm_judge / build_judge resolve.
import foresightrlcore.optimizer.judges  # noqa: F401
from foresightrlcore.config.optimizer_schema import (
    DEFAULT_CPU_INSTANCE_TYPE,
    DEFAULT_GPU_INSTANCE_TYPE,
    OptimizerJobConfig,
)
from foresightrlcore.config.optimizer_validation import validate_optimizer_config


def _base_dict(**overrides):
    cfg = {
        "job": {"name": "opt"},
        "optimizer": {
            "datasets": [{"name": "bts_parquet", "path": "s3://b/data.parquet"}],
            "prompt_set": "bts_v1",
            "prompts_path": "s3://b/prompts/bts_v1_prompts.yaml",
            "models": [{"name": "gemini", "provider": "genos"}],
            "modes": ["summary_state"],
            "judges": [{"name": "length", "weight": 0}],
        },
        "execution": {"type": "aws_training_job"},
        "output": {"s3_path": "s3://b/out/"},
        "container": {"recipe": "qwen3_5"},
    }
    cfg.update(overrides)
    return cfg


class TestResume:
    """Checkpointing must actually persist, not merely be switched on.

    `checkpoint.enabled` defaults to true and the runner honours it, but the
    rows only outlive the job when a checkpoint S3 prefix exists for the
    launcher to attach a SageMaker CheckpointConfig to. Leaving that to the
    operator produced runs that looked configured, wrote checkpoints to
    container disk, and could not resume -- with nothing said about it.
    """

    def test_a_remote_run_can_resume_without_being_asked_to(self):
        config = OptimizerJobConfig.model_validate(_base_dict())
        assert config.output.checkpoint_s3_path == "s3://b/out/checkpoints/"

    def test_an_explicit_checkpoint_location_is_left_alone(self):
        config = OptimizerJobConfig.model_validate(
            _base_dict(
                output={"s3_path": "s3://b/out/", "checkpoint_s3_path": "s3://x/c/"}
            )
        )
        assert config.output.checkpoint_s3_path == "s3://x/c/"

    def test_disabling_checkpointing_does_not_quietly_re_enable_it(self):
        d = _base_dict()
        d["optimizer"]["checkpoint"] = {"enabled": False}
        config = OptimizerJobConfig.model_validate(d)
        assert config.output.checkpoint_s3_path is None

    def test_a_local_run_gets_no_s3_checkpoint_prefix(self):
        config = OptimizerJobConfig.model_validate(
            _base_dict(execution={"type": "local"})
        )
        assert config.output.checkpoint_s3_path is None

    def test_validation_reports_where_checkpoints_are_kept(self, caplog):
        import logging

        with caplog.at_level(logging.INFO):
            validate_optimizer_config(OptimizerJobConfig.model_validate(_base_dict()))

        assert "s3://b/out/checkpoints/" in caplog.text

    def test_validation_warns_that_a_run_without_checkpointing_cannot_resume(
        self, caplog
    ):
        import logging

        d = _base_dict()
        d["optimizer"]["checkpoint"] = {"enabled": False}

        with caplog.at_level(logging.WARNING):
            validate_optimizer_config(OptimizerJobConfig.model_validate(d))

        assert "cannot resume" in caplog.text


class TestSchema:
    @pytest.mark.parametrize("provider", ["bedrock", "genos"])
    def test_candidate_temperatures_survive_yaml_roundtrip(self, tmp_path, provider):
        data = _base_dict()
        data["optimizer"]["models"] = [
            {"name": "omitted", "provider": provider, "temperature": None},
            {"name": "zero", "provider": provider, "temperature": 0.0},
            {"name": "sampled", "provider": provider, "temperature": 0.7},
            {"name": "default", "provider": provider},
        ]
        config = OptimizerJobConfig.model_validate(data)
        path = tmp_path / "optimizer.yaml"

        config.to_yaml(path)
        restored = OptimizerJobConfig.from_yaml(path)

        assert [model.temperature for model in restored.optimizer.models] == [
            None,
            0.0,
            0.7,
            0.0,
        ]
        assert restored.model_dump() == config.model_dump()

    @pytest.mark.parametrize("temperature", [-0.1, 2.1])
    def test_candidate_temperature_bounds_still_apply(self, temperature):
        data = _base_dict()
        data["optimizer"]["models"][0]["temperature"] = temperature
        with pytest.raises(ValueError, match="temperature"):
            OptimizerJobConfig.model_validate(data)

    def test_defaults_api_only_is_cpu(self):
        config = OptimizerJobConfig.model_validate(_base_dict())
        assert config.requires_gpu() is False
        assert config.resolved_instance_type() == DEFAULT_CPU_INSTANCE_TYPE

    def test_local_provider_requires_gpu(self):
        d = _base_dict()
        d["optimizer"]["models"] = [{"name": "Qwen/Qwen3.5-0.8B", "provider": "local"}]
        config = OptimizerJobConfig.model_validate(d)
        assert config.requires_gpu() is True
        assert config.resolved_instance_type() == DEFAULT_GPU_INSTANCE_TYPE

    def test_explicit_instance_type_wins(self):
        d = _base_dict()
        d["optimizer"]["instance_type"] = "ml.c5.4xlarge"
        config = OptimizerJobConfig.model_validate(d)
        assert config.resolved_instance_type() == "ml.c5.4xlarge"

    def test_local_provider_gets_default_serving(self):
        d = _base_dict()
        d["optimizer"]["models"] = [{"name": "m", "provider": "local"}]
        config = OptimizerJobConfig.model_validate(d)
        assert config.optimizer.models[0].serving is not None

    def test_model_id_defaults_to_name(self):
        config = OptimizerJobConfig.model_validate(_base_dict())
        assert config.optimizer.models[0].resolved_model_id == "gemini"

    def test_invalid_provider_rejected(self):
        d = _base_dict()
        d["optimizer"]["models"] = [{"name": "x", "provider": "openai"}]
        with pytest.raises(ValueError):
            OptimizerJobConfig.model_validate(d)

    def test_unsupported_backend_rejected(self):
        d = _base_dict()
        d["execution"] = {"type": "aws_serverless"}
        with pytest.raises(ValueError):
            OptimizerJobConfig.model_validate(d)

    def test_uses_llm_judges_detection(self):
        d = _base_dict()
        d["optimizer"]["judges"] = [{"name": "correctness", "weight": 1.0}]
        d["optimizer"]["judge_clients"] = {
            "default": {"provider": "genos", "genos": {"model_id": "m"}}
        }
        config = OptimizerJobConfig.model_validate(d)
        assert config.uses_llm_judges() is True

    def test_from_yaml_roundtrip(self, tmp_path):
        p = tmp_path / "opt.yaml"
        p.write_text(textwrap.dedent("""
                job:
                  name: opt-yaml
                optimizer:
                  datasets:
                    - name: bts_parquet
                      path: s3://b/data.parquet
                  prompt_set: bts_v1
                  models:
                    - name: gemini
                      provider: genos
                  modes: [summary_state]
                  judges:
                    - {name: length, weight: 0}
                execution:
                  type: aws_training_job
                output:
                  s3_path: s3://b/out/
                container:
                  recipe: qwen3_5
                """))
        config = OptimizerJobConfig.from_yaml(p)
        assert config.job.name == "opt-yaml"
        assert config._source_path == str(p)

    def test_merge_overrides_output_shortcut(self):
        config = OptimizerJobConfig.model_validate(_base_dict())
        merged = config.merge_overrides(output_path="s3://other/out/")
        assert str(merged.output.s3_path) == "s3://other/out/"


class TestValidation:
    def test_valid_config_has_no_errors(self):
        config = OptimizerJobConfig.model_validate(_base_dict())
        assert validate_optimizer_config(config) == []

    def test_aws_requires_s3_output(self):
        d = _base_dict()
        d["output"] = {"s3_path": "/local/out"}
        config = OptimizerJobConfig.model_validate(d)
        errors = validate_optimizer_config(config)
        assert any("s3://" in e for e in errors)

    def test_aws_dataset_requires_s3_path(self):
        d = _base_dict()
        d["optimizer"]["datasets"] = [
            {"name": "bts_parquet", "path": "/local/data.parquet"}
        ]
        config = OptimizerJobConfig.model_validate(d)
        errors = validate_optimizer_config(config)
        assert any("s3://" in e for e in errors)

    def test_llm_judge_without_clients_errors(self):
        d = _base_dict()
        d["optimizer"]["judges"] = [{"name": "correctness", "weight": 1.0}]
        config = OptimizerJobConfig.model_validate(d)
        errors = validate_optimizer_config(config)
        assert any("judge_clients" in e for e in errors)

    def test_llm_judge_unknown_client_errors(self):
        d = _base_dict()
        d["optimizer"]["judges"] = [
            {"name": "correctness", "weight": 1.0, "client": "missing"}
        ]
        d["optimizer"]["judge_clients"] = {
            "default": {"provider": "genos", "genos": {"model_id": "m"}}
        }
        config = OptimizerJobConfig.model_validate(d)
        errors = validate_optimizer_config(config)
        assert any("missing" in e for e in errors)

    def test_local_serving_on_cpu_instance_errors(self):
        d = _base_dict()
        d["optimizer"]["models"] = [{"name": "m", "provider": "local"}]
        d["optimizer"]["instance_type"] = "ml.m5.2xlarge"
        config = OptimizerJobConfig.model_validate(d)
        errors = validate_optimizer_config(config)
        assert any("GPU" in e for e in errors)

    def test_local_serving_without_image_errors(self):
        d = _base_dict()
        d["optimizer"]["models"] = [{"name": "m", "provider": "local"}]
        d["container"] = {}  # no recipe / image
        config = OptimizerJobConfig.model_validate(d)
        errors = validate_optimizer_config(config)
        assert any("GPU image" in e or "vLLM-capable" in e for e in errors)

    def test_local_serving_valid_on_gpu(self):
        d = _base_dict()
        d["optimizer"]["models"] = [{"name": "m", "provider": "local"}]
        d["optimizer"]["instance_type"] = "ml.g5.12xlarge"
        config = OptimizerJobConfig.model_validate(d)
        assert validate_optimizer_config(config) == []

    def test_aws_requires_prompts_path(self):
        d = _base_dict()
        del d["optimizer"]["prompts_path"]
        config = OptimizerJobConfig.model_validate(d)
        errors = validate_optimizer_config(config)
        assert any("prompts_path" in e for e in errors)

    def test_aws_prompts_path_must_be_s3(self):
        d = _base_dict()
        d["optimizer"]["prompts_path"] = "/local/prompts.yaml"
        config = OptimizerJobConfig.model_validate(d)
        errors = validate_optimizer_config(config)
        assert any("s3://" in e and "prompts_path" in e for e in errors)

    def test_packaged_prompt_set_needs_no_prompts_path(self):
        """A prompt set shipped inside foresightrlcore travels with the code.

        The launcher stages the package into the job, so single_v1's YAML is
        already in the container. Demanding an S3 upload for it would make the
        built-in prompt sets unusable on aws_training_job.
        """
        d = _base_dict()
        d["optimizer"]["prompt_set"] = "single_v1"
        del d["optimizer"]["prompts_path"]
        config = OptimizerJobConfig.model_validate(d)
        assert not any("prompts_path" in e for e in validate_optimizer_config(config))

    def test_prompt_set_outside_any_package_still_needs_prompts_path(self):
        """Being inside the repo is not the same as being inside a package.

        bts_v1's YAML lives in intuit/examples/prompts/, which is under the repo
        root in a source checkout but is *not* part of the foresightrl_intuit
        package -- so it never reaches the container. Naming the extension in
        extra_packages must not change that.
        """
        pytest.importorskip("foresightrl_intuit")
        d = _base_dict()
        del d["optimizer"]["prompts_path"]
        d["container"]["extra_packages"] = ["foresightrl_intuit"]
        config = OptimizerJobConfig.model_validate(d)
        assert any("prompts_path" in e for e in validate_optimizer_config(config))

    def test_unresolvable_prompt_set_reports_prompts_path(self):
        """An unknown name must not crash the offline check."""
        d = _base_dict()
        d["optimizer"]["prompt_set"] = "no_such_prompt_set"
        del d["optimizer"]["prompts_path"]
        config = OptimizerJobConfig.model_validate(d)
        assert any("prompts_path" in e for e in validate_optimizer_config(config))

    def test_local_backend_prompts_path_optional(self):
        d = _base_dict()
        d["execution"] = {"type": "local"}
        d["output"] = {"s3_path": "/tmp/out"}
        d["optimizer"]["datasets"] = [
            {"name": "bts_parquet", "path": "/tmp/data.parquet"}
        ]
        del d["optimizer"]["prompts_path"]
        config = OptimizerJobConfig.model_validate(d)
        # No prompts_path error on the local backend (uses the packaged default).
        errors = validate_optimizer_config(config)
        assert not any("prompts_path" in e for e in errors)

    def test_idps_secrets_without_vpc_errors(self):
        d = _base_dict()
        d["secrets"] = {
            "provider": "idps",
            "idps": {
                "policy_id": "p-x",
                "environment": "e2e",
                "endpoint": "vkm.example.com",
            },
            "entries": [{"env_var": "GENOS_APP_SECRET", "secret_path": "app_secret"}],
        }
        config = OptimizerJobConfig.model_validate(d)
        errors = validate_optimizer_config(config)
        assert any("vpc" in e.lower() for e in errors)

    def test_idps_secrets_with_vpc_ok(self):
        d = _base_dict()
        d["secrets"] = {
            "provider": "idps",
            "idps": {
                "policy_id": "p-x",
                "environment": "e2e",
                "endpoint": "vkm.example.com",
            },
            "entries": [{"env_var": "GENOS_APP_SECRET", "secret_path": "app_secret"}],
        }
        d["optimizer"]["vpc"] = {
            "subnets": ["subnet-abc"],
            "security_groups": ["sg-abc"],
        }
        config = OptimizerJobConfig.model_validate(d)
        assert validate_optimizer_config(config) == []
