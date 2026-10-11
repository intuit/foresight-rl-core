"""Tests for foresightrlcore.config.perf_validation -- validate_perf_config."""

from __future__ import annotations

import yaml

from foresightrlcore.config.perf_schema import PerfTestJobConfig
from foresightrlcore.config.perf_validation import validate_perf_config


def _make_config(tmp_path, **patch) -> PerfTestJobConfig:
    base = {
        "job": {"name": "perf-test"},
        "model": {"name": "s3://bucket/model/"},
        "perf_test": {
            "name": "run1",
            "serving": {"tensor_parallel_size": 1},
            "gatling": {"data_s3_path": "s3://bucket/data/"},
        },
        "execution": {"type": "aws_training_job"},
        "output": {"s3_path": "s3://bucket/out/"},
        "container": {"recipe": "qwen3_5"},
    }
    for k, v in patch.items():
        base[k] = v
    p = tmp_path / "c.yaml"
    p.write_text(yaml.dump(base))
    return PerfTestJobConfig.from_yaml(p)


class TestValidateExecution:
    def test_valid_aws_passes(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert validate_perf_config(cfg) == []

    def test_local_requires_gpus(self, tmp_path):
        cfg = _make_config(tmp_path, execution={"type": "local"})
        errors = validate_perf_config(cfg)
        assert any("execution.local.gpus" in e for e in errors)


class TestValidateOutput:
    def test_missing_output_path(self, tmp_path):
        cfg = _make_config(tmp_path, output={})
        errors = validate_perf_config(cfg)
        assert any("output.s3_path" in e for e in errors)

    def test_non_s3_output_path(self, tmp_path):
        cfg = _make_config(tmp_path, output={"s3_path": "/local/out"})
        errors = validate_perf_config(cfg)
        assert any("s3://" in e for e in errors)


class TestValidateContainer:
    def test_recipe_resolves_perf_test_image_passes(self, tmp_path):
        cfg = _make_config(tmp_path)
        assert validate_perf_config(cfg) == []

    def test_no_image_no_recipe_fails(self, tmp_path):
        cfg = _make_config(tmp_path, container={})
        errors = validate_perf_config(cfg)
        assert any("perf_test_image_uri" in e or "image_uri" in e for e in errors)


class TestValidateRun:
    def test_eagle3_requires_model(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            perf_test={
                "name": "r1",
                "serving": {
                    "speculative": {
                        "method": "eagle3",
                        "num_speculative_tokens": 5,
                    }
                },
                "gatling": {"data_s3_path": "s3://b/d/"},
            },
        )
        errors = validate_perf_config(cfg)
        assert any("speculative.model" in e for e in errors)

    def test_invalid_quantization(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            perf_test={
                "name": "r1",
                "serving": {"quantization": "weird"},
                "gatling": {"data_s3_path": "s3://b/d/"},
            },
        )
        errors = validate_perf_config(cfg)
        assert any("quantization" in e for e in errors)

    def test_compressed_tensors_quantization_allowed(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            perf_test={
                "name": "r1",
                "serving": {"quantization": "compressed-tensors"},
                "gatling": {"data_s3_path": "s3://b/d/"},
            },
        )
        errors = validate_perf_config(cfg)
        assert not any("quantization" in e for e in errors)


class TestValidateExecutionEdges:
    def test_unsupported_backend_type_returns_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        # Bypass the schema-level model_validator by mutating after parse.
        cfg.execution.type = "aws_serverless"
        errors = validate_perf_config(cfg)
        assert any("aws_serverless" in e for e in errors)


class TestValidateContainerEdges:
    def test_recipe_without_perf_test_image_uri_flags_error(
        self, tmp_path, monkeypatch
    ):
        # Inject a fake recipe that doesn't declare perf_test_image_uri to
        # exercise the branch in _validate_container.
        from foresightrlcore.config import recipes

        monkeypatch.setitem(
            recipes.CONTAINER_RECIPES,
            "_no_perf",
            {"framework_version": "x", "requirements": []},
        )
        cfg = _make_config(tmp_path, container={"recipe": "_no_perf"})
        # Pydantic recipe resolver runs at construction; ensure the resolved
        # field is also cleared so we hit the branch.
        cfg.container.perf_test_image_uri = None
        cfg.container.image_uri = None
        errors = validate_perf_config(cfg)
        assert any("does not declare a" in e for e in errors)


class TestValidateRunWarnings:
    def test_empty_simulation_class_errors(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.perf_test.gatling.simulation_class = ""
        errors = validate_perf_config(cfg)
        assert any("simulation_class" in e for e in errors)

    def test_high_tps_emits_warning(self, tmp_path, caplog):
        # max_num_seqs=1 with tps=10 -> 10*100 > 1*10, warning branch.
        cfg = _make_config(
            tmp_path,
            perf_test={
                "name": "r1",
                "serving": {"max_num_seqs": 1},
                "gatling": {"data_s3_path": "s3://b/d/", "tps_levels": [10]},
            },
        )
        with caplog.at_level(
            "WARNING", logger="foresightrlcore.config.perf_validation"
        ):
            errors = validate_perf_config(cfg)
        # Warning, not an error.
        assert errors == []
        assert "max_num_seqs" in caplog.text


class TestValidateSecretsWarning:
    def test_secret_looking_env_var_warns(self, tmp_path, caplog):
        cfg = _make_config(tmp_path, environment={"MY_API_KEY": "shhh"})
        with caplog.at_level(
            "WARNING", logger="foresightrlcore.config.perf_validation"
        ):
            errors = validate_perf_config(cfg)
        assert errors == []
        assert "MY_API_KEY" in caplog.text


class TestValidateIdpsRequiresVpc:
    """IDPS secrets pip-install idps-sdk from the Intuit-internal PyPI.

    That host is only reachable from inside the VPC, so the combination of
    ``secrets.provider: idps`` and ``aws_training_job`` without a
    ``perf_test.vpc`` block is a config error rather than a runtime surprise.
    Mirrors the same rule in optimizer_validation.
    """

    IDPS_SECRETS = {
        "provider": "idps",
        "idps": {
            "policy_id": "p-abc",
            "environment": "e2e",
            "endpoint": "vkm-e2e.example.com",
        },
        "entries": [{"env_var": "TOKEN", "secret_path": "APP_SECRET"}],
    }
    VPC = {
        "subnets": ["subnet-a"],
        "security_groups": ["sg-1"],
    }

    def _perf_block(self, **extra):
        block = {
            "name": "run1",
            "serving": {"tensor_parallel_size": 1},
            "gatling": {"data_s3_path": "s3://bucket/data/"},
        }
        block.update(extra)
        return block

    def test_idps_without_vpc_is_an_error(self, tmp_path):
        cfg = _make_config(
            tmp_path, secrets=self.IDPS_SECRETS, perf_test=self._perf_block()
        )
        errors = validate_perf_config(cfg)
        assert any("perf_test.vpc" in e for e in errors)

    def test_idps_with_vpc_passes(self, tmp_path):
        cfg = _make_config(
            tmp_path,
            secrets=self.IDPS_SECRETS,
            perf_test=self._perf_block(vpc=self.VPC),
        )
        assert validate_perf_config(cfg) == []

    def test_no_secrets_without_vpc_is_fine(self, tmp_path):
        """The rule is about IDPS specifically, not about VPC in general."""
        cfg = _make_config(tmp_path, perf_test=self._perf_block())
        assert validate_perf_config(cfg) == []
