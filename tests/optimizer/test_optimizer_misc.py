"""Operation local path, public launch API, CLI error paths, schema + validation.

Fills the remaining uncovered branches across the optimizer's thin glue modules.
"""

from __future__ import annotations

import argparse
import os
from typing import Any
from unittest.mock import patch

import foresightrlcore.optimizer.judges  # noqa: F401  (registry side effects)
from foresightrlcore.config.optimizer_schema import OptimizerJobConfig
from foresightrlcore.config.optimizer_validation import validate_optimizer_config
from foresightrlcore.operations.optimizer import OptimizerOperation
from foresightrlcore.optimizer import cli as optimizer_cli


def _base(**over):
    d = {
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
    for k, v in over.items():
        d[k] = v
    return d


class TestRunLocal:
    def test_exports_environment_and_runs(self, monkeypatch):
        d = _base(
            execution={"type": "local"},
            output={"s3_path": "/tmp/opt_local_out"},
            environment={"GENOS_ENV_TEST_XYZ": "e2e"},
        )
        config = OptimizerJobConfig.model_validate(d)
        monkeypatch.delenv("GENOS_ENV_TEST_XYZ", raising=False)
        with patch("foresightrlcore.optimizer.runner.run_optimizer") as run_optimizer:
            result = OptimizerOperation().run(config)
        run_optimizer.assert_called_once()
        assert os.environ["GENOS_ENV_TEST_XYZ"] == "e2e"
        assert result == ["opt"]

    def test_s3_output_path_still_runs(self):
        d = _base(execution={"type": "local"}, output={"s3_path": "s3://b/out/"})
        config = OptimizerJobConfig.model_validate(d)
        with patch("foresightrlcore.optimizer.runner.run_optimizer") as run_optimizer:
            result = OptimizerOperation().run(config)
        run_optimizer.assert_called_once()
        assert result == ["opt"]

    def test_secrets_are_resolved(self, tmp_path):
        d = _base(
            execution={"type": "local"},
            output={"s3_path": "/tmp/opt_secret_out"},
            secrets={
                "provider": "local",
                "local": {"env_file": ".env"},
                "entries": [
                    {"env_var": "GENOS_APP_SECRET", "secret_path": "app_secret"}
                ],
            },
        )
        cfg_path = tmp_path / "opt.yaml"
        OptimizerJobConfig.model_validate(d).to_yaml(cfg_path)
        config = OptimizerJobConfig.from_yaml(cfg_path)
        with (
            patch("foresightrlcore.optimizer.runner.run_optimizer"),
            patch("foresightrlcore.auth.resolve_secrets") as resolve_secrets,
        ):
            OptimizerOperation().run(config)
        resolve_secrets.assert_called_once()

    def test_secret_resolution_failure_propagates(self, tmp_path):
        d = _base(
            execution={"type": "local"},
            output={"s3_path": "/tmp/opt_secret_out"},
            secrets={
                "provider": "local",
                "local": {"env_file": ".env"},
                "entries": [
                    {"env_var": "GENOS_APP_SECRET", "secret_path": "app_secret"}
                ],
            },
        )
        cfg_path = tmp_path / "opt.yaml"
        OptimizerJobConfig.model_validate(d).to_yaml(cfg_path)
        config = OptimizerJobConfig.from_yaml(cfg_path)
        with patch(
            "foresightrlcore.auth.resolve_secrets",
            side_effect=RuntimeError("bad secret"),
        ):
            try:
                OptimizerOperation().run(config)
                assert False, "expected RuntimeError"
            except RuntimeError as exc:
                assert "bad secret" in str(exc)


class TestPublicLaunch:
    def test_launch_loads_and_runs(self, tmp_path):
        import foresightrlcore.optimizer as optimizer_pkg

        cfg_path = tmp_path / "opt.yaml"
        OptimizerJobConfig.model_validate(_base()).to_yaml(cfg_path)
        with patch.object(OptimizerOperation, "run", return_value=["job-xyz"]) as run:
            result = optimizer_pkg.launch(str(cfg_path), wait=True)
        run.assert_called_once()
        assert result == ["job-xyz"]

    def test_launch_applies_overrides(self, tmp_path):
        import foresightrlcore.optimizer as optimizer_pkg

        cfg_path = tmp_path / "opt.yaml"
        OptimizerJobConfig.model_validate(_base()).to_yaml(cfg_path)
        captured = {}

        def fake_run(self, config, wait=False):
            captured["concurrency"] = config.optimizer.concurrency
            return ["job"]

        overrides: dict[str, Any] = {"optimizer.concurrency": 9}
        with patch.object(OptimizerOperation, "run", fake_run):
            optimizer_pkg.launch(str(cfg_path), **overrides)
        assert captured["concurrency"] == 9


class TestCliErrorPaths:
    def test_dispatch_unknown_command(self):
        args = argparse.Namespace(command="bogus")
        assert optimizer_cli.dispatch(args) == 1

    def test_validate_missing_file(self):
        args = argparse.Namespace(
            command="validate-optimizer", config="/no/such/file.yaml"
        )
        assert optimizer_cli.dispatch(args) == 1

    def test_validate_malformed_config(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("this: [is, not, valid: optimizer config")
        args = argparse.Namespace(command="validate-optimizer", config=str(bad))
        assert optimizer_cli.dispatch(args) == 1

    def test_optimizer_launch_failure_returns_1(self):
        args = argparse.Namespace(command="optimizer", config="c.yaml", wait=False)
        with patch(
            "foresightrlcore.optimizer.launch", side_effect=RuntimeError("nope")
        ):
            assert optimizer_cli.dispatch(args) == 1


class TestSchemaMergeOverrides:
    def test_dotted_key_override(self):
        config = OptimizerJobConfig.model_validate(_base())
        merged = config.merge_overrides(**{"optimizer.concurrency": 12})
        assert merged.optimizer.concurrency == 12

    def test_none_override_is_ignored(self):
        config = OptimizerJobConfig.model_validate(_base())
        merged = config.merge_overrides(output_path=None)
        assert str(merged.output.s3_path) == "s3://b/out/"

    def test_as_provider_spec_includes_extra_keys(self):
        config = OptimizerJobConfig.model_validate(
            _base(
                optimizer={
                    "datasets": [
                        {
                            "name": "parquet",
                            "path": "s3://b/d.parquet",
                            "reference_column": "gold",
                        }
                    ],
                    "prompt_set": "bts_v1",
                    "prompts_path": "s3://b/p.yaml",
                    "models": [{"name": "g", "provider": "genos"}],
                    "modes": ["summary_state"],
                    "judges": [{"name": "length", "weight": 0}],
                }
            )
        )
        spec = config.optimizer.datasets[0].as_provider_spec()
        assert spec["reference_column"] == "gold"


class TestValidationExtras:
    def test_aws_missing_output_path_errors(self):
        d = _base(output={})
        config = OptimizerJobConfig.model_validate(d)
        errors = validate_optimizer_config(config)
        assert any("output.s3_path" in e for e in errors)

    def test_dataset_missing_path_errors(self):
        d = _base()
        d["optimizer"]["datasets"] = [{"name": "bts_parquet"}]
        config = OptimizerJobConfig.model_validate(d)
        errors = validate_optimizer_config(config)
        assert any("path" in e for e in errors)

    def test_bedrock_model_without_region_warns(self, monkeypatch, caplog):
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        d = _base()
        d["optimizer"]["models"] = [{"name": "claude", "provider": "bedrock"}]
        config = OptimizerJobConfig.model_validate(d)
        # A warning, not an error - validation must still pass.
        errors = validate_optimizer_config(config)
        assert not any("region" in e for e in errors)

    def test_gpu_instance_for_api_only_warns_not_errors(self):
        d = _base()
        d["optimizer"]["instance_type"] = "ml.g5.12xlarge"
        config = OptimizerJobConfig.model_validate(d)
        # API-only on a GPU instance is wasteful but valid.
        assert validate_optimizer_config(config) == []

    def test_secret_like_env_var_warns_not_errors(self):
        d = _base(environment={"MY_API_KEY": "shh"})
        config = OptimizerJobConfig.model_validate(d)
        assert validate_optimizer_config(config) == []
