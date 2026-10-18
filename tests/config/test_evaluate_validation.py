"""Pre-flight validation: every misconfiguration caught on the laptop."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from foresightrlcore.config.evaluate_schema import EvaluateJobConfig
from foresightrlcore.config.evaluate_validation import validate_evaluate_config


def _config(**overrides: Any) -> EvaluateJobConfig:
    payload: Dict[str, Any] = {
        "job": {"name": "eval"},
        "execution": {"type": "local"},
        "evaluate": {
            "benchmarks": [{"name": "demo_mcq"}],
            "models": [
                {"name": "claude", "provider": "bedrock", "region": "us-west-2"}
            ],
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = {**payload[key], **value}
        else:
            payload[key] = value
    return EvaluateJobConfig.model_validate(payload)


def _errors(**overrides: Any):
    return validate_evaluate_config(_config(**overrides))


def _aws(evaluate: Dict[str, Any] | None = None, **overrides: Any):
    base = {
        "execution": {"type": "aws_training_job"},
        "output": {"s3_path": "s3://bucket/eval/"},
    }
    base.update(overrides)
    if evaluate:
        base["evaluate"] = evaluate
    return _errors(**base)


def test_a_minimal_local_config_is_valid():
    assert _errors() == []


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def test_an_unknown_benchmark_lists_the_ones_that_exist():
    errors = _errors(
        evaluate={
            "benchmarks": [{"name": "not_a_benchmark"}],
            "models": [{"name": "claude"}],
        }
    )

    assert any("not in the catalog" in e and "demo_mcq" in e for e in errors)


def test_an_unknown_benchmark_type_lists_the_registered_ones():
    errors = _errors(
        evaluate={
            "benchmarks": [{"name": "b", "type": "telepathy"}],
            "models": [{"name": "claude"}],
        }
    )

    assert any("unknown type 'telepathy'" in e and "mcq" in e for e in errors)


def test_an_external_process_benchmark_needs_a_command():
    errors = _errors(
        evaluate={
            "benchmarks": [{"name": "b", "type": "external_process"}],
            "models": [{"name": "claude"}],
        }
    )

    assert any("requires a 'command'" in e for e in errors)


def test_venv_isolation_on_a_record_benchmark_is_meaningless():
    """A record-tier benchmark never sees the interpreter, so asking is a bug."""
    errors = _errors(
        evaluate={
            "benchmarks": [{"name": "b", "type": "mcq", "isolate": "venv"}],
            "models": [{"name": "claude"}],
        }
    )

    assert any("isolate=venv applies to harness-tier" in e for e in errors)


def test_a_plugin_harness_may_ask_for_its_own_interpreter():
    """It shells out with ``$EVALUATE_PYTHON`` exactly as external_process does.

    Tying the venv to ``command`` instead of to the tier locked out every
    harness that runs a subprocess without declaring one in YAML.
    """
    errors = _errors(
        evaluate={
            "benchmarks": [
                {
                    "name": "b",
                    "type": "plugin",
                    "isolate": "venv",
                    "requirements": ["tqdm"],
                    "params": {"target": "some.module:Harness"},
                }
            ],
            "models": [{"name": "claude"}],
        }
    )

    assert not any("isolate=venv" in e for e in errors)


def test_duplicate_benchmark_names_are_rejected():
    errors = _errors(
        evaluate={
            "benchmarks": [{"name": "demo_mcq"}, {"name": "demo_mcq"}],
            "models": [{"name": "claude"}],
        }
    )

    assert any("duplicate benchmark name" in e for e in errors)


def test_an_unknown_metric_lists_the_registered_ones():
    errors = _errors(
        evaluate={
            "benchmarks": [{"name": "b", "type": "mcq", "metrics": ["vibes"]}],
            "models": [{"name": "claude"}],
        }
    )

    assert any("unknown metric 'vibes'" in e and "exact_match" in e for e in errors)


def test_a_judge_metric_without_judge_clients_is_rejected():
    errors = _errors(
        evaluate={
            "benchmarks": [
                {"name": "b", "type": "generative", "metrics": ["llm_judge"]}
            ],
            "models": [{"name": "claude"}],
        }
    )

    assert any("judge_clients" in e for e in errors)


def test_a_judge_metric_with_judge_clients_is_accepted():
    errors = _errors(
        evaluate={
            "benchmarks": [
                {"name": "b", "type": "generative", "metrics": ["llm_judge"]}
            ],
            "models": [{"name": "claude"}],
            "judge_clients": {
                "default": {
                    "provider": "bedrock",
                    "bedrock": {"model_id": "anthropic.claude", "region": "us-west-2"},
                }
            },
        }
    )

    assert errors == []


def test_an_unpinned_git_source_warns_but_does_not_block(caplog):
    errors = _errors(
        evaluate={
            "benchmarks": [
                {
                    "name": "b",
                    "type": "external_process",
                    "command": ["run.sh"],
                    "source": {"git": "https://example.com/bench"},
                }
            ],
            "models": [{"name": "claude"}],
        }
    )

    assert errors == []
    assert "pins no source.ref" in caplog.text


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_duplicate_model_names_are_rejected():
    errors = _errors(
        evaluate={
            "benchmarks": [{"name": "demo_mcq"}],
            "models": [{"name": "claude"}, {"name": "claude"}],
        }
    )

    assert any("duplicate model name" in e for e in errors)


def test_a_model_group_referencing_an_unknown_model_is_rejected():
    errors = _errors(
        evaluate={
            "benchmarks": [{"name": "demo_mcq"}],
            "models": [{"name": "claude"}],
            "model_groups": {"frontier": ["claude", "ghost"]},
        }
    )

    assert any("unknown model(s) ['ghost']" in e for e in errors)


def test_a_benchmark_selecting_an_unknown_model_is_rejected():
    errors = _errors(
        evaluate={
            "benchmarks": [{"name": "demo_mcq", "models": ["ghost"]}],
            "models": [{"name": "claude"}],
        }
    )

    assert any("selects unknown model(s)" in e for e in errors)


def test_the_echo_provider_warns_that_scores_are_meaningless(caplog):
    _errors(
        evaluate={
            "benchmarks": [{"name": "demo_mcq"}],
            "models": [{"name": "e", "provider": "echo"}],
        }
    )

    assert "provider=echo" in caplog.text


# ---------------------------------------------------------------------------
# AWS-specific requirements
# ---------------------------------------------------------------------------


def test_an_aws_run_requires_an_output_path():
    errors = _errors(execution={"type": "aws_training_job"})
    assert any("requires output.s3_path" in e for e in errors)


def test_an_aws_output_path_must_be_an_s3_uri():
    errors = _errors(
        execution={"type": "aws_training_job"}, output={"s3_path": "/tmp/out"}
    )
    assert any("requires an s3:// URI" in e for e in errors)


def test_a_local_benchmark_source_cannot_be_used_on_aws(tmp_path):
    errors = _aws(
        evaluate={
            "benchmarks": [
                {
                    "name": "b",
                    "type": "external_process",
                    "command": ["run.sh"],
                    "source": {"path": str(tmp_path)},
                }
            ],
            "models": [{"name": "claude"}],
        }
    )

    assert any("not available inside a SageMaker job" in e for e in errors)


def test_a_local_dataset_path_cannot_be_used_on_aws():
    errors = _aws(
        evaluate={
            "benchmarks": [
                {
                    "name": "b",
                    "type": "mcq",
                    "dataset": {"loader": "csv", "path": "data/local.csv"},
                }
            ],
            "models": [{"name": "claude"}],
        }
    )

    assert any("requires an s3:// dataset path" in e for e in errors)


def test_an_s3_dataset_path_is_accepted_on_aws():
    errors = _aws(
        evaluate={
            "benchmarks": [
                {
                    "name": "b",
                    "type": "mcq",
                    "dataset": {"loader": "csv", "path": "s3://bucket/data.csv"},
                }
            ],
            "models": [{"name": "claude", "region": "us-west-2"}],
        }
    )

    assert errors == []


def test_a_broken_split_selection_is_reported_as_a_config_error():
    errors = _aws(
        evaluate={
            "benchmarks": [
                {
                    "name": "b",
                    "type": "mcq",
                    "dataset": {"splits": {"a": "s3://x/a", "b": "s3://x/b"}},
                }
            ],
            "models": [{"name": "claude"}],
        }
    )

    assert any("no `split` was selected" in e for e in errors)


# ---------------------------------------------------------------------------
# Infra sizing
# ---------------------------------------------------------------------------


def test_local_serving_on_a_cpu_instance_is_rejected():
    errors = _aws(
        evaluate={
            "benchmarks": [{"name": "demo_mcq"}],
            "models": [{"name": "Qwen/Qwen3-8B", "provider": "local"}],
            "instance_type": "ml.m5.2xlarge",
        }
    )

    assert any("is not a GPU family" in e for e in errors)


def test_local_serving_requires_a_vllm_capable_image():
    errors = _aws(
        evaluate={
            "benchmarks": [{"name": "demo_mcq"}],
            "models": [{"name": "Qwen/Qwen3-8B", "provider": "local"}],
        }
    )

    assert any("vLLM-capable GPU image" in e for e in errors)


def test_local_serving_with_a_recipe_is_accepted():
    errors = _aws(
        evaluate={
            "benchmarks": [{"name": "demo_mcq"}],
            "models": [{"name": "Qwen/Qwen3-8B", "provider": "local"}],
        },
        container={"recipe": "qwen3_5"},
    )

    assert errors == []


def test_a_gpu_instance_for_an_api_only_run_warns_about_cost(caplog):
    _aws(
        evaluate={
            "benchmarks": [{"name": "demo_mcq"}],
            "models": [{"name": "claude", "region": "us-west-2"}],
            "instance_type": "ml.g5.12xlarge",
        }
    )

    assert "wastes GPU cost" in caplog.text


# ---------------------------------------------------------------------------
# Reporters, secrets, backends
# ---------------------------------------------------------------------------


def test_an_unknown_reporter_lists_the_registered_ones():
    errors = _errors(
        evaluate={
            "benchmarks": [{"name": "demo_mcq"}],
            "models": [{"name": "claude"}],
            "reporters": ["carrier_pigeon"],
        }
    )

    assert any("unknown reporter" in e and "table" in e for e in errors)


def test_idps_secrets_on_aws_require_a_vpc():
    errors = _aws(
        secrets={
            "provider": "idps",
            "idps": {
                "endpoint": "https://idps.example.com",
                "policy_id": "p",
                "app_id": "a",
            },
            "entries": [{"secret_path": "x", "env_var": "Y"}],
        }
    )

    assert any("requires evaluate.vpc" in e for e in errors)


def test_a_secret_looking_env_var_is_flagged(caplog):
    _errors(environment={"GENOS_APP_SECRET": "hunter2"})
    assert "looks like a secret" in caplog.text
