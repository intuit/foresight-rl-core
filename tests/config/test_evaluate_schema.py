"""The evaluate config schema: defaults, merging, and infra resolution."""

from __future__ import annotations

from typing import Any, Dict

import pytest
import yaml
from pydantic import ValidationError

from foresightrlcore.config.evaluate_schema import (
    DEFAULT_CPU_INSTANCE_TYPE,
    DEFAULT_GPU_INSTANCE_TYPE,
    BenchmarkSpec,
    EvalDatasetSpec,
    EvaluateJobConfig,
    EvaluateModelConfig,
    EvaluateRunConfig,
)

MINIMAL: Dict[str, Any] = {
    "job": {"name": "eval-demo"},
    "evaluate": {
        "benchmarks": [{"name": "demo_mcq"}],
        "models": [{"name": "claude", "provider": "bedrock"}],
    },
}


def _config(**overrides) -> EvaluateJobConfig:
    payload = {**MINIMAL, **overrides}
    return EvaluateJobConfig.model_validate(payload)


# ---------------------------------------------------------------------------
# The headline claim: a benchmark is one line
# ---------------------------------------------------------------------------


def test_a_benchmark_needs_only_a_name():
    spec = BenchmarkSpec(name="demo_mcq")

    assert spec.type is None  # resolved from the catalog later
    assert spec.failure_tolerance == 0.0
    assert spec.isolate == "none"
    assert spec.timeout_seconds == 3600


def test_a_config_entry_overrides_only_what_it_sets():
    catalog_entry = BenchmarkSpec(
        name="b",
        type="mcq",
        metrics=["choice_match"],
        timeout_seconds=7200,
        dataset=EvalDatasetSpec(loader="csv", path="s3://bucket/full.csv"),
    )
    override = BenchmarkSpec(name="b", timeout_seconds=60)

    merged = override.merged_over(catalog_entry)

    assert merged.timeout_seconds == 60
    assert merged.type == "mcq"
    assert merged.metrics == ["choice_match"]
    assert merged.dataset is not None
    assert merged.dataset.path == "s3://bucket/full.csv"


def test_unknown_keys_reach_the_benchmark_type():
    spec = BenchmarkSpec.model_validate(
        {"name": "b", "type": "mcq", "number_of_choices": 4, "params": {"k": 1}}
    )

    params = spec.type_params()

    assert params["number_of_choices"] == 4
    assert params["k"] == 1


def test_explicit_params_win_over_a_stray_top_level_key():
    spec = BenchmarkSpec.model_validate(
        {"name": "b", "shots": 1, "params": {"shots": 5}}
    )
    assert spec.type_params()["shots"] == 5


@pytest.mark.parametrize("value", ["docker", "chroot", ""])
def test_an_unsupported_isolation_mode_is_rejected(value):
    with pytest.raises(ValidationError, match="isolate must be one of"):
        BenchmarkSpec(name="b", isolate=value)


# ---------------------------------------------------------------------------
# Datasets and splits
# ---------------------------------------------------------------------------


def test_a_lone_split_needs_no_selection():
    spec = EvalDatasetSpec(splits={"test": "s3://b/test.csv"})
    assert spec.resolved_path() == "s3://b/test.csv"


def test_a_named_split_is_selected():
    spec = EvalDatasetSpec(
        split="dev", splits={"dev": "s3://b/dev.csv", "test": "s3://b/test.csv"}
    )
    assert spec.resolved_path() == "s3://b/dev.csv"


def test_an_unknown_split_is_reported():
    spec = EvalDatasetSpec(split="train", splits={"dev": "s3://b/dev.csv"})

    with pytest.raises(ValueError, match="is not in splits"):
        spec.resolved_path()


def test_ambiguous_splits_must_be_disambiguated():
    spec = EvalDatasetSpec(splits={"dev": "a", "test": "b"})

    with pytest.raises(ValueError, match="no `split` was selected"):
        spec.resolved_path()


def test_loader_specific_keys_survive_to_the_loader():
    spec = EvalDatasetSpec.model_validate(
        {"loader": "csv", "path": "d.csv", "delimiter": ";"}
    )
    assert spec.as_loader_spec()["delimiter"] == ";"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def test_a_model_id_defaults_to_the_candidate_name():
    model = EvaluateModelConfig(name="anthropic.claude-sonnet")
    assert model.resolved_model_id == "anthropic.claude-sonnet"


def test_a_local_model_gets_default_serving_parameters():
    model = EvaluateModelConfig(name="Qwen/Qwen3-8B", provider="local")
    assert model.serving is not None


def test_an_unknown_provider_is_rejected():
    with pytest.raises(ValidationError, match="model.provider must be one of"):
        EvaluateModelConfig(name="m", provider="telepathy")


def test_an_agentic_candidate_is_recognised():
    model = EvaluateModelConfig.model_validate(
        {"name": "claude", "agent_runtime": {"kind": "claude_code", "max_turns": 20}}
    )
    assert model.is_agentic
    assert model.agent_runtime is not None
    assert model.agent_runtime.max_turns == 20


# ---------------------------------------------------------------------------
# Model groups and per-benchmark selection
# ---------------------------------------------------------------------------


def _run(**kwargs) -> EvaluateRunConfig:
    payload = {
        "benchmarks": [{"name": "b", "type": "mcq"}],
        "models": [
            {"name": "m1"},
            {"name": "m2"},
            {"name": "m3"},
        ],
    }
    payload.update(kwargs)
    return EvaluateRunConfig.model_validate(payload)


def test_a_benchmark_without_a_selector_runs_against_everything():
    run = _run()
    assert [m.name for m in run.resolve_models_for(run.benchmarks[0])] == [
        "m1",
        "m2",
        "m3",
    ]


def test_a_group_expands_to_its_members():
    run = _run(
        model_groups={"small": ["m2", "m3"]},
        benchmarks=[{"name": "b", "type": "mcq", "models": ["small"]}],
    )
    assert [m.name for m in run.resolve_models_for(run.benchmarks[0])] == ["m2", "m3"]


def test_overlapping_groups_do_not_duplicate_work():
    run = _run(
        model_groups={"a": ["m1", "m2"], "b": ["m2", "m3"]},
        benchmarks=[{"name": "b", "type": "mcq", "models": ["a", "b"]}],
    )
    assert [m.name for m in run.resolve_models_for(run.benchmarks[0])] == [
        "m1",
        "m2",
        "m3",
    ]


def test_a_per_benchmark_trial_count_wins_over_the_run_default():
    run = _run(trials=3, benchmarks=[{"name": "b", "type": "mcq", "trials": 7}])
    assert run.trials_for(run.benchmarks[0]) == 7


def test_a_run_needs_at_least_one_benchmark_and_model():
    with pytest.raises(ValidationError, match="at least one entry"):
        _run(benchmarks=[])
    with pytest.raises(ValidationError, match="at least one entry"):
        _run(models=[])


# ---------------------------------------------------------------------------
# Infra profile
# ---------------------------------------------------------------------------


def test_an_api_only_run_asks_for_a_cpu_instance():
    config = _config()

    assert config.requires_gpu() is False
    assert config.resolved_instance_type() == DEFAULT_CPU_INSTANCE_TYPE


def test_a_locally_served_candidate_asks_for_a_gpu_instance():
    config = _config(
        evaluate={
            "benchmarks": [{"name": "demo_mcq"}],
            "models": [{"name": "Qwen/Qwen3-8B", "provider": "local"}],
        }
    )

    assert config.requires_gpu() is True
    assert config.resolved_instance_type() == DEFAULT_GPU_INSTANCE_TYPE


def test_an_explicit_instance_type_wins():
    evaluate = dict(MINIMAL["evaluate"])
    evaluate["instance_type"] = "ml.c5.9xlarge"
    config = _config(evaluate=evaluate)
    assert config.resolved_instance_type() == "ml.c5.9xlarge"


# ---------------------------------------------------------------------------
# Backends and IO
# ---------------------------------------------------------------------------


def test_an_unsupported_backend_is_rejected_at_parse_time():
    with pytest.raises(ValidationError, match="does not support execution.type"):
        _config(execution={"type": "aws_serverless"})


def test_a_config_round_trips_through_yaml(tmp_path):
    path = tmp_path / "evaluate.yaml"
    path.write_text(yaml.safe_dump(MINIMAL))

    config = EvaluateJobConfig.from_yaml(path)

    assert config.job.name == "eval-demo"
    assert config._source_path == str(path)

    out = tmp_path / "round-trip.yaml"
    config.to_yaml(out)
    assert EvaluateJobConfig.from_yaml(out).job.name == "eval-demo"


def test_dotted_overrides_reach_nested_fields():
    merged = _config().merge_overrides(**{"evaluate.trials": 5})
    assert merged.evaluate.trials == 5


def test_the_output_path_shortcut_is_honoured():
    merged = _config().merge_overrides(output_path="s3://bucket/eval/")
    assert str(merged.output.s3_path) == "s3://bucket/eval/"


def test_encryption_is_on_by_default():
    """ACP IS031.1: opt-out, never opt-in."""
    assert _config().evaluate.encrypt_inter_container_traffic is True


def test_a_source_must_pin_exactly_one_location():
    from foresightrlcore.config.evaluate_schema import BenchmarkSource

    assert BenchmarkSource(git="https://x/y", ref="v1").git
    with pytest.raises(ValidationError, match="exactly one"):
        BenchmarkSource(git="https://x/y", path="/tmp/x")


# ---------------------------------------------------------------------------
# Catalog merge
# ---------------------------------------------------------------------------


def test_overriding_a_nested_field_keeps_the_rest_of_the_block():
    """`dataset: {limit: 10}` must not discard the catalog's dataset.path."""
    base = BenchmarkSpec.model_validate(
        {
            "name": "bench",
            "type": "mcq",
            "metrics": ["choice_match"],
            "dataset": {
                "loader": "csv",
                "path": "s3://bucket/bench.csv",
                "number_of_choices": 4,
            },
        }
    )
    entry = BenchmarkSpec.model_validate({"name": "bench", "dataset": {"limit": 10}})

    merged = entry.merged_over(base)

    assert merged.dataset is not None
    assert merged.dataset.path == "s3://bucket/bench.csv"
    assert merged.dataset.limit == 10
    assert merged.dataset.loader == "csv"
    assert merged.type == "mcq"


def test_a_list_field_is_replaced_not_appended():
    base = BenchmarkSpec.model_validate(
        {"name": "b", "type": "mcq", "metrics": ["a", "b"]}
    )
    entry = BenchmarkSpec.model_validate({"name": "b", "metrics": ["c"]})

    assert entry.merged_over(base).metrics == ["c"]


def test_params_merge_key_by_key():
    base = BenchmarkSpec.model_validate(
        {"name": "b", "type": "mcq", "params": {"instruction": "old", "max_tokens": 10}}
    )
    entry = BenchmarkSpec.model_validate(
        {"name": "b", "params": {"instruction": "new"}}
    )

    merged = entry.merged_over(base)

    assert merged.params["instruction"] == "new"
    assert merged.params["max_tokens"] == 10


# ---------------------------------------------------------------------------
# Round trip to a remote job
#
# The launcher writes the config to YAML and the container reads it back. If
# that trip promotes unset fields to explicit defaults, a benchmark entry stops
# being a thin override and starts clobbering the catalog it names -- and only
# ever remotely, which is the worst place to discover it.
# ---------------------------------------------------------------------------


def _round_trip(config, tmp_path):
    path = tmp_path / "args.yaml"
    config.to_yaml(path)
    return EvaluateJobConfig.from_yaml(str(path)), path


def _catalog_entry_config():
    payload = dict(MINIMAL)
    payload["evaluate"] = {
        "benchmarks": [{"name": "demo_mcq", "dataset": {"limit": 10}}],
        "models": [{"name": "m", "provider": "echo"}],
    }
    return EvaluateJobConfig.model_validate(payload)


def test_a_catalog_entry_survives_serialization(tmp_path):
    reloaded, path = _round_trip(_catalog_entry_config(), tmp_path)

    entry = reloaded.evaluate.benchmarks[0]
    assert entry.metrics == [], "author set no metrics"
    assert "metrics: []" not in path.read_text(), (
        "an unset field was written as an explicit empty list; merged over a "
        "catalog it would replace the catalog's metrics"
    )


def test_the_catalog_still_supplies_what_the_entry_omits(tmp_path):
    from foresightrlcore.evaluate.catalog import load_catalog, resolve_benchmark

    catalog = load_catalog()
    before = resolve_benchmark(_catalog_entry_config().evaluate.benchmarks[0], catalog)

    reloaded, _ = _round_trip(_catalog_entry_config(), tmp_path)
    after = resolve_benchmark(reloaded.evaluate.benchmarks[0], catalog)

    assert after.metrics == before.metrics
    assert after.type == before.type
    assert after.dataset is not None and before.dataset is not None
    assert after.dataset.loader == before.dataset.loader
    assert after.dataset.limit == 10, "the author's override must still win"


def test_an_explicit_metric_list_still_overrides_the_catalog(tmp_path):
    """exclude_unset must not swallow a value the author did set."""
    payload = dict(MINIMAL)
    payload["evaluate"] = {
        "benchmarks": [{"name": "demo_mcq", "metrics": ["exact_match"]}],
        "models": [{"name": "m", "provider": "echo"}],
    }
    config = EvaluateJobConfig.model_validate(payload)

    reloaded, _ = _round_trip(config, tmp_path)

    assert reloaded.evaluate.benchmarks[0].metrics == ["exact_match"]


def test_cli_overrides_do_not_promote_defaults(tmp_path):
    """`--set` reconstructs the config; that must not set every field."""
    config = _catalog_entry_config().merge_overrides(**{"evaluate.trials": 3})

    reloaded, path = _round_trip(config, tmp_path)

    assert reloaded.evaluate.trials == 3
    assert "metrics: []" not in path.read_text()
