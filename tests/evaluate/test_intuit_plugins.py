"""The Intuit evaluate extension: catalog registration and the Aleta reporter.

Skipped entirely in the ``oss`` tox env, which is the point -- nothing here may
be reachable from the core.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

pytestmark = pytest.mark.requires_intuit


@pytest.fixture(autouse=True)
def _extension_loaded():
    pytest.importorskip("foresightrl_intuit.evaluate")


def _result(**overrides):
    from foresightrlcore.evaluate.spec import EvaluateResult

    payload = {
        "job_name": "bench-run",
        "benchmarks": ["b1"],
        "models": ["m1"],
        "trials": 1,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:05:00+00:00",
        "n_results": 1,
        "aggregates": [
            {
                "benchmark": "b1",
                "model": "m1",
                "trials": 1,
                "mean": 0.75,
                "stdev": 0.0,
                "ci_low": 0.75,
                "ci_high": 0.75,
                "ci_level": 0.95,
                "scores": [0.75],
                "sub_scores": {},
                "n_samples": 20,
                "n_errors": 2,
                "cost_usd": 0.0,
                "valid": True,
            }
        ],
        "results": [{"benchmark": "b1", "model": "m1", "trial": 0, "n_errors": 2}],
        "samples": [],
    }
    payload.update(overrides)
    return EvaluateResult.model_validate(payload)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_the_extension_registers_its_reporter_and_catalog():
    from foresightrl_intuit.evaluate import CATALOG_PATH, _register_evaluate_plugins

    from foresightrlcore.evaluate.catalog import registered_catalogs
    from foresightrlcore.evaluate.registry import REPORTERS

    _register_evaluate_plugins()

    assert "aleta" in REPORTERS.list()
    assert CATALOG_PATH in registered_catalogs()


def test_the_extension_catalog_file_is_valid():
    """An unparseable catalog would break every evaluate run, not just ours."""
    from foresightrl_intuit.evaluate import CATALOG_PATH

    from foresightrlcore.evaluate.catalog import read_catalog_file

    read_catalog_file(CATALOG_PATH)


def test_the_extension_is_registered_for_remote_jobs():
    """SageMaker jobs have no entry-point metadata, so `_register` must do it."""
    import inspect

    import foresightrl_intuit

    source = inspect.getsource(foresightrl_intuit._register)
    assert "_register_evaluate_plugins" in source


def test_the_entry_point_group_is_declared():
    import tomllib

    pyproject = Path(__file__).resolve().parents[2] / "intuit" / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    group = data["project"]["entry-points"]["foresightrlcore.evaluate.plugins"]
    assert any("_register_evaluate_plugins" in v for v in group.values())


# ---------------------------------------------------------------------------
# Aleta payload construction
#
# These assert leaderboard parity with `llm-training-sample`'s
# pipelines/leaderboard_cli.py: metric = benchmark name, scaled to 0-100; one
# run per model; identity carried in run tags. A change that breaks one of
# these does not just alter our output, it makes our numbers incomparable with
# the series already on the board.
# ---------------------------------------------------------------------------

ASSET = "asset-test"


def _reporter(**kwargs):
    from foresightrl_intuit.evaluate.reporters.aleta import AletaReporter

    kwargs.setdefault("identifier", ASSET)
    kwargs.setdefault("batch_id", "testbatch")
    return AletaReporter(**kwargs)


def test_scores_are_reported_as_percentages_by_default():
    assert _reporter().metrics_for(_result(), "m1")["b1"] == 75.0


def test_raw_scores_are_reported_when_scaling_is_off():
    metrics = _reporter(scale_by_100=False).metrics_for(_result(), "m1")

    assert metrics["b1"] == 0.75


def test_the_metric_name_is_the_bare_benchmark_name():
    """The model is the run, not part of the metric key -- see the CSV columns
    in llm-training-sample's artificial.csv."""
    metrics = _reporter().metrics_for(_result(), "m1")

    assert "b1" in metrics
    assert not any("/" in name for name in metrics)


def test_failure_counts_travel_with_the_score():
    """A blank dashboard cell must be explainable without opening the run."""
    assert "failures_b1" in _reporter().metrics_for(_result(), "m1")


def test_an_invalid_benchmark_publishes_no_score():
    """The sentinel for 'not measured' must never reach the board as a zero."""
    from foresightrlcore.evaluate.spec import INVALID_SCORE

    result = _result()
    result.aggregates[0].update({"valid": False, "mean": INVALID_SCORE})

    metrics = _reporter().metrics_for(result, "m1")

    assert "b1" not in metrics
    assert "failures_b1" in metrics


def test_a_zero_score_is_dropped_like_the_reference_pipeline():
    """Deliberate parity with upstream, which skips any falsy metric.

    Publishing a 0.0 we drop would put a number on the board that the existing
    series never contains. Fixing it is an upstream change, not ours.
    """
    result = _result()
    result.aggregates[0]["mean"] = 0.0

    assert "b1" not in _reporter().metrics_for(result, "m1")


def test_a_confidence_metric_is_emitted_only_for_multi_trial_runs():
    assert "b1_CONFIDENCE" not in _reporter().metrics_for(_result(), "m1")

    result = _result()
    result.aggregates[0].update({"trials": 3, "ci_low": 0.65, "ci_high": 0.85})

    metrics = _reporter().metrics_for(result, "m1")
    assert metrics["b1_CONFIDENCE"] == pytest.approx(10.0)  # half-width, scaled


def test_a_models_metrics_exclude_every_other_model():
    result = _result(models=["m1", "m2"])
    result.aggregates.append({**result.aggregates[0], "model": "m2", "mean": 0.5})

    assert _reporter().metrics_for(result, "m1")["b1"] == 75.0
    assert _reporter().metrics_for(result, "m2")["b1"] == 50.0


def test_run_tags_carry_the_leaderboard_identity():
    result = _result(
        model_info={
            "m1": {
                "display_name": "Llama-3.1-8B",
                "leaderboard": {"model_size": "8B", "model_type": "pretrained"},
            }
        }
    )

    tags = _reporter().run_tags(result, "m1")

    assert tags["platform_llm_name"] == "Llama-3.1-8B"
    assert tags["model_size"] == "8B"
    assert tags["run_type"] == "genai_evaluation_and_benchmarking"
    assert tags["batch_id"] == "testbatch"


def test_a_model_without_metadata_still_publishes():
    """Metadata is optional; a missing block must not cost us the scores."""
    tags = _reporter().run_tags(_result(), "m1")

    assert tags["platform_llm_name"] == "m1"
    assert "model_size" not in tags


def test_the_leaderboard_tag_can_be_withheld():
    tags = _reporter(publish_to_leaderboard=False).run_tags(_result(), "m1")

    assert "run_type" not in tags


def test_code_artifact_logging_is_off():
    """Left on, the SDK uploads the whole staged source tree from inside a job."""
    assert _reporter().client_kwargs("mlmodel_asset")["log_code_artifacts"] is False


# ---------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------


def _fake_sdk():
    """A stand-in for the real SDK, so these run without it installed.

    Returns the ``sys.modules`` entries to patch and the client they produce.
    ``aleta.utils`` is included because the reporter imports ``IdentifierType``
    from it; patching only ``aleta`` makes that submodule import fail, which the
    reporter then reports as "not installed" and every assertion here passes
    vacuously.
    """
    module = mock.MagicMock()
    client = module.Aleta.return_value
    client.set_experiment.return_value.experiment_id = "exp-1"
    client.create_run.return_value.__enter__.return_value.run_id = "run-1"
    return {"aleta": module, "aleta.utils": mock.MagicMock()}, client


def test_the_fake_sdk_is_actually_reached(tmp_path):
    """Guards the vacuous-pass failure mode the fixture above describes."""
    modules, client = _fake_sdk()
    with mock.patch.dict(sys.modules, modules):
        _reporter(upload_artifacts=False).report(_result(), tmp_path)

    client.create_run.assert_called()


def test_the_run_is_not_lost_when_aleta_is_missing(tmp_path, caplog, monkeypatch):
    monkeypatch.setitem(sys.modules, "aleta", None)  # forces ImportError

    _reporter().report(_result(), tmp_path)

    assert "not installed" in caplog.text


def test_publishing_without_an_identifier_declines_rather_than_guesses(
    tmp_path, caplog
):
    from foresightrl_intuit.evaluate.reporters.aleta import AletaReporter

    modules, _ = _fake_sdk()
    with mock.patch.dict(sys.modules, modules):
        AletaReporter().report(_result(), tmp_path)

    assert "without an 'identifier'" in caplog.text
    modules["aleta"].Aleta.assert_not_called()


def test_a_failing_publish_is_a_warning_not_a_crash(tmp_path, caplog):
    modules, client = _fake_sdk()
    client.set_experiment.side_effect = RuntimeError("tracking service down")

    with mock.patch.dict(sys.modules, modules):
        _reporter().report(_result(), tmp_path)

    assert "could not open experiment" in caplog.text


def test_one_model_failing_does_not_lose_the_others(tmp_path, caplog):
    modules, client = _fake_sdk()
    client.create_run.side_effect = [RuntimeError("boom"), mock.MagicMock()]

    with mock.patch.dict(sys.modules, modules):
        _reporter().report(_result(models=["m1", "m2"]), tmp_path)

    assert client.create_run.call_count == 2


def test_metrics_and_artifacts_are_published(tmp_path):
    from foresightrl_intuit.evaluate.reporters.aleta import ARTIFACTS

    for name in ARTIFACTS:
        (tmp_path / name).write_text("x")

    modules, client = _fake_sdk()
    with mock.patch.dict(sys.modules, modules):
        _reporter(experiment="llm-leaderboard").report(_result(), tmp_path)

    client.set_experiment.assert_called_once_with("llm-leaderboard")
    logged = {}
    for call in client.log_metric.call_args_list:
        logged.update(call.args[0])
    assert logged["b1"] == 75.0
    assert client.log_artifact.call_count == len(ARTIFACTS)


def test_the_client_is_built_from_the_configured_asset(tmp_path):
    modules, _ = _fake_sdk()
    with mock.patch.dict(sys.modules, modules):
        _reporter(env="E2E", upload_artifacts=False).report(_result(), tmp_path)

    kwargs = modules["aleta"].Aleta.call_args.kwargs
    assert kwargs["identifier"] == ASSET
    assert kwargs["env"] == "E2E"
    assert "configs" not in kwargs  # see client_kwargs: configs= sends a bad devp_type


def test_the_identifier_type_is_passed_as_an_enum_member_not_a_string():
    """The SDK's keyword path does `identifier_type.value`, so a bare string
    would raise -- and its `configs=` path, which accepts one, interpolates the
    enum into a query string and gets a 422 back on Python 3.11+."""
    IdentifierType = pytest.importorskip("aleta.utils").IdentifierType

    member = IdentifierType("mlmodel_asset")

    assert _reporter().client_kwargs(member)["identifier_type"] is member
    assert member.value == "mlmodel_asset"
    # The reason the configs= path cannot be used, asserted rather than claimed.
    assert f"{member}" != member.value


def test_one_run_is_created_per_model(tmp_path):
    """A leaderboard row is a model. Two candidates must be two runs."""
    modules, client = _fake_sdk()
    result = _result(models=["m1", "m2"])
    result.aggregates.append({**result.aggregates[0], "model": "m2"})

    with mock.patch.dict(sys.modules, modules):
        _reporter(upload_artifacts=False).report(result, tmp_path)

    assert client.create_run.call_count == 2
    named = {
        c.kwargs["tags"]["platform_llm_name"] for c in client.create_run.call_args_list
    }
    assert named == {"m1", "m2"}


# ---------------------------------------------------------------------------
# The production benchmark catalog
# ---------------------------------------------------------------------------


def _intuit_catalog():
    from foresightrl_intuit.evaluate import CATALOG_PATH

    from foresightrlcore.evaluate.catalog import read_catalog_file

    return read_catalog_file(CATALOG_PATH)


def _leaderboard_suite():
    """The eleven entries we score ourselves.

    The catalog also carries harness-tier entries that hand scoring to an
    external runner. Those are a different kind of thing -- no prompt of ours,
    no metric of ours -- so the guarantees below are about the suite, not about
    every line in the file.
    """
    return {
        name: spec
        for name, spec in _intuit_catalog().items()
        if name.startswith("LLM_BENCHMARK_")
    }


def test_the_catalog_ships_the_production_suite():
    suite = _leaderboard_suite()

    assert len(suite) == 11
    assert "LLM_BENCHMARK_irs_bench" in suite


def test_every_suite_entry_is_runnable_without_further_config():
    """A catalog entry must carry everything a one-line config needs."""
    for name, spec in _leaderboard_suite().items():
        assert spec.type == "mcq", name
        assert spec.dataset is not None, name
        assert spec.dataset.loader == "csv", name
        assert (spec.dataset.path or "").startswith("s3://"), name
        assert getattr(spec.dataset, "number_of_choices", None), name
        assert spec.metrics, name


#: Tables that carry both a bare ``question`` and a ``prompt`` holding the
#: passage it refers to. Column precedence prefers ``question``, which asks the
#: model to reason about text it was never shown -- LogiQA2 scored below its
#: own chance baseline that way, with 57% of answers replying that the passage
#: was missing, and MuSR loses ~4.3k characters of narrative per row.
_NEEDS_EXPLICIT_PROMPT_COLUMN = (
    "LLM_BENCHMARK_logiqa2",
    "LLM_BENCHMARK_musr_complete",
    "LLM_BENCHMARK_finqa_l1",
)


def test_benchmarks_whose_question_needs_a_passage_pin_the_prompt_column():
    suite = _leaderboard_suite()
    for name in _NEEDS_EXPLICIT_PROMPT_COLUMN:
        spec = suite[name]
        assert getattr(spec.dataset, "question_column", None) == "prompt", (
            f"{name} must name its question column explicitly; the loader's "
            "default would send the question without its passage"
        )


def test_the_reported_score_is_the_legacy_metric():
    """First metric wins, and it must stay the one the leaderboard used."""
    for name, spec in _leaderboard_suite().items():
        assert spec.metrics[0] == "chat_completion_match", name


def test_every_suite_entry_renders_a_prompt_without_placeholders_left_over():
    from typing import cast

    from foresightrlcore.evaluate.benchmarks.mcq import McqBenchmark
    from foresightrlcore.evaluate.interfaces import EvalRecord
    from foresightrlcore.evaluate.registry import BENCHMARK_TYPES

    record = EvalRecord(
        id="r", inputs={"question": "Q?"}, gold="0", choices=["one", "two"]
    )
    for name, spec in _leaderboard_suite().items():
        benchmark = cast(McqBenchmark, BENCHMARK_TYPES.get(spec.type)(spec=spec))
        text = benchmark.build_prompt(record).messages[-1]["content"]
        assert "{question}" not in text and "{choices}" not in text, name
        assert "{{" not in text, name
        assert "Q?" in text, name


#: Benchmark types whose scoring belongs to someone else's code, so the entry
#: has to bring that code and an interpreter with it. Selected by type rather
#: than by name: the catalog also holds record-tier entries that are neither
#: leaderboard MCQs nor harnesses, and those stage nothing.
_HARNESS_TYPES = frozenset({"plugin", "external_process", "lm_eval", "scenario_suite"})


def test_a_harness_entry_stages_its_own_source_and_interpreter():
    """The other kind of catalog entry: scoring belongs to someone else's code."""
    external = {
        name: spec
        for name, spec in _intuit_catalog().items()
        if spec.type in _HARNESS_TYPES
    }

    assert external, "expected at least one externally-scored entry"
    for name, spec in external.items():
        assert spec.source is not None, name
        assert spec.isolate == "venv", name
        assert spec.requirements, name


# ---------------------------------------------------------------------------
# The legacy scoring rule
# ---------------------------------------------------------------------------


def _score(gold, choices, text):
    from foresightrl_intuit.evaluate.metrics.chat_completion import (
        ChatCompletionMatchMetric,
    )

    from foresightrlcore.evaluate.interfaces import EvalRecord, Prediction

    record = EvalRecord(id="r", inputs={}, gold=gold, choices=choices)
    return ChatCompletionMatchMetric().score(record, Prediction(text=text))


def test_the_metric_is_registered():
    from foresightrlcore.evaluate.registry import METRICS

    assert "chat_completion_match" in METRICS


def test_the_gold_option_appearing_anywhere_counts_as_correct():
    verdict = _score(
        "0",
        ["A. Paris", "B. London"],
        '{"explanation": "...", "correct_answer": "A) Paris"}',
    )

    assert verdict.value == 1.0
    assert verdict.correct is True


def test_normalisation_ignores_periods_apostrophes_and_parens():
    from foresightrl_intuit.evaluate.metrics.chat_completion import normalize_legacy

    assert normalize_legacy("A) It's a Corp.") == "a its a corp"


def test_a_different_option_scores_zero():
    verdict = _score("0", ["A. Paris", "B. London"], "The answer is B. London")

    assert verdict.value == 0.0
    assert verdict.correct is False


def test_an_empty_generation_scores_zero_rather_than_flagging():
    """Upstream behaviour, reproduced knowingly: empty is wrong, not unscored."""
    verdict = _score("0", ["A. Paris", "B. London"], "")

    assert verdict.value == 0.0
    assert verdict.parse_failed is False
    assert verdict.detail["empty"] is True


def test_an_unresolvable_gold_is_a_parse_failure_not_a_zero():
    verdict = _score(None, ["A. Paris"], "A. Paris")

    assert verdict.parse_failed is True


def test_a_failed_generation_is_never_scored():
    from foresightrl_intuit.evaluate.metrics.chat_completion import (
        ChatCompletionMatchMetric,
    )

    from foresightrlcore.evaluate.interfaces import EvalRecord, Prediction

    record = EvalRecord(id="r", inputs={}, gold="0", choices=["A. Paris"])
    verdict = ChatCompletionMatchMetric().score(
        record, Prediction(text="", error="429 throttled")
    )

    assert verdict.value is None or verdict.value == 0.0
    assert verdict.detail["error"] == "429 throttled"


# ---------------------------------------------------------------------------
# Upstream parity
#
# The catalog exists so scores stay comparable with the published leaderboard
# series, and both halves of that -- the prompt sent and the rule applied --
# were copied from `filmbenchmarking`. A copy drifts silently: trailing
# whitespace gets stripped by an editor, an example gets "cleaned up", and the
# numbers move for a reason no one can see. These tests hold the upstream text
# and the upstream scoring rule as literals and fail on any divergence.
# ---------------------------------------------------------------------------

UPSTREAM_MCQ_INSTRUCTION = "\n".join(
    [
        'The output should be ONLY a JSON with 2 fields: "explanation" and '
        '"correct_answer". The "explanation" is the explanation of your '
        'decision. The "correct_answer" is the full correct answer as presented '
        "in the choices, including the number or letter of the correct answer, "
        "it should be the exact match of the correct choice. ",
        "There is only one correct answer.",
        'For example, for the question "What is the capital of France?" with '
        'the choices "A) Paris B) London C) Berlin D) Rome", the output should '
        "be: ",
        "{",
        ' "explanation": "Paris is the capital of France",',
        ' "correct_answer": "A) Paris"',
        "}",
        'Another example, for the question "What is a credit score?" with the '
        'choices "A) A score that represents the creditworthiness of an '
        "individual or business B) A score that represents the creditworthiness "
        "of a business C) A score that represents the creditworthiness of an "
        "individual D) A score that represents the creditworthiness of a "
        'country", the output should be:',
        "{",
        ' "explanation": "A credit score is a numerical expression based on a '
        "level analysis of a person's credit files, to represent the "
        'creditworthiness of an individual.",',
        ' "correct_answer": "A) A score that represents the creditworthiness of '
        'an individual or business"',
        "}",
    ]
)

UPSTREAM_BOOL_INSTRUCTION = "\n".join(
    [
        'The output should be ONLY a JSON with 2 fields: "explanation" and '
        '"correct_answer". ',
        'The "explanation" is the explanation of your decision. The '
        '"correct_answer" value should be either "True" or "False".  ',
        'for example, for the statement "The Earth orbits around the Sun", the '
        "output should be: ",
        "{",
        '    "explanation": "The Earth orbits around the Sun due to the '
        "gravitational pull exerted by the Sun's massive mass,",
        '     which keeps the Earth in a stable elliptical orbit.",',
        '    "correct_answer": "True",     ',
        "}",
    ]
)


def _shared_block():
    import yaml
    from foresightrl_intuit.evaluate import CATALOG_PATH

    return yaml.safe_load(CATALOG_PATH.read_text())["_shared"]


@pytest.mark.requires_intuit
@pytest.mark.parametrize(
    "key, expected",
    [
        ("mcq_instruction", UPSTREAM_MCQ_INSTRUCTION),
        ("bool_instruction", UPSTREAM_BOOL_INSTRUCTION),
    ],
)
def test_catalog_prompts_match_upstream(key, expected):
    assert _shared_block()[key] == expected, (
        f"{key} no longer matches filmbenchmarking verbatim -- most likely "
        "trailing whitespace was stripped. The prompt is part of the "
        "measurement; changing it invalidates comparison with published scores."
    )


def _upstream_compute_chat_completion(prediction: str, reference: str) -> int:
    """`BaseTaskConfig.chat_completion` from filmbenchmarking, transcribed."""

    def process(values):
        return [
            v.lower().replace(".", "").replace("'", "").replace(")", "").strip()
            for v in values
        ]

    if prediction:
        return 1 if process([reference])[0] in process([prediction])[0] else 0
    return 0


@pytest.mark.requires_intuit
@pytest.mark.parametrize(
    "generation, choices, gold",
    [
        ("", ["A. x", "B. y"], "0"),
        ("A. X", ["a. x", "B. y"], "0"),
        ("the answer is (B) foo.", ["A. bar", "B) foo"], "1"),
        ("It's B's answer", ["A. bar", "Bs answer"], "1"),
        ("no match here", ["A. bar", "B. baz"], "0"),
        ("  spaced  ", ["  spaced  ", "z"], "0"),
        ('{"correct_answer": "C. $25,000"}', ["A. $1", "B. $2", "C. $25,000"], "2"),
    ],
)
def test_the_metric_agrees_with_upstream(generation, choices, gold):
    from foresightrl_intuit.evaluate.metrics.chat_completion import (
        ChatCompletionMatchMetric,
    )

    from foresightrlcore.evaluate.interfaces import EvalRecord, Prediction

    record = EvalRecord(id="r", inputs={}, choices=list(choices), gold=gold)
    ours = ChatCompletionMatchMetric().score(record, Prediction(text=generation))

    assert ours.value == float(
        _upstream_compute_chat_completion(generation, choices[int(gold)])
    )
