"""Benchmark types, both tiers."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import cast
from unittest import mock

import pytest

from foresightrlcore.config.evaluate_schema import BenchmarkSpec
from foresightrlcore.evaluate.benchmarks.external_process import (
    ExternalProcessBenchmark,
    expand,
    read_outcome,
)
from foresightrlcore.evaluate.benchmarks.generative import GenerativeBenchmark
from foresightrlcore.evaluate.benchmarks.lm_eval_adapter import LmEvalBenchmark
from foresightrlcore.evaluate.benchmarks.mcq import McqBenchmark
from foresightrlcore.evaluate.benchmarks.plugin import (
    PluginBenchmark,
    coerce_outcome,
    import_target,
)
from foresightrlcore.evaluate.benchmarks.scenario_suite import ScenarioSuiteBenchmark
from foresightrlcore.evaluate.interfaces import (
    EvalRecord,
    HarnessContext,
    HarnessOutcome,
    Prediction,
    is_harness_benchmark,
    is_record_benchmark,
)
from foresightrlcore.evaluate.registry import BENCHMARK_TYPES
from tests.evaluate.conftest import FakeChatClient


def _mcq_record():
    return EvalRecord(
        id="r1",
        inputs={"question": "Capital of France?"},
        gold="Paris",
        choices=["Berlin", "Madrid", "Paris", "Rome"],
    )


def _ctx(endpoint, tmp_path, **kwargs) -> HarnessContext:
    workspace = Path(tmp_path)
    return HarnessContext(
        benchmark=kwargs.pop("benchmark", "bench"),
        endpoint=endpoint,
        workspace=workspace,
        output_file=workspace / "evaluate_result.json",
        env=kwargs.pop("env", {}),
        logger=logging.getLogger("test"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tier detection
# ---------------------------------------------------------------------------


def test_the_two_tiers_are_told_apart_by_shape_not_by_a_flag():
    assert is_record_benchmark(McqBenchmark())
    assert not is_harness_benchmark(McqBenchmark())

    suite = ScenarioSuiteBenchmark()
    assert is_harness_benchmark(suite)
    assert not is_record_benchmark(suite)


# ---------------------------------------------------------------------------
# mcq
# ---------------------------------------------------------------------------


def test_mcq_labels_the_choices_and_asks_for_a_letter():
    request = McqBenchmark().build_prompt(_mcq_record())
    text = request.messages[0]["content"]

    assert "A. Berlin" in text and "C. Paris" in text
    assert "letter" in text.lower()
    assert [m["role"] for m in request.messages] == ["user"]


def test_mcq_omits_the_system_turn_unless_one_is_configured():
    with_system = McqBenchmark(system_prompt="Be terse.").build_prompt(_mcq_record())
    assert [m["role"] for m in with_system.messages] == ["system", "user"]


def test_mcq_truncates_to_number_of_choices():
    benchmark = McqBenchmark(number_of_choices=2)
    assert benchmark.render_choices(_mcq_record()) == "A. Berlin\nB. Madrid"


def test_mcq_scores_with_its_first_metric_and_records_the_rest():
    from foresightrlcore.evaluate.metrics.text_match import ExactMatchMetric

    benchmark = McqBenchmark(metrics=[ExactMatchMetric(), ExactMatchMetric()])
    verdict = benchmark.score(_mcq_record(), Prediction(text="Paris"))

    assert verdict.correct
    assert verdict.detail["ExactMatchMetric"] == 1.0


def test_mcq_factory_reads_params_off_the_spec():
    spec = BenchmarkSpec(
        name="b", type="mcq", params={"number_of_choices": 3, "instruction": "Pick."}
    )
    benchmark = cast(McqBenchmark, BENCHMARK_TYPES.get("mcq")(spec=spec, metrics=None))

    assert benchmark.number_of_choices == 3
    assert benchmark.instruction == "Pick."


# ---------------------------------------------------------------------------
# generative
# ---------------------------------------------------------------------------


def test_generative_appends_its_instruction_to_the_question():
    request = GenerativeBenchmark(instruction="Answer in one word.").build_prompt(
        _mcq_record()
    )
    text = request.messages[0]["content"]

    assert text.startswith("Capital of France?")
    assert text.endswith("Answer in one word.")


def test_generative_carries_a_per_benchmark_token_cap():
    request = GenerativeBenchmark(max_tokens=64).build_prompt(_mcq_record())
    assert request.max_tokens == 64


# ---------------------------------------------------------------------------
# scenario_suite
# ---------------------------------------------------------------------------


def _scenarios():
    return [
        EvalRecord(
            id="s1",
            inputs={"question": "Report the invoice."},
            gold={"status": "overdue", "amount": "120.00"},
        ),
        EvalRecord(
            id="s2",
            inputs={"question": "Report the payment."},
            gold={"status": "paid", "balance": "0.00"},
        ),
    ]


def test_scenario_suite_reports_binary_and_fractional_views(endpoint, tmp_path):
    # s1 fully correct; s2 half correct -> binary 0.5, micro 3/4, macro 0.75.
    client = FakeChatClient(
        {
            "invoice": json.dumps({"status": "overdue", "amount": "120.00"}),
            "payment": json.dumps({"status": "paid", "balance": "9.99"}),
        }
    )
    benchmark = ScenarioSuiteBenchmark(concurrency=1)

    with mock.patch(
        "foresightrlcore.evaluate.benchmarks.scenario_suite.ChatClient",
        return_value=client,
    ):
        outcome = benchmark.run(_ctx(endpoint, tmp_path, records=_scenarios()))

    assert outcome.sub_scores["binary"] == pytest.approx(0.5)
    assert outcome.sub_scores["fractional_micro"] == pytest.approx(0.75)
    assert outcome.sub_scores["fractional_macro"] == pytest.approx(0.75)
    assert outcome.score == outcome.sub_scores["binary"]
    assert outcome.n_samples == 2


def test_scenario_suite_fails_every_check_when_the_answer_is_not_json(
    endpoint, tmp_path
):
    client = FakeChatClient({"*": "I could not do that."})
    with mock.patch(
        "foresightrlcore.evaluate.benchmarks.scenario_suite.ChatClient",
        return_value=client,
    ):
        outcome = ScenarioSuiteBenchmark(concurrency=1).run(
            _ctx(endpoint, tmp_path, records=_scenarios())
        )

    assert outcome.score == 0.0
    assert outcome.n_errors == 0  # a bad answer is a failure, not a harness error


def test_scenario_suite_keeps_going_when_one_scenario_errors(endpoint, tmp_path):
    client = FakeChatClient(
        {"invoice": json.dumps({"status": "overdue", "amount": "120.00"})},
        fail_on=["payment"],
    )
    with mock.patch(
        "foresightrlcore.evaluate.benchmarks.scenario_suite.ChatClient",
        return_value=client,
    ):
        outcome = ScenarioSuiteBenchmark(
            concurrency=1, max_retries=1, base_delay=0
        ).run(_ctx(endpoint, tmp_path, records=_scenarios()))

    assert outcome.n_errors == 1
    assert outcome.n_samples == 1
    assert outcome.sub_scores["binary"] == 1.0


def test_scenario_suite_numeric_grader_allows_a_tolerance():
    benchmark = ScenarioSuiteBenchmark(grader="numeric", tolerance=0.05)
    record = EvalRecord(id="s", gold={"amount": "10.00"})

    passed, total, _ = benchmark.grade(record, {"amount": "10.02"})
    assert (passed, total) == (1, 1)
    assert benchmark.grade(record, {"amount": "11.00"})[0] == 0


def test_scenario_suite_rejects_an_unknown_grader():
    with pytest.raises(ValueError, match="unknown grader"):
        ScenarioSuiteBenchmark(grader="vibes")


def test_scenario_suite_without_records_says_so(endpoint, tmp_path):
    outcome = ScenarioSuiteBenchmark().run(_ctx(endpoint, tmp_path))
    assert "no records" in outcome.error


# ---------------------------------------------------------------------------
# external_process
# ---------------------------------------------------------------------------


def test_contract_vars_are_expanded_in_the_command_line():
    assert expand("--out $EVALUATE_OUTPUT_FILE", {"EVALUATE_OUTPUT_FILE": "/x"}) == (
        "--out /x"
    )
    # An unknown name is left visible rather than blanked, so a typo shows up.
    assert expand("$NOPE", {}) == "$NOPE"


def test_external_process_reads_the_result_file(endpoint, tmp_path):
    script = tmp_path / "run.py"
    script.write_text(
        "import json, os\n"
        "open(os.environ['EVALUATE_OUTPUT_FILE'], 'w').write(json.dumps("
        "{'score': 0.8, 'n_samples': 5, 'sub_scores': {'binary': 0.6}}))\n"
    )
    benchmark = ExternalProcessBenchmark(
        [sys.executable, str(script)], python=sys.executable
    )
    ctx = _ctx(endpoint, tmp_path)
    ctx.env.update({"EVALUATE_OUTPUT_FILE": str(ctx.output_file)})

    outcome = benchmark.run(ctx)

    assert outcome.score == 0.8
    assert outcome.sub_scores == {"binary": 0.6}
    assert not outcome.error


def test_external_process_reports_a_non_zero_exit_with_its_stderr(endpoint, tmp_path):
    script = tmp_path / "run.py"
    script.write_text("import sys; sys.stderr.write('it broke'); sys.exit(3)\n")

    outcome = ExternalProcessBenchmark([sys.executable, str(script)]).run(
        _ctx(endpoint, tmp_path)
    )

    assert "exited with code 3" in outcome.error
    assert "it broke" in outcome.detail["stderr_tail"]


def test_external_process_reports_a_timeout(endpoint, tmp_path):
    script = tmp_path / "run.py"
    script.write_text("import time; time.sleep(30)\n")

    outcome = ExternalProcessBenchmark([sys.executable, str(script)]).run(
        _ctx(endpoint, tmp_path, timeout_seconds=1)
    )

    assert "timed out" in outcome.error


def test_external_process_reports_a_command_that_cannot_start(endpoint, tmp_path):
    outcome = ExternalProcessBenchmark(["/nonexistent/binary"]).run(
        _ctx(endpoint, tmp_path)
    )
    assert "could not start" in outcome.error


def test_an_empty_command_is_rejected_at_construction():
    with pytest.raises(ValueError, match="non-empty command"):
        ExternalProcessBenchmark([])


def test_the_isolated_interpreter_is_taken_from_the_environment():
    benchmark = ExternalProcessBenchmark(["python", "-m", "runner"])
    argv = benchmark._resolve_command({"EVALUATE_PYTHON": "/venv/bin/python"})
    assert argv[0] == "/venv/bin/python"


def test_a_result_on_stdout_is_accepted_when_no_file_was_written(tmp_path):
    outcome = read_outcome(
        tmp_path / "missing.json",
        stdout='running...\n{"score": 0.5, "n_samples": 2}\n',
    )
    assert outcome.score == 0.5


def test_a_bare_number_is_accepted_as_a_score(tmp_path):
    path = tmp_path / "r.json"
    path.write_text("0.42")
    assert read_outcome(path).score == 0.42


@pytest.mark.parametrize(
    "body, message",
    [
        ("not json at all", "not valid JSON"),
        ('{"n_samples": 3}', "no 'score' field"),
        ('{"score": "high"}', "not a number"),
        ('["a"]', "must be an object or a number"),
    ],
)
def test_a_malformed_result_is_reported_clearly(tmp_path, body, message):
    path = tmp_path / "r.json"
    path.write_text(body)
    assert message in read_outcome(path).error


def test_no_result_at_all_is_reported(tmp_path):
    assert "produced no result" in read_outcome(tmp_path / "missing.json").error


# ---------------------------------------------------------------------------
# plugin
# ---------------------------------------------------------------------------


class _DemoSuite:
    def __init__(self, score: float = 0.6) -> None:
        self.score = score

    def run(self, ctx):
        return {"score": self.score, "n_samples": 3}


def test_plugin_instantiates_a_class_target(endpoint, tmp_path):
    benchmark = PluginBenchmark(f"{__name__}:_DemoSuite", init={"score": 0.9})
    outcome = benchmark.run(_ctx(endpoint, tmp_path))

    assert outcome.score == 0.9
    assert outcome.n_samples == 3


def test_plugin_accepts_a_plain_callable(endpoint, tmp_path):
    benchmark = PluginBenchmark(f"{__name__}:_demo_callable")
    assert benchmark.run(_ctx(endpoint, tmp_path)).score == 0.25


def _demo_callable(ctx):
    return HarnessOutcome(score=0.25, n_samples=1)


def test_an_unimportable_plugin_fails_its_benchmark_not_the_run(endpoint, tmp_path):
    outcome = PluginBenchmark("no.such.module:Thing").run(_ctx(endpoint, tmp_path))
    assert "could not load plugin target" in outcome.error


def test_a_raising_plugin_is_reported_as_a_benchmark_error(endpoint, tmp_path):
    outcome = PluginBenchmark(f"{__name__}:_raising_callable").run(
        _ctx(endpoint, tmp_path)
    )
    assert "RuntimeError" in outcome.error


def _raising_callable(ctx):
    raise RuntimeError("nope")


def test_import_target_accepts_both_path_spellings():
    assert import_target(f"{__name__}:_DemoSuite") is _DemoSuite
    assert import_target(f"{__name__}._DemoSuite") is _DemoSuite


@pytest.mark.parametrize("target", ["", "bare_name"])
def test_import_target_rejects_a_malformed_path(target):
    with pytest.raises(ValueError):
        import_target(target)


def test_coerce_outcome_accepts_dicts_numbers_and_outcomes():
    assert coerce_outcome(0.5).score == 0.5
    assert coerce_outcome({"score": 0.5, "unknown": 1}).score == 0.5
    assert coerce_outcome(HarnessOutcome(score=0.5)).score == 0.5
    with pytest.raises(TypeError):
        coerce_outcome("0.5")


# ---------------------------------------------------------------------------
# lm_eval
# ---------------------------------------------------------------------------


def test_lm_eval_points_the_harness_at_our_endpoint(endpoint, tmp_path):
    benchmark = LmEvalBenchmark(["gsm8k"])
    args = benchmark._model_args(_ctx(endpoint, tmp_path))

    assert args["base_url"] == f"{endpoint.base_url}/chat/completions"
    assert args["model"] == endpoint.model_id


def test_lm_eval_picks_a_known_metric_and_averages_across_tasks():
    outcome = LmEvalBenchmark(["a", "b"]).to_outcome(
        {
            "results": {
                "a": {"exact_match,none": 0.6},
                "b": {"acc,none": 0.8, "acc_stderr,none": 0.01},
            },
            "n-samples": {"a": 10, "b": 10},
        }
    )

    assert outcome.sub_scores == {"a": 0.6, "b": 0.8}
    assert outcome.score == pytest.approx(0.7)
    assert outcome.n_samples == 20


def test_lm_eval_reports_when_no_metric_was_recognised():
    outcome = LmEvalBenchmark(["a"]).to_outcome({"results": {"a": {"weird": "x"}}})
    assert "no numeric metric" in outcome.error


def test_lm_eval_reports_an_empty_result():
    assert "no results" in LmEvalBenchmark(["a"]).to_outcome({}).error


def test_lm_eval_without_the_package_is_actionable(endpoint, tmp_path):
    with mock.patch.dict(sys.modules, {"lm_eval": None}):
        outcome = LmEvalBenchmark(["gsm8k"]).run(_ctx(endpoint, tmp_path))
    assert "lm-eval" in outcome.error


def test_lm_eval_requires_at_least_one_task():
    with pytest.raises(ValueError, match="tasks"):
        LmEvalBenchmark([])


def test_choices_can_be_rendered_without_added_labels():
    """Datasets whose options already read `A. ...` must not be labelled twice."""
    from foresightrlcore.evaluate.benchmarks.mcq import McqBenchmark

    record = EvalRecord(
        id="r1",
        inputs={"question": "Which?"},
        gold="0",
        choices=["A. first option", "B. second option"],
    )

    labelled = McqBenchmark().render_choices(record)
    assert labelled == "A. A. first option\nB. B. second option"

    raw = McqBenchmark(label_choices=False).render_choices(record)
    assert raw == "A. first option\nB. second option"


def test_label_choices_is_readable_off_the_spec():
    from typing import cast

    from foresightrlcore.config.evaluate_schema import BenchmarkSpec
    from foresightrlcore.evaluate.benchmarks.mcq import McqBenchmark
    from foresightrlcore.evaluate.registry import BENCHMARK_TYPES

    spec = BenchmarkSpec.model_validate(
        {"name": "b", "type": "mcq", "params": {"label_choices": False}}
    )
    benchmark = cast(McqBenchmark, BENCHMARK_TYPES.get("mcq")(spec=spec))

    assert benchmark.label_choices is False


# ---------------------------------------------------------------------------
# Secondary metric verdicts
# ---------------------------------------------------------------------------


class _AlwaysUnreadable:
    """A metric that can tell an unreadable answer from a wrong one."""

    def score(self, record, prediction):
        from foresightrlcore.evaluate.interfaces import SampleScore

        return SampleScore(parse_failed=True, detail={"reason": "unreadable"})


class _AlwaysZero:
    """A containment-style metric: scores 0 and never flags a parse failure."""

    def score(self, record, prediction):
        from foresightrlcore.evaluate.interfaces import SampleScore

        return SampleScore(value=0.0, correct=False)


def test_a_secondary_metric_reports_whether_it_could_read_the_answer():
    """Keeping only the secondary metric's *value* throws away its verdict.

    The leaderboard metric scores an unreadable answer 0 rather than flagging
    it, so if the strict metric's parse verdict is dropped the run reports no
    malformed answers however many there were.
    """
    from foresightrlcore.evaluate.benchmarks.mcq import McqBenchmark
    from foresightrlcore.evaluate.interfaces import (
        PARSE_FAILED_SUFFIX,
        EvalRecord,
        Prediction,
    )

    bench = McqBenchmark(metrics=[_AlwaysZero(), _AlwaysUnreadable()])
    record = EvalRecord(id="r1", inputs={"question": "q"}, gold=0, choices=["a", "b"])

    verdict = bench.score(record, Prediction(text="waffle"))

    assert verdict.parse_failed is False, "the primary metric's verdict is unchanged"
    assert verdict.detail[f"_AlwaysUnreadable{PARSE_FAILED_SUFFIX}"] is True
