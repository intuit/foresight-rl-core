"""The runner: cross-product, endpoint bracketing, resume, trials, tolerance."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Dict, List
from unittest import mock

import pytest

from foresightrlcore.config.evaluate_schema import EvaluateJobConfig, LeaderboardInfo
from foresightrlcore.evaluate.registry import BENCHMARK_TYPES
from foresightrlcore.evaluate.runner import EvaluateRunner, cost_usd
from foresightrlcore.evaluate.spec import INVALID_SCORE
from foresightrlcore.inference.interfaces import InferenceEndpoint
from tests.evaluate.conftest import FakeChatClient

MCQ_RECORDS = [
    {
        "id": "q1",
        "question": "Capital of France?",
        "choices": ["Berlin", "Paris"],
        "answer": "Paris",
    },
    {
        "id": "q2",
        "question": "Largest ocean?",
        "choices": ["Pacific", "Arctic"],
        "answer": "Pacific",
    },
]


def _config(tmp_path, **overrides: Any) -> EvaluateJobConfig:
    run: Dict[str, Any] = {
        "benchmarks": [
            {
                "name": "mcq",
                "type": "mcq",
                "metrics": ["choice_match"],
                "dataset": {"loader": "inline", "records": MCQ_RECORDS},
            }
        ],
        "models": [
            {"name": "m1", "provider": "echo", "price_in": 1.0, "price_out": 2.0}
        ],
        "concurrency": 1,
        "reporters": [],
    }
    run.update(overrides)
    return EvaluateJobConfig.model_validate(
        {
            "job": {"name": "t"},
            "execution": {"type": "local"},
            "output": {},
            "evaluate": run,
        }
    )


@contextmanager
def _fake_endpoint(model):
    yield InferenceEndpoint(
        base_url="http://127.0.0.1:9/v1",
        api_key="k",
        model_id=model.name,
        model_name=model.name,
    )


def _runner(tmp_path, config=None, **kwargs) -> EvaluateRunner:
    return EvaluateRunner(
        config or _config(tmp_path),
        output_dir=tmp_path / "out",
        checkpoint_dir=tmp_path / "ckpt",
        workspace_root=tmp_path / "ws",
        endpoint_factory=_fake_endpoint,
        **kwargs,
    )


def _fingerprint(config, benchmark: str = "mcq", model: str = "m1") -> str:
    """The fingerprint the runner will expect for one (benchmark, model)."""
    from foresightrlcore.evaluate.catalog import load_catalog, resolve_benchmark
    from foresightrlcore.evaluate.runner import measurement_fingerprint

    catalog = load_catalog(config.evaluate.catalog_paths)
    spec = next(
        resolve_benchmark(entry, catalog)
        for entry in config.evaluate.benchmarks
        if entry.name == benchmark
    )
    candidate = next(m for m in config.evaluate.models if m.name == model)
    return measurement_fingerprint(spec, candidate)


def _run_with(runner, client):
    with mock.patch("foresightrlcore.evaluate.runner.ChatClient", return_value=client):
        return runner.run()


# ---------------------------------------------------------------------------
# Record tier
# ---------------------------------------------------------------------------


def test_a_record_tier_run_scores_every_record(tmp_path):
    result = _run_with(_runner(tmp_path), FakeChatClient({"France": "B", "ocean": "A"}))

    assert result.n_results == 1
    row = result.results[0]
    assert row["score"] == 1.0
    assert row["n_samples"] == 2
    assert len(result.samples) == 2


def test_per_record_samples_carry_cost_and_tokens(tmp_path):
    result = _run_with(_runner(tmp_path), FakeChatClient({"*": "A"}))

    sample = result.samples[0]
    # 10 prompt + 5 completion tokens at $1/$2 per 1K.
    assert sample["cost_usd"] == pytest.approx(cost_usd(10, 5, 1.0, 2.0))
    assert sample["prompt_tokens"] == 10


def test_a_wrong_answer_scores_zero_without_being_an_error(tmp_path):
    result = _run_with(_runner(tmp_path), FakeChatClient({"*": "A"}))

    row = result.results[0]
    assert row["score"] == 0.5  # q1 right, q2 wrong
    assert row["n_errors"] == 0


def test_an_endpoint_failure_becomes_an_error_row_not_a_zero(tmp_path):
    config = _config(tmp_path)
    config.evaluate.retry.max_retries = 1
    config.evaluate.retry.base_delay = 0
    # Default tolerance is 0.0, so any error invalidates the benchmark rather
    # than scoring it on the records that happened to work.
    result = _run_with(
        _runner(tmp_path, config),
        FakeChatClient({"*": "B"}, fail_on=["ocean"]),
    )

    row = result.results[0]
    assert row["n_errors"] == 1
    assert row["score"] == INVALID_SCORE
    assert result.aggregates[0]["valid"] is False


def test_errors_within_tolerance_still_produce_a_score(tmp_path):
    config = _config(tmp_path)
    config.evaluate.benchmarks[0].failure_tolerance = 0.5
    config.evaluate.retry.max_retries = 1
    config.evaluate.retry.base_delay = 0

    result = _run_with(
        _runner(tmp_path, config),
        FakeChatClient({"France": "B"}, fail_on=["ocean"]),
    )

    row = result.results[0]
    assert row["n_errors"] == 1
    assert row["score"] == 1.0  # the one scored record was correct


def test_parse_failures_are_reported_separately_from_accuracy(tmp_path):
    result = _run_with(_runner(tmp_path), FakeChatClient({"*": "no idea"}))

    row = result.results[0]
    assert row["sub_parse_failure_rate"] == 1.0
    assert row["sub_accuracy"] == 0.0


def test_truncated_generations_are_reported_rather_than_read_as_wrong(tmp_path):
    # A run cut off at max_tokens scores like a bad model. The rate is what says
    # the budget was the problem, so it has to reach the result.
    result = _run_with(
        _runner(tmp_path),
        FakeChatClient({"*": "no idea"}, finish_reason="length"),
    )

    assert result.results[0]["sub_truncation_rate"] == 1.0
    assert all(s["finish_reason"] == "length" for s in result.samples)


def test_a_complete_generation_reports_no_truncation(tmp_path):
    result = _run_with(_runner(tmp_path), FakeChatClient({"*": "A"}))

    assert result.results[0]["sub_truncation_rate"] == 0.0


# ---------------------------------------------------------------------------
# Cross-product and model selection
# ---------------------------------------------------------------------------


def test_every_benchmark_runs_against_every_model(tmp_path):
    config = _config(tmp_path)
    config.evaluate.models.append(
        type(config.evaluate.models[0])(name="m2", provider="echo")
    )
    result = _run_with(_runner(tmp_path, config), FakeChatClient({"*": "B"}))

    assert {r["model"] for r in result.results} == {"m1", "m2"}


def test_a_benchmark_can_select_a_model_group(tmp_path):
    config = _config(tmp_path)
    model_cls = type(config.evaluate.models[0])
    config.evaluate.models.append(model_cls(name="m2", provider="echo"))
    config.evaluate.model_groups = {"small": ["m2"]}
    config.evaluate.benchmarks[0].models = ["small"]

    result = _run_with(_runner(tmp_path, config), FakeChatClient({"*": "B"}))

    assert {r["model"] for r in result.results} == {"m2"}


def test_selecting_an_unknown_model_is_reported(tmp_path):
    config = _config(tmp_path)
    config.evaluate.benchmarks[0].models = ["ghost"]

    with pytest.raises(ValueError, match="unknown model"):
        config.evaluate.resolve_models_for(config.evaluate.benchmarks[0])


def test_the_endpoint_is_brought_up_once_per_model(tmp_path):
    starts: List[str] = []

    @contextmanager
    def counting_factory(model):
        starts.append(model.name)
        with _fake_endpoint(model) as endpoint:
            yield endpoint

    config = _config(tmp_path)
    config.evaluate.benchmarks.append(
        config.evaluate.benchmarks[0].model_copy(update={"name": "mcq2"})
    )
    runner = EvaluateRunner(
        config,
        output_dir=tmp_path / "out",
        checkpoint_dir=tmp_path / "ckpt",
        workspace_root=tmp_path / "ws",
        endpoint_factory=counting_factory,
    )

    _run_with(runner, FakeChatClient({"*": "B"}))

    # Two benchmarks, one model: one endpoint lifecycle, not two.
    assert starts == ["m1"]


# ---------------------------------------------------------------------------
# Trials and confidence
# ---------------------------------------------------------------------------


def test_multiple_trials_aggregate_into_an_interval(tmp_path):
    config = _config(tmp_path, trials=3)
    result = _run_with(_runner(tmp_path, config), FakeChatClient({"*": "B"}))

    assert result.n_results == 3
    aggregate = result.aggregates[0]
    assert aggregate["trials"] == 3
    assert aggregate["scores"] == [0.5, 0.5, 0.5]
    assert aggregate["ci_level"] == 0.95


def test_a_per_benchmark_trial_count_overrides_the_run_default(tmp_path):
    config = _config(tmp_path, trials=1)
    config.evaluate.benchmarks[0].trials = 2

    result = _run_with(_runner(tmp_path, config), FakeChatClient({"*": "B"}))

    assert result.n_results == 2


def test_targeted_retrials_only_touch_benchmarks_that_disagreed(tmp_path):
    """Stable benchmarks are already measured; repeating them buys nothing."""
    config = _config(tmp_path, trials=2)
    config.evaluate.confidence.retrial_flaky = True
    config.evaluate.confidence.max_retrials = 1

    result = _run_with(_runner(tmp_path, config), FakeChatClient({"*": "B"}))

    # The echo-backed benchmark is deterministic, so no extra trial is spent.
    assert result.n_results == 2


def test_an_extra_trial_is_checkpointed_like_any_other(tmp_path):
    """A retrial costs what a trial costs, so losing it costs the same.

    Retrials were run but never written to the checkpoint, so a resumed run
    re-ran every one of them from nothing however much had already been spent
    on them -- and, because they are spent precisely on the benchmarks that
    disagreed, they are the expensive ones.
    """
    scores = iter([0.2, 0.9, 0.5])

    class Flaky:
        def run(self, ctx):
            from foresightrlcore.evaluate.interfaces import HarnessOutcome

            value = next(scores)
            return HarnessOutcome(score=value, n_samples=1, sub_scores={"part": value})

    config = _config(
        tmp_path,
        benchmarks=[{"name": "flaky", "type": "_retrial_ckpt_type"}],
        trials=2,
    )
    config.evaluate.confidence.retrial_flaky = True
    config.evaluate.confidence.max_retrials = 1

    BENCHMARK_TYPES.register("_retrial_ckpt_type")(lambda **_: Flaky())
    try:
        _run_with(_runner(tmp_path, config), FakeChatClient())
    finally:
        BENCHMARK_TYPES._items.pop("_retrial_ckpt_type", None)

    rows = [
        json.loads(line)
        for line in (tmp_path / "ckpt" / "evaluate_checkpoint.jsonl")
        .read_text()
        .splitlines()
    ]
    trials = sorted(row["trial"] for row in rows)
    assert trials == [0, 1, 2], f"the extra trial was never checkpointed: {trials}"
    assert all(row.get("resume_key") for row in rows)


def test_an_unstable_benchmark_earns_extra_trials(tmp_path):
    scores = iter([0.2, 0.9, 0.5])

    class Flaky:
        def run(self, ctx):
            from foresightrlcore.evaluate.interfaces import HarnessOutcome

            value = next(scores)
            return HarnessOutcome(score=value, n_samples=1, sub_scores={"part": value})

    config = _config(
        tmp_path,
        benchmarks=[{"name": "flaky", "type": "_flaky_test_type"}],
        trials=2,
    )
    config.evaluate.confidence.retrial_flaky = True
    config.evaluate.confidence.max_retrials = 1

    BENCHMARK_TYPES.register("_flaky_test_type")(lambda **_: Flaky())
    try:
        result = _run_with(_runner(tmp_path, config), FakeChatClient())
    finally:
        BENCHMARK_TYPES._items.pop("_flaky_test_type", None)

    assert result.n_results == 3
    assert result.aggregates[0]["trials"] == 3


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------


def test_a_completed_trial_is_not_run_again(tmp_path):
    client = FakeChatClient({"France": "B", "ocean": "A"})
    _run_with(_runner(tmp_path), client)
    calls_after_first = len(client.calls)

    second = FakeChatClient({"France": "B", "ocean": "A"})
    result = _run_with(_runner(tmp_path), second)

    assert calls_after_first == 2
    assert second.calls == []  # nothing re-evaluated
    assert result.n_results == 1  # the prior result is carried forward
    assert result.results[0]["score"] == 1.0


def test_records_scored_before_a_crash_are_reused(tmp_path):
    """Losing a whole trial to a restart is the thing resume must prevent."""
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "evaluate_samples.jsonl").write_text(
        json.dumps(
            {
                "benchmark": "mcq",
                "model": "m1",
                "fingerprint": _fingerprint(_config(tmp_path)),
                "trial": 0,
                "record_id": "q1",
                "score": 1.0,
                "correct": True,
                "parse_failed": False,
                "error": "",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "latency_ms": 0.0,
                "cost_usd": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    client = FakeChatClient({"*": "A"})
    result = _run_with(_runner(tmp_path), client)

    # Only q2 was re-sent; q1 came from the checkpoint.
    assert len(client.calls) == 1
    assert result.results[0]["n_samples"] == 2


def test_records_are_checkpointed_as_they_are_scored(tmp_path):
    """A record must reach the checkpoint when it is scored, not when the
    trial ends.

    A trial is the unit of reporting, not of work: one benchmark over a large
    dataset is a single trial that can run for hours. Checkpointing it only on
    completion means an interruption discards every record it generated, which
    is what a live test found -- a 1,000-record run held nothing in S3 for
    twenty minutes and then wrote everything at once. The counter also
    advanced once per *trial* despite being named ``every_n_rows``, so a run
    with fewer trials than that number never synced early at all.

    Counting lines at each sync separates the two behaviours without depending
    on thread timing: the callback appends and syncs back to back, so the
    first sync sees exactly one sample. Under the old code the first sync came
    after the trial and saw both.
    """
    from foresightrlcore.common.io import CheckpointWriter

    lines_at_sync: List[int] = []
    real_sync = CheckpointWriter.sync

    def spy(self: CheckpointWriter) -> None:
        if self.path.name == "evaluate_samples.jsonl" and self.path.exists():
            lines_at_sync.append(len(self.path.read_text().splitlines()))
        real_sync(self)

    config = _config(tmp_path, checkpoint={"enabled": True, "every_n_rows": 1})
    with mock.patch.object(CheckpointWriter, "sync", spy):
        _run_with(_runner(tmp_path, config), FakeChatClient({"*": "A"}))

    assert lines_at_sync, "the samples checkpoint was never synced"
    assert lines_at_sync[0] == 1, (
        "the first sync should follow the first scored record; seeing both "
        f"records already written means the trial finished first: {lines_at_sync}"
    )
    assert lines_at_sync[-1] == 2


def _sample_row(config, record_id: str, *, error: str = "", **over: Any) -> str:
    """One checkpointed sample row, as the runner would have written it."""
    row: Dict[str, Any] = {
        "benchmark": "mcq",
        "model": "m1",
        "fingerprint": _fingerprint(config),
        "trial": 0,
        "record_id": record_id,
        "score": 0.0 if error else 1.0,
        "correct": not error,
        "parse_failed": False,
        "error": error,
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "latency_ms": 0.0,
        "cost_usd": 0.0,
    }
    row.update(over)
    return json.dumps(row) + "\n"


def test_a_record_that_errored_is_retried_on_resume(tmp_path):
    """An error is a record we failed to measure, not a measurement.

    Reusing it means never measuring it: the runner exhausts its retries,
    checkpoints the failure, and every later resume skips that record because
    a row exists for it. A transient gateway outage then outlives itself and
    reappears in the results of every resumed run. The GenOS sweeps hit this,
    and the documented remedy -- start from a fresh checkpoint prefix --
    discards every good record to re-reach a handful of bad ones.
    """
    config = _config(tmp_path)
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "evaluate_samples.jsonl").write_text(
        _sample_row(config, "q1")
        + _sample_row(config, "q2", error="RuntimeError: gateway timeout"),
        encoding="utf-8",
    )

    client = FakeChatClient({"*": "A"})
    result = _run_with(_runner(tmp_path, config), client)

    # q1 was measured and is reused; q2 was not, so it is re-sent.
    assert len(client.calls) == 1
    assert result.results[0]["n_samples"] == 2
    assert result.results[0]["n_errors"] == 0


def test_a_trial_that_finished_with_errors_is_not_treated_as_done(tmp_path):
    """The record-level fix is useless if the trial is skipped wholesale.

    Resume works at two levels: a completed trial is skipped entirely, and
    within a replanned trial the finished records are reused. A trial that
    errored is checkpointed like any other, so without this the trial never
    gets replanned and its errored records are never reconsidered.
    """
    config = _config(tmp_path)
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "evaluate_samples.jsonl").write_text(
        _sample_row(config, "q1")
        + _sample_row(config, "q2", error="RuntimeError: gateway timeout"),
        encoding="utf-8",
    )
    (checkpoint_dir / "evaluate_checkpoint.jsonl").write_text(
        json.dumps(
            {
                "benchmark": "mcq",
                "model": "m1",
                "fingerprint": _fingerprint(config),
                "trial": 0,
                "resume_key": "mcq|m1|0",
                "score": 0.5,
                "n_samples": 1,
                "n_errors": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    client = FakeChatClient({"*": "A"})
    result = _run_with(_runner(tmp_path, config), client)

    assert len(client.calls) == 1, "the damaged trial was skipped as if complete"
    # One result, not two: the stale damaged row must not survive alongside
    # the fresh one.
    assert result.n_results == 1
    assert result.results[0]["n_errors"] == 0


def test_a_repaired_trial_stops_being_treated_as_damaged(tmp_path):
    """Replanning a damaged trial must end, not recur forever.

    The checkpoint is append-only, so re-running a trial that errored leaves
    the old errored row sitting behind its clean replacement. Judging damage
    from any matching row rather than the latest one means the trial is
    replanned on every future resume, for the life of the prefix -- found by
    running the errored-record smoke twice, where the second run still
    reported "re-running 1 trial(s) that finished with errors" against a trial
    it had already repaired.
    """
    config = _config(tmp_path)
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "evaluate_samples.jsonl").write_text(
        _sample_row(config, "q1") + _sample_row(config, "q2"),
        encoding="utf-8",
    )

    def trial_row(n_errors: int) -> str:
        return (
            json.dumps(
                {
                    "benchmark": "mcq",
                    "model": "m1",
                    "fingerprint": _fingerprint(config),
                    "trial": 0,
                    "resume_key": "mcq|m1|0",
                    "score": 1.0,
                    "n_samples": 2,
                    "n_errors": n_errors,
                }
            )
            + "\n"
        )

    # The damaged row, then the clean row that superseded it.
    (checkpoint_dir / "evaluate_checkpoint.jsonl").write_text(
        trial_row(1) + trial_row(0), encoding="utf-8"
    )

    client = FakeChatClient({"*": "A"})
    result = _run_with(_runner(tmp_path, config), client)

    assert not client.calls, "a repaired trial was replanned again"
    assert (
        result.n_results == 1
    ), "the superseded row was reported alongside its replacement"


def test_duplicate_record_ids_do_not_collapse_into_one_record(tmp_path):
    """Two questions sharing an id must still be two questions.

    Resume is keyed on the record id, so a dataset that reuses one makes the
    two records indistinguishable: the first one's checkpointed result is
    reused for both and the second is never measured, while the sample count
    still reads full. The `logiqa2` catalog benchmark does exactly this --
    found when a resumed 1000-record run reported 1000 samples but carried
    999 rows, having scored id 10591 twice and one other record never.
    """
    config = _config(
        tmp_path,
        benchmarks=[
            {
                "name": "dupes",
                "type": "mcq",
                "metrics": ["choice_match"],
                "dataset": {
                    "loader": "inline",
                    "records": [
                        {
                            "id": "x",
                            "question": "First?",
                            "choices": ["A", "B"],
                            "answer": "A",
                        },
                        {
                            "id": "x",
                            "question": "Second?",
                            "choices": ["A", "B"],
                            "answer": "B",
                        },
                    ],
                },
            }
        ],
    )

    result = _run_with(_runner(tmp_path, config), FakeChatClient({"*": "A"}))

    ids = [s["record_id"] for s in result.samples]
    assert sorted(ids) == ["x", "x#2"], f"the two records were conflated: {ids}"
    assert result.results[0]["n_samples"] == 2


def test_a_rescored_record_is_reported_once(tmp_path):
    """The report should reflect the run, not the checkpoint's history.

    The samples checkpoint is append-only, so a record scored more than once
    -- after a repair, or because a checkpoint predates a change in how ids
    are assigned -- leaves every attempt on disk. Seeding the report from all
    of them reports the record twice and inflates the sample count.
    """
    config = _config(tmp_path)
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "evaluate_samples.jsonl").write_text(
        _sample_row(config, "q1", score=0.0, correct=False)
        + _sample_row(config, "q1", score=1.0, correct=True)
        + _sample_row(config, "q2"),
        encoding="utf-8",
    )

    result = _run_with(_runner(tmp_path, config), FakeChatClient({"*": "A"}))

    ids = [s["record_id"] for s in result.samples]
    assert sorted(ids) == ["q1", "q2"], f"a re-scored record was reported twice: {ids}"
    # The later row is the current one.
    assert [s for s in result.samples if s["record_id"] == "q1"][0]["score"] == 1.0


def test_checkpoint_rows_carry_a_resume_key(tmp_path):
    _run_with(_runner(tmp_path), FakeChatClient({"*": "B"}))

    rows = [
        json.loads(line)
        for line in (tmp_path / "ckpt" / "evaluate_checkpoint.jsonl")
        .read_text()
        .splitlines()
    ]
    assert rows[0]["resume_key"] == "mcq|m1|0"


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


def test_an_unbuildable_benchmark_fails_alone(tmp_path):
    config = _config(
        tmp_path,
        benchmarks=[
            {"name": "broken", "type": "plugin"},  # no target -> cannot run
            {
                "name": "mcq",
                "type": "mcq",
                "metrics": ["choice_match"],
                "dataset": {"loader": "inline", "records": MCQ_RECORDS},
            },
        ],
    )

    result = _run_with(
        _runner(tmp_path, config), FakeChatClient({"France": "B", "ocean": "A"})
    )

    by_name = {r["benchmark"]: r for r in result.results}
    assert by_name["broken"]["score"] == INVALID_SCORE
    assert by_name["mcq"]["score"] == 1.0


def test_a_benchmark_with_no_dataset_is_reported_not_crashed(tmp_path):
    config = _config(tmp_path, benchmarks=[{"name": "empty", "type": "mcq"}])

    result = _run_with(_runner(tmp_path, config), FakeChatClient())

    assert "no records" in result.results[0]["error"]


def test_a_failing_reporter_does_not_lose_the_run(tmp_path):
    from foresightrlcore.config.evaluate_schema import ReporterSpec

    config = _config(tmp_path)
    config.evaluate.reporters = [
        ReporterSpec(type="_exploding_test_reporter"),
        ReporterSpec(type="json"),
    ]

    from foresightrlcore.evaluate.registry import REPORTERS

    class Exploding:
        def report(self, result, output_dir):
            raise RuntimeError("sink down")

    REPORTERS.register("_exploding_test_reporter")(lambda: Exploding())
    try:
        _run_with(_runner(tmp_path, config), FakeChatClient({"*": "B"}))
    finally:
        REPORTERS._items.pop("_exploding_test_reporter", None)

    assert (tmp_path / "out" / "evaluate_results.json").exists()


def test_a_reporter_receives_its_configured_params(tmp_path):
    """Reporters were previously constructed with no arguments, so a sink
    needing an address to publish to could not be given one from YAML."""
    from foresightrlcore.config.evaluate_schema import ReporterSpec
    from foresightrlcore.evaluate.registry import REPORTERS

    seen = {}

    class Recording:
        def __init__(self, **params):
            seen.update(params)

        def report(self, result, output_dir):
            pass

    config = _config(tmp_path)
    config.evaluate.reporters = [
        ReporterSpec(type="_recording_test_reporter", params={"endpoint": "e2e"})
    ]

    REPORTERS.register("_recording_test_reporter")(Recording)
    try:
        _run_with(_runner(tmp_path, config), FakeChatClient({"*": "B"}))
    finally:
        REPORTERS._items.pop("_recording_test_reporter", None)

    assert seen == {"endpoint": "e2e"}


def test_the_result_document_carries_each_models_report_label(tmp_path):
    config = _config(tmp_path)
    config.evaluate.models[0].display_name = "Model One"

    result = _run_with(_runner(tmp_path, config), FakeChatClient({"*": "B"}))

    assert result.label("m1") == "Model One"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_the_summary_reports_a_mean_per_model(tmp_path):
    result = _run_with(_runner(tmp_path), FakeChatClient({"France": "B", "ocean": "A"}))

    summary = result.summary
    assert summary["n_pairs"] == 1
    assert summary["by_model"] == [{"model": "m1", "benchmarks": 1, "mean_score": 1.0}]


# ---------------------------------------------------------------------------
# Staged weights
#
# The launcher mounts S3 weights as input channels; the runner has to point
# `vllm serve` at the mount instead of the original URI. Getting this wrong
# only shows up inside a GPU job, so it is pinned here.
# ---------------------------------------------------------------------------


def _local_model(name, **kw):
    from foresightrlcore.config.evaluate_schema import EvaluateModelConfig

    return EvaluateModelConfig(name=name, provider="local", **kw)


def _config_with(models):
    from foresightrlcore.config.evaluate_schema import EvaluateJobConfig

    return EvaluateJobConfig.model_validate(
        {
            "job": {"name": "j"},
            "execution": {"type": "local"},
            "evaluate": {
                "benchmarks": [{"name": "b", "type": "mcq"}],
                "models": [m.model_dump() for m in models],
            },
        }
    )


def test_staged_weights_are_served_without_renaming_the_candidate(
    tmp_path, monkeypatch
):
    from foresightrlcore.evaluate.runner import mount_staged_model_weights

    mount = tmp_path / "model"
    mount.mkdir()
    monkeypatch.setenv("SM_CHANNEL_MODEL", str(mount))

    config = _config_with([_local_model("s3://bucket/qwen/")])
    mount_staged_model_weights(config)

    model = config.evaluate.models[0]
    assert model.serving_path == str(mount)
    assert model.name == "s3://bucket/qwen/", "the leaderboard label must not move"


def test_several_local_models_each_get_their_own_channel(tmp_path, monkeypatch):
    from foresightrlcore.evaluate.runner import mount_staged_model_weights

    first, second = tmp_path / "model", tmp_path / "model1"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv("SM_CHANNEL_MODEL", str(first))
    monkeypatch.setenv("SM_CHANNEL_MODEL1", str(second))

    config = _config_with(
        [_local_model("s3://bucket/a/"), _local_model("s3://bucket/b/")]
    )
    mount_staged_model_weights(config)

    assert config.evaluate.models[0].serving_path == str(first)
    assert config.evaluate.models[1].serving_path == str(second)


def test_a_hub_id_is_left_alone(tmp_path, monkeypatch):
    """Only S3 candidates are staged; an HF id must reach vLLM untouched."""
    from foresightrlcore.evaluate.runner import mount_staged_model_weights

    monkeypatch.setenv("SM_CHANNEL_MODEL", str(tmp_path))

    config = _config_with([_local_model("Qwen/Qwen3-0.6B")])
    mount_staged_model_weights(config)

    assert config.evaluate.models[0].serving_path == "Qwen/Qwen3-0.6B"


def test_nothing_happens_off_sagemaker(monkeypatch):
    from foresightrlcore.evaluate.runner import mount_staged_model_weights

    monkeypatch.delenv("SM_CHANNEL_MODEL", raising=False)
    config = _config_with([_local_model("s3://bucket/qwen/")])
    mount_staged_model_weights(config)

    assert config.evaluate.models[0].serving_path == "s3://bucket/qwen/"


def test_the_endpoint_serves_the_staged_path(tmp_path):
    """The launcher/runner handshake has to reach `vllm serve` itself."""
    from unittest import mock

    from foresightrlcore.config.perf_schema import VllmServingConfig
    from foresightrlcore.inference import endpoint_for

    model = _local_model(
        "s3://bucket/qwen/", serving=VllmServingConfig(startup_timeout=60)
    )
    model.weights_path = str(tmp_path)

    with mock.patch(
        "foresightrlcore.inference.local_vllm.VllmServerEndpoint"
    ) as endpoint:
        with endpoint_for(model):
            pass

    assert endpoint.call_args.args[0] == str(tmp_path)
    assert endpoint.call_args.kwargs["model_name"] == "s3://bucket/qwen/"


# ---------------------------------------------------------------------------
# Checkpoint invalidation
#
# Resume is keyed on names, so without a fingerprint an edit to a prompt or a
# metric leaves the old scores looking current: the re-run replays them and
# reports numbers produced by a configuration that no longer exists. This was
# not hypothetical -- it silently returned stale SageMaker scores twice.
# ---------------------------------------------------------------------------


def test_changing_the_metric_invalidates_the_checkpoint(tmp_path):
    _run_with(_runner(tmp_path), FakeChatClient({"France": "B", "ocean": "A"}))

    changed = _config(tmp_path)
    changed.evaluate.benchmarks[0].metrics = ["exact_match"]
    client = FakeChatClient({"France": "B", "ocean": "A"})
    result = _run_with(_runner(tmp_path, changed), client)

    assert client.calls, "a changed metric must re-run, not replay"
    assert result.n_results == 1, "the stale trial must not also be reported"


def test_changing_the_dataset_invalidates_the_checkpoint(tmp_path):
    _run_with(_runner(tmp_path), FakeChatClient({"France": "B", "ocean": "A"}))

    changed = _config(tmp_path)
    dataset = changed.evaluate.benchmarks[0].dataset
    assert dataset is not None
    dataset.limit = 1
    client = FakeChatClient({"France": "B", "ocean": "A"})
    _run_with(_runner(tmp_path, changed), client)

    assert client.calls, "a changed dataset must re-run"


def test_an_unchanged_config_still_resumes(tmp_path):
    """The fix must not defeat resume, which is the point of the checkpoint."""
    _run_with(_runner(tmp_path), FakeChatClient({"France": "B", "ocean": "A"}))

    second = FakeChatClient({"France": "B", "ocean": "A"})
    result = _run_with(_runner(tmp_path, _config(tmp_path)), second)

    assert second.calls == []
    assert result.n_results == 1


def test_a_result_for_a_dropped_benchmark_is_not_reported(tmp_path):
    """A checkpoint outlives the config that made it; it must not leak in."""
    _run_with(_runner(tmp_path), FakeChatClient({"France": "B", "ocean": "A"}))

    other = _config(tmp_path)
    other.evaluate.benchmarks[0].name = "mcq_renamed"
    result = _run_with(
        _runner(tmp_path, other), FakeChatClient({"France": "B", "ocean": "A"})
    )

    assert {row["benchmark"] for row in result.results} == {"mcq_renamed"}


def test_the_fingerprint_ignores_runtime_weight_staging(tmp_path):
    """`weights_path` is filled in by SageMaker; it is not part of the measurement."""
    from foresightrlcore.evaluate.catalog import load_catalog, resolve_benchmark
    from foresightrlcore.evaluate.runner import measurement_fingerprint

    config = _config(tmp_path)
    spec = resolve_benchmark(config.evaluate.benchmarks[0], load_catalog())
    model = config.evaluate.models[0]

    before = measurement_fingerprint(spec, model)
    model.weights_path = "/opt/ml/input/data/model"

    assert measurement_fingerprint(spec, model) == before


def test_relabelling_a_candidate_keeps_the_scores_it_already_has(tmp_path):
    """`display_name` and `leaderboard` decide presentation, not measurement."""
    from foresightrlcore.evaluate.catalog import load_catalog, resolve_benchmark
    from foresightrlcore.evaluate.runner import measurement_fingerprint

    config = _config(tmp_path)
    spec = resolve_benchmark(config.evaluate.benchmarks[0], load_catalog())
    model = config.evaluate.models[0]

    before = measurement_fingerprint(spec, model)
    model.display_name = "Something Readable"
    model.leaderboard = LeaderboardInfo(model_size="8B")

    assert measurement_fingerprint(spec, model) == before


def test_changing_how_a_candidate_is_sampled_invalidates_its_scores(tmp_path):
    """The opposite case: `top_p` changes the answers, so the old ones must go."""
    from foresightrlcore.evaluate.catalog import load_catalog, resolve_benchmark
    from foresightrlcore.evaluate.runner import measurement_fingerprint

    config = _config(tmp_path)
    spec = resolve_benchmark(config.evaluate.benchmarks[0], load_catalog())
    model = config.evaluate.models[0]

    before = measurement_fingerprint(spec, model)
    model.top_p = 0.95

    assert measurement_fingerprint(spec, model) != before


# ---------------------------------------------------------------------------
# Valid-answer rate
# ---------------------------------------------------------------------------


def test_valid_answer_rate_comes_from_a_metric_that_can_tell():
    """`parse_failure_rate` alone reported 0% on a run that was 86% garbage.

    It is computed from the benchmark's *reported* metric, which for the
    leaderboard suite is a containment rule that cannot distinguish an
    unreadable answer from a wrong one. The strict metric running alongside it
    can, and its verdict has to reach the report or the run looks healthy.
    """
    from foresightrlcore.evaluate.interfaces import PARSE_FAILED_SUFFIX
    from foresightrlcore.evaluate.runner import _secondary_metric_sub_scores
    from foresightrlcore.evaluate.spec import SampleResult

    key = "ChoiceMatchMetric"
    samples = [
        SampleResult(
            benchmark="b",
            model="m",
            record_id=str(i),
            score=0.0,
            parse_failed=False,  # the legacy metric never flags
            detail={
                key: 1.0 if i < 2 else 0.0,
                f"{key}{PARSE_FAILED_SUFFIX}": i >= 8,
                "gold_index": 3,  # numeric detail that is not a metric
            },
        )
        for i in range(10)
    ]

    sub = _secondary_metric_sub_scores(samples)

    assert sub[f"{key}_valid_rate"] == pytest.approx(0.8)
    assert sub[f"{key}_score"] == pytest.approx(0.2)
    assert "gold_index_score" not in sub, "plain detail is not a metric"
    assert "gold_index_valid_rate" not in sub
