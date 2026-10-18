"""Evaluating a reasoning model: sampling controls and scratchpad removal.

Two defects motivate this file, both reported from a real run against a
thinking model at ``temperature=1, top_p=0.95, top_k=64``:

* the candidate config had no way to express ``top_p`` / ``top_k``, so the run
  could not reproduce the sampling the model was tuned for; and
* the answer extractor read the model's scratchpad, where a reasoning model
  writes the answers it considered and rejected -- so a correct response could
  score zero.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict
from unittest import mock

import pytest

from foresightrlcore.common.thinking import DEFAULT_THINKING_TAGS, strip_thinking
from foresightrlcore.config.evaluate_schema import (
    EvaluateJobConfig,
    EvaluateModelConfig,
)
from foresightrlcore.evaluate.interfaces import EvalRecord, Prediction
from foresightrlcore.evaluate.metrics.choice_match import ChoiceMatchMetric
from foresightrlcore.evaluate.runner import EvaluateRunner
from foresightrlcore.inference.interfaces import InferenceEndpoint
from tests.evaluate.conftest import FakeChatClient

CHOICES = ["Accounts Payable", "Accounts Receivable", "Deferred Revenue"]


def _record(gold: Any = 1) -> EvalRecord:
    return EvalRecord(id="r1", gold=gold, choices=CHOICES)


# ---------------------------------------------------------------------------
# strip_thinking
# ---------------------------------------------------------------------------


def test_balanced_block_is_removed():
    assert strip_thinking("<think>noise</think>B") == "B"
    assert strip_thinking("<reasoning>noise</reasoning>B") == "B"


def test_every_block_is_removed_not_just_the_first():
    text = "<think>one</think>A<think>two</think>B"
    assert strip_thinking(text) == "AB"


def test_a_closing_tag_without_an_opener_leaves_only_the_tail():
    # Some chat templates pre-fill the opening tag, so it never appears in the
    # completion and only the closer marks where the answer starts.
    assert strip_thinking("deliberating</think>Answer: C") == "Answer: C"
    assert strip_thinking("a</think>b</think>final") == "final"


def test_an_opening_tag_without_a_closer_yields_nothing():
    # Generation hit the token cap mid-thought. There is no answer, and saying
    # so beats grading the scratchpad.
    assert strip_thinking("<think>still weighing A against C") == ""


def test_tags_are_matched_regardless_of_case_or_attributes():
    assert strip_thinking("<THINK>x</THINK>B") == "B"
    assert strip_thinking('<think id="1">x</think>B') == "B"


def test_custom_tags_replace_the_defaults_rather_than_adding_to_them():
    text = "<analysis>hidden</analysis>B"
    assert strip_thinking(text, ["analysis"]) == "B"
    # Naming a tag set is a statement about this model, so `<think>` is no
    # longer assumed to be reasoning.
    assert strip_thinking("<think>x</think>B", ["analysis"]) == "<think>x</think>B"


def test_tags_may_be_written_with_angle_brackets():
    assert strip_thinking("<muse>x</muse>B", ["<muse>"]) == "B"
    assert strip_thinking("<muse>x</muse>B", ["</muse>"]) == "B"


def test_text_without_tags_is_returned_unchanged():
    assert strip_thinking("plain answer") == "plain answer"
    assert strip_thinking("") == ""
    assert strip_thinking("a < b and c > d") == "a < b and c > d"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_choice_match_ignores_an_answer_the_model_rejected():
    """The reported defect: the scratchpad's discarded answer won."""
    response = (
        "<think>The answer is A, since payables... no -- the customer owes us, "
        "so it is receivable.</think>\nAnswer: B"
    )
    verdict = ChoiceMatchMetric().score(_record(), Prediction(text=response))
    assert verdict.correct and verdict.value == 1.0


def test_choice_match_reports_a_truncated_reasoning_response_as_unparseable():
    verdict = ChoiceMatchMetric().score(
        _record(), Prediction(text="<think>Maybe A, or possibly C, let me")
    )
    assert verdict.parse_failed
    assert not verdict.correct


def test_choice_match_still_reads_an_answer_that_mentions_no_tags():
    verdict = ChoiceMatchMetric().score(_record(), Prediction(text="Answer: B"))
    assert verdict.correct


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _model(**overrides: Any) -> EvaluateModelConfig:
    return EvaluateModelConfig.model_validate({"name": "m", **overrides})


def test_sampling_controls_default_to_unset_so_a_provider_keeps_its_own():
    model = _model()
    assert model.top_p is None and model.top_k is None


def test_sampling_controls_are_range_checked():
    assert _model(top_p=0.95).top_p == 0.95
    assert _model(top_k=64).top_k == 64
    for bad in ({"top_p": 1.5}, {"top_p": 0.0}, {"top_k": 0}):
        with pytest.raises(ValueError):
            _model(**bad)


# ---------------------------------------------------------------------------
# \boxed{} answers -- the DeepSeek-R1 house style
# ---------------------------------------------------------------------------
#
# Found by a live smoke, not by review: R1-Distill scored 0.10 against an
# 8B non-reasoning model's 0.46, because every one of its answers was
# unreadable. On a leaderboard an unparseable answer and a wrong one look
# identical, so this failed silently and looked like a weak model.


@pytest.mark.parametrize(
    "response,expected",
    [
        (r"The final answer is \boxed{B}.", "B"),
        (r"\boxed{B}", "B"),
        (r"\boxed{\text{B}}", "B"),
        (r"So \boxed{$B$}", "B"),
        (r"\boxed{Accounts Receivable}", "Accounts Receivable"),
    ],
)
def test_a_boxed_answer_is_read(response, expected):
    from foresightrlcore.evaluate.metrics.choice_match import extract_choice

    assert extract_choice(response, CHOICES) == expected


def test_the_last_box_wins_because_working_is_boxed_too():
    """A model that shows its working boxes intermediate results as well."""
    from foresightrlcore.evaluate.metrics.choice_match import extract_choice

    response = r"first \boxed{A}, on reflection \boxed{B}"
    assert extract_choice(response, CHOICES) == "B"


def test_a_boxed_answer_beats_a_letter_mentioned_in_prose():
    """The box is explicit; the prose patterns only infer."""
    from foresightrlcore.evaluate.metrics.choice_match import extract_choice

    response = r"The answer is not A. Final: \boxed{B}"
    assert extract_choice(response, CHOICES) == "B"


def test_a_reasoning_model_boxing_its_answer_scores():
    """End to end, in the shape R1-Distill actually emits."""
    response = "<think>Payables? No.</think>\n\n" + r"\boxed{B}"
    verdict = ChoiceMatchMetric().score(_record(gold=1), Prediction(text=response))
    assert verdict.correct and not verdict.parse_failed


def test_a_config_that_opts_out_keeps_its_reasoning_tokens():
    """`enabled: false` must mean it, all the way down to the metric.

    The metric strips the conventional tags on its own for configs that never
    mention reasoning. That safety net must not countermand a config that
    deliberately turned stripping off.
    """
    response = "<think>Answer: A</think>"
    opted_out = Prediction(text=response, thinking_handled=True)

    assert ChoiceMatchMetric().score(_record(gold=0), opted_out).correct


def test_the_safety_net_still_applies_when_a_config_says_nothing():
    """An unconfigured reasoning model is the case the net exists for."""
    response = "<think>Answer: A</think>"
    unconfigured = Prediction(text=response, thinking_handled=False)

    verdict = ChoiceMatchMetric().score(_record(gold=0), unconfigured)
    assert not verdict.correct and verdict.parse_failed


def test_the_runner_marks_a_declared_opt_out_as_handled():
    """The flag has to originate from the config, or none of this connects."""
    from foresightrlcore.evaluate.runner import _as_prediction

    class _Chat:
        text = "<think>Answer: A</think>"
        prompt_tokens = 1
        completion_tokens = 1
        latency_ms = 1.0
        finish_reason = "stop"

    # thinking: {enabled: false} -> no tags to strip, but still *declared*.
    prediction = _as_prediction(_Chat(), None, declared=True)

    assert prediction.thinking_handled
    assert prediction.text == "<think>Answer: A</think>"


# ---------------------------------------------------------------------------
# store_responses
# ---------------------------------------------------------------------------


def _runner_with(tmp_path: Any, **overrides: Any) -> EvaluateRunner:
    """A runner whose run-config carries ``overrides``."""
    config = _config()
    for key, value in overrides.items():
        setattr(config.evaluate, key, value)
    return EvaluateRunner(config, output_dir=tmp_path)


def test_responses_are_not_kept_unless_asked_for(tmp_path):
    """Storing every generation is expensive; default off."""
    runner = _runner_with(tmp_path)
    kept, raw = runner._responses_to_store(
        Prediction(text="A", raw_text="<think>x</think>A")
    )
    assert (kept, raw) == ("", "")


def test_a_stored_raw_response_is_the_evidence_stripping_ran(tmp_path):
    """`raw_text` is set only when stripping changed the text."""
    runner = _runner_with(tmp_path, store_responses=True)
    kept, raw = runner._responses_to_store(
        Prediction(text="A", raw_text="<think>maybe B</think>A")
    )
    assert kept == "A"
    assert raw == "<think>maybe B</think>A"


def test_an_unstripped_response_stores_no_raw_copy(tmp_path):
    """Nothing was removed, so there is no second version to keep."""
    runner = _runner_with(tmp_path, store_responses=True)
    kept, raw = runner._responses_to_store(Prediction(text="A"))
    assert kept == "A" and raw == ""


def test_a_long_response_is_truncated_and_says_so(tmp_path):
    runner = _runner_with(tmp_path, store_responses=True, store_responses_max_chars=10)
    kept, _ = runner._responses_to_store(Prediction(text="x" * 25))
    assert kept.startswith("x" * 10) and "15 more chars" in kept


def test_a_zero_limit_keeps_the_response_whole(tmp_path):
    runner = _runner_with(tmp_path, store_responses=True, store_responses_max_chars=0)
    kept, _ = runner._responses_to_store(Prediction(text="x" * 25))
    assert kept == "x" * 25


def test_an_implicit_temperature_yields_to_an_explicit_top_p():
    """Anthropic rejects a request carrying both, and 0.0 here is a default.

    Found by the SageMaker smoke: a candidate that configured only `top_p`
    failed every call with `temperature and top_p cannot both be specified`,
    for a parameter the config never mentioned.
    """
    assert _model(top_p=0.95).effective_temperature is None


def test_an_explicit_temperature_is_sent_even_alongside_top_p():
    """Asking for both is a choice; the provider gets to answer for it."""
    model = _model(top_p=0.95, temperature=1.0)
    assert model.effective_temperature == 1.0


def test_the_default_temperature_still_applies_when_nothing_else_is_set():
    """Benchmarks want greedy decoding by default; that must not regress."""
    assert _model().effective_temperature == 0.0
    assert _model(top_k=64).effective_temperature == 0.0


def test_thinking_is_off_until_the_block_is_written():
    assert _model().thinking_tags is None
    assert _model(thinking={}).thinking_tags == list(DEFAULT_THINKING_TAGS)
    assert _model(thinking={"tags": ["muse"]}).thinking_tags == ["muse"]


def test_thinking_can_be_switched_off_without_losing_its_tag_list():
    model = _model(thinking={"enabled": False, "tags": ["muse"]})
    assert model.thinking_tags is None
    assert model.thinking is not None and model.thinking.tags == ["muse"]


def test_an_empty_tag_list_is_rejected_rather_than_silently_doing_nothing():
    with pytest.raises(ValueError):
        _model(thinking={"tags": []})
    with pytest.raises(ValueError):
        _model(thinking={"tags": ["<>"]})


# ---------------------------------------------------------------------------
# Runner wiring
# ---------------------------------------------------------------------------

REASONED = "<think>Let me see, the answer is Berlin.</think>Paris"

RECORDS = [
    {
        "id": "q1",
        "question": "Capital of France?",
        "choices": ["Berlin", "Paris"],
        "answer": "Paris",
    }
]


def _config(**model_overrides: Any) -> EvaluateJobConfig:
    model: Dict[str, Any] = {"name": "m1", "provider": "echo"}
    model.update(model_overrides)
    return EvaluateJobConfig.model_validate(
        {
            "job": {"name": "t"},
            "execution": {"type": "local"},
            "output": {},
            "evaluate": {
                "benchmarks": [
                    {
                        "name": "mcq",
                        "type": "mcq",
                        "metrics": ["choice_match"],
                        "dataset": {"loader": "inline", "records": RECORDS},
                    }
                ],
                "models": [model],
                "concurrency": 1,
                "reporters": [],
            },
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


def _run(tmp_path, config, client):
    runner = EvaluateRunner(
        config,
        output_dir=tmp_path / "out",
        checkpoint_dir=tmp_path / "ckpt",
        workspace_root=tmp_path / "ws",
        endpoint_factory=_fake_endpoint,
    )
    with mock.patch("foresightrlcore.evaluate.runner.ChatClient", return_value=client):
        return runner.run()


def test_runner_forwards_the_configured_sampling_parameters(tmp_path):
    client = FakeChatClient({"*": "Paris"})
    _run(tmp_path, _config(top_p=0.95, top_k=64, temperature=1.0), client)

    sent = client.call_kwargs[0]
    assert sent["top_p"] == 0.95
    assert sent["top_k"] == 64
    assert sent["temperature"] == 1.0


def test_runner_omits_sampling_parameters_that_were_not_configured(tmp_path):
    client = FakeChatClient({"*": "Paris"})
    _run(tmp_path, _config(), client)

    sent = client.call_kwargs[0]
    assert sent["top_p"] is None and sent["top_k"] is None


def test_runner_strips_the_scratchpad_with_the_model_s_own_tags(tmp_path):
    client = FakeChatClient({"*": "<muse>the answer is Berlin</muse>Paris"})
    result = _run(tmp_path, _config(thinking={"tags": ["muse"]}), client)

    assert result.results[0]["score"] == 1.0


def test_a_custom_tagged_scratchpad_is_graded_when_the_model_does_not_declare_it(
    tmp_path,
):
    """The counter-case: an unknown tag is invisible, and the rejected answer wins.

    This is what the `thinking:` block buys. It is asserted so that a later
    change making tag handling unconditional does not pass unnoticed.
    """
    client = FakeChatClient({"*": "<muse>the answer is Berlin</muse>Paris"})
    result = _run(tmp_path, _config(), client)

    assert result.results[0]["score"] == 0.0


def test_an_undeclared_reasoning_model_is_not_graded_on_its_scratchpad(tmp_path):
    """Default tags are handled by the metric even with no `thinking:` block."""
    client = FakeChatClient({"*": REASONED})
    result = _run(tmp_path, _config(), client)

    assert result.results[0]["score"] == 1.0
