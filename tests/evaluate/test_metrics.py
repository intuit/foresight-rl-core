"""Metrics and trial statistics."""

from __future__ import annotations

import math

import pytest

from foresightrlcore.evaluate.interfaces import EvalRecord, Prediction
from foresightrlcore.evaluate.metrics.aggregate import (
    mean_sub_scores,
    mean_with_ci,
    score_population,
    t_critical,
    unstable_sub_scores,
)
from foresightrlcore.evaluate.metrics.choice_match import (
    ChoiceMatchMetric,
    extract_choice,
    resolve_gold_index,
)
from foresightrlcore.evaluate.metrics.judge import JudgeMetric
from foresightrlcore.evaluate.metrics.text_match import (
    ExactMatchMetric,
    SubstringMatchMetric,
    normalize,
)
from foresightrlcore.evaluate.spec import INVALID_SCORE

CHOICES = ["Berlin", "Madrid", "Paris", "Rome"]


def _record(gold="Paris", choices=None):
    return EvalRecord(
        id="r1",
        inputs={"question": "Capital of France?"},
        gold=gold,
        choices=CHOICES if choices is None else choices,
    )


# ---------------------------------------------------------------------------
# Text matching
# ---------------------------------------------------------------------------


def test_normalize_strips_case_edge_punctuation_and_a_leading_article():
    assert normalize("  The  Paris. ") == "paris"
    assert normalize("PARIS!") == normalize("paris")
    # Interior punctuation is left alone on purpose: normalising it away starts
    # marking wrong answers correct.
    assert normalize("Paris, France") == "paris, france"


def test_exact_match_is_normalized_not_literal():
    metric = ExactMatchMetric()
    verdict = metric.score(_record(), Prediction(text="  paris. "))
    assert verdict.correct and verdict.value == 1.0

    assert not metric.score(_record(), Prediction(text="Berlin")).correct


def test_substring_match_accepts_an_embedded_answer():
    verdict = SubstringMatchMetric().score(
        _record(), Prediction(text="I believe the answer is Paris.")
    )
    assert verdict.correct


def test_a_generation_error_is_not_scored_as_wrong():
    verdict = ExactMatchMetric().score(_record(), Prediction(error="boom"))
    assert verdict.value == 0.0
    assert not verdict.parse_failed
    assert verdict.detail["error"] == "boom"


# ---------------------------------------------------------------------------
# Choice matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "C",
        "Answer: C",
        "**C**",
        "(c)",
        '{"correct_answer": "C"}',
        '```json\n{"answer": "Paris"}\n```',
        "Paris",
    ],
)
def test_choice_match_accepts_the_common_answer_formats(text):
    assert ChoiceMatchMetric().score(_record(), Prediction(text=text)).correct


@pytest.mark.parametrize("gold", ["Paris", "C", "c", 2, "2"])
def test_gold_answers_may_be_text_letter_or_index(gold):
    assert resolve_gold_index(gold, CHOICES) == 2


# A prompt may ask for "the number or letter of the correct answer" while
# rendering its options unlabelled, which several catalog benchmarks do. The
# model then invents a label, and the answer text underneath it must still
# count -- otherwise a compliant, correct answer scores zero.
@pytest.mark.parametrize("answer", ["C) Paris", "(c) Paris", "3. Paris", "c - Paris"])
def test_a_label_the_model_invented_does_not_hide_a_correct_answer(answer):
    verdict = ChoiceMatchMetric().score(
        _record(), Prediction(text='{"correct_answer": "%s"}' % answer)
    )
    assert verdict.correct
    assert not verdict.parse_failed


def test_the_answer_text_wins_over_a_label_that_contradicts_it():
    """The text is the answer; the label is decoration the prompt asked for."""
    verdict = ChoiceMatchMetric().score(
        _record(), Prediction(text='{"correct_answer": "A) Paris"}')
    )
    assert verdict.detail["predicted_index"] == 2


def test_options_carrying_their_own_markers_still_match_whole():
    labelled = ["A) Berlin", "B) Madrid", "C) Paris", "D) Rome"]
    verdict = ChoiceMatchMetric().score(
        _record(gold="C) Paris", choices=labelled),
        Prediction(text='{"correct_answer": "C) Paris"}'),
    )
    assert verdict.correct
    assert verdict.detail["predicted_index"] == 2


def test_a_label_needs_a_separator_to_count_as_one():
    """`10 years` is an answer, not option 10 followed by `years`.

    Without the separator requirement this would resolve, and any answer
    beginning with a number would be silently truncated.
    """
    assert resolve_gold_index("10 years", ["years", "Rome"]) is None


def test_a_bare_integer_falls_back_to_one_based_when_zero_based_overflows():
    assert resolve_gold_index(4, CHOICES) == 3


@pytest.mark.parametrize("answer", ["②①⑤④③", "²"])
def test_digits_int_cannot_read_are_a_parse_failure_not_a_crash(answer):
    # Taken from a GPT-5.4 leaderboard run: one such answer raised out of the
    # metric and ended the whole job.
    verdict = ChoiceMatchMetric().score(_record(), Prediction(text=answer))
    assert verdict.parse_failed
    assert resolve_gold_index(answer, CHOICES) is None


def test_unparseable_output_is_distinguished_from_a_wrong_answer():
    verdict = ChoiceMatchMetric().score(
        _record(), Prediction(text="I would rather not say.")
    )
    assert verdict.parse_failed
    assert not verdict.correct


def test_an_unresolvable_gold_is_a_harness_failure_not_a_model_failure():
    verdict = ChoiceMatchMetric().score(_record(gold="Atlantis"), Prediction(text="C"))
    assert verdict.parse_failed
    assert "gold" in verdict.detail


def test_extract_choice_returns_none_for_empty_output():
    assert extract_choice("", CHOICES) is None


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------


class _FakeManager:
    def __init__(self, reply):
        self.reply = reply
        self.refreshed = 0

    def get(self):
        manager = self

        class _Client:
            def bind(self, **_):
                return self

            def invoke(self, _messages):
                if isinstance(manager.reply, Exception):
                    raise manager.reply
                return type("Msg", (), {"content": manager.reply})()

        return _Client()

    def force_refresh(self):
        self.refreshed += 1


def test_judge_rescales_a_five_point_rating_to_unit_interval():
    verdict = JudgeMetric(_FakeManager("4")).score(_record(), Prediction(text="Paris"))
    assert verdict.value == pytest.approx(0.75)
    assert verdict.correct


def test_judge_output_without_a_rating_is_a_parse_failure():
    verdict = JudgeMetric(_FakeManager("no idea")).score(
        _record(), Prediction(text="Paris")
    )
    assert verdict.parse_failed


def test_an_unavailable_judge_is_a_harness_failure():
    metric = JudgeMetric(
        _FakeManager(RuntimeError("judge down")), max_retries=1, base_delay=0
    )
    verdict = metric.score(_record(), Prediction(text="Paris"))
    assert verdict.parse_failed
    assert "judge call failed" in verdict.detail["reason"]


def test_judge_without_a_client_says_so():
    verdict = JudgeMetric(None).score(_record(), Prediction(text="Paris"))
    assert verdict.parse_failed
    assert "no llm client" in verdict.detail["reason"]


# ---------------------------------------------------------------------------
# Trial statistics
# ---------------------------------------------------------------------------


def test_t_critical_matches_the_published_table():
    assert t_critical(1, 0.95) == pytest.approx(12.706)
    assert t_critical(4, 0.95) == pytest.approx(2.776)
    # Past the tabulated range it converges on the normal quantile.
    assert t_critical(500, 0.95) == pytest.approx(1.96, abs=0.01)
    assert t_critical(0) == 0.0


def test_an_interval_from_three_trials_is_wider_than_the_normal_one():
    values = [0.70, 0.75, 0.80]
    mean, stdev, low, high = mean_with_ci(values, 0.95)

    assert mean == pytest.approx(0.75)
    normal_halfwidth = 1.96 * stdev / math.sqrt(len(values))
    assert (high - low) / 2 > normal_halfwidth


def test_a_single_trial_reports_no_fabricated_spread():
    mean, stdev, low, high = mean_with_ci([0.42])
    assert (mean, stdev, low, high) == (0.42, 0.0, 0.42, 0.42)


def test_mean_sub_scores_averages_over_the_trials_that_reported_each_key():
    averaged = mean_sub_scores([{"a": 1.0, "b": 0.0}, {"a": 0.0}])
    assert averaged == {"a": 0.5, "b": 0.0}


def test_unstable_sub_scores_names_only_the_ones_that_moved():
    trials = [{"stable": 0.5, "flaky": 0.1}, {"stable": 0.5, "flaky": 0.9}]
    assert unstable_sub_scores(trials) == ["flaky"]
    assert unstable_sub_scores([{"a": 1.0}]) == []


def test_too_many_errors_reports_invalid_rather_than_an_optimistic_mean():
    # 2 of 4 errored against a 10% tolerance: the surviving records are a
    # different population from the one the benchmark claims to measure.
    assert score_population([1.0, 1.0], 2, 0.1, INVALID_SCORE) == INVALID_SCORE
    assert score_population([1.0, 0.0], 0, 0.1, INVALID_SCORE) == 0.5
    assert score_population([], 0, 0.1, INVALID_SCORE) == INVALID_SCORE
