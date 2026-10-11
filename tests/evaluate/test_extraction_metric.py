"""Tests for the extraction metric: answer parsing and outcome classification."""

from __future__ import annotations

from foresightrlcore.evaluate.interfaces import EvalRecord, Prediction
from foresightrlcore.evaluate.metrics.extraction import (
    ExtractionMatchMetric,
    as_answer,
    classify,
)


def record(gold=None, **meta):
    return EvalRecord(id="r", inputs={"question": "q"}, gold=gold, meta=meta)


class TestAsAnswer:
    """Missing values arrive spelled several ways and all mean the same thing."""

    def test_none_is_empty(self):
        assert as_answer(None) == ""

    def test_pandas_nan_string_is_empty(self):
        # A blank in a float column reads back as the string "nan".
        assert as_answer("nan") == ""

    def test_json_null_words_are_empty(self):
        assert as_answer("None") == ""
        assert as_answer("null") == ""

    def test_real_value_is_stripped(self):
        assert as_answer("  111427:112125 ") == "111427:112125"


class TestClassify:
    def test_exact_match_is_tp(self):
        assert classify("111:222", "111:222") == "TP"

    def test_wrong_answer_on_a_gold_row_is_fp_not_fn(self):
        # The distinction the whole report rests on: answering wrongly is a
        # different failure from staying silent, and only silence is FN.
        assert classify("111:222", "111") == "FP"

    def test_silence_on_a_gold_row_is_fn(self):
        assert classify("111:222", "") == "FN"

    def test_silence_with_no_gold_is_tn(self):
        assert classify("", "") == "TN"

    def test_answering_when_there_was_nothing_is_fp(self):
        assert classify("", "999") == "FP"


class TestExtractionMatchMetric:
    def test_partial_answer_scores_zero_not_half(self):
        verdict = ExtractionMatchMetric().score(
            record(gold="111:222"), Prediction(text="111")
        )
        assert verdict.value == 0.0
        assert verdict.correct is False
        assert verdict.detail["outcome"] == "FP"

    def test_correct_abstention_is_correct_but_scores_zero(self):
        # value feeds the runner's mean, which is a TP rate; correct feeds
        # accuracy. A right "no match" belongs in the second, not the first.
        verdict = ExtractionMatchMetric().score(record(gold=None), Prediction(text=""))
        assert verdict.correct is True
        assert verdict.value == 0.0
        assert verdict.detail["outcome"] == "TN"

    def test_exact_match_is_correct_and_scores_one(self):
        verdict = ExtractionMatchMetric().score(
            record(gold="111:222"), Prediction(text="111:222")
        )
        assert (verdict.value, verdict.correct) == (1.0, True)

    def test_a_failed_request_scores_nothing_and_records_why(self):
        verdict = ExtractionMatchMetric().score(
            record(gold="111"), Prediction(error="boom")
        )
        assert verdict.detail == {"error": "boom"}
        assert verdict.correct is False

    def test_unparseable_output_is_a_parse_failure_not_an_abstention(self):
        # The model said something; nothing survived parsing. That is the
        # harness's problem to see, distinct from a deliberate "no match".
        verdict = ExtractionMatchMetric().score(
            record(gold="111"), Prediction(text="nan")
        )
        assert verdict.parse_failed is True

    def test_a_genuine_empty_reply_is_not_a_parse_failure(self):
        verdict = ExtractionMatchMetric().score(record(gold="111"), Prediction(text=""))
        assert verdict.parse_failed is False
