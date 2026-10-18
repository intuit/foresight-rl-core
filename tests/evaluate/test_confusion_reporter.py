"""The confusion reporter: aggregate counts, slices, and trial handling."""

from __future__ import annotations

import json

from foresightrlcore.evaluate.reporters.confusion import (
    ConfusionReporter,
    abstention_accuracy,
    precision_recall,
)


def sample(outcome, expected, got, **extra):
    """A flat sample row of the shape the runner writes."""
    return {
        "benchmark": "b",
        "model": "m",
        "error": "",
        "detail_outcome": outcome,
        "detail_expected": expected,
        "detail_got": got,
        **extra,
    }


class TestPrecisionRecall:
    def test_recall_denominator_is_gold_rows_not_tp_plus_fn(self):
        # 1 right, 1 answered wrongly, 0 silent, over 2 gold rows. TP/(TP+FN)
        # would read 1.0 and excuse the wrong answer; this reads 0.5.
        counts = {"TP": 1, "FP": 1, "FN": 0, "TN": 0}
        assert precision_recall(counts, num_gold=2)["recall"] == 0.5

    def test_no_predictions_gives_zero_precision_rather_than_dividing_by_zero(self):
        assert precision_recall({"TP": 0, "FP": 0}, num_gold=5)["precision"] == 0.0

    def test_f1_is_zero_when_both_halves_are_zero(self):
        assert precision_recall({"TP": 0, "FP": 3}, num_gold=4)["f1"] == 0.0


class TestAbstentionAccuracy:
    def test_accuracy_is_the_share_that_held_back(self):
        rows = [sample("TN", "", ""), sample("FP", "", "999"), sample("TN", "", "  ")]
        result = abstention_accuracy(rows)
        assert (result["num_abstained"], result["num_over_matched"]) == (2, 1)
        assert result["accuracy"] == 2 / 3

    def test_empty_segment_does_not_divide_by_zero(self):
        assert abstention_accuracy([])["accuracy"] == 0.0


class TestConfusionReporter:
    def _summarise(self, rows, **kwargs):
        return ConfusionReporter(**kwargs).summarise(rows)

    def test_overall_counts_and_rates(self):
        rows = [
            sample("TP", "111", "111"),
            sample("FP", "222", "999"),
            sample("FN", "333", ""),
            sample("TN", "", ""),
        ]
        summary = self._summarise(rows)
        assert summary["counts"] == {"TP": 1, "FP": 1, "FN": 1, "TN": 1}
        assert summary["num_gold"] == 3
        assert summary["precision"] == 0.5
        assert summary["recall"] == 1 / 3

    def test_errored_rows_are_excluded_and_counted(self):
        rows = [sample("TP", "111", "111"), {**sample("", "", ""), "error": "boom"}]
        summary = self._summarise(rows)
        assert (summary["num_records"], summary["num_errors"]) == (1, 1)

    def test_truncated_rows_are_surfaced(self):
        rows = [
            sample("FN", "111", "", finish_reason="length"),
            sample("TP", "222", "222", finish_reason="stop"),
        ]
        assert self._summarise(rows)["num_truncated"] == 1

    def test_groups_are_scored_independently(self):
        rows = [
            sample("TP", "111", "111", detail_group="1:1"),
            sample("FP", "222", "999", detail_group="1:N"),
        ]
        by_group = self._summarise(rows)["by_group"]
        assert by_group["1:1"]["precision"] == 1.0
        assert by_group["1:N"]["precision"] == 0.0

    def test_abstention_segments_get_accuracy_and_no_precision(self):
        rows = [
            sample("TP", "111", "111", detail_segment="matchable"),
            sample("TN", "", "", detail_segment="no_match"),
            sample("FP", "333", "999", detail_segment="unretrievable"),
        ]
        by_segment = self._summarise(
            rows, abstention_segments=["no_match", "unretrievable"]
        )["by_segment"]

        assert by_segment["matchable"]["expected"] == "match"
        assert by_segment["matchable"]["precision"] == 1.0

        for name in ("no_match", "unretrievable"):
            assert by_segment[name]["expected"] == "no_match"
            assert "precision" not in by_segment[name]
        assert by_segment["no_match"]["accuracy"] == 1.0
        assert by_segment["unretrievable"]["accuracy"] == 0.0

    def test_breakdowns_are_omitted_when_the_rows_carry_no_slice(self):
        summary = self._summarise([sample("TP", "111", "111")])
        assert "by_group" not in summary
        assert "by_segment" not in summary

    def test_report_writes_one_document_per_benchmark_and_model(self, tmp_path):
        from foresightrlcore.evaluate.spec import EvaluateResult

        result = EvaluateResult(
            job_name="j",
            samples=[
                sample("TP", "111", "111"),
                {**sample("FP", "222", "999"), "model": "other"},
            ],
        )
        ConfusionReporter().report(result, tmp_path)

        document = json.loads((tmp_path / "confusion.json").read_text())
        assert sorted(document["b"]) == ["m", "other"]
        assert document["b"]["m"]["counts"]["TP"] == 1
        assert document["b"]["other"]["counts"]["FP"] == 1

    def test_report_on_a_harness_only_run_writes_nothing(self, tmp_path):
        from foresightrlcore.evaluate.spec import EvaluateResult

        ConfusionReporter().report(EvaluateResult(job_name="j", samples=[]), tmp_path)
        assert not (tmp_path / "confusion.json").exists()


class TestTrials:
    """F1 is a ratio of counts, so it is computed per trial and averaged after."""

    def _trials(self, rows, **kwargs):
        return ConfusionReporter(**kwargs).summarise_trials(rows)

    def test_a_single_trial_reads_exactly_like_an_unrepeated_run(self):
        rows = [
            sample("TP", "111", "111", trial=0),
            sample("FP", "222", "999", trial=0),
        ]

        assert self._trials(rows) == ConfusionReporter().summarise(rows)

    def test_each_trial_is_scored_on_its_own_rows(self):
        rows = [
            sample("TP", "111", "111", trial=0),
            sample("TP", "222", "222", trial=1),
            sample("FP", "333", "999", trial=1),
        ]

        summary = self._trials(rows)

        assert summary["n_trials"] == 2
        assert summary["trials"]["0"]["precision"] == 1.0
        assert summary["trials"]["1"]["precision"] == 0.5

    def test_the_mean_is_over_per_trial_metrics_not_pooled_rows(self):
        # Pooling would give 2 TP / 3 predicted = 0.667. Averaging the trials
        # gives (1.0 + 0.5) / 2 = 0.75. The second is the honest answer: the
        # measurement was taken twice and it moved.
        rows = [
            sample("TP", "111", "111", trial=0),
            sample("TP", "222", "222", trial=1),
            sample("FP", "333", "999", trial=1),
        ]

        assert self._trials(rows)["aggregate"]["precision"]["mean"] == 0.75

    def test_the_interval_widens_with_disagreement(self):
        agree = [
            sample("TP", "111", "111", trial=0),
            sample("TP", "111", "111", trial=1),
        ]
        disagree = [
            sample("TP", "111", "111", trial=0),
            sample("FP", "111", "999", trial=1),
        ]

        agreed = self._trials(agree)["aggregate"]["f1"]
        disagreed = self._trials(disagree)["aggregate"]["f1"]

        assert agreed["stdev"] == 0.0
        assert disagreed["stdev"] > 0.0
        assert disagreed["ci_high"] > disagreed["ci_low"]

    def test_a_multi_trial_summary_still_renders(self):
        from foresightrlcore.evaluate.reporters.confusion import format_summary

        rows = [
            sample("TP", "111", "111", trial=0),
            sample("FP", "222", "999", trial=1),
        ]

        text = format_summary(self._trials(rows), "b — m")

        assert "2 trials" in text
        assert "trial 0" in text and "trial 1" in text
