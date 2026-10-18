"""The QBAA matching benchmark: prompting, id extraction, and the two slices.

Skipped in the ``oss`` tox env — the benchmark is an extension type, and
nothing here may be reachable from the core.

The parsing cases are not hypothetical. Each was taken from a real 1000-row run
of Qwen3.8, K2-Horizon, Ornith or Inkling; between them the reasoning-draft
cases alone accounted for 560–678 discarded answers per model.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.requires_intuit


@pytest.fixture(autouse=True)
def _extension_loaded():
    pytest.importorskip("foresightrl_intuit.evaluate.benchmarks.qbaa_matching")


def _fenced(*txn_ids: str) -> str:
    dataset = [{"txn_id": txn_id} for txn_id in txn_ids]
    return "```json\n" + json.dumps({"data": {"dataset": dataset}}) + "\n```"


def _bare(txn_id: str) -> str:
    return json.dumps({"data": {"dataset": [{"txn_id": txn_id}]}})


def _record(gold=None, prompt="q", **meta):
    from foresightrlcore.evaluate.interfaces import EvalRecord

    return EvalRecord(id="r", inputs={"question": prompt}, gold=gold, meta=meta)


def _benchmark(**kwargs):
    from foresightrl_intuit.evaluate.benchmarks.qbaa_matching import (
        QbaaMatchingBenchmark,
    )

    return QbaaMatchingBenchmark(**kwargs)


class TestExtractMatchedIds:
    def _extract(self, *args, **kwargs):
        from foresightrl_intuit.evaluate.benchmarks.qbaa_matching import (
            extract_matched_ids,
        )

        return extract_matched_ids(*args, **kwargs)

    def test_single_id(self):
        assert self._extract(_fenced("263726")) == "263726"

    def test_ids_are_sorted_numerically_and_joined(self):
        assert self._extract(_fenced("112125", "111427")) == "111427:112125"

    def test_numeric_sort_beats_lexicographic(self):
        assert self._extract(_fenced("1000", "999")) == "999:1000"
        assert self._extract(_fenced("1000", "999"), sort_numeric=False) == "1000:999"

    def test_duplicate_ids_collapse(self):
        assert self._extract(_fenced("42", "42")) == "42"

    def test_integer_ids_are_stringified(self):
        response = '```json\n{"data": {"dataset": [{"txn_id": 37606}]}}\n```'
        assert self._extract(response) == "37606"

    def test_invoice_id_is_collected_alongside_txn_id(self):
        response = (
            '```json\n{"data": {"dataset": ['
            '{"txn_id": "None", "invoice_id": "500"},'
            '{"txn_id": "400", "invoice_id": "None"}]}}\n```'
        )
        assert self._extract(response) == "400:500"

    @pytest.mark.parametrize("sentinel", ["None", "", "null"])
    def test_sentinel_ids_are_not_suggestions(self, sentinel):
        response = (
            f'```json\n{{"data": {{"dataset": [{{"txn_id": "{sentinel}"}}]}}}}\n```'
        )
        assert self._extract(response) == ""

    def test_non_numeric_id_means_no_suggestion(self):
        # Gold ids are ordered by int, so sorted(key=int) raises and the
        # reference reader swallows that into "no suggestion".
        response = '```json\n{"data": {"dataset": [{"txn_id": "abc"}]}}\n```'
        assert self._extract(response) == ""

    def test_multiple_fenced_blocks_score_as_no_answer(self):
        # Spanning the first ```json to the last ``` yields invalid JSON.
        assert self._extract(_fenced("1") + "\nand also\n" + _fenced("2")) == ""

    @pytest.mark.parametrize(
        "response",
        [
            "",
            "no json here",
            "```json\n{not valid}\n```",
            '```json\n{"data": {"dataset": []}}\n```',
            '```json\n{"data": {}}\n```',
            '```json\n{"data": {"txn_id": "37606"}}\n```',
            '```json\n{"data": {"dataset": [{"txn_',
        ],
    )
    def test_unparseable_responses_mean_no_suggestion(self, response):
        assert self._extract(response) == ""


class TestUnfencedJson:
    """Ornith answers in bare JSON, which a fenced-only reader discards."""

    def _extract(self, response, require_fenced=True):
        from foresightrl_intuit.evaluate.benchmarks.qbaa_matching import (
            extract_matched_ids,
        )

        return extract_matched_ids(response, require_fenced=require_fenced)

    def test_bare_json_is_ignored_by_default(self):
        assert self._extract(_bare("111")) == ""

    def test_bare_json_is_read_when_the_fence_is_not_required(self):
        assert self._extract(_bare("111"), require_fenced=False) == "111"

    def test_bare_json_surrounded_by_prose_is_read(self):
        response = f"Here is my answer:\n{_bare('111')}\nHope that helps."
        assert self._extract(response, require_fenced=False) == "111"

    def test_a_fence_still_wins_over_a_bare_object(self):
        response = f"considered {_bare('999')}\n\n" + _fenced("111")
        assert self._extract(response, require_fenced=False) == "111"

    def test_braces_inside_explanations_do_not_end_the_object(self):
        payload = {
            "data": {
                "dataset": [
                    {"txn_id": "111", "explanation": "matched on {ref} } literally"}
                ]
            }
        }
        assert self._extract(json.dumps(payload), require_fenced=False) == "111"


class TestDraftedJsonDuringReasoning:
    """Reasoning models sketch the schema before answering it."""

    def _extract(self, response):
        from foresightrl_intuit.evaluate.benchmarks.qbaa_matching import (
            extract_matched_ids,
        )

        return extract_matched_ids(response, require_fenced=False)

    def test_a_placeholder_sketch_does_not_beat_the_real_answer(self):
        # Qwen3.8's habit: restate the schema with `...` holes, then answer.
        response = (
            'Need output schema exactly?\n{\n "data": {\n   "dataset": [ {...} ],\n'
            '   "confidence_score": ...\n }\n}\n\nFinal:\n' + _bare("111")
        )
        assert self._extract(response) == "111"

    def test_a_complete_earlier_draft_loses_to_the_final_answer(self):
        # Both parse, so only "last wins" separates them.
        response = f"First attempt:\n{_bare('999')}\nOn reflection:\n" + _bare("111")
        assert self._extract(response) == "111"

    def test_an_unclosed_brace_in_reasoning_does_not_hide_the_answer(self):
        # A brace-counting reader never returns to depth zero here, so it
        # treats the real answer as nested and finds nothing.
        response = "I will emit {\n\nActually, the answer is:\n" + _bare("111")
        assert self._extract(response) == "111"

    def test_the_answer_wins_over_its_own_nested_records(self):
        # Scanning backwards would return the last inner dataset record.
        payload = {"data": {"dataset": [{"txn_id": "111"}, {"txn_id": "222"}]}}
        assert self._extract(json.dumps(payload)) == "111:222"


class TestThinking:
    def test_the_answer_is_taken_from_after_the_thinking_block(self):
        completion = f"<think>maybe {_fenced('999')}</think>\n{_fenced('111')}"
        assert _benchmark().parse(completion) == "111"

    def test_a_custom_close_tag_is_honoured(self):
        # K2-Horizon closes with </ifm|think>, not </think>.
        completion = f"reasoning {_fenced('999')}</ifm|think>\n{_fenced('111')}"
        assert _benchmark(think_close_tag="</ifm|think>").parse(completion) == "111"

    def test_the_wrong_close_tag_lets_a_draft_win(self):
        # Quiet rather than loud: the answer still parses, but out of the whole
        # response, so a block drafted mid-reasoning can beat the final one.
        completion = f"<think>maybe {_fenced('999')}</think>\n{_fenced('111')}"
        assert _benchmark(think_close_tag="</nope>").parse(completion) == ""

    def test_thinking_can_be_left_in_place(self):
        completion = f"<think>{_fenced('999')}</think>\n{_fenced('111')}"
        assert _benchmark(strip_thinking=False).parse(completion) == ""


class TestPerModelThinking:
    """A candidate's own ``thinking:`` block wins over the benchmark params."""

    @staticmethod
    def _served(text, tags, *, declared=True):
        from foresightrlcore.evaluate.runner import _as_prediction

        class _Chat:
            prompt_tokens = 1
            completion_tokens = 1
            latency_ms = 1.0
            finish_reason = "stop"

        chat = _Chat()
        chat.text = text  # type: ignore[attr-defined]
        return _as_prediction(chat, tags, declared=declared)

    def test_a_models_own_tag_is_used_whatever_the_benchmark_says(self):
        # One run, two conventions: the benchmark's tag fits Qwen, and the
        # K2-Horizon candidate brings its own through thinking.tags.
        completion = f"reasoning {_fenced('999')}</ifm|think>\n{_fenced('111')}"
        prediction = self._served(completion, ["ifm|think"])

        verdict = _benchmark().score(_record(gold="111"), prediction)

        assert verdict.correct

    def test_a_declared_opt_out_is_not_overridden(self):
        # thinking: {enabled: false} -> nothing stripped upstream, and the
        # benchmark must not strip on the model's behalf either.
        completion = f"<think>{_fenced('999')}</think>\n{_fenced('111')}"
        prediction = self._served(completion, None)

        assert _benchmark().parse(completion) == "111"
        assert _benchmark().score(_record(gold="111"), prediction).correct is False

    def test_an_undeclared_model_falls_back_to_the_benchmark_params(self):
        completion = f"<think>maybe {_fenced('999')}</think>\n{_fenced('111')}"
        prediction = self._served(completion, None, declared=False)

        assert _benchmark().score(_record(gold="111"), prediction).correct

    def test_the_raw_completion_keeps_the_reasoning_the_runner_removed(self):
        completion = f"<think>maybe {_fenced('999')}</think>\n{_fenced('111')}"
        prediction = self._served(completion, ["think"])

        verdict = _benchmark(keep_raw_completion=True).score(
            _record(gold="111"), prediction
        )

        assert verdict.detail["raw_completion"] == completion


class TestIsIdInPrompt:
    def _check(self, gold_id, prompt):
        from foresightrl_intuit.evaluate.benchmarks.qbaa_matching import is_id_in_prompt

        return is_id_in_prompt(gold_id, prompt)

    def test_an_offered_candidate_is_found(self):
        assert self._check("123", "candidates: 456, 123, 789")

    def test_a_short_id_does_not_match_inside_a_longer_one(self):
        # Without digit bounds the row looks answerable when it is not.
        assert not self._check("123", "candidates: 1234, 51239")

    def test_an_empty_id_is_never_present(self):
        assert not self._check("", "anything")


class TestClassifyRetrievability:
    def _classify(self, gold, prompt):
        from foresightrl_intuit.evaluate.benchmarks.qbaa_matching import (
            classify_retrievability,
        )

        return classify_retrievability(gold, prompt)

    def test_no_gold_is_no_match(self):
        assert self._classify("", "candidates: 111") == "no_match"

    def test_gold_shown_in_the_prompt_is_matchable(self):
        assert self._classify("111", "candidates: 111, 222") == "matchable"

    def test_gold_absent_from_the_prompt_is_unretrievable(self):
        assert self._classify("999", "candidates: 111, 222") == "unretrievable"

    def test_a_partially_retrieved_label_is_unretrievable(self):
        # Exact-set scoring is all-or-nothing, so a half-retrieved label is as
        # unproducible as a wholly missing one.
        assert self._classify("111:999", "candidates: 111, 222") == "unretrievable"


class TestBuildPrompt:
    def test_the_rows_own_system_prompt_is_used(self):
        record = _record(prompt="which txn?", system_prompt="You are a bookkeeper.")
        messages = _benchmark().build_prompt(record).messages
        assert messages[0] == {"role": "system", "content": "You are a bookkeeper."}
        assert messages[-1]["content"] == "which txn?"

    def test_a_blank_system_prompt_is_omitted_rather_than_sent_empty(self):
        messages = _benchmark().build_prompt(_record(system_prompt="   ")).messages
        assert [m["role"] for m in messages] == ["user"]

    def test_max_tokens_overrides_the_model_default(self):
        assert _benchmark(max_tokens=4096).build_prompt(_record()).max_tokens == 4096


class TestScore:
    def _score(self, gold, completion, *, prompt="candidates: 111, 222", **kwargs):
        from foresightrlcore.evaluate.interfaces import Prediction

        record = _record(gold=gold, prompt=prompt, **kwargs.pop("meta", {}))
        return _benchmark(require_fenced_json=False, **kwargs).score(
            record, Prediction(text=completion)
        )

    def test_an_exact_set_scores_one(self):
        verdict = self._score("111:222", _fenced("111", "222"))
        assert (verdict.value, verdict.correct) == (1.0, True)
        assert verdict.detail["outcome"] == "TP"

    def test_a_partial_set_is_a_false_positive(self):
        verdict = self._score("111:222", _fenced("111"))
        assert verdict.detail["outcome"] == "FP"
        assert verdict.value == 0.0

    def test_the_segment_and_group_ride_along_for_the_reporter(self):
        verdict = self._score("111", _fenced("111"), meta={"ofx_review_type": "1:1"})
        assert verdict.detail["segment"] == "matchable"
        assert verdict.detail["group"] == "1:1"

    def test_an_unretrieved_gold_row_is_segmented_as_such(self):
        verdict = self._score("999", "no match found")
        assert verdict.detail["segment"] == "unretrievable"
        # FN, not FP: the model stayed silent, which is the right answer here
        # even though a gold label exists. The reporter scores it by abstention.
        assert verdict.detail["outcome"] == "FN"

    def test_a_missing_review_type_is_reported_as_unknown(self):
        assert self._score("111", _fenced("111")).detail["group"] == "unknown"

    def test_the_raw_completion_is_not_kept_by_default(self):
        # The completion repeats sensitive transaction data from the prompt.
        assert "raw_completion" not in self._score("111", _fenced("111")).detail

    def test_a_run_meant_for_rescoring_can_keep_the_completion(self):
        completion = _fenced("111")
        verdict = self._score("111", completion, keep_raw_completion=True)
        assert verdict.detail["raw_completion"] == completion

    def test_a_failed_request_records_the_error_and_nothing_else(self):
        from foresightrlcore.evaluate.interfaces import Prediction

        verdict = _benchmark().score(_record(gold="111"), Prediction(error="boom"))
        assert verdict.detail == {"error": "boom"}

    def test_truncation_survives_into_the_metric(self):
        from foresightrlcore.evaluate.interfaces import Prediction

        # The benchmark re-wraps the prediction around the parsed answer; if it
        # dropped finish_reason, a truncated run would look merely wrong.
        record = _record(gold="111", prompt="candidates: 111")
        verdict = _benchmark().score(
            record, Prediction(text="<think>...", finish_reason="length")
        )
        assert verdict.detail["outcome"] == "FN"


class TestRegistration:
    def test_the_type_is_registered_and_builds_from_spec_params(self):
        from typing import cast

        from foresightrl_intuit.evaluate.benchmarks.qbaa_matching import (
            QbaaMatchingBenchmark,
        )

        from foresightrlcore.config.evaluate_schema import BenchmarkSpec
        from foresightrlcore.evaluate.registry import BENCHMARK_TYPES

        spec = BenchmarkSpec(
            name="qbaa_matching",
            type="qbaa_matching",
            params={"require_fenced_json": False, "think_close_tag": "</ifm|think>"},
        )
        benchmark = cast(
            QbaaMatchingBenchmark,
            BENCHMARK_TYPES.get("qbaa_matching")(spec=spec, metrics=None),
        )
        assert benchmark.require_fenced_json is False
        assert benchmark.think_close_tag == "</ifm|think>"

    def test_the_catalog_entry_resolves(self):
        from foresightrl_intuit.evaluate import CATALOG_PATH

        from foresightrlcore.evaluate.catalog import load_catalog

        entry = load_catalog([str(CATALOG_PATH)])["qbaa_matching"]
        assert entry.type == "qbaa_matching"
        assert entry.dataset is not None
        assert entry.dataset.loader == "parquet"
