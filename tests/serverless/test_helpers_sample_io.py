"""Tests for :mod:`foresightrlcore.serverless.helpers.sample_io`."""

from __future__ import annotations

import json

from foresightrlcore.serverless.helpers import sample_io


class TestExtractResponse:
    def test_top_level_response_wins(self):
        sample = {
            "response": "TOP",
            "messages": [{"role": "assistant", "content": "FROM_MSG"}],
        }
        assert sample_io.extract_response(sample) == "TOP"

    def test_falls_back_to_messages(self):
        sample = {"messages": [{"role": "assistant", "content": "from-msg"}]}
        assert sample_io.extract_response(sample) == "from-msg"

    def test_falls_back_to_prompt_appended_assistant(self):
        sample = {
            "prompt": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "appended"},
            ]
        }
        assert sample_io.extract_response(sample) == "appended"

    def test_returns_empty_when_no_response_found(self):
        assert sample_io.extract_response({"foo": "bar"}) == ""


class TestExtractState:
    def test_uses_reference_answer_text(self):
        sample = {"reference_answer": {"text": "REFERENCE"}}
        assert sample_io.extract_state(sample) == "REFERENCE"

    def test_falls_back_to_reward_model_ground_truth(self):
        sample = {"reward_model": {"ground_truth": "GT", "style": "rule"}}
        assert sample_io.extract_state(sample) == "GT"

    def test_last_user_message_as_last_resort(self):
        sample = {
            "messages": [
                {"role": "system", "content": "ignore"},
                {"role": "user", "content": "the question"},
            ]
        }
        assert sample_io.extract_state(sample) == "the question"


class TestCoerceBatch:
    def test_list_event(self):
        assert sample_io.coerce_batch([{"a": 1}, {"b": 2}]) == [{"a": 1}, {"b": 2}]

    def test_batch_wrapper(self):
        assert sample_io.coerce_batch({"batch": [{"a": 1}]}) == [{"a": 1}]

    def test_body_json_string(self):
        body = json.dumps({"batch": [{"a": 1}, {"a": 2}]})
        assert sample_io.coerce_batch({"body": body}) == [{"a": 1}, {"a": 2}]

    def test_single_sample_dict(self):
        # Looks like a single sample because of fingerprint keys.
        sample = {"prompt": [{"role": "user", "content": "q"}]}
        assert sample_io.coerce_batch(sample) == [sample]

    def test_unknown_shape_returns_empty(self):
        assert sample_io.coerce_batch({"unknown": 1}) == []
