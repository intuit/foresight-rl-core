"""Tests for foresightrlcore.tasks.math — DeepMath GRPO accuracy reward.

Covers:
* :func:`reward_logic.accuracy_score` — math_verify equivalence, boxed
  answers, think-tag stripping, non-math string fallback, and error
  handling.
* Public interface (``get_reward_function``, ``get_callbacks``,
  ``get_dataset_loader``) — the TRL contract (``list[float]`` output,
  ``(completions, ground_truth)`` signature) and predictions-log writing.

``math_verify`` is a real dependency here (installed in the dev venv and
bundled in the qwen3 container recipe). ``transformers`` / ``datasets``
are stubbed by ``tests/conftest.py``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

for _mod in ("transformers", "datasets"):
    sys.modules.setdefault(_mod, MagicMock())

import pytest  # noqa: E402

from foresightrlcore.tasks.math import (  # noqa: E402
    get_callbacks,
    get_dataset_loader,
    get_reward_function,
)
from foresightrlcore.tasks.math.reward_logic import accuracy_score  # noqa: E402

math_verify = pytest.importorskip(
    "math_verify", reason="math_verify not installed in this env"
)


# ---------------------------------------------------------------------------
# accuracy_score
# ---------------------------------------------------------------------------


def test_accuracy_score_boxed_numeric_match():
    assert accuracy_score("The answer is \\boxed{1}.", "$1$") == 1.0


def test_accuracy_score_wrong_answer():
    assert accuracy_score("\\boxed{2}", "$1$") == 0.0


def test_accuracy_score_strips_thinking_tags():
    completion = "<think>lots of reasoning that mentions 5</think>\\boxed{42}"
    assert accuracy_score(completion, "$42$") == 1.0


def test_accuracy_score_latex_equivalence():
    # 1/2 and 0.5 are mathematically equivalent.
    assert accuracy_score("\\boxed{\\frac{1}{2}}", "0.5") == 1.0


def test_accuracy_score_text_fallback_match():
    # Non-mathematical answers fall back to normalized string compare.
    assert accuracy_score("Therefore the answer is Yes", "Yes") == 0.0
    assert accuracy_score("Yes", "yes") == 1.0


def test_accuracy_score_empty_inputs():
    assert accuracy_score("", "") == 0.0
    assert accuracy_score("anything", "") == 0.0


def test_accuracy_score_verify_error_falls_back_to_string_match():
    # math_verify.verify raising on a parsed-but-unverifiable pair must not
    # crash the reward batch — it falls back to normalized string compare.
    with patch("math_verify.verify", side_effect=ValueError("boom")):
        assert accuracy_score("\\boxed{1}", "$1$") == 0.0


# ---------------------------------------------------------------------------
# public interface
# ---------------------------------------------------------------------------


def test_get_callbacks_empty():
    assert get_callbacks({}) == []


def test_get_reward_function_contract():
    reward_fn = get_reward_function({})
    completions = ["\\boxed{1}", "\\boxed{9}"]
    ground_truth = ["$1$", "$1$"]
    rewards = reward_fn(completions, ground_truth)
    assert rewards == [1.0, 0.0]


def test_get_reward_function_conversational_completions():
    reward_fn = get_reward_function({})
    completions = [[{"role": "assistant", "content": "\\boxed{7}"}]]
    rewards = reward_fn(completions, ["$7$"])
    assert rewards == [1.0]


def test_reward_function_writes_predictions_log():
    with tempfile.TemporaryDirectory() as d:
        log_path = os.path.join(d, "sub", "preds.jsonl")
        reward_fn = get_reward_function({"predictions_log": log_path})
        reward_fn(["\\boxed{1}"], ["$1$"])
        assert os.path.isfile(log_path)
        with open(log_path, encoding="utf-8") as f:
            rec = json.loads(f.readline())
        assert rec["reward"] == 1.0
        assert rec["ground_truth"] == "$1$"


def test_get_dataset_loader_is_callable():
    # Loader construction must not touch the network (HF load is lazy, inside
    # the returned closure). transformers/datasets are stubbed by conftest.
    loader = get_dataset_loader({"max_samples": 5}, "Qwen/Qwen3-0.6B")
    assert callable(loader)


class _FakeDataset:
    """Minimal stand-in for a HF ``datasets.Dataset``."""

    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self):
        return len(self.rows)

    def shuffle(self, seed=None):  # noqa: ARG002
        return self

    def select(self, indices):
        return _FakeDataset([self.rows[i] for i in indices])

    def map(self, fn, remove_columns=None):  # noqa: ARG002
        return _FakeDataset([fn(row) for row in self.rows])


def test_dataset_loader_maps_conversational_prompt_and_solution():
    raw_rows = [
        {"prompt": [{"role": "user", "content": "2+2?"}], "solution": "4"},
    ]
    fake_tokenizer = MagicMock()
    fake_tokenizer.apply_chat_template.return_value = "<templated prompt>"

    with (
        patch("datasets.load_dataset", return_value=_FakeDataset(raw_rows)),
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer
        ),
    ):
        loader = get_dataset_loader(
            {"max_samples": 1, "shuffle_seed": 1}, "Qwen/Qwen3-0.6B"
        )
        ds = loader("/ignored/mounted/path")

    assert len(ds) == 1
    assert ds.rows[0]["prompt"] == "<templated prompt>"
    assert ds.rows[0]["ground_truth"] == "4"
    fake_tokenizer.apply_chat_template.assert_called_once()


def test_dataset_loader_falls_back_when_enable_thinking_unsupported():
    raw_rows = [{"prompt": [{"role": "user", "content": "hi"}], "solution": "ok"}]
    fake_tokenizer = MagicMock()
    fake_tokenizer.apply_chat_template.side_effect = [
        TypeError("enable_thinking not supported"),
        "<plain templated prompt>",
    ]

    with (
        patch("datasets.load_dataset", return_value=_FakeDataset(raw_rows)),
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer
        ),
    ):
        loader = get_dataset_loader({}, "Qwen/Qwen2.5-0.6B-Instruct")
        ds = loader("/ignored/mounted/path")

    assert ds.rows[0]["prompt"] == "<plain templated prompt>"
    assert fake_tokenizer.apply_chat_template.call_count == 2


def test_dataset_loader_passes_through_non_list_prompt():
    raw_rows = [{"prompt": "already a string prompt", "solution": "42"}]

    with (
        patch("datasets.load_dataset", return_value=_FakeDataset(raw_rows)),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()),
    ):
        loader = get_dataset_loader({}, "Qwen/Qwen3-0.6B")
        ds = loader("/ignored/mounted/path")

    assert ds.rows[0]["prompt"] == "already a string prompt"
    assert ds.rows[0]["ground_truth"] == "42"


def test_reward_function_predictions_log_write_failure_is_swallowed(caplog):
    # An OSError while writing the predictions log (e.g. disk full, bad
    # path) must be logged and swallowed, never raised out of the reward
    # function — a logging failure can't be allowed to kill a GRPO step.
    reward_fn = get_reward_function({"predictions_log": "/ignored/path.jsonl"})
    with patch("builtins.open", side_effect=OSError("disk full")):
        rewards = reward_fn(["\\boxed{1}"], ["$1$"])
    assert rewards == [1.0]
    assert "Failed to write prediction log" in caplog.text
