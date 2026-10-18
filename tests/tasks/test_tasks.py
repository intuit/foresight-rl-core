"""Tests for foresightrlcore.tasks — task loader and thinking utils."""

from __future__ import annotations

import pytest

from foresightrlcore.tasks.thinking_utils import (
    extract_thinking_length,
    parse_json_from_response,
    strip_thinking_tags,
)


class TestStripThinkingTags:
    """Test thinking tag removal."""

    def test_removes_think_tags(self):
        text = "<think>Some reasoning here</think>Final answer"
        assert strip_thinking_tags(text) == "Final answer"

    def test_removes_reasoning_tags(self):
        text = "<reasoning>Deep thought</reasoning>Result"
        assert strip_thinking_tags(text) == "Result"

    def test_handles_missing_open_tag(self):
        text = "Some text before</think>After thinking"
        result = strip_thinking_tags(text)
        assert result == "After thinking"

    def test_no_tags_unchanged(self):
        text = "Plain text without any tags"
        assert strip_thinking_tags(text) == text

    def test_empty_string(self):
        assert strip_thinking_tags("") == ""


class TestExtractThinkingLength:
    """Test thinking length measurement."""

    def test_normal_think_tags(self):
        text = "<think>12345</think>rest"
        assert extract_thinking_length(text) == 5

    def test_missing_open_tag(self):
        text = "some prefix text</think>rest"
        length = extract_thinking_length(text)
        assert length == len("some prefix text")

    def test_no_tags_returns_zero(self):
        assert extract_thinking_length("no tags here") == 0

    def test_empty_thinking(self):
        assert extract_thinking_length("<think></think>rest") == 0


class TestParseJsonFromResponse:
    """Test JSON extraction from LLM responses."""

    def test_json_in_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = parse_json_from_response(text)
        assert result == {"key": "value"}

    def test_raw_json(self):
        text = 'Here is the answer: {"recommendation": "do this"}'
        result = parse_json_from_response(text)
        assert result["recommendation"] == "do this"

    def test_json_after_thinking(self):
        text = '<think>reasoning</think>```json\n{"key": 1}\n```'
        result = parse_json_from_response(text)
        assert result == {"key": 1}

    def test_no_json_returns_empty(self):
        result = parse_json_from_response("No JSON here at all")
        assert result == {}


class TestTaskLoader:
    """Test task module loading."""

    def test_load_nonexistent_task_raises(self):
        from foresightrlcore.tasks import load_task_module

        with pytest.raises(ValueError, match="Unknown task type"):
            load_task_module("totally_fake_task_xyz")

    def test_load_from_nonexistent_path_raises(self):
        from foresightrlcore.tasks import load_task_module

        with pytest.raises(FileNotFoundError):
            load_task_module("/nonexistent/path/module.py")

    @pytest.mark.requires_intuit
    def test_load_advice_task_canonical(self):
        """advice should resolve via foresightrl_intuit.tasks.advice (canonical path)."""
        import sys
        from unittest.mock import MagicMock

        for mod in ("transformers", "langchain_openai", "datasets", "peft"):
            sys.modules.setdefault(mod, MagicMock())

        from foresightrlcore.tasks import load_task_module

        mod = load_task_module("advice")
        assert callable(mod.get_reward_function)
        assert callable(mod.get_callbacks)
        assert callable(mod.get_dataset_loader)

    def test_load_external_module(self, tmp_path):
        module_file = tmp_path / "custom_task.py"
        module_file.write_text(
            "def get_reward_function(tc): return lambda *a, **k: [0.5]\n"
            "def get_callbacks(tc): return []\n"
            "def get_dataset_loader(tc, mn): return lambda p: None\n"
        )
        from foresightrlcore.tasks import load_task_module

        mod = load_task_module(str(module_file))
        assert callable(mod.get_reward_function)
        assert callable(mod.get_callbacks)
        assert callable(mod.get_dataset_loader)
