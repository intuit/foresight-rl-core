"""Tests for foresightrlcore.tasks.email_outcome — email outcome reward module.

Focus is the pure-Python reward computation and parsing helpers. The
``get_dataset_loader`` path is exercised only at the contract level (returns
a callable) because actually loading data requires ``datasets`` + an
``AutoTokenizer`` round-trip that we don't want to spin up here.
"""

from __future__ import annotations

from collections import deque
from unittest.mock import patch

import pytest

from foresightrlcore.tasks.email_outcome import (
    CANONICAL_NAMES,
    CATEGORIES,
    CATEGORY_INDEX,
    VALID_CATEGORIES,
    _build_reward_fn,
    _check_explanation_quality,
    _compute_thinking_penalty,
    _extract_content,
    _extract_json_from_markdown,
    _extract_thinking_length,
    _get_category_scale,
    _lookup_reward,
    _maybe_free_memory,
    _parse_prediction,
    _strip_thinking_tags,
    get_callbacks,
    get_dataset_loader,
    get_reward_function,
)

# ─── _extract_content ────────────────────────────────────────────────────


class TestExtractContent:
    def test_string(self):
        assert _extract_content("hello") == "hello"

    def test_message_list(self):
        completion = [{"role": "assistant", "content": "answer"}]
        assert _extract_content(completion) == "answer"

    def test_empty_list(self):
        assert _extract_content([]) == ""

    def test_message_list_missing_content_key(self):
        # First element has no "content" — falls back to empty string
        completion = [{"role": "assistant"}]
        assert _extract_content(completion) == ""

    def test_dict(self):
        assert _extract_content({"content": "x"}) == "x"

    def test_dict_missing_content(self):
        assert _extract_content({"other": "y"}) == ""

    def test_fallback_to_str(self):
        # Falls through the dict/list/str checks → str()
        assert _extract_content(123) == "123"


# ─── _strip_thinking_tags ────────────────────────────────────────────────


class TestStripThinkingTags:
    def test_no_tags(self):
        assert _strip_thinking_tags("plain text") == "plain text"

    def test_full_think_block(self):
        assert _strip_thinking_tags("<think>inner</think>visible") == "visible"

    def test_full_reasoning_block(self):
        assert _strip_thinking_tags("<reasoning>inner</reasoning>visible") == "visible"

    def test_orphan_close_tag(self):
        # </think> without matching <think> → keep only the part after it
        assert _strip_thinking_tags("prefix</think>visible") == "visible"

    def test_orphan_close_tag_with_balanced_block_uses_rsplit(self):
        # When there's an unbalanced extra </think>, rsplit takes the last
        # segment.
        assert _strip_thinking_tags("a</think>middle</think>final") == "final"

    def test_multiline_think(self):
        assert _strip_thinking_tags("<think>line1\nline2</think>after") == "after"


# ─── _extract_thinking_length ────────────────────────────────────────────


class TestExtractThinkingLength:
    def test_balanced(self):
        assert _extract_thinking_length("<think>12345</think>rest") == 5

    def test_orphan_close_tag(self):
        # No opening <think>, count everything before </think>
        assert _extract_thinking_length("abc</think>rest") == 3

    def test_no_tags(self):
        assert _extract_thinking_length("plain") == 0

    def test_empty(self):
        assert _extract_thinking_length("") == 0


# ─── _compute_thinking_penalty ───────────────────────────────────────────


class TestComputeThinkingPenalty:
    def test_below_soft_cap(self):
        # 10 chars of thinking, soft cap 100 → no penalty
        content = "<think>" + "x" * 10 + "</think>"
        assert _compute_thinking_penalty(content, 100, 200, 0.5) == 0.0

    def test_at_soft_cap_boundary(self):
        content = "<think>" + "x" * 100 + "</think>"
        # At cap → still 0 (cap inclusive)
        assert _compute_thinking_penalty(content, 100, 200, 0.5) == 0.0

    def test_partial_excess(self):
        # 150 chars, soft=100, hard=200, weight=0.5 → 0.5 × (50/100) = 0.25
        content = "<think>" + "x" * 150 + "</think>"
        assert _compute_thinking_penalty(content, 100, 200, 0.5) == pytest.approx(0.25)

    def test_above_hard_cap_clamps_to_one(self):
        # 500 chars, soft=100, hard=200 — excess 400 / range 100 = 4.0 → clamp to 1.0
        # penalty = weight × 1.0 = 0.5
        content = "<think>" + "x" * 500 + "</think>"
        assert _compute_thinking_penalty(content, 100, 200, 0.5) == 0.5


# ─── _extract_json_from_markdown ─────────────────────────────────────────


class TestExtractJsonFromMarkdown:
    def test_json_block(self):
        s = 'pre ```json\n{"a":1}\n``` post'
        assert _extract_json_from_markdown(s).strip() == '{"a":1}'

    def test_no_fence_returns_input(self):
        s = '{"a":1}'
        assert _extract_json_from_markdown(s) == s

    def test_unclosed_fence_returns_input(self):
        # When there's only one ``` (the json one), rfind == start, so we don't
        # strip — return content unchanged.
        s = "```json hello"
        assert _extract_json_from_markdown(s) == s


# ─── _parse_prediction ───────────────────────────────────────────────────


class TestParsePrediction:
    def test_valid_pydantic_json(self):
        s = (
            '{"explanation": "user is highly engaged", '
            '"predicted_category": "clicked", "confidence": 0.85}'
        )
        cat, conf, method = _parse_prediction(s)
        assert cat == "clicked"
        assert conf == pytest.approx(0.85)
        assert method == "pydantic"

    def test_valid_in_json_codefence(self):
        s = (
            "```json\n"
            '{"explanation": "ok", "predicted_category": "purchased", '
            '"confidence": 0.95}\n'
            "```"
        )
        cat, conf, method = _parse_prediction(s)
        assert cat == "purchased"
        assert method == "pydantic"

    def test_loose_json_canonical_mapping(self):
        # Falls past pydantic (missing required explanation field) then loose JSON.
        # "click" → CANONICAL_NAMES → "clicked"
        s = '{"predicted_category": "click", "confidence": 0.5}'
        cat, conf, method = _parse_prediction(s)
        assert cat == "clicked"
        assert method in {"pydantic", "json"}

    def test_loose_json_alternate_field_name(self):
        # "outcome" key instead of "predicted_category"
        s = '{"outcome": "opened", "confidence": 0.4}'
        cat, _, method = _parse_prediction(s)
        assert cat == "opened"
        assert method == "json"

    def test_unknown_category_falls_through(self):
        s = '{"predicted_category": "absolute_garbage", "confidence": 0.5}'
        cat, conf, method = _parse_prediction(s)
        assert cat == ""
        assert conf == 0.0
        assert method == "none"

    def test_non_string_category_does_not_crash(self):
        # Regression test for the cat.strip() crash on float values.
        # The fix at line 472 guards with isinstance(cat, str).
        s = '{"predicted_category": 0.7, "confidence": 0.5}'
        cat, conf, method = _parse_prediction(s)
        assert cat == ""
        assert method == "none"

    def test_confidence_out_of_range_falls_back_to_zero(self):
        # Loose JSON path: confidence outside [0, 1] is silently set to 0.
        s = '{"predicted_category": "opened", "confidence": 2.5}'
        cat, conf, method = _parse_prediction(s)
        assert cat == "opened"
        assert conf == 0.0

    def test_confidence_non_numeric_falls_back_to_zero(self):
        s = '{"predicted_category": "opened", "confidence": "very high"}'
        cat, conf, _ = _parse_prediction(s)
        assert conf == 0.0

    def test_invalid_json(self):
        cat, conf, method = _parse_prediction("not json at all")
        assert cat == ""
        assert method == "none"

    def test_strips_thinking_before_parsing(self):
        s = (
            "<think>reasoning here</think>"
            '{"predicted_category": "clicked", "confidence": 0.6, '
            '"explanation": "engagement"}'
        )
        cat, _, _ = _parse_prediction(s)
        assert cat == "clicked"

    def test_canonical_names_complete_for_valid_categories(self):
        # Every category in the official list must map to itself via
        # CANONICAL_NAMES — otherwise the lookup table is internally inconsistent.
        for c in VALID_CATEGORIES:
            assert CANONICAL_NAMES.get(c) == c


# ─── _get_category_scale ─────────────────────────────────────────────────


class TestGetCategoryScale:
    def test_short_history_returns_one(self):
        history = deque(["clicked"] * 5, maxlen=200)
        # < 14 → uniform scaling
        assert _get_category_scale("clicked", history) == 1.0

    def test_balanced_history_scale_around_one(self):
        # Uniform distribution across 7 categories → ratio = 1 → scale = 1
        history = deque([c for c in CATEGORIES for _ in range(2)], maxlen=200)
        assert _get_category_scale("clicked", history) == pytest.approx(1.0)

    def test_dominant_predicted_gets_low_scale(self):
        # 14 of "clicked" only → freq = 1.0, ratio = 7, raw 1/7 ≈ 0.143;
        # clamp = max(0.3, min(2.0, 1/ratio)) → 0.3 (the floor wins).
        history = deque(["clicked"] * 14, maxlen=200)
        assert _get_category_scale("clicked", history) == pytest.approx(0.3)

    def test_clamps_to_low_floor(self):
        # Even more dominant → still floored at 0.3
        history = deque(["clicked"] * 100, maxlen=200)
        assert _get_category_scale("clicked", history) == pytest.approx(0.3)

    def test_rare_predicted_gets_full_two(self):
        # 14 of "opened", asking about "purchased" never seen → ratio < 0.01 → 2.0
        history = deque(["opened"] * 14, maxlen=200)
        assert _get_category_scale("purchased", history) == 2.0


# ─── _lookup_reward ──────────────────────────────────────────────────────


class TestLookupReward:
    def _make_history(self):
        # Uniform history so cat_scale ≈ 1
        return deque([c for c in CATEGORIES for _ in range(2)], maxlen=200)

    def test_correct_high_confidence(self):
        history = self._make_history()
        reward, risk, base, conf_bonus = _lookup_reward(
            actual="clicked",
            predicted="clicked",
            confidence=0.9,
            correct_base=0.4,
            correct_conf_scale=0.6,
            wrong_base=-0.4,
            wrong_conf_scale=0.5,
            near_miss_over={1: 0.1, 2: 0.05},
            near_miss_under={1: 0.1, 2: 0.05},
            pred_history=history,
        )
        # base 0.4 + 0.6 × 0.9 = 0.94, scaled by ~1 → ~0.94
        assert base == 0.4
        assert conf_bonus == pytest.approx(0.54)
        assert reward == pytest.approx(0.94, abs=0.1)
        assert risk == "none"

    def test_wrong_distant_no_near_miss(self):
        history = self._make_history()
        reward, risk, _, _ = _lookup_reward(
            actual="purchased",  # idx 6
            predicted="ignored",  # idx 1, dist=5
            confidence=0.5,
            correct_base=0.4,
            correct_conf_scale=0.6,
            wrong_base=-0.4,
            wrong_conf_scale=0.5,
            near_miss_over={1: 0.1, 2: 0.05},
            near_miss_under={1: 0.1, 2: 0.05},
            pred_history=history,
        )
        # No near-miss bonus (dist=5), so base stays -0.4, conf_bonus -0.25
        # risk = "wrong" (not within 2 steps)
        assert risk == "wrong"
        assert reward < 0  # net negative

    def test_wrong_near_miss_underestimation(self):
        history = self._make_history()
        # actual is more engaged than predicted → underestimation; near_miss_under applies
        reward, risk, base, _ = _lookup_reward(
            actual="clicked",  # idx 3
            predicted="opened",  # idx 2, dist 1
            confidence=0.5,
            correct_base=0.4,
            correct_conf_scale=0.6,
            wrong_base=-0.4,
            wrong_conf_scale=0.5,
            near_miss_over={1: 0.1, 2: 0.05},
            near_miss_under={1: 0.15, 2: 0.08},  # underestimation bonus
            pred_history=history,
        )
        # base = -0.4 + 0.15 (under-1) = -0.25
        assert base == pytest.approx(-0.25)
        assert risk == "wrong_dist1"

    def test_wrong_near_miss_overestimation(self):
        history = self._make_history()
        # predicted more engaged than actual → overestimation; near_miss_over applies
        reward, risk, base, _ = _lookup_reward(
            actual="opened",  # idx 2
            predicted="clicked",  # idx 3, dist 1
            confidence=0.5,
            correct_base=0.4,
            correct_conf_scale=0.6,
            wrong_base=-0.4,
            wrong_conf_scale=0.5,
            near_miss_over={1: 0.1, 2: 0.05},
            near_miss_under={1: 0.15, 2: 0.08},
            pred_history=history,
        )
        assert base == pytest.approx(-0.3)  # -0.4 + 0.1
        assert risk == "wrong_dist1"

    def test_unknown_actual_returns_fixed_penalty(self):
        reward, risk, base, _ = _lookup_reward(
            actual="not_a_category",
            predicted="opened",
            confidence=0.5,
            correct_base=0.4,
            correct_conf_scale=0.6,
            wrong_base=-0.4,
            wrong_conf_scale=0.5,
            near_miss_over={},
            near_miss_under={},
            pred_history=deque(maxlen=200),
        )
        assert reward == -0.5
        assert risk == "unknown_actual"
        assert base == -0.5

    def test_unknown_predicted_returns_fixed_penalty(self):
        reward, risk, _, _ = _lookup_reward(
            actual="opened",
            predicted="not_a_category",
            confidence=0.5,
            correct_base=0.4,
            correct_conf_scale=0.6,
            wrong_base=-0.4,
            wrong_conf_scale=0.5,
            near_miss_over={},
            near_miss_under={},
            pred_history=deque(maxlen=200),
        )
        assert reward == -0.5
        assert risk == "unknown_predicted"


# ─── _check_explanation_quality ──────────────────────────────────────────


class TestCheckExplanationQuality:
    def test_long_explanation_no_penalty(self):
        s = (
            '{"explanation": "' + "x" * 100 + '", '
            '"predicted_category": "clicked", "confidence": 0.5}'
        )
        assert _check_explanation_quality(s, min_length=80, lazy_penalty=-0.3) == 0.0

    def test_short_explanation_gets_penalty(self):
        s = (
            '{"explanation": "too short", '
            '"predicted_category": "clicked", "confidence": 0.5}'
        )
        assert _check_explanation_quality(s, min_length=80, lazy_penalty=-0.3) == -0.3

    def test_no_json_falls_back_to_total_length(self):
        # No parseable JSON → measure total cleaned-content length
        assert (
            _check_explanation_quality("short", min_length=80, lazy_penalty=-0.3)
            == -0.3
        )

    def test_no_json_long_text_no_penalty(self):
        assert (
            _check_explanation_quality("x" * 100, min_length=80, lazy_penalty=-0.3)
            == 0.0
        )


# ─── _maybe_free_memory ──────────────────────────────────────────────────


class TestMaybeFreeMemory:
    def test_no_torch_path_runs(self):
        # Pretend torch is unimportable — should not raise.
        with patch.dict("sys.modules", {"torch": None}):
            _maybe_free_memory()

    def test_called_directly_does_not_raise(self):
        # In the actual lint environment torch is not installed; the function
        # falls back to gc.collect() and exits cleanly.
        _maybe_free_memory()


# ─── Public TaskModule API ───────────────────────────────────────────────


class TestPublicAPI:
    def test_get_callbacks_is_empty(self):
        assert get_callbacks({}) == []
        assert get_callbacks({"reward_mode": "rule_based"}) == []

    def test_get_dataset_loader_returns_callable(self):
        loader = get_dataset_loader({}, model_name="Qwen/Qwen3-8B")
        assert callable(loader)

    def test_get_reward_function_mock_mode(self):
        fn = get_reward_function({"reward_mode": "mock", "mock_constant_reward": 0.42})
        rewards = fn(["a", "b", "c"])
        assert rewards == [0.42, 0.42, 0.42]

    def test_get_reward_function_mock_uses_default_constant(self):
        fn = get_reward_function({"reward_mode": "mock"})
        rewards = fn(["x", "y"])
        # Default mock_constant_reward = 0.5
        assert rewards == [0.5, 0.5]

    def test_get_reward_function_rule_based_returns_callable(self):
        fn = get_reward_function({"reward_mode": "rule_based"})
        assert callable(fn)

    def test_get_reward_function_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown reward_mode"):
            get_reward_function({"reward_mode": "nonsense"})

    def test_get_reward_function_defaults_to_rule_based_when_unset(self):
        # No reward_mode in config → defaults dict provides "rule_based"
        fn = get_reward_function({})
        assert callable(fn)


# ─── End-to-end reward computation ───────────────────────────────────────


class TestRuleBasedRewardEndToEnd:
    def _fn(self, **overrides):
        cfg = {
            "reward_mode": "rule_based",
            "think_length_soft_cap": 200,
            "think_length_hard_cap": 400,
            "think_penalty_weight": 0.5,
            "correct_base": 0.4,
            "correct_conf_scale": 0.6,
            "wrong_base": -0.4,
            "wrong_conf_scale": 0.5,
            "near_miss_bonus_over": {1: 0.1},
            "near_miss_bonus_under": {1: 0.15},
            "min_explanation_length": 50,
            "lazy_explanation_penalty": -0.2,
            "format_penalty_unparseable": -1.0,
            "pred_window_size": 50,
        }
        cfg.update(overrides)
        return _build_reward_fn(cfg)

    def _build_completion(self, category: str, confidence: float = 0.8) -> str:
        # Long explanation to pass quality check
        return (
            '{"explanation": "' + "x" * 80 + '", '
            f'"predicted_category": "{category}", '
            f'"confidence": {confidence}}}'
        )

    def test_correct_high_confidence_positive_reward(self):
        fn = self._fn()
        rewards = fn(
            completions=[self._build_completion("clicked", 0.9)],
            outcome_category=["clicked"],
        )
        assert len(rewards) == 1
        assert rewards[0] > 0

    def test_unparseable_gets_format_penalty(self):
        fn = self._fn()
        rewards = fn(
            completions=["complete garbage no JSON here"],
            outcome_category=["clicked"],
        )
        # format_penalty (-1.0) plus think penalty (0) plus explanation penalty
        # (0 because the unparseable branch sets it to 0 in the reward loop)
        assert rewards[0] == pytest.approx(-1.0, abs=0.01)

    def test_wrong_direction_negative_reward(self):
        fn = self._fn()
        rewards = fn(
            completions=[self._build_completion("ignored", 0.9)],
            outcome_category=["purchased"],
        )
        # Far miss, high confidence → very negative
        assert rewards[0] < 0

    def test_outcome_category_none_treated_as_empty_strings(self):
        # Passing outcome_category=None means actuals are unknown — every
        # completion's actual becomes "" → falls into the unknown_actual path
        # and gets -0.5 base reward (not -1.0 format penalty).
        fn = self._fn()
        rewards = fn(
            completions=[self._build_completion("clicked", 0.5)],
            outcome_category=None,
        )
        assert len(rewards) == 1
        # Not the format penalty (-1.0) — that's reserved for unparseable
        # completions. We parsed successfully but actual is unknown.
        assert rewards[0] != pytest.approx(-1.0, abs=0.01)

    def test_batch_returns_one_reward_per_completion(self):
        fn = self._fn()
        completions = [
            self._build_completion("purchased", 0.9),
            self._build_completion("clicked", 0.6),
            self._build_completion("ignored", 0.4),
        ]
        rewards = fn(
            completions=completions,
            outcome_category=["purchased", "opened", "ignored"],
        )
        assert len(rewards) == 3

    def test_thinking_penalty_applied(self):
        fn = self._fn()
        # Big <think> block above hard cap → max think penalty
        long_think = "<think>" + "y" * 1000 + "</think>"
        with_think = long_think + self._build_completion("clicked", 0.9)
        without_think = self._build_completion("clicked", 0.9)
        with_reward = fn(completions=[with_think], outcome_category=["clicked"])[0]
        without_reward = fn(completions=[without_think], outcome_category=["clicked"])[
            0
        ]
        # Big think block costs the full weight (0.5)
        assert without_reward - with_reward == pytest.approx(0.5, abs=0.05)


# ─── Constants sanity ────────────────────────────────────────────────────


class TestModuleConstants:
    def test_valid_categories_and_categories_match(self):
        assert sorted(VALID_CATEGORIES) == sorted(CATEGORIES)

    def test_category_index_is_self_consistent(self):
        for i, c in enumerate(CATEGORIES):
            assert CATEGORY_INDEX[c] == i

    def test_canonical_names_only_map_to_valid_categories(self):
        for src, dest in CANONICAL_NAMES.items():
            assert dest in VALID_CATEGORIES
