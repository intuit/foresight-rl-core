"""
Task module: email_outcome

Predicts e-commerce email campaign outcomes (purchased, added_to_cart, etc.)
using GRPO reward shaping. Extracted from train_grpo_update.py (V6 reward).

Implements the TaskModule protocol:
  get_reward_function(task_config) -> Callable
  get_callbacks(task_config) -> list
  get_dataset_loader(task_config, model_name) -> Callable
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
from collections import deque
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel, Field, ValidationError

    class OutcomeCategory(str, Enum):
        purchased = "purchased"
        added_to_cart = "added_to_cart"
        browsed_after = "browsed_after"
        clicked = "clicked"
        opened = "opened"
        ignored = "ignored"
        unsubscribed = "unsubscribed"

    class C2OutcomePrediction(BaseModel):
        explanation: str = Field(
            description="Step-by-step reasoning for why you chose this category. "
            "Reference specific signals from the user profile, engagement "
            "history, and campaign content."
        )
        predicted_category: OutcomeCategory = Field(
            description="The single most likely outcome category for this user."
        )
        confidence: float = Field(
            description="Your confidence in the predicted category, between 0.0 and 1.0.",
            ge=0.0,
            le=1.0,
        )

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    ValidationError = Exception

VALID_CATEGORIES = [
    "purchased",
    "added_to_cart",
    "browsed_after",
    "clicked",
    "opened",
    "ignored",
    "unsubscribed",
]

# Engagement funnel order (low → high engagement)
CATEGORIES = [
    "unsubscribed",
    "ignored",
    "opened",
    "clicked",
    "browsed_after",
    "added_to_cart",
    "purchased",
]
CATEGORY_INDEX = {c: i for i, c in enumerate(CATEGORIES)}

CANONICAL_NAMES = {
    "ignored": "ignored",
    "ignore": "ignored",
    "opened": "opened",
    "open": "opened",
    "clicked": "clicked",
    "click": "clicked",
    "browsed_after": "browsed_after",
    "browsed": "browsed_after",
    "browsed_products": "browsed_after",
    "added_to_cart": "added_to_cart",
    "add_to_cart": "added_to_cart",
    "cart": "added_to_cart",
    "unsubscribed": "unsubscribed",
    "unsubscribe": "unsubscribed",
    "purchased": "purchased",
    "purchase": "purchased",
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You predict e-commerce email campaign outcomes.

INPUT: Brand info, User Profile, Campaign History, New Campaign, Derived Signals.

OUTCOME CATEGORIES (consider all — no single category dominates):
purchased | added_to_cart | browsed_after | clicked | opened | ignored | unsubscribed

PREDICTION RULES (apply in priority order):

1. **outcome_dist is your strongest signal.** If a user historically browsed_after 40% of the time, weight that heavily. Past behavior predicts future behavior.

2. **Session data outweighs email metrics.** sessions_30d > 0 means the user is engaged even if open_rate is zero. Trust site behavior over inbox behavior.

3. **purchased requires at least two conditions:** orders_alltime > 0 AND campaign-product match. Missing either → predict browsed_after or clicked instead.

4. **opened vs clicked:** opened means clicks_30d=0 AND sessions_30d=0. Any site activity → predict clicked or higher.

5. **unsubscribed comes from cross-brand behavior:** high unsubs_30d/90d + high brands_subscribed. Long ignore streaks alone do NOT predict unsubscribe.

6. **Recency matters most:** days_since_visit < 7 → likely active engagement. > 30 days → likely ignored or unsubscribed.

7. **Sale/non-sale pattern:** If user only engages with sale campaigns and this campaign is non-sale, drop engagement prediction by at least one tier.

8. **cart items + campaign product overlap** → raises tier probability even for cold-trend users.

REASONING: In <think>, analyze all signals freely. In the JSON explanation, summarize all of your findings and state the final signals that caused the prediction.

CONFIDENCE: 0.3-0.5 = ambiguous, 0.5-0.7 = leaning, 0.7-0.9 = strong, >0.9 = near-certain.

OUTPUT — valid JSON only:
```json
{
  "explanation": "Summarize all of your findings and state the final signals that caused the prediction.",
  "predicted_category": "one of the 7 categories",
  "confidence": 0.0-1.0
}
"""


# ---------------------------------------------------------------------------
# Reward defaults (V6)
# ---------------------------------------------------------------------------

_DEFAULTS = dict(
    think_length_soft_cap=1500,
    think_length_hard_cap=4000,
    think_penalty_weight=0.5,
    correct_base=0.4,
    correct_conf_scale=0.6,
    wrong_base=-0.4,
    wrong_conf_scale=0.7,
    near_miss_bonus_over={1: 0.08, 2: 0.04},
    near_miss_bonus_under={1: 0.15, 2: 0.08},
    min_explanation_length=80,
    lazy_explanation_penalty=-0.3,
    format_penalty_unparseable=-1.0,
    pred_window_size=200,
    reward_mode="rule_based",
    mock_constant_reward=0.5,
)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def get_reward_function(task_config: dict[str, Any]) -> Callable:
    cfg = {**_DEFAULTS, **task_config}
    reward_mode = cfg["reward_mode"]

    if reward_mode == "mock":
        constant = float(cfg["mock_constant_reward"])

        def mock_reward(completions, **kwargs):
            return [constant] * len(completions)

        return mock_reward

    if reward_mode == "rule_based":
        return _build_reward_fn(cfg)

    raise ValueError(f"Unknown reward_mode: {reward_mode!r}")


def get_callbacks(task_config: dict[str, Any]) -> list:
    return []


def get_dataset_loader(task_config: dict[str, Any], model_name: str) -> Callable:
    enable_thinking = task_config.get("enable_thinking", True)

    def load(path: str):
        import glob as _glob

        from datasets import Dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Support both a directory of JSONL files and a single JSONL file.
        if os.path.isdir(path):
            file_paths = sorted(_glob.glob(os.path.join(path, "*.jsonl")))
        else:
            file_paths = [path]

        records = []
        skipped = 0
        for fpath in file_paths:
            with open(fpath, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                        prompt = raw["prompt"]
                        outcome = raw.get("outcome_category", "")

                        # Apply chat template if prompt is a message list.
                        if isinstance(prompt, list):
                            # Inject / replace system prompt
                            has_system = any(m.get("role") == "system" for m in prompt)
                            if has_system:
                                prompt = [
                                    (
                                        {"role": "system", "content": SYSTEM_PROMPT}
                                        if m.get("role") == "system"
                                        else m
                                    )
                                    for m in prompt
                                ]
                            else:
                                prompt = [
                                    {"role": "system", "content": SYSTEM_PROMPT}
                                ] + prompt

                            prompt = tokenizer.apply_chat_template(
                                prompt,
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=enable_thinking,
                            )

                        records.append({"prompt": prompt, "outcome_category": outcome})
                    except (json.JSONDecodeError, KeyError) as e:
                        skipped += 1
                        if skipped <= 5:
                            logger.warning(f"Skipping line {i+1} in {fpath}: {e}")

        logger.info(f"Loaded {len(records)} records from {path} (skipped {skipped})")
        return Dataset.from_list(records)

    return load


# ---------------------------------------------------------------------------
# Reward internals
# ---------------------------------------------------------------------------


def _build_reward_fn(cfg: dict) -> Callable:
    think_soft = cfg["think_length_soft_cap"]
    think_hard = cfg["think_length_hard_cap"]
    think_weight = cfg["think_penalty_weight"]
    correct_base = cfg["correct_base"]
    correct_conf_scale = cfg["correct_conf_scale"]
    wrong_base = cfg["wrong_base"]
    wrong_conf_scale = cfg["wrong_conf_scale"]
    near_miss_over = cfg["near_miss_bonus_over"]
    near_miss_under = cfg["near_miss_bonus_under"]
    min_expl_len = cfg["min_explanation_length"]
    lazy_penalty = cfg["lazy_explanation_penalty"]
    format_penalty = cfg["format_penalty_unparseable"]
    window_size = cfg["pred_window_size"]

    # Seed rolling prediction history with uniform distribution.
    _pred_history: deque = deque(
        [c for c in CATEGORIES for _ in range(28)],
        maxlen=window_size,
    )
    _call_count = [0]

    def reward_fn(completions, outcome_category=None, **kwargs):
        _call_count[0] += 1
        batch_id = _call_count[0]
        if outcome_category is None:
            outcome_category = [""] * len(completions)

        logger.info(
            f"[Batch {batch_id}] Computing rewards for {len(completions)} completions"
        )
        rewards = []

        for idx, (completion, actual_cat) in enumerate(
            zip(completions, outcome_category)
        ):
            content = _extract_content(completion)
            actual = (actual_cat or "").strip().lower().replace(" ", "_")

            predicted, confidence, parse_method = _parse_prediction(content)

            if not predicted:
                reward = format_penalty
                risk_level = "unparseable"
                base_reward = format_penalty
                confidence_bonus = 0.0
                explanation_penalty = 0.0
                cat_scale = 1.0
            else:
                cat_scale = _get_category_scale(predicted, _pred_history)
                reward, risk_level, base_reward, confidence_bonus = _lookup_reward(
                    actual,
                    predicted,
                    confidence,
                    correct_base,
                    correct_conf_scale,
                    wrong_base,
                    wrong_conf_scale,
                    near_miss_over,
                    near_miss_under,
                    _pred_history,
                )
                explanation_penalty = _check_explanation_quality(
                    content, min_expl_len, lazy_penalty
                )
                reward += explanation_penalty

            think_len = _extract_thinking_length(content)
            think_penalty = _compute_thinking_penalty(
                content, think_soft, think_hard, think_weight
            )
            reward -= think_penalty
            rewards.append(reward)

            is_correct = actual == predicted
            logger.info(
                f"[Batch {batch_id} | {idx+1}/{len(completions)}] "
                f"reward={reward:.4f} base={base_reward:.4f} "
                f"conf_bonus={confidence_bonus:.4f} "
                f"think_penalty={think_penalty:.4f} "
                f"think_len={think_len} risk={risk_level} "
                f"confidence={confidence:.4f} parse={parse_method} "
                f"correct={is_correct} actual='{actual}' predicted='{predicted}'"
            )

        avg = sum(rewards) / len(rewards) if rewards else 0.0
        correct = sum(1 for r in rewards if r >= correct_base)
        unparseable = sum(1 for r in rewards if r == format_penalty)
        std = (
            (sum((r - avg) ** 2 for r in rewards) / len(rewards)) ** 0.5
            if rewards
            else 0.0
        )
        logger.info(
            f"[Batch {batch_id}] Done -- avg={avg:.4f}, std={std:.4f}, "
            f"correct={correct}/{len(rewards)}, unparseable={unparseable}/{len(rewards)}, "
            f"min={min(rewards):.4f}, max={max(rewards):.4f}"
        )

        _maybe_free_memory()
        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _extract_content(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return completion[0].get("content", "") if completion else ""
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def _strip_thinking_tags(content: str) -> str:
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    content = re.sub(r"<reasoning>.*?</reasoning>", "", content, flags=re.DOTALL)
    if "</think>" in content and "<think>" not in content:
        content = content.rsplit("</think>", 1)[-1]
    else:
        content = re.sub(r"</think>", "", content)
    return content.strip()


def _extract_thinking_length(content: str) -> int:
    match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
    if match:
        return len(match.group(1))
    if "</think>" in content and "<think>" not in content:
        return len(content.split("</think>")[0])
    return 0


def _compute_thinking_penalty(
    content: str, soft_cap: int, hard_cap: int, weight: float
) -> float:
    think_len = _extract_thinking_length(content)
    if think_len <= soft_cap:
        return 0.0
    excess = think_len - soft_cap
    cap_range = hard_cap - soft_cap
    return weight * min(1.0, excess / cap_range)


def _extract_json_from_markdown(content: str) -> str:
    start_tag = "```json"
    start = content.find(start_tag)
    if start >= 0:
        end = content.rfind("```")
        if end > start:
            return content[start + len(start_tag) : end].strip()
    return content


def _parse_prediction(content: str) -> tuple[str, float, str]:
    """Returns (category_str, confidence_float, parse_method_str).
    Only accepts valid JSON. No regex/keyword fallbacks.
    """
    content_clean = _strip_thinking_tags(content)
    json_str = _extract_json_from_markdown(content_clean)

    # 1. Pydantic (strict)
    if PYDANTIC_AVAILABLE:
        for candidate in [json_str, content_clean]:
            try:
                pred = C2OutcomePrediction.model_validate_json(candidate)
                return pred.predicted_category.value, pred.confidence, "pydantic"
            except (ValidationError, json.JSONDecodeError, Exception):
                pass

    # 2. Loose JSON
    for candidate in [json_str, content_clean]:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                confidence = 0.0
                raw_conf = data.get("confidence")
                if raw_conf is not None:
                    try:
                        parsed = float(raw_conf)
                        if 0.0 <= parsed <= 1.0:
                            confidence = parsed
                    except (ValueError, TypeError):
                        pass
                cat = (
                    data.get("predicted_category", "")
                    or data.get("outcome_category", "")
                    or data.get("prediction", "")
                    or data.get("outcome", "")
                    or data.get("predicted_outcome", "")
                    or data.get("category", "")
                )
                # Guard with isinstance: models occasionally emit non-string
                # category values (e.g. {"predicted_category": 0.7}) which
                # bypass the falsy check but blow up on .strip(). Falling
                # through to "unparseable" gives the model a small format
                # penalty rather than crashing the rank.
                # See bug analysis in intuit/docs/hyperpod-eks-setup-notes.md.
                if isinstance(cat, str) and cat:
                    normalized = cat.strip().lower().replace(" ", "_")
                    if normalized in VALID_CATEGORIES:
                        return normalized, confidence, "json"
                    mapped = CANONICAL_NAMES.get(normalized, "")
                    if mapped:
                        return mapped, confidence, "json"
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

    return "", 0.0, "none"


# ---------------------------------------------------------------------------
# Reward computation helpers
# ---------------------------------------------------------------------------


def _get_category_scale(predicted: str, pred_history: deque) -> float:
    """Scale in [0.3, 2.0] based on prediction frequency in rolling window."""
    if len(pred_history) < 14:
        return 1.0
    freq = sum(1 for p in pred_history if p == predicted) / len(pred_history)
    target = 1.0 / len(CATEGORIES)
    ratio = freq / target if target > 0 else 1.0
    if ratio < 0.01:
        return 2.0
    return max(0.3, min(2.0, 1.0 / ratio))


def _lookup_reward(
    actual: str,
    predicted: str,
    confidence: float,
    correct_base: float,
    correct_conf_scale: float,
    wrong_base: float,
    wrong_conf_scale: float,
    near_miss_over: dict,
    near_miss_under: dict,
    pred_history: deque,
) -> tuple[float, str, float, float]:
    ai = CATEGORY_INDEX.get(actual)
    pi = CATEGORY_INDEX.get(predicted)

    if ai is None:
        return -0.5, "unknown_actual", -0.5, 0.0
    if pi is None:
        return -0.5, "unknown_predicted", -0.5, 0.0

    is_correct = actual == predicted

    if is_correct:
        base_reward = correct_base
        confidence_bonus = correct_conf_scale * confidence
    else:
        base_reward = wrong_base
        confidence_bonus = -wrong_conf_scale * confidence
        dist = abs(ai - pi)
        if pi > ai:  # overestimation
            base_reward += near_miss_over.get(dist, 0.0)
        else:  # underestimation
            base_reward += near_miss_under.get(dist, 0.0)

    raw_reward = base_reward + confidence_bonus

    cat_scale = _get_category_scale(predicted, pred_history)
    if is_correct:
        raw_reward *= cat_scale
    else:
        wrong_scale = max(0.3, min(2.0, 1.0 / cat_scale))
        raw_reward *= wrong_scale

    pred_history.append(predicted)

    dist = abs(ai - pi)
    risk_level = (
        f"wrong_dist{dist}"
        if not is_correct and dist <= 2
        else ("none" if is_correct else "wrong")
    )
    return raw_reward, risk_level, base_reward, confidence_bonus


def _check_explanation_quality(
    content: str, min_length: int, lazy_penalty: float
) -> float:
    clean = _strip_thinking_tags(content)
    json_str = _extract_json_from_markdown(clean)
    for candidate in [json_str, clean]:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                explanation = data.get("explanation", "")
                return lazy_penalty if len(explanation) < min_length else 0.0
        except (json.JSONDecodeError, TypeError):
            pass
    return lazy_penalty if len(clean) < min_length else 0.0


def _maybe_free_memory():
    try:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        gc.collect()
