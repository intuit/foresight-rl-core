"""Math reasoning task — DeepMath GRPO reward + HF-Hub dataset loader.

Reproduces TRL's accuracy-reward GRPO recipe
(https://huggingface.co/docs/trl/en/grpo_trainer) inside the framework:

* Dataset: ``trl-lib/DeepMath-103K`` loaded **directly from the Hugging
  Face Hub** (the mounted SageMaker train channel is ignored — the loader
  path argument is accepted for protocol compatibility but not read).
  Rows have a conversational ``prompt`` (list of chat messages) and a
  ``solution`` ground-truth answer.
* Reward: deterministic ``math_verify`` final-answer accuracy — **no LLM
  judge, no Genos, no secrets**. Scoring lives in
  :mod:`foresightrlcore.tasks.math.reward_logic`.

Transport-free helpers are reused from
:mod:`foresightrlcore.tasks.thinking_utils` (``completion_to_str``) so no
logic is duplicated from the ``advice`` task.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, List

from foresightrlcore.tasks.math.reward_logic import accuracy_score
from foresightrlcore.tasks.thinking_utils import completion_to_str

logger = logging.getLogger(__name__)

DEFAULT_HF_DATASET = "trl-lib/DeepMath-103K"
DEFAULT_HF_SPLIT = "train"
# Column names in trl-lib/DeepMath-103K (verified against the live dataset).
PROMPT_COLUMN = "prompt"
SOLUTION_COLUMN = "solution"
# The extra column TRL forwards to the reward function as a kwarg.
GROUND_TRUTH_COLUMN = "ground_truth"


# ============================================================================
# DATASET LOADER (Hugging Face Hub, not the mounted channel)
# ============================================================================


def _build_dataset_loader(task_config: Dict[str, Any], model_name: str) -> Callable:
    """Return a loader that pulls DeepMath from the HF Hub and chat-templates it."""

    hf_dataset = task_config.get("hf_dataset", DEFAULT_HF_DATASET)
    hf_split = task_config.get("hf_split", DEFAULT_HF_SPLIT)
    max_samples = task_config.get("max_samples")
    shuffle_seed = task_config.get("shuffle_seed", 42)
    enable_thinking = task_config.get("enable_thinking", True)

    def load_dataset_from_path(path: str):  # noqa: ARG001 - path ignored by design
        from datasets import load_dataset
        from transformers import AutoTokenizer

        logger.info(
            "Loading HF dataset %s (split=%s) — mounted path %r ignored",
            hf_dataset,
            hf_split,
            path,
        )
        ds = load_dataset(hf_dataset, split=hf_split)

        if shuffle_seed is not None:
            ds = ds.shuffle(seed=shuffle_seed)
        if max_samples is not None:
            ds = ds.select(range(min(int(max_samples), len(ds))))

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def _apply_template(messages: list) -> str:
            # `enable_thinking` is a Qwen3-family kwarg. Non-thinking chat
            # templates (e.g. Qwen2.5-Instruct) either ignore it or raise a
            # TypeError, so fall back to a plain call for those models.
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

        def _map_row(row: Dict[str, Any]) -> Dict[str, Any]:
            prompt = row[PROMPT_COLUMN]
            if isinstance(prompt, list):
                prompt = _apply_template(prompt)
            return {
                PROMPT_COLUMN: prompt,
                GROUND_TRUTH_COLUMN: str(row[SOLUTION_COLUMN]),
            }

        ds = ds.map(_map_row, remove_columns=ds.column_names)
        logger.info("Loaded %d DeepMath records", len(ds))
        return ds

    return load_dataset_from_path


# ============================================================================
# REWARD FUNCTION (deterministic math accuracy)
# ============================================================================


def _build_reward_function(predictions_log: str | None) -> Callable:
    reward_call_count = 0

    def math_reward_func(
        completions: List[Any],
        ground_truth: List[str],
        **kwargs: Any,
    ) -> List[float]:
        nonlocal reward_call_count
        reward_call_count += 1
        batch_id = reward_call_count
        total = len(completions)

        rewards: List[float] = []
        records: List[dict] = []
        for idx, (completion, gt) in enumerate(zip(completions, ground_truth)):
            content = completion_to_str(completion)
            score = accuracy_score(content, gt)
            rewards.append(score)
            records.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "batch_id": batch_id,
                    "index": idx,
                    "reward": score,
                    "ground_truth": gt,
                    "completion": content,
                }
            )

        avg = sum(rewards) / len(rewards) if rewards else 0.0
        logger.info(
            "[Batch %d] math accuracy — n=%d, avg=%.4f, correct=%d/%d",
            batch_id,
            total,
            avg,
            sum(1 for r in rewards if r > 0.5),
            total,
        )

        if predictions_log:
            try:
                log_dir = os.path.dirname(predictions_log)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                with open(predictions_log, "a", encoding="utf-8") as pf:
                    for rec in records:
                        pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except OSError as e:
                logger.warning("Failed to write prediction log: %s", e)

        return rewards

    return math_reward_func


# ============================================================================
# PUBLIC INTERFACE (TaskModule protocol)
# ============================================================================


def get_reward_function(task_config: Dict[str, Any]) -> Callable:
    """Factory: return the deterministic DeepMath accuracy reward function."""
    pred_log = task_config.get("predictions_log")
    return _build_reward_function(pred_log if isinstance(pred_log, str) else None)


def get_callbacks(task_config: Dict[str, Any]) -> list:  # noqa: ARG001
    """No custom callbacks — TRL already logs mean ``reward`` to TensorBoard."""
    return []


def get_dataset_loader(task_config: Dict[str, Any], model_name: str) -> Callable:
    """Return a callable that loads DeepMath from the HF Hub."""
    return _build_dataset_loader(task_config, model_name)
