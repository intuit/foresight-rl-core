"""Math reasoning task — DeepMath GRPO with a deterministic accuracy reward.

Reproduces TRL's accuracy-reward GRPO recipe on ``trl-lib/DeepMath-103K``
(https://huggingface.co/docs/trl/en/grpo_trainer). Unlike the ``advice``
task there is no LLM judge, no Genos, and no ``secrets:`` section — the
reward is a deterministic ``math_verify`` final-answer check.

Submodules
----------
* :mod:`foresightrlcore.tasks.math.reward_logic` — stdlib + ``math_verify``
  scoring (``accuracy_score``). Single source of truth for the reward math.
* :mod:`foresightrlcore.tasks.math.trainer` — HF-Hub dataset loader + TRL
  reward function.

Public interface (consumed by ``foresightrlcore.tasks.load_task_module``):
:func:`get_reward_function`, :func:`get_callbacks`,
:func:`get_dataset_loader`.
"""

from __future__ import annotations

# Wrapped in try/except so importing this package never fails on machines
# without the training stack (``transformers`` / ``datasets``) — mirrors the
# ``advice`` package guard so registry/CLI paths stay importable.
try:
    from foresightrlcore.tasks.math.trainer import (  # noqa: F401
        get_callbacks,
        get_dataset_loader,
        get_reward_function,
    )
except ImportError:
    pass


__all__ = [
    "get_callbacks",
    "get_dataset_loader",
    "get_reward_function",
]
