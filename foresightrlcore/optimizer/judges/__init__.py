"""Judge implementations + helpers.

Importing this package registers the built-in judges (deterministic + LLM) in
:data:`foresightrlcore.optimizer.registry.JUDGES`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

# Side-effect imports populate the JUDGES registry.
from foresightrlcore.optimizer.judges import (  # noqa: F401
    aggregate,
    llm_judges,
)
from foresightrlcore.optimizer.judges.llm_judges import LLM_JUDGE_NAMES
from foresightrlcore.optimizer.registry import JUDGES

if TYPE_CHECKING:
    from foresightrlcore.config.optimizer_schema import JudgeSpec
    from foresightrlcore.optimizer.interfaces import Judge


def is_llm_judge(name: str) -> bool:
    """Return True if ``name`` is an LLM judge (and thus needs a client)."""
    return name in LLM_JUDGE_NAMES


def build_judge(spec: "JudgeSpec", manager: Any = None) -> "Judge":
    """Instantiate a judge from a :class:`JudgeSpec` and optional client manager.

    Args:
        spec: The judge selection (name / params / client).
        manager: An LLM client manager (required for LLM judges, ignored by
            deterministic judges).
    """
    factory = JUDGES.get(spec.name)
    judge = factory(spec.params, manager=manager)
    return cast("Judge", judge)


__all__ = [
    "is_llm_judge",
    "build_judge",
    "aggregate",
    "llm_judges",
    "LLM_JUDGE_NAMES",
]
