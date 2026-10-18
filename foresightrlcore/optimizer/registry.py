"""The optimizer's pluggable-axis registries.

The generic :class:`Registry` lives in :mod:`foresightrlcore.common.registry`
and is re-exported here for callers that already import it from this module.
"""

from __future__ import annotations

from typing import Callable

from foresightrlcore.common.registry import Registry

# The four pluggable axes of the optimizer.
DATASETS: "Registry[Callable[..., object]]" = Registry("dataset")
PROMPT_SETS: "Registry[Callable[..., object]]" = Registry("prompt_set")
MODES: "Registry[Callable[..., object]]" = Registry("mode")
JUDGES: "Registry[Callable[..., object]]" = Registry("judge")


__all__ = ["Registry", "DATASETS", "PROMPT_SETS", "MODES", "JUDGES"]
