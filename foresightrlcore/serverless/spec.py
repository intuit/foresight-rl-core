"""Serverless customization-technique output schemas + Mapper contract.

This module is the **pluggable schema layer** for the dataset-converter
CLI (``rl-train serverless convert``). Each SageMaker Serverless
customization technique (RLVR, RLAIF, SFT, DPO, KTO, ...) has its own
JSONL record shape; we expose one pydantic model per technique and
register them all in :data:`SCHEMAS`.

Today only :class:`RLVRRecord` is implemented. ``SFTRecord``,
``DPORecord``, and ``KTORecord`` are reserved as named keys in
:data:`SCHEMAS` so adding them later is a single-file addition with no
CLI or mapper-contract changes.

The per-task mapper contract lives here too (:data:`Mapper` typedef).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# RLVR — implemented today
# ---------------------------------------------------------------------------


class RewardModel(BaseModel):
    """Per-sample reward-function payload inside an :class:`RLVRRecord`.

    ``ground_truth`` is whatever string the reward function needs at
    scoring time. For rule-based rewards (e.g. classification) it's the
    correct label; for LLM-judged rewards it's typically the context
    text the judge verifies the model's output against.

    ``style`` is dataset metadata — ``"rule"`` for deterministic
    rule-based rewards, ``"model"`` for LLM-judged rewards. The serverless
    trainer does not act on it; the reward Lambda does (or ignores it).
    """

    model_config = ConfigDict(extra="forbid")

    ground_truth: str = Field(
        description="Payload string handed to the reward Lambda at scoring time."
    )
    style: str = Field(
        description="Reward style — typically 'rule' or 'model'.",
    )


class RLVRRecord(BaseModel):
    """SageMaker Serverless RLVR / verl-style training record.

    This is the wire shape every RLVR converter must produce. The
    trainer feeds this dict to the configured custom reward Lambda at
    each rollout step.
    """

    model_config = ConfigDict(extra="forbid")

    data_source: str = Field(
        description="Dataset-level tag — used by the Lambda to dispatch "
        "task-specific scoring when one Lambda serves multiple tasks.",
    )
    prompt: List[Dict[str, Any]] = Field(
        description="Chat-template message list, e.g. "
        "[{'role': 'system', 'content': '...'}, {'role': 'user', ...}].",
    )
    ability: str = Field(
        description="Task family — e.g. 'generation', 'classification', "
        "'reasoning'. Free-form metadata, not validated against an enum.",
    )
    reward_model: RewardModel = Field(
        description="Per-sample reward-function payload (ground truth + style)."
    )
    extra_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional verl-convention pass-through bag for "
        "non-PII metadata the trainer/reward Lambda does not need.",
    )


# ---------------------------------------------------------------------------
# Deferred schemas — keys reserved so the CLI surface is stable.
#
# Implementing one of these later is a single-file addition: replace the
# ``None`` placeholder with the pydantic model, add a one-line test, and
# remove the matching ``NotImplementedError`` branch in ``get_schema``.
# ---------------------------------------------------------------------------

_DEFERRED_SCHEMAS: Dict[str, str] = {
    "sft": "Supervised fine-tuning input/output pairs.",
    "dpo": "Direct preference optimization (chosen vs. rejected pairs).",
    "kto": "Kahneman-Tversky optimization (one-sided preferences).",
}


# ---------------------------------------------------------------------------
# Schema registry
# ---------------------------------------------------------------------------

SCHEMAS: Dict[str, Optional[Type[BaseModel]]] = {
    "rlvr": RLVRRecord,
    # Deferred — see _DEFERRED_SCHEMAS for descriptions.
    "sft": None,
    "dpo": None,
    "kto": None,
}


def get_schema(name: str) -> Type[BaseModel]:
    """Return the pydantic model for a registered schema.

    Raises ``NotImplementedError`` for reserved-but-not-implemented
    schemas (``sft``, ``dpo``, ``kto``) with a clear message; raises
    ``ValueError`` for unknown names.
    """
    if name not in SCHEMAS:
        valid = sorted(SCHEMAS)
        raise ValueError(
            f"Unknown schema {name!r}. Available: {valid}. "
            "Add a new pydantic model in foresightrlcore/serverless/spec.py "
            "and register it in SCHEMAS to add a new one."
        )
    model = SCHEMAS[name]
    if model is None:
        reason = _DEFERRED_SCHEMAS.get(name, "deferred")
        raise NotImplementedError(
            f"Schema {name!r} is reserved but not implemented yet "
            f"({reason}). Implement the pydantic model in "
            f"foresightrlcore/serverless/spec.py and assign it to "
            f"SCHEMAS[{name!r}]."
        )
    return model


def available_schemas() -> List[str]:
    """Return schemas that are implemented (i.e. usable with ``--schema``)."""
    return sorted(name for name, model in SCHEMAS.items() if model is not None)


def reserved_schemas() -> List[str]:
    """Return schemas with reserved keys but no implementation yet."""
    return sorted(name for name, model in SCHEMAS.items() if model is None)


# ---------------------------------------------------------------------------
# Mapper contract
# ---------------------------------------------------------------------------

# A Mapper receives one source dict and returns:
#   * One ``BaseModel`` instance (the canonical case),
#   * ``None`` to skip the record (the mapper found it unusable but does
#     not want to raise), or
#   * An iterator/generator of ``BaseModel`` instances for fanout (one
#     input -> many outputs, e.g. data augmentation).
#
# The CLI loop validates each produced model against the requested
# ``--schema`` before writing.
MapperOutput = Union[BaseModel, None, Iterator[BaseModel]]
Mapper = Callable[..., MapperOutput]


__all__ = [
    "Mapper",
    "MapperOutput",
    "RLVRRecord",
    "RewardModel",
    "SCHEMAS",
    "available_schemas",
    "get_schema",
    "reserved_schemas",
]
