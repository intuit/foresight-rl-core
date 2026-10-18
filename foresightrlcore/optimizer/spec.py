"""Result schemas for the optimizer runner.

Pure data classes describing the JSON / parquet rows written to the output
directory by :mod:`foresightrlcore.optimizer.runner`. ``RowResult`` mirrors the
reference row keys; :meth:`RowResult.to_flat_row` produces the flattened dict
(one ``<judge>_score`` / ``<judge>_passed`` column per judge) that lands in
``results.parquet`` and the JSONL checkpoint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RowResult(BaseModel):
    """Result for a single ``(dataset, record, model, mode)`` combination."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    realm_id: str
    epoch_date: str
    model: str
    mode: str
    shard: Optional[Any] = None

    generation_error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    # Per-judge results keyed by judge name.
    judge_scores: Dict[str, int] = Field(default_factory=dict)
    judge_passed: Dict[str, bool] = Field(default_factory=dict)

    primary_metric_value: float = 0.0
    evaluation_status: str = "fail"
    gates_passed: bool = False
    all_passed: bool = False

    def resume_key(self) -> str:
        """Stable identifier used for checkpoint resume (skip completed rows)."""
        return (
            f"{self.dataset}|{self.realm_id}|{self.epoch_date}|{self.model}|{self.mode}"
        )

    def to_flat_row(self) -> Dict[str, Any]:
        """Flatten to the reference row schema (per-judge columns expanded)."""
        row: Dict[str, Any] = {
            "dataset": self.dataset,
            "realm_id": self.realm_id,
            "epoch_date": self.epoch_date,
            "model": self.model,
            "mode": self.mode,
            "shard": self.shard,
            "generation_error": self.generation_error,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
        }
        for name, score in self.judge_scores.items():
            row[f"{name}_score"] = score
        for name, passed in self.judge_passed.items():
            row[f"{name}_passed"] = passed
        row["primary_metric_value"] = self.primary_metric_value
        row["evaluation_status"] = self.evaluation_status
        row["gates_passed"] = self.gates_passed
        row["all_passed"] = self.all_passed
        return row


class OptimizerResult(BaseModel):
    """Top-level result document written as ``optimizer_results.json``.

    Self-contained: carries both the full per-row data (``rows`` - the same
    flattened records as ``results.parquet``) and a structured ``summary``
    report (overall + per-model/mode + per-judge stats), so the JSON alone is
    enough to reconstruct the run without reading the parquet.
    """

    model_config = ConfigDict(extra="forbid")

    job_name: str
    prompt_set: str
    baseline_model: Optional[str] = None
    modes: List[str] = Field(default_factory=list)
    judges: List[str] = Field(default_factory=list)
    acceptance_threshold: float = 0.80
    started_at: str = ""
    finished_at: Optional[str] = None
    n_rows: int = 0
    # Structured report: {"overall": {...}, "by_model_mode": [...],
    # "per_judge": [...]}. Mirrors the summary.md tables as data.
    summary: Dict[str, Any] = Field(default_factory=dict)
    # Full per-row results (flattened, one dict per dataset x record x model x
    # mode) - identical to the rows in results.parquet / the checkpoint JSONL.
    rows: List[Dict[str, Any]] = Field(default_factory=list)


__all__ = ["RowResult", "OptimizerResult"]
