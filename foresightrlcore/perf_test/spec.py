"""Result schemas for the perf_test runner.

These are pure data classes -- they describe the JSON/YAML files written
to the SageMaker output directory by :mod:`foresightrlcore.perf_test.runner`.
No business logic lives here.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TpsResult(BaseModel):
    """Aggregate statistics for a single TPS level."""

    model_config = ConfigDict(extra="forbid")

    target_tps: int = Field(description="Configured target TPS for this run.")
    duration_seconds: int = Field(description="Steady-state duration in seconds.")
    actual_tps: Optional[float] = Field(
        default=None,
        description="Mean observed RPS reported by Gatling.",
    )
    requests_total: Optional[int] = Field(default=None)
    requests_ok: Optional[int] = Field(default=None)
    requests_ko: Optional[int] = Field(default=None)
    mean_latency_ms: Optional[float] = Field(default=None)
    p50_latency_ms: Optional[float] = Field(default=None)
    p75_latency_ms: Optional[float] = Field(default=None)
    p95_latency_ms: Optional[float] = Field(default=None)
    p99_latency_ms: Optional[float] = Field(default=None)
    max_latency_ms: Optional[float] = Field(default=None)
    gatling_report_dir: Optional[str] = Field(
        default=None,
        description="Path to the Gatling HTML report directory inside the "
        "container, copied to the SageMaker output mount.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error string when the run failed before producing stats.",
    )


class RunResult(BaseModel):
    """Result for one :class:`PerfTestRunConfig`."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Logical run name (mirrors config).")
    model: str = Field(description="Served model identifier.")
    simulation_class: str = Field(description="Gatling simulation FQCN.")
    started_at: str = Field(description="ISO-8601 UTC timestamp.")
    finished_at: Optional[str] = Field(default=None)
    tps_results: List[TpsResult] = Field(default_factory=list)
    vllm_startup_seconds: Optional[float] = Field(default=None)
    extra: Dict[str, str] = Field(default_factory=dict)


__all__ = ["TpsResult", "RunResult"]
