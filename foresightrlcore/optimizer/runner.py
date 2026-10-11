"""Optimizer runner: resolve records -> cross-product eval -> results + summary.

Faithful synchronous generalization of the reference ``run.py``. Differences:

* **Concurrency**: a bounded ``ThreadPoolExecutor`` runs the per-row evaluation
  (generation + judging). This only changes execution order/speed -- per-row
  results are identical to the reference's sequential loop.
* **Grouped serving lifecycle**: the cross-product is grouped by candidate model
  so a ``provider=local`` candidate brackets its whole batch with a single
  ``vllm serve`` lifecycle (only one local model resident at a time).
* **Checkpoint / resume**: each completed row is appended to a JSONL checkpoint;
  a resumed run skips already-evaluated combinations.

Outputs written to the output dir: ``results.parquet``, ``summary.md``, and
``optimizer_results.json``.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from foresightrlcore.common import io as optim_io
from foresightrlcore.config.optimizer_schema import (
    JudgeSpec,
    OptimizerJobConfig,
    OptimizerModelConfig,
)
from foresightrlcore.optimizer.interfaces import (
    DatasetProvider,
    GenerationClient,
    GenerationMode,
    GenResult,
    Record,
)
from foresightrlcore.optimizer.judges import build_judge, is_llm_judge
from foresightrlcore.optimizer.judges.aggregate import (
    aggregate_judge_scores,
    calculate_summary_statistics,
)
from foresightrlcore.optimizer.prompts import make_prompt_set
from foresightrlcore.optimizer.registry import DATASETS, MODES
from foresightrlcore.optimizer.spec import OptimizerResult, RowResult

logger = logging.getLogger("foresightrlcore.optimizer.runner")

# Container paths (SageMaker training-job convention).
DEFAULT_OUTPUT_DIR = "/opt/ml/output/data"
DEFAULT_CHECKPOINT_DIR = "/opt/ml/checkpoints"
DEFAULT_CONFIG_PATH = "/opt/ml/input/data/config/args.yaml"
DEFAULT_DATASET_DOWNLOAD_DIR = "/tmp/optimizer_data"

_CHECKPOINT_FILENAME = "optimizer_checkpoint.jsonl"

# Factory types (overridable in tests).
GenClientFactory = Callable[[OptimizerModelConfig], GenerationClient]
ManagerResolver = Callable[[str], Any]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _cost_usd(
    prompt_tokens: int, completion_tokens: int, price_in: float, price_out: float
) -> float:
    """USD cost given per-1K-token prices (reference formula)."""
    return (prompt_tokens / 1000.0) * price_in + (
        completion_tokens / 1000.0
    ) * price_out


class OptimizerRunner:
    """Run one :class:`OptimizerJobConfig` end-to-end."""

    def __init__(
        self,
        config: OptimizerJobConfig,
        *,
        output_dir: Path,
        checkpoint_dir: Optional[Path] = None,
        dataset_download_dir: Optional[Path] = None,
        gen_client_factory: Optional[GenClientFactory] = None,
        manager_resolver: Optional[ManagerResolver] = None,
    ) -> None:
        self.config = config
        self.run_cfg = config.optimizer
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir else self.output_dir
        )
        self.dataset_download_dir = Path(
            dataset_download_dir or DEFAULT_DATASET_DOWNLOAD_DIR
        )
        self._gen_client_factory = gen_client_factory or self._default_gen_factory
        self._manager_resolver = manager_resolver or self._default_manager_resolver
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Default factories
    # ------------------------------------------------------------------

    def _default_gen_factory(self, model: OptimizerModelConfig) -> GenerationClient:
        from foresightrlcore.generation import make_generation_client

        return make_generation_client(
            model,
            max_retries=self.run_cfg.retry.max_retries,
            base_delay=self.run_cfg.retry.base_delay,
        )

    def _default_manager_resolver(self, client_name: str) -> Any:
        from foresightrlcore.clients.llm import resolve_llm_manager

        if self.run_cfg.judge_clients is None:
            raise ValueError(
                "an LLM judge is configured but optimizer.judge_clients is empty"
            )
        return resolve_llm_manager(self.run_cfg.judge_clients.get(client_name))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _resolve_records(self) -> List[Record]:
        sample = {
            "size": self.run_cfg.sample.size,
            "seed": self.run_cfg.sample.seed,
            "stratify_by": self.run_cfg.sample.stratify_by,
        }
        records: List[Record] = []
        for ds in self.run_cfg.datasets:
            spec = ds.as_provider_spec()
            if spec.get("path"):
                local = optim_io.resolve_dataset_path(
                    spec["path"], download_dir=self.dataset_download_dir
                )
                spec["path"] = str(local)
            provider = cast(DatasetProvider, DATASETS.get(spec["name"])())
            records.extend(provider.records(spec, sample))
        logger.info(
            "resolved %d records across %d dataset(s)",
            len(records),
            len(self.run_cfg.datasets),
        )
        return records

    def _build_judges(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        judges: Dict[str, Any] = {}
        weights: Dict[str, float] = {}
        managers: Dict[str, Any] = {}
        for spec in self.run_cfg.judges:
            manager = None
            if is_llm_judge(spec.name):
                if spec.client not in managers:
                    managers[spec.client] = self._manager_resolver(spec.client)
                manager = managers[spec.client]
            judge = build_judge(spec, manager)
            # Thread configured retry into LLM judges when supported.
            if hasattr(judge, "_max_retries"):
                setattr(judge, "_max_retries", self.run_cfg.retry.max_retries)
                setattr(judge, "_base_delay", self.run_cfg.retry.base_delay)
            judges[spec.name] = judge
            weights[spec.name] = spec.weight
        return judges, weights

    # ------------------------------------------------------------------
    # Per-row evaluation
    # ------------------------------------------------------------------

    def _run_one(
        self,
        record: Record,
        model: OptimizerModelConfig,
        mode_name: str,
        client: GenerationClient,
        prompt_set: Any,
        judges: Dict[str, Any],
        weights: Dict[str, float],
        dataset_name: str,
    ) -> RowResult:
        mode = cast(GenerationMode, MODES.get(mode_name)())
        try:
            gen: GenResult = mode.generate(
                record, client, prompt_set, max_tokens=model.max_tokens
            )
        except Exception as exc:  # noqa: BLE001 - record error, keep going
            gen = GenResult(text="", error=f"{type(exc).__name__}: {exc}")

        # Short-circuit a failed generation: do NOT judge empty output or let it
        # count as a pass. Record the error and mark the row's status "error" so
        # it is excluded from pass% / the primary metric (but stays in the total).
        if gen.error:
            logger.warning(
                "generation failed for %s/%s (model=%s): %s",
                record.realm_id,
                mode_name,
                model.name,
                gen.error,
            )
            return RowResult(
                dataset=dataset_name,
                realm_id=record.realm_id,
                epoch_date=record.epoch_date,
                model=model.name,
                mode=mode_name,
                shard=record.meta.get("shard"),
                generation_error=gen.error,
                prompt_tokens=gen.prompt_tokens,
                completion_tokens=gen.completion_tokens,
                total_tokens=gen.total_tokens,
                latency_ms=round(gen.latency_ms, 1),
                cost_usd=round(
                    _cost_usd(
                        gen.prompt_tokens,
                        gen.completion_tokens,
                        model.price_in,
                        model.price_out,
                    ),
                    6,
                ),
                judge_scores={},
                judge_passed={},
                primary_metric_value=0.0,
                evaluation_status="error",
                gates_passed=False,
                all_passed=False,
            )

        scores = {}
        judge_scores: Dict[str, int] = {}
        judge_passed: Dict[str, bool] = {}
        for name, judge in judges.items():
            verdict = judge.score(record, gen)
            scores[name] = verdict
            judge_scores[name] = verdict.score
            judge_passed[name] = verdict.passed

        agg = aggregate_judge_scores(scores, weights, self.run_cfg.acceptance_threshold)

        return RowResult(
            dataset=dataset_name,
            realm_id=record.realm_id,
            epoch_date=record.epoch_date,
            model=model.name,
            mode=mode_name,
            shard=record.meta.get("shard"),
            generation_error=gen.error or "",
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            total_tokens=gen.total_tokens,
            latency_ms=round(gen.latency_ms, 1),
            cost_usd=round(
                _cost_usd(
                    gen.prompt_tokens,
                    gen.completion_tokens,
                    model.price_in,
                    model.price_out,
                ),
                6,
            ),
            judge_scores=judge_scores,
            judge_passed=judge_passed,
            primary_metric_value=agg["primary_metric_value"],
            evaluation_status=agg["evaluation_status"],
            gates_passed=agg["gates_passed"],
            all_passed=agg["all_passed"],
        )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> OptimizerResult:
        from foresightrlcore.plugins import load_optimizer_plugins

        load_optimizer_plugins()

        started_at = _now_iso()
        prompts_path = self.run_cfg.prompts_path
        if prompts_path:
            prompts_path = str(
                optim_io.resolve_prompts_path(
                    prompts_path, download_dir=self.dataset_download_dir
                )
            )
        prompt_set = make_prompt_set(self.run_cfg.prompt_set, prompts_path)
        judges, weights = self._build_judges()
        records = self._resolve_records()

        checkpoint_path = self.checkpoint_dir / _CHECKPOINT_FILENAME
        existing_rows = optim_io.read_checkpoint_rows(checkpoint_path)
        done_keys = optim_io.completed_resume_keys(existing_rows)
        if done_keys:
            logger.info("resuming: %d completed rows found", len(done_keys))

        checkpoint_s3 = None
        if self.config.output.checkpoint_s3_path:
            base = self.config.output.checkpoint_s3_path.rstrip("/")
            checkpoint_s3 = f"{base}/{_CHECKPOINT_FILENAME}"
        writer = optim_io.CheckpointWriter(checkpoint_path, checkpoint_s3)

        all_rows: List[Dict[str, Any]] = list(existing_rows)
        rows_since_sync = 0

        # Group work by candidate model so local candidates bracket their batch
        # with a single vllm serve lifecycle.
        for model in self.run_cfg.models:
            tasks: List[Tuple[Record, str, str]] = []
            for record in records:
                ds_name = record.meta.get("dataset", self.run_cfg.datasets[0].name)
                for mode_name in self.run_cfg.modes:
                    key = (
                        f"{ds_name}|{record.realm_id}|{record.epoch_date}"
                        f"|{model.name}|{mode_name}"
                    )
                    if key in done_keys:
                        continue
                    tasks.append((record, mode_name, ds_name))

            if not tasks:
                logger.info("model %r: all rows already complete, skipping", model.name)
                continue

            def checkpoint_row(row: RowResult) -> None:
                """Persist one row as soon as it is generated and judged."""
                nonlocal rows_since_sync
                flat = row.to_flat_row()
                with self._lock:
                    all_rows.append(flat)
                    writer.append(flat)
                    rows_since_sync += 1
                    if (
                        self.run_cfg.checkpoint.enabled
                        and rows_since_sync >= self.run_cfg.checkpoint.every_n_rows
                    ):
                        writer.sync()
                        rows_since_sync = 0

            client = self._gen_client_factory(model)
            start = getattr(client, "start", None)
            if callable(start):
                logger.info("starting local serving for model %r", model.name)
                start()

            try:
                self._run_model_batch(
                    model,
                    tasks,
                    client,
                    prompt_set,
                    judges,
                    weights,
                    on_row=checkpoint_row,
                )
            finally:
                client.close()

            # A finished model is worth persisting immediately, whatever the
            # row counter says.
            if self.run_cfg.checkpoint.enabled:
                with self._lock:
                    writer.sync()
                    rows_since_sync = 0

        writer.sync()

        result = OptimizerResult(
            job_name=self.config.job.name,
            prompt_set=self.run_cfg.prompt_set,
            baseline_model=self.run_cfg.baseline_model,
            modes=list(self.run_cfg.modes),
            judges=self.run_cfg.judge_names(),
            acceptance_threshold=self.run_cfg.acceptance_threshold,
            started_at=started_at,
            finished_at=_now_iso(),
            n_rows=len(all_rows),
            summary=build_summary_stats(self.config, all_rows),
            rows=all_rows,
        )
        self._write_outputs(all_rows, result)
        return result

    def _run_model_batch(
        self,
        model: OptimizerModelConfig,
        tasks: List[Tuple[Record, str, str]],
        client: GenerationClient,
        prompt_set: Any,
        judges: Dict[str, Any],
        weights: Dict[str, float],
        on_row: Optional[Callable[[RowResult], None]] = None,
    ) -> List[RowResult]:
        rows: List[RowResult] = []
        max_workers = max(1, self.run_cfg.concurrency)
        logger.info(
            "model %r: evaluating %d rows with concurrency=%d",
            model.name,
            len(tasks),
            max_workers,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    self._run_one,
                    record,
                    model,
                    mode_name,
                    client,
                    prompt_set,
                    judges,
                    weights,
                    ds_name,
                )
                for (record, mode_name, ds_name) in tasks
            ]
            for fut in as_completed(futures):
                row = fut.result()
                rows.append(row)
                # Checkpoint as each row lands, not when the batch ends. A
                # batch is the whole cross-product for one model and can run
                # for hours; checkpointing only on completion means an
                # interruption discards every row it produced. `as_completed`
                # yields on this thread, so the callback needs no lock of its
                # own -- the one it takes inside guards the shared writer.
                if on_row is not None:
                    on_row(row)
        return rows

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def _write_outputs(
        self, rows: List[Dict[str, Any]], result: OptimizerResult
    ) -> None:
        import json

        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import pandas as pd

            df = pd.DataFrame(rows)
            df.to_parquet(self.output_dir / "results.parquet")
        except Exception as exc:  # noqa: BLE001 - parquet optional if pandas missing
            logger.warning("failed to write results.parquet: %s", exc)
            df = None

        summary = render_summary(self.config, rows)
        (self.output_dir / "summary.md").write_text(summary, encoding="utf-8")

        with (self.output_dir / "optimizer_results.json").open("w") as f:
            json.dump(result.model_dump(mode="json"), f, indent=2, default=str)

        logger.info("wrote outputs to %s", self.output_dir)

        out_s3 = self.config.output.s3_path
        if out_s3 and optim_io.is_s3_uri(out_s3):
            try:
                optim_io.s3_sync_dir(self.output_dir, out_s3.rstrip("/"))
            except Exception as exc:  # noqa: BLE001 - upload best-effort
                logger.warning("output S3 sync failed: %s", exc)


# ---------------------------------------------------------------------------
# Summary rendering (ported from reference _render_summary)
# ---------------------------------------------------------------------------


def build_summary_stats(
    config: OptimizerJobConfig, rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Structured report for the JSON manifest (pure-Python, no pandas).

    Returns ``{"overall": {...}, "by_model_mode": [...], "per_judge": [...]}``
    mirroring the summary.md tables. ``overall`` includes pass/fail/error
    counts and the primary-metric statistics; the per-model/mode entries add
    token/cost/latency averages; per-judge entries cover only judged (non-error)
    rows.
    """
    run = config.optimizer
    overall = calculate_summary_statistics(rows)

    def _avg(items: List[Dict[str, Any]], key: str) -> float:
        vals = [float(r.get(key, 0) or 0) for r in items]
        return sum(vals) / len(vals) if vals else 0.0

    groups: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((r.get("model"), r.get("mode")), []).append(r)

    by_model_mode: List[Dict[str, Any]] = []
    for (model, mode), g in groups.items():
        stats = calculate_summary_statistics(g)
        n = len(g)
        gates = 100.0 * sum(1 for r in g if r.get("gates_passed")) / n if n else 0.0
        by_model_mode.append(
            {
                "model": model,
                "mode": mode,
                "n": n,
                "pass_rate": round(stats["success_rate"], 1),
                "error_rate": round(stats["error_rate"], 1),
                "gates_pass_rate": round(gates, 1),
                "primary_median": stats["primary_metric_value"],
                "avg_tokens": round(_avg(g, "total_tokens"), 1),
                "avg_cost_usd": round(_avg(g, "cost_usd"), 6),
                "avg_latency_ms": round(_avg(g, "latency_ms"), 1),
            }
        )

    per_judge: List[Dict[str, Any]] = []
    for spec in run.judges:
        sc, pc = f"{spec.name}_score", f"{spec.name}_passed"
        judged = [r for r in rows if r.get(sc) is not None and sc in r]
        n = len(judged)
        pass_rate = 100.0 * sum(1 for r in judged if r.get(pc)) / n if n else 0.0
        per_judge.append(
            {
                "judge": spec.name,
                "weight": spec.weight,
                "n_judged": n,
                "pass_rate": round(pass_rate, 1),
                "mean_score": round(_avg(judged, sc), 2),
            }
        )

    return {
        "overall": overall,
        "by_model_mode": by_model_mode,
        "per_judge": per_judge,
    }


def render_summary(config: OptimizerJobConfig, rows: List[Dict[str, Any]]) -> str:
    """Render the markdown report (three reference tables + error note)."""
    run = config.optimizer
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: List[str] = []
    lines.append("# GenAI Optimizer - Run Summary")
    lines.append("")
    lines.append(f"- Generated: {ts}")
    lines.append(f"- Rows: {len(rows)}")
    lines.append(f"- Prompt set: `{run.prompt_set}`")
    lines.append(f"- Baseline: `{run.baseline_model}`")
    lines.append(f"- Modes: {', '.join(run.modes)}")
    lines.append(f"- Judges: {', '.join(run.judge_names())}")
    lines.append(f"- Acceptance threshold (primary > x): {run.acceptance_threshold}")
    lines.append("")

    if not rows:
        lines.append("_No rows produced._")
        return "\n".join(lines) + "\n"

    import pandas as pd

    df = pd.DataFrame(rows)

    lines.append("## Quality & cost by model x mode")
    lines.append("")
    lines.append(
        "| model | mode | n | pass% | error% | gates_pass% | primary(median) "
        "| avg_tokens | avg_cost_usd | avg_latency_ms |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (model, mode), g in df.groupby(["model", "mode"]):
        summary = calculate_summary_statistics(g.to_dict("records"))
        gates = 100.0 * g["gates_passed"].mean() if "gates_passed" in g else 0.0
        avg_tokens = g["total_tokens"].mean()
        avg_cost = g["cost_usd"].mean()
        avg_lat = g["latency_ms"].mean()
        lines.append(
            f"| {model} | {mode} | {len(g)} | {summary['success_rate']:.1f} | "
            f"{summary['error_rate']:.1f} | "
            f"{gates:.1f} | {summary['primary_metric_value']:.2f} | "
            f"{avg_tokens:.0f} | {avg_cost:.6f} | {avg_lat:.1f} |"
        )
    lines.append("")

    lines.append("## Per-judge pass rate")
    lines.append("")
    lines.append("| judge | pass% | mean_score |")
    lines.append("|---|---:|---:|")
    for spec in run.judges:
        pc = f"{spec.name}_passed"
        sc = f"{spec.name}_score"
        if pc in df.columns:
            lines.append(
                f"| {spec.name} | {100.0 * df[pc].mean():.1f} | "
                f"{df[sc].mean():.2f} |"
            )
    lines.append("")

    lines.append("## Cost projection (per 10K states/month)")
    lines.append("")
    lines.append("| model | mode | avg_cost_usd/state | projected_monthly_usd |")
    lines.append("|---|---|---:|---:|")
    for (model, mode), g in df.groupby(["model", "mode"]):
        per = g["cost_usd"].mean()
        lines.append(f"| {model} | {mode} | {per:.6f} | {per * 10000:.2f} |")
    lines.append("")

    if (df.get("generation_error", "") != "").any():
        n_err = int((df["generation_error"] != "").sum())
        lines.append(f"> Note: {n_err} row(s) had generation errors.")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Public run + CLI
# ---------------------------------------------------------------------------


def run_optimizer(
    config: OptimizerJobConfig,
    *,
    output_dir: str | Path,
    checkpoint_dir: str | Path | None = None,
) -> OptimizerResult:
    """Load registries and run the optimizer, returning the result document."""
    from foresightrlcore.plugins import load_optimizer_plugins

    load_optimizer_plugins()

    runner = OptimizerRunner(
        config,
        output_dir=Path(output_dir),
        checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
    )
    return runner.run()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="foresightrlcore.optimizer.runner",
        description="Container-side orchestrator for optimizer runs.",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("OPTIMIZER_CONFIG", DEFAULT_CONFIG_PATH),
        help="Path to the optimizer YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("OPTIMIZER_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        help="Output directory for results + summary.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=os.environ.get("OPTIMIZER_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR),
        help="Directory for the incremental checkpoint JSONL.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = OptimizerJobConfig.from_yaml(args.config)
    result = run_optimizer(
        config,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
    logger.info("optimizer finished: %d rows", result.n_rows)
    return 0


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    raise SystemExit(main())
