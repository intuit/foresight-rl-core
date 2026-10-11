"""Evaluate runner: benchmarks x models x trials -> results + report.

Follows :mod:`foresightrlcore.optimizer.runner`, with three differences that
come straight from what benchmarks need:

* **One endpoint per model.** Work is grouped by candidate so the model's
  endpoint -- a ``vllm serve`` process, or the in-process proxy -- comes up
  once and covers its whole batch. A local candidate never overlaps another.

* **Two tiers in one loop.** A record-tier benchmark is fanned out over a
  thread pool here; a harness-tier benchmark is handed a workspace and a live
  endpoint and left alone. Both land in the same :class:`BenchmarkResult`.

* **Trials, not repeats.** ``trials: K`` produces K independent results that
  aggregate into a mean with a t-distribution interval. Where a benchmark's
  sub-scores disagree across trials, ``confidence.retrial_flaky`` spends extra
  trials on *those benchmarks only* and tells the benchmark which sub-scores
  were unstable (``$EVALUATE_RETRIAL_SUBSCORES``), so one that can narrow its
  own work does.

Resume is at two granularities: a completed ``(benchmark, model, trial)`` is
skipped outright, and within an unfinished record-tier trial, records already
scored are reused. Outputs land in the output dir: ``results.parquet``,
``benchmark_scores.csv``, ``summary.md``, ``evaluate_results.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, cast

from foresightrlcore.common import io as common_io
from foresightrlcore.common.thinking import strip_thinking
from foresightrlcore.config.evaluate_schema import (
    BenchmarkSpec,
    EvaluateJobConfig,
    EvaluateModelConfig,
    channel_mount_path,
    local_adapter_channels,
    local_model_channels,
)
from foresightrlcore.evaluate.catalog import load_catalog, resolve_benchmark
from foresightrlcore.evaluate.interfaces import (
    PARSE_FAILED_SUFFIX,
    DatasetLoader,
    EvalRecord,
    HarnessContext,
    HarnessOutcome,
    Metric,
    Prediction,
    RecordBenchmark,
    is_harness_benchmark,
)
from foresightrlcore.evaluate.metrics.aggregate import (
    mean_sub_scores,
    mean_with_ci,
    score_population,
    unstable_sub_scores,
)
from foresightrlcore.evaluate.registry import (
    BENCHMARK_TYPES,
    DATASET_LOADERS,
    METRICS,
    REPORTERS,
)
from foresightrlcore.evaluate.spec import (
    INVALID_SCORE,
    AggregateScore,
    BenchmarkResult,
    EvaluateResult,
    SampleResult,
)
from foresightrlcore.evaluate.workspace import WorkspaceManager
from foresightrlcore.inference.client import ChatClient
from foresightrlcore.inference.interfaces import InferenceEndpoint

logger = logging.getLogger("foresightrlcore.evaluate.runner")

# Container paths (SageMaker training-job convention).
DEFAULT_OUTPUT_DIR = "/opt/ml/output/data"
DEFAULT_CHECKPOINT_DIR = "/opt/ml/checkpoints"
DEFAULT_CONFIG_PATH = "/opt/ml/input/data/config/args.yaml"
DEFAULT_DATASET_DOWNLOAD_DIR = "/tmp/evaluate_data"

RESULT_CHECKPOINT = "evaluate_checkpoint.jsonl"
SAMPLE_CHECKPOINT = "evaluate_samples.jsonl"

# Checkpoint-only column: what the stored row measured. See
# :func:`measurement_fingerprint`.
FINGERPRINT_KEY = "fingerprint"

DEFAULT_BENCHMARK_TYPE = "mcq"

#: Signature of the endpoint context manager, so tests can inject a fake.
EndpointFactory = Callable[[EvaluateModelConfig], Any]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _secondary_metric_sub_scores(
    samples: Sequence[SampleResult],
) -> Dict[str, float]:
    """Mean score and valid-answer rate for each non-primary metric.

    A benchmark reports one number, but it can run several metrics, and the
    ones that are not reported still know something worth publishing. Two
    things come out of them here: the alternative score, so a stricter reading
    sits next to the headline without changing it, and the share of answers
    that metric could read at all.

    The valid-answer rate is the reason this exists. For the leaderboard suite
    the reported metric is a containment rule that scores an unreadable answer
    0 instead of flagging it, so ``parse_failure_rate`` above is structurally
    0 no matter how garbled the responses are -- it once read 0.0 on a run
    where 86% of answers were unparseable. A metric that *can* tell the
    difference runs alongside it, and this is where its verdict surfaces.

    Numeric detail that is not a metric verdict (``gold_index`` and friends) is
    ignored: a key is only treated as a metric when its parse verdict was
    recorded next to it.
    """
    values: Dict[str, List[float]] = {}
    failures: Dict[str, List[bool]] = {}
    for sample in samples:
        for key, raw in (sample.detail or {}).items():
            if key.endswith(PARSE_FAILED_SUFFIX):
                name = key[: -len(PARSE_FAILED_SUFFIX)]
                failures.setdefault(name, []).append(bool(raw))
            elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
                values.setdefault(key, []).append(float(raw))

    out: Dict[str, float] = {}
    for name, flags in failures.items():
        out[f"{name}_valid_rate"] = 1.0 - (sum(flags) / len(flags)) if flags else 0.0
        vals = values.get(name)
        if vals:
            out[f"{name}_score"] = sum(vals) / len(vals)
    return out


def _disambiguate_record_ids(
    benchmark: str, records: List[EvalRecord]
) -> List[EvalRecord]:
    """Make record ids unique, because resume is keyed on them.

    Record-level resume treats the id as identifying the record, so a dataset
    that reuses an id makes two different questions indistinguishable: the
    first one's result is reused for both, and the second is never measured,
    yet the sample count still reads full. The real `logiqa2` catalog
    benchmark does this -- two distinct questions share id ``10591``, and a
    resumed run scored one of them twice and the other never, with nothing in
    the output to say so.

    Suffixing repeats keeps the ids stable for the common case (no duplicates,
    nothing rewritten) while making the pathological case correct and visible.
    """
    seen: Dict[str, int] = {}
    out: List[EvalRecord] = []
    collisions = 0
    for record in records:
        count = seen.get(record.id, 0)
        seen[record.id] = count + 1
        if count:
            collisions += 1
            record = replace(record, id=f"{record.id}#{count + 1}")
        out.append(record)
    if collisions:
        logger.warning(
            "benchmark %r: %d duplicate record id(s) in the dataset; the "
            "repeats were suffixed (e.g. 'x' -> 'x#2') so that each record "
            "resumes as itself. Checkpoints written before this run keyed "
            "those records on the bare id and will not be reused for them.",
            benchmark,
            collisions,
        )
    return out


def cost_usd(
    prompt_tokens: int, completion_tokens: int, price_in: float, price_out: float
) -> float:
    """USD cost from per-1K-token prices."""
    return (prompt_tokens / 1000.0) * price_in + (
        completion_tokens / 1000.0
    ) * price_out


def _as_prediction(
    chat: Any, thinking_tags: Optional[List[str]], *, declared: bool = False
) -> Prediction:
    """Wrap a chat result, removing the scratchpad for a reasoning model.

    Stripping happens here rather than inside a metric so that every metric
    gets it: an answer extractor, a string comparison and a judge prompt are
    all equally misled by narration, and only the runner knows which candidate
    produced the response.

    Args:
        chat: The raw chat result.
        thinking_tags: Tags to strip, or ``None`` to leave the text alone.
        declared: Whether the candidate configured ``thinking:`` at all. This
            travels on the prediction so a metric's own safety net knows to
            stand down -- including when the config's answer was "do not
            strip", which is the case a safety net would otherwise override.
    """
    text = chat.text
    stripped = strip_thinking(text, thinking_tags) if thinking_tags else text
    return Prediction(
        text=stripped,
        raw_text=text if stripped != text else None,
        prompt_tokens=chat.prompt_tokens,
        completion_tokens=chat.completion_tokens,
        latency_ms=chat.latency_ms,
        thinking_handled=declared,
        finish_reason=chat.finish_reason,
    )


class EvaluateRunner:
    """Run one :class:`EvaluateJobConfig` end to end.

    Args:
        config: The validated job config.
        output_dir: Where reports are written.
        checkpoint_dir: Where the resume checkpoints live. Defaults to
            ``output_dir``.
        dataset_download_dir: Scratch space for S3 datasets.
        workspace_root: Root for staged sources and trial workspaces.
        endpoint_factory: Returns a context manager yielding an
            :class:`InferenceEndpoint` for a model. Injected in tests.
    """

    def __init__(
        self,
        config: EvaluateJobConfig,
        *,
        output_dir: Path,
        checkpoint_dir: Optional[Path] = None,
        dataset_download_dir: Optional[Path] = None,
        workspace_root: Optional[Path] = None,
        endpoint_factory: Optional[EndpointFactory] = None,
    ) -> None:
        self.config = config
        self.run_cfg = config.evaluate
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir else self.output_dir
        )
        self.dataset_download_dir = Path(
            dataset_download_dir or DEFAULT_DATASET_DOWNLOAD_DIR
        )
        self.workspace_root = Path(workspace_root) if workspace_root else None
        self._endpoint_factory = endpoint_factory or self._default_endpoint_factory
        self._lock = threading.Lock()
        self._managers: Dict[str, Any] = {}
        self._records_cache: Dict[str, List[EvalRecord]] = {}

    # ------------------------------------------------------------------
    # Defaults that talk to the outside world (overridable in tests)
    # ------------------------------------------------------------------

    def _default_endpoint_factory(self, model: EvaluateModelConfig) -> Any:
        from foresightrlcore.inference import endpoint_for

        return endpoint_for(
            model,
            max_retries=self.run_cfg.retry.max_retries,
            base_delay=self.run_cfg.retry.base_delay,
        )

    def _manager_for(self, client_name: str) -> Any:
        from foresightrlcore.clients.llm import resolve_llm_manager

        if self.run_cfg.judge_clients is None:
            raise ValueError(
                "a judge metric is configured but evaluate.judge_clients is empty"
            )
        if client_name not in self._managers:
            self._managers[client_name] = resolve_llm_manager(
                self.run_cfg.judge_clients.get(client_name)
            )
        return self._managers[client_name]

    # ------------------------------------------------------------------
    # Building benchmarks
    # ------------------------------------------------------------------

    def _build_metrics(self, spec: BenchmarkSpec) -> List[Metric]:
        metrics: List[Metric] = []
        for name in spec.metrics:
            factory = METRICS.get(name)
            if name == "llm_judge":
                client = str(spec.type_params().get("judge_client", "default"))
                metrics.append(cast(Metric, factory(self._manager_for(client))))
            else:
                metrics.append(cast(Metric, factory()))
        return metrics

    def _build_benchmark(self, spec: BenchmarkSpec) -> Any:
        factory = BENCHMARK_TYPES.get(spec.type or DEFAULT_BENCHMARK_TYPE)
        return factory(spec=spec, metrics=self._build_metrics(spec))

    def _load_records(self, spec: BenchmarkSpec) -> List[EvalRecord]:
        """Load (and cache) a benchmark's records; empty when it has no dataset."""
        if spec.dataset is None:
            return []
        if spec.name in self._records_cache:
            return self._records_cache[spec.name]

        loader_spec = spec.dataset.as_loader_spec()
        path = loader_spec.get("path")
        if path:
            loader_spec["path"] = str(
                common_io.resolve_dataset_path(
                    path, download_dir=self.dataset_download_dir
                )
            )

        sample = spec.sample.model_dump() if spec.sample else None
        loader = cast(DatasetLoader, DATASET_LOADERS.get(spec.dataset.loader)())
        records = list(loader.records(loader_spec, sample))
        records = _disambiguate_record_ids(spec.name, records)
        logger.info("benchmark %r: %d records", spec.name, len(records))
        self._records_cache[spec.name] = records
        return records

    # ------------------------------------------------------------------
    # Record tier
    # ------------------------------------------------------------------

    def _responses_to_store(self, prediction: Prediction) -> Tuple[str, str]:
        """Return the (scored, pre-stripping) texts to keep on the row.

        The second is empty unless stripping actually changed the text, so a
        populated ``raw_response`` is itself the evidence that stripping ran.
        """
        if not self.run_cfg.store_responses:
            return "", ""

        limit = self.run_cfg.store_responses_max_chars

        def clip(text: str) -> str:
            if limit and len(text) > limit:
                return text[:limit] + f"... [{len(text) - limit} more chars]"
            return text

        raw = prediction.raw_text
        return clip(prediction.text), clip(raw) if raw is not None else ""

    def _score_record(
        self,
        benchmark: RecordBenchmark,
        spec: BenchmarkSpec,
        model: EvaluateModelConfig,
        trial: int,
        record: EvalRecord,
        client: ChatClient,
    ) -> SampleResult:
        """Evaluate one record. Never raises: a failure becomes an error row."""
        from foresightrlcore.common.retry import call_with_retry

        request = benchmark.build_prompt(record)
        try:
            chat = call_with_retry(
                lambda: client.chat(
                    request.messages,
                    max_tokens=request.max_tokens or model.max_tokens,
                    temperature=(
                        request.temperature
                        if request.temperature is not None
                        else model.effective_temperature
                    ),
                    top_p=(request.top_p if request.top_p is not None else model.top_p),
                    top_k=(request.top_k if request.top_k is not None else model.top_k),
                    repetition_penalty=(
                        request.repetition_penalty
                        if request.repetition_penalty is not None
                        else model.repetition_penalty
                    ),
                    stop=request.stop,
                    extra={**model.extra_body, **request.extra},
                ),
                max_retries=self.run_cfg.retry.max_retries,
                base_delay=self.run_cfg.retry.base_delay,
                description=f"{spec.name}/{record.id}",
            )
            prediction = _as_prediction(
                chat, model.thinking_tags, declared=model.thinking is not None
            )
        except Exception as exc:  # noqa: BLE001 - one record must not end the run
            prediction = Prediction(error=f"{type(exc).__name__}: {exc}")

        verdict = benchmark.score(record, prediction)
        kept, kept_raw = self._responses_to_store(prediction)
        return SampleResult(
            benchmark=spec.name,
            model=model.name,
            trial=trial,
            record_id=record.id,
            score=verdict.value,
            correct=verdict.correct,
            parse_failed=verdict.parse_failed,
            error=prediction.error,
            response=kept,
            raw_response=kept_raw,
            finish_reason=prediction.finish_reason,
            prompt_tokens=prediction.prompt_tokens,
            completion_tokens=prediction.completion_tokens,
            latency_ms=round(prediction.latency_ms, 1),
            cost_usd=round(
                cost_usd(
                    prediction.prompt_tokens,
                    prediction.completion_tokens,
                    model.price_in,
                    model.price_out,
                ),
                6,
            ),
            detail=verdict.detail,
        )

    def _run_record_trial(
        self,
        benchmark: RecordBenchmark,
        spec: BenchmarkSpec,
        model: EvaluateModelConfig,
        trial: int,
        endpoint: InferenceEndpoint,
        done_records: Dict[str, SampleResult],
        on_sample: Optional[Callable[[SampleResult], None]] = None,
    ) -> Tuple[BenchmarkResult, List[SampleResult]]:
        started_at = _now_iso()
        records = self._load_records(spec)
        if not records:
            return (
                self._empty_result(
                    spec, model, trial, started_at, "benchmark has no records"
                ),
                [],
            )

        pending = [r for r in records if r.id not in done_records]
        if len(pending) < len(records):
            logger.info(
                "benchmark %r trial %d: reusing %d checkpointed record(s)",
                spec.name,
                trial,
                len(records) - len(pending),
            )

        client = ChatClient(endpoint, timeout=spec.timeout_seconds)
        fresh: List[SampleResult] = []
        with ThreadPoolExecutor(max_workers=max(1, self.run_cfg.concurrency)) as pool:
            futures = [
                pool.submit(
                    self._score_record, benchmark, spec, model, trial, record, client
                )
                for record in pending
            ]
            for future in as_completed(futures):
                sample = future.result()
                fresh.append(sample)
                # Checkpoint as each record lands, not when the trial ends.
                # A trial is the unit of *reporting*, not of work: one
                # benchmark over a large dataset is a single trial that can
                # run for hours, and checkpointing it only on completion
                # means an interruption discards every record it generated.
                # `as_completed` yields on this thread, so the callback needs
                # no lock.
                if on_sample is not None:
                    on_sample(sample)

        samples = [done_records[r.id] for r in records if r.id in done_records] + fresh
        return self._aggregate_samples(spec, model, trial, samples, started_at), fresh

    def _aggregate_samples(
        self,
        spec: BenchmarkSpec,
        model: EvaluateModelConfig,
        trial: int,
        samples: Sequence[SampleResult],
        started_at: str,
    ) -> BenchmarkResult:
        """Roll per-record scores into the trial's result.

        Records that errored are excluded from the mean and counted; past the
        benchmark's ``failure_tolerance`` the score is reported invalid rather
        than computed on whatever happened to succeed.
        """
        scored = [s for s in samples if not s.error]
        n_errors = len(samples) - len(scored)
        score = score_population(
            [s.score for s in scored],
            n_errors,
            spec.failure_tolerance,
            INVALID_SCORE,
        )
        parse_failures = sum(1 for s in scored if s.parse_failed)
        truncated = sum(1 for s in scored if s.finish_reason == "length")

        sub_scores: Dict[str, float] = {
            "accuracy": (
                sum(1.0 for s in scored if s.correct) / len(scored) if scored else 0.0
            ),
            "parse_failure_rate": (parse_failures / len(scored) if scored else 0.0),
            # A non-zero rate here means the score understates the model:
            # generation hit max_tokens, so the answer was cut off rather than
            # wrong. Raise the budget and rerun before reading further.
            "truncation_rate": (truncated / len(scored) if scored else 0.0),
        }
        sub_scores.update(_secondary_metric_sub_scores(scored))

        return BenchmarkResult(
            benchmark=spec.name,
            model=model.name,
            trial=trial,
            score=score if score is not None else INVALID_SCORE,
            n_samples=len(scored),
            n_errors=n_errors,
            sub_scores=sub_scores,
            prompt_tokens=sum(s.prompt_tokens for s in samples),
            completion_tokens=sum(s.completion_tokens for s in samples),
            cost_usd=round(sum(s.cost_usd for s in samples), 6),
            started_at=started_at,
            finished_at=_now_iso(),
        )

    # ------------------------------------------------------------------
    # Harness tier
    # ------------------------------------------------------------------

    def _run_harness_trial(
        self,
        benchmark: Any,
        spec: BenchmarkSpec,
        model: EvaluateModelConfig,
        trial: int,
        endpoint: InferenceEndpoint,
        workspaces: WorkspaceManager,
        retrial_sub_scores: Sequence[str] = (),
    ) -> BenchmarkResult:
        started_at = _now_iso()

        source_dir: Optional[Path] = None
        python: Optional[Path] = None
        try:
            if spec.source is not None:
                source_dir = workspaces.stage_source(spec.name, spec.source)
            if spec.isolate == "venv":
                python = workspaces.ensure_venv(spec.name, spec.requirements)
            workspace = workspaces.trial_workspace(spec.name, model.name, trial)
        except Exception as exc:  # noqa: BLE001 - staging failure fails the benchmark
            logger.warning("benchmark %r: staging failed: %s", spec.name, exc)
            return self._empty_result(
                spec, model, trial, started_at, f"staging failed: {exc}"
            )

        output_file = workspace / "evaluate_result.json"
        env = {
            **endpoint.env(),
            "EVALUATE_BENCHMARK": spec.name,
            "EVALUATE_WORKSPACE": str(workspace),
            "EVALUATE_OUTPUT_FILE": str(output_file),
            "EVALUATE_TRIAL": str(trial),
            **({"EVALUATE_SOURCE_DIR": str(source_dir)} if source_dir else {}),
            **({"EVALUATE_PYTHON": str(python)} if python else {}),
            **(
                {"EVALUATE_RETRIAL_SUBSCORES": ",".join(retrial_sub_scores)}
                if retrial_sub_scores
                else {}
            ),
            **spec.env,
        }

        params = dict(spec.type_params())
        if retrial_sub_scores:
            params["retrial_sub_scores"] = list(retrial_sub_scores)

        ctx = HarnessContext(
            benchmark=spec.name,
            endpoint=endpoint,
            workspace=workspace,
            output_file=output_file,
            records=self._load_records(spec),
            source_dir=source_dir,
            params=params,
            env=env,
            trial=trial,
            timeout_seconds=spec.timeout_seconds,
            agent_runtime=model.agent_runtime,
            logger=logging.getLogger(f"foresightrlcore.evaluate.{spec.name}"),
        )

        try:
            outcome = benchmark.run(ctx)
        except Exception as exc:  # noqa: BLE001 - one benchmark must not end the run
            logger.warning("benchmark %r raised: %s", spec.name, exc)
            outcome = HarnessOutcome(error=f"{type(exc).__name__}: {exc}")

        return self._harness_result(spec, model, trial, outcome, started_at)

    def _harness_result(
        self,
        spec: BenchmarkSpec,
        model: EvaluateModelConfig,
        trial: int,
        outcome: HarnessOutcome,
        started_at: str,
    ) -> BenchmarkResult:
        total = outcome.n_samples + outcome.n_errors
        error_rate = (outcome.n_errors / total) if total else 0.0

        score = outcome.score
        if outcome.error:
            score = INVALID_SCORE
        elif error_rate > spec.failure_tolerance:
            logger.warning(
                "benchmark %r: %d/%d samples errored (>%.0f%% tolerance); "
                "reporting an invalid score",
                spec.name,
                outcome.n_errors,
                total,
                spec.failure_tolerance * 100,
            )
            score = INVALID_SCORE

        return BenchmarkResult(
            benchmark=spec.name,
            model=model.name,
            trial=trial,
            score=score,
            n_samples=outcome.n_samples,
            n_errors=outcome.n_errors,
            sub_scores=outcome.sub_scores,
            detail=outcome.detail,
            error=outcome.error,
            prompt_tokens=outcome.prompt_tokens,
            completion_tokens=outcome.completion_tokens,
            cost_usd=round(
                cost_usd(
                    outcome.prompt_tokens,
                    outcome.completion_tokens,
                    model.price_in,
                    model.price_out,
                ),
                6,
            ),
            started_at=started_at,
            finished_at=_now_iso(),
        )

    def _empty_result(
        self,
        spec: BenchmarkSpec,
        model: EvaluateModelConfig,
        trial: int,
        started_at: str,
        error: str,
    ) -> BenchmarkResult:
        return BenchmarkResult(
            benchmark=spec.name,
            model=model.name,
            trial=trial,
            score=INVALID_SCORE,
            error=error,
            started_at=started_at,
            finished_at=_now_iso(),
        )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> EvaluateResult:
        from foresightrlcore.plugins import load_evaluate_plugins

        load_evaluate_plugins()
        started_at = _now_iso()

        catalog = load_catalog(self.run_cfg.catalog_paths)
        specs = [resolve_benchmark(entry, catalog) for entry in self.run_cfg.benchmarks]
        by_name = {spec.name: spec for spec in specs}

        fingerprints = {
            (spec.name, model.name): measurement_fingerprint(spec, model)
            for spec in specs
            for model in self.run_cfg.models
        }

        def _still_current(row: Dict[str, Any]) -> bool:
            """True when a checkpointed row measures what this run measures."""
            expected = fingerprints.get(
                (str(row.get("benchmark", "")), str(row.get("model", "")))
            )
            return expected is not None and row.get(FINGERPRINT_KEY) == expected

        prior_results, dropped_results = _partition(
            common_io.read_checkpoint_rows(self.checkpoint_dir / RESULT_CHECKPOINT),
            _still_current,
        )
        prior_samples, _ = _partition(
            common_io.read_checkpoint_rows(self.checkpoint_dir / SAMPLE_CHECKPOINT),
            _still_current,
        )
        if dropped_results:
            logger.info(
                "ignoring %d checkpointed trial(s): the benchmark or model they "
                "measured has changed, or is no longer in this run",
                dropped_results,
            )
        # A trial that errored is not finished, it is damaged. Treating it as
        # done is how a transient outage becomes a permanent hole: the runner
        # exhausts its retries on a record, checkpoints the failure, and every
        # later resume skips that record forever because a row exists for it.
        # The GenOS sweeps hit exactly this -- the documented remedy was to
        # start from a fresh checkpoint prefix, which throws away every good
        # record to re-reach a handful of bad ones.
        #
        # So a trial carrying errors is replanned, and the errored records
        # inside it are withheld from `done_records` (see
        # `_checkpointed_samples`) while its successful ones are reused. The
        # re-run therefore costs only the records that actually failed.
        #
        # A record that fails for a real reason rather than a transient one
        # does get re-attempted on each resume. That is the right trade: each
        # attempt is bounded by `retry.max_retries`, and a permanently bad
        # record is one `failure_tolerance` already knows how to report.
        # The checkpoint is an append-only log, so replanning a trial leaves
        # both its old row and its replacement behind. The last row written
        # for a resume_key is the current one; anything earlier is history.
        # Collapsing them matters in both directions: a trial that once
        # errored would otherwise stay damaged forever, replanned on every
        # future resume long after it was repaired, and a trial with two rows
        # would be reported twice.
        latest: Dict[Any, Dict[str, Any]] = {}
        for row in prior_results:
            latest[row.get("resume_key")] = row
        prior_results = list(latest.values())

        damaged = {
            row.get("resume_key")
            for row in prior_results
            if int(row.get("n_errors") or 0) > 0
        }
        done_trials = {row.get("resume_key") for row in prior_results} - damaged
        if done_trials:
            logger.info("resuming: %d completed trial(s) found", len(done_trials))
        if damaged:
            logger.info(
                "re-running %d trial(s) that finished with errors; their "
                "successful records are reused and only the failures repeat",
                len(damaged),
            )

        result_writer = self._writer(RESULT_CHECKPOINT)
        sample_writer = self._writer(SAMPLE_CHECKPOINT)

        # Damaged trials are replanned, so their prior rows must not also be
        # carried into the report -- they would be counted twice, once stale
        # and once fresh.
        results: List[BenchmarkResult] = [
            BenchmarkResult.model_validate(_without_resume_key(row))
            for row in prior_results
            if row.get("resume_key") not in damaged
        ]
        # Same append-only reasoning as the trial rows above: a record that was
        # re-scored has both rows on disk, and the report should carry the last
        # one rather than both. This also absorbs history a checkpoint may
        # already contain -- rows written before record ids were disambiguated,
        # for instance -- so what is reported depends on the current run, not
        # on how the prefix came to look the way it does.
        latest_samples: Dict[Any, Dict[str, Any]] = {}
        for row in prior_samples:
            if row.get("error"):
                continue
            key = (
                row.get("benchmark"),
                row.get("model"),
                row.get("trial"),
                row.get("record_id"),
            )
            latest_samples[key] = {k: v for k, v in row.items() if k != FINGERPRINT_KEY}
        samples: List[Dict[str, Any]] = list(latest_samples.values())
        rows_since_sync = 0

        def sample_checkpointer(
            benchmark: str, model_name: str
        ) -> Callable[[SampleResult], None]:
            """Build the per-record checkpoint callback for one measurement.

            The fingerprint is checkpoint bookkeeping, so it stays out of the
            reported rows and the schema of samples.parquet does not move.
            """
            fingerprint = fingerprints[(benchmark, model_name)]

            def _append(sample: SampleResult) -> None:
                nonlocal rows_since_sync
                sample_writer.append(
                    {**sample.to_flat_row(), FINGERPRINT_KEY: fingerprint}
                )
                rows_since_sync += 1
                if (
                    self.run_cfg.checkpoint.enabled
                    and rows_since_sync >= self.run_cfg.checkpoint.every_n_rows
                ):
                    # Only the samples file gains rows here; result rows are
                    # written per trial and flushed by `record_result`.
                    sample_writer.sync()
                    rows_since_sync = 0

            return _append

        def record_result(result: BenchmarkResult) -> None:
            """Checkpoint a finished trial and flush.

            A completed trial flushes regardless of the record counter: its
            result row is what lets the next run skip the trial wholesale
            rather than re-reading it record by record.
            """
            nonlocal rows_since_sync
            row = result.to_flat_row()
            row["resume_key"] = result.resume_key()
            row[FINGERPRINT_KEY] = fingerprints[(result.benchmark, result.model)]
            result_writer.append(row)
            if self.run_cfg.checkpoint.enabled:
                result_writer.sync()
                sample_writer.sync()
                rows_since_sync = 0

        with WorkspaceManager(self.workspace_root) as workspaces:
            for model in self.run_cfg.models:
                planned = list(self._plan_for_model(model, specs, done_trials))
                if not planned:
                    logger.info("model %r: nothing left to run", model.name)
                    continue

                logger.info(
                    "model %r: %d trial(s) across %d benchmark(s)",
                    model.name,
                    len(planned),
                    len({spec.name for spec, _ in planned}),
                )
                with self._endpoint_factory(model) as endpoint:
                    for spec, trial in planned:
                        result, fresh = self._run_trial(
                            spec,
                            model,
                            trial,
                            endpoint,
                            workspaces,
                            self._checkpointed_samples(
                                prior_samples, spec.name, model.name, trial
                            ),
                            on_sample=sample_checkpointer(spec.name, model.name),
                        )
                        results.append(result)
                        record_result(result)
                        # Samples reached the checkpoint as they were scored;
                        # this list is the in-memory one the report is built
                        # from, and is seeded with the resumed rows already.
                        samples.extend(sample.to_flat_row() for sample in fresh)

                    extra = self._retrials(
                        model,
                        by_name,
                        results,
                        endpoint,
                        workspaces,
                        on_sample=sample_checkpointer,
                    )
                    # Retrials are trials, and cost the same. Leaving them out
                    # of the checkpoint meant every resumed run re-ran them
                    # from nothing, however much had already been spent.
                    for result in extra:
                        record_result(result)
                    results.extend(extra)

        result_writer.sync()
        sample_writer.sync()

        aggregates = self.aggregate(results)
        document = EvaluateResult(
            job_name=self.config.job.name,
            benchmarks=[spec.name for spec in specs],
            models=self.run_cfg.model_names(),
            trials=self.run_cfg.trials,
            started_at=started_at,
            finished_at=_now_iso(),
            n_results=len(results),
            summary=build_summary_stats(aggregates),
            aggregates=[a.model_dump() for a in aggregates],
            results=[r.to_flat_row() for r in results],
            samples=samples,
            model_info={m.name: m.report_info() for m in self.run_cfg.models},
        )
        self._report(document)
        return document

    def _writer(self, filename: str) -> common_io.CheckpointWriter:
        s3_base = self.config.output.checkpoint_s3_path
        s3_uri = f"{s3_base.rstrip('/')}/{filename}" if s3_base else None
        return common_io.CheckpointWriter(self.checkpoint_dir / filename, s3_uri)

    def _plan_for_model(
        self,
        model: EvaluateModelConfig,
        specs: Sequence[BenchmarkSpec],
        done_trials: set,
    ) -> Iterator[Tuple[BenchmarkSpec, int]]:
        """Yield the ``(spec, trial)`` pairs still outstanding for ``model``."""
        for spec in specs:
            if model.name not in {
                m.name for m in self.run_cfg.resolve_models_for(spec)
            }:
                continue
            for trial in range(self.run_cfg.trials_for(spec)):
                if f"{spec.name}|{model.name}|{trial}" in done_trials:
                    continue
                yield spec, trial

    @staticmethod
    def _checkpointed_samples(
        rows: Sequence[Dict[str, Any]], benchmark: str, model: str, trial: int
    ) -> Dict[str, SampleResult]:
        """Return reusable checkpointed records for one trial, keyed by id.

        An errored record is not reusable. It is a record we failed to
        measure, and reusing it means never measuring it -- the failure
        outlives the outage that caused it and silently reappears in every
        resumed run's results. Withholding it here puts it back in `pending`,
        so the re-run pays for the records that failed and reuses the rest.
        """
        found: Dict[str, SampleResult] = {}
        for row in rows:
            if row.get("error"):
                continue
            if (
                row.get("benchmark") == benchmark
                and row.get("model") == model
                and row.get("trial") == trial
            ):
                try:
                    sample = SampleResult.model_validate(_strip_detail_columns(row))
                except Exception:  # noqa: BLE001 - a stale row is not fatal
                    continue
                found[sample.record_id] = sample
        return found

    def _run_trial(
        self,
        spec: BenchmarkSpec,
        model: EvaluateModelConfig,
        trial: int,
        endpoint: InferenceEndpoint,
        workspaces: WorkspaceManager,
        done_records: Dict[str, SampleResult],
        retrial_sub_scores: Sequence[str] = (),
        on_sample: Optional[Callable[[SampleResult], None]] = None,
    ) -> Tuple[BenchmarkResult, List[SampleResult]]:
        started_at = _now_iso()
        try:
            benchmark = self._build_benchmark(spec)
        except Exception as exc:  # noqa: BLE001 - a bad spec fails its benchmark only
            logger.warning("benchmark %r could not be built: %s", spec.name, exc)
            return self._empty_result(spec, model, trial, started_at, str(exc)), []

        if is_harness_benchmark(benchmark):
            return (
                self._run_harness_trial(
                    benchmark,
                    spec,
                    model,
                    trial,
                    endpoint,
                    workspaces,
                    retrial_sub_scores,
                ),
                [],
            )
        return self._run_record_trial(
            cast(RecordBenchmark, benchmark),
            spec,
            model,
            trial,
            endpoint,
            done_records,
            on_sample,
        )

    def _retrials(
        self,
        model: EvaluateModelConfig,
        specs: Dict[str, BenchmarkSpec],
        results: Sequence[BenchmarkResult],
        endpoint: InferenceEndpoint,
        workspaces: WorkspaceManager,
        on_sample: Optional[
            Callable[[str, str], Callable[[SampleResult], None]]
        ] = None,
    ) -> List[BenchmarkResult]:
        """Spend extra trials only where trials disagreed.

        A benchmark whose sub-scores were identical across trials is already
        measured; repeating it buys nothing. The unstable sub-score names are
        passed down, so a benchmark able to re-run just those scenarios can.
        """
        confidence = self.run_cfg.confidence
        if not confidence.retrial_flaky or confidence.max_retrials < 1:
            return []

        extra: List[BenchmarkResult] = []
        for name, spec in specs.items():
            trials = [
                r
                for r in results
                if r.benchmark == name and r.model == model.name and r.is_valid
            ]
            if len(trials) < 2:
                continue
            unstable = unstable_sub_scores([r.sub_scores for r in trials])
            if not unstable:
                continue

            logger.info(
                "benchmark %r on %r: unstable sub-scores %s; running %d extra trial(s)",
                name,
                model.name,
                unstable,
                confidence.max_retrials,
            )
            base = max(r.trial for r in trials) + 1
            for offset in range(confidence.max_retrials):
                result, _ = self._run_trial(
                    spec,
                    model,
                    base + offset,
                    endpoint,
                    workspaces,
                    {},
                    retrial_sub_scores=unstable,
                    on_sample=(on_sample(name, model.name) if on_sample else None),
                )
                extra.append(result)
        return extra

    # ------------------------------------------------------------------
    # Aggregation + reporting
    # ------------------------------------------------------------------

    def aggregate(self, results: Sequence[BenchmarkResult]) -> List[AggregateScore]:
        """Collapse per-trial results into one row per ``(benchmark, model)``."""
        groups: Dict[Tuple[str, str], List[BenchmarkResult]] = {}
        for result in results:
            groups.setdefault((result.benchmark, result.model), []).append(result)

        aggregates: List[AggregateScore] = []
        for (benchmark, model), trials in groups.items():
            valid = [t for t in trials if t.is_valid]
            scores = [t.score for t in valid]
            mean, stdev, low, high = mean_with_ci(scores, self.run_cfg.confidence.level)
            aggregates.append(
                AggregateScore(
                    benchmark=benchmark,
                    model=model,
                    trials=len(trials),
                    mean=mean if valid else INVALID_SCORE,
                    stdev=stdev,
                    ci_low=low,
                    ci_high=high,
                    ci_level=self.run_cfg.confidence.level,
                    scores=scores,
                    sub_scores=mean_sub_scores([t.sub_scores for t in valid]),
                    n_samples=sum(t.n_samples for t in trials),
                    n_errors=sum(t.n_errors for t in trials),
                    cost_usd=round(sum(t.cost_usd for t in trials), 6),
                    valid=bool(valid),
                )
            )
        return aggregates

    def _report(self, document: EvaluateResult) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for spec in self.run_cfg.reporters:
            try:
                reporter = REPORTERS.get(spec.type)(**spec.params)
                cast(Any, reporter).report(document, self.output_dir)
            except Exception as exc:  # noqa: BLE001 - one sink must not lose the rest
                logger.warning("reporter %r failed: %s", spec.type, exc)

        out_s3 = self.config.output.s3_path
        if out_s3 and common_io.is_s3_uri(out_s3):
            try:
                common_io.s3_sync_dir(self.output_dir, out_s3.rstrip("/"))
            except Exception as exc:  # noqa: BLE001 - upload best-effort
                logger.warning("output S3 sync failed: %s", exc)


def _partition(
    rows: Sequence[Dict[str, Any]], keep: Callable[[Dict[str, Any]], bool]
) -> Tuple[List[Dict[str, Any]], int]:
    """Split ``rows`` into the ones to keep and a count of the ones dropped."""
    kept = [row for row in rows if keep(row)]
    return kept, len(rows) - len(kept)


def measurement_fingerprint(spec: BenchmarkSpec, model: Any) -> str:
    """Digest of everything that decides what a ``(benchmark, model)`` measures.

    Resume is otherwise keyed on names alone, so editing a prompt, swapping a
    metric, or pointing a benchmark at a different dataset would leave the old
    scores looking current and a re-run would replay them without calling a
    model. Folding the resolved spec and the model config into the checkpoint
    key means such an edit simply misses, and the affected trials re-run.

    ``weights_path`` and ``adapter_path`` are excluded because they are filled
    in at runtime from the SageMaker mounts: they differ between a laptop and a
    job for the same candidate, and are not part of what is being measured. The
    ``adapter`` they were staged from is included, so swapping adapters
    invalidates the trials that used the old one. ``display_name`` and
    ``leaderboard`` are excluded for the opposite reason -- they only decide how
    a result is labelled, so relabelling a candidate should not throw away the
    scores it already has.
    """
    payload = {
        "benchmark": spec.model_dump(mode="json", exclude_none=True),
        "model": model.model_dump(
            mode="json",
            exclude_none=True,
            exclude={
                "weights_path",
                "adapter_path",
                "display_name",
                "leaderboard",
            },
        ),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _without_resume_key(row: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild a BenchmarkResult payload from a flattened checkpoint row."""
    data = {k: v for k, v in row.items() if not k.startswith("sub_")}
    data.pop("resume_key", None)
    data.pop(FINGERPRINT_KEY, None)
    data["sub_scores"] = {
        k[len("sub_") :]: v for k, v in row.items() if k.startswith("sub_")
    }
    data.setdefault("detail", {})
    return data


def _strip_detail_columns(row: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild a SampleResult payload from a flattened checkpoint row."""
    data = {
        k: v
        for k, v in row.items()
        if not k.startswith("detail_") and k != FINGERPRINT_KEY
    }
    data["detail"] = {
        k[len("detail_") :]: v for k, v in row.items() if k.startswith("detail_")
    }
    return data


def build_summary_stats(aggregates: Sequence[AggregateScore]) -> Dict[str, Any]:
    """Structured summary for the JSON manifest (no pandas)."""
    by_model: Dict[str, List[float]] = {}
    for agg in aggregates:
        if agg.valid:
            by_model.setdefault(agg.model, []).append(agg.mean)

    return {
        "n_pairs": len(aggregates),
        "n_invalid": sum(1 for a in aggregates if not a.valid),
        "total_cost_usd": round(sum(a.cost_usd for a in aggregates), 6),
        "by_model": [
            {
                "model": model,
                "benchmarks": len(scores),
                "mean_score": round(sum(scores) / len(scores), 4),
            }
            for model, scores in sorted(by_model.items())
        ],
    }


# ---------------------------------------------------------------------------
# Public run + CLI
# ---------------------------------------------------------------------------


def run_evaluate(
    config: EvaluateJobConfig,
    *,
    output_dir: str | Path,
    checkpoint_dir: str | Path | None = None,
) -> EvaluateResult:
    """Run an evaluate job and return its result document."""
    runner = EvaluateRunner(
        config,
        output_dir=Path(output_dir),
        checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
    )
    return runner.run()


def mount_staged_model_weights(config: EvaluateJobConfig) -> None:
    """Point locally served models at the weights SageMaker already downloaded.

    The launcher mounts each S3-backed candidate as an input channel, so by the
    time this runs the weights are on local disk. Left alone, ``vllm serve``
    would be handed the original ``s3://`` URI and try to fetch the model
    itself -- which fails, because pulling from object storage needs an extra
    streaming package the image does not carry.

    A no-op outside SageMaker, where no channel directory exists.
    """
    for channel, model in local_model_channels(config.evaluate.models):
        mount = channel_mount_path(channel)
        if not os.path.isdir(mount):
            continue
        logger.info("serving %r from staged weights at %s", model.name, mount)
        model.weights_path = mount

    for channel, model in local_adapter_channels(config.evaluate.models):
        mount = channel_mount_path(channel)
        if not os.path.isdir(mount):
            continue
        logger.info("serving adapter for %r from %s", model.name, mount)
        model.adapter_path = mount


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="foresightrlcore.evaluate.runner",
        description="Container-side orchestrator for evaluate runs.",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("EVALUATE_CONFIG", DEFAULT_CONFIG_PATH),
        help="Path to the evaluate YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("EVALUATE_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        help="Output directory for reports.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=os.environ.get("EVALUATE_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR),
        help="Directory for the incremental checkpoints.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = EvaluateJobConfig.from_yaml(args.config)
    mount_staged_model_weights(config)
    result = run_evaluate(
        config,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
    logger.info("evaluate finished: %d result(s)", result.n_results)
    return 0


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    raise SystemExit(main())
