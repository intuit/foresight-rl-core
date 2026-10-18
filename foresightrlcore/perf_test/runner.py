"""Container-side orchestrator for the perf_test operation.

Invoked by ``perf_entrypoint.sh`` after Gradle and the perf-test Python
requirements have been installed. Responsibilities:

1. Load the config from ``args.yaml`` (SageMaker pattern, mirroring
   :mod:`foresightrlcore.runtime.scripts.train`).
2. Copy user-supplied test data (``prompts.csv``, ``system-message.txt``,
   ``chat-config.json``) from the SageMaker data channel into the vendored
   Gatling resources directory.
3. Start ``vllm serve`` as a subprocess; poll ``/health`` until 200.
4. For each ``tps_level`` in the run, execute ``gradle gatlingRun`` with
   the right JVM system properties; parse ``global_stats.json`` for
   percentile latencies.
5. Write a JSON + markdown summary to ``/opt/ml/output/data/`` and return
   a non-zero exit code on failure.

Designed so that the same module can also be run on a developer laptop
against a local vLLM server -- everything is parametrised by env vars or
CLI args.
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from foresightrlcore.config.perf_schema import (
    GatlingConfig,
    PerfTestJobConfig,
    PerfTestRunConfig,
    VllmServingConfig,
)
from foresightrlcore.perf_test.spec import RunResult, TpsResult

logger = logging.getLogger("foresightrlcore.perf_test.runner")


# Paths inside the SageMaker container.
DEFAULT_OUTPUT_DIR = "/opt/ml/output/data"
DEFAULT_MODEL_CHANNEL = "/opt/ml/input/data/model"
DEFAULT_DATA_CHANNEL = "/opt/ml/input/data/perf_data"
DEFAULT_CONFIG_PATH = "/opt/ml/input/data/config/args.yaml"

# Gradle resolves its plugin classpath over the network on every invocation,
# and Maven Central throttles the shared NAT address SageMaker egresses
# through. The resulting 429 is transient, so the sweep retries rather than
# writing off the whole run.
GRADLE_RESOLVE_ATTEMPTS = 4
GRADLE_RETRY_BACKOFF_SECONDS = 20

# Substrings that mark a Gradle failure as "couldn't fetch dependencies"
# rather than "the simulation itself failed".
_GRADLE_DEPENDENCY_FAILURE_MARKERS = (
    "Could not resolve all artifacts",
    "Could not resolve all dependencies",
    "Too Many Requests",
    "Could not GET",
    "Could not get resource",
)

# Vendored Gatling project location inside the source tarball.
_GATLING_PROJECT_DIR = Path(__file__).resolve().parent / "gatling"
_GATLING_RESOURCES_DIR = _GATLING_PROJECT_DIR / "src" / "gatling" / "resources"
_GATLING_JAVA_DIR = _GATLING_PROJECT_DIR / "src" / "gatling" / "java"

#: Directory inside an extension package holding extra Gatling sources, laid
#: out as a Java package tree (``com/example/Foo.java``). Everything under it is
#: copied into the vendored project's source set before Gradle runs, so an
#: extension can ship simulations that compile against the OSS classes without
#: forking the project.
_EXTENSION_JAVA_SUBDIR = Path("perf_test") / "gatling_java"
# `chat-config.json` is the only mandatory file -- ChatRequest.java loads it
# statically at class init for default model/temperature/max_tokens. Everything
# else (prompts.csv, prompts.jsonl, system-message.txt, ...) is simulation
# specific and copied opportunistically if the user supplies it.
_GATLING_REQUIRED_DATA = ("chat-config.json",)
# Files we know how to recognise from the data channel and mirror into the
# Gatling resources/ directory at runtime. Order matters only for the
# required-files check above.
_GATLING_OPTIONAL_DATA = (
    "prompts.csv",
    "prompts.jsonl",
    "system-message.txt",
)


class PerfTestRunner:
    """Run one :class:`PerfTestRunConfig` end-to-end."""

    def __init__(
        self,
        run: PerfTestRunConfig,
        *,
        model_path: str,
        data_channel: str,
        output_dir: Path,
        gatling_project_dir: Path = _GATLING_PROJECT_DIR,
    ) -> None:
        self.run = run
        self.model_path = model_path
        self.data_channel = data_channel
        self.output_dir = output_dir
        self.gatling_project_dir = gatling_project_dir
        self._vllm_proc: Optional[subprocess.Popen[bytes]] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self) -> RunResult:
        started_at = _now_iso()
        result = RunResult(
            name=self.run.name,
            model=self.model_path,
            simulation_class=self.run.gatling.simulation_class,
            started_at=started_at,
        )

        try:
            self._stage_data_channel()
            self._stage_extension_simulations()
            self._stage_speculative_draft()
            startup = self._start_vllm()
            result.vllm_startup_seconds = startup

            if self.run.run_gatling_test:
                for tps in self.run.gatling.tps_levels:
                    tps_result = self._run_gatling_tps(tps)
                    result.tps_results.append(tps_result)
            else:
                logger.info(
                    "run %r: run_gatling_test=False, skipping Gatling sweep "
                    "(smoke-test mode)",
                    self.run.name,
                )
        finally:
            self._stop_vllm()
            result.finished_at = _now_iso()

        return result

    # ------------------------------------------------------------------
    # Stage 1: data channel
    # ------------------------------------------------------------------

    def _stage_data_channel(self) -> None:
        """Copy user-supplied test data into the vendored Gatling resources."""
        src = Path(self.data_channel)
        if not src.exists():
            raise RuntimeError(
                f"perf_test data channel not found at {src}. "
                f"The launcher should have mounted "
                f"gatling.data_s3_path here."
            )

        missing = [f for f in _GATLING_REQUIRED_DATA if not (src / f).exists()]
        if missing:
            raise RuntimeError(
                f"Required test data files missing from {src}: {missing}. "
                f"Expected each of: {list(_GATLING_REQUIRED_DATA)}."
            )

        _GATLING_RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
        # Copy every required + recognised-optional file present in the
        # channel. We don't recursively mirror the channel because SageMaker
        # inputs can include adjacent unrelated files; sticking to the
        # known whitelist keeps the resources dir predictable.
        copied = []
        for fname in (*_GATLING_REQUIRED_DATA, *_GATLING_OPTIONAL_DATA):
            srcpath = src / fname
            if not srcpath.exists():
                continue
            dstpath = _GATLING_RESOURCES_DIR / fname
            shutil.copy2(srcpath, dstpath)
            copied.append(fname)
            logger.info("Copied %s -> %s", srcpath, dstpath)

        if not any(f in copied for f in ("prompts.csv", "prompts.jsonl")):
            logger.warning(
                "No prompts.csv or prompts.jsonl found in %s -- simulations "
                "that read a feeder file will fail at startup.",
                src,
            )

    # ------------------------------------------------------------------
    # Stage 1b: extension-provided simulations
    # ------------------------------------------------------------------

    def _stage_extension_simulations(self) -> List[str]:
        """Copy extension Gatling sources into the vendored project's source set.

        The OSS project ships only provider-neutral simulations. Anything
        specific to one deployment -- a served-model name, bespoke sampling
        parameters, a legacy class name that existing configs still reference --
        lives in an extension package under ``perf_test/gatling_java`` and is
        merged in here, so Gradle compiles both as one source set.

        Returns the staged paths, relative to the Java source root.
        """
        java_dir = self.gatling_project_dir / "src" / "gatling" / "java"
        staged: List[str] = []

        for root in _extension_java_roots():
            for source in sorted(root.rglob("*.java")):
                relative = source.relative_to(root)
                destination = java_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                staged.append(str(relative))

        if staged:
            logger.info(
                "Staged %d extension simulation source(s): %s",
                len(staged),
                ", ".join(staged),
            )
        return staged

    # ------------------------------------------------------------------
    # Stage 1b: speculative decoding draft staging
    # ------------------------------------------------------------------

    def _stage_speculative_draft(self) -> None:
        """Download an s3:// speculative-decoding draft to a local path.

        vLLM's ``--speculative-config`` expects a local filesystem path or
        HF-Hub id for the draft model, not an S3 URI. When the config has
        ``serving.speculative.model`` set to an ``s3://...`` prefix the
        runner mirrors that prefix to ``/tmp/speculative_draft_model/`` and
        rewrites the speculative config to point at the local directory.
        Mirrors :func:`runtime.scripts.train.py` behaviour for the GRPO
        backend (see docs/speculative-decoding.md, "S3 staging").
        """
        spec = self.run.serving.speculative
        if spec is None:
            return
        model_ref = getattr(spec, "model", None)
        if not isinstance(model_ref, str) or not model_ref.startswith("s3://"):
            return

        local_dir = Path("/tmp/speculative_draft_model")
        local_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Staging speculative draft %s -> %s ...", model_ref, local_dir)
        cmd = ["aws", "s3", "sync", model_ref, str(local_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to stage speculative draft from {model_ref}: "
                f"{result.stderr.strip()}"
            )
        # Rewrite the in-memory speculative config so _build_vllm_command
        # emits the local path to vLLM. We don't mutate the source YAML --
        # only the runtime copy held by this PerfTestRunner.
        spec.model = str(local_dir)
        logger.info(
            "Speculative draft staged: %d files in %s",
            len(list(local_dir.iterdir())),
            local_dir,
        )

    # ------------------------------------------------------------------
    # Stage 2: vllm serve
    # ------------------------------------------------------------------

    def _start_vllm(self) -> float:
        cmd = self._build_vllm_command()
        env = self._build_vllm_env()
        logger.info("Starting vllm serve: %s", " ".join(cmd))

        # New session so we can kill the whole process group on shutdown.
        self._vllm_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
            start_new_session=True,
        )

        start = time.monotonic()
        deadline = start + self.run.serving.startup_timeout
        while time.monotonic() < deadline:
            if self._vllm_proc.poll() is not None:
                raise RuntimeError(
                    f"vllm serve exited with code {self._vllm_proc.returncode} "
                    f"before becoming healthy"
                )
            if _vllm_health_ok(self.run.serving.port):
                elapsed = time.monotonic() - start
                logger.info("vllm serve healthy after %.1fs", elapsed)
                return elapsed
            time.sleep(5)

        raise RuntimeError(
            f"vllm serve did not become healthy within "
            f"{self.run.serving.startup_timeout}s"
        )

    def _build_vllm_command(self) -> List[str]:
        s = self.run.serving
        cmd: List[str] = [
            "vllm",
            "serve",
            self.model_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(s.port),
            "--tensor-parallel-size",
            str(s.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(s.gpu_memory_utilization),
            "--max-num-seqs",
            str(s.max_num_seqs),
        ]
        if s.max_model_len is not None:
            cmd += ["--max-model-len", str(s.max_model_len)]
        if s.served_model_name:
            cmd += ["--served-model-name", s.served_model_name]
        if s.trust_remote_code:
            cmd.append("--trust-remote-code")
        if s.enforce_eager:
            cmd.append("--enforce-eager")
        if s.dtype:
            cmd += ["--dtype", s.dtype]
        if s.quantization:
            cmd += ["--quantization", s.quantization]
        if s.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")
        if s.speculative is not None:
            cmd += [
                "--speculative-config",
                json.dumps(
                    s.speculative.model_dump(exclude_none=True),
                    separators=(",", ":"),
                ),
            ]
        # Lift VLLM_ATTENTION_BACKEND from vllm_env to the --attention-backend
        # CLI flag for vLLM >= 0.13, which deprecated the env var (see vLLM
        # PR #26315). Done here so existing configs using vllm_env keep
        # working without users needing to know about the migration. Only
        # added if the flag isn't already set via extra_args.
        attention_backend_env = self.run.serving.vllm_env.get("VLLM_ATTENTION_BACKEND")
        if attention_backend_env and "attention-backend" not in s.extra_args:
            cmd += ["--attention-backend", str(attention_backend_env)]

        for flag, value in s.extra_args.items():
            # Empty value => store_true option; see build_serve_command.
            cmd += [f"--{flag}"] if value == "" else [f"--{flag}", str(value)]
        return cmd

    # Names of legacy vLLM env vars that have been deprecated/removed
    # upstream and should not be passed to the child process. See vLLM
    # PR #26315 (VLLM_ATTENTION_BACKEND -> --attention-backend) and the V0
    # engine removal (VLLM_USE_V1 has no effect from ~0.13 onwards).
    _LEGACY_VLLM_ENV_VARS = {
        "VLLM_USE_V1": (
            "VLLM_USE_V1 has no effect since vLLM ~0.13 (V0 engine was "
            "removed; V1 is the only engine). Dropping silently."
        ),
        "VLLM_ATTENTION_BACKEND": (
            "VLLM_ATTENTION_BACKEND was replaced by --attention-backend "
            "in vLLM 0.13 (PR #26315). Lifting to the CLI flag."
        ),
    }

    def _build_vllm_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        for key, value in self.run.serving.vllm_env.items():
            if key in self._LEGACY_VLLM_ENV_VARS:
                logger.warning("vllm_env[%s]: %s", key, self._LEGACY_VLLM_ENV_VARS[key])
                env.pop(key, None)
                continue
            env[key] = value
        if self.run.serving.gpu_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.run.serving.gpu_devices
        return env

    def _stop_vllm(self) -> None:
        proc = self._vllm_proc
        if proc is None:
            return
        if proc.poll() is not None:
            logger.info("vllm serve already exited with code %s", proc.returncode)
            return

        logger.info("Sending SIGTERM to vllm serve (PID %s)", proc.pid)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError) as exc:
            logger.warning("Could not signal vllm serve cleanly: %s", exc)

        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning("vllm serve did not exit in 30s; sending SIGKILL")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.error("vllm serve still alive after SIGKILL; giving up")

    # ------------------------------------------------------------------
    # Stage 3: Gatling sweep
    # ------------------------------------------------------------------

    def _run_gatling_tps(self, tps: int) -> TpsResult:
        gatling = self.run.gatling
        result = TpsResult(
            target_tps=tps,
            duration_seconds=gatling.duration_seconds,
        )

        report_dir = self.output_dir / "gatling" / self.run.name / f"tps_{tps}"
        report_dir.mkdir(parents=True, exist_ok=True)

        cmd = self._build_gatling_command(tps, gatling, report_dir)
        logger.info("Running Gatling (TPS=%d): %s", tps, " ".join(cmd))

        returncode = self._run_gradle_with_retry(cmd, tps)
        if returncode != 0:
            result.error = f"gradle gatlingRun exited with code {returncode}"
            logger.error("Gatling run failed for TPS=%d: %s", tps, result.error)
            return result

        stats = _find_global_stats(self.gatling_project_dir, report_dir)
        if stats is None:
            result.error = "global_stats.json not found in Gatling report"
            return result

        _populate_result_from_stats(result, stats)
        return result

    def _run_gradle_with_retry(self, cmd: List[str], tps: int) -> int:
        """Run ``gradle gatlingRun``, retrying while Gradle can't fetch its deps.

        Gradle resolves the plugin classpath from Maven Central, which
        rate-limits the shared AWS NAT egress address SageMaker jobs go out
        through. When it does, every repository in ``settings.gradle`` answers
        ``429 Too Many Requests`` and the build dies during configuration:

            > Could not resolve io.gatling:gatling-enterprise-plugin-commons:1.9.3
              > Received status code 429 from server: Too Many Requests

        The throttle is transient, so a bounded backoff usually clears it.
        Only dependency-resolution failures are retried — a simulation that
        compiles and runs but fails on its own terms is reported immediately,
        since re-running it would just burn GPU time on the same result.
        """
        attempts = GRADLE_RESOLVE_ATTEMPTS
        for attempt in range(1, attempts + 1):
            completed = subprocess.run(
                cmd,
                cwd=str(self.gatling_project_dir),
                check=False,
                capture_output=True,
                text=True,
            )
            # Gradle writes the build log to stdout; echo it so container logs
            # stay as readable as they were before retries existed.
            if completed.stdout:
                sys.stdout.write(completed.stdout)
            if completed.stderr:
                sys.stderr.write(completed.stderr)

            if completed.returncode == 0:
                return 0

            output = (completed.stdout or "") + (completed.stderr or "")
            if not _is_dependency_resolution_failure(output):
                return completed.returncode

            if attempt == attempts:
                logger.error(
                    "Gatling TPS=%d: dependency resolution still failing after "
                    "%d attempts. Maven Central is throttling this egress "
                    "address; a cached/mirrored repository is needed for a "
                    "durable fix.",
                    tps,
                    attempts,
                )
                return completed.returncode

            delay = GRADLE_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Gatling TPS=%d: Gradle could not resolve its dependencies "
                "(attempt %d/%d, likely HTTP 429 from Maven Central). "
                "Retrying in %ds.",
                tps,
                attempt,
                attempts,
                delay,
            )
            time.sleep(delay)

        return 1

    def _build_gatling_command(
        self,
        tps: int,
        gatling: GatlingConfig,
        report_dir: Path,
    ) -> List[str]:
        # Size the closed-model standing pool to the production concurrency the
        # deployment is sized for (Little's Law: concurrency = tps * latency).
        # An explicit override wins; otherwise derive from the latency budget.
        # The legacy `tps * 100` over-provisioned the pool ~5x at the 20-TPS /
        # ~20s target and produced client-timeout KOs on a healthy server.
        # Only meaningful for the closed injection model; the open model uses a
        # fixed arrival rate (`constantUsersPerSec(tps)`) and ignores it.
        concurrent_users = gatling.concurrent_users or (
            tps * gatling.latency_budget_seconds
        )
        cmd: List[str] = [
            "gradle",
            "gatlingRun",
            f"--simulation={gatling.simulation_class}",
            f"-DsimulationTps={tps}",
            f"-DsimulationDuration={gatling.duration_seconds}",
            f"-DconcurrentUsers={concurrent_users}",
            f"-DinjectionModel={gatling.injection_model}",
            f"-DsimulationWorkspace={report_dir}",
        ]
        # Gatling reads -Dgatling.http.requestTimeout as a millisecond override
        # for its built-in 60_000 ms default. We always set it explicitly so
        # the run is reproducible regardless of how `gatling.conf` is bundled
        # in the vendored project. Users can override via gatling.extra_jvm_args
        # if they want a different value -- their entry will appear later in
        # the command line and win.
        cmd.append(f"-Dgatling.http.requestTimeout={gatling.request_timeout_ms}")
        for k, v in gatling.extra_jvm_args.items():
            flag = k if k.startswith("-D") else f"-D{k}"
            cmd.append(f"{flag}={v}")
        return cmd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    )


def _vllm_health_ok(port: int) -> bool:
    try:
        import requests
    except ImportError:  # pragma: no cover -- requirements ensure this
        logger.debug("requests not installed; falling back to subprocess curl")
        return _vllm_health_via_curl(port)
    try:
        resp = requests.get(f"http://localhost:{port}/health", timeout=2)
    except Exception:  # noqa: BLE001 -- broad ok during startup polling
        return False
    return resp.status_code == 200


def _vllm_health_via_curl(port: int) -> bool:
    try:
        result = subprocess.run(
            ["curl", "-sf", f"http://localhost:{port}/health"],
            capture_output=True,
            timeout=2,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _find_global_stats(project_dir: Path, report_dir: Path) -> Optional[Dict[str, Any]]:
    """Locate ``global_stats.json`` in the Gatling report tree.

    Gatling writes reports to ``<project>/build/reports/gatling/<simulation>-<ts>/js/global_stats.json``.
    We also copy the report tree into ``report_dir`` for the SageMaker output.
    """
    reports_root = project_dir / "build" / "reports" / "gatling"
    if not reports_root.exists():
        return None

    candidates = sorted(reports_root.glob("**/js/global_stats.json"))
    if not candidates:
        return None
    latest = candidates[-1]

    sim_dir = latest.parent.parent
    dest = report_dir / sim_dir.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(sim_dir, dest)

    with open(latest, "r") as f:
        return json.load(f)


def _populate_result_from_stats(result: TpsResult, stats: Dict[str, Any]) -> None:
    """Extract a few headline numbers from Gatling's ``global_stats.json``.

    Gatling's JSON is deeply nested; we read only the fields we summarise.
    """
    num_requests = stats.get("numberOfRequests", {})
    result.requests_total = _opt_int(num_requests.get("total"))
    result.requests_ok = _opt_int(num_requests.get("ok"))
    result.requests_ko = _opt_int(num_requests.get("ko"))

    mean_response = stats.get("meanResponseTime", {})
    result.mean_latency_ms = _opt_float(mean_response.get("total"))

    result.p50_latency_ms = _opt_float(stats.get("percentiles1", {}).get("total"))
    result.p75_latency_ms = _opt_float(stats.get("percentiles2", {}).get("total"))
    result.p95_latency_ms = _opt_float(stats.get("percentiles3", {}).get("total"))
    result.p99_latency_ms = _opt_float(stats.get("percentiles4", {}).get("total"))
    result.max_latency_ms = _opt_float(stats.get("maxResponseTime", {}).get("total"))
    result.actual_tps = _opt_float(
        stats.get("meanNumberOfRequestsPerSecond", {}).get("total")
    )


def _opt_int(value: Any) -> Optional[int]:
    if value is None or value == "-":
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _opt_float(value: Any) -> Optional[float]:
    if value is None or value == "-":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def write_markdown_summary(results: List[RunResult], path: Path) -> None:
    """Render a compact per-TPS markdown table for each run."""
    lines: List[str] = []
    for run_result in results:
        lines.append(f"# {run_result.name}")
        lines.append("")
        lines.append(f"* Model: `{run_result.model}`")
        lines.append(f"* Simulation: `{run_result.simulation_class}`")
        lines.append(f"* Started: {run_result.started_at}")
        if run_result.finished_at:
            lines.append(f"* Finished: {run_result.finished_at}")
        if run_result.vllm_startup_seconds is not None:
            lines.append(f"* vLLM startup: {run_result.vllm_startup_seconds:.1f}s")
        lines.append("")
        lines.append(
            "| TPS | actual TPS | p50 (ms) | p95 (ms) | p99 (ms) | "
            "max (ms) | ok | ko |"
        )
        lines.append(
            "| --- | ---------- | -------- | -------- | -------- | "
            "-------- | -- | -- |"
        )
        for tps in run_result.tps_results:
            row = (
                f"| {tps.target_tps} | {_fmt(tps.actual_tps)} | "
                f"{_fmt(tps.p50_latency_ms)} | {_fmt(tps.p95_latency_ms)} | "
                f"{_fmt(tps.p99_latency_ms)} | {_fmt(tps.max_latency_ms)} | "
                f"{tps.requests_ok or '-'} | {tps.requests_ko or '-'} |"
            )
            lines.append(row)
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"


def run_all(
    config: PerfTestJobConfig,
    *,
    output_dir: Path,
    model_path: str,
    data_channel: str,
) -> List[RunResult]:
    """Execute the single :class:`PerfTestRunConfig` in *config*."""
    runner = PerfTestRunner(
        config.perf_test,
        model_path=model_path,
        data_channel=data_channel,
        output_dir=output_dir,
    )
    return [runner.execute()]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="foresightrlcore.perf_test.runner",
        description="Container-side orchestrator for perf_test runs.",
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("PERF_TEST_CONFIG", DEFAULT_CONFIG_PATH),
        help="Path to the perf-test YAML config.",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("PERF_TEST_MODEL_PATH", DEFAULT_MODEL_CHANNEL),
        help="Local model path (e.g. SageMaker model channel mount).",
    )
    parser.add_argument(
        "--data-channel",
        default=os.environ.get("PERF_TEST_DATA_CHANNEL", DEFAULT_DATA_CHANNEL),
        help="Local directory with user-supplied prompts/system-message/chat-config.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("PERF_TEST_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        help="Output directory for results JSON + markdown.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = PerfTestJobConfig.from_yaml(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_all(
        config,
        output_dir=output_dir,
        model_path=args.model_path,
        data_channel=args.data_channel,
    )

    json_path = output_dir / "perf_results.json"
    with open(json_path, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)
    logger.info("Wrote results JSON: %s", json_path)

    md_path = output_dir / "perf_results.md"
    write_markdown_summary(results, md_path)
    logger.info("Wrote results markdown: %s", md_path)

    failed = [
        r for r in results if any(t.error for t in r.tps_results) or not r.tps_results
    ]
    if failed:
        for run in failed:
            for tps in run.tps_results:
                if tps.error:
                    logger.error(
                        "Run %r TPS=%s failed: %s", run.name, tps.target_tps, tps.error
                    )
            if not run.tps_results:
                logger.error("Run %r produced no TPS results", run.name)
        return 1
    return 0


def _extension_java_roots() -> List[Path]:
    """Locate ``perf_test/gatling_java`` inside each staged extension package.

    Reads the same ``FORESIGHT_EXTRA_PACKAGES`` list the plugin loader uses, so
    an extension is picked up here for exactly the jobs that already declare it
    under ``container.extra_packages`` -- no second place to configure.
    """
    from foresightrlcore.plugins import EXTRA_PACKAGES_ENV

    raw = os.environ.get(EXTRA_PACKAGES_ENV, "")
    roots: List[Path] = []
    for name in [part.strip() for part in raw.split(",") if part.strip()]:
        try:
            module = importlib.import_module(name)
        except ImportError:
            logger.warning(
                "Extension package %r is listed in %s but could not be "
                "imported; its Gatling simulations (if any) will be missing.",
                name,
                EXTRA_PACKAGES_ENV,
            )
            continue
        for base in getattr(module, "__path__", []):
            candidate = Path(base) / _EXTENSION_JAVA_SUBDIR
            if candidate.is_dir():
                roots.append(candidate)
    return roots


def _is_dependency_resolution_failure(output: str) -> bool:
    """True when a Gradle build died fetching dependencies, not running them."""
    return any(marker in output for marker in _GRADLE_DEPENDENCY_FAILURE_MARKERS)


def _load_yaml(path: str) -> Dict[str, Any]:  # pragma: no cover -- helper
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":  # pragma: no cover -- container entrypoint
    raise SystemExit(main())
