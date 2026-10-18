"""Tests for foresightrlcore.perf_test.runner -- orchestrator unit tests.

Heavy ops (subprocess, vLLM, Gatling) are mocked so the tests run without
GPUs or network access.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from foresightrlcore.config.perf_schema import (
    GatlingConfig,
    PerfTestJobConfig,
    PerfTestRunConfig,
    VllmServingConfig,
)
from foresightrlcore.config.schema import SpeculativeConfig
from foresightrlcore.perf_test import runner as runner_mod
from foresightrlcore.perf_test.runner import (
    PerfTestRunner,
    _opt_float,
    _opt_int,
    _populate_result_from_stats,
    write_markdown_summary,
)
from foresightrlcore.perf_test.spec import RunResult, TpsResult


def _make_run(**kwargs) -> PerfTestRunConfig:
    base = dict(
        name="run1",
        instance_type="ml.g5.xlarge",
        serving=VllmServingConfig(tensor_parallel_size=2, port=8080),
        gatling=GatlingConfig(
            data_s3_path="s3://b/d/",
            tps_levels=[1, 2],
            duration_seconds=60,
        ),
    )
    base.update(kwargs)
    return PerfTestRunConfig(**base)  # type: ignore[arg-type]


class TestBuildVllmCommand:
    def test_basic_command(self, tmp_path):
        run = _make_run()
        r = PerfTestRunner(
            run,
            model_path="/local/model",
            data_channel=str(tmp_path),
            output_dir=tmp_path / "out",
        )
        cmd = r._build_vllm_command()
        assert cmd[:3] == ["vllm", "serve", "/local/model"]
        assert "--port" in cmd and "8080" in cmd
        assert "--tensor-parallel-size" in cmd and "2" in cmd
        assert "--trust-remote-code" in cmd
        assert "--max-num-seqs" in cmd

    def test_speculative_serialised_as_json(self, tmp_path):
        run = _make_run(
            serving=VllmServingConfig(
                tensor_parallel_size=1,
                speculative=SpeculativeConfig(
                    method="eagle3",
                    num_speculative_tokens=5,
                    model="/draft/model",
                ),
            )
        )
        r = PerfTestRunner(
            run,
            model_path="/local/model",
            data_channel=str(tmp_path),
            output_dir=tmp_path / "out",
        )
        cmd = r._build_vllm_command()
        idx = cmd.index("--speculative-config")
        payload = json.loads(cmd[idx + 1])
        assert payload["method"] == "eagle3"
        assert payload["num_speculative_tokens"] == 5
        assert payload["model"] == "/draft/model"

    def test_extra_args_passed_through(self, tmp_path):
        run = _make_run(
            serving=VllmServingConfig(
                tensor_parallel_size=1,
                extra_args={"foo": "bar"},
            )
        )
        r = PerfTestRunner(
            run,
            model_path="/m",
            data_channel=str(tmp_path),
            output_dir=tmp_path / "out",
        )
        cmd = r._build_vllm_command()
        assert "--foo" in cmd and "bar" in cmd

    def test_quantization_and_dtype(self, tmp_path):
        run = _make_run(
            serving=VllmServingConfig(
                tensor_parallel_size=1,
                quantization="awq",
                dtype="bfloat16",
            )
        )
        r = PerfTestRunner(
            run,
            model_path="/m",
            data_channel=str(tmp_path),
            output_dir=tmp_path / "out",
        )
        cmd = r._build_vllm_command()
        assert "--quantization" in cmd and "awq" in cmd
        assert "--dtype" in cmd and "bfloat16" in cmd

    def test_enable_prefix_caching_is_bare_flag(self, tmp_path):
        # vLLM >= 0.10 treats --enable-prefix-caching as a BooleanOptionalAction
        # (no positional value). The pre-0.13 idiom of passing "yes" makes
        # vllm serve exit with "unrecognized arguments: yes". Regression guard
        # for the DELPHIAI-487 perf-test job that hit this on vLLM 0.20.x.
        run = _make_run(
            serving=VllmServingConfig(
                tensor_parallel_size=1,
                enable_prefix_caching=True,
            )
        )
        r = PerfTestRunner(
            run,
            model_path="/m",
            data_channel=str(tmp_path),
            output_dir=tmp_path / "out",
        )
        cmd = r._build_vllm_command()
        assert "--enable-prefix-caching" in cmd
        idx = cmd.index("--enable-prefix-caching")
        # The next token must not be "yes" / "no" / "true" / "false" -- it
        # should either be the end of the command or another --flag.
        if idx + 1 < len(cmd):
            assert cmd[idx + 1].startswith("--"), (
                f"--enable-prefix-caching should be a bare flag, got value "
                f"{cmd[idx + 1]!r} after it"
            )

    def test_enable_prefix_caching_omitted_when_false(self, tmp_path):
        run = _make_run(
            serving=VllmServingConfig(
                tensor_parallel_size=1,
                enable_prefix_caching=False,
            )
        )
        r = PerfTestRunner(
            run,
            model_path="/m",
            data_channel=str(tmp_path),
            output_dir=tmp_path / "out",
        )
        cmd = r._build_vllm_command()
        assert "--enable-prefix-caching" not in cmd


class TestBuildGatlingCommand:
    def test_request_timeout_default_matches_gatling_stock(self, tmp_path):
        # The runner always emits an explicit
        # -Dgatling.http.requestTimeout=<ms> so the value is reproducible and
        # visible in the gradle command line. The schema default 60_000 ms
        # matches Gatling's stock default (ramp TPS until requests KO --
        # saturation is the measurement).
        run = _make_run()
        r = PerfTestRunner(
            run,
            model_path="/m",
            data_channel=str(tmp_path),
            output_dir=tmp_path / "out",
        )
        cmd = r._build_gatling_command(
            tps=2, gatling=run.gatling, report_dir=tmp_path / "rpt"
        )
        assert "-Dgatling.http.requestTimeout=60000" in cmd

    def test_request_timeout_override(self, tmp_path):
        run = _make_run(
            gatling=GatlingConfig(
                data_s3_path="s3://b/d/",
                tps_levels=[1],
                duration_seconds=60,
                request_timeout_ms=120_000,
            )
        )
        r = PerfTestRunner(
            run,
            model_path="/m",
            data_channel=str(tmp_path),
            output_dir=tmp_path / "out",
        )
        cmd = r._build_gatling_command(
            tps=1, gatling=run.gatling, report_dir=tmp_path / "rpt"
        )
        assert "-Dgatling.http.requestTimeout=120000" in cmd

    def test_extra_jvm_args_can_override_request_timeout(self, tmp_path):
        # If the user explicitly sets gatling.http.requestTimeout via
        # extra_jvm_args, their entry must come after the runner-supplied
        # default so it wins -- Gatling/JVM use last-wins for repeated -D
        # flags.
        run = _make_run(
            gatling=GatlingConfig(
                data_s3_path="s3://b/d/",
                tps_levels=[1],
                duration_seconds=60,
                extra_jvm_args={"gatling.http.requestTimeout": "30000"},
            )
        )
        r = PerfTestRunner(
            run,
            model_path="/m",
            data_channel=str(tmp_path),
            output_dir=tmp_path / "out",
        )
        cmd = r._build_gatling_command(
            tps=1, gatling=run.gatling, report_dir=tmp_path / "rpt"
        )
        idx_default = cmd.index("-Dgatling.http.requestTimeout=60000")
        idx_override = cmd.index("-Dgatling.http.requestTimeout=30000")
        assert idx_default < idx_override

    def test_concurrent_users_derived_from_latency_budget(self, tmp_path):
        # When concurrent_users is unset, the pool is sized via Little's Law:
        # concurrentUsers = tps * latency_budget_seconds. This replaces the
        # legacy tps*100 which over-provisioned the standing pool ~5x at the
        # 20-TPS / ~20-30s-latency target and produced client-timeout KOs.
        run = _make_run(
            gatling=GatlingConfig(
                data_s3_path="s3://b/d/",
                tps_levels=[20],
                duration_seconds=60,
                latency_budget_seconds=30,
            )
        )
        r = PerfTestRunner(
            run,
            model_path="/m",
            data_channel=str(tmp_path),
            output_dir=tmp_path / "out",
        )
        cmd = r._build_gatling_command(
            tps=20, gatling=run.gatling, report_dir=tmp_path / "rpt"
        )
        assert "-DconcurrentUsers=600" in cmd
        assert "-DconcurrentUsers=2000" not in cmd

    def test_concurrent_users_explicit_override_wins(self, tmp_path):
        run = _make_run(
            gatling=GatlingConfig(
                data_s3_path="s3://b/d/",
                tps_levels=[20],
                duration_seconds=60,
                latency_budget_seconds=30,
                concurrent_users=400,
            )
        )
        r = PerfTestRunner(
            run,
            model_path="/m",
            data_channel=str(tmp_path),
            output_dir=tmp_path / "out",
        )
        cmd = r._build_gatling_command(
            tps=20, gatling=run.gatling, report_dir=tmp_path / "rpt"
        )
        assert "-DconcurrentUsers=400" in cmd

    def test_injection_model_defaults_to_open(self, tmp_path):
        # The open injection model (constantUsersPerSec) is the default and is
        # passed verbatim to the simulation so it measures latency at a fixed
        # arrival rate instead of flooding the server's admission queue.
        run = _make_run(
            gatling=GatlingConfig(
                data_s3_path="s3://b/d/",
                tps_levels=[20],
                duration_seconds=60,
            )
        )
        r = PerfTestRunner(
            run,
            model_path="/m",
            data_channel=str(tmp_path),
            output_dir=tmp_path / "out",
        )
        cmd = r._build_gatling_command(
            tps=20, gatling=run.gatling, report_dir=tmp_path / "rpt"
        )
        assert "-DinjectionModel=open" in cmd

    def test_injection_model_closed_passthrough(self, tmp_path):
        run = _make_run(
            gatling=GatlingConfig(
                data_s3_path="s3://b/d/",
                tps_levels=[20],
                duration_seconds=60,
                injection_model="closed",
            )
        )
        r = PerfTestRunner(
            run,
            model_path="/m",
            data_channel=str(tmp_path),
            output_dir=tmp_path / "out",
        )
        cmd = r._build_gatling_command(
            tps=20, gatling=run.gatling, report_dir=tmp_path / "rpt"
        )
        assert "-DinjectionModel=closed" in cmd


class TestVllmEnv:
    def test_vllm_env_merged_into_subprocess_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EXISTING", "yes")
        run = _make_run(
            serving=VllmServingConfig(
                tensor_parallel_size=1,
                vllm_env={"MY_CUSTOM_VAR": "1"},
                gpu_devices="0,1",
            )
        )
        r = PerfTestRunner(
            run,
            model_path="/m",
            data_channel=str(tmp_path),
            output_dir=tmp_path / "out",
        )
        env = r._build_vllm_env()
        assert env["EXISTING"] == "yes"
        assert env["MY_CUSTOM_VAR"] == "1"
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1"

    def test_vllm_use_v1_is_dropped(self, tmp_path, monkeypatch):
        # VLLM_USE_V1 has no effect on vLLM >= 0.13 (V0 engine was removed).
        # The runner should drop it silently instead of propagating it.
        monkeypatch.setenv("VLLM_USE_V1", "0")
        run = _make_run(
            serving=VllmServingConfig(
                tensor_parallel_size=1,
                vllm_env={"VLLM_USE_V1": "1"},
            )
        )
        r = PerfTestRunner(
            run,
            model_path="/local/model",
            data_channel=str(tmp_path),
            output_dir=tmp_path / "out",
        )
        env = r._build_vllm_env()
        assert "VLLM_USE_V1" not in env

    def test_vllm_attention_backend_is_lifted_to_cli(self, tmp_path):
        # VLLM_ATTENTION_BACKEND was deprecated by vLLM PR #26315 in favor of
        # the --attention-backend CLI flag. The runner should lift the env
        # var to the CLI flag and drop it from the subprocess env.
        run = _make_run(
            serving=VllmServingConfig(
                tensor_parallel_size=1,
                vllm_env={"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"},
            )
        )
        r = PerfTestRunner(
            run,
            model_path="/local/model",
            data_channel=str(tmp_path),
            output_dir=tmp_path / "out",
        )
        env = r._build_vllm_env()
        cmd = r._build_vllm_command()
        assert "VLLM_ATTENTION_BACKEND" not in env
        assert "--attention-backend" in cmd
        assert cmd[cmd.index("--attention-backend") + 1] == "FLASH_ATTN"

    def test_extra_args_attention_backend_wins_over_vllm_env(self, tmp_path):
        # If the user has already set attention-backend via extra_args, the
        # runner should not also add it from VLLM_ATTENTION_BACKEND.
        run = _make_run(
            serving=VllmServingConfig(
                tensor_parallel_size=1,
                vllm_env={"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"},
                extra_args={"attention-backend": "TRITON"},
            )
        )
        r = PerfTestRunner(
            run,
            model_path="/local/model",
            data_channel=str(tmp_path),
            output_dir=tmp_path / "out",
        )
        cmd = r._build_vllm_command()
        # Only one --attention-backend should be present, taking the
        # extra_args value (TRITON), not the legacy env-var value.
        assert cmd.count("--attention-backend") == 1
        assert cmd[cmd.index("--attention-backend") + 1] == "TRITON"


class TestDataChannelStaging:
    def test_missing_channel_raises(self, tmp_path):
        run = _make_run()
        r = PerfTestRunner(
            run,
            model_path="/m",
            data_channel=str(tmp_path / "missing"),
            output_dir=tmp_path / "out",
        )
        with pytest.raises(RuntimeError, match="not found"):
            r._stage_data_channel()

    def test_missing_files_raises(self, tmp_path):
        data = tmp_path / "data"
        data.mkdir()
        (data / "prompts.csv").write_text("prompt\nhi\n")
        # system-message.txt + chat-config.json missing
        run = _make_run()
        r = PerfTestRunner(
            run,
            model_path="/m",
            data_channel=str(data),
            output_dir=tmp_path / "out",
        )
        with pytest.raises(RuntimeError, match="missing"):
            r._stage_data_channel()

    def test_copies_user_files_to_gatling_resources(self, tmp_path):
        data = tmp_path / "data"
        data.mkdir()
        (data / "prompts.csv").write_text("prompt\nhi\n")
        (data / "system-message.txt").write_text("system")
        (data / "chat-config.json").write_text(
            json.dumps({"model": "x", "max_tokens": 100})
        )

        # Redirect the vendored resources dir into the temp tree so we don't
        # mutate the repository.
        fake_resources = tmp_path / "gatling_resources"
        with patch.object(
            runner_mod,
            "_GATLING_RESOURCES_DIR",
            fake_resources,
        ):
            r = PerfTestRunner(
                _make_run(),
                model_path="/m",
                data_channel=str(data),
                output_dir=tmp_path / "out",
            )
            r._stage_data_channel()

        for fname in ("prompts.csv", "system-message.txt", "chat-config.json"):
            assert (fake_resources / fname).exists()


class TestStatsParsing:
    def test_opt_int_handles_dash(self):
        assert _opt_int("-") is None
        assert _opt_int(None) is None
        assert _opt_int(5) == 5

    def test_opt_float_handles_dash(self):
        assert _opt_float("-") is None
        assert _opt_float(None) is None
        assert _opt_float(1.5) == 1.5

    def test_populate_result_from_stats(self):
        stats = {
            "numberOfRequests": {"total": 100, "ok": 95, "ko": 5},
            "meanResponseTime": {"total": 250.0},
            "percentiles1": {"total": 200.0},
            "percentiles2": {"total": 250.0},
            "percentiles3": {"total": 350.0},
            "percentiles4": {"total": 400.0},
            "maxResponseTime": {"total": 800.0},
            "meanNumberOfRequestsPerSecond": {"total": 3.4},
        }
        result = TpsResult(target_tps=4, duration_seconds=60)
        _populate_result_from_stats(result, stats)
        assert result.requests_total == 100
        assert result.requests_ok == 95
        assert result.requests_ko == 5
        assert result.mean_latency_ms == 250.0
        assert result.p99_latency_ms == 400.0
        assert result.actual_tps == 3.4


class TestMarkdownSummary:
    def test_renders_table(self, tmp_path):
        run_result = RunResult(
            name="r1",
            model="m",
            simulation_class="x.Y",
            started_at="2026-01-01T00:00:00+00:00",
            tps_results=[
                TpsResult(
                    target_tps=1,
                    duration_seconds=60,
                    actual_tps=0.95,
                    requests_ok=60,
                    requests_ko=0,
                    p50_latency_ms=200.0,
                    p99_latency_ms=400.0,
                )
            ],
        )
        out = tmp_path / "summary.md"
        write_markdown_summary([run_result], out)
        text = out.read_text()
        assert "# r1" in text
        assert "| 1 |" in text
        assert "400.0" in text

    def test_renders_with_optional_fields_set(self, tmp_path):
        run_result = RunResult(
            name="r1",
            model="m",
            simulation_class="x.Y",
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T01:00:00+00:00",
            vllm_startup_seconds=12.3,
            tps_results=[],
        )
        out = tmp_path / "summary.md"
        write_markdown_summary([run_result], out)
        text = out.read_text()
        assert "Finished: 2026-01-01T01:00:00+00:00" in text
        assert "vLLM startup: 12.3s" in text


# ---------------------------------------------------------------------------
# Coverage for the previously-uncovered orchestration surface.
# ---------------------------------------------------------------------------


class TestStartVllm:
    """Exercise the polling loop in :meth:`PerfTestRunner._start_vllm`.

    All subprocess + HTTP calls are mocked so the test runs without vLLM.
    """

    def _runner(self, tmp_path):
        run = _make_run(serving=VllmServingConfig(tensor_parallel_size=1, port=9999))
        # Shorten the startup timeout floor so failure tests don't sleep.
        run.serving.startup_timeout = 60
        return PerfTestRunner(
            run,
            model_path="/m",
            data_channel=str(tmp_path),
            output_dir=tmp_path / "out",
        )

    def test_start_vllm_returns_when_healthy(self, tmp_path):
        r = self._runner(tmp_path)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with (
            patch.object(runner_mod.subprocess, "Popen", return_value=mock_proc),
            patch.object(runner_mod, "_vllm_health_ok", return_value=True),
            patch.object(runner_mod.time, "sleep"),
        ):
            elapsed = r._start_vllm()

        assert elapsed >= 0
        assert r._vllm_proc is mock_proc

    def test_start_vllm_raises_if_process_exits(self, tmp_path):
        r = self._runner(tmp_path)
        mock_proc = MagicMock()
        # First poll returns 1 (exited), so we go into the raise branch.
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1

        with (
            patch.object(runner_mod.subprocess, "Popen", return_value=mock_proc),
            patch.object(runner_mod, "_vllm_health_ok", return_value=False),
            patch.object(runner_mod.time, "sleep"),
        ):
            with pytest.raises(RuntimeError, match="exited with code"):
                r._start_vllm()

    def test_start_vllm_raises_on_timeout(self, tmp_path):
        r = self._runner(tmp_path)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        # Make the monotonic clock jump past the deadline on the first
        # iteration so the loop exits via the timeout branch.
        times = iter([0.0, 0.0, 9999.0])
        with (
            patch.object(runner_mod.subprocess, "Popen", return_value=mock_proc),
            patch.object(runner_mod, "_vllm_health_ok", return_value=False),
            patch.object(runner_mod.time, "sleep"),
            patch.object(runner_mod.time, "monotonic", side_effect=lambda: next(times)),
        ):
            with pytest.raises(RuntimeError, match="did not become healthy"):
                r._start_vllm()


class TestStopVllm:
    def _runner(self, tmp_path):
        return PerfTestRunner(
            _make_run(),
            model_path="/m",
            data_channel=str(tmp_path),
            output_dir=tmp_path / "out",
        )

    def test_stop_when_no_proc_noop(self, tmp_path):
        # _vllm_proc is None -> early return; no exception.
        self._runner(tmp_path)._stop_vllm()

    def test_stop_when_already_exited_logs_and_returns(self, tmp_path):
        r = self._runner(tmp_path)
        proc = MagicMock()
        proc.poll.return_value = 0
        proc.returncode = 0
        r._vllm_proc = proc
        r._stop_vllm()  # Must not call killpg.

    def test_stop_sigterm_clean_exit(self, tmp_path):
        r = self._runner(tmp_path)
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 12345
        r._vllm_proc = proc

        with (
            patch.object(runner_mod.os, "killpg") as mock_killpg,
            patch.object(runner_mod.os, "getpgid", return_value=12345),
        ):
            r._stop_vllm()

        mock_killpg.assert_called_once()
        proc.wait.assert_called_once_with(timeout=30)

    def test_stop_escalates_to_sigkill_on_timeout(self, tmp_path):
        import subprocess as _subprocess

        r = self._runner(tmp_path)
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 12345
        proc.wait.side_effect = [
            _subprocess.TimeoutExpired(cmd="vllm", timeout=30),
            None,  # the second wait() (after SIGKILL) returns cleanly.
        ]
        r._vllm_proc = proc

        with (
            patch.object(runner_mod.os, "killpg") as mock_killpg,
            patch.object(runner_mod.os, "getpgid", return_value=12345),
        ):
            r._stop_vllm()

        # SIGTERM + SIGKILL == 2 killpg calls.
        assert mock_killpg.call_count == 2

    def test_stop_handles_process_lookup_error_on_signal(self, tmp_path):
        r = self._runner(tmp_path)
        proc = MagicMock()
        proc.poll.return_value = None
        proc.pid = 12345
        r._vllm_proc = proc

        with (
            patch.object(runner_mod.os, "killpg", side_effect=ProcessLookupError),
            patch.object(runner_mod.os, "getpgid", return_value=12345),
        ):
            # Should swallow the error and still call wait().
            r._stop_vllm()
        proc.wait.assert_called_once()


class TestHealthCheck:
    """Cover the two ``_vllm_health_*`` helpers."""

    def test_health_ok_returns_true_on_200(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            assert runner_mod._vllm_health_ok(8080) is True

    def test_health_ok_returns_false_on_non_200(self):
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 503
            assert runner_mod._vllm_health_ok(8080) is False

    def test_health_ok_returns_false_on_exception(self):
        with patch("requests.get", side_effect=ConnectionError("nope")):
            assert runner_mod._vllm_health_ok(8080) is False

    def test_curl_fallback_success(self):
        completed = MagicMock()
        completed.returncode = 0
        with patch.object(runner_mod.subprocess, "run", return_value=completed):
            assert runner_mod._vllm_health_via_curl(8080) is True

    def test_curl_fallback_failure_returns_false(self):
        completed = MagicMock()
        completed.returncode = 7
        with patch.object(runner_mod.subprocess, "run", return_value=completed):
            assert runner_mod._vllm_health_via_curl(8080) is False

    def test_curl_fallback_missing_binary_returns_false(self):
        with patch.object(runner_mod.subprocess, "run", side_effect=FileNotFoundError):
            assert runner_mod._vllm_health_via_curl(8080) is False


class TestFindGlobalStats:
    def test_returns_none_when_reports_root_missing(self, tmp_path):
        # No build/reports/gatling at all.
        report_dir = tmp_path / "report_dir"
        report_dir.mkdir()
        assert runner_mod._find_global_stats(tmp_path / "proj", report_dir) is None

    def test_returns_none_when_no_global_stats(self, tmp_path):
        proj = tmp_path / "proj"
        (proj / "build" / "reports" / "gatling").mkdir(parents=True)
        report_dir = tmp_path / "report_dir"
        report_dir.mkdir()
        assert runner_mod._find_global_stats(proj, report_dir) is None

    def test_parses_global_stats_and_copies_report(self, tmp_path):
        proj = tmp_path / "proj"
        sim_dir = (
            proj
            / "build"
            / "reports"
            / "gatling"
            / "filmperformancesimulation-20260101"
        )
        (sim_dir / "js").mkdir(parents=True)
        (sim_dir / "js" / "global_stats.json").write_text(
            json.dumps({"numberOfRequests": {"total": 5}})
        )
        report_dir = tmp_path / "report_dir"
        report_dir.mkdir()

        stats = runner_mod._find_global_stats(proj, report_dir)
        assert stats == {"numberOfRequests": {"total": 5}}
        # The simulation directory was copied into the destination.
        copied = report_dir / sim_dir.name
        assert copied.exists()
        assert (copied / "js" / "global_stats.json").exists()


class TestRunGatlingTps:
    """Exercise ``_run_gatling_tps`` end-to-end without invoking gradle."""

    def _runner(self, tmp_path):
        return PerfTestRunner(
            _make_run(),
            model_path="/m",
            data_channel=str(tmp_path),
            output_dir=tmp_path / "out",
        )

    @staticmethod
    def _completed(returncode, stdout=""):
        """A CompletedProcess double with real string streams.

        The runner captures and re-emits gradle's output, so these must be
        ``str`` rather than bare MagicMock attributes.
        """
        proc = MagicMock()
        proc.returncode = returncode
        proc.stdout = stdout
        proc.stderr = ""
        return proc

    def test_records_gradle_failure(self, tmp_path):
        r = self._runner(tmp_path)
        completed = self._completed(2, "Simulation failed")
        with patch.object(runner_mod.subprocess, "run", return_value=completed):
            tps_result = r._run_gatling_tps(tps=1)
        assert tps_result.error and "exited with code 2" in tps_result.error

    def test_records_missing_global_stats(self, tmp_path):
        r = self._runner(tmp_path)
        completed = self._completed(0)
        with (
            patch.object(runner_mod.subprocess, "run", return_value=completed),
            patch.object(runner_mod, "_find_global_stats", return_value=None),
        ):
            tps_result = r._run_gatling_tps(tps=1)
        assert tps_result.error == "global_stats.json not found in Gatling report"

    def test_records_happy_path_stats(self, tmp_path):
        r = self._runner(tmp_path)
        completed = self._completed(0)
        stats = {
            "numberOfRequests": {"total": 10, "ok": 10, "ko": 0},
            "meanResponseTime": {"total": 100.0},
            "percentiles1": {"total": 90.0},
            "percentiles2": {"total": 95.0},
            "percentiles3": {"total": 110.0},
            "percentiles4": {"total": 150.0},
            "maxResponseTime": {"total": 300.0},
            "meanNumberOfRequestsPerSecond": {"total": 1.0},
        }
        with (
            patch.object(runner_mod.subprocess, "run", return_value=completed),
            patch.object(runner_mod, "_find_global_stats", return_value=stats),
        ):
            tps_result = r._run_gatling_tps(tps=1)
        assert tps_result.error is None
        assert tps_result.requests_ok == 10
        assert tps_result.p99_latency_ms == 150.0


class TestSpeculativeDraftStaging:
    def _runner(self, tmp_path, speculative=None):
        serving = VllmServingConfig(tensor_parallel_size=1, speculative=speculative)
        run = _make_run(serving=serving)
        return PerfTestRunner(
            run,
            model_path="/m",
            data_channel=str(tmp_path),
            output_dir=tmp_path / "out",
        )

    def test_noop_when_no_speculative(self, tmp_path):
        # No speculative config -> early return, no subprocess.
        self._runner(tmp_path)._stage_speculative_draft()

    def test_noop_when_model_is_local(self, tmp_path):
        from foresightrlcore.config.schema import SpeculativeConfig

        spec = SpeculativeConfig(
            method="eagle3", num_speculative_tokens=5, model="/local/draft"
        )
        r = self._runner(tmp_path, speculative=spec)
        with patch.object(runner_mod.subprocess, "run") as mock_run:
            r._stage_speculative_draft()
        mock_run.assert_not_called()
        assert spec.model == "/local/draft"

    def test_stages_s3_draft_and_rewrites_model_path(self, tmp_path):
        from foresightrlcore.config.schema import SpeculativeConfig

        spec = SpeculativeConfig(
            method="eagle3", num_speculative_tokens=5, model="s3://bucket/draft/"
        )
        r = self._runner(tmp_path, speculative=spec)

        completed = MagicMock()
        completed.returncode = 0
        completed.stderr = ""

        local_dir = tmp_path / "speculative_local"
        local_dir.mkdir()
        (local_dir / "model.safetensors").write_text("weights")

        with (
            patch.object(runner_mod.subprocess, "run", return_value=completed),
            patch.object(
                runner_mod,
                "Path",
                side_effect=lambda p: (
                    local_dir if p == "/tmp/speculative_draft_model" else Path(p)
                ),
            ),
        ):
            r._stage_speculative_draft()

        assert spec.model == str(local_dir)

    def test_raises_when_s3_sync_fails(self, tmp_path):
        from foresightrlcore.config.schema import SpeculativeConfig

        spec = SpeculativeConfig(
            method="eagle3", num_speculative_tokens=5, model="s3://bucket/draft/"
        )
        r = self._runner(tmp_path, speculative=spec)

        completed = MagicMock()
        completed.returncode = 1
        completed.stderr = "boom"

        with patch.object(runner_mod.subprocess, "run", return_value=completed):
            with pytest.raises(RuntimeError, match="Failed to stage speculative draft"):
                r._stage_speculative_draft()


class TestExecuteOrchestration:
    def _runner(self, tmp_path, run_gatling=True):
        data = tmp_path / "data"
        data.mkdir()
        (data / "chat-config.json").write_text("{}")
        (data / "prompts.csv").write_text("prompt\nhi\n")
        (data / "system-message.txt").write_text("system")

        run = _make_run()
        run.run_gatling_test = run_gatling
        r = PerfTestRunner(
            run,
            model_path="/m",
            data_channel=str(data),
            output_dir=tmp_path / "out",
        )
        return r

    def test_execute_runs_full_loop(self, tmp_path):
        r = self._runner(tmp_path)
        fake_resources = tmp_path / "gatling_resources"

        with (
            patch.object(runner_mod, "_GATLING_RESOURCES_DIR", fake_resources),
            patch.object(r, "_start_vllm", return_value=1.5),
            patch.object(r, "_stop_vllm"),
            patch.object(
                r,
                "_run_gatling_tps",
                side_effect=lambda tps: TpsResult(target_tps=tps, duration_seconds=60),
            ),
        ):
            result = r.execute()

        assert result.vllm_startup_seconds == 1.5
        assert len(result.tps_results) == len(r.run.gatling.tps_levels)
        assert result.finished_at is not None

    def test_execute_skips_gatling_in_smoke_mode(self, tmp_path):
        r = self._runner(tmp_path, run_gatling=False)
        fake_resources = tmp_path / "gatling_resources"

        with (
            patch.object(runner_mod, "_GATLING_RESOURCES_DIR", fake_resources),
            patch.object(r, "_start_vllm", return_value=2.0),
            patch.object(r, "_stop_vllm"),
            patch.object(r, "_run_gatling_tps") as mock_gatling,
        ):
            result = r.execute()

        assert result.tps_results == []
        mock_gatling.assert_not_called()

    def test_execute_calls_stop_even_on_failure(self, tmp_path):
        r = self._runner(tmp_path)
        fake_resources = tmp_path / "gatling_resources"

        with (
            patch.object(runner_mod, "_GATLING_RESOURCES_DIR", fake_resources),
            patch.object(r, "_start_vllm", side_effect=RuntimeError("nope")),
            patch.object(r, "_stop_vllm") as mock_stop,
        ):
            with pytest.raises(RuntimeError):
                r.execute()

        mock_stop.assert_called_once()


class TestStageDataChannelWarnsWithoutPrompts:
    def test_warns_when_no_prompts(self, tmp_path, caplog):
        data = tmp_path / "data"
        data.mkdir()
        (data / "chat-config.json").write_text("{}")
        # prompts.csv / prompts.jsonl deliberately omitted.

        r = PerfTestRunner(
            _make_run(),
            model_path="/m",
            data_channel=str(data),
            output_dir=tmp_path / "out",
        )
        fake_resources = tmp_path / "gatling_resources"
        with patch.object(runner_mod, "_GATLING_RESOURCES_DIR", fake_resources):
            with caplog.at_level("WARNING", logger="foresightrlcore.perf_test.runner"):
                r._stage_data_channel()
        assert "No prompts.csv" in caplog.text


class TestRunAll:
    def test_run_all_wraps_runner_execute(self, tmp_path):
        data = tmp_path / "data"
        data.mkdir()
        (data / "chat-config.json").write_text("{}")
        (data / "prompts.csv").write_text("prompt\nhi\n")

        # Minimal in-memory config.
        cfg_dict = {
            "job": {"name": "perf-test"},
            "model": {"name": "/local/model"},
            "perf_test": {
                "name": "run1",
                "serving": {"tensor_parallel_size": 1},
                "gatling": {"data_s3_path": "s3://b/d/"},
            },
            "execution": {"type": "local"},
            "output": {"s3_path": str(tmp_path / "out")},
            "container": {"recipe": "qwen3_5"},
        }
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text(yaml.dump(cfg_dict))
        config = PerfTestJobConfig.from_yaml(cfg_path)

        fake_result = RunResult(
            name="run1",
            model="m",
            simulation_class="x",
            started_at="2026-01-01T00:00:00+00:00",
        )
        with patch.object(
            runner_mod.PerfTestRunner, "execute", return_value=fake_result
        ):
            results = runner_mod.run_all(
                config,
                output_dir=tmp_path / "out",
                model_path="/m",
                data_channel=str(data),
            )

        assert results == [fake_result]


class TestMainCli:
    def test_main_writes_results_and_returns_zero(self, tmp_path):
        data = tmp_path / "data"
        data.mkdir()
        (data / "chat-config.json").write_text("{}")
        (data / "prompts.csv").write_text("prompt\nhi\n")

        cfg_dict = {
            "job": {"name": "perf-test"},
            "model": {"name": "/local/m"},
            "perf_test": {
                "name": "run1",
                "serving": {"tensor_parallel_size": 1},
                "gatling": {"data_s3_path": "s3://b/d/"},
            },
            "execution": {"type": "local"},
            "output": {"s3_path": str(tmp_path / "out")},
            "container": {"recipe": "qwen3_5"},
        }
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text(yaml.dump(cfg_dict))
        out_dir = tmp_path / "out_dir"

        # Stub execute so we don't try to start vllm.
        ok_result = RunResult(
            name="run1",
            model="/local/m",
            simulation_class="x.Y",
            started_at="2026-01-01T00:00:00+00:00",
            tps_results=[TpsResult(target_tps=1, duration_seconds=60, requests_ok=10)],
        )
        with patch.object(runner_mod.PerfTestRunner, "execute", return_value=ok_result):
            rc = runner_mod.main(
                [
                    "--config",
                    str(cfg_path),
                    "--model-path",
                    "/local/m",
                    "--data-channel",
                    str(data),
                    "--output-dir",
                    str(out_dir),
                ]
            )

        assert rc == 0
        assert (out_dir / "perf_results.json").exists()
        assert (out_dir / "perf_results.md").exists()

    def _run_main(self, tmp_path, run_result):
        """Drive main() with ``execute`` stubbed to return ``run_result``."""
        data = tmp_path / "data"
        data.mkdir()
        (data / "chat-config.json").write_text("{}")
        (data / "prompts.csv").write_text("prompt\nhi\n")

        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text(
            yaml.dump(
                {
                    "job": {"name": "perf-test"},
                    "model": {"name": "/local/m"},
                    "perf_test": {
                        "name": "run1",
                        "serving": {"tensor_parallel_size": 1},
                        "gatling": {"data_s3_path": "s3://b/d/"},
                    },
                    "execution": {"type": "local"},
                    "output": {"s3_path": str(tmp_path / "out")},
                    "container": {"recipe": "qwen3_5"},
                }
            )
        )
        with patch.object(
            runner_mod.PerfTestRunner, "execute", return_value=run_result
        ):
            return runner_mod.main(
                [
                    "--config",
                    str(cfg_path),
                    "--model-path",
                    "/local/m",
                    "--data-channel",
                    str(data),
                    "--output-dir",
                    str(tmp_path / "out_dir"),
                ]
            )

    def test_main_returns_one_when_every_tps_level_errored(self, tmp_path):
        """A total Gatling wipeout must fail the job, not report success.

        Regression test. The previous guard was
        ``if failed and any(r.tps_results for r in results) is False``, which
        only tripped when no run produced any TpsResult at all. When Gatling
        died at Gradle configuration time every level still produced a
        TpsResult carrying an ``error``, so the guard was False and main()
        returned 0 — SageMaker reported Completed for a run in which not one
        request was ever sent.
        """
        rc = self._run_main(
            tmp_path,
            RunResult(
                name="run1",
                model="/local/m",
                simulation_class="x.Y",
                started_at="2026-01-01T00:00:00+00:00",
                tps_results=[
                    TpsResult(
                        target_tps=1,
                        duration_seconds=60,
                        error="gradle gatlingRun exited with code 1",
                    ),
                    TpsResult(
                        target_tps=2,
                        duration_seconds=60,
                        error="gradle gatlingRun exited with code 1",
                    ),
                ],
            ),
        )
        assert rc == 1

    def test_main_returns_one_when_a_single_tps_level_errored(self, tmp_path):
        """A partial failure is still a failure — one bad level fails the run."""
        rc = self._run_main(
            tmp_path,
            RunResult(
                name="run1",
                model="/local/m",
                simulation_class="x.Y",
                started_at="2026-01-01T00:00:00+00:00",
                tps_results=[
                    TpsResult(target_tps=1, duration_seconds=60, requests_ok=10),
                    TpsResult(
                        target_tps=2,
                        duration_seconds=60,
                        error="global_stats.json not found in Gatling report",
                    ),
                ],
            ),
        )
        assert rc == 1

    def test_main_returns_one_when_run_produced_no_tps_results(self, tmp_path):
        rc = self._run_main(
            tmp_path,
            RunResult(
                name="run1",
                model="/local/m",
                simulation_class="x.Y",
                started_at="2026-01-01T00:00:00+00:00",
                tps_results=[],
            ),
        )
        assert rc == 1


class TestGradleRetry:
    """Gradle dependency fetches are retried; simulation failures are not.

    Maven Central answers 429 to the shared NAT address SageMaker egresses
    through, which kills the build during configuration before a single
    request is sent. That is transient, so it is worth retrying. A simulation
    that actually ran and failed is not.
    """

    @staticmethod
    def _runner(tmp_path):
        r = object.__new__(runner_mod.PerfTestRunner)
        r.gatling_project_dir = tmp_path
        return r

    @staticmethod
    def _completed(returncode, stdout=""):
        proc = MagicMock()
        proc.returncode = returncode
        proc.stdout = stdout
        proc.stderr = ""
        return proc

    THROTTLED = (
        "> Could not resolve all artifacts for configuration ':classpath'.\n"
        "   > Could not GET 'https://repo1.maven.org/maven2/io/gatling/"
        "gatling-enterprise-plugin-commons/1.9.3/...pom'. "
        "Received status code 429 from server: Too Many Requests"
    )

    def test_retries_then_succeeds_on_throttled_resolution(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(runner_mod.time, "sleep", lambda s: calls.append(s))
        results = [
            self._completed(1, self.THROTTLED),
            self._completed(1, self.THROTTLED),
            self._completed(0, "BUILD SUCCESSFUL"),
        ]
        with patch.object(
            runner_mod.subprocess, "run", side_effect=results
        ) as mock_run:
            rc = self._runner(tmp_path)._run_gradle_with_retry(["gradle"], tps=1)

        assert rc == 0
        assert mock_run.call_count == 3
        assert calls == [20, 40], "backoff should double between attempts"

    def test_gives_up_after_configured_attempts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(runner_mod.time, "sleep", lambda s: None)
        throttled = self._completed(1, self.THROTTLED)
        with patch.object(
            runner_mod.subprocess, "run", return_value=throttled
        ) as mock_run:
            rc = self._runner(tmp_path)._run_gradle_with_retry(["gradle"], tps=1)

        assert rc == 1
        assert mock_run.call_count == runner_mod.GRADLE_RESOLVE_ATTEMPTS

    def test_simulation_failure_is_not_retried(self, tmp_path, monkeypatch):
        """A build that ran and failed must not be re-run — that burns GPU time."""
        monkeypatch.setattr(runner_mod.time, "sleep", lambda s: None)
        failed = self._completed(1, "Simulation finished with 12 failed requests")
        with patch.object(
            runner_mod.subprocess, "run", return_value=failed
        ) as mock_run:
            rc = self._runner(tmp_path)._run_gradle_with_retry(["gradle"], tps=1)

        assert rc == 1
        assert mock_run.call_count == 1

    def test_success_on_first_attempt_does_not_sleep(self, tmp_path, monkeypatch):
        slept = []
        monkeypatch.setattr(runner_mod.time, "sleep", lambda s: slept.append(s))
        with patch.object(
            runner_mod.subprocess, "run", return_value=self._completed(0)
        ) as mock_run:
            rc = self._runner(tmp_path)._run_gradle_with_retry(["gradle"], tps=1)

        assert rc == 0
        assert mock_run.call_count == 1
        assert slept == []

    @pytest.mark.parametrize(
        "output,expected",
        [
            ("Received status code 429 from server: Too Many Requests", True),
            ("> Could not resolve all artifacts for configuration ':classpath'.", True),
            ("> Could not get resource 'https://repo1.maven.org/...'", True),
            ("Simulation finished with 12 failed requests", False),
            ("", False),
        ],
    )
    def test_dependency_failure_detection(self, output, expected):
        assert runner_mod._is_dependency_resolution_failure(output) is expected


class TestGatlingPluginRepositories:
    """The Gradle buildscript classpath needs a directly reachable mirror.

    ``gradlePluginPortal()`` serves plugin markers itself but answers requests
    for a plugin's transitive dependencies with a 303 to
    repo.maven.apache.org, which the SageMaker VPC cannot reach. Without a
    ``pluginManagement`` block declaring another repository, resolving
    io.gatling:gatling-enterprise-plugin-commons fails and every gatlingRun
    dies during configuration.
    """

    @staticmethod
    def _settings_text():
        """settings.gradle with `//` comments stripped.

        The rationale comment names these same repositories, so matching on the
        raw text would assert against prose rather than directives.
        """
        raw = (
            Path(runner_mod.__file__).parent / "gatling" / "settings.gradle"
        ).read_text()
        return "\n".join(
            line for line in raw.splitlines() if not line.strip().startswith("//")
        )

    def test_settings_declares_plugin_management_repositories(self):
        text = self._settings_text()
        assert "pluginManagement" in text
        assert "gradlePluginPortal()" in text

    def test_direct_maven_mirror_precedes_the_plugin_portal(self):
        """repo1 must come first so transitive artifacts skip the redirect."""
        text = self._settings_text()
        mirror = text.find("repo1.maven.org")
        portal = text.find("gradlePluginPortal()")
        assert mirror != -1, "no direct Maven mirror declared for the classpath"
        assert mirror < portal


class TestFmtAndOpts:
    def test_fmt_none_returns_dash(self):
        from foresightrlcore.perf_test.runner import _fmt

        assert _fmt(None) == "-"

    def test_fmt_value_formatted(self):
        from foresightrlcore.perf_test.runner import _fmt

        assert _fmt(1.234) == "1.2"

    def test_opt_int_invalid_string_returns_none(self):
        assert _opt_int("abc") is None

    def test_opt_float_invalid_string_returns_none(self):
        assert _opt_float("abc") is None


class TestStageExtensionSimulations:
    """Merging extension-provided Gatling sources into the vendored project.

    The OSS project ships only provider-neutral simulations; anything tied to a
    specific deployment arrives from a package listed in
    `container.extra_packages`. Gradle compiles one source set, so the runner
    has to copy that tree in before it runs.
    """

    @staticmethod
    def _runner(tmp_path):
        project = tmp_path / "gatling"
        (project / "src" / "gatling" / "java").mkdir(parents=True)
        return PerfTestRunner(
            _make_run(),
            model_path="/local/model",
            data_channel=str(tmp_path),
            output_dir=tmp_path / "out",
            gatling_project_dir=project,
        )

    @staticmethod
    def _fake_package(tmp_path, name, files):
        """Build a package dir laid out the way a staged extension looks."""
        pkg_root = tmp_path / name
        java_root = pkg_root / "perf_test" / "gatling_java"
        for relative, body in files.items():
            path = java_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
        module = MagicMock()
        module.__path__ = [str(pkg_root)]
        return module

    def test_no_env_var_stages_nothing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FORESIGHT_EXTRA_PACKAGES", raising=False)
        r = self._runner(tmp_path)
        assert r._stage_extension_simulations() == []

    def test_copies_sources_preserving_package_layout(self, tmp_path, monkeypatch):
        module = self._fake_package(
            tmp_path,
            "ext_pkg",
            {"com/example/perf/MySim.java": "package com.example.perf;\n"},
        )
        monkeypatch.setenv("FORESIGHT_EXTRA_PACKAGES", "ext_pkg")
        r = self._runner(tmp_path)

        with patch.object(runner_mod.importlib, "import_module", return_value=module):
            staged = r._stage_extension_simulations()

        assert staged == ["com/example/perf/MySim.java"]
        # The directory layout must survive the copy: javac rejects a file whose
        # package declaration doesn't match its path.
        copied = (
            r.gatling_project_dir
            / "src"
            / "gatling"
            / "java"
            / "com"
            / "example"
            / "perf"
            / "MySim.java"
        )
        assert copied.read_text(encoding="utf-8") == "package com.example.perf;\n"

    def test_non_java_files_are_ignored(self, tmp_path, monkeypatch):
        module = self._fake_package(
            tmp_path,
            "ext_pkg",
            {"com/example/perf/MySim.java": "x", "com/example/perf/notes.md": "y"},
        )
        monkeypatch.setenv("FORESIGHT_EXTRA_PACKAGES", "ext_pkg")
        r = self._runner(tmp_path)

        with patch.object(runner_mod.importlib, "import_module", return_value=module):
            staged = r._stage_extension_simulations()

        assert staged == ["com/example/perf/MySim.java"]

    def test_package_without_gatling_java_is_skipped(self, tmp_path, monkeypatch):
        """Most extensions ship no Java at all; that must not be an error."""
        pkg_root = tmp_path / "plain_pkg"
        pkg_root.mkdir()
        module = MagicMock()
        module.__path__ = [str(pkg_root)]
        monkeypatch.setenv("FORESIGHT_EXTRA_PACKAGES", "plain_pkg")
        r = self._runner(tmp_path)

        with patch.object(runner_mod.importlib, "import_module", return_value=module):
            assert r._stage_extension_simulations() == []

    def test_unimportable_package_warns_but_does_not_raise(
        self, tmp_path, monkeypatch, caplog
    ):
        """A perf run for an OSS simulation shouldn't die over a missing extra."""
        monkeypatch.setenv("FORESIGHT_EXTRA_PACKAGES", "missing_pkg")
        r = self._runner(tmp_path)

        with patch.object(
            runner_mod.importlib, "import_module", side_effect=ImportError("nope")
        ):
            with caplog.at_level("WARNING"):
                assert r._stage_extension_simulations() == []

        assert "missing_pkg" in caplog.text

    def test_multiple_packages_are_all_staged(self, tmp_path, monkeypatch):
        first = self._fake_package(tmp_path, "pkg_a", {"a/A.java": "package a;\n"})
        second = self._fake_package(tmp_path, "pkg_b", {"b/B.java": "package b;\n"})
        monkeypatch.setenv("FORESIGHT_EXTRA_PACKAGES", "pkg_a, pkg_b")
        r = self._runner(tmp_path)

        with patch.object(
            runner_mod.importlib, "import_module", side_effect=[first, second]
        ):
            staged = r._stage_extension_simulations()

        assert sorted(staged) == ["a/A.java", "b/B.java"]
