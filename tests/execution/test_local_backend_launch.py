"""Tests for foresightrlcore.execution.local — LocalBackend launch/subprocess."""

from __future__ import annotations

import os
import signal
import subprocess
from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest
import yaml

from foresightrlcore.config.schema import TrainingJobConfig
from foresightrlcore.execution.local import LocalBackend


def _make_local_config(overrides: dict | None = None) -> TrainingJobConfig:
    import tempfile

    base = {
        "job": {"name": "local-test"},
        "model": {"name": "Qwen/Qwen3-4B"},
        "method": {"type": "grpo"},
        "training": {"grpo": {"num_generations": 4}},
        "infra": {"vllm_mode": "server", "vllm": {"port": 8002}},
        "data": {"train_path": "/data/train"},
        "output": {"s3_path": "/output/model"},
        "execution": {
            "type": "local",
            "local": {
                "gpus": "0,1,2,3,4,5,6,7",
                "output_dir": "/tmp/test_training_output",
            },
        },
        "task": {"type": "advice", "config": {"reward_mode": "mock"}},
    }
    if overrides:
        _deep_merge(base, overrides)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(base, f)
        path = f.name
    try:
        return TrainingJobConfig.from_yaml(path)
    finally:
        os.unlink(path)


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


PATCH_ENSURE_DATA = "foresightrlcore.execution.local.LocalBackend._ensure_local_data"


class TestLocalBackendLaunch:
    def test_launch_server_mode(self, tmp_path):
        config = _make_local_config(
            {
                "execution": {
                    "type": "local",
                    "local": {
                        "gpus": "0,1,2,3,4,5,6,7",
                        "output_dir": str(tmp_path),
                    },
                }
            }
        )
        backend = LocalBackend()

        mock_popen = MagicMock()
        mock_popen.poll.return_value = None
        mock_result = MagicMock()
        mock_result.returncode = 0

        with (
            patch("subprocess.Popen", return_value=mock_popen) as mock_popen_cls,
            patch("subprocess.run", return_value=mock_result),
            patch("time.sleep"),
            patch("os.killpg"),
            patch("os.getpgid", return_value=12345),
            patch(PATCH_ENSURE_DATA),
        ):
            job_id = backend.launch(config)

        assert job_id.startswith("local-local-test-")
        mock_popen_cls.assert_called_once()

    def test_launch_none_mode(self, tmp_path):
        config = _make_local_config(
            {
                "infra": {"vllm_mode": "none"},
                "execution": {
                    "type": "local",
                    "local": {"gpus": "0,1,2,3,4,5,6,7", "output_dir": str(tmp_path)},
                },
            }
        )
        backend = LocalBackend()

        mock_result = MagicMock()
        mock_result.returncode = 0

        with (
            patch("subprocess.run", return_value=mock_result) as mock_run,
            patch(PATCH_ENSURE_DATA),
        ):
            job_id = backend.launch(config)

        assert job_id.startswith("local-local-test-")
        cmd = mock_run.call_args[0][0]
        assert "torchrun" in cmd

    def test_launch_colocate_mode(self, tmp_path):
        config = _make_local_config(
            {
                "infra": {"vllm_mode": "colocate"},
                "execution": {
                    "type": "local",
                    "local": {"gpus": "0,1,2,3,4,5,6,7", "output_dir": str(tmp_path)},
                },
            }
        )
        backend = LocalBackend()

        mock_result = MagicMock()
        mock_result.returncode = 0

        with (
            patch("subprocess.run", return_value=mock_result),
            patch(PATCH_ENSURE_DATA),
        ):
            job_id = backend.launch(config)

        assert job_id.startswith("local-")

    def test_launch_training_failure_raises(self, tmp_path):
        config = _make_local_config(
            {
                "infra": {"vllm_mode": "none"},
                "execution": {
                    "type": "local",
                    "local": {"gpus": "0,1,2,3,4,5,6,7", "output_dir": str(tmp_path)},
                },
            }
        )
        backend = LocalBackend()

        mock_result = MagicMock()
        mock_result.returncode = 1

        with (
            patch("subprocess.run", return_value=mock_result),
            patch(PATCH_ENSURE_DATA),
        ):
            with pytest.raises(RuntimeError, match="exited with code 1"):
                backend.launch(config)

    def test_launch_writes_config(self, tmp_path):
        config = _make_local_config(
            {
                "execution": {
                    "type": "local",
                    "local": {"gpus": "0,1", "output_dir": str(tmp_path)},
                },
                "infra": {"vllm_mode": "none"},
            }
        )
        backend = LocalBackend()

        mock_result = MagicMock()
        mock_result.returncode = 0

        with (
            patch("subprocess.run", return_value=mock_result),
            patch(PATCH_ENSURE_DATA),
        ):
            backend.launch(config)

        assert (tmp_path / "args.yaml").exists()

    def test_launch_with_config_path(self, tmp_path):
        config_file = tmp_path / "existing.yaml"
        config_file.write_text("dummy: true")

        config = _make_local_config(
            {
                "execution": {
                    "type": "local",
                    "local": {
                        "gpus": "0,1",
                        "output_dir": str(tmp_path),
                        "config_path": str(config_file),
                    },
                },
                "infra": {"vllm_mode": "none"},
            }
        )
        backend = LocalBackend()

        mock_result = MagicMock()
        mock_result.returncode = 0

        with (
            patch("subprocess.run", return_value=mock_result) as mock_run,
            patch(PATCH_ENSURE_DATA),
        ):
            backend.launch(config)

        cmd = mock_run.call_args[0][0]
        assert str(config_file) in cmd


class TestOutputDirFallback:
    def test_s3_path_creates_local_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = _make_local_config(
            {
                "execution": {
                    "type": "local",
                    "local": {"gpus": "0,1", "output_dir": None},
                },
                "infra": {"vllm_mode": "none"},
                "output": {"s3_path": "s3://bucket/prefix/job-output"},
            }
        )
        backend = LocalBackend()

        mock_result = MagicMock()
        mock_result.returncode = 0

        with (
            patch("subprocess.run", return_value=mock_result),
            patch(PATCH_ENSURE_DATA),
        ):
            backend.launch(config)

        expected_dir = tmp_path / "output" / config.job.name
        assert expected_dir.is_dir()
        assert (expected_dir / "args.yaml").exists()
        assert not (tmp_path / "s3:").exists()

    def test_local_path_used_directly(self, tmp_path):
        output = str(tmp_path / "my-output")
        config = _make_local_config(
            {
                "execution": {
                    "type": "local",
                    "local": {"gpus": "0,1", "output_dir": output},
                },
                "infra": {"vllm_mode": "none"},
            }
        )
        backend = LocalBackend()

        mock_result = MagicMock()
        mock_result.returncode = 0

        with (
            patch("subprocess.run", return_value=mock_result),
            patch(PATCH_ENSURE_DATA),
        ):
            backend.launch(config)

        assert os.path.isdir(output)
        assert os.path.exists(os.path.join(output, "args.yaml"))


class TestLocalBackendGetStatus:
    def test_get_status(self):
        backend = LocalBackend()
        status = backend.get_status("local-test-123")
        assert status["job_id"] == "local-test-123"
        assert status["status"] == "LOCAL_COMPLETED"


class TestStartVllmServer:
    def test_start_success(self):
        config = _make_local_config()
        backend = LocalBackend()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with patch("subprocess.Popen", return_value=mock_proc), patch("time.sleep"):
            proc = backend._start_vllm_server(config, [0, 1, 2, 3, 4, 5, 6, 7])

        assert proc is mock_proc

    def test_start_premature_exit(self):
        config = _make_local_config()
        backend = LocalBackend()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1

        with patch("subprocess.Popen", return_value=mock_proc), patch("time.sleep"):
            with pytest.raises(RuntimeError, match="exited prematurely"):
                backend._start_vllm_server(config, [0, 1])

    def test_enforce_eager_flag(self):
        config = _make_local_config(
            {"infra": {"vllm": {"enforce_eager": True, "port": 8002}}}
        )
        backend = LocalBackend()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with (
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch("time.sleep"),
        ):
            backend._start_vllm_server(config, [0, 1])

        cmd = mock_popen.call_args[0][0]
        assert "--enforce_eager" in cmd

    def test_speculative_config_appended_as_json(self):
        # Local-path draft — no staging needed, JSON passes through untouched.
        config = _make_local_config(
            {
                "infra": {
                    "vllm": {
                        "speculative": {
                            "method": "eagle3",
                            "num_speculative_tokens": 5,
                            "model": "/tmp/local_draft_model",
                            "draft_tensor_parallel_size": 1,
                        }
                    }
                }
            }
        )
        backend = LocalBackend()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with (
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch("time.sleep"),
        ):
            backend._start_vllm_server(config, [0, 1])

        cmd = mock_popen.call_args[0][0]
        assert "--speculative-config" in cmd
        idx = cmd.index("--speculative-config")
        assert cmd[idx + 1] == (
            '{"method":"eagle3","num_speculative_tokens":5,'
            '"model":"/tmp/local_draft_model","draft_tensor_parallel_size":1}'
        )

    def test_speculative_mtp_no_draft_model(self):
        # Built-in MTP head — no `model` field. JSON should omit it.
        config = _make_local_config(
            {
                "infra": {
                    "vllm": {
                        "speculative": {
                            "method": "qwen3_next_mtp",
                            "num_speculative_tokens": 5,
                        }
                    }
                }
            }
        )
        backend = LocalBackend()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with (
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch("time.sleep"),
        ):
            backend._start_vllm_server(config, [0, 1])

        cmd = mock_popen.call_args[0][0]
        idx = cmd.index("--speculative-config")
        assert cmd[idx + 1] == '{"method":"qwen3_next_mtp","num_speculative_tokens":5}'

    def test_speculative_s3_draft_is_staged_locally(self, tmp_path):
        s3_uri = "s3://my-bucket/path/to/draft/"
        config = _make_local_config(
            {
                "execution": {
                    "type": "local",
                    "local": {
                        "gpus": "0,1,2,3,4,5,6,7",
                        "output_dir": str(tmp_path),
                    },
                },
                "infra": {
                    "vllm": {
                        "speculative": {
                            "method": "eagle3",
                            "num_speculative_tokens": 5,
                            "model": s3_uri,
                            "draft_tensor_parallel_size": 1,
                        }
                    }
                },
            }
        )
        backend = LocalBackend()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        sync_result = MagicMock()
        sync_result.returncode = 0

        with (
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch("subprocess.run", return_value=sync_result) as mock_run,
            patch("time.sleep"),
        ):
            backend._start_vllm_server(config, [0, 1])

        # aws s3 sync was invoked with the right source/dest
        sync_cmd = mock_run.call_args[0][0]
        expected_stage_dir = str(tmp_path / "speculative_draft_model")
        assert sync_cmd[:3] == ["aws", "s3", "sync"]
        assert sync_cmd[3] == s3_uri
        assert sync_cmd[4] == expected_stage_dir

        # The vllm-serve cmd has the local stage dir, not the s3:// URI.
        cmd = mock_popen.call_args[0][0]
        idx = cmd.index("--speculative-config")
        assert s3_uri not in cmd[idx + 1]
        assert expected_stage_dir in cmd[idx + 1]

    def test_speculative_s3_sync_failure_raises(self, tmp_path):
        config = _make_local_config(
            {
                "execution": {
                    "type": "local",
                    "local": {
                        "gpus": "0,1,2,3,4,5,6,7",
                        "output_dir": str(tmp_path),
                    },
                },
                "infra": {
                    "vllm": {
                        "speculative": {
                            "method": "eagle3",
                            "num_speculative_tokens": 5,
                            "model": "s3://my-bucket/missing/",
                        }
                    }
                },
            }
        )
        backend = LocalBackend()

        sync_result = MagicMock()
        sync_result.returncode = 1
        sync_result.stderr = "NoSuchKey"

        with (
            patch("subprocess.Popen") as mock_popen,
            patch("subprocess.run", return_value=sync_result),
            patch("time.sleep"),
        ):
            with pytest.raises(RuntimeError, match="Failed to stage speculative draft"):
                backend._start_vllm_server(config, [0, 1])

        # vLLM was never launched
        mock_popen.assert_not_called()

    def test_speculative_omitted_when_unset(self):
        config = _make_local_config()
        backend = LocalBackend()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with (
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch("time.sleep"),
        ):
            backend._start_vllm_server(config, [0, 1])

        cmd = mock_popen.call_args[0][0]
        assert "--speculative-config" not in cmd

    def test_enable_prefix_caching_flag(self):
        config = _make_local_config(
            {"infra": {"vllm": {"enable_prefix_caching": True}}}
        )
        backend = LocalBackend()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with (
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch("time.sleep"),
        ):
            backend._start_vllm_server(config, [0, 1])

        cmd = mock_popen.call_args[0][0]
        assert "--enable-prefix-caching" in cmd
        idx = cmd.index("--enable-prefix-caching")
        assert cmd[idx + 1] == "yes"

    def test_enable_prefix_caching_omitted_when_false(self):
        config = _make_local_config(
            {"infra": {"vllm": {"enable_prefix_caching": False}}}
        )
        backend = LocalBackend()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with (
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch("time.sleep"),
        ):
            backend._start_vllm_server(config, [0, 1])

        cmd = mock_popen.call_args[0][0]
        assert "--enable-prefix-caching" not in cmd

    def test_launches_via_vllm_launcher_not_trl_cli(self):
        """The local backend must route vLLM through vllm_launcher.py.

        Calling ``trl vllm-serve`` directly skips spawn-safety setup and
        hits "Cannot re-initialize CUDA in forked subprocess" on H100/H200.
        """
        config = _make_local_config()
        backend = LocalBackend()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with (
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch("time.sleep"),
        ):
            backend._start_vllm_server(config, [0, 1])

        cmd = mock_popen.call_args[0][0]
        # First two args must be `<python> -m foresightrlcore.runtime.scripts.vllm_launcher`
        assert cmd[1] == "-m"
        assert cmd[2] == "foresightrlcore.runtime.scripts.vllm_launcher"
        # `trl` should not appear anywhere in the command
        assert "trl" not in cmd
        assert "vllm-serve" not in cmd

    def test_patch_env_vars_set_when_yaml_opts_in(self):
        """``infra.vllm.{reasoning_config,compilation_config,disable_custom_all_reduce}``
        must reach the launcher via env vars."""
        config = _make_local_config(
            {
                "infra": {
                    "vllm": {
                        "reasoning_parser": "qwen3",
                        "reasoning_config": {
                            "reasoning_start_str": "<think>",
                            "reasoning_end_str": "</think>",
                        },
                        "compilation_config": {"cudagraph_capture_sizes": [1, 2, 4]},
                        "disable_custom_all_reduce": True,
                    }
                }
            }
        )
        backend = LocalBackend()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with (
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch("time.sleep"),
        ):
            backend._start_vllm_server(config, [0, 1])

        import json

        env = mock_popen.call_args.kwargs["env"]
        # JSON key order isn't guaranteed across YAML loaders, so compare
        # parsed dicts rather than raw strings.
        assert json.loads(env["VLLM_REASONING_CONFIG_JSON"]) == {
            "reasoning_start_str": "<think>",
            "reasoning_end_str": "</think>",
        }
        assert json.loads(env["VLLM_COMPILATION_CONFIG_JSON"]) == {
            "cudagraph_capture_sizes": [1, 2, 4]
        }
        assert env["VLLM_DISABLE_CUSTOM_ALL_REDUCE"] == "1"

    def test_patch_env_vars_omitted_when_no_opt_in(self):
        """No YAML opt-ins → launcher env stays clean (no-op fast path)."""
        config = _make_local_config()
        backend = LocalBackend()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

        with (
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch("time.sleep"),
        ):
            backend._start_vllm_server(config, [0, 1])

        env = mock_popen.call_args.kwargs["env"]
        assert "VLLM_REASONING_CONFIG_JSON" not in env
        assert "VLLM_COMPILATION_CONFIG_JSON" not in env
        assert "VLLM_DISABLE_CUSTOM_ALL_REDUCE" not in env


class TestStopVllmServer:
    def test_graceful_stop(self):
        backend = LocalBackend()
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.wait.return_value = 0

        with patch("os.killpg") as mock_killpg, patch("os.getpgid", return_value=12345):
            backend._stop_vllm_server(mock_proc)

        mock_killpg.assert_called_with(12345, signal.SIGTERM)

    def test_force_kill_on_timeout(self):
        backend = LocalBackend()
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="proc", timeout=15)

        with patch("os.killpg") as mock_killpg, patch("os.getpgid", return_value=12345):
            backend._stop_vllm_server(mock_proc)

        calls = mock_killpg.call_args_list
        assert any(c == call(12345, signal.SIGTERM) for c in calls)
        assert any(c == call(12345, signal.SIGKILL) for c in calls)

    def test_process_already_gone(self):
        backend = LocalBackend()
        mock_proc = MagicMock()
        mock_proc.pid = 99999

        with (
            patch("os.killpg", side_effect=ProcessLookupError),
            patch("os.getpgid", return_value=99999),
        ):
            backend._stop_vllm_server(mock_proc)


class TestRunTraining:
    def test_accelerate_command(self, tmp_path):
        config = _make_local_config()
        backend = LocalBackend()
        output_dir = str(tmp_path / "out")

        mock_result = MagicMock()
        mock_result.returncode = 0

        with (
            patch("subprocess.run", return_value=mock_result) as mock_run,
            patch(PATCH_ENSURE_DATA),
        ):
            backend._run_training(
                config=config,
                train_gpus=[2, 3, 4, 5, 6, 7],
                config_path="/path/config.yaml",
                output_dir=output_dir,
                method_type="grpo",
                vllm_mode="server",
            )

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "accelerate"
        assert "--num_processes" in cmd
        assert "6" in cmd

        env = mock_run.call_args[1]["env"]
        assert env["SM_CHECKPOINT_DIR"] == os.path.abspath(
            os.path.join(output_dir, "checkpoints")
        )
        assert env["SM_MODEL_DIR"] == os.path.abspath(os.path.join(output_dir, "model"))
        assert env["SM_CHANNEL_TRAIN"] == os.path.abspath(
            os.path.join(output_dir, "data", "train")
        )

    def test_torchrun_command(self, tmp_path):
        config = _make_local_config()
        backend = LocalBackend()
        output_dir = str(tmp_path / "out")

        mock_result = MagicMock()
        mock_result.returncode = 0

        with (
            patch("subprocess.run", return_value=mock_result) as mock_run,
            patch(PATCH_ENSURE_DATA),
        ):
            backend._run_training(
                config=config,
                train_gpus=[0, 1, 2, 3],
                config_path="/path/config.yaml",
                output_dir=output_dir,
                method_type="grpo",
                vllm_mode="none",
            )

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "torchrun"
        assert "--nproc_per_node=4" in cmd

        env = mock_run.call_args[1]["env"]
        assert "SM_CHECKPOINT_DIR" in env
        assert "SM_MODEL_DIR" in env
        assert "SM_CHANNEL_TRAIN" in env


class TestEnsureLocalData:
    def test_s3_download(self, tmp_path):
        backend = LocalBackend()
        local_dir = str(tmp_path / "train")
        os.makedirs(local_dir)

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            backend._ensure_local_data("s3://bucket/data/train.jsonl", local_dir)

        cmd = mock_run.call_args[0][0]
        assert cmd == [
            "aws",
            "s3",
            "cp",
            "s3://bucket/data/train.jsonl",
            local_dir + "/",
        ]

    def test_s3_download_failure_raises(self, tmp_path):
        backend = LocalBackend()
        local_dir = str(tmp_path / "train")
        os.makedirs(local_dir)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "NoSuchKey"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="Failed to download"):
                backend._ensure_local_data("s3://bucket/missing.jsonl", local_dir)

    def test_skips_if_data_already_present(self, tmp_path):
        backend = LocalBackend()
        local_dir = str(tmp_path / "train")
        os.makedirs(local_dir)
        (tmp_path / "train" / "data.jsonl").write_text('{"text": "hello"}')

        with patch("subprocess.run") as mock_run:
            backend._ensure_local_data("s3://bucket/data.jsonl", local_dir)

        mock_run.assert_not_called()

    def test_local_file_symlinked(self, tmp_path):
        backend = LocalBackend()
        src_file = tmp_path / "source" / "data.jsonl"
        src_file.parent.mkdir()
        src_file.write_text('{"text": "hello"}')

        local_dir = str(tmp_path / "train")
        os.makedirs(local_dir)

        backend._ensure_local_data(str(src_file), local_dir)

        dst = tmp_path / "train" / "data.jsonl"
        assert dst.is_symlink()
        assert dst.read_text() == '{"text": "hello"}'

    def test_local_dir_symlinked(self, tmp_path):
        backend = LocalBackend()
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        (src_dir / "a.jsonl").write_text('{"a": 1}')
        (src_dir / "b.json").write_text('{"b": 2}')
        (src_dir / "readme.txt").write_text("ignore me")

        local_dir = str(tmp_path / "train")
        os.makedirs(local_dir)

        backend._ensure_local_data(str(src_dir), local_dir)

        assert (tmp_path / "train" / "a.jsonl").is_symlink()
        assert (tmp_path / "train" / "b.json").is_symlink()
        assert not (tmp_path / "train" / "readme.txt").exists()

    def test_missing_local_path_raises(self, tmp_path):
        backend = LocalBackend()
        local_dir = str(tmp_path / "train")
        os.makedirs(local_dir)

        with pytest.raises(FileNotFoundError, match="not found"):
            backend._ensure_local_data("/nonexistent/path", local_dir)
