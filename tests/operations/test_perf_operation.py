"""Tests for PerfTestOperation -- registry wiring + dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import yaml

from foresightrlcore.config.perf_schema import PerfTestJobConfig
from foresightrlcore.operations.perf_test import PerfTestOperation


def _make_config(tmp_path, **patch_dict) -> PerfTestJobConfig:
    base = {
        "job": {"name": "perf-test"},
        "model": {"name": "s3://bucket/model/"},
        "perf_test": {
            "name": "run1",
            "serving": {"tensor_parallel_size": 1},
            "gatling": {"data_s3_path": "s3://bucket/data/"},
        },
        "execution": {"type": "aws_training_job"},
        "output": {"s3_path": "s3://bucket/out/"},
        "container": {"recipe": "qwen3_5"},
    }
    for k, v in patch_dict.items():
        base[k] = v
    p = tmp_path / "c.yaml"
    p.write_text(yaml.dump(base))
    return PerfTestJobConfig.from_yaml(p)


class TestPerfTestOperationAttributes:
    def test_name(self):
        assert PerfTestOperation.name == "perf_test"

    def test_registered_in_registry(self):
        import foresightrlcore.operations.perf_test  # noqa: F401
        from foresightrlcore.operations import get_operation

        cls = get_operation("perf_test")
        assert cls is PerfTestOperation


class TestPerfTestOperationRun:
    def test_invalid_config_raises(self, tmp_path):
        cfg = _make_config(tmp_path, container={})
        op = PerfTestOperation()
        with pytest.raises(ValueError, match="validation failed"):
            op.run(cfg)

    def test_aws_backend_calls_launcher(self, tmp_path):
        cfg = _make_config(tmp_path)
        op = PerfTestOperation()

        with patch(
            "foresightrlcore.execution.perf_test_launcher.PerfTestSageMakerLauncher"
        ) as mock_launcher_cls:
            mock_launcher = MagicMock()
            mock_launcher.launch.return_value = "job-1"
            mock_launcher_cls.return_value = mock_launcher

            ids = op.run(cfg, wait=False)

            mock_launcher_cls.assert_called_once_with(config=cfg)
            mock_launcher.launch.assert_called_once_with(wait=False)
            assert ids == ["job-1"]

    def test_unsupported_backend_raises(self, tmp_path):
        # The schema enforces aws_training_job | local in model_validator, so
        # bypass it by mutating the parsed config.
        cfg = _make_config(tmp_path)
        cfg.execution.type = "aws_serverless"
        op = PerfTestOperation()
        with pytest.raises(ValueError, match="aws_serverless"):
            op.run(cfg)


class TestPerfTestOperationLocalBackend:
    """Cover the ``_run_local`` path.

    The local backend dispatches to ``PerfTestRunner.execute`` after wiring
    up output and data directories. We mock the runner so the test stays
    GPU-free; the unit under test is the wiring, not the runner itself.
    """

    def _local_config(self, tmp_path) -> PerfTestJobConfig:
        # data_dir is required for local; gpus is required by perf_validation.
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        cfg = _make_config(
            tmp_path,
            execution={
                "type": "local",
                "local": {
                    "data_dir": str(data_dir),
                    "output_dir": str(tmp_path / "out"),
                    "gpus": "0",
                },
            },
        )
        return cfg

    def test_local_invokes_runner_with_wired_paths(self, tmp_path):
        cfg = self._local_config(tmp_path)
        op = PerfTestOperation()

        with patch(
            "foresightrlcore.perf_test.runner.PerfTestRunner"
        ) as mock_runner_cls:
            mock_runner = MagicMock()
            mock_runner_cls.return_value = mock_runner
            ids = op.run(cfg)

        assert ids == [cfg.perf_test.name]
        mock_runner_cls.assert_called_once()
        kwargs = mock_runner_cls.call_args.kwargs
        assert kwargs["model_path"] == cfg.model.name
        assert kwargs["data_channel"] == str(tmp_path / "data")
        assert kwargs["output_dir"] == tmp_path / "out" / cfg.perf_test.name
        # The output subdirectory should have been created.
        assert (tmp_path / "out" / cfg.perf_test.name).is_dir()
        mock_runner.execute.assert_called_once()

    def test_local_missing_data_dir_raises(self, tmp_path):
        # If execution.local has no data_dir, _run_local raises before any
        # runner work is done.
        cfg = _make_config(
            tmp_path,
            execution={
                "type": "local",
                "local": {"output_dir": str(tmp_path / "out"), "gpus": "0"},
            },
        )
        op = PerfTestOperation()
        with pytest.raises(ValueError, match="data_dir"):
            op.run(cfg)

    def test_local_falls_back_to_output_s3_path(self, tmp_path):
        # When local.output_dir is unset, the operation should fall back to
        # output.s3_path (treated as a filesystem path).
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        fallback_out = tmp_path / "fallback_out"
        cfg = _make_config(
            tmp_path,
            execution={
                "type": "local",
                "local": {"data_dir": str(data_dir), "gpus": "0"},
            },
            output={"s3_path": str(fallback_out)},
        )
        op = PerfTestOperation()

        with patch(
            "foresightrlcore.perf_test.runner.PerfTestRunner"
        ) as mock_runner_cls:
            mock_runner_cls.return_value = MagicMock()
            op.run(cfg)

        kwargs = mock_runner_cls.call_args.kwargs
        assert kwargs["output_dir"] == fallback_out / cfg.perf_test.name
