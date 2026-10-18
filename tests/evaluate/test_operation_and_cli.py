"""The operation, the public launch() API, and the CLI surface."""

from __future__ import annotations

import logging
from typing import Any, Dict
from unittest import mock

import pytest
import yaml

from foresightrlcore.config.evaluate_schema import EvaluateJobConfig
from foresightrlcore.evaluate import launch
from foresightrlcore.evaluate.cli import dispatch
from foresightrlcore.operations import get_operation
from foresightrlcore.operations.evaluate import EvaluateOperation

RUN = {
    "benchmarks": [
        {
            "name": "demo",
            "type": "mcq",
            "metrics": ["choice_match"],
            "dataset": {
                "loader": "inline",
                "records": [
                    {
                        "id": "q1",
                        "question": "Capital of France?",
                        "choices": ["Berlin", "Paris"],
                        "answer": "Paris",
                    }
                ],
            },
        }
    ],
    "models": [{"name": "echo-model", "provider": "echo"}],
    "reporters": ["json"],
}


def _payload(**overrides: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "job": {"name": "eval-demo"},
        "execution": {"type": "local"},
        "evaluate": RUN,
    }
    payload.update(overrides)
    return payload


def _write(tmp_path, **overrides) -> str:
    path = tmp_path / "evaluate.yaml"
    path.write_text(yaml.safe_dump(_payload(**overrides)))
    return str(path)


# ---------------------------------------------------------------------------
# Registration and dispatch
# ---------------------------------------------------------------------------


def test_evaluate_is_a_registered_operation():
    import foresightrlcore  # noqa: F401 - triggers registration

    assert get_operation("evaluate") is EvaluateOperation


def test_an_invalid_config_is_refused_before_anything_runs():
    config = EvaluateJobConfig.model_validate(
        _payload(
            evaluate={
                "benchmarks": [{"name": "not_in_the_catalog"}],
                "models": [{"name": "m"}],
            }
        )
    )

    with pytest.raises(ValueError, match="validation failed"):
        EvaluateOperation().run(config)


def test_an_aws_run_goes_to_the_sagemaker_launcher():
    config = EvaluateJobConfig.model_validate(
        _payload(
            execution={"type": "aws_training_job"},
            output={"s3_path": "s3://bucket/eval/"},
            container={"evaluate_image_uri": "img"},
        )
    )

    with mock.patch(
        "foresightrlcore.execution.evaluate_launcher.EvaluateSageMakerLauncher"
    ) as launcher:
        launcher.return_value.launch.return_value = "eval-demo-123"
        assert EvaluateOperation().run(config, wait=True) == ["eval-demo-123"]

    launcher.return_value.launch.assert_called_once_with(wait=True)


def test_a_local_run_executes_in_process(tmp_path):
    config = EvaluateJobConfig.model_validate(
        _payload(execution={"type": "local", "local": {"output_dir": str(tmp_path)}})
    )

    assert EvaluateOperation().run(config) == ["eval-demo"]
    assert (tmp_path / "evaluate_results.json").exists()


def test_an_s3_output_on_a_local_run_still_writes_to_disk():
    config = EvaluateJobConfig.model_validate(
        _payload(output={"s3_path": "s3://bucket/eval/"})
    )

    with mock.patch("foresightrlcore.evaluate.runner.run_evaluate") as run:
        EvaluateOperation().run(config)

    output_dir = run.call_args.kwargs["output_dir"]
    assert str(output_dir).startswith("/tmp/evaluate_output")


def test_config_environment_is_exported_for_a_local_run(monkeypatch):
    monkeypatch.delenv("GENOS_ENV", raising=False)
    config = EvaluateJobConfig.model_validate(
        _payload(environment={"GENOS_ENV": "e2e"})
    )

    with mock.patch("foresightrlcore.evaluate.runner.run_evaluate"):
        EvaluateOperation().run(config)

    import os

    assert os.environ["GENOS_ENV"] == "e2e"


# ---------------------------------------------------------------------------
# Resume / checkpointing
#
# A remote evaluate job runs for hours against paid endpoints, so whether its
# work survives an interruption is load-bearing. The trap these cover: the
# runner writes checkpoints whenever `checkpoint.enabled` is true (the
# default), but they only outlive the job if a checkpoint S3 prefix exists to
# attach a SageMaker CheckpointConfig to. When that was left to the operator,
# a config could look fully enabled, write checkpoints all run long to
# container disk, and have nothing to resume from -- discovered only after a
# 13-hour run ended.
# ---------------------------------------------------------------------------


def _aws(**overrides: Any) -> EvaluateJobConfig:
    settings: Dict[str, Any] = {
        "execution": {"type": "aws_training_job"},
        "output": {"s3_path": "s3://bucket/eval/"},
        "container": {"evaluate_image_uri": "img"},
    }
    settings.update(overrides)
    return EvaluateJobConfig.model_validate(_payload(**settings))


def test_a_remote_run_can_resume_without_being_asked_to():
    assert _aws().output.checkpoint_s3_path == "s3://bucket/eval/checkpoints/"


def test_an_explicit_checkpoint_location_is_left_alone():
    config = _aws(
        output={
            "s3_path": "s3://bucket/eval/",
            "checkpoint_s3_path": "s3://elsewhere/ck/",
        }
    )

    assert config.output.checkpoint_s3_path == "s3://elsewhere/ck/"


def test_disabling_checkpointing_does_not_quietly_re_enable_it():
    """`enabled: false` is how an operator forces a clean measurement."""
    run = dict(RUN, checkpoint={"enabled": False})

    assert _aws(evaluate=run).output.checkpoint_s3_path is None


def test_a_local_run_gets_no_s3_checkpoint_prefix():
    """Nothing to sync: the output is already a directory on disk."""
    config = EvaluateJobConfig.model_validate(
        _payload(output={"s3_path": "s3://bucket/eval/"})
    )

    assert config.output.checkpoint_s3_path is None


def test_validation_reports_where_a_resumable_run_keeps_its_checkpoints(caplog):
    from foresightrlcore.config.evaluate_validation import validate_evaluate_config

    with caplog.at_level(logging.INFO):
        assert validate_evaluate_config(_aws()) == []

    assert "s3://bucket/eval/checkpoints/" in caplog.text


def test_validation_warns_that_a_run_without_checkpointing_cannot_resume(caplog):
    from foresightrlcore.config.evaluate_validation import validate_evaluate_config

    config = _aws(evaluate=dict(RUN, checkpoint={"enabled": False}))

    with caplog.at_level(logging.WARNING):
        assert validate_evaluate_config(config) == []

    assert "cannot resume" in caplog.text


# ---------------------------------------------------------------------------
# The public API
# ---------------------------------------------------------------------------


def test_launch_reads_a_yaml_file_and_returns_job_ids(tmp_path):
    path = _write(
        tmp_path, execution={"type": "local", "local": {"output_dir": str(tmp_path)}}
    )

    assert launch(path) == ["eval-demo"]


def test_launch_applies_dotted_overrides(tmp_path):
    path = _write(tmp_path)

    with mock.patch(
        "foresightrlcore.operations.evaluate.EvaluateOperation.run", return_value=["x"]
    ) as run:
        overrides: Dict[str, Any] = {"evaluate.trials": 3}
        launch(path, **overrides)

    assert run.call_args.args[0].evaluate.trials == 3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _args(command, **kwargs):
    import argparse

    return argparse.Namespace(command=command, **kwargs)


def test_the_cli_validates_a_good_config(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    path = _write(tmp_path)

    assert dispatch(_args("validate-evaluate", config=path)) == 0
    assert "Evaluate config is valid" in caplog.text


def test_the_cli_reports_validation_errors_and_exits_nonzero(tmp_path, caplog):
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            _payload(
                evaluate={
                    "benchmarks": [{"name": "unknown_bench"}],
                    "models": [{"name": "m"}],
                }
            )
        )
    )

    assert dispatch(_args("validate-evaluate", config=str(path))) == 1
    assert "not in the catalog" in caplog.text


def test_a_missing_config_file_is_reported(caplog):
    assert dispatch(_args("validate-evaluate", config="/nope/evaluate.yaml")) == 1
    assert "not found" in caplog.text


def test_list_benchmarks_shows_the_catalog_and_every_plug_point(caplog):
    caplog.set_level(logging.INFO)

    assert dispatch(_args("list-benchmarks", catalog=[])) == 0

    assert "demo_mcq" in caplog.text
    assert "Benchmark types:" in caplog.text
    assert "Dataset loaders:" in caplog.text
    assert "Metrics:" in caplog.text
    assert "Reporters:" in caplog.text


def test_list_benchmarks_includes_an_extra_catalog(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    extra = tmp_path / "extra.yaml"
    extra.write_text(
        yaml.safe_dump({"benchmarks": [{"name": "my_bench", "type": "mcq"}]})
    )

    dispatch(_args("list-benchmarks", catalog=[str(extra)]))

    assert "my_bench" in caplog.text


def test_the_cli_runs_an_evaluation(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    path = _write(
        tmp_path, execution={"type": "local", "local": {"output_dir": str(tmp_path)}}
    )

    assert dispatch(_args("evaluate", config=path, wait=False)) == 0
    assert "Job ID: eval-demo" in caplog.text


def test_a_failing_run_exits_nonzero(tmp_path, caplog):
    path = _write(tmp_path)

    with mock.patch(
        "foresightrlcore.evaluate.launch", side_effect=RuntimeError("boom")
    ):
        assert dispatch(_args("evaluate", config=path, wait=False)) == 1

    assert "boom" in caplog.text


def test_an_unknown_subcommand_is_reported():
    assert dispatch(_args("evaluate-something-else")) == 1


def test_the_main_cli_exposes_every_evaluate_subcommand():
    from foresightrlcore.cli import main

    argv_by_command = {
        "evaluate": ["evaluate", "--config", "x.yaml"],
        "validate-evaluate": ["validate-evaluate", "--config", "x.yaml"],
        "list-benchmarks": ["list-benchmarks"],
    }

    for command, argv in argv_by_command.items():
        with mock.patch(
            "foresightrlcore.evaluate.cli.dispatch", return_value=0
        ) as dispatched:
            assert main(argv) == 0
            assert dispatched.call_args.args[0].command == command
