"""Tests for the ``rl-train serverless`` subcommand group."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from foresightrlcore.cli import main
from foresightrlcore.serverless import registry


def _registered_lambda_names() -> list[str]:
    """Reward lambdas the installed extension registered.

    ``discover_tasks`` imports entry points. When the extension is only on
    ``PYTHONPATH``, the advice alias is registered but the module is not
    imported until something loads the task.
    """
    from foresightrlcore.tasks import load_task_module

    registry.discover_tasks()
    load_task_module("advice")
    names = [
        name
        for name, spec in registry.LAMBDAS.items()
        if spec.main_module.endswith(".lambda_reward")
    ]
    assert names, "the extension registered no reward lambda"
    return names


class TestServerlessList:
    def test_list_all_returns_zero(self, capsys):
        result = main(["serverless", "list", "all"])
        assert result == 0
        out = capsys.readouterr().out
        assert "[lambdas]" in out
        assert "[schemas]" in out

    @pytest.mark.requires_intuit
    def test_list_lambdas_contains_registered_names(self, capsys):
        names = _registered_lambda_names()
        result = main(["serverless", "list", "lambdas"])
        assert result == 0
        out = capsys.readouterr().out
        for name in names:
            assert name in out

    def test_list_schemas_marks_rlvr_implemented(self, capsys):
        result = main(["serverless", "list", "schemas"])
        assert result == 0
        out = capsys.readouterr().out
        assert "rlvr" in out
        assert "implemented" in out
        # Reserved keys must still appear:
        assert "sft" in out


class TestServerlessBundle:
    @pytest.mark.requires_intuit
    def test_bundle_advice_to_stdout(self, capsys):
        name = _registered_lambda_names()[0]
        result = main(["serverless", "bundle", "--name", name])
        assert result == 0
        out = capsys.readouterr().out
        assert "lambda_handler" in out
        assert "BUNDLER_STRIP" not in out

    @pytest.mark.requires_intuit
    def test_bundle_advice_to_file(self, tmp_path: Path):
        out_path = tmp_path / "bundle.py"
        name = _registered_lambda_names()[0]
        result = main(
            [
                "serverless",
                "bundle",
                "--name",
                name,
                "--output",
                str(out_path),
            ]
        )
        assert result == 0
        text = out_path.read_text()
        assert "lambda_handler" in text

    def test_bundle_unknown_name_fails(self):
        result = main(["serverless", "bundle", "--name", "no-such-lambda"])
        assert result == 1


class TestServerlessConvert:
    @pytest.mark.requires_intuit
    def test_convert_advice_jsonl(self, tmp_path: Path):
        src = tmp_path / "in.jsonl"
        dst = tmp_path / "out.jsonl"
        record = {
            "prompt": [{"role": "user", "content": "what should we do?"}],
            "state": "context: $1M revenue",
            "realmid": "abc",
        }
        src.write_text(json.dumps(record) + "\n")

        from foresightrlcore.tasks import load_task_module

        load_task_module("advice")
        result = main(
            [
                "serverless",
                "convert",
                "--input",
                str(src),
                "--output",
                str(dst),
                "--converter",
                "advice",
            ]
        )
        assert result == 0
        out_lines = dst.read_text().strip().splitlines()
        assert len(out_lines) == 1
        rec = json.loads(out_lines[0])
        assert rec["data_source"] == "advice"
        assert rec["reward_model"]["ground_truth"] == "context: $1M revenue"

    def test_convert_rejects_both_converter_and_mapping(self, tmp_path: Path, capsys):
        src = tmp_path / "in.jsonl"
        src.write_text("")
        dst = tmp_path / "out.jsonl"
        # argparse mutually exclusive group should error out:
        with pytest.raises(SystemExit):
            main(
                [
                    "serverless",
                    "convert",
                    "--input",
                    str(src),
                    "--output",
                    str(dst),
                    "--converter",
                    "advice",
                    "--mapping",
                    "x.yaml",
                ]
            )

    def test_convert_unknown_converter_fails(self, tmp_path: Path):
        src = tmp_path / "in.jsonl"
        src.write_text("{}\n")
        dst = tmp_path / "out.jsonl"
        result = main(
            [
                "serverless",
                "convert",
                "--input",
                str(src),
                "--output",
                str(dst),
                "--converter",
                "no-such-converter",
            ]
        )
        assert result == 1


class TestServerlessRegisterArgParsing:
    def test_register_with_unregistered_name_and_no_source_fails(self, capsys):
        # No actual AWS call should happen because we fail before
        # creating a session.
        result = main(
            [
                "serverless",
                "register",
                "--name",
                "totally-fake-not-registered",
                "--role",
                "arn:aws:iam::000000000000:role/fake",
            ]
        )
        assert result == 1
