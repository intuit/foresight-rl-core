"""The delphi-bench harness shim: what it asks the CLI for, and what it reads back.

The CLI itself is not run here -- it lives in another repository and needs a
network. What is worth guarding is the translation on either side of it, and in
particular the ``agent:`` model prefix, which is the single thing that makes an
unmodified clone talk to our endpoint instead of authenticating to Intuit's LLM
execution service and ignoring it.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, cast

import pytest

pytestmark = pytest.mark.requires_intuit


@pytest.fixture(autouse=True)
def _extension_loaded():
    pytest.importorskip("foresightrl_intuit.evaluate")


def _ctx(tmp_path: Path, *, records: int = 2, source: bool = True):
    from foresightrlcore.evaluate.interfaces import EvalRecord, HarnessContext

    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)
    source_dir = tmp_path / "src"
    source_dir.mkdir(parents=True, exist_ok=True)

    # The shim never touches the endpoint object -- it reads the URL out of
    # ctx.env, exactly as an external process would read it out of its own
    # environment -- so there is nothing to stand up here.
    no_endpoint = cast(Any, None)

    return HarnessContext(
        benchmark="delphi",
        endpoint=no_endpoint,
        workspace=workspace,
        output_file=workspace / "evaluate_result.json",
        records=[
            EvalRecord(id=str(i), inputs={"question": f"q{i}"}, gold="True")
            for i in range(records)
        ],
        source_dir=source_dir if source else None,
        env={
            "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
            "EVALUATE_MODEL_NAME": "claude-haiku",
            "EVALUATE_PYTHON": "/venv/bin/python",
        },
    )


def _harness(**kwargs: Any):
    from foresightrl_intuit.evaluate.harnesses.delphi_bench import DelphiBenchHarness

    return DelphiBenchHarness(**kwargs)


class _FakeRun:
    """Stands in for ``subprocess.run``, recording argv and writing a result."""

    def __init__(self, summary: Dict[str, Any] | None, returncode: int = 0) -> None:
        self.summary = summary
        self.returncode = returncode
        self.argv: List[str] = []
        self.cwd: str = ""
        self.env: Dict[str, str] = {}

    def __call__(self, argv, **kwargs):
        self.argv = list(argv)
        self.cwd = kwargs.get("cwd", "")
        self.env = kwargs.get("env", {})
        if self.summary is not None:
            out = Path(argv[argv.index("--output-dir") + 1])
            target = out / "bench"
            target.mkdir(parents=True, exist_ok=True)
            (target / "model.json").write_text(
                json.dumps({"summary": self.summary}), encoding="utf-8"
            )
        return subprocess.CompletedProcess(argv, self.returncode, "out", "err")


def _flag(argv: List[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_the_model_is_prefixed_so_the_clone_uses_our_endpoint(tmp_path, monkeypatch):
    """Without ``agent:`` the CLI authenticates to GenOS and never calls us."""
    fake = _FakeRun({"total": 2, "correct": 2, "errors": 0})
    monkeypatch.setattr(subprocess, "run", fake)

    _harness().run(_ctx(tmp_path))

    assert _flag(fake.argv, "--model") == "agent:claude-haiku"
    assert _flag(fake.argv, "--agent-url") == "http://127.0.0.1:9/v1"


def test_it_runs_under_the_isolated_interpreter(tmp_path, monkeypatch):
    fake = _FakeRun({"total": 2, "correct": 1, "errors": 0})
    monkeypatch.setattr(subprocess, "run", fake)

    ctx = _ctx(tmp_path)
    _harness().run(ctx)

    assert fake.argv[0] == "/venv/bin/python"
    # The package is not installed into that venv, only its dependencies are.
    assert str(ctx.source_dir / "src") in fake.env["PYTHONPATH"]


def test_records_are_handed_over_in_the_two_columns_the_loader_demands(
    tmp_path, monkeypatch
):
    """The core dataset layer owns limit, sampling and S3; this just reshapes."""
    fake = _FakeRun({"total": 3, "correct": 3, "errors": 0})
    monkeypatch.setattr(subprocess, "run", fake)

    _harness().run(_ctx(tmp_path, records=3))

    rows = list(
        csv.DictReader(open(_flag(fake.argv, "--benchmarks"), encoding="utf-8"))
    )
    assert [r["user_prompt"] for r in rows] == ["q0", "q1", "q2"]
    assert {r["correct_answer"] for r in rows} == {"True"}


def test_the_score_is_a_fraction_not_a_percentage(tmp_path, monkeypatch):
    """The hub reports 0-100; every score in this framework is 0-1."""
    monkeypatch.setattr(
        subprocess, "run", _FakeRun({"total": 4, "correct": 3, "errors": 1})
    )

    outcome = _harness().run(_ctx(tmp_path))

    assert outcome.score == pytest.approx(0.75)
    assert outcome.n_samples == 4
    assert outcome.n_errors == 1
    assert not outcome.error


def test_a_confidence_interval_is_carried_through_when_the_cli_computes_one(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        subprocess,
        "run",
        _FakeRun(
            {
                "total": 10,
                "correct": 6,
                "errors": 0,
                "confidence_interval": {"lower": 0.3, "upper": 0.85},
            }
        ),
    )

    outcome = _harness().run(_ctx(tmp_path))

    assert outcome.sub_scores["ci_lower"] == pytest.approx(0.3)
    assert outcome.sub_scores["ci_upper"] == pytest.approx(0.85)


def test_a_failing_cli_reports_its_stdout(tmp_path, monkeypatch):
    """The CLI catches its own exceptions and prints them, so stdout is the clue."""
    monkeypatch.setattr(subprocess, "run", _FakeRun(None, returncode=1))

    outcome = _harness().run(_ctx(tmp_path))

    assert "exited with code 1" in outcome.error
    assert outcome.detail["output_tail"] == "out"


def test_a_silent_cli_is_a_failure_not_a_zero(tmp_path, monkeypatch):
    """Exit 0 with no result document must not be reported as a score of 0."""
    monkeypatch.setattr(subprocess, "run", _FakeRun(None))

    outcome = _harness().run(_ctx(tmp_path))

    assert "no result document" in outcome.error


def test_a_missing_source_tree_fails_before_anything_is_launched(tmp_path):
    outcome = _harness().run(_ctx(tmp_path, source=False))

    assert "requires source" in outcome.error


def test_the_catalog_entry_resolves_to_this_harness():
    """Guards the dotted path in the catalog against a rename here."""
    from foresightrl_intuit.evaluate import CATALOG_PATH

    from foresightrlcore.evaluate.benchmarks.plugin import import_target
    from foresightrlcore.evaluate.catalog import load_catalog

    entry = load_catalog([str(CATALOG_PATH)])["DELPHI_BENCH_IA_tax"]

    assert entry.type == "plugin"
    assert entry.isolate == "venv"
    target = import_target(entry.type_params()["target"])
    assert target.__name__ == "DelphiBenchHarness"
