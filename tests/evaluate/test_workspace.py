"""Source staging and per-trial workspaces."""

from __future__ import annotations

import subprocess
from unittest import mock

import pytest

from foresightrlcore.config.evaluate_schema import BenchmarkSource
from foresightrlcore.evaluate.workspace import WorkspaceError, WorkspaceManager


def test_a_local_source_is_copied_and_reused(tmp_path):
    source = tmp_path / "bench"
    (source / "pkg").mkdir(parents=True)
    (source / "pkg" / "run.py").write_text("print('hi')")

    with WorkspaceManager(tmp_path / "root") as manager:
        first = manager.stage_source("b", BenchmarkSource(path=str(source)))
        second = manager.stage_source("b", BenchmarkSource(path=str(source)))

        assert (first / "pkg" / "run.py").exists()
        # Cached: a four-model sweep must not re-fetch per model.
        assert first == second


def test_a_source_subdir_becomes_the_root(tmp_path):
    source = tmp_path / "repo"
    (source / "benchmarks" / "mine").mkdir(parents=True)
    (source / "benchmarks" / "mine" / "marker").write_text("x")

    with WorkspaceManager(tmp_path / "root") as manager:
        root = manager.stage_source(
            "b", BenchmarkSource(path=str(source), subdir="benchmarks/mine")
        )

    assert (root / "marker").exists()


def test_a_missing_subdir_is_reported(tmp_path):
    source = tmp_path / "repo"
    source.mkdir()

    with WorkspaceManager(tmp_path / "root") as manager:
        with pytest.raises(WorkspaceError, match="subdir"):
            manager.stage_source("b", BenchmarkSource(path=str(source), subdir="gone"))


def test_a_missing_local_source_is_reported(tmp_path):
    with WorkspaceManager(tmp_path / "root") as manager:
        with pytest.raises(WorkspaceError, match="not a directory"):
            manager.stage_source("b", BenchmarkSource(path=str(tmp_path / "nope")))


def test_a_source_sets_exactly_one_location():
    with pytest.raises(ValueError, match="exactly one"):
        BenchmarkSource(git="https://example.com/x", s3="s3://b/p/")
    with pytest.raises(ValueError, match="exactly one"):
        BenchmarkSource()


def test_git_falls_back_to_a_full_clone_for_a_commit_sha(tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        # A shallow --branch clone fails for a raw SHA; the full clone works.
        code = 1 if "--depth" in cmd else 0
        if code == 0 and cmd[:2] == ["git", "clone"]:
            from pathlib import Path

            Path(cmd[-1]).mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(cmd, code, "", "no such ref")

    with WorkspaceManager(tmp_path / "root") as manager:
        with mock.patch("subprocess.run", side_effect=fake_run):
            manager.stage_source(
                "b", BenchmarkSource(git="https://example.com/x", ref="abc123")
            )

    assert any("--depth" in c for c in calls)
    assert any("checkout" in c for c in calls)


def test_a_failing_fetch_carries_the_tools_own_stderr(tmp_path):
    with WorkspaceManager(tmp_path / "root") as manager:
        with mock.patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess([], 128, "", "fatal: not found"),
        ):
            with pytest.raises(WorkspaceError, match="fatal: not found"):
                manager.stage_source("b", BenchmarkSource(git="https://example.com/x"))


def test_each_trial_gets_its_own_empty_workspace(tmp_path):
    with WorkspaceManager(tmp_path / "root") as manager:
        first = manager.trial_workspace("bench", "model/a", 0)
        (first / "scratch.txt").write_text("mutated")

        second = manager.trial_workspace("bench", "model/a", 1)
        # Agentic benchmarks are graded on what they leave behind, so trials
        # must not share state.
        assert second != first
        assert list(second.iterdir()) == []

        # Re-requesting a trial resets it rather than reusing dirty state.
        again = manager.trial_workspace("bench", "model/a", 0)
        assert list(again.iterdir()) == []


def test_a_workspace_can_be_seeded_with_scenario_inputs(tmp_path):
    seed = tmp_path / "inputs"
    seed.mkdir()
    (seed / "ledger.csv").write_text("a,b\n")

    with WorkspaceManager(tmp_path / "root") as manager:
        workspace = manager.trial_workspace("b", "m", 0, seed_from=seed)

    assert (workspace / "ledger.csv").exists()


def test_seeding_from_a_missing_directory_is_reported(tmp_path):
    with WorkspaceManager(tmp_path / "root") as manager:
        with pytest.raises(WorkspaceError, match="seed directory"):
            manager.trial_workspace("b", "m", 0, seed_from=tmp_path / "gone")


def test_an_isolated_interpreter_is_created_once(tmp_path):
    with WorkspaceManager(tmp_path / "root") as manager:
        with (
            mock.patch("foresightrlcore.evaluate.workspace._run") as run,
            mock.patch("pathlib.Path.exists", return_value=True),
        ):
            first = manager.ensure_venv("b", ["pytest"])
            second = manager.ensure_venv("b", ["pytest"])

    assert first == second
    # One venv creation + one pip install, not two of each.
    assert run.call_count == 2


def test_keep_leaves_the_tree_behind_for_inspection(tmp_path):
    manager = WorkspaceManager(tmp_path / "root", keep=True)
    manager.trial_workspace("b", "m", 0)
    manager.close()

    assert (tmp_path / "root").exists()


def test_a_temporary_root_is_removed_on_close():
    manager = WorkspaceManager()
    root = manager.root
    manager.close()

    assert not root.exists()
