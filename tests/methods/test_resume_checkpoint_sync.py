"""Tests for _download_resume_checkpoint — single-sync-per-node behavior."""

from __future__ import annotations

import fcntl
import os
import threading
import time
from collections.abc import Generator
from unittest.mock import patch

import pytest

from foresightrlcore.methods.base import _download_resume_checkpoint

S3_URI = "s3://bucket/path/to/checkpoint-100"
SENTINEL = "/tmp/resume_checkpoint.done"
FAILURE = "/tmp/resume_checkpoint.failed"
_LOCK_PATH = "/tmp/resume_checkpoint.test.lock"


def _cleanup() -> None:
    for p in (SENTINEL, FAILURE):
        if os.path.exists(p):
            os.remove(p)


@pytest.fixture(autouse=True)
def _isolate() -> Generator[None, None, None]:
    """Serialize and isolate tests across parallel tox envs.

    The production code uses hardcoded /tmp/ paths, so parallel tox envs
    (py311/py312/py313) collide on the same files. An fcntl flock on a
    shared file serializes access across processes; _cleanup() then keeps
    each test's view of the sentinel/failure files isolated.
    """
    lock_fd = os.open(_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        _cleanup()
        yield
        _cleanup()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def test_missing_aws_cli_raises_runtime_error() -> None:
    with (patch("shutil.which", return_value=None),):
        with pytest.raises(RuntimeError, match="AWS CLI"):
            _download_resume_checkpoint(S3_URI)


def test_rank0_runs_aws_sync_and_writes_sentinel() -> None:
    with (
        patch.dict(os.environ, {"LOCAL_RANK": "0"}, clear=False),
        patch("shutil.which", return_value="/usr/bin/aws"),
        patch("subprocess.run") as mock_run,
    ):
        result = _download_resume_checkpoint(S3_URI)

    assert result == "/tmp/resume_checkpoint"
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert args[0] == ["aws", "s3", "sync", S3_URI, "/tmp/resume_checkpoint"]
    assert kwargs.get("check") is True
    assert os.path.exists(SENTINEL)


def test_non_rank0_does_not_call_aws_sync_and_waits() -> None:
    # rank0 has already finished — sentinel exists.
    with open(SENTINEL, "w") as f:
        f.write("ok")

    with (
        patch.dict(os.environ, {"LOCAL_RANK": "3"}, clear=False),
        patch("shutil.which", return_value="/usr/bin/aws"),
        patch("subprocess.run") as mock_run,
    ):
        result = _download_resume_checkpoint(S3_URI)

    assert result == "/tmp/resume_checkpoint"
    mock_run.assert_not_called()


def test_non_rank0_raises_when_rank0_fails() -> None:
    with open(FAILURE, "w") as f:
        f.write("boom")

    with (
        patch.dict(os.environ, {"LOCAL_RANK": "5"}, clear=False),
        patch("shutil.which", return_value="/usr/bin/aws"),
        patch("subprocess.run") as mock_run,
    ):
        with pytest.raises(RuntimeError, match="rank0 failed"):
            _download_resume_checkpoint(S3_URI)
    mock_run.assert_not_called()


def test_non_rank0_times_out() -> None:
    with (
        patch.dict(
            os.environ,
            {"LOCAL_RANK": "1", "CHECKPOINT_SYNC_TIMEOUT": "1"},
            clear=False,
        ),
        patch("shutil.which", return_value="/usr/bin/aws"),
        patch("subprocess.run") as mock_run,
    ):
        with pytest.raises(TimeoutError):
            _download_resume_checkpoint(S3_URI)
    mock_run.assert_not_called()


def test_rank0_writes_failure_marker_on_subprocess_error() -> None:
    import subprocess

    def _fail(*args, **kwargs):
        raise subprocess.CalledProcessError(255, args[0])

    with (
        patch.dict(os.environ, {"LOCAL_RANK": "0"}, clear=False),
        patch("shutil.which", return_value="/usr/bin/aws"),
        patch("subprocess.run", side_effect=_fail),
    ):
        with pytest.raises(subprocess.CalledProcessError):
            _download_resume_checkpoint(S3_URI)
    assert os.path.exists(FAILURE)
    assert not os.path.exists(SENTINEL)


def test_concurrent_ranks_only_rank0_calls_aws() -> None:
    """Simulate the real failure mode: 8 ranks calling concurrently.

    Only rank0 should invoke aws s3 sync. The other 7 ranks should block
    until rank0 writes the sentinel, then return.
    """
    call_count = {"n": 0}
    sync_started = threading.Event()

    def fake_sync(*args, **kwargs):
        call_count["n"] += 1
        sync_started.set()
        time.sleep(0.2)  # simulate download

    errors: list[BaseException] = []
    results: list[str] = []

    # os.environ is one process-wide dict shared by every thread here —
    # unittest.mock.patch.dict's "capture original, mutate, restore
    # original on __exit__" pattern is not thread-safe against concurrent
    # mutation of the *same* key: each thread's __exit__ can restore a
    # stale snapshot captured before a sibling thread's own mutation,
    # permanently leaking LOCAL_RANK/CHECKPOINT_SYNC_TIMEOUT into every
    # later test in the same pytest process (reproduced directly — this
    # leaked LOCAL_RANK into TestEnsureTransformersVersionCompatible's
    # own-rank-default tests, which then hung for the full 600s timeout
    # waiting on a sentinel nobody was going to write). Each thread here
    # only *sets* the two keys (never tries to restore a per-thread
    # snapshot); cleanup happens exactly once, after every thread has
    # joined, in the outer try/finally below.
    def worker(rank: int) -> None:
        try:
            os.environ["LOCAL_RANK"] = str(rank)
            os.environ["CHECKPOINT_SYNC_TIMEOUT"] = "10"
            results.append(_download_resume_checkpoint(S3_URI))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    try:
        with (
            patch("shutil.which", return_value="/usr/bin/aws"),
            patch("subprocess.run", side_effect=fake_sync),
        ):
            threads = [threading.Thread(target=worker, args=(r,)) for r in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)
    finally:
        os.environ.pop("LOCAL_RANK", None)
        os.environ.pop("CHECKPOINT_SYNC_TIMEOUT", None)

    assert not errors, f"unexpected errors: {errors}"
    assert len(results) == 8
    assert call_count["n"] == 1, "aws s3 sync must be called exactly once per node"
