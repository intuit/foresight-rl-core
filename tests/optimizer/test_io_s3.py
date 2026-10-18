"""S3 transfer helpers + dataset-path resolution (aws CLI subprocess mocked)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from foresightrlcore.common import io as optim_io
from foresightrlcore.common.io import (
    CheckpointWriter,
    resolve_dataset_path,
    s3_download,
    s3_sync_dir,
    s3_upload,
)


def _ok():
    return MagicMock(returncode=0, stdout="", stderr="")


def _fail():
    return MagicMock(returncode=1, stdout="", stderr="access denied")


class TestS3Download:
    def test_success_invokes_aws_cli(self, tmp_path, monkeypatch):
        run = MagicMock(return_value=_ok())
        monkeypatch.setattr(optim_io.subprocess, "run", run)
        out = s3_download("s3://b/k.parquet", tmp_path / "sub" / "k.parquet")
        assert out == tmp_path / "sub" / "k.parquet"
        args = run.call_args[0][0]
        assert args[:3] == ["aws", "s3", "cp"]

    def test_failure_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(optim_io.subprocess, "run", MagicMock(return_value=_fail()))
        with pytest.raises(RuntimeError, match="aws s3 cp failed"):
            s3_download("s3://b/k", tmp_path / "k")


class TestS3Upload:
    def test_success(self, tmp_path, monkeypatch):
        run = MagicMock(return_value=_ok())
        monkeypatch.setattr(optim_io.subprocess, "run", run)
        s3_upload(tmp_path / "f", "s3://b/f")
        assert run.call_args[0][0][:3] == ["aws", "s3", "cp"]

    def test_failure_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(optim_io.subprocess, "run", MagicMock(return_value=_fail()))
        with pytest.raises(RuntimeError, match="aws s3 cp failed uploading"):
            s3_upload(tmp_path / "f", "s3://b/f")


class TestS3SyncDir:
    def test_success(self, tmp_path, monkeypatch):
        run = MagicMock(return_value=_ok())
        monkeypatch.setattr(optim_io.subprocess, "run", run)
        s3_sync_dir(tmp_path, "s3://b/prefix")
        assert run.call_args[0][0][:3] == ["aws", "s3", "sync"]

    def test_failure_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(optim_io.subprocess, "run", MagicMock(return_value=_fail()))
        with pytest.raises(RuntimeError, match="aws s3 sync failed"):
            s3_sync_dir(tmp_path, "s3://b/prefix")


class TestResolveDatasetPath:
    def test_s3_downloads(self, tmp_path, monkeypatch):
        captured = {}

        def fake_download(uri, local):
            captured["uri"] = uri
            return local

        monkeypatch.setattr(optim_io, "s3_download", fake_download)
        monkeypatch.setattr(optim_io, "s3_object_exists", lambda uri: True)
        out = resolve_dataset_path("s3://b/data.parquet", download_dir=tmp_path)
        assert captured["uri"] == "s3://b/data.parquet"
        assert out == tmp_path / "data.parquet"

    def test_s3_trailing_slash_syncs_the_whole_prefix(self, tmp_path, monkeypatch):
        # A trailing slash names a prefix, not an object, so every part file
        # under it has to come down — one GetObject would fetch nothing.
        monkeypatch.setattr(optim_io, "s3_download_prefix", lambda uri, local: local)
        out = resolve_dataset_path("s3://b/prefix/", download_dir=tmp_path)
        assert out.parent == tmp_path
        assert out.name.startswith("prefix-")

    def test_local_existing(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.write_text("x")
        assert resolve_dataset_path(str(p), download_dir=tmp_path) == p

    def test_local_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="dataset path not found"):
            resolve_dataset_path(str(tmp_path / "nope.parquet"), download_dir=tmp_path)


class TestCheckpointWriterSync:
    def test_sync_uploads_when_s3_configured(self, tmp_path, monkeypatch):
        upload = MagicMock()
        monkeypatch.setattr(optim_io, "s3_upload", upload)
        writer = CheckpointWriter(tmp_path / "ckpt.jsonl", s3_uri="s3://b/ckpt.jsonl")
        writer.append({"dataset": "d", "realm_id": "r"})
        writer.sync()
        upload.assert_called_once()

    def test_sync_swallows_upload_errors(self, tmp_path, monkeypatch):
        def boom(*_a, **_k):
            raise RuntimeError("network down")

        monkeypatch.setattr(optim_io, "s3_upload", boom)
        writer = CheckpointWriter(tmp_path / "ckpt.jsonl", s3_uri="s3://b/ckpt.jsonl")
        writer.sync()  # best-effort: must not raise

    def test_sync_is_silent_when_nothing_has_been_written(
        self, tmp_path, monkeypatch, caplog
    ):
        """Nothing to upload is not a failed upload.

        Callers sync on a row count or a timer and cannot always know whether
        a given writer has received anything yet. A long single-trial
        evaluate run logged a sync failure every few seconds for a results
        file that had no completed trial yet -- 113 warnings in one run,
        which is how a real sync failure comes to be overlooked.
        """
        upload = MagicMock()
        monkeypatch.setattr(optim_io, "s3_upload", upload)
        writer = CheckpointWriter(tmp_path / "ckpt.jsonl", s3_uri="s3://b/ckpt.jsonl")

        with caplog.at_level(logging.WARNING):
            writer.sync()

        upload.assert_not_called()
        assert "sync failed" not in caplog.text
