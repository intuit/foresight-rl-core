"""Checkpoint JSONL round-trip + resume-key computation + S3 URI helpers."""

from __future__ import annotations

import pytest

from foresightrlcore.common import io as optim_io
from foresightrlcore.common.io import (
    CheckpointWriter,
    completed_resume_keys,
    is_s3_uri,
    read_checkpoint_rows,
    resolve_prompts_path,
)


def _row(realm_id: str, model: str = "m", mode: str = "summary_state") -> dict:
    return {
        "dataset": "ds",
        "realm_id": realm_id,
        "epoch_date": "2024-01-01",
        "model": model,
        "mode": mode,
        "primary_metric_value": 4.0,
    }


def test_is_s3_uri():
    assert is_s3_uri("s3://bucket/key")
    assert not is_s3_uri("/local/path")
    assert not is_s3_uri("relative/path.parquet")


def test_checkpoint_append_and_read(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    writer = CheckpointWriter(path)
    writer.append(_row("r1"))
    writer.append(_row("r2"))

    rows = read_checkpoint_rows(path)
    assert len(rows) == 2
    assert {r["realm_id"] for r in rows} == {"r1", "r2"}


def test_read_missing_checkpoint_returns_empty(tmp_path):
    assert read_checkpoint_rows(tmp_path / "nope.jsonl") == []


def test_read_skips_malformed_lines(tmp_path):
    path = tmp_path / "ckpt.jsonl"
    path.write_text('{"dataset":"ds","realm_id":"r1"}\nnot-json\n\n')
    rows = read_checkpoint_rows(path)
    assert len(rows) == 1
    assert rows[0]["realm_id"] == "r1"


def test_completed_resume_keys():
    rows = [_row("r1"), _row("r2", model="m2")]
    keys = completed_resume_keys(rows)
    assert "ds|r1|2024-01-01|m|summary_state" in keys
    assert "ds|r2|2024-01-01|m2|summary_state" in keys
    assert len(keys) == 2


def test_completed_resume_keys_skips_incomplete_rows():
    keys = completed_resume_keys([{"dataset": "ds", "realm_id": "r1"}])
    assert keys == set()


def test_sync_noop_without_s3_uri(tmp_path):
    writer = CheckpointWriter(tmp_path / "ckpt.jsonl")
    writer.append(_row("r1"))
    # No s3_uri configured -> sync is a no-op and must not raise.
    writer.sync()


def test_resolve_prompts_path_local(tmp_path):
    p = tmp_path / "prompts.yaml"
    p.write_text("section: {key: hi}\n")
    assert resolve_prompts_path(str(p), download_dir=tmp_path) == p


def test_resolve_prompts_path_local_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_prompts_path(str(tmp_path / "nope.yaml"), download_dir=tmp_path)


def test_resolve_prompts_path_s3_downloads(tmp_path, monkeypatch):
    captured = {}

    def fake_download(s3_uri, local_path):
        captured["s3_uri"] = s3_uri
        captured["local"] = local_path
        return local_path

    monkeypatch.setattr(optim_io, "s3_download", fake_download)
    out = resolve_prompts_path(
        "s3://bucket/prompts/bts_v1_prompts.yaml", download_dir=tmp_path
    )
    assert captured["s3_uri"] == "s3://bucket/prompts/bts_v1_prompts.yaml"
    assert out == tmp_path / "bts_v1_prompts.yaml"
