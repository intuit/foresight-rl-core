import json
from unittest.mock import MagicMock, patch

import pytest

from foresightrlcore.tokenizer.io import (
    download_dir_from_s3,
    read_jsonl_shards,
    upload_dir_to_s3,
    write_metadata,
    write_packed_parquet,
)
from foresightrlcore.tokenizer.pack import PackedRow


def test_read_jsonl_shards_single_file(tmp_path):
    f = tmp_path / "data.jsonl"
    f.write_text('{"a": 1}\n{"a": 2}\n')
    rows = list(read_jsonl_shards(str(f)))
    assert rows == [{"a": 1}, {"a": 2}]


def test_read_jsonl_shards_directory_of_shards(tmp_path):
    (tmp_path / "shard0.jsonl").write_text('{"a": 1}\n')
    (tmp_path / "shard1.jsonl").write_text('{"a": 2}\n{"a": 3}\n')
    rows = list(read_jsonl_shards(str(tmp_path)))
    assert sorted(r["a"] for r in rows) == [1, 2, 3]


def test_read_jsonl_shards_skips_blank_lines(tmp_path):
    f = tmp_path / "data.jsonl"
    f.write_text('{"a": 1}\n\n{"a": 2}\n')
    rows = list(read_jsonl_shards(str(f)))
    assert len(rows) == 2


def test_read_jsonl_shards_raises_on_no_match(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(read_jsonl_shards(str(tmp_path / "missing.jsonl")))


def test_write_packed_parquet_round_trips(tmp_path):
    import pandas as pd

    p1 = PackedRow(input_ids=[1, 2, 3], loss_mask=[0, 1, 1], seq_start_id=[0])
    p2 = PackedRow(input_ids=[4, 5], loss_mask=[0, 0], seq_start_id=[0])
    out = tmp_path / "training_8192.idx.parquet"

    write_packed_parquet([p1, p2], str(out))
    df = pd.read_parquet(out)

    assert list(df.columns) == ["input_ids", "loss_mask", "seq_start_id"]
    assert list(df["input_ids"].iloc[0]) == [1, 2, 3]
    assert list(df["loss_mask"].iloc[1]) == [0, 0]
    assert list(df["seq_start_id"].iloc[0]) == [0]


def test_write_metadata_round_trips(tmp_path):
    out = tmp_path / "8192_metadata.jsonl"
    write_metadata([{"split": "training", "packing_factor": 2.5}], str(out))

    loaded = json.loads(out.read_text().splitlines()[0])
    assert loaded == [{"split": "training", "packing_factor": 2.5}]


def test_upload_dir_to_s3_uploads_every_file_with_the_right_key(tmp_path):
    (tmp_path / "training_8192.idx.parquet").write_bytes(b"x")
    (tmp_path / "8192_metadata.jsonl").write_text("{}")
    (tmp_path / "subdir").mkdir()  # non-recursive: must be skipped, not uploaded

    mock_client = MagicMock()
    with patch("boto3.client", return_value=mock_client) as mock_boto3_client:
        upload_dir_to_s3(str(tmp_path), "s3://my-bucket/some/prefix/")

    mock_boto3_client.assert_called_once_with("s3")
    calls = {c.args[1:] for c in mock_client.upload_file.call_args_list}
    assert calls == {
        ("my-bucket", "some/prefix/training_8192.idx.parquet"),
        ("my-bucket", "some/prefix/8192_metadata.jsonl"),
    }


def test_upload_dir_to_s3_rejects_non_s3_uri(tmp_path):
    with pytest.raises(ValueError, match="s3://"):
        upload_dir_to_s3(str(tmp_path), "/local/path/")


def test_upload_dir_to_s3_rejects_uri_without_trailing_slash(tmp_path):
    with pytest.raises(ValueError, match="end with"):
        upload_dir_to_s3(str(tmp_path), "s3://my-bucket/prefix")


def _mock_paginator(contents: list[dict]):
    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [{"Contents": contents}]
    return mock_paginator


def test_download_dir_from_s3_downloads_flat_files_only(tmp_path):
    mock_client = MagicMock()
    mock_client.get_paginator.return_value = _mock_paginator(
        [
            {"Key": "some/prefix/tokenizer.json"},
            {"Key": "some/prefix/tokenizer_config.json"},
            {"Key": "some/prefix/"},  # directory marker, must be skipped
            {"Key": "some/prefix/nested/foo.json"},  # nested, must be skipped
        ]
    )

    local_dir = tmp_path / "tok"
    with patch("boto3.client", return_value=mock_client) as mock_boto3_client:
        download_dir_from_s3("s3://my-bucket/some/prefix/", str(local_dir))

    mock_boto3_client.assert_called_once_with("s3")
    mock_client.get_paginator.assert_called_once_with("list_objects_v2")
    calls = {c.args for c in mock_client.download_file.call_args_list}
    assert calls == {
        (
            "my-bucket",
            "some/prefix/tokenizer.json",
            str(local_dir / "tokenizer.json"),
        ),
        (
            "my-bucket",
            "some/prefix/tokenizer_config.json",
            str(local_dir / "tokenizer_config.json"),
        ),
    }
    assert local_dir.is_dir()


def test_download_dir_from_s3_normalizes_prefix_without_trailing_slash(tmp_path):
    mock_client = MagicMock()
    mock_client.get_paginator.return_value = _mock_paginator(
        [{"Key": "some/prefix/tokenizer.json"}]
    )

    local_dir = tmp_path / "tok"
    with patch("boto3.client", return_value=mock_client):
        # No trailing slash on the prefix — must still resolve keys correctly.
        download_dir_from_s3("s3://my-bucket/some/prefix", str(local_dir))

    mock_client.get_paginator.return_value.paginate.assert_called_once_with(
        Bucket="my-bucket", Prefix="some/prefix/"
    )
    calls = {c.args for c in mock_client.download_file.call_args_list}
    assert calls == {
        ("my-bucket", "some/prefix/tokenizer.json", str(local_dir / "tokenizer.json"))
    }


def test_download_dir_from_s3_rejects_non_s3_uri(tmp_path):
    with pytest.raises(ValueError, match="s3://"):
        download_dir_from_s3("/local/path/", str(tmp_path))


def test_download_dir_from_s3_raises_when_no_files_found(tmp_path):
    mock_client = MagicMock()
    mock_client.get_paginator.return_value = _mock_paginator([])

    with patch("boto3.client", return_value=mock_client):
        with pytest.raises(FileNotFoundError, match="No files found"):
            download_dir_from_s3("s3://my-bucket/empty/", str(tmp_path / "tok"))


def test_download_dir_from_s3_excludes_model_weight_extensions(tmp_path):
    """Regression guard: a model's S3 prefix commonly holds its tokenizer

    files flat alongside its full weight shards (model.name/--tokenizer
    often point at the exact same prefix) — reproduced directly: this
    downloaded 55GB of Muse Glimmer 30B's safetensors shards just to
    reach a 27MB tokenizer.json, with no filtering at all.
    """
    mock_client = MagicMock()
    mock_client.get_paginator.return_value = _mock_paginator(
        [
            {"Key": "some/prefix/tokenizer.json", "Size": 1_000},
            {
                "Key": "some/prefix/model-00001-of-00002.safetensors",
                "Size": 50_000_000_000,
            },
            {"Key": "some/prefix/pytorch_model.bin", "Size": 20_000_000_000},
        ]
    )

    local_dir = tmp_path / "tok"
    with patch("boto3.client", return_value=mock_client):
        download_dir_from_s3(
            "s3://my-bucket/some/prefix/",
            str(local_dir),
            exclude_extensions=(".safetensors", ".bin"),
        )

    calls = {c.args for c in mock_client.download_file.call_args_list}
    assert calls == {
        (
            "my-bucket",
            "some/prefix/tokenizer.json",
            str(local_dir / "tokenizer.json"),
        )
    }


def test_download_dir_from_s3_enforces_max_file_size(tmp_path):
    mock_client = MagicMock()
    mock_client.get_paginator.return_value = _mock_paginator(
        [
            {"Key": "some/prefix/tokenizer.json", "Size": 1_000},
            {"Key": "some/prefix/unexpectedly_huge.json", "Size": 200_000_000},
        ]
    )

    local_dir = tmp_path / "tok"
    with patch("boto3.client", return_value=mock_client):
        download_dir_from_s3(
            "s3://my-bucket/some/prefix/",
            str(local_dir),
            max_file_size_bytes=64 * 1024 * 1024,
        )

    calls = {c.args for c in mock_client.download_file.call_args_list}
    assert calls == {
        (
            "my-bucket",
            "some/prefix/tokenizer.json",
            str(local_dir / "tokenizer.json"),
        )
    }


def test_download_dir_from_s3_defaults_download_everything(tmp_path):
    """exclude_extensions/max_file_size_bytes are opt-in — the plain

    2-argument call (used for a genuine JSONL data prefix, which can
    legitimately contain large files) must keep downloading everything,
    same as before either parameter existed.
    """
    mock_client = MagicMock()
    mock_client.get_paginator.return_value = _mock_paginator(
        [{"Key": "some/prefix/big_shard.jsonl", "Size": 5_000_000_000}]
    )

    local_dir = tmp_path / "data"
    with patch("boto3.client", return_value=mock_client):
        download_dir_from_s3("s3://my-bucket/some/prefix/", str(local_dir))

    calls = {c.args for c in mock_client.download_file.call_args_list}
    assert calls == {
        (
            "my-bucket",
            "some/prefix/big_shard.jsonl",
            str(local_dir / "big_shard.jsonl"),
        )
    }
