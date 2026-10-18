"""JSONL shard reading and packed Parquet + metadata sidecar writing."""

from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator

from foresightrlcore.tokenizer.pack import PackedRow

logger = logging.getLogger(__name__)


def read_jsonl_shards(path: str) -> Iterator[dict[str, Any]]:
    """Yield rows from a JSONL file, a directory of ``*.jsonl`` shards, or a
    glob pattern — same resolution rule as
    :func:`foresightrlcore.methods.sft_verl._glob_data_files`.
    """
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    elif any(ch in path for ch in "*?["):
        files = sorted(glob.glob(path))
    else:
        files = [path]

    if not files:
        raise FileNotFoundError(f"No .jsonl files found under {path!r}")

    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def write_packed_parquet(packed_rows: list[PackedRow], output_path: str) -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "input_ids": [p.input_ids for p in packed_rows],
            "loss_mask": [p.loss_mask for p in packed_rows],
            "seq_start_id": [p.seq_start_id for p in packed_rows],
        }
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)


def write_metadata(metadata: Any, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(json.dumps(metadata) + "\n")


_MODEL_WEIGHT_EXTENSIONS = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".h5",
    ".msgpack",
    ".gguf",
)


def download_dir_from_s3(
    s3_uri: str,
    local_dir: str,
    *,
    exclude_extensions: tuple[str, ...] = (),
    max_file_size_bytes: int | None = None,
) -> None:
    """Download top-level files under an s3:// prefix into local_dir.

    Nested keys are skipped defensively rather than silently flattened —
    both current callers (a JSONL data prefix, a tokenizer prefix) are
    flat file sets by convention.

    ``exclude_extensions``/``max_file_size_bytes`` are opt-in (both no-ops
    by default, preserving "download everything" for a genuine data
    prefix) — :func:`foresightrlcore.tokenizer.cli._resolve_tokenizer_path`
    passes both, because a model's S3 prefix commonly holds its tokenizer
    files (tokenizer.json, a few KB-MB) flat alongside its full weight
    shards (model-*.safetensors, tens of GB) — the SAME prefix is exactly
    what model.name/--tokenizer already point at for a pre-staged model
    (see CLAUDE.md "Pre-Staged Models"). Reproduced directly: pointing
    --tokenizer at such a prefix downloaded 55GB of Muse Glimmer 30B's
    safetensors shards just to reach its 27MB tokenizer.json, before this
    filter existed.
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got: {s3_uri!r}")

    import boto3

    bucket, _, prefix = s3_uri[len("s3://") :].partition("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    s3 = boto3.client("s3")
    os.makedirs(local_dir, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    found_any = False
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            name = key[len(prefix) :]
            if not name or "/" in name:
                continue
            if exclude_extensions and name.lower().endswith(exclude_extensions):
                logger.info(
                    f"Skipping {name} ({obj['Size']} bytes) — matches an "
                    "excluded extension, not needed here"
                )
                continue
            if max_file_size_bytes is not None and obj["Size"] > max_file_size_bytes:
                logger.warning(
                    f"Skipping {name} ({obj['Size']} bytes) — larger than "
                    f"the {max_file_size_bytes} byte cap for this download; "
                    "investigate if it's actually required"
                )
                continue
            found_any = True
            s3.download_file(bucket, key, os.path.join(local_dir, name))

    if not found_any:
        raise FileNotFoundError(f"No files found under {s3_uri!r}")


def upload_dir_to_s3(local_dir: str, s3_uri: str) -> None:
    """Upload every file in ``local_dir`` (non-recursive — tokenize-pack's
    output is a flat directory of Parquet + one metadata sidecar) to
    ``s3_uri``. Plain, user-supplied destination — no bucket/prefix is
    hardcoded anywhere in this module or its caller.
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"--upload-to must be an s3:// URI, got: {s3_uri!r}")
    if not s3_uri.endswith("/"):
        raise ValueError(f"--upload-to must end with '/', got: {s3_uri!r}")

    import boto3

    bucket, _, prefix = s3_uri[len("s3://") :].partition("/")
    s3 = boto3.client("s3")

    for name in sorted(os.listdir(local_dir)):
        local_path = os.path.join(local_dir, name)
        if not os.path.isfile(local_path):
            continue
        key = prefix + name
        s3.upload_file(local_path, bucket, key)
