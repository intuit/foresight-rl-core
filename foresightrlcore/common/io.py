"""I/O helpers: S3 transfer (via aws CLI) + JSONL checkpoint read/write.

S3 access uses the ``aws`` CLI as a subprocess (matching ``execution/local.py``
and ``perf_test/runner.py``) rather than adding a boto3-transfer dependency to
the hot path. The checkpoint is a newline-delimited JSON file: one flattened row
per line. Resume reads the completed rows' resume-keys so already-evaluated
combinations are skipped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

logger = logging.getLogger(__name__)


def is_s3_uri(path: str) -> bool:
    return path.startswith("s3://")


def s3_download(s3_uri: str, local_path: str | Path) -> Path:
    """Download a single S3 object to ``local_path`` via ``aws s3 cp``."""
    local = Path(local_path)
    local.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", s3_uri, local)
    result = subprocess.run(
        ["aws", "s3", "cp", s3_uri, str(local)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"aws s3 cp failed for {s3_uri}: {result.stderr.strip() or result.stdout.strip()}"
        )
    return local


def s3_upload(local_path: str | Path, s3_uri: str) -> None:
    """Upload a single file to S3 via ``aws s3 cp``."""
    logger.info("Uploading %s -> %s", local_path, s3_uri)
    result = subprocess.run(
        ["aws", "s3", "cp", str(local_path), s3_uri],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"aws s3 cp failed uploading to {s3_uri}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def s3_sync_dir(local_dir: str | Path, s3_uri: str) -> None:
    """Sync a local directory to an S3 prefix via ``aws s3 sync``."""
    logger.info("Syncing %s -> %s", local_dir, s3_uri)
    result = subprocess.run(
        ["aws", "s3", "sync", str(local_dir), s3_uri],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"aws s3 sync failed to {s3_uri}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def s3_download_prefix(s3_uri: str, local_dir: str | Path) -> Path:
    """Mirror every object under an S3 prefix into ``local_dir``.

    ``--delete`` makes it a mirror rather than a merge: readers take every file
    in the directory, so a part file left over from an earlier sync would be
    scored as if it were still in the dataset.
    """
    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s/ -> %s", s3_uri.rstrip("/"), local)
    result = subprocess.run(
        ["aws", "s3", "sync", "--delete", s3_uri.rstrip("/") + "/", str(local)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"aws s3 sync failed for {s3_uri}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    if not any(local.iterdir()):
        raise FileNotFoundError(f"no objects found under {s3_uri}")
    return local


def s3_object_exists(s3_uri: str) -> bool:
    """True when ``s3_uri`` names an object, rather than a prefix or nothing.

    A 404 is the only answer that means "not an object". Every other failure --
    expired credentials, a policy that grants GetObject but withholds
    HeadObject, throttling -- is raised rather than reported as absence, because
    the caller's fallback is to read the path as a prefix, and doing that to a
    file reports a missing dataset for one that is sitting right there. 403 is
    included deliberately: S3 answers that both for an object you may not head
    and for a missing key when you may not list, so it cannot be read as either.
    """
    without_scheme = s3_uri[len("s3://") :]
    bucket, _, key = without_scheme.partition("/")
    if not key:
        return False
    result = subprocess.run(
        ["aws", "s3api", "head-object", "--bucket", bucket, "--key", key],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return True
    detail = (result.stderr or result.stdout or "").strip()
    if "(404)" in detail or "Not Found" in detail:
        return False
    # HeadObject responses carry no body, so an expired or missing token
    # arrives as a bare 400 with nothing naming the cause.
    hint = (
        " (a bare 400 from HeadObject usually means expired or missing AWS "
        "credentials)"
        if "(400)" in detail
        else ""
    )
    raise RuntimeError(
        f"could not determine whether {s3_uri} is an object: {detail}{hint}"
    )


def resolve_dataset_path(path: str, *, download_dir: str | Path) -> Path:
    """Return a local path for ``path``; download it first if it is an S3 URI.

    A prefix is as valid as a single object: anything written by Spark or by
    ``to_parquet`` with partitioning is a directory of part files, and a loader
    that only accepts one file cannot read its own dataset.

    Which one a URI names is asked of S3 rather than inferred from the spelling.
    A trailing slash is taken at face value as a prefix, but otherwise the name
    says nothing useful: object keys are routinely written without an extension
    (a key with no extension can still be a single parquet file), and reading
    those as directories would sync an empty prefix and report a missing
    dataset for a file that is sitting right there.
    """
    if is_s3_uri(path):
        name = path.rstrip("/").split("/")[-1]
        if not path.endswith("/") and s3_object_exists(path):
            return s3_download(path, Path(download_dir) / name)
        # Keyed by the whole URI, not just its last segment: two prefixes
        # ending .../train/data/ and .../eval/data/ would otherwise share a
        # directory, and the second sync would mirror over the first.
        digest = hashlib.sha256(path.rstrip("/").encode("utf-8")).hexdigest()[:8]
        local_name = f"{name or 'dataset'}-{digest}"
        return s3_download_prefix(path, Path(download_dir) / local_name)
    local = Path(path).expanduser()
    if not local.exists():
        raise FileNotFoundError(f"dataset path not found: {local}")
    return local


def resolve_prompts_path(path: str, *, download_dir: str | Path) -> Path:
    """Return a local path for a prompts YAML; download it if it is an S3 URI.

    Mirrors :func:`resolve_dataset_path` so prompt files can live in S3 next to
    the dataset (no local bundling / no code deploy to change prompts).
    """
    if is_s3_uri(path):
        filename = path.rstrip("/").split("/")[-1] or "prompts.yaml"
        return s3_download(path, Path(download_dir) / filename)
    local = Path(path).expanduser()
    if not local.exists():
        raise FileNotFoundError(f"prompts path not found: {local}")
    return local


class CheckpointWriter:
    """Append flattened result rows to a JSONL checkpoint file.

    Args:
        path: Local checkpoint file path.
        s3_uri: Optional S3 URI the checkpoint is periodically uploaded to.
    """

    def __init__(self, path: str | Path, s3_uri: str | None = None) -> None:
        self.path = Path(path)
        self.s3_uri = s3_uri
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def sync(self) -> None:
        """Upload the checkpoint file to S3 if an S3 URI is configured.

        A file that has not been written yet is not a failure to upload, it is
        nothing to upload. Callers sync on a timer or a row count and cannot
        always know whether a particular writer has received anything, so
        without this a long single-trial run logged a sync failure every few
        seconds for a results file that simply had no completed trial yet --
        113 warnings in one run, which is how a real sync failure would come
        to be overlooked.
        """
        if self.s3_uri and self.path.exists():
            try:
                s3_upload(self.path, self.s3_uri)
            except Exception as exc:  # noqa: BLE001 - checkpoint upload best-effort
                logger.warning("checkpoint S3 sync failed: %s", exc)


def read_checkpoint_rows(path: str | Path) -> List[Dict[str, Any]]:
    """Read all rows from a JSONL checkpoint file (empty list if absent)."""
    p = Path(path)
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("skipping malformed checkpoint line")
    return rows


def completed_resume_keys(rows: Iterable[Dict[str, Any]]) -> Set[str]:
    """Compute the set of resume-keys from previously checkpointed rows."""
    keys: Set[str] = set()
    for row in rows:
        try:
            keys.add("{dataset}|{realm_id}|{epoch_date}|{model}|{mode}".format(**row))
        except (KeyError, IndexError):
            continue
    return keys


__all__ = [
    "is_s3_uri",
    "s3_download",
    "s3_upload",
    "s3_sync_dir",
    "resolve_dataset_path",
    "resolve_prompts_path",
    "CheckpointWriter",
    "read_checkpoint_rows",
    "completed_resume_keys",
]
