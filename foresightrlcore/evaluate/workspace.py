"""Staging benchmark code and per-trial workspaces.

Two jobs, both of which exist so a benchmark author does not have to solve
them again:

**Source staging.** A benchmark that lives in its own repository or an S3
prefix is fetched once per run and reused across models and trials. Fetching is
cached by benchmark name, so a four-model sweep clones once.

**Trial workspaces.** Every ``(benchmark, model, trial)`` gets its own empty,
writable directory. Agentic benchmarks *mutate* their workspace -- that is what
they are graded on -- so sharing one between trials would silently corrupt the
measurement.

Optionally each benchmark gets an isolated virtualenv, so a benchmark's
dependencies can conflict with the job's without either breaking.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from foresightrlcore.config.evaluate_schema import BenchmarkSource

GIT_CLONE_TIMEOUT = 900
VENV_CREATE_TIMEOUT = 300
PIP_INSTALL_TIMEOUT = 1800


class WorkspaceError(RuntimeError):
    """Raised when staging fails; always carries the tool's own stderr."""


def _run(cmd: List[str], *, timeout: int, what: str) -> None:
    logger.info("%s: %s", what, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise WorkspaceError(
            f"{what} failed (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


class WorkspaceManager:
    """Owns every directory an evaluate run creates on disk.

    Args:
        root: Directory to create workspaces under. A temporary directory is
            used when omitted.
        keep: Leave staged trees and workspaces in place on close. Invaluable
            when a benchmark fails and you need to see what it left behind.
    """

    def __init__(self, root: Optional[Path] = None, *, keep: bool = False) -> None:
        self._owns_root = root is None
        self.root = Path(root) if root else Path(tempfile.mkdtemp(prefix="evaluate_"))
        self.keep = keep
        self.root.mkdir(parents=True, exist_ok=True)
        self._sources: Dict[str, Path] = {}
        self._venvs: Dict[str, Path] = {}

    # ------------------------------------------------------------------
    # Source staging
    # ------------------------------------------------------------------

    def stage_source(self, name: str, source: "BenchmarkSource") -> Path:
        """Fetch ``source`` once and return its root directory.

        Repeated calls for the same ``name`` reuse the first fetch.
        """
        if name in self._sources:
            return self._sources[name]

        dest = self.root / "sources" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)

        if source.git:
            self._clone_git(source.git, source.ref, dest)
        elif source.s3:
            self._sync_s3(source.s3, dest)
        elif source.path:
            local = Path(source.path).expanduser()
            if not local.is_dir():
                raise WorkspaceError(
                    f"benchmark source.path is not a directory: {local}"
                )
            shutil.copytree(local, dest)
        else:  # pragma: no cover - the schema validator forbids this
            raise WorkspaceError(f"benchmark {name!r}: source sets no location")

        root = dest / source.subdir if source.subdir else dest
        if not root.is_dir():
            raise WorkspaceError(
                f"benchmark {name!r}: source.subdir {source.subdir!r} is not a "
                f"directory in the fetched tree"
            )

        self._sources[name] = root
        logger.info("staged benchmark %r source at %s", name, root)
        return root

    @staticmethod
    def _clone_git(url: str, ref: Optional[str], dest: Path) -> None:
        """Clone ``url`` at ``ref``.

        A shallow branch/tag clone is tried first; it fails for a raw commit
        SHA, so that case falls back to a full clone plus checkout.
        """
        if ref:
            try:
                _run(
                    ["git", "clone", "--depth", "1", "--branch", ref, url, str(dest)],
                    timeout=GIT_CLONE_TIMEOUT,
                    what="git clone",
                )
                return
            except WorkspaceError:
                logger.info(
                    "shallow clone of ref %r failed; retrying as a commit-ish", ref
                )
                if dest.exists():
                    shutil.rmtree(dest)

        _run(
            ["git", "clone", url, str(dest)],
            timeout=GIT_CLONE_TIMEOUT,
            what="git clone",
        )
        if ref:
            _run(
                ["git", "-C", str(dest), "checkout", ref],
                timeout=GIT_CLONE_TIMEOUT,
                what="git checkout",
            )

    @staticmethod
    def _sync_s3(uri: str, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        _run(
            ["aws", "s3", "sync", uri.rstrip("/") + "/", str(dest)],
            timeout=GIT_CLONE_TIMEOUT,
            what="aws s3 sync",
        )

    # ------------------------------------------------------------------
    # Per-trial workspaces
    # ------------------------------------------------------------------

    def trial_workspace(
        self,
        benchmark: str,
        model: str,
        trial: int,
        *,
        seed_from: Optional[Path] = None,
    ) -> Path:
        """Return a fresh, empty workspace for one trial.

        Args:
            benchmark: Benchmark name.
            model: Candidate model name.
            trial: 0-based trial index.
            seed_from: Directory whose contents are copied in as the starting
                state (a scenario's input files, typically).
        """
        safe_model = model.replace("/", "_").replace(":", "_")
        path = self.root / "runs" / benchmark / safe_model / f"trial_{trial}"
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

        if seed_from is not None:
            if not seed_from.is_dir():
                raise WorkspaceError(f"seed directory not found: {seed_from}")
            shutil.copytree(seed_from, path, dirs_exist_ok=True)

        return path

    # ------------------------------------------------------------------
    # Isolated interpreters
    # ------------------------------------------------------------------

    def ensure_venv(self, name: str, requirements: List[str]) -> Path:
        """Create (once) a virtualenv for ``name`` and return its interpreter.

        A benchmark with pinned dependencies that clash with the job's own is
        the normal case, not the exception -- this keeps both working.
        """
        if name in self._venvs:
            return self._venvs[name]

        venv_dir = self.root / "venvs" / name
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            timeout=VENV_CREATE_TIMEOUT,
            what=f"create venv for {name}",
        )
        python = venv_dir / "bin" / "python"
        if not python.exists():  # pragma: no cover - Windows layout
            python = venv_dir / "Scripts" / "python.exe"

        if requirements:
            _run(
                [str(python), "-m", "pip", "install", "--no-cache-dir", *requirements],
                timeout=PIP_INSTALL_TIMEOUT,
                what=f"install requirements for {name}",
            )

        self._venvs[name] = python
        logger.info("isolated interpreter for %r at %s", name, python)
        return python

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Remove the root directory, unless ``keep`` was set."""
        if self.keep or not self._owns_root:
            return
        shutil.rmtree(self.root, ignore_errors=True)

    def __enter__(self) -> "WorkspaceManager":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["WorkspaceError", "WorkspaceManager"]
