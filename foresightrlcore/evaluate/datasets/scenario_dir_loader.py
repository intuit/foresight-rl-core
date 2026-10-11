"""Loader for benchmarks whose unit of work is a *directory*, not a row.

Agentic benchmarks do not ask a question and read an answer -- they drop an
agent into a workspace with input files and grade what it leaves behind. The
record for such a benchmark is a scenario directory: some inputs, and a
specification of what the result should look like.

Layout expected under ``path``::

    scenarios/
      scenario_001/
        manifest.json        # optional metadata (category, difficulty, ...)
        expected.json        # optional grading spec, passed through as `gold`
        inputs/              # optional; whatever the agent starts with
      scenario_002/
        ...

Everything except the directory itself is optional, because scenario suites
disagree about their own conventions. What the loader guarantees is a stable
record id (the directory name) and the absolute path in
``inputs["scenario_dir"]``; the benchmark interprets the rest.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from foresightrlcore.evaluate.datasets.tabular_loader import _apply_sample
from foresightrlcore.evaluate.interfaces import EvalRecord
from foresightrlcore.evaluate.registry import DATASET_LOADERS

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_NAME = "manifest.json"
DEFAULT_EXPECTED_NAME = "expected.json"


def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("%s is not valid JSON (%s); ignoring", path, exc)
        return None


class ScenarioDirLoader:
    """Yields one :class:`EvalRecord` per scenario directory."""

    def records(
        self, spec: Dict[str, Any], sample: Optional[Dict[str, Any]] = None
    ) -> List[EvalRecord]:
        path_value = spec.get("path")
        if not path_value:
            raise ValueError("dataset loader 'scenario_dir' requires a 'path'")
        root = Path(path_value).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"scenario root is not a directory: {root}")

        manifest_name = spec.get("manifest_name", DEFAULT_MANIFEST_NAME)
        expected_name = spec.get("expected_name", DEFAULT_EXPECTED_NAME)
        only = set(spec.get("scenarios") or [])

        records: List[EvalRecord] = []
        for scenario in sorted(p for p in root.iterdir() if p.is_dir()):
            if scenario.name.startswith("."):
                continue
            if only and scenario.name not in only:
                continue

            manifest = _read_json(scenario / manifest_name) or {}
            if not isinstance(manifest, dict):
                logger.warning(
                    "%s/%s is not a JSON object; ignoring", scenario, manifest_name
                )
                manifest = {}
            expected = _read_json(scenario / expected_name)

            meta: Dict[str, Any] = {"scenario": scenario.name, **manifest}
            meta.setdefault("category", manifest.get("category", "default"))

            records.append(
                EvalRecord(
                    id=str(manifest.get("id") or scenario.name),
                    inputs={
                        "scenario_dir": str(scenario.resolve()),
                        # Carried through so a benchmark that only needs the
                        # task text (rather than the directory) works against
                        # scenario data unchanged.
                        "question": str(
                            manifest.get("prompt") or manifest.get("question") or ""
                        ),
                    },
                    gold=expected,
                    meta=meta,
                )
            )

        if not records:
            logger.warning("no scenario directories found under %s", root)

        records = _apply_sample(records, sample)
        limit = spec.get("limit")
        if limit:
            records = records[: int(limit)]
        logger.info("loaded %d scenario(s) from %s", len(records), root)
        return records


@DATASET_LOADERS.register("scenario_dir")
def _make() -> ScenarioDirLoader:
    return ScenarioDirLoader()


__all__ = ["ScenarioDirLoader"]
