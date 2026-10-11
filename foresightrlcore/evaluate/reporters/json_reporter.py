"""Writes the self-contained ``evaluate_results.json`` manifest.

Everything the run produced -- config echo, per-trial rows, aggregates, and
per-record samples -- in one document, so downstream tooling needs neither the
parquet nor this package to read a result.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from foresightrlcore.evaluate.registry import REPORTERS
from foresightrlcore.evaluate.spec import EvaluateResult

logger = logging.getLogger(__name__)

FILENAME = "evaluate_results.json"


class JsonReporter:
    """Dumps the :class:`EvaluateResult` document as JSON."""

    def report(self, result: EvaluateResult, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / FILENAME
        with path.open("w", encoding="utf-8") as handle:
            json.dump(result.model_dump(mode="json"), handle, indent=2, default=str)
        logger.info("wrote %s", path)


@REPORTERS.register("json")
def _make() -> JsonReporter:
    return JsonReporter()


__all__ = ["FILENAME", "JsonReporter"]
