"""Records written directly in the config, for smoke tests and tiny checks.

Keeps a "does my wiring work" run from needing a file, an S3 bucket, or a
network::

    dataset:
      loader: inline
      records:
        - id: q1
          question: "2 + 2 = ?"
          answer: "4"
          choices: ["3", "4", "5"]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from foresightrlcore.evaluate.datasets.tabular_loader import _apply_sample
from foresightrlcore.evaluate.interfaces import EvalRecord
from foresightrlcore.evaluate.registry import DATASET_LOADERS

logger = logging.getLogger(__name__)

_RESERVED = {"id", "question", "answer", "choices", "meta"}


class InlineDatasetLoader:
    """Builds records from a list of dicts in the dataset spec."""

    def records(
        self, spec: Dict[str, Any], sample: Optional[Dict[str, Any]] = None
    ) -> List[EvalRecord]:
        raw = spec.get("records")
        if not raw:
            raise ValueError("dataset loader 'inline' requires a 'records' list")

        records: List[EvalRecord] = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(
                    f"inline record #{index} must be a mapping, got {type(item).__name__}"
                )
            meta = dict(item.get("meta") or {})
            meta.update({k: v for k, v in item.items() if k not in _RESERVED})
            meta.setdefault("category", "default")
            records.append(
                EvalRecord(
                    id=str(item.get("id", index)),
                    inputs={"question": str(item.get("question", ""))},
                    gold=item.get("answer"),
                    choices=[str(c) for c in (item.get("choices") or [])],
                    meta=meta,
                )
            )

        records = _apply_sample(records, sample)
        limit = spec.get("limit")
        if limit:
            records = records[: int(limit)]
        return records


@DATASET_LOADERS.register("inline")
def _make() -> InlineDatasetLoader:
    return InlineDatasetLoader()


__all__ = ["InlineDatasetLoader"]
