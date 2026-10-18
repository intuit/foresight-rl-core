"""CSV / JSONL / Parquet loaders for row-per-question benchmarks.

One implementation serves all three: they differ only in how rows are read.
CSV and JSONL use the stdlib so a smoke run needs no pandas; parquet requires
pandas, which the container recipe installs.

Column mapping is configurable because benchmark CSVs never agree on names.
The defaults follow the most common convention (``question``/``answer``), and
numbered choice columns (``1``, ``2``, ...) or a single ``choices`` column are
both understood.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from foresightrlcore.common.sampling import stratified_sample_indices
from foresightrlcore.evaluate.interfaces import EvalRecord
from foresightrlcore.evaluate.registry import DATASET_LOADERS

logger = logging.getLogger(__name__)

DEFAULT_QUESTION_COLUMNS = ("question", "prompt", "input", "text")
DEFAULT_ANSWER_COLUMNS = ("answer", "label", "gold", "target")
DEFAULT_ID_COLUMNS = ("id", "example_id", "record_id")


#: A chosen question column is suspicious when another candidate column holds
#: this many times more text. The ratio is deliberately blunt: a column that
#: merely phrases things differently varies by a few percent, whereas a column
#: that drops the passage a question refers to loses most of the row.
_TRUNCATION_RATIO = 1.5


def _first_present(row: Dict[str, Any], names: Iterable[str]) -> Optional[str]:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return name
    return None


def _warn_if_truncating(rows: List[Dict[str, Any]], chosen: str, path: Path) -> None:
    """Warn when a rejected candidate column carries far more text than ``chosen``.

    A table with both a ``question`` and a ``prompt`` column usually means the
    question alone is not answerable: ``prompt`` prepends the passage it refers
    to. Precedence picks ``question``, so the model is asked what can be
    concluded from information it was never shown, and answers that it cannot.
    That reads on a leaderboard as a weak model rather than a broken harness,
    which is why this is worth a warning rather than a silent default.
    """
    sample = rows[: min(len(rows), 200)]

    def mean_len(column: str) -> float:
        return sum(len(str(r.get(column) or "")) for r in sample) / len(sample)

    chosen_len = mean_len(chosen)
    if not chosen_len:
        return
    for candidate in DEFAULT_QUESTION_COLUMNS:
        if candidate == chosen or candidate not in sample[0]:
            continue
        other_len = mean_len(candidate)
        if other_len > chosen_len * _TRUNCATION_RATIO:
            logger.warning(
                "%s: using column %r (%.0f chars/row on average) but column %r "
                "holds %.0f -- if %r prepends a passage the question refers to, "
                "the model is being asked to reason about text it cannot see. "
                "Set dataset.question_column explicitly to silence this.",
                path,
                chosen,
                chosen_len,
                candidate,
                other_len,
                candidate,
            )
            return


def _extract_choices(
    row: Dict[str, Any], number_of_choices: Optional[int]
) -> List[str]:
    """Pull answer options from either a ``choices`` column or numbered columns."""
    raw = row.get("choices")
    if isinstance(raw, list):
        return [str(c) for c in raw]
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and parsed:
                return [str(c) for c in parsed]
        except json.JSONDecodeError:
            parts = [part.strip() for part in raw.split("|") if part.strip()]
            if len(parts) > 1:
                return parts
        # A single unsplittable blob is not a choice list -- it is usually a
        # repr of one ("['a', \"b's\"]", which is not JSON). Fall through to
        # the numbered columns, which are authoritative when both exist.

    # Numbered choice columns: "1", "2", "3", "4" (the common leaderboard shape).
    # Blank cells are dropped rather than rendered as empty options: a table
    # wide enough for its longest question leaves trailing blanks on every
    # shorter one, and a gold index counts only the real options.
    numbered: List[str] = []
    index = 1
    while True:
        if number_of_choices is not None and index > number_of_choices:
            break
        key = str(index)
        if key not in row:
            break
        value = row[key]
        if value is not None and str(value).strip():
            numbered.append(str(value).strip())
        index += 1
    return numbered


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("%s:%d is not valid JSON; skipping", path, lineno)
    return rows


def _read_parquet(path: Path) -> List[Dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(path)
    return [
        {k: v for k, v in record.items()} for record in frame.to_dict(orient="records")
    ]


class TabularDatasetLoader:
    """Loads records from a flat table.

    Args:
        reader: ``"csv"``, ``"jsonl"``, or ``"parquet"``.
    """

    def __init__(self, reader: str) -> None:
        self.reader = reader

    def _read_one(self, path: Path) -> List[Dict[str, Any]]:
        if self.reader == "csv":
            return _read_csv(path)
        if self.reader == "jsonl":
            return _read_jsonl(path)
        if self.reader == "parquet":
            return _read_parquet(path)
        raise ValueError(f"unknown tabular reader: {self.reader!r}")

    def _read(self, path: Path) -> List[Dict[str, Any]]:
        """Read one file, or every part file in a directory.

        Partitioned output is the normal shape for anything Spark or pandas
        wrote, so a directory is a dataset too. Parts are read in sorted name
        order, which is what makes row indices — and therefore the default
        record ids — stable across runs.
        """
        if not path.is_dir():
            return self._read_one(path)

        parts = sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and not p.name.startswith(("_", "."))
        )
        if not parts:
            raise FileNotFoundError(f"no data files under {path}")
        rows: List[Dict[str, Any]] = []
        for part in parts:
            rows.extend(self._read_one(part))
        logger.info("read %d part file(s) from %s", len(parts), path)
        return rows

    def records(
        self, spec: Dict[str, Any], sample: Optional[Dict[str, Any]] = None
    ) -> List[EvalRecord]:
        path_value = spec.get("path")
        if not path_value:
            raise ValueError(f"dataset loader {self.reader!r} requires a 'path'")
        path = Path(path_value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"dataset path not found: {path}")

        rows = self._read(path)
        if not rows:
            logger.warning("dataset %s is empty", path)
            return []

        question_col = spec.get("question_column")
        answer_col = spec.get("answer_column")
        id_col = spec.get("id_column")
        # Datasets keyed by a compound business key (customer + document, say)
        # have no single id column. Falling back to the row index makes resume
        # depend on row order holding across runs, which is not a safe bet.
        id_cols = [str(c) for c in (spec.get("id_columns") or [])]
        category_col = spec.get("category_column", "category")
        number_of_choices = spec.get("number_of_choices")

        first = rows[0]
        explicit_question_col = question_col is not None
        question_col = question_col or _first_present(first, DEFAULT_QUESTION_COLUMNS)
        answer_col = answer_col or _first_present(first, DEFAULT_ANSWER_COLUMNS)
        id_col = id_col or _first_present(first, DEFAULT_ID_COLUMNS)
        if question_col is None:
            raise ValueError(
                f"{path}: could not find a question column (looked for "
                f"{list(DEFAULT_QUESTION_COLUMNS)}); set dataset.question_column"
            )
        if not explicit_question_col:
            _warn_if_truncating(rows, question_col, path)
        # A missing name would key every row by an empty segment; the duplicate
        # ids then get renamed by position, quietly restoring the row-order
        # dependence id_columns is there to remove.
        missing_id_cols = [c for c in id_cols if c not in first]
        if missing_id_cols:
            raise ValueError(
                f"{path}: dataset.id_columns {id_cols} names columns the data "
                f"does not have: {missing_id_cols}"
            )

        records: List[EvalRecord] = []
        for index, row in enumerate(rows):
            meta = {k: v for k, v in row.items() if k not in (question_col, answer_col)}
            meta["row_index"] = index
            if category_col in row:
                meta["category"] = row[category_col]
            if id_cols:
                record_id = "|".join(str(row.get(c, "")) for c in id_cols)
            elif id_col:
                record_id = str(row.get(id_col, index))
            else:
                record_id = str(index)
            records.append(
                EvalRecord(
                    id=record_id,
                    inputs={"question": str(row.get(question_col, ""))},
                    gold=row.get(answer_col) if answer_col else None,
                    choices=_extract_choices(row, number_of_choices),
                    meta=meta,
                )
            )

        records = _apply_sample(records, sample)
        limit = spec.get("limit")
        if limit:
            records = records[: int(limit)]
        logger.info("loaded %d records from %s", len(records), path)
        return records


def _apply_sample(
    records: List[EvalRecord], sample: Optional[Dict[str, Any]]
) -> List[EvalRecord]:
    """Deterministically sub-sample ``records`` when a sample config is given."""
    if not sample:
        return records
    size = int(sample.get("size", 0) or 0)
    if size <= 0 or size >= len(records):
        return records
    stratify_by = sample.get("stratify_by") or "category"
    strata = [r.meta.get(stratify_by, "_") for r in records]
    indices = stratified_sample_indices(strata, size, int(sample.get("seed", 13)))
    return [records[i] for i in indices]


@DATASET_LOADERS.register("csv")
def _make_csv() -> TabularDatasetLoader:
    return TabularDatasetLoader("csv")


@DATASET_LOADERS.register("jsonl")
def _make_jsonl() -> TabularDatasetLoader:
    return TabularDatasetLoader("jsonl")


@DATASET_LOADERS.register("parquet")
def _make_parquet() -> TabularDatasetLoader:
    return TabularDatasetLoader("parquet")


__all__ = ["TabularDatasetLoader"]
