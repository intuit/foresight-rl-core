"""Parquet dataset providers: generic column-mapped loader.

The ``parquet`` provider is driven by a configurable column mapping
(``reference_column``, ``input_columns``, ``constant_inputs``, ``id_column``,
``epoch_column``) so new use cases need no bespoke loader.

Accepts a local filesystem path (S3 URIs are downloaded to a local path by
the runner before ``records`` is called).

Intuit-specific presets (e.g. ``bts_parquet``) live in ``foresightrl_intuit``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from foresightrlcore.common.sampling import stratified_sample_indices
from foresightrlcore.optimizer.interfaces import Record
from foresightrlcore.optimizer.registry import DATASETS

COL_REALM = "realm_id"
COL_EPOCH = "epoch_date"
COL_SHARD = "shard"


def _apply_filter(df: Any, filt: Dict[str, Any]) -> Any:
    import pandas as pd

    if not filt:
        return df
    mask = pd.Series(True, index=df.index)
    for col, expected in filt.items():
        if col in df.columns:
            mask &= df[col] == expected
    return df[mask]


class GenericParquetDataset:
    """Column-mapped parquet provider for arbitrary use cases."""

    def records(self, spec: Dict[str, Any], sample: Dict[str, Any]) -> Iterable[Record]:
        import pandas as pd

        path = spec.get("path")
        if not path:
            raise ValueError("parquet dataset requires a 'path'")

        df = pd.read_parquet(path)
        df = _apply_filter(df, spec.get("filter") or {})
        df = df.reset_index(drop=True)

        input_columns: Dict[str, str] = dict(spec.get("input_columns") or {})
        constant_inputs: Dict[str, Any] = dict(spec.get("constant_inputs") or {})
        reference_column = spec.get("reference_column")
        id_column = spec.get("id_column", COL_REALM)
        epoch_column = spec.get("epoch_column", COL_EPOCH)

        missing = [c for c in input_columns.values() if c not in df.columns]
        if reference_column and reference_column not in df.columns:
            missing.append(reference_column)
        if missing:
            raise ValueError(
                f"parquet at {path} missing mapped columns: {missing}; "
                f"present: {list(df.columns)}"
            )

        size = int(sample.get("size", 50))
        seed = int(sample.get("seed", 13))
        stratify_by = sample.get("stratify_by", COL_SHARD)
        if stratify_by in df.columns:
            strata = df[stratify_by].tolist()
        else:
            strata = [0] * len(df)
        idxs = stratified_sample_indices(strata, size, seed)

        dataset_name = spec.get("name", "parquet")
        records: List[Record] = []
        for i in idxs:
            row = df.iloc[i]
            inputs: Dict[str, Any] = {
                name: row[col] for name, col in input_columns.items()
            }
            inputs.update(constant_inputs)
            realm_id = str(row[id_column]) if id_column in df.columns else str(i)
            epoch = str(row[epoch_column]) if epoch_column in df.columns else ""
            meta: Dict[str, Any] = {"dataset": dataset_name}
            if stratify_by in df.columns:
                meta["shard"] = row[stratify_by]
            reference = str(row[reference_column]) if reference_column else ""
            records.append(
                Record(
                    realm_id=realm_id,
                    epoch_date=epoch,
                    inputs=inputs,
                    reference=reference,
                    meta=meta,
                )
            )
        return records


@DATASETS.register("parquet")
def _make_generic_parquet() -> GenericParquetDataset:
    return GenericParquetDataset()


__all__ = ["GenericParquetDataset"]
