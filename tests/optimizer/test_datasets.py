"""Dataset providers (BTS + generic parquet) and stratified sampling.

The parquet tests require pandas/pyarrow (the optimizer extra) and are skipped
if unavailable; sampling is pure-stdlib and always runs.
"""

from __future__ import annotations

import pytest

from foresightrlcore.common.sampling import stratified_sample_indices


class TestStratifiedSampling:
    def test_non_positive_size_returns_empty(self):
        assert stratified_sample_indices([0, 1, 2], 0, seed=1) == []
        assert stratified_sample_indices([0, 1, 2], -5, seed=1) == []

    def test_size_ge_population_returns_all_indices(self):
        assert stratified_sample_indices([0, 0, 1], 3, seed=1) == [0, 1, 2]
        assert stratified_sample_indices([0, 0, 1], 99, seed=1) == [0, 1, 2]

    def test_sample_size_is_respected(self):
        strata = [0] * 10 + [1] * 10
        idxs = stratified_sample_indices(strata, 6, seed=7)
        assert len(idxs) == 6
        assert idxs == sorted(idxs)  # stable ordering
        assert all(0 <= i < 20 for i in idxs)

    def test_deterministic_for_seed(self):
        strata = [i % 3 for i in range(30)]
        a = stratified_sample_indices(strata, 9, seed=42)
        b = stratified_sample_indices(strata, 9, seed=42)
        assert a == b

    def test_proportional_allocation_across_strata(self):
        # 15 in shard 0, 5 in shard 1; sampling 8 should keep ~3:1 proportion.
        strata = [0] * 15 + [1] * 5
        idxs = stratified_sample_indices(strata, 8, seed=3)
        from_shard0 = sum(1 for i in idxs if i < 15)
        from_shard1 = len(idxs) - from_shard0
        assert len(idxs) == 8
        assert from_shard0 == 6  # round(8 * 15/20)
        assert from_shard1 == 2  # round(8 * 5/20)


pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
# Absent from OSS checkouts — skip the module rather than fail collection.
pytest.importorskip("foresightrl_intuit")

from foresightrl_intuit.optimizer.datasets.bts_parquet import (  # noqa: E402
    BtsParquetDataset,
)

from foresightrlcore.optimizer.datasets.parquet_dataset import (  # noqa: E402
    GenericParquetDataset,
    _apply_filter,
)


def _bts_frame():
    return pd.DataFrame(
        {
            "realm_id": [f"r{i}" for i in range(6)],
            "epoch_date": [20240101 + i for i in range(6)],
            "state": [f"state text {i}" for i in range(6)],
            "evidence_bundle": [f"evidence {i}" for i in range(6)],
            "data_complete": [True, True, False, True, False, True],
            "shard": [0, 0, 1, 1, 2, 2],
        }
    )


def _write_parquet(tmp_path, frame, name="data.parquet"):
    path = tmp_path / name
    frame.to_parquet(path)
    return str(path)


class TestApplyFilter:
    def test_empty_filter_is_passthrough(self):
        df = _bts_frame()
        assert _apply_filter(df, {}).equals(df)

    def test_equality_filter(self):
        df = _bts_frame()
        out = _apply_filter(df, {"data_complete": True})
        assert len(out) == 4
        assert bool(out["data_complete"].all())

    def test_unknown_column_is_ignored(self):
        df = _bts_frame()
        out = _apply_filter(df, {"does_not_exist": 1})
        assert len(out) == len(df)


class TestBtsParquetDataset:
    def test_requires_path(self):
        with pytest.raises(ValueError, match="requires a 'path'"):
            list(BtsParquetDataset().records({"name": "bts_parquet"}, {"size": 2}))

    def test_missing_required_columns_raises(self, tmp_path):
        frame = pd.DataFrame({"realm_id": ["r0"], "epoch_date": [1]})
        path = _write_parquet(tmp_path, frame)
        with pytest.raises(ValueError, match="missing required columns"):
            list(BtsParquetDataset().records({"path": path}, {"size": 1}))

    def test_records_shape_and_inputs(self, tmp_path):
        path = _write_parquet(tmp_path, _bts_frame())
        records = list(
            BtsParquetDataset().records(
                {"name": "bts", "path": path},
                {"size": 4, "seed": 13, "stratify_by": "shard"},
            )
        )
        assert len(records) == 4
        rec = records[0]
        assert rec.inputs["start_date"] == ""
        assert rec.inputs["end_date"] == str(rec.inputs["end_date"])
        assert "evidence_bundle" in rec.inputs
        assert rec.meta["dataset"] == "bts"
        assert "shard" in rec.meta
        assert isinstance(rec.meta["data_complete"], bool)

    def test_filter_reduces_population(self, tmp_path):
        path = _write_parquet(tmp_path, _bts_frame())
        records = list(
            BtsParquetDataset().records(
                {"path": path, "filter": {"data_complete": True}},
                {"size": 50},  # size >= filtered rows -> take all matching
            )
        )
        assert len(records) == 4

    def test_stratify_fallback_when_column_absent(self, tmp_path):
        frame = _bts_frame().drop(columns=["shard"])
        path = _write_parquet(tmp_path, frame)
        records = list(
            BtsParquetDataset().records(
                {"path": path}, {"size": 3, "stratify_by": "shard"}
            )
        )
        assert len(records) == 3
        assert "shard" not in records[0].meta


class TestGenericParquetDataset:
    def test_requires_path(self):
        with pytest.raises(ValueError, match="requires a 'path'"):
            list(GenericParquetDataset().records({"name": "parquet"}, {"size": 1}))

    def test_missing_mapped_columns_raises(self, tmp_path):
        frame = pd.DataFrame({"a": [1, 2], "shard": [0, 1]})
        path = _write_parquet(tmp_path, frame)
        spec = {
            "path": path,
            "input_columns": {"foo": "nonexistent"},
            "reference_column": "also_missing",
        }
        with pytest.raises(ValueError, match="missing mapped columns"):
            list(GenericParquetDataset().records(spec, {"size": 1}))

    def test_column_mapping_and_constants(self, tmp_path):
        frame = pd.DataFrame(
            {
                "company": ["c0", "c1", "c2"],
                "evidence": ["e0", "e1", "e2"],
                "gold": ["g0", "g1", "g2"],
                "when": [2024, 2025, 2026],
                "shard": [0, 1, 0],
            }
        )
        path = _write_parquet(tmp_path, frame)
        spec = {
            "name": "generic",
            "path": path,
            "input_columns": {"evidence_bundle": "evidence"},
            "constant_inputs": {"start_date": ""},
            "reference_column": "gold",
            "id_column": "company",
            "epoch_column": "when",
        }
        records = list(GenericParquetDataset().records(spec, {"size": 3, "seed": 1}))
        assert len(records) == 3
        rec = records[0]
        assert rec.inputs["start_date"] == ""
        assert rec.inputs["evidence_bundle"].startswith("e")
        assert rec.realm_id.startswith("c")
        assert rec.reference.startswith("g")
        assert rec.meta["dataset"] == "generic"
        assert "shard" in rec.meta

    def test_id_and_epoch_fallback_when_columns_absent(self, tmp_path):
        frame = pd.DataFrame({"evidence": ["e0", "e1"]})
        path = _write_parquet(tmp_path, frame)
        spec = {
            "path": path,
            "input_columns": {"evidence_bundle": "evidence"},
        }
        records = list(GenericParquetDataset().records(spec, {"size": 2, "seed": 1}))
        # id_column defaults to realm_id (absent) -> index; epoch defaults to "".
        assert {r.realm_id for r in records} == {"0", "1"}
        assert all(r.epoch_date == "" for r in records)
        assert all(r.reference == "" for r in records)
