"""Tests for :mod:`foresightrlcore.serverless.convert` (generic loop + mapper resolution)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest

from foresightrlcore.serverless import convert, registry
from foresightrlcore.serverless.spec import RewardModel, RLVRRecord


@pytest.fixture(autouse=True)
def _isolate_registry():
    snap_c = dict(registry.CONVERTERS)
    try:
        yield
    finally:
        registry.CONVERTERS.clear()
        registry.CONVERTERS.update(snap_c)


def _good_record():
    return RLVRRecord(
        data_source="t",
        prompt=[{"role": "user", "content": "q"}],
        ability="generation",
        reward_model=RewardModel(ground_truth="gt", style="model"),
    )


# ---------------------------------------------------------------------------
# Mapper resolution
# ---------------------------------------------------------------------------


def _identity_mapper(record):
    return _good_record()


class TestResolveMapper:
    def test_dotted_path(self):
        mapper = convert.resolve_mapper(
            converter="tests.serverless.test_convert:_identity_mapper",
            schema="rlvr",
        )
        assert callable(mapper)

    def test_registry_name(self):
        registry.register_converter(
            name="t-reg",
            mapper="tests.serverless.test_convert:_identity_mapper",
        )
        mapper = convert.resolve_mapper(converter="t-reg", schema="rlvr")
        assert callable(mapper)

    def test_unsupported_schema_for_converter(self):
        registry.register_converter(
            name="t-rlvr-only",
            mapper="tests.serverless.test_convert:_identity_mapper",
            schemas={"rlvr"},
        )
        with pytest.raises(ValueError, match="supports schemas"):
            convert.resolve_mapper(converter="t-rlvr-only", schema="dpo")

    def test_both_converter_and_mapping_is_an_error(self):
        with pytest.raises(ValueError, match="exactly one"):
            convert.resolve_mapper(converter="x", mapping="y.yaml", schema="rlvr")

    def test_neither_is_an_error(self):
        with pytest.raises(ValueError, match="required"):
            convert.resolve_mapper(schema="rlvr")


# ---------------------------------------------------------------------------
# YAML mapping spec
# ---------------------------------------------------------------------------


def test_yaml_mapping_spec_round_trip(tmp_path: Path):
    yaml_text = """
schema: rlvr
data_source: mailchimp
ability: classification
reward_style: rule
prompt_from: prompt
ground_truth_from: outcome
extra_info_passthrough:
  - trace_id
"""
    mapping_path = tmp_path / "mapping.yaml"
    mapping_path.write_text(yaml_text)

    mapper = convert.resolve_mapper(mapping=str(mapping_path), schema="rlvr")
    record = mapper(
        {
            "prompt": [{"role": "user", "content": "classify this"}],
            "outcome": "positive",
            "trace_id": "tr-1",
            "ignored": "should not appear",
        }
    )
    assert isinstance(record, RLVRRecord)
    assert record.data_source == "mailchimp"
    assert record.reward_model.ground_truth == "positive"
    assert record.extra_info == {"trace_id": "tr-1"}


def test_yaml_mapping_skips_missing(tmp_path: Path):
    yaml_text = """
schema: rlvr
data_source: m
prompt_from: prompt
ground_truth_from: outcome
"""
    mp = tmp_path / "m.yaml"
    mp.write_text(yaml_text)
    mapper = convert.resolve_mapper(mapping=str(mp), schema="rlvr")
    assert mapper({"prompt": "p"}) is None  # missing outcome -> skip


# ---------------------------------------------------------------------------
# convert_file loop
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_convert_file_happy_path(tmp_path: Path):
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    _write_jsonl(src, [{"x": 1}, {"x": 2}])

    summary = convert.convert_file(
        input_path=str(src),
        output_path=str(dst),
        mapper=_identity_mapper,
        schema="rlvr",
    )
    assert summary.total_in == 2
    assert summary.total_out == 2
    assert summary.skipped == 0


def test_convert_file_skips_on_mapper_returning_none(tmp_path: Path):
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    _write_jsonl(src, [{"a": 1}, {"a": 2}, {"a": 3}])

    def mapper(record):
        return None if record["a"] == 2 else _good_record()

    summary = convert.convert_file(
        input_path=str(src), output_path=str(dst), mapper=mapper, schema="rlvr"
    )
    assert summary.total_in == 3
    assert summary.total_out == 2
    assert summary.skipped == 1


def test_convert_file_supports_iterator_fanout(tmp_path: Path):
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    _write_jsonl(src, [{"x": 1}])

    def fanout(record) -> Iterator:
        yield _good_record()
        yield _good_record()

    summary = convert.convert_file(
        input_path=str(src), output_path=str(dst), mapper=fanout, schema="rlvr"
    )
    assert summary.total_in == 1
    assert summary.total_out == 2


def test_convert_file_skips_on_mapper_raising(tmp_path: Path):
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    _write_jsonl(src, [{"x": 1}, {"x": 2}])

    def boom(record):
        if record["x"] == 1:
            raise ValueError("kaboom")
        return _good_record()

    summary = convert.convert_file(
        input_path=str(src), output_path=str(dst), mapper=boom, schema="rlvr"
    )
    assert summary.total_out == 1
    assert summary.skipped == 1


def test_convert_file_skips_on_schema_validation_failure(tmp_path: Path):
    src = tmp_path / "in.jsonl"
    dst = tmp_path / "out.jsonl"
    _write_jsonl(src, [{"x": 1}])

    def bad(record):
        return {"data_source": "x"}  # incomplete -> validation error

    summary = convert.convert_file(
        input_path=str(src), output_path=str(dst), mapper=bad, schema="rlvr"
    )
    assert summary.total_out == 0
    assert summary.skipped == 1
