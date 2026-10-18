"""Tests for :mod:`foresightrlcore.serverless.spec` (schema registry)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from foresightrlcore.serverless import spec


class TestRLVRRecord:
    def test_minimal_record_round_trips(self):
        record = spec.RLVRRecord(
            data_source="advice",
            prompt=[{"role": "user", "content": "q"}],
            ability="generation",
            reward_model=spec.RewardModel(ground_truth="gt", style="model"),
        )
        dumped = record.model_dump_json(exclude_none=True)
        re_parsed = spec.RLVRRecord.model_validate_json(dumped)
        assert re_parsed == record

    def test_extra_keys_forbidden(self):
        with pytest.raises(ValidationError):
            spec.RLVRRecord(
                data_source="advice",
                prompt=[{"role": "user", "content": "q"}],
                ability="generation",
                reward_model=spec.RewardModel(ground_truth="gt", style="model"),
                unexpected_key="boom",  # type: ignore[call-arg]
            )

    def test_missing_required_fails(self):
        with pytest.raises(ValidationError):
            spec.RLVRRecord(  # type: ignore[call-arg]
                data_source="advice",
                prompt=[{"role": "user", "content": "q"}],
                ability="generation",
                # reward_model missing
            )


class TestGetSchema:
    def test_returns_rlvr(self):
        assert spec.get_schema("rlvr") is spec.RLVRRecord

    def test_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown schema"):
            spec.get_schema("not-a-thing")

    @pytest.mark.parametrize("name", ["sft", "dpo", "kto"])
    def test_reserved_raise_not_implemented(self, name):
        with pytest.raises(NotImplementedError, match=name):
            spec.get_schema(name)


class TestSchemaCatalog:
    def test_available_schemas_includes_rlvr(self):
        assert "rlvr" in spec.available_schemas()

    def test_reserved_lists_sft_dpo_kto(self):
        reserved = spec.reserved_schemas()
        for key in ("sft", "dpo", "kto"):
            assert key in reserved
