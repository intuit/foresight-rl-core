"""Tests for foresightrlcore.methods.verl_sft_dataset.

The module must import cleanly without ``verl`` installed (it isn't a
dev/test dependency — see the module docstring and the identical convention
in foresightrlcore/methods/base.py and grpo.py for trl/peft). The lazy
import boundary is tested first, below.

``PackedSFTDataset``'s one piece of pure logic — truncation +
per-sub-sequence ``position_ids`` resets — is factored out into
``truncate_and_reset_positions()``, tested directly in
``TestTruncateAndResetPositions`` below with no torch/verl dependency at
all.

The REST of each dataset class's real logic (``_read_files_and_process``,
``PackedSFTDataset.__getitem__``) is exercised in
``TestFlatPromptResponseSFTDatasetRealLogic``/``TestPackedSFTDatasetRealLogic``
by injecting fake ``verl.utils.dataset.*`` modules into ``sys.modules`` for
the duration of each test (torch/pandas are real dependencies already, so
only verl itself needs faking). This is the same class of technique
``tests/conftest.py`` already uses at session scope for ``peft``/
``transformers``/``datasets`` (stubbed there as bare ``MagicMock()``s so
``foresightrlcore/methods/grpo.py``/``base.py`` can be exercised without
those installed either) — here it's done per-test instead of globally,
because the classes under test need SPECIFIC, not just non-crashing,
return values (dataframe contents, position_ids, etc.) to assert against.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from foresightrlcore.methods.verl_sft_dataset import truncate_and_reset_positions


def test_module_imports_without_verl_installed():
    import foresightrlcore.methods.verl_sft_dataset as mod

    assert callable(mod.FlatPromptResponseSFTDataset)
    assert callable(mod.PackedSFTDataset)


@pytest.mark.parametrize(
    "dataset_cls_name", ["FlatPromptResponseSFTDataset", "PackedSFTDataset"]
)
def test_instantiation_fails_with_clear_import_error_when_verl_missing(
    dataset_cls_name,
):
    import foresightrlcore.methods.verl_sft_dataset as mod

    dataset_cls = getattr(mod, dataset_cls_name)

    try:
        import verl  # noqa: F401

        pytest.skip(
            "verl is installed in this environment; lazy-import boundary not exercised"
        )
    except ImportError:
        pass

    with pytest.raises(ImportError):
        dataset_cls(
            parquet_files=["/tmp/does-not-matter.jsonl"],
            tokenizer=object(),
            config={},
        )


class TestTruncateAndResetPositions:
    """The real, torch/verl-free logic behind PackedSFTDataset.__getitem__."""

    def test_single_sequence_gets_plain_arange_positions(self):
        input_ids, loss_mask, position_ids = truncate_and_reset_positions(
            input_ids=[10, 11, 12, 13],
            loss_mask=[0, 0, 1, 1],
            seq_start_id=[0],
            max_length=100,
            truncation="error",
        )
        assert input_ids == [10, 11, 12, 13]
        assert loss_mask == [0, 0, 1, 1]
        assert position_ids == [0, 1, 2, 3]

    def test_multi_document_row_resets_positions_at_each_boundary(self):
        # Two packed docs of length 3 and 2.
        input_ids, loss_mask, position_ids = truncate_and_reset_positions(
            input_ids=[1, 2, 3, 4, 5],
            loss_mask=[0, 1, 1, 0, 1],
            seq_start_id=[0, 3],
            max_length=100,
            truncation="error",
        )
        assert position_ids == [0, 1, 2, 0, 1]
        # input_ids/loss_mask pass through unchanged when no truncation is needed.
        assert input_ids == [1, 2, 3, 4, 5]
        assert loss_mask == [0, 1, 1, 0, 1]

    def test_three_documents(self):
        _, _, position_ids = truncate_and_reset_positions(
            input_ids=list(range(10)),
            loss_mask=[0] * 10,
            seq_start_id=[0, 4, 7],
            max_length=100,
            truncation="error",
        )
        assert position_ids == [0, 1, 2, 3, 0, 1, 2, 0, 1, 2]

    def test_no_truncation_needed_below_max_length(self):
        input_ids, loss_mask, position_ids = truncate_and_reset_positions(
            input_ids=[1, 2, 3],
            loss_mask=[0, 0, 1],
            seq_start_id=[0],
            max_length=3,
            truncation="error",
        )
        assert len(input_ids) == 3
        assert position_ids == [0, 1, 2]

    def test_truncation_error_raises_when_over_max_length(self):
        with pytest.raises(ValueError, match="larger than"):
            truncate_and_reset_positions(
                input_ids=[1, 2, 3, 4],
                loss_mask=[0, 0, 1, 1],
                seq_start_id=[0],
                max_length=3,
                truncation="error",
            )

    def test_truncation_right_drops_the_tail_and_keeps_boundaries_correct(self):
        # Two docs of length 3 and 3; truncate to 4 -> keeps doc0 fully (3
        # tokens) plus 1 token of doc1, whose boundary at offset 3 survives.
        input_ids, loss_mask, position_ids = truncate_and_reset_positions(
            input_ids=[1, 2, 3, 4, 5, 6],
            loss_mask=[0, 0, 1, 0, 0, 1],
            seq_start_id=[0, 3],
            max_length=4,
            truncation="right",
        )
        assert input_ids == [1, 2, 3, 4]
        assert loss_mask == [0, 0, 1, 0]
        assert position_ids == [0, 1, 2, 0]

    def test_truncation_left_drops_the_head_and_shifts_boundaries(self):
        # Two docs of length 3 and 3; truncate to 4 from the left -> drops
        # the first 2 tokens of doc0, keeping its last token plus all of doc1.
        input_ids, loss_mask, position_ids = truncate_and_reset_positions(
            input_ids=[1, 2, 3, 4, 5, 6],
            loss_mask=[0, 0, 1, 0, 0, 1],
            seq_start_id=[0, 3],
            max_length=4,
            truncation="left",
        )
        assert input_ids == [3, 4, 5, 6]
        assert loss_mask == [1, 0, 0, 1]
        # A new sub-sequence boundary is forced at 0 (surviving remainder of
        # doc0), plus doc1's shifted boundary at 1.
        assert position_ids == [0, 0, 1, 2]

    def test_truncation_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown truncation method"):
            truncate_and_reset_positions(
                input_ids=[1, 2, 3, 4],
                loss_mask=[0, 0, 1, 1],
                seq_start_id=[0],
                max_length=3,
                truncation="bogus",
            )


class _FakeMultiTurnSFTDataset:
    """Stand-in for verl's real base class — mirrors exactly the attributes

    both real subclasses in verl_sft_dataset.py read (messages_key, shuffle,
    seed, apply_chat_template_kwargs, pad_mode, max_length, truncation),
    then dispatches to ``self._read_files_and_process()`` polymorphically,
    exactly like the real base class does per the module docstring.
    """

    def __init__(
        self, parquet_files, tokenizer, config, processor=None, max_samples=-1
    ):
        cfg = config or {}
        self.parquet_files = parquet_files
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.messages_key = cfg.get("messages_key", "messages")
        self.shuffle = cfg.get("shuffle", False)
        self.seed = cfg.get("seed", None)
        self.apply_chat_template_kwargs = cfg.get("apply_chat_template_kwargs", {})
        self.pad_mode = cfg.get("pad_mode", "no_padding")
        self.max_length = cfg.get("max_length", 8192)
        self.truncation = cfg.get("truncation", "error")
        self._read_files_and_process()

    def _read_files_and_process(self) -> None:
        """Overridden by every real subclass — see module docstring."""
        raise NotImplementedError


@pytest.fixture
def fake_verl_dataset_modules(monkeypatch):
    """Inject fake verl.utils.dataset.* modules so _dataset_class()/

    _packed_dataset_class() build their real subclasses against a
    controllable base, then clear both lru_caches so every test gets a
    fresh class built from ITS OWN fakes (the cache would otherwise pin
    whichever fakes were installed by the first test to ever call it).
    """
    import foresightrlcore.methods.verl_sft_dataset as mod

    calls: dict = {"extract_system_prompt_and_generation": []}

    def fake_extract_system_prompt_and_generation(tokenizer, **kwargs):
        calls["extract_system_prompt_and_generation"].append(kwargs)
        return ("SYS_PREFIX", "GEN_PREFIX")

    multiturn_mod: Any = ModuleType("verl.utils.dataset.multiturn_sft_dataset")
    multiturn_mod.MultiTurnSFTDataset = _FakeMultiTurnSFTDataset

    dataset_utils_mod: Any = ModuleType("verl.utils.dataset.dataset_utils")
    dataset_utils_mod.DatasetPadMode = SimpleNamespace(NO_PADDING="no_padding")

    py_functional_mod: Any = ModuleType("verl.utils.py_functional")
    py_functional_mod.convert_nested_value_to_list_recursive = lambda v: v

    chat_template_mod: Any = ModuleType("verl.utils.tokenizer.chat_template")
    chat_template_mod.extract_system_prompt_and_generation = (
        fake_extract_system_prompt_and_generation
    )

    monkeypatch.setitem(
        sys.modules, "verl.utils.dataset.multiturn_sft_dataset", multiturn_mod
    )
    monkeypatch.setitem(
        sys.modules, "verl.utils.dataset.dataset_utils", dataset_utils_mod
    )
    monkeypatch.setitem(sys.modules, "verl.utils.py_functional", py_functional_mod)
    monkeypatch.setitem(
        sys.modules, "verl.utils.tokenizer.chat_template", chat_template_mod
    )

    mod._dataset_class.cache_clear()
    mod._packed_dataset_class.cache_clear()
    yield calls
    mod._dataset_class.cache_clear()
    mod._packed_dataset_class.cache_clear()


class TestFlatPromptResponseSFTDatasetRealLogic:
    """Exercises _FlatPromptResponseSFTDataset._read_files_and_process —

    the real messages-synthesis logic, not just the lazy-import boundary.
    """

    def test_builds_messages_from_prompt_response_columns(
        self, tmp_path, fake_verl_dataset_modules
    ):
        from foresightrlcore.methods.verl_sft_dataset import (
            FlatPromptResponseSFTDataset,
        )

        path = tmp_path / "data.parquet"
        pd.DataFrame(
            {"prompt": ["hi", "howdy"], "response": ["hi back", "howdy back"]}
        ).to_parquet(path)

        ds = FlatPromptResponseSFTDataset(
            parquet_files=[str(path)],
            tokenizer=object(),
            config={"prompt_key": "prompt", "response_key": "response"},
        )

        assert ds.messages[0] == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hi back"},
        ]
        assert ds.messages[1] == [
            {"role": "user", "content": "howdy"},
            {"role": "assistant", "content": "howdy back"},
        ]
        assert list(ds.dataframe["messages"]) == ds.messages
        assert ds.tools is None
        assert ds.enable_thinking is None
        assert ds.system_prompt == "SYS_PREFIX"
        assert ds.generation_prompt == "GEN_PREFIX"

    def test_includes_system_prompt_when_configured(
        self, tmp_path, fake_verl_dataset_modules
    ):
        from foresightrlcore.methods.verl_sft_dataset import (
            FlatPromptResponseSFTDataset,
        )

        path = tmp_path / "data.parquet"
        pd.DataFrame({"prompt": ["hi"], "response": ["hi back"]}).to_parquet(path)

        ds = FlatPromptResponseSFTDataset(
            parquet_files=[str(path)],
            tokenizer=object(),
            config={
                "prompt_key": "prompt",
                "response_key": "response",
                "system_prompt": "You are helpful.",
            },
        )

        assert ds.messages[0][0] == {"role": "system", "content": "You are helpful."}

    def test_max_samples_below_total_truncates_without_shuffle(
        self, tmp_path, fake_verl_dataset_modules
    ):
        from foresightrlcore.methods.verl_sft_dataset import (
            FlatPromptResponseSFTDataset,
        )

        path = tmp_path / "data.parquet"
        pd.DataFrame(
            {
                "prompt": [f"p{i}" for i in range(5)],
                "response": [f"r{i}" for i in range(5)],
            }
        ).to_parquet(path)

        ds = FlatPromptResponseSFTDataset(
            parquet_files=[str(path)],
            tokenizer=object(),
            config={"prompt_key": "prompt", "response_key": "response"},
            max_samples=2,
        )

        assert len(ds.dataframe) == 2
        # shuffle defaults to False -> first N rows kept, not a random sample.
        assert ds.messages[0][0]["content"] == "p0"
        assert ds.messages[1][0]["content"] == "p1"

    def test_max_samples_with_shuffle_is_seeded_and_deterministic(
        self, tmp_path, fake_verl_dataset_modules
    ):
        from foresightrlcore.methods.verl_sft_dataset import (
            FlatPromptResponseSFTDataset,
        )

        path = tmp_path / "data.parquet"
        pd.DataFrame(
            {
                "prompt": [f"p{i}" for i in range(10)],
                "response": [f"r{i}" for i in range(10)],
            }
        ).to_parquet(path)

        def _build():
            return FlatPromptResponseSFTDataset(
                parquet_files=[str(path)],
                tokenizer=object(),
                config={
                    "prompt_key": "prompt",
                    "response_key": "response",
                    "shuffle": True,
                    "seed": 7,
                },
                max_samples=3,
            )

        ds1 = _build()
        ds2 = _build()
        assert len(ds1.dataframe) == 3
        assert ds1.messages == ds2.messages


class TestPackedSFTDatasetRealLogic:
    """Exercises _PackedSFTDataset._read_files_and_process/__getitem__ —

    the real pad_mode assertion and position_ids-reset wiring, not just
    the lazy-import boundary.
    """

    def test_read_files_and_process_sets_expected_state(
        self, tmp_path, fake_verl_dataset_modules
    ):
        from foresightrlcore.methods.verl_sft_dataset import PackedSFTDataset

        path = tmp_path / "packed.parquet"
        pd.DataFrame(
            {
                "input_ids": [[1, 2, 3], [4, 5]],
                "loss_mask": [[0, 0, 1], [0, 1]],
                "seq_start_id": [[0], [0]],
            }
        ).to_parquet(path)

        ds = PackedSFTDataset(
            parquet_files=[str(path)],
            tokenizer=object(),
            config={"pad_mode": "no_padding", "max_length": 100, "truncation": "error"},
        )

        assert len(ds.dataframe) == 2
        assert ds.messages == [None, None]
        assert ds.tools is None
        assert ds.enable_thinking is None

    def test_rejects_any_pad_mode_other_than_no_padding(
        self, tmp_path, fake_verl_dataset_modules
    ):
        from foresightrlcore.methods.verl_sft_dataset import PackedSFTDataset

        path = tmp_path / "packed.parquet"
        pd.DataFrame(
            {"input_ids": [[1, 2]], "loss_mask": [[0, 1]], "seq_start_id": [[0]]}
        ).to_parquet(path)

        with pytest.raises(AssertionError, match="no_padding"):
            PackedSFTDataset(
                parquet_files=[str(path)],
                tokenizer=object(),
                config={"pad_mode": "right", "max_length": 100, "truncation": "error"},
            )

    def test_getitem_resets_position_ids_at_document_boundaries(
        self, tmp_path, fake_verl_dataset_modules
    ):
        import torch

        from foresightrlcore.methods.verl_sft_dataset import PackedSFTDataset

        path = tmp_path / "packed.parquet"
        # One packed row: two documents of length 3 and 2.
        pd.DataFrame(
            {
                "input_ids": [[10, 11, 12, 13, 14]],
                "loss_mask": [[0, 1, 1, 0, 1]],
                "seq_start_id": [[0, 3]],
            }
        ).to_parquet(path)

        ds = PackedSFTDataset(
            parquet_files=[str(path)],
            tokenizer=object(),
            config={"pad_mode": "no_padding", "max_length": 100, "truncation": "error"},
        )

        item = ds[0]
        assert torch.equal(item["input_ids"], torch.tensor([10, 11, 12, 13, 14]))
        assert torch.equal(item["loss_mask"], torch.tensor([0, 1, 1, 0, 1]))
        assert torch.equal(item["position_ids"], torch.tensor([0, 1, 2, 0, 1]))
        for v in item.values():
            assert v.dtype == torch.long

    def test_max_samples_truncates_without_shuffle(
        self, tmp_path, fake_verl_dataset_modules
    ):
        from foresightrlcore.methods.verl_sft_dataset import PackedSFTDataset

        path = tmp_path / "packed.parquet"
        pd.DataFrame(
            {
                "input_ids": [[i] for i in range(5)],
                "loss_mask": [[1] for _ in range(5)],
                "seq_start_id": [[0] for _ in range(5)],
            }
        ).to_parquet(path)

        ds = PackedSFTDataset(
            parquet_files=[str(path)],
            tokenizer=object(),
            config={"pad_mode": "no_padding", "max_length": 100, "truncation": "error"},
            max_samples=2,
        )

        assert len(ds.dataframe) == 2
        assert list(ds.dataframe["input_ids"].iloc[0]) == [0]
        assert list(ds.dataframe["input_ids"].iloc[1]) == [1]

    def test_max_samples_with_shuffle_is_seeded_and_deterministic(
        self, tmp_path, fake_verl_dataset_modules
    ):
        from foresightrlcore.methods.verl_sft_dataset import PackedSFTDataset

        path = tmp_path / "packed.parquet"
        pd.DataFrame(
            {
                "input_ids": [[i] for i in range(10)],
                "loss_mask": [[1] for _ in range(10)],
                "seq_start_id": [[0] for _ in range(10)],
            }
        ).to_parquet(path)

        def _build():
            return PackedSFTDataset(
                parquet_files=[str(path)],
                tokenizer=object(),
                config={
                    "pad_mode": "no_padding",
                    "max_length": 100,
                    "truncation": "error",
                    "shuffle": True,
                    "seed": 7,
                },
                max_samples=3,
            )

        ds1 = _build()
        ds2 = _build()
        assert len(ds1.dataframe) == 3
        assert list(ds1.dataframe["input_ids"]) == list(ds2.dataframe["input_ids"])
