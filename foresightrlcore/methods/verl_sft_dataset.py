"""Datasets for verl's SFT trainer — two shapes, one file.

verl's ``create_sft_dataset()`` (``verl.trainer.sft_trainer``) resolves
``data.custom_cls.{path,name}`` via ``load_extern_object(path, name)`` and then
calls the resolved object as ``dataset_cls(parquet_files=..., tokenizer=...,
config=..., processor=..., max_samples=...)``. Nothing requires that object to
be a class — a plain callable with the same signature works just as well. We
use that to keep this module free of a module-level ``import verl``: verl is
only installed inside the training container (via the ``verl_sft`` recipe),
never in the dev/lint/test environment. This mirrors the lazy-import
convention already used for ``trl``/``peft`` in
:mod:`foresightrlcore.methods.base` and :mod:`foresightrlcore.methods.grpo` —
those modules import their heavy deps inside functions, never at module scope,
so this package stays importable (and unit-testable) without the GPU stack
installed.

This module provides two custom classes:

``FlatPromptResponseSFTDataset`` — used for
``training.sft_verl.data_format: "flat"`` (rows with separate ``prompt``/
``response``/optional ``system_prompt`` string fields; this class
synthesizes the ``messages`` list verl's tokenization expects). It
genuinely reads Parquet (matching the ``parquet_files`` parameter name for
real, not just by convention) — the *source* data is JSONL, but
:func:`foresightrlcore.methods.sft_verl._cache_jsonl_as_parquet` converts
it to Parquet once per node before this class ever sees a path, so every
rank reads the much-cheaper binary format instead of each redundantly
re-parsing the same JSON text 8 times per node.

``PackedSFTDataset`` — used for ``data_format: "packed"`` (rows produced
offline by ``rl-train tokenize-pack``, already fully tokenized: one packed
row = one or more original conversations concatenated end to end, with
``input_ids``/``loss_mask``/``seq_start_id`` columns; see
:mod:`foresightrlcore.tokenizer`). No tokenization or chat-template work
happens here at all — only ``position_ids`` construction, resetting to 0 at
every ``seq_start_id`` boundary. That reset is load-bearing, not cosmetic:
verl's own ``use_remove_padding=False`` path threads ``position_ids``
straight into the HF model's forward call unmodified, and (confirmed
directly against the real Gemma 4 modeling code,
``transformers/models/gemma4/modeling_gemma4.py``, and
``transformers.masking_utils.find_packed_sequence_indices``) HF's own
generic mask-preparation utilities detect packed-sequence boundaries
*purely from where position_ids resets* and build the correct block-
diagonal + causal (+ sliding-window) mask automatically. Without the
reset, a multi-document packed row would look like one long monotonic
sequence to the model, and attention would silently leak across the
packed sub-sequences — wrong gradients, no crash. No custom attention_mask
or verl-engine change is needed; this dataset-level fix is sufficient.

For ``data_format: "messages"`` (rows that already have a full ``messages``
list) there is deliberately no custom class here at all: once the same
caching step has produced genuine Parquet with a ``messages`` column,
verl's own built-in ``MultiTurnSFTDataset`` (verl's default when
``data.custom_cls`` is left unset) reads it directly — no synthesis needed,
so there's nothing for a custom class to do. See
:mod:`foresightrlcore.methods.sft_verl`'s ``run()`` for where
``data.custom_cls`` is conditionally set for the "flat" and "packed" cases.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _dataset_class() -> type:
    """Build (once) a ``MultiTurnSFTDataset`` subclass reading flat Parquet
    (cached from the source JSONL — see module docstring).

    verl's ``MultiTurnSFTDataset.__getitem__`` reads
    ``self.dataframe.iloc[item].to_dict()`` and passes it through
    ``self._build_messages(row_dict)``, which looks up
    ``example[self.messages_key]`` (default column name ``"messages"``) — it
    does **not** use the ``self.messages`` list for anything besides
    ``__len__``. So the only thing this override needs to do is populate
    ``self.dataframe`` with a ``messages_key`` column built from the flat
    prompt/response fields; every downstream per-turn tokenization /
    loss-masking (loss over the *full* assistant turn, matching the "full
    assistant turn" loss-masking decision for this recipe) / thinking-template
    / Ulysses-SP-aware collation logic in the base class is reused unchanged.
    """
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

    class _FlatPromptResponseSFTDataset(MultiTurnSFTDataset):
        def __init__(
            self,
            parquet_files: Any,
            tokenizer: Any,
            config: Any,
            processor: Any = None,
            max_samples: int = -1,
        ) -> None:
            cfg = config or {}
            # Captured before super().__init__() runs _read_files_and_process()
            # (which dispatches polymorphically to our override below).
            self._prompt_key = cfg.get("prompt_key", "prompt")
            self._response_key = cfg.get("response_key", "response")
            self._system_prompt_text = cfg.get("system_prompt", None)
            super().__init__(
                parquet_files=parquet_files,
                tokenizer=tokenizer,
                config=config,
                processor=processor,
                max_samples=max_samples,
            )

        def _read_files_and_process(self) -> None:
            import random

            import pandas as pd
            from verl.utils.py_functional import (
                convert_nested_value_to_list_recursive,
            )
            from verl.utils.tokenizer.chat_template import (
                extract_system_prompt_and_generation,
            )

            # Genuinely parquet now (see module docstring): sft_verl.py's
            # _cache_jsonl_as_parquet() converts the source JSONL once per
            # node before this class ever sees a path.
            frames = [pd.read_parquet(path) for path in self.parquet_files]
            df = pd.concat(frames, ignore_index=True)

            total = len(df)
            logger.info(f"dataset len: {total}")
            if self.max_samples > 0 and self.max_samples < total:
                indices: list[int]
                if self.shuffle:
                    rnd = (
                        random.Random(self.seed)
                        if self.seed is not None
                        else random.Random()
                    )
                    indices = rnd.sample(range(total), self.max_samples)
                else:
                    indices = list(range(self.max_samples))
                df = df.iloc[indices]
                logger.info(
                    f"selected {self.max_samples} random samples out of {total}"
                )

            prompt_key, response_key = self._prompt_key, self._response_key
            system_text = self._system_prompt_text

            def _row_to_messages(row: Any) -> list[dict[str, str]]:
                msgs = []
                if system_text:
                    msgs.append({"role": "system", "content": system_text})
                msgs.append({"role": "user", "content": row[prompt_key]})
                msgs.append({"role": "assistant", "content": row[response_key]})
                return msgs

            df[self.messages_key] = df.apply(_row_to_messages, axis=1)
            self.dataframe = df.reset_index(drop=True)

            self.messages = (
                self.dataframe[self.messages_key]
                .apply(convert_nested_value_to_list_recursive)
                .tolist()
            )
            # Flat schema carries no tool calls or per-row thinking flag.
            # self.enable_thinking stays None deliberately: thinking mode is
            # forced on via the STATIC data.apply_chat_template_kwargs (set in
            # foresightrlcore.methods.sft_verl), not this per-row mechanism —
            # confirmed directly against the real tokenizer that
            # enable_thinking_default (dynamic, per-row) desyncs the
            # self.system_prompt length
            # measurement below from what's actually rendered per-turn,
            # corrupting the loss on every assistant turn. Do not change this
            # without re-verifying against the real tokenizer.
            self.tools = None
            self.enable_thinking = None

            self.system_prompt, self.generation_prompt = (
                extract_system_prompt_and_generation(
                    self.tokenizer, **self.apply_chat_template_kwargs
                )
            )

    return _FlatPromptResponseSFTDataset


def FlatPromptResponseSFTDataset(
    parquet_files: Any,
    tokenizer: Any,
    config: Any,
    processor: Any = None,
    max_samples: int = -1,
) -> Any:
    """Entry point resolved by verl's ``data.custom_cls`` mechanism."""
    cls = _dataset_class()
    return cls(
        parquet_files=parquet_files,
        tokenizer=tokenizer,
        config=config,
        processor=processor,
        max_samples=max_samples,
    )


@lru_cache(maxsize=1)
def _packed_dataset_class() -> type:
    """Build (once) a ``MultiTurnSFTDataset`` subclass reading pre-packed,
    pre-tokenized Parquet directly — see module docstring for why the
    per-document ``position_ids`` reset is the only real logic here.
    """
    from verl.utils.dataset.dataset_utils import DatasetPadMode
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

    class _PackedSFTDataset(MultiTurnSFTDataset):
        def _read_files_and_process(self) -> None:
            import random

            import pandas as pd

            assert self.pad_mode == DatasetPadMode.NO_PADDING, (
                "PackedSFTDataset only supports pad_mode='no_padding' — "
                "verl's SFT forward path asserts this regardless of "
                "use_remove_padding, and packing correctness here depends "
                "on position_ids resets, which the 'right' pad_mode branch "
                "in the base class does not compute."
            )

            frames = [pd.read_parquet(path) for path in self.parquet_files]
            df = pd.concat(frames, ignore_index=True)

            total = len(df)
            logger.info(f"dataset len: {total}")
            if self.max_samples > 0 and self.max_samples < total:
                indices: list[int]
                if self.shuffle:
                    rnd = (
                        random.Random(self.seed)
                        if self.seed is not None
                        else random.Random()
                    )
                    indices = rnd.sample(range(total), self.max_samples)
                else:
                    indices = list(range(self.max_samples))
                df = df.iloc[indices]
                logger.info(
                    f"selected {self.max_samples} random samples out of {total}"
                )

            self.dataframe = df.reset_index(drop=True)
            # Only __len__ reads this (len(self.messages)) — the real
            # per-row data is read directly from self.dataframe in
            # __getitem__ below, since it's already tokenized.
            self.messages = [None] * len(self.dataframe)
            self.tools = None
            self.enable_thinking = None

        def __getitem__(self, item: int) -> dict[str, Any]:
            import torch

            row = self.dataframe.iloc[item]
            input_ids, loss_mask, position_ids = truncate_and_reset_positions(
                list(row["input_ids"]),
                list(row["loss_mask"]),
                list(row["seq_start_id"]),
                self.max_length,
                self.truncation,
            )
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "position_ids": torch.tensor(position_ids, dtype=torch.long),
                "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
            }

    return _PackedSFTDataset


def truncate_and_reset_positions(
    input_ids: list[int],
    loss_mask: list[int],
    seq_start_id: list[int],
    max_length: int,
    truncation: str,
) -> tuple[list[int], list[int], list[int]]:
    """Pure-Python core of ``PackedSFTDataset.__getitem__``: truncate a
    packed row to ``max_length`` per ``truncation`` (mirroring
    ``MultiTurnSFTDataset``'s own truncation semantics), and compute
    ``position_ids`` that reset to 0 at every ``seq_start_id`` boundary —
    the mechanism (see module docstring) that keeps HF's own mask
    preparation from letting attention cross packed sub-sequences.

    No torch/verl dependency, so this is directly unit-testable without
    either installed — the one thing here worth testing thoroughly, since
    correctness depends entirely on it.
    """
    sequence_length = len(input_ids)
    if sequence_length > max_length:
        if truncation == "error":
            raise ValueError(f"{sequence_length=} is larger than {max_length=}")
        elif truncation == "left":
            keep_from = sequence_length - max_length
            input_ids = input_ids[keep_from:]
            loss_mask = loss_mask[keep_from:]
            seq_start_id = sorted(
                {max(0, s - keep_from) for s in seq_start_id if s >= keep_from} | {0}
            )
        elif truncation == "right":
            input_ids = input_ids[:max_length]
            loss_mask = loss_mask[:max_length]
            seq_start_id = [s for s in seq_start_id if s < max_length]
        else:
            raise ValueError(f"Unknown truncation method {truncation}")
        sequence_length = len(input_ids)

    position_ids = [0] * sequence_length
    boundaries = seq_start_id + [sequence_length]
    for start, end in zip(boundaries, boundaries[1:]):
        for offset in range(end - start):
            position_ids[start + offset] = offset

    return input_ids, loss_mask, position_ids


def PackedSFTDataset(
    parquet_files: Any,
    tokenizer: Any,
    config: Any,
    processor: Any = None,
    max_samples: int = -1,
) -> Any:
    """Entry point resolved by verl's ``data.custom_cls`` mechanism."""
    cls = _packed_dataset_class()
    return cls(
        parquet_files=parquet_files,
        tokenizer=tokenizer,
        config=config,
        processor=processor,
        max_samples=max_samples,
    )
