"""Offline tokenize + pack: a generic, model-agnostic preprocessing utility.

Produces the same ``input_ids`` / ``loss_mask`` / ``seq_start_id`` packed
Parquet shape as the NeMo/Megatron-Bridge SFT packing pipeline (see a real
example under ``qwen35_122b_sft_100k_8k_packed_cp2`` in S3) — but implemented
from scratch with plain ``transformers.AutoTokenizer``, no Megatron/NeMo
dependency. Runs entirely on CPU; intended to be invoked once, offline
(``rl-train tokenize-pack``), before a training job ever starts, so training
jobs read pre-packed Parquet directly instead of re-tokenizing every run.

Modules:
    loss_mask — template-agnostic per-turn loss masking via incremental
        tokenization (works even when a model's chat template has no
        ``{% generation %}`` block — confirmed true for both Qwen3.5's and
        Gemma 4's real shipped templates).
    pack — greedy bin-packing of tokenized rows into fixed-length packed
        sequences, tracking each sub-sequence's start offset.
    io — JSONL shard reading, Parquet + metadata sidecar writing.
    cli — ``rl-train tokenize-pack`` argument parsing and orchestration
        (multiprocessing across rows).
"""

from __future__ import annotations
