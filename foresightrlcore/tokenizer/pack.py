"""Greedy bin-packing of tokenized rows into fixed-length packed sequences.

Produces the same ``input_ids`` / ``loss_mask`` / ``seq_start_id`` shape as
the NeMo/Megatron-Bridge packer (``seq_start_id`` is the sub-sequence start
offset array — the boundary information an attention mechanism needs to
avoid attending across packed examples), implemented from scratch with a
plain best-fit-decreasing heuristic — no Megatron/NeMo dependency.

Rows over ``pack_size`` on their own are dropped (matches the source
recipe's ``truncation: error`` philosophy — fail loud on outliers via a
count in the returned metadata, not a silent truncation).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TokenizedRow:
    input_ids: list[int]
    loss_mask: list[int]

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.loss_mask):
            raise ValueError(
                f"input_ids ({len(self.input_ids)}) and loss_mask "
                f"({len(self.loss_mask)}) length mismatch"
            )


@dataclass
class PackedRow:
    input_ids: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    seq_start_id: list[int] = field(default_factory=list)
    _remaining: int = 0

    def add(self, row: TokenizedRow) -> None:
        self.seq_start_id.append(len(self.input_ids))
        self.input_ids.extend(row.input_ids)
        self.loss_mask.extend(row.loss_mask)
        self._remaining -= len(row.input_ids)

    @property
    def length(self) -> int:
        return len(self.input_ids)


def pack_rows(rows: list[TokenizedRow], pack_size: int) -> tuple[list[PackedRow], int]:
    """Best-fit-decreasing bin-packing.

    Sorting longest-first and always placing each row into the
    open bin with the *least* remaining space that still fits (falling
    back to opening a new bin) is the standard best-fit-decreasing bin-
    packing heuristic — good packing efficiency, and simple enough to
    reason about correctness for (no row ever crosses a bin boundary,
    no token is ever dropped or duplicated).

    Returns ``(packed_rows, num_oversized_dropped)`` — rows longer than
    ``pack_size`` on their own cannot be packed at all and are dropped;
    the caller decides whether that count is acceptable.
    """
    oversized = sum(1 for r in rows if len(r.input_ids) > pack_size)
    fittable = [r for r in rows if len(r.input_ids) <= pack_size]
    fittable.sort(key=lambda r: len(r.input_ids), reverse=True)

    open_bins: list[PackedRow] = []
    closed_bins: list[PackedRow] = []

    for row in fittable:
        row_len = len(row.input_ids)
        best_idx = None
        best_remaining = None
        for idx, bin_ in enumerate(open_bins):
            if bin_._remaining >= row_len:
                if best_remaining is None or bin_._remaining < best_remaining:
                    best_idx = idx
                    best_remaining = bin_._remaining
        if best_idx is None:
            new_bin = PackedRow(_remaining=pack_size)
            new_bin.add(row)
            open_bins.append(new_bin)
        else:
            open_bins[best_idx].add(row)
            # A bin with no room left for even the shortest remaining rows
            # can be closed out immediately to keep the open-bin scan short.
            if open_bins[best_idx]._remaining <= 0:
                closed_bins.append(open_bins.pop(best_idx))

    closed_bins.extend(open_bins)
    return closed_bins, oversized


def compute_metadata(
    num_raw_rows: int,
    packed_rows: list[PackedRow],
    pack_size: int,
    num_oversized_dropped: int,
) -> dict[str, float | int]:
    """Mirrors the ``<pack_size>_metadata.jsonl`` sidecar Megatron-Bridge
    writes (``packing_factor``, ``packing_efficiency``, ...) so packed
    artifacts from either pipeline are comparable at a glance.
    """
    if not packed_rows:
        return {
            "pack_size": pack_size,
            "num_raw_rows": num_raw_rows,
            "num_packed_rows": 0,
            "num_oversized_dropped": num_oversized_dropped,
            "packing_factor": 0.0,
            "packing_efficiency": 0.0,
            "min_packed_seqlen": 0,
            "max_packed_seqlen": 0,
        }

    lengths = [p.length for p in packed_rows]
    packable_rows = num_raw_rows - num_oversized_dropped
    return {
        "pack_size": pack_size,
        "num_raw_rows": num_raw_rows,
        "num_packed_rows": len(packed_rows),
        "num_oversized_dropped": num_oversized_dropped,
        "packing_factor": round(packable_rows / len(packed_rows), 4),
        "packing_efficiency": round(
            100 * sum(lengths) / (pack_size * len(packed_rows)), 2
        ),
        "min_packed_seqlen": min(lengths),
        "max_packed_seqlen": max(lengths),
    }
