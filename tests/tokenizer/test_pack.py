from foresightrlcore.tokenizer.pack import TokenizedRow, compute_metadata, pack_rows


def _row(n: int, tag: int) -> TokenizedRow:
    """A row of length n whose token ids are all `tag`, so packed output can
    be traced back to its source row for correctness checks.
    """
    return TokenizedRow(input_ids=[tag] * n, loss_mask=[1] * n)


def test_tokenized_row_rejects_length_mismatch():
    try:
        TokenizedRow(input_ids=[1, 2, 3], loss_mask=[0, 1])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_no_row_crosses_a_bin_boundary():
    rows = [_row(3000, 1), _row(3000, 2), _row(3000, 3), _row(3000, 4)]
    packed, oversized = pack_rows(rows, pack_size=8192)

    assert oversized == 0
    for bin_ in packed:
        assert bin_.length <= 8192


def test_every_token_is_preserved_exactly_once():
    rows = [_row(3000, 1), _row(3000, 2), _row(3000, 3), _row(3000, 4)]
    packed, oversized = pack_rows(rows, pack_size=8192)

    total_in = sum(len(r.input_ids) for r in rows)
    total_out = sum(bin_.length for bin_ in packed)
    assert total_in == total_out

    # every source row's tag appears contiguously in exactly one bin
    tags_seen = []
    for bin_ in packed:
        for start, end in zip(bin_.seq_start_id, bin_.seq_start_id[1:] + [bin_.length]):
            segment = bin_.input_ids[start:end]
            assert len(set(segment)) == 1  # one source row's tag, uninterrupted
            tags_seen.append(segment[0])
    assert sorted(tags_seen) == [1, 2, 3, 4]


def test_seq_start_id_offsets_are_correct():
    # Best-fit-decreasing processes longest rows first, so the 200-token
    # row lands first in the bin regardless of input order.
    rows = [_row(100, 1), _row(200, 2)]
    packed, _ = pack_rows(rows, pack_size=8192)

    assert len(packed) == 1
    bin_ = packed[0]
    assert bin_.seq_start_id == [0, 200]
    assert bin_.length == 300


def test_oversized_rows_are_dropped_not_truncated():
    rows = [_row(100, 1), _row(9000, 2)]  # second row alone exceeds pack_size
    packed, oversized = pack_rows(rows, pack_size=8192)

    assert oversized == 1
    assert sum(bin_.length for bin_ in packed) == 100
    for bin_ in packed:
        assert 2 not in bin_.input_ids


def test_compute_metadata_formulas():
    rows = [_row(4000, 1), _row(4000, 2), _row(2000, 3)]
    packed, oversized = pack_rows(rows, pack_size=8192)
    meta = compute_metadata(
        num_raw_rows=len(rows),
        packed_rows=packed,
        pack_size=8192,
        num_oversized_dropped=oversized,
    )

    assert meta["num_raw_rows"] == 3
    assert meta["num_oversized_dropped"] == 0
    assert meta["num_packed_rows"] == len(packed)
    assert meta["packing_factor"] == round(3 / len(packed), 4)
    total_tokens = 4000 + 4000 + 2000
    expected_efficiency = round(100 * total_tokens / (8192 * len(packed)), 2)
    assert meta["packing_efficiency"] == expected_efficiency


def test_compute_metadata_handles_empty_input():
    meta = compute_metadata(
        num_raw_rows=0, packed_rows=[], pack_size=8192, num_oversized_dropped=0
    )
    assert meta["num_packed_rows"] == 0
    assert meta["packing_efficiency"] == 0.0
