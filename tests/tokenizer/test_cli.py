import argparse
import json
import os
from unittest.mock import MagicMock, patch

from foresightrlcore.tokenizer import cli


def _parse_tokenize_pack_args(extra_args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    cli.build_parser(subparsers)
    return parser.parse_args(
        ["tokenize-pack", "--input", "in", "--tokenizer", "tok", "--output", "out"]
        + extra_args
    )


def test_enable_thinking_defaults_to_true():
    # Matches model.enable_thinking's own training-schema default — packed
    # rows are never re-tokenized, so a mismatched default here would
    # silently train on the wrong chat-template rendering.
    args = _parse_tokenize_pack_args([])
    assert args.enable_thinking is True


def test_no_enable_thinking_flag_turns_it_off():
    args = _parse_tokenize_pack_args(["--no-enable-thinking"])
    assert args.enable_thinking is False


def test_allow_drop_oversized_defaults_to_false():
    args = _parse_tokenize_pack_args([])
    assert args.allow_drop_oversized is False


def test_allow_drop_oversized_flag_sets_true():
    args = _parse_tokenize_pack_args(["--allow-drop-oversized"])
    assert args.allow_drop_oversized is True


def test_allow_tokenize_failures_defaults_to_false():
    args = _parse_tokenize_pack_args([])
    assert args.allow_tokenize_failures is False


def test_allow_tokenize_failures_flag_sets_true():
    args = _parse_tokenize_pack_args(["--allow-tokenize-failures"])
    assert args.allow_tokenize_failures is True


def test_resolve_tokenizer_path_passes_through_local_path():
    assert cli._resolve_tokenizer_path("/local/tokenizer/dir") == "/local/tokenizer/dir"


def test_resolve_tokenizer_path_downloads_s3_prefix(tmp_path, monkeypatch):
    """Regression guard: model.name/--tokenizer often point at the SAME

    S3 prefix that also holds the full model checkpoint (tokenizer.json
    sitting flat alongside model-*.safetensors, tens of GB) — reproduced
    directly: pointing --tokenizer at Muse Glimmer 30B's model prefix
    downloaded 55GB of weight shards just to reach the tokenizer. This
    call must pass exclude_extensions/max_file_size_bytes so
    download_dir_from_s3 filters those out.
    """
    captured = {}

    def fake_download(
        s3_uri, local_dir, *, exclude_extensions=(), max_file_size_bytes=None
    ):
        captured["s3_uri"] = s3_uri
        captured["local_dir"] = local_dir
        captured["exclude_extensions"] = exclude_extensions
        captured["max_file_size_bytes"] = max_file_size_bytes

    monkeypatch.setattr(
        "foresightrlcore.tokenizer.io.download_dir_from_s3", fake_download
    )

    resolved = cli._resolve_tokenizer_path("s3://my-bucket/models/tok/")

    assert captured["s3_uri"] == "s3://my-bucket/models/tok/"
    assert ".safetensors" in captured["exclude_extensions"]
    assert ".bin" in captured["exclude_extensions"]
    assert captured["max_file_size_bytes"] is not None
    assert resolved == captured["local_dir"]


def test_resolve_tokenizer_path_registers_cleanup_of_temp_dir(tmp_path, monkeypatch):
    """Regression guard: tempfile.mkdtemp() here was never cleaned up —

    every S3 --tokenizer run left its downloaded temp dir in /tmp forever.
    """
    import shutil

    monkeypatch.setattr(
        "foresightrlcore.tokenizer.io.download_dir_from_s3", lambda *a, **k: None
    )
    registered = []
    monkeypatch.setattr(
        "atexit.register", lambda fn, *a, **k: registered.append((fn, a, k))
    )

    resolved = cli._resolve_tokenizer_path("s3://my-bucket/models/tok/")

    assert len(registered) == 1
    fn, args, kwargs = registered[0]
    assert fn is shutil.rmtree
    assert args == (resolved,)
    assert kwargs == {"ignore_errors": True}


def test_resolve_input_path_passes_through_local_path():
    assert cli._resolve_input_path("/local/data/dir") == "/local/data/dir"


def test_resolve_input_path_downloads_s3_prefix(tmp_path, monkeypatch):
    """Regression guard: --input s3://... previously raised

    `FileNotFoundError: [Errno 2] No such file or directory` — read_jsonl_shards()
    only understands local paths, and nothing resolved an s3:// --input to
    a local one first (unlike --tokenizer, which already went through
    _resolve_tokenizer_path).
    """
    captured = {}

    def fake_download(s3_uri, local_dir):
        captured["s3_uri"] = s3_uri
        captured["local_dir"] = local_dir

    monkeypatch.setattr(
        "foresightrlcore.tokenizer.io.download_dir_from_s3", fake_download
    )

    resolved = cli._resolve_input_path("s3://my-bucket/data/train/")

    assert captured["s3_uri"] == "s3://my-bucket/data/train/"
    assert resolved == captured["local_dir"]


def test_resolve_input_path_registers_cleanup_of_temp_dir(tmp_path, monkeypatch):
    """Regression guard: tempfile.mkdtemp() here was never cleaned up —

    every S3 --input run left a full copy of the downloaded dataset
    (potentially GBs, not a small tokenizer) in /tmp forever.
    """
    import shutil

    monkeypatch.setattr(
        "foresightrlcore.tokenizer.io.download_dir_from_s3", lambda *a, **k: None
    )
    registered = []
    monkeypatch.setattr(
        "atexit.register", lambda fn, *a, **k: registered.append((fn, a, k))
    )

    resolved = cli._resolve_input_path("s3://my-bucket/data/train/")

    assert len(registered) == 1
    fn, args, kwargs = registered[0]
    assert fn is shutil.rmtree
    assert args == (resolved,)
    assert kwargs == {"ignore_errors": True}


def test_tokenize_pack_end_to_end_with_s3_input(tmp_path, monkeypatch):
    """Full _cmd_tokenize_pack run with --input as an s3:// URI — the

    actual reported bug: `read_jsonl_shards` only understands local
    paths, so an unresolved s3:// --input raised `FileNotFoundError:
    [Errno 2] No such file or directory` before this was wired in.
    """
    shard_content = (
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        )
        + "\n"
    )

    def fake_download(s3_uri, local_dir):
        with open(os.path.join(local_dir, "shard.jsonl"), "w") as f:
            f.write(shard_content)

    monkeypatch.setattr(
        "foresightrlcore.tokenizer.io.download_dir_from_s3", fake_download
    )
    monkeypatch.setattr(cli, "_init_worker", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_tokenize_one", lambda row: ([1, 2, 3], [0, 0, 1]))

    output_dir = tmp_path / "out"
    args = _build_tokenize_pack_args(
        "s3://my-bucket/data/train/", output_dir, allow_drop_oversized=False
    )

    assert cli._cmd_tokenize_pack(args) == 0
    assert (output_dir / "training_50.idx.parquet").exists()


def test_split_train_eval_absolute_count():
    rows = [{"i": i} for i in range(10)]
    train, eval_ = cli._split_train_eval(rows, eval_split_size=3, seed=0)
    assert len(train) == 7
    assert len(eval_) == 3
    assert {r["i"] for r in train} | {r["i"] for r in eval_} == set(range(10))


def test_split_train_eval_fraction():
    rows = [{"i": i} for i in range(20)]
    train, eval_ = cli._split_train_eval(rows, eval_split_size=0.25, seed=0)
    assert len(eval_) == 5
    assert len(train) == 15


def test_split_train_eval_disabled_when_zero():
    rows = [{"i": i} for i in range(5)]
    train, eval_ = cli._split_train_eval(rows, eval_split_size=0, seed=0)
    assert train == rows
    assert eval_ == []


def test_split_train_eval_caps_at_dataset_size():
    rows = [{"i": i} for i in range(3)]
    train, eval_ = cli._split_train_eval(rows, eval_split_size=10, seed=0)
    assert len(eval_) == 3
    assert train == []


def test_tokenize_pack_end_to_end(tmp_path, monkeypatch):
    """Exercises the full CLI orchestration (read -> split -> tokenize ->
    pack -> write parquet + metadata) with the tokenizer/loss-mask step
    faked out, since that's already covered directly in test_loss_mask.py
    and doesn't need a real HF model here.
    """
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    lengths = (10, 20, 30, 40)
    (input_dir / "shard.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "x" * n},
                        {"role": "assistant", "content": "y" * n},
                    ]
                }
            )
            for n in lengths
        )
    )
    output_dir = tmp_path / "out"

    def fake_init_worker(*args, **kwargs):
        pass

    def fake_tokenize_one(row):
        n = len(row["messages"][0]["content"])
        return [1] * n, [0] * n

    monkeypatch.setattr(cli, "_init_worker", fake_init_worker)
    monkeypatch.setattr(cli, "_tokenize_one", fake_tokenize_one)

    args = argparse.Namespace(
        input=str(input_dir),
        tokenizer="unused",
        output=str(output_dir),
        pack_size=50,
        allow_drop_oversized=False,
        allow_tokenize_failures=False,
        data_format="messages",
        prompt_key="prompt",
        response_key="response",
        system_prompt=None,
        enable_thinking=False,
        thinking_kwarg="enable_thinking",
        eval_split_size=0,
        split_name="training",
        seed=0,
        num_workers=1,
        trust_remote_code=False,
        upload_to=None,
    )

    rc = cli._cmd_tokenize_pack(args)
    assert rc == 0

    import pandas as pd

    df = pd.read_parquet(output_dir / "training_50.idx.parquet")
    assert len(df) == 2  # (40+10) and (30+20) each pack exactly to 50
    assert set(df.columns) == {"input_ids", "loss_mask", "seq_start_id"}
    for row_len in df["input_ids"].apply(len):
        assert row_len == 50

    metadata = json.loads(
        (output_dir / "50_metadata.jsonl").read_text().splitlines()[0]
    )
    assert metadata[0]["split"] == "training"
    assert metadata[0]["num_raw_rows"] == 4
    assert metadata[0]["num_packed_rows"] == 2
    assert metadata[0]["packing_efficiency"] == 100.0
    assert metadata[0]["enable_thinking"] is False
    assert metadata[0]["thinking_kwarg"] == "enable_thinking"
    assert (output_dir / "validation_50.idx.parquet").exists() is False


def _build_tokenize_pack_args(
    input_dir, output_dir, *, allow_drop_oversized, allow_tokenize_failures=False
):
    return argparse.Namespace(
        input=str(input_dir),
        tokenizer="unused",
        output=str(output_dir),
        pack_size=50,
        allow_drop_oversized=allow_drop_oversized,
        allow_tokenize_failures=allow_tokenize_failures,
        data_format="messages",
        prompt_key="prompt",
        response_key="response",
        system_prompt=None,
        enable_thinking=False,
        thinking_kwarg="enable_thinking",
        eval_split_size=0,
        split_name="training",
        seed=0,
        num_workers=1,
        trust_remote_code=False,
        upload_to=None,
    )


def _write_rows_with_one_oversized(input_dir):
    input_dir.mkdir()
    # 10 packs fine on its own; 60 alone exceeds pack_size=50 and can never
    # be packed no matter what else is in the batch.
    lengths = (10, 60)
    (input_dir / "shard.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "x" * n},
                        {"role": "assistant", "content": "y" * n},
                    ]
                }
            )
            for n in lengths
        )
    )


def _fake_tokenize_one(row):
    n = len(row["messages"][0]["content"])
    return [1] * n, [0] * n


def test_tokenize_pack_oversized_drop_fails_by_default(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    _write_rows_with_one_oversized(input_dir)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(cli, "_init_worker", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_tokenize_one", _fake_tokenize_one)

    args = _build_tokenize_pack_args(input_dir, output_dir, allow_drop_oversized=False)
    rc = cli._cmd_tokenize_pack(args)
    assert rc == 1

    # Metadata/parquet are still written — a partial drop isn't the
    # "produced zero packed rows" case, just a non-zero exit signal.
    metadata = json.loads(
        (output_dir / "50_metadata.jsonl").read_text().splitlines()[0]
    )
    assert metadata[0]["num_oversized_dropped"] == 1
    assert (output_dir / "training_50.idx.parquet").exists()


def test_tokenize_pack_skips_upload_when_oversized_drop_rejected(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    _write_rows_with_one_oversized(input_dir)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(cli, "_init_worker", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_tokenize_one", _fake_tokenize_one)

    args = _build_tokenize_pack_args(input_dir, output_dir, allow_drop_oversized=False)
    args.upload_to = "s3://my-bucket/prefix/"

    mock_client = MagicMock()
    with patch("boto3.client", return_value=mock_client):
        rc = cli._cmd_tokenize_pack(args)

    assert rc == 1
    mock_client.upload_file.assert_not_called()


def test_tokenize_pack_oversized_drop_allowed_with_flag(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    _write_rows_with_one_oversized(input_dir)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(cli, "_init_worker", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_tokenize_one", _fake_tokenize_one)

    args = _build_tokenize_pack_args(input_dir, output_dir, allow_drop_oversized=True)
    rc = cli._cmd_tokenize_pack(args)
    assert rc == 0


def _write_rows_with_one_bad_row(input_dir):
    input_dir.mkdir()
    # Both lengths fit well within pack_size=50 on their own — unlike
    # _write_rows_with_one_oversized, this exercises the OTHER silent-loss
    # path: a per-row exception during tokenization, not an oversized drop.
    lengths = (10, 20)
    (input_dir / "shard.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "x" * n},
                        {"role": "assistant", "content": "y" * n},
                    ]
                }
            )
            for n in lengths
        )
    )


def _fake_tokenize_one_with_one_failure(row):
    # Mirrors the real _tokenize_one's own contract (catches the
    # tokenization exception itself and returns None) rather than raising
    # — the real function never lets an exception propagate out of the
    # map/pool.map call, so a fake standing in for it must not either.
    n = len(row["messages"][0]["content"])
    if n == 20:
        return None
    return [1] * n, [0] * n


def test_tokenize_pack_tokenize_failure_fails_by_default(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    _write_rows_with_one_bad_row(input_dir)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(cli, "_init_worker", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_tokenize_one", _fake_tokenize_one_with_one_failure)

    args = _build_tokenize_pack_args(input_dir, output_dir, allow_drop_oversized=False)
    rc = cli._cmd_tokenize_pack(args)
    assert rc == 1

    # Metadata/parquet are still written — a partial failure isn't the
    # "produced zero packed rows" case, just a non-zero exit signal.
    metadata = json.loads(
        (output_dir / "50_metadata.jsonl").read_text().splitlines()[0]
    )
    assert metadata[0]["num_tokenize_failed"] == 1
    assert (output_dir / "training_50.idx.parquet").exists()


def test_tokenize_pack_tokenize_failure_allowed_with_flag(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    _write_rows_with_one_bad_row(input_dir)
    output_dir = tmp_path / "out"

    monkeypatch.setattr(cli, "_init_worker", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_tokenize_one", _fake_tokenize_one_with_one_failure)

    args = _build_tokenize_pack_args(
        input_dir,
        output_dir,
        allow_drop_oversized=False,
        allow_tokenize_failures=True,
    )
    rc = cli._cmd_tokenize_pack(args)
    assert rc == 0


def test_tokenize_pack_records_enable_thinking_true_in_metadata(tmp_path, monkeypatch):
    """Packed data is never re-tokenized at train time, so whether

    --enable-thinking was passed must be discoverable from the packed
    output itself, not just from whoever's shell history ran the command.
    """
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "shard.jsonl").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        )
        + "\n"
    )
    output_dir = tmp_path / "out"

    monkeypatch.setattr(cli, "_init_worker", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_tokenize_one", lambda row: ([1, 2, 3], [0, 0, 1]))

    args = argparse.Namespace(
        input=str(input_dir),
        tokenizer="unused",
        output=str(output_dir),
        pack_size=50,
        allow_drop_oversized=False,
        allow_tokenize_failures=False,
        data_format="messages",
        prompt_key="prompt",
        response_key="response",
        system_prompt=None,
        enable_thinking=True,
        thinking_kwarg="enable_thinking",
        eval_split_size=0,
        split_name="training",
        seed=0,
        num_workers=1,
        trust_remote_code=False,
        upload_to=None,
    )

    assert cli._cmd_tokenize_pack(args) == 0

    metadata = json.loads(
        (output_dir / "50_metadata.jsonl").read_text().splitlines()[0]
    )
    assert metadata[0]["enable_thinking"] is True


def test_tokenize_pack_split_name_labels_a_separately_packed_eval_corpus(
    tmp_path, monkeypatch
):
    """A corpus packed on its own (--eval-split-size 0) must be labeled

    "validation" via --split-name when it's semantically an eval set —
    otherwise it's always named training_*.idx.parquet regardless of what
    the data actually is, and rl-train train's data_format=packed path
    globs data.eval_path for validation_*.idx.parquet specifically.
    """
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "shard.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "x"}]}) + "\n"
    )
    output_dir = tmp_path / "out"

    monkeypatch.setattr(cli, "_init_worker", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_tokenize_one", lambda row: ([1, 2, 3], [0, 0, 1]))

    args = argparse.Namespace(
        input=str(input_dir),
        tokenizer="unused",
        output=str(output_dir),
        pack_size=50,
        allow_drop_oversized=False,
        allow_tokenize_failures=False,
        data_format="messages",
        prompt_key="prompt",
        response_key="response",
        system_prompt=None,
        enable_thinking=False,
        thinking_kwarg="enable_thinking",
        eval_split_size=0,
        split_name="validation",
        seed=0,
        num_workers=1,
        trust_remote_code=False,
        upload_to=None,
    )

    assert cli._cmd_tokenize_pack(args) == 0

    assert (output_dir / "validation_50.idx.parquet").exists()
    assert not (output_dir / "training_50.idx.parquet").exists()
    metadata = json.loads(
        (output_dir / "50_metadata.jsonl").read_text().splitlines()[0]
    )
    assert metadata[0]["split"] == "validation"


def test_tokenize_pack_rejects_split_name_validation_with_nonzero_eval_split_size(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "shard.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "x"}]}) + "\n"
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("should not tokenize on a rejected argument combo")

    monkeypatch.setattr(cli, "_init_worker", fail_if_called)
    monkeypatch.setattr(cli, "_tokenize_one", fail_if_called)

    args = argparse.Namespace(
        input=str(input_dir),
        tokenizer="unused",
        output=str(tmp_path / "out"),
        pack_size=50,
        allow_drop_oversized=False,
        allow_tokenize_failures=False,
        data_format="messages",
        prompt_key="prompt",
        response_key="response",
        system_prompt=None,
        enable_thinking=False,
        thinking_kwarg="enable_thinking",
        eval_split_size=1,
        split_name="validation",
        seed=0,
        num_workers=1,
        trust_remote_code=False,
        upload_to=None,
    )

    assert cli._cmd_tokenize_pack(args) == 1


def test_tokenize_pack_rejects_bad_upload_to_before_doing_any_work(
    tmp_path, monkeypatch
):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "shard.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "x"}]}) + "\n"
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("should not tokenize when --upload-to is malformed")

    monkeypatch.setattr(cli, "_init_worker", fail_if_called)
    monkeypatch.setattr(cli, "_tokenize_one", fail_if_called)

    args = argparse.Namespace(
        input=str(input_dir),
        tokenizer="unused",
        output=str(tmp_path / "out"),
        pack_size=50,
        allow_drop_oversized=False,
        allow_tokenize_failures=False,
        data_format="messages",
        prompt_key="prompt",
        response_key="response",
        system_prompt=None,
        enable_thinking=False,
        thinking_kwarg="enable_thinking",
        eval_split_size=0,
        split_name="training",
        seed=0,
        num_workers=1,
        trust_remote_code=False,
        upload_to="not-an-s3-uri",
    )

    rc = cli._cmd_tokenize_pack(args)
    assert rc == 1


def test_tokenize_pack_fails_when_every_row_fails_tokenization(
    tmp_path, monkeypatch, caplog
):
    """A systematic failure (wrong --data-format, tokenizer mismatch, etc.)

    must not report success with a silently empty pack — see the
    `data_format='packed'`/use_remove_padding bug class this guards
    against.
    """
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "shard.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "x"}]}) + "\n"
    )
    output_dir = tmp_path / "out"

    monkeypatch.setattr(cli, "_init_worker", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_tokenize_one", lambda row: None)

    args = argparse.Namespace(
        input=str(input_dir),
        tokenizer="unused",
        output=str(output_dir),
        pack_size=50,
        allow_drop_oversized=False,
        allow_tokenize_failures=False,
        data_format="messages",
        prompt_key="prompt",
        response_key="response",
        system_prompt=None,
        enable_thinking=False,
        thinking_kwarg="enable_thinking",
        eval_split_size=0,
        split_name="training",
        seed=0,
        num_workers=1,
        trust_remote_code=False,
        upload_to=None,
    )

    import logging

    with caplog.at_level(logging.ERROR):
        rc = cli._cmd_tokenize_pack(args)

    assert rc == 1
    assert any("zero packed rows" in r.message for r in caplog.records)
    assert not (output_dir / "training_50.idx.parquet").exists()


def test_tokenize_pack_removes_earlier_split_output_when_a_later_split_fails(
    tmp_path, monkeypatch
):
    """training succeeding then validation failing must not leave a

    training parquet behind with no metadata sidecar — a training job
    pointed at that directory later would find real data and no way to
    tell the run failed partway (and no enable_thinking/pack_size to
    cross-check against, since _check_packed_data_matches_config also
    can't see a sidecar that was never written).
    """
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "shard.jsonl").write_text(
        json.dumps({"messages": [{"role": "user", "content": "x"}]}) + "\n"
    )
    output_dir = tmp_path / "out"

    from foresightrlcore.tokenizer.pack import PackedRow

    call_count = {"n": 0}

    def fake_split(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # training: succeeds with one real packed row.
            return [
                PackedRow(input_ids=[1, 2, 3], loss_mask=[0, 0, 1], seq_start_id=[0])
            ], {
                "pack_size": 50,
                "num_raw_rows": 1,
            }
        # validation: fails, mirroring the real "zero packed rows" guard.
        raise RuntimeError("[validation] produced zero packed rows from 1 input rows")

    monkeypatch.setattr(cli, "_tokenize_and_pack_split", fake_split)

    args = argparse.Namespace(
        input=str(input_dir),
        tokenizer="unused",
        output=str(output_dir),
        pack_size=50,
        allow_drop_oversized=False,
        allow_tokenize_failures=False,
        data_format="messages",
        prompt_key="prompt",
        response_key="response",
        system_prompt=None,
        enable_thinking=False,
        thinking_kwarg="enable_thinking",
        eval_split_size=1,
        split_name="training",
        seed=0,
        num_workers=1,
        trust_remote_code=False,
        upload_to=None,
    )
    # Force exactly one row into each split regardless of shuffle order.
    monkeypatch.setattr(
        cli, "_split_train_eval", lambda rows, *a, **k: (rows, list(rows))
    )

    rc = cli._cmd_tokenize_pack(args)

    assert rc == 1
    assert not (output_dir / "training_50.idx.parquet").exists()
    assert not (output_dir / "validation_50.idx.parquet").exists()
    assert not (output_dir / "50_metadata.jsonl").exists()


def test_tokenize_pack_uploads_when_upload_to_is_set(tmp_path, monkeypatch):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "shard.jsonl").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }
        )
        + "\n"
    )
    output_dir = tmp_path / "out"

    monkeypatch.setattr(cli, "_init_worker", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_tokenize_one", lambda row: ([1, 2, 3], [0, 0, 1]))

    mock_client = MagicMock()
    args = argparse.Namespace(
        input=str(input_dir),
        tokenizer="unused",
        output=str(output_dir),
        pack_size=50,
        allow_drop_oversized=False,
        allow_tokenize_failures=False,
        data_format="messages",
        prompt_key="prompt",
        response_key="response",
        system_prompt=None,
        enable_thinking=False,
        thinking_kwarg="enable_thinking",
        eval_split_size=0,
        split_name="training",
        seed=0,
        num_workers=1,
        trust_remote_code=False,
        upload_to="s3://my-bucket/prefix/",
    )

    with patch("boto3.client", return_value=mock_client):
        rc = cli._cmd_tokenize_pack(args)

    assert rc == 0
    uploaded_keys = {c.args[2] for c in mock_client.upload_file.call_args_list}
    assert "prefix/training_50.idx.parquet" in uploaded_keys
    assert "prefix/50_metadata.jsonl" in uploaded_keys
