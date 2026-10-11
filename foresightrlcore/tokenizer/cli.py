"""``rl-train tokenize-pack`` — offline tokenize + pack a JSONL SFT dataset.

CPU-only, runs locally (no SageMaker job needed) — multiprocessing across
rows, one process per core by default. Writes ``training_<pack_size>.idx.parquet``
/ ``validation_<pack_size>.idx.parquet`` plus a ``<pack_size>_metadata.jsonl``
sidecar, matching the shape of the NeMo/Megatron-Bridge packed artifacts
already used for Qwen (e.g. ``qwen35_122b_sft_100k_8k_packed_cp2`` in S3),
but produced by a plain HF-tokenizer implementation with no NeMo/Megatron
dependency.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import time
from typing import Any

logger = logging.getLogger("foresightrlcore.tokenizer.cli")

# Populated once per worker process by _init_worker — avoids re-loading the
# tokenizer (can be a large file, e.g. Gemma 4's tokenizer.json is ~32MB) on
# every call, and avoids pickling it through the multiprocessing queue.
_tokenizer: Any = None
_chat_template_kwargs: dict[str, Any] | None = None
_data_format: str = "messages"
_prompt_key: str = "prompt"
_response_key: str = "response"
_system_prompt: str | None = None


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``tokenize-pack`` subcommand."""
    p = subparsers.add_parser(
        "tokenize-pack",
        help="Offline tokenize + pack a JSONL SFT dataset (CPU-only, local)",
    )
    p.add_argument(
        "--input", required=True, help="Input JSONL file, directory, or glob"
    )
    p.add_argument(
        "--tokenizer",
        required=True,
        help="HF tokenizer path (local dir or S3 prefix ending in '/')",
    )
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument(
        "--pack-size", type=int, default=8192, help="Max tokens per packed row"
    )
    p.add_argument(
        "--allow-drop-oversized",
        action="store_true",
        help="Exit 0 even if some rows were too long to pack on their own "
        "(dropped, count recorded in metadata either way). Without this, "
        "any oversized-row drop is a hard failure (exit 1) — a silent "
        "partial data loss otherwise.",
    )
    p.add_argument(
        "--allow-tokenize-failures",
        action="store_true",
        help="Exit 0 even if some rows raised an exception during "
        "tokenization (skipped, count recorded in metadata's "
        "num_tokenize_failed either way). Without this, any tokenize "
        "failure is a hard failure (exit 1) — same reasoning as "
        "--allow-drop-oversized: a systematic issue (wrong --data-format, "
        "a malformed row, a chat-template edge case) that drops even a "
        "few rows should not silently exit 0 and look like a clean run.",
    )
    p.add_argument(
        "--data-format",
        choices=("flat", "messages"),
        default="messages",
        help="'flat': separate prompt/response fields. 'messages': rows "
        "already have a full messages list.",
    )
    p.add_argument("--prompt-key", default="prompt")
    p.add_argument("--response-key", default="response")
    p.add_argument(
        "--system-prompt", default=None, help="Only used when --data-format=flat"
    )
    p.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set the thinking-mode chat-template kwarg (default: True, "
        "matching model.enable_thinking's own training-schema default — "
        "packed rows are never re-tokenized at train time, so this must "
        "match or training silently uses the wrong chat-template "
        "rendering). Pass --no-enable-thinking to turn it off.",
    )
    p.add_argument(
        "--thinking-kwarg",
        default="enable_thinking",
        help="apply_chat_template() kwarg name that toggles thinking mode",
    )
    p.add_argument(
        "--eval-split-size",
        type=float,
        default=0,
        help="Rows held out for validation — int for an absolute count, "
        "0<x<1 for a fraction. 0 disables the eval split.",
    )
    p.add_argument(
        "--split-name",
        choices=("training", "validation"),
        default="training",
        help="Role of this run's single split when --eval-split-size is 0 "
        "and --input is a separately-packed corpus (e.g. an eval set "
        "packed in its own invocation rather than split out of --input). "
        "Sets the output filename prefix (training_*.idx.parquet vs "
        "validation_*.idx.parquet) that rl-train train's data_format="
        "packed path globs for on data.train_path/eval_path — without "
        "this, a separately-packed eval corpus is always named "
        "training_*, and pointing data.eval_path at it raises "
        "FileNotFoundError even though the parquet is right there. "
        "Must stay 'training' when --eval-split-size > 0, since that "
        "produces both splits (correctly labeled) from one run.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Tokenization worker processes. 0 (default) = os.cpu_count().",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Passed to AutoTokenizer.from_pretrained",
    )
    p.add_argument(
        "--upload-to",
        default=None,
        help="Optional s3://bucket/prefix/ to upload the output directory "
        "to after packing completes. Any bucket/prefix — nothing is "
        "hardcoded.",
    )


def _resolve_input_path(input_path: str) -> str:
    """Download an s3:// input data prefix to a local temp dir once, up front.

    read_jsonl_shards() only understands local files/dirs/globs (same as
    foresightrlcore.methods.sft_verl._glob_data_files, which every shipped
    example config's own header comment already documents an s3:// --input
    as if it worked) — reproduced directly: `--input s3://.../` raised
    `FileNotFoundError: [Errno 2] No such file or directory` since
    os.path.isdir() on an s3:// string is always False, so it fell through
    to treating the whole URI as one literal (nonexistent) local path.
    Mirrors :func:`_resolve_tokenizer_path`'s exact download-once pattern.
    """
    if not input_path.startswith("s3://"):
        return input_path

    import atexit
    import shutil
    import tempfile

    from foresightrlcore.tokenizer.io import download_dir_from_s3

    local_dir = tempfile.mkdtemp(prefix="tokenize_pack_input_")
    # Otherwise every S3 --input run leaks a full copy of the downloaded
    # dataset (potentially GBs — this is the raw source data, not a small
    # tokenizer) in /tmp forever; registered before the download itself so
    # a partial/failed download is still cleaned up on exit.
    atexit.register(shutil.rmtree, local_dir, ignore_errors=True)
    logger.info(f"Downloading input data {input_path} -> {local_dir}")
    download_dir_from_s3(input_path, local_dir)
    return local_dir


def _resolve_tokenizer_path(tokenizer_path: str) -> str:
    """Download an s3:// tokenizer prefix to a local temp dir once, up front.

    AutoTokenizer.from_pretrained doesn't understand s3:// URIs — resolving
    here (in the main process, before the worker pool starts) avoids every
    worker redundantly re-downloading the same files.

    Excludes model-weight-shaped files and caps individual file size —
    model.name/--tokenizer commonly point at the SAME pre-staged S3 prefix
    that also holds the full model checkpoint (config.json, tokenizer.json,
    a few KB-MB, sitting flat alongside model-*.safetensors shards, tens of
    GB — see CLAUDE.md "Pre-Staged Models"). Reproduced directly: pointing
    this at Muse Glimmer 30B's model prefix downloaded 55GB of
    safetensors shards just to reach a 27MB tokenizer.json, with no
    filtering at all. AutoTokenizer/AutoProcessor never need a file bigger
    than a few tens of MB (even a large multilingual vocab), so 64MB is a
    safe, generous cap that still excludes every real weight shard.
    """
    if not tokenizer_path.startswith("s3://"):
        return tokenizer_path

    import atexit
    import shutil
    import tempfile

    from foresightrlcore.tokenizer.io import (
        _MODEL_WEIGHT_EXTENSIONS,
        download_dir_from_s3,
    )

    local_dir = tempfile.mkdtemp(prefix="tokenizer_")
    # Otherwise every S3 --tokenizer run leaks this directory in /tmp
    # forever; registered before the download itself so a partial/failed
    # download is still cleaned up on exit.
    atexit.register(shutil.rmtree, local_dir, ignore_errors=True)
    logger.info(f"Downloading tokenizer {tokenizer_path} -> {local_dir}")
    download_dir_from_s3(
        tokenizer_path,
        local_dir,
        exclude_extensions=_MODEL_WEIGHT_EXTENSIONS,
        max_file_size_bytes=64 * 1024 * 1024,
    )
    return local_dir


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "tokenize-pack":
        return _cmd_tokenize_pack(args)
    logger.error("Unknown tokenizer subcommand: %r", args.command)
    return 1


def _init_worker(
    tokenizer_path: str,
    trust_remote_code: bool,
    chat_template_kwargs: dict[str, Any],
    data_format: str,
    prompt_key: str,
    response_key: str,
    system_prompt: str | None,
) -> None:
    global _tokenizer, _chat_template_kwargs, _data_format
    global _prompt_key, _response_key, _system_prompt

    from transformers import AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=trust_remote_code
    )
    _chat_template_kwargs = chat_template_kwargs
    _data_format = data_format
    _prompt_key = prompt_key
    _response_key = response_key
    _system_prompt = system_prompt


def _tokenize_one(row: dict[str, Any]) -> tuple[list[int], list[int]] | None:
    from foresightrlcore.tokenizer.loss_mask import tokenize_with_loss_mask

    if _data_format == "flat":
        messages = []
        if _system_prompt:
            messages.append({"role": "system", "content": _system_prompt})
        messages.append({"role": "user", "content": row[_prompt_key]})
        messages.append({"role": "assistant", "content": row[_response_key]})
    else:
        messages = row["messages"]

    try:
        return tokenize_with_loss_mask(_tokenizer, messages, _chat_template_kwargs)
    except Exception as e:
        logger.warning(f"Skipping a row due to a tokenization error: {e}")
        return None


def _split_train_eval(
    rows: list[dict[str, Any]], eval_split_size: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not eval_split_size:
        return rows, []

    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)

    n_eval = (
        int(eval_split_size)
        if eval_split_size >= 1
        else int(round(len(shuffled) * eval_split_size))
    )
    n_eval = min(n_eval, len(shuffled))
    return shuffled[n_eval:], shuffled[:n_eval]


def _tokenize_and_pack_split(
    rows: list[dict[str, Any]],
    *,
    split_name: str,
    tokenizer_path: str,
    trust_remote_code: bool,
    chat_template_kwargs: dict[str, Any],
    data_format: str,
    prompt_key: str,
    response_key: str,
    system_prompt: str | None,
    pack_size: int,
    num_workers: int,
) -> tuple[Any, dict[str, Any]] | None:
    from foresightrlcore.tokenizer.pack import TokenizedRow, compute_metadata, pack_rows

    if not rows:
        return None

    t0 = time.time()
    init_args = (
        tokenizer_path,
        trust_remote_code,
        chat_template_kwargs,
        data_format,
        prompt_key,
        response_key,
        system_prompt,
    )
    if num_workers <= 1:
        _init_worker(*init_args)
        results = [_tokenize_one(r) for r in rows]
    else:
        with mp.Pool(num_workers, initializer=_init_worker, initargs=init_args) as pool:
            results = pool.map(_tokenize_one, rows, chunksize=50)
    elapsed = time.time() - t0

    tokenized = [
        TokenizedRow(input_ids=ids, loss_mask=mask)
        for r in results
        if r is not None
        for ids, mask in [r]
    ]
    num_failed = len(rows) - len(tokenized)
    # A logging line must never crash the run — elapsed can be 0.0 for a
    # trivially small/fast batch (sub-tick timer resolution), which would
    # otherwise raise ZeroDivisionError here.
    rows_per_sec = len(rows) / elapsed if elapsed > 0 else float("inf")
    logger.info(
        f"[{split_name}] tokenized {len(tokenized)}/{len(rows)} rows in "
        f"{elapsed:.1f}s ({rows_per_sec:.1f} rows/sec, "
        f"{num_failed} failed)"
    )

    packed_rows, num_oversized = pack_rows(tokenized, pack_size)
    metadata = compute_metadata(len(rows), packed_rows, pack_size, num_oversized)
    metadata["num_tokenize_failed"] = num_failed
    logger.info(f"[{split_name}] packed: {metadata}")

    if not packed_rows:
        # Every row either failed tokenization or was dropped as oversized —
        # writing an empty parquet here would look like a successful, if
        # small, run rather than the systematic failure (wrong
        # --data-format, --tokenizer, --thinking-kwarg, or --pack-size) it
        # actually is.
        raise RuntimeError(
            f"[{split_name}] produced zero packed rows from {len(rows)} "
            f"input rows ({num_failed} failed tokenization, {num_oversized} "
            "dropped as oversized). Refusing to write an empty pack."
        )
    return packed_rows, metadata


def _cmd_tokenize_pack(args: argparse.Namespace) -> int:
    from foresightrlcore.tokenizer.io import (
        read_jsonl_shards,
        upload_dir_to_s3,
        write_metadata,
        write_packed_parquet,
    )

    if args.upload_to is not None:
        # Fail before doing any of the (potentially long) tokenize/pack work
        # rather than after, on a malformed destination.
        if not args.upload_to.startswith("s3://") or not args.upload_to.endswith("/"):
            logger.error(
                f"--upload-to must be an s3://bucket/prefix/ URI ending in "
                f"'/', got: {args.upload_to!r}"
            )
            return 1

    if args.split_name != "training" and args.eval_split_size:
        logger.error(
            "--split-name is only meaningful for a single-split run "
            f"(--eval-split-size 0); got --eval-split-size="
            f"{args.eval_split_size!r} with --split-name={args.split_name!r}."
        )
        return 1

    try:
        resolved_input = _resolve_input_path(args.input)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    try:
        rows = list(read_jsonl_shards(resolved_input))
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    logger.info(f"Loaded {len(rows)} rows from {args.input}")

    train_rows, eval_rows = _split_train_eval(rows, args.eval_split_size, args.seed)
    logger.info(f"Split: {len(train_rows)} train, {len(eval_rows)} eval")

    num_workers = args.num_workers or (os.cpu_count() or 1)
    chat_template_kwargs = {args.thinking_kwarg: args.enable_thinking}
    tokenizer_path = _resolve_tokenizer_path(args.tokenizer)
    # --enable-thinking defaults to False and is baked into every row's
    # chat-template rendering permanently — packed data is never
    # re-tokenized at train time, so there's no second chance to fix a
    # forgotten flag. Logged loudly (not just recorded in metadata below)
    # since this step and the training job that consumes its output are
    # two separate, disconnected processes with no other shared state.
    logger.info(f"Packing with {args.thinking_kwarg}={args.enable_thinking}")

    common = dict(
        tokenizer_path=tokenizer_path,
        trust_remote_code=args.trust_remote_code,
        chat_template_kwargs=chat_template_kwargs,
        data_format=args.data_format,
        prompt_key=args.prompt_key,
        response_key=args.response_key,
        system_prompt=args.system_prompt,
        pack_size=args.pack_size,
        num_workers=num_workers,
    )

    all_metadata = []
    written_parquet_paths: list[str] = []
    # args.split_name is validated above to be "training" whenever
    # eval_split_size > 0 (both splits produced by this same run, always
    # correctly labeled) — it only ever overrides the label for a
    # single-split run, where eval_rows is always empty regardless.
    for split_name, split_rows in (
        (args.split_name, train_rows),
        ("validation", eval_rows),
    ):
        try:
            result = _tokenize_and_pack_split(
                split_rows, split_name=split_name, **common
            )
        except RuntimeError as e:
            logger.error(str(e))
            # A later split failing (e.g. validation, after training
            # already succeeded) must not leave an unpaired parquet file
            # behind with no metadata sidecar — a training job pointed at
            # this directory would find real data and no way to tell the
            # run failed partway.
            for path in written_parquet_paths:
                os.remove(path)
            return 1
        if result is None:
            continue
        packed_rows, metadata = result
        metadata["split"] = split_name
        metadata["enable_thinking"] = args.enable_thinking
        metadata["thinking_kwarg"] = args.thinking_kwarg
        all_metadata.append(metadata)
        out_parquet = os.path.join(
            args.output, f"{split_name}_{args.pack_size}.idx.parquet"
        )
        write_packed_parquet(packed_rows, out_parquet)
        written_parquet_paths.append(out_parquet)
        logger.info(f"[{split_name}] wrote {out_parquet}")

    metadata_path = os.path.join(args.output, f"{args.pack_size}_metadata.jsonl")
    write_metadata(all_metadata, metadata_path)
    logger.info(f"Done. Output: {args.output}")

    # The "produced zero packed rows" case above already hard-fails loudly.
    # This catches the partial case — some rows packed fine, some were
    # individually too long to pack at all and got silently dropped
    # (recorded in metadata regardless, written above either way) — which
    # otherwise exits 0 despite real, silent data loss. Checked *before* the
    # upload below so a rejected pack never lands a complete-looking prefix
    # on S3 for a later training job to silently consume.
    total_oversized = sum(m.get("num_oversized_dropped", 0) for m in all_metadata)
    if total_oversized > 0 and not args.allow_drop_oversized:
        logger.error(
            f"{total_oversized} row(s) across all splits were too long to "
            f"pack on their own (> --pack-size {args.pack_size}) and were "
            "dropped — see per-split num_oversized_dropped in "
            f"{metadata_path}. Pass --allow-drop-oversized to accept this "
            "and exit 0, or increase --pack-size."
        )
        return 1

    # Same reasoning as the oversized check above, for the other silent-
    # row-loss path: _tokenize_one catches any exception per row (a
    # malformed row, an encoding issue, a chat-template edge case) and
    # returns None, and _tokenize_and_pack_split only hard-fails when
    # EVERY row in a split failed. A partial failure — some rows tokenize
    # fine, a few don't — otherwise exits 0 with the same silent data loss
    # the oversized check exists to catch, despite indicating a real,
    # systematic problem (wrong --data-format/--tokenizer, or a genuinely
    # malformed subset of rows) rather than an expected, benign drop.
    total_tokenize_failed = sum(m.get("num_tokenize_failed", 0) for m in all_metadata)
    if total_tokenize_failed > 0 and not args.allow_tokenize_failures:
        logger.error(
            f"{total_tokenize_failed} row(s) across all splits raised an "
            "exception during tokenization and were skipped — see "
            f"per-split num_tokenize_failed in {metadata_path}. Pass "
            "--allow-tokenize-failures to accept this and exit 0, or "
            "investigate --data-format/--tokenizer/the input rows."
        )
        return 1

    if args.upload_to is not None:
        logger.info(f"Uploading {args.output} -> {args.upload_to}")
        upload_dir_to_s3(args.output, args.upload_to)
        logger.info(f"Upload complete: {args.upload_to}")

    return 0
