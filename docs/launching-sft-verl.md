# Launching a verl FSDP SFT Run (`method.type: sft_verl`)

## What this is

`sft_verl` is LoRA supervised fine-tuning via [verl](https://github.com/volcengine/verl)'s
FSDP `SFTTrainer` — a plain `torchrun` job (no Ray, no vLLM), launched the
same way as GRPO/TRL methods on the `aws_training_job` backend but with a
different, framework-native trainer underneath.

Use it over the TRL/GRPO SFT path when your model needs verl's FSDP2
sharding, or a `model.profile` whose LoRA targets can't be expressed as
plain PEFT `target_modules` (fused-expert MoE weights via
`target_parameters` — see `foresightrlcore/config/model_profiles.py`).
For a dense model with no such quirks, either path works; nothing about
`sft_verl` requires an MoE model.

Config schema: `VerlSftConfig` in `foresightrlcore/config/schema.py`
(`training.sft_verl:`). Validation: `_validate_sft_verl()` in
`foresightrlcore/config/validation.py` (this is the single canonical
implementation — `VerlSftMethodHandler.validate()` in
`foresightrlcore/methods/sft_verl.py` delegates to it).

## Quick start

For a cheap, OSS-clean, real end-to-end run before committing to a full
recipe — public 0.6B model, tiny inline dataset, 3 optimizer steps — see
`examples/smoke_tests/sft_verl_train_smoke.yaml` and its merge-only
follow-up `sft_verl_merge_smoke.yaml` (`examples/smoke_tests/README.md`
has the full walkthrough for both).

For the full production shape:

```bash
rl-train validate --config examples/configs/sagemaker_sft_verl_example.yaml
rl-train train    --config examples/configs/sagemaker_sft_verl_example.yaml
```

That example uses `data_format: packed` (offline-tokenized, see below).
For a first run with no offline step at all, set
`training.sft_verl.data_format: messages` instead — rows are tokenized
live at train time, same as GRPO/TRL. Account/IAM setup is identical to
any other `aws_training_job` — see
`docs/account-setup/aws/sagemaker-training-job.md`.

## Data formats

`training.sft_verl.data_format` is one of:

- **`messages`** — each row already has a full `messages` list. Simplest;
  no offline step. Tokenized live, one process per node caches the source
  JSONL as Parquet once (`_cache_jsonl_as_parquet`) so all local ranks
  read the cheaper binary format instead of re-parsing JSON per rank.
- **`flat`** — separate `prompt`/`response` (and optional `system_prompt`)
  string fields; `sft_verl` synthesizes the `messages` list itself.
- **`packed`** — pre-tokenized and pre-packed Parquet, produced offline by
  `rl-train tokenize-pack`. No tokenization happens in the training job at
  all. Use this when you need sequence-packing efficiency but the model's
  own attention backend can't use verl's native `use_remove_padding`
  packing path (see the next section) — packing efficiency then comes
  entirely from the offline step.

  Runs standalone (no GPU stack needed), but does need its own
  dependencies — install with `uv pip install -e ".[tokenizer]"` (or
  `pip install -e ".[tokenizer]"`) first: a plain checkout has neither
  `transformers` nor `pandas`/`pyarrow` installed, and the base
  package alone doesn't pull them in.

  ```bash
  rl-train tokenize-pack \
    --input <dir-of-jsonl-shards> \
    --tokenizer <hf-model-id-or-local/s3-path> \
    --output <local-out-dir> \
    --pack-size 18432 \
    --data-format messages \
    --eval-split-size 0.02
  # upload <local-out-dir>'s contents (training_*.idx.parquet,
  # validation_*.idx.parquet, *_metadata.jsonl) to S3, and point
  # data.train_path/eval_path at that prefix. Drop --eval-split-size (or
  # set it to 0) and also drop data.eval_path from the training config if
  # you don't want a held-out split — sft_verl's packed-data loader globs
  # for validation_*.idx.parquet and hard-fails if data.eval_path is set
  # but no such file was produced.
  ```

  `--enable-thinking` (default: on, matching `model.enable_thinking`'s own
  default) and `--thinking-kwarg` must match the training config's
  `model.enable_thinking`/`model.thinking_kwarg` exactly — packed rows are
  never re-tokenized at train time, so a mismatch trains on the wrong
  chat-template rendering with no error unless caught. `sft_verl.py`
  cross-checks this automatically against the `*_metadata.jsonl` sidecar
  `tokenize-pack` writes, and hard-fails if that sidecar is missing for
  `data_format: packed` specifically (that data_format implies the
  sidecar should exist).

## `use_remove_padding` — when packing is actually safe

`training.sft_verl.use_remove_padding` controls verl's *native* packing
path (concatenates a micro-batch into one flash-attention `cu_seqlens`
call). It requires `model.attn_implementation: flash_attention_2` and
`data_format` other than `packed` (verl's native packing operates at the
micro-batch row boundary, not the document boundary an offline-packed row
already has internally — the two packing mechanisms are mutually
exclusive, not stackable).

This is defaulted to `False` automatically whenever
`model.attn_implementation` is set to anything other than
`flash_attention_2`, or `data_format: packed` — so a config using a
`model.profile` that forces a different attention backend (e.g.
`gemma4_moe` → `sdpa`, because Gemma 4's global-attention layers exceed
FlashAttention-2's head_dim cap) doesn't need to also remember to set this
by hand. Setting it explicitly to `true` in an incompatible combination
is still a hard validation error — the wrong pairing trains with silently
corrupted gradients, not a crash, so it's never silently coerced past a
real conflict.

## LoRA targeting and `model.profile`

`model.lora.{target_modules,exclude_modules,target_parameters}` map
directly onto `peft.LoraConfig`. Two things worth knowing:

- An empty `target_modules: []` means "let PEFT auto-detect via
  `all-linear`" — needed for models whose linear layer names don't match
  the schema's non-empty default list.
- `exclude_modules` accepts either a list (exact trailing path segment
  match, e.g. `["router.proj"]`) **or** a single regex string (full-path
  match via `re.fullmatch`) when you need to exclude an entire subtree a
  list can't express that way.

`model.profile: <name>` (see `foresightrlcore/config/model_profiles.py`)
resolves all of the above plus `thinking_kwarg`/`trust_remote_code`/
`attn_implementation` from a per-model-family preset, verified once
against the real checkpoint. Fields you set explicitly under `model.lora`
always win over the profile.

## Checkpoints and merging

verl's `SFTTrainer` has no separate "save final model" step — it only
ever writes checkpoints to `output.checkpoint_s3_path` (via SageMaker's
`CheckpointConfig` S3 sync), so that field is **required** for a normal
training run; a job without it "succeeds" with no durable output anywhere.

By default (`save_lora_only: true`), checkpoints contain only the adapter
weights (a few GiB vs the full base model). To get a plain, merged HF
model out for downstream use (e.g. as an RL policy's starting point), run
a separate `merge_lora_only: true` job pointed at the checkpoint —
mirrors the pattern already covered here at a high level; see
`foresightrlcore/methods/verl_checkpoint_merge.py`'s module docstring for
the full mechanics and why a real merge (not verl's own adapter-export
tool) is used for `target_parameters`-based MoE LoRA.

## Container recipe

`container.recipe: verl_sft` — model-agnostic (verl's FSDP SFT stack is
the same regardless of `model.profile`); see
`intuit/docs/building-docker-images.md` for the pre-built image option.

## Common failures

| Symptom | Cause |
|---|---|
| `use_remove_padding must be false when model.attn_implementation=...` | Explicit `use_remove_padding: true` with a non-flash_attention_2 backend or `data_format: packed` — see the section above. |
| `output.checkpoint_s3_path is required for sft_verl training` | Missing on a non-merge run — verl never writes checkpoints anywhere else. |
| `data_format='packed' but no *_metadata.jsonl sidecar found` | `data_format: packed` data wasn't produced by (or lost the sidecar from) `rl-train tokenize-pack`. |
| `LOCAL_WORLD_SIZE is not set` | Not launched via torchrun (directly, or SageMaker's SDK-native Torchrun distribution) — this handler has no other supported launch path. |
