# sft_verl smoke data

`train.jsonl` — 8 tiny `messages`-format rows (`training.sft_verl.data_format:
messages`), one line per row. Deliberately trivial single-turn Q&A, enough to
prove the tokenize → pack → FSDP-SFT → LoRA-checkpoint path works end to end,
not to teach the model anything.

## Uploading

`data.train_path` is mounted locally as a *directory* by SageMaker's input
channel (`_glob_data_files` globs `*.jsonl` inside it), not a single file —
upload into a prefix, not to an object key:

```bash
AWS_DEFAULT_REGION=us-west-2 aws s3 cp \
  examples/smoke_tests/sft_verl_data/train.jsonl \
  s3://your-bucket/sft_verl_smoke_data/train.jsonl
```

Then point `data.train_path` in [`sft_verl_train_smoke.yaml`](../sft_verl_train_smoke.yaml)
at `s3://your-bucket/sft_verl_smoke_data/` (the prefix, ending in `/` — not
the object key itself).
