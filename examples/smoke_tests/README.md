# Smoke tests

Minimal, cheap runs of each operation, meant to prove the pipeline works before
you spend real money. They use small public models, tiny datasets, and short
durations — the goal is a green run, not a meaningful number.

| Config | Operation | Instance | Roughly |
|--------|-----------|----------|---------|
| [`perf_test_smoke.yaml`](perf_test_smoke.yaml) | `perf_test` | 1x `ml.g5.12xlarge` | 13 min |
| [`optimizer_smoke.yaml`](optimizer_smoke.yaml) | `optimizer` | 1x `ml.m5.2xlarge` (CPU) | 5 min |
| [`evaluate_smoke.yaml`](evaluate_smoke.yaml) | `evaluate` | none — runs locally | seconds |
| [`sft_verl_train_smoke.yaml`](sft_verl_train_smoke.yaml) | `train` (`method.type: sft_verl`) | 1x `ml.p4d.24xlarge` | a few min |
| [`sft_verl_merge_smoke.yaml`](sft_verl_merge_smoke.yaml) | `train` (`sft_verl` merge-only) | 1x `ml.m5.4xlarge` (CPU) | <1 min |
All five are open-source only: public HuggingFace models, Bedrock for judging,
and no `secrets:` block. The first two and sft_verl cannot supply an S3 bucket, so replace
`your-bucket` before launching those. The evaluate smoke test needs nothing at
all — no bucket, no credentials, no GPU.

## Perf test

Fully self-contained — the prompt data is in this repo.

```bash
# 1. Upload the tiny synthetic dataset
AWS_DEFAULT_REGION=us-west-2 aws s3 cp \
  examples/smoke_tests/perf_test_data/ \
  s3://your-bucket/perf_test/smoke_data/ \
  --recursive --exclude README.md

# 2. Point the config at your bucket, then
rl-train validate-perf --config examples/smoke_tests/perf_test_smoke.yaml
rl-train perf-test     --config examples/smoke_tests/perf_test_smoke.yaml
```

A healthy run ends with two Gatling stages at KO=0. Reference numbers on
`g5.12xlarge`: TPS=1 gives ~80 requests at 0.91 req/s, TPS=2 gives ~131 at
1.46 req/s.

If Gradle fails to resolve `io.gatling:...` with an HTTP 429, add a `vpc:` block
— see the note at the bottom of the config for why.

## Optimizer

Needs an evaluation parquet of your own with a question column and a
reference-answer column; five rows is plenty.

```bash
rl-train validate-optimizer --config examples/smoke_tests/optimizer_smoke.yaml
rl-train optimizer          --config examples/smoke_tests/optimizer_smoke.yaml
```

## Evaluate

Two catalog benchmarks against a synthetic `provider: echo` candidate, run
in-process. It proves the whole path — endpoint, benchmark, metric, reporters —
without a bucket, a credential, or a GPU. The scores are meaningless by
construction; a green run and a written `summary.md` are the result.

```bash
rl-train evaluate --config examples/smoke_tests/evaluate_smoke.yaml
```

Confirm the execution role can call `bedrock:InvokeModel` on both candidates and
the judge model first. Validation is offline and cannot check that for you.

Re-running resumes from the JSONL checkpoint. That is usually what you want, but
after changing the config, clear the checkpoint under `output.s3_path` — a run
that finds every row already complete exits successfully having done nothing.

## sft_verl train

`sft_verl` (verl's FSDP SFT trainer) needs no task/reward-function/LLM-judge
plugin at all — unlike GRPO, a generic tiny dataset is the whole story, so
this one gets a real smoke test.

```bash
# 1. Upload the tiny messages-format dataset
AWS_DEFAULT_REGION=us-west-2 aws s3 cp \
  examples/smoke_tests/sft_verl_data/train.jsonl \
  s3://your-bucket/sft_verl_smoke_data/train.jsonl

# 2. Point the config at your bucket, then
rl-train validate --config examples/smoke_tests/sft_verl_train_smoke.yaml
rl-train train    --config examples/smoke_tests/sft_verl_train_smoke.yaml
```

Ends with one LoRA-only checkpoint at `global_step_3`. See the config's own
header for why `ml.p4d.24xlarge` (not a cheaper single-GPU box) is currently
the smallest instance this can use.

## sft_verl merge

Follow-up to the train smoke above: exercises
`training.sft_verl.merge_lora_only` (reconstruct the LoRA checkpoint, merge it
into the real base model, save a plain HF model) against the checkpoint the
train smoke just produced. CPU-only — merge never touches the GPU/attention
path.

```bash
rl-train validate --config examples/smoke_tests/sft_verl_merge_smoke.yaml
rl-train train    --config examples/smoke_tests/sft_verl_merge_smoke.yaml
```

`model.name` and `model.lora.r`/`alpha` must match whatever SFT run produced
the checkpoint you point `output.checkpoint_s3_path` at — the config's own
comments call out exactly which fields.

## No GRPO training smoke test here

[`../configs/local_example.yaml`](../configs/local_example.yaml) is a one-GPU
GRPO run with a sample dataset already in the repo.
[`../configs/oss_quickstart.yaml`](../configs/oss_quickstart.yaml) is the
SageMaker version of the same task; it needs your own `train.jsonl` in S3.

## Extension smoke tests

Deployment-specific smoke tests live with their extension package rather than
here — for example `intuit/examples/smoke_tests/`, which covers the `advice`
task, Genos and IDPS. Anything in this directory runs against the OSS core
alone.
