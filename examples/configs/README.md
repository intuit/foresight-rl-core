# Example configs

Templates for train, perf-test, optimizer, and evaluate. All of them run on
open-source building blocks only: public HuggingFace models and the tasks
that ship in `foresightrlcore`. None require a `secrets:` block. The local
training template scores rewards in-process. The SageMaker training template
uses Bedrock, authenticated by the execution role.

| File | Operation | Command |
|------|-----------|---------|
| [`oss_quickstart.yaml`](oss_quickstart.yaml) | `train` (GRPO on SageMaker) | `rl-train train --config ...` |
| [`local_example.yaml`](local_example.yaml) | `train` (GRPO on one local GPU) | `rl-train train --config ...` |
| [`perf_test_example.yaml`](perf_test_example.yaml) | `perf_test` | `rl-train perf-test --config ...` |
| [`optimizer_example.yaml`](optimizer_example.yaml) | `optimizer` | `rl-train optimizer --config ...` |
| [`evaluate_example.yaml`](evaluate_example.yaml) | `evaluate` | `rl-train evaluate --config ...` |
| [`sagemaker_serverless_example.yaml`](sagemaker_serverless_example.yaml) | `train` via Serverless Model Customization | `rl-train train --config ...` |

Every one of them validates offline, so start there:

```bash
rl-train validate           --config examples/configs/oss_quickstart.yaml
rl-train validate           --config examples/configs/local_example.yaml
rl-train validate-perf      --config examples/configs/perf_test_example.yaml
rl-train validate-optimizer --config examples/configs/optimizer_example.yaml
rl-train validate-evaluate  --config examples/configs/evaluate_example.yaml
```

Validation is purely static — schema, cross-field rules, and per-backend path
checks. It never calls AWS, so it cannot tell you whether a bucket exists or
whether your execution role may invoke Bedrock.

## Before you launch

Replace every `your-bucket` placeholder in the SageMaker templates. That is the
only edit those files strictly need; everything else has a working default.

[`local_example.yaml`](local_example.yaml) has no placeholders. From the repo
root it trains on GPU 0 and reads
[`../datasets/email_outcome/train.jsonl`](../datasets/email_outcome/train.jsonl).

## Smaller, cheaper starting points

[`../smoke_tests/`](../smoke_tests/) holds minimal runs of the same operations —
a 0.8B model, a 60-second load test, five evaluation rows, a two-benchmark
evaluate run that needs no credentials at all. Use those to prove
the pipeline works before spending money on a real run.

## This directory is gitignored by default

`.gitignore` ignores `examples/configs/*` and allowlists the templates
individually, because this is also where working configs with real bucket names
and account IDs tend to accumulate. Two consequences:

- Your own configs here stay out of git automatically.
- A **new template must be added to the allowlist** or it will not be committed,
  and `scripts/make_oss_tree.sh` — which copies only tracked files — will not
  publish it.

## Configuration reference

The schema is `foresightrlcore.config.schema.TrainingJobConfig` (Pydantic v2);
`optimizer_schema.py`, `perf_schema.py` and `evaluate_schema.py` cover the other
operations. Field
docs live on the models themselves, and the `docs/launching-*.md` guides walk
through each backend end to end.
