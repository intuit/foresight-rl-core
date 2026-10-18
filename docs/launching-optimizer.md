# Launching a GenAI Optimizer Run

The `optimizer` operation batch-evaluates candidate LLMs on **quality**
(deterministic gates + LLM-as-judge) and **cost** (token pricing) over a
sampled dataset, producing a results table and a markdown report. It is a
generic, pluggable component — datasets, prompt sets, generation modes, and
judges are all registry-driven — shipped with the **BTS "State of the Business"**
use case fully configured as the reference test case.

It supports:

- **Three generation providers**: `genos` (Intuit LLM Execution), `bedrock`
  (AWS Bedrock `converse`), and `local` (an in-job `vllm serve` process for
  self-hosted candidates).
- **Provider-agnostic LLM judges**: correctness / hallucination judges built on
  `resolve_llm_manager` + structured output — work against Genos **or** Bedrock.
- **Cross-product evaluation**: every `(dataset × model × mode)` combination is
  generated and judged, with a bounded thread pool for concurrency.
- **Resilient calls**: exponential-backoff retry with auth-token refresh on
  generation and judge calls.
- **Checkpoint + resume**: each row is appended to a JSONL checkpoint as soon
  as it is generated and judged — not when its model's batch finishes — so an
  interruption costs the rows in flight rather than everything the batch had
  produced. A resumed run skips already-evaluated combinations. On
  `aws_training_job` the checkpoint prefix defaults to
  `<output.s3_path>/checkpoints/`, so resume works without being configured;
  `validate-optimizer` prints where checkpoints will live. Never point two
  concurrent jobs at one prefix — each sync uploads the whole file, so they
  overwrite rather than merge.
- **Automatic infra sizing**: API-only runs (all genos/bedrock) default to a
  CPU instance; a `provider: local` candidate forces a GPU instance + a
  vLLM-capable image.

## Architecture

```
rl-train optimizer --config optimizer.yaml
 → OptimizerJobConfig.from_yaml() + validate_optimizer_config()
 → OptimizerOperation.run()
 → aws_training_job → OptimizerSageMakerLauncher.launch() → 1 CreateTrainingJob
   local            → run in-process via OptimizerRunner
```

Inside the container (or in-process for `local`):

```
optimizer_entrypoint.sh
 ├─ pip install optimizer deps (pandas, pyarrow, langchain_openai, boto3, ...)
 ├─ resolve secrets via IDPS (when a secrets: section is present)
 └─ python -m foresightrlcore.optimizer.runner
      ├─ resolve records (download s3:// dataset → sample → stratify)
      ├─ for each candidate model (grouped so local models bracket one vllm serve):
      │    ├─ [local] start vllm serve, poll /health
      │    ├─ ThreadPool: per row → mode.generate() → judges → cost → RowResult
      │    ├─ append each row to the JSONL checkpoint (sync to S3 every N rows)
      │    └─ [local] stop vllm serve
      └─ write results.parquet + summary.md + optimizer_results.json to output dir
```

The three pluggable axes for the BTS use case:

| Axis | Registry key | Notes |
|---|---|---|
| Dataset | `bts_parquet` | reference-fidelity loader; `parquet` is the generic column-mapped provider |
| Prompt set | `bts_v1` | shipped at `intuit/examples/prompts/optimizer/bts_v1_prompts.yaml` |
| Mode | `summary_state` | 3-call chain: business_summary → commentary → state_of_business |
| Judges | `sections`, `no_recommendations`, `numeric_overlap`, `length` (deterministic gates); `correctness`, `hallucination` (LLM) | |

## Quick start

1. **Copy and customize the template** — fill in your S3 paths, models, judge
   client, VPC, and secrets:

   ```bash
   cp examples/configs/optimizer_example.yaml my_optimizer.yaml
   ```

2. **Validate locally** (always do this first — round-trips are expensive):

   ```bash
   rl-train validate-optimizer --config my_optimizer.yaml
   ```

3. **Upload the dataset (and prompts) to S3** at the paths given by
   `optimizer.datasets[].path` and `optimizer.prompts_path`:

   ```bash
   AWS_DEFAULT_REGION=us-west-2 aws s3 cp \
     ./eval.parquet s3://your-bucket/optimizer/data/eval.parquet
   AWS_DEFAULT_REGION=us-west-2 aws s3 cp \
     ./my_prompts.yaml s3://your-bucket/optimizer/prompts/my_prompts.yaml
   ```

4. **Submit**:

   ```bash
   AWS_DEFAULT_REGION=us-west-2 rl-train optimizer --config my_optimizer.yaml
   ```

   Add `--wait` to block until the job reaches a terminal state.

## Configuration

The schema is `foresightrlcore.config.optimizer_schema.OptimizerJobConfig`.
Everything the run needs lives under the top-level `optimizer:` block plus the
shared `job` / `execution` / `output` / `container` / `secrets` / `environment`
sections reused from the training schema.

```yaml
optimizer:
  sample:                       # deterministic stratified sampling
    size: 50
    seed: 13
    stratify_by: shard
  datasets:
    - name: bts_parquet         # registered dataset provider
      path: s3://.../data.parquet
      filter: { data_complete: true }
  prompt_set: bts_v1            # registered prompt set
  prompts_path: s3://.../prompts/bts_v1_prompts.yaml  # S3 on aws_training_job
  models:                       # candidate cross-product axis
    - name: gemini-2.5-flash-lite
      provider: genos           # genos | bedrock | local
      price_in: 0.0             # USD / 1K input tokens
      price_out: 0.0            # USD / 1K output tokens
      max_tokens: 16000
  modes: [summary_state]        # registered generation modes
  judges:
    - { name: sections, weight: 0 }          # weight 0 → deterministic gate
    - { name: correctness, weight: 1.0, client: default }  # LLM-as-judge (1–5)
  judge_clients:                # required when any judge is an LLM judge
    default:
      provider: genos
      genos: { model_id: "anthropic.claude-opus-4-5-20251101-v1-0" }
  acceptance_threshold: 0.80    # primary metric pass cutoff (strict >)
  concurrency: 4
  retry: { max_retries: 10, base_delay: 3.0 }
  checkpoint: { enabled: true, every_n_rows: 20 }
```

### Provider notes

For Bedrock and Genos candidates, `optimizer.models[].temperature` defaults to
`0.0` and accepts values from `0.0` to `2.0`. Set it explicitly to `null` to omit
temperature for a model that rejects that parameter. Explicit null survives
config serialization to the worker, and Genos retains the configured value
when its client refreshes. This setting applies to candidate generation;
`judge_clients` has its own configuration.

- **genos** — set the non-secret `GENOS_*` vars in `environment:` and the
  `GENOS_APP_SECRET` in a `secrets:` block (`idps` or `local`). Used for both
  generation (`models[].provider: genos`) and LLM judges (`judge_clients`).
  On `aws_training_job` this requires an `optimizer.vpc` block (see below) —
  both `idps-sdk` and the Genos endpoint live on the Intuit internal network.
  `GENOS_ENV` defaults to `e2e`. Pre-prod slugs are inserted into the service
  host; `prd` and `prod` use the unsuffixed host. A `prd` infix does not
  resolve. The SageMaker execution role must
  be listed on the IDPS policy (`secrets.idps.environment` matching
  `GENOS_ENV`).
- **bedrock** — no `secrets:` section needed; auth is via the SageMaker
  execution role (`bedrock:InvokeModel` on the target model ARNs). Set `region`
  or rely on `AWS_DEFAULT_REGION`.
- **local** — the candidate is served in-job by `vllm serve` on a GPU instance.
  `name` is the HF id / S3 prefix to serve; add a `serving:` block (the same
  `VllmServingConfig` as `perf_test`). Forces GPU infra sizing automatically.

### VPC (Intuit internal network)

Any `aws_training_job` run that resolves secrets via `secrets.provider: idps`
must attach an `optimizer.vpc` block so the container can reach the Intuit
internal PyPI (to pip-install `idps-sdk`) and internal endpoints (e.g. Genos).
Without it the `idps-sdk` install times out and the job fails; validation now
errors out before submission. Use the same private subnets + security group as
the training smoke tests:

```yaml
optimizer:
  vpc:
    subnets:
      - "subnet-06b51b9e8722f4741"  # PrivateSubnetAz1 (us-west-2a)
      - "subnet-0ffa7029ac081597e"  # PrivateSubnetAz2 (us-west-2b)
      - "subnet-0d309336b21b985bb"  # PrivateSubnetAz3 (us-west-2c)
    security_groups:
      - "sg-00089838eb38085c1"
```

Bedrock-only runs (no `secrets:` section) don't need a VPC — Bedrock is
reachable over the public AWS endpoint via the execution role.

### In-transit encryption

On `aws_training_job`, the optimizer job sets SageMaker
`EnableInterContainerTrafficEncryption` from
`optimizer.encrypt_inter_container_traffic` (default `true`), satisfying Intuit
ACP policy `IS031.1` (`AwsSagemakerTrainingjob` must be encrypted in-transit).
It only affects multi-instance jobs but is harmless for the single-instance
default and does not require a VPC. Set it to `false` only if you have a
specific reason.

### Prompts in S3

`optimizer.prompts_path` may be a local path or an `s3://` URI. On
`aws_training_job` it **must** be an `s3://` URI: the launcher ships only the
`foresightrlcore` package + scripts into the container, not `intuit/examples/prompts/`,
so a repo-relative default would not exist there. Keep the prompt-set YAML in S3
next to the dataset — the runner downloads it at start (like the dataset), so
prompt edits need no code deploy:

```bash
AWS_DEFAULT_REGION=us-west-2 aws s3 cp \
  intuit/examples/prompts/optimizer/bts_v1_prompts.yaml \
  s3://<bucket>/<prefix>/optimizer/prompts/bts_v1_prompts.yaml
```

```yaml
optimizer:
  prompt_set: bts_v1
  prompts_path: "s3://<bucket>/<prefix>/optimizer/prompts/bts_v1_prompts.yaml"
```

The `local` backend may omit `prompts_path` and fall back to the packaged
default file.

## Outputs

Written to `output.s3_path` (SageMaker) or the local output dir:

- `results.parquet` — one flattened row per `(dataset, record, model, mode)`
  with per-judge `<judge>_score` / `<judge>_passed` columns, tokens, cost,
  latency, and the aggregated `primary_metric_value` / `evaluation_status`.
  `evaluation_status` is one of `pass` / `fail` / `error`. A row whose
  generation failed is **short-circuited**: judges are skipped (no per-judge
  columns, `primary_metric_value=0.0`), `generation_error` holds the exception,
  and the status is `error` — so a failed generation is never counted as a pass.
- `summary.md` — quality & cost by model×mode (with a `pass%` and `error%`
  column), per-judge pass rates, and a cost projection (per 10K states/month).
- `optimizer_results.json` — the self-contained run manifest
  (`OptimizerResult`): run metadata, a structured `summary` report
  (`overall` pass/fail/error counts + primary-metric stats, `by_model_mode`
  token/cost/latency averages, `per_judge` pass rates) **and** the full `rows`
  list (the same flattened records as `results.parquet`).
- `optimizer_checkpoint.jsonl` — incremental checkpoint (under
  `output.checkpoint_s3_path` when set); resumed runs skip completed rows.

## Execution backends

| Backend | `execution.type` | Notes |
|---|---|---|
| SageMaker Training Job | `aws_training_job` | one `CreateTrainingJob` per submission; requires `output.s3_path` and s3:// dataset paths |
| Local (dev / GPU box) | `local` | runs in-process; resolves secrets + env locally |

Only these two backends are supported.

## Examples

- `examples/configs/optimizer_example.yaml` — generic reference template
  (generic `parquet` dataset + `single_call` mode + Genos generation & LLM
  judges, N=50, API-only / CPU) with placeholders and inline comments.
- `intuit/examples/smoke_tests/optimizer_genos_smoke.yaml` — N=5 Genos e2e
  generation + Genos judges (CPU).
- `intuit/examples/smoke_tests/optimizer/optimizer_bedrock_judge_smoke.yaml` — N=5 Bedrock
  generation + Bedrock judges, no `secrets:` (CPU).
- `intuit/examples/smoke_tests/optimizer/optimizer_local_smoke.yaml` — N=5 local `vllm serve`
  candidate + Bedrock judges (GPU).

## Extending to a new use case

1. Add a dataset provider (or use the generic `parquet` provider with an
   `input_columns` / `reference_column` mapping) and register it in
   `foresightrlcore/optimizer/datasets/`.
2. Add a prompts YAML and register a prompt set (or point `prompts_path` at a
   file directly).
3. Add a generation mode in `foresightrlcore/optimizer/modes/` (or reuse the
   generic `single_call` mode) and register it.
4. Reuse the deterministic / LLM judges or add new ones in
   `foresightrlcore/optimizer/judges/`.

No core edits are needed — each axis is a class plus a `@REGISTRY.register(...)`
decorator and a config entry.
