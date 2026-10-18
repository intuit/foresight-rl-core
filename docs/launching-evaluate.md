# Launching an Evaluate Run

The `evaluate` operation runs **benchmarks** against **candidate models** and
writes a leaderboard. Its organising idea is one sentence:

> Every model is served as an OpenAI-compatible HTTP endpoint, so a benchmark
> never learns which provider is behind it.

That is what makes a benchmark living in someone else's repository a first-class
citizen here. Point it at `$OPENAI_BASE_URL` and it runs against Bedrock, Genos,
or a local `vllm serve` without knowing the difference — and without being
vendored into this repo.

It supports:

- **Two benchmark tiers** — prompt-in/answer-out benchmarks the runner drives
  record by record, and harnesses that drive their own loop (external
  processes, agentic scenario suites, `lm-eval`).
- **Four providers** — `bedrock`, `genos` (via `foresightrl-intuit`), `local`
  (in-job `vllm serve`), and `echo` (offline, for plumbing checks).
- **A benchmark catalog** — benchmarks as data. Adding one is a YAML entry.
- **Trials with confidence intervals** — a stochastic benchmark measured once
  is not measured; `trials > 1` reports a mean with a Student's-t interval.
- **Checkpoint + resume** — on by default for remote runs, at trial *and*
  record granularity (a record is checkpointed when it is scored, not when its
  benchmark finishes), so a restart does not re-pay for work already done.
  Resume is scoped to what was actually measured: each checkpoint row carries a
  fingerprint of the resolved benchmark spec and the model config, so changing
  a prompt, a metric, or a dataset re-runs the affected trials instead of
  replaying stale scores, while an unchanged config resumes as before. Note
  that reusing a checkpoint path across *different* configurations is therefore
  safe, but reusing it across runs you want independently sampled is not — give
  those a separate `output.checkpoint_s3_path`, or set
  `evaluate.checkpoint.enabled: false`. See [Resume](#resume).
- **Automatic infra sizing** — CPU for API-only runs, GPU as soon as one
  candidate serves locally.

## Architecture

```
rl-train evaluate --config evaluate.yaml
 → EvaluateJobConfig.from_yaml() + validate_evaluate_config()
 → EvaluateOperation.run()
 → aws_training_job → EvaluateSageMakerLauncher.launch() → 1 CreateTrainingJob
   local            → run in-process via EvaluateRunner
```

Inside the container (or in-process for `local`):

```
evaluate_entrypoint.sh
 ├─ pip install evaluate deps (pandas, pyarrow, ...)
 ├─ resolve secrets via IDPS (when a secrets: section is present)
 └─ python -m foresightrlcore.evaluate.runner
      ├─ load the catalog + merge each config entry over it
      ├─ for each candidate model (work is grouped so one endpoint serves all
      │  of that model's benchmarks):
      │    ├─ start the endpoint (proxy for API providers, vllm serve for local)
      │    ├─ for each benchmark x trial:
      │    │    ├─ [record tier]  ThreadPool over records → metric → SampleScore
      │    │    └─ [harness tier] stage source, prepare workspace, hand over
      │    │                      the endpoint, read back the aggregate
      │    ├─ append each completed trial to the JSONL checkpoint
      │    └─ stop the endpoint
      └─ aggregate trials → run every reporter
```

## Quick start

```bash
# What can I run, and what can I plug in?
rl-train list-benchmarks

# Prove the plumbing with no credentials and no data (provider: echo).
rl-train evaluate --config examples/smoke_tests/evaluate_smoke.yaml

# A real run.
rl-train validate-evaluate --config my-evaluate.yaml
AWS_DEFAULT_REGION=us-west-2 rl-train evaluate --config my-evaluate.yaml
```

Account and IAM prerequisites for this organization's AWS accounts are in
[`intuit/docs/account-setup/aws/sagemaker-training-job.md`](../intuit/docs/account-setup/aws/sagemaker-training-job.md);
evaluate uses the same role and image path as a training job. Bedrock candidates
additionally need `bedrock:InvokeModel` on the target model ARNs.

The Python API mirrors the CLI:

```python
from foresightrlcore.evaluate import launch

job_ids = launch("my-evaluate.yaml", wait=False, **{"evaluate.trials": 3})
```

## Configuration

Full annotated template: [`examples/configs/evaluate_example.yaml`](../examples/configs/evaluate_example.yaml).
Schema: `foresightrlcore/config/evaluate_schema.py`.

### Benchmarks

A benchmark entry has exactly one required field, `name`. If that name is in the
catalog, everything else is inherited and **only the fields you set override
it**:

```yaml
evaluate:
  benchmarks:
    - name: demo_mcq                # the whole thing, from the catalog
    - name: demo_mcq
      dataset: {split: test}        # the catalog entry, on the test split
```

An entry whose name is *not* in the catalog defines the benchmark inline and
must declare its `type`.

| Field | Purpose |
|---|---|
| `type` | Registered benchmark type. Default `mcq`. |
| `source` | Where the benchmark's code comes from: `git` + `ref`, `s3`, or `path` (exactly one). |
| `dataset` | `loader` + `path`, or named `splits` with `split` selecting one. An S3 `path` may be a prefix (every file under it is read and concatenated) or a single object, including one with no file extension. `id_columns` names the columns that identify a record, so resume matches rows rather than positions. |
| `metrics` | Registered metric names. |
| `command` | argv for harness types. May reference the contract env vars. |
| `env` | Extra environment, layered over the contract vars — use it to alias contract names onto what the benchmark already reads. |
| `isolate` | `venv` gives a harness-tier benchmark a dedicated interpreter and installs `requirements`. |
| `trials` | Per-benchmark repeat count, overriding `evaluate.trials`. |
| `models` | Restrict to named models or model groups. |
| `failure_tolerance` | Fraction of records allowed to error before the score is reported invalid rather than computed on a partial population. |
| `timeout_seconds` | Wall-clock budget for one trial. |

### Models

```yaml
  models:
    - name: claude-sonnet
      provider: bedrock
      model_id: "anthropic.claude-sonnet-4-20250514-v1:0"
      region: us-west-2
      price_in: 0.003          # USD per 1K tokens; feeds the cost column
      price_out: 0.015

    - name: "s3://bucket/models/Qwen3-8B/"
      display_name: Qwen3-8B   # what reports render; `name` stays the key
      provider: local          # one of these forces a GPU instance
      serving: {tensor_parallel_size: 4}
      leaderboard:             # optional, forwarded to tracking reporters
        model_size: "8B"
        model_type: instruct
        context_window: "128K"
```

`name` is the candidate's **identity** — checkpoints, fingerprints, and
aggregate groups all key on it, which for a locally served model means an S3
prefix. Set `display_name` whenever that is the case: reports render the label
and never the key, so renaming for readability invalidates nothing.

`leaderboard` is descriptive metadata only. It never affects scoring; it exists
so a published run can be read without opening the config that produced it.

`model_groups` names a subset once so a benchmark can say `models: [frontier]`
instead of repeating every name.

### Sampling and reasoning models

```yaml
  models:
    - name: my-reasoning-model
      provider: genos
      temperature: 1.0
      top_p: 0.95
      top_k: 64
      thinking:                # this model narrates before it answers
        tags: [think]          # defaults to [think, reasoning]
```

`top_p` and `top_k` are sent only when set, so anything left out keeps the
provider's own default. `top_k` is not part of the OpenAI schema and travels as
a provider extension — vLLM reads it from the request body, Bedrock forwards it
in `additionalModelRequestFields`, and a provider that does not support it
ignores it rather than failing.

`temperature` is different: it defaults to `0.0` rather than being unset,
because a benchmark wants greedy decoding unless told otherwise. **That default
yields to an explicit `top_p`**, because Anthropic models on Bedrock reject a
request carrying both (`temperature and top_p cannot both be specified for this
model`) — without the exemption, a config naming only `top_p` would fail every
call for a field it never set. Writing both is still allowed and still sends
both; that is a deliberate choice, and `validate-evaluate` warns before the job
starts rather than letting it fail 20 minutes in.

The `thinking` block says the candidate emits its scratchpad inline. The runner
then removes it before **any** metric sees the response, which matters more than
it sounds: the scratchpad is where a model records the answers it considered and
rejected, so an answer extractor reading it returns the rejected one. A response
of `<think>the answer is A — no, B</think> B` scores as A without this.

Writing the block at all turns it on; `enabled: false` switches it off while
keeping the tag list. Set `tags` for a model that uses its own — they are not
standardised, and a model whose tags are not listed is scored on its narration.

For a candidate that says nothing about reasoning, the multiple-choice
extractor strips the two conventional tags on its own, because a reasoning
model is easy to evaluate without noticing it is one. **Writing the block —
either way — turns that safety net off**, so `enabled: false` genuinely keeps
the reasoning tokens rather than having them removed by a metric anyway. The
config is the authority; nothing downstream second-guesses it.

**For a model you serve yourself (`provider: local`), prefer vLLM's reasoning
parser to tag stripping:**

```yaml
    - name: s3://bucket/models/Qwen3.8-27B/
      provider: local
      serving:
        extra_args:
          reasoning-parser: qwen3   # deepseek_r1, gemma4, olmo3, openai_gptoss, ...
```

vLLM then returns the reasoning in a separate field and the answer alone as
the content, which is all the client reads — so no tags reach a benchmark and
there is nothing to strip. The parser is maintained per model family, so it
knows tags we would otherwise have to list, and it handles the two shapes text
stripping gets wrong: an opening tag pre-filled by the chat template, and a
response cut off mid-thought, which arrives as an empty answer with
`finish_reason: length` rather than as a scratchpad an extractor might mine for
a draft answer. Tag stripping stays as the fallback, and is still needed for
candidates behind an API (Genos, Bedrock), where no server flag can be set and
some open models return their reasoning inline, and for a model with no vLLM
parser. Leaving `thinking:` set alongside the parser is harmless: with no tags
left in the content, stripping changes nothing. One cost: the client discards
the separated reasoning, so stored responses show the answer only.

Changing any of these re-runs the affected trials rather than resuming, because
they change the answers. Changing `display_name` or `leaderboard` does not.

### Seeing what the model actually said

```yaml
evaluate:
  store_responses: true            # default false
  store_responses_max_chars: 8000  # 0 keeps it whole
```

Keeps each response on its result row. Off by default — storing every
generation is expensive, and on a run that behaved it tells you nothing. Turn
it on when a score needs explaining, because a zero can mean the model was
wrong, that it answered in a format the metric cannot read, or that
reasoning-tag stripping took the answer along with the narration, and **the
numbers cannot tell those three apart**. `raw_response` holds the text before
stripping and is written only when stripping changed something, so its presence
alone answers "did stripping run" before you read a word of the response.

Smoke and gate configs should set it: a gate that fails without saying why
costs more than the storage does.

An `agent_runtime` block drives a candidate as an **agent** rather than calling
it directly — the agent still reaches the model only through the endpoint, so no
provider knowledge leaks into the benchmark.

#### Sampling

Sampling belongs on the model, not the benchmark: it describes the candidate,
not the question. A benchmark sets one of these only when the task itself
dictates it, and wins where it does.

```yaml
    - name: "s3://bucket/models/Qwen3-8B/"
      provider: local
      temperature: 0.6
      top_p: 0.95
      top_k: 20
      repetition_penalty: 1.1
      extra_body:
        chat_template_kwargs: {reasoning_effort: xhigh}
```

`top_p`, `top_k` and `repetition_penalty` default to unset, which means *do not
send the field*, so a model that omits them behaves exactly as before. The last
two are not in the OpenAI schema but vLLM accepts them as top-level body fields;
an API provider that does not recognise them ignores them. `top_k: -1` disables
the cutoff rather than meaning "zero tokens".

`extra_body` is merged into every request body for provider options with no field
of their own. It is applied last, so it can override any named field when a
provider spells one differently.

A score is only comparable to a published one if the sampling that produced it
can be reproduced, which is why these are named fields rather than left to a
passthrough: a parameter the proxy does not know to forward is dropped in
silence, and the resulting number looks like a model difference.

#### Serving a LoRA adapter

A fine-tune can be evaluated straight from its training output — no merged
checkpoint required.

```yaml
    - name: qwen3-8b-sft            # identity, so it stays distinct in reports
      base_model: "s3://bucket/models/Qwen3-8B/"
      adapter: "s3://bucket/sft/.../output/model/"
      provider: local
      serving: {max_lora_rank: 32}
```

`base_model` carries the weights when `name` is a label rather than a path, which
is what lets a base model and the same base plus an adapter appear as two
candidates. Channels are keyed on the weights URI, so those two download one copy
between them instead of tens of gigabytes twice.

Set `serving.max_lora_rank` to the adapter's `r`: vLLM defaults it to 16 and
refuses anything larger at load time.

Local serving requires a vLLM recent enough to apply LoRA at your tensor-parallel
size. 0.29.0 is known good at TP8; earlier builds hit a Triton assertion in the
LoRA expand kernel. Pin it with `container.extra_requirements` if the image's
version is older.

### Trials and confidence

`trials: 1` (the default) reports a single number. `trials > 1` reports the mean
with a **Student's-t** interval — the normal approximation is wrong at three or
five samples, which is the range that actually gets used.

`confidence.retrial_flaky` spends extra trials only on benchmarks whose
sub-scores disagreed across trials. A benchmark that scored identically every
time is already measured; re-running it buys nothing.

## The endpoint contract

This is the whole integration surface for an external benchmark. The operation
sets these before running it:

| Variable | Meaning |
|---|---|
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` | The endpoint and a placeholder key. |
| `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY` | The same endpoint, for agent CLIs that follow the Anthropic convention. |
| `EVALUATE_MODEL_ID` | Model id to send in requests. |
| `EVALUATE_MODEL_NAME` | Logical candidate name, for the benchmark's own logs. |
| `EVALUATE_OUTPUT_FILE` | Where to write the result JSON. |
| `EVALUATE_WORKSPACE` | Writable scratch directory for this trial. |
| `EVALUATE_PYTHON` | Interpreter to use (the isolated one under `isolate: venv`). |

A benchmark that reads different variable names does not need patching — alias
them:

```yaml
      env:
        YOUR_BENCH_API_BASE: "$OPENAI_BASE_URL"
```

The result JSON is `{"score": 0.87, "n_samples": 120, "sub_scores": {...}}`. A
bare number is accepted too.

## Outputs

Written to `output.s3_path` (or the local output dir):

| File | Contents |
|---|---|
| `benchmark_scores.csv` | The leaderboard matrix: benchmarks x models. |
| `summary.md` | The same matrix with intervals and cost, plus a **Failures** section explaining every missing cell. |
| `results.parquet` | One row per (benchmark, model, trial). |
| `samples.parquet` | One row per scored record (record-tier benchmarks), carrying `finish_reason` alongside the score. |
| `evaluate_results.json` | The self-contained manifest — everything above in one file. |

Every record-tier benchmark reports `truncation_rate` beside its score. Read it
first when a model scores badly: a generation that hit its token ceiling is
scored wrong, so an undersized `max_tokens` reads as a weak model rather than as
a misconfiguration.

A cell reads `n/a` when the benchmark could not be measured: too many records
errored for its `failure_tolerance`, or the harness failed outright. That is
deliberate. A benchmark that failed is not a benchmark that scored zero, and
reporting it as zero silently corrupts a leaderboard.

## Execution backends

| Backend | Use it for |
|---|---|
| `aws_training_job` | The normal path. One `CreateTrainingJob`; CPU or GPU resolved from the candidates. |
| `local` | A dev box or GPU box. Runs in-process. |

`aws_serverless` and `aws_hyperpod_eks` are rejected at parse time.

### Resume

On `aws_training_job`, resume is on by default: completed trials survive a
restart, so an interruption costs the trial in flight rather than the run. The
checkpoint prefix defaults to `<output.s3_path>/checkpoints/`; set
`output.checkpoint_s3_path` to put it elsewhere, and
`evaluate.checkpoint.enabled: false` to force a clean measurement with no
reuse. `validate-evaluate` prints which of the two you are getting.

Resuming by default is safe because a row is only reused when the benchmark
spec and model config that produced it still hash the same — see the
fingerprint note at the top of this document. An edited prompt or a swapped
metric re-runs the trials it affects.

Resume is record-granular, not just trial-granular: each record is
checkpointed as it is scored, so a run killed partway through a benchmark
resumes the records it had already finished rather than repeating the
benchmark. `evaluate.checkpoint.every_n_rows` (default 20) controls how often
the checkpoint is pushed to S3, counted in records; a completed trial always
flushes regardless. Lowering it narrows the window of work at risk but costs
more, because each sync re-uploads the whole file.

A record that errored is **not** treated as measured: a resumed run replans
any trial that finished with errors and re-sends only the records that failed,
reusing the rest. So re-running after an outage is the normal path and needs no
special handling — point it at the same prefix and the gateway timeouts are
retried while everything else is kept. A record that fails for a real reason
rather than a transient one is re-attempted on each resume; each attempt is
bounded by `retry.max_retries`, and `failure_tolerance` governs how many such
records a benchmark may carry before its score is reported invalid.

Two limitations worth knowing:

- **Never point two concurrent jobs at one checkpoint prefix.** Each sync
  uploads that job's whole checkpoint file, so the second writer does not merge
  with the first, it replaces it — the other job's rows are destroyed, not
  interleaved. Give concurrent runs separate prefixes; a distinct
  `output.s3_path` per job does this for you.
- **Duplicate record ids are suffixed, with a warning.** Resume is keyed on
  the record id, so a dataset that reuses one would make two records
  indistinguishable — the first one's result reused for both, the second never
  measured, and the sample count still reading full. Repeats therefore become
  `<id>#2`, `<id>#3`, and the run warns. Ids are untouched when a dataset has
  no duplicates. Checkpoints written before this keyed those records on the
  bare id and will not be reused for them; everything else still resumes.
- **Harness-tier benchmarks checkpoint only on completion.** A harness owns its
  own loop (see the two tiers, above), so the runner cannot see its progress
  and has nothing to checkpoint until it returns. An interrupted harness trial
  is lost and repeats in full. Record-tier benchmarks — every built-in except
  the external-process, `lm_eval`, `scenario_suite` and `plugin` types — resume
  per record.

## Adding a benchmark

Three cases, in ascending order of effort. Most benchmarks are the first one.

### 1. You have the data, we do the asking and the scoring

No Python and no repository. Put the rows somewhere readable and describe them.

**Prepare the file.** CSV, JSONL or Parquet, local for a `local` run or `s3://`
for a SageMaker one. One row per question. The loader finds the columns by name
without being told, provided they are conventionally named:

| Role | Auto-detected from | Override with |
|---|---|---|
| Question | `question`, `prompt`, `input`, `text` | `question_column` |
| Gold answer | `answer`, `label`, `gold`, `target` | `answer_column` |
| Record id | `id`, `example_id`, `record_id` | `id_column` |
| Choices | a `choices` column (JSON list or `\|`-separated), or numbered `choice_1…choice_N` columns | `number_of_choices` to cap how many are read |

A gold answer may be the option's text, its letter, or its 0-based index — all
three resolve.

> **If your table has both a `question` and a `prompt` column, set
> `question_column` yourself.** Detection prefers `question`, and on the
> datasets that carry both, `question` is usually the bare question while
> `prompt` prepends the passage it refers to. Taking the short one asks the
> model what can be concluded from information it was never shown; it replies
> that the passage is missing, and the benchmark scores below its own chance
> baseline while looking like a weak model rather than a broken harness. The
> loader warns when the column it picked is much shorter than one it rejected,
> but naming the column is the fix.

**Write the entry.** In a catalog YAML to make it available by name everywhere,
or inline in one config if it is a one-off:

```yaml
- name: my_tax_bench
  type: mcq                     # or `generative` for free-text answers
  display_name: Tax knowledge   # column header in the leaderboard
  dataset:
    loader: csv
    path: s3://my-bucket/benchmarks/tax/v1.csv
    number_of_choices: 4
  metrics: [choice_match]
```

**Check it is visible, then pre-flight it.** `rl-train list-benchmarks` prints
every registered name; if yours is absent the catalog was not loaded. Then run
it against `provider: echo` with `dataset: {limit: 5}` — that exercises loading,
prompt rendering, scoring and reporting without credentials or spend. Read the
rendered prompt in `samples.parquet` before trusting any score.

> **Make the prompt and the scorer agree.** If your instruction asks for "the
> letter of the correct answer" while the options render unlabelled, a
> compliant model answers `C) Paris` and older scorers marked that wrong. The
> `choice_match` metric now strips an invented label before comparing, but the
> general rule stands: a high `sub_parse_failure_rate` means the format is
> wrong, not the model.

**Listing more than one metric.** The first metric is the reported score;
the rest run alongside it and land in `sub_scores` as `<MetricName>_score`,
without changing what the headline number means. Each also reports
`<MetricName>_valid_rate` — the share of answers it could read at all.

That rate is the one to check first on a disappointing result, because
`sub_parse_failure_rate` cannot always tell you. It comes from the *reported*
metric, and a containment-style metric such as `chat_completion_match` scores
an unreadable answer 0 rather than flagging it, so its parse-failure rate is
structurally zero no matter how garbled the responses are. Pairing it with
`choice_match` is what makes malformed answers visible:

```yaml
  metrics: [chat_completion_match, choice_match]   # headline, then the strict read
```

### 2. The benchmark is someone else's repository

**If it already speaks OpenAI, no Python is involved.** Add a catalog entry, or
put the same block straight in your config:

```yaml
benchmarks:
  - name: my_suite
    type: external_process
    source: {git: "https://github.com/my-org/my-bench", ref: v2.1.0}
    isolate: venv
    requirements: ["-r requirements.txt"]
    command: [python, -m, my_bench, --endpoint=$OPENAI_BASE_URL, --out=$EVALUATE_OUTPUT_FILE]
```

For a benchmark that is a Python object rather than a CLI, use
`type: plugin` with `target: my_pkg.bench:MyBenchmark`. A plugin may set
`isolate: venv` too — it receives the interpreter as `$EVALUATE_PYTHON`, which
is what you want when the plugin is a thin shim around a subprocess.

### 3. It is someone else's repository and it does not speak OpenAI

**Write a shim, not an adapter.** A repository whose runner has its own CLI
shape and its own result format needs translating, but only that: staging, the
venv, the endpoint, workspaces, checkpointing and reporting are already done.
`foresightrl_intuit.evaluate.harnesses.delphi_bench` is the worked example —
a `plugin` benchmark of roughly 200 lines that writes a CSV, builds an argv,
and parses a result document, of which the load-bearing part is one flag. Two
things to check before you start:

* **How does the runner choose its endpoint?** Some take a base URL as an
  argument or an env var, in which case pass `$OPENAI_BASE_URL` and you are
  done. Some hardcode a provider's URL as a module constant, and those cannot
  be pointed anywhere without a patch upstream — worth finding out first,
  because it decides whether this is an afternoon or a pull request against
  someone else's repository.
* **Where do its records come from?** Prefer loading them through
  `dataset:` and materialising them into whatever file the runner reads, as the
  delphi-bench shim does. That way `limit`, sampling and S3 behave the same for
  it as for every other benchmark, instead of the runner having a private way
  to say "only 25 rows".

To make a benchmark available by name everywhere, add it to a catalog YAML.
Built-in: `foresightrlcore/evaluate/catalog/benchmarks.yaml`. Extensions
register their own via the `foresightrlcore.evaluate.plugins` entry point;
one-off catalogs go in `evaluate.catalog_paths`.

## Adding a benchmark type

One class plus a decorator. Pick the tier by which methods you implement — the
runner detects it, there is no flag:

```python
from foresightrlcore.evaluate.interfaces import EvalRecord, Prediction, PromptRequest, SampleScore
from foresightrlcore.evaluate.registry import BENCHMARK_TYPES

@BENCHMARK_TYPES.register("my_type")
class MyBenchmark:
    """Record tier: the runner owns the loop."""

    def __init__(self, **params):
        self.shots = params.get("shots", 0)

    def build_prompt(self, record: EvalRecord) -> PromptRequest:
        return PromptRequest.user(record.inputs["question"])

    def score(self, record: EvalRecord, prediction: Prediction) -> SampleScore:
        correct = prediction.text.strip() == str(record.gold)
        return SampleScore(value=float(correct), correct=correct)
```

For the harness tier, implement `run(ctx) -> HarnessOutcome` instead; `ctx`
carries the endpoint, the workspace, the staged source, and the params.

The other three axes work the same way: `DATASET_LOADERS`, `METRICS`, and
`REPORTERS`. `rl-train list-benchmarks` prints all four registries, so an
extension's contributions are visible without reading its source.

## Reporters

Every run writes the files above to `output.s3_path`. Reporters are what happens
*in addition* to that, and `table` + `json` (the files themselves) are on by
default. A reporter that raises is logged and skipped: a tracking sink being
down must never lose a run whose artifacts are already on disk.

An entry is either a bare name or a mapping with `params`:

```yaml
  reporters:
    - table
    - json
    - type: my_reporter
      params:
        destination: "s3://your-bucket/reports/"
```

`table` and `json` write the files described above. Any other name is a
reporter registered by an extension. A reporter that is not installed logs a
warning and is skipped.

An internal reporter that publishes scores to an experiment tracker is
documented in
[`intuit/docs/publishing-to-aleta.md`](../intuit/docs/publishing-to-aleta.md).

## Troubleshooting

| Symptom | Cause |
|---|---|
| `benchmark 'x' is not in the catalog and defines no 'type'` | Typo in the name, or an inline benchmark missing `type`. `rl-train list-benchmarks` prints the valid names. |
| `unknown task type` / `unknown judge` inside a job | An Intuit-only benchmark, reporter, or provider without `container.extra_packages: ["foresightrl_intuit"]`. |
| A cell reads `n/a` | The benchmark exceeded its `failure_tolerance` or errored. The **Failures** section of `summary.md` says which. |
| `source.path ... not available inside a SageMaker job` | A local benchmark directory does not exist in the container. Use `source.git` or `source.s3`. |
| `is not a GPU family` | A `provider: local` candidate on a CPU instance. Remove `instance_type` and let it resolve, or set an `ml.g*`/`ml.p*` type. |
| Scores are suspiciously round | A `provider: echo` candidate is still in the config. Its replies are synthetic. |
| High `sub_parse_failure_rate` | The model answered in the wrong format, not the wrong answer. Check the prompt before concluding anything about quality. |
| Low `sub_<Metric>_valid_rate` | Same diagnosis, from a metric that can tell. Trust this over `sub_parse_failure_rate` when the reported metric is a containment rule, which cannot flag a malformed answer at all. |
| A score below the chance baseline | Something is wrong with the harness, not the model — a 4-option benchmark scoring under 25% is not measuring what you think. Read a few responses in `samples.parquet`: models usually say what is missing. The common cause is the prompt omitting a passage the question refers to; see `question_column` above. |

## Related

- [`docs/evaluate-extension-contract.md`](evaluate-extension-contract.md) — the
  guarantees an externally-authored benchmark can rely on: the endpoint
  contract, both benchmark protocols, the result shape, and which seams are
  registries. Read this before writing a benchmark that lives in another
  repository.
- [`docs/launching-optimizer.md`](launching-optimizer.md) — quality + cost
  comparison across candidates, the sibling operation.
- [`docs/launching-performance-test.md`](launching-performance-test.md) —
  throughput and latency, not quality.
