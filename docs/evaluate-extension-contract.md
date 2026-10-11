# `evaluate` — Extension Contract

This is the **stable surface** the `evaluate` operation promises to benchmark
authors. Everything here is versioned with the package; everything *not* here is
an implementation detail that may change without notice.

Read this if you are:

- porting a benchmark from `delphi-benchmarks-hub` (or anywhere else) onto
  `foresightrlcore`,
- writing a new benchmark type, metric, dataset loader, or reporter,
- deciding whether your benchmark belongs in the OSS core or in an extension
  package.

> Status: contract frozen for the first implementation. Open questions are
> listed under [Pending sign-off](#pending-sign-off).

---

## 1. The three seams

`evaluate` is a **substrate**, not a benchmark collection. It owns three things
and nothing else:

| Seam | What the core provides | What you provide |
|---|---|---|
| **Inference** | An OpenAI-compatible `base_url` + `api_key`, whatever the model actually is | Nothing — just consume the endpoint |
| **Execution** | Workspace staging, process supervision, retries, checkpoint/resume, K-trial statistics | The benchmark body (in-process or a command line) |
| **Reporting** | Normalised `BenchmarkResult` → parquet/JSON/markdown/table | Scores in the documented shape |

The core never hardcodes a benchmark, a dataset path, or a vendor. Real
benchmark definitions live in **catalog YAML** (data, not code), and Intuit-only
catalogs live in the `foresightrl_intuit` extension.

---

## 2. Inference contract

Every benchmark — in-process or external subprocess — talks to models through
**one** interface: an OpenAI-compatible HTTP endpoint.

### 2.1 What the core guarantees

For each model in the run, the core brings up an endpoint and guarantees:

- `POST {base_url}/chat/completions` accepts the OpenAI request body and returns
  the OpenAI response body.
- `GET {base_url}/models` lists at least the served model id.
- The endpoint is **healthy before** the benchmark starts and stays up for the
  benchmark's whole batch.

Three providers resolve to that same contract:

| `provider` | How the endpoint is produced | Ships in |
|---|---|---|
| `bedrock` | stdlib proxy translating OpenAI ⇄ `boto3.bedrock-runtime.converse` | OSS core |
| `genos` | same proxy, dispatching through `resolve_llm_manager` | core proxy, Genos client via `foresightrl_intuit` |
| `local` | `vllm serve` subprocess, already OpenAI-native | OSS core |

Adding a fourth provider means registering an LLM provider — it does **not**
mean touching `evaluate`.

### 2.2 Environment variables handed to a subprocess benchmark

An external benchmark is launched with these variables set. **These names are
the contract** — they are what a hub adapter needs to read:

| Variable | Meaning |
|---|---|
| `OPENAI_BASE_URL` | Endpoint root, e.g. `http://127.0.0.1:8123/v1` |
| `OPENAI_API_KEY` | Always set; a placeholder when the endpoint needs no auth |
| `EVALUATE_MODEL_ID` | The model id to send in the request body |
| `EVALUATE_MODEL_NAME` | Logical candidate name used in reports |
| `EVALUATE_WORKSPACE` | Absolute path to this trial's writable workspace |
| `EVALUATE_OUTPUT_FILE` | Absolute path the benchmark writes its result JSON to |
| `EVALUATE_TRIAL` | 0-based trial index for multi-trial runs |

`ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY` are set to the same endpoint as well,
so agent CLIs that speak the Anthropic env convention work unchanged.

Anything under a benchmark's `env:` block in YAML is layered on top, so a
benchmark that expects its own variable names can alias them without code.

---

## 3. Benchmark contract — two tiers

A benchmark declares its tier by which protocol it implements. The runner
detects the tier; there is no flag.

### 3.1 `RecordBenchmark` — the core drives the loop

For classic prompt → completion → score benchmarks. You get one record at a
time and return one score. The core handles concurrency, retries, cost
accounting, and checkpointing per record.

```python
class RecordBenchmark(Protocol):
    def build_prompt(self, record: EvalRecord) -> PromptRequest: ...
    def score(self, record: EvalRecord, prediction: Prediction) -> SampleScore: ...
```

### 3.2 `HarnessBenchmark` — the benchmark drives its own loop

For agentic benchmarks, external repositories, and anything that needs to run a
whole suite in one shot. You get a prepared workspace and a live endpoint, and
you return the aggregate.

```python
class HarnessBenchmark(Protocol):
    def run(self, ctx: HarnessContext) -> HarnessOutcome: ...
```

`HarnessContext` carries the endpoint, the workspace path, the resolved dataset
split, the trial index, and a logger. This is the tier a `delphi-benchmarks-hub`
adapter maps onto — the adapter's `run()` becomes this `run()`.

It also carries `agent_runtime`, set when the candidate names one. That block
says which agent program to launch and how; the agent then reaches the model
through the same endpoint every other benchmark uses, so declaring it leaks no
provider knowledge. A harness that ignores the field is unaffected.

### 3.3 Result shape

Both tiers normalise into the same record. This is what reporters consume and
what lands in `results.parquet`:

```python
BenchmarkResult(
    benchmark: str,          # benchmark name
    model: str,              # logical candidate name
    trial: int,              # 0-based
    score: float,            # primary metric, higher is better
    n_samples: int,
    n_errors: int,
    sub_scores: dict[str, float],   # named secondary metrics
    detail: dict[str, Any],         # free-form, preserved verbatim
    started_at: str, finished_at: str,
)
```

`sub_scores` is the extension point for benchmarks that report more than one
number (BookkeeperBench's binary / fractional / rubric triple, for example).
Nothing in the core interprets `detail`; it round-trips to the report.

---

## 4. `BenchmarkSpec` — the YAML surface

One entry adds a benchmark. Unknown keys are allowed and forwarded to the
benchmark type, so a benchmark can take parameters the core has never heard of.

```yaml
benchmarks:
  - name: my_benchmark          # catalog key, or a new definition inline
    type: external_process      # registered benchmark type
    source:                     # optional: external repo to stage
      git: https://github.example.com/org/repo
      ref: main
      # or:  s3: s3://bucket/prefix/
      # or:  path: ./local/dir
    dataset:
      loader: csv               # registered dataset loader
      path: s3://bucket/data.csv
      split: test
    command: ["python", "-m", "runner", "--out", "$EVALUATE_OUTPUT_FILE"]
    env:
      MY_BENCHMARK_MODE: strict
    isolate: venv               # venv | none — isolated interpreter for `command`
    timeout_seconds: 3600
    trials: 3
    params:                     # forwarded verbatim to the benchmark type
      grader: exact_value
```

**Placement rule:** a spec whose `source` or `dataset.path` names an internal
host, bucket, or account belongs in the extension catalog
(`foresightrl_intuit/evaluate/catalog/`), never in the OSS tree. The OSS
fingerprint scan (`scripts/check_intuit_fingerprints.sh`) enforces this at CI
time.

---

## 5. Multi-trial statistics

`trials: K` runs the benchmark K times and reports mean with a **t-distribution
95% confidence interval** (not normal — K is small). This is required for
benchmarks with stochastic scenarios: BookkeeperBench measured ~22% of its
scenarios as non-deterministic, so a single run is not a measurement.

Where a benchmark reports per-scenario detail, the runner can re-run only the
scenarios that disagreed across trials rather than the whole suite.

---

## 6. Extending without forking

| You want to add | You write | You register with |
|---|---|---|
| A benchmark | A YAML catalog entry | nothing — it is data |
| A benchmark kind | One class | `@BENCHMARK_TYPES.register("name")` |
| A data format | One loader | `@DATASET_LOADERS.register("name")` |
| A scoring rule | One function | `@METRICS.register("name")` |
| An output sink | One reporter | `@REPORTERS.register("name")` |

Extension packages register through the `foresightrlcore.evaluate.plugins`
entry-point group and are staged into remote jobs via
`container.extra_packages`, exactly like optimizer and perf-test extensions.

### The one seam that is not a registry

**Inference providers are a core edit, deliberately.** An endpoint is not a
scoring rule that can be swapped per run — it decides what the whole operation
is measuring, and every candidate in a run is compared on the assumption that
they were reached the same way. Adding one touches a known, small set of
places:

| If the provider is… | Add |
|---|---|
| Reached over HTTP | its name to `PROXIED_PROVIDERS` and `MODEL_PROVIDERS`; the client is found through `resolve_llm_manager` |
| A process you start | an `EndpointHandle` implementation, a branch in `make_endpoint_handle`, its name in `MODEL_PROVIDERS`, and `GPU_PROVIDERS` if it needs hardware |

`VllmServerEndpoint` is the worked example of the second kind, and it is the
shape an **agent server** would take: start it, wait for health, hand back a
URL, tear it down. Because the benchmark only ever sees the URL, putting an
agent there instead of a model requires no change to any benchmark — which is
the property the whole contract exists to protect.

Note the distinction from `agent_runtime` (§3.2). That block drives an agent on
*our* side of the endpoint, as the harness; a provider puts one on the *far*
side, as the subject under test. They are not alternatives — a single run could
use each, and only the second changes what the score is a statement about.

---

## Pending sign-off

The first two items below are settled by what shipped; they are kept as a
record of what was decided rather than as open questions.

1. ~~**Reference benchmark.**~~ Settled. The extension catalog carries eleven
   `LLM_BENCHMARK_*` entries we score ourselves, plus `DELPHI_BENCH_IA_tax`,
   which hands scoring to `delphi-bench`'s own runner and is the proof that an
   unmodified external repository runs on the endpoint contract.
2. ~~**Env-var aliases.**~~ Settled, with a caveat worth carrying forward. Hub
   benchmarks do not read `OPENAI_BASE_URL`; they take a base URL as a CLI
   argument, which the shim supplies from the contract. So the question for any
   further port is not *which variable* but *whether the URL is reachable at
   all* — TAT-QA hardcodes it as a module constant and cannot be redirected by
   configuration.
3. **`sub_scores` keys.** Still open, and now concrete. `delphi-bench` emits
   `accuracy` plus `ci_lower` / `ci_upper` / `ci_margin_of_error`. Agree these
   names before more harness benchmarks land, because a leaderboard column
   renamed later silently splits a series.

### Not yet supported

Evaluating an **agent as the subject** — the hub's majority case, where a
server such as `fgi-labs-svc` wraps a model in tools and skills and the
benchmark measures that — needs a process-owning provider (see §6) and an
answer to what a candidate is called in a report when it is not a model. The
inverse case, an agent as the *harness*, is supported today via `agent_runtime`
and needs only the agent's CLI present in the image.
