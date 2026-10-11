# Launching a Performance Test

The `perf_test` operation runs Gatling load tests against a `vllm serve`
process inside a SageMaker training container. It supports:

- **One run per job**: a single `perf_test` block per YAML config, one
  SageMaker training job per submission. To compare configurations (e.g.
  baseline vs speculative decoding), launch separate jobs with separate configs.
- **Pluggable Gatling simulations**: pick a Java class from the vendored
  project at `foresightrlcore/perf_test/gatling/`, or drop a new `.java`
  file beside it and set `gatling.simulation_class` in YAML.
- **First-class speculative decoding**: reuses the same `SpeculativeConfig`
  schema as training (`infra.vllm.speculative`).
- **User-supplied test data**: prompts, system message, and chat config
  are mounted from an S3 prefix you specify via
  `gatling.data_s3_path` — nothing is bundled by default.

## Architecture

```
rl-train perf-test --config perf.yaml
 → PerfTestJobConfig.from_yaml() + validate_perf_config()
 → PerfTestOperation.run()
 → PerfTestSageMakerLauncher.launch()
 → 1 CreateTrainingJob request
```

Inside the container:

```
perf_entrypoint.sh
 ├─ install Gradle 8.10  (~30s; Java + vLLM already in DJL base image)
 ├─ pip install perf-test deps
 ├─ export vllm_env values
 └─ python -m foresightrlcore.perf_test.runner
      ├─ copy prompts.csv / system-message.txt / chat-config.json from data channel
      ├─ start vllm serve (subprocess), poll /health
      ├─ for each tps_level: gradle gatlingRun --simulation=... -DsimulationTps=...
      └─ write perf_results.json + perf_results.md to /opt/ml/output/data
```

## Quick start

1. **Validate locally** (always do this first — round-trips are expensive):

   ```bash
   rl-train validate-perf --config examples/configs/perf_test_example.yaml
   ```

2. **Upload test data to S3**. The runner requires files at the prefix given by
   `gatling.data_s3_path`:

   - `chat-config.json` — model parameters
     (`{"model": "...", "max_tokens": 1024, "temperature": 0.4, ...}`)
   - `prompts.csv` — CSV with at least a `prompt` column (for `ThrottledTpsCsvSimulation`)
   - `prompts.jsonl` — line-delimited JSON with `prompt` array field (for `JsonlChatCompletionsSimulation`)
   - `system-message.txt` — system prompt (optional; required only for CSV-based simulations)

3. **Submit**:

   ```bash
   AWS_DEFAULT_REGION=us-west-2 rl-train perf-test \
       --config examples/configs/perf_test_example.yaml
   ```

   Python API equivalent:

   ```python
   from foresightrlcore.perf_test import launch
   job_ids = launch("examples/configs/perf_test_example.yaml")
   ```

## Configuration reference

See `foresightrlcore/config/perf_schema.py` for the canonical
`PerfTestJobConfig` schema. Key blocks:

### `perf_test:` (`PerfTestRunConfig`)

The `perf_test` block is a single run definition. Fields:

| Field            | Default            | Notes                                     |
| ---------------- | ------------------ | ----------------------------------------- |
| `name`           | required           | Alphanumeric, `-`, `_` only. Used as job-name suffix and output subdir. |
| `instance_type`  | `ml.g5.xlarge`     | SageMaker instance type.                  |
| `instance_count` | `1`                | Almost always `1` for perf tests.         |
| `volume_size_gb` | `200`              | EBS volume size.                          |
| `encrypt_inter_container_traffic` | `true` | Encrypt inter-container traffic (SageMaker `EnableInterContainerTrafficEncryption`). On by default to satisfy Intuit ACP policy `IS031.1`. |
| `serving`        | see below          | vLLM serve parameters.                    |
| `gatling`        | see below          | Load-test parameters.                     |
| `run_gatling_test` | `true`           | Set to `false` for vLLM-only smoke runs.  |

### `serving:` (`VllmServingConfig`)

Maps onto `vllm serve <model> --<flag> <value>...` plus a few env vars.
Covers the production DJL serving properties used in foresight-rl-core
deployments.

| YAML key                  | CLI flag / behavior                                  |
| ------------------------- | ---------------------------------------------------- |
| `port`                    | `--port` (default 8080 — matches simulation default) |
| `tensor_parallel_size`    | `--tensor-parallel-size`                              |
| `gpu_memory_utilization`  | `--gpu-memory-utilization`                            |
| `max_model_len`           | `--max-model-len`                                     |
| `max_num_seqs`            | `--max-num-seqs` (DJL `max_rolling_batch_size`)       |
| `served_model_name`       | `--served-model-name`                                 |
| `trust_remote_code`       | `--trust-remote-code` (default `true`)                |
| `enforce_eager`           | `--enforce-eager`                                     |
| `dtype`                   | `--dtype`                                             |
| `quantization`            | `--quantization` (e.g. `awq`)                         |
| `enable_prefix_caching`   | `--enable-prefix-caching`                             |
| `speculative`             | `--speculative-config '<json>'` (reuses `SpeculativeConfig`) |
| `vllm_env`                | Exported as env vars before vllm serve launches      |
| `extra_args`              | Generic `--flag value` escape hatch. An empty value emits a bare flag, which is how vLLM's `store_true` options are reached |
| `gpu_devices`             | Sets `CUDA_VISIBLE_DEVICES` for the subprocess        |
| `startup_timeout`         | Seconds to wait for `/health` (default 1800)          |

### `gatling:` (`GatlingConfig`)

| YAML key            | Default                                                | Notes                                |
| ------------------- | ------------------------------------------------------ | ------------------------------------ |
| `simulation_class`  | `org.foresightrl.perf.ThrottledTpsCsvSimulation`       | Java FQCN, must exist on classpath.  |
| `tps_levels`        | `[1, 2, 3, 4]`                                         | Run order matches the list order.    |
| `duration_seconds`  | `600`                                                  | Steady-state duration of each run.   |
| `data_s3_path`      | **required**                                           | S3 prefix with the data files.       |
| `concurrent_users`  | `None` (→ `tps * 100` for the standard simulation)     | Override for non-TPS simulations.    |
| `extra_jvm_args`    | `{}`                                                   | Pass-through `-DkEY=value` flags.    |
| `request_timeout_ms` | `60_000`                                              | Gatling HTTP timeout in ms (60s default matches stock Gatling + reference repos). KO = saturation signal. |

### `container:`

Two ways to pick the perf-test image:

1. **Recipe-driven (recommended)**: `container.recipe: qwen3_5`. The
   recipe resolves `perf_test_image_uri` to the DJL inference base
   image (`djl-inference:0.36.0-lmi25.0.0-cu130`) and overlays the
   `qwen3_5-perf-test.txt` requirements file.
2. **Explicit override**: `container.perf_test_image_uri: <uri>` or
   `container.image_uri: <uri>` for a custom image.

## Available simulations (load harnesses)

Five chat-completion simulations ship in the vendored project. They differ
on **two axes**: (a) the **injection model** — how virtual users / requests
are scheduled — and (b) the **data source** — where prompts come from.
Picking the right one is the difference between measuring *"can the server
hold the SLO at this arrival rate?"* and *"what is the maximum this one box
can sustain?"* — they answer different questions.

### Injection models — the key concept

| Model | What it does | What it measures | When to use |
| --- | --- | --- | --- |
| **Throttled-closed** | Large standing pool (`tps × 100` users) **plus** a `.throttle(jumpToRps(tps))` cap. Send rate is paced to the target TPS. | Whether latency/KO hold at a **requested arrival rate**. Reported throughput ≈ the TPS you asked for (until the server can't keep up). | Reproducing a paced-TPS methodology; "does the box hold the SLO at X TPS?" |
| **Unthrottled-closed** | Fixed pool of exactly `N` users, **no throttle**. Each user loops send→wait→send flat-out. | The **maximum sustained throughput** at concurrency `N` (Gatling `mean req/s`), plus latency at that point. Cannot produce runaway timeouts. | Sizing — "what is the per-instance TPS ceiling?" Sweep `N` to trace the throughput-vs-concurrency curve. |
| **Open** | Fixed **arrival rate** (`constantUsersPerSec(tps)`), decoupled from server response time. New requests arrive on schedule regardless of whether old ones finished. | Real-world arrival behavior; queue build-up under sustained load. | Modeling production traffic that doesn't back off. (Note: the open path has a known transport issue — see Troubleshooting — prefer closed models for trustworthy numbers today.) |

Set the injection model with `gatling.injection_model` (`closed` or `open`)
where the simulation supports it.

### The simulation classes

| Class (FQCN suffix) | Injection model | Data source | `max_tokens` | Use for |
| --- | --- | --- | --- | --- |
| `ThrottledTpsCsvSimulation` | throttled-closed (`tps × 100`) | `prompts.csv` + `system-message.txt` | **600** (hardcoded in template) | **Default / simplest first run** — needs only a prompt CSV. ⚠ short output — not representative of long-generation workloads. |
| `JsonlChatCompletionsSimulation` | **`open` or `closed`** (honors `injection_model`) | `prompts.jsonl` | from `chat-config.json` | Flexible JSONL-driven runs. Open path for arrival-rate tests; closed path derives pool from `concurrent_users` (or `tps × latency_budget_seconds`). |
| `ClosedConcurrencySimulation` | unthrottled-closed (fixed `N`) | `prompts.jsonl` | from `chat-config.json` | **Per-instance sizing.** Sweep `tps_levels` (reused as `N`); read `mean req/s` at the largest `N` whose p95 ≤ SLO. |
| `ThrottledTpsJsonlSimulation` | **throttled-closed (`tps × 100`)** — same profile as the CSV default | `prompts.jsonl` | from `chat-config.json` | **Fixed-target-TPS SLO check on production-shaped data.** Paces sends to a target TPS over real JSONL prompts — answers "does the server hold the SLO at N TPS?" and isolates output length when reconciling against the CSV baseline. |

All four live in the OSS Gatling project under `org.foresightrl.perf`. Extension
packages can add their own; see "Adding a new simulation" below.

**Choosing quickly:**

- *"How many instances do I need?"* → `ClosedConcurrencySimulation` (sweep `N`, read the SLO-respecting ceiling).
- *"Does one box hold the SLO at a target TPS?"* / *"reproduce the reference test on my real data"* → `ThrottledTpsJsonlSimulation`.
- *"Model production arrival patterns"* → `JsonlChatCompletionsSimulation` with `injection_model: open`.
- *"Just prove the pipeline works"* → `ThrottledTpsCsvSimulation` (the default; a prompt CSV is all you need). Prefer the JSONL classes for real measurements so output length matches production.

### `concurrent_users` / `latency_budget_seconds` interaction

For **closed** models the standing pool size matters:

- `ClosedConcurrencySimulation` uses the pool directly as the concurrency `N`.
- `ThrottledTpsJsonlSimulation` **hardcodes `tps × 100`** (the reference
  formula) and only honors an explicit `concurrent_users` override — so it
  matches that paced-TPS formula regardless of runner config.
- `JsonlChatCompletionsSimulation` (closed path) and the runner default the
  pool to `concurrent_users` if set, else `tps × latency_budget_seconds`
  (Little's Law). The legacy `tps × 100` over-provisioned ~5× and produced
  spurious client-timeout KOs on a healthy server — hence the latency-budget
  default.

Open models ignore the pool entirely (they use `constantUsersPerSec(tps)`).

## Adding a new Gatling simulation

The vendored project lives at
`foresightrlcore/perf_test/gatling/src/gatling/java/com/intuit/film/test/`.
To add a new simulation:

1. Drop a new `.java` file alongside the existing simulations, under
   `foresightrlcore/perf_test/gatling/src/gatling/java/org/foresightrl/perf/`.
   Subclass `io.gatling.javaapi.core.Simulation`. For JSONL-driven request
   bodies, copy the feeder + `requestBodyBuilder` from
   `JsonlChatCompletionsSimulation`; for CSV, use
   `ChatRequestTemplates.template`.
2. Read the JVM properties the runner sets: `-DsimulationTps`,
   `-DsimulationDuration`, `-DconcurrentUsers`, `-DinjectionModel`,
   `-DsimulationWorkspace` (and `-Dgatling.http.requestTimeout`).
3. Set `gatling.simulation_class: org.foresightrl.perf.MyNewSimulation`
   in your YAML.

No Python or `build.gradle` changes are required — Gradle's Gatling
plugin auto-discovers classes on the classpath. Refer to
`foresightrlcore/perf_test/gatling/README.md` for project layout.

### Simulations that don't belong in the OSS project

A simulation carrying something deployment-specific — a served-model name,
tuned sampling parameters, or a legacy class name existing configs still
reference — belongs in an extension package instead. Put it in a Java package
tree under `<package>/perf_test/gatling_java/` and list the package in
`container.extra_packages`:

```yaml
container:
  extra_packages: ["foresightrl_intuit"]

perf_test:
  gatling:
    simulation_class: com.example.ext.GptOssEaglePerformanceSimulation
```

The runner merges that tree into `src/gatling/java/` before Gradle runs, so the
extension's classes compile as one source set with the OSS ones and can import
`org.foresightrl.perf.*`. The same `extra_packages` list already drives Python
plugin loading, so there is nothing extra to configure.

## Output

For a config with `output.s3_path: s3://bucket/perf-results/exp/` and run name
`eagle3-tp4`:

```
s3://bucket/perf-results/exp/
├── source/sourcedir.tar.gz       # the source code SageMaker downloads
├── config/args.yaml              # the resolved YAML config
└── eagle3-tp4/
    └── output/data/
        ├── perf_results.json     # machine-readable results
        ├── perf_results.md       # human-readable summary
        └── gatling/eagle3-tp4/tps_*/  # full Gatling HTML reports
```

`perf_results.json` matches `foresightrlcore.perf_test.spec.RunResult`.

## Why the DJL inference base image?

The DJL inference image already ships with Java Corretto 17, vLLM (via
LMI), CUDA 13.0, and Python 3.12. Only Gradle needs to be installed at
container startup (~30s overhead). Avoiding a custom-baked image keeps
deployment simple while still supporting different model recipes — the
recipe system only swaps the pip requirements layer, not the base image.

## Troubleshooting

- **`/opt/ml/input/data/perf_data` missing**: `gatling.data_s3_path`
  failed to mount. Confirm the S3 prefix exists and the SageMaker role
  has `s3:ListBucket` + `s3:GetObject` on it.
- **vLLM `/health` never returns 200**: increase
  `serving.startup_timeout` (default 1800s). Large quantized models or
  speculative-decoding setups can take longer than expected on cold
  start.
- **Gradle fails to download dependencies**: the SageMaker container
  needs outbound access to `services.gradle.org` and Maven Central. Use
  a custom container with these mirrored if you're behind a strict
  egress policy.
- **`global_stats.json not found`**: usually means Gatling crashed
  before producing reports; check container logs for the simulation
  stack trace.
