# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment Constraint

**This machine has no GPU.** Code is written/reviewed locally and executed on a remote GPU machine or via SageMaker training jobs. Never suggest running training, vLLM inference, or any GPU-dependent code locally. Debug exclusively via logs, stack traces, and static analysis. Each round-trip to the GPU machine is expensive — propose well-reasoned fixes aimed at first-attempt success. Lightweight verification of non-GPU logic (config parsing, data preprocessing, reward function pure logic) can be done with `python -c`.

## AWS Region

The local AWS config has no default region. **Always prefix AWS/SageMaker commands with `AWS_DEFAULT_REGION=us-west-2`**, e.g. `AWS_DEFAULT_REGION=us-west-2 rl-train train --config ...`.

## Quality Gate

**Every code change must pass `uv run tox` before it is considered complete.** Run after every change:

```bash
source .venv/bin/activate && uv run tox
```

This runs isort, black, mypy (lint env) and pytest across py311/py312/py313. If lint fails, auto-fix with `uv run isort .` and `uv run black .`, then fix mypy manually. Re-run until exit code 0. For a faster feedback loop, run `uv run tox -e lint` first.

## Setup

Dependencies are managed via `pyproject.toml` with optional extras. Install with:

```bash
uv sync --all-extras --all-groups   # recommended (all deps + dev/test)
uv pip install -e ".[aws]"         # core + SageMaker support
```

For local GPU machines, install the base package and generate GPU deps from a recipe:

```bash
uv pip install -e "."
rl-train setup-local-gpu --recipe qwen3_5   # generates local-requirements.txt
pip install -r local-requirements.txt
```

For SageMaker job containers, dependencies are in `foresightrlcore/runtime/scripts/requirements.txt`.

## Architecture

### OSS vs Intuit monorepo split

The publishable OSS core is `foresightrlcore/` at the repo root. Intuit-specific
code (advice task, Genos/IDPS clients, BTS optimizer presets, the evaluate
benchmark catalog + Aleta reporter + legacy leaderboard metric + per-repository
harness shims, internal smoke tests) lives under `intuit/foresightrl_intuit/`
and is **excluded from the OSS wheel** (`source-exclude` in root
`pyproject.toml`). The rule for what belongs there is *names an internal host,
bucket, account, service, or repository* — not *is useful to us*; anything
structural goes upstream. See [`intuit/README.md`](intuit/README.md).

Install both for internal development:

```bash
uv pip install -e ".[aws,optimizer]"
uv pip install -e intuit/
```

Extensions register via setuptools entry points (`foresightrlcore.plugins`,
`foresightrlcore.tasks`, `foresightrlcore.llm_providers`, etc.). See
[`intuit/README.md`](intuit/README.md).

**Shipping an extension to a remote job (`container.extra_packages`)**: entry
points only work where the extension is pip-installed. SageMaker and HyperPod
jobs run from a staged source directory with no installed metadata, so an
extension listed in `container.extra_packages` is handled explicitly:

1. Each launcher calls `stage_extra_packages()`
   (`execution/sagemaker_common.py`), which resolves the package by import name
   and copies its source tree next to `foresightrlcore` in the job source dir.
2. The same launcher sets `FORESIGHT_EXTRA_PACKAGES=<comma-separated names>` in
   the container environment.
3. Inside the job, `load_plugins()` / `load_optimizer_plugins()` call
   `load_extra_packages()`, which imports each name and invokes its
   `_register()` hook — no `entry_points()` lookup involved.

Any config that uses an Intuit-only task, judge, dataset, or provider must list
`extra_packages: ["foresightrl_intuit"]` under `container:`. Without it the job
fails at runtime with `Unknown task type 'advice'` or
`unknown judge: 'sections'`. The field is on the shared `ContainerConfig`, so it
works for training, optimizer, and perf-test configs alike.

**Extension Gatling simulations**: the same `extra_packages` list also carries
Java. The OSS Gatling project ships only provider-neutral simulations under
`org.foresightrl.perf`; anything deployment-specific lives in the extension at
`foresightrl_intuit/perf_test/gatling_java/<java/package/path>/`. Before Gradle
runs, `PerfTestRunner._stage_extension_simulations()` copies each listed
package's `perf_test/gatling_java/` tree into `src/gatling/java/`, so both
compile as one source set and extension classes can import
`org.foresightrl.perf.*`. A perf config naming
`com.example.ext.*` therefore needs `extra_packages` set, exactly like an
Intuit task does. `tests/perf_test/test_gatling_sources.py` guards the split
(no vendor identifiers in the OSS tree, packages match directories, extension
imports still resolve) — CI has no JDK, so these are structural checks, not a
compile.

**OSS isolation check**: `uv run tox -e oss` runs the suite with
`FORESIGHT_OSS_ONLY=1` and `pythonpath=.`, so `intuit/` is off `sys.path` and the
core is exercised standalone. Tests needing the extension are marked
`@pytest.mark.requires_intuit` and skipped there; they still run in the `intuit`
env. That env also runs `scripts/check_intuit_fingerprints.sh`.

### Package Structure

The canonical package is `foresightrlcore/`. The `foresightrlcore/runtime/scripts/` directory contains the SageMaker container's bash entrypoint and the training driver — no plugin code lives there. All Python plugins (tasks, clients) live under the canonical package and are shipped to the container by the launcher's package copy.

```
foresightrlcore/
  cli.py                    # rl-train CLI (train, validate, setup-local-gpu, perf-test, validate-perf, optimizer, validate-optimizer, evaluate, validate-evaluate, list-benchmarks, serverless)
  config/                   # Pydantic schema, profiles, validation
  config/recipes.py         # Container + local-GPU recipe registries; ECR coordinates
  config/requirements/      # Single-source pip requirement files (used by launcher + Docker)
  operations/               # Operation protocol (train, perf_test, optimizer, evaluate; future: test, deploy)
  methods/                  # MethodHandler protocol (grpo, future: sft, dpo)
  execution/                # ExecutionBackend protocol (aws_training_job, aws_serverless, local)
  auth/                     # Secret resolution (local provider; IDPS via foresightrl-intuit)
  clients/                  # Bedrock LLM client (Genos via foresightrl-intuit)
  common/                   # Shared plumbing: io.py (S3 + CheckpointWriter), retry.py,
                            # sampling.py, vllm_process.py (serve cmd/env + supervision)
  generation/               # Provider generation clients (bedrock/local-vllm/stub) + factory
  inference/                # Model-as-endpoint substrate (see "Inference Substrate")
    proxy.py                # stdlib OpenAI-compatible HTTP server over any ChatBackend
    backends.py             # ManagerChatBackend (any resolve_llm_manager provider), EchoChatBackend
    local_vllm.py           # `vllm serve` lifecycle exposed as an endpoint
    client.py               # minimal urllib OpenAI chat client
  tasks/                    # OSS task plugins (email_outcome, thinking_utils)
  plugins.py                # Entry-point loader for extension packages
  serverless/               # Generic helpers for Serverless reward Lambdas + dataset converters
    registry.py             # LAMBDAS, DATASETS, CONVERTERS dicts + register_*/resolve_* helpers
    spec.py                 # RLVRRecord pydantic model + SCHEMAS registry + Mapper typedef
    bundler.py              # Single-file Lambda bundler (SageMaker Studio paste path)
    convert.py              # Generic JSONL conversion loop (pluggable per --schema)
    cli.py                  # `rl-train serverless {register,bundle,convert,list}` impl
    helpers/                # BedrockJudge, coerce_batch, extract_response/state — stdlib + boto3
  perf_test/                # `perf_test` operation: vllm serve + Gatling load tests
    runner.py               # container orchestrator (start vllm, poll /health, sweep TPS)
    spec.py                 # result schemas (TpsResult, RunResult)
    cli.py                  # `rl-train perf-test` / `validate-perf` impl
    gatling/                # vendored Gatling Java project, package org.foresightrl.perf
  optimizer/                # `optimizer` operation: GenAI quality + cost evaluation
    interfaces.py           # dataclasses (Record/CallUsage/GenResult/JudgeScore) + protocols
    registry.py             # DATASETS / PROMPT_SETS / MODES / JUDGES registries
    runner.py               # cross-product orchestrator (generate→judge→cost, checkpoint/resume)
    retry.py                # backoff + auth force_refresh wrapper; io.py: S3 + JSONL checkpoint
    generation/             # bedrock / local-vllm / stub clients + factory (genos via intuit)
    judges/                 # LLM judges (correctness/hallucination); BTS deterministic judges in intuit
    modes/                  # generic single_call (BTS summary_state in intuit)
    datasets/               # generic parquet provider (bts_parquet in intuit)
    prompts.py              # YamlPromptSet helper (bts_v1 in intuit)
    spec.py                 # result schemas (RowResult, OptimizerResult)
    cli.py                  # `rl-train optimizer` / `validate-optimizer` impl
  evaluate/                 # `evaluate` operation: benchmarks x models -> leaderboard
    interfaces.py           # two-tier protocols (RecordBenchmark / HarnessBenchmark) + carriers
    registry.py             # BENCHMARK_TYPES / DATASET_LOADERS / METRICS / REPORTERS
    catalog.py              # benchmarks-as-data loader; catalog/benchmarks.yaml is built-in
    runner.py               # cross-product orchestrator (endpoint bracketing, trials, resume)
    workspace.py            # source staging (git/s3/path) + per-trial workspaces + venvs
    benchmarks/             # mcq, generative (record tier); external_process, plugin,
                            # scenario_suite, lm_eval (harness tier)
    datasets/               # csv / jsonl / parquet / scenario_dir / inline loaders
    metrics/                # exact_match, substring_match, choice_match, llm_judge, aggregate
    reporters/              # table (parquet + leaderboard CSV + markdown), json
    spec.py                 # result schemas (SampleResult, BenchmarkResult, EvaluateResult)
    cli.py                  # `rl-train evaluate` / `validate-evaluate` / `list-benchmarks`
  runtime/scripts/          # Container glue only: entrypoint.sh, train.py, perf_entrypoint.sh,
                            # optimizer_entrypoint.sh, evaluate_entrypoint.sh,
                            # resolve_secrets.py, nccl_env.py,
                            # ds_zero{2,3}_config.json, requirements.txt(_torchrun)

intuit/docker/              # Internal ECR images (stripped from the public tree)
  Dockerfile.base           # Shared base: DLC + idps-sdk
  Dockerfile.qwen3          # qwen3 vllm variant
  Dockerfile.qwen3-torchrun # qwen3 no-vllm variant
  Dockerfile.qwen3_5        # qwen3_5 vllm variant
  build.sh                  # Build + tag + push helper

examples/configs/           # YAML config templates (user-owned, not committed)
intuit/examples/prompts/           # Static prompt text files referenced by configs (per-task subfolders, e.g. advice/)
examples/smoke_tests/       # Per-task smoke-test YAMLs (e.g. advice/serverless_*.yaml)
intuit/examples/legacy/            # Archived standalone scripts (historical reference)
```

### Config-First Principle

**All training parameters flow through YAML config files.** The canonical schema is `foresightrlcore.config.schema.TrainingJobConfig`. Config files are user-owned and not committed to the repo — `examples/configs/` has templates.

### Launch Flow

```
rl-train train --config job.yaml
  → TrainingJobConfig.from_yaml("job.yaml")
  → config.execution.type → get_backend("aws_training_job")
  → TrainOperation.run(config, backend)
  → backend.launch(config)
```

### Operations Layer

Top-level actions registered in `foresightrlcore/operations/`:
- `train` — current focus, fully implemented
- `perf_test` — Gatling load tests against a `vllm serve` process; see `docs/launching-performance-test.md`. Five pluggable simulation harnesses differ by **injection model** (throttled-closed = paced to a target TPS; unthrottled-closed = fixed `N` run flat-out for per-instance sizing; open = fixed arrival rate) and **data source** (CSV vs production `prompts.jsonl`). For sizing use `ClosedConcurrencySimulation` (sweep `N`, read SLO-respecting `mean req/s`); to reproduce reference TPS numbers on real data use `ThrottledTpsJsonlSimulation`. The "Available simulations" section of the doc is the catalog.
- `optimizer` — GenAI quality + cost evaluation across candidate LLMs; see `docs/launching-optimizer.md`. Batch cross-product `(dataset × model × mode)` generate→judge→cost, producing `results.parquet` + `summary.md`. Generic and registry-driven (four pluggable axes: datasets, prompt sets, generation modes, judges), with the **BTS "State of the Business"** use case shipped fully configured. Generation supports `genos` / `bedrock` / `local` (in-job `vllm serve`); LLM judges reuse `resolve_llm_manager` + structured output (correctness/hallucination prompts ported from `openevals` as constants). Resilient retry+refresh, JSONL checkpoint/resume, and automatic infra sizing (CPU for API-only, GPU when any candidate serves locally). Backends: `aws_training_job` + `local` only. See `docs/launching-optimizer.md` for the config schema (`config/optimizer_schema.py`) and extension recipe.
- `evaluate` — benchmark suites against candidate models, producing a leaderboard; see `docs/launching-evaluate.md`. Cross-product `(benchmark × model)` repeated over `trials`, writing `benchmark_scores.csv` + `summary.md` + `results.parquet` + `evaluate_results.json`. Registry-driven on four axes (benchmark types, dataset loaders, metrics, reporters) plus a **benchmark catalog** so adding a benchmark is a YAML entry. Backends: `aws_training_job` + `local` only. Two design points carry the operation:
  - **Model-as-endpoint.** Every candidate is served as an OpenAI-compatible HTTP endpoint (see "Inference Substrate"), so a benchmark never learns its provider. That is what lets a benchmark living in another repository run unmodified via `source: {git, ref}` — no vendoring.
  - **Checkpoints are keyed on what they measured**, not just on names. Each row carries a `fingerprint` (`runner.measurement_fingerprint`) over the resolved benchmark spec and the model config, so editing a prompt, swapping a metric, or repointing a dataset invalidates exactly the affected trials and they re-run; an unchanged config still resumes. Rows whose benchmark or model has left the config are dropped rather than reported. Without this, a re-run replays stored generations and reports numbers a since-deleted configuration produced — which is not hypothetical, it silently returned stale SageMaker scores.
  - **Two benchmark tiers**, detected from which protocol a type implements, with no flag. `RecordBenchmark` (`build_prompt` + `score`) hands the loop to the runner, which owns concurrency, retries, cost accounting, and record-level resume. `HarnessBenchmark` (`run(ctx)`) owns its own loop and receives a workspace, a staged source tree, and a live endpoint — this is the tier for agentic suites, external processes, and `lm-eval`.
- `test`, `deploy` — future, same protocol

### Methods

Registered in `foresightrlcore/methods/`:
- `grpo` — GRPO via TRL's GRPOTrainer (fully implemented)
- `dpo` — currently via SageMaker serverless (`aws_serverless` backend); future: also via TRL's DPOTrainer on `aws_training_job`/`local`
- `rlvr`, `rlaif` — via SageMaker serverless trainers (`aws_serverless` backend)
- `sft`, `kto` — future, same pattern

Method handlers use the **`TrlMethodHandler` base class** (`methods/base.py`) which provides the shared training pipeline (task loading, datasets, LoRA, hooks, train/save). Concrete handlers like `GrpoMethodHandler` subclass it and add only method-specific validation and trainer args. The `TRL_TRAINERS` registry in `methods/__init__.py` maps method type strings to TRL trainer/config class dotted paths.

**Liger kernel opt-in** (`training.use_liger_kernel`, default `False`): when `True`, `build_trainer_args` forwards `use_liger_kernel=True` to the TRL trainer config, which routes the model through `LigerForCausalLM` for fused RoPE/SwiGLU/LayerNorm/cross-entropy ops (~10–20% throughput on Qwen-class models). The `liger-kernel >=0.5.0` wheel is bundled in both the qwen3_5 container recipe (`config/requirements/qwen3_5.txt`) and the qwen3_5 local-GPU recipe (`LOCAL_GPU_RECIPES["qwen3_5"]` in `config/recipes.py`). For other recipes the package must be added explicitly. Because the flag lives on the base handler, every TRL-backed method (GRPO/SFT/DPO/RLVR/RLAIF) picks it up automatically.

The SageMaker entry point (`runtime/scripts/train.py`) uses **`get_method()` registry dispatch** — no hardcoded if/elif per method. Adding a new TRL method requires only a thin subclass and `register_method()` call.

Method handlers support **training hooks** (`training.hooks` in YAML) for customizing TRL behavior: extra `TrainerCallback` classes, pre/post training callables, and custom trainer subclasses — all specified as dotted import paths. See `foresightrlcore/hooks.py` and `docs/tasks/training-hooks.md`.

**Cross-validation**: `config/validation.py` verifies that `method.type` matches the corresponding `training.<method_type>` hyperparameter block and warns if mismatched blocks are present.

### Execution Backends

Registered in `foresightrlcore/execution/`:
- `aws_training_job` — SageMaker Training Jobs (SDK v2/v3)
- `aws_serverless` — SageMaker Serverless Model Customization (SDK v3, DPO/RLVR/RLAIF)
- `local` — local GPU via accelerate/torchrun

### Serverless Model Customization (`aws_serverless`)

The `aws_serverless` backend wraps SageMaker SDK v3's serverless trainers (`DPOTrainer`, `RLVRTrainer`, `RLAIFTrainer`). Unlike `aws_training_job`, it requires no infrastructure management — no instance types, topology, vLLM, DeepSpeed, or custom containers. AWS runs the training loop; we configure and submit.

- **Config**: `execution.type: aws_serverless` with `execution.serverless` section containing `model_id`, `accept_eula`, dataset/MPG names, and method-specific fields
- **Hyperparameters**: `training.serverless` section (`global_batch_size`, `max_epochs`, `learning_rate`, `extra`)
- **Resource resolution**: Datasets, evaluators, and model package groups are referenced by AI Registry name. RLAIF requires `reward_model_id` and `reward_prompt_name`
- **Schema**: `ServerlessConfig` and `ServerlessHyperparametersConfig` in `config/schema.py`
- **Validation**: `_validate_serverless()` in `config/validation.py`; infra/vLLM checks are skipped for serverless
- **Import caveat**: Trainer imports trigger `boto3.client("sagemaker")` at import time; the backend sets `AWS_DEFAULT_REGION` before importing
- **In-transit encryption** (`execution.serverless.encrypt_inter_container_traffic`, default `true`, ACP IS031.1): the SDK's serverless trainers hardcode their `TrainingJob.create(...)` args and expose no encryption parameter, so `serverless_inter_container_encryption` (in `execution/sagemaker_common.py`, shared with the training-job launchers) wraps `TrainingJob.create` around the submit call to inject `enable_inter_container_traffic_encryption`. Fail-fast if a future SDK drops the parameter; set the field to `false` to skip patching entirely
- **Example configs**: `examples/configs/serverless_rlvr.yaml`, `serverless_rlaif.yaml`, `serverless_dpo.yaml`
- **Future**: Schema includes placeholder fields for evaluation and deployment (unused in Phase 1)

### Infrastructure Modes

**Generation backend** (`infra.vllm_mode`):
- `server`: vLLM as separate process on dedicated GPUs
- `colocate`: vLLM inside TRL trainer process
- `none`: No vLLM — torchrun with native HF generation

**Speculative decoding** (`infra.vllm.speculative`, optional, **server-mode only**): forwards a JSON object verbatim to `vllm serve --speculative-config`, supporting EAGLE/Medusa custom drafts and built-in MTP heads (Qwen3-Next). Schema: `SpeculativeConfig` in `config/schema.py` with `extra="allow"` for forward-compat. Validation warns and ignores the field for `colocate`/`none` since TRL's `GRPOConfig` has no speculative kwarg today. See `docs/speculative-decoding.md` and `examples/configs/sagemaker_advice_v4_run_speculative.yaml`.

**vLLM single-entry-point launcher** (`runtime/scripts/vllm_launcher.py`, **all `vllm_mode = server` paths**): every backend that starts vLLM in server mode — SageMaker `entrypoint.sh` (single_node, multi_node, multi_node_server, dedicated_vllm), local-GPU `execution/local.py`, and HyperPod EKS `aws_hyperpod_eks/manifests.py` — invokes this launcher instead of `trl vllm-serve` directly. It does two things before importing TRL: (a) applies spawn-safety env defaults (`VLLM_WORKER_MULTIPROC_METHOD=spawn`, `CUDA_MODULE_LOADING=LAZY`, `VLLM_USE_V1=0`, `VLLM_ALLOW_INSECURE_SERIALIZATION=1`) via `setdefault` so operators can override, then calls `multiprocessing.set_start_method("spawn", force=True)`; (b) reads three opt-in env vars (`VLLM_REASONING_CONFIG_JSON`, `VLLM_COMPILATION_CONFIG_JSON`, `VLLM_DISABLE_CUSTOM_ALL_REDUCE`) and calls `patch_trl_vllm.patch(...)` to inject `reasoning_config`/`compilation_config`/`disable_custom_all_reduce=True` into TRL's `LLM(...)` constructor. Without the launcher, vLLM TP workers segfault on H100/H200 with "Cannot re-initialize CUDA in forked subprocess" because exporting the env var from bash is too late — Python's `multiprocessing` module caches its default context as soon as TRL's CLI module is imported. The patch is fail-fast (non-zero exit if TRL upstream rewrote the marker line) so silent feature regressions are impossible. Backends populate the patch env vars from YAML: SageMaker via `entrypoint.sh`, local via `LocalBackend._build_vllm_patch_env`, EKS via `KubeflowManifestBuilder._build_vllm_patch_env_lines`. See `runtime/scripts/vllm_launcher.py` and `runtime/scripts/patch_trl_vllm.py` for the contracts.

**Reasoning + CUDA-graph control** (`infra.vllm.reasoning_parser`, `reasoning_config`, `compilation_config`, `disable_custom_all_reduce`, optional, **server-mode only**): enables vLLM's `ThinkingTokenBudgetLogitsProcessor` (thinking models like Qwen3.5), bounded CUDA-graph capture (avoids the H100 graph-compile hang), and the disable-custom-all-reduce workaround for vLLM 0.20.x's broken kernel at TP>1. Delivery is via the vLLM launcher (see above) — the launcher invokes `patch_trl_vllm.patch(...)` to inject the kwargs into TRL's `LLM(...)` call. Validation warns and ignores the fields for `colocate`/`none`. See `docs/reasoning-and-cuda-graph.md`.

**qwen3.5 container runtime hardening** (`runtime/scripts/entrypoint.sh`): three additive guards required for H100/P5 reliability — (1) pip installs run with `CUDA_VISIBLE_DEVICES=""` so deepspeed's CUDA probe doesn't leave a parent CUDA context that breaks vLLM's spawn workers, (2) `VLLM_WORKER_MULTIPROC_METHOD=spawn` is enforced inside `vllm_launcher.py` (not just exported by bash) so the spawn context is set BEFORE TRL imports torch, (3) `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=0` + `VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING=0` on the vLLM subprocess `env -u PYTORCH_CUDA_ALLOC_CONF ...` invocations (`start_vllm_server`, `run_multi_node`) to avoid the Inductor autotune CUDA-graph hang. Training keeps default Inductor behavior; only the vLLM subprocess is affected.

**Placement topology** (`infra.topology`, relevant for `vllm_mode: server`):
- `single_node`: GPU split on one instance (default)
- `multi_node`: Heterogeneous SageMaker instance groups (SDK v2 only)
- `multi_node_server`: Homogeneous cluster, node 0 runs vLLM split
- `dedicated_vllm`: Node 0 all GPUs → vLLM, node 1+ → training. Homogeneous by default (`instance_type` × `instance_count`). Opt-in mixed types: set `infra.vllm_instance_type` to submit SageMaker `InstanceGroups` (`vllm` + `training`) via boto3; `instance_count` stays total cluster size. Mixed roles use `SM_CURRENT_INSTANCE_GROUP`; host lists come from `SM_TRAINING_ENV.instance_groups_dict` (`algo-1` is not necessarily vLLM). SageMaker `CheckpointConfig` is omitted for mixed jobs (two-way S3 sync from every InstanceGroup races DeepSpeed shards); training ranks `aws s3 sync` `/opt/ml/checkpoints` to `checkpoint_s3_path` instead. Same GPU family/EFA required unless `allow_mixed_gpu_families: true`. VPC jobs must use non-EFA types (SageMaker InstanceGroups reject EFA in VPC mode). See `docs/launching-sagemaker-training-job.md`.

**Recommended settings for all multi-node `vllm_mode: server` configs:**
- **Never set `enforce_eager: true`** — disables CUDA graph compilation in vLLM, measured 13.7% throughput penalty. Schema default is `false`; do not override it.
- **Scale `vllm_gpu_count` with instance count** — on 8-GPU nodes: 2-inst → `vllm_gpu_count: 2, TP: 2`; 4-inst → `vllm_gpu_count: 4, TP: 4`. Keeping 2 GPUs at 4 instances causes vLLM bottleneck (5.6 vs 20.5 steps/hr measured).
- **Use `ds_zero2_lora_config.json`** for LoRA training — 50MB buckets vs 500MB default; properly sized for LoRA gradient volumes (~100–200MB).
- **Remove `NCCL_PROTO: "Simple"`** from environment — confirmed no-op; let NCCL auto-select.
- **Add EFA buffer vars** to multi-node environment: `NCCL_BUFFSIZE: "8388608"`, `NCCL_P2P_NET_CHUNKSIZE: "524288"`.
- **Never set `NCCL_ALGO: "Tree"`** — incompatible with AllGather ops used by vLLM, causes immediate failure.

### Task Framework

Tasks are pluggable **packages** under `foresightrlcore/tasks/<task>/`, each providing reward functions, callbacks, dataset loaders, an optional serverless Lambda, and an optional JSONL converter. The `advice` task supports `reward_mode: llm_judge | mock`.

Standard per-task file layout:

| File              | Role                                                           |
|-------------------|----------------------------------------------------------------|
| `trainer.py`      | TRL reward function, dataset loader, callbacks                 |
| `judge_logic.py`  | Stdlib-only shared logic — prompts, scoring math, etc.         |
| `lambda_reward.py`| AWS Lambda handler used by SageMaker Serverless RLVR           |
| `convert.py`      | Record-level mapper used by `rl-train serverless convert`      |
| `__init__.py`     | Re-exports protocol functions + registers lambda/converter/dataset |

The `__init__.py` registers entries in `foresightrlcore.serverless.registry` **before** re-exporting the trainer (which pulls heavy training-stack imports), so `rl-train serverless list` and bundling work on machines without `transformers` / `torch`.

### Serverless Section

`foresightrlcore/serverless/` is a peer of `tasks/` holding only framework-grade code shared across tasks — no task-specific prompts, criteria, or field mappings:

- **`registry.py`** — `LAMBDAS`, `DATASETS`, `CONVERTERS` dicts; one identifier (the SageMaker AI Registry evaluator name) doubles as the registry key. The backend `_resolve_evaluator` falls back to `registry.LAMBDAS` when no `custom_reward_function_source` is set.
- **`spec.py`** — pydantic schemas registered in `SCHEMAS`. Today: `RLVRRecord`. Reserved keys: `sft`, `dpo`, `kto` (raise `NotImplementedError` until implemented).
- **`bundler.py`** — generalized AST bundler. Drives the single-file Lambda bundle from a `LambdaSpec` (`# BUNDLER_STRIP_BEGIN`/`END` markers in the source).
- **`convert.py`** — generic JSONL loop with per-line skip-on-error, schema validation, and YAML-mapping-spec support for no-code converters.
- **`helpers/`** — `BedrockJudge` (boto3 `converse` wrapper with retry), `coerce_batch` / `extract_response` / `extract_state` (Lambda event shredding). Stdlib + boto3 only.

The `rl-train serverless` CLI (`register`, `bundle`, `convert`, `list`) replaces the previous standalone scripts in `serverless/advice/`.

### Pre-Staged Models (S3 model.name)

`model.name` accepts either an HF Hub identifier (e.g. `Qwen/Qwen3.5-27B`) **or** an S3 prefix (e.g. `s3://bucket/models/Qwen3.6-27B/`). When an S3 prefix is provided, the `aws_sagemaker_launcher` adds a SageMaker `model` input channel; SageMaker downloads the prefix to `/opt/ml/input/data/model` before the entrypoint runs, and both `entrypoint.sh` (for `trl vllm-serve --model ...`) and `train.py`'s `_apply_sagemaker_overrides` swap `model.name` for the local mount path. The container therefore loads weights from local disk and never touches the Hugging Face Hub.

This is the recommended path for any model ≥10B on SageMaker. The motivating failure was a multi-rank race in `huggingface_hub.snapshot_download()` where 6–8 trainer ranks concurrently consult the same cache and one rank reads a stub or mid-write file (`OSError: shard N not found`, with N differing run-to-run). Pre-staging eliminates the cache entirely. The S3 URI must end with `/` (validated by `ModelConfig._validate_name`).

### Container and Local-GPU Recipes

Recipes (`foresightrlcore/config/recipes.py`) are named presets that resolve dependency versions for a model family.

- **Container recipes** (`CONTAINER_RECIPES`): Map a recipe name (e.g. `qwen3`, `qwen3_5`) to `framework_version`, optional `image_uri`, and pip `requirements`. Set `container.recipe: qwen3_5` in YAML — the `ContainerConfig` model validator calls `resolve_container_recipe()` to populate defaults. Explicit fields (`image_uri`, `requirements`, `framework_version`) override recipe values. The SageMaker launcher writes resolved requirements to `requirements.txt` at packaging time.
- **Local-GPU recipes** (`LOCAL_GPU_RECIPES`): Pip-only presets for bare-metal machines (no DLC image or framework_version). Generated via `rl-train setup-local-gpu --recipe <name>` which writes a `local-requirements.txt`.
- **Requirements files** (`foresightrlcore/config/requirements/*.txt`): Single source of truth for pip dependencies. `recipes.py` reads them at import time; Docker builds `COPY` them directly. Edit a `.txt` file to update both the SageMaker launcher and the Docker image in one place.
- **Adding a recipe**: Add a `.txt` file in `config/requirements/`, add entries to `CONTAINER_RECIPES` and `DOCKER_IMAGES` in `recipes.py`, add an `intuit/docker/Dockerfile.<recipe>`, add tests in `tests/config/test_config_recipes.py`, run `uv run tox`.

### Docker Training Images

Pre-built ECR training images live in `intuit/docker/` and are stripped from the public tree. They bake in `idps-sdk` (unavailable at container runtime) on top of the AWS DLC base.

- **ECR coordinates** (`ECR_REPO_NAME`, `ECR_ACCOUNT`, `ECR_REGION` in `recipes.py`): single place to rename the repo or retarget account/region.
- **`DOCKER_IMAGES` dict** in `recipes.py`: maps `(recipe, mode)` tuples to ECR `latest` URIs via `ecr_image_uri()`.
- **Tag format**: `{recipe}-{mode}-{semver}-{git-sha}` (immutable) + `{recipe}-{mode}-latest` (mutable alias).
- **Build**: `intuit/docker/build.sh --recipe qwen3 --mode vllm --account <id> --repo <name> --region <region> [--push]`
- **Full reference**: `intuit/docs/building-docker-images.md`. The public `docs/building-docker-images.md` is only a pointer.

### LLM Client Providers

Tasks that need an LLM judge / scorer configure clients under a single
reusable block: `task.config.llm_clients` is **always a mapping** of
task-defined names to provider-agnostic configurations. Single-client
tasks use the conventional name `default`; multi-client tasks
(e.g. a future RLAIF task) use any names that make sense for them.
Providers can be mixed across clients within the same task. See
[`docs/tasks/llm-client-config.md`](docs/tasks/llm-client-config.md) for
the full reusability contract.

Two providers are supported today (no `langchain-aws` or other new pip
dependency — Bedrock is built on `boto3`):

- **`bedrock`** — AWS Bedrock via `boto3.bedrock-runtime.converse(...)` with an
  in-house shim that exposes `with_structured_output(SchemaCls).invoke(messages)`.
  Auth is via the SageMaker execution role (one-time IAM update:
  `bedrock:InvokeModel` on the target model ARNs). **No `secrets:` section
  needed** when every client uses Bedrock.
- **`genos`** — Intuit Genos LLM Execution via
  `foresightrlcore/clients/genos.py`. Requires `GENOS_APP_SECRET` via the
  `secrets:` block (IDPS or local provider) and non-secret `GENOS_*` env
  vars in the top-level `environment:` section. Genos credentials are
  loaded inside `resolve_llm_manager` — tasks never own env-var reading.

Tasks call `resolve_llm_manager(cfg)` to dispatch on provider; the
returned manager exposes the same `.get()` / `.force_refresh()` surface
regardless of provider.

### Inference Substrate (`foresightrlcore/inference/`)

Turns any candidate model into an **OpenAI-compatible HTTP endpoint**. This is
what the `evaluate` operation is built on, and it is deliberately generic — any
future operation that must hand a model to code it does not own should use it
rather than passing client objects around.

`endpoint_for(model)` is the whole API: a context manager that yields an
`InferenceEndpoint` and tears the server down afterwards, including when the
body raises. Two implementations sit behind it:

- **API providers** (`bedrock` / `genos` / `echo`) → `OpenAICompatProxy`, a
  `http.server`-based server (no FastAPI, no uvicorn — a new web dependency to
  serve one loopback process is not worth it) wrapping a `ChatBackend`.
  `ManagerChatBackend` dispatches through `resolve_llm_manager`, so **every
  registered LLM provider gets an endpoint for free**; registering a new
  provider requires no change here. Routes: `/chat/completions` (streaming and
  not), `/completions`, `/models`, `/health`, with or without the `/v1` prefix.
- **`local`** → `VllmServerEndpoint`, which owns a `vllm serve` process and
  hands back its endpoint once `/health` answers. `vllm serve` already speaks
  OpenAI, so nothing is translated. Command/env construction and process
  supervision live in `common/vllm_process.py`, shared with the optimizer's
  local generation client.

`InferenceEndpoint.env()` produces the **endpoint contract** — the environment
an external process is given: `OPENAI_BASE_URL` / `OPENAI_API_KEY`,
`ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY` (for agent CLIs following that
convention), `EVALUATE_MODEL_ID`, `EVALUATE_MODEL_NAME`,
`EVALUATE_OUTPUT_FILE`, `EVALUATE_WORKSPACE`, `EVALUATE_PYTHON`. A benchmark
that reads different variable names is aliased in YAML via `env:`, never
patched.

In-package record-tier benchmarks use `inference/client.py` (a minimal `urllib`
OpenAI client) rather than calling the backend directly, so they traverse the
same HTTP path an external benchmark does. If the contract breaks, it breaks for
the built-ins first.

### Secrets Management

Secrets (for example IDPS-backed credentials or values from a local `.env` file) are resolved **at worker runtime**, not at launch time. For SageMaker with `provider: idps`, no secret values appear in job metadata, S3 config, or CloudTrail.

- **Module**: `foresightrlcore/auth/` — `SecretsProvider` protocol + `IDPSProvider` and `local` provider (`foresightrlcore.auth.local`, dotenv-style `.env` file)
- **Config**: `secrets:` section in YAML with `provider` (`idps` or `local`), provider-specific connection config, and `entries` list mapping secret paths to env vars
- **Runtime**: `resolve_secrets.py` runs inside the container (or local worker) before training, resolves secrets via the configured provider, and sets them as env vars
- **Providers**: `idps` (Intuit IDPS via `idps-sdk`) or `local` (reads `KEY=VALUE` lines from a `.env` file for local GPU runs where IDPS is unavailable). There is no legacy JSON injection.
- **Non-secret Genos vars**: `GENOS_ENV`, `GENOS_EXPERIENCE_ID`, `GENOS_INTUIT_ORIGINATING_ASSETALIAS`, `GENOS_OFFLINE_JOB_ID` are set as plain env vars in the top-level `environment` section (backend-agnostic; the SageMaker launcher passes this directly as the container environment).

### Container Pipeline (SageMaker)

1. `foresightrlcore/runtime/launcher.py` submits the job
2. `foresightrlcore/runtime/scripts/entrypoint.sh` dispatches inside the container
3. `resolve_secrets.py` resolves secrets as env vars via the configured provider (if `secrets:` section present)
4. `foresightrlcore/runtime/scripts/train.py` parses YAML, resolves the method handler via `get_method()`, applies SageMaker overrides, and runs training

## Account & Permissions Setup

Pre-launch account and IAM setup is documented under `intuit/docs/account-setup/`. Use it whenever a user is setting up a new AWS account, a new IAM role, an IDPS policy, or a new HyperPod cluster — these are the canonical answers for "what permissions / assets do I need before I can run `rl-train train`". The tree is Intuit-only and is stripped from the public package.

```
intuit/docs/account-setup/
  README.md                          # which guide do I need (decision tree)
  aws/
    README.md                        # shared AWS prereqs + per-backend matrix
    intuit-account-provisioning.md   # DevPortal Cloud Account flow, JIT, default roles
    local-gpu.md                     # execution.type: local
    sagemaker-training-job.md        # execution.type: aws_training_job
    sagemaker-serverless.md          # execution.type: aws_serverless
    hyperpod-eks.md                  # execution.type: aws_hyperpod_eks
```

Each per-backend guide follows the same shape: Overview → Required Assets → IAM roles & policies (with JSON) → Intuit-specific bits (DevPortal/IDPS/Genos) → Verification Checklist → Common Failures. The `docs/launching-*.md` references link back to the matching account-setup page in their Quick Start block.

## Key Design Patterns

- **Config-first**: YAML → Pydantic validation → training driver. Never hardcode hyperparameters.
- **Task modularity**: Create `foresightrlcore/tasks/<name>/` package with `trainer.py` (defining `get_reward_function`, `get_callbacks`, `get_dataset_loader`), `judge_logic.py`, `lambda_reward.py` (if serverless), `convert.py` (if needed), and `__init__.py` that registers the lambda/converter/dataset.
- **Method handlers**: Subclass `TrlMethodHandler` in `foresightrlcore/methods/<name>.py`, add method-specific args, call `register_method()`.
- **Execution backends**: Implement `ExecutionBackend` protocol in `foresightrlcore/execution/<name>.py`.
- **Container recipes**: Named presets in `config/recipes.py` resolve DLC image + pip deps per model family. Set `container.recipe` in YAML. Override with explicit `image_uri` or `requirements`.
- **Local-GPU recipes**: Pip-only presets for bare-metal machines. Generate via `rl-train setup-local-gpu --recipe <name>`.
- **Custom container images**: Set `container.image_uri` to a fully-qualified ECR URI to bypass DLC image resolution entirely.
- **Training hooks**: Callbacks, pre/post hooks, and custom trainer classes via `training.hooks` in YAML config. No framework code changes needed.
- **Benchmarks as data**: Adding a benchmark is a YAML entry in a catalog (`evaluate/catalog/benchmarks.yaml`, an extension's catalog, or `evaluate.catalog_paths`). A config entry naming a catalog benchmark is merged over it, entry wins per-field. Adding a benchmark *kind* is one class plus `@BENCHMARK_TYPES.register("name")` — implement `build_prompt`/`score` for the record tier or `run(ctx)` for the harness tier; the tier is detected, not declared.
- **Model-as-endpoint**: Hand a model to code you do not own via `inference.endpoint_for(model)`, never by passing a client object. The consumer sees an OpenAI-compatible URL and cannot tell Bedrock from Genos from a local `vllm serve`, which is what makes an unmodified third-party benchmark runnable.
- **Validate before submit**: `rl-train validate --config job.yaml` catches errors locally.
- **S3 paths**: Required for `aws_training_job`; optional for `aws_serverless` (uses AI Registry dataset names); local paths for local backend. Validated per-backend.
- **Secrets via `secrets:`**: Use the `secrets:` config section for runtime resolution (`idps` or `local`). Never put secret values in YAML `environment` or hardcode them in scripts.
