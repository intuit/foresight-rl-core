# Library Architecture & Roadmap

**Status:** Draft — planning document for evolving this repository into a reusable training library.

**Audience:** Maintainers and contributors defining package boundaries, extension points, and migration steps.

---

## 1. Goals

| Goal | Implication |
|------|-------------|
| **Multiple training methods** | GRPO, SFT, DPO, and future methods (PPO, ORPO, KTO, etc.) share infrastructure but plug in different trainers, loss wiring, and optional generation stacks. |
| **Multiple execution backends** | Same YAML-centric config must run locally (GPU workstation), on **AWS SageMaker Training Jobs** (current path), and on **other AWS patterns** (e.g. customizable / serverless-style offerings) and **other clouds** (Vertex AI, Azure ML, RunPod, Modal, etc.) without forking the core. |
| **Minimal default install** | Laptop/CI submits jobs and validates configs with **no torch/GPU stack** unless the user opts into `local` / `train` extras. |
| **Config-first** | One training config YAML (plus optional includes) is the source of truth; CLI and Python API only override or point at files. |
| **Modular installs** | `pip install package`, `package[aws]`, etc., plus `rl-train setup-local-gpu --recipe <name>` for GPU machines — consumers pull only what they need. |
| **Stable extension model** | New *methods*, *providers*, and *operations* (eval, test, deploy) are added via documented entry points (or equivalent), not by editing core dispatch logic. |
| **Pluggable rewards (GRPO / RL)** | Training can use a built-in critic, a test/mock path without credentials, or **user-supplied reward code** loaded from a configurable location at launch—without forking the library. |

---

## 2. Current repository inventory

### 2.1 What exists today (strengths)

- **`foresightrlcore/runtime/config.py`** — Pydantic models for `SageMakerTrainingConfig`: job, infra (topology, vLLM modes, instance types, training plans, VPC), model/LoRA, `TrainingConfig` nested `GrpoConfig`, data (S3), output, container, and **task** (`TaskConfig` with `type` + opaque `config` dict). YAML load/validate, `merge_overrides` with dotted keys.
- **`foresightrlcore/runtime/launcher.py`** — `GRPOJobLauncher`: SDK v2/v3 detection, image URI resolution, packaging of `scripts/` for SageMaker `code` channel, boto3 fallback for training plans, heterogeneous topologies (single-node GPU split, multi-node, dedicated vLLM), torchrun vs vLLM requirements split.
- **`foresightrlcore/runtime/scripts/train_grpo.py`** — Container-side driver: loads YAML, resolves SageMaker paths (`SM_*`), dynamic task import, TRL `GRPOTrainer`, vLLM patches (e.g. `enforce_eager`). Logs `task.config.reward_mode` when present.
- **`foresightrlcore/tasks/`** — Plugin contract: `get_reward_function`, `get_callbacks`, `get_dataset_loader` per task module name (`advice` implements production logic). Tasks are part of the installed package and are shipped to SageMaker containers via the launcher's package copy.
- **`advice` task — `reward_mode` (implemented)** — Selects how rewards are computed without changing `task.type`:
  - **`llm_judge` (default):** Full LLM critic pipeline, thinking-length penalties, criterion TensorBoard callbacks (`CriterionLoggingCallback`), credential loading for the critic.
  - **`mock`:** Fixed float reward per completion, optional sleep to mimic judge latency, optional JSONL prediction logging—**no LLM calls, no credentials**. Callback list is empty so TensorBoard does not expect criterion metrics. Used for pipeline/infra tests and dry runs (`foresightrlcore/runtime/scripts/tests/test_advice_reward_mode.py`).
- **`foresightrlcore/runtime/scripts/entrypoint.sh`** — Robust orchestration: vLLM server vs colocate vs torchrun, cleanup traps, YAML parsing helpers.
- **`foresightrlcore/runtime/launch_job.py`** — CLI to submit jobs from a dev machine.

### 2.2 Gaps relative to the vision

- **Single method** — Config and launcher names assume GRPO; `TrainingConfig` embeds `GrpoConfig` unconditionally. GrpoConfig now exposes all TRL GRPOConfig parameters (35 fields) for full parity with upstream TRL.
- **Two config stories** — Legacy scripts (`grpo_matching.py`, `grpo_advice.py`) use **module-level globals**; SageMaker path is YAML-driven. README still documents globals for `grpo_matching.py`.
- **`grpo_matching.py` is a second task type** — Implements F1-based transaction-matching rewards (`reference_matching_reward_func`), its own dataset transform, and thinking-penalty logic. This is **not just legacy**—it represents a **second domain reward** that should migrate to `tasks/matching.py` following the same contract as `advice`. Currently standalone with hardcoded globals, not usable from the YAML path.
- **No installable package metadata** — No `pyproject.toml` / pinned extras; `prerun.sh` is **interactive** (prompts for env name, blocks CI) and assumes SageMaker notebook paths. Neither is suitable for a distributable library.
- **SageMaker-centric layout** — Python package is `sagemaker_training`, which is confusing if the library is vendor-neutral.
- **Heavy container requirements** always bundled for in-container install (`trl[vllm]`, etc.); there is no separate **“launcher-only”** dependency set for CI/laptops (boto3 + pydantic + yaml).
- **TRL version inconsistency** — `requirements.txt` pins `trl[vllm]==0.28.0`; `requirements_torchrun.txt` pins `trl==0.29.1`. Different TRL point releases can change `GRPOConfig` fields, vLLM integration, and trainer APIs. The library should lock to **one TRL version** per release (or document the split and test both).
- **`DataConfig` / `OutputConfig` hardcode S3** — `validate_s3_path` rejects non-`s3://` URIs, but the library's goal is local paths when `execution.type: local`. Validators must become **execution-aware**.
- **`_gpus_per_node` is hardcoded** — `launcher.py` uses `if "g5.12" in it or "g6e.12" in it: return 4; return 8`. Any new instance type needs a code change. Should be derived from instance metadata, an explicit YAML field, or a lookup table.
- **DeepSpeed config is a filename, not inline** — `training.deepspeed` is a relative path resolved at runtime in the container. Both ZeRO-2 and ZeRO-3 JSON files live in `scripts/`. The library should decide: bundle defaults, allow inline YAML, or accept user-provided paths.
- **`TaskConfig.config` is opaque** — `Dict[str, Any]` with no schema validation. Typos in task config keys (e.g. `rewrd_mode`) are **silent** until the container crashes. Future: typed sub-models per `task.type` (discriminated union), or a `model_validator` that warns on unknown keys.
- **No `version` field in YAML schema** — Proposed in section 4.4 but absent from current `SageMakerTrainingConfig`.
- **`gradient_checkpointing_kwargs` not exposed** — `train_grpo.py` hardcodes `{"use_reentrant": False}` but `TrainingConfig` has no field for it. Same gap for `max_steps` (debug cap) and `log_completions` (always `True`).
- **Container image / `framework_version`** — `ContainerConfig.framework_version` maps to PyTorch SageMaker images via `image_uris.retrieve()`. Non-AWS backends need a different strategy (local: no image; Vertex: GCP images; custom: Docker URI). Treat as execution-backend-specific config.
- **NCCL env duplication** — `grpo_advice_dedicated_vllm.yaml` sets `NCCL_*` in the top-level `environment`, AND `entrypoint.sh` sets the same vars based on `infra.nccl_transport`. Both run, potentially conflicting. Section 5.6 proposes centralization; this is a concrete bug risk today.
- **Secret injection (`idps` | `local`)** — Secrets are resolved at worker runtime via `resolve_secrets.py` (IDPS via `idps-sdk`, or a dotenv-style `.env` file for `provider: local`). Non-secret Genos identifiers (`GENOS_ENV`, `GENOS_EXPERIENCE_ID`, etc.) live in the top-level `environment` section (backend-agnostic; passed directly to the SageMaker container). The target Genos stack is configured via `GENOS_ENV` and consumed by `foresightrlcore/clients/genos.py`.
- **`enable_thinking` is model-specific** — Threaded from `model.enable_thinking` to `task_config` to dataset loader for Qwen3 `<think>` tags. Future models may have different feature flags; the schema should allow model-specific feature config without hard-wiring per-model logic.
- **`reward_weights` and multi-reward composition** — `GrpoConfig.reward_weights` exists, and TRL accepts lists of reward functions. Section 4.7 should address composing multiple reward sources (e.g. built-in format check + external domain reward), not just single-function replacement.
- **Configs** — Only one YAML (`grpo_advice_dedicated_vllm.yaml`) is committed to `foresightrlcore/runtime/configs/`. `CLAUDE.md` references several variants not in version control. Canonical examples should be committed as the library stabilizes.
- **External reward modules (planned)** — Today, custom logic requires a **new task module** under `tasks/<name>.py` shipped with the repo/container, or forking `advice.py`. First-class support for **loading reward code from an arbitrary path** (or uploaded artifact) at launch is not implemented yet; Section 4.7 specifies the target contract.

### 2.3 Utilities outside the training core

- `preprocess_*.py`, `estimates/`, `run_*_inference.py` — Data and eval tooling; should remain **optional** dev/ops scripts or migrate to `package.devtools` extra, not core dependencies.

---

## 3. Reference designs (industry patterns)

These are not prescriptions but anchors for API shape and layering:

| Project / pattern | Takeaway for this library |
|-------------------|---------------------------|
| **Hugging Face TRL** | Single library exposes `SFTTrainer`, `DPOTrainer`, `GRPOTrainer`, etc., with shared conventions (`TrainingArguments`, model/tokenizer). *Methods* map naturally to thin adapters that build the right trainer + config. |
| **Hugging Face Accelerate / DeepSpeed** | Backend-agnostic launch; “how many processes” and “which launcher” belong in **runtime/local** backend config, not in algorithm YAML. |
| **Axolotl** | YAML-heavy, recipe-oriented; good inspiration for **one file per run** and optional plugin hooks—avoid copying their monolith structure. |
| **PyTorch Lightning** | Clear separation: `LightningModule` (analog: **method + model recipe**) vs `Trainer` vs cluster environment—useful mental model. |
| **Ray Train / Anyscale** | Provider abstraction: same training function, different “execution context”; good reference for **backend interface** design. |
| **SageMaker Python SDK** | Estimator / `ModelTrainer` / raw `CreateTrainingJob`—already partially abstracted in `launcher.py`; keep a **narrow AWS adapter** behind an interface. |
| **Entry points (importlib.metadata)** | Standard way to register third-party `method` and `provider` plugins without editing core code. |

**AWS note:** Product names evolve (Training Jobs, HyperPod, Processing, “serverless” inference vs training). The library should expose **`provider: aws` + `execution: { kind: ... }`** (or separate top-level keys) so new AWS execution modes plug in without renaming the whole package. Exact mapping to SageMaker APIs belongs in an `aws` extra and documentation, not in the core schema names.

---

## 4. Proposed conceptual architecture

### 4.1 Layered model

```
┌───────────────────────────────────────────────────────────────────────┐
│  CLI / Python API                                                      │
│  rl-train train --config job.yaml                                      │
│  rl-train eval  --config job.yaml   (future)                           │
│  rl-train test  --config job.yaml   (future)                           │
│  rl-train validate --config job.yaml                                   │
└───────────────────────────┬───────────────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────────────┐
│  Config layer (Pydantic v2)                                            │
│  - Single root model: JobConfig                                        │
│  - Discriminated unions: method.*, execution.*, data.*                 │
│  - Load: YAML + env overlay + CLI overrides                            │
└───────────────────────────┬───────────────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────────────┐
│  Operation dispatcher                                                  │
│  - "train" → Training pipeline (current focus)                         │
│  - "eval"  → Evaluation pipeline (future)                              │
│  - "test"  → Test/benchmark pipeline (future)                          │
│  - Custom operations via registry                                      │
└───────────────────────────┬───────────────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ Method         │ │ Execution      │ │ Task / Recipe  │
│ registry       │ │ registry       │ │ (reward, data) │
│ GRPO, SFT, DPO │ │ local, aws, …  │ │ plugin ID      │
└────────┬───────┘ └────────┬───────┘ └────────┬───────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Runner orchestration                                                  │
│  - Resolve operation → select pipeline                                 │
│  - Resolve method → build Trainer / evaluator / test harness           │
│  - Resolve execution → package code, env, entrypoint, submit           │
│  - Wire paths (local fs vs S3 vs channel mounts)                       │
└───────────────────────────────────────────────────────────────────────┘
```

**Principles:**
- *Operation* (what you're doing: train, perf_test, optimizer, …) is the top-level dispatch. Training was the first operation; `perf_test` (Gatling load testing) and `optimizer` (GenAI quality + cost evaluation) are now shipped alongside it, and the architecture does not assume training is the only one.
- *Method* (how you train: GRPO, SFT, DPO) and *execution* (where it runs: local, AWS, etc.) are **orthogonal** except for a small matrix of capabilities.
- Future operations (eval, test, deploy) follow the same pattern: config → operation dispatch → method/execution resolution → run.

### 4.2 Suggested package layout (rename in migration)

Illustrative package names—the final PyPI name is a product decision:

```
foresightrlcore/
  __init__.py                        # public API: load_config, train, validate
  __main__.py                        # python -m foresightrlcore
  cli.py                             # subcommands: train, eval, validate, ...

  config/
    schema.py                        # JobConfig (root), MethodConfig, ExecutionConfig
    profiles.py                      # infra_profile presets
    validation.py                    # capability matrix checks
    recipes.py                       # container + local-GPU recipe registries
    io.py                            # path resolution (S3, local, SM channels)

  operations/                        # top-level operation dispatch
    __init__.py                      # registry: get_operation, register_operation
    base.py                          # Operation protocol
    train.py                         # TrainOperation (primary)
    perf_test.py                     # PerfTestOperation (Gatling load testing)
    optimizer.py                     # OptimizerOperation (GenAI quality + cost eval)
    # eval.py                        # (future) EvalOperation
    # test.py                        # (future) TestOperation
    # deploy.py                      # (future) DeployOperation

  methods/
    __init__.py                      # registry: get_method, register_method
    base.py                          # MethodHandler protocol + TrlMethodHandler base class
    grpo.py                          # GrpoMethodHandler (subclass of TrlMethodHandler)
    sft.py
    dpo.py

  execution/
    __init__.py                      # registry + entry point discovery
    base.py                          # ExecutionBackend protocol
    aws_training_job.py              # AWS SageMaker Training Job (current)
    local.py                         # local GPU training (extracted from grpo_advice.py)
    # aws_serverless.py              # (future) AWS serverless/customizable
    # vertex.py                      # (future) GCP Vertex AI
    # modal.py                       # (future) Modal
    # runpod.py                      # (future) RunPod

  auth/                              # runtime secret resolution
    __init__.py                      # SecretsProvider protocol + registry
    idps.py                          # IDPS provider (idps-sdk)
    local.py                         # Local .env file provider

  tasks/                             # pluggable per-task components
    __init__.py
    _thinking.py
    advice.py
    matching.py

  runtime/                           # container-side code
    scripts/
      entrypoint.sh
      resolve_secrets.py             # IDPS secret resolution (runs before training)
      train.py                       # unified container driver
      nccl_env.py
      ds_zero2_config.json
      ds_zero3_config.json
      requirements.txt
```

**Operations as extension points:** Each operation (train, perf_test, optimizer; future eval, test, deploy) follows the same pattern: load config → resolve dependencies → run. The `operations/` module provides a registry so new operations register without modifying core dispatch.

**Legacy path:** The `foresightrlcore/runtime/scripts/` directory contains container-side code that runs inside SageMaker training containers.

### 4.3 Core abstractions (protocols)

**Operation** (top-level dispatch):

- `name: str` — e.g. `"train"`, `"perf_test"`, `"optimizer"`
- `validate(config)` — operation-specific config checks
- `run(config, execution_backend)` — execute the operation
- Training was the first operation; `perf_test` and `optimizer` ship today, and eval, test, deploy follow the same protocol.
- Registered via `operations/` registry; CLI subcommands map 1:1 to operations.

**MethodHandler** (protocol in `methods/base.py`) and **TrlMethodHandler** (concrete base):

- `MethodHandler` protocol: `validate()`, `build_trainer_args()`, `run()`.
- `TrlMethodHandler` base class provides the shared training pipeline: task loading, dataset handling, LoRA, vLLM patching, hooks resolution, trainer construction, train/save.
- Concrete handlers (e.g. `GrpoMethodHandler`) subclass `TrlMethodHandler` and add only method-specific validation and trainer args.
- `TRL_TRAINERS` registry maps method type strings to TRL trainer/config class dotted paths for lazy import.
- **Training hooks** (`training.hooks` in YAML): Method handlers resolve callbacks, pre/post hooks, and custom trainer classes from dotted import paths at training time. This allows users to customize TRL behavior (e.g. dynamic `max_completion_length`, iterative GRPO, backprop reduction) without modifying framework code. See `foresightrlcore/hooks.py` and `docs/tasks/training-hooks.md`.
- **Cross-validation**: `config/validation.py` ensures `method.type` matches the corresponding `training.<method_type>` hyperparameter block.

**ExecutionBackend** (pluggable per provider):

- `backend_type: str` — e.g. `"aws_training_job"`, `"local"`, `"vertex"`, `"modal"`
- `validate(cfg) -> list[str]` — provider-specific checks (e.g. S3 URIs for AWS, local paths for local, project ID for Vertex)
- `launch(job_config, wait) -> str` — submit remote job OR invoke local subprocess; returns a run identifier
- `get_status(run_id) -> str` — check job status (optional, returns `"unknown"` if unsupported)
- Registered via `execution/` registry or Python entry points for third-party backends
- The `execution.type` field in config selects which backend; provider-specific config lives under `execution.<type>.*` or existing top-level keys (e.g. `infra.*` for AWS)
- **Adding a new provider** requires: one Python file implementing the protocol, register it, document its config keys, add its dependency to an optional extra — no changes to core config, CLI, methods, or operations

**TaskPlugin** (retain current pattern):

- `get_reward_function`, `get_callbacks`, `get_dataset_loader` for GRPO-style tasks; SFT/DPO might use `get_dataset` only — methods declare which hooks they call.
- **Reward strategies** can be selected **inside** a task via config (e.g. `advice` + `reward_mode: llm_judge | mock`) or, in the future, by **loading an external module** that satisfies the same contract (Section 4.7).

**SecretsProvider** (implemented — `foresightrlcore/auth/`):

- `resolve(secret_path: str) -> str` — resolve a single secret path to its plaintext value.
- Providers are registered via `register_provider(name, cls)` and selected by `secrets.provider` in config (`idps` or `local` today). New providers (e.g. AWS Secrets Manager) implement the same protocol.
- **Built-in providers:** **`idps`** (uses `idps-sdk`) and **`local`** (reads `KEY=VALUE` lines from `secrets.local.env_file` for workstations without IDPS).
- Secrets are resolved **at worker runtime** via `resolve_secrets.py`, not at launch time. Only secret *paths* appear in config/S3/CloudTrail — never values (local runs read values from a user-supplied file path, not from SageMaker metadata).
- Config schema: `SecretsConfig` with `provider`, provider-specific connection config (`idps` or `local`), and `entries` list mapping secret paths to target env vars.

Use **typing.Protocol** or explicit ABCs; avoid heavy frameworks.

### 4.4 Configuration schema (YAML)

**Top-level sketch** (exact field names to be bikeshedded):

```yaml
version: 1

job:
  name: my-experiment
  tags: { team: nlp }

task:
  type: advice
  config:
    reward_mode: llm_judge   # llm_judge | mock — external modules: see §4.7
    # mock: mock_constant_reward, mock_sleep_seconds, predictions_log, ...

method:
  type: grpo   # sft | dpo | grpo | ...
  grpo:
    num_generations: 8
    # ...
  # sft: { ... }
  # dpo: { beta: 0.1, ... }

model:
  name: Qwen/Qwen3-14B
  lora: { r: 32, alpha: 64, ... }

training:
  epochs: 1
  per_device_batch_size: 4
  optimizer: ...
  # shared training hyperparams; method-specific pieces live under method.*

data:
  train_path: s3://bucket/train.jsonl   # or local path when execution allows
  eval_path: ...

output:
  uri: s3://bucket/out/                 # or local dir

execution:
  type: aws_training_job                # local | aws_training_job | aws_serverless | vertex | modal | ...

  # Provider-specific config lives under the matching key.
  # Only the active provider's block is validated.

  # --- AWS SageMaker Training Job (current) ---
  # Uses existing infra.* and container.* sections for backward compat
  aws:
    region: us-west-2
    role: arn:aws:iam::...
    # infra.instance_type, infra.topology, etc. migrated from current InfraConfig

  # --- Local GPU (replaces grpo_advice.py manual workflow) ---
  # local:
  #   gpus: [0, 1, 2, 3, 4, 5, 6, 7]
  #   vllm_mode: server          # server | colocate | none
  #   vllm_gpu_count: 2          # GPUs 0-1 for vLLM, 2-7 for training
  #   vllm_port: 8002
  #   deepspeed_config: ds_zero2_config.json

  # --- Modal (future) ---
  # modal:
  #   gpu: A100
  #   gpu_count: 4
  #   image: my-training-image:latest

  # --- Any future provider: add key here, implement ExecutionBackend ---

secrets:                              # (implemented) runtime secret resolution
  provider: idps                      # idps | local — SecretsProvider registry key
  idps:
    policy_id: "p-examplepolicy"
    environment: e2e                  # prd | e2e (or set endpoint directly)
  # local:                            # when provider: local
  #   env_file: "secrets/e2e.env"     # KEY=VALUE lines; entries map secret_path -> keys in file
  entries:
    - env_var: GENOS_APP_SECRET
      secret_path: "app_secret"
```

**Discriminated unions** in Pydantic keep invalid combinations from loading. Shared **`training`** block stays as narrow as possible; method-specific knobs live under **`method.<type>`**. **Task** plugins (`task.type` + `task.config`) hold reward mode, dataset hints, and future **`reward_spec`** for external code (§4.7).

### 4.5 Dependency groups (per-provider extras)

| Extra | Purpose | Example deps |
|-------|---------|----------------|
| *(none)* | Config validation, CLI “validate”, docs generation | `pydantic`, `pyyaml`, `typing_extensions` |
| **Execution backends (install per provider):** | | |
| `aws` | Submit SageMaker Training Jobs from laptop | `boto3`, `sagemaker` SDK |
| `vertex` | (future) GCP Vertex AI | `google-cloud-aiplatform` |
| `modal` | (future) Modal | `modal` |
| `runpod` | (future) RunPod | `runpod` |
| **Local GPU training:** | | |
| *(recipe-based)* | `rl-train setup-local-gpu --recipe <name>` generates a `local-requirements.txt` with `torch`, `trl`, `peft`, `accelerate`, `deepspeed`, optional `vllm` | Installed via `pip install -r local-requirements.txt` on the GPU machine |
| `train-cpu` | Smoke tests, imports without CUDA | `torch` CPU-only pin (optional) |
| **DX extras:** | | |
| `cli` | Rich output, better DX | `typer` or `click` |
| `dev` | Testing and linting | `pytest`, `ruff` |

**Container images** for cloud jobs use **container recipes** (`container.recipe` in YAML) to pin dependencies; local GPU machines use **local-GPU recipes** via `rl-train setup-local-gpu` — same schema, different environment.

### 4.6 Execution backend plugin model

The execution backend is one of the three core extension points (alongside methods and tasks). The design goal is: **changing where training runs should never require changing how training works.**

**How it works:**

1. **Config drives selection** — `execution.type` in the YAML determines which backend class handles the job.
2. **Registry resolves** — `get_backend("aws_training_job")` returns the backend class. Unknown types give a clear error listing available backends.
3. **Provider-specific config** — Each backend reads its own config keys (AWS reads `infra.*` + `container.*`; local reads `execution.local.*`; Modal would read `execution.modal.*`). The core config parser doesn't need to know about provider fields.
4. **Validation is backend-specific** — AWS checks S3 paths exist, instance types are valid, SDK versions match topology. Local checks GPU availability. Each backend's `validate()` returns errors before any resources are allocated.
5. **Third-party backends** — External packages can register backends via Python entry points without modifying the rl-train codebase:
   ```toml
   # Third-party pyproject.toml
   [project.entry-points."foresightrlcore.execution"]
   my_provider = "my_package:MyProviderBackend"
   ```

**Adding a new execution backend (recipe):**

| Step | What to do |
|------|-----------|
| 1 | Create a class implementing `ExecutionBackend` protocol (`validate`, `launch`, `get_status`) |
| 2 | Register it: built-in backends in `execution/__init__.py`; third-party via entry points |
| 3 | Document the provider-specific YAML keys (under `execution.<type>.*`) |
| 4 | Add the provider's SDK as an optional dependency extra in `pyproject.toml` |
| 5 | Add capability matrix entries for method/infra combos this provider supports |
| 6 | (Optional) Add a Cursor rule / Claude rule documenting provider-specific constraints |

**Example: what each backend is responsible for:**

| Backend | What it does |
|---------|-------------|
| `aws_training_job` | Packages code to S3, resolves SageMaker image, submits via SDK v2/v3/boto3, monitors job |
| `local` | **Already proven** — extracted from `grpo_advice.py`/`grpo_matching.py`. Sets `CUDA_VISIBLE_DEVICES`, optionally starts vLLM server on dedicated GPUs, launches `accelerate launch` with DeepSpeed, runs unified `train.py` |
| `aws_serverless` (future) | Submits to serverless/customizable training endpoint with different API |
| `vertex` (future) | Packages to GCS, submits Vertex AI CustomJob, monitors |
| `modal` (future) | Builds Modal image, deploys function with GPU, runs training |

### 4.7 CLI surface (operation-based)

The CLI maps subcommands to **operations**. Training was the first; `perf-test` and `optimizer` ship today, others plug in later.

```
rl-train train             --config job.yaml [--wait] [--method grpo] [overrides...]
rl-train validate          --config job.yaml
rl-train perf-test         --config perf.yaml [--wait]                # Gatling load test
rl-train validate-perf     --config perf.yaml
rl-train optimizer         --config opt.yaml [--wait]                 # GenAI quality + cost eval
rl-train validate-optimizer --config opt.yaml
rl-train setup-local-gpu   --recipe qwen3_5 [--output local-requirements.txt]
rl-train setup-local-gpu   --list
rl-train eval              --config job.yaml [--model-path s3://...]  # future
rl-train test              --config job.yaml [--benchmark mmlu]       # future
```

Each subcommand resolves config → operation → execution backend → run. The `validate` subcommand is built-in (no execution backend needed).

Python API:

```python
from foresightrlcore import load_config, train, validate

config = load_config("job.yaml", overrides={"model.name": "Qwen/Qwen3-8B"})
errors = validate(config)
job_id = train(config, wait=True)
```

Future operations expose the same pattern: `from foresightrlcore import evaluate`, `from foresightrlcore import test_model`.

### 4.8 Custom reward functions and user-supplied code (target design)

**Today (in-repo tasks):** Rewards for GRPO come from the task plugin loaded by `task.type` (e.g. `tasks.advice`). The `advice` task switches behavior with **`reward_mode`**:

| `reward_mode` | Behavior |
|---------------|----------|
| `llm_judge` | LLM critic, credentials, thinking penalties, criterion callbacks. |
| `mock` | Deterministic rewards, no judge, no callbacks that depend on critic metrics. |

**Goal:** Let integrators supply **their own reward implementation** without committing it into this repository: configure **where the code lives** and **how to load it** when the training worker starts (local or container).

**Recommended loading model (two tiers):**

1. **Tier A — Full task module (preferred for non-trivial work)**  
   Same contract as built-in `tasks/*.py`: Python module(s) providing:
   - `get_reward_function(task_config: dict) -> Callable`
   - `get_callbacks(task_config) -> list` (optional; default `[]`)
   - `get_dataset_loader(task_config, model_name) -> Callable` (optional if dataset stays in base task)

   Config sketch:

   ```yaml
   task:
     type: external
     config:
       reward_spec:
         loader: import_path          # e.g. my_package.rewards.advice_custom
         # OR
         # loader: file
         # path: /opt/ml/input/data/reward/reward_plugin.py  # SageMaker channel or synced path
   ```

   Implementation inserts the directory into `sys.path` (for `file`) or imports by dotted path (`import_path`), then calls the same hooks as `get_task_components` today.

2. **Tier B — Reward-only override**  
   A single callable registered by config for tasks that already define dataset loading in-repo (e.g. keep `task.type: advice` but replace only the reward):

   ```yaml
   task:
     type: advice
     config:
       reward_mode: external
       reward_spec:
         module: /opt/ml/input/data/reward/custom_reward.py
         factory: build_reward   # callable name; returns (completions, prompts, ...) -> list[float]
   ```

   The **exact signature** must match what the task/TRL integration expects (for current `advice`, align with the closure built in `advice.py`—batch of completions and prompt/state fields). Document the signature in the library and validate at load time where possible.

**SageMaker / remote execution:** User code must arrive on the worker **before** `train_grpo.py` imports the module. Practical options:

- **Extra input channel** (e.g. `reward` → `/opt/ml/input/data/reward/`) containing a `.py` file or small wheel.
- **S3 URI** copied in `entrypoint.sh` before training starts.
- **Bundled in `source_dir`** when launching the job (simplest for teams that own the launcher repo).

**Security:** Loading arbitrary Python is equivalent to arbitrary code execution. Mitigations: document that only **trusted** paths/channels are allowed; optional allowlist of module prefixes; run in isolated containers; avoid `eval` of YAML strings.

**Library responsibilities:**

- Resolve `reward_spec` → imported module or callable.
- Clear errors when exports are missing or signatures do not match.
- Unit tests with a **temporary module file** on disk (same pattern as `test_advice_reward_mode.py`).

**Multi-reward composition:** TRL's `GRPOTrainer` accepts a **list** of reward functions with optional `reward_weights`. The external reward design should support registering **multiple sources** (e.g. a built-in format checker + an external domain reward), and the YAML should surface `reward_weights` at the task level to control blending. This also means the `reward_spec` could be a list of specs, not only a single entry.

**Migration note:** Refactor `advice.get_reward_function` dispatch (`llm_judge` / `mock`) into a small registry so adding `external` is a new branch, not a copy-paste of import logic in multiple places.

---

## 5. Infrastructure profiles, NCCL transport, and method × infra compatibility

This section reviews **today’s** SageMaker implementation (`InfraConfig` in `foresightrlcore/runtime/config.py`, `GRPOJobLauncher`, `scripts/entrypoint.sh`), proposes a **clearer aggregated model** for configuration, and defines how the **future library** should validate **which training methods can run on which infrastructure and providers**.

### 5.1 Two orthogonal axes (mental model)

Operations today mix two independent concerns. Treating them separately avoids combinatorial confusion and keeps YAML extensible:

| Axis | Purpose | Today’s primary knobs |
|------|---------|------------------------|
| **A — Generation / rollout backend** | How completions are produced for RL (or ignored for SFT). | `infra.vllm_mode`: `server` \| `colocate` \| `none` |
| **B — Placement topology** | How GPUs and nodes are allocated (split on one machine, dedicated nodes, heterogeneous fleet). | `infra.topology`: `single_node` \| `multi_node` \| `multi_node_server` \| `dedicated_vllm` |

**Entrypoint dispatch** (`entrypoint.sh`): `vllm_mode` is evaluated **first**. If `none`, the script runs **`run_torchrun`** and **does not branch on `topology`** (multi-node torchrun is still supported via `SM_HOSTS`). If `colocate`, **`run_colocate`** runs (multi-node aware). Otherwise (`server`), **`topology`** selects among `run_single_node`, `run_multi_node`, `run_multi_node_server`, and `run_dedicated_vllm`.

**Launcher** (`launcher.py`): `vllm_mode == none` switches to **`train_grpo.py`** as the SageMaker entry point with `torch_distributed` / `Torchrun`, and swaps in **`requirements_torchrun.txt`** (no vLLM). All other modes use **`entrypoint.sh`**.

### 5.2 Mode catalog (current behavior)

| `vllm_mode` | `topology` (if applicable) | What runs | Notes |
|-------------|----------------------------|-----------|--------|
| `none` | *(ignored for outer switch)* | `torchrun` → `train_grpo.py`, TRL `use_vllm=False` | All GPUs for training; multi-node NCCL only between training ranks. **Lightest** container deps. |
| `colocate` | *(N/A)* | `accelerate launch` → `train_grpo.py`, vLLM inside trainer | Multi-node: EFA/socket NCCL for **training** ranks; vLLM colocated per rank/TP. |
| `server` | `single_node` | vLLM on `vllm_gpu_count` GPUs, training on `training_gpu_count` | Classic split on one instance (`CUDA_VISIBLE_DEVICES` ranges). |
| `server` | `multi_node` | **Heterogeneous** SageMaker instance groups: `vllm` nodes vs `training` nodes | Resolved via `SM_CURRENT_INSTANCE_GROUP` + vLLM host discovery. |
| `server` | `multi_node_server` | **Homogeneous** cluster: node 0 runs vLLM (split) + training; other nodes training only, HTTP to vLLM on node 0 | Avoids colocate EFA issues; vLLM is separate process. |
| `server` | `dedicated_vllm` | Node 0 **all GPUs → vLLM**; node 1+ **all GPUs → training** | Training–vLLM traffic includes **cross-node NCCL** (weight sync per TRL comments). TCP sentinel coordinates shutdown. Requires `instance_count >= 2`. |

**SDK caveat:** **Heterogeneous `multi_node`** (separate `InstanceGroup`s) is built in **`_build_multi_node_estimator`** on the **SDK v2** path only. The **v3 `ModelTrainer`** path today uses a single `Compute` (`instance_type` × `instance_count`). Jobs that need heterogeneous groups should use SDK v2 or boto3 until v3 gains parity—this should be encoded in **compatibility validation** (below).

### 5.3 NCCL transport: `efa` vs `sockets`

`infra.nccl_transport` (`efa` default, or `sockets`) controls **libfabric / NCCL** behavior on **multi-node** paths where the shell exports environment variables:

- **Applied in:** `run_colocate` (when `host_count > 1`), `run_dedicated_vllm`, `run_multi_node_server`, `run_torchrun` (when `host_count > 1`). **Not** set in **`run_single_node`** (single-machine training does not need cross-node EFA/socket tuning).
- **`efa`:** Sets `FI_PROVIDER=efa`, RDMA-related flags, `NCCL_PROTO` (often `simple`), long timeouts—appropriate for **p4d/p5**-class instances with **EFA** and matching **security group** rules.
- **`sockets`:** Sets `NCCL_NET=Socket`, clears plugin, sets `NCCL_SOCKET_IFNAME`—**TCP fallback** when EFA handshakes fail (e.g. error 4126) or VPC constraints; slower but more forgiving.

**Future-proofing:** Prefer a small **nested model** instead of a free-form string, e.g. `infra.network_transport: { kind: aws_efa | tcp_socket, socket_interface: ... }`, with the entrypoint mapping to the same env vars—so new options (e.g. **IB**, **custom NCCL env**) are **data-driven** rather than new bash branches everywhere.

### 5.4 Aggregating configuration (target schema)

Replace implicit coupling with an explicit **profile** or **structured union** (names illustrative):

```yaml
infra:
  # Option A — keep backward compatible flat keys (today)
  topology: single_node
  vllm_mode: server
  nccl_transport: efa

  # Option B — future: named profile presets + overrides
  # profile: sm_grpo_p4d_split_vllm   # expands to topology + vllm + network defaults
  # overrides:
  #   vllm_gpu_count: 2
```

**Profile presets** (e.g. `sm_grpo_single_split`, `sm_grpo_dedicated_vllm`, `sm_grpo_torchrun`) bundle sane defaults for **topology + vllm_mode + network + entrypoint** so users add **only deltas**. New AWS instance families or topologies become **new preset IDs**, not new required YAML fields for everyone.

### 5.5 Capability matrix: methods × infrastructure × provider

Not every **training method** is meaningful on every **infra profile** or **cloud provider**. The library should enforce this **at config validation time** (and optionally in the CLI before submit).

**Dimensions:**

1. **Method** — `grpo`, `sft`, `dpo`, … (each declares *requirements*: e.g. GRPO needs batched generation; SFT may not need vLLM at all.)
2. **Infra profile** — Abstract label resolved to `{ topology, vllm_mode, launcher entrypoint, network }` as above.
3. **Provider** — `aws_sagemaker_training_job`, `local_process`, `gcp_vertex`, …

**Implementation options** (pick one or combine):

| Mechanism | Role |
|-----------|------|
| **Static compatibility table** | Versioned JSON/YAML in the package: `(method, provider, infra_profile) → { allowed: bool, reason?: str }`. Easy to test; no imports. |
| **Pydantic `model_validator`** | After parsing `TrainingJobConfig`, call `assert_compatible(method, execution, infra_profile)`. Central place for rules. |
| **Protocol hooks** | Each `MethodHandler` implements `supported_execution_tags() -> frozenset[str]`; each `ExecutionBackend` advertises `tags`; intersection must be non-empty. **Extensible** when third parties add methods. |
| **Entry-point metadata** | `importlib.metadata` entry points declare capabilities for plugins. |

**Example rules (illustrative):**

- **GRPO + `vllm_mode: none`** — Allowed (HF generate); may be slow for large models—**warning**, not error.
- **GRPO + `topology: dedicated_vllm` + provider that cannot do multi-instance** — **Error** at validate.
- **SFT** — Typically **any** topology that can run `Trainer`; **no** requirement for vLLM—mark `vllm_mode: none` as **recommended default** for simplicity.

**User-facing errors:** When validation fails, return a **clear message**: e.g. “Method `grpo` with `topology: multi_node` requires SageMaker SDK v2 heterogeneous `InstanceGroup` support; use SDK v2 launcher or switch to `multi_node_server` on homogeneous clusters.”

### 5.6 Evolution checklist

- [ ] Document the **v2 vs v3** heterogeneous-cluster limitation next to `multi_node` in operator docs.
- [ ] Introduce **`infra_profile`** presets while keeping flat `topology` / `vllm_mode` for backward compatibility.
- [ ] Centralize **NCCL env** mapping in one module (Python generates a small `env.sh` or the launcher sets env vars via the top-level `environment` section) to reduce drift between `entrypoint.sh` branches.
- [ ] Add **`validate_method_infra_provider(config)`** with the capability matrix and unit tests for golden configs.

---

## 6. AWS-specific design notes

- **Training Jobs (current):** Keep `GRPOJobLauncher` logic as `AwsTrainingJobBackend`; generalize naming from GRPO to “training job with configurable entrypoint and hyperparameters.”
- **New AWS modes:** Implement as separate `ExecutionBackend` classes sharing S3/session helpers; avoid a god-class `if serverless: ... elif training_job: ...` beyond a small factory.
- **Training Plan ARN / boto3 fallback:** Preserve existing behavior; document in AWS adapter README.
- **Entry point / script packaging:** `entrypoint.sh` + `train_*.py` remain artifacts **versioned with the library** and uploaded as `source_dir` or container `WORKDIR` content—same as today.

---

## 7. Migration plan (phased)

> **Note:** The detailed implementation plan referenced here (`docs/implementation-plan.md`) was a planning artifact and was not carried forward. The phases below summarize the migration that has since been completed.

**Phase summary** (Phases 1–4 can be developed in parallel):

| Phase | Goal | Key deliverables |
|-------|------|------------------|
| **0 — Foundation** | `pyproject.toml`, new `foresightrlcore/` package skeleton, tests, TRL pin | Installable package, example configs, test infrastructure |
| **1 — Config** | Move + enhance config schema into `foresightrlcore/config/` | `TrainingJobConfig`, infra profiles, capability matrix, typed TaskConfig |
| **2 — Tasks** | Move tasks, extract shared utils, add matching | `foresightrlcore/tasks/` with advice, matching, thinking utils, external loader |
| **3 — Methods** | Extract training logic into pluggable handlers | `foresightrlcore/methods/` with GRPO, SFT, DPO handlers |
| **4 — Execution** | Refactor launcher into execution backend | `foresightrlcore/execution/` with `AwsTrainingJobBackend` |
| **5 — Training driver** | Unified `train.py` replacing `train_grpo.py` | Method-dispatching driver, NCCL centralization. **Requires GPU validation.** |
| **6 — CLI + API** | Clean CLI with subcommands, Python API | `rl-train launch/validate`, `from foresightrlcore import load_config, launch` |
| **7 — External rewards** | User-supplied reward code loading | `task.type: external`, Tier A/B, `reward_mode: external` |
| **8 — Polish** | Cleanup and documentation | Remove `train_grpo.py`, update all docs |

**What "don't break" means:** We freely rename files, move code, and restructure modules. The constraint is preserving **validated runtime logic** — the GRPO trainer wiring, the 6 entrypoint dispatch paths, the reward functions, and the SageMaker submission flow must produce identical behavior after refactoring. Existing YAML keys are never renamed (new keys with defaults are fine). `foresightrlcore/runtime/` becomes a backward-compat shim with deprecation warnings.

**Config philosophy:** YAML config files are user-owned, never committed to the repo. The library ships example templates under `examples/configs/` for reference. Users place their configs anywhere and pass the path at runtime.

---

## 8. Testing & quality (no GPU on dev laptops)

- **Unit:** Pydantic schema validation, merge overrides, path resolution, adapter `validate()` without AWS calls (mocks).
- **Reward logic:** Tests for `reward_mode` branches (`mock` vs `llm_judge` callbacks), and future tests that load a **temporary reward module** from disk to verify the external-loader contract.
- **Contract tests:** Each `MethodHandler` returns expected trainer *types* with mocked models (small `hf-internal-testing/tiny-random-*`).
- **Integration:** Optional CI job with CPU-only torch for import smoke tests behind `train-cpu` extra.
- **Golden YAMLs:** Versioned example configs that must parse across releases.
- **Capability matrix:** Unit tests that **forbidden** `(method, provider, topology, vllm_mode)` tuples fail validation with expected error substrings; **allowed** golden paths pass.

---

## 9. AI-assisted development: rules and skills

The repository includes structured guidance for AI coding assistants (Cursor rules and Claude Code's `CLAUDE.md`). These are **living artifacts** that evolve with the library and encode the core workflows, contracts, and constraints so that any AI session starts with the right context.

### 9.1 Cursor rules (`.cursor/rules/`)

Rules are `.mdc` files with YAML frontmatter that apply automatically or when matching files are open.

| Rule file | Scope | Purpose |
|-----------|-------|---------|
| `remote-execution-workflow.mdc` | Always apply | No GPU locally; debug via logs/traces, validate configs statically, each round-trip is expensive. |
| `config-first-workflow.mdc` | `foresightrlcore/runtime/**/*.{py,yaml,yml}` | YAML is source of truth; never hardcode params; validate before submit; how to add new config fields. |
| `task-plugin-contract.mdc` | `foresightrlcore/tasks/**/*.py` | Required exports (`get_reward_function`, `get_callbacks`, `get_dataset_loader`), reward mode pattern, how to register new tasks. |

**When to update rules:**
- Adding a new training method → update `config-first-workflow.mdc` with new schema sections.
- Changing the task plugin contract → update `task-plugin-contract.mdc`.
- Adding a new execution backend → consider a new rule for that provider's constraints.
- Changing the debugging workflow → update `remote-execution-workflow.mdc`.

### 9.2 Claude Code guidance (`CLAUDE.md`)

The root `CLAUDE.md` mirrors the Cursor rules for Claude Code sessions. It covers:
- Development environment constraint (no GPU)
- Setup and dependencies
- Full architecture walkthrough (config pipeline, SageMaker flow, infra modes, reward modes)
- Key design patterns (config-first, task modularity, validate-before-submit)
- Pointer to this design document for the library roadmap

**Keep `CLAUDE.md` in sync** with rules when the architecture changes significantly.

### 9.3 Skills

Skills are structured workflows that guide AI assistants through multi-step tasks. Stored in both `~/.cursor/skills/` (Cursor) and `~/.claude/skills/` (Claude Code).

| Skill | Location | Purpose |
|-------|----------|---------|
| `launch-training-job` | Both Cursor and Claude Code | End-to-end workflow: determine backend (SageMaker / serverless / local) → pick/create config → validate locally → launch with CLI or Python API. Covers all three execution backends, references the dedicated docs for each. |

**When to update skills:**
- Adding new CLI flags → add to the flags table in the skill.
- Adding new topologies or vLLM modes → add rows to the infrastructure mode table.
- Adding new reward modes → add to the reward mode table.
- Adding new training methods → create a new skill (e.g. `launch-sft-training-job`) or generalize to `launch-training-job`.
- Changing the config schema → update the config structure reference in the skill.

### 9.4 Future rules and skills (as the library evolves)

| Trigger | New rule or skill |
|---------|-------------------|
| `pyproject.toml` + extras added | Rule: `dependency-management.mdc` — which extras to install when, how to add new deps. |
| Multi-method support | Rule: `method-handler-contract.mdc` — protocol for `MethodHandler`, how to register new methods. |
| Multi-provider support | Rule per provider (e.g. `aws-execution.mdc`, `vertex-execution.mdc`) — provider-specific constraints and validation. |
| External reward loading | Rule: `external-reward-loading.mdc` — security model, file paths, channel setup, testing. |
| CLI generalization | Skill: `launch-training-job` — generalized skill covering all methods, not just GRPO. |
| Capability matrix | Rule: `infra-compatibility.mdc` — which method × topology × provider combos are valid. |
| New operations (eval, test, deploy) | Rule per operation (e.g. `eval-operation.mdc`) with operation-specific constraints. Skill per operation for end-to-end workflow. |

---

## 10. Risks and open decisions

| Risk | Mitigation |
|------|------------|
| Schema explosion | Version field (`version: 1`) + migration helpers; keep unions strict. |
| TRL API churn | Pin TRL per major library version; abstract trainer construction behind method handlers. |
| Duplication vs TRL | Do not reimplement losses; library is glue + execution + config. |
| Third-party providers | Start with protocols + one reference (`aws`); add Vertex/Modal when needed. |
| Naming | Pick PyPI name early; `sagemaker_training` is misleading for a neutral lib. |
| Arbitrary reward code | Treat as trusted code; document security model; optional signing or internal-only channels. |
| Heterogeneous SageMaker + SDK v3 | Document and validate: `multi_node` topology may require v2/boto3 until v3 supports equivalent `InstanceGroup` APIs. |
| TRL version split (0.28 vs 0.29) | Pin one version; test both requirements files in CI before release. |
| Silent task config typos | Opaque `Dict[str, Any]` in `TaskConfig.config`; typed sub-models or unknown-key warnings needed. |
| NCCL env var conflicts | Top-level `environment` and `entrypoint.sh` can set the same vars; centralize source of truth. |
| Hardcoded GPU counts | `_gpus_per_node` only knows g5.12 / g6e.12 / default 8; new instance types silently get wrong value. |

**Open decisions for stakeholders:**

- Final **package name** and namespace.
- Whether **tasks** remain first-class for all methods or only RL-ish methods.
- **Operations beyond training**: Which operations (eval, test, deploy) to implement first, and whether they share the same config schema or have operation-specific top-level sections.
- Minimum Python version (3.10+ recommended for union syntax and Pydantic v2).
- Single **lockfile** strategy for SageMaker containers vs extras.
- **Default mechanism for user reward code** on AWS: extra SageMaker input channel vs S3 copy vs baked into `source_dir`.
- **Canonical list of `infra_profile` preset IDs** and whether presets replace or augment flat `topology` / `vllm_mode` keys.
- **DeepSpeed config management**: bundle ZeRO-2/3 defaults in the package, allow inline YAML, or accept user-provided paths?
- **Container image strategy for non-AWS backends**: `framework_version` is AWS-specific; Vertex/local need different image resolution.
- ~~**Secret injection**: keep `GENOS_*`-specific pattern or generalize?~~ **Resolved:** Implemented `secrets:` config section with `SecretsProvider` protocol and providers `idps` | `local`. Secrets resolved at worker runtime via `resolve_secrets.py`. No secret values in config/S3/CloudTrail. Genos LLM client logic extracted to `foresightrlcore/clients/genos.py`; non-secret `GENOS_*` identifiers live in top-level `environment`; target stack configurable via `GENOS_ENV` (defaults to `e2e`).
- **Model-specific features**: `enable_thinking` is Qwen3-specific; how to handle future model feature flags generically?

---

## 11. Summary

The existing codebase already demonstrates the right ideas: **Pydantic YAML config**, **SageMaker launcher abstraction**, **task plugins**, and a **container entrypoint** that separates orchestration from training. **Infrastructure** is controlled by **`vllm_mode` × `topology`** plus **`nccl_transport`** for multi-node NCCL (§5). **`reward_mode`** on the `advice` task shows how to swap **LLM judge** vs **credential-free mock** rewards without changing the GRPO driver. **Secrets management** is implemented via the `SecretsProvider` protocol (`foresightrlcore/auth/`) with built-in providers **`idps`** and **`local`** — secrets are resolved at worker runtime, keeping values out of config files, S3, and CloudTrail for cloud jobs (§4.3). Non-secret Genos env vars use the top-level **`environment`** section. The next steps for customization are **first-class loading of user-defined reward modules** (§4.7), **named infra presets** and a **method × infra × provider capability matrix** validated at config time (§5.5). **AI-assisted development rules and skills** (§9) encode the core workflows, contracts, and constraints for Cursor and Claude Code so that every AI session starts with the right context — and should evolve alongside the codebase. The architecture introduces an **operations layer** (§4.1–4.3) so that training is the first operation but not the only one — evaluation, testing, and deployment can be added as new operations following the same protocol, with the same config-first approach and execution backend abstraction. The path to a full library is to **generalize method and execution dimensions**, **split dependencies into extras**, **centralize configuration** (retire script globals), and **formalize extension points** so new algorithms, **reward implementations**, **operations**, **infra profiles**, and cloud providers ship without touching the core loop.

This document should evolve with ADRs (Architecture Decision Records) for major schema changes and provider additions.
