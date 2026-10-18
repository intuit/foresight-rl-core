# rl-train: Executive Summary

**A unified, config-driven library for reinforcement learning fine-tuning — one YAML file to train any model, with any method, on any infrastructure.**

---

## The Problem

RL fine-tuning today is fragmented. Teams maintain separate scripts for each training method (GRPO, DPO), each cloud provider (SageMaker, local GPU clusters), and each infrastructure topology (single-node, multi-node, colocated vLLM). Every new experiment means copy-pasting scripts, hard-coding hyperparameters, and debugging deployment-specific edge cases.

The result: brittle pipelines, siloed knowledge, and weeks of engineering effort just to try a new approach.

## The Solution

**`rl-train`** replaces this fragmentation with a single, extensible framework. One YAML config file describes the *what* (model, method, task) and the *where* (infrastructure, execution backend). The framework handles everything else.

```yaml
# This is all you need
job:
  name: my-training-run
model:
  name: Qwen/Qwen3-0.6B
method:
  type: grpo                    # swap to dpo, rlvr, rlaif — same interface
execution:
  type: aws_training_job        # swap to aws_serverless or local — same config
task:
  type: advice                  # plug in any reward task
data:
  train_path: s3://my-bucket/dataset/
output:
  s3_path: s3://my-bucket/output/
```

```bash
rl-train validate --config job.yaml   # catch errors locally
rl-train train --config job.yaml      # launch
```

Or from Python:

```python
from foresightrlcore import train
job_id = train("job.yaml", model_name="Qwen/Qwen3-8B")  # override on the fly
```

---

## Architecture

The framework is built on four pluggable layers. Each layer defines a protocol (interface), and new implementations are registered with a single function call. No forks, no subclasses, no framework lock-in.

```
┌─────────────────────────────────────────────────────────┐
│                      YAML Config                        │
│            TrainingJobConfig (Pydantic v2)               │
│     Validated, typed, with infra profiles & overrides    │
└──────────────────────┬──────────────────────────────────┘
                       │
              ┌────────▼────────┐
              │   Operations    │   What action to take
              │ train/optimizer │   (extensible registry)
              └────────┬────────┘
                       │
          ┌────────────┼────────────┐
          ▼                         ▼
┌─────────────────┐      ┌──────────────────┐
│    Methods      │      │    Execution     │
│  grpo / sft /   │      │    Backends      │
│  dpo            │      │  aws / local     │
│ (how to train)  │      │ (where to run)   │
└─────────────────┘      └──────────────────┘
          │                         │
          └────────────┬────────────┘
                       ▼
              ┌────────────────┐
              │     Tasks      │   Domain-specific logic
              │  reward fns,   │   (fully pluggable)
              │  datasets,     │
              │  callbacks     │
              └────────────────┘
```

### The Four Layers

| Layer | Purpose | Protocol | Implemented | Planned |
|-------|---------|----------|-------------|---------|
| **Operations** | Top-level actions | `Operation` | `train`, `perf_test`, `optimizer` | `eval`, `test`, `deploy` |
| **Methods** | Training algorithms | `MethodHandler` | `grpo`, `dpo`, `rlvr`, `rlaif` | `sft`, `kto` |
| **Execution** | Where jobs run | `ExecutionBackend` | `aws_training_job`, `aws_serverless`, `local` | `k8s` |
| **Tasks** | Reward & data logic | `TaskModule` | `advice` (LLM judge + mock) | Any custom domain |

Cross-cutting: **`foresightrlcore/auth/`** provides runtime secret resolution via the `SecretsProvider` protocol. Built-in providers are **`idps`** and **`local`** (dotenv-style `.env` file for local GPU when IDPS is unavailable). Secrets are fetched at worker startup — the launch path never embeds secret values in SageMaker metadata.

---

## Key Design Principles

### 1. Config-First

Every parameter — hyperparameters, infrastructure, model, method — lives in YAML. Nothing is hardcoded. Configs are validated with Pydantic before anything launches, catching errors locally instead of 30 minutes into a cloud job.

### 2. Pluggable by Design

Each layer uses Python's `Protocol` typing. Adding a new training method, execution backend, or reward task requires implementing a simple interface and registering it — no changes to the core framework.

**Add a new training method:**
```python
class SftMethodHandler:
    method_type = "sft"
    def validate(self, config): ...
    def build_trainer_args(self, config): ...
    def run(self, config): ...

register_method("sft", SftMethodHandler)
```

**Add a new execution backend:**
```python
class K8sBackend:
    backend_type = "k8s"
    def validate(self, config): ...
    def launch(self, config, wait=False): ...
    def get_status(self, job_id): ...

register_backend("k8s", K8sBackend)
```

**Add a new task (reward function + dataset):**
```python
# foresightrlcore/tasks/code_review.py
def get_reward_function(task_config):
    """Return a callable that scores model outputs."""
    ...

def get_callbacks(task_config):
    """Return trainer callbacks (or [])."""
    ...

def get_dataset_loader(task_config, model_name):
    """Return a callable that loads the training dataset."""
    ...
```

Then reference it in config: `task.type: code_review`. That's it.

### 3. Infrastructure Profiles

Complex GPU topologies are abstracted into named profiles. Users don't need to understand NCCL settings or vLLM placement — they pick a profile.

| Profile | What It Does | When to Use |
|---------|-------------|-------------|
| `single_split_vllm` | Single node, GPUs split between vLLM + training | Default — most experiments |
| `torchrun` | Pure torchrun, no vLLM | Small models, fast iteration |
| `colocate` | vLLM inside the trainer process | Memory-constrained setups |
| `dedicated_vllm` | Node 0 = vLLM server, remaining nodes = training | Large-scale production runs |
| `multi_node_server` | Homogeneous multi-node with vLLM split | High-throughput training |

```yaml
infra:
  infra_profile: dedicated_vllm   # one line replaces 10+ settings
  instance_type: ml.p4d.24xlarge
  instance_count: 3
```

### 4. Secrets Stay Secret

Credentials are resolved **at worker runtime**, not at launch time. With `provider: idps`, the YAML config only stores secret *paths* — never values — and nothing sensitive appears in SageMaker job metadata, S3 config uploads, or CloudTrail logs. With `provider: local`, the config references a `.env` file path on the machine running the job (typical for local GPU).

```yaml
secrets:
  provider: idps
  idps:
    policy_id: "p-examplepolicy"
    environment: e2e
  entries:
    - env_var: GENOS_APP_SECRET
      secret_path: "app_secret"
```

At worker startup, `resolve_secrets.py` resolves each configured secret (`idps` via `idps-sdk`, or `local` from a `.env` file) and exports them as environment variables before training begins. Non-secret Genos settings (experience ID, asset alias, offline job ID, and target environment) are set as plain env vars in the top-level `environment` section (backend-agnostic; for SageMaker the launcher passes `config.environment` directly as the container environment):

```yaml
environment:
  GENOS_ENV: "e2e"                # Genos environment: e2e | prf | prd
  GENOS_EXPERIENCE_ID: "182c28b1-..."
  GENOS_INTUIT_ORIGINATING_ASSETALIAS: "Example.team.service.myapp"
  GENOS_OFFLINE_JOB_ID: "<offline-job-id>"
```

The `GENOS_ENV` variable selects which Genos stack is targeted (identity auth and LLM Execution API). The `SecretsProvider` protocol is pluggable — **`idps`** and **`local`** are built-in today; additional providers (e.g. AWS Secrets Manager) follow the same pattern as methods and backends.

**Local provider example** (same `entries` list; values read from the env file):

```yaml
secrets:
  provider: local
  local:
    env_file: "secrets/e2e.env"
  entries:
    - env_var: GENOS_APP_SECRET
      secret_path: "app_secret"
```

### 5. Validate Before You Spend

The `validate` command checks config correctness, infra compatibility, and backend requirements — all locally, before any cloud resources are provisioned.

```bash
$ rl-train validate --config job.yaml
Config is valid: job.yaml
  Job:       grpo-advice-run
  Model:     Qwen/Qwen3-0.6B
  Method:    grpo
  Task:      advice
  Execution: aws_training_job
```

---

## Package Structure

```
foresightrlcore/
├── cli.py                 # rl-train CLI (validate, train, setup-local-gpu)
├── __init__.py            # train() Python API
├── config/
│   ├── schema.py          # TrainingJobConfig — full Pydantic schema
│   ├── validation.py      # Cross-field validation rules
│   ├── profiles.py        # Infrastructure presets + GPU catalog
│   └── recipes.py         # Container + local-GPU recipe registries
├── operations/
│   ├── base.py            # Operation protocol
│   ├── train.py           # TrainOperation — orchestrates launch
│   ├── perf_test.py       # PerfTestOperation — Gatling load testing
│   └── optimizer.py       # OptimizerOperation — GenAI quality + cost eval
├── methods/
│   ├── base.py            # MethodHandler protocol
│   └── grpo.py            # GRPO via TRL's GRPOTrainer
├── execution/
│   ├── base.py            # ExecutionBackend protocol
│   ├── aws_training_job.py # SageMaker Training Jobs
│   ├── aws_sagemaker_launcher.py  # SDK v2/v3 + boto3 launcher
│   ├── aws_serverless.py  # SageMaker Serverless Model Customization (SDK v3)
│   └── local.py           # Local GPU via accelerate/torchrun
├── auth/
│   ├── __init__.py        # SecretsProvider protocol + registry
│   ├── idps.py            # IDPS provider (idps-sdk)
│   └── local.py           # Local .env file provider
├── clients/
│   └── genos.py           # Genos LLM client (identity auth + LLM Execution API)
├── tasks/
│   ├── __init__.py        # Dynamic task loader (built-in, external, file path)
│   └── thinking_utils.py  # Shared utilities for reasoning models
└── runtime/scripts/       # Container-side code (shipped to SageMaker)
    ├── entrypoint.sh      # GPU topology orchestration
    ├── resolve_secrets.py  # IDPS secret resolution (runs before training)
    ├── train.py            # Training driver inside containers
    └── tasks/advice.py    # Advice task (LLM judge + mock rewards)
```

---

## The Launch Flow

Every training job follows the same path, regardless of method or backend:

```
User writes YAML config
        │
        ▼
rl-train train --config job.yaml
        │
        ▼
TrainingJobConfig.from_yaml()     ← Pydantic validation + profile resolution
        │
        ▼
validate_config()                 ← Cross-field checks (infra ↔ method ↔ backend)
        │
        ▼
get_backend("aws_training_job")   ← Registry lookup
        │
        ▼
backend.validate(config)          ← Backend-specific checks
        │
        ▼
backend.launch(config)            ← Job submitted (secret paths, not values, in config)
        │
        ▼
Job ID returned                   ← "grpo-advice-qwen3-20260414..."
```

Inside the container, before training starts:

```
entrypoint.sh
        │
        ▼
resolve_secrets.py                ← Resolves secrets (IDPS or local .env), exports as env vars
        │
        ▼
train.py                          ← Training driver (secrets available via os.environ)
```

---

## Ease of Use

| Persona | What They Do | How rl-train Helps |
|---------|-------------|-------------------|
| **ML Researcher** | Experiment with models & hyperparameters | Edit one YAML file, run one command. No infrastructure knowledge needed. |
| **ML Engineer** | Scale training to production | Switch `infra_profile`, bump `instance_count`. Same config, same command. |
| **Platform Engineer** | Add new backends or methods | Implement a 3-method protocol, register it. Zero changes to existing code. |
| **New Team Member** | Get started quickly | Clone the repo, `uv sync`, copy a template config, `rl-train train`. |

### 6. Container Recipes

Named presets resolve the DLC image, framework version, and pip dependencies for a model family — no manual version management:

```yaml
container:
  recipe: qwen3_5    # resolves image_uri, framework_version, and all pip deps
```

Available recipes: `qwen3` (Qwen3-14B era), `qwen3_5` (Qwen3.5-27B era). Override any default by setting fields explicitly — `image_uri` overrides the image, `requirements` overrides pip deps. Add new recipes in `foresightrlcore/config/recipes.py`.

For **local GPU** machines (no container), generate a requirements file from a recipe:

```bash
rl-train setup-local-gpu --recipe qwen3_5   # → writes local-requirements.txt
pip install -r local-requirements.txt
```

### Getting Started

```bash
uv sync --all-extras --all-groups   # recommended (all deps + dev/test)
uv pip install -e ".[aws]"         # core + SageMaker support
```

For local GPU machines, generate and install GPU deps from a recipe:

```bash
rl-train setup-local-gpu --recipe qwen3_5   # generates local-requirements.txt
pip install -r local-requirements.txt
```

---

## What's Been Delivered

| Capability | Status |
|-----------|--------|
| **Training hooks** | Implemented — callbacks, pre/post hooks, custom trainer classes via `training.hooks` in YAML |
| **DPO / RLVR / RLAIF methods** | Implemented — via SageMaker Serverless Model Customization (`aws_serverless` backend) |
| **Serverless backend** | Implemented — `aws_serverless` wraps SageMaker SDK v3 trainers (no infra management) |
| **Custom reward tasks** | Implemented — pluggable task modules with `get_reward_function`, `get_callbacks`, `get_dataset_loader` |
| **Performance testing** | Implemented — `perf_test` operation: Gatling load tests against `vllm serve` (TPS sweep, speculative decoding) |
| **GenAI optimizer** | Implemented — `optimizer` operation: quality (deterministic gates + LLM-as-judge) + cost evaluation across candidate LLMs, registry-driven datasets/prompts/modes/judges |

## Future Roadmap

The architecture is designed to grow without rewriting:

| Capability | Extension Point | Effort |
|-----------|----------------|--------|
| **SFT / KTO methods** | Subclass `TrlMethodHandler` in `methods/` | ~50 lines each (shared pipeline in base) |
| **Evaluation pipeline** | New `Operation` in `operations/` | Follows `train` pattern |
| **Kubernetes backend** | New `ExecutionBackend` in `execution/` | Implements 3 methods |
| **Model testing / deployment** | New `Operation` entries | Same registry pattern |

Each addition is **additive** — existing configs, commands, and integrations remain untouched.

---

## Documentation

| Guide | Description |
|-------|-------------|
| [SageMaker Training Jobs](launching-sagemaker-training-job.md) | Full config reference for `aws_training_job` (infra, topology, vLLM, containers) |
| [Serverless Jobs](launching-serverless-job.md) | Config reference for `aws_serverless` (DPO/RLVR/RLAIF, no infra management) |
| [Local GPU Training](launching-local-job.md) | Config reference for `local` execution backend |
| [Performance Testing](launching-performance-test.md) | `perf_test` operation — Gatling load tests against `vllm serve` (TPS sweep, speculative decoding) |
| [Optimizer](launching-optimizer.md) | `optimizer` operation — GenAI quality + cost evaluation across candidate LLMs |
| [Architecture Design](library-architecture-design.md) | Full technical design and migration history |
| [Creating a Task](tasks/creating-a-task.md) | Step-by-step guide to adding a new task plugin |
| [Multi-Reward Functions](tasks/multi-reward-functions.md) | Per-item dispatch and multi-objective reward patterns |
| [Training Hooks](tasks/training-hooks.md) | Callbacks, pre/post hooks, and custom trainer classes |

---

## Summary

`rl-train` turns RL fine-tuning from a collection of ad-hoc scripts into a **structured, extensible platform**:

- **One config** to describe any training job
- **One command** to validate and launch
- **Four protocols** that make every component swappable
- **Zero code changes** needed to add new methods, backends, or tasks
- **Validated before launch** so cloud costs aren't wasted on config errors
- **Secrets at runtime** so credentials never leave the container boundary

The framework handles the complexity so teams can focus on what matters: the research.
