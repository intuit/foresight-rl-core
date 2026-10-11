# Launching a Local GPU Training Job

Complete reference for configuring and running training jobs on a local GPU machine with `rl-train`. Every parameter lives in a YAML config file — nothing is hardcoded.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Environment Setup with Recipes](#environment-setup-with-recipes)
- [Config Structure](#config-structure)
- [Job Identity](#job-identity)
- [Model Configuration](#model-configuration)
- [Methods](#methods)
- [Execution Backend](#execution-backend)
- [Infrastructure Settings](#infrastructure-settings)
  - [vLLM Mode](#vllm-mode)
  - [vLLM Server Settings](#vllm-server-settings)
- [Training Hyperparameters](#training-hyperparameters)
  - [GRPO Parameters](#grpo-parameters)
- [Data Configuration](#data-configuration)
- [Output Configuration](#output-configuration)
- [Environment](#environment)
- [Secrets Management](#secrets-management)
- [Task Configuration](#task-configuration)
  - [Advice Task](#advice-task)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [Validation](#validation)
- [Full Config Example](#full-config-example)

---

> **First time on this backend?** Complete the one-time machine setup before launching:
> [intuit/docs/account-setup/aws/local-gpu.md](../intuit/docs/account-setup/aws/local-gpu.md) — GPU drivers, uv venv, recipe install, optional AWS creds for S3 data, `.env` for local secrets. The steps below assume those exist.

## Quick Start

Run these commands **on the GPU machine** (not the local dev machine):

```bash
# 1. Install the package
uv pip install -e "."

# 2. GPU dependencies for the Qwen3 model in local_example.yaml
rl-train setup-local-gpu --recipe qwen3
pip install -r local-requirements.txt

# 3. Validate. This step does not need a GPU.
rl-train validate --config examples/configs/local_example.yaml

# 4. Launch from the repository root
rl-train train --config examples/configs/local_example.yaml
```

---

## Prerequisites

- A machine with **NVIDIA GPUs** and CUDA drivers installed
- Python 3.10+ with the package installed (`uv pip install -e "."`) plus GPU deps from a recipe (`rl-train setup-local-gpu --recipe <name>`)
- For `vllm_mode: server` — enough GPUs for both vLLM and training (e.g., 8 GPUs with a 2/6 split)
- For `vllm_mode: none` — any number of GPUs (all used for training via torchrun)

---

## Environment Setup with Recipes

Local-GPU recipes provide pinned dependency sets for bare-metal machines. Use `rl-train setup-local-gpu` to generate a requirements file matching the model family you are training:

```bash
# List available recipes
rl-train setup-local-gpu --list

# Generate a requirements file for Qwen3.5 models
rl-train setup-local-gpu --recipe qwen3_5
# → writes local-requirements.txt

# Install the generated requirements
pip install -r local-requirements.txt
```

| Recipe | Key Dependencies | Target Models |
|--------|-----------------|---------------|
| `qwen3` | torch >=2.6, trl[vllm] >=0.28, transformers >=4.57 | Qwen3-14B and similar |
| `qwen3_5` | torch >=2.11, trl >=1.3, vllm >=0.20, transformers >=5.6 | Qwen3.5-27B and similar |

Recipes are defined in `foresightrlcore/config/recipes.py` under `LOCAL_GPU_RECIPES`. To add a new recipe, add an entry there and a test in `tests/test_config_recipes.py`.

> **Note:** Local-GPU recipes are separate from container recipes (used for SageMaker). Container recipes also set `framework_version` and `image_uri`; local recipes are pip-only since there is no DLC base image.

---

## Config Structure

A local training job config uses these top-level sections:

```yaml
version: 1

job:        { ... }       # Job identity
model:      { ... }       # Base model and LoRA settings
method:     { ... }       # Training algorithm (grpo, sft, dpo)
execution:  { ... }       # Must be local + GPU selection
infra:      { ... }       # vLLM mode and server settings (no topology/instances)
training:   { ... }       # Optimizer, schedule, method-specific hyperparameters
data:       { ... }       # Training data (local paths or S3 URIs)
output:     { ... }       # Output directory (local path or S3)
environment: { ... }     # Env vars (Genos identifiers, etc.)
secrets:    { ... }       # Local .env file for credentials
task:       { ... }       # Task plugin (reward function, dataset loader)
```

No `container` section is needed — there is no SageMaker container. Dependencies are installed directly on the GPU machine using `rl-train setup-local-gpu --recipe <name>` (see [Environment Setup with Recipes](#environment-setup-with-recipes)).

---

## Job Identity

```yaml
job:
  name: advice-grpo-local
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *(required)* | Base name for the job. The local backend returns a synthetic job ID: `local-{name}-{timestamp}` |

---

## Model Configuration

```yaml
model:
  name: Qwen/Qwen3-14B
  enable_thinking: true
  lora:
    r: 32
    alpha: 64
    dropout: 0.05
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj
```

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `name` | `str` | *(required)* | — | HuggingFace model ID (e.g. `Qwen/Qwen3-14B`) or local/S3 path |
| `enable_thinking` | `bool` | `true` | — | Enables `<think>` token support for reasoning models |
| `lora.r` | `int` | `32` | `≥ 1` | LoRA rank |
| `lora.alpha` | `int` | `64` | `≥ 1` | LoRA scaling factor (typically `2 x r`) |
| `lora.dropout` | `float` | `0.05` | `0–1` | Dropout on LoRA layers |
| `lora.target_modules` | `list[str]` | Qwen-style defaults | — | Which model modules get LoRA adapters |

---

## Methods

```yaml
method:
  type: grpo
```

The local backend supports the same methods as SageMaker Training Jobs. See [Launching a SageMaker Training Job](launching-sagemaker-training-job.md#methods) for the full method reference including all GRPO parameters.

---

## Execution Backend

The local backend runs training directly on a GPU machine using accelerate or torchrun.

```yaml
execution:
  type: local
  local:
    gpus: "0,1,2,3,4,5,6,7"       # Comma-separated GPU IDs (required)
    # output_dir: ./output/my-run  # Override output directory
    # config_path: null            # Path to pre-written args.yaml (auto-generated if unset)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `local.gpus` | `str` | *(required)* | Comma-separated GPU device IDs visible on the machine |
| `local.output_dir` | `str` | `output.s3_path` value | Local output directory. Defaults to the `output.s3_path` value. |
| `local.config_path` | `str` | auto-generated | Path to write the config YAML for the training script |

**How it works by vLLM mode:**

| vLLM Mode | Behavior |
|-----------|----------|
| `server` | Starts vLLM via `python -m foresightrlcore.runtime.scripts.vllm_launcher` (the shared spawn-safe entry point — see [Reasoning + CUDA-graph Control](reasoning-and-cuda-graph.md)) on the first `vllm_gpu_count` GPUs, then `accelerate launch` on the remaining GPUs |
| `colocate` | vLLM runs inside the training process; all GPUs used for training |
| `none` | `torchrun` with all GPUs; no vLLM process |

---

## Infrastructure Settings

The local backend uses a subset of the `infra` section. Topology and instance settings are not applicable.

```yaml
infra:
  topology: single_node           # Always single_node for local
  vllm_mode: server               # server | colocate | none
  vllm_gpu_count: 2               # GPUs dedicated to vLLM (first N from gpus list)
  training_gpu_count: 6           # GPUs dedicated to training (remaining from gpus list)
```

### vLLM Mode

| vLLM Mode | Description | When to Use |
|-----------|-------------|-------------|
| `server` | Separate vLLM process on dedicated GPUs | Default — best throughput for GRPO. GPU count must be `≥ vllm_gpu_count + training_gpu_count`. |
| `colocate` | vLLM runs inside the TRL trainer process | Memory-constrained setups. |
| `none` | No vLLM server. Uses native HuggingFace generation with torchrun. | Small models, fast iteration, or non-GRPO methods. |

### vLLM Server Settings

When `vllm_mode: server` or `vllm_mode: colocate`:

```yaml
infra:
  vllm:
    port: 8002
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.9
    enforce_eager: false
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `port` | `int` | `8002` | vLLM HTTP server port |
| `tensor_parallel_size` | `int` | `2` | Tensor parallelism size |
| `gpu_memory_utilization` | `float` | `0.9` | Fraction of GPU memory vLLM can use |
| `enforce_eager` | `bool` | `false` | Disable CUDA graphs for debugging |
| `reasoning_parser` / `reasoning_config` / `compilation_config` / `disable_custom_all_reduce` | various | `null` / `false` | Server-mode opt-ins delivered via `vllm_launcher.py` and `patch_trl_vllm`. Same schema and semantics as the SageMaker backend — see [Reasoning + CUDA-graph Control](reasoning-and-cuda-graph.md). |

---

## Training Hyperparameters

General training settings shared across methods.

```yaml
training:
  epochs: 1
  per_device_batch_size: 4
  gradient_accumulation_steps: 2
  learning_rate: 5.0e-05
  lr_scheduler: cosine
  warmup_ratio: 0.03
  bf16: true
  gradient_checkpointing: true
  deepspeed: ds_zero2_config.json
  logging_steps: 5
  save_steps: 50
  eval_steps: 100
  report_to: tensorboard
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `epochs` | `int` | `1` | Number of training epochs |
| `per_device_batch_size` | `int` | `4` | Batch size per GPU |
| `gradient_accumulation_steps` | `int` | `2` | Gradient accumulation steps |
| `learning_rate` | `float` | `5e-5` | Learning rate |
| `weight_decay` | `float` | `0.0` | AdamW weight decay |
| `lr_scheduler` | `str` | `"cosine"` | HuggingFace scheduler type |
| `warmup_ratio` | `float` | `0.03` | Fraction of steps for LR warmup |
| `bf16` | `bool` | `true` | Use bfloat16 mixed precision |
| `gradient_checkpointing` | `bool` | `true` | Enable activation checkpointing |
| `deepspeed` | `str` | `"ds_zero2_config.json"` | DeepSpeed config filename |
| `logging_steps` | `int` | `5` | Log metrics every N steps |
| `save_steps` | `int` | `50` | Save checkpoint every N steps |
| `eval_steps` | `int` | `100` | Evaluate every N steps |
| `max_steps` | `int \| null` | `null` | Override total steps (useful for smoke tests) |
| `max_grad_norm` | `float \| null` | `null` | Gradient clipping norm |
| `log_completions` | `bool` | `true` | Log sample completions |
| `use_liger_kernel` | `bool` | `false` | Use Liger Kernel fused ops (~10–20% throughput on Qwen-class models). The `qwen3_5` local-GPU recipe already includes `liger-kernel`; for other recipes install it manually. |
| `report_to` | `str` | `"tensorboard"` | Reporting integration |

### GRPO Parameters

GRPO-specific settings under `training.grpo`. For the complete 35+ parameter reference, see [Launching a SageMaker Training Job — GRPO](launching-sagemaker-training-job.md#grpo).

```yaml
training:
  grpo:
    num_generations: 8
    max_completion_length: 8192
    beta: 0.001
    epsilon: 0.2
    epsilon_high: 0.28
    loss_type: dr_grpo
    temperature: 0.6
    top_p: 0.95
    top_k: 20
    scale_rewards: false
    mask_truncated_completions: true
    vllm_importance_sampling_correction: false
    num_iterations: 1
```

---

## Data Configuration

Local training accepts both local file paths and S3 URIs (auto-downloaded).

```yaml
data:
  train_path: "/data/train.jsonl"             # Local path
  # train_path: "s3://your-bucket/data/train.jsonl"  # S3 URI (auto-downloaded)
  eval_split_size: 100
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `train_path` | `str` | *(required)* | Local file path or S3 URI |
| `eval_path` | `str \| null` | `null` | Separate eval data |
| `eval_split_size` | `int` | `100` | Holdout samples for eval when no `eval_path` is provided |

**Dataset format:** JSONL files. Each record must have a `prompt` column (string or list of chat messages). Additional columns become keyword arguments to the reward function.

---

## Output Configuration

```yaml
output:
  s3_path: "./output/grpo-advice"   # Local path for output
  # s3_path: "s3://bucket/output/"  # S3 also supported
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `s3_path` | `str` | *(required)* | Output directory (local path or S3 URI) |

---

## Environment

Non-secret environment variables for the training process. For local runs, these are injected into the training subprocess.

```yaml
environment:
  GENOS_ENV: "e2e"
  GENOS_EXPERIENCE_ID: "YOUR_EXPERIENCE_ID"
  GENOS_INTUIT_ORIGINATING_ASSETALIAS: "YOUR_ASSET_ALIAS"
  GENOS_OFFLINE_JOB_ID: "YOUR_OFFLINE_JOB_ID"
```

---

## Secrets Management

For local GPU runs, use `provider: local` which reads credentials from a `.env` file on the machine.

```yaml
secrets:
  provider: local
  local:
    env_file: ".env"                          # Path to the .env file on the GPU machine
  entries:
    - env_var: GENOS_APP_SECRET
      secret_path: "GENOS_APP_SECRET"         # Key name in the .env file
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `str` | — | Use `local` for local GPU runs |
| `local.env_file` | `str` | *(required)* | Path to a `.env` file (`KEY=VALUE` per line) |
| `entries[].env_var` | `str` | — | Environment variable name to set |
| `entries[].secret_path` | `str` | — | Key name in the `.env` file |

The `.env` file format is simple `KEY=VALUE` pairs:

```
GENOS_APP_SECRET=your-secret-value-here
ANOTHER_SECRET=another-value
```

At training startup, `resolve_secrets.py` reads the file, looks up each `secret_path` as a key, and exports the corresponding value as the `env_var`.

> For SageMaker runs using IDPS-backed secrets, see [Launching a SageMaker Training Job — Secrets](launching-sagemaker-training-job.md#secrets-management).

---

## Task Configuration

Tasks provide domain-specific logic: reward functions, callbacks, and dataset loaders. The same task system works on both local and SageMaker.

```yaml
task:
  type: advice
  config:
    reward_mode: llm_judge
    judge_parallel_workers: 10
    think_length_soft_cap: 15000
    think_length_hard_cap: 30000
    think_penalty_weight: 0.15
    predictions_log: "logs/grpo_advice_predictions.jsonl"
    llm_clients:
      default:
        provider: genos
        genos:
          model_id: "YOUR_CRITIC_MODEL_ID"
```

> **Migration note.** The flat `critic_model_id` field has been removed
> and the prototype `llm_judge:` block was renamed to
> `llm_clients.default`. To switch providers without touching the rest
> of the task config, change the `default` entry to
> `provider: bedrock` with a `bedrock:` subkey instead of `genos:` —
> see [LLM Client Configuration](tasks/llm-client-config.md) for the
> full provider comparison and the multi-client pattern.

### Advice Task

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `reward_mode` | `str` | `"llm_judge"` | `llm_judge` for real evaluation, `mock` for testing |
| `llm_clients` | `dict` | *(required for `llm_judge`)* | Mapping of named client configs (see [llm-client-config.md](tasks/llm-client-config.md)). Use `default` for single-client tasks. |
| `judge_parallel_workers` | `int` | — | Thread pool size for parallel LLM evaluation |
| `think_length_soft_cap` | `int` | — | Soft cap on thinking token length |
| `think_length_hard_cap` | `int` | — | Hard cap on thinking token length |
| `think_penalty_weight` | `float` | — | Weight of the thinking-length penalty |
| `predictions_log` | `str` | `"logs/grpo_advice_predictions.jsonl"` | Prediction logging path |
| `mock_constant_reward` | `float` | `0.5` | Fixed reward value in mock mode |
| `mock_sleep_seconds` | `float` | `0` | Simulated latency in mock mode |

For creating custom tasks, see [Creating a Task](tasks/creating-a-task.md). For multi-reward patterns, see [Multi-Reward Functions](tasks/multi-reward-functions.md). For training hooks, see [Training Hooks](tasks/training-hooks.md).

---

## CLI Reference

### Setup Local GPU Environment

```bash
# List available local-GPU recipes
rl-train setup-local-gpu --list

# Generate requirements file from a recipe
rl-train setup-local-gpu --recipe qwen3_5
# → writes local-requirements.txt (install with: pip install -r local-requirements.txt)

# Custom output path
rl-train setup-local-gpu --recipe qwen3_5 --output my-reqs.txt
```

### Validate

```bash
rl-train validate --config my_local_job.yaml
```

### Launch

```bash
# Fire-and-forget
rl-train train --config my_local_job.yaml

# Block until complete (for local, this always blocks)
rl-train train --config my_local_job.yaml --wait
```

> No `AWS_DEFAULT_REGION` needed for local runs (unless your data is on S3).

---

## Python API

```python
from foresightrlcore import train

# Basic launch
job_id = train("my_local_job.yaml")

# With overrides
job_id = train(
    "my_local_job.yaml",
    model_name="Qwen/Qwen3-8B",
    train_data="/data/new-train.jsonl",
    output_path="./output/new-run/",
)
```

---

## Validation

Local-specific validation rules:

| Rule | Effect |
|------|--------|
| `execution.local.gpus` required | Error if not set when `execution.type: local` |
| GPU count check (`vllm_mode: server`) | GPU count must be `≥ vllm_gpu_count + training_gpu_count` |
| S3 paths not required | `data.train_path` and `output.s3_path` can be local paths |
| Infrastructure checks | Topology, instance type, multi-node settings are not validated for local |
| Container section ignored | No container configuration needed |

---

## Full Config Example

A complete config for GRPO training on a local 8-GPU machine:

```yaml
version: 1

job:
  name: advice-grpo-local

method:
  type: grpo

execution:
  type: local
  local:
    gpus: "0,1,2,3,4,5,6,7"

infra:
  topology: single_node
  vllm_mode: server
  vllm_gpu_count: 2
  training_gpu_count: 6
  vllm:
    port: 8002
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.9
    enforce_eager: false

model:
  name: Qwen/Qwen3-14B
  enable_thinking: true
  lora:
    r: 32
    alpha: 64
    dropout: 0.05
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj

training:
  epochs: 1
  per_device_batch_size: 4
  gradient_accumulation_steps: 2
  learning_rate: 5.0e-05
  lr_scheduler: cosine
  warmup_ratio: 0.03
  bf16: true
  gradient_checkpointing: true
  deepspeed: ds_zero2_config.json
  logging_steps: 5
  save_steps: 50
  eval_steps: 100
  report_to: tensorboard
  grpo:
    num_generations: 8
    max_completion_length: 8192
    beta: 0.001
    epsilon: 0.2
    epsilon_high: 0.28
    loss_type: dr_grpo
    temperature: 0.6
    top_p: 0.95
    top_k: 20
    scale_rewards: false
    mask_truncated_completions: true
    vllm_importance_sampling_correction: false
    num_iterations: 1

data:
  train_path: "/data/train.jsonl"
  eval_split_size: 100

output:
  s3_path: "./output/grpo-advice"

environment:
  GENOS_EXPERIENCE_ID: "YOUR_EXPERIENCE_ID"
  GENOS_INTUIT_ORIGINATING_ASSETALIAS: "YOUR_ASSET_ALIAS"
  GENOS_OFFLINE_JOB_ID: "YOUR_OFFLINE_JOB_ID"

secrets:
  provider: local
  local:
    env_file: ".env"
  entries:
    - env_var: GENOS_APP_SECRET
      secret_path: "GENOS_APP_SECRET"

task:
  type: advice
  config:
    judge_parallel_workers: 10
    think_length_soft_cap: 15000
    think_length_hard_cap: 30000
    think_penalty_weight: 0.15
    predictions_log: "logs/grpo_advice_predictions.jsonl"
    llm_clients:
      default:
        provider: genos
        genos:
          model_id: "YOUR_CRITIC_MODEL_ID"
```

**Torchrun mode** (no vLLM — smaller models):

```yaml
infra:
  topology: single_node
  vllm_mode: none

execution:
  type: local
  local:
    gpus: "0,1,2,3"
```

**Smoke test** (fast validation run):

```yaml
training:
  max_steps: 10
task:
  config:
    reward_mode: mock
    mock_constant_reward: 0.5
    mock_sleep_seconds: 0
```
