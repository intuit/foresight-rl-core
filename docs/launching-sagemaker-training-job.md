# Launching a SageMaker Training Job

Complete reference for configuring and launching SageMaker training jobs with `rl-train`. Every parameter lives in a YAML config file — nothing is hardcoded.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Config Structure](#config-structure)
- [Job Identity](#job-identity)
- [Model Configuration](#model-configuration)
- [Methods](#methods)
  - [GRPO](#grpo)
- [Execution Backend](#execution-backend)
- [Infrastructure & Runtime](#infrastructure--runtime)
  - [Topology](#topology)
  - [vLLM Mode](#vllm-mode)
  - [Infrastructure Profiles](#infrastructure-profiles)
  - [GPU Allocation](#gpu-allocation)
  - [Hardware Settings](#hardware-settings)
  - [Managed Spot Training](#managed-spot-training)
  - [vLLM Server Settings](#vllm-server-settings)
  - [Multi-Node Scaling: vLLM GPU Allocation](#multi-node-scaling-vllm-gpu-allocation)
  - [Mixed instance types (dedicated_vllm)](#mixed-instance-types-dedicated_vllm)
  - [Multi-Node Configuration](#multi-node-configuration)
  - [VPC Configuration](#vpc-configuration)
- [Training Hyperparameters](#training-hyperparameters)
- [Data Configuration](#data-configuration)
- [Output Configuration](#output-configuration)
- [Environment](#environment)
- [Container Configuration](#container-configuration)
  - [Container Recipes](#container-recipes)
  - [Custom Container Images](#custom-container-images)
- [Secrets Management](#secrets-management)
- [Task Configuration](#task-configuration)
  - [Advice Task](#advice-task)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [Validation](#validation)
- [Container Pipeline](#container-pipeline)
- [Full Config Example](#full-config-example)

---

> **First time on this backend?** Complete the one-time account setup before launching:
> [intuit/docs/account-setup/aws/sagemaker-training-job.md](../intuit/docs/account-setup/aws/sagemaker-training-job.md) — IAM execution role, S3 bucket, GPU quotas, IDPS policy. The steps below assume those exist.

## Quick Start

```bash
# 1. Install
uv sync --all-extras --all-groups  # or: uv pip install -e ".[aws]"

# 2. Copy a template and customize
cp examples/configs/grpo_advice_sagemaker.yaml my_job.yaml

# 3. Validate (no GPU needed)
rl-train validate --config my_job.yaml

# 4. Launch
AWS_DEFAULT_REGION=us-west-2 rl-train train --config my_job.yaml
```

> **No GPU locally.** Jobs run on SageMaker. Validate configs before submitting — each round-trip is expensive.

---

## Config Structure

A SageMaker training job config is a single YAML file with these top-level sections:

```yaml
version: 1               # Config format version

job:        { ... }       # Job identity and runtime limits
model:      { ... }       # Base model and LoRA settings
method:     { ... }       # Training algorithm (grpo, sft, dpo)
execution:  { ... }       # Must be aws_training_job
infra:      { ... }       # Hardware, topology, vLLM, networking
training:   { ... }       # Optimizer, schedule, GRPO hyperparameters
data:       { ... }       # Training and evaluation data paths (S3)
output:     { ... }       # Output and checkpoint locations (S3)
environment: { ... }     # Backend-agnostic env vars (e.g. Genos identifiers)
container:  { ... }       # SageMaker image and framework selection
secrets:    { ... }       # Runtime secret resolution (idps | local)
task:       { ... }       # Task plugin (reward function, dataset loader)
```

The schema is defined in `foresightrlcore/config/schema.py` as `TrainingJobConfig` (Pydantic v2). All fields are validated before anything launches.

---

## Job Identity

Controls job naming and runtime limits.

```yaml
job:
  name: grpo-advice-qwen3        # Base name for the job (required)
  max_runtime_hours: 12           # Max wall-clock time (1–120, default: 12)
```

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `name` | `str` | *(required)* | — | Base name appended with a timestamp to form the job ID |
| `max_runtime_hours` | `int` | `12` | `1–120` | SageMaker `MaxRuntimeInSeconds` (multiplied by 3600) |
| `max_wait_hours` | `int \| null` | `null` | `1–432` | SageMaker `MaxWaitTimeInSeconds` for [Managed Spot Training](#managed-spot-training). Only used when `infra.use_spot_instances` is `true`; must be `≥ max_runtime_hours`. Defaults to `max_runtime_hours` if unset. |

---

## Model Configuration

Specifies the base model and LoRA adapter settings.

```yaml
model:
  name: Qwen/Qwen3-0.6B          # HuggingFace model ID or S3 path (required)
  enable_thinking: true            # Enable thinking tokens for reasoning models
  lora:
    r: 32                          # LoRA rank
    alpha: 64                      # LoRA alpha
    dropout: 0.05                  # LoRA dropout
    target_modules:                # Modules to adapt
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - up_proj
      - down_proj
      - gate_proj
```

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `name` | `str` | *(required)* | — | HuggingFace model ID (e.g. `Qwen/Qwen3-0.6B`) or S3 prefix (e.g. `s3://bucket/models/Qwen3.6-27B/`) — see [Pre-Staged Models](#pre-staged-models-s3-modelname) |
| `enable_thinking` | `bool` | `true` | — | Enables `<think>` token support for reasoning models (Qwen3, etc.) |
| `lora.r` | `int` | `32` | `≥ 1` | LoRA rank — higher = more parameters, more capacity |
| `lora.alpha` | `int` | `64` | `≥ 1` | LoRA scaling factor (typically `2 × r`) |
| `lora.dropout` | `float` | `0.05` | `0–1` | Dropout on LoRA layers |
| `lora.target_modules` | `list[str]` | Qwen-style defaults | — | Which model modules get LoRA adapters |

### Pre-Staged Models (`s3://` model.name)

`model.name` accepts an **S3 prefix** in addition to a HuggingFace Hub ID. When you provide an `s3://` value:

```yaml
model:
  name: s3://my-bucket/models/Qwen3.6-27B/
```

the SageMaker launcher automatically adds a `model` input channel pointing at that prefix. SageMaker downloads the weights to `/opt/ml/input/data/model` **before** the container entrypoint runs, and both `entrypoint.sh` and `train.py`'s `_apply_sagemaker_overrides` swap the YAML model name for the local mount path. The container therefore loads weights from local disk and never calls the Hugging Face Hub.

**When to use it:**

- Multi-rank training jobs hitting `huggingface_hub.snapshot_download()` cache races (`OSError: ... does not appear to have a file named model.safetensors-NN-of-NN.safetensors`).
- Air-gapped accounts or VPCs without outbound HF Hub access.
- Reproducibility — pinning the exact weight files used for a run.

**How to stage a model to S3:**

```bash
huggingface-cli download Qwen/Qwen3.5-27B --local-dir /tmp/qwen35-27b
AWS_DEFAULT_REGION=us-west-2 aws s3 sync /tmp/qwen35-27b s3://my-bucket/models/Qwen3.5-27B/
```

No other config changes are required — the launcher detects the `s3://` prefix via `config.model.is_s3_uri` and wires the channel for you.

---

## Methods

Methods define *how* the model is trained. Each method implements the `MethodHandler` protocol.

```yaml
method:
  type: grpo                     # Training algorithm (default: "grpo")
```

| Method | Status | Description |
|--------|--------|-------------|
| `grpo` | **Implemented** | Group Relative Policy Optimization via TRL's GRPOTrainer |
| `sft` | *Planned* | Supervised Fine-Tuning |
| `dpo` | *Planned* | Direct Preference Optimization |
| `kto` | *Planned* | Kahneman-Tversky Optimization |

### GRPO

GRPO (Group Relative Policy Optimization) generates multiple completions per prompt, scores them with a reward function, and updates the policy using relative advantage within the group.

GRPO-specific settings live under `training.grpo`:

```yaml
training:
  grpo:
    num_generations: 8              # Completions per prompt (≥ 2)
    max_completion_length: 8192     # Max new tokens per completion
    beta: 0.001                     # KL penalty coefficient
    epsilon: 0.2                    # PPO-style clip low bound
    epsilon_high: 0.28              # PPO-style clip high bound
    loss_type: dr_grpo              # TRL loss variant
    temperature: 0.6                # Sampling temperature
    top_p: 0.95                     # Nucleus sampling threshold
    top_k: 20                       # Top-k sampling
    scale_rewards: false            # Normalize rewards
    mask_truncated_completions: true
    num_iterations: 1               # GRPO inner iterations
```

**Core parameters:**

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `num_generations` | `int` | `8` | `≥ 2` (enforced by handler) | Number of completions sampled per prompt |
| `max_completion_length` | `int` | `8192` | `≥ 64` | Maximum new tokens generated per completion |
| `beta` | `float` | `0.001` | `≥ 0` | KL divergence penalty weight. `0` skips reference model loading. |
| `epsilon` | `float` | `0.2` | `≥ 0` | PPO-style clipping lower bound |
| `epsilon_high` | `float` | `0.28` | `≥ 0` | PPO-style clipping upper bound (DAPO recommends 0.28) |
| `loss_type` | `str` | `"dr_grpo"` | See below | TRL loss function variant |
| `temperature` | `float` | `0.6` | `≥ 0` | Generation sampling temperature |
| `top_p` | `float` | `0.95` | `0–1` | Nucleus sampling probability mass |
| `top_k` | `int` | `20` | `≥ 0` | Top-k sampling (0 = disabled) |
| `min_p` | `float \| null` | `null` | `0–1` | Minimum token probability, scaled by top token. Typical: 0.01–0.2. |
| `repetition_penalty` | `float` | `1.0` | `≥ 0` | Token repetition penalty. >1.0 encourages diversity. |
| `scale_rewards` | `str \| bool` | `false` | `"group"`, `"batch"`, `"none"`, or bool | Reward scaling strategy |
| `mask_truncated_completions` | `bool` | `true` | — | Mask loss for completions that hit the length limit |
| `vllm_importance_sampling_correction` | `bool` | `false` | — | Importance sampling correction for vLLM |
| `num_iterations` | `int` | `1` | `≥ 1` | Number of GRPO inner optimization iterations |
| `importance_sampling_level` | `str \| null` | `null` | — | IS granularity: `"token"` or `"sequence"` |
| `reward_weights` | `list[float] \| null` | `null` | — | Weights for multi-objective rewards (see [Multi-Reward Functions](tasks/multi-reward-functions.md)) |

**Loss types** (`loss_type`): `grpo`, `dr_grpo`, `dapo`, `bnpo`, `cispo`, `sapo`, `luspo`, `vespo`. See [TRL docs](https://huggingface.co/docs/trl/main/en/grpo_trainer) for details on each variant.

**Generation control:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `generation_batch_size` | `int \| null` | `null` | Explicit generation batch size. Mutually exclusive with TRL's `steps_per_generation`. |
| `num_generations_eval` | `int \| null` | `null` | Fewer generations during eval to save compute |
| `generation_kwargs` | `dict \| null` | `null` | Escape hatch for arbitrary generation params (suppress_tokens, num_beams, etc.) |

**Advanced loss and training:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `delta` | `float \| null` | `null` | Two-sided GRPO clipping upper bound (INTELLECT-2). Recommended > 1 + epsilon. |
| `sync_ref_model` | `bool` | `false` | Periodically sync reference model with active model (TR-DPO). Requires `beta > 0`. |
| `ref_model_mixup_alpha` | `float` | `0.6` | Mix ratio for ref model sync: `pi_ref = alpha * pi_theta + (1-alpha) * pi_ref_prev` |
| `ref_model_sync_steps` | `int` | `512` | Steps between reference model syncs |
| `multi_objective_aggregation` | `str` | `"sum_then_normalize"` | Multi-reward aggregation: `sum_then_normalize` or `normalize_then_sum` (GDPO) |
| `top_entropy_quantile` | `float` | `1.0` | Keep only top-rho tokens by entropy (Beyond 80/20 Rule). 1.0 = all tokens, paper recommends 0.2. |

**Importance sampling (extended):**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vllm_importance_sampling_mode` | `str \| null` | `null` | IS mode: `token_truncate`, `token_mask`, `sequence_truncate`, `sequence_mask` |
| `vllm_importance_sampling_cap` | `float` | `3.0` | IS cap C for truncation/masking |
| `off_policy_mask_threshold` | `float \| null` | `null` | Off-policy sequence masking threshold (DeepSeek-V3.2, e.g. 0.5) |
| `use_bias_correction_kl` | `bool` | `false` | Unbiased KL estimator with IS correction (DeepSeek-V3.2) |

**Model control:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `disable_dropout` | `bool` | `false` | Disable dropout for consistent logprobs with reference model |
| `cast_lm_head_to_fp32` | `bool` | `false` | Cast LM head to fp32 (ScaleRL recommendation for untied embeddings) |

**Data and distributed:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ds3_gather_for_generation` | `bool` | `true` | DeepSpeed ZeRO-3: gather weights during generation for speed |
| `shuffle_dataset` | `bool` | `true` | Whether to shuffle the training dataset |

**Agentic training:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_tool_calling_iterations` | `int \| null` | `null` | Max tool-calling turns when training an agent. `null` = no limit. |

---

## Execution Backend

The SageMaker backend submits training jobs using PyTorch training images.

```yaml
execution:
  type: aws_training_job
```

**Requirements:**
- `boto3` and `sagemaker` SDK installed (`uv pip install -e ".[aws]"`)
- `data.train_path` must start with `s3://`
- `output.s3_path` must start with `s3://`
- AWS credentials configured (no default region — always set `AWS_DEFAULT_REGION=us-west-2`)

**How it works:**
1. Uploads the YAML config to S3 under the output prefix
2. Creates a SageMaker estimator with the configured instance type, image, and channels
3. Submits the job with `train`, `config`, and optional `eval` input channels
4. Returns the SageMaker job ID

**Branching logic based on topology:**
- `topology: multi_node` → Heterogeneous instance groups (SDK v2) using `infra.multi_node` settings
- All other topologies → Homogeneous estimator (SDK v2) or `ModelTrainer` (SDK v3)
- `vllm_mode: none` → Torchrun distribution strategy with `requirements_torchrun.txt`
- `training_plan_arn` set + SDK v3 → Falls back to boto3 `CreateTrainingJob` to inject the plan ARN

> For other backends, see [Launching a Serverless Job](launching-serverless-job.md) and [Launching a Local GPU Job](launching-local-job.md).

---

## Infrastructure & Runtime

The `infra` section controls hardware topology, GPU allocation, and networking.

```yaml
infra:
  instance_type: ml.p4d.24xlarge
  instance_count: 1
  topology: single_node
  vllm_mode: server
  vllm_gpu_count: 2
  training_gpu_count: 6
  volume_size_gb: 100
  keep_alive_seconds: 0
  nccl_transport: efa
  encrypt_inter_container_traffic: true
  # Mixed dedicated_vllm (opt-in): different instance type for the vLLM node.
  # See docs/launching-sagemaker-training-job.md#mixed-instance-types-dedicated_vllm
  # topology: dedicated_vllm
  # instance_count: 2
  # vllm_instance_type: ml.g6.16xlarge
  # vllm_instance_count: 1
```

### Topology

Topology defines how instances and GPUs are organized in the training cluster.

```yaml
infra:
  topology: single_node            # Cluster layout
```

| Topology | Description | When to Use |
|----------|-------------|-------------|
| `single_node` | All GPUs on one instance, split between vLLM and training | Default — most experiments. Single-node, no multi-node overhead. |
| `multi_node` | Heterogeneous instance groups (separate vLLM and training instance types) | Different hardware for generation vs. training (e.g., g5 for vLLM, p4d for training). Requires SDK v2. |
| `multi_node_server` | Homogeneous multi-node cluster, node 0 runs vLLM server | High-throughput training where all nodes are the same type. |
| `dedicated_vllm` | Homogeneous by default (node 0 all-vLLM, remaining nodes train). Opt-in mixed types via `infra.vllm_instance_type` (SageMaker InstanceGroups). | Large-scale production. Requires `instance_count ≥ 2`. See [Mixed instance types](#mixed-instance-types-dedicated_vllm). |

### vLLM Mode

Controls how the vLLM generation server is managed during GRPO training.

```yaml
infra:
  vllm_mode: server                # How vLLM runs
```

| vLLM Mode | Description | When to Use |
|-----------|-------------|-------------|
| `server` | Separate vLLM process on dedicated GPUs | Default — best throughput for GRPO. Requires GPU allocation via `vllm_gpu_count` / `training_gpu_count`. |
| `colocate` | vLLM runs inside the TRL trainer process | Memory-constrained setups. Configured via `infra.vllm.tensor_parallel_size`. |
| `none` | No vLLM server. Uses native HuggingFace generation with torchrun. | Small models, fast iteration, or non-GRPO methods (SFT/DPO). Slower generation. |

**Interaction matrix (topology x vllm_mode):**

| | `server` | `colocate` | `none` |
|---|---|---|---|
| `single_node` | GPUs split on one instance | vLLM in-process | torchrun, all GPUs train |
| `multi_node` | Heterogeneous groups | N/A | torchrun multi-node |
| `multi_node_server` | Node 0 runs vLLM split | N/A | N/A |
| `dedicated_vllm` | Homogeneous: node 0 all-vLLM, rest train. Mixed: roles from `SM_CURRENT_INSTANCE_GROUP` | N/A | N/A |

### Infrastructure Profiles

Named presets that map to a `topology` + `vllm_mode` combination.

```yaml
infra:
  infra_profile: dedicated_vllm   # Optional named preset
```

| Profile | Topology | vLLM Mode | Description |
|---------|----------|-----------|-------------|
| `single_split_vllm` | `single_node` | `server` | Default — single node with GPU split |
| `torchrun` | `single_node` | `none` | Pure torchrun, no vLLM |
| `colocate` | `single_node` | `colocate` | vLLM inside trainer process |
| `dedicated_vllm` | `dedicated_vllm` | `server` | Full node for vLLM |
| `multi_node_server` | `multi_node_server` | `server` | Homogeneous multi-node |

### GPU Allocation

When `vllm_mode: server`, GPUs are explicitly allocated between vLLM and training:

```yaml
infra:
  vllm_gpu_count: 2               # GPUs for vLLM server (default: 2)
  training_gpu_count: 6            # GPUs for training (default: 6)
```

For an 8-GPU instance like `ml.p4d.24xlarge`, a `2/6` split is typical. Adjust based on model size and generation throughput needs.

### Hardware Settings

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `instance_type` | `str` | `ml.p4d.24xlarge` | — | SageMaker instance type (training nodes; also vLLM when `vllm_instance_type` is unset) |
| `instance_count` | `int` | `1` | `≥ 1` | Number of instances in the cluster (total size, including the vLLM node) |
| `vllm_instance_type` | `str \| null` | `null` | dedicated_vllm + aws_training_job | Opt-in mixed types: vLLM group instance type. See [Mixed instance types](#mixed-instance-types-dedicated_vllm). |
| `vllm_instance_count` | `int` | `1` | `≥ 1`, currently must be `1` when mixed | vLLM group size when `vllm_instance_type` is set. Ignored otherwise. |
| `allow_mixed_gpu_families` | `bool` | `false` | — | Permit mixing GPU families / EFA capability (e.g. g5 + p4d). Rejected by default because TRL NCCL-syncs weights to vLLM. |
| `vllm_gpu_count` | `int` | `2` | `≥ 1` | GPUs reserved for vLLM (server mode) |
| `training_gpu_count` | `int` | `6` | `≥ 1` | GPUs reserved for training (server mode) |
| `volume_size_gb` | `int` | `100` | `≥ 1` | EBS volume per instance (GB) |
| `keep_alive_seconds` | `int` | `0` | `≥ 0` | SageMaker warm pool / keep-alive duration |
| `use_spot_instances` | `bool` | `false` | — | Enable [Managed Spot Training](#managed-spot-training). Requires `output.checkpoint_s3_path`. Homogeneous topologies only. |
| `role` | `str \| null` | `null` | — | SageMaker execution role ARN (auto-resolved if unset) |
| `nccl_transport` | `str` | `"efa"` | — | NCCL transport layer (used by `nccl_env.py` inside container) |
| `training_plan_arn` | `str \| null` | `null` | — | SageMaker Training Plan ARN (forces boto3 submit path if set with SDK v3) |
| `encrypt_inter_container_traffic` | `bool` | `true` | — | Encrypt inter-container traffic (TLS) between compute instances. Maps to SageMaker `EnableInterContainerTrafficEncryption` on all launch paths (SDK v2 `encrypt_inter_container_traffic`, SDK v3 `Networking.enable_inter_container_traffic_encryption`, boto3 `EnableInterContainerTrafficEncryption`). **On by default** to satisfy Intuit ACP policy `IS031.1`. Only affects multi-instance jobs; harmless for single-instance. No VPC required. |

### Managed Spot Training

Run training on EC2 Spot instances to save up to ~70-90% vs on-demand. SageMaker manages spot interruptions for you: an interrupted job is automatically restarted on new capacity (status transitions `Training → Interrupted → Starting → Training`) until it completes or the `max_wait_hours` window is exhausted.

```yaml
job:
  max_runtime_hours: 24
  max_wait_hours: 48            # >= max_runtime_hours; includes spot-capacity wait time

infra:
  use_spot_instances: true

output:
  s3_path: s3://my-bucket/output/my-run
  checkpoint_s3_path: s3://my-bucket/checkpoints/my-run   # REQUIRED for spot
```

**The three requirements** (all handled by the launcher when `use_spot_instances: true`):

1. `EnableManagedSpotTraining=True` — set on all launch paths (SDK v2 `use_spot_instances`, SDK v3 `Compute.enable_managed_spot_training`, boto3 `EnableManagedSpotTraining`).
2. `max_wait_hours ≥ max_runtime_hours` — maps to `MaxWaitTimeInSeconds`. If omitted, it defaults to `max_runtime_hours`. `max_wait` is the total time including waiting for spot capacity plus run time across interruptions, so give it buffer.
3. `output.checkpoint_s3_path` **must be set** — validation fails otherwise. SageMaker continuously syncs `/opt/ml/checkpoints` ↔ this S3 prefix, and restores it before each restart.

**Automatic resume.** Intermediate `checkpoint-N/` dirs are written to `/opt/ml/checkpoints` every `training.save_steps`. On a restart, the trainer auto-detects the latest checkpoint under that directory and resumes from it — you do **not** need to set `training.resume_from_checkpoint`. Because a spot interruption loses at most `save_steps` of progress, prefer a modest `save_steps` (e.g. `25`) for long spot runs; the trade-off is more frequent S3 sync.

**Constraints.**

- Spot is supported for homogeneous topologies only: `single_node`, `multi_node_server`, and homogeneous `dedicated_vllm` (no `vllm_instance_type`). The heterogeneous `multi_node` topology and mixed-type `dedicated_vllm` are rejected by validation (not an AWS limitation — those launcher paths are deferred).
- Spot GPU capacity for `p4d`/`p4de`/`p5` classes can be scarce in a region; interruptions add wall-clock latency. If the wait window is exceeded the job ends with `MaxWaitTimeExceeded`.

### vLLM Server Settings

Fine-grained control over the vLLM generation server.

```yaml
infra:
  vllm:
    port: 8002                     # Server listen port
    tensor_parallel_size: 2        # Tensor parallelism degree
    gpu_memory_utilization: 0.9    # GPU memory fraction for vLLM
    max_model_len: null            # Max sequence length (null = auto)
    enforce_eager: false           # Disable CUDA graph compilation
```

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `port` | `int` | `8002` | `1024–65535` | vLLM HTTP server port |
| `tensor_parallel_size` | `int` | `2` | `≥ 1` | Tensor parallelism size (also used for colocate mode via TRL) |
| `gpu_memory_utilization` | `float` | `0.9` | `0.1–1.0` | Fraction of GPU memory vLLM can use |
| `max_model_len` | `int \| null` | `null` | `≥ 256` if set | Maximum sequence length; `null` lets vLLM auto-detect |
| `enforce_eager` | `bool` | `false` | — | Disable CUDA graphs. **Recommended: keep `false`** — enabling it disables CUDA graph compilation and imposes a ~14% throughput penalty. Only use `true` to debug OOM or graph compilation errors. |
| `startup_timeout` | `int` | `1800` | `60–7200` | Seconds to wait for vLLM server to become healthy. Training nodes add a 300s buffer on top. Increase for very large models. |
| `enable_prefix_caching` | `bool \| null` | `null` | — | Forwarded to `vllm serve --enable-prefix-caching`. `null` keeps vLLM's default. |
| `speculative` | `dict \| null` | `null` | — | Speculative-decoding config. Server-mode only. See [Speculative Decoding](speculative-decoding.md). |
| `reasoning_parser` | `str \| null` | `null` | — | Reasoning parser identifier (e.g. `"qwen3"`). Server-mode only. Enables vLLM's `ThinkingTokenBudgetLogitsProcessor` for thinking models. See [Reasoning + CUDA-graph Control](reasoning-and-cuda-graph.md). |
| `reasoning_config` | `dict \| null` | `null` | — | `{reasoning_start_str, reasoning_end_str}`. Required alongside `reasoning_parser`. Server-mode only. |
| `compilation_config` | `dict \| null` | `null` | — | Kwargs forwarded to vLLM's `CompilationConfig` (e.g. `{cudagraph_capture_sizes: [1,2,4], cudagraph_num_of_warmups: 1}`). Use to bound CUDA-graph capture and avoid the H100 graph-compile hang. Server-mode only. |
| `disable_custom_all_reduce` | `bool` | `false` | — | Inject `disable_custom_all_reduce=True` into vLLM's `LLM(...)` call. Workaround for the broken vLLM 0.20.x custom all-reduce kernel at `tensor_parallel_size > 1` on H100/H200. Server-mode only. |

> `enforce_eager: true` disables CUDA graph compilation in vLLM, causing a measured **13.7% throughput slowdown**. The schema default is `false` — never override it to `true` in production configs.
>
> **Reasoning / compilation_config / disable_custom_all_reduce delivery.** TRL's `vllm-serve` CLI does not expose these kwargs. Every server-mode launch goes through `runtime/scripts/vllm_launcher.py` (not `trl vllm-serve` directly), which reads three env vars (`VLLM_REASONING_CONFIG_JSON`, `VLLM_COMPILATION_CONFIG_JSON`, `VLLM_DISABLE_CUSTOM_ALL_REDUCE`) populated from YAML by `entrypoint.sh` and calls `patch_trl_vllm.patch(...)` to inject them into TRL's `LLM(...)` call. The patch is idempotent and **fails fast** if TRL upstream rewrites its source — the job exits non-zero rather than silently dropping the feature. The same launcher is used by the local and HyperPod EKS backends, so opt-in behavior is identical across backends. See [Reasoning + CUDA-graph Control](reasoning-and-cuda-graph.md).

### Multi-Node Scaling: vLLM GPU Allocation

When using `multi_node_server` or `dedicated_vllm` with multiple instances, scale `vllm_gpu_count` and `tensor_parallel_size` proportionally with instance count. A fixed 2-GPU vLLM server cannot keep up with the generation requests from 3+ training nodes and becomes the throughput bottleneck.

**Recommended GPU split for `ml.p4de.24xlarge` (8 GPUs/node):**

| Instance Count | `vllm_gpu_count` | `training_gpu_count` | `tensor_parallel_size` |
|----------------|-----------------|----------------------|------------------------|
| 2 | 2 | 6 | 2 |
| 4 | **4** | **4** | **4** |

Always match `tensor_parallel_size` to `vllm_gpu_count`.

### Mixed instance types (`dedicated_vllm`)

A SageMaker Training Job can attach two instance types in one cluster via `ResourceConfig.InstanceGroups`. On `topology: dedicated_vllm` this is **opt-in**: omit `vllm_instance_type` and the job stays homogeneous (`instance_type` × `instance_count`), identical to today.

```yaml
infra:
  topology: dedicated_vllm
  instance_type: ml.g6e.48xlarge     # training group
  instance_count: 2                  # still total cluster size
  vllm_instance_type: ml.g6e.12xlarge  # OPT-IN: enables InstanceGroups
  vllm_instance_count: 1               # currently must be 1
```

Semantics:

- `instance_count` does **not** change meaning. Training nodes = `instance_count - vllm_instance_count`.
- Groups are named `vllm` and `training`. The entrypoint assigns roles from `SM_CURRENT_INSTANCE_GROUP`, not `SM_HOSTS` rank. Host lists come from `SM_TRAINING_ENV.instance_groups_dict` / `resource_config.instance_groups` (a list of `{instance_group_name, hosts}`). `algo-1` is not necessarily the vLLM node.
- Mixed jobs always submit via boto3 `CreateTrainingJob` (SDK v3 `ModelTrainer` cannot safely send InstanceGroups today).
- SageMaker `CheckpointConfig` is **omitted** for mixed jobs even if `output.checkpoint_s3_path` is set. That agent two-way-syncs one prefix from every instance; the vLLM group's empty `/opt/ml/checkpoints` races the training group's DeepSpeed shards (S3 SlowDown, job Failed after a successful train). Instead, **training ranks** `aws s3 sync` `/opt/ml/checkpoints` ↔ `checkpoint_s3_path` on restore and after each save (same live-checkpoint URI as homogeneous jobs; the vLLM node never writes it). Do not pass `--delete` — multiple training nodes upload complementary shards. Spot is still rejected on this path. Final weights still upload from `/opt/ml/model` to `output.s3_path`.
- DeepSpeed GRPO on the training group needs a process group. A 1-GPU training instance makes accelerate use `simple_launcher`, which never calls `init_process_group`. Use a training type with **≥2 GPUs** (for example `ml.g6.12xlarge`, not `ml.g6.16xlarge`).
- `infra.vllm.tensor_parallel_size` must be ≤ GPUs on **`vllm_instance_type`**, not the training type.
- Spot, warm pools (`keep_alive_seconds > 0`), local, HyperPod, and serverless are rejected for this path.

**GPU family / EFA.** TRL server-mode NCCL-syncs LoRA/weights from trainers to the vLLM process. Mixing families (A10G `g5` + A100 `p4d` + H100 `p5`) or mixing EFA vs non-EFA (`g5.4xlarge` has no EFA, `g5.8xlarge+` does) is rejected unless `allow_mixed_gpu_families: true`. Preferred pairs: same family, different size.

**VPC + EFA.** SageMaker rejects InstanceGroups in VPC mode when any group uses an EFA-capable instance (`Currently heterogeneous cluster only supports non-EFA instance in VPC mode.`). That includes `g5.8xlarge` and larger (listed as EFA-supported even without RDMA). For EFA pairs (`g6e.12xlarge` + `g6e.48xlarge`, `p4d` + `p4de`) omit `infra.vpc` and use a Bedrock judge (no Genos/IDPS). VPC jobs must use a true non-EFA pair such as `g5.2xlarge` + `g5.4xlarge`. One container image serves both nodes; the launcher resolves the DLC URI from the higher-capability type.

This is independent of `topology: multi_node` (legacy heterogeneous groups, SDK v2 only). Do not switch topologies to get mixed types.

### Multi-Node NCCL and EFA Tuning

For multi-node jobs on EFA-enabled instances (`ml.p4d`, `ml.p4de`, `ml.p5`), add these recommended env vars:

```yaml
environment:
  NCCL_BUFFSIZE: "8388608"           # 8MB EFA send buffer
  NCCL_P2P_NET_CHUNKSIZE: "524288"   # 512KB P2P chunk size
```

**Avoid these known-bad settings:**
- `NCCL_PROTO: "Simple"` — confirmed zero measurable effect vs NCCL auto-selection; overrides optimal protocol choice unnecessarily.
- `NCCL_ALGO: "Tree"` — incompatible with AllGather ops used by vLLM; causes immediate job failure.

### Multi-Node Configuration

Only used when `topology: multi_node`. Defines heterogeneous instance groups for separate vLLM and training hardware. This is the **legacy SDK v2** path. For mixed types on `dedicated_vllm`, use `infra.vllm_instance_type` instead — see [Mixed instance types](#mixed-instance-types-dedicated_vllm).

```yaml
infra:
  topology: multi_node
  multi_node:
    vllm_instance_type: ml.g5.12xlarge
    vllm_instance_count: 1
    training_instance_type: ml.p4d.24xlarge
    training_instance_count: 1
```

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `vllm_instance_type` | `str` | `ml.g5.12xlarge` | — | Instance type for the vLLM generation group |
| `vllm_instance_count` | `int` | `1` | `≥ 1` | Number of vLLM instances |
| `training_instance_type` | `str` | `ml.p4d.24xlarge` | — | Instance type for the training group |
| `training_instance_count` | `int` | `1` | `≥ 1` | Number of training instances |

### VPC Configuration

Optional — only needed if the SageMaker job must run inside a specific VPC.

Listing multiple subnets across availability zones is recommended — SageMaker will select one with available capacity, which avoids IP-exhaustion failures and gives AZ diversity for free.

```yaml
infra:
  vpc:
    subnets:
      - subnet-abc123   # AZ-1
      - subnet-def456   # AZ-2
      - subnet-ghi789   # AZ-3
    security_groups:
      - sg-xyz000
```

Mixed `dedicated_vllm` (`infra.vllm_instance_type` set) cannot combine VPC with EFA-capable instance types. SageMaker rejects that with `heterogeneous cluster only supports non-EFA instance in VPC mode`. See [Mixed instance types](#mixed-instance-types-dedicated_vllm).

---

## Training Hyperparameters

General training settings shared across methods.

```yaml
training:
  epochs: 1
  per_device_batch_size: 4
  gradient_accumulation_steps: 2
  learning_rate: 5e-5
  weight_decay: 0.0
  lr_scheduler: cosine
  warmup_ratio: 0.03
  bf16: true
  gradient_checkpointing: true
  deepspeed: ds_zero2_config.json
  logging_steps: 5
  save_steps: 50
  eval_steps: 100
  max_steps: null
  max_grad_norm: null
  log_completions: true
  use_liger_kernel: false          # Opt-in fused RoPE/SwiGLU/LN/CE kernels
  report_to: tensorboard
```

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `epochs` | `int` | `1` | `≥ 1` | Number of training epochs |
| `per_device_batch_size` | `int` | `4` | `≥ 1` | Batch size per GPU |
| `gradient_accumulation_steps` | `int` | `2` | `≥ 1` | Gradient accumulation steps |
| `learning_rate` | `float` | `5e-5` | `> 0` | Learning rate |
| `weight_decay` | `float` | `0.0` | `≥ 0` | AdamW weight decay coefficient |
| `lr_scheduler` | `str` | `"cosine"` | — | HuggingFace scheduler type |
| `warmup_ratio` | `float` | `0.03` | `0–1` | Fraction of steps for LR warmup |
| `bf16` | `bool` | `true` | — | Use bfloat16 mixed precision |
| `fp16` | `bool` | `false` | — | Use fp16 mixed precision (for older hardware without bf16 support). Cannot be used with `bf16: true`. |
| `gradient_checkpointing` | `bool` | `true` | — | Enable activation checkpointing to save memory |
| `gradient_checkpointing_kwargs` | `dict \| null` | `null` | — | Extra kwargs; GRPO defaults `{"use_reentrant": false}` |
| `deepspeed` | `str` | `"ds_zero2_config.json"` | — | DeepSpeed config filename (relative to runtime scripts) |
| `logging_steps` | `int` | `5` | `≥ 1` | Log metrics every N steps |
| `save_steps` | `int` | `50` | `≥ 1` | Save checkpoint every N steps |
| `eval_steps` | `int` | `100` | `≥ 1` | Evaluate every N steps (when eval data exists) |
| `max_steps` | `int \| null` | `null` | — | Override total steps (useful for smoke tests) |
| `max_grad_norm` | `float \| null` | `null` | `≥ 0` | Gradient clipping norm |
| `log_completions` | `bool` | `true` | — | Log sample completions during training |
| `use_liger_kernel` | `bool` | `false` | — | Route the model through `LigerForCausalLM` for fused RoPE / SwiGLU / LayerNorm / cross-entropy ops (~10–20% throughput uplift on Qwen-class models). Requires the `liger-kernel` pip package — bundled in the `qwen3_5` recipe; add manually for other recipes. Forwarded by `TrlMethodHandler` to every TRL trainer (GRPO/SFT/DPO/RLVR/RLAIF). |
| `report_to` | `str` | `"tensorboard"` | — | Reporting integration (tensorboard, wandb, etc.) |

**DeepSpeed configs available:**
- `ds_zero2_config.json` — ZeRO Stage 2, 500MB allgather/reduce buckets. Default for single-node and full-parameter fine-tuning.
- `ds_zero2_lora_config.json` — ZeRO Stage 2, **50MB buckets sized for LoRA gradients** (~100–200MB total). **Recommended for LoRA multi-node training.**
- `ds_zero3_config.json` — ZeRO Stage 3 (for very large models that don't fit in GPU memory)
- `ds_zero3_hpz_config.json` — ZeRO Stage 3 with hierarchical partitioning (experimental)

> **Multi-node LoRA tip:** Use `ds_zero2_lora_config.json` instead of `ds_zero2_config.json` for LoRA runs. The 500MB default buckets rarely fill with LoRA gradients, causing unnecessary flush delays.

---

## Data Configuration

Paths to training and evaluation datasets. All paths must be S3 URIs for SageMaker.

```yaml
data:
  train_path: s3://my-bucket/datasets/advice/train/
  eval_path: s3://my-bucket/datasets/advice/eval/     # Optional
  eval_split_size: 100
```

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `train_path` | `str` | *(required)* | Must start with `s3://` | Directory containing JSONL training data |
| `eval_path` | `str \| null` | `null` | — | Separate eval data directory. If unset, a holdout split is created from training data. |
| `eval_split_size` | `int` | `100` | `≥ 1` | Number of samples held out for eval when no `eval_path` is provided |

**Dataset format:** JSONL files in a directory. Each record must have a `prompt` column (either a string or a list of chat messages). Additional columns become keyword arguments to the reward function.

```jsonl
{"prompt": [{"role": "user", "content": "Give me advice about..."}], "action": "some context"}
```

---

## Output Configuration

Where trained models and checkpoints are stored. All paths must be S3 URIs.

```yaml
output:
  s3_path: s3://my-bucket/output/my-experiment/
  checkpoint_s3_path: s3://my-bucket/checkpoints/my-experiment/   # Optional
```

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `s3_path` | `str` | *(required)* | Must start with `s3://` | Final model output location |
| `checkpoint_s3_path` | `str \| null` | `null` | — | Mid-run checkpoint URI. Homogeneous jobs: SageMaker two-way-syncs `/opt/ml/checkpoints`. Mixed `dedicated_vllm`: SageMaker `CheckpointConfig` is omitted; training ranks `aws s3 sync` the same URI. **Required** when `infra.use_spot_instances` is `true` (homogeneous only; see [Managed Spot Training](#managed-spot-training)). |

---

## Environment

Plain environment variables available to the training process. Use this for non-secret Genos client settings and other backend-agnostic configuration. The SageMaker launcher passes `config.environment` directly as the container environment.

```yaml
environment:
  GENOS_ENV: "e2e"
  GENOS_EXPERIENCE_ID: "182c28b1-..."
  GENOS_INTUIT_ORIGINATING_ASSETALIAS: "Example.team.service.myapp"
  GENOS_OFFLINE_JOB_ID: "<offline-job-id>"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| *(top-level `environment`)* | `dict[str, str]` | `{}` | Passed directly as the SageMaker container environment. All env vars (including NCCL/Torch tuning) belong here. |

**Warning:** Validation warns if any key in `environment` contains `SECRET`, `TOKEN`, `PASSWORD`, or `KEY` — use the `secrets` section for sensitive values instead.

---

## Container Configuration

Controls the SageMaker training container image, framework selection, and pip dependencies. The recommended approach is to use a **container recipe** — a named preset that resolves the correct image, framework version, and pip requirements for a model family.

### Container Recipes

Recipes eliminate manual version management. Set `container.recipe` and the framework resolves everything else:

```yaml
container:
  recipe: qwen3_5              # resolves framework_version, image_uri, and pip deps
```

| Recipe | Framework Version | Image | Pip Stack | Notes |
|--------|------------------|-------|-----------|-------|
| `qwen3` | `2.8.0` | SDK-resolved PyTorch DLC | TRL 0.28 + vLLM + Transformers 4.57+ | Qwen3-14B era; includes torchrun variant |
| `qwen3_5` | `2.9.0` | Pinned PyTorch 2.9.0 DLC (`pytorch-training:2.9.0-gpu-py312-cu130-...`) | TRL 1.3+ + vLLM 0.20+ + Transformers 5.6+ | Qwen3.5-27B era; no torchrun variant |

**Override recipe defaults** — any explicit field takes precedence over the recipe:

```yaml
container:
  recipe: qwen3_5
  image_uri: "your-ecr-repo/custom-image:latest"   # overrides recipe image
  requirements:                                      # overrides recipe pip deps
    - "trl>=1.3.0"
    - "my-custom-package>=1.0"
```

**Resolution priority:**
1. Explicit `image_uri` → used as-is (no SDK lookup, recipe image ignored)
2. Explicit `requirements` → override recipe pip deps entirely
3. Explicit `framework_version` → override recipe version (validation warns if it differs from the recipe default)
4. No recipe set → `framework_version` defaults to `2.8.0`; pip deps come from the static `requirements.txt` shipped with the container scripts

**Adding a new recipe:** Add an entry to `CONTAINER_RECIPES` in `foresightrlcore/config/recipes.py`, add a test in `tests/test_config_recipes.py`, run `uv run tox`.

### Container Fields Reference

```yaml
container:
  recipe: qwen3_5              # Named recipe (recommended)
  # framework_version: "2.8.0" # Manual override (used when no recipe is set)
  # framework: pytorch          # "pytorch" (default) or "huggingface"
  # image_uri: "..."           # Custom ECR image (overrides everything)
  # requirements:              # Inline pip requirements (overrides recipe requirements)
  #   - "trl>=1.3.0"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `recipe` | `str \| null` | `null` | Named container recipe. Resolves `framework_version`, `image_uri`, and pip requirements automatically. |
| `framework_version` | `str` | `"2.8.0"` (or from recipe) | DLC image version (e.g. PyTorch 2.8.0) |
| `framework` | `str` | `"pytorch"` | DLC framework family: `pytorch` or `huggingface`. Controls which AWS Deep Learning Container is resolved. |
| `image_uri` | `str \| null` | `null` | Fully-qualified container image URI. When set, overrides `framework` + `framework_version` resolution entirely. |
| `requirements` | `list[str] \| null` | `null` | Inline pip requirements. When set, overrides recipe requirements. |

### Image Resolution

The image used for the training container is determined in this order:

1. If `image_uri` is set explicitly (or resolved from a recipe like `qwen3_5`), it is used as-is.
2. Otherwise, the SageMaker SDK resolves the image via `image_uris.retrieve(framework=<framework>, version=<framework_version>, ...)`.

### Custom Container Images

For advanced use cases (custom CUDA builds, pre-baked dependencies, private base images), set `image_uri` to your fully-qualified ECR URI:

```yaml
container:
  image_uri: "123456789.dkr.ecr.us-west-2.amazonaws.com/my-training:latest"
```

The image must be compatible with SageMaker Training (expected entry point at `/opt/ml/code/`, standard SageMaker environment variables). The framework's `entrypoint.sh` and `requirements.txt` are still uploaded as the source directory.

**Framework families** (when not using a custom `image_uri`):

| Framework | Image Pattern | Pre-installed Libraries | When to Use |
|-----------|---------------|------------------------|-------------|
| `pytorch` | `763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:<version>-gpu-...` | PyTorch, CUDA, NCCL | Default — RL training with TRL (dependencies installed at startup) |
| `huggingface` | `763104351884.dkr.ecr.<region>.amazonaws.com/huggingface-pytorch-training:<version>-transformers<ver>-gpu-...` | PyTorch + Transformers + Datasets + Tokenizers pre-installed | Faster startup when using HuggingFace ecosystem; classification fine-tuning |

**Finding available DLC images:**

AWS publishes all Deep Learning Container images in the [AWS DLC GitHub repository](https://github.com/aws/deep-learning-containers/blob/master/available_images.md). To list available images programmatically:

```python
from sagemaker import image_uris

# PyTorch training images
image_uris.retrieve("pytorch", "us-west-2", version="2.8.0",
                    instance_type="ml.p4d.24xlarge", image_scope="training")

# HuggingFace training images
image_uris.retrieve("huggingface", "us-west-2", version="2.8.0",
                    instance_type="ml.p4d.24xlarge", image_scope="training")
```

### Pip Dependency Resolution (Inside the Container)

When a SageMaker job starts, `entrypoint.sh` installs dependencies from `requirements.txt` (or `requirements_torchrun.txt` for `vllm_mode: none`). The launcher writes this file at packaging time based on the resolution priority:

1. **Inline `requirements`** → written directly to `requirements.txt`
2. **Recipe `resolved_requirements`** → written to `requirements.txt` (and `resolved_requirements_torchrun` to `requirements_torchrun.txt` if present)
3. **Neither set** → the static `foresightrlcore/runtime/scripts/requirements.txt` is used as-is

---

## Secrets Management

Secrets are resolved **at worker runtime**, not from plaintext in SageMaker job metadata. For `provider: idps`, no secret values appear in S3 config uploads or CloudTrail logs.

```yaml
secrets:
  provider: idps
  idps:
    policy_id: "p-examplepolicy"
    environment: e2e                # e2e | prd
  entries:
    - env_var: GENOS_APP_SECRET
      secret_path: "app_secret"
```

**Local provider** (dotenv-style `.env` file — for local GPU runs where IDPS is not available; same `entries` shape, `secret_path` names a key in the file):

```yaml
secrets:
  provider: local
  local:
    env_file: "secrets/e2e.env"
  entries:
    - env_var: GENOS_APP_SECRET
      secret_path: "app_secret"
```

### How It Works

1. The YAML config stores only secret **paths** (never inline secret values)
2. At worker startup, `resolve_secrets.py` runs before training begins
3. **`idps`:** connects via `idps-sdk` and fetches each secret by path. **`local`:** reads `KEY=VALUE` lines from `secrets.local.env_file` and maps `secret_path` to keys in that file
4. Secrets are exported as environment variables that the training script reads

### Secrets Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `provider` | `str` | `"idps"` | Secret resolution provider: `idps` or `local` |
| `idps.policy_id` | `str` | *(required when `provider: idps`)* | IDPS policy identifier |
| `idps.environment` | `str` | `"e2e"` | IDPS environment (`e2e` or `prd`) — maps to VKM host |
| `idps.endpoint` | `str \| null` | `null` | Override the auto-resolved IDPS endpoint |
| `local.env_file` | `str` | *(required when `provider: local`)* | Path to a `.env` file (`KEY=VALUE` per line) |
| `entries[].env_var` | `str` | — | Environment variable name to set (valid shell identifier) |
| `entries[].secret_path` | `str` | — | Logical secret path: IDPS path when using `idps`, or key name in the env file when using `local` |

### Validation Rules

- If `entries` is non-empty, `provider` is required
- For `idps` provider, `secrets.idps` with non-empty `policy_id` is required
- For `local` provider, `secrets.local.env_file` must be set
- No duplicate `env_var` names allowed
- Each `secret_path` must be non-empty

---

## Task Configuration

Tasks provide domain-specific logic: reward functions, callbacks, and dataset loaders.

```yaml
task:
  type: advice                     # Task module name (required)
  config:                           # Opaque per-task configuration
    reward_mode: llm_judge
    llm_clients:
      default:
        provider: genos             # bedrock | genos
        genos:
          model_id: anthropic.claude-opus-4-5-20251101-v1-0
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `str` | *(required)* | Task module name or path |
| `config` | `dict` | `{}` | Task-specific configuration (not validated by core schema) |

### Task Resolution Order

When you set `task.type`, the loader resolves the module in this order:

1. **File path** — absolute path or `.py` suffix: `task.type: /opt/ml/code/my_task.py`
2. **Dotted Python path** — `task.type: my_package.tasks.custom`
3. **Built-in** — `foresightrlcore.tasks.<name>`
4. **Runtime fallback** — `tasks.<name>` (SageMaker container `sys.path`)

### Advice Task

The built-in `advice` task supports LLM-as-judge evaluation and mock modes.

**LLM Judge mode (Genos provider):**

```yaml
task:
  type: advice
  config:
    reward_mode: llm_judge
    judge_parallel_workers: 8
    think_length_soft_cap: 500
    think_length_hard_cap: 2000
    think_penalty_weight: 0.3
    predictions_log: logs/grpo_advice_predictions.jsonl
    llm_clients:
      default:
        provider: genos
        genos:
          model_id: anthropic.claude-opus-4-5-20251101-v1-0
```

**LLM Judge mode (Bedrock provider — no `secrets:` block needed):**

```yaml
task:
  type: advice
  config:
    reward_mode: llm_judge
    judge_parallel_workers: 8
    llm_clients:
      default:
        provider: bedrock
        bedrock:
          model_id: us.anthropic.claude-sonnet-4-5-20250929-v1:0
          region: us-west-2
```

Bedrock auth uses the SageMaker execution role; ensure it has
`bedrock:InvokeModel` (and `bedrock:InvokeModelWithResponseStream` if
streaming is later enabled) on the target model ARNs in the configured
region. See [LLM Client Configuration](tasks/llm-client-config.md) for
the full reusability contract, single- vs multi-client patterns, and
the provider comparison.

> **Migration note.** The flat `critic_model_id` field has been
> removed, and the prototype `llm_judge:` block has been renamed to
> `llm_clients.default`. Both old shapes are rejected at task-load time
> with a clear migration error.

**Mock mode (for pipeline testing):**

```yaml
task:
  type: advice
  config:
    reward_mode: mock
    mock_constant_reward: 0.5
    mock_sleep_seconds: 0
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `reward_mode` | `str` | `"llm_judge"` | `llm_judge` for real evaluation, `mock` for testing |
| `llm_clients` | `dict` | *(required for `llm_judge`)* | Mapping of named client configs (see [llm-client-config.md](tasks/llm-client-config.md)). Use the name `default` for single-client tasks. |
| `judge_parallel_workers` | `int` | — | Thread pool size for parallel LLM evaluation |
| `think_length_soft_cap` | `int` | — | Soft cap on thinking token length (penalty starts here) |
| `think_length_hard_cap` | `int` | — | Hard cap on thinking token length (max penalty) |
| `think_penalty_weight` | `float` | — | Weight of the thinking-length penalty in the reward |
| `dimension_weights` | `dict` | — | Weights for individual quality scoring dimensions |
| `predictions_log` | `str` | `"logs/grpo_advice_predictions.jsonl"` | Path for logging prediction details |
| `mock_constant_reward` | `float` | `0.5` | Fixed reward value in mock mode |
| `mock_sleep_seconds` | `float` | `0` | Simulated latency in mock mode |
| `mock_log_message` | `str` | — | Custom log message in mock mode |

For creating custom tasks, see [Creating a Task](tasks/creating-a-task.md). For multi-reward patterns, see [Multi-Reward Functions](tasks/multi-reward-functions.md). For training hooks, see [Training Hooks](tasks/training-hooks.md).

---

## CLI Reference

The `rl-train` CLI provides two commands:

### `rl-train validate`

Validates a config file locally without launching anything.

```bash
rl-train validate --config job.yaml
```

Output on success:
```
Config is valid: job.yaml
  Job:       grpo-advice-run
  Model:     Qwen/Qwen3-0.6B
  Method:    grpo
  Task:      advice
  Execution: aws_training_job
```

### `rl-train train`

Validates and launches a training job.

```bash
AWS_DEFAULT_REGION=us-west-2 rl-train train --config job.yaml
```

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | Yes | Path to the YAML config file |
| `--wait` | No | Block until the job completes (default: return immediately) |

### `rl-train setup-local-gpu`

Generates a pip requirements file for local GPU machines from a named recipe. Not used for SageMaker jobs (container recipes handle this automatically). See [Local GPU Training — Environment Setup](launching-local-job.md#environment-setup-with-recipes).

### Version

```bash
rl-train --version
```

---

## Python API

For programmatic use, the `train()` function accepts a config path and optional overrides:

```python
from foresightrlcore import train

# Basic launch
job_id = train("job.yaml")

# With overrides
job_id = train(
    "job.yaml",
    model_name="Qwen/Qwen3-8B",
    instance_type="ml.p5.48xlarge",
    train_data="s3://my-bucket/new-data/",
    output_path="s3://my-bucket/new-output/",
)

# Wait for completion
job_id = train("job.yaml", wait=True)
```

### Shortcut Override Keys

| Kwarg | Maps To |
|-------|---------|
| `model_name` | `model.name` |
| `instance_type` | `infra.instance_type` |
| `train_data` | `data.train_path` |
| `eval_data` | `data.eval_path` |
| `output_path` | `output.s3_path` |

### Dotted Key Overrides

Any key containing `.` sets nested fields directly:

```python
job_id = train("job.yaml", **{"training.max_steps": 10, "training.grpo.temperature": 0.8})
```

---

## Validation

The framework validates at multiple levels before any resources are provisioned:

| Level | What It Checks |
|-------|----------------|
| **Pydantic schema** | Types, ranges, required fields, defaults, format constraints |
| **Cross-field validation** (`validation.py`) | Infra / method / backend compatibility (see below) |
| **Backend validation** | Backend-specific requirements (S3 paths, SDK installed) |

### Cross-Field Validation Rules

| Rule | Effect |
|------|--------|
| `topology: dedicated_vllm` | Requires `instance_count ≥ 2` |
| `infra.vllm_instance_type` set | Only `topology: dedicated_vllm` + `aws_training_job`; `vllm_instance_count` currently must be `1`; `instance_count > vllm_instance_count`; spot/warm-pool rejected; GPU family/EFA must match unless `allow_mixed_gpu_families: true`; VPC + any EFA type rejected |
| `infra.vllm.tensor_parallel_size` (mixed) | Must be ≤ GPUs on `vllm_instance_type`, not the training type |
| `method: grpo` + `vllm_mode: none` | **Warning** — native HF generation is significantly slower |
| `method: sft/dpo` + `vllm_mode != none` | **Warning** — vLLM is unnecessary for SFT/DPO |
| `execution: aws_training_job` | `data.train_path` and `output.s3_path` must start with `s3://` |
| `container.recipe` + `image_uri` | **Warning** — `image_uri` takes precedence; recipe image is ignored |
| `container.recipe` + `requirements` | **Warning** — inline requirements override recipe requirements |
| `container.recipe` + different `framework_version` | **Warning** — explicit version overrides recipe version |
| `environment` keys | **Warning** if any key looks like a secret name |
| Secrets with non-empty entries | Provider-specific config must be complete (`idps` or `local`); no duplicate env vars |
| `grpo.loss_type` | Must be one of: `grpo`, `dr_grpo`, `dapo`, `bnpo`, `cispo`, `sapo`, `luspo`, `vespo` |
| `grpo.scale_rewards` (string) | Must be `"group"`, `"batch"`, or `"none"` (bools also accepted) |
| `grpo.multi_objective_aggregation` | Must be `"sum_then_normalize"` or `"normalize_then_sum"` |
| `grpo.vllm_importance_sampling_mode` | Must be `token_truncate`, `token_mask`, `sequence_truncate`, or `sequence_mask` |
| `grpo.sync_ref_model: true` | Requires `beta > 0` (reference model is only loaded when beta > 0) |
| `grpo.loss_type: luspo` | Requires `importance_sampling_level: sequence` |
| `bf16: true` + `fp16: true` | **Error** — cannot enable both simultaneously |

---

## Container Pipeline

What happens inside the SageMaker container after the job starts:

```
entrypoint.sh
    |
    +---> Install dependencies (requirements.txt)
    |
    +---> Resolve secrets (if secrets: section present)
    |       +---> resolve_secrets.py -> idps or local .env -> export env vars
    |
    +---> Dispatch based on vllm_mode + topology:
            |
            +-- vllm_mode: none
            |     +---> torchrun -> train.py
            |
            +-- vllm_mode: colocate
            |     +---> accelerate launch -> train.py (vLLM in-process)
            |
            +-- vllm_mode: server + single_node
            |     +---> Start vLLM server -> accelerate launch -> train.py
            |
            +-- vllm_mode: server + dedicated_vllm
            |     +---> Homogeneous: node 0 vLLM (all GPUs); node 1+ train
            |     +---> Mixed InstanceGroups: SM_CURRENT_INSTANCE_GROUP
            |           (vllm vs training); host lists from instance_groups_dict
            |
            +-- vllm_mode: server + multi_node_server
                  +---> Node 0: vLLM split + training
                  +---> Node 1+: training only
```

### Runtime Scripts

| Script | Purpose |
|--------|---------|
| `entrypoint.sh` | GPU topology orchestration, dependency installation, vLLM/training dispatch |
| `resolve_secrets.py` | Resolves secrets (`idps` or `local` .env) and exports as environment variables |
| `train.py` | Training driver — loads config, initializes task, builds trainer, runs training |
| `instance_group_hosts.py` | Resolves SageMaker InstanceGroup host lists from `SM_TRAINING_ENV` / `resourceconfig.json` |
| `nccl_env.py` | Configures NCCL environment variables for multi-node communication |

---

## Full Config Example

A complete annotated config for GRPO training on SageMaker with LLM-as-judge evaluation:

```yaml
version: 1

job:
  name: grpo-advice-qwen3
  max_runtime_hours: 24

model:
  name: Qwen/Qwen3-0.6B
  enable_thinking: true
  lora:
    r: 32
    alpha: 64
    dropout: 0.05
    target_modules: [q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj]

method:
  type: grpo

execution:
  type: aws_training_job

infra:
  topology: single_node
  vllm_mode: server
  instance_type: ml.p4d.24xlarge
  instance_count: 1
  vllm_gpu_count: 2
  training_gpu_count: 6
  volume_size_gb: 100
  vllm:
    port: 8002
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.9

training:
  epochs: 1
  per_device_batch_size: 4
  gradient_accumulation_steps: 2
  learning_rate: 5e-5
  lr_scheduler: cosine
  warmup_ratio: 0.03
  bf16: true
  gradient_checkpointing: true
  deepspeed: ds_zero2_config.json
  logging_steps: 5
  save_steps: 50
  grpo:
    num_generations: 8
    max_completion_length: 8192
    beta: 0.001
    temperature: 0.6
    top_p: 0.95
    top_k: 20
    loss_type: dr_grpo

data:
  train_path: s3://my-bucket/datasets/advice/train/
  eval_split_size: 100

output:
  s3_path: s3://my-bucket/output/grpo-advice-qwen3/
  # checkpoint_s3_path: s3://my-bucket/checkpoints/grpo-advice-qwen3/  # live mid-run checkpoints

environment:
  GENOS_ENV: "e2e"
  GENOS_EXPERIENCE_ID: "your-experience-id"
  GENOS_INTUIT_ORIGINATING_ASSETALIAS: "your-asset-alias"
  GENOS_OFFLINE_JOB_ID: "your-job-id"

container:
  recipe: qwen3_5                # Named recipe — resolves image, framework_version, pip deps
  # framework_version: "2.8.0"  # Manual override when not using a recipe
  # image_uri: "..."            # Custom ECR image (overrides recipe + framework resolution)

secrets:
  provider: idps
  idps:
    policy_id: "your-policy-id"
    environment: e2e
  entries:
    - env_var: GENOS_APP_SECRET
      secret_path: "app_secret"

task:
  type: advice
  config:
    reward_mode: llm_judge
    judge_parallel_workers: 8
    llm_clients:
      default:
        provider: genos
        genos:
          model_id: anthropic.claude-opus-4-5-20251101-v1-0
```

**Smoke test pattern** — add to YAML for quick pipeline validation:

```yaml
training:
  max_steps: 10
task:
  config:
    reward_mode: mock
    mock_constant_reward: 0.5
    mock_sleep_seconds: 0
```
