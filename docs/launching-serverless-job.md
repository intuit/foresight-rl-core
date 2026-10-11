# Launching a Serverless Model Customization Job

Complete reference for configuring and launching SageMaker Serverless Model Customization jobs with `rl-train`. Serverless training requires no infrastructure management — no instance types, topology, vLLM, DeepSpeed, or custom containers. AWS manages the training loop; you configure and submit.

---

## Table of Contents

- [Quick Start](#quick-start)
- [What is Serverless Model Customization](#what-is-serverless-model-customization)
- [Supported Methods](#supported-methods)
- [Config Structure](#config-structure)
- [Job Identity](#job-identity)
- [Model Selection](#model-selection)
- [Execution Backend](#execution-backend)
  - [Core Settings](#core-settings)
  - [Dataset Management](#dataset-management)
  - [Model Package Group](#model-package-group)
  - [RLAIF-Specific Settings](#rlaif-specific-settings)
  - [RLVR-Specific Settings](#rlvr-specific-settings)
  - [MLflow Tracking](#mlflow-tracking)
  - [Job Configuration](#job-configuration)
  - [VPC / Networking](#vpc--networking)
- [Training Hyperparameters](#training-hyperparameters)
- [Output Configuration](#output-configuration)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [Validation](#validation)
- [Full Config Examples](#full-config-examples)
  - [DPO Example](#dpo-example)
  - [RLVR Example](#rlvr-example)
  - [RLAIF Example](#rlaif-example)

---

> **First time on this backend?** Complete the one-time account setup before launching:
> [intuit/docs/account-setup/aws/sagemaker-serverless.md](../intuit/docs/account-setup/aws/sagemaker-serverless.md) — execution role, AI Registry access, EULA acceptance, optional MLflow server. The steps below assume those exist.

## Quick Start

```bash
# 1. Install (requires SageMaker SDK v3)
uv sync --all-extras --all-groups  # or: uv pip install -e ".[aws]"

# 2. Copy a template and customize
cp examples/configs/sagemaker_serverless_example.yaml my_serverless_job.yaml

# 3. Validate (no GPU needed)
rl-train validate --config my_serverless_job.yaml

# 4. Launch
AWS_DEFAULT_REGION=us-west-2 rl-train train --config my_serverless_job.yaml
```

---

## What is Serverless Model Customization

SageMaker Serverless Model Customization (via SDK v3) lets you fine-tune foundation models without managing any infrastructure. Compared to SageMaker Training Jobs (`aws_training_job`):

| Aspect | SageMaker Training Job | Serverless |
|--------|----------------------|------------|
| Infrastructure | You choose instance types, GPU splits, topology, vLLM mode | Fully managed by AWS |
| Container | You configure framework, image, DeepSpeed | No container management |
| Methods | Any (GRPO, SFT, DPO via TRL) | DPO, RLVR, RLAIF (AWS-managed trainers) |
| Datasets | S3 paths (JSONL in directories) | AI Registry dataset names (get-or-create) |
| Hyperparameters | Full TRL control (35+ GRPO params) | Simplified (`global_batch_size`, `max_epochs`, `learning_rate`) |
| Model versioning | Manual S3 output | Automatic via Model Package Groups |
| Tracking | TensorBoard / custom | MLflow integration built-in |

Use serverless when you want simplicity and don't need fine-grained control over the training infrastructure.

> For infrastructure-managed training, see [Launching a SageMaker Training Job](launching-sagemaker-training-job.md). For local GPU, see [Launching a Local GPU Job](launching-local-job.md).

---

## Supported Methods

| Method | SDK Trainer | Description |
|--------|-------------|-------------|
| `dpo` | `sagemaker.train.dpo_trainer.DPOTrainer` | Direct Preference Optimization |
| `rlvr` | `sagemaker.train.rlvr_trainer.RLVRTrainer` | RL with Verifiable Rewards (custom reward function) |
| `rlaif` | `sagemaker.train.rlaif_trainer.RLAIFTrainer` | RL from AI Feedback (reward model + prompt) |

Each method maps to a dedicated SageMaker SDK v3 trainer class. The backend resolves the trainer automatically from `method.type`.

---

## Config Structure

A serverless config is simpler than a SageMaker Training Job config. No `infra`, `container`, `secrets`, or `task` sections are needed:

```yaml
version: 1

job:        { ... }               # Job identity
model:      { ... }               # Model name (informational)
method:     { ... }               # dpo | rlvr | rlaif
execution:  { ... }               # type: aws_serverless + serverless settings
training:   { ... }               # serverless hyperparameters
output:     { ... }               # S3 output path (optional)
```

The bulk of the configuration lives under `execution.serverless`.

---

## Job Identity

```yaml
job:
  name: serverless-dpo-qwen25
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *(required)* | Human-readable job identifier |

---

## Model Selection

```yaml
model:
  name: huggingface-llm-qwen2-5-7b-instruct
```

The `model.name` field is informational for tracking. The actual model used for training is specified in `execution.serverless.model_id` (the SageMaker Hub model ID).

---

## Execution Backend

```yaml
execution:
  type: aws_serverless
  serverless:
    model_id: huggingface-llm-qwen2-5-7b-instruct
    training_type: lora
    accept_eula: true
    # ... (dataset, MPG, method-specific settings)
```

### Core Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_id` | `str` | *(required)* | SageMaker Hub model ID (e.g. `huggingface-llm-qwen2-5-7b-instruct`) |
| `training_type` | `str` | `"lora"` | Training type: `lora` or `full` |
| `accept_eula` | `bool` | `false` | Accept the model EULA (**must be `true`** to run) |

### Dataset Management

Datasets use **get-or-create** semantics via the AI Registry. If a dataset with the given name already exists, it is reused. If not, and a source is provided, it is created automatically.

```yaml
execution:
  serverless:
    training_dataset_name: my-train-dataset
    training_dataset_source: "s3://bucket/path/train.jsonl"   # Optional: auto-create
    # training_dataset_source: "./data/train.jsonl"           # Local files also supported

    validation_dataset_name: my-val-dataset                   # Optional
    # validation_dataset_source: "s3://bucket/path/val.jsonl"

    test_dataset_name: my-test-dataset                        # Optional
    # test_dataset_source: "s3://bucket/path/test.jsonl"

    customization_technique: dpo                              # Optional: hint for DataSet.create
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `training_dataset_name` | `str \| null` | `null` | AI Registry dataset name. Either this or `data.train_path` must be set. |
| `training_dataset_source` | `str \| null` | `null` | S3 URI or local file path for auto-creating the dataset |
| `validation_dataset_name` | `str \| null` | `null` | AI Registry validation dataset name |
| `validation_dataset_source` | `str \| null` | `null` | Source for auto-creating the validation dataset |
| `test_dataset_name` | `str \| null` | `null` | AI Registry test dataset name (for evaluation) |
| `test_dataset_source` | `str \| null` | `null` | Source for auto-creating the test dataset |
| `customization_technique` | `str \| null` | `null` | Technique hint for `DataSet.create` (e.g. `dpo`). If omitted, derived from `method.type`. |

**How dataset resolution works:**

1. `DataSet.get(name=...)` — tries to find an existing dataset in AI Registry
2. If not found and `source` is provided — `DataSet.create(name=..., source=..., customization_technique=...)` creates it
3. If not found and no source — raises an error

### Model Package Group

Automatic model versioning via SageMaker Model Package Groups.

```yaml
execution:
  serverless:
    model_package_group: my-model-package-group
    model_package_group_description: "Fine-tuned Qwen2.5-7B for my use case"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_package_group` | `str \| null` | `null` | ModelPackageGroup name. Get-or-create semantics. |
| `model_package_group_description` | `str \| null` | `null` | Description used when auto-creating the group |

### RLAIF-Specific Settings

RLAIF requires a reward model and a reward prompt evaluator.

```yaml
method:
  type: rlaif

execution:
  serverless:
    reward_model_id: "openai.gpt-oss-120b-1:0"
    reward_prompt_name: my-reward-prompt

    # Option A: reference an existing evaluator (by name above)
    # Option B: auto-create from S3 or local file
    reward_prompt_source: "s3://bucket/path/reward_prompt.txt"
    # Option C: inline text (written to temp file, registered as evaluator)
    # reward_prompt_text: |
    #   You are an expert evaluator. Given a prompt and response, rate 1-10.
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `reward_model_id` | `str` | *(required for RLAIF)* | Reward model ID (e.g. `openai.gpt-oss-120b-1:0`) |
| `reward_prompt_name` | `str` | *(required for RLAIF)* | AI Registry evaluator name for the reward prompt |
| `reward_prompt_source` | `str \| null` | `null` | S3 URI or local file path for auto-creating the evaluator |
| `reward_prompt_text` | `str \| null` | `null` | Inline reward prompt text. Written to a temp file and registered as an evaluator. |

**Reward prompt resolution order:**

1. `Evaluator.get(name=reward_prompt_name)` — tries existing evaluator
2. If not found, creates from `reward_prompt_source` (S3/local path)
3. If no source but `reward_prompt_text` is set, writes text to a temp file and creates from that

### RLVR-Specific Settings

RLVR uses a custom reward function Lambda registered as an AI Registry
Evaluator. The framework auto-creates the evaluator the first time a
job runs against an unfamiliar name.

```yaml
method:
  type: rlvr

execution:
  serverless:
    custom_reward_function_name: example-reward-fn                 # AI Registry name AND registry key
    custom_reward_function_role: "arn:aws:iam::<acct>:role/<role>"  # Lambda execution role
    # custom_reward_function_source: "./my_team/lambda.py"          # OPTIONAL — escape hatch
    # custom_reward_function_extra_files: ["./my_team/util.py"]     # OPTIONAL — multi-file zip
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `custom_reward_function_name`        | `str \| null` | `null` | AI Registry evaluator name. Doubles as the in-package registry key (see [`docs/serverless-reward-functions.md`](serverless-reward-functions.md)). |
| `custom_reward_function_role`        | `str \| null` | `null` | IAM role ARN for the Lambda execution role (required for first-time creation). |
| `custom_reward_function_source`      | `str \| null` | `null` | Explicit Lambda `.py` file path or S3 URI. Overrides the registry lookup. |
| `custom_reward_function_extra_files` | `list[str]`   | `[]`   | Additional files bundled at the zip root alongside the main source. Only valid with `_source`. |

Resolution order at launch time:

1. **`Evaluator.get(name=...)`** — if an AI Registry evaluator with
   that name already exists, use it.
2. **`custom_reward_function_source` set** — create the evaluator from
   that file (multi-file zip via `_extra_files` if any).
3. **Look up the registry** — `foresightrlcore.serverless.registry.LAMBDAS[name]`
   supplies the main module + extra modules + env defaults; the
   framework zips them and creates the evaluator.

When both `_source` and a registry entry exist, `_source` wins
(explicit beats implicit) and a warning is logged.

For the per-task folder layout, the bundler (`rl-train serverless
bundle`), the generic JSONL converter (`rl-train serverless convert`),
and the schema-pluggable converter contract, see
[`docs/serverless-reward-functions.md`](serverless-reward-functions.md).

### MLflow Tracking

Built-in MLflow integration for experiment tracking.

```yaml
execution:
  serverless:
    mlflow_experiment_name: serverless-dpo-experiment
    mlflow_run_name: run-001
    mlflow_resource_arn: "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/my-server"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mlflow_experiment_name` | `str \| null` | `null` | MLflow experiment name |
| `mlflow_run_name` | `str \| null` | `null` | MLflow run name within the experiment |
| `mlflow_resource_arn` | `str \| null` | `null` | MLflow tracking server ARN. If unset, uses the default MLflow experience. |

### Job Configuration

```yaml
execution:
  serverless:
    base_job_name: custom-job-prefix
    kms_key_id: "arn:aws:kms:us-west-2:123456789012:key/my-key-id"
    max_runtime_seconds: 86400
    wait_timeout: 86400
    tags:
      team: my-team
      project: my-project
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `base_job_name` | `str \| null` | `null` | Custom prefix for the SageMaker job name. If unset, derived from model_id + method. |
| `kms_key_id` | `str \| null` | `null` | KMS key ID for encrypting training job outputs |
| `encrypt_inter_container_traffic` | `bool` | `true` | Encrypt traffic between the compute instances backing the job (TLS). Maps to SageMaker `EnableInterContainerTrafficEncryption`. **On by default** to satisfy Intuit ACP policy `IS031.1`. See [In-transit encryption](#in-transit-encryption) below. |
| `max_runtime_seconds` | `int \| null` | `null` | Max training runtime in seconds. Default is SageMaker service limit (24h). |
| `wait_timeout` | `int \| null` | `null` | Max seconds to wait for job completion when `wait=True` |
| `tags` | `dict[str, str] \| null` | `null` | AWS resource tags applied to the training job |

#### In-transit encryption

The SDK v3 serverless trainers build their `TrainingJob.create(...)` arguments
from a fixed dict inside `train()` and expose no constructor parameter for
`enable_inter_container_traffic_encryption` — their `**kwargs` are consumed by
`BaseTrainer`. `TrainingJob.create` itself accepts the parameter, so the
framework wraps that call for the duration of the submit
(`serverless_inter_container_encryption` in
`foresightrlcore/execution/sagemaker_common.py`, the same module the
training-job launchers use for the raw boto3 path).

The wrapper is fail-fast: if a future SDK version drops the parameter from
`TrainingJob.create`, launch raises `RuntimeError` rather than silently
submitting an unencrypted job. Setting `encrypt_inter_container_traffic: false`
skips the wrapping entirely and restores stock SDK behaviour — use it as the
escape hatch if AWS rejects the parameter for serverless jobs.

### VPC / Networking

```yaml
execution:
  serverless:
    networking:
      subnets: ["subnet-abc123"]
      security_groups: ["sg-xyz000"]
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `networking.subnets` | `list[str]` | — | VPC subnet IDs |
| `networking.security_groups` | `list[str]` | — | Security group IDs |

---

## Training Hyperparameters

Serverless training exposes a simplified set of hyperparameters under `training.serverless`:

```yaml
training:
  serverless:
    global_batch_size: 64
    max_epochs: 3
    learning_rate: 5.0e-05
    extra:
      custom_param: value
```

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `global_batch_size` | `int \| null` | `null` | `≥ 1` | Total batch size across all workers |
| `max_epochs` | `int \| null` | `null` | `≥ 1` | Number of training epochs |
| `learning_rate` | `float \| null` | `null` | `> 0` | Learning rate |
| `extra` | `dict[str, Any]` | `{}` | — | Pass-through hyperparameters. Any key/value pair set here is applied directly to the trainer's hyperparameters object via `setattr`. |

---

## Output Configuration

```yaml
output:
  s3_path: "s3://your-bucket/output/serverless-example"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `s3_path` | `str \| null` | `null` | S3 output path. Optional for serverless (model versioning via MPG is the primary output mechanism). |

---

## CLI Reference

### Validate

```bash
rl-train validate --config my_serverless_job.yaml
```

### Launch

```bash
# Fire-and-forget
AWS_DEFAULT_REGION=us-west-2 rl-train train --config my_serverless_job.yaml

# Block until complete
AWS_DEFAULT_REGION=us-west-2 rl-train train --config my_serverless_job.yaml --wait
```

> **Always set `AWS_DEFAULT_REGION=us-west-2`** — the local AWS config has no default region. The serverless backend also sets this automatically if missing, but explicit is better.

---

## Python API

```python
import os
os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
from foresightrlcore import train

job_id = train("my_serverless_job.yaml")

# Wait for completion
job_id = train("my_serverless_job.yaml", wait=True)
```

---

## Validation

Serverless validation checks different things than SageMaker Training Job validation:

| Rule | Effect |
|------|--------|
| `execution.serverless` section required | Error if missing when `execution.type: aws_serverless` |
| `accept_eula: true` required | Error if not accepted |
| Dataset or train_path required | Either `training_dataset_name` or `data.train_path` must be set |
| RLAIF requires reward config | `reward_model_id` and `reward_prompt_name` are required |
| SageMaker SDK v3 required | Error if SDK version < 3.0.0 |
| `method.type` must be supported | Only `dpo`, `rlvr`, `rlaif` are supported |
| Infrastructure checks skipped | Topology, vLLM mode, GPU allocation are not validated for serverless |

---

## Full Config Examples

### DPO Example

```yaml
version: 1

job:
  name: serverless-dpo-qwen25

model:
  name: huggingface-llm-qwen2-5-7b-instruct

method:
  type: dpo

execution:
  type: aws_serverless
  serverless:
    model_id: huggingface-llm-qwen2-5-7b-instruct
    training_type: lora
    accept_eula: true
    training_dataset_name: my-dpo-train-dataset
    training_dataset_source: "s3://my-bucket/data/dpo_train.jsonl"
    customization_technique: dpo
    model_package_group: my-dpo-models
    mlflow_experiment_name: dpo-experiments

training:
  serverless:
    global_batch_size: 64
    max_epochs: 3
    learning_rate: 5.0e-05

output:
  s3_path: "s3://my-bucket/output/serverless-dpo"
```

### RLVR Example

```yaml
version: 1

job:
  name: serverless-rlvr-qwen25

model:
  name: huggingface-llm-qwen2-5-7b-instruct

method:
  type: rlvr

execution:
  type: aws_serverless
  serverless:
    model_id: huggingface-llm-qwen2-5-7b-instruct
    training_type: lora
    accept_eula: true
    training_dataset_name: my-rlvr-train-dataset
    custom_reward_function_name: my-reward-evaluator
    model_package_group: my-rlvr-models
    mlflow_experiment_name: rlvr-experiments

training:
  serverless:
    global_batch_size: 64

output:
  s3_path: "s3://my-bucket/output/serverless-rlvr"
```

### RLAIF Example

```yaml
version: 1

job:
  name: serverless-rlaif-qwen25

model:
  name: huggingface-llm-qwen2-5-7b-instruct

method:
  type: rlaif

execution:
  type: aws_serverless
  serverless:
    model_id: huggingface-llm-qwen2-5-7b-instruct
    training_type: lora
    accept_eula: true
    training_dataset_name: my-rlaif-train-dataset
    reward_model_id: "openai.gpt-oss-120b-1:0"
    reward_prompt_name: my-reward-prompt
    reward_prompt_text: |
      You are an expert evaluator. Given a prompt and response,
      rate the quality of the response on a scale of 1-10.
    model_package_group: my-rlaif-models
    mlflow_experiment_name: rlaif-experiments

training:
  serverless:
    global_batch_size: 64
    max_epochs: 3

output:
  s3_path: "s3://my-bucket/output/serverless-rlaif"
```
