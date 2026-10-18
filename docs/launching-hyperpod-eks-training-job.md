# Launching a HyperPod EKS Training Job

Complete reference for configuring and launching GRPO training jobs on a
SageMaker HyperPod EKS cluster with `rl-train`. Every parameter lives in
a YAML config file — nothing is hardcoded.

This doc is the per-job launch reference. For the one-time
cluster-bootstrap setup (IAM roles, SSM tunnel, FSx provisioning, EKS
add-ons), see [HyperPod EKS Setup Notes](../intuit/docs/hyperpod-eks-setup-notes.md).

> **First time on this backend?** Start with the structured account-setup
> guide: [intuit/docs/account-setup/aws/hyperpod-eks.md](../intuit/docs/account-setup/aws/hyperpod-eks.md) —
> covers the three IAM roles, FSx, ECR, IDPS, IRSA, and the
> kubeconfig/SSM bastion pattern in account-setup form. The deep-dive
> operational notes in `intuit/docs/hyperpod-eks-setup-notes.md` remain the canonical
> reference for day-2 cluster operations.

---

## Table of Contents

- [Quick Start](#quick-start)
- [How HyperPod EKS Differs from SageMaker Training Jobs](#how-hyperpod-eks-differs-from-sagemaker-training-jobs)
- [Config Structure](#config-structure)
- [Execution Backend](#execution-backend)
  - [Cluster Identity](#cluster-identity)
  - [Training Image](#training-image)
  - [Pod IAM (IRSA)](#pod-iam-irsa)
  - [FSx for Lustre](#fsx-for-lustre)
  - [Restart Policy](#restart-policy)
  - [SSM Tunnel (Laptop Launches)](#ssm-tunnel-laptop-launches)
- [Topology & vLLM Mode](#topology--vllm-mode)
- [Container Configuration](#container-configuration)
- [Manifests Generated](#manifests-generated)
- [Validation](#validation)
- [CLI Reference](#cli-reference)
- [Debugging](#debugging)
- [Full Config Example](#full-config-example)

---

## Quick Start

```bash
# 1. Install
uv sync --all-extras --all-groups

# 2. Copy the example at the bottom of this page and fill in your
#    cluster IDs, image URI, and S3 paths.

# 3. Validate (no GPU, no cluster access needed)
rl-train validate --config my_job.yaml

# 4. Launch — auto-establishes the SSM tunnel + applies manifests
AWS_DEFAULT_REGION=us-east-1 rl-train train --config my_job.yaml
```

> **Region.** HyperPod clusters live in a specific region. Always set
> `AWS_DEFAULT_REGION` to the region the cluster was created in.

---

## How HyperPod EKS Differs from SageMaker Training Jobs

The two AWS backends share the YAML schema, but the runtime path is
different. Use this table to decide which backend fits a given job.

| Concern | `aws_training_job` | `aws_hyperpod_eks` |
|---|---|---|
| **Submission API** | SageMaker `CreateTrainingJob` (boto3 / SDK) | Kubernetes `apply -f` (kubectl → EKS API) |
| **Resource shape** | SageMaker `TrainingJob` resource | Kubeflow `PyTorchJob` (+ optional `Deployment` + `Service` for the vLLM split) |
| **Distribution** | `torchrun`-via-DLC entry or `accelerate launch` | `torchrun` + `elasticPolicy.rdzvBackend: c10d` (operator-injected `PET_*` env vars) |
| **Cluster provisioning** | Per-job (instances spun up + torn down) | Pre-provisioned (instances stay warm; jobs queue on them) |
| **Cold-start latency** | ~5–10 min per job | Seconds (no instance provisioning) |
| **Multi-week runs** | Limited (`max_runtime_hours` ≤ 120) | Native (`max_restarts`-based auto-resume via the Kubeflow operator) |
| **Model + dataset caching** | Per-job download to ephemeral disk | Shared FSx mount across all pods (set up once, reused everywhere) |
| **Pod IAM** | `RoleArn` on the TrainingJob | Service account → IRSA → trust policy |
| **Secrets** | SageMaker secrets-manager hookup or runtime IDPS | Runtime IDPS via the pod's IAM role |

**Choose HyperPod EKS when** you have a pre-provisioned cluster and
want fast launches + persistent shared FSx + long auto-resume windows.

**Choose SageMaker Training Jobs when** you want zero cluster ops, can
absorb 5–10 min per-job startup, and your jobs fit in the 120-hour cap.

For other backends, see [Launching a SageMaker Training Job](launching-sagemaker-training-job.md)
and [Launching a Local GPU Job](launching-local-job.md).

---

## Config Structure

A HyperPod EKS training job config is a single YAML file with the same
top-level sections as a SageMaker config:

```yaml
version: 1                # Config format version
job:        { ... }        # Job identity and runtime limits
model:      { ... }        # Base model and LoRA settings
method:     { ... }        # Training algorithm (grpo, sft, dpo)
execution:                 # Must be aws_hyperpod_eks
  type: aws_hyperpod_eks
  hyperpod_eks: { ... }    # HyperPod-specific (cluster IDs, FSx, SSM, ...)
infra:      { ... }        # Hardware, topology, vLLM
training:   { ... }        # Optimizer, schedule, GRPO hyperparameters
data:       { ... }        # Training data path (must be s3://)
output:     { ... }        # Output and checkpoint locations (s3://)
environment: { ... }      # Backend-agnostic env vars (e.g. Genos)
container:  { ... }        # Recipe (defaults to qwen3 for HyperPod EKS)
secrets:    { ... }        # Runtime secret resolution (local, or an extension provider)
task:       { ... }        # Task plugin (reward function, dataset loader)
```

Backend-agnostic sections — `job`, `model`, `method`, `training`,
`data`, `output`, `environment`, `secrets`, `task` — work identically
to the SageMaker path. See [the SageMaker doc](launching-sagemaker-training-job.md)
for full field references on those. The rest of this doc covers the
HyperPod EKS-specific bits.

---

## Execution Backend

```yaml
execution:
  type: aws_hyperpod_eks
  hyperpod_eks:
    cluster_name: "your-hyperpod-cluster"
    eks_cluster_name: "your-eks-cluster"
    namespace: default

    image_uri: "123456789012.dkr.ecr.us-east-1.amazonaws.com/your-training-image:qwen3-vllm-latest"
    service_account: default

    fsx_filesystem_id: "fs-0123456789abcdef0"
    fsx_mount_path: /fsx
    pvc_name: "fsx-claim"

    restart_policy: OnFailure
    max_restarts: 3

    # SSM tunnel — opens localhost:6443 → EKS API automatically.
    # Omit when launching from inside the cluster.
    ssm_relay_instance_id: i-0123456789abcdef0
    kubeconfig_path: kubeconfigs/your-cluster.yaml
```

| Field | Type | Default | Description |
|---|---|---|---|
| `cluster_name` | `str` | *(required)* | HyperPod cluster name (SageMaker control plane) |
| `eks_cluster_name` | `str` | *(required)* | Underlying EKS cluster name (data plane) |
| `namespace` | `str` | `"default"` | Kubernetes namespace for the PyTorchJob and pods |
| `image_uri` | `str` | *(required)* | ECR image with training code + recipe deps. See [Training Image](#training-image). |
| `service_account` | `str` | `"default"` | K8s service account (bound to a pod IAM role via IRSA) |
| `fsx_filesystem_id` | `str` | `""` | FSx for Lustre filesystem ID. Set this to enable shared model + dataset caching. |
| `fsx_mount_path` | `str` | `"/fsx"` | Where the FSx volume mounts inside each pod |
| `pvc_name` | `str` | `""` | Name of the FSx-backed `PersistentVolumeClaim` (created out-of-band; see setup doc) |
| `restart_policy` | `str` | `"OnFailure"` | Pod restart policy: `OnFailure` or `Never` |
| `max_restarts` | `int` | `3` | Max worker restarts before the PyTorchJob is marked failed |
| `ssm_relay_instance_id` | `str \| null` | `null` | EC2 instance ID of an SSM-managed bastion. Auto-opens a port-forwarding tunnel from the launching machine. |
| `kubeconfig_path` | `str \| null` | `null` | Kubeconfig file. Required when `ssm_relay_instance_id` is set. Relative paths are resolved against the working directory. The backend exports `KUBECONFIG` to this path on every kubectl/SSM call, and does not write to your home directory. |

### Cluster Identity

A HyperPod cluster has two names because SageMaker (which manages
queueing, resilience, and node groups) sits on top of an EKS cluster
(which actually runs the pods). Both are required: `cluster_name` is
used for SageMaker control-plane API calls; `eks_cluster_name` is used
to resolve the EKS API endpoint for kubectl.

### Training Image

The cluster pulls the training image from ECR. Two options:

1. **Public AWS Deep Learning Container** — point `image_uri` at a
   public DLC (`763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:...`).
   The container `entrypoint.sh` installs the recipe-resolved
   requirements at startup. Pro: no image build required. Con: ~3–5
   min startup per pod for the pip install. Not suitable when the job
   needs a package that is not on public PyPI.

2. **Custom ECR image** — bake the recipe dependencies into an image you
   push to your own registry. Pro: fast startup. Con: a large image push
   the first time. An internal build that also installs a private secrets
   client is documented in
   [intuit/docs/building-docker-images.md](../intuit/docs/building-docker-images.md).

### Pod IAM (IRSA)

The training pod's identity comes from the Kubernetes service account
bound to an IAM role via IRSA (IAM Roles for Service Accounts). The
service account → role mapping is configured at cluster setup time;
see the [setup doc](../intuit/docs/hyperpod-eks-setup-notes.md) for the trust policy.

When the job resolves secrets through an extension provider, that
provider's policy must trust this role.

### FSx for Lustre

Strongly recommended. Without an FSx mount, every training pod
re-downloads the model from Hugging Face Hub at startup, which:
- Adds ~5–15 min cold start per pod (model size dependent)
- Hits the HF Hub rate limiter at scale — we observed HTTP 429s on
  every pod simultaneously when launching 9-node runs.

With FSx:
- The model is downloaded once to a shared `HF_HUB_CACHE` path.
- All pods read the cache in parallel from FSx (no network egress).
- Same backing volume serves model + dataset + checkpoint paths.

See [FSx model cache in the setup doc](../intuit/docs/hyperpod-eks-setup-notes.md#shared-hf-model-cache-on-fsx)
for the one-time provisioning + how to pre-stage a model.

### Restart Policy

The Kubeflow PyTorchJob restarts failed worker pods up to
`max_restarts` times before giving up. `OnFailure` restarts only when
the pod's primary container exits non-zero; `Never` makes the worker
recover from a kernel/OOM crash impossible. **Always use `OnFailure`
for multi-day runs** — instance-level events (spot reclaim,
unscheduled reboots) will eventually happen.

### SSM Tunnel (Laptop Launches)

The EKS control plane is private (no public endpoint by default). When
launching from outside the cluster — i.e. from your laptop or any
machine that isn't already in the VPC — the launcher needs to reach
the EKS API server.

Set `ssm_relay_instance_id` to an SSM-managed bastion that *can* reach
the EKS API. The launcher auto-opens a port-forwarding session
(`aws ssm start-session --document-name AWS-StartPortForwardingSessionToRemoteHost`)
that tunnels `localhost:6443` → `<eks-api>:443` via the bastion, then
sets `kubectl` to use `https://localhost:6443`. The tunnel is torn down
after the launch completes (or after the job finishes if `wait=True`).

`kubeconfig_path` must point at a kubeconfig wired for this idiom —
`server: https://localhost:6443` and `tls-server-name: <real EKS
endpoint>`.

```yaml
kubeconfig_path: kubeconfigs/your-cluster.yaml
```

The backend exports `KUBECONFIG=<this path>` on every `kubectl` / `aws
ssm` invocation. Your home directory is not touched. A worked example
for one internal cluster, including the kubeconfig file, is in
[`intuit/docs/hyperpod-eks-setup-notes.md`](../intuit/docs/hyperpod-eks-setup-notes.md).

If you want to run `kubectl` from a regular shell while the job is
running (e.g. ad-hoc log tails), it's convenient — *but optional* — to
copy the same file into your home directory once:

```bash
mkdir -p ~/.kube
cp kubeconfigs/your-cluster.yaml ~/.kube/hyperpod-ssm
chmod 600 ~/.kube/hyperpod-ssm
export KUBECONFIG=~/.kube/hyperpod-ssm
```

The launcher does not depend on the home-directory copy.

Skip both fields (`ssm_relay_instance_id`, `kubeconfig_path`) when
launching from inside the cluster — e.g. from a CI runner on a node
group, or from a pod with kubectl pre-configured.

The setup doc has a [step-by-step tunnel + kubectl health-check
ladder](../intuit/docs/hyperpod-eks-setup-notes.md#network-access) for when the
tunnel goes silent mid-job.

---

## Topology & vLLM Mode

The HyperPod EKS backend supports a subset of the SageMaker topology /
vllm_mode matrix:

| `topology` | `vllm_mode=none` | `vllm_mode=colocate` | `vllm_mode=server` |
|---|---|---|---|
| `single_node` | ✓ (torchrun) | ✓ (torchrun, vLLM in-process) | ✗ |
| `multi_node_server` | ✓ (torchrun) | ✓ (torchrun, vLLM in-process) | ✗ |
| `dedicated_vllm` | N/A | N/A | ✓ — **canonical HyperPod path** |

`vllm_mode=server` is only meaningful on `dedicated_vllm` because the
in-pod vLLM split that the SageMaker backend uses for the
`single_node` and `multi_node_server` paths isn't implemented on
Kubernetes. Setting an unsupported combination raises a
`NotImplementedError` with a clear message at manifest-build time, and
the same combinations are rejected by `rl-train validate` upfront.

`dedicated_vllm` requires `instance_count ≥ 2` (node 0 runs the vLLM
Deployment, nodes 1..N-1 run the PyTorchJob workers). This is enforced
in validation.

The vLLM pod's command does **not** invoke `trl vllm-serve` directly. Like
the SageMaker and local backends, it runs `python /tmp/code/vllm_launcher.py
...` so the same spawn-safety hygiene and `patch_trl_vllm` opt-ins
(`reasoning_config`, `compilation_config`, `disable_custom_all_reduce`)
apply identically here. See
[Reasoning + CUDA-graph Control](reasoning-and-cuda-graph.md) for the
full env-var contract — the EKS manifest builder emits `export VAR=...`
lines before the launcher invocation so the launcher sees the opt-ins in
its process environment.

For the field-level reference of the rest of the `infra` block
(`instance_type`, `vllm.*`, etc.), see
[Infrastructure & Runtime in the SageMaker doc](launching-sagemaker-training-job.md#infrastructure--runtime) —
they're shared.

---

## Container Configuration

The HyperPod EKS backend resolves `container.recipe` the same way the
SageMaker launcher does, with **one important deviation**:

| Scenario | Behaviour |
|---|---|
| `container.requirements` set (inline list) | Used as-is, written to `requirements.txt` in the source bundle. |
| `container.recipe` set | Resolved to `container.resolved_requirements` via `foresightrlcore/config/recipes.py`. Written to `requirements.txt`. |
| Both unset | **Default to the `qwen3` recipe with a warning**. (The SageMaker launcher falls back to the static `runtime/scripts/requirements.txt` here, but that file is synced to `qwen3_5` which uses `transformers>=5.6` — incompatible with the trl 0.28.x line that the HyperPod path runs today. The default protects backward compat for older HyperPod configs that predate the recipe system.) |

To silence the warning, set `container.recipe: qwen3` explicitly.

> **Why the qwen3 cap on transformers.** The `qwen3.txt` recipe pins
> `transformers>=4.57,<5`. The `<5` cap is non-negotiable for trl 0.28:
> in `transformers` 5.x, `_is_package_available(..., return_version=False)`
> returns a `(bool, None)` tuple instead of a plain bool, which makes
> every `if is_<X>_available()` guard in trl 0.28 evaluate truthy and
> unconditionally imports `vllm_ascend` (Huawei NPU only) + `weave`
> (W&B), crashing both worker and vLLM pods. Revisit when trl
> moves past 0.28; the `qwen3_5` recipe is already on trl>=1.3 +
> transformers>=5.6 for that codepath.

See [Container Configuration in the SageMaker doc](launching-sagemaker-training-job.md#container-configuration)
for the full recipe resolution priority + how to override.

---

## Manifests Generated

The launcher applies one or three Kubernetes resources depending on
topology.

### `dedicated_vllm` topology (canonical path)

Three resources, applied in order:

1. **`Deployment`** — `<job-name>-vllm`. Runs the vLLM server pod on
   node 0 (all GPUs → vLLM). Stays up after the job finishes; **must
   be deleted manually after non-wait launches** — the launcher emits
   a warning with the exact `kubectl delete` command when `wait=False`.
2. **`Service`** — `<job-name>-vllm`. Headless service (`clusterIP:
   None`) so the TRL trainer's NCCL TCPStore can reach vLLM on
   port 51216 directly without the kube-proxy hop.
3. **`PyTorchJob`** (Kubeflow `kubeflow.org/v1`) — `<job-name>`. Runs
   the training workers on nodes 1..N-1, with
   `elasticPolicy.rdzvBackend: c10d` so the Kubeflow operator injects
   the `PET_*` env vars torchrun needs for peer discovery.

### `single_node` / `multi_node_server` topology

One resource:

- **`PyTorchJob`** — same shape as above but with a single homogeneous
  Worker replica group. `vllm_mode=colocate` runs vLLM in-process
  inside each worker; `vllm_mode=none` runs pure torchrun with native
  HF generation.

The full manifest building code lives in
`foresightrlcore/execution/aws_hyperpod_eks.py`. To preview the
manifest for a config without applying it, see
[Generating and Applying Manifests in the setup doc](../intuit/docs/hyperpod-eks-setup-notes.md#generating-and-applying-manifests).

---

## Validation

`rl-train validate --config my_job.yaml` runs three layers of checks
before anything is submitted:

| Level | What's checked |
|---|---|
| **Pydantic schema** | Types, ranges, required fields |
| **Cross-field validation** (`foresightrlcore/config/validation.py`) | HyperPod-specific config rules, listed below |
| **Backend runtime validate** (`AwsHyperPodEksBackend.validate`) | Process-level concerns (`kubectl` available, `boto3` importable) |

### HyperPod EKS Cross-Field Rules

| Rule | Effect |
|---|---|
| `execution.hyperpod_eks` is required when `execution.type == "aws_hyperpod_eks"` | Error |
| `execution.hyperpod_eks.image_uri` is required | Error |
| `data.train_path` must start with `s3://` | Error — workers fetch the data via `aws s3 cp` |
| `output.s3_path` must start with `s3://` | Error — the launcher uploads the source tarball there before submitting |
| `topology: dedicated_vllm` | Requires `instance_count ≥ 2` |
| `vllm_mode: server` | Requires `topology` in `{single_node, dedicated_vllm, multi_node_server}` — only `dedicated_vllm` is actually implemented for HyperPod EKS today |

A common foot-gun caught by these rules:
`output.s3_path: /fsx/output/grpo-advice` (an FSx-shaped path). The
launcher uses this path to upload the source bundle to S3, so an FSx
path silently parses as `bucket=""` and crashes deep inside
`boto3.upload_file`. The validation now rejects it upfront with a
clear message pointing at the offending field.

For the SageMaker / GRPO / Secrets validation rules that apply
regardless of backend, see
[Validation in the SageMaker doc](launching-sagemaker-training-job.md#validation).

---

## CLI Reference

```bash
# Validate without launching
rl-train validate --config my_job.yaml

# Launch (returns immediately with the K8s job-name)
AWS_DEFAULT_REGION=us-east-1 rl-train train --config my_job.yaml

# Launch and wait for completion
AWS_DEFAULT_REGION=us-east-1 rl-train train --config my_job.yaml --wait
```

The launcher prints `Job ID: <job-name>` on success. Use that name with
`kubectl` for monitoring:

```bash
# Point kubectl at the same kubeconfig the launcher used.
KUBE=kubeconfigs/your-cluster.yaml
# KUBE=$HOME/.kube/hyperpod-ssm                     # if you did the optional cp
JOB=<job-name>

kubectl --kubeconfig $KUBE get pytorchjob $JOB -n default
kubectl --kubeconfig $KUBE get pods -n default -l job-id=$JOB
kubectl --kubeconfig $KUBE logs -n default ${JOB}-worker-0 --tail 200 -f
```

Log every successful launch to `job_tracking/<job-id>.yaml` per the
[`track-training-jobs` skill](../.claude/skills/track-training-jobs/SKILL.md).

---

## Debugging

For end-to-end debugging recipes (when the tunnel goes silent mid-job,
when the operator misroutes pods, when vLLM hangs on warmup, etc.),
see [Debugging Commands](../intuit/docs/hyperpod-eks-setup-notes.md#debugging-commands)
in the setup doc.

---

## Full Config Example

A complete annotated config for GRPO training on HyperPod EKS with
LLM-as-judge evaluation:

```yaml
version: 1

job:
  name: grpo-advice-qwen3-hyperpod
  max_runtime_hours: 24

model:
  name: Qwen/Qwen3-8B
  enable_thinking: true
  lora:
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules: [q_proj, k_proj, v_proj, o_proj]

method:
  type: grpo

execution:
  type: aws_hyperpod_eks
  hyperpod_eks:
    cluster_name: "your-hyperpod-cluster"
    eks_cluster_name: "your-eks-cluster"
    namespace: default
    image_uri: "123456789012.dkr.ecr.us-east-1.amazonaws.com/your-training-image:qwen3-vllm-latest"
    service_account: default
    fsx_filesystem_id: "fs-0123456789abcdef0"
    fsx_mount_path: /fsx
    pvc_name: "fsx-claim"
    restart_policy: OnFailure
    max_restarts: 3
    ssm_relay_instance_id: i-0123456789abcdef0
    kubeconfig_path: kubeconfigs/your-cluster.yaml

infra:
  topology: dedicated_vllm
  vllm_mode: server
  instance_type: ml.p4d.24xlarge
  instance_count: 2
  vllm:
    port: 8002
    tensor_parallel_size: 8
    gpu_memory_utilization: 0.9
    enforce_eager: false

training:
  epochs: 1
  per_device_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-06
  lr_scheduler: cosine
  warmup_ratio: 0.03
  bf16: true
  gradient_checkpointing: true
  deepspeed: ds_zero2_config.json
  logging_steps: 5
  save_steps: 50
  eval_steps: 100
  max_grad_norm: 0.5
  grpo:
    num_generations: 16
    max_completion_length: 4096
    beta: 0.08
    temperature: 1.0
    top_p: 0.95
    loss_type: dr_grpo
    scale_rewards: false
    mask_truncated_completions: true
    num_iterations: 1

data:
  train_path: "s3://my-bucket/data/train.jsonl"
  eval_split_size: 100

output:
  s3_path: "s3://my-bucket/output/grpo-advice-hyperpod"

environment:
  GENOS_ENV: "e2e"
  GENOS_EXPERIENCE_ID: "your-experience-id"
  GENOS_INTUIT_ORIGINATING_ASSETALIAS: "your-asset-alias"
  GENOS_OFFLINE_JOB_ID: "your-job-id"

container:
  recipe: qwen3

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
    judge_parallel_workers: 10
    think_length_soft_cap: 15000
    think_length_hard_cap: 25000
    think_penalty_weight: 0.20
    llm_clients:
      default:
        provider: genos
        genos:
          model_id: anthropic.claude-opus-4-5-20251101-v1-0
```

For a minimal end-to-end smoke test of the same path with a tiny
model, see [intuit/examples/smoke_tests/advice/hyperpod_eks_dedicated_vllm_smoke_test.yaml](../intuit/examples/smoke_tests/advice/hyperpod_eks_dedicated_vllm_smoke_test.yaml).
