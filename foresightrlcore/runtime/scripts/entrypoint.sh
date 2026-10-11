#!/bin/bash
# =============================================================================
# SageMaker GRPO Training Entrypoint
#
# Orchestrates vLLM server + GRPO training inside the SageMaker container.
# Supports two topologies:
#   - single_node: vLLM and training share GPUs on one instance
#   - multi_node:  vLLM runs on a dedicated instance group
# =============================================================================
set -euo pipefail

# Global vLLM PID (set by start_vllm_server, used by cleanup trap)
VLLM_PID=""

# ── Cleanup trap ─────────────────────────────────────────────────────────────
# Registered on EXIT so every code path (normal exit, error, signal) kills any
# lingering vLLM processes before the script returns.
#
# Why process-group kill matters:
#   The vLLM server (uvicorn, PID 140) forks worker processes (e.g. PID 373).
#   When uvicorn exits after a graceful-shutdown attempt, surviving workers are
#   reparented to the container init (tini).  A plain `kill $VLLM_PID` or even
#   `pkill -P $VLLM_PID` finds nothing once the parent is gone.
#   However, forked workers *inherit* their parent's process-group ID (PGID) and
#   keep it even after reparenting.  Because start_vllm_server uses `setsid`,
#   the vLLM process becomes session leader with PGID == its own PID.  All its
#   descendants share that PGID, so `kill -KILL -$PGID` reaches every survivor
#   regardless of which init process adopted them.
cleanup_vllm() {
    # ── Server mode: VLLM_PID was set by start_vllm_server ─────────────────
    if [ -n "${VLLM_PID:-}" ]; then
        echo "Cleanup: sending SIGTERM to vLLM process group (PGID=$VLLM_PID)..."
        # Signal the entire process group (main server + all worker children)
        kill -TERM -"$VLLM_PID" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null || true

        # Wait up to 20 s for the uvicorn main process to exit gracefully
        local waited=0
        while kill -0 "$VLLM_PID" 2>/dev/null && [ "$waited" -lt 20 ]; do
            sleep 2
            waited=$((waited + 2))
        done

        # Force-kill the entire process group.  Any worker that survived SIGTERM
        # (e.g. stuck in a CUDA kernel) still carries PGID=$VLLM_PID, so this
        # reaches it even if it was reparented by the time we get here.
        echo "Cleanup: sending SIGKILL to vLLM process group (PGID=$VLLM_PID)..."
        kill -KILL -"$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi

    # ── Colocate mode: TRL manages an internal vLLM server ─────────────────
    # After `accelerate launch` exits, TRL's vllm-serve process and any workers
    # it forked may still be running.  Kill by name as a safety net.
    pkill -KILL -f "trl vllm-serve" 2>/dev/null || true
    pkill -KILL -f "vllm_serve\\.py" 2>/dev/null || true
}
trap cleanup_vllm EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SM_CHANNEL_CONFIG:-/opt/ml/input/data/config}/args.yaml"

TRAIN_SCRIPT="${SCRIPT_DIR}/train.py"

# Fallback: if args.yaml doesn't exist, try to find any yaml in the config dir
if [ ! -f "$CONFIG_PATH" ]; then
    CONFIG_DIR="${SM_CHANNEL_CONFIG:-/opt/ml/input/data/config}"
    CONFIG_PATH=$(find "$CONFIG_DIR" -name "*.yaml" -o -name "*.yml" | head -1)
    if [ -z "$CONFIG_PATH" ]; then
        echo "ERROR: No YAML config found in $CONFIG_DIR"
        exit 1
    fi
fi

echo "============================================="
echo "SageMaker GRPO Training Entrypoint"
echo "Config: $CONFIG_PATH"
echo "Host: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "============================================="

# ── Helper: parse YAML values via Python ─────────────────────────────────
yaml_get() {
    python3 -c "
import yaml, sys
with open('$CONFIG_PATH') as f:
    c = yaml.safe_load(f)
keys = '$1'.split('.')
v = c
for k in keys:
    v = v.get(k, {}) if isinstance(v, dict) else {}
print(v if v != {} else '$2')
"
}

# ── Parse config ─────────────────────────────────────────────────────────
TOPOLOGY=$(yaml_get "infra.topology" "single_node")
VLLM_MODE=$(yaml_get "infra.vllm_mode" "server")
MODEL_NAME=$(yaml_get "model.name" "Qwen/Qwen3-14B")

# Pre-staged model: if the launcher attached a SageMaker 'model' input channel
# (set when YAML model.name is an s3:// URI), prefer the local mount over the
# HF Hub identifier so vLLM (and the trainer) load from disk. This eliminates
# the multi-rank HF cache race that crashes from_pretrained() with
# "shard N not found" on large models.
SM_MODEL_CHANNEL_DIR="${SM_CHANNEL_MODEL:-/opt/ml/input/data/model}"
if [[ -f "$SM_MODEL_CHANNEL_DIR/config.json" ]]; then
    echo "Pre-staged model detected at $SM_MODEL_CHANNEL_DIR — overriding MODEL_NAME (was: $MODEL_NAME)"
    MODEL_NAME="$SM_MODEL_CHANNEL_DIR"
fi
VLLM_PORT=$(yaml_get "infra.vllm.port" "8002")
VLLM_TP_SIZE=$(yaml_get "infra.vllm.tensor_parallel_size" "2")
VLLM_GPU_MEM=$(yaml_get "infra.vllm.gpu_memory_utilization" "0.9")
VLLM_MAX_MODEL_LEN=$(yaml_get "infra.vllm.max_model_len" "None")
VLLM_ENFORCE_EAGER=$(yaml_get "infra.vllm.enforce_eager" "False")
VLLM_ENABLE_PREFIX_CACHING=$(yaml_get "infra.vllm.enable_prefix_caching" "False")
VLLM_STARTUP_TIMEOUT=$(yaml_get "infra.vllm.startup_timeout" "1800")
# Patch opt-ins forwarded to vllm_launcher.py via env vars. The launcher
# reads them at startup and calls patch_trl_vllm.patch() before importing
# TRL — see foresightrlcore/runtime/scripts/vllm_launcher.py for the
# contract. Schema field: infra.vllm.disable_custom_all_reduce.
VLLM_DISABLE_CUSTOM_ALL_REDUCE_YAML=$(yaml_get "infra.vllm.disable_custom_all_reduce" "False")
if [ "$VLLM_DISABLE_CUSTOM_ALL_REDUCE_YAML" = "True" ]; then
    export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
fi

# Nested speculative-decoding config — emitted as a single JSON object
# (or empty string if absent). yaml_get can't handle nested dicts, so do it inline.
VLLM_SPECULATIVE_JSON=$(python3 -c "
import json, yaml
c = yaml.safe_load(open('$CONFIG_PATH')) or {}
spec = ((c.get('infra') or {}).get('vllm') or {}).get('speculative')
print(json.dumps(spec, separators=(',',':')) if spec else '')
" 2>/dev/null)

# Reasoning parser identifier (e.g. 'qwen3'). Empty string when absent.
VLLM_REASONING_PARSER=$(python3 -c "
import yaml
c = yaml.safe_load(open('$CONFIG_PATH')) or {}
v = ((c.get('infra') or {}).get('vllm') or {}).get('reasoning_parser')
print(v if v else '')
" 2>/dev/null)

# Reasoning config (start/end delimiter dict) — JSON object, empty when absent.
# Exported (not just a shell var) so vllm_launcher.py — which keys on
# VLLM_REASONING_CONFIG_JSON — sees it in the subprocess env.
VLLM_REASONING_CONFIG_JSON=$(python3 -c "
import json, yaml
c = yaml.safe_load(open('$CONFIG_PATH')) or {}
rc = ((c.get('infra') or {}).get('vllm') or {}).get('reasoning_config')
print(json.dumps(rc, separators=(',',':')) if rc else '')
" 2>/dev/null)
[ -n "$VLLM_REASONING_CONFIG_JSON" ] && export VLLM_REASONING_CONFIG_JSON

# CompilationConfig kwargs — JSON object, empty when absent.
# Exported so vllm_launcher.py — which keys on VLLM_COMPILATION_CONFIG_JSON
# — sees it in the subprocess env.
VLLM_COMPILATION_CONFIG_JSON=$(python3 -c "
import json, yaml
c = yaml.safe_load(open('$CONFIG_PATH')) or {}
cc = ((c.get('infra') or {}).get('vllm') or {}).get('compilation_config')
print(json.dumps(cc, separators=(',',':')) if cc else '')
" 2>/dev/null)
[ -n "$VLLM_COMPILATION_CONFIG_JSON" ] && export VLLM_COMPILATION_CONFIG_JSON

# Local staging dir for an S3-sourced speculative draft model. Populated
# lazily by stage_speculative_draft() right before vLLM launches, so we don't
# pay the download cost on import or when speculative is unset.
SPECULATIVE_DRAFT_STAGE_DIR="/tmp/speculative_draft_model"

# If the speculative config points at an s3:// draft model, sync it to local
# disk and rewrite VLLM_SPECULATIVE_JSON to use that local path. vLLM's
# `--speculative-config model:` only accepts a local path or HF Hub ID.
# Idempotent: a second call is a no-op if the JSON already has a local path.
stage_speculative_draft() {
    [ -z "$VLLM_SPECULATIVE_JSON" ] && return 0

    local draft_uri
    draft_uri=$(python3 -c "
import json
spec = json.loads('''$VLLM_SPECULATIVE_JSON''')
print(spec.get('model') or '')
")
    case "$draft_uri" in
        s3://*) ;;
        *) return 0 ;;  # local path or HF ID — nothing to stage
    esac

    echo "Staging speculative draft model from $draft_uri to $SPECULATIVE_DRAFT_STAGE_DIR ..."
    mkdir -p "$SPECULATIVE_DRAFT_STAGE_DIR"
    if ! aws s3 sync "$draft_uri" "$SPECULATIVE_DRAFT_STAGE_DIR" --only-show-errors; then
        echo "ERROR: failed to sync speculative draft from $draft_uri"
        return 1
    fi
    echo "Speculative draft staged. Files:"
    ls -la "$SPECULATIVE_DRAFT_STAGE_DIR" | head -20

    # Rewrite the JSON in-place so downstream `--speculative-config` callers
    # see the local path.
    VLLM_SPECULATIVE_JSON=$(python3 -c "
import json
spec = json.loads('''$VLLM_SPECULATIVE_JSON''')
spec['model'] = '$SPECULATIVE_DRAFT_STAGE_DIR'
print(json.dumps(spec, separators=(',',':')))
")
    echo "Rewrote speculative-config model -> $SPECULATIVE_DRAFT_STAGE_DIR"
}

echo "Topology: $TOPOLOGY"
echo "vLLM mode: $VLLM_MODE"
echo "Model: $MODEL_NAME"
echo "vLLM port: $VLLM_PORT, TP size: $VLLM_TP_SIZE"
echo "vLLM startup timeout: ${VLLM_STARTUP_TIMEOUT}s"

# ── Install dependencies ──────────────────────────────────────────────────
# Hide GPUs during pip install so build scripts (e.g. deepspeed) don't probe
# CUDA and taint the process with an early CUDA context — that context would
# later prevent vLLM workers from using multiprocessing.spawn cleanly.
REQ_FILE="${SCRIPT_DIR}/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    echo "Installing dependencies from $REQ_FILE ..."
    CUDA_VISIBLE_DEVICES="" pip install --no-cache-dir -r "$REQ_FILE"
    echo "Dependencies installed successfully"
else
    echo "WARNING: No requirements.txt found at $REQ_FILE"
fi

# ── Ensure flash-attn is loadable against the on-host CUDA toolchain ──────
# The PyPI flash-attn wheels are built for libcudart.so.12. The qwen3_5 recipe
# runs on a cu130 DLC (libcudart.so.13), where that prebuilt wheel fails to
# import; the qwen3 recipe runs on a cu12 DLC, where it imports natively. The
# Qwen3 model code (Qwen3_VisionTransformer.ApplyRotaryEmb) imports flash_attn
# directly, so VLLM_ATTENTION_BACKEND can't sidestep a broken wheel.
#
# Therefore: probe first, rebuild only if needed. Importing flash_attn_2_cuda
# loads the compiled .so and surfaces any libcudart mismatch. If it imports
# cleanly (cu12 path) we skip the ~15+ min source build entirely — this is what
# kept the qwen3 single_node smoke fast before FILM-5305 made the rebuild
# unconditional. We hide GPUs (CUDA_VISIBLE_DEVICES="") for the probe because
# `import torch` would otherwise initialize a CUDA context in this parent shell,
# which later taints vLLM's forked TP workers ("Cannot re-initialize CUDA in
# forked subprocess").
if CUDA_VISIBLE_DEVICES="" python3 -c "import torch; import flash_attn_2_cuda" 2>/dev/null; then
    echo "flash-attn already loads against on-host CUDA; skipping source rebuild"
else
    echo "flash-attn not loadable against on-host CUDA — building from source..."
    # --force-reinstall: the requirements.txt step may have pulled in a cu12-built
    #   flash-attn wheel via vllm; replace it with a source build for this CUDA.
    # --no-deps: avoid reinstalling torch/numpy etc.
    # --no-build-isolation: build against the already-installed torch (else it
    #   fetches a fresh torch with mismatched CUDA).
    # Note: flash-attn needs CUDA headers at build time but CUDA_VISIBLE_DEVICES=""
    # only hides runtime GPU access, not the CUDA toolkit — the build still works.
    #
    # MAX_JOBS caps the number of parallel nvcc compile units. Without it, ninja
    # spawns one job per vCPU (48 on g5.12xlarge), and each flash-attn nvcc unit
    # peaks at several GB — concurrently that exhausts host RAM and SageMaker
    # kills the instance with the misleading "Please use an instance type with
    # more memory" ClientError (it really is host-RAM OOM during the build, not
    # disk). 4 is the empirically safe ceiling; override via FLASH_ATTN_MAX_JOBS.
    FLASH_ATTN_MAX_JOBS="${FLASH_ATTN_MAX_JOBS:-4}"
    echo "flash-attn build parallelism: MAX_JOBS=${FLASH_ATTN_MAX_JOBS}"
    CUDA_VISIBLE_DEVICES="" MAX_JOBS="${FLASH_ATTN_MAX_JOBS}" \
        pip install --no-cache-dir --force-reinstall --no-deps --no-build-isolation \
        "flash-attn>=2.7.0" \
        || { echo "ERROR: Failed to build flash-attn from source"; exit 1; }
    CUDA_VISIBLE_DEVICES="" python3 -c "import torch; import flash_attn_2_cuda; print('flash-attn source build OK')" \
        || { echo "ERROR: flash-attn rebuilt but flash_attn_2_cuda still fails to import"; exit 1; }
    echo "flash-attn built and installed successfully"
fi

# ── TRL vllm_serve.py patching ────────────────────────────────────────────
# Moved into vllm_launcher.py: the launcher reads the three opt-in env vars
# (VLLM_REASONING_CONFIG_JSON, VLLM_COMPILATION_CONFIG_JSON,
# VLLM_DISABLE_CUSTOM_ALL_REDUCE) and calls patch_trl_vllm.patch() before
# importing TRL. This guarantees the patch runs in every backend (SageMaker
# entrypoint, local GPU box, HyperPod EKS pod) without duplicating logic
# across each backend's launch glue.

# ── Resolve secrets (if configured) ──────────────────────────────────────
# provider=local is resolved in resolve_secrets.py. provider=idps is
# delegated to foresightrl_intuit (container.extra_packages). The internal
# training image bakes the secrets client in (intuit/docker/Dockerfile.base);
# this script does not install it and does not name an internal index.
# For HyperPod EKS: pod authenticates to IDPS VKM via IRSA (no credentials injected).
# For SageMaker: same flow; SageMaker execution role must be trusted by the IDPS policy.
SECRETS_ENV_FILE="/tmp/resolved_secrets.env"
if python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG_PATH')); exit(0 if c.get('secrets') else 1)" 2>/dev/null; then
    echo "Resolving secrets..."
    python3 "${SCRIPT_DIR}/resolve_secrets.py" --config "$CONFIG_PATH" --env-file "$SECRETS_ENV_FILE" \
        || { echo "ERROR: Secret resolution failed"; exit 1; }

    if [ -f "$SECRETS_ENV_FILE" ]; then
        echo "Sourcing resolved secrets from $SECRETS_ENV_FILE"
        source "$SECRETS_ENV_FILE"
    fi
    echo "Secret resolution complete."
fi

# ── Environment setup ────────────────────────────────────────────────────
# hf_transfer is the Rust-backed parallel downloader — much faster on
# multi-shard models, but has a known race where the snapshot index
# reports completion before the last (smallest) shard lands on disk.
# Default to enabled; honor a user override (set HF_HUB_ENABLE_HF_TRANSFER
# in the YAML environment block to "0" if the model fails to load
# cleanly via from_pretrained after a "Fetching N files: 100%" log line).
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM=false
# Force spawn multiprocessing for vLLM TP workers — on H100/P5, fork-based
# workers can hit "Cannot re-initialize CUDA in forked subprocess" when CUDA
# was probed in the parent (e.g. by deepspeed during pip install). Honor a
# user override if VLLM_WORKER_MULTIPROC_METHOD is already set in the env.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

# ── vLLM health check function ──────────────────────────────────────────
wait_for_vllm() {
    local host="${1:-localhost}"
    local port="${2:-$VLLM_PORT}"
    local timeout="${3:-600}"
    local interval=5
    local elapsed=0

    echo "Waiting for vLLM server at $host:$port (timeout: ${timeout}s)..."

    while [ $elapsed -lt $timeout ]; do
        if curl -s "http://$host:$port/health" > /dev/null 2>&1; then
            echo "vLLM server is healthy after ${elapsed}s"
            return 0
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "  Still waiting for vLLM... (${elapsed}s elapsed)"
        fi
        # Print a compact GPU snapshot every 15s — live visibility into
        # weight-load progress (memory ramp) and GPU utilisation while
        # vLLM is starting.
        if [ $((elapsed % 15)) -eq 0 ]; then
            echo "  GPU snapshot @ ${elapsed}s:"
            nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
                --format=csv,noheader 2>/dev/null | sed 's/^/    /' || true
        fi
    done

    echo "ERROR: vLLM server did not become healthy within ${timeout}s"
    return 1
}

TORCH_RANK_LOG_DIR="${TORCH_RANK_LOG_DIR:-/opt/ml/output/torch_rank_logs}"

# ── Detect GPU P2P support ────────────────────────────────────────────────
# Returns 0 (true) if NVLink P2P is available, 1 (false) if PCIe-only.
#
# Primary: SageMaker sets SM_CURRENT_INSTANCE_TYPE reliably.  NVLink-capable
# families (p3, p4d, p4de, p5, p5e, p5en) are matched by prefix.
# Fallback: nvidia-smi topo probe for local GPU runs outside SageMaker.
gpu_p2p_available() {
    local instance_type="${SM_CURRENT_INSTANCE_TYPE:-}"
    if [ -n "$instance_type" ]; then
        local family
        family=$(echo "$instance_type" | sed 's/^ml\.//' | cut -d. -f1)
        case "$family" in
            p[3-9]*|p[1-9][0-9]*)
                echo "GPU P2P: NVLink — instance $instance_type (family=$family)"
                return 0
                ;;
            *)
                echo "GPU P2P: PCIe-only — instance $instance_type (family=$family)"
                return 1
                ;;
        esac
    fi

    if nvidia-smi topo -m 2>/dev/null | grep -q "NV[0-9]"; then
        echo "GPU P2P: NVLink detected via nvidia-smi topo"
        return 0
    fi

    echo "GPU P2P: No NVLink detected (nvidia-smi topo fallback)"
    return 1
}

# After a failed accelerate/torchrun launch: grep + tail each rank stderr (see --log_dir in launch_training).
dump_training_diagnostics() {
    local exit_code="${1:-1}"
    [ "${exit_code}" -eq 0 ] && return 0
    echo "--- Training failure diagnostics (exit=${exit_code}) ---"
    echo "If you see NCCL 'Connection closed by localRank N', that rank exited first—check its stderr below."
    if [ -d "${TORCH_RANK_LOG_DIR}" ]; then
        find "${TORCH_RANK_LOG_DIR}" -type f -name 'stderr.log' 2>/dev/null | sort | while read -r f; do
            echo "===== ${f} (grep: Traceback/OOM/errors) ====="
            grep -E -n 'Traceback|CUDA out of memory|OutOfMemory|RuntimeError|Error:|Exception|Killed|deepspeed|DeepSpeed' "${f}" 2>/dev/null | tail -n 35 || true
            echo "===== ${f} (last 100 lines) ====="
            tail -n 100 "${f}" 2>/dev/null || true
        done
        if [ "$(find "${TORCH_RANK_LOG_DIR}" -type f -name 'stderr.log' 2>/dev/null | wc -l | tr -d ' ')" -eq 0 ]; then
            echo "WARNING: no stderr.log under ${TORCH_RANK_LOG_DIR}"
            find "${TORCH_RANK_LOG_DIR}" -type f 2>/dev/null | head -n 30 || true
        fi
    else
        echo "WARNING: ${TORCH_RANK_LOG_DIR} missing"
    fi
    find /tmp -maxdepth 6 \( -name 'torchelastic*.json' -o -name 'elastic_error*.json' \) 2>/dev/null \
        | while read -r f; do
            echo "===== ${f} ====="
            cat "${f}" 2>/dev/null || true
        done
    echo "--- end diagnostics ---"
}

# ── Start vLLM server ───────────────────────────────────────────────────
start_vllm_server() {
    local cuda_devices="$1"

    stage_speculative_draft || return 1

    # Route through vllm_launcher.py instead of `trl vllm-serve` directly so
    # multiprocessing.set_start_method("spawn", force=True) and
    # CUDA_MODULE_LOADING=LAZY are applied BEFORE TRL imports torch/vllm.
    # Without this the vLLM TP workers segfault on H100/H200 with
    # "Cannot re-initialize CUDA in forked subprocess". See vllm_launcher.py
    # docstring for the full rationale.
    local vllm_cmd="python ${SCRIPT_DIR}/vllm_launcher.py --model $MODEL_NAME"
    vllm_cmd="$vllm_cmd --host 0.0.0.0 --port $VLLM_PORT"
    vllm_cmd="$vllm_cmd --tensor-parallel-size $VLLM_TP_SIZE"
    vllm_cmd="$vllm_cmd --gpu-memory-utilization $VLLM_GPU_MEM"

    if [ "$VLLM_MAX_MODEL_LEN" != "None" ]; then
        vllm_cmd="$vllm_cmd --max-model-len $VLLM_MAX_MODEL_LEN"
    fi
    if [ "$VLLM_ENFORCE_EAGER" = "True" ]; then
        vllm_cmd="$vllm_cmd --enforce-eager"
    fi
    if [ "$VLLM_ENABLE_PREFIX_CACHING" = "True" ]; then
        vllm_cmd="$vllm_cmd --enable-prefix-caching yes"
    fi
    if [ -n "$VLLM_SPECULATIVE_JSON" ]; then
        # Single-quote-wrap so the JSON survives the `bash -c "$vllm_cmd ..."` below.
        # The JSON itself only contains double quotes, so single quotes are safe.
        vllm_cmd="$vllm_cmd --speculative-config '$VLLM_SPECULATIVE_JSON'"
    fi

    echo "Starting vLLM server on GPUs: $cuda_devices"
    echo "Command: CUDA_VISIBLE_DEVICES=$cuda_devices $vllm_cmd"
    echo "vLLM stdout/stderr will be streamed below with [vllm] prefix."

    # setsid gives the vLLM server its own session, making its PID == its PGID.
    # All worker processes it forks inherit that PGID, so cleanup_vllm can
    # kill the entire group with `kill -KILL -$PGID` even after workers have
    # been reparented by tini (which happens when the uvicorn master exits first).
    #
    # `2>&1 | stdbuf -oL sed 's/^/[vllm] /'`:
    #   - merges stderr into stdout so warnings/errors are captured too
    #   - prefixes every line with [vllm] so it's easy to spot/filter in CloudWatch
    #   - stdbuf -oL forces line-buffered output so we see progress live, not
    #     in 4KB chunks after the model finally loads
    # Unset PYTORCH_CUDA_ALLOC_CONF for the vLLM subprocess only — vLLM's
    # custom all-reduce kernel uses CUDA IPC handles which are incompatible
    # with the expandable_segments allocator. `env -u` removes the variable
    # entirely (not just sets it to empty), so PyTorch falls back to its
    # built-in default. Training keeps the inherited value from the container env.
    # VLLM_ENABLE_INDUCTOR_*=0 disables Inductor autotune paths that can trigger
    # the H100 CUDA-graph compile hang. Applied to the vLLM subprocess only —
    # training keeps default Inductor behavior.
    env -u PYTORCH_CUDA_ALLOC_CONF \
        PYTHONUNBUFFERED=1 \
        VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=0 \
        VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING=0 \
        CUDA_VISIBLE_DEVICES="$cuda_devices" setsid bash -c \
        "$vllm_cmd 2>&1 | stdbuf -oL sed 's/^/[vllm] /'" &
    VLLM_PID=$!
    echo "vLLM server started with PID: $VLLM_PID (PGID=$VLLM_PID)"
}

# ── Launch training ──────────────────────────────────────────────────────
launch_training() {
    local cuda_devices="$1"
    local num_processes="$2"
    local vllm_host="${3:-0.0.0.0}"

    mkdir -p "${TORCH_RANK_LOG_DIR}"
    local train_cmd="accelerate launch --num_processes $num_processes"
    train_cmd="$train_cmd --log_dir ${TORCH_RANK_LOG_DIR} --tee 3"
    train_cmd="$train_cmd ${TRAIN_SCRIPT}"
    train_cmd="$train_cmd --config $CONFIG_PATH"

    if [ "$vllm_host" != "0.0.0.0" ] && [ "$vllm_host" != "localhost" ]; then
        train_cmd="$train_cmd --vllm-host $vllm_host"
    fi

    echo "Launching training on GPUs: $cuda_devices ($num_processes processes)"
    echo "Command: CUDA_VISIBLE_DEVICES=$cuda_devices $train_cmd"

    CUDA_VISIBLE_DEVICES="$cuda_devices" $train_cmd
    return $?
}

# ── Build CUDA_VISIBLE_DEVICES strings ───────────────────────────────────
build_gpu_ranges() {
    local vllm_count="$1"
    local train_count="$2"
    local total_gpus

    total_gpus=$(nvidia-smi -L 2>/dev/null | wc -l)
    total_gpus=$((total_gpus + 0))  # ensure numeric

    if [ $((vllm_count + train_count)) -gt "$total_gpus" ]; then
        echo "WARNING: Requested $((vllm_count + train_count)) GPUs but only $total_gpus available"
        echo "Adjusting: vLLM=$vllm_count, training=$((total_gpus - vllm_count))"
        train_count=$((total_gpus - vllm_count))
    fi

    # vLLM gets GPUs 0..(vllm_count-1)
    VLLM_GPUS=""
    for i in $(seq 0 $((vllm_count - 1))); do
        [ -n "$VLLM_GPUS" ] && VLLM_GPUS="$VLLM_GPUS,"
        VLLM_GPUS="$VLLM_GPUS$i"
    done

    # Training gets GPUs vllm_count..(vllm_count+train_count-1)
    TRAIN_GPUS=""
    for i in $(seq "$vllm_count" $((vllm_count + train_count - 1))); do
        [ -n "$TRAIN_GPUS" ] && TRAIN_GPUS="$TRAIN_GPUS,"
        TRAIN_GPUS="$TRAIN_GPUS$i"
    done

    TRAIN_NUM_PROCESSES=$train_count

    echo "GPU allocation: vLLM=[$VLLM_GPUS], Training=[$TRAIN_GPUS] ($TRAIN_NUM_PROCESSES processes)"
}

# =============================================================================
# TOPOLOGY: SINGLE NODE
# =============================================================================
run_single_node() {
    local vllm_gpu_count
    local train_gpu_count

    vllm_gpu_count=$(yaml_get "infra.vllm_gpu_count" "2")
    train_gpu_count=$(yaml_get "infra.training_gpu_count" "6")

    build_gpu_ranges "$vllm_gpu_count" "$train_gpu_count"

    # vLLM server mode uses NCCL to sync weights from training → vLLM.
    # On PCIe-only GPUs (e.g. A10G on g5) CUDA P2P is unsupported and the
    # NCCL communicator init crashes.  Disable P2P so NCCL falls back to
    # network transport.  On NVLink GPUs this is skipped, keeping fast P2P.
    if ! gpu_p2p_available; then
        echo "Disabling NCCL P2P and SHM for vLLM weight sync (PCIe-only)"
        export NCCL_P2P_DISABLE=1
        export NCCL_SHM_DISABLE=1
        # With P2P and SHM both off, NCCL falls through to its *network*
        # transport. The DLC ships the Libfabric/EFA plugin on
        # LD_LIBRARY_PATH, so NCCL loads NET/OFI even on instances with no
        # EFA device (g5/g6), and the weight-sync all_reduce dies with
        # "NET/OFI ... RC: 103. Error: 4126 (Unresponsive receiver)".
        # This is single-node, so the socket transport is the right path.
        # Respect an explicit operator override from config.environment.
        if [ -z "${NCCL_NET_PLUGIN:-}" ]; then
            export NCCL_NET_PLUGIN=none
        fi
        if [ -z "${NCCL_NET:-}" ]; then
            export NCCL_NET=Socket
        fi
        echo "NCCL net transport: plugin=${NCCL_NET_PLUGIN} net=${NCCL_NET}"
    else
        echo "NVLink P2P enabled for vLLM weight sync"
    fi

    # Start vLLM server in background
    start_vllm_server "$VLLM_GPUS"

    # Wait for vLLM to be ready.
    # 1800s (30 min) accommodates Qwen3.5-27B weight load (~5 min) plus
    # cudagraph compile (~5-10 min) on first cold start.
    if ! wait_for_vllm "localhost" "$VLLM_PORT" 1800; then
        echo "Killing vLLM server (PID: $VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        exit 1
    fi

    # Launch training
    TRAIN_EXIT_CODE=0
    launch_training "$TRAIN_GPUS" "$TRAIN_NUM_PROCESSES" "0.0.0.0" || TRAIN_EXIT_CODE=$?

    dump_training_diagnostics "${TRAIN_EXIT_CODE}"

    # Cleanup — the EXIT trap (cleanup_vllm) will handle killing the vLLM
    # server and all its worker subprocesses, so just report and exit here.
    echo "Training finished with exit code: $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
}

# =============================================================================
# VLLM MODE: COLOCATE (single-node or multi-node)
# vLLM runs inside the trainer process — no separate server needed.
# Multi-node is auto-detected from SageMaker environment variables.
# =============================================================================
run_colocate() {
    local total_gpus
    total_gpus=$(nvidia-smi -L 2>/dev/null | wc -l)

    # Detect multi-node from SageMaker environment
    local host_count=1
    local node_rank=0
    local master_addr="localhost"
    local master_port="${SM_MASTER_PORT:-29500}"

    if [ -n "${SM_HOSTS:-}" ]; then
        host_count=$(python3 -c "import json; print(len(json.loads('${SM_HOSTS}')))")
        if [ "$host_count" -gt 1 ]; then
            master_addr=$(python3 -c "import json; print(json.loads('${SM_HOSTS}')[0])")
            node_rank=$(python3 -c "import json; print(json.loads('${SM_HOSTS}').index('${SM_CURRENT_HOST}'))")
        fi
    fi

    # In colocate mode each training rank owns tensor_parallel_size GPUs for its vLLM instance.
    # Dividing by TP size prevents multiple ranks from fighting over the same GPUs, which causes
    # "No available memory for the cache blocks" during GRPOTrainer init.
    local total_processes=$(( (total_gpus * host_count) / VLLM_TP_SIZE ))

    echo "Colocate mode: ${host_count} node(s), ${total_gpus} GPUs/node, TP=${VLLM_TP_SIZE}, ${total_processes} total processes"

    mkdir -p "${TORCH_RANK_LOG_DIR}"

    local train_cmd="accelerate launch --log_dir ${TORCH_RANK_LOG_DIR} --tee 3"
    train_cmd="$train_cmd --mixed_precision bf16"

    if [ "$host_count" -gt 1 ]; then
        echo "Multi-node colocate: rank=$node_rank, master=$master_addr:$master_port"

        # ── NCCL transport selection ──────────────────────────────────────
        # By default, try EFA (RDMA).  If the YAML config sets
        # infra.nccl_transport: sockets  we fall back to TCP — slower but
        # avoids EFA error-4126 handshake failures that occur on some VPCs.
        NCCL_TRANSPORT=$(yaml_get "infra.nccl_transport" "efa")
        echo "NCCL transport: $NCCL_TRANSPORT"

        if [ "$NCCL_TRANSPORT" = "sockets" ]; then
            # Bypass the aws-ofi-nccl plugin entirely — it crashes (SIGSEGV)
            # when the libfabric sockets provider can't do fi_mr_reg.
            # NCCL_NET=Socket forces NCCL's built-in TCP socket transport.
            export NCCL_NET="Socket"
            export NCCL_NET_PLUGIN=""
            export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
        else
            export FI_PROVIDER="${FI_PROVIDER:-efa}"
            export FI_EFA_USE_DEVICE_RDMA="${FI_EFA_USE_DEVICE_RDMA:-1}"
            export FI_EFA_FORK_SAFE="${FI_EFA_FORK_SAFE:-1}"
            export RDMAV_FORK_SAFE="${RDMAV_FORK_SAFE:-1}"
            export NCCL_PROTO="${NCCL_PROTO:-simple}"
            export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
        fi

        # Verbose NCCL logging — set NCCL_DEBUG=INFO via config or env to
        # see full transport negotiation and connection details.
        export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

        # Extend NCCL timeout to 30 min. NCCL_TIMEOUT is in milliseconds;
        # TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC is in seconds.
        # Non-blocking wait (=0) prevents the watchdog from aborting
        # while NCCL proxies are busy during vLLM init.
        export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800000}"
        export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
        export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"

        train_cmd="$train_cmd --num_machines $host_count"
        train_cmd="$train_cmd --machine_rank $node_rank"
        train_cmd="$train_cmd --main_process_ip $master_addr"
        train_cmd="$train_cmd --main_process_port $master_port"
    fi

    train_cmd="$train_cmd --num_processes $total_processes"
    train_cmd="$train_cmd ${TRAIN_SCRIPT}"
    train_cmd="$train_cmd --config $CONFIG_PATH"

    echo "Command: $train_cmd"
    # Capture exit code explicitly — set -e is intentionally bypassed here so
    # we can still run cleanup via the EXIT trap before propagating the status.
    local colocate_exit=0
    $train_cmd || colocate_exit=$?

    echo "Colocate training finished with exit code: $colocate_exit"

    dump_training_diagnostics "${colocate_exit}"

    # EXIT trap (cleanup_vllm) fires automatically; explicit exit ensures the
    # correct code is returned to SageMaker so the job status is set properly.
    exit $colocate_exit
}

# Print hosts for a SageMaker instance group, one per line. Empty if not found.
# Reads SM_TRAINING_ENV.instance_groups_dict / resource_config.instance_groups
# (list of {instance_group_name, hosts}). algo-1 is not necessarily vLLM.
instance_group_hosts() {
    python3 "${SCRIPT_DIR}/instance_group_hosts.py" "$1" || true
}

# =============================================================================
# TOPOLOGY: MULTI NODE (legacy heterogeneous — kept for backward compat)
# =============================================================================
run_multi_node() {
    local instance_group="${SM_CURRENT_INSTANCE_GROUP:-training}"
    echo "Multi-node mode: instance group = $instance_group"

    if [ "$instance_group" = "vllm" ]; then
        # This node runs the vLLM server on all available GPUs
        local total_gpus
        total_gpus=$(nvidia-smi -L 2>/dev/null | wc -l)

        ALL_GPUS=""
        for i in $(seq 0 $((total_gpus - 1))); do
            [ -n "$ALL_GPUS" ] && ALL_GPUS="$ALL_GPUS,"
            ALL_GPUS="$ALL_GPUS$i"
        done

        echo "Running as vLLM server node (all $total_gpus GPUs)"
        stage_speculative_draft || exit 1
        # Use a bash array so the speculative JSON survives as a single argv entry
        # (no shell re-parse on this path).
        if [ -n "$VLLM_SPECULATIVE_JSON" ]; then
            spec_args=(--speculative-config "$VLLM_SPECULATIVE_JSON")
        else
            spec_args=()
        fi
        if [ "$VLLM_ENABLE_PREFIX_CACHING" = "True" ]; then
            prefix_cache_args=(--enable-prefix-caching yes)
        else
            prefix_cache_args=()
        fi
        # Run in foreground -- SageMaker keeps this alive until the training group finishes.
        # Unset PYTORCH_CUDA_ALLOC_CONF for the vLLM subprocess only — vLLM's
        # custom all-reduce kernel uses CUDA IPC handles which are incompatible
        # with the expandable_segments allocator. `env -u` removes the variable
        # entirely so PyTorch falls back to its built-in default. Training nodes
        # keep the inherited value from the container env.
        # Route through vllm_launcher.py for the same spawn-safety reason
        # documented on start_vllm_server() above.
        env -u PYTORCH_CUDA_ALLOC_CONF \
            VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE=0 \
            VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING=0 \
            CUDA_VISIBLE_DEVICES="$ALL_GPUS" python "${SCRIPT_DIR}/vllm_launcher.py" \
            --model "$MODEL_NAME" \
            --host 0.0.0.0 \
            --port "$VLLM_PORT" \
            --tensor-parallel-size "$VLLM_TP_SIZE" \
            --gpu-memory-utilization "$VLLM_GPU_MEM" \
            "${prefix_cache_args[@]}" \
            "${spec_args[@]}"

    elif [ "$instance_group" = "training" ]; then
        # Resolve vLLM server hostname from SageMaker environment
        local vllm_host=""

        vllm_host=$(instance_group_hosts vllm | head -n 1)

        if [ -z "$vllm_host" ]; then
            echo "ERROR: Could not resolve vLLM server hostname"
            exit 1
        fi

        echo "Running as training node, vLLM server at: $vllm_host:$VLLM_PORT"

        # Wait for vLLM server to be ready
        local train_timeout=$((VLLM_STARTUP_TIMEOUT + 300))
        if ! wait_for_vllm "$vllm_host" "$VLLM_PORT" "$train_timeout"; then
            exit 1
        fi

        # Use all GPUs on this node for training
        local total_gpus
        total_gpus=$(nvidia-smi -L 2>/dev/null | wc -l)

        ALL_GPUS=""
        for i in $(seq 0 $((total_gpus - 1))); do
            [ -n "$ALL_GPUS" ] && ALL_GPUS="$ALL_GPUS,"
            ALL_GPUS="$ALL_GPUS$i"
        done

        launch_training "$ALL_GPUS" "$total_gpus" "$vllm_host"
    else
        echo "ERROR: Unknown instance group '$instance_group'. Expected 'vllm' or 'training'."
        exit 1
    fi
}

# =============================================================================
# TOPOLOGY: DEDICATED VLLM (one node / group vLLM, remaining nodes train)
#
# Homogeneous (default): Node 0 (algo-1) ALL GPUs → vLLM; node 1+ ALL GPUs →
# training. Role assignment is SM_HOSTS rank.
#
# Mixed instance types (opt-in via infra.vllm_instance_type): SageMaker
# InstanceGroups named "vllm" and "training". Role assignment is
# SM_CURRENT_INSTANCE_GROUP. GPU counts may differ per group — always use
# local nvidia-smi. Host discovery reads SM_TRAINING_ENV.instance_groups_dict
# and resource_config.instance_groups (list of {instance_group_name, hosts});
# algo-1 is not necessarily the vLLM node.
#
# Key advantage: with 2 nodes, training is purely single-node so there is ZERO
# cross-node NCCL traffic.  With 3+ nodes, cross-node NCCL runs only between
# training nodes — the vLLM node is fully isolated.
#
# HTTP generation traffic uses standard TCP. TRL server-mode also NCCL-syncs
# weights from trainers to vLLM after each step, so NCCL/EFA env is set on
# BOTH groups.
#
# A lightweight TCP sentinel (port 29600) coordinates shutdown: the first
# training node connects when training finishes, which tells the vLLM node to
# stop vLLM and exit so SageMaker can mark the job complete.
# =============================================================================
SENTINEL_PORT=29600

run_dedicated_vllm() {
    local total_gpus
    total_gpus=$(nvidia-smi -L 2>/dev/null | wc -l)

    local host_count=1
    local node_rank=0
    local master_host=""
    local train_node_count=0
    local train_rank=-1
    local train_master=""
    local instance_group="${SM_CURRENT_INSTANCE_GROUP:-}"

    if [ "$instance_group" = "vllm" ] || [ "$instance_group" = "training" ]; then
        # Heterogeneous InstanceGroups: do not use SM_HOSTS rank (per-group).
        local vllm_hosts_raw train_hosts_raw
        vllm_hosts_raw=$(instance_group_hosts vllm)
        train_hosts_raw=$(instance_group_hosts training)
        if [ -z "$vllm_hosts_raw" ] || [ -z "$train_hosts_raw" ]; then
            echo "ERROR: dedicated_vllm InstanceGroups could not resolve hosts"
            echo "  vLLM group hosts: ${vllm_hosts_raw:-<empty>}"
            echo "  training group hosts: ${train_hosts_raw:-<empty>}"
            python3 "${SCRIPT_DIR}/instance_group_hosts.py" --dump || true
            exit 1
        fi
        master_host=$(printf '%s\n' "$vllm_hosts_raw" | head -n 1)
        train_master=$(printf '%s\n' "$train_hosts_raw" | head -n 1)
        train_node_count=$(printf '%s\n' "$train_hosts_raw" | grep -c .)
        local vllm_node_count
        vllm_node_count=$(printf '%s\n' "$vllm_hosts_raw" | grep -c .)
        host_count=$((vllm_node_count + train_node_count))
        if [ "$instance_group" = "vllm" ]; then
            node_rank=0
            train_rank=-1
        else
            train_rank=$(printf '%s\n' "$train_hosts_raw" | grep -n -x "${SM_CURRENT_HOST}" | head -n 1 | cut -d: -f1)
            if [ -z "$train_rank" ]; then
                echo "ERROR: current host ${SM_CURRENT_HOST} not in training instance group"
                exit 1
            fi
            train_rank=$((train_rank - 1))
            node_rank=$((train_rank + 1))
        fi
        echo "Dedicated vLLM mode (heterogeneous InstanceGroups):"
        echo "  Group: $instance_group  host=${SM_CURRENT_HOST}"
        echo "  vLLM hosts ($vllm_node_count): $(printf '%s' "$vllm_hosts_raw" | tr '\n' ' ')"
        echo "  training hosts ($train_node_count): $(printf '%s' "$train_hosts_raw" | tr '\n' ' ')"
    else
        # Homogeneous cluster: role by SM_HOSTS rank (node 0 = vLLM).
        if [ -n "${SM_HOSTS:-}" ]; then
            host_count=$(python3 -c "import json; print(len(json.loads('${SM_HOSTS}')))")
            master_host=$(python3 -c "import json; print(json.loads('${SM_HOSTS}')[0])")
            if [ "$host_count" -gt 1 ]; then
                node_rank=$(python3 -c "import json; print(json.loads('${SM_HOSTS}').index('${SM_CURRENT_HOST}'))")
            fi
        fi

        if [ "$host_count" -lt 2 ]; then
            echo "ERROR: dedicated_vllm topology requires instance_count >= 2"
            exit 1
        fi

        train_node_count=$((host_count - 1))
        train_rank=$((node_rank - 1))
        train_master=$(python3 -c "import json; print(json.loads('${SM_HOSTS}')[1])")
        echo "Dedicated vLLM mode:"
        echo "  Nodes: $host_count (1 vLLM + $train_node_count training), this node rank: $node_rank"
    fi

    # Build full GPU list (0,1,2,...,N-1) from THIS node — mixed types may
    # have different GPU counts per group.
    local ALL_GPUS=""
    for i in $(seq 0 $((total_gpus - 1))); do
        [ -n "$ALL_GPUS" ] && ALL_GPUS="$ALL_GPUS,"
        ALL_GPUS="$ALL_GPUS$i"
    done

    # Training process count is only consumed on training nodes, where
    # total_gpus is the training instance's GPU count.
    local total_train_procs=$((total_gpus * train_node_count))
    local train_port="${SM_MASTER_PORT:-29500}"

    echo "  GPUs on this node: $total_gpus"
    echo "  vLLM server at $master_host (all GPUs on the vLLM node/group)"
    if [ "$train_node_count" -eq 1 ]; then
        echo "  Training: $train_master (single-node, all local GPUs)"
    else
        echo "  Training: $train_node_count nodes, $total_train_procs processes (this node local GPUs × training nodes)"
        echo "  Training master: $train_master:$train_port"
    fi

    # ── NCCL transport (required on ALL nodes) ────────────────────────
    # TRL's server mode uses NCCL to push updated weights from training
    # to the vLLM server after each training step.  This cross-node NCCL
    # link must be configured on both the vLLM node and training nodes.
    NCCL_TRANSPORT=$(yaml_get "infra.nccl_transport" "efa")
    echo "NCCL transport: $NCCL_TRANSPORT"
    if [ "$NCCL_TRANSPORT" = "sockets" ]; then
        export NCCL_NET="Socket"
        export NCCL_NET_PLUGIN=""
        export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
    else
        export FI_PROVIDER="${FI_PROVIDER:-efa}"
        export FI_EFA_USE_DEVICE_RDMA="${FI_EFA_USE_DEVICE_RDMA:-1}"
        export FI_EFA_FORK_SAFE="${FI_EFA_FORK_SAFE:-1}"
        export RDMAV_FORK_SAFE="${RDMAV_FORK_SAFE:-1}"
        export NCCL_PROTO="${NCCL_PROTO:-simple}"
        export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
    fi
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800000}"
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
    export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"

    # ── Node 0: vLLM server ───────────────────────────────────────────
    if [ "$node_rank" -eq 0 ]; then
        start_vllm_server "$ALL_GPUS"
        if ! wait_for_vllm "localhost" "$VLLM_PORT" 1800; then
            echo "vLLM server failed to start"
            kill "$VLLM_PID" 2>/dev/null || true
            exit 1
        fi

        echo "vLLM server healthy. Waiting for training to signal completion on port $SENTINEL_PORT..."

        python3 -c "
import socket
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('0.0.0.0', $SENTINEL_PORT))
s.settimeout(90000)  # 25h — longer than any job
s.listen(1)
try:
    conn, addr = s.accept()
    conn.close()
    print(f'Received completion signal from {addr}')
except socket.timeout:
    print('Sentinel timed out')
s.close()
" 2>&1

        echo "Training complete — shutting down vLLM server"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        exit 0

    # ── Node 1+: training ─────────────────────────────────────────────
    else
        local train_timeout=$((VLLM_STARTUP_TIMEOUT + 300))
        echo "Waiting for vLLM server on $master_host:$VLLM_PORT (timeout: ${train_timeout}s)..."
        if ! wait_for_vllm "$master_host" "$VLLM_PORT" "$train_timeout"; then
            echo "ERROR: vLLM server on $master_host did not become healthy"
            exit 1
        fi

        local train_exit=0

        if [ "$train_node_count" -eq 1 ]; then
            echo "Launching single-node training on $total_gpus GPUs, vLLM at $master_host"
            launch_training "$ALL_GPUS" "$total_gpus" "$master_host"
            train_exit=$?
        else
            # Multi-node distributed training across training nodes
            echo "Multi-node training: $train_node_count nodes, rank $train_rank, master=$train_master:$train_port"
            echo "Total training processes: $total_train_procs"

            mkdir -p "${TORCH_RANK_LOG_DIR}"

            local train_cmd="accelerate launch --log_dir ${TORCH_RANK_LOG_DIR} --tee 3"
            train_cmd="$train_cmd --mixed_precision bf16"
            train_cmd="$train_cmd --num_machines $train_node_count"
            train_cmd="$train_cmd --machine_rank $train_rank"
            train_cmd="$train_cmd --main_process_ip $train_master"
            train_cmd="$train_cmd --main_process_port $train_port"
            train_cmd="$train_cmd --num_processes $total_train_procs"
            train_cmd="$train_cmd ${TRAIN_SCRIPT}"
            train_cmd="$train_cmd --config $CONFIG_PATH"
            train_cmd="$train_cmd --vllm-host $master_host"

            echo "Command: CUDA_VISIBLE_DEVICES=$ALL_GPUS $train_cmd"
            CUDA_VISIBLE_DEVICES="$ALL_GPUS" $train_cmd || train_exit=$?
        fi

        # Only the first training node (train_rank 0) signals vLLM to shut down.
        # Other training nodes exit immediately after training completes.
        if [ "$train_rank" -eq 0 ]; then
            echo "Signaling vLLM node ($master_host:$SENTINEL_PORT) to shut down..."
            python3 -c "
import socket
try:
    s = socket.socket()
    s.settimeout(30)
    s.connect(('$master_host', $SENTINEL_PORT))
    s.close()
    print('Completion signal sent')
except Exception as e:
    print(f'Warning: could not signal vLLM node: {e}')
" 2>&1
        fi

        echo "Training finished with exit code: $train_exit"
        exit $train_exit
    fi
}

# =============================================================================
# TOPOLOGY: MULTI NODE SERVER (homogeneous cluster, GPU split per node)
#
# Every node in a homogeneous SageMaker cluster runs the same entrypoint.
# Node 0 (algo-1): vLLM server on GPUs 0..(TP-1), training on remaining GPUs.
# Other nodes:     training only on the same GPU range (lower GPUs idle).
# All training ranks connect to the vLLM server on node 0 via HTTP.
#
# This avoids the EFA-handshake issues of colocate mode because vLLM runs as
# a separate process with its own CUDA context — NCCL is used only for
# training communication.
# =============================================================================
run_multi_node_server() {
    local total_gpus
    total_gpus=$(nvidia-smi -L 2>/dev/null | wc -l)

    local vllm_gpu_count
    vllm_gpu_count=$(yaml_get "infra.vllm_gpu_count" "$VLLM_TP_SIZE")
    local train_gpu_count=$((total_gpus - vllm_gpu_count))

    # Detect multi-node from SageMaker environment
    local host_count=1
    local node_rank=0
    local master_addr="localhost"
    local master_port="${SM_MASTER_PORT:-29500}"
    local master_host=""

    if [ -n "${SM_HOSTS:-}" ]; then
        host_count=$(python3 -c "import json; print(len(json.loads('${SM_HOSTS}')))")
        master_host=$(python3 -c "import json; print(json.loads('${SM_HOSTS}')[0])")
        if [ "$host_count" -gt 1 ]; then
            master_addr="$master_host"
            node_rank=$(python3 -c "import json; print(json.loads('${SM_HOSTS}').index('${SM_CURRENT_HOST}'))")
        fi
    fi

    # Build GPU ranges: vLLM gets low GPUs, training gets the rest.
    build_gpu_ranges "$vllm_gpu_count" "$train_gpu_count"

    local total_train_procs=$((train_gpu_count * host_count))

    echo "Multi-node server mode:"
    echo "  Nodes: $host_count, this node rank: $node_rank"
    echo "  GPUs per node: $total_gpus (vLLM: $vllm_gpu_count, training: $train_gpu_count)"
    echo "  Total training processes: $total_train_procs"
    echo "  vLLM server on node 0 ($master_host), GPUs: $VLLM_GPUS"

    # ── NCCL transport (same as colocate) ──────────────────────────────
    NCCL_TRANSPORT=$(yaml_get "infra.nccl_transport" "efa")
    echo "NCCL transport: $NCCL_TRANSPORT"

    if [ "$NCCL_TRANSPORT" = "sockets" ]; then
        export NCCL_NET="Socket"
        export NCCL_NET_PLUGIN=""
        export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
    else
        export FI_PROVIDER="${FI_PROVIDER:-efa}"
        export FI_EFA_USE_DEVICE_RDMA="${FI_EFA_USE_DEVICE_RDMA:-1}"
        export FI_EFA_FORK_SAFE="${FI_EFA_FORK_SAFE:-1}"
        export RDMAV_FORK_SAFE="${RDMAV_FORK_SAFE:-1}"
        export NCCL_PROTO="${NCCL_PROTO:-simple}"
        export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
    fi
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800000}"
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
    export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"

    mkdir -p "${TORCH_RANK_LOG_DIR}"

    # ── Node 0: start vLLM server, then training ──────────────────────
    if [ "$node_rank" -eq 0 ]; then
        start_vllm_server "$VLLM_GPUS"
        if ! wait_for_vllm "localhost" "$VLLM_PORT" 1800; then
            echo "Killing vLLM server (PID: $VLLM_PID)"
            kill "$VLLM_PID" 2>/dev/null || true
            exit 1
        fi
    else
        # Other nodes: wait for vLLM on node 0
        local train_timeout=$((VLLM_STARTUP_TIMEOUT + 300))
        echo "Waiting for vLLM server on $master_host:$VLLM_PORT (timeout: ${train_timeout}s)..."
        if ! wait_for_vllm "$master_host" "$VLLM_PORT" "$train_timeout"; then
            echo "ERROR: vLLM server on $master_host did not become healthy"
            exit 1
        fi
    fi

    # ── Launch distributed training on the training GPUs ──────────────
    local train_cmd="accelerate launch --log_dir ${TORCH_RANK_LOG_DIR} --tee 3"
    train_cmd="$train_cmd --mixed_precision bf16"

    if [ "$host_count" -gt 1 ]; then
        train_cmd="$train_cmd --num_machines $host_count"
        train_cmd="$train_cmd --machine_rank $node_rank"
        train_cmd="$train_cmd --main_process_ip $master_addr"
        train_cmd="$train_cmd --main_process_port $master_port"
    fi

    train_cmd="$train_cmd --num_processes $total_train_procs"
    train_cmd="$train_cmd ${TRAIN_SCRIPT}"
    train_cmd="$train_cmd --config $CONFIG_PATH"
    train_cmd="$train_cmd --vllm-host $master_host"

    echo "Command: CUDA_VISIBLE_DEVICES=$TRAIN_GPUS $train_cmd"

    local train_exit=0
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" $train_cmd || train_exit=$?

    echo "Training finished with exit code: $train_exit"
    dump_training_diagnostics "${train_exit}"
    exit $train_exit
}

# =============================================================================
# VLLM MODE: NONE (torchrun — no vLLM, native HF generation)
#
# Uses torchrun to launch the training script directly.  TRL falls back to
# the model's own generate() method for GRPO completions — simpler setup,
# no vLLM server lifecycle to manage, all GPUs available for training.
#
# Suitable for smaller models or environments where vLLM adds unwanted
# complexity (e.g. CUDA graph compilation issues, memory contention).
#
# Multi-node is auto-detected from SageMaker environment variables.
# =============================================================================
run_torchrun() {
    local total_gpus
    total_gpus=$(nvidia-smi -L 2>/dev/null | wc -l)

    # Detect multi-node from SageMaker environment
    local host_count=1
    local node_rank=0
    local master_addr="localhost"
    local master_port="${SM_MASTER_PORT:-29500}"

    if [ -n "${SM_HOSTS:-}" ]; then
        host_count=$(python3 -c "import json; print(len(json.loads('${SM_HOSTS}')))")
        if [ "$host_count" -gt 1 ]; then
            master_addr=$(python3 -c "import json; print(json.loads('${SM_HOSTS}')[0])")
            node_rank=$(python3 -c "import json; print(json.loads('${SM_HOSTS}').index('${SM_CURRENT_HOST}'))")
        fi
    fi

    echo "Torchrun mode (no vLLM): ${host_count} node(s), ${total_gpus} GPUs/node"

    # ── NCCL configuration for multi-node ────────────────────────────
    # Without vLLM, generation runs on the training ranks and can block
    # for minutes per batch.  Use 2-hour timeouts so NCCL connections survive.
    if [ "$host_count" -gt 1 ]; then
        NCCL_TRANSPORT=$(yaml_get "infra.nccl_transport" "efa")
        echo "NCCL transport: $NCCL_TRANSPORT"

        if [ "$NCCL_TRANSPORT" = "sockets" ]; then
            export NCCL_NET="Socket"
            export NCCL_NET_PLUGIN=""
            export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
        else
            export FI_PROVIDER="${FI_PROVIDER:-efa}"
            export FI_EFA_USE_DEVICE_RDMA="${FI_EFA_USE_DEVICE_RDMA:-1}"
            export FI_EFA_FORK_SAFE="${FI_EFA_FORK_SAFE:-1}"
            export RDMAV_FORK_SAFE="${RDMAV_FORK_SAFE:-1}"
            export NCCL_PROTO="${NCCL_PROTO:-Simple}"
            export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
        fi

        export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
        export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
        export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
        export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
        export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-0}"
    fi

    mkdir -p "${TORCH_RANK_LOG_DIR}"

    # ── Build torchrun command ────────────────────────────────────────
    local train_cmd="torchrun"
    train_cmd="$train_cmd --nproc_per_node=$total_gpus"
    train_cmd="$train_cmd --log_dir ${TORCH_RANK_LOG_DIR} --redirects 3"

    if [ "$host_count" -gt 1 ]; then
        echo "Multi-node torchrun: rank=$node_rank, master=$master_addr:$master_port"
        train_cmd="$train_cmd --nnodes=$host_count"
        train_cmd="$train_cmd --node_rank=$node_rank"
        train_cmd="$train_cmd --master_addr=$master_addr"
        train_cmd="$train_cmd --master_port=$master_port"
    fi

    train_cmd="$train_cmd ${TRAIN_SCRIPT}"
    train_cmd="$train_cmd --config $CONFIG_PATH"

    echo "Command: $train_cmd"

    local torchrun_exit=0
    $train_cmd || torchrun_exit=$?

    echo "Torchrun training finished with exit code: $torchrun_exit"

    dump_training_diagnostics "${torchrun_exit}"

    exit $torchrun_exit
}

# =============================================================================
# MAIN
# =============================================================================

echo "Starting with topology: $TOPOLOGY, vllm_mode: $VLLM_MODE"

if [ "$VLLM_MODE" = "none" ]; then
    run_torchrun
elif [ "$VLLM_MODE" = "colocate" ]; then
    run_colocate
else
    case "$TOPOLOGY" in
        single_node)
            run_single_node
            ;;
        multi_node)
            run_multi_node
            ;;
        multi_node_server)
            run_multi_node_server
            ;;
        dedicated_vllm)
            run_dedicated_vllm
            ;;
        *)
            echo "ERROR: Unknown topology '$TOPOLOGY'. Expected 'single_node', 'multi_node', 'multi_node_server', or 'dedicated_vllm'."
            exit 1
            ;;
    esac
fi