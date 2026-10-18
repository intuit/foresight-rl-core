#!/bin/bash
# =============================================================================
# SageMaker Evaluate Entrypoint
#
# Runs inside either the DJL inference image (GPU path -- a candidate is served
# locally with vLLM) or a CPU image (API-only path). Responsibilities, in order:
#
#   1. Install evaluate Python requirements (pandas, pyarrow, boto3, ...).
#   2. Resolve secrets via IDPS when a `secrets:` section is present, so a
#      provider that needs credentials can authenticate.
#   3. Hand off to foresightrlcore.evaluate.runner, which brings up an endpoint
#      per model, runs the benchmarks, and writes the report.
#
# Benchmarks that stage their own source and their own virtualenv do that from
# inside the runner -- nothing benchmark-specific belongs in this script.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================="
echo "SageMaker Evaluate Entrypoint"
echo "Host: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "============================================="

CONFIG_PATH="${EVALUATE_CONFIG:-/opt/ml/input/data/config/args.yaml}"
OUTPUT_DIR="${EVALUATE_OUTPUT_DIR:-/opt/ml/output/data}"
CHECKPOINT_DIR="${EVALUATE_CHECKPOINT_DIR:-/opt/ml/checkpoints}"

mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR"

# ── Install evaluate Python requirements ─────────────────────────────────
# CUDA_VISIBLE_DEVICES="" keeps a package's CUDA probe from leaving a parent
# CUDA context behind, which breaks vLLM's spawned workers on the GPU path.
REQ_FILE="${SCRIPT_DIR}/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    echo "Installing evaluate Python requirements from $REQ_FILE..."
    CUDA_VISIBLE_DEVICES="" pip install --no-cache-dir -r "$REQ_FILE"
else
    echo "WARNING: No requirements.txt at $REQ_FILE -- skipping pip install"
fi

# ── Resolve secrets (only when a secrets section is present) ─────────────
# IDPS is implemented by foresightrl_intuit. This script does not install
# a secrets client and does not name an internal package index.
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

# ── Dispatch to the Python orchestrator ──────────────────────────────────
RUNNER_ARGS=(
    --config "$CONFIG_PATH"
    --output-dir "$OUTPUT_DIR"
    --checkpoint-dir "$CHECKPOINT_DIR"
)

echo "Launching evaluate runner: python -m foresightrlcore.evaluate.runner ${RUNNER_ARGS[*]}"
exec python -m foresightrlcore.evaluate.runner "${RUNNER_ARGS[@]}"
