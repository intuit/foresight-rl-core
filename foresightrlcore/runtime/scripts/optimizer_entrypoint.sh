#!/bin/bash
# =============================================================================
# SageMaker Optimizer Entrypoint
#
# Runs inside either the DJL inference image (GPU / local-serving path, vLLM
# baked in) or a CPU image (API-only path). Responsibilities, in order:
#
#   1. Install optimizer Python requirements (pandas, pyarrow, langchain_openai,
#      boto3, requests, pydantic, pyyaml).
#   2. Resolve secrets via IDPS when a `secrets:` section is present (so genos
#      auth works), sourcing them into the environment.
#   3. Hand off to foresightrlcore.optimizer.runner, which resolves records,
#      runs the cross-product evaluation, and writes results + summary.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================="
echo "SageMaker Optimizer Entrypoint"
echo "Host: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "============================================="

CONFIG_PATH="${OPTIMIZER_CONFIG:-/opt/ml/input/data/config/args.yaml}"
OUTPUT_DIR="${OPTIMIZER_OUTPUT_DIR:-/opt/ml/output/data}"
CHECKPOINT_DIR="${OPTIMIZER_CHECKPOINT_DIR:-/opt/ml/checkpoints}"

mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR"

# ── Install optimizer Python requirements ────────────────────────────────
REQ_FILE="${SCRIPT_DIR}/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    echo "Installing optimizer Python requirements from $REQ_FILE..."
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

echo "Launching optimizer runner: python -m foresightrlcore.optimizer.runner ${RUNNER_ARGS[*]}"
exec python -m foresightrlcore.optimizer.runner "${RUNNER_ARGS[@]}"
