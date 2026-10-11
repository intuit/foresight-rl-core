#!/bin/bash
# =============================================================================
# SageMaker Perf-Test Entrypoint
#
# Runs inside the DJL inference base image (Java + vLLM + CUDA already
# baked in). Responsibilities, in order:
#
#   1. Install Gradle 8.10 if not present.
#   2. Install perf-test Python requirements (lightweight: requests, psutil,
#      pydantic, pyyaml, boto3, py-markdown-table).
#   3. Re-export any vllm_env values forwarded by the launcher.
#   4. Hand off to foresightrlcore.perf_test.runner, which starts
#      vllm serve, polls /health, runs Gatling, and writes results.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================="
echo "SageMaker Perf-Test Entrypoint"
echo "Host: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "============================================="

# ── Install Gradle 8.10 (~30s; Java already in DJL base) ────────────────
if ! command -v gradle >/dev/null 2>&1; then
    echo "Installing Gradle 8.10..."
    GRADLE_DIST_URL="https://services.gradle.org/distributions/gradle-8.10-bin.zip"
    GRADLE_ZIP="/tmp/gradle-8.10-bin.zip"
    wget -q -O "$GRADLE_ZIP" "$GRADLE_DIST_URL"
    unzip -qd /opt "$GRADLE_ZIP"
    ln -sfn /opt/gradle-8.10 /opt/gradle
    rm -f "$GRADLE_ZIP"
    export PATH="/opt/gradle/bin:$PATH"
    echo "Gradle installed: $(gradle --version | head -3)"
else
    echo "Gradle already installed: $(gradle --version | head -1)"
fi

# ── Install perf-test Python requirements ────────────────────────────────
REQ_FILE="${SCRIPT_DIR}/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    echo "Installing perf-test Python requirements from $REQ_FILE..."
    CUDA_VISIBLE_DEVICES="" pip install --no-cache-dir -r "$REQ_FILE"
else
    echo "WARNING: No requirements.txt at $REQ_FILE -- skipping pip install"
fi

# ── Re-export vllm_env values forwarded by the launcher ──────────────────
# The launcher serialises VllmServingConfig.vllm_env into
# PERF_TEST_VLLM_ENV_JSON so we can export each KEY=VALUE pair into the
# environment that runner.py inherits when it launches vllm serve.
if [ -n "${PERF_TEST_VLLM_ENV_JSON:-}" ]; then
    echo "Exporting vllm_env from PERF_TEST_VLLM_ENV_JSON..."
    eval "$(python3 -c '
import json, os, sys, shlex
data = json.loads(os.environ["PERF_TEST_VLLM_ENV_JSON"])
for k, v in data.items():
    sys.stdout.write(f"export {k}={shlex.quote(str(v))}\n")
')"
fi

# ── Dispatch to the Python orchestrator ──────────────────────────────────
CONFIG_PATH="${PERF_TEST_CONFIG:-/opt/ml/input/data/config/args.yaml}"
DATA_CHANNEL="${PERF_TEST_DATA_CHANNEL:-/opt/ml/input/data/perf_data}"
OUTPUT_DIR="${PERF_TEST_OUTPUT_DIR:-/opt/ml/output/data}"
MODEL_PATH="${PERF_TEST_MODEL_PATH:-/opt/ml/input/data/model}"

mkdir -p "$OUTPUT_DIR"

RUNNER_ARGS=(
    --config "$CONFIG_PATH"
    --data-channel "$DATA_CHANNEL"
    --output-dir "$OUTPUT_DIR"
    --model-path "$MODEL_PATH"
)

echo "Launching perf-test runner: python -m foresightrlcore.perf_test.runner ${RUNNER_ARGS[*]}"
exec python -m foresightrlcore.perf_test.runner "${RUNNER_ARGS[@]}"
