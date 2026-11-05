#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_RUNNER="${REPO_ROOT}/experiments/vllm_prefill_p3_pinned_uva_compare/run_compare.sh"

export BENCHMARK="${SCRIPT_DIR}/benchmark_steady_batches.py"
export VERIFY="${SCRIPT_DIR}/verify_steady_compare.py"
export ORDER="${ORDER:-PCUCUPUPCUCPPUCCPU}"
export CONCURRENCY="${CONCURRENCY:-4}"
export WARMUP_REQUESTS="${WARMUP_REQUESTS:-4}"
export DISABLE_CHUNKED_PREFILL="${DISABLE_CHUNKED_PREFILL:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
export STEADY_CONDITIONING_BATCHES="${STEADY_CONDITIONING_BATCHES:-3}"
export STEADY_MEASUREMENT_BATCHES="${STEADY_MEASUREMENT_BATCHES:-5}"
export STEADY_INTER_BATCH_IDLE_MS="${STEADY_INTER_BATCH_IDLE_MS:-0}"
export RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_steady6}"

exec "${BASE_RUNNER}" "$@"
