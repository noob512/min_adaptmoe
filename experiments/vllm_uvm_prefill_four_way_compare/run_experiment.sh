#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_original_va_graph_integration/run_integration.sh"

export RUNNER_OVERRIDE="${SCRIPT_DIR}/benchmark_compare_steps.py"
export VERIFY_OVERRIDE="${SCRIPT_DIR}/verify_experiment.py"
export TOP_LEVEL_HARNESS="${SCRIPT_DIR}/run_experiment.sh"
export ATTENTION_BACKEND="TRITON_ATTN"
export EXTENDED_PREFILL_MATRIX=1

mode="${1:-help}"
case "${mode}" in
  static-preflight)
    "${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python" -m pytest -q \
      "${SCRIPT_DIR}/test_verify_experiment.py"
    exec "${BASE_HARNESS}" static-preflight
    ;;
  smoke)
    export ORDER="${ORDER:-G}"
    export REQUIRED_ARMS="${REQUIRED_ARMS:-G}"
    ;;
  qualify)
    export ORDER="${ORDER:-NAGRCF}"
    export REQUIRED_ARMS="${REQUIRED_ARMS:-NAGRCF}"
    ;;
  formal|run)
    # Three counterbalanced blocks. F is the native FULL-Graph correctness
    # control for G; it is not one of the four reported prefill mechanisms.
    export ORDER="${ORDER:-NAGRCFFCRGANGFNCRA}"
    export REQUIRED_ARMS="${REQUIRED_ARMS:-NAGRCF}"
    ;;
  verify)
    [[ $# -eq 2 ]] || { echo "verify requires RUN_ROOT" >&2; exit 2; }
    exec "${BASE_HARNESS}" verify "$2"
    ;;
  *)
    echo "usage: $0 {static-preflight|smoke|qualify|formal|verify RUN_ROOT}" >&2
    exit 2
    ;;
esac

if [[ -z "${RUN_ROOT:-}" ]]; then
  export RUN_ROOT="${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_${mode}"
fi
exec "${BASE_HARNESS}" "${mode}"
