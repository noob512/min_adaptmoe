#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
E10_RUNNER="${REPO_ROOT}/experiments/vllm_uvm_e10_multilevel_pressure/run_experiment.sh"
INPUT_SET="${REPO_ROOT}/experiments/vllm_uvm_e10_multilevel_pressure/inputs/formal_k2_k4_k6_k8_k10"

[[ -x "${E10_RUNNER}" ]] || {
  echo "missing E10 runner: ${E10_RUNNER}" >&2
  exit 1
}
[[ -f "${INPUT_SET}/plan_classes.json" ]] || {
  echo "missing frozen E10 plan classes: ${INPUT_SET}" >&2
  exit 1
}

export E10_INPUT_SET_DIR="${INPUT_SET}"
export E10_TIER_SPECS="2:1364:cold70,4:2730:cold80,8:5460:cold99,10:6826:cold108"
export KV_MAX_BYTES=10737942528
export PLAN_CLASS_MAP='{"1364":"cold70","2730":"cold80","5460":"cold99","6826":"cold108"}'
export BACKUP_RESERVE_MAP='{"1364":1348,"2730":2714,"5460":5444,"6826":6810}'
export E10_TRIGGER_MODE=pressure

exec "${E10_RUNNER}" "${1:-qualify}"
