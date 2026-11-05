#!/usr/bin/env bash
set -euo pipefail

E10_COMPARE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E10_COMPARE_ROOT="$(cd "${E10_COMPARE_DIR}/../.." && pwd)"
SINGLE_RUN="${E10_COMPARE_DIR}/run_experiment.sh"
COMPARE="${E10_COMPARE_DIR}/compare_trigger_modes.py"
VLLM_PY="${E10_COMPARE_ROOT}/workloads/vllm/vllm/.venv/bin/python"

die_compare() {
  echo "E10 trigger comparison: $*" >&2
  exit 1
}

static_preflight() {
  bash -n "${BASH_SOURCE[0]}"
  "${VLLM_PY}" -m py_compile "${COMPARE}"
  E10_TRIGGER_MODE=pressure bash "${SINGLE_RUN}" static-preflight
  E10_TRIGGER_MODE=direct bash "${SINGLE_RUN}" static-preflight
  echo "E10 trigger comparison static preflight: OK"
}

run_comparison() {
  [[ "${ALLOW_GPU_RUN:-0}" == 1 ]] \
    || die_compare "GPU execution requires ALLOW_GPU_RUN=1"
  static_preflight
  local run_root
  run_root="${RUN_ROOT:-${E10_COMPARE_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_trigger_comparison}"
  [[ ! -e "${run_root}" ]] || die_compare "RUN_ROOT already exists: ${run_root}"
  mkdir -p "${run_root}"
  run_root="$(cd "${run_root}" && pwd)"

  "${VLLM_PY}" - "${run_root}/protocol.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "schema": "eurosys_e10_trigger_mode_comparison_protocol_v1",
    "qualification_only": True,
    "arms": ["auto_pressure", "direct_tier"],
    "arm_order": ["auto_pressure", "direct_tier"],
    "independent_cold_processes": True,
    "complete_sequences_per_arm": 1,
    "statistical_repetitions": 0,
    "automatic_retry": False,
    "shared_tiers_gib": [2, 4, 6],
    "shared_target_blocks": [1364, 2730, 4095],
    "shared_plan_classes": ["cold70", "cold80", "cold89"],
    "comparison_contract": {
        "transition_cost": "compare separately",
        "prefill": "auto compute excluding in-Prefill transition vs direct request Prefill",
        "decode": "compare Prefill-to-Decode switch and steady Decode separately",
    },
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

  local auto_rc=0 direct_rc=0
  set +e
  RUN_ROOT="${run_root}/auto_pressure" \
    E10_TRIGGER_MODE=pressure \
    ALLOW_GPU_RUN=1 \
    bash "${SINGLE_RUN}" qualify
  auto_rc=$?
  RUN_ROOT="${run_root}/direct_tier" \
    E10_TRIGGER_MODE=direct \
    ALLOW_GPU_RUN=1 \
    bash "${SINGLE_RUN}" qualify
  direct_rc=$?
  set -e
  printf '%s\n' "${auto_rc}" >"${run_root}/auto_pressure.returncode.txt"
  printf '%s\n' "${direct_rc}" >"${run_root}/direct_tier.returncode.txt"
  (( auto_rc == 0 )) || die_compare "auto-pressure arm failed; it was not retried"
  (( direct_rc == 0 )) || die_compare "direct-tier arm failed; it was not retried"

  "${VLLM_PY}" "${COMPARE}" "${run_root}" \
    --output "${run_root}/comparison.json" \
    >"${run_root}/compare.log" 2>"${run_root}/compare.err"
  echo "RUN_ROOT=${run_root}"
  echo "comparison=${run_root}/comparison.json"
}

case "${1:-help}" in
  static-preflight)
    static_preflight
    ;;
  qualify|qualification|run)
    run_comparison
    ;;
  compare)
    [[ $# -eq 2 ]] || die_compare "compare requires RUN_ROOT"
    "${VLLM_PY}" "${COMPARE}" "$2" --output "$2/comparison.json"
    ;;
  *)
    echo "usage: $0 {static-preflight|qualify|compare RUN_ROOT}" >&2
    exit 2
    ;;
esac
