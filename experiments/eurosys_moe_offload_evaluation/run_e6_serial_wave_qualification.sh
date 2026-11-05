#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
DIFFERENCE_GENERATOR="${SCRIPT_DIR}/generate_e6_difference_plans.py"
SERIAL_GENERATOR="${SCRIPT_DIR}/generate_e6_serial_wave_plans.py"
VERIFIER="${SCRIPT_DIR}/verify_e6_serial_wave_qualification.py"
BASE_RUNNER="${REPO_ROOT}/experiments/hybrid_offload_2_4_6_8_compare/run_graph_plan_class_sequence.sh"
BASE_PLAN="${REPO_ROOT}/experiments/hybrid_offload_2_4_6_8_compare/inputs/cold_plan_8g.json"
NAMES="${REPO_ROOT}/experiments/hybrid_offload_2_4_6_8_compare/inputs/exact_parameter_names.txt"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/e6-serial-wave-$(date -u +%Y%m%dT%H%M%SZ)-qualification}"

die() { echo "E6 serial-wave qualification: $*" >&2; exit 1; }
[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
for path in "${PYTHON}" "${DIFFERENCE_GENERATOR}" "${SERIAL_GENERATOR}" \
  "${VERIFIER}" "${BASE_RUNNER}" "${BASE_PLAN}" "${NAMES}"; do
  [[ -e "${path}" ]] || die "missing input: ${path}"
done
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"

"${PYTHON}" "${DIFFERENCE_GENERATOR}" --base-plan "${BASE_PLAN}" \
  --output-dir "${RUN_ROOT}/inputs" >"${RUN_ROOT}/plan_generation.stdout.json"
"${PYTHON}" "${SERIAL_GENERATOR}" --input-dir "${RUN_ROOT}/inputs" \
  >"${RUN_ROOT}/serial_generation.stdout.json"

"${PYTHON}" - "${RUN_ROOT}/run_config.json" <<'PY'
import json, time, sys
payload = {
    "schema": "eurosys_e6_serial_wave_qualification_run_v1",
    "created_at": time.time(),
    "process_blocks": 1,
    "percentages": [5, 10, 25, 50, 100],
    "wave_counts": [1, 2, 5, 10, 19],
    "model": "DeepSeek-V2-Lite-Chat BF16",
    "kv_cache_bytes": 268435456,
    "slot_budget_bytes": 10737418240,
    "max_tokens": 8,
    "seed": 20260828,
}
open(sys.argv[1], "x", encoding="utf-8").write(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY

printf 'block\tdirectory\treturncode\n' >"${RUN_ROOT}/execution_status.tsv"
block_root="${RUN_ROOT}/block_01"
serial_plan_classes="${RUN_ROOT}/inputs/serial_plan_classes.json"
serial_transition_paths="${RUN_ROOT}/inputs/serial_transition_paths.json"
set +e
RUN_ROOT="${block_root}" \
PLAN_CLASSES="${serial_plan_classes}" \
TRANSITION_PATHS="${serial_transition_paths}" \
NAMES="${NAMES}" \
PLAN_CLASS_SEQUENCE="base,d005,base,d010,base,d025,base,d050,base,d100,base" \
MAX_TOKENS=8 \
KV_CACHE_BYTES=268435456 \
SLOT_BUDGET_BYTES=10737418240 \
SEED=20260828 \
ALLOW_GPU_RUN=1 \
  bash "${BASE_RUNNER}" >"${RUN_ROOT}/block_01.stdout.log" 2>&1
rc=$?
set -e
printf '1\tblock_01\t%d\n' "${rc}" >>"${RUN_ROOT}/execution_status.tsv"
(( rc == 0 )) || die "block failed; see block_01.stdout.log"

"${PYTHON}" "${VERIFIER}" "${RUN_ROOT}" --output "${RUN_ROOT}/verification.json"
echo "RUN_ROOT=${RUN_ROOT}"
