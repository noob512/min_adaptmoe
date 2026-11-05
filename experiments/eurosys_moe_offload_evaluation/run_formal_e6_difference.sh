#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
GENERATOR="${SCRIPT_DIR}/generate_e6_difference_plans.py"
VERIFIER="${SCRIPT_DIR}/verify_e6_difference_formal.py"
BASE_RUNNER="${REPO_ROOT}/experiments/hybrid_offload_2_4_6_8_compare/run_graph_plan_class_sequence.sh"
BASE_PLAN="${REPO_ROOT}/experiments/hybrid_offload_2_4_6_8_compare/inputs/cold_plan_8g.json"
NAMES="${REPO_ROOT}/experiments/hybrid_offload_2_4_6_8_compare/inputs/exact_parameter_names.txt"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/e6-difference-$(date -u +%Y%m%dT%H%M%SZ)}"
PROCESS_BLOCKS="${PROCESS_BLOCKS:-10}"
MODE="${MODE:-formal}"

die() { echo "E6 difference formal: $*" >&2; exit 1; }
[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
case "${MODE}" in
  qualification) minimum_blocks=1 ;;
  formal) minimum_blocks=10 ;;
  *) die "MODE must be qualification or formal" ;;
esac
[[ "${PROCESS_BLOCKS}" =~ ^[0-9]+$ ]] && (( PROCESS_BLOCKS >= minimum_blocks )) \
  || die "PROCESS_BLOCKS must be an integer >=${minimum_blocks} for MODE=${MODE}"
for path in "${PYTHON}" "${GENERATOR}" "${VERIFIER}" "${BASE_RUNNER}" "${BASE_PLAN}" "${NAMES}"; do
  [[ -e "${path}" ]] || die "missing input: ${path}"
done
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"

"${PYTHON}" "${GENERATOR}" --base-plan "${BASE_PLAN}" \
  --output-dir "${RUN_ROOT}/inputs" >"${RUN_ROOT}/plan_generation.stdout.json"
PLAN_CLASSES="${RUN_ROOT}/inputs/plan_classes.json"

"${PYTHON}" - "${RUN_ROOT}/run_config.json" "${PROCESS_BLOCKS}" "${MODE}" \
  "${BASE_PLAN}" "${PLAN_CLASSES}" <<'PY'
import hashlib, json, os, sys, time
path, blocks, mode, base, classes = sys.argv[1:]
payload = {
    "schema": "eurosys_e6_difference_formal_run_v1",
    "created_at": time.time(),
    "mode": mode,
    "process_blocks": int(blocks),
    "percentages": [5, 10, 25, 50, 100],
    "balanced_order": "odd ascending, even descending; base between differences",
    "base_plan": os.path.realpath(base),
    "base_plan_sha256": hashlib.sha256(open(base, "rb").read()).hexdigest(),
    "plan_classes": os.path.realpath(classes),
    "plan_classes_sha256": hashlib.sha256(open(classes, "rb").read()).hexdigest(),
    "model": "DeepSeek-V2-Lite-Chat BF16",
    "kv_cache_bytes": 268435456,
    "slot_budget_bytes": 10737418240,
    "max_tokens": 8,
}
open(path, "x", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

printf 'block\tdirectory\treturncode\n' >"${RUN_ROOT}/execution_status.tsv"
ascending="base,d005,base,d010,base,d025,base,d050,base,d100,base"
descending="base,d100,base,d050,base,d025,base,d010,base,d005,base"
for ((block=1; block<=PROCESS_BLOCKS; ++block)); do
  sequence="${ascending}"
  (( block % 2 == 0 )) && sequence="${descending}"
  directory="block_$(printf '%02d' "${block}")"
  block_root="${RUN_ROOT}/${directory}"
  set +e
  RUN_ROOT="${block_root}" \
  PLAN_CLASSES="${PLAN_CLASSES}" \
  NAMES="${NAMES}" \
  PLAN_CLASS_SEQUENCE="${sequence}" \
  MAX_TOKENS=8 \
  KV_CACHE_BYTES=268435456 \
  SLOT_BUDGET_BYTES=10737418240 \
  SEED=20260828 \
  ALLOW_GPU_RUN=1 \
    bash "${BASE_RUNNER}" >"${RUN_ROOT}/${directory}.stdout.log" 2>&1
  rc=$?
  set -e
  printf '%d\t%s\t%d\n' "${block}" "${directory}" "${rc}" \
    >>"${RUN_ROOT}/execution_status.tsv"
  (( rc == 0 )) || die "block ${block} failed; see ${directory}.stdout.log"
done

"${PYTHON}" "${VERIFIER}" "${RUN_ROOT}" --output "${RUN_ROOT}/verification.json"
echo "RUN_ROOT=${RUN_ROOT}"
