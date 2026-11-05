#!/usr/bin/env bash
set -euo pipefail

E12_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E12_ROOT="$(cd "${E12_DIR}/../.." && pwd)"
export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
export ACCESS_COUNTER_MIGRATION_ENABLE=0
export UVM_PREFETCH_ENABLE=0
export UVM_THRASHING_ENABLE=0
export STATIC_ALIAS_DUAL_VIEW_ENABLE=1
export STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL=1
# shellcheck source=/dev/null
source "${E12_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY

PROBE="${E12_ROOT}/experiments/uvm_static_alias_original_va/full_dual_view_geometry_probe.py"
VERIFY_E12="${E12_DIR}/verify_e12_dual_view_soak.py"
CYCLES="${CYCLES:-1000}"
MINIMUM_SECONDS="${MINIMUM_SECONDS:-0}"
VERIFY_EVERY="${VERIFY_EVERY:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"

require_file "${PROBE}"
require_file "${VERIFY_E12}"
[[ "${CYCLES}" =~ ^[1-9][0-9]*$ ]] || die "CYCLES must be positive"
[[ "${VERIFY_EVERY}" =~ ^[1-9][0-9]*$ ]] || die "VERIFY_EVERY must be positive"
[[ "${CHECKPOINT_EVERY}" =~ ^[0-9]+$ ]] || die "CHECKPOINT_EVERY must be non-negative"
"${VLLM_PY}" -m py_compile "${PROBE}" "${VERIFY_E12}"
runtime_preflight

run_root="${RUN_ROOT:-${E12_DIR}/artifacts/e12-dual-view-soak-$(date -u +%Y%m%dT%H%M%SZ)-attempt-01-qualification}"
[[ ! -e "${run_root}" ]] || die "RUN_ROOT already exists: ${run_root}"
mkdir -p "${run_root}"
run_root="$(cd "${run_root}" && pwd)"
RUN_ROOT_ACTIVE="${run_root}"
module_snapshot "${run_root}/module.before.json" "${STOCK_UVM}"
load_custom_module
module_snapshot "${run_root}/module.custom.json" "${CUSTOM_UVM}"
cp -- "${STATS_PATH}" "${run_root}/latency_stats.before.txt"
git -C "${E12_ROOT}" status --short >"${run_root}/git_status.txt"
sha256sum "${PROBE}" "${VERIFY_E12}" "${CUSTOM_UVM}" \
  >"${run_root}/input_sha256.txt"
"${VLLM_PY}" - "${run_root}/configuration.json" "${CYCLES}" \
  "${MINIMUM_SECONDS}" "${VERIFY_EVERY}" "${CHECKPOINT_EVERY}" <<'PY'
import json, sys
path, cycles, seconds, verify_every, checkpoint_every = sys.argv[1:]
value = {
    "cycles": int(cycles),
    "minimum_seconds": float(seconds),
    "verify_every": int(verify_every),
    "checkpoint_every": int(checkpoint_every),
    "scope": "PARK/RESUME mechanism soak; not the six-hour model-serving soak",
}
open(path, "w", encoding="utf-8").write(
    json.dumps(value, indent=2, sort_keys=True) + "\n"
)
PY

kernel_epoch="$(date +%s)"
set +e
(
  cd "${VLLM_ROOT}"
  env PYTHONPATH="${VLLM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
      CUDA_VISIBLE_DEVICES=0 \
      /usr/bin/time -v -o "${run_root}/resource_time.txt" \
      "${VLLM_PY}" "${PROBE}" \
        --output "${run_root}/result.json" \
        --cycles "${CYCLES}" \
        --minimum-seconds "${MINIMUM_SECONDS}" \
        --verify-every "${VERIFY_EVERY}" \
        --checkpoint-every "${CHECKPOINT_EVERY}" \
        --progress "${run_root}/progress.jsonl"
) >"${run_root}/run.log" 2>&1
runner_rc=$?
sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
  >"${run_root}/kernel.since.log" 2>"${run_root}/kernel.since.err"
journal_rc=$?
set -e
printf '%s\n' "${runner_rc}" >"${run_root}/returncode.txt"
printf '%s\n' "${journal_rc}" >"${run_root}/kernel_log_returncode.txt"
cp -- "${STATS_PATH}" "${run_root}/latency_stats.after.txt"
wait_gpu_clean || die "GPU/client state did not clean after E12 soak"
restore_stock || die "failed to restore stock UVM"
module_snapshot "${run_root}/module.after.json" "${STOCK_UVM}"
restart_exporters

set +e
"${VLLM_PY}" "${VERIFY_E12}" "${run_root}" \
  --minimum-cycles "${CYCLES}" \
  --output "${run_root}/verification.json" \
  >"${run_root}/verify.log" 2>"${run_root}/verify.err"
verify_rc=$?
set -e
echo "RUN_ROOT=${run_root}"
echo "verification=${run_root}/verification.json"
(( runner_rc == 0 && journal_rc == 0 && verify_rc == 0 )) || {
  tail -n 200 "${run_root}/run.log" >&2 || true
  cat "${run_root}/verify.err" >&2 || true
  die "E12 dual-view soak qualification failed"
}
