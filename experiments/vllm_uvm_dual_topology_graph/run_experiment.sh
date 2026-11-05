#!/usr/bin/env bash
set -euo pipefail

DUAL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DUAL_REPO_ROOT="$(cd "${DUAL_SCRIPT_DIR}/../.." && pwd)"
BASE_HARNESS="${DUAL_REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
# shellcheck source=/dev/null
source "${BASE_HARNESS}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY
export VLLM_PARK_RESUME_EXPERIMENT=1
TRANSITION_MODE="${TRANSITION_MODE:-grouped}"
[[ "${TRANSITION_MODE}" == grouped || "${TRANSITION_MODE}" == sequential ]] \
  || die "TRANSITION_MODE must be grouped or sequential"
export VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_TRANSITION_MODE="${TRANSITION_MODE}"

SCRIPT_DIR="${DUAL_SCRIPT_DIR}"
REPO_ROOT="${DUAL_REPO_ROOT}"
RUNNER="${SCRIPT_DIR}/benchmark_dual_topology_steps.py"
VERIFY="${SCRIPT_DIR}/verify_experiment.py"
TEST_VERIFY="${SCRIPT_DIR}/test_dual_topology_verify.py"

arm_name() {
  [[ "$1" == D ]] || die "dual-topology harness only accepts arm D"
  echo alias_graph_prefill_remote_graph_decode
}

phase_mode() {
  [[ "$1" == D ]] || die "dual-topology harness only accepts arm D"
  echo prefill_graph_remote_decode
}

generate_case_plan() {
  local output="$1" mode="$2" repeats="$3"
  "${VLLM_PY}" - "${output}" "${mode}" "${repeats}" <<'PY'
import csv, sys
path, mode, repeats_text = sys.argv[1:]
repeats = int(repeats_text)
batches = [1] if mode == "smoke" else [1, 4]
rows = [("audit", 0, batch, "D") for batch in batches]
if mode == "formal":
    for sample in range(1, repeats + 1):
        batch_order = batches if sample % 2 else list(reversed(batches))
        rows.extend(("performance", sample, batch, "D") for batch in batch_order)
with open(path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    writer.writerow(("ordinal", "kind", "sample", "batch", "symbol"))
    for ordinal, row in enumerate(rows, 1):
        writer.writerow((ordinal, *row))
PY
}

write_run_config() {
  local output="$1" mode="$2" repeats="$3" model="$4" fixture="$5"
  "${VLLM_PY}" - "${output}" "${mode}" "${repeats}" "${model}" "${fixture}" \
    "${KV_CACHE_BYTES}" "${SLOT_BUDGET_BYTES}" "${STOCK_UVM}" \
    "${STOCK_SRCVERSION}" "${STOCK_SHA256}" "${CUSTOM_UVM}" \
    "${CUSTOM_SRCVERSION}" "${CUSTOM_SHA256}" "${WARMUP_DECODE_STEPS}" \
    "${MEASURED_DECODE_STEPS}" "${TRANSITION_MODE}" \
    "${STATIC_ALIAS_DUAL_VIEW_ENABLE}" \
    "${STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL}" <<'PY'
import json, os, sys
(path, mode, repeats, model, fixture, kv, budget, stock_path, stock_src,
 stock_sha, custom_path, custom_src, custom_sha, warmup, measured,
 transition_mode, dual_view_enable, dual_view_require_full) = sys.argv[1:]
payload = {
    "schema": "vllm_uvm_dual_topology_graph_park_resume_run_v3",
    "mode": mode,
    "performance_repeats": int(repeats),
    "model": os.path.realpath(model),
    "fixture": os.path.realpath(fixture),
    "batch_sizes": [1] if mode == "smoke" else [1, 4],
    "arms": ["D"],
    "phase_mode": "prefill_graph_remote_decode",
    "park_resume": True,
    "park_batch_required": True,
    "transition_mode": transition_mode,
    "static_alias_dual_view_enable": int(dual_view_enable),
    "static_alias_dual_view_require_full": int(dual_view_require_full),
    "roundtrip_cycles": 2,
    "attention_backend": "TRITON_ATTN",
    "cudagraph_mode": "FULL_SEPARATE",
    "cudagraph_capture_sizes": [1, 4, 314, 1217],
    "remote_uniform_entries": [1, 4],
    "alias_nonuniform_entries": [314, 1217],
    "warmup_decode_steps": int(warmup),
    "measured_decode_steps": int(measured),
    "logical_bytes_per_forward": 2_415_919_104,
    "dual_bank_backing_bytes": 1_207_959_552,
    "kv_cache_memory_bytes": int(kv),
    "slot_budget_bytes": int(budget),
    "manual_migration_fault_mode": 2,
    "module_cycle": True,
    "stock_module_path": os.path.realpath(stock_path),
    "stock_module_srcversion": stock_src,
    "stock_module_sha256": stock_sha,
    "custom_module_path": os.path.realpath(custom_path),
    "custom_module_srcversion": custom_src,
    "custom_module_sha256": custom_sha,
}
open(path, "w", encoding="utf-8").write(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY
}

dual_static_preflight() {
  static_preflight
  rg -q 'FULL_SEPARATE' "${VLLM_ROOT}/vllm/config/compilation.py" \
    || die "FULL_SEPARATE mode is absent"
  rg -q 'certify_cpu_offload_uvm_original_va_dual_graph_partition' \
    "${VLLM_ROOT}/vllm/model_executor/models/utils.py" \
    "${VLLM_ROOT}/vllm/v1/worker/gpu_model_runner.py" \
    || die "dual Graph partition certificate is absent"
  rg -q 'driver_static_alias_park' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_static_alias.py" \
    || die "static-alias PARK userspace ABI is absent"
  rg -q 'driver_static_alias_resume' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_static_alias.py" \
    || die "static-alias RESUME userspace ABI is absent"
  rg -q 'TRANSITION_FLAG_FORCE_SEQUENTIAL' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_static_alias.py" \
    "${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h" \
    || die "controlled sequential transition ABI is absent"
  rg -q 'VLLM_ATTENTION_BACKEND=TRITON_ATTN' "${BASE_HARNESS}" \
    || die "D arm does not pin the TRITON_ATTN backend"
  bash -n "${BASH_SOURCE[0]}"
  echo "dual-topology static preflight: OK"
}

case "${1:-help}" in
  static-preflight) dual_static_preflight ;;
  smoke) run_suite smoke ;;
  qualify|qualification) run_suite qualification ;;
  formal|run) run_suite formal ;;
  verify)
    [[ $# -eq 2 ]] || die "verify requires RUN_ROOT"
    "${VLLM_PY}" "${VERIFY}" "$2" --output "$2/verification.json"
    ;;
  *)
    echo "usage: $0 {static-preflight|smoke|qualify|formal|verify RUN_ROOT}" >&2
    exit 2
    ;;
esac
