#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_PY:-${VLLM_ROOT}/.venv/bin/python}"
RUNNER="${SCRIPT_DIR}/benchmark_phase_switch_steps.py"
VERIFY="${SCRIPT_DIR}/verify_experiment.py"
TEST_VERIFY="${SCRIPT_DIR}/test_verify_experiment.py"
PREPARE="${REPO_ROOT}/experiments/vllm_slot_vs_pinned_copy/prepare_inputs.py"
PHASE_RUNNER="${REPO_ROOT}/experiments/vllm_p3_pinned_uva_phase_compare/benchmark_phase_steps.py"
GRAPH_RUNNER="${REPO_ROOT}/experiments/vllm_p3_static_alias_graph_integration/benchmark_full_graph_steps.py"
UVM_ALLOCATOR_SO="${VLLM_ROOT}/uvm_test/uvm_allocator.so"
CUSTOM_UVM="${CUSTOM_UVM:-${REPO_ROOT}/.build/nvidia-595.71.05/kernel-open/nvidia-uvm.ko}"
STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"

DEFAULT_MODEL="${MODEL:-}"
DEFAULT_FIXTURE="${REPO_ROOT}/experiments/adaptmoe_token_threshold_sharegpt_compare/inputs/sharegpt_deepseek_p1024_n128_20261024.json"

DUAL_BANK_BYTES=1207959552
LOGICAL_BYTES=2415919104
SLOT_BUDGET_BYTES="${SLOT_BUDGET_BYTES:-${DUAL_BANK_BYTES}}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-167772160}"
GPU_INDEX="${GPU_INDEX:-0}"
CPUSET="${CPUSET:-0-13}"
CPU_THREADS="${CPU_THREADS:-14}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"
GPU_CLEAN_TIMEOUT_SEC="${GPU_CLEAN_TIMEOUT_SEC:-240}"
MANUAL_MIGRATION_FAULT_MODE="${MANUAL_MIGRATION_FAULT_MODE:-2}"
REMOTE_READ_FULL_TLB_INVALIDATE="${REMOTE_READ_FULL_TLB_INVALIDATE:-0}"
PHASE_F_ROUTE_LOG_ENABLE="${PHASE_F_ROUTE_LOG_ENABLE:-0}"
ACCESS_COUNTER_MIGRATION_ENABLE="${ACCESS_COUNTER_MIGRATION_ENABLE:--1}"
UVM_PREFETCH_ENABLE="${UVM_PREFETCH_ENABLE:-1}"
UVM_THRASHING_ENABLE="${UVM_THRASHING_ENABLE:-1}"
STATIC_ALIAS_DUAL_VIEW_ENABLE="${STATIC_ALIAS_DUAL_VIEW_ENABLE:-0}"
STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL="${STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL:-0}"

RUN_ROOT_ACTIVE=""
STOCK_UVM=""
STOCK_SRCVERSION=""
STOCK_SHA256=""
CUSTOM_SRCVERSION=""
CUSTOM_SHA256=""
MODULE_RESTORED=0
MODULE_CYCLE_ARMED=0
SERVICES_RESTARTED=0
ACTIVE_SERVICES=()

die() { echo "error: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }

uvm_clients() {
  local clients=""
  clients="$(sudo fuser /dev/nvidia-uvm /dev/nvidia-uvm-tools 2>/dev/null || true)"
  tr ' ' '\n' <<<"${clients}" | awk '$1 ~ /^[0-9]+$/ {print $1}' | sort -nu
}

gpu_used_mib() {
  nvidia-smi -i "${GPU_INDEX}" --query-gpu=memory.used \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]'
}

compute_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1}' | sort -nu
}

gpu_clean() {
  local used pids clients
  used="$(gpu_used_mib)" || return 1
  pids="$(compute_pids)" || return 1
  clients="$(uvm_clients)" || return 1
  [[ "${used}" == 0 && -z "${pids}" && -z "${clients}" ]]
}

wait_gpu_clean() {
  local attempt
  for ((attempt = 0; attempt < GPU_CLEAN_TIMEOUT_SEC; ++attempt)); do
    gpu_clean && return 0
    sleep 1
  done
  return 1
}

loaded_srcversion() {
  [[ -r /sys/module/nvidia_uvm/srcversion ]] || return 0
  tr -d '[:space:]' </sys/module/nvidia_uvm/srcversion
}

gpu_state_json() {
  local output="$1" used pids clients loaded clean=false
  used="$(gpu_used_mib 2>/dev/null || true)"
  pids="$(compute_pids 2>/dev/null | paste -sd, - || true)"
  clients="$(uvm_clients 2>/dev/null | paste -sd, - || true)"
  loaded="$(loaded_srcversion)"
  [[ "${used}" == 0 && -z "${pids}" && -z "${clients}" ]] && clean=true
  "${VLLM_PY}" - "${output}" "${GPU_INDEX}" "${used:-unknown}" \
    "${pids}" "${clients}" "${loaded}" "${clean}" <<'PY'
import json, sys, time
path, gpu, used, pids, clients, loaded, clean = sys.argv[1:]
payload = {
    "schema": "vllm_uvm_phase_switch_gpu_state_v1",
    "timestamp": time.time(),
    "gpu_index": int(gpu),
    "memory_used_mib": int(used) if used.isdigit() else used,
    "compute_pids": [int(x) for x in pids.split(",") if x],
    "uvm_client_pids": [int(x) for x in clients.split(",") if x],
    "loaded_srcversion": loaded or None,
    "gpu_clean": clean == "true",
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

module_snapshot() {
  local output="$1" module_path="$2" used pids clients loaded clean=false
  local module_sha="" budget="" remap="" manual_fault_mode="" full_tlb_invalidate=""
  local access_counter_migration="" prefetch_enable="" thrashing_enable=""
  local static_alias_dual_view="" static_alias_dual_view_require_full=""
  [[ -f "${module_path}" ]] && module_sha="$(sha256sum "${module_path}" | awk '{print $1}')"
  used="$(gpu_used_mib 2>/dev/null || true)"
  pids="$(compute_pids 2>/dev/null | paste -sd, - || true)"
  clients="$(uvm_clients 2>/dev/null | paste -sd, - || true)"
  loaded="$(loaded_srcversion)"
  [[ "${used}" == 0 && -z "${pids}" && -z "${clients}" ]] && clean=true
  [[ -r /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes ]] \
    && budget="$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes)"
  [[ -r /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable ]] \
    && remap="$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable)"
  [[ -r /sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode ]] \
    && manual_fault_mode="$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode)"
  [[ -r /sys/module/nvidia_uvm/parameters/uvm_phase_f_remote_read_full_tlb_invalidate ]] \
    && full_tlb_invalidate="$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_phase_f_remote_read_full_tlb_invalidate)"
  [[ -r /sys/module/nvidia_uvm/parameters/uvm_perf_access_counter_migration_enable ]] \
    && access_counter_migration="$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_perf_access_counter_migration_enable)"
  [[ -r /sys/module/nvidia_uvm/parameters/uvm_perf_prefetch_enable ]] \
    && prefetch_enable="$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_perf_prefetch_enable)"
  [[ -r /sys/module/nvidia_uvm/parameters/uvm_perf_thrashing_enable ]] \
    && thrashing_enable="$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_perf_thrashing_enable)"
  [[ -r /sys/module/nvidia_uvm/parameters/uvm_moe_static_alias_dual_view_enable ]] \
    && static_alias_dual_view="$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_moe_static_alias_dual_view_enable)"
  [[ -r /sys/module/nvidia_uvm/parameters/uvm_moe_static_alias_dual_view_require_full ]] \
    && static_alias_dual_view_require_full="$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_moe_static_alias_dual_view_require_full)"
  "${VLLM_PY}" - "${output}" "${module_path}" "${module_sha}" "${loaded}" \
    "${used:-unknown}" "${pids}" "${clients}" "${clean}" "${budget}" \
    "${remap}" "${manual_fault_mode}" "${full_tlb_invalidate}" \
    "${access_counter_migration}" "${prefetch_enable}" \
    "${thrashing_enable}" "${static_alias_dual_view}" \
    "${static_alias_dual_view_require_full}" <<'PY'
import json, os, sys, time
(path, module_path, module_sha, loaded, used, pids, clients, clean,
 budget, remap, manual_fault_mode, full_tlb_invalidate,
 access_counter_migration, prefetch_enable, thrashing_enable,
 static_alias_dual_view, static_alias_dual_view_require_full) = sys.argv[1:]
payload = {
    "schema": "vllm_uvm_phase_switch_module_state_v1",
    "timestamp": time.time(),
    "module_path": os.path.realpath(module_path),
    "module_sha256": module_sha or None,
    "loaded_srcversion": loaded or None,
    "memory_used_mib": int(used) if used.isdigit() else used,
    "compute_pids": [int(x) for x in pids.split(",") if x],
    "uvm_client_pids": [int(x) for x in clients.split(",") if x],
    "gpu_clean": clean == "true",
    "slot_budget_bytes": int(budget) if budget.isdigit() else None,
    "slot_pte_remap_enable": int(remap) if remap.lstrip("-").isdigit() else None,
    "manual_migration_fault_mode": int(manual_fault_mode) if manual_fault_mode.lstrip("-").isdigit() else None,
    "phase_f_remote_read_full_tlb_invalidate": int(full_tlb_invalidate) if full_tlb_invalidate.lstrip("-").isdigit() else None,
    "access_counter_migration_enable": int(access_counter_migration) if access_counter_migration.lstrip("-").isdigit() else None,
    "prefetch_enable": int(prefetch_enable) if prefetch_enable.lstrip("-").isdigit() else None,
    "thrashing_enable": int(thrashing_enable) if thrashing_enable.lstrip("-").isdigit() else None,
    "static_alias_dual_view_enable": int(static_alias_dual_view) if static_alias_dual_view.lstrip("-").isdigit() else None,
    "static_alias_dual_view_require_full": int(static_alias_dual_view_require_full) if static_alias_dual_view_require_full.lstrip("-").isdigit() else None,
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

capture_and_stop_exporters() {
  local service
  for service in node-exporter.service gpu-exporter.service dcgm-exporter.service; do
    if [[ "$(systemctl is-active "${service}" 2>/dev/null || true)" == active ]]; then
      ACTIVE_SERVICES+=("${service}")
      sudo systemctl stop "${service}"
    fi
  done
}

restart_exporters() {
  local service
  (( SERVICES_RESTARTED == 0 )) || return 0
  for service in "${ACTIVE_SERVICES[@]}"; do
    sudo systemctl start "${service}" || true
  done
  SERVICES_RESTARTED=1
}

restore_stock() {
  local loaded rc=0 had_errexit=0
  [[ $- == *e* ]] && had_errexit=1
  if (( MODULE_CYCLE_ARMED == 0 )); then
    MODULE_RESTORED=1
    return 0
  fi
  (( MODULE_RESTORED == 0 )) || return 0
  set +e
  loaded="$(loaded_srcversion)"
  if [[ "${loaded}" != "${STOCK_SRCVERSION}" ]]; then
    if ! wait_gpu_clean; then
      echo "recovery: refusing UVM unload because GPU/client state is dirty" >&2
      rc=1
    else
      sudo modprobe -r nvidia_uvm || rc=1
      sudo modprobe nvidia || rc=1
      sudo modprobe nvidia_uvm || rc=1
    fi
  fi
  [[ "$(loaded_srcversion)" == "${STOCK_SRCVERSION}" ]] || rc=1
  wait_gpu_clean || rc=1
  (( rc == 0 )) && MODULE_RESTORED=1
  if (( had_errexit == 1 )); then
    set -e
  else
    set +e
  fi
  return "${rc}"
}

on_exit() {
  local original_rc="$1" recovery_rc=0
  trap - EXIT INT TERM
  restore_stock || recovery_rc=$?
  if [[ -n "${RUN_ROOT_ACTIVE}" && -d "${RUN_ROOT_ACTIVE}" \
      && "${MODULE_RESTORED}" == 1 \
      && ! -e "${RUN_ROOT_ACTIVE}/module.after.json" ]]; then
    module_snapshot "${RUN_ROOT_ACTIVE}/module.after.json" "${STOCK_UVM}" || recovery_rc=1
  fi
  restart_exporters
  (( original_rc == 0 )) || exit "${original_rc}"
  exit "${recovery_rc}"
}

trap 'on_exit $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

static_preflight() {
  require_file "${RUNNER}"
  require_file "${VERIFY}"
  require_file "${TEST_VERIFY}"
  require_file "${PREPARE}"
  require_file "${PHASE_RUNNER}"
  require_file "${GRAPH_RUNNER}"
  require_file "${UVM_ALLOCATOR_SO}"
  require_file "${CUSTOM_UVM}"
  [[ -x "${VLLM_PY}" ]] || die "missing vLLM Python: ${VLLM_PY}"
  [[ "${SLOT_BUDGET_BYTES}" =~ ^[0-9]+$ ]] || die "slot budget is not an integer"
  [[ "${KV_CACHE_BYTES}" =~ ^[0-9]+$ ]] || die "KV cache size is not an integer"
  (( SLOT_BUDGET_BYTES >= DUAL_BANK_BYTES )) || die "slot budget is below 1.125 GiB"
  command -v nvidia-smi >/dev/null || die "nvidia-smi unavailable"
  command -v modinfo >/dev/null || die "modinfo unavailable"
  command -v flock >/dev/null || die "flock unavailable"
  command -v timeout >/dev/null || die "timeout unavailable"
  command -v taskset >/dev/null || die "taskset unavailable"
  command -v journalctl >/dev/null || die "journalctl unavailable"
  command -v sudo >/dev/null || die "sudo unavailable"
  # Do not use grep -q here under pipefail: after an early match it can close
  # the pipe while modinfo is still writing the long parameter list, turning
  # a valid module into an intermittent SIGPIPE preflight failure.
  modinfo -p "${CUSTOM_UVM}" | grep '^uvm_manual_migration_fault_mode:' >/dev/null \
    || die "custom UVM lacks manual migration fail-fast mode"
  modinfo -p "${CUSTOM_UVM}" | grep '^uvm_phase_f_remote_read_full_tlb_invalidate:' >/dev/null \
    || die "custom UVM lacks Phase-F remote-read full TLB invalidate"
  modinfo -p "${CUSTOM_UVM}" | grep '^uvm_moe_static_alias_dual_view_enable:' >/dev/null \
    || die "custom UVM lacks static-alias dual-view mode"
  modinfo -p "${CUSTOM_UVM}" | grep '^uvm_moe_static_alias_dual_view_require_full:' >/dev/null \
    || die "custom UVM lacks full dual-view coverage guard"
  rg -q 'VLLM_UVM_ORIGINAL_VA_PHASE_MODE' \
    "${VLLM_ROOT}/vllm/model_executor/models/utils.py" \
    || die "original-VA phase mode is absent"
  rg -q 'graph_static_alias_actual_h2d_bytes' \
    "${VLLM_ROOT}/vllm/model_executor/models/utils.py" \
    || die "actual H2D audit field is absent"
  rg -q 'prepare_cpu_offload_uvm_original_va_remote_graph_capture' \
    "${VLLM_ROOT}/vllm/v1/worker/gpu_model_runner.py" \
    || die "remote Graph capture preparation is absent"
  rg -q 'activate_cpu_offload_uvm_original_va_prefill_topology' \
    "${VLLM_ROOT}/vllm/v1/worker/gpu_worker.py" \
    "${VLLM_ROOT}/vllm/v1/worker/gpu_model_runner.py" \
    || die "post-capture alias activation is absent"
  "${VLLM_PY}" -m py_compile "${RUNNER}" "${VERIFY}" "${TEST_VERIFY}"
  "${VLLM_PY}" -m pytest -q "${TEST_VERIFY}"
  bash -n "${BASH_SOURCE[0]}"
  echo "static preflight: OK"
}

runtime_preflight() {
  static_preflight
  [[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
  STOCK_UVM="$(modinfo -F filename nvidia-uvm)"
  STOCK_SRCVERSION="$(modinfo -F srcversion nvidia-uvm | tr -d '[:space:]')"
  STOCK_SHA256="$(sha256sum "${STOCK_UVM}" | awk '{print $1}')"
  CUSTOM_SRCVERSION="$(modinfo -F srcversion "${CUSTOM_UVM}" | tr -d '[:space:]')"
  CUSTOM_SHA256="$(sha256sum "${CUSTOM_UVM}" | awk '{print $1}')"
  [[ -n "${STOCK_SRCVERSION}" && -n "${CUSTOM_SRCVERSION}" ]] || die "module srcversion absent"
  [[ "$(loaded_srcversion)" == "${STOCK_SRCVERSION}" ]] || die "initial UVM is not stock"
  MODULE_CYCLE_ARMED=1
  capture_and_stop_exporters
  wait_gpu_clean || die "GPU must start at exactly 0 MiB with no UVM client"
  exec 9>"/tmp/vllm_uvm_prefill_alias_decode_remote_graph.lock"
  flock -n 9 || die "another phase-switch experiment holds the lock"
}

load_custom_module() {
  wait_gpu_clean || die "GPU state changed before custom module load"
  sudo modprobe -r nvidia_uvm
  sudo modprobe nvidia
  sudo insmod "${CUSTOM_UVM}" \
    uvm_moe_slot_pte_remap_enable=1 \
    uvm_moe_slot_pool_vidmem_budget_bytes="${SLOT_BUDGET_BYTES}" \
    uvm_manual_migration_fault_mode="${MANUAL_MIGRATION_FAULT_MODE}" \
    uvm_phase_f_remote_read_full_tlb_invalidate="${REMOTE_READ_FULL_TLB_INVALIDATE}" \
    uvm_phase_f_route_log_enable="${PHASE_F_ROUTE_LOG_ENABLE}" \
    uvm_perf_access_counter_migration_enable="${ACCESS_COUNTER_MIGRATION_ENABLE}" \
    uvm_perf_prefetch_enable="${UVM_PREFETCH_ENABLE}" \
    uvm_perf_thrashing_enable="${UVM_THRASHING_ENABLE}" \
    uvm_moe_static_alias_dual_view_enable="${STATIC_ALIAS_DUAL_VIEW_ENABLE}" \
    uvm_moe_static_alias_dual_view_require_full="${STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL}"
  [[ "$(loaded_srcversion)" == "${CUSTOM_SRCVERSION}" ]] || die "custom UVM srcversion differs"
  wait_gpu_clean || die "custom module load left GPU state"
  [[ "$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable)" == 1 ]] \
    || die "slot PTE remap is not enabled"
  [[ "$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode)" == "${MANUAL_MIGRATION_FAULT_MODE}" ]] \
    || die "manual migration fault mode differs from requested mode ${MANUAL_MIGRATION_FAULT_MODE}"
  [[ "$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_phase_f_remote_read_full_tlb_invalidate)" == "${REMOTE_READ_FULL_TLB_INVALIDATE}" ]] \
    || die "remote-read full TLB invalidate differs from requested mode ${REMOTE_READ_FULL_TLB_INVALIDATE}"
  [[ "$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_perf_access_counter_migration_enable)" == "${ACCESS_COUNTER_MIGRATION_ENABLE}" ]] \
    || die "access-counter migration differs from requested mode ${ACCESS_COUNTER_MIGRATION_ENABLE}"
  [[ "$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_perf_prefetch_enable)" == "${UVM_PREFETCH_ENABLE}" ]] \
    || die "UVM prefetch differs from requested mode ${UVM_PREFETCH_ENABLE}"
  [[ "$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_perf_thrashing_enable)" == "${UVM_THRASHING_ENABLE}" ]] \
    || die "UVM thrashing differs from requested mode ${UVM_THRASHING_ENABLE}"
  [[ "$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_moe_static_alias_dual_view_enable)" == "${STATIC_ALIAS_DUAL_VIEW_ENABLE}" ]] \
    || die "static-alias dual-view mode differs from requested mode ${STATIC_ALIAS_DUAL_VIEW_ENABLE}"
  [[ "$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_moe_static_alias_dual_view_require_full)" == "${STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL}" ]] \
    || die "full dual-view guard differs from requested mode ${STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL}"
  [[ -r "${STATS_PATH}" ]] || die "latency stats unavailable after custom module load"
}

arm_name() {
  case "$1" in
    H) echo alias_prefill_remote_eager_decode ;;
    G) echo alias_prefill_remote_graph_decode ;;
    R) echo remote_prefill_remote_eager_decode ;;
    C) echo remote_prefill_remote_graph_decode ;;
    D) echo alias_graph_prefill_remote_graph_decode ;;
    *) die "unknown arm: $1" ;;
  esac
}

phase_mode() {
  case "$1" in
    H|G) echo prefill_remote_decode ;;
    R|C) echo remote_only ;;
    D) echo prefill_graph_remote_decode ;;
    *) die "unknown arm: $1" ;;
  esac
}

generate_case_plan() {
  local output="$1" mode="$2" repeats="$3"
  "${VLLM_PY}" - "${output}" "${mode}" "${repeats}" <<'PY'
import csv, sys
path, mode, repeats_text = sys.argv[1:]
repeats = int(repeats_text)
batches = [1] if mode == "smoke" else [1, 4]
rows = []
audit_order = {1: "HGCR", 4: "RCGH"}
for batch in batches:
    for symbol in audit_order[batch]:
        rows.append(("audit", 0, batch, symbol))
if mode == "formal":
    sequences = ("HGCR", "GRHC", "RCGH", "CHRG")
    for sample in range(1, repeats + 1):
        batch_order = batches if sample % 2 else list(reversed(batches))
        for batch in batch_order:
            offset = 0 if batch == 1 else 2
            for symbol in sequences[(sample - 1 + offset) % len(sequences)]:
                rows.append(("performance", sample, batch, symbol))
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
    "${MEASURED_DECODE_STEPS}" <<'PY'
import json, os, sys
(path, mode, repeats, model, fixture, kv, budget, stock_path, stock_src,
 stock_sha, custom_path, custom_src, custom_sha, warmup, measured) = sys.argv[1:]
payload = {
    "schema": "vllm_uvm_prefill_alias_decode_remote_graph_run_v1",
    "mode": mode,
    "performance_repeats": int(repeats),
    "model": os.path.realpath(model),
    "fixture": os.path.realpath(fixture),
    "batch_sizes": [1] if mode == "smoke" else [1, 4],
    "arms": ["H", "G", "R", "C"],
    "cudagraph_capture_sizes": [1, 4],
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "warmup_decode_steps": int(warmup),
    "measured_decode_steps": int(measured),
    "logical_bytes_per_forward": 2_415_919_104,
    "dual_bank_backing_bytes": 1_207_959_552,
    "alias_count": 6_912,
    "alias_bytes": 3_623_878_656,
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
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

run_arm() {
  local symbol="$1" batch="$2" kind="$3" sample="$4" arm_dir="$5" names="$6"
  local arm phase execution trace graph_args=()
  arm="$(arm_name "${symbol}")"
  phase="$(phase_mode "${symbol}")"
  trace=0
  [[ "${kind}" == audit ]] && trace=1
  execution=eager
  if [[ "${symbol}" == G || "${symbol}" == C || "${symbol}" == D ]]; then
    execution=full_graph_decode
  fi
  [[ "${kind}" == audit ]] && graph_args+=(--audit-trace)
  mkdir "${arm_dir}"
  : >"${arm_dir}/cpu_offload_manifest.jsonl"
  wait_gpu_clean || die "GPU is not clean before ${arm}"
  gpu_state_json "${arm_dir}/gpu.before.json"
  cp -- "${STATS_PATH}" "${arm_dir}/latency_stats.before.txt"
  local kernel_epoch
  kernel_epoch="$(date +%s)"

  local -a common_env=(
    VLLM_USE_UVM=0
    VLLM_ENABLE_V1_MULTIPROCESSING=0
    VLLM_CPU_OFFLOAD_BACKEND=copy
    VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${names}"
    VLLM_CPU_OFFLOAD_STATIC_BANK=0
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES="${REGION_BYTES:-524288}"
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE="${PTE_MODE:-auto}"
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
    VLLM_UVM_ORIGINAL_VA_PHASE_MODE="${phase}"
    VLLM_UVM_ORIGINAL_VA_PHASE_RESULT_FILE="${arm_dir}/phase_transition_result.json"
    VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ=0
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
    VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
    VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0
    VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0
    VLLM_CPU_OFFLOAD_COPY_TRACE="${trace}"
    VLLM_CPU_OFFLOAD_COPY_SYNC=0
    VLLM_CPU_OFFLOAD_MANIFEST_FILE="${arm_dir}/cpu_offload_manifest.jsonl"
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1
    VLLM_UVM_LOG_FILE="${arm_dir}/uvm_allocations.log"
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    TOKENIZERS_PARALLELISM=false
    OMP_NUM_THREADS="${CPU_THREADS}"
    MKL_NUM_THREADS="${CPU_THREADS}"
    CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
  )
  if [[ "${symbol}" == D ]]; then
    common_env+=(VLLM_ATTENTION_BACKEND=TRITON_ATTN)
  fi
  if [[ "${VLLM_PARK_RESUME_EXPERIMENT:-0}" == 1 ]]; then
    [[ "${symbol}" == D ]] || die "park/resume experiment requires arm D"
    common_env+=(VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1)
    common_env+=(
      VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES="${PARK_RESUME_MIN_FREE_BYTES:-201326592}"
    )
    graph_args+=(--park-resume-roundtrip)
  fi
  printf '%s\n' "${common_env[@]}" >"${arm_dir}/environment.txt"

  local model="${MODEL:-${DEFAULT_MODEL}}" fixture="${EVAL_FIXTURE:-${DEFAULT_FIXTURE}}"
  local capture_sizes=1,4
  [[ "${symbol}" == D ]] && capture_sizes=1,4,314,1217
  local -a args=(
    --model "${model}"
    --fixture "${fixture}"
    --exact-names "${names}"
    --offload-manifest "${arm_dir}/cpu_offload_manifest.jsonl"
    --output "${arm_dir}/result.json"
    --arm "${symbol}"
    --batch-size "${batch}"
    --cudagraph-capture-sizes "${capture_sizes}"
    --warmup-decode-steps "${WARMUP_DECODE_STEPS}"
    --measured-decode-steps "${MEASURED_DECODE_STEPS}"
    --kv-cache-memory-bytes "${KV_CACHE_BYTES}"
    --cpu-offload-gb 3.0
    --max-model-len "${MAX_MODEL_LEN:-512}"
    --max-num-seqs 4
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.97}"
    --seed 0
    "${graph_args[@]}"
  )
  {
    printf 'cd %q && env ' "${VLLM_ROOT}"
    printf '%q ' "${common_env[@]}"
    printf 'timeout --kill-after=30s %qs taskset -c %q %q %q ' \
      "${TIMEOUT_SEC}" "${CPUSET}" "${VLLM_PY}" "${RUNNER}"
    printf '%q ' "${args[@]}"
    printf '\n'
  } >"${arm_dir}/command.txt"

  local rc journal_rc
  set +e
  (
    cd "${VLLM_ROOT}"
    env "${common_env[@]}" \
      /usr/bin/time -v -o "${arm_dir}/resource_time.txt" \
      timeout --kill-after=30s "${TIMEOUT_SEC}s" \
      taskset -c "${CPUSET}" \
      "${VLLM_PY}" "${RUNNER}" "${args[@]}"
  ) >"${arm_dir}/run.log" 2>&1
  rc=$?
  if [[ -r /proc/driver/nvidia-uvm/phase_f_route_log ]]; then
    cp -- /proc/driver/nvidia-uvm/phase_f_route_log \
      "${arm_dir}/phase_f_route_log.txt"
  fi
  if [[ -r /proc/driver/nvidia-uvm/phase_f_route_log_stats ]]; then
    cp -- /proc/driver/nvidia-uvm/phase_f_route_log_stats \
      "${arm_dir}/phase_f_route_log_stats.txt"
  fi
  sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
    >"${arm_dir}/kernel.since.log" 2>"${arm_dir}/kernel.since.err"
  journal_rc=$?
  set -e
  echo "${rc}" >"${arm_dir}/returncode.txt"
  echo "${journal_rc}" >"${arm_dir}/kernel_log_returncode.txt"
  cp -- "${STATS_PATH}" "${arm_dir}/latency_stats.after.txt"
  (( journal_rc == 0 )) || return 97
  if ! wait_gpu_clean; then
    gpu_state_json "${arm_dir}/gpu.after.json"
    die "GPU/client state did not cleanly shut down after ${arm}"
  fi
  gpu_state_json "${arm_dir}/gpu.after.json"
  return "${rc}"
}

run_suite() {
  local mode="$1" repeats
  runtime_preflight
  local model="${MODEL:-${DEFAULT_MODEL}}" fixture="${EVAL_FIXTURE:-${DEFAULT_FIXTURE}}"
  require_file "${model}/config.json"
  require_file "${fixture}"
  case "${mode}" in
    smoke)
      repeats=0
      WARMUP_DECODE_STEPS="${WARMUP_DECODE_STEPS:-1}"
      MEASURED_DECODE_STEPS="${MEASURED_DECODE_STEPS:-2}"
      ;;
    qualification)
      repeats=0
      WARMUP_DECODE_STEPS="${WARMUP_DECODE_STEPS:-2}"
      MEASURED_DECODE_STEPS="${MEASURED_DECODE_STEPS:-8}"
      ;;
    formal)
      repeats="${FORMAL_PERFORMANCE_REPEATS:-3}"
      [[ "${repeats}" =~ ^[1-9][0-9]*$ ]] || die "FORMAL_PERFORMANCE_REPEATS must be positive"
      WARMUP_DECODE_STEPS="${WARMUP_DECODE_STEPS:-5}"
      MEASURED_DECODE_STEPS="${MEASURED_DECODE_STEPS:-16}"
      ;;
    *) die "unknown mode: ${mode}" ;;
  esac

  local run_root="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_${mode}}"
  [[ ! -e "${run_root}" ]] || die "RUN_ROOT already exists: ${run_root}"
  mkdir -p "${run_root}/inputs"
  run_root="$(cd "${run_root}" && pwd)"
  RUN_ROOT_ACTIVE="${run_root}"
  module_snapshot "${run_root}/module.before.json" "${STOCK_UVM}"
  load_custom_module
  module_snapshot "${run_root}/module.custom.json" "${CUSTOM_UVM}"

  "${VLLM_PY}" "${PREPARE}" --output-dir "${run_root}/inputs" \
    >"${run_root}/prepare_inputs.stdout.json"
  local names="${run_root}/inputs/exact_parameter_names.txt"
  require_file "${names}"
  [[ "$(wc -l <"${names}")" -eq 8 ]] || die "expected eight exact parameter names"
  generate_case_plan "${run_root}/case_plan.tsv" "${mode}" "${repeats}"
  write_run_config "${run_root}/run_config.json" "${mode}" "${repeats}" "${model}" "${fixture}"
  sha256sum \
    "${model}/config.json" "${fixture}" "${names}" "${BASH_SOURCE[0]}" \
    "${RUNNER}" "${VERIFY}" "${TEST_VERIFY}" "${PREPARE}" \
    "${PHASE_RUNNER}" "${GRAPH_RUNNER}" \
    "${VLLM_ROOT}/vllm/model_executor/models/utils.py" \
    "${VLLM_ROOT}/vllm/v1/worker/gpu_model_runner.py" \
    "${VLLM_ROOT}/vllm/v1/worker/gpu_worker.py" \
    "${VLLM_ROOT}/vllm/device_allocator/uvm.py" \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_static_alias.py" \
    "${UVM_ALLOCATOR_SO}" "${CUSTOM_UVM}" "${STOCK_UVM}" \
    >"${run_root}/input_sha256.txt"
  if git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "${REPO_ROOT}" status --short >"${run_root}/git_status.txt"
  else
    printf 'Source snapshot without Git history\n' >"${run_root}/git_status.txt"
  fi
  printf 'ordinal\tkind\tsample\tbatch\tsymbol\tarm\tdirectory\treturncode\n' \
    >"${run_root}/execution_status.tsv"

  local failures=0 ordinal kind sample batch symbol arm arm_dir rc
  while IFS=$'\t' read -r ordinal kind sample batch symbol; do
    [[ "${ordinal}" == ordinal ]] && continue
    arm="$(arm_name "${symbol}")"
    arm_dir="${run_root}/$(printf '%02d_%s_s%s_b%s_%s' "${ordinal}" "${kind}" "${sample}" "${batch}" "${arm}")"
    if run_arm "${symbol}" "${batch}" "${kind}" "${sample}" "${arm_dir}" "${names}"; then
      rc=0
    else
      rc=$?
      failures=$((failures + 1))
      tail -n 200 "${arm_dir}/run.log" >&2 || true
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${ordinal}" "${kind}" "${sample}" "${batch}" "${symbol}" "${arm}" \
      "$(basename "${arm_dir}")" "${rc}" >>"${run_root}/execution_status.tsv"
  done <"${run_root}/case_plan.tsv"

  restore_stock || die "failed to restore stock nvidia_uvm"
  module_snapshot "${run_root}/module.after.json" "${STOCK_UVM}"
  restart_exporters

  local verify_rc=0
  if "${VLLM_PY}" "${VERIFY}" "${run_root}" \
      --output "${run_root}/verification.json" \
      >"${run_root}/verify.log" 2>"${run_root}/verify.err"; then
    verify_rc=0
  else
    verify_rc=$?
    cat "${run_root}/verify.err" >&2 || true
  fi
  echo "RUN_ROOT=${run_root}"
  echo "verification=${run_root}/verification.json"
  (( failures == 0 && verify_rc == 0 )) \
    || die "suite failed: arm_failures=${failures}, verifier=${verify_rc}"
}

if [[ "${VLLM_PHASE_SWITCH_SOURCE_ONLY:-0}" != 1 ]]; then
  case "${1:-help}" in
    static-preflight) static_preflight ;;
    smoke) run_suite smoke ;;
    qualify|qualification) run_suite qualification ;;
    formal|run) run_suite formal ;;
    verify)
      [[ $# -eq 2 ]] || die "verify requires RUN_ROOT"
      "${VLLM_PY}" "${VERIFY}" "$2"
      ;;
    *)
      echo "usage: $0 {static-preflight|smoke|qualify|formal|verify RUN_ROOT}" >&2
      exit 2
      ;;
  esac
fi
