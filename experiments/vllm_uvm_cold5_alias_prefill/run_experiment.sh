#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
export PYTHONPATH="${VLLM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
VLLM_PY="${VLLM_PY:-${VLLM_ROOT}/.venv/bin/python}"
RUNNER="${SCRIPT_DIR}/benchmark_cold5_steps.py"
VERIFY="${SCRIPT_DIR}/verify_experiment.py"
FREEZER="${SCRIPT_DIR}/freeze_cold5_plan.py"
PLAN="${SCRIPT_DIR}/cold5_expert_plan.json"
NAMES="${SCRIPT_DIR}/exact_parameter_names.txt"
BASE_RUNNER="${REPO_ROOT}/experiments/vllm_p3_static_alias_graph_integration/benchmark_full_graph_steps.py"
CUSTOM_UVM="${CUSTOM_UVM:-${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
DEFAULT_MODEL="${REPO_ROOT}/../.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782"
DEFAULT_FIXTURE="${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json"

BACKING_BYTES=47185920
SELECTED_BYTES=1132462080
SLOT_BUDGET_BYTES="${SLOT_BUDGET_BYTES:-67108864}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-167772160}"
GPU_INDEX="${GPU_INDEX:-0}"
CPUSET="${CPUSET:-0-13}"
CPU_THREADS="${CPU_THREADS:-14}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"
GPU_CLEAN_TIMEOUT_SEC="${GPU_CLEAN_TIMEOUT_SEC:-240}"

RUN_ROOT_ACTIVE=""
STOCK_UVM=""
STOCK_SRCVERSION=""
CUSTOM_SRCVERSION=""
MODULE_CYCLE_ARMED=0
MODULE_RESTORED=0
ACTIVE_SERVICES=()

die() { echo "error: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }

uvm_clients() {
  sudo fuser /dev/nvidia-uvm /dev/nvidia-uvm-tools 2>/dev/null \
    | tr ' ' '\n' | awk '$1 ~ /^[0-9]+$/ {print $1}' | sort -nu || true
}

gpu_used_mib() {
  nvidia-smi -i "${GPU_INDEX}" --query-gpu=memory.used \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]'
}

compute_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk '$1 ~ /^[0-9]+$/ {print $1}' | sort -nu
}

gpu_clean() {
  [[ "$(gpu_used_mib)" == "0" && -z "$(compute_pids)" && -z "$(uvm_clients)" ]]
}

wait_gpu_clean() {
  local attempt service
  for ((attempt = 0; attempt < GPU_CLEAN_TIMEOUT_SEC; ++attempt)); do
    if (( MODULE_CYCLE_ARMED == 1 && MODULE_RESTORED == 0 )); then
      for service in node-exporter.service gpu-exporter.service dcgm-exporter.service; do
        if systemctl is-active --quiet "${service}" 2>/dev/null; then
          sudo systemctl stop "${service}"
        fi
      done
    fi
    gpu_clean && return 0
    sleep 1
  done
  return 1
}

loaded_srcversion() {
  [[ -r /sys/module/nvidia_uvm/srcversion ]] || return 0
  tr -d '[:space:]' </sys/module/nvidia_uvm/srcversion
}

snapshot_gpu() {
  local output="$1" used pids clients loaded
  used="$(gpu_used_mib 2>/dev/null || true)"
  pids="$(compute_pids | paste -sd, - || true)"
  clients="$(uvm_clients | paste -sd, - || true)"
  loaded="$(loaded_srcversion)"
  "${VLLM_PY}" - "${output}" "${used}" "${pids}" "${clients}" "${loaded}" <<'PY'
import json, sys, time
path, used, pids, clients, loaded = sys.argv[1:]
json.dump({
    "timestamp": time.time(),
    "memory_used_mib": int(used) if used.isdigit() else used,
    "compute_pids": [int(x) for x in pids.split(",") if x],
    "uvm_client_pids": [int(x) for x in clients.split(",") if x],
    "loaded_srcversion": loaded or None,
}, open(path, "w", encoding="utf-8"), indent=2, sort_keys=True)
open(path, "a", encoding="utf-8").write("\n")
PY
}

stop_exporters() {
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
  for service in "${ACTIVE_SERVICES[@]}"; do
    sudo systemctl start "${service}" || true
  done
  ACTIVE_SERVICES=()
}

snapshot_services() {
  local output="$1" service status
  : >"${output}"
  for service in node-exporter.service gpu-exporter.service dcgm-exporter.service; do
    status="$(systemctl is-active "${service}" 2>/dev/null || true)"
    printf '%s\t%s\n' "${service}" "${status}" >>"${output}"
  done
}

restore_stock() {
  local rc=0
  (( MODULE_CYCLE_ARMED == 1 )) || return 0
  (( MODULE_RESTORED == 0 )) || return 0
  set +e
  if [[ "$(loaded_srcversion)" != "${STOCK_SRCVERSION}" ]]; then
    wait_gpu_clean || rc=1
    if (( rc == 0 )); then
      sudo modprobe -r nvidia_uvm || rc=1
      sudo modprobe nvidia || rc=1
      sudo modprobe nvidia_uvm || rc=1
    fi
  fi
  [[ "$(loaded_srcversion)" == "${STOCK_SRCVERSION}" ]] || rc=1
  wait_gpu_clean || rc=1
  (( rc == 0 )) && MODULE_RESTORED=1
  set -e
  return "${rc}"
}

on_exit() {
  local original_rc="$1" recovery_rc=0
  trap - EXIT INT TERM
  restore_stock || recovery_rc=$?
  if [[ -n "${RUN_ROOT_ACTIVE}" && -d "${RUN_ROOT_ACTIVE}" ]]; then
    snapshot_gpu "${RUN_ROOT_ACTIVE}/gpu.final.json" || recovery_rc=1
  fi
  restart_exporters
  (( original_rc == 0 )) || exit "${original_rc}"
  exit "${recovery_rc}"
}

trap 'on_exit $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

static_preflight() {
  local path
  for path in "${RUNNER}" "${VERIFY}" "${FREEZER}" "${PLAN}" "${NAMES}" \
      "${BASE_RUNNER}" "${CUSTOM_UVM}"; do
    require_file "${path}"
  done
  [[ -x "${VLLM_PY}" ]] || die "missing vLLM Python"
  [[ "$(awk 'NF {count++} END {print count+0}' "${NAMES}")" == 96 ]] \
    || die "exact-name count is not 96"
  (( SLOT_BUDGET_BYTES >= BACKING_BYTES )) || die "slot budget is below 45 MiB"
  "${VLLM_PY}" "${FREEZER}" >/dev/null
  "${VLLM_PY}" -m py_compile "${RUNNER}" "${VERIFY}" "${FREEZER}"
  "${VLLM_PY}" -m pytest -q \
    "${VLLM_ROOT}/tests/test_cpu_offload_exact_static.py"
  bash -n "${BASH_SOURCE[0]}"
  modinfo -p "${CUSTOM_UVM}" | grep '^uvm_manual_migration_fault_mode:' >/dev/null \
    || die "custom UVM lacks manual migration mode"
  echo "static preflight: OK"
}

runtime_preflight() {
  static_preflight
  [[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
  STOCK_UVM="$(modinfo -F filename nvidia-uvm)"
  STOCK_SRCVERSION="$(modinfo -F srcversion nvidia-uvm | tr -d '[:space:]')"
  CUSTOM_SRCVERSION="$(modinfo -F srcversion "${CUSTOM_UVM}" | tr -d '[:space:]')"
  [[ "$(loaded_srcversion)" == "${STOCK_SRCVERSION}" ]] \
    || die "initial nvidia_uvm is not stock"
  exec 9>"/tmp/vllm_uvm_cold5_alias_prefill.lock"
  flock -n 9 || die "another cold-5 run holds the lock"
  stop_exporters
  wait_gpu_clean || die "GPU/UVM is not clean"
  MODULE_CYCLE_ARMED=1
}

load_custom_module() {
  wait_gpu_clean || die "GPU changed before custom module load"
  sudo modprobe -r nvidia_uvm
  sudo modprobe nvidia
  sudo insmod "${CUSTOM_UVM}" \
    uvm_moe_slot_pte_remap_enable=1 \
    uvm_moe_slot_pool_vidmem_budget_bytes="${SLOT_BUDGET_BYTES}" \
    uvm_manual_migration_fault_mode=2
  [[ "$(loaded_srcversion)" == "${CUSTOM_SRCVERSION}" ]] \
    || die "custom UVM srcversion differs"
  [[ "$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode)" == 2 ]] \
    || die "manual migration mode is not 2"
  wait_gpu_clean || die "custom UVM load left GPU state"
}

arm_name() {
  case "$1" in
    T) echo route_trace_control ;;
    Z) echo all_gpu_graph ;;
    A) echo alias_eager ;;
    G) echo alias_graph ;;
    *) die "unknown arm $1" ;;
  esac
}

run_arm() {
  local symbol="$1" arm_dir="$2" batch_order="$3"
  local arm execution capture_sizes offload_gb alias alias_eager trace copy_trace
  arm="$(arm_name "${symbol}")"
  mkdir "${arm_dir}"
  : >"${arm_dir}/cpu_offload_manifest.jsonl"
  : >"${arm_dir}/expert_trace.jsonl"
  wait_gpu_clean || die "GPU is not clean before ${arm}"
  snapshot_gpu "${arm_dir}/gpu.before.json"

  execution=eager
  capture_sizes=1,4
  offload_gb=0
  alias=0
  alias_eager=0
  trace=0
  copy_trace=0
  case "${symbol}" in
    T) trace=1 ;;
    Z) execution=full_graph_all; capture_sizes=1,4,314,1217 ;;
    A) alias=1; alias_eager=1; offload_gb=1.25 ;;
    G) alias=1; offload_gb=1.25; execution=full_graph_all; capture_sizes=1,4,314,1217 ;;
  esac
  if [[ "${alias}" == 1 && "${PERFORMANCE_COPY_TRACE}" == 1 ]]; then
    copy_trace=1
  fi

  local -a envs=(
    VLLM_USE_UVM=0
    VLLM_ENABLE_V1_MULTIPROCESSING=0
    VLLM_CPU_OFFLOAD_BACKEND=copy
    VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="$([[ "${alias}" == 1 ]] && echo "${NAMES}" || true)"
    VLLM_CPU_OFFLOAD_STATIC_BANK=0
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS="${alias}"
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER="${alias_eager}"
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND="$([[ "${alias}" == 1 ]] && echo uvm_original_va || echo vmm)"
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="$([[ "${alias}" == 1 ]] && echo "${PLAN}" || true)"
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=4k
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPECTED_ARENA_COUNT=4
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPECTED_BACKING_BYTES="${BACKING_BYTES}"
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED="${alias}"
    VLLM_UVM_ORIGINAL_VA_PHASE_MODE=off
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
    VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
    VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0
    VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0
    VLLM_CPU_OFFLOAD_COPY_TRACE="${copy_trace}"
    VLLM_CPU_OFFLOAD_COPY_SYNC=0
    VLLM_CPU_OFFLOAD_MANIFEST_FILE="${arm_dir}/cpu_offload_manifest.jsonl"
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1
    VLLM_UVM_LOG_FILE="${arm_dir}/uvm_allocations.log"
    VLLM_COLD5_ARM="${symbol}"
    VLLM_MOE_TOKEN_EXPERT_TRACE_ENABLE="${trace}"
    VLLM_MOE_TOKEN_EXPERT_TRACE_FILE="${arm_dir}/expert_trace.jsonl"
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    TOKENIZERS_PARALLELISM=false
    OMP_NUM_THREADS="${CPU_THREADS}"
    MKL_NUM_THREADS="${CPU_THREADS}"
    CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
    VLLM_ATTENTION_BACKEND=TRITON_ATTN
  )
  printf '%s\n' "${envs[@]}" >"${arm_dir}/environment.txt"

  local model="${MODEL:-${DEFAULT_MODEL}}"
  local fixture="${EVAL_FIXTURE:-${DEFAULT_FIXTURE}}"
  local -a args=(
    --model "${model}"
    --fixture "${fixture}"
    --output "${arm_dir}/result.json"
    --offload-manifest "${arm_dir}/cpu_offload_manifest.jsonl"
    --profile-label "qwen_cold5_${arm}"
    --mechanism-class "vllm_${arm}_cold5"
    --execution-mode "${execution}"
    --batch-sizes "${batch_order}"
    --cudagraph-capture-sizes "${capture_sizes}"
    --conditioning-repeats "${CONDITIONING_REPEATS}"
    --measurement-repeats "${MEASUREMENT_REPEATS}"
    --warmup-decode-steps "${WARMUP_DECODE_STEPS}"
    --measured-decode-steps "${MEASURED_DECODE_STEPS}"
    --kv-cache-memory-bytes "${KV_CACHE_BYTES}"
    --cpu-offload-gb "${offload_gb}"
    --max-model-len 512
    --max-num-seqs 4
    --max-num-batched-tokens 2048
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.97}"
    --seed 0
  )
  {
    printf 'cd %q && env ' "${VLLM_ROOT}"
    printf '%q ' "${envs[@]}"
    printf 'timeout --kill-after=30s %qs taskset -c %q %q %q ' \
      "${TIMEOUT_SEC}" "${CPUSET}" "${VLLM_PY}" "${RUNNER}"
    printf '%q ' "${args[@]}"
    printf '\n'
  } >"${arm_dir}/command.txt"

  local rc
  set +e
  (
    cd "${VLLM_ROOT}"
    env "${envs[@]}" /usr/bin/time -v -o "${arm_dir}/resource_time.txt" \
      timeout --kill-after=30s "${TIMEOUT_SEC}s" taskset -c "${CPUSET}" \
      "${VLLM_PY}" "${RUNNER}" "${args[@]}"
  ) >"${arm_dir}/run.log" 2>&1
  rc=$?
  set -e
  echo "${rc}" >"${arm_dir}/returncode.txt"
  wait_gpu_clean || die "GPU did not clean after ${arm}"
  snapshot_gpu "${arm_dir}/gpu.after.json"
  return "${rc}"
}

run_suite() {
  local mode="$1" order
  runtime_preflight
  require_file "${MODEL:-${DEFAULT_MODEL}}/config.json"
  require_file "${EVAL_FIXTURE:-${DEFAULT_FIXTURE}}"
  case "${mode}" in
    smoke)
      order="${ORDER:-G}"
      CONDITIONING_REPEATS="${CONDITIONING_REPEATS:-0}"
      MEASUREMENT_REPEATS="${MEASUREMENT_REPEATS:-1}"
      WARMUP_DECODE_STEPS="${WARMUP_DECODE_STEPS:-1}"
      MEASURED_DECODE_STEPS="${MEASURED_DECODE_STEPS:-2}"
      PERFORMANCE_COPY_TRACE="${PERFORMANCE_COPY_TRACE:-1}"
      ;;
    qualify)
      order="${ORDER:-TZAG}"
      CONDITIONING_REPEATS="${CONDITIONING_REPEATS:-1}"
      MEASUREMENT_REPEATS="${MEASUREMENT_REPEATS:-2}"
      WARMUP_DECODE_STEPS="${WARMUP_DECODE_STEPS:-2}"
      MEASURED_DECODE_STEPS="${MEASURED_DECODE_STEPS:-4}"
      PERFORMANCE_COPY_TRACE="${PERFORMANCE_COPY_TRACE:-1}"
      ;;
    formal)
      # Keep the historical three-process order by default, but allow the
      # publication run to use an even number of independent cold processes
      # per performance arm.  Each adjacent arm pair receives both 1,4 and
      # 4,1 batch orders; rotating the pair order limits temporal arm bias.
      local formal_processes="${FORMAL_PROCESSES_PER_ARM:-3}"
      [[ "${formal_processes}" =~ ^[0-9]+$ ]] \
        || die "FORMAL_PROCESSES_PER_ARM must be an integer"
      (( formal_processes >= 3 )) \
        || die "FORMAL_PROCESSES_PER_ARM must be at least 3"
      if [[ -n "${ORDER:-}" ]]; then
        order="${ORDER}"
      elif (( formal_processes == 3 )); then
        order=TZAGGZAAZG
      else
        (( formal_processes % 2 == 0 )) \
          || die "FORMAL_PROCESSES_PER_ARM above 3 must be even"
        order=T
        local pair_block
        for ((pair_block = 0; pair_block < formal_processes / 2; ++pair_block)); do
          case $((pair_block % 3)) in
            0) order+=ZZAAGG ;;
            1) order+=AAGGZZ ;;
            2) order+=GGZZAA ;;
          esac
        done
      fi
      CONDITIONING_REPEATS="${CONDITIONING_REPEATS:-1}"
      MEASUREMENT_REPEATS="${MEASUREMENT_REPEATS:-3}"
      WARMUP_DECODE_STEPS="${WARMUP_DECODE_STEPS:-5}"
      MEASURED_DECODE_STEPS="${MEASURED_DECODE_STEPS:-8}"
      # Performance samples must not include per-forward JSON trace I/O.  The
      # alias geometry is certified from driver/runtime diagnostics instead.
      PERFORMANCE_COPY_TRACE="${PERFORMANCE_COPY_TRACE:-0}"
      ;;
    *) die "unknown mode ${mode}" ;;
  esac
  [[ "${order}" =~ ^[TZAG]+$ ]] || die "ORDER contains an unknown arm"
  [[ "${PERFORMANCE_COPY_TRACE}" =~ ^[01]$ ]] \
    || die "PERFORMANCE_COPY_TRACE must be 0 or 1"

  local run_root="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_${mode}}"
  [[ ! -e "${run_root}" ]] || die "RUN_ROOT already exists"
  mkdir -p "${run_root}"
  run_root="$(cd "${run_root}" && pwd)"
  RUN_ROOT_ACTIVE="${run_root}"
  printf '%s\n' "${ACTIVE_SERVICES[@]}" >"${run_root}/services.initially_active.txt"
  snapshot_gpu "${run_root}/gpu.before.json"
  load_custom_module
  snapshot_gpu "${run_root}/gpu.custom.json"
  sha256sum "${PLAN}" "${NAMES}" "${FREEZER}" "${RUNNER}" "${VERIFY}" \
    "${BASE_RUNNER}" "${VLLM_ROOT}/vllm/model_executor/models/utils.py" \
    "${CUSTOM_UVM}" "${STOCK_UVM}" >"${run_root}/input_sha256.txt"
  git -C "${REPO_ROOT}" status --short >"${run_root}/git_status.txt"
  "${VLLM_PY}" - "${run_root}/run_config.json" "${mode}" "${order}" \
    "${CONDITIONING_REPEATS}" "${MEASUREMENT_REPEATS}" \
    "${WARMUP_DECODE_STEPS}" "${MEASURED_DECODE_STEPS}" \
    "${MODEL:-${DEFAULT_MODEL}}" "${EVAL_FIXTURE:-${DEFAULT_FIXTURE}}" \
    "${PERFORMANCE_COPY_TRACE}" <<'PY'
import hashlib, json, sys
from pathlib import Path

(
    path,
    mode,
    order,
    conditioning_repeats,
    measurement_repeats,
    warmup_decode_steps,
    measured_decode_steps,
    model,
    fixture,
    performance_copy_trace,
) = sys.argv[1:]
model_path = Path(model).resolve()
fixture_path = Path(fixture).resolve()
json.dump({
    "schema": "vllm_uvm_cold5_alias_prefill_run_v1",
    "mode": mode,
    "order": order,
    "formal_processes_per_arm": (
        order.count("Z") if mode == "formal" else None
    ),
    "batch_sizes": [1, 4],
    "conditioning_repeats": int(conditioning_repeats),
    "measurement_repeats": int(measurement_repeats),
    "warmup_decode_steps": int(warmup_decode_steps),
    "measured_decode_steps": int(measured_decode_steps),
    "performance_copy_trace": int(performance_copy_trace),
    "model": str(model_path),
    "model_config_sha256": hashlib.sha256(
        (model_path / "config.json").read_bytes()
    ).hexdigest(),
    "fixture": str(fixture_path),
    "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
    "selected_experts_per_layer": 5,
    "layer_count": 48,
    "parameter_count": 96,
    "selected_bytes_per_prefill": 1_132_462_080,
    "backing_bytes": 47_185_920,
    "manual_migration_fault_mode": 2,
}, open(path, "w", encoding="utf-8"), indent=2, sort_keys=True)
open(path, "a", encoding="utf-8").write("\n")
PY
  printf 'ordinal\tsymbol\tarm\tbatch_order\tdirectory\treturncode\n' \
    >"${run_root}/execution_status.tsv"

  local failures=0 index symbol arm batch_order arm_dir rc
  for ((index = 0; index < ${#order}; ++index)); do
    symbol="${order:index:1}"
    arm="$(arm_name "${symbol}")"
    batch_order=1,4
    (( index % 2 == 1 )) && batch_order=4,1
    arm_dir="${run_root}/$(printf '%02d_%s' "$((index + 1))" "${arm}")"
    if run_arm "${symbol}" "${arm_dir}" "${batch_order}"; then
      rc=0
    else
      rc=$?
      failures=$((failures + 1))
      tail -n 160 "${arm_dir}/run.log" >&2 || true
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$((index + 1))" "${symbol}" \
      "${arm}" "${batch_order}" "$(basename "${arm_dir}")" "${rc}" \
      >>"${run_root}/execution_status.tsv"
  done

  restore_stock || die "failed to restore stock UVM"
  snapshot_gpu "${run_root}/gpu.stock.json"
  restart_exporters
  snapshot_services "${run_root}/services.after.tsv"
  while IFS= read -r service; do
    [[ -z "${service}" ]] && continue
    [[ "$(systemctl is-active "${service}" 2>/dev/null || true)" == active ]] \
      || die "failed to restore ${service}"
  done <"${run_root}/services.initially_active.txt"
  if [[ "${mode}" != smoke && ${failures} -eq 0 ]]; then
    "${VLLM_PY}" "${VERIFY}" "${run_root}" \
      --output "${run_root}/verification.json" >"${run_root}/verify.log"
  fi
  echo "RUN_ROOT=${run_root}"
  (( failures == 0 )) || die "${failures} arm(s) failed"
}

case "${1:-help}" in
  static-preflight) static_preflight ;;
  smoke|qualify|formal) run_suite "$1" ;;
  verify)
    [[ $# == 2 ]] || die "verify requires RUN_ROOT"
    "${VLLM_PY}" "${VERIFY}" "$2"
    ;;
  *) echo "usage: $0 {static-preflight|smoke|qualify|formal|verify RUN_ROOT}" >&2; exit 2 ;;
esac
