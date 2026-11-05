#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_PY:-${VLLM_ROOT}/.venv/bin/python}"
RUNNER="${RUNNER_OVERRIDE:-${SCRIPT_DIR}/benchmark_original_va_graph_steps.py}"
VERIFY="${VERIFY_OVERRIDE:-${SCRIPT_DIR}/verify_integration.py}"
PREPARE="${REPO_ROOT}/experiments/vllm_slot_vs_pinned_copy/prepare_inputs.py"
BASE_RUNNER="${REPO_ROOT}/experiments/vllm_p3_static_alias_graph_integration/benchmark_full_graph_steps.py"
PHASE_RUNNER="${REPO_ROOT}/experiments/vllm_p3_pinned_uva_phase_compare/benchmark_phase_steps.py"
UVM_ALLOCATOR_SO="${VLLM_ROOT}/uvm_test/uvm_allocator.so"
CUSTOM_UVM="${CUSTOM_UVM:-${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
DEFAULT_MODEL="${REPO_ROOT}/../.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782"
DEFAULT_FIXTURE="${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json"

DUAL_BANK_BYTES=1207959552
LOGICAL_BYTES=2415919104
SLOT_BUDGET_BYTES="${SLOT_BUDGET_BYTES:-${DUAL_BANK_BYTES}}"
GPU_INDEX="${GPU_INDEX:-0}"
CPUSET="${CPUSET:-0-13}"
CPU_THREADS="${CPU_THREADS:-14}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"
GPU_CLEAN_TIMEOUT_SEC="${GPU_CLEAN_TIMEOUT_SEC:-240}"
# The original-VA arm retains 1.125 GiB of dual-bank backing in addition to
# the resident non-selected model weights. 256 MiB leaves less than one MoE
# scratch allocation free on this 32 GiB GPU; 160 MiB still covers the frozen
# 4-request workload (at most 1,536 live prompt+output tokens).
KV_CACHE_BYTES="${KV_CACHE_BYTES:-167772160}"
MANUAL_MIGRATION_FAULT_MODE=2

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
  [[ "${used}" == "0" && -z "${pids}" && -z "${clients}" ]]
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
  [[ "${used}" == "0" && -z "${pids}" && -z "${clients}" ]] && clean=true
  "${VLLM_PY}" - "${output}" "${GPU_INDEX}" "${used:-unknown}" \
    "${pids}" "${clients}" "${loaded}" "${clean}" <<'PY'
import json, sys, time
path, gpu, used, pids, clients, loaded, clean = sys.argv[1:]
payload = {
    "schema": "vllm_uvm_original_va_gpu_state_v1",
    "timestamp": time.time(),
    "gpu_index": int(gpu),
    "memory_used_mib": int(used) if used.isdigit() else used,
    "compute_pids": [int(x) for x in pids.split(",") if x],
    "uvm_client_pids": [int(x) for x in clients.split(",") if x],
    "loaded_srcversion": loaded or None,
    "gpu_clean": clean == "true",
}
open(path, "w", encoding="utf-8").write(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY
}

module_snapshot() {
  local output="$1" module_path="$2" used pids clients loaded clean=false
  local module_sha="" budget="" remap="" manual_fault_mode=""
  [[ -f "${module_path}" ]] && module_sha="$(sha256sum "${module_path}" | awk '{print $1}')"
  used="$(gpu_used_mib 2>/dev/null || true)"
  pids="$(compute_pids 2>/dev/null | paste -sd, - || true)"
  clients="$(uvm_clients 2>/dev/null | paste -sd, - || true)"
  loaded="$(loaded_srcversion)"
  [[ "${used}" == "0" && -z "${pids}" && -z "${clients}" ]] && clean=true
  [[ -r /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes ]] \
    && budget="$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes)"
  [[ -r /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable ]] \
    && remap="$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable)"
  [[ -r /sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode ]] \
    && manual_fault_mode="$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode)"
  "${VLLM_PY}" - "${output}" "${module_path}" "${module_sha}" \
    "${loaded}" "${used:-unknown}" "${pids}" "${clients}" "${clean}" \
    "${budget}" "${remap}" "${manual_fault_mode}" <<'PY'
import json, os, sys, time
(path, module_path, module_sha, loaded, used, pids, clients, clean,
 budget, remap, manual_fault_mode) = sys.argv[1:]
payload = {
    "schema": "vllm_uvm_original_va_module_state_v1",
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
    "manual_migration_fault_mode": (
        int(manual_fault_mode)
        if manual_fault_mode.lstrip("-").isdigit()
        else None
    ),
}
open(path, "w", encoding="utf-8").write(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
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
  local loaded rc=0
  if (( MODULE_CYCLE_ARMED == 0 )); then
    MODULE_RESTORED=1
    return 0
  fi
  (( MODULE_RESTORED == 0 )) || return 0
  set +e
  loaded="$(loaded_srcversion)"
  if [[ "${loaded}" != "${STOCK_SRCVERSION}" ]]; then
    if ! wait_gpu_clean; then
      echo "recovery: GPU/client state is not clean; refusing UVM unload" >&2
      rc=1
    else
      sudo modprobe -r nvidia_uvm || rc=1
      sudo modprobe nvidia || rc=1
      sudo modprobe nvidia_uvm || rc=1
    fi
  fi
  loaded="$(loaded_srcversion)"
  [[ "${loaded}" == "${STOCK_SRCVERSION}" ]] || rc=1
  wait_gpu_clean || rc=1
  (( rc == 0 )) && MODULE_RESTORED=1
  set -e
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
  require_file "${PREPARE}"
  require_file "${BASE_RUNNER}"
  require_file "${PHASE_RUNNER}"
  require_file "${UVM_ALLOCATOR_SO}"
  require_file "${CUSTOM_UVM}"
  if [[ -n "${TOP_LEVEL_HARNESS:-}" ]]; then
    require_file "${TOP_LEVEL_HARNESS}"
  fi
  [[ -x "${VLLM_PY}" ]] || die "missing vLLM Python: ${VLLM_PY}"
  [[ "${SLOT_BUDGET_BYTES}" =~ ^[0-9]+$ ]] || die "SLOT_BUDGET_BYTES is not an integer"
  [[ "${KV_CACHE_BYTES}" =~ ^[0-9]+$ ]] || die "KV_CACHE_BYTES is not an integer"
  (( SLOT_BUDGET_BYTES >= DUAL_BANK_BYTES )) \
    || die "slot budget must be at least ${DUAL_BANK_BYTES} bytes"
  (( KV_CACHE_BYTES > 0 )) || die "KV_CACHE_BYTES must be positive"
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  command -v modinfo >/dev/null || die "modinfo is unavailable"
  command -v flock >/dev/null || die "flock is unavailable"
  command -v timeout >/dev/null || die "timeout is unavailable"
  command -v taskset >/dev/null || die "taskset is unavailable"
  command -v sudo >/dev/null || die "sudo is unavailable"
  modinfo -p "${CUSTOM_UVM}" | grep -q '^uvm_manual_migration_fault_mode:' \
    || die "custom UVM lacks uvm_manual_migration_fault_mode"
  "${VLLM_PY}" -m py_compile "${RUNNER}" "${VERIFY}"
  bash -n "${BASH_SOURCE[0]}"
  "${VLLM_PY}" - "${RUNNER}" <<'PY'
import importlib.util, pathlib, sys
path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("harness_runner_static_test", path)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module.DUAL_BANK_BYTES == 1_207_959_552
assert module.LOGICAL_BYTES == 2_415_919_104
PY
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
  [[ -n "${STOCK_SRCVERSION}" && -n "${CUSTOM_SRCVERSION}" ]] \
    || die "module srcversion is absent"
  [[ "$(loaded_srcversion)" == "${STOCK_SRCVERSION}" ]] \
    || die "the initially loaded nvidia_uvm is not stock"
  MODULE_CYCLE_ARMED=1
  capture_and_stop_exporters
  wait_gpu_clean || die "GPU must start at exactly 0 MiB with no UVM client"
  exec 9>"/tmp/vllm_uvm_original_va_graph_integration.lock"
  flock -n 9 || die "another original-VA integration run holds the lock"
}

load_custom_module() {
  wait_gpu_clean || die "GPU/client state changed before custom module load"
  sudo modprobe -r nvidia_uvm
  sudo modprobe nvidia
  sudo insmod "${CUSTOM_UVM}" \
    uvm_moe_slot_pte_remap_enable=1 \
    uvm_moe_slot_pool_vidmem_budget_bytes="${SLOT_BUDGET_BYTES}" \
    uvm_manual_migration_fault_mode="${MANUAL_MIGRATION_FAULT_MODE}"
  [[ "$(loaded_srcversion)" == "${CUSTOM_SRCVERSION}" ]] \
    || die "loaded custom nvidia_uvm srcversion differs"
  wait_gpu_clean || die "custom module load left a GPU allocation or client"
  [[ "$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable)" == 1 ]] \
    || die "slot PTE remap parameter was not enabled"
  [[ "$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode)" == "${MANUAL_MIGRATION_FAULT_MODE}" ]] \
    || die "manual migration fail-fast mode was not enabled"
  local realized_budget
  realized_budget="$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes)"
  (( realized_budget >= DUAL_BANK_BYTES )) || die "realized slot budget is too small"
}

arm_name() {
  case "$1" in
    E) echo eager_pinned ;;
    U) echo uvm_original_va_graph ;;
    N) echo native_copy_eager ;;
    A) echo alias_eager ;;
    G) echo alias_full_graph ;;
    R) echo remote_eager ;;
    C) echo remote_decode_graph ;;
    F) echo native_copy_full_graph_control ;;
    *) die "unknown arm symbol: $1" ;;
  esac
}

validate_order() {
  local order="$1" required="$2" block_len index block offset
  [[ "${order}" =~ ^[EUNAGRCF]+$ ]] || die "ORDER contains an unknown symbol"
  block_len="${#required}"
  (( block_len > 0 && ${#order} % block_len == 0 )) \
    || die "ORDER does not contain complete ${required} blocks"
  for ((index = 0; index < ${#order}; index += block_len)); do
    block="${order:index:block_len}"
    for ((offset = 0; offset < block_len; ++offset)); do
      [[ "${block}" == *"${required:offset:1}"* ]] \
        || die "ORDER block ${block} is not a ${required} permutation"
    done
  done
}

reverse_csv() {
  "${VLLM_PY}" - "$1" <<'PY'
import sys
print(",".join(reversed([x.strip() for x in sys.argv[1].split(",") if x.strip()])))
PY
}

run_arm() {
  local symbol="$1" arm_dir="$2" names="$3" batch_order="$4"
  local arm execution profile mechanism alias managed alias_eager phase_mode
  local capture_sizes
  arm="$(arm_name "${symbol}")"
  mkdir "${arm_dir}"
  : >"${arm_dir}/cpu_offload_manifest.jsonl"
  wait_gpu_clean || die "GPU is not clean before ${arm}"
  gpu_state_json "${arm_dir}/gpu.before.json"

  alias_eager=0
  phase_mode=off
  capture_sizes=1,4
  case "${symbol}" in
    U)
      execution=full_graph_decode
      profile=qwen_l0_l3_uvm_original_va_full_graph
      mechanism=vllm_uvm_original_va_static_alias_dual_bank_l0_l3
      alias=1
      managed=1
      ;;
    E)
      execution=eager
      profile=qwen_l0_l3_eager_pinned
      mechanism=vllm_pinned_copy_exact_l0_l3
      alias=0
      managed=0
      ;;
    N)
      execution=eager
      profile=qwen_l0_l3_native_copy_eager
      mechanism=vllm_native_copy_on_forward_exact_l0_l3
      alias=0
      managed=0
      if [[ "${N_FULL_GRAPH_PROBE:-0}" == 1 ]]; then
        execution=full_graph_all
        capture_sizes=1,4,314,1217
      fi
      ;;
    F)
      execution=full_graph_all
      profile=qwen_l0_l3_native_copy_full_graph_control
      mechanism=vllm_native_copy_on_forward_full_graph_control_l0_l3
      alias=0
      managed=0
      capture_sizes=1,4,314,1217
      ;;
    A)
      execution=eager
      profile=qwen_l0_l3_uvm_original_va_alias_eager
      mechanism=vllm_uvm_original_va_static_alias_eager_l0_l3
      alias=1
      managed=1
      alias_eager=1
      ;;
    G)
      execution=full_graph_all
      profile=qwen_l0_l3_uvm_original_va_alias_full_graph
      mechanism=vllm_uvm_original_va_static_alias_full_graph_l0_l3
      alias=1
      managed=1
      phase_mode="${G_PHASE_MODE_OVERRIDE:-off}"
      capture_sizes=1,4,314,1217
      ;;
    R)
      execution=eager
      profile=qwen_l0_l3_uvm_original_va_remote_eager
      mechanism=vllm_uvm_original_va_remote_read_eager_l0_l3
      alias=1
      managed=1
      phase_mode=remote_only
      ;;
    C)
      execution=full_graph_decode
      profile=qwen_l0_l3_uvm_original_va_remote_graph
      mechanism=vllm_uvm_original_va_remote_read_graph_l0_l3
      alias=1
      managed=1
      phase_mode=remote_only
      ;;
    *) die "unknown arm symbol: ${symbol}" ;;
  esac

  local -a common_env=(
    VLLM_USE_UVM=0
    VLLM_ENABLE_V1_MULTIPROCESSING=0
    VLLM_CPU_OFFLOAD_BACKEND=copy
    VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${names}"
    VLLM_CPU_OFFLOAD_STATIC_BANK=0
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS="${alias}"
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER="${alias_eager}"
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND="$([[ "${alias}" == 1 ]] && echo uvm_original_va || echo vmm)"
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES="${REGION_BYTES:-524288}"
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE="${PTE_MODE:-auto}"
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED="${managed}"
    VLLM_UVM_ORIGINAL_VA_PHASE_MODE="${phase_mode}"
    VLLM_PREFILL_COMPARE_ARM="${symbol}"
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
    VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
    VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0
    VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0
    VLLM_CPU_OFFLOAD_COPY_TRACE="${COPY_TRACE}"
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
  if [[ -n "${ATTENTION_BACKEND:-}" ]]; then
    common_env+=(VLLM_ATTENTION_BACKEND="${ATTENTION_BACKEND}")
  fi
  printf '%s\n' "${common_env[@]}" >"${arm_dir}/environment.txt"

  local model="${MODEL:-${DEFAULT_MODEL}}" fixture="${EVAL_FIXTURE:-${DEFAULT_FIXTURE}}"
  local -a args=(
    --model "${model}"
    --fixture "${fixture}"
    --output "${arm_dir}/result.json"
    --offload-manifest "${arm_dir}/cpu_offload_manifest.jsonl"
    --profile-label "${profile}"
    --mechanism-class "${mechanism}"
    --execution-mode "${execution}"
    --batch-sizes "${batch_order}"
    --cudagraph-capture-sizes "${capture_sizes}"
    --conditioning-repeats "${CONDITIONING_REPEATS}"
    --measurement-repeats "${MEASUREMENT_REPEATS}"
    --warmup-decode-steps "${WARMUP_DECODE_STEPS}"
    --measured-decode-steps "${MEASURED_DECODE_STEPS}"
    --kv-cache-memory-bytes "${KV_CACHE_BYTES}"
    --cpu-offload-gb 3.0
    --max-model-len "${MAX_MODEL_LEN:-512}"
    --max-num-seqs 4
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.97}"
    --seed 0
  )
  {
    printf 'cd %q && env ' "${VLLM_ROOT}"
    printf '%q ' "${common_env[@]}"
    printf 'timeout --kill-after=30s %qs taskset -c %q %q %q ' \
      "${TIMEOUT_SEC}" "${CPUSET}" "${VLLM_PY}" "${RUNNER}"
    printf '%q ' "${args[@]}"
    printf '\n'
  } >"${arm_dir}/command.txt"

  local rc
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
  set -e
  echo "${rc}" >"${arm_dir}/returncode.txt"
  if ! wait_gpu_clean; then
    gpu_state_json "${arm_dir}/gpu.after.json"
    die "GPU/client state did not cleanly shut down after ${arm}"
  fi
  gpu_state_json "${arm_dir}/gpu.after.json"
  return "${rc}"
}

write_run_config() {
  local output="$1" mode="$2" order="$3" required="$4" model="$5" fixture="$6"
  "${VLLM_PY}" - "${output}" "${mode}" "${order}" "${required}" \
    "${model}" "${fixture}" "${CONDITIONING_REPEATS}" \
    "${MEASUREMENT_REPEATS}" "${WARMUP_DECODE_STEPS}" \
    "${MEASURED_DECODE_STEPS}" "${KV_CACHE_BYTES}" "${SLOT_BUDGET_BYTES}" \
    "${MANUAL_MIGRATION_FAULT_MODE}" "${STOCK_UVM}" \
    "${STOCK_SRCVERSION}" "${STOCK_SHA256}" "${CUSTOM_UVM}" \
    "${CUSTOM_SRCVERSION}" "${CUSTOM_SHA256}" "${REGION_BYTES:-524288}" \
    "${PTE_MODE:-auto}" "${ATTENTION_BACKEND:-}" \
    "${EXTENDED_PREFILL_MATRIX:-0}" <<'PY'
import json, os, sys
(path, mode, order, required, model, fixture, conditioning, measurement,
 warmup, measured, kv_cache_bytes, budget, manual_fault_mode, stock_path,
 stock_src, stock_sha, custom_path,
 custom_src, custom_sha, region_bytes, pte_mode, attention_backend,
 extended_prefill_matrix) = sys.argv[1:]

arm_catalog = {
    "E": {
        "arm": "eager_pinned",
        "execution_mode": "eager",
        "cudagraph_mode": "NONE",
        "cudagraph_capture_sizes": [],
        "cpu_offload_backend": "copy",
        "graph_static_alias": False,
        "graph_static_alias_backend": None,
        "managed_expert_weights": False,
        "original_va_phase_mode": "off",
    },
    "U": {
        "arm": "uvm_original_va_graph",
        "execution_mode": "full_graph_decode",
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 4],
        "cpu_offload_backend": "copy",
        "graph_static_alias": True,
        "graph_static_alias_backend": "uvm_original_va",
        "managed_expert_weights": True,
        "original_va_phase_mode": "off",
    },
    "N": {
        "arm": "native_copy_eager",
        "profile_label": "qwen_l0_l3_native_copy_eager",
        "mechanism_class": "vllm_native_copy_on_forward_exact_l0_l3",
        "execution_mode": "eager",
        "cudagraph_mode": "NONE",
        "cudagraph_capture_sizes": [],
        "cpu_offload_backend": "copy",
        "graph_static_alias": False,
        "graph_static_alias_backend": None,
        "managed_expert_weights": False,
        "original_va_phase_mode": "off",
    },
    "A": {
        "arm": "alias_eager",
        "profile_label": "qwen_l0_l3_uvm_original_va_alias_eager",
        "mechanism_class": "vllm_uvm_original_va_static_alias_eager_l0_l3",
        "execution_mode": "eager",
        "cudagraph_mode": "NONE",
        "cudagraph_capture_sizes": [],
        "cpu_offload_backend": "copy",
        "graph_static_alias": True,
        "graph_static_alias_backend": "uvm_original_va",
        "managed_expert_weights": True,
        "original_va_phase_mode": "off",
    },
    "G": {
        "arm": "alias_full_graph",
        "profile_label": "qwen_l0_l3_uvm_original_va_alias_full_graph",
        "mechanism_class": "vllm_uvm_original_va_static_alias_full_graph_l0_l3",
        "execution_mode": "full_graph_all",
        "cudagraph_mode": "FULL",
        "cudagraph_capture_sizes": [1, 4, 314, 1217],
        "cpu_offload_backend": "copy",
        "graph_static_alias": True,
        "graph_static_alias_backend": "uvm_original_va",
        "managed_expert_weights": True,
        "original_va_phase_mode": "off",
    },
    "R": {
        "arm": "remote_eager",
        "profile_label": "qwen_l0_l3_uvm_original_va_remote_eager",
        "mechanism_class": "vllm_uvm_original_va_remote_read_eager_l0_l3",
        "execution_mode": "eager",
        "cudagraph_mode": "NONE",
        "cudagraph_capture_sizes": [],
        "cpu_offload_backend": "copy",
        "graph_static_alias": True,
        "graph_static_alias_backend": "uvm_original_va",
        "managed_expert_weights": True,
        "original_va_phase_mode": "remote_only",
    },
    "C": {
        "arm": "remote_decode_graph",
        "profile_label": "qwen_l0_l3_uvm_original_va_remote_graph",
        "mechanism_class": "vllm_uvm_original_va_remote_read_graph_l0_l3",
        "execution_mode": "full_graph_decode",
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 4],
        "cpu_offload_backend": "copy",
        "graph_static_alias": True,
        "graph_static_alias_backend": "uvm_original_va",
        "managed_expert_weights": True,
        "original_va_phase_mode": "remote_only",
    },
    "F": {
        "arm": "native_copy_full_graph_control",
        "profile_label": "qwen_l0_l3_native_copy_full_graph_control",
        "mechanism_class": "vllm_native_copy_on_forward_full_graph_control_l0_l3",
        "execution_mode": "full_graph_all",
        "cudagraph_mode": "FULL",
        "cudagraph_capture_sizes": [1, 4, 314, 1217],
        "cpu_offload_backend": "copy",
        "graph_static_alias": False,
        "graph_static_alias_backend": None,
        "managed_expert_weights": False,
        "original_va_phase_mode": "off",
    },
}
symbols = sorted(set(order), key=order.index)
arm_configs = {symbol: arm_catalog[symbol] for symbol in symbols}
extended = extended_prefill_matrix == "1"
payload = {
    "schema": "vllm_uvm_original_va_graph_run_v1",
    "mode": mode,
    "order": order,
    "required_arms": required,
    "model": os.path.realpath(model),
    "fixture": os.path.realpath(fixture),
    "batch_sizes": [1, 4],
    "cudagraph_capture_sizes": None if extended else [1, 4],
    "cudagraph_mode": "PER_ARM" if extended else "FULL_DECODE_ONLY",
    "arm_configs": arm_configs,
    "attention_backend": attention_backend or None,
    "extended_prefill_matrix": extended,
    "conditioning_repeats": int(conditioning),
    "measurement_repeats": int(measurement),
    "warmup_decode_steps": int(warmup),
    "measured_decode_steps": int(measured),
    "kv_cache_memory_bytes": int(kv_cache_bytes),
    "layers": [0, 1, 2, 3],
    "parameter_count": 8,
    "logical_bytes_per_forward": 2_415_919_104,
    "dual_bank_backing_bytes": 1_207_959_552,
    "slot_budget_bytes": int(budget),
    "manual_migration_fault_mode": int(manual_fault_mode),
    "region_bytes": int(region_bytes),
    "pte_mode": pte_mode,
    "module_cycle": True,
    "gpu_clean_required": True,
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

run_suite() {
  local mode="$1" order required
  runtime_preflight
  local model="${MODEL:-${DEFAULT_MODEL}}" fixture="${EVAL_FIXTURE:-${DEFAULT_FIXTURE}}"
  require_file "${model}/config.json"
  require_file "${fixture}"

  case "${mode}" in
    smoke)
      order="${ORDER:-U}"
      required="${REQUIRED_ARMS:-U}"
      CONDITIONING_REPEATS="${CONDITIONING_REPEATS:-0}"
      MEASUREMENT_REPEATS="${MEASUREMENT_REPEATS:-1}"
      WARMUP_DECODE_STEPS="${WARMUP_DECODE_STEPS:-1}"
      MEASURED_DECODE_STEPS="${MEASURED_DECODE_STEPS:-2}"
      COPY_TRACE="${COPY_TRACE:-1}"
      ;;
    qualification)
      order="${ORDER:-EU}"
      required="${REQUIRED_ARMS:-EU}"
      CONDITIONING_REPEATS="${CONDITIONING_REPEATS:-1}"
      MEASUREMENT_REPEATS="${MEASUREMENT_REPEATS:-1}"
      WARMUP_DECODE_STEPS="${WARMUP_DECODE_STEPS:-2}"
      MEASURED_DECODE_STEPS="${MEASURED_DECODE_STEPS:-8}"
      COPY_TRACE="${COPY_TRACE:-1}"
      ;;
    formal)
      order="${ORDER:-EUUEEU}"
      required="${REQUIRED_ARMS:-EU}"
      CONDITIONING_REPEATS="${CONDITIONING_REPEATS:-1}"
      MEASUREMENT_REPEATS="${MEASUREMENT_REPEATS:-3}"
      WARMUP_DECODE_STEPS="${WARMUP_DECODE_STEPS:-5}"
      MEASURED_DECODE_STEPS="${MEASURED_DECODE_STEPS:-16}"
      COPY_TRACE="${COPY_TRACE:-0}"
      ;;
    *) die "unknown suite mode: ${mode}" ;;
  esac
  validate_order "${order}" "${required}"

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
  [[ "$(wc -l <"${names}")" -eq 8 ]] || die "expected eight exact names"
  write_run_config "${run_root}/run_config.json" "${mode}" "${order}" \
    "${required}" "${model}" "${fixture}"
  local -a provenance_files=(
    "${model}/config.json" \
    "${fixture}" \
    "${names}" \
    "${BASH_SOURCE[0]}" \
    "${RUNNER}" \
    "${VERIFY}" \
    "${BASE_RUNNER}" \
    "${PHASE_RUNNER}" \
    "${PREPARE}" \
    "${VLLM_ROOT}/vllm/model_executor/models/utils.py" \
    "${VLLM_ROOT}/vllm/model_executor/layers/fused_moe/layer.py" \
    "${VLLM_ROOT}/vllm/model_executor/layers/quantization/fp8.py" \
    "${VLLM_ROOT}/vllm/v1/worker/gpu_model_runner.py" \
    "${VLLM_ROOT}/vllm/compilation/cuda_graph.py" \
    "${VLLM_ROOT}/vllm/device_allocator/uvm.py" \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_static_alias.py" \
    "${UVM_ALLOCATOR_SO}" \
    "${CUSTOM_UVM}" \
    "${STOCK_UVM}"
  )
  if [[ -n "${TOP_LEVEL_HARNESS:-}" ]]; then
    provenance_files+=("${TOP_LEVEL_HARNESS}")
  fi
  sha256sum "${provenance_files[@]}" >"${run_root}/input_sha256.txt"
  git -C "${REPO_ROOT}" status --short >"${run_root}/git_status.txt"
  printf 'ordinal\tblock\tsymbol\tarm\tbatch_order\tdirectory\treturncode\n' \
    >"${run_root}/execution_status.tsv"

  local failures=0 index ordinal block symbol arm arm_dir batch_order rc
  local block_len="${#required}"
  for ((index = 0; index < ${#order}; ++index)); do
    ordinal=$((index + 1))
    block=$((index / block_len + 1))
    symbol="${order:index:1}"
    arm="$(arm_name "${symbol}")"
    batch_order=1,4
    (( block % 2 == 0 )) && batch_order="$(reverse_csv "${batch_order}")"
    arm_dir="${run_root}/$(printf '%02d_%s' "${ordinal}" "${arm}")"
    if run_arm "${symbol}" "${arm_dir}" "${names}" "${batch_order}"; then
      rc=0
    else
      rc=$?
      failures=$((failures + 1))
      tail -n 200 "${arm_dir}/run.log" >&2 || true
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${ordinal}" "${block}" "${symbol}" "${arm}" "${batch_order}" \
      "$(basename "${arm_dir}")" "${rc}" >>"${run_root}/execution_status.tsv"
  done

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

case "${1:-help}" in
  static-preflight) static_preflight ;;
  smoke) run_suite smoke ;;
  qualify) run_suite qualification ;;
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
