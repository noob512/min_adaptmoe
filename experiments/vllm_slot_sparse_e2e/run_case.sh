#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_PY:-${VLLM_ROOT}/.venv/bin/python}"
PREPARE="${REPO_ROOT}/experiments/vllm_slot_vs_pinned_copy/prepare_inputs.py"
VERIFY="${SCRIPT_DIR}/verify_case.py"
BENCHMARK="${REPO_ROOT}/experiments/vllm_slot_direct_integration/benchmark_sequential_requests.py"
COMMON_BENCHMARK="${REPO_ROOT}/experiments/broad_offload_compare/benchmark_custom_offload.py"
CUSTOM_MODULE="${CUSTOM_MODULE:-${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
MODEL="${MODEL:-${REPO_ROOT}/../.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
FIXTURE="${EVAL_FIXTURE:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json}"
FENCE_LIB="${VLLM_ROOT}/uvm_test/libuvm_slot_cuda_fence_kernels.so"
STATS_PATH="/proc/driver/nvidia-uvm/latency_stats"
PARAM_ROOT="/sys/module/nvidia_uvm/parameters"

COLD_EXPERT_IDS="0,7,15,23,31,47,63,79,95,111,119,127"
LAYER_BYTES=56623104
REGIONS_PER_LAYER=108
DUAL_BANK_BYTES=113246208
DUAL_BANK_SLOTS=216
SEQUENTIAL_REQUESTS=3
SHADOW_PTE_PREFILL="${SHADOW_PTE_PREFILL:-1}"

die() { echo "error: $*" >&2; exit 1; }

validate_switches() {
  case "${SHADOW_PTE_PREFILL}" in
    0|1) ;;
    *) die "SHADOW_PTE_PREFILL must be 0 or 1, got ${SHADOW_PTE_PREFILL}" ;;
  esac
}

read_parameter() {
  local path="${PARAM_ROOT}/$1"
  [[ -r "${path}" ]] || die "module parameter unavailable: ${path}"
  tr -d '[:space:]' < "${path}"
}

require_parameter() {
  local actual
  actual="$(read_parameter "$1")"
  [[ "${actual}" == "$2" ]] || die "$1=${actual}, expected $2"
}

gpu_clean() {
  local used pids
  used="$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1}')"
  [[ "${used}" == "0" && -z "${pids}" ]]
}

wait_gpu_clean() {
  local attempt
  for ((attempt = 0; attempt < 180; ++attempt)); do
    gpu_clean && return 0
    sleep 1
  done
  return 1
}

module_contract() {
  validate_switches
  require_parameter uvm_moe_slot_pte_remap_enable 1
  require_parameter uvm_moe_slot_plan_overlap_copy_prepare 0
  require_parameter uvm_moe_slot_plan_register_pte_prewarm 1
  require_parameter uvm_moe_slot_plan_cache_copy_runs 1
  require_parameter uvm_moe_slot_plan_cache_metadata_recipe 1
  require_parameter uvm_moe_slot_transition_fast_index 1
  require_parameter uvm_moe_slot_transition_direct_cutover 1
  require_parameter uvm_moe_slot_plan_fast_validation 1
  require_parameter uvm_moe_slot_plan_shadow_pte_prefill "${SHADOW_PTE_PREFILL}"
  require_parameter uvm_moe_slot_plan_dual_bank_event 1
  require_parameter uvm_moe_slot_plan_cuda_fence 1
  local budget
  budget="$(${VLLM_PY} -c 'import sys; print(int(sys.argv[1], 0))' "$(read_parameter uvm_moe_slot_pool_vidmem_budget_bytes)")"
  (( budget >= DUAL_BANK_BYTES )) || die "slot budget ${budget} is below ${DUAL_BANK_BYTES}"
}

snapshot_gpu() {
  {
    nvidia-smi -i 0 --query-gpu=index,name,uuid,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>&1 || true
  } > "$1"
}

write_module_state() {
  {
    echo "loaded_srcversion=$(tr -d '[:space:]' < /sys/module/nvidia_uvm/srcversion)"
    echo "custom_srcversion=$(modinfo -F srcversion "${CUSTOM_MODULE}")"
    for name in \
      uvm_moe_slot_pte_remap_enable \
      uvm_moe_slot_pool_vidmem_budget_bytes \
      uvm_moe_slot_plan_overlap_copy_prepare \
      uvm_moe_slot_plan_register_pte_prewarm \
      uvm_moe_slot_plan_cache_copy_runs \
      uvm_moe_slot_plan_cache_metadata_recipe \
      uvm_moe_slot_transition_fast_index \
      uvm_moe_slot_transition_direct_cutover \
      uvm_moe_slot_plan_fast_validation \
      uvm_moe_slot_plan_shadow_pte_prefill \
      uvm_moe_slot_plan_dual_bank_event \
      uvm_moe_slot_plan_cuda_fence; do
      echo "${name}=$(read_parameter "${name}")"
    done
  } > "$1"
}

static_preflight() {
  validate_switches
  for path in "${VLLM_PY}" "${PREPARE}" "${VERIFY}" "${BENCHMARK}" \
              "${COMMON_BENCHMARK}" "${CUSTOM_MODULE}" "${MODEL}/config.json" \
              "${FIXTURE}" "${FENCE_LIB}"; do
    [[ -e "${path}" ]] || die "missing input: ${path}"
  done
  bash -n "${BASH_SOURCE[0]}"
  "${VLLM_PY}" -m py_compile "${PREPARE}" "${VERIFY}"
  "${VLLM_PY}" -m pytest -q \
    "${REPO_ROOT}/experiments/vllm_slot_vs_pinned_copy/test_prepare_inputs.py"
  echo "static preflight: OK"
}

run_case() {
  [[ "${ALLOW_GPU_RUN:-0}" == "1" ]] || die "run requires ALLOW_GPU_RUN=1"
  local run_root="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_sparse48x12_fence_v1}"
  [[ ! -e "${run_root}" ]] || die "RUN_ROOT already exists: ${run_root}"
  static_preflight
  gpu_clean || die "GPU 0 must be exactly 0 MiB with no compute client"
  module_contract
  local module_src loaded_src
  module_src="$(modinfo -F srcversion "${CUSTOM_MODULE}")"
  loaded_src="$(tr -d '[:space:]' < /sys/module/nvidia_uvm/srcversion)"
  [[ -n "${module_src}" && "${loaded_src}" == "${module_src}" ]] || die "loaded UVM is not the custom module"
  [[ -r "${STATS_PATH}" ]] || die "latency stats unavailable"

  mkdir -p "$(dirname "${run_root}")"
  mkdir "${run_root}"
  run_root="$(cd "${run_root}" && pwd)"
  mkdir "${run_root}/inputs"
  snapshot_gpu "${run_root}/gpu.before.csv"
  "${VLLM_PY}" "${PREPARE}" \
    --output-dir "${run_root}/inputs" \
    --cold-layers 0-47 \
    --cold-expert-ids "${COLD_EXPERT_IDS}" \
    --cold-expert-rotation-step 1 > "${run_root}/prepare_inputs.log"

  local plan="${run_root}/inputs/exact_l0_l3_plan.json"
  "${VLLM_PY}" - "${plan}" <<'PY' > "${run_root}/plan_validation.txt"
import json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
cold = {(x["layer_id"], x["expert_id"]) for x in p["offload_plan"]}
hot = {(x["layer_id"], x["expert_id"]) for x in p["hot_plan"]}
all_keys = {(layer, expert) for layer in range(48) for expert in range(128)}
assert len(cold) == 48 * 12
assert not cold & hot and cold | hot == all_keys
assert all(sum(layer == item[0] for item in cold) == 12 for layer in range(48))
m = p["metadata"]
assert m["cold_regions"] == 48 * 108
assert m["actual_offload_main_bytes"] == 48 * 56_623_104
assert m["shared_hbm_bank_bytes"] == 56_623_104
assert m["shared_hbm_bank_slots"] == 108
print("sparse plan validation: 48 layers x 12 experts x 9 regions; 216 dual-bank slots")
PY

  cp -- "${STATS_PATH}" "${run_root}/latency_stats.before.txt"
  : > "${run_root}/slot_trace.jsonl"
  write_module_state "${run_root}/module_state.before.env"
  sha256sum "${CUSTOM_MODULE}" "${plan}" "${MODEL}/config.json" "${FIXTURE}" > "${run_root}/input_sha256.txt"
  git -C "${REPO_ROOT}" status --short > "${run_root}/git_status.txt"

  local -a env_args=(
    VLLM_USE_UVM=0
    VLLM_UVM_COLD_SLOT_POOL_DRIVER_MODE=slot_va_block_remap
    VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP=1
    VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0
    VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0
    VLLM_UVM_KV_CACHE_MANAGED=0
    VLLM_UVM_KV_RUNTIME_ENABLE=0
    VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0
    VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
    VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0
    VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0
    VLLM_UVM_POST_WARMUP_REBALANCE_FAIL_FAST=1
    VLLM_UVM_WEIGHT_EXECUTOR=cpp_batch
    VLLM_UVM_PREFILL_ORACLE_PREFETCH_MODE=off
    VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0
    VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST=
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
    VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1
    VLLM_UVM_EXPERT_ENVELOPE_BYTES=524288
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup
    VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${plan}"
    VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN=1
    VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=slot_restore
    VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD=1
    VLLM_UVM_SLOT_DIRECT_INIT_BANK=1
    VLLM_UVM_SLOT_DIRECT_REUSE_BANK=1
    VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES=1
    VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION=1
    VLLM_UVM_SLOT_DIRECT_SHARED_PUSH=1
    VLLM_UVM_SLOT_DIRECT_64K_PTE=1
    VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN=1
    VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT=1
    VLLM_UVM_SLOT_DIRECT_CUDA_FENCE=1
    VLLM_UVM_SLOT_DIRECT_CUDA_FENCE_KERNEL_LIB="${FENCE_LIB}"
    VLLM_UVM_SLOT_DIRECT_CACHE_BYTES="${DUAL_BANK_BYTES}"
    VLLM_UVM_SLOT_DIRECT_MAX_SLOTS="${DUAL_BANK_SLOTS}"
    VLLM_UVM_SLOT_DIRECT_TRACE_FILE="${run_root}/slot_trace.jsonl"
    VLLM_ENABLE_V1_MULTIPROCESSING=0
    VLLM_CPU_OFFLOAD_BACKEND=uva
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    TOKENIZERS_PARALLELISM=false
    OMP_NUM_THREADS=14
    MKL_NUM_THREADS=14
    CUDA_VISIBLE_DEVICES=0
  )
  printf '%s\n' "${env_args[@]}" > "${run_root}/environment.txt"

  local -a benchmark_args=(
    --common-benchmark "${COMMON_BENCHMARK}"
    --model "${MODEL}"
    --evaluation-fixture "${FIXTURE}"
    --output "${run_root}/result.json"
    --profile-label sparse48x12_registered_dual_bank_fence
    --mechanism-class vllm_sparse512k_registered_plan_dual_bank_fence
    --sequential-requests "${SEQUENTIAL_REQUESTS}"
    --fixture-request-index 0
    --output-tokens 4
    --kv-cache-memory-bytes 1288490189
    --max-model-len 544
    --max-num-batched-tokens 2048
    --max-num-seqs 32
    --gpu-memory-utilization 0.97
    --seed 0
  )
  {
    printf 'cd %q && env ' "${VLLM_ROOT}"
    printf '%q ' "${env_args[@]}"
    printf 'timeout 3600s taskset -c 0-13 %q %q ' "${VLLM_PY}" "${BENCHMARK}"
    printf '%q ' "${benchmark_args[@]}"
    printf '\n'
  } > "${run_root}/command.txt"

  local benchmark_rc
  set +e
  (
    cd "${VLLM_ROOT}"
    env "${env_args[@]}" timeout 3600s taskset -c 0-13 \
      "${VLLM_PY}" "${BENCHMARK}" "${benchmark_args[@]}"
  ) > "${run_root}/run.log" 2>&1
  benchmark_rc=$?
  set -e
  echo "${benchmark_rc}" > "${run_root}/returncode.txt"
  cp -- "${STATS_PATH}" "${run_root}/latency_stats.after.txt"
  write_module_state "${run_root}/module_state.after.env"
  module_contract
  wait_gpu_clean || die "GPU did not return to 0 MiB after benchmark"
  snapshot_gpu "${run_root}/gpu.after.csv"
  (( benchmark_rc == 0 )) || { tail -n 160 "${run_root}/run.log" >&2; die "benchmark failed rc=${benchmark_rc}"; }

  "${VLLM_PY}" "${VERIFY}" "${run_root}" \
    --shadow-pte-prefill "${SHADOW_PTE_PREFILL}" \
    --output "${run_root}/verification.json" \
    > "${run_root}/verify.log"
  echo "RUN_ROOT=${run_root}"
  echo "verification=${run_root}/verification.json"
}

case "${1:-help}" in
  static-preflight) static_preflight ;;
  run) run_case ;;
  *) echo "usage: $0 {static-preflight|run}" >&2; exit 2 ;;
esac
