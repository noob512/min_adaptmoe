#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_PY:-${VLLM_ROOT}/.venv/bin/python}"
BENCHMARK="${REPO_ROOT}/experiments/broad_offload_compare/benchmark_custom_offload.py"
PREPARE="${REPO_ROOT}/experiments/vllm_slot_vs_pinned_copy/prepare_inputs.py"
VERIFY="${SCRIPT_DIR}/verify_compare.py"
CUSTOM_MODULE="${CUSTOM_MODULE:-${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
FENCE_LIB="${VLLM_ROOT}/uvm_test/libuvm_slot_cuda_fence_kernels.so"
STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"
PARAM_ROOT="${PARAM_ROOT:-/sys/module/nvidia_uvm/parameters}"

DEFAULT_MODEL="${REPO_ROOT}/../.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782"
DEFAULT_FIXTURE="${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json"

COLD_EXPERT_IDS="0,7,15,23,31,47,63,79,95,111,119,127"
LAYER_BYTES=56623104
TOTAL_BYTES=2717908992
REGIONS_PER_LAYER=108
TOTAL_REGIONS=5184
DUAL_BANK_BYTES=113246208
DUAL_BANK_SLOTS=216
OUTPUT_TOKENS=8

die() { echo "error: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }

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
  used="$(nvidia-smi -i "${GPU_INDEX:-0}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1}')"
  [[ "${used}" == "0" && -z "${pids}" ]]
}

wait_gpu_clean() {
  local attempt
  for ((attempt = 0; attempt < ${GPU_CLEAN_TIMEOUT_SEC:-240}; ++attempt)); do
    gpu_clean && return 0
    sleep 1
  done
  return 1
}

snapshot_gpu() {
  {
    nvidia-smi -i "${GPU_INDEX:-0}" --query-gpu=index,name,uuid,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>&1 || true
  } > "$1"
}

static_preflight() {
  require_file "${VLLM_PY}"
  require_file "${BENCHMARK}"
  require_file "${PREPARE}"
  require_file "${VERIFY}"
  require_file "${CUSTOM_MODULE}"
  require_file "${FENCE_LIB}"
  bash -n "${BASH_SOURCE[0]}"
  "${VLLM_PY}" -m py_compile "${BENCHMARK}" "${PREPARE}" "${VERIFY}"
  "${VLLM_PY}" "${BENCHMARK}" --help | rg -q -- '--disable-chunked-prefill'
  rg -q 'VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ' \
    "${VLLM_ROOT}/vllm/v1/worker/gpu_model_runner.py" \
    || die "P3-prefill/remote-decode gate is absent"
  rg -q 'transition_uvm_regions_to_remote_read' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_pool_residency.py" \
    || die "audited remote-read transition is absent"
  rg -q 'VLLM_UVM_PREFILL_P3_DECODE_REMOTE_BATCH' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_pool_residency.py" \
    || die "batched audited remote-read transition is absent"
  "${VLLM_PY}" -m pytest -q \
    "${REPO_ROOT}/experiments/vllm_slot_vs_pinned_copy/test_prepare_inputs.py" \
    "${SCRIPT_DIR}/test_prefill_p3_decode_remote_verify.py"
  echo "static preflight: OK"
}

runtime_preflight() {
  static_preflight
  command -v nvidia-smi >/dev/null || die "nvidia-smi unavailable"
  command -v timeout >/dev/null || die "timeout unavailable"
  command -v taskset >/dev/null || die "taskset unavailable"
  [[ -r "${STATS_PATH}" ]] || die "latency stats unavailable"
  local file_src loaded_src budget
  file_src="$(modinfo -F srcversion "${CUSTOM_MODULE}")"
  loaded_src="$(tr -d '[:space:]' < /sys/module/nvidia_uvm/srcversion)"
  [[ -n "${file_src}" && "${loaded_src}" == "${EXPECTED_CUSTOM_SRCVERSION:-${file_src}}" ]] \
    || die "loaded UVM is not the requested custom module"
  require_parameter uvm_moe_slot_pte_remap_enable 1
  require_parameter uvm_moe_slot_plan_overlap_copy_prepare 0
  require_parameter uvm_moe_slot_plan_register_pte_prewarm 1
  require_parameter uvm_moe_slot_plan_cache_copy_runs 1
  require_parameter uvm_moe_slot_plan_cache_metadata_recipe 1
  require_parameter uvm_moe_slot_transition_fast_index 1
  require_parameter uvm_moe_slot_transition_direct_cutover 1
  require_parameter uvm_moe_slot_plan_fast_validation 1
  require_parameter uvm_moe_slot_plan_shadow_pte_prefill 1
  require_parameter uvm_moe_slot_plan_dual_bank_event 1
  require_parameter uvm_moe_slot_plan_cuda_fence 1
  budget="$("${VLLM_PY}" -c 'import sys; print(int(sys.argv[1], 0))' "$(read_parameter uvm_moe_slot_pool_vidmem_budget_bytes)")"
  (( budget >= DUAL_BANK_BYTES )) || die "slot budget ${budget} is below ${DUAL_BANK_BYTES}"
}

arm_name() {
  case "$1" in
    H) echo hybrid ;;
    P) echo p3 ;;
    U) echo remote ;;
    *) die "unknown arm symbol $1" ;;
  esac
}

run_arm() {
  local symbol="$1" ordinal="$2" arm_dir="$3" plan="$4"
  local arm profile mechanism
  arm="$(arm_name "${symbol}")"
  mkdir "${arm_dir}"
  : > "${arm_dir}/slot_trace.jsonl"
  : > "${arm_dir}/cpu_offload_manifest.jsonl"
  wait_gpu_clean || die "GPU is not clean before ${arm}"
  snapshot_gpu "${arm_dir}/gpu.before.csv"
  cp -- "${STATS_PATH}" "${arm_dir}/latency_stats.before.txt"

  local -a common_env=(
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
    VLLM_UVM_EXPERT_ENVELOPE_BYTES=524288
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup
    VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${plan}"
    VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN=1
    VLLM_UVM_SLOT_DIRECT_TRACE_FILE="${arm_dir}/slot_trace.jsonl"
    VLLM_CPU_OFFLOAD_MANIFEST_FILE="${arm_dir}/cpu_offload_manifest.jsonl"
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1
    VLLM_ENABLE_V1_MULTIPROCESSING=0
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    TOKENIZERS_PARALLELISM=false
    OMP_NUM_THREADS="${CPU_THREADS:-14}"
    MKL_NUM_THREADS="${CPU_THREADS:-14}"
    CUDA_VISIBLE_DEVICES="${GPU_INDEX:-0}"
  )
  local -a arm_env
  case "${arm}" in
    hybrid|p3)
      if [[ "${arm}" == "hybrid" ]]; then
        profile=sparse48x12_prefill_p3_decode_remote
        mechanism=vllm_sparse512k_p3_prefill_custom_remote_decode
        arm_env=(
          VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ=1
          VLLM_UVM_PREFILL_P3_DECODE_REMOTE_BATCH=1
          VLLM_UVM_PREFILL_P3_DECODE_REMOTE_RESULT_FILE="${arm_dir}/transition_result.json"
        )
      else
        profile=sparse48x12_pure_p3_prefill_decode
        mechanism=vllm_sparse512k_pure_p3_all_token_forwards
        arm_env=(
          VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ=0
          VLLM_UVM_PREFILL_P3_DECODE_REMOTE_BATCH=0
          VLLM_UVM_PREFILL_P3_DECODE_REMOTE_RESULT_FILE=
        )
      fi
      arm_env+=(
        VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST=
        VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
        VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1
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
        VLLM_CPU_OFFLOAD_BACKEND=uva
        VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE=
        VLLM_CPU_OFFLOAD_STATIC_BANK=0
        VLLM_CPU_OFFLOAD_COPY_TRACE=0
        VLLM_CPU_OFFLOAD_COPY_SYNC=0
      )
      ;;
    remote)
      profile=sparse48x12_pure_custom_remote_read
      mechanism=custom_sparse512k_remote_read_prefill_decode
      arm_env=(
        VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ=0
        VLLM_UVM_PREFILL_P3_DECODE_REMOTE_BATCH=0
        VLLM_UVM_PREFILL_P3_DECODE_REMOTE_RESULT_FILE=
        VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST="${arm_dir}/runtime_remote_placement_manifest.json"
        VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
        VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1
        VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=remote_read
        VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD=0
        VLLM_UVM_SLOT_DIRECT_INIT_BANK=0
        VLLM_UVM_SLOT_DIRECT_REUSE_BANK=0
        VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES=0
        VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION=0
        VLLM_UVM_SLOT_DIRECT_SHARED_PUSH=0
        VLLM_UVM_SLOT_DIRECT_64K_PTE=0
        VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN=0
        VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT=0
        VLLM_UVM_SLOT_DIRECT_CUDA_FENCE=0
        VLLM_UVM_SLOT_DIRECT_CUDA_FENCE_KERNEL_LIB=
        VLLM_UVM_SLOT_DIRECT_CACHE_BYTES=
        VLLM_UVM_SLOT_DIRECT_MAX_SLOTS=
        VLLM_CPU_OFFLOAD_BACKEND=uva
        VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE=
        VLLM_CPU_OFFLOAD_STATIC_BANK=0
        VLLM_CPU_OFFLOAD_COPY_TRACE=0
        VLLM_CPU_OFFLOAD_COPY_SYNC=0
      )
      ;;
  esac
  printf '%s\n' "${common_env[@]}" "${arm_env[@]}" > "${arm_dir}/environment.txt"
  {
    echo "ordinal=${ordinal}"
    echo "symbol=${symbol}"
    echo "arm=${arm}"
    echo "started_at=$(date --iso-8601=seconds)"
  } > "${arm_dir}/execution.env"

  local model="${MODEL:-${DEFAULT_MODEL}}" fixture="${EVAL_FIXTURE:-${DEFAULT_FIXTURE}}"
  local -a args=(
    --model "${model}"
    --evaluation-fixture "${fixture}"
    --output "${arm_dir}/result.json"
    --profile-label "${profile}"
    --mechanism-class "${mechanism}"
    --concurrency 1
    --output-tokens "${OUTPUT_TOKENS}"
    --warmup-requests 0
    --warmup-output-tokens 1
    --kv-cache-memory-bytes "${KV_CACHE_BYTES:-1288490189}"
    --max-model-len "${MAX_MODEL_LEN:-544}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
    --max-num-seqs 1
    --disable-chunked-prefill
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.97}"
    --cpu-offload-gb 0
    --seed 0
    --require-prompt-token-ids
  )
  {
    printf 'cd %q && env ' "${VLLM_ROOT}"
    printf '%q ' "${common_env[@]}" "${arm_env[@]}"
    printf 'timeout %qs taskset -c %q %q %q ' \
      "${TIMEOUT_SEC:-3600}" "${CPUSET:-0-13}" "${VLLM_PY}" "${BENCHMARK}"
    printf '%q ' "${args[@]}"
    printf '\n'
  } > "${arm_dir}/command.txt"

  local rc
  set +e
  (
    cd "${VLLM_ROOT}"
    env "${common_env[@]}" "${arm_env[@]}" \
      /usr/bin/time -v -o "${arm_dir}/resource_time.txt" \
      timeout "${TIMEOUT_SEC:-3600}s" taskset -c "${CPUSET:-0-13}" \
      "${VLLM_PY}" "${BENCHMARK}" "${args[@]}"
  ) > "${arm_dir}/run.log" 2>&1
  rc=$?
  set -e
  echo "${rc}" > "${arm_dir}/returncode.txt"
  cp -- "${STATS_PATH}" "${arm_dir}/latency_stats.after.txt"
  {
    echo "ended_at=$(date --iso-8601=seconds)"
    echo "returncode=${rc}"
  } >> "${arm_dir}/execution.env"
  wait_gpu_clean || die "GPU did not return to 0 MiB after ${arm}"
  snapshot_gpu "${arm_dir}/gpu.after.csv"
  return "${rc}"
}

run_compare() {
  [[ "${ALLOW_GPU_RUN:-0}" == "1" ]] || die "run requires ALLOW_GPU_RUN=1"
  runtime_preflight
  local order="${ORDER:-HPUUPH}"
  [[ "${order}" =~ ^[HPU]+$ ]] || die "ORDER may contain only H, P, and U"
  [[ "${order}" == *H* && "${order}" == *P* && "${order}" == *U* ]] \
    || die "ORDER must contain all three arms"
  local model="${MODEL:-${DEFAULT_MODEL}}" fixture="${EVAL_FIXTURE:-${DEFAULT_FIXTURE}}"
  require_file "${model}/config.json"
  require_file "${fixture}"
  gpu_clean || die "GPU must start at exactly 0 MiB with no compute client"

  local run_root="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_${order}}"
  [[ ! -e "${run_root}" ]] || die "RUN_ROOT already exists: ${run_root}"
  mkdir -p "${run_root}/inputs"
  run_root="$(cd "${run_root}" && pwd)"
  "${VLLM_PY}" "${PREPARE}" \
    --output-dir "${run_root}/inputs" \
    --cold-layers 0-47 \
    --cold-expert-ids "${COLD_EXPERT_IDS}" \
    --cold-expert-rotation-step 1 \
    > "${run_root}/prepare_inputs.stdout.json"
  local plan="${run_root}/inputs/exact_l0_l3_plan.json"
  sha256sum "${CUSTOM_MODULE}" "${model}/config.json" "${fixture}" "${plan}" \
    > "${run_root}/input_sha256.txt"
  git -C "${REPO_ROOT}" status --short > "${run_root}/git_status.txt"
  snapshot_gpu "${run_root}/gpu.initial.csv"
  "${VLLM_PY}" - "${run_root}/run_config.json" "${order}" <<'PY'
import json, sys
path, order = sys.argv[1:]
payload = {
    "schema": "vllm_prefill_p3_decode_remote_run_v1",
    "order": order,
    "layers": 48,
    "cold_experts_per_layer": 12,
    "regions_per_layer": 108,
    "bytes_per_layer": 56623104,
    "total_regions": 5184,
    "total_bytes": 2717908992,
    "concurrency": 1,
    "output_tokens": 8,
    "max_num_seqs": 1,
    "chunked_prefill": False,
    "enforce_eager": True,
    "hybrid_remote_batch_max_ranges": 64,
    "hybrid_remote_batch_ioctl_calls": 81,
    "hybrid_remote_mapped_bytes": 2717908992,
    "mechanism_classes": {
        "hybrid": "vllm_sparse512k_p3_prefill_custom_remote_decode",
        "p3": "vllm_sparse512k_pure_p3_all_token_forwards",
        "remote": "custom_sparse512k_remote_read_prefill_decode",
    },
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  printf 'ordinal\tarm\tdirectory\treturncode\n' > "${run_root}/execution_status.tsv"

  exec 9>"/tmp/vllm_prefill_p3_decode_remote.lock"
  flock -n 9 || die "another hybrid comparison holds the lock"
  local failures=0 index ordinal symbol arm arm_dir rc
  for ((index = 0; index < ${#order}; ++index)); do
    ordinal=$((index + 1))
    symbol="${order:index:1}"
    arm="$(arm_name "${symbol}")"
    arm_dir="${run_root}/$(printf '%02d_%s' "${ordinal}" "${arm}")"
    if run_arm "${symbol}" "${ordinal}" "${arm_dir}" "${plan}"; then
      rc=0
    else
      rc=$?
      failures=$((failures + 1))
      tail -n 160 "${arm_dir}/run.log" >&2 || true
    fi
    printf '%s\t%s\t%s\t%s\n' "${ordinal}" "${arm}" "$(basename "${arm_dir}")" "${rc}" \
      >> "${run_root}/execution_status.tsv"
  done
  snapshot_gpu "${run_root}/gpu.final.csv"
  local verify_rc=0
  if "${VLLM_PY}" "${VERIFY}" "${run_root}" --output "${run_root}/verification.json" \
      > "${run_root}/verify.log" 2> "${run_root}/verify.err"; then
    verify_rc=0
  else
    verify_rc=$?
    cat "${run_root}/verify.err" >&2 || true
    cat "${run_root}/verify.log" >&2 || true
  fi
  echo "RUN_ROOT=${run_root}"
  echo "verification=${run_root}/verification.json"
  (( failures == 0 && verify_rc == 0 )) \
    || die "comparison failed: arms=${failures} verifier=${verify_rc}"
}

case "${1:-help}" in
  static-preflight) static_preflight ;;
  run) run_compare ;;
  verify)
    [[ $# -eq 2 ]] || die "verify requires RUN_ROOT"
    "${VLLM_PY}" "${VERIFY}" "$2"
    ;;
  *) echo "usage: $0 {static-preflight|run|verify RUN_ROOT}" >&2; exit 2 ;;
esac
