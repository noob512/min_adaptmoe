#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_PY:-${VLLM_ROOT}/.venv/bin/python}"
BENCHMARK="${BENCHMARK:-${REPO_ROOT}/experiments/broad_offload_compare/benchmark_custom_offload.py}"
PREPARE="${REPO_ROOT}/experiments/vllm_slot_vs_pinned_copy/prepare_inputs.py"
VERIFY="${VERIFY:-${SCRIPT_DIR}/verify_compare.py}"
CUSTOM_MODULE="${CUSTOM_MODULE:-${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
FENCE_LIB="${VLLM_ROOT}/uvm_test/libuvm_slot_cuda_fence_kernels.so"
STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"
PARAM_ROOT="${PARAM_ROOT:-/sys/module/nvidia_uvm/parameters}"

DEFAULT_MODEL="${REPO_ROOT}/../.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782"
DEFAULT_FIXTURE="${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json"
DEFAULT_WARMUP_FIXTURE="${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/calibration_fixture_token_ids.json"

LAYER_BYTES=603979776
TOTAL_BYTES=2415919104
DUAL_BANK_BYTES=1207959552
DUAL_BANK_SLOTS=2304

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
  "${VLLM_PY}" -m pytest -q \
    "${REPO_ROOT}/experiments/vllm_slot_vs_pinned_copy/test_prepare_inputs.py"
  rg -q 'VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE' \
    "${VLLM_ROOT}/vllm/model_executor/models/utils.py" \
    || die "exact-name pinned path is absent"
  rg -q 'register_uvm_slot_direct_layer_plans' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_pool_residency.py" \
    || die "registered P3 path is absent"
  rg -q '^_UVM_MOE_SLOT_PLAN_EXECUTE_FENCED_IOCTL = 96$' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm.py" \
    || die "P3 must use the synchronous fenced cmd96 executor"
  if rg -n -i \
    'cmd97|cmd98|async_execute|drain_uvm_registered_layer_plans_async|submitted_layer' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm.py" \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_pool_residency.py" \
    "${VLLM_ROOT}/vllm/v1/worker/gpu_model_runner.py"; then
    die "P3 host-async execution path is still present"
  fi
  rg -q 'remote_read=True' \
    "${VLLM_ROOT}/vllm/model_executor/layers/fused_moe/layer.py" \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_pool_residency.py" \
    || die "managed-UVA remote-read path is absent"
  echo "static preflight: OK"
}

runtime_preflight() {
  static_preflight
  command -v nvidia-smi >/dev/null || die "nvidia-smi unavailable"
  command -v taskset >/dev/null || die "taskset unavailable"
  command -v timeout >/dev/null || die "timeout unavailable"
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
  require_parameter uvm_moe_slot_plan_epoch_validation "${EPOCH_VALIDATION:-0}"
  require_parameter uvm_moe_slot_plan_shadow_pte_prefill 1
  require_parameter uvm_moe_slot_plan_dual_bank_event 1
  require_parameter uvm_moe_slot_plan_cuda_fence 1
  budget="$(${VLLM_PY} -c 'import sys; print(int(sys.argv[1], 0))' "$(read_parameter uvm_moe_slot_pool_vidmem_budget_bytes)")"
  (( budget >= DUAL_BANK_BYTES )) || die "slot budget ${budget} is below ${DUAL_BANK_BYTES}"
}

arm_name() {
  case "$1" in
    P) echo p3 ;;
    C) echo pinned ;;
    U) echo uva ;;
    *) die "unknown arm symbol $1" ;;
  esac
}

run_arm() {
  local symbol="$1" ordinal="$2" arm_dir="$3" plan="$4" names="$5"
  local arm profile mechanism cpu_offload_gb
  arm="$(arm_name "${symbol}")"
  mkdir "${arm_dir}"
  : > "${arm_dir}/slot_trace.jsonl"
  : > "${arm_dir}/cpu_offload_manifest.jsonl"
  : > "${arm_dir}/routing_trace.jsonl"
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
    VLLM_UVM_MOE_ROUTING_TRACE_ENABLE="${ROUTING_TRACE_ENABLE:-0}"
    VLLM_UVM_MOE_ROUTING_TRACE_FILE="${arm_dir}/routing_trace.jsonl"
    VLLM_UVM_EXPERT_ENVELOPE_BYTES=524288
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup
    VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${plan}"
    VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN=1
    VLLM_UVM_SLOT_DIRECT_TRACE_FILE="${arm_dir}/slot_trace.jsonl"
    VLLM_UVM_SLOT_DIRECT_TRACE_MODE="${TRACE_MODE:-full}"
    VLLM_UVM_SLOT_DIRECT_BANK_LEVEL_COMMIT="${BANK_LEVEL_COMMIT:-0}"
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
    p3)
      profile=prefill_p3_registered_full_l0_l3
      mechanism=vllm_prefill_p3_registered_512k_dual_bank_fence
      cpu_offload_gb=0
      arm_env=(
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
        VLLM_CPU_OFFLOAD_COPY_TRACE="${COPY_TRACE_ENABLE:-0}"
        VLLM_CPU_OFFLOAD_COPY_SYNC=0
      )
      ;;
    pinned)
      profile=prefill_pinned_copy_exact_full_l0_l3
      mechanism=vllm_prefill_pinned_copy_exact_l0_l3
      cpu_offload_gb=3.0
      arm_env=(
        VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST=
        VLLM_UVM_EXPERT_WEIGHTS_MANAGED=0
        VLLM_UVM_EXPERT_ENVELOPE_ENABLE=0
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
        VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=restore
        VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD=0
        VLLM_CPU_OFFLOAD_BACKEND=copy
        VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${names}"
        VLLM_CPU_OFFLOAD_STATIC_BANK=0
        VLLM_CPU_OFFLOAD_COPY_TRACE="${COPY_TRACE_ENABLE:-0}"
        VLLM_CPU_OFFLOAD_COPY_SYNC=0
      )
      ;;
    uva)
      profile=prefill_custom_uva_remote_read_full_l0_l3
      mechanism=custom_prefill_managed_uva_remote_read_exact_l0_l3
      cpu_offload_gb=0
      arm_env=(
        VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST="${arm_dir}/runtime_remote_placement_manifest.json"
        VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
        VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1
        VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=remote_read
        VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD=0
        VLLM_CPU_OFFLOAD_BACKEND=uva
        VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE=
        VLLM_CPU_OFFLOAD_STATIC_BANK=0
        VLLM_CPU_OFFLOAD_COPY_TRACE="${COPY_TRACE_ENABLE:-0}"
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
  local concurrency="${CONCURRENCY:-1}"
  local warmup_requests="${WARMUP_REQUESTS:-0}"
  local warmup_fixture="${WARMUP_FIXTURE:-${DEFAULT_WARMUP_FIXTURE}}"
  local disable_chunked_prefill="${DISABLE_CHUNKED_PREFILL:-0}"
  local -a args=(
    --model "${model}"
    --evaluation-fixture "${fixture}"
    --output "${arm_dir}/result.json"
    --profile-label "${profile}"
    --mechanism-class "${mechanism}"
    --concurrency "${concurrency}"
    --output-tokens 1
    --warmup-requests "${warmup_requests}"
    --warmup-output-tokens 1
    --kv-cache-memory-bytes "${KV_CACHE_BYTES:-1288490189}"
    --max-model-len "${MAX_MODEL_LEN:-544}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
    --max-num-seqs "${MAX_NUM_SEQS:-32}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.97}"
    --cpu-offload-gb "${cpu_offload_gb}"
    --seed 0
    --require-prompt-token-ids
  )
  if (( warmup_requests > 0 )); then
    args+=(--warmup-fixture "${warmup_fixture}")
  fi
  if [[ "${disable_chunked_prefill}" == "1" ]]; then
    args+=(--disable-chunked-prefill)
  elif [[ "${disable_chunked_prefill}" != "0" ]]; then
    die "DISABLE_CHUNKED_PREFILL must be 0 or 1"
  fi
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
  case "${BANK_LEVEL_COMMIT:-0}" in 0|1) ;; *) die "BANK_LEVEL_COMMIT must be 0 or 1" ;; esac
  case "${TRACE_MODE:-full}" in full|compact|errors|off) ;; *) die "TRACE_MODE must be full, compact, errors, or off" ;; esac
  case "${EPOCH_VALIDATION:-0}" in 0|1) ;; *) die "EPOCH_VALIDATION must be 0 or 1" ;; esac
  local order="${ORDER:-PCUUCP}"
  [[ "${order}" =~ ^[PCU]+$ ]] || die "ORDER may contain only P, C, U"
  [[ "${order}" == *P* && "${order}" == *C* && "${order}" == *U* ]] \
    || die "ORDER must contain all three arms"
  local model="${MODEL:-${DEFAULT_MODEL}}" fixture="${EVAL_FIXTURE:-${DEFAULT_FIXTURE}}"
  local concurrency="${CONCURRENCY:-1}"
  local warmup_requests="${WARMUP_REQUESTS:-0}"
  local warmup_fixture="${WARMUP_FIXTURE:-${DEFAULT_WARMUP_FIXTURE}}"
  local disable_chunked_prefill="${DISABLE_CHUNKED_PREFILL:-0}"
  require_file "${model}/config.json"
  require_file "${fixture}"
  if (( warmup_requests > 0 )); then
    require_file "${warmup_fixture}"
  fi
  gpu_clean || die "GPU must start at exactly 0 MiB with no compute client"

  local run_root="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_${order}}"
  [[ ! -e "${run_root}" ]] || die "RUN_ROOT already exists: ${run_root}"
  mkdir -p "${run_root}/inputs"
  run_root="$(cd "${run_root}" && pwd)"
  "${VLLM_PY}" "${PREPARE}" --output-dir "${run_root}/inputs" \
    > "${run_root}/prepare_inputs.stdout.json"
  local plan="${run_root}/inputs/exact_l0_l3_plan.json"
  local names="${run_root}/inputs/exact_parameter_names.txt"
  local -a hashed_inputs=(
    "${CUSTOM_MODULE}" "${model}/config.json" "${fixture}" "${plan}" "${names}"
  )
  if (( warmup_requests > 0 )); then
    hashed_inputs+=("${warmup_fixture}")
  fi
  sha256sum "${hashed_inputs[@]}" > "${run_root}/input_sha256.txt"
  git -C "${REPO_ROOT}" status --short > "${run_root}/git_status.txt"
  snapshot_gpu "${run_root}/gpu.initial.csv"
  "${VLLM_PY}" - "${run_root}/run_config.json" "${order}" \
    "${concurrency}" "${warmup_requests}" "${disable_chunked_prefill}" \
    "${fixture}" "${warmup_fixture}" "${BANK_LEVEL_COMMIT:-0}" \
    "${TRACE_MODE:-full}" "${EPOCH_VALIDATION:-0}" <<'PY'
import json, sys
(
    path,
    order,
    concurrency,
    warmup_requests,
    disable_chunked_prefill,
    fixture,
    warmup_fixture,
    bank_level_commit,
    trace_mode,
    epoch_validation,
) = sys.argv[1:]
payload = {
    "schema": "vllm_prefill_p3_pinned_uva_run_v1",
    "order": order,
    "layers": [0, 1, 2, 3],
    "experts_per_layer": 128,
    "bytes_per_layer": 603979776,
    "total_bytes": 2415919104,
    "concurrency": int(concurrency),
    "output_tokens": 1,
    "warmup_requests": int(warmup_requests),
    "disable_chunked_prefill": bool(int(disable_chunked_prefill)),
    "evaluation_fixture": fixture,
    "warmup_fixture": warmup_fixture if int(warmup_requests) else None,
    "bank_level_commit": bool(int(bank_level_commit)),
    "trace_mode": trace_mode,
    "epoch_validation": bool(int(epoch_validation)),
    "mechanism_classes": {
        "p3": "vllm_prefill_p3_registered_512k_dual_bank_fence",
        "pinned": "vllm_prefill_pinned_copy_exact_l0_l3",
        "uva": "custom_prefill_managed_uva_remote_read_exact_l0_l3",
    },
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
  printf 'ordinal\tarm\tdirectory\treturncode\n' > "${run_root}/execution_status.tsv"

  exec 9>"/tmp/vllm_prefill_p3_pinned_uva_compare.lock"
  flock -n 9 || die "another three-arm comparison holds the lock"
  local failures=0 index ordinal symbol arm arm_dir rc
  for ((index = 0; index < ${#order}; ++index)); do
    ordinal=$((index + 1))
    symbol="${order:index:1}"
    arm="$(arm_name "${symbol}")"
    arm_dir="${run_root}/$(printf '%02d_%s' "${ordinal}" "${arm}")"
    if run_arm "${symbol}" "${ordinal}" "${arm_dir}" "${plan}" "${names}"; then
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
