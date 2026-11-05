#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_PY:-${VLLM_ROOT}/.venv/bin/python}"
BENCHMARK="${SCRIPT_DIR}/benchmark_phase_steps.py"
VERIFY="${SCRIPT_DIR}/verify_phase_compare.py"
PREPARE="${REPO_ROOT}/experiments/vllm_slot_vs_pinned_copy/prepare_inputs.py"
CUSTOM_MODULE="${CUSTOM_MODULE:-${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
FENCE_LIB="${VLLM_ROOT}/uvm_test/libuvm_slot_cuda_fence_kernels.so"
STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"
PARAM_ROOT="${PARAM_ROOT:-/sys/module/nvidia_uvm/parameters}"

DEFAULT_MODEL="${REPO_ROOT}/../.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782"
DEFAULT_FIXTURE="${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json"

LAYER_BYTES=603979776
TOTAL_BYTES=2415919104
REGIONS_PER_LAYER=1152
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
    nvidia-smi -i "${GPU_INDEX:-0}" \
      --query-gpu=index,name,uuid,pstate,clocks.current.sm,clocks.current.memory,power.draw,memory.used,memory.free,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>&1 || true
  } > "$1"
}

snapshot_host() {
  {
    uname -a
    lscpu
    rg -n '^(performance|powersave|schedutil)$' \
      /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true
  } > "$1"
}

static_preflight() {
  require_file "${VLLM_PY}"
  require_file "${BENCHMARK}"
  require_file "${VERIFY}"
  require_file "${PREPARE}"
  require_file "${CUSTOM_MODULE}"
  require_file "${FENCE_LIB}"
  bash -n "${BASH_SOURCE[0]}"
  "${VLLM_PY}" -m py_compile "${BENCHMARK}" "${VERIFY}" "${PREPARE}"
  "${VLLM_PY}" "${BENCHMARK}" --help >/dev/null
  rg -q '^_UVM_MOE_SLOT_PLAN_EXECUTE_FENCED_IOCTL = 96$' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm.py" \
    || die "synchronous fenced cmd96 executor is absent"
  if rg -n -i \
    'cmd97|cmd98|async_execute|drain_uvm_registered_layer_plans_async|submitted_layer' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm.py" \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_pool_residency.py" \
    "${VLLM_ROOT}/vllm/v1/worker/gpu_model_runner.py"; then
    die "retired host-async P3 path is present"
  fi
  rg -q 'register_uvm_slot_direct_layer_plans' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_pool_residency.py" \
    || die "registered P3 path is absent"
  rg -q 'remote_read=True' \
    "${VLLM_ROOT}/vllm/model_executor/layers/fused_moe/layer.py" \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_pool_residency.py" \
    || die "custom managed-UVA remote-read path is absent"
  "${VLLM_PY}" -m pytest -q "${SCRIPT_DIR}/test_phase_compare.py"
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
  require_parameter uvm_moe_slot_plan_epoch_validation 1
  require_parameter uvm_moe_slot_plan_shadow_pte_prefill 1
  require_parameter uvm_moe_slot_plan_dual_bank_event 1
  require_parameter uvm_moe_slot_plan_cuda_fence 1
  budget="$("${VLLM_PY}" -c 'import sys; print(int(sys.argv[1], 0))' "$(read_parameter uvm_moe_slot_pool_vidmem_budget_bytes)")"
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

reverse_csv() {
  "${VLLM_PY}" - "$1" <<'PY'
import sys
values = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
print(",".join(reversed(values)))
PY
}

run_arm() {
  local symbol="$1" ordinal="$2" block="$3" arm_dir="$4" plan="$5" names="$6" batch_order="$7"
  local arm profile mechanism cpu_offload_gb
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
    VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ=0
    VLLM_UVM_PREFILL_P3_DECODE_REMOTE_BATCH=0
    VLLM_UVM_PREFILL_P3_DECODE_REMOTE_RESULT_FILE=
    VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0
    VLLM_UVM_EXPERT_ENVELOPE_BYTES=524288
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup
    VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${plan}"
    VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN=1
    VLLM_UVM_SLOT_DIRECT_TRACE_FILE="${arm_dir}/slot_trace.jsonl"
    VLLM_UVM_SLOT_DIRECT_TRACE_MODE=compact
    VLLM_UVM_SLOT_DIRECT_BANK_LEVEL_COMMIT=1
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
      profile=phase_sync_p3_exact_l0_l3
      mechanism=sync_cmd96_p3_registered_512k_dual_bank
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
        VLLM_CPU_OFFLOAD_COPY_TRACE=0
        VLLM_CPU_OFFLOAD_COPY_SYNC=0
      )
      ;;
    pinned)
      profile=phase_pinned_exact_l0_l3
      mechanism=vllm_pinned_copy_exact_l0_l3
      cpu_offload_gb=3.0
      arm_env=(
        VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST=
        VLLM_UVM_EXPERT_WEIGHTS_MANAGED=0
        VLLM_UVM_EXPERT_ENVELOPE_ENABLE=0
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
        VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=restore
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
        VLLM_CPU_OFFLOAD_BACKEND=copy
        VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${names}"
        VLLM_CPU_OFFLOAD_STATIC_BANK=0
        VLLM_CPU_OFFLOAD_COPY_TRACE=0
        VLLM_CPU_OFFLOAD_COPY_SYNC=0
      )
      ;;
    uva)
      profile=phase_custom_managed_uva_exact_l0_l3
      mechanism=custom_managed_uva_remote_read_exact_l0_l3
      cpu_offload_gb=0
      arm_env=(
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
    echo "block=${block}"
    echo "symbol=${symbol}"
    echo "arm=${arm}"
    echo "batch_order=${batch_order}"
    echo "started_at=$(date --iso-8601=seconds)"
  } > "${arm_dir}/execution.env"

  local model="${MODEL:-${DEFAULT_MODEL}}" fixture="${EVAL_FIXTURE:-${DEFAULT_FIXTURE}}"
  local -a args=(
    --model "${model}"
    --fixture "${fixture}"
    --output "${arm_dir}/result.json"
    --profile-label "${profile}"
    --mechanism-class "${mechanism}"
    --batch-sizes "${batch_order}"
    --conditioning-repeats "${CONDITIONING_REPEATS:-1}"
    --measurement-repeats "${MEASUREMENT_REPEATS:-3}"
    --warmup-decode-steps "${WARMUP_DECODE_STEPS:-5}"
    --measured-decode-steps "${MEASURED_DECODE_STEPS:-16}"
    --kv-cache-memory-bytes "${KV_CACHE_BYTES:-268435456}"
    --cpu-offload-gb "${cpu_offload_gb}"
    --max-model-len "${MAX_MODEL_LEN:-512}"
    --max-num-seqs "${MAX_NUM_SEQS:-4}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.97}"
    --seed 0
  )
  {
    printf 'cd %q && env ' "${VLLM_ROOT}"
    printf '%q ' "${common_env[@]}" "${arm_env[@]}"
    printf 'timeout --kill-after=30s %qs taskset -c %q %q %q ' \
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
      timeout --kill-after=30s "${TIMEOUT_SEC:-3600}s" \
      taskset -c "${CPUSET:-0-13}" \
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
  local order="${ORDER:-PCUCUPUPC}"
  [[ "${order}" =~ ^[PCU]+$ && $((${#order} % 3)) -eq 0 ]] \
    || die "ORDER must contain complete PCU permutation blocks"
  local index block_text
  for ((index = 0; index < ${#order}; index += 3)); do
    block_text="${order:index:3}"
    [[ "${block_text}" == *P* && "${block_text}" == *C* && "${block_text}" == *U* ]] \
      || die "ORDER block ${block_text} is not a PCU permutation"
  done
  local batch_sizes="${BATCH_SIZES:-1,4}"
  "${VLLM_PY}" - "${batch_sizes}" <<'PY' >/dev/null
import sys
values = [int(item.strip()) for item in sys.argv[1].split(",") if item.strip()]
assert values and len(values) == len(set(values)) and all(value > 0 for value in values)
assert 1 in values and 4 in values
PY
  local model="${MODEL:-${DEFAULT_MODEL}}" fixture="${EVAL_FIXTURE:-${DEFAULT_FIXTURE}}"
  require_file "${model}/config.json"
  require_file "${fixture}"
  gpu_clean || die "GPU must start at exactly 0 MiB with no compute client"

  local run_root="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_phase_PCU}"
  [[ ! -e "${run_root}" ]] || die "RUN_ROOT already exists: ${run_root}"
  mkdir -p "${run_root}/inputs"
  run_root="$(cd "${run_root}" && pwd)"
  "${VLLM_PY}" "${PREPARE}" --output-dir "${run_root}/inputs" \
    > "${run_root}/prepare_inputs.stdout.json"
  local plan="${run_root}/inputs/exact_l0_l3_plan.json"
  local names="${run_root}/inputs/exact_parameter_names.txt"
  sha256sum "${CUSTOM_MODULE}" "${model}/config.json" "${fixture}" \
    "${plan}" "${names}" "${BENCHMARK}" "${VERIFY}" \
    > "${run_root}/input_sha256.txt"
  git -C "${REPO_ROOT}" status --short > "${run_root}/git_status.txt"
  snapshot_gpu "${run_root}/gpu.initial.csv"
  snapshot_host "${run_root}/host.txt"
  "${VLLM_PY}" - "${run_root}/run_config.json" "${order}" "${batch_sizes}" \
    "${model}" "${fixture}" "${CONDITIONING_REPEATS:-1}" \
    "${MEASUREMENT_REPEATS:-3}" "${WARMUP_DECODE_STEPS:-5}" \
    "${MEASURED_DECODE_STEPS:-16}" "${MAX_MODEL_LEN:-512}" \
    "${MAX_NUM_SEQS:-4}" "${MAX_NUM_BATCHED_TOKENS:-2048}" <<'PY'
import json, sys
(
    path, order, raw_batches, model, fixture, conditioning, measurement,
    warmup_steps, measured_steps, max_model_len, max_num_seqs,
    max_num_batched_tokens,
) = sys.argv[1:]
batches = [int(item.strip()) for item in raw_batches.split(",") if item.strip()]
payload = {
    "schema": "vllm_p3_pinned_uva_phase_compare_run_v1",
    "order": order,
    "process_blocks": [order[index:index + 3] for index in range(0, len(order), 3)],
    "batch_sizes": batches,
    "model": model,
    "fixture": fixture,
    "conditioning_repeats": int(conditioning),
    "measurement_repeats": int(measurement),
    "warmup_decode_steps": int(warmup_steps),
    "measured_decode_steps": int(measured_steps),
    "max_model_len": int(max_model_len),
    "max_num_seqs": int(max_num_seqs),
    "max_num_batched_tokens": int(max_num_batched_tokens),
    "layers": [0, 1, 2, 3],
    "experts_per_layer": 128,
    "regions_per_layer": 1152,
    "bytes_per_layer": 603979776,
    "total_bytes": 2415919104,
    "p3_executor": "synchronous_fenced_cmd96",
    "bank_level_commit": True,
    "trace_mode": "compact",
    "epoch_validation": True,
    "enforce_eager": True,
    "chunked_prefill": False,
    "mechanism_classes": {
        "p3": "sync_cmd96_p3_registered_512k_dual_bank",
        "pinned": "vllm_pinned_copy_exact_l0_l3",
        "uva": "custom_managed_uva_remote_read_exact_l0_l3",
    },
}
open(path, "w", encoding="utf-8").write(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY
  printf 'ordinal\tblock\tsymbol\tarm\tbatch_order\tdirectory\treturncode\n' \
    > "${run_root}/execution_status.tsv"

  exec 9>"/tmp/vllm_p3_pinned_uva_phase_compare.lock"
  flock -n 9 || die "another phase comparison holds the lock"
  local failures=0 ordinal symbol arm arm_dir rc block batch_order
  for ((index = 0; index < ${#order}; ++index)); do
    ordinal=$((index + 1))
    block=$((index / 3 + 1))
    symbol="${order:index:1}"
    arm="$(arm_name "${symbol}")"
    batch_order="${batch_sizes}"
    if (( block % 2 == 0 )); then
      batch_order="$(reverse_csv "${batch_sizes}")"
    fi
    arm_dir="${run_root}/$(printf '%02d_%s' "${ordinal}" "${arm}")"
    if run_arm "${symbol}" "${ordinal}" "${block}" "${arm_dir}" \
        "${plan}" "${names}" "${batch_order}"; then
      rc=0
    else
      rc=$?
      failures=$((failures + 1))
      tail -n 160 "${arm_dir}/run.log" >&2 || true
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${ordinal}" "${block}" "${symbol}" "${arm}" "${batch_order}" \
      "$(basename "${arm_dir}")" "${rc}" >> "${run_root}/execution_status.tsv"
  done
  snapshot_gpu "${run_root}/gpu.final.csv"
  local verify_rc=0
  if "${VLLM_PY}" "${VERIFY}" "${run_root}" \
      --output "${run_root}/verification.json" \
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
    || die "phase comparison failed: arms=${failures} verifier=${verify_rc}"
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
