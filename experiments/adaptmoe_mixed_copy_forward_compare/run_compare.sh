#!/usr/bin/env bash
set -euo pipefail

COMPARE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${COMPARE_DIR}/../.." && pwd)"
E11_DIR="${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation"
E11_RUNNER="${E11_DIR}/e11_continuous_arrival_runner.py"
COMPARE_VERIFY="${COMPARE_DIR}/verify_compare.py"
FROZEN_ENV="${E11_DIR}/configs/e2_ours_2g_env.json"
ADAPT_NAMES="${E11_DIR}/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/calibration_assets/plans/exact_parameter_names.txt"
ADAPT_PLAN="${E11_DIR}/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/calibration_assets/plans_adjusted_11_17_23_29/cold_plan_2g.json"
NATIVE_NAMES="${COMPARE_DIR}/inputs/native_near_equal_parameter_names.txt"
BYTE_COORDINATE="${COMPARE_DIR}/inputs/byte_coordinate.json"
MODEL="${MODEL:?Set MODEL to the local DeepSeek-V2-Lite-Chat snapshot directory}"
FIXTURE="${FIXTURE:-${E11_DIR}/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/evaluation_short_dialogue.json}"
REFERENCE_RESULT="${REFERENCE_RESULT:-${REPO_ROOT}/experiments/eurosys_additional_nonintrusive_20260909/runs/A_e11_payload_retry2/remote_continuous/result.json}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-167772160}"
PAIRED_SAMPLES="${PAIRED_SAMPLES:-2}"
TIMEOUT_SEC="${TIMEOUT_SEC:-2400}"
CPUSET="${CPUSET:-0-13}"
NSYS_PROFILE="${NSYS_PROFILE:-0}"
NSYS_BIN="${NSYS_BIN:-/usr/local/cuda-12.8/bin/nsys}"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY

[[ "${PAIRED_SAMPLES}" =~ ^[1-9][0-9]*$ ]] \
  || die "PAIRED_SAMPLES must be a positive integer"
[[ "${NSYS_PROFILE}" == 0 || "${NSYS_PROFILE}" == 1 ]] \
  || die "NSYS_PROFILE must be 0 or 1"
if [[ "${NSYS_PROFILE}" == 1 ]]; then
  [[ -x "${NSYS_BIN}" ]] || die "Nsight Systems is missing: ${NSYS_BIN}"
fi
for path in "${E11_RUNNER}" "${COMPARE_VERIFY}" "${FROZEN_ENV}" "${ADAPT_NAMES}" \
  "${ADAPT_PLAN}" "${NATIVE_NAMES}" "${BYTE_COORDINATE}" \
  "${MODEL}/config.json" "${FIXTURE}" "${REFERENCE_RESULT}"; do
  require_file "${path}"
done
[[ "$(wc -l <"${ADAPT_NAMES}")" -eq 52 ]] \
  || die "AdaptMoE exact-name input must contain 52 parameters"
[[ "$(wc -l <"${NATIVE_NAMES}")" -eq 9 ]] \
  || die "native exact-name input must contain nine parameters"
"${VLLM_PY}" -m py_compile "${E11_RUNNER}" "${COMPARE_VERIFY}"
bash -n "${BASH_SOURCE[0]}"
runtime_preflight

run_root="${RUN_ROOT:-${COMPARE_DIR}/artifacts/adaptmoe-vs-native-copy-mixed-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ ! -e "${run_root}" ]] || die "RUN_ROOT already exists: ${run_root}"
mkdir -p "${run_root}/provenance" "${run_root}/arms"
run_root="$(cd "${run_root}" && pwd)"
RUN_ROOT_ACTIVE="${run_root}"

cp -- "${ADAPT_NAMES}" "${run_root}/provenance/adaptmoe_parameter_names.txt"
cp -- "${ADAPT_PLAN}" "${run_root}/provenance/adaptmoe_expert_plan.json"
cp -- "${NATIVE_NAMES}" "${run_root}/provenance/native_parameter_names.txt"
cp -- "${BYTE_COORDINATE}" "${run_root}/provenance/byte_coordinate.json"
cp -- "${FROZEN_ENV}" "${run_root}/provenance/frozen_environment.json"

module_snapshot "${run_root}/module.before.json" "${STOCK_UVM}"
load_custom_module
module_snapshot "${run_root}/module.custom.json" "${CUSTOM_UVM}"
wait_gpu_clean || die "GPU is not clean before the paired experiment"

mapfile -t frozen_env < <(
  "${VLLM_PY}" - "${FROZEN_ENV}" <<'PY'
import json, sys
for key, value in json.load(open(sys.argv[1], encoding="utf-8")).items():
    print(f"{key}={value}")
PY
)

"${VLLM_PY}" - "${run_root}/run_config.json" "${PAIRED_SAMPLES}" \
  "${MODEL}" "${FIXTURE}" "${REFERENCE_RESULT}" "${KV_CACHE_BYTES}" <<'PY'
import json, os, sys
path, samples, model, fixture, reference, kv = sys.argv[1:]
payload = {
    "schema": "adaptmoe_mixed_copy_forward_compare_run_v1",
    "paired_samples": int(samples),
    "arm_order": "AB/BA alternating",
    "model": os.path.realpath(model),
    "fixture": os.path.realpath(fixture),
    "reference_result": os.path.realpath(reference),
    "kv_cache_memory_bytes": int(kv),
    "initial_requests": 4,
    "late_requests": 4,
    "late_after_engine_steps": 4,
    "prompt_tokens": 128,
    "output_tokens": 16,
    "seed": 20260831,
    "cudagraph_mode": "NONE",
    "scheduler": "continuous",
}
open(path, "w", encoding="utf-8").write(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY

sha256sum "${E11_RUNNER}" "${COMPARE_VERIFY}" "${FROZEN_ENV}" "${ADAPT_NAMES}" \
  "${ADAPT_PLAN}" "${NATIVE_NAMES}" "${BYTE_COORDINATE}" "${FIXTURE}" \
  "${MODEL}/config.json" "${REFERENCE_RESULT}" \
  "${VLLM_ROOT}/vllm/model_executor/models/utils.py" \
  "${VLLM_ROOT}/vllm/v1/worker/gpu_model_runner.py" "${CUSTOM_UVM}" \
  >"${run_root}/input_sha256.txt"
git -C "${REPO_ROOT}" status --short >"${run_root}/git_status.txt"
printf 'ordinal\tsample\tarm\tdirectory\treturncode\n' \
  >"${run_root}/execution_status.tsv"

run_arm() {
  local sample="$1" arm="$2" ordinal="$3"
  local arm_dir="${run_root}/arms/$(printf '%02d_s%02d_%s' \
    "${ordinal}" "${sample}" "${arm}")"
  local mode exact_names runner_rc kernel_epoch
  local -a arm_env runner_prefix
  mkdir "${arm_dir}"
  : >"${arm_dir}/cpu_offload_manifest.jsonl"
  wait_gpu_clean || die "GPU is not clean before ${arm} sample ${sample}"
  gpu_state_json "${arm_dir}/gpu.before.json"
  cp -- "${STATS_PATH}" "${arm_dir}/latency_stats.before.txt"

  arm_env=(
    "VLLM_UVM_ORIGINAL_VA_PHASE_MODE=off"
    "VLLM_UVM_PHASE_BATCHED_SCHEDULING=0"
    "VLLM_UVM_PHASE_KV_AWARE_COHORTS=0"
    "VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=0"
    "VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE=0"
    "VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=0"
    "VLLM_TRITON_MLA_EXPERIMENTAL_FULL_CUDAGRAPH=0"
    "VLLM_CPU_OFFLOAD_COPY_SYNC=0"
    "VLLM_CPU_OFFLOAD_COPY_TRACE=1"
    "VLLM_CPU_OFFLOAD_MANIFEST_FILE=${arm_dir}/cpu_offload_manifest.jsonl"
    "VLLM_CPU_OFFLOAD_MANIFEST_RESET=1"
    "VLLM_UVM_LOG_FILE=${arm_dir}/uvm.log"
    "PYTORCH_ALLOC_CONF=expandable_segments:True"
  )
  runner_prefix=()
  if [[ "${NSYS_PROFILE}" == 1 ]]; then
    arm_env+=("VLLM_E11_CUDA_PROFILER_CAPTURE=1")
    runner_prefix=(
      "${NSYS_BIN}" profile
      --trace=cuda,nvtx,osrt
      --sample=none
      --cpuctxsw=none
      --capture-range=cudaProfilerApi
      --capture-range-end=stop
      --force-overwrite=true
      "--output=${arm_dir}/timeline"
    )
  fi
  if [[ "${arm}" == adaptmoe ]]; then
    mode=adaptmoe_copy_eager_continuous
    exact_names="${ADAPT_NAMES}"
    arm_env+=(
      "VLLM_CPU_OFFLOAD_BACKEND=copy"
      "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE=${ADAPT_NAMES}"
      "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1"
      "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER=1"
      "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va"
      "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE=${ADAPT_PLAN}"
      "VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1"
    )
  elif [[ "${arm}" == native ]]; then
    mode=native_copy_eager_continuous
    exact_names="${NATIVE_NAMES}"
    arm_env+=(
      "VLLM_CPU_OFFLOAD_BACKEND=copy"
      "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE=${NATIVE_NAMES}"
      "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=0"
      "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER=0"
      "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=vmm"
      "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="
      "VLLM_UVM_EXPERT_WEIGHTS_MANAGED=0"
    )
  else
    die "unknown arm: ${arm}"
  fi

  printf '%s\n' "${frozen_env[@]}" "${arm_env[@]}" \
    >"${arm_dir}/environment.txt"
  {
    printf 'cd %q && env ' "${VLLM_ROOT}"
    printf '%q ' "${frozen_env[@]}" "${arm_env[@]}"
    printf 'timeout --kill-after=30s %qs taskset -c %q ' \
      "${TIMEOUT_SEC}" "${CPUSET}"
    printf '%q ' "${runner_prefix[@]}" "${VLLM_PY}" "${E11_RUNNER}"
    printf '%q ' --model "${MODEL}" --fixture "${FIXTURE}" \
      --output "${arm_dir}/result.json" --mode "${mode}" \
      --kv-cache-memory-bytes "${KV_CACHE_BYTES}"
    printf '\n'
  } >"${arm_dir}/command.txt"

  kernel_epoch="$(date +%s)"
  set +e
  (
    cd "${VLLM_ROOT}"
    env "${frozen_env[@]}" "${arm_env[@]}" \
      /usr/bin/time -v -o "${arm_dir}/resource_time.txt" \
      timeout --kill-after=30s "${TIMEOUT_SEC}s" \
      taskset -c "${CPUSET}" \
      "${runner_prefix[@]}" "${VLLM_PY}" "${E11_RUNNER}" \
        --model "${MODEL}" \
        --fixture "${FIXTURE}" \
        --output "${arm_dir}/result.json" \
        --mode "${mode}" \
        --kv-cache-memory-bytes "${KV_CACHE_BYTES}"
  ) >"${arm_dir}/run.log" 2>&1
  runner_rc=$?
  sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
    >"${arm_dir}/kernel.since.log" 2>"${arm_dir}/kernel.since.err" || true
  set -e
  printf '%s\n' "${runner_rc}" >"${arm_dir}/returncode.txt"
  cp -- "${STATS_PATH}" "${arm_dir}/latency_stats.after.txt"
  wait_gpu_clean || die "GPU/client state did not clean after ${arm} sample ${sample}"
  gpu_state_json "${arm_dir}/gpu.after.json"
  sha256sum "${exact_names}" "${E11_RUNNER}" \
    >"${arm_dir}/input_sha256.txt"
  printf '%s\t%s\t%s\t%s\t%s\n' "${ordinal}" "${sample}" "${arm}" \
    "$(basename "${arm_dir}")" "${runner_rc}" \
    >>"${run_root}/execution_status.tsv"
  return "${runner_rc}"
}

failures=0
ordinal=0
for ((sample = 1; sample <= PAIRED_SAMPLES; ++sample)); do
  if (( sample % 2 == 1 )); then
    order=(adaptmoe native)
  else
    order=(native adaptmoe)
  fi
  for arm in "${order[@]}"; do
    ordinal=$((ordinal + 1))
    if ! run_arm "${sample}" "${arm}" "${ordinal}"; then
      failures=$((failures + 1))
    fi
  done
done

restore_stock || die "failed to restore stock nvidia_uvm"
module_snapshot "${run_root}/module.after.json" "${STOCK_UVM}"
restart_exporters

set +e
"${VLLM_PY}" "${COMPARE_VERIFY}" "${run_root}" \
  --reference "${REFERENCE_RESULT}" \
  --output "${run_root}/verification.json" \
  --report "${run_root}/REPORT.md" \
  >"${run_root}/verify.log" 2>"${run_root}/verify.err"
verify_rc=$?
set -e
echo "RUN_ROOT=${run_root}"
echo "verification=${run_root}/verification.json"
echo "report=${run_root}/REPORT.md"
(( failures == 0 && verify_rc == 0 )) || {
  cat "${run_root}/verify.err" >&2 || true
  die "paired experiment failed: arm_failures=${failures}, verifier=${verify_rc}"
}
