#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
KT_PY="${REPO_ROOT}/KTransformers/.venv/bin/python"
MODEL="/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7"
FIXTURE_ROOT="${SCRIPT_DIR}/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16"
EVAL_FIXTURE="${FIXTURE_ROOT}/evaluation_short_dialogue.json"
WARMUP_FIXTURE="${FIXTURE_ROOT}/calibration_short_dialogue.json"
TRACE="${FIXTURE_ROOT}/calibration_assets/routing/routing_trace.jsonl"
PROFILE="${SCRIPT_DIR}/configs/c16_short_mechanism_runtime_hot_audit_off_pilot.json"
NATIVE_ENV="${SCRIPT_DIR}/configs/c1_native_common_env.json"
OURS_ENV="${SCRIPT_DIR}/configs/c1_ours_runtime_hot_audit_off_env.json"
KT_ENV="${SCRIPT_DIR}/configs/e2_ktransformers_env.json"
MOEI_ENV="${SCRIPT_DIR}/configs/e2_moe_infinity_env.json"
VLLM_RUNNER="${SCRIPT_DIR}/direct_batch_c16_vllm_runner.py"
KT_RUNNER="${SCRIPT_DIR}/direct_batch_c16_ktransformers_runner.py"
MOEI_RUNNER="${SCRIPT_DIR}/direct_batch_c16_moe_infinity_runner.py"
MOEI_MODEL_RUNNER="${REPO_ROOT}/MoE-Infinity/moe_infinity/serving/model_runner.py"
C16_VERIFY="${SCRIPT_DIR}/verify_direct_batch_c16_case.py"
PLAN_BUILDER="${REPO_ROOT}/experiments/hybrid_offload_2_4_6_8_compare/prepare_cold_plans.py"
KT_PLAN_BUILDER="${REPO_ROOT}/experiments/hybrid_offload_2_4_6_8_compare/prepare_kt_frequency_plans.py"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
MOEI_OFFLOAD="/home/ubuntu/moe_infinity_offload/deepseek_v2_lite_chat_85864749cd611b4353ce1decdb286193298f64c7"
MOEI_RECEIPT="${SCRIPT_DIR}/artifacts/c1-mechanism-min-offload-pilot-20260827-attempt-01/provenance/moe_infinity_store_build/store_build_receipt.json"
RUN_ROOT="${1:-${SCRIPT_DIR}/artifacts/c16-short-direct-batch-pilot-$(date -u +%Y%m%dT%H%M%SZ)}"
TIMEOUT_SECONDS="${C16_CASE_TIMEOUT_SECONDS:-2400}"

die() { echo "error: $*" >&2; exit 1; }

[[ ! -e "${RUN_ROOT}" ]] || die "refusing to overwrite ${RUN_ROOT}"
for path in "${MODEL}/config.json" "${EVAL_FIXTURE}" "${WARMUP_FIXTURE}" \
  "${TRACE}" "${PROFILE}" "${NATIVE_ENV}" "${OURS_ENV}" "${KT_ENV}" \
  "${MOEI_ENV}" "${VLLM_RUNNER}" "${KT_RUNNER}" "${MOEI_RUNNER}" \
  "${MOEI_MODEL_RUNNER}" "${C16_VERIFY}" "${PLAN_BUILDER}" \
  "${KT_PLAN_BUILDER}" "${BASE_HARNESS}" \
  "${MOEI_RECEIPT}"; do
  [[ -e "${path}" ]] || die "required path is missing: ${path}"
done
[[ -d "${MOEI_OFFLOAD}" ]] || die "MoE-Infinity offload store is missing"
command -v jq >/dev/null || die "jq is unavailable"
command -v flock >/dev/null || die "flock is unavailable"

# Serialize against both the C1 mechanism matrix and all formal E2 GPU work.
exec 7>"/tmp/eurosys_c1_mechanism_runtime_hot_audit_off.lock"
flock -n 7 || die "another C1 matrix holds the GPU lock"
exec 8>"/tmp/eurosys_moe_offload_e2_gpu0.lock"
flock -n 8 || die "another E2 runner holds the GPU lock"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
export ALLOW_GPU_RUN=1
export SLOT_BUDGET_BYTES=1207959552
export MANUAL_MIGRATION_FAULT_MODE=2
# shellcheck source=/dev/null
source "${BASE_HARNESS}"

mkdir -p "${RUN_ROOT}/provenance" "${RUN_ROOT}/cases"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
RUN_ROOT_ACTIVE="${RUN_ROOT}"
EXECUTION_TSV="${RUN_ROOT}/execution.tsv"
printf 'created_at\tarm\tcase_root\treturncode\tverification\n' >"${EXECUTION_TSV}"

runtime_preflight
module_snapshot "${RUN_ROOT}/module.before.json" "${STOCK_UVM}"
cp "${PROFILE}" "${RUN_ROOT}/provenance/profile.json"
git -C "${REPO_ROOT}" status --short >"${RUN_ROOT}/provenance/git_status.txt"
sha256sum "${PROFILE}" "${VLLM_RUNNER}" "${KT_RUNNER}" "${MOEI_RUNNER}" \
  "${MOEI_MODEL_RUNNER}" \
  "${C16_VERIFY}" "${EVAL_FIXTURE}" "${WARMUP_FIXTURE}" "${TRACE}" \
  "${MODEL}/config.json" >"${RUN_ROOT}/provenance/input_sha256.txt"

# Freeze the same frequency-ranked cold1 placement for ours and KTransformers.
PLAN_ROOT="${RUN_ROOT}/provenance/cold_plans"
"${VLLM_PY}" "${PLAN_BUILDER}" --trace "${TRACE}" --output-dir "${PLAN_ROOT}" \
  --serving-start-step 3 --prefill-step-count 1 --tier-widths cold01=1 \
  >"${RUN_ROOT}/provenance/prepare_cold_plan.stdout.json"
KT_INPUT="${RUN_ROOT}/provenance/kt_input"
KT_PLAN_ROOT="${RUN_ROOT}/provenance/kt_plan"
mkdir -p "${KT_INPUT}"
for tier in 2g 4g 6g 8g; do
  cp "${PLAN_ROOT}/cold_plan_cold01.json" "${KT_INPUT}/cold_plan_${tier}.json"
done
"${VLLM_PY}" "${KT_PLAN_BUILDER}" --input-dir "${KT_INPUT}" \
  --output-dir "${KT_PLAN_ROOT}" >"${RUN_ROOT}/provenance/prepare_kt_plan.stdout.txt"

start_sampler() {
  local case_root="$1"
  nvidia-smi \
    --query-gpu=timestamp,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,clocks.current.sm,clocks.current.memory,pstate \
    --format=csv,noheader,nounits --loop-ms=200 \
    >"${case_root}/nvml.csv" 2>"${case_root}/nvml.err" &
  SAMPLER_PID=$!
}

stop_sampler() {
  kill -TERM "${SAMPLER_PID}" 2>/dev/null || true
  set +e
  wait "${SAMPLER_PID}"
  local rc=$?
  set -e
  printf '%s\n' "${rc}" >"$1/nvml_returncode.txt"
}

load_env_file() {
  local env_file="$1" key value
  while IFS=$'\t' read -r key value; do
    [[ "${key}" =~ ^[A-Z_][A-Z0-9_]*$ ]] || die "invalid env key: ${key}"
    export "${key}=${value}"
  done < <(jq -r 'to_entries[] | [.key, .value] | @tsv' "${env_file}")
}

run_case() {
  local arm="$1" env_file="$2" python="$3"
  shift 3
  local case_root="${RUN_ROOT}/cases/${arm}" kernel_epoch rc verify_rc
  local -a command=("$@")
  mkdir -p "${case_root}"
  gpu_state_json "${case_root}/gpu.before.json"
  kernel_epoch="$(date +%s)"
  {
    printf 'cd %q && ' "${REPO_ROOT}"
    printf 'timeout --kill-after=30s %qs taskset -c 0-13 %q ' \
      "${TIMEOUT_SECONDS}" "${python}"
    printf '%q ' "${command[@]}"
    printf '\n'
  } >"${case_root}/command.txt"
  start_sampler "${case_root}"
  set +e
  (
    load_env_file "${env_file}"
    cd "${REPO_ROOT}"
    /usr/bin/time -v -o "${case_root}/resource_time.txt" \
      timeout --kill-after=30s "${TIMEOUT_SECONDS}s" \
      taskset -c 0-13 "${python}" "${command[@]}"
  ) >"${case_root}/run.log" 2>&1
  rc=$?
  set -e
  stop_sampler "${case_root}"
  printf '%s\n' "${rc}" >"${case_root}/returncode.txt"
  sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
    >"${case_root}/kernel.since.log" 2>"${case_root}/kernel.since.err" || true
  if ! wait_gpu_clean; then
    gpu_state_json "${case_root}/gpu.after.json"
    die "GPU did not clean up after ${arm}"
  fi
  gpu_state_json "${case_root}/gpu.after.json"
  set +e
  "${VLLM_PY}" "${C16_VERIFY}" "${case_root}" --arm "${arm}" \
    >"${case_root}/verify.stdout.txt" 2>"${case_root}/verify.stderr.txt"
  verify_rc=$?
  set -e
  printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${arm}" "${case_root}" "${rc}" "${verify_rc}" >>"${EXECUTION_TSV}"
  return "${verify_rc}"
}

total_failures=0
COMMON_VLLM_ARGS=(
  --model "${MODEL}"
  --fixture "${EVAL_FIXTURE}"
  --warmup-fixture "${WARMUP_FIXTURE}"
  --kv-cache-memory-bytes 100663296
  --max-model-len 192
  --max-num-batched-tokens 2048
)

for arm in native_all_gpu native_pinned_uva native_copy_on_forward; do
  case_root="${RUN_ROOT}/cases/${arm}"
  cpu_gb=0
  [[ "${arm}" == native_all_gpu ]] || cpu_gb=0.01
  if ! (
    if [[ "${arm}" == native_pinned_uva ]]; then
      export VLLM_CPU_OFFLOAD_BACKEND=uva
      export VLLM_CPU_OFFLOAD_MANIFEST_FILE="${case_root}/offload_manifest.jsonl"
      export VLLM_CPU_OFFLOAD_MANIFEST_RESET=1
    elif [[ "${arm}" == native_copy_on_forward ]]; then
      export VLLM_CPU_OFFLOAD_BACKEND=copy
      export VLLM_CPU_OFFLOAD_COPY_SYNC=0
      export VLLM_CPU_OFFLOAD_COPY_TRACE=0
      export VLLM_CPU_OFFLOAD_MANIFEST_FILE="${case_root}/offload_manifest.jsonl"
      export VLLM_CPU_OFFLOAD_MANIFEST_RESET=1
    else
      unset VLLM_CPU_OFFLOAD_BACKEND VLLM_CPU_OFFLOAD_MANIFEST_FILE || true
    fi
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    run_case "${arm}" "${NATIVE_ENV}" "${VLLM_PY}" \
      "${VLLM_RUNNER}" --arm "${arm}" "${COMMON_VLLM_ARGS[@]}" \
      --cpu-offload-gb "${cpu_gb}" --output "${case_root}/result.json"
  ); then
    total_failures=$((total_failures + 1))
  fi
done

load_custom_module
module_snapshot "${RUN_ROOT}/module.custom.json" "${CUSTOM_UVM}"
OURS_CASE="${RUN_ROOT}/cases/ours"
mkdir -p "${OURS_CASE}"
OURS_EFFECTIVE_ENV="${OURS_CASE}/environment.json"
jq \
  --arg names "${PLAN_ROOT}/exact_parameter_names.txt" \
  --arg plan "${PLAN_ROOT}/cold_plan_cold01.json" \
  --arg manifest "${OURS_CASE}/offload_manifest.jsonl" \
  --arg uvm_log "${OURS_CASE}/uvm.log" \
  --arg phase "${OURS_CASE}/phase_transition.json" \
  '. + {
    VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE: $names,
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE: $plan,
    VLLM_CPU_OFFLOAD_MANIFEST_FILE: $manifest,
    VLLM_UVM_LOG_FILE: $uvm_log,
    VLLM_UVM_ORIGINAL_VA_PHASE_RESULT_FILE: $phase,
    VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_BYTES: "100663296",
    VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS: "2048",
    VLLM_UVM_FUSED_MOE_POST_WORKSPACE_AUDIT_MIN_TOKENS: "2048",
    VLLM_FUSED_MOE_CHUNK_SIZE: "2048",
    VLLM_UVM_RUNTIME_HOT_RESIDENCY_AUDIT: "0"
  } | del(.VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE)' \
  "${OURS_ENV}" >"${OURS_EFFECTIVE_ENV}"
if ! (
  unset VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE
  unset FORMAL_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE
  run_case ours "${OURS_EFFECTIVE_ENV}" "${VLLM_PY}" \
    "${VLLM_RUNNER}" --arm ours "${COMMON_VLLM_ARGS[@]}" \
    --cpu-offload-gb 30 --output "${OURS_CASE}/result.json"
); then
  total_failures=$((total_failures + 1))
fi
restore_stock || die "failed to restore stock nvidia_uvm"
module_snapshot "${RUN_ROOT}/module.after_ours.json" "${STOCK_UVM}"

KT_CASE="${RUN_ROOT}/cases/ktransformers"
if ! run_case ktransformers "${KT_ENV}" "${KT_PY}" \
  "${KT_RUNNER}" --model "${MODEL}" --fixture "${EVAL_FIXTURE}" \
  --warmup-fixture "${WARMUP_FIXTURE}" \
  --frequency-plan "${KT_PLAN_ROOT}/kt_frequency_2g.pt" \
  --gpu-experts 63 --context-length 194 --max-total-tokens 3104 \
  --max-prefill-tokens 2048 --output "${KT_CASE}/result.json"; then
  total_failures=$((total_failures + 1))
fi

extension="$(find "${REPO_ROOT}/MoE-Infinity/moe_infinity" -maxdepth 1 \
  -type f -name '_store*.so' -print -quit)"
[[ -n "${extension}" ]] || die "MoE-Infinity _store extension is missing"
[[ "$(sha256sum "${extension}" | awk '{print $1}')" == \
   "$(jq -r '.extension_sha256' "${MOEI_RECEIPT}")" ]] \
  || die "MoE-Infinity extension differs from its build receipt"
MOEI_CASE="${RUN_ROOT}/cases/moe_infinity"
if ! run_case moe_infinity "${MOEI_ENV}" "${VLLM_PY}" \
  "${MOEI_RUNNER}" --model "${MODEL}" --offload-path "${MOEI_OFFLOAD}" \
  --fixture "${EVAL_FIXTURE}" --warmup-fixture "${WARMUP_FIXTURE}" \
  --device-memory-ratio 0.80 --kv-cache-ratio 0.0145625 \
  --expected-kv-blocks 192 --max-tokens-per-step 2048 \
  --output "${MOEI_CASE}/result.json"; then
  total_failures=$((total_failures + 1))
fi

printf '%s\n' "${total_failures}" >"${RUN_ROOT}/failure_count.txt"
module_snapshot "${RUN_ROOT}/module.after.json" "${STOCK_UVM}"
restart_exporters
echo "RUN_ROOT=${RUN_ROOT}"
echo "FAILURES=${total_failures}"
(( total_failures == 0 ))
