#!/usr/bin/env bash
set -euo pipefail

E2E_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E2E_REPO_ROOT="$(cd "${E2E_DIR}/../.." && pwd)"
E2E_EVAL_DIR="${E2E_REPO_ROOT}/experiments/eurosys_moe_offload_evaluation"
E2E_RUNNER="${E2E_EVAL_DIR}/e11_continuous_arrival_runner.py"
E2E_VERIFY="${E2E_DIR}/verify_e2e.py"
E2E_FROZEN_ENV="${E2E_EVAL_DIR}/configs/e2_ours_2g_env.json"
E2E_PLAN="${E2E_EVAL_DIR}/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/calibration_assets/plans_adjusted_11_17_23_29/cold_plan_2g.json"
MODEL="${MODEL:?Set MODEL to the local DeepSeek-V2-Lite-Chat snapshot directory}"
FIXTURE="${FIXTURE:-${E2E_EVAL_DIR}/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/evaluation_short_dialogue.json}"
REFERENCE="${REFERENCE:-${E2E_REPO_ROOT}/experiments/eurosys_additional_nonintrusive_20260909/runs/A_e11_payload_retry2/remote_continuous/result.json}"
TOKEN_THRESHOLD="${TOKEN_THRESHOLD:-64}"
RESUME_TOKEN_THRESHOLD="${RESUME_TOKEN_THRESHOLD:-128}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-167772160}"
TIMEOUT_SEC="${TIMEOUT_SEC:-2400}"
CPUSET="${CPUSET:-0-13}"

export STATIC_ALIAS_DUAL_VIEW_ENABLE=1
export STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL=1
export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
# shellcheck source=/dev/null
source "${E2E_REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY

for path in "${E2E_RUNNER}" "${E2E_VERIFY}" "${E2E_FROZEN_ENV}" "${E2E_PLAN}" \
  "${MODEL}/config.json" "${FIXTURE}" "${REFERENCE}"; do
  require_file "${path}"
done
[[ "${TOKEN_THRESHOLD}" =~ ^[1-9][0-9]*$ ]] \
  || die "TOKEN_THRESHOLD must be a positive integer"
[[ "${RESUME_TOKEN_THRESHOLD}" =~ ^[1-9][0-9]*$ ]] \
  || die "RESUME_TOKEN_THRESHOLD must be a positive integer"
(( RESUME_TOKEN_THRESHOLD > TOKEN_THRESHOLD )) \
  || die "RESUME_TOKEN_THRESHOLD must be greater than TOKEN_THRESHOLD"
"${VLLM_PY}" -m py_compile "${E2E_RUNNER}" "${E2E_VERIFY}"
runtime_preflight

run_root="${RUN_ROOT:-${E2E_DIR}/artifacts/token-threshold-2g-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ ! -e "${run_root}" ]] || die "RUN_ROOT already exists: ${run_root}"
mkdir -p "${run_root}"
run_root="$(cd "${run_root}" && pwd)"
RUN_ROOT_ACTIVE="${run_root}"
module_snapshot "${run_root}/module.before.json" "${STOCK_UVM}"
load_custom_module
module_snapshot "${run_root}/module.custom.json" "${CUSTOM_UVM}"
wait_gpu_clean || die "GPU is not clean before token-threshold E2E"
gpu_state_json "${run_root}/gpu.before.json"
: >"${run_root}/cpu_offload_manifest.jsonl"

mapfile -t frozen_env < <(
  "${VLLM_PY}" - "${E2E_FROZEN_ENV}" <<'PY'
import json, sys
for key, value in json.load(open(sys.argv[1], encoding="utf-8")).items():
    print(f"{key}={value}")
PY
)
env_overrides=(
  "VLLM_UVM_ORIGINAL_VA_PHASE_MODE=token_threshold"
  "VLLM_UVM_ORIGINAL_VA_TOKEN_THRESHOLD=${TOKEN_THRESHOLD}"
  "VLLM_UVM_ORIGINAL_VA_RESUME_TOKEN_THRESHOLD=${RESUME_TOKEN_THRESHOLD}"
  "VLLM_UVM_TOKEN_THRESHOLD_FAST_TRANSITION=1"
  "VLLM_UVM_RUNTIME_POST_MAP_RESIDENCY_AUDIT=0"
  "VLLM_UVM_RUNTIME_HOT_RESIDENCY_AUDIT=0"
  "VLLM_UVM_PHASE_BATCHED_SCHEDULING=0"
  "VLLM_UVM_PHASE_KV_AWARE_COHORTS=0"
  "VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1"
  "VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE=0"
  "VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=0"
  "VLLM_TRITON_MLA_EXPERIMENTAL_FULL_CUDAGRAPH=0"
  "VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW=1"
  "VLLM_CPU_OFFLOAD_PLAN_CLASS_IMMUTABLE_CPU_SHADOW=1"
  "VLLM_CPU_OFFLOAD_DUAL_VIEW_FULL_PARAMETER_SHADOW=1"
  "VLLM_CPU_OFFLOAD_COPY_TRACE=1"
  "VLLM_CPU_OFFLOAD_COPY_SYNC=0"
  "VLLM_CUDAGRAPH_DISPATCH_AUDIT=1"
  "VLLM_CPU_OFFLOAD_MANIFEST_FILE=${run_root}/cpu_offload_manifest.jsonl"
  "VLLM_CPU_OFFLOAD_MANIFEST_RESET=1"
  "VLLM_UVM_LOG_FILE=${run_root}/uvm.log"
)
printf '%s\n' "${frozen_env[@]}" "${env_overrides[@]}" \
  >"${run_root}/environment.txt"
{
  printf 'cd %q && env ' "${VLLM_ROOT}"
  printf '%q ' "${frozen_env[@]}" "${env_overrides[@]}"
  printf 'timeout --kill-after=30s %qs taskset -c %q %q %q ' \
    "${TIMEOUT_SEC}" "${CPUSET}" "${VLLM_PY}" "${E2E_RUNNER}"
  printf '%q ' --model "${MODEL}" --fixture "${FIXTURE}" \
    --output "${run_root}/result.json" \
    --mode adaptmoe_copy_eager_continuous \
    --kv-cache-memory-bytes "${KV_CACHE_BYTES}"
  printf '\n'
} >"${run_root}/command.txt"
sha256sum "${E2E_RUNNER}" "${E2E_VERIFY}" "${E2E_FROZEN_ENV}" "${E2E_PLAN}" \
  "${FIXTURE}" "${MODEL}/config.json" \
  "${VLLM_ROOT}/vllm/model_executor/models/utils.py" \
  "${VLLM_ROOT}/vllm/v1/worker/gpu_model_runner.py" \
  "${VLLM_ROOT}/vllm/v1/worker/gpu_worker.py" "${CUSTOM_UVM}" \
  >"${run_root}/input_sha256.txt"
git -C "${E2E_REPO_ROOT}" status --short >"${run_root}/git_status.txt"

kernel_epoch="$(date +%s)"
set +e
(
  cd "${VLLM_ROOT}"
  env "${frozen_env[@]}" "${env_overrides[@]}" \
    /usr/bin/time -v -o "${run_root}/resource_time.txt" \
    timeout --kill-after=30s "${TIMEOUT_SEC}s" \
    taskset -c "${CPUSET}" \
    "${VLLM_PY}" "${E2E_RUNNER}" \
      --model "${MODEL}" \
      --fixture "${FIXTURE}" \
      --output "${run_root}/result.json" \
      --mode adaptmoe_copy_eager_continuous \
      --kv-cache-memory-bytes "${KV_CACHE_BYTES}"
) >"${run_root}/run.log" 2>&1
runner_rc=$?
sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
  >"${run_root}/kernel.since.log" 2>"${run_root}/kernel.since.err" || true
set -e
printf '%s\n' "${runner_rc}" >"${run_root}/returncode.txt"
wait_gpu_clean || die "GPU/client state did not clean after token-threshold E2E"
gpu_state_json "${run_root}/gpu.after.json"

restore_stock || die "failed to restore stock nvidia_uvm"
module_snapshot "${run_root}/module.after.json" "${STOCK_UVM}"
restart_exporters

set +e
"${VLLM_PY}" "${E2E_VERIFY}" "${run_root}" \
  --plan "${E2E_PLAN}" --reference "${REFERENCE}" \
  --threshold "${TOKEN_THRESHOLD}" \
  --resume-threshold "${RESUME_TOKEN_THRESHOLD}" \
  --output "${run_root}/verification.json" \
  >"${run_root}/verify.log" 2>"${run_root}/verify.err"
verify_rc=$?
set -e
echo "RUN_ROOT=${run_root}"
echo "verification=${run_root}/verification.json"
(( runner_rc == 0 && verify_rc == 0 )) || {
  tail -n 240 "${run_root}/run.log" >&2 || true
  cat "${run_root}/verify.err" >&2 || true
  die "token-threshold E2E failed: runner=${runner_rc}, verifier=${verify_rc}"
}
