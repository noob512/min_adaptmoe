#!/usr/bin/env bash
set -euo pipefail

E11_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E11_ROOT="$(cd "${E11_DIR}/../.." && pwd)"
export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
# shellcheck source=/dev/null
source "${E11_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY

RUNNER="${E11_DIR}/e11_continuous_arrival_runner.py"
VERIFY_E11="${E11_DIR}/verify_e11_continuous_arrival.py"
FROZEN_ENV="${E11_DIR}/configs/e2_ours_2g_env.json"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7}"
FIXTURE="${FIXTURE:-${E11_DIR}/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/evaluation_short_dialogue.json}"
TIMEOUT_SEC="${TIMEOUT_SEC:-5400}"
CPUSET="${CPUSET:-0-13}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-2147483648}"

require_file "${RUNNER}"
require_file "${VERIFY_E11}"
require_file "${FROZEN_ENV}"
require_file "${MODEL}/config.json"
require_file "${FIXTURE}"
"${VLLM_PY}" -m py_compile "${RUNNER}" "${VERIFY_E11}"
(
  cd "${VLLM_ROOT}"
  "${VLLM_PY}" -m pytest -q tests/v1/core/test_scheduler.py \
    -k 'phase_batched_scheduler'
)
runtime_preflight

run_root="${RUN_ROOT:-${E11_DIR}/artifacts/e11-continuous-arrival-$(date -u +%Y%m%dT%H%M%SZ)-attempt-01-qualification}"
[[ ! -e "${run_root}" ]] || die "RUN_ROOT already exists: ${run_root}"
mkdir -p "${run_root}"
run_root="$(cd "${run_root}" && pwd)"
RUN_ROOT_ACTIVE="${run_root}"
module_snapshot "${run_root}/module.before.json" "${STOCK_UVM}"
load_custom_module
module_snapshot "${run_root}/module.custom.json" "${CUSTOM_UVM}"
git -C "${E11_ROOT}" status --short >"${run_root}/git_status.txt"
sha256sum "${RUNNER}" "${VERIFY_E11}" "${FROZEN_ENV}" "${FIXTURE}" \
  "${MODEL}/config.json" "${CUSTOM_UVM}" >"${run_root}/input_sha256.txt"

mapfile -t frozen_env < <(
  "${VLLM_PY}" - "${FROZEN_ENV}" <<'PY'
import json, sys
for key, value in json.load(open(sys.argv[1], encoding="utf-8")).items():
    print(f"{key}={value}")
PY
)

failures=0
for mode in remote_phase_epoch remote_continuous hybrid_fail_closed; do
  arm_root="${run_root}/${mode}"
  mkdir "${arm_root}"
  : >"${arm_root}/cpu_offload_manifest.jsonl"
  wait_gpu_clean || die "GPU is not clean before ${mode}"
  gpu_state_json "${arm_root}/gpu.before.json"
  kernel_epoch="$(date +%s)"
  phase_mode=prefill_graph_remote_decode
  phase_batched=0
  park_resume=1
  prefill_capture=1
  expected_rc=1
  if [[ "${mode}" == remote_phase_epoch ]]; then
    phase_mode=remote_only
    phase_batched=1
    park_resume=0
    prefill_capture=0
    expected_rc=0
  elif [[ "${mode}" == remote_continuous ]]; then
    phase_mode=remote_only
    park_resume=0
    prefill_capture=0
    expected_rc=0
  fi
  env_overrides=(
    "VLLM_UVM_ORIGINAL_VA_PHASE_MODE=${phase_mode}"
    "VLLM_UVM_PHASE_BATCHED_SCHEDULING=${phase_batched}"
    "VLLM_UVM_PHASE_KV_AWARE_COHORTS=${phase_batched}"
    "VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=${park_resume}"
    "VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=${prefill_capture}"
    "VLLM_CPU_OFFLOAD_MANIFEST_FILE=${arm_root}/cpu_offload_manifest.jsonl"
    "VLLM_CPU_OFFLOAD_MANIFEST_RESET=1"
    "VLLM_UVM_LOG_FILE=${arm_root}/uvm.log"
  )
  printf '%s\n' "${frozen_env[@]}" "${env_overrides[@]}" \
    >"${arm_root}/environment.txt"
  {
    printf 'cd %q && env ' "${VLLM_ROOT}"
    printf '%q ' "${frozen_env[@]}" "${env_overrides[@]}"
    printf 'timeout --kill-after=30s %qs taskset -c %q %q %q ' \
      "${TIMEOUT_SEC}" "${CPUSET}" "${VLLM_PY}" "${RUNNER}"
    printf '%q ' --model "${MODEL}" --fixture "${FIXTURE}" \
      --output "${arm_root}/result.json" --mode "${mode}" \
      --kv-cache-memory-bytes "${KV_CACHE_BYTES}"
    printf '\n'
  } >"${arm_root}/command.txt"

  set +e
  (
    cd "${VLLM_ROOT}"
    env "${frozen_env[@]}" "${env_overrides[@]}" \
      /usr/bin/time -v -o "${arm_root}/resource_time.txt" \
      timeout --kill-after=30s "${TIMEOUT_SEC}s" \
      taskset -c "${CPUSET}" \
      "${VLLM_PY}" "${RUNNER}" \
        --model "${MODEL}" \
        --fixture "${FIXTURE}" \
        --output "${arm_root}/result.json" \
        --mode "${mode}" \
        --kv-cache-memory-bytes "${KV_CACHE_BYTES}"
  ) >"${arm_root}/run.log" 2>&1
  rc=$?
  sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
    >"${arm_root}/kernel.since.log" 2>"${arm_root}/kernel.since.err"
  set -e
  printf '%s\n' "${rc}" >"${arm_root}/returncode.txt"
  wait_gpu_clean || die "GPU/client state did not clean after ${mode}"
  gpu_state_json "${arm_root}/gpu.after.json"
  if (( rc != expected_rc )); then
    failures=$((failures + 1))
    tail -n 240 "${arm_root}/run.log" >&2 || true
  fi
done

restore_stock || die "failed to restore stock nvidia_uvm"
module_snapshot "${run_root}/module.after.json" "${STOCK_UVM}"
restart_exporters
set +e
"${VLLM_PY}" "${VERIFY_E11}" "${run_root}" \
  --output "${run_root}/verification.json" \
  >"${run_root}/verify.log" 2>"${run_root}/verify.err"
verify_rc=$?
set -e
echo "RUN_ROOT=${run_root}"
echo "verification=${run_root}/verification.json"
(( failures == 0 && verify_rc == 0 )) \
  || die "E11 qualification failed: arm_failures=${failures}, verifier=${verify_rc}"
