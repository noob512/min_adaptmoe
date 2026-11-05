#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
PROBE_RUNNER="${SCRIPT_DIR}/benchmark_qwen_fragmentation_probe.py"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-${SCRIPT_DIR}/artifacts/e3-calibration-20260825T091247Z}"
PLAN="${CALIBRATION_ROOT}/calibration_assets/cold_decode_plan_2g.json"
NAMES="${CALIBRATION_ROOT}/calibration_assets/exact_parameter_names.txt"
FIXTURE="${SCRIPT_DIR}/inputs/generated/eurosys-pa-v2-20260824/qwen3_30b_a3b_fp8/evaluation_short_dialogue.json"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/qwen_fragmentation_mixed_probe_$(date -u +%Y%m%dT%H%M%SZ)_attempt_01}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-1800000000}"
CPUSET="${CPUSET:-0-13}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"
POST_KV_HOT_RECONCILE="${POST_KV_HOT_RECONCILE:-1}"
INIT_WEIGHT_OFFLOAD_ENABLE="${INIT_WEIGHT_OFFLOAD_ENABLE:-0}"
INIT_WEIGHT_OFFLOAD_STAGE="${INIT_WEIGHT_OFFLOAD_STAGE:-post_warmup}"
INIT_WEIGHT_OFFLOAD_MODE="${INIT_WEIGHT_OFFLOAD_MODE:-trace_only}"
EXPERT_ENVELOPE_ENABLE="${EXPERT_ENVELOPE_ENABLE:-0}"
FAULT_MODE_AFTER_WARMUP="${FAULT_MODE_AFTER_WARMUP:-}"
FAULT_MODE_CONTROLLER_TIMEOUT_SEC="${FAULT_MODE_CONTROLLER_TIMEOUT_SEC:-300}"
SELECTED_BYTES=2264924160
FULL_BYTES=28991029248
BACKING_BYTES=94371840
MONITOR_PID=""
FAULT_MODE_SWITCH_PID=""
CUSTOM_LOADED=0

die() { echo "error: $*" >&2; exit 1; }

stop_monitor() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=""
  fi
}

stop_fault_mode_switch() {
  if [[ -n "${FAULT_MODE_SWITCH_PID}" ]]; then
    kill "${FAULT_MODE_SWITCH_PID}" 2>/dev/null || true
    wait "${FAULT_MODE_SWITCH_PID}" 2>/dev/null || true
    FAULT_MODE_SWITCH_PID=""
  fi
}

cleanup_probe() {
  stop_fault_mode_switch
  stop_monitor
  if [[ "${CUSTOM_LOADED}" == 1 ]]; then
    restore_stock || true
    CUSTOM_LOADED=0
    restart_exporters || true
  fi
}

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
for path in "${VLLM_PY}" "${PROBE_RUNNER}" "${MODEL}/config.json" "${PLAN}" "${NAMES}" "${FIXTURE}"; do
  [[ -f "${path}" ]] || die "missing file: ${path}"
done
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"
mkdir -p "${RUN_ROOT}/provenance" "${RUN_ROOT}/case"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
# shellcheck source=/dev/null
source "${BASE_HARNESS}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY
trap cleanup_probe EXIT INT TERM

RUN_ROOT_ACTIVE="${RUN_ROOT}"
runtime_preflight || die "custom runtime preflight failed"
sha256sum "${PLAN}" "${NAMES}" "${FIXTURE}" "${MODEL}/config.json" "${PROBE_RUNNER}" \
  >"${RUN_ROOT}/provenance/inputs.sha256"
git -C "${REPO_ROOT}" status --short >"${RUN_ROOT}/provenance/repo.status.txt"
nvidia-smi -q >"${RUN_ROOT}/provenance/nvidia-smi-q.txt"
cp "${PLAN}" "${RUN_ROOT}/provenance/cold_decode_plan_2g.json"
module_snapshot "${RUN_ROOT}/provenance/module.before.json" "${STOCK_UVM}"
load_custom_module || die "could not load custom UVM"
CUSTOM_LOADED=1
module_snapshot "${RUN_ROOT}/provenance/module.custom.json" "${CUSTOM_UVM}"

case_dir="${RUN_ROOT}/case"
: >"${case_dir}/manifest.jsonl"
fault_mode_marker="${case_dir}/fault_mode_switch.request"
fault_mode_result="${case_dir}/fault_mode_switch.result"
if [[ -n "${FAULT_MODE_AFTER_WARMUP}" ]]; then
  [[ "${FAULT_MODE_AFTER_WARMUP}" =~ ^[012]$ ]] \
    || die "FAULT_MODE_AFTER_WARMUP must be 0, 1, or 2"
  (
    deadline=$((SECONDS + FAULT_MODE_CONTROLLER_TIMEOUT_SEC))
    while ((SECONDS < deadline)); do
      if [[ -f "${fault_mode_marker}" ]]; then
        before="$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode)"
        printf '%s\n' "${FAULT_MODE_AFTER_WARMUP}" \
          | sudo tee /sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode \
          >/dev/null
        after="$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode)"
        printf 'before=%s\ntarget=%s\nafter=%s\n' \
          "${before}" "${FAULT_MODE_AFTER_WARMUP}" "${after}" \
          >"${fault_mode_result}"
        [[ "${after}" == "${FAULT_MODE_AFTER_WARMUP}" ]]
        exit
      fi
      sleep 0.05
    done
    printf 'error=marker_timeout\n' >"${fault_mode_result}"
    exit 1
  ) >"${case_dir}/fault_mode_switch.log" 2>&1 &
  FAULT_MODE_SWITCH_PID="$!"
fi
kernel_epoch="$(date +%s)"
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits -lms 200 >"${case_dir}/nvml.csv" 2>"${case_dir}/nvml.err" &
MONITOR_PID="$!"

set +e
env VLLM_ATTENTION_BACKEND=TRITON_ATTN VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  VLLM_CPU_OFFLOAD_BACKEND=copy VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${NAMES}" \
  VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1 VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER=1 \
  VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va \
  VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="${PLAN}" \
  VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288 \
  VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=4k \
  VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPECTED_ARENA_COUNT=4 \
  VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPECTED_BACKING_BYTES="${BACKING_BYTES}" \
  VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 VLLM_UVM_ORIGINAL_VA_PHASE_MODE=off \
  VLLM_UVM_ORIGINAL_VA_PHASE_RESULT_FILE="${case_dir}/transition.json" \
  VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ=0 \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE="${INIT_WEIGHT_OFFLOAD_ENABLE}" \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE="${INIT_WEIGHT_OFFLOAD_STAGE}" \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE="${INIT_WEIGHT_OFFLOAD_MODE}" \
  VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${PLAN}" \
  VLLM_UVM_EXPERT_ENVELOPE_ENABLE="${EXPERT_ENVELOPE_ENABLE}" \
  VLLM_UVM_POST_WARMUP_REBALANCE_FAIL_FAST=1 \
  VLLM_UVM_PRE_REBALANCE_MANUAL_MIGRATION_FAULT_MODE="${FAULT_MODE_AFTER_WARMUP}" \
  VLLM_UVM_PRE_REBALANCE_FAULT_MODE_MARKER="${fault_mode_marker}" \
  VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0 \
  VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0 VLLM_CPU_OFFLOAD_COPY_TRACE=0 \
  VLLM_CPU_OFFLOAD_COPY_SYNC=0 VLLM_CPU_OFFLOAD_MANIFEST_FILE="${case_dir}/manifest.jsonl" \
  VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 VLLM_UVM_LOG_FILE="${case_dir}/uvm.log" \
  VLLM_UVM_PHASE_F_DEBUG_LOG="${case_dir}/phase_f_debug.log" \
  VLLM_UVM_FUSED_MOE_POST_WORKSPACE_RESIDENCY_AUDIT=1 \
  VLLM_UVM_FUSED_MOE_POST_WORKSPACE_AUDIT_MIN_TOKENS=7000 \
  VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE=1 \
  VLLM_UVM_POST_KV_HOT_RECONCILE="${POST_KV_HOT_RECONCILE}" \
  VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE_GRANULARITY_BYTES=2097152 \
  VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE_RUNTIME_GRANULARITY_BYTES=2097152 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 CUDA_VISIBLE_DEVICES=0 \
  timeout --kill-after=30s "${TIMEOUT_SEC}s" taskset -c "${CPUSET}" \
    "${VLLM_PY}" "${PROBE_RUNNER}" --model "${MODEL}" --fixture "${FIXTURE}" \
    --exact-names "${NAMES}" --expert-plan "${PLAN}" \
    --offload-manifest "${case_dir}/manifest.jsonl" --output "${case_dir}/result.json" \
    --progress "${case_dir}/progress.jsonl" --kv-cache-memory-bytes "${KV_CACHE_BYTES}" \
    --pure-prefill-tokens 7168 --mixed-prefill-tokens 7167 --lead-prefill-tokens 512 \
    --max-model-len 7172 --gpu-memory-utilization 0.97 --cpu-offload-gb 30.0 \
    --seed 0 --expected-selected-bytes "${SELECTED_BYTES}" \
  >"${case_dir}/run.log" 2>&1
rc=$?
set -e
stop_monitor
stop_fault_mode_switch
printf '%s\n' "${rc}" >"${case_dir}/returncode.txt"
sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
  >"${case_dir}/kernel.since.log" 2>"${case_dir}/kernel.since.err" || true
cp -- "${STATS_PATH}" "${case_dir}/latency_stats.after.txt" || true

wait_gpu_clean || gpu_state_json "${case_dir}/gpu.cleanup_failure.json"
restore_stock || die "failed to restore stock nvidia_uvm"
CUSTOM_LOADED=0
restart_exporters
module_snapshot "${RUN_ROOT}/provenance/module.after.json" "${STOCK_UVM}"
printf 'RUN_ROOT=%s\n' "${RUN_ROOT}"
printf 'returncode=%s\n' "${rc}"
exit "${rc}"
