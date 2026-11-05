#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_PY:-${VLLM_ROOT}/.venv/bin/python}"
BENCHMARK="${SCRIPT_DIR}/benchmark_clock_phases.py"
VERIFY="${SCRIPT_DIR}/verify_cpu_dma_latency_no_offload.py"
GUARD="${SCRIPT_DIR}/cpu_dma_latency_guard.py"
DEVICE="${CPU_DMA_LATENCY_DEVICE:-/dev/cpu_dma_latency}"

DEFAULT_MODEL="${REPO_ROOT}/../.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782"
DEFAULT_EVAL="${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json"
DEFAULT_WARMUP="${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/calibration_fixture_token_ids.json"

MODE_ORDER=BHHB
SLEEP_GAPS=110,500,500,110
BUSY_GAPS=110,500,500,110
LOCKED_GAPS=0
RECOVERY_GAPS=0

die() { echo "error: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }

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
      --query-gpu=index,name,uuid,memory.used,memory.free,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>&1 || true
  } > "$1"
}

reset_clocks() {
  sudo -n nvidia-smi -i "${GPU_INDEX:-0}" -rgc >/dev/null 2>&1 || true
  sudo -n nvidia-smi -i "${GPU_INDEX:-0}" -rmc >/dev/null 2>&1 || true
  sudo -n nvidia-smi -i "${GPU_INDEX:-0}" -pm 0 >/dev/null 2>&1 || true
}

GUARD_JOB_PID=""
GUARD_REAL_PID=""

stop_guard() {
  local rc=0
  if [[ -n "${GUARD_REAL_PID}" ]]; then
    sudo -n kill -TERM "${GUARD_REAL_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${GUARD_JOB_PID}" ]]; then
    wait "${GUARD_JOB_PID}" || rc=$?
  fi
  GUARD_JOB_PID=""
  GUARD_REAL_PID=""
  return "${rc}"
}

start_guard() {
  local control_dir="$1"
  local ready_file="${control_dir}/guard.ready.json"
  sudo -n python3 "${GUARD}" hold \
    --device "${DEVICE}" \
    --latency-us 0 \
    --timeout-seconds "${GUARD_TIMEOUT_SECONDS:-3600}" \
    --ready-file "${ready_file}" \
    > "${control_dir}/guard.events.jsonl" \
    2> "${control_dir}/guard.stderr.log" &
  GUARD_JOB_PID=$!

  local attempt
  for ((attempt = 0; attempt < 200; ++attempt)); do
    if [[ -s "${ready_file}" ]]; then
      GUARD_REAL_PID="$("${VLLM_PY}" -c \
        'import json,sys; print(int(json.load(open(sys.argv[1]))["pid"]))' \
        "${ready_file}")"
      sudo -n kill -0 "${GUARD_REAL_PID}" >/dev/null
      return 0
    fi
    kill -0 "${GUARD_JOB_PID}" >/dev/null 2>&1 \
      || die "cpu_dma_latency guard exited before becoming ready"
    sleep 0.05
  done
  die "timed out waiting for cpu_dma_latency guard"
}

NODE_WAS_ACTIVE=""
GPU_WAS_ACTIVE=""

restore_services() {
  [[ "${NODE_WAS_ACTIVE}" == "active" ]] \
    && sudo -n systemctl start node-exporter.service >/dev/null 2>&1 || true
  [[ "${GPU_WAS_ACTIVE}" == "active" ]] \
    && sudo -n systemctl start gpu-exporter.service >/dev/null 2>&1 || true
}

on_exit() {
  local original_rc="$1" cleanup_rc=0
  trap - EXIT INT TERM
  set +e
  stop_guard || cleanup_rc=$?
  reset_clocks
  restore_services
  set -e
  (( original_rc == 0 )) || exit "${original_rc}"
  exit "${cleanup_rc}"
}

static_preflight() {
  require_file "${VLLM_PY}"
  require_file "${BENCHMARK}"
  require_file "${VERIFY}"
  require_file "${GUARD}"
  require_file "${MODEL:-${DEFAULT_MODEL}}/config.json"
  require_file "${EVAL_FIXTURE:-${DEFAULT_EVAL}}"
  require_file "${WARMUP_FIXTURE:-${DEFAULT_WARMUP}}"
  bash -n "${BASH_SOURCE[0]}"
  "${VLLM_PY}" -m py_compile "${BENCHMARK}" "${VERIFY}" "${GUARD}"
  python3 "${GUARD}" inspect --device "${DEVICE}"
}

write_run_config() {
  local path="$1" stock_src="$2" loaded_src="$3"
  "${VLLM_PY}" - "${path}" "${stock_src}" "${loaded_src}" <<'PY'
import json, sys
path, stock_srcversion, loaded_srcversion = sys.argv[1:]
payload = {
    "schema": "p3_cpu_dma_latency_no_offload_v1",
    "mode_order": "BHHB",
    "modes": {"B": "baseline", "H": "dma_latency_held"},
    "continuous_batches": 2,
    "sleep_gaps_ms": [110, 500, 500, 110],
    "busy_gaps_ms": [110, 500, 500, 110],
    "locked_gaps_ms": [0],
    "recovery_gaps_ms": [0],
    "cpu_dma_latency_us": 0,
    "stock_srcversion": stock_srcversion,
    "loaded_srcversion": loaded_srcversion,
    "custom_module_cycle": False,
}
open(path, "w", encoding="utf-8").write(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY
}

run_case() {
  local symbol="$1" ordinal="$2" case_dir="$3"
  local label
  if [[ "${symbol}" == "B" ]]; then
    label=baseline
  else
    label=dma_latency_held
  fi
  mkdir -p "${case_dir}/control"
  : > "${case_dir}/cpu_offload_manifest.jsonl"
  : > "${case_dir}/slot_trace.jsonl"
  : > "${case_dir}/routing_trace.jsonl"
  wait_gpu_clean || die "GPU is not clean before case ${ordinal}"
  snapshot_gpu "${case_dir}/gpu.before.csv"

  local -a environment=(
    VLLM_USE_UVM=0
    VLLM_ENABLE_V1_MULTIPROCESSING=0
    VLLM_UVM_COLD_SLOT_POOL_DRIVER_MODE=slot_va_block_remap
    VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP=1
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED=0
    VLLM_UVM_EXPERT_ENVELOPE_ENABLE=0
    VLLM_UVM_EXPERT_ENVELOPE_BYTES=524288
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
    VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN=1
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup
    VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0
    VLLM_UVM_INIT_WEIGHT_PLAN_FILE=
    VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST=
    VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=restore
    VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD=0
    VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0
    VLLM_UVM_WEIGHT_EXECUTOR=cpp_batch
    VLLM_UVM_PREFILL_ORACLE_PREFETCH_MODE=off
    VLLM_UVM_POST_WARMUP_REBALANCE_FAIL_FAST=1
    VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0
    VLLM_UVM_KV_CACHE_MANAGED=0
    VLLM_UVM_KV_RUNTIME_ENABLE=0
    VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0
    VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
    VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0
    VLLM_CPU_OFFLOAD_BACKEND=uva
    VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE=
    VLLM_CPU_OFFLOAD_STATIC_BANK=0
    VLLM_CPU_OFFLOAD_COPY_TRACE=0
    VLLM_CPU_OFFLOAD_COPY_SYNC=0
    VLLM_CPU_OFFLOAD_MANIFEST_FILE="${case_dir}/cpu_offload_manifest.jsonl"
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1
    VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0
    VLLM_UVM_MOE_ROUTING_TRACE_FILE="${case_dir}/routing_trace.jsonl"
    VLLM_UVM_SLOT_DIRECT_TRACE_FILE="${case_dir}/slot_trace.jsonl"
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    TOKENIZERS_PARALLELISM=false
    OMP_NUM_THREADS="${CPU_THREADS:-14}"
    MKL_NUM_THREADS="${CPU_THREADS:-14}"
    CUDA_VISIBLE_DEVICES="${GPU_INDEX:-0}"
  )
  local -a arguments=(
    --model "${MODEL:-${DEFAULT_MODEL}}"
    --evaluation-fixture "${EVAL_FIXTURE:-${DEFAULT_EVAL}}"
    --warmup-fixture "${WARMUP_FIXTURE:-${DEFAULT_WARMUP}}"
    --output "${case_dir}/result.json"
    --profile-label "cpu_dma_latency_no_offload_${label}"
    --mechanism-class vllm_no_offload_control
    --concurrency 4
    --output-tokens 1
    --warmup-requests 4
    --warmup-output-tokens 1
    --kv-cache-memory-bytes "${KV_CACHE_BYTES:-1288490189}"
    --max-model-len 512
    --max-num-batched-tokens 2048
    --max-num-seqs 4
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.97}"
    --cpu-offload-gb 0
    --seed 0
    --require-prompt-token-ids
    --disable-chunked-prefill
    --continuous-batches 2
    --auto-gaps-ms "${SLEEP_GAPS}"
    --busy-gaps-ms "${BUSY_GAPS}"
    --locked-gaps-ms "${LOCKED_GAPS}"
    --recovery-gaps-ms "${RECOVERY_GAPS}"
    --locked-sm-mhz "${LOCKED_SM_MHZ:-2400}"
    --locked-memory-mhz "${LOCKED_MEMORY_MHZ:-14001}"
  )
  printf '%s\n' "${environment[@]}" > "${case_dir}/environment.txt"
  {
    echo "ordinal=${ordinal}"
    echo "symbol=${symbol}"
    echo "label=${label}"
    echo "started_at=$(date --iso-8601=seconds)"
  } > "${case_dir}/execution.env"
  {
    printf 'cd %q && env ' "${VLLM_ROOT}"
    printf '%q ' "${environment[@]}"
    printf 'timeout %qs taskset -c %q %q %q ' \
      "${TIMEOUT_SEC:-1800}" "${CPUSET:-0-13}" "${VLLM_PY}" "${BENCHMARK}"
    printf '%q ' "${arguments[@]}"
    printf '\n'
  } > "${case_dir}/command.txt"

  if [[ "${symbol}" == "H" ]]; then
    start_guard "${case_dir}/control"
  fi
  local rc
  set +e
  (
    cd "${VLLM_ROOT}"
    env "${environment[@]}" \
      /usr/bin/time -v -o "${case_dir}/resource_time.txt" \
      timeout "${TIMEOUT_SEC:-1800}s" taskset -c "${CPUSET:-0-13}" \
      "${VLLM_PY}" "${BENCHMARK}" "${arguments[@]}"
  ) > "${case_dir}/run.log" 2>&1
  rc=$?
  set -e
  echo "${rc}" > "${case_dir}/returncode.txt"
  if [[ "${symbol}" == "H" ]]; then
    sudo -n kill -0 "${GUARD_REAL_PID}" >/dev/null \
      || die "cpu_dma_latency guard ended during case ${ordinal}"
    stop_guard
  fi
  {
    echo "ended_at=$(date --iso-8601=seconds)"
    echo "returncode=${rc}"
  } >> "${case_dir}/execution.env"
  wait_gpu_clean || die "GPU did not become clean after case ${ordinal}"
  snapshot_gpu "${case_dir}/gpu.after.csv"
  if (( rc != 0 )); then
    tail -n 160 "${case_dir}/run.log" >&2 || true
    die "case ${ordinal}/${label} failed with ${rc}"
  fi
}

run_suite() {
  [[ "${ALLOW_GPU_RUN:-0}" == "1" ]] \
    || die "run requires explicit ALLOW_GPU_RUN=1"
  static_preflight >/dev/null
  command -v nvidia-smi >/dev/null || die "nvidia-smi unavailable"
  command -v taskset >/dev/null || die "taskset unavailable"
  command -v timeout >/dev/null || die "timeout unavailable"
  sudo -n true
  sudo -n test -c "${DEVICE}"
  sudo -n test -w "${DEVICE}"

  local stock_src loaded_src
  stock_src="$(modinfo -F srcversion nvidia-uvm)"
  loaded_src="$(tr -d '[:space:]' < /sys/module/nvidia_uvm/srcversion)"
  [[ -n "${stock_src}" && "${loaded_src}" == "${stock_src}" ]] \
    || die "stock nvidia-uvm must already be loaded"
  gpu_clean || die "GPU must start at exactly 0 MiB with no compute process"

  local suite_root
  suite_root="${SUITE_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_no_offload_dma_BHHB_v1}"
  [[ ! -e "${suite_root}" ]] || die "suite root already exists: ${suite_root}"
  mkdir -p "${suite_root}"
  write_run_config "${suite_root}/run_config.json" "${stock_src}" "${loaded_src}"
  snapshot_gpu "${suite_root}/gpu.initial.csv"

  NODE_WAS_ACTIVE="$(systemctl is-active node-exporter.service || true)"
  GPU_WAS_ACTIVE="$(systemctl is-active gpu-exporter.service || true)"
  trap 'on_exit $?' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  reset_clocks
  sudo -n systemctl stop node-exporter.service gpu-exporter.service

  printf 'ordinal\tsymbol\tlabel\tdirectory\treturncode\n' \
    > "${suite_root}/execution_status.tsv"
  local index symbol label case_dir
  for ((index = 0; index < ${#MODE_ORDER}; ++index)); do
    symbol="${MODE_ORDER:index:1}"
    [[ "${symbol}" == "B" ]] && label=baseline || label=dma_latency_held
    case_dir="${suite_root}/$(printf '%02d_%s' "$((index + 1))" "${label}")"
    run_case "${symbol}" "$((index + 1))" "${case_dir}"
    printf '%s\t%s\t%s\t%s\t0\n' \
      "$((index + 1))" "${symbol}" "${label}" "$(basename "${case_dir}")" \
      >> "${suite_root}/execution_status.tsv"
  done
  snapshot_gpu "${suite_root}/gpu.final.csv"
  "${VLLM_PY}" "${VERIFY}" "${suite_root}" \
    --output "${suite_root}/verification.json" \
    > "${suite_root}/verify.log" 2> "${suite_root}/verify.err"
  echo "SUITE_ROOT=${suite_root}"
  echo "verification=${suite_root}/verification.json"
}

case "${1:-help}" in
  static-preflight) static_preflight ;;
  run) run_suite ;;
  verify)
    [[ $# -eq 2 ]] || die "verify requires SUITE_ROOT"
    "${VLLM_PY}" "${VERIFY}" "$2"
    ;;
  *)
    echo "usage: $0 {static-preflight|run|verify SUITE_ROOT}" >&2
    exit 2
    ;;
esac
