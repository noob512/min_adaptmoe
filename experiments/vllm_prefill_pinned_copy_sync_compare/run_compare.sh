#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_PY:-${VLLM_ROOT}/.venv/bin/python}"
BENCHMARK="${REPO_ROOT}/experiments/vllm_prefill_p3_steady_compare/benchmark_steady_batches.py"
PREPARE="${REPO_ROOT}/experiments/vllm_slot_vs_pinned_copy/prepare_inputs.py"
VERIFY="${SCRIPT_DIR}/verify_compare.py"

DEFAULT_MODEL="${REPO_ROOT}/../.cache/huggingface/hub/"\
"models--Qwen--Qwen3-30B-A3B-FP8/snapshots/"\
"d206ba732169f29bb77fbf80fc2c4b81d4d30782"
DEFAULT_EVAL="${REPO_ROOT}/experiments/broad_offload_compare/"\
"../eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json"
DEFAULT_WARMUP="${REPO_ROOT}/experiments/broad_offload_compare/"\
"../eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/calibration_fixture_token_ids.json"

ORDER=ASSA
LAYER_BYTES=603979776
TOTAL_BYTES=2415919104

die() { echo "error: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }

gpu_clean() {
  local used pids
  used="$(
    nvidia-smi -i "${GPU_INDEX:-0}" --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]'
  )"
  pids="$(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1}'
  )"
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

NODE_WAS_ACTIVE=""
GPU_WAS_ACTIVE=""

restore_services() {
  [[ "${NODE_WAS_ACTIVE}" == "active" ]] \
    && sudo -n systemctl start node-exporter.service >/dev/null 2>&1 || true
  [[ "${GPU_WAS_ACTIVE}" == "active" ]] \
    && sudo -n systemctl start gpu-exporter.service >/dev/null 2>&1 || true
}

on_exit() {
  local original_rc="$1"
  trap - EXIT INT TERM
  set +e
  reset_clocks
  restore_services
  set -e
  exit "${original_rc}"
}

static_preflight() {
  require_file "${VLLM_PY}"
  require_file "${BENCHMARK}"
  require_file "${PREPARE}"
  require_file "${VERIFY}"
  require_file "${MODEL:-${DEFAULT_MODEL}}/config.json"
  require_file "${EVAL_FIXTURE:-${DEFAULT_EVAL}}"
  require_file "${WARMUP_FIXTURE:-${DEFAULT_WARMUP}}"
  bash -n "${BASH_SOURCE[0]}"
  "${VLLM_PY}" -m py_compile "${BENCHMARK}" "${PREPARE}" "${VERIFY}"
  rg -q 'VLLM_CPU_OFFLOAD_COPY_SYNC' \
    "${VLLM_ROOT}/vllm/model_executor/models/utils.py"
  rg -q 'copy_cuda_ms' "${VLLM_ROOT}/vllm/model_executor/models/utils.py"
}

write_run_config() {
  local path="$1" stock_src="$2" loaded_src="$3" names="$4"
  "${VLLM_PY}" - "${path}" "${stock_src}" "${loaded_src}" "${names}" <<'PY'
import hashlib, json, sys
path, stock_srcversion, loaded_srcversion, names_path = sys.argv[1:]
names_bytes = open(names_path, "rb").read()
payload = {
    "schema": "vllm_pinned_copy_sync_steady_v1",
    "order": "ASSA",
    "modes": {"A": "async", "S": "sync"},
    "stock_srcversion": stock_srcversion,
    "loaded_srcversion": loaded_srcversion,
    "custom_module_cycle": False,
    "conditioning_batches": 3,
    "measurement_batches": 5,
    "inter_batch_idle_ms": 0.0,
    "copy_trace": True,
    "exact_names_file": names_path,
    "exact_names_sha256": hashlib.sha256(names_bytes).hexdigest(),
    "layers": [0, 1, 2, 3],
    "bytes_per_layer": 603979776,
    "total_bytes_per_batch": 2415919104,
}
open(path, "w", encoding="utf-8").write(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY
}

run_case() {
  local symbol="$1" ordinal="$2" case_dir="$3" plan="$4" names="$5"
  local mode copy_sync
  if [[ "${symbol}" == "A" ]]; then
    mode=async
    copy_sync=0
  else
    mode=sync
    copy_sync=1
  fi
  mkdir -p "${case_dir}"
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
    VLLM_UVM_MOE_ROUTING_TRACE_FILE="${case_dir}/routing_trace.jsonl"
    VLLM_UVM_EXPERT_ENVELOPE_BYTES=524288
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup
    VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${plan}"
    VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN=1
    VLLM_UVM_SLOT_DIRECT_TRACE_FILE="${case_dir}/slot_trace.jsonl"
    VLLM_CPU_OFFLOAD_MANIFEST_FILE="${case_dir}/cpu_offload_manifest.jsonl"
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1
    VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST=
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED=0
    VLLM_UVM_EXPERT_ENVELOPE_ENABLE=0
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
    VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=restore
    VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD=0
    VLLM_CPU_OFFLOAD_BACKEND=copy
    VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${names}"
    VLLM_CPU_OFFLOAD_STATIC_BANK=0
    VLLM_CPU_OFFLOAD_COPY_TRACE=1
    VLLM_CPU_OFFLOAD_COPY_SYNC="${copy_sync}"
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
    --profile-label "pinned_copy_${mode}_steady_trace"
    --mechanism-class "vllm_prefill_pinned_copy_exact_l0_l3_${mode}"
    --concurrency 4
    --output-tokens 1
    --warmup-requests 4
    --warmup-output-tokens 1
    --kv-cache-memory-bytes "${KV_CACHE_BYTES:-1288490189}"
    --max-model-len 544
    --max-num-batched-tokens 2048
    --max-num-seqs 32
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.97}"
    --cpu-offload-gb 3.0
    --seed 0
    --require-prompt-token-ids
    --disable-chunked-prefill
    --conditioning-batches 3
    --measurement-batches 5
    --inter-batch-idle-ms 0
  )
  printf '%s\n' "${environment[@]}" > "${case_dir}/environment.txt"
  {
    echo "ordinal=${ordinal}"
    echo "symbol=${symbol}"
    echo "mode=${mode}"
    echo "copy_sync=${copy_sync}"
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
  {
    echo "ended_at=$(date --iso-8601=seconds)"
    echo "returncode=${rc}"
  } >> "${case_dir}/execution.env"
  wait_gpu_clean || die "GPU did not become clean after case ${ordinal}"
  snapshot_gpu "${case_dir}/gpu.after.csv"
  if (( rc != 0 )); then
    tail -n 160 "${case_dir}/run.log" >&2 || true
    die "case ${ordinal}/${mode} failed with ${rc}"
  fi
}

run_suite() {
  [[ "${ALLOW_GPU_RUN:-0}" == "1" ]] \
    || die "run requires explicit ALLOW_GPU_RUN=1"
  static_preflight
  command -v nvidia-smi >/dev/null || die "nvidia-smi unavailable"
  command -v taskset >/dev/null || die "taskset unavailable"
  command -v timeout >/dev/null || die "timeout unavailable"
  sudo -n true

  local stock_src loaded_src
  stock_src="$(modinfo -F srcversion nvidia-uvm)"
  loaded_src="$(tr -d '[:space:]' < /sys/module/nvidia_uvm/srcversion)"
  [[ -n "${stock_src}" && "${loaded_src}" == "${stock_src}" ]] \
    || die "stock nvidia-uvm must already be loaded"
  gpu_clean || die "GPU must start at exactly 0 MiB with no compute process"

  local run_root
  run_root="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_ASSA_v1}"
  [[ ! -e "${run_root}" ]] || die "run root already exists: ${run_root}"
  mkdir -p "${run_root}/inputs"
  "${VLLM_PY}" "${PREPARE}" --output-dir "${run_root}/inputs" \
    > "${run_root}/prepare_inputs.stdout.json"
  local plan="${run_root}/inputs/exact_l0_l3_plan.json"
  local names="${run_root}/inputs/exact_parameter_names.txt"
  require_file "${plan}"
  require_file "${names}"
  write_run_config "${run_root}/run_config.json" \
    "${stock_src}" "${loaded_src}" "${names}"
  sha256sum \
    "${MODEL:-${DEFAULT_MODEL}}/config.json" \
    "${EVAL_FIXTURE:-${DEFAULT_EVAL}}" \
    "${WARMUP_FIXTURE:-${DEFAULT_WARMUP}}" \
    "${plan}" "${names}" > "${run_root}/input_sha256.txt"
  git -C "${REPO_ROOT}" status --short > "${run_root}/git_status.txt"
  snapshot_gpu "${run_root}/gpu.initial.csv"

  NODE_WAS_ACTIVE="$(systemctl is-active node-exporter.service || true)"
  GPU_WAS_ACTIVE="$(systemctl is-active gpu-exporter.service || true)"
  trap 'on_exit $?' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  reset_clocks
  sudo -n systemctl stop node-exporter.service gpu-exporter.service

  printf 'ordinal\tsymbol\tmode\tdirectory\treturncode\n' \
    > "${run_root}/execution_status.tsv"
  local index symbol mode case_dir
  for ((index = 0; index < ${#ORDER}; ++index)); do
    symbol="${ORDER:index:1}"
    [[ "${symbol}" == "A" ]] && mode=async || mode=sync
    case_dir="${run_root}/$(printf '%02d_%s' "$((index + 1))" "${mode}")"
    run_case "${symbol}" "$((index + 1))" "${case_dir}" "${plan}" "${names}"
    printf '%s\t%s\t%s\t%s\t0\n' \
      "$((index + 1))" "${symbol}" "${mode}" "$(basename "${case_dir}")" \
      >> "${run_root}/execution_status.tsv"
  done
  snapshot_gpu "${run_root}/gpu.final.csv"
  "${VLLM_PY}" "${VERIFY}" "${run_root}" \
    --output "${run_root}/verification.json" \
    > "${run_root}/verify.log" 2> "${run_root}/verify.err"
  echo "RUN_ROOT=${run_root}"
  echo "verification=${run_root}/verification.json"
}

case "${1:-help}" in
  static-preflight) static_preflight ;;
  run) run_suite ;;
  verify)
    [[ $# -eq 2 ]] || die "verify requires RUN_ROOT"
    "${VLLM_PY}" "${VERIFY}" "$2"
    ;;
  *)
    echo "usage: $0 {static-preflight|run|verify RUN_ROOT}" >&2
    exit 2
    ;;
esac
