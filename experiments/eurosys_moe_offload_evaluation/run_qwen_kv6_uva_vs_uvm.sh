#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_PY="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
RUNNER="${SCRIPT_DIR}/direct_batch_c16_vllm_runner.py"
SUMMARIZER="${SCRIPT_DIR}/summarize_qwen_kv6_uva_vs_uvm.py"
FIXTURE_ROOT="${SCRIPT_DIR}/inputs/generated/eurosys-pa-v2-20260824/qwen3_30b_a3b_fp8"
MODEL="/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782"
RUN_ROOT="${1:-${REPO_ROOT}/my_paper/data/runs/Qwen_kv6_uva_vs_stock_uvm_$(date -u +%Y%m%dT%H%M%SZ)}"
CPUSET="${CPUSET:-0-13}"
CASE_TIMEOUT_SECONDS="${CASE_TIMEOUT_SECONDS:-7200}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-6442450944}"
UVA_OFFLOAD_GIB="${UVA_OFFLOAD_GIB:-6}"
MODE_FILTER="${MODE_FILTER:-native_uva_6g,stock_uvm_demand}"
WORKLOAD_FILTER="${WORKLOAD_FILTER:-short_dialogue,ordinary_chat,medium_context,rag_summary,long_generation}"
CONCURRENCY_FILTER="${CONCURRENCY_FILTER:-1,16}"
RESUME="${RESUME:-0}"
SAMPLER_PID=""

die() { echo "Qwen KV6 UVA-vs-UVM: $*" >&2; exit 1; }

selected() {
  local csv="$1" value="$2"
  [[ ",${csv}," == *",${value},"* ]]
}

completed() {
  local mode="$1" workload="$2" concurrency="$3"
  awk -F '\t' -v mode="${mode}" -v workload="${workload}" -v concurrency="${concurrency}" \
    'NR > 1 && $1 == mode && $2 == workload && $3 == concurrency {found=1} END {exit !found}' \
    "${RUN_ROOT}/execution.tsv"
}

wait_gpu_clean() {
  local used pids
  for _ in $(seq 1 300); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= 128 )) \
        && [[ -z "${pids//[[:space:]]/}" ]]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

stop_sampler() {
  if [[ -n "${SAMPLER_PID}" ]]; then
    kill -TERM "${SAMPLER_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}" 2>/dev/null || true
    SAMPLER_PID=""
  fi
}
trap stop_sampler EXIT INT TERM

profile() {
  local workload="$1" prompt output max_model timeout_ttft timeout_tpot
  case "${workload}" in
    short_dialogue)
      prompt=128; output=32; max_model=160; timeout_ttft=2000; timeout_tpot=100 ;;
    ordinary_chat)
      prompt=512; output=128; max_model=640; timeout_ttft=5000; timeout_tpot=150 ;;
    medium_context)
      prompt=2048; output=128; max_model=2176; timeout_ttft=10000; timeout_tpot=200 ;;
    rag_summary)
      prompt=8192; output=128; max_model=8320; timeout_ttft=30000; timeout_tpot=200 ;;
    long_generation)
      prompt=512; output=512; max_model=1024; timeout_ttft=5000; timeout_tpot=150 ;;
    *) return 1 ;;
  esac
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${prompt}" "${output}" "${max_model}" "${timeout_ttft}" "${timeout_tpot}"
}

run_case() {
  local mode="$1" workload="$2" concurrency="$3"
  local case_root prompt output max_model max_batched ttft_slo tpot_slo
  local arm cpu_offload rc status
  local -a mode_env command

  IFS=$'\t' read -r prompt output max_model ttft_slo tpot_slo \
    <<<"$(profile "${workload}")"
  max_batched=$((prompt * concurrency))
  (( max_batched < max_model )) && max_batched="${max_model}"
  case_root="${RUN_ROOT}/cases/${mode}/qwen/${workload}/c${concurrency}"
  mkdir -p "${case_root}"

  if [[ "${mode}" == native_uva_6g ]]; then
    arm=native_pinned_uva
    cpu_offload="${UVA_OFFLOAD_GIB}"
    : >"${case_root}/offload_manifest.jsonl"
    mode_env=(
      VLLM_USE_UVM=0
      PYTORCH_ALLOC_CONF=expandable_segments:True
      VLLM_CPU_OFFLOAD_BACKEND=uva
      VLLM_CPU_OFFLOAD_MANIFEST_FILE="${case_root}/offload_manifest.jsonl"
      VLLM_CPU_OFFLOAD_MANIFEST_RESET=1
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=0
      VLLM_UVM_KV_CACHE_MANAGED=0
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
      VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
      VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0
      VLLM_UVM_KV_RUNTIME_ENABLE=0
      VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0
    )
  else
    arm=native_all_gpu
    cpu_offload=0
    mode_env=(
      VLLM_USE_UVM=1
      VLLM_UVM_PREFETCH=0
      VLLM_UVM_VERBOSE=0
      VLLM_UVM_POLICY_ENABLE=0
      VLLM_UVM_POLICY_WARMUP_PREFETCH=0
      VLLM_UVM_POLICY_WARMUP_ADVISE_GPU=0
      VLLM_UVM_GAP_WATCH_ENABLE=0
      VLLM_UVM_DEVICE_DIRECT_ENABLE=0
      VLLM_UVM_KV_DEVICE_DIRECT_ENABLE=0
      VLLM_UVM_FIXED_WEIGHT_DEVICE_DIRECT_ENABLE=0
      VLLM_UVM_SCRATCH_POOL_ENABLE=0
      VLLM_UVM_WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE=0
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
      VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
      VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0
      VLLM_UVM_MATCHED_CPU_OFFLOAD_ENABLE=0
      VLLM_UVM_KV_RUNTIME_ENABLE=0
      VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0
      VLLM_UVM_LOG_FILE="${case_root}/uvm_allocations.log"
    )
  fi

  command=(
    "${VLLM_PY}" "${RUNNER}"
    --arm "${arm}"
    --model "${MODEL}"
    --fixture "${FIXTURE_ROOT}/evaluation_${workload}.json"
    --warmup-fixture "${FIXTURE_ROOT}/calibration_${workload}.json"
    --output "${case_root}/result.json"
    --workload "${workload}"
    --concurrency "${concurrency}"
    --prompt-tokens "${prompt}"
    --output-tokens "${output}"
    --warmup-requests 3
    --warmup-output-tokens 8
    --seed 20260823
    --max-model-len "${max_model}"
    --max-num-batched-tokens "${max_batched}"
    --kv-cache-memory-bytes "${KV_CACHE_BYTES}"
    --cpu-offload-gb "${cpu_offload}"
    --ttft-slo-ms "${ttft_slo}"
    --tpot-slo-ms "${tpot_slo}"
  )

  wait_gpu_clean || die "GPU is not clean before ${mode}/${workload}/c${concurrency}"
  nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw \
    --format=csv,noheader,nounits --loop-ms=200 >"${case_root}/nvml.csv" 2>"${case_root}/nvml.err" &
  SAMPLER_PID="$!"
  printf '%q ' env "${mode_env[@]}" VLLM_ATTENTION_BACKEND=TRITON_ATTN "${command[@]}" \
    >"${case_root}/command.txt"
  printf '\n' >>"${case_root}/command.txt"
  printf '%s\n' "${mode_env[@]}" VLLM_ATTENTION_BACKEND=TRITON_ATTN \
    >"${case_root}/environment.txt"

  echo "KV6_CELL_START mode=${mode} workload=${workload} c=${concurrency}" >&2
  env -u VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE \
      -u VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS \
      -u VLLM_UVM_PRE_REBALANCE_MANUAL_MIGRATION_FAULT_MODE \
      CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 \
      VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONUNBUFFERED=1 \
      VLLM_ATTENTION_BACKEND=TRITON_ATTN \
      "${mode_env[@]}" \
      timeout --kill-after=30s "${CASE_TIMEOUT_SECONDS}" \
      taskset -c "${CPUSET}" "${command[@]}" >"${case_root}/run.log" 2>&1
  rc=$?
  stop_sampler
  echo "${rc}" >"${case_root}/returncode.txt"
  status=error
  if (( rc == 0 )) && jq -e \
      '.status == "ok" and .measurement.successful_requests == .configuration.concurrency' \
      "${case_root}/result.json" >/dev/null 2>&1; then
    status=ok
  elif rg -q 'OutOfMemoryError|CUDA out of memory' "${case_root}/run.log"; then
    status=OOM
  elif (( rc == 124 || rc == 137 )); then
    status=timeout
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${mode}" "${workload}" "${concurrency}" "${status}" "${rc}" \
    "${KV_CACHE_BYTES}" "${case_root}" >>"${RUN_ROOT}/execution.tsv"
  echo "KV6_CELL_DONE mode=${mode} workload=${workload} c=${concurrency} status=${status} rc=${rc}" >&2
  wait_gpu_clean || die "GPU did not clean up after ${mode}/${workload}/c${concurrency}"
}

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "set ALLOW_GPU_RUN=1"
[[ "${KV_CACHE_BYTES}" == 6442450944 ]] || die "this protocol requires exactly 6 GiB of KV cache"
[[ "${UVA_OFFLOAD_GIB}" == 6 || "${UVA_OFFLOAD_GIB}" == 6.0 ]] \
  || die "this protocol requires cpu_offload_gb=6 for UVA"
[[ "${RESUME}" == 0 || "${RESUME}" == 1 ]] || die "RESUME must be 0 or 1"
if [[ "${RESUME}" == 1 ]]; then
  [[ -d "${RUN_ROOT}" && -f "${RUN_ROOT}/execution.tsv" ]] \
    || die "resume root is missing execution.tsv: ${RUN_ROOT}"
else
  [[ ! -e "${RUN_ROOT}" ]] || die "run root already exists: ${RUN_ROOT}"
fi
for path in "${VLLM_PY}" "${RUNNER}" "${SUMMARIZER}" "${MODEL}/config.json"; do
  [[ -e "${path}" ]] || die "missing required path: ${path}"
done
for workload in short_dialogue ordinary_chat medium_context rag_summary long_generation; do
  [[ -f "${FIXTURE_ROOT}/evaluation_${workload}.json" ]] \
    || die "missing evaluation fixture for ${workload}"
  [[ -f "${FIXTURE_ROOT}/calibration_${workload}.json" ]] \
    || die "missing calibration fixture for ${workload}"
done
stock_src="$(modinfo -F srcversion nvidia-uvm)"
loaded_src="$(cat /sys/module/nvidia_uvm/srcversion)"
[[ "${stock_src}" == "${loaded_src}" ]] || die "loaded nvidia-uvm is not the stock module"
exec 9>"/tmp/qwen_kv6_uva_vs_uvm.lock"
flock -n 9 || die "another Qwen KV6 run holds the GPU lock"

if [[ "${RESUME}" == 0 ]]; then
  mkdir -p "${RUN_ROOT}/provenance" "${RUN_ROOT}/cases"
  RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
  printf 'mode\tworkload\tconcurrency\tstatus\treturncode\tkv_cache_bytes\tcase_root\n' \
    >"${RUN_ROOT}/execution.tsv"
  git -C "${REPO_ROOT}" rev-parse HEAD >"${RUN_ROOT}/provenance/repo.commit"
  git -C "${REPO_ROOT}" status --short >"${RUN_ROOT}/provenance/repo.status.txt"
  nvidia-smi -q >"${RUN_ROOT}/provenance/nvidia-smi-q.txt"
  lscpu >"${RUN_ROOT}/provenance/lscpu.txt"
  uname -a >"${RUN_ROOT}/provenance/uname.txt"
  printf '%s\n' "${stock_src}" >"${RUN_ROOT}/provenance/stock_uvm_srcversion.txt"
  printf '%s\n' "${loaded_src}" >"${RUN_ROOT}/provenance/loaded_uvm_srcversion.txt"
  sha256sum "${RUNNER}" "${MODEL}/config.json" "${FIXTURE_ROOT}"/evaluation_*.json \
    "${FIXTURE_ROOT}"/calibration_*.json >"${RUN_ROOT}/provenance/inputs.sha256"
  jq -n --arg stock_src "${stock_src}" --arg runner "${RUNNER}" \
    --arg mode_filter "${MODE_FILTER}" --arg workload_filter "${WORKLOAD_FILTER}" \
    --arg concurrency_filter "${CONCURRENCY_FILTER}" \
    '{schema:"qwen_kv6_uva_vs_stock_uvm_v1",repeats:1,
      arms:["native_uva_6g","stock_uvm_demand"],model:"Qwen3-30B-A3B FP8",
      workloads:["short_dialogue","ordinary_chat","medium_context","rag_summary","long_generation"],
      concurrencies:[1,16],mode_filter:$mode_filter,workload_filter:$workload_filter,
      concurrency_filter:$concurrency_filter,kv_cache_bytes:6442450944,kv_cache_gib:6,
      transport:"none",execution:"eager direct batch",attention_backend:"TRITON_ATTN",
      warmup_requests:3,warmup_output_tokens:8,seed:20260823,
      uva:{backend:"pinned UVA",cpu_offload_gb:6},
      uvm:{allocation:"cudaMallocManaged",cpu_offload_gb:0,advice:false,prefetch:false,
           policy:false,stock_srcversion:$stock_src},runner:$runner}' >"${RUN_ROOT}/protocol.json"
else
  RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
  [[ "$(jq -r '.schema' "${RUN_ROOT}/protocol.json")" == qwen_kv6_uva_vs_stock_uvm_v1 ]] \
    || die "resume root has an incompatible protocol"
fi

for mode in native_uva_6g stock_uvm_demand; do
  selected "${MODE_FILTER}" "${mode}" || continue
  for workload in short_dialogue ordinary_chat medium_context rag_summary long_generation; do
    selected "${WORKLOAD_FILTER}" "${workload}" || continue
    for concurrency in 1 16; do
      selected "${CONCURRENCY_FILTER}" "${concurrency}" || continue
      if completed "${mode}" "${workload}" "${concurrency}"; then
        echo "KV6_CELL_SKIP mode=${mode} workload=${workload} c=${concurrency} reason=completed" >&2
        continue
      fi
      [[ ! -e "${RUN_ROOT}/cases/${mode}/qwen/${workload}/c${concurrency}" ]] \
        || die "incomplete case directory exists; preserve or move it before resume: ${mode}/${workload}/c${concurrency}"
      run_case "${mode}" "${workload}" "${concurrency}"
    done
  done
done

"${VLLM_PY}" "${SUMMARIZER}" "${RUN_ROOT}" || true
echo "RUN_ROOT=${RUN_ROOT}"
