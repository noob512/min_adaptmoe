#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/workloads/vllm/vllm${PYTHONPATH:+:${PYTHONPATH}}"
VLLM_PY="${VLLM_PY:-${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python}"
RUNNER="${SCRIPT_DIR}/direct_batch_c16_vllm_runner.py"
FIXTURE_BASE="${SCRIPT_DIR}/inputs/generated/eurosys-pa-v2-20260824"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-}"
QWEN_MODEL="${QWEN_MODEL:-}"
RUN_ROOT="${1:-${SCRIPT_DIR}/artifacts/H_vllm_stock_uvm_$(date -u +%Y%m%dT%H%M%SZ)}"
CPUSET="${CPUSET:-0-13}"
CASE_TIMEOUT_SECONDS="${CASE_TIMEOUT_SECONDS:-7200}"
SAMPLER_PID=""

die() { echo "H vLLM Stock-UVM: $*" >&2; exit 1; }

wait_gpu_clean() {
  local used pids
  for _ in $(seq 1 180); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
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
  local model_key="$1" concurrency="$2" workload="$3"
  if [[ "${model_key}" == deepseek ]]; then
    case "${concurrency}:${workload}" in
      1:short_dialogue) echo $'128\t32\t192\t128\t6291456\t2000\t100' ;;
      1:ordinary_chat) echo $'512\t128\t672\t512\t22020096\t5000\t150' ;;
      1:rag_summary) echo $'8192\t128\t8320\t8192\t272629760\t30000\t200' ;;
      1:long_generation) echo $'512\t512\t1056\t512\t34603008\t5000\t150' ;;
      16:short_dialogue) echo $'128\t32\t192\t2048\t100663296\t2000\t100' ;;
      16:ordinary_chat) echo $'512\t128\t672\t8192\t352321536\t5000\t150' ;;
      16:rag_summary) echo $'8192\t128\t8320\t16384\t4362076160\t30000\t200' ;;
      16:long_generation) echo $'512\t512\t1056\t8192\t553648128\t5000\t150' ;;
      *) return 1 ;;
    esac
    return
  fi

  local prompt output max_model max_batched rounded kv
  case "${workload}" in
    short_dialogue) prompt=128; output=32; max_model=160 ;;
    ordinary_chat) prompt=512; output=128; max_model=640 ;;
    medium_context) prompt=2048; output=128; max_model=2176 ;;
    rag_summary) prompt=8192; output=128; max_model=8320 ;;
    long_generation) prompt=512; output=512; max_model=1024 ;;
    *) return 1 ;;
  esac
  max_batched=$((prompt * concurrency))
  (( max_batched < max_model )) && max_batched="${max_model}"
  rounded=$(( (prompt + output + 16 + 15) / 16 * 16 ))
  kv=$((rounded * concurrency * 98304))
  (( kv < 167772160 )) && kv=167772160
  case "${workload}" in
    short_dialogue) echo -e "${prompt}\t${output}\t${max_model}\t${max_batched}\t${kv}\t2000\t100" ;;
    ordinary_chat|long_generation) echo -e "${prompt}\t${output}\t${max_model}\t${max_batched}\t${kv}\t5000\t150" ;;
    medium_context) echo -e "${prompt}\t${output}\t${max_model}\t${max_batched}\t${kv}\t10000\t200" ;;
    rag_summary) echo -e "${prompt}\t${output}\t${max_model}\t${max_batched}\t${kv}\t30000\t200" ;;
  esac
}

run_case() {
  local mode="$1" model_key="$2" workload="$3" concurrency="$4"
  local model fixture_root case_root prompt output max_model max_batched kv ttft tpot rc status
  local -a mode_env backend_env command
  if [[ "${model_key}" == deepseek ]]; then
    model="${DEEPSEEK_MODEL}"
    fixture_root="${FIXTURE_BASE}/deepseek_v2_lite_bf16"
    backend_env=()
  else
    model="${QWEN_MODEL}"
    fixture_root="${FIXTURE_BASE}/qwen3_30b_a3b_fp8"
    backend_env=(VLLM_ATTENTION_BACKEND=TRITON_ATTN)
  fi
  IFS=$'\t' read -r prompt output max_model max_batched kv ttft tpot \
    <<<"$(profile "${model_key}" "${concurrency}" "${workload}")"
  case_root="${RUN_ROOT}/cases/${mode}/${model_key}/${workload}/c${concurrency}"
  mkdir -p "${case_root}"

  if [[ "${mode}" == stock_uvm_demand ]]; then
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
      VLLM_UVM_MATCHED_CPU_OFFLOAD_ENABLE=0
      VLLM_UVM_LOG_FILE="${case_root}/uvm_allocations.log"
    )
  else
    mode_env=(VLLM_USE_UVM=0 PYTORCH_ALLOC_CONF=expandable_segments:True)
  fi

  command=(
    "${VLLM_PY}" "${RUNNER}"
    --arm native_all_gpu
    --model "${model}"
    --fixture "${fixture_root}/evaluation_${workload}.json"
    --warmup-fixture "${fixture_root}/calibration_${workload}.json"
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
    --kv-cache-memory-bytes "${kv}"
    --cpu-offload-gb 0
    --ttft-slo-ms "${ttft}"
    --tpot-slo-ms "${tpot}"
  )

  wait_gpu_clean || die "GPU is not clean before ${mode}/${model_key}/${workload}/c${concurrency}"
  nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw \
    --format=csv,noheader,nounits --loop-ms=200 >"${case_root}/nvml.csv" 2>"${case_root}/nvml.err" &
  SAMPLER_PID="$!"
  printf '%q ' env "${mode_env[@]}" "${backend_env[@]}" "${command[@]}" >"${case_root}/command.txt"
  printf '\n' >>"${case_root}/command.txt"
  printf '%s\n' "${mode_env[@]}" "${backend_env[@]}" >"${case_root}/environment.txt"
  echo "H_CELL_START mode=${mode} model=${model_key} workload=${workload} c=${concurrency}" >&2
  env -u VLLM_CPU_OFFLOAD_BACKEND \
      -u VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE \
      -u VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS \
      -u VLLM_UVM_PRE_REBALANCE_MANUAL_MIGRATION_FAULT_MODE \
      CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 \
      VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONUNBUFFERED=1 \
      "${mode_env[@]}" "${backend_env[@]}" \
      timeout --kill-after=30s "${CASE_TIMEOUT_SECONDS}" \
      taskset -c "${CPUSET}" "${command[@]}" >"${case_root}/run.log" 2>&1
  rc=$?
  stop_sampler
  echo "${rc}" >"${case_root}/returncode.txt"
  status=error
  if (( rc == 0 )) && jq -e '.status == "ok" and .measurement.successful_requests == .configuration.concurrency' \
      "${case_root}/result.json" >/dev/null 2>&1; then
    status=ok
  elif rg -q 'OutOfMemoryError|CUDA out of memory' "${case_root}/run.log"; then
    status=OOM
  elif (( rc == 124 || rc == 137 )); then
    status=timeout
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${mode}" "${model_key}" "${workload}" "${concurrency}" "${status}" "${rc}" "${case_root}" \
    >>"${RUN_ROOT}/execution.tsv"
  echo "H_CELL_DONE mode=${mode} model=${model_key} workload=${workload} c=${concurrency} status=${status} rc=${rc}" >&2
  wait_gpu_clean || die "GPU did not clean up after ${mode}/${model_key}/${workload}/c${concurrency}"
}

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "set ALLOW_GPU_RUN=1"
[[ -n "${DEEPSEEK_MODEL}" && -n "${QWEN_MODEL}" ]] || die "set DEEPSEEK_MODEL and QWEN_MODEL to local model snapshots"
[[ ! -e "${RUN_ROOT}" ]] || die "run root already exists: ${RUN_ROOT}"
for path in "${VLLM_PY}" "${RUNNER}" "${DEEPSEEK_MODEL}/config.json" "${QWEN_MODEL}/config.json"; do
  [[ -e "${path}" ]] || die "missing required path: ${path}"
done
stock_src="$(modinfo -F srcversion nvidia-uvm)"
loaded_src="$(cat /sys/module/nvidia_uvm/srcversion)"
[[ "${stock_src}" == "${loaded_src}" ]] || die "loaded nvidia-uvm is not the stock module"
exec 9>"/tmp/h_vllm_stock_uvm.lock"
flock -n 9 || die "another H vLLM run holds the GPU lock"

mkdir -p "${RUN_ROOT}/provenance" "${RUN_ROOT}/cases"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
printf 'mode\tmodel\tworkload\tconcurrency\tstatus\treturncode\tcase_root\n' >"${RUN_ROOT}/execution.tsv"
git -C "${REPO_ROOT}" rev-parse HEAD >"${RUN_ROOT}/provenance/repo.commit"
git -C "${REPO_ROOT}" status --short >"${RUN_ROOT}/provenance/repo.status.txt"
nvidia-smi -q >"${RUN_ROOT}/provenance/nvidia-smi-q.txt"
printf '%s\n' "${stock_src}" >"${RUN_ROOT}/provenance/stock_uvm_srcversion.txt"
printf '%s\n' "${loaded_src}" >"${RUN_ROOT}/provenance/loaded_uvm_srcversion.txt"
sha256sum "${DEEPSEEK_MODEL}/config.json" "${QWEN_MODEL}/config.json" \
  "${FIXTURE_BASE}"/*/evaluation_*.json >"${RUN_ROOT}/provenance/inputs.sha256"
jq -n --arg stock_src "${stock_src}" --arg runner "${RUNNER}" \
  '{schema:"h_vllm_stock_uvm_v1",repeats:1,arms:["device_resident","stock_uvm_demand"],
    models:["DeepSeek-V2-Lite BF16","Qwen3-30B-A3B FP8"],concurrencies:[1,16],
    transport:"none",execution:"eager direct batch",cpu_offload_gb:0,warmup_requests:3,
    uvm:{allocation:"cudaMallocManaged",advice:false,prefetch:false,policy:false,stock_srcversion:$stock_src},
    runner:$runner}' >"${RUN_ROOT}/protocol.json"

for mode in device_resident stock_uvm_demand; do
  for model_key in deepseek qwen; do
    if [[ "${model_key}" == deepseek ]]; then
      workloads=(short_dialogue ordinary_chat rag_summary long_generation)
    else
      workloads=(short_dialogue ordinary_chat medium_context rag_summary long_generation)
    fi
    for workload in "${workloads[@]}"; do
      for concurrency in 1 16; do
        run_case "${mode}" "${model_key}" "${workload}" "${concurrency}"
      done
    done
  done
done

echo "RUN_ROOT=${RUN_ROOT}"
