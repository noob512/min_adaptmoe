#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
export PYTHONPATH="${VLLM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
RUNNER="${E3_NATIVE_PHASE_RUNNER_OVERRIDE:-${REPO_ROOT}/experiments/vllm_p3_pinned_uva_phase_compare/benchmark_phase_steps.py}"
MODEL="${MODEL:?Set MODEL to the local Qwen3-30B-A3B-FP8 snapshot directory}"
FIXTURE_ROOT="${FIXTURE_ROOT:-${SCRIPT_DIR}/inputs/generated/eurosys-pa-v2-20260824/qwen3_30b_a3b_fp8}"
RUN_ROOT="${1:-${SCRIPT_DIR}/artifacts/e3-native-copy-vs-uva-phase-$(date -u +%Y%m%dT%H%M%SZ)}"
CPUSET="${CPUSET:-0-13}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASH_ATTN}"
CONDITIONING_REPEATS="${CONDITIONING_REPEATS:-1}"
MEASUREMENT_REPEATS="${MEASUREMENT_REPEATS:-1}"
BATCH_FILTER="${BATCH_FILTER:-1,8}"
ARM_FILTER="${ARM_FILTER:-copy,uva}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-NONE}"
WORKLOAD_FILTER="${WORKLOAD_FILTER:-short_dialogue,ordinary_chat,medium_context,rag_summary,long_generation}"
TIMEOUT_SEC="${TIMEOUT_SEC:-7200}"
KV_BYTES_PER_TOKEN=98304
MONITOR_PID=""

die() { echo "E3 native phase: $*" >&2; exit 1; }

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | head -1 | tr -d '[:space:]'
}

compute_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk '$1 ~ /^[0-9]+$/ {print $1}'
}

wait_gpu_clean() {
  local attempt used
  for ((attempt=0; attempt<300; ++attempt)); do
    used="$(gpu_used_mib)"
    if [[ "${used}" == 0 && -z "$(compute_pids)" ]]; then return 0; fi
    sleep 1
  done
  echo "GPU did not become clean: used=$(gpu_used_mib) MiB pids=$(compute_pids | paste -sd, -)" >&2
  return 1
}

stop_monitor() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=""
  fi
}

cleanup() {
  stop_monitor
}

kv_bytes() {
  local prompt="$1" output="$2" batch="$3" rounded value
  rounded=$(( (prompt + output + 16 + 15) / 16 * 16 ))
  value=$((rounded * batch * KV_BYTES_PER_TOKEN))
  (( value < 167772160 )) && value=167772160
  echo "${value}"
}

# The native arms cannot select expert slices. RAG/C16 Copy FULL_SEPARATE
# requires the documented extra capture headroom (21.298364 GiB realized).
offload_gib() {
  local workload="$1" batch="$2" arm="$3"
  if [[ "${workload}" == rag_summary && "${batch}" == 16 \
      && "${arm}" == copy && "${CUDAGRAPH_MODE}" == FULL_SEPARATE ]]; then
    echo "${RAG_B16_COPY_GRAPH_OFFLOAD_GIB:-21.0}"
  elif [[ "${workload}" == rag_summary && "${batch}" == 16 ]]; then
    echo "${RAG_B16_OFFLOAD_GIB:-20.25}"
  elif [[ "${workload}" == rag_summary && "${batch}" == 8 ]]; then
    echo 13.5
  elif [[ "${workload}" == rag_summary \
      || ( "${batch}" == 16 \
           && ( "${workload}" == ordinary_chat \
                || "${workload}" == medium_context \
                || "${workload}" == long_generation ) ) \
      || ( "${workload}" == medium_context && "${batch}" == 8 ) ]]; then
    echo 7.875
  else
    echo 2.25
  fi
}

run_one() {
  local arm="$1" workload="$2" prompt="$3" output_tokens="$4" batch="$5"
  local case_dir label kv max_model max_batched measured offload rc status peak
  label="${workload}_b${batch}_${arm}"
  case_dir="${RUN_ROOT}/cases/${label}"
  mkdir "${case_dir}"
  : >"${case_dir}/manifest.jsonl"
  kv="$(kv_bytes "${prompt}" "${output_tokens}" "${batch}")"
  max_model=$((prompt + output_tokens))
  max_batched=$((prompt * batch))
  (( max_batched < max_model )) && max_batched="${max_model}"
  measured=$((output_tokens - 1 - 5))
  (( measured > 0 )) || die "invalid measured Decode steps for ${label}"
  offload="$(offload_gib "${workload}" "${batch}" "${arm}")"

  wait_gpu_clean || die "GPU is dirty before ${label}"
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu \
    --format=csv,noheader,nounits -lms 100 >"${case_dir}/nvml.csv" 2>"${case_dir}/nvml.err" &
  MONITOR_PID="$!"

  local -a environment=(
    VLLM_ATTENTION_BACKEND="${ATTENTION_BACKEND}"
    VLLM_USE_UVM=0
    VLLM_ENABLE_V1_MULTIPROCESSING=0
    VLLM_CPU_OFFLOAD_BACKEND="${arm}"
    VLLM_CPU_OFFLOAD_MANIFEST_FILE="${case_dir}/manifest.jsonl"
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1
    VLLM_CPU_OFFLOAD_COPY_TRACE=0
    VLLM_CPU_OFFLOAD_COPY_SYNC=0
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED=0
    VLLM_UVM_KV_CACHE_MANAGED=0
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
    VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
    VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0
    VLLM_UVM_KV_RUNTIME_ENABLE=0
    VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    TOKENIZERS_PARALLELISM=false
    OMP_NUM_THREADS=14
    MKL_NUM_THREADS=14
    CUDA_VISIBLE_DEVICES=0
  )
  if [[ "${CUDAGRAPH_MODE}" != NONE ]]; then
    environment+=(VLLM_CUDAGRAPH_DISPATCH_AUDIT=1)
  fi
  printf '%s\n' "${environment[@]}" >"${case_dir}/environment.txt"
  {
    printf 'cd %q && env ' "${VLLM_ROOT}"
    printf '%q ' "${environment[@]}"
    printf 'timeout --kill-after=30s %qs taskset -c %q %q %q ' \
      "${TIMEOUT_SEC}" "${CPUSET}" "${VLLM_PY}" "${RUNNER}"
    printf '%q ' --model "${MODEL}" \
      --fixture "${FIXTURE_ROOT}/evaluation_${workload}.json" \
      --output "${case_dir}/result.json" \
      --profile-label "e3_native_${arm}_${workload}_b${batch}" \
      --mechanism-class "vllm_native_${arm}_phase_crossover" \
      --batch-sizes "${batch}" \
      --conditioning-repeats "${CONDITIONING_REPEATS}" \
      --measurement-repeats "${MEASUREMENT_REPEATS}" \
      --warmup-decode-steps 5 --measured-decode-steps "${measured}" \
      --kv-cache-memory-bytes "${kv}" --cpu-offload-gb "${offload}" \
      --max-model-len "${max_model}" --max-num-seqs "${batch}" \
      --max-num-batched-tokens "${max_batched}" \
      --gpu-memory-utilization 0.97 --seed 0 \
      --cudagraph-mode "${CUDAGRAPH_MODE}"
    printf '\n'
  } >"${case_dir}/command.txt"

  echo "START ${label} offload_gib=${offload} kv_bytes=${kv} prefill_tokens=$((prompt * batch)) decode_steps=${measured}"
  set +e
  (
    cd "${VLLM_ROOT}"
    env "${environment[@]}" timeout --kill-after=30s "${TIMEOUT_SEC}s" \
      taskset -c "${CPUSET}" "${VLLM_PY}" "${RUNNER}" \
        --model "${MODEL}" \
        --fixture "${FIXTURE_ROOT}/evaluation_${workload}.json" \
        --output "${case_dir}/result.json" \
        --profile-label "e3_native_${arm}_${workload}_b${batch}" \
        --mechanism-class "vllm_native_${arm}_phase_crossover" \
        --batch-sizes "${batch}" \
        --conditioning-repeats "${CONDITIONING_REPEATS}" \
        --measurement-repeats "${MEASUREMENT_REPEATS}" \
        --warmup-decode-steps 5 --measured-decode-steps "${measured}" \
        --kv-cache-memory-bytes "${kv}" --cpu-offload-gb "${offload}" \
        --max-model-len "${max_model}" --max-num-seqs "${batch}" \
        --max-num-batched-tokens "${max_batched}" \
        --gpu-memory-utilization 0.97 --seed 0 \
        --cudagraph-mode "${CUDAGRAPH_MODE}"
  ) >"${case_dir}/run.log" 2>&1
  rc=$?
  set +e
  stop_monitor
  echo "${rc}" >"${case_dir}/returncode.txt"
  status=error
  if (( rc == 0 )) && jq -e --arg b "${batch}" \
    '.status=="ok" and (.config.batch_sizes==[($b|tonumber)]) and
     (.phase_summary[$b].measurement_repeats==1)' \
    "${case_dir}/result.json" >/dev/null 2>&1; then
    status=ok
  fi
  peak="$(awk -F, 'NR>1 {gsub(/ /,"",$2); if ($2+0>m) m=$2+0} END{print m+0}' "${case_dir}/nvml.csv")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${workload}" "${prompt}" "${output_tokens}" "${batch}" "${arm}" \
    "${offload}" "${kv}" "${status}" "${rc}" "${peak}" \
    >>"${RUN_ROOT}/execution.tsv"
  wait_gpu_clean || die "GPU did not clean after ${label}"
  echo "END ${label} status=${status} rc=${rc} peak_mib=${peak}"
  [[ "${status}" == ok ]]
}

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
for path in "${VLLM_PY}" "${RUNNER}" "${MODEL}/config.json" \
  "${FIXTURE_ROOT}/evaluation_short_dialogue.json" \
  "${FIXTURE_ROOT}/evaluation_ordinary_chat.json" \
  "${FIXTURE_ROOT}/evaluation_medium_context.json" \
  "${FIXTURE_ROOT}/evaluation_rag_summary.json" \
  "${FIXTURE_ROOT}/evaluation_long_generation.json"; do
  [[ -f "${path}" ]] || die "missing required file: ${path}"
done
[[ ! -e "${RUN_ROOT}" ]] || die "output already exists: ${RUN_ROOT}"
wait_gpu_clean || die "GPU is not initially clean"
mkdir -p "${RUN_ROOT}/cases" "${RUN_ROOT}/provenance"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
trap cleanup EXIT INT TERM

printf 'workload\tprompt\toutput\tbatch\tarm\toffload_gib\tkv_bytes\tstatus\treturncode\tpeak_hbm_mib\n' \
  >"${RUN_ROOT}/execution.tsv"
sha256sum "${RUNNER}" "${MODEL}/config.json" "${FIXTURE_ROOT}"/evaluation_*.json \
  >"${RUN_ROOT}/provenance/inputs.sha256"
git -C "${REPO_ROOT}" rev-parse HEAD >"${RUN_ROOT}/provenance/repo.commit"
git -C "${REPO_ROOT}" status --short >"${RUN_ROOT}/provenance/repo.status.txt"
nvidia-smi -q >"${RUN_ROOT}/provenance/nvidia-smi-q.txt"
lscpu >"${RUN_ROOT}/provenance/lscpu.txt"
uname -a >"${RUN_ROOT}/provenance/uname.txt"
jq -n --arg attention "${ATTENTION_BACKEND}" --arg batch_filter "${BATCH_FILTER}" \
  --arg arm_filter "${ARM_FILTER}" --arg cudagraph_mode "${CUDAGRAPH_MODE}" \
  --arg workload_filter "${WORKLOAD_FILTER}" \
  --argjson conditioning "${CONDITIONING_REPEATS}" \
  --argjson measurement "${MEASUREMENT_REPEATS}" \
  '{schema:"eurosys_e3_native_copy_vs_uva_phase_v2",process_blocks:1,
    arms:["copy","uva"],batch_filter:$batch_filter,workload_filter:$workload_filter,
    arm_filter:$arm_filter,cudagraph_mode:$cudagraph_mode,
    workloads:["short_dialogue","ordinary_chat","medium_context","rag_summary","long_generation"],
    phases:["strict_complete_prefill_step","strict_cached_one_token_decode_steps"],
    transport:"none",attention_backend:$attention,
    conditioning_repeats:$conditioning,measurement_repeats:$measurement,
    warmup_decode_steps:5,native_coordinate_contract:"same whole-Parameter budget except documented RAG/C16 Copy FULL_SEPARATE capture headroom"}' \
  >"${RUN_ROOT}/configuration.json"

workloads=(short_dialogue ordinary_chat medium_context rag_summary long_generation)
prompts=(128 512 2048 8192 512)
outputs=(32 128 128 128 512)
IFS=',' read -r -a batches <<<"${BATCH_FILTER}"
failed=0
cell_index=0
for wi in "${!workloads[@]}"; do
  [[ ",${WORKLOAD_FILTER}," == *",${workloads[$wi]},"* ]] || continue
  for batch in "${batches[@]}"; do
    if (( cell_index % 2 == 0 )); then arms=(uva copy); else arms=(copy uva); fi
    for arm in "${arms[@]}"; do
      [[ ",${ARM_FILTER}," == *",${arm},"* ]] || continue
      if ! run_one "${arm}" "${workloads[$wi]}" "${prompts[$wi]}" \
        "${outputs[$wi]}" "${batch}"; then
        failed=1
        echo "FAILED ${workloads[$wi]} b${batch} ${arm}; continuing independent cells" >&2
      fi
    done
    cell_index=$((cell_index + 1))
  done
done

wait_gpu_clean || die "GPU is not clean after matrix"
echo "RUN_ROOT=${RUN_ROOT}"
exit "${failed}"
