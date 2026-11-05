#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_RUNNER="${SCRIPT_DIR}/run_formal_e3.sh"
DIRECT_KT="${SCRIPT_DIR}/direct_batch_e3_ktransformers_runner.py"
VERIFIER="${SCRIPT_DIR}/verify_formal_e3.py"
PLOTTER="${SCRIPT_DIR}/plot_e3_heatmap.py"
KT_ROOT="${REPO_ROOT}/KTransformers"
KT_PY="${KT_ROOT}/.venv/bin/python"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-${SCRIPT_DIR}/artifacts/e3-calibration-20260825T091247Z}"
KT_STATS="${KT_STATS:-${CALIBRATION_ROOT}/calibration_assets/kt_stats/cold_decode_plan_2g_logical_count.pt}"
FIXTURE_ROOT="${FIXTURE_ROOT:-${SCRIPT_DIR}/inputs/generated/eurosys-pa-v2-20260824/qwen3_30b_a3b_fp8}"
RUN_ROOT="${1:-${SCRIPT_DIR}/artifacts/e3-single-round-no-http-$(date -u +%Y%m%dT%H%M%SZ)}"
CPUSET="${CPUSET:-0-13}"
ACTIVE_PID=""
MONITOR_PID=""

die() { echo "E3 no-HTTP: $*" >&2; exit 1; }

stop_active() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=""
  fi
  if [[ -n "${ACTIVE_PID}" ]] && kill -0 "${ACTIVE_PID}" 2>/dev/null; then
    kill -INT -- "-${ACTIVE_PID}" 2>/dev/null || true
    sleep 2
    kill -TERM -- "-${ACTIVE_PID}" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-${ACTIVE_PID}" 2>/dev/null || true
    wait "${ACTIVE_PID}" 2>/dev/null || true
  fi
  ACTIVE_PID=""
}
trap stop_active EXIT INT TERM

wait_gpu_clean() {
  local used pids
  for _ in $(seq 1 180); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= 128 )) \
      && [[ -z "${pids//[[:space:]]/}" ]]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

kv_bytes() {
  local prompt="$1" output="$2" batch="$3" rounded value
  rounded=$(( (prompt + output + 16 + 15) / 16 * 16 ))
  value=$((rounded * batch * 98304))
  (( value < 167772160 )) && value=167772160
  echo "${value}"
}

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"
for path in "${BASE_RUNNER}" "${DIRECT_KT}" "${VERIFIER}" "${PLOTTER}" \
  "${KT_PY}" "${MODEL}/config.json" "${KT_STATS}"; do
  [[ -f "${path}" ]] || die "missing required file: ${path}"
done
wait_gpu_clean || die "GPU is not clean before execution"

echo "E3 no-HTTP: starting custom H/A/R B1/B8"
ALLOW_GPU_RUN=1 PROCESS_BLOCKS=1 BATCH_FILTER=1,8 RUN_KT=0 \
RUN_ROOT="${RUN_ROOT}" CPUSET="${CPUSET}" \
  bash "${BASE_RUNNER}" || die "custom H/A/R execution failed"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"

jq -n \
  --arg schema eurosys_e3_single_round_no_http_v1 \
  --arg model "$(realpath "${MODEL}")" \
  --arg batches "1,8" \
  --arg transport none \
  '{schema:$schema,process_blocks:1,batches:$batches,transport:$transport,
    model:$model,custom_arms:["H","A","R"],kt_mode:"direct",
    mixed_prefill_decode:false,measurement_repeats:1}' \
  > "${RUN_ROOT}/single_round_config.json"

printf 'block\tworkload\tprompt\toutput\tbatches\tcase_dir\tkv_bytes_b8\tstatus\treturncode\n' \
  > "${RUN_ROOT}/kt/execution.tsv"

workloads=(short_dialogue ordinary_chat medium_context rag_summary long_generation)
prompts=(128 512 2048 8192 512)
outputs=(32 128 128 128 512)
for wi in "${!workloads[@]}"; do
  workload="${workloads[$wi]}"
  prompt="${prompts[$wi]}"
  output="${outputs[$wi]}"
  supported=()
  max_kv=0
  for batch in 1 8; do
    kv="$(kv_bytes "${prompt}" "${output}" "${batch}")"
    if (( kv <= 2800000000 )); then
      supported+=("${batch}")
      (( kv > max_kv )) && max_kv="${kv}"
    fi
  done
  batch_csv="$(IFS=,; echo "${supported[*]}")"
  label="b0_${workload}"
  case_dir="${RUN_ROOT}/kt/${label}"
  mkdir "${case_dir}"
  if [[ -z "${batch_csv}" ]]; then
    jq -n '{status:"N/A",reason:"B1/B8 exceed frozen KT KV headroom"}' > "${case_dir}/na.json"
    printf '0\t%s\t%s\t%s\t-\t%s\t%s\tN/A\t-\n' \
      "${workload}" "${prompt}" "${output}" "${label}" "${max_kv}" >> "${RUN_ROOT}/kt/execution.tsv"
    continue
  fi
  wait_gpu_clean || die "GPU is not clean before KT ${workload}"
  since="$(date --iso-8601=seconds)"
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu \
    --format=csv,noheader,nounits -lms 200 > "${case_dir}/nvml.csv" 2> "${case_dir}/nvml.err" &
  MONITOR_PID="$!"
  printf '%q ' taskset -c "${CPUSET}" "${KT_PY}" "${DIRECT_KT}" \
    --model "${MODEL}" --fixture "${FIXTURE_ROOT}/evaluation_${workload}.json" \
    --frequency-plan "${KT_STATS}" --output "${case_dir}/result.json" \
    --workload "${workload}" --prompt-tokens "${prompt}" --output-tokens "${output}" \
    --batch-sizes "${batch_csv}" --gpu-experts 118 --cpu-threads 14 \
    > "${case_dir}/command.txt"
  printf '\n' >> "${case_dir}/command.txt"
  echo "E3 no-HTTP: starting KT ${workload} batches=${batch_csv}"
  setsid timeout --kill-after=30s 3600s taskset -c "${CPUSET}" \
    env PATH="${KT_ROOT}/.venv/bin:${PATH}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 \
      "${KT_PY}" "${DIRECT_KT}" \
        --model "${MODEL}" --fixture "${FIXTURE_ROOT}/evaluation_${workload}.json" \
        --frequency-plan "${KT_STATS}" --output "${case_dir}/result.json" \
        --workload "${workload}" --prompt-tokens "${prompt}" --output-tokens "${output}" \
        --batch-sizes "${batch_csv}" --gpu-experts 118 --cpu-threads 14 \
      > "${case_dir}/run.log" 2>&1 &
  ACTIVE_PID="$!"
  wait "${ACTIVE_PID}"
  rc=$?
  ACTIVE_PID=""
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
  MONITOR_PID=""
  journalctl -k --since "${since}" --no-pager > "${case_dir}/kernel.since.log" 2> "${case_dir}/kernel.since.err" || true
  status=error
  if (( rc == 0 )) && jq -e \
    '.status=="ok" and .configuration.transport=="none"
     and .configuration.http_server_started==false
     and .configuration.mixed_chunk_enabled==false
     and (.cells|all(.single_generate_call==true and .all_requests_submitted_together==true
       and .transport=="none" and .mixed_chunk_enabled==false
       and .successful_requests==.request_count))' \
    "${case_dir}/result.json" >/dev/null 2>&1 \
    && ! rg -q 'Uvicorn running|POST /generate|http://127\.0\.0\.1' "${case_dir}/run.log" \
    && ! rg -qi 'NVRM: Xid|fault_pde|CUDA illegal' "${case_dir}/kernel.since.log"; then
    status=ok
  fi
  echo "${rc}" > "${case_dir}/returncode.txt"
  printf '0\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${workload}" "${prompt}" "${output}" "${batch_csv}" "${label}" "${max_kv}" "${status}" "${rc}" \
    >> "${RUN_ROOT}/kt/execution.tsv"
  wait_gpu_clean || die "GPU did not clean up after KT ${workload}"
done

"${KT_PY}" "${VERIFIER}" --profile qwen3 --batches 1,8 --kt-mode direct "${RUN_ROOT}" \
  > "${RUN_ROOT}/verification.stdout.json"
verify_rc=$?
if (( verify_rc == 0 )); then
  "${KT_PY}" "${PLOTTER}" "${RUN_ROOT}/analysis/best_strategy_heatmap.csv" \
    "${RUN_ROOT}/analysis/best_strategy_heatmap.svg"
fi
wait_gpu_clean || die "GPU is not clean after E3"
echo "RUN_ROOT=${RUN_ROOT}"
exit "${verify_rc}"
