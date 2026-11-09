#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KT_ROOT="${REPO_ROOT}/KTransformers"
KT_PY="${KT_ROOT}/.venv/bin/python"
RUNNER="${SCRIPT_DIR}/direct_batch_e3_ktransformers_runner.py"
CONVERTER="${REPO_ROOT}/experiments/ktransformers_cpu_expert_compare/cold_plan_to_kt_stats.py"
VERIFIER="${SCRIPT_DIR}/verify_e3_ktransformers_c1_c16.py"
MODEL="${MODEL:?Set MODEL to the local Qwen3-30B-A3B-FP8 snapshot directory}"
FIXTURE_ROOT="${SCRIPT_DIR}/inputs/generated/eurosys-pa-v2-20260824/qwen3_30b_a3b_fp8"
PLAN_LOW="${E3_KT_PLAN_LOW:-${REPO_ROOT}/experiments/qwen_decode_six_mechanism_compare/inputs/cold_decode_plan_2g.json}"
PLAN_MID="${E3_KT_PLAN_MID:-${REPO_ROOT}/experiments/qwen_decode_six_mechanism_compare/inputs/cold_decode_plan_8g.json}"
PLAN_HIGH="${E3_KT_PLAN_HIGH:-${REPO_ROOT}/experiments/vllm_uvm_e10_multilevel_pressure/inputs/cold89.json}"
COLD_LOW="${E3_KT_COLD_LOW:-10}"
COLD_MID="${E3_KT_COLD_MID:-38}"
COLD_HIGH="${E3_KT_COLD_HIGH:-89}"
TIER_LOW="cold${COLD_LOW}"
TIER_MID="cold${COLD_MID}"
TIER_HIGH="cold${COLD_HIGH}"
D_REFERENCE="${D_REFERENCE:-${SCRIPT_DIR}/artifacts/e3-c1-c16-full-dual-20260901-reproduce-all-attempt-02}"
RUN_ROOT="${1:-${SCRIPT_DIR}/artifacts/e3-ktransformers-c1-c16-$(date -u +%Y%m%dT%H%M%SZ)}"
CPU_THREADS="${CPU_THREADS:-14}"
# Keep the scheduler/runtime free to use all physical cores while limiting
# only the KTransformers CPU expert pool through kt_cpuinfer/OMP/MKL.
CPUSET="${CPUSET:-0-13}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
ATTENTION_BACKEND_RAG_B16="${ATTENTION_BACKEND_RAG_B16:-${ATTENTION_BACKEND}}"
DECODE_ATTENTION_BACKEND="${DECODE_ATTENTION_BACKEND:-}"
GPU_PREFILL_TOKEN_THRESHOLD="${GPU_PREFILL_TOKEN_THRESHOLD:-0}"
GPU_PREFILL_TOKEN_THRESHOLD_MEDIUM_B16="${GPU_PREFILL_TOKEN_THRESHOLD_MEDIUM_B16:-${GPU_PREFILL_TOKEN_THRESHOLD}}"
GPU_PREFILL_TOKEN_THRESHOLD_RAG_B16="${GPU_PREFILL_TOKEN_THRESHOLD_RAG_B16:-${GPU_PREFILL_TOKEN_THRESHOLD}}"
DISABLE_OVERLAP_SCHEDULE="${DISABLE_OVERLAP_SCHEDULE:-0}"
CELL_FILTER="${E3_KT_CELL_FILTER:-}"
SKIP_VERIFY="${E3_KT_SKIP_VERIFY:-0}"
CELL_PREFILL_THRESHOLDS_JSON="${E3_KT_CELL_PREFILL_THRESHOLDS_JSON:-}"
CELL_ATTENTION_BACKENDS_JSON="${E3_KT_CELL_ATTENTION_BACKENDS_JSON:-}"
CELL_DECODE_BACKENDS_JSON="${E3_KT_CELL_DECODE_BACKENDS_JSON:-}"
[[ -n "${CELL_PREFILL_THRESHOLDS_JSON}" ]] || CELL_PREFILL_THRESHOLDS_JSON='{}'
[[ -n "${CELL_ATTENTION_BACKENDS_JSON}" ]] || CELL_ATTENTION_BACKENDS_JSON='{}'
[[ -n "${CELL_DECODE_BACKENDS_JSON}" ]] || CELL_DECODE_BACKENDS_JSON='{}'
MONITOR_PID=""

die() { echo "E3 KTransformers C1/C16: $*" >&2; exit 1; }

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
trap cleanup EXIT
trap 'exit 130' INT TERM

wait_gpu_clean() {
  local second used
  for second in $(seq 1 90); do
    used="$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{sum += $1} END {print sum + 0}')"
    [[ "${used}" == 0 ]] && return 0
    sleep 1
  done
  return 1
}

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
[[ "${ATTENTION_BACKEND}" == triton || "${ATTENTION_BACKEND}" == flashinfer ]] \
  || die "ATTENTION_BACKEND must be triton or flashinfer"
[[ "${ATTENTION_BACKEND_RAG_B16}" == triton \
   || "${ATTENTION_BACKEND_RAG_B16}" == flashinfer ]] \
  || die "ATTENTION_BACKEND_RAG_B16 must be triton or flashinfer"
[[ -z "${DECODE_ATTENTION_BACKEND}" || "${DECODE_ATTENTION_BACKEND}" == triton \
   || "${DECODE_ATTENTION_BACKEND}" == flashinfer ]] \
  || die "DECODE_ATTENTION_BACKEND must be empty, triton, or flashinfer"
[[ "${DISABLE_OVERLAP_SCHEDULE}" == 0 || "${DISABLE_OVERLAP_SCHEDULE}" == 1 ]] \
  || die "DISABLE_OVERLAP_SCHEDULE must be 0 or 1"
for cold in "${COLD_LOW}" "${COLD_MID}" "${COLD_HIGH}"; do
  [[ "${cold}" =~ ^[0-9]+$ && "${cold}" -ge 1 && "${cold}" -le 127 ]] \
    || die "KTransformers cold tier must be in [1, 127], got ${cold}"
done
for mapping in "${CELL_PREFILL_THRESHOLDS_JSON}" \
  "${CELL_ATTENTION_BACKENDS_JSON}" "${CELL_DECODE_BACKENDS_JSON}"; do
  jq -e 'type == "object"' <<<"${mapping}" >/dev/null \
    || die "per-cell override must be a JSON object"
done
[[ "${GPU_PREFILL_TOKEN_THRESHOLD}" =~ ^[0-9]+$ \
   && "${GPU_PREFILL_TOKEN_THRESHOLD_MEDIUM_B16}" =~ ^[0-9]+$ \
   && "${GPU_PREFILL_TOKEN_THRESHOLD_RAG_B16}" =~ ^[0-9]+$ ]] \
  || die "GPU prefill thresholds must be non-negative integers"
[[ ! -e "${RUN_ROOT}" ]] || die "output already exists: ${RUN_ROOT}"
for path in "${KT_PY}" "${RUNNER}" "${CONVERTER}" "${VERIFIER}" \
  "${MODEL}/config.json" "${PLAN_LOW}" "${PLAN_MID}" "${PLAN_HIGH}"; do
  [[ -f "${path}" ]] || die "missing file: ${path}"
done
if [[ "${SKIP_VERIFY}" != 1 ]]; then
  [[ -f "${D_REFERENCE}/ours/short_dialogue_b1_D/custom/b0_short_dialogue_b1_D/result.json" ]] \
    || die "Set D_REFERENCE to a completed AdaptMoE run for cross-system comparison"
fi
for workload in short_dialogue ordinary_chat medium_context rag_summary long_generation; do
  [[ -f "${FIXTURE_ROOT}/evaluation_${workload}.json" ]] \
    || die "missing fixture: evaluation_${workload}.json"
done
wait_gpu_clean || die "GPU has an active compute process"

mkdir -p "${RUN_ROOT}/cases" "${RUN_ROOT}/provenance"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
PLAN_ROOT="${RUN_ROOT}/provenance/kt_plans"
mkdir -p "${PLAN_ROOT}"
"${KT_PY}" "${CONVERTER}" "${PLAN_LOW}" "${PLAN_MID}" "${PLAN_HIGH}" \
  --output-dir "${PLAN_ROOT}" >"${RUN_ROOT}/provenance/prepare_plans.stdout.json"
PLAN_LOW_PT="${PLAN_ROOT}/$(basename "${PLAN_LOW}" .json)_logical_count.pt"
PLAN_MID_PT="${PLAN_ROOT}/$(basename "${PLAN_MID}" .json)_logical_count.pt"
PLAN_HIGH_PT="${PLAN_ROOT}/$(basename "${PLAN_HIGH}" .json)_logical_count.pt"

git -C "${REPO_ROOT}" rev-parse HEAD >"${RUN_ROOT}/provenance/repo.commit"
git -C "${KT_ROOT}" rev-parse HEAD >"${RUN_ROOT}/provenance/ktransformers.commit"
git -C "${KT_ROOT}/third_party/sglang" rev-parse HEAD >"${RUN_ROOT}/provenance/sglang.commit"
git -C "${REPO_ROOT}" status --short >"${RUN_ROOT}/provenance/repo.status.txt"
sha256sum "${RUNNER}" "${VERIFIER}" "${MODEL}/config.json" \
  "${FIXTURE_ROOT}"/evaluation_*.json "${PLAN_LOW}" "${PLAN_MID}" "${PLAN_HIGH}" \
  >"${RUN_ROOT}/provenance/inputs.sha256"
nvidia-smi -q >"${RUN_ROOT}/provenance/nvidia-smi-q.txt"
lscpu >"${RUN_ROOT}/provenance/lscpu.txt"
uname -a >"${RUN_ROOT}/provenance/uname.txt"

jq -n \
  --arg model "$(realpath "${MODEL}")" \
  --arg fixture_root "$(realpath "${FIXTURE_ROOT}")" \
  --arg d_reference "$(realpath "${D_REFERENCE}")" \
  --arg backend "${ATTENTION_BACKEND}" \
  --arg rag_b16_backend "${ATTENTION_BACKEND_RAG_B16}" \
  --arg decode_backend "${DECODE_ATTENTION_BACKEND:-${ATTENTION_BACKEND}}" \
  --arg tier_low "${TIER_LOW}" --arg tier_mid "${TIER_MID}" \
  --arg tier_high "${TIER_HIGH}" \
  --arg cpuset "${CPUSET}" \
  --argjson cpu_threads "${CPU_THREADS}" \
  --argjson gpu_prefill_threshold "${GPU_PREFILL_TOKEN_THRESHOLD}" \
  --argjson medium_b16_prefill_threshold "${GPU_PREFILL_TOKEN_THRESHOLD_MEDIUM_B16}" \
  --argjson rag_b16_prefill_threshold "${GPU_PREFILL_TOKEN_THRESHOLD_RAG_B16}" \
  --argjson overlap_enabled "$((1 - DISABLE_OVERLAP_SCHEDULE))" \
  --argjson cell_prefill_thresholds "${CELL_PREFILL_THRESHOLDS_JSON}" \
  --argjson cell_attention_backends "${CELL_ATTENTION_BACKENDS_JSON}" \
  --argjson cell_decode_backends "${CELL_DECODE_BACKENDS_JSON}" \
  '{schema:"eurosys_e3_ktransformers_c1_c16_config_v1",
    model:$model,fixture_root:$fixture_root,d_reference:$d_reference,
    batches:[1,16],attention_backend:$backend,
    attention_backend_overrides:{"rag_summary/B16":$rag_b16_backend},
    decode_attention_backend:($decode_backend | select(length > 0) // $backend),
    cpu_threads:$cpu_threads,process_cpuset:$cpuset,
    gpu_prefill_token_threshold:$gpu_prefill_threshold,
    gpu_prefill_token_threshold_overrides:{
      "medium_context/B16":$medium_b16_prefill_threshold,
      "rag_summary/B16":$rag_b16_prefill_threshold},
    cell_gpu_prefill_token_threshold_overrides:$cell_prefill_thresholds,
    cell_attention_backend_overrides:$cell_attention_backends,
    cell_decode_attention_backend_overrides:$cell_decode_backends,
    overlap_schedule_enabled:($overlap_enabled == 1),
    repeats:1,transport:"none",
    cuda_graph:true,mixed_chunk:false,chunked_prefill:false,
    placement_by_cell:{
      "short_dialogue/B1":$tier_low,"short_dialogue/B16":$tier_low,
      "ordinary_chat/B1":$tier_low,"ordinary_chat/B16":$tier_mid,
      "medium_context/B1":$tier_low,"medium_context/B16":$tier_mid,
      "rag_summary/B1":$tier_mid,"rag_summary/B16":$tier_high,
      "long_generation/B1":$tier_low,"long_generation/B16":$tier_mid},
    comparison_scope:"complete KTransformers CUDA Graph framework baseline; not a pure page-table or remote-access ablation"}' \
  >"${RUN_ROOT}/configuration.json"

printf 'workload\tprompt\toutput\tbatch\ttier\tgpu_experts\tcase_dir\tstatus\treturncode\n' \
  >"${RUN_ROOT}/execution.tsv"

run_cell() {
  local workload="$1" prompt="$2" output="$3" batch="$4" tier="$5"
  local gpu_experts plan case_dir label rc status start_epoch prefill_threshold
  local cell_attention cell_decode_attention coordinate
  case "${tier}" in
    "${TIER_LOW}") gpu_experts=$((128 - COLD_LOW)); plan="${PLAN_LOW_PT}" ;;
    "${TIER_MID}") gpu_experts=$((128 - COLD_MID)); plan="${PLAN_MID_PT}" ;;
    "${TIER_HIGH}") gpu_experts=$((128 - COLD_HIGH)); plan="${PLAN_HIGH_PT}" ;;
    *) die "unknown tier: ${tier}" ;;
  esac
  label="${workload}_b${batch}_${tier}"
  coordinate="${workload}/B${batch}"
  prefill_threshold="${GPU_PREFILL_TOKEN_THRESHOLD}"
  cell_attention="${ATTENTION_BACKEND}"
  cell_decode_attention="${DECODE_ATTENTION_BACKEND}"
  if [[ "${workload}" == medium_context && "${batch}" == 16 ]]; then
    prefill_threshold="${GPU_PREFILL_TOKEN_THRESHOLD_MEDIUM_B16}"
  fi
  if [[ "${workload}" == rag_summary && "${batch}" == 16 ]]; then
    cell_attention="${ATTENTION_BACKEND_RAG_B16}"
    prefill_threshold="${GPU_PREFILL_TOKEN_THRESHOLD_RAG_B16}"
  fi
  if jq -e --arg coordinate "${coordinate}" 'has($coordinate)' \
    <<<"${CELL_PREFILL_THRESHOLDS_JSON}" >/dev/null; then
    prefill_threshold="$(jq -er --arg coordinate "${coordinate}" \
      '.[$coordinate]' <<<"${CELL_PREFILL_THRESHOLDS_JSON}")"
  fi
  if jq -e --arg coordinate "${coordinate}" 'has($coordinate)' \
    <<<"${CELL_ATTENTION_BACKENDS_JSON}" >/dev/null; then
    cell_attention="$(jq -er --arg coordinate "${coordinate}" \
      '.[$coordinate]' <<<"${CELL_ATTENTION_BACKENDS_JSON}")"
  fi
  if jq -e --arg coordinate "${coordinate}" 'has($coordinate)' \
    <<<"${CELL_DECODE_BACKENDS_JSON}" >/dev/null; then
    cell_decode_attention="$(jq -er --arg coordinate "${coordinate}" \
      '.[$coordinate]' <<<"${CELL_DECODE_BACKENDS_JSON}")"
  fi
  [[ "${prefill_threshold}" =~ ^[0-9]+$ ]] \
    || die "invalid Prefill threshold for ${coordinate}: ${prefill_threshold}"
  [[ "${cell_attention}" == triton || "${cell_attention}" == flashinfer ]] \
    || die "invalid attention backend for ${coordinate}: ${cell_attention}"
  [[ -z "${cell_decode_attention}" || "${cell_decode_attention}" == triton \
     || "${cell_decode_attention}" == flashinfer ]] \
    || die "invalid Decode attention backend for ${coordinate}: ${cell_decode_attention}"
  case_dir="${RUN_ROOT}/cases/${label}"
  mkdir "${case_dir}"
  wait_gpu_clean || die "GPU is not clean before ${label}"
  start_epoch="$(date +%s)"
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu \
    --format=csv,noheader,nounits -lms 200 >"${case_dir}/nvml.csv" 2>"${case_dir}/nvml.err" &
  MONITOR_PID="$!"
  local -a overlap_args=()
  local -a decode_backend_args=()
  if [[ "${DISABLE_OVERLAP_SCHEDULE}" == 1 ]]; then
    overlap_args+=(--disable-overlap-schedule)
  fi
  if [[ -n "${cell_decode_attention}" ]]; then
    decode_backend_args+=(--decode-attention-backend "${cell_decode_attention}")
  fi
  printf '%q ' env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}" SGLANG_E3_SERIALIZE_SCHEDULER_TIME_STATS=1 \
    taskset -c "${CPUSET}" "${KT_PY}" "${RUNNER}" \
    --model "${MODEL}" --fixture "${FIXTURE_ROOT}/evaluation_${workload}.json" \
    --frequency-plan "${plan}" --output "${case_dir}/result.json" \
    --workload "${workload}" --prompt-tokens "${prompt}" --output-tokens "${output}" \
    --batch-sizes "${batch}" --gpu-experts "${gpu_experts}" --cpu-threads "${CPU_THREADS}" \
    --gpu-prefill-token-threshold "${prefill_threshold}" \
    --attention-backend "${cell_attention}" --seed 0 --warmup-output-tokens 8 \
    "${decode_backend_args[@]}" \
    --warmup-decode-steps 5 --watchdog-timeout-seconds 7200 --enable-cuda-graph \
    "${overlap_args[@]}" \
    >"${case_dir}/command.txt"
  printf '\n' >>"${case_dir}/command.txt"
  set +e
  env PATH="${KT_ROOT}/.venv/bin:${PATH}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}" \
    SGLANG_E3_SERIALIZE_SCHEDULER_TIME_STATS=1 \
    timeout --kill-after=60s 7200s taskset -c "${CPUSET}" "${KT_PY}" "${RUNNER}" \
      --model "${MODEL}" --fixture "${FIXTURE_ROOT}/evaluation_${workload}.json" \
      --frequency-plan "${plan}" --output "${case_dir}/result.json" \
      --workload "${workload}" --prompt-tokens "${prompt}" --output-tokens "${output}" \
      --batch-sizes "${batch}" --gpu-experts "${gpu_experts}" --cpu-threads "${CPU_THREADS}" \
      --gpu-prefill-token-threshold "${prefill_threshold}" \
      --attention-backend "${cell_attention}" --seed 0 --warmup-output-tokens 8 \
      "${decode_backend_args[@]}" \
      --warmup-decode-steps 5 --watchdog-timeout-seconds 7200 --enable-cuda-graph \
      "${overlap_args[@]}" \
      >"${case_dir}/run.log" 2>&1
  rc=$?
  set -e
  stop_monitor
  journalctl -k --since "@${start_epoch}" --no-pager \
    >"${case_dir}/kernel.since.log" 2>"${case_dir}/kernel.since.err" || true
  echo "${rc}" >"${case_dir}/returncode.txt"
  status=error
  if (( rc == 0 )) && jq -e \
    --argjson batch "${batch}" --argjson output "${output}" \
    '.status == "ok" and (.cells|length) == 1 and
     .cells[0].batch_size == $batch and
     .cells[0].successful_requests == $batch and
     (.cells[0].output_token_ids|length) == $batch and
     (.cells[0].output_token_ids|all(length == $output)) and
     (.cells[0].prefill_step_ms > 0) and
     (.cells[0].measured_decode_step_ms.count == ($output - 7))' \
    "${case_dir}/result.json" >/dev/null; then
    status=ok
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${workload}" "${prompt}" "${output}" "${batch}" "${tier}" \
    "${gpu_experts}" "${label}" "${status}" "${rc}" >>"${RUN_ROOT}/execution.tsv"
  wait_gpu_clean || echo dirty >"${case_dir}/gpu_cleanup_error.txt"
}

run_selected_cell() {
  local workload="$1" batch="$4" coordinate
  coordinate="${workload}_b${batch}"
  if [[ -n "${CELL_FILTER}" ]]; then
    case ",${CELL_FILTER}," in
      *",${coordinate},"*) ;;
      *) return 0 ;;
    esac
  fi
  run_cell "$@"
}

# Keep each cell in an independent cold process and preserve the latest D
# placement tier for that exact request coordinate.
run_selected_cell short_dialogue 128 32 1 "${TIER_LOW}"
run_selected_cell short_dialogue 128 32 16 "${TIER_LOW}"
run_selected_cell ordinary_chat 512 128 1 "${TIER_LOW}"
run_selected_cell ordinary_chat 512 128 16 "${TIER_MID}"
run_selected_cell medium_context 2048 128 1 "${TIER_LOW}"
run_selected_cell medium_context 2048 128 16 "${TIER_MID}"
run_selected_cell rag_summary 8192 128 1 "${TIER_MID}"
run_selected_cell rag_summary 8192 128 16 "${TIER_HIGH}"
run_selected_cell long_generation 512 512 1 "${TIER_LOW}"
run_selected_cell long_generation 512 512 16 "${TIER_MID}"

if [[ "${SKIP_VERIFY}" == 1 ]]; then
  if [[ -z "${CELL_FILTER}" ]]; then
    "${KT_PY}" "${VERIFIER}" --root "${RUN_ROOT}" --standalone \
      --output "${RUN_ROOT}/verification.json" || exit 1
  else
    echo "cross-system verification deferred for filtered continuation"
  fi
else
  "${KT_PY}" "${VERIFIER}" --root "${RUN_ROOT}" --d-reference "${D_REFERENCE}" \
    --output "${RUN_ROOT}/verification.json"
fi
echo "RUN_ROOT=${RUN_ROOT}"
