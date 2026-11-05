#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MOEI_ROOT="${REPO_ROOT}/MoE-Infinity"
PYTHON="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
RUNNER="${REPO_ROOT}/experiments/broad_offload_compare/qualify_moe_infinity.py"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7}"
OFFLOAD_DIR="${OFFLOAD_DIR:-/home/ubuntu/moe_infinity_offload/deepseek_v2_lite_chat_85864749cd611b4353ce1decdb286193298f64c7}"
EVAL_FIXTURE="${EVAL_FIXTURE:-${REPO_ROOT}/experiments/broad_offload_compare/fixtures/moei_deepseek_v2_lite_evaluation_prompt512_v1.json}"
WARMUP_FIXTURE="${WARMUP_FIXTURE:-${REPO_ROOT}/experiments/broad_offload_compare/fixtures/moei_deepseek_v2_lite_calibration_prompt512_v1.json}"
TIERS="${TIERS:-2g,4g,6g,8g}"
REPEATS="${REPEATS:-3}"
ALL_GPU_REFERENCE_MIB="${ALL_GPU_REFERENCE_MIB:-31295}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/moe_infinity_$(date -u +%Y%m%dT%H%M%SZ)}"
MONITOR_PID=""

die() { echo "error: $*" >&2; exit 1; }

ratio_for_tier() {
  case "$1" in
    2g) echo 0.75 ;;
    4g) echo 0.69 ;;
    6g) echo 0.63 ;;
    8g) echo 0.57 ;;
    *) return 1 ;;
  esac
}

gpu_clean() {
  local used pids
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ "${used}" == 0 && -z "${pids//[[:space:]]/}" ]]
}

wait_gpu_clean() {
  local attempt
  for ((attempt = 0; attempt < 300; ++attempt)); do
    gpu_clean && return 0
    sleep 1
  done
  return 1
}

stop_monitor() {
  [[ -n "${MONITOR_PID}" ]] || return 0
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
  MONITOR_PID=""
}
trap 'stop_monitor' EXIT INT TERM

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
[[ "${REPEATS}" =~ ^[0-9]+$ ]] && ((REPEATS >= 2)) \
  || die "REPEATS must be at least 2"
[[ -x "${PYTHON}" && -f "${RUNNER}" ]] || die "missing MoE-Infinity runner"
[[ -f "${MODEL}/config.json" ]] || die "missing model"
[[ -d "${OFFLOAD_DIR}" && -f "${OFFLOAD_DIR}/archer_index" ]] \
  || die "missing prepared MoE-Infinity store"
[[ -f "${EVAL_FIXTURE}" && -f "${WARMUP_FIXTURE}" ]] || die "missing fixtures"
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"

printf 'tier\tratio\tpeak_mib\testimated_saved_gib\tresult\n' \
  >"${RUN_ROOT}/execution.tsv"
IFS=',' read -r -a tiers <<<"${TIERS}"
for tier in "${tiers[@]}"; do
  ratio="$(ratio_for_tier "${tier}")" || die "unsupported tier: ${tier}"
  case_dir="${RUN_ROOT}/${tier}"
  mkdir "${case_dir}"
  wait_gpu_clean || die "GPU is not clean before ${tier}"
  nvidia-smi \
    --query-gpu=timestamp,index,memory.used,utilization.gpu,utilization.memory,power.draw \
    --format=csv,noheader,nounits -lms 200 \
    >"${case_dir}/nvml.csv" 2>"${case_dir}/nvml.err" &
  MONITOR_PID="$!"
  set +e
  env PYTHONPATH="${MOEI_ROOT}:${PYTHONPATH:-}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
    timeout --signal=INT --kill-after=30s 2400s taskset -c 0-13 \
    "${PYTHON}" "${RUNNER}" \
      --model "${MODEL}" --offload-dir "${OFFLOAD_DIR}" \
      --output "${case_dir}/result.json" --mode demand-only \
      --prompt-length 512 --output-tokens 12 --warmup-requests 1 \
      --repetitions "${REPEATS}" --device-memory-ratio "${ratio}" \
      --num-threads 8 --event-sample-limit 0 \
      --prompt-token-ids-file "${EVAL_FIXTURE}" \
      --warmup-prompt-token-ids-file "${WARMUP_FIXTURE}" \
      >"${case_dir}/stdout.log" 2>"${case_dir}/stderr.log"
  rc=$?
  set -e
  stop_monitor
  echo "${rc}" >"${case_dir}/returncode.txt"
  wait_gpu_clean || die "GPU did not clean up after ${tier}"
  if ((rc != 0)); then
    tail -n 160 "${case_dir}/stderr.log" >&2 || true
    tail -n 160 "${case_dir}/stdout.log" >&2 || true
    die "MoE-Infinity case failed: ${tier}"
  fi
  jq -e '.status == "ok" and .qualification.status == "ok" and
         .qualification.output_hash_stable == true and
         .qualification.warmup_output_hash_stable == true and
         .qualification.cross_phase_output_match_required == false' \
    "${case_dir}/result.json" >/dev/null \
    || die "MoE-Infinity result contract failed: ${tier}"
  peak_mib="$(awk -F, 'BEGIN{m=0} {gsub(/ /,"",$3); if($3+0>m)m=$3+0} END{print m}' \
    "${case_dir}/nvml.csv")"
  saved_gib="$(awk -v ref="${ALL_GPU_REFERENCE_MIB}" -v peak="${peak_mib}" \
    'BEGIN{printf "%.6f", (ref-peak)/1024}')"
  jq -n --arg tier "${tier}" --argjson ratio "${ratio}" \
    --argjson all_gpu_reference_mib "${ALL_GPU_REFERENCE_MIB}" \
    --argjson peak_mib "${peak_mib}" --argjson estimated_saved_gib "${saved_gib}" \
    '{tier:$tier,device_memory_ratio:$ratio,
      all_gpu_reference_mib:$all_gpu_reference_mib,peak_mib:$peak_mib,
      estimated_saved_gib:$estimated_saved_gib,
      budget_semantics:"observed peak reduction versus common all-GPU reference; dynamic cache, not a fixed expert byte set"}' \
    >"${case_dir}/memory_budget.json"
  printf '%s\t%s\t%s\t%s\t%s\n' "${tier}" "${ratio}" "${peak_mib}" \
    "${saved_gib}" "${case_dir}/result.json" >>"${RUN_ROOT}/execution.tsv"
done

echo "RUN_ROOT=${RUN_ROOT}"
