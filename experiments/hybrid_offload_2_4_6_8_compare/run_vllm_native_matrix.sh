#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
PYTHON="${VLLM_ROOT}/.venv/bin/python"
RUNNER="${REPO_ROOT}/experiments/vllm_p3_pinned_uva_phase_compare/benchmark_phase_steps.py"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7}"
FIXTURE="${FIXTURE:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/deepseek_v2_lite_bf16/fixtures/evaluation_fixture.json}"
TIERS="${TIERS:-2,4,6,8}"
REPEATS="${REPEATS:-3}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/vllm_native_$(date -u +%Y%m%dT%H%M%SZ)}"

die() { echo "error: $*" >&2; exit 1; }
[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
[[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]] || die "REPEATS must be positive"
[[ -x "${PYTHON}" ]] || die "missing Python: ${PYTHON}"
[[ -f "${RUNNER}" ]] || die "missing runner: ${RUNNER}"
[[ -f "${MODEL}/config.json" ]] || die "missing model: ${MODEL}"
[[ -f "${FIXTURE}" ]] || die "missing fixture: ${FIXTURE}"
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"

gpu_clean() {
  local used pids
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  [[ "${used}" == 0 && -z "${pids//[[:space:]]/}" ]]
}

wait_gpu_clean() {
  local attempt
  for ((attempt = 0; attempt < 240; ++attempt)); do
    gpu_clean && return 0
    sleep 1
  done
  return 1
}

printf 'tier_gib\tresult\tmanifest\treturncode\n' >"${RUN_ROOT}/execution.tsv"
IFS=',' read -r -a tiers <<<"${TIERS}"
for tier in "${tiers[@]}"; do
  [[ "${tier}" =~ ^(2|4|6|8)$ ]] || die "unsupported tier: ${tier}"
  case_dir="${RUN_ROOT}/${tier}g"
  mkdir "${case_dir}"
  manifest="${case_dir}/manifest.jsonl"
  : >"${manifest}"
  wait_gpu_clean || die "GPU is not clean before ${tier} GiB"
  set +e
  env -u VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE \
    -u VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS \
    -u VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE \
    -u VLLM_UVM_ORIGINAL_VA_PHASE_MODE \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_CPU_OFFLOAD_BACKEND=uva \
    VLLM_CPU_OFFLOAD_MANIFEST_FILE="${manifest}" \
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 \
    VLLM_CPU_OFFLOAD_COPY_TRACE=0 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 CUDA_VISIBLE_DEVICES=0 \
    timeout --kill-after=30s 1800s taskset -c 0-13 \
    "${PYTHON}" "${RUNNER}" \
      --model "${MODEL}" --fixture "${FIXTURE}" \
      --output "${case_dir}/result.json" \
      --profile-label "vllm_native_uva_${tier}g" \
      --mechanism-class vllm_native_cpu_offload_pinned_uva \
      --batch-sizes 1 --conditioning-repeats 1 \
      --measurement-repeats "${REPEATS}" \
      --warmup-decode-steps 2 --measured-decode-steps 8 \
      --kv-cache-memory-bytes 167772160 --cpu-offload-gb "${tier}" \
      --max-model-len 544 --max-num-seqs 1 \
      --max-num-batched-tokens 1024 --gpu-memory-utilization 0.97 --seed 0 \
      >"${case_dir}/run.log" 2>&1
  rc=$?
  set -e
  echo "${rc}" >"${case_dir}/returncode.txt"
  wait_gpu_clean || die "GPU did not clean up after ${tier} GiB"
  printf '%s\t%s\t%s\t%s\n' "${tier}" "${case_dir}/result.json" \
    "${manifest}" "${rc}" >>"${RUN_ROOT}/execution.tsv"
  if ((rc != 0)); then
    tail -n 160 "${case_dir}/run.log" >&2 || true
    die "vLLM native case failed: ${tier} GiB"
  fi
  jq -e '.status == "ok"' "${case_dir}/result.json" >/dev/null \
    || die "vLLM native result is not ok: ${tier} GiB"
  jq -s -e --argjson budget "$((tier * 1073741824))" '
    [ .[] | select(.event == "cpu_offload_parameter") ] as $p
    | ($p | map(.bytes) | add) as $bytes
    | ($p | max_by(.offload_order) | .bytes) as $last
    | ($p | length) > 0
      and ($p | all(.offload_backend == "uva" and .uva_offloading == true
                    and .cpu_data_pinned == true))
      and ($bytes >= $budget)
      and (($bytes - $last) < $budget)
  ' "${manifest}" >/dev/null || die "native manifest contract failed: ${tier} GiB"
done

echo "RUN_ROOT=${RUN_ROOT}"
