#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KT_RUNNER="${REPO_ROOT}/experiments/ktransformers_cpu_expert_compare/run_ktransformers_compare.sh"
KT_PY="${REPO_ROOT}/KTransformers/.venv/bin/python"
VLLM_PY="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
PREPARE_FIXTURES="${REPO_ROOT}/experiments/custom_cold_qwen_sharegpt_concurrency_snapshot/prepare_sharegpt_fixtures.py"
REFERENCE_SNAPSHOT="${REPO_ROOT}/experiments/native_uva_qwen_sharegpt_concurrency_snapshot/snapshot_20260812.json"
PLAN_BUILDER="${SCRIPT_DIR}/build_layer_first_plan.py"
VLLM_RUNNER="${SCRIPT_DIR}/run_vllm_native_uva_case.sh"
SUMMARIZER="${SCRIPT_DIR}/summarize_pilot.py"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
DATASET="${DATASET:-${REPO_ROOT}/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RUN_ROOT="${RUN_ROOT:-/tmp/ktransformers_vs_vllm_native_offload_6g_$(date +%Y%m%d_%H%M%S)}"
OFFLOAD_GIB="${OFFLOAD_GIB:-6.0}"
COLD_EXPERTS="${COLD_EXPERTS:-1408}"
CONCURRENCIES_ASC="${CONCURRENCIES_ASC:-1,4,8}"
CONCURRENCIES_DESC="${CONCURRENCIES_DESC:-8,4,1}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-64}"
WARM_CONCURRENCY="${WARM_CONCURRENCY:-8}"
WARM_OUTPUT_TOKENS="${WARM_OUTPUT_TOKENS:-8}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-1288490189}"
KV_TOKENS="${KV_TOKENS:-13104}"
CPUSET="${CPUSET:-0-13}"

FIXTURE_DIR="${RUN_ROOT}/fixtures"
FIXTURE="${FIXTURE_DIR}/evaluation_fixture.json"
PLAN_DIR="${RUN_ROOT}/plans"
PLAN="${PLAN_DIR}/layer_first_cold_${COLD_EXPERTS}.json"

die() {
  echo "error: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "missing file: $1"
}

prepare_assets() {
  mkdir -p "${FIXTURE_DIR}" "${PLAN_DIR}"
  if [[ ! -s "${FIXTURE}" ]]; then
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      "${VLLM_PY}" "${PREPARE_FIXTURES}" \
        --dataset "${DATASET}" \
        --model "${MODEL}" \
        --reference-snapshot "${REFERENCE_SNAPSHOT}" \
        --output-dir "${FIXTURE_DIR}" \
        --request-count 32 \
        --prompt-min-tokens 300 \
        --prompt-max-tokens 320 \
        --output-tokens "${OUTPUT_TOKENS}" \
        > "${RUN_ROOT}/prepare_fixtures.log"
  fi
  if [[ ! -s "${PLAN}" ]]; then
    "${KT_PY}" "${PLAN_BUILDER}" \
      --cold-experts "${COLD_EXPERTS}" \
      --output "${PLAN}" \
      --label "vllm_native_uva_${OFFLOAD_GIB}gib_total_byte_match" \
      > "${RUN_ROOT}/build_plan.log"
  fi
  sha256sum "${FIXTURE}" "${PLAN}" > "${RUN_ROOT}/asset_sha256.txt"
}

run_vllm() {
  local label="$1"
  local concurrencies="$2"
  RUN_ROOT="${RUN_ROOT}/${label}" \
  FIXTURE="${FIXTURE}" \
  MODEL="${MODEL}" \
  OFFLOAD_GIB="${OFFLOAD_GIB}" \
  CONCURRENCIES="${concurrencies}" \
  OUTPUT_TOKENS="${OUTPUT_TOKENS}" \
  WARM_CONCURRENCY="${WARM_CONCURRENCY}" \
  WARM_OUTPUT_TOKENS="${WARM_OUTPUT_TOKENS}" \
  KV_CACHE_BYTES="${KV_CACHE_BYTES}" \
  CPUSET="${CPUSET}" \
    bash "${VLLM_RUNNER}"
}

run_kt() {
  local label="$1"
  local concurrencies="$2"
  RUN_ROOT="${RUN_ROOT}/${label}" \
  FIXTURE="${FIXTURE}" \
  MODEL="${MODEL}" \
  CONCURRENCIES="${concurrencies}" \
  OUTPUT_TOKENS="${OUTPUT_TOKENS}" \
  WARM_CONCURRENCY="${WARM_CONCURRENCY}" \
  WARM_OUTPUT_TOKENS="${WARM_OUTPUT_TOKENS}" \
  KV_TOKENS="${KV_TOKENS}" \
  KT_CPU_THREADS=14 \
  CPUSET="${CPUSET}" \
    bash "${KT_RUNNER}" kt-case "${PLAN}"
}

require_file "${KT_RUNNER}"
require_file "${PREPARE_FIXTURES}"
require_file "${REFERENCE_SNAPSHOT}"
require_file "${PLAN_BUILDER}"
require_file "${VLLM_RUNNER}"
require_file "${SUMMARIZER}"

if [[ -e "${RUN_ROOT}/vllm_a1" || -e "${RUN_ROOT}/kt_b1" \
   || -e "${RUN_ROOT}/kt_b2" || -e "${RUN_ROOT}/vllm_a2" ]]; then
  die "RUN_ROOT already contains pilot cases: ${RUN_ROOT}"
fi

mkdir -p "${RUN_ROOT}"
bash "${KT_RUNNER}" preflight
prepare_assets

printf '%s\n' \
  'design=A(asc)-B(asc)-B(desc)-A(desc)' \
  'A=vllm_native_cpu_offload_pinned_uva' \
  'B=ktransformers_cpu_expert' \
  "offload_gib=${OFFLOAD_GIB}" \
  "kt_cold_experts=${COLD_EXPERTS}" \
  "concurrencies_asc=${CONCURRENCIES_ASC}" \
  "concurrencies_desc=${CONCURRENCIES_DESC}" \
  "output_tokens=${OUTPUT_TOKENS}" \
  "kv_cache_bytes_vllm=${KV_CACHE_BYTES}" \
  "kv_tokens_kt=${KV_TOKENS}" \
  > "${RUN_ROOT}/pilot_config.txt"

run_vllm vllm_a1 "${CONCURRENCIES_ASC}"
run_kt kt_b1 "${CONCURRENCIES_ASC}"
run_kt kt_b2 "${CONCURRENCIES_DESC}"
run_vllm vllm_a2 "${CONCURRENCIES_DESC}"

"${VLLM_PY}" "${SUMMARIZER}" "${RUN_ROOT}" \
  --output "${RUN_ROOT}/snapshot.json" \
  | tee "${RUN_ROOT}/summarize.stdout.json"

echo "RUN_ROOT=${RUN_ROOT}"
echo "SNAPSHOT=${RUN_ROOT}/snapshot.json"
