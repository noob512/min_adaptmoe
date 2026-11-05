#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
KT_PY="${REPO_ROOT}/KTransformers/.venv/bin/python"
BENCHMARK="${REPO_ROOT}/experiments/broad_offload_compare/benchmark_custom_offload.py"
PLAN_BUILDER="${REPO_ROOT}/experiments/hybrid_offload_2_4_6_8_compare/prepare_cold_plans.py"
KT_BUILDER="${REPO_ROOT}/experiments/hybrid_offload_2_4_6_8_compare/prepare_kt_frequency_plans.py"

PROTOCOL_ID="${PROTOCOL_ID:-eurosys-pa-v1-20260823}"
MODEL="${MODEL:?Set MODEL to the local DeepSeek-V2-Lite-Chat snapshot directory}"
FIXTURE_ROOT="${SCRIPT_DIR}/inputs/generated/${PROTOCOL_ID}/deepseek_v2_lite_bf16"
CALIBRATION_FIXTURE="${CALIBRATION_FIXTURE:-${FIXTURE_ROOT}/calibration_ordinary_chat.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FIXTURE_ROOT}/calibration_assets}"

ROUTING_CONCURRENCY="${ROUTING_CONCURRENCY:-32}"
CPU_OFFLOAD_GIB="${CPU_OFFLOAD_GIB:-8}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-1073741824}"
GPU_CLEAN_MIB="${GPU_CLEAN_MIB:-128}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"

die() {
  echo "error: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "missing file: $1"
}

wait_for_clean_gpu() {
  local used_mib="" pids="" attempt
  for attempt in $(seq 1 180); do
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] \
      && (( used_mib <= GPU_CLEAN_MIB )) \
      && [[ -z "${pids//[[:space:]]/}" ]]; then
      return 0
    fi
    sleep 1
  done
  die "GPU did not become clean (used=${used_mib:-unknown} MiB, pids=${pids:-none})"
}

write_failure() {
  local stage="$1" rc="$2"
  jq -n \
    --arg protocol_id "${PROTOCOL_ID}" \
    --arg stage "${stage}" \
    --argjson returncode "${rc}" \
    --arg created_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    '{schema_version: 1, status: "failed", protocol_id: $protocol_id,
      stage: $stage, returncode: $returncode, created_at: $created_at}' \
    >"${OUTPUT_ROOT}/FAILED.json"
}

[[ "${ALLOW_GPU_RUN:-0}" == "1" ]] \
  || die "GPU execution requires ALLOW_GPU_RUN=1"
[[ "${ROUTING_CONCURRENCY}" == "32" ]] \
  || die "formal calibration requires all 32 frozen requests"
[[ ! -e "${OUTPUT_ROOT}" ]] || die "refusing to overwrite ${OUTPUT_ROOT}"
require_file "${VLLM_PY}"
require_file "${KT_PY}"
require_file "${BENCHMARK}"
require_file "${PLAN_BUILDER}"
require_file "${KT_BUILDER}"
require_file "${MODEL}/config.json"
require_file "${CALIBRATION_FIXTURE}"
command -v jq >/dev/null || die "jq is unavailable"
command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
command -v taskset >/dev/null || die "taskset is unavailable"

jq -e --arg protocol "${PROTOCOL_ID}" '
  .metadata.protocol_id == $protocol
  and .metadata.fixture_role == "calibration"
  and .metadata.workload == "ordinary_chat"
  and .metadata.requests == 32
  and .metadata.prompt_tokens == 512
  and .metadata.output_tokens == 128
  and ([.requests[].prompt_token_ids | length] | all(. == 512))
' "${CALIBRATION_FIXTURE}" >/dev/null \
  || die "calibration fixture does not match the frozen protocol"

mkdir -p "${OUTPUT_ROOT}/routing" "${OUTPUT_ROOT}/plans" "${OUTPUT_ROOT}/kt_placements"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"
ROUTING_DIR="${OUTPUT_ROOT}/routing"
ROUTING_TRACE="${ROUTING_DIR}/routing_trace.jsonl"
RESULT="${ROUTING_DIR}/result.json"
CPU_MANIFEST="${ROUTING_DIR}/cpu_offload_manifest.jsonl"

command=(
  timeout --kill-after=30s "${TIMEOUT_SECONDS}s"
  taskset -c 0-13
  "${VLLM_PY}" "${BENCHMARK}"
  --model "${MODEL}"
  --evaluation-fixture "${CALIBRATION_FIXTURE}"
  --output "${RESULT}"
  --profile-label formal_deepseek_routing_calibration
  --mechanism-class calibration_only_native_pinned_uva
  --concurrency "${ROUTING_CONCURRENCY}"
  --output-tokens 128
  --warmup-requests 0
  --warmup-output-tokens 8
  --kv-cache-memory-bytes "${KV_CACHE_BYTES}"
  --max-model-len 640
  --max-num-batched-tokens 20480
  --max-num-seqs 32
  --disable-chunked-prefill
  --gpu-memory-utilization 0.97
  --cpu-offload-gb "${CPU_OFFLOAD_GIB}"
  --seed 20260823
  --require-prompt-token-ids
)
printf 'env VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_CPU_OFFLOAD_BACKEND=uva ' \
  >"${ROUTING_DIR}/command.txt"
printf 'VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=1 ' >>"${ROUTING_DIR}/command.txt"
printf 'VLLM_UVM_MOE_ROUTING_TRACE_FILE=%q ' "${ROUTING_TRACE}" \
  >>"${ROUTING_DIR}/command.txt"
printf '%q ' "${command[@]}" >>"${ROUTING_DIR}/command.txt"
printf '\n' >>"${ROUTING_DIR}/command.txt"

git -C "${REPO_ROOT}" rev-parse HEAD >"${ROUTING_DIR}/repo.commit"
git -C "${REPO_ROOT}/KTransformers" rev-parse HEAD >"${ROUTING_DIR}/ktransformers.commit"
git -C "${REPO_ROOT}/KTransformers/third_party/sglang" rev-parse HEAD \
  >"${ROUTING_DIR}/sglang.commit"
nvidia-smi -q >"${ROUTING_DIR}/nvidia_smi_q.before.txt"
: >"${CPU_MANIFEST}"
wait_for_clean_gpu

set +e
(
  cd "${VLLM_ROOT}"
  env \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_CPU_OFFLOAD_BACKEND=uva \
    VLLM_CPU_OFFLOAD_MANIFEST_FILE="${CPU_MANIFEST}" \
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 \
    VLLM_CPU_OFFLOAD_COPY_TRACE=0 \
    VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=1 \
    VLLM_UVM_MOE_ROUTING_TRACE_FILE="${ROUTING_TRACE}" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 CUDA_VISIBLE_DEVICES=0 \
    "${command[@]}"
) >"${ROUTING_DIR}/run.log" 2>&1
rc=$?
set -e
printf '%s\n' "${rc}" >"${ROUTING_DIR}/returncode.txt"
wait_for_clean_gpu
nvidia-smi -q >"${ROUTING_DIR}/nvidia_smi_q.after.txt"
if (( rc != 0 )); then
  write_failure routing "${rc}"
  tail -n 240 "${ROUTING_DIR}/run.log" >&2 || true
  exit "${rc}"
fi

require_file "${RESULT}"
require_file "${ROUTING_TRACE}"
jq -e '
  .status == "ok"
  and .measurement.completed_requests == 32
  and .measurement.output_tokens == 4096
  and ([.measurement.requests[].prompt_token_ids_exact] | all(. == true))
  and ([.measurement.requests[].output_tokens] | all(. == 128))
' "${RESULT}" >/dev/null || {
  write_failure routing_verification 1
  die "routing calibration result failed verification"
}

"${VLLM_PY}" "${PLAN_BUILDER}" \
  --trace "${ROUTING_TRACE}" \
  --output-dir "${OUTPUT_ROOT}/plans" \
  --serving-start-step 3 \
  --prefill-step-count 1 \
  >"${OUTPUT_ROOT}/plans/prepare.stdout.json"

PATH="${REPO_ROOT}/KTransformers/.venv/bin:${PATH}" \
  "${KT_PY}" "${KT_BUILDER}" \
    --input-dir "${OUTPUT_ROOT}/plans" \
    --output-dir "${OUTPUT_ROOT}/kt_placements" \
    >"${OUTPUT_ROOT}/kt_placements/prepare.stdout.txt"

jq -e '
  .schema == "deepseek_v2_lite_cold_budget_matrix_v1"
  and (.tiers | keys == ["2g", "4g", "6g", "8g"])
  and .coverage.scheduler_steps.prefill == 1
  and .coverage.scheduler_steps.decode == 127
  and .coverage.prefill_num_tokens == [16384]
  and .coverage.decode_num_tokens_min == 32
  and .coverage.decode_num_tokens_max == 32
' "${OUTPUT_ROOT}/plans/manifest.json" >/dev/null \
  || {
    write_failure cold_plan_trace_coverage 1
    die "cold-plan trace coverage indicates preemption, recompute, or truncation"
  }
jq -e '
  .schema == "deepseek_v2_lite_kt_frequency_placements_v1"
  and (.tiers | keys == ["2g", "4g", "6g", "8g"])
' "${OUTPUT_ROOT}/kt_placements/kt_frequency_manifest.json" >/dev/null \
  || die "KT placement manifest verification failed"

fixture_sha="$(sha256sum "${CALIBRATION_FIXTURE}" | awk '{print $1}')"
trace_sha="$(sha256sum "${ROUTING_TRACE}" | awk '{print $1}')"
result_sha="$(sha256sum "${RESULT}" | awk '{print $1}')"
plans_sha="$(sha256sum "${OUTPUT_ROOT}/plans/manifest.json" | awk '{print $1}')"
kt_sha="$(sha256sum "${OUTPUT_ROOT}/kt_placements/kt_frequency_manifest.json" | awk '{print $1}')"
jq -n \
  --arg protocol_id "${PROTOCOL_ID}" \
  --arg created_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg model "$(realpath "${MODEL}")" \
  --arg fixture "$(realpath "${CALIBRATION_FIXTURE}")" \
  --arg fixture_sha256 "${fixture_sha}" \
  --arg trace "${ROUTING_TRACE}" \
  --arg trace_sha256 "${trace_sha}" \
  --arg result "${RESULT}" \
  --arg result_sha256 "${result_sha}" \
  --arg plans_manifest "${OUTPUT_ROOT}/plans/manifest.json" \
  --arg plans_manifest_sha256 "${plans_sha}" \
  --arg kt_manifest "${OUTPUT_ROOT}/kt_placements/kt_frequency_manifest.json" \
  --arg kt_manifest_sha256 "${kt_sha}" \
  --arg repo_commit "$(cat "${ROUTING_DIR}/repo.commit")" \
  --arg kt_commit "$(cat "${ROUTING_DIR}/ktransformers.commit")" \
  --arg sglang_commit "$(cat "${ROUTING_DIR}/sglang.commit")" \
  '{schema_version: 1, status: "ok", protocol_id: $protocol_id,
    created_at: $created_at, purpose: "calibration_only_not_performance",
    model: $model,
    calibration_fixture: {path: $fixture, sha256: $fixture_sha256, requests: 32},
    routing_trace: {path: $trace, sha256: $trace_sha256},
    routing_result: {path: $result, sha256: $result_sha256},
    cold_plans: {manifest: $plans_manifest, sha256: $plans_manifest_sha256},
    kt_exact_placements: {manifest: $kt_manifest, sha256: $kt_manifest_sha256},
    provenance: {repo_commit: $repo_commit, ktransformers_commit: $kt_commit,
      sglang_commit: $sglang_commit}}' >"${OUTPUT_ROOT}/asset_manifest.json"

jq -n \
  --arg protocol_id "${PROTOCOL_ID}" \
  --arg completed_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg manifest_sha256 "$(sha256sum "${OUTPUT_ROOT}/asset_manifest.json" | awk '{print $1}')" \
  '{schema_version: 1, status: "complete", protocol_id: $protocol_id,
    completed_at: $completed_at, asset_manifest_sha256: $manifest_sha256}' \
  >"${OUTPUT_ROOT}/COMPLETED.json"

echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
