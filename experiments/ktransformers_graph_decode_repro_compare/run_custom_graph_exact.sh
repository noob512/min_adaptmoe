#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
EXACT_RUNNER="${RUNNER:-${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/benchmark_phase_switch_steps.py}"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
FIXTURE="${FIXTURE:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json}"
PLAN="${PLAN:-${SCRIPT_DIR}/inputs/cold_decode_plan_4g.json}"
NAMES="${NAMES:-${SCRIPT_DIR}/inputs/exact_parameter_names.txt}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/custom_graph_exact_$(date -u +%Y%m%dT%H%M%SZ)}"
BATCHES="${BATCHES:-1,4}"
CAPTURE_SIZES="${CAPTURE_SIZES:-${BATCHES}}"
WARMUP_DECODE_STEPS="${WARMUP_DECODE_STEPS:-5}"
MEASURED_DECODE_STEPS="${MEASURED_DECODE_STEPS:-57}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-167772160}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GRAPH_MIN_FREE_BYTES="${GRAPH_MIN_FREE_BYTES:-268435456}"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
# shellcheck source=/dev/null
source "${BASE_HARNESS}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY
RUNNER="${EXACT_RUNNER}"

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
require_file "${MODEL}/config.json"
require_file "${FIXTURE}"
require_file "${PLAN}"
require_file "${NAMES}"
[[ "$(wc -l <"${NAMES}")" == 96 ]] || die "expected 96 exact parameter names"
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"

selected_bytes="$(jq -er '.metadata.selected_main_bytes' "${PLAN}")"
cold_experts="$(jq -er '.metadata.offload_plan_records' "${PLAN}")"
[[ "${selected_bytes}" =~ ^[1-9][0-9]*$ ]] || die "invalid selected byte count"
[[ "${cold_experts}" =~ ^[1-9][0-9]*$ ]] || die "invalid cold expert count"

mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
RUN_ROOT_ACTIVE="${RUN_ROOT}"
runtime_preflight
module_snapshot "${RUN_ROOT}/module.before.json" "${STOCK_UVM}"
load_custom_module
module_snapshot "${RUN_ROOT}/module.custom.json" "${CUSTOM_UVM}"
printf 'batch\tselected_bytes\tcold_experts\tdirectory\treturncode\n' \
  >"${RUN_ROOT}/execution.tsv"

IFS=',' read -r -a batch_list <<<"${BATCHES}"
for batch in "${batch_list[@]}"; do
  [[ "${batch}" =~ ^(1|4|8)$ ]] || die "batch must be 1, 4, or 8"
  case_dir="${RUN_ROOT}/b${batch}"
  mkdir "${case_dir}"
  : >"${case_dir}/manifest.jsonl"
  wait_gpu_clean || die "GPU is not clean before custom Graph B${batch}"
  gpu_state_json "${case_dir}/gpu.before.json"
  set +e
  env \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_CPU_OFFLOAD_BACKEND=copy \
    VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${NAMES}" \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1 \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="${PLAN}" \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288 \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=4k \
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
    VLLM_UVM_ORIGINAL_VA_PHASE_MODE=remote_only \
    VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES="${GRAPH_MIN_FREE_BYTES}" \
    VLLM_UVM_ORIGINAL_VA_PHASE_RESULT_FILE="${case_dir}/transition.json" \
    VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ=0 \
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0 \
    VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0 \
    VLLM_CPU_OFFLOAD_COPY_TRACE=0 \
    VLLM_CPU_OFFLOAD_COPY_SYNC=0 \
    VLLM_CPU_OFFLOAD_MANIFEST_FILE="${case_dir}/manifest.jsonl" \
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 \
    VLLM_UVM_LOG_FILE="${case_dir}/uvm.log" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0 \
    timeout --kill-after=30s 1800s taskset -c 0-13 \
    "${VLLM_PY}" "${RUNNER}" \
      --model "${MODEL}" --fixture "${FIXTURE}" \
      --exact-names "${NAMES}" --expert-plan "${PLAN}" \
      --offload-manifest "${case_dir}/manifest.jsonl" \
      --output "${case_dir}/result.json" --arm C \
      --batch-size "${batch}" --cudagraph-capture-sizes "${CAPTURE_SIZES}" \
      --warmup-decode-steps "${WARMUP_DECODE_STEPS}" \
      --measured-decode-steps "${MEASURED_DECODE_STEPS}" \
      --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
      --cpu-offload-gb 30.0 --max-model-len 512 \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --gpu-memory-utilization 0.97 \
      --seed 0 --expected-parameter-count 96 \
      --expected-full-parameter-bytes 28991029248 \
      --expected-selected-bytes "${selected_bytes}" \
    >"${case_dir}/run.log" 2>&1
  rc=$?
  set -e
  printf '%s\n' "${rc}" >"${case_dir}/returncode.txt"
  wait_gpu_clean || die "GPU did not clean after custom Graph B${batch}"
  gpu_state_json "${case_dir}/gpu.after.json"
  printf '%s\t%s\t%s\t%s\t%s\n' "${batch}" "${selected_bytes}" \
    "${cold_experts}" "$(basename "${case_dir}")" "${rc}" \
    >>"${RUN_ROOT}/execution.tsv"
  if (( rc != 0 )); then
    tail -n 240 "${case_dir}/run.log" >&2 || true
    die "custom Graph B${batch} failed"
  fi
  jq -e '.status == "ok"' "${case_dir}/result.json" >/dev/null \
    || die "custom Graph result is not ok"
done

restore_stock
module_snapshot "${RUN_ROOT}/module.after.json" "${STOCK_UVM}"
restart_exporters
echo "RUN_ROOT=${RUN_ROOT}"
