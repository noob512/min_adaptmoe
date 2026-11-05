#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
RUNNER="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/benchmark_phase_switch_steps.py"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7}"
FIXTURE="${FIXTURE:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/deepseek_v2_lite_bf16/fixtures/evaluation_fixture.json}"
INPUTS="${SCRIPT_DIR}/inputs"
NAMES="${INPUTS}/exact_parameter_names.txt"
TIERS="${TIERS:-2g,4g,6g,8g}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/custom_$(date -u +%Y%m%dT%H%M%SZ)}"
REPEATS="${REPEATS:-1}"
CUSTOM_ARM="${CUSTOM_ARM:-H}"

case "${CUSTOM_ARM}" in
  H)
    PHASE_MODE=prefill_remote_decode
    ;;
  D)
    PHASE_MODE=prefill_graph_remote_decode
    ;;
  *)
    echo "error: CUSTOM_ARM must be H (eager) or D (FULL_SEPARATE Graph)" >&2
    exit 1
    ;;
esac

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
export SLOT_BUDGET_BYTES="${SLOT_BUDGET_BYTES:-1207959552}"
# shellcheck source=/dev/null
source "${BASE_HARNESS}"

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
[[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]] || die "REPEATS must be positive"
require_file "${MODEL}/config.json"
require_file "${FIXTURE}"
require_file "${NAMES}"
require_file "${INPUTS}/manifest.json"
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
RUN_ROOT_ACTIVE="${RUN_ROOT}"

runtime_preflight
module_snapshot "${RUN_ROOT}/module.before.json" "${STOCK_UVM}"
load_custom_module
module_snapshot "${RUN_ROOT}/module.custom.json" "${CUSTOM_UVM}"

IFS=',' read -r -a tier_list <<<"${TIERS}"
printf 'tier\trepeat\tselected_bytes\tresult\treturncode\n' >"${RUN_ROOT}/execution.tsv"
for tier in "${tier_list[@]}"; do
  plan="${INPUTS}/cold_plan_${tier}.json"
  require_file "${plan}"
  selected_bytes="$(jq -er '.selected_main_bytes' "${plan}")"
  for ((sample = 1; sample <= REPEATS; ++sample)); do
    case_dir="${RUN_ROOT}/${tier}_r${sample}"
    mkdir "${case_dir}"
    : >"${case_dir}/manifest.jsonl"
    wait_gpu_clean || die "GPU is not clean before ${tier}/r${sample}"
    gpu_state_json "${case_dir}/gpu.before.json"
    set +e
    env \
      VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      VLLM_CPU_OFFLOAD_BACKEND=copy \
      VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${NAMES}" \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1 \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="${plan}" \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288 \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=auto \
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
      VLLM_UVM_ORIGINAL_VA_PHASE_MODE="${PHASE_MODE}" \
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
      OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 CUDA_VISIBLE_DEVICES=0 \
      timeout --kill-after=30s 1800s taskset -c 0-13 \
      "${VLLM_PY}" "${RUNNER}" \
        --model "${MODEL}" \
        --fixture "${FIXTURE}" \
        --exact-names "${NAMES}" \
        --expert-plan "${plan}" \
        --offload-manifest "${case_dir}/manifest.jsonl" \
        --output "${case_dir}/result.json" \
        --arm "${CUSTOM_ARM}" --batch-size 1 --cudagraph-capture-sizes 1,512 \
        --warmup-decode-steps "${WARMUP_DECODE_STEPS:-2}" \
        --measured-decode-steps "${MEASURED_DECODE_STEPS:-8}" \
        --kv-cache-memory-bytes "${KV_CACHE_BYTES:-268435456}" \
        --cpu-offload-gb 30.0 --max-model-len 544 --max-num-seqs 1 \
        --max-num-batched-tokens 1024 --gpu-memory-utilization 0.97 --seed 0 \
        --expected-parameter-count 52 \
        --expected-full-parameter-bytes 28789702656 \
        --expected-selected-bytes "${selected_bytes}" \
      >"${case_dir}/run.log" 2>&1
    rc=$?
    set -e
    echo "${rc}" >"${case_dir}/returncode.txt"
    wait_gpu_clean || die "GPU/client state did not clean up after ${tier}/r${sample}"
    gpu_state_json "${case_dir}/gpu.after.json"
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "${tier}" "${sample}" "${selected_bytes}" \
      "${case_dir}/result.json" "${rc}" >>"${RUN_ROOT}/execution.tsv"
    if (( rc != 0 )); then
      tail -n 240 "${case_dir}/run.log" >&2 || true
      die "custom case failed: ${tier}/r${sample}"
    fi
    jq -e '.status == "ok"' "${case_dir}/result.json" >/dev/null \
      || die "custom result did not reach status=ok: ${tier}/r${sample}"
  done
done

restore_stock
module_snapshot "${RUN_ROOT}/module.after.json" "${STOCK_UVM}"
restart_exporters
echo "RUN_ROOT=${RUN_ROOT}"
