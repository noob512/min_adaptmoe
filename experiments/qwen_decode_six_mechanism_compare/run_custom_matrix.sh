#!/usr/bin/env bash
set -euo pipefail

OWN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OWN_REPO="$(cd "${OWN_DIR}/../.." && pwd)"
VLLM_ROOT="${OWN_REPO}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
BASE_HARNESS="${OWN_REPO}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
RUNNER="${OWN_REPO}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/benchmark_phase_switch_steps.py"
OWN_PREPARE="${OWN_DIR}/prepare_inputs.py"
INPUTS="${OWN_DIR}/inputs"
NAMES="${INPUTS}/exact_parameter_names.txt"
CALIBRATION="${OWN_REPO}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/routing/expert_distribution.pt"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
FIXTURE="${FIXTURE:-${OWN_REPO}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json}"
RUN_ROOT="${RUN_ROOT:-${OWN_DIR}/artifacts/custom_$(date -u +%Y%m%dT%H%M%SZ)}"
TIERS="${TIERS:-2g,4g}"
BATCHES="${BATCHES:-1,4}"
ARMS="${ARMS:-R,C}"
WARMUP_DECODE_STEPS="${WARMUP_DECODE_STEPS:-5}"
MEASURED_DECODE_STEPS="${MEASURED_DECODE_STEPS:-16}"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
# shellcheck source=/dev/null
source "${BASE_HARNESS}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
require_file "${MODEL}/config.json"
require_file "${FIXTURE}"
require_file "${CALIBRATION}"
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"
mkdir -p "${INPUTS}" "${RUN_ROOT}"
"${VLLM_PY}" "${OWN_PREPARE}" --source "${CALIBRATION}" --output-dir "${INPUTS}" \
  >"${RUN_ROOT}/prepare_inputs.stdout.json"
[[ "$(wc -l <"${NAMES}")" == 96 ]] || die "expected 96 exact names"

RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
RUN_ROOT_ACTIVE="${RUN_ROOT}"
runtime_preflight
module_snapshot "${RUN_ROOT}/module.before.json" "${STOCK_UVM}"
load_custom_module
module_snapshot "${RUN_ROOT}/module.custom.json" "${CUSTOM_UVM}"
printf 'tier\tarm\tbatch\tselected_bytes\tdirectory\treturncode\n' \
  >"${RUN_ROOT}/execution.tsv"

IFS=',' read -r -a tier_list <<<"${TIERS}"
IFS=',' read -r -a arm_list <<<"${ARMS}"
IFS=',' read -r -a batch_list <<<"${BATCHES}"
for tier in "${tier_list[@]}"; do
  plan="${INPUTS}/cold_decode_plan_${tier}.json"
  require_file "${plan}"
  selected_bytes="$(jq -er '.metadata.selected_main_bytes' "${plan}")"
  for arm in "${arm_list[@]}"; do
    [[ "${arm}" == R || "${arm}" == C ]] || die "custom arm must be R or C"
    execution=eager
    [[ "${arm}" == C ]] && execution=graph
    for batch in "${batch_list[@]}"; do
      case_dir="${RUN_ROOT}/${tier}_${execution}_b${batch}"
      mkdir "${case_dir}"
      : >"${case_dir}/manifest.jsonl"
      wait_gpu_clean || die "GPU is not clean before ${tier}/${execution}/b${batch}"
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
        VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=4k \
        VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
        VLLM_UVM_ORIGINAL_VA_PHASE_MODE=remote_only \
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
          --model "${MODEL}" --fixture "${FIXTURE}" \
          --exact-names "${NAMES}" --expert-plan "${plan}" \
          --offload-manifest "${case_dir}/manifest.jsonl" \
          --output "${case_dir}/result.json" --arm "${arm}" \
          --batch-size "${batch}" --cudagraph-capture-sizes 1,4 \
          --warmup-decode-steps "${WARMUP_DECODE_STEPS}" \
          --measured-decode-steps "${MEASURED_DECODE_STEPS}" \
          --kv-cache-memory-bytes 167772160 --cpu-offload-gb 30.0 \
          --max-model-len 512 --max-num-seqs 4 \
          --max-num-batched-tokens 2048 --gpu-memory-utilization 0.97 \
          --seed 0 --expected-parameter-count 96 \
          --expected-full-parameter-bytes 28991029248 \
          --expected-selected-bytes "${selected_bytes}" \
        >"${case_dir}/run.log" 2>&1
      rc=$?
      set -e
      echo "${rc}" >"${case_dir}/returncode.txt"
      wait_gpu_clean || die "GPU did not clean after ${tier}/${execution}/b${batch}"
      gpu_state_json "${case_dir}/gpu.after.json"
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${tier}" "${arm}" "${batch}" "${selected_bytes}" \
        "$(basename "${case_dir}")" "${rc}" >>"${RUN_ROOT}/execution.tsv"
      if (( rc != 0 )); then
        tail -n 240 "${case_dir}/run.log" >&2 || true
        die "custom case failed: ${tier}/${execution}/b${batch}"
      fi
      jq -e '.status == "ok"' "${case_dir}/result.json" >/dev/null \
        || die "custom result is not ok"
    done
  done
done

restore_stock
module_snapshot "${RUN_ROOT}/module.after.json" "${STOCK_UVM}"
restart_exporters
echo "RUN_ROOT=${RUN_ROOT}"
