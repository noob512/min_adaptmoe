#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
GRAPH_RUNNER_PATH="${REPO_ROOT}/experiments/vllm_p3_static_alias_graph_integration/benchmark_full_graph_steps.py"
INPUTS="${REPO_ROOT}/experiments/qwen_decode_six_mechanism_compare/inputs"
NAMES="${INPUTS}/exact_parameter_names.txt"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
FIXTURE="${FIXTURE:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/custom_$(date -u +%Y%m%dT%H%M%SZ)}"
TIERS="${TIERS:-2g,4g}"
MODES="${MODES:-eager,graph}"
CONDITIONING_REPEATS="${CONDITIONING_REPEATS:-1}"
MEASUREMENT_REPEATS="${MEASUREMENT_REPEATS:-3}"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
# shellcheck source=/dev/null
source "${BASE_HARNESS}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY
RUNNER="${GRAPH_RUNNER_PATH}"

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
require_file "${RUNNER}"
require_file "${MODEL}/config.json"
require_file "${FIXTURE}"
require_file "${NAMES}"
[[ "$(wc -l <"${NAMES}")" == 96 ]] || die "expected 96 exact parameter names"
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
RUN_ROOT_ACTIVE="${RUN_ROOT}"

runtime_preflight
module_snapshot "${RUN_ROOT}/module.before.json" "${STOCK_UVM}"
load_custom_module
module_snapshot "${RUN_ROOT}/module.custom.json" "${CUSTOM_UVM}"
printf 'tier\tmode\tselected_main_bytes\tbacking_bytes\tdirectory\treturncode\n' \
  >"${RUN_ROOT}/execution.tsv"

IFS=',' read -r -a tier_list <<<"${TIERS}"
IFS=',' read -r -a mode_list <<<"${MODES}"
for tier in "${tier_list[@]}"; do
  plan="${INPUTS}/cold_decode_plan_${tier}.json"
  require_file "${plan}"
  selected_bytes="$(jq -er '.metadata.selected_main_bytes' "${plan}")"
  layers="$(jq -er '.metadata.num_layers' "${plan}")"
  # Four reusable arenas: two parameter kinds times two banks.  Each bank is
  # sized for one layer's selected expert slices.
  backing_bytes="$((selected_bytes * 2 / layers))"
  for mode in "${mode_list[@]}"; do
    [[ "${mode}" == eager || "${mode}" == graph ]] \
      || die "custom mode must be eager or graph"
    case_dir="${RUN_ROOT}/${tier}_${mode}"
    mkdir "${case_dir}"
    : >"${case_dir}/manifest.jsonl"
    execution_mode=eager
    capture_sizes=1,4
    if [[ "${mode}" == graph ]]; then
      execution_mode=full_graph_all
      capture_sizes=1,4,314,1217
    fi
    wait_gpu_clean || die "GPU is not clean before ${tier}/${mode}"
    gpu_state_json "${case_dir}/gpu.before.json"
    printf '%s\n' "${selected_bytes}" >"${case_dir}/selected_main_bytes.txt"
    printf '%s\n' "${backing_bytes}" >"${case_dir}/expected_backing_bytes.txt"
    sha256sum "${FIXTURE}" "${plan}" "${NAMES}" >"${case_dir}/input_sha256.txt"
    set +e
    env \
      VLLM_ATTENTION_BACKEND=TRITON_ATTN \
      VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      VLLM_CPU_OFFLOAD_BACKEND=copy \
      VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${NAMES}" \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1 \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="${plan}" \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288 \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=4k \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPECTED_ARENA_COUNT=4 \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPECTED_BACKING_BYTES="${backing_bytes}" \
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
      timeout --kill-after=30s 3600s taskset -c 0-13 \
      "${VLLM_PY}" "${RUNNER}" \
        --model "${MODEL}" --fixture "${FIXTURE}" \
        --output "${case_dir}/result.json" \
        --offload-manifest "${case_dir}/manifest.jsonl" \
        --profile-label "qwen3_cold_${tier}_remote_${mode}_prefill" \
        --mechanism-class "custom_original_va_remote_read_${mode}" \
        --execution-mode "${execution_mode}" --batch-sizes 1,4 \
        --cudagraph-capture-sizes "${capture_sizes}" \
        --conditioning-repeats "${CONDITIONING_REPEATS}" \
        --measurement-repeats "${MEASUREMENT_REPEATS}" \
        --warmup-decode-steps 1 --measured-decode-steps 1 \
        --kv-cache-memory-bytes 167772160 --cpu-offload-gb 30.0 \
        --max-model-len 512 --max-num-seqs 4 \
        --max-num-batched-tokens 2048 --gpu-memory-utilization 0.97 \
        --seed 0 >"${case_dir}/run.log" 2>&1
    rc=$?
    set -e
    echo "${rc}" >"${case_dir}/returncode.txt"
    wait_gpu_clean || die "GPU did not clean after ${tier}/${mode}"
    gpu_state_json "${case_dir}/gpu.after.json"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${tier}" "${mode}" "${selected_bytes}" "${backing_bytes}" \
      "$(basename "${case_dir}")" "${rc}" >>"${RUN_ROOT}/execution.tsv"
    if (( rc != 0 )); then
      tail -n 240 "${case_dir}/run.log" >&2 || true
      die "custom case failed: ${tier}/${mode}"
    fi
    jq -e --arg mode "${mode}" --argjson selected "${selected_bytes}" \
      --argjson backing "${backing_bytes}" '
        .status == "ok"
        and .static_alias_diagnostics_after_load.phase_mode == "remote_only"
        and .static_alias_diagnostics_after_load.remote_map_active == true
        and .static_alias_diagnostics_after_load.selected_parameter_bytes_per_forward == $selected
        and .static_alias_diagnostics_after_load.backing_bytes == $backing
        and (.phase_summary["1"].measurement_repeats == 3)
        and (.phase_summary["4"].measurement_repeats == 3)
        and (if $mode == "graph" then
          .config.realized_cudagraph_mode == "FULL"
          and .static_alias_diagnostics_after_load.remote_graph_certificate.actual_h2d_bytes == 0
          and ([.cycles[].prefill_graph_wrapper_delta.full_replay] | all(. == 1))
        else .config.realized_cudagraph_mode == "NONE" end)
      ' "${case_dir}/result.json" >/dev/null \
      || die "custom result verification failed: ${tier}/${mode}"
  done
done

restore_stock
module_snapshot "${RUN_ROOT}/module.after.json" "${STOCK_UVM}"
restart_exporters
echo "RUN_ROOT=${RUN_ROOT}"
