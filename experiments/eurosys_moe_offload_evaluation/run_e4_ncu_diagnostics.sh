#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
RUNNER="${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/benchmark_e4_concurrency.py"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
NCU="${NCU:-/usr/local/cuda-12.8/bin/ncu}"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-${SCRIPT_DIR}/artifacts/e3-calibration-20260825T091247Z}"
PLAN="${CALIBRATION_ROOT}/calibration_assets/cold_decode_plan_2g.json"
NAMES="${CALIBRATION_ROOT}/calibration_assets/exact_parameter_names.txt"
FIXTURE="${SCRIPT_DIR}/inputs/generated/eurosys-pa-v2-20260824/qwen3_30b_a3b_fp8/evaluation_short_dialogue.json"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/e4-ncu-$(date -u +%Y%m%dT%H%M%SZ)}"
CPUSET="${CPUSET:-0-13}"
SELECTED_BYTES=2264924160
FULL_BYTES=28991029248
BACKING_BYTES=94371840
CUSTOM_LOADED=0

die() { echo "error: $*" >&2; exit 1; }
cleanup() {
  if [[ "${CUSTOM_LOADED}" == 1 ]]; then
    restore_stock || true
    CUSTOM_LOADED=0
    restart_exporters || true
  fi
}
trap cleanup EXIT INT TERM

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
for path in "${VLLM_PY}" "${RUNNER}" "${NCU}" "${MODEL}/config.json" "${PLAN}" "${NAMES}" "${FIXTURE}"; do
  [[ -f "${path}" ]] || die "missing file: ${path}"
done
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"
mkdir -p "${RUN_ROOT}/provenance" "${RUN_ROOT}/cases"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
# shellcheck source=/dev/null
source "${BASE_HARNESS}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY
# The shared harness has a legacy RUNNER variable; restore the E4 diagnostic.
RUNNER="${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/benchmark_e4_concurrency.py"
RUN_ROOT_ACTIVE="${RUN_ROOT}"
runtime_preflight || die "custom runtime preflight failed"
sha256sum "${RUNNER}" "${PLAN}" "${NAMES}" "${FIXTURE}" >"${RUN_ROOT}/provenance/inputs.sha256"
"${NCU}" --version >"${RUN_ROOT}/provenance/ncu-version.txt"
module_snapshot "${RUN_ROOT}/provenance/module.before.json" "${STOCK_UVM}"
load_custom_module || die "could not load custom UVM"
CUSTOM_LOADED=1
module_snapshot "${RUN_ROOT}/provenance/module.custom.json" "${CUSTOM_UVM}"

printf 'concurrency\tstatus\treturncode\n' >"${RUN_ROOT}/execution.tsv"
for concurrency in 1 16 32; do
  case_dir="${RUN_ROOT}/cases/c${concurrency}"
  mkdir "${case_dir}"
  : >"${case_dir}/manifest.jsonl"
  wait_gpu_clean || die "GPU not clean before C${concurrency}"
  set +e
  env VLLM_ATTENTION_BACKEND=TRITON_ATTN VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_CPU_OFFLOAD_BACKEND=copy VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${NAMES}" \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1 VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER=0 \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="${PLAN}" \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288 \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=4k \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPECTED_ARENA_COUNT=4 \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPECTED_BACKING_BYTES="${BACKING_BYTES}" \
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 VLLM_UVM_ORIGINAL_VA_PHASE_MODE=remote_only \
    VLLM_UVM_ORIGINAL_VA_PHASE_RESULT_FILE="${case_dir}/transition.json" \
    VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ=0 VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0 \
    VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0 VLLM_CPU_OFFLOAD_COPY_TRACE=0 \
    VLLM_CPU_OFFLOAD_COPY_SYNC=0 VLLM_CPU_OFFLOAD_MANIFEST_FILE="${case_dir}/manifest.jsonl" \
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 VLLM_UVM_LOG_FILE="${case_dir}/uvm.log" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 CUDA_VISIBLE_DEVICES=0 \
    taskset -c "${CPUSET}" sudo -n -E "${NCU}" --profile-from-start off --target-processes all \
      --cache-control none --replay-mode application \
      --kernel-name 'regex:.*fused_moe_kernel.*' --launch-count 8 \
      --metrics smsp__warp_issue_stalled_long_scoreboard_per_warp_active,smsp__warp_issue_stalled_lg_throttle_per_warp_active,smsp__warp_issue_stalled_mio_throttle_per_warp_active \
      --csv --page raw "${VLLM_PY}" "${RUNNER}" --model "${MODEL}" --fixture "${FIXTURE}" \
      --exact-names "${NAMES}" --expert-plan "${PLAN}" --offload-manifest "${case_dir}/manifest.jsonl" \
      --output "${case_dir}/result.json" --progress "${case_dir}/progress.jsonl" \
      --concurrencies "${concurrency}" --skip-warmup-phase --output-tokens 16 \
      --warm-output-tokens 16 --warmup-decode-steps 5 --cuda-profiler-measured-steps 1 \
      --kv-cache-memory-bytes 650000000 --cpu-offload-gb 30.0 --max-model-len 272 \
      --max-num-batched-tokens 4096 --gpu-memory-utilization 0.97 --request-rotation 0 --seed 0 \
      --expected-parameter-count 96 --expected-full-parameter-bytes "${FULL_BYTES}" \
      --expected-selected-bytes "${SELECTED_BYTES}" \
    >"${case_dir}/ncu.csv" 2>"${case_dir}/ncu.stderr"
  rc=$?
  set +e
  status=error
  if (( rc == 0 )) && jq -e '
      .status == "ok"
      and (.phases | length == 1)
      and .shutdown_contract.diagnostics_after_shutdown.driver.session_count == 0
    ' "${case_dir}/result.json" >/dev/null 2>&1 \
    && rg -q 'smsp__warp_issue_stalled_long_scoreboard_per_warp_active' "${case_dir}/ncu.csv"; then
    status=ok
  fi
  printf '%s\n' "${rc}" >"${case_dir}/returncode.txt"
  printf '%s\t%s\t%s\n' "${concurrency}" "${status}" "${rc}" >>"${RUN_ROOT}/execution.tsv"
  wait_gpu_clean || echo dirty >"${case_dir}/gpu_cleanup_error.txt"
done

restore_stock
CUSTOM_LOADED=0
restart_exporters
module_snapshot "${RUN_ROOT}/provenance/module.after.json" "${STOCK_UVM}"
echo "RUN_ROOT=${RUN_ROOT}"
