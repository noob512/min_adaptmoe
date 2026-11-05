#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
PYTHON="${VLLM_ROOT}/.venv/bin/python"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
E6_RUNNER="${SCRIPT_DIR}/run_e6_full_rebuild_sequence.py"
E6_GENERATOR="${SCRIPT_DIR}/generate_e6_difference_plans.py"
E6_VERIFIER="${SCRIPT_DIR}/verify_e6_full_rebuild_qualification.py"
BASE_PLAN="${REPO_ROOT}/experiments/hybrid_offload_2_4_6_8_compare/inputs/cold_plan_8g.json"
NAMES="${REPO_ROOT}/experiments/hybrid_offload_2_4_6_8_compare/inputs/exact_parameter_names.txt"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7}"
FIXTURE="${FIXTURE:-${SCRIPT_DIR}/inputs/legacy_pilot/deepseek_v2_lite_bf16/fixtures/evaluation_fixture.json}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/e6-full-rebuild-$(date -u +%Y%m%dT%H%M%SZ)-qualification}"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
# shellcheck source=/dev/null
source "${BASE_HARNESS}"

die_e6() { echo "E6 full-rebuild qualification: $*" >&2; exit 1; }
[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die_e6 "GPU execution requires ALLOW_GPU_RUN=1"
for path in "${PYTHON}" "${E6_RUNNER}" "${E6_GENERATOR}" "${E6_VERIFIER}" \
  "${BASE_PLAN}" "${NAMES}" "${MODEL}/config.json" "${FIXTURE}"; do
  [[ -e "${path}" ]] || die_e6 "missing input: ${path}"
done
[[ ! -e "${RUN_ROOT}" ]] || die_e6 "RUN_ROOT already exists: ${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
RUN_ROOT_ACTIVE="${RUN_ROOT}"

"${PYTHON}" "${E6_GENERATOR}" --base-plan "${BASE_PLAN}" \
  --output-dir "${RUN_ROOT}/inputs" >"${RUN_ROOT}/plan_generation.stdout.json"
PLAN_CLASSES="${RUN_ROOT}/inputs/plan_classes.json"
: >"${RUN_ROOT}/manifest.jsonl"
"${PYTHON}" - "${RUN_ROOT}/run_config.json" "${PLAN_CLASSES}" <<'PY'
import json, os, sys, time
payload = {
    "schema": "eurosys_e6_full_rebuild_qualification_run_v1",
    "created_at": time.time(),
    "process_blocks": 1,
    "percentages": [5, 10, 25, 50, 100],
    "model": "DeepSeek-V2-Lite-Chat BF16",
    "plan_classes": os.path.realpath(sys.argv[2]),
    "kv_cache_bytes": 268435456,
    "slot_budget_bytes": 10737418240,
    "max_tokens": 8,
    "seed": 20260828,
}
open(sys.argv[1], "x", encoding="utf-8").write(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY

runtime_preflight
module_snapshot "${RUN_ROOT}/module.before.json" "${STOCK_UVM}"
kernel_epoch="$(date +%s)"
load_custom_module
module_snapshot "${RUN_ROOT}/module.custom.json" "${CUSTOM_UVM}"
gpu_state_json "${RUN_ROOT}/gpu.before.json"
wait_gpu_clean || die_e6 "GPU is not clean before the full-rebuild sequence"

set +e
env \
  VLLM_PHASE_SWITCH_CONTROLLED_HARD_EXIT_AFTER_CLEANUP=1 \
  VLLM_MLA_DISABLE=1 \
  VLLM_ATTENTION_BACKEND=TRITON_ATTN \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  VLLM_CPU_OFFLOAD_BACKEND=copy \
  VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${NAMES}" \
  VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1 \
  VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va \
  VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="${RUN_ROOT}/inputs/base_c8.json" \
  VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288 \
  VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=auto \
  VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1 \
  VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
  VLLM_UVM_ORIGINAL_VA_PHASE_MODE=prefill_graph_remote_decode \
  VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES=268435456 \
  VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ=0 \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
  VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
  VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0 \
  VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0 \
  VLLM_CPU_OFFLOAD_COPY_TRACE=0 \
  VLLM_CPU_OFFLOAD_COPY_SYNC=0 \
  VLLM_CPU_OFFLOAD_MANIFEST_FILE="${RUN_ROOT}/manifest.jsonl" \
  VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 \
  VLLM_UVM_LOG_FILE="${RUN_ROOT}/uvm.log" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 CUDA_VISIBLE_DEVICES=0 \
  timeout --kill-after=60s 3600s taskset -c 0-13 \
  "${PYTHON}" "${E6_RUNNER}" \
    --model "${MODEL}" \
    --fixture "${FIXTURE}" \
    --exact-names "${NAMES}" \
    --plan-classes "${PLAN_CLASSES}" \
    --output "${RUN_ROOT}/result.json" \
    --max-tokens 8 \
    --kv-cache-memory-bytes 268435456 \
    --cpu-offload-gb 30.0 \
    --gpu-memory-utilization 0.97 \
    --seed 20260828 \
  >"${RUN_ROOT}/run.log" 2>&1
rc=$?
sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
  >"${RUN_ROOT}/kernel.since.log" 2>"${RUN_ROOT}/kernel.since.err"
journal_rc=$?
set -e
printf '%s\n' "${rc}" >"${RUN_ROOT}/returncode.txt"
printf '%s\n' "${journal_rc}" >"${RUN_ROOT}/kernel_log_returncode.txt"
wait_gpu_clean || die_e6 "GPU/client state did not clean after the sequence"
gpu_state_json "${RUN_ROOT}/gpu.after.json"

(( journal_rc == 0 )) || die_e6 "failed to collect kernel log"
if rg -qi \
  'NVRM: Xid|FAULT_PDE|manual[_ -]migration[_ -]fault|(^|[^[:alnum:]_])Oops|(^|[^[:alnum:]_])BUG:|kernel panic|general protection fault|segfault|Call Trace' \
  "${RUN_ROOT}/kernel.since.log"; then
  restore_stock
  restart_exporters
  die_e6 "kernel log contains a fatal GPU/UVM signature"
fi
if (( rc != 0 )); then
  tail -n 300 "${RUN_ROOT}/run.log" >&2 || true
  restore_stock
  restart_exporters
  die_e6 "full-rebuild sequence failed"
fi

restore_stock
module_snapshot "${RUN_ROOT}/module.after.json" "${STOCK_UVM}"
restart_exporters
"${PYTHON}" "${E6_VERIFIER}" "${RUN_ROOT}" --output "${RUN_ROOT}/verification.json"
echo "RUN_ROOT=${RUN_ROOT}"
