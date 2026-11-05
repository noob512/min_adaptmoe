#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
PLAN_CLASS_RUNNER="${SCRIPT_DIR}/run_graph_plan_class_sequence.py"
INPUTS="${SCRIPT_DIR}/inputs"
PLAN_CLASSES="${PLAN_CLASSES:-${INPUTS}/graph_plan_classes.json}"
NAMES="${NAMES:-${INPUTS}/exact_parameter_names.txt}"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7}"
FIXTURE="${FIXTURE:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/deepseek_v2_lite_bf16/fixtures/evaluation_fixture.json}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/graph_classes_$(date -u +%Y%m%dT%H%M%SZ)}"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
export SLOT_BUDGET_BYTES="${SLOT_BUDGET_BYTES:-10737418240}"
export KV_CACHE_BYTES="${KV_CACHE_BYTES:-268435456}"
# Current custom-path contract: both CUDA Graph phases remain captured while
# P->D uses complete frozen page-table children.  Equal-width E6 difference
# classes are not nested, so they use the full-Parameter shadow and the
# synchronous differential migration transaction rather than nested-tier
# source prefill.
export STATIC_ALIAS_DUAL_VIEW_ENABLE="${STATIC_ALIAS_DUAL_VIEW_ENABLE:-1}"
export STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL="${STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL:-1}"
# shellcheck source=/dev/null
source "${BASE_HARNESS}"

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
require_file "${MODEL}/config.json"
require_file "${FIXTURE}"
require_file "${NAMES}"
require_file "${PLAN_CLASSES}"
require_file "${PLAN_CLASS_RUNNER}"
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
RUN_ROOT_ACTIVE="${RUN_ROOT}"

runtime_preflight
module_snapshot "${RUN_ROOT}/module.before.json" "${STOCK_UVM}"
kernel_epoch="$(date +%s)"
load_custom_module
module_snapshot "${RUN_ROOT}/module.custom.json" "${CUSTOM_UVM}"
gpu_state_json "${RUN_ROOT}/gpu.before.json"
wait_gpu_clean || die "GPU is not clean before the plan-class sequence"
: >"${RUN_ROOT}/manifest.jsonl"

set +e
runner_extra_args=()
if [[ -n "${TRANSITION_PATHS:-}" ]]; then
  require_file "${TRANSITION_PATHS}"
  runner_extra_args+=(--transition-paths "${TRANSITION_PATHS}")
fi
env \
  VLLM_MLA_DISABLE=1 \
  VLLM_ATTENTION_BACKEND=TRITON_ATTN \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  VLLM_CPU_OFFLOAD_BACKEND=copy \
  VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${NAMES}" \
  VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1 \
  VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va \
  VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASSES_FILE="${PLAN_CLASSES}" \
  VLLM_CPU_OFFLOAD_PLAN_CLASS_PREFILL_SOURCES="${VLLM_CPU_OFFLOAD_PLAN_CLASS_PREFILL_SOURCES:-0}" \
  VLLM_CPU_OFFLOAD_PLAN_CLASS_IMMUTABLE_CPU_SHADOW="${VLLM_CPU_OFFLOAD_PLAN_CLASS_IMMUTABLE_CPU_SHADOW:-1}" \
  VLLM_CPU_OFFLOAD_PLAN_CLASS_FORCE_SYNC_MIGRATION="${VLLM_CPU_OFFLOAD_PLAN_CLASS_FORCE_SYNC_MIGRATION:-1}" \
  VLLM_CPU_OFFLOAD_DUAL_VIEW_FULL_PARAMETER_SHADOW="${VLLM_CPU_OFFLOAD_DUAL_VIEW_FULL_PARAMETER_SHADOW:-1}" \
  VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW="${VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW:-1}" \
  VLLM_CPU_OFFLOAD_PLAN_CLASS_PRECOMPILE="${VLLM_CPU_OFFLOAD_PLAN_CLASS_PRECOMPILE:-1}" \
  VLLM_CPU_OFFLOAD_PLAN_CLASS_QUIESCED_DELTA="${VLLM_CPU_OFFLOAD_PLAN_CLASS_QUIESCED_DELTA:-1}" \
  VLLM_CPU_OFFLOAD_PLAN_CLASS_INCREMENTAL_DUAL_REFRESH="${VLLM_CPU_OFFLOAD_PLAN_CLASS_INCREMENTAL_DUAL_REFRESH:-1}" \
  VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288 \
  VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=auto \
  VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1 \
  VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
  VLLM_UVM_ORIGINAL_VA_PHASE_MODE=prefill_graph_remote_decode \
  VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES="${VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES:-268435456}" \
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
  "${VLLM_PY}" "${PLAN_CLASS_RUNNER}" \
    --model "${MODEL}" \
    --fixture "${FIXTURE}" \
    --exact-names "${NAMES}" \
    --plan-classes "${PLAN_CLASSES}" \
    --output "${RUN_ROOT}/result.json" \
    --sequence "${PLAN_CLASS_SEQUENCE:-c2,c4,c6,c8,c2}" \
    --max-tokens "${MAX_TOKENS:-8}" \
    --kv-cache-memory-bytes "${KV_CACHE_BYTES}" \
    --cpu-offload-gb "${CPU_OFFLOAD_GB:-30.0}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.97}" \
    --seed "${SEED:-0}" \
    "${runner_extra_args[@]}" \
  >"${RUN_ROOT}/run.log" 2>&1
rc=$?
sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
  >"${RUN_ROOT}/kernel.since.log" 2>"${RUN_ROOT}/kernel.since.err"
journal_rc=$?
set -e
echo "${rc}" >"${RUN_ROOT}/returncode.txt"
echo "${journal_rc}" >"${RUN_ROOT}/kernel_log_returncode.txt"
wait_gpu_clean || die "GPU/client state did not clean up after the sequence"
gpu_state_json "${RUN_ROOT}/gpu.after.json"

(( journal_rc == 0 )) || die "failed to collect the kernel log"
if rg -qi \
    'NVRM: Xid|FAULT_PDE|manual[_ -]migration[_ -]fault|(^|[^[:alnum:]_])Oops|(^|[^[:alnum:]_])BUG:|kernel panic|general protection fault|segfault|Call Trace' \
    "${RUN_ROOT}/kernel.since.log"; then
  rg -in \
    'NVRM: Xid|FAULT_PDE|manual[_ -]migration[_ -]fault|(^|[^[:alnum:]_])Oops|(^|[^[:alnum:]_])BUG:|kernel panic|general protection fault|segfault|Call Trace' \
    "${RUN_ROOT}/kernel.since.log" >&2 || true
  die "kernel log contains a fatal GPU/UVM signature"
fi

if (( rc != 0 )); then
  tail -n 300 "${RUN_ROOT}/run.log" >&2 || true
  restore_stock
  restart_exporters
  die "C2/C4/C6/C8 Graph plan-class sequence failed"
fi
jq -e '
  .status == "ok"
  and .capture_count_after_load == .capture_count_final
  and ([.records[].class_name] == .sequence)
  and ([.records[].output_sha256] | unique | length == 1)
' "${RUN_ROOT}/result.json" >/dev/null \
  || die "plan-class result contract failed"

restore_stock
module_snapshot "${RUN_ROOT}/module.after.json" "${STOCK_UVM}"
restart_exporters
echo "RUN_ROOT=${RUN_ROOT}"
