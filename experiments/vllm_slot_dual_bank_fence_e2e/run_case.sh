#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_PY:-${VLLM_ROOT}/.venv/bin/python}"
BENCHMARK="${BENCHMARK:-${REPO_ROOT}/experiments/vllm_slot_direct_integration/benchmark_sequential_requests.py}"
COMMON_BENCHMARK="${COMMON_BENCHMARK:-${REPO_ROOT}/experiments/broad_offload_compare/benchmark_custom_offload.py}"
PREPARE_INPUTS="${PREPARE_INPUTS:-${REPO_ROOT}/experiments/vllm_slot_vs_pinned_copy/prepare_inputs.py}"
VERIFY="${SCRIPT_DIR}/verify_case.py"
SELF_TEST="${SCRIPT_DIR}/test_runner.py"
CUSTOM_MODULE="${CUSTOM_MODULE:-${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"
PARAM_ROOT="${PARAM_ROOT:-/sys/module/nvidia_uvm/parameters}"
CUDA_FENCE_KERNEL_LIB="${CUDA_FENCE_KERNEL_LIB:-${VLLM_ROOT}/uvm_test/libuvm_slot_cuda_fence_kernels.so}"

DEFAULT_MODEL="${REPO_ROOT}/../.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782"
DEFAULT_FIXTURE="${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json"

SLOT_CACHE_BYTES=1207959552
SLOT_CACHE_MAX_SLOTS=2304
SEQUENTIAL_REQUESTS=3
LAYER_BYTES=603979776

COMMAND="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

die() {
  echo "error: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "missing file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "missing directory: $1"
}

require_command() {
  command -v "$1" >/dev/null || die "missing command: $1"
}

usage() {
  cat <<'EOF'
Usage:
  run_case.sh static-preflight
  run_case.sh run CASE_LABEL FENCE
  run_case.sh verify RUN_DIR FENCE [METADATA_RECIPE [FAST_VALIDATION [SHADOW_PTE_PREFILL]]]

This script never loads/unloads NVIDIA modules and never writes a module
parameter. The caller must load the intended custom nvidia-uvm.ko first.

The two fence arms are separate invocations so each one can fail closed against
the already-loaded module configuration:

  ALLOW_GPU_RUN=1 METADATA_RECIPE=1 ./run_case.sh run fence_off 0
  # externally switch/reload the UVM fence parameter
  ALLOW_GPU_RUN=1 METADATA_RECIPE=1 ./run_case.sh run fence_on 1

Both cases always use one LLM instance, exactly three sequential requests, the
exact Qwen L0-L3 all-expert plan, two 576 MiB banks, and no compute/copy
prefetch. Required module parameters are checked before and after the run.

Optional environment:
  MODEL=<Qwen3-30B-A3B-FP8 snapshot>
  EVAL_FIXTURE=<frozen token-ID fixture>
  RUN_ROOT=<new output directory>
  METADATA_RECIPE=0|1                 default 1
  OVERLAP_COPY_PREPARE=0|1            default 1
  FAST_VALIDATION=0|1                 default 0
  SHADOW_PTE_PREFILL=0|1              default 0
  EXPECTED_CUSTOM_SRCVERSION=<hex>
  GPU_INDEX=0 CPUSET=0-13 CPU_THREADS=14
  OUTPUT_TOKENS=4 KV_CACHE_BYTES=1288490189
  MAX_MODEL_LEN=544 MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=32
  GPU_MEMORY_UTILIZATION=0.97 TIMEOUT_SEC=3600
  GPU_CLEAN_TIMEOUT_SEC=180
EOF
}

validate_bit() {
  local name="$1"
  local value="$2"
  [[ "${value}" == "0" || "${value}" == "1" ]] \
    || die "${name} must be 0 or 1"
}

read_parameter() {
  local name="$1"
  local path="${PARAM_ROOT}/${name}"
  [[ -r "${path}" ]] || die "module parameter is unavailable: ${path}"
  tr -d '[:space:]' < "${path}"
}

require_parameter() {
  local name="$1"
  local expected="$2"
  local actual
  actual="$(read_parameter "${name}")"
  [[ "${actual}" == "${expected}" ]] \
    || die "${name}=${actual}, expected ${expected}"
}

static_preflight() {
  require_dir "${VLLM_ROOT}"
  require_file "${VLLM_PY}"
  require_file "${BENCHMARK}"
  require_file "${COMMON_BENCHMARK}"
  require_file "${PREPARE_INPUTS}"
  require_file "${VERIFY}"
  require_file "${SELF_TEST}"
  require_file "${VLLM_ROOT}/vllm/device_allocator/uvm.py"
  require_file "${VLLM_ROOT}/vllm/device_allocator/uvm_pool_residency.py"
  require_file "${VLLM_ROOT}/vllm/model_executor/layers/fused_moe/layer.py"
  require_file "${VLLM_ROOT}/vllm/v1/worker/gpu_model_runner.py"
  require_file "${CUDA_FENCE_KERNEL_LIB}"
  require_command rg

  rg -q '_UVM_MOE_SLOT_PLAN_EXECUTE_FENCED_IOCTL = 96' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm.py" \
    || die "vLLM fenced registered-plan ioctl 96 is missing"
  rg -q 'cuStreamWriteValue32_v2' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm.py" \
    || die "vLLM compute fence write is missing"
  rg -q 'cuStreamWaitValue32_v2' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm.py" \
    || die "vLLM completion stream wait is missing"
  rg -q 'vllm_uvm_slot_fence_signal' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm.py" \
    || die "vLLM raw-UVA compute fence launcher is missing"
  rg -q 'VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_pool_residency.py" \
    || die "vLLM dual-bank registered-plan mode is missing"
  rg -q 'uvm_slot_cuda_fence_enabled' \
    "${VLLM_ROOT}/vllm/model_executor/layers/fused_moe/layer.py" \
    || die "FusedMoE fence synchronization gate is missing"
  rg -q 'uvm_moe_slot_plan_cache_metadata_recipe' \
    "${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c" \
    || die "kernel metadata-recipe cache parameter is missing"
  rg -q 'slot_direct_clear_recipe_cache_hits' \
    "${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c" \
    || die "kernel metadata-recipe telemetry is missing"
  rg -q 'uvm_moe_slot_plan_fast_validation' \
    "${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c" \
    || die "kernel fast-validation parameter is missing"
  rg -q 'uvm_moe_slot_plan_shadow_pte_prefill' \
    "${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c" \
    || die "kernel shadow-PTE parameter is missing"
  rg -q 'slot_direct_shadow_pte_prefill_calls' \
    "${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c" \
    || die "kernel shadow-PTE telemetry is missing"

  bash -n "${BASH_SOURCE[0]}"
  "${VLLM_PY}" - \
    "${BENCHMARK}" "${COMMON_BENCHMARK}" "${PREPARE_INPUTS}" "${VERIFY}" "${SELF_TEST}" <<'PY'
import ast
import sys
from pathlib import Path

for raw in sys.argv[1:]:
    path = Path(raw)
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("python syntax: OK")
PY
  "${VLLM_PY}" "${BENCHMARK}" --help >/dev/null
  "${VLLM_PY}" "${VERIFY}" --help >/dev/null
  PYTHONDONTWRITEBYTECODE=1 "${VLLM_PY}" "${SELF_TEST}"
  echo "static preflight: OK (no CUDA workload launched)"
}

snapshot_gpu() {
  local output="$1"
  local gpu_index="$2"
  {
    nvidia-smi -i "${gpu_index}" \
      --query-gpu=index,name,uuid,memory.used,memory.free,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>&1 || true
  } > "${output}"
}

gpu_clean_state() {
  local gpu_index="$1"
  local used_mib gpu_uuid process_count
  used_mib="$(nvidia-smi -i "${gpu_index}" --query-gpu=memory.used \
    --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]')"
  gpu_uuid="$(nvidia-smi -i "${gpu_index}" --query-gpu=uuid \
    --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]')"
  [[ "${used_mib}" =~ ^[0-9]+$ && -n "${gpu_uuid}" ]] || return 1
  process_count="$(nvidia-smi \
    --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' -v uuid="${gpu_uuid}" '$1 ~ uuid {count += 1} END {print count + 0}')"
  [[ "${used_mib}" == "0" && "${process_count}" == "0" ]]
}

require_clean_gpu_now() {
  local output="$1"
  local gpu_index="$2"
  snapshot_gpu "${output}" "${gpu_index}"
  gpu_clean_state "${gpu_index}" \
    || die "GPU ${gpu_index} is not at exactly 0 MiB with no compute process"
}

wait_clean_gpu() {
  local output="$1"
  local gpu_index="$2"
  local timeout_sec="$3"
  local elapsed
  for ((elapsed = 0; elapsed <= timeout_sec; ++elapsed)); do
    if gpu_clean_state "${gpu_index}"; then
      snapshot_gpu "${output}" "${gpu_index}"
      return 0
    fi
    sleep 1
  done
  snapshot_gpu "${output}" "${gpu_index}" || true
  nvidia-smi >&2 || true
  die "GPU ${gpu_index} did not return to exactly 0 MiB"
}

assert_module_contract() {
  local fence="$1"
  local metadata_recipe="$2"
  local overlap_copy_prepare="$3"
  local fast_validation="$4"
  local shadow_pte_prefill="$5"
  require_parameter uvm_moe_slot_pte_remap_enable 1
  require_parameter uvm_moe_slot_plan_overlap_copy_prepare "${overlap_copy_prepare}"
  require_parameter uvm_moe_slot_plan_register_pte_prewarm 1
  require_parameter uvm_moe_slot_plan_cache_copy_runs 1
  require_parameter uvm_moe_slot_plan_cache_metadata_recipe "${metadata_recipe}"
  require_parameter uvm_moe_slot_transition_fast_index 1
  require_parameter uvm_moe_slot_transition_direct_cutover 1
  require_parameter uvm_moe_slot_plan_fast_validation "${fast_validation}"
  require_parameter uvm_moe_slot_plan_shadow_pte_prefill "${shadow_pte_prefill}"
  require_parameter uvm_moe_slot_plan_dual_bank_event 1
  require_parameter uvm_moe_slot_plan_cuda_fence "${fence}"

  local budget_raw budget
  budget_raw="$(read_parameter uvm_moe_slot_pool_vidmem_budget_bytes)"
  budget="$(${VLLM_PY} -c 'import sys; print(int(sys.argv[1], 0))' "${budget_raw}")" \
    || die "invalid slot-pool budget: ${budget_raw}"
  (( budget >= SLOT_CACHE_BYTES )) \
    || die "slot-pool budget ${budget} is below two-bank requirement ${SLOT_CACHE_BYTES}"
}

write_module_state() {
  local output="$1"
  local loaded_srcversion="$2"
  local module_srcversion="$3"
  {
    echo "loaded_srcversion=${loaded_srcversion}"
    echo "custom_srcversion=${module_srcversion}"
    for name in \
      uvm_moe_slot_pte_remap_enable \
      uvm_moe_slot_pool_vidmem_budget_bytes \
      uvm_moe_slot_plan_overlap_copy_prepare \
      uvm_moe_slot_plan_register_pte_prewarm \
      uvm_moe_slot_plan_cache_copy_runs \
      uvm_moe_slot_plan_cache_metadata_recipe \
      uvm_moe_slot_transition_fast_index \
      uvm_moe_slot_transition_direct_cutover \
      uvm_moe_slot_plan_fast_validation \
      uvm_moe_slot_plan_shadow_pte_prefill \
      uvm_moe_slot_plan_dual_bank_event \
      uvm_moe_slot_plan_cuda_fence; do
      echo "${name}=$(read_parameter "${name}")"
    done
  } > "${output}"
}

validate_generated_plan() {
  local plan="$1"
  "${VLLM_PY}" - "${plan}" "${LAYER_BYTES}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
layer_bytes = int(sys.argv[2])
plan = json.loads(path.read_text(encoding="utf-8"))
cold = {(int(x["layer_id"]), int(x["expert_id"])) for x in plan["offload_plan"]}
hot = {(int(x["layer_id"]), int(x["expert_id"])) for x in plan["hot_plan"]}
expected_cold = {(layer, expert) for layer in range(4) for expert in range(128)}
expected_all = {(layer, expert) for layer in range(48) for expert in range(128)}
if cold != expected_cold or cold & hot or cold | hot != expected_all:
    raise SystemExit("generated plan is not exact Qwen L0-L3")
if plan.get("metadata", {}).get("actual_offload_main_bytes") != 4 * layer_bytes:
    raise SystemExit("generated plan has the wrong byte total")
print("fixed plan validation: OK")
PY
}

write_run_config() {
  local output="$1"
  local case_label="$2"
  local fence="$3"
  local metadata_recipe="$4"
  local overlap_copy_prepare="$5"
  local fast_validation="$6"
  local shadow_pte_prefill="$7"
  local model="$8"
  local fixture="$9"
  local plan="${10}"
  local module_srcversion="${11}"
  local loaded_srcversion="${12}"
  local gpu_index="${13}"
  "${VLLM_PY}" - "${output}" "${case_label}" "${fence}" \
    "${metadata_recipe}" "${overlap_copy_prepare}" "${fast_validation}" \
    "${shadow_pte_prefill}" "${model}" "${fixture}" \
    "${plan}" "${CUSTOM_MODULE}" "${module_srcversion}" "${loaded_srcversion}" \
    "${gpu_index}" "${BENCHMARK}" "${COMMON_BENCHMARK}" "${REPO_ROOT}" <<'PY'
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output, case_label, fence, metadata_recipe, overlap_copy_prepare,
    fast_validation, shadow_pte_prefill, model, fixture, plan, custom_module,
    module_srcversion, loaded_srcversion, gpu_index, benchmark,
    common_benchmark, repo_root,
) = sys.argv[1:]

def sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

try:
    commit = subprocess.run(
        ["git", "-C", repo_root, "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
except Exception:
    commit = None

value = {
    "schema": "vllm_slot_dual_bank_fence_e2e_run_v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "case_label": case_label,
    "fence": int(fence),
    "metadata_recipe": int(metadata_recipe),
    "overlap_copy_prepare": int(overlap_copy_prepare),
    "fast_validation": int(fast_validation),
    "shadow_pte_prefill": int(shadow_pte_prefill),
    "benchmark": {
        "script": str(Path(benchmark).resolve()),
        "common_benchmark": str(Path(common_benchmark).resolve()),
        "sequential_requests": 3,
        "model": str(Path(model).resolve()),
        "evaluation_fixture": str(Path(fixture).resolve()),
        "plan": str(Path(plan).resolve()),
    },
    "fixed_bank": {"bank_count": 2, "bytes": 1207959552, "slots": 2304},
    "inputs": {
        "model_config_sha256": sha(str(Path(model) / "config.json")),
        "evaluation_fixture_sha256": sha(fixture),
        "plan_sha256": sha(plan),
    },
    "module": {
        "path": str(Path(custom_module).resolve()),
        "sha256": sha(custom_module),
        "custom_srcversion": module_srcversion,
        "loaded_srcversion": loaded_srcversion,
    },
    "gpu_index": int(gpu_index),
    "repo_commit": commit,
    "synchronization": (
        "main-stream write -> fenced ioctl 96 -> main-stream wait"
        if int(fence)
        else "device synchronize -> legacy registered-plan ioctl 93 -> device synchronize"
    ),
    "compute_copy_overlap": False,
}
Path(output).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_case() {
  [[ $# -eq 2 ]] || die "run requires CASE_LABEL and FENCE"
  [[ "${ALLOW_GPU_RUN:-0}" == "1" ]] || die "run requires ALLOW_GPU_RUN=1"
  local case_label="$1"
  local fence="$2"
  local metadata_recipe="${METADATA_RECIPE:-1}"
  local overlap_copy_prepare="${OVERLAP_COPY_PREPARE:-1}"
  local fast_validation="${FAST_VALIDATION:-0}"
  local shadow_pte_prefill="${SHADOW_PTE_PREFILL:-0}"
  local model="${MODEL:-${DEFAULT_MODEL}}"
  local fixture="${EVAL_FIXTURE:-${DEFAULT_FIXTURE}}"
  local gpu_index="${GPU_INDEX:-0}"
  local cpuset="${CPUSET:-0-13}"
  local cpu_threads="${CPU_THREADS:-14}"
  local output_tokens="${OUTPUT_TOKENS:-4}"
  local kv_cache_bytes="${KV_CACHE_BYTES:-1288490189}"
  local max_model_len="${MAX_MODEL_LEN:-544}"
  local max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-2048}"
  local max_num_seqs="${MAX_NUM_SEQS:-32}"
  local gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.97}"
  local timeout_sec="${TIMEOUT_SEC:-3600}"
  local clean_timeout="${GPU_CLEAN_TIMEOUT_SEC:-180}"
  local run_root="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_${case_label}_fence${fence}_recipe${metadata_recipe}_fast${fast_validation}_shadow${shadow_pte_prefill}}"

  [[ "${case_label}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || die "CASE_LABEL must match [A-Za-z0-9][A-Za-z0-9._-]*"
  validate_bit FENCE "${fence}"
  validate_bit METADATA_RECIPE "${metadata_recipe}"
  validate_bit OVERLAP_COPY_PREPARE "${overlap_copy_prepare}"
  validate_bit FAST_VALIDATION "${fast_validation}"
  validate_bit SHADOW_PTE_PREFILL "${shadow_pte_prefill}"
  if [[ "${shadow_pte_prefill}" == "1" && "${overlap_copy_prepare}" != "0" ]]; then
    die "SHADOW_PTE_PREFILL=1 requires OVERLAP_COPY_PREPARE=0"
  fi
  [[ "${gpu_index}" =~ ^[0-9]+$ ]] || die "GPU_INDEX must be non-negative"
  [[ "${cpu_threads}" =~ ^[1-9][0-9]*$ ]] || die "CPU_THREADS must be positive"
  [[ "${output_tokens}" =~ ^[1-9][0-9]*$ ]] || die "OUTPUT_TOKENS must be positive"
  [[ "${clean_timeout}" =~ ^[1-9][0-9]*$ ]] \
    || die "GPU_CLEAN_TIMEOUT_SEC must be positive"

  static_preflight
  require_command nvidia-smi
  require_command modinfo
  require_command timeout
  require_command taskset
  require_command flock
  require_command sha256sum
  require_file "${CUSTOM_MODULE}"
  require_dir "${model}"
  require_file "${model}/config.json"
  require_file "${fixture}"
  [[ -r "${STATS_PATH}" ]] || die "latency stats are unavailable: ${STATS_PATH}"
  taskset -c "${cpuset}" true >/dev/null 2>&1 || die "invalid CPUSET=${cpuset}"

  local module_srcversion loaded_srcversion expected_srcversion
  module_srcversion="$(modinfo -F srcversion "${CUSTOM_MODULE}" 2>/dev/null || true)"
  loaded_srcversion="$(sed -n '1p' /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)"
  expected_srcversion="${EXPECTED_CUSTOM_SRCVERSION:-${module_srcversion}}"
  [[ -n "${module_srcversion}" && -n "${loaded_srcversion}" ]] \
    || die "custom/loaded module srcversion is unavailable"
  [[ "${module_srcversion}" == "${expected_srcversion}" ]] \
    || die "custom module does not match EXPECTED_CUSTOM_SRCVERSION"
  [[ "${loaded_srcversion}" == "${expected_srcversion}" ]] \
    || die "loaded UVM is not the requested custom module"
  assert_module_contract "${fence}" "${metadata_recipe}" "${overlap_copy_prepare}" \
    "${fast_validation}" "${shadow_pte_prefill}"
  for counter in \
    slot_plan_register_calls \
    slot_plan_execute_calls \
    slot_direct_copy_run_cache_hits \
    slot_direct_metadata_recipe_cache_attempts \
    slot_direct_target_recipe_cache_hits \
    slot_direct_clear_recipe_cache_hits \
    slot_direct_shadow_pte_prefill_calls \
    slot_direct_shadow_pte_prefill_blocks; do
    rg -q "^${counter}[[:space:]]" "${STATS_PATH}" \
      || die "loaded UVM lacks telemetry counter ${counter}"
  done
  rg -q '^  shadow_pte_prefill[[:space:]]+calls=' "${STATS_PATH}" \
    || die "loaded UVM lacks shadow-PTE prefill profile telemetry"
  rg -q '^  shadow_pte_publish[[:space:]]+calls=' "${STATS_PATH}" \
    || die "loaded UVM lacks shadow-PTE publish profile telemetry"

  exec 9>"/tmp/vllm_slot_dual_bank_fence_e2e.lock"
  flock -n 9 || die "another dual-bank fence E2E case is active"
  [[ ! -e "${run_root}" ]] || die "RUN_ROOT already exists: ${run_root}"
  mkdir -p "$(dirname "${run_root}")"
  mkdir "${run_root}"
  run_root="$(cd "${run_root}" && pwd)"
  mkdir "${run_root}/inputs"
  require_clean_gpu_now "${run_root}/gpu.before.csv" "${gpu_index}"

  "${VLLM_PY}" "${PREPARE_INPUTS}" --output-dir "${run_root}/inputs" \
    > "${run_root}/prepare_inputs.log"
  local plan="${run_root}/inputs/exact_l0_l3_plan.json"
  validate_generated_plan "${plan}" > "${run_root}/plan_validation.txt"
  cp -- "${STATS_PATH}" "${run_root}/latency_stats.before.txt"
  : > "${run_root}/slot_trace.jsonl"
  write_module_state "${run_root}/module_state.before.env" \
    "${loaded_srcversion}" "${module_srcversion}"
  write_run_config "${run_root}/run_config.json" "${case_label}" "${fence}" \
    "${metadata_recipe}" "${overlap_copy_prepare}" "${fast_validation}" \
    "${shadow_pte_prefill}" "${model}" "${fixture}" "${plan}" \
    "${module_srcversion}" "${loaded_srcversion}" "${gpu_index}"
  git -C "${REPO_ROOT}" status --short > "${run_root}/git_status.txt"
  modinfo "${CUSTOM_MODULE}" > "${run_root}/custom_module.modinfo.txt"
  sha256sum "${CUSTOM_MODULE}" "${fixture}" "${plan}" "${model}/config.json" \
    > "${run_root}/input_sha256.txt"

  local -a env_args=(
    VLLM_USE_UVM=0
    VLLM_UVM_COLD_SLOT_POOL_DRIVER_MODE=slot_va_block_remap
    VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP=1
    VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0
    VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0
    VLLM_UVM_KV_CACHE_MANAGED=0
    VLLM_UVM_KV_RUNTIME_ENABLE=0
    VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0
    VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
    VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0
    VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0
    VLLM_UVM_POST_WARMUP_REBALANCE_FAIL_FAST=1
    VLLM_UVM_WEIGHT_EXECUTOR=cpp_batch
    VLLM_UVM_PREFILL_ORACLE_PREFETCH_MODE=off
    VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0
    VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST=
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
    VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1
    VLLM_UVM_EXPERT_ENVELOPE_BYTES=524288
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup
    VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${plan}"
    VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN=1
    VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=slot_restore
    VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD=1
    VLLM_UVM_SLOT_DIRECT_INIT_BANK=1
    VLLM_UVM_SLOT_DIRECT_REUSE_BANK=1
    VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES=1
    VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION=1
    VLLM_UVM_SLOT_DIRECT_SHARED_PUSH=1
    VLLM_UVM_SLOT_DIRECT_64K_PTE=1
    VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN=1
    VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT=1
    VLLM_UVM_SLOT_DIRECT_CUDA_FENCE="${fence}"
    VLLM_UVM_SLOT_DIRECT_CUDA_FENCE_KERNEL_LIB="${CUDA_FENCE_KERNEL_LIB}"
    VLLM_UVM_SLOT_DIRECT_CACHE_BYTES="${SLOT_CACHE_BYTES}"
    VLLM_UVM_SLOT_DIRECT_MAX_SLOTS="${SLOT_CACHE_MAX_SLOTS}"
    VLLM_UVM_SLOT_DIRECT_TRACE_FILE="${run_root}/slot_trace.jsonl"
    VLLM_ENABLE_V1_MULTIPROCESSING=0
    VLLM_CPU_OFFLOAD_BACKEND=uva
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    TOKENIZERS_PARALLELISM=false
    OMP_NUM_THREADS="${cpu_threads}"
    MKL_NUM_THREADS="${cpu_threads}"
    CUDA_VISIBLE_DEVICES="${gpu_index}"
  )
  printf '%s\n' "${env_args[@]}" > "${run_root}/environment.txt"

  local profile_label="dual_bank_${case_label}_fence${fence}_recipe${metadata_recipe}_fast${fast_validation}_shadow${shadow_pte_prefill}"
  local mechanism_class="vllm_registered_plan_dual_bank_fence${fence}_recipe${metadata_recipe}_fast${fast_validation}_shadow${shadow_pte_prefill}"
  local -a benchmark_args=(
    --common-benchmark "${COMMON_BENCHMARK}"
    --model "${model}"
    --evaluation-fixture "${fixture}"
    --output "${run_root}/result.json"
    --profile-label "${profile_label}"
    --mechanism-class "${mechanism_class}"
    --sequential-requests "${SEQUENTIAL_REQUESTS}"
    --fixture-request-index 0
    --output-tokens "${output_tokens}"
    --kv-cache-memory-bytes "${kv_cache_bytes}"
    --max-model-len "${max_model_len}"
    --max-num-batched-tokens "${max_num_batched_tokens}"
    --max-num-seqs "${max_num_seqs}"
    --gpu-memory-utilization "${gpu_memory_utilization}"
    --seed 0
  )
  local -a timing_prefix=()
  if [[ -x /usr/bin/time ]]; then
    timing_prefix=(/usr/bin/time -v -o "${run_root}/resource_time.txt")
  fi
  {
    printf 'cd %q && env ' "${VLLM_ROOT}"
    printf '%q ' "${env_args[@]}"
    printf '%q ' "${timing_prefix[@]}" timeout "${timeout_sec}s" \
      taskset -c "${cpuset}" "${VLLM_PY}" "${BENCHMARK}" "${benchmark_args[@]}"
    printf '\n'
  } > "${run_root}/command.txt"
  {
    echo "case_label=${case_label}"
    echo "fence=${fence}"
    echo "metadata_recipe=${metadata_recipe}"
    echo "fast_validation=${fast_validation}"
    echo "shadow_pte_prefill=${shadow_pte_prefill}"
    echo "started_at=$(date --iso-8601=seconds)"
  } > "${run_root}/execution.env"

  local benchmark_rc
  set +e
  (
    cd "${VLLM_ROOT}"
    env "${env_args[@]}" "${timing_prefix[@]}" \
      timeout "${timeout_sec}s" taskset -c "${cpuset}" \
      "${VLLM_PY}" "${BENCHMARK}" "${benchmark_args[@]}"
  ) > "${run_root}/run.log" 2>&1
  benchmark_rc=$?
  set -e
  cp -- "${STATS_PATH}" "${run_root}/latency_stats.after.txt"
  printf '%s\n' "${benchmark_rc}" > "${run_root}/returncode.txt"
  {
    echo "ended_at=$(date --iso-8601=seconds)"
    echo "returncode=${benchmark_rc}"
  } >> "${run_root}/execution.env"

  assert_module_contract "${fence}" "${metadata_recipe}" "${overlap_copy_prepare}" \
    "${fast_validation}" "${shadow_pte_prefill}"
  write_module_state "${run_root}/module_state.after.env" \
    "${loaded_srcversion}" "${module_srcversion}"
  wait_clean_gpu "${run_root}/gpu.after.csv" "${gpu_index}" "${clean_timeout}"

  local verify_rc
  set +e
  "${VLLM_PY}" "${VERIFY}" "${run_root}" --fence "${fence}" \
    --metadata-recipe "${metadata_recipe}" --fast-validation "${fast_validation}" \
    --shadow-pte-prefill "${shadow_pte_prefill}" \
    --output "${run_root}/verification.json" \
    > "${run_root}/verify.log" 2> "${run_root}/verify.err"
  verify_rc=$?
  set -e
  echo "RUN_ROOT=${run_root}"
  if (( benchmark_rc != 0 || verify_rc != 0 )); then
    tail -n 120 "${run_root}/run.log" >&2 || true
    cat "${run_root}/verify.err" >&2 || true
    cat "${run_root}/verify.log" >&2 || true
    die "case failed: benchmark_rc=${benchmark_rc} verifier_rc=${verify_rc}"
  fi
  echo "verification=${run_root}/verification.json"
}

case "${COMMAND}" in
  static-preflight|preflight)
    [[ $# -eq 0 ]] || die "static-preflight takes no arguments"
    static_preflight
    ;;
  run)
    run_case "$@"
    ;;
  verify)
    [[ $# -ge 2 && $# -le 5 ]] \
      || die "verify requires RUN_DIR FENCE [METADATA_RECIPE [FAST_VALIDATION [SHADOW_PTE_PREFILL]]]"
    "${VLLM_PY}" "${VERIFY}" "$1" --fence "$2" \
      --metadata-recipe "${3:-1}" --fast-validation "${4:-0}" \
      --shadow-pte-prefill "${5:-0}" --output "$1/verification.json"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    die "unknown command: ${COMMAND}"
    ;;
esac
