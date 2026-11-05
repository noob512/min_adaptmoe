#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_PY:-${VLLM_ROOT}/.venv/bin/python}"
BENCHMARK="${BENCHMARK:-${REPO_ROOT}/experiments/broad_offload_compare/benchmark_custom_offload.py}"
SEQUENTIAL_BENCHMARK="${SEQUENTIAL_BENCHMARK:-${SCRIPT_DIR}/benchmark_sequential_requests.py}"
VERIFIER="${SCRIPT_DIR}/verify_results.py"
SELF_TEST="${SCRIPT_DIR}/test_verify_results.py"
CUSTOM_MODULE="${CUSTOM_MODULE:-${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"
SLOT_ENABLE_PATH="${SLOT_ENABLE_PATH:-/sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable}"
SLOT_BUDGET_PATH="${SLOT_BUDGET_PATH:-/sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes}"
MIN_REQUIRED_SLOT_BUDGET_BYTES=$((5 * 1024 * 1024 * 1024 / 2))

DEFAULT_MODEL="${REPO_ROOT}/../.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782"
DEFAULT_FIXTURE="${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json"
DEFAULT_PLAN="${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/plans/kt_exact_custom_remote_plan.json"

COMMAND="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

ACTIVE_MONITOR_PID=""

cleanup_monitor() {
  if [[ -n "${ACTIVE_MONITOR_PID}" ]]; then
    kill "${ACTIVE_MONITOR_PID}" 2>/dev/null || true
    wait "${ACTIVE_MONITOR_PID}" 2>/dev/null || true
    ACTIVE_MONITOR_PID=""
  fi
}
trap cleanup_monitor EXIT INT TERM

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

usage_text() {
  cat <<'EOF'
Usage:
  run_integration.sh static-preflight
  run_integration.sh run
  run_integration.sh run-cache
  run_integration.sh verify /path/to/run_root

The runner never loads, unloads, or changes an NVIDIA module. The caller must
load the custom module with slot remap enabled and at least 2.5 GiB slot budget.

run environment (defaults shown):
  ALLOW_GPU_RUN=1                         mandatory acknowledgement
  ORDER=AB                                A=generic restore, B=slot restore; AB|BA
  MODEL=<Qwen3-30B-A3B-FP8 snapshot>
  EVAL_FIXTURE=<frozen evaluation token-ID fixture>
  PLAN=<kt_exact_custom_remote_plan.json>
  RUN_ROOT=<this-dir>/artifacts/<UTC>_<order>
  CONCURRENCY=1 OUTPUT_TOKENS=4 WARMUP_REQUESTS=0
  KV_CACHE_BYTES=1288490189 MAX_MODEL_LEN=544
  MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=32
  GPU_MEMORY_UTILIZATION=0.97 TIMEOUT_SEC=3600
  GPU_INDEX=0 GPU_CLEAN_MAX_MIB=0 GPU_CLEAN_TIMEOUT_SEC=180
  CPUSET=0-13 CPU_THREADS=14
  CUSTOM_MODULE=/path/to/custom/nvidia-uvm.ko
  EXPECTED_CUSTOM_SRCVERSION=<optional pinned srcversion>

run-cache additionally uses one LLM instance for repeated sequential generate:
  SEQUENTIAL_REQUESTS=2
  SLOT_CACHE_BYTES=268435456             256 MiB userspace cache
  SLOT_CACHE_MAX_SLOTS=512
EOF
}

static_preflight() {
  require_dir "${VLLM_ROOT}"
  require_file "${VLLM_PY}"
  require_file "${BENCHMARK}"
  require_file "${SEQUENTIAL_BENCHMARK}"
  require_file "${VERIFIER}"
  require_file "${SELF_TEST}"
  require_file "${VLLM_ROOT}/vllm/device_allocator/uvm.py"
  require_file "${VLLM_ROOT}/vllm/device_allocator/uvm_pool_residency.py"
  require_file "${VLLM_ROOT}/vllm/model_executor/layers/fused_moe/layer.py"
  command -v rg >/dev/null || die "rg is unavailable"
  command -v timeout >/dev/null || die "timeout is unavailable"
  command -v taskset >/dev/null || die "taskset is unavailable"
  command -v sha256sum >/dev/null || die "sha256sum is unavailable"

  rg -q '^def driver_slot_preallocate_ranges\(' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm.py" \
    || die "mode-8 userspace API is missing"
  rg -q '^def driver_slot_restore_ranges\(' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm.py" \
    || die "mode-10 userspace API is missing"
  rg -q '^def restore_uvm_regions_to_slots\(' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_pool_residency.py" \
    || die "slot restore policy is missing"
  rg -q 'VLLM_UVM_SLOT_DIRECT_TRACE_FILE' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_pool_residency.py" \
    || die "slot-direct trace hook is missing"
  rg -q '"slot_restore"' \
    "${VLLM_ROOT}/vllm/model_executor/layers/fused_moe/layer.py" \
    || die "FusedMoE slot_restore access mode is missing"

  bash -n "${BASH_SOURCE[0]}"
  "${VLLM_PY}" - "${BENCHMARK}" "${SEQUENTIAL_BENCHMARK}" \
    "${VERIFIER}" "${SELF_TEST}" <<'PY'
import ast
import sys
from pathlib import Path

for raw in sys.argv[1:]:
    path = Path(raw)
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("python syntax: OK")
PY
  "${VLLM_PY}" "${BENCHMARK}" --help >/dev/null
  "${VLLM_PY}" "${SEQUENTIAL_BENCHMARK}" --help >/dev/null
  "${VLLM_PY}" "${VERIFIER}" --help >/dev/null
  "${VLLM_PY}" "${SELF_TEST}"
  echo "static preflight: OK (no CUDA workload launched)"
}

cache_source_preflight() {
  rg -q 'VLLM_UVM_SLOT_DIRECT_CACHE_BYTES' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_pool_residency.py" \
    || die "userspace slot cache byte-limit hook is missing"
  rg -q 'VLLM_UVM_SLOT_DIRECT_MAX_SLOTS' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_pool_residency.py" \
    || die "userspace slot cache slot-limit hook is missing"
  rg -q '"operation": "release"' \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_pool_residency.py" \
    || die "mode19 release trace hook is missing"
}

snapshot_gpu() {
  local output="$1"
  nvidia-smi \
    --query-gpu=index,name,uuid,memory.used,memory.free,memory.total,utilization.gpu \
    --format=csv,noheader,nounits > "${output}"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits >> "${output}" 2>&1 || true
}

wait_for_clean_gpu() {
  local snapshot="$1"
  local gpu_index="$2"
  local max_mib="$3"
  local timeout_seconds="$4"
  local used_mib=""
  local pids=""
  local elapsed
  for ((elapsed = 0; elapsed <= timeout_seconds; ++elapsed)); do
    used_mib="$(nvidia-smi -i "${gpu_index}" --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d '[:space:]')"
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
      2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1}')"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] \
        && (( used_mib <= max_mib )) \
        && [[ -z "${pids}" ]]; then
      snapshot_gpu "${snapshot}"
      return 0
    fi
    sleep 1
  done
  snapshot_gpu "${snapshot}" || true
  nvidia-smi >&2 || true
  die "GPU is not clean after ${timeout_seconds}s (used=${used_mib:-unknown} MiB, pids=${pids:-none})"
}

validate_inputs() {
  local model="$1"
  local fixture="$2"
  local plan="$3"
  require_dir "${model}"
  require_file "${model}/config.json"
  require_file "${fixture}"
  require_file "${plan}"
  "${VLLM_PY}" - "${fixture}" "${plan}" <<'PY'
import json
import sys
from pathlib import Path

fixture_path, plan_path = map(Path, sys.argv[1:])
fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
requests = fixture.get("requests") if isinstance(fixture, dict) else None
if not isinstance(requests, list) or not requests:
    raise SystemExit("evaluation fixture has no requests")
for index, request in enumerate(requests):
    if not isinstance(request, dict):
        raise SystemExit(f"fixture request {index} is not an object")
    count = request.get("prompt_tokens")
    ids = request.get("prompt_token_ids")
    if not isinstance(count, int) or isinstance(count, bool):
        raise SystemExit(f"fixture request {index} has invalid prompt_tokens")
    if not isinstance(ids, list) or len(ids) != count:
        raise SystemExit(f"fixture request {index} lacks exact prompt token IDs")

plan = json.loads(plan_path.read_text(encoding="utf-8"))
cold = plan.get("offload_plan") if isinstance(plan, dict) else None
hot = plan.get("hot_plan", plan.get("prefetch_plan")) if isinstance(plan, dict) else None
if not isinstance(cold, list) or not isinstance(hot, list) or not cold:
    raise SystemExit("plan must contain nonempty cold and complete hot lists")

def coordinates(records, label):
    result = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SystemExit(f"{label}[{index}] is not an object")
        layer = record.get("layer_id")
        expert = record.get("expert_id")
        if not isinstance(layer, int) or isinstance(layer, bool) or not 0 <= layer < 48:
            raise SystemExit(f"{label}[{index}] has invalid layer_id")
        if not isinstance(expert, int) or isinstance(expert, bool) or not 0 <= expert < 128:
            raise SystemExit(f"{label}[{index}] has invalid expert_id")
        result.append((layer, expert))
    if len(result) != len(set(result)):
        raise SystemExit(f"{label} has duplicate coordinates")
    return set(result)

cold_keys = coordinates(cold, "offload_plan")
hot_keys = coordinates(hot, "hot_plan")
expected = {(layer, expert) for layer in range(48) for expert in range(128)}
if cold_keys & hot_keys or cold_keys | hot_keys != expected:
    raise SystemExit("plan does not partition the complete Qwen 48x128 topology")
remote_bytes = plan.get("metadata", {}).get("actual_offload_main_bytes")
if not isinstance(remote_bytes, int) or remote_bytes <= 0:
    raise SystemExit("plan lacks positive metadata.actual_offload_main_bytes")
print(f"inputs: requests={len(requests)} cold={len(cold)} hot={len(hot)}")
PY
}

runtime_preflight() {
  local model="$1"
  local fixture="$2"
  local plan="$3"
  local gpu_index="$4"
  local cpuset="$5"
  static_preflight
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  command -v modinfo >/dev/null || die "modinfo is unavailable"
  command -v flock >/dev/null || die "flock is unavailable"
  require_file "${CUSTOM_MODULE}"
  [[ -r "${STATS_PATH}" ]] || die "latency stats is not readable: ${STATS_PATH}"
  [[ -r "${SLOT_ENABLE_PATH}" ]] || die "slot enable parameter is unavailable"
  [[ -r "${SLOT_BUDGET_PATH}" ]] || die "slot budget parameter is unavailable"
  taskset -c "${cpuset}" true >/dev/null 2>&1 || die "invalid CPUSET=${cpuset}"
  [[ "${gpu_index}" =~ ^[0-9]+$ ]] || die "GPU_INDEX must be a non-negative integer"
  validate_inputs "${model}" "${fixture}" "${plan}"

  local loaded_srcversion module_srcversion pinned_srcversion enable budget_raw budget_bytes
  loaded_srcversion="$(sed -n '1p' /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)"
  module_srcversion="$(modinfo -F srcversion "${CUSTOM_MODULE}" 2>/dev/null || true)"
  pinned_srcversion="${EXPECTED_CUSTOM_SRCVERSION:-${module_srcversion}}"
  [[ -n "${loaded_srcversion}" && -n "${module_srcversion}" ]] \
    || die "loaded/custom module srcversion is unavailable"
  [[ "${module_srcversion}" == "${pinned_srcversion}" ]] \
    || die "custom module srcversion does not match EXPECTED_CUSTOM_SRCVERSION"
  [[ "${loaded_srcversion}" == "${pinned_srcversion}" ]] \
    || die "loaded UVM is not the requested custom module: loaded=${loaded_srcversion} expected=${pinned_srcversion}"
  enable="$(tr -d '[:space:]' < "${SLOT_ENABLE_PATH}")"
  [[ "${enable}" == "1" ]] || die "uvm_moe_slot_pte_remap_enable must equal 1"
  budget_raw="$(tr -d '[:space:]' < "${SLOT_BUDGET_PATH}")"
  budget_bytes="$("${VLLM_PY}" -c 'import sys; print(int(sys.argv[1], 0))' "${budget_raw}")" \
    || die "invalid slot budget: ${budget_raw}"
  [[ "${budget_bytes}" =~ ^[0-9]+$ ]] || die "invalid slot budget: ${budget_raw}"
  (( budget_bytes >= MIN_REQUIRED_SLOT_BUDGET_BYTES )) \
    || die "slot budget ${budget_bytes} is below required ${MIN_REQUIRED_SLOT_BUDGET_BYTES} bytes (2.5 GiB)"
}

write_run_config() {
  local output="$1"
  local order="$2"
  local generic_dir="$3"
  local slot_dir="$4"
  local model="$5"
  local fixture="$6"
  local plan="$7"
  local concurrency="$8"
  local output_tokens="$9"
  shift 9
  local kv_cache_bytes="$1"
  local max_model_len="$2"
  local max_num_batched_tokens="$3"
  local max_num_seqs="$4"
  local gpu_memory_utilization="$5"
  local module_srcversion="$6"
  local loaded_srcversion="$7"
  local budget_bytes="$8"
  local gpu_index="$9"
  shift 9
  local experiment_kind="$1"
  local sequential_requests="$2"
  local cache_bytes="$3"
  local cache_max_slots="$4"
  local benchmark_script="${BENCHMARK}"
  [[ "${experiment_kind}" == "cache_reuse" ]] \
    && benchmark_script="${SEQUENTIAL_BENCHMARK}"
  "${VLLM_PY}" - "${output}" "${order}" "${generic_dir}" "${slot_dir}" \
    "${model}" "${fixture}" "${plan}" "${concurrency}" "${output_tokens}" \
    "${kv_cache_bytes}" "${max_model_len}" "${max_num_batched_tokens}" \
    "${max_num_seqs}" "${gpu_memory_utilization}" "${module_srcversion}" \
    "${loaded_srcversion}" "${budget_bytes}" "${gpu_index}" "${CUSTOM_MODULE}" \
    "${benchmark_script}" "${STATS_PATH}" "${REPO_ROOT}" \
    "${experiment_kind}" "${sequential_requests}" "${cache_bytes}" \
    "${cache_max_slots}" <<'PY'
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output, order, generic_dir, slot_dir, model, fixture, plan,
    concurrency, output_tokens, kv_cache_bytes, max_model_len,
    max_num_batched_tokens, max_num_seqs, gpu_memory_utilization,
    module_srcversion, loaded_srcversion, budget_bytes, gpu_index,
    custom_module, benchmark, stats_path, repo_root, experiment_kind,
    sequential_requests, cache_bytes, cache_max_slots,
) = sys.argv[1:]

def sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

repo = Path(repo_root).resolve()
try:
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
except Exception:
    commit = None
value = {
    "schema": "vllm_slot_direct_integration_run_v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "experiment_kind": experiment_kind,
    "order": order,
    "arm_aliases": {"A": "generic", "B": "slot"},
    "arm_dirs": {"generic": generic_dir, "slot": slot_dir},
    "benchmark": {
        "script": str(Path(benchmark).resolve()),
        "model": str(Path(model).resolve()),
        "evaluation_fixture": str(Path(fixture).resolve()),
        "plan": str(Path(plan).resolve()),
        "concurrency": int(concurrency),
        "output_tokens": int(output_tokens),
        "warmup_requests": 0,
        "sequential_requests": int(sequential_requests),
        "kv_cache_bytes": int(kv_cache_bytes),
        "max_model_len": int(max_model_len),
        "max_num_batched_tokens": int(max_num_batched_tokens),
        "max_num_seqs": int(max_num_seqs),
        "gpu_memory_utilization": float(gpu_memory_utilization),
    },
    "slot_cache": {
        "bytes": int(cache_bytes),
        "max_slots": int(cache_max_slots),
    } if experiment_kind == "cache_reuse" else None,
    "inputs": {
        "model_config_sha256": sha(str(Path(model) / "config.json")),
        "evaluation_fixture_sha256": sha(fixture),
        "plan_sha256": sha(plan),
    },
    "module": {
        "path": str(Path(custom_module).resolve()),
        "module_sha256": sha(custom_module),
        "custom_srcversion": module_srcversion,
        "loaded_srcversion": loaded_srcversion,
        "slot_enable": 1,
        "slot_budget_bytes": int(budget_bytes),
        "minimum_required_slot_budget_bytes": 5 * 1024**3 // 2,
        "latency_stats_path": stats_path,
    },
    "gpu_index": int(gpu_index),
    "repo_commit": commit,
    "process_semantics": (
        "one independent cold process per arm; one LLM load and repeated sequential generate calls"
        if experiment_kind == "cache_reuse"
        else "one independent cold process per arm; no application warmup"
    ),
}
Path(output).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_arm() {
  local arm="$1"
  local ordinal="$2"
  local arm_dir="$3"
  local model="$4"
  local fixture="$5"
  local plan="$6"
  local concurrency="$7"
  local output_tokens="$8"
  local kv_cache_bytes="$9"
  shift 9
  local max_model_len="$1"
  local max_num_batched_tokens="$2"
  local max_num_seqs="$3"
  local gpu_memory_utilization="$4"
  local timeout_sec="$5"
  local gpu_index="$6"
  local clean_mib="$7"
  local clean_timeout="$8"
  local cpuset="$9"
  shift 9
  local cpu_threads="$1"
  local experiment_kind="$2"
  local sequential_requests="$3"
  local cache_bytes="$4"
  local cache_max_slots="$5"
  local access_mode profile_label mechanism_class
  if [[ "${arm}" == "generic" ]]; then
    access_mode="restore"
    profile_label="generic_restore"
    mechanism_class="vllm_generic_uvm_restore"
    [[ "${experiment_kind}" == "cache_reuse" ]] \
      && mechanism_class="vllm_generic_uvm_restore_sequential"
  elif [[ "${arm}" == "slot" ]]; then
    access_mode="slot_restore"
    profile_label="slot_restore"
    mechanism_class="vllm_slot_direct_mode8_mode10_restore"
    [[ "${experiment_kind}" == "cache_reuse" ]] \
      && mechanism_class="vllm_bounded_slot_cache_mode8_mode10_mode19"
  else
    die "unknown arm: ${arm}"
  fi

  mkdir "${arm_dir}"
  : > "${arm_dir}/slot_trace.jsonl"
  wait_for_clean_gpu "${arm_dir}/gpu.before.csv" "${gpu_index}" "${clean_mib}" "${clean_timeout}"
  cp -- "${STATS_PATH}" "${arm_dir}/latency_stats.before.txt"
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
    VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE="${access_mode}"
    VLLM_UVM_SLOT_DIRECT_TRACE_FILE="${arm_dir}/slot_trace.jsonl"
    VLLM_ENABLE_V1_MULTIPROCESSING=0
    VLLM_CPU_OFFLOAD_BACKEND=uva
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    TOKENIZERS_PARALLELISM=false
    OMP_NUM_THREADS="${cpu_threads}"
    MKL_NUM_THREADS="${cpu_threads}"
    CUDA_VISIBLE_DEVICES="${gpu_index}"
  )
  if [[ "${experiment_kind}" == "cache_reuse" ]]; then
    env_args+=(
      VLLM_UVM_SLOT_DIRECT_CACHE_BYTES="${cache_bytes}"
    VLLM_UVM_SLOT_DIRECT_MAX_SLOTS="${cache_max_slots}"
    )
  fi
  printf '%s\n' "${env_args[@]}" > "${arm_dir}/environment.txt"

  local selected_benchmark="${BENCHMARK}"
  local -a benchmark_args=()
  if [[ "${experiment_kind}" == "cache_reuse" ]]; then
    selected_benchmark="${SEQUENTIAL_BENCHMARK}"
    benchmark_args=(
      --common-benchmark "${BENCHMARK}"
      --model "${model}"
      --evaluation-fixture "${fixture}"
      --output "${arm_dir}/result.json"
      --profile-label "${profile_label}"
      --mechanism-class "${mechanism_class}"
      --sequential-requests "${sequential_requests}"
      --fixture-request-index 0
      --output-tokens "${output_tokens}"
      --kv-cache-memory-bytes "${kv_cache_bytes}"
      --max-model-len "${max_model_len}"
      --max-num-batched-tokens "${max_num_batched_tokens}"
      --max-num-seqs "${max_num_seqs}"
      --gpu-memory-utilization "${gpu_memory_utilization}"
      --seed 0
    )
  else
    benchmark_args=(
      --model "${model}"
      --evaluation-fixture "${fixture}"
      --output "${arm_dir}/result.json"
      --profile-label "${profile_label}"
      --mechanism-class "${mechanism_class}"
      --concurrency "${concurrency}"
      --output-tokens "${output_tokens}"
      --warmup-requests 0
      --warmup-output-tokens 1
      --kv-cache-memory-bytes "${kv_cache_bytes}"
      --max-model-len "${max_model_len}"
      --max-num-batched-tokens "${max_num_batched_tokens}"
      --max-num-seqs "${max_num_seqs}"
      --gpu-memory-utilization "${gpu_memory_utilization}"
      --seed 0
      --require-prompt-token-ids
    )
  fi
  local -a timing_prefix=()
  if [[ -x /usr/bin/time ]]; then
    timing_prefix=(/usr/bin/time -v -o "${arm_dir}/resource_time.txt")
  fi
  {
    printf 'cd %q && env ' "${VLLM_ROOT}"
    printf '%q ' "${env_args[@]}"
    printf '%q ' "${timing_prefix[@]}" timeout "${timeout_sec}s" \
      taskset -c "${cpuset}" "${VLLM_PY}" "${selected_benchmark}" "${benchmark_args[@]}"
    printf '\n'
  } > "${arm_dir}/command.txt"
  {
    echo "ordinal=${ordinal}"
    echo "arm=${arm}"
    echo "started_at=$(date --iso-8601=seconds)"
  } > "${arm_dir}/execution.env"

  nvidia-smi dmon -i "${gpu_index}" -s pucvmet -d 1 -o DT \
    > "${arm_dir}/gpu_dmon.log" 2>&1 &
  ACTIVE_MONITOR_PID="$!"
  local rc
  set +e
  (
    cd "${VLLM_ROOT}"
    env "${env_args[@]}" "${timing_prefix[@]}" \
      timeout "${timeout_sec}s" taskset -c "${cpuset}" \
      "${VLLM_PY}" "${selected_benchmark}" "${benchmark_args[@]}"
  ) > "${arm_dir}/run.log" 2>&1
  rc=$?
  set -e
  cleanup_monitor
  cp -- "${STATS_PATH}" "${arm_dir}/latency_stats.after.txt"
  printf '%s\n' "${rc}" > "${arm_dir}/returncode.txt"
  {
    echo "ended_at=$(date --iso-8601=seconds)"
    echo "returncode=${rc}"
  } >> "${arm_dir}/execution.env"
  wait_for_clean_gpu "${arm_dir}/gpu.after.csv" "${gpu_index}" "${clean_mib}" "${clean_timeout}"
  return "${rc}"
}

run_experiment() {
  local experiment_kind="${1:-cold_pair}"
  [[ "${ALLOW_GPU_RUN:-0}" == "1" ]] \
    || die "run requires ALLOW_GPU_RUN=1"
  local order="${ORDER:-AB}"
  local model="${MODEL:-${DEFAULT_MODEL}}"
  local fixture="${EVAL_FIXTURE:-${DEFAULT_FIXTURE}}"
  local plan="${PLAN:-${DEFAULT_PLAN}}"
  local concurrency="${CONCURRENCY:-1}"
  local output_tokens="${OUTPUT_TOKENS:-4}"
  local warmup_requests="${WARMUP_REQUESTS:-0}"
  local kv_cache_bytes="${KV_CACHE_BYTES:-1288490189}"
  local max_model_len="${MAX_MODEL_LEN:-544}"
  local max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-2048}"
  local max_num_seqs="${MAX_NUM_SEQS:-32}"
  local gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.97}"
  local timeout_sec="${TIMEOUT_SEC:-3600}"
  local gpu_index="${GPU_INDEX:-0}"
  local clean_mib="${GPU_CLEAN_MAX_MIB:-0}"
  local clean_timeout="${GPU_CLEAN_TIMEOUT_SEC:-180}"
  local cpuset="${CPUSET:-0-13}"
  local cpu_threads="${CPU_THREADS:-14}"
  local sequential_requests="${SEQUENTIAL_REQUESTS:-2}"
  local cache_bytes="${SLOT_CACHE_BYTES:-268435456}"
  local cache_max_slots="${SLOT_CACHE_MAX_SLOTS:-512}"
  local run_label="${order}"
  [[ "${experiment_kind}" == "cache_reuse" ]] && run_label="cache_${order}"
  local run_root="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_${run_label}}"

  [[ "${experiment_kind}" == "cold_pair" || "${experiment_kind}" == "cache_reuse" ]] \
    || die "unknown experiment kind: ${experiment_kind}"
  [[ "${order}" == "AB" || "${order}" == "BA" ]] || die "ORDER must be AB or BA"
  [[ "${concurrency}" =~ ^[1-9][0-9]*$ ]] || die "CONCURRENCY must be positive"
  [[ "${output_tokens}" =~ ^[1-9][0-9]*$ ]] || die "OUTPUT_TOKENS must be positive"
  [[ "${warmup_requests}" == "0" ]] || die "WARMUP_REQUESTS must be exactly 0"
  [[ "${max_num_seqs}" =~ ^[1-9][0-9]*$ ]] || die "MAX_NUM_SEQS must be positive"
  (( max_num_seqs >= concurrency )) || die "MAX_NUM_SEQS is below CONCURRENCY"
  [[ "${clean_mib}" =~ ^[0-9]+$ ]] || die "GPU_CLEAN_MAX_MIB must be non-negative"
  [[ "${clean_timeout}" =~ ^[1-9][0-9]*$ ]] || die "GPU_CLEAN_TIMEOUT_SEC must be positive"
  [[ "${cpu_threads}" =~ ^[1-9][0-9]*$ ]] || die "CPU_THREADS must be positive"
  if [[ "${experiment_kind}" == "cache_reuse" ]]; then
    [[ "${concurrency}" == "1" ]] \
      || die "run-cache requires CONCURRENCY=1; requests are separate generate calls"
    [[ "${sequential_requests}" =~ ^[0-9]+$ ]] \
      && (( sequential_requests >= 2 )) \
      || die "SEQUENTIAL_REQUESTS must be at least 2"
    [[ "${cache_bytes}" =~ ^[1-9][0-9]*$ ]] \
      || die "SLOT_CACHE_BYTES must be positive"
    [[ "${cache_max_slots}" =~ ^[1-9][0-9]*$ ]] \
      || die "SLOT_CACHE_MAX_SLOTS must be positive"
    (( cache_bytes >= 524288 )) \
      || die "SLOT_CACHE_BYTES must fit at least one 512 KiB envelope"
  else
    sequential_requests=0
    cache_bytes=0
    cache_max_slots=0
  fi

  runtime_preflight "${model}" "${fixture}" "${plan}" "${gpu_index}" "${cpuset}"
  [[ "${experiment_kind}" != "cache_reuse" ]] || cache_source_preflight
  exec 9>"/tmp/vllm_slot_direct_integration.lock"
  flock -n 9 || die "another slot-direct integration run holds the experiment lock"

  local parent
  parent="$(dirname "${run_root}")"
  mkdir -p "${parent}"
  [[ ! -e "${run_root}" ]] || die "RUN_ROOT already exists: ${run_root}"
  mkdir "${run_root}"
  run_root="$(cd "${run_root}" && pwd)"
  model="$(cd "${model}" && pwd)"
  fixture="$(cd "$(dirname "${fixture}")" && pwd)/$(basename "${fixture}")"
  plan="$(cd "$(dirname "${plan}")" && pwd)/$(basename "${plan}")"

  local generic_dir_name slot_dir_name
  local -a sequence
  if [[ "${order}" == "AB" ]]; then
    sequence=(generic slot)
    generic_dir_name="01_generic"
    slot_dir_name="02_slot"
  else
    sequence=(slot generic)
    slot_dir_name="01_slot"
    generic_dir_name="02_generic"
  fi
  printf 'ordinal\talias\tarm\tdirectory\n' > "${run_root}/schedule.tsv"
  local index arm alias arm_dir_name
  for index in 0 1; do
    arm="${sequence[index]}"
    if [[ "${arm}" == "generic" ]]; then
      alias="A"
      arm_dir_name="${generic_dir_name}"
    else
      alias="B"
      arm_dir_name="${slot_dir_name}"
    fi
    printf '%s\t%s\t%s\t%s\n' "$((index + 1))" "${alias}" "${arm}" "${arm_dir_name}" \
      >> "${run_root}/schedule.tsv"
  done

  local module_srcversion loaded_srcversion budget_raw budget_bytes
  module_srcversion="$(modinfo -F srcversion "${CUSTOM_MODULE}")"
  loaded_srcversion="$(sed -n '1p' /sys/module/nvidia_uvm/srcversion)"
  budget_raw="$(tr -d '[:space:]' < "${SLOT_BUDGET_PATH}")"
  budget_bytes="$("${VLLM_PY}" -c 'import sys; print(int(sys.argv[1], 0))' "${budget_raw}")"
  write_run_config "${run_root}/run_config.json" "${order}" \
    "${generic_dir_name}" "${slot_dir_name}" "${model}" "${fixture}" "${plan}" \
    "${concurrency}" "${output_tokens}" "${kv_cache_bytes}" "${max_model_len}" \
    "${max_num_batched_tokens}" "${max_num_seqs}" "${gpu_memory_utilization}" \
    "${module_srcversion}" "${loaded_srcversion}" "${budget_bytes}" "${gpu_index}" \
    "${experiment_kind}" "${sequential_requests}" "${cache_bytes}" \
    "${cache_max_slots}"
  git -C "${REPO_ROOT}" status --short > "${run_root}/git_status.txt"
  modinfo "${CUSTOM_MODULE}" > "${run_root}/custom_module.modinfo.txt"
  sha256sum "${CUSTOM_MODULE}" "${fixture}" "${plan}" "${model}/config.json" \
    > "${run_root}/input_sha256.txt"
  {
    echo "loaded_srcversion=${loaded_srcversion}"
    echo "custom_srcversion=${module_srcversion}"
    echo "slot_enable=$(tr -d '[:space:]' < "${SLOT_ENABLE_PATH}")"
    echo "slot_budget_bytes=${budget_bytes}"
    echo "stats_path=${STATS_PATH}"
  } > "${run_root}/module_state.env"
  wait_for_clean_gpu "${run_root}/gpu.initial.csv" "${gpu_index}" "${clean_mib}" "${clean_timeout}"

  printf 'ordinal\tarm\treturncode\n' > "${run_root}/execution_status.tsv"
  local failures=0 rc arm_dir
  for index in 0 1; do
    arm="${sequence[index]}"
    arm_dir_name="${generic_dir_name}"
    [[ "${arm}" == "slot" ]] && arm_dir_name="${slot_dir_name}"
    arm_dir="${run_root}/${arm_dir_name}"
    if run_arm "${arm}" "$((index + 1))" "${arm_dir}" "${model}" "${fixture}" \
        "${plan}" "${concurrency}" "${output_tokens}" "${kv_cache_bytes}" \
        "${max_model_len}" "${max_num_batched_tokens}" "${max_num_seqs}" \
        "${gpu_memory_utilization}" "${timeout_sec}" "${gpu_index}" \
        "${clean_mib}" "${clean_timeout}" "${cpuset}" "${cpu_threads}" \
        "${experiment_kind}" "${sequential_requests}" "${cache_bytes}" \
        "${cache_max_slots}"; then
      rc=0
    else
      rc=$?
      failures=$((failures + 1))
      tail -n 120 "${arm_dir}/run.log" >&2 || true
    fi
    printf '%s\t%s\t%s\n' "$((index + 1))" "${arm}" "${rc}" \
      >> "${run_root}/execution_status.tsv"
  done

  local verify_rc=0
  if "${VLLM_PY}" "${VERIFIER}" "${run_root}" \
      > >(tee "${run_root}/verify.log") \
      2> >(tee "${run_root}/verify.err" >&2); then
    verify_rc=0
  else
    verify_rc=$?
  fi
  wait_for_clean_gpu "${run_root}/gpu.final.csv" "${gpu_index}" "${clean_mib}" "${clean_timeout}"
  echo "RUN_ROOT=${run_root}"
  if (( failures > 0 || verify_rc > 0 )); then
    die "integration pair failed: arm_failures=${failures} verifier_rc=${verify_rc}"
  fi
  echo "summary=${run_root}/summary.json"
}

case "${COMMAND}" in
  static-preflight|preflight)
    static_preflight
    ;;
  run)
    run_experiment cold_pair
    ;;
  run-cache)
    run_experiment cache_reuse
    ;;
  verify)
    [[ $# -eq 1 ]] || die "verify requires exactly one run-root argument"
    "${VLLM_PY}" "${VERIFIER}" "$1"
    ;;
  help|-h|--help)
    usage_text
    ;;
  *)
    usage_text >&2
    die "unknown command: ${COMMAND}"
    ;;
esac
