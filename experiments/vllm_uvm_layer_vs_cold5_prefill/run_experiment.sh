#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_PY:-${VLLM_ROOT}/.venv/bin/python}"
RUNNER="${RUNNER_OVERRIDE:-${SCRIPT_DIR}/benchmark_compare_steps.py}"
VERIFY="${VERIFY_OVERRIDE:-${SCRIPT_DIR}/verify_experiment.py}"
COLD5_DIR="${REPO_ROOT}/experiments/vllm_uvm_cold5_alias_prefill"
COLD5_NAMES="${COLD5_DIR}/exact_parameter_names.txt"
COLD5_PLAN="${COLD5_DIR}/cold5_expert_plan.json"
COLD5_FREEZER="${COLD5_DIR}/freeze_cold5_plan.py"
L0_L1_NAMES="${SCRIPT_DIR}/exact_l0_l1_names.txt"
L0_L3_NAMES="${SCRIPT_DIR}/exact_l0_l3_names.txt"
BASE_RUNNER="${REPO_ROOT}/experiments/vllm_p3_static_alias_graph_integration/benchmark_full_graph_steps.py"
CUSTOM_UVM="${CUSTOM_UVM:-${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
DEFAULT_MODEL="${REPO_ROOT}/../.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782"
DEFAULT_FIXTURE="${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json"

COLD5_SELECTED_BYTES=1132462080
COLD5_BACKING_BYTES=47185920
L0_L1_SELECTED_BYTES=1207959552
L0_L3_SELECTED_BYTES=2415919104
LAYER_BACKING_BYTES=1207959552
MIN_SLOT_BUDGET_BYTES=1207959552
SLOT_BUDGET_BYTES="${SLOT_BUDGET_BYTES:-1342177280}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-167772160}"
GPU_INDEX="${GPU_INDEX:-0}"
CPUSET="${CPUSET:-0-13}"
CPU_THREADS="${CPU_THREADS:-14}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"
GPU_CLEAN_TIMEOUT_SEC="${GPU_CLEAN_TIMEOUT_SEC:-240}"

RUN_ROOT_ACTIVE=""
STOCK_UVM=""
STOCK_SRCVERSION=""
CUSTOM_SRCVERSION=""
MODULE_CYCLE_ARMED=0
MODULE_RESTORED=0
ACTIVE_SERVICES=()

die() { echo "error: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }

uvm_clients() {
  sudo fuser /dev/nvidia-uvm /dev/nvidia-uvm-tools 2>/dev/null \
    | tr ' ' '\n' | awk '$1 ~ /^[0-9]+$/ {print $1}' | sort -nu || true
}

gpu_used_mib() {
  nvidia-smi -i "${GPU_INDEX}" --query-gpu=memory.used \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]'
}

compute_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk '$1 ~ /^[0-9]+$/ {print $1}' | sort -nu
}

gpu_clean() {
  [[ "$(gpu_used_mib)" == "0" && -z "$(compute_pids)" && -z "$(uvm_clients)" ]]
}

wait_gpu_clean() {
  local attempt
  for ((attempt = 0; attempt < GPU_CLEAN_TIMEOUT_SEC; ++attempt)); do
    gpu_clean && return 0
    sleep 1
  done
  return 1
}

loaded_srcversion() {
  [[ -r /sys/module/nvidia_uvm/srcversion ]] || return 0
  tr -d '[:space:]' </sys/module/nvidia_uvm/srcversion
}

snapshot_gpu() {
  local output="$1" used pids clients loaded
  used="$(gpu_used_mib 2>/dev/null || true)"
  pids="$(compute_pids | paste -sd, - || true)"
  clients="$(uvm_clients | paste -sd, - || true)"
  loaded="$(loaded_srcversion)"
  "${VLLM_PY}" - "${output}" "${used}" "${pids}" "${clients}" "${loaded}" <<'PY'
import json
import sys
import time

path, used, pids, clients, loaded = sys.argv[1:]
json.dump({
    "timestamp": time.time(),
    "memory_used_mib": int(used) if used.isdigit() else used,
    "compute_pids": [int(x) for x in pids.split(",") if x],
    "uvm_client_pids": [int(x) for x in clients.split(",") if x],
    "loaded_srcversion": loaded or None,
}, open(path, "w", encoding="utf-8"), indent=2, sort_keys=True)
open(path, "a", encoding="utf-8").write("\n")
PY
}

stop_exporters() {
  local service
  for service in node-exporter.service gpu-exporter.service dcgm-exporter.service; do
    if [[ "$(systemctl is-active "${service}" 2>/dev/null || true)" == active ]]; then
      ACTIVE_SERVICES+=("${service}")
      sudo systemctl stop "${service}"
    fi
  done
}

restart_exporters() {
  local service
  for service in "${ACTIVE_SERVICES[@]}"; do
    sudo systemctl start "${service}" || true
  done
  ACTIVE_SERVICES=()
}

snapshot_services() {
  local output="$1" service status
  : >"${output}"
  for service in node-exporter.service gpu-exporter.service dcgm-exporter.service; do
    status="$(systemctl is-active "${service}" 2>/dev/null || true)"
    printf '%s\t%s\n' "${service}" "${status}" >>"${output}"
  done
}

restore_stock() {
  local rc=0
  (( MODULE_CYCLE_ARMED == 1 )) || return 0
  (( MODULE_RESTORED == 0 )) || return 0
  set +e
  if [[ "$(loaded_srcversion)" != "${STOCK_SRCVERSION}" ]]; then
    wait_gpu_clean || rc=1
    if (( rc == 0 )); then
      sudo modprobe -r nvidia_uvm || rc=1
      sudo modprobe nvidia || rc=1
      sudo modprobe nvidia_uvm || rc=1
    fi
  fi
  [[ "$(loaded_srcversion)" == "${STOCK_SRCVERSION}" ]] || rc=1
  wait_gpu_clean || rc=1
  (( rc == 0 )) && MODULE_RESTORED=1
  set -e
  return "${rc}"
}

on_exit() {
  local original_rc="$1" recovery_rc=0
  trap - EXIT INT TERM
  restore_stock || recovery_rc=$?
  if [[ -n "${RUN_ROOT_ACTIVE}" && -d "${RUN_ROOT_ACTIVE}" ]]; then
    snapshot_gpu "${RUN_ROOT_ACTIVE}/gpu.final.json" || recovery_rc=1
  fi
  restart_exporters
  (( original_rc == 0 )) || exit "${original_rc}"
  exit "${recovery_rc}"
}

trap 'on_exit $?' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

static_preflight() {
  local path
  for path in "${RUNNER}" "${VERIFY}" "${COLD5_NAMES}" "${COLD5_PLAN}" \
      "${COLD5_FREEZER}" "${L0_L1_NAMES}" "${L0_L3_NAMES}" \
      "${BASE_RUNNER}" "${CUSTOM_UVM}"; do
    require_file "${path}"
  done
  [[ -x "${VLLM_PY}" ]] || die "missing vLLM Python: ${VLLM_PY}"
  [[ "${SLOT_BUDGET_BYTES}" =~ ^[0-9]+$ ]] \
    || die "SLOT_BUDGET_BYTES is not an integer"
  [[ "${KV_CACHE_BYTES}" =~ ^[0-9]+$ ]] \
    || die "KV_CACHE_BYTES is not an integer"
  (( SLOT_BUDGET_BYTES >= MIN_SLOT_BUDGET_BYTES )) \
    || die "slot budget is below the 1152 MiB whole-layer backing"
  (( KV_CACHE_BYTES > 0 )) || die "KV_CACHE_BYTES must be positive"
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  command -v modinfo >/dev/null || die "modinfo is unavailable"
  command -v flock >/dev/null || die "flock is unavailable"
  command -v timeout >/dev/null || die "timeout is unavailable"
  command -v taskset >/dev/null || die "taskset is unavailable"
  command -v sudo >/dev/null || die "sudo is unavailable"
  "${VLLM_PY}" - "${COLD5_NAMES}" "${L0_L1_NAMES}" "${L0_L3_NAMES}" <<'PY'
from pathlib import Path
import sys

cold, l01, l03 = map(Path, sys.argv[1:])

def lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]

expected_l03 = [
    f"model.layers.{layer}.mlp.experts.{weight}"
    for layer in range(4)
    for weight in ("w13_weight", "w2_weight")
]
assert len(lines(cold)) == 96, "cold-5 exact-name count is not 96"
assert lines(l01) == expected_l03[:4], "L0-L1 exact names differ"
assert lines(l03) == expected_l03, "L0-L3 exact names differ"
PY
  "${VLLM_PY}" "${COLD5_FREEZER}" >/dev/null
  "${VLLM_PY}" -m py_compile "${RUNNER}" "${VERIFY}" "${COLD5_FREEZER}"
  "${VLLM_PY}" -m pytest -q \
    "${VLLM_ROOT}/tests/test_cpu_offload_exact_static.py"
  bash -n "${BASH_SOURCE[0]}"
  modinfo -p "${CUSTOM_UVM}" | grep -q '^uvm_manual_migration_fault_mode:' \
    || die "custom UVM lacks manual migration mode"
  echo "static preflight: OK"
}

runtime_preflight() {
  static_preflight
  [[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
  STOCK_UVM="$(modinfo -F filename nvidia-uvm)"
  STOCK_SRCVERSION="$(modinfo -F srcversion nvidia-uvm | tr -d '[:space:]')"
  CUSTOM_SRCVERSION="$(modinfo -F srcversion "${CUSTOM_UVM}" | tr -d '[:space:]')"
  [[ "$(loaded_srcversion)" == "${STOCK_SRCVERSION}" ]] \
    || die "initial nvidia_uvm is not stock"
  exec 9>"/tmp/vllm_uvm_layer_vs_cold5_prefill.lock"
  flock -n 9 || die "another layer-vs-cold5 run holds the lock"
  stop_exporters
  wait_gpu_clean || die "GPU/UVM is not clean"
  MODULE_CYCLE_ARMED=1
}

load_custom_module() {
  wait_gpu_clean || die "GPU changed before custom module load"
  sudo modprobe -r nvidia_uvm
  sudo modprobe nvidia
  sudo insmod "${CUSTOM_UVM}" \
    uvm_moe_slot_pte_remap_enable=1 \
    uvm_moe_slot_pool_vidmem_budget_bytes="${SLOT_BUDGET_BYTES}" \
    uvm_manual_migration_fault_mode=2
  [[ "$(loaded_srcversion)" == "${CUSTOM_SRCVERSION}" ]] \
    || die "custom UVM srcversion differs"
  [[ "$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode)" == 2 ]] \
    || die "manual migration mode is not 2"
  [[ "$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes)" == "${SLOT_BUDGET_BYTES}" ]] \
    || die "slot budget differs after module load"
  wait_gpu_clean || die "custom UVM load left GPU state"
}

arm_name() {
  case "$1" in
    Z) echo all_gpu_graph ;;
    C) echo cold5_alias_graph ;;
    B) echo l0_l1_alias_graph ;;
    L) echo l0_l3_alias_graph ;;
    *) die "unknown arm $1" ;;
  esac
}

run_arm() {
  local symbol="$1" arm_dir="$2" batch_order="$3"
  local arm alias offload_gb names_file plan_file expected_backing
  arm="$(arm_name "${symbol}")"
  mkdir "${arm_dir}"
  : >"${arm_dir}/cpu_offload_manifest.jsonl"
  wait_gpu_clean || die "GPU is not clean before ${arm}"
  snapshot_gpu "${arm_dir}/gpu.before.json"

  alias=0
  offload_gb=0
  names_file=""
  plan_file=""
  expected_backing=0
  case "${symbol}" in
    Z) ;;
    C)
      alias=1
      offload_gb=1.25
      names_file="${COLD5_NAMES}"
      plan_file="${COLD5_PLAN}"
      expected_backing="${COLD5_BACKING_BYTES}"
      ;;
    B)
      alias=1
      offload_gb=1.5
      names_file="${L0_L1_NAMES}"
      expected_backing="${LAYER_BACKING_BYTES}"
      ;;
    L)
      alias=1
      offload_gb=3.0
      names_file="${L0_L3_NAMES}"
      expected_backing="${LAYER_BACKING_BYTES}"
      ;;
  esac

  local -a envs=(
    VLLM_USE_UVM=0
    VLLM_ENABLE_V1_MULTIPROCESSING=0
    VLLM_CPU_OFFLOAD_BACKEND=copy
    VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${names_file}"
    VLLM_CPU_OFFLOAD_STATIC_BANK=0
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS="${alias}"
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER=0
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND="$([[ "${alias}" == 1 ]] && echo uvm_original_va || echo vmm)"
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="${plan_file}"
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=4k
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPECTED_ARENA_COUNT=4
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPECTED_BACKING_BYTES="${expected_backing}"
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED="${alias}"
    VLLM_UVM_ORIGINAL_VA_PHASE_MODE=off
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
    VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
    VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0
    VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0
    VLLM_MOE_TOKEN_EXPERT_TRACE_ENABLE=0
    VLLM_CPU_OFFLOAD_COPY_TRACE="$([[ "${alias}" == 1 ]] && echo 1 || echo 0)"
    VLLM_CPU_OFFLOAD_COPY_SYNC=0
    VLLM_CPU_OFFLOAD_MANIFEST_FILE="${arm_dir}/cpu_offload_manifest.jsonl"
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1
    VLLM_UVM_LOG_FILE="${arm_dir}/uvm_allocations.log"
    VLLM_LAYER_COMPARE_ARM="${symbol}"
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    TOKENIZERS_PARALLELISM=false
    OMP_NUM_THREADS="${CPU_THREADS}"
    MKL_NUM_THREADS="${CPU_THREADS}"
    CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
    VLLM_ATTENTION_BACKEND=TRITON_ATTN
  )
  printf '%s\n' "${envs[@]}" >"${arm_dir}/environment.txt"

  local model="${MODEL:-${DEFAULT_MODEL}}"
  local fixture="${EVAL_FIXTURE:-${DEFAULT_FIXTURE}}"
  local -a args=(
    --model "${model}"
    --fixture "${fixture}"
    --output "${arm_dir}/result.json"
    --offload-manifest "${arm_dir}/cpu_offload_manifest.jsonl"
    --profile-label "qwen_layer_compare_${arm}"
    --mechanism-class "vllm_${arm}_layer_compare"
    --execution-mode full_graph_all
    --batch-sizes "${batch_order}"
    --cudagraph-capture-sizes 1,4,314,1217
    --conditioning-repeats "${CONDITIONING_REPEATS}"
    --measurement-repeats "${MEASUREMENT_REPEATS}"
    --warmup-decode-steps "${WARMUP_DECODE_STEPS}"
    --measured-decode-steps "${MEASURED_DECODE_STEPS}"
    --kv-cache-memory-bytes "${KV_CACHE_BYTES}"
    --cpu-offload-gb "${offload_gb}"
    --max-model-len 512
    --max-num-seqs 4
    --max-num-batched-tokens 2048
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.97}"
    --seed 0
  )
  {
    printf 'cd %q && env ' "${VLLM_ROOT}"
    printf '%q ' "${envs[@]}"
    printf 'timeout --kill-after=30s %qs taskset -c %q %q %q ' \
      "${TIMEOUT_SEC}" "${CPUSET}" "${VLLM_PY}" "${RUNNER}"
    printf '%q ' "${args[@]}"
    printf '\n'
  } >"${arm_dir}/command.txt"

  local rc
  set +e
  (
    cd "${VLLM_ROOT}"
    env "${envs[@]}" /usr/bin/time -v -o "${arm_dir}/resource_time.txt" \
      timeout --kill-after=30s "${TIMEOUT_SEC}s" taskset -c "${CPUSET}" \
      "${VLLM_PY}" "${RUNNER}" "${args[@]}"
  ) >"${arm_dir}/run.log" 2>&1
  rc=$?
  set -e
  echo "${rc}" >"${arm_dir}/returncode.txt"
  wait_gpu_clean || die "GPU did not clean after ${arm}"
  snapshot_gpu "${arm_dir}/gpu.after.json"
  return "${rc}"
}

write_run_config() {
  local output="$1" mode="$2" order="$3" model="$4" fixture="$5"
  "${VLLM_PY}" - "${output}" "${mode}" "${order}" \
    "${CONDITIONING_REPEATS}" "${MEASUREMENT_REPEATS}" \
    "${WARMUP_DECODE_STEPS}" "${MEASURED_DECODE_STEPS}" \
    "${model}" "${fixture}" "${COLD5_NAMES}" "${COLD5_PLAN}" \
    "${L0_L1_NAMES}" "${L0_L3_NAMES}" "${SLOT_BUDGET_BYTES}" \
    "${KV_CACHE_BYTES}" "${GPU_MEMORY_UTILIZATION:-0.97}" <<'PY'
import collections
import hashlib
import json
import sys
from pathlib import Path

(
    output, mode, order, conditioning_repeats, measurement_repeats,
    warmup_decode_steps, measured_decode_steps, model, fixture, cold_names,
    cold_plan, l01_names, l03_names, slot_budget, kv_cache,
    gpu_memory_utilization,
) = sys.argv[1:]

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def source(path: str) -> dict[str, object]:
    value = Path(path).resolve()
    return {"path": str(value), "sha256": digest(value)}

model_path = Path(model).resolve()
fixture_path = Path(fixture).resolve()
names_c = source(cold_names)
plan_c = source(cold_plan)
names_b = source(l01_names)
names_l = source(l03_names)
arm_names = {
    "Z": "all_gpu_graph",
    "C": "cold5_alias_graph",
    "B": "l0_l1_alias_graph",
    "L": "l0_l3_alias_graph",
}
executions = [
    {
        "ordinal": index + 1,
        "symbol": symbol,
        "arm": arm_names[symbol],
        "batch_sizes": [1, 4] if index % 2 == 0 else [4, 1],
    }
    for index, symbol in enumerate(order)
]
geometry = {
    "Z": {
        "arm": arm_names["Z"],
        "selection": "none",
        "alias_enabled": False,
        "exact_parameter_names": None,
        "expert_plan": None,
        "selected_layers": [],
        "selected_layer_count": 0,
        "selected_parameter_count": 0,
        "selected_expert_count": 0,
        "selected_bytes": 0,
        "selected_parameter_bytes_per_prefill": 0,
        "expected_h2d_bytes_per_prefill": 0,
        "backing_bytes": 0,
        "expected_backing_bytes": 0,
        "expected_alias_count": 0,
        "expected_alias_bytes": 0,
        "theoretical_net_residency_relief_bytes": 0,
        "theoretical_net_bytes": 0,
        "cpu_offload_gb": 0.0,
    },
    "C": {
        "arm": arm_names["C"],
        "selection": "five calibration-cold experts in every MoE layer",
        "alias_enabled": True,
        "exact_parameter_names": names_c,
        "expert_plan": plan_c,
        "selected_layers": list(range(48)),
        "selected_layer_count": 48,
        "selected_parameter_count": 96,
        "experts_per_selected_layer": 5,
        "selected_expert_count": 240,
        "expected_diagnostics_selected_expert_count": 240,
        "full_selected_parameter_bytes": 28_991_029_248,
        "selected_bytes": 1_132_462_080,
        "selected_parameter_bytes_per_prefill": 1_132_462_080,
        "expected_h2d_bytes_per_prefill": 1_132_462_080,
        "expected_copy_bytes_per_selected_layer": 23_592_960,
        "backing_bytes": 47_185_920,
        "expected_backing_bytes": 47_185_920,
        "expected_alias_count": 2_250,
        "expected_alias_bytes": 1_179_648_000,
        "theoretical_net_residency_relief_bytes": 1_085_276_160,
        "theoretical_net_bytes": 1_085_276_160,
        "cpu_offload_gb": 1.25,
    },
    "B": {
        "arm": arm_names["B"],
        "selection": "all experts in layers L0-L1",
        "alias_enabled": True,
        "exact_parameter_names": names_b,
        "expert_plan": None,
        "selected_layers": [0, 1],
        "selected_layer_count": 2,
        "selected_parameter_count": 4,
        "experts_per_selected_layer": 128,
        "selected_expert_count": 256,
        "expected_diagnostics_selected_expert_count": 0,
        "full_selected_parameter_bytes": 1_207_959_552,
        "selected_bytes": 1_207_959_552,
        "selected_parameter_bytes_per_prefill": 1_207_959_552,
        "expected_h2d_bytes_per_prefill": 1_207_959_552,
        "expected_copy_bytes_per_selected_layer": 603_979_776,
        "backing_bytes": 1_207_959_552,
        "expected_backing_bytes": 1_207_959_552,
        "expected_alias_count": 4_608,
        "expected_alias_bytes": 2_415_919_104,
        "theoretical_net_residency_relief_bytes": 0,
        "theoretical_net_bytes": 0,
        "cpu_offload_gb": 1.5,
    },
    "L": {
        "arm": arm_names["L"],
        "selection": "all experts in layers L0-L3",
        "alias_enabled": True,
        "exact_parameter_names": names_l,
        "expert_plan": None,
        "selected_layers": [0, 1, 2, 3],
        "selected_layer_count": 4,
        "selected_parameter_count": 8,
        "experts_per_selected_layer": 128,
        "selected_expert_count": 512,
        "expected_diagnostics_selected_expert_count": 0,
        "full_selected_parameter_bytes": 2_415_919_104,
        "selected_bytes": 2_415_919_104,
        "selected_parameter_bytes_per_prefill": 2_415_919_104,
        "expected_h2d_bytes_per_prefill": 2_415_919_104,
        "expected_copy_bytes_per_selected_layer": 603_979_776,
        "backing_bytes": 1_207_959_552,
        "expected_backing_bytes": 1_207_959_552,
        "expected_alias_count": 6_912,
        "expected_alias_bytes": 3_623_878_656,
        "theoretical_net_residency_relief_bytes": 1_207_959_552,
        "theoretical_net_bytes": 1_207_959_552,
        "cpu_offload_gb": 3.0,
    },
}
payload = {
    "schema": "vllm_uvm_layer_vs_cold5_prefill_run_v1",
    "mode": mode,
    "order": order,
    "arm_names": arm_names,
    "arm_process_counts": dict(sorted(collections.Counter(order).items())),
    "executions": executions,
    "batch_sizes": [1, 4],
    "conditioning_repeats": int(conditioning_repeats),
    "measurement_repeats": int(measurement_repeats),
    "warmup_decode_steps": int(warmup_decode_steps),
    "measured_decode_steps": int(measured_decode_steps),
    "cudagraph_capture_sizes": [1, 4, 314, 1217],
    "kv_cache_memory_bytes": int(kv_cache),
    "gpu_memory_utilization": float(gpu_memory_utilization),
    "max_model_len": 512,
    "max_num_seqs": 4,
    "max_num_batched_tokens": 2048,
    "seed": 0,
    "common_contract": {
        "execution_mode": "full_graph_all",
        "batch_sizes": [1, 4],
        "cudagraph_capture_sizes": [1, 4, 314, 1217],
        "attention_backend": "TRITON_ATTN",
        "conditioning_repeats": int(conditioning_repeats),
        "measurement_repeats": int(measurement_repeats),
        "warmup_decode_steps": int(warmup_decode_steps),
        "measured_decode_steps": int(measured_decode_steps),
        "kv_cache_memory_bytes": int(kv_cache),
        "alias_backend": "uvm_original_va",
        "alias_region_bytes": 524_288,
        "alias_pte_mode": "4k",
        "expected_arena_count": 4,
        "slot_budget_bytes": int(slot_budget),
        "manual_migration_fault_mode": 2,
    },
    "model": str(model_path),
    "model_config_sha256": digest(model_path / "config.json"),
    "fixture": str(fixture_path),
    "fixture_sha256": digest(fixture_path),
    "arm_geometry": geometry,
}
Path(output).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
}

run_suite() {
  local mode="$1" order model fixture
  runtime_preflight
  model="${MODEL:-${DEFAULT_MODEL}}"
  fixture="${EVAL_FIXTURE:-${DEFAULT_FIXTURE}}"
  require_file "${model}/config.json"
  require_file "${fixture}"
  model="$(cd "${model}" && pwd)"
  fixture="$(cd "$(dirname "${fixture}")" && pwd)/$(basename "${fixture}")"
  case "${mode}" in
    smoke)
      order="${ORDER:-ZCBL}"
      CONDITIONING_REPEATS="${CONDITIONING_REPEATS:-0}"
      MEASUREMENT_REPEATS="${MEASUREMENT_REPEATS:-1}"
      WARMUP_DECODE_STEPS="${WARMUP_DECODE_STEPS:-1}"
      MEASURED_DECODE_STEPS="${MEASURED_DECODE_STEPS:-2}"
      ;;
    qualify)
      order="${ORDER:-ZCBL}"
      CONDITIONING_REPEATS="${CONDITIONING_REPEATS:-1}"
      MEASUREMENT_REPEATS="${MEASUREMENT_REPEATS:-2}"
      WARMUP_DECODE_STEPS="${WARMUP_DECODE_STEPS:-2}"
      MEASURED_DECODE_STEPS="${MEASURED_DECODE_STEPS:-4}"
      ;;
    formal)
      # Each arm receives three fresh processes and both 1,4 and 4,1 batch
      # orders. The unavoidable per-arm batch-order imbalance is only 2:1.
      order="${ORDER:-ZCBLBLZCCZLB}"
      CONDITIONING_REPEATS="${CONDITIONING_REPEATS:-1}"
      MEASUREMENT_REPEATS="${MEASUREMENT_REPEATS:-3}"
      WARMUP_DECODE_STEPS="${WARMUP_DECODE_STEPS:-5}"
      MEASURED_DECODE_STEPS="${MEASURED_DECODE_STEPS:-8}"
      ;;
    *) die "unknown mode ${mode}" ;;
  esac
  [[ "${order}" =~ ^[ZCBL]+$ ]] || die "ORDER contains an unknown arm"

  local run_root="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_${mode}}"
  [[ ! -e "${run_root}" ]] || die "RUN_ROOT already exists"
  mkdir -p "${run_root}"
  run_root="$(cd "${run_root}" && pwd)"
  RUN_ROOT_ACTIVE="${run_root}"
  printf '%s\n' "${ACTIVE_SERVICES[@]}" >"${run_root}/services.initially_active.txt"
  snapshot_gpu "${run_root}/gpu.before.json"
  load_custom_module
  snapshot_gpu "${run_root}/gpu.custom.json"
  sha256sum "${COLD5_PLAN}" "${COLD5_NAMES}" "${L0_L1_NAMES}" \
    "${L0_L3_NAMES}" "${COLD5_FREEZER}" "${RUNNER}" "${VERIFY}" \
    "${SCRIPT_DIR}/run_experiment.sh" "${BASE_RUNNER}" \
    "${VLLM_ROOT}/vllm/model_executor/models/utils.py" \
    "${model}/config.json" "${fixture}" "${CUSTOM_UVM}" "${STOCK_UVM}" \
    >"${run_root}/input_sha256.txt"
  git -C "${REPO_ROOT}" status --short >"${run_root}/git_status.txt"
  write_run_config "${run_root}/run_config.json" "${mode}" "${order}" \
    "${model}" "${fixture}"
  printf 'ordinal\tsymbol\tarm\tbatch_order\tdirectory\treturncode\n' \
    >"${run_root}/execution_status.tsv"

  local failures=0 index symbol arm batch_order arm_dir rc
  for ((index = 0; index < ${#order}; ++index)); do
    symbol="${order:index:1}"
    arm="$(arm_name "${symbol}")"
    batch_order=1,4
    (( index % 2 == 1 )) && batch_order=4,1
    arm_dir="${run_root}/$(printf '%02d_%s' "$((index + 1))" "${arm}")"
    if run_arm "${symbol}" "${arm_dir}" "${batch_order}"; then
      rc=0
    else
      rc=$?
      failures=$((failures + 1))
      tail -n 160 "${arm_dir}/run.log" >&2 || true
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$((index + 1))" "${symbol}" \
      "${arm}" "${batch_order}" "$(basename "${arm_dir}")" "${rc}" \
      >>"${run_root}/execution_status.tsv"
  done

  restore_stock || die "failed to restore stock UVM"
  snapshot_gpu "${run_root}/gpu.stock.json"
  restart_exporters
  snapshot_services "${run_root}/services.after.tsv"
  while IFS= read -r service; do
    [[ -z "${service}" ]] && continue
    [[ "$(systemctl is-active "${service}" 2>/dev/null || true)" == active ]] \
      || die "failed to restore ${service}"
  done <"${run_root}/services.initially_active.txt"
  if [[ "${mode}" != smoke && ${failures} -eq 0 ]]; then
    "${VLLM_PY}" "${VERIFY}" "${run_root}" \
      --output "${run_root}/verification.json" >"${run_root}/verify.log"
  fi
  echo "RUN_ROOT=${run_root}"
  (( failures == 0 )) || die "${failures} arm(s) failed"
}

case "${1:-help}" in
  static-preflight) static_preflight ;;
  smoke|qualify|formal) run_suite "$1" ;;
  verify)
    [[ $# == 2 ]] || die "verify requires RUN_ROOT"
    "${VLLM_PY}" "${VERIFY}" "$2"
    ;;
  *)
    echo "usage: $0 {static-preflight|smoke|qualify|formal|verify RUN_ROOT}" >&2
    exit 2
    ;;
esac
