#!/usr/bin/env bash
set -euo pipefail

E10_MULTI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E10_MULTI_ROOT="$(cd "${E10_MULTI_DIR}/../.." && pwd)"
E10_MULTI_BASE="${E10_MULTI_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
E10_MULTI_VERIFY_OVERRIDE="${VERIFY-}"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
export ACCESS_COUNTER_MIGRATION_ENABLE="${ACCESS_COUNTER_MIGRATION_ENABLE:-0}"
export UVM_PREFETCH_ENABLE="${UVM_PREFETCH_ENABLE:-0}"
export UVM_THRASHING_ENABLE="${UVM_THRASHING_ENABLE:-0}"
export STATIC_ALIAS_DUAL_VIEW_ENABLE=1
export STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL=1
# shellcheck source=/dev/null
source "${E10_MULTI_BASE}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY

SCRIPT_DIR="${E10_MULTI_DIR}"
REPO_ROOT="${E10_MULTI_ROOT}"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
RUNNER="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/benchmark_phase_switch_steps.py"
VERIFY="${E10_MULTI_VERIFY_OVERRIDE:-${SCRIPT_DIR}/verify_experiment.py}"
PREPARE_CLASSES="${SCRIPT_DIR}/prepare_plan_classes.py"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
FIXTURE="${FIXTURE:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json}"
NAMES="${REPO_ROOT}/experiments/vllm_uvm_large_kv_multilevel/inputs/exact_parameter_names.txt"
BASE_PLAN="${REPO_ROOT}/experiments/vllm_uvm_large_kv_multilevel/inputs/cold_decode_plan_base70.json"
WAVE_PLAN="${REPO_ROOT}/experiments/vllm_uvm_large_kv_multilevel/inputs/incremental_expert_waves.json"
INPUT_SET_DIR="${E10_INPUT_SET_DIR:-${SCRIPT_DIR}/inputs}"
COLD70="${INPUT_SET_DIR}/cold70.json"
PLAN_CLASSES="${INPUT_SET_DIR}/plan_classes.json"

E10_TIER_SPECS="${E10_TIER_SPECS:-2:1364:cold70,4:2730:cold80,6:4095:cold89}"
KV_MAX_BYTES="${KV_MAX_BYTES:-8589410304}"
PLAN_CLASS_MAP="${PLAN_CLASS_MAP:-{\"1364\":\"cold70\",\"2730\":\"cold80\",\"4095\":\"cold89\"}}"
BACKUP_RESERVE_MAP="${BACKUP_RESERVE_MAP:-{\"1364\":1348,\"2730\":2714,\"4095\":4079}}"
first_tier_spec="${E10_TIER_SPECS%%,*}"
last_tier_spec="${E10_TIER_SPECS##*,}"
IFS=: read -r KV_INITIAL_GIB KV_INITIAL_BLOCKS KV_INITIAL_CLASS \
  <<<"${first_tier_spec}"
IFS=: read -r KV_MAX_GIB KV_MAX_TARGET_BLOCKS KV_MAX_CLASS \
  <<<"${last_tier_spec}"
ALLOWED_TARGET_BLOCKS="$(
  "${VLLM_PY}" - "${E10_TIER_SPECS}" <<'PY'
import sys
print(",".join(item.split(":")[1] for item in sys.argv[1].split(",")))
PY
)"
TIMEOUT_SEC="${TIMEOUT_SEC:-7200}"
CPUSET="${CPUSET:-0-13}"
E10_TRIGGER_MODE="${E10_TRIGGER_MODE:-pressure}"
export E10_TRIGGER_MODE

die_multilevel() {
  echo "E10 multilevel pressure: $*" >&2
  exit 1
}

require_multilevel_file() {
  [[ -f "$1" ]] || die_multilevel "missing file: $1"
}

static_preflight_multilevel() {
  [[ "${E10_TRIGGER_MODE}" == "pressure" || "${E10_TRIGGER_MODE}" == "direct" ]] \
    || die_multilevel "E10_TRIGGER_MODE must be pressure or direct"
  local path
  for path in "${RUNNER}" "${VERIFY}" "${PREPARE_CLASSES}" \
    "${MODEL}/config.json" "${FIXTURE}" "${NAMES}" "${BASE_PLAN}" \
    "${WAVE_PLAN}" "${COLD70}" \
    "${PLAN_CLASSES}"; do
    require_multilevel_file "${path}"
  done
  [[ "$(wc -l <"${NAMES}")" -eq 96 ]] \
    || die_multilevel "expected 96 exact parameter names"
  bash -n "${BASH_SOURCE[0]}"
  "${VLLM_PY}" -m py_compile "${RUNNER}" "${VERIFY}" "${PREPARE_CLASSES}"
  "${VLLM_PY}" - "${PLAN_CLASSES}" "${KV_MAX_BYTES}" \
    "${PLAN_CLASS_MAP}" "${BACKUP_RESERVE_MAP}" \
    "${E10_TIER_SPECS}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
maximum = int(sys.argv[2])
class_map = {int(k): v for k, v in json.loads(sys.argv[3]).items()}
reserve_map = {int(k): int(v) for k, v in json.loads(sys.argv[4]).items()}
specs = [item.split(":") for item in sys.argv[5].split(",")]
gib = [int(item[0]) for item in specs]
targets = [int(item[1]) for item in specs]
classes = [item[2] for item in specs]
assert gib == sorted(set(gib))
assert targets == sorted(set(targets))
assert manifest["initial_class"] == classes[0]
assert set(manifest["classes"]) >= set(classes)
assert class_map == dict(zip(targets, classes))
assert reserve_map == {target: target - 16 for target in targets}
block_bytes = 1_572_864
assert maximum % block_bytes == 0
pool_blocks = maximum // block_bytes
# E10 reserves an 8 GiB virtual-address pool (5,461 blocks), while this
# qualification intentionally grows physical GPU residency only to the
# 6 GiB tier (4,095 blocks).  The pool size must therefore be checked
# independently from the final physical target.
assert pool_blocks == 5_461
assert targets[-1] < pool_blocks
expected = {name: int(name.removeprefix("cold")) for name in classes}
previous = None
for name, width in expected.items():
    path = sys.path[0]
    plan_file = manifest["classes"][name]["plan_file"]
    import pathlib
    plan = json.load(open(pathlib.Path(sys.argv[1]).parent / plan_file, encoding="utf-8"))
    by_layer = {int(k): set(map(int, v)) for k, v in plan["selected_by_layer"].items()}
    assert set(by_layer) == set(range(48))
    assert {len(v) for v in by_layer.values()} == {width}
    if previous is not None:
        assert all(previous[layer] < by_layer[layer] for layer in range(48))
    previous = by_layer
PY
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" -m pytest -q \
      tests/test_cpu_offload_exact_static.py \
      tests/test_uvm_static_alias_ioctl.py \
      tests/v1/core/test_uvm_dynamic_pool_queues.py \
      tests/v1/engine/test_uvm_adaptive_kv.py \
      tests/v1/worker/test_uvm_graph_plan_classes.py \
      tests/v1/worker/test_uvm_post_warmup_rebalance.py \
      -k 'transition_to_remote_audits_metadata or static_alias_ioctl_abi_matches_kernel_header or e10_4500_entry_delta or defer_dual_view_refresh or deferred_dual_view_refresh or backup_queue_target_can_follow or backup_queue_is_used_only or backup_threshold_growth_ignores_prediction or growth_switches_frozen_plan_class_before_physical_kv or resumes_prefill_before_plan_switch or worker_kv_resize_refreshes_full_dual_view_after_sync or worker_kv_resize_refresh_failure_fails_closed'
  )
  echo "E10 multilevel pressure static preflight: OK"
}

run_case_multilevel() {
  local run_root="$1" case_root="${1}/sequence"
  local manifest="${case_root}/cpu_offload_manifest.jsonl"
  mkdir -p "${case_root}"
  : >"${manifest}"
  gpu_state_json "${case_root}/gpu.before.json"

  set +e
  (
    cd "${VLLM_ROOT}"
    env \
      VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      VLLM_ATTENTION_BACKEND=TRITON_ATTN \
      VLLM_CPU_OFFLOAD_BACKEND=copy \
      VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${NAMES}" \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1 \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="${COLD70}" \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASSES_FILE="${PLAN_CLASSES}" \
      VLLM_CPU_OFFLOAD_PLAN_CLASS_PREFILL_SOURCES=1 \
      VLLM_CPU_OFFLOAD_PLAN_CLASS_IMMUTABLE_CPU_SHADOW=1 \
      VLLM_CPU_OFFLOAD_DUAL_VIEW_FULL_PARAMETER_SHADOW=1 \
      VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW=1 \
      VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE=0 \
      VLLM_UVM_FUSED_MOE_PRE_KERNEL_ALIAS_READ_GATE=0 \
      VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE=0 \
      VLLM_CPU_OFFLOAD_PLAN_CLASS_PRECOMPILE=1 \
      VLLM_CPU_OFFLOAD_PLAN_CLASS_QUIESCED_DELTA=1 \
      VLLM_CPU_OFFLOAD_PLAN_CLASS_INCREMENTAL_DUAL_REFRESH=1 \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288 \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=auto \
      VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1 \
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
      VLLM_UVM_ORIGINAL_VA_PHASE_MODE=prefill_graph_remote_decode \
      VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES=201326592 \
      VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ=0 \
      VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1 \
      VLLM_UVM_EXPERT_ENVELOPE_BYTES=524288 \
      VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=remote_read \
      VLLM_UVM_RUNTIME_POST_MAP_RESIDENCY_AUDIT=0 \
      VLLM_UVM_RUNTIME_HOT_RESIDENCY_AUDIT=0 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup \
      VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0 \
      VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0 \
      VLLM_UVM_WEIGHT_EXECUTOR=cpp_batch \
      VLLM_CPU_OFFLOAD_COPY_TRACE=0 \
      VLLM_CPU_OFFLOAD_COPY_SYNC=0 \
      VLLM_CPU_OFFLOAD_MANIFEST_FILE="${manifest}" \
      VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 \
      VLLM_PHASE_SWITCH_CONTROLLED_HARD_EXIT_AFTER_CLEANUP=1 \
      VLLM_UVM_LOG_FILE="${case_root}/uvm.log" \
      VLLM_UVM_KV_CACHE_MANAGED=1 \
      VLLM_UVM_KV_CACHE_LAZY_RESERVE=1 \
      VLLM_UVM_KV_RUNTIME_ENABLE=1 \
      VLLM_UVM_KV_RUNTIME_MODE=enforce \
      VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=1 \
      VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS="${KV_INITIAL_BLOCKS}" \
      VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS="${KV_INITIAL_BLOCKS}" \
      VLLM_UVM_KV_BACKUP_RESERVE_BLOCKS="$((KV_INITIAL_BLOCKS - 16))" \
      VLLM_UVM_KV_BACKUP_RESERVE_TARGET_MAP="${BACKUP_RESERVE_MAP}" \
      VLLM_UVM_KV_BACKUP_GROW_BLOCKS=1366 \
      VLLM_UVM_KV_BACKUP_PRESSURE_STANDALONE_GROW_ENABLE=0 \
      VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0 \
      VLLM_UVM_KV_AUTO_SCALE_ENABLE=0 \
      VLLM_UVM_KV_SCHEDULER_BOUNDARY_RESIZE_ENABLE=1 \
      VLLM_UVM_KV_AUTO_SCALE_SYNCHRONIZED_TIER_STEPS=1 \
      VLLM_UVM_KV_AUTO_SCALE_BACKUP_ONLY=1 \
      VLLM_UVM_KV_AUTO_SCALE_BACKUP_PRESSURE_THRESHOLD_BLOCKS=4 \
      VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_MAP="${PLAN_CLASS_MAP}" \
      VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS="${ALLOWED_TARGET_BLOCKS}" \
      VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_BEFORE_GROW=1 \
      VLLM_UVM_KV_AUTO_SCALE_RESUME_PREFILL_BEFORE_PLAN_CLASS=1 \
      VLLM_UVM_KV_AUTO_SCALE_LOW_PRESSURE_STREAK=1000000 \
      VLLM_UVM_KV_AUTO_SCALE_GROW_FACTOR=2 \
      VLLM_UVM_KV_AUTO_SCALE_SHRINK_DIVISOR=2 \
      VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS="${KV_INITIAL_BLOCKS}" \
      VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS="${KV_MAX_TARGET_BLOCKS}" \
      VLLM_UVM_KV_AUTO_SCALE_SAFETY_BLOCKS=0 \
      VLLM_UVM_KV_AUTO_SCALE_MIN_RESIDENCY_STEPS=0 \
      VLLM_UVM_KV_AUTO_SCALE_RESIZE_GRANULARITY_BLOCKS=1365 \
      VLLM_UVM_KV_AUTO_SCALE_EXPERT_RESTORE_ENABLE=0 \
      VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_ENABLE=1 \
      VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_EXECUTOR=cpp_batch \
      VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_SYNC=1 \
      VLLM_UVM_POST_WARMUP_KV_OFFLOAD_EXECUTOR=cpp_batch \
      VLLM_UVM_KV_ADAPTIVE_FAIL_FAST=1 \
      VLLM_UVM_POST_WARMUP_REBALANCE_FAIL_FAST=1 \
      VLLM_USE_UVM=0 \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      TOKENIZERS_PARALLELISM=false \
      OMP_NUM_THREADS=14 \
      MKL_NUM_THREADS=14 \
      CUDA_VISIBLE_DEVICES=0 \
      timeout --kill-after=30s "${TIMEOUT_SEC}s" \
      taskset -c "${CPUSET}" \
      "${VLLM_PY}" "${RUNNER}" \
        --model "${MODEL}" \
        --fixture "${FIXTURE}" \
        --exact-names "${NAMES}" \
        --expert-plan "${COLD70}" \
        --offload-manifest "${manifest}" \
        --output "${case_root}/result.json" \
        --arm D \
        --batch-size 1 \
        --cudagraph-capture-sizes 1,314 \
        --warmup-decode-steps 5 \
        --measured-decode-steps 16 \
        --kv-multilevel-pressure-2-4-6 \
        --kv-multilevel-trigger-mode "${E10_TRIGGER_MODE}" \
        --kv-multilevel-tier-specs "${E10_TIER_SPECS}" \
        --kv-cache-memory-bytes "${KV_MAX_BYTES}" \
        --cpu-offload-gb 30.0 \
        --max-model-len 512 \
        --max-num-seqs 1 \
        --max-num-batched-tokens 512 \
        --gpu-memory-utilization 0.97 \
        --seed 0 \
        --expected-parameter-count 96 \
        --expected-full-parameter-bytes 28991029248 \
        --expected-selected-bytes 15854469120
  ) >"${case_root}/run.log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "${rc}" >"${case_root}/returncode.txt"
  if ! wait_gpu_clean; then
    gpu_state_json "${case_root}/gpu.after.json"
    die_multilevel "GPU did not clean after the single sequence"
  fi
  gpu_state_json "${case_root}/gpu.after.json"
  if (( rc != 0 )); then
    tail -n 300 "${case_root}/run.log" >&2 || true
  fi
  return "${rc}"
}

run_qualification_multilevel() {
  [[ "${ALLOW_GPU_RUN:-0}" == 1 ]] \
    || die_multilevel "GPU execution requires ALLOW_GPU_RUN=1"
  static_preflight_multilevel
  runtime_preflight
  local run_root="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_qualification}"
  [[ ! -e "${run_root}" ]] || die_multilevel "RUN_ROOT already exists: ${run_root}"
  mkdir -p "${run_root}"
  run_root="$(cd "${run_root}" && pwd)"
  RUN_ROOT_ACTIVE="${run_root}"

  "${VLLM_PY}" - "${run_root}/protocol.json" \
    "${E10_TIER_SPECS}" "${KV_MAX_BYTES}" <<'PY'
import json, pathlib
import sys
path = pathlib.Path(sys.argv[1])
specs = [item.split(":") for item in sys.argv[2].split(",")]
path.write_text(json.dumps({
    "schema": "eurosys_e10_kv_multilevel_pressure_protocol_v2",
    "qualification_only": True,
    "complete_sequences": 1,
    "statistical_repetitions": 0,
    "kv_reserved_va_bytes": int(sys.argv[3]),
    "kv_targets_blocks": [int(item[1]) for item in specs],
    "kv_targets_gib": [int(item[0]) for item in specs],
    "expert_classes": [item[2] for item in specs],
    "expert_strategy": "cold-frequency",
    "trigger_mode": __import__('os').environ.get('E10_TRIGGER_MODE', 'pressure'),
    "pressure_source": (
        "real scheduler consumption of a tier-sized backup queue"
        if __import__('os').environ.get('E10_TRIGGER_MODE', 'pressure') == 'pressure'
        else "direct target applied at an idle request boundary"
    ),
    "prediction_enabled": False,
    "runtime_post_map_residency_audit": False,
    "runtime_hot_residency_audit": False,
    "migration_heuristics": {
        "manual_fault_mode": 2,
        "access_counter_migration": 0,
        "prefetch": 0,
        "thrashing": 0,
    },
    "dual_view_parent_switch": {
        "enabled": True,
        "require_full_coverage": True,
        "prebuilt_views": ["parameter_to_cpu", "parameter_to_gpu_slot"],
        "full_parameter_shadow": True,
        "retained_tiers": 1,
        "rebuild_after_each_kv_tier_change": True,
        "managed_policy_granularity": "whole_parameter",
        "switch_granularity_bytes": 2097152,
        "switch_batch_blocks": 64,
    },
    "rebalance_order": [
        "prefill_topology_resume",
        "expert_plan_class_deferred_dual_refresh",
        "physical_kv_prefix",
        "current_tier_dual_view_rebuild",
    ],
    "requests_per_tier": 1,
    "decode_warmup_steps_per_tier": 5,
    "decode_measured_steps_per_tier": 16,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  git -C "${REPO_ROOT}" status --short >"${run_root}/git_status.txt"
  nvidia-smi -i 0 --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader,nounits >"${run_root}/hardware.csv"
  module_snapshot "${run_root}/module.before.json" "${STOCK_UVM}"
  load_custom_module
  module_snapshot "${run_root}/module.custom.json" "${CUSTOM_UVM}"
  wait_gpu_clean || die_multilevel "GPU is not clean before qualification"
  cp -- "${STATS_PATH}" "${run_root}/latency_stats.before.txt"
  sha256sum \
    "${MODEL}/config.json" "${FIXTURE}" "${NAMES}" \
    "${INPUT_SET_DIR}"/cold*.json "${PLAN_CLASSES}" \
    "${RUNNER}" "${VERIFY}" "${PREPARE_CLASSES}" "${BASH_SOURCE[0]}" \
    "${VLLM_ROOT}/vllm/v1/core/block_pool.py" \
    "${VLLM_ROOT}/vllm/v1/core/uvm_kv_runtime_policy.py" \
    "${VLLM_ROOT}/vllm/v1/engine/core.py" \
    "${VLLM_ROOT}/vllm/v1/worker/gpu_model_runner.py" \
    "${VLLM_ROOT}/vllm/model_executor/models/utils.py" \
    "${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko" \
    "${STOCK_UVM}" >"${run_root}/input_sha256.txt"

  local kernel_epoch runner_rc journal_rc
  kernel_epoch="$(date +%s)"
  if run_case_multilevel "${run_root}"; then
    runner_rc=0
  else
    runner_rc=$?
  fi
  set +e
  sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
    >"${run_root}/kernel.since.log" 2>"${run_root}/kernel.since.err"
  journal_rc=$?
  set -e
  printf '%s\n' "${journal_rc}" >"${run_root}/kernel_log_returncode.txt"
  cp -- "${STATS_PATH}" "${run_root}/latency_stats.after.txt"
  restore_stock || die_multilevel "failed to restore stock nvidia_uvm"
  module_snapshot "${run_root}/module.after.json" "${STOCK_UVM}"
  restart_exporters
  (( journal_rc == 0 )) || die_multilevel "kernel journal collection failed"
  (( runner_rc == 0 )) || die_multilevel "the only permitted sequence failed"
  "${VLLM_PY}" "${VERIFY}" "${run_root}" \
    --expected-trigger-mode "${E10_TRIGGER_MODE}" \
    --tier-specs "${E10_TIER_SPECS}" \
    --kv-max-bytes "${KV_MAX_BYTES}" \
    --plan-classes "${PLAN_CLASSES}" \
    --output "${run_root}/verification.json" \
    >"${run_root}/verify.log" 2>"${run_root}/verify.err"
  echo "RUN_ROOT=${run_root}"
  echo "verification=${run_root}/verification.json"
}

case "${1:-help}" in
  static-preflight)
    static_preflight_multilevel
    ;;
  qualify|qualification|run)
    run_qualification_multilevel
    ;;
  verify)
    [[ $# -eq 2 ]] || die_multilevel "verify requires RUN_ROOT"
    "${VLLM_PY}" "${VERIFY}" "$2" \
      --expected-trigger-mode "${E10_TRIGGER_MODE}" \
      --tier-specs "${E10_TIER_SPECS}" \
      --kv-max-bytes "${KV_MAX_BYTES}" \
      --plan-classes "${PLAN_CLASSES}" \
      --output "$2/verification.json"
    ;;
  *)
    echo "usage: $0 {static-preflight|qualify|verify RUN_ROOT}" >&2
    exit 2
    ;;
esac
