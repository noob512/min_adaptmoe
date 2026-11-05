#!/usr/bin/env bash
set -euo pipefail

THREE_QUEUE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THREE_QUEUE_REPO_ROOT="$(cd "${THREE_QUEUE_SCRIPT_DIR}/../.." && pwd)"
THREE_QUEUE_BASE_HARNESS="${THREE_QUEUE_REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
# shellcheck source=/dev/null
source "${THREE_QUEUE_BASE_HARNESS}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY

SCRIPT_DIR="${THREE_QUEUE_SCRIPT_DIR}"
REPO_ROOT="${THREE_QUEUE_REPO_ROOT}"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
RUNNER="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/benchmark_phase_switch_steps.py"
VERIFY="${SCRIPT_DIR}/verify_experiment.py"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
FIXTURE="${FIXTURE:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json}"
NAMES="${REPO_ROOT}/experiments/qwen_decode_six_mechanism_compare/inputs/exact_parameter_names.txt"
STATIC_PLAN="${REPO_ROOT}/experiments/vllm_uvm_physical_kv_pool_resize/inputs/cold_decode_plan_6g.json"
SUPERSET_PLAN="${REPO_ROOT}/experiments/vllm_uvm_physical_kv_pool_resize/inputs/cold_decode_plan_7g.json"
INCREMENTAL_PLAN="${SCRIPT_DIR}/inputs/incremental_wave_plan.json"
TIER30_PLAN="${SCRIPT_DIR}/inputs/cold_frequency_tier_30.json"
PLAN_CLASSES="${SCRIPT_DIR}/inputs/cold_frequency_plan_classes.json"
PLAN_CLASS_MAP='{"24":"cold29","40":"cold30"}'
KV_MAX_BYTES="${KV_MAX_BYTES:-1610612736}"
BASELINE_TARGET_BLOCKS="${BASELINE_TARGET_BLOCKS:-40}"
QUEUE_INITIAL_TARGET_BLOCKS="${QUEUE_INITIAL_TARGET_BLOCKS:-24}"
BACKUP_RESERVE_BLOCKS="${BACKUP_RESERVE_BLOCKS:-8}"
BACKUP_GROW_BLOCKS="${BACKUP_GROW_BLOCKS:-16}"
BACKUP_PRESSURE_THRESHOLD_BLOCKS="${BACKUP_PRESSURE_THRESHOLD_BLOCKS:-4}"
AUTOSCALE_MIN_TARGET_BLOCKS="${AUTOSCALE_MIN_TARGET_BLOCKS:-20}"
AUTOSCALE_MAX_TARGET_BLOCKS="${AUTOSCALE_MAX_TARGET_BLOCKS:-40}"
AUTOSCALE_LOW_PROMPT_TOKENS="${AUTOSCALE_LOW_PROMPT_TOKENS:-16}"
AUTOSCALE_LOW_OUTPUT_TOKENS="${AUTOSCALE_LOW_OUTPUT_TOKENS:-5}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"
CPUSET="${CPUSET:-0-13}"
EXPERT_STRATEGY="${EXPERT_STRATEGY:-cold-frequency}"

die_three_queue() {
  echo "three-queue qualification: $*" >&2
  exit 1
}

require_three_queue_file() {
  [[ -f "$1" ]] || die_three_queue "missing file: $1"
}

static_preflight_three_queue() {
  require_three_queue_file "${RUNNER}"
  require_three_queue_file "${VERIFY}"
  require_three_queue_file "${MODEL}/config.json"
  require_three_queue_file "${FIXTURE}"
  require_three_queue_file "${NAMES}"
  require_three_queue_file "${STATIC_PLAN}"
  require_three_queue_file "${SUPERSET_PLAN}"
  require_three_queue_file "${INCREMENTAL_PLAN}"
  require_three_queue_file "${TIER30_PLAN}"
  require_three_queue_file "${PLAN_CLASSES}"
  [[ "$(wc -l <"${NAMES}")" -eq 96 ]] \
    || die_three_queue "expected 96 exact parameter names"
  [[ "${EXPERT_STRATEGY}" == cold-frequency ]] \
    || die_three_queue "this frozen E10 runner only supports cold-frequency"
  (( AUTOSCALE_MIN_TARGET_BLOCKS < QUEUE_INITIAL_TARGET_BLOCKS )) \
    || die_three_queue "autoscale minimum must be below initial target"
  (( QUEUE_INITIAL_TARGET_BLOCKS < AUTOSCALE_MAX_TARGET_BLOCKS )) \
    || die_three_queue "autoscale maximum must exceed initial target"
  (( BASELINE_TARGET_BLOCKS == AUTOSCALE_MAX_TARGET_BLOCKS )) \
    || die_three_queue "baseline target must equal autoscale maximum"
  (( AUTOSCALE_LOW_PROMPT_TOKENS > 0 )) \
    || die_three_queue "autoscale low prompt must be positive"
  (( AUTOSCALE_LOW_OUTPUT_TOKENS >= 4 )) \
    || die_three_queue "autoscale low request needs at least four outputs"
  (( BACKUP_PRESSURE_THRESHOLD_BLOCKS > 0 )) \
    || die_three_queue "backup pressure threshold must be positive"
  (( BACKUP_PRESSURE_THRESHOLD_BLOCKS <= BACKUP_RESERVE_BLOCKS )) \
    || die_three_queue "backup pressure threshold exceeds the reserve"
  bash -n "${BASH_SOURCE[0]}"
  "${VLLM_PY}" -m py_compile "${RUNNER}" "${VERIFY}"
  "${VLLM_PY}" - "${STATIC_PLAN}" "${SUPERSET_PLAN}" \
    "${INCREMENTAL_PLAN}" "${TIER30_PLAN}" "${PLAN_CLASSES}" <<'PY'
import json
import sys

static = json.load(open(sys.argv[1], encoding="utf-8"))
superset = json.load(open(sys.argv[2], encoding="utf-8"))
incremental = json.load(open(sys.argv[3], encoding="utf-8"))
tier30 = json.load(open(sys.argv[4], encoding="utf-8"))
classes = json.load(open(sys.argv[5], encoding="utf-8"))
static_keys = {
    (int(item["layer_id"]), int(item["expert_id"]))
    for item in static["offload_plan"]
}
superset_keys = {
    (int(item["layer_id"]), int(item["expert_id"]))
    for item in superset["offload_plan"]
}
wave = incremental["offload_plan"]
wave_keys = {
    (int(item["layer_id"]), int(item["expert_id"])) for item in wave
}
assert len(wave) == len(wave_keys) == 48
assert {layer for layer, _ in wave_keys} == set(range(48))
assert static_keys < superset_keys
assert wave_keys <= (superset_keys - static_keys)
assert not (wave_keys & static_keys)
assert "minimum summed calibration-decode route counts per layer" in (
    static.get("selection_rule") or ""
)
assert "minimum summed calibration-decode route counts per layer" in (
    superset.get("selection_rule") or ""
)
assert incremental["metadata"]["policy"] == "cold_only_one_per_layer"
assert incremental["metadata"]["source_plan"] == (
    "7 GiB minus 6 GiB nested cold-plan difference"
)
assert classes["initial_class"] == "cold29"
assert set(classes["classes"]) == {"cold29", "cold30"}
assert set(tier30["selected_by_layer"]) == {str(layer) for layer in range(48)}
assert {len(values) for values in tier30["selected_by_layer"].values()} == {30}
assert tier30["selected_main_bytes"] - static["metadata"]["selected_main_bytes"] \
    == 226492416
for layer in range(48):
    base = set(static["selected_by_layer"][str(layer)])
    high = set(tier30["selected_by_layer"][str(layer)])
    assert base < high and len(high - base) == 1
PY
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" -m pytest -q \
      tests/v1/core/test_uvm_dynamic_pool_queues.py \
      tests/v1/engine/test_uvm_adaptive_kv.py \
      tests/v1/worker/test_uvm_post_warmup_rebalance.py \
      -k 'backup or autoscale or incremental_expert or plan_class'
  )
  echo "three-queue static preflight: OK"
}

run_three_queue_case() {
  local run_root="$1" label="$2" initial_target="$3" reserve="$4"
  local qualification="$5" tiered_batch="$6" autoscale="$7"
  local case_root="${run_root}/${label}"
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
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="${STATIC_PLAN}" \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASSES_FILE="${PLAN_CLASSES}" \
      VLLM_CPU_OFFLOAD_PLAN_CLASS_PREFILL_SOURCES=1 \
      VLLM_CPU_OFFLOAD_PLAN_CLASS_IMMUTABLE_CPU_SHADOW=1 \
      VLLM_CPU_OFFLOAD_PLAN_CLASS_PRECOMPILE=1 \
      VLLM_CPU_OFFLOAD_PLAN_CLASS_QUIESCED_DELTA=1 \
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
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup \
      VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${INCREMENTAL_PLAN}" \
      VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN=0 \
      VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0 \
      VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0 \
      VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST= \
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
      VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS="${initial_target}" \
      VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS="${initial_target}" \
      VLLM_UVM_KV_BACKUP_RESERVE_BLOCKS="${reserve}" \
      VLLM_UVM_KV_BACKUP_GROW_BLOCKS="${BACKUP_GROW_BLOCKS}" \
      VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0 \
      VLLM_UVM_KV_AUTO_SCALE_ENABLE="${autoscale}" \
      VLLM_UVM_KV_AUTO_SCALE_BACKUP_ONLY="${tiered_batch}" \
      VLLM_UVM_KV_AUTO_SCALE_BACKUP_PRESSURE_THRESHOLD_BLOCKS="${BACKUP_PRESSURE_THRESHOLD_BLOCKS}" \
      VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_MAP="${PLAN_CLASS_MAP}" \
      VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_BEFORE_GROW="${tiered_batch}" \
      VLLM_UVM_KV_AUTO_SCALE_LOW_PRESSURE_STREAK=3 \
      VLLM_UVM_KV_AUTO_SCALE_GROW_FACTOR=2 \
      VLLM_UVM_KV_AUTO_SCALE_SHRINK_DIVISOR=2 \
      VLLM_UVM_KV_AUTO_SCALE_HIGH_WATERMARK=0.80 \
      VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS="${AUTOSCALE_MIN_TARGET_BLOCKS}" \
      VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS="${AUTOSCALE_MAX_TARGET_BLOCKS}" \
      VLLM_UVM_KV_AUTO_SCALE_SAFETY_BLOCKS=0 \
      VLLM_UVM_KV_AUTO_SCALE_MIN_RESIDENCY_STEPS=0 \
      VLLM_UVM_KV_AUTO_SCALE_RESIZE_GRANULARITY_BLOCKS=1 \
      VLLM_UVM_KV_AUTO_SCALE_EXPERT_RESTORE_ENABLE=0 \
      VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_ENABLE=1 \
      VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_EXECUTOR=cpp_batch \
      VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_SYNC=1 \
      VLLM_UVM_POST_WARMUP_KV_OFFLOAD_EXECUTOR=cpp_batch \
      VLLM_UVM_KV_ADAPTIVE_FAIL_FAST=1 \
      VLLM_UVM_POST_WARMUP_REBALANCE_FAIL_FAST=1 \
      VLLM_UVM_THREE_QUEUE_QUALIFICATION="${qualification}" \
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
        --expert-plan "${STATIC_PLAN}" \
        --offload-manifest "${manifest}" \
        --output "${case_root}/result.json" \
        --arm D \
        --batch-size 1 \
        --cudagraph-capture-sizes "1,${AUTOSCALE_LOW_PROMPT_TOKENS},314" \
        --warmup-decode-steps 1 \
        --measured-decode-steps 3 \
        --kv-autoscale-roundtrip \
        --kv-autoscale-low-prompt-tokens "${AUTOSCALE_LOW_PROMPT_TOKENS}" \
        --kv-autoscale-low-output-tokens "${AUTOSCALE_LOW_OUTPUT_TOKENS}" \
        --kv-cache-memory-bytes "${KV_MAX_BYTES}" \
        --cpu-offload-gb 30.0 \
        --max-model-len 512 \
        --max-num-seqs 1 \
        --max-num-batched-tokens 1024 \
        --gpu-memory-utilization 0.97 \
        --seed 0 \
        --expected-parameter-count 96 \
        --expected-full-parameter-bytes 28991029248 \
        --expected-selected-bytes 6568280064
  ) >"${case_root}/run.log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "${rc}" >"${case_root}/returncode.txt"
  wait_gpu_clean || die_three_queue "GPU did not clean after ${label}"
  gpu_state_json "${case_root}/gpu.after.json"
  if (( rc != 0 )); then
    tail -n 300 "${case_root}/run.log" >&2 || true
  fi
  return "${rc}"
}

run_qualification_three_queue() {
  [[ "${ALLOW_GPU_RUN:-0}" == 1 ]] \
    || die_three_queue "GPU execution requires ALLOW_GPU_RUN=1"
  static_preflight_three_queue
  runtime_preflight
  local run_root="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_qualification}"
  [[ ! -e "${run_root}" ]] || die_three_queue "RUN_ROOT already exists: ${run_root}"
  mkdir -p "${run_root}"
  run_root="$(cd "${run_root}" && pwd)"
  RUN_ROOT_ACTIVE="${run_root}"

  "${VLLM_PY}" - "${run_root}/protocol.json" "${EXPERT_STRATEGY}" \
    "${STATIC_PLAN}" "${SUPERSET_PLAN}" "${INCREMENTAL_PLAN}" \
    "${TIER30_PLAN}" "${PLAN_CLASSES}" \
    "${BACKUP_PRESSURE_THRESHOLD_BLOCKS}" <<'PY'
import hashlib
import json
import pathlib
import sys

(
    output,
    strategy,
    static_text,
    superset_text,
    incremental_text,
    tier30_text,
    classes_text,
    pressure_threshold_text,
) = sys.argv[1:]
paths = {
    "static_cold_plan": pathlib.Path(static_text).resolve(),
    "superset_cold_plan": pathlib.Path(superset_text).resolve(),
    "incremental_wave_plan": pathlib.Path(incremental_text).resolve(),
    "cold30_tier_plan": pathlib.Path(tier30_text).resolve(),
    "plan_class_manifest": pathlib.Path(classes_text).resolve(),
}
documents = {
    key: json.loads(path.read_text(encoding="utf-8"))
    for key, path in paths.items()
}
payload = {
    "schema": "eurosys_e10_three_queue_single_round_protocol_v2",
    "qualification_only": True,
    "statistical_repetitions_per_case": 1,
    "case_order": ["baseline", "three_queue"],
    "expert_strategy": strategy,
    "strategy_definition": (
        "per-layer minimum frozen calibration-decode route count; "
        "ties by expert_id"
    ),
    "cases": {
        "baseline": {
            "fixed_target_blocks": 40,
            "backup": False,
            "autoscale": False,
            "expert_transition": False,
        },
        "three_queue": {
            "initial_target_blocks": 24,
            "minimum_target_blocks": 20,
            "maximum_target_blocks": 40,
            "backup_reserve_blocks": 8,
            "backup_pressure_threshold_blocks": int(pressure_threshold_text),
            "autoscale": True,
            "prediction_enabled": False,
            "growth_trigger_policy": "backup_threshold_only",
            "expert_transition": "grouped_plan_class",
            "optimizations": {
                "startup_cpu_source_prefill": True,
                "immutable_cpu_shadow_cutover": True,
                "precompiled_transition_instructions": True,
                "quiesced_delta_alias_update": True,
            },
            "plan_class_map": {"24": "cold29", "40": "cold30"},
            "tier_selected_bytes": {
                "cold29": 6568280064,
                "cold30": 6794772480,
            },
        },
    },
    "inputs": {
        key: {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "selection_rule": documents[key].get("selection_rule"),
            "metadata": documents[key].get("metadata"),
        }
        for key, path in paths.items()
    },
}
pathlib.Path(output).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
  git -C "${REPO_ROOT}" status --short >"${run_root}/git_status.txt"

  module_snapshot "${run_root}/module.before.json" "${STOCK_UVM}"
  load_custom_module
  module_snapshot "${run_root}/module.custom.json" "${CUSTOM_UVM}"
  if [[ "${E10_PHASE_F_ROUTE_LOG_CAPTURE:-0}" == full ]]; then
    printf '1\n' | sudo tee \
      /sys/module/nvidia_uvm/parameters/uvm_phase_f_route_log_enable \
      >/dev/null
  fi
  wait_gpu_clean || die_three_queue "GPU is not clean before qualification"
  cp -- "${STATS_PATH}" "${run_root}/latency_stats.before.txt"
  sha256sum \
    "${MODEL}/config.json" "${FIXTURE}" "${NAMES}" \
    "${STATIC_PLAN}" "${SUPERSET_PLAN}" "${INCREMENTAL_PLAN}" \
    "${TIER30_PLAN}" "${PLAN_CLASSES}" \
    "${SCRIPT_DIR}/prepare_cold_frequency_tiers.py" \
    "${RUNNER}" "${VERIFY}" "${BASH_SOURCE[0]}" \
    "${VLLM_ROOT}/vllm/v1/engine/core.py" \
    "${VLLM_ROOT}/vllm/v1/worker/gpu_model_runner.py" \
    "${VLLM_ROOT}/vllm/model_executor/models/utils.py" \
    "${VLLM_ROOT}/vllm/device_allocator/uvm_static_alias.py" \
    "${VLLM_ROOT}/vllm/model_executor/layers/fused_moe/layer.py" \
    "${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c" \
    "${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h" \
    "${CUSTOM_UVM}" "${STOCK_UVM}" >"${run_root}/input_sha256.txt"
  local kernel_epoch journal_rc baseline_rc queue_rc
  kernel_epoch="$(date +%s)"

  if [[ "${E10_DIAGNOSTIC_SKIP_BASELINE:-0}" == 1 ]]; then
    baseline_rc=0
  elif run_three_queue_case \
      "${run_root}" baseline "${BASELINE_TARGET_BLOCKS}" 0 0 0 0; then
      baseline_rc=0
  else
    baseline_rc=$?
  fi
  if (( baseline_rc == 0 )); then
    if run_three_queue_case \
      "${run_root}" three_queue "${QUEUE_INITIAL_TARGET_BLOCKS}" \
      "${BACKUP_RESERVE_BLOCKS}" 1 1 1; then
      queue_rc=0
    else
      queue_rc=$?
    fi
  else
    queue_rc=99
  fi
  set +e
  sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
    >"${run_root}/kernel.since.log" 2>"${run_root}/kernel.since.err"
  journal_rc=$?
  set -e
  printf '%s\n' "${journal_rc}" >"${run_root}/kernel_log_returncode.txt"

  wait_gpu_clean || die_three_queue "GPU did not clean after qualification"
  cp -- "${STATS_PATH}" "${run_root}/latency_stats.after.txt"
  if [[ "${E10_PHASE_F_ROUTE_LOG_CAPTURE:-0}" != 0 ]]; then
    sudo cp -- /proc/driver/nvidia-uvm/phase_f_route_log \
      "${run_root}/phase_f_route.log"
    sudo chown "$(id -u):$(id -g)" "${run_root}/phase_f_route.log"
  fi
  restore_stock || die_three_queue "failed to restore stock nvidia_uvm"
  module_snapshot "${run_root}/module.after.json" "${STOCK_UVM}"
  restart_exporters
  (( journal_rc == 0 )) || die_three_queue "kernel journal collection failed"
  (( baseline_rc == 0 )) || die_three_queue "baseline runner failed"
  (( queue_rc == 0 )) || die_three_queue "three-queue runner failed"
  "${VLLM_PY}" "${VERIFY}" "${run_root}" \
    --output "${run_root}/verification.json"
  echo "RUN_ROOT=${run_root}"
  echo "verification=${run_root}/verification.json"
}

case "${1:-help}" in
  static-preflight)
    static_preflight_three_queue
    ;;
  qualify|qualification|run)
    run_qualification_three_queue
    ;;
  verify)
    [[ $# -eq 2 ]] || die_three_queue "verify requires RUN_ROOT"
    "${VLLM_PY}" "${VERIFY}" "$2" --output "$2/verification.json"
    ;;
  *)
    echo "usage: $0 {static-preflight|qualify|verify RUN_ROOT}" >&2
    exit 2
    ;;
esac
