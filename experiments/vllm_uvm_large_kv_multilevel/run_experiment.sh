#!/usr/bin/env bash
set -euo pipefail

LARGE_KV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LARGE_KV_REPO_ROOT="$(cd "${LARGE_KV_SCRIPT_DIR}/../.." && pwd)"
LARGE_KV_BASE_HARNESS="${LARGE_KV_REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
# The sourced base harness defines its own VERIFY default.  Preserve only an
# explicit caller override so this derived qualification cannot accidentally
# run the base experiment's incompatible verifier.
LARGE_KV_VERIFY_OVERRIDE="${VERIFY-}"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
# shellcheck source=/dev/null
source "${LARGE_KV_BASE_HARNESS}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY

SCRIPT_DIR="${LARGE_KV_SCRIPT_DIR}"
REPO_ROOT="${LARGE_KV_REPO_ROOT}"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
RUNNER="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/benchmark_phase_switch_steps.py"
VERIFY="${LARGE_KV_VERIFY_OVERRIDE:-${SCRIPT_DIR}/verify_experiment.py}"
PLAN_GENERATOR="${SCRIPT_DIR}/prepare_qwen_large_plan.py"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
FIXTURE="${FIXTURE:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json}"
NAMES="${NAMES:-${SCRIPT_DIR}/inputs/exact_parameter_names.txt}"
INPUT_MANIFEST="${INPUT_MANIFEST:-${SCRIPT_DIR}/inputs/large_kv_manifest.json}"
EXPERT_PLAN="${EXPERT_PLAN:-${SCRIPT_DIR}/inputs/cold_decode_plan_base70.json}"
INCREMENTAL_WAVE_PLAN="${INCREMENTAL_WAVE_PLAN:-${SCRIPT_DIR}/inputs/incremental_expert_waves.json}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_ATTN}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-1,314}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
EXPECTED_PARAMETER_COUNT="${EXPECTED_PARAMETER_COUNT:-96}"
EXPECTED_LAYER_COUNT="${EXPECTED_LAYER_COUNT:-48}"
EXPECTED_COLD_EXPERTS_PER_LAYER="${EXPECTED_COLD_EXPERTS_PER_LAYER:-70}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-30.0}"

# Qwen3 uses a 1.5 MiB physical KV block. Targets exclude vLLM's permanently
# reserved null block; the corresponding physical GPU prefixes are within
# 512 KiB of 2/4/8/10 GiB.
KV_BLOCK_BYTES="${KV_BLOCK_BYTES:-1572864}"
KV_MAX_BYTES="${KV_MAX_BYTES:-10737942528}"
KV_2G_BLOCKS="${KV_2G_BLOCKS:-1364}"
KV_4G_BLOCKS="${KV_4G_BLOCKS:-2730}"
KV_8G_BLOCKS="${KV_8G_BLOCKS:-5460}"
KV_10G_BLOCKS="${KV_10G_BLOCKS:-6826}"
EXPECTED_FULL_PARAMETER_BYTES="${EXPECTED_FULL_PARAMETER_BYTES:-28991029248}"
EXPECTED_SELECTED_BYTES="${EXPECTED_SELECTED_BYTES:-15854469120}"
EXPECTED_TOTAL_MODEL_BYTES="${EXPECTED_TOTAL_MODEL_BYTES:-32444792832}"
EXPECTED_ALIAS_BACKING_BYTES="${EXPECTED_ALIAS_BACKING_BYTES:-660602880}"
WORKSPACE_RESERVE_BYTES="${WORKSPACE_RESERVE_BYTES:-536870912}"
MIN_RAW_HEADROOM_BYTES="${MIN_RAW_HEADROOM_BYTES:-2147483648}"
TIMEOUT_SEC="${TIMEOUT_SEC:-7200}"
CPUSET="${CPUSET:-0-13}"

die_large_kv() {
  echo "large-kv multilevel: $*" >&2
  exit 1
}

require_large_kv_file() {
  [[ -f "$1" ]] || die_large_kv "missing file: $1"
}

static_preflight_large_kv() {
  local path
  for path in "${RUNNER}" "${VERIFY}" "${MODEL}/config.json" "${FIXTURE}" \
    "${NAMES}" "${INPUT_MANIFEST}" "${EXPERT_PLAN}" \
    "${INCREMENTAL_WAVE_PLAN}" "${PLAN_GENERATOR}"; do
    require_large_kv_file "${path}"
  done
  [[ "$(wc -l <"${NAMES}")" -eq "${EXPECTED_PARAMETER_COUNT}" ]] \
    || die_large_kv "unexpected exact parameter-name count"
  bash -n "${BASH_SOURCE[0]}"
  "${VLLM_PY}" -m py_compile "${RUNNER}" "${VERIFY}" "${PLAN_GENERATOR}"
  "${VLLM_PY}" - "${EXPERT_PLAN}" "${INCREMENTAL_WAVE_PLAN}" \
    "${INPUT_MANIFEST}" \
    "${KV_BLOCK_BYTES}" \
    "${KV_MAX_BYTES}" "${KV_2G_BLOCKS}" "${KV_4G_BLOCKS}" \
    "${KV_8G_BLOCKS}" "${KV_10G_BLOCKS}" \
    "${EXPECTED_SELECTED_BYTES}" "${EXPECTED_LAYER_COUNT}" \
    "${EXPECTED_COLD_EXPERTS_PER_LAYER}" <<'PY'
import hashlib
import json
import sys

(plan_path, wave_path, manifest_path, block_bytes, maximum_bytes, target_2g, target_4g,
 target_8g, target_10g, selected_bytes, layer_count, layer_width) = sys.argv[1:]
block_bytes = int(block_bytes)
maximum_bytes = int(maximum_bytes)
targets = {
    2: int(target_2g),
    4: int(target_4g),
    8: int(target_8g),
    10: int(target_10g),
}
plan = json.load(open(plan_path, encoding="utf-8"))
waves = json.load(open(wave_path, encoding="utf-8"))
manifest = json.load(open(manifest_path, encoding="utf-8"))
metadata = plan.get("metadata") or {}
actual_selected = metadata.get("selected_main_bytes", plan.get("selected_main_bytes"))
selected_by_layer = plan["selected_by_layer"]
assert int(actual_selected) == int(selected_bytes)
assert len(selected_by_layer) == int(layer_count)
assert {len(value) for value in selected_by_layer.values()} == {int(layer_width)}
assert sum(map(len, selected_by_layer.values())) == int(layer_count) * int(layer_width)
assert set(selected_by_layer) == {str(layer) for layer in range(int(layer_count))}
assert all(len(experts) == len(set(experts)) for experts in selected_by_layer.values())
cold = {
    (int(layer), int(expert))
    for layer, experts in selected_by_layer.items()
    for expert in experts
}
offload = {
    (int(item["layer_id"]), int(item["expert_id"]))
    for item in plan["offload_plan"]
}
hot = {
    (int(item["layer_id"]), int(item["expert_id"]))
    for item in plan["hot_plan"]
}
universe = {
    (layer, expert)
    for layer in range(int(layer_count))
    for expert in range(128)
}
assert cold == offload
assert len(offload) == int(layer_count) * int(layer_width)
assert len(hot) == int(layer_count) * (128 - int(layer_width))
assert not offload & hot
assert offload | hot == universe
assert int(metadata["offload_plan_records"]) == len(offload)
assert int(metadata["hot_plan_records"]) == len(hot)
plan_sha256 = hashlib.sha256(open(plan_path, "rb").read()).hexdigest()
wave_sha256 = hashlib.sha256(open(wave_path, "rb").read()).hexdigest()
assert manifest["base_plan_sha256"] == plan_sha256
assert manifest["wave_plan_sha256"] == wave_sha256
assert waves["schema"] == "qwen_large_kv_incremental_expert_waves_v1"
assert int(waves["base_experts_per_layer"]) == int(layer_width)
assert int(waves["maximum_experts_per_layer"]) == 108
assert len(waves["waves"]) == 38
expected_wave_layers = {
    f"model.layers.{layer}.mlp.experts" for layer in range(int(layer_count))
}
seen = {layer: set() for layer in expected_wave_layers}
for index, wave in enumerate(waves["waves"], 1):
    assert int(wave["wave"]) == index
    layer_experts = wave["layer_experts"]
    assert set(layer_experts) == expected_wave_layers
    for layer, expert in layer_experts.items():
        assert 0 <= int(expert) < 128
        assert int(expert) not in seen[layer]
        seen[layer].add(int(expert))
for tier_name, width, wave_count in (
    ("2g", 70, 0), ("4g", 80, 10), ("8g", 99, 29), ("10g", 108, 38)
):
    tier = manifest["tiers"][tier_name]
    assert int(tier["experts_per_layer"]) == width
    assert int(tier["incremental_waves"]) == wave_count
assert int(manifest["tiers"]["2g"]["selected_main_bytes"]) == int(selected_bytes)
assert maximum_bytes == (targets[10] + 1) * block_bytes
for gib, target in targets.items():
    physical = (target + 1) * block_bytes
    assert abs(physical - gib * 2**30) <= 512 * 1024
assert list(targets.values()) == sorted(targets.values())
PY
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" -m pytest -q \
      tests/v1/core/test_uvm_dynamic_pool_policy.py \
      tests/v1/core/test_uvm_dynamic_pool_queues.py \
      tests/v1/engine/test_uvm_adaptive_kv.py \
      tests/v1/worker/test_uvm_post_warmup_rebalance.py \
      -k 'dynamic_pool or kv_resize or physical_span_snapshot or explicit_incremental_wave_does_not_enable_init_stage_m or runtime_expert_registry_skips_static_alias_owned_ranges or incremental_wave_rejects_static_alias_owned_expert'
  )
  echo "large-kv multilevel static preflight: OK"
}

runtime_budget_preflight_large_kv() {
  "${VLLM_PY}" - "${EXPECTED_TOTAL_MODEL_BYTES}" \
    "${EXPECTED_SELECTED_BYTES}" "${KV_MAX_BYTES}" \
    "${EXPECTED_ALIAS_BACKING_BYTES}" "${WORKSPACE_RESERVE_BYTES}" \
    "${MIN_RAW_HEADROOM_BYTES}" <<'PY'
import sys
import torch

model, cold, kv, backing, workspace, safety = map(int, sys.argv[1:])
device = int(torch.cuda.get_device_properties(0).total_memory)
hot = model - cold
required = hot + kv + backing + workspace + safety
remaining = device - required
if remaining < 0:
    raise SystemExit(
        "10 GiB tier violates conservative HBM budget: "
        f"device={device} resident_model={hot} kv={kv} backing={backing} "
        f"workspace={workspace} safety={safety} shortfall={-remaining}"
    )
print(
    "large-kv HBM preflight: "
    f"device={device} resident_model={hot} kv={kv} backing={backing} "
    f"workspace={workspace} safety={safety} remaining={remaining}"
)
PY
}

write_sequence_large_kv() {
  local output="$1" mode="$2"
  "${VLLM_PY}" - "${output}" "${mode}" "${KV_2G_BLOCKS}" \
    "${KV_4G_BLOCKS}" "${KV_8G_BLOCKS}" "${KV_10G_BLOCKS}" <<'PY'
import json
import sys

output, mode = sys.argv[1:3]
targets = {
    "2g": int(sys.argv[3]),
    "4g": int(sys.argv[4]),
    "8g": int(sys.argv[5]),
    "10g": int(sys.argv[6]),
}

def records(prefix, tier):
    waves = {"2g": 0, "4g": 10, "8g": 29, "10g": 38}
    return [
        {
            "label": f"{prefix}_{tier}_{index}",
            "batch_size": 1,
            "max_tokens": 23,
            "decode_warmup_steps": 5,
            "expected_pressure_class": "explicit",
            "target_blocks": targets[tier],
            "expert_waves": waves[tier],
        }
        for index in range(1, 4)
    ]

if mode == "stepped":
    payload = []
    for prefix, tier in (
        ("ascending", "4g"),
        ("ascending", "8g"),
        ("ascending", "10g"),
        ("descending", "8g"),
        ("descending", "4g"),
        ("descending", "2g"),
    ):
        payload.extend(records(prefix, tier))
elif mode in targets:
    payload = records("direct", mode)
else:
    raise SystemExit(f"unknown sequence mode: {mode}")

with open(output, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
}

run_case_large_kv() {
  local run_root="$1" label="$2" mode="$3"
  local case_root="${run_root}/${label}"
  local sequence="${case_root}/physical_kv_sequence.json"
  local manifest="${case_root}/cpu_offload_manifest.jsonl"
  mkdir -p "${case_root}"
  write_sequence_large_kv "${sequence}" "${mode}"
  : >"${manifest}"
  gpu_state_json "${case_root}/gpu.before.json"

  set +e
  (
    cd "${VLLM_ROOT}"
    env \
      VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      VLLM_ATTENTION_BACKEND="${ATTENTION_BACKEND}" \
      VLLM_CPU_OFFLOAD_BACKEND=copy \
      VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${NAMES}" \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1 \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="${EXPERT_PLAN}" \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288 \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=auto \
      VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1 \
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
      VLLM_UVM_ORIGINAL_VA_PHASE_MODE=prefill_graph_remote_decode \
      VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES=201326592 \
      VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ=0 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu \
      VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=remote_read \
      VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1 \
      VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=1 \
      VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0 \
      VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0 \
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
      VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS="${KV_2G_BLOCKS}" \
      VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS="${KV_2G_BLOCKS}" \
      VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0 \
      VLLM_UVM_KV_AUTO_SCALE_ENABLE=0 \
      VLLM_UVM_POST_WARMUP_KV_OFFLOAD_EXECUTOR=cpp_batch \
      VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_ENABLE=1 \
      VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_EXECUTOR=cpp_batch \
      VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_SYNC=1 \
      VLLM_UVM_KV_ADAPTIVE_FAIL_FAST=1 \
      VLLM_UVM_POST_WARMUP_REBALANCE_FAIL_FAST=1 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup \
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
        --expert-plan "${EXPERT_PLAN}" \
        --incremental-expert-wave-plan "${INCREMENTAL_WAVE_PLAN}" \
        --dynamic-kv-sequence "${sequence}" \
        --allow-explicit-dynamic-kv-targets \
        --offload-manifest "${manifest}" \
        --output "${case_root}/result.json" \
        --arm D \
        --batch-size 1 \
        --cudagraph-capture-sizes "${CUDAGRAPH_CAPTURE_SIZES}" \
        --warmup-decode-steps 5 \
        --measured-decode-steps 16 \
        --kv-cache-memory-bytes "${KV_MAX_BYTES}" \
        --cpu-offload-gb "${CPU_OFFLOAD_GB}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs 1 \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --gpu-memory-utilization 0.97 \
        --seed 0 \
        --expected-parameter-count "${EXPECTED_PARAMETER_COUNT}" \
        --expected-full-parameter-bytes "${EXPECTED_FULL_PARAMETER_BYTES}" \
        --expected-selected-bytes "${EXPECTED_SELECTED_BYTES}"
  ) >"${case_root}/run.log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "${rc}" >"${case_root}/returncode.txt"
  wait_gpu_clean || die_large_kv "GPU did not clean after ${label}"
  gpu_state_json "${case_root}/gpu.after.json"
  if (( rc != 0 )); then
    tail -n 260 "${case_root}/run.log" >&2 || true
  fi
  return "${rc}"
}

run_qualification_large_kv() {
  [[ "${ALLOW_GPU_RUN:-0}" == 1 ]] \
    || die_large_kv "GPU execution requires ALLOW_GPU_RUN=1"
  static_preflight_large_kv
  runtime_preflight
  runtime_budget_preflight_large_kv
  local run_root="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_qualification}"
  [[ ! -e "${run_root}" ]] || die_large_kv "RUN_ROOT already exists: ${run_root}"
  mkdir -p "${run_root}"
  run_root="$(cd "${run_root}" && pwd)"
  RUN_ROOT_ACTIVE="${run_root}"
  nvidia-smi -i 0 --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader,nounits >"${run_root}/hardware.csv"

  module_snapshot "${run_root}/module.before.json" "${STOCK_UVM}"
  load_custom_module
  module_snapshot "${run_root}/module.custom.json" "${CUSTOM_UVM}"
  wait_gpu_clean || die_large_kv "GPU is not clean before qualification"
  gpu_state_json "${run_root}/gpu.before.json"
  local kernel_epoch journal_rc case_rc=0 label mode
  local case_labels="${CASE_LABELS:-stepped direct_2g direct_4g direct_8g direct_10g}"
  kernel_epoch="$(date +%s)"

  for label in ${case_labels}; do
    mode="${label#direct_}"
    [[ "${label}" == stepped ]] && mode=stepped
    if (( case_rc == 0 )) && run_case_large_kv \
      "${run_root}" "${label}" "${mode}"; then
      :
    else
      case_rc=$?
      break
    fi
  done

  set +e
  sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
    >"${run_root}/kernel.since.log" 2>"${run_root}/kernel.since.err"
  journal_rc=$?
  set -e
  printf '%s\n' "${journal_rc}" >"${run_root}/kernel_log_returncode.txt"

  wait_gpu_clean || die_large_kv "GPU did not clean after qualification"
  gpu_state_json "${run_root}/gpu.after.json"
  restore_stock || die_large_kv "failed to restore stock nvidia_uvm"
  module_snapshot "${run_root}/module.after.json" "${STOCK_UVM}"
  restart_exporters
  (( journal_rc == 0 )) || die_large_kv "kernel journal collection failed"
  (( case_rc == 0 )) || die_large_kv "large-KV runner failed"
  if [[ "${VERIFY_AFTER_RUN:-1}" == 1 ]]; then
    "${VLLM_PY}" "${VERIFY}" "${run_root}" \
      --output "${run_root}/verification.json"
  fi
  echo "RUN_ROOT=${run_root}"
  [[ "${VERIFY_AFTER_RUN:-1}" == 1 ]] \
    && echo "verification=${run_root}/verification.json"
  return 0
}

if [[ "${VLLM_LARGE_KV_SOURCE_ONLY:-0}" == 1 ]]; then
  return 0 2>/dev/null || exit 0
fi

case "${1:-help}" in
  static-preflight)
    static_preflight_large_kv
    ;;
  qualify|qualification|run)
    run_qualification_large_kv
    ;;
  verify)
    [[ $# -eq 2 ]] || die_large_kv "verify requires RUN_ROOT"
    "${VLLM_PY}" "${VERIFY}" "$2" --output "$2/verification.json"
    ;;
  *)
    echo "usage: $0 {static-preflight|qualify|verify RUN_ROOT}" >&2
    exit 2
    ;;
esac
