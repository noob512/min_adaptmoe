#!/usr/bin/env bash
set -euo pipefail

E7_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E7_REPO_ROOT="$(cd "${E7_SCRIPT_DIR}/../.." && pwd)"
E7_BASE_HARNESS="${E7_REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
export STATIC_ALIAS_DUAL_VIEW_ENABLE="${STATIC_ALIAS_DUAL_VIEW_ENABLE:-1}"
export STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL="${STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL:-1}"
export PHASE_F_ROUTE_LOG_ENABLE="${PHASE_F_ROUTE_LOG_ENABLE:-0}"
# shellcheck source=/dev/null
source "${E7_BASE_HARNESS}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY

SCRIPT_DIR="${E7_SCRIPT_DIR}"
REPO_ROOT="${E7_REPO_ROOT}"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_PY:-${VLLM_ROOT}/.venv/bin/python}"
RUNNER="${SCRIPT_DIR}/run_e7_independent_arm.py"
VERIFY="${SCRIPT_DIR}/verify_e7_independent_cold_qualification.py"
SOURCE_RESULT="${SOURCE_RESULT:-${SCRIPT_DIR}/artifacts/e7-workload-shift-REPRO-20260829-attempt-03-qualification/result.json}"
SOURCE_RESULT_SHA256="3190d36d37a0ca9eaccb5bbeb35d4dcfc3b9f553815e0ea7240df1ade917187c"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
FIXTURE_ROOT="${SCRIPT_DIR}/inputs/generated/eurosys-pa-v2-20260824/qwen3_30b_a3b_fp8"
SHIFT_FIXTURE="${SHIFT_FIXTURE:-${FIXTURE_ROOT}/shift_short_dialogue.json}"
NAMES="${REPO_ROOT}/experiments/qwen_decode_six_mechanism_compare/inputs/exact_parameter_names.txt"
STATIC_PLAN="${REPO_ROOT}/experiments/vllm_uvm_physical_kv_pool_resize/inputs/cold_decode_plan_6g.json"
SUPERSET_PLAN="${REPO_ROOT}/experiments/vllm_uvm_physical_kv_pool_resize/inputs/cold_decode_plan_7g.json"
TIMEOUT_SEC="${TIMEOUT_SEC:-7200}"
CPUSET="${CPUSET:-0-13}"
CPU_THREADS="${CPU_THREADS:-14}"
ARM_ORDER=(
  frequency_cold
  random
  frozen_static
  gpu_heat_cpu_ewma
  oracle
  hot_adversarial
)

die_e7() {
  echo "E7 independent-cold qualification: $*" >&2
  exit 1
}

require_e7_file() {
  [[ -f "$1" ]] || die_e7 "missing file: $1"
}

validate_source_and_inputs() {
  "${VLLM_PY}" - "${SOURCE_RESULT}" "${SOURCE_RESULT_SHA256}" \
    "${SHIFT_FIXTURE}" "${STATIC_PLAN}" "${SUPERSET_PLAN}" <<'PY'
import hashlib, json, pathlib, sys
source_text, expected_sha, shift_text, static_text, superset_text = sys.argv[1:]
source_path = pathlib.Path(source_text).resolve()
raw = source_path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == expected_sha
source = json.loads(raw)
assert source["schema"] == "eurosys_e7_workload_shift_qualification_v1"
assert source["status"] == "ok"
assert source["statistical_repetitions"] == 1
shift = json.loads(pathlib.Path(shift_text).read_text(encoding="utf-8"))
assert shift["metadata"]["fixture_role"] == "shift"
assert shift["metadata"]["frozen_prompt_token_ids"] is True
assert len(shift["requests"]) == 32
assert {len(item["prompt_token_ids"]) for item in shift["requests"]} == {128}
static = json.loads(pathlib.Path(static_text).read_text(encoding="utf-8"))["offload_plan"]
superset = json.loads(pathlib.Path(superset_text).read_text(encoding="utf-8"))["offload_plan"]
a = {(int(item["layer_id"]), int(item["expert_id"])) for item in static}
b = {(int(item["layer_id"]), int(item["expert_id"])) for item in superset}
assert a < b
diff = b - a
assert len(diff) == 192
assert all(sum(layer == candidate for candidate, _ in diff) == 4 for layer in range(48))
PY
}

static_preflight_e7() {
  require_e7_file "${RUNNER}"
  require_e7_file "${VERIFY}"
  require_e7_file "${SOURCE_RESULT}"
  require_e7_file "${MODEL}/config.json"
  require_e7_file "${SHIFT_FIXTURE}"
  require_e7_file "${NAMES}"
  require_e7_file "${STATIC_PLAN}"
  require_e7_file "${SUPERSET_PLAN}"
  require_e7_file "${CUSTOM_UVM}"
  [[ -x "${VLLM_PY}" ]] || die_e7 "missing vLLM Python: ${VLLM_PY}"
  [[ "$(wc -l <"${NAMES}")" -eq 96 ]] || die_e7 "expected 96 exact names"
  command -v nvidia-smi >/dev/null || die_e7 "nvidia-smi unavailable"
  command -v timeout >/dev/null || die_e7 "timeout unavailable"
  command -v taskset >/dev/null || die_e7 "taskset unavailable"
  command -v flock >/dev/null || die_e7 "flock unavailable"
  bash -n "${BASH_SOURCE[0]}"
  "${VLLM_PY}" -m py_compile "${RUNNER}" "${VERIFY}"
  validate_source_and_inputs
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" -m pytest -q \
      tests/v1/engine/test_uvm_adaptive_kv.py \
      tests/test_uvm_explicit_expert_regions.py \
      -k 'dynamic_expert_swap_plan or dynamic_swap'
  )
  echo "E7 independent-cold static preflight: OK"
}

runtime_preflight_e7() {
  static_preflight_e7
  [[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die_e7 "GPU execution requires ALLOW_GPU_RUN=1"
  STOCK_UVM="$(modinfo -F filename nvidia-uvm)"
  STOCK_SRCVERSION="$(modinfo -F srcversion nvidia-uvm | tr -d '[:space:]')"
  STOCK_SHA256="$(sha256sum "${STOCK_UVM}" | awk '{print $1}')"
  CUSTOM_SRCVERSION="$(modinfo -F srcversion "${CUSTOM_UVM}" | tr -d '[:space:]')"
  CUSTOM_SHA256="$(sha256sum "${CUSTOM_UVM}" | awk '{print $1}')"
  [[ -n "${STOCK_SRCVERSION}" && -n "${CUSTOM_SRCVERSION}" ]] \
    || die_e7 "module srcversion absent"
  [[ "$(loaded_srcversion)" == "${STOCK_SRCVERSION}" ]] \
    || die_e7 "initial UVM is not stock"
  MODULE_CYCLE_ARMED=1
  capture_and_stop_exporters
  wait_gpu_clean || die_e7 "GPU must start at 0 MiB with no compute/UVM client"
  exec 9>"/tmp/vllm_uvm_prefill_alias_decode_remote_graph.lock"
  flock -n 9 || die_e7 "another phase-switch experiment holds the lock"
}

freeze_targets() {
  local output="$1"
  "${VLLM_PY}" - "${SOURCE_RESULT}" "${SOURCE_RESULT_SHA256}" "${output}" <<'PY'
import hashlib, json, pathlib, sys, time
source_path = pathlib.Path(sys.argv[1]).resolve()
expected_sha = sys.argv[2]
output_path = pathlib.Path(sys.argv[3])
raw = source_path.read_bytes()
actual_sha = hashlib.sha256(raw).hexdigest()
assert actual_sha == expected_sha
source = json.loads(raw)
arms = {arm["name"]: arm for arm in source["arms"]}
targets = {
    "frequency_cold": source["calibration"]["frequency_cold_target"],
    "random": arms["random"]["start_target"],
    "frozen_static": arms["frozen_static"]["start_target"],
    "gpu_heat_cpu_ewma": arms["gpu_heat_cpu_ewma"]["start_target"],
    "oracle": source["oracle"]["target"],
    "hot_adversarial": source["oracle"]["hot_adversarial_target"],
}
assert list(targets) == [
    "frequency_cold", "random", "frozen_static", "gpu_heat_cpu_ewma",
    "oracle", "hot_adversarial",
]
assert all(len(target) == 48 for target in targets.values())
assert all(
    len(values) == 4
    for target in targets.values()
    for values in target.values()
)
payload = {
    "schema": "eurosys_e7_independent_cold_targets_v1",
    "frozen_before_run": True,
    "frozen_time_unix": time.time(),
    "source_result": str(source_path),
    "source_result_sha256": actual_sha,
    "source_verification_sha256": "77e1d258e7f236ab35db00acd88d00ba8b08f98f19286003526b8b2d929ffb29",
    "source_route_counts_sha256": source["reference_route_counts_sha256"],
    "oracle_provenance": source["oracle"]["provenance"],
    "oracle_future_knowledge": True,
    "targets": targets,
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
PY
}

run_one_arm() {
  local run_root="$1" arm="$2" frozen_targets="$3"
  local arm_root="${run_root}/arms/${arm}"
  local manifest="${arm_root}/cpu_offload_manifest.jsonl"
  local runner_rc
  mkdir -p "${arm_root}"
  : >"${manifest}"
  wait_gpu_clean || die_e7 "GPU is not clean before arm ${arm}"
  gpu_state_json "${arm_root}/gpu.before.json"
  cp -- "${STATS_PATH}" "${arm_root}/latency_stats.before.txt"
  set +e
  (
    cd "${VLLM_ROOT}"
    env \
      VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      VLLM_ATTENTION_BACKEND=TRITON_ATTN \
      VLLM_LOGGING_LEVEL=WARNING \
      VLLM_CPU_OFFLOAD_BACKEND=copy \
      VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${NAMES}" \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1 \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="${STATIC_PLAN}" \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288 \
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=auto \
      VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1 \
      VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW=1 \
      VLLM_CPU_OFFLOAD_PLAN_CLASS_IMMUTABLE_CPU_SHADOW=1 \
      VLLM_CPU_OFFLOAD_DUAL_VIEW_FULL_PARAMETER_SHADOW=1 \
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
      VLLM_UVM_ORIGINAL_VA_PHASE_MODE=prefill_graph_remote_decode \
      VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE=1 \
      VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=1 \
      VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES=201326592 \
      VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ=0 \
      VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1 \
      VLLM_UVM_EXPERT_ENVELOPE_BYTES=524288 \
      VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=remote_read \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup \
      VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${SUPERSET_PLAN}" \
      VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN=0 \
      VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0 \
      VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=1 \
      VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=1 \
      VLLM_UVM_DYNAMIC_EXPERT_AUTO_REBALANCE_ENABLE=0 \
      VLLM_UVM_DYNAMIC_EXPERT_HEAT_EWMA_ALPHA=0.25 \
      VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST="${arm_root}/placement.json" \
      VLLM_UVM_WEIGHT_EXECUTOR=cpp_batch \
      VLLM_CPU_OFFLOAD_COPY_TRACE=0 \
      VLLM_CPU_OFFLOAD_COPY_SYNC=0 \
      VLLM_CPU_OFFLOAD_MANIFEST_FILE="${manifest}" \
      VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 \
      VLLM_PHASE_SWITCH_CONTROLLED_HARD_EXIT_AFTER_CLEANUP=1 \
      VLLM_UVM_LOG_FILE="${arm_root}/uvm.log" \
      VLLM_USE_UVM=0 \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      TOKENIZERS_PARALLELISM=false \
      OMP_NUM_THREADS="${CPU_THREADS}" \
      MKL_NUM_THREADS="${CPU_THREADS}" \
      CUDA_VISIBLE_DEVICES=0 \
      timeout --kill-after=30s "${TIMEOUT_SEC}s" \
      taskset -c "${CPUSET}" \
      "${VLLM_PY}" "${RUNNER}" \
        --arm "${arm}" \
        --model "${MODEL}" \
        --shift-fixture "${SHIFT_FIXTURE}" \
        --static-plan "${STATIC_PLAN}" \
        --superset-plan "${SUPERSET_PLAN}" \
        --frozen-targets "${frozen_targets}" \
        --output "${arm_root}/result.json" \
        --max-tokens 32 \
        --window-size 4 \
        --ewma-alpha 0.25 \
        --max-swaps-per-layer 1 \
        --min-count-delta 1 \
        --kv-cache-memory-bytes 268435456 \
        --cpu-offload-gb 30.0 \
        --gpu-memory-utilization 0.97 \
        --seed 20260829
  ) >"${arm_root}/run.log" 2>&1
  runner_rc=$?
  set -e
  printf '%s\n' "${runner_rc}" >"${arm_root}/returncode.txt"
  sudo cat /proc/driver/nvidia-uvm/phase_f_route_log \
    >"${arm_root}/phase_f_route_log.txt" 2>"${arm_root}/phase_f_route_log.err" \
    || true
  wait_gpu_clean || die_e7 "GPU did not clean after arm ${arm}"
  gpu_state_json "${arm_root}/gpu.after.json"
  cp -- "${STATS_PATH}" "${arm_root}/latency_stats.after.txt"
  return "${runner_rc}"
}

run_e7() {
  runtime_preflight_e7
  local run_root="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/e7-independent-cold-REPRO-20260829-attempt-01-qualification}"
  [[ ! -e "${run_root}" ]] || die_e7 "RUN_ROOT already exists: ${run_root}"
  mkdir -p "${run_root}/arms"
  run_root="$(cd "${run_root}" && pwd)"
  RUN_ROOT_ACTIVE="${run_root}"
  local frozen_targets="${run_root}/frozen_targets.json"
  local kernel_epoch journal_rc arm any_failure=0
  freeze_targets "${frozen_targets}"
  module_snapshot "${run_root}/module.before.json" "${STOCK_UVM}"
  load_custom_module
  module_snapshot "${run_root}/module.custom.json" "${CUSTOM_UVM}"
  wait_gpu_clean || die_e7 "GPU is not clean after custom UVM load"
  git -C "${REPO_ROOT}" status --short >"${run_root}/git_status.txt"
  sha256sum \
    "${MODEL}/config.json" "${SHIFT_FIXTURE}" "${NAMES}" \
    "${STATIC_PLAN}" "${SUPERSET_PLAN}" "${SOURCE_RESULT}" \
    "${frozen_targets}" "${RUNNER}" "${VERIFY}" "${BASH_SOURCE[0]}" \
    "${VLLM_ROOT}/vllm/model_executor/layers/fused_moe/layer.py" \
    "${VLLM_ROOT}/vllm/v1/engine/core.py" \
    "${VLLM_ROOT}/vllm/v1/worker/gpu_model_runner.py" \
    "${CUSTOM_UVM}" "${STOCK_UVM}" >"${run_root}/input_sha256.txt"
  kernel_epoch="$(date +%s)"
  for arm in "${ARM_ORDER[@]}"; do
    echo "E7 independent-cold: starting ${arm}"
    if run_one_arm "${run_root}" "${arm}" "${frozen_targets}"; then
      echo "E7 independent-cold: completed ${arm}"
    else
      any_failure=1
      echo "E7 independent-cold: ${arm} failed; no retry will be made" >&2
    fi
  done
  set +e
  sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
    >"${run_root}/kernel.since.log" 2>"${run_root}/kernel.since.err"
  journal_rc=$?
  set -e
  printf '%s\n' "${journal_rc}" >"${run_root}/kernel_log_returncode.txt"
  restore_stock || die_e7 "failed to restore stock nvidia_uvm"
  module_snapshot "${run_root}/module.after.json" "${STOCK_UVM}"
  restart_exporters
  (( journal_rc == 0 )) || die_e7 "kernel journal collection failed"
  if (( any_failure != 0 )); then
    die_e7 "at least one arm failed; every arm was attempted exactly once; artifact=${run_root}"
  fi
  "${VLLM_PY}" "${VERIFY}" "${run_root}" \
    --output "${run_root}/verification.json" \
    >"${run_root}/verify.log" 2>"${run_root}/verify.err" \
    || {
      cat "${run_root}/verify.err" >&2 || true
      die_e7 "verification failed; artifact retained at ${run_root}"
    }
  echo "RUN_ROOT=${run_root}"
  echo "verification=${run_root}/verification.json"
}

run_e7_diagnostic_arm() {
  runtime_preflight_e7
  local arm="${E7_DIAGNOSTIC_ARM:-frequency_cold}"
  local run_root="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/e7-independent-cold-diagnostic}"
  local frozen_targets="${run_root}/frozen_targets.json"
  local valid=0 runner_rc journal_rc kernel_epoch
  local candidate
  for candidate in "${ARM_ORDER[@]}"; do
    [[ "${candidate}" == "${arm}" ]] && valid=1
  done
  (( valid == 1 )) || die_e7 "invalid E7_DIAGNOSTIC_ARM: ${arm}"
  [[ ! -e "${run_root}" ]] || die_e7 "RUN_ROOT already exists: ${run_root}"
  mkdir -p "${run_root}/arms"
  run_root="$(cd "${run_root}" && pwd)"
  RUN_ROOT_ACTIVE="${run_root}"
  frozen_targets="${run_root}/frozen_targets.json"
  freeze_targets "${frozen_targets}"
  module_snapshot "${run_root}/module.before.json" "${STOCK_UVM}"
  load_custom_module
  module_snapshot "${run_root}/module.custom.json" "${CUSTOM_UVM}"
  kernel_epoch="$(date +%s)"
  set +e
  run_one_arm "${run_root}" "${arm}" "${frozen_targets}"
  runner_rc=$?
  sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
    >"${run_root}/kernel.since.log" 2>"${run_root}/kernel.since.err"
  journal_rc=$?
  set -e
  printf '%s\n' "${journal_rc}" >"${run_root}/kernel_log_returncode.txt"
  restore_stock || die_e7 "failed to restore stock nvidia_uvm"
  module_snapshot "${run_root}/module.after.json" "${STOCK_UVM}"
  restart_exporters
  (( journal_rc == 0 )) || die_e7 "kernel journal collection failed"
  echo "RUN_ROOT=${run_root}"
  echo "diagnostic_arm=${arm}"
  return "${runner_rc}"
}

case "${1:-help}" in
  preflight) static_preflight_e7 ;;
  qualify|qualification|run) run_e7 ;;
  diagnose|diagnostic) run_e7_diagnostic_arm ;;
  verify)
    [[ $# -eq 2 ]] || die_e7 "verify requires RUN_ROOT"
    "${VLLM_PY}" "${VERIFY}" "$2" --output "$2/verification.json"
    ;;
  *)
    echo "usage: $0 {preflight|qualify|diagnose|verify RUN_ROOT}" >&2
    exit 2
    ;;
esac
