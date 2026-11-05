#!/usr/bin/env bash
set -euo pipefail

# One-process, one-run qualification of a single cold70 Prefill/Decode dual
# view. This intentionally does not activate cold80/cold89 or resize the KV
# cache. The second two-token request exercises Decode->Prefill->Decode reuse.

SINGLE_TIER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SINGLE_TIER_DIR}/../.." && pwd)"
BASE="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
export ACCESS_COUNTER_MIGRATION_ENABLE=0
export UVM_PREFETCH_ENABLE=0
export UVM_THRASHING_ENABLE=0
export STATIC_ALIAS_DUAL_VIEW_ENABLE=1
export STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL=1
# shellcheck source=/dev/null
source "${BASE}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY

VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
RUNNER="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/benchmark_phase_switch_steps.py"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
FIXTURE="${FIXTURE:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json}"
NAMES="${REPO_ROOT}/experiments/vllm_uvm_large_kv_multilevel/inputs/exact_parameter_names.txt"
COLD70="${SINGLE_TIER_DIR}/inputs/cold70.json"
PLAN_CLASSES="${SINGLE_TIER_DIR}/inputs/plan_classes_cold70_only.json"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/artifacts/e10-single-tier-full-dual-view-20260830-attempt-01-qualification}"
TIMEOUT_SEC="${TIMEOUT_SEC:-7200}"
CPUSET="${CPUSET:-0-13}"
KV_BYTES=2145386496

for path in "${RUNNER}" "${MODEL}/config.json" "${FIXTURE}" "${NAMES}" \
            "${COLD70}" "${PLAN_CLASSES}" "${CUSTOM_UVM}"; do
  require_file "${path}"
done
[[ ! -e "${RUN_ROOT}" ]] || die "run directory already exists: ${RUN_ROOT}"
bash -n "${BASH_SOURCE[0]}"
"${VLLM_PY}" -m py_compile "${RUNNER}"

runtime_preflight
RUN_ROOT_ACTIVE="${RUN_ROOT}"
mkdir -p "${RUN_ROOT_ACTIVE}/sequence"
RUN_ROOT_ACTIVE="$(cd "${RUN_ROOT_ACTIVE}" && pwd)"
CASE_ROOT="${RUN_ROOT_ACTIVE}/sequence"
MANIFEST="${CASE_ROOT}/cpu_offload_manifest.jsonl"
: >"${MANIFEST}"

module_snapshot "${RUN_ROOT_ACTIVE}/module.before.json" "${STOCK_UVM}"
gpu_state_json "${CASE_ROOT}/gpu.before.json"
load_custom_module
module_snapshot "${RUN_ROOT_ACTIVE}/module.custom.json" "${CUSTOM_UVM}"
cp -- "${STATS_PATH}" "${RUN_ROOT_ACTIVE}/latency_stats.before.txt"

kernel_epoch="$(date +%s)"
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
    VLLM_CPU_OFFLOAD_PLAN_CLASS_PRECOMPILE=1 \
    VLLM_CPU_OFFLOAD_PLAN_CLASS_QUIESCED_DELTA=1 \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288 \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=4k \
    VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1 \
    VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW=1 \
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
    VLLM_CPU_OFFLOAD_MANIFEST_FILE="${MANIFEST}" \
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 \
    VLLM_PHASE_SWITCH_CONTROLLED_HARD_EXIT_AFTER_CLEANUP=1 \
    VLLM_UVM_LOG_FILE="${CASE_ROOT}/uvm.log" \
    VLLM_USE_UVM=0 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=14 \
    MKL_NUM_THREADS=14 \
    CUDA_VISIBLE_DEVICES=0 \
    /usr/bin/time -v -o "${CASE_ROOT}/resource_time.txt" \
    timeout --kill-after=30s "${TIMEOUT_SEC}s" \
    taskset -c "${CPUSET}" \
    "${VLLM_PY}" "${RUNNER}" \
      --model "${MODEL}" \
      --fixture "${FIXTURE}" \
      --exact-names "${NAMES}" \
      --expert-plan "${COLD70}" \
      --offload-manifest "${MANIFEST}" \
      --output "${CASE_ROOT}/result.json" \
      --arm D \
      --batch-size 1 \
      --cudagraph-capture-sizes 1,314 \
      --warmup-decode-steps 5 \
      --measured-decode-steps 16 \
      --park-resume-roundtrip \
      --kv-cache-memory-bytes "${KV_BYTES}" \
      --cpu-offload-gb 30.0 \
      --max-model-len 512 \
      --max-num-seqs 1 \
      --max-num-batched-tokens 512 \
      --gpu-memory-utilization 0.97 \
      --seed 0 \
      --expected-parameter-count 96 \
      --expected-full-parameter-bytes 28991029248 \
      --expected-selected-bytes 15854469120
) >"${CASE_ROOT}/run.log" 2>&1
runner_rc=$?
sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
  >"${CASE_ROOT}/kernel.since.log" 2>"${CASE_ROOT}/kernel.since.err"
journal_rc=$?
set -e
printf '%s\n' "${runner_rc}" >"${CASE_ROOT}/returncode.txt"
printf '%s\n' "${journal_rc}" >"${CASE_ROOT}/kernel_log_returncode.txt"
cp -- "${STATS_PATH}" "${RUN_ROOT_ACTIVE}/latency_stats.after.txt"

wait_gpu_clean || die "GPU/client state did not cleanly shut down"
gpu_state_json "${CASE_ROOT}/gpu.after.json"
restore_stock || die "failed to restore stock UVM"
module_snapshot "${RUN_ROOT_ACTIVE}/module.after.json" "${STOCK_UVM}"
restart_exporters

(( journal_rc == 0 )) || die "kernel log collection failed"
if (( runner_rc != 0 )); then
  tail -n 240 "${CASE_ROOT}/run.log" >&2 || true
  die "single-tier full-dual-view runner failed"
fi

"${VLLM_PY}" - "${CASE_ROOT}/result.json" "${CASE_ROOT}/kernel.since.log" \
  "${RUN_ROOT_ACTIVE}/verification.json" <<'PY'
import json
import pathlib
import re
import sys

result_path, kernel_path, output_path = map(pathlib.Path, sys.argv[1:])
result = json.loads(result_path.read_text(encoding="utf-8"))
kernel = kernel_path.read_text(encoding="utf-8", errors="replace")
transitions = result["diagnostics"]["final"]["model"]["phase_transition_history"]
remote = [
    x for x in transitions
    if x.get("event") == "cpu_offload_uvm_original_va_remote_transition"
    and x.get("reason") == "first_decode_cutover"
]
roundtrip = (result.get("steps") or {}).get("park_resume_roundtrip") or {}
roundtrip_remote = [
    x for x in roundtrip.get("repark_transitions", [])
    if x.get("event") == "cpu_offload_uvm_original_va_remote_transition"
]
roundtrip_resume = [
    x for x in roundtrip.get("resume_transitions", [])
    if x.get("event") == "cpu_offload_uvm_original_va_alias_prefill_activated"
]
all_remote = [*remote, *roundtrip_remote]
parks = [x["park"] for x in all_remote if isinstance(x.get("park"), dict)]
resumes = [x["resume"] for x in roundtrip_resume if isinstance(x.get("resume"), dict)]
errors = []
if result.get("status") != "ok": errors.append("runner status is not ok")
if len(parks) != 2 or len(resumes) != 1: errors.append("expected PARK/PARK/RESUME sequence is absent")
for label, records in (("park", parks), ("resume", resumes)):
    for item in records:
        if item.get("ok") is not True: errors.append(f"{label} failed")
        if item.get("dual_view_full_coverage") is not True: errors.append(f"{label} lacks full coverage")
        if int(item.get("dual_view_block_count", 0)) <= 0: errors.append(f"{label} block count is zero")
for item in all_remote:
    if item.get("remote_map_operation") != "prebuilt_full_dual_view":
        errors.append("Decode performed the legacy range-map operation")
    if int((item.get("remote_map") or {}).get("ioctl_calls", -1)) != 0:
        errors.append("Decode remote-map ioctl count is nonzero")
if re.search(
    r"NVRM:\s+Xid|MMU fault|\bBUG:|NULL pointer|general protection fault",
    kernel,
    re.I,
):
    errors.append("kernel/GPU fault found")
payload = {
    "schema": "e10_single_tier_full_dual_view_qualification_v1",
    "status": "ok" if not errors else "failed",
    "errors": errors,
    "park": parks,
    "resume": resumes,
    "roundtrip": roundtrip,
    "output_token_ids_sha256": result.get("output_token_ids_sha256"),
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if errors:
    raise SystemExit("; ".join(errors))
PY

echo "result=${CASE_ROOT}/result.json"
echo "verification=${RUN_ROOT_ACTIVE}/verification.json"
