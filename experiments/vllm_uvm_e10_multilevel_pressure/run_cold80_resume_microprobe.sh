#!/usr/bin/env bash
set -euo pipefail

# One-shot microprobe for the cold80 Decode->Prefill RESUME path.  The dual
# parent-table optimization is deliberately disabled so that all 34,560
# consumer aliases are installed through the original grouped PTE path.

PROBE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PROBE_DIR}/../.." && pwd)"
BASE_RUNNER="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
export ACCESS_COUNTER_MIGRATION_ENABLE=0
export UVM_PREFETCH_ENABLE=0
export UVM_THRASHING_ENABLE=0
export STATIC_ALIAS_DUAL_VIEW_ENABLE=0
# shellcheck source=/dev/null
source "${BASE_RUNNER}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
FIXTURE="${FIXTURE:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json}"
NAMES="${REPO_ROOT}/experiments/vllm_uvm_large_kv_multilevel/inputs/exact_parameter_names.txt"
COLD80="${PROBE_DIR}/inputs/cold80.json"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/artifacts/e10-cold80-resume-microprobe-20260830-attempt-01-qualification}"
KV_BYTES=2145386496
EXPECTED_SELECTED_BYTES=18119393280
EXPECTED_FULL_BYTES=28991029248
EXPECTED_ALIASES=34560
TIMEOUT_SEC="${TIMEOUT_SEC:-7200}"

for path in "${MODEL}/config.json" "${FIXTURE}" "${NAMES}" "${COLD80}"; do
  require_file "${path}"
done
[[ "$(wc -l <"${NAMES}")" -eq 96 ]] || die "expected 96 exact parameter names"
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"

"${VLLM_PY}" - "${COLD80}" "${EXPECTED_SELECTED_BYTES}" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
assert set(map(int, plan["selected_by_layer"])) == set(range(48))
assert {len(v) for v in plan["selected_by_layer"].values()} == {80}
assert int(plan["metadata"]["selected_main_bytes"]) == int(sys.argv[2])
assert 48 * 80 * 9 == 34_560
PY

runtime_preflight
mkdir -p "${RUN_ROOT}/sequence"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
RUN_ROOT_ACTIVE="${RUN_ROOT}"
CASE_ROOT="${RUN_ROOT}/sequence"
MANIFEST="${CASE_ROOT}/cpu_offload_manifest.jsonl"
: >"${MANIFEST}"

"${VLLM_PY}" - "${RUN_ROOT}/protocol.json" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "schema": "e10_cold80_resume_microprobe_v1",
    "qualification_only": True,
    "complete_sequences": 1,
    "statistical_repetitions": 0,
    "expert_plan": "cold80",
    "consumer_alias_count": 34560,
    "operation": "Decode remote_parked -> Prefill sealed RESUME",
    "dual_parent_switch_enabled": False,
    "pte_mode_requested": "auto",
    "reference_resume_ioctl_ms": 471.784742,
    "reference_pte_map_ms": 459.223813,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

git -C "${REPO_ROOT}" status --short >"${RUN_ROOT}/git_status.txt"
nvidia-smi -i 0 --query-gpu=name,memory.total,driver_version \
  --format=csv,noheader,nounits >"${RUN_ROOT}/hardware.csv"
module_snapshot "${RUN_ROOT}/module.before.json" "${STOCK_UVM}"
load_custom_module
module_snapshot "${RUN_ROOT}/module.custom.json" "${CUSTOM_UVM}"
[[ "$(tr -d '[:space:]' </sys/module/nvidia_uvm/parameters/uvm_moe_static_alias_dual_view_enable)" == 0 ]] \
  || die "dual-view fast path was not disabled"
wait_gpu_clean || die "GPU is not clean before the microprobe"
gpu_state_json "${CASE_ROOT}/gpu.before.json"
cp -- "${STATS_PATH}" "${RUN_ROOT}/latency_stats.before.txt"

sha256sum \
  "${MODEL}/config.json" "${FIXTURE}" "${NAMES}" "${COLD80}" \
  "${RUNNER}" "${BASH_SOURCE[0]}" "${CUSTOM_UVM}" "${STOCK_UVM}" \
  >"${RUN_ROOT}/input_sha256.txt"

kernel_epoch="$(date +%s)"
set +e
(
  cd "${VLLM_ROOT}"
  env \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_ATTENTION_BACKEND=TRITON_ATTN \
    VLLM_CPU_OFFLOAD_BACKEND=copy \
    VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${NAMES}" \
    VLLM_CPU_OFFLOAD_STATIC_BANK=0 \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1 \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="${COLD80}" \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288 \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=auto \
    VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1 \
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
    VLLM_UVM_ORIGINAL_VA_PHASE_MODE=prefill_graph_remote_decode \
    VLLM_UVM_ORIGINAL_VA_PHASE_RESULT_FILE="${CASE_ROOT}/phase_transition_result.json" \
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
    taskset -c 0-13 \
    "${VLLM_PY}" "${RUNNER}" \
      --model "${MODEL}" \
      --fixture "${FIXTURE}" \
      --exact-names "${NAMES}" \
      --expert-plan "${COLD80}" \
      --offload-manifest "${MANIFEST}" \
      --output "${CASE_ROOT}/result.json" \
      --arm D \
      --batch-size 1 \
      --cudagraph-capture-sizes 1,314 \
      --warmup-decode-steps 1 \
      --measured-decode-steps 1 \
      --park-resume-roundtrip \
      --kv-cache-memory-bytes "${KV_BYTES}" \
      --cpu-offload-gb 30.0 \
      --max-model-len 512 \
      --max-num-seqs 1 \
      --max-num-batched-tokens 512 \
      --gpu-memory-utilization 0.97 \
      --seed 0 \
      --expected-parameter-count 96 \
      --expected-full-parameter-bytes "${EXPECTED_FULL_BYTES}" \
      --expected-selected-bytes "${EXPECTED_SELECTED_BYTES}"
) >"${CASE_ROOT}/run.log" 2>&1
runner_rc=$?
sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
  >"${RUN_ROOT}/kernel.since.log" 2>"${RUN_ROOT}/kernel.since.err"
journal_rc=$?
set -e
printf '%s\n' "${runner_rc}" >"${CASE_ROOT}/returncode.txt"
printf '%s\n' "${journal_rc}" >"${RUN_ROOT}/kernel_log_returncode.txt"
cp -- "${STATS_PATH}" "${RUN_ROOT}/latency_stats.after.txt"
wait_gpu_clean || die "GPU/client state did not clean after the microprobe"
gpu_state_json "${CASE_ROOT}/gpu.after.json"

restore_stock || die "failed to restore stock nvidia_uvm"
module_snapshot "${RUN_ROOT}/module.after.json" "${STOCK_UVM}"
restart_exporters

(( journal_rc == 0 )) || die "kernel journal collection failed"
if (( runner_rc != 0 )); then
  tail -n 300 "${CASE_ROOT}/run.log" >&2 || true
  die "the only permitted GPU microprobe failed"
fi

"${VLLM_PY}" - \
  "${CASE_ROOT}/result.json" "${MANIFEST}" \
  "${RUN_ROOT}/module.custom.json" "${RUN_ROOT}/kernel.since.log" \
  "${RUN_ROOT}/summary.json" <<'PY'
import json, pathlib, re, sys

result_path, manifest_path, module_path, kernel_path, output_path = map(pathlib.Path, sys.argv[1:])
result = json.loads(result_path.read_text(encoding="utf-8"))
module = json.loads(module_path.read_text(encoding="utf-8"))
assert result["status"] == "ok"
assert result["config"]["expert_plan"].endswith("/cold80.json")
assert result["config"]["park_resume_roundtrip"] is True
assert module["static_alias_du_view_enable"] == 0 if "static_alias_du_view_enable" in module else module["static_alias_dual_view_enable"] == 0

roundtrip = result["steps"]["park_resume_roundtrip"]
assert roundtrip is not None
events = [
    item for item in roundtrip["resume_transitions"]
    if item.get("event") == "cpu_offload_uvm_original_va_alias_prefill_activated"
]
assert len(events) == 1
event = events[0]
resume = event["resume"]
assert event["ok"] is True and event["pre_state"] == "remote_parked" and event["post_state"] == "sealed"
assert resume["ok"] is True and resume["rm_status"] == 0
assert resume["consumer_alias_count"] == resume["processed_alias_count"] == 34_560
assert resume["tlb_invalidate_count"] == 34_560
assert resume["transition_batch_count"] == 8

finalize = None
for line in manifest_path.read_text(encoding="utf-8").splitlines():
    record = json.loads(line)
    if record.get("event") == "cpu_offload_uvm_static_alias_finalized" and record.get("ok") is True:
        finalize = record
assert finalize is not None and finalize["pte_mode"] == "4K"

kernel_text = kernel_path.read_text(encoding="utf-8", errors="replace")
bad_kernel_lines = [
    line for line in kernel_text.splitlines()
    if re.search(r"NVRM: Xid|BUG:|Oops:|general protection fault|uvm_gpu_fault_entry", line, re.I)
]
assert not bad_kernel_lines, bad_kernel_lines

fields = (
    "validation_ns", "build_sort_ns", "preflight_ns", "metadata_ns",
    "cpu_unmap_ns", "pte_map_ns", "wait_ns", "bookkeeping_ns",
)
breakdown_ms = {name.removesuffix("_ns") + "_ms": float(resume.get(name, 0)) / 1e6 for name in fields}
reference_pte_ms = 459.223813
actual_pte_ms = breakdown_ms["pte_map_ms"]
summary = {
    "schema": "e10_cold80_resume_microprobe_summary_v1",
    "status": "ok",
    "single_run": True,
    "statistical_repetitions": 0,
    "dual_parent_switch_enabled": False,
    "pte_mode": finalize["pte_mode"],
    "consumer_alias_count": resume["consumer_alias_count"],
    "processed_alias_count": resume["processed_alias_count"],
    "transition_batch_count": resume["transition_batch_count"],
    "tlb_invalidate_count": resume["tlb_invalidate_count"],
    "event_wall_ms": float(event["wall_ns"]) / 1e6,
    "resume_wall_ms": float(resume["wall_ns"]) / 1e6,
    "ioctl_wall_ms": float(resume["ioctl_wall_ns"]) / 1e6,
    "driver_breakdown_ms": breakdown_ms,
    "driver_breakdown_sum_ms": sum(breakdown_ms.values()),
    "reference_pte_map_ms": reference_pte_ms,
    "pte_map_delta_ms": actual_pte_ms - reference_pte_ms,
    "pte_map_delta_percent": (actual_pte_ms / reference_pte_ms - 1.0) * 100.0,
    "roundtrip_prefill_step_ms": float(roundtrip["prefill"]["elapsed_ms"]),
    "graph_recaptures": (
        int(roundtrip["compilation_counter_after"]["num_cudagraph_captured"])
        - int(roundtrip["compilation_counter_before"]["num_cudagraph_captured"])
    ),
    "kernel_faults": 0,
    "output_prefix_verified_by_runner": True,
}
assert summary["graph_recaptures"] == 0
output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "RUN_ROOT=${RUN_ROOT}"
echo "summary=${RUN_ROOT}/summary.json"
