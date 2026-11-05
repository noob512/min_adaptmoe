#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
BENCHMARK="${SCRIPT_DIR}/benchmark_custom_offload.py"
PLAN_CONVERTER="${SCRIPT_DIR}/build_custom_plan_from_kt_stats.py"
ABBA_SUMMARIZER="${SCRIPT_DIR}/summarize_custom_abba.py"
UVM_MODULE="${UVM_MODULE:-${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
CUSTOM_IMPLEMENTATION_FILES=(
  workloads/vllm/vllm/vllm/device_allocator/uvm.py
  workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py
  workloads/vllm/vllm/vllm/device_allocator/uvm_region.py
  workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py
  workloads/vllm/vllm/vllm/model_executor/models/qwen3_moe.py
  workloads/vllm/vllm/vllm/model_executor/models/utils.py
  workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py
  workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py
)

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

usage() {
  sed -n '/^usage_text() {$/,/^}$/p' "${BASH_SOURCE[0]}" \
    | sed '1d;$d;s/^  //'
}

usage_text() {
  cat <<'EOF'
Usage:
  run_custom_comparison.sh preflight
  run_custom_comparison.sh matrix
  PLAN=/path/plan.json MODEL_FAMILY=qwen PLAN_KIND=nonempty \
    run_custom_comparison.sh validate-plan
  run_custom_comparison.sh run-case
  run_custom_comparison.sh run-abba

run-case is configured through environment variables:
  ALLOW_GPU_RUN=1                         mandatory safety acknowledgement
  PROFILE=all_gpu|custom_empty|custom_remote|custom_restore|oracle_sync|oracle_async
  MODEL_FAMILY=qwen|deepseek
  MODEL=/absolute/model/snapshot
  EVAL_FIXTURE=/path/evaluation_fixture.json
  WARMUP_FIXTURE=/path/calibration_fixture.json  (unless warmup is zero)
  PLAN=/path/plan.json                    required for remote/restore/oracle
  EMPTY_PLAN=/path/empty.json             used by run-abba arm A
  RUN_ROOT=/new/case/directory
  CONCURRENCY=8 OUTPUT_TOKENS=64
  KV_CACHE_BYTES=1288490189

Optional:
  WARMUP_REQUESTS=8 WARMUP_OUTPUT_TOKENS=8 MAX_MODEL_LEN=544
  MAX_NUM_BATCHED_TOKENS=2048 MAX_NUM_SEQS=32 GPU_MEMORY_UTILIZATION=0.97
  TIMEOUT_SEC=3600 ROUTING_TRACE=0 GPU_CLEAN_MIB=128
  FORMAL_RUN=0                            set to 1 for formal cross-framework runs
  REQUIRE_PROMPT_TOKEN_IDS=0              forced to 1 when FORMAL_RUN=1
  CPUSET=0-13 CPU_THREADS=14              common CPU/affinity budget
  CONCURRENCIES=1,4,8 ABBA_BLOCKS=3       run-abba only
  ALLOW_ORACLE_ABLATION=1                 mandatory for oracle profiles
EOF
}

static_preflight() {
  require_dir "${VLLM_ROOT}"
  require_file "${VLLM_PY}"
  require_file "${BENCHMARK}"
  require_file "${PLAN_CONVERTER}"
  require_file "${ABBA_SUMMARIZER}"
  require_file "${SCRIPT_DIR}/CUSTOM_COMPARISON_PLAN.md"
  require_file "${VLLM_ROOT}/vllm/model_executor/layers/fused_moe/layer.py"
  require_file "${VLLM_ROOT}/vllm/model_executor/models/qwen3_moe.py"
  require_file "${VLLM_ROOT}/vllm/v1/worker/gpu_model_runner.py"
  command -v rg >/dev/null || die "rg is unavailable"
  command -v sha256sum >/dev/null || die "sha256sum is unavailable"
  command -v timeout >/dev/null || die "timeout is unavailable"
  command -v taskset >/dev/null || die "taskset is unavailable"

  rg -q 'VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE' \
    "${VLLM_ROOT}/vllm/model_executor/layers/fused_moe/layer.py" \
    || die "custom remote-read source hook is missing"
  rg -q 'prepare_uvm_prefill_oracle_experts' \
    "${VLLM_ROOT}/vllm/model_executor/layers/fused_moe/layer.py" \
    || die "oracle prepare source hook is missing"
  rg -q '_Qwen3MoePrefillOraclePrefetch' \
    "${VLLM_ROOT}/vllm/model_executor/models/qwen3_moe.py" \
    || die "Qwen oracle controller is missing"
  rg -q 'run_uvm_init_time_weight_placement' \
    "${VLLM_ROOT}/vllm/v1/worker/gpu_model_runner.py" \
    || die "post-warmup placement hook is missing"
  rg -q '_write_uvm_init_weight_placement_manifest' \
    "${VLLM_ROOT}/vllm/v1/worker/gpu_model_runner.py" \
    || die "runtime placement manifest hook is missing"

  bash -n "${BASH_SOURCE[0]}"
  "${VLLM_PY}" - "${BENCHMARK}" "${PLAN_CONVERTER}" "${ABBA_SUMMARIZER}" <<'PY'
import ast
import sys
from pathlib import Path

for value in sys.argv[1:]:
    path = Path(value)
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("python syntax: OK")
PY
  "${VLLM_PY}" "${BENCHMARK}" --help >/dev/null
  "${VLLM_PY}" "${PLAN_CONVERTER}" --help >/dev/null
  "${VLLM_PY}" "${ABBA_SUMMARIZER}" --help >/dev/null

  local repo_commit diff_sha loaded_srcversion module_srcversion
  repo_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  diff_sha="$(git -C "${REPO_ROOT}" diff --binary HEAD -- \
    "${CUSTOM_IMPLEMENTATION_FILES[@]}" \
    | sha256sum | awk '{print $1}')"
  echo "repo_commit=${repo_commit}"
  echo "custom_patch_sha256=${diff_sha}"
  if ! git -C "${REPO_ROOT}" diff --quiet HEAD -- \
    "${CUSTOM_IMPLEMENTATION_FILES[@]}"; then
    echo "warning: custom implementation is dirty; commit plus patch SHA are required"
  fi

  if [[ -f "${UVM_MODULE}" ]] && command -v modinfo >/dev/null; then
    module_srcversion="$(modinfo -F srcversion "${UVM_MODULE}" 2>/dev/null || true)"
    loaded_srcversion="$(sed -n '1p' /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)"
    echo "uvm_module_srcversion=${module_srcversion:-missing}"
    echo "uvm_loaded_srcversion=${loaded_srcversion:-missing}"
    [[ -n "${loaded_srcversion}" && -n "${module_srcversion}" ]] \
      || die "loaded/worktree UVM srcversion is unavailable"
    if [[ "${loaded_srcversion}" != "${module_srcversion}" ]]; then
      die "loaded/worktree UVM srcversion mismatch"
    fi
  fi
  echo "custom comparison static preflight: OK (no CUDA workload launched)"
}

print_matrix() {
  cat <<'EOF'
stage	model_pair	tier	profile	prompt/output	concurrency	process_semantics
qualification	Qwen self	E	all_gpu,custom_empty,custom_remote	300-320/64	1,8	3 cold reps each
qualification	Qwen oracle	ablation	remote,oracle_sync,oracle_async	1217 aggregate/1	4	no app warmup; never main table
qualification	DeepSeek self	E	all_gpu,custom_empty,custom_remote	512/32	1	3 cold reps each
core	Qwen custom-vs-KT	E	coordinate-exact frozen KT mask	512/32;ShareGPT/64	1,2,4,8,16,32	3 ABBA blocks per point
core	Qwen custom-vs-KT	V	equal ready-VRAM	ShareGPT/64	1,4,8,16,32	3 ABBA blocks per point
core	Qwen custom-vs-KT	S	best-exact,best-quality-bounded	ShareGPT/64	1,4,8,16,32	separate quality table
core	DeepSeek custom-vs-MoEI	E	batch1 exact	512/32	1	paper-protocol anchor, 3 ABBA blocks
core	DeepSeek custom-vs-MoEI	V	equal ready-VRAM	ShareGPT/64	1,4,8,16	3 ABBA blocks per point
core	DeepSeek custom-vs-MoEI	S	best-exact,best-quality-bounded	ShareGPT/64	1,4,8,16	separate quality table
extended	model-local	E,V,S	budget sweep	128,512,2048 / 32,128	1,8	independent cold process per point
extended	model-local	E,V,S	distribution shift	ShareGPT,code,long-text	1,8	frozen calibration plan
extended	custom oracle only	ablation	envelope/lookahead/reuse sweep	variable	1,8	bounded-cache=false label
EOF
}

validate_plan() {
  local plan="$1"
  local family="$2"
  local expected_kind="${3:-any}"
  require_file "${plan}"
  "${VLLM_PY}" - "${plan}" "${family}" "${expected_kind}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
family = sys.argv[2]
expected_kind = sys.argv[3]
data = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(data, dict):
    raise SystemExit("plan is not an object")
cold = data.get("offload_plan")
hot = data.get("hot_plan", data.get("prefetch_plan"))
if not isinstance(cold, list) or not isinstance(hot, list):
    raise SystemExit("plan must contain offload_plan and hot_plan/prefetch_plan")

if family == "qwen":
    valid_layers = set(range(48))
    experts = 128
elif family == "deepseek":
    valid_layers = set(range(1, 27))
    experts = 64
else:
    raise SystemExit(f"unknown family {family!r}")

def coordinates(records, label):
    values = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SystemExit(f"{label}[{index}] is not an object")
        layer = record.get("layer_id")
        expert = record.get("expert_id")
        if isinstance(layer, bool) or not isinstance(layer, int) or layer not in valid_layers:
            raise SystemExit(f"{label}[{index}] invalid layer_id={layer!r}")
        if isinstance(expert, bool) or not isinstance(expert, int) or not 0 <= expert < experts:
            raise SystemExit(f"{label}[{index}] invalid expert_id={expert!r}")
        values.append((layer, expert))
    if len(values) != len(set(values)):
        raise SystemExit(f"{label} contains duplicate coordinates")
    return set(values)

cold_keys = coordinates(cold, "offload_plan")
hot_keys = coordinates(hot, "hot_plan")
if cold_keys & hot_keys:
    raise SystemExit("cold and hot coordinates overlap")
expected = {(layer, expert) for layer in valid_layers for expert in range(experts)}
if cold_keys | hot_keys != expected:
    missing = sorted(expected - (cold_keys | hot_keys))
    extra = sorted((cold_keys | hot_keys) - expected)
    raise SystemExit(f"plan does not cover topology: missing={missing[:8]} extra={extra[:8]}")
if data.get("offload_plan_records", len(cold)) != len(cold):
    raise SystemExit("offload_plan_records does not match list length")
if expected_kind == "empty" and cold:
    raise SystemExit(f"empty-plan profile received {len(cold)} cold experts")
if expected_kind == "nonempty" and not cold:
    raise SystemExit("offload profile received an empty cold plan")
if expected_kind not in {"any", "empty", "nonempty"}:
    raise SystemExit(f"unknown expected plan kind: {expected_kind}")
print(f"plan validation: family={family} cold={len(cold)} hot={len(hot)}")
PY
}

wait_for_clean_gpu() {
  local threshold="$1"
  local used_mib=""
  local compute_processes=""
  local _
  for _ in $(seq 1 180); do
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | head -n 1 | tr -d ' ')"
    compute_processes="$(nvidia-smi --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] \
      && (( used_mib <= threshold )) \
      && [[ -z "${compute_processes//[[:space:]]/}" ]]; then
      return 0
    fi
    sleep 1
  done
  nvidia-smi >&2 || true
  die "GPU did not become clean (used=${used_mib:-unknown}, pids=${compute_processes:-none})"
}

write_metadata() {
  local case_root="$1"
  local model="$2"
  local eval_fixture="$3"
  local warmup_fixture="$4"
  local plan="$5"
  {
    echo "repo_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "model=${model}"
    echo "profile=${PROFILE}"
    echo "model_family=${MODEL_FAMILY}"
  } > "${case_root}/metadata.env"
  git -C "${REPO_ROOT}" status --short > "${case_root}/git_status.txt"
  git -C "${REPO_ROOT}" diff --binary HEAD -- \
    "${CUSTOM_IMPLEMENTATION_FILES[@]}" \
    > "${case_root}/custom_implementation.patch"
  sha256sum "${case_root}/custom_implementation.patch" \
    > "${case_root}/custom_implementation.patch.sha256"
  sha256sum "${eval_fixture}" > "${case_root}/evaluation_fixture.sha256"
  if [[ -n "${warmup_fixture}" ]]; then
    sha256sum "${warmup_fixture}" > "${case_root}/warmup_fixture.sha256"
  fi
  if [[ -n "${plan}" ]]; then
    sha256sum "${plan}" > "${case_root}/plan.sha256"
  fi
  if [[ -f "${model}/config.json" ]]; then
    sha256sum "${model}/config.json" > "${case_root}/model_config.sha256"
  fi
  uname -a > "${case_root}/uname.txt"
  lscpu > "${case_root}/lscpu.txt"
  if "${VLLM_PY}" -m pip --version >/dev/null 2>&1; then
    "${VLLM_PY}" -m pip freeze > "${case_root}/pip_freeze.txt"
  elif command -v uv >/dev/null 2>&1; then
    uv pip freeze --python "${VLLM_PY}" > "${case_root}/pip_freeze.txt"
  else
    die "cannot freeze Python environment: neither pip nor uv is available"
  fi
  if [[ -f "${UVM_MODULE}" ]] && command -v modinfo >/dev/null; then
    modinfo "${UVM_MODULE}" > "${case_root}/uvm_module_modinfo.txt" 2>&1 || true
    if [[ -r /sys/module/nvidia_uvm/srcversion ]]; then
      cp /sys/module/nvidia_uvm/srcversion "${case_root}/uvm_loaded_srcversion.txt"
    fi
  fi
}

verify_case_result() {
  local case_root="$1"
  local expected_concurrency="$2"
  local expected_output_tokens="$3"
  local profile="$4"
  local plan="$5"
  local family="$6"
  local routing_trace_enabled="$7"
  local formal_run="$8"
  "${VLLM_PY}" - "${case_root}" "${expected_concurrency}" \
    "${expected_output_tokens}" "${profile}" "${plan}" "${family}" \
    "${routing_trace_enabled}" "${formal_run}" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_concurrency = int(sys.argv[2])
expected_output = int(sys.argv[3])
profile = sys.argv[4]
plan_path = Path(sys.argv[5]) if sys.argv[5] else None
family = sys.argv[6]
routing_trace_enabled = sys.argv[7] == "1"
formal_run = sys.argv[8] == "1"
result = json.loads((root / "result.json").read_text(encoding="utf-8"))
if result.get("status") != "ok":
    raise SystemExit("benchmark status is not ok")
measurement = result.get("measurement", {})
if measurement.get("completed_requests") != expected_concurrency:
    raise SystemExit("completed request count mismatch")
if measurement.get("output_tokens") != expected_concurrency * expected_output:
    raise SystemExit("output token count mismatch")
requests = measurement.get("requests", [])
if len(requests) != expected_concurrency:
    raise SystemExit("request record count mismatch")
if any(item.get("prompt_tokens") != item.get("expected_prompt_tokens") for item in requests):
    raise SystemExit("prompt token count gate failed")
if any(item.get("prompt_token_ids_exact") is False for item in requests):
    raise SystemExit("prompt token ID gate failed")
if formal_run and any(item.get("prompt_token_ids_exact") is not True for item in requests):
    raise SystemExit("formal run lacks exact frozen prompt token IDs")
if any(item.get("output_tokens") != expected_output for item in requests):
    raise SystemExit("per-request output token gate failed")

expected_tpot_count = expected_concurrency if expected_output > 1 else 0
metric_counts = {
    "ttft_ms": expected_concurrency,
    "request_latency_ms": expected_concurrency,
    "tpot_ms": expected_tpot_count,
}
metrics_complete = all(
    isinstance(measurement.get(name), dict)
    and measurement[name].get("count") == expected_count
    for name, expected_count in metric_counts.items()
)
required_request_metrics = ["ttft_ms", "request_latency_ms"]
if expected_output > 1:
    required_request_metrics.append("tpot_ms")
metric_values_finite_nonnegative = all(
    isinstance(item.get(name), (int, float))
    and not isinstance(item.get(name), bool)
    and math.isfinite(float(item[name]))
    and float(item[name]) >= 0.0
    for item in requests
    for name in required_request_metrics
)
elapsed = measurement.get("elapsed_seconds")
throughput = measurement.get("output_tokens_per_second")
aggregate_metrics_finite_positive = all(
    isinstance(value, (int, float))
    and not isinstance(value, bool)
    and math.isfinite(float(value))
    and float(value) > 0.0
    for value in (elapsed, throughput)
)
if formal_run and not (
    metrics_complete
    and metric_values_finite_nonnegative
    and aggregate_metrics_finite_positive
):
    raise SystemExit(
        f"formal metric validity gate failed: expected={metric_counts}"
    )

prompt_ids_frozen = all(
    item.get("prompt_token_ids_exact") is True for item in requests
)
formal_metrics_valid = (
    metrics_complete
    and metric_values_finite_nonnegative
    and aggregate_metrics_finite_positive
)
all_gpu_eligible = formal_run and prompt_ids_frozen and formal_metrics_valid

gate = {
    "status": "pass",
    "profile": profile,
    "request_count_exact": True,
    "prompt_token_count_exact": True,
    "formal_run": formal_run,
    "prompt_token_ids_frozen": prompt_ids_frozen,
    "output_token_count_exact": True,
    "metric_counts_expected": metric_counts,
    "metric_counts_complete": metrics_complete,
    "metric_values_finite_nonnegative": metric_values_finite_nonnegative,
    "aggregate_metrics_finite_positive": aggregate_metrics_finite_positive,
    "aggregate_output_token_ids_sha256": measurement.get(
        "aggregate_output_token_ids_sha256"
    ),
    "runtime_remote_placement_manifest_available": False,
    "publication_eligible": profile == "all_gpu" and all_gpu_eligible,
}
if profile == "all_gpu" and not all_gpu_eligible:
    gate["publication_blocker"] = (
        "all-GPU reference is screening-only unless FORMAL_RUN=1, prompt IDs "
        "are exact, and latency metrics are complete"
    )
if profile.startswith("oracle_"):
    trace_path = root / "oracle_trace.jsonl"
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if trace_path.exists() else []
    matching = [item for item in records if item.get("event") == "qwen3_moe_prefill_oracle_prefetch"]
    if len(matching) != 1:
        raise SystemExit(f"oracle trace count is {len(matching)}, expected one")
    trace = matching[0]
    if trace.get("layer_count") != 48:
        raise SystemExit("oracle did not prepare/commit all 48 layers")
    layers = trace.get("layers", [])
    if not isinstance(layers, list) or len(layers) != 48:
        raise SystemExit("oracle per-layer trace is incomplete")
    expected_mode = profile.removeprefix("oracle_")
    if trace.get("mode") != expected_mode:
        raise SystemExit("oracle trace mode/profile mismatch")
    layer_ids = [item.get("target_layer_id") for item in layers]
    if len(set(layer_ids)) != 48 or set(layer_ids) != set(range(48)):
        raise SystemExit("oracle trace does not contain 48 unique Qwen layer IDs")
    if plan_path is None:
        raise SystemExit("oracle verification requires the frozen plan")
    plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    planned_by_layer = {layer: set() for layer in range(48)}
    for item in plan_data.get("offload_plan", []):
        planned_by_layer[int(item["layer_id"])].add(int(item["expert_id"]))
    expert_main_bytes = plan_data.get("metadata", {}).get("expert_main_bytes")
    if not isinstance(expert_main_bytes, int) or expert_main_bytes <= 0:
        raise SystemExit("oracle plan lacks positive expert_main_bytes provenance")
    requested_sum = 0
    expert_sum = 0
    for item in layers:
        layer_id = int(item["target_layer_id"])
        experts = item.get("expert_ids")
        if not isinstance(experts, list) or set(experts) != planned_by_layer[layer_id]:
            raise SystemExit(f"oracle expert plan mismatch at layer {layer_id}")
        if item.get("expert_count") != len(experts):
            raise SystemExit(f"oracle expert count mismatch at layer {layer_id}")
        requested = item.get("requested_bytes")
        if requested != len(experts) * expert_main_bytes:
            raise SystemExit(f"oracle requested-byte mismatch at layer {layer_id}")
        if item.get("initial_access_mode") != "remote_read":
            raise SystemExit(f"oracle initial access mode mismatch at layer {layer_id}")
        expected_dispatch = (
            "foreground"
            if expected_mode == "sync" or layer_id == 0
            else "async_worker"
        )
        if item.get("dispatch_kind") != expected_dispatch:
            raise SystemExit(f"oracle dispatch mismatch at layer {layer_id}")
        requested_sum += int(requested)
        expert_sum += len(experts)
    planned_main_bytes = plan_data.get("metadata", {}).get(
        "actual_offload_main_bytes"
    )
    if requested_sum != trace.get("requested_bytes") or requested_sum != planned_main_bytes:
        raise SystemExit("oracle aggregate requested bytes do not match plan/trace")
    if expert_sum != trace.get("expert_count"):
        raise SystemExit("oracle aggregate expert count mismatch")
    if trace.get("num_tokens") != measurement.get("prompt_tokens"):
        raise SystemExit("oracle trace token count does not match measured prompts")
    gate.update({
        "oracle_trace_complete": True,
        "oracle_mode_matches_profile": True,
        "oracle_unique_layer_ids": True,
        "oracle_prepare_commit_success_by_completed_trace": True,
        "oracle_expert_coordinates_match_plan": True,
        "oracle_requested_bytes_match_plan_and_trace": True,
        "oracle_requested_bytes": requested_sum,
        "same_request_oracle": True,
        "bounded_cache": False,
        "publication_eligible": False,
    })
elif profile in {"custom_empty", "custom_remote", "custom_restore"}:
    if plan_path is None:
        raise SystemExit("custom placement verification requires a plan")
    manifest_path = root / "runtime_remote_placement_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("runtime placement manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise SystemExit("runtime placement manifest is not an object")
    plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    expected_remote = sorted(
        [int(item["layer_id"]), int(item["expert_id"])]
        for item in plan_data.get("offload_plan", [])
    )
    expected_hot = sorted(
        [int(item["layer_id"]), int(item["expert_id"])]
        for item in plan_data.get("hot_plan", plan_data.get("prefetch_plan", []))
    )
    manifest_remote = sorted(manifest.get("remote_experts", []))
    manifest_hot = sorted(manifest.get("hot_experts", []))
    if manifest.get("status") != "ok" or manifest.get("complete") is not True:
        raise SystemExit(
            f"runtime placement manifest failed: {manifest.get('failures')}"
        )
    if manifest.get("failures") != []:
        raise SystemExit("runtime placement manifest has non-empty failures")
    expected_layers = 48 if family == "qwen" else 26
    expected_experts = expected_layers * (128 if family == "qwen" else 64)
    if manifest.get("layer_count") != expected_layers:
        raise SystemExit("runtime placement layer count mismatch")
    if manifest.get("expert_count") != expected_experts:
        raise SystemExit("runtime placement expert count mismatch")
    if manifest_remote != expected_remote or manifest_hot != expected_hot:
        raise SystemExit("runtime expert coordinates do not match the frozen plan")
    if manifest.get("remote_expert_count") != len(expected_remote):
        raise SystemExit("runtime remote expert count mismatch")
    if manifest.get("hot_expert_count") != len(expected_hot):
        raise SystemExit("runtime hot expert count mismatch")
    if manifest.get("offloaded_expert_count") != 0:
        raise SystemExit("remote-read profile contains restore-mode experts")
    if manifest.get("plan_sha256") != __import__("hashlib").sha256(
        plan_path.read_bytes()
    ).hexdigest():
        raise SystemExit("runtime placement plan SHA mismatch")
    expected_bytes = plan_data.get("metadata", {}).get(
        "actual_offload_main_bytes"
    )
    if not isinstance(expected_bytes, int) or expected_bytes < 0:
        raise SystemExit("plan lacks valid actual_offload_main_bytes")
    if manifest.get("remote_region_bytes") != expected_bytes:
        raise SystemExit("runtime remote byte count mismatch")
    release_audits = manifest.get("release_audits", [])
    if not isinstance(release_audits, list):
        raise SystemExit("runtime release audit is not a list")
    if len(release_audits) != manifest.get("remote_region_count"):
        raise SystemExit("runtime release audit/region count mismatch")
    if manifest.get("release_time_remote_mapping_verified") is not True:
        raise SystemExit("release-time remote mapping was not verified")
    physical = manifest.get("snapshot_time_remote_physical_residency")
    if not isinstance(physical, dict) or physical.get("ok") is not True:
        raise SystemExit("snapshot-time physical residency query failed")
    if expected_bytes:
        if (
            physical.get("queried_bytes") != expected_bytes
            or physical.get("cpu_bytes") != expected_bytes
            or physical.get("gpu_bytes") != 0
            or physical.get("unknown_bytes") != 0
        ):
            raise SystemExit("snapshot-time remote bytes are not wholly CPU-resident")
    gate["release_fail_fast"] = True
    gate["plan_topology_validated"] = True
    gate["runtime_remote_placement_manifest_available"] = True
    gate["runtime_remote_placement_manifest_sha256"] = __import__(
        "hashlib"
    ).sha256(manifest_path.read_bytes()).hexdigest()
    gate["runtime_remote_expert_coordinates_match_plan"] = True
    gate["runtime_remote_expert_count"] = len(expected_remote)
    gate["runtime_hot_expert_count"] = len(expected_hot)
    gate["runtime_remote_region_count"] = manifest.get("remote_region_count")
    gate["runtime_remote_region_bytes"] = manifest.get("remote_region_bytes")
    gate["release_time_remote_mapping_verified"] = True
    gate["snapshot_time_remote_cpu_residency_verified"] = True
    gate["snapshot_time_remote_mapping_query_available"] = (
        manifest.get("snapshot_time_remote_mapping_query_available") is True
    )
    gate["placement_qualified"] = True
    gate["publication_eligible"] = False
    gate["publication_blocker"] = (
        "requires a completed formal cross-profile output-hash/ABBA gate"
    )

if routing_trace_enabled:
    trace_path = root / "routing_trace.jsonl"
    if not trace_path.exists():
        raise SystemExit("routing trace was requested but is missing")
    start = float(result["measurement_start_epoch"])
    end = float(result["measurement_end_epoch"])
    trace = []
    for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        timestamp = float(value.get("timestamp", -1))
        if start <= timestamp <= end:
            trace.append(value)
    expected_layers = set(range(48)) if family == "qwen" else set(range(1, 27))
    layer_records = {layer: 0 for layer in expected_layers}
    cold_by_layer = {layer: set() for layer in expected_layers}
    if plan_path is not None:
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
        for item in plan_data.get("offload_plan", []):
            cold_by_layer[int(item["layer_id"])].add(int(item["expert_id"]))
    total_slots = 0
    cold_slots = 0
    for item in trace:
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", str(item.get("layer_name", "")))
        if match is None:
            continue
        layer = int(match.group(1))
        if layer not in expected_layers:
            continue
        layer_records[layer] += 1
        counts = item.get("expert_token_counts", {})
        if not isinstance(counts, dict):
            raise SystemExit("routing trace expert_token_counts is not an object")
        for expert_text, count_value in counts.items():
            expert = int(expert_text)
            count = int(count_value)
            if count < 0:
                raise SystemExit("routing trace contains a negative count")
            total_slots += count
            if expert in cold_by_layer[layer]:
                cold_slots += count
    missing_layers = [layer for layer, count in layer_records.items() if count == 0]
    if missing_layers:
        raise SystemExit(f"measured routing trace misses layers {missing_layers}")
    if total_slots <= 0:
        raise SystemExit("measured routing trace has zero routed slots")
    gate.update({
        "routing_trace_measured_window_complete": True,
        "routing_trace_records": len(trace),
        "routing_total_slots": total_slots,
        "routing_cold_slots": cold_slots,
        "routing_cold_slot_fraction": cold_slots / total_slots,
        "routing_layer_record_counts": {
            str(layer): layer_records[layer] for layer in sorted(layer_records)
        },
    })
(root / "integrity_gate.json").write_text(
    json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(gate, indent=2, sort_keys=True))
PY
}

run_case() {
  [[ "${ALLOW_GPU_RUN:-0}" == "1" ]] \
    || die "run-case requires ALLOW_GPU_RUN=1"
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  static_preflight

  PROFILE="${PROFILE:-}"
  MODEL_FAMILY="${MODEL_FAMILY:-}"
  local model="${MODEL:-}"
  local eval_fixture="${EVAL_FIXTURE:-}"
  local warmup_fixture="${WARMUP_FIXTURE:-}"
  local plan="${PLAN:-}"
  local case_root="${RUN_ROOT:-}"
  local concurrency="${CONCURRENCY:-1}"
  local output_tokens="${OUTPUT_TOKENS:-32}"
  local warmup_requests="${WARMUP_REQUESTS:-8}"
  local warmup_output_tokens="${WARMUP_OUTPUT_TOKENS:-8}"
  local kv_cache_bytes="${KV_CACHE_BYTES:-1288490189}"
  local max_model_len="${MAX_MODEL_LEN:-544}"
  local max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-2048}"
  local max_num_seqs="${MAX_NUM_SEQS:-32}"
  local gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.97}"
  local timeout_sec="${TIMEOUT_SEC:-3600}"
  local clean_mib="${GPU_CLEAN_MIB:-128}"
  local routing_trace="${ROUTING_TRACE:-0}"
  local formal_run="${FORMAL_RUN:-0}"
  local require_prompt_token_ids="${REQUIRE_PROMPT_TOKEN_IDS:-0}"
  local cpuset="${CPUSET:-0-13}"
  local cpu_threads="${CPU_THREADS:-14}"
  local mechanism_class=""
  local oracle_mode="off"
  local monitor_pid=""
  local rc=0
  local -a benchmark_args=()
  local -a timing_prefix=()

  [[ "${PROFILE}" =~ ^(all_gpu|custom_empty|custom_remote|custom_restore|oracle_sync|oracle_async)$ ]] \
    || die "unknown PROFILE=${PROFILE:-missing}"
  [[ "${MODEL_FAMILY}" == "qwen" || "${MODEL_FAMILY}" == "deepseek" ]] \
    || die "MODEL_FAMILY must be qwen or deepseek"
  [[ -n "${model}" && -n "${eval_fixture}" && -n "${case_root}" ]] \
    || die "MODEL, EVAL_FIXTURE and RUN_ROOT are required"
  require_dir "${model}"
  require_file "${eval_fixture}"
  [[ "${concurrency}" =~ ^[1-9][0-9]*$ ]] || die "invalid CONCURRENCY"
  [[ "${output_tokens}" =~ ^[1-9][0-9]*$ ]] || die "invalid OUTPUT_TOKENS"
  [[ "${warmup_requests}" =~ ^[0-9]+$ ]] || die "invalid WARMUP_REQUESTS"
  [[ "${cpu_threads}" =~ ^[1-9][0-9]*$ ]] || die "invalid CPU_THREADS"
  [[ "${formal_run}" == "0" || "${formal_run}" == "1" ]] \
    || die "FORMAL_RUN must be 0 or 1"
  [[ "${require_prompt_token_ids}" == "0" || "${require_prompt_token_ids}" == "1" ]] \
    || die "REQUIRE_PROMPT_TOKEN_IDS must be 0 or 1"
  if [[ "${formal_run}" == "1" ]]; then
    require_prompt_token_ids=1
  fi
  taskset -c "${cpuset}" true >/dev/null 2>&1 || die "invalid CPUSET=${cpuset}"
  if (( warmup_requests > 0 )); then
    require_file "${warmup_fixture}"
  fi
  if [[ -e "${case_root}/result.json" || -e "${case_root}/run.log" ]]; then
    die "case output already exists under ${case_root}"
  fi
  mkdir -p "${case_root}"

  case "${PROFILE}" in
    all_gpu)
      mechanism_class="framework_all_gpu_reference"
      plan=""
      ;;
    custom_empty)
      mechanism_class="custom_managed_envelope_empty_plan"
      [[ -n "${plan}" ]] || die "custom_empty requires PLAN pointing to an empty complete plan"
      validate_plan "${plan}" "${MODEL_FAMILY}" empty
      ;;
    custom_remote)
      mechanism_class="custom_fixed_cold_managed_uvm_remote_read_exact"
      [[ -n "${plan}" ]] || die "custom_remote requires PLAN"
      validate_plan "${plan}" "${MODEL_FAMILY}" nonempty
      ;;
    custom_restore)
      mechanism_class="custom_demand_restore_unbounded_ablation"
      [[ -n "${plan}" ]] || die "custom_restore requires PLAN"
      validate_plan "${plan}" "${MODEL_FAMILY}" nonempty
      [[ "${warmup_requests}" == "0" ]] \
        || die "custom_restore requires WARMUP_REQUESTS=0 because restore mutates placement"
      ;;
    oracle_sync|oracle_async)
      [[ "${MODEL_FAMILY}" == "qwen" ]] || die "oracle profiles are Qwen-only"
      [[ "${ALLOW_ORACLE_ABLATION:-0}" == "1" ]] \
        || die "oracle profile requires ALLOW_ORACLE_ABLATION=1"
      [[ "${warmup_requests}" == "0" ]] \
        || die "oracle profile requires WARMUP_REQUESTS=0; warmup consumes its only run"
      [[ -n "${plan}" ]] || die "oracle profile requires PLAN"
      validate_plan "${plan}" "${MODEL_FAMILY}" nonempty
      oracle_mode="${PROFILE#oracle_}"
      mechanism_class="same_request_oracle_unbounded_${oracle_mode}_ablation"
      ;;
  esac

  write_metadata "${case_root}" "${model}" "${eval_fixture}" \
    "${warmup_fixture}" "${plan}"

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
    VLLM_UVM_PREFILL_ORACLE_PREFETCH_MODE="${oracle_mode}"
    VLLM_UVM_PREFILL_ORACLE_MIN_TOKENS=2
    VLLM_UVM_PREFILL_ORACLE_FAIL_FAST=1
    VLLM_UVM_PREFILL_ORACLE_TRACE_FILE="${case_root}/oracle_trace.jsonl"
    VLLM_UVM_MOE_ROUTING_TRACE_ENABLE="${routing_trace}"
    VLLM_UVM_MOE_ROUTING_TRACE_FILE="${case_root}/routing_trace.jsonl"
    VLLM_ENABLE_V1_MULTIPROCESSING=0
    VLLM_CPU_OFFLOAD_BACKEND=uva
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    TOKENIZERS_PARALLELISM=false
    OMP_NUM_THREADS="${cpu_threads}"
    MKL_NUM_THREADS="${cpu_threads}"
    CUDA_VISIBLE_DEVICES=0
  )
  if [[ "${PROFILE}" == "all_gpu" ]]; then
    env_args+=(
      VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST=
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=0
      VLLM_UVM_EXPERT_ENVELOPE_ENABLE=0
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
      VLLM_UVM_INIT_WEIGHT_PLAN_FILE=
      VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=remote_read
    )
  else
    env_args+=(
      VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST="${case_root}/runtime_remote_placement_manifest.json"
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
      VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1
      VLLM_UVM_EXPERT_ENVELOPE_BYTES=524288
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup
      VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${plan}"
      VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN=1
    )
    if [[ "${PROFILE}" == "custom_restore" ]]; then
      env_args+=(VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=restore)
    else
      env_args+=(VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=remote_read)
    fi
  fi
  printf '%s\n' "${env_args[@]}" > "${case_root}/environment.txt"

  benchmark_args=(
    --model "${model}"
    --evaluation-fixture "${eval_fixture}"
    --output "${case_root}/result.json"
    --profile-label "${PROFILE}"
    --mechanism-class "${mechanism_class}"
    --concurrency "${concurrency}"
    --output-tokens "${output_tokens}"
    --warmup-requests "${warmup_requests}"
    --warmup-output-tokens "${warmup_output_tokens}"
    --kv-cache-memory-bytes "${kv_cache_bytes}"
    --max-model-len "${max_model_len}"
    --max-num-batched-tokens "${max_num_batched_tokens}"
    --max-num-seqs "${max_num_seqs}"
    --gpu-memory-utilization "${gpu_memory_utilization}"
    --seed 0
  )
  if [[ -n "${warmup_fixture}" ]]; then
    benchmark_args+=(--warmup-fixture "${warmup_fixture}")
  fi
  if [[ "${require_prompt_token_ids}" == "1" ]]; then
    benchmark_args+=(--require-prompt-token-ids)
  fi
  if [[ -x /usr/bin/time ]]; then
    timing_prefix=(/usr/bin/time -v -o "${case_root}/resource_time.txt")
  fi
  {
    printf 'env '
    printf '%q ' "${env_args[@]}"
    printf '%q ' "${timing_prefix[@]}" timeout "${timeout_sec}s" \
      taskset -c "${cpuset}" "${VLLM_PY}" "${BENCHMARK}" \
      "${benchmark_args[@]}"
    printf '\n'
  } > "${case_root}/command.txt"

  wait_for_clean_gpu "${clean_mib}"
  nvidia-smi dmon -s pucvmet -d 1 -o DT \
    > "${case_root}/gpu_dmon.csv" 2>&1 &
  monitor_pid="$!"
  set +e
  (
    cd "${VLLM_ROOT}"
    env "${env_args[@]}" "${timing_prefix[@]}" \
      timeout "${timeout_sec}s" taskset -c "${cpuset}" \
      "${VLLM_PY}" "${BENCHMARK}" "${benchmark_args[@]}"
  ) > "${case_root}/run.log" 2>&1
  rc=$?
  set -e
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  printf '%s\n' "${rc}" > "${case_root}/returncode.txt"
  if [[ "${rc}" -ne 0 ]]; then
    tail -n 240 "${case_root}/run.log" >&2 || true
    return "${rc}"
  fi
  verify_case_result "${case_root}" "${concurrency}" "${output_tokens}" \
    "${PROFILE}" "${plan}" "${MODEL_FAMILY}" "${routing_trace}" "${formal_run}"
  wait_for_clean_gpu "${clean_mib}"
  echo "result=${case_root}/result.json"
  echo "integrity_gate=${case_root}/integrity_gate.json"
}

run_abba() {
  [[ "${ALLOW_GPU_RUN:-0}" == "1" ]] \
    || die "run-abba requires ALLOW_GPU_RUN=1"
  local abba_root="${RUN_ROOT:-}"
  local empty_plan="${EMPTY_PLAN:-}"
  local treatment_plan="${PLAN:-}"
  local concurrencies="${CONCURRENCIES:-1,4,8}"
  local blocks="${ABBA_BLOCKS:-3}"
  local family="${MODEL_FAMILY:-}"
  [[ -n "${abba_root}" && -n "${empty_plan}" && -n "${treatment_plan}" ]] \
    || die "RUN_ROOT, EMPTY_PLAN and PLAN are required"
  [[ "${blocks}" =~ ^[1-9][0-9]*$ ]] || die "invalid ABBA_BLOCKS"
  [[ "${family}" == "qwen" || "${family}" == "deepseek" ]] \
    || die "MODEL_FAMILY must be qwen or deepseek"
  validate_plan "${empty_plan}" "${family}" empty
  validate_plan "${treatment_plan}" "${family}" nonempty
  if [[ -e "${abba_root}/abba_order.tsv" ]]; then
    die "ABBA root already contains abba_order.tsv"
  fi
  mkdir -p "${abba_root}"
  printf 'concurrency\tblock\tslot\tarm\tprofile\tcase_root\n' \
    > "${abba_root}/abba_order.tsv"

  local -a concurrency_values=()
  IFS=',' read -r -a concurrency_values <<< "${concurrencies}"
  local concurrency block slot arm profile case_root selected_plan
  local -a sequence=()

  # Freeze the complete schedule before launching the first process.  This
  # preserves the denominator even when an individual cold process fails.
  for concurrency in "${concurrency_values[@]}"; do
    [[ "${concurrency}" =~ ^[1-9][0-9]*$ ]] \
      || die "invalid concurrency in CONCURRENCIES: ${concurrency}"
    for block in $(seq 1 "${blocks}"); do
      if (( block % 2 == 1 )); then
        sequence=(A B B A)
      else
        sequence=(B A A B)
      fi
      slot=0
      for arm in "${sequence[@]}"; do
        slot=$((slot + 1))
        if [[ "${arm}" == "A" ]]; then
          profile="custom_empty"
        else
          profile="custom_remote"
        fi
        case_root="${abba_root}/c${concurrency}/block${block}/slot${slot}_${arm}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
          "${concurrency}" "${block}" "${slot}" "${arm}" \
          "${profile}" "${case_root}" >> "${abba_root}/abba_order.tsv"
      done
    done
  done

  printf 'concurrency\tblock\tslot\tarm\tprofile\tcase_root\tstatus\treturncode\tstarted_at\tended_at\n' \
    > "${abba_root}/execution_status.tsv"
  local status started_at ended_at rc
  local failed_cases=0
  while IFS=$'\t' read -r concurrency block slot arm profile case_root; do
    if [[ "${arm}" == "A" ]]; then
      selected_plan="${empty_plan}"
    else
      selected_plan="${treatment_plan}"
    fi
    started_at="$(date --iso-8601=seconds)"
    if ALLOW_GPU_RUN=1 \
        PROFILE="${profile}" \
        MODEL_FAMILY="${family}" \
        MODEL="${MODEL:-}" \
        EVAL_FIXTURE="${EVAL_FIXTURE:-}" \
        WARMUP_FIXTURE="${WARMUP_FIXTURE:-}" \
        PLAN="${selected_plan}" \
        RUN_ROOT="${case_root}" \
        CONCURRENCY="${concurrency}" \
        OUTPUT_TOKENS="${OUTPUT_TOKENS:-64}" \
        WARMUP_REQUESTS="${WARMUP_REQUESTS:-8}" \
        WARMUP_OUTPUT_TOKENS="${WARMUP_OUTPUT_TOKENS:-8}" \
        KV_CACHE_BYTES="${KV_CACHE_BYTES:-1288490189}" \
        MAX_MODEL_LEN="${MAX_MODEL_LEN:-544}" \
        MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}" \
        MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}" \
        GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}" \
        TIMEOUT_SEC="${TIMEOUT_SEC:-3600}" \
        ROUTING_TRACE="${ROUTING_TRACE:-0}" \
        FORMAL_RUN="${FORMAL_RUN:-0}" \
        REQUIRE_PROMPT_TOKEN_IDS="${REQUIRE_PROMPT_TOKEN_IDS:-0}" \
        CPUSET="${CPUSET:-0-13}" \
        CPU_THREADS="${CPU_THREADS:-14}" \
        GPU_CLEAN_MIB="${GPU_CLEAN_MIB:-128}" \
          bash "${BASH_SOURCE[0]}" run-case; then
      status="success"
      rc=0
    else
      rc=$?
      status="failed"
      failed_cases=$((failed_cases + 1))
    fi
    ended_at="$(date --iso-8601=seconds)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${concurrency}" "${block}" "${slot}" "${arm}" "${profile}" \
      "${case_root}" "${status}" "${rc}" "${started_at}" "${ended_at}" \
      >> "${abba_root}/execution_status.tsv"
  done < <(tail -n +2 "${abba_root}/abba_order.tsv")

  "${VLLM_PY}" "${ABBA_SUMMARIZER}" "${abba_root}"
  echo "abba_order=${abba_root}/abba_order.tsv"
  echo "execution_status=${abba_root}/execution_status.tsv"
  echo "abba_summary=${abba_root}/custom_abba_summary.json"
  echo "note=This ABBA validates custom-empty vs custom-remote only; vendor ABBA needs the top-level orchestrator."
  if (( failed_cases > 0 )); then
    echo "error: ${failed_cases} scheduled ABBA cases failed; all scheduled cases were retained" >&2
    return 1
  fi
}

case "${COMMAND}" in
  preflight)
    static_preflight
    ;;
  matrix)
    print_matrix
    ;;
  validate-plan)
    [[ -n "${PLAN:-}" && -n "${MODEL_FAMILY:-}" ]] \
      || die "validate-plan requires PLAN and MODEL_FAMILY"
    validate_plan "${PLAN}" "${MODEL_FAMILY}" "${PLAN_KIND:-any}"
    ;;
  run-case)
    run_case
    ;;
  run-abba)
    run_abba
    ;;
  help|-h|--help)
    usage_text
    ;;
  *)
    usage_text >&2
    die "unknown command: ${COMMAND}"
    ;;
esac
