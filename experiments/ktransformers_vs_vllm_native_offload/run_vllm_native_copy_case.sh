#!/usr/bin/env bash
set -euo pipefail

# Shared instrumented case runner.  Its legacy default is the supplemental
# copy-on-forward ablation; run_vllm_native_uva_case.sh selects the native UVA
# backend used by the main comparison.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_INSTALL_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_ROOT="${VLLM_SOURCE_ROOT:-${VLLM_INSTALL_ROOT}}"
VLLM_PY="${VLLM_INSTALL_ROOT}/.venv/bin/python"
PHASED_RUNNER="${VLLM_ROOT}/uvm_test/vllm_generate_phased.py"
UVM_MODULE="${UVM_MODULE:-${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"

EXPECTED_REPO_COMMIT="${EXPECTED_REPO_COMMIT:-e7a4db7f49f7b200316a762074476c41d3c1de78}"
EXPECTED_VLLM_VERSION="${EXPECTED_VLLM_VERSION:-0.1.dev11683+gd9d342d21.d20260330}"
EXPECTED_MODEL_SNAPSHOT="${EXPECTED_MODEL_SNAPSHOT:-d206ba732169f29bb77fbf80fc2c4b81d4d30782}"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/${EXPECTED_MODEL_SNAPSHOT}}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_native_copy_case_$(date +%Y%m%d_%H%M%S)}"
FIXTURE="${FIXTURE:-}"
OFFLOAD_GIB="${OFFLOAD_GIB:-6.0}"
OFFLOAD_BACKEND="${OFFLOAD_BACKEND:-copy}"
CONCURRENCIES="${CONCURRENCIES:-1,4,8}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-64}"
WARM_OUTPUT_TOKENS="${WARM_OUTPUT_TOKENS:-8}"
CPUSET="${CPUSET:-0-13}"

# Qwen3-30B-A3B-FP8 reports exactly 13,104 KV tokens for this byte budget.
KV_CACHE_BYTES="${KV_CACHE_BYTES:-1288490189}"
EXPECTED_KV_TOKENS="${EXPECTED_KV_TOKENS:-13104}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"
GPU_CLEAN_MIB="${GPU_CLEAN_MIB:-128}"
REQUIRE_CLEAN_VLLM_SOURCE="${REQUIRE_CLEAN_VLLM_SOURCE:-0}"

RESULT="${RUN_ROOT}/result.json"
PROGRESS="${RUN_ROOT}/progress.jsonl"
RUN_LOG="${RUN_ROOT}/run.log"
RETURN_CODE="${RUN_ROOT}/returncode.txt"
OFFLOAD_MANIFEST="${RUN_ROOT}/cpu_offload_manifest.jsonl"
CASE_SUMMARY="${RUN_ROOT}/case_summary.json"

RUN_PID=""
MONITOR_PIDS=()

die() {
  echo "error: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "missing file: $1"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

stop_monitors() {
  local pid
  for pid in "${MONITOR_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]]; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  for pid in "${MONITOR_PIDS[@]:-}"; do
    if [[ -n "${pid}" ]]; then
      wait "${pid}" 2>/dev/null || true
    fi
  done
  MONITOR_PIDS=()
}

stop_run() {
  local _
  if [[ -z "${RUN_PID}" ]]; then
    return 0
  fi
  if kill -0 "${RUN_PID}" 2>/dev/null; then
    kill -TERM -- "-${RUN_PID}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 "${RUN_PID}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
  fi
  if kill -0 "${RUN_PID}" 2>/dev/null; then
    kill -KILL -- "-${RUN_PID}" 2>/dev/null || true
  fi
  wait "${RUN_PID}" 2>/dev/null || true
  RUN_PID=""
}

cleanup() {
  stop_run
  stop_monitors
}
trap cleanup EXIT INT TERM

wait_for_clean_gpu() {
  local used_mib=""
  local compute_pids=""
  for _ in $(seq 1 180); do
    used_mib="$(
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
        head -n 1 | tr -d ' '
    )"
    compute_pids="$(
      nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        2>/dev/null || true
    )"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] \
      && (( used_mib <= GPU_CLEAN_MIB )) \
      && [[ -z "${compute_pids//[[:space:]]/}" ]]; then
      return 0
    fi
    sleep 1
  done
  nvidia-smi >&2 || true
  die "GPU is not clean: used=${used_mib:-unknown} MiB compute_pids=${compute_pids:-none}"
}

preflight() {
  require_file "${VLLM_PY}"
  require_file "${PHASED_RUNNER}"
  require_file "${UVM_MODULE}"
  require_file "${MODEL}/config.json"
  [[ -n "${FIXTURE}" ]] || die "FIXTURE must point to the shared evaluation fixture"
  require_file "${FIXTURE}"
  require_command git
  require_command nvidia-smi
  require_command taskset
  require_command timeout
  require_command sha256sum
  require_command lscpu
  require_command ps

  local repo_commit source_commit source_status model_snapshot loaded_uvm expected_uvm
  repo_commit="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  source_commit="$(git -C "${VLLM_ROOT}" rev-parse HEAD)"
  source_status="$(git -C "${VLLM_ROOT}" status --short --untracked-files=no)"
  model_snapshot="$(basename "${MODEL}")"
  loaded_uvm="$(cat /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)"
  expected_uvm="$(modinfo -F srcversion "${UVM_MODULE}" 2>/dev/null || true)"

  [[ "${repo_commit}" == "${EXPECTED_REPO_COMMIT}" ]] \
    || die "repository commit drift: ${repo_commit}"
  [[ "${source_commit}" == "${EXPECTED_REPO_COMMIT}" ]] \
    || die "vLLM source commit drift: ${source_commit}"
  if [[ "${REQUIRE_CLEAN_VLLM_SOURCE}" == "1" && -n "${source_status}" ]]; then
    die "vLLM source worktree is not clean: ${source_status}"
  fi
  [[ "${model_snapshot}" == "${EXPECTED_MODEL_SNAPSHOT}" ]] \
    || die "model snapshot drift: ${model_snapshot}"
  [[ -n "${loaded_uvm}" && "${loaded_uvm}" == "${expected_uvm}" ]] \
    || die "nvidia_uvm srcversion mismatch: loaded=${loaded_uvm:-missing} expected=${expected_uvm:-missing}"

  PYTHONPATH="${VLLM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    PATH="${VLLM_INSTALL_ROOT}/.venv/bin:${PATH}" "${VLLM_PY}" - \
    "${EXPECTED_VLLM_VERSION}" "${FIXTURE}" "${CONCURRENCIES}" \
    "${MAX_NUM_SEQS}" "${MAX_MODEL_LEN}" "${OUTPUT_TOKENS}" \
    "${OFFLOAD_BACKEND}" <<'PY'
import json
import sys

import torch
import vllm

expected_version, fixture_path, raw_concurrencies = sys.argv[1:4]
max_num_seqs, max_model_len, output_tokens = map(int, sys.argv[4:7])
offload_backend = sys.argv[7]
assert vllm.__version__ == expected_version, (vllm.__version__, expected_version)
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability() == (12, 0)

parts = [part.strip() for part in raw_concurrencies.split(",") if part.strip()]
assert parts, "CONCURRENCIES is empty"
concurrencies = [int(part) for part in parts]
assert all(value > 0 for value in concurrencies), concurrencies
assert len(set(concurrencies)) == len(concurrencies), concurrencies
assert max(concurrencies) <= max_num_seqs, (concurrencies, max_num_seqs)
assert output_tokens >= 2, output_tokens
assert offload_backend in {"copy", "uva"}, offload_backend

payload = json.load(open(fixture_path, encoding="utf-8"))
requests = payload.get("requests") if isinstance(payload, dict) else payload
assert isinstance(requests, list) and len(requests) >= max(concurrencies)
for index, item in enumerate(requests[: max(concurrencies)]):
    assert isinstance(item, dict) and isinstance(item.get("prompt"), str)
    declared = item.get("prompt_tokens")
    if declared is not None:
        assert int(declared) + output_tokens <= max_model_len, (index, declared)
print(
    f"vLLM preflight: version={vllm.__version__} torch={torch.__version__} "
    f"fixture_requests={len(requests)} concurrencies={concurrencies}"
)
PY

  local online_cpus
  online_cpus="$(nproc)"
  [[ "${online_cpus}" == "14" ]] \
    || die "online CPU count drift: ${online_cpus}, expected 14"
  wait_for_clean_gpu
}

build_phases() {
  local -n names_ref="$1"
  local -n counts_ref="$2"
  local -n outputs_ref="$3"
  local -a values=()
  local value max_concurrency=0

  IFS=',' read -r -a values <<< "${CONCURRENCIES}"
  for value in "${values[@]}"; do
    value="${value//[[:space:]]/}"
    (( value > max_concurrency )) && max_concurrency="${value}"
  done

  names_ref=("warm_c${max_concurrency}")
  counts_ref=("${max_concurrency}")
  outputs_ref=("${WARM_OUTPUT_TOKENS}")
  for value in "${values[@]}"; do
    value="${value//[[:space:]]/}"
    names_ref+=("sharegpt_c${value}")
    counts_ref+=("${value}")
    outputs_ref+=("${OUTPUT_TOKENS}")
  done
}

join_by_comma() {
  local IFS=','
  echo "$*"
}

record_metadata() {
  mkdir -p "${RUN_ROOT}"
  [[ ! -e "${RESULT}" && ! -e "${RUN_LOG}" ]] \
    || die "RUN_ROOT already contains a run: ${RUN_ROOT}"

  git -C "${REPO_ROOT}" rev-parse HEAD > "${RUN_ROOT}/repository.commit"
  git -C "${REPO_ROOT}" status --short > "${RUN_ROOT}/repository.status"
  git -C "${VLLM_ROOT}" rev-parse HEAD > "${RUN_ROOT}/vllm_source.commit"
  git -C "${VLLM_ROOT}" status --short > "${RUN_ROOT}/vllm_source.status"
  printf '%s\n' "$(realpath "${VLLM_ROOT}")" > "${RUN_ROOT}/vllm_source_root.txt"
  if command -v uv >/dev/null 2>&1; then
    uv pip freeze --python "${VLLM_PY}" \
      > "${RUN_ROOT}/vllm_pip_freeze.txt"
  else
    "${VLLM_PY}" - > "${RUN_ROOT}/vllm_pip_freeze.txt" <<'PY'
from importlib.metadata import distributions

rows = []
for distribution in distributions():
    name = distribution.metadata.get("Name")
    if name:
        rows.append(f"{name}=={distribution.version}")
print("\n".join(sorted(rows, key=str.casefold)))
PY
  fi
  lscpu > "${RUN_ROOT}/lscpu.txt"
  nvidia-smi -q > "${RUN_ROOT}/nvidia_smi_q_before.txt"
  sha256sum "${FIXTURE}" > "${RUN_ROOT}/fixture.sha256"
  printf '%s\n' "${OFFLOAD_GIB}" > "${RUN_ROOT}/cpu_offload_gib.txt"
  printf '%s\n' "${OFFLOAD_BACKEND}" > "${RUN_ROOT}/cpu_offload_backend.txt"
  printf '%s\n' "${CONCURRENCIES}" > "${RUN_ROOT}/concurrencies.txt"
  printf '%s\n' "${CPUSET}" > "${RUN_ROOT}/cpuset.txt"
  {
    echo "loaded_srcversion=$(cat /sys/module/nvidia_uvm/srcversion)"
    echo "module_srcversion=$(modinfo -F srcversion "${UVM_MODULE}")"
    echo "module=$(realpath "${UVM_MODULE}")"
  } > "${RUN_ROOT}/uvm_module.txt"

  "${VLLM_PY}" - "${FIXTURE}" "${RUN_ROOT}/fixture_metadata.json" <<'PY'
import hashlib
import json
import sys

source, output = sys.argv[1:]
raw = open(source, "rb").read()
payload = json.loads(raw)
requests = payload.get("requests") if isinstance(payload, dict) else payload
prompts = [item["prompt"] for item in requests]
metadata = {
    "path": source,
    "sha256": hashlib.sha256(raw).hexdigest(),
    "request_count": len(requests),
    "prompt_list_sha256": hashlib.sha256(
        json.dumps(prompts, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest(),
    "declared_prompt_tokens": [item.get("prompt_tokens") for item in requests],
}
open(output, "w", encoding="utf-8").write(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n"
)
PY
}

start_gpu_monitors() {
  nvidia-smi \
    --query-gpu=timestamp,index,memory.used,utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu,pcie.link.gen.current,pcie.link.width.current \
    --format=csv,noheader,nounits -lms 200 \
    > "${RUN_ROOT}/resource_nvml.csv" 2> "${RUN_ROOT}/resource_nvml.err" &
  MONITOR_PIDS+=("$!")

  nvidia-smi dmon -i 0 -s t -d 1 -o DT \
    > "${RUN_ROOT}/resource_pcie_dmon.txt" 2>&1 &
  MONITOR_PIDS+=("$!")
}

start_process_monitor() {
  (
    while kill -0 "${RUN_PID}" 2>/dev/null; do
      date --iso-8601=ns
      ps -s "${RUN_PID}" -o pid=,ppid=,rss=,vsz=,pcpu=,stat=,comm=
      sleep 0.2
    done
  ) > "${RUN_ROOT}/resource_process_tree.txt" 2>&1 &
  MONITOR_PIDS+=("$!")
}

validate_and_summarize() {
  "${VLLM_PY}" - \
    "${RESULT}" "${PROGRESS}" "${OFFLOAD_MANIFEST}" "${RUN_LOG}" \
    "${RUN_ROOT}/fixture_metadata.json" "${CASE_SUMMARY}" \
    "${CONCURRENCIES}" "${OUTPUT_TOKENS}" "${WARM_OUTPUT_TOKENS}" \
    "${KV_CACHE_BYTES}" "${EXPECTED_KV_TOKENS}" "${OFFLOAD_GIB}" \
    "${OFFLOAD_BACKEND}" <<'PY'
import hashlib
import json
import re
import sys

(
    result_path,
    progress_path,
    manifest_path,
    log_path,
    fixture_metadata_path,
    output_path,
    raw_concurrencies,
    output_tokens,
    warm_output_tokens,
    kv_cache_bytes,
    expected_kv_tokens,
    offload_gib,
    offload_backend,
) = sys.argv[1:]
output_tokens = int(output_tokens)
warm_output_tokens = int(warm_output_tokens)
kv_cache_bytes = int(kv_cache_bytes)
expected_kv_tokens = int(expected_kv_tokens)
concurrencies = [int(part.strip()) for part in raw_concurrencies.split(",") if part.strip()]
max_concurrency = max(concurrencies)

result = json.load(open(result_path, encoding="utf-8"))
fixture = json.load(open(fixture_metadata_path, encoding="utf-8"))
assert result.get("status") == "ok", result.get("error")
config = result["config"]
assert int(config["kv_cache_memory_bytes"]) == kv_cache_bytes
assert float(config["cpu_offload_gb"]) == float(offload_gib)
assert int(config["max_model_len"]) == 512
assert int(config["max_num_seqs"]) == 32
assert int(config["max_num_batched_tokens"]) == 2048
assert config["enable_prefix_caching"] is False
assert int(config["seed"]) == 0

expected = [
    (f"warm_c{max_concurrency}", max_concurrency, warm_output_tokens),
    *[(f"sharegpt_c{value}", value, output_tokens) for value in concurrencies],
]
phases = result.get("phases") or []
assert len(phases) == len(expected), (len(phases), expected)
declared_lengths = fixture["declared_prompt_tokens"]
normalized_phases = []
for phase, (name, count, per_request) in zip(phases, expected):
    assert phase["phase"] == name, (phase["phase"], name)
    assert int(phase["request_count"]) == count
    assert int(phase["completed_requests"]) == count
    assert int(phase["output_tokens_per_request"]) == per_request
    assert int(phase["output_tokens"]) == count * per_request
    assert int(phase["prompt_tokens"]) == sum(map(int, declared_lengths[:count]))
    requests = phase["requests"]
    assert len(requests) == count
    assert all(int(item["output_tokens"]) == per_request for item in requests)
    normalized_phases.append(
        {
            "phase": name,
            "concurrency": count,
            "output_tokens_per_request": per_request,
            "prompt_tokens": int(phase["prompt_tokens"]),
            "output_tokens": int(phase["output_tokens"]),
            "elapsed_seconds": float(phase["elapsed_seconds"]),
            "output_tokens_per_second": float(phase["output_tokens_per_second"]),
            "mean_ttft_ms": phase.get("mean_ttft_ms"),
            "p99_ttft_ms": phase.get("p99_ttft_ms"),
            "mean_tpot_ms": phase.get("mean_tpot_ms"),
            "p99_tpot_ms": phase.get("p99_tpot_ms"),
            "output_token_ids_sha256": hashlib.sha256(
                json.dumps(
                    [item["output_token_ids"] for item in requests],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
    )

manifest = [
    json.loads(line)
    for line in open(manifest_path, encoding="utf-8")
    if line.strip()
]
parameters = [item for item in manifest if item.get("event") == "cpu_offload_parameter"]
copy_traces = [item for item in manifest if item.get("event") == "cpu_offload_copy_forward"]
assert parameters, "CPU offload manifest contains no parameters"
assert not copy_traces, "performance run unexpectedly enabled per-forward copy tracing"
assert offload_backend in {"copy", "uva"}, offload_backend
assert all(item.get("offload_backend") == offload_backend for item in parameters)
assert all(item.get("cpu_data_pinned") is True for item in parameters)
if offload_backend == "copy":
    assert all(item.get("copy_on_forward") is True for item in parameters)
    assert all(item.get("device_after") == "cpu" for item in parameters)
    mechanism = "vllm_cpu_offload_copy_on_forward_ablation"
else:
    assert all(item.get("uva_offloading") is True for item in parameters)
    assert all(item.get("copy_on_forward") in (None, False) for item in parameters)
    assert all(str(item.get("device_after", "")).startswith("cuda") for item in parameters)
    mechanism = "vllm_native_cpu_offload_pinned_uva"

log_text = open(log_path, encoding="utf-8", errors="replace").read()
matches = re.findall(r"GPU KV cache size:\s*([\d,]+) tokens", log_text)
assert matches, "run log has no GPU KV cache size"
reported_kv_tokens = int(matches[-1].replace(",", ""))
assert reported_kv_tokens == expected_kv_tokens, (
    reported_kv_tokens,
    expected_kv_tokens,
)
bad_log_patterns = [
    pattern
    for pattern in (r"CUDA out of memory", r"EngineCore encountered a fatal error", r"preempt")
    if re.search(pattern, log_text, flags=re.IGNORECASE)
]
assert not bad_log_patterns, bad_log_patterns

signature = [
    {
        "name": item.get("full_parameter_name"),
        "shape": item.get("shape"),
        "dtype": item.get("dtype"),
        "bytes": int(item.get("bytes") or 0),
        "order": int(item.get("offload_order") or 0),
    }
    for item in parameters
]
signature_sha = hashlib.sha256(
    json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
summary = {
    "status": "ok",
    "mechanism": mechanism,
    "measurement_scope": "offline_burst_complete_runtime",
    "fixture": fixture,
    "config": {
        "cpu_offload_gib_requested": float(offload_gib),
        "cpu_offload_backend": offload_backend,
        "cpu_offload_bytes_actual": sum(item["bytes"] for item in signature),
        "offloaded_parameter_count": len(signature),
        "offload_signature_sha256": signature_sha,
        "copy_trace": False,
        "copy_sync": False,
        "kv_cache_memory_bytes": kv_cache_bytes,
        "kv_cache_tokens_reported": reported_kv_tokens,
        "concurrencies": concurrencies,
        "output_tokens": output_tokens,
    },
    "phases": normalized_phases,
    "integrity": {
        "all_requests_complete": True,
        "fixed_output_lengths": True,
        "prefix_cache_disabled": True,
        "performance_trace_and_sync_disabled": True,
        "all_offloaded_parameters_pinned": True,
        "offload_backend_verified": True,
        "no_oom_fatal_or_preemption_log": True,
    },
}
open(output_path, "w", encoding="utf-8").write(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

main() {
  local -a phase_names=()
  local -a phase_counts=()
  local -a phase_outputs=()
  local -a command=()
  local runner_rc

  preflight
  record_metadata
  build_phases phase_names phase_counts phase_outputs

  local -a environment=(
    VLLM_USE_UVM=0
    VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP=0
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED=0
    VLLM_UVM_KV_CACHE_MANAGED=0
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
    VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
    VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0
    VLLM_UVM_KV_RUNTIME_ENABLE=0
    VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0
    VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0
    VLLM_UVM_MATCHED_CPU_OFFLOAD_ENABLE=0
    VLLM_UVM_FIXED_WEIGHT_DEVICE_DIRECT_ENABLE=0
    VLLM_UVM_PREFILL_ORACLE_PREFETCH_MODE=off
    VLLM_MOE_TOKEN_EXPERT_TRACE_ENABLE=0
    VLLM_QWEN3_MOE_EXPERT_TRACE_ENABLE=0
    VLLM_ENABLE_V1_MULTIPROCESSING=0
    VLLM_CPU_OFFLOAD_BACKEND="${OFFLOAD_BACKEND}"
    VLLM_CPU_OFFLOAD_MANIFEST_FILE="${OFFLOAD_MANIFEST}"
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1
    VLLM_CPU_OFFLOAD_COPY_TRACE=0
    VLLM_CPU_OFFLOAD_COPY_SYNC=0
    VLLM_LOG_BATCHSIZE_INTERVAL=-1
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    TOKENIZERS_PARALLELISM=false
    PYTHONHASHSEED=0
    PYTHONPATH="${VLLM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  )

  command=(
    timeout --signal=TERM --kill-after=30s "${TIMEOUT_SEC}s"
    taskset -c "${CPUSET}"
    "${VLLM_PY}" "${PHASED_RUNNER}"
    --model "${MODEL}"
    --fixture "${FIXTURE}"
    --output "${RESULT}"
    --progress "${PROGRESS}"
    --phase-names "$(join_by_comma "${phase_names[@]}")"
    --phase-request-counts "$(join_by_comma "${phase_counts[@]}")"
    --phase-output-tokens "$(join_by_comma "${phase_outputs[@]}")"
    --kv-cache-memory-bytes "${KV_CACHE_BYTES}"
    --cpu-offload-gb "${OFFLOAD_GIB}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  )

  printf '%s\n' "${environment[@]}" > "${RUN_ROOT}/environment.txt"
  printf '%q ' env "${environment[@]}" "${command[@]}" \
    > "${RUN_ROOT}/command.txt"
  printf '\n' >> "${RUN_ROOT}/command.txt"

  start_gpu_monitors
  set +e
  (
    cd "${VLLM_ROOT}"
    exec setsid env "${environment[@]}" "${command[@]}"
  ) > "${RUN_LOG}" 2>&1 &
  RUN_PID="$!"
  echo "${RUN_PID}" > "${RUN_ROOT}/run.pid"
  start_process_monitor
  wait "${RUN_PID}"
  runner_rc=$?
  RUN_PID=""
  set -e
  stop_monitors
  printf '%s\n' "${runner_rc}" > "${RETURN_CODE}"

  if [[ "${runner_rc}" -ne 0 ]]; then
    tail -n 240 "${RUN_LOG}" >&2 || true
    die "vLLM copy case failed with return code ${runner_rc}"
  fi

  validate_and_summarize > "${RUN_ROOT}/case_summary.stdout.json"
  nvidia-smi --query-gpu=timestamp,memory.used \
    --format=csv,noheader,nounits > "${RUN_ROOT}/gpu_after.csv"
  wait_for_clean_gpu
  echo "RUN_ROOT=${RUN_ROOT}"
  echo "CASE_SUMMARY=${CASE_SUMMARY}"
}

main "$@"
