#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KT_ROOT="${REPO_ROOT}/KTransformers"
KT_PY="${KT_ROOT}/.venv/bin/python"
VLLM_PY="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
BENCHMARK="${SCRIPT_DIR}/benchmark_sglang_stream.py"
CONVERTER="${SCRIPT_DIR}/cold_plan_to_kt_stats.py"
PREPARE_FIXTURES="${REPO_ROOT}/experiments/custom_cold_qwen_sharegpt_concurrency_snapshot/prepare_sharegpt_fixtures.py"
REFERENCE_SNAPSHOT="${REPO_ROOT}/experiments/native_uva_qwen_sharegpt_concurrency_snapshot/snapshot_20260812.json"

EXPECTED_KT_COMMIT="eb9b70c4115cff151ace2cae5b0fc9db3690e31e"
EXPECTED_SGLANG_COMMIT="bc7f0058fafe6c05dc6ad52d6b1c854af2a673b2"
EXPECTED_MODEL_SNAPSHOT="d206ba732169f29bb77fbf80fc2c4b81d4d30782"
EXPECTED_DATASET_SHA256="35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/${EXPECTED_MODEL_SNAPSHOT}}"
DATASET="${DATASET:-${REPO_ROOT}/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
PORT="${PORT:-30000}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"
KT_CPU_THREADS="${KT_CPU_THREADS:-14}"
CPUSET="${CPUSET:-0-13}"
KV_TOKENS="${KV_TOKENS:-13104}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
MAX_CONTEXT_LENGTH="${MAX_CONTEXT_LENGTH:-512}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-2048}"
CONCURRENCIES="${CONCURRENCIES:-1,2,3,4,8,16,24,32}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-64}"
WARM_CONCURRENCY="${WARM_CONCURRENCY:-32}"
WARM_OUTPUT_TOKENS="${WARM_OUTPUT_TOKENS:-8}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
GPU_CLEAN_MIB="${GPU_CLEAN_MIB:-128}"

COMMAND="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi
RUN_ROOT="${RUN_ROOT:-/tmp/ktransformers_cpu_expert_compare_${COMMAND}_$(date +%Y%m%d_%H%M%S)}"
FIXTURE="${FIXTURE:-${RUN_ROOT}/fixtures/evaluation_fixture.json}"

SERVER_PID=""
MONITOR_PIDS=()

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

wait_for_clean_gpu() {
  local used_mib=""
  local compute_processes=""
  for _ in $(seq 1 180); do
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    compute_processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] \
      && (( used_mib <= GPU_CLEAN_MIB )) \
      && [[ -z "${compute_processes//[[:space:]]/}" ]]; then
      return 0
    fi
    sleep 1
  done
  nvidia-smi >&2 || true
  die "GPU did not become clean (used=${used_mib:-unknown} MiB, compute_pids=${compute_processes:-none})"
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

stop_server() {
  local _
  stop_monitors
  if [[ -z "${SERVER_PID}" ]]; then
    return 0
  fi
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -INT -- "-${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
  fi
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    sleep 2
  fi
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
  fi
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
}

cleanup() {
  stop_server
}
trap cleanup EXIT

preflight_kt() {
  require_dir "${KT_ROOT}"
  require_file "${KT_PY}"
  require_file "${BENCHMARK}"
  require_file "${CONVERTER}"
  require_dir "${MODEL}"
  require_file "${MODEL}/config.json"
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  command -v curl >/dev/null || die "curl is unavailable"
  command -v taskset >/dev/null || die "taskset is unavailable"

  local kt_commit sglang_commit model_snapshot
  kt_commit="$(git -C "${KT_ROOT}" rev-parse HEAD)"
  sglang_commit="$(git -C "${KT_ROOT}/third_party/sglang" rev-parse HEAD)"
  model_snapshot="$(basename "${MODEL}")"
  [[ "${kt_commit}" == "${EXPECTED_KT_COMMIT}" ]] \
    || die "KTransformers commit drift: ${kt_commit}"
  [[ "${sglang_commit}" == "${EXPECTED_SGLANG_COMMIT}" ]] \
    || die "SGLang commit drift: ${sglang_commit}"
  [[ "${model_snapshot}" == "${EXPECTED_MODEL_SNAPSHOT}" ]] \
    || die "model snapshot drift: ${model_snapshot}"

  PATH="${KT_ROOT}/.venv/bin:${PATH}" "${KT_PY}" - <<'PY'
import torch
import kt_kernel
from kt_kernel import kt_kernel_ext

assert torch.cuda.is_available()
assert torch.cuda.get_device_capability() == (12, 0)
assert getattr(kt_kernel, "__version__", None) == "0.6.4"
assert getattr(kt_kernel, "__cpu_variant__", None) == "avx512_bf16"
assert hasattr(kt_kernel_ext.CPUInfer, "submit_with_cuda_stream")
print("KT preflight: version=0.6.4 cpu_variant=avx512_bf16 capability=12.0")
PY

  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    die "${BASE_URL} is already serving; choose another PORT or stop it"
  fi
  wait_for_clean_gpu
}

preflight_full_compare() {
  preflight_kt
  require_file "${DATASET}"
  local dataset_sha loaded_uvm expected_uvm link_width link_width_max
  dataset_sha="$(sha256sum "${DATASET}" | awk '{print $1}')"
  [[ "${dataset_sha}" == "${EXPECTED_DATASET_SHA256}" ]] \
    || die "ShareGPT hash drift: ${dataset_sha}"
  loaded_uvm="$(cat /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)"
  expected_uvm="$(modinfo -F srcversion "${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko" 2>/dev/null || true)"
  [[ -n "${loaded_uvm}" && "${loaded_uvm}" == "${expected_uvm}" ]] \
    || die "nvidia_uvm srcversion mismatch: loaded=${loaded_uvm:-missing} worktree=${expected_uvm:-missing}"
  IFS=',' read -r link_width link_width_max < <(
    nvidia-smi --query-gpu=pcie.link.width.current,pcie.link.width.max \
      --format=csv,noheader,nounits | head -n 1 | tr -d ' '
  )
  [[ "${link_width}" == "16" && "${link_width_max}" == "16" ]] \
    || die "PCIe width is x${link_width:-unknown}, expected active/max x16"
  echo "full comparison preflight: OK"
}

prepare_fixture() {
  if [[ -s "${FIXTURE}" ]]; then
    return 0
  fi
  require_file "${DATASET}"
  require_file "${REFERENCE_SNAPSHOT}"
  require_file "${PREPARE_FIXTURES}"
  require_file "${VLLM_PY}"
  mkdir -p "$(dirname "${FIXTURE}")"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "${VLLM_PY}" "${PREPARE_FIXTURES}" \
      --dataset "${DATASET}" \
      --model "${MODEL}" \
      --reference-snapshot "${REFERENCE_SNAPSHOT}" \
      --output-dir "$(dirname "${FIXTURE}")" \
      --request-count 32 \
      --prompt-min-tokens 300 \
      --prompt-max-tokens 320 \
      --output-tokens 64 \
      > "${RUN_ROOT}/prepare_fixture.log"
  require_file "${FIXTURE}"
}

record_metadata() {
  mkdir -p "${RUN_ROOT}"
  git -C "${KT_ROOT}" rev-parse HEAD > "${RUN_ROOT}/ktransformers.commit"
  git -C "${KT_ROOT}" submodule status > "${RUN_ROOT}/ktransformers.submodules"
  git -C "${REPO_ROOT}" rev-parse HEAD > "${RUN_ROOT}/uvm_repo.commit"
  "${KT_PY}" -m pip freeze > "${RUN_ROOT}/kt_pip_freeze.txt"
  lscpu > "${RUN_ROOT}/lscpu.txt"
  nvidia-smi -q > "${RUN_ROOT}/nvidia_smi_q_before.txt"
  sha256sum "${FIXTURE}" > "${RUN_ROOT}/fixture.sha256"
}

prepare_plan() {
  local plan_file="$1"
  local stats_dir="${RUN_ROOT}/kt_stats"
  require_file "${plan_file}"
  mkdir -p "${stats_dir}" "${RUN_ROOT}/input_plan"
  cp -- "${plan_file}" "${RUN_ROOT}/input_plan/$(basename "${plan_file}")"
  plan_file="${RUN_ROOT}/input_plan/$(basename "${plan_file}")"
  PATH="${KT_ROOT}/.venv/bin:${PATH}" \
    "${KT_PY}" "${CONVERTER}" "${plan_file}" --output-dir "${stats_dir}" \
      > "${RUN_ROOT}/convert_plan.log"

  readarray -t PLAN_FIELDS < <("${KT_PY}" - "${stats_dir}/manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
assert len(manifest["plans"]) == 1
entry = manifest["plans"][0]
print(entry["kt_gpu_experts_ratio_cli"])
print(entry["logical_count_path"])
PY
  )
  KT_RATIO="${PLAN_FIELDS[0]}"
  KT_STATS="${PLAN_FIELDS[1]}"

  PATH="${KT_ROOT}/.venv/bin:${PATH}" \
    "${KT_PY}" - "${KT_STATS}" "${KT_RATIO}" <<'PY'
import sys
import torch
from kt_kernel import generate_gpu_experts_masks

path, ratio = sys.argv[1], float(sys.argv[2])
logical_count = torch.load(path, map_location="cpu", weights_only=True)["logical_count"]
expected = logical_count.sum(0).bool()
count = int(expected.numel() * ratio)
actual = generate_gpu_experts_masks(logical_count.sum(0).float(), count)
assert torch.equal(actual, expected)
print(f"actual KT top-k verified: shape={tuple(expected.shape)} hot={count}")
PY
}

verify_runtime_mask_counts() {
  local manifest="$1"
  local server_log="$2"
  "${KT_PY}" - "${manifest}" "${server_log}" <<'PY'
import json
import re
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
expected = manifest["plans"][0]["hot_experts_per_layer"]
text = open(sys.argv[2], encoding="utf-8", errors="replace").read()
pairs = re.findall(r"KT GPU experts: layer (\d+) \(MoE\) has (\d+) GPU experts", text)
observed = {int(layer): int(count) for layer, count in pairs}
assert len(observed) == 48, f"expected 48 per-layer mask log entries, got {len(observed)}"
actual = [observed[layer] for layer in range(48)]
assert actual == expected, f"runtime per-layer counts differ: expected={expected} actual={actual}"
assert sum(actual) == manifest["plans"][0]["hot_experts"]
print(f"runtime mask-count log verified: layers=48 hot={sum(actual)}")
PY
}

wait_for_server() {
  local elapsed
  for elapsed in $(seq 1 "${SERVER_START_TIMEOUT}"); do
    if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
      echo "server ready after ${elapsed}s"
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      tail -n 240 "${RUN_ROOT}/server.log" >&2 || true
      die "KTransformers server exited during startup"
    fi
    sleep 1
  done
  tail -n 240 "${RUN_ROOT}/server.log" >&2 || true
  die "KTransformers server did not become ready within ${SERVER_START_TIMEOUT}s"
}

start_monitors() {
  nvidia-smi \
    --query-gpu=timestamp,index,memory.used,utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu,pcie.link.gen.current,pcie.link.width.current \
    --format=csv,noheader,nounits -lms 200 \
    > "${RUN_ROOT}/phase_nvml.csv" 2> "${RUN_ROOT}/phase_nvml.err" &
  MONITOR_PIDS+=("$!")

  nvidia-smi dmon -i 0 -s t -d 1 -o DT \
    > "${RUN_ROOT}/phase_pcie_dmon.txt" 2>&1 &
  MONITOR_PIDS+=("$!")

  (
    while kill -0 "${SERVER_PID}" 2>/dev/null; do
      date --iso-8601=ns
      ps -s "${SERVER_PID}" -o pid=,ppid=,rss=,vsz=,pcpu=,stat=,comm=
      sleep 0.2
    done
  ) > "${RUN_ROOT}/phase_process_tree.txt" 2>&1 &
  MONITOR_PIDS+=("$!")

  if [[ -n "${PCM_MEMORY_BIN:-}" ]]; then
    "${PCM_MEMORY_BIN}" 1 -csv="${RUN_ROOT}/phase_pcm_memory.csv" \
      > "${RUN_ROOT}/phase_pcm_memory.log" 2>&1 &
    MONITOR_PIDS+=("$!")
  fi
}

run_benchmark_client() {
  local output="$1"
  local concurrencies="$2"
  local output_tokens="$3"
  PATH="${KT_ROOT}/.venv/bin:${PATH}" \
    "${KT_PY}" "${BENCHMARK}" \
      --fixture "${FIXTURE}" \
      --output "${output}" \
      --base-url "${BASE_URL}" \
      --concurrencies "${concurrencies}" \
      --output-tokens "${output_tokens}" \
      --request-timeout "${REQUEST_TIMEOUT}"
}

run_kt_server_case() {
  local mode="$1"
  local plan_file="${2:-}"
  local run_warmup="$3"
  local -a placement_args=()

  [[ ! -e "${RUN_ROOT}/server.log" && ! -e "${RUN_ROOT}/benchmark.json" ]] \
    || die "RUN_ROOT already contains a server run: ${RUN_ROOT}"

  preflight_kt
  prepare_fixture
  record_metadata

  if [[ "${mode}" == "plan" ]]; then
    prepare_plan "${plan_file}"
    placement_args=(
      --kt-gpu-experts-ratio "${KT_RATIO}"
      --kt-expert-placement-strategy frequency
      --init-expert-location "${KT_STATS}"
    )
  elif [[ "${mode}" == "all_cpu" ]]; then
    placement_args=(
      --kt-num-gpu-experts 0
      --kt-expert-placement-strategy uniform
    )
  else
    die "unknown server mode: ${mode}"
  fi

  local -a server_command=(
    taskset -c "${CPUSET}"
    "${KT_PY}" -m sglang.launch_server
    --model-path "${MODEL}"
    --host 127.0.0.1
    --port "${PORT}"
    --tp-size 1
    --context-length "${MAX_CONTEXT_LENGTH}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --max-total-tokens "${KV_TOKENS}"
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
    --max-prefill-tokens "${MAX_PREFILL_TOKENS}"
    --schedule-policy fcfs
    --stream-interval 1
    --random-seed 0
    --kt-weight-path "${MODEL}"
    --kt-method FP8
    --kt-cpuinfer "${KT_CPU_THREADS}"
    --kt-threadpool-count 1
    --kt-numa-nodes 0
    --kt-max-deferred-experts-per-token 0
    --kt-gpu-prefill-token-threshold 0
    --attention-backend flashinfer
    --fp8-gemm-backend triton
    --disable-shared-experts-fusion
    --disable-radix-cache
    --disable-cuda-graph
    --disable-overlap-schedule
    "${placement_args[@]}"
  )

  printf '%q ' "${server_command[@]}" > "${RUN_ROOT}/server_command.txt"
  printf '\n' >> "${RUN_ROOT}/server_command.txt"
  {
    echo "PATH=${KT_ROOT}/.venv/bin:${PATH}"
    echo "HF_HUB_OFFLINE=1"
    echo "TRANSFORMERS_OFFLINE=1"
    echo "TOKENIZERS_PARALLELISM=false"
  } > "${RUN_ROOT}/server_environment.txt"

  wait_for_clean_gpu
  setsid env \
    PATH="${KT_ROOT}/.venv/bin:${PATH}" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    "${server_command[@]}" \
      > "${RUN_ROOT}/server.log" 2>&1 &
  SERVER_PID="$!"
  echo "${SERVER_PID}" > "${RUN_ROOT}/server.pid"
  wait_for_server

  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits > "${RUN_ROOT}/ready_compute_processes.csv"
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader,nounits > "${RUN_ROOT}/ready_gpu.csv"
  nvidia-smi --query-gpu=timestamp,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max \
    --format=csv,noheader,nounits > "${RUN_ROOT}/ready_pcie_link.csv"
  taskset -pc "${SERVER_PID}" > "${RUN_ROOT}/server_affinity.txt"

  if [[ "${mode}" == "plan" ]]; then
    verify_runtime_mask_counts "${RUN_ROOT}/kt_stats/manifest.json" "${RUN_ROOT}/server.log"
  fi

  if [[ "${run_warmup}" == "1" ]]; then
    run_benchmark_client "${RUN_ROOT}/warmup.json" "${WARM_CONCURRENCY}" "${WARM_OUTPUT_TOKENS}" \
      > "${RUN_ROOT}/warmup.log" 2>&1
  fi

  start_monitors
  set +e
  run_benchmark_client "${RUN_ROOT}/benchmark.json" "${CONCURRENCIES}" "${OUTPUT_TOKENS}" \
    > "${RUN_ROOT}/benchmark.log" 2>&1
  local benchmark_rc=$?
  set -e
  stop_monitors
  echo "${benchmark_rc}" > "${RUN_ROOT}/benchmark.returncode"
  if [[ "${benchmark_rc}" -ne 0 ]]; then
    tail -n 200 "${RUN_ROOT}/benchmark.log" >&2 || true
    return "${benchmark_rc}"
  fi

  stop_server
  wait_for_clean_gpu
  nvidia-smi --query-gpu=timestamp,memory.used \
    --format=csv,noheader,nounits > "${RUN_ROOT}/gpu_after.csv"
  echo "completed: ${RUN_ROOT}"
}

prepare_assets() {
  preflight_kt
  prepare_fixture
  record_metadata
  if [[ $# -gt 0 ]]; then
    mkdir -p "${RUN_ROOT}/kt_stats"
    PATH="${KT_ROOT}/.venv/bin:${PATH}" \
      "${KT_PY}" "${CONVERTER}" "$@" --output-dir "${RUN_ROOT}/kt_stats" \
        > "${RUN_ROOT}/convert_plans.log"
  else
    echo "fixture prepared; pass plan_g*.json paths to also build KT masks"
  fi
  echo "prepared: ${RUN_ROOT}"
}

show_help() {
  cat <<'EOF'
Usage:
  run_ktransformers_compare.sh preflight-kt
  run_ktransformers_compare.sh preflight
  run_ktransformers_compare.sh prepare [plan_g2.json ... plan_g18.json]
  run_ktransformers_compare.sh smoke
  run_ktransformers_compare.sh kt-case PLAN_JSON

Commands:
  preflight-kt  Check the pinned KT/model/GPU environment only.
  preflight     Also require the loaded nvidia_uvm module to match this worktree.
  prepare       Rebuild the fixed ShareGPT fixture and optionally exact KT masks.
  smoke         Run c1 x 8 tokens with all routed experts computed on CPU.
  kt-case       Run one exact-plan KT case; defaults to warm c32 then the full scan.

Important overrides:
  RUN_ROOT, MODEL, DATASET, PORT, KV_TOKENS, KT_CPU_THREADS, CPUSET,
  CONCURRENCIES, OUTPUT_TOKENS, WARM_CONCURRENCY, WARM_OUTPUT_TOKENS.

This runner covers the KTransformers side. The ABBA orchestrator must pair each
kt-case cold process with the existing UVM runner after `preflight` passes.
EOF
}

case "${COMMAND}" in
  preflight-kt)
    preflight_kt
    ;;
  preflight)
    preflight_full_compare
    ;;
  prepare)
    prepare_assets "$@"
    ;;
  smoke)
    CONCURRENCIES="1"
    OUTPUT_TOKENS="8"
    run_kt_server_case all_cpu "" 0
    ;;
  kt-case)
    [[ $# -eq 1 ]] || die "kt-case requires exactly one plan JSON"
    run_kt_server_case plan "$1" 1
    ;;
  help|-h|--help)
    show_help
    ;;
  *)
    show_help >&2
    die "unknown command: ${COMMAND}"
    ;;
esac
