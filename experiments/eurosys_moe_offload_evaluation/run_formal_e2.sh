#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FORMAL_SCRIPT_DIR="${SCRIPT_DIR}"
RUN_ROOT="${1:-}"
PROTOCOL_ID="eurosys-pa-v2-20260824"
MODEL="${FORMAL_MODEL:?Set FORMAL_MODEL to the local DeepSeek-V2-Lite-Chat snapshot directory}"
MODEL_KEY="${FORMAL_MODEL_KEY:-deepseek_v2_lite_bf16}"
MODEL_DTYPE="${FORMAL_MODEL_DTYPE:-bfloat16}"
VLLM_PY="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
KT_PY="${REPO_ROOT}/KTransformers/.venv/bin/python"
BLOCK_RUNNER="${SCRIPT_DIR}/formal_openai_block.py"
CLIENT="${SCRIPT_DIR}/streaming_client.py"
WORKLOADS="${FORMAL_WORKLOAD_CONFIG:-${SCRIPT_DIR}/configs/e2_matrix_cells.json}"
FIXTURE_ROOT="${FORMAL_FIXTURE_ROOT:-${SCRIPT_DIR}/inputs/generated/${PROTOCOL_ID}/deepseek_v2_lite_bf16}"
ASSET_ROOT="${FORMAL_ASSET_ROOT:-${FIXTURE_ROOT}/calibration_assets}"
MOEI_ROOT="${REPO_ROOT}/MoE-Infinity"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
MAX_BATCHED_TOKENS="${FORMAL_MAX_BATCHED_TOKENS:-16384}"
E2_ROOT=""
EXECUTION_TSV=""

die() {
  echo "error: $*" >&2
  exit 1
}

# The clean-GPU probe is intentionally retained as a state check, but it is
# not an atomic ownership primitive: two independently queued runners can see
# an empty GPU at the same instant and both launch.  Hold one host-local lock
# for the full E2 process so every tier/process block has exclusive GPU 0 and
# port ownership.  FD 8 avoids the phase-switch harness' existing FD 9 lock.
E2_GPU_LOCK_FILE="${FORMAL_E2_GPU_LOCK_FILE:-/tmp/eurosys_moe_offload_e2_gpu0.lock}"
command -v flock >/dev/null || die "flock unavailable"
exec 8>"${E2_GPU_LOCK_FILE}"
flock -n 8 || die "another E2 runner holds the GPU/port lock: ${E2_GPU_LOCK_FILE}"

contains_csv() {
  local needle="$1" haystack=",$2,"
  [[ "${haystack}" == *",${needle},"* ]]
}

[[ -n "${RUN_ROOT}" ]] || die "usage: $0 RUN_ROOT"
[[ -d "${RUN_ROOT}" && -f "${RUN_ROOT}/protocol.json" ]] \
  || die "formal run root is invalid: ${RUN_ROOT}"
[[ "$(jq -r '.protocol_id' "${RUN_ROOT}/protocol.json")" == "${PROTOCOL_ID}" ]] \
  || die "formal protocol ID differs"
[[ -x "${VLLM_PY}" && -x "${KT_PY}" ]] || die "missing Python environment"
[[ -f "${MODEL}/config.json" && -f "${WORKLOADS}" ]] || die "missing frozen input"

SYSTEM_FILTER="${FORMAL_SYSTEMS:-ours,ktransformers,moe_infinity}"
TIER_FILTER="${FORMAL_TIERS:-0g,2g,4g,6g,8g}"
REPEAT_FILTER="${FORMAL_REPEATS:-1,2,3,4,5}"
E2_ROOT="${RUN_ROOT}/cases/E2/track_b_equal_system_hbm"
mkdir -p "${E2_ROOT}"
EXECUTION_TSV="${E2_ROOT}/execution.tsv"
if [[ ! -e "${EXECUTION_TSV}" ]]; then
  printf 'created_at\tsystem\ttier\trepeat\tattempt\tcase_root\treturncode\n' \
    >"${EXECUTION_TSV}"
fi

successful_attempt_exists() {
  local repeat_root="$1" verification
  [[ -d "${repeat_root}" ]] || return 1
  while IFS= read -r verification; do
    jq -e '.status == "ok"' "${verification}" >/dev/null 2>&1 && return 0
  done < <(find "${repeat_root}" -mindepth 2 -maxdepth 2 \
    -name verification.json -type f | sort)
  return 1
}

next_attempt() {
  local repeat_root="$1" attempt=1
  while [[ -e "${repeat_root}/attempt_${attempt}" ]]; do
    attempt=$((attempt + 1))
  done
  echo "${attempt}"
}

validate_moe_infinity_build_receipt() {
  local receipt="${FORMAL_MOE_INFINITY_BUILD_RECEIPT:-}"
  local extension commit patch_sha destination
  [[ -n "${receipt}" && -f "${receipt}" ]] \
    || die "FORMAL_MOE_INFINITY_BUILD_RECEIPT must name a fresh build receipt"
  extension="$(find "${MOEI_ROOT}/moe_infinity" -maxdepth 1 -type f \
    -name '_store*.so' -print -quit)"
  [[ -n "${extension}" && -f "${extension}" ]] \
    || die "active MoE-Infinity _store extension is absent"
  commit="$(git -C "${MOEI_ROOT}" rev-parse HEAD)"
  patch_sha="$(git -C "${MOEI_ROOT}" diff --binary | sha256sum | awk '{print $1}')"
  "${VLLM_PY}" - "${receipt}" "${commit}" "${patch_sha}" "${extension}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

receipt_path = Path(sys.argv[1])
commit, patch_sha = sys.argv[2:4]
extension = Path(sys.argv[4]).resolve()
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
expected = {
    "status": "complete",
    "returncode": 0,
    "moe_infinity_commit": commit,
    "moe_infinity_patch_sha256": patch_sha,
}
for key, value in expected.items():
    if receipt.get(key) != value:
        raise SystemExit(
            f"MoE-Infinity build receipt {key}={receipt.get(key)!r}, expected {value!r}"
        )
extension_sha = hashlib.sha256(extension.read_bytes()).hexdigest()
if receipt.get("extension_sha256") != extension_sha:
    raise SystemExit("MoE-Infinity receipt does not describe the active extension")
if os.path.realpath(str(receipt.get("extension", ""))) != str(extension):
    raise SystemExit("MoE-Infinity receipt extension path differs from the active extension")
for path_key, hash_key in (
    ("build_log", "build_log_sha256"),
    ("runtime_elf_fingerprint", "runtime_elf_fingerprint_sha256"),
):
    evidence = receipt_path.parent / str(receipt.get(path_key, ""))
    if not evidence.is_file():
        raise SystemExit(f"MoE-Infinity receipt evidence is absent: {evidence}")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    if receipt.get(hash_key) != digest:
        raise SystemExit(f"MoE-Infinity receipt evidence hash mismatch: {evidence}")
runtime_path = receipt_path.parent / str(receipt["runtime_elf_fingerprint"])
runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
if runtime.get("file_sha256") != extension_sha:
    raise SystemExit("MoE-Infinity runtime fingerprint does not describe active extension")
if runtime.get("runtime_elf_sha256") != receipt.get("runtime_elf_sha256"):
    raise SystemExit("MoE-Infinity runtime ELF fingerprint mismatch")
PY
  mkdir -p "${E2_ROOT}/provenance"
  destination="${E2_ROOT}/provenance/moe_infinity_store_build_receipt.json"
  if [[ -e "${destination}" ]]; then
    cmp -s "${receipt}" "${destination}" \
      || die "E2 MoE-Infinity receipt differs from the already frozen receipt"
  else
    cp --reflink=auto "${receipt}" "${destination}"
    sha256sum "${destination}" >"${destination}.sha256"
  fi
}

run_block() {
  local system="$1" tier="$2" repeat="$3"
  local repeat_root attempt case_root port base_url env_file client_python
  local gpu_experts frequency kv_bytes kv_ratio device_ratio compilation_config
  local kt_context_length kt_mem_fraction_static kt_mem_fraction_env
  local capture_sizes_csv capture_sizes_json max_capture_size
  local -a server_command runner_command overrides

  repeat_root="${E2_ROOT}/${system}/${MODEL_KEY}/budget_${tier}/repeat_$(printf '%02d' "${repeat}")"
  if [[ "${FORMAL_FORCE_RETRY:-0}" != 1 ]] \
      && successful_attempt_exists "${repeat_root}"; then
    echo "skip verified E2 case: ${system}/${tier}/repeat_${repeat}"
    return 0
  fi
  mkdir -p "${repeat_root}"
  attempt="$(next_attempt "${repeat_root}")"
  case_root="${repeat_root}/attempt_${attempt}"
  overrides=()

  case "${system}" in
    ours)
      port=30201
      env_file="${FORMAL_OURS_ENV_FILE:-${FORMAL_SCRIPT_DIR}/configs/e2_ours_${tier}_env.json}"
      client_python="${VLLM_PY}"
      if [[ "${tier}" == 0g ]]; then
        server_command=(
          taskset -c 0-13 "${VLLM_PY}" -m vllm.entrypoints.openai.api_server
          --host 127.0.0.1 --port "${port}" --disable-frontend-multiprocessing
          --model "${MODEL}" --trust-remote-code --dtype "${MODEL_DTYPE}"
          --max-model-len 8320 --gpu-memory-utilization 0.97
          --cpu-offload-gb 0 --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
          --max-num-seqs 16 --generation-config vllm
        )
      else
        case "${tier}" in
          2g) kv_bytes="${FORMAL_OURS_KV_BYTES_2G:-2147483648}" ;;
          4g) kv_bytes="${FORMAL_OURS_KV_BYTES_4G:-4294967296}" ;;
          6g) kv_bytes="${FORMAL_OURS_KV_BYTES_6G:-6442450944}" ;;
          8g) kv_bytes="${FORMAL_OURS_KV_BYTES_8G:-8589934592}" ;;
          *) die "unsupported ours tier: ${tier}" ;;
        esac
        [[ "${kv_bytes}" =~ ^[1-9][0-9]*$ ]] \
          || die "invalid ours KV bytes for ${tier}: ${kv_bytes}"
        overrides+=(
          --server-env-override "VLLM_UVM_ORIGINAL_VA_PHASE_RESULT_FILE=${case_root}/phase_transition.json"
          --server-env-override "VLLM_CPU_OFFLOAD_MANIFEST_FILE=${case_root}/offload_manifest.jsonl"
          --server-env-override "VLLM_UVM_LOG_FILE=${case_root}/uvm.log"
        )
        if [[ -n "${FORMAL_UVM_PHASE_MODE:-}" ]]; then
          overrides+=(
            --server-env-override "VLLM_UVM_ORIGINAL_VA_PHASE_MODE=${FORMAL_UVM_PHASE_MODE}"
          )
        fi
        if [[ -n "${FORMAL_UVM_PARK_RESUME:-}" ]]; then
          overrides+=(
            --server-env-override "VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=${FORMAL_UVM_PARK_RESUME}"
          )
        fi
        if [[ "${FORMAL_CUDA_LAUNCH_BLOCKING:-0}" == 1 ]]; then
          overrides+=(--server-env-override "CUDA_LAUNCH_BLOCKING=1")
        fi
        if [[ -n "${FORMAL_UVM_HOT_CUDA_READ_GATE:-}" ]]; then
          overrides+=(
            --server-env-override "VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE=${FORMAL_UVM_HOT_CUDA_READ_GATE}"
          )
        fi
        if [[ -n "${FORMAL_UVM_PHASE_BATCHED_SCHEDULING:-}" ]]; then
          overrides+=(
            --server-env-override "VLLM_UVM_PHASE_BATCHED_SCHEDULING=${FORMAL_UVM_PHASE_BATCHED_SCHEDULING}"
          )
        fi
        if [[ -n "${FORMAL_UVM_PREFILL_LAYER_MEMORY_TRACE:-}" ]]; then
          overrides+=(
            --server-env-override "VLLM_UVM_PREFILL_LAYER_MEMORY_TRACE=${FORMAL_UVM_PREFILL_LAYER_MEMORY_TRACE}"
          )
        fi
        if [[ -n "${FORMAL_UVM_PREFILL_LAYER_MEMORY_TRACE_MIN_TOKENS:-}" ]]; then
          overrides+=(
            --server-env-override "VLLM_UVM_PREFILL_LAYER_MEMORY_TRACE_MIN_TOKENS=${FORMAL_UVM_PREFILL_LAYER_MEMORY_TRACE_MIN_TOKENS}"
          )
        fi
        if [[ -n "${FORMAL_UVM_FUSED_MOE_POST_WORKSPACE_RESIDENCY_AUDIT:-}" ]]; then
          overrides+=(
            --server-env-override "VLLM_UVM_FUSED_MOE_POST_WORKSPACE_RESIDENCY_AUDIT=${FORMAL_UVM_FUSED_MOE_POST_WORKSPACE_RESIDENCY_AUDIT}"
          )
        fi
        if [[ -n "${FORMAL_UVM_FUSED_MOE_POST_WORKSPACE_AUDIT_MIN_TOKENS:-}" ]]; then
          overrides+=(
            --server-env-override "VLLM_UVM_FUSED_MOE_POST_WORKSPACE_AUDIT_MIN_TOKENS=${FORMAL_UVM_FUSED_MOE_POST_WORKSPACE_AUDIT_MIN_TOKENS}"
          )
        fi
        if [[ -n "${FORMAL_UVM_FUSED_MOE_POST_WORKSPACE_AUDIT_CAPTURE_WARMUP:-}" ]]; then
          overrides+=(
            --server-env-override "VLLM_UVM_FUSED_MOE_POST_WORKSPACE_AUDIT_CAPTURE_WARMUP=${FORMAL_UVM_FUSED_MOE_POST_WORKSPACE_AUDIT_CAPTURE_WARMUP}"
          )
        fi
        if [[ -n "${FORMAL_UVM_FUSED_MOE_PRE_KERNEL_ALIAS_READ_GATE:-}" ]]; then
          overrides+=(
            --server-env-override "VLLM_UVM_FUSED_MOE_PRE_KERNEL_ALIAS_READ_GATE=${FORMAL_UVM_FUSED_MOE_PRE_KERNEL_ALIAS_READ_GATE}"
          )
        fi
        if [[ -n "${FORMAL_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE:-}" ]]; then
          overrides+=(
            --server-env-override "VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE=${FORMAL_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE}"
          )
        fi
        if [[ -n "${FORMAL_UVM_EXPERT_PLAN_FILE:-}" ]]; then
          [[ -f "${FORMAL_UVM_EXPERT_PLAN_FILE}" ]] \
            || die "missing formal expert plan: ${FORMAL_UVM_EXPERT_PLAN_FILE}"
          overrides+=(
            --server-env-override "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE=${FORMAL_UVM_EXPERT_PLAN_FILE}"
          )
        fi
        if [[ -n "${FORMAL_UVM_EXACT_NAMES_FILE:-}" ]]; then
          [[ -f "${FORMAL_UVM_EXACT_NAMES_FILE}" ]] \
            || die "missing formal exact-name file: ${FORMAL_UVM_EXACT_NAMES_FILE}"
          overrides+=(
            --server-env-override "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE=${FORMAL_UVM_EXACT_NAMES_FILE}"
          )
        fi
        if [[ -n "${FORMAL_ATTENTION_BACKEND:-}" ]]; then
          overrides+=(
            --server-env-override "VLLM_ATTENTION_BACKEND=${FORMAL_ATTENTION_BACKEND}"
          )
        fi
        if [[ -n "${FORMAL_UVM_PREFILL_RESERVE_WARMUP_TOKENS:-}" ]]; then
          [[ "${FORMAL_UVM_PREFILL_RESERVE_WARMUP_TOKENS}" =~ ^[1-9][0-9]*$ ]] \
            || die "formal Prefill reserve tokens must be a positive integer"
          overrides+=(
            --server-env-override "VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS=${FORMAL_UVM_PREFILL_RESERVE_WARMUP_TOKENS}"
          )
        fi
        if [[ -n "${FORMAL_UVM_NATIVE_HOLE_SENTINEL_BYTES:-}" ]]; then
          [[ "${FORMAL_UVM_NATIVE_HOLE_SENTINEL_BYTES}" =~ ^[1-9][0-9]*$ ]] \
            || die "formal native-hole sentinel bytes must be a positive integer"
          overrides+=(
            --server-env-override "VLLM_UVM_NATIVE_HOLE_SENTINEL_BYTES=${FORMAL_UVM_NATIVE_HOLE_SENTINEL_BYTES}"
            --server-env-override "VLLM_UVM_NATIVE_HOLE_SENTINEL_EMPTY_CACHE=${FORMAL_UVM_NATIVE_HOLE_SENTINEL_EMPTY_CACHE:-0}"
          )
        fi
        if [[ -n "${FORMAL_UVM_CAPTURE_NATIVE_HOLE_SENTINEL_BYTES:-}" ]]; then
          [[ "${FORMAL_UVM_CAPTURE_NATIVE_HOLE_SENTINEL_BYTES}" =~ ^[1-9][0-9]*$ ]] \
            || die "formal capture native-hole sentinel bytes must be a positive integer"
          overrides+=(
            --server-env-override "VLLM_UVM_CAPTURE_NATIVE_HOLE_SENTINEL_BYTES=${FORMAL_UVM_CAPTURE_NATIVE_HOLE_SENTINEL_BYTES}"
            --server-env-override "VLLM_UVM_CAPTURE_NATIVE_HOLE_SENTINEL_EMPTY_CACHE=${FORMAL_UVM_CAPTURE_NATIVE_HOLE_SENTINEL_EMPTY_CACHE:-0}"
          )
        fi
        if [[ -n "${FORMAL_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_BYTES:-}" ]]; then
          [[ "${FORMAL_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_BYTES}" =~ ^[1-9][0-9]*$ ]] \
            || die "formal pre-KV native-hole sentinel bytes must be a positive integer"
          overrides+=(
            --server-env-override "VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_BYTES=${FORMAL_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_BYTES}"
            --server-env-override "VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_EMPTY_CACHE=${FORMAL_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_EMPTY_CACHE:-0}"
          )
        fi
        if [[ -n "${FORMAL_FUSED_MOE_CHUNK_SIZE:-}" ]]; then
          [[ "${FORMAL_FUSED_MOE_CHUNK_SIZE}" =~ ^[1-9][0-9]*$ ]] \
            || die "formal fused-MoE chunk size must be a positive integer"
          overrides+=(
            --server-env-override "VLLM_FUSED_MOE_CHUNK_SIZE=${FORMAL_FUSED_MOE_CHUNK_SIZE}"
          )
        fi
        # The capture-size list is the union of uniform Decode sizes and pure
        # Prefill token shapes. Dual-topology capture partitions it at
        # max_num_seqs: <=16 becomes remote Decode, >16 becomes sealed Prefill.
        # Formal pure-stage runs must include every intended Prefill shape so
        # the dispatcher cannot silently fall back to eager execution.
        capture_sizes_csv="${FORMAL_CUDAGRAPH_CAPTURE_SIZES:-1,4,8,16,128,512,1024,2048,4096,8192,16384}"
        [[ "${capture_sizes_csv}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] \
          || die "invalid formal CUDA Graph capture sizes: ${capture_sizes_csv}"
        capture_sizes_json="$(jq -cn --arg value "${capture_sizes_csv}" \
          '$value | split(",") | map(tonumber) | unique | sort')"
        max_capture_size="$(jq -r 'max' <<<"${capture_sizes_json}")"
        compilation_config="$(jq -cn \
          --argjson sizes "${capture_sizes_json}" \
          --argjson maximum "${max_capture_size}" \
          '{mode:"NONE",cudagraph_mode:"FULL_SEPARATE",cudagraph_capture_sizes:$sizes,max_cudagraph_capture_size:$maximum}')"
        if [[ "${FORMAL_CUDAGRAPH_MODE:-}" == NONE ]]; then
          compilation_config='{"mode":"NONE","cudagraph_mode":"NONE"}'
        fi
        server_command=(
          taskset -c 0-13 "${VLLM_PY}" -m vllm.entrypoints.openai.api_server
          --host 127.0.0.1 --port "${port}" --disable-frontend-multiprocessing
          --model "${MODEL}" --trust-remote-code --dtype "${MODEL_DTYPE}"
          --max-model-len 8320 --gpu-memory-utilization 0.97
          --kv-cache-memory-bytes "${kv_bytes}" --cpu-offload-gb 30
          --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" --max-num-seqs 16
          --no-enable-prefix-caching
          --generation-config vllm
          --compilation-config "${compilation_config}"
        )
      fi
      ;;
    ktransformers)
      port=30202
      env_file="${FORMAL_SCRIPT_DIR}/configs/e2_ktransformers_env.json"
      client_python="${KT_PY}"
      # SGLang reserves two internal token slots, so 8322 provides the same
      # effective 8320-token request envelope used by the other E2 systems.
      kt_context_length="${FORMAL_KT_CONTEXT_LENGTH:-8322}"
      [[ "${kt_context_length}" =~ ^[1-9][0-9]*$ ]] \
        || die "invalid KTransformers context length: ${kt_context_length}"
      # Leave enough non-static HBM for the largest 16,384-token Prefill
      # activation workspace.  The earlier 0.97 default exhausted HBM during
      # ordinary/RAG concurrency even after KV-pool profiling was corrected.
      # Allow calibration to freeze a different safe allocator fraction per
      # nominal tier while preserving the documented odd/even tier ordering
      # in one runner invocation.  The global override remains the fallback
      # for backwards compatibility.
      kt_mem_fraction_env="FORMAL_KT_MEM_FRACTION_STATIC_${tier^^}"
      kt_mem_fraction_static="${!kt_mem_fraction_env:-${FORMAL_KT_MEM_FRACTION_STATIC:-0.93}}"
      [[ "${kt_mem_fraction_static}" =~ ^0\.[0-9]+$ ]] \
        || die "invalid KTransformers static-memory fraction: ${kt_mem_fraction_static}"
      case "${tier}" in
        0g) gpu_experts=64; frequency="" ;;
        2g) gpu_experts=59; frequency="${ASSET_ROOT}/kt_placements/kt_frequency_2g.pt" ;;
        4g) gpu_experts=54; frequency="${ASSET_ROOT}/kt_placements/kt_frequency_4g.pt" ;;
        6g) gpu_experts=50; frequency="${ASSET_ROOT}/kt_placements/kt_frequency_6g.pt" ;;
        8g) gpu_experts=45; frequency="${ASSET_ROOT}/kt_placements/kt_frequency_8g.pt" ;;
        *) die "unsupported KTransformers tier: ${tier}" ;;
      esac
      server_command=(
        taskset -c 0-13 "${KT_PY}" -m sglang.launch_server
        --model-path "${MODEL}" --host 127.0.0.1 --port "${port}"
        --tp-size 1 --context-length "${kt_context_length}" --max-running-requests 16
        --mem-fraction-static "${kt_mem_fraction_static}" --chunked-prefill-size "${MAX_BATCHED_TOKENS}"
        --max-prefill-tokens "${MAX_BATCHED_TOKENS}" --schedule-policy fcfs --stream-interval 1
        --random-seed 20260823 --trust-remote-code --kt-weight-path "${MODEL}"
        --kt-method BF16 --kt-cpuinfer 14 --kt-threadpool-count 1
        --kt-numa-nodes 0 --kt-num-gpu-experts "${gpu_experts}"
        --kt-expert-placement-strategy "$([[ -n "${frequency}" ]] && echo frequency || echo uniform)"
        --kt-max-deferred-experts-per-token 0 --kt-gpu-prefill-token-threshold 0
        --attention-backend flashinfer --cuda-graph-max-bs 16
        --cuda-graph-bs 1 4 8 16 --enable-mixed-chunk
        --disable-shared-experts-fusion --disable-radix-cache --enable-metrics
      )
      if [[ -n "${frequency}" ]]; then
        server_command+=(--init-expert-location "${frequency}")
      fi
      ;;
    moe_infinity)
      port=30203
      env_file="${FORMAL_SCRIPT_DIR}/configs/e2_moe_infinity_env.json"
      client_python="${VLLM_PY}"
      case "${tier}" in
        0g) device_ratio=0.75; kv_ratio=0.01 ;;
        2g) device_ratio=0.68; kv_ratio=0.07 ;;
        4g) device_ratio=0.61; kv_ratio=0.14 ;;
        6g) device_ratio=0.54; kv_ratio=0.21 ;;
        8g) device_ratio=0.47; kv_ratio=0.28 ;;
        *) die "unsupported MoE-Infinity tier: ${tier}" ;;
      esac
      server_command=(
        taskset -c 0-13 "${VLLM_PY}" "${FORMAL_SCRIPT_DIR}/launch_moe_infinity_server.py"
        --model "${MODEL}"
        --offload-path "${FORMAL_MOEI_OFFLOAD_PATH:?Set FORMAL_MOEI_OFFLOAD_PATH for the MoE-Infinity offload directory}"
        --host 127.0.0.1 --port "${port}" --device-memory-ratio "${device_ratio}"
        --kv-cache-ratio "${kv_ratio}" --max-batch-size 16 --num-threads 1
      )
      ;;
    *)
      die "unknown E2 system: ${system}"
      ;;
  esac

  base_url="http://127.0.0.1:${port}"
  runner_command=(
    "${VLLM_PY}" "${BLOCK_RUNNER}"
    --case-root "${case_root}" --protocol-id "${PROTOCOL_ID}"
    --system "${system}" --model "${MODEL}" --base-url "${base_url}"
    --workload-config "${WORKLOADS}" --fixture-root "${FIXTURE_ROOT}"
    --server-env "${env_file}"
    "${overrides[@]}"
    --python "${client_python}" --streaming-client "${CLIENT}"
    --ready-timeout 1200 --request-timeout 7200 --gpu-clean-timeout 240
  )
  if [[ "${FORMAL_STOP_ON_FAILURE:-0}" == 1 ]]; then
    runner_command+=(--stop-on-first-workload-failure)
  fi
  if [[ "${FORMAL_STOP_AFTER_READY:-0}" == 1 ]]; then
    runner_command+=(--stop-after-ready)
  fi
  if [[ -n "${FORMAL_WORKLOAD_FILTER:-}" ]]; then
    runner_command+=(--workload-filter "${FORMAL_WORKLOAD_FILTER}")
  fi
  runner_command+=(-- "${server_command[@]}")

  set +e
  "${runner_command[@]}" >"${case_root}.runner.log" 2>&1
  local rc=$?
  set -e
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${system}" "${tier}" "${repeat}" \
    "${attempt}" "${case_root}" "${rc}" >>"${EXECUTION_TSV}"
  return "${rc}"
}

run_selected_system() {
  local system="$1" tier repeat failures=0
  IFS=',' read -r -a tiers <<<"${TIER_FILTER}"
  IFS=',' read -r -a repeats <<<"${REPEAT_FILTER}"
  for repeat in "${repeats[@]}"; do
    [[ "${repeat}" =~ ^[1-5]$ ]] || die "invalid repeat: ${repeat}"
    if (( repeat % 2 == 1 )); then
      ordered_tiers=("${tiers[@]}")
    else
      ordered_tiers=()
      for ((index=${#tiers[@]} - 1; index >= 0; --index)); do
        ordered_tiers+=("${tiers[index]}")
      done
    fi
    for tier in "${ordered_tiers[@]}"; do
      if ! run_block "${system}" "${tier}" "${repeat}"; then
        failures=$((failures + 1))
        if [[ "${FORMAL_STOP_ON_FAILURE:-0}" == 1 ]]; then
          return 1
        fi
      fi
    done
  done
  return "${failures}"
}

total_failures=0
if contains_csv moe_infinity "${SYSTEM_FILTER}"; then
  validate_moe_infinity_build_receipt
fi
if contains_csv ours "${SYSTEM_FILTER}"; then
  export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
  export ALLOW_GPU_RUN=1
  export SLOT_BUDGET_BYTES="${SLOT_BUDGET_BYTES:-1207959552}"
  # shellcheck source=/dev/null
  source "${BASE_HARNESS}"
  RUN_ROOT_ACTIVE="${E2_ROOT}"
  runtime_preflight
  module_snapshot "${E2_ROOT}/module.before.json" "${STOCK_UVM}"
  load_custom_module
  module_snapshot "${E2_ROOT}/module.custom.json" "${CUSTOM_UVM}"
  ours_rc=0
  run_selected_system ours || ours_rc=$?
  total_failures=$((total_failures + ours_rc))
  restore_stock || die "failed to restore stock UVM after E2 ours"
  module_snapshot "${E2_ROOT}/module.after_ours.json" "${STOCK_UVM}"
  restart_exporters
  if (( ours_rc != 0 )) && [[ "${FORMAL_STOP_ON_FAILURE:-0}" == 1 ]]; then
    echo "E2_ROOT=${E2_ROOT}"
    echo "E2_FAILURES=${total_failures}"
    exit "${ours_rc}"
  fi
fi

for system in ktransformers moe_infinity; do
  if contains_csv "${system}" "${SYSTEM_FILTER}"; then
    system_rc=0
    run_selected_system "${system}" || system_rc=$?
    total_failures=$((total_failures + system_rc))
    if (( system_rc != 0 )) && [[ "${FORMAL_STOP_ON_FAILURE:-0}" == 1 ]]; then
      echo "E2_ROOT=${E2_ROOT}"
      echo "E2_FAILURES=${total_failures}"
      exit "${system_rc}"
    fi
  fi
done

echo "E2_ROOT=${E2_ROOT}"
echo "E2_FAILURES=${total_failures}"
((total_failures == 0))
