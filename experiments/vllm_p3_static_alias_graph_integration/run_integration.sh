#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_PY:-${VLLM_ROOT}/.venv/bin/python}"
RUNNER="${SCRIPT_DIR}/benchmark_full_graph_steps.py"
SMOKE="${SCRIPT_DIR}/smoke_static_alias_graph.py"
VERIFY="${SCRIPT_DIR}/verify_integration.py"
PREPARE="${REPO_ROOT}/experiments/vllm_slot_vs_pinned_copy/prepare_inputs.py"
DEFAULT_MODEL="${REPO_ROOT}/../.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782"
DEFAULT_FIXTURE="${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json"

LAYER_BYTES=603979776
LOGICAL_BYTES=2415919104
DUAL_BANK_BYTES=1207959552

die() { echo "error: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }

gpu_clean() {
  local used pids
  used="$(nvidia-smi -i "${GPU_INDEX:-0}" --query-gpu=memory.used \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1}')"
  [[ "${used}" == "0" && -z "${pids}" ]]
}

wait_gpu_clean() {
  local attempt
  for ((attempt = 0; attempt < ${GPU_CLEAN_TIMEOUT_SEC:-240}; ++attempt)); do
    gpu_clean && return 0
    sleep 1
  done
  return 1
}

require_stock_uvm() {
  local loaded installed
  [[ -r /sys/module/nvidia_uvm/srcversion ]] \
    || die "nvidia_uvm is not loaded or exposes no srcversion"
  loaded="$(tr -d '[:space:]' < /sys/module/nvidia_uvm/srcversion)"
  installed="$(modinfo -F srcversion nvidia-uvm | tr -d '[:space:]')"
  [[ -n "${installed}" && "${loaded}" == "${installed}" ]] \
    || die "loaded nvidia_uvm ${loaded} differs from installed stock ${installed}"
}

snapshot_gpu() {
  {
    nvidia-smi -i "${GPU_INDEX:-0}" \
      --query-gpu=index,name,uuid,driver_version,pstate,clocks.current.sm,clocks.current.memory,power.draw,memory.used,memory.free,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>&1 || true
  } > "$1"
}

static_preflight() {
  require_file "${RUNNER}"
  require_file "${SMOKE}"
  require_file "${VERIFY}"
  require_file "${PREPARE}"
  [[ -x "${VLLM_PY}" ]] || die "missing vLLM Python: ${VLLM_PY}"
  "${VLLM_PY}" -m py_compile "${RUNNER}" "${SMOKE}" "${VERIFY}"
  "${VLLM_PY}" - <<'PY'
from vllm.device_allocator.cumem import StaticAliasAllocation
assert callable(StaticAliasAllocation.create)
PY
  "${VLLM_PY}" -m pytest -q "${SCRIPT_DIR}/test_integration.py"
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" -m pytest -q tests/test_cpu_offload_exact_static.py \
      -k graph_static_alias
  )
  echo "static preflight: OK"
}

runtime_preflight() {
  static_preflight
  require_stock_uvm
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  command -v taskset >/dev/null || die "taskset is unavailable"
  command -v timeout >/dev/null || die "timeout is unavailable"
  command -v flock >/dev/null || die "flock is unavailable"
  gpu_clean || die "GPU must start at exactly 0 MiB with no compute client"
}

run_smoke() {
  [[ "${ALLOW_GPU_RUN:-0}" == "1" ]] || die "smoke requires ALLOW_GPU_RUN=1"
  runtime_preflight
  local output="${1:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_allocator_smoke.json}"
  [[ ! -e "${output}" ]] || die "output already exists: ${output}"
  mkdir -p "$(dirname "${output}")"
  local manifest="${output}.manifest.jsonl"
  local command_file="${output}.command.txt"
  local gpu_before="${output}.gpu.before.csv"
  local gpu_after="${output}.gpu.after.csv"
  snapshot_gpu "${gpu_before}"
  {
    printf 'cd %q && env CUDA_VISIBLE_DEVICES=%q taskset -c %q %q %q ' \
      "${VLLM_ROOT}" "${GPU_INDEX:-0}" "${CPUSET:-0-13}" \
      "${VLLM_PY}" "${SMOKE}"
    printf '%q ' --output "${output}" --manifest "${manifest}" \
      --layer-mib "${SMOKE_LAYER_MIB:-8}" \
      --warmup-replays "${SMOKE_WARMUPS:-3}" \
      --measured-replays "${SMOKE_REPLAYS:-20}"
    printf '\n'
  } > "${command_file}"
  exec 9>"/tmp/vllm_p3_static_alias_graph_integration.lock"
  flock -n 9 || die "another static-alias integration run holds the lock"
  local rc
  set +e
  (
    cd "${VLLM_ROOT}"
    env CUDA_VISIBLE_DEVICES="${GPU_INDEX:-0}" \
      taskset -c "${CPUSET:-0-13}" \
      "${VLLM_PY}" "${SMOKE}" \
      --output "${output}" \
      --manifest "${manifest}" \
      --layer-mib "${SMOKE_LAYER_MIB:-8}" \
      --warmup-replays "${SMOKE_WARMUPS:-3}" \
      --measured-replays "${SMOKE_REPLAYS:-20}"
  )
  rc=$?
  set -e
  wait_gpu_clean || die "GPU did not return to 0 MiB after allocator smoke"
  snapshot_gpu "${gpu_after}"
  echo "output=${output}"
  echo "manifest=${manifest}"
  echo "command=${command_file}"
  (( rc == 0 )) || die "allocator smoke failed with return code ${rc}"
}

arm_name() {
  case "$1" in
    E) echo eager_pinned ;;
    A) echo graph_static_alias ;;
    D) echo graph_direct_static_bank ;;
    *) die "unknown arm symbol $1" ;;
  esac
}

validate_order() {
  local order="$1" required="$2" block_len index block symbol
  block_len="${#required}"
  (( block_len > 0 && ${#order} % block_len == 0 )) \
    || die "ORDER must contain complete ${required} permutation blocks"
  [[ "${order}" =~ ^[EAD]+$ ]] || die "ORDER contains an unknown symbol"
  for ((index = 0; index < ${#order}; index += block_len)); do
    block="${order:index:block_len}"
    for ((symbol = 0; symbol < block_len; ++symbol)); do
      [[ "${block}" == *"${required:symbol:1}"* ]] \
        || die "ORDER block ${block} is not a ${required} permutation"
    done
  done
}

reverse_csv() {
  "${VLLM_PY}" - "$1" <<'PY'
import sys
print(",".join(reversed([x.strip() for x in sys.argv[1].split(",") if x.strip()])))
PY
}

run_arm() {
  local symbol="$1" ordinal="$2" block="$3" arm_dir="$4" names="$5"
  local batch_order="$6" copy_trace="$7"
  local arm execution_mode profile mechanism
  arm="$(arm_name "${symbol}")"
  mkdir "${arm_dir}"
  : > "${arm_dir}/cpu_offload_manifest.jsonl"
  wait_gpu_clean || die "GPU is not clean before ${arm}"
  snapshot_gpu "${arm_dir}/gpu.before.csv"

  case "${symbol}" in
    E)
      execution_mode=eager
      profile=qwen_l0_l3_eager_pinned
      mechanism=vllm_pinned_copy_exact_l0_l3
      ;;
    A)
      execution_mode=full_graph_decode
      profile=qwen_l0_l3_full_graph_static_alias
      mechanism=vllm_full_decode_graph_dual_bank_static_alias_l0_l3
      ;;
    D)
      execution_mode=full_graph_decode
      profile=qwen_l0_l3_full_graph_direct_static_bank
      mechanism=vllm_full_decode_graph_direct_static_bank_l0_l3
      ;;
  esac

  local static_bank=0 graph_static_alias=0
  [[ "${symbol}" == D ]] && static_bank=1
  [[ "${symbol}" == A ]] && graph_static_alias=1
  local -a common_env=(
    VLLM_USE_UVM=0
    VLLM_ENABLE_V1_MULTIPROCESSING=0
    VLLM_CPU_OFFLOAD_BACKEND=copy
    VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${names}"
    VLLM_CPU_OFFLOAD_STATIC_BANK="${static_bank}"
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS="${graph_static_alias}"
    VLLM_CPU_OFFLOAD_COPY_TRACE="${copy_trace}"
    VLLM_CPU_OFFLOAD_COPY_SYNC=0
    VLLM_CPU_OFFLOAD_MANIFEST_FILE="${arm_dir}/cpu_offload_manifest.jsonl"
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    TOKENIZERS_PARALLELISM=false
    OMP_NUM_THREADS="${CPU_THREADS:-14}"
    MKL_NUM_THREADS="${CPU_THREADS:-14}"
    CUDA_VISIBLE_DEVICES="${GPU_INDEX:-0}"
  )
  printf '%s\n' "${common_env[@]}" > "${arm_dir}/environment.txt"

  local model="${MODEL:-${DEFAULT_MODEL}}" fixture="${EVAL_FIXTURE:-${DEFAULT_FIXTURE}}"
  local -a args=(
    --model "${model}"
    --fixture "${fixture}"
    --output "${arm_dir}/result.json"
    --offload-manifest "${arm_dir}/cpu_offload_manifest.jsonl"
    --profile-label "${profile}"
    --mechanism-class "${mechanism}"
    --execution-mode "${execution_mode}"
    --batch-sizes "${batch_order}"
    --cudagraph-capture-sizes 1,4
    --conditioning-repeats "${CONDITIONING_REPEATS}"
    --measurement-repeats "${MEASUREMENT_REPEATS}"
    --warmup-decode-steps "${WARMUP_DECODE_STEPS}"
    --measured-decode-steps "${MEASURED_DECODE_STEPS}"
    --kv-cache-memory-bytes "${KV_CACHE_BYTES:-268435456}"
    --cpu-offload-gb 3.0
    --max-model-len "${MAX_MODEL_LEN:-512}"
    --max-num-seqs 4
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.97}"
    --seed 0
  )
  {
    printf 'cd %q && env ' "${VLLM_ROOT}"
    printf '%q ' "${common_env[@]}"
    printf 'timeout --kill-after=30s %qs taskset -c %q %q %q ' \
      "${TIMEOUT_SEC:-3600}" "${CPUSET:-0-13}" "${VLLM_PY}" "${RUNNER}"
    printf '%q ' "${args[@]}"
    printf '\n'
  } > "${arm_dir}/command.txt"

  local rc
  set +e
  (
    cd "${VLLM_ROOT}"
    env "${common_env[@]}" \
      /usr/bin/time -v -o "${arm_dir}/resource_time.txt" \
      timeout --kill-after=30s "${TIMEOUT_SEC:-3600}s" \
      taskset -c "${CPUSET:-0-13}" \
      "${VLLM_PY}" "${RUNNER}" "${args[@]}"
  ) > "${arm_dir}/run.log" 2>&1
  rc=$?
  set -e
  echo "${rc}" > "${arm_dir}/returncode.txt"
  wait_gpu_clean || die "GPU did not return to 0 MiB after ${arm}"
  snapshot_gpu "${arm_dir}/gpu.after.csv"
  return "${rc}"
}

run_qwen() {
  local mode="$1"
  [[ "${ALLOW_GPU_RUN:-0}" == "1" ]] || die "${mode} requires ALLOW_GPU_RUN=1"
  runtime_preflight
  local model="${MODEL:-${DEFAULT_MODEL}}" fixture="${EVAL_FIXTURE:-${DEFAULT_FIXTURE}}"
  require_file "${model}/config.json"
  require_file "${fixture}"

  local default_order default_required default_copy_trace
  if [[ "${mode}" == qualification ]]; then
    default_order=EA
    default_required=EA
    default_copy_trace=1
    CONDITIONING_REPEATS="${CONDITIONING_REPEATS:-1}"
    MEASUREMENT_REPEATS="${MEASUREMENT_REPEATS:-1}"
    WARMUP_DECODE_STEPS="${WARMUP_DECODE_STEPS:-2}"
    MEASURED_DECODE_STEPS="${MEASURED_DECODE_STEPS:-8}"
  else
    default_order=EAAEEA
    default_required=EA
    default_copy_trace=0
    CONDITIONING_REPEATS="${CONDITIONING_REPEATS:-1}"
    MEASUREMENT_REPEATS="${MEASUREMENT_REPEATS:-3}"
    WARMUP_DECODE_STEPS="${WARMUP_DECODE_STEPS:-5}"
    MEASURED_DECODE_STEPS="${MEASURED_DECODE_STEPS:-16}"
  fi
  local order="${ORDER:-${default_order}}" required="${REQUIRED_ARMS:-${default_required}}"
  local copy_trace="${COPY_TRACE:-${default_copy_trace}}"
  validate_order "${order}" "${required}"
  [[ "${copy_trace}" == 0 || "${copy_trace}" == 1 ]] \
    || die "COPY_TRACE must be 0 or 1"

  local run_root="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_${mode}}"
  [[ ! -e "${run_root}" ]] || die "RUN_ROOT already exists: ${run_root}"
  mkdir -p "${run_root}/inputs"
  run_root="$(cd "${run_root}" && pwd)"
  "${VLLM_PY}" "${PREPARE}" --output-dir "${run_root}/inputs" \
    > "${run_root}/prepare_inputs.stdout.json"
  local names="${run_root}/inputs/exact_parameter_names.txt"
  require_file "${names}"
  [[ "$(wc -l < "${names}")" -eq 8 ]] \
    || die "expected eight exact Qwen parameter names"

  sha256sum "${model}/config.json" "${fixture}" "${names}" "${RUNNER}" \
    "${SMOKE}" "${VERIFY}" > "${run_root}/input_sha256.txt"
  git -C "${REPO_ROOT}" status --short > "${run_root}/git_status.txt"
  snapshot_gpu "${run_root}/gpu.initial.csv"
  "${VLLM_PY}" - "${run_root}/run_config.json" "${mode}" "${order}" \
    "${required}" "${copy_trace}" "${model}" "${fixture}" \
    "${CONDITIONING_REPEATS}" "${MEASUREMENT_REPEATS}" \
    "${WARMUP_DECODE_STEPS}" "${MEASURED_DECODE_STEPS}" <<'PY'
import json, sys
(path, mode, order, required, trace, model, fixture, conditioning,
 measurement, warmup, measured) = sys.argv[1:]
payload = {
    "schema": "vllm_p3_static_alias_graph_run_v1",
    "mode": mode,
    "order": order,
    "required_arms": required,
    "copy_trace": bool(int(trace)),
    "model": model,
    "fixture": fixture,
    "batch_sizes": [1, 4],
    "cudagraph_capture_sizes": [1, 4],
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "conditioning_repeats": int(conditioning),
    "measurement_repeats": int(measurement),
    "warmup_decode_steps": int(warmup),
    "measured_decode_steps": int(measured),
    "layers": [0, 1, 2, 3],
    "parameter_count": 8,
    "bytes_per_layer": 603979776,
    "logical_bytes_per_forward": 2415919104,
    "dual_bank_backing_bytes": 1207959552,
    "stock_uvm_required": True,
    "module_cycle": False,
}
open(path, "w", encoding="utf-8").write(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY

  printf 'ordinal\tblock\tsymbol\tarm\tbatch_order\tdirectory\treturncode\n' \
    > "${run_root}/execution_status.tsv"
  exec 9>"/tmp/vllm_p3_static_alias_graph_integration.lock"
  flock -n 9 || die "another static-alias integration run holds the lock"
  local failures=0 index ordinal block symbol arm arm_dir batch_order rc
  local block_len="${#required}"
  for ((index = 0; index < ${#order}; ++index)); do
    ordinal=$((index + 1))
    block=$((index / block_len + 1))
    symbol="${order:index:1}"
    arm="$(arm_name "${symbol}")"
    batch_order=1,4
    (( block % 2 == 0 )) && batch_order="$(reverse_csv "${batch_order}")"
    arm_dir="${run_root}/$(printf '%02d_%s' "${ordinal}" "${arm}")"
    if run_arm "${symbol}" "${ordinal}" "${block}" "${arm_dir}" \
      "${names}" "${batch_order}" "${copy_trace}"; then
      rc=0
    else
      rc=$?
      failures=$((failures + 1))
      tail -n 180 "${arm_dir}/run.log" >&2 || true
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${ordinal}" "${block}" "${symbol}" "${arm}" "${batch_order}" \
      "$(basename "${arm_dir}")" "${rc}" >> "${run_root}/execution_status.tsv"
  done
  snapshot_gpu "${run_root}/gpu.final.csv"

  local verify_rc=0
  if "${VLLM_PY}" "${VERIFY}" "${run_root}" \
    --output "${run_root}/verification.json" \
    > "${run_root}/verify.log" 2> "${run_root}/verify.err"; then
    verify_rc=0
  else
    verify_rc=$?
    cat "${run_root}/verify.err" >&2 || true
    cat "${run_root}/verify.log" >&2 || true
  fi
  echo "RUN_ROOT=${run_root}"
  echo "verification=${run_root}/verification.json"
  (( failures == 0 && verify_rc == 0 )) \
    || die "integration run failed: arms=${failures} verifier=${verify_rc}"
}

case "${1:-help}" in
  static-preflight) static_preflight ;;
  smoke) run_smoke "${2:-}" ;;
  qualify) run_qwen qualification ;;
  run) run_qwen performance ;;
  verify)
    [[ $# -eq 2 ]] || die "verify requires RUN_ROOT"
    "${VLLM_PY}" "${VERIFY}" "$2"
    ;;
  *) echo "usage: $0 {static-preflight|smoke [OUTPUT]|qualify|run|verify RUN_ROOT}" >&2; exit 2 ;;
esac
