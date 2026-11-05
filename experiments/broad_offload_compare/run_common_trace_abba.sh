#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTROL="${SCRIPT_DIR}/common_trace_abba.py"
MICROBENCH="${SCRIPT_DIR}/common_trace_moe_microbench.py"
KT_ROOT="${REPO_ROOT}/KTransformers"
KT_PY="${KT_ROOT}/.venv/bin/python"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"

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

static_preflight() {
  require_file "${CONTROL}"
  require_file "${MICROBENCH}"
  require_file "${KT_PY}"
  require_file "${VLLM_PY}"
  command -v jq >/dev/null || die "jq is unavailable"
  command -v taskset >/dev/null || die "taskset is unavailable"
  command -v timeout >/dev/null || die "timeout is unavailable"
  bash -n "${BASH_SOURCE[0]}"
  "${VLLM_PY}" -m py_compile "${CONTROL}" "${MICROBENCH}"
  "${VLLM_PY}" "${CONTROL}" --help >/dev/null
  "${VLLM_PY}" "${MICROBENCH}" --help >/dev/null
  echo "common-trace ABBA static preflight: OK (no CUDA workload launched)"
}

preregister() {
  static_preflight
  local run_root="${RUN_ROOT:-}"
  local model="${MODEL:-}"
  local fixture="${FIXTURE:-}"
  local fixture_manifest="${FIXTURE_MANIFEST:-}"
  local reference_output="${REFERENCE_OUTPUT:-}"
  local tolerance="${TOLERANCE:-}"
  [[ -n "${run_root}" && -n "${model}" && -n "${fixture}" ]] \
    || die "RUN_ROOT, MODEL and FIXTURE are required"
  [[ -n "${fixture_manifest}" && -n "${reference_output}" && -n "${tolerance}" ]] \
    || die "FIXTURE_MANIFEST, REFERENCE_OUTPUT and TOLERANCE are required"
  taskset -c "${CPUSET:-0-13}" true >/dev/null 2>&1 \
    || die "invalid CPUSET=${CPUSET:-0-13}"
  "${VLLM_PY}" "${CONTROL}" preregister \
    --run-root "${run_root}" \
    --model "${model}" \
    --fixture "${fixture}" \
    --fixture-manifest "${fixture_manifest}" \
    --reference-output "${reference_output}" \
    --tolerance "${tolerance}" \
    --blocks "${ABBA_BLOCKS:-3}" \
    --threads "${CPU_THREADS:-12}" \
    --numa-node "${NUMA_NODE:-0}" \
    --cpuset "${CPUSET:-0-13}" \
    --warmup "${WARMUP:-3}" \
    --iterations "${ITERATIONS:-20}"
}

wait_for_clean_gpu() {
  local clean_mib="${GPU_CLEAN_MIB:-128}"
  local used_mib=""
  local compute_processes=""
  local attempt
  for attempt in $(seq 1 180); do
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | head -n 1 | tr -d ' ')"
    compute_processes="$(nvidia-smi --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] \
      && (( used_mib <= clean_mib )) \
      && [[ -z "${compute_processes//[[:space:]]/}" ]]; then
      return 0
    fi
    sleep 1
  done
  nvidia-smi >&2 || true
  die "GPU did not become clean (used=${used_mib:-unknown}, pids=${compute_processes:-none})"
}

run_one() {
  local run_root="$1"
  local block="$2"
  local slot="$3"
  local mechanism="$4"
  local case_root="$5"
  local slot_root
  local model fixture fixture_manifest reference tolerance
  local atol rtol relative_l1 threads numa cpuset warmup iterations
  local monitor_pid=""
  local rc=0
  local verify_rc=0
  local -a command=()

  slot_root="$(dirname "${case_root}")"
  mkdir -p "${slot_root}"
  model="$(jq -r '.model.path' "${run_root}/run_manifest.json")"
  fixture="$(jq -r '.inputs.fixture.snapshot_path' "${run_root}/run_manifest.json")"
  fixture_manifest="$(jq -r '.inputs.fixture_manifest.snapshot_path' "${run_root}/run_manifest.json")"
  reference="$(jq -r '.inputs.reference_output.snapshot_path' "${run_root}/run_manifest.json")"
  tolerance="$(jq -r '.inputs.tolerance.snapshot_path' "${run_root}/run_manifest.json")"
  atol="$(jq -r '.thresholds.atol' "${run_root}/run_manifest.json")"
  rtol="$(jq -r '.thresholds.rtol' "${run_root}/run_manifest.json")"
  relative_l1="$(jq -r '.thresholds.max_relative_l1' "${run_root}/run_manifest.json")"
  threads="$(jq -r '.runtime.threads' "${run_root}/run_manifest.json")"
  numa="$(jq -r '.runtime.numa_node' "${run_root}/run_manifest.json")"
  cpuset="$(jq -r '.runtime.cpuset' "${run_root}/run_manifest.json")"
  warmup="$(jq -r '.runtime.warmup' "${run_root}/run_manifest.json")"
  iterations="$(jq -r '.runtime.iterations' "${run_root}/run_manifest.json")"
  require_file "${fixture}"
  require_file "${fixture_manifest}"
  require_file "${reference}"
  require_file "${tolerance}"

  "${VLLM_PY}" "${CONTROL}" update-status \
    --run-root "${run_root}" --block "${block}" --slot "${slot}" \
    --status running > "${slot_root}/status_start.json"
  wait_for_clean_gpu

  if [[ "${mechanism}" == "kt" ]]; then
    command=(
      env
      "PATH=${KT_ROOT}/.venv/bin:${PATH}"
      "OMP_NUM_THREADS=${threads}"
      "MKL_NUM_THREADS=${threads}"
      taskset -c "${cpuset}"
      "${KT_PY}" "${MICROBENCH}" run-kt
      --model "${model}"
      --fixture "${fixture}"
      --manifest "${fixture_manifest}"
      --output-dir "${case_root}"
      --threads "${threads}"
      --numa-node "${numa}"
      --warmup "${warmup}"
      --iterations "${iterations}"
      --reference-output "${reference}"
      --atol "${atol}"
      --rtol "${rtol}"
      --max-relative-l1 "${relative_l1}"
      --allow-gpu-run
    )
  elif [[ "${mechanism}" == "custom" ]]; then
    command=(
      env
      "PYTHONPATH=${VLLM_ROOT}"
      "OMP_NUM_THREADS=${threads}"
      "MKL_NUM_THREADS=${threads}"
      taskset -c "${cpuset}"
      "${VLLM_PY}" "${MICROBENCH}" run-custom
      --model "${model}"
      --fixture "${fixture}"
      --manifest "${fixture_manifest}"
      --output-dir "${case_root}"
      --warmup "${warmup}"
      --iterations "${iterations}"
      --reference-output "${reference}"
      --atol "${atol}"
      --rtol "${rtol}"
      --max-relative-l1 "${relative_l1}"
      --allow-gpu-run
    )
  else
    die "unknown scheduled mechanism: ${mechanism}"
  fi
  {
    printf 'cd %q\n' "${VLLM_ROOT}"
    printf '%q ' timeout "${SLOT_TIMEOUT:-1800}s" /usr/bin/time -v \
      -o "${slot_root}/resource_time.txt" "${command[@]}"
    printf '\n'
  } > "${slot_root}/command.txt"

  nvidia-smi dmon -s pucvmet -d 1 -o DT \
    > "${slot_root}/gpu_dmon.csv" 2>&1 &
  monitor_pid="$!"
  set +e
  (
    cd "${VLLM_ROOT}"
    timeout "${SLOT_TIMEOUT:-1800}s" /usr/bin/time -v \
      -o "${slot_root}/resource_time.txt" "${command[@]}"
  ) > "${slot_root}/stdout.json" 2> "${slot_root}/stderr.log"
  rc=$?
  set -e
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  printf '%s\n' "${rc}" > "${slot_root}/returncode.txt"

  if [[ "${rc}" -eq 0 ]]; then
    set +e
    "${VLLM_PY}" "${CONTROL}" verify-slot \
      --run-root "${run_root}" --block "${block}" --slot "${slot}" \
      > "${slot_root}/verify_slot.stdout.json" \
      2> "${slot_root}/verify_slot.stderr.log"
    verify_rc=$?
    set -e
  else
    verify_rc="${rc}"
  fi
  wait_for_clean_gpu
  if [[ "${rc}" -eq 0 && "${verify_rc}" -eq 0 ]]; then
    "${VLLM_PY}" "${CONTROL}" update-status \
      --run-root "${run_root}" --block "${block}" --slot "${slot}" \
      --status success --returncode 0 > "${slot_root}/status_end.json"
    return 0
  fi
  "${VLLM_PY}" "${CONTROL}" update-status \
    --run-root "${run_root}" --block "${block}" --slot "${slot}" \
    --status failed --returncode "${rc}" \
    --detail "microbench_rc=${rc},verify_rc=${verify_rc}" \
    > "${slot_root}/status_end.json"
  return 1
}

run_schedule() {
  [[ "${ALLOW_GPU_RUN:-0}" == "1" ]] \
    || die "run requires ALLOW_GPU_RUN=1"
  static_preflight
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
  local run_root="${RUN_ROOT:-}"
  local failures=0
  local block slot mechanism case_root status
  [[ -n "${run_root}" ]] || die "RUN_ROOT is required"
  require_file "${run_root}/run_manifest.json"
  require_file "${run_root}/execution_status.json"
  "${VLLM_PY}" "${CONTROL}" validate-manifest --run-root "${run_root}" \
    > "${run_root}/pre_run_manifest_validation.json"
  while IFS=$'\t' read -r block slot mechanism case_root; do
    status="$(jq -r --argjson block "${block}" --argjson slot "${slot}" \
      '.slots[] | select(.block == $block and .slot == $slot) | .status' \
      "${run_root}/execution_status.json")"
    if [[ "${status}" != "pending" ]]; then
      echo "skip block=${block} slot=${slot} status=${status}"
      [[ "${status}" == "success" ]] || failures=$((failures + 1))
      continue
    fi
    if ! run_one "${run_root}" "${block}" "${slot}" "${mechanism}" "${case_root}"; then
      failures=$((failures + 1))
    fi
  done < <(
    jq -r '.schedule[] | [.block,.slot,.mechanism,.case_root] | @tsv' \
      "${run_root}/run_manifest.json"
  )
  set +e
  "${VLLM_PY}" "${CONTROL}" summarize --run-root "${run_root}" \
    > "${run_root}/summarize.stdout.json" \
    2> "${run_root}/summarize.stderr.log"
  local summary_rc=$?
  set -e
  if (( failures > 0 || summary_rc != 0 )); then
    echo "error: ${failures} slot(s) failed; all preregistered failures were retained" >&2
    return 1
  fi
  echo "summary=${run_root}/COMMON_TRACE_ABBA_SUMMARY.json"
}

summarize_only() {
  static_preflight
  [[ -n "${RUN_ROOT:-}" ]] || die "RUN_ROOT is required"
  "${VLLM_PY}" "${CONTROL}" validate-manifest --run-root "${RUN_ROOT}" \
    > "${RUN_ROOT}/pre_summary_manifest_validation.json"
  "${VLLM_PY}" "${CONTROL}" summarize --run-root "${RUN_ROOT}"
}

show_help() {
  cat <<'EOF'
Usage:
  run_common_trace_abba.sh preflight
  RUN_ROOT=... MODEL=... FIXTURE=... FIXTURE_MANIFEST=... \
    REFERENCE_OUTPUT=... TOLERANCE=... run_common_trace_abba.sh preregister
  ALLOW_GPU_RUN=1 RUN_ROOT=... run_common_trace_abba.sh run
  RUN_ROOT=... run_common_trace_abba.sh summarize

The formal default is three cold-process ABBA blocks (12 processes total).
Odd blocks use KT-custom-custom-KT; even blocks reverse the order.  The
summarizer resamples whole blocks and never treats in-process iterations as
independent observations.
EOF
}

case "${COMMAND}" in
  preflight)
    static_preflight
    ;;
  preregister)
    preregister
    ;;
  run)
    run_schedule
    ;;
  summarize)
    summarize_only
    ;;
  help|-h|--help)
    show_help
    ;;
  *)
    show_help >&2
    die "unknown command: ${COMMAND}"
    ;;
esac
