#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUARD="${SCRIPT_DIR}/cpu_dma_latency_guard.py"
DIAGNOSIS="${SCRIPT_DIR}/run_diagnosis.sh"
PYTHON="${PYTHON:-python3}"
DEVICE="${CPU_DMA_LATENCY_DEVICE:-/dev/cpu_dma_latency}"

die() { echo "error: $*" >&2; exit 1; }

static_preflight() {
  [[ -f "${GUARD}" && -x "${DIAGNOSIS}" ]] || die "diagnosis scripts are missing"
  bash -n "${BASH_SOURCE[0]}" "${DIAGNOSIS}"
  "${PYTHON}" -m py_compile "${GUARD}"
  "${PYTHON}" "${GUARD}" inspect --device "${DEVICE}"
}

GUARD_JOB_PID=""
GUARD_REAL_PID=""

stop_guard() {
  local rc=0
  if [[ -n "${GUARD_REAL_PID}" ]]; then
    sudo -n kill -TERM "${GUARD_REAL_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${GUARD_JOB_PID}" ]]; then
    wait "${GUARD_JOB_PID}" || rc=$?
  fi
  GUARD_JOB_PID=""
  GUARD_REAL_PID=""
  return "${rc}"
}

start_guard() {
  local control_dir="$1"
  local ready_file="${control_dir}/guard.ready.json"
  sudo -n "${PYTHON}" "${GUARD}" hold \
    --device "${DEVICE}" \
    --latency-us "${CPU_DMA_LATENCY_US:-0}" \
    --timeout-seconds "${GUARD_TIMEOUT_SECONDS:-7200}" \
    --ready-file "${ready_file}" \
    > "${control_dir}/guard.events.jsonl" \
    2> "${control_dir}/guard.stderr.log" &
  GUARD_JOB_PID=$!

  local attempt
  for ((attempt = 0; attempt < 200; ++attempt)); do
    if [[ -s "${ready_file}" ]]; then
      GUARD_REAL_PID="$("${PYTHON}" -c \
        'import json,sys; print(int(json.load(open(sys.argv[1]))["pid"]))' \
        "${ready_file}")"
      sudo -n kill -0 "${GUARD_REAL_PID}" >/dev/null
      return 0
    fi
    kill -0 "${GUARD_JOB_PID}" >/dev/null 2>&1 \
      || die "cpu_dma_latency guard exited before becoming ready"
    sleep 0.05
  done
  die "timed out waiting for cpu_dma_latency guard"
}

run_suite() {
  [[ "${ALLOW_GPU_RUN:-0}" == "1" ]] \
    || die "run requires explicit ALLOW_GPU_RUN=1"
  static_preflight >/dev/null
  sudo -n true
  sudo -n test -c "${DEVICE}"
  sudo -n test -w "${DEVICE}"

  local mode_order="${DMA_MODE_ORDER:-BH}"
  [[ "${mode_order}" =~ ^[BH]+$ && "${mode_order}" == *B* && "${mode_order}" == *H* ]] \
    || die "DMA_MODE_ORDER must contain both B (baseline) and H (held)"
  local default_root
  default_root="${SCRIPT_DIR}/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_cpu_dma_latency_${mode_order}_v1"
  local suite_root="${SUITE_ROOT:-${default_root}}"
  [[ ! -e "${suite_root}" ]] || die "suite root already exists: ${suite_root}"
  mkdir -p "${suite_root}"
  {
    echo "schema=p3_cpu_dma_latency_compare_v1"
    echo "mode_order=${mode_order}"
    echo "latency_us=${CPU_DMA_LATENCY_US:-0}"
    echo "device=${DEVICE}"
    echo "started_at=$(date --iso-8601=seconds)"
  } > "${suite_root}/suite.env"

  trap 'stop_guard || true' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  local index symbol label case_dir diagnosis_root
  for ((index = 0; index < ${#mode_order}; ++index)); do
    symbol="${mode_order:index:1}"
    if [[ "${symbol}" == "B" ]]; then
      label=baseline
    else
      label=dma_latency_held
    fi
    case_dir="${suite_root}/$(printf '%02d_%s' "$((index + 1))" "${label}")"
    diagnosis_root="${case_dir}/diagnosis"
    mkdir -p "${case_dir}/control"
    {
      echo "symbol=${symbol}"
      echo "label=${label}"
      echo "started_at=$(date --iso-8601=seconds)"
    } > "${case_dir}/case.env"

    if [[ "${symbol}" == "H" ]]; then
      start_guard "${case_dir}/control"
    fi
    RUN_ROOT="${diagnosis_root}" "${DIAGNOSIS}"
    if [[ "${symbol}" == "H" ]]; then
      sudo -n kill -0 "${GUARD_REAL_PID}" >/dev/null \
        || die "cpu_dma_latency guard ended during diagnosis"
      stop_guard
    fi
    echo "ended_at=$(date --iso-8601=seconds)" >> "${case_dir}/case.env"
  done
  trap - EXIT INT TERM
  echo "completed_at=$(date --iso-8601=seconds)" >> "${suite_root}/suite.env"
  echo "SUITE_ROOT=${suite_root}"
}

case "${1:-help}" in
  static-preflight) static_preflight ;;
  run) run_suite ;;
  *)
    echo "usage: $0 {static-preflight|run}" >&2
    echo "formal order: ALLOW_GPU_RUN=1 DMA_MODE_ORDER=BHHB $0 run" >&2
    exit 2
    ;;
esac
