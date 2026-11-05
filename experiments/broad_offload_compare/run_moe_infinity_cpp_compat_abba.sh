#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MOEI_ROOT="${REPO_ROOT}/MoE-Infinity"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python}"
QUALIFIER="${SCRIPT_DIR}/qualify_moe_infinity.py"
SUMMARIZER="${SCRIPT_DIR}/summarize_moe_infinity_cpp_compat_abba.py"
ACTIVE_STORE="${MOEI_ROOT}/moe_infinity/_store.cpython-312-x86_64-linux-gnu.so"
ALL_CHUNK_STORE="${ALL_CHUNK_STORE:-/home/ubuntu/moe_infinity_builds/store_all_chunk_08c5c1e8_a4a78d11.so}"
MINIMAL_STORE="${MINIMAL_STORE:-/home/ubuntu/moe_infinity_builds/store_minimal_6dab7b08.so}"
ALL_CHUNK_SHA256="a4a78d1166a3fbf16c9c7585110fefabcb7eac7e399a40a136249bc95b679a7d"
MINIMAL_STORE_SHA256="b0b2b0e3ed3f80839039c33cccf38454aba14b0d6579962c379f064ee60b905b"
EXPECTED_COMMIT="ba5651897a80d9c9b7a1500cef2c68adaa63db0f"
EXPECTED_PATCH_SHA256="6dab7b086ccc3e53cec83113799683cf8f5825cda7f50d1ba3a9263d3e4e955f"
FIXTURE="${SCRIPT_DIR}/fixtures/moei_deepseek_v2_lite_calibration_prompt512_v1.json"
FIXTURE_SHA256="14ce7cb4960109fc3744ecd10fbcbb7a8c8fca41a82ee89b7de5a118fd4eaffe"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7}"
OFFLOAD_DIR="${OFFLOAD_DIR:-/home/ubuntu/moe_infinity_offload/deepseek_v2_lite_chat_85864749cd611b4353ce1decdb286193298f64c7}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/moe_infinity_cpp_compat_abba_$(date +%Y%m%d_%H%M%S)}"
CPUSET="${CPUSET:-0-13}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"
COMMAND="${1:-help}"

die() {
  echo "error: $*" >&2
  exit 1
}

sha256() {
  sha256sum "$1" | awk '{print $1}'
}

validate_source_plan() {
  [[ -x "${PYTHON_BIN}" ]] || die "missing Python: ${PYTHON_BIN}"
  [[ -f "${QUALIFIER}" && -f "${SUMMARIZER}" ]] || die "missing experiment helper"
  [[ -f "${FIXTURE}" ]] || die "missing calibration fixture"
  [[ "$(sha256 "${FIXTURE}")" == "${FIXTURE_SHA256}" ]] || die "fixture drift"
  [[ -f "${MODEL}/config.json" ]] || die "model snapshot is absent"
  [[ "$(git -C "${MOEI_ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] \
    || die "MoE-Infinity commit drift"
  local patch_sha
  patch_sha="$(git -C "${MOEI_ROOT}" diff --binary | sha256sum | awk '{print $1}')"
  [[ "${patch_sha}" == "${EXPECTED_PATCH_SHA256}" ]] || die "source patch drift"
  [[ -f "${ALL_CHUNK_STORE}" ]] || die "missing preserved all-chunk extension"
  [[ "$(sha256 "${ALL_CHUNK_STORE}")" == "${ALL_CHUNK_SHA256}" ]] \
    || die "preserved all-chunk extension drift"
  case "$(readlink -m "${RUN_ROOT}")" in
    /tmp|/tmp/*) die "RUN_ROOT must be persistent" ;;
  esac
  case "$(readlink -m "${OFFLOAD_DIR}")" in
    /tmp|/tmp/*|"$(readlink -m "${REPO_ROOT}")"/*)
      die "OFFLOAD_DIR must be persistent and outside the repository"
      ;;
  esac
}

validate_minimal_store() {
  [[ -f "${MINIMAL_STORE}" ]] || die "missing minimal extension: ${MINIMAL_STORE}"
  [[ "$(sha256 "${MINIMAL_STORE}")" == "${MINIMAL_STORE_SHA256}" ]] \
    || die "minimal extension hash mismatch"
  [[ "${MINIMAL_STORE_SHA256}" != "${ALL_CHUNK_SHA256}" ]] \
    || die "minimal and all-chunk hashes are identical"
}

wait_for_clean_gpu() {
  local used_mib pids
  for _ in $(seq 1 180); do
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits || true)"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] && (( used_mib <= 128 )) \
      && [[ -z "${pids//[[:space:]]/}" ]]; then
      return 0
    fi
    sleep 1
  done
  die "GPU did not become clean"
}

activate_store() {
  local source="$1" expected_sha="$2" temporary
  wait_for_clean_gpu
  temporary="${ACTIVE_STORE}.switch.$$"
  cp --reflink=auto -- "${source}" "${temporary}"
  chmod --reference="${source}" "${temporary}"
  mv -f -- "${temporary}" "${ACTIVE_STORE}"
  [[ "$(sha256 "${ACTIVE_STORE}")" == "${expected_sha}" ]] \
    || die "active extension switch failed"
}

restore_minimal() {
  if [[ -f "${MINIMAL_STORE}" && "${MINIMAL_STORE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    local temporary="${ACTIVE_STORE}.restore.$$"
    cp --reflink=auto -- "${MINIMAL_STORE}" "${temporary}" || return 0
    chmod --reference="${MINIMAL_STORE}" "${temporary}" || true
    mv -f -- "${temporary}" "${ACTIVE_STORE}" || true
  fi
}

write_plan() {
  mkdir -p "${RUN_ROOT}/metadata"
  git -C "${MOEI_ROOT}" diff --binary > "${RUN_ROOT}/metadata/moe_infinity.patch"
  sha256sum "${RUN_ROOT}/metadata/moe_infinity.patch" \
    > "${RUN_ROOT}/metadata/moe_infinity.patch.sha256"
  sha256sum "${ALL_CHUNK_STORE}" "${MINIMAL_STORE}" \
    > "${RUN_ROOT}/metadata/store_binaries.sha256"
  "${PYTHON_BIN}" - "${RUN_ROOT}/plan.json" "${EXPECTED_COMMIT}" \
    "${EXPECTED_PATCH_SHA256}" "${FIXTURE_SHA256}" "${ALL_CHUNK_SHA256}" \
    "${MINIMAL_STORE_SHA256}" <<'PY'
import json, os, sys
from pathlib import Path
path = Path(sys.argv[1])
arms = ("all-chunk", "minimal", "minimal", "all-chunk")
hashes = {"all-chunk": sys.argv[5], "minimal": sys.argv[6]}
value = {
    "schema_version": 1,
    "experiment": "moe_infinity_cpp_compatibility_abba",
    "moe_infinity_commit": sys.argv[2],
    "moe_infinity_patch_sha256": sys.argv[3],
    "fixture_sha256": sys.argv[4],
    "protocol": {"prompt_tokens": 512, "output_tokens": 32,
                 "warmup_requests": 2, "measured_repetitions": 3,
                 "expert_compute_threads": 1},
    "schedule": [
        {"slot": index, "arm": arm, "store_sha256": hashes[arm]}
        for index, arm in enumerate(arms, 1)
    ],
}
tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
PY
}

run_slot() {
  local slot="$1" arm="$2" source="$3" expected_sha="$4"
  local label case_root monitor_pid rc
  label="${arm//-/_}"
  case_root="${RUN_ROOT}/cases/slot${slot}_${label}"
  [[ ! -e "${case_root}" ]] || die "case already exists: ${case_root}"
  mkdir -p "${case_root}"
  activate_store "${source}" "${expected_sha}"
  sha256sum "${ACTIVE_STORE}" > "${case_root}/active_store.sha256"
  nvidia-smi \
    --query-gpu=timestamp,index,memory.used,utilization.gpu,utilization.memory,power.draw \
    --format=csv,noheader,nounits -lms 200 \
    > "${case_root}/nvml.csv" 2> "${case_root}/nvml.err" &
  monitor_pid=$!
  set +e
  env PYTHONPATH="${MOEI_ROOT}:${PYTHONPATH:-}" HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=1 timeout --signal=INT --kill-after=30s "${TIMEOUT_SECONDS}s" \
    taskset -c "${CPUSET}" "${PYTHON_BIN}" "${QUALIFIER}" \
      --model "${MODEL}" --offload-dir "${OFFLOAD_DIR}" \
      --output "${case_root}/result.json" --mode demand-only \
      --prompt-length 512 --output-tokens 32 --warmup-requests 2 \
      --repetitions 3 --device-memory-ratio 0.75 --num-threads 1 \
      --event-sample-limit 256 --prompt-token-ids-file "${FIXTURE}" \
      > "${case_root}/stdout.log" 2> "${case_root}/stderr.log"
  rc=$?
  set -e
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  printf '%s\n' "${rc}" > "${case_root}/returncode.txt"
  (( rc == 0 )) || die "slot ${slot} failed with return code ${rc}"
  wait_for_clean_gpu
}

run_abba() {
  validate_source_plan
  validate_minimal_store
  [[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists"
  command -v nvidia-smi >/dev/null || die "nvidia-smi unavailable"
  command -v timeout >/dev/null || die "timeout unavailable"
  wait_for_clean_gpu
  trap restore_minimal EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  write_plan
  run_slot 1 all-chunk "${ALL_CHUNK_STORE}" "${ALL_CHUNK_SHA256}"
  run_slot 2 minimal "${MINIMAL_STORE}" "${MINIMAL_STORE_SHA256}"
  run_slot 3 minimal "${MINIMAL_STORE}" "${MINIMAL_STORE_SHA256}"
  run_slot 4 all-chunk "${ALL_CHUNK_STORE}" "${ALL_CHUNK_SHA256}"
  restore_minimal
  trap - EXIT INT TERM
  "${PYTHON_BIN}" "${SUMMARIZER}" "${RUN_ROOT}" \
    --all-chunk-sha256 "${ALL_CHUNK_SHA256}" \
    --minimal-sha256 "${MINIMAL_STORE_SHA256}"
  echo "completed=${RUN_ROOT}/COMPLETED.json"
}

case "${COMMAND}" in
  plan)
    validate_source_plan
    echo "source_plan=OK"
    echo "reviewed_patch_sha256=${EXPECTED_PATCH_SHA256}"
    echo "all_chunk_store_sha256=${ALL_CHUNK_SHA256}"
    echo "minimal_store_sha256=${MINIMAL_STORE_SHA256}"
    ;;
  run)
    run_abba
    ;;
  help|-h|--help)
    echo "Usage: run_moe_infinity_cpp_compat_abba.sh plan|run"
    ;;
  *)
    die "unknown command: ${COMMAND}"
    ;;
esac
