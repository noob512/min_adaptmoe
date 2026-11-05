#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
QUALIFICATION_RUNNER="${SCRIPT_DIR}/run_moe_infinity_qualification.sh"
SUMMARIZER="${SCRIPT_DIR}/summarize_moe_infinity_thread_calibration.py"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python}"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7}"
CALIBRATION_FIXTURE="${CALIBRATION_FIXTURE:-${SCRIPT_DIR}/fixtures/moei_deepseek_v2_lite_calibration_prompt512_v1.json}"
EXPECTED_FIXTURE_SHA256="14ce7cb4960109fc3744ecd10fbcbb7a8c8fca41a82ee89b7de5a118fd4eaffe"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-${SCRIPT_DIR}/artifacts/moe_infinity_thread_calibration_$(date +%Y%m%d_%H%M%S)}"
OFFLOAD_DIR="${OFFLOAD_DIR:-/home/ubuntu/moe_infinity_offload/deepseek_v2_lite_chat_85864749cd611b4353ce1decdb286193298f64c7}"
THREAD_ORDER="${THREAD_ORDER:-1,4,8,14,14,8,4,1}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"
REPETITIONS="${REPETITIONS:-3}"
PROMPT_LENGTH="${PROMPT_LENGTH:-512}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-32}"
DEVICE_MEMORY_RATIO="${DEVICE_MEMORY_RATIO:-0.75}"
CPUSET="${CPUSET:-0-13}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-3600}"

COMMAND="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

die() {
  echo "error: $*" >&2
  exit 1
}

fixture_sha256() {
  sha256sum "${CALIBRATION_FIXTURE}" | awk '{print $1}'
}

validate_plan() {
  [[ -f "${QUALIFICATION_RUNNER}" ]] || die "missing ${QUALIFICATION_RUNNER}"
  [[ -f "${SUMMARIZER}" ]] || die "missing ${SUMMARIZER}"
  [[ -f "${CALIBRATION_FIXTURE}" ]] || die "missing ${CALIBRATION_FIXTURE}"
  [[ "$(fixture_sha256)" == "${EXPECTED_FIXTURE_SHA256}" ]] \
    || die "calibration fixture SHA-256 drift"
  [[ "${PROMPT_LENGTH}" == "512" ]] || die "PROMPT_LENGTH must be 512"
  [[ "${OUTPUT_TOKENS}" == "32" ]] || die "OUTPUT_TOKENS must be 32"
  [[ "${WARMUP_REQUESTS}" == "2" ]] || die "WARMUP_REQUESTS must be 2"
  [[ "${REPETITIONS}" == "3" ]] || die "REPETITIONS must be 3"
  [[ "${THREAD_ORDER}" == "1,4,8,14,14,8,4,1" ]] \
    || die "THREAD_ORDER must remain preregistered"
  case "$(readlink -m "${CALIBRATION_ROOT}")" in
    /tmp|/tmp/*) die "CALIBRATION_ROOT must not be under /tmp" ;;
  esac
  case "$(readlink -m "${OFFLOAD_DIR}")" in
    /tmp|/tmp/*|"$(readlink -m "${REPO_ROOT}")"/*)
      die "OFFLOAD_DIR must be persistent and outside the repository"
      ;;
  esac
  PROMPT_TOKEN_IDS_FILE="${CALIBRATION_FIXTURE}" \
  PROMPT_LENGTH="${PROMPT_LENGTH}" \
  RUN_ROOT="${CALIBRATION_ROOT}/static_preflight_placeholder" \
  OFFLOAD_DIR="${OFFLOAD_DIR}" \
    bash "${QUALIFICATION_RUNNER}" static-preflight
}

write_plan() {
  mkdir -p "${CALIBRATION_ROOT}"
  {
    echo "calibration_root=$(readlink -m "${CALIBRATION_ROOT}")"
    echo "offload_dir=$(readlink -m "${OFFLOAD_DIR}")"
    echo "model=$(readlink -m "${MODEL}")"
    echo "calibration_fixture=$(readlink -m "${CALIBRATION_FIXTURE}")"
    echo "calibration_fixture_sha256=$(fixture_sha256)"
    echo "thread_order=${THREAD_ORDER}"
    echo "warmup_requests=${WARMUP_REQUESTS}"
    echo "repetitions=${REPETITIONS}"
    echo "selection_metric=median_decode_tpot_across_all_measured_trials"
    echo "confirmation=independent_ABBA_winner_vs_runner_up"
  } > "${CALIBRATION_ROOT}/calibration_plan.env"
  sha256sum "${BASH_SOURCE[0]}" "${QUALIFICATION_RUNNER}" "${SUMMARIZER}" \
    "${CALIBRATION_FIXTURE}" \
    > "${CALIBRATION_ROOT}/calibration_inputs.sha256"
}

run_scan() {
  validate_plan
  [[ ! -e "${CALIBRATION_ROOT}" ]] \
    || die "CALIBRATION_ROOT already exists; use a new root or summarize it"
  write_plan
  IFS=',' read -r -a thread_counts <<< "${THREAD_ORDER}"
  local index=0 threads label child_root
  for threads in "${thread_counts[@]}"; do
    index=$((index + 1))
    printf -v label 'scan_%02d_t%s' "${index}" "${threads}"
    child_root="${CALIBRATION_ROOT}/runs/${label}"
    echo "starting=${label}"
    MODEL="${MODEL}" \
    RUN_ROOT="${child_root}" \
    OFFLOAD_DIR="${OFFLOAD_DIR}" \
    PROMPT_TOKEN_IDS_FILE="${CALIBRATION_FIXTURE}" \
    PROMPT_LENGTH="${PROMPT_LENGTH}" \
    OUTPUT_TOKENS="${OUTPUT_TOKENS}" \
    WARMUP_REQUESTS="${WARMUP_REQUESTS}" \
    REPETITIONS="${REPETITIONS}" \
    NUM_THREADS="${threads}" \
    DEVICE_MEMORY_RATIO="${DEVICE_MEMORY_RATIO}" \
    CPUSET="${CPUSET}" \
    TIMEOUT_SECONDS="${TIMEOUT_SECONDS}" \
      bash "${QUALIFICATION_RUNNER}" run-case demand-only
  done
  summarize_scan
}

summarize_scan() {
  "${PYTHON_BIN}" "${SUMMARIZER}" "${CALIBRATION_ROOT}" \
    --order "${THREAD_ORDER}" \
    --warmups "${WARMUP_REQUESTS}" \
    --repetitions "${REPETITIONS}" \
    --fixture-sha256 "$(fixture_sha256)" \
    --output-json "${CALIBRATION_ROOT}/thread_calibration_snapshot.json" \
    --output-md "${CALIBRATION_ROOT}/THREAD_CALIBRATION_RESULTS.md" \
    > "${CALIBRATION_ROOT}/summarizer.stdout.json"
  echo "snapshot=${CALIBRATION_ROOT}/thread_calibration_snapshot.json"
}

show_help() {
  cat <<'EOF'
Usage:
  run_moe_infinity_thread_calibration.sh plan
  run_moe_infinity_thread_calibration.sh run
  run_moe_infinity_thread_calibration.sh summarize

The run command uses independent cold processes in preregistered order
1,4,8,14,14,8,4,1 on a calibration-only frozen token-ID fixture. The selected
winner remains provisional until an independent winner/runner-up A/B/B/A run.
EOF
}

case "${COMMAND}" in
  plan)
    [[ $# -eq 0 ]] || die "plan takes no arguments"
    validate_plan
    ;;
  run)
    [[ $# -eq 0 ]] || die "run takes no arguments"
    run_scan
    ;;
  summarize)
    [[ $# -eq 0 ]] || die "summarize takes no arguments"
    summarize_scan
    ;;
  help|-h|--help)
    show_help
    ;;
  *)
    show_help >&2
    die "unknown command: ${COMMAND}"
    ;;
esac
