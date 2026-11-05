#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KT_ROOT="${REPO_ROOT}/KTransformers"
CONTROL="${SCRIPT_DIR}/kt_dense_fp8_ablation.py"
TUNING_CONTROL="${SCRIPT_DIR}/kt_dense_fp8_tuning.py"
QUALIFICATION_RUNNER="${SCRIPT_DIR}/run_kt_qualification.sh"

EVALUATION_FIXTURE="${SCRIPT_DIR}/../eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json"
WARMUP_FIXTURE="${SCRIPT_DIR}/../eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/calibration_fixture_token_ids.json"
FREQUENCY_STATS="${SCRIPT_DIR}/../eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/routing/frequency_stats.pt"
FREQUENCY_STATS_MANIFEST="${SCRIPT_DIR}/../eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/routing/frequency_stats_manifest.json"

GPU_CLEAN_MIB=128
ACTIVE_RUN_ROOT=""
INITIAL_STATE=""
RESTORE_DONE=0

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

require_absent() {
  [[ ! -e "$1" ]] || die "refusing to overwrite existing path: $1"
}

controller() {
  python3 "${CONTROL}" "$@"
}

gpu_compute_pids() {
  local raw
  raw="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)" \
    || return 1
  awk '$1 ~ /^[0-9]+$/ {print $1}' <<< "${raw}"
}

gpu_has_no_compute_process() {
  local pids
  pids="$(gpu_compute_pids)" || return 1
  [[ -z "${pids}" ]]
}

gpu_is_clean() {
  local used_mib pids
  used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | head -n 1 | tr -d ' ')" || return 1
  pids="$(gpu_compute_pids)" || return 1
  [[ "${used_mib}" =~ ^[0-9]+$ ]] \
    && (( used_mib <= GPU_CLEAN_MIB )) \
    && [[ -z "${pids}" ]]
}

wait_for_clean_gpu() {
  local attempts="${1:-180}"
  local _
  for _ in $(seq 1 "${attempts}"); do
    if gpu_is_clean; then
      return 0
    fi
    sleep 1
  done
  return 1
}

safe_copy() {
  local source="$1"
  local target="$2"
  require_file "${source}"
  require_absent "${target}"
  cp -p -- "${source}" "${target}"
}

read_initial_state() {
  jq -er '.initial_config_state.state | select(. == "default" or . == "tuned")' \
    "$1/run_manifest.json"
}

restore_initial_config() {
  local run_root="${ACTIVE_RUN_ROOT}"
  local bundle_dir current_state journal receipt_name
  [[ -n "${run_root}" && -f "${run_root}/run_manifest.json" ]] || return 0
  [[ "${RESTORE_DONE}" == "0" ]] || return 0
  bundle_dir="${run_root}/bundle_snapshot"
  if [[ -z "${INITIAL_STATE}" ]]; then
    INITIAL_STATE="$(read_initial_state "${run_root}")" || return 1
  fi
  if ! gpu_has_no_compute_process; then
    echo "error: GPU still has a compute process; configuration was not switched during cleanup." >&2
    echo "error: after the GPU is clean, run: $0 recover ${run_root}" >&2
    return 1
  fi
  current_state="$(controller config-state \
    --bundle-dir "${bundle_dir}" --kt-root "${KT_ROOT}" | jq -er '.state')" \
    || return 1
  if [[ "${current_state}" == "invalid" ]]; then
    echo "error: dense config state is partial/different; refusing automatic recovery" >&2
    return 1
  fi
  receipt_name="final_restore_to_${INITIAL_STATE}.json"
  journal="${run_root}/transitions/${receipt_name}"
  if [[ -e "${journal}" ]]; then
    if [[ "${current_state}" == "${INITIAL_STATE}" ]]; then
      RESTORE_DONE=1
      return 0
    fi
    echo "error: restore receipt exists but current config is ${current_state}" >&2
    return 1
  fi
  require_absent "${run_root}/logs/final_restore.stdout.json"
  controller transition \
    --bundle-dir "${bundle_dir}" \
    --kt-root "${KT_ROOT}" \
    --target "${INITIAL_STATE}" \
    --journal "${journal}" \
    > "${run_root}/logs/final_restore.stdout.json" || return 1
  current_state="$(controller config-state \
    --bundle-dir "${bundle_dir}" --kt-root "${KT_ROOT}" | jq -er '.state')" \
    || return 1
  [[ "${current_state}" == "${INITIAL_STATE}" ]] || return 1
  RESTORE_DONE=1
}

on_exit() {
  local returncode=$?
  trap - EXIT INT TERM HUP
  set +e
  if [[ -n "${ACTIVE_RUN_ROOT}" && "${RESTORE_DONE}" == "0" ]]; then
    if ! restore_initial_config; then
      returncode=1
    fi
  fi
  exit "${returncode}"
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

preflight_control() {
  require_dir "${KT_ROOT}"
  require_file "${CONTROL}"
  command -v python3 >/dev/null || die "python3 is unavailable"
  command -v jq >/dev/null || die "jq is unavailable"
  command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
}

preflight_files() {
  preflight_control
  require_file "${TUNING_CONTROL}"
  require_file "${QUALIFICATION_RUNNER}"
  require_file "${EVALUATION_FIXTURE}"
  require_file "${WARMUP_FIXTURE}"
  require_file "${FREQUENCY_STATS}"
  require_file "${FREQUENCY_STATS_MANIFEST}"
}

update_slot_status() {
  local run_root="$1"
  local slot="$2"
  local state="$3"
  local returncode="$4"
  local detail="$5"
  local snapshot="${run_root}/status/slot${slot}_${state}.json"
  require_absent "${snapshot}"
  controller update-status \
    --run-root "${run_root}" \
    --slot "${slot}" \
    --status "${state}" \
    --returncode "${returncode}" \
    --detail "${detail}" \
    > "${snapshot}"
}

skip_remaining_slots() {
  local run_root="$1"
  local first_slot="$2"
  local slot
  for slot in $(seq "${first_slot}" 4); do
    update_slot_status "${run_root}" "${slot}" skipped 75 \
      "not started because the GPU/config transition safety gate stopped the sequence"
  done
}

run_slot() {
  local run_root="$1"
  local slot="$2"
  local arm="$3"
  local target_state="$4"
  local case_label="block1/slot${slot}_${arm}"
  local case_dir="${run_root}/cases/${case_label}"
  local pretransition="${run_root}/states/slot${slot}_pretransition.json"
  local before="${run_root}/states/slot${slot}_before_server.json"
  local after="${run_root}/states/slot${slot}_after_server.json"
  local transition="${run_root}/transitions/slot${slot}_to_${target_state}.json"
  local qual_stdout="${run_root}/logs/slot${slot}_qualification.stdout.log"
  local qual_stderr="${run_root}/logs/slot${slot}_qualification.stderr.log"
  local qual_returncode="${run_root}/logs/slot${slot}_qualification.returncode.txt"
  local audit_stdout="${run_root}/logs/slot${slot}_dense_audit.stdout.json"
  local verify_stdout="${run_root}/logs/slot${slot}_integrity.stdout.json"
  local qualification_rc=0
  local audit_rc=0
  local verify_rc=0
  local clean_rc=0
  local slot_rc=0

  update_slot_status "${run_root}" "${slot}" running 0 \
    "preparing ${arm}/${target_state}" || die "failed to start slot ${slot} status"
  require_absent "${case_dir}"
  require_absent "${pretransition}"
  require_absent "${before}"
  require_absent "${after}"
  require_absent "${transition}"
  require_absent "${qual_stdout}"
  require_absent "${qual_stderr}"
  require_absent "${qual_returncode}"

  if ! controller config-state \
      --bundle-dir "${run_root}/bundle_snapshot" \
      --kt-root "${KT_ROOT}" \
      --output "${pretransition}" >/dev/null; then
    update_slot_status "${run_root}" "${slot}" failed 78 \
      "could not inspect config before transition" || true
    return 75
  fi
  if ! controller transition \
      --bundle-dir "${run_root}/bundle_snapshot" \
      --kt-root "${KT_ROOT}" \
      --target "${target_state}" \
      --journal "${transition}" \
      > "${run_root}/logs/slot${slot}_transition.stdout.json"; then
    update_slot_status "${run_root}" "${slot}" failed 78 \
      "hash-restricted config transition failed" || true
    return 75
  fi
  if ! controller config-state \
      --bundle-dir "${run_root}/bundle_snapshot" \
      --kt-root "${KT_ROOT}" \
      --output "${before}" >/dev/null; then
    update_slot_status "${run_root}" "${slot}" failed 78 \
      "could not inspect config after transition" || true
    return 75
  fi
  if [[ "$(jq -er '.state' "${before}")" != "${target_state}" ]]; then
    update_slot_status "${run_root}" "${slot}" failed 78 \
      "config did not reach ${target_state} before server start" || true
    return 75
  fi

  if env \
    RUN_ROOT="${run_root}" \
    CASE_LABEL="${case_label}" \
    PORT="${KT_DENSE_PORT:-30000}" \
    CPUSET="0-13" \
    GPU_CLEAN_MIB="${GPU_CLEAN_MIB}" \
    GPU_EXPERTS_RATIO="0.77083333333333337" \
    KV_TOKENS="13104" \
    MAX_CONTEXT_LENGTH="512" \
    MAX_RUNNING_REQUESTS="32" \
    QUAL_CHUNKED_PREFILL_SIZE="4096" \
    QUAL_MAX_PREFILL_TOKENS="4096" \
    QUAL_CONCURRENCIES="8" \
    QUAL_OUTPUT_TOKENS="64" \
    WARM_CONCURRENCY="8" \
    WARM_OUTPUT_TOKENS="8" \
    REQUIRE_PROMPT_TOKEN_IDS="1" \
    EVALUATION_FIXTURE_OVERRIDE="${run_root}/inputs/evaluation_fixture_token_ids.json" \
    CALIBRATION_FIXTURE_OVERRIDE="${run_root}/inputs/calibration_fixture_token_ids.json" \
    FREQUENCY_STATS_OVERRIDE="${FREQUENCY_STATS}" \
    FREQUENCY_STATS_MANIFEST_OVERRIDE="${FREQUENCY_STATS_MANIFEST}" \
    bash "${QUALIFICATION_RUNNER}" run-case exact 12 \
      > "${qual_stdout}" 2> "${qual_stderr}"; then
    qualification_rc=0
  else
    qualification_rc=$?
  fi
  printf '%s\n' "${qualification_rc}" > "${qual_returncode}"

  if wait_for_clean_gpu 180; then
    clean_rc=0
  else
    clean_rc=75
  fi

  if ! controller config-state \
      --bundle-dir "${run_root}/bundle_snapshot" \
      --kt-root "${KT_ROOT}" \
      --output "${after}" >/dev/null; then
    update_slot_status "${run_root}" "${slot}" failed 78 \
      "qualification=${qualification_rc}; could not inspect post-server config" || true
    return 75
  fi
  mkdir -p "${case_dir}"
  safe_copy "${pretransition}" "${case_dir}/config_state_pretransition.json"
  safe_copy "${before}" "${case_dir}/config_state_before.json"
  safe_copy "${after}" "${case_dir}/config_state_after.json"
  safe_copy "${transition}" "${case_dir}/config_transition_receipt.json"
  safe_copy "${qual_stdout}" "${case_dir}/outer_runner.stdout.log"
  safe_copy "${qual_stderr}" "${case_dir}/outer_runner.stderr.log"
  safe_copy "${qual_returncode}" "${case_dir}/runner.returncode.txt"

  if [[ -f "${case_dir}/server.log" ]]; then
    if python3 "${TUNING_CONTROL}" audit-log \
      --log "${case_dir}/server.log" \
      --expect "${target_state}" \
      --output "${case_dir}/dense_fp8_warning_gate.json" \
      > "${audit_stdout}"; then
      audit_rc=0
    else
      audit_rc=$?
    fi
  else
    audit_rc=76
  fi

  if controller verify-slot \
    --case-dir "${case_dir}" \
    --run-manifest "${run_root}/run_manifest.json" \
    --arm "${arm}" \
    --output "${case_dir}/integrity_gate.json" \
    > "${verify_stdout}"; then
    verify_rc=0
  else
    verify_rc=$?
  fi

  if (( qualification_rc != 0 )); then
    slot_rc="${qualification_rc}"
  elif (( audit_rc != 0 )); then
    slot_rc="${audit_rc}"
  elif (( verify_rc != 0 )); then
    slot_rc="${verify_rc}"
  elif (( clean_rc != 0 )); then
    slot_rc="${clean_rc}"
  fi

  if (( slot_rc == 0 )); then
    update_slot_status "${run_root}" "${slot}" ok 0 \
      "qualification=0,audit=0,integrity=0,gpu_clean=0" \
      || die "failed to finish slot ${slot} status"
  else
    update_slot_status "${run_root}" "${slot}" failed "${slot_rc}" \
      "qualification=${qualification_rc},audit=${audit_rc},integrity=${verify_rc},gpu_clean=${clean_rc}" \
      || die "failed to finish slot ${slot} status"
  fi

  if (( clean_rc != 0 )); then
    echo "error: slot ${slot} left the GPU unclean; no further config switch is allowed" >&2
    return 75
  fi
  if (( slot_rc != 0 )); then
    return 1
  fi
  return 0
}

summarize_run() {
  local run_root="$1"
  require_file "${run_root}/run_manifest.json"
  require_file "${run_root}/execution_status.json"
  require_absent "${run_root}/DENSE_FP8_DTTD_SUMMARY.json"
  require_absent "${run_root}/DENSE_FP8_DTTD_SUMMARY.md"
  controller summarize --run-root "${run_root}"
}

run_ablation() {
  local run_root bundle_dir preregister_tmp
  local -a arms=(D T T D)
  local -a states=(default tuned tuned default)
  local index slot
  local failed_slots=0
  local unsafe_stop=0
  local summary_rc=0

  [[ "${ALLOW_GPU_RUN:-0}" == "1" ]] \
    || die "run requires ALLOW_GPU_RUN=1 after the dense tuner has released the GPU"
  [[ $# -eq 2 ]] || die "run requires RUN_ROOT and BUNDLE_DIR"
  preflight_files
  run_root="$(realpath -m "$1")"
  bundle_dir="$(realpath "$2")"
  require_absent "${run_root}"
  require_dir "${bundle_dir}"
  if ! gpu_is_clean; then
    die "GPU is not clean; preregistration/execution was not started"
  fi

  preregister_tmp="$(mktemp)"
  if controller preregister \
      --run-root "${run_root}" \
      --bundle-dir "${bundle_dir}" \
      --kt-root "${KT_ROOT}" \
      --evaluation-fixture "${EVALUATION_FIXTURE}" \
      --warmup-fixture "${WARMUP_FIXTURE}" \
      --frequency-stats "${FREQUENCY_STATS}" \
      --stats-manifest "${FREQUENCY_STATS_MANIFEST}" \
      > "${preregister_tmp}"; then
    mv -- "${preregister_tmp}" "${run_root}/logs/preregister.stdout.json"
  else
    echo "error: preregistration failed; controller output retained at ${preregister_tmp}" >&2
    return 1
  fi
  ACTIVE_RUN_ROOT="${run_root}"
  INITIAL_STATE="$(read_initial_state "${run_root}")"

  for index in 0 1 2 3; do
    slot=$((index + 1))
    if run_slot "${run_root}" "${slot}" "${arms[$index]}" "${states[$index]}"; then
      :
    else
      case_rc=$?
      failed_slots=$((failed_slots + 1))
      if (( case_rc == 75 )); then
        unsafe_stop=1
        if (( slot < 4 )); then
          skip_remaining_slots "${run_root}" "$((slot + 1))"
        fi
        break
      fi
      echo "warning: slot ${slot} failed after cleanup; continuing the preregistered schedule" >&2
    fi
  done

  if ! restore_initial_config; then
    die "failed to restore initial config; use recover after the GPU is clean"
  fi
  if summarize_run "${run_root}" > "${run_root}/logs/summarize.stdout.json"; then
    summary_rc=0
  else
    summary_rc=$?
  fi
  echo "run_root=${run_root}"
  echo "initial_config_state=${INITIAL_STATE}"
  echo "restored=true"
  echo "failed_slots=${failed_slots}"
  echo "summary=${run_root}/DENSE_FP8_DTTD_SUMMARY.json"
  if (( unsafe_stop != 0 || failed_slots != 0 || summary_rc != 0 )); then
    return 1
  fi
}

recover_run() {
  local run_root initial_state bundle_dir expected_sha actual_sha journal state_output recovered_state
  [[ $# -eq 1 ]] || die "recover requires RUN_ROOT"
  preflight_control
  run_root="$(realpath "$1")"
  require_file "${run_root}/run_manifest.json"
  require_file "${run_root}/initial_config_state.json"
  bundle_dir="${run_root}/bundle_snapshot"
  require_dir "${bundle_dir}"
  require_file "${bundle_dir}/manifest.json"
  if ! gpu_has_no_compute_process; then
    die "GPU has a compute process; refusing to switch dense configs"
  fi
  initial_state="$(read_initial_state "${run_root}")"
  expected_sha="$(jq -er '.bundle.manifest_sha256' "${run_root}/run_manifest.json")"
  actual_sha="$(sha256sum "${bundle_dir}/manifest.json" | awk '{print $1}')"
  [[ "${actual_sha}" == "${expected_sha}" ]] || die "bundle snapshot manifest SHA differs"
  journal="${run_root}/transitions/recover_to_${initial_state}_$(date -u +%Y%m%dT%H%M%SZ)_$$.json"
  state_output="${run_root}/states/recover_after_$(date -u +%Y%m%dT%H%M%SZ)_$$.json"
  require_absent "${journal}"
  require_absent "${state_output}"
  controller transition \
    --bundle-dir "${bundle_dir}" \
    --kt-root "${KT_ROOT}" \
    --target "${initial_state}" \
    --journal "${journal}"
  controller config-state \
    --bundle-dir "${bundle_dir}" \
    --kt-root "${KT_ROOT}" \
    --output "${state_output}" >/dev/null
  recovered_state="$(jq -er '.state' "${state_output}")"
  [[ "${recovered_state}" == "${initial_state}" ]] \
    || die "recovery did not restore ${initial_state}"
  echo "recovered_state=${recovered_state}"
  echo "receipt=${journal}"
  echo "state=${state_output}"
}

show_help() {
  cat <<'EOF'
Usage:
  ALLOW_GPU_RUN=1 run_kt_dense_fp8_ablation.sh run RUN_ROOT BUNDLE_DIR
  run_kt_dense_fp8_ablation.sh recover RUN_ROOT
  run_kt_dense_fp8_ablation.sh summarize RUN_ROOT

`run` preregisters all four cold-process slots before the first server and
executes default-tuned-tuned-default at frozen exact/t12, c8/output64, and
KV=13,104. Every config transition is hash-journaled. The EXIT trap restores
the initial default/tuned state only after compute processes are gone. If a
server cannot be cleaned, no subsequent config switch occurs; use `recover`
after the GPU becomes idle. This ablation screens KT dense configs only and is
not a custom-vs-KT speedup sample.
EOF
}

COMMAND="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${COMMAND}" in
  run)
    run_ablation "$@"
    ;;
  recover)
    recover_run "$@"
    ;;
  summarize)
    [[ $# -eq 1 ]] || die "summarize requires RUN_ROOT"
    summarize_run "$(realpath "$1")"
    ;;
  help|-h|--help)
    show_help
    ;;
  *)
    show_help >&2
    die "unknown command: ${COMMAND}"
    ;;
esac
