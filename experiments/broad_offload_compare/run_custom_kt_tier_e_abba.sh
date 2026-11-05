#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTROL="${SCRIPT_DIR}/custom_kt_tier_e_abba.py"
CUSTOM_RUNNER="${SCRIPT_DIR}/run_custom_comparison.sh"
KT_RUNNER="${SCRIPT_DIR}/run_kt_qualification.sh"
TUNING_CONTROL="${SCRIPT_DIR}/kt_dense_fp8_tuning.py"
DENSE_STATE_CONTROL="${SCRIPT_DIR}/kt_dense_fp8_ablation.py"
KT_ROOT="${REPO_ROOT}/KTransformers"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
EVALUATION_FIXTURE="${EVALUATION_FIXTURE:-${SCRIPT_DIR}/../eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json}"
WARMUP_FIXTURE="${WARMUP_FIXTURE:-${SCRIPT_DIR}/../eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/calibration_fixture_token_ids.json}"
CUSTOM_PLAN="${CUSTOM_PLAN:-${SCRIPT_DIR}/../eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/plans/kt_exact_custom_remote_plan.json}"
FREQUENCY_STATS="${FREQUENCY_STATS:-${SCRIPT_DIR}/../eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/routing/frequency_stats.pt}"
STATS_MANIFEST="${STATS_MANIFEST:-${SCRIPT_DIR}/../eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/routing/frequency_stats_manifest.json}"
QUALIFICATION_SNAPSHOT="${QUALIFICATION_SNAPSHOT:-}"

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

controller() {
  python3 "${CONTROL}" "$@"
}

static_check() {
  require_file "${CONTROL}"
  require_file "${CUSTOM_RUNNER}"
  require_file "${KT_RUNNER}"
  require_file "${TUNING_CONTROL}"
  require_file "${DENSE_STATE_CONTROL}"
  command -v python3 >/dev/null || die "python3 is unavailable"
  command -v rg >/dev/null || die "rg is unavailable"
  bash -n "${BASH_SOURCE[0]}"
  python3 -m py_compile "${CONTROL}"
  python3 "${CONTROL}" --help >/dev/null
  rg -q 'FORMAL_RUN' "${CUSTOM_RUNNER}" \
    || die "custom formal-run contract is missing"
  rg -q 'run-case\)' "${CUSTOM_RUNNER}" \
    || die "custom run-case contract is missing"
  rg -q 'run-case exact 12' "${KT_RUNNER}" \
    || die "KT exact/t12 run-case contract is missing"
  rg -q 'audit-log' "${TUNING_CONTROL}" \
    || die "dense warning audit contract is missing"
  echo "custom-vs-KT Tier-E static check: OK (CPU-only; no CUDA query)"
}

preregister_run() {
  local run_root="$1"
  local dense_bundle="$2"
  [[ -n "${QUALIFICATION_SNAPSHOT}" ]] \
    || die "QUALIFICATION_SNAPSHOT must point to a fresh qualification result"
  require_file "${QUALIFICATION_SNAPSHOT}"
  controller preregister \
    --run-root "${run_root}" \
    --model "${MODEL}" \
    --evaluation-fixture "${EVALUATION_FIXTURE}" \
    --warmup-fixture "${WARMUP_FIXTURE}" \
    --plan "${CUSTOM_PLAN}" \
    --frequency-stats "${FREQUENCY_STATS}" \
    --stats-manifest "${STATS_MANIFEST}" \
    --qualification-snapshot "${QUALIFICATION_SNAPSHOT}" \
    --dense-bundle-dir "${dense_bundle}" \
    --kt-root "${KT_ROOT}"
}

mark_remaining_skipped() {
  local run_root="$1"
  local detail="$2"
  local index slot_id concurrency block slot arm framework profile slot_root artifact_root
  while IFS=$'\t' read -r index slot_id concurrency block slot arm framework profile slot_root artifact_root; do
    controller mark --run-root "${run_root}" --slot-id "${slot_id}" \
      --status skipped --returncode 78 --detail "${detail}" >/dev/null
  done < <(tail -n +2 "${run_root}/schedule.tsv")
}

run_custom_slot() {
  local run_root="$1"
  local slot_id="$2"
  local concurrency="$3"
  local slot_root="$4"
  mkdir -p "${slot_root}/outer_logs"
  set +e
  env \
    ALLOW_GPU_RUN=1 \
    PROFILE=custom_remote \
    MODEL_FAMILY=qwen \
    MODEL="${MODEL}" \
    EVAL_FIXTURE="${run_root}/inputs/evaluation_fixture.json" \
    WARMUP_FIXTURE="${run_root}/inputs/warmup_fixture.json" \
    PLAN="${run_root}/inputs/custom_plan.json" \
    RUN_ROOT="${slot_root}" \
    CONCURRENCY="${concurrency}" \
    OUTPUT_TOKENS=64 \
    WARMUP_REQUESTS=8 \
    WARMUP_OUTPUT_TOKENS=8 \
    KV_CACHE_BYTES=1288490189 \
    MAX_MODEL_LEN=512 \
    MAX_NUM_BATCHED_TOKENS=4096 \
    MAX_NUM_SEQS=32 \
    GPU_MEMORY_UTILIZATION=0.97 \
    TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-3600}" \
    ROUTING_TRACE=0 \
    FORMAL_RUN=1 \
    REQUIRE_PROMPT_TOKEN_IDS=1 \
    CPUSET=0-13 \
    CPU_THREADS=12 \
    GPU_CLEAN_MIB="${GPU_CLEAN_MIB:-128}" \
    bash "${CUSTOM_RUNNER}" run-case \
      > "${slot_root}/outer_logs/custom_runner.stdout.log" \
      2> "${slot_root}/outer_logs/custom_runner.stderr.log"
  local rc=$?
  set -e
  printf '%s\n' "${rc}" > "${slot_root}/outer_logs/custom_runner.returncode.txt"
  return "${rc}"
}

run_kt_slot() {
  local run_root="$1"
  local concurrency="$2"
  local slot_root="$3"
  local artifact_root="$4"
  local qualification_rc=0
  local audit_rc=0
  local state_rc=0
  mkdir -p "${slot_root}/outer_logs"

  if ! python3 "${DENSE_STATE_CONTROL}" config-state \
      --bundle-dir "${run_root}/inputs/dense_bundle" \
      --kt-root "${KT_ROOT}" \
      --output "${slot_root}/kt_config_state_before.json" >/dev/null; then
    return 78
  fi
  if [[ "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "${slot_root}/kt_config_state_before.json")" != "tuned" ]]; then
    return 78
  fi

  set +e
  env \
    RUN_ROOT="${slot_root}/kt_runner" \
    CASE_LABEL=formal \
    PORT="${KT_PORT:-30000}" \
    CPUSET=0-13 \
    GPU_CLEAN_MIB="${GPU_CLEAN_MIB:-128}" \
    GPU_EXPERTS_RATIO=0.77083333333333337 \
    KV_TOKENS=13104 \
    MAX_CONTEXT_LENGTH=512 \
    MAX_RUNNING_REQUESTS=32 \
    QUAL_CHUNKED_PREFILL_SIZE=4096 \
    QUAL_MAX_PREFILL_TOKENS=4096 \
    QUAL_CONCURRENCIES="${concurrency}" \
    QUAL_OUTPUT_TOKENS=64 \
    WARM_CONCURRENCY=8 \
    WARM_OUTPUT_TOKENS=8 \
    REQUIRE_PROMPT_TOKEN_IDS=1 \
    EVALUATION_FIXTURE_OVERRIDE="${run_root}/inputs/evaluation_fixture.json" \
    CALIBRATION_FIXTURE_OVERRIDE="${run_root}/inputs/warmup_fixture.json" \
    FREQUENCY_STATS_OVERRIDE="${run_root}/inputs/frequency_stats.pt" \
    FREQUENCY_STATS_MANIFEST_OVERRIDE="${run_root}/inputs/frequency_stats_manifest.json" \
    bash "${KT_RUNNER}" run-case exact 12 \
      > "${slot_root}/outer_logs/kt_runner.stdout.log" \
      2> "${slot_root}/outer_logs/kt_runner.stderr.log"
  qualification_rc=$?
  set -e
  printf '%s\n' "${qualification_rc}" \
    > "${slot_root}/outer_logs/kt_runner.returncode.txt"

  if [[ -f "${artifact_root}/server.log" ]]; then
    set +e
    python3 "${TUNING_CONTROL}" audit-log \
      --log "${artifact_root}/server.log" \
      --expect tuned \
      --output "${artifact_root}/dense_fp8_warning_gate.json" \
      > "${slot_root}/outer_logs/dense_warning_audit.stdout.json"
    audit_rc=$?
    set -e
  else
    audit_rc=76
  fi

  set +e
  python3 "${DENSE_STATE_CONTROL}" config-state \
    --bundle-dir "${run_root}/inputs/dense_bundle" \
    --kt-root "${KT_ROOT}" \
    --output "${slot_root}/kt_config_state_after.json" >/dev/null
  state_rc=$?
  set -e
  if (( qualification_rc != 0 )); then
    return "${qualification_rc}"
  fi
  if (( audit_rc != 0 )); then
    return "${audit_rc}"
  fi
  if (( state_rc != 0 )); then
    return "${state_rc}"
  fi
  return 0
}

execute_run() {
  [[ "${ALLOW_GPU_RUN:-0}" == "1" ]] \
    || die "run requires ALLOW_GPU_RUN=1 after tuning has fully completed"
  [[ $# -eq 2 ]] || die "run requires RUN_ROOT and completed DENSE_BUNDLE_DIR"
  local run_root dense_bundle initial_state
  run_root="$(realpath -m "$1")"
  dense_bundle="$(realpath "$2")"
  static_check >/dev/null
  [[ ! -e "${run_root}" ]] || die "RUN_ROOT already exists: ${run_root}"
  preregister_run "${run_root}" "${dense_bundle}" \
    > "${run_root}.preregister.stdout.json"

  set +e
  python3 "${DENSE_STATE_CONTROL}" config-state \
    --bundle-dir "${run_root}/inputs/dense_bundle" \
    --kt-root "${KT_ROOT}" \
    --output "${run_root}/states/initial_dense_config_state.json" >/dev/null
  local initial_state_rc=$?
  set -e
  initial_state="invalid"
  if (( initial_state_rc == 0 )); then
    initial_state="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "${run_root}/states/initial_dense_config_state.json")"
  fi
  if [[ "${initial_state}" != "tuned" ]]; then
    mark_remaining_skipped "${run_root}" \
      "dense bundle was preregistered, but the installed KT config state was not tuned"
    controller summarize --run-root "${run_root}" \
      > "${run_root}/logs/summarize.stdout.json" || true
    die "installed dense config state is ${initial_state}; no GPU case was launched"
  fi

  local index slot_id concurrency block slot arm framework profile slot_root artifact_root
  local runner_rc verify_rc slot_rc failed_slots=0
  while IFS=$'\t' read -r index slot_id concurrency block slot arm framework profile slot_root artifact_root; do
    controller mark --run-root "${run_root}" --slot-id "${slot_id}" \
      --status running --returncode 0 \
      --detail "launching preregistered ${profile} cold process" >/dev/null
    runner_rc=0
    if [[ "${arm}" == "A" ]]; then
      run_custom_slot "${run_root}" "${slot_id}" "${concurrency}" "${slot_root}" \
        || runner_rc=$?
    else
      run_kt_slot "${run_root}" "${concurrency}" "${slot_root}" "${artifact_root}" \
        || runner_rc=$?
    fi
    set +e
    controller verify-slot --run-root "${run_root}" --slot-id "${slot_id}" \
      --output "${slot_root}/cross_framework_integrity_gate.json" \
      > "${slot_root}/outer_logs/cross_framework_verify.stdout.json"
    verify_rc=$?
    set -e
    slot_rc="${runner_rc}"
    if (( slot_rc == 0 && verify_rc != 0 )); then
      slot_rc="${verify_rc}"
    fi
    if (( slot_rc == 0 )); then
      controller mark --run-root "${run_root}" --slot-id "${slot_id}" \
        --status success --returncode 0 \
        --detail "runner=0, cross-framework-integrity=0" >/dev/null
    else
      failed_slots=$((failed_slots + 1))
      controller mark --run-root "${run_root}" --slot-id "${slot_id}" \
        --status failed --returncode "${slot_rc}" \
        --detail "runner=${runner_rc}, cross-framework-integrity=${verify_rc}; failure retained" >/dev/null
      echo "warning: ${slot_id} failed; continuing the frozen schedule" >&2
    fi
  done < <(tail -n +2 "${run_root}/schedule.tsv")

  controller summarize --run-root "${run_root}" \
    > "${run_root}/logs/summarize.stdout.json"
  echo "run_root=${run_root}"
  echo "summary=${run_root}/CUSTOM_KT_TIER_E_SUMMARY.json"
  echo "failed_slots=${failed_slots}"
  echo "note=cross-framework quality evidence is a separate mandatory publication gate"
  (( failed_slots == 0 ))
}

summarize_run() {
  [[ $# -ge 1 && $# -le 2 ]] || die "summarize requires RUN_ROOT [QUALITY_GATE_JSON]"
  local -a args=(summarize --run-root "$1")
  if [[ $# -eq 2 ]]; then
    args+=(--quality-gate "$2")
  fi
  controller "${args[@]}"
}

show_help() {
  cat <<'EOF'
Usage:
  run_custom_kt_tier_e_abba.sh static-check
  ALLOW_GPU_RUN=1 run_custom_kt_tier_e_abba.sh run RUN_ROOT COMPLETED_DENSE_BUNDLE_DIR
  run_custom_kt_tier_e_abba.sh summarize RUN_ROOT [QUALITY_GATE_JSON]

`static-check` is CPU-only and does not query CUDA. `run` first validates and
snapshots a completed dense bundle, atomically preregisters all 36 c1/c4/c8
slots, requires the matching configs to already be installed in tuned state,
then calls the existing custom and KT run-case interfaces. It never changes
dense config files. Failed slots remain in the frozen denominator.
EOF
}

case "${COMMAND}" in
  static-check)
    [[ $# -eq 0 ]] || die "static-check takes no arguments"
    static_check
    ;;
  run)
    execute_run "$@"
    ;;
  summarize)
    summarize_run "$@"
    ;;
  help|-h|--help)
    show_help
    ;;
  *)
    show_help >&2
    die "unknown command: ${COMMAND}"
    ;;
esac
