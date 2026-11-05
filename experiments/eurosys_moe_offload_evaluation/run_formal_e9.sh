#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KT_HARNESS="${REPO_ROOT}/experiments/broad_offload_compare/run_kt_qualification.sh"
VERIFIER="${SCRIPT_DIR}/verify_formal_e9.py"
PYTHON="${REPO_ROOT}/KTransformers/.venv/bin/python"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-${SCRIPT_DIR}/artifacts/e9-kt-REPRO-20260828-attempt-01}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/e9-formal-$(date -u +%Y%m%dT%H%M%SZ)}"
PROCESS_BLOCKS="${PROCESS_BLOCKS:-5}"
MODE="${MODE:-formal}"
PORT="${PORT:-30000}"

die() { echo "E9 formal: $*" >&2; exit 1; }
[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
case "${MODE}" in
  qualification) minimum_blocks=1 ;;
  formal) minimum_blocks=5 ;;
  *) die "MODE must be qualification or formal" ;;
esac
[[ "${PROCESS_BLOCKS}" =~ ^[0-9]+$ ]] && (( PROCESS_BLOCKS >= minimum_blocks )) \
  || die "PROCESS_BLOCKS must be >=${minimum_blocks} for MODE=${MODE}"
for path in \
  "${KT_HARNESS}" "${VERIFIER}" "${PYTHON}" \
  "${CALIBRATION_ROOT}/fixtures/evaluation_fixture.json" \
  "${CALIBRATION_ROOT}/fixtures/calibration_fixture.json" \
  "${CALIBRATION_ROOT}/fixtures/fixture_metadata.json" \
  "${CALIBRATION_ROOT}/calibration/frequency_stats.pt" \
  "${CALIBRATION_ROOT}/calibration/frequency_stats_manifest.json"; do
  [[ -f "${path}" ]] || die "missing required input: ${path}"
done
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"

"${PYTHON}" - "${RUN_ROOT}/run_config.json" "${PROCESS_BLOCKS}" "${MODE}" \
  "${CALIBRATION_ROOT}" <<'PY'
import hashlib, json, os, sys, time
path, blocks, mode, calibration = sys.argv[1:]
arms = {
    "baseline": {"threads": 14, "graph": True, "placement": "frequency", "overlap": True, "mixed": True},
    "threads7": {"threads": 7, "graph": True, "placement": "frequency", "overlap": True, "mixed": True},
    "threads28": {"threads": 28, "graph": True, "placement": "frequency", "overlap": True, "mixed": True},
    "eager": {"threads": 14, "graph": False, "placement": "frequency", "overlap": True, "mixed": True},
    "uniform": {"threads": 14, "graph": True, "placement": "uniform", "overlap": True, "mixed": True},
    "no_overlap": {"threads": 14, "graph": True, "placement": "frequency", "overlap": False, "mixed": True},
    "no_mixed": {"threads": 14, "graph": True, "placement": "frequency", "overlap": True, "mixed": False},
}
frequency = os.path.realpath(os.path.join(calibration, "calibration", "frequency_stats.pt"))
payload = {
    "schema": "eurosys_e9_ktransformers_formal_v1",
    "created_at": time.time(),
    "mode": mode,
    "process_blocks": int(blocks),
    "arms": arms,
    "block_order": "rotating forward/reverse",
    "calibration_root": os.path.realpath(calibration),
    "frequency_stats_sha256": hashlib.sha256(open(frequency, "rb").read()).hexdigest(),
    "concurrencies": [1, 8],
    "prompt_tokens": [300, 320],
    "output_tokens": 32,
    "numa_scope": "single-node host; cross-NUMA arm is N/A",
}
open(path, "x", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

sha256sum \
  "${BASH_SOURCE[0]}" "${KT_HARNESS}" "${VERIFIER}" \
  "${CALIBRATION_ROOT}/fixtures/evaluation_fixture.json" \
  "${CALIBRATION_ROOT}/fixtures/calibration_fixture.json" \
  "${CALIBRATION_ROOT}/calibration/frequency_stats.pt" \
  "${CALIBRATION_ROOT}/calibration/frequency_stats_manifest.json" \
  > "${RUN_ROOT}/input_sha256.txt"
git -C "${REPO_ROOT}" status --short > "${RUN_ROOT}/git_status.txt"
printf 'block\tposition\tarm\tcase_label\treturncode\n' > "${RUN_ROOT}/execution_status.tsv"

arms=(baseline threads7 threads28 eager uniform no_overlap no_mixed)
arm_count="${#arms[@]}"
for ((block=1; block<=PROCESS_BLOCKS; ++block)); do
  for ((position=0; position<arm_count; ++position)); do
    if (( block % 2 == 1 )); then
      arm_index=$(( (block - 1 + position) % arm_count ))
    else
      arm_index=$(( (block - 1 - position + arm_count * 4) % arm_count ))
    fi
    arm="${arms[arm_index]}"
    threads=14
    graph=0
    placement=frequency
    disable_overlap=0
    mixed=1
    case "${arm}" in
      baseline) ;;
      threads7) threads=7 ;;
      threads28) threads=28 ;;
      eager) graph=1 ;;
      uniform) placement=uniform ;;
      no_overlap) disable_overlap=1 ;;
      no_mixed) mixed=0 ;;
      *) die "unknown arm: ${arm}" ;;
    esac
    case_label="b$(printf '%02d' "${block}")_p$(printf '%02d' "$((position + 1))")_${arm}"
    command_file="${RUN_ROOT}/${case_label}.command.txt"
    printf '%s\n' \
      "RUN_ROOT=${RUN_ROOT}" \
      "CASE_LABEL=${case_label}" \
      "QUAL_DISABLE_CUDA_GRAPH=${graph}" \
      "QUAL_PLACEMENT_STRATEGY=${placement}" \
      "QUAL_DISABLE_OVERLAP_SCHEDULE=${disable_overlap}" \
      "QUAL_ENABLE_MIXED_CHUNK=${mixed}" \
      "bash ${KT_HARNESS} run-case exact ${threads}" > "${command_file}"
    set +e
    RUN_ROOT="${RUN_ROOT}" \
    CASE_LABEL="${case_label}" \
    PORT="${PORT}" \
    EVALUATION_FIXTURE_OVERRIDE="${CALIBRATION_ROOT}/fixtures/evaluation_fixture.json" \
    CALIBRATION_FIXTURE_OVERRIDE="${CALIBRATION_ROOT}/fixtures/calibration_fixture.json" \
    FREQUENCY_STATS_OVERRIDE="${CALIBRATION_ROOT}/calibration/frequency_stats.pt" \
    FREQUENCY_STATS_MANIFEST_OVERRIDE="${CALIBRATION_ROOT}/calibration/frequency_stats_manifest.json" \
    QUAL_DISABLE_CUDA_GRAPH="${graph}" \
    QUAL_PLACEMENT_STRATEGY="${placement}" \
    QUAL_DISABLE_OVERLAP_SCHEDULE="${disable_overlap}" \
    QUAL_ENABLE_MIXED_CHUNK="${mixed}" \
    QUAL_CONCURRENCIES=1,8 \
    QUAL_OUTPUT_TOKENS=32 \
    WARM_CONCURRENCY=8 \
    WARM_OUTPUT_TOKENS=8 \
      bash "${KT_HARNESS}" run-case exact "${threads}" \
      > "${RUN_ROOT}/${case_label}.stdout.log" 2>&1
    rc=$?
    set -e
    printf '%d\t%d\t%s\t%s\t%d\n' \
      "${block}" "$((position + 1))" "${arm}" "${case_label}" "${rc}" \
      >> "${RUN_ROOT}/execution_status.tsv"
  done
done

"${PYTHON}" "${VERIFIER}" "${RUN_ROOT}" --output "${RUN_ROOT}/verification.json"
echo "RUN_ROOT=${RUN_ROOT}"
