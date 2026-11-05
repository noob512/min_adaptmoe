#!/usr/bin/env bash
# Fresh four-arm replay; no dependency on historical artifacts or measurements.
set -euo pipefail

compare_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
compare_repo="$(cd "${compare_dir}/../.." && pwd)"
if [[ "${1:-}" == --help || $# == 0 ]]; then
  echo 'Usage: run_kv1_compare.sh NEW_OUTPUT_DIRECTORY [QPS ...]'
  echo 'Default: QPS=1, SAMPLES=1. PREPARE_ONLY=1 prepares inputs without GPU use.'
  echo 'GPU execution requires ALLOW_GPU_RUN=1, MODEL, and SHAREGPT_DATASET.'
  exit 0
fi
compare_output="$1"
shift
compare_rates=("$@")
if (( ${#compare_rates[@]} == 0 )); then compare_rates=(1); fi
compare_samples="${SAMPLES:-1}"
compare_prepare="${PREPARE_ONLY:-0}"
export VLLM_PY="${VLLM_PY:-${compare_repo}/workloads/vllm/vllm/.venv/bin/python}"
[[ "${compare_samples}" == 1 || "${compare_samples}" == 2 ]] || { echo 'SAMPLES must be 1 or 2' >&2; exit 1; }
[[ "${compare_prepare}" == 0 || "${compare_prepare}" == 1 ]] || { echo 'PREPARE_ONLY must be 0 or 1' >&2; exit 1; }
[[ -x "${VLLM_PY}" ]] || { echo "Missing Python: ${VLLM_PY}" >&2; exit 1; }
[[ ! -e "${compare_output}" ]] || { echo "Output already exists: ${compare_output}" >&2; exit 1; }
declare -A compare_seen=()
for rate in "${compare_rates[@]}"; do
  case "${rate}" in 1|2|4|8|10|20|30) ;; *) echo "Unsupported QPS: ${rate}" >&2; exit 1 ;; esac
  [[ -z "${compare_seen[${rate}]:-}" ]] || { echo "Duplicate QPS: ${rate}" >&2; exit 1; }
  compare_seen["${rate}"]=1
done
if [[ "${compare_prepare}" == 0 ]]; then
  [[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || { echo 'Set ALLOW_GPU_RUN=1' >&2; exit 1; }
  : "${MODEL:?Set MODEL to the local DeepSeek-V2-Lite-Chat snapshot}"
  : "${SHAREGPT_DATASET:?Set SHAREGPT_DATASET to ShareGPT_V3_unfiltered_cleaned_split.json}"
fi
mkdir -p "${compare_output}"
compare_output="$(cd "${compare_output}" && pwd)"
for rate in "${compare_rates[@]}"; do
  mkdir "${compare_output}/qps${rate}"
  "${VLLM_PY}" "${compare_dir}/five_arm_k1_compare.py" \
    "${compare_output}/qps${rate}/inputs" --prepare \
    >"${compare_output}/qps${rate}/prepare.log"
done
"${VLLM_PY}" - "${compare_output}" "${compare_samples}" "${compare_dir}" "${compare_rates[@]}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root, samples, source, *rates = sys.argv[1:]
root, source = Path(root), Path(source)
fixture = source / 'inputs/sharegpt_deepseek_p1024_n128_20261024.json'
protocol = {
    'qps': sorted(map(int, rates)), 'new_qps': list(map(int, rates)),
    'reused_qps': [], 'samples_per_new_arm': int(samples),
    'roots': {q: str(root / f'qps{q}') for q in rates},
    'report_stem': 'ADAPTMOE_KV1_LATEST',
    'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
    'capacity_certificate': json.loads((root / f'qps{rates[0]}/inputs/capacity_certificate.json').read_text()),
}
assert protocol['fixture_sha256'] == 'a4cc14060c94efb8c86f0524f2c49fbf85c0ff284eb12636e1e5c133d85bb9f2'
(root / 'sweep_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
PY
if [[ "${compare_prepare}" == 1 ]]; then
  echo "Prepared inputs only: ${compare_output}"
  exit 0
fi
unset ADAPT_RUNTIME_SETTINGS_FILE
for rate in "${compare_rates[@]}"; do
  rate_root="${compare_output}/qps${rate}"
  echo "RATE_BEGIN QPS=${rate} samples=${compare_samples}"
  env COMPARE_RUNNER_OVERRIDE="${compare_dir}/five_arm_k1_runner.py" \
    COMPARE_VERIFY_OVERRIDE="${compare_dir}/kv1_qps_sweep_compare.py" \
    KV1_SWEEP_EXPECTED_QPS="${rate}" \
    ADAPT_FIXED_RUNTIME_SETTINGS_FILE="${rate_root}/inputs/adaptive_fixed_1g_runtime_settings.json" \
    ADAPT_FIXED1_RUNTIME_SETTINGS_FILE="${rate_root}/inputs/adaptive_fixed_1g_runtime_settings.json" \
    ADAPT_DYNAMIC_RUNTIME_SETTINGS_FILE="${rate_root}/inputs/adaptive_dynamic_runtime_settings.json" \
    NATIVE_NAMES="${rate_root}/inputs/native_cold8_near_equal_names.txt" \
    NATIVE_EXPECTED_BYTES=3690987520 NATIVE_OFFLOAD_GB=4.0 \
    SAMPLES="${compare_samples}" ARRIVAL_QPS="${rate}" \
    COMPARE_ARMS='native_uva native_copy adaptive_fixed_1g adaptive_dynamic' \
    RUN_ROOT="${rate_root}/vllm" \
    FIXTURE="${compare_dir}/inputs/sharegpt_deepseek_p1024_n128_20261024.json" \
    INITIAL_REQUESTS=64 LATE_REQUESTS=64 LATE_AFTER_ENGINE_STEPS=4 OUTPUT_TOKENS=16 \
    MAX_MODEL_LEN=1088 MAX_NUM_BATCHED_TOKENS=8192 MAX_NUM_SEQS=128 \
    KV_CACHE_BYTES=1073741824 ADAPT_FIXED1_KV_CACHE_BYTES=2147483648 \
    ADAPT_DYNAMIC_KV_CACHE_BYTES=2147483648 \
    PARK_TOKEN_THRESHOLD=16 RESUME_TOKEN_THRESHOLD=128 NSYS_PROFILE=0 \
    bash "${compare_dir}/run_compare.sh" >"${rate_root}/run.log" 2>&1
  echo "RATE_END QPS=${rate}"
done
"${VLLM_PY}" "${compare_dir}/kv1_qps_sweep_compare.py" "${compare_output}"
echo "Report: ${compare_output}/REPORT.md"
