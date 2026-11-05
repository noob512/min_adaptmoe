#!/usr/bin/env bash
set -euo pipefail
scan_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${scan_dir}/../.." && pwd)"
scan_root="${1:-${scan_dir}/artifacts/p512-c16-qps1-4-8-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ ! -e "${scan_root}" || "${RESUME_SCAN:-0}" == 1 ]] || { echo "Output already exists: ${scan_root}" >&2; exit 1; }
mkdir -p "${scan_root}"
scan_root="$(cd "${scan_root}" && pwd)"
if [[ ! -e "${scan_root}/scan_status.tsv" ]]; then
  printf 'qps\treturncode\n' >"${scan_root}/scan_status.tsv"
fi
for rate in 1 4 8; do
  if [[ -d "${scan_root}/qps${rate}" ]]; then
    [[ "${RESUME_SCAN:-0}" == 1 ]] || exit 1
    "${repo_dir}/workloads/vllm/vllm/.venv/bin/python" - "${scan_root}/qps${rate}/verification.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
assert value['status'] in {'ok', 'ok_with_output_drift'} and not value['errors']
assert len(value['cases']) == 6
PY
    printf '%s\t0\n' "${rate}" >>"${scan_root}/scan_status.tsv"
    echo "RATE_REUSED qps=${rate} completed verified results"
    continue
  fi
  echo "RATE_BEGIN qps=${rate}"
  if env ALLOW_GPU_RUN=1 SAMPLES=2 ARRIVAL_QPS="${rate}" \
    RUN_ROOT="${scan_root}/qps${rate}" \
    FIXTURE="${repo_dir}/experiments/eurosys_moe_offload_evaluation/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/evaluation_ordinary_chat.json" \
    INITIAL_REQUESTS=8 LATE_REQUESTS=8 LATE_AFTER_ENGINE_STEPS=4 \
    OUTPUT_TOKENS=16 MAX_MODEL_LEN=576 MAX_NUM_BATCHED_TOKENS=8192 \
    MAX_NUM_SEQS=16 KV_CACHE_BYTES=536870912 \
    PARK_TOKEN_THRESHOLD=16 RESUME_TOKEN_THRESHOLD=128 \
    bash "${scan_dir}/run_compare.sh" >"${scan_root}/qps${rate}.log" 2>&1; then
    printf '%s\t0\n' "${rate}" >>"${scan_root}/scan_status.tsv"
    echo "RATE_END qps=${rate} rc=0"
  else
    scan_rc=$?
    printf '%s\t%s\n' "${rate}" "${scan_rc}" >>"${scan_root}/scan_status.tsv"
    tail -n 100 "${scan_root}/qps${rate}.log" >&2
    exit "${scan_rc}"
  fi
done
"${repo_dir}/workloads/vllm/vllm/.venv/bin/python" \
  "${scan_dir}/summarize_p512_qps_scan.py" "${scan_root}"
echo "SCAN_ROOT=${scan_root}"
