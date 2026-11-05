#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_ROOT="${1:?new absolute run root required}"
[[ ! -e "${RUN_ROOT}" ]] || { echo "Run root already exists" >&2; exit 1; }
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
PY="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
"${PY}" "${SCRIPT_DIR}/build_pressure_wave.py" \
  --source "${SCRIPT_DIR}/../inputs/sharegpt_trace_20261013/sharegpt_qwen_trace_low.jsonl" \
  --output "${RUN_ROOT}/pressure_wave.jsonl" >"${RUN_ROOT}/trace_build.json"
export TRACE_FILE="${RUN_ROOT}/pressure_wave.jsonl" ARRIVAL_SCALE=1 CUDA_GRAPH=0
export MAX_NUM_SEQS=16 MAX_BATCHED=8192 WARMUP=4 WARMUP_SELECTION=trace_prefix
export REPLAY_MAX_CONNECTIONS=512 PHASE_F_ROUTE_LOG_ENABLE=0
# Native baselines retain the same peak KV capacity. Native uses whole
# Parameters selected by a fixed byte budget; actual bytes are audited later.
export KV_CACHE_BYTES=4295491584 NATIVE_OFFLOAD_GIB=5.90625
export ADAPT_DYNAMIC_ACTIVE_HIGH_WATERMARK_GROW=1
for arm in adaptmoe native_copy native_uva; do
  printf '%s START %s\n' "$(date -u +%FT%TZ)" "${arm}" | tee -a "${RUN_ROOT}/progress.log"
  if [[ "${arm}" == adaptmoe ]]; then
    bash "${SCRIPT_DIR}/run_k2_k4_cold28_qps2.sh" "${RUN_ROOT}/${arm}" >"${RUN_ROOT}/${arm}.launcher.log" 2>&1
  else
    bash "${SCRIPT_DIR}/run_sharegpt_trace_arm.sh" "${arm}" low "${RUN_ROOT}/${arm}" >"${RUN_ROOT}/${arm}.launcher.log" 2>&1
  fi
  printf '%s COMPLETE %s\n' "$(date -u +%FT%TZ)" "${arm}" | tee -a "${RUN_ROOT}/progress.log"
done
