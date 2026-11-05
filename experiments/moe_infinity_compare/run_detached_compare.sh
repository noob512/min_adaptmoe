#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLLM_ROOT="${ROOT}/workloads/vllm/vllm"
PY="${VLLM_ROOT}/.venv/bin/python"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
DATASET="${DATASET:-${ROOT}/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RUN_ROOT="${RUN_ROOT:-/tmp/moe_infinity_compare_$(date +%Y%m%d_%H%M%S)}"
MOE_ROOT="${ROOT}/MoE-Infinity"
export PYTHONPATH="${MOE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/logs"

{
  echo "run_root=${RUN_ROOT}"
  echo "model_id=${MODEL_ID:-${MODEL}}"
  echo "model=${MODEL}"
  echo "dataset=${DATASET}"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "moe_infinity_commit=$(git -C "${MOE_ROOT}" rev-parse HEAD 2>/dev/null || true)"
  echo "current_repo_commit=$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || true)"
} > "${RUN_ROOT}/metadata.env"

export MODEL DATASET RUN_ROOT
export RUN_CASES="${RUN_CASES:-pilot,adaptive}"
export REQUEST_COUNT="${REQUEST_COUNT:-256}"
export PILOT_REQUEST_COUNT="${PILOT_REQUEST_COUNT:-${REQUEST_COUNT}}"
export PROMPT_MIN_TOKENS="${PROMPT_MIN_TOKENS:-320}"
export PROMPT_MAX_TOKENS="${PROMPT_MAX_TOKENS:-360}"
export OUTPUT_TOKENS="${OUTPUT_TOKENS:-64}"
export PHASE_NAMES="${PHASE_NAMES:-low,stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8,stage9}"
export PHASE_COUNTS="${PHASE_COUNTS:-8,64,96,128,160,192,224,256,256,256}"
export PHASE_OUTPUTS="${PHASE_OUTPUTS:-64,64,64,64,64,64,64,64,64,64}"
export TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"

set +e
bash "${VLLM_ROOT}/uvm_test/run_vllm_online_hot_cold_kv10g_scale.sh" \
  > "${RUN_ROOT}/logs/current_vllm_runner.log" 2>&1
current_rc="$?"
set -e
printf '%s\n' "${current_rc}" > "${RUN_ROOT}/current_vllm_returncode.txt"

set +e
"${PY}" "${ROOT}/experiments/moe_infinity_compare/moe_infinity_phased_benchmark.py" \
  --model "${MODEL}" \
  --fixture "${RUN_ROOT}/sharegpt_256_prompt320_360.json" \
  --output "${RUN_ROOT}/moe_infinity/result.json" \
  --progress "${RUN_ROOT}/moe_infinity/progress.jsonl" \
  --offload-dir "${RUN_ROOT}/moe_infinity/offload" \
  --device-memory-ratio "${MOE_INFINITY_DEVICE_MEMORY_RATIO:-0.75}" \
  --phase-names "${MOE_INFINITY_PHASE_NAMES:-final}" \
  --phase-request-counts "${MOE_INFINITY_PHASE_COUNTS:-256}" \
  --phase-output-tokens "${MOE_INFINITY_PHASE_OUTPUTS:-64}" \
  ${MOE_INFINITY_USE_NATIVE_ENGINE:+--use-native-engine} \
  ${MOE_INFINITY_PREFETCH:+--prefetch} \
  > "${RUN_ROOT}/logs/moe_infinity_runner.log" 2>&1
moe_rc="$?"
set -e
printf '%s\n' "${moe_rc}" > "${RUN_ROOT}/moe_infinity_returncode.txt"

"${PY}" - "${RUN_ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
summary = {
    'run_root': str(root),
    'current_vllm_returncode': (root / 'current_vllm_returncode.txt').read_text().strip() if (root / 'current_vllm_returncode.txt').exists() else None,
    'moe_infinity_returncode': (root / 'moe_infinity_returncode.txt').read_text().strip() if (root / 'moe_infinity_returncode.txt').exists() else None,
    'current_vllm_summary': read_json(root / 'summary.json'),
    'moe_infinity_result': read_json(root / 'moe_infinity' / 'result.json'),
}

current_cases = (summary['current_vllm_summary'] or {}).get('cases', {})
current_case = current_cases.get('adaptive', {})
current_phases = current_case.get('phases', [])
moe_result = summary['moe_infinity_result'] or {}
moe_phases = moe_result.get('phases', [])
current_final = current_phases[-1] if current_phases else {}
moe_final = moe_phases[-1] if moe_phases else {}
current_tps = current_final.get('output_tokens_per_second')
moe_tps = moe_final.get('output_tokens_per_second')
current_elapsed = current_final.get('elapsed_seconds')
moe_elapsed = moe_final.get('elapsed_seconds')
current_load = current_case.get('load_seconds')
moe_load = moe_result.get('load_seconds')

def divide(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator

summary['comparison'] = {
    'fixture_scope': {
        'dataset': 'ShareGPT',
        'request_count': 64,
        'prompt_tokens': '320-360',
        'output_tokens_per_request': 32,
        'same_fixture': True,
    },
    'scheduling_scope': {
        'current_vllm': '64 requests generated as one batched high phase',
        'moe_infinity': '64 requests generated sequentially with MoE.generate()',
        'same_batching_model': False,
        'interpretation': 'The result compares the exercised end-to-end mechanisms, but batching contributes materially to the throughput gap.',
    },
    'current_vllm_case': 'adaptive',
    'current_vllm_final_phase': current_final,
    'moe_infinity_final_phase': moe_final,
    'metrics': {
        'current_vllm_load_seconds': current_load,
        'moe_infinity_cached_offload_load_seconds': moe_load,
        'load_time_ratio_moe_infinity_over_current_vllm': divide(moe_load, current_load),
        'current_vllm_output_tokens_per_second': current_tps,
        'moe_infinity_output_tokens_per_second': moe_tps,
        'throughput_ratio_moe_infinity_over_current_vllm': divide(moe_tps, current_tps),
        'throughput_speedup_current_vllm_over_moe_infinity': divide(current_tps, moe_tps),
        'current_vllm_generation_seconds': current_elapsed,
        'moe_infinity_generation_seconds': moe_elapsed,
        'generation_time_ratio_moe_infinity_over_current_vllm': divide(moe_elapsed, current_elapsed),
    },
}
(root / 'comparison_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2, sort_keys=True))
PY
