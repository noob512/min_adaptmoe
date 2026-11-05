#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${REPO_ROOT}/workloads/vllm/vllm/uvm_test/run_vllm_online_hot_cold_kv10g_scale.sh"
SUMMARIZER="${SCRIPT_DIR}/summarize_sharegpt_concurrency_snapshot.py"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
DATASET="${DATASET:-${REPO_ROOT}/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RUN_ROOT="${RUN_ROOT:-/tmp/qwen_native_uva_sharegpt_concurrency_$(date +%Y%m%d_%H%M%S)}"
OFFLOAD_GIB_ORDER="${OFFLOAD_GIB_ORDER:-2,18,6,14,10}"
CONCURRENCIES="${CONCURRENCIES:-1,2,3,4,8,16,24,32}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-64}"
WARM_OUTPUT_TOKENS="${WARM_OUTPUT_TOKENS:-8}"
PROMPT_MIN_TOKENS="${PROMPT_MIN_TOKENS:-300}"
PROMPT_MAX_TOKENS="${PROMPT_MAX_TOKENS:-320}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-1288490189}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"
SELECTED_DATASET="${RUN_ROOT}/sharegpt_human_first_selected.json"
SELECTION_METADATA="${RUN_ROOT}/sharegpt_human_first_selection_metadata.json"

IFS=',' read -r -a OFFLOAD_LEVELS <<< "${OFFLOAD_GIB_ORDER}"
IFS=',' read -r -a ASC_CONCURRENCIES <<< "${CONCURRENCIES}"
if [[ "${#OFFLOAD_LEVELS[@]}" -eq 0 || "${#ASC_CONCURRENCIES[@]}" -eq 0 ]]; then
  echo "OFFLOAD_GIB_ORDER and CONCURRENCIES must not be empty" >&2
  exit 2
fi

MAX_CONCURRENCY=0
for concurrency in "${ASC_CONCURRENCIES[@]}"; do
  if [[ ! "${concurrency}" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid concurrency: ${concurrency}" >&2
    exit 2
  fi
  if (( concurrency > MAX_CONCURRENCY )); then
    MAX_CONCURRENCY="${concurrency}"
  fi
done
for offload_gib in "${OFFLOAD_LEVELS[@]}"; do
  if [[ ! "${offload_gib}" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid offload GiB: ${offload_gib}" >&2
    exit 2
  fi
done
if (( PROMPT_MAX_TOKENS + OUTPUT_TOKENS > MAX_MODEL_LEN )); then
  echo "prompt + output exceeds MAX_MODEL_LEN" >&2
  exit 2
fi
if [[ ! -s "${DATASET}" ]]; then
  echo "missing ShareGPT dataset: ${DATASET}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}"

prepare_selected_dataset() {
  if [[ -s "${SELECTED_DATASET}" && -s "${SELECTION_METADATA}" ]]; then
    return 0
  fi
  (
    cd "${REPO_ROOT}/workloads/vllm/vllm"
    .venv/bin/python - "${DATASET}" "${MODEL}" "${SELECTED_DATASET}" \
      "${SELECTION_METADATA}" "${MAX_CONCURRENCY}" \
      "${PROMPT_MIN_TOKENS}" "${PROMPT_MAX_TOKENS}" <<'PY'
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

source_path = Path(sys.argv[1])
model = sys.argv[2]
output_path = Path(sys.argv[3])
metadata_path = Path(sys.argv[4])
request_count = int(sys.argv[5])
minimum = int(sys.argv[6])
maximum = int(sys.argv[7])

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
source = json.loads(source_path.read_text(encoding="utf-8"))
selected = []
source_indices = []
prompt_lengths = []
roles = []
for source_index, item in enumerate(source):
    conversations = item.get("conversations") if isinstance(item, dict) else None
    if not isinstance(conversations, list) or len(conversations) < 2:
        continue
    first = conversations[0]
    if not isinstance(first, dict):
        continue
    role = str(first.get("from", "")).strip().lower()
    if role not in {"human", "user"}:
        continue
    prompt = first.get("value")
    if not isinstance(prompt, str) or not prompt:
        continue
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if not minimum <= prompt_tokens <= maximum:
        continue
    selected.append(item)
    source_indices.append(source_index)
    prompt_lengths.append(prompt_tokens)
    roles.append(role)
    if len(selected) == request_count:
        break

if len(selected) != request_count:
    raise SystemExit(f"expected {request_count} human-first requests, found {len(selected)}")

metadata = {
    "source": str(source_path),
    "selection": "first_length_matched_human_or_user_first_turn",
    "request_count": request_count,
    "source_indices": source_indices,
    "roles": roles,
    "prompt_token_min": min(prompt_lengths),
    "prompt_token_max": max(prompt_lengths),
    "prompt_token_mean": sum(prompt_lengths) / len(prompt_lengths),
    "prompt_token_total": sum(prompt_lengths),
}
output_path.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
metadata_path.write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(metadata, indent=2, sort_keys=True))
PY
  )
}

join_by_comma() {
  local IFS=','
  echo "$*"
}

wait_for_clean_gpu() {
  local used_mib
  for _ in $(seq 1 180); do
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] && [[ "${used_mib}" -le 128 ]]; then
      return 0
    fi
    sleep 2
  done
  echo "GPU did not become clean: ${used_mib} MiB" >&2
  nvidia-smi >&2 || true
  return 1
}

build_phases() {
  local order="$1"
  local -n names_ref="$2"
  local -n counts_ref="$3"
  local -n outputs_ref="$4"
  local ordered=()

  if [[ "${order}" == "asc" ]]; then
    ordered=("${ASC_CONCURRENCIES[@]}")
  else
    for ((index=${#ASC_CONCURRENCIES[@]} - 1; index >= 0; index--)); do
      ordered+=("${ASC_CONCURRENCIES[index]}")
    done
  fi

  names_ref=("warm_c${MAX_CONCURRENCY}")
  counts_ref=("${MAX_CONCURRENCY}")
  outputs_ref=("${WARM_OUTPUT_TOKENS}")
  for concurrency in "${ordered[@]}"; do
    names_ref+=("sharegpt_c${concurrency}")
    counts_ref+=("${concurrency}")
    outputs_ref+=("${OUTPUT_TOKENS}")
  done
}

run_one() {
  local pair_index="$1"
  local requested_gib="$2"
  local label="$3"
  local actual_offload_gib="$4"
  local order="$5"
  local case_root="${RUN_ROOT}/pair${pair_index}_g${requested_gib}_${label}"
  local monitor_pid=""
  local phase_names=()
  local phase_counts=()
  local phase_outputs=()

  build_phases "${order}" phase_names phase_counts phase_outputs
  mkdir -p "${case_root}"
  printf '%s\n' "${order}" > "${case_root}/scan_order.txt"
  printf '%s\n' "${requested_gib}" > "${case_root}/requested_offload_gib.txt"
  wait_for_clean_gpu
  nvidia-smi dmon -s t -d 1 -o DT > "${case_root}/dmon_pcie.csv" 2>&1 &
  monitor_pid=$!

  echo "RUN_BEGIN pair=${pair_index} requested_gib=${requested_gib} label=${label} order=${order}"
  set +e
  RUN_ROOT="${case_root}" \
  MODEL="${MODEL}" \
  DATASET="${SELECTED_DATASET}" \
  REQUEST_COUNT="${MAX_CONCURRENCY}" \
  PROMPT_MIN_TOKENS="${PROMPT_MIN_TOKENS}" \
  PROMPT_MAX_TOKENS="${PROMPT_MAX_TOKENS}" \
  OUTPUT_TOKENS="${OUTPUT_TOKENS}" \
  FIXTURE_OUTPUT_TOKENS="${OUTPUT_TOKENS}" \
  LARGE_KV_BYTES="${KV_CACHE_BYTES}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
  MAX_NUM_SEQS="${MAX_CONCURRENCY}" \
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
  GPU_MEMORY_UTILIZATION=0.97 \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  NATIVE_CPU_OFFLOAD_GB="${actual_offload_gib}" \
  NATIVE_PHASE_NAMES="$(join_by_comma "${phase_names[@]}")" \
  NATIVE_PHASE_COUNTS="$(join_by_comma "${phase_counts[@]}")" \
  NATIVE_PHASE_OUTPUTS="$(join_by_comma "${phase_outputs[@]}")" \
  TIMEOUT_SEC="${TIMEOUT_SEC}" \
  RUN_CASES=native_uva \
  bash "${RUNNER}" > "${case_root}/orchestrator.log" 2>&1
  local runner_rc=$?
  set -e

  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true

  local case_rc=missing
  if [[ -f "${case_root}/native_uva/returncode.txt" ]]; then
    case_rc="$(tr -d '[:space:]' < "${case_root}/native_uva/returncode.txt")"
  fi
  echo "RUN_END pair=${pair_index} requested_gib=${requested_gib} label=${label} runner_rc=${runner_rc} case_rc=${case_rc}"
  if [[ "${runner_rc}" -ne 0 || "${case_rc}" != "0" ]]; then
    tail -n 160 "${case_root}/orchestrator.log" >&2 || true
    return 1
  fi
  wait_for_clean_gpu
}

pair_index=0
prepare_selected_dataset
for requested_gib in "${OFFLOAD_LEVELS[@]}"; do
  pair_index=$((pair_index + 1))
  if (( pair_index % 2 == 1 )); then
    scan_order=asc
    run_one "${pair_index}" "${requested_gib}" baseline 0.0 "${scan_order}"
    run_one "${pair_index}" "${requested_gib}" offload "${requested_gib}.0" "${scan_order}"
  else
    scan_order=desc
    run_one "${pair_index}" "${requested_gib}" offload "${requested_gib}.0" "${scan_order}"
    run_one "${pair_index}" "${requested_gib}" baseline 0.0 "${scan_order}"
  fi
done

"${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python" \
  "${SUMMARIZER}" "${RUN_ROOT}" |
  tee "${RUN_ROOT}/snapshot_summary.stdout.json"

echo "RUN_ROOT=${RUN_ROOT}"
