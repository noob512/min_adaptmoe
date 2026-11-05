#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
VLLM_PY="${VLLM_PY:-.venv/bin/python}"
DATASET="${DATASET:-/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_online_hot_cold_kv10g_scale_$(date +%Y%m%d_%H%M%S)}"

REQUEST_COUNT="${REQUEST_COUNT:-256}"
PILOT_REQUEST_COUNT="${PILOT_REQUEST_COUNT:-${REQUEST_COUNT}}"
PILOT_OUTPUT_TOKENS="${PILOT_OUTPUT_TOKENS:-${OUTPUT_TOKENS:-64}}"
PROMPT_MIN_TOKENS="${PROMPT_MIN_TOKENS:-320}"
PROMPT_MAX_TOKENS="${PROMPT_MAX_TOKENS:-360}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-64}"
FIXTURE_OUTPUT_TOKENS="${FIXTURE_OUTPUT_TOKENS:-${OUTPUT_TOKENS}}"
SMALL_KV_BYTES="${SMALL_KV_BYTES:-1288490189}"
LARGE_KV_BYTES="${LARGE_KV_BYTES:-10737418240}"
INITIAL_KV_BLOCKS="${INITIAL_KV_BLOCKS:-818}"
FINAL_KV_BLOCKS="${FINAL_KV_BLOCKS:-6825}"
COLD_EXPERTS_PER_LAYER="${COLD_EXPERTS_PER_LAYER:-45}"
EXPERT_EST_BYTES="${EXPERT_EST_BYTES:-4718592}"
MODEL_NUM_LAYERS="${MODEL_NUM_LAYERS:-48}"
MODEL_NUM_EXPERTS="${MODEL_NUM_EXPERTS:-128}"
FIRST_MOE_LAYER="${FIRST_MOE_LAYER:-0}"
NATIVE_CPU_OFFLOAD_GB="${NATIVE_CPU_OFFLOAD_GB:-10.0}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"
RUN_CASES="${RUN_CASES:-pilot,adaptive,static_custom,native_uva}"
MAINTENANCE_WAVES_PER_PHASE_LIST="${MAINTENANCE_WAVES_PER_PHASE_LIST:-}"

PHASE_NAMES="${PHASE_NAMES:-low,stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8,stage9}"
PHASE_COUNTS="${PHASE_COUNTS:-8,64,96,128,160,192,224,256,256,256}"
PHASE_OUTPUTS="${PHASE_OUTPUTS:-64,64,64,64,64,64,64,64,64,64}"
STATIC_PHASE_NAMES="${STATIC_PHASE_NAMES:-warm,final}"
STATIC_PHASE_COUNTS="${STATIC_PHASE_COUNTS:-256,256}"
STATIC_PHASE_OUTPUTS="${STATIC_PHASE_OUTPUTS:-1,64}"
NATIVE_PHASE_NAMES="${NATIVE_PHASE_NAMES:-warm,final}"
NATIVE_PHASE_COUNTS="${NATIVE_PHASE_COUNTS:-256,256}"
NATIVE_PHASE_OUTPUTS="${NATIVE_PHASE_OUTPUTS:-1,64}"

FIXTURE="${RUN_ROOT}/sharegpt_256_prompt320_360.json"
FIXTURE_META="${RUN_ROOT}/fixture_metadata.json"
PLAN="${RUN_ROOT}/hot_cold_plan_k45.json"
TARGETS_FILE="${RUN_ROOT}/kv_grow_targets.txt"
SUMMARY="${RUN_ROOT}/summary.json"

mkdir -p "${RUN_ROOT}"

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    head -n 1 | tr -d ' '
}

wait_for_clean_gpu() {
  local label="$1"
  local used=999999
  for _ in $(seq 1 180); do
    used="$(gpu_used_mib || echo 999999)"
    if [[ "${used}" =~ ^[0-9]+$ ]] && [[ "${used}" -le 1024 ]]; then
      return 0
    fi
    sleep 2
  done
  echo "GPU is not clean before ${label}: ${used} MiB" >&2
  nvidia-smi >&2 || true
  return 1
}

should_run_case() {
  local name="$1"
  [[ ",${RUN_CASES}," == *",${name},"* ]]
}

prepare_fixture() {
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - "${DATASET}" "${MODEL}" "${FIXTURE}" "${FIXTURE_META}" \
      "${REQUEST_COUNT}" "${PROMPT_MIN_TOKENS}" "${PROMPT_MAX_TOKENS}" \
      "${FIXTURE_OUTPUT_TOKENS}" <<'PY'
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

dataset_path = Path(sys.argv[1])
model = sys.argv[2]
fixture_path = Path(sys.argv[3])
metadata_path = Path(sys.argv[4])
request_count = int(sys.argv[5])
min_tokens = int(sys.argv[6])
max_tokens = int(sys.argv[7])
fixture_output_tokens = int(sys.argv[8])

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
source = json.loads(dataset_path.read_text(encoding="utf-8"))
requests = []
source_indices = []
prompt_lengths = []
for index, item in enumerate(source):
    conversations = item.get("conversations") if isinstance(item, dict) else None
    if not isinstance(conversations, list) or not conversations:
        continue
    first = conversations[0]
    prompt = first.get("value") if isinstance(first, dict) else None
    if not isinstance(prompt, str) or not prompt:
        continue
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if not min_tokens <= prompt_tokens <= max_tokens:
        continue
    requests.append({
        "request": len(requests),
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "max_tokens": fixture_output_tokens,
    })
    source_indices.append(index)
    prompt_lengths.append(prompt_tokens)
    if len(requests) == request_count:
        break

if len(requests) != request_count:
    raise SystemExit(f"expected {request_count} requests, found {len(requests)}")

metadata = {
    "source": str(dataset_path),
    "selection": "first_length_matched",
    "source_indices": source_indices,
    "request_count": len(requests),
    "total_prompt_tokens": sum(prompt_lengths),
    "mean_prompt_tokens": sum(prompt_lengths) / len(prompt_lengths),
    "min_prompt_tokens": min(prompt_lengths),
    "max_prompt_tokens": max(prompt_lengths),
}
fixture_path.write_text(
    json.dumps({"metadata": metadata, "requests": requests}, indent=2, sort_keys=True)
    + "\n",
    encoding="utf-8",
)
metadata_path.write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(metadata, indent=2, sort_keys=True))
PY
  )
}

generate_targets() {
  "${VLLM_ROOT}/${VLLM_PY}" - "${INITIAL_KV_BLOCKS}" "${FINAL_KV_BLOCKS}" \
    "${COLD_EXPERTS_PER_LAYER}" "${TARGETS_FILE}" <<'PY'
import sys
from pathlib import Path

initial = int(sys.argv[1])
final = int(sys.argv[2])
waves = int(sys.argv[3])
path = Path(sys.argv[4])
targets = [initial + round((final - initial) * wave / waves) for wave in range(1, waves + 1)]
if targets[-1] != final or any(a >= b for a, b in zip(targets, targets[1:])):
    raise SystemExit(f"invalid targets: {targets}")
path.write_text(",".join(str(value) for value in targets) + "\n", encoding="utf-8")
print(path.read_text().strip())
PY
}

run_case() {
  local name="$1"
  local mode="$2"
  local kv_bytes="$3"
  local phase_names="$4"
  local phase_counts="$5"
  local phase_outputs="$6"
  local dir="${RUN_ROOT}/${name}"
  mkdir -p "${dir}"
  rm -f \
    "${dir}/result.json" \
    "${dir}/progress.jsonl" \
    "${dir}/run.log" \
    "${dir}/returncode.txt" \
    "${dir}/routing_trace.jsonl" \
    "${dir}/cpu_offload_manifest.jsonl"
  wait_for_clean_gpu "${name}"

  local -a env_args=(
    VLLM_USE_UVM=0
    VLLM_UVM_COLD_SLOT_POOL_DRIVER_MODE=slot_va_block_remap
    VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP=1
    VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0
    VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0
  )
  local -a extra_args=()
  local cpu_offload_gb=0

  case "${mode}" in
    pilot)
      env_args+=(
        VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
        VLLM_UVM_KV_CACHE_MANAGED=1
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
        VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
        VLLM_UVM_KV_RUNTIME_ENABLE=0
        VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0
        VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=1
        VLLM_UVM_MOE_ROUTING_TRACE_FILE="${dir}/routing_trace.jsonl"
      )
      ;;
    adaptive|static_custom)
      env_args+=(
        VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
        VLLM_UVM_KV_CACHE_MANAGED=1
        VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1
        VLLM_UVM_EXPERT_ENVELOPE_BYTES=524288
        VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=remote_read
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup
        VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${PLAN}"
        VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN=1
        VLLM_UVM_WEIGHT_EXECUTOR=cpp_batch
        VLLM_UVM_KV_RUNTIME_ENABLE=1
        VLLM_UVM_KV_RUNTIME_MODE=trace_only
        VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=1
        VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS="${INITIAL_KV_BLOCKS}"
        VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_ENABLE=1
        VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_EXECUTOR=cpp_batch
        VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_SYNC=1
      )
      if [[ "${mode}" == "adaptive" ]]; then
        env_args+=(
          VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=1
          VLLM_UVM_INCREMENTAL_EXPERTS_PER_LAYER_PER_ROUND=1
        )
        extra_args=(
          --grow-target-blocks "$(<"${TARGETS_FILE}")"
          --release-expert-wave
        )
        if [[ -n "${MAINTENANCE_WAVES_PER_PHASE_LIST}" ]]; then
          extra_args+=(
            --maintenance-waves-per-phase-list
            "${MAINTENANCE_WAVES_PER_PHASE_LIST}"
          )
        else
          extra_args+=(--maintenance-waves-per-phase 5)
        fi
      else
        env_args+=(VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0)
        extra_args=(--pre-phase-target-blocks "${FINAL_KV_BLOCKS}")
      fi
      ;;
    native_uva)
      cpu_offload_gb="${NATIVE_CPU_OFFLOAD_GB}"
      env_args+=(
        VLLM_CPU_OFFLOAD_BACKEND=uva
        VLLM_CPU_OFFLOAD_MANIFEST_FILE="${dir}/cpu_offload_manifest.jsonl"
        VLLM_CPU_OFFLOAD_MANIFEST_RESET=1
        VLLM_UVM_EXPERT_WEIGHTS_MANAGED=0
        VLLM_UVM_KV_CACHE_MANAGED=0
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
        VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
        VLLM_UVM_KV_RUNTIME_ENABLE=0
        VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0
        VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0
      )
      ;;
    *)
      echo "unknown mode: ${mode}" >&2
      return 2
      ;;
  esac

  set +e
  (
    cd "${VLLM_ROOT}"
    env "${env_args[@]}" timeout "${TIMEOUT_SEC}s" "${VLLM_PY}" \
      "${SCRIPT_DIR}/vllm_generate_phased.py" \
      --model "${MODEL}" \
      --fixture "${FIXTURE}" \
      --output "${dir}/result.json" \
      --progress "${dir}/progress.jsonl" \
      --phase-names "${phase_names}" \
      --phase-request-counts "${phase_counts}" \
      --phase-output-tokens "${phase_outputs}" \
      --kv-cache-memory-bytes "${kv_bytes}" \
      --cpu-offload-gb "${cpu_offload_gb}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      "${extra_args[@]}"
  ) > "${dir}/run.log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "${rc}" > "${dir}/returncode.txt"
  wait_for_clean_gpu "after_${name}" || true
  return "${rc}"
}

build_plan() {
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" "${SCRIPT_DIR}/build_hot_cold_expert_plan.py" \
      --routing-trace "${RUN_ROOT}/pilot/routing_trace.jsonl" \
      --output "${PLAN}" \
      --num-layers "${MODEL_NUM_LAYERS}" \
      --num-experts "${MODEL_NUM_EXPERTS}" \
      --first-moe-layer "${FIRST_MOE_LAYER}" \
      --cold-experts-per-layer "${COLD_EXPERTS_PER_LAYER}" \
      --expert-est-bytes "${EXPERT_EST_BYTES}"
  )
}

write_summary() {
  "${VLLM_ROOT}/${VLLM_PY}" - "${RUN_ROOT}" "${SUMMARY}" "${FIXTURE_META}" "${PLAN}" "${TARGETS_FILE}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
fixture_meta_path = Path(sys.argv[3])
plan_path = Path(sys.argv[4])
targets_path = Path(sys.argv[5])


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_case(name):
    case_dir = root / name
    result = read_json(case_dir / "result.json") or {}
    returncode = None
    returncode_path = case_dir / "returncode.txt"
    if returncode_path.exists():
        returncode = int(returncode_path.read_text().strip())
    phases = [
        {key: value for key, value in phase.items() if key != "requests"}
        for phase in result.get("phases", [])
    ]
    return {
        "returncode": returncode,
        "status": result.get("status"),
        "error": result.get("error"),
        "load_seconds": result.get("load_seconds"),
        "phases": phases,
        "maintenance": result.get("maintenance", []),
        "maintenance_total_seconds": sum(
            float(item.get("elapsed_seconds") or 0)
            for item in result.get("maintenance", [])
        ),
    }


cases = {}
for name in ("pilot", "adaptive", "static_custom", "native_uva"):
    if (root / name / "result.json").exists():
        cases[name] = summarize_case(name)

if "native_uva" in cases:
    parameters = [
        record for record in read_jsonl(root / "native_uva" / "cpu_offload_manifest.jsonl")
        if record.get("event") == "cpu_offload_parameter"
    ]
    cases["native_uva"]["cpu_offload_parameter_records"] = len(parameters)
    cases["native_uva"]["cpu_offload_bytes"] = sum(
        int(record.get("bytes") or 0) for record in parameters
    )

summary = {
    "run_root": str(root),
    "fixture": read_json(fixture_meta_path),
    "plan": (read_json(plan_path) or {}).get("metadata", {}),
    "kv_grow_targets": [
        int(value) for value in (targets_path.read_text().split(",") if targets_path.exists() else [])
        if value.strip()
    ],
    "cases": cases,
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

prepare_fixture
generate_targets

if should_run_case pilot; then
  run_case pilot pilot "${SMALL_KV_BYTES}" pilot \
    "${PILOT_REQUEST_COUNT}" "${PILOT_OUTPUT_TOKENS}" || true
fi

needs_plan=0
if should_run_case pilot || should_run_case adaptive || should_run_case static_custom; then
  needs_plan=1
fi
if [[ "${needs_plan}" -eq 1 ]] && [[ ! -f "${PLAN}" ]]; then
  if [[ ! -f "${RUN_ROOT}/pilot/routing_trace.jsonl" ]]; then
    echo "missing ${PLAN} and pilot/routing_trace.jsonl" >&2
    exit 1
  fi
  build_plan
fi

if should_run_case adaptive; then
  run_case adaptive adaptive "${LARGE_KV_BYTES}" \
    "${PHASE_NAMES}" "${PHASE_COUNTS}" "${PHASE_OUTPUTS}" || true
fi
if should_run_case static_custom; then
  run_case static_custom static_custom "${LARGE_KV_BYTES}" \
    "${STATIC_PHASE_NAMES}" "${STATIC_PHASE_COUNTS}" \
    "${STATIC_PHASE_OUTPUTS}" || true
fi
if should_run_case native_uva; then
  run_case native_uva native_uva "${LARGE_KV_BYTES}" \
    "${NATIVE_PHASE_NAMES}" "${NATIVE_PHASE_COUNTS}" \
    "${NATIVE_PHASE_OUTPUTS}" || true
fi

write_summary
echo "summary=${SUMMARY}"
