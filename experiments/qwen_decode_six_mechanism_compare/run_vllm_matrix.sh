#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
RUNNER="${VLLM_ROOT}/uvm_test/vllm_decode_step_benchmark.py"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
FIXTURE="${FIXTURE:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/vllm_$(date -u +%Y%m%dT%H%M%SZ)}"
TIERS="${TIERS:-2g,4g}"
BACKENDS="${BACKENDS:-uva,copy}"
WARMUP_DECODE_STEPS="${WARMUP_DECODE_STEPS:-5}"
MEASURED_DECODE_STEPS="${MEASURED_DECODE_STEPS:-16}"

die() { echo "error: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }
gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | head -1 | tr -d '[:space:]'
}
compute_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk '$1 ~ /^[0-9]+$/ {print $1}'
}
uvm_clients() {
  sudo fuser /dev/nvidia-uvm /dev/nvidia-uvm-tools 2>/dev/null \
    | tr ' ' '\n' | awk '$1 ~ /^[0-9]+$/ {print $1}' || true
}
wait_gpu_clean() {
  local attempt
  for ((attempt=0; attempt<240; ++attempt)); do
    if [[ "$(gpu_used_mib)" == 0 && -z "$(compute_pids)" && -z "$(uvm_clients)" ]]; then
      return 0
    fi
    sleep 1
  done
  return 1
}
gpu_snapshot() {
  local output="$1"
  "${VLLM_PY}" - "${output}" "$(gpu_used_mib)" \
    "$(compute_pids | paste -sd, -)" "$(uvm_clients | paste -sd, -)" \
    "$(tr -d '[:space:]' </sys/module/nvidia_uvm/srcversion)" <<'PY'
import json, sys, time
path, used, compute, clients, module = sys.argv[1:]
json.dump({
    "timestamp": time.time(), "memory_used_mib": int(used),
    "compute_pids": [int(x) for x in compute.split(",") if x],
    "uvm_client_pids": [int(x) for x in clients.split(",") if x],
    "loaded_srcversion": module,
}, open(path, "w", encoding="utf-8"), indent=2, sort_keys=True)
open(path, "a", encoding="utf-8").write("\n")
PY
}

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
require_file "${RUNNER}"
require_file "${MODEL}/config.json"
require_file "${FIXTURE}"
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
wait_gpu_clean || die "GPU/UVM is not clean"
gpu_snapshot "${RUN_ROOT}/gpu.before.json"
printf 'tier\tbackend\trequested_gib\tdirectory\treturncode\n' \
  >"${RUN_ROOT}/execution.tsv"

IFS=',' read -r -a tier_list <<<"${TIERS}"
IFS=',' read -r -a backend_list <<<"${BACKENDS}"
for tier in "${tier_list[@]}"; do
  requested="${tier%g}.0"
  for backend in "${backend_list[@]}"; do
    [[ "${backend}" == uva || "${backend}" == copy ]] \
      || die "backend must be uva or copy"
    case_dir="${RUN_ROOT}/${tier}_${backend}"
    mkdir "${case_dir}"
    : >"${case_dir}/manifest.jsonl"
    wait_gpu_clean || die "GPU is not clean before ${tier}/${backend}"
    gpu_snapshot "${case_dir}/gpu.before.json"
    set +e
    (
      cd "${VLLM_ROOT}"
      env \
        VLLM_USE_UVM=0 \
        VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        VLLM_CPU_OFFLOAD_BACKEND="${backend}" \
        VLLM_CPU_OFFLOAD_MANIFEST_FILE="${case_dir}/manifest.jsonl" \
        VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 \
        VLLM_CPU_OFFLOAD_COPY_TRACE=0 \
        VLLM_CPU_OFFLOAD_COPY_SYNC=0 \
        VLLM_UVM_EXPERT_WEIGHTS_MANAGED=0 \
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
        VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
        VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0 \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
        OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 CUDA_VISIBLE_DEVICES=0 \
        timeout --kill-after=30s 1800s taskset -c 0-13 \
        "${VLLM_PY}" "${RUNNER}" \
          --model "${MODEL}" --fixture "${FIXTURE}" \
          --output "${case_dir}/result.json" \
          --progress "${case_dir}/progress.jsonl" --batch-sizes 1,4 \
          --warmup-decode-steps "${WARMUP_DECODE_STEPS}" \
          --measured-decode-steps "${MEASURED_DECODE_STEPS}" \
          --kv-cache-memory-bytes 167772160 --cpu-offload-gb "${requested}" \
          --max-model-len 512 --max-num-seqs 4 \
          --max-num-batched-tokens 2048 --gpu-memory-utilization 0.97
    ) >"${case_dir}/run.log" 2>&1
    rc=$?
    set -e
    echo "${rc}" >"${case_dir}/returncode.txt"
    wait_gpu_clean || die "GPU did not clean after ${tier}/${backend}"
    gpu_snapshot "${case_dir}/gpu.after.json"
    printf '%s\t%s\t%s\t%s\t%s\n' "${tier}" "${backend}" "${requested}" \
      "$(basename "${case_dir}")" "${rc}" >>"${RUN_ROOT}/execution.tsv"
    if (( rc != 0 )); then
      tail -n 240 "${case_dir}/run.log" >&2 || true
      die "vLLM case failed: ${tier}/${backend}"
    fi
    jq -e '.status == "ok"' "${case_dir}/result.json" >/dev/null \
      || die "vLLM result is not ok"
  done
done

gpu_snapshot "${RUN_ROOT}/gpu.after.json"
echo "RUN_ROOT=${RUN_ROOT}"
