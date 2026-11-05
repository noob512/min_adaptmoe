#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
RUNNER="${REPO_ROOT}/experiments/vllm_p3_pinned_uva_phase_compare/benchmark_phase_steps.py"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
FIXTURE="${FIXTURE:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/vllm_$(date -u +%Y%m%dT%H%M%SZ)}"
TIERS="${TIERS:-2g,4g}"
BACKENDS="${BACKENDS:-uva,copy}"
CONDITIONING_REPEATS="${CONDITIONING_REPEATS:-1}"
MEASUREMENT_REPEATS="${MEASUREMENT_REPEATS:-3}"
ACTIVE_SERVICES=()

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
wait_gpu_clean() {
  local attempt
  for ((attempt=0; attempt<240; ++attempt)); do
    if [[ "$(gpu_used_mib)" == 0 && -z "$(compute_pids)" ]]; then return 0; fi
    sleep 1
  done
  return 1
}
capture_and_stop_exporters() {
  local service
  for service in node-exporter.service gpu-exporter.service dcgm-exporter.service; do
    if [[ "$(systemctl is-active "${service}" 2>/dev/null || true)" == active ]]; then
      ACTIVE_SERVICES+=("${service}")
      sudo systemctl stop "${service}"
    fi
  done
}
restart_exporters() {
  local service
  for service in "${ACTIVE_SERVICES[@]}"; do sudo systemctl start "${service}" || true; done
}
trap restart_exporters EXIT INT TERM

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
require_file "${RUNNER}"
require_file "${MODEL}/config.json"
require_file "${FIXTURE}"
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
capture_and_stop_exporters
wait_gpu_clean || die "GPU is not initially clean"
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
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      >"${case_dir}/gpu.before.csv"
    sha256sum "${FIXTURE}" >"${case_dir}/input_sha256.txt"
    set +e
    (
      cd "${VLLM_ROOT}"
      env \
        VLLM_ATTENTION_BACKEND=TRITON_ATTN \
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
          --profile-label "qwen3_layer_order_${tier}_${backend}_prefill" \
          --mechanism-class "vllm_native_${backend}" --batch-sizes 1,4 \
          --conditioning-repeats "${CONDITIONING_REPEATS}" \
          --measurement-repeats "${MEASUREMENT_REPEATS}" \
          --warmup-decode-steps 1 --measured-decode-steps 1 \
          --kv-cache-memory-bytes 167772160 --cpu-offload-gb "${requested}" \
          --max-model-len 512 --max-num-seqs 4 \
          --max-num-batched-tokens 2048 --gpu-memory-utilization 0.97 \
          --seed 0
    ) >"${case_dir}/run.log" 2>&1
    rc=$?
    set -e
    echo "${rc}" >"${case_dir}/returncode.txt"
    wait_gpu_clean || die "GPU did not clean after ${tier}/${backend}"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      >"${case_dir}/gpu.after.csv"
    printf '%s\t%s\t%s\t%s\t%s\n' "${tier}" "${backend}" "${requested}" \
      "$(basename "${case_dir}")" "${rc}" >>"${RUN_ROOT}/execution.tsv"
    if (( rc != 0 )); then
      tail -n 240 "${case_dir}/run.log" >&2 || true
      die "vLLM case failed: ${tier}/${backend}"
    fi
    jq -e --arg backend "${backend}" '
      .status == "ok"
      and .config.enforce_eager == true
      and .config.batch_sizes == [1,4]
      and .phase_summary["1"].measurement_repeats == 3
      and .phase_summary["4"].measurement_repeats == 3
      and .environment.VLLM_CPU_OFFLOAD_BACKEND == $backend
    ' "${case_dir}/result.json" >/dev/null || die "vLLM result is not ok"
    "${VLLM_PY}" - "${case_dir}/manifest.jsonl" "${case_dir}/placement.json" <<'PY'
import json, re, sys
records=[json.loads(line) for line in open(sys.argv[1], encoding='utf-8') if line.strip()]
params=[x for x in records if x.get('event')=='cpu_offload_parameter']
layers=[]
for item in params:
    match=re.search(r'\.layers\.(\d+)\.', str(item.get('full_parameter_name','')))
    if match: layers.append(int(match.group(1)))
payload={
    'parameter_count': len(params),
    'realized_bytes': sum(int(x['bytes']) for x in params),
    'realized_gib': sum(int(x['bytes']) for x in params)/(1<<30),
    'layers_touched': sorted(set(layers)),
    'first_parameter': params[0]['full_parameter_name'],
    'last_parameter': params[-1]['full_parameter_name'],
    'backend': params[0]['offload_backend'],
}
open(sys.argv[2], 'w', encoding='utf-8').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
  done
done

restart_exporters
ACTIVE_SERVICES=()
trap - EXIT INT TERM
echo "RUN_ROOT=${RUN_ROOT}"
