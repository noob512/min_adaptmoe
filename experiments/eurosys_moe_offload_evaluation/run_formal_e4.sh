#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
E4_RUNNER="${SCRIPT_DIR}/benchmark_e4_concurrency.py"
NVML_MONITOR="${SCRIPT_DIR}/monitor_e4_nvml.py"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-${SCRIPT_DIR}/artifacts/e3-calibration-20260825T091247Z}"
PLAN="${CALIBRATION_ROOT}/calibration_assets/cold_decode_plan_2g.json"
NAMES="${CALIBRATION_ROOT}/calibration_assets/exact_parameter_names.txt"
FIXTURE="${FIXTURE:-${SCRIPT_DIR}/inputs/generated/eurosys-pa-v2-20260824/qwen3_30b_a3b_fp8/evaluation_short_dialogue.json}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/e4-formal-$(date -u +%Y%m%dT%H%M%SZ)}"
PROCESS_BLOCKS="${PROCESS_BLOCKS:-5}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
WARM_OUTPUT_TOKENS="${WARM_OUTPUT_TOKENS:-16}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-1000000000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-272}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
CONCURRENCIES="${CONCURRENCIES:-$(seq -s, 1 32)}"
CPUSET="${CPUSET:-0-13}"
TIMEOUT_SEC="${TIMEOUT_SEC:-7200}"
PCM_BIN="${PCM_BIN:-}"
SELECTED_BYTES=2264924160
FULL_BYTES=28991029248
BACKING_BYTES=94371840
CUSTOM_LOADED=0
MONITOR_PIDS=()

die() { echo "error: $*" >&2; exit 1; }

stop_monitors() {
  local pid
  for pid in "${MONITOR_PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${MONITOR_PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  MONITOR_PIDS=()
}

cleanup() {
  stop_monitors
  if [[ "${CUSTOM_LOADED}" == 1 ]]; then
    restore_stock || true
    CUSTOM_LOADED=0
    restart_exporters || true
  fi
}
trap cleanup EXIT INT TERM

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
[[ "${PROCESS_BLOCKS}" =~ ^[1-9][0-9]*$ ]] || die "invalid PROCESS_BLOCKS"
for path in "${VLLM_PY}" "${E4_RUNNER}" "${NVML_MONITOR}" \
  "${MODEL}/config.json" "${PLAN}" "${NAMES}" "${FIXTURE}"; do
  [[ -f "${path}" ]] || die "missing file: ${path}"
done
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"
mkdir -p "${RUN_ROOT}/provenance" "${RUN_ROOT}/capability" "${RUN_ROOT}/blocks"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
# shellcheck source=/dev/null
source "${BASE_HARNESS}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY
RUN_ROOT_ACTIVE="${RUN_ROOT}"
runtime_preflight || die "custom runtime preflight failed"

sha256sum "${PLAN}" "${NAMES}" "${FIXTURE}" "${MODEL}/config.json" \
  "${E4_RUNNER}" "${NVML_MONITOR}" >"${RUN_ROOT}/provenance/inputs.sha256"
git -C "${REPO_ROOT}" rev-parse HEAD >"${RUN_ROOT}/provenance/repo.commit"
git -C "${REPO_ROOT}" status --short >"${RUN_ROOT}/provenance/repo.status.txt"
nvidia-smi -q >"${RUN_ROOT}/provenance/nvidia-smi-q.txt"
lscpu >"${RUN_ROOT}/provenance/lscpu.txt"
uname -a >"${RUN_ROOT}/provenance/uname.txt"
systemd-detect-virt >"${RUN_ROOT}/provenance/virtualization.txt" 2>&1 || true
cp "${PLAN}" "${RUN_ROOT}/provenance/cold_decode_plan_2g.json"
cp "${CALIBRATION_ROOT}/calibration_assets/manifest.json" \
  "${RUN_ROOT}/provenance/calibration_manifest.json"
nvidia-smi --query-gpu=pci.bus_id,pcie.link.gen.max,pcie.link.width.max \
  --format=csv,noheader >"${RUN_ROOT}/provenance/pcie_link.txt"

if [[ -n "${PCM_BIN}" && -x "${PCM_BIN}" ]]; then
  set +e
  sudo -n timeout 4s "${PCM_BIN}" 1 -csv \
    >"${RUN_ROOT}/capability/pcm-memory.stdout" \
    2>"${RUN_ROOT}/capability/pcm-memory.stderr"
  pcm_rc=$?
  set +e
  printf '%s\n' "${pcm_rc}" >"${RUN_ROOT}/capability/pcm-memory.returncode"
else
  printf '%s\n' "PCM_BIN unavailable" >"${RUN_ROOT}/capability/pcm-memory.stderr"
  printf '%s\n' "127" >"${RUN_ROOT}/capability/pcm-memory.returncode"
fi
jq -n --arg virtualization "$(systemd-detect-virt 2>/dev/null || echo unknown)" \
  --arg pcm_commit "${PCM_COMMIT:-unknown}" \
  --arg reason "KVM does not expose Intel IMC uncore PMUs or server uncore PCI configuration space; PCM reports no memory controllers" \
  '{status:"N/A",metric:"aggregate_cpu_dram_read_write_bandwidth",virtualization:$virtualization,pcm_commit:$pcm_commit,reason:$reason,lower_bound:"GPU PCIe RX is a lower bound on host-memory traffic for the remote-read path"}' \
  >"${RUN_ROOT}/capability/cpu_dram_bandwidth.json"

printf 'block\torder\tcase_dir\tstatus\treturncode\n' >"${RUN_ROOT}/execution.tsv"
module_snapshot "${RUN_ROOT}/provenance/module.before.json" "${STOCK_UVM}"
load_custom_module || die "could not load custom UVM"
CUSTOM_LOADED=1
module_snapshot "${RUN_ROOT}/provenance/module.custom.json" "${CUSTOM_UVM}"

IFS=',' read -r -a configured_concurrencies <<<"${CONCURRENCIES}"
ascending=("${configured_concurrencies[@]}")
descending=()
for ((index=${#ascending[@]}-1; index>=0; --index)); do
  descending+=("${ascending[$index]}")
done
expected_phase_count="${#ascending[@]}"

for ((block=0; block<PROCESS_BLOCKS; ++block)); do
  if (( block % 2 == 0 )); then
    order=asc
    scan=("${ascending[@]}")
  else
    order=desc
    scan=("${descending[@]}")
  fi
  label="b${block}_${order}"
  case_dir="${RUN_ROOT}/blocks/${label}"
  mkdir "${case_dir}"
  scan_csv="$(IFS=,; echo "${scan[*]}")"
  printf '%s\n' "${order}" >"${case_dir}/scan_order.txt"
  printf '%s\n' "${scan_csv}" >"${case_dir}/concurrencies.txt"
  : >"${case_dir}/manifest.jsonl"

  wait_gpu_clean || die "GPU not clean before ${label}"
  "${VLLM_PY}" "${NVML_MONITOR}" --output "${case_dir}/nvml.csv" \
    --interval-ms 200 >"${case_dir}/nvml.stdout" 2>"${case_dir}/nvml.stderr" &
  MONITOR_PIDS+=("$!")
  nvidia-smi dmon --gpm-metrics 1,2,3,10,20,21 -d 1 -o DT \
    >"${case_dir}/gpm_dmon.txt" 2>"${case_dir}/gpm_dmon.stderr" &
  MONITOR_PIDS+=("$!")
  sar -u ALL 1 >"${case_dir}/cpu_sar.txt" 2>"${case_dir}/cpu_sar.stderr" &
  MONITOR_PIDS+=("$!")

  set +e
  env VLLM_ATTENTION_BACKEND=TRITON_ATTN VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_CPU_OFFLOAD_BACKEND=copy VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${NAMES}" \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1 VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER=0 \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="${PLAN}" \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288 \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=4k \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPECTED_ARENA_COUNT=4 \
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPECTED_BACKING_BYTES="${BACKING_BYTES}" \
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 VLLM_UVM_ORIGINAL_VA_PHASE_MODE=remote_only \
    VLLM_UVM_ORIGINAL_VA_PHASE_RESULT_FILE="${case_dir}/transition.json" \
    VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ=0 VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0 \
    VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0 VLLM_CPU_OFFLOAD_COPY_TRACE=0 \
    VLLM_CPU_OFFLOAD_COPY_SYNC=0 VLLM_CPU_OFFLOAD_MANIFEST_FILE="${case_dir}/manifest.jsonl" \
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 VLLM_UVM_LOG_FILE="${case_dir}/uvm.log" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 CUDA_VISIBLE_DEVICES=0 \
    timeout --kill-after=30s "${TIMEOUT_SEC}s" taskset -c "${CPUSET}" \
      "${VLLM_PY}" "${E4_RUNNER}" --model "${MODEL}" --fixture "${FIXTURE}" \
      --exact-names "${NAMES}" --expert-plan "${PLAN}" \
      --offload-manifest "${case_dir}/manifest.jsonl" --output "${case_dir}/result.json" \
      --progress "${case_dir}/progress.jsonl" --concurrencies "${scan_csv}" \
      --warm-concurrency 32 --warm-output-tokens "${WARM_OUTPUT_TOKENS}" \
      --output-tokens "${OUTPUT_TOKENS}" --warmup-decode-steps 5 \
      --kv-cache-memory-bytes "${KV_CACHE_BYTES}" --cpu-offload-gb 30.0 \
      --max-model-len "${MAX_MODEL_LEN}" --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
      --gpu-memory-utilization 0.97 --request-rotation "$((block * 7))" --seed 0 \
      --expected-parameter-count 96 --expected-full-parameter-bytes "${FULL_BYTES}" \
      --expected-selected-bytes "${SELECTED_BYTES}" \
    >"${case_dir}/run.log" 2>&1
  rc=$?
  set +e
  stop_monitors
  status=error
  if (( rc == 0 )) && jq -e --argjson n "${OUTPUT_TOKENS}" --argjson count "${expected_phase_count}" '
      .status == "ok"
      and (.phases | length == $count)
      and (.phases | all(.output_tokens_per_request == $n and .preemptions == 0))
      and .placement_audit.selected_expert_count == 480
      and .placement_audit.selected_parameter_bytes_per_forward == 2264924160
      and .placement_audit.remote_physical_residency.cpu_bytes == 2264924160
      and .placement_audit.remote_physical_residency.gpu_bytes == 0
      and .shutdown_contract.diagnostics_after_shutdown.driver.session_count == 0
    ' "${case_dir}/result.json" >/dev/null 2>&1; then
    status=ok
  fi
  printf '%s\n' "${rc}" >"${case_dir}/returncode.txt"
  printf '%s\t%s\t%s\t%s\t%s\n' "${block}" "${order}" "${label}" "${status}" "${rc}" \
    >>"${RUN_ROOT}/execution.tsv"
  wait_gpu_clean || echo dirty >"${case_dir}/gpu_cleanup_error.txt"
done

restore_stock
CUSTOM_LOADED=0
restart_exporters
module_snapshot "${RUN_ROOT}/provenance/module.after.json" "${STOCK_UVM}"
echo "RUN_ROOT=${RUN_ROOT}"
