#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
PREFILL_RUNNER="${SCRIPT_DIR}/e3_prefill_only_runner.py"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
FIXTURE_ROOT="${FIXTURE_ROOT:-${SCRIPT_DIR}/inputs/generated/eurosys-pa-v2-20260824/qwen3_30b_a3b_fp8}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-${SCRIPT_DIR}/artifacts/e3-calibration-20260825T091247Z}"
NAMES="${CALIBRATION_ROOT}/calibration_assets/exact_parameter_names.txt"
PLAN_10="${REPO_ROOT}/experiments/qwen_decode_six_mechanism_compare/inputs/cold_decode_plan_2g.json"
PLAN_38="${REPO_ROOT}/experiments/qwen_decode_six_mechanism_compare/inputs/cold_decode_plan_8g.json"
PLAN_63="${SCRIPT_DIR}/inputs/frozen/e3_qwen_cold63_20260828/cold_decode_plan_2g.json"
TOP_ROOT="${1:-${SCRIPT_DIR}/artifacts/e3-prefill-only-four-mechanism-$(date -u +%Y%m%dT%H%M%SZ)}"
CPUSET="${CPUSET:-0-13}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_ATTN}"
KV_BYTES_PER_TOKEN=98304
FULL_PARAMETER_BYTES=28991029248
CUSTOM_ARM_FILTER="${CUSTOM_ARM_FILTER:-graph_copy,remote_eager,alias_eager}"
RUN_CUSTOM="${RUN_CUSTOM:-1}"
RUN_NATIVE="${RUN_NATIVE:-1}"
MONITOR_PID=""
CUSTOM_LOADED=0

die() { echo "E3 Prefill-only: $*" >&2; exit 1; }

stop_monitor() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=""
  fi
}

cleanup_prefill() {
  stop_monitor
  if [[ "${CUSTOM_LOADED}" == 1 ]]; then
    restore_stock || true
    CUSTOM_LOADED=0
  fi
  restart_exporters || true
}

contains_csv() {
  local needle="$1" haystack=",${2},"
  [[ "${haystack}" == *",${needle},"* ]]
}

kv_bytes() {
  local prompt="$1" original_output="$2" batch="$3" rounded value
  rounded=$(( (prompt + original_output + 16 + 15) / 16 * 16 ))
  value=$((rounded * batch * KV_BYTES_PER_TOKEN))
  (( value < 167772160 )) && value=167772160
  echo "${value}"
}

cell_geometry() {
  local workload="$1" batch="$2"
  if [[ "${workload}" == rag_summary && "${batch}" == 8 ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${PLAN_63}" 14269022208 594542592 24 13.5 20480
  elif [[ "${workload}" == rag_summary || ( "${workload}" == medium_context && "${batch}" == 8 ) ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${PLAN_38}" 8606711808 358612992 14 7.875 "$([[ "${workload}" == rag_summary ]] && echo 8192 || echo 16384)"
  else
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${PLAN_10}" 2264924160 94371840 4 2.25 0
  fi
}

run_one() {
  local arm="$1" workload="$2" prompt="$3" original_output="$4" batch="$5"
  local plan selected backing native_layers native_gib reserve kv label case_dir
  IFS=$'\t' read -r plan selected backing native_layers native_gib reserve \
    < <(cell_geometry "${workload}" "${batch}")
  kv="$(kv_bytes "${prompt}" "${original_output}" "${batch}")"
  label="${workload}_b${batch}_${arm}"
  case_dir="${TOP_ROOT}/cases/${label}"
  mkdir "${case_dir}"
  : >"${case_dir}/manifest.jsonl"
  local exact_names="${NAMES}" expected_count=96 expected_full="${FULL_PARAMETER_BYTES}"
  local cpu_offload_gb=30.0 expert_args=(--expert-plan "${plan}")
  local alias=1 alias_eager=0 phase=remote_only coordinate=exact
  local graph_prefill=0 native_bytes=0

  case "${arm}" in
    graph_copy)
      phase=prefill_graph_remote_decode
      graph_prefill=1
      ;;
    remote_eager)
      phase=remote_only
      ;;
    alias_eager)
      phase=off
      alias_eager=1
      ;;
    native_copy_on_forward)
      alias=0
      phase=off
      coordinate=nearest_byte_full_parameter
      exact_names="${TOP_ROOT}/provenance/native_names_${native_layers}_layers.txt"
      expected_count=$((native_layers * 2))
      native_bytes=$((native_layers * 603979776))
      expected_full="${native_bytes}"
      selected="${native_bytes}"
      cpu_offload_gb="${native_gib}"
      expert_args=()
      ;;
    *) die "unknown arm: ${arm}" ;;
  esac

  # On this RTX 5090/TRITON_ATTN build, eager one-step Prefill at 8192 or
  # more scheduled tokens is a fail-closed backend limit.  The same frozen
  # boundary was already registered by E3 after descriptor-700/illegal-address
  # failures.  FULL_SEPARATE Prefill Graph is the only certified path above
  # this boundary; do not silently change the batch or chunk the requests.
  if (( prompt * batch >= 8192 && graph_prefill == 0 )); then
    jq -n --arg arm "${arm}" --arg coordinate "${coordinate}" \
      --argjson prefill_tokens "$((prompt * batch))" \
      --argjson selected_bytes "${selected}" --argjson kv_bytes "${kv}" \
      '{status:"N/A",arm:$arm,coordinate_contract:$coordinate,
        prefill_tokens:$prefill_tokens,selected_bytes:$selected_bytes,
        kv_bytes:$kv_bytes,
        reason:"fail-closed backend limit: eager one-step Prefill >=8192 tokens triggers TMA descriptor 700 / CUDA illegal-address on this RTX 5090/TRITON_ATTN runtime; batch, token geometry, and backend were not changed"}' \
      >"${case_dir}/na.json"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\tN/A\t-\t-\n' \
      "${workload}" "${prompt}" "${batch}" "${arm}" "${coordinate}" \
      "${selected}" "${kv}" >>"${TOP_ROOT}/execution.tsv"
    echo "N/A ${label} prefill_tokens=$((prompt * batch)) eager_backend_limit"
    return 0
  fi

  wait_gpu_clean || die "GPU is dirty before ${label}"
  module_snapshot "${case_dir}/module.before.json" "$([[ "${alias}" == 1 ]] && echo "${CUSTOM_UVM}" || echo "${STOCK_UVM}")"
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu \
    --format=csv,noheader,nounits -lms 100 >"${case_dir}/nvml.csv" 2>"${case_dir}/nvml.err" &
  MONITOR_PID="$!"

  # Preserve the frozen E3 context envelope even though this Prefill-only run
  # samples one token and stops.  Shrinking this to prompt+1 changes internal
  # warmup shapes and is not configuration-equivalent to the source matrix.
  local max_model=$((prompt + original_output))
  local max_batched=$((prompt * batch))
  (( max_batched < max_model )) && max_batched="${max_model}"
  local chunk=16384 autotune=0
  if (( prompt * batch >= 8192 )); then chunk=4096; autotune=4096; fi

  local -a environment=(
    VLLM_ATTENTION_BACKEND="${ATTENTION_BACKEND}"
    VLLM_ENABLE_V1_MULTIPROCESSING=0
    VLLM_CPU_OFFLOAD_BACKEND=copy
    VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${exact_names}"
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS="${alias}"
    VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER="${alias_eager}"
    VLLM_UVM_ORIGINAL_VA_PHASE_MODE="${phase}"
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED="${alias}"
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
    VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
    VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0
    VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0
    VLLM_CPU_OFFLOAD_COPY_TRACE=0
    VLLM_CPU_OFFLOAD_COPY_SYNC=0
    VLLM_CPU_OFFLOAD_MANIFEST_FILE="${case_dir}/manifest.jsonl"
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1
    VLLM_UVM_LOG_FILE="${case_dir}/uvm.log"
    VLLM_UVM_RUNTIME_HOT_RESIDENCY_AUDIT=0
    VLLM_FUSED_MOE_CHUNK_SIZE="${chunk}"
    VLLM_FLASHINFER_AUTOTUNE_MAX_TOKENS="${autotune}"
    HF_HUB_OFFLINE=1
    TRANSFORMERS_OFFLINE=1
    TOKENIZERS_PARALLELISM=false
    OMP_NUM_THREADS=14
    MKL_NUM_THREADS=14
    CUDA_VISIBLE_DEVICES=0
  )
  if [[ "${alias}" == 1 ]]; then
    environment+=(
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="${plan}"
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=4k
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPECTED_ARENA_COUNT=4
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPECTED_BACKING_BYTES="${backing}"
      VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_BYTES="${kv}"
      VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_EMPTY_CACHE=1
    )
  fi
  if [[ "${graph_prefill}" == 1 ]]; then
    environment+=(
      VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1
      VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE=1
      VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=1
      VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES=201326592
      VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS="${reserve}"
      VLLM_PHASE_SWITCH_CONTROLLED_HARD_EXIT_AFTER_CLEANUP=1
    )
  fi

  {
    printf 'cd %q && env ' "${REPO_ROOT}"
    printf '%q ' "${environment[@]}"
    printf 'timeout --kill-after=30s 3600s taskset -c %q %q %q ' \
      "${CPUSET}" "${VLLM_PY}" "${PREFILL_RUNNER}"
    printf '%q ' --arm "${arm}" --model "${MODEL}" \
      --fixture "${FIXTURE_ROOT}/evaluation_${workload}.json" \
      --exact-names "${exact_names}" "${expert_args[@]}" \
      --offload-manifest "${case_dir}/manifest.jsonl" \
      --output "${case_dir}/result.json" --workload "${workload}" \
      --prompt-tokens "${prompt}" --batch-size "${batch}" \
      --kv-cache-memory-bytes "${kv}" --max-model-len "${max_model}" \
      --max-num-batched-tokens "${max_batched}" \
      --cpu-offload-gb "${cpu_offload_gb}" \
      --expected-parameter-count "${expected_count}" \
      --expected-full-parameter-bytes "${expected_full}" \
      --expected-selected-bytes "${selected}" --seed 0
    printf '\n'
  } >"${case_dir}/command.txt"

  echo "START ${label} coordinate=${coordinate} selected_bytes=${selected} kv_bytes=${kv}"
  set +e
  env "${environment[@]}" timeout --kill-after=30s 3600s taskset -c "${CPUSET}" \
    "${VLLM_PY}" "${PREFILL_RUNNER}" --arm "${arm}" --model "${MODEL}" \
    --fixture "${FIXTURE_ROOT}/evaluation_${workload}.json" \
    --exact-names "${exact_names}" "${expert_args[@]}" \
    --offload-manifest "${case_dir}/manifest.jsonl" \
    --output "${case_dir}/result.json" --workload "${workload}" \
    --prompt-tokens "${prompt}" --batch-size "${batch}" \
    --kv-cache-memory-bytes "${kv}" --max-model-len "${max_model}" \
    --max-num-batched-tokens "${max_batched}" --cpu-offload-gb "${cpu_offload_gb}" \
    --expected-parameter-count "${expected_count}" \
    --expected-full-parameter-bytes "${expected_full}" \
    --expected-selected-bytes "${selected}" --seed 0 \
    >"${case_dir}/run.log" 2>&1
  local rc=$?
  set -e
  stop_monitor
  echo "${rc}" >"${case_dir}/returncode.txt"
  local status=error
  if (( rc == 0 )) && jq -e --argjson batch "${batch}" \
    '.status=="ok" and (.outputs|length)==$batch and
     (.outputs|all(.finished and (.output_token_ids|length)==1)) and
     .configuration.decode_steps==0' "${case_dir}/result.json" >/dev/null 2>&1; then
    status=ok
  fi
  local peak=0
  if [[ -s "${case_dir}/nvml.csv" ]]; then
    peak="$(awk -F, '{gsub(/ /,"",$2); if ($2+0>m) m=$2+0} END{print m+0}' "${case_dir}/nvml.csv")"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${workload}" "${prompt}" "${batch}" "${arm}" "${coordinate}" \
    "${selected}" "${kv}" "${status}" "${rc}" "${peak}" \
    >>"${TOP_ROOT}/execution.tsv"
  wait_gpu_clean || die "GPU did not clean up after ${label}"
  echo "END ${label} status=${status} rc=${rc} peak_mib=${peak}"
  [[ "${status}" == ok ]] || return 1
}

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
[[ ! -e "${TOP_ROOT}" ]] || die "output already exists: ${TOP_ROOT}"
for path in "${VLLM_PY}" "${PREFILL_RUNNER}" "${BASE_HARNESS}" "${MODEL}/config.json" \
  "${NAMES}" "${PLAN_10}" "${PLAN_38}" "${PLAN_63}"; do
  [[ -f "${path}" ]] || die "missing required file: ${path}"
done
mkdir -p "${TOP_ROOT}/cases" "${TOP_ROOT}/provenance"
TOP_ROOT="$(cd "${TOP_ROOT}" && pwd)"

# Native vLLM can select complete packed Parameters only.  These three frozen
# prefix-layer sets are the nearest byte matches to cold10/cold38/cold63.
head -n 8 "${NAMES}" >"${TOP_ROOT}/provenance/native_names_4_layers.txt"
head -n 28 "${NAMES}" >"${TOP_ROOT}/provenance/native_names_14_layers.txt"
head -n 48 "${NAMES}" >"${TOP_ROOT}/provenance/native_names_24_layers.txt"

printf 'workload\tprompt\tbatch\tarm\tcoordinate_contract\tselected_bytes\tkv_bytes\tstatus\treturncode\tpeak_hbm_mib\n' \
  >"${TOP_ROOT}/execution.tsv"
sha256sum "${PREFILL_RUNNER}" "${NAMES}" "${PLAN_10}" "${PLAN_38}" "${PLAN_63}" \
  "${MODEL}/config.json" "${FIXTURE_ROOT}"/evaluation_*.json \
  >"${TOP_ROOT}/provenance/inputs.sha256"
git -C "${REPO_ROOT}" rev-parse HEAD >"${TOP_ROOT}/provenance/repo.commit"
git -C "${REPO_ROOT}" status --short >"${TOP_ROOT}/provenance/repo.status.txt"
nvidia-smi -q >"${TOP_ROOT}/provenance/nvidia-smi-q.txt"
lscpu >"${TOP_ROOT}/provenance/lscpu.txt"
uname -a >"${TOP_ROOT}/provenance/uname.txt"
jq -n --arg custom_arm_filter "${CUSTOM_ARM_FILTER}" \
  --arg attention_backend "${ATTENTION_BACKEND}" \
  --argjson run_custom "${RUN_CUSTOM}" --argjson run_native "${RUN_NATIVE}" \
  '{schema:"eurosys_e3_prefill_only_four_mechanism_v1",process_blocks:1,
  batches:[1,8],workloads:["short_dialogue","ordinary_chat","medium_context","rag_summary","long_generation"],
  custom_arm_filter:$custom_arm_filter,run_custom:($run_custom==1),
  run_native:($run_native==1),
  output_tokens:1,decode_steps:0,transport:"none",pure_prefill:true,
  attention_backend:$attention_backend,
  request_admission:"one complete batch before first EngineCore step",
  native_coordinate_contract:"nearest-byte complete packed Parameters; auxiliary, not exact expert coordinates"}' \
  >"${TOP_ROOT}/configuration.json"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
# shellcheck source=/dev/null
source "${BASE_HARNESS}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY
RUN_ROOT_ACTIVE="${TOP_ROOT}"
trap cleanup_prefill EXIT INT TERM
runtime_preflight || die "custom runtime preflight failed"
module_snapshot "${TOP_ROOT}/provenance/module.before.json" "${STOCK_UVM}"

workloads=(short_dialogue ordinary_chat medium_context rag_summary long_generation)
prompts=(128 512 2048 8192 512)
original_outputs=(32 128 128 128 512)
batches=(1 8)

if [[ "${RUN_CUSTOM}" == 1 ]]; then
  load_custom_module || die "could not load custom UVM"
  CUSTOM_LOADED=1
  module_snapshot "${TOP_ROOT}/provenance/module.custom.json" "${CUSTOM_UVM}"
  for wi in "${!workloads[@]}"; do
    for batch in "${batches[@]}"; do
      for arm in graph_copy remote_eager alias_eager; do
        contains_csv "${arm}" "${CUSTOM_ARM_FILTER}" || continue
        run_one "${arm}" "${workloads[$wi]}" "${prompts[$wi]}" \
          "${original_outputs[$wi]}" "${batch}" \
          || die "custom cell failed; artifact retained"
      done
    done
  done
  restore_stock || die "failed to restore stock UVM before native baseline"
  CUSTOM_LOADED=0
fi

if [[ "${RUN_NATIVE}" == 1 ]]; then
  module_snapshot "${TOP_ROOT}/provenance/module.stock_for_native.json" "${STOCK_UVM}"
  for wi in "${!workloads[@]}"; do
    for batch in "${batches[@]}"; do
      run_one native_copy_on_forward "${workloads[$wi]}" "${prompts[$wi]}" \
        "${original_outputs[$wi]}" "${batch}" \
        || die "native copy-on-forward cell failed; artifact retained"
    done
  done
fi

module_snapshot "${TOP_ROOT}/provenance/module.after.json" "${STOCK_UVM}"
echo "TOP_ROOT=${TOP_ROOT}"
