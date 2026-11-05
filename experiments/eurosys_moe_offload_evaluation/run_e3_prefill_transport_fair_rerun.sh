#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_PY="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
PREFILL_RUNNER="${SCRIPT_DIR}/e3_prefill_only_runner.py"
PLAN_PREP="${SCRIPT_DIR}/prepare_e3_prefill_transport_matched_plans.py"
ANALYZE="${SCRIPT_DIR}/analyze_e3_prefill_transport_fair_rerun.py"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
FIXTURE_ROOT="${FIXTURE_ROOT:-${SCRIPT_DIR}/inputs/generated/eurosys-pa-v2-20260824/qwen3_30b_a3b_fp8}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-${SCRIPT_DIR}/artifacts/e3-calibration-20260825T091247Z}"
NAMES="${CALIBRATION_ROOT}/calibration_assets/exact_parameter_names.txt"
ROUTING_SOURCE="${SCRIPT_DIR}/inputs/legacy_pilot/qwen3_30b_a3b_fp8/routing/expert_distribution.pt"
TOP_ROOT="${1:-${SCRIPT_DIR}/artifacts/e3-prefill-transport-fair-rerun-$(date -u +%Y%m%dT%H%M%SZ)}"
CPUSET="${CPUSET:-0-13}"
PROCESS_BLOCKS="${PROCESS_BLOCKS:-2}"
CONDITIONING_REPEATS="${CONDITIONING_REPEATS:-1}"
MEASUREMENT_REPEATS="${MEASUREMENT_REPEATS:-10}"
WORKLOAD_FILTER="${WORKLOAD_FILTER:-short_dialogue,ordinary_chat,medium_context,rag_summary,long_generation}"
BATCH_FILTER="${BATCH_FILTER:-1,8}"
ARM_FILTER="${ARM_FILTER:-adaptmoe,native_copy}"
E3_GEOMETRY_OVERRIDE="${E3_GEOMETRY_OVERRIDE:-auto}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
KV_BYTES_PER_TOKEN=98304
FULL_PARAMETER_BYTES=28991029248
EXPECTED_PARAMETER_COUNT=96
MONITOR_PID=""
CUSTOM_LOADED=0

die() { echo "E3 fair Prefill rerun: $*" >&2; exit 1; }

contains_csv() {
  local needle="$1" haystack=",$2,"
  [[ "${haystack}" == *",${needle},"* ]]
}

stop_monitor() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=""
  fi
}

cleanup_fair_prefill() {
  stop_monitor
  if [[ "${CUSTOM_LOADED}" == 1 ]]; then
    restore_stock || true
    CUSTOM_LOADED=0
  fi
  restart_exporters || true
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
  if [[ "${E3_GEOMETRY_OVERRIDE}" == high ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      high 14722007040 613416960 13.7109375 72 14722007040 0
  elif [[ "${E3_GEOMETRY_OVERRIDE}" == mid ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      mid 8606711808 358612992 8.015625 44 8606711808 0
  elif [[ "${E3_GEOMETRY_OVERRIDE}" == low ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      low 2491416576 103809024 2.3203125 16 2491416576 0
  elif [[ "${workload}" == rag_summary && "${batch}" == 8 ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      high 14722007040 613416960 13.7109375 72 14722007040 20480
  elif [[ "${workload}" == rag_summary \
      || ("${workload}" == medium_context && "${batch}" == 8) ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      mid 8606711808 358612992 8.015625 44 8606711808 \
      "$([[ "${workload}" == rag_summary ]] && echo 8192 || echo 16384)"
  else
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      low 2491416576 103809024 2.3203125 16 2491416576 0
  fi
}

run_one() {
  local block="$1" mechanism="$2" workload="$3" prompt="$4"
  local original_output="$5" batch="$6"
  local tier selected backing native_gib native_count native_bytes reserve
  IFS=$'\t' read -r tier selected backing native_gib native_count native_bytes reserve \
    < <(cell_geometry "${workload}" "${batch}")
  local plan="${TOP_ROOT}/provenance/plans/${tier}.json"
  local kv label case_dir arm cpu_offload_gb expected_native_count
  local expected_native_bytes coordinate
  kv="$(kv_bytes "${prompt}" "${original_output}" "${batch}")"
  label="b${block}_${workload}_b${batch}_${mechanism}"
  case_dir="${TOP_ROOT}/cases/${label}"
  mkdir "${case_dir}"
  : >"${case_dir}/manifest.jsonl"

  arm=graph_copy
  cpu_offload_gb=30.0
  expected_native_count=0
  expected_native_bytes=0
  coordinate=exact_expert_slices
  if [[ "${mechanism}" == native_copy ]]; then
    arm=native_graph_copy_on_forward
    cpu_offload_gb="${native_gib}"
    expected_native_count="${native_count}"
    expected_native_bytes="${native_bytes}"
    coordinate=whole_parameters
  fi

  wait_gpu_clean || die "GPU is dirty before ${label}"
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu \
    --format=csv,noheader,nounits -lms 100 >"${case_dir}/nvml.csv" 2>"${case_dir}/nvml.err" &
  MONITOR_PID="$!"

  local max_model=$((prompt + original_output))
  local max_batched=$((prompt * batch))
  (( max_batched < max_model )) && max_batched="${max_model}"
  local chunk=16384 autotune=0
  if (( prompt * batch >= 8192 )); then chunk=4096; autotune=4096; fi

  local -a environment=(
    VLLM_ATTENTION_BACKEND=TRITON_ATTN
    VLLM_ENABLE_V1_MULTIPROCESSING=0
    VLLM_CPU_OFFLOAD_BACKEND=copy
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
  local -a expert_args=()
  if [[ "${mechanism}" == adaptmoe ]]; then
    expert_args=(--expert-plan "${plan}")
    environment+=(
      VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${NAMES}"
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER=0
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="${plan}"
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=4k
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPECTED_ARENA_COUNT=4
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPECTED_BACKING_BYTES="${backing}"
      VLLM_UVM_ORIGINAL_VA_PHASE_MODE=prefill_graph_remote_decode
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
      VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_BYTES="${kv}"
      VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_EMPTY_CACHE=1
      VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1
      VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE=1
      VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=1
      VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES=201326592
      VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS="${reserve}"
      VLLM_PHASE_SWITCH_CONTROLLED_HARD_EXIT_AFTER_CLEANUP=1
    )
  else
    environment+=(
      VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${TOP_ROOT}/provenance/plans/native_names_${tier}.txt"
      VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=0
      VLLM_UVM_ORIGINAL_VA_PHASE_MODE=off
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=0
      VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=0
      VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE=0
      VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=0
      VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS=0
    )
  fi

  local -a command=(
    "${VLLM_PY}" "${PREFILL_RUNNER}"
    --arm "${arm}"
    --model "${MODEL}"
    --fixture "${FIXTURE_ROOT}/evaluation_${workload}.json"
    --exact-names "${NAMES}"
    "${expert_args[@]}"
    --offload-manifest "${case_dir}/manifest.jsonl"
    --output "${case_dir}/result.json"
    --workload "${workload}"
    --prompt-tokens "${prompt}"
    --batch-size "${batch}"
    --kv-cache-memory-bytes "${kv}"
    --max-model-len "${max_model}"
    --max-num-batched-tokens "${max_batched}"
    --cpu-offload-gb "${cpu_offload_gb}"
    --expected-parameter-count "${EXPECTED_PARAMETER_COUNT}"
    --expected-full-parameter-bytes "${FULL_PARAMETER_BYTES}"
    --expected-selected-bytes "${selected}"
    --expected-native-parameter-count "${expected_native_count}"
    --expected-native-parameter-bytes "${expected_native_bytes}"
    --conditioning-repeats "${CONDITIONING_REPEATS}"
    --measurement-repeats "${MEASUREMENT_REPEATS}"
    --seed 0
  )
  local -a profile_prefix=()
  if [[ "${E3_NSYS_PROFILE:-0}" == 1 ]]; then
    command -v nsys >/dev/null || die "E3_NSYS_PROFILE=1 but nsys is unavailable"
    environment+=(
      E3_NSYS_PREFILL_CAPTURE=1
      E3_NSYS_PREFILL_CAPTURE_ROLE=measurement
      E3_NSYS_PREFILL_CAPTURE_ROLE_INDEX=0
    )
    profile_prefix=(
      nsys profile
      --trace=cuda,nvtx
      --cuda-graph-trace="${E3_NSYS_CUDA_GRAPH_TRACE:-node}"
      --sample=none
      --cpuctxsw=none
      --capture-range=cudaProfilerApi
      --capture-range-end=stop
      --force-overwrite=true
      --export=sqlite
      --output="${case_dir}/prefill-steady"
    )
  fi
  {
    printf 'cd %q && env -u VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE ' "${REPO_ROOT}"
    printf '%q ' "${environment[@]}"
    printf 'timeout --kill-after=30s 3600s '
    printf '%q ' "${profile_prefix[@]}"
    printf 'taskset -c %q ' "${CPUSET}"
    printf '%q ' "${command[@]}"
    printf '\n'
  } >"${case_dir}/command.txt"

  echo "START ${label} tier=${tier} coordinate=${coordinate} selected=${selected} native=${native_bytes}"
  set +e
  env -u VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE "${environment[@]}" \
    timeout --kill-after=30s 3600s "${profile_prefix[@]}" \
      taskset -c "${CPUSET}" "${command[@]}" \
    >"${case_dir}/run.log" 2>&1
  local rc=$?
  set +e
  stop_monitor
  echo "${rc}" >"${case_dir}/returncode.txt"

  local status=error
  if (( rc == 0 )) && jq -e \
    --argjson conditioning "${CONDITIONING_REPEATS}" \
    --argjson measurement "${MEASUREMENT_REPEATS}" \
    --argjson batch "${batch}" '
      .status == "ok"
      and ([.cycles[] | select(.role=="conditioning")] | length) == $conditioning
      and ([.cycles[] | select(.role=="measurement")] | length) == $measurement
      and (.cycles | all(
        (.outputs | length) == $batch
        and (.outputs | all(.finished and (.output_token_ids | length) == 1))
        and .graph_wrapper_delta_prefill.full_replay == 1))
      and ([.cycles[].output_token_ids_sha256] | unique | length) == 1
      and .shutdown_contract.diagnostics_after_shutdown.driver.session_count == 0
    ' "${case_dir}/result.json" >/dev/null 2>&1; then
    status=ok
  fi
  local peak=0 first_ms=null steady_ms=null actual_bytes=0
  if [[ -s "${case_dir}/nvml.csv" ]]; then
    peak="$(awk -F, '{gsub(/ /,"",$2); if ($2+0>m) m=$2+0} END{print m+0}' "${case_dir}/nvml.csv")"
  fi
  if [[ -f "${case_dir}/result.json" ]]; then
    first_ms="$(jq -r '.timing.first_request_prefill_step_ms // "null"' "${case_dir}/result.json")"
    steady_ms="$(jq -r '.timing.prefill_step_ms // "null"' "${case_dir}/result.json")"
    if [[ "${mechanism}" == adaptmoe ]]; then
      actual_bytes="$(jq -r '.diagnostics.after_load.model.selected_parameter_bytes_per_forward // 0' "${case_dir}/result.json")"
    else
      actual_bytes="$(jq -r '.manifest.offloaded_parameter_bytes // 0' "${case_dir}/result.json")"
    fi
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${block}" "${workload}" "${prompt}" "${batch}" "${mechanism}" "${tier}" \
    "${selected}" "${native_bytes}" "${actual_bytes}" "${first_ms}" "${steady_ms}" \
    "${status}" "${rc}" "${peak}" >>"${TOP_ROOT}/execution.tsv"
  wait_gpu_clean || die "GPU did not clean up after ${label}"
  echo "END ${label} status=${status} first_ms=${first_ms} steady_median_ms=${steady_ms}"
  [[ "${status}" == ok ]]
}

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
[[ "${PROCESS_BLOCKS}" =~ ^[1-9][0-9]*$ ]] || die "invalid PROCESS_BLOCKS"
[[ "${CONDITIONING_REPEATS}" =~ ^[1-9][0-9]*$ ]] || die "conditioning must be positive"
[[ "${MEASUREMENT_REPEATS}" =~ ^[1-9][0-9]*$ ]] || die "measurement must be positive"
[[ "${E3_GEOMETRY_OVERRIDE}" =~ ^(auto|low|mid|high)$ ]] \
  || die "invalid E3_GEOMETRY_OVERRIDE"
[[ ! -e "${TOP_ROOT}" ]] || die "output already exists: ${TOP_ROOT}"
for path in "${VLLM_PY}" "${PREFILL_RUNNER}" "${PLAN_PREP}" "${ANALYZE}" \
  "${BASE_HARNESS}" "${MODEL}/config.json" "${NAMES}" "${ROUTING_SOURCE}"; do
  [[ -f "${path}" ]] || die "missing required file: ${path}"
done

mkdir -p "${TOP_ROOT}/cases" "${TOP_ROOT}/provenance/plans"
TOP_ROOT="$(cd "${TOP_ROOT}" && pwd)"
"${VLLM_PY}" "${PLAN_PREP}" --source "${ROUTING_SOURCE}" \
  --output-dir "${TOP_ROOT}/provenance/plans" \
  >"${TOP_ROOT}/provenance/plan-generation.stdout.json"
printf 'block\tworkload\tprompt\tbatch\tmechanism\ttier\tadaptmoe_bytes\tnative_bytes\tactual_bytes\tfirst_ms\tsteady_median_ms\tstatus\treturncode\tpeak_hbm_mib\n' \
  >"${TOP_ROOT}/execution.tsv"
sha256sum "${PREFILL_RUNNER}" "${PLAN_PREP}" "${ANALYZE}" "${NAMES}" \
  "${TOP_ROOT}/provenance/plans"/* "${MODEL}/config.json" \
  "${FIXTURE_ROOT}"/evaluation_*.json >"${TOP_ROOT}/provenance/inputs.sha256"
git -C "${REPO_ROOT}" rev-parse HEAD >"${TOP_ROOT}/provenance/repo.commit"
git -C "${REPO_ROOT}" status --short >"${TOP_ROOT}/provenance/repo.status.txt"
nvidia-smi -q >"${TOP_ROOT}/provenance/nvidia-smi-q.txt"
lscpu >"${TOP_ROOT}/provenance/lscpu.txt"
uname -a >"${TOP_ROOT}/provenance/uname.txt"
jq -n \
  --arg workload_filter "${WORKLOAD_FILTER}" \
  --arg batch_filter "${BATCH_FILTER}" \
  --arg arm_filter "${ARM_FILTER}" \
  --arg geometry_override "${E3_GEOMETRY_OVERRIDE}" \
  --argjson process_blocks "${PROCESS_BLOCKS}" \
  --argjson conditioning_repeats "${CONDITIONING_REPEATS}" \
  --argjson measurement_repeats "${MEASUREMENT_REPEATS}" \
  '{schema:"eurosys_e3_prefill_transport_fair_rerun_v1",
    attention_backend:"TRITON_ATTN",cudagraph_mode:"FULL_SEPARATE",
    process_blocks:$process_blocks,conditioning_repeats:$conditioning_repeats,
    measurement_repeats:$measurement_repeats,
    geometry_override:$geometry_override,
    workload_filter:$workload_filter,batch_filter:$batch_filter,
    arm_filter:$arm_filter,output_tokens:1,decode_steps:0,
    timing_boundary:"synchronized llm_engine.step wall time",
    conditioning:"complete real batch followed by untimed empty cleanup",
    arm_order:"AB then BA in alternating blocks",
    statistical_scope:(if $process_blocks < 5 then
      "diagnostic replication only; insufficient independent blocks for a stable CI"
      else "independent-block distribution and bootstrap CI" end),
    byte_contract:"exact H2D byte equality using complete native Parameters and uniform-width expert slices"}' \
  >"${TOP_ROOT}/configuration.json"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
# shellcheck source=/dev/null
source "${BASE_HARNESS}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY
RUN_ROOT_ACTIVE="${TOP_ROOT}"
trap cleanup_fair_prefill EXIT INT TERM
runtime_preflight || die "custom runtime preflight failed"
module_snapshot "${TOP_ROOT}/provenance/module.before.json" "${STOCK_UVM}"
load_custom_module || die "could not load custom UVM"
CUSTOM_LOADED=1
module_snapshot "${TOP_ROOT}/provenance/module.custom.json" "${CUSTOM_UVM}"
kernel_epoch="$(date +%s)"

workloads=(short_dialogue ordinary_chat medium_context rag_summary long_generation)
prompts=(128 512 2048 8192 512)
original_outputs=(32 128 128 128 512)
batches=(1 8)
for wi in "${!workloads[@]}"; do
  workload="${workloads[$wi]}"
  contains_csv "${workload}" "${WORKLOAD_FILTER}" || continue
  for batch in "${batches[@]}"; do
    contains_csv "${batch}" "${BATCH_FILTER}" || continue
    for ((block=0; block<PROCESS_BLOCKS; ++block)); do
      if (( block % 2 == 0 )); then
        mechanisms=(adaptmoe native_copy)
      else
        mechanisms=(native_copy adaptmoe)
      fi
      for mechanism in "${mechanisms[@]}"; do
        contains_csv "${mechanism}" "${ARM_FILTER}" || continue
        if ! run_one "${block}" "${mechanism}" "${workload}" \
          "${prompts[$wi]}" "${original_outputs[$wi]}" "${batch}"; then
          if [[ "${STOP_ON_ERROR}" == 1 ]]; then
            die "cell failed: block=${block} ${workload}/B${batch}/${mechanism}"
          fi
        fi
      done
    done
  done
done

sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
  >"${TOP_ROOT}/kernel.since.log" 2>"${TOP_ROOT}/kernel.since.err" || true
restore_stock || die "failed to restore stock UVM"
CUSTOM_LOADED=0
module_snapshot "${TOP_ROOT}/provenance/module.after.json" "${STOCK_UVM}"
restart_exporters || true
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits \
  >"${TOP_ROOT}/gpu.after.csv"

set +e
"${VLLM_PY}" "${ANALYZE}" "${TOP_ROOT}" \
  >"${TOP_ROOT}/analysis.stdout.txt" 2>"${TOP_ROOT}/analysis.stderr.txt"
analysis_rc=$?
set +e
echo "${analysis_rc}" >"${TOP_ROOT}/analysis.returncode.txt"
echo "TOP_ROOT=${TOP_ROOT}"
echo "ANALYSIS_RC=${analysis_rc}"
(( analysis_rc == 0 ))
