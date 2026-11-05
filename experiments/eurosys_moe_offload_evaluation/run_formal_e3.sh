#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${REPO_ROOT}/workloads/vllm/vllm"
VLLM_PY="${VLLM_ROOT}/.venv/bin/python"
KT_ROOT="${REPO_ROOT}/KTransformers"
KT_PY="${KT_ROOT}/.venv/bin/python"
PHASE_RUNNER="${PHASE_RUNNER_OVERRIDE:-${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/benchmark_phase_switch_steps.py}"
KT_RUNNER="${REPO_ROOT}/experiments/ktransformers_l0_l3_phase_compare/benchmark_phase_stream.py"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
CALIBRATION_ROOT="${CALIBRATION_ROOT:-${SCRIPT_DIR}/artifacts/e3-calibration-20260825T091247Z}"
PLAN="${PLAN:-${CALIBRATION_ROOT}/calibration_assets/cold_decode_plan_2g.json}"
NAMES="${NAMES:-${CALIBRATION_ROOT}/calibration_assets/exact_parameter_names.txt}"
KT_STATS="${KT_STATS:-${CALIBRATION_ROOT}/calibration_assets/kt_stats/cold_decode_plan_2g_logical_count.pt}"
CALIBRATION_MANIFEST="${CALIBRATION_MANIFEST:-${CALIBRATION_ROOT}/calibration_assets/manifest.json}"
FIXTURE_ROOT="${FIXTURE_ROOT:-${SCRIPT_DIR}/inputs/generated/eurosys-pa-v2-20260824/qwen3_30b_a3b_fp8}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/e3-formal-$(date -u +%Y%m%dT%H%M%SZ)}"
PROCESS_BLOCKS="${PROCESS_BLOCKS:-1}"
PORT="${PORT:-30000}"
CPUSET="${CPUSET:-0-13}"
CUSTOM_KV_LIMIT_BYTES="${CUSTOM_KV_LIMIT_BYTES:-2400000000}"
KT_KV_LIMIT_BYTES="${KT_KV_LIMIT_BYTES:-2800000000}"
CUSTOM_PREFILL_NA_THRESHOLD="${CUSTOM_PREFILL_NA_THRESHOLD:-8192}"
STOP_ON_CUSTOM_ERROR="${STOP_ON_CUSTOM_ERROR:-0}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_ATTN}"
CUSTOM_PTE_MODE="${CUSTOM_PTE_MODE:-4k}"
EXPECTED_ARENA_COUNT="${EXPECTED_ARENA_COUNT:-4}"
SELECTED_BYTES="${SELECTED_BYTES:-2264924160}"
FULL_BYTES="${FULL_BYTES:-28991029248}"
EXPECTED_PARAMETER_COUNT="${EXPECTED_PARAMETER_COUNT:-96}"
COORDINATE_COUNT="${COORDINATE_COUNT:-480}"
BACKING_BYTES="${BACKING_BYTES:-94371840}"
KV_BYTES_PER_TOKEN="${KV_BYTES_PER_TOKEN:-98304}"
KT_METHOD="${KT_METHOD:-FP8}"
KT_GPU_EXPERTS_RATIO="${KT_GPU_EXPERTS_RATIO:-0.921875}"
KT_EXPECTED_GPU_EXPERTS="${KT_EXPECTED_GPU_EXPERTS:-118}"
KT_EXPECTED_MOE_LAYERS="${KT_EXPECTED_MOE_LAYERS:-48}"
KT_TRUST_REMOTE_CODE="${KT_TRUST_REMOTE_CODE:-0}"
RUN_CUSTOM="${RUN_CUSTOM:-1}"
RUN_KT="${RUN_KT:-1}"
WORKLOAD_FILTER="${WORKLOAD_FILTER:-short_dialogue,ordinary_chat,medium_context,rag_summary,long_generation}"
BATCH_FILTER="${BATCH_FILTER:-1,4,8}"
ARM_FILTER="${ARM_FILTER:-H,A,R}"
CUSTOM_ARMS="${CUSTOM_ARMS:-H,A,R}"
CUDAGRAPH_CAPTURE_SIZES="${CUDAGRAPH_CAPTURE_SIZES:-1,4,8}"
RUNTIME_HOT_RESIDENCY_AUDIT="${RUNTIME_HOT_RESIDENCY_AUDIT:-0}"
FUSED_MOE_CHUNK_SIZE="${FUSED_MOE_CHUNK_SIZE:-16384}"
PRE_KV_NATIVE_HOLE_SENTINEL="${PRE_KV_NATIVE_HOLE_SENTINEL:-0}"
SERVING_NATIVE_HOLE_SENTINEL="${SERVING_NATIVE_HOLE_SENTINEL:-${VLLM_UVM_NATIVE_HOLE_SENTINEL_BYTES:-0}}"
FLASHINFER_AUTOTUNE_MAX_TOKENS="${FLASHINFER_AUTOTUNE_MAX_TOKENS:-0}"
E3_CURRENT_TIER_DUAL_VIEW="${E3_CURRENT_TIER_DUAL_VIEW:-0}"
SERVER_PID=""
MONITOR_PID=""
CUSTOM_LOADED=0

die() { echo "error: $*" >&2; exit 1; }
contains_csv() { local needle="$1" haystack=",$2,"; [[ "${haystack}" == *",${needle},"* ]]; }
stop_monitor() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=""
  fi
}
stop_server() {
  stop_monitor
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -INT -- "-${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "${SERVER_PID}" 2>/dev/null || break; sleep 1; done
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
cleanup() {
  stop_server
  if [[ "${CUSTOM_LOADED}" == 1 ]]; then restore_stock || true; CUSTOM_LOADED=0; fi
}
trap cleanup EXIT INT TERM

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
for path in "${VLLM_PY}" "${KT_PY}" "${PHASE_RUNNER}" "${KT_RUNNER}" \
  "${MODEL}/config.json" "${PLAN}" "${NAMES}" "${KT_STATS}" "${CALIBRATION_MANIFEST}"; do
  [[ -f "${path}" ]] || die "missing file: ${path}"
done
[[ "${PROCESS_BLOCKS}" =~ ^[1-9][0-9]*$ ]] || die "invalid PROCESS_BLOCKS"
[[ "${E3_CURRENT_TIER_DUAL_VIEW}" == 0 || "${E3_CURRENT_TIER_DUAL_VIEW}" == 1 ]] \
  || die "E3_CURRENT_TIER_DUAL_VIEW must be 0 or 1"
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"
mkdir -p "${RUN_ROOT}/provenance" "${RUN_ROOT}/custom" "${RUN_ROOT}/kt" "${RUN_ROOT}/capability"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
if [[ "${E3_CURRENT_TIER_DUAL_VIEW}" == 1 ]]; then
  # The module switch is global for this process block. Full coverage remains
  # a per-serving-process software requirement, so B/C retain their original
  # no-switch execution paths while G/D fail closed unless their fixed tier is
  # completely covered.
  export STATIC_ALIAS_DUAL_VIEW_ENABLE=1
  export STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL=0
  export ACCESS_COUNTER_MIGRATION_ENABLE=0
  export UVM_PREFETCH_ENABLE=0
  export UVM_THRASHING_ENABLE=0
fi
# shellcheck source=/dev/null
source "${BASE_HARNESS}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY
# The shared harness defines its own legacy PHASE_RUNNER.  Rebind the formal
# E3 runner after sourcing so its H/A/R arm contract cannot be shadowed.
PHASE_RUNNER="${PHASE_RUNNER_OVERRIDE:-${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/benchmark_phase_switch_steps.py}"
RUN_ROOT_ACTIVE="${RUN_ROOT}"
runtime_preflight || die "custom runtime preflight failed"

sha256sum "${PLAN}" "${NAMES}" "${KT_STATS}" "${MODEL}/config.json" \
  "${FIXTURE_ROOT}"/evaluation_*.json >"${RUN_ROOT}/provenance/inputs.sha256"
git -C "${REPO_ROOT}" rev-parse HEAD >"${RUN_ROOT}/provenance/repo.commit"
git -C "${KT_ROOT}" rev-parse HEAD >"${RUN_ROOT}/provenance/ktransformers.commit"
git -C "${KT_ROOT}/third_party/sglang" rev-parse HEAD >"${RUN_ROOT}/provenance/sglang.commit"
git -C "${REPO_ROOT}" status --short >"${RUN_ROOT}/provenance/repo.status.txt"
nvidia-smi -q >"${RUN_ROOT}/provenance/nvidia-smi-q.txt"
lscpu >"${RUN_ROOT}/provenance/lscpu.txt"
uname -a >"${RUN_ROOT}/provenance/uname.txt"
cp "${CALIBRATION_MANIFEST}" "${RUN_ROOT}/provenance/calibration_manifest.json"
cp "${PLAN}" "${RUN_ROOT}/provenance/expert_plan.json"
# Preserve the frozen 2-GiB filename consumed by the existing H/A/R verifier.
if [[ "${SELECTED_BYTES}" == 2264924160 ]]; then
  cp "${PLAN}" "${RUN_ROOT}/provenance/cold_decode_plan_2g.json"
fi
jq -n \
  --arg plan "$(realpath "${PLAN}")" \
  --arg custom_arms "${CUSTOM_ARMS}" \
  --arg batch_filter "${BATCH_FILTER}" \
  --arg workload_filter "${WORKLOAD_FILTER}" \
  --arg attention_backend "${ATTENTION_BACKEND}" \
  --arg capture_sizes "${CUDAGRAPH_CAPTURE_SIZES}" \
  --arg runtime_hot_residency_audit "${RUNTIME_HOT_RESIDENCY_AUDIT}" \
  --arg fused_moe_chunk_size "${FUSED_MOE_CHUNK_SIZE}" \
  --arg pre_kv_native_hole_sentinel "${PRE_KV_NATIVE_HOLE_SENTINEL}" \
  --arg serving_native_hole_sentinel "${SERVING_NATIVE_HOLE_SENTINEL}" \
  --arg flashinfer_autotune_max_tokens "${FLASHINFER_AUTOTUNE_MAX_TOKENS}" \
  --arg current_tier_dual_view "${E3_CURRENT_TIER_DUAL_VIEW}" \
  --argjson selected_bytes "${SELECTED_BYTES}" \
  --argjson backing_bytes "${BACKING_BYTES}" \
  --argjson process_blocks "${PROCESS_BLOCKS}" \
  '{schema:"eurosys_e3_run_config_v1",plan:$plan,custom_arms:$custom_arms,
    batch_filter:$batch_filter,workload_filter:$workload_filter,
    attention_backend:$attention_backend,cudagraph_capture_sizes:$capture_sizes,
    runtime_hot_residency_audit:$runtime_hot_residency_audit,
    fused_moe_chunk_size:$fused_moe_chunk_size,
    pre_kv_native_hole_sentinel:$pre_kv_native_hole_sentinel,
    serving_native_hole_sentinel:$serving_native_hole_sentinel,
    flashinfer_autotune_max_tokens:$flashinfer_autotune_max_tokens,
    current_tier_dual_view:($current_tier_dual_view == "1"),
    selected_bytes:$selected_bytes,backing_bytes:$backing_bytes,
    process_blocks:$process_blocks,transport:"none"}' \
  >"${RUN_ROOT}/provenance/run_config.json"

jq -n --arg reason "native vLLM cpu_offload allocates whole Parameter tensors and has no expert-slice plan interface; it cannot reproduce the frozen ${COORDINATE_COUNT} (layer,expert) coordinates" \
  '{schema:"eurosys_e3_capability_v1",track:"A_exact_coordinate",pinned_uva:{status:"N/A",reason:$reason},copy_on_forward:{status:"N/A",reason:$reason}}' \
  >"${RUN_ROOT}/capability/vllm_native.json"
printf 'block\tworkload\tprompt\toutput\tbatch\tarm\tcase_dir\tkv_bytes\tstatus\treturncode\n' >"${RUN_ROOT}/custom/execution.tsv"
printf 'block\tworkload\tprompt\toutput\tbatches\tcase_dir\tkv_bytes_b8\tstatus\treturncode\n' >"${RUN_ROOT}/kt/execution.tsv"

workloads=(short_dialogue ordinary_chat medium_context rag_summary long_generation)
prompts=(128 512 2048 8192 512)
outputs=(32 128 128 128 512)
batches=(1 4 8 16)
IFS=',' read -r -a arms <<<"${CUSTOM_ARMS}"
for arm in "${arms[@]}"; do
  case "${arm}" in
    H|A|R|G|B|C|D|E|F) ;;
    *) die "invalid custom arm in CUSTOM_ARMS: ${arm}" ;;
  esac
done

kv_bytes() {
  local prompt="$1" output="$2" batch="$3" rounded
  # One extra block per request is required while the final sampled token is
  # materialized; an exact prompt+output capacity preempts the last request.
  rounded=$(( (prompt + output + 16 + 15) / 16 * 16 ))
  local value=$((rounded * batch * KV_BYTES_PER_TOKEN))
  (( value < 167772160 )) && value=167772160
  echo "${value}"
}

module_snapshot "${RUN_ROOT}/provenance/module.before.json" "${STOCK_UVM}"
if [[ "${RUN_CUSTOM}" == 1 ]]; then
  load_custom_module || die "could not load custom UVM"
  CUSTOM_LOADED=1
  module_snapshot "${RUN_ROOT}/provenance/module.custom.json" "${CUSTOM_UVM}"
fi

cell=0
if [[ "${RUN_CUSTOM}" == 1 ]]; then
for ((block=0; block<PROCESS_BLOCKS; ++block)); do
  for wi in "${!workloads[@]}"; do
    workload="${workloads[$wi]}"; prompt="${prompts[$wi]}"; output="${outputs[$wi]}"
    contains_csv "${workload}" "${WORKLOAD_FILTER}" || continue
    fixture="${FIXTURE_ROOT}/evaluation_${workload}.json"
    for batch in "${batches[@]}"; do
      contains_csv "${batch}" "${BATCH_FILTER}" || continue
      kv="$(kv_bytes "${prompt}" "${output}" "${batch}")"
      prefill_tokens=$((prompt * batch))
      arm_count="${#arms[@]}"
      (( arm_count > 0 )) || die "CUSTOM_ARMS must not be empty"
      rotation=$(( (cell + block) % arm_count ))
      for ((offset=0; offset<arm_count; ++offset)); do
        arm="${arms[$(((rotation + offset) % arm_count))]}"
        contains_csv "${arm}" "${ARM_FILTER}" || continue
        label="b${block}_${workload}_b${batch}_${arm}"
        case_dir="${RUN_ROOT}/custom/${label}"
        mkdir "${case_dir}"
        if (( CUSTOM_PREFILL_NA_THRESHOLD > 0 && prefill_tokens >= CUSTOM_PREFILL_NA_THRESHOLD )); then
          jq -n --arg status N/A --arg reason "backend preflight: >=8192-token single-step Prefill triggers TMA descriptor 700 / illegal-address on this platform" \
            --argjson prefill_tokens "${prefill_tokens}" \
            '{status:$status,reason:$reason,prefill_tokens:$prefill_tokens,observed_in_prior_attempt:true}' >"${case_dir}/na.json"
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tN/A\t-\n' "${block}" "${workload}" "${prompt}" "${output}" "${batch}" "${arm}" "${label}" "${kv}" >>"${RUN_ROOT}/custom/execution.tsv"
          continue
        fi
        if (( kv > CUSTOM_KV_LIMIT_BYTES )); then
          jq -n --arg status N/A --arg reason "pre-registered KV bytes exceed custom HBM headroom" \
            --argjson kv "${kv}" --argjson limit "${CUSTOM_KV_LIMIT_BYTES}" \
            '{status:$status,reason:$reason,required_kv_bytes:$kv,limit_bytes:$limit}' >"${case_dir}/na.json"
          printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tN/A\t-\n' "${block}" "${workload}" "${prompt}" "${output}" "${batch}" "${arm}" "${label}" "${kv}" >>"${RUN_ROOT}/custom/execution.tsv"
          continue
        fi
        wait_gpu_clean || { echo dirty >"${case_dir}/preflight_error.txt"; continue; }
        : >"${case_dir}/manifest.jsonl"
        nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu \
          --format=csv,noheader,nounits -lms 200 >"${case_dir}/nvml.csv" 2>"${case_dir}/nvml.err" &
        MONITOR_PID="$!"
        phase=remote_only; alias_eager=0
        [[ "${arm}" == H || "${arm}" == G ]] && phase=prefill_remote_decode
        [[ "${arm}" == D ]] && phase=prefill_graph_remote_decode
        [[ "${arm}" == A || "${arm}" == B || "${arm}" == E ]] && { phase=off; alias_eager=1; }
        measured=$((output - 7))
        max_model=$((prompt + output))
        max_batched=$((prompt * batch)); (( max_batched < max_model )) && max_batched="${max_model}"
        capture_sizes="${CUDAGRAPH_CAPTURE_SIZES}"
        if [[ "${capture_sizes}" == per_batch ]]; then
          capture_sizes="${batch}"
          [[ "${arm}" == D || "${arm}" == E || "${arm}" == F ]] \
            && capture_sizes="${batch},${prefill_tokens}"
        fi
        sentinel_bytes="${PRE_KV_NATIVE_HOLE_SENTINEL}"
        [[ "${sentinel_bytes}" == auto || "${sentinel_bytes}" == kv ]] && sentinel_bytes="${kv}"
        custom_expect_env=()
        dual_view_env=()
        arm_runtime_env=()
        if (( EXPECTED_ARENA_COUNT > 0 )); then
          custom_expect_env+=(VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPECTED_ARENA_COUNT="${EXPECTED_ARENA_COUNT}")
        fi
        if (( BACKING_BYTES > 0 )); then
          custom_expect_env+=(VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPECTED_BACKING_BYTES="${BACKING_BYTES}")
        fi
        if [[ "${E3_CURRENT_TIER_DUAL_VIEW}" == 1 \
            && ("${arm}" == G || "${arm}" == D) ]]; then
          dual_view_env+=(
            VLLM_CPU_OFFLOAD_PLAN_CLASS_IMMUTABLE_CPU_SHADOW=1
            VLLM_CPU_OFFLOAD_DUAL_VIEW_FULL_PARAMETER_SHADOW=1
            VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW=1
            VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1
            VLLM_UVM_RUNTIME_POST_MAP_RESIDENCY_AUDIT=0
          )
        fi
        if [[ "${arm}" == C ]]; then
          # Preserve the frozen original-E3 C path. C has eager remote
          # Prefill and remote Decode Graph, so none of D's all-remote
          # profiling, Prefill-Graph capture, PARK/RESUME, or allocator-pool
          # preparation may leak into this fresh process.
          arm_runtime_env+=(
            VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=0
            VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE=0
            VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=0
            VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS=0
          )
        elif [[ "${arm}" == F ]]; then
          # F is an always-remote FULL_SEPARATE control. It never owns an
          # alias-prefill topology, so inherited D-only PARK/RESUME and
          # allocator-reserve settings must not leak into this fresh process.
          arm_runtime_env+=(
            VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=0
            VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS=0
          )
        fi
        printf '%s\n' "${dual_view_env[@]}" >"${case_dir}/dual_view_environment.txt"
        printf '%s\n' "${arm_runtime_env[@]}" >"${case_dir}/arm_runtime_environment.txt"
        set +e
        env "${custom_expect_env[@]}" "${dual_view_env[@]}" "${arm_runtime_env[@]}" VLLM_ATTENTION_BACKEND="${ATTENTION_BACKEND}" VLLM_ENABLE_V1_MULTIPROCESSING=0 \
          VLLM_CPU_OFFLOAD_BACKEND=copy VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${NAMES}" \
          VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1 VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER="${alias_eager}" \
          VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va \
          VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="${PLAN}" \
          VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288 VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE="${CUSTOM_PTE_MODE}" \
          VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 VLLM_UVM_ORIGINAL_VA_PHASE_MODE="${phase}" \
          VLLM_UVM_ORIGINAL_VA_PHASE_RESULT_FILE="${case_dir}/transition.json" \
          VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ=0 VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
          VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0 \
          VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0 VLLM_CPU_OFFLOAD_COPY_TRACE=0 \
          VLLM_UVM_RUNTIME_HOT_RESIDENCY_AUDIT="${RUNTIME_HOT_RESIDENCY_AUDIT}" \
          VLLM_UVM_POST_RECONCILE_FULL_TLB_INVALIDATE="${REMOTE_READ_FULL_TLB_INVALIDATE}" \
          VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_BYTES="${sentinel_bytes}" \
          VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_EMPTY_CACHE=1 \
          VLLM_UVM_NATIVE_HOLE_SENTINEL_BYTES="${SERVING_NATIVE_HOLE_SENTINEL}" \
          VLLM_CPU_OFFLOAD_COPY_SYNC=0 VLLM_CPU_OFFLOAD_MANIFEST_FILE="${case_dir}/manifest.jsonl" \
          VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 VLLM_UVM_LOG_FILE="${case_dir}/uvm.log" \
          HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
          VLLM_FUSED_MOE_CHUNK_SIZE="${FUSED_MOE_CHUNK_SIZE}" \
          VLLM_FLASHINFER_AUTOTUNE_MAX_TOKENS="${FLASHINFER_AUTOTUNE_MAX_TOKENS}" \
          OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 CUDA_VISIBLE_DEVICES=0 \
          timeout --kill-after=30s 3600s taskset -c "${CPUSET}" "${VLLM_PY}" "${PHASE_RUNNER}" \
            --model "${MODEL}" --fixture "${fixture}" --exact-names "${NAMES}" --expert-plan "${PLAN}" \
            --offload-manifest "${case_dir}/manifest.jsonl" --output "${case_dir}/result.json" \
            --arm "${arm}" --batch-size "${batch}" --cudagraph-capture-sizes "${capture_sizes}" \
            --warmup-decode-steps 5 --measured-decode-steps "${measured}" \
            --kv-cache-memory-bytes "${kv}" --cpu-offload-gb 30.0 --max-model-len "${max_model}" \
            --max-num-seqs "${batch}" --max-num-batched-tokens "${max_batched}" \
            --gpu-memory-utilization 0.97 --seed 0 --expected-parameter-count "${EXPECTED_PARAMETER_COUNT}" \
            --expected-full-parameter-bytes "${FULL_BYTES}" --expected-selected-bytes "${SELECTED_BYTES}" \
          >"${case_dir}/run.log" 2>&1
        rc=$?
        set +e
        stop_monitor
        status=error
        if (( rc == 0 )) && jq -e --argjson out "${output}" --argjson selected "${SELECTED_BYTES}" '
          .status == "ok" and (.outputs|all(.output_token_ids|length == $out))
          and .diagnostics.after_load.model.selected_parameter_bytes_per_forward == $selected
          and .shutdown_contract.diagnostics_after_shutdown.driver.session_count == 0
        ' "${case_dir}/result.json" >/dev/null 2>&1; then status=ok; fi
        echo "${rc}" >"${case_dir}/returncode.txt"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${block}" "${workload}" "${prompt}" "${output}" "${batch}" "${arm}" "${label}" "${kv}" "${status}" "${rc}" >>"${RUN_ROOT}/custom/execution.tsv"
        if [[ -r /proc/driver/nvidia-uvm/phase_f_route_log ]]; then
          cp /proc/driver/nvidia-uvm/phase_f_route_log \
            "${case_dir}/phase_f_route.log" || true
        fi
        if [[ -r /proc/driver/nvidia-uvm/phase_f_route_log_stats ]]; then
          cp /proc/driver/nvidia-uvm/phase_f_route_log_stats \
            "${case_dir}/phase_f_route_stats.txt" || true
        fi
        wait_gpu_clean || echo dirty >"${case_dir}/gpu_cleanup_error.txt"
        if [[ "${status}" != ok && "${STOP_ON_CUSTOM_ERROR}" == 1 ]]; then
          if [[ -r "${STATS_PATH}" ]]; then
            cp "${STATS_PATH}" "${RUN_ROOT}/provenance/latency_stats.custom_after.txt"
          fi
          restore_stock || true
          CUSTOM_LOADED=0
          module_snapshot "${RUN_ROOT}/provenance/module.stock_after_custom_error.json" "${STOCK_UVM}"
          die "custom E3 cell failed: ${label}"
        fi
      done
      cell=$((cell + 1))
    done
  done
done
fi

if [[ "${RUN_CUSTOM}" == 1 ]]; then
  if [[ -r "${STATS_PATH}" ]]; then
    cp "${STATS_PATH}" "${RUN_ROOT}/provenance/latency_stats.custom_after.txt"
  fi
  # A successful module reload can briefly make nvidia-smi unavailable while
  # the driver finishes reinitializing.  Retry the post-reload clean-state
  # check instead of abandoning the next independent cell after its result was
  # already committed.
  stock_restored=0
  for _ in $(seq 1 5); do
    if restore_stock; then
      stock_restored=1
      break
    fi
    sleep 2
  done
  (( stock_restored == 1 )) || die "could not restore stock UVM after custom E3 cell"
  CUSTOM_LOADED=0
  module_snapshot "${RUN_ROOT}/provenance/module.stock_for_kt.json" "${STOCK_UVM}"
  restart_exporters
fi

if [[ "${RUN_KT}" == 1 ]]; then
for ((block=0; block<PROCESS_BLOCKS; ++block)); do
  for wi in "${!workloads[@]}"; do
    workload="${workloads[$wi]}"; prompt="${prompts[$wi]}"; output="${outputs[$wi]}"
    contains_csv "${workload}" "${WORKLOAD_FILTER}" || continue
    fixture="${FIXTURE_ROOT}/evaluation_${workload}.json"
    supported=()
    max_kv=0
    for batch in "${batches[@]}"; do
      contains_csv "${batch}" "${BATCH_FILTER}" || continue
      kv="$(kv_bytes "${prompt}" "${output}" "${batch}")"
      if (( kv <= KT_KV_LIMIT_BYTES )); then supported+=("${batch}"); ((kv > max_kv)) && max_kv="${kv}"; fi
    done
    batch_csv="$(IFS=,; echo "${supported[*]}")"
    label="b${block}_${workload}"
    case_dir="${RUN_ROOT}/kt/${label}"
    mkdir "${case_dir}"
    if [[ -z "${batch_csv}" ]]; then
      jq -n '{status:"N/A",reason:"all batches exceed pre-registered KT KV headroom"}' >"${case_dir}/na.json"
      printf '%s\t%s\t%s\t%s\t-\t%s\t%s\tN/A\t-\n' "${block}" "${workload}" "${prompt}" "${output}" "${label}" "${max_kv}" >>"${RUN_ROOT}/kt/execution.tsv"
      continue
    fi
    context=$((prompt + output + 16))
    total_tokens=$(( (prompt + output + 16) * supported[${#supported[@]}-1] + 128 ))
    prefill=$((prompt * supported[${#supported[@]}-1])); (( prefill < context )) && prefill="${context}"
    command=(taskset -c "${CPUSET}" "${KT_PY}" -m sglang.launch_server --model-path "${MODEL}"
      --host 127.0.0.1 --port "${PORT}" --tp-size 1 --context-length "${context}"
      --max-running-requests "${supported[${#supported[@]}-1]}" --max-total-tokens "${total_tokens}"
      --chunked-prefill-size "${prefill}" --mem-fraction-static 0.97 --max-prefill-tokens "${prefill}"
      --schedule-policy fcfs --stream-interval 1 --random-seed 0 --kt-weight-path "${MODEL}" --kt-method "${KT_METHOD}"
      --kt-cpuinfer 14 --kt-threadpool-count 1 --kt-numa-nodes 0 --kt-max-deferred-experts-per-token 0
      --kt-gpu-prefill-token-threshold 0 --kt-gpu-experts-ratio "${KT_GPU_EXPERTS_RATIO}" --kt-expert-placement-strategy frequency
      --init-expert-location "${KT_STATS}" --attention-backend flashinfer
      --disable-shared-experts-fusion --disable-radix-cache --enable-metrics --enable-mixed-chunk --disable-cuda-graph)
    if [[ "${KT_METHOD}" == FP8 ]]; then command+=(--fp8-gemm-backend triton); fi
    if [[ "${KT_TRUST_REMOTE_CODE}" == 1 ]]; then command+=(--trust-remote-code); fi
    printf '%q ' "${command[@]}" >"${case_dir}/server_command.txt"; printf '\n' >>"${case_dir}/server_command.txt"
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,clocks.sm,temperature.gpu \
      --format=csv,noheader,nounits -lms 200 >"${case_dir}/nvml.csv" 2>"${case_dir}/nvml.err" &
    MONITOR_PID="$!"
    setsid env PATH="${KT_ROOT}/.venv/bin:${PATH}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      TOKENIZERS_PARALLELISM=false "${command[@]}" >"${case_dir}/server.log" 2>&1 &
    SERVER_PID="$!"; echo "${SERVER_PID}" >"${case_dir}/server.pid"
    ready=0
    for second in $(seq 1 900); do
      if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then ready=1; echo "${second}" >"${case_dir}/startup_seconds.txt"; break; fi
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    rc=1
    if [[ "${ready}" == 1 ]]; then
      measured=$((output - 6))
      set +e
      "${KT_PY}" "${KT_RUNNER}" --formal-e3 --fixture "${fixture}" --output "${case_dir}/result.json" \
        --base-url "http://127.0.0.1:${PORT}" --batch-sizes "${batch_csv}" --conditioning-repeats 1 \
        --measurement-repeats 1 --warmup-decode-steps 5 --measured-decode-steps "${measured}" \
        --timeout-seconds 3600 >"${case_dir}/client.stdout" 2>"${case_dir}/client.stderr"
      rc=$?
      set +e
      curl -fsS "http://127.0.0.1:${PORT}/metrics" >"${case_dir}/metrics_final.txt" 2>/dev/null || true
    fi
    stop_server
    status=error
    if (( rc == 0 )) && jq -e --argjson out "${output}" '.status == "ok" and (.cycles|map(select(.role=="measurement"))|all(.output_token_ids|all(length == $out)))' "${case_dir}/result.json" >/dev/null 2>&1 \
      && [[ "$(rg -c "KT GPU experts: layer [0-9]+ \\(MoE\\) has ${KT_EXPECTED_GPU_EXPERTS} GPU experts" "${case_dir}/server.log")" == "${KT_EXPECTED_MOE_LAYERS}" ]]; then status=ok; fi
    echo "${rc}" >"${case_dir}/returncode.txt"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${block}" "${workload}" "${prompt}" "${output}" "${batch_csv}" "${label}" "${max_kv}" "${status}" "${rc}" >>"${RUN_ROOT}/kt/execution.tsv"
  done
done
fi

module_snapshot "${RUN_ROOT}/provenance/module.after.json" "${STOCK_UVM}"
echo "RUN_ROOT=${RUN_ROOT}"
