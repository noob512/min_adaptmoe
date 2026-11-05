#!/usr/bin/env bash
set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EVAL_DIR}/../.." && pwd)"
VLLM_PY="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
KT_PY="${REPO_ROOT}/KTransformers/.venv/bin/python"
MODEL="/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7"
FIXTURE_ROOT="${EVAL_DIR}/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16"
PROFILE="${E2_C16_PROFILE:-${EVAL_DIR}/configs/e2_c16_direct_batch_matrix.json}"
NATIVE_ENV="${EVAL_DIR}/configs/c1_native_common_env.json"
OURS_ENV="${EVAL_DIR}/configs/c1_ours_runtime_hot_audit_off_env.json"
KT_ENV="${EVAL_DIR}/configs/e2_ktransformers_env.json"
MOEI_ENV="${EVAL_DIR}/configs/e2_moe_infinity_env.json"
VLLM_RUNNER="${EVAL_DIR}/direct_batch_c16_vllm_runner.py"
KT_RUNNER="${EVAL_DIR}/direct_batch_c16_ktransformers_runner.py"
MOEI_RUNNER="${EVAL_DIR}/direct_batch_c16_moe_infinity_runner.py"
C16_VERIFY="${EVAL_DIR}/verify_direct_batch_c16_case.py"
DUAL_VIEW_VERIFY="${EVAL_DIR}/verify_ours_full_dual_view.py"
TRACE="${FIXTURE_ROOT}/calibration_assets/routing/routing_trace.jsonl"
PLAN_BUILDER="${REPO_ROOT}/experiments/hybrid_offload_2_4_6_8_compare/prepare_cold_plans.py"
KT_PLAN_BUILDER="${REPO_ROOT}/experiments/hybrid_offload_2_4_6_8_compare/prepare_kt_frequency_plans.py"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
MOEI_OFFLOAD="/home/ubuntu/moe_infinity_offload/deepseek_v2_lite_chat_85864749cd611b4353ce1decdb286193298f64c7"
MOEI_RECEIPT="${EVAL_DIR}/artifacts/c1-mechanism-min-offload-pilot-20260827-attempt-01/provenance/moe_infinity_store_build/store_build_receipt.json"
RUN_ROOT="${1:-${EVAL_DIR}/artifacts/e2-c16-direct-batch-$(date -u +%Y%m%dT%H%M%SZ)}"
CASE_TIMEOUT_SECONDS="${E2_C16_CASE_TIMEOUT_SECONDS:-14400}"
SYSTEM_FILTER="${E2_C16_SYSTEMS:-native_all_gpu,native_pinned_uva,native_copy_on_forward,ours,ktransformers,moe_infinity}"
WORKLOAD_FILTER="${E2_C16_WORKLOADS:-short_dialogue,ordinary_chat,rag_summary,long_generation}"
CONCURRENCY="${E2_DIRECT_CONCURRENCY:-16}"

die() { echo "error: $*" >&2; exit 1; }
contains_csv() { [[ ",$2," == *",$1,"* ]]; }

if [[ -e "${RUN_ROOT}" && "${E2_C16_RESUME:-0}" != 1 ]]; then
  die "refusing to overwrite ${RUN_ROOT}; set E2_C16_RESUME=1 to resume"
fi
for path in "${MODEL}/config.json" "${VLLM_PY}" "${KT_PY}" "${PROFILE}" \
  "${NATIVE_ENV}" "${OURS_ENV}" "${KT_ENV}" "${MOEI_ENV}" "${VLLM_RUNNER}" \
  "${KT_RUNNER}" "${MOEI_RUNNER}" "${C16_VERIFY}" "${TRACE}" "${PLAN_BUILDER}" \
  "${KT_PLAN_BUILDER}" "${BASE_HARNESS}" "${MOEI_RECEIPT}" \
  "${DUAL_VIEW_VERIFY}"; do
  [[ -e "${path}" ]] || die "required path is missing: ${path}"
done
[[ -d "${MOEI_OFFLOAD}" ]] || die "MoE-Infinity offload store is missing"
command -v jq >/dev/null || die "jq is unavailable"
command -v flock >/dev/null || die "flock is unavailable"
[[ "${CONCURRENCY}" == 1 || "${CONCURRENCY}" == 16 ]] \
  || die "E2_DIRECT_CONCURRENCY must be 1 or 16"
if [[ "${CONCURRENCY}" != 16 ]] \
    && contains_csv moe_infinity "${SYSTEM_FILTER}"; then
  die "generic direct runner does not support C1 for MoE-Infinity"
fi

exec 7>"/tmp/eurosys_c16_direct_batch_matrix.lock"
flock -n 7 || die "another C16 direct-batch matrix holds the GPU lock"
exec 8>"/tmp/eurosys_moe_offload_e2_gpu0.lock"
flock -n 8 || die "another E2 runner holds the GPU lock"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
export ALLOW_GPU_RUN=1
export SLOT_BUDGET_BYTES=1207959552
export MANUAL_MIGRATION_FAULT_MODE=2
if contains_csv ours "${SYSTEM_FILTER}"; then
  export STATIC_ALIAS_DUAL_VIEW_ENABLE=1
  export STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL=1
  export ACCESS_COUNTER_MIGRATION_ENABLE=0
  export UVM_PREFETCH_ENABLE=0
  export UVM_THRASHING_ENABLE=0
fi
# shellcheck source=/dev/null
source "${BASE_HARNESS}"

mkdir -p "${RUN_ROOT}/provenance" "${RUN_ROOT}/cases" "${RUN_ROOT}/retries"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
RUN_ROOT_ACTIVE="${RUN_ROOT}"
EXECUTION_TSV="${RUN_ROOT}/execution.tsv"
if [[ ! -f "${EXECUTION_TSV}" ]]; then
  printf 'created_at\tarm\tworkload\tsetting\tcase_root\treturncode\tverification\n' >"${EXECUTION_TSV}"
fi

profile_value() {
  jq -er --arg workload "$1" --arg key "$2" \
    '.workloads[] | select(.name == $workload) | .[$key]' "${PROFILE}"
}

profile_value_default() {
  jq -cr --arg workload "$1" --arg key "$2" --argjson default "$3" \
    '.workloads[] | select(.name == $workload) | (.[$key] // $default)' \
    "${PROFILE}"
}

profile_array() {
  jq -er --arg workload "$1" --arg key "$2" \
    '.workloads[] | select(.name == $workload) | .[$key][]' "${PROFILE}"
}

case_already_ok() {
  [[ -f "$1/verification.json" ]] \
    && jq -e '.status == "ok"' "$1/verification.json" >/dev/null
}

arm_workload_already_ok() {
  find "${RUN_ROOT}/cases/$1/$2" -mindepth 2 -maxdepth 2 \
    -name verification.json -exec jq -e '.status == "ok"' {} \; \
    2>/dev/null | grep -q true
}

start_sampler() {
  nvidia-smi \
    --query-gpu=timestamp,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,clocks.current.sm,clocks.current.memory,pstate \
    --format=csv,noheader,nounits --loop-ms=200 \
    >"$1/nvml.csv" 2>"$1/nvml.err" &
  SAMPLER_PID=$!
}

stop_sampler() {
  kill -TERM "${SAMPLER_PID}" 2>/dev/null || true
  set +e
  wait "${SAMPLER_PID}"
  local sampler_rc=$?
  set -e
  printf '%s\n' "${sampler_rc}" >"$1/nvml_returncode.txt"
}

load_env_file() {
  local env_file="$1" key value
  while IFS=$'\t' read -r key value; do
    [[ "${key}" =~ ^[A-Z_][A-Z0-9_]*$ ]] || die "invalid env key: ${key}"
    export "${key}=${value}"
  done < <(jq -r 'to_entries[] | [.key, .value] | @tsv' "${env_file}")
}

run_case() {
  local arm="$1" workload="$2" setting="$3" case_root="$4" env_file="$5" python="$6"
  shift 6
  local kernel_epoch rc verify_rc
  local -a command=("$@")
  mkdir -p "${case_root}"
  gpu_state_json "${case_root}/gpu.before.json"
  kernel_epoch="$(date +%s)"
  {
    printf 'cd %q && ' "${REPO_ROOT}"
    printf 'timeout --kill-after=30s %qs taskset -c 0-13 %q ' \
      "${CASE_TIMEOUT_SECONDS}" "${python}"
    printf '%q ' "${command[@]}"
    printf '\n'
  } >"${case_root}/command.txt"
  start_sampler "${case_root}"
  set +e
  (
    load_env_file "${env_file}"
    cd "${REPO_ROOT}"
    /usr/bin/time -v -o "${case_root}/resource_time.txt" \
      timeout --kill-after=30s "${CASE_TIMEOUT_SECONDS}s" \
      taskset -c 0-13 "${python}" "${command[@]}"
  ) >"${case_root}/run.log" 2>&1
  rc=$?
  set -e
  stop_sampler "${case_root}"
  printf '%s\n' "${rc}" >"${case_root}/returncode.txt"
  sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
    >"${case_root}/kernel.since.log" 2>"${case_root}/kernel.since.err" || true
  if ! wait_gpu_clean; then
    gpu_state_json "${case_root}/gpu.after.json"
    die "GPU did not clean up after ${arm}/${workload}/${setting}"
  fi
  gpu_state_json "${case_root}/gpu.after.json"
  set +e
  "${VLLM_PY}" "${C16_VERIFY}" "${case_root}" --arm "${arm}" \
    >"${case_root}/verify.stdout.txt" 2>"${case_root}/verify.stderr.txt"
  verify_rc=$?
  if [[ "${arm}" == ours && "${E2_SKIP_DUAL_VIEW_VERIFY:-0}" != 1 ]]; then
    "${VLLM_PY}" "${DUAL_VIEW_VERIFY}" "${case_root}" \
      >>"${case_root}/verify.stdout.txt" 2>>"${case_root}/verify.stderr.txt"
    (( $? == 0 )) || verify_rc=1
  fi
  set -e
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${arm}" "${workload}" \
    "${setting}" "${case_root}" "${rc}" "${verify_rc}" >>"${EXECUTION_TSV}"
  return "${verify_rc}"
}

runtime_preflight
module_snapshot "${RUN_ROOT}/module.before.json" "${STOCK_UVM}"
cp "${PROFILE}" "${RUN_ROOT}/provenance/profile.json"
git -C "${REPO_ROOT}" status --short >"${RUN_ROOT}/provenance/git_status.txt"
sha256sum "${PROFILE}" "${VLLM_RUNNER}" "${KT_RUNNER}" "${MOEI_RUNNER}" \
  "${C16_VERIFY}" "${TRACE}" "${MODEL}/config.json" \
  "${REPO_ROOT}/MoE-Infinity/moe_infinity/serving/model_runner.py" \
  "${REPO_ROOT}/MoE-Infinity/moe_infinity/serving/engine.py" \
  >"${RUN_ROOT}/provenance/input_sha256.txt"

PLAN_ROOT="${RUN_ROOT}/provenance/cold_plans"
if { contains_csv ours "${SYSTEM_FILTER}" || contains_csv ktransformers "${SYSTEM_FILTER}"; } \
    && [[ ! -f "${PLAN_ROOT}/manifest.json" ]]; then
  widths=""
  for cold in $(seq -w 1 24); do
    [[ -z "${widths}" ]] || widths+=","
    widths+="cold${cold}=$((10#${cold}))"
  done
  mkdir -p "${PLAN_ROOT}"
  "${VLLM_PY}" "${PLAN_BUILDER}" --trace "${TRACE}" --output-dir "${PLAN_ROOT}" \
    --serving-start-step 3 --prefill-step-count 1 --tier-widths "${widths}" \
    >"${RUN_ROOT}/provenance/prepare_cold_plans.stdout.json"
fi

if contains_csv ktransformers "${SYSTEM_FILTER}"; then
  for cold in $(seq -w 1 24); do
    KT_INPUT="${RUN_ROOT}/provenance/kt_inputs/cold${cold}"
    KT_OUTPUT="${RUN_ROOT}/provenance/kt_plans/cold${cold}"
    [[ -f "${KT_OUTPUT}/kt_frequency_2g.pt" ]] && continue
    mkdir -p "${KT_INPUT}" "${KT_OUTPUT}"
    for tier in 2g 4g 6g 8g; do
      cp "${PLAN_ROOT}/cold_plan_cold${cold}.json" "${KT_INPUT}/cold_plan_${tier}.json"
    done
    "${VLLM_PY}" "${KT_PLAN_BUILDER}" --input-dir "${KT_INPUT}" \
      --output-dir "${KT_OUTPUT}" >"${KT_OUTPUT}/prepare.stdout.txt"
  done
fi

run_native_workload() {
  local arm="$1" workload="$2" backend="" cpu_gb tag case_root effective_env
  local exact_names_file="" layer_count="" setting="" tmp_env exact_name_count="0"
  local prompt output max_len kv_bytes max_batched ttft tpot
  local -a candidates
  prompt="$(profile_value "${workload}" prompt_tokens)"
  output="${E2_MEASUREMENT_OUTPUT_TOKENS:-$(profile_value "${workload}" output_tokens)}"
  max_len="$(profile_value "${workload}" max_model_len)"
  kv_bytes="$(profile_value "${workload}" kv_cache_bytes)"
  max_batched="$(profile_value "${workload}" max_batched_tokens)"
  ttft="$(profile_value_default "${workload}" ttft_slo_ms 30000)"
  tpot="$(profile_value_default "${workload}" tpot_slo_ms 1000)"
  arm_workload_already_ok "${arm}" "${workload}" && return 0
  case "${arm}" in
    native_all_gpu) backend=""; candidates=(0) ;;
    native_pinned_uva) backend=uva; mapfile -t candidates < <(profile_array "${workload}" native_offload_gb_candidates) ;;
    native_copy_on_forward)
      backend=copy
      exact_name_count="$(profile_value_default "${workload}" native_copy_exact_parameter_names '[]' | jq 'length')"
      if [[ "${exact_name_count}" -gt 0 ]]; then
        candidates=(exact_names)
      elif [[ "$(profile_value "${workload}" native_copy_exact_w2_layer_candidates | jq 'length')" -gt 0 ]]; then
        mapfile -t candidates < <(profile_array "${workload}" native_copy_exact_w2_layer_candidates)
      else
        mapfile -t candidates < <(profile_array "${workload}" native_offload_gb_candidates)
      fi
      ;;
    *) die "unknown native arm ${arm}" ;;
  esac
  for cpu_gb in "${candidates[@]}"; do
    exact_names_file=""
    layer_count=""
    if [[ "${arm}" == native_copy_on_forward && "${exact_name_count}" -gt 0 ]]; then
      cpu_gb="$(profile_value "${workload}" native_copy_exact_offload_gb)"
      tag="exact_names"
      setting="exact_parameter_names=${exact_name_count};cpu_offload_gb=${cpu_gb}"
    elif [[ "${arm}" == native_copy_on_forward \
          && "$(profile_value "${workload}" native_copy_exact_w2_layer_candidates | jq 'length')" -gt 0 ]]; then
      layer_count="${cpu_gb}"
      cpu_gb="$(jq -nr --argjson layers "${layer_count}" '$layers * 0.34375')"
      tag="w2_layers_${layer_count}"
      setting="exact_w2_layers=${layer_count};cpu_offload_gb=${cpu_gb}"
    else
      tag="${cpu_gb/./p}g"
      setting="cpu_offload_gb=${cpu_gb}"
    fi
    case_root="${RUN_ROOT}/cases/${arm}/${workload}/offload_${tag}"
    case_already_ok "${case_root}" && return 0
    [[ ! -e "${case_root}" ]] || continue
    mkdir -p "${case_root}"
    if [[ "${exact_name_count}" -gt 0 ]]; then
      exact_names_file="${case_root}/exact_parameter_names.json"
      profile_value "${workload}" native_copy_exact_parameter_names >"${exact_names_file}"
    elif [[ -n "${layer_count}" ]]; then
      exact_names_file="${case_root}/exact_w2_parameter_names.json"
      jq -n --argjson layers "${layer_count}" \
        '[range(1; $layers + 1) | "model.layers.\(.).mlp.experts.w2_weight"]' \
        >"${exact_names_file}"
    fi
    effective_env="${case_root}/environment.json"
    jq --arg backend "${backend}" --arg manifest "${case_root}/offload_manifest.jsonl" \
      --arg exact_names_file "${exact_names_file}" \
      '. + {
        PYTORCH_ALLOC_CONF: "expandable_segments:True",
        VLLM_UVM_PHASE_BATCHED_SCHEDULING: "1",
        VLLM_UVM_PHASE_KV_AWARE_COHORTS: "1",
        VLLM_UVM_PHASE_COHORT_KV_RESERVE_BLOCKS: "1"
      }
      | if $backend == "" then
          del(.VLLM_CPU_OFFLOAD_BACKEND, .VLLM_CPU_OFFLOAD_MANIFEST_FILE)
        else
          . + {VLLM_CPU_OFFLOAD_BACKEND: $backend,
               VLLM_CPU_OFFLOAD_MANIFEST_FILE: $manifest,
               VLLM_CPU_OFFLOAD_MANIFEST_RESET: "1"}
        end
      | if $backend == "copy" then
          . + {VLLM_CPU_OFFLOAD_COPY_SYNC: "0", VLLM_CPU_OFFLOAD_COPY_TRACE: "0"}
          | if $exact_names_file != "" then
              . + {VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE: $exact_names_file}
            else del(.VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE) end
        else . end' "${NATIVE_ENV}" >"${effective_env}"
    if [[ ( "${arm}" == native_pinned_uva \
          || "${arm}" == native_copy_on_forward ) \
          && ( "${E2_C16_FORCE_CUDAGRAPH:-0}" == 1 \
               || "${E2_C16_CUDAGRAPH_MODE:-NONE}" != NONE ) ]]; then
      tmp_env="${effective_env}.tmp"
      jq '. + {
        VLLM_TRITON_MLA_EXPERIMENTAL_FULL_CUDAGRAPH: "1",
        VLLM_CUDAGRAPH_DISPATCH_AUDIT: "1"
      }' \
        "${effective_env}" >"${tmp_env}"
      mv "${tmp_env}" "${effective_env}"
    fi
    if [[ "${arm}" == native_copy_on_forward \
          && "${E2_C16_NATIVE_COPY_STATIC_BANK:-0}" == 1 ]]; then
      tmp_env="${effective_env}.tmp"
      jq '. + {
        VLLM_CPU_OFFLOAD_STATIC_BANK: "1",
        VLLM_DISABLE_SHARED_EXPERTS_STREAM: "1"
      }' \
        "${effective_env}" >"${tmp_env}"
      mv "${tmp_env}" "${effective_env}"
    fi
    if run_case "${arm}" "${workload}" "${setting}" \
      "${case_root}" "${effective_env}" "${VLLM_PY}" "${VLLM_RUNNER}" \
      --arm "${arm}" --model "${MODEL}" \
      --fixture "${FIXTURE_ROOT}/evaluation_${workload}.json" \
      --warmup-fixture "${FIXTURE_ROOT}/calibration_${workload}.json" \
      --workload "${workload}" --concurrency "${CONCURRENCY}" \
      --prompt-tokens "${prompt}" --output-tokens "${output}" \
      --warmup-requests "${E2_C16_WARMUP_REQUESTS:-3}" \
      --max-model-len "${max_len}" --max-num-batched-tokens "${max_batched}" \
      --kv-cache-memory-bytes "${kv_bytes}" --cpu-offload-gb "${cpu_gb}" \
      --ttft-slo-ms "${ttft}" --tpot-slo-ms "${tpot}" \
      --output "${case_root}/result.json"; then
      return 0
    fi
  done
  return 1
}

run_ours_workload() {
  local workload="$1" cold cold_tag case_root effective_env plan compilation
  local prompt output max_len kv_bytes max_batched prefill ttft tpot
  local -a candidates
  prompt="$(profile_value "${workload}" prompt_tokens)"
  output="${E2_MEASUREMENT_OUTPUT_TOKENS:-$(profile_value "${workload}" output_tokens)}"
  max_len="$(profile_value "${workload}" max_model_len)"
  kv_bytes="$(profile_value "${workload}" kv_cache_bytes)"
  max_batched="$(profile_value_default "${workload}" ours_max_batched_tokens \
    "$(profile_value "${workload}" max_batched_tokens)")"
  prefill="$(profile_value "${workload}" prefill_graph_tokens)"
  ttft="$(profile_value_default "${workload}" ttft_slo_ms 30000)"
  tpot="$(profile_value_default "${workload}" tpot_slo_ms 1000)"
  arm_workload_already_ok ours "${workload}" && return 0
  compilation="$(jq -cn --argjson prefill "${prefill}" \
    --argjson concurrency "${CONCURRENCY}" \
    '{mode:"NONE",cudagraph_mode:"FULL_SEPARATE",
      cudagraph_capture_sizes:([1,$concurrency,$prefill]|unique|sort),
      max_cudagraph_capture_size:$prefill}')"
  mapfile -t candidates < <(profile_array "${workload}" ours_cold_candidates)
  for cold in "${candidates[@]}"; do
    cold_tag="$(printf '%02d' "${cold}")"
    case_root="${RUN_ROOT}/cases/ours/${workload}/cold_${cold_tag}"
    case_already_ok "${case_root}" && return 0
    [[ ! -e "${case_root}" ]] || continue
    mkdir -p "${case_root}"
    plan="${PLAN_ROOT}/cold_plan_cold${cold_tag}.json"
    effective_env="${case_root}/environment.json"
    jq --arg names "${PLAN_ROOT}/exact_parameter_names.txt" --arg plan "${plan}" \
      --arg manifest "${case_root}/offload_manifest.jsonl" \
      --arg uvm_log "${case_root}/uvm.log" \
      --arg phase "${case_root}/phase_transition.json" \
      --arg kv "${kv_bytes}" --arg prefill "${prefill}" \
      '. + {
        VLLM_CPU_OFFLOAD_PLAN_CLASS_IMMUTABLE_CPU_SHADOW: "1",
        VLLM_CPU_OFFLOAD_DUAL_VIEW_FULL_PARAMETER_SHADOW: "1",
        VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW: "1",
        VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE: $names,
        VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE: $plan,
        VLLM_CPU_OFFLOAD_MANIFEST_FILE: $manifest,
        VLLM_UVM_LOG_FILE: $uvm_log,
        VLLM_UVM_ORIGINAL_VA_PHASE_RESULT_FILE: $phase,
        VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_BYTES: $kv,
        VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS: $prefill,
        VLLM_UVM_FUSED_MOE_POST_WORKSPACE_AUDIT_MIN_TOKENS: $prefill,
        VLLM_FUSED_MOE_CHUNK_SIZE: $prefill,
        VLLM_UVM_RUNTIME_HOT_RESIDENCY_AUDIT: "0",
        VLLM_UVM_RUNTIME_POST_MAP_RESIDENCY_AUDIT: "0",
        VLLM_UVM_PHASE_BATCHED_SCHEDULING: "1",
        VLLM_UVM_PHASE_KV_AWARE_COHORTS: "1",
        VLLM_UVM_PHASE_COHORT_KV_RESERVE_BLOCKS: "1",
        VLLM_CUDAGRAPH_DISPATCH_AUDIT: "1"
      } | del(.VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE)' \
      "${OURS_ENV}" >"${effective_env}"
    if run_case ours "${workload}" "cold_per_layer=${cold}" \
      "${case_root}" "${effective_env}" "${VLLM_PY}" "${VLLM_RUNNER}" \
      --arm ours --model "${MODEL}" \
      --fixture "${FIXTURE_ROOT}/evaluation_${workload}.json" \
      --warmup-fixture "${FIXTURE_ROOT}/calibration_${workload}.json" \
      --workload "${workload}" --concurrency "${CONCURRENCY}" \
      --prompt-tokens "${prompt}" --output-tokens "${output}" \
      --max-model-len "${max_len}" --max-num-batched-tokens "${max_batched}" \
      --kv-cache-memory-bytes "${kv_bytes}" --cpu-offload-gb 30 \
      --ttft-slo-ms "${ttft}" --tpot-slo-ms "${tpot}" \
      --output "${case_root}/result.json"; then
      return 0
    fi
  done
  return 1
}

run_kt_workload() {
  local workload="$1" cold cold_tag case_root context max_total gpu_experts effective_env
  local prefill_threshold
  local prompt output max_len max_batched ttft tpot
  local -a candidates
  prompt="$(profile_value "${workload}" prompt_tokens)"
  output="$(profile_value "${workload}" output_tokens)"
  max_len="$(profile_value "${workload}" max_model_len)"
  max_batched="$(profile_value "${workload}" max_batched_tokens)"
  ttft="$(profile_value_default "${workload}" ttft_slo_ms 30000)"
  tpot="$(profile_value_default "${workload}" tpot_slo_ms 1000)"
  context=$((max_len + 2))
  max_total="$(profile_value_default "${workload}" kt_max_total_tokens \
    "$((context * CONCURRENCY))")"
  prefill_threshold="${E2_KT_GPU_PREFILL_TOKEN_THRESHOLD:-0}"
  if [[ -n "${E2_KT_GPU_PREFILL_TOKEN_THRESHOLD_BY_WORKLOAD:-}" ]]; then
    prefill_threshold="$(jq -er --arg workload "${workload}" \
      '.[$workload]' <<<"${E2_KT_GPU_PREFILL_TOKEN_THRESHOLD_BY_WORKLOAD}")" \
      || die "E2_KT_GPU_PREFILL_TOKEN_THRESHOLD_BY_WORKLOAD lacks ${workload}"
  fi
  [[ "${prefill_threshold}" =~ ^[0-9]+$ ]] \
    || die "KTransformers GPU Prefill threshold must be non-negative"
  arm_workload_already_ok ktransformers "${workload}" && return 0
  local cold_override="${E2_KT_CPU_EXPERTS_PER_LAYER:-}"
  if [[ -n "${E2_KT_CPU_EXPERTS_BY_WORKLOAD:-}" ]]; then
    cold_override="$(jq -er --arg workload "${workload}" \
      '.[$workload]' <<<"${E2_KT_CPU_EXPERTS_BY_WORKLOAD}")" \
      || die "E2_KT_CPU_EXPERTS_BY_WORKLOAD lacks ${workload}"
  fi
  if [[ -n "${cold_override}" ]]; then
    [[ "${cold_override}" =~ ^[0-9]+$ \
       && "${cold_override}" -ge 1 \
       && "${cold_override}" -le 63 ]] \
      || die "E2_KT_CPU_EXPERTS_PER_LAYER must be in [1, 63]"
    candidates=("${cold_override}")
  else
    mapfile -t candidates < <(profile_array "${workload}" kt_cold_candidates)
  fi
  for cold in "${candidates[@]}"; do
    cold_tag="$(printf '%02d' "${cold}")"
    gpu_experts=$((64 - cold))
    case_root="${RUN_ROOT}/cases/ktransformers/${workload}/cold_${cold_tag}"
    case_already_ok "${case_root}" && return 0
    [[ ! -e "${case_root}" ]] || continue
    effective_env="${case_root}/environment.json"
    mkdir -p "${case_root}"
    jq --arg threads "${E2_KT_CPU_THREADS:-14}" \
      '. + {OMP_NUM_THREADS: $threads, MKL_NUM_THREADS: $threads}' \
      "${KT_ENV}" >"${effective_env}"
    if run_case ktransformers "${workload}" \
      "cold_per_layer=${cold},cpu_threads=${E2_KT_CPU_THREADS:-14},gpu_prefill_token_threshold=${prefill_threshold}" \
      "${case_root}" "${effective_env}" "${KT_PY}" "${KT_RUNNER}" \
      --model "${MODEL}" --fixture "${FIXTURE_ROOT}/evaluation_${workload}.json" \
      --warmup-fixture "${FIXTURE_ROOT}/calibration_${workload}.json" \
      --frequency-plan "${RUN_ROOT}/provenance/kt_plans/cold${cold_tag}/kt_frequency_2g.pt" \
      --workload "${workload}" --concurrency "${CONCURRENCY}" \
      --prompt-tokens "${prompt}" --output-tokens "${output}" \
      --gpu-experts "${gpu_experts}" --cpu-threads "${E2_KT_CPU_THREADS:-14}" \
      --gpu-prefill-token-threshold "${prefill_threshold}" \
      --context-length "${context}" \
      --max-total-tokens "${max_total}" --max-prefill-tokens "${max_batched}" \
      --ttft-slo-ms "${ttft}" --tpot-slo-ms "${tpot}" \
      --output "${case_root}/result.json"; then
      return 0
    fi
  done
  return 1
}

validate_moei_receipt() {
  local extension expected actual
  extension="$(find "${REPO_ROOT}/MoE-Infinity/moe_infinity" -maxdepth 1 -type f -name '_store*.so' -print -quit)"
  [[ -n "${extension}" ]] || die "MoE-Infinity _store extension is missing"
  expected="$(jq -r '.extension_sha256' "${MOEI_RECEIPT}")"
  actual="$(sha256sum "${extension}" | awk '{print $1}')"
  [[ "${expected}" == "${actual}" ]] || die "MoE-Infinity extension differs from receipt"
}

run_moei_workload() {
  local workload="$1" ratio ratio_tag kv_ratio kv_tag case_root
  local prompt output max_len max_batched ttft tpot blocks block_bytes available
  local -a candidates
  prompt="$(profile_value "${workload}" prompt_tokens)"
  output="$(profile_value "${workload}" output_tokens)"
  max_len="$(profile_value "${workload}" max_model_len)"
  max_batched="$(profile_value "${workload}" max_batched_tokens)"
  ttft="$(profile_value "${workload}" ttft_slo_ms)"
  tpot="$(profile_value "${workload}" tpot_slo_ms)"
  blocks="$(profile_value "${workload}" moe_infinity_kv_blocks)"
  block_bytes="$(jq -er '.moe_infinity_kv_calibration.block_bytes' "${PROFILE}")"
  available="$(jq -er '.moe_infinity_kv_calibration.available_bytes' "${PROFILE}")"
  arm_workload_already_ok moe_infinity "${workload}" && return 0
  mapfile -t candidates < <(profile_array "${workload}" moe_infinity_device_ratio_candidates)
  for ratio in "${candidates[@]}"; do
    kv_ratio="$(jq -nr --argjson blocks "${blocks}" --argjson block_bytes "${block_bytes}" \
      --argjson available "${available}" --argjson ratio "${ratio}" \
      '((($blocks*$block_bytes)/($available*$ratio)*10000000)|ceil)/10000000')"
    ratio_tag="${ratio/./p}"
    kv_tag="${kv_ratio/./p}"
    case_root="${RUN_ROOT}/cases/moe_infinity/${workload}/ratio_${ratio_tag}_kv_${kv_tag}"
    case_already_ok "${case_root}" && return 0
    [[ ! -e "${case_root}" ]] || continue
    if run_case moe_infinity "${workload}" \
      "device_ratio=${ratio},kv_ratio=${kv_ratio},kv_blocks=${blocks}" \
      "${case_root}" "${MOEI_ENV}" "${VLLM_PY}" "${MOEI_RUNNER}" \
      --model "${MODEL}" --offload-path "${MOEI_OFFLOAD}" \
      --fixture "${FIXTURE_ROOT}/evaluation_${workload}.json" \
      --warmup-fixture "${FIXTURE_ROOT}/calibration_${workload}.json" \
      --workload "${workload}" --prompt-tokens "${prompt}" --output-tokens "${output}" \
      --max-seq-length "${max_len}" --max-tokens-per-step "${max_batched}" \
      --device-memory-ratio "${ratio}" --kv-cache-ratio "${kv_ratio}" \
      --expected-kv-blocks "${blocks}" --ttft-slo-ms "${ttft}" \
      --tpot-slo-ms "${tpot}" --output "${case_root}/result.json"; then
      return 0
    fi
  done
  return 1
}

total_failures=0
for arm in native_all_gpu native_pinned_uva native_copy_on_forward; do
  contains_csv "${arm}" "${SYSTEM_FILTER}" || continue
  for workload in short_dialogue ordinary_chat rag_summary long_generation; do
    contains_csv "${workload}" "${WORKLOAD_FILTER}" || continue
    run_native_workload "${arm}" "${workload}" || total_failures=$((total_failures + 1))
  done
done

if contains_csv ours "${SYSTEM_FILTER}"; then
  load_custom_module
  module_snapshot "${RUN_ROOT}/module.custom.json" "${CUSTOM_UVM}"
  for workload in short_dialogue ordinary_chat rag_summary long_generation; do
    contains_csv "${workload}" "${WORKLOAD_FILTER}" || continue
    run_ours_workload "${workload}" || total_failures=$((total_failures + 1))
  done
  restore_stock || die "failed to restore stock nvidia_uvm"
  module_snapshot "${RUN_ROOT}/module.after_ours.json" "${STOCK_UVM}"
fi

if contains_csv ktransformers "${SYSTEM_FILTER}"; then
  for workload in short_dialogue ordinary_chat rag_summary long_generation; do
    contains_csv "${workload}" "${WORKLOAD_FILTER}" || continue
    run_kt_workload "${workload}" || total_failures=$((total_failures + 1))
  done
fi

if contains_csv moe_infinity "${SYSTEM_FILTER}"; then
  validate_moei_receipt
  for workload in short_dialogue ordinary_chat rag_summary long_generation; do
    contains_csv "${workload}" "${WORKLOAD_FILTER}" || continue
    run_moei_workload "${workload}" || total_failures=$((total_failures + 1))
  done
fi

printf '%s\n' "${total_failures}" >"${RUN_ROOT}/failure_count.txt"
module_snapshot "${RUN_ROOT}/module.after.json" "${STOCK_UVM}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "FAILURES=${total_failures}"
(( total_failures == 0 ))
