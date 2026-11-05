#!/usr/bin/env bash
set -euo pipefail

MATRIX_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${MATRIX_SCRIPT_DIR}/../.." && pwd)"
RUN_ROOT="${1:-}"
MODEL="${MATRIX_MODEL:-/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7}"
MODEL_KEY="deepseek_v2_lite_bf16"
PROTOCOL_ID="eurosys-pa-v2-20260824"
VLLM_PY="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
KT_PY="${REPO_ROOT}/KTransformers/.venv/bin/python"
BLOCK_RUNNER="${MATRIX_SCRIPT_DIR}/formal_openai_block.py"
CLIENT="${MATRIX_SCRIPT_DIR}/streaming_client.py"
WORKLOAD_CONFIG="${MATRIX_WORKLOAD_CONFIG:-${MATRIX_SCRIPT_DIR}/configs/e2_matrix_cells.json}"
MATRIX_CONFIG="${MATRIX_PROFILE_CONFIG:-${MATRIX_SCRIPT_DIR}/configs/c1_mechanism_runtime_hot_audit_off_matrix.json}"
NATIVE_ENV="${MATRIX_SCRIPT_DIR}/configs/c1_native_common_env.json"
OURS_ENV="${MATRIX_SCRIPT_DIR}/configs/c1_ours_runtime_hot_audit_off_env.json"
KT_ENV="${MATRIX_SCRIPT_DIR}/configs/e2_ktransformers_env.json"
MOEI_ENV="${MATRIX_SCRIPT_DIR}/configs/e2_moe_infinity_env.json"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
TRACE="${MATRIX_SCRIPT_DIR}/inputs/generated/${PROTOCOL_ID}/${MODEL_KEY}/calibration_assets/routing/routing_trace.jsonl"
PLAN_BUILDER="${REPO_ROOT}/experiments/hybrid_offload_2_4_6_8_compare/prepare_cold_plans.py"
KT_PLAN_BUILDER="${REPO_ROOT}/experiments/hybrid_offload_2_4_6_8_compare/prepare_kt_frequency_plans.py"
MOEI_RECEIPT="${MATRIX_MOEI_RECEIPT:-${MATRIX_SCRIPT_DIR}/artifacts/c1-mechanism-min-offload-pilot-20260827-attempt-01/provenance/moe_infinity_store_build/store_build_receipt.json}"
MOEI_OFFLOAD="/home/ubuntu/moe_infinity_offload/deepseek_v2_lite_chat_85864749cd611b4353ce1decdb286193298f64c7"
DUAL_VIEW_VERIFY="${MATRIX_SCRIPT_DIR}/verify_ours_full_dual_view.py"
SYSTEM_FILTER="${MATRIX_SYSTEMS:-native_all_gpu,native_pinned_uva,native_copy_on_forward,ours,ktransformers,moe_infinity}"
WORKLOAD_FILTER="${MATRIX_WORKLOADS:-short_dialogue,ordinary_chat,rag_summary,long_generation}"
SINGLE_ATTEMPT="${MATRIX_SINGLE_ATTEMPT:-0}"
GPU_LOCK="${MATRIX_GPU_LOCK:-/tmp/eurosys_c1_mechanism_runtime_hot_audit_off.lock}"
EXECUTION_TSV=""

die() { echo "error: $*" >&2; exit 1; }

contains_csv() {
  local needle="$1" haystack=",$2,"
  [[ "${haystack}" == *",${needle},"* ]]
}

record_execution() {
  local arm="$1" workload="$2" setting="$3" case_root="$4" rc="$5"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${arm}" "${workload}" \
    "${setting}" "${case_root}" "${rc}" >>"${EXECUTION_TSV}"
}

profile_value() {
  local workload="$1" key="$2"
  jq -er --arg workload "${workload}" --arg key "${key}" \
    '.workloads[] | select(.name == $workload) | .[$key]' "${MATRIX_CONFIG}"
}

profile_value_default() {
  local workload="$1" key="$2" default_json="$3"
  jq -cr --arg workload "${workload}" --arg key "${key}" \
    --argjson default "${default_json}" \
    '.workloads[] | select(.name == $workload) | (.[$key] // $default)' \
    "${MATRIX_CONFIG}"
}

run_block() {
  local arm="$1" workload="$2" setting="$3" case_root="$4"
  local client_system="$5" client_python="$6" env_file="$7" port="$8"
  shift 8
  local request_timeout="${MATRIX_REQUEST_TIMEOUT:-7200}"
  local -a runner overrides server
  overrides=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do
    overrides+=(--server-env-override "$1")
    shift
  done
  [[ $# -gt 0 && "$1" == "--" ]] || die "run_block missing -- separator"
  shift
  server=("$@")
  runner=(
    "${VLLM_PY}" "${BLOCK_RUNNER}"
    --case-root "${case_root}"
    --protocol-id "${PROTOCOL_ID}"
    --system "${client_system}"
    --model "${MODEL}"
    --base-url "http://127.0.0.1:${port}"
    --workload-config "${WORKLOAD_CONFIG}"
    --workload-filter "$(profile_value "${workload}" cell)"
    --server-env "${env_file}"
    "${overrides[@]}"
    --python "${client_python}"
    --streaming-client "${CLIENT}"
    --ready-timeout 1200
    --request-timeout "${request_timeout}"
    --gpu-clean-timeout 240
    --stop-on-first-workload-failure
    -- "${server[@]}"
  )
  set +e
  "${runner[@]}" >"${case_root}.runner.log" 2>&1
  local rc=$?
  set -e
  record_execution "${arm}" "${workload}" "${setting}" "${case_root}" "${rc}"
  return "${rc}"
}

prepare_plans() {
  local plan_root="${RUN_ROOT}/provenance/cold_plans"
  local widths="" cold tier input output
  if [[ ! -f "${plan_root}/manifest.json" ]]; then
    mkdir -p "${plan_root}"
    for cold in $(seq -w 1 16); do
      [[ -z "${widths}" ]] || widths+=","
      widths+="cold${cold}=$((10#${cold}))"
    done
    "${VLLM_PY}" "${PLAN_BUILDER}" --trace "${TRACE}" \
      --output-dir "${plan_root}" --serving-start-step 3 \
      --prefill-step-count 1 --tier-widths "${widths}" \
      >"${RUN_ROOT}/provenance/prepare_cold_plans.stdout.json"
  fi
  for cold in $(seq -w 1 16); do
    output="${RUN_ROOT}/provenance/kt_plans/cold${cold}"
    [[ -f "${output}/kt_frequency_2g.pt" ]] && continue
    input="${RUN_ROOT}/provenance/kt_inputs/cold${cold}"
    mkdir -p "${input}" "${output}"
    for tier in 2g 4g 6g 8g; do
      cp "${plan_root}/cold_plan_cold${cold}.json" \
        "${input}/cold_plan_${tier}.json"
    done
    "${VLLM_PY}" "${KT_PLAN_BUILDER}" --input-dir "${input}" \
      --output-dir "${output}" \
      >"${output}/prepare.stdout.txt"
  done
}

run_native_arm() {
  local arm="$1" workload="$2" port cpu_gb backend case_root
  local max_len kv_bytes max_batched max_num_seqs attempt_tag
  local graph_mode compilation exact_names_file exact_name_count
  local -a graph_args
  max_len="$(profile_value "${workload}" max_model_len)"
  kv_bytes="$(profile_value "${workload}" kv_cache_bytes)"
  max_batched="$(profile_value "${workload}" max_batched_tokens)"
  max_num_seqs="$(profile_value_default "${workload}" concurrency 1)"
  case "${arm}" in
    native_all_gpu) port=30401; cpu_gb=0; backend="" ;;
    native_pinned_uva)
      port=30402
      cpu_gb="$(profile_value_default "${workload}" native_offload_gb 0.01)"
      backend=uva
      ;;
    native_copy_on_forward)
      port=30403
      cpu_gb="$(profile_value_default "${workload}" native_offload_gb 0.01)"
      backend=copy
      ;;
    *) die "unknown native arm ${arm}" ;;
  esac
  graph_mode="NONE"
  if [[ "${arm}" == native_pinned_uva \
        || "${arm}" == native_copy_on_forward ]]; then
    graph_mode="${MATRIX_NATIVE_CUDAGRAPH_MODE:-${MATRIX_NATIVE_UVA_CUDAGRAPH_MODE:-NONE}}"
  fi
  case "${graph_mode}" in
    NONE|FULL_DECODE_ONLY|FULL_SEPARATE) ;;
    *) die "unsupported native CUDA Graph mode: ${graph_mode}" ;;
  esac

  exact_names_file=""
  exact_name_count="$(profile_value_default "${workload}" native_copy_exact_parameter_names '[]' | jq 'length')"
  if [[ "${arm}" == native_copy_on_forward && "${exact_name_count}" -gt 0 ]]; then
    cpu_gb="$(profile_value "${workload}" native_copy_exact_offload_gb)"
    exact_names_file="${RUN_ROOT}/provenance/native_copy_exact_${workload}.json"
    profile_value "${workload}" native_copy_exact_parameter_names >"${exact_names_file}"
  fi

  # Preserve a previously successful cell when resuming an interrupted matrix.
  if find "${RUN_ROOT}/cases/${arm}/${workload}" -mindepth 2 -maxdepth 2 \
      -name verification.json -exec jq -e '.status == "ok"' {} \; \
      2>/dev/null | grep -q true; then
    return 0
  fi

  # The 8K MLA Prefill requests a transient 512 MiB allocation.  First retry
  # the identical one-step Prefill with expandable allocator segments.  Only
  # if that still fails, keep the request unchanged and reduce the Prefill
  # chunk size to 4096 and then 2048 tokens.
  local -a candidate_tags candidate_chunks
  if [[ "${workload}" == rag_summary && "${SINGLE_ATTEMPT}" == 1 \
        && "${max_batched}" -lt 2048 ]]; then
    # The DeepSeek Triton MLA Native-UVA path is only qualified for
    # 128-token pure-Prefill Graphs.  Respect an explicitly smaller frozen
    # profile instead of silently widening it back to the historical 2K
    # eager/Decode-Graph fallback.
    candidate_tags=("attempt_01_frozen_${max_batched}")
    candidate_chunks=("${max_batched}")
  elif [[ "${workload}" == rag_summary && "${SINGLE_ATTEMPT}" == 1 ]]; then
    candidate_tags=(attempt_01_frozen_2048)
    candidate_chunks=(2048)
  elif [[ "${workload}" == rag_summary ]]; then
    candidate_tags=(attempt_02_expandable_8192 attempt_03_expandable_4096 attempt_04_expandable_2048)
    candidate_chunks=(8192 4096 2048)
  else
    candidate_tags=(attempt_01)
    candidate_chunks=("${max_batched}")
  fi

  local idx chunk
  for idx in "${!candidate_tags[@]}"; do
    attempt_tag="${candidate_tags[$idx]}"
    chunk="${candidate_chunks[$idx]}"
    case_root="${RUN_ROOT}/cases/${arm}/${workload}/${attempt_tag}"
    if [[ -f "${case_root}/verification.json" ]] \
      && jq -e '.status == "ok"' "${case_root}/verification.json" >/dev/null; then
      return 0
    fi
    [[ ! -e "${case_root}" && ! -e "${case_root}.runner.log" ]] || continue
    mkdir -p "$(dirname "${case_root}")"
    local -a overrides=("PYTORCH_ALLOC_CONF=expandable_segments:True")
    graph_args=(--enforce-eager)
    compilation='{"mode":"NONE","cudagraph_mode":"NONE"}'
    if [[ "${graph_mode}" == FULL_DECODE_ONLY ]]; then
      compilation='{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1],"max_cudagraph_capture_size":1}'
      graph_args=(--compilation-config "${compilation}")
      overrides+=(
        "VLLM_TRITON_MLA_EXPERIMENTAL_FULL_CUDAGRAPH=1"
        "VLLM_CUDAGRAPH_DISPATCH_AUDIT=1"
      )
    elif [[ "${graph_mode}" == FULL_SEPARATE ]]; then
      compilation="$(jq -cn --argjson prefill "${chunk}" --argjson decode 1 \
        --argjson copy_inputs "${MATRIX_NATIVE_CUDAGRAPH_COPY_INPUTS:-false}" \
        '{mode:"NONE",cudagraph_mode:"FULL_SEPARATE",
          cudagraph_capture_sizes:([$decode,$prefill]|unique|sort),
          max_cudagraph_capture_size:([$decode,$prefill]|max),
          cudagraph_copy_inputs:$copy_inputs}')"
      graph_args=(--compilation-config "${compilation}")
      overrides+=(
        "VLLM_TRITON_MLA_EXPERIMENTAL_FULL_CUDAGRAPH=1"
        "VLLM_CUDAGRAPH_DISPATCH_AUDIT=1"
      )
    fi
    if [[ -n "${backend}" ]]; then
      overrides+=("VLLM_CPU_OFFLOAD_BACKEND=${backend}")
      overrides+=("VLLM_CPU_OFFLOAD_MANIFEST_FILE=${case_root}/offload_manifest.jsonl")
      overrides+=("VLLM_CPU_OFFLOAD_MANIFEST_RESET=1")
    fi
    if [[ "${backend}" == copy ]]; then
      overrides+=("VLLM_CPU_OFFLOAD_COPY_SYNC=0" "VLLM_CPU_OFFLOAD_COPY_TRACE=0")
      if [[ -n "${exact_names_file}" ]]; then
        overrides+=("VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE=${exact_names_file}")
      fi
      if [[ "${MATRIX_NATIVE_COPY_STATIC_BANK:-0}" == 1 ]]; then
        overrides+=(
          "VLLM_CPU_OFFLOAD_STATIC_BANK=1"
          "VLLM_DISABLE_SHARED_EXPERTS_STREAM=1"
        )
      fi
    fi
    if run_block "${arm}" "${workload}" \
      "cpu_offload_gb=${cpu_gb},max_batched_tokens=${chunk},expandable_segments=true,cudagraph_mode=${graph_mode}" \
      "${case_root}" ours "${VLLM_PY}" "${NATIVE_ENV}" "${port}" \
      "${overrides[@]}" -- \
      taskset -c 0-13 "${VLLM_PY}" -m vllm.entrypoints.openai.api_server \
      --host 127.0.0.1 --port "${port}" --disable-frontend-multiprocessing \
      --model "${MODEL}" --trust-remote-code --dtype bfloat16 \
      --max-model-len "${max_len}" --gpu-memory-utilization 0.97 \
      --kv-cache-memory-bytes "${kv_bytes}" --cpu-offload-gb "${cpu_gb}" \
      --max-num-batched-tokens "${chunk}" --max-num-seqs "${max_num_seqs}" \
      --no-enable-prefix-caching --generation-config vllm \
      "${graph_args[@]}"; then
      return 0
    fi
  done
  return 1
}

run_ours_workload() {
  local workload="$1" cold cold_tag case_root plan names compilation
  local max_len kv_bytes max_batched prefill reserve chunk execution_mode case_suffix
  local max_num_seqs decode_graph_sizes
  local -a execution_overrides execution_args
  max_len="$(profile_value "${workload}" max_model_len)"
  kv_bytes="$(profile_value "${workload}" kv_cache_bytes)"
  max_batched="$(profile_value "${workload}" ours_max_batched_tokens)"
  prefill="$(profile_value "${workload}" prefill_graph_tokens)"
  reserve="$(profile_value "${workload}" prefill_reserve_tokens)"
  chunk="$(profile_value "${workload}" fused_moe_chunk_tokens)"
  max_num_seqs="$(profile_value_default "${workload}" concurrency 1)"
  decode_graph_sizes="$(profile_value_default \
    "${workload}" decode_graph_batch_sizes '[1]')"
  names="${RUN_ROOT}/provenance/cold_plans/exact_parameter_names.txt"
  execution_mode="${MATRIX_OURS_CUDAGRAPH_MODE:-FULL_SEPARATE}"
  execution_overrides=()
  execution_args=()
  case_suffix=""
  case "${execution_mode}" in
    FULL_SEPARATE)
      execution_overrides+=("VLLM_CUDAGRAPH_DISPATCH_AUDIT=1")
      compilation="$(jq -cn --argjson prefill "${prefill}" \
        --argjson decode "${decode_graph_sizes}" \
        '($decode + [$prefill] | unique | sort) as $sizes |
         {mode:"NONE",cudagraph_mode:"FULL_SEPARATE",
          cudagraph_capture_sizes:$sizes,
          max_cudagraph_capture_size:($sizes | max)}')"
      ;;
    NONE)
      compilation='{"mode":"NONE","cudagraph_mode":"NONE"}'
      case_suffix="_eager_warmup_cache"
      execution_overrides+=(
        "VLLM_UVM_ORIGINAL_VA_PHASE_MODE=prefill_remote_decode"
        "VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=0"
        # The explicit allocator reserve is a graph-only, post-capture hook.
        # Eager mode instead relies on the three real 2K warmup requests to
        # establish allocator/workspace cache before the measured request.
        "VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS=0"
        "VLLM_TRITON_MLA_EXPERIMENTAL_FULL_CUDAGRAPH=0"
      )
      execution_args+=(--enforce-eager)
      ;;
    FULL_DECODE_ONLY)
      compilation="$(jq -cn --argjson decode "${decode_graph_sizes}" \
        '{mode:"NONE",cudagraph_mode:"FULL_DECODE_ONLY",
         cudagraph_capture_sizes:($decode | unique | sort),
         max_cudagraph_capture_size:($decode | max)}')"
      case_suffix="_decode_graph_prefill_eager_warmup_cache"
      execution_overrides+=(
        "VLLM_UVM_ORIGINAL_VA_PHASE_MODE=prefill_remote_decode"
        "VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=0"
        # Only Decode is captured.  Three real Prefill warmups establish the
        # eager MLA/MoE workspace cache before the measured request.
        "VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS=0"
        "VLLM_TRITON_MLA_EXPERIMENTAL_FULL_CUDAGRAPH=1"
      )
      ;;
    *) die "unsupported MATRIX_OURS_CUDAGRAPH_MODE=${execution_mode}" ;;
  esac
  local -a cold_candidates
  if [[ "${SINGLE_ATTEMPT}" == 1 ]]; then
    cold_candidates=("$(profile_value_default "${workload}" ours_cold_per_layer 1)")
  else
    mapfile -t cold_candidates < <(seq 1 16)
  fi
  for cold in "${cold_candidates[@]}"; do
    cold_tag="$(printf '%02d' "${cold}")"
    plan="${RUN_ROOT}/provenance/cold_plans/cold_plan_cold${cold_tag}.json"
    if [[ "${max_batched}" != "$(profile_value "${workload}" max_batched_tokens)" ]]; then
      case_root="${RUN_ROOT}/cases/ours/${workload}/cold_${cold_tag}_chunk_${max_batched}${case_suffix}"
    else
      case_root="${RUN_ROOT}/cases/ours/${workload}/cold_${cold_tag}${case_suffix}"
    fi
    if [[ -f "${case_root}/verification.json" ]] \
      && jq -e '.status == "ok"' "${case_root}/verification.json" >/dev/null; then
      return 0
    fi
    [[ ! -e "${case_root}" && ! -e "${case_root}.runner.log" ]] || continue
    mkdir -p "$(dirname "${case_root}")"
    if run_block ours "${workload}" \
      "cold_per_layer=${cold},prefill_chunk_tokens=${max_batched},cudagraph_mode=${execution_mode}" \
      "${case_root}" ours "${VLLM_PY}" "${OURS_ENV}" 30404 \
      "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE=${names}" \
      "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE=${plan}" \
      "VLLM_CPU_OFFLOAD_MANIFEST_FILE=${case_root}/offload_manifest.jsonl" \
      "VLLM_UVM_LOG_FILE=${case_root}/uvm.log" \
      "VLLM_UVM_ORIGINAL_VA_PHASE_RESULT_FILE=${case_root}/phase_transition.json" \
      "VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_BYTES=${kv_bytes}" \
      "VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS=${reserve}" \
      "VLLM_UVM_FUSED_MOE_POST_WORKSPACE_AUDIT_MIN_TOKENS=${prefill}" \
      "VLLM_FUSED_MOE_CHUNK_SIZE=${chunk}" \
      "${execution_overrides[@]}" -- \
      taskset -c 0-13 "${VLLM_PY}" -m vllm.entrypoints.openai.api_server \
      --host 127.0.0.1 --port 30404 --disable-frontend-multiprocessing \
      --model "${MODEL}" --trust-remote-code --dtype bfloat16 \
      "${execution_args[@]}" \
      --max-model-len "${max_len}" --gpu-memory-utilization 0.97 \
      --kv-cache-memory-bytes "${kv_bytes}" --cpu-offload-gb 30 \
      --max-num-batched-tokens "${max_batched}" --max-num-seqs "${max_num_seqs}" \
      --no-enable-prefix-caching --generation-config vllm \
      --compilation-config "${compilation}"; then
      if "${VLLM_PY}" "${DUAL_VIEW_VERIFY}" "${case_root}"; then
        return 0
      fi
    fi
  done
  return 1
}

run_kt_workload() {
  local workload="$1" cold cold_tag case_root frequency gpu_experts
  local max_len kv_tokens max_batched context max_running
  local configured_kt_tokens
  local -a cuda_graph_bs
  max_len="$(profile_value "${workload}" max_model_len)"
  kv_tokens="$(profile_value "${workload}" kv_tokens)"
  max_batched="$(profile_value "${workload}" max_batched_tokens)"
  max_running="$(profile_value_default "${workload}" concurrency 1)"
  configured_kt_tokens="$(profile_value_default "${workload}" kt_kv_tokens null)"
  mapfile -t cuda_graph_bs < <(profile_value_default \
    "${workload}" decode_graph_batch_sizes '[1]' | jq -r '.[]')
  context=$((max_len + 2))
  # SGLang reserves two internal token slots.  A pool equal to prompt+output
  # truncates the exact-limit 8192+128 RAG cell to 126 generated tokens.
  # Use the already configured context ceiling for this cell only.
  if [[ "${configured_kt_tokens}" != null ]]; then
    kv_tokens="${configured_kt_tokens}"
  elif [[ "${workload}" == rag_summary ]]; then
    kv_tokens="${context}"
  fi
  local -a cold_candidates
  if [[ "${SINGLE_ATTEMPT}" == 1 ]]; then
    cold_candidates=(1)
  else
    mapfile -t cold_candidates < <(seq 1 16)
  fi
  for cold in "${cold_candidates[@]}"; do
    cold_tag="$(printf '%02d' "${cold}")"
    frequency="${RUN_ROOT}/provenance/kt_plans/cold${cold_tag}/kt_frequency_2g.pt"
    gpu_experts=$((64 - cold))
    if [[ "${workload}" == rag_summary ]]; then
      case_root="${RUN_ROOT}/cases/ktransformers/${workload}/cold_${cold_tag}_kv_plus_2"
    else
      case_root="${RUN_ROOT}/cases/ktransformers/${workload}/cold_${cold_tag}"
    fi
    if [[ -f "${case_root}/verification.json" ]] \
      && jq -e '.status == "ok"' "${case_root}/verification.json" >/dev/null; then
      return 0
    fi
    [[ ! -e "${case_root}" && ! -e "${case_root}.runner.log" ]] || continue
    mkdir -p "$(dirname "${case_root}")"
    if run_block ktransformers "${workload}" \
      "cold_per_layer=${cold},max_total_tokens=${kv_tokens},cpu_threads=${MATRIX_KT_CPU_THREADS:-14}" \
      "${case_root}" ktransformers "${KT_PY}" "${KT_ENV}" 30405 \
      "OMP_NUM_THREADS=${MATRIX_KT_CPU_THREADS:-14}" \
      "MKL_NUM_THREADS=${MATRIX_KT_CPU_THREADS:-14}" -- \
      taskset -c 0-13 "${KT_PY}" -m sglang.launch_server \
      --model-path "${MODEL}" --host 127.0.0.1 --port 30405 --tp-size 1 \
      --context-length "${context}" --max-running-requests "${max_running}" \
      --mem-fraction-static 0.97 --max-total-tokens "${kv_tokens}" \
      --chunked-prefill-size "${max_batched}" --max-prefill-tokens "${max_batched}" \
      --schedule-policy fcfs --stream-interval 1 --random-seed 20260823 \
      --trust-remote-code --kt-weight-path "${MODEL}" --kt-method BF16 \
      --kt-cpuinfer "${MATRIX_KT_CPU_THREADS:-14}" \
      --kt-threadpool-count 1 --kt-numa-nodes 0 \
      --kt-num-gpu-experts "${gpu_experts}" \
      --kt-expert-placement-strategy frequency --init-expert-location "${frequency}" \
      --kt-max-deferred-experts-per-token 0 --kt-gpu-prefill-token-threshold 0 \
      --attention-backend flashinfer --cuda-graph-max-bs "${max_running}" \
      --cuda-graph-bs "${cuda_graph_bs[@]}" \
      --enable-mixed-chunk --disable-shared-experts-fusion \
      --disable-radix-cache --enable-metrics; then
      return 0
    fi
  done
  return 1
}

validate_moei_receipt() {
  local extension expected actual
  jq -e '.status == "complete" and .returncode == 0' "${MOEI_RECEIPT}" >/dev/null
  extension="$(find "${REPO_ROOT}/MoE-Infinity/moe_infinity" -maxdepth 1 \
    -type f -name '_store*.so' -print -quit)"
  [[ -n "${extension}" ]] || die "MoE-Infinity _store extension is absent"
  expected="$(jq -r '.extension_sha256' "${MOEI_RECEIPT}")"
  actual="$(sha256sum "${extension}" | awk '{print $1}')"
  [[ "${expected}" == "${actual}" ]] || die "MoE-Infinity receipt drift"
}

run_moei_workload() {
  local workload="$1" ratio ratio_int case_root kv_ratio kv_tag
  local kv_blocks kv_block_bytes kv_available_bytes max_tokens_per_step max_batch_size
  kv_blocks="$(profile_value "${workload}" moe_infinity_kv_blocks)"
  max_tokens_per_step="$(profile_value "${workload}" max_batched_tokens)"
  max_batch_size="$(profile_value_default "${workload}" concurrency 1)"
  kv_block_bytes="$(jq -er '.moe_infinity_kv_calibration.block_bytes' "${MATRIX_CONFIG}")"
  kv_available_bytes="$(jq -er '.moe_infinity_kv_calibration.available_bytes' "${MATRIX_CONFIG}")"
  # Preserve a previously successful cell when resuming an interrupted matrix.
  if [[ "${MATRIX_MOEI_FORCE_SEARCH:-0}" != 1 ]] \
    && find "${RUN_ROOT}/cases/moe_infinity/${workload}" -mindepth 2 -maxdepth 2 \
      -name verification.json -exec jq -e '.status == "ok"' {} \; \
      2>/dev/null | grep -q true; then
    return 0
  fi
  # Use the already established coarse boundary for each workload, then walk
  # downward at 0.01 resolution.  The first success is therefore the maximum
  # feasible device-cache ratio (least offload) inside the frozen bracket.
  local -a candidates
  if [[ "${SINGLE_ATTEMPT}" == 1 ]]; then
    case "${workload}" in
      short_dialogue) candidates=(83) ;;
      ordinary_chat|long_generation) candidates=(85) ;;
      rag_summary) candidates=(35) ;;
      *) die "missing frozen MoE-Infinity setting for ${workload}" ;;
    esac
  elif [[ -n "${MATRIX_MOEI_CANDIDATES:-}" ]]; then
    read -r -a candidates <<<"${MATRIX_MOEI_CANDIDATES//,/ }"
  else
    case "${workload}" in
      short_dialogue) candidates=(85 84 83) ;;
      ordinary_chat|long_generation) candidates=(90 89 88 87 86 85) ;;
      rag_summary) candidates=(40 39 38 37 36 35) ;;
      *) die "missing MoE-Infinity search bracket for ${workload}" ;;
    esac
  fi
  for ratio_int in "${candidates[@]}"; do
    ratio="$(printf '0.%02d' "${ratio_int}")"
    # Keep the absolute KV block count fixed while device_memory_ratio changes.
    # Round upward at 1e-6 so floor(kv_bytes / block_bytes) cannot under-size it.
    kv_ratio="$(jq -nr \
      --argjson blocks "${kv_blocks}" \
      --argjson block_bytes "${kv_block_bytes}" \
      --argjson available_bytes "${kv_available_bytes}" \
      --argjson device_ratio "${ratio}" \
      '((($blocks * $block_bytes) / ($available_bytes * $device_ratio) * 1000000) | ceil) / 1000000')"
    kv_tag="${kv_ratio/./}"
    case_root="${RUN_ROOT}/cases/moe_infinity/${workload}/ratio_${ratio_int}_kv_${kv_tag}_step_${max_tokens_per_step}"
    if [[ -f "${case_root}/verification.json" ]] \
      && jq -e '.status == "ok"' "${case_root}/verification.json" >/dev/null; then
      return 0
    fi
    [[ ! -e "${case_root}" && ! -e "${case_root}.runner.log" ]] || continue
    mkdir -p "$(dirname "${case_root}")"
    if run_block moe_infinity "${workload}" \
      "device_ratio=${ratio},kv_ratio=${kv_ratio},kv_blocks=${kv_blocks},max_tokens_per_step=${max_tokens_per_step}" \
      "${case_root}" moe_infinity "${VLLM_PY}" "${MOEI_ENV}" 30406 -- \
      taskset -c 0-13 "${VLLM_PY}" "${MATRIX_SCRIPT_DIR}/launch_moe_infinity_server.py" \
      --model "${MODEL}" --offload-path "${MOEI_OFFLOAD}" \
      --host 127.0.0.1 --port 30406 --device-memory-ratio "${ratio}" \
      --max-tokens-per-step "${max_tokens_per_step}" \
      --kv-cache-ratio "${kv_ratio}" --max-batch-size "${max_batch_size}" \
      --num-threads 1; then
      return 0
    fi
  done
  return 1
}

[[ -n "${RUN_ROOT}" ]] || die "usage: $0 RUN_ROOT"
[[ "${SINGLE_ATTEMPT}" == 0 || "${SINGLE_ATTEMPT}" == 1 ]] \
  || die "MATRIX_SINGLE_ATTEMPT must be 0 or 1"
if [[ -e "${RUN_ROOT}" && "${MATRIX_RESUME:-0}" != 1 ]]; then
  die "refusing to overwrite ${RUN_ROOT}; set MATRIX_RESUME=1 to resume"
fi
for path in "${MODEL}/config.json" "${VLLM_PY}" "${KT_PY}" \
  "${BLOCK_RUNNER}" "${CLIENT}" "${WORKLOAD_CONFIG}" "${MATRIX_CONFIG}" \
  "${NATIVE_ENV}" "${OURS_ENV}" "${KT_ENV}" "${MOEI_ENV}" "${TRACE}" \
  "${PLAN_BUILDER}" "${KT_PLAN_BUILDER}" "${BASE_HARNESS}" \
  "${DUAL_VIEW_VERIFY}"; do
  [[ -e "${path}" ]] || die "missing required path: ${path}"
done
command -v flock >/dev/null || die "flock unavailable"
exec 7>"${GPU_LOCK}"
flock -n 7 || die "another C1 mechanism matrix holds ${GPU_LOCK}"

mkdir -p "${RUN_ROOT}/provenance" "${RUN_ROOT}/cases"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
EXECUTION_TSV="${RUN_ROOT}/execution.tsv"
if [[ ! -f "${EXECUTION_TSV}" ]]; then
  printf 'created_at\tarm\tworkload\tsetting\tcase_root\treturncode\n' >"${EXECUTION_TSV}"
fi
if [[ ! -f "${RUN_ROOT}/provenance/matrix_config.json" ]]; then
  cp "${MATRIX_CONFIG}" "${RUN_ROOT}/provenance/matrix_config.json"
fi
matrix_config_sha="$(sha256sum "${MATRIX_CONFIG}" | awk '{print $1}')"
matrix_config_snapshot="${RUN_ROOT}/provenance/matrix_config.${matrix_config_sha}.json"
if [[ ! -f "${matrix_config_snapshot}" ]]; then
  cp "${MATRIX_CONFIG}" "${matrix_config_snapshot}"
fi
git -C "${REPO_ROOT}" status --short \
  >"${RUN_ROOT}/provenance/git_status.resume.$(date -u +%Y%m%dT%H%M%SZ).txt"
if contains_csv ours "${SYSTEM_FILTER}" \
    || contains_csv ktransformers "${SYSTEM_FILTER}"; then
  prepare_plans
fi

total_failures=0
for arm in native_all_gpu native_pinned_uva native_copy_on_forward; do
  contains_csv "${arm}" "${SYSTEM_FILTER}" || continue
  for workload in short_dialogue ordinary_chat rag_summary long_generation; do
    contains_csv "${workload}" "${WORKLOAD_FILTER}" || continue
    run_native_arm "${arm}" "${workload}" || total_failures=$((total_failures + 1))
  done
done

if contains_csv ours "${SYSTEM_FILTER}"; then
  export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
  export ALLOW_GPU_RUN=1
  export SLOT_BUDGET_BYTES=1207959552
  export MANUAL_MIGRATION_FAULT_MODE=2
  export STATIC_ALIAS_DUAL_VIEW_ENABLE=1
  export STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL=1
  export ACCESS_COUNTER_MIGRATION_ENABLE=0
  export UVM_PREFETCH_ENABLE=0
  export UVM_THRASHING_ENABLE=0
  # shellcheck source=/dev/null
  source "${BASE_HARNESS}"
  RUN_ROOT_ACTIVE="${RUN_ROOT}"
  runtime_preflight
  module_snapshot "${RUN_ROOT}/module.before.json" "${STOCK_UVM}"
  load_custom_module
  module_snapshot "${RUN_ROOT}/module.custom.json" "${CUSTOM_UVM}"
  for workload in short_dialogue ordinary_chat rag_summary long_generation; do
    contains_csv "${workload}" "${WORKLOAD_FILTER}" || continue
    run_ours_workload "${workload}" || total_failures=$((total_failures + 1))
  done
  restore_stock
  module_snapshot "${RUN_ROOT}/module.after_ours.json" "${STOCK_UVM}"
  restart_exporters
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
echo "RUN_ROOT=${RUN_ROOT}"
echo "FAILURES=${total_failures}"
(( total_failures == 0 ))
