#!/usr/bin/env bash
set -euo pipefail

COMPARE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${COMPARE_DIR}/../.." && pwd)"
EVAL_DIR="${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation"
COMPARE_RUNNER="${COMPARE_RUNNER_OVERRIDE:-${EVAL_DIR}/e11_continuous_arrival_runner.py}"
COMPARE_VERIFY="${COMPARE_VERIFY_OVERRIDE:-${COMPARE_DIR}/verify_compare.py}"
FROZEN_ENV="${EVAL_DIR}/configs/e2_ours_2g_env.json"
ADAPT_NAMES="${EVAL_DIR}/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/calibration_assets/plans/exact_parameter_names.txt"
ADAPT_PLAN="${EVAL_DIR}/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/calibration_assets/plans_adjusted_11_17_23_29/cold_plan_2g.json"
NATIVE_NAMES="${NATIVE_NAMES:?Use run_kv1_compare.sh to prepare native parameter names}"
MODEL="${MODEL:?Set MODEL to the local DeepSeek-V2-Lite-Chat snapshot}"
FIXTURE="${FIXTURE:-${COMPARE_DIR}/inputs/sharegpt_deepseek_p1024_n128_20261024.json}"
SHAREGPT_DATASET="${SHAREGPT_DATASET:-${REPO_ROOT}/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
SAMPLES="${SAMPLES:-2}"
COMPARE_ARMS="${COMPARE_ARMS:-adaptive native_uva native_copy}"
read -r -a selected_arms <<<"${COMPARE_ARMS}"
ARRIVAL_QPS="${ARRIVAL_QPS:-}"
arrival_args=()
if [[ -n "${ARRIVAL_QPS}" ]]; then
  arrival_args=(--arrival-qps "${ARRIVAL_QPS}")
fi
NATIVE_OFFLOAD_GB="${NATIVE_OFFLOAD_GB:-5.0}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-167772160}"
INITIAL_REQUESTS="${INITIAL_REQUESTS:-4}"
LATE_REQUESTS="${LATE_REQUESTS:-4}"
LATE_AFTER_ENGINE_STEPS="${LATE_AFTER_ENGINE_STEPS:-4}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
PARK_TOKEN_THRESHOLD="${PARK_TOKEN_THRESHOLD:-64}"
RESUME_TOKEN_THRESHOLD="${RESUME_TOKEN_THRESHOLD:-128}"
TIMEOUT_SEC="${TIMEOUT_SEC:-2400}"
CPUSET="${CPUSET:-0-13}"
NSYS_PROFILE="${NSYS_PROFILE:-0}"
NSYS_BIN="${NSYS_BIN:-/usr/local/cuda-12.8/bin/nsys}"

export STATIC_ALIAS_DUAL_VIEW_ENABLE=1
export STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL=1
export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
# shellcheck source=/dev/null
source "${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY

[[ "${SAMPLES}" =~ ^[1-9][0-9]*$ ]] || die "SAMPLES must be positive"
(( ${#selected_arms[@]} > 0 )) || die "COMPARE_ARMS must be non-empty"
declare -A seen_arms=()
for arm in "${selected_arms[@]}"; do
  case "${arm}" in
    adaptive|adaptive_fixed|adaptive_dynamic|adaptive_fixed_1g|adaptive_fixed_2g|adaptive_fixed_4g|native_uva|native_copy) ;;
    *) die "unknown arm in COMPARE_ARMS: ${arm}" ;;
  esac
  [[ -z "${seen_arms[${arm}]:-}" ]] || die "duplicate arm: ${arm}"
  seen_arms["${arm}"]=1
done
[[ "${NSYS_PROFILE}" == 0 || "${NSYS_PROFILE}" == 1 ]] \
  || die "NSYS_PROFILE must be 0 or 1"
if [[ "${NSYS_PROFILE}" == 1 ]]; then
  [[ -x "${NSYS_BIN}" ]] || die "Nsight Systems is missing: ${NSYS_BIN}"
fi
for value in "${INITIAL_REQUESTS}" "${LATE_REQUESTS}" \
  "${LATE_AFTER_ENGINE_STEPS}" "${OUTPUT_TOKENS}" "${MAX_MODEL_LEN}" \
  "${MAX_NUM_BATCHED_TOKENS}" "${MAX_NUM_SEQS}" \
  "${PARK_TOKEN_THRESHOLD}" "${RESUME_TOKEN_THRESHOLD}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "request/scheduler values must be positive"
done
(( PARK_TOKEN_THRESHOLD < RESUME_TOKEN_THRESHOLD )) \
  || die "PARK_TOKEN_THRESHOLD must be below RESUME_TOKEN_THRESHOLD"
for path in "${COMPARE_RUNNER}" "${COMPARE_VERIFY}" "${FROZEN_ENV}" "${ADAPT_NAMES}" \
  "${ADAPT_PLAN}" "${NATIVE_NAMES}" "${MODEL}/config.json" "${FIXTURE}" \
  "${SHAREGPT_DATASET}"; do
  require_file "${path}"
done
"${VLLM_PY}" -m py_compile "${COMPARE_RUNNER}" "${COMPARE_VERIFY}"
bash -n "${BASH_SOURCE[0]}"
runtime_preflight

run_root="${RUN_ROOT:-${COMPARE_DIR}/artifacts/sharegpt-three-arm-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ ! -e "${run_root}" ]] || die "RUN_ROOT already exists: ${run_root}"
mkdir -p "${run_root}/arms" "${run_root}/provenance"
run_root="$(cd "${run_root}" && pwd)"
RUN_ROOT_ACTIVE="${run_root}"

mapfile -t frozen_env < <(
  "${VLLM_PY}" - "${FROZEN_ENV}" "${REPO_ROOT}" <<'PY'
import json, sys
for key, value in json.load(open(sys.argv[1], encoding="utf-8")).items():
    value = value.replace("__REPO_ROOT__", sys.argv[2])
    print(f"{key}={value}")
PY
)

"${VLLM_PY}" - "${run_root}/run_config.json" "${SAMPLES}" \
  "${MODEL}" "${FIXTURE}" "${SHAREGPT_DATASET}" \
  "${NATIVE_OFFLOAD_GB}" "${KV_CACHE_BYTES}" "${INITIAL_REQUESTS}" \
  "${LATE_REQUESTS}" "${LATE_AFTER_ENGINE_STEPS}" "${OUTPUT_TOKENS}" \
  "${MAX_MODEL_LEN}" "${MAX_NUM_BATCHED_TOKENS}" "${MAX_NUM_SEQS}" \
  "${PARK_TOKEN_THRESHOLD}" "${RESUME_TOKEN_THRESHOLD}" "${ARRIVAL_QPS}" "${COMPARE_ARMS}" <<'PY'
import hashlib, json, math, os, sys
(
    path, samples, model, fixture_path, dataset, native_gb, kv,
    initial, late, late_after, output_tokens, max_model_len,
    max_batched, max_seqs, park_threshold, resume_threshold, arrival_qps, arms_text,
) = sys.argv[1:]
arms = arms_text.split()
rate = float(arrival_qps) if arrival_qps else None
if rate is not None and (not math.isfinite(rate) or rate <= 0):
    raise SystemExit("ARRIVAL_QPS must be finite and positive")
fixture = json.load(open(fixture_path, encoding="utf-8"))
metadata = fixture["metadata"]
request_count = int(initial) + int(late)
records = fixture["requests"][:request_count]
if len(records) != request_count:
    raise SystemExit("fixture does not contain enough pressure-scan requests")
prompt_lengths = {int(record["prompt_tokens"]) for record in records}
if len(prompt_lengths) != 1:
    raise SystemExit("pressure-scan fixture prompt lengths are not uniform")
prompt_tokens = prompt_lengths.pop()
payload = {
    "schema": "adaptmoe_token_threshold_sharegpt_compare_run_v2",
    "samples_per_arm": int(samples),
    "arm_order": f"odd samples: {','.join(arms)}; even samples: {','.join(reversed(arms))}",
    "arms": arms,
    "model": os.path.realpath(model),
    "fixture": os.path.realpath(fixture_path),
    "fixture_sha256": hashlib.sha256(open(fixture_path, "rb").read()).hexdigest(),
    "sharegpt_dataset": os.path.realpath(dataset),
    "sharegpt_dataset_sha256": metadata["source_dataset_sha256"],
    "source_domain": metadata["source_domain"],
    "source_indices": sorted({
        int(index) for record in records for index in record["source_indices"]
    }),
    "prompt_token_hashes": [record["prompt_token_ids_sha256"] for record in records],
    "requests": request_count,
    "prompt_tokens_per_request": prompt_tokens,
    "output_tokens_per_request": int(output_tokens),
    "initial_requests": int(initial),
    "late_requests": int(late),
    "late_after_engine_steps": int(late_after),
    "arrival_qps": rate,
    "arrival_policy": "fixed_interval" if rate else "two_cohort_steps",
    "planned_arrival_offsets_seconds": [i / rate for i in range(request_count)] if rate else None,
    "scheduler": "vLLM v1 mixed continuous",
    "cudagraph_mode": "NONE",
    "kv_cache_memory_bytes": int(kv),
    "arm_kv_cache_memory_bytes": {
        arm: int(os.environ.get("ADAPT_FIXED1_KV_CACHE_BYTES", "1073741824")) if arm == "adaptive_fixed_1g" else
        int(os.environ.get("ADAPT_FIXED2_KV_CACHE_BYTES", "2147483648")) if arm == "adaptive_fixed_2g" else
        int(os.environ.get("ADAPT_FIXED4_KV_CACHE_BYTES", "6442450944"))
        if arm == "adaptive_fixed_4g" else
        int(os.environ.get("ADAPT_FIXED_KV_CACHE_BYTES", "536870912"))
        if arm == "adaptive_fixed" else
        int(os.environ.get("ADAPT_DYNAMIC_KV_CACHE_BYTES", "6442450944"))
        if arm == "adaptive_dynamic" else int(kv)
        for arm in arms
    },
    "max_model_len": int(max_model_len),
    "max_num_batched_tokens": int(max_batched),
    "max_num_seqs": int(max_seqs),
    "adaptive_selected_bytes": 4_948_230_144,
    "native_expected_bytes": int(os.environ.get("NATIVE_EXPECTED_BYTES", "4798283776")),
    "native_offload_gb": float(native_gb),
    "park_token_threshold": int(park_threshold),
    "resume_token_threshold": int(resume_threshold),
}
open(path, "w", encoding="utf-8").write(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY

sha256sum "${COMPARE_RUNNER}" "${COMPARE_VERIFY}" "${FROZEN_ENV}" "${ADAPT_NAMES}" \
  "${ADAPT_PLAN}" "${NATIVE_NAMES}" "${FIXTURE}" "${MODEL}/config.json" \
  "${VLLM_ROOT}/vllm/model_executor/models/utils.py" \
  "${VLLM_ROOT}/vllm/v1/worker/gpu_model_runner.py" "${CUSTOM_UVM}" \
  >"${run_root}/input_sha256.txt"
if git -C "${REPO_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "${REPO_ROOT}" status --short >"${run_root}/git_status.txt"
else
  printf 'Git metadata unavailable\n' >"${run_root}/git_status.txt"
fi
printf 'ordinal\tsample\tarm\tdirectory\treturncode\n' \
  >"${run_root}/execution_status.tsv"

module_snapshot "${run_root}/module.before.json" "${STOCK_UVM}"
load_custom_module
module_snapshot "${run_root}/module.custom.json" "${CUSTOM_UVM}"
wait_gpu_clean || die "GPU is not clean before comparison"

run_arm() {
  local sample="$1" arm="$2" ordinal="$3"
  local arm_dir mode cpu_offload_gb runner_rc kernel_epoch exact_names
  local arm_kv_cache_bytes="${KV_CACHE_BYTES}"
  case "${arm}" in
    adaptive_fixed) arm_kv_cache_bytes="${ADAPT_FIXED_KV_CACHE_BYTES:-536870912}" ;;
    adaptive_fixed_1g) arm_kv_cache_bytes="${ADAPT_FIXED1_KV_CACHE_BYTES:-1073741824}" ;;
    adaptive_fixed_2g) arm_kv_cache_bytes="${ADAPT_FIXED2_KV_CACHE_BYTES:-2147483648}" ;;
    adaptive_fixed_4g) arm_kv_cache_bytes="${ADAPT_FIXED4_KV_CACHE_BYTES:-6442450944}" ;;
    adaptive_dynamic) arm_kv_cache_bytes="${ADAPT_DYNAMIC_KV_CACHE_BYTES:-6442450944}" ;;
  esac
  local -a arm_env runner_prefix
  arm_dir="${run_root}/arms/$(printf '%02d_s%02d_%s' \
    "${ordinal}" "${sample}" "${arm}")"
  mkdir "${arm_dir}"
  : >"${arm_dir}/cpu_offload_manifest.jsonl"
  wait_gpu_clean || die "GPU is not clean before ${arm} sample ${sample}"
  gpu_state_json "${arm_dir}/gpu.before.json"

  arm_env=(
    "VLLM_UVM_PHASE_BATCHED_SCHEDULING=0"
    "VLLM_UVM_PHASE_KV_AWARE_COHORTS=0"
    "VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE=0"
    "VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=0"
    "VLLM_TRITON_MLA_EXPERIMENTAL_FULL_CUDAGRAPH=0"
    "VLLM_CPU_OFFLOAD_COPY_SYNC=0"
    "VLLM_CPU_OFFLOAD_COPY_TRACE=1"
    "VLLM_CUDAGRAPH_DISPATCH_AUDIT=1"
    "VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE=0"
    "VLLM_UVM_FUSED_MOE_PRE_KERNEL_ALIAS_READ_GATE=0"
    # The frozen E2 environment enables a large-Prefill residency diagnostic.
    # It scans all hot ranges four times in DeepSeek layer 1 and is not part of
    # the transport mechanism, so keep it out of performance measurements.
    "VLLM_UVM_FUSED_MOE_POST_WORKSPACE_RESIDENCY_AUDIT=0"
    "VLLM_UVM_FUSED_MOE_POST_WORKSPACE_AUDIT_CAPTURE_WARMUP=0"
    "VLLM_CPU_OFFLOAD_MANIFEST_FILE=${arm_dir}/cpu_offload_manifest.jsonl"
    "VLLM_CPU_OFFLOAD_MANIFEST_RESET=1"
    "VLLM_UVM_LOG_FILE=${arm_dir}/uvm.log"
    "PYTORCH_ALLOC_CONF=expandable_segments:True"
  )
  runner_prefix=()
  if [[ "${NSYS_PROFILE}" == 1 ]]; then
    arm_env+=("VLLM_E11_CUDA_PROFILER_CAPTURE=1")
    runner_prefix=(
      "${NSYS_BIN}" profile
      --trace=cuda,nvtx,osrt
      --sample=none
      --cpuctxsw=none
      --capture-range=cudaProfilerApi
      --capture-range-end=stop
      --force-overwrite=true
      "--output=${arm_dir}/timeline"
    )
  fi

  case "${arm}" in
    adaptive|adaptive_fixed|adaptive_dynamic|adaptive_fixed_1g|adaptive_fixed_2g|adaptive_fixed_4g)
      mode=adaptmoe_copy_eager_continuous
      cpu_offload_gb=30
      exact_names="${ADAPT_NAMES}"
      arm_env+=(
        "VLLM_CPU_OFFLOAD_BACKEND=copy"
        "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE=${ADAPT_NAMES}"
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1"
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER=0"
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va"
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE=${ADAPT_PLAN}"
        "VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1"
        "VLLM_UVM_ORIGINAL_VA_PHASE_MODE=token_threshold"
        "VLLM_UVM_ORIGINAL_VA_TOKEN_THRESHOLD=${PARK_TOKEN_THRESHOLD}"
        "VLLM_UVM_ORIGINAL_VA_RESUME_TOKEN_THRESHOLD=${RESUME_TOKEN_THRESHOLD}"
        "VLLM_UVM_TOKEN_THRESHOLD_FAST_TRANSITION=1"
        "VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1"
        "VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW=1"
        "VLLM_CPU_OFFLOAD_PLAN_CLASS_IMMUTABLE_CPU_SHADOW=1"
        "VLLM_CPU_OFFLOAD_DUAL_VIEW_FULL_PARAMETER_SHADOW=1"
        "VLLM_UVM_RUNTIME_POST_MAP_RESIDENCY_AUDIT=0"
        "VLLM_UVM_RUNTIME_HOT_RESIDENCY_AUDIT=0"
      )
      ;;
    native_uva)
      mode=native_uva_eager_continuous
      cpu_offload_gb="${NATIVE_OFFLOAD_GB}"
      exact_names="${NATIVE_NAMES}"
      arm_env+=(
        "VLLM_CPU_OFFLOAD_BACKEND=uva"
        "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE=${NATIVE_NAMES}"
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=0"
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER=0"
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=vmm"
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="
        "VLLM_UVM_EXPERT_WEIGHTS_MANAGED=0"
        "VLLM_UVM_ORIGINAL_VA_PHASE_MODE=off"
        "VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=0"
        "VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW=0"
        "VLLM_CPU_OFFLOAD_PLAN_CLASS_IMMUTABLE_CPU_SHADOW=0"
        "VLLM_CPU_OFFLOAD_DUAL_VIEW_FULL_PARAMETER_SHADOW=0"
        "VLLM_UVM_KV_CACHE_MANAGED=0"
        "VLLM_UVM_KV_RUNTIME_ENABLE=0"
        "VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0"
        "VLLM_UVM_KV_AUTO_SCALE_ENABLE=0"
      )
      ;;
    native_copy)
      mode=native_copy_eager_continuous
      cpu_offload_gb="${NATIVE_OFFLOAD_GB}"
      exact_names="${NATIVE_NAMES}"
      arm_env+=(
        "VLLM_CPU_OFFLOAD_BACKEND=copy"
        "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE=${NATIVE_NAMES}"
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=0"
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER=0"
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=vmm"
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="
        "VLLM_UVM_EXPERT_WEIGHTS_MANAGED=0"
        "VLLM_UVM_ORIGINAL_VA_PHASE_MODE=off"
        "VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=0"
        "VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW=0"
        "VLLM_CPU_OFFLOAD_PLAN_CLASS_IMMUTABLE_CPU_SHADOW=0"
        "VLLM_CPU_OFFLOAD_DUAL_VIEW_FULL_PARAMETER_SHADOW=0"
        "VLLM_UVM_KV_CACHE_MANAGED=0"
        "VLLM_UVM_KV_RUNTIME_ENABLE=0"
        "VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0"
        "VLLM_UVM_KV_AUTO_SCALE_ENABLE=0"
      )
      ;;
    *) die "unknown arm: ${arm}" ;;
  esac

  local adaptive_runtime_settings="${ADAPT_RUNTIME_SETTINGS_FILE:-}"
  case "${arm}" in
    adaptive_fixed) adaptive_runtime_settings="${ADAPT_FIXED_RUNTIME_SETTINGS_FILE:?}" ;;
    adaptive_fixed_1g) adaptive_runtime_settings="${ADAPT_FIXED1_RUNTIME_SETTINGS_FILE:-${ADAPT_FIXED_RUNTIME_SETTINGS_FILE:?}}" ;;
    adaptive_fixed_2g) adaptive_runtime_settings="${ADAPT_FIXED2_RUNTIME_SETTINGS_FILE:-${ADAPT_FIXED_RUNTIME_SETTINGS_FILE:?}}" ;;
    adaptive_dynamic) adaptive_runtime_settings="${ADAPT_DYNAMIC_RUNTIME_SETTINGS_FILE:?}" ;;
    adaptive_fixed_4g) adaptive_runtime_settings="${ADAPT_FIXED4_RUNTIME_SETTINGS_FILE:?}" ;;
  esac
  if [[ "${arm}" == adaptive* && -n "${adaptive_runtime_settings}" ]]; then
    require_file "${adaptive_runtime_settings}"
    mapfile -t runtime_settings < <(
      "${VLLM_PY}" - "${adaptive_runtime_settings}" "${arm_dir}" <<'PY'
import json, sys
settings = json.load(open(sys.argv[1]))
for key, value in settings.items():
    if not key.startswith('VLLM_') or not isinstance(value, str) or '\n' in value:
        raise SystemExit(f'invalid runtime setting: {key}')
    print(f'{key}={value.replace("__CASE_DIR__", sys.argv[2])}')
PY
    )
    (( ${#runtime_settings[@]} > 0 )) || die "empty/invalid adaptive runtime settings"
    arm_env+=("${runtime_settings[@]}")
    cp "${adaptive_runtime_settings}" "${arm_dir}/runtime_settings.json"
  fi

  printf '%s\n' "${frozen_env[@]}" "${arm_env[@]}" \
    >"${arm_dir}/environment.txt"
  {
    printf 'cd %q && env ' "${VLLM_ROOT}"
    printf '%q ' "${frozen_env[@]}" "${arm_env[@]}"
    printf 'timeout --kill-after=30s %qs taskset -c %q ' \
      "${TIMEOUT_SEC}" "${CPUSET}"
    printf '%q ' "${runner_prefix[@]}" "${VLLM_PY}" "${COMPARE_RUNNER}"
    printf '%q ' --model "${MODEL}" --fixture "${FIXTURE}" \
      --output "${arm_dir}/result.json" --mode "${mode}" \
      --kv-cache-memory-bytes "${arm_kv_cache_bytes}" \
      --cpu-offload-gb "${cpu_offload_gb}" \
      --initial-requests "${INITIAL_REQUESTS}" \
      --late-requests "${LATE_REQUESTS}" \
      --late-after-engine-steps "${LATE_AFTER_ENGINE_STEPS}" \
      --output-tokens "${OUTPUT_TOKENS}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --max-num-seqs "${MAX_NUM_SEQS}" "${arrival_args[@]}"
    printf '\n'
  } >"${arm_dir}/command.txt"

  echo "RUN_BEGIN sample=${sample} arm=${arm} ordinal=${ordinal}"
  kernel_epoch="$(date +%s)"
  set +e
  (
    cd "${VLLM_ROOT}"
    env "${frozen_env[@]}" "${arm_env[@]}" \
      /usr/bin/time -v -o "${arm_dir}/resource_time.txt" \
      timeout --kill-after=30s "${TIMEOUT_SEC}s" \
      taskset -c "${CPUSET}" \
      "${runner_prefix[@]}" "${VLLM_PY}" "${COMPARE_RUNNER}" \
        --model "${MODEL}" --fixture "${FIXTURE}" \
        --output "${arm_dir}/result.json" --mode "${mode}" \
        --kv-cache-memory-bytes "${arm_kv_cache_bytes}" \
        --cpu-offload-gb "${cpu_offload_gb}" \
        --initial-requests "${INITIAL_REQUESTS}" \
        --late-requests "${LATE_REQUESTS}" \
        --late-after-engine-steps "${LATE_AFTER_ENGINE_STEPS}" \
        --output-tokens "${OUTPUT_TOKENS}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --max-num-seqs "${MAX_NUM_SEQS}" "${arrival_args[@]}"
  ) >"${arm_dir}/run.log" 2>&1
  runner_rc=$?
  sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
    >"${arm_dir}/kernel.since.log" 2>"${arm_dir}/kernel.since.err" || true
  set -e
  printf '%s\n' "${runner_rc}" >"${arm_dir}/returncode.txt"
  wait_gpu_clean || die "GPU did not clean after ${arm} sample ${sample}"
  gpu_state_json "${arm_dir}/gpu.after.json"
  printf '%s\t%s\t%s\t%s\t%s\n' "${ordinal}" "${sample}" "${arm}" \
    "$(basename "${arm_dir}")" "${runner_rc}" \
    >>"${run_root}/execution_status.tsv"
  echo "RUN_END sample=${sample} arm=${arm} rc=${runner_rc}"
  if (( runner_rc != 0 )); then
    tail -n 200 "${arm_dir}/run.log" >&2 || true
    return "${runner_rc}"
  fi
  if [[ -n "${exact_names}" ]]; then
    sha256sum "${exact_names}" >"${arm_dir}/selection_sha256.txt"
  fi
}

ordinal=0
for ((sample = 1; sample <= SAMPLES; ++sample)); do
  if (( sample % 2 == 1 )); then
    order=("${selected_arms[@]}")
  else
    order=()
    for ((arm_index = ${#selected_arms[@]} - 1; arm_index >= 0; --arm_index)); do
      order+=("${selected_arms[${arm_index}]}")
    done
  fi
  for arm in "${order[@]}"; do
    ordinal=$((ordinal + 1))
    run_arm "${sample}" "${arm}" "${ordinal}" \
      || die "arm failed: sample=${sample} arm=${arm}"
  done
done

restore_stock || die "failed to restore stock nvidia_uvm"
module_snapshot "${run_root}/module.after.json" "${STOCK_UVM}"
restart_exporters

"${VLLM_PY}" "${COMPARE_VERIFY}" "${run_root}" \
  --output "${run_root}/verification.json" \
  --report "${run_root}/REPORT.md"
echo "RUN_ROOT=${run_root}"
echo "verification=${run_root}/verification.json"
echo "report=${run_root}/REPORT.md"
