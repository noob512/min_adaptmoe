#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ROOT="${1:-}"
MODEL="/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7"
VLLM_PY="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
KT_PY="${REPO_ROOT}/KTransformers/.venv/bin/python"
BLOCK_RUNNER="${SCRIPT_DIR}/formal_openai_block.py"
CLIENT="${SCRIPT_DIR}/streaming_client.py"
WORKLOADS="${FORMAL_WORKLOAD_CONFIG:-${SCRIPT_DIR}/configs/e1_short_c1.json}"
VERIFIER="${SCRIPT_DIR}/verify_formal_e1.py"
EXECUTION_TSV=""

die() {
  echo "error: $*" >&2
  exit 1
}

[[ -n "${RUN_ROOT}" ]] || die "usage: $0 RUN_ROOT"
[[ -d "${RUN_ROOT}" ]] || die "run root does not exist: ${RUN_ROOT}"
[[ -f "${RUN_ROOT}/protocol.json" ]] || die "missing formal protocol"
PROTOCOL_ID="$(jq -er '.protocol_id' "${RUN_ROOT}/protocol.json")" \
  || die "formal protocol ID is absent"
FIXTURE_ROOT="${SCRIPT_DIR}/inputs/generated/${PROTOCOL_ID}/deepseek_v2_lite_bf16"
[[ -x "${VLLM_PY}" && -x "${KT_PY}" ]] || die "missing Python environment"
[[ -f "${MODEL}/config.json" ]] || die "missing model snapshot"
[[ -f "${WORKLOADS}" ]] || die "missing workload config: ${WORKLOADS}"
[[ "$(jq -r '.protocol_id' "${WORKLOADS}")" == "${PROTOCOL_ID}" ]] \
  || die "workload protocol ID differs"
jq -e '.workloads | length == 1
  and .[0].concurrencies == [1]
  and .[0].measurement_requests == 10
  and .[0].minimum_request_start_interval_seconds == 6' \
  "${WORKLOADS}" >/dev/null \
  || die "E1 workload must be C1, ten requests, and <=10 starts/minute"

E1_ROOT="${RUN_ROOT}/cases/E1"
mkdir -p "${E1_ROOT}"
EXECUTION_TSV="${E1_ROOT}/execution.tsv"
if [[ ! -e "${EXECUTION_TSV}" ]]; then
  printf 'created_at\trepeat\torder\tsystem\tattempt\tcase_root\treturncode\n' \
    >"${EXECUTION_TSV}"
fi

SYSTEM_FILTER="${FORMAL_SYSTEMS:-ours,ktransformers,moe_infinity}"
REPEAT_FILTER="${FORMAL_REPEATS:-1,2,3,4,5}"

contains_csv() {
  local needle="$1" haystack=",$2,"
  [[ "${haystack}" == *",${needle},"* ]]
}

successful_attempt_exists() {
  local repeat_root="$1" verification
  [[ -d "${repeat_root}" ]] || return 1
  while IFS= read -r verification; do
    jq -e '.status == "ok"' "${verification}" >/dev/null 2>&1 && return 0
  done < <(find "${repeat_root}" -mindepth 2 -maxdepth 2 \
    -name verification.json -type f | sort)
  return 1
}

next_attempt() {
  local repeat_root="$1" attempt=1
  while [[ -e "${repeat_root}/attempt_${attempt}" ]]; do
    attempt=$((attempt + 1))
  done
  echo "${attempt}"
}

run_case() {
  local repeat="$1" order="$2" system="$3"
  local repeat_root attempt case_root port base_url env_file block_python
  local -a server_command runner_command
  repeat_root="${E1_ROOT}/${system}/deepseek_v2_lite_bf16/maximum_resident/repeat_$(printf '%02d' "${repeat}")"
  if [[ "${FORMAL_FORCE_RETRY:-0}" != 1 ]] \
      && successful_attempt_exists "${repeat_root}"; then
    echo "skip verified E1 case: ${system}/repeat_${repeat}"
    return 0
  fi
  mkdir -p "${repeat_root}"
  attempt="$(next_attempt "${repeat_root}")"
  case_root="${repeat_root}/attempt_${attempt}"

  case "${system}" in
    ours)
      port=30101
      env_file="${SCRIPT_DIR}/configs/e1_ours_env.json"
      block_python="${VLLM_PY}"
      server_command=(
        taskset -c 0-13 "${VLLM_PY}" -m vllm.entrypoints.openai.api_server
        --host 127.0.0.1 --port "${port}" --disable-frontend-multiprocessing
        --model "${MODEL}" --trust-remote-code --dtype bfloat16
        --enforce-eager
        --max-model-len 192 --gpu-memory-utilization 0.97
        --kv-cache-memory-bytes 67108864 --cpu-offload-gb 0
        --max-num-batched-tokens 192 --max-num-seqs 1
        --generation-config vllm
      )
      ;;
    ktransformers)
      port=30102
      env_file="${SCRIPT_DIR}/configs/e1_ktransformers_env.json"
      block_python="${VLLM_PY}"
      server_command=(
        taskset -c 0-13 "${KT_PY}" -m sglang.launch_server
        --model-path "${MODEL}" --host 127.0.0.1 --port "${port}"
        --tp-size 1 --context-length 192 --max-running-requests 1
        --mem-fraction-static 0.97 --max-total-tokens 1024
        --chunked-prefill-size 256 --max-prefill-tokens 256
        --schedule-policy fcfs --stream-interval 1 --random-seed 20260823
        --trust-remote-code --kt-weight-path "${MODEL}" --kt-method BF16
        --kt-cpuinfer 14 --kt-threadpool-count 1 --kt-numa-nodes 0
        --kt-num-gpu-experts 64 --kt-expert-placement-strategy uniform
        --kt-max-deferred-experts-per-token 0 --kt-gpu-prefill-token-threshold 0
        --attention-backend flashinfer --cuda-graph-max-bs 1 --cuda-graph-bs 1
        --enable-mixed-chunk --disable-shared-experts-fusion
        --disable-radix-cache --enable-metrics
      )
      ;;
    moe_infinity)
      port=30103
      env_file="${SCRIPT_DIR}/configs/e1_moe_infinity_env.json"
      block_python="${VLLM_PY}"
      server_command=(
        taskset -c 0-13 "${VLLM_PY}" "${SCRIPT_DIR}/launch_moe_infinity_server.py"
        --model "${MODEL}"
        --offload-path "/home/ubuntu/moe_infinity_offload/deepseek_v2_lite_chat_85864749cd611b4353ce1decdb286193298f64c7"
        --host 127.0.0.1 --port "${port}" --device-memory-ratio 0.75
        --kv-cache-ratio 0.01 --max-batch-size 1 --num-threads 1
      )
      ;;
    *)
      die "unknown E1 system: ${system}"
      ;;
  esac
  base_url="http://127.0.0.1:${port}"
  runner_command=(
    "${VLLM_PY}" "${BLOCK_RUNNER}"
    --case-root "${case_root}" --protocol-id "${PROTOCOL_ID}"
    --system "${system}" --model "${MODEL}" --base-url "${base_url}"
    --workload-config "${WORKLOADS}" --server-env "${env_file}"
    --python "${block_python}" --streaming-client "${CLIENT}"
    --ready-timeout 900 --request-timeout 3600 --gpu-clean-timeout 180
    -- "${server_command[@]}"
  )

  "${runner_command[@]}" >"${case_root}.runner.log" 2>&1
  local rc=$?
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${repeat}" "${order}" \
    "${system}" "${attempt}" "${case_root}" "${rc}" \
    >>"${EXECUTION_TSV}"
  return "${rc}"
}

failures=0
IFS=',' read -r -a selected_repeats <<<"${REPEAT_FILTER}"
for repeat in "${selected_repeats[@]}"; do
  [[ "${repeat}" =~ ^[1-5]$ ]] || die "invalid repeat: ${repeat}"
  case "$(((repeat - 1) % 3))" in
    0) systems=(ours ktransformers moe_infinity) ;;
    1) systems=(ktransformers moe_infinity ours) ;;
    2) systems=(moe_infinity ours ktransformers) ;;
  esac
  order=0
  for system in "${systems[@]}"; do
    contains_csv "${system}" "${SYSTEM_FILTER}" || continue
    order=$((order + 1))
    if ! run_case "${repeat}" "${order}" "${system}"; then
      failures=$((failures + 1))
    fi
  done
done

echo "E1_ROOT=${E1_ROOT}"
echo "E1_FAILURES=${failures}"
if ((failures == 0)) && [[ "${FORMAL_SKIP_TOP_LEVEL_VERIFY:-0}" != 1 ]]; then
  verification="${E1_ROOT}/verification.json"
  [[ ! -e "${verification}" ]] || die "refusing to overwrite ${verification}"
  "${VLLM_PY}" "${VERIFIER}" "${E1_ROOT}" \
    --output "${verification}" --protocol-id "${PROTOCOL_ID}"
  jq -e '.status == "ok"' "${verification}" >/dev/null \
    || die "E1 top-level verification failed"
fi
((failures == 0))
