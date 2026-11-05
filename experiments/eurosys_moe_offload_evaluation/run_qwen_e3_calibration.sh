#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KT_ROOT="${REPO_ROOT}/KTransformers"
KT_PY="${KT_ROOT}/.venv/bin/python"
CLIENT="${REPO_ROOT}/experiments/ktransformers_cpu_expert_compare/benchmark_sglang_stream.py"
FREEZER="${SCRIPT_DIR}/freeze_qwen_e3_plan.py"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
FIXTURE="${FIXTURE:-${SCRIPT_DIR}/inputs/generated/eurosys-pa-v2-20260824/qwen3_30b_a3b_fp8/calibration_ordinary_chat.json}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/e3-calibration-$(date -u +%Y%m%dT%H%M%SZ)}"
PORT="${PORT:-30000}"
CPUSET="${CPUSET:-0-13}"
SERVER_PID=""

die() { echo "error: $*" >&2; exit 1; }
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -INT -- "-${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "${SERVER_PID}" 2>/dev/null || break; sleep 1; done
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
[[ -f "${KT_PY}" && -f "${CLIENT}" && -f "${FREEZER}" ]] || die "tooling is absent"
[[ -f "${MODEL}/config.json" && -f "${FIXTURE}" ]] || die "model or fixture is absent"
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"
mkdir -p "${RUN_ROOT}/recorder_raw"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"

used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
[[ "${used}" == 0 ]] || die "GPU is not clean: ${used} MiB"
sha256sum "${FIXTURE}" "${MODEL}/config.json" >"${RUN_ROOT}/inputs.sha256"
git -C "${REPO_ROOT}" rev-parse HEAD >"${RUN_ROOT}/repo.commit"
git -C "${KT_ROOT}" rev-parse HEAD >"${RUN_ROOT}/ktransformers.commit"
git -C "${KT_ROOT}/third_party/sglang" rev-parse HEAD >"${RUN_ROOT}/sglang.commit"

command=(
  taskset -c "${CPUSET}" "${KT_PY}" -m sglang.launch_server
  --model-path "${MODEL}" --host 127.0.0.1 --port "${PORT}" --tp-size 1
  --context-length 1024 --max-running-requests 32 --max-total-tokens 18000
  --chunked-prefill-size 4096 --max-prefill-tokens 4096
  --schedule-policy fcfs --stream-interval 1 --random-seed 0
  --kt-weight-path "${MODEL}" --kt-method FP8 --kt-cpuinfer 14
  --kt-threadpool-count 1 --kt-numa-nodes 0 --kt-gpu-experts-ratio 0.77083333333333337
  --kt-expert-placement-strategy uniform --kt-max-deferred-experts-per-token 0
  --kt-gpu-prefill-token-threshold 0 --attention-backend flashinfer
  --fp8-gemm-backend triton --expert-distribution-recorder-mode stat
  --expert-distribution-recorder-buffer-size -1 --disable-shared-experts-fusion
  --disable-radix-cache --disable-cuda-graph --disable-overlap-schedule
)
printf '%q ' "${command[@]}" >"${RUN_ROOT}/server_command.txt"
printf '\n' >>"${RUN_ROOT}/server_command.txt"
setsid env PATH="${KT_ROOT}/.venv/bin:${PATH}" HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="${RUN_ROOT}/recorder_raw" \
  "${command[@]}" >"${RUN_ROOT}/server.log" 2>&1 &
SERVER_PID="$!"
echo "${SERVER_PID}" >"${RUN_ROOT}/server.pid"
for second in $(seq 1 900); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "${second}" >"${RUN_ROOT}/startup_seconds.txt"; break
  fi
  kill -0 "${SERVER_PID}" 2>/dev/null || { tail -n 200 "${RUN_ROOT}/server.log" >&2; die "server exited"; }
  [[ "${second}" != 900 ]] || die "server startup timed out"
  sleep 1
done
curl -fsS -X POST "http://127.0.0.1:${PORT}/start_expert_distribution_record" >"${RUN_ROOT}/recorder_start.txt"
"${KT_PY}" "${CLIENT}" --fixture "${FIXTURE}" \
  --output "${RUN_ROOT}/calibration_traffic.json" \
  --base-url "http://127.0.0.1:${PORT}" --concurrencies 32 --output-tokens 8 \
  --request-timeout 3600 --flush-cache-before-phase --require-prompt-token-ids \
  >"${RUN_ROOT}/client.stdout" 2>"${RUN_ROOT}/client.stderr"
curl -fsS -X POST "http://127.0.0.1:${PORT}/stop_expert_distribution_record" >"${RUN_ROOT}/recorder_stop.txt"
curl -fsS -X POST "http://127.0.0.1:${PORT}/dump_expert_distribution_record" >"${RUN_ROOT}/recorder_dump.txt"
cleanup
SERVER_PID=""

mapfile -t stats < <(find "${RUN_ROOT}/recorder_raw" -maxdepth 1 -type f -name 'expert_distribution_recorder_*.pt' -print | sort)
[[ "${#stats[@]}" == 1 ]] || die "expected one recorder file, got ${#stats[@]}"
assets="${RUN_ROOT}/calibration_assets"
"${KT_PY}" "${FREEZER}" --source "${stats[0]}" \
  --calibration-fixture "${FIXTURE}" --output-dir "${assets}" \
  >"${RUN_ROOT}/freeze.stdout.json"
"${KT_PY}" "${REPO_ROOT}/experiments/ktransformers_cpu_expert_compare/cold_plan_to_kt_stats.py" \
  "${assets}/cold_decode_plan_2g.json" --output-dir "${assets}/kt_stats" \
  >"${RUN_ROOT}/kt_convert.stdout.json"
sha256sum "${assets}/cold_decode_plan_2g.json" "${assets}/exact_parameter_names.txt" \
  "${assets}/kt_stats/manifest.json" >"${RUN_ROOT}/outputs.sha256"
echo "RUN_ROOT=${RUN_ROOT}"
