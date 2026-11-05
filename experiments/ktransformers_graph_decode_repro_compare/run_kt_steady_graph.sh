#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KT_ROOT="${REPO_ROOT}/KTransformers"
KT_PY="${KT_ROOT}/.venv/bin/python"
BENCHMARK="${REPO_ROOT}/experiments/ktransformers_l0_l3_phase_compare/benchmark_phase_stream.py"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
FIXTURE="${FIXTURE:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json}"
STATS="${STATS:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/routing/frequency_stats.pt}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/kt_steady_graph_$(date -u +%Y%m%dT%H%M%SZ)}"
PORT="${PORT:-30000}"
CPUSET="${CPUSET:-0-13}"
SERVER_PID=""
MONITOR_PID=""

die() { echo "error: $*" >&2; exit 1; }
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
cleanup() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=""
  fi
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -INT -- "-${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT INT TERM

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
[[ -x "${KT_PY}" ]] || die "missing KT Python"
[[ -f "${BENCHMARK}" && -f "${FIXTURE}" && -f "${STATS}" ]] \
  || die "missing benchmark input"
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
wait_gpu_clean || die "GPU is not initially clean"

command=(
  taskset -c "${CPUSET}" "${KT_PY}" -m sglang.launch_server
  --model-path "${MODEL}" --host 127.0.0.1 --port "${PORT}"
  --tp-size 1 --context-length 512 --max-running-requests 32
  --max-total-tokens 13104 --chunked-prefill-size 4096
  --mem-fraction-static 0.79 --max-prefill-tokens 4096
  --schedule-policy fcfs --stream-interval 1 --random-seed 0
  --trust-remote-code --kt-weight-path "${MODEL}" --kt-method FP8
  --kt-cpuinfer 12 --kt-threadpool-count 1 --kt-numa-nodes 0
  --kt-gpu-experts-ratio 0.77083333333333337
  --kt-expert-placement-strategy frequency --init-expert-location "${STATS}"
  --kt-max-deferred-experts-per-token 0 --kt-gpu-prefill-token-threshold 0
  --attention-backend flashinfer --fp8-gemm-backend triton
  --enable-mixed-chunk --disable-shared-experts-fusion --disable-radix-cache
  --enable-metrics
)
printf '%q ' "${command[@]}" >"${RUN_ROOT}/server_command.txt"
printf '\n' >>"${RUN_ROOT}/server_command.txt"
git -C "${KT_ROOT}" rev-parse HEAD >"${RUN_ROOT}/ktransformers.commit"
git -C "${KT_ROOT}/third_party/sglang" rev-parse HEAD \
  >"${RUN_ROOT}/sglang.commit"
sha256sum "${FIXTURE}" "${STATS}" >"${RUN_ROOT}/input_sha256.txt"

setsid env PATH="${KT_ROOT}/.venv/bin:${PATH}" HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  "${command[@]}" >"${RUN_ROOT}/server.log" 2>&1 &
SERVER_PID="$!"
echo "${SERVER_PID}" >"${RUN_ROOT}/server.pid"
for second in $(seq 1 900); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "${second}" >"${RUN_ROOT}/startup_seconds.txt"
    break
  fi
  kill -0 "${SERVER_PID}" 2>/dev/null \
    || { tail -n 240 "${RUN_ROOT}/server.log" >&2; die "KT server exited"; }
  [[ "${second}" != 900 ]] || die "KT server startup timed out"
  sleep 1
done

nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits >"${RUN_ROOT}/ready_compute.csv"
nvidia-smi --query-gpu=memory.used,power.draw --format=csv,noheader,nounits \
  >"${RUN_ROOT}/ready_gpu.csv"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader,nounits -lms 100 >"${RUN_ROOT}/phase_nvml.csv" \
  2>"${RUN_ROOT}/phase_nvml.err" &
MONITOR_PID="$!"
"${KT_PY}" "${BENCHMARK}" --fixture "${FIXTURE}" \
  --output "${RUN_ROOT}/result.json" \
  --base-url "http://127.0.0.1:${PORT}" --batch-sizes 1,4 \
  --conditioning-repeats 1 --measurement-repeats 3 \
  --warmup-decode-steps 5 --measured-decode-steps 58 \
  >"${RUN_ROOT}/benchmark.stdout" 2>"${RUN_ROOT}/benchmark.stderr"
kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
MONITOR_PID=""
curl -fsS "http://127.0.0.1:${PORT}/metrics" >"${RUN_ROOT}/metrics_final.txt"

"${KT_PY}" - "${RUN_ROOT}" <<'PY'
import json, pathlib, re, sys
root = pathlib.Path(sys.argv[1])
result = json.load(open(root / "result.json", encoding="utf-8"))
assert result["status"] == "ok"
text = (root / "server.log").read_text(errors="replace")
pairs = {int(a): int(b) for a, b in re.findall(
    r"KT GPU experts: layer (\d+) \(MoE\) has (\d+) GPU experts", text)}
assert len(pairs) == 48 and sum(pairs.values()) == 4736
for batch in ("1", "4"):
    summary = result["phase_summary"][batch]
    assert summary["measurement_repeats"] == 3
    assert summary["cuda_graph_pass_delta"]["true"] > 0
    assert summary["cuda_graph_pass_delta"]["false"] == 0
    assert len(summary["output_token_ids_sha256"]) == 1
(root / "run_summary.json").write_text(json.dumps({
    "status": "ok",
    "cuda_graph_verified": True,
    "runtime_mask_layers": len(pairs),
    "hot_experts": sum(pairs.values()),
    "cold_experts": 6144 - sum(pairs.values()),
    "cold_main_weight_bytes": (6144 - sum(pairs.values())) * 4_718_592,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

cleanup
wait_gpu_clean || die "GPU did not clean after KT steady run"
echo "RUN_ROOT=${RUN_ROOT}"
