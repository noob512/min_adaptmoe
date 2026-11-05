#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KT_ROOT="${REPO_ROOT}/KTransformers"
KT_PY="${KT_ROOT}/.venv/bin/python"
CONVERTER="${REPO_ROOT}/experiments/ktransformers_cpu_expert_compare/cold_plan_to_kt_stats.py"
BENCHMARK="${SCRIPT_DIR}/benchmark_phase_stream.py"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
FIXTURE="${FIXTURE:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json}"
PLAN="${PLAN:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/plans/exact_l0_l3_plan.json}"
MODE="${1:-}"
RUN_ROOT="${RUN_ROOT:-/tmp/ktransformers_l0_l3_${MODE}_$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-30000}"
BASE_URL="http://127.0.0.1:${PORT}"
CPUSET="${CPUSET:-0-13}"
OVERLAP_MODE="${OVERLAP_MODE:-serialized}"
SERVER_PID=""
MONITOR_PID=""

die() { echo "error: $*" >&2; exit 1; }
[[ "${MODE}" == "eager" || "${MODE}" == "graph" ]] || die "usage: $0 eager|graph"
[[ "${OVERLAP_MODE}" == "serialized" || "${OVERLAP_MODE}" == "overlap" ]] || \
  die "OVERLAP_MODE must be serialized or overlap"
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"
[[ -f "${MODEL}/config.json" && -f "${FIXTURE}" && -f "${PLAN}" ]] || die "input missing"

cleanup() {
  if [[ -n "${MONITOR_PID}" ]]; then kill "${MONITOR_PID}" 2>/dev/null || true; wait "${MONITOR_PID}" 2>/dev/null || true; fi
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -INT -- "-${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "${SERVER_PID}" 2>/dev/null || break; sleep 1; done
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
[[ "${used}" == "0" && -z "${pids//[[:space:]]/}" ]] || die "GPU not clean: ${used} MiB, pids=${pids:-none}"

mkdir -p "${RUN_ROOT}/kt_stats"
"${KT_PY}" "${CONVERTER}" "${PLAN}" --output-dir "${RUN_ROOT}/kt_stats" > "${RUN_ROOT}/convert.log"
readarray -t placement < <("${KT_PY}" - "${RUN_ROOT}/kt_stats/manifest.json" <<'PY'
import json, sys
entry=json.load(open(sys.argv[1]))["plans"][0]
assert entry["cold_experts"] == 512
assert entry["hot_experts_per_layer"] == [0,0,0,0] + [128]*44
print(entry["kt_gpu_experts_ratio_cli"])
print(entry["logical_count_path"])
PY
)
ratio="${placement[0]}"
stats="${placement[1]}"

graph_args=(--disable-cuda-graph)
if [[ "${MODE}" == "graph" ]]; then graph_args=(--cuda-graph-bs 1 4); fi
schedule_args=(--disable-overlap-schedule)
if [[ "${OVERLAP_MODE}" == "overlap" ]]; then schedule_args=(--enable-mixed-chunk); fi
command=(
  taskset -c "${CPUSET}" "${KT_PY}" -m sglang.launch_server
  --model-path "${MODEL}" --host 127.0.0.1 --port "${PORT}"
  --tp-size 1 --context-length 512 --max-running-requests 4 --max-total-tokens 1536
  --chunked-prefill-size 2048 --max-prefill-tokens 2048
  --schedule-policy fcfs --stream-interval 1 --random-seed 0
  --kt-weight-path "${MODEL}" --kt-method FP8 --kt-cpuinfer 14
  --kt-threadpool-count 1 --kt-numa-nodes 0
  --kt-max-deferred-experts-per-token 0 --kt-gpu-prefill-token-threshold 0
  --kt-gpu-experts-ratio "${ratio}" --kt-expert-placement-strategy frequency
  --init-expert-location "${stats}"
  --attention-backend flashinfer --fp8-gemm-backend triton
  --disable-shared-experts-fusion --disable-radix-cache
  --enable-metrics "${schedule_args[@]}" "${graph_args[@]}"
)
printf '%q ' "${command[@]}" > "${RUN_ROOT}/server_command.txt"
printf '\n' >> "${RUN_ROOT}/server_command.txt"
git -C "${KT_ROOT}" rev-parse HEAD > "${RUN_ROOT}/ktransformers.commit"
git -C "${KT_ROOT}/third_party/sglang" rev-parse HEAD > "${RUN_ROOT}/sglang.commit"
sha256sum "${FIXTURE}" "${PLAN}" > "${RUN_ROOT}/input_sha256.txt"
cat /sys/module/nvidia_uvm/srcversion > "${RUN_ROOT}/uvm_srcversion.txt"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader > "${RUN_ROOT}/gpu_before.csv"

setsid env PATH="${KT_ROOT}/.venv/bin:${PATH}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  "${command[@]}" > "${RUN_ROOT}/server.log" 2>&1 &
SERVER_PID="$!"
echo "${SERVER_PID}" > "${RUN_ROOT}/server.pid"
for second in $(seq 1 900); do
  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then echo "${second}" > "${RUN_ROOT}/startup_seconds.txt"; break; fi
  kill -0 "${SERVER_PID}" 2>/dev/null || { tail -n 240 "${RUN_ROOT}/server.log" >&2; die "server exited"; }
  [[ "${second}" != "900" ]] || die "server startup timed out"
  sleep 1
done

nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits > "${RUN_ROOT}/ready_compute.csv"
nvidia-smi --query-gpu=memory.used,power.draw,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits > "${RUN_ROOT}/ready_gpu.csv"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,utilization.memory,power.draw,pcie.link.gen.current,pcie.link.width.current \
  --format=csv,noheader,nounits -lms 100 > "${RUN_ROOT}/phase_nvml.csv" 2> "${RUN_ROOT}/phase_nvml.err" &
MONITOR_PID="$!"
"${KT_PY}" "${BENCHMARK}" --fixture "${FIXTURE}" --output "${RUN_ROOT}/result.json" --base-url "${BASE_URL}" \
  > "${RUN_ROOT}/benchmark.stdout" 2> "${RUN_ROOT}/benchmark.stderr"
kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
MONITOR_PID=""
curl -fsS "${BASE_URL}/metrics" > "${RUN_ROOT}/metrics_final.txt"

"${KT_PY}" - "${RUN_ROOT}" <<'PY'
import csv, json, pathlib, re, sys
root=pathlib.Path(sys.argv[1])
manifest=json.load(open(root/'kt_stats/manifest.json'))['plans'][0]
text=(root/'server.log').read_text(errors='replace')
pairs={int(a):int(b) for a,b in re.findall(r'KT GPU experts: layer (\d+) \(MoE\) has (\d+) GPU experts', text)}
assert len(pairs)==48 and [pairs[i] for i in range(48)]==manifest['hot_experts_per_layer']
rows=list(csv.reader(open(root/'phase_nvml.csv')))
memory=[int(row[1].strip()) for row in rows if len(row)>1 and row[1].strip().isdigit()]
ready=int(next(csv.reader(open(root/'ready_gpu.csv')))[0].strip())
summary={
 'runtime_mask_verified': True,
 'ready_gpu_memory_mib': ready,
 'phase_peak_gpu_memory_mib': max(memory),
 'cold_experts': 512,
 'hot_experts': 5632,
 'cold_main_weight_bytes': 512*4718592,
 'cold_scale_bytes': 512*1152,
 'cold_bytes_including_scales': 512*(4718592+1152),
}
(root/'run_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps(summary,sort_keys=True))
PY

cleanup
SERVER_PID=""
trap - EXIT
used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
[[ "${used}" == "0" && -z "${pids//[[:space:]]/}" ]] || die "GPU did not clean after run"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > "${RUN_ROOT}/gpu_after.csv"
echo "RUN_ROOT=${RUN_ROOT}"
