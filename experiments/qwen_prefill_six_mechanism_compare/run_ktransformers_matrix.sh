#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
KT_ROOT="${REPO_ROOT}/KTransformers"
KT_PY="${KT_ROOT}/.venv/bin/python"
CONVERTER="${REPO_ROOT}/experiments/ktransformers_cpu_expert_compare/cold_plan_to_kt_stats.py"
BENCHMARK="${REPO_ROOT}/experiments/ktransformers_l0_l3_phase_compare/benchmark_phase_stream.py"
INPUTS="${REPO_ROOT}/experiments/qwen_decode_six_mechanism_compare/inputs"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
FIXTURE="${FIXTURE:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/artifacts/ktransformers_$(date -u +%Y%m%dT%H%M%SZ)}"
TIERS="${TIERS:-2g,4g}"
MODES="${MODES:-eager,graph}"
PORT="${PORT:-30000}"
CPUSET="${CPUSET:-0-13}"
CONDITIONING_REPEATS="${CONDITIONING_REPEATS:-1}"
MEASUREMENT_REPEATS="${MEASUREMENT_REPEATS:-3}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.97}"
DISABLE_OVERLAP_SCHEDULE="${DISABLE_OVERLAP_SCHEDULE:-0}"

SERVER_PID=""
MONITOR_PID=""
ACTIVE_SERVICES=()

die() { echo "error: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }
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
capture_and_stop_exporters() {
  local service
  for service in node-exporter.service gpu-exporter.service dcgm-exporter.service; do
    if [[ "$(systemctl is-active "${service}" 2>/dev/null || true)" == active ]]; then
      ACTIVE_SERVICES+=("${service}")
      sudo systemctl stop "${service}"
    fi
  done
}
restart_exporters() {
  local service
  for service in "${ACTIVE_SERVICES[@]}"; do sudo systemctl start "${service}" || true; done
  ACTIVE_SERVICES=()
}
cleanup_case() {
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
on_exit() {
  local rc="$?"
  trap - EXIT INT TERM
  cleanup_case
  restart_exporters
  exit "${rc}"
}
trap on_exit EXIT INT TERM

run_case() {
  local tier="$1" mode="$2" case_dir plan ratio stats cold hot_per_layer
  local -a graph_args schedule_args command
  case_dir="${RUN_ROOT}/${tier}_${mode}"
  plan="${INPUTS}/cold_decode_plan_${tier}.json"
  mkdir "${case_dir}" "${case_dir}/kt_stats"
  "${KT_PY}" "${CONVERTER}" "${plan}" --output-dir "${case_dir}/kt_stats" \
    >"${case_dir}/convert.log"
  readarray -t placement < <("${KT_PY}" - "${case_dir}/kt_stats/manifest.json" <<'PY'
import json, sys
entry=json.load(open(sys.argv[1], encoding='utf-8'))['plans'][0]
assert entry['min_cold_experts_per_layer'] == entry['max_cold_experts_per_layer']
assert entry['hot_experts_per_layer'] == [entry['hot_experts_per_layer'][0]] * 48
print(entry['kt_gpu_experts_ratio_cli'])
print(entry['logical_count_path'])
print(entry['cold_experts'])
print(entry['hot_experts_per_layer'][0])
PY
  )
  ratio="${placement[0]}"
  stats="${placement[1]}"
  cold="${placement[2]}"
  hot_per_layer="${placement[3]}"

  graph_args=(--disable-cuda-graph)
  if [[ "${mode}" == graph ]]; then
    graph_args=(
      --cuda-graph-bs 1 4
      --enable-piecewise-cuda-graph
      --piecewise-cuda-graph-tokens 314 1217
      --piecewise-cuda-graph-compiler eager
    )
  fi
  schedule_args=(--enable-mixed-chunk)
  if [[ "${DISABLE_OVERLAP_SCHEDULE}" == 1 ]]; then
    schedule_args=(--disable-overlap-schedule)
  fi
  command=(
    taskset -c "${CPUSET}" "${KT_PY}" -m sglang.launch_server
    --model-path "${MODEL}" --host 127.0.0.1 --port "${PORT}"
    --tp-size 1 --context-length 512 --max-running-requests 4
    --max-total-tokens 1536 --chunked-prefill-size 2048
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --max-prefill-tokens 2048 --schedule-policy fcfs --stream-interval 1
    --random-seed 0 --kt-weight-path "${MODEL}" --kt-method FP8
    --kt-cpuinfer 14 --kt-threadpool-count 1 --kt-numa-nodes 0
    --kt-max-deferred-experts-per-token 0 --kt-gpu-prefill-token-threshold 0
    --kt-gpu-experts-ratio "${ratio}"
    --kt-expert-placement-strategy frequency --init-expert-location "${stats}"
    --attention-backend flashinfer --fp8-gemm-backend triton
    --disable-shared-experts-fusion --disable-radix-cache --enable-metrics
    --enable-request-time-stats-logging "${schedule_args[@]}"
    "${graph_args[@]}"
  )
  printf '%q ' "${command[@]}" >"${case_dir}/server_command.txt"
  printf '\n' >>"${case_dir}/server_command.txt"
  git -C "${KT_ROOT}" rev-parse HEAD >"${case_dir}/ktransformers.commit"
  git -C "${KT_ROOT}/third_party/sglang" rev-parse HEAD >"${case_dir}/sglang.commit"
  sha256sum "${FIXTURE}" "${plan}" >"${case_dir}/input_sha256.txt"
  wait_gpu_clean || die "GPU is not clean before KT ${tier}/${mode}"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    >"${case_dir}/gpu.before.csv"

  setsid env PATH="${KT_ROOT}/.venv/bin:${PATH}" HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    "${command[@]}" >"${case_dir}/server.log" 2>&1 &
  SERVER_PID="$!"
  echo "${SERVER_PID}" >"${case_dir}/server.pid"
  for second in $(seq 1 1200); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "${second}" >"${case_dir}/startup_seconds.txt"
      break
    fi
    kill -0 "${SERVER_PID}" 2>/dev/null \
      || { tail -n 260 "${case_dir}/server.log" >&2; die "KT server exited"; }
    [[ "${second}" != 1200 ]] || die "KT server startup timed out"
    sleep 1
  done

  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits >"${case_dir}/ready_compute.csv"
  nvidia-smi --query-gpu=memory.used,power.draw --format=csv,noheader,nounits \
    >"${case_dir}/ready_gpu.csv"
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader,nounits -lms 100 >"${case_dir}/phase_nvml.csv" \
    2>"${case_dir}/phase_nvml.err" &
  MONITOR_PID="$!"
  "${KT_PY}" "${BENCHMARK}" --fixture "${FIXTURE}" \
    --output "${case_dir}/result.json" \
    --base-url "http://127.0.0.1:${PORT}" --batch-sizes 1,4 \
    --conditioning-repeats "${CONDITIONING_REPEATS}" \
    --measurement-repeats "${MEASUREMENT_REPEATS}" \
    --warmup-decode-steps 1 --measured-decode-steps 1 \
    >"${case_dir}/benchmark.stdout" 2>"${case_dir}/benchmark.stderr"
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
  MONITOR_PID=""
  curl -fsS "http://127.0.0.1:${PORT}/metrics" >"${case_dir}/metrics_final.txt"

  "${KT_PY}" - "${case_dir}" "${mode}" "${cold}" "${hot_per_layer}" <<'PY'
import csv, json, pathlib, re, sys
root=pathlib.Path(sys.argv[1]); mode=sys.argv[2]
cold=int(sys.argv[3]); hot_per_layer=int(sys.argv[4])
manifest=json.load(open(root/'kt_stats/manifest.json', encoding='utf-8'))['plans'][0]
result=json.load(open(root/'result.json', encoding='utf-8'))
assert result['status']=='ok'
text=(root/'server.log').read_text(errors='replace')
pairs={int(a):int(b) for a,b in re.findall(
    r'KT GPU experts: layer (\d+) \(MoE\) has (\d+) GPU experts', text)}
assert len(pairs)==48 and [pairs[i] for i in range(48)]==[hot_per_layer]*48
prefill={314: [], 1217: []}
for tokens, graph in re.findall(
    r'Prefill batch[^\n]*#new-token: (314|1217),[^\n]*cuda graph: (True|False)', text):
    prefill[int(tokens)].append(graph == 'True')
for tokens in (314,1217):
    assert len(prefill[tokens]) >= 4, (tokens, prefill[tokens])
    expected=(mode=='graph')
    assert all(value is expected for value in prefill[tokens][-4:]), prefill[tokens]
if mode=='graph':
    assert 'Capture piecewise CUDA graph begin' in text
    assert 'Capture piecewise CUDA graph end' in text
    assert re.search(r'Capture cuda graph num tokens \[314, 1217\]', text)
else:
    assert 'enable_piecewise_cuda_graph=True' not in text
for batch in ('1','4'):
    phase=result['phase_summary'][batch]
    assert phase['measurement_repeats']==3
    assert len(phase['output_token_ids_sha256'])==1
rows=list(csv.reader(open(root/'phase_nvml.csv')))
memory=[int(row[1].strip()) for row in rows if len(row)>1 and row[1].strip().isdigit()]
ready=int(next(csv.reader(open(root/'ready_gpu.csv')))[0].strip())
summary={
  'prefill_graph_verified': mode=='graph',
  'prefill_graph_observations': {str(k): v for k,v in prefill.items()},
  'ready_gpu_memory_mib': ready,
  'phase_peak_gpu_memory_mib': max(memory),
  'cold_experts': cold,
  'hot_experts': 6144-cold,
  'cold_main_weight_bytes': cold*4_718_592,
  'cold_scale_bytes': cold*1_152,
  'cold_bytes_including_scales': cold*(4_718_592+1_152),
  'plan_sha256': manifest['input_plan_sha256'],
}
(root/'run_summary.json').write_text(
    json.dumps(summary,indent=2,sort_keys=True)+'\n', encoding='utf-8')
PY
  cleanup_case
  wait_gpu_clean || die "GPU did not clean after KT ${tier}/${mode}"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    >"${case_dir}/gpu.after.csv"
  printf '%s\t%s\t%s\t%s\t%s\n' "${tier}" "${mode}" "${cold}" \
    "$(basename "${case_dir}")" 0 >>"${RUN_ROOT}/execution.tsv"
}

[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || die "GPU execution requires ALLOW_GPU_RUN=1"
require_file "${KT_PY}"
require_file "${CONVERTER}"
require_file "${BENCHMARK}"
require_file "${MODEL}/config.json"
require_file "${FIXTURE}"
[[ ! -e "${RUN_ROOT}" ]] || die "RUN_ROOT already exists: ${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
capture_and_stop_exporters
wait_gpu_clean || die "GPU is not initially clean"
printf 'tier\tmode\tcold_experts\tdirectory\treturncode\n' \
  >"${RUN_ROOT}/execution.tsv"
IFS=',' read -r -a tier_list <<<"${TIERS}"
IFS=',' read -r -a mode_list <<<"${MODES}"
for tier in "${tier_list[@]}"; do
  require_file "${INPUTS}/cold_decode_plan_${tier}.json"
  for mode in "${mode_list[@]}"; do
    [[ "${mode}" == eager || "${mode}" == graph ]] \
      || die "KT mode must be eager or graph"
    run_case "${tier}" "${mode}"
  done
done
restart_exporters
trap - EXIT INT TERM
echo "RUN_ROOT=${RUN_ROOT}"
