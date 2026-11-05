#!/usr/bin/env bash
set -euo pipefail
sweep_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sweep_repo="$(cd "${sweep_dir}/../.." && pwd)"
sweep_root="${1:?new artifact directory required}"
[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || { echo 'Set ALLOW_GPU_RUN=1' >&2; exit 1; }
[[ ! -e "${sweep_root}" ]] || { echo "Output exists: ${sweep_root}" >&2; exit 1; }
mkdir -p "${sweep_root}"
sweep_root="$(cd "${sweep_root}" && pwd)"
sweep_python="${sweep_repo}/workloads/vllm/vllm/.venv/bin/python"
sweep_reference="${sweep_dir}/artifacts/p1024-qps30-five-mechanisms-fixed-k1-dynamic-k1-k2-20261024-attempt-01"
export KV1_SWEEP_NEW_QPS="${KV1_SWEEP_NEW_QPS-4 8 10 20}"
export KV1_SWEEP_REUSE_QPS="${KV1_SWEEP_REUSE_QPS-30}"
export KV1_SWEEP_REPORT_STEM="${KV1_SWEEP_REPORT_STEM-P1024_KV1_QPS4_8_10_20_30_FOUR_MECHANISMS_20261024}"
export KV1_SWEEP_SAMPLES="${KV1_SWEEP_SAMPLES-2}"
[[ "${KV1_SWEEP_SAMPLES}" == 1 || "${KV1_SWEEP_SAMPLES}" == 2 ]] || { echo 'KV1_SWEEP_SAMPLES must be 1 or 2' >&2; exit 1; }
read -r -a sweep_new_rates <<<"${KV1_SWEEP_NEW_QPS}"
sweep_marker="/run/adaptmoe-kv1-sweep-${BASHPID}.active"
sweep_services=()
sweep_dropins=()
restore_sweep_exporters() {
  local original_rc=$?
  trap - EXIT INT TERM
  set +e
  if (( ${#sweep_dropins[@]} )); then
    sudo rm -f "${sweep_marker}" "${sweep_dropins[@]}"
    sudo systemctl daemon-reload
  fi
  for service in "${sweep_services[@]}"; do sudo systemctl start "${service}"; done
  exit "${original_rc}"
}
trap restore_sweep_exporters EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for service in node-exporter.service gpu-exporter.service; do
  if [[ "$(systemctl is-active "${service}" 2>/dev/null || true)" != active ]]; then continue; fi
  sweep_services+=("${service}")
  sweep_dropin_dir="/run/systemd/system/${service}.d"
  sweep_dropin="${sweep_dropin_dir}/90-adaptmoe-kv1-sweep.conf"
  sudo install -d -m 0755 "${sweep_dropin_dir}"
  [[ ! -e "${sweep_dropin}" ]] || { echo 'Runtime override already exists' >&2; exit 1; }
  sweep_dropins+=("${sweep_dropin}")
  sudo touch "${sweep_marker}"
  sudo tee "${sweep_dropin}" >/dev/null <<EOF
[Unit]
ConditionPathExists=!${sweep_marker}
EOF
  sudo systemctl daemon-reload
  sudo systemctl stop "${service}"
done
mkdir "${sweep_root}/source_snapshots"
for name in run_kv1_qps_sweep.sh kv1_qps_sweep_compare.py five_arm_k1_compare.py five_arm_k1_runner.py qps30_k1_k2_compare.py run_compare.sh \
  qps30_kv_pressure_runner.py c128_kv_runner.py dynamic_kv_p512_runner.py \
  prepare_dynamic_kv_p512.py verify_dynamic_kv_p512.py verify_p1024_dynamic_kv.py \
  verify_p1024_fixed_vs_dynamic_kv.py analyze_qps30_kv_pressure.py; do
  cp "${sweep_dir}/${name}" "${sweep_root}/source_snapshots/${name}"
done
"${sweep_python}" - "${sweep_repo}" "${sweep_root}" "${sweep_reference}" <<'PY'
import hashlib,json,os,shutil,sys
from pathlib import Path
repo,root,reference=map(Path,sys.argv[1:])
exp=repo/'experiments/adaptmoe_token_threshold_sharegpt_compare'
source_pairs={
 'base_runner.py':'experiments/eurosys_moe_offload_evaluation/e11_continuous_arrival_runner.py',
 'engine_core.py':'workloads/vllm/vllm/vllm/v1/engine/core.py',
 'gpu_model_runner.py':'workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py',
 'gpu_worker.py':'workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py',
 'model_utils.py':'workloads/vllm/vllm/vllm/model_executor/models/utils.py',
 'scheduler.py':'workloads/vllm/vllm/vllm/v1/core/sched/scheduler.py'}
for name in ('kv_cache_manager.py','uvm_kv_runtime_policy.py','block_pool.py'):
 source_pairs[name]='workloads/vllm/vllm/vllm/v1/core/'+name
for name,path in source_pairs.items():
 shutil.copy2(repo/path,root/'source_snapshots'/name)
 assert (repo/path).read_bytes()==(reference/'source_snapshots'/name).read_bytes(),name
for name in ('five_arm_k1_runner.py','qps30_kv_pressure_runner.py','c128_kv_runner.py','dynamic_kv_p512_runner.py','prepare_dynamic_kv_p512.py','run_compare.sh'):
 assert (exp/name).read_bytes()==(reference/'source_snapshots'/name).read_bytes(),name
fixture=exp/'inputs/sharegpt_deepseek_p1024_n128_20261024.json'
new_rates=[int(q) for q in os.environ['KV1_SWEEP_NEW_QPS'].split()]
reused_rates=[int(q) for q in os.environ['KV1_SWEEP_REUSE_QPS'].split()]
assert new_rates and len(set(new_rates+reused_rates))==len(new_rates+reused_rates)
assert all(q in (1,2,4,8,10,20,30) for q in new_rates) and set(reused_rates)<= {30}
stem=os.environ['KV1_SWEEP_REPORT_STEM']
assert Path(stem).name==stem and stem not in ('','.', '..')
protocol={'qps':sorted(new_rates+reused_rates),'new_qps':new_rates,'reused_qps':reused_rates,
 'report_stem':stem,
 'samples_per_new_arm':int(os.environ['KV1_SWEEP_SAMPLES']),
 'roots':{str(q):str(root/f'qps{q}') for q in new_rates},
 'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest(),
 'capacity_certificate':json.loads((reference/'inputs/capacity_certificate.json').read_text())}
if 30 in reused_rates:protocol['roots']['30']=str(reference)
(root/'sweep_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
(root/'source_sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'source_snapshots').iterdir()},indent=2)+'\n')
PY
export COMPARE_RUNNER_OVERRIDE="${sweep_dir}/five_arm_k1_runner.py"
export COMPARE_VERIFY_OVERRIDE="${sweep_dir}/kv1_qps_sweep_compare.py"
unset ADAPT_RUNTIME_SETTINGS_FILE
for rate in "${sweep_new_rates[@]}"; do
  rate_root="${sweep_root}/qps${rate}"
  mkdir "${rate_root}"
  "${sweep_python}" "${sweep_dir}/five_arm_k1_compare.py" "${rate_root}/inputs" --prepare >"${rate_root}/inputs_prepare.log"
  export ADAPT_FIXED_RUNTIME_SETTINGS_FILE="${rate_root}/inputs/adaptive_fixed_1g_runtime_settings.json"
  export ADAPT_FIXED1_RUNTIME_SETTINGS_FILE="${rate_root}/inputs/adaptive_fixed_1g_runtime_settings.json"
  export ADAPT_FIXED2_RUNTIME_SETTINGS_FILE="${rate_root}/inputs/adaptive_fixed_2g_runtime_settings.json"
  export ADAPT_DYNAMIC_RUNTIME_SETTINGS_FILE="${rate_root}/inputs/adaptive_dynamic_runtime_settings.json"
  export NATIVE_NAMES="${rate_root}/inputs/native_cold8_near_equal_names.txt"
  export NATIVE_EXPECTED_BYTES=3690987520 KV1_SWEEP_EXPECTED_QPS="${rate}"
  echo "RATE_BEGIN QPS=${rate} fixed-KV1/dynamic-KV1-2; samples=${KV1_SWEEP_SAMPLES}; no KTransformers"
  if env ALLOW_GPU_RUN=1 SAMPLES="${KV1_SWEEP_SAMPLES}" ARRIVAL_QPS="${rate}" \
    COMPARE_ARMS='native_uva native_copy adaptive_fixed_1g adaptive_dynamic' RUN_ROOT="${rate_root}/vllm" \
    FIXTURE="${sweep_dir}/inputs/sharegpt_deepseek_p1024_n128_20261024.json" \
    INITIAL_REQUESTS=64 LATE_REQUESTS=64 LATE_AFTER_ENGINE_STEPS=4 OUTPUT_TOKENS=16 \
    MAX_MODEL_LEN=1088 MAX_NUM_BATCHED_TOKENS=8192 MAX_NUM_SEQS=128 \
    KV_CACHE_BYTES=1073741824 ADAPT_FIXED1_KV_CACHE_BYTES=2147483648 \
    ADAPT_FIXED2_KV_CACHE_BYTES=2147483648 ADAPT_DYNAMIC_KV_CACHE_BYTES=2147483648 \
    PARK_TOKEN_THRESHOLD=16 RESUME_TOKEN_THRESHOLD=128 NSYS_PROFILE=0 NATIVE_OFFLOAD_GB=4.0 \
    bash "${sweep_dir}/run_compare.sh" >"${rate_root}/run.log" 2>&1; then
    echo "RATE_END QPS=${rate} rc=0"
  else
    rate_rc=$?
    tail -n 90 "${rate_root}/run.log" >&2
    exit "${rate_rc}"
  fi
done
"${sweep_python}" "${sweep_dir}/kv1_qps_sweep_compare.py" "${sweep_root}" --plot
echo "KV1_QPS_SWEEP_COMPLETED=${sweep_root}"
