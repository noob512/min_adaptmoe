#!/usr/bin/env bash
set -euo pipefail
pressure_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pressure_repo="$(cd "${pressure_dir}/../.." && pwd)"
pressure_root="${1:?new artifact directory required}"
pressure_max_gib="${PRESSURE_KV_MAX_GIB:-6}"
[[ "${pressure_max_gib}" == 4 || "${pressure_max_gib}" == 6 ]] || { echo 'PRESSURE_KV_MAX_GIB must be 4 or 6' >&2; exit 1; }
[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || { echo 'Set ALLOW_GPU_RUN=1' >&2; exit 1; }
[[ ! -e "${pressure_root}" ]] || { echo "Output exists: ${pressure_root}" >&2; exit 1; }
mkdir -p "${pressure_root}"
pressure_root="$(cd "${pressure_root}" && pwd)"
pressure_python="${pressure_repo}/workloads/vllm/vllm/.venv/bin/python"
pressure_marker="/run/adaptmoe-qps30-kv-pressure-${BASHPID}.active"
pressure_active_services=()
pressure_created_dropins=()
restore_pressure_exporters() {
  local original_rc=$?
  trap - EXIT INT TERM
  set +e
  if (( ${#pressure_created_dropins[@]} )); then
    sudo rm -f "${pressure_marker}" "${pressure_created_dropins[@]}"
    sudo systemctl daemon-reload
  fi
  for service in "${pressure_active_services[@]}"; do sudo systemctl start "${service}"; done
  exit "${original_rc}"
}
trap restore_pressure_exporters EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for service in node-exporter.service gpu-exporter.service; do
  if [[ "$(systemctl is-active "${service}" 2>/dev/null || true)" != active ]]; then continue; fi
  pressure_active_services+=("${service}")
  pressure_dropin_dir="/run/systemd/system/${service}.d"
  pressure_dropin="${pressure_dropin_dir}/90-adaptmoe-qps30-kv-pressure.conf"
  sudo install -d -m 0755 "${pressure_dropin_dir}"
  [[ ! -e "${pressure_dropin}" ]] || { echo 'Runtime override already exists' >&2; exit 1; }
  pressure_created_dropins+=("${pressure_dropin}")
  sudo touch "${pressure_marker}"
  sudo tee "${pressure_dropin}" >/dev/null <<EOF
[Unit]
ConditionPathExists=!${pressure_marker}
EOF
  sudo systemctl daemon-reload
  sudo systemctl stop "${service}"
done
mkdir "${pressure_root}/source_snapshots"
if [[ "${pressure_max_gib}" == 4 ]]; then
  for name in verify_qps30_k2_k4.py analyze_qps30_k2_k4.py; do
    cp "${pressure_dir}/${name}" "${pressure_root}/source_snapshots/${name}"
  done
fi
for name in run_qps30_kv_pressure.sh qps30_kv_pressure_runner.py run_compare.sh c128_kv_runner.py \
  dynamic_kv_p512_runner.py prepare_dynamic_kv_p512.py verify_dynamic_kv_p512.py \
  verify_p1024_dynamic_kv.py verify_p1024_fixed_vs_dynamic_kv.py analyze_qps30_kv_pressure.py; do
  cp "${pressure_dir}/${name}" "${pressure_root}/source_snapshots/${name}"
done
cp "${pressure_repo}/experiments/eurosys_moe_offload_evaluation/e11_continuous_arrival_runner.py" "${pressure_root}/source_snapshots/base_runner.py"
for name in core.py; do
  cp "${pressure_repo}/workloads/vllm/vllm/vllm/v1/engine/${name}" "${pressure_root}/source_snapshots/engine_${name}"
done
for name in kv_cache_manager.py uvm_kv_runtime_policy.py block_pool.py; do
  cp "${pressure_repo}/workloads/vllm/vllm/vllm/v1/core/${name}" "${pressure_root}/source_snapshots/${name}"
done
cp "${pressure_repo}/workloads/vllm/vllm/vllm/v1/core/sched/scheduler.py" "${pressure_root}/source_snapshots/scheduler.py"
cp "${pressure_dir}/inputs/sharegpt_deepseek_p1024_n128_20261024.manifest.json" "${pressure_root}/fixture_manifest.json"
"${pressure_python}" "${pressure_dir}/prepare_dynamic_kv_p512.py" "${pressure_root}/inputs" \
  --max-num-seqs 128 --prompt-tokens 1024 >"${pressure_root}/inputs_prepare.log"
"${pressure_python}" - "${pressure_root}/inputs" "${pressure_max_gib}" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);dynamic=json.loads((p/'runtime_settings.json').read_text())
if sys.argv[2]=='4':
    dynamic.update({'VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS':'4313,8629',
                    'VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS':'8629'})
dynamic.update({'VLLM_EXPERIMENT_KV_VARIANT':'dynamic','VLLM_EXPERIMENT_RESIZE_TELEMETRY':'1'})
fixed={**dynamic,'VLLM_EXPERIMENT_KV_VARIANT':'fixed_managed','VLLM_UVM_KV_AUTO_SCALE_ENABLE':'0'}
for name,settings in (('dynamic_runtime_settings.json',dynamic),('fixed_runtime_settings.json',fixed)):
    (p/name).write_text(json.dumps(settings,indent=2)+'\n')
(p/'tier_cap.json').write_text(json.dumps({'max_gpu_kv_gib':int(sys.argv[2]),
    'allowed_target_blocks':[int(v) for v in dynamic['VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS'].split(',')],
    'reserved_virtual_kv_gib':6,'initial_gpu_kv_gib':2,
    'expert_registry_preserved_for_identical_initialization':True},indent=2)+'\n')
PY
export COMPARE_RUNNER_OVERRIDE="${pressure_dir}/qps30_kv_pressure_runner.py"
export COMPARE_VERIFY_OVERRIDE="${pressure_dir}/verify_p1024_fixed_vs_dynamic_kv.py"
if [[ "${pressure_max_gib}" == 4 ]]; then
  export COMPARE_VERIFY_OVERRIDE="${pressure_dir}/verify_qps30_k2_k4.py"
fi
export ADAPT_FIXED_RUNTIME_SETTINGS_FILE="${pressure_root}/inputs/fixed_runtime_settings.json"
export ADAPT_DYNAMIC_RUNTIME_SETTINGS_FILE="${pressure_root}/inputs/dynamic_runtime_settings.json"
unset ADAPT_RUNTIME_SETTINGS_FILE
echo "QPS30_BEGIN P1024/N128 C128 fixed K2 versus dynamic capped at K${pressure_max_gib}; two samples per arm"
if env ALLOW_GPU_RUN=1 SAMPLES=2 ARRIVAL_QPS=30 \
  COMPARE_ARMS='adaptive_fixed adaptive_dynamic' RUN_ROOT="${pressure_root}/qps30" \
  FIXTURE="${pressure_dir}/inputs/sharegpt_deepseek_p1024_n128_20261024.json" \
  INITIAL_REQUESTS=64 LATE_REQUESTS=64 LATE_AFTER_ENGINE_STEPS=4 OUTPUT_TOKENS=16 \
  MAX_MODEL_LEN=1088 MAX_NUM_BATCHED_TOKENS=8192 MAX_NUM_SEQS=128 \
  KV_CACHE_BYTES=6442450944 ADAPT_FIXED_KV_CACHE_BYTES=6442450944 ADAPT_DYNAMIC_KV_CACHE_BYTES=6442450944 \
  PARK_TOKEN_THRESHOLD=16 RESUME_TOKEN_THRESHOLD=128 NSYS_PROFILE=0 NATIVE_OFFLOAD_GB=5.0 \
  bash "${pressure_dir}/run_compare.sh" >"${pressure_root}/qps30.log" 2>&1; then
  echo 'QPS30_END rc=0'
else
  pressure_rc=$?
  tail -n 85 "${pressure_root}/qps30.log" >&2
  exit "${pressure_rc}"
fi
if [[ "${pressure_max_gib}" == 4 ]]; then
  "${pressure_python}" "${pressure_dir}/analyze_qps30_k2_k4.py" "${pressure_root}"
else
  "${pressure_python}" "${pressure_dir}/analyze_qps30_kv_pressure.py" "${pressure_root}"
fi
echo "PRESSURE_RUN_COMPLETED=${pressure_root}"
