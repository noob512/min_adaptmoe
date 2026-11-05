#!/usr/bin/env bash
set -euo pipefail
fixed_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fixed_repo="$(cd "${fixed_dir}/../.." && pwd)"
fixed_root="${1:?new artifact directory required}"
[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || { echo 'Set ALLOW_GPU_RUN=1' >&2; exit 1; }
[[ ! -e "${fixed_root}" ]] || { echo "Output exists: ${fixed_root}" >&2; exit 1; }
mkdir -p "${fixed_root}"
fixed_root="$(cd "${fixed_root}" && pwd)"
fixed_python="${fixed_repo}/workloads/vllm/vllm/.venv/bin/python"
fixed_marker="/run/adaptmoe-fixed-k2-k4-${BASHPID}.active"
fixed_active_services=()
fixed_dropins=()
restore_fixed_exporters() {
  local original_rc=$?
  trap - EXIT INT TERM
  set +e
  if (( ${#fixed_dropins[@]} )); then
    sudo rm -f "${fixed_marker}" "${fixed_dropins[@]}"
    sudo systemctl daemon-reload
  fi
  for service in "${fixed_active_services[@]}"; do sudo systemctl start "${service}"; done
  exit "${original_rc}"
}
trap restore_fixed_exporters EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for service in node-exporter.service gpu-exporter.service; do
  if [[ "$(systemctl is-active "${service}" 2>/dev/null || true)" != active ]]; then continue; fi
  fixed_active_services+=("${service}")
  fixed_dropin_dir="/run/systemd/system/${service}.d"
  fixed_dropin="${fixed_dropin_dir}/90-adaptmoe-fixed-k2-k4.conf"
  sudo install -d -m 0755 "${fixed_dropin_dir}"
  [[ ! -e "${fixed_dropin}" ]] || { echo 'Runtime override already exists' >&2; exit 1; }
  fixed_dropins+=("${fixed_dropin}")
  sudo touch "${fixed_marker}"
  sudo tee "${fixed_dropin}" >/dev/null <<EOF
[Unit]
ConditionPathExists=!${fixed_marker}
EOF
  sudo systemctl daemon-reload
  sudo systemctl stop "${service}"
done
mkdir "${fixed_root}/source_snapshots"
for name in run_qps30_fixed_k2_k4.sh qps30_fixed_k2_k4.py run_compare.sh \
  qps30_kv_pressure_runner.py c128_kv_runner.py dynamic_kv_p512_runner.py \
  prepare_dynamic_kv_p512.py verify_dynamic_kv_p512.py verify_p1024_dynamic_kv.py \
  verify_p1024_fixed_vs_dynamic_kv.py analyze_qps30_kv_pressure.py; do
  cp "${fixed_dir}/${name}" "${fixed_root}/source_snapshots/${name}"
done
cp "${fixed_repo}/experiments/eurosys_moe_offload_evaluation/e11_continuous_arrival_runner.py" "${fixed_root}/source_snapshots/base_runner.py"
cp "${fixed_repo}/workloads/vllm/vllm/vllm/v1/engine/core.py" "${fixed_root}/source_snapshots/engine_core.py"
for name in kv_cache_manager.py uvm_kv_runtime_policy.py block_pool.py; do
  cp "${fixed_repo}/workloads/vllm/vllm/vllm/v1/core/${name}" "${fixed_root}/source_snapshots/${name}"
done
cp "${fixed_repo}/workloads/vllm/vllm/vllm/v1/core/sched/scheduler.py" "${fixed_root}/source_snapshots/scheduler.py"
cp "${fixed_repo}/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py" "${fixed_root}/source_snapshots/gpu_model_runner.py"
cp "${fixed_repo}/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py" "${fixed_root}/source_snapshots/gpu_worker.py"
cp "${fixed_repo}/workloads/vllm/vllm/vllm/model_executor/models/utils.py" "${fixed_root}/source_snapshots/model_utils.py"
cp "${fixed_dir}/inputs/sharegpt_deepseek_p1024_n128_20261024.manifest.json" "${fixed_root}/fixture_manifest.json"
"${fixed_python}" "${fixed_dir}/prepare_dynamic_kv_p512.py" "${fixed_root}/inputs" \
  --max-num-seqs 128 --prompt-tokens 1024 >"${fixed_root}/inputs_prepare.log"
"${fixed_python}" - "${fixed_root}/inputs" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);base=json.loads((p/'runtime_settings.json').read_text())
base.update({'VLLM_UVM_KV_AUTO_SCALE_ENABLE':'0',
             # Fixed-capacity runs have no external capacity controller.
             # Otherwise the worker's bootstrap-K2 control file can undo
             # pre-service K4 preparation on the first scheduler poll.
             'VLLM_UVM_KV_DYNAMIC_POOL_CONTROL_FILE':'',
             'VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS':'4313,8629',
             'VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS':'8629',
             'VLLM_EXPERIMENT_KV_VARIANT':'fixed_managed',
             'VLLM_EXPERIMENT_RESIZE_TELEMETRY':'1'})
tiers=json.loads((p/'manifest.json').read_text())['tiers']
configs=[]
bootstrap=next(t for t in tiers if t['kv_gib_label']==2)
for gib in (2,4):
    tier=next(t for t in tiers if t['kv_gib_label']==gib)
    # Nested expert source preparation must start from the smallest class.
    # Use the existing pre-service transaction, outside all request timing,
    # to prepare K4/cold17; autoscaling stays disabled in both arms.
    settings={**base,'VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS':str(bootstrap['target_blocks']),
              'VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS':str(bootstrap['target_blocks']),
              'VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_GROW_TARGET_BLOCKS':str(tier['target_blocks']),
              'VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE':bootstrap['plan_file']}
    (p/f'fixed_k{gib}_runtime_settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    configs.append({**tier,'arm':'adaptive_fixed' if gib==2 else 'adaptive_fixed_4g',
                    'autoscaling':False,'reserved_virtual_kv_gib':6,
                    'bootstrap_target_blocks':bootstrap['target_blocks'],
                    'pre_service_target_blocks':tier['target_blocks']})
(p/'fixed_configs.json').write_text(json.dumps(configs,indent=2)+'\n')
PY
export COMPARE_RUNNER_OVERRIDE="${fixed_dir}/qps30_kv_pressure_runner.py"
export COMPARE_VERIFY_OVERRIDE="${fixed_dir}/qps30_fixed_k2_k4.py"
export ADAPT_FIXED_RUNTIME_SETTINGS_FILE="${fixed_root}/inputs/fixed_k2_runtime_settings.json"
export ADAPT_FIXED4_RUNTIME_SETTINGS_FILE="${fixed_root}/inputs/fixed_k4_runtime_settings.json"
unset ADAPT_RUNTIME_SETTINGS_FILE
echo 'FIXED_KV_BEGIN QPS30 P1024/N128 C128; K2/cold11 versus K4/cold17; two samples per arm'
if env ALLOW_GPU_RUN=1 SAMPLES=2 ARRIVAL_QPS=30 \
  COMPARE_ARMS='adaptive_fixed adaptive_fixed_4g' RUN_ROOT="${fixed_root}/qps30" \
  FIXTURE="${fixed_dir}/inputs/sharegpt_deepseek_p1024_n128_20261024.json" \
  INITIAL_REQUESTS=64 LATE_REQUESTS=64 LATE_AFTER_ENGINE_STEPS=4 OUTPUT_TOKENS=16 \
  MAX_MODEL_LEN=1088 MAX_NUM_BATCHED_TOKENS=8192 MAX_NUM_SEQS=128 \
  KV_CACHE_BYTES=6442450944 ADAPT_FIXED_KV_CACHE_BYTES=6442450944 ADAPT_FIXED4_KV_CACHE_BYTES=6442450944 \
  PARK_TOKEN_THRESHOLD=16 RESUME_TOKEN_THRESHOLD=128 NSYS_PROFILE=0 NATIVE_OFFLOAD_GB=5.0 \
  bash "${fixed_dir}/run_compare.sh" >"${fixed_root}/qps30.log" 2>&1; then
  echo 'FIXED_KV_END rc=0'
else
  fixed_rc=$?
  tail -n 85 "${fixed_root}/qps30.log" >&2
  exit "${fixed_rc}"
fi
"${fixed_python}" "${fixed_dir}/qps30_fixed_k2_k4.py" "${fixed_root}"
echo "FIXED_KV_COMPLETED=${fixed_root}"
