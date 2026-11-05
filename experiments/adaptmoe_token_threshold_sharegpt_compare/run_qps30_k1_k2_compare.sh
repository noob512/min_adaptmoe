#!/usr/bin/env bash
set -euo pipefail
small_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
small_repo="$(cd "${small_dir}/../.." && pwd)"
small_root="${1:?new artifact directory required}"
[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || { echo 'Set ALLOW_GPU_RUN=1' >&2; exit 1; }
[[ ! -e "${small_root}" ]] || { echo "Output exists: ${small_root}" >&2; exit 1; }
mkdir -p "${small_root}"
small_root="$(cd "${small_root}" && pwd)"
small_python="${small_repo}/workloads/vllm/vllm/.venv/bin/python"
small_marker="/run/adaptmoe-k1-k2-${BASHPID}.active"
small_active_services=()
small_dropins=()
restore_small_exporters() {
  local original_rc=$?
  trap - EXIT INT TERM
  set +e
  if (( ${#small_dropins[@]} )); then
    sudo rm -f "${small_marker}" "${small_dropins[@]}"
    sudo systemctl daemon-reload
  fi
  for service in "${small_active_services[@]}"; do sudo systemctl start "${service}"; done
  exit "${original_rc}"
}
trap restore_small_exporters EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for service in node-exporter.service gpu-exporter.service; do
  if [[ "$(systemctl is-active "${service}" 2>/dev/null || true)" != active ]]; then continue; fi
  small_active_services+=("${service}")
  small_dropin_dir="/run/systemd/system/${service}.d"
  small_dropin="${small_dropin_dir}/90-adaptmoe-k1-k2.conf"
  sudo install -d -m 0755 "${small_dropin_dir}"
  [[ ! -e "${small_dropin}" ]] || { echo 'Runtime override already exists' >&2; exit 1; }
  small_dropins+=("${small_dropin}")
  sudo touch "${small_marker}"
  sudo tee "${small_dropin}" >/dev/null <<EOF
[Unit]
ConditionPathExists=!${small_marker}
EOF
  sudo systemctl daemon-reload
  sudo systemctl stop "${service}"
done
mkdir "${small_root}/source_snapshots"
for name in run_qps30_k1_k2_compare.sh qps30_k1_k2_compare.py run_compare.sh \
  qps30_kv_pressure_runner.py c128_kv_runner.py dynamic_kv_p512_runner.py \
  prepare_dynamic_kv_p512.py verify_dynamic_kv_p512.py verify_p1024_dynamic_kv.py \
  verify_p1024_fixed_vs_dynamic_kv.py analyze_qps30_kv_pressure.py; do
  cp "${small_dir}/${name}" "${small_root}/source_snapshots/${name}"
done
cp "${small_repo}/experiments/eurosys_moe_offload_evaluation/e11_continuous_arrival_runner.py" "${small_root}/source_snapshots/base_runner.py"
cp "${small_repo}/workloads/vllm/vllm/vllm/v1/engine/core.py" "${small_root}/source_snapshots/engine_core.py"
for name in kv_cache_manager.py uvm_kv_runtime_policy.py block_pool.py; do
  cp "${small_repo}/workloads/vllm/vllm/vllm/v1/core/${name}" "${small_root}/source_snapshots/${name}"
done
cp "${small_repo}/workloads/vllm/vllm/vllm/v1/core/sched/scheduler.py" "${small_root}/source_snapshots/scheduler.py"
cp "${small_repo}/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py" "${small_root}/source_snapshots/gpu_model_runner.py"
cp "${small_repo}/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py" "${small_root}/source_snapshots/gpu_worker.py"
cp "${small_repo}/workloads/vllm/vllm/vllm/model_executor/models/utils.py" "${small_root}/source_snapshots/model_utils.py"
cp "${small_dir}/inputs/sharegpt_deepseek_p1024_n128_20261024.manifest.json" "${small_root}/fixture_manifest.json"
"${small_python}" "${small_dir}/qps30_k1_k2_compare.py" "${small_root}/inputs" --prepare >"${small_root}/inputs_prepare.log"
export COMPARE_RUNNER_OVERRIDE="${small_dir}/qps30_kv_pressure_runner.py"
export COMPARE_VERIFY_OVERRIDE="${small_dir}/qps30_k1_k2_compare.py"
export ADAPT_FIXED_RUNTIME_SETTINGS_FILE="${small_root}/inputs/adaptive_fixed_1g_runtime_settings.json"
export ADAPT_FIXED1_RUNTIME_SETTINGS_FILE="${small_root}/inputs/adaptive_fixed_1g_runtime_settings.json"
export ADAPT_FIXED2_RUNTIME_SETTINGS_FILE="${small_root}/inputs/adaptive_fixed_2g_runtime_settings.json"
export ADAPT_DYNAMIC_RUNTIME_SETTINGS_FILE="${small_root}/inputs/adaptive_dynamic_runtime_settings.json"
unset ADAPT_RUNTIME_SETTINGS_FILE
echo 'K1_K2_BEGIN QPS30 P1024/N128 C128; fixed K1, fixed K2, automatic K1/K2; two samples per arm'
if env ALLOW_GPU_RUN=1 SAMPLES=2 ARRIVAL_QPS=30 \
  COMPARE_ARMS='adaptive_fixed_1g adaptive_fixed_2g adaptive_dynamic' RUN_ROOT="${small_root}/qps30" \
  FIXTURE="${small_dir}/inputs/sharegpt_deepseek_p1024_n128_20261024.json" \
  INITIAL_REQUESTS=64 LATE_REQUESTS=64 LATE_AFTER_ENGINE_STEPS=4 OUTPUT_TOKENS=16 \
  MAX_MODEL_LEN=1088 MAX_NUM_BATCHED_TOKENS=8192 MAX_NUM_SEQS=128 \
  KV_CACHE_BYTES=2147483648 ADAPT_FIXED1_KV_CACHE_BYTES=2147483648 \
  ADAPT_FIXED2_KV_CACHE_BYTES=2147483648 ADAPT_DYNAMIC_KV_CACHE_BYTES=2147483648 \
  PARK_TOKEN_THRESHOLD=16 RESUME_TOKEN_THRESHOLD=128 NSYS_PROFILE=0 NATIVE_OFFLOAD_GB=5.0 \
  bash "${small_dir}/run_compare.sh" >"${small_root}/qps30.log" 2>&1; then
  echo 'K1_K2_END rc=0'
else
  small_rc=$?
  tail -n 85 "${small_root}/qps30.log" >&2
  exit "${small_rc}"
fi
"${small_python}" "${small_dir}/qps30_k1_k2_compare.py" "${small_root}"
echo "K1_K2_COMPLETED=${small_root}"
