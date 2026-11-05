#!/usr/bin/env bash
set -euo pipefail
scan_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${scan_dir}/../.." && pwd)"
scan_root="${1:?new artifact directory required}"
[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || { echo 'Set ALLOW_GPU_RUN=1' >&2; exit 1; }
[[ ! -e "${scan_root}" ]] || { echo "Output exists: ${scan_root}" >&2; exit 1; }
mkdir -p "${scan_root}"
scan_root="$(cd "${scan_root}" && pwd)"
scan_python="${repo_dir}/workloads/vllm/vllm/.venv/bin/python"
exporter_marker="/run/adaptmoe-p1024-kv-compare-${BASHPID}.active"
exporter_active_services=()
exporter_created_dropins=()
restore_exporter() {
  local original_rc=$?
  trap - EXIT INT TERM
  set +e
  if (( ${#exporter_created_dropins[@]} )); then
    sudo rm -f "${exporter_marker}" "${exporter_created_dropins[@]}"
    sudo systemctl daemon-reload
  fi
  for service in "${exporter_active_services[@]}"; do sudo systemctl start "${service}"; done
  exit "${original_rc}"
}
trap restore_exporter EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for service in node-exporter.service gpu-exporter.service; do
  if [[ "$(systemctl is-active "${service}" 2>/dev/null || true)" != active ]]; then continue; fi
  exporter_active_services+=("${service}")
  exporter_dropin_dir="/run/systemd/system/${service}.d"
  exporter_dropin="${exporter_dropin_dir}/90-adaptmoe-p1024-kv-compare.conf"
  sudo install -d -m 0755 "${exporter_dropin_dir}"
  [[ ! -e "${exporter_dropin}" ]] || { echo 'Runtime override already exists' >&2; exit 1; }
  exporter_created_dropins+=("${exporter_dropin}")
  sudo touch "${exporter_marker}"
  sudo tee "${exporter_dropin}" >/dev/null <<EOF
[Unit]
ConditionPathExists=!${exporter_marker}
EOF
  sudo systemctl daemon-reload
  sudo systemctl stop "${service}"
done
mkdir "${scan_root}/source_snapshots"
for name in run_p1024_fixed_vs_dynamic_kv_scan.sh run_compare.sh c128_kv_runner.py \
  dynamic_kv_p512_runner.py prepare_dynamic_kv_p512.py verify_dynamic_kv_p512.py \
  verify_p1024_dynamic_kv.py verify_p1024_fixed_vs_dynamic_kv.py \
  summarize_p1024_fixed_vs_dynamic_kv.py P1024_FIXED_VS_DYNAMIC_KV_PROTOCOL_20261024.md; do
  cp "${scan_dir}/${name}" "${scan_root}/source_snapshots/${name}"
done
cp "${repo_dir}/experiments/eurosys_moe_offload_evaluation/e11_continuous_arrival_runner.py" "${scan_root}/source_snapshots/base_runner.py"
cp "${repo_dir}/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py" "${scan_root}/source_snapshots/gpu_model_runner.py"
cp "${repo_dir}/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py" "${scan_root}/source_snapshots/gpu_worker.py"
cp "${repo_dir}/workloads/vllm/vllm/vllm/device_allocator/uvm.py" "${scan_root}/source_snapshots/uvm.py"
cp "${scan_dir}/P1024_FIXED_VS_DYNAMIC_KV_PROTOCOL_20261024.md" "${scan_root}/PROTOCOL.md"
cp "${scan_dir}/inputs/sharegpt_deepseek_p1024_n128_20261024.manifest.json" "${scan_root}/fixture_manifest.json"
"${scan_python}" "${scan_dir}/prepare_dynamic_kv_p512.py" "${scan_root}/inputs" \
  --max-num-seqs 128 --prompt-tokens 1024 >"${scan_root}/inputs_prepare.log"
"${scan_python}" - "${scan_root}/inputs" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);dynamic=json.loads((p/'runtime_settings.json').read_text())
dynamic.update({'VLLM_EXPERIMENT_KV_VARIANT':'dynamic','VLLM_EXPERIMENT_RESIZE_TELEMETRY':'1'})
fixed={**dynamic,'VLLM_EXPERIMENT_KV_VARIANT':'fixed_managed','VLLM_UVM_KV_AUTO_SCALE_ENABLE':'0'}
for name,settings in (('dynamic_runtime_settings.json',dynamic),('fixed_runtime_settings.json',fixed)):
    (p/name).write_text(json.dumps(settings,indent=2)+'\n')
PY
export COMPARE_RUNNER_OVERRIDE="${scan_dir}/c128_kv_runner.py"
export COMPARE_VERIFY_OVERRIDE="${scan_dir}/verify_p1024_fixed_vs_dynamic_kv.py"
export ADAPT_FIXED_RUNTIME_SETTINGS_FILE="${scan_root}/inputs/fixed_runtime_settings.json"
export ADAPT_DYNAMIC_RUNTIME_SETTINGS_FILE="${scan_root}/inputs/dynamic_runtime_settings.json"
unset ADAPT_RUNTIME_SETTINGS_FILE
printf 'qps\treturncode\n' >"${scan_root}/status.tsv"
for rate in 20 30 4 8 10; do
  echo "RATE_BEGIN qps=${rate} P1024 C128 fixed K2 + dynamic K2/K4/K6"
  if env ALLOW_GPU_RUN=1 SAMPLES=2 ARRIVAL_QPS="${rate}" \
    COMPARE_ARMS='adaptive_fixed adaptive_dynamic' RUN_ROOT="${scan_root}/qps${rate}" \
    FIXTURE="${scan_dir}/inputs/sharegpt_deepseek_p1024_n128_20261024.json" \
    INITIAL_REQUESTS=64 LATE_REQUESTS=64 LATE_AFTER_ENGINE_STEPS=4 OUTPUT_TOKENS=16 \
    MAX_MODEL_LEN=1088 MAX_NUM_BATCHED_TOKENS=8192 MAX_NUM_SEQS=128 \
    KV_CACHE_BYTES=6442450944 ADAPT_FIXED_KV_CACHE_BYTES=6442450944 ADAPT_DYNAMIC_KV_CACHE_BYTES=6442450944 \
    PARK_TOKEN_THRESHOLD=16 RESUME_TOKEN_THRESHOLD=128 NSYS_PROFILE=0 NATIVE_OFFLOAD_GB=5.0 \
    bash "${scan_dir}/run_compare.sh" >"${scan_root}/qps${rate}.log" 2>&1; then
    printf '%s\t0\n' "${rate}" >>"${scan_root}/status.tsv"
    echo "RATE_END qps=${rate} rc=0"
  else
    rate_rc=$?
    printf '%s\t%s\n' "${rate}" "${rate_rc}" >>"${scan_root}/status.tsv"
    tail -n 85 "${scan_root}/qps${rate}.log" >&2
    exit "${rate_rc}"
  fi
done
"${scan_python}" "${scan_dir}/summarize_p1024_fixed_vs_dynamic_kv.py" "${scan_root}"
echo "SCAN_COMPLETED=${scan_root}"
