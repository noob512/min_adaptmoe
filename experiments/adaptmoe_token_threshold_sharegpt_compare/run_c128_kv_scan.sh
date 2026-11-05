#!/usr/bin/env bash
set -euo pipefail
scan_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${scan_dir}/../.." && pwd)"
scan_root="${1:?new artifact directory required}"
[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || { echo 'Set ALLOW_GPU_RUN=1' >&2; exit 1; }
resume_scan="${RESUME_SCAN:-0}"
if [[ "${resume_scan}" == 1 ]]; then
  [[ -f "${scan_root}/inputs/fixed_runtime_settings.json" ]] || { echo 'Invalid resume root' >&2; exit 1; }
else
  [[ ! -e "${scan_root}" ]] || { echo "Output exists: ${scan_root}" >&2; exit 1; }
  mkdir -p "${scan_root}"
fi
scan_root="$(cd "${scan_root}" && pwd)"
scan_python="${repo_dir}/workloads/vllm/vllm/.venv/bin/python"
exporter_marker="/run/adaptmoe-c128-kv-${BASHPID}.active"
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
  exporter_dropin="${exporter_dropin_dir}/90-adaptmoe-c128-kv.conf"
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
snapshot_dir="${scan_root}/source_snapshots"
status_file="${scan_root}/status.tsv"
if [[ "${resume_scan}" == 1 ]]; then
  snapshot_dir="${scan_root}/resume_source_snapshots_$(date -u +%Y%m%dT%H%M%SZ)"
  status_file="${snapshot_dir}/status.tsv"
fi
mkdir "${snapshot_dir}"
for name in run_c128_kv_scan.sh run_compare.sh c128_kv_runner.py dynamic_kv_p512_runner.py \
  prepare_dynamic_kv_p512.py verify_dynamic_kv_p512.py verify_c128_kv.py \
  summarize_c128_kv.py C128_KV_PROTOCOL_20261024.md; do
  cp "${scan_dir}/${name}" "${snapshot_dir}/${name}"
done
if [[ "${resume_scan}" != 1 ]]; then
cp "${scan_dir}/C128_KV_PROTOCOL_20261024.md" "${scan_root}/PROTOCOL.md"
cp "${repo_dir}/experiments/eurosys_moe_offload_evaluation/e11_continuous_arrival_runner.py" "${snapshot_dir}/base_runner.py"
"${scan_python}" "${scan_dir}/prepare_dynamic_kv_p512.py" "${scan_root}/inputs" --max-num-seqs 128 >"${scan_root}/inputs_prepare.log"
"${scan_python}" - "${scan_root}/inputs" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);dynamic=json.loads((p/'runtime_settings.json').read_text())
dynamic['VLLM_EXPERIMENT_KV_VARIANT']='dynamic'
(p/'runtime_settings.json').write_text(json.dumps(dynamic,indent=2)+'\n')
fixed={'VLLM_EXPERIMENT_KV_VARIANT':'fixed',
       'VLLM_UVM_KV_CACHE_MANAGED':'0','VLLM_UVM_KV_CACHE_LAZY_RESERVE':'0',
       'VLLM_UVM_KV_RUNTIME_ENABLE':'0','VLLM_UVM_KV_DYNAMIC_POOL_ENABLE':'0',
       'VLLM_UVM_KV_AUTO_SCALE_ENABLE':'0',
       'VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASSES_FILE':'',
       'VLLM_CPU_OFFLOAD_PLAN_CLASS_EAGER_ENABLE':'0',
       'VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_BYTES':'2147483648',
       'VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE':'0'}
(p/'fixed_runtime_settings.json').write_text(json.dumps(fixed,indent=2)+'\n')
PY
fi
export COMPARE_RUNNER_OVERRIDE="${scan_dir}/c128_kv_runner.py"
export COMPARE_VERIFY_OVERRIDE="${scan_dir}/verify_c128_kv.py"
export ADAPT_FIXED_RUNTIME_SETTINGS_FILE="${scan_root}/inputs/fixed_runtime_settings.json"
export ADAPT_DYNAMIC_RUNTIME_SETTINGS_FILE="${scan_root}/inputs/runtime_settings.json"
unset ADAPT_RUNTIME_SETTINGS_FILE
printf 'qps\treturncode\n' >"${status_file}"
for rate in 4 8 10 20 30; do
  if [[ "${resume_scan}" == 1 && -d "${scan_root}/qps${rate}" ]]; then
    if [[ -f "${scan_root}/qps${rate}/verification.json" ]]; then
      cp "${scan_root}/qps${rate}/verification.json" "${snapshot_dir}/qps${rate}.verification.before.json"
    fi
    "${scan_python}" "${scan_dir}/verify_c128_kv.py" "${scan_root}/qps${rate}" \
      --output "${scan_root}/qps${rate}/verification.json" --report "${scan_root}/qps${rate}/REPORT.md" \
      >"${snapshot_dir}/qps${rate}.reverify.log"
    printf '%s\t0\n' "${rate}" >>"${status_file}"
    echo "RATE_REVERIFIED qps=${rate}; reuse completed processes"
    continue
  fi
  echo "RATE_BEGIN qps=${rate} C=128 fixed+dynamic"
  if env ALLOW_GPU_RUN=1 SAMPLES=2 ARRIVAL_QPS="${rate}" COMPARE_ARMS='adaptive_fixed adaptive_dynamic' \
    RUN_ROOT="${scan_root}/qps${rate}" \
    FIXTURE="${scan_dir}/inputs/sharegpt_deepseek_p512_n128_20261023.json" \
    INITIAL_REQUESTS=64 LATE_REQUESTS=64 LATE_AFTER_ENGINE_STEPS=4 OUTPUT_TOKENS=16 \
    MAX_MODEL_LEN=576 MAX_NUM_BATCHED_TOKENS=8192 MAX_NUM_SEQS=128 \
    KV_CACHE_BYTES=6442450944 ADAPT_FIXED_KV_CACHE_BYTES=536870912 ADAPT_DYNAMIC_KV_CACHE_BYTES=6442450944 \
    PARK_TOKEN_THRESHOLD=16 RESUME_TOKEN_THRESHOLD=128 NSYS_PROFILE=0 NATIVE_OFFLOAD_GB=5.0 \
    bash "${scan_dir}/run_compare.sh" >"${scan_root}/qps${rate}.log" 2>&1; then
    printf '%s\t0\n' "${rate}" >>"${status_file}"
    echo "RATE_END qps=${rate} rc=0"
  else
    rate_rc=$?
    printf '%s\t%s\n' "${rate}" "${rate_rc}" >>"${status_file}"
    tail -n 85 "${scan_root}/qps${rate}.log" >&2
    exit "${rate_rc}"
  fi
done
"${scan_python}" "${scan_dir}/summarize_c128_kv.py" "${scan_root}"
echo "SCAN_COMPLETED=${scan_root}"
