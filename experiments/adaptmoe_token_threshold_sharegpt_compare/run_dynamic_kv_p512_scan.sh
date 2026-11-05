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
exporter_dropin_dir=/run/systemd/system/node-exporter.service.d
exporter_dropin=${exporter_dropin_dir}/90-adaptmoe-dynamic-p512.conf
exporter_marker="/run/adaptmoe-dynamic-p512-${BASHPID}.active"
exporter_was_active=0
exporter_override_created=0

restore_exporter() {
  local original_rc=$?
  trap - EXIT INT TERM
  set +e
  if (( exporter_override_created )); then
    sudo rm -f "${exporter_marker}" "${exporter_dropin}"
    sudo systemctl daemon-reload
  fi
  if (( exporter_was_active )); then sudo systemctl start node-exporter.service; fi
  exit "${original_rc}"
}
trap restore_exporter EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "$(systemctl is-active node-exporter.service 2>/dev/null || true)" == active ]]; then
  exporter_was_active=1
  sudo install -d -m 0755 "${exporter_dropin_dir}"
  [[ ! -e "${exporter_dropin}" ]] || { echo 'Runtime override already exists' >&2; exit 1; }
  exporter_override_created=1
  sudo touch "${exporter_marker}"
  sudo tee "${exporter_dropin}" >/dev/null <<EOF
[Unit]
ConditionPathExists=!${exporter_marker}
EOF
  sudo systemctl daemon-reload
  sudo systemctl stop node-exporter.service
fi

cp "${scan_dir}/DYNAMIC_KV_P512_PROTOCOL_20261024.md" "${scan_root}/PROTOCOL.md"
cp "${BASH_SOURCE[0]}" "${scan_root}/driver.sh.snapshot"
cp "${scan_dir}/run_compare.sh" "${scan_root}/compare.sh.snapshot"
cp "${scan_dir}/dynamic_kv_p512_runner.py" "${scan_root}/runner.py.snapshot"
cp "${repo_dir}/experiments/eurosys_moe_offload_evaluation/e11_continuous_arrival_runner.py" "${scan_root}/base_runner.py.snapshot"
"${scan_python}" "${scan_dir}/prepare_dynamic_kv_p512.py" "${scan_root}/inputs" >"${scan_root}/inputs_prepare.log"
export COMPARE_RUNNER_OVERRIDE="${scan_dir}/dynamic_kv_p512_runner.py"
export COMPARE_VERIFY_OVERRIDE="${scan_dir}/verify_dynamic_kv_p512.py"
export ADAPT_RUNTIME_SETTINGS_FILE="${scan_root}/inputs/runtime_settings.json"
printf 'qps\treturncode\n' >"${scan_root}/status.tsv"
for rate in 4 8 10 20 30; do
  echo "RATE_BEGIN qps=${rate}"
  if env ALLOW_GPU_RUN=1 SAMPLES=2 ARRIVAL_QPS="${rate}" COMPARE_ARMS=adaptive \
    RUN_ROOT="${scan_root}/qps${rate}" \
    FIXTURE="${scan_dir}/inputs/sharegpt_deepseek_p512_n128_20261023.json" \
    INITIAL_REQUESTS=64 LATE_REQUESTS=64 LATE_AFTER_ENGINE_STEPS=4 OUTPUT_TOKENS=16 \
    MAX_MODEL_LEN=576 MAX_NUM_BATCHED_TOKENS=8192 MAX_NUM_SEQS=16 \
    KV_CACHE_BYTES=6442450944 PARK_TOKEN_THRESHOLD=16 RESUME_TOKEN_THRESHOLD=128 \
    NSYS_PROFILE=0 NATIVE_OFFLOAD_GB=5.0 \
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
"${scan_python}" "${scan_dir}/summarize_dynamic_kv_p512.py" "${scan_root}"
echo "SCAN_COMPLETED=${scan_root}"
