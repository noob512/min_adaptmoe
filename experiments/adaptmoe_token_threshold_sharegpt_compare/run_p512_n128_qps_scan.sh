#!/usr/bin/env bash
set -euo pipefail
scan_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${scan_dir}/../.." && pwd)"
read -r -a scan_rates <<<"${QPS_RATES:-10 20 30}"
(( ${#scan_rates[@]} > 0 )) || { echo "QPS_RATES must be nonempty" >&2; exit 1; }
declare -A seen_rates=()
for rate in "${scan_rates[@]}"; do
  [[ "${rate}" =~ ^[1-9][0-9]*$ && -z "${seen_rates[${rate}]:-}" ]] \
    || { echo "QPS_RATES must contain unique positive integers" >&2; exit 1; }
  seen_rates["${rate}"]=1
done
rate_label="$(IFS=-; echo "${scan_rates[*]}")"
scan_root="${1:-${scan_dir}/artifacts/p512-n128-c16-qps${rate_label}-$(date -u +%Y%m%dT%H%M%SZ)}"
fixture="${scan_dir}/inputs/sharegpt_deepseek_p512_n128_20261023.json"
[[ -f "${fixture}" ]] || { echo "Build the frozen 128-request fixture first" >&2; exit 1; }
[[ -f "${scan_dir}/summarize_p512_n128_qps_scan.py" ]] || exit 1
[[ ! -e "${scan_root}" || "${RESUME_SCAN:-0}" == 1 ]] || { echo "Output exists: ${scan_root}" >&2; exit 1; }
mkdir -p "${scan_root}"
scan_root="$(cd "${scan_root}" && pwd)"
exporter_dropin_dir=/run/systemd/system/node-exporter.service.d
exporter_dropin=${exporter_dropin_dir}/90-adaptmoe-benchmark.conf
exporter_marker="/run/adaptmoe-p512-n128-qps-scan-${BASHPID}.active"
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
  if (( exporter_was_active )); then
    sudo systemctl start node-exporter.service
  fi
  exit "${original_rc}"
}
trap restore_exporter EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "$(systemctl is-active node-exporter.service 2>/dev/null || true)" == active ]]; then
  exporter_was_active=1
  sudo install -d -m 0755 "${exporter_dropin_dir}"
  [[ ! -e "${exporter_dropin}" ]] || {
    echo "Refusing to replace existing runtime override: ${exporter_dropin}" >&2
    exit 1
  }
  exporter_override_created=1
  sudo touch "${exporter_marker}"
  sudo tee "${exporter_dropin}" >/dev/null <<EOF
[Unit]
ConditionPathExists=!${exporter_marker}
EOF
  sudo systemctl daemon-reload
  sudo systemctl stop node-exporter.service
fi

if [[ ! -e "${scan_root}/PROTOCOL.md" ]]; then
  cp "${scan_dir}/P512_N128_QPS_SCAN_PROTOCOL.md" "${scan_root}/PROTOCOL.md"
fi
"${repo_dir}/workloads/vllm/vllm/.venv/bin/python" - \
  "${scan_root}/scan_config.json" "${fixture}" "${scan_rates[@]}" <<'PY'
import hashlib, json, pathlib, sys
path, fixture, *rate_args = sys.argv[1:]
rates = [int(rate) for rate in rate_args]
value = {'rates': rates, 'samples_per_arm': 2, 'processes': len(rates) * 6,
         'requests': 128, 'fixture_sha256': hashlib.sha256(pathlib.Path(fixture).read_bytes()).hexdigest(),
         'arrival_spans_seconds': {rate: 127 / rate for rate in rates}}
target = pathlib.Path(path)
if target.exists():
    assert json.loads(target.read_text()) == json.loads(json.dumps(value)), 'resume scan configuration differs'
else:
    target.write_text(json.dumps(value, indent=2) + '\n')
PY
if [[ ! -e "${scan_root}/scan_status.tsv" ]]; then
  printf 'qps\treturncode\n' >"${scan_root}/scan_status.tsv"
fi
for rate in "${scan_rates[@]}"; do
  if [[ -d "${scan_root}/qps${rate}" ]]; then
    [[ "${RESUME_SCAN:-0}" == 1 ]] || exit 1
    "${repo_dir}/workloads/vllm/vllm/.venv/bin/python" - "${scan_root}/qps${rate}/verification.json" "${fixture}" <<'PY'
import hashlib, json, sys
value = json.load(open(sys.argv[1]))
assert not value['errors'] and value['status'] in {'ok', 'ok_with_output_drift'}
assert len(value['cases']) == 6 and value['protocol']['requests'] == 128
assert value['protocol']['fixture_sha256'] == hashlib.sha256(open(sys.argv[2], 'rb').read()).hexdigest()
PY
    printf '%s\t0\n' "${rate}" >>"${scan_root}/scan_status.tsv"
    echo "RATE_REUSED qps=${rate}"
    continue
  fi
  echo "RATE_BEGIN qps=${rate}"
  if env ALLOW_GPU_RUN=1 SAMPLES=2 ARRIVAL_QPS="${rate}" \
    COMPARE_ARMS='adaptive native_uva native_copy' \
    RUN_ROOT="${scan_root}/qps${rate}" FIXTURE="${fixture}" \
    INITIAL_REQUESTS=64 LATE_REQUESTS=64 LATE_AFTER_ENGINE_STEPS=4 \
    OUTPUT_TOKENS=16 MAX_MODEL_LEN=576 MAX_NUM_BATCHED_TOKENS=8192 \
    MAX_NUM_SEQS=16 KV_CACHE_BYTES=536870912 \
    PARK_TOKEN_THRESHOLD=16 RESUME_TOKEN_THRESHOLD=128 \
    NSYS_PROFILE=0 NATIVE_OFFLOAD_GB=5.0 \
    bash "${scan_dir}/run_compare.sh" >"${scan_root}/qps${rate}.log" 2>&1; then
    printf '%s\t0\n' "${rate}" >>"${scan_root}/scan_status.tsv"
    echo "RATE_END qps=${rate} rc=0"
  else
    scan_rc=$?
    printf '%s\t%s\n' "${rate}" "${scan_rc}" >>"${scan_root}/scan_status.tsv"
    tail -n 100 "${scan_root}/qps${rate}.log" >&2
    exit "${scan_rc}"
  fi
done
"${repo_dir}/workloads/vllm/vllm/.venv/bin/python" \
  "${scan_dir}/summarize_p512_n128_qps_scan.py" "${scan_root}" --rates "${scan_rates[@]}"
echo "SCAN_ROOT=${scan_root}"
