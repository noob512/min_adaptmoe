#!/usr/bin/env bash
set -euo pipefail
scan_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${scan_dir}/../.." && pwd)"
scan_root="${1:-${scan_dir}/artifacts/kt-p512-n128-qps4-8-10-20-30-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "${ALLOW_GPU_RUN:-0}" == 1 ]] || { echo 'Set ALLOW_GPU_RUN=1' >&2; exit 1; }
[[ ! -e "${scan_root}" ]] || { echo "Output exists: ${scan_root}" >&2; exit 1; }
mkdir -p "${scan_root}"
scan_root="$(cd "${scan_root}" && pwd)"
kt_python="${repo_dir}/KTransformers/.venv/bin/python"
fixture="${scan_dir}/inputs/sharegpt_deepseek_p512_n128_20261023.json"
cold_plan="${repo_dir}/experiments/eurosys_moe_offload_evaluation/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/calibration_assets/plans_adjusted_11_17_23_29/cold_plan_2g.json"
model="${MODEL:?Set MODEL to the local DeepSeek-V2-Lite-Chat snapshot directory}"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=14 MKL_NUM_THREADS=14
export SGLANG_APPLY_CONFIG_BACKUP=none SGLANG_E3_SERIALIZE_SCHEDULER_TIME_STATS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="${repo_dir}/KTransformers/.venv/bin:${PATH}"

gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
[[ -z "${gpu_pids}" ]] || { echo "GPU already in use: ${gpu_pids}" >&2; exit 1; }
cp "${scan_dir}/KTRANSFORMERS_P512_PROTOCOL.md" "${scan_root}/PROTOCOL.md"
cp "${scan_dir}/ktransformers_p512_runner.py" "${scan_root}/runner.py.snapshot"
cp "${BASH_SOURCE[0]}" "${scan_root}/driver.sh.snapshot"
sha256sum "${fixture}" "${cold_plan}" "${scan_dir}/ktransformers_p512_runner.py" >"${scan_root}/input_and_code_sha256.txt"
git -C "${repo_dir}/KTransformers" rev-parse HEAD >"${scan_root}/ktransformers.commit"
git -C "${repo_dir}/KTransformers/third_party/sglang" rev-parse HEAD >"${scan_root}/sglang.commit"
git -C "${repo_dir}/KTransformers/third_party/sglang" diff >"${scan_root}/sglang.local.diff"
nvidia-smi -q >"${scan_root}/gpu.before.txt"
lscpu >"${scan_root}/cpu.txt"
"${kt_python}" "${scan_dir}/ktransformers_p512_runner.py" --prepare-placement \
  --cold-plan "${cold_plan}" --frequency-plan "${scan_root}/placement.pt" >"${scan_root}/placement_prepare.log" 2>&1
"${kt_python}" - <<'PY' >"${scan_root}/build_preflight.log" 2>&1
import time
from sglang.jit_kernel.deepseek_v4 import _jit_torch_cublas_bf16_fp32
started = time.perf_counter()
_jit_torch_cublas_bf16_fp32()
print(f'linear_bf16_fp32 build/load preflight complete in {time.perf_counter()-started:.3f}s; no model requests')
PY
printf 'sample\tqps\treturncode\n' >"${scan_root}/status.tsv"
for sample in 1 2; do
  if (( sample == 1 )); then scan_rates=(4 8 10 20 30); else scan_rates=(30 20 10 8 4); fi
  for rate in "${scan_rates[@]}"; do
    case_dir="${scan_root}/qps${rate}/sample${sample}"
    mkdir -p "${case_dir}"
    case_started="$(date -u '+%Y-%m-%d %H:%M:%S')"
    echo "CASE_BEGIN sample=${sample} qps=${rate} time=${case_started}"
    nvidia-smi --query-gpu=timestamp,memory.used,memory.free --format=csv >"${case_dir}/gpu.before.csv"
    if /usr/bin/time -v -o "${case_dir}/resource_time.txt" timeout --signal=TERM --kill-after=20s 1200s \
      "${kt_python}" "${scan_dir}/ktransformers_p512_runner.py" \
      --model "${model}" --fixture "${fixture}" --frequency-plan "${scan_root}/placement.pt" \
      --qps "${rate}" --output "${case_dir}/result.json" >"${case_dir}/run.log" 2>&1; then
      case_rc=0
    else
      case_rc=$?
    fi
    printf '%s\n' "${case_rc}" >"${case_dir}/returncode.txt"
    printf '%s\t%s\t%s\n' "${sample}" "${rate}" "${case_rc}" >>"${scan_root}/status.tsv"
    nvidia-smi --query-gpu=timestamp,memory.used,memory.free --format=csv >"${case_dir}/gpu.after.csv"
    sudo journalctl -k --since "${case_started} UTC" --no-pager >"${case_dir}/kernel.log"
    if (( case_rc != 0 )); then
      tail -n 80 "${case_dir}/run.log" >&2
      echo "STOPPED sample=${sample} qps=${rate} rc=${case_rc}" >&2
      exit "${case_rc}"
    fi
    "${kt_python}" "${scan_dir}/summarize_ktransformers_p512_scan.py" "${scan_root}" --verify-case "${case_dir}"
    echo "CASE_END sample=${sample} qps=${rate} rc=0"
  done
done
"${kt_python}" "${scan_dir}/summarize_ktransformers_p512_scan.py" "${scan_root}"
nvidia-smi -q >"${scan_root}/gpu.after.txt"
echo "SCAN_COMPLETED=${scan_root}"
