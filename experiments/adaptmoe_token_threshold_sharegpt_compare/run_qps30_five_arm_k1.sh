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
small_marker="/run/adaptmoe-five-k1-${BASHPID}.active"
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
  small_dropin="${small_dropin_dir}/90-adaptmoe-five-k1.conf"
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
for name in run_qps30_five_arm_k1.sh five_arm_k1_compare.py five_arm_k1_runner.py ktransformers_p512_runner.py qps30_k1_k2_compare.py run_compare.sh \
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
"${small_python}" "${small_dir}/five_arm_k1_compare.py" "${small_root}/inputs" --prepare >"${small_root}/inputs_prepare.log"
export COMPARE_RUNNER_OVERRIDE="${small_dir}/five_arm_k1_runner.py"
export COMPARE_VERIFY_OVERRIDE="${small_dir}/five_arm_k1_compare.py"
export ADAPT_FIXED_RUNTIME_SETTINGS_FILE="${small_root}/inputs/adaptive_fixed_1g_runtime_settings.json"
export ADAPT_FIXED1_RUNTIME_SETTINGS_FILE="${small_root}/inputs/adaptive_fixed_1g_runtime_settings.json"
export ADAPT_FIXED2_RUNTIME_SETTINGS_FILE="${small_root}/inputs/adaptive_fixed_2g_runtime_settings.json"
export ADAPT_DYNAMIC_RUNTIME_SETTINGS_FILE="${small_root}/inputs/adaptive_dynamic_runtime_settings.json"
export NATIVE_NAMES="${small_root}/inputs/native_cold8_near_equal_names.txt"
export NATIVE_EXPECTED_BYTES=3690987520
unset ADAPT_RUNTIME_SETTINGS_FILE
echo 'FIVE_ARM_BEGIN QPS30 P1024/N128 C128; native UVA/copy, AdaptMoE fixed1/dynamic1-2, KT fixed1; two samples each'
if env ALLOW_GPU_RUN=1 SAMPLES=2 ARRIVAL_QPS=30 \
  COMPARE_ARMS='native_uva native_copy adaptive_fixed_1g adaptive_dynamic' RUN_ROOT="${small_root}/vllm" \
  FIXTURE="${small_dir}/inputs/sharegpt_deepseek_p1024_n128_20261024.json" \
  INITIAL_REQUESTS=64 LATE_REQUESTS=64 LATE_AFTER_ENGINE_STEPS=4 OUTPUT_TOKENS=16 \
  MAX_MODEL_LEN=1088 MAX_NUM_BATCHED_TOKENS=8192 MAX_NUM_SEQS=128 \
  KV_CACHE_BYTES=1073741824 ADAPT_FIXED1_KV_CACHE_BYTES=2147483648 \
  ADAPT_FIXED2_KV_CACHE_BYTES=2147483648 ADAPT_DYNAMIC_KV_CACHE_BYTES=2147483648 \
  PARK_TOKEN_THRESHOLD=16 RESUME_TOKEN_THRESHOLD=128 NSYS_PROFILE=0 NATIVE_OFFLOAD_GB=4.0 \
  bash "${small_dir}/run_compare.sh" >"${small_root}/qps30.log" 2>&1; then
  echo 'VLLM_FOUR_ARMS_END rc=0'
else
  small_rc=$?
  tail -n 85 "${small_root}/qps30.log" >&2
  exit "${small_rc}"
fi
kt_python="${small_repo}/KTransformers/.venv/bin/python"
kt_model="${MODEL:?Set MODEL to the local DeepSeek-V2-Lite-Chat snapshot directory}"
kt_fixture="${small_dir}/inputs/sharegpt_deepseek_p1024_n128_20261024.json"
kt_plan="${small_repo}/experiments/eurosys_moe_offload_evaluation/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/calibration_assets/plans/cold_plan_8_per_layer.json"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=14 MKL_NUM_THREADS=14
export SGLANG_APPLY_CONFIG_BACKUP=none SGLANG_E3_SERIALIZE_SCHEDULER_TIME_STATS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="${small_repo}/KTransformers/.venv/bin:${PATH}"
mkdir "${small_root}/ktransformers"
git -C "${small_repo}/KTransformers" rev-parse HEAD >"${small_root}/ktransformers.commit"
git -C "${small_repo}/KTransformers/third_party/sglang" rev-parse HEAD >"${small_root}/sglang.commit"
git -C "${small_repo}/KTransformers/third_party/sglang" diff >"${small_root}/sglang.local.diff"
"${kt_python}" "${small_dir}/ktransformers_p512_runner.py" --prepare-placement \
  --cold-plan "${kt_plan}" --frequency-plan "${small_root}/ktransformers/placement.pt" \
  >"${small_root}/ktransformers/placement_prepare.log" 2>&1
"${kt_python}" - <<'PY' >"${small_root}/ktransformers/build_preflight.log" 2>&1
from sglang.jit_kernel.deepseek_v4 import _jit_torch_cublas_bf16_fp32
_jit_torch_cublas_bf16_fp32()
print('BF16/FP32 cuBLAS extension build/load complete; no model requests')
PY
printf 'sample\treturncode\n' >"${small_root}/ktransformers/status.tsv"
for kt_sample in 1 2; do
  kt_case="${small_root}/ktransformers/sample${kt_sample}"
  mkdir "${kt_case}"
  kt_started="$(date +%s)"
  echo "KT_CASE_BEGIN sample=${kt_sample} QPS30 P1024 C128 KV1 cold8"
  nvidia-smi --query-gpu=timestamp,memory.used,memory.free --format=csv >"${kt_case}/gpu.before.csv"
  if /usr/bin/time -v -o "${kt_case}/resource_time.txt" \
    timeout --signal=TERM --kill-after=20s 1200s taskset -c 0-13 \
    "${kt_python}" "${small_dir}/ktransformers_p512_runner.py" \
    --model "${kt_model}" --fixture "${kt_fixture}" \
    --frequency-plan "${small_root}/ktransformers/placement.pt" --qps 30 \
    --prompt-tokens 1024 --max-num-seqs 128 --max-model-len 1088 \
    --kv-budget-bytes 1073741824 --kv-effective-tokens 34480 \
    --output "${kt_case}/result.json" >"${kt_case}/run.log" 2>&1; then
    kt_rc=0
  else
    kt_rc=$?
  fi
  printf '%s\n' "${kt_rc}" >"${kt_case}/returncode.txt"
  printf '%s\t%s\n' "${kt_sample}" "${kt_rc}" >>"${small_root}/ktransformers/status.tsv"
  sudo journalctl -k --since "@${kt_started}" --no-pager -o short-unix >"${kt_case}/kernel.log"
  nvidia-smi --query-gpu=timestamp,memory.used,memory.free --format=csv >"${kt_case}/gpu.after.csv"
  if (( kt_rc != 0 )); then
    tail -n 90 "${kt_case}/run.log" >&2
    exit "${kt_rc}"
  fi
  "${small_python}" "${small_dir}/five_arm_k1_compare.py" "${kt_case}" --verify-kt-case
  echo "KT_CASE_END sample=${kt_sample} rc=0"
done
"${small_python}" "${small_dir}/five_arm_k1_compare.py" "${small_root}"
echo "FIVE_ARM_COMPLETED=${small_root}"
