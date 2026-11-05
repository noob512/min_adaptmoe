#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="${DEVICE:-0}"
TOTAL_LAYERS="${TOTAL_LAYERS:-16}"
GROUP_SIZE="${GROUP_SIZE:-8}"
NUM_IN_GROUP="${NUM_IN_GROUP:-2}"
PREFETCH_STEP="${PREFETCH_STEP:-2}"
LAYER_MIB="${LAYER_MIB:-16}"
TENSORS="${TENSORS:-4}"
COMPUTE_US="${COMPUTE_US:-200}"
WARMUPS="${WARMUPS:-1}"
REPEATS="${REPEATS:-5}"
EQUIVALENT_RATIO="${EQUIVALENT_RATIO:-0.90}"
CUDA_ARCH="${CUDA_ARCH:-native}"
RESULT_FILE="${RESULT_FILE:-/tmp/vllm_group_offload_$(date +%Y%m%d_%H%M%S).txt}"

command -v nvcc >/dev/null
command -v nvidia-smi >/dev/null
nvidia-smi --id="${DEVICE}" \
  --query-gpu=name,driver_version,memory.used,memory.free \
  --format=csv,noheader

make -C "${SCRIPT_DIR}" CUDA_ARCH="${CUDA_ARCH}"
mkdir -p "$(dirname "${RESULT_FILE}")"
"${SCRIPT_DIR}/group_offload_microbench" \
  --device "${DEVICE}" \
  --total-layers "${TOTAL_LAYERS}" \
  --group-size "${GROUP_SIZE}" \
  --num-in-group "${NUM_IN_GROUP}" \
  --prefetch-step "${PREFETCH_STEP}" \
  --layer-mib "${LAYER_MIB}" \
  --tensors "${TENSORS}" \
  --compute-us "${COMPUTE_US}" \
  --warmups "${WARMUPS}" \
  --repeats "${REPEATS}" \
  --equivalent-ratio "${EQUIVALENT_RATIO}" | tee "${RESULT_FILE}"

selected=()
slots=()
initial=()
triggers=()
for ((layer = 0; layer < TOTAL_LAYERS; ++layer)); do
  if ((layer % GROUP_SIZE >= GROUP_SIZE - NUM_IN_GROUP)); then
    selected+=("${layer}")
  fi
done
for ((position = 0; position < ${#selected[@]}; ++position)); do
  slots+=("$((position % PREFETCH_STEP))")
  if ((position < PREFETCH_STEP)); then
    initial+=("${selected[position]}")
  fi
  target=$((position + PREFETCH_STEP))
  if ((target < ${#selected[@]})); then
    triggers+=("${selected[position]}->${selected[target]}")
  fi
done
join_by_comma() {
  local IFS=,
  echo "$*"
}
trigger_text="$(join_by_comma "${triggers[@]}")"
if ((${#triggers[@]} == 0)); then
  trigger_text=none
fi
expected_schedule="SCHEDULE selected_layers=$(join_by_comma "${selected[@]}") slots=$(join_by_comma "${slots[@]}") initial=$(join_by_comma "${initial[@]}") triggers=${trigger_text}"
grep -Fqx "${expected_schedule}" "${RESULT_FILE}"
grep -q '^RESULT backend=pinned .*functional_ok=1$' "${RESULT_FILE}"
grep -q '^RESULT backend=managed .*in_place_va=1 consumer_va_stable=1 .*functional_ok=1$' "${RESULT_FILE}"
grep -q '^RESULT backend=uva .*in_place_va=1 consumer_va_stable=1 .*functional_ok=1$' "${RESULT_FILE}"
grep -q '^COMPARISON ' "${RESULT_FILE}"
grep -q "^benchmark_result functional_success=1 selected_count=${#selected[@]} total_cases=3 managed_va_preserved=1 " "${RESULT_FILE}"

echo "RESULT_FILE=${RESULT_FILE}"
