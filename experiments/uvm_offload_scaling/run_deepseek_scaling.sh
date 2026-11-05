#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLLM_ROOT="$ROOT/workloads/vllm/vllm"
PY="$VLLM_ROOT/.venv/bin/python"
RUNNER="$VLLM_ROOT/uvm_test/run_vllm_online_hot_cold_kv10g_scale.sh"
PLAN_BUILDER="$VLLM_ROOT/uvm_test/build_hot_cold_expert_plan.py"
SUMMARIZER="$ROOT/experiments/uvm_offload_scaling/summarize_deepseek_scaling.py"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7}"
DATASET="${DATASET:-$ROOT/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
PILOT_TRACE="${PILOT_TRACE:-/tmp/deepseek_v2_lite_compare_20260804_192435/pilot/routing_trace.jsonl}"
RUN_ROOT="${RUN_ROOT:-/tmp/deepseek_offload_scaling_$(date +%Y%m%d_%H%M%S)}"
PHASE_NAMES="prepare,warm64,c64,c96,c128,c160"
PHASE_COUNTS="1,64,64,96,128,160"
PHASE_OUTPUTS="1,1,32,32,32,32"

mkdir -p "$RUN_ROOT/plans" "$RUN_ROOT/logs"
printf '%s\n' "$RUN_ROOT" > /tmp/deepseek_offload_scaling_latest

restore_service() {
  sudo systemctl start node-exporter.service >/dev/null 2>&1 || true
}
trap restore_service EXIT
sudo systemctl stop node-exporter.service || true

wait_clean_gpu() {
  local used
  for _ in $(seq 1 300); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used <= 1024 )); then
      return 0
    fi
    sleep 2
  done
  return 1
}

monitor_gpu() {
  local pid="$1" output="$2"
  printf 'timestamp,used_mib,total_mib,utilization_gpu\n' > "$output"
  while true; do
    local state
    state=$(ps -o stat= -p "$pid" 2>/dev/null | awk '{print $1}')
    if [[ -z "$state" || "$state" == Z* ]]; then
      break
    fi
    nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >> "$output" 2>/dev/null || true
    sleep 0.1
  done
  nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >> "$output" 2>/dev/null || true
}

build_plan() {
  local experts="$1" output="$2"
  "$PY" "$PLAN_BUILDER" \
    --routing-trace "$PILOT_TRACE" \
    --output "$output" \
    --num-layers 27 \
    --num-experts 64 \
    --first-moe-layer 1 \
    --cold-experts-per-layer "$experts" \
    --expert-est-bytes 17301504 \
    > "$output.log" 2>&1
}

run_case() {
  local name="$1" mode="$2" experts="$3" final_blocks="$4" schedule="$5"
  local case_root="$RUN_ROOT/$name"
  local plan="$RUN_ROOT/plans/k${experts}.json"
  mkdir -p "$case_root"
  cp "$plan" "$case_root/hot_cold_plan_k45.json"
  if ! wait_clean_gpu; then
    printf '90\n' > "$case_root/launcher_returncode.txt"
    return 90
  fi

  env \
    MODEL="$MODEL" DATASET="$DATASET" RUN_ROOT="$case_root" RUN_CASES="$mode" \
    REQUEST_COUNT=160 PROMPT_MIN_TOKENS=320 PROMPT_MAX_TOKENS=360 \
    OUTPUT_TOKENS=32 FIXTURE_OUTPUT_TOKENS=32 \
    LARGE_KV_BYTES=8589934592 INITIAL_KV_BLOCKS=512 FINAL_KV_BLOCKS="$final_blocks" \
    COLD_EXPERTS_PER_LAYER="$experts" EXPERT_EST_BYTES=17301504 \
    MODEL_NUM_LAYERS=27 MODEL_NUM_EXPERTS=64 FIRST_MOE_LAYER=1 \
    MAX_MODEL_LEN=512 MAX_NUM_SEQS=160 MAX_NUM_BATCHED_TOKENS=2048 \
    GPU_MEMORY_UTILIZATION=0.97 TIMEOUT_SEC=3600 \
    PHASE_NAMES="$PHASE_NAMES" PHASE_COUNTS="$PHASE_COUNTS" PHASE_OUTPUTS="$PHASE_OUTPUTS" \
    STATIC_PHASE_NAMES="$PHASE_NAMES" STATIC_PHASE_COUNTS="$PHASE_COUNTS" STATIC_PHASE_OUTPUTS="$PHASE_OUTPUTS" \
    MAINTENANCE_WAVES_PER_PHASE_LIST="$schedule" \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    bash "$RUNNER" > "$case_root/launcher.log" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" > "$case_root/pid"
  monitor_gpu "$pid" "$case_root/gpu.csv" &
  local monitor_pid=$!
  wait "$pid"
  local rc=$?
  wait "$monitor_pid" 2>/dev/null || true
  printf '%s\n' "$rc" > "$case_root/launcher_returncode.txt"
  return "$rc"
}

{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "run_root=$RUN_ROOT"
  echo "model=$MODEL"
  echo "dataset=$DATASET"
  echo "pilot_trace=$PILOT_TRACE"
  echo "current_repo_commit=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
} > "$RUN_ROOT/metadata.env"

build_plan 9 "$RUN_ROOT/plans/k9.json"
build_plan 16 "$RUN_ROOT/plans/k16.json"
build_plan 23 "$RUN_ROOT/plans/k23.json"

printf 'stage=fixed_4gb\n' >> "$RUN_ROOT/status.env"
run_case fixed_4gb static_custom 9 2528 "" || true
printf 'stage=fixed_7gb\n' >> "$RUN_ROOT/status.env"
run_case fixed_7gb static_custom 16 4096 "" || true
printf 'stage=fixed_10gb\n' >> "$RUN_ROOT/status.env"
run_case fixed_10gb static_custom 23 5664 "" || true
printf 'stage=adaptive_4_7_10gb\n' >> "$RUN_ROOT/status.env"
run_case adaptive_4_7_10gb adaptive 23 5664 "9,0,0,7,7,0" || true

"$PY" "$SUMMARIZER" "$RUN_ROOT" > "$RUN_ROOT/logs/summarizer.log" 2>&1 || true
printf 'stage=done\nfinished_at=%s\n' "$(date --iso-8601=seconds)" >> "$RUN_ROOT/status.env"
