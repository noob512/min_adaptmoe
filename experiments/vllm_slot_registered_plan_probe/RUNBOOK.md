# Registered-plan phase-1 GPU runbook

This runbook validates the immutable registered-plan path without changing the
experiment geometry. It runs, in order:

1. a one-block isolated smoke;
2. the full 288-block / 576 MiB isolated probe;
3. one trace-enabled `CS` vLLM diagnostic;
4. a trace-disabled `SCCS` baseline without init-bank/registered-plan;
5. a trace-disabled `CSSC` init-bank-only arm;
6. a trace-disabled `SCCS` init-bank/registered-plan arm;
7. a gated three-arm comparison which separates the two gains;
8. stock-UVM restoration and cleanup checks.

Run all command blocks below, in order, from the same Bash session. Do not run
another UVM experiment concurrently. Every fixed artifact directory must be
absent before starting; use a new suffix for a repeat instead of deleting an
earlier result.

The executable isolated probe remains in
`experiments/uvm_slot_registered_plan_probe`. This directory contains only the
vLLM comparison runbook.

## 1. Safety and static preflight

The commands assume GPU 0, the repository at the path below, and the stock UVM
installed for the current kernel. The expected stock `srcversion` and this
host's `uvm_disable_hmm=1` modprobe policy are pinned so recovery cannot
silently load a different module or parameter state.

```bash
set -Eeuo pipefail

REPO_ROOT=/home/ubuntu/nvidia-uvm-gpu
GPU_INDEX=0
CUSTOM_UVM="${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko"
STOCK_UVM="/lib/modules/$(uname -r)/kernel/drivers/video/nvidia-uvm.ko"
EXPECTED_STOCK_SRCVERSION=85A79790636BBD99BA3E43B
EXPECTED_STOCK_UVM_DISABLE_HMM=Y
SLOT_BUDGET_BYTES=603979776
STAGE1_RECOVERY_DONE=0

stage1_restore_environment() {
  local recovery_rc=0
  local loaded_srcversion=""
  local gpu_used_mib=""
  local compute_pids=""
  local residual_pids=""
  local attempt

  if [[ "${STAGE1_RECOVERY_DONE:-0}" == 1 ]]; then
    return 0
  fi

  for ((attempt = 0; attempt < 60; ++attempt)); do
    gpu_used_mib="$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
    compute_pids="$(nvidia-smi --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk '$1 ~ /^[0-9]+$/ {print $1}')"
    if [[ "${gpu_used_mib}" == 0 && -z "${compute_pids}" ]]; then
      break
    fi
    sleep 1
  done

  if [[ "${gpu_used_mib}" != 0 || -n "${compute_pids}" ]]; then
    echo "recovery: GPU is not clean; refusing unsafe UVM unload" >&2
    recovery_rc=1
  else
    loaded_srcversion="$(tr -d '[:space:]' \
      < /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)"
    if [[ "${loaded_srcversion}" != "${EXPECTED_STOCK_SRCVERSION}" ]]; then
      if [[ -n "${loaded_srcversion}" ]] && ! sudo modprobe -r nvidia_uvm; then
        echo "recovery: failed to unload custom nvidia_uvm" >&2
        recovery_rc=1
      fi
      if ! sudo modprobe nvidia; then
        echo "recovery: failed to load base nvidia module" >&2
        recovery_rc=1
      fi
      if ! sudo modprobe nvidia_uvm; then
        echo "recovery: failed to load stock nvidia_uvm" >&2
        recovery_rc=1
      fi
    fi
  fi

  if ! sudo systemctl start node-exporter.service gpu-exporter.service; then
    echo "recovery: failed to restart exporters" >&2
    recovery_rc=1
  fi

  loaded_srcversion="$(tr -d '[:space:]' \
    < /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)"
  gpu_used_mib="$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=memory.used \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
  compute_pids="$(nvidia-smi --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk '$1 ~ /^[0-9]+$/ {print $1}')"
  residual_pids="$(pgrep -f \
    '[b]enchmark_custom_offload.py|[u]vm_slot_registered_plan_probe/run_probe.py|nvidia-smi [d]mon' \
    || true)"

  [[ "${loaded_srcversion}" == "${EXPECTED_STOCK_SRCVERSION}" ]] \
    || recovery_rc=1
  [[ "$(readlink -f "$(modinfo -n nvidia_uvm 2>/dev/null || true)")" \
    == "$(readlink -f "${STOCK_UVM}")" ]] || recovery_rc=1
  [[ "$(tr -d '[:space:]' \
    < /sys/module/nvidia_uvm/parameters/uvm_disable_hmm 2>/dev/null || true)" \
    == "${EXPECTED_STOCK_UVM_DISABLE_HMM}" ]] || recovery_rc=1
  [[ "${gpu_used_mib}" == 0 && -z "${compute_pids}" \
    && -z "${residual_pids}" ]] || recovery_rc=1
  systemctl is-active --quiet node-exporter.service || recovery_rc=1
  systemctl is-active --quiet gpu-exporter.service || recovery_rc=1

  if (( recovery_rc == 0 )); then
    STAGE1_RECOVERY_DONE=1
    echo "recovery: stock UVM/exporters restored; GPU ${gpu_used_mib} MiB"
  else
    echo "recovery: final-state verification failed" >&2
  fi
  return "${recovery_rc}"
}

stage1_err_trap() {
  local command_rc="$1"
  local line_number="$2"
  echo "stage1 command failed at line ${line_number} (rc=${command_rc})" >&2
  return "${command_rc}"
}

stage1_exit_trap() {
  local command_rc=$?
  local recovery_rc
  trap - ERR EXIT INT TERM
  set +e
  stage1_restore_environment
  recovery_rc=$?
  if (( command_rc == 0 && recovery_rc != 0 )); then
    command_rc="${recovery_rc}"
  fi
  exit "${command_rc}"
}

cd "${REPO_ROOT}"
test -f "${CUSTOM_UVM}"
test -f "${STOCK_UVM}"
test "$(modinfo -F srcversion "${STOCK_UVM}")" = "${EXPECTED_STOCK_SRCVERSION}"
test "$(readlink -f "$(modinfo -n nvidia_uvm)")" = "$(readlink -f "${STOCK_UVM}")"
test "$(tr -d '[:space:]' < /sys/module/nvidia_uvm/srcversion)" = "${EXPECTED_STOCK_SRCVERSION}"
test "$(tr -d '[:space:]' \
  < /sys/module/nvidia_uvm/parameters/uvm_disable_hmm)" \
  = "${EXPECTED_STOCK_UVM_DISABLE_HMM}"

GPU_USED_MIB="$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=memory.used \
  --format=csv,noheader,nounits | tr -d '[:space:]')"
COMPUTE_PIDS="$(nvidia-smi --query-compute-apps=pid \
  --format=csv,noheader,nounits 2>/dev/null \
  | awk '$1 ~ /^[0-9]+$/ {print $1}')"
test "${GPU_USED_MIB}" = 0
test -z "${COMPUTE_PIDS}"

python3 -m unittest -v \
  experiments/uvm_slot_registered_plan_probe/test_run_probe.py
experiments/vllm_slot_vs_pinned_copy/run_compare.sh static-preflight

trap 'stage1_err_trap "$?" "$LINENO"' ERR
trap stage1_exit_trap EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
```

Do not proceed if the GPU is not exactly `0 MiB`, a compute PID is present, the
stock module provenance check fails, or either userspace preflight fails.

## 2. Load the custom UVM

Only `nvidia_uvm` is replaced. The base NVIDIA modules remain loaded. Stopping
the exporters prevents them from racing the module transition or sampling
latency counters during the experiment.

```bash
sudo systemctl stop gpu-exporter.service node-exporter.service

GPU_USED_MIB="$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=memory.used \
  --format=csv,noheader,nounits | tr -d '[:space:]')"
COMPUTE_PIDS="$(nvidia-smi --query-compute-apps=pid \
  --format=csv,noheader,nounits 2>/dev/null \
  | awk '$1 ~ /^[0-9]+$/ {print $1}')"
test "${GPU_USED_MIB}" = 0
test -z "${COMPUTE_PIDS}"

sudo modprobe -r nvidia_uvm
# Removing the last UVM user can also unload base nvidia on this host. A
# direct insmod does not resolve that dependency, so restore it explicitly.
sudo modprobe nvidia
sudo insmod "${CUSTOM_UVM}" \
  uvm_moe_slot_pte_remap_enable=1 \
  uvm_moe_slot_transition_fast_index=1 \
  uvm_moe_slot_transition_direct_cutover=1 \
  uvm_moe_slot_pool_vidmem_budget_bytes="${SLOT_BUDGET_BYTES}"

CUSTOM_SRCVERSION="$(modinfo -F srcversion "${CUSTOM_UVM}")"
LOADED_SRCVERSION="$(tr -d '[:space:]' < /sys/module/nvidia_uvm/srcversion)"
test -n "${CUSTOM_SRCVERSION}"
test "${LOADED_SRCVERSION}" = "${CUSTOM_SRCVERSION}"
test "$(tr -d '[:space:]' < \
  /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable)" = 1
test "$(tr -d '[:space:]' < \
  /sys/module/nvidia_uvm/parameters/uvm_moe_slot_transition_fast_index)" = 1
test "$(tr -d '[:space:]' < \
  /sys/module/nvidia_uvm/parameters/uvm_moe_slot_transition_direct_cutover)" = 1
LOADED_SLOT_BUDGET="$(tr -d '[:space:]' < \
  /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes)"
(( LOADED_SLOT_BUDGET >= SLOT_BUDGET_BYTES ))
test -r /proc/driver/nvidia-uvm/latency_stats

for COUNTER_NAME in \
  slot_plan_register_calls \
  slot_plan_execute_calls \
  slot_plan_destroy_calls \
  slot_legacy_ioctl_calls \
  slot_legacy_transition_calls \
  slot_session_abort_calls \
  slot_session_abort_successes \
  slot_session_abort_failures \
  slot_session_abort_released_slots \
  slot_session_abort_released_bytes \
  slot_session_abort_remaining_slots; do
  rg -q "^${COUNTER_NAME}[[:space:]]" \
    /proc/driver/nvidia-uvm/latency_stats
done
```

Any `ERR`, normal `EXIT`, `INT`, or `TERM` after preflight invokes recovery
automatically. It waits up to 60 seconds for GPU 0 to become idle, refuses an
unsafe module unload if a compute process remains, restores stock UVM, restarts
both exporters, and validates the final state. The explicit final section
disarms these traps only after the same checks pass.

## 3. Isolated registered-plan probes

The smoke and full probe each use a short-lived worker process. The controller
requires exact VA/checksum/generation/counter behavior and verifies that GPU
memory returns to `0 MiB` after the CUDA context exits.

### 3.1 One-block smoke

```bash
python3 experiments/uvm_slot_registered_plan_probe/run_probe.py \
  --blocks 1 \
  --device "${GPU_INDEX}" \
  --output experiments/uvm_slot_registered_plan_probe/artifacts/20260816T_phase1_registered_smoke_1block.json
```

### 3.2 Full 288-block / 576 MiB shape

```bash
python3 experiments/uvm_slot_registered_plan_probe/run_probe.py \
  --blocks 288 \
  --device "${GPU_INDEX}" \
  --output experiments/uvm_slot_registered_plan_probe/artifacts/20260816T_phase1_registered_full_288block.json
```

Both commands must exit zero and write top-level `status: pass`. For each
session, all five registrations must return the same nonzero generation `G`;
the five valid executions must advance `G -> G+1 -> ... -> G+5`. The lifecycle
must also show one deliberately stale execution, five destroys, zero legacy
mode-20 transitions, stable managed VAs/checksums, zero slots after close, and
`0 MiB` after worker exit. Because generation is monotonic for the loaded
module's lifetime, the full probe is expected to start above the smoke probe's
final generation rather than returning to `1`.

## 4. Trace-enabled vLLM diagnostic

This is a single `CS` correctness diagnostic, not a performance sample. Trace
records must prove one init-bank allocation, five plan registrations, four
compact executions, one shared nonzero registration generation `G`, the chain
`G -> G+1 -> ... -> G+4`, three direct cutovers, and zero legacy mode-20
transition ioctls. Its per-slot-process driver control delta must be exactly
`register=5`, `execute=4`, `destroy=5`, `legacy ioctl=2`, and
`legacy transition=0`. In particular, `destroy=5` proves explicit descriptor
destruction; VA-space teardown alone does not satisfy the verifier. The two
legacy entries are the mode-8 candidate-fd ioctls recorded by
`fd_candidate_ioctl_calls`. Reliable cleanup then uses one scoped session-abort
audit and must report `calls/successes/failures=1/1/0`, release exactly 1,152
slots / 603,979,776 bytes, and leave zero remaining slots.

```bash
env \
  ALLOW_GPU_RUN=1 \
  ORDER=CS \
  TRACE_ENABLE=1 \
  REQUIRE_ABBA=0 \
  SLOT_EXPECTED_MODE8_FD_CANDIDATE_IOCTL_CALLS=2 \
  SLOT_REUSE_BANK=1 \
  SLOT_PRECOMPUTED_LANES=1 \
  SLOT_LAYER_TRANSITION=1 \
  SLOT_INIT_BANK=1 \
  SLOT_REGISTERED_PLAN=1 \
  SLOT_SHARED_PUSH=1 \
  SLOT_64K_PTE=1 \
  SLOT_DIRECT_CUTOVER=1 \
  SLOT_REQUIRE_COMBINED=1 \
  GPU_INDEX="${GPU_INDEX}" \
  RUN_ROOT=experiments/vllm_slot_vs_pinned_copy/artifacts/20260816T_phase1_registered_trace_on_CS \
  experiments/vllm_slot_vs_pinned_copy/run_compare.sh run
```

The runner invokes the strict verifier automatically. Do not use this trace-on
result for the latency comparison below.

The three formal groups keep model, fixture, payload, 576 MiB bank, shared
push, 64 KiB PTE, direct cutover, CPU affinity, and copy arm unchanged. Only
the following two mechanism switches vary:

| group | init-bank | registered-plan | order |
|---|---:|---:|---|
| baseline | 0 | 0 | SCCS |
| init-bank-only | 1 | 0 | CSSC |
| init-bank + plan | 1 | 1 | SCCS |

## 5. Trace-disabled baseline ABBA

This arm keeps fixed-bank reuse, layer batching, shared push, 64 KiB PTEs, and
direct cutover enabled, but disables init-bank and registered plans. `SCCS` is
one valid ABBA ordering accepted by the runner gate. Every baseline slot
process must report the exact control delta `register=0`, `execute=0`,
`destroy=0`, `legacy ioctl=7`, and `legacy transition=3`. This is two actual
mode-8 candidate-fd entries, four layer copy/transition entries, and one final
mode-19 release.

```bash
env \
  ALLOW_GPU_RUN=1 \
  ORDER=SCCS \
  TRACE_ENABLE=0 \
  REQUIRE_ABBA=1 \
  PINNED_ANCHOR_MAX_DRIFT_PCT=3.0 \
  SLOT_EXPECTED_MODE8_FD_CANDIDATE_IOCTL_CALLS=2 \
  SLOT_REUSE_BANK=1 \
  SLOT_PRECOMPUTED_LANES=1 \
  SLOT_LAYER_TRANSITION=1 \
  SLOT_INIT_BANK=0 \
  SLOT_REGISTERED_PLAN=0 \
  SLOT_SHARED_PUSH=1 \
  SLOT_64K_PTE=1 \
  SLOT_DIRECT_CUTOVER=1 \
  SLOT_REQUIRE_COMBINED=1 \
  GPU_INDEX="${GPU_INDEX}" \
  RUN_ROOT=experiments/vllm_slot_vs_pinned_copy/artifacts/20260816T_phase1_baseline_trace_off_SCCS \
  experiments/vllm_slot_vs_pinned_copy/run_compare.sh run
```

## 6. Trace-disabled init-bank-only ABBA

This differs from the baseline only by enabling init-bank. Registered plans
remain disabled, so the slot process still uses the legacy layer transitions
and must report `register=0`, `execute=0`, `destroy=0`, `legacy ioctl=7`, and
`legacy transition=3`. The request-period mode-8 count changes from one to
zero, while the whole-process mode-8 count remains one. This arm isolates the
benefit of moving bank initialization out of the measured request. The
candidate-fd expectation is established independently by section 4 and is
recorded before each trace-off run; the verifier never infers it from the
driver's legacy total.

```bash
env \
  ALLOW_GPU_RUN=1 \
  ORDER=CSSC \
  TRACE_ENABLE=0 \
  REQUIRE_ABBA=1 \
  PINNED_ANCHOR_MAX_DRIFT_PCT=3.0 \
  SLOT_EXPECTED_MODE8_FD_CANDIDATE_IOCTL_CALLS=2 \
  SLOT_REUSE_BANK=1 \
  SLOT_PRECOMPUTED_LANES=1 \
  SLOT_LAYER_TRANSITION=1 \
  SLOT_INIT_BANK=1 \
  SLOT_REGISTERED_PLAN=0 \
  SLOT_SHARED_PUSH=1 \
  SLOT_64K_PTE=1 \
  SLOT_DIRECT_CUTOVER=1 \
  SLOT_REQUIRE_COMBINED=1 \
  GPU_INDEX="${GPU_INDEX}" \
  RUN_ROOT=experiments/vllm_slot_vs_pinned_copy/artifacts/20260816T_phase1_init_only_trace_off_CSSC \
  experiments/vllm_slot_vs_pinned_copy/run_compare.sh run
```

## 7. Trace-disabled init-bank/registered-plan ABBA

This differs from the init-bank-only arm only by enabling registered plans.
The pre-registered `SCCS` ordering matches the baseline ordering. Every
optimized slot process must report `register=5`, `execute=4`, `destroy=5`,
`legacy ioctl=2`, and `legacy transition=0`, followed by the exact scoped-abort
cleanup proof described in section 4. This arm, compared with section 6,
isolates descriptor registration/compact execution from init-bank placement.

```bash
env \
  ALLOW_GPU_RUN=1 \
  ORDER=SCCS \
  TRACE_ENABLE=0 \
  REQUIRE_ABBA=1 \
  PINNED_ANCHOR_MAX_DRIFT_PCT=3.0 \
  SLOT_EXPECTED_MODE8_FD_CANDIDATE_IOCTL_CALLS=2 \
  SLOT_REUSE_BANK=1 \
  SLOT_PRECOMPUTED_LANES=1 \
  SLOT_LAYER_TRANSITION=1 \
  SLOT_INIT_BANK=1 \
  SLOT_REGISTERED_PLAN=1 \
  SLOT_SHARED_PUSH=1 \
  SLOT_64K_PTE=1 \
  SLOT_DIRECT_CUTOVER=1 \
  SLOT_REQUIRE_COMBINED=1 \
  GPU_INDEX="${GPU_INDEX}" \
  RUN_ROOT=experiments/vllm_slot_vs_pinned_copy/artifacts/20260816T_phase1_init_plan_trace_off_SCCS \
  experiments/vllm_slot_vs_pinned_copy/run_compare.sh run
```

## 8. Compare the three trace-off summaries

Each run must independently report `status=pass`, equal output tokens, two copy
samples, and two slot samples. All three commands pre-register the same 3%
maximum pinned-anchor drift. The verifier refuses missing/different thresholds
or arm/order mismatches, then gates on the spread between the maximum and
minimum of all three pinned medians. It exits nonzero when that spread exceeds
3%, while still writing an explicit fail artifact.

On a passing anchor gate, the output separates:

- init-bank benefit: baseline slot median divided by init-bank-only slot median;
- registered-plan benefit: init-bank-only slot median divided by registered-plan
  slot median;
- combined benefit: baseline slot median divided by registered-plan slot median.

```bash
BASELINE_ROOT=experiments/vllm_slot_vs_pinned_copy/artifacts/20260816T_phase1_baseline_trace_off_SCCS
INIT_ONLY_ROOT=experiments/vllm_slot_vs_pinned_copy/artifacts/20260816T_phase1_init_only_trace_off_CSSC
OPTIMIZED_ROOT=experiments/vllm_slot_vs_pinned_copy/artifacts/20260816T_phase1_init_plan_trace_off_SCCS

workloads/vllm/vllm/.venv/bin/python \
  experiments/vllm_slot_vs_pinned_copy/verify_compare.py \
  --compare-stage1-arms \
  "${BASELINE_ROOT}/summary.json" \
  "${INIT_ONLY_ROOT}/summary.json" \
  "${OPTIMIZED_ROOT}/summary.json" \
  --anchor-output "${OPTIMIZED_ROOT}/stage1_three_arm_comparison.json"
```

Only an output with `status: pass` and `pinned_anchor_pass: true` is a valid
comparison. On `status: fail`, repeat all three trace-off groups with new
artifact suffixes before attributing either gain to the mechanism.

## 9. Restore stock UVM and verify cleanup

On success, call the same idempotent recovery function explicitly and disarm
the traps. On an earlier failure or signal, the installed `ERR`/`EXIT` traps run
this function automatically.

```bash
stage1_restore_environment
trap - ERR EXIT INT TERM
```
