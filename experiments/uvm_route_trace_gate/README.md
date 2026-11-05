# UVM route trace production gate

High-frequency `phase_f_route_log` records are disabled by default through the
writable module parameter:

```text
/sys/module/nvidia_uvm/parameters/uvm_phase_f_route_log_enable
```

The header-level gate prevents disabled calls from evaluating format
arguments. The emit function checks the parameter again before timestamping,
formatting, or taking the route-log mutex. Two explicit policy fallback events
use the low-frequency always-on path; allocated-ioctl allocation and
`copy_{from,to}_user` failures remain on `UVM_ERR_PRINT`.

## Static verification

```bash
python3 experiments/uvm_route_trace_gate/verify_static.py
```

## What the GPU A/B isolates

[`run_gpu_abba.py`](./run_gpu_abba.py) uses the existing strict 576 MiB,
1,152-entry fixed-bank probe. Every arm runs in a fresh process and performs
the same mode-21 restore followed by the same mode-20 transition. The primary
measurements are the two wrapper-reported `ioctl_wall_ns` values, so dataset
creation, checksum reads, and process teardown are outside the timing window.

The same loaded module is used for every arm. The controller changes only
`uvm_phase_f_route_log_enable`, in symmetric `OFF → ON → ON → OFF` order for
three cycles. It rejects an arm unless bytes, entry count, PTE mode, copy
pushes, VA stability, checksums, direct-cutover mode, and module srcversion all
remain fixed. This isolates the cost of timestamp/format/mutex route records;
it does not mix in a different CE path or a different kernel build.

For an OFF arm, deltas for `debug_emit_calls`, `timestamp_calls`,
`format_calls`, `mutex_acquires`, `records`, and `bytes` must all be zero. For
an ON arm, every debug emit must correspond to exactly one timestamp, format,
mutex acquisition, and record. `always_emit_calls` must stay zero because this
success probe must not take a fallback path.

The controller restores the original route gate in a `finally` block and
generates `summary.json` plus `REPORT.md`. It does not load or unload modules.

## Dry-run validation

This prints the complete ABBA command sequence without reading sysfs or
accessing the GPU:

```bash
python3 experiments/uvm_route_trace_gate/run_gpu_abba.py \
  --expected-srcversion DRY_RUN \
  --cycles 1 \
  --output-dir /tmp/uvm-route-trace-dry-run \
  --dry-run
```

## Minimal module load and formal A/B

These commands are an operator runbook; they were not executed while preparing
the experiment. First record service state, require an idle GPU, and load the
newly built stage-2 module:

```bash
repo_root="$PWD"
custom_uvm="$repo_root/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko"
stock_uvm="$(modinfo -F filename nvidia-uvm)"
stage2_src="$(modinfo -F srcversion "$custom_uvm")"
stock_src="$(modinfo -F srcversion "$stock_uvm")"
node_exporter_was_active="$(systemctl is-active node-exporter || true)"
gpu_exporter_was_active="$(systemctl is-active gpu-exporter || true)"

test "$(nvidia-smi -i 0 --query-gpu=memory.used \
  --format=csv,noheader,nounits | tr -d ' ')" = 0
if sudo fuser -v /dev/nvidia-uvm; then
  echo "nvidia-uvm still has a live client" >&2
  exit 1
fi
sudo systemctl stop node-exporter gpu-exporter
sudo modprobe -r nvidia_uvm
sudo insmod "$custom_uvm" \
  uvm_moe_slot_pte_remap_enable=1 \
  uvm_moe_slot_transition_fast_index=1 \
  uvm_moe_slot_transition_direct_cutover=1 \
  uvm_moe_slot_pool_vidmem_budget_bytes=1073741824

test "$(tr -d '[:space:]' < /sys/module/nvidia_uvm/srcversion)" = "$stage2_src"
test "$(cat /sys/module/nvidia_uvm/parameters/uvm_phase_f_route_log_enable)" = 0
test "$(cat /sys/module/nvidia_uvm/parameters/uvm_phase_f_route_log_delay_ms)" = 0
test -r /proc/driver/nvidia-uvm/phase_f_route_log_stats
```

Run one smoke cycle, then the formal three-cycle ABBA. Each output directory
must be new:

```bash
run_tag="$(date -u +%Y%m%dT%H%M%SZ)"
python3 experiments/uvm_route_trace_gate/run_gpu_abba.py \
  --expected-srcversion "$stage2_src" \
  --blocks 1 \
  --cycles 1 \
  --device 0 \
  --output-dir "experiments/uvm_route_trace_gate/artifacts/${run_tag}_smoke"

run_tag="$(date -u +%Y%m%dT%H%M%SZ)"
python3 experiments/uvm_route_trace_gate/run_gpu_abba.py \
  --expected-srcversion "$stage2_src" \
  --blocks 288 \
  --cycles 3 \
  --device 0 \
  --output-dir "experiments/uvm_route_trace_gate/artifacts/${run_tag}_full"
```

The default worker interpreter is
`workloads/vllm/vllm/.venv/bin/python`; override it with `--python` only if the
vLLM CUDA environment has moved.

## Recovery to stock UVM

Do not force removal if `fuser` or `modprobe -r` reports a live client. Once the
worker has exited and GPU memory is zero:

```bash
echo 0 | sudo tee \
  /sys/module/nvidia_uvm/parameters/uvm_phase_f_route_log_enable
test "$(nvidia-smi -i 0 --query-gpu=memory.used \
  --format=csv,noheader,nounits | tr -d ' ')" = 0
if sudo fuser -v /dev/nvidia-uvm; then
  echo "nvidia-uvm still has a live client" >&2
  exit 1
fi
sudo modprobe -r nvidia_uvm
sudo modprobe nvidia_uvm
test "$(tr -d '[:space:]' < /sys/module/nvidia_uvm/srcversion)" = "$stock_src"

if test "$node_exporter_was_active" = active; then
  sudo systemctl start node-exporter
fi
if test "$gpu_exporter_was_active" = active; then
  sudo systemctl start gpu-exporter
fi
test "$(nvidia-smi -i 0 --query-gpu=memory.used \
  --format=csv,noheader,nounits | tr -d ' ')" = 0
```
