# UVM prefetch diagnostics

This directory contains a driver-profile collector independent of vLLM. It is
intended to wrap a small migration benchmark before the full server is started.

## What is already available

All of these paths preserve the user-visible virtual address:

- `cudaMemPrefetchAsync`: exposed by `uvm_batch_prefetch` in
  `workloads/vllm/vllm/uvm_test/uvm_allocator.cpp`. Despite its name, this is a
  loop of CUDA calls on one stream, not one multi-range CUDA operation.
- `UVM_MIGRATE`: the standard UVM ioctl used by the custom Phase-F prefetch
  action. It migrates managed backing and updates mappings without changing the
  allocation's VA.
- `UVM_PHASE_F_BATCH_EXECUTOR` and `UVM_PHASE_F_BATCH_REGIONS`: up to 64 VA
  ranges or registered region IDs per ioctl. Ordinary CPU/GPU prefetch still
  loops over ranges and calls `uvm_api_migrate` once per entry. The specialized
  expert read-duplicate action has a real shared batch path.
- `cudaMemAdvise`/Phase-F `ADVISE_CPU`: policy hints only; they do not prove
  migration completion.

There is also a distinct, fault-driven kernel heuristic in
`uvm_perf_prefetch.c`. With the current defaults (`enable=1`, threshold 51%,
minimum one fault), it expands a replayable-fault migration to nearby pages
whose density crosses the bitmap-tree threshold. Its scope is one UVM VA block
(at most 2 MiB), it avoids thrashing pages, and it cannot proactively fetch the
next layer or the next VA block before the first fault. Changing these module
parameters requires a module reload because they are read-only after load. An
A/B test of this heuristic is therefore different from an explicit
`cudaMemPrefetchAsync` test and must use identical first-touch kernels and a
fresh process/module configuration.

The driver also exports cumulative migration profiles at
`/proc/driver/nvidia-uvm/latency_stats` and resets them through
`latency_stats_reset`. The collector reports the API split, VA-block work,
populate/copy/map/TLB stages, and copy-method granularity. Profile stages are
nested, so their times are shown as non-additive ratios rather than summed.

## Usage

First verify that the loaded module matches the workspace build. The `run`
command performs this check when `--module` is supplied.

```bash
python3 experiments/uvm_prefetch_diagnostics/uvm_profile.py run \
  --module kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko \
  --output /tmp/uvm-profile.json \
  -- ./small_uvm_prefetch_benchmark --regions 64 --region-kib 512
```

Root permission may be needed only for `--reset`; ordinary before/after deltas
do not require a reset. Because the counters are module-global, keep unrelated
UVM work idle during each case.

A range manifest can be scanned separately or attached to `run`:

```json
{
  "ranges": [
    {"base": "0x100000000", "length": 524288, "allocation_id": "arena-0"},
    {"base": "0x100080000", "length": 524288, "allocation_id": "arena-0"}
  ]
}
```

```bash
python3 experiments/uvm_prefetch_diagnostics/uvm_profile.py layout ranges.json
```

`allocation_id` matters: touching ranges may be merged only when they are known
to belong to the same managed allocation. Crossing two merely adjacent
`cudaMallocManaged` allocations is invalid.

## How to interpret the result

- `api_tracker_wait / api_total` is completion or queued-device-work exposure;
  `api_core / api_total` is host-side UVM work; `api_locks_lookup / api_total`
  isolates setup and lookup.
- `block_create`, `block_lock_migrate`, `block_make_resident`, `block_map`,
  `populate_*`, `make_resident_finish`, `map_*`, `unmap_*`, and
  `tlb_phys_invalidate` narrow down metadata and mapping overhead. They overlap
  with parent stages and must not be added together.
- `copy_push_memcopy.bytes / calls` is the CE copy-method size. The current UVM
  implementation emits a large method only when both sides of an entire VA
  block are physically contiguous; otherwise it can fall back to page-sized
  methods. A 4 KiB-like result is therefore a strong fragmentation proxy in a
  full-range migration, but sparse page masks can produce the same result.

The current kernel does **not** export actual success/failure counts for CPU
2 MiB, 64 KiB, and 4 KiB chunk allocations. The module parameter
`uvm_cpu_chunk_allocation_sizes` only says which sizes are allowed. Likewise, a
single managed arena proves virtual continuity, not CPU physical continuity.
Use repeated cases with equal total bytes but different allocation count,
range size, and range ordering, and correlate them with copy-method size,
VA-block calls, `block_create`, `populate_cpu`, and `/proc/buddyinfo`. Direct
proof would require adding per-size attempt/success/fallback counters in
`uvm_pmm_sysmem.c` and exporting them through `latency_stats`.

The standard UVM Tools interface can count HtoD/DtoH bytes and CPU/GPU faults;
its migration events also carry `UvmEventMigrationCausePrefetch`. The custom
`latency_stats` profile does not currently split ordinary fault-demand bytes
from fault-prefetch bytes, so cause-specific validation needs UVM Tools/Nsight
events or one additional cause counter in the driver.
