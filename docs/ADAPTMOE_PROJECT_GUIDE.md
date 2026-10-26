# Paper-to-code guide

AdaptMoE separates two decisions: how the current step accesses offloaded
experts, and how much physical GPU memory is assigned to experts versus KV.
The first can change every scheduler step; the second changes discrete
capacity classes when pressure and safety conditions permit.

## Implementation map

Runtime paths below are relative to `workloads/vllm/vllm/`.

| Concept | Main implementation |
|---|---|
| Stable Parameter VA, CPU shadow, upload aliases, shared slot and plan registry | `vllm/model_executor/models/utils.py` |
| Access controller and worker-side capacity transitions | `vllm/v1/worker/gpu_model_runner.py`, `gpu_worker.py` |
| Driver bindings, session lifecycle and ioctl ABI | `vllm/device_allocator/uvm_static_alias.py`, `uvm.py` |
| Capacity controller and pending grow/shrink | `vllm/v1/engine/core.py` |
| KV free/backup/offline queues and physical ranges | `vllm/v1/core/block_pool.py`, `uvm_kv_runtime_policy.py` |
| Mixed scheduling and optional phase cohorts | `vllm/v1/core/sched/scheduler.py` |
| Dual views, backing ownership and publication | `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c`, `uvm_mmu.c` (relative to repository root) |
| Driver transactions | `uvm_ioctl.h` in that driver directory, and `uvm_static_alias.py` |

## Stable addresses and the access controller

In the copy view, a writable upload alias and stable consumer Parameter VA
refer to reusable GPU backing. Explicit H2D copies fill the slot before the
fused MoE operator reads it. Reuse across layers requires dependencies between
copy, compute and the next overwrite. In the remote view, the same consumer VA
reads CPU-backed weights over PCIe. Remote access uses prepared mappings; it
differs from stock UVM demand paging and fault-driven migration.

PARK selects the remote view; RESUME selects the copy view. Neither operation
changes KV capacity or frees the shared copy backing. The driver prepares two
sets of 4 KiB leaf mappings and publishes the selected parent PDE per 2 MiB
VA block. Dependency waits, permission checks and TLB invalidation remain
part of the operation.

The mixed mode is `token_threshold`. Its entry point is
`_maybe_switch_uvm_original_va_prefill_to_decode_remote()` in the GPU runner.
It reads `scheduler_output.total_num_scheduled_tokens`, including both prefill
and decode tokens. One access view is selected for the whole step.

| State | Scheduled tokens | Action in the documented experiment |
|---|---:|---|
| Copy (`sealed`) | ≤16 | PARK |
| Remote (`remote_parked`) | ≥128 | RESUME |
| Either | Between the thresholds | Keep the current view |

Offered QPS, request count and the configured token budget are not the input
to this decision. Transition checks validate lifecycle, session and backing
state; unrecoverable inconsistencies poison the session and stop execution.
Performance configurations disable repeated full residency diagnostics while
retaining required synchronization and transition validation.

## Capacity controller

KV reserves a maximum stable VA range and varies the allocatable GPU-resident
prefix. The current implementation uses managed allocations; inactive tails
can have CPU backing. They are not universally physically unbacked ranges.
The block pool tracks GPU-free, GPU-backup and CPU-offline queues in addition
to active blocks held by requests. A shrink cannot reclaim active tail blocks.

The engine uses demand estimates, occupancy, backup consumption and preemption
signals. A safe boundary commits a pending transition: growth first moves to
a plan with more cold experts, then grows KV; shrink first reclaims free KV
tail blocks, then restores experts. Class validation checks nested expert
sets, slot geometry, source identity and memory budgets. Bounded delta
batches, mapping refresh and post-transition residency checks protect the
cutover. Transaction/rollback paths have limits: arbitrary OOM recovery is
not guaranteed.

For the retained DeepSeek-V2-Lite-Chat BF16 experiment:

| Class | Usable KV blocks | KV tokens | Cold experts per MoE layer | Main expert bytes offloaded |
|---|---:|---:|---:|---:|
| K1 | 2,155 | 34,480 | 8 | 3,598,712,832 |
| K2 | 4,313 | 69,008 | 11 | 4,948,230,144 |

There are 26 MoE layers. KV uses 16-token blocks, 497,664 model-wide bytes per
block, aligned residency units and a reserved null block. Both AdaptMoE arms
reserve the same K2 maximum VA. The fixed arm exposes only K1; the dynamic arm
can change between K1 and K2. The generated settings retain an 80% high
watermark, four-step minimum dwell and three-step low-pressure confirmation.
An enabled policy does not imply that a run actually resized; inspect its
events and observed capacity.

Native UVA and Native Copy use the same seven whole Parameters, totaling
3,690,987,520 bytes, 2.56% more than cold8. Their usable KV capacity is aligned
with K1. This is a system configuration comparison with a documented offload
granularity difference.

## Graph and evaluation scope

| Runtime mode | Mixed steps | CUDA Graph |
|---|---|---|
| `token_threshold` | Supported | Must be `NONE` |
| `prefill_graph_remote_decode` | Rejected | `FULL_SEPARATE` |
| `prefill_remote_decode` | Rejected | Eager phase path |

The runtime validator rejects mixed `token_threshold` plus CUDA Graph. Stable
addresses alone do not resolve attention metadata, token bucket, batch layout
and copy-node dependency constraints. The phase-separated Graph code is
retained for inspection and focused contract tests; the canonical experiment
does not measure it.

The [reproduction guide](REPRODUCING_ADAPTMOE.md) defines what the retained
runner measures. Dynamic KV can reduce queueing at high pressure while extra
offloaded experts and larger batches increase token latency. Improvements
must be stated per metric and workload. Request completion, absence of GPU
faults and numerical equivalence are separate checks.
