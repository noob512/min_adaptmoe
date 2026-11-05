# UVM original-VA static alias probe

This probe validates the cmd97–99 static-alias path against two slices of one
`cudaMallocManaged` allocation. Both original VAs map the same preallocated
VIDMEM slot: the upload VA is RW and the consumer VA is RO. A CUDA Graph
replays pinned H2D into the upload VA and reads through the consumer VA.
The slices are selected inside a complete 2 MiB-aligned UVM VA block so the
probe exercises the 64 KiB PTE path rather than silently falling back to 4 KiB.

The module-cycle wrapper refuses to unload UVM unless GPU memory is zero and
`/dev/nvidia-uvm` has no clients. It restores the stock module on exit.

```bash
experiments/uvm_static_alias_original_va/run_with_module_cycle.sh
```

The result includes VA stability, replay correctness/timing, zero custom alias
ioctls during replay, rejection of a direct child-process CPU load while
sealed, and restoration of the untouched CPU backing after Graph teardown and
alias destruction.

It also exercises the runtime mutation guards on the RO consumer alias. While
the session is sealed, all of the following must return `NV_ERR_IN_USE`:

- set and unset preferred location;
- set and unset accessed-by;
- enable and disable read duplication;
- migrate to GPU and migrate to CPU.

After `DESTROY`, the probe repeats the same operations. Every operation must
succeed, each policy set is paired with its inverse, migration ends on CPU, and
the original backing contents are checked again. The policy ioctl structures
have explicit size/offset ABI assertions in the probe.

Validated artifacts from 2026-08-20:

- `result_policy_runtime_20260820.json`: 64 KiB PTEs, 50 Graph replays;
- `result_policy_runtime_4k_20260820.json`: 4 KiB PTEs, 20 Graph replays.

Both runs rejected all eight operations with `rmStatus=23` while sealed,
accepted all eight with `rmStatus=0` after `DESTROY`, and reported
`remote_cpu_backing_restored=true`.

Post-failure-path regressions
`result_post_failpath_fix_64k_cpu_guard_20260820.json` and
`result_post_failpath_fix_4k_20260820.json` additionally show the sealed CPU
load child terminating with `SIGSEGV`; the parent remains healthy and cmd99
restores the original CPU value afterward.

## Dual-bank, multi-region topology

`topology_probe.py` exercises the intended prefill layout rather than a single
alias. Every 512 KiB lane has one physical VIDMEM slot, one RW upload VA, and
two RO original consumer VAs. Four logical layers alternate over two banks:

```text
pinned L0 -> upload B0 -> original consumer VA L0
pinned L1 -> upload B1 -> original consumer VA L1
pinned L2 -> upload B0 -> original consumer VA L2
pinned L3 -> upload B1 -> original consumer VA L3
```

The full Qwen geometry can be reproduced with:

```bash
PROBE=$PWD/experiments/uvm_static_alias_original_va/topology_probe.py \
REGIONS=1152 \
REPLAYS=5 \
SLOT_BUDGET_BYTES=1207959552 \
OUTPUT=$PWD/experiments/uvm_static_alias_original_va/result_full_1152_current_20260820.json \
experiments/uvm_static_alias_original_va/run_with_module_cycle.sh
```

The validated full-scale artifact contains 2,304 physical slots, 6,912
aliases, 1.125 GiB unique VIDMEM backing, and 3.375 GiB aliased VA coverage.
All six cmd97 batches succeeded. Five measured Graph replays each transferred
2.25 GiB H2D and performed a 2.25 GiB D2D consumer verification. GPU time was
50.263 ms/replay and Graph host submission was 12.01 us/replay. Replay issued
zero cmd97--99 ioctls. This is a topology/data-plane probe, not a full Qwen MoE
forward benchmark.

## Failure convergence

cmd97 publishes exact teardown metadata before changing a target GPU PTE. A
failed MAP is retained as a poisoned, destroy-only session. cmd99 restores a
normal CPU-backed remote GPU mapping even when a target was invalidated during
prepare but never received a slot PTE. It releases backing only after that
cleanup completes.

The directed probe makes the second target GPU-resident so cmd97 fails after
preparing the first, lower-VA target:

```bash
PROBE=$PWD/experiments/uvm_static_alias_original_va/failure_recovery_probe.py \
OUTPUT=$PWD/experiments/uvm_static_alias_original_va/result_failure_recovery_current_20260820.json \
experiments/uvm_static_alias_original_va/run_with_module_cycle.sh
```

The artifact reports a poisoned MAP with both teardown records retained,
successful cmd99 release of both aliases and both 512 KiB slots, stable
original VAs, restored contents, and an empty legacy userspace owner cache.

## What may be deferred

The fixed expert-to-slot plan removes all per-layer remapping from the prefill
hot path, but it does not make address-translation state optional:

| State | May wait until compute completes? | Reason |
|---|---:|---|
| GPU leaf PTE publish | No | The first consumer instruction needs it. |
| TLB invalidation/completion | No | A stale translation can read the old backing. |
| Exact VA/slot teardown ranges | No | A partial failure must never free backing behind a live PTE. |
| Session/slot lifetime ownership | No | It prevents migrate/evict/release while Graphs are live. |
| Statistics, trace, logical scheduler owner | Yes | They do not participate in address translation or lifetime safety. |

Thus cmd97/cmd98 run once before capture, CUDA Graph replay contains only H2D
and compute, and bookkeeping which is not needed for teardown can be updated
later. cmd99 requires `drain -> destroy Graph -> synchronize streams` first.
It can then restore CPU remote access for decode or release/rebuild banks for a
new KV-cache/expert budget.
