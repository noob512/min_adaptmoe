# Registered-plan phase-1 probe

This directory contains the isolated probe for the immutable fixed-bank plan
interface. It does not start vLLM, load a kernel module, unload a module, or
write any module parameter.

## What the probe proves

For four independent managed datasets, the happy-path GPU worker performs this
exact sequence:

1. Reserve one fixed physical bank with one mode-8 call.
2. Register five immutable plans while the generation remains an arbitrary
   positive value `G` returned by the driver:
   - L0 initial mode-21 plan;
   - L0, L1, L2, and L3 mode-20 transition plans.
   The registration interval must report exactly
   `slot_plan_register_calls=5` and zero execute/destroy/legacy calls.
3. Execute `L0 → L1 → L2 → L3 → L0` with compact plan ioctls and require the
   generation sequence `G→G+1→G+2→G+3→G+4→G+5`. Every successful operation must add
   exactly one `slot_plan_execute_calls` and no legacy ioctl.
4. After the first L0 execute, submit a raw kernel ioctl using stale generation
   `G` while the live generation is `G+1`. The rejection must report generation
   `G+1`, leave every result/copy/map field zero, leave all slot-direct activity
   core counters unchanged, add exactly one `slot_plan_execute_calls`, preserve
   every managed VA, and preserve L0's checksum. This distinguishes a real
   kernel generation rejection from a userspace-only rejection.
5. Re-read every dataset VA and verify the active dataset checksum after every
   successful execute. The four layer signatures must be distinct, and the
   final L0 signature must equal the first L0 signature.
6. Destroy all five descriptor plans at generation `G+5`. The first four
   destroys retain `G+5`; the last reports registry generation `0` through the
   ABI while the module-lifetime generation allocator remains monotonic.
   Destroy is also required to leave the live L0 checksum and slot-direct
   counters unchanged. The destroy interval must report exactly
   `slot_plan_destroy_calls=5`.
7. Close the slot session. Mode 19 must release every physical lane and all
   logical bytes. The wrapper state must then contain zero owners, zero plans,
   no generation, and no open session.
8. The controller waits for the worker process and CUDA context to exit, then
   requires `nvidia-smi` to report exactly `0 MiB` on the selected GPU.

Across the complete lifecycle, the strict control-plane delta must be:

```text
slot_plan_register_calls      5
slot_plan_execute_calls       6  # five valid plus one stale
slot_plan_destroy_calls       5
slot_legacy_ioctl_calls       2  # mode 8 plus mode 19
slot_legacy_transition_calls  0
```

The worker uses the existing low-level wrappers and CUDA dataset shim. Raw
ioctls are confined to deliberate negative checks; they are necessary because
the public wrapper correctly rejects some invalid states before they can reach
the kernel gates being tested.

After the happy worker exits, the controller runs these independent one-block
workers. Every worker starts and ends at `0 MiB`, so one rejected or aborted
session cannot contaminate another:

- reject a second mode-21 bootstrap registration;
- reject mode-20 geometry mismatches in count, slot identity, bytes,
  slot offset, GPU UUID, and duplicate logical-layer identity;
- while any plan is alive, reject a raw legacy mode-20 ioctl, increment exactly
  one legacy and one legacy-transition control counter, and leave all core
  counters unchanged;
- destroy/close a session, recreate the same plan ID, require `G2` to be larger
  than the old session's final generation, and reject the old token before the
  configured core (the plan-ID/generation ABA check);
- register a target plan, release and reallocate every cached physical slot
  backing while the descriptor remains alive, then require execute to reject
  the stale cached backing generation even if pool index and allocator pointer
  are reused, before clear, copy, PTE, TLB, owner-cutover, or generation work;
- register a target plan, free and recreate its managed allocation, then require
  the old plan/invalidation epoch to be rejected during preflight before any
  copy push, PTE map, or PTE write. The kernel must preserve the ACTIVE generation/session and
  report it unpoisoned; the production userspace wrapper still fails closed by
  locally poisoning and aborting the session. Reusing the same VA does not make
  the old allocation identity valid.

Each rejected register adds exactly one `slot_plan_register_calls`; each raw
execute rejection adds exactly one `slot_plan_execute_calls`. Except for the
explicit legacy case, all legacy counters remain unchanged. Registration,
legacy exclusion, and stale/ABA rejection require zero delta for every
slot-direct core counter. Allocation-identity rejection deliberately enters the
configured core far enough to validate its cached allocation IDs: it reports
one error call plus four requested ranges / 2 MiB requested bytes, but zero
completed ranges, blocks, copy methods/pushes, PTE work, owner cutover, or TLB
invalidate. The raw kernel result keeps the generation unchanged and the
registry ACTIVE/unpoisoned; serving userspace remains conservative and aborts
after the rejection.

Backing-identity rejection likewise enters the configured core and reaches the
cached slot validator (`slot_direct_transition_fast_index_calls=1`), but must
still report zero clear, copy, PTE, TLB, or owner-cutover work and preserve the
generation. The module-lifetime backing generation makes this independent of
whether the allocator happens to reuse the same `uvm_mem_t` pointer.

## Preconditions

The operator must load the intended module and configure it before running the
probe. The script only reads and verifies:

- `uvm_moe_slot_transition_fast_index=1`;
- `uvm_moe_slot_transition_direct_cutover=1`;
- `uvm_moe_slot_plan_fast_validation=1`;
- `uvm_moe_slot_pool_vidmem_budget_bytes >= blocks × 2 MiB`;
- `/proc/driver/nvidia-uvm/latency_stats` exists;
- the selected GPU reports `0 MiB` before the worker starts.

For `--blocks 288`, the slot budget must be at least `603979776` bytes. The
default one-block smoke needs only 2 MiB of slot backing.

## Pure-userspace verification

This command imports no CUDA or vLLM module and does not access a GPU:

```bash
workloads/vllm/vllm/.venv/bin/python -m unittest -v \
  experiments/uvm_slot_registered_plan_probe/test_run_probe.py
```

The thirteen tests cover an arbitrary-positive-generation happy lifecycle, every
isolated negative worker, the monotonic ABA contract, allocation identity
and cached-backing rejection before copy/PTE work, exact legacy counters, and the original
simulated stale-side-effect, VA-mutation, and cleanup failures.

## GPU smoke and full-shape commands

One-block smoke:

```bash
workloads/vllm/vllm/.venv/bin/python \
  experiments/uvm_slot_registered_plan_probe/run_probe.py \
  --blocks 1 \
  --device 0 \
  --output experiments/uvm_slot_registered_plan_probe/artifacts/smoke.json
```

Full 576 MiB / 1,152-entry bank:

```bash
workloads/vllm/vllm/.venv/bin/python \
  experiments/uvm_slot_registered_plan_probe/run_probe.py \
  --blocks 288 \
  --device 0 \
  --output experiments/uvm_slot_registered_plan_probe/artifacts/full.json
```

The controller emits a failed JSON artifact as well as a non-zero exit code if
any invariant fails. The official result is valid only when all of these are
true:

- top-level `status == "pass"`;
- worker `execution_order == [0, 1, 2, 3, 0]`;
- worker `generation_start == G > 0` and `generation_end == G + 5`;
- worker `stale_generation.profile_delta` has zero core activity;
- worker `lifecycle_control_delta` equals the five-counter vector above;
- worker `zero_slots_after_close == true`;
- every entry in `negative_worker_results` has `status == "pass"`, zero core
  activity for its rejection interval, `zero_slots_after_close == true`, and a
  `0 MiB` post-worker observation;
- top-level `zero_mib_after_worker_exit == true`.

## Current readiness result

As of 2026-08-17, the current reliability build (`srcversion
54E75E286AC26D12EBCF4FA`) passed both the one-block smoke and the full 288-block
/ 576 MiB matrix. The full run advanced generation `1 -> 6`, passed all ten
isolated negative workers, left every worker at zero slots, and returned GPU 0
to `0 MiB`. Its artifact is
`artifacts/20260817T_reliable_full_288block_v2.json`.

Use the vLLM virtual-environment interpreter shown above. The system
`/usr/bin/python3` on this host does not provide torch and therefore fails
before entering any GPU or driver path.
