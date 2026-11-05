# AdaptMoE K2 -> K5, QPS=2: failure diagnosis and rerun

Date: 2026-10-19. Status: allocation failure identified; the allocation-order
repair passes the resize itself but fails the next Forward in cold29. Further
K2/K5 work was stopped when the user changed the requested experiment to K2/K4.
Incomplete replays are excluded from performance results.

## Unchanged serving protocol

Qwen3-30B-A3B-FP8; frozen ShareGPT low trace, SHA-256
`416fe6e9bd464fec30259bdc375c462a1486e252b5a2555176452ab7253d98a2`;
C16, 8192 scheduled-token bucket, TRITON_ATTN, eager, no CUDA Graph.
Token-threshold hysteresis remains PARK <=16 and RESUME >=128. Dynamic KV
remains 1364 ->3413 policy blocks with physical migration and the original
plan-class/immutable-shadow/full-dual-view protocol. Four trace-prefix warmup
requests are outside measurement. Alias records remain 512 KiB.

K2/cold14 offloads 3,170,893,824 B of main expert weights. K5/cold29 offloads
6,568,280,064 B, an increase of 3,397,386,240 B. KV grows by 3,222,798,336 B.
The expert increment exceeds KV growth by 174,587,904 B before slot backing
and page-table costs. Larger cold30/cold32/cold34 diagnostic variants had
also failed; they are not successful performance samples.

The full 128-request trace has offered QPS=2 with arrival_scale
`0.13185410310639004`. The 40-request prefix has an actual offered QPS of
1.727515 because it samples only the first portion of the same burst sequence.

## Direct allocation evidence

Artifact (relative to experiments/eurosys_moe_offload_evaluation):
`artifacts/token-threshold-dynamic-kv-k2-k5-cold29-qps2-diagnostics-20261019-attempt-04/`

Its `phase_f_route_log.txt` retains the exact failure:

```text
page_directory_alloc_failed stage=physical depth=5 page_size=4096 physical_bytes=4096 location=11 flags=0 sys_fallback=1 status=0x51
dual_prepare_alloc_failed child=cpu block=0x720f05a00000 nested=1 ready=0 selected=512 status=0x51
dual_prepare_batch_failed block=0x720f05a00000 index=419 candidates=4362 prepared=196 reused=223 skipped=0 nested=1 status=0x51
```

`location=11` is the VID aperture. The failure is an actual non-evicting GPU
page-table allocation, not a host metadata allocation. `status=0x51` is
`NV_ERR_NO_MEMORY`. Detached allocation uses `UVM_PMM_ALLOC_FLAGS_NONE` and
returns directly on failure. Ordinary UVM table allocation has a separate
retry outside the VA-block lock with eviction enabled. The configured sysmem
fallback is also conditional on the eviction flag and therefore is not used
here.

Increasing the number of cold experts alone does not establish that such a
non-evicting KERNEL allocation can succeed: PMM USER and KERNEL free lists are
distinct. Released expert memory can remain in USER chunks. This is a concrete
allocator limitation; the contribution of cached roots versus true physical
shortage is being tested, and has not yet been assumed.

The preceding route-log-enabled attempt03 completed 8/40 and failed in the
same seal. Its 64 KiB route ring was overwritten by teardown; attempt04 disabled
high-frequency logging so its three failure records remained available.

## Conservative allocation retry under qualification

`uvm_page_tree_alloc_table_detached()` first tries its original allocation.
On failure, it may return one completely free USER root chunk to PMA and retry.
The existing PMM free-root routine waits its trackers and does not evict live
pages. No active expert/KV mapping, residency policy, alias granularity or
PARK/RESUME protocol changes in this retry.

Driver builds and shell syntax checks pass. The actual cold29 qualification is
`artifacts/token-threshold-dynamic-kv-k2-k5-cold29-qps2-free-root-retry-20261019-attempt-06/`.
It failed at the same block index 419/4362, with 8/40 requests complete. No
`detached_table_retry` event occurred: there was no wholly free USER root to
return. Attempt05 is invalid because a running shell script was edited during
that attempt, causing Bash to fail before measurement.

The cold32 retry also completed only 8/40:
`artifacts/token-threshold-dynamic-kv-k2-k5-cold32-qps2-free-root-retry-20261019-attempt-02/`.
Its immutable cutover reports 4,076,863,488 B physically released and CPU-only,
but post-KV preparation fails at index 173/4809 allocating a GPU child. Again,
there is no reclaimable whole USER root. This rules out cached *whole* free
roots as the sole cause; fragmented subchunks and physical allocation pressure
remain relevant.

## Prepare page tables before KV growth

The opt-in `VLLM_UVM_KV_PREPARE_DUAL_BEFORE_GROW=1` first seals the deferred
children before physically growing KV. After the migration it performs an
explicit non-nested refresh, reusing those children while rebuilding ordinary
leaves from current residency. It retains immutable shadows, plan classes,
512 KiB aliases, physical KV migration, and the full dual-view PARK/RESUME
mechanism. Preparation failure prevents KV migration. Eight targeted worker
tests pass, including both failure boundaries; the older disabled path also
remains covered.

The cold29 qualification is
`artifacts/token-threshold-dynamic-kv-k2-k5-cold29-qps2-prepare-dual-first-20261019-attempt-01/`.
It demonstrates that allocation order matters:

- cold14 -> cold29 plan switch succeeds in 482.86 ms.
- Pre-growth preparation succeeds in 149.18 ms.
- Post-growth non-nested refresh succeeds in 157.39 ms.
- KV target 3413 is committed at scheduler step 251.

However, the next 16-token Forward fails with a new Xid31/PDE read fault at
`0x732fe6000000`, exactly the first expert Parameter's starting address. That
is a hot expert address in layer 0 under both plans. Only 8/40 requests finish.
Passing SEAL is therefore insufficient to claim a working repair or publish
performance. A cold32 qualification with the same ordering and additional
648 MiB expert headroom was stopped on the user's switch to K2/K4, so it has
no accepted performance result. Its captured kernel log also includes Xid31
for EngineCore PID 241422, so it cannot be treated as a passing qualification.
See `RESULTS_TOKEN_THRESHOLD_DYNAMIC_KV_K2_K4_QPS2_20261019.md` for the subsequent
smaller-step tests.

## Reproduction

```bash
# 40-request qualification
PHASE_F_ROUTE_LOG_ENABLE=0 LIMIT=40 K5_COLD_EXPERTS=32 bash experiments/eurosys_moe_offload_evaluation/sharegpt_trace/run_k2_k5_qps2.sh /absolute/new/artifact
# Full trace (only after qualification succeeds)
PHASE_F_ROUTE_LOG_ENABLE=0 K5_COLD_EXPERTS=32 bash experiments/eurosys_moe_offload_evaluation/sharegpt_trace/run_k2_k5_qps2.sh /absolute/new/full-artifact
```
