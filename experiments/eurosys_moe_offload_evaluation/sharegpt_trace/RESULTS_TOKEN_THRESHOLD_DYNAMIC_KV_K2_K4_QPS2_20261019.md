# AdaptMoE K2 -> K4 at offered QPS=2

Date: 2026-10-19. Status: **128/128 full-trace requests succeed**, following
a separate 40/40 qualification. The valid pairing is K2/cold14 -> K4/cold28.
QPS=2 and the full dual-view mechanisms are retained. Three cold24
qualifications and the cold27 headroom control failed and were stopped;
incomplete replays are excluded from performance results. Stock UVM is
restored and GPU memory is 0 MiB.

## Full-trace result

Artifact relative to experiments/eurosys_moe_offload_evaluation:
`artifacts/token-threshold-dynamic-kv-k2-k4-cold28-qps2-full-20261019-attempt-01/`.

| Metric | Result |
|---|---:|
| Successful / failed requests | 128 / 0 |
| Output tokens | 24,254 |
| Offered QPS | 2.000000 |
| Achieved QPS | 0.828077 |
| Output throughput | 156.908 tok/s |
| Measurement window | 154.575 s |
| TTFT mean / p50 / p95 / p99 | 26,836.95 / 26,618.91 / 54,548.62 / 58,039.48 ms |
| TPOT mean / p50 / p95 / p99 | 85.02 / 85.23 / 93.66 / 99.45 ms |
| E2E mean / p50 / p95 / p99 | 42,753.56 / 44,281.43 / 87,658.17 / 91,734.13 ms |

All HTTP statuses are 200 and every output length matches the requested
length. There are no CUDA errors, illegal-memory accesses, Xids, MMU faults,
or driver allocation failures in the captured server/kernel logs. The server
emits a ProcessGroupNCCL destruction warning during intentional shutdown,
after all measured requests have finished; this is not a serving failure.
This is one full cold-start measurement, not repeated-run confidence data.

Exactly one cold14 -> cold28 transition occurs during measurement. It finishes
about 20 seconds into the formal window, and KV target 2730 commits at
scheduler step 257. No shrink back to K2 is observed. The transaction takes
423.82 ms; pre-growth detached-table preparation takes 132.50 ms, hot-expert
reconciliation 171.72 ms, and final non-nested refresh 148.85 ms. These
component times do not include all physical KV migration and control work.

The post-growth audit records all **4,293,918,720 B** of policy-visible KV on
GPU, with zero CPU/unknown bytes. All **22,649,241,600 B** of hot expert
weights are already GPU-resident, so no restorative hot prefetch is needed.
Free GPU memory is **292,290,560 B (278.75 MiB)**, above the unchanged 192 MiB
requirement. The full trace then completes without a Forward fault.

During the formal window, token-threshold access switching remains active:
93 PARK and 93 RESUME events total 1,973.73 ms of transition wall time. At the
10-second server-log sampling interval, KV usage reaches 93.4% and waiting
requests reach 48. Offered load exceeds achieved throughput, so queueing
remains substantial despite successful KV growth.

## Reference to the previous fixed-KV QPS=2 run

The older successful fixed run uses the same trace hash, arrival scaling,
C16, 8192-token bucket, thresholds, and eager execution:
`sharegpt_trace/artifacts/token-threshold-low-c16-qps2-adaptmoe-20261013/`.

| Configuration | QPS | TTFT mean (s) | TPOT mean (ms) | E2E mean (s) |
|---|---:|---:|---:|---:|
| Prior fixed 1.75 GiB KV / cold11 | 0.665297 | 45.746 | 62.60 | 57.132 |
| Current dynamic K2 -> K4 / cold14 -> cold28 | 0.828077 | 26.837 | 85.02 | 42.754 |

QPS increases 24.47%, mean TTFT decreases 41.33%, and mean E2E decreases
25.17%, while mean TPOT increases 35.81%. This is a system-configuration
comparison: both KV capacity and expert placement differ. It does not isolate
the controller's benefit from having more KV capacity, and no fixed-K4/cold28
control is measured in this task. Results at offered QPS=1 are not treated as
load-matched controls.

## Protocol

Qwen3-30B-A3B-FP8 on RTX 5090; the same 128-request frozen ShareGPT low trace
as the K2/K5 experiment, SHA-256
`416fe6e9bd464fec30259bdc375c462a1486e252b5a2555176452ab7253d98a2`.
`arrival_scale=0.13185410310639004` gives offered QPS=2 for the full trace.
The first-40 qualification prefix has offered QPS=1.727515 because the burst
sequence is not uniform. Four prefix warmup requests are outside measurement.

C16, 8192-token scheduler bucket, TRITON_ATTN, eager, no CUDA Graph.
Token-threshold hysteresis remains PARK <=16 and RESUME >=128. Immutable
CPU shadows, nested plan classes, physical KV migration, 512 KiB alias
granularity, and full dual-view/PTE switching are retained. The in-progress
K2/K5 run was explicitly stopped at the user's request before starting this
new configuration; it is not a performance sample.

| Setting | Value |
|---|---:|
| Initial policy KV blocks | 1364 |
| Maximum policy KV blocks | 2730 |
| Physical KV pool blocks | 2731 |
| Physical pool bytes | 4,295,491,584 |
| One growth step | 1366 blocks / 2,148,532,224 B |
| Prior K2/K5 growth | 2049 blocks / 3,222,798,336 B |
| Change in growth size | -33.33% |
| Initial / maximum expert tier | cold14 / cold28 |
| Main expert offload, initial | 3,170,893,824 B |
| Main expert offload, maximum | 6,341,787,648 B |
| Expert offload increment | 3,170,893,824 B |
| Extra maximum offload over failed cold24 | 905,969,664 B / 864 MiB |

For the failed cold14/cold24 pairing, expert release exceeds KV growth by
only 111 MiB; backing growth consumes another 90 MiB, leaving just 21 MiB
before additional page tables and runtime workspace. The valid cold28 pairing
releases 975 MiB more expert bytes than KV growth and adds 126 MiB of backing.
Neither byte calculation certifies physical headroom: the actual hot/KV
residency and free-memory audits are required.

The existing opt-in allocation-order protection is enabled:
`VLLM_UVM_KV_PREPARE_DUAL_BEFORE_GROW=1`. It prepares deferred detached children
before growing KV, then fully refreshes ordinary leaves after migration.
This is the same setting as the immediately preceding K2/K5 attempt; it does
not change the access-mode or KV-pressure policy. The valid rerun also enables
`VLLM_UVM_KV_RECONCILE_HOT_AFTER_GROW=1`, which audits/restores hot experts and
audits all active KV after growth. A successful resize alone is insufficient:
subsequent Forward steps must also finish without CUDA/UVM/Xid faults.

## Prepare-first qualification

Artifact relative to experiments/eurosys_moe_offload_evaluation:
`artifacts/token-threshold-dynamic-kv-k2-k4-cold24-qps2-prepare-dual-first-20261019-attempt-01/`.

Result: **incomplete, 8/40 requests**, exit code 2. The growth transaction itself
passes:

| Component | Result |
|---|---:|
| cold14 -> cold24 plan-class switch | 303.24 ms, successful |
| Pre-growth dual-view preparation | 89.81 ms, successful |
| Post-growth non-nested refresh | 126.50 ms, successful |
| KV target commit | 2730 blocks, scheduler step 256 |
| Next scheduled step | 16 tokens, fails |

`server.log` records `Failed to initialize the TMA descriptor 700` followed by
illegal memory access. The new kernel error for this run is:

```text
Xid 31, pid=243435: MMU Fault: ENGINE GRAPHICS GPC0 GPCCLIENT_T1_0
faulted @ 0x7e25_b2300000. FAULT_PDE ACCESS_TYPE_VIRT_READ
```

The fault address is 3,145,728 B into expert Parameter index 0 (the first
layer's w13 Parameter), corresponding to expert 1. This expert is hot under
both cold14 and cold24. The location is therefore outside the deliberately
remote cold set. This identifies an invalid mapping at the subsequent expert
read; it does not by itself prove which migration or PDE write invalidated it.
There is no page-directory allocation failure in this attempt's error ring.
Stock UVM was restored and `module.after.json` confirms GPU memory at 0 MiB.

## Original-order control

To distinguish a smaller growth step alone from the prepare-first change,
the same K2/K4 pair and arrival sequence are rerun with
`VLLM_UVM_KV_PREPARE_DUAL_BEFORE_GROW=0`.

Artifact:
`artifacts/token-threshold-dynamic-kv-k2-k4-cold24-qps2-original-order-20261019-attempt-01/`.

Result: **incomplete, 8/40 requests**, exit code 2. The expert switch succeeds
in 312.59 ms. Its cutover reports all 2,264,924,160 B of added cold weights
physically released and CPU-only, with zero unknown bytes. The post-KV nested
refresh then fails in 10.76 ms. Its error ring records:

```text
page_directory_alloc_failed stage=physical depth=5 page_size=4096 physical_bytes=4096 location=11 flags=0 sys_fallback=1 status=0x51
dual_prepare_alloc_failed child=gpu block=0x751514400000 nested=1 ready=0 selected=384 status=0x51
dual_prepare_batch_failed block=0x751514400000 index=460 candidates=3621 prepared=181 reused=279 skipped=0 nested=1 status=0x51
```

`0x51` is `NV_ERR_NO_MEMORY`, `location=11` is VID. This is a GPU physical
page-table allocation failure with non-evicting flags, not Python/host metadata
allocation. No free-root retry event occurs. This attempt does not commit the
2730-block target and fails before the next Forward. Both configurations
retain the same full dual-view mechanism; only the placement of table
preparation within the resize transaction changes. Cleanup again confirms
stock UVM and 0 MiB GPU memory.

## Post-growth hot-expert residency qualification

Artifact:
`artifacts/token-threshold-dynamic-kv-k2-k4-cold24-qps2-hot-reconcile-20261019-attempt-01/`.

The opt-in `VLLM_UVM_KV_RECONCILE_HOT_AFTER_GROW=1` checks and restores the
hot complement after physical growth, then checks every active KV block
before the final dual-view refresh. The restore preserves allocator reserves
and cold aliases. It fails closed if hot restore, minimum headroom, or active
KV residency is incomplete. Eleven focused worker tests pass, including
success and separate hot/KV audit failures.

Result: **incomplete, 8/40**, stopped before the next Forward, without a new
Xid or page-table allocation error in the captured logs.

| Post-growth audit | Bytes |
|---|---:|
| Expected hot-expert GPU residency | 23,555,211,264 |
| Actual GPU residency before repair | 23,341,301,760 |
| Missing GPU bytes | 213,909,504 (204 MiB) |
| Actual GPU residency after repair | 23,555,211,264 |
| Free GPU memory after repair | 2,883,584 (2.75 MiB) |
| Existing minimum free-memory requirement | 201,326,592 (192 MiB) |

This directly demonstrates that physical KV growth under the cold24 pairing
leaves hot expert pages without GPU backing. Manual migration mode 2 rejects
implicit data migration (while allowing mapping-only fault repair), so a
subsequent hot expert read cannot rely on an ordinary demand migration. The
repair restores hot residency but has insufficient headroom; it therefore
stops before the full active-KV audit or another Forward can pass. This does
not certify that active KV survived the repair, nor that every prior PDE fault
had this sole cause.

The following qualification retains cold14 at K2 and changes only the K4 plan to
cold27, adding 679,477,248 B (648 MiB) of main expert offload over cold24.
Cold27 is generated from the same frozen calibration and contains cold14.
Its main offload is 6,115,295,232 B (5.6953125 GiB). The original pool,
1364/2730 targets, QPS=2 trace, token thresholds, physical migration,
immutable CPU shadows, and dual-view/PTE access mechanism are preserved.

## cold27 headroom control

Artifact:
`artifacts/token-threshold-dynamic-kv-k2-k4-cold27-qps2-hot-reconcile-20261019-attempt-01/`.

Result: **incomplete, 8/40**, stopped before another Forward. Post-growth hot
residency is already exact: all 22,875,734,016 B are on the GPU, with zero
unknown bytes and no restorative prefetch required. The previous hot-residency
loss is absent. Free GPU memory is 151,781,376 B (144.75 MiB), below the
unchanged 192 MiB minimum, so the qualification is correctly rejected. The
next pairing adds one more expert per layer: K2/cold14 -> K4/cold28. Final main
expert offload is 6,341,787,648 B (5.90625 GiB), 864 MiB above cold24.

## Successful cold28 qualification

Artifact:
`artifacts/token-threshold-dynamic-kv-k2-k4-cold28-qps2-hot-reconcile-20261019-attempt-01/`.

**40/40 requests succeed**, exit code 0; no CUDA error, illegal memory access,
Xid, MMU fault, or driver allocation failure in captured logs. The KV target
commits to 2730 at scheduler step 256, and all subsequent Forward steps finish.
Post-growth hot experts are already GPU-resident (22,649,241,600 B, zero
unknown bytes); restorative hot prefetch is skipped. The full active-KV audit
passes before non-nested refresh and target commit. Free memory is
292,290,560 B (278.75 MiB), exceeding the unchanged 192 MiB minimum. Cleanup
restores stock UVM and 0 MiB GPU memory.

The cold14 -> cold28 transaction takes 422.19 ms; pre-growth detached-table
preparation takes 124.18 ms, hot reconciliation 158.59 ms, and post-growth
non-nested refresh 143.87 ms. These are different components, not the entire
KV migration wall time. This shortened prefix is a correctness qualification;
its measured QPS/TTFT are not substituted for the full-trace result.

## Error cause and repair

| K2 -> K4 variant | Completed | Failure boundary |
|---|---:|---|
| Original preparation order | 8/40 | After physical KV growth, 4 KiB GPU page-table allocation fails |
| Prepare before growing KV | 8/40 | Resize commits, then the next 16-token Forward faults on a hot expert address |
| cold24 + post-growth hot audit | 8/40 | 204 MiB hot GPU residency missing; restoration leaves only 2.75 MiB free |
| cold27 + post-growth hot audit | 8/40 | Hot residency exact, but 144.75 MiB free is below the unchanged 192 MiB gate |
| cold28 qualification | 40/40 | Hot/KV residency exact, 278.75 MiB free; subsequent Forward succeeds |
| cold28 full trace | 128/128 | Same audits pass; complete serving run succeeds |

Reducing the KV growth step alone did not resolve the cold24 failure. The
confirmed defects are insufficient physical headroom for non-evicting GPU
page-table allocation after growth, plus missing hot-expert GPU residency
after KV migration under the original pairing. Mode 2 cannot demand-migrate
those hot weights back on a later Forward. The precise PDE write involved in
the older fault is not separately traced, so residency loss is not presented
as proof that every earlier PDE failure had only one cause.

The repair prepares detached tables before KV consumes memory, reconciles
hot experts and all active KV at the quiescent boundary, refreshes both full
views after migration, and retains enough physical headroom with cold28.
It preserves the token scheduler, K2/K4 targets, QPS=2, physical KV migration,
immutable shadows, alias granularity, and full dual-view PARK/RESUME mechanism.
The extra four cold experts per layer are 3.125% of the 128-expert pool per
layer; K2 remains cold14. All failure metrics are excluded from the valid
full-trace result.

## Reproduction

```bash
PHASE_F_ROUTE_LOG_ENABLE=0 LIMIT=40 bash experiments/eurosys_moe_offload_evaluation/sharegpt_trace/run_k2_k4_cold28_qps2.sh /absolute/new/qualification-artifact
# Full measurement, same configuration:
PHASE_F_ROUTE_LOG_ENABLE=0 bash experiments/eurosys_moe_offload_evaluation/sharegpt_trace/run_k2_k4_cold28_qps2.sh /absolute/new/full-artifact
# Original-order control:
PHASE_F_ROUTE_LOG_ENABLE=0 LIMIT=40 VLLM_UVM_KV_PREPARE_DUAL_BEFORE_GROW=0 VLLM_UVM_KV_RECONCILE_HOT_AFTER_GROW=0 bash experiments/eurosys_moe_offload_evaluation/sharegpt_trace/run_k2_k4_qps2.sh /absolute/new/control-artifact
```

The final full measurement follows successful qualification and completes
128/128. The 8 completed requests from failed variants exclude those harmed
by failure and cannot supply unbiased TTFT/TPOT/QPS results.
