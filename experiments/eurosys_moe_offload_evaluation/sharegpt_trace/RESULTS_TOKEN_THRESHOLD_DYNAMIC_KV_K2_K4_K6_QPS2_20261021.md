# AdaptMoE K2 -> K4 -> K6, offered QPS=2

Date: 2026-10-21. Status: **128/128 requests succeed with both K2->K4 and
K4->K6 committed**, after a separate 64/64 qualification. The valid pairing
is cold14/cold28/cold44 with the opt-in proactive trigger. The matched proactive
K2/K4 control also completes 128/128. The legacy-trigger full trace only used K4;
proactive cold40 was rejected on physical headroom and is excluded.

## Matched full-trace comparison

Both arms use the same actual requests, arrivals (offered QPS=2), C16,
8192-token bucket, TRITON_ATTN, eager execution, four trace-prefix warmups,
token thresholds, and the new 80%-watermark proactive trigger. Both finish
128/128 requests and 24,254 output tokens; all HTTP statuses and output
lengths pass. These are one independent cold process per arm, not repeated
samples or confidence intervals.

| Configuration | Achieved QPS | TTFT mean (s) | TPOT mean (ms) | E2E mean (s) |
|---|---:|---:|---:|---:|
| K2->K4 / cold14->cold28, proactive matched control | 0.916834 | 22.166 | 76.95 | 36.491 |
| K2->K4->K6 / cold14->cold28->cold44, proactive | 0.824467 | 26.639 | 87.02 | 42.796 |
| Three tiers vs matched control | -10.07% | +20.18% | +13.09% | +17.28% |

The two actual growth steps work, but adding K6 does not improve this
configuration's serving performance. At 10-second sampling, the K4 control
reaches 95.7% KV use; after K6 commits the three-tier arm peaks at 63.2%
(roughly 3.8 GiB in use). Both remain limited to 16 running requests and
both reach 48 waiting requests. The legacy-trigger run also completes on K4
without requesting K6. This supports limited additional capacity benefit
under this trace/C16 setting; it does not prove that K6 cannot help a workload
with more concurrently admitted KV demand.

The paired K6 class increases main expert offload from 5.90625 to 9.28125 GiB,
an additional 3.375 GiB / 57.14%. It adds a second synchronous resize and
increases cold-weight traffic in copy mode and the remote expert set. Both
arms execute 91 PARK and 91 RESUME operations in the formal window; their
aggregate transition wall times are 1,795.60 ms (K4) and 2,607.50 ms (K6).
These observations are consistent with paying more transport/control work
without enough additional useful KV concurrency. No profiler decomposition
was collected, so they do not quantitatively separate copy, remote access,
forward computation, and controller costs. Mean TPOT includes mixed-step
prefill and resize stalls; it is not an isolated decode-kernel latency.

Control artifact:
`artifacts/token-threshold-dynamic-kv-k2-k4-cold28-qps2-proactive-control-full-20261021-attempt-01/`.
Its K4 commit is step 189; its one cold14->cold28 transaction is 265.29 ms.
The measurement window is 139.611 s and output throughput is 173.726 tok/s.
No CUDA/UVM/Xid fault occurs. Both valid arms restore stock UVM and 0 MiB
GPU memory after measurement.

## Relation to the historical result in the request

The 2026-10-19 K2/K4 result was QPS 0.828077, TTFT 26.837 s, TPOT 85.02 ms,
and E2E 42.754 s. The new K6 run differs by -0.44%, -0.74%, +2.35%, and
+0.10%, respectively. This is effectively similar at single-run resolution,
but the growth trigger and date differ; it is not a matched K6-benefit test.
The same-session proactive control above is the relevant new comparison.

For context, today's legacy-trigger run with a K6 maximum only committed K4
and achieved QPS 0.918839, mean TTFT 21.247 s, TPOT 76.03 ms, E2E 35.479 s.
This reinforces the need for a current control rather than attributing every
difference from the historical 0.828077 result to the additional KV tier.

## Output and statistical limitations

All input trace hashes, request ordering, expected output lengths and total
output counts match. First output tokens agree for all 128 requests between
the matched arms, but complete generated token sequences agree for only
49/128. No bitwise output-equivalence claim is made. For context, today's
legacy-trigger K4-only execution and the older K4 execution agree completely
for only 37/128 despite identical first tokens for all 128. Batch schedules
differ across continuous-serving runs; the source of complete-sequence
divergence is not separately established here. The results are serving
performance measurements with fixed requested lengths, not a numerical or
semantic equivalence certification. Repeated paired runs are needed for
statistical confidence.

## Full-trace K2/K4/K6 result

Artifact relative to experiments/eurosys_moe_offload_evaluation:
`artifacts/token-threshold-dynamic-kv-k2-k4-k6-cold44-qps2-proactive-full-20261021-attempt-01/`.

| Metric | Result |
|---|---:|
| Successful / failed requests | 128 / 0 |
| HTTP errors / output-length mismatches | 0 / 0 |
| Output tokens | 24,254 |
| Offered / achieved QPS | 2.000000 / 0.824467 |
| Output throughput | 156.224 tok/s |
| Measurement window | 155.252 s |
| TTFT mean / p50 / p95 / p99 | 26,638.74 / 26,028.31 / 56,816.58 / 60,383.89 ms |
| TPOT mean / p50 / p95 / p99 | 87.02 / 86.29 / 104.93 / 113.39 ms |
| E2E mean / p50 / p95 / p99 | 42,796.39 / 43,360.66 / 90,407.83 / 92,801.01 ms |

No CUDA error, illegal memory access, Xid, MMU fault, or driver allocation
failure occurs in captured logs. The ProcessGroupNCCL warning occurs during
intentional shutdown after measurement. Stock UVM is restored and GPU memory
returns to 0 MiB. This is a single cold-start run and not a confidence estimate.

| Growth | Commit step | Expert class transaction | Pre-growth table preparation | Hot audit | Final full refresh | Free GPU memory after hot audit |
|---|---:|---:|---:|---:|---:|---:|
| K2->K4 | 186 | 262.57 ms | 69.95 ms | 155.52 ms | 137.04 ms | 260.75 MiB |
| K4->K6 | 486 | 321.66 ms | 77.46 ms | 167.86 ms | 208.90 ms | 366.75 MiB |

These are component times, not the complete physical KV growth stall. After
the first growth, all 4,293,918,720 B of active KV and 22,649,241,600 B of hot
experts are on GPU. After the second, all 6,440,878,080 B of active KV and
19,025,362,944 B of hot experts are on GPU. KV has zero CPU/unknown bytes;
hot restorative prefetch is skipped because residency is already complete.
Subsequent Forward finishes all requests. No reverse tier transition occurs.

## Protocol

Qwen3-30B-A3B-FP8, RTX 5090 32 GB, the same frozen 128-request ShareGPT low
trace, SHA-256 `416fe6e9bd464fec30259bdc375c462a1486e252b5a2555176452ab7253d98a2`.
Arrival scale `0.13185410310639004` retains full-trace offered QPS=2.
C16, 8192-token scheduler bucket, TRITON_ATTN, eager, no CUDA Graph, four
trace-prefix warmup requests outside measurement. PARK <=16 and RESUME >=128.

The existing synchronized tier transaction commits adjacent KV/expert pairs
separately; it does not skip K4 when a prediction requests K6. Each growth
prepares deferred detached page tables first, migrates KV physically, audits
and restores hot experts if necessary, audits every active KV block, then
refreshes the complete dual view before publishing capacity. The existing
192 MiB minimum headroom requirement is retained. Immutable CPU shadows,
512 KiB alias granularity, full dual-view PARK/RESUME, and expert selection
remain unchanged. The proactive variant below explicitly changes the pressure
trigger and is distinguished from the legacy-trigger control.

| Tier | Policy KV blocks | Policy KV bytes | Cold experts per layer | Main expert offload |
|---|---:|---:|---:|---:|
| K2 | 1364 | 2,145,386,496 | 14 | 3,170,893,824 B / 2.953125 GiB |
| K4 | 2730 | 4,293,918,720 | 28 | 6,341,787,648 B / 5.90625 GiB |
| K6 (current pairing) | 4095 | 6,440,878,080 | 44 | 9,965,666,304 B / 9.28125 GiB |
| K6 (rejected pairing) | 4095 | 6,440,878,080 | 40 | 9,059,696,640 B / 8.4375 GiB |

The maximum managed KV reservation is 4096 blocks / 6,442,450,944 B (6 GiB),
including the null block. The inactive tail is not all GPU-resident at K2.
K2->K4 migrates 1366 blocks / 2,148,532,224 B; K4->K6 migrates 1365 blocks /
2,146,959,360 B. All three expert plans are nested and generated from the same
frozen routing calibration. K2/cold14 and K4/cold28 are identical to the
previous successful two-tier experiment. K6 initially pairs cold40, adding
2.53125 GiB of main expert offload and 108 MiB slot backing over cold28.
The rejected physical-residency audit below requires changing K6 to cold44,
adding another 864 MiB of main offload and 36 MiB of backing. K2 and K4 stay
unchanged. Byte balance alone does not certify physical allocation headroom.

No synthetic request or higher concurrency is substituted to force the second
growth. The report must distinguish configured tiers from actually committed
tiers, and invalid attempts must not supply serving performance metrics.

## Validation

Fourteen initial focused worker/controller tests pass, including adjacent K2/K4/K6
transactions, rejection of direct tier skips, preservation of active KV, hot
residency repair and failure of incomplete KV audits. Launcher syntax and JSON
validation pass. The first GPU qualification uses the first 64 trace requests;
it is separate from the completed full 128-request measurements.

Qualification artifact relative to experiments/eurosys_moe_offload_evaluation:
`artifacts/token-threshold-dynamic-kv-k2-k4-k6-cold40-qps2-qualify-20261021-attempt-01/`.

The 64-request prefix has offered QPS=2.147276 with the same arrival scale;
the full trace remains exactly QPS=2. The prefix commits K4 at scheduler
step 275. All 4,293,918,720 B of active KV and all 22,649,241,600 B of hot
experts are GPU-resident. Free GPU memory is 292,290,560 B (278.75 MiB).
No K4->K6 pressure event is observed. Therefore this qualifies startup, K4
growth, and subsequent inference; it does not yet qualify physical K6 growth.
Prefix performance is not substituted for full-trace performance. Stock UVM
and 0 MiB GPU memory are restored before the next run.

Full natural-pressure artifact:
`artifacts/token-threshold-dynamic-kv-k2-k4-k6-cold40-qps2-full-20261021-attempt-01/`.

This full trace completes 128/128 with QPS 0.918839, mean TTFT 21,246.70 ms,
mean TPOT 76.03 ms, mean E2E 35,479.28 ms. It commits only K4 at scheduler
step 274. There is no cold28->cold40 event. This is **not** a K6 result even
though the maximum reservation is 6 GiB. At 10-second sampling, K4 KV use
reaches 92.1% and waiting requests reach 48. No CUDA/UVM/Xid errors occur.

## Active high-watermark trigger

The legacy controller returns `active_cohort_fits_current_target` for
non-urgent high-watermark pressure while requests run. It waits until backup
activation, preemption or predicted demand proves the current tier inadequate.
This explains why registering a K6 maximum does not ensure its use on this
trace. A queue can grow due to C16 and compute limits even when K4 fits the
currently admitted requests.

New opt-in environment setting:
`VLLM_UVM_KV_AUTO_SCALE_ACTIVE_HIGH_WATERMARK_GROW=1`. Its default is 0, so
existing protocols retain their behavior. When enabled, non-urgent pressure
at the existing 0.8 high watermark may grow while requests run, subject to
minimum residency, no in-flight execution, and the same scheduler-boundary
and tier-transaction checks. It sets neither a manual target nor an elapsed
time trigger and inserts no synthetic requests. The launcher records the
setting as `adapt_dynamic_active_high_watermark_grow`.

The primary three-tier experiment and its matched K2/K4 control both enable
this option, so a performance difference is not
silently attributed to K6 when growth timing also differs.

The full controller test module passes 57/57 after this change. Targeted
controller/worker checks pass 19/19, including the default legacy hold,
proactive growth at actual pressure, adjacent transactions, and deferral
while a model execution is in flight.

## Rejected proactive cold40 qualification

Artifact:
`artifacts/token-threshold-dynamic-kv-k2-k4-k6-cold40-qps2-proactive-qualify-20261021-attempt-01/`.

Incomplete 27/64, exit 2. K4 commits at step 187. Later cold28->cold40
succeeds (249.97 ms), detached-table preparation succeeds (59.90 ms), but
post-growth hot GPU residency is 19,709,034,496 B versus 19,931,332,608 B
required: **222,298,112 B (212 MiB) missing**. Hot restoration completes but
free GPU memory is only 2,883,584 B (2.75 MiB), below the unchanged 192 MiB
gate. The process stops before K6 can be published and before another
Forward. Partial metrics are excluded. Stock UVM and 0 MiB are restored.

This is the same physical headroom failure class observed in the prior K2/K4
work, not a failure to reach the second pressure trigger. The rerun changes
only K6 from cold40 to cold44 (+4 cold experts out of 128 per layer) while
retaining trace, QPS, KV targets, high watermark, and access/migration protocol.

## Reproduction

```bash
PHASE_F_ROUTE_LOG_ENABLE=0 LIMIT=64 bash experiments/eurosys_moe_offload_evaluation/sharegpt_trace/run_k2_k4_k6_proactive_cold44_qps2.sh /absolute/new/qualification
PHASE_F_ROUTE_LOG_ENABLE=0 bash experiments/eurosys_moe_offload_evaluation/sharegpt_trace/run_k2_k4_k6_proactive_cold44_qps2.sh /absolute/new/full
# Matched K2/K4 control with the same proactive trigger:
PHASE_F_ROUTE_LOG_ENABLE=0 ADAPT_DYNAMIC_ACTIVE_HIGH_WATERMARK_GROW=1 bash experiments/eurosys_moe_offload_evaluation/sharegpt_trace/run_k2_k4_cold28_qps2.sh /absolute/new/control
```

## Successful cold44 qualification

Artifact:
`artifacts/token-threshold-dynamic-kv-k2-k4-k6-cold44-qps2-proactive-qualify-20261021-attempt-01/`.

64/64 succeeds, exit 0, with both K2->K4 and K4->K6 committed. K6 commits
at scheduler step 490. Full active KV residency is exactly 6,440,878,080 B
GPU, zero CPU and zero unknown. All 19,025,362,944 B of hot expert weights
are already GPU-resident; no restorative prefetch is necessary. Free GPU
memory is 384,565,248 B (366.75 MiB), above the existing 192 MiB gate.
Subsequent Forward requests finish normally. No serving CUDA/UVM/Xid fault
is observed; stock UVM and 0 MiB GPU memory are restored.

This prefix validates the mechanism; its metrics are excluded from the
full-trace performance table. The prefix arrival scale is unchanged, so its
offered QPS is 2.147276 due to the original burst sequence; the full 128
requests remain offered QPS=2.000000.
