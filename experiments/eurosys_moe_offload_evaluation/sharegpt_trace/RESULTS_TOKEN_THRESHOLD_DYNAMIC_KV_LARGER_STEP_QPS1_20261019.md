# AdaptMoE dynamic KV larger-step experiment

Date: 2026-10-19

## Protocol

- Model: Qwen3-30B-A3B-FP8
- Trace: ShareGPT low, 128 requests, SHA-256
  `416fe6e9bd464fec30259bdc375c462a1486e252b5a2555176452ab7253d98a2`
- Offered load: 1.0 req/s, C16, `max_num_batched_tokens=8192`
- TRITON_ATTN, eager execution, CUDA Graph disabled
- Token-threshold access mode: PARK at 16 scheduled tokens and RESUME at 128
- Four trace-prefix warmup requests outside the measurement window
- One independent cold-start run per compared dynamic-KV configuration

The valid larger step keeps the same K2/cold14 initial tier as the previous
run and changes only the maximum paired tier:

| configuration | KV transition | KV increase | expert transition | expert increase |
|---|---:|---:|---|---:|
| Previous | 1364 -> 2047 blocks | 1,074,266,112 B | cold14 -> cold19 | 1,132,738,560 B |
| Larger step | 1364 -> 2217 blocks | 1,341,652,992 B | cold14 -> cold20 | 1,359,286,272 B |

The new KV increase is 24.89% larger. Its KV and expert increases differ by
only 17,633,280 B (16.82 MiB). The physical KV pool contains 2218 blocks;
2217 blocks are policy-visible.

## Result

Valid artifact:
`artifacts/token-threshold-dynamic-kv-k2-k3-25-qps1-full-20261019-attempt-01/`

| metric | Previous K2 -> K3 | Larger K2 -> K3.25 | change |
|---|---:|---:|---:|
| requests | 128/128 | 128/128 | equal |
| QPS | 0.750846 | 0.756581 | +0.76% |
| output throughput | 142.273 tok/s | 143.360 tok/s | +0.76% |
| TTFT mean | 8,353.36 ms | 6,543.00 ms | -21.67% |
| TTFT p50 | 8,403.80 ms | 6,274.17 ms | -25.34% |
| TTFT p95 | 20,095.68 ms | 17,159.70 ms | -14.61% |
| TPOT mean | 78.36 ms | 80.36 ms | +2.54% |
| TPOT p50 | 77.76 ms | 80.65 ms | +3.72% |
| TPOT p95 | 85.13 ms | 86.71 ms | +1.85% |
| E2E mean | 23,088.56 ms | 21,667.18 ms | -6.16% |
| E2E p50 | 21,565.78 ms | 19,746.03 ms | -8.44% |
| E2E p95 | 52,474.81 ms | 49,587.05 ms | -5.50% |
| measurement window | 170.474 s | 169.182 s | -0.76% |

The larger-step run performed exactly one successful growth at scheduler step
447. The previous run grew at step 460. Peak reported KV use fell from 99.5%
to 96.2%; both runs reached 16 running requests and 18 waiting requests.

The larger transaction itself cost more:

| transaction component | Previous K2 -> K3 | Larger K2 -> K3.25 |
|---|---:|---:|
| plan-class switch | 176.89 ms | 198.36 ms |
| deferred full dual-view refresh | 46.74 ms | 61.27 ms |
| recorded switch + refresh | 223.63 ms | 259.63 ms |

This metric split is consistent with extra KV capacity reducing queueing while
the larger expert topology and transition add work: TTFT/E2E improved but TPOT
regressed slightly. One run is insufficient for causal attribution. The run
completed without nonzero `rm_status`,
CUDA error, Xid, MMU/PTE/PDE fault, or poisoned session. Cleanup restored the
stock UVM module and GPU memory to 0 MiB.

## Safety boundary found during qualification

Two still-larger candidates were rejected before the full run:

- K2 -> K3.5 (1364 -> 2388 blocks, cold14 -> cold21) failed the post-resize
  dual-view seal with `rm_status=81`; the attempted topology had 9450 aliases
  and 198,180,864 B of backing.
- K2 -> K4 (1364 -> 2730 blocks, cold14 -> cold24) failed at the same point
  with `rm_status=81`; it had 10800 aliases and 226,492,416 B of backing.

The valid K2 -> K3.25 topology has 9000 aliases and 188,743,680 B of backing.
Thus the current safe boundary lies above cold20/9000 aliases but below
cold21/9450 aliases on this GPU and driver state. Force-synchronous migration
was not used because an earlier K2/K4 force-sync run moved the failure to the
next fused-MoE forward and caused an Xid 31/PDE fault.

## Interpretation limits

The larger step is better in this single full run for QPS, TTFT, and E2E, but
it is not yet a statistically established improvement: each configuration has
only one cold-start sample. All 128 requests completed with the same requested
and produced token counts. Greedy output token IDs were identical for 32/128
requests; continuous-batching changes can alter floating-point execution order
and therefore greedy token choices, so this experiment is a serving-performance
comparison rather than an output-hash equivalence test.

