# AdaptMoE token-threshold + dynamic KV: K2/K3 result

Date: 2026-10-19

## Protocol

- Model: Qwen3-30B-A3B-FP8
- Trace: ShareGPT low, 128 requests, SHA-256 `416fe6e9bd464fec30259bdc375c462a1486e252b5a2555176452ab7253d98a2`
- Offered load: 1.0 req/s
- C16, `max_num_batched_tokens=8192`, eager mode, no CUDA Graph
- Token-threshold access mode: PARK at 16 scheduled tokens, RESUME at 128
- Dynamic KV: K2 (`1364` blocks, initial) -> K3 (`2047` blocks, maximum)
- Paired expert plans: cold14 -> cold19, nested by construction
- Force-sync migration: disabled in the successful run
- Warmup requests: 4

KV and expert increments are deliberately near-equal:

| transition | bytes |
|---|---:|
| KV K2 -> K3 | 1,074,266,112 |
| Expert cold14 -> cold19 | 1,132,738,560 |
| Difference | 58,472,448 (55.8 MiB) |

## Full result

Artifact: `artifacts/token-threshold-dynamic-kv-k2-k3-qps1-full-20261019-attempt-01/`

| metric | result |
|---|---:|
| requests | 128/128 |
| QPS | 0.744035 |
| TTFT mean | 8,810.77 ms |
| TTFT p50 / p95 / p99 | 9,361.68 / 20,847.86 / 23,034.63 ms |
| TPOT mean | 79.34 ms |
| TPOT p50 / p95 / p99 | 78.96 / 86.02 / 90.56 ms |
| E2E mean | 23,736.73 ms |
| E2E p50 / p95 / p99 | 22,333.89 / 53,897.40 / 60,826.02 ms |
| output throughput | 140.98 tok/s |

The run performed exactly one successful dynamic tier transition:

- `cold14 -> cold19`: 169.20 ms
- Added cold expert ranges: 480, `1,132,462,080 B`, all CPU-only after cutover
- Full dual-view refresh: 49.98 ms, `rm_status=0`, `alias_count=8550`, `backing_bytes=179,306,496`
- No CUDA/UVM error, Xid, MMU fault, or kernel warning during the run
- Final cleanup restored stock UVM and GPU memory to 0 MiB

## Qualification and failed variants

The first K2/K4 pairing (cold12/cold22) reached the expert plan switch but failed the subsequent dual-view seal with `NV_ERR_NO_MEMORY` (`rm_status=81`). Increasing resident headroom alone did not remove the failure. A force-synchronous K2/K4 retry made the seal succeed but then produced an Xid 31/PDE fault on the next fused-MoE forward, so it is not a valid serving configuration. Those runs are retained as diagnostic artifacts and are excluded from the result above.

## Baseline comparison

For reference, the earlier fixed-K2 token-threshold run used cold11 and reported QPS `0.623387`, TTFT mean `24,608.96 ms`, TPOT mean `62.31 ms`, and E2E mean `36,013.24 ms` (`artifacts/token-threshold-low-c16-qps1-adaptmoe-20261013/`). The K2/K3 run is +19.35% QPS, -64.20% TTFT mean, and -34.09% E2E mean, while TPOT is +27.34%.

This is an operational comparison, not an isolated proof of the dynamic-KV controller: the new run changes both the maximum KV capacity (2 GiB -> 3 GiB) and the paired expert plan (cold11 -> cold14/cold19). A controlled attribution experiment would need a fixed 3 GiB KV/cold19 run and a dynamic-KV run with identical initial/final plans.
