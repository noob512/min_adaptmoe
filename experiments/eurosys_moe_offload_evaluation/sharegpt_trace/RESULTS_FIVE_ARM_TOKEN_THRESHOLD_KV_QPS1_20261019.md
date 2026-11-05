# ShareGPT QPS=1: AdaptMoE fixed/dynamic KV, UVA, UVM, and Native Copy

Date: 2026-10-19

## Protocol

- Model: Qwen3-30B-A3B-FP8.
- Trace: ShareGPT low, 128 requests, SHA-256
  `416fe6e9bd464fec30259bdc375c462a1486e252b5a2555176452ab7253d98a2`.
- Offered load: 1.0 request/s; open-loop arrival order and offsets are identical.
- C16, `max_num_batched_tokens=8192`, 3 GiB physical KV allocation.
- TRITON_ATTN, eager execution, CUDA Graph disabled.
- Four warmup requests are outside the formal trace window.
- AdaptMoE access threshold: PARK at <=16 scheduled tokens and RESUME at >=128.
- One independent cold process per valid arm. Results are single-run measurements,
  not confidence-interval estimates.

The two AdaptMoE policies use paired KV/expert tiers:

| Tier | Policy-visible KV | Expert plan | Expert bytes |
|---|---:|---|---:|
| K2 | 1364 blocks, 1.998 GiB | cold14 | 2.953 GiB |
| K3 | 2047 blocks, 2.999 GiB | cold19 | 4.008 GiB |

`fixed K3/cold19` starts through the driver-safe cold14 topology, switches once
during warmup, and remains at K3/cold19 for the entire formal trace. Directly
sealing cold19 did not qualify on the current driver/configuration. `dynamic K2->K3`
starts the formal trace at K2/cold14 and lets measured workload pressure trigger
the paired K3/cold19 transition.

Here `UVM` specifically means stock `cudaMallocManaged` demand paging with
prefetch, advice, and the custom policy disabled. `UVA` means vLLM pinned-host
direct remote reads. They are distinct mechanisms.

## Results

All five valid arms completed 128/128 requests and 24,254 output tokens.

| Mechanism | QPS | Output tok/s | Window (s) | TTFT mean (ms) | TPOT mean (ms) | E2E mean (ms) |
|---|---:|---:|---:|---:|---:|---:|
| AdaptMoE token-threshold, fixed K3/cold19 | 0.748346 | 141.800 | 171.044 | 8,469.75 | 78.33 | 23,238.64 |
| AdaptMoE token-threshold, dynamic K2->K3 | **0.750846** | **142.273** | **170.474** | **8,353.36** | 78.36 | **23,088.56** |
| Native UVA, 4.064 GiB actual offload | 0.590274 | 111.848 | 216.848 | 30,484.81 | 107.37 | 50,304.18 |
| Stock UVM demand paging | 0.089433 | 16.946 | 1,431.238 | 520,509.60 | 805.17 | 669,215.58 |
| Native Copy, 4.064 GiB actual offload | 0.609530 | 115.496 | 209.998 | 20,410.05 | 101.71 | 39,525.19 |

Tail latency:

| Mechanism | TTFT p50 / p95 (ms) | TPOT p50 / p95 (ms) | E2E p50 / p95 (ms) |
|---|---:|---:|---:|
| AdaptMoE fixed | 8,695.85 / 20,317.27 | 78.18 / 83.97 | 21,689.06 / 53,072.31 |
| AdaptMoE dynamic | 8,403.80 / 20,095.68 | 77.76 / 85.13 | 21,565.78 / 52,474.81 |
| Native UVA | 33,265.38 / 59,488.55 | 105.40 / 151.89 | 53,215.90 / 100,739.85 |
| Stock UVM | 505,928.28 / 1,081,546.90 | 782.47 / 1,078.65 | 648,646.82 / 1,279,617.73 |
| Native Copy | 20,303.41 / 42,974.47 | 101.29 / 107.09 | 40,102.18 / 87,066.12 |

## Interpretation

- Fixed and dynamic AdaptMoE are effectively tied in this single run. Dynamic
  is 0.33% higher in QPS, 1.37% lower in mean TTFT, 0.04% higher in mean TPOT,
  and 0.65% lower in mean E2E. These differences are too small for a one-run
  significance claim.
- Dynamic AdaptMoE is 27.20% higher-QPS than Native UVA and 23.18% higher-QPS
  than Native Copy. Its mean TTFT is lower by 72.60% and 59.07%, respectively;
  mean TPOT is lower by 27.01% and 22.96%.
- Native Copy is 3.26% higher-QPS than Native UVA. At offered QPS=1 both native
  arms are below offered load, so queue buildup amplifies the service-rate
  difference in TTFT and E2E.
- Stock UVM reaches only 0.0894 QPS. Dynamic AdaptMoE is 8.40x higher-QPS. This
  is expected because stock UVM demand-pages broad managed model allocations;
  it is not an equal-byte cold-expert transport microbenchmark.

## Mechanism audit

- Fixed AdaptMoE: one cold14->cold19 transaction took 161.53 ms and completed
  1.847 s before the formal trace. No class change occurred in the trace.
- Dynamic AdaptMoE: one cold14->cold19 transaction took 176.89 ms, 33.94 s
  after formal trace start. The associated full dual-view refresh took 46.74 ms.
- Formal-window token-threshold transitions were 72 PARK / 71 RESUME for fixed
  and 70 PARK / 70 RESUME for dynamic. Their aggregate transition wall time was
  1,139.07 ms and 1,055.93 ms, respectively.
- Native UVA and Native Copy selected the same 114 whole Parameters and exactly
  4,364,094,520 bytes (4.064 GiB), 1.39% above cold19's 4.008 GiB expert slices.
- All valid server logs are free of CUDA error, illegal-memory, Xid, MMU/PTE/PDE
  fault signatures. GPU memory returned to 0 MiB and stock UVM srcversion
  `85A79790636BBD99BA3E43B` was restored.

## Invalid attempts retained for diagnosis

- Direct fixed-cold19 seal failed before serving with `rm_status=86`. It is not
  included. The valid fixed arm uses the proven cold14 seal plus quiesced append.
- Publishing K3 during engine startup left only 2.75 MiB free for the required
  192 MiB hot-expert reconciliation gate. It failed before serving and is not
  included.
- The first Native Copy formal attempt completed 127/128 because request
  `sharegpt-0104` saw `ServerDisconnectedError` before receiving an HTTP status.
  The server had no CUDA/OOM/traceback error and continued serving other
  requests. The independent `native-copy-rerun02` completed 128/128 and is the
  only Native Copy result reported above.

## Artifacts

Valid runs are under:

`artifacts/five-arm-qps1-low-eager-20261019-attempt-01/`

- `adaptmoe-fixed-k3-cold19/`
- `adaptmoe-dynamic-k2-k3/`
- `native-uva/`
- `stock-uvm-demand/`
- `native-copy-rerun02/`

Each directory contains the frozen configuration, command, server log, and
request-level `replay.json`. Qualification and rejected attempts remain under
`sharegpt_trace/artifacts/` and are not mixed into the result table.

## Limitations

- This is one cold-start run per mechanism, so small differences, especially
  fixed versus dynamic AdaptMoE, require repeated ABBA runs before ranking.
- Fixed AdaptMoE uses longest-prompt warmup to complete its one-time K3 setup;
  dynamic and native arms use trace-prefix warmup. Warmup is excluded from all
  formal metrics, but it can still change cache state.
- Stock UVM's managed-allocation scope is broader than the exact cold-expert
  slices in AdaptMoE and the 114 whole Parameters in Native UVA/Copy. Treat its
  row as a system-level baseline, not a byte-matched transport comparison.
