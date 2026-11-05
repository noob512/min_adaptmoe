# ShareGPT mixed-scheduler pressure scan (2026-09-15)

> **Correction (2026-09-15):** the original `512 / 32, PARK=16` Adaptive
> result was contaminated by the large-Prefill residency diagnostic inherited
> from the frozen E2 environment. A uniform three-load rerun with `PARK=16`,
> that diagnostic disabled, and two independent processes per arm gives
> Adaptive/UVA/native-copy throughput of `5.017/5.356/3.833` for 128/8,
> `8.681/4.457/6.838` for 512/16, and `10.895/4.840/10.934` for 512/32.
> These supersede the performance table below. See
> `PARK16_AUDIT_OFF_SCAN_20260915.md` for the authoritative rerun and
> `PARK16_STEP_CAUSE_20260915.md` for per-step and Nsight attribution.

## Protocol

- Model: DeepSeek-V2-Lite-Chat BF16, TRITON_MLA, eager execution.
- Arms: Adaptive PARK/RESUME, native UVA, native copy-on-forward.
- Frozen ShareGPT non-code prompts; all successful cells have strict cross-arm output-hash equivalence.
- Adaptive offload: 286 expert slices, 4,948,230,144 B (4.608 GiB).
- Native arms: the same 9 whole Parameters, 4,798,283,776 B (4.469 GiB).
- Initial and late cohorts; late requests arrive after engine step 4.
- Qualification scan: unless noted otherwise, each cell has one independent process per arm. The results establish large effect directions, not confidence intervals.

## Authoritative PARK=16, audit-off rerun

| Prompt / requests | Arm | req/s | TTFT ms | TPOT ms | E2E ms | measured H2D GB |
|---|---|---:|---:|---:|---:|---:|
| 128 / 8 (4+4) | Adaptive | 5.017 | 158.25 | 73.31 | 1257.92 | 9.896 |
| 128 / 8 (4+4) | Native UVA | 5.356 | 186.50 | 66.17 | 1179.04 | 0 |
| 128 / 8 (4+4) | Native copy | 3.833 | 140.30 | 101.40 | 1661.27 | 95.966 |
| 512 / 16 (8+8) | Adaptive | 8.681 | 233.97 | 82.16 | 1466.38 | 9.896 |
| 512 / 16 (8+8) | Native UVA | 4.457 | 1038.71 | 119.42 | 2830.02 | 0 |
| 512 / 16 (8+8) | Native copy | 6.838 | 217.66 | 109.47 | 1859.71 | 95.966 |
| 512 / 32 (16+16) | Adaptive | 10.895 | 348.51 | 122.08 | 2179.78 | 64.327 |
| 512 / 32 (16+16) | Native UVA | 4.840 | 2216.15 | 178.87 | 4899.12 | 0 |
| 512 / 32 (16+16) | Native copy | 10.934 | 329.83 | 118.20 | 2102.83 | 100.764 |

Every value is the mean of two independent processes per arm, with ABC/CBA
order reversal. All 18 processes completed, all output hashes agree across
arms, and no Xid/PTE error occurred.

## Historical qualification cells (superseded)

The following one-process cells are retained only for provenance. They mix
PARK thresholds and, for the original PARK=16 512/32 row, include the
large-Prefill residency diagnostic. Do not use them as the current ranking.

| Prompt / requests | PARK threshold | Arm | req/s | TTFT ms | TPOT ms | E2E ms | measured H2D GB |
|---|---:|---|---:|---:|---:|---:|---:|
| 128 / 8 (4+4) | 64 | Adaptive | 5.485 | 155.23 | 66.12 | 1147.01 | 9.896 |
| 128 / 8 (4+4) | - | Native UVA | 5.560 | 184.12 | 63.90 | 1142.58 | 0 |
| 128 / 8 (4+4) | - | Native copy | 3.867 | 138.74 | 100.64 | 1648.28 | 95.966 |
| 512 / 16 (8+8) | 64 | Adaptive | 8.882 | 233.01 | 80.33 | 1438.02 | 9.896 |
| 512 / 16 (8+8) | - | Native UVA | 4.491 | 1032.16 | 118.30 | 2806.69 | 0 |
| 512 / 16 (8+8) | - | Native copy | 6.916 | 216.84 | 108.19 | 1839.68 | 95.966 |
| 512 / 32 (16+16) | 64 | Adaptive | 5.098 | 564.23 | 123.51 | 2416.81 | 9.896 |
| 512 / 32 (16+16) | - | Native UVA | 4.929 | 2173.28 | 176.46 | 4820.21 | 0 |
| 512 / 32 (16+16) | - | Native copy | 11.150 | 327.85 | 116.44 | 2074.42 | 100.764 |
| 512 / 32 (16+16) | 16 | Adaptive | 9.833 | 553.82 | 126.75 | 2455.02 | 64.327 |
| 512 / 32 (16+16) | - | Native UVA | 4.946 | 2169.59 | 176.33 | 4814.47 | 0 |
| 512 / 32 (16+16) | - | Native copy | 11.124 | 328.08 | 116.53 | 2076.00 | 100.764 |

The 128-token cell is the mean of two processes per arm. The other table entries are qualification cells with one process per arm. A separate 512/16 diagnostic-patch regression reproduced the first result (Adaptive 8.986, UVA 4.485, copy 6.916 req/s), but it is not averaged into the table because the kernel source differed during that diagnostic run and the patch was later reverted.

## Historical findings (superseded by the audit-off rerun where applicable)

1. Pressure explains the original Adaptive/UVA tie. At 128 tokens and 8 requests, Adaptive/UVA throughput differs by only -1.35%. At 512 tokens and 16 requests, Adaptive is 97.8% faster than UVA and reduces TTFT by 77.4%.
2. Total scheduled tokens alone are not a sufficient access-mode predictor. The 512/32 trace is `8192, 16, 16, 16, 8192, 47, 32..., 16..., 1`. PARK=64 sends the 32-token Decode interval to UVA, where 32 independently routed requests create expensive random PCIe reads. Adaptive then reaches only 5.098 req/s, close to UVA's 4.929 req/s.
3. Lowering PARK to 16 keeps the 47/32-token interval in copy mode. Adaptive throughput rises from 5.098 to 9.833 req/s (+92.9%) and becomes 98.8% faster than UVA. H2D rises from 9.896 to 64.327 GB.
4. Control-plane switching is not the dominant remaining cost. Both 512/32 Adaptive traces perform two PARKs and one RESUME, totaling about 26-30 ms, below 1% of their makespan.
5. The former claim that native copy remained 13.1% faster was caused by the
   enabled residency diagnostic and is withdrawn. In the authoritative
   audit-off 512/32 rerun, native copy is only 0.35% faster than Adaptive.

## 2048-token blocker

No 2048-token performance number is valid. Four Adaptive attempts stopped before request admission:

- 32 requests, 65,536-token bucket, 2.5 GiB KV: profile-time Xid 31 / FAULT_PDE under extreme workspace pressure.
- 32 requests, 8,192-token bucket, 2.5 GiB KV: final warm-up Xid 31 / FAULT_PDE.
- 16 requests, 8,192-token bucket, 1 GiB KV: final warm-up Xid 31 / FAULT_PTE.
- The same 16-request cell after a detached-child republish diagnostic patch: still failed; the patch was reverted.

The fault occurs after KV allocation and hot-residency/full-dual refresh, before PARK/RESUME or real requests. In one fully resolved case, the fault VA was 68 KiB beyond a switched cold range, inside the adjacent hot fragment, while the driver residency audit reported all 23.84 GB of hot ranges GPU-resident. This supports a stale boundary PTE/PDE or visibility defect after the larger KV/VA layout, not random request behavior or a normal performance slowdown.

Every failed run stopped before the other arms, preserved its logs, restored stock `nvidia_uvm`, and left the GPU clean.

## Artifacts

- Authoritative PARK=16 audit-off rerun:
  `artifacts/park16-audit-off-p128-c8-20260915-attempt-01/`,
  `artifacts/park16-audit-off-p512-c16-20260915-attempt-01/`, and
  `artifacts/park16-audit-off-p512-c32-20260915-attempt-01/`.

- Baseline 128/8: `artifacts/sharegpt-three-arm-20260915T090142Z/`
- 512/16: `artifacts/pressure-p512-c16-20260915T091402Z/`
- 512/32, PARK=64: `artifacts/pressure-p512-c32-20260915T094037Z/`
- 512/32, PARK=16: `artifacts/pressure-p512-c32-park16-20260915T094537Z/`
- 2048 failures: `artifacts/pressure-p2048-c32-20260915T092019Z/`, `artifacts/pressure-p2048-c32-b8192-20260915T092314Z/`, `artifacts/pressure-p2048-c16-b8192-20260915T092522Z/`, and `artifacts/pressure-p2048-c16-republish-20260915T093739Z/`.
