# ShareGPT low/high/low pressure wave: AdaptMoE K2/K4 versus Native Copy/UVA

Status: complete. Each arm finishes **384/384** HTTP requests and **72,762**
service-reported output tokens, without CUDA/UVM errors. Stock UVM is restored
and GPU memory returns to 0 MiB. One cold-process run per mechanism.

## Results

QPS is completions in the wall-clock interval. Mean latency groups requests
by arrival phase, including those completing later. TPOT uses the corrected
service-token denominator explained below.

| Time / input rate | Mechanism | Completion QPS | TTFT mean (s) | TPOT mean (ms) |
|---|---|---:|---:|---:|
| 0--512 s / 0.25 req/s | AdaptMoE K2/K4 | 0.2422 | 0.247 | 66.05 |
| 0--512 s / 0.25 req/s | Native Copy K4 | 0.2402 | 0.290 | 137.07 |
| 0--512 s / 0.25 req/s | Native UVA K4 | 0.2422 | 1.568 | 89.48 |
| 512--576 s / 2 req/s | AdaptMoE K2/K4 | **0.9844** | **21.831** | **76.12** |
| 512--576 s / 2 req/s | Native Copy K4 | 0.4375 | 66.044 | 144.08 |
| 512--576 s / 2 req/s | Native UVA K4 | 0.3438 | 84.758 | 164.43 |
| 576--1088 s / 0.25 req/s | AdaptMoE K2/K4 | 0.3770 | 3.033 | 66.47 |
| 576--1088 s / 0.25 req/s | Native Copy K4 | 0.4453 | 25.569 | 140.83 |
| 576--1088 s / 0.25 req/s | Native UVA K4 | 0.4570 | 58.181 | 146.01 |

Higher native recovery QPS reflects more old high-pressure work remaining,
not faster handling of new requests. Recovery has 128 new arrivals but
193 / 228 / 234 completions for AdaptMoE / Copy / UVA. After 1088 s,
4 / 5 / 4 final completions remain; all are retained in drain windows.

Recovery **last 128 seconds (960--1088 s)**, the same final 25 arriving
requests in each arm, has local arrival QPS 0.1953 within the burst pattern:

| Mechanism | Completion QPS | TTFT mean (s) | TPOT mean (ms) |
|---|---:|---:|---:|
| AdaptMoE | 0.1953 | 0.258 | 66.29 |
| Native Copy | 0.1953 | 0.294 | 135.23 |
| Native UVA | 0.2969 | 1.553 | 77.41 |

UVA still completes some earlier requests in that interval.

![30-second time series](../artifacts/pressure-wave-k2-k4-vs-native-low-high-low-20261021-attempt-01/pressure_wave.png)

Standalone QPS figure, generated 2026-10-22 from the same 30-second windows:
[PNG](../artifacts/pressure-wave-k2-k4-vs-native-low-high-low-20261021-attempt-01/qps_over_time.png),
[SVG](../artifacts/pressure-wave-k2-k4-vs-native-low-high-low-20261021-attempt-01/qps_over_time.svg),
[PDF](../artifacts/pressure-wave-k2-k4-vs-native-low-high-low-20261021-attempt-01/qps_over_time.pdf).
Its curves stop at each arm's last completion window instead of showing
the common-horizon zero padding after that arm finishes. Source data are
in `qps_over_time.csv`; reproduce with `plot_pressure_wave_qps.py RUN_ROOT`.

## Interpretation and mechanism audit

The result supports **less accumulated work, relatively small TPOT variation,
and faster recovery** for this AdaptMoE configuration. It does not support
pressure-invariant TTFT or capacity sufficient for 2 req/s.

- AdaptMoE TPOT increases 15.25% at high load, 66.05 -> 76.12 ms. UVA
  increases 83.77%, 89.48 -> 164.43 ms. Copy changes less (5.12%), but has
  substantially higher absolute TPOT (137.07 -> 144.08 ms).
- AdaptMoE high-phase completion QPS is 2.25x Copy and 2.86x UVA. Its
  high-arrival TTFT is 66.94% / 74.24% lower, but increases sharply from its
  own initial low phase. Input exceeds service performance and all arms
  queue. More KV cannot eliminate compute/transport saturation.
- After the 576 s load drop, the first server sample with zero waiting is
  at 636.15 / 826.16 / 965.18 s: **60 / 250 / 389 seconds to recover** for
  AdaptMoE / Copy / UVA. These are approximately 10-second samples, not
  exact instants or a guarantee of zero waiting thereafter. Peak sampled
  waiting counts after high starts are 54 / 90 / 93; all reach C16.
- AdaptMoE recovery-tail TTFT is 4.75% above its initial-low mean; TPOT is
  0.37% above it. Single-run observations do not establish equivalence.

Actual KV capacity publications (core log resolution about 1 second):

| Relative time (s) | Scheduler step | Committed KV / expert change |
|---:|---:|---|
| 224.15 | 3227 | K2 -> K4 / cold14 -> cold28 |
| 670.15 | 9613 | K4 -> K2 / cold28 -> cold14 |
| 800.15 | 11474 | K2 -> K4 / cold14 -> cold28 |

The first growth occurs during nominal low pressure because bursts and long
prompts can exceed K2 at low average arrival rate. High therefore starts on
K4. Recovery shrinks to K2, then the repeated long-prompt burst grows to K4.
Targets follow actual KV demand, not the envelope labels; none is forced.

Formal access transitions are 273 PARK and 273 RESUME; aggregate manifest
wall times are 2.727 + 2.157 = 4.883 s. Token-threshold access switching and
capacity adjustment both operate. Without a fixed-K4 AdaptMoE ablation or
profiler this cannot isolate the contribution of resizing, access choice,
expert placement or copy granularity.

Overall QPS is 0.3472 / 0.3368 / 0.3472 for AdaptMoE / Copy / UVA. Most
elapsed time is low pressure and all arms finish, so this mostly measures
arrival timing and final drain, rather than high-load service capacity.

## Validation and limitations

Both native arms select identical **181 whole Parameters**, 6,656,924,760 B
(6.19975 GiB), 4.969% more than K4/cold28's 6,341,787,648 B (5.90625 GiB).
AdaptMoE K2/cold14 offloads 2.953125 GiB. Native selection includes attention
and auxiliary Parameters. This is a system-configuration comparison, not
identical expert coordinates or exact byte-matched transport.

Trace SHA-256 is `40fe92cd6d419cf550f9ab962e22a654675209ea2059281502317dc515482d6d`;
source is `416fe6e9bd464fec30259bdc375c462a1486e252b5a2555176452ab7253d98a2`.
All service-reported lengths match. All completions sum to 384 across each
arm's 38 windows, including the final endpoint. Maximum send lag is
10.93 / 10.56 / 7.28 ms; client admission is not the large-latency cause.

The original client prefix heuristic loses one initial repeated token for
`sharegpt-0092` in each phase: 109 IDs/timestamps captured versus 110 tokens
reported by the service, three records per arm. First and last timestamps
remain available. Reported phase TPOT is corrected to
`(last_token_time-first_token_time)/(service_completion_tokens-1)` without
inventing timestamps or modifying raw replays. Overall correction is at
most 0.011 ms. The plotted observed-gap series retains this tiny missing-gap
count limitation. Future client parsing is fixed using continuous-usage
counts and tested for both delta and cumulative token streams.

Original `replay.json` TPOT retains its old denominator. Corrected values are
in `phase_summary.csv` and `analysis.json.stream_capture_audit`. The exact
client used is preserved as `replay_client_used.py`, SHA-256
`a98dc52f66487913cb4661983228e835fc5845e9ca03655cb2df29d6f097c4ab`,
matching all three provenance files.

Generated outputs are not all identical. Versus AdaptMoE, first tokens match
382/384 for Copy and 381/384 for UVA. Of 381 records with complete captured
IDs, full sequences match 101 for Copy and 108 for UVA. Divergence cause is
not independently established. Results measure fixed-input, fixed-requested-
length serving performance; they do not certify numerical/semantic equivalence.

Three focused tests cover trace content/burst preservation, window boundaries
and late completions, stream parsing, and TPOT correction. Shell syntax,
Python compilation and `git diff --check` pass. No controller/worker code is
changed in this experiment. Single-run results have no confidence intervals.

## Fixed protocol

- Qwen3-30B-A3B-FP8, RTX 5090, TRITON_ATTN, eager, CUDA Graph disabled.
- Repeat the same frozen 128 ShareGPT requests in the same order in each
  phase. All prompt token IDs and requested output lengths remain unchanged.
  Repeated phase instances have distinct request IDs; prefix caching is off.
- Initial low: 0--512 s, 128 arrivals, 0.25 req/s.
- High: 512--576 s, 128 arrivals, 2 req/s.
- Recovery low: 576--1088 s, 128 arrivals, 0.25 req/s.
- The frozen burst pattern is affinely scaled within each phase. There is no
  pause, process restart, or queue drain at phase boundaries. Three arms use
  exactly the same 384-request JSONL and arrival offsets.
- C16, 8192-token scheduler budget, 512 client connections to avoid a client
  admission bottleneck. Four identical trace-prefix warmups are excluded.
- AdaptMoE: existing K2/cold14 and K4/cold28 plans; proactive growth at 80%
  KV watermark, existing audited shrink and expert restoration enabled;
  PARK <=16 and RESUME >=128 scheduled tokens. No manual tier forcing.
- Native Copy and Native UVA: fixed 4,295,491,584 B KV allocation matching
  AdaptMoE's peak reservation; 5.90625 GiB nominal whole-Parameter offload
  budget matching the peak cold28 expert bytes. Actual native selection is
  audited from each manifest. Dynamic slices versus static whole Parameters
  are a system-configuration comparison, not an equal-coordinate microbenchmark.
- One independent cold process per mechanism. No preselected stability claim
  and no statistical confidence claim from this single sweep.

## Metric definitions

Time origin is formal trace start after warmup. QPS is successful completions
in a wall-clock interval divided by its duration. Input QPS is reported
separately. A recovery window can complete old high-pressure requests and
therefore exceed its current input rate.

TTFT and per-request mean TPOT are grouped by request **arrival** time;
requests completing after the interval remain in their arrival cohort.
The time-series plot additionally reports actual token intervals grouped by
the receive time of each interval's final token. This separates generation
observed during recovery from full-lifetime request averages. Token timing
is from streaming HTTP reception and includes serving/queue/resize stalls;
it is not isolated GPU-kernel latency.

The analysis emits 30-second windows and complete phase summaries, retains
final drain completions, and reports the last 128 seconds of recovery
separately. Empty latency windows are null, not zero. KV tier commit events
are taken from core publication logs with one-second timestamp resolution.

## Reproduction and artifacts

Launcher:
`sharegpt_trace/run_pressure_wave_three_arm.sh /absolute/new/run-root`

Run root relative to `experiments/eurosys_moe_offload_evaluation`:
`artifacts/pressure-wave-k2-k4-vs-native-low-high-low-20261021-attempt-01/`.

Raw streaming timestamps and request results are in each arm's `replay.json`;
completed-request checkpoints are in `replay.progress.jsonl`. The frozen
arrival manifest and trace are stored at the run root. Final artifacts
include `time_windows.csv`, `phase_summary.csv`, `server_samples.csv`, `analysis.json`, and
`pressure_wave.png` / `pressure_wave.svg`.

Recompute using `sharegpt_trace/analyze_pressure_wave.py RUN_ROOT` with
matplotlib available. `monitor_pressure_wave.py` provides read-only progress
snapshots. Existing experiments retain their default trace/arrival scale.
