# Reproducing the figure experiments

`run_figure_experiments.py` connects the 21 stored figure/table CSVs in
`evaluation/figures/data` to their GPU experiment scripts. It runs one top-level
round; warmup, conditioning, request lengths, batch sizes and samples inside a
protocol are retained. These runs provide point estimates rather than new
confidence intervals. Existing figure CSVs are never overwritten.

## Environment

Build the driver, allocator and bundled vLLM as described in
[the build guide](../../docs/REPRODUCING_ADAPTMOE.md). The experiments use a single
RTX 5090 with about 32 GB HBM and 14 physical CPU cores. Other configurations can
change both feasibility and performance. A model-free test is available through
[the static alias probe](STATIC_ALIAS_REPRODUCTION.md).

The runtime Python must be available at `workloads/vllm/vllm/.venv/bin/python`.
To reuse an already compatible build in another local workspace:

```bash
python3 scripts/borrow_runtime.py "$RUNTIME_WORKSPACE" --baselines
```

This links ignored compiled extensions, the Python environment, the UVM module
and optionally the pinned baseline checkouts. It refuses to replace existing
files. Local experiment processes still import this checkout's vLLM Python
source; `runtime-import.txt` records and checks that location. Keep build inputs
out of Git. Rebuild when their source or ABI differs from the current checkout.

Set external input paths before starting:

```bash
export DEEPSEEK_MODEL=/path/to/DeepSeek-V2-Lite-Chat
export QWEN_MODEL=/path/to/Qwen3-30B-A3B-FP8
export SHAREGPT_DATASET=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json
```

The frozen requests and expert plans are included. Qwen near-equal plan generation
uses `inputs/frozen/qwen_decode_calibration_counts.json` under the evaluation
experiment directory. The generator verifies its checksum and the original
calibration tensor fingerprint. It produces the same cold11/39/90 coordinates
as that original tensor, without requiring a binary `.pt` input.

The direct DeepSeek runner accepts the frozen C1 and C16 candidate arrays.
KTransformers uses seven expert threads and a GPU Prefill threshold of zero
except for RAG/C16, which uses 4096. RAG/C1 retains zero; enabling the C16
full-GPU fallback there can exhaust memory. Native Copy Graph runs leave the
extra static bank disabled, matching the successful stored runs: its single
stream guard is incompatible with vLLM's separate capture stream.

Optional KTransformers and MoE-Infinity sources/builds follow
[the baseline guide](../../baselines/README.md). The two `legacy-six-systems-*`
jobs also need an existing DeepSeek MoE-Infinity store and its matching extension
build receipt:

```bash
export MOEI_OFFLOAD=/path/to/moe-infinity-store
export MOEI_RECEIPT=/path/to/store_build_receipt.json
```

The default locations are `.build/moe-infinity-offload` and
`.build/moe-infinity-build-receipt.json`. The receipt must match the loaded
`_store` extension. Store data and receipts are local runtime inputs.

Install plotting dependencies in a separate environment, if needed:

```bash
python3 -m venv .build/plot-venv
.build/plot-venv/bin/python -m pip install -r evaluation/requirements.txt
export FIGURE_PYTHON="$PWD/.build/plot-venv/bin/python"
```

E13 uses this environment when present, or `FIGURE_PYTHON` when explicitly set.

## Run

```bash
python3 experiments/reproduce/run_figure_experiments.py --list
ALLOW_GPU_RUN=1 python3 -u experiments/reproduce/run_figure_experiments.py \
  --output .build/figure-experiments-once
```

Use `--only` with one or more job names to run a subset:

```bash
ALLOW_GPU_RUN=1 python3 -u experiments/reproduce/run_figure_experiments.py \
  --only page-table-parent page-table-leaf placement \
  --output .build/figure-mechanisms-once
```

Every output directory must be new. Retain failed directories and use a new
directory after fixing an input or script. The suite serializes GPU jobs and
checks that HBM is empty and stock UVM is restored between jobs. The module
harness needs passwordless `sudo` for module/service management. It temporarily
stops monitoring exporters during module cycles and restores their initial
state afterwards; it also handles exporters restarted during a long run.

| Jobs | Figure/table input stems |
|---|---|
| `page-table-parent`, `page-table-leaf` | `fig_pd_page_table_switch`, `tab_pd_lifecycle_invariants` |
| `placement` | `fig_e14_placement_scaling` |
| `e6-batched`, `e6-full`, `e6-serial` | `fig_batched_plan_update` |
| `e5-graph`, `e5-four-paths`, `e5-prebuild-on`, `e5-prebuild-off`, `e5-stable`, `e5-nonstable` | `fig_core_mechanism_ablation` |
| `kv-lifecycle` | `fig_plan_memory_accounting` |
| `kv-roundtrip` | `fig_fixed_plan_tradeoff`, `fig_adaptive_steady_state` |
| `kv-rag` | `fig_e13_rag_pressure_throughput` |
| `kv-partition` | `fig_kv_expert_partition_crossover` |
| `pressure-p128-c8`, `pressure-p512-c16`, `pressure-p512-c32` | `fig_adaptmoe_pressure_throughput` |
| `stock-uvm` | `fig_stock_uvm_demand_paging`, `fig_stock_uvm_vs_native_uva`, `fig_stock_uvm_vs_native_uva_e2e` |
| `qwen-stock-kv6` | `fig_qwen_stock_uvm_vs_native_uva` |
| `deepseek-c{1,16}-*` | `fig3a_deepseek_performance`, `fig_e2e_six_systems_normalized`, `fig_e2e_six_systems_normalized_bars` |
| `qwen-adaptmoe`, `qwen-native-*`, `qwen-kt` | `fig3b_qwen_performance`, `fig_phase_path_preference`, both normalized systems figures |
| `legacy-six-systems-c1`, `legacy-six-systems-c16` | `fig_e2e_six_systems` |

The current six-mechanism matrices and the older six-system comparison have
different placements/protocols. C1 in the older comparison uses its original
HTTP/server runner, including MoE-Infinity; C16 uses the direct batch runner.
Cross-framework completion does not establish exact generated-token equivalence.
The standalone Qwen KTransformers job verifies all ten cells' output lengths
and hashes, placement, Graph capture/replay, configured backend/threads/Prefill
thresholds, timings and cleanup. It does not require an old AdaptMoE result
directory and does not claim cross-framework output equivalence.

E5 `e5-four-paths` runs the frozen cold5 route/all-GPU/eager/Graph qualification
with performance tracing disabled. `e5-prebuild-on/off` run the matched 4K PTE
dual-Graph protocols. Together with H/G and stable/nonstable runs these cover the
figure's original mechanisms and the later incremental request-path comparison.

Two Native Copy Graph cells retain the extra offload documented in the final
experiment report. DeepSeek RAG/C1 `FULL_SEPARATE` uses five exact parameter names
(1.843994 GiB realized); its eager cell retains the lower placement. Qwen RAG/C16
Copy `FULL_SEPARATE` requests 21.0 GiB (21.298364 GiB realized in the stored run),
while UVA and eager Copy retain the 20.25 GiB request. These adjustments provide
capture headroom without reducing request length, concurrency or KV capacity.

The E5 H/G harness explicitly enables
`VLLM_UVM_STARTUP_EMPTY_CACHE_NO_HOT=1`. Explicit KV allocation skips vLLM's
ordinary profile-cache cleanup, and kernel/sampler warmup can retain additional
unused native allocator segments. The opt-in releases unused segments at the
startup sealed-residency reconciliation boundaries. It rejects placements with
any managed hot-expert complement before releasing cache: those placements need
their reserve to prevent later allocator growth from evicting hot pages. Live
tensors and Graph-owned allocations stay allocated. The 256 MiB free-memory
gate applies before KV allocation and before kernel/Graph warmup. After startup,
the same helper checks the zero-hot condition, topology and Parameter VAs, but
does not repeat the pre-warmup budget after live Graph workspaces have consumed
part of it. The receipt records `minimum_free_headroom_checked=false` for that
final zero-hot audit. Every managed-hot placement retains the free-memory gate
at every stage, as does the default path without this option. This option does
not change other experiment families' default cache-preservation policy.

The H/G wrapper uses the 128 MiB KV budget from the stored incremental ablation,
matching the dual-Graph and stable/nonstable runs; the generic phase harness's
160 MiB default is a different configuration.
Changing `GPU_MEMORY_UTILIZATION` does not change an explicitly allocated KV
budget.

The earlier cleanup-only attempts completed H but stopped G after Graph capture:
free HBM was 214,695,936 bytes, below the 268,435,456-byte pre-warmup budget.
Allocator expansion and split-size trials did not resolve this. The final audit
was incorrectly applying that startup budget again after allocating the live
Graph pool. Preserve those failed attempts when validating the corrected
startup phases; they are not successful reproductions. The stored September
run used an earlier runtime and a different UVM build.

## Inspect fresh results

`execution.json` records commands, selected environment, entry script checksums,
return codes and verifier locations. Each job retains its own case logs,
provenance and outputs. Examine the cell records as well as launcher exit codes:
the stock UVM protocols can complete while recording OOM or timeout observations.

```bash
python3 experiments/reproduce/figure_run_report.py \
  .build/figure-experiments-once --output .build/figure-run-report.json
```

The report preserves unsuccessful execution rows and separates functional
verifier status from performance claim gates. For example E14 `status=ok` with
`claim_gate.requested_pattern_observed=false` means the mechanism ran correctly
but the requested performance pattern was not reproduced. Preserve that result.
For the two stock-UVM protocols it also recomputes output hashes from the saved
token IDs, checks completed request counts, and compares prompts/outputs for
matching completed baseline/UVM cells. OOM cells remain explicit observations
and are excluded from output equivalence comparisons.

After all three E6 jobs complete, aggregate them with:

```bash
python3 experiments/eurosys_moe_offload_evaluation/aggregate_e6_qualification.py \
  --difference-root .build/figure-experiments-once/e6-batched \
  --full-rebuild-root .build/figure-experiments-once/e6-full \
  --serial-wave-root .build/figure-experiments-once/e6-serial \
  --output .build/e6-summary/summary.json
```

Compare the one-round stable/nonstable runs without bootstrap intervals:

```bash
python3 experiments/eurosys_moe_offload_evaluation/compare_e5_stable_vs_nonstable_va.py \
  .build/figure-experiments-once/e5-stable \
  .build/figure-experiments-once/e5-nonstable --single-round \
  --output .build/e5-summary/comparison.json
```

The default comparison still requires at least three process pairs.

Rendering the stored tables is a separate CPU-only operation:

```bash
make figures FIGURE_PYTHON="$FIGURE_PYTHON"
```

The suite retains fresh structured results and each protocol's own summaries;
E13 also renders its fresh measurements directly. It does not replace all 21
stored CSVs with newly measured data. The extraction scripts under
`evaluation/figures/scripts` accept different inputs: some read result trees,
while others parse the report tables used for the stored figure. Check each
script's `--help` and its annotations before applying it to a new run. Keep
fresh and historical results separate, retaining failed cells, source
fingerprints, output checks and cleanup evidence.
