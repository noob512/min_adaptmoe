# Experiments and figures

This index connects the runtime experiments to the plotting inputs in
`evaluation/figures/data/`. Each linked protocol records its model, frozen
requests, memory budgets, process order, correctness gates and output schema.
Build the bundled runtime and driver using [the reproduction guide](REPRODUCING_ADAPTMOE.md)
before starting GPU experiments.

| Evaluation topic | Protocol / experiment entry | Figure input stems |
|---|---|---|
| Mixed scheduling, token thresholds, fixed/dynamic KV | [Four-mechanism comparison](../experiments/adaptmoe_token_threshold_sharegpt_compare/README.md); `run_kv1_compare.sh` in that directory | New runs produce `summary.csv`, `output_checks.csv`, `REPORT.md` |
| End-to-end systems (E2) | [E2/E3 protocol](../experiments/reproduce/REPRODUCE_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md); [formal E2 runner](../experiments/eurosys_moe_offload_evaluation/run_formal_e2.sh) | `fig_e2e_six_systems`, `fig_e2e_six_systems_normalized`, `fig_e2e_six_systems_normalized_bars` |
| Prefill/decode paths (E3), DeepSeek/Qwen | [E3 three mechanisms](../experiments/reproduce/REPRODUCE_E3_PREFILL_DECODE_THREE_MECHANISM_B1_B8.md); [Qwen reproduction](../experiments/reproduce/REPRODUCTION_QWEN3.md) | `fig3a_deepseek_performance`, `fig3b_qwen_performance`, `fig_phase_path_preference` |
| Mechanism ablations and stable VA (E5) | [E5 protocol](../experiments/reproduce/REPRODUCE_E5_CURRENT.md); [nonstable VA runner](../experiments/eurosys_moe_offload_evaluation/run_e5_nonstable_va_recapture_formal.sh) | `fig_core_mechanism_ablation` |
| Plan differences, batching and memory accounting (E6) | [E6 protocol](../experiments/reproduce/REPRODUCE_E6_PLAN_SWITCH_QUALIFICATION.md); [plan difference runner](../experiments/eurosys_moe_offload_evaluation/run_formal_e6_difference.sh) | `fig_batched_plan_update`, `fig_plan_memory_accounting` |
| Page-table switching and lifecycle | [Dual-view phase experiment](../experiments/vllm_uvm_prefill_alias_decode_remote_graph/README.md); [static alias probes](../experiments/uvm_static_alias_original_va/README.md) | `fig_pd_page_table_switch`, `tab_pd_lifecycle_invariants` |
| Fixed-plan and adaptive capacity (E10/E13) | [E10 protocol](../experiments/reproduce/REPRODUCE_E10_CURRENT_TIER_DUAL_VIEW_FULL_RUN.md); [E13 protocol](../experiments/reproduce/REPRODUCE_E13_KV_ADAPTIVE.md); [round trip](../experiments/reproduce/REPRODUCE_E13_FULL_SEPARATE_K2_K6_ADAPTIVE_ROUNDTRIP.md) | `fig_fixed_plan_tradeoff`, `fig_kv_expert_partition_crossover`, `fig_adaptive_steady_state`, `fig_e13_rag_pressure_throughput`, `fig_adaptmoe_pressure_throughput` |
| Placement scaling (E14) | [E14 protocol](../experiments/eurosys_moe_offload_evaluation/e14/README.md); [E14 runner](../experiments/eurosys_moe_offload_evaluation/e14/run_e14.sh) | `fig_e14_placement_scaling` |
| Stock UVM demand paging and Native UVA | [Stock UVM runner](../experiments/eurosys_moe_offload_evaluation/run_h_vllm_stock_uvm.sh); [result summarizer](../experiments/eurosys_moe_offload_evaluation/summarize_h_vllm_stock_uvm.py); [Qwen KV6 runner](../experiments/eurosys_moe_offload_evaluation/run_qwen_kv6_uva_vs_uvm.sh) | `fig_stock_uvm_demand_paging`, `fig_stock_uvm_vs_native_uva`, `fig_stock_uvm_vs_native_uva_e2e`, `fig_qwen_stock_uvm_vs_native_uva` |

The [E0–E14 index](../experiments/reproduce/README.md) also retains qualification,
diagnostic, failed and incomplete experiments with their original status. An
entry in that index does not imply that the entire experiment passed. Historical
protocols can refer to external run directories and machine-specific paths;
preserve their recorded module/source fingerprints when interpreting the stored
measurements, and record fresh fingerprints for new runs.

## Inputs and optional baselines

Frozen JSON requests, per-layer expert plans and their provenance manifests are
included under `experiments/`. The P512 and P1024 mixed-scheduling fixtures retain
their original checksums. Model weights, the complete ShareGPT dataset, downloaded
dependencies and compiled libraries are external inputs. Set `MODEL` and
`SHAREGPT_DATASET` explicitly for the current four-mechanism entry point.

The Stock UVM runner accepts `DEEPSEEK_MODEL`, `QWEN_MODEL`, `VLLM_PY` and
`CPUSET`; pass a new output directory as its first argument. For E14, set
`MODEL` and use `config_request_specific.json` and the frozen
`request_specific_cold29_plan.json` as documented in its protocol.

Fetch optional systems with [the baseline helper](../baselines/README.md).
KTransformers placement tensors are generated from the included JSON plans,
rather than distributed as `.pt` files. For the frozen DeepSeek E2 plans:

```bash
ASSET_ROOT="$PWD/experiments/eurosys_moe_offload_evaluation/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/calibration_assets"
workloads/vllm/vllm/.venv/bin/python \
  experiments/hybrid_offload_2_4_6_8_compare/prepare_kt_frequency_plans.py \
  --input-dir "$ASSET_ROOT/plans" --output-dir .build/kt-placements
```

This creates `kt_frequency_{2g,4g,6g,8g}.pt` and a new manifest. For the historical
`run_formal_e2.sh` layout, generate into `$ASSET_ROOT/kt_placements` instead;
this replaces its provenance manifest with one describing the new local tensors.
Other plan formats use
`experiments/ktransformers_cpu_expert_compare/cold_plan_to_kt_stats.py`.
Routing recalibration uses
`experiments/eurosys_moe_offload_evaluation/prepare_deepseek_calibration_assets.sh`
and requires a model, a GPU and a fresh `OUTPUT_ROOT`. Calibration traces and
plans in `inputs/` are frozen inputs, not newly collected results.

## Rebuild the figures

All 21 stored CSV tables can be rendered without model weights or a GPU:

```bash
python3 -m pip install -r evaluation/requirements.txt
make figures
```

The shared renderer validates the inputs and writes plots, the lifecycle table
and input checksums into `.build/figures/`. The original per-figure scripts in
`evaluation/figures/scripts/` also extract tables from structured results; their
`--help` output documents the corresponding result/report arguments. Provenance
columns and the summaries in `evaluation/data/processed/` distinguish historical
measurements, failed cells, single runs and repeated samples. Rendering a stored
table does not repeat its GPU measurement.
