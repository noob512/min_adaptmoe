# vLLM prefill: P3 vs pinned copy vs UVA remote read

This experiment compares three real vLLM prefill paths with an identical
Qwen3-30B-A3B-FP8 weight selection:

- `p3`: the registered 512-KiB P3 plan, two fixed banks, CUDA fences, and no
  compute/copy overlap;
- `pinned`: vLLM exact-name pinned-host copy-on-forward;
- `uva`: the custom managed-UVA remote-read path used by the planned decode
  phase.

The footprint-matched selection is all 128 experts in layers 0 through 3, or
`4 x 576 MiB = 2.25 GiB` of offloaded main expert weights. This full-layer
selection is intentional: packed Qwen expert
parameters cannot be independently selected by vLLM's pinned parameter
offloader. Comparing sparse-12 P3 against packed pinned copy would compare
different offloaded footprints. P3 and pinned copy transfer the complete
selected layer weights. UVA remote read maps the same 2.25-GiB selection but
generates PCIe reads only for cache lines actually consumed by routed experts;
its dynamic bus traffic is therefore intentionally not asserted to equal the
P3/pinned transfer byte count.

The runner defaults to one frozen token-ID prompt, concurrency 1, no warmup,
and exactly one output token. Consequently request TTFT is the prefill metric;
there is no model decode step after the first sampled token. Concurrency,
warmup fixture/count, chunked-prefill behavior, and tracing are parameterized
so a formal run can reproduce the earlier four-request/1217-token workload.
P3 still does not overlap layer computation with weight copy.

The default process order is `PCUUCP` (`P=p3`, `C=pinned`, `U=uva`), which
gives two fresh-process observations per mechanism and mirrors order drift.
Use a longer balanced order for a formal run.

The historical balanced 1217-token report was cleared on 2026-08-23. Rerun it
under a fresh protocol ID before drawing performance conclusions.

## Static validation

This does not initialize CUDA or modify a kernel module:

```bash
./run_compare.sh static-preflight
```

## Runtime

The runner never loads, unloads, or changes `nvidia_uvm`. Load the P3 module
externally with the same contract used by the full-layer dual-bank E2E test:

```text
uvm_moe_slot_pte_remap_enable=1
uvm_moe_slot_pool_vidmem_budget_bytes>=1207959552
uvm_moe_slot_plan_overlap_copy_prepare=0
uvm_moe_slot_plan_register_pte_prewarm=1
uvm_moe_slot_plan_cache_copy_runs=1
uvm_moe_slot_plan_cache_metadata_recipe=1
uvm_moe_slot_transition_fast_index=1
uvm_moe_slot_transition_direct_cutover=1
uvm_moe_slot_plan_fast_validation=1
uvm_moe_slot_plan_shadow_pte_prefill=1
uvm_moe_slot_plan_dual_bank_event=1
uvm_moe_slot_plan_cuda_fence=1
```

Then run:

```bash
ALLOW_GPU_RUN=1 ./run_compare.sh run
```

Useful overrides are `ORDER`, `RUN_ROOT`, `MODEL`, `EVAL_FIXTURE`,
`GPU_INDEX`, `CPUSET`, `CPU_THREADS`, and `TIMEOUT_SEC`. The GPU must be at
exactly 0 MiB with no compute clients before and after every arm.

`verification.json` reports per-arm TTFT samples and medians, P3 deltas versus
pinned and UVA, output-token equality, plan byte equality, pinned allocation
evidence, UVA remote-mapping evidence, and P3 registered-plan/fence evidence.

The `uva` label here means the project's custom managed-UVA remote-read
mechanism, not vLLM's generic `cpu_offload_gb` native-UVA selection. The latter
cannot currently select these eight exact Qwen parameters and therefore is not
a byte-matched control.
