# Fragmented 512 KiB registered-plan probe

This probe tests whether the registered-copy cache, dual-bank copy-first path,
fast validation, metadata recipes, and fixed-VA transaction still work when a
576 MiB layer is not expressible as 288 complete 2 MiB VA blocks.

It provides complete-block controls plus three partial-capable layouts:

| layout | managed allocations | 512 KiB entries | expected VA blocks |
|---|---:|---:|---:|
| `packed` | 288 × 2 MiB | 1,152 | 288 complete |
| `gapped` | 288 × 4 MiB, only the first 2 MiB active | 1,152 | 288 complete, globally non-contiguous |
| `fragmented` | 1,152 × 2 MiB, only the first 512 KiB active | 1,152 | 1,152 partial |
| `sparse-experts` | `9C × 2 MiB`, only the first 512 KiB active | `9C` | `9C` partial, for a selected set of `C` experts |
| `qwen-packed-sparse` | two contiguous tensors (`w13=384 MiB`, `w2=192 MiB`) | `9C` selected views | packed full/partial mix determined by the cold IDs |

The fragmented shape is the conservative endpoint for a Qwen3 layer with 128
experts and `6 + 3` envelopes per expert (`3 MiB + 1.5 MiB`). It preserves the
same 1,152-entry/576-MiB payload while guaranteeing that no result depends on
accidental adjacency between tensors.

`sparse-experts` models a cold subset of the 128 experts. Each selected expert
owns six 512 KiB `w13` envelopes (3 MiB) and three 512 KiB `w2` envelopes
(1.5 MiB). Every envelope has an independent managed allocation, so the probe
does not depend on adjacency either within an expert or between experts. All
nine entries retain the same expert owner ID while their slot IDs remain
distinct. Layers may select different expert IDs as long as every layer keeps
the same expert count and nine-envelope lane geometry.

`qwen-packed-sparse` reproduces the actual Qwen3-30B-A3B-FP8 main-weight
geometry observed by vLLM: `w13_weight` has shape `[128,1536,2048]`, byte
stride `[3145728,2048,1]`, and 3 MiB per expert; `w2_weight` has shape
`[128,2048,768]`, byte stride `[1572864,768,1]`, and 1.5 MiB per expert.
The tensors are distinct allocations, but each tensor is internally packed.
The dataset returns only the selected experts' six plus three 512 KiB views to
the registered plan. Its checksum covers the complete 576 MiB tensor payload,
so unselected hot experts in partial blocks are checked without the
independent-allocation complement shim.

The `gapped` arm separates a different concern: 64 KiB grouping requires each
touched 2 MiB block to be complete, but it does not require those blocks (or
the packed `w13` and `w2` tensors) to be adjacent in the global VA space.

The script never loads UVM and never writes a module parameter. The operator
must configure the intended module before each arm. Every successful arm runs
bootstrap L0 followed by three complete `L1 -> L2 -> L3 -> L0` cycles; it does
not run the 11 registered-plan negative workers. Steady-state statistics begin
with the second L0 visit, after every layer has allocated its PTE hierarchy.

Each artifact contains:

- exact complete/partial VA-block occupancy;
- wall time and full `latency_stats` delta for every preallocate, register,
  execute, destroy, and close operation;
- extracted per-stage `slot_direct_profile.*.ns` timings;
- generation sequence, checksum after every execute, and compute-event order;
- active 64 KiB/4 KiB leaf counts and touched 2 MiB block counts;
- exact registration-time full/partial hierarchy-prewarm counters and the
  execute-time `partial_64k_blocks` count;
- for partial-64K, a GPU checksum of every unowned 1.5 MiB complement after
  every transition;
- zero userspace slots after close and `0 MiB` after worker exit.

## Partial 64 KiB contract

`partial64k` is a probe label, not a new UAPI mode. It uses the existing mode
21 bootstrap and mode 20 with the existing 64 KiB transition flag. The worker
requires PTE prewarm, verifies that every touched block is reported as 64 KiB,
checks exactly eight active 64 KiB leaves per 512 KiB envelope, and rejects any
silent 4 KiB fallback. Every registration must report one successful prewarm
and exact full/partial block counts for the submitted geometry. Every execute
must report the same partial-block count.

Shadow eligibility is accounted per block. Complete groups may use shadow leaf
prefill while partial groups use the live-leaf path, because activating a
partial shadow parent would replace unrelated leaves in the same block. With
the module switch enabled, a non-bootstrap mixed transition must therefore
report exactly `complete_2m_blocks` shadow hits; with it disabled the count must
remain zero.

Before registration, the guard shim seeds the unowned 1.5 MiB complement of
every 2 MiB allocation through a GPU kernel. After each transition, another
GPU kernel verifies that this complement remains readable and byte-for-byte
unchanged. This catches implementations which accidentally invalidate or
replace the whole 2 MiB block.

The guard fault can also pull neighbouring pages of the same managed
allocation into GPU residency. As the final residency operation before
preallocation/registration, the shim therefore issues an exact 512 KiB
`cudaMemPrefetchAsync(..., cudaCpuDeviceId)` for every active envelope and
synchronizes the probe stream. It does not prefetch the remaining 1.5 MiB, so
the complement stays GPU-accessible. Artifacts report these operations in
`active_source_cpu_restore`; registration must not be timed without them.

Build both additive userspace shim ABIs without loading UVM:

```bash
make -C experiments/uvm_512x4m_batch_compare
make -C experiments/uvm_slot_fragmented_plan_probe
```

The batch shim adds
`uvm512_dataset_create_qwen_packed_sparse(expert_ids, count, 128, 512KiB,
6, 3, w13_offset, w2_offset)`. Existing constructors and pointer order are
unchanged.

Before any bank preallocation or plan registration, this arm now reproduces
the serving residency split explicitly:

1. the checksum kernel touches and validates all 1,152 envelopes in the full
   packed `w13+w2` tensors;
2. only the selected cold 512 KiB regions are prefetched back to CPU and the
   stream is synchronized;
3. page-granular residency queries require every cold byte on CPU and every
   remaining hot byte on GPU.

The artifact records this under `registration_residency_setup`, including the
operation order, per-layer checksums, full/cold/hot byte counts, final
residency, and the exact CUDA prefetch call topology. No checksum or other GPU
touch occurs between the final residency validation and preallocation.

## Real packed Qwen sparse layout

This arm accepts arbitrary cold IDs, including consecutive experts. Regions
are emitted in expert-major order (`w13[0..5]`, then `w2[0..2]`) while their VAs
retain the two tensor-major packed strides. Consequently a 2 MiB block may
contain regions from multiple experts, and one plan may contain both complete
and partial blocks. Artifacts include block-entry histograms, cold 64 KiB leaf
mask histograms, preserved-hot-leaf capacity, and exact tensor bases/strides.

`--qwen-base-offset-cycle` makes the four probe layers request all legal
512 KiB-aligned tensor-base residues modulo 2 MiB. This does not alter either
expert stride. It is separate from `--cycle-active-offsets`, which applies only
to independent allocations.

```bash
workloads/vllm/vllm/.venv/bin/python \
  experiments/uvm_slot_fragmented_plan_probe/run_probe.py \
  --layout qwen-packed-sparse --pte-mode partial64k \
  --qwen-base-offset-cycle --cold-expert-ids 0,7,31,64,127 \
  --output /tmp/qwen_packed_sparse_partial64k.json
```

Use a range such as `--cold-expert-ids 5-12` to cover consecutive cold
experts. Per-layer ID syntax is identical to `sparse-experts`.

## Required non-2 MiB-aligned Qwen gate

`--cycle-active-offsets` retains a 2 MiB allocation per envelope but cycles
the active 512 KiB view through offsets `0`, `512 KiB`, `1 MiB`, and `1.5 MiB`.
The dataset ABI returns, reseeds, and checksums that active view rather than
the allocation base. The guard receives both the active pointer and offset, so
its checksum covers the discontiguous complement before and after the active
view; prefix plus suffix remains exactly 1.5 MiB per allocation.

This flag is the required Qwen non-2 MiB-alignment gate. It is explicit so old
zero-offset commands and artifacts remain reproducible. Artifacts report the
observed histogram under `layout_metrics.*.active_offsets` and must contain all
four offsets when at least four envelopes are selected.

Full 128-expert fragmented arm (`1,152` envelopes, `576 MiB`, `1,152` touched
blocks, `9,216` active 64 KiB leaves):

```bash
workloads/vllm/vllm/.venv/bin/python \
  experiments/uvm_slot_fragmented_plan_probe/run_probe.py \
  --layout fragmented --pte-mode partial64k --cycle-active-offsets \
  --output /tmp/fragmented_partial64k.json
```

Sparse, non-contiguous cold-expert arm:

```bash
workloads/vllm/vllm/.venv/bin/python \
  experiments/uvm_slot_fragmented_plan_probe/run_probe.py \
  --layout sparse-experts --pte-mode partial64k --cycle-active-offsets \
  --cold-expert-ids 0,7,31,64,127 \
  --output /tmp/sparse_experts_partial64k.json
```

That five-expert example has 45 envelopes/touched blocks, 22.5 MiB per bank,
360 active 64 KiB leaves, and 1,440 leaves of activated child-table capacity.
`--cold-expert-ids` applies one explicit set to all four layers. To exercise
different cold sets with identical geometry, use:

```bash
workloads/vllm/vllm/.venv/bin/python \
  experiments/uvm_slot_fragmented_plan_probe/run_probe.py \
  --layout sparse-experts --pte-mode partial64k --cycle-active-offsets \
  --cold-expert-ids-by-layer \
  'L0=0,7,31,64,127;L1=1,8,32,65,126;L2=2,9,33,66,125;L3=3,10,34,67,124' \
  --output /tmp/per_layer_sparse_experts_partial64k.json
```

Every layer must be present exactly once and contain the same number of
experts. Without either explicit-ID option, `--cold-expert-count N` generates
four deterministic, different sparse sets for `N < 128`; at `N=128` all four
sets necessarily contain every expert. Artifacts record
`expert_ids_by_layer` and `expert_ids_vary_by_layer` so this coverage is
auditable.

## Historical 4 KiB control

The fragmented 4 KiB control uses the registered shared-push 4 KiB bootstrap
(mode 11) and a mode-20 transition with the 64 KiB policy flag clear. Shadow
leaf prefill and 64 KiB PTE prewarm are not applicable; copy-run caching,
dual-bank copy-first, event ordering, fast validation, metadata recipes, and
shared CE push remain in scope.

`--expect-registration-rejection` is retained only to reproduce artifacts from
an older module which rejected partial 64 KiB plans. It is not a valid arm for
the current partial-64K implementation and must not be compared as a timing
sample.

```bash
workloads/vllm/vllm/.venv/bin/python \
  experiments/uvm_slot_fragmented_plan_probe/run_probe.py \
  --layout fragmented --pte-mode 64k \
  --expect-registration-rejection \
  --output experiments/uvm_slot_fragmented_plan_probe/artifacts/fragmented_64k_reject.json
```

## Minimal performance matrix

Run three rounds of each successful arm after setting module parameters.
`dual_bank_event` and `copy_run_cache` may each be 0 or 1 for ablation, except
that dual-bank copy-first requires the copy cache. Every transition asserts
that `slot_direct_dual_bank_copy_first_calls` is exactly 1 when dual-bank is
enabled and exactly 0 when disabled. The other common settings are legacy
overlap=0, fast-index=1, direct-cutover=1, CUDA-fence=0, and a VIDMEM budget of
at least 1,207,959,552 bytes.

Packed 64 KiB baseline/candidate:

```bash
# Baseline: fast_validation=0, shadow_pte_prefill=0
workloads/vllm/vllm/.venv/bin/python \
  experiments/uvm_slot_fragmented_plan_probe/run_probe.py \
  --layout packed --pte-mode 64k --output /tmp/packed_64k_f0s0_r1.json

# Candidate: fast_validation=1, shadow_pte_prefill=1
workloads/vllm/vllm/.venv/bin/python \
  experiments/uvm_slot_fragmented_plan_probe/run_probe.py \
  --layout packed --pte-mode 64k --output /tmp/packed_64k_f1s1_r1.json
```

Fragmented 4 KiB baseline/candidate:

```bash
# Both arms require shadow_pte_prefill=0. Toggle fast_validation 0 -> 1.
workloads/vllm/vllm/.venv/bin/python \
  experiments/uvm_slot_fragmented_plan_probe/run_probe.py \
  --layout fragmented --pte-mode 4k --output /tmp/fragmented_4k_f0_r1.json

workloads/vllm/vllm/.venv/bin/python \
  experiments/uvm_slot_fragmented_plan_probe/run_probe.py \
  --layout fragmented --pte-mode 4k --output /tmp/fragmented_4k_f1_r1.json
```

Fragmented or sparse partial-64K baseline/candidate uses the same commands
shown above while toggling module parameters around the worker. A useful
ablation is:

| arm | copy cache | dual bank | fast validation | recipe | shadow |
|---|---:|---:|---:|---:|---:|
| P0 registered baseline | 0 | 0 | 0 | 0 | 0 |
| P1 cached copy | 1 | 0 | 0 | 0 | 0 |
| P2 inactive-bank copy-first | 1 | 1 | 0 | 0 | 0 |
| P3 full partial candidate | 1 | 1 | 1 | 1 | 0 |
| P4 shadow-request safety check | 1 | 1 | 1 | 1 | 1 |

PTE prewarm remains 1, legacy overlap remains 0, fast index/direct cutover
remain 1, and CUDA fence remains 0 in every arm. For `fragmented` and
`sparse-experts`, P4 must remain on the live-leaf partial path with zero shadow
hits; it is a safety check, not an expected speedup. A mixed
`qwen-packed-sparse` plan instead requires exactly one shadow hit per complete
2 MiB block while every partial block remains live-leaf. Compare P3 with
packed/64K, fragmented/4K, and the selected sparse expert count using the same
parameter settings.

Compare `steady_ioctl_wall_ns_median` and the nine steady samples at indices
4 through 12. Do not compare a 64 KiB registration rejection against
successful 4 KiB timings.

Pure-userspace geometry tests (no CUDA import or GPU access):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  experiments/uvm_slot_fragmented_plan_probe/test_run_probe.py
```
