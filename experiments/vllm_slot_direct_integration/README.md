# vLLM slot-direct integration comparison

This harness compares generic UVM expert restore with the complete vLLM
slot-direct path. It reuses the Qwen3-30B-A3B-FP8 model, frozen token-ID
fixture, exact cold-expert plan, and single-phase benchmark from
`broad_offload_compare`.

There are two GPU commands:

- `run` starts one cold process per arm and validates mode 8 preallocation plus
  mode 10 copy/PTE remap.
- `run-cache` keeps one LLM instance alive for repeated, non-overlapping
  `generate` calls. A deliberately small userspace LRU cache forces mode 19
  slot release, slot reuse, and later mode 8/mode 10 restoration in the same
  process.

The harness does **not** load, unload, or modify kernel modules. Before either
process starts it requires all of the following:

- loaded `nvidia_uvm` srcversion equals the supplied custom `.ko` srcversion;
- `uvm_moe_slot_pte_remap_enable=1`;
- `uvm_moe_slot_pool_vidmem_budget_bytes >= 2684354560` (2.5 GiB);
- no compute process and GPU 0 memory usage at or below
  `GPU_CLEAN_MAX_MIB` (strict default: 0 MiB);
- a complete, non-overlapping Qwen 48-layer by 128-expert plan and a frozen
  prompt-token-ID fixture.

Run CPU-only static checks first:

```bash
experiments/vllm_slot_direct_integration/run_integration.sh static-preflight
```

After an external orchestrator has loaded the custom module with the required
parameters, run the default generic-then-slot pair:

```bash
ALLOW_GPU_RUN=1 \
experiments/vllm_slot_direct_integration/run_integration.sh run
```

Optionally set `EXPECTED_CUSTOM_SRCVERSION` to a separately recorded
`modinfo -F srcversion` value. Without it, the runner still requires the loaded
module to match `CUSTOM_MODULE` exactly.

Set `ORDER=BA` for slot-then-generic. `A` always means generic restore and `B`
always means slot restore. Defaults are concurrency 1, four output tokens, and
zero application warmup requests. Every arm is a newly started process.

After the cold-pair test succeeds, exercise the bounded cache lifecycle:

```bash
ALLOW_GPU_RUN=1 \
ORDER=AB \
SEQUENTIAL_REQUESTS=2 \
SLOT_CACHE_BYTES=268435456 \
SLOT_CACHE_MAX_SLOTS=512 \
experiments/vllm_slot_direct_integration/run_integration.sh run-cache
```

`run-cache` requires `CONCURRENCY=1` because each request must be a separate
`generate` call. `SEQUENTIAL_REQUESTS` must be at least 2. The byte and slot
limits are passed to vLLM as `VLLM_UVM_SLOT_DIRECT_CACHE_BYTES` and
`VLLM_UVM_SLOT_DIRECT_MAX_SLOTS`. The defaults (256 MiB and 512 slots) fit one
complete observed per-layer working set (up to about 148.5 MiB and 297 slots)
while forcing LRU capacity release as execution moves across layers. They are
test limits, not recommended production sizing.

Each arm preserves:

- `command.txt` and `environment.txt`;
- `result.json` and `run.log`;
- `slot_trace.jsonl` (intentionally empty for generic restore);
- raw `latency_stats.before.txt` and `latency_stats.after.txt`;
- process return code/resource use and before/after GPU snapshots.

The root also stores the frozen schedule, input hashes, module state, final
`summary.json`, and `SUMMARY.md`.

The verifier requires exact prompt and generated token identity across arms.
Every slot trace must contain a successful mode 8 preallocation and mode 10
copy/remap with complete per-batch count, byte, status, TLB, and push closure.
It independently checks module-global counter deltas for zero fallback, error,
eviction, and timed mode-10 backing allocation, then reports TTFT, request E2E,
output tokens/s, total placement time/bandwidth, and mode-10 time/bandwidth.

For `run-cache`, verification is stricter: both arms must use one process and
one LLM instance for all sequential requests; every cache snapshot must remain
within both limits; each capacity release must contain a successful mode 19
result; at least one released logical expert/region identity must later pass
mode 8 and mode 10 again; and the final explicit session close must leave zero
cached bytes and slots. Generic and slot arms must produce identical token IDs
on every repeated request. Any fallback, driver eviction, timed allocation,
conflict, partial result, or error fails the run.

The existing runtime placement manifest is deliberately disabled here. Its
current schema accepts `remote_read` placement only and rejects the
`offloaded` state used by both `restore` and `slot_restore`; enabling it would
fail model initialization before the requested comparison. Initial-placement
equivalence is instead fixed by the identical plan SHA, complete topology,
identical placement environment, cold-process boundary, and fail-fast model
initialization. This harness does not claim a separate snapshot-time physical
residency proof.

To re-verify an existing artifact without launching CUDA:

```bash
experiments/vllm_slot_direct_integration/run_integration.sh verify /path/to/run
```

Because `latency_stats` is module-global, do not run another UVM workload at
the same time. Module restoration and the final stock-UVM/GPU-0-MiB check are
the responsibility of the external experiment orchestrator.
