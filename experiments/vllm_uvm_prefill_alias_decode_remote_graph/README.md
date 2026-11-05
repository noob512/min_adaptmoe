# vLLM original-VA alias-prefill / remote-decode experiment

This experiment answers four questions with the real
Qwen3-30B-A3B-FP8 vLLM scheduler workload:

1. can the driver-backed, original-VA dual-bank copy topology serve prefill and
   then change to audited CPU-backed remote READ PTEs for decode;
2. do the Parameter VAs, driver session state, physical residency, and remote
   mapping metadata remain internally consistent across that cutover;
3. how does alias-copy prefill compare with custom remote-access prefill; and
4. can remote decode use a pre-captured `FULL_DECODE_ONLY` CUDA Graph, and what
   is its steady-state benefit or cost.

The implementation under test uses
`VLLM_UVM_ORIGINAL_VA_PHASE_MODE=prefill_remote_decode|remote_only` and
`uvm_manual_migration_fault_mode=2`. Automatic fault migration is therefore
not a fallback: an omitted or stale PTE makes the process fail.

## Four arms

| Arm | Prefill topology | Decode topology | Decode execution |
|---|---|---|---|
| `H` | sealed dual-bank alias + pinned H2D | original-VA remote READ | eager |
| `G` | sealed dual-bank alias + pinned H2D | original-VA remote READ | CUDA Graph |
| `R` | original-VA remote READ | original-VA remote READ | eager |
| `C` | original-VA remote READ | original-VA remote READ | CUDA Graph |

`G` and `C` capture their decode Graphs while all eight original Parameter VAs
already have audited remote READ mappings. Consequently those Graphs contain
no upload memcpy nodes. After capture, `G` alone restores the sealed alias
topology for serving prefill. Its first decode synchronizes the GPU, issues
cmd99, audits all eight CPU-backed remote mappings, verifies physical
residency, and then replays the already captured remote-only Graph.

Each process runs exactly one request cycle and one batch size. Reusing the
same process for another prefill after the one-way cutover would invalidate the
contract and is intentionally unsupported.

## Timing boundaries

The runner synchronizes CUDA around every scheduler step and records these
non-overlapping views:

- `prefill_step_ms`: the complete scheduler prefill step;
- `first_decode_step_ms`: the first decode, including the H/G cutover;
- `cutover_wall_ms`: the in-hook sync + cmd99 + remote-map + physical-audit
  time reported by the transition itself;
- `first_decode_excluding_cutover_ms`: first-decode wall minus the recorded
  transition (a useful decomposition, not a pure kernel timer);
- `steady_decode_step_ms`: median of later measured decode steps, after
  separate warmup steps;
- `cycle_e2e_ms`: request insertion through final synchronized decode.

The primary comparisons are:

- H/R and G/C `prefill_step_ms`: alias-copy versus remote prefill;
- G/H `steady_decode_step_ms`: Graph benefit in the hybrid phase-switch path;
- C/R `steady_decode_step_ms`: Graph benefit in the pure remote path.

The first-decode comparisons are reported separately and never mixed into the
steady decode median.

## Correctness and metadata gates

The verifier fails unless it observes all of the following:

- the same eight unique `Parameter.data_ptr()` values after load, prefill,
  first decode, and final decode;
- exactly 2,415,919,104 selected bytes and the fixed L0-L3 w13/w2 name set;
- sealed states with a nonzero session and exactly 1,207,959,552 active backing
  bytes, followed by remote states with session 0 and no active driver backing;
- cmd99 releasing exactly 6,912 aliases, 3,623,878,656 alias bytes, and
  1,207,959,552 backing bytes, with every remaining count zero;
- an audited eight-range remote map covering at least 2,415,919,104 bytes,
  with CPU-resident/mapped/covered bytes complete and unknown bytes zero;
- a post-map physical query reporting exactly 2,415,919,104 CPU bytes, zero GPU
  bytes, and zero unknown bytes;
- remote Graph capture topology for G/C and one FULL replay for every decode;
- identical output-token hashes across all four arms for each batch;
- no `manual_migration_fault` kernel record in mode 2;
- Graph reset before tensor/process teardown, no userspace alias session, a
  clean 0 MiB GPU after every process, and exact stock UVM restoration.

Audit processes set `VLLM_CPU_OFFLOAD_COPY_TRACE=1` and additionally prove:

- H/G serving prefill submits exactly 2.25 GiB actual H2D;
- R/C serving prefill submits zero actual H2D;
- every eager remote decode submits zero H2D;
- every captured G/C forward is in `remote` state and submits zero H2D; and
- Graph replay never re-enters the Python offload hook.

Performance processes disable this file-I/O trace. The formal suite includes
one independent audit process plus three independent trace-free performance
processes per arm and batch by default. Performance is reported but is not a
correctness acceptance gate.

## Commands

Static checks do not change kernel modules:

```bash
./experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh static-preflight
```

Run all four arms for batch 1 with short decode loops:

```bash
ALLOW_GPU_RUN=1 \
  ./experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh smoke
```

Run one audited process per arm for batches 1 and 4:

```bash
ALLOW_GPU_RUN=1 \
  ./experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh qualify
```

Run audited cases plus three trace-free processes per arm and batch:

```bash
ALLOW_GPU_RUN=1 \
  ./experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh formal
```

Useful overrides include `MODEL`, `EVAL_FIXTURE`, `RUN_ROOT`, `CUSTOM_UVM`,
`SLOT_BUDGET_BYTES`, `KV_CACHE_BYTES`, `GPU_INDEX`, `CPUSET`,
`FORMAL_PERFORMANCE_REPEATS`, `WARMUP_DECODE_STEPS`, and
`MEASURED_DECODE_STEPS`. `RUN_ROOT` must not already exist.

The harness starts only from the installed stock UVM module and a GPU at
exactly 0 MiB with no compute process or UVM client. It stops only exporter
services that were initially active, installs the local custom module, and
uses an exit trap to restore the exact stock module. It refuses a destructive
module unload if GPU/client state is not clean.

## Artifacts

Every case records its exact environment and command, raw result, offload
manifest, UVM allocation log, scheduler timings, CUDA Graph call deltas,
before/after driver latency statistics, mode-2 kernel log window, resource
usage, and GPU state. The run root also contains module snapshots, input
SHA-256 hashes, the counterbalanced `case_plan.tsv`, execution status, and the
final `verification.json` with process-level samples, medians, and all seven
requested comparisons.

The 2026-08-20 report and locked artifact were cleared on 2026-08-23. Use this
directory only as an implementation/qualification entry point and regenerate
all formal data with a fresh EuroSys run ID.
