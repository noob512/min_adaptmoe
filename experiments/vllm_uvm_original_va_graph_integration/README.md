# vLLM UVM original-VA CUDA Graph integration

Historical mode-2 results were cleared on 2026-08-23. This directory retains
the integration harness only; formal results must be regenerated.

This directory runs the real Qwen3-30B-A3B-FP8 vLLM workload against the
driver-backed static-alias implementation (cmd97 MAP, cmd98 SEAL, cmd99
DESTROY).  It is separate from the CUDA-VMM static-alias experiment so those
published artifacts and schemas remain unchanged.

The experiment compares:

- `E`: vLLM exact-name pinned H2D in eager execution;
- `U`: two driver-backed banks, original managed parameter VAs, and
  `FULL_DECODE_ONLY` CUDA Graph replay.

Both arms use the same frozen token fixture, scheduler-level step timer,
Qwen layers 0-3, and the exact eight `w13_weight`/`w2_weight` parameters.  The
logical copy volume is 2.25 GiB per model forward and the UVM backing budget is
exactly 1.125 GiB by default.

## Safety contract

`run_integration.sh` refuses a GPU run unless all of the following hold:

- `ALLOW_GPU_RUN=1` was explicitly supplied;
- GPU 0 reports exactly 0 MiB, no compute process, and no `/dev/nvidia-uvm`
  client;
- the initially loaded module matches the installed stock module;
- exporter services which were active are stopped before the module cycle;
- the configured slot budget is at least 1,207,959,552 bytes.
- the custom module exposes and is loaded with
  `uvm_manual_migration_fault_mode=2`, so any unexpected replayable GPU fault
  is logged and cancelled instead of being serviced by automatic migration.

The script installs the local custom `nvidia-uvm.ko` with slot remapping
enabled, holds an exclusive experiment lock, and installs an EXIT/INT/TERM
trap.  The trap restores the stock UVM module and restarts only services which
were active before the run.  A dirty GPU/client state blocks module unload
instead of forcing it.

Each vLLM process also performs an in-process ordered shutdown:

1. return from the benchmark so the vLLM engine and CUDA Graph objects die;
2. garbage collect and synchronize CUDA;
3. issue cmd99 with `graphs_destroyed=True`;
4. verify lifecycle `remote`, session id 0, and an empty userspace session
   cache;
5. exit and let the outer harness verify GPU 0 MiB/no client.

## Modes

Run static checks without changing modules:

```bash
./experiments/vllm_uvm_original_va_graph_integration/run_integration.sh static-preflight
```

Run one minimal real-model UVM process:

```bash
ALLOW_GPU_RUN=1 \
  ./experiments/vllm_uvm_original_va_graph_integration/run_integration.sh smoke
```

Run one eager and one UVM qualification process:

```bash
ALLOW_GPU_RUN=1 \
  ./experiments/vllm_uvm_original_va_graph_integration/run_integration.sh qualify
```

Run three independently loaded processes per arm, with counterbalanced order:

```bash
ALLOW_GPU_RUN=1 \
  ./experiments/vllm_uvm_original_va_graph_integration/run_integration.sh formal
```

Useful explicit overrides are `MODEL`, `EVAL_FIXTURE`, `RUN_ROOT`,
`CUSTOM_UVM`, `SLOT_BUDGET_BYTES`, `GPU_INDEX`, `CPUSET`, `PTE_MODE`, and
`REGION_BYTES`.  `RUN_ROOT` must not already exist. The default KV cache is
160 MiB: the former 256 MiB setting left insufficient MoE scratch space once
the 1.125 GiB dual-bank backing and all non-selected resident layers were
accounted for. It can be changed with `KV_CACHE_BYTES`.

The UVM arm fixes the relevant mechanism environment to:

```text
VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1
VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va
VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288
VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=auto
VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0
```

The harness loads the driver with `uvm_manual_migration_fault_mode=2`. This is
stronger than merely observing migration counters: any managed VA omitted from
the sealed static topology makes the run fail instead of silently migrating.

The vLLM compilation contract is `CompilationMode.NONE` plus
`CUDAGraphMode.FULL_DECODE_ONLY`, capture sizes 1 and 4, one GPU, and no DBO.

## Recorded evidence and acceptance

Every run records the exact command and environment, input/module SHA-256
hashes, git status, resource usage, GPU state before/after every process,
module state before/custom/after the cycle, raw JSON results, logs, and a final
`verification.json`.

The verifier fails unless it observes:

- eight managed parameters whose `Parameter.data` VA equals the recorded
  original/consumer VA before and after topology construction;
- four upload arenas, eight unique original VAs, backend `uvm_original_va`, a
  nonzero session id, SEAL success, and exactly 1.125 GiB backing;
- exactly 12 managed allocator calls: eight selected expert parameters plus
  four upload allocations (no unselected expert may silently become managed);
- one FULL Graph replay for every measured/warmup decode step and no Python
  offload-hook re-entry during replay;
- deterministic token hashes within a process and identical hashes across
  eager/UVM processes;
- successful cmd99 teardown, no surviving userspace session, clean per-process
  GPU shutdown, and restoration of the exact stock module hash/srcversion.

Performance is reported but is not an acceptance gate.  `verification.json`
contains process medians for prefill, decode, and end-to-end cycles plus UVM
over eager ratios.  Smoke numbers are functional only; use `formal` for the
performance claim.
