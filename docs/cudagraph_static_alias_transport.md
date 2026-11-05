# CUDA Graph-safe static-alias transport

> This document describes the earlier Graph transport. For the current
> token-adaptive eager mixed scheduler and its distinction from separate-phase
> Graphs, read [the project guide](ADAPTMOE_PROJECT_GUIDE.md).

## Scope

The current production experiment uses a fixed topology:

- one GPU;
- vLLM V1 with `CUDAGraphMode.FULL`;
- the driver-backed `uvm_original_va` backend;
- exact-name parameter selection, optionally narrowed by a frozen expert plan;
- topology construction before capture and ordered destruction after all
  captured Graph executables have been released.

Dynamic expert selection, concurrent Graph replay, multi-GPU ownership, and
changing the offload set under KV-cache pressure are not validated.

## Data path

```text
pinned CPU source
      |
      | cudaMemcpyAsync (captured once, replayed by CUDA Graph)
      v
stable RW upload VA ----+
                        | same preallocated device backing
original Parameter VA <-+  (RO consumer mapping)
      |
      v
MoE kernel reads the unchanged Parameter.data_ptr()
```

The selected original parameters are UVM managed allocations. Checkpoint
initialization explicitly prefetches them to the GPU, so fail-fast migration
mode does not depend on a first-touch fault. Finalization then:

1. creates pinned CPU sources;
2. makes the selected original ranges CPU resident;
3. preallocates the fixed device backing banks;
4. maps upload and original consumer VAs to that backing;
5. seals the topology;
6. captures H2D plus model execution in CUDA Graphs.

At steady state, replay performs H2D and compute only. No mapping ioctl is
allowed during capture or replay.

## Driver ABI and lifecycle

The UVM extension adds three session-scoped ioctls:

| Command | Operation | Contract |
|---:|---|---|
| 97 | `UVM_MOE_STATIC_ALIAS_MAP` | Create/append fixed upload and consumer mappings |
| 98 | `UVM_MOE_STATIC_ALIAS_SEAL` | Validate counts and freeze the topology |
| 99 | `UVM_MOE_STATIC_ALIAS_DESTROY` | Remove aliases and optionally release backing |

The driver rejects policy mutation, normal migration/discard/eviction, and CPU
fault servicing for sealed alias ranges. MAP failures roll back; a poisoned or
partially committed session is destroy-only. The userspace ABI wrapper also
rejects all three ioctls while CUDA Graph capture is active.

Teardown ordering is mandatory:

1. destroy and synchronize every Graph that captured an alias VA;
2. issue cmd99 and verify exact alias/backing release;
3. close the slot session;
4. release model/runtime resources.

The vLLM worker follows this ordering and fails closed if Graph destruction or
cmd99 fails.

## Key source files

Driver:

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.h`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_api.h`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_policy.c`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_space.c`

vLLM:

- `vllm/device_allocator/uvm_static_alias.py`: cmd97/98/99 ABI wrapper and
  userspace session cache;
- `vllm/model_executor/models/utils.py`: selection plans, source/arena
  construction, capture certificates, H2D hooks, diagnostics, and teardown;
- `vllm/model_executor/layers/fused_moe/layer.py`: selective original-VA UVM
  allocation and explicit initialization prefetch;
- `vllm/v1/worker/gpu_model_runner.py`: runtime validation, pre-capture seal,
  capture context, VA lifetime guard, and Graph-first shutdown;
- `vllm/v1/worker/gpu_worker.py`: ordered worker shutdown.

Paths under `workloads/vllm/vllm` are relative to that vLLM checkout.

## Runtime configuration

The essential environment for the validated path is:

```text
VLLM_ENABLE_V1_MULTIPROCESSING=0
VLLM_CPU_OFFLOAD_BACKEND=copy
VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE=/path/to/exact_names.txt
VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1
VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va
VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE=
VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288
VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=4k
VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
VLLM_UVM_ORIGINAL_VA_PHASE_MODE=off
VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0
VLLM_CPU_OFFLOAD_COPY_SYNC=0
```

Set `VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE` to a frozen plan for
partial expert slices; leave it empty to transport each selected parameter in
full. The experiment harnesses also set exact expected arena/backing counts so
geometry drift fails before measurement.

The custom module is loaded with:

```text
uvm_moe_slot_pte_remap_enable=1
uvm_moe_slot_pool_vidmem_budget_bytes=<at-least-the-required-backing>
uvm_manual_migration_fault_mode=2
```

Mode 2 logs and cancels an unexpected replayable GPU fault instead of silently
migrating a range, making an incomplete sealed topology fail fast.

## Build and tests

Build the module against the running kernel headers (or pass an explicit
`SYSSRC`):

```bash
make -C kernel-module/nvidia-module modules -j"$(nproc)"
```

Run the userspace contracts from the repository root:

```bash
workloads/vllm/vllm/.venv/bin/python -m pytest -q \
  workloads/vllm/vllm/tests/test_cpu_offload_exact_static.py \
  workloads/vllm/vllm/tests/test_uvm_static_alias_ioctl.py
```

Run static integration checks:

```bash
./experiments/vllm_uvm_original_va_graph_integration/run_integration.sh \
  static-preflight
```

A real GPU run intentionally requires an explicit opt-in and an unused GPU:

```bash
ALLOW_GPU_RUN=1 \
RUN_ROOT=/absolute/new/output/path \
./experiments/vllm_uvm_original_va_graph_integration/run_integration.sh smoke
```

The harness stops only exporters that were initially active, refuses a dirty
GPU/UVM state, loads the custom module, and restores the exact stock UVM module
and service state on every normal/error exit.

## Qualification boundary

The implementation has dedicated harnesses for original-VA stability, exact
H2D accounting, Graph replay, output consistency, and cmd99 resource release.
Historical performance reports and run artifacts were cleared on 2026-08-23;
they are not paper data. New measurements must follow
[`eurosys_moe_offload_evaluation_plan.md`](eurosys_moe_offload_evaluation_plan.md)
and pass its unified provenance and verification gates.

Generated `artifacts/`, local model paths, logs, kernel objects, and calibration
checkouts are intentionally excluded from Git.

## Other paths in the source tree

The CUDA-VMM static-alias allocator is a portable foundation with newly
allocated consumer VAs. It is not the backend used by the formal original-VA
results. `CUDAGraphMode.FULL_SEPARATE` and non-`off` original-VA phase modes
support experimental alias-prefill/remote-decode topology switching; they are
not part of the validated single-topology claim above.
