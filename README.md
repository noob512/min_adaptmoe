# AdaptMoE

AdaptMoE is a single-GPU research prototype for memory-constrained MoE
inference. It extends vLLM v1 and NVIDIA UVM to preserve expert parameter
virtual addresses while changing expert access paths and the memory split
between experts and the KV cache.

- The access controller chooses explicit copy or remote CPU-memory reads from
  the total tokens scheduled in the current step, including mixed steps.
- The capacity controller pairs discrete KV capacities with expert offload
  plans and applies changes at safe scheduler boundaries.
- The driver maintains local/remote page-table views and publishes the
  selected view with synchronization and TLB invalidation.

Start with the [paper-to-code guide](docs/ADAPTMOE_PROJECT_GUIDE.md) and
[build and reproduction instructions](docs/REPRODUCING_ADAPTMOE.md).
The [four-arm experiment](experiments/adaptmoe_token_threshold_sharegpt_compare/README.md)
compares Native UVA, Native Copy, fixed-KV AdaptMoE and dynamic-KV AdaptMoE
on the same frozen ShareGPT requests.

| Directory | Contents |
|---|---|
| `kernel-module/` | Modified UVM sources and pinned upstream build instructions |
| `workloads/vllm/vllm/` | Runtime, CUDA sources, build metadata and allocator |
| `experiments/adaptmoe_token_threshold_sharegpt_compare/` | Canonical mixed-scheduling experiment, verifier and frozen trace |
| Other `experiments/` directories | Shared runners and low-level alias probes |
| `tests/` | Focused controller, queue, plan and ioctl regression tests |
| `scripts/` | Source checks and driver assembly/build helpers |

```bash
# Standard-library checks; no model, CUDA installation or GPU required.
make check
make inputs
```

The mixed-scheduling experiment uses eager execution with CUDA Graph disabled.
The runtime also supports a separate Prefill/Decode Graph mode with different
scheduling constraints; see the project guide for supported configurations.

Model weights and the complete ShareGPT source dataset are supplied separately.
The frozen request payload and expert plans are included. Each experiment writes
its measurements to a fresh output directory and reports output consistency
alongside performance metrics. See the reproduction guide for the workload,
memory budgets and measurement definitions.

Third-party license and copyright notices are retained: the root
[LICENSE](LICENSE), vLLM's [Apache license](workloads/vllm/vllm/LICENSE), and
NVIDIA's [license terms](kernel-module/nvidia-module/COPYING).
