# AdaptMoE: reviewer artifact

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

| Directory | Review content |
|---|---|
| `kernel-module/` | Modified UVM sources and pinned upstream build instructions |
| `workloads/vllm/vllm/` | Runtime, CUDA sources, build metadata and allocator |
| `experiments/adaptmoe_token_threshold_sharegpt_compare/` | Canonical mixed-scheduling experiment, verifier and frozen trace |
| Other retained `experiments/` directories | Shared runner dependencies and low-level alias probes |
| `tests/` | Focused controller, queue, plan and ioctl regression tests |
| `scripts/` | CPU artifact check and driver assembly/build helpers |

```bash
# Standard-library checks; no model, CUDA installation or GPU required.
make check
make inputs
```

The documented mixed-scheduling configuration uses eager execution with CUDA
Graph disabled. The separate Prefill/Decode Graph implementation remains in
the runtime; it has different scheduling constraints and is not the protocol
of this comparison. This artifact is a focused implementation and reproducible
experiment, rather than a reproduction package for every paper figure.

This source snapshot contains no original Git history or project remote.
Model weights, the complete source dataset, historical performance results,
profiler captures and compiled artifacts are excluded. The frozen request
payload and expert plans needed by the documented experiment are included.
New measurements are written to a fresh output directory. Source checks and
successful requests alone do not establish output equivalence or speedups;
inspect the verifier's output comparisons.

Third-party license and copyright notices are retained: the root
[LICENSE](LICENSE), vLLM's [Apache license](workloads/vllm/vllm/LICENSE), and
NVIDIA's [license terms](kernel-module/nvidia-module/COPYING).
