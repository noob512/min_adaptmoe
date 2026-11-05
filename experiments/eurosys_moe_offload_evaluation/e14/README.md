# E14: eager UVA per-layer cold experts versus fixed whole layers

E14 compares two near-equal Qwen3-30B-A3B-FP8 placements under increasing
batch concurrency. The current authoritative variant derives its cold experts
from the same 32 frozen E14 requests:

- `D`: 29 lowest combined Prefill+Decode route-count experts from the same
  request set in every MoE layer (6.117 GiB);
- `W`: every expert in fixed layers 0--10 (6.188 GiB).

Both arms use the same original-parameter-VA managed UVA `remote_read`
mechanism for Prefill and Decode. They use `CUDAGraphMode.NONE`,
`CompilationMode.NONE`, and `enforce_eager=True`. The current protocol enables
`VLLM_UVM_REMOTE_ONLY_DIRECT_FORWARD=1` and disables per-forward copy tracing,
so serving calls the original module forward directly after the audited remote
mapping is installed. There is no native copy-on-forward arm and no CUDA Graph arm. Each concurrency point uses that
many distinct frozen requests, each with an 8-token Prefill prefix. The
formal matrix uses B1/B4/B16/B32, three independent cold processes per arm,
and three timed cycles per batch in each process.

The trend gate treats B1 throughput within 25% as the same performance scale,
requires `D/W >= 1.15` in both phases at B32, and compares B32 throughput with
ideal linear scaling from B1. `D` must retain at least 90% efficiency in both
phases while `W` must fall to at most 75%. This is a scaling-efficiency claim:
it does not require whole-layer throughput to stop increasing absolutely.

Qualification run (one cold process per arm; not formal statistics):

```bash
ALLOW_GPU_RUN=1 \
E14_CONFIG=experiments/eurosys_moe_offload_evaluation/e14/config_request_specific.json \
E14_COLD_PLAN=experiments/eurosys_moe_offload_evaluation/e14/inputs/request_specific_cold29_plan.json \
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/e14-<tag> \
./experiments/eurosys_moe_offload_evaluation/e14/run_e14.sh qualification
```

Collect one measured B32 routing window and generate a request-specific plan:

```bash
ALLOW_GPU_RUN=1 \
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/e14-request-routes-<tag> \
./experiments/eurosys_moe_offload_evaluation/e14/run_e14.sh route-profile
```

The generated plan must then be frozen at
`e14/inputs/request_specific_cold29_plan.json`. The current plan covers 1.033%
of the combined route slots in the profiled request window. Because selection
and evaluation use the same request set, this is an in-distribution oracle
experiment rather than evidence of cross-workload generalization.

Static preflight:

```bash
E14_CONFIG=experiments/eurosys_moe_offload_evaluation/e14/config_request_specific.json \
E14_COLD_PLAN=experiments/eurosys_moe_offload_evaluation/e14/inputs/request_specific_cold29_plan.json \
./experiments/eurosys_moe_offload_evaluation/e14/run_e14.sh static-preflight
```

Formal run (the destination must not already exist):

```bash
ALLOW_GPU_RUN=1 \
E14_CONFIG=experiments/eurosys_moe_offload_evaluation/e14/config_request_specific.json \
E14_COLD_PLAN=experiments/eurosys_moe_offload_evaluation/e14/inputs/request_specific_cold29_plan.json \
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/e14-<tag> \
./experiments/eurosys_moe_offload_evaluation/e14/run_e14.sh formal
```

The harness restores the stock UVM module and requires a clean 0-MiB GPU at
both boundaries.  `verification.json`, `summary.csv`, and `RESULTS_E14.md` are
generated only after all process, no-Graph, UVA topology, output, fault, and
cleanup checks pass.
