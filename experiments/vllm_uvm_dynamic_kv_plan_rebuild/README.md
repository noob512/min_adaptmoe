# Request-boundary KV-pressure expert-plan rebuild

This experiment validates a same-process serving sequence on Qwen3-30B-A3B-FP8:

```text
initial 2 GiB cold plan -> B1 request
idle scheduler barrier
predicted high next-request KV pressure -> rebuild 4 GiB plan -> B4 request
idle scheduler barrier
predicted low next-request KV pressure -> rebuild 2 GiB plan -> B1 request
```

At each boundary the model runner resets all four concrete CUDA Graphs,
destroys the prior alias session/backing, mutates the long-lived per-module
bindings to the new cold expert slices, restores the new hot complement to
GPU, seals a new prefill upload topology, and captures new remote-decode and
alias-prefill Graphs. The 96 complete expert-weight Parameters and their
virtual addresses remain unchanged.

The pressure signal is deliberately reported in two parts. vLLM allocates the
160 MiB physical KV cache once during initialization, so a completed request
does not return that HBM allocation. The dynamic decision therefore uses the
next request's predicted logical KV blocks: B4 is above the frozen 1,000-token
threshold and selects 4 GiB; B1 is below it and returns to 2 GiB.

Static checks do not load a kernel module or use the GPU:

```bash
./experiments/vllm_uvm_dynamic_kv_plan_rebuild/run_experiment.sh static-preflight
```

The qualification serializes GPU use and performs the custom/stock UVM module
lifecycle:

```bash
ALLOW_GPU_RUN=1 \
  ./experiments/vllm_uvm_dynamic_kv_plan_rebuild/run_experiment.sh qualify
```

The run is accepted only when `verification.json` reports `status=ok`, both
plan generations recapture/replay correctly, output and Parameter VA checks
pass, every remote range is CPU resident with zero unknown bytes, shutdown
releases all Graph/session/backing resources, and the kernel/UVM logs contain
no Xid, PDE, migration-fault, BUG, or Oops signature.

The 2026-08-22 result report was cleared. This README now serves only as the
implementation and rerun entry point for fresh protocol runs.
