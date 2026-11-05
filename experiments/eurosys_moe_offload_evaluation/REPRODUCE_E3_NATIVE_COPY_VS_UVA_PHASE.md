# Reproduce E3 native copy-on-forward vs pinned-UVA phase crossover

## Experiment contract

- Model: Qwen3-30B-A3B-FP8 snapshot `d206ba732169f29bb77fbf80fc2c4b81d4d30782`.
- Mechanisms: native vLLM `copy` and pinned `uva` only.
- Workloads: 128→32, 512→128, 2048→128, 8192→128, and 512→512.
- Batch sizes: B1 and B8.
- One conditioning cycle and one measured cycle per process.
- No HTTP or EngineCore multiprocessing.
- The complete batch is inserted before the first engine step.
- Exactly one complete Prefill step, followed by cached-only one-token Decode steps.
- Eager execution and `VLLM_ATTENTION_BACKEND=FLASH_ATTN`.

## Preconditions

Run from the repository root. Confirm that GPU 0 has no compute process or allocated memory and that the model snapshot and five frozen evaluation fixtures exist. Do not reuse an existing artifact path; the runner fails closed if it already exists.

## Command

```bash
ALLOW_GPU_RUN=1 \
  experiments/eurosys_moe_offload_evaluation/run_e3_native_copy_vs_uva_phase.sh \
  experiments/eurosys_moe_offload_evaluation/artifacts/e3-native-copy-vs-uva-phase-<new-attempt>
```

The exact command, environment, result, manifest, NVML trace, log, and return code for every arm-cell are stored in its `cases/<workload>_b<batch>_<arm>/` directory.

## Important failure guards

- Do not switch back to `TRITON_ATTN` for the 8192-token cells: this platform previously produced TMA descriptor/illegal-address failures at that scale. `FLASH_ATTN` is a shared control for both arms.
- Do not enable chunked Prefill. The contract requires one complete Prefill scheduler step.
- Do not submit B8 as two cohorts. The runner validates eight new requests in the Prefill step and eight cached requests in every Decode step.
- Keep the paired `cpu_offload_gb` value identical. Native vLLM rounds at whole-Parameter boundaries, so compare realized manifest signatures rather than only requested GiB.
- Large cells need larger offload budgets: 7.875 GiB for Medium/B8 and RAG/B1; 13.5 GiB for RAG/B8. Reducing these values can make model load, KV allocation, or Prefill workspace allocation fail.
- Each process must leave the GPU at 0 MiB before the next cell. Treat a surviving process or allocation as a failed cleanup rather than starting the next arm.

## Acceptance

There must be 20 zero return codes and 20 `status=ok` results. For every copy/UVA pair, output hashes and realized Parameter signatures must match. Logs must contain no OOM, CUDA illegal access, Xid, PTE/PDE fault, or traceback. Final GPU use must be 0 MiB.
