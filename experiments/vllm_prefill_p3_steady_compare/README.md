# P3 / pinned / managed-UVA steady-prefill comparison

This experiment measures the same frozen four-request, 1217-token prefill and
the same exact L0-L3/all-expert 2.25 GiB selection in all three arms:

- `P`: registered P3 fixed-VA copy;
- `C`: vLLM pinned copy-on-forward;
- `U`: custom managed-UVA remote read.

Each independent process performs one disjoint warmup batch, three evaluation
conditioning batches, then five measured evaluation batches. There is no
intentional inter-batch idle and the benchmark performs no NVML query,
subprocess, file write, result inspection, or serialization between the final
conditioning batch and the measurements. The process median of the five
measurements is the statistical sample; the five batches are not treated as
five independent processes.

The default order contains six balanced three-process blocks:
`PCU CUP UPC UCP PUC CPU`.
The wrapper also pins the legacy reproduction geometry to
`max_model_len=512`, `max_num_batched_tokens=2048`, and `max_num_seqs=4`.

Static validation does not initialize CUDA:

```bash
bash experiments/vllm_prefill_p3_steady_compare/run_steady_compare.sh \
  static-preflight
```

The full GPU experiment still requires the expected custom UVM module and an
otherwise clean GPU:

```bash
ALLOW_GPU_RUN=1 \
bash experiments/vllm_prefill_p3_steady_compare/run_steady_compare.sh run
```

The verifier checks frozen prompts and outputs, exact pinned/UVA placement,
every P3 L0-L3 transition, copy/metadata cache hits, CUDA compute/completion
fences, stream waits, 64 KiB PTE coverage, and the absence of poison, fallback,
or transition errors.
