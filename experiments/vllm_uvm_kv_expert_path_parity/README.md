# KV / expert path-parity qualification

This experiment checks whether arriving at a joint KV/expert state through the
runtime pressure controller leaves a persistent performance penalty compared
with starting directly in the same state.

It runs three isolated Qwen3-30B-A3B-FP8 processes on one RTX 5090:

- `dynamic`: consume the 8-block GPU backup reserve, grow KV `24 -> 40`,
  offload exactly one expert from every MoE layer, then shrink `40 -> 20` and
  restore the same 48 experts;
- `direct_high`: start at 40 KV blocks and directly offload the exact same
  expert IDs;
- `direct_low`: start at 20 KV blocks with all incremental experts resident.

Autoscaling is frozen during performance sampling. Each final state is sampled
with three requests, five Decode warmups per request, and 16 measured Decode
steps per request. The primary Prefill and first-Decode metrics subtract the
PARK/RESUME topology transition time; resize and expert migration belong to
the conditioning path and are not mixed into steady-state samples.

## Run

```bash
./experiments/vllm_uvm_kv_expert_path_parity/run_experiment.sh static-preflight

RUN_ROOT="$PWD/experiments/vllm_uvm_kv_expert_path_parity/artifacts/my_run" \
ALLOW_GPU_RUN=1 \
./experiments/vllm_uvm_kv_expert_path_parity/run_experiment.sh qualify
```

The qualification switches to the custom UVM module, runs all three isolated
cases, scans the kernel log, restores the stock module, and invokes the
independent verifier.

## Results

Historical qualification artifacts and their standalone result table were
removed by the repository cleanup policy. This directory retains only the
runner, verifier, frozen input plan, and reproduction instructions. New results
must use a new run ID.
