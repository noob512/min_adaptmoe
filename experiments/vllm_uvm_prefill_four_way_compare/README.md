# vLLM UVM prefill four-way comparison

This experiment compares, under one frozen Qwen fixture and one timing
boundary:

| Symbol | Prefill data path | Execution |
|---|---|---|
| `N` | vLLM native copy-on-forward (`value.to(cuda)`) | eager |
| `A` | custom UVM original-VA dual-bank alias H2D | eager |
| `G` | custom UVM original-VA dual-bank alias H2D | FULL CUDA Graph |
| `R` | audited original-VA CPU remote READ | eager |
| `C` | audited original-VA CPU remote READ | eager prefill / Graph decode |
| `F` | vLLM native copy-on-forward | FULL CUDA Graph control |

`TRITON_ATTN` is deliberately fixed for every arm.  The installed FA2 backend
advertises only uniform-batch Graph support and makes vLLM downgrade FULL
prefill capture to `FULL_DECODE_ONLY`; Triton attention advertises FULL mixed
prefill/decode support.  Arm G captures exact token sizes `314` and `1217` in
addition to decode sizes `1` and `4`.

`F` is not a fifth reported prefill mechanism. It is a semantic control for
`G`: this vLLM/TRITON configuration produces a different batch-4 greedy token
sequence in FULL mode than in eager mode. The verifier therefore requires
bit-exact output identity between `G` and `F`, and separately among
`N/A/R/C`, so a generic FULL-mode difference is not misattributed to UVM
aliasing.

Commands:

```bash
./experiments/vllm_uvm_prefill_four_way_compare/run_experiment.sh static-preflight

ALLOW_GPU_RUN=1 \
  ./experiments/vllm_uvm_prefill_four_way_compare/run_experiment.sh smoke

ALLOW_GPU_RUN=1 \
  ./experiments/vllm_uvm_prefill_four_way_compare/run_experiment.sh qualify

ALLOW_GPU_RUN=1 \
  ./experiments/vllm_uvm_prefill_four_way_compare/run_experiment.sh formal
```

The wrapper inherits the fail-closed module cycle, GPU-zero checks, process
isolation, exact L0-L3 parameter preparation, and stock-UVM restoration from
the original-VA integration harness.
