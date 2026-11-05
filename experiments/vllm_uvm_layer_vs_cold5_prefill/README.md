# vLLM UVM cold-5 versus whole-layer prefill comparison

This experiment compares four Qwen3-30B-A3B-FP8 FULL-prefill-CUDA-Graph
arms on the same original-VA static-alias implementation:

- `Z`: all-GPU Graph control;
- `C`: five frozen calibration-cold experts in every one of 48 MoE layers;
- `B`: all experts in layers L0-L1 (near-equal-H2D diagnostic);
- `L`: all experts in layers L0-L3 (near-equal theoretical net-capacity
  comparison).

All alias arms use 512 KiB regions, forced 4K PTEs, fail-fast UVM migration
mode 2, the same frozen fixture, and capture sizes 1/4/314/1217. Historical
results were cleared; do not reuse those settings as a paper protocol without
the new fairness and provenance gates.

Static validation:

```bash
./experiments/vllm_uvm_layer_vs_cold5_prefill/run_experiment.sh static-preflight
```

Formal GPU run (the output directory must not already exist):

```bash
ALLOW_GPU_RUN=1 \
RUN_ROOT=/absolute/new/output/path \
./experiments/vllm_uvm_layer_vs_cold5_prefill/run_experiment.sh formal
```
