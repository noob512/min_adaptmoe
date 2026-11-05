# Mixed scheduling: Native UVA, Native Copy, fixed and dynamic AdaptMoE

The public entry point is `run_kv1_compare.sh`. It prepares fresh inputs and
uses the existing measured runner/verifier stack without requiring historical
artifacts. Default: one cold process per arm, QPS=1, no CUDA Graph.

Read [the build and protocol guide](../../docs/REPRODUCING_ADAPTMOE.md) first.

```bash
# CPU-only preparation; choose an output directory that does not exist.
PREPARE_ONLY=1 bash experiments/adaptmoe_token_threshold_sharegpt_compare/run_kv1_compare.sh \
  /absolute/new/prepared-inputs 1

# Real GPU replay: four arms, one process each, no automatic repetitions.
MODEL=/absolute/path/to/DeepSeek-V2-Lite-Chat \
SHAREGPT_DATASET=/absolute/path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
ALLOW_GPU_RUN=1 SAMPLES=1 \
bash experiments/adaptmoe_token_threshold_sharegpt_compare/run_kv1_compare.sh \
  /absolute/new/artifacts/kv1-qps1 1
```

The trace contains 128 requests, P1024/O16, with planned arrival `i / QPS`.
C128, 8192-token scheduling budget, TRITON_MLA, eager mixed scheduling and
disabled prefix caching are common to every arm. AdaptMoE uses PARK≤16 and
RESUME≥128. Fixed usable KV is 34,480 tokens; dynamic KV can grow from K1 to K2
by changing the expert plan from cold8 to cold11.

The Native arms use the same whole Parameters and offload 2.56% more weight
bytes than AdaptMoE cold8. These are system configurations, not a perfectly
isolated comparison of copy mechanisms. Actual QPS includes final drain;
queueing, real capacity changes and output drift remain in the reports.

`run_compare.sh` manages module switching and cleanup. `five_arm_k1_runner.py`
aligns Native KV capacity; the other runner wrappers observe the existing
capacity policy. `five_arm_k1_compare.py`, `qps30_k1_k2_compare.py` and the
verification helpers are retained as dependencies despite historical filenames.
The entry point runs four vLLM arms; it does not run KTransformers.

The frozen input is in `inputs/sharegpt_deepseek_p1024_n128_20261024.json`.
Its SHA-256 is
`a4cc14060c94efb8c86f0524f2c49fbf85c0ff284eb12636e1e5c133d85bb9f2`.
The ShareGPT V3 source SHA-256 is
`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`.
The original model snapshot was
`85864749cd611b4353ce1decdb286193298f64c7`.
Raw runs, generated plans/settings, profiler files and report snapshots stay
local. Request success does not imply token-by-token numerical equivalence;
inspect `output_checks.csv` and the final status.
