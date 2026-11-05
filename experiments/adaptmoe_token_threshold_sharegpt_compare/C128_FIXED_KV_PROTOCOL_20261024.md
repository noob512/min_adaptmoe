# Fixed 1/2 GiB KV, C128

Rerun only fixed KV at 1 GiB (1,073,741,824 B) and 2 GiB (2,147,483,648 B).
Both use non-managed KV, with dynamic pool/autoscaling/expert swapping off.
Retain cold11 expert coordinates and 4,948,230,144 B offload in both arms.
Retain the established temporary 2 GiB pre-KV native-hole sentinel; release
before serving. Physical blocks including null: 2157 / 4315, respectively.

Initialization qualification found that maximum-token kernel autotuning evicts
hot weights during its own model forward, after the pre-kernel audit. An
after-kernel repair is too late: CUDA failure is already pending. Both arms
therefore use the existing temporary all-remote startup topology for this
kernel_warmup only, via the experiment wrapper. Then reconcile hot residency,
refresh full dual page tables and resume the original copy/alias topology.
Preserve the populated native allocator cache. Record before/after residency
and require every hot byte GPU-resident before sampler dummy128. This is
outside measurement; keep manual fault mode2. No per-step repair, automatic
migration fallback or model/controller modification.

Same frozen DeepSeek-V2-Lite-Chat BF16 ShareGPT input as the previous C128
experiment: 128 requests, 512 prompt tokens and 16 greedy output tokens.
Fixture SHA256: febf1babf6981abb9f1d65cfd600e33e7a8833ad799b87e09e66cc913b8edef7.
Mixed continuous scheduling, max_num_seqs128, max_model_len576, 8192-token
budget, TRITON_MLA, PARK<=16 / RESUME>=128, prefix cache off, enforce_eager,
compilation and CUDA Graph NONE. No extra conditioning request.

Offered QPS 4/8/10/20/30, arrivals at i/QPS. Two independent processes per
arm/rate in 1G/2G/2G/1G order (20 processes, 2560 requests). QPS includes
final drain, TTFT includes admission and queueing, TPOT includes mixed steps.
Block-level usage/concurrency observation is the same O(1) wrapper as before.
Stop on runtime errors, retain artifacts, restore stock UVM and services.
Inhibit automatic exporter restart consistently across all rates.

Check input hashes, arrival offsets, output lengths, output hash differences,
actual KV block count, disabled autoscaling, token-threshold configuration,
transition success, kernel faults and GPU cleanup. Report output differences
explicitly without assuming numerical equivalence. Compare old dynamic K2
only as a historical reference; it is not rerun and its managed allocator,
backup reserve and slightly different usable block count remain confounds.
