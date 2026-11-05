# KTransformers P512/N128 comparison

Replay the identical frozen 128 ShareGPT token-ID prompts (512 input / 16 generated tokens)
at i/QPS, for QPS 4, 8, 10, 20, 30. DeepSeek-V2-Lite-Chat BF16, greedy,
ignore EOS, seed 20260831, C16, context length 576, max prefill and mixed
chunk budget 8192. No client concurrency semaphore: all planned arrivals
are admitted to the backend queue; max_running_requests=16 limits active sequences.
Mixed chunk scheduling enabled, prefix cache disabled, CUDA Graph and torch compile disabled.

Match the AdaptMoE cold plan exactly: 26 MoE layers, each 53 GPU / 11 CPU experts,
286 cold experts, 4,948,230,144 BF16 weight bytes. Frequency masks are verified using
the actual kt_kernel.generate_gpu_experts_masks implementation. Dense layer zero
does not consume the global MoE top-k count. Dynamic expert updates disabled.
BF16 CPU expert kernels, 14 CPU threads, one thread pool, NUMA node0.
kt_gpu_prefill_token_threshold=0 and max_deferred_experts_per_token=0, consistent
with the project's existing DeepSeek KTransformers baseline. No expert relocation sweep.

KV: match vLLM's effective 17,248-token capacity for its 512-MiB budget.
SGLang MLA BF16 pool has 27*(512+64)*2 bytes per token and a one-token dummy slot:
17,249*31,104 bytes (record the exact amount in configuration). page_size=1.
KV effective capacity must equal 17,248 after backend profiling/clamping.
Attention backend is SGLang triton (MLA); original arms use vLLM TRITON_MLA.
Same algorithm family does not imply identical kernels or scheduler implementation.

Two independent engine processes per rate, matching the original sample count.
Run first pass ascending and second descending rates. Normal engine initialization
precedes measurement, with no extra conditioning requests and no sample exclusions.
Before the formal matrix, perform a software-build preflight for the BF16/FP32
cuBLAS extension, without generating a model request. The first-use qualification
trace in attempt-03 discovered a previously absent extension and is preserved
separately (QUALIFICATION.md), explicitly designated before aggregate timings
were available. The formal matrix starts all five rates again after build readiness;
disk build/JIT caches are not cleared, consistent with the historical vLLM environment.
Do not include startup or shutdown in trace metrics. TTFT starts at planned arrival;
TPOT=(last observed token-first observed token)/15. Stream interval1; retain grouped
stream event counts. E2E ends at stream completion. Actual QPS=128/time from first
planned arrival to final request completion; finite trace includes final drain.

Use direct Engine async streaming to avoid HTTP overhead. Unlike the embedded vLLM
runner, SGLang uses tokenizer/scheduler/detokenizer IPC and its own overlap scheduler.
This is a system comparison, not a pure expert-transport microbenchmark.
Record per-request inputs, outputs, raw timings, engine configuration, placement,
commit hashes, existing local SGLang diff, logs, resources and kernel errors.
Verify all 128 requests produce exactly16 tokens and planned arrivals/input hashes match.
Report output hash disagreements explicitly; do not claim strict output equivalence.
Stop matrix immediately on process/configuration/GPU/kernel errors; preserve partial artifacts.
Historical three-arm baselines remain unchanged and are joined by frozen fixture hash.
