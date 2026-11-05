# Bounded-context mixed scheduling: single sweep

The experiment measures actual outcomes. A dynamic-KV win is a hypothesis,
not a selection criterion; unsuccessful or slower cells remain in the record.

Model: Qwen3-30B-A3B-FP8 on one RTX 5090 (32 GB). CUDA Graph is disabled in
both vLLM and KTransformers/SGLang. vLLM uses TRITON_ATTN, eager execution,
chunked mixed scheduling, max_num_seqs=64 and an 8192-token step budget.
AdaptMoE uses token-threshold access switching (PARK<=16, RESUME>=128),
with phase cohorts disabled. KTransformers enables mixed chunk scheduling.

The source is the previously frozen ShareGPT low trace. Select its 64 unique
requests with at least 1024 prompt tokens, preserving order. Take deterministic
128/512/1024-token prefixes for Short/Medium/Long. Generate 512 tokens for every
request, ignoring EOS. These are transformed real prompts with controlled
lengths, not a natural ShareGPT length distribution. Long is relative here:
the maximum prompt+output context is only 1536 tokens.
The 64 source identities produce 63 unique prefix hashes in each trace (two
distinct source requests share their first 1024 tokens). Every arm receives
that identical multiset; prefix/radix caching is disabled in every engine.

All three traces have the same request identities and arrival sequence:

| Arrival phase | Requests | Target arrival rate | Time interval |
|---|---:|---:|---:|
| Low | 8 | 0.25 req/s | [0,32) s |
| High | 48 | 4 req/s | [32,44) s |
| Recovery | 8 | 0.25 req/s | [44,76) s |

There is no client admission cap below the 64 requests: replay connection limit
is 96. Each cell gets four cloned 8-output-token warmup requests, then the
formal trace exactly once. Six configurations per trace, 18 cells total:

| Configuration | Active KV | Cold experts/layer |
|---|---|---|
| AdaptMoE dynamic | K2<->K4 | 14<->28 |
| AdaptMoE fixed K2 | K2 | 14 |
| AdaptMoE fixed K4 | K4 | 28 |
| Native UVA | K4 | Whole-parameter budget 5.907 GiB |
| Native Copy | K4 | Whole-parameter budget 5.907 GiB |
| KTransformers | K4 token capacity | Same cold28 coordinates |

K2 is 1364 policy blocks (~1.998 GiB), K4 is 2730 blocks (~3.999 GiB),
16 tokens/block. AdaptMoE reserves the same K4 virtual pool for all three
controls. Fixed K2 has min=max=1364 and only that allowed target. Fixed K4
uses the established safe K2 initialization and a warmup growth transaction;
a 1% growth watermark and a 1,000,000-step low-pressure streak hold K4 during
formal measurement. Confirm the growth completed before the formal window.
This control is fixed during measurement, not initialized directly at K4.

Dynamic KV uses 80% proactive growth, the existing safe shrink transaction,
immutable CPU weight shadows and full dual-view preparation. Physical hot/KV
residency and the 192 MiB headroom gate remain enabled for resizing. Runtime
full audits and diagnostic read gates stay disabled, as in the prior fast
switching experiment. Expert-plan classes remain coupled to KV capacity.

Report overall completed requests divided by full replay duration, request
mean TTFT, TPOT and E2E, preemptions, mixed-step counts, tier transitions,
and arrival-phase latency. Whole-run QPS includes final drain. Wall-clock
phase completion QPS and arrival-cohort latency have different denominators
and must not be conflated. Compare dynamic against both fixed capacities;
a win over K2 alone does not isolate a dynamic-control benefit from capacity.

Stop the sweep on an error, retain failed artifacts, diagnose before retrying.
One valid replay per cell; no confidence intervals or statistical significance
claims. Different engines/quantized kernels may produce different outputs;
check complete token counts and record that limitation.

Launch:

```bash
bash experiments/eurosys_moe_offload_evaluation/sharegpt_trace/run_bounded_mixed_matrix.sh /absolute/new/artifact-root
```
