# P512/N128 AdaptMoE automatic K2/K4/K6 KV experiment

Preserve the frozen 128 ShareGPT prompts, 512 input/16 greedy output tokens,
DeepSeek-V2-Lite-Chat BF16, TRITON_MLA, C16,8192-token scheduler budget,
context576,mixed continuous scheduling,PARK<=16/RESUME>=128,no Graph,
prefix cache disabled, initialization then no extra conditioning.
Set enforce_eager=True explicitly, as required by the existing eager plan-class
protocol; the original fixed runner specified compilation/Graph NONE only.
Five rates4/8/10/20/30, two independent processes per rate. No sample selection.
TTFT is from planned arrival i/QPS; actual QPS includes the complete trace drain.

Dynamic KV uses the previously validated scheduler-boundary automatic controller,
managed lazy KV reservation, synchronized adjacent KV/expert transactions,
80% active high watermark trigger,16 backup+16 safety blocks,4-step minimum
residency,3-step low-pressure streak and existing safe shrink checks.
No time-based manual trigger or synthetic pressure. Keep physical migration,
pre-growth dual preparation/hot reconciliation and192-MiB headroom gate.
Legacy independent expert swapping remains disabled; plan-class transaction
owns expert/KV pairing, matching the existing K2/K4/K6 protocol.

Model-specific MLA block:16 tokens*27 layers*(512+64)*BF16=497,664 B.
The6-GiB virtual pool has12,944 physical blocks including one null block.
Each layer block is18,432 B; pair blocks so all4-KiB UVM query/migration
boundaries align. Set num_gpu_blocks_override=12944 and round each tier's
physical block count down to an even number (policy target excludes null).
This removes at most one block (~0.475MiB) per nominal tier. Retain all audits.

| Tier | Policy target blocks | Active policy KV B | Cold/layer | Main expert B |
|---|---:|---:|---:|---:|
| K2 |4313|2146424832|11|4948230144|
| K4 |8629|4294342656|17|7647264768|
| K6 |12943|6441265152|23|10346299392|

Use the immutable existing DeepSeek plans_adjusted_11_17_23_29 cold_plan_2g/4g/6g
plans; exact bytes, hashes, nesting and settings are recorded in inputs/manifest.json.
K2 preserves exactly the historical fixed512-MiB arm's cold11 coordinates and bytes.
The old Qwen cold14/28/44 and1364/2730/4095-block values do not apply to DeepSeek.

Important expected limitation, declared before measurements: C16 at512+16 tokens
uses at most528 blocks /262,766,592 B (~250.59 MiB). Even the historical512-MiB
fixed KV already fits this active working set. K2 is unlikely to reach80% pressure
and K4/K6 may never activate. More queued requests do not override C16. Report
configured versus actually used tiers separately; a speed difference without
transitions is not evidence of dynamic growth benefit. Keep trace and C16 fixed.

Preserve the original e11 replay algorithm and timing through a wrapper that
records existing controller decisions at each completed step without extra
whole-pool scans. Include observation overhead and all automatic actions.
Stop on real errors; retain failed artifacts and restore stock UVM/services.
Compare with frozen historical fixed512 results. This changes KV capacity,
allocator/controller framework and registered expert plan classes; it is not
an isolated controller ablation or a repeated paired same-day fixed comparison.
Record output hash divergence and do not assert strict output equivalence.
