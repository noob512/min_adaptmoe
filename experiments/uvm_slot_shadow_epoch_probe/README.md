# Shadow-epoch dual-bank A/B summarizer

This directory contains a pure-user-space acceptance and comparison tool for
the next registered-plan optimization. It does not load UVM, launch a GPU
workload, or modify vLLM. The input is the JSON produced by
`uvm_slot_dual_bank_event_probe` after each candidate has been exercised by the
same 576 MiB, two-bank event protocol.

The intended A/B arms are:

- arm A: the existing safe dual-bank implementation;
- arm B: CUDA-fence/stream-wait integration plus cached clear recipes, target
  block masks, and preflight shadow epochs.

## Why a separate summarizer

The five-execution sequence is `[0, 1, 2, 3, 0]` over banks
`[0, 1, 0, 1, 0]`. Index 0 is bootstrap and index 1 is the first fill of empty
Bank B. Only indices 2–4 reuse a bank, so each JSON is reduced to the median of
those three steady samples. Three paired rounds are then compared as `A - B`;
a positive delta means the candidate is faster.

The pooled nine-sample median is retained only as a diagnostic. With three
paired rounds the output is descriptive and makes no statistical-significance
claim.

## Safety gates

Every artifact must prove all of the following before it contributes timing:

- 288 blocks, 1152 entries and 576 MiB per bank, with 1152 MiB total backing;
- `dual_bank_event=1`, copy-run cache and PTE prewarm enabled, legacy overlap
  disabled, fast index and direct cutover enabled;
- enough vidmem budget for both banks;
- no compute/copy overlap, and a completed kernel-visible compute event before
  every transition;
- successful ioctl, full-bank byte/range completion, event checksum, plan
  destruction, session close, and zero post-close slots/GPU memory;
- one copy-run cache hit per execution;
- a contiguous, exactly-once generation advance for every successful execute,
  followed by destruction of every registered plan and generation reset on
  final destroy;
- `copy_complete` and `map_complete` telemetry remains zero.

The final rule is intentional. In this probe, `copy` and `map` measure host
submission only. Device completion is joined into `wait`; reporting the former
as completed device work would understate ready-to-compute latency.

Newly instrumented artifacts additionally carry `fast_validation` and
`shadow_pte_prefill` module parameters. When shadow prefill is enabled, every
non-bootstrap transition must report exactly one
`slot_direct_shadow_pte_prefill_calls` hit, all 288 target blocks, and positive
`shadow_pte_prefill` and `shadow_pte_publish` profile timings. Disabled arms
must report zero for all of those fields. A successful fast-validation execute
also proves the registered validation recipe existed: that mode rejects a
missing recipe before transition state.

Archived artifacts without the two new parameters remain readable. They do not
claim shadow/fast evidence, and the tool refuses to compare an archived arm
against a newly instrumented arm because their metric schemas differ.

## Usage

```bash
python3 experiments/uvm_slot_shadow_epoch_probe/summarize.py \
  --arm-a-label safe \
  --arm-a safe_r1.json safe_r2.json safe_r3.json \
  --arm-b-label shadow_epoch \
  --arm-b candidate_r1.json candidate_r2.json candidate_r3.json \
  --expect-a fast_validation=0 \
  --expect-a shadow_pte_prefill=0 \
  --expect-b fast_validation=1 \
  --expect-b shadow_pte_prefill=1 \
  --output /tmp/shadow_epoch_ab_summary.json
```

An undotted `--expect-a/--expect-b` key addresses `module_parameters`. A dotted
key, for example `module_parameters.shadow_epoch_cache=1`, addresses the JSON
root explicitly. Values use JSON scalar parsing.

Recommended artifact names are:

```text
<UTC>_shadow_epoch_safe_r01_<srcversion>.json
<UTC>_shadow_epoch_candidate_r01_<srcversion>.json
<UTC>_shadow_epoch_ab_summary.json
```

Each input path, SHA-256, module `srcversion`, feature evidence, steady samples,
per-round median, paired delta, speedup, and win/tie/loss count is recorded in
the output. Shadow prefill and publish timing are reported independently from
the existing map/copy submission and joined device-wait stages.

## Pure-user-space tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v \
  experiments/uvm_slot_shadow_epoch_probe/test_summarize.py
```

For a read-only smoke test of the parser and current contract, the same three
known-good dual-bank artifacts can be supplied to both arms. The expected
paired delta is zero; this is a tooling check, not a performance result.
