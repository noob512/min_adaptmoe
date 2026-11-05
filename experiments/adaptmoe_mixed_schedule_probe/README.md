# AdaptMoE mixed-schedule probe

This qualification probe asks one narrow question: can continuous scheduling,
including a step containing late Prefill requests and older Decode requests,
run while AdaptMoE remains in its copy-backed topology for the whole service?

The tested policy is `always-copy`: the Parameter VAs stay on sealed GPU
aliases and no PARK/RESUME or remote Decode transition occurs. Prefill and the
mixed step execute eagerly with explicit H2D copies; uniform Decode replays
`FULL_DECODE_ONLY` Graphs whose captured nodes also include the H2D copies.
Thus every scheduler step uses the搬运 mechanism, while scheduling remains
continuous. This is not a per-token dual-topology design.

Run the one-process, eight-request probe with:

```bash
ALLOW_GPU_RUN=1 bash experiments/adaptmoe_mixed_schedule_probe/run_probe.sh
```

The verifier requires a real 4-Prefill + 4-Decode mixed scheduler step, sealed
aliases for the whole run, actual H2D records in eager and captured-Graph
forwards, zero remote transitions, eight complete outputs, no GPU/kernel fault,
and exact output-token hashes against the existing same-trace continuous remote
reference. It also reports end-to-end throughput/TTFT/TPOT against that
reference. This is a compatibility/qualification experiment, not an
offered-load or publishable performance matrix.
