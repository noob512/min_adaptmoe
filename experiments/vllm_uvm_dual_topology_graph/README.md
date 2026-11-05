# vLLM original-VA dual-topology CUDA Graph experiment

This experiment tests the reversible same-process mechanism requested for
serving.  The static-alias session now has the topology state machine
`sealed -> remote_parked -> sealed`:

```text
initialization
  sealed original-VA dual-bank aliases
    -> PARK only the read-only consumer PTEs
    -> install audited CPU remote READ PTEs
    -> capture uniform decode Graphs: (1,true), (4,true)
    -> RESUME the retained consumer-alias records
    -> capture non-uniform prefill Graphs: (314,false), (1217,false)

serving
  alias Graph replay for prefill
    -> synchronize
    -> PARK 4,608 read-only consumer aliases
    -> audit CPU remote READ PTEs and physical residency
    -> replay the two preserved uniform remote decode Graph entries
  next prefill epoch
    -> synchronize and RESUME the retained consumer aliases
    -> replay the same preserved non-uniform alias Graph entries

shutdown
  destroy all four Graph entries
    -> cmd99 releases all 6,912 aliases and 1.125 GiB backing
```

The phase mode is `prefill_graph_remote_decode`; the vLLM CUDA Graph mode is
the experimental `FULL_SEPARATE`.  The latter gives mixed/prefill and
uniform-decode batches distinct `BatchDescriptor` keys, so a decode cannot
accidentally dispatch to a Graph captured under alias PTEs.

PARK leaves the 2,304 writable upload aliases, 1.125 GiB backing, session,
immutable consumer records, and all four CUDA Graph entries alive.  RESUME
reinstalls the consumer PTEs from those records without recreating any of
those resources.  The original eight `Parameter.data_ptr()` values never
change.  Only their PTE/backing topology changes.  Automatic UVM fault
migration remains disabled with `uvm_manual_migration_fault_mode=2`.

## Fail-closed certificates

The run is rejected unless it records all of the following:

- an independent remote capture certificate: eight selected-layer forwards,
  zero H2D bytes, and the original-VA fingerprint;
- an independent alias capture certificate: eight selected-layer forwards,
  exactly two logical uploads (4.5 GiB captured H2D), the same VA fingerprint,
  and 1.125 GiB backing;
- an exact partition certificate containing only remote uniform `1/4` and
  alias non-uniform `314/1217` entries;
- one FULL replay for each serving prefill and decode phase;
- exact PARK and RESUME counts proving 4,608 consumer aliases /
  2,415,919,104 bytes switch while 2,304 upload aliases and
  1,207,959,552 bytes of backing remain live;
- a second `PARK -> RESUME -> PARK` cycle with no additional Graph capture;
- complete remote mapping/physical-residency audits, unchanged VAs,
  deterministic output hashes, mode-2 no-fault logs, and shutdown release of
  all four Graphs, aliases, backing, and the session.

## Commands

Static validation does not change the kernel module or use the GPU:

```bash
./experiments/vllm_uvm_dual_topology_graph/run_experiment.sh static-preflight
```

GPU runs are serialized with the same global lock and stock/custom/stock UVM
module lifecycle as the earlier phase-switch experiment:

```bash
ALLOW_GPU_RUN=1 \
  ./experiments/vllm_uvm_dual_topology_graph/run_experiment.sh smoke

ALLOW_GPU_RUN=1 \
  ./experiments/vllm_uvm_dual_topology_graph/run_experiment.sh qualify

ALLOW_GPU_RUN=1 FORMAL_PERFORMANCE_REPEATS=3 \
  ./experiments/vllm_uvm_dual_topology_graph/run_experiment.sh formal
```

`smoke` runs batch 1. `qualify` and `formal` run batch 1 and 4; formal adds
three trace-free performance processes per batch after the audited processes.
The verifier reports prefill, first-decode, steady-decode, cutover, cycle E2E,
RESUME, roundtrip, and CUDA memory summaries without combining results from
separate mechanisms.  Because retained backing is part of the protocol, the
harness uses a 192 MiB minimum post-residency headroom gate.

No GPU result is claimed until a generated `verification.json` has status
`ok`.  In particular, the earlier
`vllm_uvm_prefill_alias_decode_remote_graph` G arm used eager alias prefill and
must not be cited as a measurement of this dual-Graph path.

Historical grouped-PTE reports and artifacts were cleared on 2026-08-23. This
directory remains the implementation/qualification entry point; all new
measurements must use a fresh EuroSys protocol run ID.
