# Static VMM aliases inside CUDA Graph replay

This standalone CUDA 12.8 microbenchmark tests a graph-compatible alternative
to changing GPU mappings during replay. It creates all physical allocations and
all virtual aliases before graph instantiation; replay performs ordinary pinned
host-to-device copies and kernels only. No `cuMemMap`, `cuMemSetAccess`, ioctl,
host callback, or page-table mutation occurs inside a replay.

Historical measurements were cleared on 2026-08-23. New measurements must use
the repository-wide EuroSys protocol in
[`../../docs/eurosys_moe_offload_evaluation_plan.md`](../../docs/eurosys_moe_offload_evaluation_plan.md).

The default shape matches the requested layer layout:

- 4 logical layers;
- 1,152 regions per layer;
- 512 KiB per region, or 576 MiB per layer;
- 2 physical 576 MiB device-memory banks;
- 2.25 GiB of fixed pinned source data in total.

The 512 KiB size is the immutable **logical region** used to generate and
validate data. It is independent of `--copy-chunk-kib`, which controls how many
adjacent logical regions are aggregated into one memcpy operation or graph
node.

Each bank is one `cuMemCreate` handle. The same handle is passed to
`cuMemMap` repeatedly:

```text
bank 0 handle (576 MiB)
  +-- upload[0]       RW
  +-- consumer[layer0] RO
  `-- consumer[layer2] RO

bank 1 handle (576 MiB)
  +-- upload[1]       RW
  +-- consumer[layer1] RO
  `-- consumer[layer3] RO
```

Thus all four layer-facing consumer VAs are unique and stable, while their HBM
data backing is only two banks (1.125 GiB). A copy writes through the bank's RW
upload VA and the validation kernel reads the same bytes through that layer's
RO alias. Allocation and mapping sizes are rounded to the device's reported
`CU_MEM_ALLOC_GRANULARITY_MINIMUM`; the default 576 MiB bank is already aligned
on the tested device.

## Explicit dependency graph

Copies can be prefetched on the two bank chains, but consumer kernels follow
Transformer layer order:

```text
reset -> copies L0 -> consumer L0 -> copies L2 ---------> consumer L2
   `---> copies L1 ----------------> consumer L1 -> copies L3 -----> consumer L3

consumer-order edges: consumer L0 -> consumer L1 -> consumer L2 -> consumer L3
```

Every verify node depends directly on every copy node for its layer. Every copy
that reuses a bank depends on the preceding consumer of that bank. A consumer
also directly depends on the preceding layer's consumer. These edges are
required because the upload and consumer pointers have different virtual
addresses, so CUDA cannot infer the physical aliasing itself. The eager path
uses corresponding cross-stream CUDA-event waits.

With defaults, `graph_region_alias` therefore has exactly:

- 4,608 memcpy nodes (`4 * 1,152`);
- 4 validation-kernel nodes;
- 1 error-reset node;
- 4,613 total nodes and 9,219 explicit edges: two edges per memcpy plus three
  consumer-order edges.

Each pinned source region contains a deterministic layer-and-region-specific
pattern. Distinct layer and region patterns expose an early overwrite, missing
region copy, wrong alias, or missing dependency.

## Consumer-check modes

`--verify-mode canary` is the default timing mode. Its one consumer kernel per
layer reads the first, middle, and last word of every logical region. At the
default shape this is 3,456 words per layer instead of 150,994,944 words, so the
timed replay primarily reflects H2D work plus a minimal ordered consumer. It
checks every region but cannot prove that unsampled interior words are intact.

`--verify-mode full` retains the exhaustive correctness mode. Its consumer
kernel scans every logical word. Use it for qualification, not when estimating
copy-node overhead. Both modes retain the same reset, copy, alias-read,
consumer-order, and bank-reuse DAG.

## Copy-node granularity

`--copy-chunk-kib` must be a positive multiple of the fixed 512 KiB logical
region. A final short copy is supported when a custom layer shape is not evenly
divisible. For the default 576 MiB layer:

| `--copy-chunk-kib` | Logical regions/node | Nodes/layer | Total memcpy nodes |
|---:|---:|---:|---:|
| 512 | 1 | 1,152 | 4,608 |
| 2,048 | 4 | 288 | 1,152 |
| 8,192 | 16 | 72 | 288 |
| 32,768 | 64 | 18 | 72 |
| 589,824 | 1,152 | 1 | 4 |

`graph_coalesced_alias` remains an explicit one-copy-per-layer control and
ignores the requested chunk for its graph construction. Its JSON record reports
the effective full-layer chunk.

## Compared modes

The default run emits all four modes sequentially, so their two-bank data
allocations are not resident at the same time:

| Mode | Mapping/data path | Submission granularity |
|---|---|---|
| `graph_region_alias` | VMM RW upload + per-layer RO alias | Graph, configurable chunk |
| `graph_region_direct` | Two ordinary `cudaMalloc` static slots | Graph, configurable chunk |
| `eager_region_alias` | VMM RW upload + per-layer RO alias | Eager `cudaMemcpyAsync`, configurable chunk |
| `graph_coalesced_alias` | VMM RW upload + per-layer RO alias | Graph, one contiguous copy per layer |

The direct static-slot control intentionally exposes only two consumer VAs
(one per bank), whereas the VMM paths expose one immutable consumer VA per
layer. The eager path implements the same two bank chains with nonblocking
copy/verify streams plus explicit CUDA events.

## Build and run

```bash
cd /home/ubuntu/nvidia-uvm-gpu/experiments/p3_static_alias_graph
make
./p3_static_alias_graph > results.jsonl
```

`-arch=native` is used by default. A cross-build can override it, for example:

```bash
make clean
make ARCH_FLAGS='-gencode arch=compute_90,code=sm_90'
```

The default allocates 2.25 GiB of pinned host memory and exercises 2.25 GiB of
logical H2D traffic per replay. A small functional test is available as:

```bash
make smoke
```

Equivalent explicit command:

```bash
./p3_static_alias_graph \
  --layers 4 --banks 2 --regions-per-layer 8 \
  --copy-chunk-kib 2048 --verify-mode canary \
  --verify-blocks 32 --warmups 1 --repeats 2
```

`make smoke` runs this canary test across all four modes, then runs a separate
small `full` scan on `graph_region_alias`.

Run only selected controls with a comma-separated list:

```bash
./p3_static_alias_graph --modes graph_region_alias,graph_coalesced_alias
```

## JSONL contract and timing boundaries

Standard output contains JSONL only: one `config` record, one `sample` record
per measured replay, and one `summary` per mode. Diagnostics go to standard
error. Important fields include:

- `correct` and `error_words`;
- `memcpy_operations`, graph memcpy/kernel/total node counts, and edge count;
- requested/effective copy chunk, copies per layer, and tail-copy size;
- check mode, checked words, and effective consumer-kernel block count;
- `host_submit_ms`, measured through submission of the final timing event;
- `gpu_latency_ms`, from the start event through reset, copies, and consumer
  check;
- `e2e_latency_ms`, wall time through completion of that final event;
- `pipeline_gib_s`, logical H2D bytes divided by `gpu_latency_ms`;
- VMM minimum/recommended granularity and rounded physical-bank size;
- `consumer_read_only`, access validation, per-layer VA uniqueness, and replay
  VA stability;
- physical HBM data backing, mapped VA space, and logical consumer alias bytes.

All timings and `pipeline_gib_s` describe a **single-concurrency replay
pipeline** containing error reset, H2D copies, and ordered consumer checks. They
must not be reported as pure H2D or raw copy-engine measurements. Graph
construction and instantiation are outside replay timing and reported
separately. `config.concurrency` is fixed to `1`. HBM backing fields count the
two weight-data banks; tiny validation counters and CUDA runtime bookkeeping
are intentionally excluded.

All streams, events, graph objects, graph executables, device counters, direct
slots, VMM mappings/reservations/handles, and pinned sources are released in
dependency-safe order before the CUDA primary context is reset.
