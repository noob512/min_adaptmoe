# AdaptMoE

AdaptMoE is a research prototype for memory-constrained MoE inference, built on
vLLM v1 and a modified NVIDIA UVM driver. It keeps expert `Parameter` virtual
addresses stable while adapting both the expert access path and the GPU memory
split between experts and the KV cache.

- **Access controller:** selects explicit H2D copies into reusable GPU buffers
  or direct reads from host memory according to the scheduled token count.
- **Capacity controller:** changes discrete KV/expert plan classes at quiescent
  scheduler boundaries, preserving operator-visible addresses.
- **Driver:** prepares local/remote page-table views and publishes them through
  parent-PDE switching, with synchronization and TLB invalidation.

This repository accompanies *AdaptMoE: Token-Adaptive MoE Inference with Stable
Virtual Addresses and Elastic KV–Expert Memory*. The implementation is an
experimental single-GPU system; the source and supported configurations, rather
than every design statement in the manuscript, define the current interface.

## Start here

- [Project guide: paper-to-code map, controllers, limitations](docs/ADAPTMOE_PROJECT_GUIDE.md)
- [Build, tests, and reproducible mixed-scheduling comparison](docs/REPRODUCING_ADAPTMOE.md)
- [Mixed-scheduling experiment entry point](experiments/adaptmoe_token_threshold_sharegpt_compare/README.md)
- [Experiment and figure index](docs/EXPERIMENTS.md)
- [Earlier separate-scheduling and Graph protocols](experiments/reproduce/README.md)
- [Static-alias CUDA Graph transport](docs/cudagraph_static_alias_transport.md)

## Execution modes

| Mode | Mixed prefill/decode steps | CUDA Graph | Capacity adjustment |
|---|---|---|---|
| `token_threshold` | Supported | Must be disabled | Optional; paired KV/expert classes |
| `prefill_graph_remote_decode` | Rejected | `FULL_SEPARATE` | Qualified fixed plan classes |
| `prefill_remote_decode` | Rejected | Eager phase path | Optional eager plan classes |

The current mixed-scheduling configuration uses two thresholds: PARK at or
below the low threshold, RESUME at or above the high threshold, and retain the
current view between them. PARK changes the access view; it does not free the
shared GPU copy buffer or change KV capacity.

## Repository layout

| Path | Purpose |
|---|---|
| `kernel-module/nvidia-module/` | NVIDIA 595.71.05 driver sources and UVM extensions |
| `workloads/vllm/vllm/` | Modified vLLM engine, allocator, scheduler integration, and tests |
| `experiments/adaptmoe_token_threshold_sharegpt_compare/` | Eager mixed-scheduling Native UVA / Native Copy / fixed and dynamic AdaptMoE comparison |
| `experiments/reproduce/` | Earlier evaluation and Graph reproduction protocols |
| `evaluation/` | Reproducible plotting scripts and frozen, derived measurement tables |
| `tests/` | Focused controller, queue, plan and ioctl regression tests |
| `scripts/` | Source checks and driver assembly/build helpers |
| `docs/` | Architecture and reproduction documentation |
| `baselines/` | Pinned optional baseline versions and measurement patches |

```bash
git clone "$REPOSITORY_URL"
cd min_adaptmoe
```

Optional baseline sources are fetched at the exact revisions recorded in
`baselines/versions.json` with `python3 scripts/fetch_baselines.py`. They are
not needed for the four-mechanism vLLM comparison. Use the bundled vLLM source
for all AdaptMoE runs.

Model weights and the full ShareGPT dataset are supplied separately. Run
directories, profiler captures, compiled modules, local manuscript drafts, and
editor settings are excluded. The frozen token-ID fixture and expert plans
needed for the documented comparison are included. Upstream components retain
their own licenses and attribution.

The source-only validation path is:

```bash
make check
make inputs
```

`make test` runs the focused Python regression suite after the vLLM environment
has been installed. `make driver` prepares the checksum-pinned NVIDIA source
tree and builds the kernel modules; `make allocator` builds the CUDA allocator
helpers. These build steps require matching CUDA, kernel headers,
PyTorch/vLLM dependencies and an NVIDIA driver, but do not install a module.

The plotting-only path needs the packages in `evaluation/requirements.txt` and
does not import vLLM or require a GPU:

```bash
python3 -m pip install -r evaluation/requirements.txt
make FIGURE_PYTHON=python3 figures
```

The output directory is generated under `.build/figures` and is ignored. The
CSV tables in `evaluation/figures/data/` are the checked source inputs for these
plots; their `source_section` fields preserve experiment provenance without
including run logs or compiled output.
