# Validation

The following checks cover source integrity, experiment setup and builds.
They do not include model inference or performance measurements.

| Check | Result |
|---|---|
| Python syntax | All Python sources parsed |
| Experiment/helper shell syntax | Passed `bash -n` |
| Portable paths | Canonical runners, build helpers and current documentation resolve paths from the repository or environment; historical protocol notes retain provenance paths |
| Frozen input | SHA-256 and 128 distinct P1024 and P512 prompts verified |
| Request payloads | 128 frozen request records; text, token IDs and order verified |
| Expert plans | cold8 is a subset of cold11 in all 26 MoE layers; offload bytes verified |
| CPU preparation | Fresh QPS=1 and QPS=20 inputs generated without GPU/model imports |
| Verifier and driver source guard tests | 13 passed |
| Focused runtime regression suite | 464 passed |
| Plotting-only evaluation path | 21 CSV tables rendered successfully without GPU/model imports |
| NVIDIA driver build | Previously recorded full pinned 595.71.05 module build passed on Linux 6.13.12 headers; this checkout does not build/load a module automatically |
| Allocator build | Previously recorded CUDA 12.8 build produced `uvm_allocator.so` and `libuvm_slot_cuda_fence_kernels.so`; use `make allocator` to rebuild locally |

The full vLLM C++/CUDA extension build and model runs require the environment
described in the reproduction guide. The focused suite above uses the bundled
vLLM source with the installed runtime dependencies and does not run inference.

The recorded driver build completed with upstream compiler warnings. The
validation commands do not install or load modules. GPU inference and the
historical performance matrices must be rerun with the external model and data
inputs; stored CSV tables are derived inputs for plotting, not a new benchmark.

`SOURCE_MANIFEST.sha256` lists source file checksums, excluding itself.
From the repository root, verify it with:

```bash
sha256sum -c SOURCE_MANIFEST.sha256
make check
```
