# Validation

The following checks cover source integrity, experiment setup and builds.
They do not include model inference or performance measurements.

| Check | Result |
|---|---|
| Python syntax | All Python sources parsed |
| Experiment/helper shell syntax | Passed `bash -n` |
| Portable paths | No hardcoded local home paths |
| Frozen input | SHA-256 and 128 distinct 1024-token prompts verified |
| Request payloads | 128 frozen request records; text, token IDs and order verified |
| Expert plans | cold8 is a subset of cold11 in all 26 MoE layers; offload bytes verified |
| CPU preparation | Fresh QPS=1 and QPS=20 inputs generated without GPU/model imports |
| Verifier and driver source guard tests | 13 passed |
| NVIDIA driver build | Full pinned 595.71.05 module build passed on Linux 6.13.12 headers |
| Allocator build | `uvm_allocator.so` and `libuvm_slot_cuda_fence_kernels.so` compiled with CUDA 12.8; fence target `sm_120` |

The full `tests/` runtime suite and vLLM C++/CUDA extension build require
PyTorch and the complete runtime dependencies. They were not run in this
validation environment, which lacks those dependencies. The 13-test set above
is independent of runtime imports. See the reproduction guide for the full
environment.

The driver build completed with upstream compiler warnings. Build validation
did not install or load modules.

`SOURCE_MANIFEST.sha256` lists source file checksums, excluding itself.
From the repository root, verify it with:

```bash
sha256sum -c SOURCE_MANIFEST.sha256
make check
```
