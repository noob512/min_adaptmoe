# Source preparation validation

This report covers the review source snapshot, not a new performance run.

| Check | Result |
|---|---|
| Python syntax | All retained Python sources parsed |
| Experiment/helper shell syntax | Passed `bash -n` |
| Known project identity and local home paths | No matches in distributed files |
| Frozen input | SHA-256 and 128 distinct 1024-token prompts verified |
| Request preservation | All 128 request records equal to the input before metadata sanitization |
| Expert plans | cold8 is a subset of cold11 in all 26 MoE layers; offload bytes verified |
| CPU preparation | Fresh QPS=1 and QPS=20 inputs generated without GPU/model imports |
| Verifier and driver source guard tests | 13 passed |
| NVIDIA driver build | Full pinned 595.71.05 module build passed on Linux 6.13.12 headers |
| Allocator build | `uvm_allocator.so` and `libuvm_slot_cuda_fence_kernels.so` compiled with CUDA 12.8; fence target `sm_120` |
| Core runtime preservation | vLLM runtime matches the source snapshot except anonymous generated version metadata; UVM implementation unchanged |
| Git/history/build residue | No original Git directory, project remote, compiled output or dependency cache distributed |

One pre-existing source guard test referred to an older split helper and error
branch layout. Its expectations were updated to the current resident-aware
split and telemetry-bearing failure branch; runtime/driver logic was not
changed for that test.

The full `tests/` runtime suite and the vLLM C++/CUDA extension build were not
rerun in this preparation environment: PyTorch and the complete runtime build
dependencies are not installed. An attempted runtime test could not collect
because `torch` was absent. The small 13-test set is independent of those
runtime imports. See the reproduction guide for the full environment.

Build products were generated outside the distributed source directory. No
module was installed or loaded, and no model inference or GPU performance
measurement was performed. Upstream driver compiler warnings were present;
the complete build exited successfully.

`ARTIFACT_MANIFEST.sha256` inventories the distributed files other than itself.
From the artifact root, verify it with:

```bash
sha256sum -c ARTIFACT_MANIFEST.sha256
make check
```
