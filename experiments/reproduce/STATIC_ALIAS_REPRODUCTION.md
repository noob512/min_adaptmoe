# Static-alias reproduction

The smallest end-to-end GPU reproduction is the original-VA static-alias
probe. It uses a 512 KiB upload region and one consumer alias, so it does not
load model weights. The probe still exercises the custom UVM module, preallocated
VIDMEM backing, 64K page-table aliases, CUDA Graph H2D transport, sealed policy
guards, destroy, and CPU backing restoration.

The runner writes its result to a new directory and restores the stock UVM
module before returning. It requires a compatible NVIDIA driver/module, `sudo`
without an interactive password, a CUDA-enabled Python environment with the
bundled vLLM source importable, and the allocator helper built in
`workloads/vllm/vllm/uvm_test/uvm_allocator.so`.

```bash
export PYTHON=/path/to/vllm-python
export CUSTOM_UVM=/path/to/nvidia-uvm.ko
experiments/reproduce/run_static_alias_once.sh \
  /absolute/new/output/static-alias.json
```

For a local NVIDIA workspace, the compatible compiled helper and module can be
borrowed without committing them. Set `NVIDIA_UVM_GPU_ROOT` to that workspace:

```bash
export NVIDIA_UVM_GPU_ROOT=/path/to/nvidia-uvm-gpu
export PYTHON="$NVIDIA_UVM_GPU_ROOT/workloads/vllm/vllm/.venv/bin/python"
export CUSTOM_UVM="$NVIDIA_UVM_GPU_ROOT/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko"
cp "$NVIDIA_UVM_GPU_ROOT/workloads/vllm/vllm/uvm_test/uvm_allocator.so" \
  workloads/vllm/vllm/uvm_test/uvm_allocator.so
experiments/reproduce/run_static_alias_once.sh \
  /absolute/new/output/static-alias.json
```

The helper is an external build input and is ignored by Git. A successful run
reports `status: ok`, `pte_mode: 64K`, `remote_cpu_backing_restored: true`,
sealed policy operations rejected with status 23, and post-destroy policy
operations accepted. The measured replay time is hardware and driver dependent.
