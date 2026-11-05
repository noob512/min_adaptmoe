# CUDA/UVM semaphore-pool fence probe

This standalone diagnostic checks the synchronization primitive proposed for
the registered MoE slot-plan fence UAPI. It uses only stock UVM interfaces:

- an existing `/dev/nvidia-uvm` file description which owns the registered-plan
  session and semaphore-pool VA range;
- `UVM_ALLOC_SEMAPHORE_POOL` (ioctl 68) and the required fixed-VA mmap;
- either `cuStreamWriteValue32`/`cuStreamWaitValue32`, or raw-UVA CUDA kernels;
- `UVM_FREE` (ioctl 34) after unmapping the CPU view.

The kernel validates both semaphore addresses against the registered-plan
session fd/VA space. Stream memory operations additionally require CUDA host
registration metadata, which is not available for the plan-owning fd on this
host. The production path therefore launches one-thread CUDA kernels with the
raw semaphore-pool UVA; UVM has already mapped that VA into the session GPU
page table, and kernel argument passing does not impose StreamMemOp's pointer
metadata check.

It validates three directions: a CUDA stream write consumed by another CUDA
stream, a CUDA stream write observed directly by the CPU, and a CPU release
consumed by a CUDA stream wait. It does not load a custom module, change module
parameters, or invoke any experimental MoE ioctl.

Build and run:

```bash
cd /home/ubuntu/nvidia-uvm-gpu/experiments/uvm_slot_cuda_fence_probe
make
./uvm_slot_cuda_fence_probe
./uvm_slot_cuda_fence_probe --raw-kernel
```

The default mode diagnoses the host-registration/StreamMemOp path. The
`--raw-kernel` mode skips host registration entirely and validates GPU-to-CPU
and CPU-to-GPU visibility over the raw UVM pool VA. Success is one JSON object
with `"status": "pass"`. The probe always destroys its
streams, unmaps and frees the semaphore pool, closes the duplicated UVM fd, and
frees the ordinary device allocation used to bootstrap the CUDA context.

The C++ probe establishes address and memory-operation interoperability only.
It does not by itself prove that a CUDA event handle can be consumed by UVM (it
cannot), nor does it exercise an experimental registered-plan ioctl.

## Fenced registered-plan contract

`run_registered_plan.py` exercises command 96 with a real two-bank registered
plan and runs each negative case in an isolated worker. It covers:

- a valid bootstrap and transition ticket, including a same-stream compute
  write and post-commit completion wait;
- a replayed/stale ticket;
- a userspace address outside the UVM semaphore pool;
- aliased compute and completion addresses;
- legacy command 93 while the command-96 gate is enabled;
- an allocation-invalidated execute which accepts the compute fence but must
  not publish a completion ticket.

Each worker first creates and retains one ordinary PyTorch CUDA tensor. This
establishes the same primary CUDA context/allocation environment as vLLM before
the probe selects its UVM fd or creates the semaphore pool. The tensor is
released only after registered-plan and fence-pool cleanup completes.

It requires an already loaded experimental module with the CUDA-fence,
dual-bank, and copy-run-cache parameters enabled. It never loads a module or
writes those parameters itself:

```bash
VLLM_UVM_SLOT_DIRECT_CUDA_FENCE_KERNEL_LIB=\
/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/\
libuvm_slot_cuda_fence_kernels.so \
python3 run_registered_plan.py --case all
```

The CPU-only ABI and result-contract suite is safe to run with no GPU work:

```bash
python3 -m unittest -v test_run_registered_plan.py
```
