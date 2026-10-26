// SPDX-License-Identifier: Apache-2.0
// Tiny CUDA launcher used by the experimental fixed-VA slot fence path.

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace {

__global__ void signal_kernel(uint32_t *address, uint32_t value)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        __threadfence_system();
        atomicExch_system(address, value);
    }
}

__global__ void wait_kernel(uint32_t *address, uint32_t value)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        while (atomicAdd_system(address, 0) < value)
            __nanosleep(128);
        __threadfence_system();
    }
}

}  // namespace

extern "C" int vllm_uvm_slot_fence_signal(uint64_t stream,
                                            uint64_t address,
                                            uint32_t value)
{
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    // The Python admission path has already rejected graph capture using the
    // driver-facing stream query. cudaStreamIsCapturing(0) in this CUDART shim
    // can wait for earlier legacy-default-stream work, which would turn cmd97
    // back into a synchronous host call before the signal kernel is launched.
    signal_kernel<<<1, 1, 0, cuda_stream>>>(
        reinterpret_cast<uint32_t *>(address), value);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int vllm_uvm_slot_fence_wait(uint64_t stream,
                                          uint64_t address,
                                          uint32_t value)
{
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    // Capture validation is intentionally centralized in the Python caller;
    // see the signal launcher above for why a CUDART query is unsafe here.
    wait_kernel<<<1, 1, 0, cuda_stream>>>(
        reinterpret_cast<uint32_t *>(address), value);
    return static_cast<int>(cudaGetLastError());
}
