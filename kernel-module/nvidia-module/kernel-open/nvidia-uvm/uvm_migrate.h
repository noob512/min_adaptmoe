/*******************************************************************************
    Copyright (c) 2018-2025 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#include "uvm_ioctl.h"
#include "uvm_va_block.h"

NV_STATUS uvm_migrate_init(void);
void uvm_migrate_exit(void);

NV_STATUS uvm_migrate_managed_range_to_cpu(uvm_va_space_t *va_space,
                                           NvU64 base,
                                           NvU64 length,
                                           NvU32 migrate_flags);

NV_STATUS uvm_api_phase_f_prefetch_gpu_batch_v2(UVM_PHASE_F_PREFETCH_GPU_BATCH_V2_PARAMS *params,
                                                struct file *filp);

// Wait on a CUDA-stream-produced 32-bit timeline value in a UVM semaphore
// pool. Both user addresses are validated under the VA-space lock before the
// MEMOPS acquire is submitted, so the caller can safely use completion_addr
// for the matching post-commit release below.
NV_STATUS uvm_user_semaphore_wait_and_validate(uvm_va_space_t *va_space,
                                               const NvProcessorUuid *gpu_uuid,
                                               NvU64 compute_addr,
                                               NvU64 completion_addr,
                                               NvU32 payload);

// Publish a CPU-side 32-bit completion value to a previously validated UVM
// semaphore-pool cell. The address is revalidated because callers drop the
// VA-space lock while performing the transition.
NV_STATUS uvm_user_semaphore_release_from_cpu(uvm_va_space_t *va_space,
                                              NvU64 completion_addr,
                                              NvU32 payload);
