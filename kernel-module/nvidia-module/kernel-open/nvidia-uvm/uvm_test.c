/*******************************************************************************
    Copyright (c) 2015-2025 NVIDIA Corporation

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

#include "uvm_api.h"
#include "uvm_test.h"
#include "uvm_test_ioctl.h"
#include "uvm_channel.h"
#include "uvm_global.h"
#include "uvm_va_space.h"
#include "uvm_va_space_mm.h"
#include "uvm_va_range.h"
#include "uvm_va_block.h"
#include "uvm_test_rng.h"
#include "uvm_kvmalloc.h"
#include "uvm_perf_events.h"
#include "uvm_tools.h"
#include "uvm_mmu.h"
#include "uvm_gpu_access_counters.h"
#include "uvm_pmm_gpu.h"
#include "uvm_pmm_sysmem.h"
#include "uvm_migrate_pageable.h"
#include "uvm_test_file.h"
#include "uvm_user_channel.h"
#include "uvm_gpu_non_replayable_faults.h"

static NV_STATUS uvm_test_get_gpu_ref_count(UVM_TEST_GET_GPU_REF_COUNT_PARAMS *params, struct file *filp)
{
    NvU64 retained_count = 0;
    uvm_gpu_t *gpu = NULL;

    uvm_mutex_lock(&g_uvm_global.global_lock);

    gpu = uvm_gpu_get_by_uuid(&params->gpu_uuid);
    if (gpu != NULL)
        retained_count = uvm_gpu_retained_count(gpu);

    uvm_mutex_unlock(&g_uvm_global.global_lock);

    params->ref_count = retained_count;
    return NV_OK;
}

static NV_STATUS uvm_test_peer_ref_count(UVM_TEST_PEER_REF_COUNT_PARAMS *params, struct file *filp)
{
    NvU64 registered_ref_count = 0;
    uvm_gpu_t *gpu0 = NULL;
    uvm_gpu_t *gpu1 = NULL;
    NV_STATUS status = NV_OK;

    uvm_mutex_lock(&g_uvm_global.global_lock);

    gpu0 = uvm_gpu_get_by_uuid(&params->gpu_uuid_1);
    gpu1 = uvm_gpu_get_by_uuid(&params->gpu_uuid_2);

    if (gpu0 && gpu1 && !uvm_gpus_are_smc_peers(gpu0, gpu1)) {
        registered_ref_count = uvm_gpu_peer_ref_count(gpu0, gpu1);
    }
    else {
        status = NV_ERR_INVALID_DEVICE;
    }

    uvm_mutex_unlock(&g_uvm_global.global_lock);

    params->ref_count = registered_ref_count;

    return status;
}

static NV_STATUS uvm_test_make_channel_stops_immediate(UVM_TEST_MAKE_CHANNEL_STOPS_IMMEDIATE_PARAMS *params,
                                                       struct file *filp)
{
    uvm_va_space_get(filp)->user_channel_stops_are_immediate = NV_TRUE;

    return NV_OK;
}

static NV_STATUS uvm_test_nv_kthread_q(UVM_TEST_NV_KTHREAD_Q_PARAMS *params, struct file *filp)
{
    // The nv-kthread-q system returns 0 or -1, because it is not actually
    // part of UVM. UVM needs to run this test, because otherwise, the
    // nv-kthread-q code would not get adequate test coverage. That's because
    // UVM is the first user of nv-kthread-q.
    int result = nv_kthread_q_run_self_test();
    if (result == 0)
        return NV_OK;

    return NV_ERR_INVALID_STATE;
}

// Callers of this function should ensure that node is not NUMA_NO_NODE in order
// to avoid overrunning the kernel's node to cpumask map.
static NV_STATUS uvm_test_verify_bh_affinity(uvm_intr_handler_t *isr, int node)
{
    UVM_ASSERT(node != NUMA_NO_NODE);

    // If the bottom half has been executed but its cpus_used_mask is empty
    // something obviously went wrong. Otherwise, check that the CPUs on which
    // the bottom half was executed is a subset of the NUMA node's cpumask.
    if ((isr->stats.bottom_half_count && cpumask_empty(&isr->stats.cpus_used_mask)) ||
        !cpumask_subset(&isr->stats.cpus_used_mask, cpumask_of_node(node))) {
        UVM_TEST_PRINT("ISR BH cpu mask check failed! BH ran on CPU cores outside NUMA %u\n",
                       node);
        return NV_ERR_INVALID_STATE;
    }
    return NV_OK;
}

static NV_STATUS uvm_test_numa_check_affinity(UVM_TEST_NUMA_CHECK_AFFINITY_PARAMS *params, struct file *filp)
{
    uvm_gpu_t *gpu;
    NV_STATUS status = NV_OK;

    if (!UVM_THREAD_AFFINITY_SUPPORTED())
        return NV_ERR_NOT_SUPPORTED;

    uvm_mutex_lock(&g_uvm_global.global_lock);

    gpu = uvm_gpu_get_by_uuid(&params->gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto unlock;
    }

    // If the GPU is not attached to a NUMA node, there is nothing to do.
    if (gpu->parent->closest_cpu_numa_node == NUMA_NO_NODE) {
        status = NV_ERR_NOT_SUPPORTED;
        goto unlock;
    }

    UVM_ASSERT(gpu->parent->isr.access_counters);
    UVM_ASSERT(gpu->parent->access_counters.buffer);

    uvm_parent_gpu_replayable_faults_isr_lock(gpu->parent);
    status = uvm_test_verify_bh_affinity(&gpu->parent->isr.replayable_faults,
                                          gpu->parent->closest_cpu_numa_node);
    uvm_parent_gpu_replayable_faults_isr_unlock(gpu->parent);
    if (status != NV_OK)
        goto unlock;

    uvm_parent_gpu_non_replayable_faults_isr_lock(gpu->parent);
    status = uvm_test_verify_bh_affinity(&gpu->parent->isr.non_replayable_faults,
                                          gpu->parent->closest_cpu_numa_node);
    uvm_parent_gpu_non_replayable_faults_isr_unlock(gpu->parent);
    if (status != NV_OK)
        goto unlock;

    if (gpu->parent->access_counters_supported) {
        // We only need to test one notification buffer, we pick index 0.
        uvm_access_counters_isr_lock(&gpu->parent->access_counters.buffer[0]);
        status = uvm_test_verify_bh_affinity(&gpu->parent->isr.access_counters[0],
                                              gpu->parent->closest_cpu_numa_node);
        uvm_access_counters_isr_unlock(&gpu->parent->access_counters.buffer[0]);
    }

unlock:
    uvm_mutex_unlock(&g_uvm_global.global_lock);
    return status;
}

static NV_STATUS uvm_test_get_kernel_virtual_address(UVM_TEST_GET_KERNEL_VIRTUAL_ADDRESS_PARAMS *params,
                                                     struct file *filp)
{
    params->addr = (NvU64)uvm_va_space_get(filp);

    return NV_OK;
}

static NV_STATUS uvm_test_get_user_space_end_address(UVM_TEST_GET_USER_SPACE_END_ADDRESS_PARAMS *params,
                                                     struct file *flip)
{
    params->user_space_end_address = TASK_SIZE;

    return NV_OK;
}

static NV_STATUS uvm_test_cgroup_accounting_supported(UVM_TEST_CGROUP_ACCOUNTING_SUPPORTED_PARAMS *params,
                                                      struct file *flip)
{
    return UVM_CGROUP_ACCOUNTING_SUPPORTED() ? NV_OK : NV_ERR_NOT_SUPPORTED;
}

static NV_STATUS uvm_test_set_p2p_suspended(UVM_TEST_SET_P2P_SUSPENDED_PARAMS *params, struct file *filp)
{
    uvm_gpu_t *gpu;
    NV_STATUS status = NV_OK;

    uvm_mutex_lock(&g_uvm_global.global_lock);

    gpu = uvm_gpu_get_by_uuid(&params->gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto unlock;
    }

    if (params->suspended)
        status = uvm_channel_manager_suspend_p2p(gpu->channel_manager);
    else
        uvm_channel_manager_resume_p2p(gpu->channel_manager);

unlock:
    uvm_mutex_unlock(&g_uvm_global.global_lock);
    return status;
}

static NV_STATUS uvm_test_inject_nvlink_error(UVM_TEST_INJECT_NVLINK_ERROR_PARAMS *params, struct file *filp)
{
    uvm_gpu_t *gpu;
    NV_STATUS status = NV_ERR_INVALID_DEVICE;

    uvm_mutex_lock(&g_uvm_global.global_lock);

    gpu = uvm_gpu_get_by_uuid(&params->gpu_uuid);
    if (gpu)
        status = uvm_gpu_inject_nvlink_error(gpu, params->error_type);

    uvm_mutex_unlock(&g_uvm_global.global_lock);
    return status;
}

static NV_STATUS uvm_test_phase_f_ping(UVM_TEST_PHASE_F_PING_PARAMS *params, struct file *filp)
{
    UVM_INFO_PRINT("phase_f_ping begin magic=0x%x\n", params->magic);
    uvm_phase_f_route_log("phase_f_ping begin magic=0x%x\n", params->magic);
    UVM_INFO_PRINT("phase_f_ping end magic=0x%x\n", params->magic);
    uvm_phase_f_route_log("phase_f_ping end magic=0x%x\n", params->magic);
    return NV_OK;
}

static NV_STATUS uvm_test_phase_f_batch_mini(UVM_TEST_PHASE_F_BATCH_MINI_PARAMS *params, struct file *filp)
{
    NV_STATUS status = NV_OK;

    params->success_count = 0;
    params->failure_count = 0;

    UVM_INFO_PRINT("phase_f_batch_mini begin action=%u count=%u base=0x%llx length=%llu\n",
                   params->action,
                   params->count,
                   params->base,
                   params->length);
    uvm_phase_f_route_log("phase_f_batch_mini begin action=%u count=%u base=0x%llx length=%llu\n",
                          params->action,
                          params->count,
                          params->base,
                          params->length);

    if (params->count > 1) {
        status = NV_ERR_INVALID_ARGUMENT;
    }
    else if (params->count == 0) {
        params->success_count = 0;
    }
    else if (params->base == 0 || params->length == 0) {
        status = NV_ERR_INVALID_ARGUMENT;
        params->failure_count = 1;
    }
    else {
        params->success_count = 1;
    }

    UVM_INFO_PRINT("phase_f_batch_mini end action=%u count=%u success=%u failure=%u status=0x%x\n",
                   params->action,
                   params->count,
                   params->success_count,
                   params->failure_count,
                   status);
    uvm_phase_f_route_log("phase_f_batch_mini end action=%u count=%u success=%u failure=%u status=0x%x\n",
                          params->action,
                          params->count,
                          params->success_count,
                          params->failure_count,
                          status);

    return status;
}

static NV_STATUS uvm_test_phase_f_batch_executor(UVM_TEST_PHASE_F_BATCH_EXECUTOR_PARAMS *params,
                                                 struct file *filp)
{
    // 遍历 batch 中每个 range 的索引。
    NvU32 i;

    // 记录第一个失败的 NV_STATUS。
    // 如果所有 range 都成功，最终保持 NV_OK。
    NV_STATUS first_error = NV_OK;

    // 防御性检查：用户态传入的 range 数量不能超过驱动定义的最大值。
    // 如果 count 过大，后续访问 params->ranges/results 可能越界。
    if (params->count > UVM_TEST_PHASE_F_MAX_RANGES) {
        // 打印到内核日志 dmesg。
        UVM_INFO_PRINT("phase_f_batch invalid count=%u max=%u action=%u\n",
                       params->count, UVM_TEST_PHASE_F_MAX_RANGES, params->action);

        // 同时写入新增的 /proc/driver/nvidia-uvm/phase_f_route_log 环形日志。
        uvm_phase_f_route_log("phase_f_batch invalid count=%u max=%u action=%u\n",
                              params->count, UVM_TEST_PHASE_F_MAX_RANGES, params->action);

        return NV_ERR_INVALID_ARGUMENT;
    }

    // 初始化返回给用户态的统计字段。
    params->success_count = 0;
    params->failure_count = 0;

    // 打印 batch executor 入口信息。
    // 如果 count == 0，则不会访问 ranges[0]，而是打印 base=0、len=0。
    UVM_INFO_PRINT("phase_f_batch begin action=%u count=%u async=%u first_base=0x%llx first_len=%llu\n",
                   params->action,
                   params->count,
                   params->async,
                   params->count > 0 ? params->ranges[0].base : 0,
                   params->count > 0 ? params->ranges[0].length : 0);

    uvm_phase_f_route_log("phase_f_batch begin action=%u count=%u async=%u first_base=0x%llx first_len=%llu\n",
                          params->action,
                          params->count,
                          params->async,
                          params->count > 0 ? params->ranges[0].base : 0,
                          params->count > 0 ? params->ranges[0].length : 0);

    // 遍历每个用户态传入的 VA range。
    // 对 ping-ioctl 来说，count == 0，因此这个循环不会执行。
    for (i = 0; i < params->count; ++i) {
        NV_STATUS status = NV_OK;
        NvU64 base = params->ranges[i].base;
        NvU64 length = params->ranges[i].length;

        // 初始化当前 range 的结果。
        params->results[i].status = NV_OK;
        params->results[i].resident_on_count = 0;
        params->results[i].resident_on_cpu = NV_FALSE;
        params->results[i].resident_on_gpu = NV_FALSE;
        params->results[i].queried_bytes = 0;
        params->results[i].resident_gpu_bytes = 0;
        params->results[i].resident_cpu_bytes = 0;
        params->results[i].resident_any_bytes = 0;
        params->results[i].resident_unknown_bytes = 0;

        // base 或 length 非法，直接标记当前 range 失败。
        if (base == 0 || length == 0) {
            status = NV_ERR_INVALID_ARGUMENT;
        }
        // 将 preferred location 设置为 CPU。
        else if (params->action == UVM_TEST_PHASE_F_ACTION_ADVISE_CPU) {
            UVM_SET_PREFERRED_LOCATION_PARAMS *advise_params =
                uvm_kvmalloc_zero(sizeof(*advise_params));

            if (!advise_params) {
                status = NV_ERR_NO_MEMORY;
            }
            else {
                advise_params->requestedBase = base;
                advise_params->length = length;
                advise_params->preferredLocation = NV_PROCESSOR_UUID_CPU_DEFAULT;
                advise_params->preferredCpuNumaNode = NUMA_NO_NODE;

                status = uvm_api_set_preferred_location(advise_params, filp);
                uvm_kvfree(advise_params);
            }
        }
        // 将 range 迁移到 CPU 或 GPU。
        else if (params->action == UVM_TEST_PHASE_F_ACTION_PREFETCH_CPU ||
                 params->action == UVM_TEST_PHASE_F_ACTION_PREFETCH_GPU) {
            UVM_MIGRATE_PARAMS *migrate_params =
                uvm_kvmalloc_zero(sizeof(*migrate_params));

            if (!migrate_params) {
                status = NV_ERR_NO_MEMORY;
            }
            else {
                migrate_params->base = base;
                migrate_params->length = length;
                migrate_params->destinationUuid =
                    params->action == UVM_TEST_PHASE_F_ACTION_PREFETCH_CPU ?
                        NV_PROCESSOR_UUID_CPU_DEFAULT : params->gpu_uuid;
                migrate_params->flags = params->async ? UVM_MIGRATE_FLAG_ASYNC : 0;
                migrate_params->semaphoreAddress = 0;
                migrate_params->semaphorePayload = 0;
                migrate_params->cpuNumaNode = NUMA_NO_NODE;

                status = uvm_api_migrate(migrate_params, filp);
                uvm_kvfree(migrate_params);
            }
        }
        // 查询当前地址驻留在哪些 processor 上。
        else if (params->action == UVM_TEST_PHASE_F_ACTION_QUERY_RESIDENCY) {
            UVM_TEST_VA_RESIDENCY_INFO_PARAMS *query_params =
                uvm_kvmalloc_zero(sizeof(*query_params));
            NvU32 resident_index;

            if (!query_params) {
                status = NV_ERR_NO_MEMORY;
            }
            else {
                query_params->lookup_address = base;
                query_params->is_async = params->async;

                status = uvm_test_va_residency_info(query_params, filp);

                // 查询成功后，把详细 UUID 结果转换成简化字段返回给用户态。
                if (status == NV_OK) {
                    params->results[i].resident_on_count = query_params->resident_on_count;

                    for (resident_index = 0;
                         resident_index < query_params->resident_on_count &&
                         resident_index < UVM_MAX_PROCESSORS;
                         ++resident_index) {
                        if (uvm_uuid_eq(&query_params->resident_on[resident_index],
                                        &NV_PROCESSOR_UUID_CPU_DEFAULT))
                            params->results[i].resident_on_cpu = NV_TRUE;

                        if (uvm_uuid_eq(&query_params->resident_on[resident_index],
                                        &params->gpu_uuid))
                            params->results[i].resident_on_gpu = NV_TRUE;
                    }
                }

                uvm_kvfree(query_params);
            }
        }
        // 未识别的 action。
        else {
            status = NV_ERR_INVALID_ARGUMENT;
        }

        // 写回当前 range 的执行状态。
        params->results[i].status = status;

        // 更新 batch 级别的成功/失败计数。
        if (status == NV_OK) {
            ++params->success_count;
        }
        else {
            ++params->failure_count;

            // 只记录第一个错误，作为函数最终返回值。
            if (first_error == NV_OK)
                first_error = status;
        }
    }

    // 打印 batch executor 结束信息。
    UVM_INFO_PRINT("phase_f_batch end action=%u count=%u success=%u failure=%u first_error=0x%x\n",
                   params->action,
                   params->count,
                   params->success_count,
                   params->failure_count,
                   first_error);

    uvm_phase_f_route_log("phase_f_batch end action=%u count=%u success=%u failure=%u first_error=0x%x\n",
                          params->action,
                          params->count,
                          params->success_count,
                          params->failure_count,
                          first_error);

    // 返回第一个错误；如果没有错误则返回 NV_OK。
    return first_error;
}


long uvm_test_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    // Disable all test entry points if the module parameter wasn't provided.
    // These should not be enabled in a production environment.
    if (!uvm_enable_builtin_tests) {
        UVM_INFO_PRINT("ioctl %d not found. Did you mean to insmod with uvm_enable_builtin_tests=1?\n", cmd);
        uvm_phase_f_route_log("ioctl %d not found. Did you mean to insmod with uvm_enable_builtin_tests=1?\n", cmd);
        return -ENOSYS;
    }
    uvm_phase_f_route_log("ioctl begin cmd=%u\n", cmd);
    switch (cmd)
    {
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_GET_GPU_REF_COUNT,            uvm_test_get_gpu_ref_count);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_RNG_SANITY,                   uvm_test_rng_sanity);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_RANGE_TREE_DIRECTED,          uvm_test_range_tree_directed);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_RANGE_TREE_RANDOM,            uvm_test_range_tree_random);
        UVM_ROUTE_CMD_ALLOC_INIT_CHECK(UVM_TEST_VA_RANGE_INFO,                uvm_test_va_range_info);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_RM_MEM_SANITY,                uvm_test_rm_mem_sanity);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_GPU_SEMAPHORE_SANITY,         uvm_test_gpu_semaphore_sanity);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_PEER_REF_COUNT,               uvm_test_peer_ref_count);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_VA_RANGE_SPLIT,               uvm_test_va_range_split);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_VA_RANGE_INJECT_SPLIT_ERROR,  uvm_test_va_range_inject_split_error);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_PAGE_TREE,                    uvm_test_page_tree);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_CHANGE_PTE_MAPPING,           uvm_test_change_pte_mapping);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_TRACKER_SANITY,               uvm_test_tracker_sanity);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_PUSH_SANITY,                  uvm_test_push_sanity);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_CHANNEL_SANITY,               uvm_test_channel_sanity);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_CHANNEL_STRESS,               uvm_test_channel_stress);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_CE_SANITY,                    uvm_test_ce_sanity);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_HOST_SANITY,                  uvm_test_host_sanity);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_VA_SPACE_MM_OR_CURRENT_RETAIN, uvm_test_va_space_mm_or_current_retain);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_VA_BLOCK_INFO,                uvm_test_va_block_info);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_LOCK_SANITY,                  uvm_test_lock_sanity);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_PERF_UTILS_SANITY,            uvm_test_perf_utils_sanity);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_KVMALLOC,                     uvm_test_kvmalloc);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_PMM_QUERY,                    uvm_test_pmm_query);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_PMM_CHECK_LEAK,               uvm_test_pmm_check_leak);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_PERF_EVENTS_SANITY,           uvm_test_perf_events_sanity);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_PERF_MODULE_SANITY,           uvm_test_perf_module_sanity);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_RANGE_ALLOCATOR_SANITY,       uvm_test_range_allocator_sanity);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_GET_RM_PTES,                  uvm_test_get_rm_ptes);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_FAULT_BUFFER_FLUSH,           uvm_test_fault_buffer_flush);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_INJECT_TOOLS_EVENT,           uvm_test_inject_tools_event);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_INCREMENT_TOOLS_COUNTER,      uvm_test_increment_tools_counter);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_MEM_SANITY,                   uvm_test_mem_sanity);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_MAKE_CHANNEL_STOPS_IMMEDIATE, uvm_test_make_channel_stops_immediate);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_VA_BLOCK_INJECT_ERROR,        uvm_test_va_block_inject_error);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_PEER_IDENTITY_MAPPINGS,       uvm_test_peer_identity_mappings);
        UVM_ROUTE_CMD_ALLOC_INIT_CHECK(UVM_TEST_VA_RESIDENCY_INFO,            uvm_test_va_residency_info);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_PMM_ASYNC_ALLOC,              uvm_test_pmm_async_alloc);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_SET_PREFETCH_FILTERING,       uvm_test_set_prefetch_filtering);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_PMM_SANITY,                   uvm_test_pmm_sanity);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_INVALIDATE_TLB,               uvm_test_invalidate_tlb);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_VA_BLOCK,                     uvm_test_va_block);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_EVICT_CHUNK,                  uvm_test_evict_chunk);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_FLUSH_DEFERRED_WORK,          uvm_test_flush_deferred_work);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_NV_KTHREAD_Q,                 uvm_test_nv_kthread_q);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_SET_PAGE_PREFETCH_POLICY,     uvm_test_set_page_prefetch_policy);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_RANGE_GROUP_TREE,             uvm_test_range_group_tree);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_RANGE_GROUP_RANGE_INFO,       uvm_test_range_group_range_info);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_RANGE_GROUP_RANGE_COUNT,      uvm_test_range_group_range_count);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_GET_PREFETCH_FAULTS_REENABLE_LAPSE,
                                       uvm_test_get_prefetch_faults_reenable_lapse);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_SET_PREFETCH_FAULTS_REENABLE_LAPSE,
                                       uvm_test_set_prefetch_faults_reenable_lapse);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_GET_KERNEL_VIRTUAL_ADDRESS,   uvm_test_get_kernel_virtual_address);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_PMA_ALLOC_FREE,               uvm_test_pma_alloc_free);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_PMM_ALLOC_FREE_ROOT,          uvm_test_pmm_alloc_free_root);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_PMM_INJECT_PMA_EVICT_ERROR,   uvm_test_pmm_inject_pma_evict_error);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_ACCESS_COUNTERS_ENABLED_BY_DEFAULT,
                                       uvm_test_access_counters_enabled_by_default);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_RECONFIGURE_ACCESS_COUNTERS,  uvm_test_reconfigure_access_counters);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_RESET_ACCESS_COUNTERS,        uvm_test_reset_access_counters);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_SET_IGNORE_ACCESS_COUNTERS,   uvm_test_set_ignore_access_counters);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_CHECK_CHANNEL_VA_SPACE,       uvm_test_check_channel_va_space);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_ENABLE_STATIC_PEER_ACCESS,    uvm_test_enable_static_peer_access);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_DISABLE_STATIC_PEER_ACCESS,   uvm_test_disable_static_peer_access);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_GET_PAGE_THRASHING_POLICY,    uvm_test_get_page_thrashing_policy);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_SET_PAGE_THRASHING_POLICY,    uvm_test_set_page_thrashing_policy);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_VA_SPACE_MM_RETAIN,           uvm_test_va_space_mm_retain);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_PMM_CHUNK_WITH_ELEVATED_PAGE, uvm_test_pmm_chunk_with_elevated_page);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_VA_SPACE_INJECT_ERROR,        uvm_test_va_space_inject_error);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_GET_GPU_TIME,                 uvm_test_get_gpu_time);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_PMM_RELEASE_FREE_ROOT_CHUNKS, uvm_test_pmm_release_free_root_chunks);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_DRAIN_REPLAYABLE_FAULTS,      uvm_test_drain_replayable_faults);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_PMA_GET_BATCH_SIZE,           uvm_test_pma_get_batch_size);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_PMM_QUERY_PMA_STATS,          uvm_test_pmm_query_pma_stats);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_NUMA_CHECK_AFFINITY,          uvm_test_numa_check_affinity);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_VA_SPACE_ADD_DUMMY_THREAD_CONTEXTS,
                                       uvm_test_va_space_add_dummy_thread_contexts);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_VA_SPACE_REMOVE_DUMMY_THREAD_CONTEXTS,
                                       uvm_test_va_space_remove_dummy_thread_contexts);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_THREAD_CONTEXT_SANITY,        uvm_test_thread_context_sanity);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_THREAD_CONTEXT_PERF,          uvm_test_thread_context_perf);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_GET_PAGEABLE_MEM_ACCESS_TYPE, uvm_test_get_pageable_mem_access_type);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_TOOLS_FLUSH_REPLAY_EVENTS,    uvm_test_tools_flush_replay_events);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_REGISTER_UNLOAD_STATE_BUFFER, uvm_test_register_unload_state_buffer);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_RB_TREE_DIRECTED,             uvm_test_rb_tree_directed);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_RB_TREE_RANDOM,               uvm_test_rb_tree_random);
        UVM_ROUTE_CMD_STACK_NO_INIT_CHECK(UVM_TEST_GET_USER_SPACE_END_ADDRESS, uvm_test_get_user_space_end_address);
        UVM_ROUTE_CMD_STACK_NO_INIT_CHECK(UVM_TEST_GET_CPU_CHUNK_ALLOC_SIZES, uvm_test_get_cpu_chunk_allocation_sizes);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_VA_RANGE_INJECT_ADD_GPU_VA_SPACE_ERROR,
                                       uvm_test_va_range_inject_add_gpu_va_space_error);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_DESTROY_GPU_VA_SPACE_DELAY,   uvm_test_destroy_gpu_va_space_delay);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_SEC2_SANITY,                  uvm_test_sec2_sanity);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_SEC2_CPU_GPU_ROUNDTRIP,       uvm_test_sec2_cpu_gpu_roundtrip);
        UVM_ROUTE_CMD_STACK_NO_INIT_CHECK(UVM_TEST_CGROUP_ACCOUNTING_SUPPORTED, uvm_test_cgroup_accounting_supported);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_SPLIT_INVALIDATE_DELAY, uvm_test_split_invalidate_delay);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_CPU_CHUNK_API, uvm_test_cpu_chunk_api);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_FORCE_CPU_TO_CPU_COPY_WITH_CE, uvm_test_force_cpu_to_cpu_copy_with_ce);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_VA_SPACE_ALLOW_MOVABLE_ALLOCATIONS,
                                       uvm_test_va_space_allow_movable_allocations);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_SKIP_MIGRATE_VMA, uvm_test_skip_migrate_vma);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_INJECT_TOOLS_EVENT_V2,        uvm_test_inject_tools_event_v2);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_SET_P2P_SUSPENDED,            uvm_test_set_p2p_suspended);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_INJECT_NVLINK_ERROR,          uvm_test_inject_nvlink_error);
        UVM_ROUTE_CMD_STACK_NO_INIT_CHECK(UVM_TEST_FILE_INITIALIZE,           uvm_test_file_initialize);
        UVM_ROUTE_CMD_STACK_NO_INIT_CHECK(UVM_TEST_FILE_UNMAP,                uvm_test_file_unmap);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_QUERY_ACCESS_COUNTERS,        uvm_test_query_access_counters);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_PHASE_F_PING,                 uvm_test_phase_f_ping);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_PHASE_F_BATCH_MINI,           uvm_test_phase_f_batch_mini);
        UVM_ROUTE_CMD_ALLOC_INIT_CHECK(UVM_TEST_PHASE_F_BATCH_EXECUTOR,       uvm_test_phase_f_batch_executor);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_VA_BLOCK_DISCARD_STATUS,      uvm_test_va_block_discard_status);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_PMM_GET_ALLOC_LIST,           uvm_test_pmm_get_alloc_list);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_DUMP_ACCESS_BITS,             uvm_test_dump_access_bits);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_DEAD_CHANNEL,                 uvm_test_dead_channel);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TEST_SET_NON_REPLAYABLE_DELAY,     uvm_test_set_non_replayable_delay);
    }

    return -EINVAL;
}
