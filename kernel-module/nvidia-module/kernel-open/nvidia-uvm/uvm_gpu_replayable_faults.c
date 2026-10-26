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

#include "linux/kernel.h"
#include "linux/sched.h"
#include "linux/sort.h"
#include "nv_uvm_interface.h"
#include "uvm_common.h"
#include "uvm_api.h"
#include "uvm_linux.h"
#include "uvm_global.h"
#include "uvm_gpu_replayable_faults.h"
#include "uvm_hal.h"
#include "uvm_kvmalloc.h"
#include "uvm_tools.h"
#include "uvm_va_block.h"
#include "uvm_va_range.h"
#include "uvm_va_space.h"
#include "uvm_va_space_mm.h"
#include "uvm_procfs.h"
#include "uvm_perf_thrashing.h"
#include "uvm_gpu_non_replayable_faults.h"
#include "uvm_ats_faults.h"
#include "uvm_test.h"

// The documentation at the beginning of uvm_gpu_non_replayable_faults.c
// provides some background for understanding replayable faults, non-replayable
// faults, and how UVM services each fault type.

// The HW fault buffer flush mode instructs RM on how to flush the hardware
// replayable fault buffer; it is only used in Confidential Computing.
//
// Unless HW_FAULT_BUFFER_FLUSH_MODE_MOVE is functionally required (because UVM
// needs to inspect the faults currently present in the HW fault buffer) it is
// recommended to use HW_FAULT_BUFFER_FLUSH_MODE_DISCARD for performance
// reasons.
typedef enum
{
    // Flush the HW fault buffer, discarding all the resulting faults. UVM never
    // gets to see these faults.
    HW_FAULT_BUFFER_FLUSH_MODE_DISCARD,

    // Flush the HW fault buffer, and move all the resulting faults to the SW
    // fault ("shadow") buffer.
    HW_FAULT_BUFFER_FLUSH_MODE_MOVE,
} hw_fault_buffer_flush_mode_t;

#define UVM_PERF_REENABLE_PREFETCH_FAULTS_LAPSE_MSEC_DEFAULT 1000

// Lapse of time in milliseconds after which prefetch faults can be re-enabled.
// 0 means it is never disabled
static unsigned uvm_perf_reenable_prefetch_faults_lapse_msec = UVM_PERF_REENABLE_PREFETCH_FAULTS_LAPSE_MSEC_DEFAULT;
module_param(uvm_perf_reenable_prefetch_faults_lapse_msec, uint, S_IRUGO);

#define UVM_PERF_FAULT_BATCH_COUNT_MIN 1
#define UVM_PERF_FAULT_BATCH_COUNT_DEFAULT 32

// Number of entries that are fetched from the GPU fault buffer and serviced in
// batch
static unsigned uvm_perf_fault_batch_count = UVM_PERF_FAULT_BATCH_COUNT_DEFAULT;
module_param(uvm_perf_fault_batch_count, uint, S_IRUGO);

#define UVM_PERF_FAULT_REPLAY_POLICY_DEFAULT UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH

// Policy that determines when to issue fault replays
static uvm_perf_fault_replay_policy_t uvm_perf_fault_replay_policy = UVM_PERF_FAULT_REPLAY_POLICY_DEFAULT;
module_param(uvm_perf_fault_replay_policy, uint, S_IRUGO);

#define UVM_PERF_FAULT_REPLAY_UPDATE_PUT_RATIO_DEFAULT 50

// Reading fault buffer GET/PUT pointers from the CPU is expensive. However,
// updating PUT before flushing the buffer helps minimizing the number of
// duplicates in the buffer as it discards faults that were not processed
// because of the batch size limit or because they arrived during servicing.
// If PUT is not updated, the replay operation will make them show up again
// in the buffer as duplicates.
//
// We keep track of the number of duplicates in each batch and we use
// UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT for the fault buffer flush after if the
// percentage of duplicate faults in a batch is greater than the ratio defined
// in the following module parameter. UVM_GPU_BUFFER_FLUSH_MODE_CACHED_PUT is
// used, otherwise.
static unsigned uvm_perf_fault_replay_update_put_ratio = UVM_PERF_FAULT_REPLAY_UPDATE_PUT_RATIO_DEFAULT;
module_param(uvm_perf_fault_replay_update_put_ratio, uint, S_IRUGO);

#define UVM_PERF_FAULT_MAX_BATCHES_PER_SERVICE_DEFAULT 4

#define UVM_PERF_FAULT_MAX_THROTTLE_PER_SERVICE_DEFAULT 5

// Maximum number of batches to be processed per execution of the bottom-half
static unsigned uvm_perf_fault_max_batches_per_service = UVM_PERF_FAULT_MAX_BATCHES_PER_SERVICE_DEFAULT;
module_param(uvm_perf_fault_max_batches_per_service, uint, S_IRUGO);

// Maximum number of batches with thrashing pages per execution of the bottom-half
static unsigned uvm_perf_fault_max_throttle_per_service = UVM_PERF_FAULT_MAX_THROTTLE_PER_SERVICE_DEFAULT;
module_param(uvm_perf_fault_max_throttle_per_service, uint, S_IRUGO);

// Under heavy oversubscription a single replayable fault service pass can keep
// mmap_lock held for read long enough to block ordinary userspace mmap/brk and
// teardown writers. Periodically drop the VA-space and mmap read locks between
// serviced fault groups to keep forward progress fair.
#define UVM_PERF_FAULT_MMAP_LOCK_YIELD_INTERVAL_DEFAULT 4
static unsigned uvm_perf_fault_mmap_lock_yield_interval = UVM_PERF_FAULT_MMAP_LOCK_YIELD_INTERVAL_DEFAULT;
module_param(uvm_perf_fault_mmap_lock_yield_interval, uint, S_IRUGO);

static unsigned uvm_perf_fault_coalesce = 1;
module_param(uvm_perf_fault_coalesce, uint, S_IRUGO);

#define UVM_MANUAL_MIGRATION_FAULT_MODE_OFF       0
#define UVM_MANUAL_MIGRATION_FAULT_MODE_OBSERVE   1
#define UVM_MANUAL_MIGRATION_FAULT_MODE_FAIL_FAST 2

static unsigned uvm_manual_migration_fault_mode = UVM_MANUAL_MIGRATION_FAULT_MODE_OFF;
module_param(uvm_manual_migration_fault_mode, uint, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_manual_migration_fault_mode,
                 "Manual migration fault mode for managed GPU replayable faults: "
                 "0=normal UVM service, 1=observe/log and continue, "
                 "2=allow mapping-only service for pages already resident on the faulting GPU, "
                 "otherwise log and cancel instead of migrating. Default: 0.");

// Per-page fault logging can produce hundreds of thousands of kernel records
// during an observe-mode diagnostic. Keep the state-machine protection active
// by default, but require an explicit opt-in for verbose fault attribution.
static unsigned uvm_manual_migration_fault_log_enable;
module_param(uvm_manual_migration_fault_log_enable, uint, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_manual_migration_fault_log_enable,
                 "Log manual migration GPU fault observations to phase_f_route_log and the kernel log. Default: 0.");

static bool manual_migration_fault_mode_enabled(void)
{
    return uvm_manual_migration_fault_mode == UVM_MANUAL_MIGRATION_FAULT_MODE_OBSERVE ||
           uvm_manual_migration_fault_mode == UVM_MANUAL_MIGRATION_FAULT_MODE_FAIL_FAST;
}

static void manual_migration_fault_log(uvm_gpu_t *gpu,
                                       uvm_va_range_managed_t *managed_range,
                                       uvm_fault_buffer_entry_t *fault_entry,
                                       uvm_page_index_t page_index,
                                       bool gpu_resident,
                                       bool cpu_resident,
                                       bool gpu_read_mapped,
                                       uvm_processor_id_t selected_residency,
                                       const char *action)
{
    if (!uvm_manual_migration_fault_log_enable)
        return;

    uvm_phase_f_route_log("manual_migration_fault mode=%u action=%s fault_addr=0x%llx access=%s "
                          "gpu_id=%u pid=%d range_base=0x%llx range_len=%llu page_index=%u "
                          "gpu_resident=%u cpu_resident=%u gpu_read_mapped=%u selected_residency=%u\n",
                          uvm_manual_migration_fault_mode,
                          action,
                          fault_entry->fault_address,
                          uvm_fault_access_type_string(fault_entry->fault_access_type),
                          uvm_id_value(gpu->id),
                          task_tgid_nr(current),
                          managed_range ? managed_range->va_range.node.start : 0,
                          managed_range ? uvm_va_range_size(&managed_range->va_range) : 0,
                          page_index,
                          gpu_resident,
                          cpu_resident,
                          gpu_read_mapped,
                          uvm_id_value(selected_residency));
    UVM_INFO_PRINT("manual_migration_fault mode=%u action=%s fault_addr=0x%llx access=%s gpu_id=%u pid=%d range_base=0x%llx range_len=%llu page_index=%u gpu_resident=%u cpu_resident=%u gpu_read_mapped=%u selected_residency=%u\n",
                   uvm_manual_migration_fault_mode,
                   action,
                   fault_entry->fault_address,
                   uvm_fault_access_type_string(fault_entry->fault_access_type),
                   uvm_id_value(gpu->id),
                   task_tgid_nr(current),
                   managed_range ? managed_range->va_range.node.start : 0,
                   managed_range ? uvm_va_range_size(&managed_range->va_range) : 0,
                   page_index,
                   gpu_resident,
                   cpu_resident,
                   gpu_read_mapped,
                   uvm_id_value(selected_residency));
}

static unsigned uvm_perf_fault_log_addresses;
module_param(uvm_perf_fault_log_addresses, uint, S_IRUGO | S_IWUSR);

static unsigned uvm_perf_fault_log_destination;
module_param(uvm_perf_fault_log_destination, uint, S_IRUGO | S_IWUSR);

static unsigned uvm_perf_fault_log_counters;
module_param(uvm_perf_fault_log_counters, uint, S_IRUGO | S_IWUSR);

static unsigned uvm_perf_fault_kv_range_enable;
module_param(uvm_perf_fault_kv_range_enable, uint, S_IRUGO | S_IWUSR);

static unsigned long long uvm_perf_fault_kv_start;
module_param(uvm_perf_fault_kv_start, ullong, S_IRUGO | S_IWUSR);

static unsigned long long uvm_perf_fault_kv_end;
module_param(uvm_perf_fault_kv_end, ullong, S_IRUGO | S_IWUSR);

bool uvm_perf_fault_address_is_in_kv_cache(NvU64 fault_address)
{
    NvU64 start = (NvU64)uvm_perf_fault_kv_start;
    NvU64 end = (NvU64)uvm_perf_fault_kv_end;

    if (!uvm_perf_fault_kv_range_enable)
        return false;

    if (start == 0 || end == 0 || end < start)
        return false;

    return fault_address >= start && fault_address <= end;
}

static NvU64 percent_x100_u64(NvU64 numerator, NvU64 denominator)
{
    if (denominator == 0)
        return 0;

    return (numerator * 10000ULL) / denominator;
}

static void log_replayable_fault_entry(uvm_parent_gpu_t *parent_gpu,
                                       const uvm_fault_buffer_entry_t *fault_entry,
                                       NvU64 fault_address_raw)
{
    if (uvm_perf_fault_log_destination == 1) {
        trace_printk("Replayable fault GPU %s: raw=0x%llx page=0x%llx type=%s access=%s utlb=%u gpc=%u client=%u ve=%u\n",
                     uvm_parent_gpu_name(parent_gpu),
                     fault_address_raw,
                     fault_entry->fault_address,
                     uvm_fault_type_string(fault_entry->fault_type),
                     uvm_fault_access_type_string(fault_entry->fault_access_type),
                     fault_entry->fault_source.utlb_id,
                     fault_entry->fault_source.gpc_id,
                     fault_entry->fault_source.client_id,
                     fault_entry->fault_source.ve_id);
    }
    else {
        UVM_INFO_PRINT("Replayable fault GPU %s: raw=0x%llx page=0x%llx type=%s access=%s utlb=%u gpc=%u client=%u ve=%u\n",
                       uvm_parent_gpu_name(parent_gpu),
                       fault_address_raw,
                       fault_entry->fault_address,
                       uvm_fault_type_string(fault_entry->fault_type),
                       uvm_fault_access_type_string(fault_entry->fault_access_type),
                       fault_entry->fault_source.utlb_id,
                       fault_entry->fault_source.gpc_id,
                       fault_entry->fault_source.client_id,
                       fault_entry->fault_source.ve_id);
    }
}

static void log_replayable_fault_counters(uvm_parent_gpu_t *parent_gpu,
                                          NvU32 batch_faults,
                                          NvU32 batch_duplicates,
                                          NvU32 batch_kv_faults,
                                          NvU32 batch_kv_duplicates)
{
    NvU32 batch_after_dedup = batch_faults >= batch_duplicates ? batch_faults - batch_duplicates : 0;
    NvU32 batch_kv_after_dedup = batch_kv_faults >= batch_kv_duplicates ? batch_kv_faults - batch_kv_duplicates : 0;
    NvU64 batch_kv_ratio_x100 = percent_x100_u64(batch_kv_faults, batch_faults);
    NvU64 batch_kv_after_dedup_ratio_x100 = percent_x100_u64(batch_kv_after_dedup, batch_after_dedup);
    NvU64 total_faults = parent_gpu->stats.num_replayable_faults;
    NvU64 total_duplicates = parent_gpu->fault_buffer.replayable.stats.num_duplicate_faults;
    NvU64 total_after_dedup = total_faults >= total_duplicates ? total_faults - total_duplicates : 0;
    NvU64 total_kv_faults = parent_gpu->fault_buffer.replayable.stats.num_kv_faults;
    NvU64 total_kv_duplicates = parent_gpu->fault_buffer.replayable.stats.num_kv_duplicate_faults;
    NvU64 total_kv_after_dedup = total_kv_faults >= total_kv_duplicates ? total_kv_faults - total_kv_duplicates : 0;
    NvU64 total_kv_ratio_x100 = percent_x100_u64(total_kv_faults, total_faults);
    NvU64 total_kv_after_dedup_ratio_x100 = percent_x100_u64(total_kv_after_dedup, total_after_dedup);

    if (uvm_perf_fault_log_destination == 1) {
        trace_printk("Replayable fault stats GPU %s: batch_faults=%u batch_duplicates=%u batch_after_dedup=%u batch_kv_faults=%u batch_kv_duplicates=%u batch_kv_after_dedup=%u batch_kv_ratio=%llu.%02llu%% batch_kv_after_dedup_ratio=%llu.%02llu%% total_faults=%llu total_duplicates=%llu total_after_dedup=%llu total_kv_faults=%llu total_kv_duplicates=%llu total_kv_after_dedup=%llu total_kv_ratio=%llu.%02llu%% total_kv_after_dedup_ratio=%llu.%02llu%%\n",
                     uvm_parent_gpu_name(parent_gpu),
                     batch_faults,
                     batch_duplicates,
                     batch_after_dedup,
                     batch_kv_faults,
                     batch_kv_duplicates,
                     batch_kv_after_dedup,
                     (unsigned long long)(batch_kv_ratio_x100 / 100ULL),
                     (unsigned long long)(batch_kv_ratio_x100 % 100ULL),
                     (unsigned long long)(batch_kv_after_dedup_ratio_x100 / 100ULL),
                     (unsigned long long)(batch_kv_after_dedup_ratio_x100 % 100ULL),
                     (unsigned long long)total_faults,
                     (unsigned long long)total_duplicates,
                     (unsigned long long)total_after_dedup,
                     (unsigned long long)total_kv_faults,
                     (unsigned long long)total_kv_duplicates,
                     (unsigned long long)total_kv_after_dedup,
                     (unsigned long long)(total_kv_ratio_x100 / 100ULL),
                     (unsigned long long)(total_kv_ratio_x100 % 100ULL),
                     (unsigned long long)(total_kv_after_dedup_ratio_x100 / 100ULL),
                     (unsigned long long)(total_kv_after_dedup_ratio_x100 % 100ULL));
    }
    else {
        UVM_INFO_PRINT("Replayable fault stats GPU %s: batch_faults=%u batch_duplicates=%u batch_after_dedup=%u batch_kv_faults=%u batch_kv_duplicates=%u batch_kv_after_dedup=%u batch_kv_ratio=%llu.%02llu%% batch_kv_after_dedup_ratio=%llu.%02llu%% total_faults=%llu total_duplicates=%llu total_after_dedup=%llu total_kv_faults=%llu total_kv_duplicates=%llu total_kv_after_dedup=%llu total_kv_ratio=%llu.%02llu%% total_kv_after_dedup_ratio=%llu.%02llu%%\n",
                       uvm_parent_gpu_name(parent_gpu),
                       batch_faults,
                       batch_duplicates,
                       batch_after_dedup,
                       batch_kv_faults,
                       batch_kv_duplicates,
                       batch_kv_after_dedup,
                       (unsigned long long)(batch_kv_ratio_x100 / 100ULL),
                       (unsigned long long)(batch_kv_ratio_x100 % 100ULL),
                       (unsigned long long)(batch_kv_after_dedup_ratio_x100 / 100ULL),
                       (unsigned long long)(batch_kv_after_dedup_ratio_x100 % 100ULL),
                       (unsigned long long)total_faults,
                       (unsigned long long)total_duplicates,
                       (unsigned long long)total_after_dedup,
                       (unsigned long long)total_kv_faults,
                       (unsigned long long)total_kv_duplicates,
                       (unsigned long long)total_kv_after_dedup,
                       (unsigned long long)(total_kv_ratio_x100 / 100ULL),
                       (unsigned long long)(total_kv_ratio_x100 % 100ULL),
                       (unsigned long long)(total_kv_after_dedup_ratio_x100 / 100ULL),
                       (unsigned long long)(total_kv_after_dedup_ratio_x100 % 100ULL));
    }
}

// This function is used for both the initial fault buffer initialization and
// the power management resume path.
static void fault_buffer_reinit_replayable_faults(uvm_parent_gpu_t *parent_gpu)
{
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;

    // Read the current get/put pointers, as this might not be the first time
    // we take control of the fault buffer since the GPU was initialized,
    // or since we may need to bring UVM's cached copies back in sync following
    // a sleep cycle.
    replayable_faults->cached_get = parent_gpu->fault_buffer_hal->read_get(parent_gpu);
    replayable_faults->cached_put = parent_gpu->fault_buffer_hal->read_put(parent_gpu);

    // (Re-)enable fault prefetching
    if (parent_gpu->fault_buffer.prefetch_faults_enabled)
        parent_gpu->arch_hal->enable_prefetch_faults(parent_gpu);
    else
        parent_gpu->arch_hal->disable_prefetch_faults(parent_gpu);
}

// There is no error handling in this function. The caller is in charge of
// calling fault_buffer_deinit_replayable_faults on failure.
static NV_STATUS fault_buffer_init_replayable_faults(uvm_parent_gpu_t *parent_gpu)
{
    NV_STATUS status = NV_OK;
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;
    uvm_fault_service_batch_context_t *batch_context = &replayable_faults->batch_service_context;

    UVM_ASSERT(parent_gpu->fault_buffer.rm_info.replayable.bufferSize %
               parent_gpu->fault_buffer_hal->entry_size(parent_gpu) == 0);

    replayable_faults->max_faults = parent_gpu->fault_buffer.rm_info.replayable.bufferSize /
                                    parent_gpu->fault_buffer_hal->entry_size(parent_gpu);

    // Check provided module parameter value
    parent_gpu->fault_buffer.max_batch_size = max(uvm_perf_fault_batch_count,
                                                  (NvU32)UVM_PERF_FAULT_BATCH_COUNT_MIN);
    parent_gpu->fault_buffer.max_batch_size = min(parent_gpu->fault_buffer.max_batch_size,
                                                  replayable_faults->max_faults);

    if (parent_gpu->fault_buffer.max_batch_size != uvm_perf_fault_batch_count) {
        UVM_INFO_PRINT("Invalid uvm_perf_fault_batch_count value on GPU %s: %u. Valid range [%u:%u] Using %u instead\n",
                       uvm_parent_gpu_name(parent_gpu),
                       uvm_perf_fault_batch_count,
                       UVM_PERF_FAULT_BATCH_COUNT_MIN,
                       replayable_faults->max_faults,
                       parent_gpu->fault_buffer.max_batch_size);
    }

    batch_context->fault_cache = uvm_kvmalloc_zero(replayable_faults->max_faults * sizeof(*batch_context->fault_cache));
    if (!batch_context->fault_cache)
        return NV_ERR_NO_MEMORY;

    // fault_cache is used to signal that the tracker was initialized.
    uvm_tracker_init(&replayable_faults->replay_tracker);

    batch_context->ordered_fault_cache = uvm_kvmalloc_zero(replayable_faults->max_faults *
                                                           sizeof(*batch_context->ordered_fault_cache));
    if (!batch_context->ordered_fault_cache)
        return NV_ERR_NO_MEMORY;

    // This value must be initialized by HAL
    UVM_ASSERT(replayable_faults->utlb_count > 0);

    batch_context->utlbs = uvm_kvmalloc_zero(replayable_faults->utlb_count * sizeof(*batch_context->utlbs));
    if (!batch_context->utlbs)
        return NV_ERR_NO_MEMORY;

    batch_context->max_utlb_id = 0;

    status = uvm_rm_locked_call(nvUvmInterfaceOwnPageFaultIntr(parent_gpu->rm_device, NV_TRUE));
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to take page fault ownership from RM: %s, GPU %s\n",
                      nvstatusToString(status),
                      uvm_parent_gpu_name(parent_gpu));
        return status;
    }

    replayable_faults->replay_policy = uvm_perf_fault_replay_policy < UVM_PERF_FAULT_REPLAY_POLICY_MAX?
                                           uvm_perf_fault_replay_policy:
                                           UVM_PERF_FAULT_REPLAY_POLICY_DEFAULT;

    if (replayable_faults->replay_policy != uvm_perf_fault_replay_policy) {
        UVM_INFO_PRINT("Invalid uvm_perf_fault_replay_policy value on GPU %s: %d. Using %d instead\n",
                       uvm_parent_gpu_name(parent_gpu),
                       uvm_perf_fault_replay_policy,
                       replayable_faults->replay_policy);
    }

    replayable_faults->replay_update_put_ratio = min(uvm_perf_fault_replay_update_put_ratio, 100u);
    if (replayable_faults->replay_update_put_ratio != uvm_perf_fault_replay_update_put_ratio) {
        UVM_INFO_PRINT("Invalid uvm_perf_fault_replay_update_put_ratio value on GPU %s: %u. Using %u instead\n",
                       uvm_parent_gpu_name(parent_gpu),
                       uvm_perf_fault_replay_update_put_ratio,
                       replayable_faults->replay_update_put_ratio);
    }

    // Re-enable fault prefetching just in case it was disabled in a previous run
    parent_gpu->fault_buffer.prefetch_faults_enabled = true;

    fault_buffer_reinit_replayable_faults(parent_gpu);

    return NV_OK;
}

static void fault_buffer_deinit_replayable_faults(uvm_parent_gpu_t *parent_gpu)
{
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;
    uvm_fault_service_batch_context_t *batch_context = &replayable_faults->batch_service_context;

    if (batch_context->fault_cache) {
        UVM_ASSERT(uvm_tracker_is_empty(&replayable_faults->replay_tracker));
        uvm_tracker_deinit(&replayable_faults->replay_tracker);
    }

    if (parent_gpu->fault_buffer.rm_info.faultBufferHandle) {
        // Re-enable prefetch faults in case we disabled them
        if (!parent_gpu->fault_buffer.prefetch_faults_enabled)
            parent_gpu->arch_hal->enable_prefetch_faults(parent_gpu);
    }

    uvm_kvfree(batch_context->fault_cache);
    uvm_kvfree(batch_context->ordered_fault_cache);
    uvm_kvfree(batch_context->utlbs);
    batch_context->fault_cache         = NULL;
    batch_context->ordered_fault_cache = NULL;
    batch_context->utlbs               = NULL;
}

NV_STATUS uvm_parent_gpu_fault_buffer_init(uvm_parent_gpu_t *parent_gpu)
{
    NV_STATUS status = NV_OK;

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    status = uvm_rm_locked_call(nvUvmInterfaceInitFaultInfo(parent_gpu->rm_device,
                                                            &parent_gpu->fault_buffer.rm_info));
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to init fault buffer info from RM: %s, GPU %s\n",
                      nvstatusToString(status),
                      uvm_parent_gpu_name(parent_gpu));

        // nvUvmInterfaceInitFaultInfo may leave fields in rm_info populated
        // when it returns an error. Set the buffer handle to zero as it is
        // used by the deinitialization logic to determine if it was correctly
        // initialized.
        parent_gpu->fault_buffer.rm_info.faultBufferHandle = 0;
        goto fail;
    }

    status = fault_buffer_init_replayable_faults(parent_gpu);
    if (status != NV_OK)
        goto fail;

    status = uvm_parent_gpu_fault_buffer_init_non_replayable_faults(parent_gpu);
    if (status != NV_OK)
        goto fail;

    return NV_OK;

fail:
    uvm_parent_gpu_fault_buffer_deinit(parent_gpu);

    return status;
}

// Reinitialize state relevant to replayable fault handling after returning
// from a power management cycle.
void uvm_parent_gpu_fault_buffer_resume(uvm_parent_gpu_t *parent_gpu)
{
    fault_buffer_reinit_replayable_faults(parent_gpu);
}

void uvm_parent_gpu_fault_buffer_deinit(uvm_parent_gpu_t *parent_gpu)
{
    NV_STATUS status = NV_OK;

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    uvm_parent_gpu_fault_buffer_deinit_non_replayable_faults(parent_gpu);

    fault_buffer_deinit_replayable_faults(parent_gpu);

    if (parent_gpu->fault_buffer.rm_info.faultBufferHandle) {
        status = uvm_rm_locked_call(nvUvmInterfaceOwnPageFaultIntr(parent_gpu->rm_device, NV_FALSE));
        UVM_ASSERT(status == NV_OK);

        uvm_rm_locked_call_void(nvUvmInterfaceDestroyFaultInfo(parent_gpu->rm_device,
                                                               &parent_gpu->fault_buffer.rm_info));

        parent_gpu->fault_buffer.rm_info.faultBufferHandle = 0;
    }
}

bool uvm_parent_gpu_replayable_faults_pending(uvm_parent_gpu_t *parent_gpu)
{
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;

    // Fast path 1: we left some faults unserviced in the buffer in the last
    // pass
    if (replayable_faults->cached_get != replayable_faults->cached_put)
        return true;

    // Fast path 2: read the valid bit of the fault buffer entry pointed by the
    // cached get pointer
    if (!parent_gpu->fault_buffer_hal->entry_is_valid(parent_gpu, replayable_faults->cached_get)) {
        // Slow path: read the put pointer from the GPU register via BAR0
        // over PCIe
        replayable_faults->cached_put = parent_gpu->fault_buffer_hal->read_put(parent_gpu);

        // No interrupt pending
        if (replayable_faults->cached_get == replayable_faults->cached_put)
            return false;
    }

    return true;
}

// Push a fault cancel method on the given client. Any failure during this
// operation may lead to application hang (requiring manual Ctrl+C from the
// user) or system crash (requiring reboot).
// In that case we log an error message.
//
// gpc_id and client_id aren't used if global_cancel is true.
//
// This function acquires both the given tracker and the replay tracker
static NV_STATUS push_cancel_on_gpu(uvm_gpu_t *gpu,
                                    uvm_gpu_phys_address_t instance_ptr,
                                    bool global_cancel,
                                    NvU32 gpc_id,
                                    NvU32 client_id,
                                    uvm_tracker_t *tracker)
{
    NV_STATUS status;
    uvm_push_t push;
    uvm_tracker_t *replay_tracker = &gpu->parent->fault_buffer.replayable.replay_tracker;

    UVM_ASSERT(tracker != NULL);

    status = uvm_tracker_add_tracker_safe(tracker, replay_tracker);
    if (status != NV_OK)
        return status;

    if (global_cancel) {
        status = uvm_push_begin_acquire(gpu->channel_manager,
                                        UVM_CHANNEL_TYPE_MEMOPS,
                                        tracker,
                                        &push,
                                        "Cancel targeting instance_ptr {0x%llx:%s}\n",
                                        instance_ptr.address,
                                        uvm_aperture_string(instance_ptr.aperture));
    }
    else {
        status = uvm_push_begin_acquire(gpu->channel_manager,
                                        UVM_CHANNEL_TYPE_MEMOPS,
                                        tracker,
                                        &push,
                                        "Cancel targeting instance_ptr {0x%llx:%s} gpc %u client %u\n",
                                        instance_ptr.address,
                                        uvm_aperture_string(instance_ptr.aperture),
                                        gpc_id,
                                        client_id);
    }

    UVM_ASSERT(status == NV_OK);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to create push and acquire trackers before pushing cancel: %s, GPU %s\n",
                      nvstatusToString(status),
                      uvm_gpu_name(gpu));
        return status;
    }

    if (global_cancel)
        gpu->parent->host_hal->cancel_faults_global(&push, instance_ptr);
    else
        gpu->parent->host_hal->cancel_faults_targeted(&push, instance_ptr, gpc_id, client_id);

    // We don't need to put the cancel in the GPU replay tracker since we wait
    // on it immediately.
    status = uvm_push_end_and_wait(&push);

    UVM_ASSERT(status == NV_OK);
    if (status != NV_OK)
        UVM_ERR_PRINT("Failed to wait for pushed cancel: %s, GPU %s\n", nvstatusToString(status), uvm_gpu_name(gpu));

    // The cancellation is complete, so the input trackers must be complete too.
    uvm_tracker_clear(tracker);
    uvm_tracker_clear(replay_tracker);

    return status;
}

static NV_STATUS push_cancel_on_gpu_targeted(uvm_gpu_t *gpu,
                                             uvm_gpu_phys_address_t instance_ptr,
                                             NvU32 gpc_id,
                                             NvU32 client_id,
                                             uvm_tracker_t *tracker)
{
    return push_cancel_on_gpu(gpu, instance_ptr, false, gpc_id, client_id, tracker);
}

static NV_STATUS push_cancel_on_gpu_global(uvm_gpu_t *gpu, uvm_gpu_phys_address_t instance_ptr, uvm_tracker_t *tracker)
{
    UVM_ASSERT(!gpu->parent->smc.enabled);

    return push_cancel_on_gpu(gpu, instance_ptr, true, 0, 0, tracker);
}

// Volta implements a targeted VA fault cancel that simplifies the fault cancel
// process. You only need to specify the address, type, and mmu_engine_id for
// the access to be cancelled. Caller must hold the VA space lock for the access
// to be cancelled.
static NV_STATUS cancel_fault_precise_va(uvm_fault_buffer_entry_t *fault_entry,
                                         uvm_fault_cancel_va_mode_t cancel_va_mode)
{
    NV_STATUS status;
    uvm_gpu_va_space_t *gpu_va_space;
    uvm_va_space_t *va_space = fault_entry->va_space;
    uvm_gpu_t *gpu = fault_entry->gpu;
    uvm_gpu_phys_address_t pdb;
    uvm_push_t push;
    uvm_replayable_fault_buffer_t *replayable_faults = &gpu->parent->fault_buffer.replayable;
    NvU64 offset;

    UVM_ASSERT(fault_entry->fatal_reason != UvmEventFatalReasonInvalid);
    UVM_ASSERT(!fault_entry->filtered);

    gpu_va_space = uvm_gpu_va_space_get(va_space, gpu);
    UVM_ASSERT(gpu_va_space);
    pdb = uvm_page_tree_pdb_address(&gpu_va_space->page_tables);

    // Record fatal fault event
    uvm_tools_record_gpu_fatal_fault(gpu->id, va_space, fault_entry, fault_entry->fatal_reason);

    status = uvm_push_begin_acquire(gpu->channel_manager,
                                    UVM_CHANNEL_TYPE_MEMOPS,
                                    &replayable_faults->replay_tracker,
                                    &push,
                                    "Precise cancel targeting PDB {0x%llx:%s} VA 0x%llx VEID %u with access type %s",
                                    pdb.address,
                                    uvm_aperture_string(pdb.aperture),
                                    fault_entry->fault_address,
                                    fault_entry->fault_source.ve_id,
                                    uvm_fault_access_type_string(fault_entry->fault_access_type));
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to create push and acquire replay tracker before pushing cancel: %s, GPU %s\n",
                      nvstatusToString(status),
                      uvm_gpu_name(gpu));
        return status;
    }

    // UVM aligns fault addresses to PAGE_SIZE as it is the smallest mapping
    // and coherence tracking granularity. However, the cancel method requires
    // the original address (4K-aligned) reported in the packet, which is lost
    // at this point. Since the access permissions are the same for the whole
    // 64K page, we issue a cancel per 4K range to make sure that the HW sees
    // the address reported in the packet.
    for (offset = 0; offset < PAGE_SIZE; offset += UVM_PAGE_SIZE_4K) {
        gpu->parent->host_hal->cancel_faults_va(&push, pdb, fault_entry, cancel_va_mode);
        fault_entry->fault_address += UVM_PAGE_SIZE_4K;
    }
    fault_entry->fault_address = UVM_PAGE_ALIGN_DOWN(fault_entry->fault_address - 1);

    // We don't need to put the cancel in the GPU replay tracker since we wait
    // on it immediately.
    status = uvm_push_end_and_wait(&push);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to wait for pushed VA global fault cancel: %s, GPU %s\n",
                      nvstatusToString(status), uvm_gpu_name(gpu));
    }

    uvm_tracker_clear(&replayable_faults->replay_tracker);

    return status;
}

static NV_STATUS push_replay_on_gpu(uvm_gpu_t *gpu,
                                    uvm_fault_replay_type_t type,
                                    uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status;
    uvm_push_t push;
    uvm_replayable_fault_buffer_t *replayable_faults = &gpu->parent->fault_buffer.replayable;
    uvm_tracker_t *tracker = NULL;

    if (batch_context)
        tracker = &batch_context->tracker;

    status = uvm_push_begin_acquire(gpu->channel_manager, UVM_CHANNEL_TYPE_MEMOPS, tracker, &push,
                                    "Replaying faults");
    if (status != NV_OK)
        return status;

    gpu->parent->host_hal->replay_faults(&push, type);

    // Do not count REPLAY_TYPE_START_ACK_ALL's toward the replay count.
    // REPLAY_TYPE_START_ACK_ALL's are issued for cancels, and the cancel
    // algorithm checks to make sure that no REPLAY_TYPE_START's have been
    // issued using batch_context->replays.
    if (batch_context && type != UVM_FAULT_REPLAY_TYPE_START_ACK_ALL) {
        uvm_tools_broadcast_replay(gpu, &push, batch_context->batch_id, UVM_FAULT_CLIENT_TYPE_GPC);
        ++batch_context->num_replays;
    }

    uvm_push_end(&push);

    // Add this push to the GPU's replay_tracker so cancel can wait on it.
    status = uvm_tracker_add_push_safe(&replayable_faults->replay_tracker, &push);

    if (uvm_procfs_is_debug_enabled()) {
        if (type == UVM_FAULT_REPLAY_TYPE_START)
            ++replayable_faults->stats.num_replays;
        else
            ++replayable_faults->stats.num_replays_ack_all;
    }

    return status;
}

static NV_STATUS push_replay_on_parent_gpu(uvm_parent_gpu_t *parent_gpu,
                                           uvm_fault_replay_type_t type,
                                           uvm_fault_service_batch_context_t *batch_context)
{
    uvm_gpu_t *gpu = uvm_parent_gpu_find_first_valid_gpu(parent_gpu);

    if (gpu)
        return push_replay_on_gpu(gpu, type, batch_context);

    return NV_OK;
}

static void write_get(uvm_parent_gpu_t *parent_gpu, NvU32 get)
{
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;

    UVM_ASSERT(uvm_sem_is_locked(&parent_gpu->isr.replayable_faults.service_lock));

    // Write get on the GPU only if it's changed.
    if (replayable_faults->cached_get == get)
        return;

    replayable_faults->cached_get = get;

    // Update get pointer on the GPU
    parent_gpu->fault_buffer_hal->write_get(parent_gpu, get);
}

// In Confidential Computing GSP-RM owns the HW replayable fault buffer.
// Flushing the fault buffer implies flushing both the HW buffer (using a RM
// API), and the SW buffer accessible by UVM ("shadow" buffer).
//
// The HW buffer needs to be flushed first. This is because, once that flush
// completes, any faults that were present in the HW buffer have been moved to
// the shadow buffer, or have been discarded by RM.
static NV_STATUS hw_fault_buffer_flush_locked(uvm_parent_gpu_t *parent_gpu, hw_fault_buffer_flush_mode_t flush_mode)
{
    NV_STATUS status;
    NvBool is_flush_mode_move;

    UVM_ASSERT(uvm_sem_is_locked(&parent_gpu->isr.replayable_faults.service_lock));
    UVM_ASSERT((flush_mode == HW_FAULT_BUFFER_FLUSH_MODE_MOVE) || (flush_mode == HW_FAULT_BUFFER_FLUSH_MODE_DISCARD));

    if (!g_uvm_global.conf_computing_enabled)
        return NV_OK;

    is_flush_mode_move = (NvBool) (flush_mode == HW_FAULT_BUFFER_FLUSH_MODE_MOVE);
    status = nvUvmInterfaceFlushReplayableFaultBuffer(&parent_gpu->fault_buffer.rm_info, is_flush_mode_move);

    UVM_ASSERT(status == NV_OK);

    return status;
}

static void fault_buffer_skip_replayable_entry(uvm_parent_gpu_t *parent_gpu, NvU32 index)
{
    UVM_ASSERT(parent_gpu->fault_buffer_hal->entry_is_valid(parent_gpu, index));

    // Flushed faults are never decrypted, but the decryption IV associated with
    // replayable faults still requires manual adjustment so it is kept in sync
    // with the encryption IV on the GSP-RM's side.
    if (g_uvm_global.conf_computing_enabled)
        uvm_conf_computing_fault_increment_decrypt_iv(parent_gpu);

    parent_gpu->fault_buffer_hal->entry_clear_valid(parent_gpu, index);
}

static NV_STATUS fault_buffer_flush_locked(uvm_parent_gpu_t *parent_gpu,
                                           uvm_gpu_t *gpu,
                                           uvm_gpu_buffer_flush_mode_t flush_mode,
                                           uvm_fault_replay_type_t fault_replay,
                                           uvm_fault_service_batch_context_t *batch_context)
{
    NvU32 get;
    NvU32 put;
    uvm_spin_loop_t spin;
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;
    NV_STATUS status;

    UVM_ASSERT(uvm_sem_is_locked(&parent_gpu->isr.replayable_faults.service_lock));

    // Wait for the prior replay to flush out old fault messages
    if (flush_mode == UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT) {
        status = uvm_tracker_wait(&replayable_faults->replay_tracker);
        if (status != NV_OK)
            return status;
    }

    // Read PUT pointer from the GPU if requested
    if (flush_mode == UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT || flush_mode == UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT) {
        status = hw_fault_buffer_flush_locked(parent_gpu, HW_FAULT_BUFFER_FLUSH_MODE_DISCARD);
        if (status != NV_OK)
            return status;

        replayable_faults->cached_put = parent_gpu->fault_buffer_hal->read_put(parent_gpu);
    }

    get = replayable_faults->cached_get;
    put = replayable_faults->cached_put;

    while (get != put) {
        // Wait until valid bit is set
        UVM_SPIN_WHILE(!parent_gpu->fault_buffer_hal->entry_is_valid(parent_gpu, get), &spin) {
            // Channels might be idle (e.g. in teardown) so check for errors
            // actively. In that case the gpu pointer is valid.
            status = gpu ? uvm_channel_manager_check_errors(gpu->channel_manager) : uvm_global_get_status();
            if (status != NV_OK) {
                write_get(parent_gpu, get);
                return status;
            }
        }

        fault_buffer_skip_replayable_entry(parent_gpu, get);
        ++get;
        if (get == replayable_faults->max_faults)
            get = 0;
    }

    write_get(parent_gpu, get);

    // Issue fault replay
    if (gpu)
        return push_replay_on_gpu(gpu, fault_replay, batch_context);

    return push_replay_on_parent_gpu(parent_gpu, fault_replay, batch_context);
}

NV_STATUS uvm_gpu_replayable_buffer_flush(uvm_gpu_t *gpu)
{
    NV_STATUS status = NV_OK;

    // Disables replayable fault interrupts and fault servicing
    uvm_parent_gpu_replayable_faults_isr_lock(gpu->parent);

    status = fault_buffer_flush_locked(gpu->parent,
                                       gpu,
                                       UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT,
                                       UVM_FAULT_REPLAY_TYPE_START,
                                       NULL);

    // This will trigger the top half to start servicing faults again, if the
    // replay brought any back in
    uvm_parent_gpu_replayable_faults_isr_unlock(gpu->parent);
    return status;
}

static inline int cmp_fault_instance_ptr(const uvm_fault_buffer_entry_t *a,
                                         const uvm_fault_buffer_entry_t *b)
{
    int result = uvm_gpu_phys_addr_cmp(a->instance_ptr, b->instance_ptr);
    // On Volta+ we need to sort by {instance_ptr + subctx_id} pair since it can
    // map to a different VA space
    if (result != 0)
        return result;
    return UVM_CMP_DEFAULT(a->fault_source.ve_id, b->fault_source.ve_id);
}

// Compare two VA spaces
static inline int cmp_va_space(const uvm_va_space_t *a, const uvm_va_space_t *b)
{
    return UVM_CMP_DEFAULT(a, b);
}

// Compare two GPUs
static inline int cmp_gpu(const uvm_gpu_t *a, const uvm_gpu_t *b)
{
    NvU32 id_a = a ? uvm_id_value(a->id) : 0;
    NvU32 id_b = b ? uvm_id_value(b->id) : 0;

    return UVM_CMP_DEFAULT(id_a, id_b);
}

// Compare two virtual addresses
static inline int cmp_addr(NvU64 a, NvU64 b)
{
    return UVM_CMP_DEFAULT(a, b);
}

// Compare two fault access types
static inline int cmp_access_type(uvm_fault_access_type_t a, uvm_fault_access_type_t b)
{
    UVM_ASSERT(a >= 0 && a < UVM_FAULT_ACCESS_TYPE_COUNT);
    UVM_ASSERT(b >= 0 && b < UVM_FAULT_ACCESS_TYPE_COUNT);

    // Check that fault access type enum values are ordered by "intrusiveness"
    BUILD_BUG_ON(UVM_FAULT_ACCESS_TYPE_ATOMIC_STRONG <= UVM_FAULT_ACCESS_TYPE_ATOMIC_WEAK);
    BUILD_BUG_ON(UVM_FAULT_ACCESS_TYPE_ATOMIC_WEAK <= UVM_FAULT_ACCESS_TYPE_WRITE);
    BUILD_BUG_ON(UVM_FAULT_ACCESS_TYPE_WRITE <= UVM_FAULT_ACCESS_TYPE_READ);
    BUILD_BUG_ON(UVM_FAULT_ACCESS_TYPE_READ <= UVM_FAULT_ACCESS_TYPE_PREFETCH);

    return b - a;
}

typedef enum
{
    // Fetch a batch of faults from the buffer. Stop at the first entry that is
    // not ready yet
    FAULT_FETCH_MODE_BATCH_READY,

    // Fetch all faults in the buffer before PUT. Wait for all faults to become
    // ready
    FAULT_FETCH_MODE_ALL,
} fault_fetch_mode_t;

static void fetch_fault_buffer_merge_entry(uvm_fault_buffer_entry_t *current_entry,
                                           uvm_fault_buffer_entry_t *last_entry)
{
    UVM_ASSERT(last_entry->num_instances > 0);

    ++last_entry->num_instances;
    uvm_fault_access_type_mask_set(&last_entry->access_type_mask, current_entry->fault_access_type);

    if (current_entry->fault_access_type > last_entry->fault_access_type) {
        // If the new entry has a higher access type, it becomes the
        // fault to be serviced. Add the previous one to the list of instances
        current_entry->access_type_mask = last_entry->access_type_mask;
        current_entry->num_instances = last_entry->num_instances;
        last_entry->filtered = true;

        // We only merge faults from different uTLBs if the new fault has an
        // access type with the same or lower level of intrusiveness.
        UVM_ASSERT(current_entry->fault_source.utlb_id == last_entry->fault_source.utlb_id);

        list_replace(&last_entry->merged_instances_list, &current_entry->merged_instances_list);
        list_add(&last_entry->merged_instances_list, &current_entry->merged_instances_list);
    }
    else {
        // Add the new entry to the list of instances for reporting purposes
        current_entry->filtered = true;
        list_add(&current_entry->merged_instances_list, &last_entry->merged_instances_list);
    }
}

static bool fetch_fault_buffer_try_merge_entry(uvm_fault_buffer_entry_t *current_entry,
                                               uvm_fault_service_batch_context_t *batch_context,
                                               uvm_fault_utlb_info_t *current_tlb,
                                               bool is_same_instance_ptr)
{
    uvm_fault_buffer_entry_t *last_tlb_entry = current_tlb->last_fault;
    uvm_fault_buffer_entry_t *last_global_entry = batch_context->last_fault;

    // Check the last coalesced fault and the coalesced fault that was
    // originated from this uTLB
    const bool is_last_tlb_fault = current_tlb->num_pending_faults > 0 &&
                                   cmp_fault_instance_ptr(current_entry, last_tlb_entry) == 0 &&
                                   current_entry->fault_address == last_tlb_entry->fault_address;

    // We only merge faults from different uTLBs if the new fault has an
    // access type with the same or lower level of intrusiveness. This is to
    // avoid having to update num_pending_faults on both uTLBs and recomputing
    // last_fault.
    const bool is_last_fault = is_same_instance_ptr &&
                               current_entry->fault_address == last_global_entry->fault_address &&
                               current_entry->fault_access_type <= last_global_entry->fault_access_type;

    if (is_last_tlb_fault) {
        fetch_fault_buffer_merge_entry(current_entry, last_tlb_entry);
        if (current_entry->fault_access_type > last_tlb_entry->fault_access_type)
            current_tlb->last_fault = current_entry;

        return true;
    }
    else if (is_last_fault) {
        fetch_fault_buffer_merge_entry(current_entry, last_global_entry);
        if (current_entry->fault_access_type > last_global_entry->fault_access_type)
            batch_context->last_fault = current_entry;

        return true;
    }

    return false;
}

// Fetch entries from the fault buffer, decode them and store them in the batch
// context. We implement the fetch modes described above.
//
// When possible, we coalesce duplicate entries to minimize the fault handling
// overhead. Basically, we merge faults with the same instance pointer and page
// virtual address. We keep track of the last fault per uTLB to detect
// duplicates due to local reuse and the last fault in the whole batch to
// detect reuse across CTAs.
//
// We will service the first fault entry with the most "intrusive" (atomic >
// write > read > prefetch) access type*. That fault entry is called the
// "representative". The rest of filtered faults have the "filtered" flag set
// and are added to a list in the representative fault entry for reporting
// purposes. The representative fault entry also contains a mask with all the
// access types that produced a fault on the page.
//
// *We only merge faults from different uTLBs if the new fault has an access
// type with the same or lower level of intrusiveness.
//
// This optimization cannot be performed during fault cancel on Pascal GPUs
// (fetch_mode == FAULT_FETCH_MODE_ALL) since we need accurate tracking of all
// the faults in each uTLB in order to guarantee precise fault attribution.
static NV_STATUS fetch_fault_buffer_entries(uvm_parent_gpu_t *parent_gpu,
                                            uvm_fault_service_batch_context_t *batch_context,
                                            fault_fetch_mode_t fetch_mode)
{
    NvU32 get;
    NvU32 put;
    NvU32 fault_index;
    NvU32 num_coalesced_faults;
    NvU32 utlb_id;
    uvm_fault_buffer_entry_t *fault_cache;
    uvm_spin_loop_t spin;
    NV_STATUS status = NV_OK;
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;
    const bool may_filter = uvm_perf_fault_coalesce;

    UVM_ASSERT(uvm_sem_is_locked(&parent_gpu->isr.replayable_faults.service_lock));

    fault_cache = batch_context->fault_cache;

    get = replayable_faults->cached_get;

    // Read put pointer from GPU and cache it
    if (get == replayable_faults->cached_put)
        replayable_faults->cached_put = parent_gpu->fault_buffer_hal->read_put(parent_gpu);

    put = replayable_faults->cached_put;

    batch_context->is_single_instance_ptr = true;
    batch_context->last_fault = NULL;

    fault_index = 0;
    num_coalesced_faults = 0;

    // Clear uTLB counters
    for (utlb_id = 0; utlb_id <= batch_context->max_utlb_id; ++utlb_id) {
        batch_context->utlbs[utlb_id].num_pending_faults = 0;
        batch_context->utlbs[utlb_id].has_fatal_faults = false;
    }
    batch_context->max_utlb_id = 0;

    if (get == put)
        goto done;

    // Parse until get != put and have enough space to cache.
    while ((get != put) &&
           (fetch_mode == FAULT_FETCH_MODE_ALL || fault_index < parent_gpu->fault_buffer.max_batch_size)) {
        bool is_same_instance_ptr = true;
        uvm_fault_buffer_entry_t *current_entry = &fault_cache[fault_index];
        uvm_fault_utlb_info_t *current_tlb;

        // We cannot just wait for the last entry (the one pointed by put) to
        // become valid, we have to do it individually since entries can be
        // written out of order
        UVM_SPIN_WHILE(!parent_gpu->fault_buffer_hal->entry_is_valid(parent_gpu, get), &spin) {
            // We have some entry to work on. Let's do the rest later.
            if (fetch_mode == FAULT_FETCH_MODE_BATCH_READY && fault_index > 0)
                goto done;

            status = uvm_global_get_status();
            if (status != NV_OK)
                goto done;
        }

        // Prevent later accesses being moved above the read of the valid bit
        smp_mb__after_atomic();

        // Got valid bit set. Let's cache.
        status = parent_gpu->fault_buffer_hal->parse_replayable_entry(parent_gpu, get, current_entry);
        if (status != NV_OK)
            goto done;

        NvU64 fault_address_raw = current_entry->fault_address;

        // The GPU aligns the fault addresses to 4k, but all of our tracking is
        // done in PAGE_SIZE chunks which might be larger.
        current_entry->fault_address = UVM_PAGE_ALIGN_DOWN(current_entry->fault_address);

        if (uvm_perf_fault_log_addresses)
            log_replayable_fault_entry(parent_gpu, current_entry, fault_address_raw);

        // Make sure that all fields in the entry are properly initialized
        current_entry->is_fatal = (current_entry->fault_type >= UVM_FAULT_TYPE_FATAL);

        if (current_entry->is_fatal) {
            // Record the fatal fault event later as we need the va_space locked
            current_entry->fatal_reason = UvmEventFatalReasonInvalidFaultType;
        }
        else {
            current_entry->fatal_reason = UvmEventFatalReasonInvalid;
        }

        current_entry->va_space = NULL;
        current_entry->gpu = NULL;
        current_entry->filtered = false;
        current_entry->replayable.cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;

        if (current_entry->fault_source.utlb_id > batch_context->max_utlb_id) {
            UVM_ASSERT(current_entry->fault_source.utlb_id < replayable_faults->utlb_count);
            batch_context->max_utlb_id = current_entry->fault_source.utlb_id;
        }

        current_tlb = &batch_context->utlbs[current_entry->fault_source.utlb_id];

        if (fault_index > 0) {
            UVM_ASSERT(batch_context->last_fault);
            is_same_instance_ptr = cmp_fault_instance_ptr(current_entry, batch_context->last_fault) == 0;

            // Coalesce duplicate faults when possible
            if (may_filter && !current_entry->is_fatal) {
                bool merged = fetch_fault_buffer_try_merge_entry(current_entry,
                                                                 batch_context,
                                                                 current_tlb,
                                                                 is_same_instance_ptr);
                if (merged)
                    goto next_fault;
            }
        }

        if (batch_context->is_single_instance_ptr && !is_same_instance_ptr)
            batch_context->is_single_instance_ptr = false;

        current_entry->num_instances = 1;
        current_entry->access_type_mask = uvm_fault_access_type_mask_bit(current_entry->fault_access_type);
        INIT_LIST_HEAD(&current_entry->merged_instances_list);

        ++current_tlb->num_pending_faults;
        current_tlb->last_fault = current_entry;
        batch_context->last_fault = current_entry;

        ++num_coalesced_faults;

    next_fault:
        ++fault_index;
        ++get;
        if (get == replayable_faults->max_faults)
            get = 0;
    }

done:
    write_get(parent_gpu, get);

    batch_context->num_cached_faults = fault_index;
    batch_context->num_coalesced_faults = num_coalesced_faults;

    return status;
}

// Sort comparator for pointers to fault buffer entries that sorts by
// instance pointer
static int cmp_sort_fault_entry_by_instance_ptr(const void *_a, const void *_b)
{
    const uvm_fault_buffer_entry_t **a = (const uvm_fault_buffer_entry_t **)_a;
    const uvm_fault_buffer_entry_t **b = (const uvm_fault_buffer_entry_t **)_b;

    return cmp_fault_instance_ptr(*a, *b);
}

// Sort comparator for pointers to fault buffer entries that sorts by va_space,
// GPU ID, fault address, and fault access type.
static int cmp_sort_fault_entry_by_va_space_gpu_address_access_type(const void *_a, const void *_b)
{
    const uvm_fault_buffer_entry_t **a = (const uvm_fault_buffer_entry_t **)_a;
    const uvm_fault_buffer_entry_t **b = (const uvm_fault_buffer_entry_t **)_b;

    int result;

    result = cmp_va_space((*a)->va_space, (*b)->va_space);
    if (result != 0)
        return result;

    result = cmp_gpu((*a)->gpu, (*b)->gpu);
    if (result != 0)
        return result;

    result = cmp_addr((*a)->fault_address, (*b)->fault_address);
    if (result != 0)
        return result;

    return cmp_access_type((*a)->fault_access_type, (*b)->fault_access_type);
}

// Translate all instance pointers to a VA space and GPU instance. Since the
// buffer is ordered by instance_ptr, we minimize the number of translations.
//
// This function returns NV_WARN_MORE_PROCESSING_REQUIRED if a fault buffer
// flush occurred and executed successfully, or the error code if it failed.
// NV_OK otherwise.
static NV_STATUS translate_instance_ptrs(uvm_parent_gpu_t *parent_gpu,
                                         uvm_fault_service_batch_context_t *batch_context)
{
    NvU32 i;
    NV_STATUS status;

    for (i = 0; i < batch_context->num_coalesced_faults; ++i) {
        uvm_fault_buffer_entry_t *current_entry;

        current_entry = batch_context->ordered_fault_cache[i];

        // If this instance pointer matches the previous instance pointer, just
        // copy over the already-translated va_space and move on.
        if (i != 0 && cmp_fault_instance_ptr(current_entry, batch_context->ordered_fault_cache[i - 1]) == 0) {
            current_entry->va_space = batch_context->ordered_fault_cache[i - 1]->va_space;
            current_entry->gpu = batch_context->ordered_fault_cache[i - 1]->gpu;
            continue;
        }

        status = uvm_parent_gpu_fault_entry_to_va_space(parent_gpu,
                                                        current_entry,
                                                        &current_entry->va_space,
                                                        &current_entry->gpu);
        if (status != NV_OK) {
            uvm_gpu_t *gpu = NULL;

            if (status == NV_ERR_PAGE_TABLE_NOT_AVAIL) {
                // The channel is valid but the subcontext is not. This can only
                // happen if the subcontext is torn down before its work is
                // complete while other subcontexts in the same TSG are still
                // executing which means some GPU is still valid under that
                // parent GPU. This is a violation of the programming model. We
                // have limited options since the VA space is gone, meaning we
                // can't target the PDB for cancel even if we wanted to. So
                // we'll just throw away precise attribution and cancel this
                // fault using the SW method, which validates that the intended
                // context (TSG) is still running so we don't cancel an innocent
                // context.
                gpu = uvm_parent_gpu_find_first_valid_gpu(parent_gpu);

                UVM_ASSERT(!current_entry->va_space);
                UVM_ASSERT(gpu);
                UVM_ASSERT(gpu->max_subcontexts > 0);

                if (parent_gpu->smc.enabled) {
                    status = push_cancel_on_gpu_targeted(gpu,
                                                         current_entry->instance_ptr,
                                                         current_entry->fault_source.gpc_id,
                                                         current_entry->fault_source.client_id,
                                                         &batch_context->tracker);
                }
                else {
                    status = push_cancel_on_gpu_global(gpu, current_entry->instance_ptr, &batch_context->tracker);
                }

                if (status != NV_OK)
                    return status;

                // Fall through and let the flush restart fault processing
            }
            else {
                UVM_ASSERT(status == NV_ERR_INVALID_CHANNEL);
            }

            // If the channel is gone then we're looking at a stale fault entry.
            // The fault must have been resolved already (serviced or
            // cancelled), so we can just flush the fault buffer.
            //
            // No need to use UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT since
            // there was a context preemption for the entries we want to flush,
            // meaning PUT must reflect them.
            status = fault_buffer_flush_locked(parent_gpu,
                                               gpu,
                                               UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT,
                                               UVM_FAULT_REPLAY_TYPE_START,
                                               batch_context);
            if (status != NV_OK)
                 return status;

            return NV_WARN_MORE_PROCESSING_REQUIRED;
        }
        else {
            UVM_ASSERT(current_entry->va_space);
            UVM_ASSERT(current_entry->gpu);
        }
    }

    return NV_OK;
}

// Fault cache preprocessing for fault coalescing
//
// This function generates an ordered view of the given fault_cache in which
// faults are sorted by VA space, fault address (aligned to 4K) and access type
// "intrusiveness". In order to minimize the number of instance_ptr to VA space
// translations we perform a first sort by instance_ptr.
//
// This function returns NV_WARN_MORE_PROCESSING_REQUIRED if a fault buffer
// flush occurred during instance_ptr translation and executed successfully, or
// the error code if it failed. NV_OK otherwise.
//
// Current scheme:
// 1) sort by instance_ptr
// 2) translate all instance_ptrs to VA spaces
// 3) sort by va_space, GPU ID, fault address (fault_address is page-aligned at
//    this point) and access type.
static NV_STATUS preprocess_fault_batch(uvm_parent_gpu_t *parent_gpu,
                                        uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status;
    NvU32 i, j;
    uvm_fault_buffer_entry_t **ordered_fault_cache = batch_context->ordered_fault_cache;

    UVM_ASSERT(batch_context->num_coalesced_faults > 0);
    UVM_ASSERT(batch_context->num_cached_faults >= batch_context->num_coalesced_faults);

    // Generate an ordered view of the fault cache in ordered_fault_cache.
    // We sort the pointers, not the entries in fault_cache

    // Initialize pointers before they are sorted. We only sort one instance per
    // coalesced fault
    for (i = 0, j = 0; i < batch_context->num_cached_faults; ++i) {
        if (!batch_context->fault_cache[i].filtered)
            ordered_fault_cache[j++] = &batch_context->fault_cache[i];
    }
    UVM_ASSERT(j == batch_context->num_coalesced_faults);

    // 1) if the fault batch contains more than one, sort by instance_ptr
    if (!batch_context->is_single_instance_ptr) {
        sort(ordered_fault_cache,
             batch_context->num_coalesced_faults,
             sizeof(*ordered_fault_cache),
             cmp_sort_fault_entry_by_instance_ptr,
             NULL);
    }

    // 2) translate all instance_ptrs to VA spaces
    status = translate_instance_ptrs(parent_gpu, batch_context);
    if (status != NV_OK)
        return status;

    // 3) sort by va_space, GPU ID, fault address (GPU already reports
    // 4K-aligned address), and access type.
    sort(ordered_fault_cache,
         batch_context->num_coalesced_faults,
         sizeof(*ordered_fault_cache),
         cmp_sort_fault_entry_by_va_space_gpu_address_access_type,
         NULL);

    return NV_OK;
}

static bool check_fault_entry_duplicate(const uvm_fault_buffer_entry_t *current_entry,
                                        const uvm_fault_buffer_entry_t *previous_entry)
{
    bool is_duplicate = false;

    if (previous_entry) {
        is_duplicate = (current_entry->va_space == previous_entry->va_space) &&
                       (current_entry->fault_address == previous_entry->fault_address);
    }

    return is_duplicate;
}

static void update_batch_and_notify_fault(uvm_gpu_t *gpu,
                                          uvm_fault_service_batch_context_t *batch_context,
                                          uvm_va_block_t *va_block,
                                          uvm_processor_id_t preferred_location,
                                          uvm_fault_buffer_entry_t *current_entry,
                                          bool is_duplicate)
{
    if (is_duplicate)
        batch_context->num_duplicate_faults += current_entry->num_instances;
    else
        batch_context->num_duplicate_faults += current_entry->num_instances - 1;

    uvm_perf_event_notify_gpu_fault(&current_entry->va_space->perf_events,
                                    va_block,
                                    gpu->id,
                                    preferred_location,
                                    current_entry,
                                    batch_context->batch_id,
                                    is_duplicate);
}

static void mark_fault_invalid_prefetch(uvm_fault_service_batch_context_t *batch_context,
                                        uvm_fault_buffer_entry_t *fault_entry)
{
    fault_entry->is_invalid_prefetch = true;

    // For block faults, the following counter might be updated more than once
    // for the same fault if block_context->num_retries > 0. As a result, this
    // counter might be higher than the actual count. In order for this counter
    // to be always accurate, block_context needs to passed down the stack from
    // all callers. But since num_retries > 0 case is uncommon and imprecise
    // invalid_prefetch counter doesn't affect functionality (other than
    // disabling prefetching if the counter indicates lots of invalid prefetch
    // faults), this is ok.
    batch_context->num_invalid_prefetch_faults += fault_entry->num_instances;
}

static void mark_fault_throttled(uvm_fault_service_batch_context_t *batch_context,
                                 uvm_fault_buffer_entry_t *fault_entry)
{
    fault_entry->is_throttled = true;
    batch_context->has_throttled_faults = true;
}

static void mark_fault_fatal(uvm_fault_service_batch_context_t *batch_context,
                             uvm_fault_buffer_entry_t *fault_entry,
                             UvmEventFatalReason fatal_reason,
                             uvm_fault_cancel_va_mode_t cancel_va_mode)
{
    uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[fault_entry->fault_source.utlb_id];

    fault_entry->is_fatal = true;
    fault_entry->fatal_reason = fatal_reason;
    fault_entry->replayable.cancel_va_mode = cancel_va_mode;

    utlb->has_fatal_faults = true;

    if (!batch_context->fatal_va_space) {
        UVM_ASSERT(fault_entry->va_space);
        batch_context->fatal_va_space = fault_entry->va_space;
        batch_context->fatal_gpu = fault_entry->gpu;
    }
}

static void fault_entry_duplicate_flags(uvm_fault_service_batch_context_t *batch_context,
                                        uvm_fault_buffer_entry_t *current_entry,
                                        const uvm_fault_buffer_entry_t *previous_entry)
{
    UVM_ASSERT(previous_entry);
    UVM_ASSERT(check_fault_entry_duplicate(current_entry, previous_entry));

    // Propagate the is_invalid_prefetch flag across all prefetch faults
    // on the page
    if (previous_entry->is_invalid_prefetch)
        mark_fault_invalid_prefetch(batch_context, current_entry);

    // If a page is throttled, all faults on the page must be skipped
    if (previous_entry->is_throttled)
        mark_fault_throttled(batch_context, current_entry);
}

// This function computes the maximum access type that can be serviced for the
// reported fault instances given the logical permissions of the VA range. If
// none of the fault instances can be serviced UVM_FAULT_ACCESS_TYPE_COUNT is
// returned instead.
//
// In the case that there are faults that cannot be serviced, this function
// also sets the flags required for fault cancellation. Prefetch faults do not
// need to be cancelled since they disappear on replay.
//
// The UVM driver considers two scenarios for logical permissions violation:
// - All access types are invalid. For example, when faulting from a processor
// that doesn't have access to the preferred location of a range group when it
// is not migratable. In this case all accesses to the page must be cancelled.
// - Write/atomic accesses are invalid. Basically, when trying to modify a
// read-only VA range. In this case we restrict fault cancelling to those types
// of accesses.
//
// Return values:
// - service_access_type: highest access type that can be serviced.
static uvm_fault_access_type_t check_fault_access_permissions(uvm_gpu_t *gpu,
                                                              uvm_fault_service_batch_context_t *batch_context,
                                                              uvm_va_block_t *va_block,
                                                              uvm_service_block_context_t *service_block_context,
                                                              uvm_fault_buffer_entry_t *fault_entry)
{
    NV_STATUS perm_status;
    UvmEventFatalReason fatal_reason;
    uvm_fault_cancel_va_mode_t cancel_va_mode;
    uvm_fault_access_type_t ret = UVM_FAULT_ACCESS_TYPE_COUNT;
    uvm_va_block_context_t *va_block_context = service_block_context->block_context;

    perm_status = uvm_va_block_check_logical_permissions(va_block,
                                                         va_block_context,
                                                         gpu->id,
                                                         uvm_va_block_cpu_page_index(va_block,
                                                                                     fault_entry->fault_address),
                                                         fault_entry->fault_access_type);
    if (perm_status == NV_OK)
        return fault_entry->fault_access_type;

    if (fault_entry->fault_access_type == UVM_FAULT_ACCESS_TYPE_PREFETCH) {
        // Only update the count the first time since logical permissions cannot
        // change while we hold the VA space lock
        // TODO: Bug 1750144: That might not be true with HMM.
        if (service_block_context->num_retries == 0)
            mark_fault_invalid_prefetch(batch_context, fault_entry);

        return ret;
    }

    // At this point we know that some fault instances cannot be serviced
    fatal_reason = uvm_tools_status_to_fatal_fault_reason(perm_status);

    if (fault_entry->fault_access_type > UVM_FAULT_ACCESS_TYPE_READ) {
        cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_WRITE_AND_ATOMIC;

        // If there are pending read accesses on the same page, we have to
        // service them before we can cancel the write/atomic faults. So we
        // retry with read fault access type.
        if (uvm_fault_access_type_mask_test(fault_entry->access_type_mask, UVM_FAULT_ACCESS_TYPE_READ)) {
            perm_status = uvm_va_block_check_logical_permissions(va_block,
                                                                 va_block_context,
                                                                 gpu->id,
                                                                 uvm_va_block_cpu_page_index(va_block,
                                                                                             fault_entry->fault_address),
                                                                 UVM_FAULT_ACCESS_TYPE_READ);
            if (perm_status == NV_OK) {
                ret = UVM_FAULT_ACCESS_TYPE_READ;
            }
            else {
                // Read accesses didn't succeed, cancel all faults
                cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
                fatal_reason = uvm_tools_status_to_fatal_fault_reason(perm_status);
            }
        }
    }
    else {
        cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
    }

    mark_fault_fatal(batch_context, fault_entry, fatal_reason, cancel_va_mode);

    return ret;
}

// 处理同一个 VA Block 内的一批缺页异常。
// 我们会先通知性能启发式算法更新状态，然后根据算法的决策执行必要的内存迁移或映射操作。
// 
// 致命错误（Fatal faults）会被标记，以便后续取消操作。由于权限不足或地址无效导致的致命错误
// 不会中断整个批次的服务。
//
// 返回码：
// - NV_OK：所有异常（包括致命和非致命）都已处理
// - NV_ERR_MORE_PROCESSING_REQUIRED：处理过程中需要重试（如等待内存分配）
// - NV_ERR_NO_MEMORY：内存不足导致无法处理
// - 其他值：UVM 全局错误
static NV_STATUS service_fault_batch_block_locked(uvm_gpu_t *gpu,
                                                  uvm_va_block_t *va_block,
                                                  uvm_va_block_retry_t *va_block_retry,
                                                  uvm_fault_service_batch_context_t *batch_context,
                                                  NvU32 first_fault_index,
                                                  const bool hmm_migratable,
                                                  NvU32 *block_faults)
{
    NV_STATUS status = NV_OK;
    NvU32 i;
    uvm_page_index_t first_page_index;
    uvm_page_index_t last_page_index;
    NvU32 page_fault_count = 0;
    uvm_range_group_range_iter_t iter;
    
    // 获取重放故障缓冲区及其相关的服务上下文
    uvm_replayable_fault_buffer_t *replayable_faults = &gpu->parent->fault_buffer.replayable;
    uvm_fault_buffer_entry_t **ordered_fault_cache = batch_context->ordered_fault_cache;
    uvm_fault_buffer_entry_t *first_fault_entry = ordered_fault_cache[first_fault_index];
    uvm_service_block_context_t *block_context = &replayable_faults->block_service_context;
    
    // 获取该块所属的虚拟地址空间（VA Space）
    uvm_va_space_t *va_space = uvm_va_block_get_va_space(va_block);
    const uvm_va_policy_t *policy;
    NvU64 end;

    // 编译期检查：确保访问类型总数能存入一个 8 位无符号整数中
    BUILD_BUG_ON(UVM_FAULT_ACCESS_TYPE_COUNT > (int)(NvU8)-1);

    // 运行期安全断言：确保调用此函数前已经获取了该 va_block 的互斥锁
    uvm_assert_mutex_locked(&va_block->lock);

    *block_faults = 0;

    // 初始化页面索引边界，用于后续计算本批次影响的页面范围
    first_page_index = PAGES_PER_UVM_VA_BLOCK;
    last_page_index = 0;

    // 初始化故障服务块上下文的掩码和计数器
    uvm_processor_mask_zero(&block_context->resident_processors);
    block_context->thrashing_pin_count = 0;
    block_context->read_duplicate_count = 0;

    // 初始化迁移迭代器，用于遍历该地址块的迁移属性
    uvm_range_group_range_migratability_iter_first(va_space, va_block->start, va_block->end, &iter);

    // 确保传入的第一个故障条目合法且确实落在当前处理的 VA Block 范围内
    UVM_ASSERT(first_fault_entry->va_space == va_space);
    UVM_ASSERT(first_fault_entry->gpu == gpu);
    UVM_ASSERT(first_fault_entry->fault_address >= va_block->start);
    UVM_ASSERT(first_fault_entry->fault_address <= va_block->end);

    // 获取内存策略（Policy）：区分是 HMM（异构内存管理）还是 UVM 传统托管范围
    if (uvm_va_block_is_hmm(va_block)) {
        policy = uvm_hmm_find_policy_end(va_block,
                                         block_context->block_context->hmm.vma,
                                         first_fault_entry->fault_address,
                                         &end);
    }
    else {
        policy = &va_block->managed_range->policy;
        end = va_block->end;
    }

    // 核心循环：扫描已排序的故障缓存数组，处理所有属于当前 block 的故障
    for (i = first_fault_index;
         i < batch_context->num_coalesced_faults &&
         ordered_fault_cache[i]->va_space == va_space &&
         ordered_fault_cache[i]->gpu == gpu &&
         ordered_fault_cache[i]->fault_address <= end;
         ++i) {
        
        uvm_fault_buffer_entry_t *current_entry = ordered_fault_cache[i];
        const uvm_fault_buffer_entry_t *previous_entry = NULL;
        bool read_duplicate;
        uvm_processor_id_t new_residency;
        uvm_perf_thrashing_hint_t thrashing_hint;
        
        // 计算故障地址在 VA block 内的页面索引
        uvm_page_index_t page_index = uvm_va_block_cpu_page_index(va_block, current_entry->fault_address);
        bool is_duplicate = false;
        uvm_fault_access_type_t service_access_type;
        NvU32 service_access_type_mask;

        // 断言确保故障访问类型是掩码中最高级的（如 Write 优先于 Read）
        UVM_ASSERT(current_entry->fault_access_type ==
                   uvm_fault_access_type_mask_highest(current_entry->access_type_mask));

        // 无法服务的故障已被调用者跳过。确保当前处理的不是致命错误
        UVM_ASSERT(!current_entry->is_fatal);
        current_entry->is_throttled    = false;
        current_entry->is_invalid_prefetch = false;

        // 检查与前一个故障条目是否为重复异常（比如同一页面上的并发请求）
        if (i > first_fault_index) {
            previous_entry = ordered_fault_cache[i - 1];
            is_duplicate = check_fault_entry_duplicate(current_entry, previous_entry);
        }

        // 首次尝试时（非重试状态），更新性能启发式指标并触发缺页通知
        // 因为在持有 VA space 锁期间，逻辑权限不会改变
        if (block_context->num_retries == 0) {
            update_batch_and_notify_fault(gpu,
                                          batch_context,
                                          va_block,
                                          policy->preferred_location,
                                          current_entry,
                                          is_duplicate);
        }

        // 如果是重复故障，合并标志位。我们只处理每页侵入性最强的故障（如写故障覆盖读故障）
        if (is_duplicate) {
            fault_entry_duplicate_flags(batch_context, current_entry, previous_entry);
            // 如果前一个非致命，说明该页面的处理意图已经被记录，跳过当前循环
            if (!previous_entry->is_fatal)
                continue;
        }

        // 推进迁移迭代器以覆盖当前的故障地址
        while (iter.end < current_entry->fault_address)
            uvm_range_group_range_migratability_iter_next(va_space, &iter, va_block->end);

        UVM_ASSERT(iter.start <= current_entry->fault_address && iter.end >= current_entry->fault_address);

        // 检查权限：确定该请求是否具有相应的逻辑访问权限
        service_access_type = check_fault_access_permissions(gpu,
                                                             batch_context,
                                                             va_block,
                                                             block_context,
                                                             current_entry);

        // 如果权限越界或无效，不直接退出，跳过本条目继续处理其余故障
        if (service_access_type == UVM_FAULT_ACCESS_TYPE_COUNT)
            continue;

        // 降级处理：如果请求的权限部分无效，重新计算需要提供服务的权限掩码
        if (service_access_type != current_entry->fault_access_type) {
            UVM_ASSERT(service_access_type < current_entry->fault_access_type);
            service_access_type_mask = uvm_fault_access_type_mask_bit(service_access_type);
        }
        else {
            service_access_type_mask = current_entry->access_type_mask;
        }

        // 优化点：如果 GPU 已经拥有所需的访问权限（页表已就绪），则无需做任何操作
        if (uvm_va_block_page_is_gpu_authorized(va_block,
                                                page_index,
                                                gpu->id,
                                                uvm_fault_access_type_to_prot(service_access_type)))
            continue;

        // 查询性能启发式算法中的 "Thrashing" (内存颠簸) 提示
        thrashing_hint = uvm_perf_thrashing_get_hint(va_block,
                                                     block_context->block_context,
                                                     current_entry->fault_address,
                                                     gpu->id);
        
        if (thrashing_hint.type == UVM_PERF_THRASHING_HINT_TYPE_THROTTLE) {
            // 节流处理 (Throttle)：页面在 CPU/GPU 间频繁迁移颠簸，算法建议暂停处理
            // CPU 会在异常处理程序中休眠，同时允许 GPU 继续处理其他页面的异常
            if (block_context->num_retries == 0)
                mark_fault_throttled(batch_context, current_entry);
            continue;
        }
        else if (thrashing_hint.type == UVM_PERF_THRASHING_HINT_TYPE_PIN) {
            // 锁定处理 (Pin)：算法建议将此页面固定在当前处理器中，避免迁移
            if (block_context->thrashing_pin_count++ == 0)
                uvm_page_mask_zero(&block_context->thrashing_pin_mask);
            uvm_page_mask_set(&block_context->thrashing_pin_mask, page_index);
        }

        // 核心决策：计算页面应该驻留的新位置（迁移目标）并更新掩码
        new_residency = uvm_va_block_select_residency(va_block,
                                                      block_context->block_context,
                                                      page_index,
                                                      gpu->id,
                                                      service_access_type_mask,
                                                      policy,
                                                      &thrashing_hint,
                                                      UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS,
                                                      hmm_migratable,
                                                      &read_duplicate);

        // manual_migration_fault_mode=2 is intended to prohibit implicit data
        // movement, not ordinary page-table repair. A replayable fault can be
        // raised after prefetch has restored the physical GPU page but before
        // a readable PTE/PDE has been installed. In that case servicing the
        // fault with the current GPU as the selected residency only installs
        // the missing mapping and does not migrate data.
        //
        // This decision has to be made while holding va_block->lock. The old
        // dispatch-level check ran before taking that lock and therefore could
        // not distinguish mapping-only faults from real CPU-to-GPU migration.
        if (manual_migration_fault_mode_enabled() && va_block->managed_range) {
            uvm_va_block_gpu_state_t *gpu_state = uvm_va_block_gpu_state_get(va_block, gpu->id);
            bool gpu_resident = gpu_state && uvm_page_mask_test(&gpu_state->resident, page_index);
            bool cpu_resident = uvm_page_mask_test(&va_block->cpu.resident, page_index);
            bool gpu_read_mapped = gpu_state &&
                                   uvm_page_mask_test(&gpu_state->pte_bits[UVM_PTE_BITS_GPU_READ], page_index);
            bool mapping_only = gpu_resident && uvm_id_equal(new_residency, gpu->id);
            bool block_migration = uvm_manual_migration_fault_mode ==
                                       UVM_MANUAL_MIGRATION_FAULT_MODE_FAIL_FAST &&
                                   !mapping_only;

            manual_migration_fault_log(gpu,
                                       va_block->managed_range,
                                       current_entry,
                                       page_index,
                                       gpu_resident,
                                       cpu_resident,
                                       gpu_read_mapped,
                                       new_residency,
                                       block_migration ? "block_migration" :
                                       mapping_only ? "allow_mapping_only" : "observe_migration");

            if (block_migration) {
                mark_fault_fatal(batch_context,
                                 current_entry,
                                 uvm_tools_status_to_fatal_fault_reason(NV_ERR_INVALID_ACCESS_TYPE),
                                 UVM_FAULT_CANCEL_VA_MODE_ALL);
                continue;
            }
        }

        // 记录新驻留位置的映射掩码，如果该驻留位置第一次出现，先清空其页面掩码
        if (!uvm_processor_mask_test_and_set(&block_context->resident_processors, new_residency))
            uvm_page_mask_zero(&block_context->per_processor_masks[uvm_id_value(new_residency)].new_residency);

        uvm_page_mask_set(&block_context->per_processor_masks[uvm_id_value(new_residency)].new_residency, page_index);

        // 如果启用了读复制，更新对应掩码
        if (read_duplicate) {
            if (block_context->read_duplicate_count++ == 0)
                uvm_page_mask_zero(&block_context->read_duplicate_mask);
            uvm_page_mask_set(&block_context->read_duplicate_mask, page_index);
        }

        ++page_fault_count;
        block_context->access_type[page_index] = service_access_type;

        // 追踪受影响的最小和最大页面索引，以便后续批量服务时确定操作范围
        if (page_index < first_page_index)
            first_page_index = page_index;
        if (page_index > last_page_index)
            last_page_index = page_index;
    }

    // 执行阶段：如果该 block 内有需要处理的缺页异常
    // 将上面收集到的状态（掩码、驻留决策）应用于内存子系统
    if (page_fault_count > 0) {
        block_context->region = uvm_va_block_region(first_page_index, last_page_index + 1);
        // 调用底层核心函数执行实际的内存迁移、页面分配和页表更新
        status = uvm_va_block_service_locked(gpu->id, va_block, va_block_retry, block_context);
    }

    // 返回本 block 成功遍历/处理的异常数量
    *block_faults = i - first_fault_index;
    
    // 递增重试计数，以防此操作返回 NV_ERR_MORE_PROCESSING_REQUIRED
    ++block_context->num_retries;

    return status;
}

// We notify the fault event for all faults within the block so that the
// performance heuristics are updated. The VA block lock is taken for the whole
// fault servicing although it might be temporarily dropped and re-taken if
// memory eviction is required.
//
// See the comments for function service_fault_batch_block_locked for
// implementation details and error codes.
static NV_STATUS service_fault_batch_block(uvm_gpu_t *gpu,
                                           uvm_va_block_t *va_block,
                                           uvm_fault_service_batch_context_t *batch_context,
                                           NvU32 first_fault_index,
                                           const bool hmm_migratable,
                                           NvU32 *block_faults)
{
    NV_STATUS status;
    uvm_va_block_retry_t va_block_retry;
    NV_STATUS tracker_status;
    uvm_replayable_fault_buffer_t *replayable_faults = &gpu->parent->fault_buffer.replayable;
    uvm_service_block_context_t *fault_block_context = &replayable_faults->block_service_context;

    fault_block_context->operation = UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS;
    fault_block_context->num_retries = 0;

    if (uvm_va_block_is_hmm(va_block))
        uvm_hmm_migrate_begin_wait(va_block);

    uvm_mutex_lock(&va_block->lock);

    status = UVM_VA_BLOCK_RETRY_LOCKED(va_block, &va_block_retry,
                                       service_fault_batch_block_locked(gpu,
                                                                        va_block,
                                                                        &va_block_retry,
                                                                        batch_context,
                                                                        first_fault_index,
                                                                        hmm_migratable,
                                                                        block_faults));

    tracker_status = uvm_tracker_add_tracker_safe(&batch_context->tracker, &va_block->tracker);

    uvm_mutex_unlock(&va_block->lock);

    if (uvm_va_block_is_hmm(va_block))
        uvm_hmm_migrate_finish(va_block);

    return status == NV_OK? tracker_status: status;
}

typedef enum
{
    // Use this mode when calling from the normal fault servicing path
    FAULT_SERVICE_MODE_REGULAR,

    // Use this mode when servicing faults from the fault cancelling algorithm.
    // In this mode no replays are issued
    FAULT_SERVICE_MODE_CANCEL,
} fault_service_mode_t;

static void service_fault_batch_fatal(uvm_fault_service_batch_context_t *batch_context,
                                      NvU32 first_fault_index,
                                      NV_STATUS status,
                                      uvm_fault_cancel_va_mode_t cancel_va_mode,
                                      NvU32 *block_faults)
{
    uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[first_fault_index];
    const uvm_fault_buffer_entry_t *previous_entry = first_fault_index > 0 ?
                                                       batch_context->ordered_fault_cache[first_fault_index - 1] : NULL;
    bool is_duplicate = check_fault_entry_duplicate(current_entry, previous_entry);

    if (is_duplicate)
        fault_entry_duplicate_flags(batch_context, current_entry, previous_entry);

    if (current_entry->fault_access_type == UVM_FAULT_ACCESS_TYPE_PREFETCH)
        mark_fault_invalid_prefetch(batch_context, current_entry);
    else
        mark_fault_fatal(batch_context, current_entry, uvm_tools_status_to_fatal_fault_reason(status), cancel_va_mode);

    (*block_faults)++;
}

static void service_fault_batch_fatal_notify(uvm_gpu_t *gpu,
                                             uvm_fault_service_batch_context_t *batch_context,
                                             NvU32 first_fault_index,
                                             NV_STATUS status,
                                             uvm_fault_cancel_va_mode_t cancel_va_mode,
                                             NvU32 *block_faults)
{
    uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[first_fault_index];
    const uvm_fault_buffer_entry_t *previous_entry = first_fault_index > 0 ?
                                                       batch_context->ordered_fault_cache[first_fault_index - 1] : NULL;
    bool is_duplicate = check_fault_entry_duplicate(current_entry, previous_entry);

    service_fault_batch_fatal(batch_context, first_fault_index, status, cancel_va_mode, block_faults);

    update_batch_and_notify_fault(gpu, batch_context, NULL, UVM_ID_INVALID, current_entry, is_duplicate);
}

static NV_STATUS service_fault_batch_ats_sub_vma(uvm_gpu_va_space_t *gpu_va_space,
                                                 struct vm_area_struct *vma,
                                                 NvU64 base,
                                                 uvm_fault_service_batch_context_t *batch_context,
                                                 NvU32 fault_index_start,
                                                 NvU32 fault_index_end,
                                                 NvU32 *block_faults)
{
    NvU32 i;
    NV_STATUS status = NV_OK;
    uvm_ats_fault_context_t *ats_context = &batch_context->ats_context;
    const uvm_page_mask_t *read_fault_mask = &ats_context->faults.read_fault_mask;
    const uvm_page_mask_t *write_fault_mask = &ats_context->faults.write_fault_mask;
    const uvm_page_mask_t *reads_serviced_mask = &ats_context->faults.reads_serviced_mask;
    uvm_page_mask_t *faults_serviced_mask = &ats_context->faults.faults_serviced_mask;
    uvm_page_mask_t *accessed_mask = &ats_context->faults.accessed_mask;

    UVM_ASSERT(vma);

    ats_context->client_type = UVM_FAULT_CLIENT_TYPE_GPC;

    uvm_page_mask_or(accessed_mask, write_fault_mask, read_fault_mask);

    status = uvm_ats_service_faults(gpu_va_space, vma, base, &batch_context->ats_context);

    // Remove SW prefetched pages from the serviced mask since fault servicing
    // failures belonging to prefetch pages need to be ignored.
    uvm_page_mask_and(faults_serviced_mask, faults_serviced_mask, accessed_mask);

    UVM_ASSERT(uvm_page_mask_subset(faults_serviced_mask, accessed_mask));

    if ((status != NV_OK) || uvm_page_mask_equal(faults_serviced_mask, accessed_mask)) {
        (*block_faults) += (fault_index_end - fault_index_start);
        return status;
    }

    // Check faults_serviced_mask and reads_serviced_mask for precise fault
    // attribution after calling the ATS servicing routine. The
    // errors returned from ATS servicing routine should only be
    // global errors such as OOM or ECC.
    // uvm_parent_gpu_service_replayable_faults() handles global errors by
    // calling cancel_fault_batch(). Precise attribution isn't currently
    // supported in such cases.
    //
    // Precise fault attribution for global errors can be handled by
    // servicing one fault at a time until fault servicing encounters an
    // error.
    // TODO: Bug 3989244: Precise ATS fault attribution for global errors.
    for (i = fault_index_start; i < fault_index_end; i++) {
        uvm_page_index_t page_index;
        uvm_fault_cancel_va_mode_t cancel_va_mode;
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
        uvm_fault_access_type_t access_type = current_entry->fault_access_type;

        page_index = (current_entry->fault_address - base) / PAGE_SIZE;

        if (uvm_page_mask_test(faults_serviced_mask, page_index)) {
            (*block_faults)++;
            continue;
        }

        if (access_type <= UVM_FAULT_ACCESS_TYPE_READ) {
            cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
        }
        else {
            UVM_ASSERT(access_type >= UVM_FAULT_ACCESS_TYPE_WRITE);
            if (uvm_fault_access_type_mask_test(current_entry->access_type_mask, UVM_FAULT_ACCESS_TYPE_READ) &&
                !uvm_page_mask_test(reads_serviced_mask, page_index))
                cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
            else
                cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_WRITE_AND_ATOMIC;
        }

        service_fault_batch_fatal(batch_context, i, NV_ERR_INVALID_ADDRESS, cancel_va_mode, block_faults);
    }

    return status;
}

static void start_new_sub_batch(NvU64 *sub_batch_base,
                                NvU64 address,
                                NvU32 *sub_batch_fault_index,
                                NvU32 fault_index,
                                uvm_ats_fault_context_t *ats_context)
{
    uvm_page_mask_zero(&ats_context->faults.read_fault_mask);
    uvm_page_mask_zero(&ats_context->faults.write_fault_mask);
    uvm_page_mask_zero(&ats_context->faults.prefetch_only_fault_mask);

    *sub_batch_fault_index = fault_index;
    *sub_batch_base = UVM_VA_BLOCK_ALIGN_DOWN(address);
}

static NV_STATUS service_fault_batch_ats_sub(uvm_gpu_va_space_t *gpu_va_space,
                                             struct vm_area_struct *vma,
                                             uvm_fault_service_batch_context_t *batch_context,
                                             NvU32 fault_index,
                                             NvU64 outer,
                                             NvU32 *block_faults)
{
    NV_STATUS status = NV_OK;
    NvU32 i = fault_index;
    NvU32 sub_batch_fault_index;
    NvU64 sub_batch_base;
    uvm_fault_buffer_entry_t *previous_entry = NULL;
    uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
    uvm_ats_fault_context_t *ats_context = &batch_context->ats_context;
    uvm_page_mask_t *read_fault_mask = &ats_context->faults.read_fault_mask;
    uvm_page_mask_t *write_fault_mask = &ats_context->faults.write_fault_mask;
    uvm_page_mask_t *prefetch_only_fault_mask = &ats_context->faults.prefetch_only_fault_mask;
    uvm_gpu_t *gpu = gpu_va_space->gpu;
    bool replay_per_va_block =
                        (gpu->parent->fault_buffer.replayable.replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BLOCK);

    UVM_ASSERT(vma);

    outer = min(outer, (NvU64) vma->vm_end);

    start_new_sub_batch(&sub_batch_base, current_entry->fault_address, &sub_batch_fault_index, i, ats_context);

    do {
        uvm_page_index_t page_index;
        NvU64 fault_address = current_entry->fault_address;
        uvm_fault_access_type_t access_type = current_entry->fault_access_type;
        bool is_duplicate = check_fault_entry_duplicate(current_entry, previous_entry);

        // ATS faults can't be unserviceable, since unserviceable faults require
        // GMMU PTEs.
        UVM_ASSERT(!current_entry->is_fatal);
        UVM_ASSERT(current_entry->gpu == gpu);

        i++;

        update_batch_and_notify_fault(gpu,
                                      batch_context,
                                      NULL,
                                      UVM_ID_INVALID,
                                      current_entry,
                                      is_duplicate);

        // End of sub-batch. Service faults gathered so far.
        if (fault_address >= (sub_batch_base + UVM_VA_BLOCK_SIZE)) {
            UVM_ASSERT(!uvm_page_mask_empty(read_fault_mask) ||
                       !uvm_page_mask_empty(write_fault_mask) ||
                       !uvm_page_mask_empty(prefetch_only_fault_mask));

            status = service_fault_batch_ats_sub_vma(gpu_va_space,
                                                     vma,
                                                     sub_batch_base,
                                                     batch_context,
                                                     sub_batch_fault_index,
                                                     i - 1,
                                                     block_faults);
            if (status != NV_OK || replay_per_va_block)
                break;

            start_new_sub_batch(&sub_batch_base, fault_address, &sub_batch_fault_index, i - 1, ats_context);
        }

        page_index = (fault_address - sub_batch_base) / PAGE_SIZE;

        // Do not check for coalesced access type. If there are multiple
        // different accesses to an address, we can disregard the prefetch one.
        if ((access_type == UVM_FAULT_ACCESS_TYPE_PREFETCH) &&
            (uvm_fault_access_type_mask_highest(current_entry->access_type_mask) == UVM_FAULT_ACCESS_TYPE_PREFETCH))
            uvm_page_mask_set(prefetch_only_fault_mask, page_index);

        if ((access_type == UVM_FAULT_ACCESS_TYPE_READ) ||
            uvm_fault_access_type_mask_test(current_entry->access_type_mask, UVM_FAULT_ACCESS_TYPE_READ))
            uvm_page_mask_set(read_fault_mask, page_index);

        if (access_type >= UVM_FAULT_ACCESS_TYPE_WRITE)
            uvm_page_mask_set(write_fault_mask, page_index);

        previous_entry = current_entry;
        current_entry = i < batch_context->num_coalesced_faults ? batch_context->ordered_fault_cache[i] : NULL;

    } while (current_entry &&
             (current_entry->fault_address < outer) &&
             (previous_entry->gpu == current_entry->gpu) &&
             (previous_entry->va_space == current_entry->va_space));

    // Service the last sub-batch.
    if ((status == NV_OK) &&
        (!uvm_page_mask_empty(read_fault_mask) ||
         !uvm_page_mask_empty(write_fault_mask) ||
         !uvm_page_mask_empty(prefetch_only_fault_mask))) {
        status = service_fault_batch_ats_sub_vma(gpu_va_space,
                                                 vma,
                                                 sub_batch_base,
                                                 batch_context,
                                                 sub_batch_fault_index,
                                                 i,
                                                 block_faults);
    }

    return status;
}

static NV_STATUS service_fault_batch_ats(uvm_gpu_va_space_t *gpu_va_space,
                                         struct mm_struct *mm,
                                         uvm_fault_service_batch_context_t *batch_context,
                                         NvU32 first_fault_index,
                                         NvU64 outer,
                                         NvU32 *block_faults)
{
    NvU32 i;
    NV_STATUS status = NV_OK;

    for (i = first_fault_index; i < batch_context->num_coalesced_faults;) {
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
        const uvm_fault_buffer_entry_t *previous_entry = i > first_fault_index ?
                                                                       batch_context->ordered_fault_cache[i - 1] : NULL;
        NvU64 fault_address = current_entry->fault_address;
        struct vm_area_struct *vma;
        NvU32 num_faults_before = (*block_faults);

        if (previous_entry &&
            (previous_entry->va_space != current_entry->va_space || previous_entry->gpu != current_entry->gpu))
            break;

        if (fault_address >= outer)
            break;

        vma = find_vma_intersection(mm, fault_address, fault_address + 1);
        if (!vma) {
            // Since a vma wasn't found, cancel all accesses on the page since
            // cancelling write and atomic accesses will not cancel pending read
            // faults and this can lead to a deadlock since read faults need to
            // be serviced first before cancelling write faults.
            service_fault_batch_fatal_notify(gpu_va_space->gpu,
                                             batch_context,
                                             i,
                                             NV_ERR_INVALID_ADDRESS,
                                             UVM_FAULT_CANCEL_VA_MODE_ALL,
                                             block_faults);

            // Do not fail due to logical errors.
            status = NV_OK;

            break;
        }

        status = service_fault_batch_ats_sub(gpu_va_space, vma, batch_context, i, outer, block_faults);
        if (status != NV_OK)
            break;

        i += ((*block_faults) - num_faults_before);
    }

    return status;
}

static NV_STATUS service_fault_batch_dispatch(uvm_va_space_t *va_space,
                                              uvm_gpu_va_space_t *gpu_va_space,
                                              uvm_fault_service_batch_context_t *batch_context,
                                              NvU32 fault_index,
                                              NvU32 *block_faults,
                                              bool replay_per_va_block,
                                              const bool hmm_migratable)
{
    NV_STATUS status;
    uvm_va_range_t *va_range = NULL;
    uvm_va_range_t *va_range_next = NULL;
    uvm_va_block_t *va_block;
    uvm_gpu_t *gpu = gpu_va_space->gpu;
    uvm_va_block_context_t *va_block_context =
        gpu->parent->fault_buffer.replayable.block_service_context.block_context;
    uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[fault_index];
    struct mm_struct *mm = va_block_context->mm;
    NvU64 fault_address = current_entry->fault_address;

    (*block_faults) = 0;

    va_range_next = uvm_va_space_iter_gmmu_mappable_first(va_space, fault_address);
    if (va_range_next && (fault_address >= va_range_next->node.start)) {
        UVM_ASSERT(fault_address < va_range_next->node.end);

        va_range = va_range_next;
        va_range_next = uvm_va_range_gmmu_mappable_next(va_range);
    }

    if (va_range)
        status = uvm_va_block_find_create_in_range(va_space, va_range, fault_address, &va_block);
    else if (mm)
        status = uvm_hmm_va_block_find_create(va_space, fault_address, &va_block_context->hmm.vma, &va_block);
    else
        status = NV_ERR_INVALID_ADDRESS;

    if (status == NV_OK)
        status = service_fault_batch_block(gpu, va_block, batch_context, fault_index, hmm_migratable, block_faults);
    else if ((status == NV_ERR_INVALID_ADDRESS) && uvm_ats_can_service_faults(gpu_va_space, mm)) {
        NvU64 outer = ~0ULL;

         UVM_ASSERT(replay_per_va_block ==
                    (gpu->parent->fault_buffer.replayable.replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BLOCK));

        // Limit outer to the minimum of next va_range.start and first
        // fault_address' next UVM_GMMU_ATS_GRANULARITY alignment so that it's
        // enough to check whether the first fault in this dispatch belongs to a
        // GMMU region.
        if (va_range_next) {
            outer = min(va_range_next->node.start,
                           UVM_ALIGN_DOWN(fault_address + UVM_GMMU_ATS_GRANULARITY, UVM_GMMU_ATS_GRANULARITY));
        }

        // ATS lookups are disabled on all addresses within the same
        // UVM_GMMU_ATS_GRANULARITY as existing GMMU mappings (see documentation
        // in uvm_mmu.h). User mode is supposed to reserve VAs as appropriate to
        // prevent any system memory allocations from falling within the NO_ATS
        // range of other GMMU mappings, so this shouldn't happen during normal
        // operation. However, since this scenario may lead to infinite fault
        // loops, we handle it by canceling the fault.
        if (uvm_ats_check_in_gmmu_region(va_space, fault_address, va_range_next)) {
            service_fault_batch_fatal_notify(gpu,
                                             batch_context,
                                             fault_index,
                                             NV_ERR_INVALID_ADDRESS,
                                             UVM_FAULT_CANCEL_VA_MODE_ALL,
                                             block_faults);

            // Do not fail due to logical errors
            status = NV_OK;
        }
        else {
            status = service_fault_batch_ats(gpu_va_space, mm, batch_context, fault_index, outer, block_faults);
        }
    }
    else {
        service_fault_batch_fatal_notify(gpu,
                                         batch_context,
                                         fault_index,
                                         status,
                                         UVM_FAULT_CANCEL_VA_MODE_ALL,
                                         block_faults);

        // Do not fail due to logical errors
        status = NV_OK;
    }

    return status;
}

// Called when a fault in the batch has been marked fatal. Flush the buffer
// under the VA and mmap locks to remove any potential stale fatal faults, then
// service all new faults for just that VA space and cancel those which are
// fatal. Faults in other VA spaces are replayed when done and will be processed
// when normal fault servicing resumes.
static NV_STATUS service_fault_batch_for_cancel(uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status = NV_OK;
    NvU32 i;
    uvm_va_space_t *va_space = batch_context->fatal_va_space;
    uvm_gpu_t *gpu = batch_context->fatal_gpu;
    uvm_gpu_va_space_t *gpu_va_space = NULL;
    struct mm_struct *mm;
    uvm_replayable_fault_buffer_t *replayable_faults = &gpu->parent->fault_buffer.replayable;
    uvm_service_block_context_t *service_context = &gpu->parent->fault_buffer.replayable.block_service_context;
    uvm_va_block_context_t *va_block_context = service_context->block_context;

    UVM_ASSERT(va_space);
    UVM_ASSERT(gpu);

    // Perform the flush and re-fetch while holding the mmap_lock and the
    // VA space lock. This avoids stale faults because it prevents any vma
    // modifications (mmap, munmap, mprotect) from happening between the time HW
    // takes the fault and we cancel it.
    mm = uvm_va_space_mm_retain_lock(va_space);
    uvm_va_block_context_init(va_block_context, mm);
    uvm_va_space_down_read(va_space);

    // We saw fatal faults in this VA space before. Flush while holding
    // mmap_lock to make sure those faults come back (aren't stale).
    //
    // We need to wait until all old fault messages have arrived before
    // flushing, hence UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT.
    status = fault_buffer_flush_locked(gpu->parent,
                                       gpu,
                                       UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT,
                                       UVM_FAULT_REPLAY_TYPE_START,
                                       batch_context);
    if (status != NV_OK)
        goto done;

    // Wait for the flush's replay to finish to give the legitimate faults a
    // chance to show up in the buffer again.
    status = uvm_tracker_wait(&replayable_faults->replay_tracker);
    if (status != NV_OK)
        goto done;

    // We expect all replayed faults to have arrived in the buffer so we can re-
    // service them. The replay-and-wait sequence above will ensure they're all
    // in the HW buffer. When GSP owns the HW buffer, we also have to wait for
    // GSP to copy all available faults from the HW buffer into the shadow
    // buffer.
    status = hw_fault_buffer_flush_locked(gpu->parent, HW_FAULT_BUFFER_FLUSH_MODE_MOVE);
    if (status != NV_OK)
        goto done;

    // If there is no GPU VA space for the GPU, ignore all faults in the VA
    // space. This can happen if the GPU VA space has been destroyed since we
    // unlocked the VA space in service_fault_batch. That means the fatal faults
    // are stale, because unregistering the GPU VA space requires preempting the
    // context and detaching all channels in that VA space. Restart fault
    // servicing from the top.
    gpu_va_space = uvm_gpu_va_space_get(va_space, gpu);
    if (!gpu_va_space)
        goto done;

    // Re-parse the new faults
    batch_context->num_invalid_prefetch_faults = 0;
    batch_context->num_duplicate_faults        = 0;
    batch_context->num_replays                 = 0;
    batch_context->fatal_va_space              = NULL;
    batch_context->fatal_gpu                   = NULL;
    batch_context->has_throttled_faults        = false;

    status = fetch_fault_buffer_entries(gpu->parent, batch_context, FAULT_FETCH_MODE_ALL);
    if (status != NV_OK)
        goto done;

    // No more faults left. Either the previously-seen fatal entry was stale, or
    // RM killed the context underneath us.
    if (batch_context->num_cached_faults == 0)
        goto done;

    ++batch_context->batch_id;

    status = preprocess_fault_batch(gpu->parent, batch_context);
    if (status != NV_OK) {
        if (status == NV_WARN_MORE_PROCESSING_REQUIRED) {
            // Another flush happened due to stale faults or a context-fatal
            // error. The previously-seen fatal fault might not exist anymore,
            // so restart fault servicing from the top.
            status = NV_OK;
        }

        goto done;
    }

    // Search for the target VA space and GPU.
    for (i = 0; i < batch_context->num_coalesced_faults; i++) {
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
        UVM_ASSERT(current_entry->va_space);
        if (current_entry->va_space == va_space && current_entry->gpu == gpu)
            break;
    }

    while (i < batch_context->num_coalesced_faults) {
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];

        if (current_entry->va_space != va_space || current_entry->gpu != gpu)
            break;

        // service_fault_batch_dispatch() doesn't expect unserviceable faults.
        // Just cancel them directly.
        if (current_entry->is_fatal) {
            status = cancel_fault_precise_va(current_entry, UVM_FAULT_CANCEL_VA_MODE_ALL);
            if (status != NV_OK)
                break;

            ++i;
        }
        else {
            uvm_ats_fault_invalidate_t *ats_invalidate = &gpu->parent->fault_buffer.replayable.ats_invalidate;
            NvU32 block_faults;
            const bool hmm_migratable = true;

            ats_invalidate->tlb_batch_pending = false;

            // Service all the faults that we can. We only really need to search
            // for fatal faults, but attempting to service all is the easiest
            // way to do that.
            status = service_fault_batch_dispatch(va_space, gpu_va_space, batch_context, i, &block_faults, false, hmm_migratable);
            if (status != NV_OK) {
                // TODO: Bug 3900733: clean up locking in service_fault_batch().
                // We need to drop lock and retry. That means flushing and
                // starting over.
                if (status == NV_WARN_MORE_PROCESSING_REQUIRED || status == NV_WARN_MISMATCHED_TARGET)
                    status = NV_OK;

                break;
            }

            // Invalidate TLBs before cancel to ensure that fatal faults don't
            // get stuck in HW behind non-fatal faults to the same line.
            status = uvm_ats_invalidate_tlbs(gpu_va_space, ats_invalidate, &batch_context->tracker);
            if (status != NV_OK)
                break;

            while (block_faults-- > 0) {
                current_entry = batch_context->ordered_fault_cache[i];
                if (current_entry->is_fatal) {
                    status = cancel_fault_precise_va(current_entry, current_entry->replayable.cancel_va_mode);
                    if (status != NV_OK)
                        break;
                }

                ++i;
            }
        }
    }

done:
    uvm_va_space_up_read(va_space);
    uvm_va_space_mm_release_unlock(va_space, mm);

    if (status == NV_OK) {
        // There are two reasons to flush the fault buffer here.
        //
        // 1) Functional. We need to replay both the serviced non-fatal faults
        //    and the skipped faults in other VA spaces. The former need to be
        //    restarted and the latter need to be replayed so the normal fault
        //    service mechanism can fetch and process them.
        //
        // 2) Performance. After cancelling the fatal faults, a flush removes
        //    any potential duplicated fault that may have been added while
        //    processing the faults in this batch. This flush also avoids doing
        //    unnecessary processing after the fatal faults have been cancelled,
        //    so all the rest are unlikely to remain after a replay because the
        //    context is probably in the process of dying.
        status = fault_buffer_flush_locked(gpu->parent,
                                           gpu,
                                           UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT,
                                           UVM_FAULT_REPLAY_TYPE_START,
                                           batch_context);
    }

    return status;
}

// Scan the ordered view of faults and group them by different va_blocks
// (managed faults) and service faults for each va_block, in batch.
// Service non-managed faults one at a time as they are encountered during the
// scan.
//
// Fatal faults are marked for later processing by the caller.
static NV_STATUS service_fault_batch(uvm_parent_gpu_t *parent_gpu,
                                     fault_service_mode_t service_mode,
                                     uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status = NV_OK;
    NvU32 i;
    uvm_va_space_t *va_space = NULL;
    uvm_gpu_va_space_t *prev_gpu_va_space = NULL;
    uvm_ats_fault_invalidate_t *ats_invalidate = &parent_gpu->fault_buffer.replayable.ats_invalidate;
    struct mm_struct *mm = NULL;
    const bool replay_per_va_block = service_mode != FAULT_SERVICE_MODE_CANCEL &&
                                     parent_gpu->fault_buffer.replayable.replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BLOCK;
    uvm_service_block_context_t *service_context =
        &parent_gpu->fault_buffer.replayable.block_service_context;
    uvm_va_block_context_t *va_block_context = service_context->block_context;
    bool hmm_migratable = true;
    NvU32 faults_since_lock_yield = 0;

    ats_invalidate->tlb_batch_pending = false;

    for (i = 0; i < batch_context->num_coalesced_faults;) {
        NvU32 block_faults;
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
        uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[current_entry->fault_source.utlb_id];
        uvm_gpu_va_space_t *gpu_va_space;
        const NvU64 fault_leaf_base = UVM_ALIGN_DOWN(current_entry->fault_address,
                                                     UVM_PAGE_SIZE_64K);

        UVM_ASSERT(current_entry->va_space);

        if (current_entry->va_space != va_space) {
            if (prev_gpu_va_space) {
                // TLB entries are invalidated per GPU VA space
                status = uvm_ats_invalidate_tlbs(prev_gpu_va_space, ats_invalidate, &batch_context->tracker);
                if (status != NV_OK)
                    goto fail;

                prev_gpu_va_space = NULL;
            }

            // Fault on a different va_space, drop the lock of the old one...
            if (va_space) {
                uvm_va_space_up_read(va_space);
                uvm_va_space_mm_release_unlock(va_space, mm);
                mm = NULL;
            }

            va_space = current_entry->va_space;

            // ... and take the lock of the new one

            // If an mm is registered with the VA space, we have to retain it
            // in order to lock it before locking the VA space. It is guaranteed
            // to remain valid until we release. If no mm is registered, we
            // can only service managed faults, not ATS/HMM faults.
            mm = uvm_va_space_mm_retain_lock(va_space);
            uvm_va_block_context_init(va_block_context, mm);

            uvm_va_space_down_read(va_space);
        }

        // Some faults could be already fatal if they cannot be handled by
        // the UVM driver
        if (current_entry->is_fatal) {
            ++i;
            if (!batch_context->fatal_va_space) {
                batch_context->fatal_va_space = va_space;
                batch_context->fatal_gpu = current_entry->gpu;
            }

            utlb->has_fatal_faults = true;
            UVM_ASSERT(utlb->num_pending_faults > 0);
            continue;
        }

        gpu_va_space = uvm_gpu_va_space_get(va_space, current_entry->gpu);

        if (prev_gpu_va_space && prev_gpu_va_space != gpu_va_space) {
            status = uvm_ats_invalidate_tlbs(prev_gpu_va_space, ats_invalidate, &batch_context->tracker);
            if (status != NV_OK)
                goto fail;
        }

        prev_gpu_va_space = gpu_va_space;

        // If there is no GPU VA space for the GPU, ignore the fault. This
        // can happen if a GPU VA space is destroyed without explicitly
        // freeing all memory ranges and there are stale entries in the
        // buffer that got fixed by the servicing in a previous batch.
        if (!gpu_va_space) {
            ++i;
            continue;
        }

        if (uvm_moe_slot_pool_range_has_mappings(va_space,
                                                 fault_leaf_base,
                                                 UVM_PAGE_SIZE_64K)) {
            if (prev_gpu_va_space) {
                status = uvm_ats_invalidate_tlbs(prev_gpu_va_space, ats_invalidate, &batch_context->tracker);
                if (status != NV_OK)
                    goto fail;
            }

            uvm_va_space_up_read(va_space);
            uvm_va_space_mm_release_unlock(va_space, mm);
            mm = NULL;
            va_space = NULL;
            prev_gpu_va_space = NULL;

            status = uvm_moe_slot_pool_clear_range_mappings_for_fault(current_entry->va_space,
                                                                      fault_leaf_base,
                                                                      UVM_PAGE_SIZE_64K);
            if (status != NV_OK)
                goto fail;

            continue;
        }

        status = service_fault_batch_dispatch(va_space,
                                              gpu_va_space,
                                              batch_context,
                                              i,
                                              &block_faults,
                                              replay_per_va_block,
                                              hmm_migratable);
        // TODO: Bug 3900733: clean up locking in service_fault_batch().
        if (status == NV_WARN_MORE_PROCESSING_REQUIRED || status == NV_WARN_MISMATCHED_TARGET) {
            if (status == NV_WARN_MISMATCHED_TARGET)
                hmm_migratable = false;
            uvm_va_space_up_read(va_space);
            uvm_va_space_mm_release_unlock(va_space, mm);
            mm = NULL;
            va_space = NULL;
            prev_gpu_va_space = NULL;
            status = NV_OK;
            continue;
        }

        if (status != NV_OK)
            goto fail;

        hmm_migratable = true;
        i += block_faults;
        faults_since_lock_yield += block_faults;

        // Don't issue replays in cancel mode
        if (replay_per_va_block && !batch_context->fatal_va_space) {
            status = push_replay_on_gpu(gpu_va_space->gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);
            if (status != NV_OK)
                goto fail;

            // Increment the batch id if UVM_PERF_FAULT_REPLAY_POLICY_BLOCK
            // is used, as we issue a replay after servicing each VA block
            // and we can service a number of VA blocks before returning.
            ++batch_context->batch_id;
        }

        if (service_mode == FAULT_SERVICE_MODE_REGULAR &&
            uvm_perf_fault_mmap_lock_yield_interval != 0 &&
            faults_since_lock_yield >= uvm_perf_fault_mmap_lock_yield_interval &&
            i < batch_context->num_coalesced_faults) {
            if (prev_gpu_va_space) {
                status = uvm_ats_invalidate_tlbs(prev_gpu_va_space, ats_invalidate, &batch_context->tracker);
                if (status != NV_OK)
                    goto fail;

                prev_gpu_va_space = NULL;
            }

            if (va_space) {
                uvm_va_space_up_read(va_space);
                uvm_va_space_mm_release_unlock(va_space, mm);
                va_space = NULL;
                mm = NULL;
            }

            faults_since_lock_yield = 0;
            cond_resched();
        }
    }

    if (prev_gpu_va_space) {
        NV_STATUS invalidate_status = uvm_ats_invalidate_tlbs(prev_gpu_va_space, ats_invalidate, &batch_context->tracker);
        if (invalidate_status != NV_OK)
            status = invalidate_status;
    }

fail:
    if (va_space) {
        uvm_va_space_up_read(va_space);
        uvm_va_space_mm_release_unlock(va_space, mm);
    }

    return status;
}

// Tells if the given fault entry is the first one in its uTLB
static bool is_first_fault_in_utlb(uvm_fault_service_batch_context_t *batch_context, NvU32 fault_index)
{
    NvU32 i;
    NvU32 utlb_id = batch_context->fault_cache[fault_index].fault_source.utlb_id;

    for (i = 0; i < fault_index; ++i) {
        uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];

        // We have found a prior fault in the same uTLB
        if (current_entry->fault_source.utlb_id == utlb_id)
            return false;
    }

    return true;
}

// Compute the number of fatal and non-fatal faults for a page in the given uTLB
static void faults_for_page_in_utlb(uvm_fault_service_batch_context_t *batch_context,
                                    uvm_va_space_t *va_space,
                                    NvU64 addr,
                                    NvU32 utlb_id,
                                    NvU32 *fatal_faults,
                                    NvU32 *non_fatal_faults)
{
    uvm_gpu_t *gpu = NULL;
    NvU32 i;

    *fatal_faults = 0;
    *non_fatal_faults = 0;

    // Fault filtering is not allowed in the TLB-based fault cancel path
    UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

    for (i = 0; i < batch_context->num_cached_faults; ++i) {
        uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];

        if (!gpu)
            gpu = current_entry->gpu;
        else
            UVM_ASSERT(current_entry->gpu == gpu);

        if (current_entry->fault_source.utlb_id == utlb_id &&
            current_entry->va_space == va_space &&
            current_entry->fault_address == addr) {
            // We have found the page
            if (current_entry->is_fatal)
                ++(*fatal_faults);
            else
                ++(*non_fatal_faults);
        }
    }
}

// Function that tells if there are addresses (reminder: they are aligned to 4K)
// with non-fatal faults only
static bool no_fatal_pages_in_utlb(uvm_fault_service_batch_context_t *batch_context,
                                   NvU32 start_index,
                                   NvU32 utlb_id)
{
    NvU32 i;

    // Fault filtering is not allowed in the TLB-based fault cancel path
    UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

    for (i = start_index; i < batch_context->num_cached_faults; ++i) {
        uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];

        if (current_entry->fault_source.utlb_id == utlb_id) {
            // We have found a fault for the uTLB
            NvU32 fatal_faults;
            NvU32 non_fatal_faults;

            faults_for_page_in_utlb(batch_context,
                                    current_entry->va_space,
                                    current_entry->fault_address,
                                    utlb_id,
                                    &fatal_faults,
                                    &non_fatal_faults);

            if (non_fatal_faults > 0 && fatal_faults == 0)
                return true;
        }
    }

    return false;
}

static void record_fatal_fault_helper(uvm_fault_buffer_entry_t *entry, UvmEventFatalReason reason)
{
    uvm_va_space_t *va_space = entry->va_space;
    uvm_gpu_t *gpu = entry->gpu;

    UVM_ASSERT(va_space);
    UVM_ASSERT(gpu);

    uvm_va_space_down_read(va_space);
    // Record fatal fault event
    uvm_tools_record_gpu_fatal_fault(gpu->id, va_space, entry, reason);
    uvm_va_space_up_read(va_space);
}

// This function tries to find and issue a cancel for each uTLB that meets
// the requirements to guarantee precise fault attribution:
// - No new faults can arrive on the uTLB (uTLB is in lockdown)
// - The first fault in the buffer for a specific uTLB is fatal
// - There are no other addresses in the uTLB with non-fatal faults only
//
// This function and the related helpers iterate over faults as read from HW,
// not through the ordered fault view
//
// TODO: Bug 1766754
// This is very costly, although not critical for performance since we are
// cancelling.
// - Build a list with all the faults within a uTLB
// - Sort by uTLB id
static NV_STATUS try_to_cancel_utlbs(uvm_fault_service_batch_context_t *batch_context)
{
    NvU32 i;

    // Fault filtering is not allowed in the TLB-based fault cancel path
    UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

    for (i = 0; i < batch_context->num_cached_faults; ++i) {
        uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];
        uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[current_entry->fault_source.utlb_id];
        NvU32 gpc_id = current_entry->fault_source.gpc_id;
        NvU32 utlb_id = current_entry->fault_source.utlb_id;
        NvU32 client_id = current_entry->fault_source.client_id;
        uvm_gpu_t *gpu = current_entry->gpu;

        // Only fatal faults are considered
        if (!current_entry->is_fatal)
            continue;

        // Only consider uTLBs in lock-down
        if (!utlb->in_lockdown)
            continue;

        // Issue a single cancel per uTLB
        if (utlb->cancelled)
            continue;

        if (is_first_fault_in_utlb(batch_context, i) &&
            !no_fatal_pages_in_utlb(batch_context, i + 1, utlb_id)) {
            NV_STATUS status;

            record_fatal_fault_helper(current_entry, current_entry->fatal_reason);

            status = push_cancel_on_gpu_targeted(gpu,
                                                 current_entry->instance_ptr,
                                                 gpc_id,
                                                 client_id,
                                                 &batch_context->tracker);
            if (status != NV_OK)
                return status;

            utlb->cancelled = true;
        }
    }

    return NV_OK;
}

static NvU32 find_fatal_fault_in_utlb(uvm_fault_service_batch_context_t *batch_context,
                                      NvU32 utlb_id)
{
    NvU32 i;

    // Fault filtering is not allowed in the TLB-based fault cancel path
    UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

    for (i = 0; i < batch_context->num_cached_faults; ++i) {
        if (batch_context->fault_cache[i].is_fatal &&
            batch_context->fault_cache[i].fault_source.utlb_id == utlb_id)
            return i;
    }

    return i;
}

static NvU32 is_fatal_fault_in_buffer(uvm_fault_service_batch_context_t *batch_context,
                                      uvm_fault_buffer_entry_t *fault)
{
    NvU32 i;

    // Fault filtering is not allowed in the TLB-based fault cancel path
    UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

    for (i = 0; i < batch_context->num_cached_faults; ++i) {
        uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];
        if (cmp_fault_instance_ptr(current_entry, fault) == 0 &&
            current_entry->fault_address == fault->fault_address &&
            current_entry->fault_access_type == fault->fault_access_type &&
            current_entry->fault_source.utlb_id == fault->fault_source.utlb_id) {
            return true;
        }
    }

    return false;
}

// Cancel all faults in the given fault service batch context, even those not
// marked as fatal.
static NV_STATUS cancel_faults_all(uvm_fault_service_batch_context_t *batch_context, UvmEventFatalReason reason)
{
    NV_STATUS status = NV_OK;
    NV_STATUS fault_status;
    uvm_gpu_t *gpu = NULL;
    NvU32 i = 0;

    UVM_ASSERT(reason != UvmEventFatalReasonInvalid);
    UVM_ASSERT(batch_context->num_coalesced_faults > 0);

    while (i < batch_context->num_coalesced_faults && status == NV_OK) {
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
        uvm_va_space_t *va_space = current_entry->va_space;
        bool skip_gpu_va_space;

        gpu = current_entry->gpu;
        UVM_ASSERT(gpu);
        UVM_ASSERT(va_space);

        uvm_va_space_down_read(va_space);

        // If there is no GPU VA space for the GPU, ignore all faults in
        // that GPU VA space. This can happen if the GPU VA space has been
        // destroyed since we unlocked the VA space in service_fault_batch.
        // Ignoring the fault avoids targetting a PDB that might have been
        // reused by another process.
        skip_gpu_va_space = !uvm_gpu_va_space_get(va_space, gpu);

        for (;
             i < batch_context->num_coalesced_faults &&
                 current_entry->va_space == va_space &&
                 current_entry->gpu == gpu;
             current_entry = batch_context->ordered_fault_cache[++i]) {
            uvm_fault_cancel_va_mode_t cancel_va_mode;

            if (skip_gpu_va_space)
                continue;

            if (current_entry->is_fatal) {
                UVM_ASSERT(current_entry->fatal_reason != UvmEventFatalReasonInvalid);
                cancel_va_mode = current_entry->replayable.cancel_va_mode;
            }
            else {
                current_entry->fatal_reason = reason;
                cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
            }

            status = cancel_fault_precise_va(current_entry, cancel_va_mode);
            if (status != NV_OK)
                break;
        }

        uvm_va_space_up_read(va_space);
    }

    // Because each cancel itself triggers a replay, there may be a large number
    // of new duplicated faults in the buffer after cancelling all the known
    // ones. Flushing the buffer discards them to avoid unnecessary processing.
    // Note that we are using one of the GPUs with a fault, but the choice of
    // which one is arbitrary.
    fault_status = fault_buffer_flush_locked(gpu->parent,
                                             gpu,
                                             UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT,
                                             UVM_FAULT_REPLAY_TYPE_START,
                                             batch_context);

    // We report the first encountered error.
    if (status == NV_OK)
        status = fault_status;

    return status;
}

// Function called when the system has found a global error and needs to
// trigger RC in RM.
static void cancel_fault_batch_tlb(uvm_fault_service_batch_context_t *batch_context, UvmEventFatalReason reason)
{
    NvU32 i;

    for (i = 0; i < batch_context->num_coalesced_faults; ++i) {
        NV_STATUS status = NV_OK;
        uvm_fault_buffer_entry_t *current_entry;
        uvm_fault_buffer_entry_t *coalesced_entry;
        uvm_va_space_t *va_space;
        uvm_gpu_t *gpu;

        current_entry = batch_context->ordered_fault_cache[i];
        va_space = current_entry->va_space;
        gpu = current_entry->gpu;

        // The list iteration below skips the entry used as 'head'.
        // Report the 'head' entry explicitly.
        uvm_va_space_down_read(va_space);
        uvm_tools_record_gpu_fatal_fault(gpu->id, va_space, current_entry, reason);

        list_for_each_entry(coalesced_entry, &current_entry->merged_instances_list, merged_instances_list)
            uvm_tools_record_gpu_fatal_fault(gpu->id, va_space, coalesced_entry, reason);
        uvm_va_space_up_read(va_space);

        // We need to cancel each instance pointer to correctly handle faults
        // from multiple contexts.
        status = push_cancel_on_gpu_global(gpu, current_entry->instance_ptr, &batch_context->tracker);
        if (status != NV_OK)
            break;
    }
}

static void cancel_fault_batch(uvm_parent_gpu_t *parent_gpu,
                               uvm_fault_service_batch_context_t *batch_context,
                               UvmEventFatalReason reason)
{
    // Return code is ignored since we're on a global error path and wouldn't be
    // able to recover anyway.
    cancel_faults_all(batch_context, reason);
}


// Current fault cancel algorithm
//
// 1- Disable prefetching to avoid new requests keep coming and flooding the
// buffer.
// LOOP
//   2- Record one fatal fault per uTLB to check if it shows up after the replay
//   3- Flush fault buffer (REPLAY_TYPE_START_ACK_ALL to prevent new faults from
//      coming to TLBs with pending faults)
//   4- Wait for replay to finish
//   5- Fetch all faults from buffer
//   6- Check what uTLBs are in lockdown mode and can be cancelled
//   7- Preprocess faults (order per va_space, fault address, access type)
//   8- Service all non-fatal faults and mark all non-serviceable faults as
//      fatal.
//      8.1- If fatal faults are not found, we are done
//   9- Search for a uTLB which can be targeted for cancel, as described in
//      try_to_cancel_utlbs. If found, cancel it.
// END LOOP
// 10- Re-enable prefetching
//
// NOTE: prefetch faults MUST NOT trigger fault cancel. We make sure that no
// prefetch faults are left in the buffer by disabling prefetching and
// flushing the fault buffer afterwards (prefetch faults are not replayed and,
// therefore, will not show up again)
static NV_STATUS cancel_faults_precise_tlb(uvm_gpu_t *gpu, uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status;
    NV_STATUS tracker_status;
    uvm_replayable_fault_buffer_t *replayable_faults = &gpu->parent->fault_buffer.replayable;
    bool first = true;

    // 1) Disable prefetching to avoid new requests keep coming and flooding
    //    the buffer
    if (gpu->parent->fault_buffer.prefetch_faults_enabled)
        gpu->parent->arch_hal->disable_prefetch_faults(gpu->parent);

    while (1) {
        NvU32 utlb_id;

        // 2) Record one fatal fault per uTLB to check if it shows up after
        // the replay. This is used to handle the case in which the uTLB is
        // being cancelled from behind our backs by RM. See the comment in
        // step 6.
        for (utlb_id = 0; utlb_id <= batch_context->max_utlb_id; ++utlb_id) {
            uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[utlb_id];

            if (!first && utlb->has_fatal_faults) {
                NvU32 idx = find_fatal_fault_in_utlb(batch_context, utlb_id);
                UVM_ASSERT(idx < batch_context->num_cached_faults);

                utlb->prev_fatal_fault = batch_context->fault_cache[idx];
            }
            else {
                utlb->prev_fatal_fault.fault_address = (NvU64)-1;
            }
        }
        first = false;

        // 3) Flush fault buffer. After this call, all faults from any of the
        // faulting uTLBs are before PUT. New faults from other uTLBs can keep
        // arriving. Therefore, in each iteration we just try to cancel faults
        // from uTLBs that contained fatal faults in the previous iterations
        // and will cause the TLB to stop generating new page faults after the
        // following replay with type UVM_FAULT_REPLAY_TYPE_START_ACK_ALL.
        //
        // No need to use UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT since we
        // don't care too much about old faults, just new faults from uTLBs
        // which faulted before the replay.
        status = fault_buffer_flush_locked(gpu->parent,
                                           gpu,
                                           UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT,
                                           UVM_FAULT_REPLAY_TYPE_START_ACK_ALL,
                                           batch_context);
        if (status != NV_OK)
            break;

        // 4) Wait for replay to finish
        status = uvm_tracker_wait(&replayable_faults->replay_tracker);
        if (status != NV_OK)
            break;

        batch_context->num_invalid_prefetch_faults = 0;
        batch_context->num_replays                 = 0;
        batch_context->fatal_va_space              = NULL;
        batch_context->fatal_gpu                   = NULL;
        batch_context->has_throttled_faults        = false;

        // 5) Fetch all faults from buffer
        status = fetch_fault_buffer_entries(gpu->parent, batch_context, FAULT_FETCH_MODE_ALL);
        if (status != NV_OK)
            break;

        ++batch_context->batch_id;

        UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

        // No more faults left, we are done
        if (batch_context->num_cached_faults == 0)
            break;

        // 6) Check what uTLBs are in lockdown mode and can be cancelled
        for (utlb_id = 0; utlb_id <= batch_context->max_utlb_id; ++utlb_id) {
            uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[utlb_id];

            utlb->in_lockdown = false;
            utlb->cancelled   = false;

            if (utlb->prev_fatal_fault.fault_address != (NvU64)-1) {
                // If a previously-reported fault shows up again we can "safely"
                // assume that the uTLB that contains it is in lockdown mode
                // and no new translations will show up before cancel.
                // A fatal fault could only be removed behind our backs by RM
                // issuing a cancel, which only happens when RM is resetting the
                // engine. That means the instance pointer can't generate any
                // new faults, so we won't have an ABA problem where a new
                // fault arrives with the same state.
                if (is_fatal_fault_in_buffer(batch_context, &utlb->prev_fatal_fault))
                    utlb->in_lockdown = true;
            }
        }

        // 7) Preprocess faults
        status = preprocess_fault_batch(gpu->parent, batch_context);
        if (status == NV_WARN_MORE_PROCESSING_REQUIRED)
            continue;
        else if (status != NV_OK)
            break;

        // 8) Service all non-fatal faults and mark all non-serviceable faults
        // as fatal
        status = service_fault_batch(gpu->parent, FAULT_SERVICE_MODE_CANCEL, batch_context);
        UVM_ASSERT(batch_context->num_replays == 0);
        if (status == NV_ERR_NO_MEMORY)
            continue;
        else if (status != NV_OK)
            break;

        // No more fatal faults left, we are done
        if (!batch_context->fatal_va_space)
            break;

        // 9) Search for uTLBs that contain fatal faults and meet the
        // requirements to be cancelled
        try_to_cancel_utlbs(batch_context);
    }

    // 10) Re-enable prefetching
    if (gpu->parent->fault_buffer.prefetch_faults_enabled)
        gpu->parent->arch_hal->enable_prefetch_faults(gpu->parent);

    if (status == NV_OK)
        status = push_replay_on_gpu(gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);

    tracker_status = uvm_tracker_wait(&batch_context->tracker);

    return status == NV_OK? tracker_status: status;
}

static NV_STATUS cancel_faults_precise(uvm_fault_service_batch_context_t *batch_context)
{
    uvm_gpu_t *gpu;

    UVM_ASSERT(batch_context->fatal_va_space);
    UVM_ASSERT(batch_context->fatal_gpu);

    gpu = batch_context->fatal_gpu;
    return service_fault_batch_for_cancel(batch_context);
}

static void enable_disable_prefetch_faults(uvm_parent_gpu_t *parent_gpu,
                                           uvm_fault_service_batch_context_t *batch_context)
{
    // If more than 66% of faults are invalid prefetch accesses, disable
    // prefetch faults for a while.
    // num_invalid_prefetch_faults may be higher than the actual count. See the
    // comment in mark_fault_invalid_prefetch(..).
    // Some tests rely on this logic (and ratio) to correctly disable prefetch
    // fault reporting. If the logic changes, the tests will have to be changed.
    if (parent_gpu->fault_buffer.prefetch_faults_enabled &&
        uvm_perf_reenable_prefetch_faults_lapse_msec > 0 &&
        ((batch_context->num_invalid_prefetch_faults * 3 > parent_gpu->fault_buffer.max_batch_size * 2) ||
         (uvm_enable_builtin_tests &&
          parent_gpu->rm_info.isSimulated &&
          batch_context->num_invalid_prefetch_faults > 5))) {
        uvm_parent_gpu_disable_prefetch_faults(parent_gpu);
    }
    else if (!parent_gpu->fault_buffer.prefetch_faults_enabled) {
        NvU64 lapse = NV_GETTIME() - parent_gpu->fault_buffer.disable_prefetch_faults_timestamp;

        // Reenable prefetch faults after some time
        if (lapse > ((NvU64)uvm_perf_reenable_prefetch_faults_lapse_msec * (1000 * 1000)))
            uvm_parent_gpu_enable_prefetch_faults(parent_gpu);
    }
}

// 处理指定 GPU 上的可重放缺页异常（Replayable Page Faults）。
// 该函数是 UVM 处理 GPU 缺页的顶层循环，它会从硬件的缺页缓冲区中按批次（Batch）
// 提取、预处理并服务缺页异常，最后根据配置的策略触发硬件重放（Replay）。
void uvm_parent_gpu_service_replayable_faults(uvm_parent_gpu_t *parent_gpu)
{
    NvU32 num_replays = 0;   // 记录触发重放操作的次数
    NvU32 num_batches = 0;   // 记录已处理的缺页批次数量
    NvU32 num_throttled = 0; // 记录因内存颠簸（Thrashing）被节流的批次数量
    NV_STATUS status = NV_OK;
    
    // 获取可重放缺页缓冲区及其批处理上下文
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;
    uvm_fault_service_batch_context_t *batch_context = &replayable_faults->batch_service_context;

    // 初始化异步操作追踪器（Tracker），用于等待 GPU 上的内存操作完成
    uvm_tracker_init(&batch_context->tracker);

    // 持续处理缓冲区中的所有缺页异常，直到缓冲区为空或达到阈值限制
    while (1) {
        // 记录处理前的 KV（内核虚拟地址）缺页数，用于后续统计本批次的增量
        NvU64 total_kv_faults_before = parent_gpu->fault_buffer.replayable.stats.num_kv_faults;
        NvU64 total_kv_duplicates_before = parent_gpu->fault_buffer.replayable.stats.num_kv_duplicate_faults;

        // 退出条件：为了防止在严重内存颠簸或高负载下长时间占用 CPU，
        // 限制每次服务调用的最大节流次数和最大批处理次数。
        if (num_throttled >= uvm_perf_fault_max_throttle_per_service ||
            num_batches >= uvm_perf_fault_max_batches_per_service) {
            break;
        }

        // 重置本批次的统计数据和状态标志
        batch_context->num_invalid_prefetch_faults = 0;
        batch_context->num_duplicate_faults        = 0;
        batch_context->num_kv_faults               = 0;
        batch_context->num_kv_duplicate_faults     = 0;
        batch_context->num_replays                 = 0;
        batch_context->fatal_va_space              = NULL;
        batch_context->fatal_gpu                   = NULL;
        batch_context->has_throttled_faults        = false;

        // 1. 获取缺页条目：从硬件环形缓冲区中读取一批缺页记录
        status = fetch_fault_buffer_entries(parent_gpu, batch_context, FAULT_FETCH_MODE_BATCH_READY);
        if (status != NV_OK)
            break;

        // 如果没有获取到任何缺页异常，说明当前缓冲区已空，退出循环
        if (batch_context->num_cached_faults == 0)
            break;

        ++batch_context->batch_id;

        // 2. 预处理：按虚拟地址排序、去重、过滤无效或已取消的地址
        status = preprocess_fault_batch(parent_gpu, batch_context);

        // 累加预处理阶段可能已经触发的重放次数（例如处理特定错误时）
        num_replays += batch_context->num_replays;

        if (status == NV_WARN_MORE_PROCESSING_REQUIRED)
            continue; // 需要重试
        else if (status != NV_OK)
            break;    // 遇到无法恢复的错误

        // 3. 执行服务：核心函数，分配物理页面、更新页表以解决这一批缺页异常
        status = service_fault_batch(parent_gpu, FAULT_SERVICE_MODE_REGULAR, batch_context);

        // 如果采用了 BLOCK 级别的重放策略，或者缓冲区被刷新，服务过程中可能已经触发了重放
        num_replays += batch_context->num_replays;

        enable_disable_prefetch_faults(parent_gpu, batch_context);

        // 计算并更新本批次内与 KV (Kernel Virtual) 相关的统计信息
        batch_context->num_kv_faults = parent_gpu->fault_buffer.replayable.stats.num_kv_faults >= total_kv_faults_before ?
                                       (NvU32)(parent_gpu->fault_buffer.replayable.stats.num_kv_faults - total_kv_faults_before) :
                                       0;
        batch_context->num_kv_duplicate_faults =
            parent_gpu->fault_buffer.replayable.stats.num_kv_duplicate_faults >= total_kv_duplicates_before ?
            (NvU32)(parent_gpu->fault_buffer.replayable.stats.num_kv_duplicate_faults - total_kv_duplicates_before) :
            0;

        // 记录性能日志
        if (uvm_perf_fault_log_counters) {
            log_replayable_fault_counters(parent_gpu,
                                          batch_context->num_cached_faults,
                                          batch_context->num_duplicate_faults,
                                          batch_context->num_kv_faults,
                                          batch_context->num_kv_duplicate_faults);
        }

        // 错误处理逻辑
        if (status != NV_OK) {
            // 全局致命错误（如 OOM 或 ECC 错误）：
            // 无条件取消批次中的所有缺页。由于发生全局错误，我们不强求精确的错误归因。
            // 忽略 cancel 函数的返回值，直接跳出循环。
            cancel_fault_batch(parent_gpu, batch_context, uvm_tools_status_to_fatal_fault_reason(status));
            break;
        }

        // 处理精确致命错误（通常是因为权限不足或地址越界，仅影响特定 VA space）
        if (batch_context->fatal_va_space) {
            // 必须先等待所有已发起的页表操作完成
            status = uvm_tracker_wait(&batch_context->tracker);
            if (status == NV_OK) {
                // 精确取消引发致命错误的具体缺页记录
                status = cancel_faults_precise(batch_context);
                if (status == NV_OK) {
                    // 取消操作至少会发起一次重放指令
                    UVM_ASSERT(batch_context->num_replays > 0);
                    ++num_batches;
                    continue; // 错误已处理，继续下一个批次
                }
            }
            break; // tracker 等待失败或精准取消失败，跳出
        }

        // 5. 根据预设策略触发重放（Replay），通知 GPU 重新执行因缺页暂停的指令
        if (replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BATCH) {
            // 每个批次处理完后立刻发送一条 START 类型的重放指令
            status = push_replay_on_parent_gpu(parent_gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);
            if (status != NV_OK)
                break;
            ++num_replays;
        }
        else if (replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH) {
            // 批次加刷新策略：根据重复率动态决定如何更新硬件的 PUT 指针
            uvm_gpu_buffer_flush_mode_t flush_mode = UVM_GPU_BUFFER_FLUSH_MODE_CACHED_PUT;

            // 如果重复率过高（超出设定比例），强制使用 UPDATE_PUT 模式，防止旧的重复缺页阻塞缓冲区
            if (batch_context->num_duplicate_faults * 100 >
                batch_context->num_cached_faults * replayable_faults->replay_update_put_ratio) {
                flush_mode = UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT;
            }

            status = fault_buffer_flush_locked(parent_gpu, NULL, flush_mode, UVM_FAULT_REPLAY_TYPE_START, batch_context);
            if (status != NV_OK)
                break;
            ++num_replays;
            // 必须等待重放完成
            status = uvm_tracker_wait(&replayable_faults->replay_tracker);
            if (status != NV_OK)
                break;
        }

        // 更新颠簸（Thrashing）节流计数
        if (batch_context->has_throttled_faults)
            ++num_throttled;

        ++num_batches;

        // 如果配置了释放 mmap_lock 的让步间隔，主动调用调度器出让 CPU，避免长时间占用
        if (uvm_perf_fault_mmap_lock_yield_interval != 0)
            cond_resched();
    }

    if (status == NV_WARN_MORE_PROCESSING_REQUIRED)
        status = NV_OK;

    // 安全兜底逻辑：
    // 如果策略要求只发一次重放（ONCE），或者整个循环由于各种原因没有发出过重放指令，
    // 我们必须在这里至少发送一次重放，以防止 GPU 卡死在旧的未记录或未处理的缺页上。
    if ((status == NV_OK && replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_ONCE) ||
        num_replays == 0)
        status = push_replay_on_parent_gpu(parent_gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);

    // 清理追踪器
    uvm_tracker_deinit(&batch_context->tracker);

    // 记录错误日志
    if (status != NV_OK)
        UVM_DBG_PRINT("Error servicing replayable faults on GPU: %s\n", uvm_parent_gpu_name(parent_gpu));
}

void uvm_parent_gpu_enable_prefetch_faults(uvm_parent_gpu_t *parent_gpu)
{
    UVM_ASSERT(parent_gpu->isr.replayable_faults.handling);

    if (!parent_gpu->fault_buffer.prefetch_faults_enabled) {
        parent_gpu->arch_hal->enable_prefetch_faults(parent_gpu);
        parent_gpu->fault_buffer.prefetch_faults_enabled = true;
    }
}

void uvm_parent_gpu_disable_prefetch_faults(uvm_parent_gpu_t *parent_gpu)
{
    UVM_ASSERT(parent_gpu->isr.replayable_faults.handling);

    if (parent_gpu->fault_buffer.prefetch_faults_enabled) {
        parent_gpu->arch_hal->disable_prefetch_faults(parent_gpu);
        parent_gpu->fault_buffer.prefetch_faults_enabled = false;
        parent_gpu->fault_buffer.disable_prefetch_faults_timestamp = NV_GETTIME();
    }
}

const char *uvm_perf_fault_replay_policy_string(uvm_perf_fault_replay_policy_t replay_policy)
{
    BUILD_BUG_ON(UVM_PERF_FAULT_REPLAY_POLICY_MAX != 4);

    switch (replay_policy) {
        UVM_ENUM_STRING_CASE(UVM_PERF_FAULT_REPLAY_POLICY_BLOCK);
        UVM_ENUM_STRING_CASE(UVM_PERF_FAULT_REPLAY_POLICY_BATCH);
        UVM_ENUM_STRING_CASE(UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH);
        UVM_ENUM_STRING_CASE(UVM_PERF_FAULT_REPLAY_POLICY_ONCE);
        UVM_ENUM_STRING_DEFAULT();
    }
}

NV_STATUS uvm_test_get_prefetch_faults_reenable_lapse(UVM_TEST_GET_PREFETCH_FAULTS_REENABLE_LAPSE_PARAMS *params,
                                                      struct file *filp)
{
    params->reenable_lapse = uvm_perf_reenable_prefetch_faults_lapse_msec;

    return NV_OK;
}

NV_STATUS uvm_test_set_prefetch_faults_reenable_lapse(UVM_TEST_SET_PREFETCH_FAULTS_REENABLE_LAPSE_PARAMS *params,
                                                      struct file *filp)
{
    uvm_perf_reenable_prefetch_faults_lapse_msec = params->reenable_lapse;

    return NV_OK;
}

NV_STATUS uvm_test_drain_replayable_faults(UVM_TEST_DRAIN_REPLAYABLE_FAULTS_PARAMS *params, struct file *filp)
{
    uvm_gpu_t *gpu;
    NV_STATUS status = NV_OK;
    uvm_spin_loop_t spin;
    bool pending = true;
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    gpu = uvm_va_space_retain_gpu_by_uuid(va_space, &params->gpu_uuid);
    if (!gpu)
        return NV_ERR_INVALID_DEVICE;

    uvm_spin_loop_init(&spin);

    do {
        uvm_parent_gpu_replayable_faults_isr_lock(gpu->parent);
        pending = uvm_parent_gpu_replayable_faults_pending(gpu->parent);
        uvm_parent_gpu_replayable_faults_isr_unlock(gpu->parent);

        if (!pending)
            break;

        if (fatal_signal_pending(current)) {
            status = NV_ERR_SIGNAL_PENDING;
            break;
        }

        UVM_SPIN_LOOP(&spin);
    } while (uvm_spin_loop_elapsed(&spin) < params->timeout_ns);

    if (pending && status == NV_OK)
        status = NV_ERR_TIMEOUT;

    uvm_gpu_release(gpu);

    return status;
}
