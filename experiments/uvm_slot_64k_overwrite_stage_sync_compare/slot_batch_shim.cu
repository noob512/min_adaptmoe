// SPDX-License-Identifier: Apache-2.0
// C ABI for the isolated normal-64K versus existing-64K-overwrite comparison.
// Reuse the exact managed dataset and context/residency helpers from the
// preceding 512x4MiB probe, then expose the four ioctl-91 modes under test and
// the paired cmd81 actions which construct the timed CPU-remote 64K state.

#include "../uvm_512x4m_batch_compare/uvm_batch_shim.cu"

namespace {

constexpr unsigned int UVM_MOE_SLOT_REMAP_BATCH_IOCTL = 91;
constexpr size_t UVM_MOE_SLOT_REMAP_MAX_ENTRIES_LOCAL = 128;
constexpr uint32_t UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC_LOCAL = 8;
constexpr uint32_t UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_LOCAL = 12;
constexpr uint32_t UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_STAGE_SYNC_LOCAL = 16;
constexpr uint32_t UVM_MOE_SLOT_REMAP_MODE_DIRECT_OVERWRITE_64K_LOCAL = 17;
constexpr uint32_t UVM_MOE_SLOT_REMAP_MODE_DIRECT_OVERWRITE_64K_STAGE_SYNC_LOCAL = 18;
constexpr uint32_t UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_LOCAL = 5;
constexpr uint32_t UVM_PHASE_F_ACTION_FORCE_REMOTE_READ_64K_LOCAL = 20;

struct MoeSlotRemapEntry {
    uint32_t layer_id;
    uint32_t expert_id;
    uint32_t slot_id;
    uint32_t role;
    uint64_t va_base;
    uint64_t bytes;
    uint64_t slot_offset;
    uint32_t flags;
    uint32_t status;
};

struct MoeSlotRemapBatchParams {
    uint32_t mode;
    uint32_t count;
    PhaseFUuid gpu_uuid;
    uint32_t validate_managed;
    uint32_t success_count;
    uint32_t failure_count;
    uint32_t conflict_count;
    uint32_t remap_count;
    uint32_t tlb_invalidate_count;
    uint32_t reserved;
    uint64_t requested_bytes;
    uint64_t accepted_bytes;
    MoeSlotRemapEntry entries[UVM_MOE_SLOT_REMAP_MAX_ENTRIES_LOCAL];
    uint32_t rm_status;
};

static_assert(sizeof(MoeSlotRemapEntry) == 48,
              "slot entry ABI size mismatch");
static_assert(offsetof(MoeSlotRemapEntry, va_base) == 16,
              "slot entry VA offset mismatch");
static_assert(offsetof(MoeSlotRemapEntry, status) == 44,
              "slot entry status offset mismatch");
static_assert(offsetof(MoeSlotRemapBatchParams, requested_bytes) == 56,
              "slot batch requested_bytes offset mismatch");
static_assert(offsetof(MoeSlotRemapBatchParams, entries) == 72,
              "slot batch entries offset mismatch");
static_assert(offsetof(MoeSlotRemapBatchParams, rm_status) == 6216,
              "slot batch rmStatus offset mismatch");
static_assert(sizeof(MoeSlotRemapBatchParams) == 6224,
              "slot batch ABI size mismatch");

bool slot_mode_supported(uint32_t mode)
{
    return mode == UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC_LOCAL ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_LOCAL ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_STAGE_SYNC_LOCAL ||
           mode == UVM_MOE_SLOT_REMAP_MODE_DIRECT_OVERWRITE_64K_LOCAL ||
           mode == UVM_MOE_SLOT_REMAP_MODE_DIRECT_OVERWRITE_64K_STAGE_SYNC_LOCAL;
}

}  // namespace

extern "C" {

uint32_t uvm512_slot_batch_ioctl_command()
{
    return UVM_MOE_SLOT_REMAP_BATCH_IOCTL;
}

size_t uvm512_slot_batch_max_entries()
{
    return UVM_MOE_SLOT_REMAP_MAX_ENTRIES_LOCAL;
}

size_t uvm512_slot_batch_params_size()
{
    return sizeof(MoeSlotRemapBatchParams);
}

uint32_t uvm512_slot_mode_vidmem_prealloc()
{
    return UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC_LOCAL;
}

uint32_t uvm512_slot_mode_direct_64k()
{
    return UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_LOCAL;
}

uint32_t uvm512_slot_mode_direct_64k_stage_sync()
{
    return UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_STAGE_SYNC_LOCAL;
}

uint32_t uvm512_slot_mode_direct_overwrite_64k()
{
    return UVM_MOE_SLOT_REMAP_MODE_DIRECT_OVERWRITE_64K_LOCAL;
}

uint32_t uvm512_slot_mode_direct_overwrite_64k_stage_sync()
{
    return UVM_MOE_SLOT_REMAP_MODE_DIRECT_OVERWRITE_64K_STAGE_SYNC_LOCAL;
}

uint32_t uvm512_phase_f_batch_executor_ioctl_command()
{
    return UVM_PHASE_F_BATCH_EXECUTOR_IOCTL;
}

size_t uvm512_phase_f_batch_executor_max_ranges()
{
    return UVM_PHASE_F_MAX_RANGES;
}

size_t uvm512_phase_f_batch_executor_params_size()
{
    return sizeof(PhaseFBatchExecutorParams);
}

uint32_t uvm512_phase_f_action_prepare_remote_read()
{
    return UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_LOCAL;
}

uint32_t uvm512_phase_f_action_force_remote_read_64k()
{
    return UVM_PHASE_F_ACTION_FORCE_REMOTE_READ_64K_LOCAL;
}

int uvm512_slot_remap_batch(
    void *opaque_context,
    uint32_t mode,
    const uintptr_t *pointers,
    const size_t *sizes,
    const uint32_t *slot_ids,
    size_t count,
    uint32_t layer_id,
    uint32_t expert_id,
    uint32_t *success_count,
    uint32_t *failure_count,
    uint32_t *conflict_count,
    uint32_t *remap_count,
    uint32_t *tlb_invalidate_count,
    uint64_t *requested_bytes,
    uint64_t *accepted_bytes,
    uint32_t *bad_entry_count,
    uint32_t *copy_push_count,
    uint32_t *rm_status)
{
    if (success_count)
        *success_count = 0;
    if (failure_count)
        *failure_count = 0;
    if (conflict_count)
        *conflict_count = 0;
    if (remap_count)
        *remap_count = 0;
    if (tlb_invalidate_count)
        *tlb_invalidate_count = 0;
    if (requested_bytes)
        *requested_bytes = 0;
    if (accepted_bytes)
        *accepted_bytes = 0;
    if (bad_entry_count)
        *bad_entry_count = 0;
    if (copy_push_count)
        *copy_push_count = 0;
    if (rm_status)
        *rm_status = 0;

    DriverBatchContext *context =
        static_cast<DriverBatchContext *>(opaque_context);
    if (!context || context->fd < 0 || !slot_mode_supported(mode) ||
        !pointers || !sizes || !slot_ids || count == 0 ||
        count > UVM_MOE_SLOT_REMAP_MAX_ENTRIES_LOCAL ||
        !success_count || !failure_count || !conflict_count ||
        !remap_count || !tlb_invalidate_count || !requested_bytes ||
        !accepted_bytes || !bad_entry_count || !copy_push_count ||
        !rm_status) {
        set_driver_error("invalid slot batch arguments");
        return -1;
    }

    MoeSlotRemapBatchParams params{};
    params.mode = mode;
    params.count = static_cast<uint32_t>(count);
    params.gpu_uuid = context->gpu_uuid;
    params.validate_managed = 1;
    for (size_t index = 0; index < count; ++index) {
        if (pointers[index] == 0 || sizes[index] == 0) {
            set_driver_error("slot batch contains an empty range");
            return -2;
        }
        MoeSlotRemapEntry &entry = params.entries[index];
        entry.layer_id = layer_id;
        entry.expert_id = expert_id;
        entry.slot_id = slot_ids[index];
        entry.role = 0;
        entry.va_base = pointers[index];
        entry.bytes = sizes[index];
        entry.slot_offset = 0;
        entry.flags = 0;
    }

    errno = 0;
    const int result =
        ioctl(context->fd, UVM_MOE_SLOT_REMAP_BATCH_IOCTL, &params);
    const int saved_errno = errno;
    *success_count = params.success_count;
    *failure_count = params.failure_count;
    *conflict_count = params.conflict_count;
    *remap_count = params.remap_count;
    *tlb_invalidate_count = params.tlb_invalidate_count;
    *requested_bytes = params.requested_bytes;
    *accepted_bytes = params.accepted_bytes;
    *copy_push_count = params.reserved;
    *rm_status = params.rm_status;
    for (size_t index = 0; index < count; ++index) {
        if (params.entries[index].status != 0)
            ++*bad_entry_count;
    }

    if (result != 0) {
        char message[320];
        std::snprintf(message, sizeof(message),
                      "ioctl91 syscall failed mode=%u count=%zu ret=%d "
                      "errno=%d rmStatus=%u success=%u failure=%u bad=%u",
                      mode, count, result, saved_errno, params.rm_status,
                      params.success_count, params.failure_count,
                      *bad_entry_count);
        set_driver_error(message);
        return -3;
    }

    g_driver_last_error.clear();
    return 0;
}

int uvm512_driver_query_residency_full(
    void *opaque_context,
    const uintptr_t *pointers,
    const size_t *sizes,
    size_t count,
    uint32_t *success_count,
    uint64_t *resident_gpu_bytes,
    uint64_t *resident_cpu_bytes,
    uint64_t *resident_any_bytes,
    uint64_t *resident_unknown_bytes)
{
    if (success_count)
        *success_count = 0;
    if (resident_gpu_bytes)
        *resident_gpu_bytes = 0;
    if (resident_cpu_bytes)
        *resident_cpu_bytes = 0;
    if (resident_any_bytes)
        *resident_any_bytes = 0;
    if (resident_unknown_bytes)
        *resident_unknown_bytes = 0;

    DriverBatchContext *context =
        static_cast<DriverBatchContext *>(opaque_context);
    if (!context || context->fd < 0 || !pointers || !sizes || count == 0 ||
        !success_count || !resident_gpu_bytes || !resident_cpu_bytes ||
        !resident_any_bytes || !resident_unknown_bytes) {
        set_driver_error("invalid full residency-query arguments");
        return -1;
    }

    for (size_t begin = 0; begin < count; begin += UVM_PHASE_F_MAX_RANGES) {
        const size_t chunk_count =
            std::min(UVM_PHASE_F_MAX_RANGES, count - begin);
        PhaseFBatchExecutorParams params{};
        params.action = UVM_PHASE_F_ACTION_QUERY_RESIDENCY;
        params.count = static_cast<uint32_t>(chunk_count);
        params.gpu_uuid = context->gpu_uuid;
        params.async = 0;
        for (size_t index = 0; index < chunk_count; ++index) {
            params.ranges[index].base = pointers[begin + index];
            params.ranges[index].length = sizes[begin + index];
        }

        errno = 0;
        const int result =
            ioctl(context->fd, UVM_PHASE_F_BATCH_EXECUTOR_IOCTL, &params);
        const int saved_errno = errno;
        if (result != 0 || params.rm_status != 0 ||
            params.success_count != chunk_count || params.failure_count != 0) {
            char message[320];
            std::snprintf(message, sizeof(message),
                          "cmd81 full residency query failed offset=%zu "
                          "count=%zu ret=%d errno=%d rmStatus=%u "
                          "success=%u failure=%u",
                          begin, chunk_count, result, saved_errno,
                          params.rm_status, params.success_count,
                          params.failure_count);
            set_driver_error(message);
            return -2;
        }
        for (size_t index = 0; index < chunk_count; ++index) {
            const PhaseFRangeResult &entry = params.results[index];
            if (entry.status != 0 ||
                entry.queried_bytes != sizes[begin + index]) {
                set_driver_error("cmd81 full residency entry contract failed");
                return -3;
            }
            *resident_gpu_bytes += entry.resident_gpu_bytes;
            *resident_cpu_bytes += entry.resident_cpu_bytes;
            *resident_any_bytes += entry.resident_any_bytes;
            *resident_unknown_bytes += entry.resident_unknown_bytes;
            ++*success_count;
        }
    }

    g_driver_last_error.clear();
    return 0;
}

int uvm512_driver_prepare_remote_read_full(
    void *opaque_context,
    const uintptr_t *pointers,
    const size_t *sizes,
    size_t count,
    uint32_t *success_count,
    uint64_t *newly_mapped_bytes,
    uint64_t *cpu_candidate_bytes,
    uint64_t *covered_bytes,
    uint64_t *unknown_bytes)
{
    if (success_count)
        *success_count = 0;
    if (newly_mapped_bytes)
        *newly_mapped_bytes = 0;
    if (cpu_candidate_bytes)
        *cpu_candidate_bytes = 0;
    if (covered_bytes)
        *covered_bytes = 0;
    if (unknown_bytes)
        *unknown_bytes = 0;

    DriverBatchContext *context =
        static_cast<DriverBatchContext *>(opaque_context);
    if (!context || context->fd < 0 || !pointers || !sizes || count == 0 ||
        !success_count || !newly_mapped_bytes || !cpu_candidate_bytes ||
        !covered_bytes || !unknown_bytes) {
        set_driver_error("invalid remote-read preparation arguments");
        return -1;
    }

    for (size_t begin = 0; begin < count; begin += UVM_PHASE_F_MAX_RANGES) {
        const size_t chunk_count =
            std::min(UVM_PHASE_F_MAX_RANGES, count - begin);
        PhaseFBatchExecutorParams params{};
        params.action = UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_LOCAL;
        params.count = static_cast<uint32_t>(chunk_count);
        params.gpu_uuid = context->gpu_uuid;
        params.async = 0;
        for (size_t index = 0; index < chunk_count; ++index) {
            params.ranges[index].base = pointers[begin + index];
            params.ranges[index].length = sizes[begin + index];
        }

        errno = 0;
        const int result =
            ioctl(context->fd, UVM_PHASE_F_BATCH_EXECUTOR_IOCTL, &params);
        const int saved_errno = errno;
        if (result != 0 || params.rm_status != 0 ||
            params.success_count != chunk_count || params.failure_count != 0) {
            char message[320];
            std::snprintf(message, sizeof(message),
                          "cmd81 remote-read preparation failed offset=%zu "
                          "count=%zu ret=%d errno=%d rmStatus=%u "
                          "success=%u failure=%u",
                          begin, chunk_count, result, saved_errno,
                          params.rm_status, params.success_count,
                          params.failure_count);
            set_driver_error(message);
            return -2;
        }

        for (size_t index = 0; index < chunk_count; ++index) {
            const PhaseFRangeResult &entry = params.results[index];
            if (entry.status != 0 ||
                entry.queried_bytes != sizes[begin + index]) {
                set_driver_error("cmd81 remote-read entry contract failed");
                return -3;
            }
            *newly_mapped_bytes += entry.resident_gpu_bytes;
            *cpu_candidate_bytes += entry.resident_cpu_bytes;
            *covered_bytes += entry.resident_any_bytes;
            *unknown_bytes += entry.resident_unknown_bytes;
            ++*success_count;
        }
    }

    g_driver_last_error.clear();
    return 0;
}

int uvm512_driver_force_remote_read_64k_full(
    void *opaque_context,
    const uintptr_t *pointers,
    const size_t *sizes,
    size_t count,
    uint32_t *success_count,
    uint64_t *split_bytes,
    uint64_t *already_64k_bytes,
    uint64_t *validated_bytes,
    uint64_t *unknown_bytes)
{
    if (success_count)
        *success_count = 0;
    if (split_bytes)
        *split_bytes = 0;
    if (already_64k_bytes)
        *already_64k_bytes = 0;
    if (validated_bytes)
        *validated_bytes = 0;
    if (unknown_bytes)
        *unknown_bytes = 0;

    DriverBatchContext *context =
        static_cast<DriverBatchContext *>(opaque_context);
    if (!context || context->fd < 0 || !pointers || !sizes || count == 0 ||
        !success_count || !split_bytes || !already_64k_bytes ||
        !validated_bytes || !unknown_bytes) {
        set_driver_error("invalid force-remote-64K arguments");
        return -1;
    }

    for (size_t begin = 0; begin < count; begin += UVM_PHASE_F_MAX_RANGES) {
        const size_t chunk_count =
            std::min(UVM_PHASE_F_MAX_RANGES, count - begin);
        PhaseFBatchExecutorParams params{};
        params.action = UVM_PHASE_F_ACTION_FORCE_REMOTE_READ_64K_LOCAL;
        params.count = static_cast<uint32_t>(chunk_count);
        params.gpu_uuid = context->gpu_uuid;
        params.async = 0;
        for (size_t index = 0; index < chunk_count; ++index) {
            params.ranges[index].base = pointers[begin + index];
            params.ranges[index].length = sizes[begin + index];
        }

        errno = 0;
        const int result =
            ioctl(context->fd, UVM_PHASE_F_BATCH_EXECUTOR_IOCTL, &params);
        const int saved_errno = errno;
        if (result != 0 || params.rm_status != 0 ||
            params.success_count != chunk_count || params.failure_count != 0) {
            char message[320];
            std::snprintf(message, sizeof(message),
                          "cmd81 force-remote-64K failed offset=%zu "
                          "count=%zu ret=%d errno=%d rmStatus=%u "
                          "success=%u failure=%u",
                          begin, chunk_count, result, saved_errno,
                          params.rm_status, params.success_count,
                          params.failure_count);
            set_driver_error(message);
            return -2;
        }

        for (size_t index = 0; index < chunk_count; ++index) {
            const PhaseFRangeResult &entry = params.results[index];
            const uint64_t expected = sizes[begin + index];
            if (entry.status != 0 || entry.queried_bytes != expected ||
                entry.resident_any_bytes != expected ||
                entry.resident_unknown_bytes != 0 ||
                entry.resident_gpu_bytes > expected ||
                entry.resident_cpu_bytes > expected ||
                entry.resident_gpu_bytes + entry.resident_cpu_bytes != expected) {
                set_driver_error("cmd81 force-remote-64K entry contract failed");
                return -3;
            }
            *split_bytes += entry.resident_gpu_bytes;
            *already_64k_bytes += entry.resident_cpu_bytes;
            *validated_bytes += entry.resident_any_bytes;
            *unknown_bytes += entry.resident_unknown_bytes;
            ++*success_count;
        }
    }

    g_driver_last_error.clear();
    return 0;
}

}  // extern "C"
