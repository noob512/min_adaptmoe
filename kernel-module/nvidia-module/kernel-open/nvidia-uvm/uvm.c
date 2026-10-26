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
#include "uvm_global.h"
#include "uvm_gpu_replayable_faults.h"
#include "uvm_tools_init.h"
#include "uvm_lock.h"
#include "uvm_test.h"
#include "uvm_va_space.h"
#include "uvm_va_space_mm.h"
#include "uvm_va_range.h"
#include "uvm_va_block.h"
#include "uvm_tools.h"
#include "uvm_common.h"
#include "uvm_fd_type.h"
#include "uvm_linux_ioctl.h"
#include "uvm_hmm.h"
#include "uvm_hal.h"
#include "uvm_mem.h"
#include "uvm_migrate.h"
#include "uvm_kvmalloc.h"
#include "uvm_test_file.h"
#include "uvm_push.h"
#include "uvm_mmu.h"

#include <linux/mutex.h>
#include <linux/sort.h>
#include <linux/uaccess.h>

#define NVIDIA_UVM_DEVICE_NAME          "nvidia-uvm"

static dev_t g_uvm_base_dev;
static struct cdev g_uvm_cdev;
static const struct file_operations uvm_fops;

static uvm_phase_f_region_t *uvm_phase_f_region_from_node(uvm_range_tree_node_t *node)
{
    if (!node)
        return NULL;

    return container_of(node, uvm_phase_f_region_t, va_node);
}

static bool uvm_phase_f_registered_region_is_valid(const uvm_phase_f_region_t *region);


#define UVM_MOE_SLOT_POOL_MAX_SLOTS 8192
#define UVM_MOE_SLOT_POOL_GPU_COPY_STAGING_CHUNK_BYTES (1ULL * 1024 * 1024)
// A static dual-bank session maps one writable upload VA plus every layer of
// the same bank onto one physical lane. Qwen3's 48 layers therefore require
// 25 ranges per lane. Keep enough fixed metadata for larger model topologies
// while retaining allocation-free policy/fault overlap checks.
#define UVM_MOE_SLOT_POOL_MAX_PTE_RANGES_PER_SLOT 64
#define UVM_MOE_SLOT_DIRECT_MAX_ENTRY_BYTES (4ULL * 1024 * 1024)
// Qwen3 MoE has 48 transformer layers. Keep one bootstrap descriptor plus a
// transition descriptor for every layer resident for the serving lifetime.
// The old 16-entry experimental cap only covered the four-layer probe and
// prevented materializing a complete prefill plan set during expert offload.
#define UVM_MOE_SLOT_PLAN_MAX_PLANS 64
#define UVM_MOE_STATIC_ALIAS_MAX_SESSIONS 4
// A maximum-sized, merely page-aligned entry can intersect three 2 MiB VA
// blocks. The layer ioctl is kvmalloc-backed, so size this fail-closed state
// for the ABI maximum rather than imposing the old 512-block ceiling.
#define UVM_MOE_SLOT_DIRECT_MAX_BLOCKS (UVM_MOE_SLOT_REMAP_MAX_ENTRIES * 3)


typedef struct
{
    NvU64 va_base;
    NvU64 bytes;
} uvm_moe_slot_pool_pte_range_t;

typedef struct
{
    NvBool valid;
    NvU32 layer_id;
    NvU32 slot_id;
    NvU32 owner_expert_id;
    NvU32 state;
    NvU32 range_count;
    NvU32 last_role;
    NvU32 flags;
    NvU32 last_status;
    NvU64 epoch;
    NvU64 batch_id;
    NvU64 first_va_base;
    NvU64 last_va_base;
    NvU64 last_bytes;
    NvU64 total_bytes;
    uvm_mem_t *slot_mem;
    uvm_gpu_t *slot_gpu;
    NvU64 slot_bytes;
    NvU64 backing_generation;
    uvm_mem_t *async_copy_staging;
    uvm_tracker_t async_copy_tracker;
    NvBool async_copy_pending;
    NvU64 async_copy_bytes;
    NvU64 async_copy_slot_offset;
    uvm_page_table_range_vec_t *pte_range_vec;
    uvm_gpu_t *pte_gpu;
    NvU64 pte_va_base;
    NvU64 pte_bytes;
    uvm_moe_slot_pool_pte_range_t pte_ranges[UVM_MOE_SLOT_POOL_MAX_PTE_RANGES_PER_SLOT];
    NvU32 pte_range_count;
    uvm_va_space_t *pte_va_space;
    NvU64 pte_slot_offset;
    NvU32 backing_kind;
    NvU32 copy_status;
    NvU32 pte_status;
    NvU64 copy_count;
    NvU64 copy_bytes;
    NvU64 copy_failed_count;
    NvU64 pte_remap_count;
    NvU64 pte_remap_bytes;
    NvU64 pte_clear_count;
    NvU64 pte_block_unmap_count;
    NvU64 pte_block_tracker_wait_count;
    NvU64 pte_rollback_count;
    NvU64 pte_rollback_bytes;
    NvU64 pte_failure_count;
    NvU64 pte_fault_clear_count;
    NvU64 pte_migrate_clear_count;
    NvU64 pte_unmap_clear_count;
    NvU64 pte_split_clear_count;
    NvU64 pte_release_clear_count;
    NvU64 pte_evict_clear_count;
    NvU64 pte_dangerous_op_reject_count;
    NvU32 pte_mode;
    NvBool pte_va_block_tracked;
    // The UVM file/VA space which owns this logical slot. layer_id and
    // slot_id are scheduler-local identifiers and are not globally unique.
    uvm_va_space_t *owner_va_space;
    // A static-alias session owns the slot PTE metadata independently of the
    // cmd96 owner/generation state. While non-zero, ordinary UVM residency or
    // policy operations must not silently replace these PTEs. Explicit
    // static-alias destroy and VA-space teardown bypass that guard.
    NvU64 static_alias_session_id;
    NvBool static_alias_sealed;
    NvBool static_alias_poisoned;
} uvm_moe_slot_pool_slot_t;

static NV_STATUS uvm_moe_slot_pool_finish_async_copy_locked(uvm_moe_slot_pool_slot_t *slot,
                                                            NvBool wait_for_completion,
                                                            NvBool *completed_out);

static void uvm_moe_slot_pte_seen_bounds_extend(uvm_va_space_t *va_space,
                                                NvU64 va_base,
                                                NvU64 bytes)
{
    NvU64 va_end = va_base + bytes;

    if (!va_space || bytes == 0 || va_end <= va_base)
        return;

    if (va_space->moe_slot_pte_seen_min == 0 ||
        va_base < va_space->moe_slot_pte_seen_min) {
        va_space->moe_slot_pte_seen_min = va_base;
    }
    if (va_end > va_space->moe_slot_pte_seen_max)
        va_space->moe_slot_pte_seen_max = va_end;
}

static NvBool uvm_moe_slot_pte_seen_bounds_overlap(const uvm_va_space_t *va_space,
                                                   NvU64 va_base,
                                                   NvU64 bytes)
{
    NvU64 va_end = va_base + bytes;
    NvU64 seen_min;
    NvU64 seen_max;

    if (!va_space || bytes == 0 || va_end <= va_base)
        return NV_FALSE;

    seen_min = va_space->moe_slot_pte_seen_min;
    seen_max = va_space->moe_slot_pte_seen_max;
    if (seen_min == 0 || seen_max <= seen_min)
        return NV_FALSE;

    return va_base < seen_max && seen_min < va_end;
}

static NvBool uvm_moe_slot_pte_seen_bounds_overlap_ranges(
    const uvm_va_space_t *va_space,
    const UVM_PHASE_F_RANGE *ranges,
    NvU32 count)
{
    NvU32 i;

    for (i = 0; i < count; ++i) {
        if (uvm_moe_slot_pte_seen_bounds_overlap(va_space,
                                                  ranges[i].base,
                                                  ranges[i].length)) {
            return NV_TRUE;
        }
    }

    return NV_FALSE;
}

static NV_STATUS uvm_moe_slot_pool_record_pte_range_locked(uvm_moe_slot_pool_slot_t *slot, NvU64 va_base, NvU64 bytes)
{
    NvU64 va_end = va_base + bytes;

    if (bytes == 0 || va_base + bytes <= va_base)
        return NV_ERR_INVALID_ARGUMENT;

    if (slot->pte_range_count >= UVM_MOE_SLOT_POOL_MAX_PTE_RANGES_PER_SLOT)
        return NV_ERR_INSUFFICIENT_RESOURCES;

    uvm_phase_f_route_log("moe_pte_range action=record layer=%u expert=%u slot=%u role=%u va=0x%llx end=0x%llx bytes=%llu range_index=%u existing_ranges=%u mode=%u va_space=%px\n",
                          slot->layer_id,
                          slot->owner_expert_id,
                          slot->slot_id,
                          slot->last_role,
                          va_base,
                          va_end,
                          bytes,
                          slot->pte_range_count,
                          slot->pte_range_count,
                          slot->pte_mode,
                          slot->pte_va_space);

    slot->pte_ranges[slot->pte_range_count].va_base = va_base;
    slot->pte_ranges[slot->pte_range_count].bytes = bytes;
    ++slot->pte_range_count;

    if (slot->pte_bytes == 0 || va_base < slot->pte_va_base)
        slot->pte_va_base = va_base;

    slot->pte_bytes += bytes;
    uvm_moe_slot_pte_seen_bounds_extend(slot->pte_va_space, va_base, bytes);

    return NV_OK;
}

typedef struct
{
    uvm_moe_slot_pool_slot_t *installed_slots[UVM_MOE_SLOT_REMAP_MAX_ENTRIES];
    NvU32 installed_indices[UVM_MOE_SLOT_REMAP_MAX_ENTRIES];
    NvU32 installed_count;
    NvBool rollback_batch;
} uvm_moe_slot_pool_pte_txn_t;

typedef struct
{
    NvBool installed[UVM_MOE_SLOT_REMAP_MAX_ENTRIES];
    NvU64 block_unmap_count[UVM_MOE_SLOT_REMAP_MAX_ENTRIES];
    NvU64 tracker_wait_count[UVM_MOE_SLOT_REMAP_MAX_ENTRIES];
} uvm_moe_slot_pool_user_backing_group_state_t;

typedef struct
{
    NvU32 layer_id;
    NvU32 slot_id;
    NvU32 entry_index;
} uvm_moe_slot_identity_key_t;

typedef struct
{
    NvU64 va_base;
    NvU64 va_end;
    NvU32 entry_index;
} uvm_moe_slot_source_key_t;

typedef struct
{
    NvU32 pool_index;
    NvU32 layer_id;
    NvU32 slot_id;
    uvm_va_space_t *owner_va_space;
    uvm_mem_t *slot_mem;
    uvm_gpu_t *slot_gpu;
    NvU64 slot_bytes;
    NvU64 backing_generation;
} uvm_moe_slot_plan_validation_entry_t;

// Immutable slot lookup recipe owned by a registered plan. Pool entries live
// in a module-lifetime static array, so retaining indices is safe. Execute
// still validates every mutable slot field under the VA-space and pool locks;
// this recipe only removes repeated sorting and the 8K-entry pool scan.
typedef struct
{
    uvm_moe_slot_plan_validation_entry_t *entries;
    uvm_moe_slot_pool_slot_t **slots;
    uvm_moe_slot_identity_key_t *identity_keys;
    NvU32 count;
    uvm_va_space_t *owner_va_space;
    uvm_gpu_t *gpu;
    NvU64 invalidation_epoch;
    NvU64 source_epoch;
    NvU64 backing_epoch;
} uvm_moe_slot_plan_validation_recipe_t;

typedef struct
{
    uvm_va_block_slot_direct_entry_t blocks[UVM_MOE_SLOT_DIRECT_MAX_BLOCKS];
    NvU32 block_entry_indices[UVM_MOE_SLOT_DIRECT_MAX_BLOCKS];
    uvm_moe_slot_pool_slot_t *slots[UVM_MOE_SLOT_REMAP_MAX_ENTRIES];
    uvm_moe_slot_identity_key_t identity_keys[UVM_MOE_SLOT_REMAP_MAX_ENTRIES];
    uvm_moe_slot_source_key_t source_keys[UVM_MOE_SLOT_REMAP_MAX_ENTRIES];
    NvU32 range_block_counts[UVM_MOE_SLOT_REMAP_MAX_ENTRIES];
    NvBool slot_tracker_initialized[UVM_MOE_SLOT_REMAP_MAX_ENTRIES];
    NvBool has_surviving_mapping[UVM_MOE_SLOT_REMAP_MAX_ENTRIES];
    NvU32 block_count;
} uvm_moe_slot_direct_batch_state_t;

typedef struct
{
    uvm_va_block_t *va_block;
    uvm_va_block_region_t region;
    NvU32 slot_index;
} uvm_moe_slot_transition_clear_slice_t;

typedef struct
{
    uvm_va_block_t *va_block;
    NvU32 first_slice;
    NvU32 slice_count;
    uvm_page_mask_t page_mask;
    // PARK may mix the grouped fast path with the conservative per-slice
    // path. One unusual shared 2 MiB block must not force every unrelated
    // block in the 4,608-entry transaction back to sequential teardown.
    NvBool grouped_eligible;
} uvm_moe_slot_transition_clear_group_t;

typedef struct
{
    uvm_moe_slot_transition_clear_slice_t slices[UVM_MOE_SLOT_DIRECT_MAX_BLOCKS];
    uvm_moe_slot_transition_clear_group_t groups[UVM_MOE_SLOT_DIRECT_MAX_BLOCKS];
    NvU32 slot_block_counts[UVM_MOE_SLOT_REMAP_MAX_ENTRIES];
    NvU32 slice_count;
    NvU32 group_count;
    NvU32 range_count;
    NvU32 unique_block_count;
    NvU64 page_count;
} uvm_moe_slot_transition_clear_state_t;

// Immutable, address-only target topology retained by a registered plan. Do
// not retain uvm_va_block_t pointers here: managed ranges may be destroyed
// while a descriptor is installed. Execute resolves each unique block again
// under the VA-space write lock after validating the plan invalidation epoch.
typedef struct
{
    NvU64 va_base;
    NvU64 bytes;
    NvU32 entry_index;
} uvm_moe_slot_plan_target_slice_t;

typedef struct
{
    NvU64 block_start;
    NvU64 block_end;
    NvU32 first_slice;
    NvU32 slice_count;
    uvm_page_mask_t page_mask;
} uvm_moe_slot_plan_target_group_t;

typedef struct
{
    uvm_moe_slot_plan_target_slice_t *slices;
    uvm_moe_slot_plan_target_group_t *groups;
    NvU32 slice_count;
    NvU32 group_count;
    NvU32 entry_count;
    NvU32 pte_mode;
    NvU64 bytes;
    NvU64 invalidation_epoch;
} uvm_moe_slot_plan_target_recipe_t;

typedef struct
{
    NvBool attempted;
    NvBool used;
    NvBool fallback;
    NvBool direct_cutover;
    NvU64 ranges;
    NvU64 unique_blocks;
    NvU64 direct_cutover_pages;
} uvm_moe_slot_transition_clear_result_t;

typedef struct
{
    uvm_moe_slot_transition_clear_state_t *clear_state;
    uvm_va_block_context_t *block_context;
    NvBool direct_cutover;
    // Set only after every old-clear operation succeeded and its completion
    // was either retained by the transaction aggregate or synchronously
    // drained by the tracker's allocation-failure fallback. Slot PTE metadata
    // still remains intact until the final aggregate wait has completed.
    NvBool completion_guarded;
    NvBool metadata_committed;
} uvm_moe_slot_transition_prepared_clear_t;

typedef struct
{
    NvU64 compute_addr;
    NvU64 completion_addr;
    NvU32 payload;
    NvBool accepted;
    NvBool signaled;
    NvBool compute_waited;
} uvm_moe_slot_plan_fence_request_t;

static DEFINE_MUTEX(g_uvm_moe_slot_pool_lock);
static uvm_moe_slot_pool_slot_t g_uvm_moe_slot_pool[UVM_MOE_SLOT_POOL_MAX_SLOTS];
static NvU64 g_uvm_moe_slot_pool_current_epoch;
static NvU64 g_uvm_moe_slot_pool_current_batch_id;
static NvU64 g_uvm_moe_slot_pool_backing_generation;
static NvU64 g_uvm_moe_slot_pool_backing_epoch;
static NvU64 g_uvm_moe_slot_pool_validated_entries;
static NvU64 g_uvm_moe_slot_pool_overwrite_count;
static NvU64 g_uvm_moe_slot_pool_range_update_count;
static NvU64 g_uvm_moe_slot_pool_full_count;
static NvU64 g_uvm_moe_slot_pool_copy_count;
static NvU64 g_uvm_moe_slot_pool_copy_bytes;
static NvU64 g_uvm_moe_slot_pool_copy_failed_count;
static NvU64 g_uvm_moe_slot_pool_vidmem_alloc_count;
static NvU64 g_uvm_moe_slot_pool_vidmem_evict_count;
static NvU64 g_uvm_moe_slot_pool_vidmem_rebind_count;
static NvU64 g_uvm_moe_slot_pool_vidmem_rebind_bytes;
static NvU64 g_uvm_moe_slot_pool_pte_remap_count;
static NvU64 g_uvm_moe_slot_pool_pte_remap_bytes;
static NvU64 g_uvm_moe_slot_pool_pte_clear_count;
static NvU64 g_uvm_moe_slot_pool_pte_block_unmap_count;
static NvU64 g_uvm_moe_slot_pool_pte_block_tracker_wait_count;
static NvU64 g_uvm_moe_slot_pool_pte_rollback_count;
static NvU64 g_uvm_moe_slot_pool_pte_rollback_bytes;
static NvU64 g_uvm_moe_slot_pool_pte_failure_count;
static NvU64 g_uvm_moe_slot_pool_pte_fault_clear_count;
static NvU64 g_uvm_moe_slot_pool_pte_migrate_clear_count;
static NvU64 g_uvm_moe_slot_pool_pte_unmap_clear_count;
static NvU64 g_uvm_moe_slot_pool_pte_split_clear_count;
static NvU64 g_uvm_moe_slot_pool_pte_release_clear_count;
static NvU64 g_uvm_moe_slot_pool_pte_evict_clear_count;
static NvU64 g_uvm_moe_slot_pool_pte_dangerous_op_reject_count;
static NvU64 g_uvm_moe_slot_pool_release_count;
static NvU64 g_uvm_moe_slot_pool_release_bytes;
static NvU64 g_uvm_moe_slot_pool_release_failed_count;
static NvU64 g_uvm_moe_slot_pool_close_reclaim_count;
static NvU64 g_uvm_moe_slot_pool_close_reclaim_bytes;
static NvU64 g_uvm_moe_slot_pool_close_reclaim_failed_count;
static NvU64 g_uvm_moe_slot_pool_owner_mismatch_count;
static NvU64 g_uvm_moe_slot_pool_no_evict_budget_reject_count;

typedef struct
{
    NvU32 driver_layer_id;
    NvU32 slot_id;
    NvU32 pool_index;
    NvU64 bytes;
    NvU64 slot_offset;
    NvU64 slot_bytes;
    uvm_mem_t *slot_mem;
} uvm_moe_slot_plan_bank_lane_t;

typedef struct
{
    NvBool valid;
    NvU64 plan_id;
    NvU32 layer_id;
    uvm_va_space_t *owner_va_space;
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *batch;
    NvU64 *allocation_ids;
    uvm_va_block_slot_copy_plan_t *copy_plan;
    uvm_moe_slot_plan_target_recipe_t *target_recipe;
    uvm_moe_slot_plan_validation_recipe_t *validation_recipe;
    NvU64 invalidation_epoch;
    NvU32 bank_id;
    NvBool pte_prewarmed;
    // Complete 2 MiB groups can use the existing unreachable-leaf/PDE
    // shadow publish path. Sparse 64 KiB groups deliberately use the
    // copy-dependent live-leaf path so mappings outside their cold mask are
    // never replaced by a parent cutover.
    NvBool shadow_pte_eligible;
    NvBool partial_64k;
} uvm_moe_slot_plan_t;

typedef enum
{
    UVM_MOE_SLOT_PLAN_STATE_INACTIVE = 0,
    UVM_MOE_SLOT_PLAN_STATE_ACTIVE,
    UVM_MOE_SLOT_PLAN_STATE_TRANSITIONING,
    UVM_MOE_SLOT_PLAN_STATE_PUBLISHING,
    UVM_MOE_SLOT_PLAN_STATE_POISONED,
} uvm_moe_slot_plan_state_t;

// The experimental serving contract is one process/VA-space and one fixed
// bank. Keep descriptor plans in a separately serialized registry so compact
// execution can hold this lock across the owner transaction without exposing
// a plan to concurrent destroy or another generation.
static DEFINE_MUTEX(g_uvm_moe_slot_plan_lock);
static uvm_moe_slot_plan_t g_uvm_moe_slot_plans[UVM_MOE_SLOT_PLAN_MAX_PLANS];
static uvm_va_space_t *g_uvm_moe_slot_plan_owner_va_space;
static NvU64 g_uvm_moe_slot_plan_generation;
static NvU64 g_uvm_moe_slot_plan_generation_counter;
static NvU64 g_uvm_moe_slot_plan_invalidation_epoch;
static NvBool g_uvm_moe_slot_plan_poisoned;
static NvBool g_uvm_moe_slot_plan_bootstrap_registered;
static NvBool g_uvm_moe_slot_plan_registration_sealed;
static uvm_moe_slot_plan_state_t g_uvm_moe_slot_plan_state;
static NvU64 g_uvm_moe_slot_plan_transition_id;
static NvU64 g_uvm_moe_slot_plan_transition_counter;
static NvProcessorUuid g_uvm_moe_slot_plan_gpu_uuid;
static NvU32 g_uvm_moe_slot_plan_bank_count;
static uvm_moe_slot_plan_bank_lane_t *g_uvm_moe_slot_plan_bank_lanes;
static uvm_moe_slot_plan_bank_lane_t *g_uvm_moe_slot_plan_second_bank_lanes;
static NvU32 g_uvm_moe_slot_plan_registered_bank_count;
static NvU32 g_uvm_moe_slot_plan_active_bank = NV_U32_MAX;
// plan_id of the mapping currently retained by each physical bank. An
// inactive bank still has live PTE metadata until that same bank is reused.
static NvU64 g_uvm_moe_slot_plan_bank_resident_plan_ids[2];
// A transaction which reserved a generation may have published only part of
// its target before failing. It is not yet a resident bank owner, so retain
// its plan id separately for the generation-free abort sweep.
static NvU64 g_uvm_moe_slot_plan_failed_target_plan_id;
static NvU64 g_uvm_moe_slot_plan_fence_compute_addr;
static NvU64 g_uvm_moe_slot_plan_fence_completion_addr;
static NvU32 g_uvm_moe_slot_plan_fence_last_payload;

typedef struct
{
    NvU64 va_base;
    NvU64 bytes;
    NvU64 slot_offset;
    NvU32 backing_layer_id;
    NvU32 backing_expert_id;
    NvU32 backing_slot_id;
    NvU32 pool_index;
    NvU32 flags;
    NvBool active;
} uvm_moe_static_alias_record_t;

typedef struct
{
    NvBool valid;
    NvBool sealed;
    NvBool parked;
    NvBool poisoned;
    NvU64 session_id;
    uvm_va_space_t *owner_va_space;
    NvProcessorUuid gpu_uuid;
    uvm_moe_static_alias_record_t *aliases;
    NvU32 alias_count;
    NvU64 alias_bytes;
    NvU64 backing_bytes;
    // Candidate read-only consumer records for a multi-ioctl REBIND. They
    // never participate in fault handling or PTE cleanup until COMMIT swaps
    // them into aliases while the session remains parked.
    uvm_moe_static_alias_record_t *rebind_aliases;
    NvU32 rebind_alias_count;
    NvU32 rebind_expected_alias_count;
    NvU64 rebind_alias_bytes;
    NvU64 rebind_expected_alias_bytes;
    NvU64 rebind_id;
    // A RESIZE transaction stages the complete upload + consumer set. The
    // ordinary form stages consumers only and preserves every upload/backing.
    NvBool rebind_resize;
    // The expensive record/slot audit is performed when a topology is
    // created or changed. PARK/RESUME may reuse that proof while the static
    // alias guards keep the directory immutable under the plan lock.
    NvU64 topology_generation;
    NvU64 certificate_generation;
    NvU64 consumer_alias_bytes;
    NvU32 consumer_alias_count;
    NvU32 upload_alias_count;
    NvBool certificate_valid;
    NvBool certificate_parked;
    // Sorted full 2 MiB VA blocks touched by consumer aliases whose complete
    // CPU and GPU 4K child tables have both been prepared. PARK/RESUME switches
    // their parent entries even when only part of a block contains experts.
    NvU64 *dual_view_block_starts;
    NvU32 dual_view_block_count;
    NvU64 dual_view_generation;
    NvBool dual_view_full_coverage;
    // SEAL(DISCARD_DUAL) is a scheduler-quiescent request to return to the
    // ordinary UVM hierarchy before a residency transaction. Full-coverage
    // mode may perform one legacy PARK while this certificate is set; an
    // implicit recipe invalidation must never grant the same exception.
    NvBool dual_view_explicitly_discarded;
} uvm_moe_static_alias_session_t;

// Static aliases are logically independent of registered cmd96 plans, but
// use the same outer serialization lock. This makes the global lock order
// unambiguous and lets legacy commands reject a live alias session without a
// second lock or a lock-order inversion:
//   plan/static-session -> retained mm -> VA-space write -> slot pool.
static uvm_moe_static_alias_session_t
    g_uvm_moe_static_alias_sessions[UVM_MOE_STATIC_ALIAS_MAX_SESSIONS];
static NvU64 g_uvm_moe_static_alias_session_counter;
static NvU64 g_uvm_moe_static_alias_rebind_counter;

static unsigned long long uvm_moe_slot_pool_vidmem_budget_bytes __read_mostly = 512ULL * 1024 * 1024;
module_param(uvm_moe_slot_pool_vidmem_budget_bytes, ullong, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_slot_pool_vidmem_budget_bytes,
                 "MoE gpu_copy_only driver-owned vidmem backing budget in bytes; 0 disables budget eviction");

static int uvm_moe_slot_pte_remap_enable __read_mostly = 0;
module_param(uvm_moe_slot_pte_remap_enable, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_slot_pte_remap_enable,
                 "Enable experimental smoke-only MoE slot-backed GPU PTE remap. Default: 0.");

static int uvm_moe_static_alias_dual_view_enable __read_mostly = 0;
module_param(uvm_moe_static_alias_dual_view_enable, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_static_alias_dual_view_enable,
                 "Prebuild complete CPU/GPU 4K children for expert-touched 2M blocks and switch their parent entries at quiesced PARK/RESUME boundaries. Experimental default: 0.");

static int uvm_moe_static_alias_dual_view_require_full __read_mostly = 0;
module_param(uvm_moe_static_alias_dual_view_require_full, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_static_alias_dual_view_require_full,
                 "Reject SEAL unless every read-only static-alias consumer is covered by the prebuilt dual-view parent switch. Experimental default: 0.");

static unsigned long long uvm_moe_slot_pte_remap_max_bytes __read_mostly = 64ULL * 1024;
module_param(uvm_moe_slot_pte_remap_max_bytes, ullong, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_slot_pte_remap_max_bytes,
                 "Maximum bytes accepted by one experimental MoE slot PTE remap entry. Default: 64KiB.");

static int uvm_moe_slot_user_backing_defer_tlb_invalidate __read_mostly = 0;
module_param(uvm_moe_slot_user_backing_defer_tlb_invalidate, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_slot_user_backing_defer_tlb_invalidate,
                 "Experimentally defer mode=7 user-backing map-side TLB invalidates and issue one batch invalidate. Default: 0.");

static int uvm_moe_slot_user_backing_grouped_remap __read_mostly = 0;
module_param(uvm_moe_slot_user_backing_grouped_remap, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_slot_user_backing_grouped_remap,
                 "Experimentally group mode=7 user-backing remap by target VA block. Default: 0.");

static int uvm_phase_f_remote_read_full_tlb_invalidate __read_mostly = 0;
module_param(uvm_phase_f_remote_read_full_tlb_invalidate, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_phase_f_remote_read_full_tlb_invalidate,
                 "After each successful audited Phase-F remote-read batch, synchronously invalidate all GPU TLBs. Default: 0.");

static int uvm_moe_slot_transition_fast_index __read_mostly = 0;
module_param(uvm_moe_slot_transition_fast_index, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_slot_transition_fast_index,
                 "Use sorted validation, one-pass slot lookup, and nlogn block sorting for mode=20. Default: 0.");

static int uvm_moe_slot_transition_direct_cutover __read_mostly = 0;
module_param(uvm_moe_slot_transition_direct_cutover, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_slot_transition_direct_cutover,
                 "For grouped mode=20 transitions, unmap old slot PTEs without eagerly restoring GPU mappings to CPU-resident pages. Default: 0.");

static int uvm_moe_slot_plan_overlap_copy_prepare __read_mostly = 1;
module_param(uvm_moe_slot_plan_overlap_copy_prepare, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_slot_plan_overlap_copy_prepare,
                 "For registered plans, submit CE copy after target invalidation and overlap it with shadow PTE hierarchy preparation. Default: 1.");

static int uvm_moe_slot_plan_register_pte_prewarm __read_mostly = 1;
module_param(uvm_moe_slot_plan_register_pte_prewarm, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_slot_plan_register_pte_prewarm,
                 "Preallocate registered-plan sliced-64K PTE hierarchy during registration without publishing valid leaf PTEs. Default: 1.");

static int uvm_moe_slot_plan_cache_copy_runs __read_mostly = 0;
module_param(uvm_moe_slot_plan_cache_copy_runs, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_slot_plan_cache_copy_runs,
                 "Cache registered-plan CE-visible source/destination runs at registration and epoch-validate them at execute. Experimental default: 0.");

static int uvm_moe_slot_plan_cache_metadata_recipe __read_mostly = 0;
module_param(uvm_moe_slot_plan_cache_metadata_recipe, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_slot_plan_cache_metadata_recipe,
                 "Cache registered-plan target topology and dual-bank old-clear recipes. Any missing or stale recipe is rejected before transition state. Experimental default: 0.");

static int uvm_moe_slot_plan_fast_validation __read_mostly = 0;
module_param(uvm_moe_slot_plan_fast_validation, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_slot_plan_fast_validation,
                 "Use registered-plan pool indices and cached identity ordering while retaining per-slot execute validation. Experimental default: 0.");

static int uvm_moe_slot_plan_epoch_validation __read_mostly = 0;
module_param(uvm_moe_slot_plan_epoch_validation, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_slot_plan_epoch_validation,
                 "Validate registered plans with aggregate VA/source/backing epochs and bind cached slot pointers without rescanning every descriptor. Requires fast validation and immutable copy/metadata recipes. Experimental default: 0.");

static int uvm_moe_slot_plan_shadow_pte_prefill __read_mostly = 0;
module_param(uvm_moe_slot_plan_shadow_pte_prefill, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_slot_plan_shadow_pte_prefill,
                 "Fill unreachable registered-plan 64K leaf PTEs while copy runs, then publish only the parent PDE after copy completion. Experimental default: 0.");

static int uvm_moe_slot_plan_dual_bank_event __read_mostly = 0;
module_param(uvm_moe_slot_plan_dual_bank_event, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_slot_plan_dual_bank_event,
                 "Enable registered-plan two-bank copy-first transitions after a userspace-synchronized main-stream CUDA event. This overlaps only transition metadata/PTE work with CE copy; it does not authorize compute/copy overlap. Experimental default: 0.");

static int uvm_moe_slot_plan_cuda_fence __read_mostly = 0;
module_param(uvm_moe_slot_plan_cuda_fence, int, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_moe_slot_plan_cuda_fence,
                 "Require command-96 UVM semaphore-pool compute/completion fences for registered-plan execution. Experimental default: 0.");

typedef enum
{
    UVM_MOE_SLOT_POOL_COPY_NONE = 0,
    UVM_MOE_SLOT_POOL_COPY_SYSMEM = 1,
    UVM_MOE_SLOT_POOL_COPY_GPU = 2,
    UVM_MOE_SLOT_POOL_PREALLOC_VIDMEM = 3,
    UVM_MOE_SLOT_POOL_COPY_GPU_ASYNC = 4,
} uvm_moe_slot_pool_copy_mode_t;

typedef enum
{
    UVM_MOE_SLOT_PTE_CLEAR_REASON_FAULT = 0,
    UVM_MOE_SLOT_PTE_CLEAR_REASON_MIGRATE,
    UVM_MOE_SLOT_PTE_CLEAR_REASON_DISCARD,
    UVM_MOE_SLOT_PTE_CLEAR_REASON_UNMAP,
    UVM_MOE_SLOT_PTE_CLEAR_REASON_SPLIT,
    UVM_MOE_SLOT_PTE_CLEAR_REASON_RELEASE,
    UVM_MOE_SLOT_PTE_CLEAR_REASON_EVICT,
    UVM_MOE_SLOT_PTE_CLEAR_REASON_DANGEROUS_REJECT,
} uvm_moe_slot_pte_clear_reason_t;

static void uvm_moe_slot_pool_release_va_space_mappings(uvm_va_space_t *va_space);
static void uvm_moe_slot_plan_destroy_va_space(uvm_va_space_t *va_space);
static NV_STATUS uvm_moe_slot_plan_next_generation_locked(NvU64 *generation_out);
static NV_STATUS uvm_moe_slot_pool_clear_range_mappings_for_release_locked(uvm_va_space_t *va_space,
                                                                           struct mm_struct *mm,
                                                                           NvU64 base,
                                                                           NvU64 bytes);

static void uvm_copy_region_to_query(uvm_phase_f_region_t *region,
                                     UVM_QUERY_REGION_PARAMS *params,
                                     NvU32 found_by_va)
{
    params->region_id = region->region_id;
    params->base = region->base;
    params->length = region->length;
    params->state = region->state;
    params->found_by_va = found_by_va;
}

static void uvm_copy_region_to_query_by_va(uvm_phase_f_region_t *region,
                                           UVM_QUERY_REGION_BY_VA_PARAMS *params)
{
    params->region_id = region->region_id;
    params->base = region->base;
    params->length = region->length;
    params->state = region->state;
    params->found_by_va = 1;
}

bool uvm_file_is_nvidia_uvm(struct file *filp)
{
    return (filp != NULL) && (filp->f_op == &uvm_fops);
}

bool uvm_file_is_nvidia_uvm_va_space(struct file *filp)
{
    return uvm_file_is_nvidia_uvm(filp) && uvm_fd_type(filp, NULL) == UVM_FD_VA_SPACE;
}

static NV_STATUS uvm_api_mm_initialize(UVM_MM_INITIALIZE_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space;
    uvm_va_space_mm_t *va_space_mm;
    struct file *uvm_file;
    struct mm_struct *mm;
    NV_STATUS status;

    uvm_file = fget(params->uvmFd);
    if (!uvm_file_is_nvidia_uvm(uvm_file)) {
        status = NV_ERR_INVALID_ARGUMENT;
        goto err;
    }

    if (uvm_fd_type(uvm_file, (void **)&va_space) != UVM_FD_VA_SPACE) {
        status = NV_ERR_INVALID_ARGUMENT;
        goto err;
    }

    // Tell userspace the MM FD is not required and it may be released
    // with no loss of functionality.
    if (!uvm_va_space_mm_enabled(va_space)) {
        status = NV_WARN_NOTHING_TO_DO;
        goto err;
    }

    status = uvm_fd_type_init(filp);
    if (status != NV_OK)
        goto err;

    va_space_mm = &va_space->va_space_mm;
    uvm_spin_lock(&va_space_mm->lock);
    switch (va_space->va_space_mm.state) {
        // We only allow the va_space_mm to be initialised once. If
        // userspace passed the UVM FD to another process it is up to
        // userspace to ensure it also passes the UVM MM FD that
        // initialised the va_space_mm or arranges some other way to keep
        // a reference on the FD.
        case UVM_VA_SPACE_MM_STATE_ALIVE:
            status = NV_ERR_INVALID_STATE;
            goto err_release_unlock;
            break;

        // Once userspace has released the va_space_mm the GPU is
        // effectively dead and no new work can be started. We don't
        // support re-initializing once userspace has closed the FD.
        case UVM_VA_SPACE_MM_STATE_RELEASED:
            status = NV_ERR_PAGE_TABLE_NOT_AVAIL;
            goto err_release_unlock;
            break;

        // Keep the warnings at bay
        case UVM_VA_SPACE_MM_STATE_UNINITIALIZED:
            mm = va_space->va_space_mm.mm;
            if (!mm || !mmget_not_zero(mm)) {
                status = NV_ERR_PAGE_TABLE_NOT_AVAIL;
                goto err_release_unlock;
            }

            va_space_mm->state = UVM_VA_SPACE_MM_STATE_ALIVE;
            break;

        default:
            UVM_ASSERT(0);
            break;
    }
    uvm_spin_unlock(&va_space_mm->lock);
    uvm_fd_type_set(filp, UVM_FD_MM, uvm_file);

    return NV_OK;

err_release_unlock:
    uvm_spin_unlock(&va_space_mm->lock);
    uvm_fd_type_set(filp, UVM_FD_UNINITIALIZED, NULL);

err:
    if (uvm_file)
        fput(uvm_file);

    return status;
}

// Called when opening /dev/nvidia-uvm. This code doesn't take any UVM locks, so
// there's no need to acquire g_uvm_global.pm.lock, but if that changes the PM
// lock will need to be taken.
static int uvm_open(struct inode *inode, struct file *filp)
{
    struct address_space *mapping;
    NV_STATUS status = uvm_global_get_status();

    if (status != NV_OK)
        return -nv_status_to_errno(status);

    mapping = uvm_kvmalloc(sizeof(*mapping));
    if (!mapping)
        return -ENOMEM;

    // By default all struct files on the same inode share the same
    // address_space structure (the inode's) across all processes. This means
    // unmap_mapping_range would unmap virtual mappings across all processes on
    // that inode.
    //
    // Since the UVM driver uses the mapping offset as the VA of the file's
    // process, we need to isolate the mappings to each process.
    address_space_init_once(mapping);
    mapping->host = inode;

    // Some paths in the kernel, for example force_page_cache_readahead which
    // can be invoked from user-space via madvise MADV_WILLNEED and fadvise
    // POSIX_FADV_WILLNEED, check the function pointers within
    // file->f_mapping->a_ops for validity. However, those paths assume that a_ops
    // itself is always valid. Handle that by using the inode's a_ops pointer,
    // which is what f_mapping->a_ops would point to anyway if we weren't re-
    // assigning f_mapping.
    mapping->a_ops = inode->i_mapping->a_ops;

    filp->private_data = NULL;
    filp->f_mapping = mapping;

    return NV_OK;
}

static int uvm_open_entry(struct inode *inode, struct file *filp)
{
   UVM_ENTRY_RET(uvm_open(inode, filp));
}

static void uvm_release_deferred(void *data)
{
    uvm_va_space_t *va_space = data;

    // Since this function is only scheduled to run when uvm_release() fails
    // to trylock-acquire the pm.lock, the following acquisition attempt
    // is expected to block this thread, and cause it to remain blocked until
    // uvm_resume() releases the lock. As a result, the deferred release
    // kthread queue may stall for long periods of time.
    uvm_down_read(&g_uvm_global.pm.lock);

    uvm_moe_slot_pool_release_va_space_mappings(va_space);
    uvm_va_space_destroy(va_space);

    uvm_up_read(&g_uvm_global.pm.lock);
}

static void uvm_release_va_space(struct file *filp, uvm_va_space_t *va_space)
{
    int ret;

    filp->private_data = NULL;
    filp->f_mapping = NULL;

    // Because the kernel discards the status code returned from this release
    // callback, early exit in case of a pm.lock acquisition failure is not
    // an option. Instead, the teardown work normally performed synchronously
    // needs to be scheduled to run after uvm_resume() releases the lock.
    if (uvm_down_read_trylock(&g_uvm_global.pm.lock)) {
        uvm_moe_slot_pool_release_va_space_mappings(va_space);
        uvm_va_space_destroy(va_space);
        uvm_up_read(&g_uvm_global.pm.lock);
    }
    else {
        // Remove references to this inode from the address_space. This isn't
        // strictly necessary, as any CPU mappings of this file have already
        // been destroyed, and va_space->mapping won't be used again. Still,
        // the va_space survives the inode if its destruction is deferred, in
        // which case the references are rendered stale.
        address_space_init_once(va_space->mapping);

        nv_kthread_q_item_init(&va_space->deferred_release_q_item, uvm_release_deferred, va_space);
        ret = nv_kthread_q_schedule_q_item(&g_uvm_global.deferred_release_q, &va_space->deferred_release_q_item);
        UVM_ASSERT(ret != 0);
    }
}

static void uvm_release_mm(struct file *filp, struct file *uvm_file)
{
    uvm_va_space_t *va_space = uvm_va_space_get(uvm_file);
    uvm_va_space_mm_t *va_space_mm = &va_space->va_space_mm;
    struct mm_struct *mm = va_space_mm->mm;

    uvm_kvfree(filp->f_mapping);

    if (uvm_va_space_mm_enabled(va_space)) {
        uvm_va_space_mm_unregister(va_space);

        if (uvm_va_space_mm_enabled(va_space))
            uvm_mmput(mm);

        va_space_mm->mm = NULL;
        fput(uvm_file);
    }
}

static int uvm_release(struct inode *inode, struct file *filp)
{
    void *ptr;
    uvm_fd_type_t fd_type = uvm_fd_type(filp, &ptr);

    switch (fd_type) {
        case UVM_FD_UNINITIALIZED:
            uvm_kvfree(filp->f_mapping);
            break;

        case UVM_FD_VA_SPACE:
            uvm_release_va_space(filp, (uvm_va_space_t *)ptr);
            break;

        case UVM_FD_MM:
            uvm_release_mm(filp, (struct file *)ptr);
            break;

        case UVM_FD_TEST:
            uvm_test_file_release(filp, (uvm_test_file_t *)ptr);
            break;

        default:
            UVM_ASSERT_MSG(0, "Unexpected fd type: %d\n", fd_type);
    }

    return 0;
}

static int uvm_release_entry(struct inode *inode, struct file *filp)
{
   UVM_ENTRY_RET(uvm_release(inode, filp));
}

static void uvm_destroy_vma_managed(struct vm_area_struct *vma, bool make_zombie)
{
    uvm_va_range_managed_t *managed_range, *managed_range_next;
    NvU64 size = 0;

    uvm_assert_rwsem_locked_write(&uvm_va_space_get(vma->vm_file)->lock);
    uvm_for_each_va_range_managed_in_vma_safe(managed_range, managed_range_next, vma) {
        // On exit_mmap (process teardown), current->mm is cleared so
        // uvm_va_range_vma_current would return NULL.
        UVM_ASSERT(uvm_va_range_vma(managed_range) == vma);
        UVM_ASSERT(managed_range->va_range.node.start >= vma->vm_start);
        UVM_ASSERT(managed_range->va_range.node.end   <  vma->vm_end);
        size += uvm_va_range_size(&managed_range->va_range);
        if (make_zombie)
            uvm_va_range_zombify(managed_range);
        else
            uvm_va_range_destroy(&managed_range->va_range, NULL);
    }

    if (vma->vm_private_data) {
        uvm_vma_wrapper_destroy(vma->vm_private_data);
        vma->vm_private_data = NULL;
    }
    UVM_ASSERT(size == vma->vm_end - vma->vm_start);
}

static void uvm_destroy_vma_semaphore_pool(struct vm_area_struct *vma)
{
    uvm_va_range_semaphore_pool_t *semaphore_pool_range;
    uvm_va_space_t *va_space;

    va_space = uvm_va_space_get(vma->vm_file);
    uvm_assert_rwsem_locked(&va_space->lock);
    semaphore_pool_range = uvm_va_range_semaphore_pool_find(va_space, vma->vm_start);
    UVM_ASSERT(semaphore_pool_range &&
               semaphore_pool_range->va_range.node.start   == vma->vm_start &&
               semaphore_pool_range->va_range.node.end + 1 == vma->vm_end);

    uvm_mem_unmap_cpu_user(semaphore_pool_range->mem);
}

// The kernel will also SIGBUS faults to vmas with valid ops but no fault
// handler, but it didn't always do that. Make it explicit so we don't rely on
// the kernel's implementation.
static vm_fault_t uvm_vm_fault_sigbus(struct vm_fault *vmf)
{
    UVM_DBG_PRINT_RL("Fault to address 0x%lx in disabled vma\n", vmf->address);
    return VM_FAULT_SIGBUS;
}

static vm_fault_t uvm_vm_fault_sigbus_entry(struct vm_fault *vmf)
{
    UVM_ENTRY_RET(uvm_vm_fault_sigbus(vmf));
}

static struct vm_operations_struct uvm_vm_ops_disabled =
{
    .fault = uvm_vm_fault_sigbus_entry,
};

static void uvm_disable_vma(struct vm_area_struct *vma)
{
    // In the case of fork, the kernel has already copied the old PTEs over to
    // the child process, so an access in the child might succeed instead of
    // causing a fault. To force a fault we'll unmap it directly here.
    //
    // Note that since the unmap works on file offset, not virtual address, this
    // unmaps both the old and new vmas.
    //
    // In the case of a move (mremap), the kernel will copy the PTEs over later,
    // so it doesn't matter if we unmap here. However, the new vma's open will
    // immediately be followed by a close on the old vma. We call
    // unmap_mapping_range for the close, which also unmaps the new vma because
    // they have the same file offset.
    unmap_mapping_range(vma->vm_file->f_mapping,
                        vma->vm_pgoff << PAGE_SHIFT,
                        vma->vm_end - vma->vm_start,
                        1);

    vma->vm_ops = &uvm_vm_ops_disabled;

    if (vma->vm_private_data) {
        uvm_vma_wrapper_destroy(vma->vm_private_data);
        vma->vm_private_data = NULL;
    }
}

// We can't return an error from uvm_vm_open so on failed splits
// we'll disable *both* vmas. This isn't great behavior for the
// user, but we don't have many options. We could leave the old VA
// range in place but that breaks the model of vmas always
// completely covering VA ranges. We'd have to be very careful
// handling later splits and closes of both that partially-covered
// VA range, and of the vmas which might or might not cover it any
// more.
//
// A failure likely means we're in OOM territory, so this should not
// be common by any means, and the process might die anyway.
static void uvm_vm_open_failure(struct vm_area_struct *original,
                                struct vm_area_struct *new)
{
    uvm_va_space_t *va_space = uvm_va_space_get(new->vm_file);
    static const bool make_zombie = false;

    UVM_ASSERT(va_space == uvm_va_space_get(original->vm_file));
    uvm_assert_rwsem_locked_write(&va_space->lock);

    uvm_destroy_vma_managed(original, make_zombie);
    uvm_disable_vma(original);
    uvm_disable_vma(new);
}

// vm_ops->open cases:
//
// 1) Parent vma is dup'd (fork)
//    This is undefined behavior in the UVM Programming Model. For convenience
//    the parent will continue operating properly, but the child is not
//    guaranteed access to the range.
//
// 2) Original vma is split (munmap, mprotect, mremap, mbind, etc)
//    The UVM Programming Model supports mbind always and supports mprotect if
//    HMM is present. Supporting either of those means all such splitting cases
//    must be handled. This involves splitting the va_range covering the split
//    location. Note that the kernel will never merge us back on two counts: we
//    set VM_MIXEDMAP and we have a ->close callback.
//
// 3) Original vma is moved (mremap)
//    This is undefined behavior in the UVM Programming Model. We'll get an open
//    on the new vma in which we disable operations on the new vma, then a close
//    on the old vma.
//
// Note that since we set VM_DONTEXPAND on the vma we're guaranteed that the vma
// will never increase in size, only shrink/split.
static void uvm_vm_open_managed(struct vm_area_struct *vma)
{
    uvm_va_space_t *va_space = uvm_va_space_get(vma->vm_file);
    uvm_va_range_managed_t *managed_range;
    struct vm_area_struct *original;
    NV_STATUS status;
    NvU64 new_end;

    // This is slightly ugly. We need to know the parent vma of this new one,
    // but we can't use the range tree to look up the original because that
    // doesn't handle a vma move operation.
    //
    // However, all of the old vma's fields have been copied into the new vma,
    // and open of the new vma is always called before close of the old (in
    // cases where close will be called immediately afterwards, like move).
    // vma->vm_private_data will thus still point to the original vma that we
    // set in mmap or open.
    //
    // Things to watch out for here:
    // - For splits, the old vma hasn't been adjusted yet so its vm_start and
    //   vm_end region will overlap with this vma's start and end.
    //
    // - For splits and moves, the new vma has not yet been inserted into the
    //   mm's list so vma->vm_prev and vma->vm_next cannot be used, nor will
    //   the new vma show up in find_vma and friends.
    original = ((uvm_vma_wrapper_t*)vma->vm_private_data)->vma;
    vma->vm_private_data = NULL;
    // On fork or move we want to simply disable the new vma
    if (vma->vm_mm != original->vm_mm ||
        (vma->vm_start != original->vm_start && vma->vm_end != original->vm_end)) {
        uvm_disable_vma(vma);
        return;
    }

    // At this point we are guaranteed that the mmap_lock is held in write
    // mode.
    uvm_record_lock_mmap_lock_write(current->mm);

    // Split vmas should always fall entirely within the old one, and be on one
    // side.
    UVM_ASSERT(vma->vm_start >= original->vm_start && vma->vm_end <= original->vm_end);
    UVM_ASSERT(vma->vm_start == original->vm_start || vma->vm_end == original->vm_end);

    // The vma is splitting, so create a new range under this vma if necessary.
    // The kernel handles splits in the middle of the vma by doing two separate
    // splits so we just have to handle one vma splitting in two here.
    if (vma->vm_start == original->vm_start)
        new_end = vma->vm_end - 1; // Left split (new_end is inclusive)
    else
        new_end = vma->vm_start - 1; // Right split (new_end is inclusive)

    uvm_va_space_down_write(va_space);

    vma->vm_private_data = uvm_vma_wrapper_alloc(vma);
    if (!vma->vm_private_data) {
        uvm_vm_open_failure(original, vma);
        goto out;
    }

    // There can be multiple ranges under the vma already. Check if one spans
    // the new split boundary. If so, split it.
    managed_range = uvm_va_range_managed_find(va_space, new_end);
    UVM_ASSERT(managed_range);
    UVM_ASSERT(uvm_va_range_vma_current(managed_range) == original);
    if (managed_range->va_range.node.end != new_end) {
        status = uvm_va_range_split(managed_range, new_end, NULL);
        if (status != NV_OK) {
            UVM_DBG_PRINT("Failed to split VA range, destroying both: %s. "
                          "original vma [0x%lx, 0x%lx) new vma [0x%lx, 0x%lx)\n",
                          nvstatusToString(status),
                          original->vm_start, original->vm_end,
                          vma->vm_start, vma->vm_end);
            uvm_vm_open_failure(original, vma);
            goto out;
        }
    }

    // Point managed_ranges to the new vma
    uvm_for_each_va_range_managed_in_vma(managed_range, vma) {
        UVM_ASSERT(uvm_va_range_vma_current(managed_range) == original);
        managed_range->vma_wrapper = vma->vm_private_data;
    }

out:
    uvm_va_space_up_write(va_space);
    uvm_record_unlock_mmap_lock_write(current->mm);
}

static void uvm_vm_open_managed_entry(struct vm_area_struct *vma)
{
   UVM_ENTRY_VOID(uvm_vm_open_managed(vma));
}

static void uvm_vm_close_managed(struct vm_area_struct *vma)
{
    uvm_va_space_t *va_space = uvm_va_space_get(vma->vm_file);
    bool make_zombie = false;
    NV_STATUS status;

    if (current->mm != NULL)
        uvm_record_lock_mmap_lock_write(current->mm);

    // current->mm will be NULL on process teardown, in which case we have
    // special handling.
    if (current->mm == NULL) {
        make_zombie = (va_space->initialization_flags & UVM_INIT_FLAGS_MULTI_PROCESS_SHARING_MODE);
        if (!make_zombie) {
            // If we're not in multi-process mode, then we want to stop all user
            // channels before unmapping the managed allocations to avoid
            // spurious MMU faults in the system log. If we have a va_space_mm
            // then this must've already happened as part of
            // uvm_va_space_mm_shutdown. Otherwise we need to handle it here.
            if (uvm_va_space_mm_enabled(va_space) && current->mm == va_space->va_space_mm.mm) {
                UVM_ASSERT(atomic_read(&va_space->user_channels_stopped));
            }
            else {
                // Stopping channels involves making RM calls, so we have to do
                // that with the VA space lock in read mode.
                uvm_va_space_down_read_rm(va_space);
                if (!atomic_read(&va_space->user_channels_stopped))
                    uvm_va_space_stop_all_user_channels(va_space);
                uvm_va_space_up_read_rm(va_space);
            }
        }
    }

    // See uvm_mmap for why we need this in addition to mmap_lock
    uvm_va_space_down_write(va_space);

    if (current->mm == NULL)
        status = uvm_moe_slot_pool_clear_range_mappings_for_release_locked(va_space,
                                                                           current->mm,
                                                                           vma->vm_start,
                                                                           vma->vm_end - vma->vm_start);
    else
        status = uvm_moe_slot_pool_clear_range_mappings_for_unmap_locked(va_space,
                                                                          current->mm,
                                                                          vma->vm_start,
                                                                          vma->vm_end - vma->vm_start);

    (void)status;

    uvm_destroy_vma_managed(vma, make_zombie);

    uvm_va_space_up_write(va_space);

    if (current->mm != NULL)
        uvm_record_unlock_mmap_lock_write(current->mm);
}

static void uvm_vm_close_managed_entry(struct vm_area_struct *vma)
{
    UVM_ENTRY_VOID(uvm_vm_close_managed(vma));
}

static vm_fault_t uvm_vm_fault(struct vm_fault *vmf)
{
    uvm_va_space_t *va_space = uvm_va_space_get(vmf->vma->vm_file);
    return uvm_va_space_cpu_fault_managed(va_space, vmf);
}

static vm_fault_t uvm_vm_fault_entry(struct vm_fault *vmf)
{
    UVM_ENTRY_RET(uvm_vm_fault(vmf));
}

static struct vm_operations_struct uvm_vm_ops_managed =
{
    .open         = uvm_vm_open_managed_entry,
    .close        = uvm_vm_close_managed_entry,
    .fault        = uvm_vm_fault_entry,
    .page_mkwrite = uvm_vm_fault_entry,
};

// vm operations on semaphore pool allocations only control CPU mappings. Unmapping GPUs,
// freeing the allocation, and destroying the range are handled by UVM_FREE.
static void uvm_vm_open_semaphore_pool(struct vm_area_struct *vma)
{
    struct vm_area_struct *origin_vma = (struct vm_area_struct *)vma->vm_private_data;
    uvm_va_space_t *va_space = uvm_va_space_get(origin_vma->vm_file);
    uvm_va_range_semaphore_pool_t *semaphore_pool_range;
    bool is_fork = (vma->vm_mm != origin_vma->vm_mm);
    NV_STATUS status;

    uvm_record_lock_mmap_lock_write(current->mm);

    uvm_va_space_down_write(va_space);

    semaphore_pool_range = uvm_va_range_semaphore_pool_find(va_space, origin_vma->vm_start);
    UVM_ASSERT(semaphore_pool_range);
    UVM_ASSERT_MSG(semaphore_pool_range &&
                   semaphore_pool_range->va_range.node.start == origin_vma->vm_start &&
                   semaphore_pool_range->va_range.node.end + 1 == origin_vma->vm_end,
                   "origin vma [0x%llx, 0x%llx); va_range [0x%llx, 0x%llx) type %d\n",
                   (NvU64)origin_vma->vm_start,
                   (NvU64)origin_vma->vm_end,
                   semaphore_pool_range->va_range.node.start,
                   semaphore_pool_range->va_range.node.end + 1,
                   semaphore_pool_range->va_range.type);

    // Semaphore pool vmas do not have vma wrappers, but some functions will
    // assume vm_private_data is a wrapper.
    vma->vm_private_data = NULL;
#if defined(VM_WIPEONFORK)
    nv_vm_flags_set(vma, VM_WIPEONFORK);
#endif

    if (is_fork) {
        // If we forked, leave the parent vma alone.
        uvm_disable_vma(vma);

        // uvm_disable_vma unmaps in the parent as well; clear the uvm_mem CPU
        // user mapping metadata and then remap.
        uvm_mem_unmap_cpu_user(semaphore_pool_range->mem);

        status = uvm_mem_map_cpu_user(semaphore_pool_range->mem, semaphore_pool_range->va_range.va_space, origin_vma);
        if (status != NV_OK) {
            UVM_DBG_PRINT("Failed to remap semaphore pool to CPU for parent after fork; status = %d (%s)",
                    status, nvstatusToString(status));
            origin_vma->vm_ops = &uvm_vm_ops_disabled;
        }
    }
    else {
        origin_vma->vm_private_data = NULL;
        origin_vma->vm_ops = &uvm_vm_ops_disabled;
        vma->vm_ops = &uvm_vm_ops_disabled;
        uvm_mem_unmap_cpu_user(semaphore_pool_range->mem);
    }

    uvm_va_space_up_write(va_space);

    uvm_record_unlock_mmap_lock_write(current->mm);
}

static void uvm_vm_open_semaphore_pool_entry(struct vm_area_struct *vma)
{
   UVM_ENTRY_VOID(uvm_vm_open_semaphore_pool(vma));
}

// vm operations on semaphore pool allocations only control CPU mappings. Unmapping GPUs,
// freeing the allocation, and destroying the va_range are handled by UVM_FREE.
static void uvm_vm_close_semaphore_pool(struct vm_area_struct *vma)
{
    uvm_va_space_t *va_space = uvm_va_space_get(vma->vm_file);

    if (current->mm != NULL)
        uvm_record_lock_mmap_lock_write(current->mm);

    uvm_va_space_down_read(va_space);

    uvm_destroy_vma_semaphore_pool(vma);

    uvm_va_space_up_read(va_space);

    if (current->mm != NULL)
        uvm_record_unlock_mmap_lock_write(current->mm);
}

static void uvm_vm_close_semaphore_pool_entry(struct vm_area_struct *vma)
{
   UVM_ENTRY_VOID(uvm_vm_close_semaphore_pool(vma));
}

static struct vm_operations_struct uvm_vm_ops_semaphore_pool =
{
    .open         = uvm_vm_open_semaphore_pool_entry,
    .close        = uvm_vm_close_semaphore_pool_entry,
    .fault        = uvm_vm_fault_sigbus_entry,
};

static void uvm_vm_open_device_p2p(struct vm_area_struct *vma)
{
    struct vm_area_struct *origin_vma = (struct vm_area_struct *)vma->vm_private_data;
    uvm_va_space_t *va_space = uvm_va_space_get(origin_vma->vm_file);
    uvm_va_range_t *va_range;
    bool is_fork = (vma->vm_mm != origin_vma->vm_mm);

    uvm_record_lock_mmap_lock_write(current->mm);

    uvm_va_space_down_write(va_space);

    va_range = uvm_va_range_find(va_space, origin_vma->vm_start);
    UVM_ASSERT(va_range);
    UVM_ASSERT_MSG(va_range->type == UVM_VA_RANGE_TYPE_DEVICE_P2P &&
                   va_range->node.start == origin_vma->vm_start &&
                   va_range->node.end + 1 == origin_vma->vm_end,
                   "origin vma [0x%llx, 0x%llx); va_range [0x%llx, 0x%llx) type %d\n",
                   (NvU64)origin_vma->vm_start, (NvU64)origin_vma->vm_end, va_range->node.start,
                   va_range->node.end + 1, va_range->type);

    // Device P2P vmas do not have vma wrappers, but some functions will
    // assume vm_private_data is a wrapper.
    vma->vm_private_data = NULL;
#if defined(VM_WIPEONFORK)
    nv_vm_flags_set(vma, VM_WIPEONFORK);
#endif

    if (is_fork) {
        // If we forked, leave the parent vma alone.
        uvm_disable_vma(vma);

        // uvm_disable_vma unmaps in the parent as well so remap the parent
        uvm_va_range_device_p2p_map_cpu(va_range->va_space, origin_vma, uvm_va_range_to_device_p2p(va_range));
    }
    else {
        // mremap will free the backing pages via unmap so we can't support it.
        origin_vma->vm_private_data = NULL;
        origin_vma->vm_ops = &uvm_vm_ops_disabled;
        vma->vm_ops = &uvm_vm_ops_disabled;
        unmap_mapping_range(va_space->mapping, va_range->node.start, uvm_va_range_size(va_range), 1);
    }

    uvm_va_space_up_write(va_space);

    uvm_record_unlock_mmap_lock_write(current->mm);
}

static void uvm_vm_open_device_p2p_entry(struct vm_area_struct *vma)
{
    UVM_ENTRY_VOID(uvm_vm_open_device_p2p(vma));
}

// Device P2P pages are only mapped on the CPU. Pages are allocated externally
// to UVM but destroying the range must unpin the RM object.
static void uvm_vm_close_device_p2p(struct vm_area_struct *vma)
{
}

static void uvm_vm_close_device_p2p_entry(struct vm_area_struct *vma)
{
    UVM_ENTRY_VOID(uvm_vm_close_device_p2p(vma));
}

static struct vm_operations_struct uvm_vm_ops_device_p2p =
{
    .open         = uvm_vm_open_device_p2p_entry,
    .close        = uvm_vm_close_device_p2p_entry,
    .fault        = uvm_vm_fault_sigbus_entry,
};

static bool va_range_type_expects_mmap(uvm_va_range_type_t type)
{
    switch (type) {
        case UVM_VA_RANGE_TYPE_SEMAPHORE_POOL:
        case UVM_VA_RANGE_TYPE_DEVICE_P2P:
            return true;

        // Although UVM_VA_RANGE_TYPE_MANAGED does support mmap, it doesn't
        // expect mmap to be called on a pre-existing range. mmap itself creates
        // the managed va range.

        default:
            return false;
    }
}

static int uvm_mmap(struct file *filp, struct vm_area_struct *vma)
{
    void *fd_type_ptr;
    uvm_va_space_t *va_space;
    NV_STATUS status = uvm_global_get_status();
    int ret = 0;
    bool vma_wrapper_allocated = false;

    if (status != NV_OK)
        return -nv_status_to_errno(status);

    switch (uvm_fd_type(filp, &fd_type_ptr)) {
        case UVM_FD_VA_SPACE:
            va_space = (uvm_va_space_t *)fd_type_ptr;
            break;

        case UVM_FD_TEST:
            return uvm_test_file_mmap((uvm_test_file_t *)fd_type_ptr, vma);

        default:
            return -EBADFD;
    }

    // When the VA space is associated with an mm, all vmas under the VA space
    // must come from that mm.
    if (uvm_va_space_mm_enabled(va_space)) {
        UVM_ASSERT(va_space->va_space_mm.mm);
        if (va_space->va_space_mm.mm != current->mm)
            return -EOPNOTSUPP;
    }

    // UVM mappings are required to set offset == VA. This simplifies things
    // since we don't have to worry about address aliasing (except for fork,
    // handled separately) and it makes unmap_mapping_range simpler.
    if (vma->vm_start != (vma->vm_pgoff << PAGE_SHIFT)) {
        UVM_DBG_PRINT_RL("vm_start 0x%lx != vm_pgoff 0x%lx\n", vma->vm_start, vma->vm_pgoff << PAGE_SHIFT);
        return -EINVAL;
    }

    // Enforce shared read/writable mappings so we get all fault callbacks
    // without the kernel doing COW behind our backs. The user can still call
    // mprotect to change protections, but that will only hurt user space.
    if ((vma->vm_flags & (VM_SHARED|VM_READ|VM_WRITE)) !=
                         (VM_SHARED|VM_READ|VM_WRITE)) {
        UVM_DBG_PRINT_RL("User requested non-shared or non-writable mapping\n");
        return -EINVAL;
    }

    // If the PM lock cannot be acquired, disable the VMA and report success
    // to the caller. The caller is expected to determine whether the
    // map operation succeeded via an ioctl() call. This is necessary to
    // safely handle MAP_FIXED, which needs to complete atomically to prevent
    // the loss of the virtual address range.
    if (!uvm_down_read_trylock(&g_uvm_global.pm.lock)) {
        uvm_disable_vma(vma);
        return 0;
    }

    uvm_record_lock_mmap_lock_write(current->mm);

    // VM_MIXEDMAP      Required to use vm_insert_page
    //
    // VM_DONTEXPAND    mremap can grow a vma in place without giving us any
    //                  callback. We need to prevent this so our ranges stay
    //                  up-to-date with the vma. This flag doesn't prevent
    //                  mremap from moving the mapping elsewhere, nor from
    //                  shrinking it. We can detect both of those cases however
    //                  with vm_ops->open() and vm_ops->close() callbacks.
    //
    // Using VM_DONTCOPY would be nice, but madvise(MADV_DOFORK) can reset that
    // so we have to handle vm_open on fork anyway. We could disable MADV_DOFORK
    // with VM_IO, but that causes other mapping issues.
    // Make the default behavior be VM_DONTCOPY to avoid the performance impact
    // of removing CPU mappings in the parent on fork()+exec(). Users can call
    // madvise(MDV_DOFORK) if the child process requires access to the
    // allocation.
    nv_vm_flags_set(vma, VM_MIXEDMAP | VM_DONTEXPAND | VM_DONTCOPY);

    vma->vm_ops = &uvm_vm_ops_managed;

    // This identity assignment is needed so uvm_vm_open can find its parent vma
    vma->vm_private_data = uvm_vma_wrapper_alloc(vma);
    if (!vma->vm_private_data) {
        ret = -ENOMEM;
        goto out;
    }
    vma_wrapper_allocated = true;

    // The kernel has taken mmap_lock in write mode, but that doesn't prevent
    // this va_space from being modified by the GPU fault path or from the ioctl
    // path where we don't have this mm for sure, so we have to lock the VA
    // space directly.
    uvm_va_space_down_write(va_space);

    // uvm_va_range_create_mmap will catch collisions. Below are some example
    // cases which can cause collisions. There may be others.
    // 1) An overlapping range was previously created with an ioctl, for example
    //    for an external mapping.
    // 2) This file was passed to another process via a UNIX domain socket
    status = uvm_va_range_create_mmap(va_space, current->mm, vma->vm_private_data, NULL);

    if (status == NV_ERR_UVM_ADDRESS_IN_USE) {
        uvm_va_range_t *existing_range = uvm_va_range_find(va_space, vma->vm_start);

        // Does the existing range exactly match the vma and expects mmap?
        if (existing_range &&
            existing_range->node.start == vma->vm_start &&
            existing_range->node.end + 1 == vma->vm_end &&
            va_range_type_expects_mmap(existing_range->type)) {

            // We speculatively initialized the managed vma before checking for
            // collisions because we expect successful insertion to be the
            // common case. Undo that.
            uvm_vma_wrapper_destroy(vma->vm_private_data);
            vma_wrapper_allocated = false;
            vma->vm_private_data = vma;

            switch (existing_range->type) {
                case UVM_VA_RANGE_TYPE_SEMAPHORE_POOL:
                    vma->vm_ops = &uvm_vm_ops_semaphore_pool;
                    status = uvm_mem_map_cpu_user(uvm_va_range_to_semaphore_pool(existing_range)->mem, va_space, vma);
                    break;

                case UVM_VA_RANGE_TYPE_DEVICE_P2P:
                    vma->vm_ops = &uvm_vm_ops_device_p2p;
                    status = uvm_va_range_device_p2p_map_cpu(va_space,
                                                             vma,
                                                             uvm_va_range_to_device_p2p(existing_range));
                    break;

                default:
                    UVM_ASSERT(0);
                    break;
            }
        }
    }

    if (status != NV_OK) {
        UVM_DBG_PRINT_RL("Failed to create or map VA range for vma [0x%lx, 0x%lx): %s\n",
                         vma->vm_start, vma->vm_end, nvstatusToString(status));
        ret = -nv_status_to_errno(status);
    }

    uvm_va_space_up_write(va_space);

out:
    if (ret != 0 && vma_wrapper_allocated)
        uvm_vma_wrapper_destroy(vma->vm_private_data);

    uvm_record_unlock_mmap_lock_write(current->mm);

    uvm_up_read(&g_uvm_global.pm.lock);

    return ret;
}

bool uvm_vma_is_managed(struct vm_area_struct *vma)
{
    return vma->vm_ops == &uvm_vm_ops_disabled ||
           vma->vm_ops == &uvm_vm_ops_managed ||
           vma->vm_ops == &uvm_vm_ops_semaphore_pool;
}

static int uvm_mmap_entry(struct file *filp, struct vm_area_struct *vma)
{
   UVM_ENTRY_RET(uvm_mmap(filp, vma));
}

static NV_STATUS uvm_api_initialize(UVM_INITIALIZE_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space;
    NV_STATUS status;
    uvm_fd_type_t old_fd_type;

    // Normally we expect private_data == UVM_FD_UNINITIALIZED. However multiple
    // threads may call this ioctl concurrently so we have to be careful to
    // avoid initializing multiple va_spaces and/or leaking memory. To do this
    // we do an atomic compare and swap. Only one thread will observe
    // UVM_FD_UNINITIALIZED and that thread will allocate and setup the
    // va_space.
    //
    // Other threads will either see UVM_FD_INITIALIZING or UVM_FD_VA_SPACE. In
    // the case of UVM_FD_VA_SPACE we return success if and only if the
    // initialization flags match. If another thread is still initializing the
    // va_space we return NV_ERR_BUSY_RETRY.
    //
    // If va_space initialization fails we return the failure code and reset the
    // FD state back to UVM_FD_UNINITIALIZED to allow another initialization
    // attempt to be made. This is safe because other threads will have only had
    // a chance to observe UVM_FD_INITIALIZING and not UVM_FD_VA_SPACE in this
    // case.
    old_fd_type = uvm_fd_type_init_cas(filp);
    switch (old_fd_type) {
        case UVM_FD_UNINITIALIZED:
            status = uvm_va_space_create(filp->f_mapping, &va_space, params->flags);
            if (status != NV_OK) {
                uvm_fd_type_set(filp, UVM_FD_UNINITIALIZED, NULL);
                return status;
            }

            uvm_fd_type_set(filp, UVM_FD_VA_SPACE, va_space);
            break;

        case UVM_FD_VA_SPACE:
            va_space = uvm_va_space_get(filp);
            if (params->flags != va_space->initialization_flags)
                status = NV_ERR_INVALID_ARGUMENT;
            else
                status = NV_OK;

            break;

        case UVM_FD_MM:
        case UVM_FD_TEST:
            status = NV_ERR_INVALID_ARGUMENT;
            break;

        case UVM_FD_INITIALIZING:
            status = NV_ERR_BUSY_RETRY;
            break;

        default:
            UVM_ASSERT(0);
            status = NV_ERR_INVALID_STATE; // Quiet compiler warnings
            break;
    }

    return status;
}

static NV_STATUS uvm_api_pageable_mem_access(UVM_PAGEABLE_MEM_ACCESS_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    params->pageableMemAccess = uvm_va_space_pageable_mem_access_supported(va_space) ? NV_TRUE : NV_FALSE;
    return NV_OK;
}

static NvU64 uvm_phase_f_page_overlap_bytes(NvU64 page_start, NvU64 query_start, NvU64 query_end)
{
    NvU64 page_end = page_start + PAGE_SIZE;
    NvU64 overlap_start = max(page_start, query_start);
    NvU64 overlap_end = min(page_end, query_end);

    if (overlap_start >= overlap_end)
        return 0;

    return overlap_end - overlap_start;
}

static void uvm_phase_f_add_resident_bytes_from_block(uvm_va_block_t *block,
                                                      uvm_gpu_t *gpu,
                                                      NvU64 query_start,
                                                      NvU64 query_end,
                                                      UVM_PHASE_F_RANGE_RESULT *result)
{
    uvm_page_mask_t *gpu_resident = uvm_va_block_resident_mask_get(block, gpu->id, NUMA_NO_NODE);
    uvm_page_mask_t *cpu_resident = uvm_va_block_resident_mask_get(block, UVM_ID_CPU, NUMA_NO_NODE);
    NvU64 first_addr;
    NvU64 last_addr;
    uvm_page_index_t first_page_index;
    uvm_page_index_t last_page_index;
    uvm_page_index_t page_index;

    if (query_start >= query_end)
        return;

    first_addr = UVM_ALIGN_DOWN(max(query_start, block->start), PAGE_SIZE);
    last_addr = UVM_ALIGN_DOWN(min(query_end - 1, block->end), PAGE_SIZE);
    first_page_index = uvm_va_block_cpu_page_index(block, first_addr);
    last_page_index = uvm_va_block_cpu_page_index(block, last_addr);

    for (page_index = first_page_index; page_index <= last_page_index; ++page_index) {
        NvU64 page_start = block->start + (NvU64)page_index * PAGE_SIZE;
        NvU64 bytes = uvm_phase_f_page_overlap_bytes(page_start, query_start, query_end);
        bool on_gpu = gpu_resident && uvm_page_mask_test(gpu_resident, page_index);
        bool on_cpu = cpu_resident && uvm_page_mask_test(cpu_resident, page_index);

        if (on_gpu)
            result->resident_gpu_bytes += bytes;
        if (on_cpu)
            result->resident_cpu_bytes += bytes;
        if (on_gpu || on_cpu)
            result->resident_any_bytes += bytes;
        else
            result->resident_unknown_bytes += bytes;
    }
}

static NV_STATUS uvm_phase_f_query_resident_bytes(const NvProcessorUuid *gpu_uuid,
                                                  NvBool async,
                                                  NvU64 base,
                                                  NvU64 length,
                                                  UVM_PHASE_F_RANGE_RESULT *result,
                                                  uvm_va_space_t *va_space)
{
    NV_STATUS status = NV_OK;
    uvm_va_range_managed_t *managed_range = NULL;
    uvm_va_range_managed_t *managed_range_last = NULL;
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    bool retained_mm = false;
    NvU64 query_start;
    NvU64 query_end;
    NvU64 last_address;

    result->queried_bytes = length;
    result->resident_gpu_bytes = 0;
    result->resident_cpu_bytes = 0;
    result->resident_any_bytes = 0;
    result->resident_unknown_bytes = 0;

    if (uvm_api_range_invalid(base, length))
        return NV_ERR_INVALID_ARGUMENT;

    query_start = base;
    query_end = base + length;
    last_address = query_end - 1;

    if (uvm_va_space_mm_enabled(va_space)) {
        mm = uvm_va_space_mm_retain(va_space);
        retained_mm = mm != NULL;
    }
    else if (!(current->flags & PF_KTHREAD)) {
        mm = current->mm;
    }

    if (mm)
        uvm_down_read_mmap_lock(mm);

    uvm_va_space_down_read(va_space);

    gpu = uvm_va_space_get_gpu_by_uuid(va_space, gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto out;
    }

    if (uvm_api_range_type_check(va_space, mm, base, length) != UVM_API_RANGE_TYPE_MANAGED) {
        status = NV_ERR_INVALID_ADDRESS;
        goto out;
    }

    uvm_for_each_va_range_managed_in_contig(managed_range, va_space, base, last_address) {
        const NvU64 range_start = max(query_start, managed_range->va_range.node.start);
        const NvU64 range_end = min(query_end - 1, managed_range->va_range.node.end);
        const size_t first_block_index = uvm_va_range_block_index(managed_range, range_start);
        const size_t last_block_index = uvm_va_range_block_index(managed_range, range_end);
        size_t block_index;

        managed_range_last = managed_range;

        for (block_index = first_block_index; block_index <= last_block_index; ++block_index) {
            uvm_va_block_t *block = uvm_va_range_block(managed_range, block_index);
            const NvU64 block_start = block ? max(query_start, block->start) : 0;
            const NvU64 block_end = block ? min(query_end, block->end + 1) : 0;

            if (!block) {
                NvU64 managed_aligned_start = UVM_ALIGN_DOWN(managed_range->va_range.node.start, UVM_VA_BLOCK_SIZE);
                NvU64 missing_block_start = max(query_start,
                                                max(managed_range->va_range.node.start,
                                                    managed_aligned_start + (NvU64)block_index * UVM_VA_BLOCK_SIZE));
                NvU64 missing_block_end = min(query_end,
                                              min(managed_range->va_range.node.end + 1,
                                                  managed_aligned_start +
                                                      (NvU64)(block_index + 1) * UVM_VA_BLOCK_SIZE));
                if (missing_block_start < missing_block_end)
                    result->resident_unknown_bytes += missing_block_end - missing_block_start;
                continue;
            }

            uvm_mutex_lock(&block->lock);
            if (!async)
                status = uvm_tracker_wait(&block->tracker);
            if (status == NV_OK)
                uvm_phase_f_add_resident_bytes_from_block(block, gpu, block_start, block_end, result);
            uvm_mutex_unlock(&block->lock);

            if (status != NV_OK)
                goto out;
        }
    }

    if (!managed_range_last || managed_range_last->va_range.node.end < last_address)
        status = NV_ERR_INVALID_ADDRESS;

out:
    uvm_va_space_up_read(va_space);
    if (mm)
        uvm_up_read_mmap_lock(mm);
    if (retained_mm)
        uvm_va_space_mm_release(va_space);

    uvm_phase_f_route_log("phase_f_resident_bytes base=0x%llx length=%llu status=0x%x queried=%llu gpu=%llu cpu=%llu any=%llu unknown=%llu\n",
                          base,
                          length,
                          status,
                          result->queried_bytes,
                          result->resident_gpu_bytes,
                          result->resident_cpu_bytes,
                          result->resident_any_bytes,
                          result->resident_unknown_bytes);
    return status;
}

static NV_STATUS uvm_phase_f_prepare_remote_read(const NvProcessorUuid *gpu_uuid,
                                                 NvBool async,
                                                 NvU64 base,
                                                 NvU64 length,
                                                 bool enqueue_bg,
                                                 bool schedule_bg,
                                                 bool audit_final_mapping,
                                                 UVM_PHASE_F_RANGE_RESULT *result,
                                                 uvm_va_space_t *va_space)
{
    NV_STATUS status = NV_OK;
    uvm_va_range_managed_t *managed_range = NULL;
    uvm_va_range_managed_t *managed_range_last = NULL;
    uvm_va_block_context_t *block_context;
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    NvU64 query_start;
    NvU64 query_end;
    NvU64 last_address;
    NvU64 covered_bytes = 0;
    NvU64 candidate_pages = 0;
    NvU64 mapped_pages = 0;
    NvU64 final_cpu_pages = 0;
    NvU64 final_remote_pages = 0;
    NvU64 bg_enqueued_pages = 0;

    result->queried_bytes = length;
    result->resident_gpu_bytes = 0;
    result->resident_cpu_bytes = 0;
    result->resident_any_bytes = 0;
    result->resident_unknown_bytes = 0;

    if (uvm_api_range_invalid(base, length))
        return NV_ERR_INVALID_ARGUMENT;

    query_start = base;
    query_end = base + length;
    last_address = query_end - 1;

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_read(va_space);

    gpu = uvm_va_space_get_gpu_by_uuid(va_space, gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto out;
    }

    if (uvm_api_range_type_check(va_space, mm, base, length) != UVM_API_RANGE_TYPE_MANAGED) {
        status = NV_ERR_INVALID_ADDRESS;
        goto out;
    }

    block_context = uvm_va_space_block_context(va_space, mm);

    uvm_for_each_va_range_managed_in_contig(managed_range, va_space, base, last_address) {
        const NvU64 range_start = max(query_start, managed_range->va_range.node.start);
        const NvU64 range_end = min(last_address, managed_range->va_range.node.end);
        const size_t first_block_index = uvm_va_range_block_index(managed_range, range_start);
        const size_t last_block_index = uvm_va_range_block_index(managed_range, range_end);
        size_t block_index;

        managed_range_last = managed_range;

        for (block_index = first_block_index; block_index <= last_block_index; ++block_index) {
            uvm_va_block_t *block;
            NvU64 block_start;
            NvU64 block_end;
            uvm_va_block_region_t block_region;
            NvU32 block_mapped_pages = 0;
            NvU32 block_bg_enqueued_pages = 0;

            status = uvm_va_range_block_create(managed_range, block_index, &block);
            if (status != NV_OK)
                goto out;

            block_start = max(query_start, block->start);
            block_end = min(last_address, block->end);
            if (block_start > block_end)
                continue;

            block_region = uvm_va_block_region_from_start_end(block, block_start, block_end);
            covered_bytes += block_end - block_start + 1;
            candidate_pages += uvm_va_block_region_size(block_region) / PAGE_SIZE;

            status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                             NULL,
                                             uvm_va_block_premap_readmostly_remote_read(block,
                                                                                        block_context,
                                                                                        gpu->id,
                                                                                        block_region,
                                                                                        !async,
                                                                                        false,
                                                                                        &block_mapped_pages));
            if (status != NV_OK)
                goto out;

            mapped_pages += block_mapped_pages;

            if (audit_final_mapping) {
                uvm_va_block_gpu_state_t *gpu_state;
                uvm_page_index_t page_index;
                NvU32 block_cpu_pages = 0;
                NvU32 block_remote_pages = 0;

                uvm_mutex_lock(&block->lock);
                gpu_state = uvm_va_block_gpu_state_get(block, gpu->id);
                for_each_va_block_page_in_region(page_index, block_region) {
                    bool cpu_resident = uvm_page_mask_test(&block->cpu.resident, page_index);
                    bool gpu_resident = gpu_state &&
                                        uvm_page_mask_test(&gpu_state->resident, page_index);
                    bool gpu_read_mapped = gpu_state &&
                                           uvm_page_mask_test(&gpu_state->pte_bits[UVM_PTE_BITS_GPU_READ],
                                                              page_index);

                    if (cpu_resident)
                        ++block_cpu_pages;
                    if (cpu_resident && !gpu_resident && gpu_read_mapped)
                        ++block_remote_pages;
                }
                uvm_mutex_unlock(&block->lock);

                final_cpu_pages += block_cpu_pages;
                final_remote_pages += block_remote_pages;
                if (block_cpu_pages != uvm_va_block_region_num_pages(block_region) ||
                    block_remote_pages != uvm_va_block_region_num_pages(block_region)) {
                    status = NV_ERR_INVALID_STATE;
                    goto out;
                }
            }

            if (enqueue_bg) {
                status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                                 NULL,
                                                 schedule_bg ?
                                                    uvm_va_block_enqueue_readmostly_bg_promote(block,
                                                                                               block_context,
                                                                                               gpu->id,
                                                                                               block_region,
                                                                                               &block_bg_enqueued_pages) :
                                                    uvm_va_block_enqueue_readmostly_bg_promote_deferred(block,
                                                                                                        block_context,
                                                                                                        gpu->id,
                                                                                                        block_region,
                                                                                                        &block_bg_enqueued_pages));
                if (status != NV_OK)
                    goto out;

                bg_enqueued_pages += block_bg_enqueued_pages;
            }
        }
    }

    if (!managed_range_last || managed_range_last->va_range.node.end < last_address)
        status = NV_ERR_INVALID_ADDRESS;

out:
    uvm_va_space_up_read(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    result->resident_gpu_bytes = (audit_final_mapping ? final_remote_pages : mapped_pages) * PAGE_SIZE;
    result->resident_cpu_bytes = (audit_final_mapping ? final_cpu_pages : candidate_pages) * PAGE_SIZE;
    result->resident_any_bytes = covered_bytes;
    result->resident_unknown_bytes = length > covered_bytes ? length - covered_bytes : 0;

    uvm_phase_f_route_log("phase_f_prepare_remote_read base=0x%llx length=%llu status=0x%x gpu=%s async=%u enqueue_bg=%u schedule_bg=%u audit_final=%u covered=%llu candidate_pages=%llu mapped_pages=%llu final_cpu_pages=%llu final_remote_pages=%llu bg_enqueued_pages=%llu\n",
                          base,
                          length,
                          status,
                          gpu ? gpu->name : "none",
                          async,
                          enqueue_bg,
                          schedule_bg,
                          audit_final_mapping,
                          covered_bytes,
                          candidate_pages,
                          mapped_pages,
                          final_cpu_pages,
                          final_remote_pages,
                          bg_enqueued_pages);

    return status;
}

static NV_STATUS uvm_phase_f_force_remote_read_64k(const NvProcessorUuid *gpu_uuid,
                                                   NvBool async,
                                                   NvU64 base,
                                                   NvU64 length,
                                                   UVM_PHASE_F_RANGE_RESULT *result,
                                                   uvm_va_space_t *va_space)
{
    NV_STATUS status = NV_OK;
    uvm_va_range_managed_t *managed_range = NULL;
    uvm_va_range_managed_t *managed_range_last = NULL;
    uvm_va_block_context_t *block_context;
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    NvU64 last_address;
    NvU64 split_bytes = 0;
    NvU64 already_64k_bytes = 0;
    NvU64 validated_bytes = 0;

    result->queried_bytes = length;
    result->resident_gpu_bytes = 0;
    result->resident_cpu_bytes = 0;
    result->resident_any_bytes = 0;
    result->resident_unknown_bytes = length;

    // The action intentionally accepts only complete VA blocks. This keeps the
    // resulting hierarchy uniform and lets strict slot-overwrite modes reject
    // every accidental partial, 4 KiB, or mixed setup.
    if (uvm_api_range_invalid(base, length) ||
        !IS_ALIGNED(base, UVM_PAGE_SIZE_2M) ||
        !IS_ALIGNED(length, UVM_PAGE_SIZE_2M)) {
        return NV_ERR_INVALID_ARGUMENT;
    }

    last_address = base + length - 1;
    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_read(va_space);

    gpu = uvm_va_space_get_gpu_by_uuid(va_space, gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto out;
    }

    if (uvm_api_range_type_check(va_space, mm, base, length) != UVM_API_RANGE_TYPE_MANAGED) {
        status = NV_ERR_INVALID_ADDRESS;
        goto out;
    }

    block_context = uvm_va_space_block_context(va_space, mm);

    uvm_for_each_va_range_managed_in_contig(managed_range, va_space, base, last_address) {
        const NvU64 range_start = max(base, managed_range->va_range.node.start);
        const NvU64 range_end = min(last_address, managed_range->va_range.node.end);
        const size_t first_block_index = uvm_va_range_block_index(managed_range, range_start);
        const size_t last_block_index = uvm_va_range_block_index(managed_range, range_end);
        size_t block_index;

        managed_range_last = managed_range;

        for (block_index = first_block_index; block_index <= last_block_index; ++block_index) {
            uvm_va_block_t *block;
            uvm_va_block_region_t block_region;
            bool split = false;
            bool already_64k = false;

            status = uvm_va_range_block_create(managed_range, block_index, &block);
            if (status != NV_OK)
                goto out;

            if (uvm_va_block_size(block) != UVM_PAGE_SIZE_2M ||
                !IS_ALIGNED(block->start, UVM_PAGE_SIZE_2M) ||
                block->start < base ||
                block->end > last_address) {
                status = NV_ERR_INVALID_ARGUMENT;
                goto out;
            }

            block_region = uvm_va_block_region_from_block(block);
            status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                             NULL,
                                             uvm_va_block_force_cpu_remote_read_64k(block,
                                                                                   block_context,
                                                                                   gpu->id,
                                                                                   block_region,
                                                                                   &split,
                                                                                   &already_64k));
            if (status != NV_OK)
                goto out;

            if (split == already_64k) {
                status = NV_ERR_INVALID_STATE;
                goto out;
            }

            if (split)
                split_bytes += UVM_PAGE_SIZE_2M;
            else
                already_64k_bytes += UVM_PAGE_SIZE_2M;

            validated_bytes += UVM_PAGE_SIZE_2M;
        }
    }

    if (!managed_range_last ||
        managed_range_last->va_range.node.end < last_address ||
        validated_bytes != length) {
        status = NV_ERR_INVALID_ADDRESS;
    }

out:
    uvm_va_space_up_read(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    result->resident_gpu_bytes = split_bytes;
    result->resident_cpu_bytes = already_64k_bytes;
    result->resident_any_bytes = validated_bytes;
    result->resident_unknown_bytes = length > validated_bytes ? length - validated_bytes : 0;

    uvm_phase_f_route_log("phase_f_force_remote_read_64k base=0x%llx length=%llu status=0x%x gpu=%s async_ignored=%u split_bytes=%llu already_64k_bytes=%llu validated_bytes=%llu unknown_bytes=%llu\n",
                          base,
                          length,
                          status,
                          gpu ? gpu->name : "none",
                          async,
                          split_bytes,
                          already_64k_bytes,
                          validated_bytes,
                          result->resident_unknown_bytes);

    return status;
}

static NV_STATUS uvm_phase_f_start_bg_promotion(const NvProcessorUuid *gpu_uuid,
                                                NvBool async,
                                                NvU64 base,
                                                NvU64 length,
                                                UVM_PHASE_F_RANGE_RESULT *result,
                                                uvm_va_space_t *va_space)
{
    NV_STATUS status = NV_OK;
    uvm_va_range_managed_t *managed_range = NULL;
    uvm_va_range_managed_t *managed_range_last = NULL;
    uvm_va_block_context_t *block_context;
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    NvU64 query_start;
    NvU64 query_end;
    NvU64 last_address;
    NvU64 covered_bytes = 0;
    NvU64 candidate_pages = 0;
    NvU64 started_pages = 0;

    result->queried_bytes = length;
    result->resident_gpu_bytes = 0;
    result->resident_cpu_bytes = 0;
    result->resident_any_bytes = 0;
    result->resident_unknown_bytes = 0;

    if (uvm_api_range_invalid(base, length))
        return NV_ERR_INVALID_ARGUMENT;

    query_start = base;
    query_end = base + length;
    last_address = query_end - 1;

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_read(va_space);

    gpu = uvm_va_space_get_gpu_by_uuid(va_space, gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto out;
    }

    if (uvm_api_range_type_check(va_space, mm, base, length) != UVM_API_RANGE_TYPE_MANAGED) {
        status = NV_ERR_INVALID_ADDRESS;
        goto out;
    }

    block_context = uvm_va_space_block_context(va_space, mm);

    uvm_for_each_va_range_managed_in_contig(managed_range, va_space, base, last_address) {
        const NvU64 range_start = max(query_start, managed_range->va_range.node.start);
        const NvU64 range_end = min(last_address, managed_range->va_range.node.end);
        const size_t first_block_index = uvm_va_range_block_index(managed_range, range_start);
        const size_t last_block_index = uvm_va_range_block_index(managed_range, range_end);
        size_t block_index;

        managed_range_last = managed_range;

        for (block_index = first_block_index; block_index <= last_block_index; ++block_index) {
            uvm_va_block_t *block;
            NvU64 block_start;
            NvU64 block_end;
            uvm_va_block_region_t block_region;
            NvU32 block_started_pages = 0;

            status = uvm_va_range_block_create(managed_range, block_index, &block);
            if (status != NV_OK)
                goto out;

            block_start = max(query_start, block->start);
            block_end = min(last_address, block->end);
            if (block_start > block_end)
                continue;

            block_region = uvm_va_block_region_from_start_end(block, block_start, block_end);
            covered_bytes += block_end - block_start + 1;
            candidate_pages += uvm_va_block_region_size(block_region) / PAGE_SIZE;

            status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                             NULL,
                                             uvm_va_block_start_readmostly_bg_promote(block,
                                                                                      block_context,
                                                                                      gpu->id,
                                                                                      block_region,
                                                                                      &block_started_pages));
            if (status != NV_OK)
                goto out;

            started_pages += block_started_pages;
        }
    }

    if (!managed_range_last || managed_range_last->va_range.node.end < last_address)
        status = NV_ERR_INVALID_ADDRESS;

out:
    uvm_va_space_up_read(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    result->resident_gpu_bytes = started_pages * PAGE_SIZE;
    result->resident_cpu_bytes = candidate_pages * PAGE_SIZE;
    result->resident_any_bytes = covered_bytes;
    result->resident_unknown_bytes = length > covered_bytes ? length - covered_bytes : 0;

    uvm_phase_f_route_log("phase_f_start_bg_promotion base=0x%llx length=%llu status=0x%x gpu=%s async=%u covered=%llu candidate_pages=%llu started_pages=%llu\n",
                          base,
                          length,
                          status,
                          gpu ? gpu->name : "none",
                          async,
                          covered_bytes,
                          candidate_pages,
                          started_pages);

    return status;
}

static NV_STATUS uvm_phase_f_set_bg_promotion_paused(NvU64 base,
                                                     NvU64 length,
                                                     bool pause,
                                                     UVM_PHASE_F_RANGE_RESULT *result,
                                                     uvm_va_space_t *va_space)
{
    NV_STATUS status = NV_OK;
    uvm_va_range_managed_t *managed_range = NULL;
    uvm_va_range_managed_t *managed_range_last = NULL;
    struct mm_struct *mm = NULL;
    NvU64 query_start;
    NvU64 query_end;
    NvU64 last_address;
    NvU64 covered_bytes = 0;
    NvU64 candidate_blocks = 0;
    NvU64 touched_blocks = 0;

    result->queried_bytes = length;
    result->resident_gpu_bytes = 0;
    result->resident_cpu_bytes = 0;
    result->resident_any_bytes = 0;
    result->resident_unknown_bytes = 0;

    if (uvm_api_range_invalid(base, length))
        return NV_ERR_INVALID_ARGUMENT;

    query_start = base;
    query_end = base + length;
    last_address = query_end - 1;

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_read(va_space);

    if (uvm_api_range_type_check(va_space, mm, base, length) != UVM_API_RANGE_TYPE_MANAGED) {
        status = NV_ERR_INVALID_ADDRESS;
        goto out;
    }

    uvm_for_each_va_range_managed_in_contig(managed_range, va_space, base, last_address) {
        const NvU64 range_start = max(query_start, managed_range->va_range.node.start);
        const NvU64 range_end = min(last_address, managed_range->va_range.node.end);
        const size_t first_block_index = uvm_va_range_block_index(managed_range, range_start);
        const size_t last_block_index = uvm_va_range_block_index(managed_range, range_end);
        size_t block_index;

        managed_range_last = managed_range;

        for (block_index = first_block_index; block_index <= last_block_index; ++block_index) {
            uvm_va_block_t *block;
            NvU64 block_start;
            NvU64 block_end;
            NvU32 block_touched = 0;

            status = uvm_va_range_block_create(managed_range, block_index, &block);
            if (status != NV_OK)
                goto out;

            block_start = max(query_start, block->start);
            block_end = min(last_address, block->end);
            if (block_start > block_end)
                continue;

            covered_bytes += block_end - block_start + 1;
            candidate_blocks++;

            status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                             NULL,
                                             uvm_va_block_set_bg_promote_paused(block,
                                                                                pause,
                                                                                &block_touched));
            if (status != NV_OK)
                goto out;

            touched_blocks += block_touched;
        }
    }

    if (!managed_range_last || managed_range_last->va_range.node.end < last_address)
        status = NV_ERR_INVALID_ADDRESS;

out:
    uvm_va_space_up_read(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    result->resident_gpu_bytes = touched_blocks;
    result->resident_cpu_bytes = candidate_blocks;
    result->resident_any_bytes = covered_bytes;
    result->resident_unknown_bytes = length > covered_bytes ? length - covered_bytes : 0;

    if (pause)
        uvm_phase_f_route_log("phase_f_pause_bg_promotion base=0x%llx length=%llu status=0x%x covered=%llu blocks=%llu touched=%llu\n",
                              base,
                              length,
                              status,
                              covered_bytes,
                              candidate_blocks,
                              touched_blocks);
    else
        uvm_phase_f_route_log("phase_f_resume_bg_promotion base=0x%llx length=%llu status=0x%x covered=%llu blocks=%llu touched=%llu\n",
                              base,
                              length,
                              status,
                              covered_bytes,
                              candidate_blocks,
                              touched_blocks);

    return status;
}

static NV_STATUS uvm_phase_f_gpu_only_soft(const NvProcessorUuid *gpu_uuid,
                                           NvBool async,
                                           NvU64 base,
                                           NvU64 length,
                                           bool read_only,
                                           bool clear,
                                           bool preserve_cpu_backing,
                                           UVM_PHASE_F_RANGE_RESULT *result,
                                           uvm_va_space_t *va_space)
{
    NV_STATUS status = NV_OK;
    uvm_va_range_managed_t *managed_range = NULL;
    uvm_va_range_managed_t *managed_range_last = NULL;
    uvm_va_block_context_t *block_context;
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    NvU64 query_start;
    NvU64 query_end;
    NvU64 last_address;
    NvU64 covered_bytes = 0;
    NvU64 candidate_pages = 0;
    NvU64 applied_pages = 0;
    NvU64 cpu_unmapped_pages = 0;
    NvU64 gpu_mapped_pages = 0;
    NvU64 cpu_preserved_pages = 0;
    NvU64 cleared_pages = 0;

    result->queried_bytes = length;
    result->resident_gpu_bytes = 0;
    result->resident_cpu_bytes = 0;
    result->resident_any_bytes = 0;
    result->resident_unknown_bytes = 0;

    if (uvm_api_range_invalid(base, length))
        return NV_ERR_INVALID_ARGUMENT;

    query_start = base;
    query_end = base + length;
    last_address = query_end - 1;

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_read(va_space);

    if (!clear) {
        gpu = uvm_va_space_get_gpu_by_uuid(va_space, gpu_uuid);
        if (!gpu) {
            status = NV_ERR_INVALID_DEVICE;
            goto out;
        }
    }

    if (uvm_api_range_type_check(va_space, mm, base, length) != UVM_API_RANGE_TYPE_MANAGED) {
        status = NV_ERR_INVALID_ADDRESS;
        goto out;
    }

    block_context = uvm_va_space_block_context(va_space, mm);

    uvm_for_each_va_range_managed_in_contig(managed_range, va_space, base, last_address) {
        const NvU64 range_start = max(query_start, managed_range->va_range.node.start);
        const NvU64 range_end = min(last_address, managed_range->va_range.node.end);
        const size_t first_block_index = uvm_va_range_block_index(managed_range, range_start);
        const size_t last_block_index = uvm_va_range_block_index(managed_range, range_end);
        size_t block_index;

        managed_range_last = managed_range;

        for (block_index = first_block_index; block_index <= last_block_index; ++block_index) {
            uvm_va_block_t *block;
            NvU64 block_start;
            NvU64 block_end;
            uvm_va_block_region_t block_region;
            NvU32 block_applied_pages = 0;
            NvU32 block_cpu_unmapped_pages = 0;
            NvU32 block_gpu_mapped_pages = 0;
            NvU32 block_cpu_preserved_pages = 0;
            NvU32 block_cleared_pages = 0;

            status = uvm_va_range_block_create(managed_range, block_index, &block);
            if (status != NV_OK)
                goto out;

            block_start = max(query_start, block->start);
            block_end = min(last_address, block->end);
            if (block_start > block_end)
                continue;

            block_region = uvm_va_block_region_from_start_end(block, block_start, block_end);
            covered_bytes += block_end - block_start + 1;
            candidate_pages += uvm_va_block_region_size(block_region) / PAGE_SIZE;

            if (clear) {
                status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                                 NULL,
                                                 uvm_va_block_clear_gpu_only_soft(block,
                                                                                  block_context,
                                                                                  block_region,
                                                                                  &block_cleared_pages));
            }
            else {
                uvm_va_block_retry_t block_retry;

                status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                                 &block_retry,
                                                 uvm_va_block_apply_gpu_only_soft(block,
                                                                                  &block_retry,
                                                                                  block_context,
                                                                                  gpu->id,
                                                                                  block_region,
                                                                                  read_only,
                                                                                  preserve_cpu_backing,
                                                                                  !async,
                                                                                  &block_applied_pages,
                                                                                  &block_cpu_unmapped_pages,
                                                                                  &block_gpu_mapped_pages,
                                                                                  &block_cpu_preserved_pages));
            }

            if (status != NV_OK)
                goto out;

            applied_pages += block_applied_pages;
            cpu_unmapped_pages += block_cpu_unmapped_pages;
            gpu_mapped_pages += block_gpu_mapped_pages;
            cpu_preserved_pages += block_cpu_preserved_pages;
            cleared_pages += block_cleared_pages;
        }
    }

    if (!managed_range_last || managed_range_last->va_range.node.end < last_address)
        status = NV_ERR_INVALID_ADDRESS;

out:
    uvm_va_space_up_read(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    result->resident_gpu_bytes = (clear ? cleared_pages : applied_pages) * PAGE_SIZE;
    result->resident_cpu_bytes = (preserve_cpu_backing ? cpu_preserved_pages : cpu_unmapped_pages) * PAGE_SIZE;
    result->resident_any_bytes = covered_bytes;
    result->resident_unknown_bytes = length > covered_bytes ? length - covered_bytes : 0;

    uvm_phase_f_route_log("phase_f_gpu_only_soft base=0x%llx length=%llu status=0x%x gpu=%s async=%u read_only=%u clear=%u preserve_cpu_backing=%u covered=%llu candidate_pages=%llu applied_pages=%llu cpu_unmapped_pages=%llu gpu_mapped_pages=%llu cpu_preserved_pages=%llu cleared_pages=%llu\n",
                          base,
                          length,
                          status,
                          gpu ? gpu->name : "none",
                          async,
                          read_only ? 1 : 0,
                          clear ? 1 : 0,
                          preserve_cpu_backing ? 1 : 0,
                          covered_bytes,
                          candidate_pages,
                          applied_pages,
                          cpu_unmapped_pages,
                          gpu_mapped_pages,
                          cpu_preserved_pages,
                          cleared_pages);

    return status;
}

static bool uvm_phase_f_managed_range_is_expert_weight(uvm_va_range_managed_t *managed_range)
{
    /* The caller already supplied an explicit VA range/region id. */
    return managed_range != NULL;
}

static NV_STATUS uvm_phase_f_expert_read_dup_fast(const NvProcessorUuid *gpu_uuid,
                                                  NvBool async,
                                                  NvU64 base,
                                                  NvU64 length,
                                                  UVM_PHASE_F_RANGE_RESULT *result,
                                                  uvm_va_space_t *va_space)
{
    NV_STATUS status = NV_OK;
    uvm_va_range_managed_t *managed_range = NULL;
    uvm_va_range_managed_t *managed_range_last = NULL;
    uvm_va_block_context_t *block_context;
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    NvU64 query_start;
    NvU64 query_end;
    NvU64 last_address;
    NvU64 covered_bytes = 0;
    NvU64 candidate_pages = 0;
    NvU64 applied_pages = 0;
    NvU64 cpu_preserved_pages = 0;
    NvU64 gpu_mapped_pages = 0;

    result->queried_bytes = length;
    result->resident_gpu_bytes = 0;
    result->resident_cpu_bytes = 0;
    result->resident_any_bytes = 0;
    result->resident_unknown_bytes = 0;

    if (uvm_api_range_invalid(base, length))
        return NV_ERR_INVALID_ARGUMENT;

    query_start = base;
    query_end = base + length;
    last_address = query_end - 1;

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_read(va_space);

    gpu = uvm_va_space_get_gpu_by_uuid(va_space, gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto out;
    }

    block_context = uvm_va_space_block_context(va_space, mm);

    uvm_for_each_va_range_managed_in_contig(managed_range, va_space, base, last_address) {
        const NvU64 range_start = max(query_start, managed_range->va_range.node.start);
        const NvU64 range_end = min(last_address, managed_range->va_range.node.end);
        const size_t first_block_index = uvm_va_range_block_index(managed_range, range_start);
        const size_t last_block_index = uvm_va_range_block_index(managed_range, range_end);
        size_t block_index;

        managed_range_last = managed_range;

        if (!uvm_phase_f_managed_range_is_expert_weight(managed_range)) {
            status = NV_ERR_INVALID_ADDRESS;
            goto out;
        }

        for (block_index = first_block_index; block_index <= last_block_index; ++block_index) {
            uvm_va_block_t *block;
            NvU64 block_start;
            NvU64 block_end;
            uvm_va_block_region_t block_region;
            NvU32 block_committed_pages = 0;
            uvm_va_block_retry_t block_retry;

            status = uvm_va_range_block_create(managed_range, block_index, &block);
            if (status != NV_OK)
                goto out;

            block_start = max(query_start, block->start);
            block_end = min(last_address, block->end);
            if (block_start > block_end)
                continue;

            block_region = uvm_va_block_region_from_start_end(block, block_start, block_end);
            covered_bytes += block_end - block_start + 1;
            candidate_pages += uvm_va_block_region_size(block_region) / PAGE_SIZE;

            status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                             &block_retry,
                                             uvm_va_block_promote_read_duplicate_now(block,
                                                                                     &block_retry,
                                                                                     block_context,
                                                                                     gpu->id,
                                                                                     block_region,
                                                                                     !async,
                                                                                     &block_committed_pages));
            if (status != NV_OK)
                goto out;

            applied_pages += block_committed_pages;
            gpu_mapped_pages += block_committed_pages;
            cpu_preserved_pages += block_committed_pages;
        }
    }

    if (!managed_range_last || managed_range_last->va_range.node.end < last_address)
        status = NV_ERR_INVALID_ADDRESS;

out:
    uvm_va_space_up_read(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    result->resident_gpu_bytes = applied_pages * PAGE_SIZE;
    result->resident_cpu_bytes = cpu_preserved_pages * PAGE_SIZE;
    result->resident_any_bytes = covered_bytes;
    result->resident_unknown_bytes = length > covered_bytes ? length - covered_bytes : 0;

    uvm_phase_f_route_log("phase_f_expert_read_dup_fast base=0x%llx length=%llu status=0x%x gpu=%s async=%u covered=%llu candidate_pages=%llu applied_pages=%llu gpu_mapped_pages=%llu cpu_preserved_pages=%llu\n",
                          base,
                          length,
                          status,
                          gpu ? gpu->name : "none",
                          async,
                          covered_bytes,
                          candidate_pages,
                          applied_pages,
                          gpu_mapped_pages,
                          cpu_preserved_pages);

    return status;
}

static NV_STATUS uvm_phase_f_cpu_remote_cutover(const NvProcessorUuid *gpu_uuid,
                                                NvBool async,
                                                NvU64 base,
                                                NvU64 length,
                                                bool require_immutable_shadow,
                                                UVM_PHASE_F_RANGE_RESULT *result,
                                                uvm_va_space_t *va_space)
{
    NV_STATUS status = NV_OK;
    uvm_va_range_managed_t *managed_range = NULL;
    uvm_va_range_managed_t *managed_range_last = NULL;
    uvm_va_block_context_t *block_context;
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    NvU64 last_address;
    NvU64 covered_bytes = 0;
    NvU64 released_bytes = 0;
    NvU64 cpu_shadow_bytes = 0;
    NvU64 remote_mapped_bytes = 0;

    result->queried_bytes = length;
    result->resident_gpu_bytes = 0;
    result->resident_cpu_bytes = 0;
    result->resident_any_bytes = 0;
    result->resident_unknown_bytes = length;

    // This operation frees physical memory and is intentionally synchronous.
    // Userspace must call it only at a scheduler-certified GPU quiescence point.
    if (async ||
        uvm_api_range_invalid(base, length) ||
        !IS_ALIGNED(base, UVM_CHUNK_SIZE_64K) ||
        !IS_ALIGNED(length, UVM_CHUNK_SIZE_64K)) {
        return NV_ERR_INVALID_ARGUMENT;
    }

    last_address = base + length - 1;
    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_write(va_space);

    gpu = uvm_va_space_get_gpu_by_uuid(va_space, gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto out;
    }

    if (uvm_api_range_type_check(va_space, mm, base, length) != UVM_API_RANGE_TYPE_MANAGED) {
        status = NV_ERR_INVALID_ADDRESS;
        goto out;
    }

    block_context = uvm_va_space_block_context(va_space, mm);
    uvm_for_each_va_range_managed_in_contig(managed_range, va_space, base, last_address) {
        const NvU64 range_start = max(base, managed_range->va_range.node.start);
        const NvU64 range_end = min(last_address, managed_range->va_range.node.end);
        const size_t first_block_index = uvm_va_range_block_index(managed_range, range_start);
        const size_t last_block_index = uvm_va_range_block_index(managed_range, range_end);
        size_t block_index;

        managed_range_last = managed_range;
        for (block_index = first_block_index; block_index <= last_block_index; ++block_index) {
            uvm_va_block_t *block;
            uvm_va_block_region_t block_region;
            uvm_va_block_gpu_state_t *gpu_state;
            uvm_page_index_t page_index;
            NvU64 block_start;
            NvU64 block_end;
            NvU64 block_bytes;
            NvU64 block_released_bytes = 0;
            NvU32 newly_mapped_pages = 0;
            NvU32 cpu_pages = 0;
            NvU32 remote_pages = 0;

            status = uvm_va_range_block_create(managed_range, block_index, &block);
            if (status != NV_OK)
                goto out;

            block_start = max(base, block->start);
            block_end = min(last_address, block->end);
            if (block_start > block_end)
                continue;

            block_region = uvm_va_block_region_from_start_end(block, block_start, block_end);
            block_bytes = uvm_va_block_region_size(block_region);

            uvm_mutex_lock(&block->lock);
            status = uvm_va_block_release_gpu_subchunks_to_cpu(block,
                                                               mm,
                                                               gpu,
                                                               block_region,
                                                               UVM_CHUNK_SIZE_64K,
                                                               require_immutable_shadow,
                                                               &block->tracker,
                                                               &block_released_bytes);
            uvm_mutex_unlock(&block->lock);
            if (status != NV_OK ||
                (require_immutable_shadow && block_released_bytes != block_bytes)) {
                if (status == NV_OK)
                    status = NV_ERR_INVALID_STATE;
                goto out;
            }

            status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                             NULL,
                                             uvm_va_block_premap_readmostly_remote_read(block,
                                                                                        block_context,
                                                                                        gpu->id,
                                                                                        block_region,
                                                                                        true,
                                                                                        uvm_va_block_region_size(block_region) !=
                                                                                            uvm_va_block_size(block),
                                                                                        &newly_mapped_pages));
            if (status != NV_OK)
                goto out;

            uvm_mutex_lock(&block->lock);
            gpu_state = uvm_va_block_gpu_state_get(block, gpu->id);
            for_each_va_block_page_in_region(page_index, block_region) {
                bool cpu_resident = uvm_page_mask_test(&block->cpu.resident, page_index);
                bool gpu_resident = gpu_state &&
                                    uvm_page_mask_test(&gpu_state->resident, page_index);
                bool gpu_read_mapped = gpu_state &&
                                       uvm_page_mask_test(&gpu_state->pte_bits[UVM_PTE_BITS_GPU_READ],
                                                          page_index);

                if (cpu_resident)
                    ++cpu_pages;
                if (cpu_resident && !gpu_resident && gpu_read_mapped)
                    ++remote_pages;
            }
            uvm_mutex_unlock(&block->lock);

            if ((NvU64)cpu_pages * PAGE_SIZE != block_bytes ||
                (NvU64)remote_pages * PAGE_SIZE != block_bytes) {
                status = NV_ERR_INVALID_STATE;
                goto out;
            }

            covered_bytes += block_bytes;
            // For the synchronous path this counter is committed cutover
            // bytes, including pages which were already CPU-only. The
            // immutable path retains its stricter newly-released meaning.
            released_bytes += require_immutable_shadow ?
                                  block_released_bytes : block_bytes;
            cpu_shadow_bytes += (NvU64)cpu_pages * PAGE_SIZE;
            remote_mapped_bytes += (NvU64)remote_pages * PAGE_SIZE;
        }
    }

    if (!managed_range_last || managed_range_last->va_range.node.end < last_address ||
        covered_bytes != length || released_bytes != length ||
        cpu_shadow_bytes != length || remote_mapped_bytes != length) {
        status = NV_ERR_INVALID_STATE;
    }

out:
    uvm_va_space_up_write(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    // These result fields have action-specific meanings for action 22:
    // gpu_bytes=released, cpu_bytes=retained shadow, any_bytes=remote mapped.
    result->resident_gpu_bytes = released_bytes;
    result->resident_cpu_bytes = cpu_shadow_bytes;
    result->resident_any_bytes = remote_mapped_bytes;
    result->resident_unknown_bytes = length > covered_bytes ? length - covered_bytes : 0;

    uvm_phase_f_route_log("phase_f_cpu_remote_cutover base=0x%llx length=%llu status=0x%x gpu=%s immutable=%u covered=%llu committed=%llu cpu=%llu remote_mapped=%llu\n",
                          base,
                          length,
                          status,
                          gpu ? gpu->name : "none",
                          require_immutable_shadow ? 1 : 0,
                          covered_bytes,
                          released_bytes,
                          cpu_shadow_bytes,
                          remote_mapped_bytes);
    return status;
}

// Rebuild the complete GPU-visible READ topology for a managed allocation
// from residency state. This is intentionally a coarse correctness repair:
// callers pass complete Parameter ranges at a scheduler-certified quiescent
// point. Rewriting the whole range keeps every shared 2 MiB parent internally
// consistent instead of repairing only a cold or hot slice beneath it.
//
// No physical allocation is moved or copied. Pages resident on the target GPU
// are mapped locally. Pages which have no target-GPU copy but do have a CPU
// copy are mapped remotely from CPU memory. A page with neither copy, or a
// slot-backed mapping whose physical owner is external to UVM residency, is a
// fail-closed error.
static NV_STATUS uvm_phase_f_rebuild_gpu_read_mappings(const NvProcessorUuid *gpu_uuid,
                                                       NvBool async,
                                                       NvU64 base,
                                                       NvU64 length,
                                                       UVM_PHASE_F_RANGE_RESULT *result,
                                                       uvm_va_space_t *va_space)
{
    NV_STATUS status = NV_OK;
    uvm_va_range_managed_t *managed_range = NULL;
    uvm_va_range_managed_t *managed_range_last = NULL;
    uvm_va_block_context_t *block_context;
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    NvU64 last_address;
    NvU64 covered_bytes = 0;
    NvU64 local_gpu_bytes = 0;
    NvU64 remote_cpu_bytes = 0;
    NvU64 read_mapped_bytes = 0;

    result->queried_bytes = length;
    result->resident_gpu_bytes = 0;
    result->resident_cpu_bytes = 0;
    result->resident_any_bytes = 0;
    result->resident_unknown_bytes = length;

    if (async ||
        uvm_api_range_invalid(base, length) ||
        !IS_ALIGNED(base, PAGE_SIZE) ||
        !IS_ALIGNED(length, PAGE_SIZE)) {
        return NV_ERR_INVALID_ARGUMENT;
    }

    last_address = base + length - 1;
    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_read(va_space);

    gpu = uvm_va_space_get_gpu_by_uuid(va_space, gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto out;
    }

    if (uvm_api_range_type_check(va_space, mm, base, length) !=
        UVM_API_RANGE_TYPE_MANAGED) {
        status = NV_ERR_INVALID_ADDRESS;
        goto out;
    }

    block_context = uvm_va_space_block_context(va_space, mm);
    uvm_for_each_va_range_managed_in_contig(managed_range,
                                             va_space,
                                             base,
                                             last_address) {
        const NvU64 range_start = max(base, managed_range->va_range.node.start);
        const NvU64 range_end = min(last_address,
                                    managed_range->va_range.node.end);
        const size_t first_block_index =
            uvm_va_range_block_index(managed_range, range_start);
        const size_t last_block_index =
            uvm_va_range_block_index(managed_range, range_end);
        size_t block_index;

        managed_range_last = managed_range;
        for (block_index = first_block_index;
             block_index <= last_block_index;
             ++block_index) {
            uvm_va_block_t *block;
            uvm_va_block_retry_t block_retry;
            uvm_va_block_region_t block_region;
            uvm_va_block_gpu_state_t *gpu_state;
            uvm_page_mask_t region_pages;
            uvm_page_mask_t gpu_pages;
            uvm_page_mask_t cpu_only_pages;
            uvm_page_mask_t scratch;
            uvm_page_index_t page_index;
            NvU64 block_start;
            NvU64 block_end;
            NvU32 newly_remote_mapped_pages = 0;

            status = uvm_va_range_block_create(managed_range,
                                               block_index,
                                               &block);
            if (status != NV_OK)
                goto out;

            block_start = max(base, block->start);
            block_end = min(last_address, block->end);
            if (block_start > block_end)
                continue;

            block_region =
                uvm_va_block_region_from_start_end(block,
                                                   block_start,
                                                   block_end);
            uvm_page_mask_init_from_region(&region_pages, block_region, NULL);

            // Snapshot the authoritative destinations before clearing stale
            // GPU PTE metadata. Read duplication is resolved in favor of the
            // local GPU copy; CPU is used only when no GPU copy exists.
            uvm_mutex_lock(&block->lock);
            gpu_state = uvm_va_block_gpu_state_get_alloc(block, gpu);
            if (!gpu_state) {
                status = NV_ERR_NO_MEMORY;
            }
            else if (uvm_page_mask_and(&scratch,
                                       &region_pages,
                                       &block->discarded_pages) ||
                     uvm_page_mask_and(&scratch,
                                       &region_pages,
                                       &gpu_state->moe_slot_pte_bits)) {
                status = NV_ERR_INVALID_STATE;
            }
            else {
                uvm_page_mask_and(&gpu_pages,
                                  &region_pages,
                                  &gpu_state->resident);
                uvm_page_mask_and(&cpu_only_pages,
                                  &region_pages,
                                  &block->cpu.resident);
                uvm_page_mask_andnot(&cpu_only_pages,
                                     &cpu_only_pages,
                                     &gpu_pages);
                uvm_page_mask_or(&scratch,
                                 &gpu_pages,
                                 &cpu_only_pages);
                if (!uvm_page_mask_subset(&region_pages, &scratch))
                    status = NV_ERR_INVALID_STATE;
            }
            uvm_mutex_unlock(&block->lock);
            if (status != NV_OK) {
                uvm_phase_f_route_log_always(
                    "phase_f_rebuild_gpu_read_reject block=0x%llx stage=residency_snapshot status=0x%x region=[%u,%u)\n",
                    block->start,
                    status,
                    block_region.first,
                    block_region.outer);
                goto out;
            }

            // A complete unmap followed by local and remote READ maps forces
            // the hardware hierarchy and UVM's software PTE bits back into
            // agreement. Allocation retries are handled by the standard block
            // helpers; no residency operation is issued here.
            status = UVM_VA_BLOCK_LOCK_RETRY(
                block,
                &block_retry,
                uvm_va_block_unmap(block,
                                   block_context,
                                   gpu->id,
                                   block_region,
                                   &region_pages,
                                   &block->tracker));
            if (status != NV_OK) {
                uvm_phase_f_route_log_always(
                    "phase_f_rebuild_gpu_read_reject block=0x%llx stage=unmap status=0x%x gpu_pages=%u cpu_only_pages=%u\n",
                    block->start,
                    status,
                    uvm_page_mask_weight(&gpu_pages),
                    uvm_page_mask_weight(&cpu_only_pages));
                goto out;
            }

            if (!uvm_page_mask_empty(&gpu_pages)) {
                status = UVM_VA_BLOCK_LOCK_RETRY(
                    block,
                    &block_retry,
                    uvm_va_block_map(block,
                                     block_context,
                                     gpu->id,
                                     block_region,
                                     &gpu_pages,
                                     UVM_PROT_READ_ONLY,
                                     UvmEventMapRemoteCausePolicy,
                                     &block->tracker));
                if (status != NV_OK) {
                    uvm_phase_f_route_log_always(
                        "phase_f_rebuild_gpu_read_reject block=0x%llx stage=map_local status=0x%x gpu_pages=%u cpu_only_pages=%u\n",
                        block->start,
                        status,
                        uvm_page_mask_weight(&gpu_pages),
                        uvm_page_mask_weight(&cpu_only_pages));
                    goto out;
                }
            }

            if (!uvm_page_mask_empty(&cpu_only_pages)) {
                status = UVM_VA_BLOCK_LOCK_RETRY(
                    block,
                    &block_retry,
                    uvm_va_block_premap_readmostly_remote_read(
                        block,
                        block_context,
                        gpu->id,
                        block_region,
                        true,
                        false,
                        &newly_remote_mapped_pages));
                if (status != NV_OK) {
                    uvm_phase_f_route_log_always(
                        "phase_f_rebuild_gpu_read_reject block=0x%llx stage=map_remote status=0x%x gpu_pages=%u cpu_only_pages=%u newly_remote=%u\n",
                        block->start,
                        status,
                        uvm_page_mask_weight(&gpu_pages),
                        uvm_page_mask_weight(&cpu_only_pages),
                        newly_remote_mapped_pages);
                    goto out;
                }
            }

            status = UVM_VA_BLOCK_LOCK_RETRY(
                block,
                &block_retry,
                uvm_va_block_force_gpu_parent_mapping_rewrite(
                    block,
                    block_context,
                    gpu,
                    &block->tracker));
            if (status != NV_OK) {
                uvm_phase_f_route_log_always(
                    "phase_f_rebuild_gpu_read_reject block=0x%llx stage=parent_rewrite status=0x%x gpu_pages=%u cpu_only_pages=%u\n",
                    block->start,
                    status,
                    uvm_page_mask_weight(&gpu_pages),
                    uvm_page_mask_weight(&cpu_only_pages));
                goto out;
            }

            uvm_mutex_lock(&block->lock);
            status = uvm_tracker_wait(&block->tracker);
            gpu_state = uvm_va_block_gpu_state_get(block, gpu->id);
            if (status == NV_OK &&
                (!gpu_state ||
                 !uvm_page_mask_subset(
                     &region_pages,
                     &gpu_state->pte_bits[UVM_PTE_BITS_GPU_READ]) ||
                 uvm_page_mask_and(
                     &scratch,
                     &region_pages,
                     &gpu_state->pte_bits[UVM_PTE_BITS_GPU_WRITE]) ||
                 uvm_page_mask_and(
                     &scratch,
                     &region_pages,
                     &gpu_state->pte_bits[UVM_PTE_BITS_GPU_ATOMIC]) ||
                 uvm_page_mask_and(&scratch,
                                   &region_pages,
                                   &gpu_state->moe_slot_pte_bits))) {
                status = NV_ERR_INVALID_STATE;
            }

            if (status == NV_OK) {
                for_each_va_block_page_in_region(page_index, block_region) {
                    if (uvm_page_mask_test(&gpu_pages, page_index))
                        local_gpu_bytes += PAGE_SIZE;
                    else if (uvm_page_mask_test(&cpu_only_pages, page_index))
                        remote_cpu_bytes += PAGE_SIZE;
                    read_mapped_bytes += PAGE_SIZE;
                }
            }
            uvm_mutex_unlock(&block->lock);
            if (status != NV_OK) {
                uvm_phase_f_route_log_always(
                    "phase_f_rebuild_gpu_read_reject block=0x%llx stage=postcheck status=0x%x gpu_pages=%u cpu_only_pages=%u\n",
                    block->start,
                    status,
                    uvm_page_mask_weight(&gpu_pages),
                    uvm_page_mask_weight(&cpu_only_pages));
                goto out;
            }

            covered_bytes += block_end - block_start + 1;
        }
    }

    if (!managed_range_last ||
        managed_range_last->va_range.node.end < last_address ||
        covered_bytes != length || read_mapped_bytes != length) {
        status = NV_ERR_INVALID_STATE;
    }

out:
    uvm_va_space_up_read(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    result->resident_gpu_bytes = local_gpu_bytes;
    result->resident_cpu_bytes = remote_cpu_bytes;
    result->resident_any_bytes = read_mapped_bytes;
    result->resident_unknown_bytes =
        length > covered_bytes ? length - covered_bytes : 0;

    uvm_phase_f_route_log_always(
        "phase_f_rebuild_gpu_read_mappings base=0x%llx length=%llu status=0x%x gpu=%s covered=%llu local_gpu=%llu remote_cpu=%llu read_mapped=%llu\n",
        base,
        length,
        status,
        gpu ? gpu->name : "none",
        covered_bytes,
        local_gpu_bytes,
        remote_cpu_bytes,
        read_mapped_bytes);
    return status;
}

static NV_STATUS uvm_phase_f_execute_range(NvU32 action,
                                           const NvProcessorUuid *gpu_uuid,
                                           NvBool async,
                                           NvU64 base,
                                           NvU64 length,
                                           UVM_PHASE_F_RANGE_RESULT *result,
                                           struct file *filp)
{
    NV_STATUS status = NV_OK;
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    struct mm_struct *mm = NULL;
    uvm_va_range_t *va_range = NULL;
    uvm_api_range_type_t range_type = UVM_API_RANGE_TYPE_INVALID;
    const char *range_type_name = "INVALID";
    NvU64 last_address = 0;
    bool range_invalid = true;
    bool range_empty = true;
    bool skip_range_diag = action == UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST ||
                           action == UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST_NATIVE_REGIONS;

    result->status = NV_OK;
    result->resident_on_count = 0;
    result->resident_on_cpu = NV_FALSE;
    result->resident_on_gpu = NV_FALSE;
    result->queried_bytes = 0;
    result->resident_gpu_bytes = 0;
    result->resident_cpu_bytes = 0;
    result->resident_any_bytes = 0;
    result->resident_unknown_bytes = 0;

    if (!skip_range_diag && base != 0 && length != 0) {
        range_invalid = uvm_api_range_invalid(base, length);
        if (!range_invalid)
            last_address = base + length - 1;

        mm = uvm_va_space_mm_or_current_retain_lock(va_space);
        uvm_va_space_down_read(va_space);

        if (!range_invalid) {
            va_range = uvm_va_range_find(va_space, base);
            range_empty = uvm_va_space_range_empty(va_space, base, last_address);
            range_type = uvm_api_range_type_check(va_space, mm, base, length);
        }

        if (range_type == UVM_API_RANGE_TYPE_MANAGED)
            range_type_name = "MANAGED";
        else if (range_type == UVM_API_RANGE_TYPE_HMM)
            range_type_name = "HMM";
        else if (range_type == UVM_API_RANGE_TYPE_ATS)
            range_type_name = "ATS";

        uvm_phase_f_route_log("phase_f_range_diag action=%u base=0x%llx length=%llu last=0x%llx range_invalid=%u base_page_aligned=%u length_page_aligned=%u type=%s range_empty=%u va_range=%p va_range_type=%d va_start=0x%llx va_end=0x%llx va_space=%p mm=%p current_mm=%p\n",
                              action,
                              base,
                              length,
                              last_address,
                              range_invalid,
                              IS_ALIGNED(base, PAGE_SIZE),
                              IS_ALIGNED(length, PAGE_SIZE),
                              range_type_name,
                              range_empty,
                              va_range,
                              va_range ? va_range->type : UVM_VA_RANGE_TYPE_INVALID,
                              va_range ? va_range->node.start : 0,
                              va_range ? va_range->node.end : 0,
                              va_space,
                              mm,
                              current->mm);

        uvm_va_space_up_read(va_space);
        uvm_va_space_mm_or_current_release_unlock(va_space, mm);
    }

    if (base == 0 || length == 0) {
        status = NV_ERR_INVALID_ARGUMENT;
    }
    else if (action == UVM_PHASE_F_ACTION_ADVISE_CPU) {
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
    else if (action == UVM_PHASE_F_ACTION_PREFETCH_CPU ||
             action == UVM_PHASE_F_ACTION_PREFETCH_CPU_NO_MAP ||
             action == UVM_PHASE_F_ACTION_PREFETCH_GPU) {
        UVM_MIGRATE_PARAMS *migrate_params =
            uvm_kvmalloc_zero(sizeof(*migrate_params));

        if (!migrate_params) {
            status = NV_ERR_NO_MEMORY;
        }
        else if (action == UVM_PHASE_F_ACTION_PREFETCH_CPU_NO_MAP) {
            status = uvm_migrate_managed_range_to_cpu(va_space,
                                                      base,
                                                      length,
                                                      (async ? UVM_MIGRATE_FLAG_ASYNC : 0) |
                                                      UVM_MIGRATE_FLAG_SKIP_CPU_MAP);
            uvm_kvfree(migrate_params);
        }
        else {
            migrate_params->base = base;
            migrate_params->length = length;
            migrate_params->destinationUuid =
                action == UVM_PHASE_F_ACTION_PREFETCH_CPU ?
                    NV_PROCESSOR_UUID_CPU_DEFAULT : *gpu_uuid;
            migrate_params->flags = async ? UVM_MIGRATE_FLAG_ASYNC : 0;
            migrate_params->semaphoreAddress = 0;
            migrate_params->semaphorePayload = 0;
            migrate_params->cpuNumaNode = NUMA_NO_NODE;

            status = uvm_api_migrate(migrate_params, filp);
            uvm_kvfree(migrate_params);
        }
    }
    else if (action == UVM_PHASE_F_ACTION_QUERY_RESIDENCY) {
        UVM_TEST_VA_RESIDENCY_INFO_PARAMS *query_params =
            uvm_kvmalloc_zero(sizeof(*query_params));
        NvU32 resident_index;

        if (!query_params) {
            status = NV_ERR_NO_MEMORY;
        }
        else {
            query_params->lookup_address = base;
            query_params->is_async = async;

            status = uvm_test_va_residency_info(query_params, filp);
            if (status == NV_OK) {
                result->resident_on_count = query_params->resident_on_count;

                for (resident_index = 0;
                     resident_index < query_params->resident_on_count &&
                     resident_index < UVM_MAX_PROCESSORS;
                     ++resident_index) {
                    if (uvm_uuid_eq(&query_params->resident_on[resident_index],
                                    &NV_PROCESSOR_UUID_CPU_DEFAULT))
                        result->resident_on_cpu = NV_TRUE;

                    if (uvm_uuid_eq(&query_params->resident_on[resident_index],
                                    gpu_uuid))
                        result->resident_on_gpu = NV_TRUE;
                }
            }

            uvm_kvfree(query_params);

            if (status == NV_OK) {
                status = uvm_phase_f_query_resident_bytes(gpu_uuid,
                                                          async,
                                                          base,
                                                          length,
                                                          result,
                                                          va_space);
            }
        }
    }
    else if (action == UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ ||
             action == UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG ||
             action == UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG_DEFERRED ||
             action == UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_AUDITED) {
        if (action == UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_AUDITED && async) {
            status = NV_ERR_INVALID_ARGUMENT;
        }
        else {
            status = uvm_phase_f_prepare_remote_read(
                gpu_uuid,
                async,
                base,
                length,
                action == UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG ||
                    action == UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG_DEFERRED,
                action == UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG,
                action == UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_AUDITED,
                result,
                va_space);
        }
    }
    else if (action == UVM_PHASE_F_ACTION_FORCE_REMOTE_READ_64K) {
        status = uvm_phase_f_force_remote_read_64k(gpu_uuid,
                                                   async,
                                                   base,
                                                   length,
                                                   result,
                                                   va_space);
    }
    else if (action == UVM_PHASE_F_ACTION_START_BG_PROMOTION) {
        status = uvm_phase_f_start_bg_promotion(gpu_uuid,
                                                async,
                                                base,
                                                length,
                                                result,
                                                va_space);
    }
    else if (action == UVM_PHASE_F_ACTION_PAUSE_BG_PROMOTION ||
             action == UVM_PHASE_F_ACTION_RESUME_BG_PROMOTION) {
        bool pause = action == UVM_PHASE_F_ACTION_PAUSE_BG_PROMOTION;

        uvm_va_block_note_bg_promote_gate_action(pause);
        status = uvm_phase_f_set_bg_promotion_paused(base,
                                                     length,
                                                     pause,
                                                     result,
                                                     va_space);
    }
    else if (action == UVM_PHASE_F_ACTION_GPU_ONLY_READ_SOFT ||
             action == UVM_PHASE_F_ACTION_GPU_ONLY_RW_SOFT ||
             action == UVM_PHASE_F_ACTION_GPU_ONLY_READ_DUP_SOFT ||
             action == UVM_PHASE_F_ACTION_CLEAR_GPU_ONLY) {
        status = uvm_phase_f_gpu_only_soft(gpu_uuid,
                                           async,
                                           base,
                                           length,
                                           action == UVM_PHASE_F_ACTION_GPU_ONLY_READ_SOFT ||
                                           action == UVM_PHASE_F_ACTION_GPU_ONLY_READ_DUP_SOFT,
                                           action == UVM_PHASE_F_ACTION_CLEAR_GPU_ONLY,
                                           action == UVM_PHASE_F_ACTION_GPU_ONLY_READ_DUP_SOFT,
                                           result,
                                           va_space);
    }
    else if (action == UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST ||
             action == UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST_NATIVE_REGIONS) {
        status = uvm_phase_f_expert_read_dup_fast(gpu_uuid,
                                                  async,
                                                  base,
                                                  length,
                                                  result,
                                                  va_space);
    }
    else if (action == UVM_PHASE_F_ACTION_IMMUTABLE_CPU_SHADOW_CUTOVER ||
             action == UVM_PHASE_F_ACTION_SYNC_CPU_REMOTE_CUTOVER) {
        status = uvm_phase_f_cpu_remote_cutover(
            gpu_uuid,
            async,
            base,
            length,
            action == UVM_PHASE_F_ACTION_IMMUTABLE_CPU_SHADOW_CUTOVER,
            result,
            va_space);
    }
    else if (action == UVM_PHASE_F_ACTION_REBUILD_GPU_READ_MAPPINGS) {
        status = uvm_phase_f_rebuild_gpu_read_mappings(gpu_uuid,
                                                       async,
                                                       base,
                                                       length,
                                                       result,
                                                       va_space);
    }
    else {
        status = NV_ERR_INVALID_ARGUMENT;
    }

    result->status = status;
    uvm_phase_f_route_log("phase_f_range_action_end action=%u base=0x%llx end=0x%llx length=%llu status=0x%x gpu_bytes=%llu cpu_bytes=%llu any_bytes=%llu unknown_bytes=%llu async=%u\n",
                          action,
                          base,
                          base + length,
                          length,
                          status,
                          result->resident_gpu_bytes,
                          result->resident_cpu_bytes,
                          result->resident_any_bytes,
                          result->resident_unknown_bytes,
                          async);
    return status;
}

static NV_STATUS uvm_phase_f_note_status(NV_STATUS status,
                                         NvU32 *success_count,
                                         NvU32 *failure_count,
                                         NV_STATUS *first_error)
{
    if (status == NV_OK) {
        ++*success_count;
    }
    else {
        ++*failure_count;
        if (*first_error == NV_OK)
            *first_error = status;
    }

    return status;
}

static NV_STATUS uvm_moe_slot_pool_user_backing_batch_tlb_invalidate(uvm_va_space_t *va_space,
                                                                      uvm_gpu_t *gpu);

typedef struct
{
    NvU64 covered_bytes;
    NvU64 candidate_pages;
    NvU64 committed_pages;
} uvm_phase_f_expert_batch_range_state_t;

typedef struct
{
    uvm_phase_f_expert_batch_range_state_t range_states[UVM_PHASE_F_MAX_RANGES];
    uvm_va_block_expert_batch_entry_t entries[UVM_PHASE_F_MAX_RANGES];
    NvU32 entry_range_indices[UVM_PHASE_F_MAX_RANGES];
    uvm_va_range_managed_t *single_managed_range;
    size_t single_first_block_index;
    size_t single_last_block_index;
    DECLARE_BITMAP(single_block_seen, UVM_PHASE_F_MAX_RANGES);
    NvU32 single_block_owner[UVM_PHASE_F_MAX_RANGES];
} uvm_phase_f_expert_batch_region_work_t;

static void uvm_phase_f_result_init(UVM_PHASE_F_RANGE_RESULT *result, NvU64 queried_bytes)
{
    result->status = NV_OK;
    result->resident_on_count = 0;
    result->resident_on_cpu = NV_FALSE;
    result->resident_on_gpu = NV_FALSE;
    result->queried_bytes = queried_bytes;
    result->resident_gpu_bytes = 0;
    result->resident_cpu_bytes = 0;
    result->resident_any_bytes = 0;
    result->resident_unknown_bytes = 0;
}

static NV_STATUS uvm_phase_f_expert_batch_count_blocks(uvm_va_space_t *va_space,
                                                       UVM_PHASE_F_BATCH_EXECUTOR_PARAMS *params,
                                                       uvm_phase_f_expert_batch_range_state_t *range_states,
                                                       size_t *entry_count_out)
{
    NvU32 range_index;

    *entry_count_out = 0;

    for (range_index = 0; range_index < params->count; ++range_index) {
        UVM_PHASE_F_RANGE *range = &params->ranges[range_index];
        uvm_phase_f_expert_batch_range_state_t *range_state = &range_states[range_index];
        uvm_va_range_managed_t *managed_range = NULL;
        uvm_va_range_managed_t *managed_range_last = NULL;
        NvU64 last_address;

        uvm_phase_f_result_init(&params->results[range_index], range->length);

        if (range->base == 0 || range->length == 0 || uvm_api_range_invalid(range->base, range->length))
            return NV_WARN_MORE_PROCESSING_REQUIRED;

        last_address = range->base + range->length - 1;

        uvm_for_each_va_range_managed_in_contig(managed_range, va_space, range->base, last_address) {
            NvU64 range_start = max(range->base, managed_range->va_range.node.start);
            NvU64 range_end = min(last_address, managed_range->va_range.node.end);
            size_t first_block_index;
            size_t last_block_index;
            size_t block_count;

            managed_range_last = managed_range;

            if (!uvm_phase_f_managed_range_is_expert_weight(managed_range))
                return NV_WARN_MORE_PROCESSING_REQUIRED;

            first_block_index = uvm_va_range_block_index(managed_range, range_start);
            last_block_index = uvm_va_range_block_index(managed_range, range_end);
            block_count = last_block_index - first_block_index + 1;

            if (block_count > UVM_PHASE_F_MAX_RANGES || *entry_count_out > UVM_PHASE_F_MAX_RANGES - block_count)
                return NV_WARN_MORE_PROCESSING_REQUIRED;

            *entry_count_out += block_count;
            range_state->covered_bytes += range_end - range_start + 1;
            range_state->candidate_pages += (range_end - range_start + 1) / PAGE_SIZE;
        }

        if (!managed_range_last || managed_range_last->va_range.node.end < last_address)
            return NV_WARN_MORE_PROCESSING_REQUIRED;

        if (range_state->covered_bytes != range->length)
            return NV_WARN_MORE_PROCESSING_REQUIRED;
    }

    if (*entry_count_out == 0)
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    return NV_OK;
}

static NV_STATUS uvm_phase_f_expert_batch_fill_entries(uvm_va_space_t *va_space,
                                                       UVM_PHASE_F_BATCH_EXECUTOR_PARAMS *params,
                                                       uvm_va_block_expert_batch_entry_t *entries,
                                                       NvU32 *entry_range_indices,
                                                       size_t entry_count)
{
    size_t entry_index = 0;
    NvU32 range_index;

    for (range_index = 0; range_index < params->count; ++range_index) {
        UVM_PHASE_F_RANGE *range = &params->ranges[range_index];
        uvm_va_range_managed_t *managed_range = NULL;
        NvU64 last_address = range->base + range->length - 1;

        uvm_for_each_va_range_managed_in_contig(managed_range, va_space, range->base, last_address) {
            NvU64 range_start = max(range->base, managed_range->va_range.node.start);
            NvU64 range_end = min(last_address, managed_range->va_range.node.end);
            size_t first_block_index = uvm_va_range_block_index(managed_range, range_start);
            size_t last_block_index = uvm_va_range_block_index(managed_range, range_end);
            size_t block_index;

            for (block_index = first_block_index; block_index <= last_block_index; ++block_index) {
                uvm_va_block_t *block;
                NvU64 block_start;
                NvU64 block_end;
                NV_STATUS status;

                UVM_ASSERT(entry_index < entry_count);

                status = uvm_va_range_block_create(managed_range, block_index, &block);
                if (status != NV_OK)
                    return status;

                block_start = max(range->base, block->start);
                block_end = min(last_address, block->end);
                if (block_start > block_end)
                    continue;

                entries[entry_index].va_block = block;
                entries[entry_index].region = uvm_va_block_region_from_start_end(block, block_start, block_end);
                entries[entry_index].committed_pages = 0;
                entry_range_indices[entry_index] = range_index;
                ++entry_index;
            }
        }
    }

    UVM_ASSERT(entry_index == entry_count);
    return NV_OK;
}

static bool uvm_phase_f_registered_region_is_valid(const uvm_phase_f_region_t *region)
{
    return region && region->region_id != 0 && region->base != 0 && region->length != 0;
}

static NV_STATUS uvm_phase_f_expert_batch_collect_region(uvm_va_space_t *va_space,
                                                         NvU64 region_id,
                                                         NvU32 range_index,
                                                         uvm_phase_f_expert_batch_range_state_t *range_state,
                                                         uvm_va_block_expert_batch_entry_t *entries,
                                                         NvU32 *entry_range_indices,
                                                         size_t *entry_count_inout)
{
    uvm_phase_f_region_t *region;
    uvm_va_range_managed_t *managed_range = NULL;
    uvm_va_range_managed_t *managed_range_last = NULL;
    NvU64 base;
    NvU64 length;
    NvU64 last_address;

    if (region_id == 0)
        return NV_ERR_INVALID_ARGUMENT;

    region = radix_tree_lookup(&va_space->phase_f_regions, region_id);
    if (!region)
        return NV_ERR_OBJECT_NOT_FOUND;

    if (!uvm_phase_f_registered_region_is_valid(region))
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    base = region->base;
    length = region->length;
    if (base == 0 || length == 0 || uvm_api_range_invalid(base, length))
        return NV_WARN_MORE_PROCESSING_REQUIRED;
    if (!IS_ALIGNED(base, UVM_VA_BLOCK_SIZE) || !IS_ALIGNED(length, UVM_VA_BLOCK_SIZE))
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    last_address = base + length - 1;
    range_state->covered_bytes = 0;
    range_state->candidate_pages = 0;
    range_state->committed_pages = 0;

    uvm_for_each_va_range_managed_in_contig(managed_range, va_space, base, last_address) {
        NvU64 range_start = max(base, managed_range->va_range.node.start);
        NvU64 range_end = min(last_address, managed_range->va_range.node.end);
        size_t first_block_index;
        size_t last_block_index;
        size_t block_index;

        managed_range_last = managed_range;

        first_block_index = uvm_va_range_block_index(managed_range, range_start);
        last_block_index = uvm_va_range_block_index(managed_range, range_end);

        if (last_block_index - first_block_index + 1 > UVM_PHASE_F_MAX_RANGES ||
            *entry_count_inout > UVM_PHASE_F_MAX_RANGES - (last_block_index - first_block_index + 1)) {
            return NV_WARN_MORE_PROCESSING_REQUIRED;
        }

        range_state->covered_bytes += range_end - range_start + 1;
        range_state->candidate_pages += (range_end - range_start + 1) / PAGE_SIZE;

        for (block_index = first_block_index; block_index <= last_block_index; ++block_index) {
            uvm_va_block_t *block;
            NvU64 block_start;
            NvU64 block_end;
            NV_STATUS status;

            status = uvm_va_range_block_create(managed_range, block_index, &block);
            if (status != NV_OK)
                return status;

            block_start = max(base, block->start);
            block_end = min(last_address, block->end);
            if (block_start > block_end)
                continue;

            entries[*entry_count_inout].va_block = block;
            entries[*entry_count_inout].region = uvm_va_block_region_from_start_end(block, block_start, block_end);
            entries[*entry_count_inout].committed_pages = 0;
            entry_range_indices[*entry_count_inout] = range_index;
            ++*entry_count_inout;
        }
    }

    if (!managed_range_last || managed_range_last->va_range.node.end < last_address)
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    if (range_state->covered_bytes != length)
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    return NV_OK;
}

static NV_STATUS uvm_phase_f_expert_batch_probe_single_va_range_region(uvm_va_space_t *va_space,
                                                                       NvU64 region_id,
                                                                       NvU32 range_index,
                                                                       uvm_phase_f_expert_batch_region_work_t *work)
{
    uvm_phase_f_region_t *region;
    uvm_va_range_t *va_range;
    uvm_va_range_managed_t *managed_range;
    uvm_phase_f_expert_batch_range_state_t *range_state = &work->range_states[range_index];
    NvU64 base;
    NvU64 length;
    NvU64 last_address;
    size_t first_block_index;
    size_t last_block_index;
    size_t block_count;

    if (region_id == 0)
        return NV_ERR_INVALID_ARGUMENT;

    region = radix_tree_lookup(&va_space->phase_f_regions, region_id);
    if (!region)
        return NV_ERR_OBJECT_NOT_FOUND;

    if (!uvm_phase_f_registered_region_is_valid(region))
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    base = region->base;
    length = region->length;
    if (base == 0 || length == 0 || uvm_api_range_invalid(base, length))
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    last_address = base + length - 1;
    va_range = uvm_va_range_find(va_space, base);
    managed_range = uvm_va_range_to_managed_or_null(va_range);
    if (!managed_range || managed_range->va_range.node.end < last_address)
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    if (work->single_managed_range && work->single_managed_range != managed_range)
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    first_block_index = uvm_va_range_block_index(managed_range, base);
    last_block_index = uvm_va_range_block_index(managed_range, last_address);
    block_count = last_block_index - first_block_index + 1;
    if (block_count > UVM_PHASE_F_MAX_RANGES)
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    if (!work->single_managed_range) {
        work->single_managed_range = managed_range;
        work->single_first_block_index = first_block_index;
        work->single_last_block_index = last_block_index;
    }
    else {
        work->single_first_block_index = min(work->single_first_block_index, first_block_index);
        work->single_last_block_index = max(work->single_last_block_index, last_block_index);
        if (work->single_last_block_index - work->single_first_block_index + 1 > UVM_PHASE_F_MAX_RANGES)
            return NV_WARN_MORE_PROCESSING_REQUIRED;
    }

    range_state->covered_bytes = length;
    range_state->candidate_pages = length / PAGE_SIZE;
    range_state->committed_pages = 0;
    return NV_OK;
}

static NV_STATUS uvm_phase_f_expert_batch_mark_single_va_range_region(uvm_va_space_t *va_space,
                                                                      NvU64 region_id,
                                                                      NvU32 range_index,
                                                                      uvm_phase_f_expert_batch_region_work_t *work)
{
    uvm_phase_f_region_t *region;
    uvm_va_range_managed_t *managed_range = work->single_managed_range;
    NvU64 base;
    NvU64 last_address;
    size_t first_block_index;
    size_t last_block_index;
    size_t block_index;

    UVM_ASSERT(managed_range);

    region = radix_tree_lookup(&va_space->phase_f_regions, region_id);
    if (!region)
        return NV_ERR_OBJECT_NOT_FOUND;

    base = region->base;
    last_address = base + region->length - 1;
    first_block_index = uvm_va_range_block_index(managed_range, base);
    last_block_index = uvm_va_range_block_index(managed_range, last_address);

    for (block_index = first_block_index; block_index <= last_block_index; ++block_index) {
        size_t seen_index = block_index - work->single_first_block_index;

        if (!test_and_set_bit(seen_index, work->single_block_seen))
            work->single_block_owner[seen_index] = range_index;
    }

    return NV_OK;
}

static NV_STATUS uvm_phase_f_expert_batch_emit_single_va_range_entries(uvm_phase_f_expert_batch_region_work_t *work,
                                                                       UVM_PHASE_F_BATCH_REGIONS_PARAMS *params,
                                                                       size_t *entry_count_out)
{
    uvm_va_range_managed_t *managed_range = work->single_managed_range;
    size_t relative_index;
    size_t span;

    *entry_count_out = 0;
    UVM_ASSERT(managed_range);

    span = work->single_last_block_index - work->single_first_block_index + 1;
    for (relative_index = 0; relative_index < span; ++relative_index) {
        size_t block_index;
        NvU32 owner;
        uvm_phase_f_region_t *region;
        uvm_va_block_t *block;
        NvU64 last_address;
        NvU64 block_start;
        NvU64 block_end;
        NV_STATUS status;

        if (!test_bit(relative_index, work->single_block_seen))
            continue;

        if (*entry_count_out >= UVM_PHASE_F_MAX_RANGES)
            return NV_WARN_MORE_PROCESSING_REQUIRED;

        block_index = work->single_first_block_index + relative_index;
        owner = work->single_block_owner[relative_index];
        region = radix_tree_lookup(&managed_range->va_range.va_space->phase_f_regions, params->region_ids[owner]);
        if (!region)
            return NV_ERR_OBJECT_NOT_FOUND;

        status = uvm_va_range_block_create(managed_range, block_index, &block);
        if (status != NV_OK)
            return status;

        last_address = region->base + region->length - 1;
        block_start = max(region->base, block->start);
        block_end = min(last_address, block->end);
        if (block_start > block_end)
            continue;

        work->entries[*entry_count_out].va_block = block;
        work->entries[*entry_count_out].region = uvm_va_block_region_from_start_end(block, block_start, block_end);
        work->entries[*entry_count_out].committed_pages = 0;
        work->entry_range_indices[*entry_count_out] = owner;
        ++*entry_count_out;
    }

    return NV_OK;
}

static void uvm_phase_f_expert_batch_sort_entries(uvm_va_block_expert_batch_entry_t *entries,
                                                  NvU32 *entry_range_indices,
                                                  size_t entry_count)
{
    size_t i;

    for (i = 1; i < entry_count; ++i) {
        uvm_va_block_expert_batch_entry_t entry = entries[i];
        NvU32 range_index = entry_range_indices[i];
        size_t j = i;

        while (j > 0 && entries[j - 1].va_block->start > entry.va_block->start) {
            entries[j] = entries[j - 1];
            entry_range_indices[j] = entry_range_indices[j - 1];
            --j;
        }

        entries[j] = entry;
        entry_range_indices[j] = range_index;
    }
}

static bool uvm_phase_f_expert_batch_has_duplicate_blocks(uvm_va_block_expert_batch_entry_t *entries,
                                                          size_t entry_count)
{
    size_t entry_index;

    for (entry_index = 1; entry_index < entry_count; ++entry_index) {
        if (entries[entry_index - 1].va_block == entries[entry_index].va_block)
            return true;
    }

    return false;
}

static NV_STATUS uvm_phase_f_expert_read_dup_fast_batch(const NvProcessorUuid *gpu_uuid,
                                                        NvBool async,
                                                        UVM_PHASE_F_BATCH_EXECUTOR_PARAMS *params,
                                                        struct file *filp)
{
    NV_STATUS status = NV_OK;
    NV_STATUS first_error = NV_OK;
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    struct mm_struct *mm = NULL;
    uvm_gpu_t *gpu = NULL;
    uvm_phase_f_expert_batch_range_state_t *range_states = NULL;
    uvm_va_block_expert_batch_entry_t *entries = NULL;
    NvU32 *entry_range_indices = NULL;
    uvm_processor_mask_t *involved_processors = NULL;
    uvm_tracker_t batch_tracker = UVM_TRACKER_INIT();
    size_t entry_count = 0;
    size_t entry_index;
    NvU32 range_index;
    NvU64 committed_pages_total = 0;
    bool mmap_lock_dropped = false;

    UVM_ASSERT(params->action == UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST);
    UVM_ASSERT(params->count > 1);

    range_states = uvm_kvmalloc_zero(sizeof(*range_states) * params->count);
    if (!range_states)
        return NV_ERR_NO_MEMORY;

    for (range_index = 0; range_index < params->count; ++range_index)
        uvm_phase_f_result_init(&params->results[range_index], params->ranges[range_index].length);

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_read(va_space);

    gpu = uvm_va_space_get_gpu_by_uuid(va_space, gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto out;
    }

    status = uvm_phase_f_expert_batch_count_blocks(va_space, params, range_states, &entry_count);
    if (status != NV_OK)
        goto out;

    entries = uvm_kvmalloc_zero(sizeof(*entries) * entry_count);
    entry_range_indices = uvm_kvmalloc_zero(sizeof(*entry_range_indices) * entry_count);
    involved_processors = uvm_processor_mask_cache_alloc();
    if (!entries || !entry_range_indices || !involved_processors) {
        status = NV_ERR_NO_MEMORY;
        goto out;
    }

    status = uvm_phase_f_expert_batch_fill_entries(va_space,
                                                   params,
                                                   entries,
                                                   entry_range_indices,
                                                   entry_count);
    if (status != NV_OK)
        goto out;

    uvm_phase_f_expert_batch_sort_entries(entries, entry_range_indices, entry_count);
    if (uvm_phase_f_expert_batch_has_duplicate_blocks(entries, entry_count)) {
        status = NV_WARN_MORE_PROCESSING_REQUIRED;
        goto out;
    }

    status = uvm_va_block_expert_promote_read_duplicate_batch(va_space,
                                                              mm,
                                                              gpu,
                                                              entries,
                                                              entry_count,
                                                              &batch_tracker,
                                                              involved_processors);
    if (status != NV_OK)
        goto out;

    if (!async) {
        if (mm) {
            uvm_up_read_mmap_lock_out_of_order(mm);
            mmap_lock_dropped = true;
        }

        status = uvm_tracker_wait(&batch_tracker);
        if (status != NV_OK)
            goto out;
    }

    for (entry_index = 0; entry_index < entry_count; ++entry_index) {
        NvU32 owner = entry_range_indices[entry_index];

        range_states[owner].committed_pages += entries[entry_index].committed_pages;
        committed_pages_total += entries[entry_index].committed_pages;
    }

out:
    if (status != NV_WARN_MORE_PROCESSING_REQUIRED) {
        params->success_count = 0;
        params->failure_count = 0;

        for (range_index = 0; range_index < params->count; ++range_index) {
            UVM_PHASE_F_RANGE_RESULT *result = &params->results[range_index];
            NV_STATUS range_status = status == NV_OK ? NV_OK : status;

            result->status = range_status;
            if (range_status == NV_OK) {
                result->resident_gpu_bytes = range_states[range_index].committed_pages * PAGE_SIZE;
                result->resident_cpu_bytes = range_states[range_index].committed_pages * PAGE_SIZE;
                result->resident_any_bytes = range_states[range_index].covered_bytes;
                result->resident_unknown_bytes =
                    params->ranges[range_index].length > range_states[range_index].covered_bytes ?
                        params->ranges[range_index].length - range_states[range_index].covered_bytes : 0;
            }

            uvm_phase_f_note_status(range_status,
                                    &params->success_count,
                                    &params->failure_count,
                                    &first_error);
        }
    }

    uvm_phase_f_route_log("phase_f_expert_read_dup_fast_batch count=%u entries=%zu status=0x%x gpu=%s async=%u committed_pages=%llu\n",
                          params->count,
                          entry_count,
                          status,
                          gpu ? gpu->name : "none",
                          async,
                          committed_pages_total);

    uvm_tracker_deinit(&batch_tracker);

    if (involved_processors)
        uvm_processor_mask_cache_free(involved_processors);
    uvm_kvfree(entry_range_indices);
    uvm_kvfree(entries);

    uvm_va_space_up_read(va_space);
    if (mmap_lock_dropped)
        uvm_va_space_mm_or_current_release(va_space, mm);
    else
        uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    uvm_kvfree(range_states);

    return status == NV_WARN_MORE_PROCESSING_REQUIRED ? status : first_error;
}

static NV_STATUS uvm_phase_f_expert_read_dup_fast_batch_regions(const NvProcessorUuid *gpu_uuid,
                                                                NvBool async,
                                                                UVM_PHASE_F_BATCH_REGIONS_PARAMS *params,
                                                                struct file *filp)
{
    NV_STATUS status = NV_OK;
    NV_STATUS first_error = NV_OK;
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    struct mm_struct *mm = NULL;
    uvm_gpu_t *gpu = NULL;
    uvm_phase_f_expert_batch_region_work_t *work = NULL;
    uvm_processor_mask_t *involved_processors = NULL;
    uvm_tracker_t batch_tracker = UVM_TRACKER_INIT();
    size_t entry_count = 0;
    size_t entry_index;
    NvU32 range_index;
    NvU64 committed_pages_total = 0;
    bool mmap_lock_dropped = false;
    bool single_va_range_fast = false;

    UVM_ASSERT(params->action == UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST);
    UVM_ASSERT(params->count > 1);

    work = uvm_kvmalloc_zero(sizeof(*work));
    if (!work)
        return NV_ERR_NO_MEMORY;

    for (range_index = 0; range_index < params->count; ++range_index)
        uvm_phase_f_result_init(&params->results[range_index], 0);

    involved_processors = uvm_processor_mask_cache_alloc();
    if (!involved_processors) {
        uvm_kvfree(work);
        return NV_ERR_NO_MEMORY;
    }

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_read(va_space);

    gpu = uvm_va_space_get_gpu_by_uuid(va_space, gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto out;
    }

    if (false) {
        NV_STATUS fast_status = NV_OK;

        for (range_index = 0; range_index < params->count; ++range_index) {
            fast_status = uvm_phase_f_expert_batch_probe_single_va_range_region(va_space,
                                                                                params->region_ids[range_index],
                                                                                range_index,
                                                                                work);
            if (fast_status != NV_OK)
                break;
        }

        if (fast_status == NV_OK) {
            for (range_index = 0; range_index < params->count; ++range_index) {
                fast_status = uvm_phase_f_expert_batch_mark_single_va_range_region(va_space,
                                                                                   params->region_ids[range_index],
                                                                                   range_index,
                                                                                   work);
                if (fast_status != NV_OK)
                    break;
            }
        }

        if (fast_status == NV_OK)
            fast_status = uvm_phase_f_expert_batch_emit_single_va_range_entries(work, params, &entry_count);

        if (fast_status == NV_OK) {
            single_va_range_fast = true;
            for (range_index = 0; range_index < params->count; ++range_index)
                params->results[range_index].queried_bytes = work->range_states[range_index].covered_bytes;
        }
        else {
            memset(work, 0, sizeof(*work));
            entry_count = 0;
        }
    }

    if (!single_va_range_fast) {
        for (range_index = 0; range_index < params->count; ++range_index) {
            status = uvm_phase_f_expert_batch_collect_region(va_space,
                                                             params->region_ids[range_index],
                                                             range_index,
                                                             &work->range_states[range_index],
                                                             work->entries,
                                                             work->entry_range_indices,
                                                             &entry_count);
            if (status != NV_OK)
                goto out;

            params->results[range_index].queried_bytes = work->range_states[range_index].covered_bytes;
        }
    }

    if (entry_count == 0) {
        status = NV_WARN_MORE_PROCESSING_REQUIRED;
        goto out;
    }

    if (!single_va_range_fast) {
        uvm_phase_f_expert_batch_sort_entries(work->entries, work->entry_range_indices, entry_count);
        if (uvm_phase_f_expert_batch_has_duplicate_blocks(work->entries, entry_count)) {
            status = NV_WARN_MORE_PROCESSING_REQUIRED;
            goto out;
        }
    }

    if (single_va_range_fast) {
        status = uvm_va_block_expert_promote_read_duplicate_batch_single_lock(va_space,
                                                                              mm,
                                                                              gpu,
                                                                              work->entries,
                                                                              entry_count,
                                                                              &batch_tracker,
                                                                              involved_processors);
    }
    else if (false) {
        status = uvm_va_block_expert_promote_read_duplicate_batch_trusted(va_space,
                                                                          mm,
                                                                          gpu,
                                                                          work->entries,
                                                                          entry_count,
                                                                          &batch_tracker,
                                                                          involved_processors);
    }
    else {
        status = uvm_va_block_expert_promote_read_duplicate_batch(va_space,
                                                                  mm,
                                                                  gpu,
                                                                  work->entries,
                                                                  entry_count,
                                                                  &batch_tracker,
                                                                  involved_processors);
    }
    if (status != NV_OK)
        goto out;

    if (!async) {
        if (mm) {
            uvm_up_read_mmap_lock_out_of_order(mm);
            mmap_lock_dropped = true;
        }

        status = uvm_tracker_wait(&batch_tracker);
        if (status != NV_OK)
            goto out;
    }

    for (entry_index = 0; entry_index < entry_count; ++entry_index) {
        NvU32 owner = work->entry_range_indices[entry_index];

        work->range_states[owner].committed_pages += work->entries[entry_index].committed_pages;
        committed_pages_total += work->entries[entry_index].committed_pages;
    }

out:
    if (status != NV_WARN_MORE_PROCESSING_REQUIRED) {
        params->success_count = 0;
        params->failure_count = 0;

        for (range_index = 0; range_index < params->count; ++range_index) {
            UVM_PHASE_F_RANGE_RESULT *result = &params->results[range_index];
            NV_STATUS range_status = status == NV_OK ? NV_OK : status;

            result->status = range_status;
            if (range_status == NV_OK) {
                result->resident_gpu_bytes = work->range_states[range_index].committed_pages * PAGE_SIZE;
                result->resident_cpu_bytes = work->range_states[range_index].committed_pages * PAGE_SIZE;
                result->resident_any_bytes = work->range_states[range_index].covered_bytes;
                result->resident_unknown_bytes = 0;
            }

            uvm_phase_f_note_status(range_status,
                                    &params->success_count,
                                    &params->failure_count,
                                    &first_error);
        }
    }

    uvm_phase_f_route_log("phase_f_expert_read_dup_fast_batch_regions count=%u entries=%zu status=0x%x gpu=%s async=%u single_va_range=%u committed_pages=%llu\n",
                          params->count,
                          entry_count,
                          status,
                          gpu ? gpu->name : "none",
                          async,
                          single_va_range_fast,
                          committed_pages_total);

    uvm_tracker_deinit(&batch_tracker);

    uvm_va_space_up_read(va_space);
    if (mmap_lock_dropped)
        uvm_va_space_mm_or_current_release(va_space, mm);
    else
        uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    uvm_processor_mask_cache_free(involved_processors);
    uvm_kvfree(work);

    return status == NV_WARN_MORE_PROCESSING_REQUIRED ? status : first_error;
}

static NV_STATUS uvm_api_phase_f_batch_executor(UVM_PHASE_F_BATCH_EXECUTOR_PARAMS *params, struct file *filp)
{
    NvU32 i;
    NV_STATUS first_error = NV_OK;

    if (params->count > UVM_PHASE_F_MAX_RANGES)
        return NV_ERR_INVALID_ARGUMENT;

    params->success_count = 0;
    params->failure_count = 0;

    UVM_INFO_PRINT("phase_f_policy_batch begin action=%u count=%u async=%u first_base=0x%llx first_len=%llu\n",
                   params->action,
                   params->count,
                   params->async,
                   params->count > 0 ? params->ranges[0].base : 0,
                   params->count > 0 ? params->ranges[0].length : 0);
    uvm_phase_f_route_log("phase_f_policy_batch begin action=%u count=%u async=%u first_base=0x%llx first_len=%llu\n",
                          params->action,
                          params->count,
                          params->async,
                          params->count > 0 ? params->ranges[0].base : 0,
                          params->count > 0 ? params->ranges[0].length : 0);

    if (params->action == UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST && params->count > 1) {
        NV_STATUS batch_status = uvm_phase_f_expert_read_dup_fast_batch(&params->gpu_uuid,
                                                                        params->async,
                                                                        params,
                                                                        filp);
        if (batch_status != NV_WARN_MORE_PROCESSING_REQUIRED) {
            first_error = batch_status;
            goto out;
        }

        params->success_count = 0;
        params->failure_count = 0;
        first_error = NV_OK;
        uvm_phase_f_route_log_always("phase_f_policy_batch expert_fast_batch_fallback count=%u\n",
                                     params->count);
    }

    for (i = 0; i < params->count; ++i) {
        NV_STATUS status = uvm_phase_f_execute_range(params->action,
                                                     &params->gpu_uuid,
                                                     params->async,
                                                     params->ranges[i].base,
                                                     params->ranges[i].length,
                                                     &params->results[i],
                                                     filp);
        uvm_phase_f_note_status(status,
                                &params->success_count,
                                &params->failure_count,
                                &first_error);
    }

out:
    UVM_INFO_PRINT("phase_f_policy_batch end action=%u count=%u success=%u failure=%u first_error=0x%x\n",
                   params->action,
                   params->count,
                   params->success_count,
                   params->failure_count,
                   first_error);
    uvm_phase_f_route_log("phase_f_policy_batch end action=%u count=%u success=%u failure=%u first_error=0x%x\n",
                          params->action,
                          params->count,
                          params->success_count,
                          params->failure_count,
                          first_error);

    return first_error;
}

static NV_STATUS uvm_api_phase_f_register_region(UVM_PHASE_F_REGISTER_REGION_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_phase_f_region_t *region;
    NV_STATUS status;
    int ret;

    if (params->region.region_id == 0 ||
        params->region.base == 0 ||
        params->region.length == 0 ||
        params->region.base + params->region.length - 1 < params->region.base)
        return NV_ERR_INVALID_ARGUMENT;

    region = uvm_kvmalloc_zero(sizeof(*region));
    if (!region)
        return NV_ERR_NO_MEMORY;

    region->region_id = params->region.region_id;
    region->base = params->region.base;
    region->length = params->region.length;
    region->state = UVM_REGION_STATE_UNKNOWN;
    region->va_node.start = region->base;
    region->va_node.end = region->base + region->length - 1;

    uvm_va_space_down_write(va_space);
    status = uvm_range_tree_add(&va_space->phase_f_region_ranges, &region->va_node);
    if (status == NV_OK) {
        ret = radix_tree_insert(&va_space->phase_f_regions, region->region_id, region);
        status = errno_to_nv_status(ret);
        if (status != NV_OK)
            uvm_range_tree_remove(&va_space->phase_f_region_ranges, &region->va_node);
    }
    uvm_va_space_up_write(va_space);

    if (status != NV_OK) {
        uvm_kvfree(region);
        return status;
    }

    uvm_phase_f_route_log("phase_f_register_region id=%llu base=0x%llx length=%llu state=%u\n",
                          region->region_id,
                          region->base,
                          region->length,
                          region->state);

    return NV_OK;
}

static NV_STATUS uvm_api_phase_f_unregister_region(UVM_PHASE_F_UNREGISTER_REGION_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_phase_f_region_t *region;

    if (params->region_id == 0)
        return NV_ERR_INVALID_ARGUMENT;

    uvm_va_space_down_write(va_space);
    region = radix_tree_delete(&va_space->phase_f_regions, params->region_id);
    if (region) {
        uvm_range_tree_remove(&va_space->phase_f_region_ranges, &region->va_node);
    }
    uvm_va_space_up_write(va_space);

    if (!region)
        return NV_ERR_OBJECT_NOT_FOUND;

    uvm_phase_f_route_log("phase_f_unregister_region id=%llu base=0x%llx length=%llu\n",
                          region->region_id,
                          region->base,
                          region->length);
    uvm_kvfree(region);

    return NV_OK;
}

// Batch region execution is the generic restore/release path. Keep the
// residency state machine tied to the operation result, including an explicit
// in-flight state for asynchronous submissions.
static void uvm_phase_f_update_region_state_after_action(uvm_va_space_t *va_space,
                                                          NvU64 region_id,
                                                          NvU32 action,
                                                          NvBool async,
                                                          NV_STATUS status)
{
    NvU32 new_state;
    uvm_phase_f_region_t *region;

    if (action == UVM_PHASE_F_ACTION_PREFETCH_GPU) {
        new_state = status == NV_OK ?
                        (async ? UVM_REGION_STATE_PREFETCHING_GPU :
                                 UVM_REGION_STATE_GPU) :
                        UVM_REGION_STATE_UNKNOWN;
    }
    else if (action == UVM_PHASE_F_ACTION_PREFETCH_CPU ||
             action == UVM_PHASE_F_ACTION_PREFETCH_CPU_NO_MAP) {
        new_state = status == NV_OK ?
                        (async ? UVM_REGION_STATE_PREFETCHING_CPU :
                                 UVM_REGION_STATE_CPU) :
                        UVM_REGION_STATE_UNKNOWN;
    }
    else {
        return;
    }

    uvm_va_space_down_write(va_space);
    region = radix_tree_lookup(&va_space->phase_f_regions, region_id);
    if (region)
        region->state = new_state;
    uvm_va_space_up_write(va_space);
}

static bool uvm_phase_f_region_action_is_idempotent(NvU32 action,
                                                     NvU32 state)
{
    if (action == UVM_PHASE_F_ACTION_PREFETCH_GPU)
        return state == UVM_REGION_STATE_GPU;

    if (action == UVM_PHASE_F_ACTION_PREFETCH_CPU)
        return state == UVM_REGION_STATE_CPU;

    return false;
}

static void uvm_phase_f_result_set_idempotent(UVM_PHASE_F_RANGE_RESULT *result,
                                               NvU32 action,
                                               NvU64 length)
{
    uvm_phase_f_result_init(result, length);
    result->resident_on_count = 1;
    result->resident_any_bytes = length;

    if (action == UVM_PHASE_F_ACTION_PREFETCH_GPU) {
        result->resident_on_gpu = NV_TRUE;
        result->resident_gpu_bytes = length;
    }
    else {
        result->resident_on_cpu = NV_TRUE;
        result->resident_cpu_bytes = length;
    }
}

static NV_STATUS uvm_api_phase_f_batch_regions(UVM_PHASE_F_BATCH_REGIONS_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    NvU32 i;
    NV_STATUS first_error = NV_OK;

    if (params->count > UVM_PHASE_F_MAX_RANGES)
        return NV_ERR_INVALID_ARGUMENT;

    params->success_count = 0;
    params->failure_count = 0;

    uvm_phase_f_route_log("phase_f_policy_batch_regions begin action=%u count=%u async=%u first_region=%llu\n",
                          params->action,
                          params->count,
                          params->async,
                          params->count > 0 ? params->region_ids[0] : 0);

    if (params->action == UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST && params->count > 1) {
        NV_STATUS batch_status = uvm_phase_f_expert_read_dup_fast_batch_regions(&params->gpu_uuid,
                                                                                params->async,
                                                                                params,
                                                                                filp);
        if (batch_status != NV_WARN_MORE_PROCESSING_REQUIRED) {
            first_error = batch_status;
            goto out;
        }

        params->success_count = 0;
        params->failure_count = 0;
        first_error = NV_OK;
        uvm_phase_f_route_log_always("phase_f_policy_batch_regions expert_fast_region_batch_fallback count=%u\n",
                                     params->count);
    }

    for (i = 0; i < params->count; ++i) {
        uvm_phase_f_region_t *region;
        NvU64 base = 0;
        NvU64 length = 0;
        NvU32 state = UVM_REGION_STATE_UNKNOWN;
        bool idempotent = false;
        NV_STATUS status = NV_OK;

        params->results[i].status = NV_OK;
        params->results[i].resident_on_count = 0;
        params->results[i].resident_on_cpu = NV_FALSE;
        params->results[i].resident_on_gpu = NV_FALSE;
        params->results[i].queried_bytes = 0;
        params->results[i].resident_gpu_bytes = 0;
        params->results[i].resident_cpu_bytes = 0;
        params->results[i].resident_any_bytes = 0;
        params->results[i].resident_unknown_bytes = 0;

        if (params->region_ids[i] == 0) {
            status = NV_ERR_INVALID_ARGUMENT;
        }
        else {
            uvm_va_space_down_read(va_space);
            region = radix_tree_lookup(&va_space->phase_f_regions, params->region_ids[i]);
            if (region) {
                base = region->base;
                length = region->length;
                state = region->state;
            }
            uvm_va_space_up_read(va_space);

            if (!region)
                status = NV_ERR_OBJECT_NOT_FOUND;
        }

        if (status == NV_OK) {
            idempotent = uvm_phase_f_region_action_is_idempotent(params->action, state);
            if (idempotent) {
                uvm_phase_f_result_set_idempotent(&params->results[i],
                                                   params->action,
                                                   length);
                goto note_status;
            }

            uvm_phase_f_route_log("phase_f_policy_batch_region_entry index=%u action=%u region=%llu base=0x%llx end=0x%llx length=%llu async=%u\n",
                                  i,
                                  params->action,
                                  params->region_ids[i],
                                  base,
                                  base + length,
                                  length,
                                  params->async);
            status = uvm_phase_f_execute_range(params->action,
                                               &params->gpu_uuid,
                                               params->async,
                                               base,
                                               length,
                                               &params->results[i],
                                               filp);
            uvm_phase_f_update_region_state_after_action(va_space,
                                                         params->region_ids[i],
                                                         params->action,
                                                         params->async,
                                                         status);
        }
        else {
            params->results[i].status = status;
        }

note_status:
        uvm_phase_f_note_status(status,
                                &params->success_count,
                                &params->failure_count,
                                &first_error);
    }

    if (params->action == UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_AUDITED &&
        params->count > 0 &&
        params->success_count == params->count &&
        first_error == NV_OK &&
        READ_ONCE(uvm_phase_f_remote_read_full_tlb_invalidate) != 0) {
        uvm_gpu_t *gpu = uvm_va_space_retain_gpu_by_uuid(va_space, &params->gpu_uuid);
        NV_STATUS invalidate_status;

        if (!gpu) {
            invalidate_status = NV_ERR_INVALID_DEVICE;
        }
        else {
            uvm_va_space_down_read(va_space);
            invalidate_status =
                uvm_moe_slot_pool_user_backing_batch_tlb_invalidate(va_space, gpu);
            uvm_va_space_up_read(va_space);
            uvm_gpu_release(gpu);
        }

        uvm_phase_f_route_log_always("phase_f_remote_read_full_tlb_invalidate count=%u status=0x%x\n",
                                     params->count,
                                     invalidate_status);
        UVM_INFO_PRINT("phase_f_remote_read_full_tlb_invalidate count=%u status=0x%x\n",
                       params->count,
                       invalidate_status);

        if (invalidate_status != NV_OK) {
            for (i = 0; i < params->count; ++i)
                params->results[i].status = invalidate_status;
            params->success_count = 0;
            params->failure_count = params->count;
            first_error = invalidate_status;
        }
    }

out:
    uvm_phase_f_route_log("phase_f_policy_batch_regions end action=%u count=%u success=%u failure=%u first_error=0x%x\n",
                          params->action,
                          params->count,
                          params->success_count,
                          params->failure_count,
                          first_error);

    return first_error;
}

static bool uvm_moe_slot_remap_mode_valid(NvU32 mode)
{
    return mode == UVM_MOE_SLOT_REMAP_MODE_DRY_RUN ||
           mode == UVM_MOE_SLOT_REMAP_MODE_VALIDATE ||
           mode == UVM_MOE_SLOT_REMAP_MODE_REMAP ||
           mode == UVM_MOE_SLOT_REMAP_MODE_COPY_ONLY ||
           mode == UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ONLY ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_PTE_REMAP ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_USER_BACKING_REMAP ||
           mode == UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC ||
           mode == UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ASYNC_SUBMIT ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_PER_BLOCK_PUSH ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_PTE ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_2M_PTE ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_OVERWRITE ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_STAGE_SYNC ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_STAGE_SYNC ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_OVERWRITE ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_OVERWRITE_STAGE_SYNC ||
           mode == UVM_MOE_SLOT_REMAP_MODE_RELEASE ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE;
}

static bool uvm_moe_slot_remap_is_layer_transition_mode(NvU32 mode)
{
    return mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION;
}

static bool uvm_moe_slot_transition_fast_index_enabled(NvU32 mode)
{
    return uvm_moe_slot_remap_is_layer_transition_mode(mode) &&
           READ_ONCE(uvm_moe_slot_transition_fast_index) != 0;
}

static bool uvm_moe_slot_transition_direct_cutover_enabled(NvU32 mode)
{
    return uvm_moe_slot_transition_fast_index_enabled(mode) &&
           READ_ONCE(uvm_moe_slot_transition_direct_cutover) != 0;
}

static bool uvm_moe_slot_remap_is_slot_direct_mode(NvU32 mode)
{
    return mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_PER_BLOCK_PUSH ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_PTE ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_2M_PTE ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_OVERWRITE ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_STAGE_SYNC ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_STAGE_SYNC ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_OVERWRITE ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_OVERWRITE_STAGE_SYNC ||
           mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE ||
           uvm_moe_slot_remap_is_layer_transition_mode(mode);
}

static NvU32 uvm_moe_slot_transition_required_flags(void)
{
    return UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT |
           UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED;
}

static NvU32 uvm_moe_slot_transition_strategy_mask(void)
{
    return UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_SHARED_PUSH |
           UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_64K_PTE;
}

static NvBool uvm_moe_slot_transition_flags_valid(NvU32 flags)
{
    const NvU32 required = uvm_moe_slot_transition_required_flags();

    return (flags & required) == required &&
           !(flags & ~(required | uvm_moe_slot_transition_strategy_mask()));
}

static NvU32 uvm_moe_slot_transition_batch_strategy(
    const UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params)
{
    if (params->count == 0)
        return 0;

    return params->entries[0].flags & uvm_moe_slot_transition_strategy_mask();
}

static NvBool uvm_moe_slot_transition_entry_strategy_valid(
    const UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    NvU32 entry_index)
{
    return uvm_moe_slot_transition_flags_valid(params->entries[entry_index].flags) &&
           (params->entries[entry_index].flags & uvm_moe_slot_transition_strategy_mask()) ==
               uvm_moe_slot_transition_batch_strategy(params);
}

static bool uvm_moe_slot_remap_entry_range_invalid(const UVM_MOE_SLOT_REMAP_ENTRY *entry)
{
    if (entry->va_base == 0 || entry->bytes == 0)
        return true;

    // All downstream users form an exclusive end. Reject ranges whose
    // inclusive end is UINT64_MAX as well as ordinary addition overflow.
    if (entry->va_base + entry->bytes <= entry->va_base)
        return true;

    if (entry->slot_offset + entry->bytes <= entry->slot_offset)
        return true;

    if (!IS_ALIGNED(entry->va_base, PAGE_SIZE) ||
        !IS_ALIGNED(entry->bytes, PAGE_SIZE) ||
        !IS_ALIGNED(entry->slot_offset, PAGE_SIZE))
        return true;

    return false;
}

static bool uvm_moe_slot_release_entry_invalid(const UVM_MOE_SLOT_REMAP_ENTRY *entry)
{
    // A release is keyed only by the scheduler identity. Requiring every
    // unused field to be zero catches accidentally mixed mode-8/mode-10
    // payloads before they can destroy backing.
    return entry->role != 0 ||
           entry->va_base != 0 ||
           entry->bytes != 0 ||
           entry->slot_offset != 0 ||
           entry->flags != 0;
}

static bool uvm_moe_slot_release_has_duplicate(const UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params, NvU32 index)
{
    const UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[index];
    NvU32 i;

    for (i = 0; i < index; ++i) {
        const UVM_MOE_SLOT_REMAP_ENTRY *prev = &params->entries[i];

        if (prev->status != UVM_MOE_SLOT_REMAP_ENTRY_OK)
            continue;

        if (prev->layer_id == entry->layer_id && prev->slot_id == entry->slot_id)
            return true;
    }

    return false;
}

static bool uvm_moe_slot_remap_entry_excluded_from_conflict_check(const UVM_MOE_SLOT_REMAP_ENTRY *entry)
{
    return entry->status == UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ARGUMENT ||
           entry->status == UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ADDRESS ||
           entry->status == UVM_MOE_SLOT_REMAP_ENTRY_SLOT_CONFLICT;
}

static bool uvm_moe_slot_remap_entry_target_ranges_overlap(const UVM_MOE_SLOT_REMAP_ENTRY *a,
                                                           const UVM_MOE_SLOT_REMAP_ENTRY *b)
{
    NvU64 a_end = a->slot_offset + a->bytes;
    NvU64 b_end = b->slot_offset + b->bytes;

    return a->slot_offset < b_end && b->slot_offset < a_end;
}

static bool uvm_moe_slot_remap_has_slot_conflict(UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params, NvU32 index)
{
    UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[index];
    NvU32 i;

    for (i = 0; i < index; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *prev = &params->entries[i];

        if (uvm_moe_slot_remap_entry_excluded_from_conflict_check(prev))
            continue;

        if (prev->layer_id != entry->layer_id || prev->slot_id != entry->slot_id)
            continue;

        if (prev->expert_id != entry->expert_id)
            return true;

        if (uvm_moe_slot_remap_entry_target_ranges_overlap(prev, entry))
            return true;
    }

    return false;
}

static NV_STATUS uvm_moe_slot_pool_uninstall_va_block_slot_mapping(uvm_va_space_t *va_space,
                                                                   uvm_gpu_t *gpu,
                                                                   struct mm_struct *mm,
                                                                   NvU64 va_base,
                                                                   NvU64 bytes,
                                                                   NvU64 *block_unmap_count_out,
                                                                   NvU64 *tracker_wait_count_out);
static NV_STATUS uvm_moe_slot_pool_uninstall_va_block_slot_mapping_mode(
    uvm_va_space_t *va_space,
    uvm_gpu_t *gpu,
    struct mm_struct *mm,
    NvU64 va_base,
    NvU64 bytes,
    NvU64 *block_unmap_count_out,
    NvU64 *tracker_wait_count_out,
    NvBool restore_cpu_mapping);

static void uvm_moe_slot_pool_note_pte_failure_locked(uvm_moe_slot_pool_slot_t *slot)
{
    if (slot)
        ++slot->pte_failure_count;

    ++g_uvm_moe_slot_pool_pte_failure_count;
}

static void uvm_moe_slot_pool_note_pte_clear_reason_locked(uvm_moe_slot_pool_slot_t *slot,
                                                           uvm_moe_slot_pte_clear_reason_t reason)
{
    if (!slot)
        return;

    switch (reason) {
        case UVM_MOE_SLOT_PTE_CLEAR_REASON_FAULT:
            ++slot->pte_fault_clear_count;
            ++g_uvm_moe_slot_pool_pte_fault_clear_count;
            break;
        case UVM_MOE_SLOT_PTE_CLEAR_REASON_MIGRATE:
            ++slot->pte_migrate_clear_count;
            ++g_uvm_moe_slot_pool_pte_migrate_clear_count;
            break;
        case UVM_MOE_SLOT_PTE_CLEAR_REASON_DISCARD:
        case UVM_MOE_SLOT_PTE_CLEAR_REASON_UNMAP:
            ++slot->pte_unmap_clear_count;
            ++g_uvm_moe_slot_pool_pte_unmap_clear_count;
            break;
        case UVM_MOE_SLOT_PTE_CLEAR_REASON_SPLIT:
            ++slot->pte_split_clear_count;
            ++g_uvm_moe_slot_pool_pte_split_clear_count;
            break;
        case UVM_MOE_SLOT_PTE_CLEAR_REASON_RELEASE:
            ++slot->pte_release_clear_count;
            ++g_uvm_moe_slot_pool_pte_release_clear_count;
            break;
        case UVM_MOE_SLOT_PTE_CLEAR_REASON_EVICT:
            ++slot->pte_evict_clear_count;
            ++g_uvm_moe_slot_pool_pte_evict_clear_count;
            break;
        case UVM_MOE_SLOT_PTE_CLEAR_REASON_DANGEROUS_REJECT:
            ++slot->pte_dangerous_op_reject_count;
            ++g_uvm_moe_slot_pool_pte_dangerous_op_reject_count;
            break;
    }
}

static bool uvm_moe_slot_pool_pte_mapping_overlaps(const uvm_moe_slot_pool_slot_t *slot,
                                                   uvm_va_space_t *va_space,
                                                   NvU64 va_base,
                                                   NvU64 bytes)
{
    NvU64 va_end = va_base + bytes;
    NvU32 i;

    if (!slot->valid || slot->pte_va_space != va_space || slot->pte_bytes == 0)
        return false;

    if (va_end <= va_base)
        return false;

    if (!slot->pte_range_vec &&
        !((slot->pte_mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP ||
           slot->pte_mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_USER_BACKING_REMAP) &&
          slot->pte_gpu))
        return false;

    if (slot->pte_range_count == 0) {
        NvU64 slot_end = slot->pte_va_base + slot->pte_bytes;

        return slot_end > slot->pte_va_base &&
               slot->pte_va_base < va_end &&
               va_base < slot_end;
    }

    for (i = 0; i < slot->pte_range_count; ++i) {
        const uvm_moe_slot_pool_pte_range_t *range = &slot->pte_ranges[i];
        NvU64 range_end = range->va_base + range->bytes;

        if (range->bytes == 0 || range_end <= range->va_base)
            continue;

        if (range->va_base < va_end && va_base < range_end)
            return true;
    }

    return false;
}

static NvBool uvm_moe_static_alias_rejects_clear_locked(
    uvm_moe_slot_pool_slot_t *slot,
    uvm_moe_slot_pte_clear_reason_t reason)
{
    if (slot->static_alias_session_id == 0)
        return NV_FALSE;

    // Managed VMA/GPU-VA teardown cannot be refused by this hook: channels
    // are stopped and the page tables are about to be destroyed. Mark that
    // path poison-only and let it converge safely. Serving-time operations
    // which could mutate residency or PTE shape are rejected instead.
    switch (reason) {
        case UVM_MOE_SLOT_PTE_CLEAR_REASON_FAULT:
        case UVM_MOE_SLOT_PTE_CLEAR_REASON_MIGRATE:
        case UVM_MOE_SLOT_PTE_CLEAR_REASON_DISCARD:
        case UVM_MOE_SLOT_PTE_CLEAR_REASON_SPLIT:
        case UVM_MOE_SLOT_PTE_CLEAR_REASON_EVICT:
            uvm_moe_slot_pool_note_pte_clear_reason_locked(
                slot,
                UVM_MOE_SLOT_PTE_CLEAR_REASON_DANGEROUS_REJECT);
            return NV_TRUE;
        default:
            slot->static_alias_poisoned = NV_TRUE;
            return NV_FALSE;
    }
}

static NV_STATUS uvm_moe_slot_pool_release_pte_mapping_mode_locked(
    uvm_moe_slot_pool_slot_t *slot,
    uvm_va_space_t *locked_va_space,
    struct mm_struct *mm,
    NvBool restore_cpu_mapping)
{
    NV_STATUS status = NV_OK;
    NvU64 old_pte_va_base = slot->pte_va_base;
    NvU64 old_pte_bytes = slot->pte_bytes;
    NvU64 old_pte_end = old_pte_va_base + old_pte_bytes;
    NvU32 old_pte_range_count = slot->pte_range_count;
    NvU32 old_pte_mode = slot->pte_mode;

    if (old_pte_bytes != 0 || slot->pte_range_vec || old_pte_range_count != 0) {
        uvm_phase_f_route_log("moe_pte_range action=release_begin layer=%u expert=%u slot=%u role=%u va=0x%llx end=0x%llx bytes=%llu ranges=%u mode=%u va_space=%px locked_va_space=%px mm=%px gpu=%s\n",
                              slot->layer_id,
                              slot->owner_expert_id,
                              slot->slot_id,
                              slot->last_role,
                              old_pte_va_base,
                              old_pte_end,
                              old_pte_bytes,
                              old_pte_range_count,
                              old_pte_mode,
                              slot->pte_va_space,
                              locked_va_space,
                              mm,
                              slot->pte_gpu ? uvm_gpu_name(slot->pte_gpu) : "unknown");
    }

    if (slot->pte_range_vec) {
        uvm_membar_t tlb_membar = UVM_MEMBAR_SYS;

        if (slot->pte_gpu && slot->slot_mem)
            tlb_membar = uvm_hal_downgrade_membar_type(slot->pte_gpu,
                                                       uvm_mem_is_local_vidmem(slot->slot_mem, slot->pte_gpu));

        uvm_phase_f_route_log("moe_pte_range action=clear_range_vec_begin layer=%u expert=%u slot=%u role=%u va=0x%llx end=0x%llx bytes=%llu ranges=%u mode=%u membar=%u\n",
                              slot->layer_id,
                              slot->owner_expert_id,
                              slot->slot_id,
                              slot->last_role,
                              slot->pte_va_base,
                              slot->pte_va_base + slot->pte_bytes,
                              slot->pte_bytes,
                              slot->pte_range_count,
                              slot->pte_mode,
                              tlb_membar);
        status = uvm_page_table_range_vec_clear_ptes(slot->pte_range_vec, tlb_membar);
        uvm_phase_f_route_log("moe_pte_range action=clear_range_vec_end layer=%u expert=%u slot=%u role=%u va=0x%llx end=0x%llx bytes=%llu status=0x%x\n",
                              slot->layer_id,
                              slot->owner_expert_id,
                              slot->slot_id,
                              slot->last_role,
                              slot->pte_va_base,
                              slot->pte_va_base + slot->pte_bytes,
                              slot->pte_bytes,
                              status);
        if (status != NV_OK)
            UVM_ERR_PRINT("MoE slot PTE clear failed: %s, GPU %s, VA [0x%llx, 0x%llx)\n",
                          nvstatusToString(status),
                          slot->pte_gpu ? uvm_gpu_name(slot->pte_gpu) : "unknown",
                          slot->pte_va_base,
                          slot->pte_va_base + slot->pte_bytes);

        uvm_page_table_range_vec_destroy(slot->pte_range_vec);
        slot->pte_range_vec = NULL;
        ++slot->pte_clear_count;
        ++g_uvm_moe_slot_pool_pte_clear_count;
    }

    if ((slot->pte_mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP ||
         slot->pte_mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_USER_BACKING_REMAP) &&
        slot->pte_bytes != 0) {
        NvU64 block_unmap_count = 0;
        NvU64 tracker_wait_count = 0;
        NvU32 i;

        if (!locked_va_space || locked_va_space != slot->pte_va_space || !slot->pte_gpu)
            return NV_ERR_INVALID_STATE;

        if (slot->pte_range_count == 0) {
            slot->pte_ranges[0].va_base = slot->pte_va_base;
            slot->pte_ranges[0].bytes = slot->pte_bytes;
            slot->pte_range_count = 1;
        }

        for (i = 0; i < slot->pte_range_count; ++i) {
            uvm_moe_slot_pool_pte_range_t *range = &slot->pte_ranges[i];
            NvU64 range_block_unmap_count = 0;
            NvU64 range_tracker_wait_count = 0;

            if (range->bytes == 0)
                continue;

            uvm_phase_f_route_log("moe_pte_range action=clear_va_block_begin layer=%u expert=%u slot=%u role=%u range_index=%u va=0x%llx end=0x%llx bytes=%llu full_va=0x%llx full_end=0x%llx full_bytes=%llu mode=%u\n",
                                  slot->layer_id,
                                  slot->owner_expert_id,
                                  slot->slot_id,
                                  slot->last_role,
                                  i,
                                  range->va_base,
                                  range->va_base + range->bytes,
                                  range->bytes,
                                  slot->pte_va_base,
                                  slot->pte_va_base + slot->pte_bytes,
                                  slot->pte_bytes,
                                  slot->pte_mode);
            status = uvm_moe_slot_pool_uninstall_va_block_slot_mapping_mode(
                locked_va_space,
                slot->pte_gpu,
                mm,
                range->va_base,
                range->bytes,
                &range_block_unmap_count,
                &range_tracker_wait_count,
                restore_cpu_mapping);
            uvm_phase_f_route_log("moe_pte_range action=clear_va_block_end layer=%u expert=%u slot=%u role=%u range_index=%u va=0x%llx end=0x%llx bytes=%llu status=0x%x block_unmaps=%llu tracker_waits=%llu\n",
                                  slot->layer_id,
                                  slot->owner_expert_id,
                                  slot->slot_id,
                                  slot->last_role,
                                  i,
                                  range->va_base,
                                  range->va_base + range->bytes,
                                  range->bytes,
                                  status,
                                  range_block_unmap_count,
                                  range_tracker_wait_count);
            if (status != NV_OK) {
                UVM_ERR_PRINT("MoE slot VA-block PTE clear failed: %s, GPU %s, VA [0x%llx, 0x%llx)\n",
                              nvstatusToString(status),
                              slot->pte_gpu ? uvm_gpu_name(slot->pte_gpu) : "unknown",
                              range->va_base,
                              range->va_base + range->bytes);
                return status;
            }

            block_unmap_count += range_block_unmap_count;
            tracker_wait_count += range_tracker_wait_count;
        }

        slot->pte_block_unmap_count += block_unmap_count;
        slot->pte_block_tracker_wait_count += tracker_wait_count;
        g_uvm_moe_slot_pool_pte_block_unmap_count += block_unmap_count;
        g_uvm_moe_slot_pool_pte_block_tracker_wait_count += tracker_wait_count;
        ++slot->pte_clear_count;
        ++g_uvm_moe_slot_pool_pte_clear_count;
    }

    if (old_pte_bytes != 0 || old_pte_range_count != 0) {
        uvm_phase_f_route_log("moe_pte_range action=release_end layer=%u expert=%u slot=%u role=%u old_va=0x%llx old_end=0x%llx old_bytes=%llu ranges=%u mode=%u status=0x%x\n",
                              slot->layer_id,
                              slot->owner_expert_id,
                              slot->slot_id,
                              slot->last_role,
                              old_pte_va_base,
                              old_pte_end,
                              old_pte_bytes,
                              old_pte_range_count,
                              old_pte_mode,
                              status);
    }

    if (slot->pte_gpu) {
        uvm_gpu_release(slot->pte_gpu);
        slot->pte_gpu = NULL;
    }

    slot->pte_va_base = 0;
    slot->pte_bytes = 0;
    slot->pte_range_count = 0;
    memset(slot->pte_ranges, 0, sizeof(slot->pte_ranges));
    slot->pte_va_space = NULL;
    slot->pte_slot_offset = 0;
    slot->pte_mode = 0;
    slot->pte_va_block_tracked = NV_FALSE;

    return status;
}

static NV_STATUS uvm_moe_slot_pool_release_pte_mapping_locked(
    uvm_moe_slot_pool_slot_t *slot,
    uvm_va_space_t *locked_va_space,
    struct mm_struct *mm)
{
    return uvm_moe_slot_pool_release_pte_mapping_mode_locked(slot,
                                                              locked_va_space,
                                                              mm,
                                                              NV_TRUE);
}

static void uvm_moe_slot_pool_forget_release_pte_mapping_locked(uvm_moe_slot_pool_slot_t *slot)
{
    if (slot->pte_gpu) {
        uvm_gpu_release(slot->pte_gpu);
        slot->pte_gpu = NULL;
    }

    slot->pte_range_vec = NULL;
    slot->pte_va_base = 0;
    slot->pte_bytes = 0;
    slot->pte_range_count = 0;
    memset(slot->pte_ranges, 0, sizeof(slot->pte_ranges));
    slot->pte_va_space = NULL;
    slot->pte_slot_offset = 0;
    slot->pte_mode = 0;
    slot->pte_va_block_tracked = NV_FALSE;
}

static NV_STATUS uvm_moe_slot_pool_rollback_pte_mapping_mode_locked(
    uvm_moe_slot_pool_slot_t *slot,
    uvm_va_space_t *locked_va_space,
    struct mm_struct *mm,
    NvBool restore_cpu_mapping)
{
    NvU64 rollback_bytes = slot->pte_bytes;
    NV_STATUS status;

    if (slot->async_copy_pending)
        (void)uvm_moe_slot_pool_finish_async_copy_locked(slot, NV_TRUE, NULL);

    status = uvm_moe_slot_pool_release_pte_mapping_mode_locked(slot,
                                                                locked_va_space,
                                                                mm,
                                                                restore_cpu_mapping);
    if (status != NV_OK)
        return status;

    ++slot->pte_rollback_count;
    slot->pte_rollback_bytes += rollback_bytes;
    ++g_uvm_moe_slot_pool_pte_rollback_count;
    g_uvm_moe_slot_pool_pte_rollback_bytes += rollback_bytes;
    slot->pte_status = UVM_MOE_SLOT_REMAP_ENTRY_ROLLED_BACK;

    return NV_OK;
}

static NV_STATUS uvm_moe_slot_pool_rollback_pte_mapping_locked(
    uvm_moe_slot_pool_slot_t *slot,
    uvm_va_space_t *locked_va_space,
    struct mm_struct *mm)
{
    return uvm_moe_slot_pool_rollback_pte_mapping_mode_locked(slot,
                                                               locked_va_space,
                                                               mm,
                                                               NV_TRUE);
}

static void uvm_moe_slot_pool_note_partial_rollback_locked(uvm_moe_slot_pool_slot_t *slot,
                                                           NvU64 rollback_bytes,
                                                           NvU64 block_unmap_count,
                                                           NvU64 tracker_wait_count)
{
    if (rollback_bytes == 0)
        return;

    ++slot->pte_rollback_count;
    slot->pte_rollback_bytes += rollback_bytes;
    ++slot->pte_clear_count;
    slot->pte_block_unmap_count += block_unmap_count;
    slot->pte_block_tracker_wait_count += tracker_wait_count;
    ++g_uvm_moe_slot_pool_pte_rollback_count;
    g_uvm_moe_slot_pool_pte_rollback_bytes += rollback_bytes;
    ++g_uvm_moe_slot_pool_pte_clear_count;
    g_uvm_moe_slot_pool_pte_block_unmap_count += block_unmap_count;
    g_uvm_moe_slot_pool_pte_block_tracker_wait_count += tracker_wait_count;
}

static void uvm_moe_slot_pool_bump_backing_epoch_locked(void)
{
    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_pool_lock));

    ++g_uvm_moe_slot_pool_backing_epoch;
    if (g_uvm_moe_slot_pool_backing_epoch == 0)
        ++g_uvm_moe_slot_pool_backing_epoch;
}

static NV_STATUS uvm_moe_slot_pool_release_backing_mode_locked(uvm_moe_slot_pool_slot_t *slot,
                                                               uvm_va_space_t *locked_va_space,
                                                               struct mm_struct *mm,
                                                               NvBool restore_cpu_mapping)
{
    NV_STATUS status;
    NvBool had_backing = slot->slot_mem != NULL || slot->slot_bytes != 0 ||
                         slot->backing_kind != UVM_MOE_SLOT_POOL_BACKING_NONE;

    if (slot->async_copy_pending) {
        status = uvm_moe_slot_pool_finish_async_copy_locked(slot, NV_TRUE, NULL);
        if (status != NV_OK)
            return status;
    }

    status = uvm_moe_slot_pool_release_pte_mapping_mode_locked(slot,
                                                                locked_va_space,
                                                                mm,
                                                                restore_cpu_mapping);
    if (status != NV_OK)
        return status;

    if (slot->slot_mem)
        uvm_mem_free(slot->slot_mem);

    slot->slot_mem = NULL;
    slot->slot_bytes = 0;
    slot->backing_generation = 0;
    slot->backing_kind = UVM_MOE_SLOT_POOL_BACKING_NONE;
    slot->copy_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
    slot->pte_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;

    if (slot->slot_gpu) {
        uvm_gpu_release(slot->slot_gpu);
        slot->slot_gpu = NULL;
    }

    if (had_backing)
        uvm_moe_slot_pool_bump_backing_epoch_locked();

    return NV_OK;
}

static NV_STATUS uvm_moe_slot_pool_release_backing_locked(uvm_moe_slot_pool_slot_t *slot,
                                                          uvm_va_space_t *locked_va_space,
                                                          struct mm_struct *mm)
{
    return uvm_moe_slot_pool_release_backing_mode_locked(slot,
                                                          locked_va_space,
                                                          mm,
                                                          NV_TRUE);
}

static NvBool uvm_moe_slot_pool_slot_has_vidmem_backing(const uvm_moe_slot_pool_slot_t *slot,
                                                        const uvm_gpu_t *gpu)
{
    return slot->slot_mem &&
           slot->slot_gpu == gpu &&
           slot->backing_kind == UVM_MOE_SLOT_POOL_BACKING_VIDMEM;
}

static NvBool uvm_moe_slot_pool_slot_has_active_pte(const uvm_moe_slot_pool_slot_t *slot)
{
    return slot->pte_range_vec || slot->pte_bytes != 0;
}

static NV_STATUS uvm_moe_slot_pool_finish_async_copy_locked(uvm_moe_slot_pool_slot_t *slot,
                                                            NvBool wait_for_completion,
                                                            NvBool *completed_out)
{
    NV_STATUS status;

    if (completed_out)
        *completed_out = NV_TRUE;

    if (!slot->async_copy_pending)
        return NV_OK;

    status = wait_for_completion ?
                 uvm_tracker_wait(&slot->async_copy_tracker) :
                 uvm_tracker_query(&slot->async_copy_tracker);
    if (!wait_for_completion && status == NV_WARN_MORE_PROCESSING_REQUIRED) {
        if (completed_out)
            *completed_out = NV_FALSE;
        return NV_OK;
    }

    uvm_tracker_deinit(&slot->async_copy_tracker);

    if (slot->async_copy_staging) {
        uvm_mem_free(slot->async_copy_staging);
        slot->async_copy_staging = NULL;
    }

    slot->async_copy_pending = NV_FALSE;

    if (status == NV_OK) {
        slot->copy_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
        slot->copy_count++;
        slot->copy_bytes += slot->async_copy_bytes;
        g_uvm_moe_slot_pool_copy_count++;
        g_uvm_moe_slot_pool_copy_bytes += slot->async_copy_bytes;
        slot->state = UVM_MOE_SLOT_POOL_STATE_READY;
    }
    else {
        slot->copy_status = UVM_MOE_SLOT_REMAP_ENTRY_COPY_FAILED;
        slot->copy_failed_count++;
        g_uvm_moe_slot_pool_copy_failed_count++;
        slot->state = UVM_MOE_SLOT_POOL_STATE_ERROR;
    }

    slot->async_copy_bytes = 0;
    slot->async_copy_slot_offset = 0;

    return status;
}

static NvU64 uvm_moe_slot_pool_vidmem_bytes_locked(const uvm_gpu_t *gpu,
                                                   const uvm_moe_slot_pool_slot_t *skip)
{
    NvU64 bytes = 0;
    NvU32 i;

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        const uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];

        if (slot == skip)
            continue;

        if (uvm_moe_slot_pool_slot_has_vidmem_backing(slot, gpu))
            bytes += slot->slot_bytes;
    }

    return bytes;
}

static NvBool uvm_moe_slot_pool_slot_can_evict_locked(const uvm_moe_slot_pool_slot_t *slot,
                                                      uvm_va_space_t *locked_va_space,
                                                      NvBool allow_active)
{
    if (slot->flags & UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT)
        return NV_FALSE;

    if (!uvm_moe_slot_pool_slot_has_active_pte(slot))
        return NV_TRUE;

    if (!allow_active || !locked_va_space || slot->pte_va_space != locked_va_space)
        return NV_FALSE;

    return NV_TRUE;
}

static uvm_moe_slot_pool_slot_t *uvm_moe_slot_pool_oldest_vidmem_slot_locked(const uvm_gpu_t *gpu,
                                                                             const uvm_moe_slot_pool_slot_t *skip,
                                                                             uvm_va_space_t *locked_va_space,
                                                                             NvBool allow_active,
                                                                             NvBool *blocked_active_out)
{
    uvm_moe_slot_pool_slot_t *victim = NULL;
    NvU32 i;

    if (blocked_active_out)
        *blocked_active_out = NV_FALSE;

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];

        if (slot == skip)
            continue;

        if (!uvm_moe_slot_pool_slot_has_vidmem_backing(slot, gpu))
            continue;

        if (!uvm_moe_slot_pool_slot_can_evict_locked(slot, locked_va_space, allow_active)) {
            if (blocked_active_out && uvm_moe_slot_pool_slot_has_active_pte(slot))
                *blocked_active_out = NV_TRUE;
            continue;
        }

        if (!victim ||
            slot->epoch < victim->epoch ||
            (slot->epoch == victim->epoch && slot->batch_id < victim->batch_id))
            victim = slot;
    }

    return victim;
}

static void uvm_moe_slot_pool_note_blocked_active_eviction_locked(const uvm_gpu_t *gpu,
                                                                  const uvm_moe_slot_pool_slot_t *skip,
                                                                  uvm_va_space_t *locked_va_space)
{
    NvU32 i;

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];

        if (slot == skip)
            continue;

        if (!uvm_moe_slot_pool_slot_has_vidmem_backing(slot, gpu))
            continue;

        if (!uvm_moe_slot_pool_slot_has_active_pte(slot))
            continue;

        if (locked_va_space && slot->pte_va_space == locked_va_space)
            continue;

        uvm_moe_slot_pool_note_pte_clear_reason_locked(slot, UVM_MOE_SLOT_PTE_CLEAR_REASON_DANGEROUS_REJECT);
    }
}

static NV_STATUS uvm_moe_slot_pool_evict_slot_locked(uvm_moe_slot_pool_slot_t *slot,
                                                     uvm_va_space_t *locked_va_space,
                                                     struct mm_struct *mm)
{
    NvBool had_active_pte = uvm_moe_slot_pool_slot_has_active_pte(slot);
    NV_STATUS status = uvm_moe_slot_pool_release_backing_locked(slot, locked_va_space, mm);

    if (status != NV_OK)
        return status;

    if (had_active_pte)
        uvm_moe_slot_pool_note_pte_clear_reason_locked(slot, UVM_MOE_SLOT_PTE_CLEAR_REASON_EVICT);

    memset(slot, 0, sizeof(*slot));
    ++g_uvm_moe_slot_pool_vidmem_evict_count;
    return NV_OK;
}

static NvU64 uvm_moe_slot_pool_enforce_vidmem_budget_locked(uvm_gpu_t *gpu,
                                                            const uvm_moe_slot_pool_slot_t *skip,
                                                            NvU64 allocation_bytes,
                                                            NvBool force_one,
                                                            uvm_va_space_t *locked_va_space,
                                                            struct mm_struct *mm)
{
    NvU64 budget = (NvU64)uvm_moe_slot_pool_vidmem_budget_bytes;
    NvU64 bytes;
    NvU64 evicted = 0;

    if (!gpu)
        return 0;

    bytes = uvm_moe_slot_pool_vidmem_bytes_locked(gpu, skip);

    while ((budget != 0 && bytes + allocation_bytes > budget) || (force_one && evicted == 0)) {
        NvBool blocked_active = NV_FALSE;
        uvm_moe_slot_pool_slot_t *victim = uvm_moe_slot_pool_oldest_vidmem_slot_locked(gpu,
                                                                                      skip,
                                                                                      locked_va_space,
                                                                                      NV_FALSE,
                                                                                      &blocked_active);
        NvU64 victim_bytes;

        if (!victim)
            victim = uvm_moe_slot_pool_oldest_vidmem_slot_locked(gpu,
                                                                 skip,
                                                                 locked_va_space,
                                                                 NV_TRUE,
                                                                 &blocked_active);

        if (!victim) {
            if (blocked_active)
                uvm_moe_slot_pool_note_blocked_active_eviction_locked(gpu, skip, locked_va_space);
            break;
        }

        victim_bytes = victim->slot_bytes;
        uvm_phase_f_route_log("moe_slot_pool_vidmem_evict layer=%u slot=%u expert=%u epoch=%llu bytes=%llu budget=%llu current=%llu pending=%llu force=%u active_pte=%u\n",
                              victim->layer_id,
                              victim->slot_id,
                              victim->owner_expert_id,
                              victim->epoch,
                              victim_bytes,
                              budget,
                              bytes,
                              allocation_bytes,
                              force_one,
                              uvm_moe_slot_pool_slot_has_active_pte(victim));

        if (uvm_moe_slot_pool_evict_slot_locked(victim, locked_va_space, mm) != NV_OK)
            break;

        bytes = (bytes > victim_bytes) ? bytes - victim_bytes : 0;
        ++evicted;
    }

    return evicted;
}

static NvBool uvm_moe_slot_pool_vidmem_budget_allows_locked(const uvm_gpu_t *gpu,
                                                            const uvm_moe_slot_pool_slot_t *skip,
                                                            NvU64 allocation_bytes)
{
    NvU64 budget = (NvU64)uvm_moe_slot_pool_vidmem_budget_bytes;
    NvU64 bytes;

    if (budget == 0)
        return NV_TRUE;

    bytes = uvm_moe_slot_pool_vidmem_bytes_locked(gpu, skip);
    return bytes <= budget && allocation_bytes <= budget - bytes;
}

static NV_STATUS uvm_moe_slot_pool_clear_metadata_keep_backing_locked(uvm_moe_slot_pool_slot_t *slot,
                                                                      uvm_va_space_t *locked_va_space,
                                                                      struct mm_struct *mm)
{
    uvm_mem_t *slot_mem = slot->slot_mem;
    uvm_gpu_t *slot_gpu = slot->slot_gpu;
    NvU64 slot_bytes = slot->slot_bytes;
    NvU64 backing_generation = slot->backing_generation;
    NvU32 backing_kind = slot->backing_kind;
    NvBool had_async_copy = slot->async_copy_pending;
    uvm_va_space_t *owner_va_space = slot->owner_va_space;
    NV_STATUS status;

    if (slot->async_copy_pending) {
        status = uvm_moe_slot_pool_finish_async_copy_locked(slot, NV_TRUE, NULL);
        if (status != NV_OK)
            return status;
    }

    status = uvm_moe_slot_pool_release_pte_mapping_locked(slot, locked_va_space, mm);
    if (status != NV_OK)
        return status;

    memset(slot, 0, sizeof(*slot));
    slot->slot_mem = slot_mem;
    slot->slot_gpu = slot_gpu;
    slot->slot_bytes = slot_bytes;
    slot->backing_generation = backing_generation;
    slot->backing_kind = backing_kind;
    slot->owner_va_space = owner_va_space;
    slot->copy_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
    slot->pte_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
    if (had_async_copy)
        slot->state = UVM_MOE_SLOT_POOL_STATE_READY;

    if (slot_mem && slot_bytes && backing_kind == UVM_MOE_SLOT_POOL_BACKING_SYSMEM)
        memset(uvm_mem_get_cpu_addr_kernel(slot_mem), 0, slot_bytes);

    return NV_OK;
}

static NV_STATUS uvm_moe_slot_pool_reset_locked(uvm_va_space_t *locked_va_space, struct mm_struct *mm)
{
    NvU32 i;

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        NV_STATUS status = uvm_moe_slot_pool_release_backing_locked(&g_uvm_moe_slot_pool[i],
                                                                    locked_va_space,
                                                                    mm);
        if (status != NV_OK)
            return status;
    }

    memset(g_uvm_moe_slot_pool, 0, sizeof(g_uvm_moe_slot_pool));
    g_uvm_moe_slot_pool_current_epoch = 0;
    g_uvm_moe_slot_pool_current_batch_id = 0;
    g_uvm_moe_slot_pool_validated_entries = 0;
    g_uvm_moe_slot_pool_overwrite_count = 0;
    g_uvm_moe_slot_pool_range_update_count = 0;
    g_uvm_moe_slot_pool_full_count = 0;
    g_uvm_moe_slot_pool_copy_count = 0;
    g_uvm_moe_slot_pool_copy_bytes = 0;
    g_uvm_moe_slot_pool_copy_failed_count = 0;
    g_uvm_moe_slot_pool_vidmem_alloc_count = 0;
    g_uvm_moe_slot_pool_vidmem_evict_count = 0;
    g_uvm_moe_slot_pool_vidmem_rebind_count = 0;
    g_uvm_moe_slot_pool_vidmem_rebind_bytes = 0;
    g_uvm_moe_slot_pool_pte_remap_count = 0;
    g_uvm_moe_slot_pool_pte_remap_bytes = 0;
    g_uvm_moe_slot_pool_pte_clear_count = 0;
    g_uvm_moe_slot_pool_pte_block_unmap_count = 0;
    g_uvm_moe_slot_pool_pte_block_tracker_wait_count = 0;
    g_uvm_moe_slot_pool_pte_rollback_count = 0;
    g_uvm_moe_slot_pool_pte_rollback_bytes = 0;
    g_uvm_moe_slot_pool_pte_failure_count = 0;
    g_uvm_moe_slot_pool_pte_fault_clear_count = 0;
    g_uvm_moe_slot_pool_pte_migrate_clear_count = 0;
    g_uvm_moe_slot_pool_pte_unmap_clear_count = 0;
    g_uvm_moe_slot_pool_pte_split_clear_count = 0;
    g_uvm_moe_slot_pool_pte_release_clear_count = 0;
    g_uvm_moe_slot_pool_pte_evict_clear_count = 0;
    g_uvm_moe_slot_pool_pte_dangerous_op_reject_count = 0;
    g_uvm_moe_slot_pool_release_count = 0;
    g_uvm_moe_slot_pool_release_bytes = 0;
    g_uvm_moe_slot_pool_release_failed_count = 0;
    g_uvm_moe_slot_pool_close_reclaim_count = 0;
    g_uvm_moe_slot_pool_close_reclaim_bytes = 0;
    g_uvm_moe_slot_pool_close_reclaim_failed_count = 0;
    g_uvm_moe_slot_pool_owner_mismatch_count = 0;
    g_uvm_moe_slot_pool_no_evict_budget_reject_count = 0;
    return NV_OK;
}

static void uvm_moe_slot_plan_target_recipe_destroy(
    uvm_moe_slot_plan_target_recipe_t *recipe)
{
    if (!recipe)
        return;

    uvm_kvfree(recipe->groups);
    uvm_kvfree(recipe->slices);
    uvm_kvfree(recipe);
}

static void uvm_moe_slot_plan_validation_recipe_destroy(
    uvm_moe_slot_plan_validation_recipe_t *recipe)
{
    if (!recipe)
        return;

    uvm_kvfree(recipe->identity_keys);
    uvm_kvfree(recipe->slots);
    uvm_kvfree(recipe->entries);
    uvm_kvfree(recipe);
}

static uvm_moe_static_alias_session_t *uvm_moe_static_alias_find_locked(
    uvm_va_space_t *va_space,
    NvU64 session_id)
{
    NvU32 i;

    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_plan_lock));

    if (!va_space || session_id == 0)
        return NULL;

    for (i = 0; i < UVM_MOE_STATIC_ALIAS_MAX_SESSIONS; ++i) {
        uvm_moe_static_alias_session_t *session = &g_uvm_moe_static_alias_sessions[i];

        if (session->valid &&
            session->owner_va_space == va_space &&
            session->session_id == session_id) {
            return session;
        }
    }

    return NULL;
}

static uvm_moe_static_alias_session_t *uvm_moe_static_alias_find_va_space_locked(
    uvm_va_space_t *va_space)
{
    NvU32 i;

    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_plan_lock));

    for (i = 0; i < UVM_MOE_STATIC_ALIAS_MAX_SESSIONS; ++i) {
        uvm_moe_static_alias_session_t *session = &g_uvm_moe_static_alias_sessions[i];

        if (session->valid && session->owner_va_space == va_space)
            return session;
    }

    return NULL;
}

static NvBool uvm_moe_static_alias_active_for_va_space_locked(uvm_va_space_t *va_space)
{
    return uvm_moe_static_alias_find_va_space_locked(va_space) != NULL;
}

static void uvm_moe_static_alias_session_free_locked(uvm_moe_static_alias_session_t *session)
{
    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_plan_lock));

    if (!session)
        return;

    uvm_kvfree(session->aliases);
    uvm_kvfree(session->rebind_aliases);
    uvm_kvfree(session->dual_view_block_starts);
    memset(session, 0, sizeof(*session));
}

static uvm_moe_static_alias_session_t *uvm_moe_static_alias_alloc_locked(
    uvm_va_space_t *va_space,
    const NvProcessorUuid *gpu_uuid)
{
    NvU32 i;

    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_plan_lock));

    if (uvm_moe_static_alias_find_va_space_locked(va_space))
        return NULL;

    for (i = 0; i < UVM_MOE_STATIC_ALIAS_MAX_SESSIONS; ++i) {
        uvm_moe_static_alias_session_t *session = &g_uvm_moe_static_alias_sessions[i];

        if (session->valid)
            continue;

        ++g_uvm_moe_static_alias_session_counter;
        if (g_uvm_moe_static_alias_session_counter == 0)
            ++g_uvm_moe_static_alias_session_counter;

        memset(session, 0, sizeof(*session));
        session->valid = NV_TRUE;
        session->session_id = g_uvm_moe_static_alias_session_counter;
        session->topology_generation = 1;
        session->owner_va_space = va_space;
        uvm_uuid_copy(&session->gpu_uuid, gpu_uuid);
        return session;
    }

    return NULL;
}

static void uvm_moe_static_alias_destroy_va_space_registry_locked(uvm_va_space_t *va_space)
{
    NvU32 i;

    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_plan_lock));

    for (i = 0; i < UVM_MOE_STATIC_ALIAS_MAX_SESSIONS; ++i) {
        uvm_moe_static_alias_session_t *session = &g_uvm_moe_static_alias_sessions[i];

        if (session->valid && session->owner_va_space == va_space)
            uvm_moe_static_alias_session_free_locked(session);
    }
}

static void uvm_moe_slot_plan_free_locked(uvm_moe_slot_plan_t *plan)
{
    uvm_va_block_slot_copy_plan_destroy(plan->copy_plan);
    uvm_moe_slot_plan_target_recipe_destroy(plan->target_recipe);
    uvm_moe_slot_plan_validation_recipe_destroy(plan->validation_recipe);
    uvm_kvfree(plan->batch);
    uvm_kvfree(plan->allocation_ids);
    memset(plan, 0, sizeof(*plan));
}

static void uvm_moe_slot_plan_clear_session_locked(NvBool reset_generation_counter)
{
    uvm_kvfree(g_uvm_moe_slot_plan_bank_lanes);
    uvm_kvfree(g_uvm_moe_slot_plan_second_bank_lanes);
    g_uvm_moe_slot_plan_bank_lanes = NULL;
    g_uvm_moe_slot_plan_second_bank_lanes = NULL;
    g_uvm_moe_slot_plan_bank_count = 0;
    g_uvm_moe_slot_plan_registered_bank_count = 0;
    g_uvm_moe_slot_plan_active_bank = NV_U32_MAX;
    memset(g_uvm_moe_slot_plan_bank_resident_plan_ids,
           0,
           sizeof(g_uvm_moe_slot_plan_bank_resident_plan_ids));
    g_uvm_moe_slot_plan_failed_target_plan_id = 0;
    g_uvm_moe_slot_plan_fence_compute_addr = 0;
    g_uvm_moe_slot_plan_fence_completion_addr = 0;
    g_uvm_moe_slot_plan_fence_last_payload = 0;
    memset(&g_uvm_moe_slot_plan_gpu_uuid, 0, sizeof(g_uvm_moe_slot_plan_gpu_uuid));
    g_uvm_moe_slot_plan_owner_va_space = NULL;
    g_uvm_moe_slot_plan_generation = 0;
    g_uvm_moe_slot_plan_invalidation_epoch = 0;
    g_uvm_moe_slot_plan_poisoned = NV_FALSE;
    g_uvm_moe_slot_plan_bootstrap_registered = NV_FALSE;
    g_uvm_moe_slot_plan_registration_sealed = NV_FALSE;
    g_uvm_moe_slot_plan_state = UVM_MOE_SLOT_PLAN_STATE_INACTIVE;
    g_uvm_moe_slot_plan_transition_id = 0;
    if (reset_generation_counter) {
        g_uvm_moe_slot_plan_generation_counter = 0;
        g_uvm_moe_slot_plan_transition_counter = 0;
    }
}

static void uvm_moe_slot_pool_reset(void)
{
    NV_STATUS status;

    mutex_lock(&g_uvm_moe_slot_plan_lock);
    {
        NvU32 i;

        for (i = 0; i < UVM_MOE_SLOT_PLAN_MAX_PLANS; ++i)
            uvm_moe_slot_plan_free_locked(&g_uvm_moe_slot_plans[i]);
        for (i = 0; i < UVM_MOE_STATIC_ALIAS_MAX_SESSIONS; ++i)
            uvm_moe_static_alias_session_free_locked(&g_uvm_moe_static_alias_sessions[i]);
        g_uvm_moe_static_alias_session_counter = 0;
        g_uvm_moe_static_alias_rebind_counter = 0;
        uvm_moe_slot_plan_clear_session_locked(NV_TRUE);
    }
    mutex_unlock(&g_uvm_moe_slot_plan_lock);

    mutex_lock(&g_uvm_moe_slot_pool_lock);
    status = uvm_moe_slot_pool_reset_locked(NULL, NULL);
    mutex_unlock(&g_uvm_moe_slot_pool_lock);

    if (status != NV_OK)
        UVM_ERR_PRINT("MoE slot pool reset failed: %s\n", nvstatusToString(status));
}

static NvU32 uvm_moe_slot_plan_destroy_va_space_locked(uvm_va_space_t *va_space)
{
    NvU32 i;
    NvU32 destroyed = 0;
    NvBool any = NV_FALSE;

    if (!va_space)
        return 0;

    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_plan_lock));

    for (i = 0; i < UVM_MOE_SLOT_PLAN_MAX_PLANS; ++i) {
        uvm_moe_slot_plan_t *plan = &g_uvm_moe_slot_plans[i];

        if (!plan->valid || plan->owner_va_space != va_space)
            continue;

        uvm_moe_slot_plan_free_locked(plan);
        ++destroyed;
    }

    for (i = 0; i < UVM_MOE_SLOT_PLAN_MAX_PLANS; ++i) {
        if (g_uvm_moe_slot_plans[i].valid) {
            any = NV_TRUE;
            break;
        }
    }

    if (!any && g_uvm_moe_slot_plan_owner_va_space == va_space) {
        // Keep the module-lifetime generation counter monotonic so a stale
        // (plan_id, generation) pair cannot become valid in a later session.
        uvm_moe_slot_plan_clear_session_locked(NV_FALSE);
    }

    return destroyed;
}

static void uvm_moe_slot_plan_destroy_va_space(uvm_va_space_t *va_space)
{
    if (!va_space)
        return;

    mutex_lock(&g_uvm_moe_slot_plan_lock);
    (void)uvm_moe_slot_plan_destroy_va_space_locked(va_space);
    uvm_moe_static_alias_destroy_va_space_registry_locked(va_space);
    mutex_unlock(&g_uvm_moe_slot_plan_lock);
}

static void uvm_moe_slot_pool_release_va_space_mappings(uvm_va_space_t *va_space)
{
    struct mm_struct *mm;
    NvU64 reclaimed_count = 0;
    NvU64 reclaimed_bytes = 0;
    NvU64 failed_count = 0;
    NvU32 i;

    if (!va_space)
        return;

    // Drop immutable descriptor caches before taking VA-space/pool locks.
    // Compact execution takes locks in plan -> VA-space -> pool order.
    uvm_moe_slot_plan_destroy_va_space(va_space);

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_write(va_space);
    mutex_lock(&g_uvm_moe_slot_pool_lock);

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];
        NvBool had_active_pte;
        NvU64 slot_bytes;
        NV_STATUS status;

        if (slot->owner_va_space != va_space) {
            // Compatibility cleanup for metadata created before ownership was
            // introduced. Never free backing which belongs to another owner.
            if (slot->pte_va_space == va_space) {
                ++g_uvm_moe_slot_pool_owner_mismatch_count;
                status = uvm_moe_slot_pool_release_pte_mapping_locked(slot, va_space, mm);
                if (status == NV_OK)
                    uvm_moe_slot_pool_note_pte_clear_reason_locked(slot,
                                                                   UVM_MOE_SLOT_PTE_CLEAR_REASON_RELEASE);
                else {
                    uvm_moe_slot_pool_note_pte_failure_locked(slot);
                    uvm_moe_slot_pool_forget_release_pte_mapping_locked(slot);
                }
            }
            continue;
        }

        had_active_pte = uvm_moe_slot_pool_slot_has_active_pte(slot);
        slot_bytes = slot->slot_bytes;
        status = uvm_moe_slot_pool_release_backing_locked(slot, va_space, mm);
        if (status == NV_OK) {
            if (had_active_pte)
                uvm_moe_slot_pool_note_pte_clear_reason_locked(slot,
                                                               UVM_MOE_SLOT_PTE_CLEAR_REASON_RELEASE);
            memset(slot, 0, sizeof(*slot));
            ++reclaimed_count;
            reclaimed_bytes += slot_bytes;
            ++g_uvm_moe_slot_pool_close_reclaim_count;
            g_uvm_moe_slot_pool_close_reclaim_bytes += slot_bytes;
        }
        else {
            uvm_moe_slot_pool_note_pte_failure_locked(slot);
            ++failed_count;
            ++g_uvm_moe_slot_pool_close_reclaim_failed_count;
            UVM_ERR_PRINT("MoE slot VA-space backing cleanup failed: %s, VA space %px, layer %u, slot %u\n",
                          nvstatusToString(status),
                          va_space,
                          slot->layer_id,
                          slot->slot_id);

            // The VA space is about to be destroyed and cannot remain as an
            // owner key. Keep the backing pinned until module reset so a rare
            // teardown failure cannot turn an outstanding PTE into a dangling
            // physical reference or let a recycled VA-space pointer claim it.
            uvm_moe_slot_pool_forget_release_pte_mapping_locked(slot);
            slot->owner_va_space = NULL;
            slot->flags |= UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT;
            slot->state = UVM_MOE_SLOT_POOL_STATE_ERROR;
        }
    }

    uvm_phase_f_route_log("moe_slot_pool_close_reclaim va_space=%px reclaimed=%llu bytes=%llu failed=%llu total_reclaimed=%llu total_bytes=%llu total_failed=%llu owner_mismatches=%llu\n",
                          va_space,
                          reclaimed_count,
                          reclaimed_bytes,
                          failed_count,
                          g_uvm_moe_slot_pool_close_reclaim_count,
                          g_uvm_moe_slot_pool_close_reclaim_bytes,
                          g_uvm_moe_slot_pool_close_reclaim_failed_count,
                          g_uvm_moe_slot_pool_owner_mismatch_count);

    mutex_unlock(&g_uvm_moe_slot_pool_lock);
    uvm_va_space_up_write(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);
}

NV_STATUS uvm_moe_slot_pool_clear_gpu_va_space_mappings_locked(uvm_va_space_t *va_space,
                                                               uvm_gpu_t *gpu,
                                                               struct mm_struct *mm)
{
    NV_STATUS first_error = NV_OK;
    NvU32 i;

    if (!va_space || !gpu)
        return NV_ERR_INVALID_ARGUMENT;

    uvm_assert_rwsem_locked_write(&va_space->lock);
    mutex_lock(&g_uvm_moe_slot_pool_lock);

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];
        NvU64 old_pte_va_base;
        NvU64 old_pte_bytes;
        NV_STATUS status;

        if (slot->pte_va_space != va_space || slot->pte_gpu != gpu || !uvm_moe_slot_pool_slot_has_active_pte(slot))
            continue;

        if (slot->static_alias_session_id != 0)
            slot->static_alias_poisoned = NV_TRUE;

        old_pte_va_base = slot->pte_va_base;
        old_pte_bytes = slot->pte_bytes;
        status = uvm_moe_slot_pool_release_pte_mapping_locked(slot, va_space, mm);
        if (status == NV_OK) {
            uvm_moe_slot_pool_note_pte_clear_reason_locked(slot, UVM_MOE_SLOT_PTE_CLEAR_REASON_RELEASE);
            uvm_phase_f_route_log("moe_slot_pte_clear_gpu_va_space_release layer=%u expert=%u slot=%u gpu=%s old_va=0x%llx old_bytes=%llu\n",
                                  slot->layer_id,
                                  slot->owner_expert_id,
                                  slot->slot_id,
                                  uvm_gpu_name(gpu),
                                  old_pte_va_base,
                                  old_pte_bytes);
        }
        else {
            uvm_moe_slot_pool_note_pte_failure_locked(slot);
            if (first_error == NV_OK)
                first_error = status;
            UVM_ERR_PRINT("MoE slot GPU VA-space PTE cleanup failed: %s, GPU %s, VA space %px, layer %u, slot %u, VA [0x%llx, 0x%llx)\n",
                          nvstatusToString(status),
                          uvm_gpu_name(gpu),
                          va_space,
                          slot->layer_id,
                          slot->slot_id,
                          old_pte_va_base,
                          old_pte_va_base + old_pte_bytes);
            uvm_moe_slot_pool_forget_release_pte_mapping_locked(slot);
        }
    }

    mutex_unlock(&g_uvm_moe_slot_pool_lock);

    return first_error;
}

static NV_STATUS uvm_moe_slot_pool_clear_range_mappings(uvm_va_space_t *va_space,
                                                        NvU64 va_base,
                                                        NvU64 bytes,
                                                        uvm_moe_slot_pte_clear_reason_t reason)
{
    struct mm_struct *mm;
    NV_STATUS first_error = NV_OK;
    NvBool plan_invalidated = NV_FALSE;
    NvU32 i;

    if (!va_space)
        return NV_ERR_INVALID_ARGUMENT;

    if (bytes == 0)
        return NV_OK;

    if (va_base + bytes <= va_base)
        return NV_ERR_INVALID_ARGUMENT;

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_write(va_space);
    if (!uvm_moe_slot_pte_seen_bounds_overlap(va_space, va_base, bytes)) {
        uvm_va_space_up_write(va_space);
        uvm_va_space_mm_or_current_release_unlock(va_space, mm);
        return NV_OK;
    }
    mutex_lock(&g_uvm_moe_slot_pool_lock);

    uvm_phase_f_route_log("moe_pte_range action=clear_guard_scan_begin reason=%u guard_va=0x%llx guard_end=0x%llx guard_bytes=%llu va_space=%px mm=%px\n",
                          reason,
                          va_base,
                          va_base + bytes,
                          bytes,
                          va_space,
                          mm);

    // Preflight the complete scan before clearing anything. This preserves
    // all-or-none behavior when a request intersects a sealed/static alias
    // together with ordinary legacy slot mappings.
    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];

        if (uvm_moe_slot_pool_pte_mapping_overlaps(slot, va_space, va_base, bytes) &&
            uvm_moe_static_alias_rejects_clear_locked(slot, reason)) {
            first_error = NV_ERR_IN_USE;
        }
    }

    if (first_error != NV_OK)
        goto done;

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];
        NvU64 old_pte_va_base;
        NvU64 old_pte_bytes;
        NV_STATUS status;

        if (!uvm_moe_slot_pool_pte_mapping_overlaps(slot, va_space, va_base, bytes))
            continue;

        // A range guard may race with another clearer between its read-side
        // overlap check and this write-side scan. Only invalidate registered
        // plans when this scan still finds a recorded slot mapping. Do so
        // before the first release attempt so even a partially failed release
        // leaves every previously registered plan fail-closed.
        if (!plan_invalidated) {
            uvm_va_space_moe_slot_plan_invalidate(va_space);
            plan_invalidated = NV_TRUE;
        }

        old_pte_va_base = slot->pte_va_base;
        old_pte_bytes = slot->pte_bytes;
        status = uvm_moe_slot_pool_release_pte_mapping_locked(slot, va_space, mm);
        if (status == NV_OK) {
            uvm_moe_slot_pool_note_pte_clear_reason_locked(slot, reason);
            uvm_phase_f_route_log("moe_slot_pte_clear_guard reason=%u layer=%u expert=%u slot=%u old_va=0x%llx old_end=0x%llx old_bytes=%llu guard_va=0x%llx guard_end=0x%llx guard_bytes=%llu\n",
                                  reason,
                                  slot->layer_id,
                                  slot->owner_expert_id,
                                  slot->slot_id,
                                  old_pte_va_base,
                                  old_pte_va_base + old_pte_bytes,
                                  old_pte_bytes,
                                  va_base,
                                  va_base + bytes,
                                  bytes);
        }
        else {
            uvm_moe_slot_pool_note_pte_failure_locked(slot);
            if (first_error == NV_OK)
                first_error = status;
            UVM_ERR_PRINT("MoE slot PTE clear guard failed: %s, VA space %px, layer %u, slot %u, VA [0x%llx, 0x%llx)\n",
                          nvstatusToString(status),
                          va_space,
                          slot->layer_id,
                          slot->slot_id,
                          old_pte_va_base,
                          old_pte_va_base + old_pte_bytes);
        }
    }

    uvm_phase_f_route_log("moe_pte_range action=clear_guard_scan_end reason=%u guard_va=0x%llx guard_end=0x%llx guard_bytes=%llu status=0x%x\n",
                          reason,
                          va_base,
                          va_base + bytes,
                          bytes,
                          first_error);

done:
    mutex_unlock(&g_uvm_moe_slot_pool_lock);
    uvm_va_space_up_write(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    return first_error;
}

static NV_STATUS uvm_moe_slot_pool_clear_range_mappings_locked(uvm_va_space_t *va_space,
                                                               struct mm_struct *mm,
                                                               NvU64 va_base,
                                                               NvU64 bytes,
                                                               uvm_moe_slot_pte_clear_reason_t reason)
{
    NV_STATUS first_error = NV_OK;
    NvBool plan_invalidated = NV_FALSE;
    NvU32 i;

    if (!va_space)
        return NV_ERR_INVALID_ARGUMENT;

    if (bytes == 0)
        return NV_OK;

    if (va_base + bytes <= va_base)
        return NV_ERR_INVALID_ARGUMENT;

    uvm_assert_rwsem_locked_write(&va_space->lock);
    if (!uvm_moe_slot_pte_seen_bounds_overlap(va_space, va_base, bytes))
        return NV_OK;

    mutex_lock(&g_uvm_moe_slot_pool_lock);

    uvm_phase_f_route_log("moe_pte_range action=clear_guard_locked_scan_begin reason=%u guard_va=0x%llx guard_end=0x%llx guard_bytes=%llu va_space=%px mm=%px\n",
                          reason,
                          va_base,
                          va_base + bytes,
                          bytes,
                          va_space,
                          mm);

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];

        if (uvm_moe_slot_pool_pte_mapping_overlaps(slot, va_space, va_base, bytes) &&
            uvm_moe_static_alias_rejects_clear_locked(slot, reason)) {
            first_error = NV_ERR_IN_USE;
        }
    }

    if (first_error != NV_OK)
        goto done;

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];
        NvU64 old_pte_va_base;
        NvU64 old_pte_bytes;
        NV_STATUS status;

        if (!uvm_moe_slot_pool_pte_mapping_overlaps(slot, va_space, va_base, bytes))
            continue;

        // See the non-locked wrapper above. The VA-space write lock prevents
        // plan execution from crossing this epoch change, and the pool lock
        // keeps the overlap record stable through the release attempt.
        if (!plan_invalidated) {
            uvm_va_space_moe_slot_plan_invalidate(va_space);
            plan_invalidated = NV_TRUE;
        }

        old_pte_va_base = slot->pte_va_base;
        old_pte_bytes = slot->pte_bytes;
        status = uvm_moe_slot_pool_release_pte_mapping_locked(slot, va_space, mm);
        if (status == NV_OK) {
            uvm_moe_slot_pool_note_pte_clear_reason_locked(slot, reason);
            uvm_phase_f_route_log("moe_slot_pte_clear_guard_locked reason=%u layer=%u expert=%u slot=%u old_va=0x%llx old_end=0x%llx old_bytes=%llu guard_va=0x%llx guard_end=0x%llx guard_bytes=%llu\n",
                                  reason,
                                  slot->layer_id,
                                  slot->owner_expert_id,
                                  slot->slot_id,
                                  old_pte_va_base,
                                  old_pte_va_base + old_pte_bytes,
                                  old_pte_bytes,
                                  va_base,
                                  va_base + bytes,
                                  bytes);
        }
        else {
            uvm_moe_slot_pool_note_pte_failure_locked(slot);
            if (first_error == NV_OK)
                first_error = status;
            UVM_ERR_PRINT("MoE slot PTE locked clear guard failed: %s, VA space %px, layer %u, slot %u, VA [0x%llx, 0x%llx)\n",
                          nvstatusToString(status),
                          va_space,
                          slot->layer_id,
                          slot->slot_id,
                          old_pte_va_base,
                          old_pte_va_base + old_pte_bytes);
        }
    }

    uvm_phase_f_route_log("moe_pte_range action=clear_guard_locked_scan_end reason=%u guard_va=0x%llx guard_end=0x%llx guard_bytes=%llu status=0x%x\n",
                          reason,
                          va_base,
                          va_base + bytes,
                          bytes,
                          first_error);

done:
    mutex_unlock(&g_uvm_moe_slot_pool_lock);

    return first_error;
}

static NV_STATUS uvm_moe_slot_pool_forget_range_mappings_locked(uvm_va_space_t *va_space,
                                                                NvU64 va_base,
                                                                NvU64 bytes,
                                                                uvm_moe_slot_pte_clear_reason_t reason)
{
    NvU32 i;

    if (!va_space)
        return NV_ERR_INVALID_ARGUMENT;

    if (bytes == 0)
        return NV_OK;

    if (va_base + bytes <= va_base)
        return NV_ERR_INVALID_ARGUMENT;

    uvm_assert_rwsem_locked_write(&va_space->lock);
    // Unlike a fault-side range guard, address-space teardown changes the
    // lifetime contract captured by every registered plan even when no slot
    // PTE is currently published. Keep this invalidation unconditional.
    uvm_va_space_moe_slot_plan_invalidate(va_space);
    mutex_lock(&g_uvm_moe_slot_pool_lock);

    uvm_phase_f_route_log("moe_pte_range action=forget_guard_scan_begin reason=%u guard_va=0x%llx guard_end=0x%llx guard_bytes=%llu va_space=%px\n",
                          reason,
                          va_base,
                          va_base + bytes,
                          bytes,
                          va_space);

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];
        NvU64 old_pte_va_base;
        NvU64 old_pte_bytes;

        if (!uvm_moe_slot_pool_pte_mapping_overlaps(slot, va_space, va_base, bytes))
            continue;

        if (slot->static_alias_session_id != 0)
            slot->static_alias_poisoned = NV_TRUE;

        old_pte_va_base = slot->pte_va_base;
        old_pte_bytes = slot->pte_bytes;
        uvm_moe_slot_pool_note_pte_clear_reason_locked(slot, reason);
        uvm_phase_f_route_log("moe_slot_pte_forget_guard reason=%u layer=%u expert=%u slot=%u old_va=0x%llx old_end=0x%llx old_bytes=%llu guard_va=0x%llx guard_end=0x%llx guard_bytes=%llu\n",
                              reason,
                              slot->layer_id,
                              slot->owner_expert_id,
                              slot->slot_id,
                              old_pte_va_base,
                              old_pte_va_base + old_pte_bytes,
                              old_pte_bytes,
                              va_base,
                              va_base + bytes,
                              bytes);
        uvm_moe_slot_pool_forget_release_pte_mapping_locked(slot);
    }

    uvm_phase_f_route_log("moe_pte_range action=forget_guard_scan_end reason=%u guard_va=0x%llx guard_end=0x%llx guard_bytes=%llu\n",
                          reason,
                          va_base,
                          va_base + bytes,
                          bytes);

    mutex_unlock(&g_uvm_moe_slot_pool_lock);

    return NV_OK;
}

bool uvm_moe_slot_pool_range_has_mappings(uvm_va_space_t *va_space, NvU64 base, NvU64 bytes)
{
    NvU32 i;
    bool found = false;

    if (!va_space || bytes == 0 || base + bytes <= base)
        return false;

    uvm_assert_rwsem_locked(&va_space->lock);

    mutex_lock(&g_uvm_moe_slot_pool_lock);

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        if (uvm_moe_slot_pool_pte_mapping_overlaps(&g_uvm_moe_slot_pool[i], va_space, base, bytes)) {
            found = true;
            break;
        }
    }

    mutex_unlock(&g_uvm_moe_slot_pool_lock);

    return found;
}

NV_STATUS uvm_moe_static_alias_reject_range_mutation_locked(uvm_va_space_t *va_space,
                                                             NvU64 base,
                                                             NvU64 bytes)
{
    NV_STATUS status = NV_OK;
    NvU32 i;

    if (!va_space || bytes == 0 || base + bytes <= base)
        return NV_ERR_INVALID_ARGUMENT;

    uvm_assert_rwsem_locked(&va_space->lock);
    mutex_lock(&g_uvm_moe_slot_pool_lock);

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];

        if (slot->static_alias_session_id == 0 ||
            !uvm_moe_slot_pool_pte_mapping_overlaps(slot, va_space, base, bytes)) {
            continue;
        }

        uvm_moe_slot_pool_note_pte_clear_reason_locked(
            slot,
            UVM_MOE_SLOT_PTE_CLEAR_REASON_DANGEROUS_REJECT);
        status = NV_ERR_IN_USE;
    }

    mutex_unlock(&g_uvm_moe_slot_pool_lock);
    return status;
}

NV_STATUS uvm_moe_slot_pool_clear_range_mappings_for_fault(uvm_va_space_t *va_space, NvU64 base, NvU64 bytes)
{
    struct mm_struct *mm;
    NV_STATUS status;

    if (!va_space)
        return NV_ERR_INVALID_ARGUMENT;

    if (bytes == 0)
        return NV_OK;

    if (base + bytes <= base)
        return NV_ERR_INVALID_ARGUMENT;

    mm = uvm_va_space_mm_retain_lock(va_space);
    uvm_va_space_down_write(va_space);
    status = uvm_moe_slot_pool_clear_range_mappings_locked(va_space,
                                                           mm,
                                                           base,
                                                           bytes,
                                                           UVM_MOE_SLOT_PTE_CLEAR_REASON_FAULT);
    uvm_va_space_up_write(va_space);
    uvm_va_space_mm_release_unlock(va_space, mm);

    return status;
}

NV_STATUS uvm_moe_slot_pool_clear_range_mappings_for_migrate(uvm_va_space_t *va_space, NvU64 base, NvU64 bytes)
{
    return uvm_moe_slot_pool_clear_range_mappings(va_space,
                                                  base,
                                                  bytes,
                                                  UVM_MOE_SLOT_PTE_CLEAR_REASON_MIGRATE);
}

NV_STATUS uvm_moe_slot_pool_clear_ranges_mappings_for_migrate_locked(uvm_va_space_t *va_space,
                                                                     struct mm_struct *mm,
                                                                     const UVM_PHASE_F_RANGE *ranges,
                                                                     NvU32 count)
{
    NV_STATUS first_error = NV_OK;
    NvU32 slot_index;

    if (!va_space || !ranges || count == 0 || count > UVM_PHASE_F_PREFETCH_GPU_BATCH_V2_MAX_RANGES)
        return NV_ERR_INVALID_ARGUMENT;

    uvm_assert_rwsem_locked_write(&va_space->lock);
    // Managed allocations used by the KV pool are disjoint from expert
    // Parameter/upload aliases. Avoid two full slot-table scans for every
    // ordinary cudaMemPrefetchAsync when the requested ranges cannot overlap
    // any slot PTE ever installed in this VA space. The monotonic bounds may
    // yield a conservative false positive after removals, never a false
    // negative for a live or subsequently reinstalled mapping.
    if (!uvm_moe_slot_pte_seen_bounds_overlap_ranges(va_space, ranges, count))
        return NV_OK;

    // Migration can change the pinned/source backing cached by a registered
    // copy plan even when none of the requested ranges currently owns a slot
    // PTE. This is a source mutation guard, not only a PTE-overlap guard.
    uvm_va_space_moe_slot_plan_invalidate(va_space);
    mutex_lock(&g_uvm_moe_slot_pool_lock);

    // The batched prefetch/migrate path scans the pool directly instead of
    // using the single-range helper, so give it the same immutable-session
    // preflight before it can release the first mapping.
    for (slot_index = 0; slot_index < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++slot_index) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[slot_index];
        NvU32 range_index;

        if (!slot->valid || slot->pte_va_space != va_space || slot->pte_bytes == 0)
            continue;

        for (range_index = 0; range_index < count; ++range_index) {
            if (uvm_moe_slot_pool_pte_mapping_overlaps(slot,
                                                       va_space,
                                                       ranges[range_index].base,
                                                       ranges[range_index].length))
                break;
        }

        if (range_index < count &&
            uvm_moe_static_alias_rejects_clear_locked(
                slot,
                UVM_MOE_SLOT_PTE_CLEAR_REASON_MIGRATE)) {
            first_error = NV_ERR_IN_USE;
        }
    }

    if (first_error != NV_OK) {
        mutex_unlock(&g_uvm_moe_slot_pool_lock);
        return first_error;
    }

    // Scan the slot table once. The common case has no active slot PTEs, so it
    // must not pay count independent 8192-slot scans or mutex acquisitions.
    for (slot_index = 0; slot_index < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++slot_index) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[slot_index];
        NvU32 range_index;

        if (!slot->valid || slot->pte_va_space != va_space || slot->pte_bytes == 0)
            continue;

        for (range_index = 0; range_index < count; ++range_index) {
            if (uvm_moe_slot_pool_pte_mapping_overlaps(slot,
                                                       va_space,
                                                       ranges[range_index].base,
                                                       ranges[range_index].length))
                break;
        }

        if (range_index < count) {
            NvU64 old_pte_va_base = slot->pte_va_base;
            NvU64 old_pte_bytes = slot->pte_bytes;
            NV_STATUS status = uvm_moe_slot_pool_release_pte_mapping_locked(slot, va_space, mm);

            if (status == NV_OK) {
                uvm_moe_slot_pool_note_pte_clear_reason_locked(slot,
                                                               UVM_MOE_SLOT_PTE_CLEAR_REASON_MIGRATE);
                uvm_phase_f_route_log("moe_slot_pte_clear_batch_migrate layer=%u expert=%u slot=%u old_va=0x%llx old_end=0x%llx old_bytes=%llu range_index=%u guard_va=0x%llx guard_end=0x%llx guard_bytes=%llu\n",
                                      slot->layer_id,
                                      slot->owner_expert_id,
                                      slot->slot_id,
                                      old_pte_va_base,
                                      old_pte_va_base + old_pte_bytes,
                                      old_pte_bytes,
                                      range_index,
                                      ranges[range_index].base,
                                      ranges[range_index].base + ranges[range_index].length,
                                      ranges[range_index].length);
            }
            else {
                uvm_moe_slot_pool_note_pte_failure_locked(slot);
                if (first_error == NV_OK)
                    first_error = status;
            }
        }
    }

    mutex_unlock(&g_uvm_moe_slot_pool_lock);

    return first_error;
}

NV_STATUS uvm_moe_slot_pool_clear_range_mappings_for_discard(uvm_va_space_t *va_space, NvU64 base, NvU64 bytes)
{
    return uvm_moe_slot_pool_clear_range_mappings(va_space,
                                                  base,
                                                  bytes,
                                                  UVM_MOE_SLOT_PTE_CLEAR_REASON_DISCARD);
}

NV_STATUS uvm_moe_slot_pool_clear_range_mappings_for_unmap_locked(uvm_va_space_t *va_space,
                                                                  struct mm_struct *mm,
                                                                  NvU64 base,
                                                                  NvU64 bytes)
{
    return uvm_moe_slot_pool_clear_range_mappings_locked(va_space,
                                                         mm,
                                                         base,
                                                         bytes,
                                                         UVM_MOE_SLOT_PTE_CLEAR_REASON_UNMAP);
}

static NV_STATUS uvm_moe_slot_pool_clear_range_mappings_for_release_locked(uvm_va_space_t *va_space,
                                                                           struct mm_struct *mm,
                                                                           NvU64 base,
                                                                           NvU64 bytes)
{
    if (!mm)
        return uvm_moe_slot_pool_forget_range_mappings_locked(va_space,
                                                              base,
                                                              bytes,
                                                              UVM_MOE_SLOT_PTE_CLEAR_REASON_RELEASE);

    return uvm_moe_slot_pool_clear_range_mappings_locked(va_space,
                                                         mm,
                                                         base,
                                                         bytes,
                                                         UVM_MOE_SLOT_PTE_CLEAR_REASON_RELEASE);
}

NV_STATUS uvm_moe_slot_pool_clear_range_mappings_for_native_invalidate(uvm_va_space_t *va_space,
                                                                       struct mm_struct *mm,
                                                                       NvU64 base,
                                                                       NvU64 bytes)
{
    NV_STATUS status;

    if (!va_space)
        return NV_ERR_INVALID_ARGUMENT;

    uvm_va_space_down_write(va_space);
    status = uvm_moe_slot_pool_clear_range_mappings_locked(va_space,
                                                           mm,
                                                           base,
                                                           bytes,
                                                           UVM_MOE_SLOT_PTE_CLEAR_REASON_UNMAP);
    uvm_va_space_up_write(va_space);

    return status;
}

NV_STATUS uvm_moe_slot_pool_clear_range_mappings_for_split_locked(uvm_va_space_t *va_space,
                                                                  struct mm_struct *mm,
                                                                  NvU64 base,
                                                                  NvU64 bytes)
{
    return uvm_moe_slot_pool_clear_range_mappings_locked(va_space,
                                                         mm,
                                                         base,
                                                         bytes,
                                                         UVM_MOE_SLOT_PTE_CLEAR_REASON_SPLIT);
}

static bool uvm_moe_slot_pool_entry_matches(const uvm_moe_slot_pool_slot_t *slot,
                                            const uvm_va_space_t *owner_va_space,
                                            NvU32 layer_id,
                                            NvU32 slot_id)
{
    return slot->valid &&
           slot->owner_va_space == owner_va_space &&
           slot->layer_id == layer_id &&
           slot->slot_id == slot_id;
}

static uvm_moe_slot_pool_slot_t *uvm_moe_slot_pool_find_locked(const uvm_va_space_t *owner_va_space,
                                                               NvU32 layer_id,
                                                               NvU32 slot_id)
{
    NvU32 i;

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        if (uvm_moe_slot_pool_entry_matches(&g_uvm_moe_slot_pool[i], owner_va_space, layer_id, slot_id))
            return &g_uvm_moe_slot_pool[i];
    }

    return NULL;
}

static NvBool uvm_moe_slot_pool_key_owned_by_other_locked(const uvm_va_space_t *owner_va_space,
                                                           NvU32 layer_id,
                                                           NvU32 slot_id)
{
    NvU32 i;

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        const uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];

        if (slot->valid &&
            slot->owner_va_space != owner_va_space &&
            slot->layer_id == layer_id &&
            slot->slot_id == slot_id) {
            return NV_TRUE;
        }
    }

    return NV_FALSE;
}

static uvm_moe_slot_pool_slot_t *uvm_moe_slot_pool_find_free_locked(void)
{
    NvU32 i;

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        if (!g_uvm_moe_slot_pool[i].valid)
            return &g_uvm_moe_slot_pool[i];
    }

    return NULL;
}

static NV_STATUS uvm_moe_slot_pool_gpu_memset(uvm_gpu_t *gpu,
                                              uvm_mem_t *mem,
                                              NvU64 offset,
                                              NvU64 bytes)
{
    NV_STATUS status;
    uvm_push_t push;
    uvm_gpu_address_t dst;

    if (bytes == 0)
        return NV_OK;

    if (bytes > (NvU64)SIZE_MAX)
        return NV_ERR_INVALID_ARGUMENT;

    status = uvm_push_begin(gpu->channel_manager,
                            UVM_CHANNEL_TYPE_GPU_INTERNAL,
                            &push,
                            "MoE slot pool zero");
    if (status != NV_OK)
        return status;

    dst = uvm_mem_gpu_address_virtual_kernel(mem, gpu);
    dst.address += offset;
    gpu->parent->ce_hal->memset_1(&push, dst, 0, (size_t)bytes);

    return uvm_push_end_and_wait(&push);
}

static NV_STATUS uvm_moe_slot_pool_gpu_memcopy(uvm_gpu_t *gpu,
                                               uvm_channel_type_t channel_type,
                                               uvm_mem_t *dst_mem,
                                               NvU64 dst_offset,
                                               uvm_mem_t *src_mem,
                                               NvU64 src_offset,
                                               NvU64 bytes)
{
    NV_STATUS status;
    uvm_push_t push;
    uvm_gpu_address_t dst;
    uvm_gpu_address_t src;

    if (bytes == 0)
        return NV_OK;

    if (bytes > (NvU64)SIZE_MAX)
        return NV_ERR_INVALID_ARGUMENT;

    status = uvm_push_begin(gpu->channel_manager,
                            channel_type,
                            &push,
                            "MoE slot pool copy");
    if (status != NV_OK)
        return status;

    dst = uvm_mem_gpu_address_virtual_kernel(dst_mem, gpu);
    dst.address += dst_offset;
    src = uvm_mem_gpu_address_virtual_kernel(src_mem, gpu);
    src.address += src_offset;
    gpu->parent->ce_hal->memcopy(&push, dst, src, (size_t)bytes);

    return uvm_push_end_and_wait(&push);
}

static NV_STATUS uvm_moe_slot_pool_ensure_sysmem_backing_locked(uvm_moe_slot_pool_slot_t *slot,
                                                                NvU64 required_bytes)
{
    NV_STATUS status;
    uvm_mem_t *new_mem = NULL;
    void *new_addr;
    NvU64 old_bytes;

    required_bytes = UVM_ALIGN_UP(required_bytes, PAGE_SIZE);
    if (required_bytes == 0)
        return NV_ERR_INVALID_ARGUMENT;

    if (slot->slot_mem &&
        slot->backing_kind == UVM_MOE_SLOT_POOL_BACKING_SYSMEM &&
        slot->slot_bytes >= required_bytes)
        return NV_OK;

    if (slot->pte_range_vec || slot->pte_bytes != 0)
        return NV_ERR_INVALID_STATE;

    status = uvm_mem_alloc_sysmem_and_map_cpu_kernel(required_bytes, current->mm, &new_mem);
    if (status != NV_OK)
        return status;

    new_addr = uvm_mem_get_cpu_addr_kernel(new_mem);
    if (!new_addr) {
        uvm_mem_free(new_mem);
        return NV_ERR_INVALID_STATE;
    }

    memset(new_addr, 0, required_bytes);

    if (slot->slot_mem && slot->backing_kind == UVM_MOE_SLOT_POOL_BACKING_SYSMEM) {
        void *old_addr = uvm_mem_get_cpu_addr_kernel(slot->slot_mem);

        old_bytes = min(slot->slot_bytes, required_bytes);
        if (old_addr && old_bytes)
            memcpy(new_addr, old_addr, old_bytes);
    }

    status = uvm_moe_slot_pool_release_backing_locked(slot, NULL, NULL);
    if (status != NV_OK) {
        uvm_mem_free(new_mem);
        return status;
    }

    slot->slot_mem = new_mem;
    slot->slot_bytes = required_bytes;
    slot->backing_generation = ++g_uvm_moe_slot_pool_backing_generation;
    if (slot->backing_generation == 0)
        slot->backing_generation = ++g_uvm_moe_slot_pool_backing_generation;
    slot->backing_kind = UVM_MOE_SLOT_POOL_BACKING_SYSMEM;
    uvm_moe_slot_pool_bump_backing_epoch_locked();
    return NV_OK;
}

static NV_STATUS uvm_moe_slot_pool_ensure_vidmem_backing_locked(uvm_moe_slot_pool_slot_t *slot,
                                                                uvm_gpu_t *gpu,
                                                                NvU64 required_bytes,
                                                                uvm_va_space_t *locked_va_space,
                                                                struct mm_struct *mm,
                                                                NvBool allow_eviction)
{
    NV_STATUS status;
    uvm_mem_t *new_mem = NULL;
    NvU64 old_bytes = 0;
    NvU64 allocation_bytes;
    NvBool preserve_old = NV_FALSE;

    if (!gpu)
        return NV_ERR_INVALID_DEVICE;

    required_bytes = UVM_ALIGN_UP(required_bytes, PAGE_SIZE);
    if (required_bytes == 0)
        return NV_ERR_INVALID_ARGUMENT;

    if (slot->slot_mem &&
        slot->slot_gpu == gpu &&
        slot->backing_kind == UVM_MOE_SLOT_POOL_BACKING_VIDMEM &&
        slot->slot_bytes >= required_bytes)
        return NV_OK;

    if (slot->pte_range_vec || slot->pte_bytes != 0)
        return NV_ERR_INVALID_STATE;

    preserve_old = slot->slot_mem &&
                   slot->slot_gpu == gpu &&
                   slot->backing_kind == UVM_MOE_SLOT_POOL_BACKING_VIDMEM;

    allocation_bytes = required_bytes;
    if (preserve_old)
        allocation_bytes += slot->slot_bytes;

    if (allow_eviction) {
        uvm_moe_slot_pool_enforce_vidmem_budget_locked(gpu,
                                                       slot,
                                                       allocation_bytes,
                                                       NV_FALSE,
                                                       locked_va_space,
                                                       mm);
    }

    if (!uvm_moe_slot_pool_vidmem_budget_allows_locked(gpu, slot, allocation_bytes)) {
        ++g_uvm_moe_slot_pool_no_evict_budget_reject_count;
        uvm_phase_f_route_log("moe_slot_pool_vidmem_budget_reject layer=%u expert=%u slot=%u required=%llu allocation=%llu budget=%llu no_evict=%u rejects=%llu\n",
                              slot->layer_id,
                              slot->owner_expert_id,
                              slot->slot_id,
                              required_bytes,
                              allocation_bytes,
                              (NvU64)uvm_moe_slot_pool_vidmem_budget_bytes,
                              !allow_eviction,
                              g_uvm_moe_slot_pool_no_evict_budget_reject_count);
        return NV_ERR_NO_MEMORY;
    }

    status = uvm_mem_alloc_vidmem(required_bytes, gpu, &new_mem);
    if (status == NV_ERR_NO_MEMORY && allow_eviction) {
        NvU64 evicted = uvm_moe_slot_pool_enforce_vidmem_budget_locked(gpu,
                                                                       slot,
                                                                       allocation_bytes,
                                                                       NV_TRUE,
                                                                       locked_va_space,
                                                                       mm);

        if (evicted) {
            uvm_phase_f_route_log("moe_slot_pool_vidmem_alloc_retry required=%llu evicted=%llu budget=%llu\n",
                                  required_bytes,
                                  evicted,
                                  (NvU64)uvm_moe_slot_pool_vidmem_budget_bytes);
            status = uvm_mem_alloc_vidmem(required_bytes, gpu, &new_mem);
        }
    }
    if (status != NV_OK)
        return status;

    status = uvm_mem_map_gpu_kernel(new_mem, gpu);
    if (status != NV_OK)
        goto error;

    status = uvm_moe_slot_pool_gpu_memset(gpu, new_mem, 0, required_bytes);
    if (status != NV_OK)
        goto error;

    if (preserve_old) {
        status = uvm_mem_map_gpu_kernel(slot->slot_mem, gpu);
        if (status != NV_OK)
            goto error;

        old_bytes = min(slot->slot_bytes, required_bytes);
        status = uvm_moe_slot_pool_gpu_memcopy(gpu,
                                               UVM_CHANNEL_TYPE_GPU_INTERNAL,
                                               new_mem,
                                               0,
                                               slot->slot_mem,
                                               0,
                                               old_bytes);
        if (status != NV_OK)
            goto error;
    }

    status = uvm_moe_slot_pool_release_backing_locked(slot, NULL, NULL);
    if (status != NV_OK)
        goto error;

    slot->slot_mem = new_mem;
    slot->slot_bytes = required_bytes;
    slot->backing_generation = ++g_uvm_moe_slot_pool_backing_generation;
    if (slot->backing_generation == 0)
        slot->backing_generation = ++g_uvm_moe_slot_pool_backing_generation;
    slot->backing_kind = UVM_MOE_SLOT_POOL_BACKING_VIDMEM;
    uvm_gpu_retain(gpu);
    slot->slot_gpu = gpu;
    uvm_moe_slot_pool_bump_backing_epoch_locked();
    ++g_uvm_moe_slot_pool_vidmem_alloc_count;

    return NV_OK;

error:
    uvm_mem_free(new_mem);
    return status;
}

static NV_STATUS uvm_moe_slot_pool_copy_sysmem_entry_locked(uvm_moe_slot_pool_slot_t *slot,
                                                            const UVM_MOE_SLOT_REMAP_ENTRY *entry)
{
    NV_STATUS status;
    void *dst;
    const void __user *src = (const void __user *)(unsigned long)entry->va_base;
    NvU64 copy_end = entry->slot_offset + entry->bytes;

    status = uvm_moe_slot_pool_ensure_sysmem_backing_locked(slot, copy_end);
    if (status != NV_OK)
        return status;

    dst = (char *)uvm_mem_get_cpu_addr_kernel(slot->slot_mem) + entry->slot_offset;
    if (!access_ok(src, entry->bytes))
        return NV_ERR_INVALID_ADDRESS;

    if (copy_from_user(dst, src, entry->bytes) != 0)
        return NV_ERR_INVALID_ADDRESS;

    return NV_OK;
}

static NV_STATUS uvm_moe_slot_pool_copy_gpu_entry_locked(uvm_moe_slot_pool_slot_t *slot,
                                                         const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                         uvm_gpu_t *gpu,
                                                         uvm_va_space_t *locked_va_space,
                                                         struct mm_struct *mm)
{
    NV_STATUS status;
    const char __user *src = (const char __user *)(unsigned long)entry->va_base;
    NvU64 copy_end = entry->slot_offset + entry->bytes;
    NvU64 copied = 0;
    NvU64 max_chunk_bytes = UVM_MOE_SLOT_POOL_GPU_COPY_STAGING_CHUNK_BYTES;

    if (!gpu)
        return NV_ERR_INVALID_DEVICE;

    if (entry->bytes > (NvU64)SIZE_MAX)
        return NV_ERR_INVALID_ARGUMENT;

    if (!access_ok(src, entry->bytes))
        return NV_ERR_INVALID_ADDRESS;

    status = uvm_moe_slot_pool_ensure_vidmem_backing_locked(
        slot,
        gpu,
        copy_end,
        locked_va_space,
        mm,
        !(entry->flags & UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT));
    if (status != NV_OK)
        return status;

    max_chunk_bytes = UVM_ALIGN_DOWN(max_chunk_bytes, PAGE_SIZE);
    if (max_chunk_bytes == 0)
        max_chunk_bytes = PAGE_SIZE;

    while (copied < entry->bytes) {
        uvm_mem_t *staging = NULL;
        void *staging_addr;
        NvU64 remaining = entry->bytes - copied;
        NvU64 chunk_bytes = min(remaining, max_chunk_bytes);

        chunk_bytes = UVM_ALIGN_DOWN(chunk_bytes, PAGE_SIZE);
        if (chunk_bytes == 0)
            chunk_bytes = remaining;

        status = uvm_mem_alloc_sysmem_dma_and_map_cpu_kernel(chunk_bytes, gpu, current->mm, &staging);
        if (status != NV_OK) {
            if (status == NV_ERR_NO_MEMORY && max_chunk_bytes > PAGE_SIZE) {
                max_chunk_bytes = UVM_ALIGN_DOWN(max_chunk_bytes / 2, PAGE_SIZE);
                if (max_chunk_bytes < PAGE_SIZE)
                    max_chunk_bytes = PAGE_SIZE;
                uvm_phase_f_route_log("moe_slot_pool_gpu_copy_staging_retry layer=%u expert=%u slot=%u offset=%llu remaining=%llu next_chunk=%llu status=0x%x\n",
                                      entry->layer_id,
                                      entry->expert_id,
                                      entry->slot_id,
                                      copied,
                                      remaining,
                                      max_chunk_bytes,
                                      status);
                continue;
            }
            return status;
        }

        staging_addr = uvm_mem_get_cpu_addr_kernel(staging);
        if (!staging_addr) {
            status = NV_ERR_INVALID_STATE;
            goto done;
        }

        if (copy_from_user(staging_addr, src + copied, chunk_bytes) != 0) {
            status = NV_ERR_INVALID_ADDRESS;
            goto done;
        }

        status = uvm_mem_map_gpu_kernel(staging, gpu);
        if (status != NV_OK)
            goto done;

        status = uvm_moe_slot_pool_gpu_memcopy(gpu,
                                               UVM_CHANNEL_TYPE_CPU_TO_GPU,
                                               slot->slot_mem,
                                               entry->slot_offset + copied,
                                               staging,
                                               0,
                                               chunk_bytes);

done:
        uvm_mem_free(staging);
        if (status != NV_OK)
            return status;

        copied += chunk_bytes;
    }

    return NV_OK;
}

static NV_STATUS uvm_moe_slot_pool_copy_gpu_entry_async_locked(uvm_moe_slot_pool_slot_t *slot,
                                                               const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                               uvm_gpu_t *gpu,
                                                               uvm_va_space_t *locked_va_space,
                                                               struct mm_struct *mm)
{
    NV_STATUS status;
    const char __user *src = (const char __user *)(unsigned long)entry->va_base;
    NvU64 copy_end = entry->slot_offset + entry->bytes;
    uvm_mem_t *staging = NULL;
    void *staging_addr;
    uvm_push_t push;
    uvm_gpu_address_t dst;
    uvm_gpu_address_t staging_src;
    NvBool completed = NV_FALSE;

    if (!gpu)
        return NV_ERR_INVALID_DEVICE;

    if (entry->bytes > (NvU64)SIZE_MAX)
        return NV_ERR_INVALID_ARGUMENT;

    if (!access_ok(src, entry->bytes))
        return NV_ERR_INVALID_ADDRESS;

    if (slot->async_copy_pending) {
        status = uvm_moe_slot_pool_finish_async_copy_locked(slot, NV_FALSE, &completed);
        if (status != NV_OK)
            return status;
        if (!completed)
            return NV_ERR_INVALID_STATE;
    }

    status = uvm_moe_slot_pool_ensure_vidmem_backing_locked(
        slot,
        gpu,
        copy_end,
        locked_va_space,
        mm,
        !(entry->flags & UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT));
    if (status != NV_OK)
        return status;

    status = uvm_mem_alloc_sysmem_dma_and_map_cpu_kernel(entry->bytes, gpu, current->mm, &staging);
    if (status != NV_OK)
        return status;

    staging_addr = uvm_mem_get_cpu_addr_kernel(staging);
    if (!staging_addr) {
        status = NV_ERR_INVALID_STATE;
        goto error;
    }

    if (copy_from_user(staging_addr, src, entry->bytes) != 0) {
        status = NV_ERR_INVALID_ADDRESS;
        goto error;
    }

    status = uvm_mem_map_gpu_kernel(staging, gpu);
    if (status != NV_OK)
        goto error;

    status = uvm_push_begin(gpu->channel_manager,
                            UVM_CHANNEL_TYPE_CPU_TO_GPU,
                            &push,
                            "Async MoE slot pool copy");
    if (status != NV_OK)
        goto error;

    dst = uvm_mem_gpu_address_virtual_kernel(slot->slot_mem, gpu);
    dst.address += entry->slot_offset;
    staging_src = uvm_mem_gpu_address_virtual_kernel(staging, gpu);
    gpu->parent->ce_hal->memcopy(&push, dst, staging_src, (size_t)entry->bytes);
    uvm_push_end(&push);

    uvm_tracker_init(&slot->async_copy_tracker);
    uvm_tracker_overwrite_with_push(&slot->async_copy_tracker, &push);
    slot->async_copy_staging = staging;
    slot->async_copy_pending = NV_TRUE;
    slot->async_copy_bytes = entry->bytes;
    slot->async_copy_slot_offset = entry->slot_offset;
    slot->copy_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
    slot->state = UVM_MOE_SLOT_POOL_STATE_LOADING;

    return NV_OK;

error:
    if (staging)
        uvm_mem_free(staging);
    return status;
}

static NV_STATUS uvm_moe_slot_pool_copy_entry_locked(uvm_moe_slot_pool_slot_t *slot,
                                                     const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                     uvm_moe_slot_pool_copy_mode_t copy_mode,
                                                     uvm_gpu_t *gpu,
                                                     uvm_va_space_t *locked_va_space,
                                                     struct mm_struct *mm)
{
    if (copy_mode == UVM_MOE_SLOT_POOL_COPY_SYSMEM)
        return uvm_moe_slot_pool_copy_sysmem_entry_locked(slot, entry);

    if (copy_mode == UVM_MOE_SLOT_POOL_COPY_GPU)
        return uvm_moe_slot_pool_copy_gpu_entry_locked(slot, entry, gpu, locked_va_space, mm);

    if (copy_mode == UVM_MOE_SLOT_POOL_COPY_GPU_ASYNC)
        return uvm_moe_slot_pool_copy_gpu_entry_async_locked(slot, entry, gpu, locked_va_space, mm);

    return NV_OK;
}

typedef struct
{
    uvm_mem_t *mem;
    uvm_gpu_t *gpu;
    NvU64 slot_offset;
} uvm_moe_slot_pte_maker_data_t;

static NvU64 uvm_moe_slot_pte_maker(uvm_page_table_range_vec_t *range_vec, NvU64 offset, void *caller_data)
{
    uvm_moe_slot_pte_maker_data_t *data = (uvm_moe_slot_pte_maker_data_t *)caller_data;
    uvm_page_tree_t *tree = range_vec->tree;
    uvm_gpu_phys_address_t phys = uvm_mem_gpu_physical(data->mem,
                                                       data->gpu,
                                                       data->slot_offset + offset,
                                                       range_vec->page_size);

    return tree->hal->make_pte(phys.aperture,
                               phys.address,
                               UVM_PROT_READ_ONLY,
                               uvm_mem_is_vidmem(data->mem) ? UVM_MMU_PTE_FLAGS_CACHED : UVM_MMU_PTE_FLAGS_NONE);
}

static NV_STATUS uvm_moe_slot_pool_install_pte_mapping_locked(uvm_va_space_t *va_space,
                                                              uvm_moe_slot_pool_slot_t *slot,
                                                              const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                              uvm_gpu_t *gpu,
                                                              struct mm_struct *mm,
                                                              NvU32 pte_mode)
{
    NV_STATUS status;
    uvm_gpu_va_space_t *gpu_va_space;
    uvm_page_table_range_vec_t *range_vec = NULL;
    uvm_page_tree_t *page_tables;
    uvm_membar_t tlb_membar;
    NvU64 map_end = entry->slot_offset + entry->bytes;
    NvU64 page_size;
    uvm_moe_slot_pte_maker_data_t pte_maker_data;

    if (!uvm_moe_slot_pte_remap_enable)
        return NV_ERR_NOT_SUPPORTED;

    if (!gpu || !slot->slot_mem || slot->backing_kind != UVM_MOE_SLOT_POOL_BACKING_VIDMEM)
        return NV_ERR_INVALID_STATE;

    if (slot->slot_gpu != gpu)
        return NV_ERR_INVALID_DEVICE;

    if (entry->bytes > (NvU64)uvm_moe_slot_pte_remap_max_bytes)
        return NV_ERR_INVALID_ARGUMENT;

    if (map_end < entry->slot_offset || map_end > slot->slot_bytes)
        return NV_ERR_INVALID_ARGUMENT;

    if (!uvm_gpu_can_address(gpu, entry->va_base, entry->bytes))
        return NV_ERR_OUT_OF_RANGE;

    gpu_va_space = va_space->gpu_va_spaces[uvm_id_gpu_index(gpu->id)];
    if (!gpu_va_space)
        return NV_ERR_INVALID_DEVICE;

    page_tables = &gpu_va_space->page_tables;
    page_size = pte_mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP ?
                    UVM_PAGE_SIZE_4K :
                    uvm_mmu_biggest_page_size_up_to(page_tables, min(entry->bytes, UVM_PAGE_SIZE_64K));
    if (page_size == 0)
        return NV_ERR_NOT_SUPPORTED;

    if (!IS_ALIGNED(entry->va_base, page_size) ||
        !IS_ALIGNED(entry->bytes, page_size) ||
        !IS_ALIGNED(entry->slot_offset, page_size))
        return NV_ERR_INVALID_ARGUMENT;

    uvm_phase_f_route_log("moe_pte_range action=install_range_vec_begin layer=%u expert=%u slot=%u role=%u va=0x%llx end=0x%llx bytes=%llu slot_offset=%llu slot_end=%llu page_size=%llu mode=%u va_space=%px mm=%px gpu=%s\n",
                          entry->layer_id,
                          entry->expert_id,
                          entry->slot_id,
                          entry->role,
                          entry->va_base,
                          entry->va_base + entry->bytes,
                          entry->bytes,
                          entry->slot_offset,
                          map_end,
                          page_size,
                          pte_mode,
                          va_space,
                          mm,
                          uvm_gpu_name(gpu));

    status = uvm_moe_slot_pool_release_pte_mapping_locked(slot, va_space, mm);
    if (status != NV_OK) {
        uvm_phase_f_route_log("moe_pte_range action=install_range_vec_release_failed layer=%u expert=%u slot=%u va=0x%llx end=0x%llx bytes=%llu status=0x%x\n",
                              entry->layer_id,
                              entry->expert_id,
                              entry->slot_id,
                              entry->va_base,
                              entry->va_base + entry->bytes,
                              entry->bytes,
                              status);
        return status;
    }

    status = uvm_page_table_range_vec_create(page_tables,
                                             entry->va_base,
                                             entry->bytes,
                                             page_size,
                                             UVM_PMM_ALLOC_FLAGS_EVICT,
                                             &range_vec);
    if (status != NV_OK) {
        uvm_phase_f_route_log("moe_pte_range action=install_range_vec_create_failed layer=%u expert=%u slot=%u va=0x%llx end=0x%llx bytes=%llu page_size=%llu status=0x%x\n",
                              entry->layer_id,
                              entry->expert_id,
                              entry->slot_id,
                              entry->va_base,
                              entry->va_base + entry->bytes,
                              entry->bytes,
                              page_size,
                              status);
        return status;
    }

    pte_maker_data.mem = slot->slot_mem;
    pte_maker_data.gpu = gpu;
    pte_maker_data.slot_offset = entry->slot_offset;

    tlb_membar = uvm_hal_downgrade_membar_type(gpu, uvm_mem_is_local_vidmem(slot->slot_mem, gpu));
    status = uvm_page_table_range_vec_write_ptes(range_vec,
                                                 tlb_membar,
                                                 uvm_moe_slot_pte_maker,
                                                 &pte_maker_data);
    if (status != NV_OK) {
        uvm_phase_f_route_log("moe_pte_range action=install_range_vec_write_failed layer=%u expert=%u slot=%u va=0x%llx end=0x%llx bytes=%llu status=0x%x\n",
                              entry->layer_id,
                              entry->expert_id,
                              entry->slot_id,
                              entry->va_base,
                              entry->va_base + entry->bytes,
                              entry->bytes,
                              status);
        (void)uvm_page_table_range_vec_clear_ptes(range_vec, tlb_membar);
        uvm_page_table_range_vec_destroy(range_vec);
        return status;
    }

    slot->pte_range_vec = range_vec;
    slot->pte_gpu = gpu;
    uvm_gpu_retain(gpu);
    slot->pte_va_base = entry->va_base;
    slot->pte_bytes = entry->bytes;
    slot->pte_va_space = va_space;
    slot->pte_slot_offset = entry->slot_offset;
    slot->pte_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
    slot->pte_mode = pte_mode;
    slot->pte_va_block_tracked = pte_mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP;
    slot->pte_remap_count++;
    slot->pte_remap_bytes += entry->bytes;
    ++g_uvm_moe_slot_pool_pte_remap_count;
    g_uvm_moe_slot_pool_pte_remap_bytes += entry->bytes;
    uvm_moe_slot_pte_seen_bounds_extend(va_space,
                                        entry->va_base,
                                        entry->bytes);

    uvm_phase_f_route_log("moe_pte_range action=install_range_vec_end layer=%u expert=%u slot=%u role=%u va=0x%llx end=0x%llx bytes=%llu slot_offset=%llu page_size=%llu mode=%u status=0x%x remap_count=%llu remap_bytes=%llu\n",
                          entry->layer_id,
                          entry->expert_id,
                          entry->slot_id,
                          entry->role,
                          entry->va_base,
                          entry->va_base + entry->bytes,
                          entry->bytes,
                          entry->slot_offset,
                          page_size,
                          pte_mode,
                          status,
                          slot->pte_remap_count,
                          slot->pte_remap_bytes);

    return NV_OK;
}

static NV_STATUS uvm_moe_slot_pool_unmap_va_block_gpu_locked(uvm_va_block_t *block,
                                                             uvm_va_block_context_t *block_context,
                                                             const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                             uvm_gpu_t *gpu)
{
    uvm_va_block_region_t region = uvm_va_block_region_from_start_size(block, entry->va_base, entry->bytes);
    uvm_page_mask_t *page_mask = &block_context->caller_page_mask;

    uvm_page_mask_init_from_region(page_mask, region, NULL);
    return uvm_va_block_unmap(block, block_context, gpu->id, region, page_mask, &block->tracker);
}

static NV_STATUS uvm_moe_slot_pool_map_va_block_slot_locked(uvm_va_block_t *block,
                                                            uvm_va_block_context_t *block_context,
                                                            const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                            uvm_gpu_t *gpu,
                                                            uvm_mem_t *slot_mem)
{
    uvm_va_block_region_t region = uvm_va_block_region_from_start_size(block, entry->va_base, entry->bytes);
    uvm_page_mask_t *page_mask = &block_context->caller_page_mask;

    uvm_page_mask_init_from_region(page_mask, region, NULL);
    return uvm_va_block_map_moe_slot_readonly(block,
                                              block_context,
                                              gpu,
                                              region,
                                              page_mask,
                                              slot_mem,
                                              entry->slot_offset,
                                              &block->tracker);
}

static NV_STATUS uvm_moe_slot_pool_unmap_va_block_slot_locked(uvm_va_block_t *block,
                                                              uvm_va_block_context_t *block_context,
                                                              NvU64 va_base,
                                                              NvU64 bytes,
                                                              uvm_gpu_t *gpu,
                                                              NvBool restore_cpu_mapping)
{
    uvm_va_block_region_t region = uvm_va_block_region_from_start_size(block, va_base, bytes);
    uvm_page_mask_t *page_mask = &block_context->caller_page_mask;

    uvm_page_mask_init_from_region(page_mask, region, NULL);
    if (restore_cpu_mapping) {
        return uvm_va_block_unmap_moe_slot(block,
                                           block_context,
                                           gpu,
                                           region,
                                           page_mask,
                                           &block->tracker);
    }

    return uvm_va_block_unmap_moe_slot_no_cpu_restore(block,
                                                       block_context,
                                                       gpu,
                                                       region,
                                                       page_mask,
                                                       &block->tracker);
}

static NV_STATUS uvm_moe_slot_pool_prepare_va_block_remap(uvm_va_space_t *va_space,
                                                          const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                          uvm_gpu_t *gpu,
                                                          struct mm_struct *mm,
                                                          NvU64 *block_unmap_count_out,
                                                          NvU64 *tracker_wait_count_out)
{
    NV_STATUS status;
    uvm_va_block_t *block = NULL;
    uvm_va_block_context_t *block_context;
    uvm_va_block_retry_t block_retry;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
    NvU64 entry_end = entry->va_base + entry->bytes - 1;

    if (block_unmap_count_out)
        *block_unmap_count_out = 0;
    if (tracker_wait_count_out)
        *tracker_wait_count_out = 0;

    if (entry_end < entry->va_base)
        return NV_ERR_INVALID_ARGUMENT;

    block_context = uvm_va_block_context_alloc(mm);
    if (!block_context)
        return NV_ERR_NO_MEMORY;

    status = uvm_va_block_find_create_managed(va_space, entry->va_base, &block);
    if (status != NV_OK)
        goto out;

    if (entry_end > block->end) {
        status = NV_ERR_INVALID_ARGUMENT;
        goto out;
    }

    status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                     &block_retry,
                                     uvm_moe_slot_pool_unmap_va_block_gpu_locked(block,
                                                                                 block_context,
                                                                                 entry,
                                                                                 gpu));
    if (status != NV_OK)
        goto out;

    if (block_unmap_count_out)
        *block_unmap_count_out = 1;

    uvm_mutex_lock(&block->lock);
    status = uvm_tracker_init_from(&local_tracker, &block->tracker);
    uvm_mutex_unlock(&block->lock);
    if (status != NV_OK)
        goto out;

    status = uvm_tracker_wait_deinit(&local_tracker);
    if (status != NV_OK)
        goto out;

    if (tracker_wait_count_out)
        *tracker_wait_count_out = 1;

out:
    uvm_va_block_context_free(block_context);
    return status;
}

static void uvm_moe_slot_pool_log_va_block_find_failed(uvm_va_space_t *va_space,
                                                       const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                       struct mm_struct *mm,
                                                       NvU64 cursor,
                                                       NvU64 remaining,
                                                       NV_STATUS status,
                                                       uvm_api_range_type_t range_type)
{
    struct mm_struct *va_space_mm = va_space ? va_space->va_space_mm.mm : NULL;
    struct vm_area_struct *vma = mm ? find_vma(mm, cursor) : NULL;
    NvU64 vma_start = vma ? (NvU64)vma->vm_start : 0;
    NvU64 vma_end = vma ? (NvU64)vma->vm_end : 0;
    unsigned long vma_flags = vma ? vma->vm_flags : 0;
    NvU32 vma_covers = vma && cursor >= (NvU64)vma->vm_start && cursor < (NvU64)vma->vm_end;
    NvU32 vma_read = vma && (vma->vm_flags & VM_READ);
    NvU32 vma_io = vma && (vma->vm_flags & VM_IO);
    NvU32 vma_pfnmap = vma && (vma->vm_flags & VM_PFNMAP);
    NvU32 vma_managed = vma && uvm_vma_is_managed(vma);

    uvm_phase_f_route_log("moe_slot_va_block_install_find_failed layer=%u expert=%u slot=%u cursor=0x%llx remaining=%llu status=0x%x range_type=%u hmm_enabled=%u mm=%px current_mm=%px va_space_mm=%px vma=%px vma=[0x%llx,0x%llx) vma_flags=0x%lx vma_covers=%u vma_read=%u vma_io=%u vma_pfnmap=%u vma_managed=%u vma_ops=%px\n",
                          entry->layer_id,
                          entry->expert_id,
                          entry->slot_id,
                          cursor,
                          remaining,
                          status,
                          range_type,
                          va_space ? (NvU32)uvm_hmm_is_enabled(va_space) : 0,
                          mm,
                          current->mm,
                          va_space_mm,
                          vma,
                          vma_start,
                          vma_end,
                          vma_flags,
                          vma_covers,
                          vma_read,
                          vma_io,
                          vma_pfnmap,
                          vma_managed,
                          vma ? vma->vm_ops : NULL);
}

static NV_STATUS uvm_moe_slot_pool_install_va_block_slot_mapping(uvm_va_space_t *va_space,
                                                                 const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                                 uvm_gpu_t *gpu,
                                                                 struct mm_struct *mm,
                                                                 uvm_mem_t *slot_mem,
                                                                 NvU64 *block_unmap_count_out,
                                                                 NvU64 *tracker_wait_count_out,
                                                                 NvU64 *rollback_bytes_out,
                                                                 NvU64 *rollback_block_unmap_count_out,
                                                                 NvU64 *rollback_tracker_wait_count_out)
{
    NV_STATUS status = NV_OK;
    uvm_va_block_context_t *block_context;
    NvU64 entry_end = entry->va_base + entry->bytes;
    NvU64 cursor = entry->va_base;
    NvU64 installed_bytes = 0;

    if (block_unmap_count_out)
        *block_unmap_count_out = 0;
    if (tracker_wait_count_out)
        *tracker_wait_count_out = 0;
    if (rollback_bytes_out)
        *rollback_bytes_out = 0;
    if (rollback_block_unmap_count_out)
        *rollback_block_unmap_count_out = 0;
    if (rollback_tracker_wait_count_out)
        *rollback_tracker_wait_count_out = 0;

    if (!slot_mem)
        return NV_ERR_INVALID_STATE;

    if (entry_end <= entry->va_base)
        return NV_ERR_INVALID_ARGUMENT;

    block_context = uvm_va_block_context_alloc(mm);
    if (!block_context)
        return NV_ERR_NO_MEMORY;

    uvm_phase_f_route_log("moe_pte_range action=install_va_block_begin layer=%u expert=%u slot=%u role=%u va=0x%llx end=0x%llx bytes=%llu slot_offset=%llu va_space=%px mm=%px gpu=%s\n",
                          entry->layer_id,
                          entry->expert_id,
                          entry->slot_id,
                          entry->role,
                          entry->va_base,
                          entry_end,
                          entry->bytes,
                          entry->slot_offset,
                          va_space,
                          mm,
                          uvm_gpu_name(gpu));

    while (cursor < entry_end) {
        UVM_MOE_SLOT_REMAP_ENTRY slice = *entry;
        uvm_va_block_t *block = NULL;
        uvm_va_block_retry_t block_retry;
        uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
        NvU64 slice_end;
        uvm_api_range_type_t range_type;

        block_context->hmm.vma = NULL;
        status = mm ?
                    uvm_va_block_find_create(va_space, cursor, &block_context->hmm.vma, &block) :
                    uvm_va_block_find_create_managed(va_space, cursor, &block);
        if (status != NV_OK) {
            range_type = uvm_api_range_type_check(va_space, mm, cursor, min(entry_end - cursor, (NvU64)PAGE_SIZE));
            uvm_moe_slot_pool_log_va_block_find_failed(va_space,
                                                       entry,
                                                       mm,
                                                       cursor,
                                                       entry_end - cursor,
                                                       status,
                                                       range_type);
            break;
        }

        if (cursor > block->end) {
            status = NV_ERR_INVALID_ARGUMENT;
            uvm_phase_f_route_log("moe_slot_va_block_install_cursor_after_block layer=%u expert=%u slot=%u cursor=0x%llx block=[0x%llx,0x%llx]\n",
                                  entry->layer_id,
                                  entry->expert_id,
                                  entry->slot_id,
                                  cursor,
                                  block->start,
                                  block->end);
            break;
        }

        slice_end = min(entry_end, block->end + 1);
        if (block_context->hmm.vma)
            slice_end = min(slice_end, (NvU64)block_context->hmm.vma->vm_end);
        if (slice_end <= cursor) {
            status = NV_ERR_INVALID_ARGUMENT;
            uvm_phase_f_route_log("moe_slot_va_block_install_empty_slice layer=%u expert=%u slot=%u cursor=0x%llx block=[0x%llx,0x%llx] entry_end=0x%llx\n",
                                  entry->layer_id,
                                  entry->expert_id,
                                  entry->slot_id,
                                  cursor,
                                  block->start,
                                  block->end,
                                  entry_end);
            break;
        }

        slice.va_base = cursor;
        slice.bytes = slice_end - cursor;
        slice.slot_offset = entry->slot_offset + (cursor - entry->va_base);

        uvm_phase_f_route_log("moe_pte_range action=install_va_block_slice_begin layer=%u expert=%u slot=%u role=%u slice_va=0x%llx slice_end=0x%llx slice_bytes=%llu slot_offset=%llu block_start=0x%llx block_end=0x%llx vma=%px\n",
                              entry->layer_id,
                              entry->expert_id,
                              entry->slot_id,
                              entry->role,
                              slice.va_base,
                              slice_end,
                              slice.bytes,
                              slice.slot_offset,
                              block->start,
                              block->end,
                              block_context->hmm.vma);

        status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                         &block_retry,
                                         uvm_moe_slot_pool_unmap_va_block_gpu_locked(block,
                                                                                     block_context,
                                                                                     &slice,
                                                                                     gpu));
        if (status != NV_OK) {
            uvm_phase_f_route_log("moe_slot_va_block_install_unmap_failed layer=%u expert=%u slot=%u slice_va=0x%llx slice_bytes=%llu block=[0x%llx,0x%llx] status=0x%x\n",
                                  entry->layer_id,
                                  entry->expert_id,
                                  entry->slot_id,
                                  slice.va_base,
                                  slice.bytes,
                                  block->start,
                                  block->end,
                                  status);
            break;
        }

        if (block_unmap_count_out)
            ++*block_unmap_count_out;

        status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                         &block_retry,
                                         uvm_moe_slot_pool_map_va_block_slot_locked(block,
                                                                                   block_context,
                                                                                   &slice,
                                                                                   gpu,
                                                                                   slot_mem));
        if (status != NV_OK) {
            uvm_phase_f_route_log("moe_slot_va_block_install_map_failed layer=%u expert=%u slot=%u slice_va=0x%llx slice_bytes=%llu slot_offset=%llu block=[0x%llx,0x%llx] status=0x%x\n",
                                  entry->layer_id,
                                  entry->expert_id,
                                  entry->slot_id,
                                  slice.va_base,
                                  slice.bytes,
                                  slice.slot_offset,
                                  block->start,
                                  block->end,
                                  status);
            break;
        }

        installed_bytes += slice.bytes;

        uvm_mutex_lock(&block->lock);
        status = uvm_tracker_init_from(&local_tracker, &block->tracker);
        uvm_mutex_unlock(&block->lock);
        if (status != NV_OK)
            break;

        status = uvm_tracker_wait_deinit(&local_tracker);
        if (status != NV_OK)
            break;

        if (tracker_wait_count_out)
            ++*tracker_wait_count_out;

        uvm_phase_f_route_log("moe_pte_range action=install_va_block_slice_end layer=%u expert=%u slot=%u role=%u slice_va=0x%llx slice_end=0x%llx slice_bytes=%llu installed_bytes=%llu status=0x%x\n",
                              entry->layer_id,
                              entry->expert_id,
                              entry->slot_id,
                              entry->role,
                              slice.va_base,
                              slice_end,
                              slice.bytes,
                              installed_bytes,
                              status);

        cursor = slice_end;
    }

    uvm_va_block_context_free(block_context);

    if (status != NV_OK && installed_bytes != 0) {
        NvU64 rollback_block_unmap_count = 0;
        NvU64 rollback_tracker_wait_count = 0;
        NV_STATUS rollback_status;

        rollback_status = uvm_moe_slot_pool_uninstall_va_block_slot_mapping(va_space,
                                                                            gpu,
                                                                            mm,
                                                                            entry->va_base,
                                                                            installed_bytes,
                                                                            &rollback_block_unmap_count,
                                                                            &rollback_tracker_wait_count);
        if (rollback_status == NV_OK) {
            if (rollback_bytes_out)
                *rollback_bytes_out = installed_bytes;
            if (rollback_block_unmap_count_out)
                *rollback_block_unmap_count_out = rollback_block_unmap_count;
            if (rollback_tracker_wait_count_out)
                *rollback_tracker_wait_count_out = rollback_tracker_wait_count;
        }
        else {
            UVM_ERR_PRINT("MoE slot partial VA-block rollback failed: %s, VA [0x%llx, 0x%llx)\n",
                          nvstatusToString(rollback_status),
                          entry->va_base,
                          entry->va_base + installed_bytes);
        }
    }

    uvm_phase_f_route_log("moe_pte_range action=install_va_block_end layer=%u expert=%u slot=%u role=%u va=0x%llx end=0x%llx bytes=%llu installed=%llu status=0x%x\n",
                          entry->layer_id,
                          entry->expert_id,
                          entry->slot_id,
                          entry->role,
                          entry->va_base,
                          entry_end,
                          entry->bytes,
                          installed_bytes,
                          status);

    return status;
}

static NV_STATUS uvm_moe_slot_pool_uninstall_va_block_slot_mapping_mode(
    uvm_va_space_t *va_space,
    uvm_gpu_t *gpu,
    struct mm_struct *mm,
    NvU64 va_base,
    NvU64 bytes,
    NvU64 *block_unmap_count_out,
    NvU64 *tracker_wait_count_out,
    NvBool restore_cpu_mapping)
{
    NV_STATUS status = NV_OK;
    uvm_va_block_context_t *block_context;
    NvU64 va_end = va_base + bytes;
    NvU64 cursor = va_base;

    if (block_unmap_count_out)
        *block_unmap_count_out = 0;
    if (tracker_wait_count_out)
        *tracker_wait_count_out = 0;

    if (!va_space || !gpu || bytes == 0)
        return NV_ERR_INVALID_ARGUMENT;

    if (va_end <= va_base)
        return NV_ERR_INVALID_ARGUMENT;

    uvm_phase_f_route_log("moe_pte_range action=uninstall_va_block_begin va=0x%llx end=0x%llx bytes=%llu va_space=%px mm=%px gpu=%s\n",
                          va_base,
                          va_end,
                          bytes,
                          va_space,
                          mm,
                          uvm_gpu_name(gpu));

    block_context = uvm_va_block_context_alloc(mm);
    if (!block_context)
        return NV_ERR_NO_MEMORY;

    while (cursor < va_end) {
        uvm_va_block_t *block = NULL;
        uvm_va_block_retry_t block_retry;
        uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
        NvU64 slice_end;

        block_context->hmm.vma = NULL;
        status = mm ?
                    uvm_va_block_find_create(va_space, cursor, &block_context->hmm.vma, &block) :
                    uvm_va_block_find(va_space, cursor, &block);
        if (status == NV_ERR_OBJECT_NOT_FOUND || status == NV_ERR_INVALID_ADDRESS) {
            NvU64 next_block = UVM_VA_BLOCK_ALIGN_UP(cursor + 1);

            uvm_phase_f_route_log("moe_slot_uninstall_skip va=0x%llx next=0x%llx end=0x%llx status=0x%x mm=%px\n",
                                  cursor,
                                  next_block,
                                  va_end,
                                  status,
                                  mm);
            cursor = min(va_end, next_block);
            status = NV_OK;
            continue;
        }
        if (status != NV_OK)
            break;

        if (cursor > block->end) {
            status = NV_ERR_INVALID_ARGUMENT;
            break;
        }

        slice_end = min(va_end, block->end + 1);
        if (block_context->hmm.vma)
            slice_end = min(slice_end, (NvU64)block_context->hmm.vma->vm_end);
        if (slice_end <= cursor) {
            status = NV_ERR_INVALID_ARGUMENT;
            break;
        }

        uvm_phase_f_route_log("moe_pte_range action=uninstall_va_block_slice_begin block_start=0x%llx block_end=0x%llx slice_va=0x%llx slice_end=0x%llx bytes=%llu mm=%px vma=%px\n",
                              block->start,
                              block->end,
                              cursor,
                              slice_end,
                              slice_end - cursor,
                              mm,
                              block_context->hmm.vma);

        status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                         &block_retry,
                                         uvm_moe_slot_pool_unmap_va_block_slot_locked(block,
                                                                                      block_context,
                                                                                      cursor,
                                                                                      slice_end - cursor,
                                                                                      gpu,
                                                                                      restore_cpu_mapping));
        uvm_phase_f_route_log("moe_pte_range action=uninstall_va_block_slice_end block_start=0x%llx block_end=0x%llx slice_va=0x%llx slice_end=0x%llx status=0x%x\n",
                              block->start,
                              block->end,
                              cursor,
                              slice_end,
                              status);
        if (status != NV_OK)
            break;

        if (block_unmap_count_out)
            ++*block_unmap_count_out;

        uvm_mutex_lock(&block->lock);
        status = uvm_tracker_init_from(&local_tracker, &block->tracker);
        uvm_mutex_unlock(&block->lock);
        if (status != NV_OK)
            break;

        status = uvm_tracker_wait_deinit(&local_tracker);
        if (status != NV_OK)
            break;

        if (tracker_wait_count_out)
            ++*tracker_wait_count_out;

        cursor = slice_end;
    }

    uvm_va_block_context_free(block_context);
    uvm_phase_f_route_log("moe_pte_range action=uninstall_va_block_end va=0x%llx end=0x%llx bytes=%llu status=0x%x block_unmaps=%llu tracker_waits=%llu\n",
                          va_base,
                          va_end,
                          bytes,
                          status,
                          block_unmap_count_out ? *block_unmap_count_out : 0,
                          tracker_wait_count_out ? *tracker_wait_count_out : 0);
    return status;
}

static NV_STATUS uvm_moe_slot_pool_uninstall_va_block_slot_mapping(
    uvm_va_space_t *va_space,
    uvm_gpu_t *gpu,
    struct mm_struct *mm,
    NvU64 va_base,
    NvU64 bytes,
    NvU64 *block_unmap_count_out,
    NvU64 *tracker_wait_count_out)
{
    return uvm_moe_slot_pool_uninstall_va_block_slot_mapping_mode(va_space,
                                                                   gpu,
                                                                   mm,
                                                                   va_base,
                                                                   bytes,
                                                                   block_unmap_count_out,
                                                                   tracker_wait_count_out,
                                                                   NV_TRUE);
}

static NV_STATUS uvm_moe_slot_pool_map_va_block_user_backing_locked(uvm_va_block_t *target_block,
                                                                    uvm_va_block_t *backing_block,
                                                                    uvm_va_block_context_t *block_context,
                                                                    const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                                    uvm_gpu_t *gpu,
                                                                    NvBool defer_tlb_invalidate)
{
    uvm_va_block_region_t region = uvm_va_block_region_from_start_size(target_block, entry->va_base, entry->bytes);
    uvm_page_mask_t *page_mask = &block_context->caller_page_mask;

    uvm_page_mask_init_from_region(page_mask, region, NULL);
    return uvm_va_block_map_moe_user_backing_readonly(target_block,
                                                      block_context,
                                                      gpu,
                                                      region,
                                                      page_mask,
                                                      backing_block,
                                                      entry->slot_offset,
                                                      defer_tlb_invalidate,
                                                      &target_block->tracker);
}

static NV_STATUS uvm_moe_slot_pool_install_va_block_user_backing_mapping(uvm_va_space_t *va_space,
                                                                         const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                                         uvm_gpu_t *gpu,
                                                                         struct mm_struct *mm,
                                                                         NvBool defer_tlb_invalidate,
                                                                         NvU64 *block_unmap_count_out,
                                                                         NvU64 *tracker_wait_count_out)
{
    NV_STATUS status;
    uvm_va_block_context_t *block_context;
    uvm_va_block_t *target_block = NULL;
    uvm_va_block_t *backing_block = NULL;
    uvm_va_block_retry_t target_retry;
    uvm_va_block_retry_t backing_retry;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
    NvU64 entry_end = entry->va_base + entry->bytes - 1;
    NvU64 backing_end = entry->slot_offset + entry->bytes - 1;

    if (block_unmap_count_out)
        *block_unmap_count_out = 0;
    if (tracker_wait_count_out)
        *tracker_wait_count_out = 0;

    if (!va_space || !gpu || entry_end < entry->va_base || backing_end < entry->slot_offset)
        return NV_ERR_INVALID_ARGUMENT;

    block_context = uvm_va_block_context_alloc(mm);
    if (!block_context)
        return NV_ERR_NO_MEMORY;

    status = uvm_va_block_find_create_managed(va_space, entry->va_base, &target_block);
    if (status != NV_OK)
        goto out;

    if (entry_end > target_block->end) {
        status = NV_ERR_INVALID_ARGUMENT;
        goto out;
    }

    status = uvm_va_block_find_create_managed(va_space, entry->slot_offset, &backing_block);
    if (status != NV_OK)
        goto out;

    if (backing_end > backing_block->end) {
        status = NV_ERR_INVALID_ARGUMENT;
        goto out;
    }

    status = UVM_VA_BLOCK_LOCK_RETRY(target_block,
                                     &target_retry,
                                     uvm_moe_slot_pool_unmap_va_block_gpu_locked(target_block,
                                                                                 block_context,
                                                                                 entry,
                                                                                 gpu));
    if (status != NV_OK)
        goto out;

    if (block_unmap_count_out)
        *block_unmap_count_out = 1;

    uvm_va_block_retry_init(&target_retry);
    uvm_va_block_retry_init(&backing_retry);

    // Lock in address order to avoid deadlock if target/backing blocks differ.
    if (target_block == backing_block) {
        uvm_mutex_lock(&target_block->lock);
    }
    else if (target_block->start < backing_block->start) {
        uvm_mutex_lock(&target_block->lock);
        uvm_mutex_lock(&backing_block->lock);
    }
    else {
        uvm_mutex_lock(&backing_block->lock);
        uvm_mutex_lock(&target_block->lock);
    }

    do {
        status = uvm_moe_slot_pool_map_va_block_user_backing_locked(target_block,
                                                                    backing_block,
                                                                    block_context,
                                                                    entry,
                                                                    gpu,
                                                                    defer_tlb_invalidate);
    } while (status == NV_ERR_MORE_PROCESSING_REQUIRED);

    if (target_block != backing_block) {
        uvm_mutex_unlock(&backing_block->lock);
        uvm_mutex_unlock(&target_block->lock);
    }
    else {
        uvm_mutex_unlock(&target_block->lock);
    }

    uvm_va_block_retry_deinit(&backing_retry, backing_block);
    uvm_va_block_retry_deinit(&target_retry, target_block);

    if (status != NV_OK)
        goto out;

    uvm_mutex_lock(&target_block->lock);
    status = uvm_tracker_init_from(&local_tracker, &target_block->tracker);
    uvm_mutex_unlock(&target_block->lock);
    if (status != NV_OK)
        goto out;

    status = uvm_tracker_wait_deinit(&local_tracker);
    if (status != NV_OK)
        goto out;

    if (tracker_wait_count_out)
        *tracker_wait_count_out = 1;

out:
    uvm_va_block_context_free(block_context);
    return status;
}

static NvBool uvm_moe_slot_pool_user_backing_entry_in_block(uvm_va_block_t *target_block,
                                                            const UVM_MOE_SLOT_REMAP_ENTRY *entry)
{
    NvU64 entry_end = entry->va_base + entry->bytes - 1;

    return entry_end >= entry->va_base &&
           entry->va_base >= target_block->start &&
           entry_end <= target_block->end;
}

static NV_STATUS uvm_moe_slot_pool_unmap_va_block_gpu_mask_locked(uvm_va_block_t *block,
                                                                  uvm_va_block_context_t *block_context,
                                                                  uvm_gpu_t *gpu,
                                                                  const uvm_page_mask_t *page_mask)
{
    return uvm_va_block_unmap(block,
                              block_context,
                              gpu->id,
                              uvm_va_block_region_from_block(block),
                              page_mask,
                              &block->tracker);
}

static NV_STATUS uvm_moe_slot_pool_map_one_user_backing_slice(uvm_va_space_t *va_space,
                                                              uvm_va_block_t *target_block,
                                                              uvm_va_block_context_t *block_context,
                                                              const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                              uvm_gpu_t *gpu,
                                                              NvBool defer_tlb_invalidate)
{
    NV_STATUS status;
    uvm_va_block_t *backing_block = NULL;
    uvm_va_block_retry_t target_retry;
    uvm_va_block_retry_t backing_retry;
    NvU64 backing_end = entry->slot_offset + entry->bytes - 1;

    if (backing_end < entry->slot_offset)
        return NV_ERR_INVALID_ARGUMENT;

    status = uvm_va_block_find_create_managed(va_space, entry->slot_offset, &backing_block);
    if (status != NV_OK)
        return status;

    if (backing_end > backing_block->end)
        return NV_ERR_INVALID_ARGUMENT;

    uvm_va_block_retry_init(&target_retry);
    uvm_va_block_retry_init(&backing_retry);

    if (target_block == backing_block) {
        uvm_mutex_lock(&target_block->lock);
    }
    else if (target_block->start < backing_block->start) {
        uvm_mutex_lock(&target_block->lock);
        uvm_mutex_lock(&backing_block->lock);
    }
    else {
        uvm_mutex_lock(&backing_block->lock);
        uvm_mutex_lock(&target_block->lock);
    }

    do {
        status = uvm_moe_slot_pool_map_va_block_user_backing_locked(target_block,
                                                                    backing_block,
                                                                    block_context,
                                                                    entry,
                                                                    gpu,
                                                                    defer_tlb_invalidate);
    } while (status == NV_ERR_MORE_PROCESSING_REQUIRED);

    if (target_block != backing_block) {
        uvm_mutex_unlock(&backing_block->lock);
        uvm_mutex_unlock(&target_block->lock);
    }
    else {
        uvm_mutex_unlock(&target_block->lock);
    }

    uvm_va_block_retry_deinit(&backing_retry, backing_block);
    uvm_va_block_retry_deinit(&target_retry, target_block);

    return status;
}

static NV_STATUS uvm_moe_slot_pool_install_user_backing_group(uvm_va_space_t *va_space,
                                                              UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
                                                              NvU32 current_index,
                                                              uvm_gpu_t *gpu,
                                                              struct mm_struct *mm,
                                                              NvBool defer_tlb_invalidate,
                                                              uvm_moe_slot_pool_user_backing_group_state_t *group_state,
                                                              NvU64 *block_unmap_count_out,
                                                              NvU64 *tracker_wait_count_out)
{
    NV_STATUS status = NV_OK;
    uvm_va_block_context_t *block_context = NULL;
    uvm_va_block_t *target_block = NULL;
    uvm_va_block_retry_t target_retry;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
    uvm_page_mask_t *combined_mask;
    NvU32 *candidate_indices = NULL;
    NvU32 candidate_count = 0;
    NvU32 i;

    if (block_unmap_count_out)
        *block_unmap_count_out = 0;
    if (tracker_wait_count_out)
        *tracker_wait_count_out = 0;

    if (!group_state || group_state->installed[current_index])
        return NV_OK;

    status = uvm_va_block_find_create_managed(va_space, params->entries[current_index].va_base, &target_block);
    if (status != NV_OK)
        return status;

    if (!uvm_moe_slot_pool_user_backing_entry_in_block(target_block, &params->entries[current_index]))
        return NV_WARN_NOTHING_TO_DO;

    block_context = uvm_va_block_context_alloc(mm);
    if (!block_context)
        return NV_ERR_NO_MEMORY;

    candidate_indices = uvm_kvmalloc(params->count * sizeof(*candidate_indices));
    if (!candidate_indices) {
        status = NV_ERR_NO_MEMORY;
        goto out;
    }

    combined_mask = &block_context->caller_page_mask;
    uvm_page_mask_zero(combined_mask);

    for (i = current_index; i < params->count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[i];
        uvm_moe_slot_pool_slot_t *slot;
        uvm_va_block_region_t region;

        if (entry->status != UVM_MOE_SLOT_REMAP_ENTRY_OK || group_state->installed[i])
            continue;

        if (!uvm_moe_slot_pool_user_backing_entry_in_block(target_block, entry))
            continue;

        slot = uvm_moe_slot_pool_find_locked(va_space, entry->layer_id, entry->slot_id);
        if (!slot || slot->owner_expert_id != entry->expert_id)
            continue;

        if (slot->pte_range_count >= UVM_MOE_SLOT_POOL_MAX_PTE_RANGES_PER_SLOT)
            continue;

        if (slot->pte_bytes != 0 &&
            (slot->pte_va_space != va_space ||
             slot->pte_gpu != gpu ||
             slot->pte_mode != UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_USER_BACKING_REMAP ||
             slot->batch_id != g_uvm_moe_slot_pool_current_batch_id))
            continue;

        region = uvm_va_block_region_from_start_size(target_block, entry->va_base, entry->bytes);
        uvm_page_mask_region_fill(combined_mask, region);
        candidate_indices[candidate_count++] = i;
    }

    if (candidate_count <= 1 || candidate_indices[0] != current_index) {
        status = NV_WARN_NOTHING_TO_DO;
        goto out;
    }

    status = UVM_VA_BLOCK_LOCK_RETRY(target_block,
                                     &target_retry,
                                     uvm_moe_slot_pool_unmap_va_block_gpu_mask_locked(target_block,
                                                                                      block_context,
                                                                                      gpu,
                                                                                      combined_mask));
    if (status != NV_OK)
        goto out;

    for (i = 0; i < candidate_count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[candidate_indices[i]];

        status = uvm_moe_slot_pool_map_one_user_backing_slice(va_space,
                                                              target_block,
                                                              block_context,
                                                              entry,
                                                              gpu,
                                                              defer_tlb_invalidate);
        if (status != NV_OK)
            goto rollback;
    }

    uvm_mutex_lock(&target_block->lock);
    status = uvm_tracker_init_from(&local_tracker, &target_block->tracker);
    uvm_mutex_unlock(&target_block->lock);
    if (status != NV_OK)
        goto rollback;

    status = uvm_tracker_wait_deinit(&local_tracker);
    if (status != NV_OK)
        goto rollback;

    for (i = 0; i < candidate_count; ++i)
        group_state->installed[candidate_indices[i]] = NV_TRUE;

    group_state->block_unmap_count[current_index] = 1;
    group_state->tracker_wait_count[current_index] = 1;

    if (block_unmap_count_out)
        *block_unmap_count_out = 1;
    if (tracker_wait_count_out)
        *tracker_wait_count_out = 1;

    goto out;

rollback:
    for (i = 0; i < candidate_count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[candidate_indices[i]];

        (void)uvm_moe_slot_pool_uninstall_va_block_slot_mapping(va_space,
                                                                gpu,
                                                                mm,
                                                                entry->va_base,
                                                                entry->bytes,
                                                                NULL,
                                                                NULL);
    }

out:
    uvm_kvfree(candidate_indices);
    uvm_va_block_context_free(block_context);
    return status;
}

static NV_STATUS uvm_moe_slot_pool_user_backing_batch_tlb_invalidate(uvm_va_space_t *va_space, uvm_gpu_t *gpu)
{
    NV_STATUS status;
    uvm_gpu_va_space_t *gpu_va_space;
    uvm_page_tree_t *tree;
    uvm_push_t push;

    if (!va_space || !gpu)
        return NV_ERR_INVALID_ARGUMENT;

    gpu_va_space = va_space->gpu_va_spaces[uvm_id_gpu_index(gpu->id)];
    if (!gpu_va_space || uvm_gpu_va_space_state(gpu_va_space) != UVM_GPU_VA_SPACE_STATE_ACTIVE)
        return NV_ERR_INVALID_DEVICE;

    tree = &gpu_va_space->page_tables;
    status = uvm_push_begin(gpu->channel_manager,
                            UVM_CHANNEL_TYPE_MEMOPS,
                            &push,
                            "Batch TLB invalidate for MoE user backing remap");
    if (status != NV_OK)
        return status;

    gpu->parent->host_hal->tlb_invalidate_all(&push,
                                              uvm_page_tree_pdb_address(tree),
                                              tree->hal->page_table_depth(UVM_PAGE_SIZE_4K),
                                              UVM_MEMBAR_SYS);

    return uvm_push_end_and_wait(&push);
}

static NV_STATUS uvm_moe_slot_pool_apply_entry_locked(UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                      NvU64 batch_id,
                                                      NvU64 epoch,
                                                      uvm_moe_slot_pool_copy_mode_t copy_mode,
                                                      uvm_gpu_t *gpu,
                                                      uvm_va_space_t *locked_va_space,
                                                      struct mm_struct *mm,
                                                      NvBool *vidmem_reused_out)
{
    uvm_moe_slot_pool_slot_t *slot;
    NV_STATUS status = NV_OK;
    NvBool new_slot = NV_FALSE;
    NvBool overwrite = NV_FALSE;
    NvBool refresh = NV_FALSE;
    NvBool same_owner;
    NvBool vidmem_reused = NV_FALSE;
    NvBool reuse_required = copy_mode == UVM_MOE_SLOT_POOL_PREALLOC_VIDMEM &&
                            (entry->flags & UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED);
    NvU64 required_bytes = entry->slot_offset + entry->bytes;

    if (vidmem_reused_out)
        *vidmem_reused_out = NV_FALSE;

    slot = uvm_moe_slot_pool_find_locked(locked_va_space, entry->layer_id, entry->slot_id);
    if (!slot && reuse_required) {
        entry->status = UVM_MOE_SLOT_REMAP_ENTRY_BACKING_ALLOC_FAILED;
        return NV_ERR_OBJECT_NOT_FOUND;
    }

    if (!slot) {
        slot = uvm_moe_slot_pool_find_free_locked();
        if (!slot) {
            ++g_uvm_moe_slot_pool_full_count;
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_BACKING_ALLOC_FAILED;
            return NV_ERR_NO_MEMORY;
        }
        new_slot = NV_TRUE;
    }

    if (copy_mode == UVM_MOE_SLOT_POOL_PREALLOC_VIDMEM) {
        vidmem_reused = uvm_moe_slot_pool_slot_has_vidmem_backing(slot, gpu) &&
                        slot->slot_bytes >= required_bytes;
        if (reuse_required && !vidmem_reused) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_BACKING_ALLOC_FAILED;
            return NV_ERR_INVALID_STATE;
        }
    }

    same_owner = slot->valid && slot->owner_expert_id == entry->expert_id;
    overwrite = slot->valid && !same_owner;
    refresh = !slot->valid || !same_owner || slot->batch_id != batch_id;

    if (new_slot || refresh) {
        status = uvm_moe_slot_pool_clear_metadata_keep_backing_locked(slot, locked_va_space, mm);
        if (status != NV_OK) {
            entry->status = copy_mode == UVM_MOE_SLOT_POOL_COPY_NONE ?
                                UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED :
                                UVM_MOE_SLOT_REMAP_ENTRY_COPY_FAILED;
            slot->state = UVM_MOE_SLOT_POOL_STATE_ERROR;
            slot->last_status = entry->status;
            slot->pte_status = entry->status;
            return status;
        }

        slot->valid = NV_TRUE;
        slot->owner_va_space = locked_va_space;
        slot->layer_id = entry->layer_id;
        slot->slot_id = entry->slot_id;
        slot->owner_expert_id = entry->expert_id;
        slot->state = UVM_MOE_SLOT_POOL_STATE_READY;
        slot->first_va_base = entry->va_base;
        if (overwrite)
            ++g_uvm_moe_slot_pool_overwrite_count;
    }
    else if (!slot->valid) {
        slot->valid = NV_TRUE;
        slot->owner_va_space = locked_va_space;
        slot->layer_id = entry->layer_id;
        slot->slot_id = entry->slot_id;
        slot->owner_expert_id = entry->expert_id;
        slot->state = UVM_MOE_SLOT_POOL_STATE_READY;
        slot->first_va_base = entry->va_base;
    }

    // Publish lifetime policy before allocation so concurrent budget decisions
    // made under this same pool lock observe the final eviction contract.
    slot->flags = entry->flags & UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT;

    if (copy_mode == UVM_MOE_SLOT_POOL_PREALLOC_VIDMEM) {
        status = uvm_moe_slot_pool_ensure_vidmem_backing_locked(slot,
                                                                gpu,
                                                                entry->slot_offset + entry->bytes,
                                                                locked_va_space,
                                                                mm,
                                                                !(entry->flags & UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT));
        if (status != NV_OK) {
            if (status == NV_ERR_NO_MEMORY)
                entry->status = UVM_MOE_SLOT_REMAP_ENTRY_BACKING_ALLOC_FAILED;
            else
                entry->status = UVM_MOE_SLOT_REMAP_ENTRY_COPY_FAILED;

            slot->state = UVM_MOE_SLOT_POOL_STATE_ERROR;
            slot->last_role = entry->role;
            slot->flags = entry->flags & UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT;
            slot->last_status = entry->status;
            slot->copy_status = entry->status;
            slot->epoch = epoch;
            slot->batch_id = batch_id;
            slot->last_va_base = entry->va_base;
            slot->last_bytes = entry->bytes;
            return status;
        }

        slot->copy_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;

        if (vidmem_reused) {
            ++g_uvm_moe_slot_pool_vidmem_rebind_count;
            g_uvm_moe_slot_pool_vidmem_rebind_bytes += entry->bytes;
            if (vidmem_reused_out)
                *vidmem_reused_out = NV_TRUE;
            uvm_phase_f_route_log("moe_slot_pool_vidmem_rebind layer=%u expert=%u slot=%u bytes=%llu capacity=%llu required=%u total=%llu total_bytes=%llu\n",
                                  entry->layer_id,
                                  entry->expert_id,
                                  entry->slot_id,
                                  entry->bytes,
                                  slot->slot_bytes,
                                  reuse_required,
                                  g_uvm_moe_slot_pool_vidmem_rebind_count,
                                  g_uvm_moe_slot_pool_vidmem_rebind_bytes);
        }
    }

    if (copy_mode != UVM_MOE_SLOT_POOL_COPY_NONE &&
        copy_mode != UVM_MOE_SLOT_POOL_PREALLOC_VIDMEM) {
        status = uvm_moe_slot_pool_copy_entry_locked(slot, entry, copy_mode, gpu, locked_va_space, mm);
        if (status != NV_OK) {
            if (status == NV_ERR_NO_MEMORY)
                entry->status = UVM_MOE_SLOT_REMAP_ENTRY_BACKING_ALLOC_FAILED;
            else
                entry->status = UVM_MOE_SLOT_REMAP_ENTRY_COPY_FAILED;

            slot->state = UVM_MOE_SLOT_POOL_STATE_ERROR;
            slot->last_role = entry->role;
            slot->flags = entry->flags & UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT;
            slot->last_status = entry->status;
            slot->copy_status = entry->status;
            slot->epoch = epoch;
            slot->batch_id = batch_id;
            slot->last_va_base = entry->va_base;
            slot->last_bytes = entry->bytes;
            slot->copy_failed_count++;
            g_uvm_moe_slot_pool_copy_failed_count++;
            return status;
        }

        if (copy_mode != UVM_MOE_SLOT_POOL_COPY_GPU_ASYNC) {
            slot->copy_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
            slot->copy_count++;
            slot->copy_bytes += entry->bytes;
            g_uvm_moe_slot_pool_copy_count++;
            g_uvm_moe_slot_pool_copy_bytes += entry->bytes;
        }
    }

    slot->range_count++;
    slot->last_role = entry->role;
    slot->flags = entry->flags & UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT;
    slot->last_status = entry->status;
    slot->epoch = epoch;
    slot->batch_id = batch_id;
    slot->last_va_base = entry->va_base;
    slot->last_bytes = entry->bytes;
    slot->total_bytes += entry->bytes;
    ++g_uvm_moe_slot_pool_range_update_count;

    return NV_OK;
}

static void uvm_moe_slot_remap_fail_ok_entries(UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
                                               NvU32 entry_status);

static NV_STATUS uvm_moe_slot_pool_apply_batch(UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
                                               uvm_moe_slot_pool_copy_mode_t copy_mode,
                                               uvm_gpu_t *gpu,
                                               uvm_va_space_t *va_space)
{
    NV_STATUS first_error = NV_OK;
    NvU64 batch_id;
    NvU32 i;
    NvBool has_candidate = NV_FALSE;
    struct mm_struct *mm = NULL;

    for (i = 0; i < params->count; ++i) {
        if (params->entries[i].status == UVM_MOE_SLOT_REMAP_ENTRY_OK) {
            has_candidate = NV_TRUE;
            break;
        }
    }

    if (!has_candidate)
        return NV_OK;

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_write(va_space);

    // PREALLOC uses the VA-space fast retain. Close the unregister race under
    // the write lock before allocating any backing or mutating slot metadata.
    if (copy_mode == UVM_MOE_SLOT_POOL_PREALLOC_VIDMEM &&
        (!gpu || uvm_va_space_get_gpu_by_uuid_with_gpu_va_space(va_space, &params->gpu_uuid) != gpu)) {
        uvm_va_space_up_write(va_space);
        uvm_va_space_mm_or_current_release_unlock(va_space, mm);
        uvm_moe_slot_remap_fail_ok_entries(params, UVM_MOE_SLOT_REMAP_ENTRY_COPY_FAILED);
        return NV_ERR_INVALID_DEVICE;
    }

    mutex_lock(&g_uvm_moe_slot_pool_lock);
    batch_id = ++g_uvm_moe_slot_pool_current_batch_id;
    for (i = 0; i < params->count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[i];
        NvU64 epoch;
        NV_STATUS status;
        NvBool vidmem_reused = NV_FALSE;

        if (entry->status != UVM_MOE_SLOT_REMAP_ENTRY_OK)
            continue;

        epoch = g_uvm_moe_slot_pool_current_epoch + 1;
        status = uvm_moe_slot_pool_apply_entry_locked(entry,
                                                      batch_id,
                                                      epoch,
                                                      copy_mode,
                                                      gpu,
                                                      va_space,
                                                      mm,
                                                      &vidmem_reused);
        if (status == NV_OK) {
            g_uvm_moe_slot_pool_current_epoch = epoch;
            ++g_uvm_moe_slot_pool_validated_entries;
            ++params->success_count;
            params->accepted_bytes += entry->bytes;
            if (vidmem_reused)
                ++params->reserved;
        }
        else {
            ++params->failure_count;
            if (first_error == NV_OK)
                first_error = status;
        }
    }
    mutex_unlock(&g_uvm_moe_slot_pool_lock);

    uvm_va_space_up_write(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    return first_error;
}

static NV_STATUS uvm_moe_slot_pool_release_batch(UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
                                                 uvm_va_space_t *va_space,
                                                 NV_STATUS validation_status)
{
    NV_STATUS first_error = validation_status;
    struct mm_struct *mm;
    NvBool has_candidate = NV_FALSE;
    NvU32 i;

    for (i = 0; i < params->count; ++i) {
        if (params->entries[i].status == UVM_MOE_SLOT_REMAP_ENTRY_OK) {
            has_candidate = NV_TRUE;
            break;
        }
    }

    if (!has_candidate)
        return first_error;

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_write(va_space);
    mutex_lock(&g_uvm_moe_slot_pool_lock);

    for (i = 0; i < params->count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[i];
        uvm_moe_slot_pool_slot_t *slot;
        NvBool had_active_pte;
        NvU64 slot_bytes;
        NV_STATUS status;

        if (entry->status != UVM_MOE_SLOT_REMAP_ENTRY_OK)
            continue;

        slot = uvm_moe_slot_pool_find_locked(va_space, entry->layer_id, entry->slot_id);
        if (!slot) {
            if (uvm_moe_slot_pool_key_owned_by_other_locked(va_space,
                                                            entry->layer_id,
                                                            entry->slot_id)) {
                ++g_uvm_moe_slot_pool_owner_mismatch_count;
            }
            status = NV_ERR_OBJECT_NOT_FOUND;
        }
        else if (slot->owner_expert_id != entry->expert_id) {
            ++g_uvm_moe_slot_pool_owner_mismatch_count;
            status = NV_ERR_INVALID_STATE;
        }
        else {
            had_active_pte = uvm_moe_slot_pool_slot_has_active_pte(slot);
            slot_bytes = slot->slot_bytes;
            status = uvm_moe_slot_pool_release_backing_locked(slot, va_space, mm);
            if (status == NV_OK) {
                if (had_active_pte)
                    uvm_moe_slot_pool_note_pte_clear_reason_locked(slot,
                                                                   UVM_MOE_SLOT_PTE_CLEAR_REASON_RELEASE);

                uvm_phase_f_route_log("moe_slot_pool_release layer=%u expert=%u slot=%u bytes=%llu active_pte=%u owner=%px\n",
                                      entry->layer_id,
                                      entry->expert_id,
                                      entry->slot_id,
                                      slot_bytes,
                                      had_active_pte,
                                      va_space);
                memset(slot, 0, sizeof(*slot));
                ++params->success_count;
                params->accepted_bytes += slot_bytes;
                ++params->remap_count;
                if (had_active_pte)
                    ++params->tlb_invalidate_count;
                ++g_uvm_moe_slot_pool_release_count;
                g_uvm_moe_slot_pool_release_bytes += slot_bytes;
                continue;
            }

            slot->state = UVM_MOE_SLOT_POOL_STATE_ERROR;
            slot->last_status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
            uvm_moe_slot_pool_note_pte_failure_locked(slot);
        }

        entry->status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
        ++params->failure_count;
        ++g_uvm_moe_slot_pool_release_failed_count;
        if (first_error == NV_OK)
            first_error = status;

        uvm_phase_f_route_log("moe_slot_pool_release_failed layer=%u expert=%u slot=%u owner=%px status=0x%x failures=%llu owner_mismatches=%llu\n",
                              entry->layer_id,
                              entry->expert_id,
                              entry->slot_id,
                              va_space,
                              status,
                              g_uvm_moe_slot_pool_release_failed_count,
                              g_uvm_moe_slot_pool_owner_mismatch_count);
    }

    uvm_phase_f_route_log("moe_slot_pool_release_batch owner=%px success=%u failure=%u reclaimed_bytes=%llu total_releases=%llu total_bytes=%llu total_failures=%llu owner_mismatches=%llu\n",
                          va_space,
                          params->success_count,
                          params->failure_count,
                          params->accepted_bytes,
                          g_uvm_moe_slot_pool_release_count,
                          g_uvm_moe_slot_pool_release_bytes,
                          g_uvm_moe_slot_pool_release_failed_count,
                          g_uvm_moe_slot_pool_owner_mismatch_count);

    mutex_unlock(&g_uvm_moe_slot_pool_lock);
    uvm_va_space_up_write(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

    return first_error;
}

static NvU32 uvm_moe_slot_pte_status_from_nv_status(NV_STATUS status)
{
    if (status == NV_ERR_NOT_SUPPORTED)
        return UVM_MOE_SLOT_REMAP_ENTRY_REMAP_NOT_SUPPORTED;

    if (status == NV_ERR_INVALID_ARGUMENT || status == NV_ERR_OUT_OF_RANGE)
        return UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ARGUMENT;

    if (status == NV_ERR_INVALID_ADDRESS)
        return UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ADDRESS;

    return UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
}

static NV_STATUS uvm_moe_slot_pool_apply_slot_pte_batch(UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
                                                        uvm_va_space_t *va_space,
                                                        uvm_gpu_t *gpu,
                                                        NvBool use_va_block_path,
                                                        NvBool use_user_backing_path)
{
    NV_STATUS first_error = NV_OK;
    NvU64 batch_id;
    NvU32 i;
    NvBool has_candidate = NV_FALSE;
    struct mm_struct *mm = NULL;
    uvm_moe_slot_pool_pte_txn_t *txn = NULL;
    uvm_moe_slot_pool_user_backing_group_state_t *group_state = NULL;
    NvBool defer_user_backing_tlb_invalidate =
        use_user_backing_path && uvm_moe_slot_user_backing_defer_tlb_invalidate != 0;
    NvBool grouped_user_backing_remap =
        use_user_backing_path && uvm_moe_slot_user_backing_grouped_remap != 0;

    if (!uvm_moe_slot_pte_remap_enable) {
        uvm_moe_slot_remap_fail_ok_entries(params, UVM_MOE_SLOT_REMAP_ENTRY_REMAP_NOT_SUPPORTED);
        return NV_ERR_NOT_SUPPORTED;
    }

    if (!gpu)
        return NV_ERR_INVALID_DEVICE;

    for (i = 0; i < params->count; ++i) {
        if (params->entries[i].status == UVM_MOE_SLOT_REMAP_ENTRY_OK) {
            has_candidate = NV_TRUE;
            break;
        }
    }

    if (!has_candidate)
        return NV_OK;

    if (use_va_block_path) {
        txn = uvm_kvmalloc_zero(sizeof(*txn));
        if (!txn) {
            uvm_moe_slot_remap_fail_ok_entries(params, UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED);
            return NV_ERR_NO_MEMORY;
        }
    }

    if (grouped_user_backing_remap) {
        group_state = uvm_kvmalloc_zero(sizeof(*group_state));
        if (!group_state) {
            uvm_moe_slot_remap_fail_ok_entries(params, UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED);
            uvm_kvfree(txn);
            return NV_ERR_NO_MEMORY;
        }
    }

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_write(va_space);

    mutex_lock(&g_uvm_moe_slot_pool_lock);
    batch_id = ++g_uvm_moe_slot_pool_current_batch_id;


    for (i = 0; i < params->count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[i];
        uvm_moe_slot_pool_slot_t *slot;
        NV_STATUS status;
        NvU64 epoch;
        NvU64 block_unmap_count = 0;
        NvU64 tracker_wait_count = 0;
        NvU64 rollback_bytes = 0;
        NvU64 rollback_block_unmap_count = 0;
        NvU64 rollback_tracker_wait_count = 0;
        NvBool release_existing_mapping = NV_TRUE;

        if (entry->status != UVM_MOE_SLOT_REMAP_ENTRY_OK)
            continue;

        slot = uvm_moe_slot_pool_find_locked(va_space, entry->layer_id, entry->slot_id);
        if (!slot || (!slot->slot_mem && !use_user_backing_path)) {
            status = NV_ERR_OBJECT_NOT_FOUND;
        }
        else if (slot->owner_expert_id != entry->expert_id) {
            status = NV_ERR_INVALID_STATE;
        }
        else if (use_va_block_path) {
            release_existing_mapping = slot->pte_bytes != 0 &&
                                       (slot->pte_va_space != va_space ||
                                        slot->pte_gpu != gpu ||
                                        slot->pte_mode != (use_user_backing_path ?
                                                              UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_USER_BACKING_REMAP :
                                                              UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP) ||
                                        slot->batch_id != batch_id);

            if (group_state && !group_state->installed[i]) {
                status = uvm_moe_slot_pool_install_user_backing_group(va_space,
                                                                      params,
                                                                      i,
                                                                      gpu,
                                                                      mm,
                                                                      defer_user_backing_tlb_invalidate,
                                                                      group_state,
                                                                      &block_unmap_count,
                                                                      &tracker_wait_count);
                if (status == NV_WARN_NOTHING_TO_DO)
                    status = NV_OK;
            }
            else {
                status = NV_OK;
            }

            if (status == NV_OK && (!group_state || !group_state->installed[i])) {
                status = release_existing_mapping ?
                             uvm_moe_slot_pool_release_pte_mapping_locked(slot, va_space, mm) :
                             NV_OK;
            }
            if (status == NV_OK) {
                if (group_state && group_state->installed[i]) {
                    block_unmap_count = group_state->block_unmap_count[i];
                    tracker_wait_count = group_state->tracker_wait_count[i];
                }
                else if (use_user_backing_path) {
                    status = uvm_moe_slot_pool_install_va_block_user_backing_mapping(va_space,
	                                                                                    entry,
	                                                                                    gpu,
	                                                                                    mm,
	                                                                                    defer_user_backing_tlb_invalidate,
	                                                                                    &block_unmap_count,
	                                                                                    &tracker_wait_count);
                }
                else {
                    status = uvm_moe_slot_pool_install_va_block_slot_mapping(va_space,
                                                                             entry,
                                                                             gpu,
                                                                             mm,
                                                                             slot->slot_mem,
                                                                             &block_unmap_count,
                                                                             &tracker_wait_count,
                                                                             &rollback_bytes,
                                                                             &rollback_block_unmap_count,
                                                                             &rollback_tracker_wait_count);
                }
            }
            if (status == NV_OK) {
                if (slot->pte_gpu != gpu) {
                    slot->pte_gpu = gpu;
                    uvm_gpu_retain(gpu);
                }
                slot->pte_va_space = va_space;
                slot->pte_slot_offset = entry->slot_offset;
                slot->pte_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
                slot->pte_mode = use_user_backing_path ?
                                      UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_USER_BACKING_REMAP :
                                      UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP;
                slot->pte_va_block_tracked = NV_TRUE;
                status = uvm_moe_slot_pool_record_pte_range_locked(slot, entry->va_base, entry->bytes);
                if (status != NV_OK) {
                    (void)uvm_moe_slot_pool_uninstall_va_block_slot_mapping(va_space,
                                                                            gpu,
                                                                            mm,
                                                                            entry->va_base,
                                                                            entry->bytes,
                                                                            NULL,
                                                                            NULL);
                }
            }

            if (status == NV_OK) {
                slot->pte_remap_count++;
                slot->pte_remap_bytes += entry->bytes;
                ++g_uvm_moe_slot_pool_pte_remap_count;
                g_uvm_moe_slot_pool_pte_remap_bytes += entry->bytes;
                g_uvm_moe_slot_pool_pte_block_unmap_count += block_unmap_count;
                g_uvm_moe_slot_pool_pte_block_tracker_wait_count += tracker_wait_count;
            }
            else if (rollback_bytes != 0) {
                uvm_moe_slot_pool_note_partial_rollback_locked(slot,
                                                               rollback_bytes,
                                                               rollback_block_unmap_count,
                                                               rollback_tracker_wait_count);
            }
        }
        else {
            status = uvm_moe_slot_pool_install_pte_mapping_locked(
                va_space,
                slot,
                entry,
                gpu,
                mm,
                UVM_MOE_SLOT_REMAP_MODE_SLOT_PTE_REMAP);
        }

        if (status == NV_OK) {
            UVM_ASSERT(slot);
            epoch = g_uvm_moe_slot_pool_current_epoch + 1;
            slot->range_count++;
            slot->last_role = entry->role;
            slot->flags = entry->flags;
            slot->last_status = entry->status;
            slot->epoch = epoch;
            slot->batch_id = batch_id;
            slot->last_va_base = entry->va_base;
            slot->last_bytes = entry->bytes;
            slot->total_bytes += entry->bytes;
            ++g_uvm_moe_slot_pool_range_update_count;
            g_uvm_moe_slot_pool_current_epoch = epoch;
            ++g_uvm_moe_slot_pool_validated_entries;
            ++params->success_count;
            params->accepted_bytes += entry->bytes;
            ++params->remap_count;
            if (!defer_user_backing_tlb_invalidate)
                ++params->tlb_invalidate_count;
            if (use_va_block_path) {
                slot->pte_block_unmap_count += block_unmap_count;
                slot->pte_block_tracker_wait_count += tracker_wait_count;
                txn->installed_slots[txn->installed_count] = slot;
                txn->installed_indices[txn->installed_count] = i;
                ++txn->installed_count;
            }
        }
        else {
            entry->status = uvm_moe_slot_pte_status_from_nv_status(status);
            uvm_moe_slot_pool_note_pte_failure_locked(slot);
            if (slot) {
                slot->state = UVM_MOE_SLOT_POOL_STATE_ERROR;
                slot->last_status = entry->status;
                slot->pte_status = entry->status;
            }
            ++params->failure_count;
            if (first_error == NV_OK)
                first_error = status;

            if (use_va_block_path) {
                NvU32 rollback_index;

                txn->rollback_batch = NV_TRUE;
                for (rollback_index = txn->installed_count; rollback_index > 0; --rollback_index) {
                    NvU32 installed_pos = rollback_index - 1;
                    NvU32 entry_index = txn->installed_indices[installed_pos];
                    uvm_moe_slot_pool_slot_t *installed_slot = txn->installed_slots[installed_pos];
                    NV_STATUS rollback_status;

                    rollback_status = uvm_moe_slot_pool_rollback_pte_mapping_locked(installed_slot, va_space, mm);
                    if (rollback_status != NV_OK) {
                        uvm_moe_slot_pool_note_pte_failure_locked(installed_slot);
                        if (first_error == NV_OK)
                            first_error = rollback_status;
                        continue;
                    }

                    params->entries[entry_index].status = UVM_MOE_SLOT_REMAP_ENTRY_ROLLED_BACK;
                    if (params->success_count > 0)
                        --params->success_count;
                    if (params->remap_count > 0)
                        --params->remap_count;
                    if (params->tlb_invalidate_count > 0)
                        --params->tlb_invalidate_count;
                    params->accepted_bytes -= min(params->accepted_bytes, params->entries[entry_index].bytes);
                    ++params->failure_count;
                }
                txn->installed_count = 0;
            }
        }

        uvm_phase_f_route_log("moe_slot_pte_remap_apply index=%u mode=%u layer=%u expert=%u slot=%u va=0x%llx bytes=%llu status=%u nv_status=0x%x remap_count=%u tlb_invalidate_count=%u\n",
                              i,
                              use_user_backing_path ?
                                  UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_USER_BACKING_REMAP :
                                  (use_va_block_path ? UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP :
                                                       UVM_MOE_SLOT_REMAP_MODE_SLOT_PTE_REMAP),
                              entry->layer_id,
                              entry->expert_id,
                              entry->slot_id,
                              entry->va_base,
                              entry->bytes,
                              entry->status,
                              status,
                              params->remap_count,
                              params->tlb_invalidate_count);

        if (txn && txn->rollback_batch)
            break;
    }

    if (txn && txn->rollback_batch) {
        NvU32 skipped;

        for (skipped = i + 1; skipped < params->count; ++skipped) {
            if (params->entries[skipped].status != UVM_MOE_SLOT_REMAP_ENTRY_OK)
                continue;

            params->entries[skipped].status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
            ++params->failure_count;
        }
    }

    if (defer_user_backing_tlb_invalidate && params->success_count > 0 && first_error == NV_OK) {
        NV_STATUS invalidate_status = uvm_moe_slot_pool_user_backing_batch_tlb_invalidate(va_space, gpu);

        if (invalidate_status == NV_OK) {
            params->tlb_invalidate_count = 1;
        }
        else {
            NvU32 failed_index;

            if (first_error == NV_OK)
                first_error = invalidate_status;

            for (failed_index = 0; failed_index < params->count; ++failed_index) {
                if (params->entries[failed_index].status == UVM_MOE_SLOT_REMAP_ENTRY_OK)
                    params->entries[failed_index].status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
            }
        }
    }

    mutex_unlock(&g_uvm_moe_slot_pool_lock);

    uvm_va_space_up_write(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);
    uvm_kvfree(group_state);
    uvm_kvfree(txn);

    return first_error;
}

static NV_STATUS uvm_moe_slot_remap_migrate_entry(UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
                                                  UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                  struct file *filp)
{
    NV_STATUS status;
    UVM_MIGRATE_PARAMS *migrate_params = uvm_kvmalloc_zero(sizeof(*migrate_params));

    if (!migrate_params)
        return NV_ERR_NO_MEMORY;

    migrate_params->base = entry->va_base;
    migrate_params->length = entry->bytes;
    migrate_params->destinationUuid = params->gpu_uuid;
    migrate_params->flags = 0;
    migrate_params->semaphoreAddress = 0;
    migrate_params->semaphorePayload = 0;
    migrate_params->cpuNumaNode = NUMA_NO_NODE;

    status = uvm_api_migrate(migrate_params, filp);
    uvm_kvfree(migrate_params);

    return status;
}

static NV_STATUS uvm_moe_slot_remap_validate_managed_range(uvm_va_space_t *va_space,
                                                           NvU64 base,
                                                           NvU64 length)
{
    NV_STATUS status = NV_OK;
    struct mm_struct *mm = NULL;
    bool retained_mm = false;

    if (uvm_va_space_mm_enabled(va_space)) {
        mm = uvm_va_space_mm_retain(va_space);
        retained_mm = mm != NULL;
    }
    else if (!(current->flags & PF_KTHREAD)) {
        mm = current->mm;
    }

    if (mm)
        uvm_down_read_mmap_lock(mm);

    uvm_va_space_down_read(va_space);
    if (uvm_api_range_type_check(va_space, mm, base, length) != UVM_API_RANGE_TYPE_MANAGED)
        status = NV_ERR_INVALID_ADDRESS;
    uvm_va_space_up_read(va_space);

    if (mm)
        uvm_up_read_mmap_lock(mm);
    if (retained_mm)
        uvm_va_space_mm_release(va_space);

    return status;
}

static bool uvm_moe_slot_remap_has_ok_entries(const UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params)
{
    NvU32 i;

    for (i = 0; i < params->count; ++i) {
        if (params->entries[i].status == UVM_MOE_SLOT_REMAP_ENTRY_OK)
            return true;
    }

    return false;
}

static void uvm_moe_slot_remap_fail_ok_entries(UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
                                               NvU32 entry_status)
{
    NvU32 i;

    for (i = 0; i < params->count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[i];

        if (entry->status != UVM_MOE_SLOT_REMAP_ENTRY_OK)
            continue;

        entry->status = entry_status;
        ++params->failure_count;
    }
}

static NV_STATUS uvm_moe_slot_pool_retain_gpu_by_uuid(const NvProcessorUuid *gpu_uuid,
                                                      uvm_gpu_t **gpu_out)
{
    uvm_rm_user_object_t user_rm_device = { 0 };
    uvm_test_parent_gpu_inject_error_t parent_gpu_error = { 0 };

    return uvm_gpu_retain_by_uuid(gpu_uuid,
                                  &user_rm_device,
                                  &parent_gpu_error,
                                  false,
                                  gpu_out);
}

static NvBool uvm_moe_slot_direct_source_ranges_overlap(const UVM_MOE_SLOT_REMAP_ENTRY *a,
                                                         const UVM_MOE_SLOT_REMAP_ENTRY *b)
{
    NvU64 a_end = a->va_base + a->bytes;
    NvU64 b_end = b->va_base + b->bytes;

    return a->va_base < b_end && b->va_base < a_end;
}

static NvBool uvm_moe_slot_direct_pte_metadata_empty(const uvm_moe_slot_pool_slot_t *slot)
{
    return !uvm_moe_slot_pool_slot_has_active_pte(slot) &&
           slot->pte_range_count == 0 &&
           slot->pte_gpu == NULL &&
           slot->pte_va_space == NULL &&
           slot->pte_mode == 0 &&
           !slot->pte_va_block_tracked;
}

static NV_STATUS uvm_moe_slot_direct_validate_slots_locked(UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
                                                           uvm_va_space_t *va_space,
                                                           uvm_gpu_t *gpu,
                                                           uvm_moe_slot_direct_batch_state_t *state)
{
    NvU32 i;

    for (i = 0; i < params->count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[i];
        uvm_moe_slot_pool_slot_t *slot;
        NvU32 j;

        if (entry->bytes > UVM_MOE_SLOT_DIRECT_MAX_ENTRY_BYTES) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ARGUMENT;
            return NV_ERR_OUT_OF_RANGE;
        }

        for (j = 0; j < i; ++j) {
            const UVM_MOE_SLOT_REMAP_ENTRY *prev = &params->entries[j];

            // A single logical slot has exactly one lifetime tracker and one
            // PTE metadata record in this restricted path.
            if (prev->layer_id == entry->layer_id && prev->slot_id == entry->slot_id) {
                entry->status = UVM_MOE_SLOT_REMAP_ENTRY_SLOT_CONFLICT;
                ++params->conflict_count;
                return NV_ERR_INVALID_ARGUMENT;
            }

            if (uvm_moe_slot_direct_source_ranges_overlap(prev, entry)) {
                entry->status = UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ARGUMENT;
                return NV_ERR_INVALID_ARGUMENT;
            }
        }

        slot = uvm_moe_slot_pool_find_locked(va_space, entry->layer_id, entry->slot_id);
        if (!slot) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_BACKING_ALLOC_FAILED;
            return NV_ERR_OBJECT_NOT_FOUND;
        }

        if (slot->owner_expert_id != entry->expert_id) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
            return NV_ERR_INVALID_STATE;
        }

        if (!uvm_moe_slot_pool_slot_has_vidmem_backing(slot, gpu) ||
            entry->slot_offset + entry->bytes > slot->slot_bytes) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_BACKING_ALLOC_FAILED;
            return NV_ERR_OUT_OF_RANGE;
        }

        if (slot->state != UVM_MOE_SLOT_POOL_STATE_READY ||
            slot->async_copy_pending ||
            slot->async_copy_staging ||
            !uvm_moe_slot_direct_pte_metadata_empty(slot)) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
            return NV_ERR_INVALID_STATE;
        }

        state->slots[i] = slot;
    }

    return NV_OK;
}

static int uvm_moe_slot_identity_key_cmp(const void *a, const void *b)
{
    const uvm_moe_slot_identity_key_t *key_a = a;
    const uvm_moe_slot_identity_key_t *key_b = b;

    if (key_a->layer_id != key_b->layer_id)
        return key_a->layer_id < key_b->layer_id ? -1 : 1;

    if (key_a->slot_id != key_b->slot_id)
        return key_a->slot_id < key_b->slot_id ? -1 : 1;

    if (key_a->entry_index != key_b->entry_index)
        return key_a->entry_index < key_b->entry_index ? -1 : 1;

    return 0;
}

static int uvm_moe_slot_source_key_cmp(const void *a, const void *b)
{
    const uvm_moe_slot_source_key_t *key_a = a;
    const uvm_moe_slot_source_key_t *key_b = b;

    if (key_a->va_base != key_b->va_base)
        return key_a->va_base < key_b->va_base ? -1 : 1;

    if (key_a->va_end != key_b->va_end)
        return key_a->va_end < key_b->va_end ? -1 : 1;

    if (key_a->entry_index != key_b->entry_index)
        return key_a->entry_index < key_b->entry_index ? -1 : 1;

    return 0;
}

static const uvm_moe_slot_identity_key_t *uvm_moe_slot_identity_key_find(
    const uvm_moe_slot_identity_key_t *keys,
    NvU32 count,
    NvU32 layer_id,
    NvU32 slot_id)
{
    NvU32 first = 0;
    NvU32 last = count;

    while (first < last) {
        NvU32 middle = first + (last - first) / 2;
        const uvm_moe_slot_identity_key_t *key = &keys[middle];

        if (key->layer_id < layer_id ||
            (key->layer_id == layer_id && key->slot_id < slot_id)) {
            first = middle + 1;
        }
        else {
            last = middle;
        }
    }

    if (first == count || keys[first].layer_id != layer_id || keys[first].slot_id != slot_id)
        return NULL;

    return &keys[first];
}

static NV_STATUS uvm_moe_slot_transition_validate_slots_fast_locked(
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    uvm_va_space_t *va_space,
    uvm_gpu_t *gpu,
    uvm_moe_slot_direct_batch_state_t *state)
{
    NvU32 i;

    for (i = 0; i < params->count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[i];

        if (entry->bytes > UVM_MOE_SLOT_DIRECT_MAX_ENTRY_BYTES ||
            !uvm_moe_slot_transition_entry_strategy_valid(params, i)) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ARGUMENT;
            return NV_ERR_INVALID_ARGUMENT;
        }

        state->identity_keys[i].layer_id = entry->layer_id;
        state->identity_keys[i].slot_id = entry->slot_id;
        state->identity_keys[i].entry_index = i;
        state->source_keys[i].va_base = entry->va_base;
        state->source_keys[i].va_end = entry->va_base + entry->bytes;
        state->source_keys[i].entry_index = i;
    }

    sort(state->identity_keys,
         params->count,
         sizeof(state->identity_keys[0]),
         uvm_moe_slot_identity_key_cmp,
         NULL);
    sort(state->source_keys,
         params->count,
         sizeof(state->source_keys[0]),
         uvm_moe_slot_source_key_cmp,
         NULL);

    for (i = 1; i < params->count; ++i) {
        const uvm_moe_slot_identity_key_t *prev = &state->identity_keys[i - 1];
        const uvm_moe_slot_identity_key_t *current_key = &state->identity_keys[i];

        // A slot-direct entry owns one tracker and one PTE metadata record,
        // so even disjoint target offsets cannot share a logical slot.
        if (prev->layer_id == current_key->layer_id && prev->slot_id == current_key->slot_id) {
            params->entries[current_key->entry_index].status = UVM_MOE_SLOT_REMAP_ENTRY_SLOT_CONFLICT;
            ++params->conflict_count;
            return NV_ERR_INVALID_ARGUMENT;
        }
    }

    for (i = 1; i < params->count; ++i) {
        const uvm_moe_slot_source_key_t *prev = &state->source_keys[i - 1];
        const uvm_moe_slot_source_key_t *current_key = &state->source_keys[i];

        if (prev->va_end > current_key->va_base) {
            NvU32 offending_index = max(prev->entry_index, current_key->entry_index);

            params->entries[offending_index].status = UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ARGUMENT;
            return NV_ERR_INVALID_ARGUMENT;
        }
    }

    // The old path scans all 8,192 slots once per entry. Here the submitted
    // identities are sorted once, then the pool is traversed only once and
    // each matching slot is placed directly at its original entry index.
    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];
        const uvm_moe_slot_identity_key_t *key;

        if (!slot->valid || slot->owner_va_space != va_space)
            continue;

        key = uvm_moe_slot_identity_key_find(state->identity_keys,
                                             params->count,
                                             slot->layer_id,
                                             slot->slot_id);
        if (!key)
            continue;

        if (state->slots[key->entry_index]) {
            params->entries[key->entry_index].status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
            return NV_ERR_INVALID_STATE;
        }

        state->slots[key->entry_index] = slot;
    }

    for (i = 0; i < params->count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[i];
        uvm_moe_slot_pool_slot_t *slot = state->slots[i];

        if (!slot) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_BACKING_ALLOC_FAILED;
            return NV_ERR_OBJECT_NOT_FOUND;
        }

        if (!(slot->flags & UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT) ||
            !uvm_moe_slot_pool_slot_has_vidmem_backing(slot, gpu) ||
            entry->slot_offset + entry->bytes > slot->slot_bytes) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_BACKING_ALLOC_FAILED;
            return NV_ERR_INVALID_STATE;
        }

        if (slot->state != UVM_MOE_SLOT_POOL_STATE_READY ||
            slot->async_copy_pending ||
            slot->async_copy_staging) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
            return NV_ERR_INVALID_STATE;
        }

        if (!uvm_moe_slot_direct_pte_metadata_empty(slot) &&
            (slot->pte_va_space != va_space ||
             slot->pte_gpu != gpu ||
             slot->pte_mode != UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP ||
             !slot->pte_va_block_tracked ||
             slot->pte_range_count == 0 ||
             slot->pte_bytes == 0)) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
            return NV_ERR_INVALID_STATE;
        }
    }

    return NV_OK;
}

static NV_STATUS uvm_moe_slot_plan_validate_slots_cached_locked(
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    uvm_va_space_t *va_space,
    uvm_gpu_t *gpu,
    uvm_moe_slot_direct_batch_state_t *state,
    const uvm_moe_slot_plan_validation_recipe_t *recipe,
    NvBool layer_transition)
{
    NvU32 i;

    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_pool_lock));

    if (!recipe || !recipe->entries || !recipe->slots || !recipe->identity_keys ||
        recipe->count != params->count || recipe->count == 0) {
        uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=slot_cached reason=recipe_shape status=0x%x recipe=%px entries=%px keys=%px recipe_count=%u params_count=%u\n",
                              NV_ERR_INVALID_STATE,
                              recipe,
                              recipe ? recipe->entries : NULL,
                              recipe ? recipe->identity_keys : NULL,
                              recipe ? recipe->count : 0,
                              params->count);
        return NV_ERR_INVALID_STATE;
    }

    for (i = 0; i < params->count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[i];
        const uvm_moe_slot_plan_validation_entry_t *cached = &recipe->entries[i];
        uvm_moe_slot_pool_slot_t *slot;

        if (cached->pool_index >= UVM_MOE_SLOT_POOL_MAX_SLOTS ||
            cached->owner_va_space != va_space ||
            cached->layer_id != entry->layer_id ||
            cached->slot_id != entry->slot_id ||
            !cached->slot_mem || cached->slot_gpu != gpu ||
            entry->bytes > UVM_MOE_SLOT_DIRECT_MAX_ENTRY_BYTES ||
            entry->slot_offset + entry->bytes < entry->slot_offset) {
            uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=slot_cached reason=cached_identity index=%u status=0x%x pool_index=%u cached_layer=%u entry_layer=%u cached_slot=%u entry_slot=%u cached_owner=%px expected_owner=%px cached_mem=%px cached_gpu=%px expected_gpu=%px bytes=%llu slot_offset=%llu\n",
                                  i,
                                  NV_ERR_INVALID_STATE,
                                  cached->pool_index,
                                  cached->layer_id,
                                  entry->layer_id,
                                  cached->slot_id,
                                  entry->slot_id,
                                  cached->owner_va_space,
                                  va_space,
                                  cached->slot_mem,
                                  cached->slot_gpu,
                                  gpu,
                                  entry->bytes,
                                  entry->slot_offset);
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
            return NV_ERR_INVALID_STATE;
        }

        slot = &g_uvm_moe_slot_pool[cached->pool_index];
        if (!slot->valid ||
            slot->owner_va_space != cached->owner_va_space ||
            slot->layer_id != cached->layer_id ||
            slot->slot_id != cached->slot_id ||
            slot->slot_mem != cached->slot_mem ||
            slot->slot_gpu != cached->slot_gpu ||
            slot->slot_bytes != cached->slot_bytes ||
            slot->backing_generation != cached->backing_generation ||
            !(slot->flags & UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT) ||
            !uvm_moe_slot_pool_slot_has_vidmem_backing(slot, gpu) ||
            entry->slot_offset + entry->bytes > slot->slot_bytes) {
            uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=slot_cached reason=backing index=%u status=0x%x pool_index=%u valid=%u slot_owner=%px cached_owner=%px slot_layer=%u cached_layer=%u slot_id=%u cached_slot=%u slot_mem=%px cached_mem=%px slot_gpu=%px cached_gpu=%px slot_bytes=%llu cached_bytes=%llu backing_generation=%llu cached_generation=%llu flags=0x%x entry_end=%llu\n",
                                  i,
                                  NV_ERR_INVALID_STATE,
                                  cached->pool_index,
                                  slot->valid,
                                  slot->owner_va_space,
                                  cached->owner_va_space,
                                  slot->layer_id,
                                  cached->layer_id,
                                  slot->slot_id,
                                  cached->slot_id,
                                  slot->slot_mem,
                                  cached->slot_mem,
                                  slot->slot_gpu,
                                  cached->slot_gpu,
                                  slot->slot_bytes,
                                  cached->slot_bytes,
                                  slot->backing_generation,
                                  cached->backing_generation,
                                  slot->flags,
                                  entry->slot_offset + entry->bytes);
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_BACKING_ALLOC_FAILED;
            return NV_ERR_INVALID_STATE;
        }

        if (slot->state != UVM_MOE_SLOT_POOL_STATE_READY ||
            slot->async_copy_pending ||
            slot->async_copy_staging) {
            uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=slot_cached reason=slot_state index=%u status=0x%x pool_index=%u state=%u copy_pending=%u staging=%px\n",
                                  i,
                                  NV_ERR_INVALID_STATE,
                                  cached->pool_index,
                                  slot->state,
                                  slot->async_copy_pending,
                                  slot->async_copy_staging);
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
            return NV_ERR_INVALID_STATE;
        }

        if (layer_transition) {
            if (!uvm_moe_slot_direct_pte_metadata_empty(slot) &&
                (slot->pte_va_space != va_space ||
                 slot->pte_gpu != gpu ||
                 slot->pte_mode != UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP ||
                 !slot->pte_va_block_tracked ||
                 slot->pte_range_count == 0 ||
                 slot->pte_bytes == 0)) {
                uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=slot_cached reason=pte_metadata index=%u status=0x%x pool_index=%u pte_va_space=%px expected_va_space=%px pte_gpu=%px expected_gpu=%px pte_mode=%u tracked=%u range_count=%u pte_bytes=%llu\n",
                                      i,
                                      NV_ERR_INVALID_STATE,
                                      cached->pool_index,
                                      slot->pte_va_space,
                                      va_space,
                                      slot->pte_gpu,
                                      gpu,
                                      slot->pte_mode,
                                      slot->pte_va_block_tracked,
                                      slot->pte_range_count,
                                      slot->pte_bytes);
                entry->status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
                return NV_ERR_INVALID_STATE;
            }
        }
        else if (slot->owner_expert_id != entry->expert_id ||
                 !uvm_moe_slot_direct_pte_metadata_empty(slot)) {
            uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=slot_cached reason=bootstrap_owner index=%u status=0x%x pool_index=%u owner_expert=%u expected_expert=%u pte_bytes=%llu pte_ranges=%u\n",
                                  i,
                                  NV_ERR_INVALID_STATE,
                                  cached->pool_index,
                                  slot->owner_expert_id,
                                  entry->expert_id,
                                  slot->pte_bytes,
                                  slot->pte_range_count);
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
            return NV_ERR_INVALID_STATE;
        }

        state->slots[i] = slot;
    }

    memcpy(state->identity_keys,
           recipe->identity_keys,
           (size_t)recipe->count * sizeof(recipe->identity_keys[0]));

    return NV_OK;
}

static NV_STATUS uvm_moe_slot_plan_bind_slots_epoch_locked(
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    uvm_va_space_t *va_space,
    uvm_gpu_t *gpu,
    uvm_moe_slot_direct_batch_state_t *state,
    const uvm_moe_slot_plan_validation_recipe_t *recipe,
    NvU64 expected_invalidation_epoch)
{
    NvU64 current_invalidation_epoch;
    NvU64 current_source_epoch;

    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_pool_lock));

    current_invalidation_epoch =
        uvm_va_space_moe_slot_plan_invalidation_epoch(va_space);
    current_source_epoch = uvm_va_space_moe_slot_copy_source_epoch(va_space);
    if (!recipe || !recipe->entries || !recipe->slots ||
        !recipe->identity_keys || recipe->count == 0 ||
        recipe->count != params->count ||
        recipe->owner_va_space != va_space || recipe->gpu != gpu ||
        recipe->invalidation_epoch != expected_invalidation_epoch ||
        recipe->invalidation_epoch != current_invalidation_epoch ||
        recipe->source_epoch != current_source_epoch ||
        recipe->backing_epoch != g_uvm_moe_slot_pool_backing_epoch) {
        // The common cached-slot rejection path accounts for one precisely
        // marked entry, then marks the remaining OK entries. Epoch rejection
        // is aggregate, so nominate entry zero to preserve failure_count ==
        // count instead of reporting one synthetic extra failure.
        params->entries[0].status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
        uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=epoch_gate reason=epoch_mismatch status=0x%x recipe=%px recipe_count=%u params_count=%u recipe_owner=%px expected_owner=%px recipe_gpu=%px expected_gpu=%px recipe_va_epoch=%llu expected_va_epoch=%llu current_va_epoch=%llu recipe_source_epoch=%llu current_source_epoch=%llu recipe_backing_epoch=%llu current_backing_epoch=%llu\n",
                              NV_ERR_INVALID_STATE,
                              recipe,
                              recipe ? recipe->count : 0,
                              params->count,
                              recipe ? recipe->owner_va_space : NULL,
                              va_space,
                              recipe ? recipe->gpu : NULL,
                              gpu,
                              recipe ? recipe->invalidation_epoch : 0,
                              expected_invalidation_epoch,
                              current_invalidation_epoch,
                              recipe ? recipe->source_epoch : 0,
                              current_source_epoch,
                              recipe ? recipe->backing_epoch : 0,
                              g_uvm_moe_slot_pool_backing_epoch);
        return NV_ERR_INVALID_STATE;
    }

    // Pool storage and registered descriptors have module/session lifetime.
    // Epoch equality proves that none of their mutable backing/source/lifetime
    // contracts changed, so only bind the already validated pointer arrays.
    memcpy(state->slots,
           recipe->slots,
           (size_t)recipe->count * sizeof(recipe->slots[0]));
    memcpy(state->identity_keys,
           recipe->identity_keys,
           (size_t)recipe->count * sizeof(recipe->identity_keys[0]));
    return NV_OK;
}

static NV_STATUS uvm_moe_slot_transition_validate_slots_locked(
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    uvm_va_space_t *va_space,
    uvm_gpu_t *gpu,
    uvm_moe_slot_direct_batch_state_t *state)
{
    NvU32 i;

    for (i = 0; i < params->count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[i];
        uvm_moe_slot_pool_slot_t *slot;
        NvU32 j;

        if (entry->bytes > UVM_MOE_SLOT_DIRECT_MAX_ENTRY_BYTES ||
            !uvm_moe_slot_transition_entry_strategy_valid(params, i)) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ARGUMENT;
            return NV_ERR_INVALID_ARGUMENT;
        }

        for (j = 0; j < i; ++j) {
            const UVM_MOE_SLOT_REMAP_ENTRY *prev = &params->entries[j];

            if (prev->layer_id == entry->layer_id && prev->slot_id == entry->slot_id) {
                entry->status = UVM_MOE_SLOT_REMAP_ENTRY_SLOT_CONFLICT;
                ++params->conflict_count;
                return NV_ERR_INVALID_ARGUMENT;
            }

            if (uvm_moe_slot_direct_source_ranges_overlap(prev, entry)) {
                entry->status = UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ARGUMENT;
                return NV_ERR_INVALID_ARGUMENT;
            }
        }

        slot = uvm_moe_slot_pool_find_locked(va_space, entry->layer_id, entry->slot_id);
        if (!slot) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_BACKING_ALLOC_FAILED;
            return NV_ERR_OBJECT_NOT_FOUND;
        }

        // This mode is a fixed-bank transition, not a hidden allocation or
        // eviction path. The old logical owner may differ from the submitted
        // owner, but the backing and its lifetime policy must already exist.
        if (!(slot->flags & UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT) ||
            !uvm_moe_slot_pool_slot_has_vidmem_backing(slot, gpu) ||
            entry->slot_offset + entry->bytes > slot->slot_bytes) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_BACKING_ALLOC_FAILED;
            return NV_ERR_INVALID_STATE;
        }

        if (slot->state != UVM_MOE_SLOT_POOL_STATE_READY ||
            slot->async_copy_pending ||
            slot->async_copy_staging) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
            return NV_ERR_INVALID_STATE;
        }

        // A freshly preallocated bank has no PTE metadata. Otherwise accept
        // only mappings produced by the audited VA-block slot-direct path so
        // teardown can update UVM software state before the backing is reused.
        if (!uvm_moe_slot_direct_pte_metadata_empty(slot) &&
            (slot->pte_va_space != va_space ||
             slot->pte_gpu != gpu ||
             slot->pte_mode != UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP ||
             !slot->pte_va_block_tracked ||
             slot->pte_range_count == 0 ||
             slot->pte_bytes == 0)) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
            return NV_ERR_INVALID_STATE;
        }

        state->slots[i] = slot;
    }

    return NV_OK;
}

static void uvm_moe_slot_transition_mark_old_owners_error_locked(
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    uvm_moe_slot_direct_batch_state_t *state)
{
    NvU32 i;

    for (i = 0; i < params->count; ++i) {
        uvm_moe_slot_pool_slot_t *slot = state->slots[i];

        if (!slot)
            continue;

        slot->state = UVM_MOE_SLOT_POOL_STATE_ERROR;
        slot->last_status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
        slot->pte_status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
    }
}

static int uvm_moe_slot_transition_clear_slice_cmp(const void *a, const void *b)
{
    const uvm_moe_slot_transition_clear_slice_t *slice_a = a;
    const uvm_moe_slot_transition_clear_slice_t *slice_b = b;

    if (slice_a->va_block->start != slice_b->va_block->start)
        return slice_a->va_block->start < slice_b->va_block->start ? -1 : 1;

    if (slice_a->region.first != slice_b->region.first)
        return slice_a->region.first < slice_b->region.first ? -1 : 1;

    if (slice_a->region.outer != slice_b->region.outer)
        return slice_a->region.outer < slice_b->region.outer ? -1 : 1;

    if (slice_a->slot_index != slice_b->slot_index)
        return slice_a->slot_index < slice_b->slot_index ? -1 : 1;

    return 0;
}

static NV_STATUS uvm_moe_slot_transition_build_clear_state_locked(
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    uvm_va_space_t *va_space,
    uvm_gpu_t *gpu,
    uvm_moe_slot_direct_batch_state_t *state,
    uvm_moe_slot_transition_clear_state_t *clear_state)
{
    NvU32 i;

    uvm_assert_rwsem_locked_write(&va_space->lock);

    for (i = 0; i < params->count; ++i) {
        uvm_moe_slot_pool_slot_t *slot = state->slots[i];
        const uvm_moe_slot_pool_pte_range_t *range;
        NvU64 range_end;
        NvU64 cursor;

        if (uvm_moe_slot_direct_pte_metadata_empty(slot))
            continue;

        // The grouped clear is deliberately restricted to metadata emitted by
        // one successful slot-direct range. Legacy/multi-range records retain
        // the existing per-slot teardown path.
        if (slot->pte_range_vec ||
            slot->pte_gpu != gpu ||
            slot->pte_va_space != va_space ||
            slot->pte_mode != UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP ||
            !slot->pte_va_block_tracked ||
            slot->pte_range_count != 1) {
            return NV_WARN_MORE_PROCESSING_REQUIRED;
        }

        range = &slot->pte_ranges[0];
        range_end = range->va_base + range->bytes;
        if (range->va_base == 0 ||
            range->bytes == 0 ||
            range_end <= range->va_base ||
            !IS_ALIGNED(range->va_base, PAGE_SIZE) ||
            !IS_ALIGNED(range->bytes, PAGE_SIZE) ||
            slot->pte_va_base != range->va_base ||
            slot->pte_bytes != range->bytes) {
            return NV_WARN_MORE_PROCESSING_REQUIRED;
        }

        cursor = range->va_base;
        while (cursor < range_end) {
            uvm_moe_slot_transition_clear_slice_t *slice;
            uvm_va_block_t *va_block = NULL;
            NvU64 slice_end;
            NV_STATUS status;

            if (clear_state->slice_count >= UVM_MOE_SLOT_DIRECT_MAX_BLOCKS)
                return NV_WARN_MORE_PROCESSING_REQUIRED;

            // Slot-direct mappings reject HMM blocks and record an already
            // existing managed VA block. Do not create new state merely to make
            // this optional fast path eligible.
            status = uvm_va_block_find(va_space, cursor, &va_block);
            if (status != NV_OK ||
                !va_block ||
                uvm_va_block_is_dead(va_block) ||
                uvm_va_block_is_hmm(va_block) ||
                uvm_va_block_get_va_space(va_block) != va_space ||
                cursor > va_block->end) {
                return NV_WARN_MORE_PROCESSING_REQUIRED;
            }

            slice_end = min(range_end, va_block->end + 1);
            if (slice_end <= cursor)
                return NV_WARN_MORE_PROCESSING_REQUIRED;

            slice = &clear_state->slices[clear_state->slice_count++];
            slice->va_block = va_block;
            slice->region = uvm_va_block_region_from_start_end(va_block, cursor, slice_end - 1);
            slice->slot_index = i;
            ++clear_state->slot_block_counts[i];
            cursor = slice_end;
        }

        ++clear_state->range_count;
    }

    sort(clear_state->slices,
         clear_state->slice_count,
         sizeof(clear_state->slices[0]),
         uvm_moe_slot_transition_clear_slice_cmp,
         NULL);

    for (i = 0; i < clear_state->slice_count; ++i) {
        const uvm_moe_slot_transition_clear_slice_t *slice = &clear_state->slices[i];
        uvm_moe_slot_transition_clear_group_t *group;

        clear_state->page_count += uvm_va_block_region_num_pages(slice->region);

        if (i == 0 || clear_state->slices[i - 1].va_block != slice->va_block) {
            ++clear_state->unique_block_count;
            group = &clear_state->groups[clear_state->group_count++];
            group->va_block = slice->va_block;
            group->first_slice = i;
            uvm_page_mask_zero(&group->page_mask);
        }
        else {
            group = &clear_state->groups[clear_state->group_count - 1];

            // Overlap would make two logical slots claim the same old PTE. The
            // legacy path remains the conservative recovery mechanism for such
            // metadata; never mutate it through the grouped path.
            if (clear_state->slices[i - 1].region.outer > slice->region.first)
                return NV_WARN_MORE_PROCESSING_REQUIRED;
        }

        ++group->slice_count;
        uvm_page_mask_region_fill(&group->page_mask, slice->region);
    }

    if (clear_state->group_count != clear_state->unique_block_count)
        return NV_ERR_INVALID_STATE;

    return NV_OK;
}

static NV_STATUS uvm_moe_slot_transition_preflight_group_clear_locked(
    uvm_va_space_t *va_space,
    uvm_gpu_t *gpu,
    const uvm_moe_slot_transition_clear_state_t *clear_state)
{
    NvU32 group_index;

    uvm_assert_rwsem_locked_write(&va_space->lock);

    for (group_index = 0; group_index < clear_state->group_count; ++group_index) {
        const uvm_moe_slot_transition_clear_group_t *group =
            &clear_state->groups[group_index];
        uvm_va_block_t *va_block = group->va_block;
        NV_STATUS status;

        uvm_mutex_lock(&va_block->lock);
        status = uvm_va_block_validate_moe_slot_unmap(va_block,
                                                      gpu,
                                                      uvm_va_block_region_from_block(va_block),
                                                      &group->page_mask);
        uvm_mutex_unlock(&va_block->lock);
        if (status != NV_OK)
            return NV_WARN_MORE_PROCESSING_REQUIRED;
    }

    return NV_OK;
}

static void uvm_moe_static_alias_classify_park_groups_locked(
    uvm_va_space_t *va_space,
    uvm_gpu_t *gpu,
    uvm_moe_slot_transition_clear_state_t *clear_state)
{
    NvU32 group_index;

    uvm_assert_rwsem_locked_write(&va_space->lock);

    for (group_index = 0; group_index < clear_state->group_count; ++group_index) {
        uvm_moe_slot_transition_clear_group_t *group =
            &clear_state->groups[group_index];
        uvm_va_block_t *va_block = group->va_block;
        NV_STATUS status;

        uvm_mutex_lock(&va_block->lock);
        status = uvm_va_block_validate_moe_slot_unmap(
            va_block,
            gpu,
            uvm_va_block_region_from_block(va_block),
            &group->page_mask);
        uvm_mutex_unlock(&va_block->lock);
        group->grouped_eligible = status == NV_OK;
    }
}

static void uvm_moe_slot_transition_commit_group_clear_metadata_locked(
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    uvm_moe_slot_direct_batch_state_t *state,
    const uvm_moe_slot_transition_clear_state_t *clear_state)
{
    NvU32 i;

    for (i = 0; i < params->count; ++i) {
        uvm_moe_slot_pool_slot_t *slot = state->slots[i];

        if (uvm_moe_slot_direct_pte_metadata_empty(slot))
            continue;

        UVM_ASSERT(slot->pte_range_vec == NULL);
        slot->pte_block_unmap_count += clear_state->slot_block_counts[i];
        if (clear_state->slot_block_counts[i] != 0)
            ++slot->pte_block_tracker_wait_count;
        ++slot->pte_clear_count;
        uvm_moe_slot_pool_forget_release_pte_mapping_locked(slot);
    }

    g_uvm_moe_slot_pool_pte_clear_count += clear_state->range_count;
    g_uvm_moe_slot_pool_pte_block_unmap_count += clear_state->unique_block_count;
    if (clear_state->unique_block_count != 0)
        ++g_uvm_moe_slot_pool_pte_block_tracker_wait_count;
}

static void uvm_moe_slot_transition_prepared_clear_deinit(
    uvm_moe_slot_transition_prepared_clear_t *prepared)
{
    if (!prepared)
        return;

    uvm_va_block_context_free(prepared->block_context);
    uvm_kvfree(prepared->clear_state);
    memset(prepared, 0, sizeof(*prepared));
}

static NV_STATUS uvm_moe_slot_transition_build_cached_clear_state_locked(
    const UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    const UVM_MOE_SLOT_REMAP_BATCH_PARAMS *resident_batch,
    uvm_va_space_t *va_space,
    uvm_gpu_t *gpu,
    uvm_moe_slot_direct_batch_state_t *state,
    const uvm_moe_slot_plan_target_recipe_t *recipe,
    NvBool resident_bank_empty,
    uvm_moe_slot_transition_clear_state_t *clear_state)
{
    NvU64 materialized_bytes = 0;
    NvU32 i;

    uvm_assert_rwsem_locked_write(&va_space->lock);

    if (resident_bank_empty) {
        if (resident_batch || recipe)
            return NV_ERR_INVALID_STATE;

        // A missing resident recipe is safe only for a genuinely empty bank.
        // Never interpret a destroyed or stale plan_id as an empty bank.
        for (i = 0; i < params->count; ++i) {
            if (!state->slots[i] ||
                !uvm_moe_slot_direct_pte_metadata_empty(state->slots[i])) {
                return NV_ERR_INVALID_STATE;
            }
        }
        return NV_OK;
    }

    if (!resident_batch || !recipe || resident_batch->count != params->count ||
        recipe->entry_count != resident_batch->count ||
        recipe->slice_count == 0 || recipe->group_count == 0 ||
        (recipe->pte_mode != UVM_SLOT_DIRECT_PTE_MODE_4K &&
         recipe->pte_mode != UVM_SLOT_DIRECT_PTE_MODE_64K) ||
        recipe->bytes != resident_batch->requested_bytes ||
        recipe->invalidation_epoch != uvm_va_space_moe_slot_plan_invalidation_epoch(va_space)) {
        return NV_ERR_INVALID_STATE;
    }

    // Validate every resident slot record before resolving or touching a VA
    // block. The recipe describes exactly the PTE metadata published by the
    // resident plan; any mismatch is a pre-transaction cache rejection.
    for (i = 0; i < resident_batch->count; ++i) {
        const UVM_MOE_SLOT_REMAP_ENTRY *resident_entry = &resident_batch->entries[i];
        const uvm_moe_slot_identity_key_t *current_key =
            uvm_moe_slot_identity_key_find(state->identity_keys,
                                            params->count,
                                            resident_entry->layer_id,
                                            resident_entry->slot_id);
        const uvm_moe_slot_pool_slot_t *slot;

        if (!current_key || current_key->entry_index >= params->count)
            return NV_ERR_INVALID_STATE;

        slot = state->slots[current_key->entry_index];
        if (!slot || slot->pte_range_vec || slot->pte_gpu != gpu ||
            slot->pte_va_space != va_space ||
            slot->pte_mode != UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP ||
            !slot->pte_va_block_tracked || slot->pte_range_count != 1 ||
            slot->pte_va_base != resident_entry->va_base ||
            slot->pte_bytes != resident_entry->bytes ||
            slot->pte_slot_offset != resident_entry->slot_offset ||
            slot->pte_ranges[0].va_base != resident_entry->va_base ||
            slot->pte_ranges[0].bytes != resident_entry->bytes) {
            return NV_ERR_INVALID_STATE;
        }
    }

    for (i = 0; i < recipe->group_count; ++i) {
        const uvm_moe_slot_plan_target_group_t *cached_group = &recipe->groups[i];
        uvm_moe_slot_transition_clear_group_t *clear_group = &clear_state->groups[i];
        uvm_page_mask_t materialized_mask;
        uvm_va_block_t *va_block = NULL;
        NvU32 slice_offset;
        NV_STATUS status;

        if (cached_group->slice_count == 0 ||
            cached_group->first_slice > recipe->slice_count ||
            cached_group->slice_count > recipe->slice_count - cached_group->first_slice ||
            cached_group->block_end < cached_group->block_start) {
            return NV_ERR_INVALID_STATE;
        }

        status = uvm_va_block_find(va_space, cached_group->block_start, &va_block);
        if (status != NV_OK || !va_block || uvm_va_block_is_dead(va_block) ||
            uvm_va_block_is_hmm(va_block) ||
            uvm_va_block_get_va_space(va_block) != va_space ||
            va_block->start != cached_group->block_start ||
            va_block->end != cached_group->block_end) {
            return NV_ERR_INVALID_STATE;
        }

        clear_group->va_block = va_block;
        clear_group->first_slice = clear_state->slice_count;
        clear_group->slice_count = cached_group->slice_count;
        uvm_page_mask_copy(&clear_group->page_mask, &cached_group->page_mask);
        uvm_page_mask_zero(&materialized_mask);

        for (slice_offset = 0; slice_offset < cached_group->slice_count; ++slice_offset) {
            const uvm_moe_slot_plan_target_slice_t *cached_slice =
                &recipe->slices[cached_group->first_slice + slice_offset];
            const UVM_MOE_SLOT_REMAP_ENTRY *resident_entry;
            const uvm_moe_slot_identity_key_t *current_key;
            uvm_moe_slot_transition_clear_slice_t *clear_slice;
            NvU64 slice_end = cached_slice->va_base + cached_slice->bytes;

            if (cached_slice->entry_index >= resident_batch->count ||
                cached_slice->bytes == 0 || slice_end <= cached_slice->va_base ||
                cached_slice->va_base < va_block->start || slice_end - 1 > va_block->end ||
                clear_state->slice_count >= UVM_MOE_SLOT_DIRECT_MAX_BLOCKS) {
                return NV_ERR_INVALID_STATE;
            }

            resident_entry = &resident_batch->entries[cached_slice->entry_index];
            current_key = uvm_moe_slot_identity_key_find(state->identity_keys,
                                                          params->count,
                                                          resident_entry->layer_id,
                                                          resident_entry->slot_id);
            if (!current_key || current_key->entry_index >= params->count)
                return NV_ERR_INVALID_STATE;

            clear_slice = &clear_state->slices[clear_state->slice_count++];
            clear_slice->va_block = va_block;
            clear_slice->region = uvm_va_block_region_from_start_end(va_block,
                                                                      cached_slice->va_base,
                                                                      slice_end - 1);
            clear_slice->slot_index = current_key->entry_index;
            ++clear_state->slot_block_counts[current_key->entry_index];
            uvm_page_mask_region_fill(&materialized_mask, clear_slice->region);
            clear_state->page_count += uvm_va_block_region_num_pages(clear_slice->region);
            materialized_bytes += cached_slice->bytes;
        }

        if (!uvm_page_mask_equal(&materialized_mask, &cached_group->page_mask))
            return NV_ERR_INVALID_STATE;
    }

    if (clear_state->slice_count != recipe->slice_count ||
        materialized_bytes != recipe->bytes) {
        return NV_ERR_INVALID_STATE;
    }

    clear_state->group_count = recipe->group_count;
    clear_state->unique_block_count = recipe->group_count;
    clear_state->range_count = resident_batch->count;
    return NV_OK;
}

static NV_STATUS uvm_moe_slot_transition_prepare_dual_clear_locked(
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    const UVM_MOE_SLOT_REMAP_BATCH_PARAMS *resident_batch,
    uvm_va_space_t *va_space,
    struct mm_struct *mm,
    uvm_gpu_t *gpu,
    uvm_moe_slot_direct_batch_state_t *state,
    const uvm_moe_slot_plan_target_recipe_t *resident_recipe,
    NvBool recipe_cache_enabled,
    NvBool resident_bank_empty,
    uvm_moe_slot_transition_clear_result_t *clear_result,
    uvm_moe_slot_transition_prepared_clear_t *prepared,
    NvBool *recipe_cache_hit_out)
{
    uvm_moe_slot_transition_clear_state_t *clear_state;
    NV_STATUS status;

    UVM_ASSERT(prepared);
    UVM_ASSERT(!prepared->clear_state);
    UVM_ASSERT(!prepared->block_context);
    UVM_ASSERT(recipe_cache_hit_out);
    *recipe_cache_hit_out = NV_FALSE;

    clear_result->attempted = NV_TRUE;
    clear_state = uvm_kvmalloc_zero(sizeof(*clear_state));
    if (!clear_state)
        return NV_ERR_NO_MEMORY;

    status = recipe_cache_enabled ?
                 uvm_moe_slot_transition_build_cached_clear_state_locked(
                     params,
                     resident_batch,
                     va_space,
                     gpu,
                     state,
                     resident_recipe,
                     resident_bank_empty,
                     clear_state) :
                 uvm_moe_slot_transition_build_clear_state_locked(params,
                                                                   va_space,
                                                                   gpu,
                                                                   state,
                                                                   clear_state);
    if (status != NV_OK)
        goto error;

    if (recipe_cache_enabled && !resident_bank_empty)
        *recipe_cache_hit_out = NV_TRUE;

    status = uvm_moe_slot_transition_preflight_group_clear_locked(va_space,
                                                                   gpu,
                                                                   clear_state);
    if (status != NV_OK)
        goto error;

    if (clear_state->slice_count != 0) {
        prepared->block_context = uvm_va_block_context_alloc(mm);
        if (!prepared->block_context) {
            status = NV_ERR_NO_MEMORY;
            goto error;
        }
    }

    prepared->clear_state = clear_state;
    prepared->direct_cutover = NV_TRUE;
    clear_result->used = NV_TRUE;
    clear_result->direct_cutover = NV_TRUE;
    clear_result->ranges = clear_state->range_count;
    clear_result->unique_blocks = clear_state->unique_block_count;
    clear_result->direct_cutover_pages = clear_state->page_count;
    return NV_OK;

error:
    clear_result->fallback = NV_TRUE;
    uvm_moe_slot_transition_prepared_clear_deinit(prepared);
    uvm_kvfree(clear_state);
    return status;
}

static NV_STATUS uvm_moe_slot_transition_submit_dual_clear_locked(
    uvm_gpu_t *gpu,
    uvm_moe_slot_transition_prepared_clear_t *prepared,
    uvm_tracker_t *deferred_tracker)
{
    uvm_moe_slot_transition_clear_state_t *clear_state = prepared->clear_state;
    uvm_tracker_t clear_tracker = UVM_TRACKER_INIT();
    NV_STATUS status = NV_OK;
    NV_STATUS tracker_status;
    NvU32 group_index;

    UVM_ASSERT(clear_state);
    UVM_ASSERT(prepared->direct_cutover);
    UVM_ASSERT(!prepared->completion_guarded);
    UVM_ASSERT(!prepared->metadata_committed);
    UVM_ASSERT(deferred_tracker);

    for (group_index = 0; group_index < clear_state->group_count; ++group_index) {
        const uvm_moe_slot_transition_clear_group_t *group =
            &clear_state->groups[group_index];
        uvm_va_block_t *va_block = group->va_block;
        uvm_va_block_retry_t block_retry;

        status = UVM_VA_BLOCK_LOCK_RETRY(
            va_block,
            &block_retry,
            uvm_va_block_unmap_moe_slot_no_cpu_restore(
                va_block,
                prepared->block_context,
                gpu,
                uvm_va_block_region_from_block(va_block),
                &group->page_mask,
                &va_block->tracker));

        // A retry can submit a push before reporting a later error. Preserve
        // every such completion in the local tracker before leaving the group.
        uvm_mutex_lock(&va_block->lock);
        tracker_status = uvm_tracker_add_tracker_safe(&clear_tracker,
                                                      &va_block->tracker);
        uvm_mutex_unlock(&va_block->lock);
        if (status == NV_OK)
            status = tracker_status;
        if (status != NV_OK)
            break;
    }

    tracker_status = uvm_tracker_add_tracker_safe(deferred_tracker, &clear_tracker);
    if (tracker_status != NV_OK) {
        NV_STATUS drain_status = uvm_tracker_wait(&clear_tracker);

        if (drain_status != NV_OK)
            tracker_status = drain_status;
    }
    if (status == NV_OK)
        status = tracker_status;
    uvm_tracker_deinit(&clear_tracker);

    // Adding the tracker only transfers its lifetime to the transaction. It
    // does not prove that the old leaf clear/TLB work has completed. Retain
    // the old slot metadata until the aggregate wait succeeds so an error can
    // still rebuild or retry the exact clear operation.
    if (status == NV_OK)
        prepared->completion_guarded = NV_TRUE;

    return status;
}

static void uvm_moe_slot_transition_commit_completed_dual_clear_locked(
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    uvm_moe_slot_direct_batch_state_t *state,
    uvm_moe_slot_transition_prepared_clear_t *prepared)
{
    UVM_ASSERT(prepared->clear_state);
    UVM_ASSERT(prepared->completion_guarded);

    if (prepared->metadata_committed)
        return;

    uvm_moe_slot_transition_commit_group_clear_metadata_locked(params,
                                                                state,
                                                                prepared->clear_state);
    prepared->metadata_committed = NV_TRUE;
}

static NV_STATUS uvm_moe_slot_transition_fail_closed_dual_clear_locked(
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    uvm_gpu_t *gpu,
    uvm_moe_slot_direct_batch_state_t *state,
    uvm_moe_slot_transition_prepared_clear_t *prepared)
{
    const uvm_moe_slot_transition_clear_state_t *clear_state = prepared->clear_state;
    uvm_tracker_t cleanup_tracker = UVM_TRACKER_INIT();
    NV_STATUS first_error = NV_OK;
    NV_STATUS wait_status;
    NvU32 group_index;

    if (prepared->metadata_committed || !clear_state || clear_state->slice_count == 0)
        return NV_OK;

    UVM_ASSERT(prepared->block_context);
    for (group_index = 0; group_index < clear_state->group_count; ++group_index) {
        const uvm_moe_slot_transition_clear_group_t *group =
            &clear_state->groups[group_index];
        uvm_va_block_t *va_block = group->va_block;
        uvm_va_block_retry_t block_retry;
        NV_STATUS status;
        NV_STATUS tracker_status;

        status = UVM_VA_BLOCK_LOCK_RETRY(
            va_block,
            &block_retry,
            uvm_va_block_unmap(va_block,
                               prepared->block_context,
                               gpu->id,
                               uvm_va_block_region_from_block(va_block),
                               &group->page_mask,
                               &va_block->tracker));
        if (status != NV_OK && first_error == NV_OK)
            first_error = status;

        uvm_mutex_lock(&va_block->lock);
        tracker_status = uvm_tracker_add_tracker_safe(&cleanup_tracker,
                                                      &va_block->tracker);
        uvm_mutex_unlock(&va_block->lock);
        if (tracker_status != NV_OK && first_error == NV_OK)
            first_error = tracker_status;
    }

    wait_status = uvm_tracker_wait(&cleanup_tracker);
    if (first_error == NV_OK)
        first_error = wait_status;
    uvm_tracker_deinit(&cleanup_tracker);

    if (first_error == NV_OK && !prepared->metadata_committed) {
        uvm_moe_slot_transition_commit_group_clear_metadata_locked(params,
                                                                    state,
                                                                    clear_state);
        prepared->metadata_committed = NV_TRUE;
    }
    return first_error;
}

static NV_STATUS uvm_moe_slot_transition_group_clear_locked(
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    uvm_va_space_t *va_space,
    struct mm_struct *mm,
    uvm_gpu_t *gpu,
    uvm_moe_slot_direct_batch_state_t *state,
    uvm_moe_slot_transition_clear_result_t *clear_result,
    uvm_tracker_t *deferred_tracker)
{
    const NvBool direct_cutover =
        uvm_moe_slot_transition_direct_cutover_enabled(params->mode);
    uvm_moe_slot_transition_clear_state_t *clear_state = NULL;
    uvm_va_block_context_t *block_context = NULL;
    uvm_tracker_t clear_tracker = UVM_TRACKER_INIT();
    NV_STATUS status;
    NV_STATUS wait_status;
    NvU32 group_index;

    clear_result->attempted = NV_TRUE;
    clear_state = uvm_kvmalloc_zero(sizeof(*clear_state));
    if (!clear_state) {
        clear_result->fallback = NV_TRUE;
        return NV_WARN_MORE_PROCESSING_REQUIRED;
    }

    status = uvm_moe_slot_transition_build_clear_state_locked(params,
                                                               va_space,
                                                               gpu,
                                                               state,
                                                               clear_state);
    if (status != NV_OK)
        goto fallback;

    status = uvm_moe_slot_transition_preflight_group_clear_locked(va_space, gpu, clear_state);
    if (status != NV_OK)
        goto fallback;

    if (clear_state->slice_count != 0) {
        block_context = uvm_va_block_context_alloc(mm);
        if (!block_context)
            goto fallback;
    }

    clear_result->used = NV_TRUE;
    clear_result->direct_cutover = direct_cutover;
    clear_result->ranges = clear_state->range_count;
    clear_result->unique_blocks = clear_state->unique_block_count;
    clear_result->direct_cutover_pages = direct_cutover ? clear_state->page_count : 0;

    // All range, block, overlap, and large-leaf checks above completed without
    // mutation. From this point onward a failure is fail-closed and must never
    // fall back to per-slot teardown because earlier groups may already have
    // submitted PTE changes.
    status = NV_OK;
    for (group_index = 0; group_index < clear_state->group_count; ++group_index) {
        const uvm_moe_slot_transition_clear_group_t *group =
            &clear_state->groups[group_index];
        uvm_va_block_t *va_block = group->va_block;
        uvm_va_block_retry_t block_retry;
        NV_STATUS tracker_status;

        // Keep the retry input outside block_context: the ordinary cleanup
        // path uses the context's caller mask as an output when it restores
        // CPU mappings. Direct cutover deliberately omits that second map.
        if (direct_cutover) {
            status = UVM_VA_BLOCK_LOCK_RETRY(
                va_block,
                &block_retry,
                uvm_va_block_unmap_moe_slot_no_cpu_restore(
                    va_block,
                    block_context,
                    gpu,
                    uvm_va_block_region_from_block(va_block),
                    &group->page_mask,
                    &va_block->tracker));
        }
        else {
            status = UVM_VA_BLOCK_LOCK_RETRY(
                va_block,
                &block_retry,
                uvm_va_block_unmap_moe_slot(va_block,
                                            block_context,
                                            gpu,
                                            uvm_va_block_region_from_block(va_block),
                                            &group->page_mask,
                                            &va_block->tracker));
        }

        // Snapshot the block tracker even if the operation reported an error:
        // it may have submitted an earlier unmap/map push before failing.
        uvm_mutex_lock(&va_block->lock);
        tracker_status = uvm_tracker_add_tracker_safe(&clear_tracker, &va_block->tracker);
        uvm_mutex_unlock(&va_block->lock);
        if (status == NV_OK)
            status = tracker_status;
        if (status != NV_OK)
            break;
    }

    // The ordinary single-bank path waits here before the destination can be
    // overwritten. In the event-guarded dual-bank path the destination bank
    // is inactive, so retain the same clear work in the transaction tracker
    // and let its completion overlap the CE copy. No compute is running: the
    // userspace main-stream event completed before this ioctl was entered.
    if (deferred_tracker) {
        wait_status = uvm_tracker_add_tracker_safe(deferred_tracker, &clear_tracker);
        if (wait_status != NV_OK) {
            // Do not let a submitted clear escape an aggregation failure. The
            // safe helper normally drains clear_tracker on allocation failure,
            // but explicitly wait here as well so every other error path has
            // the same lifetime guarantee before clear_tracker is destroyed.
            NV_STATUS drain_status = uvm_tracker_wait(&clear_tracker);

            if (drain_status != NV_OK)
                wait_status = drain_status;
        }
    }
    else {
        wait_status = uvm_tracker_wait(&clear_tracker);
    }
    if (status == NV_OK)
        status = wait_status;
    uvm_tracker_deinit(&clear_tracker);

    if (status == NV_OK)
        uvm_moe_slot_transition_commit_group_clear_metadata_locked(params, state, clear_state);

    uvm_va_block_context_free(block_context);
    uvm_kvfree(clear_state);
    return status;

fallback:
    // No PTE or slot metadata has changed on this path. Preserve compatibility
    // with unusual legacy metadata by handing the batch to the audited per-slot
    // release implementation.
    clear_result->fallback = NV_TRUE;
    uvm_va_block_context_free(block_context);
    uvm_tracker_deinit(&clear_tracker);
    uvm_kvfree(clear_state);
    return NV_WARN_MORE_PROCESSING_REQUIRED;
}

static void uvm_moe_slot_transition_mark_loading_locked(
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    uvm_moe_slot_direct_batch_state_t *state,
    NvU64 batch_id)
{
    NvU32 i;

    for (i = 0; i < params->count; ++i) {
        uvm_moe_slot_pool_slot_t *slot = state->slots[i];

        slot->state = UVM_MOE_SLOT_POOL_STATE_LOADING;
        slot->batch_id = batch_id;
    }
}

static NV_STATUS uvm_moe_slot_transition_begin_locked(
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    uvm_va_space_t *va_space,
    struct mm_struct *mm,
    uvm_gpu_t *gpu,
    uvm_moe_slot_direct_batch_state_t *state,
    NvU64 batch_id,
    NvBool try_group_clear,
    uvm_moe_slot_transition_clear_result_t *clear_result,
    uvm_tracker_t *deferred_tracker)
{
    NvBool group_clear_used = NV_FALSE;
    NvU32 i;

    if (try_group_clear) {
        NV_STATUS status = uvm_moe_slot_transition_group_clear_locked(params,
                                                                       va_space,
                                                                       mm,
                                                                       gpu,
                                                                       state,
                                                                       clear_result,
                                                                       deferred_tracker);

        group_clear_used = clear_result->used;
        if (status != NV_OK && group_clear_used) {
            params->entries[0].status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
            uvm_moe_slot_transition_mark_old_owners_error_locked(params, state);
            return status;
        }
    }

    // Keep every old owner unchanged until every old slot PTE has been
    // removed. A failure in this loop may leave some old VAs unmapped, but no
    // backing has been overwritten and userspace can unambiguously retain the
    // old owner table (remap_count remains zero) and poison the session.
    if (!group_clear_used) {
        for (i = 0; i < params->count; ++i) {
            uvm_moe_slot_pool_slot_t *slot = state->slots[i];
            NV_STATUS status = uvm_moe_slot_pool_release_pte_mapping_locked(slot,
                                                                             va_space,
                                                                             mm);

            if (status != NV_OK) {
                params->entries[i].status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
                uvm_moe_slot_transition_mark_old_owners_error_locked(params, state);
                return status;
            }
        }
    }

    // The old mappings are now gone, but the new contents and PTEs are not yet
    // ready. Keep the old logical owner intact and expose only a non-active
    // slot state while the copy and shadow PTE transaction is in flight. This
    // is the fail-closed TRANSITIONING state: fault, eviction, and cleanup
    // paths must not mistake the target owner for an ACTIVE mapping.
    uvm_moe_slot_transition_mark_loading_locked(params, state, batch_id);

    return NV_OK;
}

static void uvm_moe_slot_transition_commit_owner_locked(
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    uvm_moe_slot_direct_batch_state_t *state)
{
    NvU32 i;

    // The aggregate tracker has completed before this function is called, so
    // every CE copy, PTE write, and TLB invalidate is globally visible. Owner
    // publication is now an infallible batch-wide metadata commit. Do not
    // memset the slots here: they already contain the newly recorded PTE
    // cleanup metadata and retain references needed by later transitions.
    for (i = 0; i < params->count; ++i) {
        const UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[i];
        uvm_moe_slot_pool_slot_t *slot = state->slots[i];

        slot->layer_id = entry->layer_id;
        slot->slot_id = entry->slot_id;
        slot->owner_expert_id = entry->expert_id;
        slot->first_va_base = entry->va_base;

        ++g_uvm_moe_slot_pool_vidmem_rebind_count;
        g_uvm_moe_slot_pool_vidmem_rebind_bytes += entry->bytes;
    }

    params->remap_count = params->count;
}

static int uvm_moe_slot_direct_block_cmp(const void *a, const void *b, const void *priv)
{
    const uvm_va_block_slot_direct_entry_t *block_a = a;
    const uvm_va_block_slot_direct_entry_t *block_b = b;

    (void)priv;

    if (block_a->va_block->start != block_b->va_block->start)
        return block_a->va_block->start < block_b->va_block->start ? -1 : 1;

    if (block_a->region.first != block_b->region.first)
        return block_a->region.first < block_b->region.first ? -1 : 1;

    if (block_a->region.outer != block_b->region.outer)
        return block_a->region.outer < block_b->region.outer ? -1 : 1;

    return 0;
}

static void uvm_moe_slot_direct_block_swap(void *a, void *b, int size, const void *priv)
{
    uvm_moe_slot_direct_batch_state_t *state = (uvm_moe_slot_direct_batch_state_t *)priv;
    uvm_va_block_slot_direct_entry_t *block_a = a;
    uvm_va_block_slot_direct_entry_t *block_b = b;
    size_t index_a = block_a - state->blocks;
    size_t index_b = block_b - state->blocks;
    uvm_va_block_slot_direct_entry_t block_tmp;
    NvU32 entry_index_tmp;

    UVM_ASSERT(size == sizeof(*block_a));
    UVM_ASSERT(index_a < state->block_count);
    UVM_ASSERT(index_b < state->block_count);

    if (block_a == block_b)
        return;

    block_tmp = *block_a;
    *block_a = *block_b;
    *block_b = block_tmp;

    entry_index_tmp = state->block_entry_indices[index_a];
    state->block_entry_indices[index_a] = state->block_entry_indices[index_b];
    state->block_entry_indices[index_b] = entry_index_tmp;
}

static void uvm_moe_slot_direct_sort_blocks(uvm_moe_slot_direct_batch_state_t *state,
                                             NvBool fast_sort)
{
    NvU32 i;

    // Static-alias sessions retain their records in VA order. Avoid sorting
    // the same immutable 30K--40K-entry topology on every RESUME while still
    // running the adjacent overlap check in the caller.
    for (i = 1; i < state->block_count; ++i) {
        if (state->blocks[i - 1].va_block->start >
                state->blocks[i].va_block->start ||
            (state->blocks[i - 1].va_block->start ==
                 state->blocks[i].va_block->start &&
             state->blocks[i - 1].region.first >
                 state->blocks[i].region.first)) {
            break;
        }
    }
    if (i == state->block_count)
        return;

    if (fast_sort) {
        sort_r(state->blocks,
               state->block_count,
               sizeof(state->blocks[0]),
               uvm_moe_slot_direct_block_cmp,
               uvm_moe_slot_direct_block_swap,
               state);
        return;
    }

    // Keep the block entry and its source range index paired without another
    // allocation. Sorting slices within a VA block is required both for the
    // overlap check below and for unique-block lock acquisition in the helper.
    for (i = 1; i < state->block_count; ++i) {
        uvm_va_block_slot_direct_entry_t block = state->blocks[i];
        NvU32 entry_index = state->block_entry_indices[i];
        NvU32 pos = i;

        while (pos > 0 &&
               (state->blocks[pos - 1].va_block->start > block.va_block->start ||
                (state->blocks[pos - 1].va_block->start == block.va_block->start &&
                 state->blocks[pos - 1].region.first > block.region.first))) {
            state->blocks[pos] = state->blocks[pos - 1];
            state->block_entry_indices[pos] = state->block_entry_indices[pos - 1];
            --pos;
        }

        state->blocks[pos] = block;
        state->block_entry_indices[pos] = entry_index;
    }
}

static NV_STATUS uvm_moe_slot_direct_flatten_blocks_locked(UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
                                                           uvm_va_space_t *va_space,
                                                           uvm_moe_slot_direct_batch_state_t *state,
                                                           NvBool fast_sort)
{
    NvU32 i;

    for (i = 0; i < params->count; ++i) {
        const UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[i];
        NvU64 entry_end = entry->va_base + entry->bytes;
        NvU64 cursor = entry->va_base;

        while (cursor < entry_end) {
            uvm_va_block_slot_direct_entry_t *block_entry;
            uvm_va_block_t *va_block = NULL;
            NvU64 slice_last;
            NvU64 slice_bytes;
            NV_STATUS status;

            if (state->block_count >= UVM_MOE_SLOT_DIRECT_MAX_BLOCKS)
                return NV_ERR_INSUFFICIENT_RESOURCES;

            status = uvm_va_block_find_create_managed(va_space, cursor, &va_block);
            if (status != NV_OK)
                return status;

            slice_last = min(entry_end - 1, va_block->end);
            slice_bytes = slice_last - cursor + 1;
            block_entry = &state->blocks[state->block_count];
            block_entry->va_block = va_block;
            block_entry->region = uvm_va_block_region_from_start_end(va_block, cursor, slice_last);
            block_entry->slot_mem = state->slots[i]->slot_mem;
            block_entry->slot_tracker = &state->slots[i]->async_copy_tracker;
            block_entry->slot_offset = entry->slot_offset + (cursor - entry->va_base);

            if (block_entry->slot_offset + slice_bytes > state->slots[i]->slot_bytes)
                return NV_ERR_OUT_OF_RANGE;

            state->block_entry_indices[state->block_count] = i;
            ++state->block_count;
            ++state->range_block_counts[i];
            cursor = slice_last + 1;
        }
    }

    uvm_moe_slot_direct_sort_blocks(state, fast_sort);

    for (i = 1; i < state->block_count; ++i) {
        const uvm_va_block_slot_direct_entry_t *prev = &state->blocks[i - 1];
        const uvm_va_block_slot_direct_entry_t *current_entry = &state->blocks[i];

        // Multiple slots may cover disjoint slices of the same 2 MiB managed
        // block. They share one block lock/tracker, but overlapping page masks
        // would install two physical owners for the same VA and are rejected.
        if (prev->va_block == current_entry->va_block &&
            prev->region.outer > current_entry->region.first) {
            return NV_ERR_INVALID_ARGUMENT;
        }
    }

    return NV_OK;
}

static NV_STATUS uvm_moe_slot_plan_target_recipe_create_locked(
    const UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    const uvm_moe_slot_direct_batch_state_t *state,
    NvU64 invalidation_epoch,
    uvm_moe_slot_plan_target_recipe_t **recipe_out)
{
    uvm_moe_slot_plan_target_recipe_t *recipe = NULL;
    NvU64 *entry_bytes = NULL;
    NV_STATUS status = NV_OK;
    NvU32 group_count = 0;
    NvU32 group_index = 0;
    NvU32 i;

    UVM_ASSERT(recipe_out);
    *recipe_out = NULL;

    if (!params || !state || params->count == 0 || state->block_count == 0)
        return NV_ERR_INVALID_ARGUMENT;

    for (i = 0; i < state->block_count; ++i) {
        if (i == 0 || state->blocks[i - 1].va_block != state->blocks[i].va_block)
            ++group_count;
    }

    recipe = uvm_kvmalloc_zero(sizeof(*recipe));
    if (!recipe)
        return NV_ERR_NO_MEMORY;

    recipe->slices = uvm_kvmalloc_zero((size_t)state->block_count * sizeof(*recipe->slices));
    recipe->groups = uvm_kvmalloc_zero((size_t)group_count * sizeof(*recipe->groups));
    entry_bytes = uvm_kvmalloc_zero((size_t)params->count * sizeof(*entry_bytes));
    if (!recipe->slices || !recipe->groups || !entry_bytes) {
        status = NV_ERR_NO_MEMORY;
        goto done;
    }

    recipe->slice_count = state->block_count;
    recipe->group_count = group_count;
    recipe->entry_count = params->count;
    recipe->pte_mode = state->blocks[0].pte_mode;
    recipe->invalidation_epoch = invalidation_epoch;

    for (i = 0; i < state->block_count; ++i) {
        const uvm_va_block_slot_direct_entry_t *block_entry = &state->blocks[i];
        const NvU32 entry_index = state->block_entry_indices[i];
        const UVM_MOE_SLOT_REMAP_ENTRY *entry;
        uvm_moe_slot_plan_target_slice_t *slice = &recipe->slices[i];
        uvm_moe_slot_plan_target_group_t *group;
        NvU64 slice_end;
        NvU64 entry_end;

        if (!block_entry->va_block || entry_index >= params->count ||
            block_entry->pte_mode != recipe->pte_mode) {
            status = NV_ERR_INVALID_STATE;
            goto done;
        }

        entry = &params->entries[entry_index];
        slice->va_base = uvm_va_block_region_start(block_entry->va_block,
                                                    block_entry->region);
        slice->bytes = uvm_va_block_region_size(block_entry->region);
        slice->entry_index = entry_index;
        slice_end = slice->va_base + slice->bytes;
        entry_end = entry->va_base + entry->bytes;
        if (slice->bytes == 0 || slice_end <= slice->va_base ||
            entry_end <= entry->va_base || slice->va_base < entry->va_base ||
            slice_end > entry_end || entry_bytes[entry_index] + slice->bytes < entry_bytes[entry_index] ||
            recipe->bytes + slice->bytes < recipe->bytes) {
            status = NV_ERR_INVALID_STATE;
            goto done;
        }
        entry_bytes[entry_index] += slice->bytes;
        recipe->bytes += slice->bytes;

        if (i == 0 || state->blocks[i - 1].va_block != block_entry->va_block) {
            group = &recipe->groups[group_index++];
            group->block_start = block_entry->va_block->start;
            group->block_end = block_entry->va_block->end;
            group->first_slice = i;
            uvm_page_mask_zero(&group->page_mask);
        }
        else {
            group = &recipe->groups[group_index - 1];
        }

        ++group->slice_count;
        uvm_page_mask_region_fill(&group->page_mask, block_entry->region);
    }

    if (group_index != recipe->group_count || recipe->bytes != params->requested_bytes) {
        status = NV_ERR_INVALID_STATE;
        goto done;
    }

    for (i = 0; i < params->count; ++i) {
        if (entry_bytes[i] != params->entries[i].bytes) {
            status = NV_ERR_INVALID_STATE;
            goto done;
        }
    }

    *recipe_out = recipe;
    recipe = NULL;

done:
    uvm_kvfree(entry_bytes);
    uvm_moe_slot_plan_target_recipe_destroy(recipe);
    return status;
}

static NV_STATUS uvm_moe_slot_plan_target_recipe_materialize_locked(
    const UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    uvm_va_space_t *va_space,
    const uvm_moe_slot_plan_target_recipe_t *recipe,
    uvm_slot_direct_pte_mode_t pte_mode,
    uvm_moe_slot_direct_batch_state_t *state)
{
    NvU64 materialized_bytes = 0;
    NvU32 output_index = 0;
    NvU32 group_index;

    uvm_assert_rwsem_locked_write(&va_space->lock);

    if (!params || !recipe || !state || recipe->entry_count != params->count ||
        recipe->pte_mode != pte_mode ||
        recipe->slice_count == 0 || recipe->slice_count > UVM_MOE_SLOT_DIRECT_MAX_BLOCKS ||
        recipe->group_count == 0 || recipe->group_count > recipe->slice_count ||
        recipe->bytes != params->requested_bytes ||
        recipe->invalidation_epoch != uvm_va_space_moe_slot_plan_invalidation_epoch(va_space)) {
        uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=target_recipe reason=recipe_header status=0x%x recipe=%px entry_count=%u params_count=%u recipe_mode=%u expected_mode=%u slices=%u groups=%u recipe_bytes=%llu requested_bytes=%llu recipe_epoch=%llu current_epoch=%llu\n",
                              NV_ERR_INVALID_STATE,
                              recipe,
                              recipe ? recipe->entry_count : 0,
                              params ? params->count : 0,
                              recipe ? recipe->pte_mode : 0,
                              pte_mode,
                              recipe ? recipe->slice_count : 0,
                              recipe ? recipe->group_count : 0,
                              recipe ? recipe->bytes : 0,
                              params ? params->requested_bytes : 0,
                              recipe ? recipe->invalidation_epoch : 0,
                              uvm_va_space_moe_slot_plan_invalidation_epoch(va_space));
        return NV_ERR_INVALID_STATE;
    }

    for (group_index = 0; group_index < recipe->group_count; ++group_index) {
        const uvm_moe_slot_plan_target_group_t *group = &recipe->groups[group_index];
        uvm_page_mask_t materialized_mask;
        uvm_va_block_t *va_block = NULL;
        NvU32 slice_offset;
        NV_STATUS status;

        if (group->slice_count == 0 || group->first_slice != output_index ||
            group->first_slice > recipe->slice_count ||
            group->slice_count > recipe->slice_count - group->first_slice ||
            group->block_end < group->block_start) {
            uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=target_recipe reason=group_shape status=0x%x group=%u first_slice=%u output_index=%u slice_count=%u recipe_slices=%u block_start=0x%llx block_end=0x%llx\n",
                                  NV_ERR_INVALID_STATE,
                                  group_index,
                                  group->first_slice,
                                  output_index,
                                  group->slice_count,
                                  recipe->slice_count,
                                  group->block_start,
                                  group->block_end);
            return NV_ERR_INVALID_STATE;
        }

        status = uvm_va_block_find(va_space, group->block_start, &va_block);
        if (status != NV_OK || !va_block || uvm_va_block_is_dead(va_block) ||
            uvm_va_block_is_hmm(va_block) ||
            uvm_va_block_get_va_space(va_block) != va_space ||
            va_block->start != group->block_start || va_block->end != group->block_end) {
            uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=target_recipe reason=block_identity status=0x%x group=%u find_status=0x%x block=%px dead=%u hmm=%u block_va_space=%px expected_va_space=%px cached_start=0x%llx cached_end=0x%llx actual_start=0x%llx actual_end=0x%llx\n",
                                  NV_ERR_INVALID_STATE,
                                  group_index,
                                  status,
                                  va_block,
                                  va_block ? uvm_va_block_is_dead(va_block) : 0,
                                  va_block ? uvm_va_block_is_hmm(va_block) : 0,
                                  va_block ? uvm_va_block_get_va_space(va_block) : NULL,
                                  va_space,
                                  group->block_start,
                                  group->block_end,
                                  va_block ? va_block->start : 0,
                                  va_block ? va_block->end : 0);
            return NV_ERR_INVALID_STATE;
        }

        uvm_page_mask_zero(&materialized_mask);
        for (slice_offset = 0; slice_offset < group->slice_count; ++slice_offset) {
            const uvm_moe_slot_plan_target_slice_t *slice =
                &recipe->slices[group->first_slice + slice_offset];
            const UVM_MOE_SLOT_REMAP_ENTRY *entry;
            uvm_va_block_slot_direct_entry_t *block_entry;
            NvU64 slice_end = slice->va_base + slice->bytes;
            NvU64 entry_end;

            if (slice->entry_index >= params->count || !state->slots[slice->entry_index] ||
                slice->bytes == 0 || slice_end <= slice->va_base ||
                slice->va_base < va_block->start || slice_end > va_block->end + 1) {
                uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=target_recipe reason=slice_shape status=0x%x group=%u slice=%u entry=%u params_count=%u slot=%px va=0x%llx bytes=%llu block_start=0x%llx block_end=0x%llx\n",
                                      NV_ERR_INVALID_STATE,
                                      group_index,
                                      group->first_slice + slice_offset,
                                      slice->entry_index,
                                      params->count,
                                      slice->entry_index < params->count ? state->slots[slice->entry_index] : NULL,
                                      slice->va_base,
                                      slice->bytes,
                                      va_block->start,
                                      va_block->end);
                return NV_ERR_INVALID_STATE;
            }

            entry = &params->entries[slice->entry_index];
            entry_end = entry->va_base + entry->bytes;
            if (entry_end <= entry->va_base || slice->va_base < entry->va_base ||
                slice_end > entry_end) {
                uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=target_recipe reason=slice_entry_range status=0x%x group=%u slice=%u entry=%u slice_va=0x%llx slice_end=0x%llx entry_va=0x%llx entry_end=0x%llx\n",
                                      NV_ERR_INVALID_STATE,
                                      group_index,
                                      group->first_slice + slice_offset,
                                      slice->entry_index,
                                      slice->va_base,
                                      slice_end,
                                      entry->va_base,
                                      entry_end);
                return NV_ERR_INVALID_STATE;
            }

            block_entry = &state->blocks[output_index];
            block_entry->va_block = va_block;
            block_entry->region = uvm_va_block_region_from_start_end(va_block,
                                                                      slice->va_base,
                                                                      slice_end - 1);
            block_entry->slot_mem = state->slots[slice->entry_index]->slot_mem;
            block_entry->slot_tracker = &state->slots[slice->entry_index]->async_copy_tracker;
            block_entry->slot_offset = entry->slot_offset + (slice->va_base - entry->va_base);
            block_entry->pte_mode = recipe->pte_mode;
            if (block_entry->slot_offset + slice->bytes < block_entry->slot_offset ||
                block_entry->slot_offset + slice->bytes > state->slots[slice->entry_index]->slot_bytes) {
                uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=target_recipe reason=slot_range status=0x%x group=%u slice=%u entry=%u slot_offset=%llu bytes=%llu slot_bytes=%llu\n",
                                      NV_ERR_INVALID_STATE,
                                      group_index,
                                      group->first_slice + slice_offset,
                                      slice->entry_index,
                                      block_entry->slot_offset,
                                      slice->bytes,
                                      state->slots[slice->entry_index]->slot_bytes);
                return NV_ERR_INVALID_STATE;
            }

            state->block_entry_indices[output_index] = slice->entry_index;
            ++state->range_block_counts[slice->entry_index];
            ++output_index;
            materialized_bytes += slice->bytes;
            uvm_page_mask_region_fill(&materialized_mask, block_entry->region);
        }

        if (!uvm_page_mask_equal(&materialized_mask, &group->page_mask)) {
            uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=target_recipe reason=page_mask status=0x%x group=%u first_slice=%u slice_count=%u\n",
                                  NV_ERR_INVALID_STATE,
                                  group_index,
                                  group->first_slice,
                                  group->slice_count);
            return NV_ERR_INVALID_STATE;
        }
    }

    if (output_index != recipe->slice_count || materialized_bytes != recipe->bytes) {
        uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=target_recipe reason=totals status=0x%x output=%u expected_slices=%u bytes=%llu expected_bytes=%llu\n",
                              NV_ERR_INVALID_STATE,
                              output_index,
                              recipe->slice_count,
                              materialized_bytes,
                              recipe->bytes);
        return NV_ERR_INVALID_STATE;
    }

    state->block_count = output_index;
    return NV_OK;
}

static void uvm_moe_slot_direct_init_slot_trackers(UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
                                                   uvm_moe_slot_direct_batch_state_t *state)
{
    NvU32 i;

    for (i = 0; i < params->count; ++i) {
        uvm_moe_slot_pool_slot_t *slot = state->slots[i];

        uvm_tracker_init(&slot->async_copy_tracker);
        slot->async_copy_staging = NULL;
        slot->async_copy_pending = NV_TRUE;
        slot->async_copy_bytes = params->entries[i].bytes;
        slot->async_copy_slot_offset = params->entries[i].slot_offset;
        slot->copy_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
        slot->state = UVM_MOE_SLOT_POOL_STATE_LOADING;
        state->slot_tracker_initialized[i] = NV_TRUE;
    }
}

static NV_STATUS uvm_moe_slot_direct_deinit_slot_trackers(UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
                                                          uvm_moe_slot_direct_batch_state_t *state,
                                                          NvBool wait_for_completion)
{
    NV_STATUS first_error = NV_OK;
    NvU32 i;

    for (i = 0; i < params->count; ++i) {
        uvm_moe_slot_pool_slot_t *slot;
        NV_STATUS status;

        if (!state->slot_tracker_initialized[i])
            continue;

        slot = state->slots[i];
        status = wait_for_completion ? uvm_tracker_wait(&slot->async_copy_tracker) : NV_OK;
        if (first_error == NV_OK)
            first_error = status;
        uvm_tracker_deinit(&slot->async_copy_tracker);
        slot->async_copy_pending = NV_FALSE;
        slot->async_copy_bytes = 0;
        slot->async_copy_slot_offset = 0;
        state->slot_tracker_initialized[i] = NV_FALSE;
    }

    return first_error;
}

static NV_STATUS uvm_moe_slot_direct_record_pte_range_locked(uvm_moe_slot_pool_slot_t *slot,
                                                             uvm_va_space_t *va_space,
                                                             uvm_gpu_t *gpu,
                                                             const UVM_MOE_SLOT_REMAP_ENTRY *entry,
                                                             NvU64 va_base,
                                                             NvU64 bytes)
{
    if (!slot->pte_gpu) {
        UVM_ASSERT(slot->pte_range_count == 0);
        UVM_ASSERT(slot->pte_bytes == 0);
        slot->pte_gpu = gpu;
        uvm_gpu_retain(gpu);
        slot->pte_va_space = va_space;
        slot->pte_slot_offset = entry->slot_offset;
        // Reuse the established VA-block cleanup path. The ioctl mode remains
        // visible through its counters; pte_mode describes cleanup semantics.
        slot->pte_mode = UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP;
        slot->pte_va_block_tracked = NV_TRUE;
        slot->pte_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
    }
    else if (slot->pte_gpu != gpu || slot->pte_va_space != va_space ||
             slot->pte_mode != UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP) {
        return NV_ERR_INVALID_STATE;
    }

    slot->last_role = entry->role;
    return uvm_moe_slot_pool_record_pte_range_locked(slot, va_base, bytes);
}

static NV_STATUS uvm_moe_slot_direct_record_all_pte_metadata_locked(
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    uvm_va_space_t *va_space,
    uvm_gpu_t *gpu,
    uvm_moe_slot_direct_batch_state_t *state)
{
    NvU32 i;

    for (i = 0; i < params->count; ++i) {
        NV_STATUS status = uvm_moe_slot_direct_record_pte_range_locked(state->slots[i],
                                                                       va_space,
                                                                       gpu,
                                                                       &params->entries[i],
                                                                       params->entries[i].va_base,
                                                                       params->entries[i].bytes);
        if (status != NV_OK)
            return status;
    }

    return NV_OK;
}

static NV_STATUS uvm_moe_slot_direct_record_surviving_pte_metadata_locked(
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    uvm_va_space_t *va_space,
    uvm_gpu_t *gpu,
    uvm_moe_slot_direct_batch_state_t *state)
{
    NvU32 i;

    for (i = 0; i < state->block_count; ++i) {
        uvm_va_block_slot_direct_entry_t *block = &state->blocks[i];
        NvU32 entry_index = state->block_entry_indices[i];

        if (block->mapped)
            state->has_surviving_mapping[entry_index] = NV_TRUE;
    }

    for (i = 0; i < params->count; ++i) {
        NV_STATUS status;

        if (!state->has_surviving_mapping[i])
            continue;

        // Track the whole original range. Cleanup unmapping an already-clear
        // slice is harmless, and this keeps one metadata range per slot even
        // if a pathological 4 MiB input spans many short VA blocks.
        status = uvm_moe_slot_direct_record_pte_range_locked(state->slots[i],
                                                              va_space,
                                                              gpu,
                                                              &params->entries[i],
                                                              params->entries[i].va_base,
                                                              params->entries[i].bytes);
        if (status != NV_OK)
            return status;
    }

    return NV_OK;
}

static NV_STATUS uvm_moe_slot_direct_rollback_recorded_pte_locked(
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    uvm_va_space_t *va_space,
    struct mm_struct *mm,
    uvm_moe_slot_direct_batch_state_t *state,
    NvBool fail_closed)
{
    NV_STATUS first_error = NV_OK;
    NvU32 i = params->count;

    while (i-- > 0) {
        uvm_moe_slot_pool_slot_t *slot = state->slots[i];
        NV_STATUS status;

        if (!slot->pte_gpu && slot->pte_bytes == 0 && slot->pte_range_count == 0)
            continue;

        status = uvm_moe_slot_pool_rollback_pte_mapping_mode_locked(slot,
                                                                    va_space,
                                                                    mm,
                                                                    !fail_closed);
        if (status != NV_OK) {
            uvm_moe_slot_pool_note_pte_failure_locked(slot);
            if (first_error == NV_OK)
                first_error = status;
        }
    }

    return first_error;
}

static NV_STATUS uvm_moe_slot_direct_fail_closed_cleanup_locked(
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    uvm_va_space_t *va_space,
    struct mm_struct *mm,
    uvm_gpu_t *gpu,
    uvm_moe_slot_direct_batch_state_t *state,
    uvm_va_block_context_t *cleanup_block_context,
    NvBool targets_flattened,
    NvBool slots_validated)
{
    NV_STATUS first_error = NV_OK;
    NV_STATUS status;

    // Clear every recorded slot PTE first. In particular, this preserves the
    // complete-leaf checks of the slot-specific unmap path before the generic
    // target sweep is allowed to split any ordinary large mapping.
    if (slots_validated) {
        status = uvm_moe_slot_direct_rollback_recorded_pte_locked(params,
                                                                  va_space,
                                                                  mm,
                                                                  state,
                                                                  NV_TRUE);
        if (first_error == NV_OK)
            first_error = status;
    }

    // Then clear the complete target set. This covers both unrecorded
    // replacement slot PTEs and ordinary CPU-backed GPU PTEs in slices which a
    // failed invalidate or prepare loop never reached. Continue after a
    // recorded-metadata error to make the best possible fail-closed sweep.
    if (targets_flattened) {
        UVM_ASSERT(cleanup_block_context);
        status = uvm_va_block_slot_direct_fail_closed_unmap_batch(
            va_space,
            cleanup_block_context,
            gpu,
            state->blocks,
            state->block_count);
        if (first_error == NV_OK)
            first_error = status;
    }

    return first_error;
}

static NvU32 uvm_moe_slot_direct_failure_entry_status(NV_STATUS status,
                                                       const uvm_va_block_slot_direct_result_t *result,
                                                       NvBool tracker_failed)
{
    if (result->failure_stage == UVM_SLOT_DIRECT_FAILURE_COPY || tracker_failed)
        return UVM_MOE_SLOT_REMAP_ENTRY_COPY_FAILED;

    if (status == NV_ERR_NOT_SUPPORTED || status == NV_WARN_MORE_PROCESSING_REQUIRED)
        return UVM_MOE_SLOT_REMAP_ENTRY_REMAP_NOT_SUPPORTED;

    return uvm_moe_slot_pte_status_from_nv_status(status);
}

static NV_STATUS uvm_moe_slot_plan_entry_identity_locked(
    uvm_va_space_t *va_space,
    const UVM_MOE_SLOT_REMAP_ENTRY *entry,
    NvU64 *identity_out)
{
    NvU64 cursor = entry->va_base;
    NvU64 end = entry->va_base + entry->bytes;
    NvU64 identity = 0;

    uvm_assert_rwsem_locked(&va_space->lock);

    if (entry->bytes == 0 || end <= cursor)
        return NV_ERR_INVALID_ADDRESS;

    while (cursor < end) {
        uvm_va_range_managed_t *managed_range = uvm_va_range_managed_find(va_space, cursor);
        NvU64 next;

        if (!managed_range ||
            !managed_range->vma_wrapper ||
            managed_range->moe_slot_allocation_id == 0 ||
            managed_range->va_range.node.start > cursor ||
            managed_range->va_range.node.end < cursor) {
            return NV_ERR_INVALID_ADDRESS;
        }

        if (identity == 0)
            identity = managed_range->moe_slot_allocation_id;
        else if (identity != managed_range->moe_slot_allocation_id)
            return NV_ERR_INVALID_ADDRESS;

        next = min(end, managed_range->va_range.node.end + 1);
        if (next <= cursor)
            return NV_ERR_INVALID_ADDRESS;
        cursor = next;
    }

    *identity_out = identity;
    return NV_OK;
}

static NV_STATUS uvm_moe_slot_plan_capture_identities_locked(
    const UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    uvm_va_space_t *va_space,
    NvU64 *identities)
{
    NvU32 i;

    for (i = 0; i < params->count; ++i) {
        NV_STATUS status = uvm_moe_slot_plan_entry_identity_locked(va_space,
                                                                   &params->entries[i],
                                                                   &identities[i]);
        if (status != NV_OK)
            return status;
    }

    return NV_OK;
}

static NV_STATUS uvm_moe_slot_plan_validate_identities_locked(
    const UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    uvm_va_space_t *va_space,
    const NvU64 *expected_identities)
{
    NvU32 i;

    for (i = 0; i < params->count; ++i) {
        NvU64 identity = 0;
        NV_STATUS status = uvm_moe_slot_plan_entry_identity_locked(va_space,
                                                                   &params->entries[i],
                                                                   &identity);
        if (status != NV_OK || identity != expected_identities[i])
            return NV_ERR_INVALID_STATE;
    }

    return NV_OK;
}

static NV_STATUS uvm_moe_slot_pool_apply_slot_direct_batch_configured(
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    uvm_va_space_t *va_space,
    NV_STATUS validation_status,
    NvBool layer_transition,
    NvBool shared_push,
    uvm_slot_direct_pte_mode_t pte_mode,
    const NvU64 *expected_identities,
    NvU64 expected_invalidation_epoch,
    const uvm_moe_slot_plan_validation_recipe_t *validation_recipe,
    const uvm_va_block_slot_copy_plan_t *copy_plan,
    const uvm_moe_slot_plan_target_recipe_t *target_recipe,
    const UVM_MOE_SLOT_REMAP_BATCH_PARAMS *resident_clear_batch,
    const uvm_moe_slot_plan_target_recipe_t *resident_clear_recipe,
    NvBool metadata_recipe_cache_enabled,
    NvBool epoch_validation_enabled,
    NvBool resident_bank_empty,
    NvBool dual_bank_copy_first,
    NvBool shadow_pte_prefill,
    NvU64 *commit_generation_out)
{
    const NvBool registered_plan = commit_generation_out != NULL;
    const NvBool fast_index = layer_transition &&
                              uvm_moe_slot_transition_fast_index_enabled(params->mode);
    const NvBool strict_64k_overwrite =
        params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_OVERWRITE ||
        params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_OVERWRITE_STAGE_SYNC;
    const NvBool skip_unmap = params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_OVERWRITE ||
                              strict_64k_overwrite;
    const NvBool stage_sync = params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_STAGE_SYNC ||
                              params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_STAGE_SYNC ||
                              params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_OVERWRITE_STAGE_SYNC;
    uvm_slot_direct_batch_telemetry_t telemetry = {
        .shared_push = shared_push,
        .stage_sync = stage_sync,
        .shadow_pte_prefill = shadow_pte_prefill,
        .metadata_recipe_cache_enabled = metadata_recipe_cache_enabled,
        .requested_ranges = params->count,
        .requested_bytes = params->requested_bytes,
    };
    uvm_va_block_slot_direct_result_t result = { 0 };
    uvm_moe_slot_transition_clear_result_t transition_clear_result = { 0 };
    uvm_moe_slot_transition_prepared_clear_t prepared_dual_clear = { 0 };
    uvm_moe_slot_direct_batch_state_t *state = NULL;
    uvm_va_block_context_t *cleanup_block_context = NULL;
    uvm_tracker_t batch_tracker = UVM_TRACKER_INIT();
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    NV_STATUS status = validation_status;
    NV_STATUS helper_status = NV_OK;
    NV_STATUS metadata_status = NV_OK;
    NV_STATUS batch_wait_status = NV_OK;
    NV_STATUS slot_wait_status = NV_OK;
    NV_STATUS rollback_status = NV_OK;
    NV_STATUS fail_closed_cleanup_status = NV_OK;
    NV_STATUS old_clear_cleanup_status = NV_OK;
    NvU64 total_start = NV_GETTIME();
    NvU64 detail_cursor = total_start;
    NvU64 detail_now = total_start;
    NvU64 helper_start = total_start;
    NvU64 helper_end = total_start;
    NvU64 total_end;
    NvU64 wait_start = 0;
    NvU64 allocation_count_before = 0;
    NvU64 eviction_count_before = 0;
    NvU64 batch_id = 0;
    NvU64 commit_generation = 0;
    NvBool pool_locked = NV_FALSE;
    NvBool va_space_locked = NV_FALSE;
    NvBool mm_retained = NV_FALSE;
    NvBool trackers_initialized = NV_FALSE;
    NvBool batch_tracker_deinitialized = NV_FALSE;
    NvBool batch_tracker_may_have_work = NV_FALSE;
    NvBool attempted = NV_FALSE;
    NvBool slots_validated = NV_FALSE;
    NvBool targets_flattened = NV_FALSE;
    NvBool fail_closed_cleanup_attempted = NV_FALSE;
    NvBool old_clear_cleanup_attempted = NV_FALSE;
    NvBool target_metadata_recorded = NV_FALSE;
    NvBool fail_closed_transaction_started = NV_FALSE;
    NvBool clear_recipe_cache_hit = NV_FALSE;
    NvU32 failure_entry_status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
    NvU32 i;

    if (commit_generation_out)
        *commit_generation_out = 0;

    if (validation_status != NV_OK) {
        uvm_moe_slot_remap_fail_ok_entries(params, UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED);
        goto done;
    }

    if (params->count == 0) {
        status = NV_OK;
        telemetry.success = true;
        goto done;
    }

    if (!uvm_moe_slot_pte_remap_enable) {
        status = NV_ERR_NOT_SUPPORTED;
        uvm_moe_slot_remap_fail_ok_entries(params, UVM_MOE_SLOT_REMAP_ENTRY_REMAP_NOT_SUPPORTED);
        goto done;
    }

    detail_now = NV_GETTIME();
    telemetry.gate_ns = detail_now - detail_cursor;
    detail_cursor = detail_now;
    state = uvm_kvmalloc_zero(sizeof(*state));
    if (!state) {
        status = NV_ERR_NO_MEMORY;
        uvm_moe_slot_remap_fail_ok_entries(params, UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED);
        goto done;
    }

    detail_now = NV_GETTIME();
    telemetry.state_alloc_ns = detail_now - detail_cursor;
    detail_cursor = detail_now;
    // Slot-direct operates on an already-registered VA space, so retain the
    // existing GPU object directly instead of rebuilding its state through
    // the full RM GetGpuInfo path on every ioctl.
    gpu = uvm_va_space_retain_gpu_by_uuid(va_space, &params->gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        uvm_moe_slot_remap_fail_ok_entries(params, UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED);
        goto done;
    }

    detail_now = NV_GETTIME();
    telemetry.gpu_retain_ns = detail_now - detail_cursor;
    detail_cursor = detail_now;
    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    mm_retained = NV_TRUE;
    uvm_va_space_down_write(va_space);
    va_space_locked = NV_TRUE;
    mutex_lock(&g_uvm_moe_slot_pool_lock);
    pool_locked = NV_TRUE;

    detail_now = NV_GETTIME();
    telemetry.outer_lock_acquire_ns = detail_now - detail_cursor;
    detail_cursor = detail_now;
    allocation_count_before = g_uvm_moe_slot_pool_vidmem_alloc_count;
    eviction_count_before = g_uvm_moe_slot_pool_vidmem_evict_count;
    batch_id = ++g_uvm_moe_slot_pool_current_batch_id;

    // Registered plans cache allocation identities rather than trusting raw
    // virtual addresses. Revalidate both the mutation epoch and every managed
    // allocation while the VA-space write lock excludes free/split/unregister,
    // and before slot validation, owner cutover, DMA, or PTE work begins.
    if (expected_identities) {
        NvU64 current_invalidation_epoch =
            uvm_va_space_moe_slot_plan_invalidation_epoch(va_space);

        if (current_invalidation_epoch != expected_invalidation_epoch) {
            status = NV_ERR_INVALID_STATE;
            uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=identity_epoch reason=epoch_mismatch batch=%llu mode=%u layer=%u status=0x%x expected_epoch=%llu current_epoch=%llu\n",
                                  batch_id,
                                  params->mode,
                                  params->entries[0].layer_id,
                                  status,
                                  expected_invalidation_epoch,
                                  current_invalidation_epoch);
            uvm_moe_slot_remap_fail_ok_entries(params, UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED);
            goto unlock;
        }

        if (!epoch_validation_enabled) {
            status = uvm_moe_slot_plan_validate_identities_locked(params,
                                                                   va_space,
                                                                   expected_identities);
            if (status != NV_OK) {
                uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=allocation_identity reason=identity_mismatch batch=%llu mode=%u layer=%u status=0x%x count=%u\n",
                                      batch_id,
                                      params->mode,
                                      params->entries[0].layer_id,
                                      status,
                                      params->count);
                status = NV_ERR_INVALID_STATE;
                uvm_moe_slot_remap_fail_ok_entries(params, UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED);
                goto unlock;
            }
        }
    }

    // The retain above protects the uvm_gpu_t lifetime, but the GPU can be
    // unregistered from this VA space before we acquire its write lock. The
    // direct PTE path requires an active GPU VA space, so revalidate it while
    // registration state is stable under the lock.
    if (uvm_va_space_get_gpu_by_uuid_with_gpu_va_space(va_space, &params->gpu_uuid) != gpu) {
        status = NV_ERR_INVALID_DEVICE;
        uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=gpu_va_space reason=gpu_unregistered batch=%llu mode=%u layer=%u status=0x%x gpu=%px\n",
                              batch_id,
                              params->mode,
                              params->entries[0].layer_id,
                              status,
                              gpu);
        uvm_moe_slot_remap_fail_ok_entries(params, UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED);
        goto unlock;
    }

    detail_now = NV_GETTIME();
    telemetry.batch_bookkeeping_ns = detail_now - detail_cursor;
    detail_cursor = detail_now;
    if (registered_plan && validation_recipe) {
        status = epoch_validation_enabled ?
                     uvm_moe_slot_plan_bind_slots_epoch_locked(params,
                                                                va_space,
                                                                gpu,
                                                                state,
                                                                validation_recipe,
                                                                expected_invalidation_epoch) :
                     uvm_moe_slot_plan_validate_slots_cached_locked(params,
                                                                     va_space,
                                                                     gpu,
                                                                     state,
                                                                     validation_recipe,
                                                                     layer_transition);
    }
    else {
        status = layer_transition ?
                     (fast_index ?
                          uvm_moe_slot_transition_validate_slots_fast_locked(params, va_space, gpu, state) :
                          uvm_moe_slot_transition_validate_slots_locked(params, va_space, gpu, state)) :
                     uvm_moe_slot_direct_validate_slots_locked(params, va_space, gpu, state);
    }
    if (status != NV_OK) {
        uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=slot_validate reason=helper_reject batch=%llu mode=%u layer=%u status=0x%x fast=%u cached=%u count=%u failures=%u\n",
                              batch_id,
                              params->mode,
                              params->entries[0].layer_id,
                              status,
                              fast_index,
                              registered_plan && validation_recipe,
                              params->count,
                              params->failure_count);
        // validate_slots marks the precise offending entry. Initial ioctl
        // validation was clean, so this is the only failure counted so far.
        ++params->failure_count;
        uvm_moe_slot_remap_fail_ok_entries(params, UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED);
        goto unlock;
    }
    slots_validated = NV_TRUE;
    if (layer_transition && fast_index)
        telemetry.transition_fast_index = true;

    detail_now = NV_GETTIME();
    telemetry.slot_validate_ns = detail_now - detail_cursor;
    detail_cursor = detail_now;
    status = metadata_recipe_cache_enabled ?
                 uvm_moe_slot_plan_target_recipe_materialize_locked(params,
                                                                     va_space,
                                                                     target_recipe,
                                                                     pte_mode,
                                                                     state) :
                 uvm_moe_slot_direct_flatten_blocks_locked(params,
                                                            va_space,
                                                            state,
                                                            fast_index);
    if (status != NV_OK) {
        uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=target_materialize reason=helper_reject batch=%llu mode=%u layer=%u status=0x%x cached=%u count=%u\n",
                              batch_id,
                              params->mode,
                              params->entries[0].layer_id,
                              status,
                              metadata_recipe_cache_enabled,
                              params->count);
        uvm_moe_slot_remap_fail_ok_entries(params,
                                           status == NV_ERR_INVALID_ARGUMENT || status == NV_ERR_OUT_OF_RANGE ?
                                               UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ARGUMENT :
                                               UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED);
        goto unlock;
    }
    targets_flattened = NV_TRUE;
    telemetry.target_recipe_cache_hit = metadata_recipe_cache_enabled;

    detail_now = NV_GETTIME();
    telemetry.flatten_sort_ns = detail_now - detail_cursor;
    detail_cursor = detail_now;
    for (i = 0; i < state->block_count; ++i)
        state->blocks[i].pte_mode = pte_mode;

    if (layer_transition || registered_plan) {
        // A target which already contains a slot PTE has ambiguous external
        // ownership, and a fail-closed generic sweep could otherwise split an
        // unrelated slot-backed large leaf. Reject it before TRANSITIONING.
        status = uvm_va_block_slot_direct_preflight_targets_not_slot_mapped(
            va_space,
            gpu,
            state->blocks,
            state->block_count);
        if (status != NV_OK) {
            result.failure_stage = UVM_SLOT_DIRECT_FAILURE_ELIGIBILITY;
            uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=target_not_slot reason=slot_mapping_present batch=%llu mode=%u layer=%u status=0x%x blocks=%u\n",
                                  batch_id,
                                  params->mode,
                                  params->entries[0].layer_id,
                                  status,
                                  state->block_count);
            uvm_moe_slot_remap_fail_ok_entries(params,
                                               UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED);
            goto unlock;
        }

        // Fail-closed cleanup must not depend on allocating memory after a copy
        // or PTE push fails. Reserve its context before entering the transaction.
        cleanup_block_context = uvm_va_block_context_alloc(mm);
        if (!cleanup_block_context) {
            status = NV_ERR_NO_MEMORY;
            uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=cleanup_context reason=allocation_failed batch=%llu mode=%u layer=%u status=0x%x\n",
                                  batch_id,
                                  params->mode,
                                  params->entries[0].layer_id,
                                  status);
            uvm_moe_slot_remap_fail_ok_entries(params,
                                               UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED);
            goto unlock;
        }
    }

    if ((layer_transition || registered_plan) &&
        pte_mode == UVM_SLOT_DIRECT_PTE_MODE_64K) {
        // Mode 20 publishes an irreversible owner cutover. Prove every 64K
        // source, full-block slice layout, and destination physical run before
        // clearing an old PTE or changing an owner. The regular helper repeats
        // these checks under its execution locks, but must never be the first
        // place a large-page incompatibility is discovered after cutover.
        status = uvm_va_block_slot_direct_preflight_64k_batch(va_space,
                                                              mm,
                                                              gpu,
                                                              state->blocks,
                                                              state->block_count);
        detail_now = NV_GETTIME();
        telemetry.transition_preflight_ns = detail_now - detail_cursor;
        detail_cursor = detail_now;
        if (status != NV_OK) {
            result.failure_stage = UVM_SLOT_DIRECT_FAILURE_ELIGIBILITY;
            uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=preflight_64k reason=helper_reject batch=%llu mode=%u layer=%u status=0x%x blocks=%u\n",
                                  batch_id,
                                  params->mode,
                                  params->entries[0].layer_id,
                                  status,
                                  state->block_count);
            params->remap_count = 0;
            failure_entry_status = uvm_moe_slot_direct_failure_entry_status(status,
                                                                             &result,
                                                                             NV_FALSE);
            uvm_moe_slot_remap_fail_ok_entries(params, failure_entry_status);
            goto unlock;
        }
    }

    // Cached physical addresses are an optimization, never an identity
    // contract. Validate immutable geometry, CPU residency/NID, destination
    // backing, and each source block's physical epoch before reserving a
    // generation or entering TRANSITIONING. A stale cache is rejected rather
    // than silently replayed or discovered after slot contents are overwritten.
    if (copy_plan && !epoch_validation_enabled) {
        NvU64 copy_cache_validate_start = NV_GETTIME();

        status = uvm_va_block_slot_copy_plan_validate(va_space,
                                                       mm,
                                                       gpu,
                                                       state->blocks,
                                                       state->block_count,
                                                       copy_plan);
        telemetry.copy_run_cache_validate_ns += NV_GETTIME() - copy_cache_validate_start;
        if (status != NV_OK) {
            result.failure_stage = UVM_SLOT_DIRECT_FAILURE_ELIGIBILITY;
            uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=copy_cache reason=validation_failed batch=%llu mode=%u layer=%u status=0x%x blocks=%u cache_entries=%zu cache_runs=%zu cache_bytes=%llu\n",
                                  batch_id,
                                  params->mode,
                                  params->entries[0].layer_id,
                                  status,
                                  state->block_count,
                                  copy_plan->entry_count,
                                  copy_plan->run_count,
                                  copy_plan->bytes);
            uvm_moe_slot_remap_fail_ok_entries(params,
                                               UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED);
            goto unlock;
        }
    }

    if (dual_bank_copy_first) {
        NvU64 dual_clear_preflight_start = NV_GETTIME();

        // Copy-first is safe only when the complete old mapping recipe and all
        // cleanup resources are proven before the first CE write. Do not fall
        // back to per-slot teardown after copy submission.
        if (!layer_transition ||
            !fast_index ||
            !uvm_moe_slot_transition_direct_cutover_enabled(params->mode)) {
            status = NV_ERR_INVALID_STATE;
        }
        else {
            status = uvm_moe_slot_transition_prepare_dual_clear_locked(
                params,
                resident_clear_batch,
                va_space,
                mm,
                gpu,
                state,
                resident_clear_recipe,
                metadata_recipe_cache_enabled,
                resident_bank_empty,
                &transition_clear_result,
                &prepared_dual_clear,
                &clear_recipe_cache_hit);
        }
        detail_now = NV_GETTIME();
        telemetry.transition_preflight_ns += detail_now - dual_clear_preflight_start;
        detail_cursor = detail_now;
        if (status != NV_OK) {
            result.failure_stage = UVM_SLOT_DIRECT_FAILURE_ELIGIBILITY;
            uvm_phase_f_route_log("moe_slot_plan_preflight_reject stage=dual_clear reason=preflight_failed batch=%llu mode=%u layer=%u status=0x%x resident_empty=%u cached_recipe=%u blocks=%u\n",
                                  batch_id,
                                  params->mode,
                                  params->entries[0].layer_id,
                                  status,
                                  resident_bank_empty,
                                  metadata_recipe_cache_enabled,
                                  state->block_count);
            uvm_moe_slot_remap_fail_ok_entries(params,
                                               UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED);
            goto unlock;
        }
    }

    // Validation, immutable-target flattening, large-leaf eligibility, and all
    // failure-cleanup resources are now complete. Only this point begins an
    // execution-stage transaction. Earlier failures have touched no PTE and do
    // not poison an otherwise active registered-plan session.
    if (registered_plan) {
        status = uvm_moe_slot_plan_next_generation_locked(&commit_generation);
        if (status != NV_OK) {
            uvm_moe_slot_remap_fail_ok_entries(params,
                                               UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED);
            goto unlock;
        }

        UVM_ASSERT(g_uvm_moe_slot_plan_state == UVM_MOE_SLOT_PLAN_STATE_ACTIVE);
        *commit_generation_out = commit_generation;
        g_uvm_moe_slot_plan_transition_id = ++g_uvm_moe_slot_plan_transition_counter;
        g_uvm_moe_slot_plan_state = UVM_MOE_SLOT_PLAN_STATE_TRANSITIONING;
    }
    fail_closed_transaction_started = layer_transition || registered_plan;
    telemetry.dual_bank_copy_first = dual_bank_copy_first;
    telemetry.clear_recipe_cache_hit = clear_recipe_cache_hit;

    if (layer_transition && !dual_bank_copy_first) {
        NvU64 transition_clear_start = NV_GETTIME();

        status = uvm_moe_slot_transition_begin_locked(params,
                                                       va_space,
                                                       mm,
                                                       gpu,
                                                       state,
                                                       batch_id,
                                                       fast_index,
                                                       &transition_clear_result,
                                                       NULL);
        detail_now = NV_GETTIME();
        telemetry.transition_clear_rebind_ns = detail_now - transition_clear_start;
        telemetry.transition_group_clear = transition_clear_result.used;
        telemetry.transition_group_clear_fallback = transition_clear_result.fallback;
        telemetry.transition_clear_ranges = transition_clear_result.ranges;
        telemetry.transition_clear_unique_blocks = transition_clear_result.unique_blocks;
        telemetry.transition_direct_cutover = transition_clear_result.direct_cutover;
        telemetry.transition_direct_cutover_pages = transition_clear_result.direct_cutover_pages;
        detail_cursor = detail_now;
        if (status != NV_OK) {
            // The precise teardown failure was marked by the helper. No owner
            // cutover occurred, so remap_count stays zero and every remaining
            // entry is failed before returning to userspace.
            ++params->failure_count;
            uvm_moe_slot_remap_fail_ok_entries(params,
                                               UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED);
            goto unlock;
        }
    }
    else if (layer_transition) {
        // The complete old-PTE recipe was preflighted above. Make the inactive
        // bank non-active in software before its first CE write; actual clear
        // pushes are submitted after copy starts.
        uvm_moe_slot_transition_mark_loading_locked(params, state, batch_id);
        telemetry.transition_group_clear = transition_clear_result.used;
        telemetry.transition_group_clear_fallback = transition_clear_result.fallback;
        telemetry.transition_clear_ranges = transition_clear_result.ranges;
        telemetry.transition_clear_unique_blocks = transition_clear_result.unique_blocks;
        telemetry.transition_direct_cutover = transition_clear_result.direct_cutover;
        telemetry.transition_direct_cutover_pages = transition_clear_result.direct_cutover_pages;
    }

    detail_now = NV_GETTIME();
    uvm_moe_slot_direct_init_slot_trackers(params, state);
    trackers_initialized = NV_TRUE;
    attempted = NV_TRUE;

    helper_start = NV_GETTIME();
    telemetry.slot_tracker_init_ns = helper_start - detail_now;
    telemetry.pre_helper_total_ns = helper_start - total_start;
    if (dual_bank_copy_first) {
        // The helper can submit one or more pushes before a later method or
        // tracker operation reports failure.
        batch_tracker_may_have_work = NV_TRUE;
        helper_status = uvm_va_block_slot_direct_submit_cached_copy(
            va_space,
            mm,
            gpu,
            state->blocks,
            state->block_count,
            shared_push,
            copy_plan,
            &g_uvm_moe_slot_pool_lock,
            &batch_tracker,
            &result);

        {
            NvU64 transition_clear_start = NV_GETTIME();
            NV_STATUS clear_status;

            // Copy is now running against the physically disjoint inactive
            // bank. Submit the already-preflighted old-PTE clear while CE is
            // busy. This runs even after a partial copy-submit failure so no
            // overwritten old owner remains GPU-visible on the poison path.
            clear_status = uvm_moe_slot_transition_submit_dual_clear_locked(
                gpu,
                &prepared_dual_clear,
                &batch_tracker);
            detail_now = NV_GETTIME();
            telemetry.transition_clear_rebind_ns =
                detail_now - transition_clear_start;
            if (clear_status != NV_OK && result.failure_stage == UVM_SLOT_DIRECT_FAILURE_NONE)
                result.failure_stage = UVM_SLOT_DIRECT_FAILURE_PREPARE;
            if (helper_status == NV_OK)
                helper_status = clear_status;
        }
    }

    if (helper_status == NV_OK) {
        helper_status = uvm_va_block_slot_direct_copy_map_batch(
            va_space,
            mm,
            gpu,
            state->blocks,
            state->block_count,
            shared_push,
            skip_unmap,
            strict_64k_overwrite,
            stage_sync,
            commit_generation != 0 &&
                uvm_moe_slot_plan_overlap_copy_prepare != 0,
            dual_bank_copy_first,
            shadow_pte_prefill,
            fail_closed_transaction_started,
            copy_plan,
            &g_uvm_moe_slot_pool_lock,
            &batch_tracker,
            &result);
    }
    helper_end = NV_GETTIME();
    telemetry.helper_total_ns = helper_end - helper_start;
    detail_cursor = helper_end;

    if (helper_status == NV_OK &&
        (result.copied_bytes != params->requested_bytes ||
         result.mapped_bytes != params->requested_bytes ||
         result.mapped_blocks != state->block_count)) {
        helper_status = NV_ERR_INVALID_STATE;
        result.failure_stage = UVM_SLOT_DIRECT_FAILURE_MAP;
    }

    detail_now = NV_GETTIME();
    telemetry.result_check_ns = detail_now - detail_cursor;
    detail_cursor = detail_now;
    if (helper_status == NV_OK && commit_generation != 0) {
        UVM_ASSERT(g_uvm_moe_slot_plan_state == UVM_MOE_SLOT_PLAN_STATE_TRANSITIONING);
        g_uvm_moe_slot_plan_state = UVM_MOE_SLOT_PLAN_STATE_PUBLISHING;
    }

    if (!dual_bank_copy_first) {
        if (helper_status == NV_OK) {
            metadata_status = uvm_moe_slot_direct_record_all_pte_metadata_locked(params,
                                                                                  va_space,
                                                                                  gpu,
                                                                                  state);
        }
        else {
            // Normally the helper rolls every partial map back. If that
            // rollback itself failed, publish exact surviving ranges before
            // dropping the pool lock so later reset/release can still find
            // and clear them.
            metadata_status =
                uvm_moe_slot_direct_record_surviving_pte_metadata_locked(params,
                                                                          va_space,
                                                                          gpu,
                                                                          state);
        }
        target_metadata_recorded = NV_TRUE;
    }

    // copy_ns/map_ns measure host submission. Normal modes account device
    // completion in this final wait interval. The diagnostic stage-sync mode
    // has already waited at each boundary, so this final wait should be near
    // zero but remains part of the synchronous ioctl contract.
    wait_start = NV_GETTIME();
    telemetry.pte_metadata_ns = wait_start - detail_cursor;
    batch_wait_status = uvm_tracker_wait(&batch_tracker);
    uvm_tracker_deinit(&batch_tracker);
    batch_tracker_deinitialized = NV_TRUE;
    telemetry.tracker_waits = 1;

    // A successful aggregate wait covers every push duplicated into each
    // slot tracker, so avoid another O(slot_count) set of synchronous waits on
    // the measurement path. Error paths still wait each tracker defensively.
    slot_wait_status = uvm_moe_slot_direct_deinit_slot_trackers(
        params,
        state,
        helper_status != NV_OK || batch_wait_status != NV_OK);
    trackers_initialized = NV_FALSE;
    detail_now = NV_GETTIME();
    telemetry.wait_ns = detail_now - wait_start;
    detail_cursor = detail_now;

    if (dual_bank_copy_first &&
        prepared_dual_clear.completion_guarded &&
        batch_wait_status == NV_OK) {
        NvU64 metadata_start = detail_cursor;

        // The aggregate owns the old-clear tracker and has now completed it.
        // Only at this point is it safe to forget the old slot PTE metadata.
        // Record replacement metadata afterwards so an old and a new mapping
        // are never coalesced in the same slot record.
        uvm_moe_slot_transition_commit_completed_dual_clear_locked(
            params,
            state,
            &prepared_dual_clear);

        if (helper_status == NV_OK) {
            metadata_status = uvm_moe_slot_direct_record_all_pte_metadata_locked(params,
                                                                                  va_space,
                                                                                  gpu,
                                                                                  state);
        }
        else {
            metadata_status =
                uvm_moe_slot_direct_record_surviving_pte_metadata_locked(params,
                                                                          va_space,
                                                                          gpu,
                                                                          state);
        }
        target_metadata_recorded = NV_TRUE;
        detail_now = NV_GETTIME();
        telemetry.pte_metadata_ns += detail_now - metadata_start;
        detail_cursor = detail_now;
    }

    status = helper_status;
    if (status == NV_OK)
        status = metadata_status;
    if (status == NV_OK)
        status = batch_wait_status;
    if (status == NV_OK)
        status = slot_wait_status;

    if (status != NV_OK) {
        if (dual_bank_copy_first &&
            prepared_dual_clear.clear_state &&
            !old_clear_cleanup_attempted) {
            old_clear_cleanup_status =
                uvm_moe_slot_transition_fail_closed_dual_clear_locked(
                    params,
                    gpu,
                    state,
                    &prepared_dual_clear);
            old_clear_cleanup_attempted = NV_TRUE;
            if (old_clear_cleanup_status != NV_OK) {
                result.failure_stage = UVM_SLOT_DIRECT_FAILURE_ROLLBACK;
                UVM_ERR_PRINT("MoE dual-bank old-VA cleanup failed: primary %s, cleanup %s\n",
                              nvstatusToString(status),
                              nvstatusToString(old_clear_cleanup_status));
            }
        }

        // If aggregate completion could not confirm the initial old clear,
        // fail-closed above retries and waits it using the retained recipe.
        // Only a successful retry may discard old metadata and make room for
        // surviving replacement metadata. If the retry failed, leave both the
        // slot metadata and registered recipe intact for abort/retry.
        if (dual_bank_copy_first &&
            prepared_dual_clear.metadata_committed &&
            !target_metadata_recorded) {
            metadata_status =
                uvm_moe_slot_direct_record_surviving_pte_metadata_locked(params,
                                                                          va_space,
                                                                          gpu,
                                                                          state);
            target_metadata_recorded = NV_TRUE;
        }

        if (fail_closed_transaction_started) {
            // Metadata recording can fail after all map pushes were submitted,
            // leaving later, unrecorded entries live. Clear recorded slot PTEs
            // with their exact metadata, then sweep the complete flattened
            // target set; neither recipe alone covers every partial-failure path.
            fail_closed_cleanup_status = uvm_moe_slot_direct_fail_closed_cleanup_locked(
                params,
                va_space,
                mm,
                gpu,
                state,
                cleanup_block_context,
                targets_flattened,
                slots_validated);
            fail_closed_cleanup_attempted = NV_TRUE;
            if (fail_closed_cleanup_status != NV_OK) {
                result.failure_stage = UVM_SLOT_DIRECT_FAILURE_ROLLBACK;
                UVM_ERR_PRINT("MoE slot fail-closed cleanup failed: primary %s, cleanup %s\n",
                              nvstatusToString(status),
                              nvstatusToString(fail_closed_cleanup_status));
            }
        }
        else {
            rollback_status = uvm_moe_slot_direct_rollback_recorded_pte_locked(params,
                                                                               va_space,
                                                                               mm,
                                                                               state,
                                                                               NV_FALSE);
            if (rollback_status != NV_OK) {
                result.failure_stage = UVM_SLOT_DIRECT_FAILURE_ROLLBACK;
                status = rollback_status;
            }
        }

        failure_entry_status = uvm_moe_slot_direct_failure_entry_status(
            status,
            &result,
            batch_wait_status != NV_OK || slot_wait_status != NV_OK);
        uvm_moe_slot_remap_fail_ok_entries(params, failure_entry_status);

        for (i = 0; i < params->count; ++i) {
            uvm_moe_slot_pool_slot_t *slot = state->slots[i];

            slot->state = UVM_MOE_SLOT_POOL_STATE_ERROR;
            slot->last_status = failure_entry_status;
            slot->pte_status = failure_entry_status;
            if (result.failure_stage == UVM_SLOT_DIRECT_FAILURE_COPY ||
                batch_wait_status != NV_OK || slot_wait_status != NV_OK) {
                slot->copy_status = UVM_MOE_SLOT_REMAP_ENTRY_COPY_FAILED;
                ++slot->copy_failed_count;
                ++g_uvm_moe_slot_pool_copy_failed_count;
            }
            else {
                slot->copy_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
            }
        }
        goto unlock;
    }

    detail_now = NV_GETTIME();
    telemetry.status_resolve_ns = detail_now - detail_cursor;
    detail_cursor = detail_now;
    if (layer_transition)
        uvm_moe_slot_transition_commit_owner_locked(params, state);

    for (i = 0; i < params->count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[i];
        uvm_moe_slot_pool_slot_t *slot = state->slots[i];
        NvU64 epoch = g_uvm_moe_slot_pool_current_epoch + 1;

        slot->state = UVM_MOE_SLOT_POOL_STATE_READY;
        slot->copy_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
        ++slot->copy_count;
        slot->copy_bytes += entry->bytes;
        ++g_uvm_moe_slot_pool_copy_count;
        g_uvm_moe_slot_pool_copy_bytes += entry->bytes;

        ++slot->range_count;
        slot->last_role = entry->role;
        // NO_EVICT is a backing lifetime policy established by preallocation.
        // A mapping request cannot accidentally drop it by leaving flags at 0.
        // REUSE and mode-20 strategy bits are per-request controls, never slot
        // lifetime state.
        slot->flags = (entry->flags &
                       ~(UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT |
                         UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED |
                         UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_SHARED_PUSH |
                         UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_64K_PTE)) |
                      (slot->flags & UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT);
        slot->last_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
        slot->pte_status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
        slot->epoch = epoch;
        slot->batch_id = batch_id;
        slot->last_va_base = entry->va_base;
        slot->last_bytes = entry->bytes;
        slot->total_bytes += entry->bytes;
        ++slot->pte_remap_count;
        slot->pte_remap_bytes += entry->bytes;
        if (!skip_unmap)
            slot->pte_block_unmap_count += state->range_block_counts[i];

        g_uvm_moe_slot_pool_current_epoch = epoch;
        ++g_uvm_moe_slot_pool_validated_entries;
        ++g_uvm_moe_slot_pool_range_update_count;
        ++g_uvm_moe_slot_pool_pte_remap_count;
        g_uvm_moe_slot_pool_pte_remap_bytes += entry->bytes;

        ++params->success_count;
        params->accepted_bytes += entry->bytes;
    }

    if (!skip_unmap)
        g_uvm_moe_slot_pool_pte_block_unmap_count += state->block_count;
    ++g_uvm_moe_slot_pool_pte_block_tracker_wait_count;
    params->remap_count = params->count;
    params->tlb_invalidate_count =
        pte_mode == UVM_SLOT_DIRECT_PTE_MODE_64K && !skip_unmap ?
            result.pte_64k_blocks :
            state->block_count;
    params->reserved = result.copy_pushes;

    if (commit_generation != 0) {
        UVM_ASSERT(g_uvm_moe_slot_plan_state == UVM_MOE_SLOT_PLAN_STATE_PUBLISHING);
        g_uvm_moe_slot_plan_generation = commit_generation;
        g_uvm_moe_slot_plan_state = UVM_MOE_SLOT_PLAN_STATE_ACTIVE;
        g_uvm_moe_slot_plan_transition_id = 0;
    }

    telemetry.success = true;
    telemetry.completed_ranges = params->count;
    telemetry.completed_bytes = params->accepted_bytes;
    telemetry.pte_map_ranges = params->count;
    telemetry.pte_map_bytes = params->accepted_bytes;

    detail_now = NV_GETTIME();
    telemetry.slot_state_commit_ns = detail_now - detail_cursor;
    detail_cursor = detail_now;
unlock:
    params->reserved = result.copy_pushes;
    telemetry.blocks = result.block_count;
    telemetry.pte_4k_blocks = result.pte_4k_blocks;
    telemetry.pte_64k_blocks = result.pte_64k_blocks;
    telemetry.partial_64k_blocks = result.partial_64k_blocks;
    telemetry.pte_2m_blocks = result.pte_2m_blocks;
    telemetry.pte_write_entries = result.pte_write_entries;
    telemetry.skipped_unmap_blocks = result.skipped_unmap_blocks;
    telemetry.methods = result.copy_methods;
    telemetry.pushes = result.copy_pushes;
    telemetry.overlap_copy_prepare = result.overlap_copy_prepare;
    telemetry.prepare_ns = result.prepare_ns;
    telemetry.invalidate_ns = result.invalidate_ns;
    telemetry.pte_prepare_ns = result.pte_prepare_ns;
    telemetry.unmap_ns = result.unmap_ns;
    telemetry.copy_ns = result.copy_ns;
    telemetry.map_ns = result.map_ns;
    telemetry.prepare_complete_ns = result.prepare_complete_ns;
    telemetry.copy_complete_ns = result.copy_complete_ns;
    telemetry.map_complete_ns = result.map_complete_ns;
    telemetry.shadow_pte_prefill_blocks = result.shadow_pte_prefill_blocks;
    telemetry.shadow_pte_prefill_ns = result.shadow_pte_prefill_ns;
    telemetry.shadow_pte_publish_ns = result.shadow_pte_publish_ns;
    telemetry.helper_context_ns = result.helper_context_ns;
    telemetry.block_lock_acquire_ns = result.block_lock_acquire_ns;
    telemetry.locked_revalidate_ns = result.locked_revalidate_ns;
    telemetry.copy_to_map_ns = result.copy_to_map_ns;
    telemetry.block_unlock_ns = result.block_unlock_ns;
    telemetry.helper_context_free_ns = result.helper_context_free_ns;
    telemetry.copy_run_cache_hit = result.copy_run_cache_hit;
    telemetry.copy_run_cache_runs = result.copy_run_cache_runs;
    telemetry.copy_run_cache_validate_ns += result.copy_run_cache_validate_ns;
    telemetry.copy_method_emit_ns = result.copy_method_emit_ns;
    telemetry.copy_tracker_attach_ns = result.copy_tracker_attach_ns;
    telemetry.backing_allocations = g_uvm_moe_slot_pool_vidmem_alloc_count - allocation_count_before;
    telemetry.backing_evictions = g_uvm_moe_slot_pool_vidmem_evict_count - eviction_count_before;

    if (!batch_tracker_deinitialized &&
        (trackers_initialized || batch_tracker_may_have_work)) {
        wait_start = NV_GETTIME();
        batch_wait_status = uvm_tracker_wait(&batch_tracker);
        uvm_tracker_deinit(&batch_tracker);
        batch_tracker_deinitialized = NV_TRUE;
        batch_tracker_may_have_work = NV_FALSE;
        telemetry.tracker_waits = 1;
        telemetry.wait_ns += NV_GETTIME() - wait_start;
        if (status == NV_OK)
            status = batch_wait_status;
    }

    if (trackers_initialized) {
        slot_wait_status = uvm_moe_slot_direct_deinit_slot_trackers(params, state, NV_TRUE);
        trackers_initialized = NV_FALSE;
        if (status == NV_OK)
            status = slot_wait_status;
    }

    // Preflight, old-PTE teardown, and invalidate/prepare failures branch here
    // before the normal post-helper cleanup block. Once the immutable targets
    // have been flattened, apply the same complete target sweep on every such
    // exit. Tracker draining above always precedes the sweep.
    if (status != NV_OK &&
        dual_bank_copy_first &&
        fail_closed_transaction_started &&
        prepared_dual_clear.clear_state &&
        !old_clear_cleanup_attempted) {
        old_clear_cleanup_status =
            uvm_moe_slot_transition_fail_closed_dual_clear_locked(
                params,
                gpu,
                state,
                &prepared_dual_clear);
        old_clear_cleanup_attempted = NV_TRUE;
        if (old_clear_cleanup_status != NV_OK) {
            result.failure_stage = UVM_SLOT_DIRECT_FAILURE_ROLLBACK;
            UVM_ERR_PRINT("MoE dual-bank old-VA cleanup failed: primary %s, cleanup %s\n",
                          nvstatusToString(status),
                          nvstatusToString(old_clear_cleanup_status));
        }
    }

    if (status != NV_OK &&
        fail_closed_transaction_started &&
        targets_flattened &&
        cleanup_block_context &&
        !fail_closed_cleanup_attempted) {
        fail_closed_cleanup_status = uvm_moe_slot_direct_fail_closed_cleanup_locked(
            params,
            va_space,
            mm,
            gpu,
            state,
            cleanup_block_context,
            targets_flattened,
            slots_validated);
        fail_closed_cleanup_attempted = NV_TRUE;
        if (fail_closed_cleanup_status != NV_OK) {
            result.failure_stage = UVM_SLOT_DIRECT_FAILURE_ROLLBACK;
            UVM_ERR_PRINT("MoE slot fail-closed cleanup failed: primary %s, cleanup %s\n",
                          nvstatusToString(status),
                          nvstatusToString(fail_closed_cleanup_status));
        }
    }

    if (status != NV_OK &&
        fail_closed_transaction_started &&
        slots_validated) {
        // The plan registry is poisoned by the outer execute path, but fault,
        // eviction, and teardown inspect slot state without taking that lock.
        // Never leave a GPU-unmapped failed bank looking READY.
        uvm_moe_slot_transition_mark_old_owners_error_locked(params, state);
    }

    telemetry.tracker_waits += result.stage_sync_waits;

    detail_now = NV_GETTIME();
    telemetry.telemetry_cleanup_ns = detail_now - detail_cursor;
    detail_cursor = detail_now;
    uvm_moe_slot_transition_prepared_clear_deinit(&prepared_dual_clear);
    uvm_va_block_context_free(cleanup_block_context);
    cleanup_block_context = NULL;
    if (pool_locked) {
        mutex_unlock(&g_uvm_moe_slot_pool_lock);
        pool_locked = NV_FALSE;
    }
    if (va_space_locked) {
        uvm_va_space_up_write(va_space);
        va_space_locked = NV_FALSE;
    }
    if (mm_retained) {
        uvm_va_space_mm_or_current_release_unlock(va_space, mm);
        mm_retained = NV_FALSE;
    }

    detail_now = NV_GETTIME();
    telemetry.outer_lock_release_ns = detail_now - detail_cursor;
    detail_cursor = detail_now;
done:
    uvm_moe_slot_transition_prepared_clear_deinit(&prepared_dual_clear);
    uvm_va_block_context_free(cleanup_block_context);
    if (!batch_tracker_deinitialized)
        uvm_tracker_deinit(&batch_tracker);
    if (gpu)
        uvm_gpu_release(gpu);
    uvm_kvfree(state);
    total_end = NV_GETTIME();
    telemetry.resource_teardown_ns = total_end - detail_cursor;
    telemetry.post_helper_total_ns = total_end - helper_end;
    telemetry.total_ns = total_end - total_start;
    if (!telemetry.success)
        telemetry.completed_ranges = telemetry.completed_bytes = telemetry.pte_map_ranges = telemetry.pte_map_bytes = 0;
    telemetry.fallback = false;
    uvm_va_block_note_slot_direct_batch(&telemetry);

    uvm_phase_f_route_log("moe_slot_direct_batch mode=%u shared=%u attempted=%u ranges=%u blocks=%llu bytes=%llu methods=%llu pushes=%llu tracker_waits=%llu allocations=%llu evictions=%llu status=0x%x\n",
                          params->mode,
                          shared_push,
                          attempted,
                          params->count,
                          telemetry.blocks,
                          params->requested_bytes,
                          telemetry.methods,
                          telemetry.pushes,
                          telemetry.tracker_waits,
                          telemetry.backing_allocations,
                          telemetry.backing_evictions,
                          status);

    return status;
}

static NvBool uvm_moe_slot_direct_mode_uses_shared_push(
    const UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params)
{
    return params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH ||
           params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE ||
           (uvm_moe_slot_remap_is_layer_transition_mode(params->mode) &&
            (uvm_moe_slot_transition_batch_strategy(params) &
             UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_SHARED_PUSH));
}

static uvm_slot_direct_pte_mode_t uvm_moe_slot_direct_pte_mode(
    const UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params)
{
    if (params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_PTE ||
        params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_STAGE_SYNC ||
        params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_OVERWRITE ||
        params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_OVERWRITE_STAGE_SYNC ||
        params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE ||
        (uvm_moe_slot_remap_is_layer_transition_mode(params->mode) &&
         (uvm_moe_slot_transition_batch_strategy(params) &
          UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_64K_PTE))) {
        return UVM_SLOT_DIRECT_PTE_MODE_64K;
    }

    if (params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_2M_PTE)
        return UVM_SLOT_DIRECT_PTE_MODE_2M;

    return UVM_SLOT_DIRECT_PTE_MODE_4K;
}

static NV_STATUS uvm_moe_slot_pool_apply_slot_direct_batch(
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params,
    uvm_va_space_t *va_space,
    NV_STATUS validation_status,
    const NvU64 *expected_identities,
    NvU64 expected_invalidation_epoch,
    const uvm_moe_slot_plan_validation_recipe_t *validation_recipe,
    const uvm_va_block_slot_copy_plan_t *copy_plan,
    const uvm_moe_slot_plan_target_recipe_t *target_recipe,
    const UVM_MOE_SLOT_REMAP_BATCH_PARAMS *resident_clear_batch,
    const uvm_moe_slot_plan_target_recipe_t *resident_clear_recipe,
    NvBool metadata_recipe_cache_enabled,
    NvBool epoch_validation_enabled,
    NvBool resident_bank_empty,
    NvBool dual_bank_copy_first,
    NvBool shadow_pte_prefill,
    NvU64 *commit_generation_out)
{
    // Keep owner-transition semantics independent from the copy/PTE strategy.
    // Ordinary modes encode strategy in the mode number; mode 20 uses uniform
    // transient entry flags. The cutover and fail-closed transaction code is
    // shared by every combination.
    return uvm_moe_slot_pool_apply_slot_direct_batch_configured(
        params,
        va_space,
        validation_status,
        uvm_moe_slot_remap_is_layer_transition_mode(params->mode),
        uvm_moe_slot_direct_mode_uses_shared_push(params),
        uvm_moe_slot_direct_pte_mode(params),
        expected_identities,
        expected_invalidation_epoch,
        validation_recipe,
        copy_plan,
        target_recipe,
        resident_clear_batch,
        resident_clear_recipe,
        metadata_recipe_cache_enabled,
        epoch_validation_enabled,
        resident_bank_empty,
        dual_bank_copy_first,
        shadow_pte_prefill,
        commit_generation_out);
}

static void uvm_expert_offload_release_zero(UVM_EXPERT_OFFLOAD_RELEASE_PARAMS *params)
{
    NvU64 region_id = params->region_id;
    NvProcessorUuid gpu_uuid = params->gpu_uuid;
    NvU32 flags = params->flags;

    memset(params, 0, sizeof(*params));
    params->region_id = region_id;
    params->gpu_uuid = gpu_uuid;
    params->flags = flags;
}

static void uvm_expert_offload_release_record_failure(UVM_EXPERT_OFFLOAD_RELEASE_PARAMS *params,
                                                       NvU64 failed_address,
                                                       NvU64 failed_bytes,
                                                       NV_STATUS status)
{
    ++params->failure_blocks;
    params->failed_bytes += failed_bytes;

    if (params->first_error == NV_OK) {
        params->first_error = status;
        params->first_failed_block = failed_address;
    }
}

static NV_STATUS uvm_expert_offload_release_block(uvm_va_space_t *va_space,
                                                  uvm_gpu_t *gpu,
                                                  uvm_va_block_t *block,
                                                  const uvm_phase_f_region_t *region,
                                                  UVM_EXPERT_OFFLOAD_RELEASE_PARAMS *params)
{
    uvm_va_block_gpu_state_t *gpu_state;
    uvm_gpu_chunk_t *chunk;
    NvU64 released_bytes = 0;
    NV_STATUS status;

    if (uvm_va_block_size(block) != UVM_CHUNK_SIZE_MAX) {
        ++params->failure_blocks;
        params->failed_bytes += uvm_va_block_size(block);
        if (params->first_error == NV_OK) {
            params->first_error = NV_ERR_NOT_SUPPORTED;
            params->first_failed_block = block->start;
        }
        return NV_ERR_NOT_SUPPORTED;
    }

    gpu_state = uvm_va_block_gpu_state_get(block, gpu->id);
    if (!gpu_state || !gpu_state->chunks || !gpu_state->chunks[0])
        return NV_WARN_NOTHING_TO_DO;

    chunk = gpu_state->chunks[0];
    if (uvm_gpu_chunk_get_size(chunk) != UVM_CHUNK_SIZE_MAX) {
        ++params->failure_blocks;
        params->failed_bytes += UVM_CHUNK_SIZE_MAX;
        if (params->first_error == NV_OK) {
            params->first_error = NV_ERR_NOT_SUPPORTED;
            params->first_failed_block = block->start;
        }
        return NV_ERR_NOT_SUPPORTED;
    }

    ++params->attempted_blocks;
    params->attempted_bytes += UVM_CHUNK_SIZE_MAX;

    status = uvm_pmm_gpu_evict_root_chunk_to_free(&gpu->pmm, chunk, &released_bytes);
    if (status == NV_OK) {
        ++params->success_blocks;
        params->released_bytes += released_bytes;
    }
    else {
        ++params->failure_blocks;
        params->failed_bytes += UVM_CHUNK_SIZE_MAX;
        if (params->first_error == NV_OK) {
            params->first_error = status;
            params->first_failed_block = block->start;
        }
    }

    return status;
}

static NV_STATUS uvm_expert_offload_release_subchunk_block(uvm_va_space_t *va_space,
                                                           struct mm_struct *mm,
                                                           uvm_gpu_t *gpu,
                                                           uvm_va_block_t *block,
                                                           const uvm_phase_f_region_t *region,
                                                           UVM_EXPERT_OFFLOAD_RELEASE_PARAMS *params)
{
    NvU64 region_end = region->base + region->length - 1;
    NvU64 release_start = max(region->base, block->start);
    NvU64 release_end = min(region_end, block->end);
    uvm_va_block_region_t block_region;
    NvU64 release_bytes;
    NvU64 released_bytes = 0;
    NV_STATUS status;

    if (release_end < release_start)
        return NV_WARN_NOTHING_TO_DO;

    block_region = uvm_va_block_region_from_start_end(block, release_start, release_end);
    release_bytes = uvm_va_block_region_size(block_region);

    if (release_bytes == 0)
        return NV_WARN_NOTHING_TO_DO;

    ++params->attempted_blocks;
    params->attempted_bytes += release_bytes;

    status = uvm_va_block_release_gpu_subchunks_to_cpu(block,
                                                       mm,
                                                       gpu,
                                                       block_region,
                                                       UVM_CHUNK_SIZE_64K,
                                                       (params->flags &
                                                        UVM_EXPERT_OFFLOAD_RELEASE_FLAG_IMMUTABLE_CPU_SHADOW) != 0,
                                                       &block->tracker,
                                                       &released_bytes);
    if (status == NV_OK) {
        ++params->success_blocks;
        params->released_bytes += released_bytes;
    }
    else {
        ++params->failure_blocks;
        params->failed_bytes += release_bytes;
        if (params->first_error == NV_OK) {
            params->first_error = status;
            params->first_failed_block = release_start;
        }
    }

    return status;
}

static NV_STATUS uvm_expert_offload_remote_map_block(uvm_gpu_t *gpu,
                                                     uvm_va_block_t *block,
                                                     uvm_va_block_context_t *block_context,
                                                     uvm_va_block_region_t block_region,
                                                     UVM_EXPERT_OFFLOAD_RELEASE_PARAMS *params)
{
    uvm_va_block_gpu_state_t *gpu_state;
    uvm_page_index_t page_index;
    NvU32 expected_pages = uvm_va_block_region_num_pages(block_region);
    NvU32 cpu_pages = 0;
    NvU32 remote_pages = 0;
    NvU32 newly_mapped_pages = 0;
    NV_STATUS status;

    status = UVM_VA_BLOCK_LOCK_RETRY(block,
                                     NULL,
                                     uvm_va_block_premap_readmostly_remote_read(block,
                                                                                block_context,
                                                                                gpu->id,
                                                                                block_region,
                                                                                true,
                                                                                false,
                                                                                &newly_mapped_pages));
    if (status != NV_OK)
        goto failed;

    uvm_mutex_lock(&block->lock);
    gpu_state = uvm_va_block_gpu_state_get(block, gpu->id);
    for_each_va_block_page_in_region(page_index, block_region) {
        bool cpu_resident = uvm_page_mask_test(&block->cpu.resident, page_index);
        bool gpu_resident = gpu_state &&
                            uvm_page_mask_test(&gpu_state->resident, page_index);
        bool gpu_read_mapped = gpu_state &&
                               uvm_page_mask_test(&gpu_state->pte_bits[UVM_PTE_BITS_GPU_READ],
                                                  page_index);

        if (cpu_resident)
            ++cpu_pages;
        if (cpu_resident && !gpu_resident && gpu_read_mapped)
            ++remote_pages;
    }
    uvm_mutex_unlock(&block->lock);

    params->cpu_resident_bytes += (NvU64)cpu_pages * PAGE_SIZE;
    params->remote_mapped_bytes += (NvU64)remote_pages * PAGE_SIZE;
    params->newly_remote_mapped_bytes += (NvU64)newly_mapped_pages * PAGE_SIZE;

    if (cpu_pages != expected_pages || remote_pages != expected_pages) {
        status = NV_ERR_INVALID_STATE;
        goto failed;
    }

    ++params->remote_map_blocks;
    return NV_OK;

failed:
    ++params->remote_map_failure_blocks;
    if (params->first_error == NV_OK) {
        params->first_error = status;
        params->first_failed_block = block->start +
                                     (NvU64)block_region.first * PAGE_SIZE;
    }
    return status;
}

static NV_STATUS uvm_api_expert_offload_release(UVM_EXPERT_OFFLOAD_RELEASE_PARAMS *params,
                                                struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    uvm_phase_f_region_t *region = NULL;
    uvm_va_block_context_t *block_context = NULL;
    NvU64 region_end;
    NvU64 addr;
    bool placement_complete = false;
    bool remote_map_only = false;
    NV_STATUS status = NV_OK;

    if (params->flags & ~(UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REQUIRE_EXPERT |
                          UVM_EXPERT_OFFLOAD_RELEASE_FLAG_SUBCHUNK_64K |
                          UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REMOTE_READ |
                          UVM_EXPERT_OFFLOAD_RELEASE_FLAG_IMMUTABLE_CPU_SHADOW)) {
        return NV_ERR_INVALID_ARGUMENT;
    }

    status = uvm_moe_slot_pool_retain_gpu_by_uuid(&params->gpu_uuid, &gpu);
    if (status != NV_OK) {
        uvm_expert_offload_release_zero(params);
        params->rmStatus = status;
        return status;
    }

    uvm_expert_offload_release_zero(params);

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_write(va_space);
    if (params->flags & UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REMOTE_READ)
        block_context = uvm_va_space_block_context(va_space, mm);

    region = radix_tree_lookup(&va_space->phase_f_regions, params->region_id);
    if (!region) {
        status = NV_ERR_OBJECT_NOT_FOUND;
        goto done_unlock;
    }

    params->base = region->base;
    params->length = region->length;

    if (region->state == UVM_REGION_STATE_CPU) {
        if (!(params->flags & UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REMOTE_READ)) {
            placement_complete = true;
            goto done_unlock;
        }

        /* CPU placement does not prove that GPU remote-read PTEs exist. Avoid
         * re-running the release primitive, which rejects already-installed
         * remote PTEs, and instead establish and validate those PTEs below. */
        remote_map_only = true;
    }
    /* UNKNOWN is not an idempotent success, but it is safe to converge by
     * executing the explicitly requested release. */
    region->state = UVM_REGION_STATE_PREFETCHING_CPU;

    if (region->length == 0) {
        status = NV_ERR_INVALID_ARGUMENT;
        goto done_unlock;
    }

    region_end = region->base + region->length - 1;
    if (region_end < region->base) {
        status = NV_ERR_INVALID_ARGUMENT;
        goto done_unlock;
    }

    if (params->flags & UVM_EXPERT_OFFLOAD_RELEASE_FLAG_SUBCHUNK_64K) {
        if (!IS_ALIGNED(region->base, UVM_CHUNK_SIZE_64K) ||
            !IS_ALIGNED(region->length, UVM_CHUNK_SIZE_64K)) {
            status = NV_ERR_INVALID_ARGUMENT;
            goto done_unlock;
        }
    }

    addr = region->base;
    while (addr <= region_end) {
        uvm_va_range_t *va_range = uvm_va_range_find(va_space, addr);
        uvm_va_range_managed_t *managed_range;
        size_t first_block_index;
        size_t last_block_index;
        size_t block_index;
        NvU64 managed_start;
        NvU64 managed_end;
        NvU64 covered_end;

        if (!va_range) {
            uvm_expert_offload_release_record_failure(params,
                                                       addr,
                                                       region_end - addr + 1,
                                                       NV_ERR_INVALID_ADDRESS);
            break;
        }

        covered_end = min(region_end, va_range->node.end);
        managed_range = uvm_va_range_to_managed_or_null(va_range);
        if (!managed_range) {
            uvm_expert_offload_release_record_failure(params,
                                                       addr,
                                                       covered_end - addr + 1,
                                                       NV_ERR_INVALID_ADDRESS);
            goto next_range;
        }

        managed_start = max(region->base, managed_range->va_range.node.start);
        managed_end = min(region_end, managed_range->va_range.node.end);
        first_block_index = uvm_va_range_block_index(managed_range, managed_start);
        last_block_index = uvm_va_range_block_index(managed_range, managed_end);

        for (block_index = first_block_index; block_index <= last_block_index; ++block_index) {
            uvm_va_block_t *block;
            uvm_va_block_region_t block_region;
            NV_STATUS block_status;

            block_status = uvm_va_range_block_create(managed_range, block_index, &block);
            if (block_status != NV_OK) {
                ++params->failure_blocks;
                if (params->first_error == NV_OK) {
                    params->first_error = block_status;
                    params->first_failed_block = managed_start;
                }
                break;
            }

            ++params->block_count;

            block_region = uvm_va_block_region_from_start_end(block,
                                                               max(region->base, block->start),
                                                               min(region_end, block->end));

            if (remote_map_only) {
                block_status = uvm_expert_offload_remote_map_block(gpu,
                                                                   block,
                                                                   block_context,
                                                                   block_region,
                                                                   params);
            }
            else if (params->flags & UVM_EXPERT_OFFLOAD_RELEASE_FLAG_SUBCHUNK_64K) {
                block_status = uvm_expert_offload_release_subchunk_block(va_space,
                                                                         mm,
                                                                         gpu,
                                                                         block,
                                                                         region,
                                                                         params);
            }
            else {
                block_status = uvm_expert_offload_release_block(va_space,
                                                                gpu,
                                                                block,
                                                                region,
                                                                params);
            }

            if (!remote_map_only &&
                (params->flags & UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REMOTE_READ) &&
                (block_status == NV_OK || block_status == NV_WARN_NOTHING_TO_DO)) {
                block_status = uvm_expert_offload_remote_map_block(gpu,
                                                                   block,
                                                                   block_context,
                                                                   block_region,
                                                                   params);
            }

            if (block_status == NV_WARN_NOTHING_TO_DO) {
                ++params->skipped_blocks;
            }
        }

next_range:
        if (covered_end == region_end) {
            placement_complete = true;
            break;
        }

        addr = covered_end + 1;
    }

done_unlock:
    if (status == NV_OK && !placement_complete) {
        status = params->first_error != NV_OK ?
                 params->first_error :
                 NV_ERR_INVALID_ADDRESS;
    }
    if (status == NV_OK && params->first_error != NV_OK)
        status = params->first_error;
    if (region) {
        region->state = status == NV_OK && placement_complete ?
                        UVM_REGION_STATE_CPU :
                        UVM_REGION_STATE_UNKNOWN;
    }

    uvm_va_space_up_write(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);
    uvm_gpu_release(gpu);

    params->rmStatus = status;

    uvm_phase_f_route_log("region_release region_id=%llu base=0x%llx length=%llu blocks=%llu attempted=%u success=%u failure=%u skipped=%u attempted_bytes=%llu released_bytes=%llu failed_bytes=%llu remote_blocks=%u remote_failures=%u cpu_bytes=%llu remote_bytes=%llu newly_remote_bytes=%llu first_failed=0x%llx status=0x%x\n",
                          params->region_id,
                          params->base,
                          params->length,
                          params->block_count,
                          params->attempted_blocks,
                          params->success_blocks,
                          params->failure_blocks,
                          params->skipped_blocks,
                          params->attempted_bytes,
                          params->released_bytes,
                          params->failed_bytes,
                          params->remote_map_blocks,
                          params->remote_map_failure_blocks,
                          params->cpu_resident_bytes,
                          params->remote_mapped_bytes,
                          params->newly_remote_mapped_bytes,
                          params->first_failed_block,
                          params->rmStatus);

    return status;
}

static uvm_moe_slot_plan_t *uvm_moe_slot_plan_find_locked(uvm_va_space_t *va_space,
                                                           NvU64 plan_id)
{
    NvU32 i;

    for (i = 0; i < UVM_MOE_SLOT_PLAN_MAX_PLANS; ++i) {
        uvm_moe_slot_plan_t *plan = &g_uvm_moe_slot_plans[i];

        if (plan->valid && plan->owner_va_space == va_space && plan->plan_id == plan_id)
            return plan;
    }

    return NULL;
}

static NvBool uvm_moe_slot_plan_is_abort_target_locked(NvU64 plan_id)
{
    NvU32 i;

    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_plan_lock));

    for (i = 0; i < ARRAY_SIZE(g_uvm_moe_slot_plan_bank_resident_plan_ids); ++i) {
        if (g_uvm_moe_slot_plan_bank_resident_plan_ids[i] == plan_id)
            return NV_TRUE;
    }

    return g_uvm_moe_slot_plan_failed_target_plan_id == plan_id;
}

static uvm_moe_slot_plan_t *uvm_moe_slot_plan_free_entry_locked(void)
{
    NvU32 i;

    for (i = 0; i < UVM_MOE_SLOT_PLAN_MAX_PLANS; ++i) {
        if (!g_uvm_moe_slot_plans[i].valid)
            return &g_uvm_moe_slot_plans[i];
    }

    return NULL;
}

static NV_STATUS uvm_moe_slot_plan_next_generation_locked(NvU64 *generation_out)
{
    if (g_uvm_moe_slot_plan_generation_counter == ~0ULL)
        return NV_ERR_INSUFFICIENT_RESOURCES;

    ++g_uvm_moe_slot_plan_generation_counter;
    *generation_out = g_uvm_moe_slot_plan_generation_counter;
    return NV_OK;
}

static NvBool uvm_moe_slot_plan_has_bootstrap_locked(void)
{
    NvU32 i;

    for (i = 0; i < UVM_MOE_SLOT_PLAN_MAX_PLANS; ++i) {
        if (g_uvm_moe_slot_plans[i].valid &&
            (g_uvm_moe_slot_plans[i].batch->mode ==
                 UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH ||
             g_uvm_moe_slot_plans[i].batch->mode ==
                 UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE)) {
            return NV_TRUE;
        }
    }

    return NV_FALSE;
}

static NvBool uvm_moe_slot_plan_layer_registered_locked(NvU32 layer_id)
{
    NvU32 i;

    for (i = 0; i < UVM_MOE_SLOT_PLAN_MAX_PLANS; ++i) {
        const uvm_moe_slot_plan_t *plan = &g_uvm_moe_slot_plans[i];

        if (plan->valid &&
            plan->batch->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION &&
            plan->layer_id == layer_id) {
            return NV_TRUE;
        }
    }

    return NV_FALSE;
}

static const uvm_moe_slot_plan_bank_lane_t *uvm_moe_slot_plan_bank_lane_find_locked(
    NvU32 bank_id,
    NvU32 driver_layer_id,
    NvU32 slot_id)
{
    const uvm_moe_slot_plan_bank_lane_t *lanes =
        bank_id == 0 ? g_uvm_moe_slot_plan_bank_lanes :
        bank_id == 1 ? g_uvm_moe_slot_plan_second_bank_lanes : NULL;
    NvU32 i;

    if (!lanes)
        return NULL;

    for (i = 0; i < g_uvm_moe_slot_plan_bank_count; ++i) {
        const uvm_moe_slot_plan_bank_lane_t *lane = &lanes[i];

        if (lane->driver_layer_id == driver_layer_id && lane->slot_id == slot_id)
            return lane;
    }

    return NULL;
}

static NvBool uvm_moe_slot_plan_candidate_matches_bank_locked(
    const uvm_moe_slot_plan_bank_lane_t *candidate_lanes,
    NvU32 count,
    NvU32 bank_id)
{
    NvU32 i;

    for (i = 0; i < count; ++i) {
        const uvm_moe_slot_plan_bank_lane_t *candidate = &candidate_lanes[i];
        const uvm_moe_slot_plan_bank_lane_t *canonical =
            uvm_moe_slot_plan_bank_lane_find_locked(bank_id,
                                                     candidate->driver_layer_id,
                                                     candidate->slot_id);

        if (!canonical ||
            canonical->pool_index != candidate->pool_index ||
            canonical->bytes != candidate->bytes ||
            canonical->slot_offset != candidate->slot_offset ||
            canonical->slot_bytes != candidate->slot_bytes ||
            canonical->slot_mem != candidate->slot_mem) {
            return NV_FALSE;
        }
    }

    return NV_TRUE;
}

static NvBool uvm_moe_slot_plan_candidate_disjoint_from_first_bank_locked(
    const uvm_moe_slot_plan_bank_lane_t *candidate_lanes,
    NvU32 count)
{
    NvU32 i;
    NvU32 j;

    for (i = 0; i < count; ++i) {
        const uvm_moe_slot_plan_bank_lane_t *candidate = &candidate_lanes[i];
        NvU64 candidate_end = candidate->slot_offset + candidate->bytes;

        if (candidate_end < candidate->slot_offset)
            return NV_FALSE;

        for (j = 0; j < g_uvm_moe_slot_plan_bank_count; ++j) {
            const uvm_moe_slot_plan_bank_lane_t *first = &g_uvm_moe_slot_plan_bank_lanes[j];
            NvU64 first_end = first->slot_offset + first->bytes;

            if (first_end < first->slot_offset)
                return NV_FALSE;
            // A bank is a set of independently owned slot records, not merely
            // a disjoint byte range. One slot has a single owner/state/PTE
            // metadata/tracker tuple, so two subranges of the same slot_mem
            // cannot safely represent simultaneously active banks.
            if (candidate->slot_mem == first->slot_mem) {
                return NV_FALSE;
            }
        }
    }

    return NV_TRUE;
}

static NV_STATUS uvm_moe_slot_plan_validate_bank_locked(
    const UVM_MOE_SLOT_REMAP_BATCH_PARAMS *batch,
    uvm_va_space_t *va_space,
    uvm_gpu_t *gpu,
    NvBool new_session,
    uvm_moe_slot_plan_bank_lane_t *candidate_lanes,
    NvU32 *bank_id_out,
    NvBool *install_second_bank_out)
{
    NvU32 i;

    UVM_ASSERT(bank_id_out);
    UVM_ASSERT(install_second_bank_out);
    *bank_id_out = NV_U32_MAX;
    *install_second_bank_out = NV_FALSE;

    for (i = 0; i < batch->count; ++i) {
        const UVM_MOE_SLOT_REMAP_ENTRY *entry = &batch->entries[i];
        uvm_moe_slot_pool_slot_t *slot = uvm_moe_slot_pool_find_locked(va_space,
                                                                       entry->layer_id,
                                                                       entry->slot_id);
        uvm_moe_slot_plan_bank_lane_t *candidate = &candidate_lanes[i];

        if (!slot ||
            !(slot->flags & UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT) ||
            !uvm_moe_slot_pool_slot_has_vidmem_backing(slot, gpu) ||
            entry->slot_offset + entry->bytes < entry->slot_offset ||
            entry->slot_offset + entry->bytes > slot->slot_bytes ||
            slot->state != UVM_MOE_SLOT_POOL_STATE_READY ||
            slot->async_copy_pending ||
            slot->async_copy_staging) {
            return NV_ERR_INVALID_STATE;
        }

        // The bootstrap descriptor is registered against the mode-8 owner
        // table which created the fixed bank. Transition plans intentionally
        // name future owners and therefore do not require an owner match.
        if (new_session && slot->owner_expert_id != entry->expert_id)
            return NV_ERR_INVALID_STATE;
        if (new_session && !uvm_moe_slot_direct_pte_metadata_empty(slot))
            return NV_ERR_INVALID_STATE;

        candidate->driver_layer_id = entry->layer_id;
        candidate->slot_id = entry->slot_id;
        candidate->pool_index = (NvU32)(slot - g_uvm_moe_slot_pool);
        candidate->bytes = entry->bytes;
        candidate->slot_offset = entry->slot_offset;
        candidate->slot_bytes = slot->slot_bytes;
        candidate->slot_mem = slot->slot_mem;

    }

    if (new_session) {
        *bank_id_out = 0;
        return NV_OK;
    }

    if (uvm_moe_slot_plan_candidate_matches_bank_locked(candidate_lanes,
                                                         batch->count,
                                                         0)) {
        *bank_id_out = 0;
        return NV_OK;
    }

    if (READ_ONCE(uvm_moe_slot_plan_dual_bank_event) == 0)
        return NV_ERR_INVALID_STATE;

    if (g_uvm_moe_slot_plan_second_bank_lanes) {
        if (!uvm_moe_slot_plan_candidate_matches_bank_locked(candidate_lanes,
                                                              batch->count,
                                                              1)) {
            return NV_ERR_INVALID_STATE;
        }
        *bank_id_out = 1;
        return NV_OK;
    }

    if (!uvm_moe_slot_plan_candidate_disjoint_from_first_bank_locked(candidate_lanes,
                                                                      batch->count)) {
        return NV_ERR_INVALID_STATE;
    }

    *bank_id_out = 1;
    *install_second_bank_out = NV_TRUE;

    return NV_OK;
}

static NV_STATUS uvm_moe_slot_plan_validation_recipe_create_locked(
    const UVM_MOE_SLOT_REMAP_BATCH_PARAMS *batch,
    uvm_va_space_t *va_space,
    uvm_gpu_t *gpu,
    const uvm_moe_slot_plan_bank_lane_t *candidate_lanes,
    uvm_moe_slot_plan_validation_recipe_t **recipe_out)
{
    uvm_moe_slot_plan_validation_recipe_t *recipe;
    NvU32 i;

    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_pool_lock));
    UVM_ASSERT(recipe_out);
    *recipe_out = NULL;

    if (!batch || !candidate_lanes || batch->count == 0)
        return NV_ERR_INVALID_ARGUMENT;

    recipe = uvm_kvmalloc_zero(sizeof(*recipe));
    if (!recipe)
        return NV_ERR_NO_MEMORY;

    recipe->entries = uvm_kvmalloc_zero((size_t)batch->count * sizeof(*recipe->entries));
    recipe->slots = uvm_kvmalloc_zero((size_t)batch->count * sizeof(*recipe->slots));
    recipe->identity_keys = uvm_kvmalloc_zero((size_t)batch->count * sizeof(*recipe->identity_keys));
    if (!recipe->entries || !recipe->slots || !recipe->identity_keys) {
        uvm_moe_slot_plan_validation_recipe_destroy(recipe);
        return NV_ERR_NO_MEMORY;
    }
    recipe->count = batch->count;
    recipe->owner_va_space = va_space;
    recipe->gpu = gpu;
    recipe->invalidation_epoch = uvm_va_space_moe_slot_plan_invalidation_epoch(va_space);
    recipe->source_epoch = uvm_va_space_moe_slot_copy_source_epoch(va_space);
    recipe->backing_epoch = g_uvm_moe_slot_pool_backing_epoch;

    for (i = 0; i < batch->count; ++i) {
        const UVM_MOE_SLOT_REMAP_ENTRY *entry = &batch->entries[i];
        const uvm_moe_slot_plan_bank_lane_t *lane = &candidate_lanes[i];
        uvm_moe_slot_plan_validation_entry_t *cached = &recipe->entries[i];
        uvm_moe_slot_pool_slot_t *slot;

        if (lane->pool_index >= UVM_MOE_SLOT_POOL_MAX_SLOTS) {
            uvm_moe_slot_plan_validation_recipe_destroy(recipe);
            return NV_ERR_INVALID_STATE;
        }

        slot = &g_uvm_moe_slot_pool[lane->pool_index];
        if (!uvm_moe_slot_pool_entry_matches(slot,
                                              va_space,
                                              entry->layer_id,
                                              entry->slot_id) ||
            lane->driver_layer_id != entry->layer_id ||
            lane->slot_id != entry->slot_id ||
            lane->slot_mem != slot->slot_mem ||
            lane->slot_bytes != slot->slot_bytes ||
            slot->slot_gpu != gpu) {
            uvm_moe_slot_plan_validation_recipe_destroy(recipe);
            return NV_ERR_INVALID_STATE;
        }

        cached->pool_index = lane->pool_index;
        cached->layer_id = entry->layer_id;
        cached->slot_id = entry->slot_id;
        cached->owner_va_space = va_space;
        cached->slot_mem = slot->slot_mem;
        cached->slot_gpu = slot->slot_gpu;
        cached->slot_bytes = slot->slot_bytes;
        cached->backing_generation = slot->backing_generation;
        recipe->slots[i] = slot;

        recipe->identity_keys[i].layer_id = entry->layer_id;
        recipe->identity_keys[i].slot_id = entry->slot_id;
        recipe->identity_keys[i].entry_index = i;
    }

    sort(recipe->identity_keys,
         recipe->count,
         sizeof(recipe->identity_keys[0]),
         uvm_moe_slot_identity_key_cmp,
         NULL);

    for (i = 1; i < recipe->count; ++i) {
        const uvm_moe_slot_identity_key_t *prev = &recipe->identity_keys[i - 1];
        const uvm_moe_slot_identity_key_t *current_key = &recipe->identity_keys[i];

        if (prev->layer_id == current_key->layer_id && prev->slot_id == current_key->slot_id) {
            uvm_moe_slot_plan_validation_recipe_destroy(recipe);
            return NV_ERR_INVALID_STATE;
        }
    }

    *recipe_out = recipe;
    return NV_OK;
}

static void uvm_moe_slot_plan_refresh_registration_source_epoch_locked(
    uvm_va_space_t *va_space)
{
    NvU64 source_epoch;
    NvU32 i;

    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_plan_lock));
    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_pool_lock));
    uvm_assert_rwsem_locked_write(&va_space->lock);

    // Building a later registered copy plan may establish CE-visible DMA
    // mappings for additional frozen source blocks. That advances the
    // conservative VA-space aggregate source epoch without invalidating runs
    // already cached for earlier plans. Registration is still unsealed and
    // serialized here, so publish one post-build epoch to every recipe. Once
    // execution seals registration this helper is no longer reachable; any
    // subsequent source mutation therefore remains a fail-closed mismatch.
    source_epoch = uvm_va_space_moe_slot_copy_source_epoch(va_space);
    for (i = 0; i < ARRAY_SIZE(g_uvm_moe_slot_plans); ++i) {
        uvm_moe_slot_plan_t *registered = &g_uvm_moe_slot_plans[i];

        if (!registered->valid || registered->owner_va_space != va_space ||
            !registered->validation_recipe) {
            continue;
        }

        registered->validation_recipe->source_epoch = source_epoch;
    }
}

static NV_STATUS uvm_api_moe_slot_plan_register(UVM_MOE_SLOT_PLAN_REGISTER_PARAMS *params,
                                                 struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *batch = NULL;
    uvm_moe_slot_plan_t *plan = NULL;
    uvm_moe_slot_plan_bank_lane_t *candidate_lanes = NULL;
    uvm_moe_slot_direct_batch_state_t *prewarm_state = NULL;
    uvm_va_block_slot_plan_pte_prewarm_result_t prewarm_result = { 0 };
    uvm_va_block_slot_copy_plan_t *copy_plan = NULL;
    uvm_moe_slot_plan_target_recipe_t *target_recipe = NULL;
    uvm_moe_slot_plan_validation_recipe_t *validation_recipe = NULL;
    NvU64 *allocation_ids = NULL;
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    NV_STATUS status = NV_OK;
    NvU64 requested_bytes = 0;
    NvU64 invalidation_epoch = 0;
    NvU64 prewarm_start = 0;
    NvU64 prewarm_ns = 0;
    NvU64 copy_cache_start = 0;
    NvU64 copy_cache_ns = 0;
    NvBool new_session = NV_FALSE;
    NvBool prewarm_enabled = NV_FALSE;
    NvBool copy_cache_enabled = NV_FALSE;
    NvBool metadata_recipe_cache_enabled = NV_FALSE;
    NvBool plan_locked = NV_FALSE;
    NvBool va_space_locked = NV_FALSE;
    NvBool pool_locked = NV_FALSE;
    NvBool mm_retained = NV_FALSE;
    NvBool install_second_bank = NV_FALSE;
    NvU32 plan_bank_id = NV_U32_MAX;
    NvU32 i;
    NvU32 j;

    uvm_va_block_note_slot_plan_register();

    params->generation = 0;
    params->requested_bytes = 0;

    if (params->plan_id == 0 ||
        params->reserved != 0 ||
        params->count == 0 ||
        params->count > UVM_MOE_SLOT_REMAP_MAX_ENTRIES ||
        (params->mode != UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION &&
         params->mode != UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH &&
         params->mode != UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE)) {
        return NV_ERR_INVALID_ARGUMENT;
    }

    batch = uvm_kvmalloc_zero(sizeof(*batch));
    if (!batch)
        return NV_ERR_NO_MEMORY;

    allocation_ids = uvm_kvmalloc_zero(params->count * sizeof(*allocation_ids));
    candidate_lanes = uvm_kvmalloc_zero(params->count * sizeof(*candidate_lanes));
    if (!allocation_ids || !candidate_lanes) {
        status = NV_ERR_NO_MEMORY;
        goto done;
    }

    batch->mode = params->mode;
    batch->count = params->count;
    batch->gpu_uuid = params->gpu_uuid;
    batch->validate_managed = 1;

    for (i = 0; i < params->count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &batch->entries[i];

        *entry = params->entries[i];
        entry->status = UVM_MOE_SLOT_REMAP_ENTRY_OK;

        if (uvm_moe_slot_remap_entry_range_invalid(entry) ||
            entry->bytes > UVM_MOE_SLOT_DIRECT_MAX_ENTRY_BYTES ||
            entry->slot_offset + entry->bytes < entry->slot_offset) {
            status = NV_ERR_INVALID_ARGUMENT;
            goto done;
        }

        if (batch->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION) {
            if (!uvm_moe_slot_transition_entry_strategy_valid(batch, i)) {
                status = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }
        }
        else if (entry->flags & ~UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT) {
            status = NV_ERR_INVALID_ARGUMENT;
            goto done;
        }

        for (j = 0; j < i; ++j) {
            const UVM_MOE_SLOT_REMAP_ENTRY *previous = &batch->entries[j];

            if ((previous->layer_id == entry->layer_id &&
                 previous->slot_id == entry->slot_id) ||
                uvm_moe_slot_direct_source_ranges_overlap(previous, entry)) {
                status = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }
        }

        if (requested_bytes + entry->bytes < requested_bytes) {
            status = NV_ERR_INVALID_ARGUMENT;
            goto done;
        }
        requested_bytes += entry->bytes;
    }
    batch->requested_bytes = requested_bytes;

    // Only complete sliced-64K plans have a registration-safe hierarchy-only
    // preparation contract. Other legal registered modes retain their exact
    // historical registration and execute behavior, including allocation
    // failure semantics.
    prewarm_enabled = READ_ONCE(uvm_moe_slot_plan_register_pte_prewarm) != 0 &&
                      uvm_moe_slot_direct_pte_mode(batch) == UVM_SLOT_DIRECT_PTE_MODE_64K;
    copy_cache_enabled = READ_ONCE(uvm_moe_slot_plan_cache_copy_runs) != 0;
    metadata_recipe_cache_enabled =
        READ_ONCE(uvm_moe_slot_plan_cache_metadata_recipe) != 0;

    // Allocate temporary flattening state before taking any registry/VA-space
    // locks. It is never retained by the plan; the durable hierarchy belongs
    // to the ordinary VA-block GPU state and follows that lifetime.
    if (prewarm_enabled || copy_cache_enabled || metadata_recipe_cache_enabled) {
        prewarm_state = uvm_kvmalloc_zero(sizeof(*prewarm_state));
        if (!prewarm_state) {
            status = NV_ERR_NO_MEMORY;
            goto done;
        }
    }

    // Retain the object before taking the plan lock, then revalidate its
    // registration under the VA-space write lock below. The plan itself does
    // not retain this pointer.
    gpu = uvm_va_space_retain_gpu_by_uuid(va_space, &params->gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto done;
    }

    mutex_lock(&g_uvm_moe_slot_plan_lock);
    plan_locked = NV_TRUE;
    if (uvm_moe_static_alias_active_for_va_space_locked(va_space)) {
        status = NV_ERR_IN_USE;
        goto done;
    }
    new_session = g_uvm_moe_slot_plan_owner_va_space == NULL;

    if (g_uvm_moe_slot_plan_owner_va_space &&
        g_uvm_moe_slot_plan_owner_va_space != va_space) {
        status = NV_ERR_IN_USE;
        goto done;
    }
    if (uvm_moe_slot_plan_find_locked(va_space, params->plan_id)) {
        status = NV_ERR_IN_USE;
        goto done;
    }
    if (g_uvm_moe_slot_plan_poisoned) {
        status = NV_ERR_INVALID_STATE;
        goto done;
    }
    if (!new_session && g_uvm_moe_slot_plan_state != UVM_MOE_SLOT_PLAN_STATE_ACTIVE) {
        status = NV_ERR_INVALID_STATE;
        goto done;
    }
    if (!new_session && g_uvm_moe_slot_plan_registration_sealed) {
        status = NV_ERR_INVALID_STATE;
        goto done;
    }

    if (params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH ||
        params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE) {
        // Mode 11 (4K) or mode 21 (64K) is the one bootstrap copy/map
        // descriptor for a bank already created by mode 8. A session cannot
        // replace or add another bootstrap.
        if (!new_session ||
            g_uvm_moe_slot_plan_bootstrap_registered ||
            uvm_moe_slot_plan_has_bootstrap_locked()) {
            status = NV_ERR_IN_USE;
            goto done;
        }
    }
    else {
        // Every later descriptor is an owner-changing mode-20 plan. Logical
        // layers are unique among transitions; bootstrap L0 and transition L0
        // are intentionally allowed to coexist for cyclic serving.
        if (new_session ||
            !g_uvm_moe_slot_plan_bootstrap_registered ||
            uvm_moe_slot_plan_layer_registered_locked(params->layer_id)) {
            status = NV_ERR_INVALID_STATE;
            goto done;
        }

        if (!uvm_uuid_eq(&g_uvm_moe_slot_plan_gpu_uuid, &params->gpu_uuid) ||
            g_uvm_moe_slot_plan_bank_count != params->count) {
            status = NV_ERR_INVALID_STATE;
            goto done;
        }
    }

    plan = uvm_moe_slot_plan_free_entry_locked();
    if (!plan) {
        status = NV_ERR_INSUFFICIENT_RESOURCES;
        goto done;
    }

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    mm_retained = NV_TRUE;
    uvm_va_space_down_write(va_space);
    va_space_locked = NV_TRUE;
    mutex_lock(&g_uvm_moe_slot_pool_lock);
    pool_locked = NV_TRUE;

    if (uvm_va_space_get_gpu_by_uuid_with_gpu_va_space(va_space, &params->gpu_uuid) != gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto done;
    }

    invalidation_epoch = uvm_va_space_moe_slot_plan_invalidation_epoch(va_space);
    if (!new_session && invalidation_epoch != g_uvm_moe_slot_plan_invalidation_epoch) {
        g_uvm_moe_slot_plan_poisoned = NV_TRUE;
        status = NV_ERR_INVALID_STATE;
        goto done;
    }

    status = uvm_moe_slot_plan_capture_identities_locked(batch, va_space, allocation_ids);
    if (status != NV_OK)
        goto done;

    status = uvm_moe_slot_plan_validate_bank_locked(batch,
                                                    va_space,
                                                    gpu,
                                                    new_session,
                                                    candidate_lanes,
                                                    &plan_bank_id,
                                                    &install_second_bank);
    if (status != NV_OK)
        goto done;

    status = uvm_moe_slot_plan_validation_recipe_create_locked(batch,
                                                                va_space,
                                                                gpu,
                                                                candidate_lanes,
                                                                &validation_recipe);
    if (status != NV_OK)
        goto done;

    if (prewarm_enabled || copy_cache_enabled || metadata_recipe_cache_enabled) {
        for (i = 0; i < batch->count; ++i) {
            prewarm_state->slots[i] = uvm_moe_slot_pool_find_locked(va_space,
                                                                    batch->entries[i].layer_id,
                                                                    batch->entries[i].slot_id);
            if (!prewarm_state->slots[i]) {
                status = NV_ERR_INVALID_STATE;
                break;
            }
        }

        if (status == NV_OK) {
            status = uvm_moe_slot_direct_flatten_blocks_locked(batch,
                                                                va_space,
                                                                prewarm_state,
                                                                NV_TRUE);
        }

        if (status == NV_OK) {
            for (i = 0; i < prewarm_state->block_count; ++i)
                prewarm_state->blocks[i].pte_mode = uvm_moe_slot_direct_pte_mode(batch);
        }
        if (status != NV_OK)
            goto done;
    }

    if (metadata_recipe_cache_enabled) {
        status = uvm_moe_slot_plan_target_recipe_create_locked(batch,
                                                                prewarm_state,
                                                                invalidation_epoch,
                                                                &target_recipe);
        if (status != NV_OK)
            goto done;
    }

    if (prewarm_enabled) {
        prewarm_start = NV_GETTIME();

        status = uvm_va_block_slot_plan_pte_prewarm(va_space,
                                                    mm,
                                                    gpu,
                                                    prewarm_state->blocks,
                                                    prewarm_state->block_count,
                                                    &prewarm_result);

        prewarm_ns = NV_GETTIME() - prewarm_start;
        uvm_va_block_note_slot_plan_pte_prewarm(status,
                                                batch->count,
                                                batch->requested_bytes,
                                                prewarm_ns,
                                                &prewarm_result);
        if (status != NV_OK)
            goto done;
    }

    if (copy_cache_enabled) {
        copy_cache_start = NV_GETTIME();
        status = uvm_va_block_slot_copy_plan_create(va_space,
                                                     mm,
                                                     gpu,
                                                     prewarm_state->blocks,
                                                     prewarm_state->block_count,
                                                     &g_uvm_moe_slot_pool_lock,
                                                     &copy_plan);
        copy_cache_ns = NV_GETTIME() - copy_cache_start;
        if (status == NV_OK && (!copy_plan || copy_plan->bytes != batch->requested_bytes))
            status = NV_ERR_INVALID_STATE;
        uvm_va_block_note_slot_plan_copy_cache(status,
                                                batch->count,
                                                batch->requested_bytes,
                                                copy_cache_ns,
                                                copy_plan ? copy_plan->entry_count : 0,
                                                copy_plan ? copy_plan->run_count : 0);
        if (status != NV_OK)
            goto done;
    }

    if (new_session) {
        NvU64 initial_generation;

        status = uvm_moe_slot_plan_next_generation_locked(&initial_generation);
        if (status != NV_OK)
            goto done;

        g_uvm_moe_slot_plan_bank_lanes = candidate_lanes;
        candidate_lanes = NULL;
        g_uvm_moe_slot_plan_bank_count = params->count;
        g_uvm_moe_slot_plan_registered_bank_count = 1;
        g_uvm_moe_slot_plan_active_bank = NV_U32_MAX;
        uvm_uuid_copy(&g_uvm_moe_slot_plan_gpu_uuid, &params->gpu_uuid);
        g_uvm_moe_slot_plan_owner_va_space = va_space;
        g_uvm_moe_slot_plan_generation = initial_generation;
        g_uvm_moe_slot_plan_invalidation_epoch = invalidation_epoch;
        g_uvm_moe_slot_plan_poisoned = NV_FALSE;
        g_uvm_moe_slot_plan_bootstrap_registered = NV_TRUE;
        g_uvm_moe_slot_plan_state = UVM_MOE_SLOT_PLAN_STATE_ACTIVE;
    }
    else if (install_second_bank) {
        UVM_ASSERT(plan_bank_id == 1);
        UVM_ASSERT(!g_uvm_moe_slot_plan_second_bank_lanes);
        g_uvm_moe_slot_plan_second_bank_lanes = candidate_lanes;
        candidate_lanes = NULL;
        g_uvm_moe_slot_plan_registered_bank_count = 2;
    }

    plan->valid = NV_TRUE;
    plan->plan_id = params->plan_id;
    plan->layer_id = params->layer_id;
    plan->owner_va_space = va_space;
    plan->batch = batch;
    plan->allocation_ids = allocation_ids;
    plan->copy_plan = copy_plan;
    plan->target_recipe = target_recipe;
    plan->validation_recipe = validation_recipe;
    plan->invalidation_epoch = invalidation_epoch;
    plan->bank_id = plan_bank_id;
    plan->pte_prewarmed = prewarm_enabled;
    plan->shadow_pte_eligible = prewarm_enabled && prewarm_result.shadow_eligible;
    plan->partial_64k = prewarm_enabled && prewarm_result.partial_block_count != 0;
    uvm_moe_slot_plan_refresh_registration_source_epoch_locked(va_space);
    batch = NULL;
    allocation_ids = NULL;
    copy_plan = NULL;
    target_recipe = NULL;
    validation_recipe = NULL;
    params->generation = g_uvm_moe_slot_plan_generation;
    params->requested_bytes = requested_bytes;

done:
    if (pool_locked)
        mutex_unlock(&g_uvm_moe_slot_pool_lock);
    if (va_space_locked)
        uvm_va_space_up_write(va_space);
    if (mm_retained)
        uvm_va_space_mm_or_current_release_unlock(va_space, mm);
    if (plan_locked)
        mutex_unlock(&g_uvm_moe_slot_plan_lock);
    if (gpu)
        uvm_gpu_release(gpu);
    uvm_kvfree(candidate_lanes);
    uvm_kvfree(prewarm_state);
    uvm_va_block_slot_copy_plan_destroy(copy_plan);
    uvm_moe_slot_plan_target_recipe_destroy(target_recipe);
    uvm_moe_slot_plan_validation_recipe_destroy(validation_recipe);
    uvm_kvfree(allocation_ids);
    uvm_kvfree(batch);
    return status;
}

static void uvm_moe_slot_plan_execute_copy_result(
    UVM_MOE_SLOT_PLAN_EXECUTE_PARAMS *params,
    const UVM_MOE_SLOT_REMAP_BATCH_PARAMS *batch)
{
    params->mode = batch->mode;
    params->count = batch->count;
    params->success_count = batch->success_count;
    params->failure_count = batch->failure_count;
    params->conflict_count = batch->conflict_count;
    params->remap_count = batch->remap_count;
    params->tlb_invalidate_count = batch->tlb_invalidate_count;
    params->copy_push_count = batch->reserved;
    params->requested_bytes = batch->requested_bytes;
    params->accepted_bytes = batch->accepted_bytes;
    params->owner_cutover =
        batch->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION &&
        batch->count != 0 && batch->remap_count == batch->count;
}

static NV_STATUS uvm_api_moe_slot_plan_execute_common(
    UVM_MOE_SLOT_PLAN_EXECUTE_PARAMS *params,
    struct file *filp,
    uvm_moe_slot_plan_fence_request_t *fence)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *batch = NULL;
    uvm_moe_slot_plan_t *plan;
    uvm_moe_slot_plan_t *resident_plan = NULL;
    const uvm_va_block_slot_copy_plan_t *execute_copy_plan = NULL;
    const uvm_moe_slot_plan_target_recipe_t *execute_target_recipe = NULL;
    const uvm_moe_slot_plan_validation_recipe_t *execute_validation_recipe = NULL;
    const UVM_MOE_SLOT_REMAP_BATCH_PARAMS *resident_clear_batch = NULL;
    const uvm_moe_slot_plan_target_recipe_t *resident_clear_recipe = NULL;
    NV_STATUS status = NV_OK;
    NvU64 commit_generation = 0;
    NvBool dual_bank_copy_first = NV_FALSE;
    NvBool dual_bank_event_enabled;
    NvBool metadata_recipe_cache_enabled;
    NvBool fast_validation_enabled;
    NvBool epoch_validation_enabled;
    NvBool shadow_pte_prefill = NV_FALSE;
    NvBool shadow_pte_prefill_requested;
    NvBool resident_bank_empty = NV_FALSE;
    NvBool cuda_fence_required;

    uvm_va_block_note_slot_plan_execute();

    params->new_generation = 0;
    params->requested_bytes = 0;
    params->accepted_bytes = 0;
    params->mode = 0;
    params->count = 0;
    params->success_count = 0;
    params->failure_count = 0;
    params->conflict_count = 0;
    params->remap_count = 0;
    params->tlb_invalidate_count = 0;
    params->copy_push_count = 0;
    params->owner_cutover = 0;
    params->poisoned = 0;
    if (params->plan_id == 0)
        return NV_ERR_INVALID_ARGUMENT;

    batch = uvm_kvmalloc_zero(sizeof(*batch));
    if (!batch)
        return NV_ERR_NO_MEMORY;

    // Serialize the complete owner transaction. A stale generation or wrong
    // target is rejected while only this registry lock is held, before the
    // configured core can acquire a VA-space lock or touch any PTE.
    mutex_lock(&g_uvm_moe_slot_plan_lock);
    params->new_generation = g_uvm_moe_slot_plan_generation;
    params->poisoned = g_uvm_moe_slot_plan_poisoned;

    plan = uvm_moe_slot_plan_find_locked(va_space, params->plan_id);
    if (!plan) {
        status = NV_ERR_OBJECT_NOT_FOUND;
        goto unlock;
    }
    if (g_uvm_moe_slot_plan_owner_va_space != va_space ||
        g_uvm_moe_slot_plan_poisoned ||
        g_uvm_moe_slot_plan_state != UVM_MOE_SLOT_PLAN_STATE_ACTIVE ||
        params->generation != g_uvm_moe_slot_plan_generation ||
        params->layer_id != plan->layer_id) {
        status = NV_ERR_INVALID_STATE;
        goto unlock;
    }

    // Module parameters are writable for A/B experiments. Snapshot both
    // switches once under the plan lock so a concurrent sysfs write cannot
    // pass the dual-bank gate with a cached plan and then hand NULL to the
    // copy-first helper.
    dual_bank_event_enabled = READ_ONCE(uvm_moe_slot_plan_dual_bank_event) != 0;
    metadata_recipe_cache_enabled =
        READ_ONCE(uvm_moe_slot_plan_cache_metadata_recipe) != 0;
    fast_validation_enabled = READ_ONCE(uvm_moe_slot_plan_fast_validation) != 0;
    epoch_validation_enabled = READ_ONCE(uvm_moe_slot_plan_epoch_validation) != 0;
    shadow_pte_prefill_requested =
        READ_ONCE(uvm_moe_slot_plan_shadow_pte_prefill) != 0;
    cuda_fence_required = READ_ONCE(uvm_moe_slot_plan_cuda_fence) != 0;
    if (cuda_fence_required != (fence != NULL)) {
        status = NV_ERR_INVALID_STATE;
        goto unlock;
    }
    if (READ_ONCE(uvm_moe_slot_plan_cache_copy_runs) != 0)
        execute_copy_plan = plan->copy_plan;
    if (fast_validation_enabled) {
        execute_validation_recipe = plan->validation_recipe;
        if (!execute_validation_recipe) {
            status = NV_ERR_INVALID_STATE;
            goto unlock;
        }
    }
    if (metadata_recipe_cache_enabled) {
        execute_target_recipe = plan->target_recipe;
        if (!execute_target_recipe) {
            status = NV_ERR_INVALID_STATE;
            goto unlock;
        }
    }
    if (epoch_validation_enabled &&
        (!fast_validation_enabled || !execute_validation_recipe ||
         !execute_copy_plan || !execute_target_recipe)) {
        status = NV_ERR_INVALID_STATE;
        goto unlock;
    }

    if (plan->batch->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH ||
        plan->batch->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE) {
        // Mode 11/21 initializes bank 0 exactly once. Replaying it after any
        // plan became active would bypass transition teardown and could
        // overwrite a bank which is still GPU-visible. This gate is
        // unconditional so a runtime module-parameter change cannot reopen
        // the bootstrap path.
        if (g_uvm_moe_slot_plan_active_bank != NV_U32_MAX || plan->bank_id != 0) {
            status = NV_ERR_INVALID_STATE;
            goto unlock;
        }
    }
    else if (dual_bank_event_enabled &&
             plan->batch->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION) {
        // Userspace has synchronized a CUDA event recorded on the only
        // compute stream before entering this ioctl. Require an actually
        // inactive second bank and an immutable copy plan before allowing the
        // copy to pass the old-PTE completion boundary. A same-bank request is
        // rejected before generation reservation or PTE mutation.
        if (g_uvm_moe_slot_plan_registered_bank_count != 2 ||
            g_uvm_moe_slot_plan_active_bank == NV_U32_MAX ||
            plan->bank_id > 1 ||
            plan->bank_id == g_uvm_moe_slot_plan_active_bank ||
            !execute_copy_plan) {
            status = NV_ERR_INVALID_STATE;
            goto unlock;
        }
        dual_bank_copy_first = NV_TRUE;

        if (g_uvm_moe_slot_plan_bank_resident_plan_ids[plan->bank_id] == 0) {
            resident_bank_empty = NV_TRUE;
        }
        else if (metadata_recipe_cache_enabled) {
            resident_plan = uvm_moe_slot_plan_find_locked(
                va_space,
                g_uvm_moe_slot_plan_bank_resident_plan_ids[plan->bank_id]);
            if (!resident_plan || resident_plan->bank_id != plan->bank_id ||
                !resident_plan->target_recipe) {
                status = NV_ERR_INVALID_STATE;
                goto unlock;
            }
            resident_clear_batch = resident_plan->batch;
            resident_clear_recipe = resident_plan->target_recipe;
        }

        if (shadow_pte_prefill_requested) {
            if (!plan->pte_prewarmed ||
                uvm_moe_slot_direct_pte_mode(plan->batch) != UVM_SLOT_DIRECT_PTE_MODE_64K ||
                !execute_copy_plan) {
                status = NV_ERR_INVALID_STATE;
                goto unlock;
            }

            // Select the shadow path when at least one touched 2 MiB block is
            // complete. The helper chooses per block: complete groups prefill
            // unreachable leaves and publish their parent after copy, while
            // sparse groups retain the live child table and install only the
            // cold-mask leaves through a copy-dependent map push.
            if (plan->shadow_pte_eligible)
                shadow_pte_prefill = NV_TRUE;
            else if (!plan->partial_64k) {
                status = NV_ERR_INVALID_STATE;
                goto unlock;
            }
        }
    }

    // A requested shadow transition must never silently fall back to a live-
    // leaf map path. Bootstrap is intentionally excluded: it has no active old
    // bank and executes before dual-bank transitions begin.
    if (shadow_pte_prefill_requested &&
        plan->batch->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION &&
        !shadow_pte_prefill &&
        !plan->partial_64k) {
        status = NV_ERR_INVALID_STATE;
        goto unlock;
    }

    if (fence) {
        if (fence->payload == 0 ||
            fence->compute_addr == 0 ||
            fence->completion_addr == 0 ||
            fence->compute_addr == fence->completion_addr ||
            (g_uvm_moe_slot_plan_fence_compute_addr != 0 &&
             (fence->compute_addr != g_uvm_moe_slot_plan_fence_compute_addr ||
              fence->completion_addr != g_uvm_moe_slot_plan_fence_completion_addr)) ||
            (g_uvm_moe_slot_plan_fence_last_payload != 0 &&
             fence->payload <= g_uvm_moe_slot_plan_fence_last_payload)) {
            status = NV_ERR_INVALID_ARGUMENT;
            goto unlock;
        }

        // This GPU-side acquire replaces the old device-wide userspace
        // synchronize. It waits only for work ordered before the stream write
        // and completes before copy/clear submission, preserving the explicit
        // no compute-copy-overlap contract.
        if (!fence->compute_waited) {
            status = uvm_user_semaphore_wait_and_validate(
                va_space,
                &plan->batch->gpu_uuid,
                fence->compute_addr,
                fence->completion_addr,
                fence->payload);
            if (status != NV_OK)
                goto unlock;
        }

        if (g_uvm_moe_slot_plan_fence_compute_addr == 0) {
            g_uvm_moe_slot_plan_fence_compute_addr = fence->compute_addr;
            g_uvm_moe_slot_plan_fence_completion_addr = fence->completion_addr;
        }
        g_uvm_moe_slot_plan_fence_last_payload = fence->payload;
        fence->accepted = NV_TRUE;
    }

    // The descriptor set is immutable once any plan is executed. Together
    // with module-lifetime generations this prevents destroy/re-register ABA
    // for a userspace-selected plan_id.
    g_uvm_moe_slot_plan_registration_sealed = NV_TRUE;

    memcpy(batch, plan->batch, sizeof(*batch));
    status = uvm_moe_slot_pool_apply_slot_direct_batch(batch,
                                                       va_space,
                                                       NV_OK,
                                                       plan->allocation_ids,
                                                       plan->invalidation_epoch,
                                                       execute_validation_recipe,
                                                       execute_copy_plan,
                                                       execute_target_recipe,
                                                       resident_clear_batch,
                                                       resident_clear_recipe,
                                                       metadata_recipe_cache_enabled,
                                                       epoch_validation_enabled,
                                                       resident_bank_empty,
                                                       dual_bank_copy_first,
                                                       shadow_pte_prefill,
                                                       &commit_generation);
    uvm_moe_slot_plan_execute_copy_result(params, batch);

    if (status == NV_OK &&
        commit_generation != 0 &&
        batch->success_count == batch->count &&
        batch->failure_count == 0 &&
        batch->accepted_bytes == batch->requested_bytes &&
        g_uvm_moe_slot_plan_generation == commit_generation &&
        g_uvm_moe_slot_plan_state == UVM_MOE_SLOT_PLAN_STATE_ACTIVE) {
        // Generation and owner state were published together while the core
        // still held the VA-space and slot-pool locks.
        g_uvm_moe_slot_plan_active_bank = plan->bank_id;
        g_uvm_moe_slot_plan_bank_resident_plan_ids[plan->bank_id] = plan->plan_id;
        g_uvm_moe_slot_plan_failed_target_plan_id = 0;
    }
    else if (commit_generation != 0) {
        // The core reserved a generation only after every preflight and cleanup
        // resource succeeded. From that point, any error is an execution-stage
        // failure which has run the complete GPU-unmap sweep and is abort-only.
        g_uvm_moe_slot_plan_poisoned = NV_TRUE;
        g_uvm_moe_slot_plan_state = UVM_MOE_SLOT_PLAN_STATE_POISONED;
        g_uvm_moe_slot_plan_failed_target_plan_id = plan->plan_id;
        if (status == NV_OK)
            status = NV_ERR_INVALID_STATE;
    }
    else {
        // Identity, device, slot, target, hierarchy, or cleanup-resource
        // rejection happened before TRANSITIONING and touched no PTE. Preserve
        // the active generation/session so the caller can correct or retry it.
        UVM_ASSERT(g_uvm_moe_slot_plan_state == UVM_MOE_SLOT_PLAN_STATE_ACTIVE);
        g_uvm_moe_slot_plan_failed_target_plan_id = 0;
        if (status == NV_OK)
            status = NV_ERR_INVALID_STATE;
    }

    params->new_generation = g_uvm_moe_slot_plan_generation;
    params->poisoned = g_uvm_moe_slot_plan_poisoned;

    if (status == NV_OK && fence) {
        NV_STATUS fence_status;

        fence_status = uvm_user_semaphore_release_from_cpu(
            va_space,
            fence->completion_addr,
            fence->payload);

        if (fence_status == NV_OK) {
            fence->signaled = NV_TRUE;
        }
        else {
            // Publication already committed, but no consumer is authorized:
            // leave completion unsignaled and force the generation-free abort
            // path before userspace can enqueue any new-weight kernel.
            g_uvm_moe_slot_plan_poisoned = NV_TRUE;
            g_uvm_moe_slot_plan_state = UVM_MOE_SLOT_PLAN_STATE_POISONED;
            params->poisoned = 1;
            status = fence_status;
        }
    }

unlock:
    mutex_unlock(&g_uvm_moe_slot_plan_lock);
    uvm_kvfree(batch);
    return status;
}

static NV_STATUS uvm_api_moe_slot_plan_execute(UVM_MOE_SLOT_PLAN_EXECUTE_PARAMS *params,
                                                struct file *filp)
{
    return uvm_api_moe_slot_plan_execute_common(params,
                                                filp,
                                                NULL);
}

static void uvm_moe_slot_plan_fenced_copy_common_out(
    UVM_MOE_SLOT_PLAN_EXECUTE_FENCED_PARAMS *params,
    const UVM_MOE_SLOT_PLAN_EXECUTE_PARAMS *common)
{
    params->new_generation = common->new_generation;
    params->requested_bytes = common->requested_bytes;
    params->accepted_bytes = common->accepted_bytes;
    params->mode = common->mode;
    params->count = common->count;
    params->success_count = common->success_count;
    params->failure_count = common->failure_count;
    params->conflict_count = common->conflict_count;
    params->remap_count = common->remap_count;
    params->tlb_invalidate_count = common->tlb_invalidate_count;
    params->copy_push_count = common->copy_push_count;
    params->owner_cutover = common->owner_cutover;
    params->poisoned = common->poisoned;
}

static NV_STATUS uvm_api_moe_slot_plan_execute_fenced(
    UVM_MOE_SLOT_PLAN_EXECUTE_FENCED_PARAMS *params,
    struct file *filp)
{
    UVM_MOE_SLOT_PLAN_EXECUTE_PARAMS common = { 0 };
    uvm_moe_slot_plan_fence_request_t fence = { 0 };
    NV_STATUS status;

    BUILD_BUG_ON(sizeof(*params) != 128);
    BUILD_BUG_ON(offsetof(UVM_MOE_SLOT_PLAN_EXECUTE_FENCED_PARAMS, fence_flags) != 84);
    BUILD_BUG_ON(offsetof(UVM_MOE_SLOT_PLAN_EXECUTE_FENCED_PARAMS, compute_semaphore_address) != 88);
    BUILD_BUG_ON(offsetof(UVM_MOE_SLOT_PLAN_EXECUTE_FENCED_PARAMS, rmStatus) != 124);

    params->fence_accepted = 0;
    params->completion_signaled = 0;
    if (params->fence_flags != UVM_MOE_SLOT_PLAN_FENCE_FLAGS_ALL ||
        params->reserved != 0 ||
        params->compute_payload == 0 ||
        params->compute_payload != params->completion_payload) {
        return NV_ERR_INVALID_ARGUMENT;
    }

    common.plan_id = params->plan_id;
    common.generation = params->generation;
    common.layer_id = params->layer_id;
    fence.compute_addr = params->compute_semaphore_address;
    fence.completion_addr = params->completion_semaphore_address;
    fence.payload = params->compute_payload;

    status = uvm_api_moe_slot_plan_execute_common(&common,
                                                   filp,
                                                   &fence);
    uvm_moe_slot_plan_fenced_copy_common_out(params, &common);
    params->fence_accepted = fence.accepted;
    params->completion_signaled = fence.signaled;
    return status;
}

static NV_STATUS uvm_api_moe_slot_plan_destroy(UVM_MOE_SLOT_PLAN_DESTROY_PARAMS *params,
                                                struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_moe_slot_plan_t *plan;
    NV_STATUS status = NV_OK;
    NvU32 i;
    NvBool any = NV_FALSE;

    uvm_va_block_note_slot_plan_destroy();

    params->current_generation = 0;
    params->destroyed = 0;
    if (params->plan_id == 0 || params->reserved != 0)
        return NV_ERR_INVALID_ARGUMENT;

    mutex_lock(&g_uvm_moe_slot_plan_lock);
    params->current_generation = g_uvm_moe_slot_plan_generation;
    plan = uvm_moe_slot_plan_find_locked(va_space, params->plan_id);
    if (!plan) {
        status = NV_ERR_OBJECT_NOT_FOUND;
        goto unlock;
    }
    if (params->generation != g_uvm_moe_slot_plan_generation ||
        g_uvm_moe_slot_plan_poisoned ||
        g_uvm_moe_slot_plan_state != UVM_MOE_SLOT_PLAN_STATE_ACTIVE) {
        status = NV_ERR_INVALID_STATE;
        goto unlock;
    }

    // A resident bank or a failed publication can still have GPU-visible PTEs
    // described only by this immutable plan. Keep that recipe until the
    // generation-free session abort has swept every target VA. Deleting it
    // here would make a later abort unable to prove that teardown is complete.
    if (uvm_moe_slot_plan_is_abort_target_locked(params->plan_id)) {
        status = NV_ERR_IN_USE;
        goto unlock;
    }

    // A partially destroyed registry cannot accept a replacement descriptor
    // with the same plan_id and generation.
    g_uvm_moe_slot_plan_registration_sealed = NV_TRUE;

    uvm_moe_slot_plan_free_locked(plan);
    params->destroyed = 1;

    for (i = 0; i < UVM_MOE_SLOT_PLAN_MAX_PLANS; ++i) {
        if (g_uvm_moe_slot_plans[i].valid) {
            any = NV_TRUE;
            break;
        }
    }
    if (!any) {
        uvm_moe_slot_plan_clear_session_locked(NV_FALSE);
        params->current_generation = 0;
    }

unlock:
    mutex_unlock(&g_uvm_moe_slot_plan_lock);
    return status;
}

static NV_STATUS uvm_moe_slot_abort_materialize_plan_targets_locked(
    const uvm_moe_slot_plan_t *plan,
    uvm_va_space_t *va_space,
    uvm_moe_slot_direct_batch_state_t *state)
{
    const UVM_MOE_SLOT_REMAP_BATCH_PARAMS *batch;
    const uvm_moe_slot_plan_target_recipe_t *recipe;
    NvU64 materialized_bytes = 0;
    NvU32 output_index = 0;
    NvU32 i;

    uvm_assert_rwsem_locked_write(&va_space->lock);
    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_plan_lock));

    if (!plan || !plan->valid || plan->owner_va_space != va_space ||
        !plan->batch || !state) {
        return NV_ERR_INVALID_STATE;
    }

    memset(state, 0, sizeof(*state));
    batch = plan->batch;
    recipe = plan->target_recipe;
    if (batch->count == 0 || batch->count > UVM_MOE_SLOT_REMAP_MAX_ENTRIES ||
        batch->requested_bytes == 0 ||
        plan->invalidation_epoch !=
            uvm_va_space_moe_slot_plan_invalidation_epoch(va_space)) {
        return NV_ERR_INVALID_STATE;
    }

    if (recipe) {
        NvU32 group_index;

        // An invalidation epoch means these immutable addresses may have been
        // freed and reused. Refuse teardown instead of unmapping a potentially
        // unrelated allocation and releasing backing on an unproven VA set.
        if (!recipe->slices || !recipe->groups ||
            recipe->invalidation_epoch != plan->invalidation_epoch ||
            recipe->entry_count != batch->count ||
            recipe->slice_count == 0 ||
            recipe->slice_count > UVM_MOE_SLOT_DIRECT_MAX_BLOCKS ||
            recipe->group_count == 0 ||
            recipe->group_count > recipe->slice_count ||
            recipe->bytes != batch->requested_bytes) {
            return NV_ERR_INVALID_STATE;
        }

        for (group_index = 0; group_index < recipe->group_count; ++group_index) {
            const uvm_moe_slot_plan_target_group_t *group = &recipe->groups[group_index];
            uvm_page_mask_t materialized_mask;
            uvm_va_block_t *va_block = NULL;
            NvU32 slice_offset;
            NV_STATUS status;

            if (group->slice_count == 0 || group->first_slice != output_index ||
                group->first_slice > recipe->slice_count ||
                group->slice_count > recipe->slice_count - group->first_slice ||
                group->block_end < group->block_start) {
                return NV_ERR_INVALID_STATE;
            }

            status = uvm_va_block_find_create_managed(va_space,
                                                       group->block_start,
                                                       &va_block);
            if (status != NV_OK || !va_block || uvm_va_block_is_dead(va_block) ||
                uvm_va_block_is_hmm(va_block) ||
                uvm_va_block_get_va_space(va_block) != va_space ||
                va_block->start != group->block_start ||
                va_block->end != group->block_end) {
                return status == NV_OK ? NV_ERR_INVALID_STATE : status;
            }

            uvm_page_mask_zero(&materialized_mask);
            for (slice_offset = 0; slice_offset < group->slice_count; ++slice_offset) {
                const uvm_moe_slot_plan_target_slice_t *slice =
                    &recipe->slices[group->first_slice + slice_offset];
                const UVM_MOE_SLOT_REMAP_ENTRY *entry;
                uvm_va_block_slot_direct_entry_t *block_entry;
                NvU64 slice_end = slice->va_base + slice->bytes;
                NvU64 entry_end;

                if (slice->entry_index >= batch->count || slice->bytes == 0 ||
                    slice_end <= slice->va_base ||
                    slice->va_base < va_block->start ||
                    slice_end > va_block->end + 1 ||
                    output_index >= UVM_MOE_SLOT_DIRECT_MAX_BLOCKS) {
                    return NV_ERR_INVALID_STATE;
                }

                entry = &batch->entries[slice->entry_index];
                entry_end = entry->va_base + entry->bytes;
                if (entry->bytes == 0 || entry_end <= entry->va_base ||
                    slice->va_base < entry->va_base || slice_end > entry_end ||
                    materialized_bytes + slice->bytes < materialized_bytes) {
                    return NV_ERR_INVALID_STATE;
                }

                block_entry = &state->blocks[output_index];
                block_entry->va_block = va_block;
                block_entry->region =
                    uvm_va_block_region_from_start_end(va_block,
                                                       slice->va_base,
                                                       slice_end - 1);
                state->block_entry_indices[output_index] = slice->entry_index;
                ++output_index;
                materialized_bytes += slice->bytes;
                uvm_page_mask_region_fill(&materialized_mask, block_entry->region);
            }

            if (!uvm_page_mask_equal(&materialized_mask, &group->page_mask))
                return NV_ERR_INVALID_STATE;
        }

        if (output_index != recipe->slice_count ||
            materialized_bytes != recipe->bytes) {
            return NV_ERR_INVALID_STATE;
        }
    }
    else {
        // Older/diagnostic descriptors may lack a cached topology. Rebuild
        // only their address slices from the kernel-owned batch; no user
        // pointer or slot metadata is needed for this abort-only sweep.
        for (i = 0; i < batch->count; ++i) {
            const UVM_MOE_SLOT_REMAP_ENTRY *entry = &batch->entries[i];
            NvU64 entry_end = entry->va_base + entry->bytes;
            NvU64 cursor = entry->va_base;

            if (entry->bytes == 0 || entry_end <= entry->va_base ||
                materialized_bytes + entry->bytes < materialized_bytes) {
                return NV_ERR_INVALID_STATE;
            }

            while (cursor < entry_end) {
                uvm_va_block_slot_direct_entry_t *block_entry;
                uvm_va_block_t *va_block = NULL;
                NvU64 slice_last;
                NV_STATUS status;

                if (output_index >= UVM_MOE_SLOT_DIRECT_MAX_BLOCKS)
                    return NV_ERR_INSUFFICIENT_RESOURCES;

                status = uvm_va_block_find_create_managed(va_space, cursor, &va_block);
                if (status != NV_OK || !va_block || uvm_va_block_is_dead(va_block) ||
                    uvm_va_block_is_hmm(va_block) ||
                    uvm_va_block_get_va_space(va_block) != va_space) {
                    return status == NV_OK ? NV_ERR_INVALID_STATE : status;
                }

                slice_last = min(entry_end - 1, va_block->end);
                block_entry = &state->blocks[output_index];
                block_entry->va_block = va_block;
                block_entry->region =
                    uvm_va_block_region_from_start_end(va_block, cursor, slice_last);
                state->block_entry_indices[output_index] = i;
                ++output_index;
                cursor = slice_last + 1;
            }

            materialized_bytes += entry->bytes;
        }

        if (materialized_bytes != batch->requested_bytes)
            return NV_ERR_INVALID_STATE;
    }

    state->block_count = output_index;
    uvm_moe_slot_direct_sort_blocks(state, NV_TRUE);
    for (i = 1; i < state->block_count; ++i) {
        const uvm_va_block_slot_direct_entry_t *previous = &state->blocks[i - 1];
        const uvm_va_block_slot_direct_entry_t *current_entry = &state->blocks[i];

        if (previous->va_block == current_entry->va_block &&
            previous->region.outer > current_entry->region.first) {
            return NV_ERR_INVALID_STATE;
        }
    }

    return state->block_count == 0 ? NV_ERR_INVALID_STATE : NV_OK;
}

static NV_STATUS uvm_moe_slot_abort_sweep_plan_targets_locked(
    uvm_va_space_t *va_space,
    uvm_gpu_t *gpu,
    uvm_va_block_context_t *block_context,
    uvm_moe_slot_direct_batch_state_t *state)
{
    const NvU64 target_plan_ids[] = {
        g_uvm_moe_slot_plan_bank_resident_plan_ids[0],
        g_uvm_moe_slot_plan_bank_resident_plan_ids[1],
        g_uvm_moe_slot_plan_failed_target_plan_id,
    };
    NV_STATUS first_error = NV_OK;
    NvU32 i;

    uvm_assert_rwsem_locked_write(&va_space->lock);
    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_plan_lock));
    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_pool_lock));
    UVM_ASSERT(gpu);
    UVM_ASSERT(block_context);
    UVM_ASSERT(state);

    for (i = 0; i < ARRAY_SIZE(target_plan_ids); ++i) {
        const uvm_moe_slot_plan_t *plan;
        NV_STATUS status;
        NvU32 previous;

        if (target_plan_ids[i] == 0)
            continue;

        for (previous = 0; previous < i; ++previous) {
            if (target_plan_ids[previous] == target_plan_ids[i])
                break;
        }
        if (previous != i)
            continue;

        plan = uvm_moe_slot_plan_find_locked(va_space, target_plan_ids[i]);
        if (!plan) {
            if (first_error == NV_OK)
                first_error = NV_ERR_INVALID_STATE;
            continue;
        }

        status = uvm_moe_slot_abort_materialize_plan_targets_locked(plan,
                                                                    va_space,
                                                                    state);
        if (status == NV_OK) {
            status = uvm_va_block_slot_direct_fail_closed_unmap_batch(
                va_space,
                block_context,
                gpu,
                state->blocks,
                state->block_count);
        }

        // Sweep every selected resident/failed target even after an error. No
        // backing is released unless the complete set succeeds, but
        // best-effort unmap reduces the GPU-visible surface before a later
        // abort retry.
        if (first_error == NV_OK)
            first_error = status;
    }

    return first_error;
}

static NV_STATUS uvm_api_moe_slot_session_abort(UVM_MOE_SLOT_SESSION_ABORT_PARAMS *params,
                                                 struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_moe_slot_direct_batch_state_t *abort_state = NULL;
    uvm_va_block_context_t *abort_block_context = NULL;
    uvm_gpu_t *abort_gpu = NULL;
    struct mm_struct *mm;
    NV_STATUS first_error = NV_OK;
    NV_STATUS target_sweep_status = NV_OK;
    NvProcessorUuid abort_gpu_uuid = { 0 };
    NvU64 target_plan_ids[3] = { 0 };
    NvBool has_plan_targets = NV_FALSE;
    NvBool resources_remain = NV_FALSE;
    NvU32 i;

    BUILD_BUG_ON(sizeof(*params) != 64);
    BUILD_BUG_ON(offsetof(UVM_MOE_SLOT_SESSION_ABORT_PARAMS, generation_before) != 8);
    BUILD_BUG_ON(offsetof(UVM_MOE_SLOT_SESSION_ABORT_PARAMS, rmStatus) != 56);

    params->generation_before = 0;
    params->released_bytes = 0;
    params->remaining_bytes = 0;
    params->destroyed_plan_count = 0;
    params->released_slot_count = 0;
    params->remaining_slot_count = 0;
    params->had_registered_session = 0;
    params->session_cleared = 0;
    params->poisoned = 0;

    if (params->flags != 0 || params->reserved != 0)
        return NV_ERR_INVALID_ARGUMENT;

    // Serialize with register, execute, destroy, and the legacy slot ioctl.
    // Keep this lock through PTE/backing release so no new owner transaction
    // can observe a half-aborted bank.
    mutex_lock(&g_uvm_moe_slot_plan_lock);

    if (uvm_moe_static_alias_active_for_va_space_locked(va_space)) {
        mutex_unlock(&g_uvm_moe_slot_plan_lock);
        return NV_ERR_IN_USE;
    }

    if (g_uvm_moe_slot_plan_owner_va_space == va_space) {
        params->generation_before = g_uvm_moe_slot_plan_generation;
        params->had_registered_session = 1;
        // No execute/register retry is safe once teardown starts. A fully
        // successful abort clears this bit together with the session below.
        g_uvm_moe_slot_plan_poisoned = NV_TRUE;
        g_uvm_moe_slot_plan_state = UVM_MOE_SLOT_PLAN_STATE_POISONED;
    }

    for (i = 0; i < UVM_MOE_SLOT_PLAN_MAX_PLANS; ++i) {
        const uvm_moe_slot_plan_t *plan = &g_uvm_moe_slot_plans[i];

        if (!plan->valid || plan->owner_va_space != va_space)
            continue;

        params->had_registered_session = 1;
    }

    // Only plans which may have installed target PTEs participate in the
    // generation-free abort sweep. A registered plan rejected by preflight
    // never entered TRANSITIONING and therefore must not make an unrelated,
    // never-published target VA part of teardown.
    target_plan_ids[0] = g_uvm_moe_slot_plan_bank_resident_plan_ids[0];
    target_plan_ids[1] = g_uvm_moe_slot_plan_bank_resident_plan_ids[1];
    target_plan_ids[2] = g_uvm_moe_slot_plan_failed_target_plan_id;

    for (i = 0; i < ARRAY_SIZE(target_plan_ids); ++i) {
        const uvm_moe_slot_plan_t *plan;
        NvU32 previous;

        if (target_plan_ids[i] == 0)
            continue;

        for (previous = 0; previous < i; ++previous) {
            if (target_plan_ids[previous] == target_plan_ids[i])
                break;
        }
        if (previous != i)
            continue;

        plan = uvm_moe_slot_plan_find_locked(va_space, target_plan_ids[i]);
        if (!plan) {
            if (target_sweep_status == NV_OK)
                target_sweep_status = NV_ERR_INVALID_STATE;
            continue;
        }

        if (!plan->batch) {
            if (target_sweep_status == NV_OK)
                target_sweep_status = NV_ERR_INVALID_STATE;
            continue;
        }

        if (!has_plan_targets) {
            uvm_uuid_copy(&abort_gpu_uuid, &plan->batch->gpu_uuid);
            has_plan_targets = NV_TRUE;
        }
        else if (!uvm_uuid_eq(&abort_gpu_uuid, &plan->batch->gpu_uuid)) {
            if (target_sweep_status == NV_OK)
                target_sweep_status = NV_ERR_INVALID_STATE;
        }
    }

    if (has_plan_targets && target_sweep_status == NV_OK) {
        abort_gpu = uvm_va_space_retain_gpu_by_uuid(va_space, &abort_gpu_uuid);
        if (!abort_gpu)
            target_sweep_status = NV_ERR_INVALID_DEVICE;
    }

    if (has_plan_targets && target_sweep_status == NV_OK) {
        abort_state = uvm_kvmalloc_zero(sizeof(*abort_state));
        if (!abort_state)
            target_sweep_status = NV_ERR_NO_MEMORY;
    }

    // Lock order is part of the live-abort ABI contract:
    // plan -> retained mm -> VA-space write -> pool.
    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    if (has_plan_targets && target_sweep_status == NV_OK) {
        abort_block_context = uvm_va_block_context_alloc(mm);
        if (!abort_block_context)
            target_sweep_status = NV_ERR_NO_MEMORY;
    }
    uvm_va_space_down_write(va_space);
    mutex_lock(&g_uvm_moe_slot_pool_lock);

    if (has_plan_targets && target_sweep_status == NV_OK) {
        target_sweep_status = uvm_moe_slot_abort_sweep_plan_targets_locked(
            va_space,
            abort_gpu,
            abort_block_context,
            abort_state);
    }

    if (target_sweep_status != NV_OK) {
        // The recipe sweep is the proof that no registered target VA can
        // still reference either bank. Without that proof, release none of
        // this session's backing. Keep slots and plans poisoned so a later
        // abort can retry with the same immutable target recipes.
        first_error = target_sweep_status;
        resources_remain = NV_TRUE;
        for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
            uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];

            if (slot->owner_va_space != va_space)
                continue;

            slot->state = UVM_MOE_SLOT_POOL_STATE_ERROR;
            slot->last_status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
            slot->pte_status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
            ++params->remaining_slot_count;
            params->remaining_bytes += slot->slot_bytes;
        }
        goto release_complete;
    }

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];
        NvBool had_active_pte;
        NvU64 slot_bytes;
        NV_STATUS status;

        if (slot->owner_va_space != va_space) {
            NvU64 mapping_bytes;

            // A healthy slot never maps one VA space while being owned by
            // another. Still clean such a residual mapping so abort cannot
            // report success while this VA space retains a live PTE. Never
            // release backing owned by the other VA space.
            if (slot->pte_va_space != va_space)
                continue;

            mapping_bytes = slot->pte_bytes;
            status = uvm_moe_slot_pool_release_pte_mapping_locked(slot, va_space, mm);
            if (status == NV_OK) {
                uvm_moe_slot_pool_note_pte_clear_reason_locked(slot,
                                                               UVM_MOE_SLOT_PTE_CLEAR_REASON_RELEASE);
                continue;
            }

            slot->state = UVM_MOE_SLOT_POOL_STATE_ERROR;
            slot->last_status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
            uvm_moe_slot_pool_note_pte_failure_locked(slot);
            ++params->remaining_slot_count;
            params->remaining_bytes += mapping_bytes;
            if (first_error == NV_OK)
                first_error = status;
            continue;
        }

        had_active_pte = uvm_moe_slot_pool_slot_has_active_pte(slot);
        slot_bytes = slot->slot_bytes;
        status = uvm_moe_slot_pool_release_backing_locked(slot, va_space, mm);
        if (status == NV_OK) {
            if (had_active_pte)
                uvm_moe_slot_pool_note_pte_clear_reason_locked(slot,
                                                               UVM_MOE_SLOT_PTE_CLEAR_REASON_RELEASE);

            memset(slot, 0, sizeof(*slot));
            ++params->released_slot_count;
            params->released_bytes += slot_bytes;
            continue;
        }

        // Preserve the owner and any backing which release did not finish.
        // This both blocks reuse and gives a later abort call an exact retry
        // target. Plans remain installed and poisoned until every slot is
        // gone, so stale execute requests fail before touching PTEs.
        slot->state = UVM_MOE_SLOT_POOL_STATE_ERROR;
        slot->last_status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
        uvm_moe_slot_pool_note_pte_failure_locked(slot);
        ++params->remaining_slot_count;
        params->remaining_bytes += slot->slot_bytes;
        if (first_error == NV_OK)
            first_error = status;
    }

release_complete:
    // Fail closed on any ownership or PTE residue, including inconsistent
    // metadata which was not part of the normal owner scan above. This check
    // also proves that empty/current-VA abort cannot clear another VA's live
    // registered session.
    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];

        if (slot->owner_va_space == va_space || slot->pte_va_space == va_space) {
            resources_remain = NV_TRUE;
            break;
        }
    }

    // Plans are the final resource released. Never discard the retry recipe
    // while any slot/PTE release failed or any current-VA record remains.
    if (params->remaining_slot_count == 0 && !resources_remain)
        params->destroyed_plan_count = uvm_moe_slot_plan_destroy_va_space_locked(va_space);

    if (!resources_remain) {
        for (i = 0; i < UVM_MOE_SLOT_PLAN_MAX_PLANS; ++i) {
            if (g_uvm_moe_slot_plans[i].valid &&
                g_uvm_moe_slot_plans[i].owner_va_space == va_space) {
                resources_remain = NV_TRUE;
                break;
            }
        }
    }

    if (g_uvm_moe_slot_plan_owner_va_space == va_space)
        resources_remain = NV_TRUE;

    if (params->remaining_slot_count == 0 && !resources_remain) {
        params->session_cleared = 1;
        params->poisoned = 0;
    }
    else {
        if (params->remaining_slot_count == 0) {
            // This is an invariant failure rather than a normal release
            // error. Surface a non-zero retry count so userspace cannot
            // mistake zeroed byte counters for successful teardown.
            params->remaining_slot_count = 1;
            if (first_error == NV_OK)
                first_error = NV_ERR_INVALID_STATE;
        }
        params->session_cleared = 0;
        params->poisoned = 1;
    }

    uvm_va_block_note_slot_session_abort(params->session_cleared,
                                         params->released_slot_count,
                                         params->released_bytes,
                                         params->remaining_slot_count);

    uvm_phase_f_route_log("moe_slot_session_abort va_space=%px generation=%llu had_session=%u released_slots=%u released_bytes=%llu remaining_slots=%u remaining_bytes=%llu destroyed_plans=%u cleared=%u poisoned=%u status=0x%x\n",
                          va_space,
                          params->generation_before,
                          params->had_registered_session,
                          params->released_slot_count,
                          params->released_bytes,
                          params->remaining_slot_count,
                          params->remaining_bytes,
                          params->destroyed_plan_count,
                          params->session_cleared,
                          params->poisoned,
                          first_error);

    mutex_unlock(&g_uvm_moe_slot_pool_lock);
    uvm_va_space_up_write(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);
    uvm_va_block_context_free(abort_block_context);
    uvm_kvfree(abort_state);
    if (abort_gpu)
        uvm_gpu_release(abort_gpu);
    mutex_unlock(&g_uvm_moe_slot_plan_lock);

    return first_error;
}

static bool uvm_moe_static_alias_range_invalid(const UVM_MOE_STATIC_ALIAS_ENTRY *entry)
{
    NvU64 va_end;
    NvU64 slot_end;

    if (entry->va_base == 0 || entry->bytes == 0)
        return true;

    va_end = entry->va_base + entry->bytes;
    slot_end = entry->slot_offset + entry->bytes;
    if (va_end <= entry->va_base || slot_end <= entry->slot_offset)
        return true;

    if (!IS_ALIGNED(entry->va_base, PAGE_SIZE) ||
        !IS_ALIGNED(entry->bytes, PAGE_SIZE) ||
        !IS_ALIGNED(entry->slot_offset, PAGE_SIZE)) {
        return true;
    }

    if (entry->bytes > UVM_MOE_SLOT_DIRECT_MAX_ENTRY_BYTES)
        return true;

    if (entry->flags & ~UVM_MOE_STATIC_ALIAS_ENTRY_FLAGS_ALL)
        return true;

    if ((entry->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K) &&
        (!IS_ALIGNED(entry->va_base, UVM_PAGE_SIZE_64K) ||
         !IS_ALIGNED(entry->bytes, UVM_PAGE_SIZE_64K) ||
         !IS_ALIGNED(entry->slot_offset, UVM_PAGE_SIZE_64K))) {
        return true;
    }

    return false;
}

static bool uvm_moe_static_alias_ranges_overlap(NvU64 a_base,
                                                NvU64 a_bytes,
                                                NvU64 b_base,
                                                NvU64 b_bytes)
{
    return a_base < b_base + b_bytes && b_base < a_base + a_bytes;
}

typedef struct
{
    NvU64 va_base;
    NvU64 va_end;
    NvU32 input_index;
    NvBool is_new;
} uvm_moe_static_alias_range_key_t;

static int uvm_moe_static_alias_range_key_cmp(const void *a, const void *b)
{
    const uvm_moe_static_alias_range_key_t *left = a;
    const uvm_moe_static_alias_range_key_t *right = b;

    if (left->va_base < right->va_base)
        return -1;
    if (left->va_base > right->va_base)
        return 1;
    if (left->va_end < right->va_end)
        return -1;
    if (left->va_end > right->va_end)
        return 1;
    if (left->is_new != right->is_new)
        return left->is_new ? 1 : -1;
    return 0;
}

// Validate a MAP delta and the committed directory with one sorted merge.
// This replaces both the O(new^2) batch scan and O(new * old) append scan.
static NV_STATUS uvm_moe_static_alias_validate_map_ranges(
    UVM_MOE_STATIC_ALIAS_MAP_PARAMS *params,
    const uvm_moe_static_alias_session_t *session)
{
    uvm_moe_static_alias_range_key_t *keys;
    size_t old_count = session ? session->alias_count : 0;
    size_t total_count = old_count + params->count;
    NV_STATUS status = NV_OK;
    size_t i;

    if (total_count < old_count ||
        total_count > SIZE_MAX / sizeof(*keys)) {
        return NV_ERR_OUT_OF_RANGE;
    }

    keys = uvm_kvmalloc_zero(total_count * sizeof(*keys));
    if (!keys)
        return NV_ERR_NO_MEMORY;

    for (i = 0; i < old_count; ++i) {
        const uvm_moe_static_alias_record_t *record = &session->aliases[i];

        if (!record->active || record->bytes == 0 ||
            record->va_base + record->bytes <= record->va_base) {
            status = NV_ERR_INVALID_STATE;
            goto out;
        }
        keys[i].va_base = record->va_base;
        keys[i].va_end = record->va_base + record->bytes;
        keys[i].input_index = NV_U32_MAX;
        keys[i].is_new = NV_FALSE;
    }
    for (i = 0; i < params->count; ++i) {
        const UVM_MOE_STATIC_ALIAS_ENTRY *entry = &params->entries[i];
        uvm_moe_static_alias_range_key_t *key = &keys[old_count + i];

        key->va_base = entry->va_base;
        key->va_end = entry->va_base + entry->bytes;
        key->input_index = (NvU32)i;
        key->is_new = NV_TRUE;
    }

    sort(keys,
         total_count,
         sizeof(*keys),
         uvm_moe_static_alias_range_key_cmp,
         NULL);
    for (i = 1; i < total_count; ++i) {
        const uvm_moe_static_alias_range_key_t *previous = &keys[i - 1];
        const uvm_moe_static_alias_range_key_t *current_key = &keys[i];
        NvU32 conflict_index;

        if (previous->va_end <= current_key->va_base)
            continue;

        if (!previous->is_new && !current_key->is_new) {
            status = NV_ERR_INVALID_STATE;
            goto out;
        }

        conflict_index = current_key->is_new ? current_key->input_index :
                                           previous->input_index;
        if (conflict_index >= params->count) {
            status = NV_ERR_INVALID_STATE;
            goto out;
        }
        params->entries[conflict_index].status =
            UVM_MOE_STATIC_ALIAS_ENTRY_ALIAS_CONFLICT;
        status = previous->is_new && current_key->is_new ?
            NV_ERR_INVALID_ARGUMENT : NV_ERR_IN_USE;
        goto out;
    }

out:
    uvm_kvfree(keys);
    return status;
}

static int uvm_moe_static_alias_record_va_cmp(const void *a, const void *b);

static void uvm_moe_static_alias_dual_recipe_reset_locked(
    uvm_moe_static_alias_session_t *session)
{
    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_plan_lock));

    uvm_kvfree(session->dual_view_block_starts);
    session->dual_view_block_starts = NULL;
    session->dual_view_block_count = 0;
    session->dual_view_generation = 0;
    session->dual_view_full_coverage = NV_FALSE;
    session->dual_view_explicitly_discarded = NV_FALSE;
}

static NV_STATUS uvm_moe_static_alias_dual_append_record(
    NvU64 *block_starts,
    NvU32 capacity,
    NvU32 *count,
    NvU64 record_start,
    NvU64 record_end)
{
    NvU64 block_start;

    if (record_end <= record_start)
        return NV_OK;

    block_start = UVM_VA_BLOCK_ALIGN_DOWN(record_start);
    while (block_start < record_end) {
        // Session records are kept in VA order, so duplicate blocks from
        // adjacent or overlapping 2 MiB parent coverage are consecutive.
        if (*count == 0 || block_starts[*count - 1] != block_start) {
            if (*count >= capacity)
                return NV_ERR_INSUFFICIENT_RESOURCES;
            block_starts[(*count)++] = block_start;
        }
        if (block_start > ~0ULL - UVM_VA_BLOCK_SIZE)
            return NV_ERR_OUT_OF_RANGE;
        block_start += UVM_VA_BLOCK_SIZE;
    }

    return NV_OK;
}

// Build the set of 2 MiB parent entries touched by read-only aliases. The
// prepare helper rejects a VA-range edge block which is not a complete 2 MiB
// UVM block, but a complete block no longer needs to consist exclusively of
// aliases: its per-page mask selects only the expert leaves which differ
// between the prebuilt CPU and GPU children.
static NV_STATUS uvm_moe_static_alias_dual_build_recipe_locked(
    const uvm_moe_static_alias_session_t *session,
    NvU64 **block_starts_out,
    NvU32 *block_count_out)
{
    NvU64 *block_starts = NULL;
    NvU64 capacity64;
    NvU32 capacity;
    NvU32 count = 0;
    NvU32 i;
    NV_STATUS status = NV_OK;

    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_plan_lock));
    *block_starts_out = NULL;
    *block_count_out = 0;

    if (!uvm_moe_static_alias_dual_view_enable)
        return NV_OK;

    // One entry is at most 4 MiB and can cross at most three 2 MiB parent
    // entries when its first byte is unaligned.
    capacity64 = (NvU64)session->consumer_alias_count * 3;
    if (capacity64 > NV_U32_MAX ||
        capacity64 > SIZE_MAX / sizeof(*block_starts)) {
        return NV_ERR_OUT_OF_RANGE;
    }
    capacity = (NvU32)capacity64;
    if (capacity == 0)
        return NV_OK;
    block_starts = uvm_kvmalloc_zero((size_t)capacity *
                                      sizeof(*block_starts));
    if (!block_starts)
        return NV_ERR_NO_MEMORY;

    for (i = 0; i < session->alias_count; ++i) {
        const uvm_moe_static_alias_record_t *record = &session->aliases[i];
        NvU64 record_end;

        record_end = record->va_base + record->bytes;
        if (!record->active || record->bytes == 0 ||
            record_end <= record->va_base ||
            (record->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW)) {
            continue;
        }

        status = uvm_moe_static_alias_dual_append_record(block_starts,
                                                          capacity,
                                                          &count,
                                                          record->va_base,
                                                          record_end);
        if (status != NV_OK)
            goto out;
    }

    *block_starts_out = block_starts;
    *block_count_out = count;
    return NV_OK;

out:
    uvm_kvfree(block_starts);
    return status;
}

static NvBool uvm_moe_static_alias_dual_contains_block(
    const uvm_moe_static_alias_session_t *session,
    NvU64 block_start)
{
    NvU32 left = 0;
    NvU32 right = session->dual_view_block_count;

    while (left < right) {
        NvU32 middle = left + (right - left) / 2;
        NvU64 candidate = session->dual_view_block_starts[middle];

        if (candidate < block_start)
            left = middle + 1;
        else
            right = middle;
    }

    return left < session->dual_view_block_count &&
           session->dual_view_block_starts[left] == block_start;
}

static NvU32 uvm_moe_static_alias_dual_filter_blocks(
    const uvm_moe_static_alias_session_t *session,
    uvm_moe_slot_direct_batch_state_t *state)
{
    NvU32 input_count = state->block_count;
    NvU32 output_index = 0;
    NvU32 input_index;

    for (input_index = 0; input_index < input_count; ++input_index) {
        if (uvm_moe_static_alias_dual_contains_block(
                session, state->blocks[input_index].va_block->start)) {
            continue;
        }
        if (output_index != input_index) {
            state->blocks[output_index] = state->blocks[input_index];
            state->block_entry_indices[output_index] =
                state->block_entry_indices[input_index];
        }
        ++output_index;
    }
    state->block_count = output_index;
    return input_count - output_index;
}

static NV_STATUS uvm_moe_static_alias_dual_prepare_locked(
    uvm_moe_static_alias_session_t *session,
    uvm_va_space_t *va_space,
    struct mm_struct *mm,
    uvm_gpu_t *gpu,
    NvBool nested_refresh)
{
    uvm_va_block_slot_alias_dual_result_t result = {0};
    NvU64 *block_starts = NULL;
    NvU32 block_count = 0;
    NvU32 candidate_block_count;
    NV_STATUS status;

    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_plan_lock));
    uvm_assert_rwsem_locked_write(&va_space->lock);

    status = uvm_moe_static_alias_dual_build_recipe_locked(session,
                                                            &block_starts,
                                                            &block_count);
    if (status != NV_OK)
        return status;

    candidate_block_count = block_count;

    status = uvm_va_block_slot_alias_dual_prepare_batch(va_space,
                                                         mm,
                                                         gpu,
                                                         block_starts,
                                                         &block_count,
                                                         nested_refresh,
                                                         &result);
    if (status != NV_OK) {
        uvm_kvfree(block_starts);
        return status;
    }

    uvm_kvfree(session->dual_view_block_starts);
    session->dual_view_block_starts = block_starts;
    session->dual_view_block_count = block_count;
    session->dual_view_generation = session->topology_generation;
    // build_recipe includes every 2 MiB parent touched by every active
    // read-only record. prepare_batch compacts that array to blocks for which
    // both complete children were actually frozen. Equality is therefore a
    // fail-closed certificate that filtering those blocks removes every
    // consumer leaf from the legacy PARK/RESUME path.
    session->dual_view_full_coverage =
        candidate_block_count != 0 && block_count == candidate_block_count;
    session->dual_view_explicitly_discarded = NV_FALSE;
    uvm_phase_f_route_log(
        "moe_static_alias_dual_prepare session=%llu generation=%llu candidates=%u blocks=%u full=%u prepared=%llu reused=%llu skipped=%llu ns=%llu\n",
        session->session_id,
        session->dual_view_generation,
        candidate_block_count,
        block_count,
        session->dual_view_full_coverage,
        result.prepared_blocks,
        result.reused_blocks,
        result.skipped_blocks,
        result.elapsed_ns);
    if (uvm_moe_static_alias_dual_view_require_full &&
        !session->dual_view_full_coverage) {
        return NV_ERR_NOT_SUPPORTED;
    }
    return NV_OK;
}

static NV_STATUS uvm_moe_static_alias_dual_discard_locked(
    uvm_moe_static_alias_session_t *session,
    uvm_va_space_t *va_space,
    struct mm_struct *mm,
    uvm_gpu_t *gpu)
{
    uvm_va_block_slot_alias_dual_result_t result = {0};
    NV_STATUS status = NV_OK;

    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_plan_lock));
    uvm_assert_rwsem_locked_write(&va_space->lock);

    if (session->dual_view_block_count != 0) {
        status = uvm_va_block_slot_alias_dual_discard_batch(
            va_space,
            mm,
            gpu,
            session->dual_view_block_starts,
            session->dual_view_block_count,
            &result);
        uvm_phase_f_route_log(
            "moe_static_alias_dual_discard session=%llu generation=%llu blocks=%u discarded=%llu hidden_gpu_ptes=%llu ns=%llu status=0x%x\n",
            session->session_id,
            session->dual_view_generation,
            session->dual_view_block_count,
            result.switched_blocks,
            result.pde_writes,
            result.elapsed_ns,
            status);
    }
    if (status == NV_OK)
        uvm_moe_static_alias_dual_recipe_reset_locked(session);
    return status;
}

static void uvm_moe_static_alias_fail_batch(UVM_MOE_STATIC_ALIAS_MAP_PARAMS *params,
                                            NvU32 remaining_status)
{
    NvU32 i;

    params->success_count = 0;
    params->failure_count = 0;
    params->mapped_bytes = 0;

    for (i = 0; i < params->count; ++i) {
        if (params->entries[i].status == UVM_MOE_STATIC_ALIAS_ENTRY_OK)
            params->entries[i].status = remaining_status;
        ++params->failure_count;
    }
}

static bool uvm_moe_static_alias_slot_has_exact_range(
    const uvm_moe_slot_pool_slot_t *slot,
    NvU64 va_base,
    NvU64 bytes)
{
    NvU32 i;

    for (i = 0; i < slot->pte_range_count; ++i) {
        if (slot->pte_ranges[i].va_base == va_base &&
            slot->pte_ranges[i].bytes == bytes) {
            return true;
        }
    }

    return false;
}

static NV_STATUS uvm_api_moe_static_alias_map(UVM_MOE_STATIC_ALIAS_MAP_PARAMS *params,
                                              struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_moe_static_alias_session_t *session = NULL;
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *legacy = NULL;
    uvm_moe_slot_direct_batch_state_t *state = NULL;
    uvm_moe_static_alias_record_t *new_aliases = NULL;
    NvU32 *pending_per_pool = NULL;
    NvBool *pool_in_session = NULL;
    uvm_moe_slot_identity_key_t *pool_keys = NULL;
    uvm_va_block_slot_direct_result_t map_result = {0};
    uvm_tracker_t batch_tracker = UVM_TRACKER_INIT();
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    NV_STATUS status = NV_OK;
    NV_STATUS tracker_status = NV_OK;
    NV_STATUS metadata_status = NV_OK;
    NvU64 requested_bytes = 0;
    NvU64 added_backing_bytes = 0;
    NvU64 added_consumer_bytes = 0;
    NvU32 old_alias_count = 0;
    NvU32 added_consumer_count = 0;
    NvU32 added_upload_count = 0;
    NvU32 pool_key_count = 0;
    NvU32 i;
    bool batch_pte_64k = false;
    bool plan_locked = false;
    bool mm_locked = false;
    bool va_locked = false;
    bool pool_locked = false;
    bool tracker_initialized = false;
    bool mutation_started = false;
    bool creating_session = params->session_id == 0;
    bool quiesced_append =
        (params->flags & UVM_MOE_STATIC_ALIAS_MAP_FLAG_QUIESCED_APPEND) != 0;
    bool defer_dual_refresh =
        (params->flags & UVM_MOE_STATIC_ALIAS_MAP_FLAG_DEFER_DUAL_REFRESH) != 0;

    BUILD_BUG_ON(sizeof(*params) != 221288);
    BUILD_BUG_ON(offsetof(UVM_MOE_STATIC_ALIAS_MAP_PARAMS, entries) != 96);
    BUILD_BUG_ON(offsetof(UVM_MOE_STATIC_ALIAS_MAP_PARAMS, rmStatus) != 221280);

    params->requested_bytes = 0;
    params->mapped_bytes = 0;
    params->total_alias_bytes = 0;
    params->backing_bytes = 0;
    params->success_count = 0;
    params->failure_count = 0;
    params->alias_count = 0;
    params->tlb_invalidate_count = 0;
    params->poisoned = 0;

    if (params->abi_version != UVM_MOE_STATIC_ALIAS_ABI_VERSION ||
        params->struct_size != sizeof(*params) ||
        (params->flags & ~UVM_MOE_STATIC_ALIAS_MAP_FLAGS_ALL) != 0 ||
        params->reserved != 0 ||
        params->count == 0 ||
        params->count > UVM_MOE_STATIC_ALIAS_MAX_ENTRIES) {
        return NV_ERR_INVALID_ARGUMENT;
    }
    if (quiesced_append && creating_session)
        return NV_ERR_INVALID_ARGUMENT;
    if (defer_dual_refresh && !quiesced_append)
        return NV_ERR_INVALID_ARGUMENT;

    for (i = 0; i < params->count; ++i) {
        UVM_MOE_STATIC_ALIAS_ENTRY *entry = &params->entries[i];

        entry->status = UVM_MOE_STATIC_ALIAS_ENTRY_OK;
        if (uvm_moe_static_alias_range_invalid(entry) ||
            requested_bytes + entry->bytes < requested_bytes) {
            entry->status = UVM_MOE_STATIC_ALIAS_ENTRY_INVALID_ARGUMENT;
            status = NV_ERR_INVALID_ARGUMENT;
            continue;
        }

        if (i == 0) {
            batch_pte_64k =
                (entry->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K) != 0;
        }
        else if (((entry->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K) != 0) !=
                 batch_pte_64k) {
            entry->status = UVM_MOE_STATIC_ALIAS_ENTRY_INVALID_ARGUMENT;
            status = NV_ERR_INVALID_ARGUMENT;
            continue;
        }

        requested_bytes += entry->bytes;
        if (entry->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW)
            ++added_upload_count;
        else {
            ++added_consumer_count;
            added_consumer_bytes += entry->bytes;
        }
    }
    params->requested_bytes = requested_bytes;

    if (status != NV_OK) {
        uvm_moe_static_alias_fail_batch(params, UVM_MOE_STATIC_ALIAS_ENTRY_ROLLED_BACK);
        return status;
    }

    legacy = uvm_kvmalloc_zero(sizeof(*legacy));
    state = uvm_kvmalloc_zero(sizeof(*state));
    pending_per_pool = uvm_kvmalloc_zero(
        UVM_MOE_SLOT_POOL_MAX_SLOTS * sizeof(*pending_per_pool));
    pool_in_session = uvm_kvmalloc_zero(
        UVM_MOE_SLOT_POOL_MAX_SLOTS * sizeof(*pool_in_session));
    pool_keys = uvm_kvmalloc_zero(
        UVM_MOE_SLOT_POOL_MAX_SLOTS * sizeof(*pool_keys));
    if (!legacy || !state || !pending_per_pool || !pool_in_session ||
        !pool_keys) {
        status = NV_ERR_NO_MEMORY;
        goto out;
    }

    legacy->count = params->count;
    legacy->mode = UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE;
    uvm_uuid_copy(&legacy->gpu_uuid, &params->gpu_uuid);
    for (i = 0; i < params->count; ++i) {
        const UVM_MOE_STATIC_ALIAS_ENTRY *entry = &params->entries[i];
        UVM_MOE_SLOT_REMAP_ENTRY *legacy_entry = &legacy->entries[i];

        legacy_entry->layer_id = entry->backing_layer_id;
        legacy_entry->expert_id = entry->backing_expert_id;
        legacy_entry->slot_id = entry->backing_slot_id;
        legacy_entry->va_base = entry->va_base;
        legacy_entry->bytes = entry->bytes;
        legacy_entry->slot_offset = entry->slot_offset;
        legacy_entry->flags = UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT;
        legacy_entry->status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
    }

    mutex_lock(&g_uvm_moe_slot_plan_lock);
    plan_locked = true;

    if (g_uvm_moe_slot_plan_owner_va_space != NULL) {
        status = NV_ERR_IN_USE;
        goto out;
    }

    if (creating_session) {
        // Candidate UVM file descriptions are probed by userspace. Do not
        // consume a registry entry, or return a non-zero session id, until
        // this VA space has passed the GPU, backing, and target-VA checks.
        // The outer plan lock remains held through the eventual allocation,
        // so deferring it does not introduce a reservation race.
        if (uvm_moe_static_alias_find_va_space_locked(va_space)) {
            status = NV_ERR_IN_USE;
            goto out;
        }
    }
    else {
        session = uvm_moe_static_alias_find_locked(va_space, params->session_id);
        if (!session) {
            status = NV_ERR_OBJECT_NOT_FOUND;
            goto out;
        }

        if (session->poisoned ||
            !uvm_uuid_eq(&session->gpu_uuid, &params->gpu_uuid)) {
            status = session->poisoned ? NV_ERR_INVALID_STATE : NV_ERR_IN_USE;
            goto out;
        }
        if (quiesced_append) {
            if (!session->sealed || session->parked ||
                session->rebind_id != 0 || session->rebind_aliases != NULL) {
                status = NV_ERR_IN_USE;
                goto out;
            }
        }
        else if (session->sealed) {
            status = NV_ERR_IN_USE;
            goto out;
        }

        if (session->alias_count > NV_U32_MAX - params->count) {
            status = NV_ERR_OUT_OF_RANGE;
            goto out;
        }

        old_alias_count = session->alias_count;
    }

    status = uvm_moe_static_alias_validate_map_ranges(params, session);
    if (status != NV_OK)
        goto out;

    new_aliases = uvm_kvmalloc_zero((old_alias_count + params->count) * sizeof(*new_aliases));
    if (!new_aliases) {
        status = NV_ERR_NO_MEMORY;
        goto out;
    }
    if (old_alias_count != 0)
        memcpy(new_aliases, session->aliases, old_alias_count * sizeof(*new_aliases));

    gpu = uvm_va_space_retain_gpu_by_uuid(va_space, &params->gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto out;
    }

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    mm_locked = true;
    uvm_va_space_down_write(va_space);
    va_locked = true;
    mutex_lock(&g_uvm_moe_slot_pool_lock);
    pool_locked = true;

    // Build one immutable lookup for this MAP transaction instead of making
    // every submitted alias scan all 8,192 pool entries. Multiple aliases may
    // legitimately share one backing slot, so index the pool (whose logical
    // identities must be unique) and binary-search it for each input record.
    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        const uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];

        if (!slot->valid || slot->owner_va_space != va_space)
            continue;

        pool_keys[pool_key_count].layer_id = slot->layer_id;
        pool_keys[pool_key_count].slot_id = slot->slot_id;
        pool_keys[pool_key_count].entry_index = i;
        ++pool_key_count;
    }
    sort(pool_keys,
         pool_key_count,
         sizeof(pool_keys[0]),
         uvm_moe_slot_identity_key_cmp,
         NULL);
    for (i = 1; i < pool_key_count; ++i) {
        if (pool_keys[i - 1].layer_id == pool_keys[i].layer_id &&
            pool_keys[i - 1].slot_id == pool_keys[i].slot_id) {
            status = NV_ERR_INVALID_STATE;
            goto out;
        }
    }

    if (session) {
        for (i = 0; i < session->alias_count; ++i) {
            const uvm_moe_static_alias_record_t *record =
                &session->aliases[i];

            if (!record->active ||
                record->pool_index >= UVM_MOE_SLOT_POOL_MAX_SLOTS) {
                status = NV_ERR_INVALID_STATE;
                goto out;
            }
            pool_in_session[record->pool_index] = NV_TRUE;
        }
    }

    for (i = 0; i < params->count; ++i) {
        UVM_MOE_STATIC_ALIAS_ENTRY *entry = &params->entries[i];
        const uvm_moe_slot_identity_key_t *pool_key;
        uvm_moe_slot_pool_slot_t *slot;
        NvU32 pool_index;

        pool_key = uvm_moe_slot_identity_key_find(pool_keys,
                                                  pool_key_count,
                                                  entry->backing_layer_id,
                                                  entry->backing_slot_id);
        slot = pool_key && pool_key->entry_index < UVM_MOE_SLOT_POOL_MAX_SLOTS ?
            &g_uvm_moe_slot_pool[pool_key->entry_index] : NULL;
        if (!slot || slot->owner_expert_id != entry->backing_expert_id) {
            entry->status = UVM_MOE_STATIC_ALIAS_ENTRY_BACKING_NOT_FOUND;
            status = NV_ERR_OBJECT_NOT_FOUND;
            break;
        }

        if (!uvm_moe_slot_pool_slot_has_vidmem_backing(slot, gpu) ||
            !(slot->flags & UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT) ||
            entry->slot_offset + entry->bytes > slot->slot_bytes) {
            entry->status = UVM_MOE_STATIC_ALIAS_ENTRY_BACKING_NOT_FOUND;
            status = NV_ERR_INVALID_STATE;
            break;
        }

        if (slot->state != UVM_MOE_SLOT_POOL_STATE_READY ||
            slot->async_copy_pending ||
            slot->async_copy_staging ||
            slot->static_alias_poisoned ||
            (slot->static_alias_session_id != 0 &&
             (!session || slot->static_alias_session_id != session->session_id))) {
            entry->status = UVM_MOE_STATIC_ALIAS_ENTRY_ALIAS_CONFLICT;
            status = NV_ERR_IN_USE;
            break;
        }

        if (slot->static_alias_session_id == 0 &&
            !uvm_moe_slot_direct_pte_metadata_empty(slot)) {
            entry->status = UVM_MOE_STATIC_ALIAS_ENTRY_ALIAS_CONFLICT;
            status = NV_ERR_IN_USE;
            break;
        }

        if (session && slot->static_alias_session_id == session->session_id &&
            (slot->pte_gpu != gpu ||
             slot->pte_va_space != va_space ||
             slot->pte_mode != UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP)) {
            entry->status = UVM_MOE_STATIC_ALIAS_ENTRY_ALIAS_CONFLICT;
            status = NV_ERR_INVALID_STATE;
            break;
        }

        pool_index = (NvU32)(slot - g_uvm_moe_slot_pool);
        ++pending_per_pool[pool_index];
        if (slot->pte_range_count + pending_per_pool[pool_index] >
            UVM_MOE_SLOT_POOL_MAX_PTE_RANGES_PER_SLOT) {
            entry->status = UVM_MOE_STATIC_ALIAS_ENTRY_ALIAS_CONFLICT;
            status = NV_ERR_INSUFFICIENT_RESOURCES;
            break;
        }

        state->slots[i] = slot;
    }

    if (status != NV_OK)
        goto out;

    status = uvm_moe_slot_direct_flatten_blocks_locked(legacy, va_space, state, NV_TRUE);
    if (status != NV_OK)
        goto out;

    for (i = 0; i < state->block_count; ++i) {
        NvU32 entry_index = state->block_entry_indices[i];
        NvU32 flags = params->entries[entry_index].flags;

        state->blocks[i].pte_mode =
            (flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K) ?
                UVM_SLOT_DIRECT_PTE_MODE_64K : UVM_SLOT_DIRECT_PTE_MODE_4K;
        state->blocks[i].writable =
            (flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW) != 0;
        state->blocks[i].slot_tracker = NULL;
    }

    status = uvm_va_block_slot_direct_preflight_targets_not_slot_mapped(va_space,
                                                                         gpu,
                                                                         state->blocks,
                                                                         state->block_count);
    if (status != NV_OK)
        goto out;

    if (creating_session) {
        session = uvm_moe_static_alias_alloc_locked(va_space, &params->gpu_uuid);
        if (!session) {
            status = NV_ERR_INSUFFICIENT_RESOURCES;
            goto out;
        }
        params->session_id = session->session_id;
    }

    // Teardown metadata is part of the safety transaction, not deferred
    // bookkeeping. Publish it before either CPU or GPU PTEs can change.
    // Capacity and identity were validated above while holding the same
    // slot-pool lock. If this unexpectedly fails partway, commit complete
    // session ownership without changing PTEs, poison the session, and let
    // DESTROY converge the harmless partial metadata.
    metadata_status = uvm_moe_slot_direct_record_all_pte_metadata_locked(legacy,
                                                                          va_space,
                                                                          gpu,
                                                                          state);

    // Commit the full requested topology and slot ownership before revoking
    // CPU PTEs. This closes the window in which an already-present writable
    // CPU mapping could be rebuilt without the static-alias fault guard seeing
    // the target range. In the metadata-failure path no PTE mutation follows,
    // so a missing per-slot range can never describe a live slot PTE.
    for (i = 0; i < params->count; ++i) {
        uvm_moe_slot_pool_slot_t *slot = state->slots[i];
        NvU32 pool_index = (NvU32)(slot - g_uvm_moe_slot_pool);
        bool already_counted = pool_in_session[pool_index];

        new_aliases[old_alias_count + i].va_base = params->entries[i].va_base;
        new_aliases[old_alias_count + i].bytes = params->entries[i].bytes;
        new_aliases[old_alias_count + i].slot_offset = params->entries[i].slot_offset;
        new_aliases[old_alias_count + i].backing_layer_id =
            params->entries[i].backing_layer_id;
        new_aliases[old_alias_count + i].backing_expert_id =
            params->entries[i].backing_expert_id;
        new_aliases[old_alias_count + i].backing_slot_id =
            params->entries[i].backing_slot_id;
        new_aliases[old_alias_count + i].pool_index = pool_index;
        new_aliases[old_alias_count + i].flags = params->entries[i].flags;
        new_aliases[old_alias_count + i].active = NV_TRUE;

        if (!already_counted) {
            added_backing_bytes += slot->slot_bytes;
            pool_in_session[pool_index] = NV_TRUE;
        }

        slot->static_alias_session_id = session->session_id;
        slot->static_alias_sealed = quiesced_append ? NV_TRUE : NV_FALSE;
        if (metadata_status != NV_OK)
            slot->static_alias_poisoned = NV_TRUE;
    }

    uvm_kvfree(session->aliases);
    session->aliases = new_aliases;
    new_aliases = NULL;
    session->alias_count = old_alias_count + params->count;
    session->alias_bytes += requested_bytes;
    session->backing_bytes += added_backing_bytes;
    session->consumer_alias_count += added_consumer_count;
    session->consumer_alias_bytes += added_consumer_bytes;
    session->upload_alias_count += added_upload_count;
    sort(session->aliases,
         session->alias_count,
         sizeof(session->aliases[0]),
         uvm_moe_static_alias_record_va_cmp,
         NULL);
    if (metadata_status != NV_OK)
        session->poisoned = NV_TRUE;

    params->alias_count = params->count;
    params->total_alias_bytes = session->alias_bytes;
    params->backing_bytes = session->backing_bytes;
    params->poisoned = session->poisoned;
    mutation_started = true;

    if (metadata_status != NV_OK) {
        status = metadata_status;
        goto out;
    }

    // Existing writable CPU PTEs do not fault, so a fault-path guard alone
    // cannot prevent CPU/sysmem and GPU/slot views from diverging. Revoke the
    // selected CPU mappings while preserving CPU residency. The committed
    // range metadata makes any later CPU refault fail closed until cmd99.
    status = uvm_va_block_slot_alias_revoke_cpu_mappings_batch(va_space,
                                                                mm,
                                                                state->blocks,
                                                                state->block_count);
    if (status != NV_OK)
        goto out;

    uvm_tracker_init(&batch_tracker);
    tracker_initialized = true;
    status = uvm_va_block_slot_alias_map_batch(va_space,
                                                mm,
                                                gpu,
                                                state->blocks,
                                                state->block_count,
                                                false,
                                                true,
                                                &g_uvm_moe_slot_pool_lock,
                                                &batch_tracker,
                                                &map_result);
    // The helper can fail after an earlier block submitted hierarchy or unmap
    // work only to that block's tracker. Snapshot every unique block before
    // returning so a failed MAP is still a PTE/TLB completion boundary.
    if (status != NV_OK) {
        for (i = 0; i < state->block_count; ++i) {
            uvm_va_block_t *block = state->blocks[i].va_block;
            NV_STATUS add_status;

            if (i > 0 && state->blocks[i - 1].va_block == block)
                continue;

            uvm_mutex_lock(&block->lock);
            add_status = uvm_tracker_add_tracker_safe(&batch_tracker,
                                                       &block->tracker);
            uvm_mutex_unlock(&block->lock);
            if (tracker_status == NV_OK)
                tracker_status = add_status;
        }
    }
    {
        NV_STATUS wait_status = uvm_tracker_wait(&batch_tracker);

        if (tracker_status == NV_OK)
            tracker_status = wait_status;
    }
    if (status == NV_OK && tracker_status != NV_OK)
        status = tracker_status;

    params->tlb_invalidate_count = (NvU32)min_t(NvU64,
                                                map_result.mapped_blocks,
                                                (NvU64)NV_U32_MAX);
    if (status == NV_OK) {
        ++session->topology_generation;
        if (session->topology_generation == 0)
            ++session->topology_generation;
        if (quiesced_append) {
            if (defer_dual_refresh) {
                // Userspace will change the physical KV/expert layout next.
                // Keep the frozen children owned by their VA blocks, but
                // revoke this session's recipe so PARK cannot publish stale
                // entries. SEAL(REFRESH_DUAL) rediscovers and refreshes every
                // child exactly once after the layout transaction commits.
                uvm_moe_static_alias_dual_recipe_reset_locked(session);
            }
            else {
                status = uvm_moe_static_alias_dual_prepare_locked(session,
                                                                   va_space,
                                                                   mm,
                                                                   gpu,
                                                                   NV_FALSE);
            }
        }
    }
    if (status == NV_OK) {
        session->certificate_generation = session->topology_generation;
        session->certificate_valid = NV_TRUE;
        session->certificate_parked = NV_FALSE;
        params->mapped_bytes = requested_bytes;
        params->success_count = params->count;
        uvm_phase_f_route_log(
            "moe_static_alias_map session=%llu quiesced_append=%u added_aliases=%u added_bytes=%llu total_aliases=%u total_bytes=%llu backing=%llu\n",
            session->session_id,
            quiesced_append,
            params->count,
            requested_bytes,
            session->alias_count,
            session->alias_bytes,
            session->backing_bytes);
    }

out:
    if (status != NV_OK) {
        NvU32 failure_status = mutation_started ?
            UVM_MOE_STATIC_ALIAS_ENTRY_MAP_FAILED :
            UVM_MOE_STATIC_ALIAS_ENTRY_ROLLED_BACK;

        if (session && mutation_started) {
            session->poisoned = NV_TRUE;
            for (i = 0; i < params->count; ++i) {
                if (state && state->slots[i] &&
                    state->slots[i]->static_alias_session_id == session->session_id) {
                    state->slots[i]->static_alias_poisoned = NV_TRUE;
                }
            }
        }
        uvm_moe_static_alias_fail_batch(params, failure_status);
        if (session) {
            params->total_alias_bytes = session->alias_bytes;
            params->backing_bytes = session->backing_bytes;
            params->poisoned = session->poisoned;
        }
    }

    if (tracker_initialized)
        uvm_tracker_deinit(&batch_tracker);
    if (pool_locked)
        mutex_unlock(&g_uvm_moe_slot_pool_lock);
    if (va_locked)
        uvm_va_space_up_write(va_space);
    if (mm_locked)
        uvm_va_space_mm_or_current_release_unlock(va_space, mm);
    if (gpu)
        uvm_gpu_release(gpu);
    if (plan_locked)
        mutex_unlock(&g_uvm_moe_slot_plan_lock);
    uvm_kvfree(new_aliases);
    uvm_kvfree(pool_keys);
    uvm_kvfree(pool_in_session);
    uvm_kvfree(pending_per_pool);
    uvm_kvfree(state);
    uvm_kvfree(legacy);

    return status;
}

static NV_STATUS uvm_api_moe_static_alias_seal(UVM_MOE_STATIC_ALIAS_SEAL_PARAMS *params,
                                               struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_moe_static_alias_session_t *session;
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    NV_STATUS status = NV_OK;
    NvU32 active_count = 0;
    NvU64 active_bytes = 0;
    NvU32 i;
    NvBool refresh_dual;
    NvBool nested_refresh;
    NvBool discard_dual;

    BUILD_BUG_ON(sizeof(*params) != 72);

    params->alias_bytes = 0;
    params->backing_bytes = 0;
    params->sealed = 0;
    params->alias_count = 0;
    params->poisoned = 0;

    if (params->abi_version != UVM_MOE_STATIC_ALIAS_ABI_VERSION ||
        params->struct_size != sizeof(*params) ||
        (params->flags & ~UVM_MOE_STATIC_ALIAS_SEAL_FLAGS_ALL) != 0 ||
        params->reserved != 0 ||
        params->session_id == 0) {
        return NV_ERR_INVALID_ARGUMENT;
    }
    refresh_dual =
        (params->flags & UVM_MOE_STATIC_ALIAS_SEAL_FLAG_REFRESH_DUAL) != 0;
    nested_refresh =
        (params->flags &
         UVM_MOE_STATIC_ALIAS_SEAL_FLAG_REFRESH_DUAL_NESTED) != 0;
    discard_dual =
        (params->flags & UVM_MOE_STATIC_ALIAS_SEAL_FLAG_DISCARD_DUAL) != 0;
    if ((nested_refresh && !refresh_dual) ||
        (discard_dual && (refresh_dual || nested_refresh)))
        return NV_ERR_INVALID_ARGUMENT;

    mutex_lock(&g_uvm_moe_slot_plan_lock);
    session = uvm_moe_static_alias_find_locked(va_space, params->session_id);
    if (!session) {
        status = NV_ERR_OBJECT_NOT_FOUND;
        goto out_plan;
    }

    params->alias_bytes = session->alias_bytes;
    params->backing_bytes = session->backing_bytes;
    params->alias_count = session->alias_count;
    params->poisoned = session->poisoned;

    if (session->poisoned || session->alias_count == 0 ||
        ((refresh_dual || discard_dual) ?
             !session->sealed : session->sealed)) {
        status = session->poisoned ? NV_ERR_INVALID_STATE :
                 (refresh_dual || discard_dual) ?
                     NV_ERR_INVALID_STATE : NV_ERR_IN_USE;
        goto out_plan;
    }

    for (i = 0; i < session->alias_count; ++i) {
        if (!session->aliases[i].active)
            continue;
        ++active_count;
        active_bytes += session->aliases[i].bytes;
    }

    if (active_count != session->alias_count ||
        active_count != params->expected_alias_count ||
        active_bytes != session->alias_bytes ||
        active_bytes != params->expected_alias_bytes) {
        status = NV_ERR_INVALID_ARGUMENT;
        goto out_plan;
    }

    gpu = uvm_va_space_retain_gpu_by_uuid(va_space, &session->gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto out_plan;
    }
    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_write(va_space);
    mutex_lock(&g_uvm_moe_slot_pool_lock);

    for (i = 0; i < session->alias_count; ++i) {
        const uvm_moe_static_alias_record_t *record = &session->aliases[i];
        uvm_moe_slot_pool_slot_t *slot;
        NvBool mapped;

        if (!record->active || record->pool_index >= UVM_MOE_SLOT_POOL_MAX_SLOTS) {
            status = NV_ERR_INVALID_STATE;
            break;
        }

        slot = &g_uvm_moe_slot_pool[record->pool_index];
        if (!slot->valid ||
            slot->owner_va_space != va_space ||
            slot->static_alias_session_id != session->session_id ||
            slot->static_alias_poisoned ||
            slot->pte_va_space != va_space ||
            slot->pte_bytes == 0 ||
            slot->pte_gpu == NULL ||
            slot->pte_mode != UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP ||
            slot->pte_range_count == 0) {
            status = NV_ERR_INVALID_STATE;
            break;
        }

        mapped = uvm_moe_static_alias_slot_has_exact_range(slot,
                                                            record->va_base,
                                                            record->bytes);
        if (!refresh_dual && !discard_dual && !mapped) {
            status = NV_ERR_INVALID_STATE;
            break;
        }
        if ((refresh_dual || discard_dual) &&
            ((record->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW) ?
                !mapped : mapped == session->parked)) {
            status = NV_ERR_INVALID_STATE;
            break;
        }

        // During a refresh the session may be PARKED, so the consumer slot
        // leaves are intentionally absent from the live software mask. The
        // dual-view refresh below validates both frozen children and rebuilds
        // their ordinary leaves; the original SEAL path still performs the
        // full live-range validation.
        if (!refresh_dual && !discard_dual) {
            status = uvm_va_block_validate_slot_alias_range(
                va_space,
                slot->pte_gpu,
                record->va_base,
                record->bytes,
                (record->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW) != 0);
            if (status != NV_OK)
                break;
        }
    }

    if (status == NV_OK) {
        if (discard_dual) {
            status = uvm_moe_static_alias_dual_discard_locked(
                session, va_space, mm, gpu);
            if (status == NV_OK)
                session->dual_view_explicitly_discarded = NV_TRUE;
        }
        else
            status = uvm_moe_static_alias_dual_prepare_locked(session,
                                                               va_space,
                                                               mm,
                                                               gpu,
                                                               nested_refresh);
    }

    if (status == NV_OK) {
        if (!refresh_dual && !discard_dual) {
            for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
                uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];

                if (slot->static_alias_session_id == session->session_id)
                    slot->static_alias_sealed = NV_TRUE;
            }
            session->sealed = NV_TRUE;
            session->certificate_generation = session->topology_generation;
            session->certificate_valid = NV_TRUE;
            session->certificate_parked = NV_FALSE;
        }
        params->sealed = 1;
    }
    else {
        session->poisoned = NV_TRUE;
        params->poisoned = 1;
    }

    mutex_unlock(&g_uvm_moe_slot_pool_lock);
    uvm_va_space_up_write(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

out_plan:
    if (gpu)
        uvm_gpu_release(gpu);
    mutex_unlock(&g_uvm_moe_slot_plan_lock);
    return status;
}

static void uvm_moe_static_alias_transition_snapshot(
    const uvm_moe_static_alias_session_t *session,
    UVM_MOE_STATIC_ALIAS_TRANSITION_PARAMS *params)
{
    params->consumer_alias_bytes = 0;
    params->backing_bytes = 0;
    params->consumer_alias_count = 0;
    params->upload_alias_count = 0;
    params->parked = 0;
    params->sealed = 0;
    params->poisoned = 0;
    params->dual_view_full_coverage = 0;
    params->dual_view_block_count = 0;

    if (!session)
        return;

    params->consumer_alias_count = session->consumer_alias_count;
    params->consumer_alias_bytes = session->consumer_alias_bytes;
    params->upload_alias_count = session->upload_alias_count;
    params->backing_bytes = session->backing_bytes;
    params->parked = session->parked;
    params->sealed = session->sealed;
    params->poisoned = session->poisoned;
    params->dual_view_full_coverage = session->dual_view_full_coverage;
    params->dual_view_block_count = session->dual_view_block_count;
}

static NV_STATUS uvm_moe_static_alias_refresh_certificate_locked(
    uvm_moe_static_alias_session_t *session)
{
    NvU64 consumer_bytes = 0;
    NvU64 total_bytes = 0;
    NvU32 consumer_count = 0;
    NvU32 upload_count = 0;
    NvU32 i;

    for (i = 0; i < session->alias_count; ++i) {
        const uvm_moe_static_alias_record_t *record = &session->aliases[i];

        if (!record->active || record->bytes == 0 ||
            total_bytes + record->bytes < total_bytes) {
            session->certificate_valid = NV_FALSE;
            return NV_ERR_INVALID_STATE;
        }
        total_bytes += record->bytes;
        if (record->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW)
            ++upload_count;
        else {
            ++consumer_count;
            consumer_bytes += record->bytes;
        }
    }
    if (total_bytes != session->alias_bytes || consumer_count == 0 ||
        upload_count == 0) {
        session->certificate_valid = NV_FALSE;
        return NV_ERR_INVALID_STATE;
    }

    session->consumer_alias_count = consumer_count;
    session->consumer_alias_bytes = consumer_bytes;
    session->upload_alias_count = upload_count;
    ++session->topology_generation;
    if (session->topology_generation == 0)
        ++session->topology_generation;
    session->certificate_generation = session->topology_generation;
    session->certificate_valid = NV_TRUE;
    session->certificate_parked = session->parked;
    return NV_OK;
}

static NV_STATUS uvm_moe_static_alias_remove_slot_range_locked(
    uvm_moe_slot_pool_slot_t *slot,
    NvU64 va_base,
    NvU64 bytes)
{
    NvU32 index;
    NvU32 i;

    for (index = 0; index < slot->pte_range_count; ++index) {
        if (slot->pte_ranges[index].va_base == va_base &&
            slot->pte_ranges[index].bytes == bytes) {
            break;
        }
    }
    if (index == slot->pte_range_count)
        return NV_ERR_OBJECT_NOT_FOUND;

    if (index + 1 < slot->pte_range_count) {
        memmove(&slot->pte_ranges[index],
                &slot->pte_ranges[index + 1],
                (slot->pte_range_count - index - 1) * sizeof(slot->pte_ranges[0]));
    }
    --slot->pte_range_count;
    memset(&slot->pte_ranges[slot->pte_range_count],
           0,
           sizeof(slot->pte_ranges[0]));

    slot->pte_va_base = 0;
    slot->pte_bytes = 0;
    for (i = 0; i < slot->pte_range_count; ++i) {
        const uvm_moe_slot_pool_pte_range_t *range = &slot->pte_ranges[i];

        if (slot->pte_bytes == 0 || range->va_base < slot->pte_va_base)
            slot->pte_va_base = range->va_base;
        slot->pte_bytes += range->bytes;
    }

    return NV_OK;
}

static NV_STATUS uvm_moe_static_alias_validate_transition_locked(
    const uvm_moe_static_alias_session_t *session,
    uvm_va_space_t *va_space,
    NvBool expect_parked)
{
    NvU32 i;

    for (i = 0; i < session->alias_count; ++i) {
        const uvm_moe_static_alias_record_t *record = &session->aliases[i];
        const uvm_moe_slot_pool_slot_t *slot;
        NvBool mapped;

        if (!record->active || record->pool_index >= UVM_MOE_SLOT_POOL_MAX_SLOTS)
            return NV_ERR_INVALID_STATE;

        slot = &g_uvm_moe_slot_pool[record->pool_index];
        if (!slot->valid ||
            slot->owner_va_space != va_space ||
            slot->static_alias_session_id != session->session_id ||
            slot->static_alias_poisoned ||
            slot->pte_va_space != va_space ||
            slot->pte_bytes == 0 ||
            slot->pte_gpu == NULL ||
            slot->pte_mode != UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP) {
            return NV_ERR_INVALID_STATE;
        }

        mapped = uvm_moe_static_alias_slot_has_exact_range(slot,
                                                            record->va_base,
                                                            record->bytes);
        if (record->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW) {
            if (!mapped)
                return NV_ERR_INVALID_STATE;
        }
        else if (mapped == expect_parked) {
            // Active topology: consumers must be mapped. Parked topology:
            // consumers must be absent while their immutable records survive.
            return NV_ERR_INVALID_STATE;
        }
    }

    return NV_OK;
}

static NV_STATUS uvm_moe_static_alias_build_park_clear_state_locked(
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *legacy,
    const uvm_moe_static_alias_session_t *session,
    uvm_va_space_t *va_space,
    uvm_gpu_t *gpu,
    uvm_moe_slot_direct_batch_state_t *state,
    uvm_moe_slot_transition_clear_state_t *clear_state)
{
    NvU32 i;

    uvm_assert_rwsem_locked_write(&va_space->lock);

    for (i = 0; i < legacy->count; ++i) {
        const UVM_MOE_SLOT_REMAP_ENTRY *entry = &legacy->entries[i];
        uvm_moe_slot_pool_slot_t *slot = state->slots[i];
        NvU64 range_end = entry->va_base + entry->bytes;
        NvU64 cursor = entry->va_base;

        if (!slot ||
            slot->pte_range_vec ||
            slot->pte_gpu != gpu ||
            slot->pte_va_space != va_space ||
            slot->pte_mode != UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP ||
            !slot->pte_va_block_tracked ||
            entry->bytes == 0 ||
            range_end <= entry->va_base ||
            !IS_ALIGNED(entry->va_base, PAGE_SIZE) ||
            !IS_ALIGNED(entry->bytes, PAGE_SIZE) ||
            !uvm_moe_static_alias_slot_has_exact_range(slot,
                                                       entry->va_base,
                                                       entry->bytes)) {
            return NV_ERR_INVALID_STATE;
        }

        while (cursor < range_end) {
            uvm_moe_slot_transition_clear_slice_t *slice;
            uvm_va_block_t *va_block = NULL;
            NvU64 slice_end;
            NV_STATUS status;

            if (clear_state->slice_count >= UVM_MOE_SLOT_DIRECT_MAX_BLOCKS)
                return NV_ERR_INSUFFICIENT_RESOURCES;

            // A sealed alias already owns PTEs in an existing managed block.
            // Never create VA state merely to make PARK eligible.
            status = uvm_va_block_find(va_space, cursor, &va_block);
            if (status != NV_OK ||
                !va_block ||
                uvm_va_block_is_dead(va_block) ||
                uvm_va_block_is_hmm(va_block) ||
                uvm_va_block_get_va_space(va_block) != va_space ||
                cursor > va_block->end) {
                return status == NV_OK ? NV_ERR_INVALID_STATE : status;
            }

            slice_end = min(range_end, va_block->end + 1);
            if (slice_end <= cursor)
                return NV_ERR_INVALID_STATE;

            if (uvm_moe_static_alias_dual_contains_block(session,
                                                          va_block->start)) {
                cursor = slice_end;
                continue;
            }

            slice = &clear_state->slices[clear_state->slice_count++];
            slice->va_block = va_block;
            slice->region =
                uvm_va_block_region_from_start_end(va_block, cursor, slice_end - 1);
            slice->slot_index = i;
            ++clear_state->slot_block_counts[i];
            cursor = slice_end;
        }

        ++clear_state->range_count;
    }

    sort(clear_state->slices,
         clear_state->slice_count,
         sizeof(clear_state->slices[0]),
         uvm_moe_slot_transition_clear_slice_cmp,
         NULL);

    // A full-coverage certificate promises that the visible parent switch is
    // the only hardware mapping mutation. Never silently run a legacy leaf
    // clear if the recipe and the current records disagree.
    if (session->dual_view_full_coverage && clear_state->slice_count != 0)
        return NV_ERR_INVALID_STATE;

    for (i = 0; i < clear_state->slice_count; ++i) {
        const uvm_moe_slot_transition_clear_slice_t *slice =
            &clear_state->slices[i];
        uvm_moe_slot_transition_clear_group_t *group;

        clear_state->page_count += uvm_va_block_region_num_pages(slice->region);
        if (i == 0 || clear_state->slices[i - 1].va_block != slice->va_block) {
            if (clear_state->group_count >= UVM_MOE_SLOT_DIRECT_MAX_BLOCKS)
                return NV_ERR_INSUFFICIENT_RESOURCES;
            ++clear_state->unique_block_count;
            group = &clear_state->groups[clear_state->group_count++];
            group->va_block = slice->va_block;
            group->first_slice = i;
            uvm_page_mask_zero(&group->page_mask);
        }
        else {
            group = &clear_state->groups[clear_state->group_count - 1];
            if (clear_state->slices[i - 1].region.outer > slice->region.first)
                return NV_ERR_INVALID_STATE;
        }

        ++group->slice_count;
        uvm_page_mask_region_fill(&group->page_mask, slice->region);
    }

    if (clear_state->range_count != legacy->count ||
        clear_state->group_count != clear_state->unique_block_count) {
        return NV_ERR_INVALID_STATE;
    }

    return NV_OK;
}

static NV_STATUS uvm_moe_static_alias_park_chunk_locked(
    uvm_moe_static_alias_session_t *session,
    uvm_va_space_t *va_space,
    uvm_gpu_t *gpu,
    struct mm_struct *mm,
    NvU32 *record_cursor,
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *legacy,
    uvm_moe_slot_direct_batch_state_t *state,
    uvm_moe_slot_transition_clear_state_t *clear_state,
    uvm_va_block_context_t *block_context,
    UVM_MOE_STATIC_ALIAS_TRANSITION_PARAMS *params,
    NvBool *mutation_started,
    NvBool force_sequential,
    NvBool defer_cpu_restore)
{
    uvm_tracker_t clear_tracker = UVM_TRACKER_INIT();
    NV_STATUS first_error = NV_OK;
    NV_STATUS wait_status;
    NvU64 chunk_bytes = 0;
    NvU64 grouped_block_unmaps = 0;
    NvU64 fallback_block_unmaps = 0;
    NvU64 fallback_tracker_waits = 0;
    NvU64 stage_start;
    NvU32 group_index;
    NvU32 i;

    memset(legacy, 0, sizeof(*legacy));
    memset(state, 0, sizeof(*state));
    memset(clear_state, 0, sizeof(*clear_state));
    legacy->mode = UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE;
    uvm_uuid_copy(&legacy->gpu_uuid, &session->gpu_uuid);

    while (*record_cursor < session->alias_count &&
           legacy->count < UVM_MOE_STATIC_ALIAS_MAX_ENTRIES) {
        const uvm_moe_static_alias_record_t *record =
            &session->aliases[*record_cursor];
        uvm_moe_slot_pool_slot_t *slot;
        UVM_MOE_SLOT_REMAP_ENTRY *entry;

        ++*record_cursor;
        if (record->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW)
            continue;
        if (!record->active || record->pool_index >= UVM_MOE_SLOT_POOL_MAX_SLOTS)
            return NV_ERR_INVALID_STATE;

        slot = &g_uvm_moe_slot_pool[record->pool_index];
        entry = &legacy->entries[legacy->count];
        entry->layer_id = slot->layer_id;
        entry->expert_id = slot->owner_expert_id;
        entry->slot_id = slot->slot_id;
        entry->role = record->flags;
        entry->va_base = record->va_base;
        entry->bytes = record->bytes;
        entry->slot_offset = record->slot_offset;
        entry->flags = UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT;
        entry->status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
        state->slots[legacy->count] = slot;
        chunk_bytes += record->bytes;
        ++legacy->count;
    }

    if (legacy->count == 0)
        return NV_OK;
    legacy->requested_bytes = chunk_bytes;

    stage_start = NV_GETTIME();
    first_error = uvm_moe_static_alias_build_park_clear_state_locked(legacy,
                                                                      session,
                                                                      va_space,
                                                                      gpu,
                                                                      state,
                                                                      clear_state);
    params->build_sort_ns += NV_GETTIME() - stage_start;
    if (first_error != NV_OK)
        goto out;

    stage_start = NV_GETTIME();
    if (force_sequential) {
        first_error = NV_WARN_MORE_PROCESSING_REQUIRED;
    }
    else {
        uvm_moe_static_alias_classify_park_groups_locked(va_space,
                                                          gpu,
                                                          clear_state);
        first_error = NV_OK;
    }
    params->preflight_ns += NV_GETTIME() - stage_start;
    if (first_error == NV_WARN_MORE_PROCESSING_REQUIRED) {
        NvU64 sequential_block_unmaps = 0;
        NvU64 sequential_tracker_waits = 0;

        // Some sparse expert layouts share a VA block with ordinary UVM
        // mappings whose current PTE shape cannot be proven safe by the
        // grouped transaction.  The grouped preflight has not mutated any
        // PTEs, so fall back to the long-standing range-at-a-time uninstaller.
        // It performs its own VA-block locking, tracker wait, CPU-remote map,
        // and TLB synchronization for each complete recorded consumer range.
        // Keep the upload ranges and backing alive exactly as PARK requires.
        *mutation_started = NV_TRUE;
        stage_start = NV_GETTIME();
        for (i = 0; i < legacy->count; ++i) {
            UVM_MOE_SLOT_REMAP_ENTRY *entry = &legacy->entries[i];
            uvm_moe_slot_pool_slot_t *slot = state->slots[i];
            NvU64 range_block_unmaps = 0;
            NvU64 range_tracker_waits = 0;

            first_error = uvm_moe_slot_pool_uninstall_va_block_slot_mapping_mode(
                va_space,
                gpu,
                mm,
                entry->va_base,
                entry->bytes,
                &range_block_unmaps,
                &range_tracker_waits,
                defer_cpu_restore ? NV_FALSE : NV_TRUE);
            if (first_error != NV_OK)
                goto out;

            first_error = uvm_moe_static_alias_remove_slot_range_locked(
                slot,
                entry->va_base,
                entry->bytes);
            if (first_error != NV_OK)
                goto out;

            slot->pte_block_unmap_count += range_block_unmaps;
            slot->pte_block_tracker_wait_count += range_tracker_waits;
            ++slot->pte_clear_count;
            sequential_block_unmaps += range_block_unmaps;
            sequential_tracker_waits += range_tracker_waits;
            ++params->processed_alias_count;
            params->processed_alias_bytes += entry->bytes;
        }
        params->pte_map_ns += NV_GETTIME() - stage_start;

        stage_start = NV_GETTIME();
        params->tlb_invalidate_count = (NvU32)min_t(
            NvU64,
            (NvU64)params->tlb_invalidate_count + sequential_block_unmaps,
            (NvU64)NV_U32_MAX);
        params->reserved += force_sequential ? legacy->count : 1;
        g_uvm_moe_slot_pool_pte_clear_count += legacy->count;
        g_uvm_moe_slot_pool_pte_block_unmap_count += sequential_block_unmaps;
        g_uvm_moe_slot_pool_pte_block_tracker_wait_count +=
            sequential_tracker_waits;
        params->bookkeeping_ns += NV_GETTIME() - stage_start;

        uvm_phase_f_route_log("moe_static_alias_park_sequential session=%llu transactions=%u ranges=%u bytes=%llu block_unmaps=%llu tracker_waits=%llu forced=%u\n",
                              session->session_id,
                              params->reserved,
                              legacy->count,
                              chunk_bytes,
                              sequential_block_unmaps,
                              sequential_tracker_waits,
                              force_sequential);
        first_error = NV_OK;
        goto out;
    }
    if (first_error != NV_OK)
        goto out;

    // All allocation, range, overlap, and large-PTE checks completed without
    // mutation. Every group below performs slot->CPU-remote replacement and
    // contributes its block tracker to one completion boundary for the chunk.
    *mutation_started = NV_TRUE;
    stage_start = NV_GETTIME();
    for (group_index = 0; group_index < clear_state->group_count; ++group_index) {
        const uvm_moe_slot_transition_clear_group_t *group =
            &clear_state->groups[group_index];
        uvm_va_block_t *va_block = group->va_block;
        uvm_va_block_retry_t block_retry;
        NV_STATUS status;
        NV_STATUS tracker_status;

        if (!group->grouped_eligible) {
            NvU32 slice_offset;

            // Preserve the long-standing fail-closed operation for only the
            // exceptional block. The slices are the exact, non-overlapping
            // portions of committed alias records which intersect this block.
            // All other blocks in the transaction remain grouped.
            for (slice_offset = 0;
                 slice_offset < group->slice_count;
                 ++slice_offset) {
                const uvm_moe_slot_transition_clear_slice_t *slice =
                    &clear_state->slices[group->first_slice + slice_offset];
                NvU64 range_block_unmaps = 0;
                NvU64 range_tracker_waits = 0;

                status = uvm_moe_slot_pool_uninstall_va_block_slot_mapping_mode(
                    va_space,
                    gpu,
                    mm,
                    uvm_va_block_region_start(slice->va_block, slice->region),
                    uvm_va_block_region_size(slice->region),
                    &range_block_unmaps,
                    &range_tracker_waits,
                    defer_cpu_restore ? NV_FALSE : NV_TRUE);
                fallback_block_unmaps += range_block_unmaps;
                fallback_tracker_waits += range_tracker_waits;
                if (status != NV_OK) {
                    first_error = status;
                    break;
                }
            }
            if (first_error != NV_OK)
                break;
            continue;
        }

        status = UVM_VA_BLOCK_LOCK_RETRY(
            va_block,
            &block_retry,
            defer_cpu_restore ?
                uvm_va_block_unmap_moe_slot_no_cpu_restore(
                    va_block,
                    block_context,
                    gpu,
                    uvm_va_block_region_from_block(va_block),
                    &group->page_mask,
                    &va_block->tracker) :
                uvm_va_block_unmap_moe_slot(
                    va_block,
                    block_context,
                    gpu,
                    uvm_va_block_region_from_block(va_block),
                    &group->page_mask,
                    &va_block->tracker));

        // Snapshot even on failure: unmap or remote-map work may already have
        // been submitted before the helper returned an error.
        uvm_mutex_lock(&va_block->lock);
        tracker_status =
            uvm_tracker_add_tracker_safe(&clear_tracker, &va_block->tracker);
        uvm_mutex_unlock(&va_block->lock);
        if (status != NV_OK && first_error == NV_OK)
            first_error = status;
        if (tracker_status != NV_OK && first_error == NV_OK)
            first_error = tracker_status;
        if (first_error != NV_OK)
            break;
        ++grouped_block_unmaps;
    }
    params->pte_map_ns += NV_GETTIME() - stage_start;

    stage_start = NV_GETTIME();
    wait_status = uvm_tracker_wait(&clear_tracker);
    params->wait_ns += NV_GETTIME() - stage_start;
    if (first_error == NV_OK)
        first_error = wait_status;
    if (first_error != NV_OK)
        goto out;

    // Commit only consumer metadata after every PTE/TLB operation completed.
    // RW upload mappings and the slot/session/backing lifetime remain intact.
    stage_start = NV_GETTIME();
    for (i = 0; i < legacy->count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &legacy->entries[i];
        uvm_moe_slot_pool_slot_t *slot = state->slots[i];

        first_error = uvm_moe_static_alias_remove_slot_range_locked(slot,
                                                                     entry->va_base,
                                                                     entry->bytes);
        if (first_error != NV_OK)
            goto out;
    }
    params->metadata_ns += NV_GETTIME() - stage_start;

    stage_start = NV_GETTIME();
    for (i = 0; i < legacy->count; ++i) {
        uvm_moe_slot_pool_slot_t *slot = state->slots[i];
        NvBool first_slot_in_chunk = NV_TRUE;
        NvU32 j;

        slot->pte_block_unmap_count += clear_state->slot_block_counts[i];
        ++slot->pte_clear_count;
        for (j = 0; j < i; ++j) {
            if (state->slots[j] == slot) {
                first_slot_in_chunk = NV_FALSE;
                break;
            }
        }
        if (first_slot_in_chunk && clear_state->slot_block_counts[i] != 0)
            ++slot->pte_block_tracker_wait_count;
    }

    params->processed_alias_count += legacy->count;
    params->processed_alias_bytes += chunk_bytes;
    params->tlb_invalidate_count = (NvU32)min_t(
        NvU64,
        (NvU64)params->tlb_invalidate_count + grouped_block_unmaps +
            fallback_block_unmaps,
        (NvU64)NV_U32_MAX);
    ++params->reserved;
    g_uvm_moe_slot_pool_pte_clear_count += legacy->count;
    g_uvm_moe_slot_pool_pte_block_unmap_count += grouped_block_unmaps +
                                                 fallback_block_unmaps;
    g_uvm_moe_slot_pool_pte_block_tracker_wait_count +=
        fallback_tracker_waits;
    if (grouped_block_unmaps != 0)
        ++g_uvm_moe_slot_pool_pte_block_tracker_wait_count;
    params->bookkeeping_ns += NV_GETTIME() - stage_start;

    uvm_phase_f_route_log("moe_static_alias_park_batch session=%llu batch=%u ranges=%u bytes=%llu slices=%u unique_blocks=%u grouped_blocks=%llu fallback_unmaps=%llu fallback_waits=%llu\n",
                          session->session_id,
                          params->reserved,
                          legacy->count,
                          chunk_bytes,
                          clear_state->slice_count,
                          clear_state->unique_block_count,
                          grouped_block_unmaps,
                          fallback_block_unmaps,
                          fallback_tracker_waits);

out:
    uvm_tracker_deinit(&clear_tracker);
    return first_error;
}

static NV_STATUS uvm_api_moe_static_alias_park(
    UVM_MOE_STATIC_ALIAS_TRANSITION_PARAMS *params,
    struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_moe_static_alias_session_t *session;
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *legacy = NULL;
    uvm_moe_slot_direct_batch_state_t *state = NULL;
    uvm_moe_slot_transition_clear_state_t *clear_state = NULL;
    uvm_va_block_context_t *block_context = NULL;
    uvm_va_block_slot_alias_dual_result_t dual_result = {0};
    NvBool *pool_has_upload = NULL;
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    NV_STATUS status = NV_OK;
    NvU32 record_cursor = 0;
    NvBool mutation_started = NV_FALSE;
    NvU64 stage_start;
    NvU32 i;

    BUILD_BUG_ON(sizeof(*params) != 152);

    params->processed_alias_bytes = 0;
    params->processed_alias_count = 0;
    params->tlb_invalidate_count = 0;
    params->validation_ns = 0;
    params->build_sort_ns = 0;
    params->preflight_ns = 0;
    params->metadata_ns = 0;
    params->cpu_unmap_ns = 0;
    params->pte_map_ns = 0;
    params->wait_ns = 0;
    params->bookkeeping_ns = 0;
    uvm_moe_static_alias_transition_snapshot(NULL, params);

    if (params->abi_version != UVM_MOE_STATIC_ALIAS_ABI_VERSION ||
        params->struct_size != sizeof(*params) ||
        (params->flags & ~UVM_MOE_STATIC_ALIAS_TRANSITION_FLAGS_ALL) != 0 ||
        params->reserved != 0 ||
        params->session_id == 0) {
        return NV_ERR_INVALID_ARGUMENT;
    }
    params->reserved = 0;

    mutex_lock(&g_uvm_moe_slot_plan_lock);
    session = uvm_moe_static_alias_find_locked(va_space, params->session_id);
    if (!session) {
        status = NV_ERR_OBJECT_NOT_FOUND;
        goto out_plan;
    }
    uvm_moe_static_alias_transition_snapshot(session, params);
    if (!session->sealed || session->parked || session->poisoned ||
        params->consumer_alias_count == 0 || params->upload_alias_count == 0) {
        status = session->poisoned ? NV_ERR_INVALID_STATE : NV_ERR_IN_USE;
        goto out_plan;
    }
    // A deferred topology append deliberately has no current dual-view
    // certificate until userspace refreshes after its physical-layout
    // transaction. Full-coverage mode must not silently fall back to leaf
    // mutations or expose an obsolete frozen child table.
    if (uvm_moe_static_alias_dual_view_require_full &&
        !(
            (session->dual_view_full_coverage &&
             session->dual_view_generation == session->topology_generation) ||
            (session->dual_view_explicitly_discarded &&
             session->dual_view_block_count == 0)
        )) {
        status = NV_ERR_INVALID_STATE;
        goto out_plan;
    }

    legacy = uvm_kvmalloc_zero(sizeof(*legacy));
    state = uvm_kvmalloc_zero(sizeof(*state));
    clear_state = uvm_kvmalloc_zero(sizeof(*clear_state));
    pool_has_upload = uvm_kvmalloc_zero(
        UVM_MOE_SLOT_POOL_MAX_SLOTS * sizeof(*pool_has_upload));
    if (!legacy || !state || !clear_state || !pool_has_upload) {
        status = NV_ERR_NO_MEMORY;
        goto out_plan;
    }

    // PARK must leave at least one writable upload alias for every backing
    // slot which loses consumer mappings. Build the ownership certificate in
    // one linear pass instead of adding an O(alias_count^2) transition check.
    for (i = 0; i < session->alias_count; ++i) {
        const uvm_moe_static_alias_record_t *record = &session->aliases[i];

        if (!record->active || record->pool_index >= UVM_MOE_SLOT_POOL_MAX_SLOTS) {
            status = NV_ERR_INVALID_STATE;
            goto out_plan;
        }
        if (record->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW)
            pool_has_upload[record->pool_index] = NV_TRUE;
    }
    for (i = 0; i < session->alias_count; ++i) {
        const uvm_moe_static_alias_record_t *record = &session->aliases[i];

        if (!(record->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW) &&
            !pool_has_upload[record->pool_index]) {
            status = NV_ERR_INVALID_STATE;
            goto out_plan;
        }
    }

    gpu = uvm_va_space_retain_gpu_by_uuid(va_space, &session->gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto out_plan;
    }

    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    block_context = uvm_va_block_context_alloc(mm);
    if (!block_context) {
        status = NV_ERR_NO_MEMORY;
        goto out_mm;
    }
    uvm_va_space_down_write(va_space);
    mutex_lock(&g_uvm_moe_slot_pool_lock);

    stage_start = NV_GETTIME();
    status = uvm_moe_static_alias_validate_transition_locked(session,
                                                             va_space,
                                                             NV_FALSE);
    params->validation_ns += NV_GETTIME() - stage_start;
    if (status != NV_OK)
        goto out_pool;

    if (session->dual_view_block_count != 0) {
        NvU64 dual_start = NV_GETTIME();

        if ((params->flags &
             UVM_MOE_STATIC_ALIAS_TRANSITION_FLAG_FORCE_SEQUENTIAL) ||
            session->dual_view_generation != session->topology_generation) {
            status = NV_ERR_INVALID_STATE;
            goto out_pool;
        }
        status = uvm_va_block_slot_alias_dual_switch_batch(
            va_space,
            gpu,
            session->dual_view_block_starts,
            session->dual_view_block_count,
            true,
            &dual_result);
        params->pte_map_ns += NV_GETTIME() - dual_start;
        if (dual_result.pde_writes != 0)
            mutation_started = NV_TRUE;
        params->tlb_invalidate_count = (NvU32)min_t(
            NvU64,
            dual_result.tlb_batches,
            (NvU64)NV_U32_MAX);
        if (status != NV_OK)
            goto out_pool;
        uvm_phase_f_route_log(
            "moe_static_alias_dual_park session=%llu generation=%llu blocks=%u pdes=%llu tlb_batches=%llu ns=%llu\n",
            session->session_id,
            session->dual_view_generation,
            session->dual_view_block_count,
            dual_result.pde_writes,
            dual_result.tlb_batches,
            dual_result.elapsed_ns);
    }

    while (status == NV_OK && record_cursor < session->alias_count) {
        status = uvm_moe_static_alias_park_chunk_locked(session,
                                                        va_space,
                                                        gpu,
                                                        mm,
                                                        &record_cursor,
                                                        legacy,
                                                        state,
                                                        clear_state,
                                                        block_context,
                                                        params,
                                                        &mutation_started,
                                                        (params->flags &
                                                         UVM_MOE_STATIC_ALIAS_TRANSITION_FLAG_FORCE_SEQUENTIAL) != 0,
                                                        (params->flags &
                                                         UVM_MOE_STATIC_ALIAS_TRANSITION_FLAG_DEFER_CPU_RESTORE) != 0);
    }

    if (status == NV_OK &&
        params->processed_alias_count == params->consumer_alias_count &&
        params->processed_alias_bytes == params->consumer_alias_bytes) {
        session->parked = NV_TRUE;
        session->certificate_generation = session->topology_generation;
        session->certificate_valid = NV_TRUE;
        session->certificate_parked = NV_TRUE;
    }
    else {
        if (status == NV_OK)
            status = NV_ERR_INVALID_STATE;
        session->parked = NV_FALSE;
        if (mutation_started || params->processed_alias_count != 0) {
            session->certificate_valid = NV_FALSE;
            session->poisoned = NV_TRUE;
            for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
                if (g_uvm_moe_slot_pool[i].static_alias_session_id ==
                    session->session_id) {
                    g_uvm_moe_slot_pool[i].static_alias_poisoned = NV_TRUE;
                }
            }
        }
    }

out_pool:
    uvm_moe_static_alias_transition_snapshot(session, params);
    mutex_unlock(&g_uvm_moe_slot_pool_lock);
    uvm_va_space_up_write(va_space);
    uvm_va_block_context_free(block_context);
    block_context = NULL;

out_mm:
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

out_plan:
    if (gpu)
        uvm_gpu_release(gpu);
    mutex_unlock(&g_uvm_moe_slot_plan_lock);
    uvm_va_block_context_free(block_context);
    uvm_kvfree(pool_has_upload);
    uvm_kvfree(clear_state);
    uvm_kvfree(state);
    uvm_kvfree(legacy);
    return status;
}

#define UVM_MOE_STATIC_ALIAS_MAX_REBIND_CONSUMERS 65536U

static void uvm_moe_static_alias_rebind_reset_locked(
    uvm_moe_static_alias_session_t *session)
{
    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_plan_lock));

    if (!session)
        return;

    uvm_kvfree(session->rebind_aliases);
    session->rebind_aliases = NULL;
    session->rebind_alias_count = 0;
    session->rebind_expected_alias_count = 0;
    session->rebind_alias_bytes = 0;
    session->rebind_expected_alias_bytes = 0;
    session->rebind_id = 0;
    session->rebind_resize = NV_FALSE;
}

static NvU32 uvm_moe_static_alias_upload_count(
    const uvm_moe_static_alias_session_t *session,
    NvU64 *upload_bytes_out)
{
    NvU64 upload_bytes = 0;
    NvU32 upload_count = 0;
    NvU32 i;

    for (i = 0; session && i < session->alias_count; ++i) {
        const uvm_moe_static_alias_record_t *record = &session->aliases[i];

        if (!record->active ||
            !(record->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW)) {
            continue;
        }
        ++upload_count;
        upload_bytes += record->bytes;
    }

    if (upload_bytes_out)
        *upload_bytes_out = upload_bytes;
    return upload_count;
}

static bool uvm_moe_static_alias_has_matching_upload(
    const uvm_moe_static_alias_session_t *session,
    NvU32 pool_index,
    const UVM_MOE_STATIC_ALIAS_ENTRY *entry)
{
    NvU32 i;

    for (i = 0; i < session->alias_count; ++i) {
        const uvm_moe_static_alias_record_t *record = &session->aliases[i];

        if (!record->active || record->pool_index != pool_index ||
            !(record->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW)) {
            continue;
        }
        if (record->bytes == entry->bytes &&
            record->slot_offset == entry->slot_offset &&
            ((record->flags ^ entry->flags) &
             UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K) == 0) {
            return true;
        }
    }

    return false;
}

static int uvm_moe_static_alias_record_va_cmp(const void *a, const void *b)
{
    const uvm_moe_static_alias_record_t *left = a;
    const uvm_moe_static_alias_record_t *right = b;

    if (left->va_base < right->va_base)
        return -1;
    if (left->va_base > right->va_base)
        return 1;
    if (left->bytes < right->bytes)
        return -1;
    if (left->bytes > right->bytes)
        return 1;
    return 0;
}

static void uvm_moe_static_alias_rebind_fail_entries(
    UVM_MOE_STATIC_ALIAS_REBIND_PARAMS *params,
    NvU32 status)
{
    NvU32 i;

    params->accepted_count = 0;
    for (i = 0; i < params->count; ++i) {
        if (params->entries[i].status == UVM_MOE_STATIC_ALIAS_ENTRY_OK)
            params->entries[i].status = status;
    }
}

static NvBool uvm_moe_static_alias_resize_has_old_upload(
    const uvm_moe_static_alias_session_t *session,
    const uvm_moe_static_alias_record_t *candidate)
{
    NvU32 i;

    for (i = 0; i < session->alias_count; ++i) {
        const uvm_moe_static_alias_record_t *record = &session->aliases[i];

        if (!record->active ||
            !(record->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW) ||
            record->pool_index != candidate->pool_index) {
            continue;
        }
        if (record->va_base == candidate->va_base &&
            record->bytes == candidate->bytes &&
            record->slot_offset == candidate->slot_offset &&
            ((record->flags ^ candidate->flags) &
             UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K) == 0) {
            return NV_TRUE;
        }
    }

    return NV_FALSE;
}

/*
 * Commit a complete parked topology while retaining every common pool slot.
 * Newly requested pool slots must already have mode-8 VIDMEM backing; this
 * transaction adopts them into the static session and maps only their upload
 * aliases. Consumer aliases remain ordinary CPU-remote mappings until RESUME.
 * Slots absent from the candidate are released only after all additions have
 * been installed successfully.
 */
static NV_STATUS uvm_moe_static_alias_resize_commit_locked(
    uvm_moe_static_alias_session_t *session,
    uvm_va_space_t *va_space,
    uvm_gpu_t *gpu,
    struct mm_struct *mm,
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *legacy,
    uvm_moe_slot_direct_batch_state_t *state,
    UVM_MOE_STATIC_ALIAS_REBIND_PARAMS *params,
    NvBool *mutation_started)
{
    NvBool *old_pool = NULL;
    NvBool *target_pool = NULL;
    NvBool *target_has_upload = NULL;
    NvU32 *target_range_count = NULL;
    NvU32 *target_upload_index = NULL;
    uvm_moe_static_alias_record_t *old_aliases = NULL;
    uvm_tracker_t batch_tracker = UVM_TRACKER_INIT();
    uvm_va_block_slot_direct_result_t map_result = {0};
    NV_STATUS status = NV_OK;
    NV_STATUS tracker_status = NV_OK;
    NvU64 target_backing_bytes = 0;
    NvU64 added_backing_bytes = 0;
    NvU64 removed_backing_bytes = 0;
    NvU32 target_upload_count = 0;
    NvU32 added_upload_count = 0;
    NvU32 removed_pool_count = 0;
    NvU32 i;
    NvBool tracker_initialized = NV_FALSE;

    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_plan_lock));
    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_pool_lock));
    uvm_assert_rwsem_locked_write(&va_space->lock);

    old_pool = uvm_kvmalloc_zero(
        UVM_MOE_SLOT_POOL_MAX_SLOTS * sizeof(*old_pool));
    target_pool = uvm_kvmalloc_zero(
        UVM_MOE_SLOT_POOL_MAX_SLOTS * sizeof(*target_pool));
    target_has_upload = uvm_kvmalloc_zero(
        UVM_MOE_SLOT_POOL_MAX_SLOTS * sizeof(*target_has_upload));
    target_range_count = uvm_kvmalloc_zero(
        UVM_MOE_SLOT_POOL_MAX_SLOTS * sizeof(*target_range_count));
    target_upload_index = uvm_kvmalloc_zero(
        UVM_MOE_SLOT_POOL_MAX_SLOTS * sizeof(*target_upload_index));
    if (!old_pool || !target_pool || !target_has_upload ||
        !target_range_count || !target_upload_index) {
        status = NV_ERR_NO_MEMORY;
        goto out;
    }
    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i)
        target_upload_index[i] = NV_U32_MAX;

    for (i = 0; i < session->alias_count; ++i) {
        const uvm_moe_static_alias_record_t *record = &session->aliases[i];

        if (!record->active || record->pool_index >= UVM_MOE_SLOT_POOL_MAX_SLOTS) {
            status = NV_ERR_INVALID_STATE;
            goto out;
        }
        old_pool[record->pool_index] = NV_TRUE;
    }

    for (i = 0; i < session->rebind_alias_count; ++i) {
        const uvm_moe_static_alias_record_t *record = &session->rebind_aliases[i];
        uvm_moe_slot_pool_slot_t *slot;
        NvU32 pool_index = record->pool_index;

        if (!record->active || pool_index >= UVM_MOE_SLOT_POOL_MAX_SLOTS) {
            status = NV_ERR_INVALID_STATE;
            goto out;
        }
        slot = &g_uvm_moe_slot_pool[pool_index];
        if (!slot->valid ||
            slot->owner_va_space != va_space ||
            slot->layer_id != record->backing_layer_id ||
            slot->owner_expert_id != record->backing_expert_id ||
            slot->slot_id != record->backing_slot_id ||
            !uvm_moe_slot_pool_slot_has_vidmem_backing(slot, gpu) ||
            !(slot->flags & UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT) ||
            slot->async_copy_pending || slot->async_copy_staging ||
            slot->static_alias_poisoned ||
            record->slot_offset + record->bytes > slot->slot_bytes) {
            status = NV_ERR_INVALID_STATE;
            goto out;
        }
        if (old_pool[pool_index]) {
            if (slot->static_alias_session_id != session->session_id ||
                !slot->static_alias_sealed ||
                slot->pte_va_space != va_space || slot->pte_gpu != gpu ||
                slot->pte_mode != UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP) {
                status = NV_ERR_INVALID_STATE;
                goto out;
            }
        }
        else if (slot->static_alias_session_id != 0 ||
                 !uvm_moe_slot_direct_pte_metadata_empty(slot) ||
                 slot->state != UVM_MOE_SLOT_POOL_STATE_READY) {
            status = NV_ERR_IN_USE;
            goto out;
        }

        target_pool[pool_index] = NV_TRUE;
        if (++target_range_count[pool_index] >
            UVM_MOE_SLOT_POOL_MAX_PTE_RANGES_PER_SLOT) {
            status = NV_ERR_INSUFFICIENT_RESOURCES;
            goto out;
        }
        if (record->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW) {
            if (target_has_upload[pool_index]) {
                status = NV_ERR_INVALID_ARGUMENT;
                goto out;
            }
            target_has_upload[pool_index] = NV_TRUE;
            target_upload_index[pool_index] = i;
            ++target_upload_count;
            target_backing_bytes += slot->slot_bytes;
            if (old_pool[pool_index]) {
                if (!uvm_moe_static_alias_resize_has_old_upload(session, record)) {
                    status = NV_ERR_INVALID_ARGUMENT;
                    goto out;
                }
            }
            else {
                ++added_upload_count;
                added_backing_bytes += slot->slot_bytes;
            }
        }
    }

    if (target_upload_count == 0 ||
        added_upload_count > UVM_MOE_STATIC_ALIAS_MAX_ENTRIES) {
        status = target_upload_count == 0 ?
            NV_ERR_INVALID_ARGUMENT : NV_ERR_INSUFFICIENT_RESOURCES;
        goto out;
    }
    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        if (target_pool[i] && !target_has_upload[i]) {
            status = NV_ERR_INVALID_ARGUMENT;
            goto out;
        }
    }
    for (i = 0; i < session->rebind_alias_count; ++i) {
        const uvm_moe_static_alias_record_t *record = &session->rebind_aliases[i];
        const uvm_moe_static_alias_record_t *upload;

        if (record->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW)
            continue;
        upload = &session->rebind_aliases[
            target_upload_index[record->pool_index]];
        if (upload->bytes != record->bytes ||
            upload->slot_offset != record->slot_offset ||
            ((upload->flags ^ record->flags) &
             UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K) != 0) {
            status = NV_ERR_INVALID_ARGUMENT;
            goto out;
        }
    }

    memset(legacy, 0, sizeof(*legacy));
    memset(state, 0, sizeof(*state));
    legacy->mode = UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE;
    uvm_uuid_copy(&legacy->gpu_uuid, &session->gpu_uuid);
    for (i = 0; i < session->rebind_alias_count; ++i) {
        const uvm_moe_static_alias_record_t *record = &session->rebind_aliases[i];
        UVM_MOE_SLOT_REMAP_ENTRY *entry;

        if (!(record->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW) ||
            old_pool[record->pool_index]) {
            continue;
        }
        entry = &legacy->entries[legacy->count];
        entry->layer_id = record->backing_layer_id;
        entry->expert_id = record->backing_expert_id;
        entry->slot_id = record->backing_slot_id;
        entry->role = record->flags;
        entry->va_base = record->va_base;
        entry->bytes = record->bytes;
        entry->slot_offset = record->slot_offset;
        entry->flags = UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT;
        entry->status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
        state->slots[legacy->count] =
            &g_uvm_moe_slot_pool[record->pool_index];
        legacy->requested_bytes += record->bytes;
        ++legacy->count;
    }

    if (legacy->count != added_upload_count) {
        status = NV_ERR_INVALID_STATE;
        goto out;
    }
    if (legacy->count != 0) {
        status = uvm_moe_slot_direct_flatten_blocks_locked(legacy,
                                                            va_space,
                                                            state,
                                                            NV_TRUE);
        if (status != NV_OK)
            goto out;
        for (i = 0; i < state->block_count; ++i) {
            NvU32 entry_index = state->block_entry_indices[i];
            NvU32 flags = legacy->entries[entry_index].role;

            state->blocks[i].pte_mode =
                (flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K) ?
                    UVM_SLOT_DIRECT_PTE_MODE_64K : UVM_SLOT_DIRECT_PTE_MODE_4K;
            state->blocks[i].writable = true;
            state->blocks[i].slot_tracker = NULL;
        }
        status = uvm_va_block_slot_direct_preflight_targets_not_slot_mapped(
            va_space, gpu, state->blocks, state->block_count);
        if (status != NV_OK)
            goto out;
    }

    // A parked dual-view session still exposes the detached CPU children, but
    // its hidden GPU children describe the old tier. Normalize those blocks
    // back to ordinary CPU mappings before any backing can be added or freed.
    // The target will be prepared again after its first legacy RESUME.
    if (session->dual_view_block_count != 0)
        *mutation_started = NV_TRUE;
    status = uvm_moe_static_alias_dual_discard_locked(session,
                                                       va_space,
                                                       mm,
                                                       gpu);
    if (status != NV_OK)
        goto poison;

    if (legacy->count != 0) {
        status = uvm_moe_slot_direct_record_all_pte_metadata_locked(legacy,
                                                                    va_space,
                                                                    gpu,
                                                                    state);
        if (status != NV_OK)
            goto out;

        /* Publish ownership before revoking CPU mappings: the fault guard must
         * see every range as soon as the first PTE can change. */
        for (i = 0; i < legacy->count; ++i) {
            state->slots[i]->static_alias_session_id = session->session_id;
            state->slots[i]->static_alias_sealed = NV_TRUE;
            state->slots[i]->static_alias_poisoned = NV_FALSE;
        }
        *mutation_started = NV_TRUE;
        status = uvm_va_block_slot_alias_revoke_cpu_mappings_batch(
            va_space, mm, state->blocks, state->block_count);
        if (status != NV_OK)
            goto poison;

        uvm_tracker_init(&batch_tracker);
        tracker_initialized = NV_TRUE;
        status = uvm_va_block_slot_alias_map_batch(
            va_space,
            mm,
            gpu,
            state->blocks,
            state->block_count,
            true,
            true,
            &g_uvm_moe_slot_pool_lock,
            &batch_tracker,
            &map_result);
        for (i = 0; i < state->block_count; ++i) {
            uvm_va_block_t *block = state->blocks[i].va_block;
            NV_STATUS add_status;

            if (i > 0 && state->blocks[i - 1].va_block == block)
                continue;
            uvm_mutex_lock(&block->lock);
            add_status =
                uvm_tracker_add_tracker_safe(&batch_tracker, &block->tracker);
            uvm_mutex_unlock(&block->lock);
            if (tracker_status == NV_OK)
                tracker_status = add_status;
        }
        {
            NV_STATUS wait_status = uvm_tracker_wait(&batch_tracker);

            if (tracker_status == NV_OK)
                tracker_status = wait_status;
        }
        if (status == NV_OK && tracker_status != NV_OK)
            status = tracker_status;
        if (status != NV_OK)
            goto poison;
    }

    /* Additions are now durable. Release only slots absent from the target;
     * common backing and its writable upload PTE never move. */
    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot;
        NvU32 record_index;

        if (!old_pool[i] || target_pool[i])
            continue;
        slot = &g_uvm_moe_slot_pool[i];
        if (slot->static_alias_session_id != session->session_id) {
            status = NV_ERR_INVALID_STATE;
            goto poison;
        }
        *mutation_started = NV_TRUE;
        removed_backing_bytes += slot->slot_bytes;
        status = uvm_moe_slot_pool_release_backing_locked(slot, va_space, mm);
        if (status != NV_OK)
            goto poison;
        for (record_index = 0; record_index < session->alias_count;
             ++record_index) {
            if (session->aliases[record_index].active &&
                session->aliases[record_index].pool_index == i) {
                session->aliases[record_index].active = NV_FALSE;
            }
        }
        memset(slot, 0, sizeof(*slot));
        ++removed_pool_count;
    }

    old_aliases = session->aliases;
    session->aliases = session->rebind_aliases;
    session->alias_count = session->rebind_alias_count;
    session->alias_bytes = session->rebind_alias_bytes;
    session->backing_bytes = target_backing_bytes;
    session->rebind_aliases = NULL;
    session->rebind_alias_count = 0;
    session->rebind_alias_bytes = 0;
    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot;

        if (!target_pool[i])
            continue;
        slot = &g_uvm_moe_slot_pool[i];
        slot->static_alias_session_id = session->session_id;
        slot->static_alias_sealed = NV_TRUE;
        slot->static_alias_poisoned = NV_FALSE;
    }
    status = uvm_moe_static_alias_refresh_certificate_locked(session);
    if (status == NV_OK)
        status = uvm_moe_static_alias_validate_transition_locked(session,
                                                                 va_space,
                                                                 NV_TRUE);
    if (status != NV_OK)
        goto poison;
    uvm_kvfree(old_aliases);
    old_aliases = NULL;
    params->backing_bytes = session->backing_bytes;
    params->total_alias_bytes = session->alias_bytes;
    params->upload_alias_count = target_upload_count;
    params->committed = 1;
    uvm_phase_f_route_log(
        "moe_static_alias_resize session=%llu retained_backing=%llu added_backing=%llu removed_backing=%llu added_uploads=%u removed_pools=%u target_aliases=%u\n",
        session->session_id,
        session->backing_bytes - added_backing_bytes,
        added_backing_bytes,
        removed_backing_bytes,
        added_upload_count,
        removed_pool_count,
        session->alias_count);
    goto out;

poison:
    session->certificate_valid = NV_FALSE;
    session->poisoned = NV_TRUE;
    params->poisoned = 1;
    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        if (g_uvm_moe_slot_pool[i].static_alias_session_id ==
            session->session_id) {
            g_uvm_moe_slot_pool[i].static_alias_poisoned = NV_TRUE;
        }
    }

out:
    if (tracker_initialized)
        uvm_tracker_deinit(&batch_tracker);
    uvm_kvfree(old_aliases);
    uvm_kvfree(target_upload_index);
    uvm_kvfree(target_range_count);
    uvm_kvfree(target_has_upload);
    uvm_kvfree(target_pool);
    uvm_kvfree(old_pool);
    return status;
}

static NV_STATUS uvm_api_moe_static_alias_rebind(
    UVM_MOE_STATIC_ALIAS_REBIND_PARAMS *params,
    struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_moe_static_alias_session_t *session = NULL;
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *legacy = NULL;
    uvm_moe_slot_direct_batch_state_t *state = NULL;
    uvm_moe_static_alias_record_t *replacement_aliases = NULL;
    NvU32 *pending_per_pool = NULL;
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    NV_STATUS status = NV_OK;
    NvU64 call_bytes = 0;
    NvU64 upload_bytes = 0;
    NvU32 upload_count = 0;
    NvU32 i;
    bool plan_locked = false;
    bool mm_locked = false;
    bool va_locked = false;
    bool pool_locked = false;
    bool transaction_started = false;
    bool resize_requested = false;
    NvBool resize_mutation_started = NV_FALSE;

    BUILD_BUG_ON(sizeof(*params) != 221288);
    BUILD_BUG_ON(offsetof(UVM_MOE_STATIC_ALIAS_REBIND_PARAMS, entries) != 96);
    BUILD_BUG_ON(offsetof(UVM_MOE_STATIC_ALIAS_REBIND_PARAMS, rmStatus) != 221280);

    params->accepted_count = 0;
    params->staged_consumer_alias_bytes = 0;
    params->backing_bytes = 0;
    params->total_alias_bytes = 0;
    params->staged_consumer_alias_count = 0;
    params->upload_alias_count = 0;
    params->committed = 0;
    params->poisoned = 0;

    if (params->abi_version != UVM_MOE_STATIC_ALIAS_ABI_VERSION ||
        params->struct_size != sizeof(*params) ||
        (params->flags & ~UVM_MOE_STATIC_ALIAS_REBIND_FLAGS_ALL) ||
        params->reserved != 0 || params->reserved2 != 0 ||
        params->session_id == 0) {
        return NV_ERR_INVALID_ARGUMENT;
    }

    resize_requested =
        (params->flags & UVM_MOE_STATIC_ALIAS_REBIND_FLAG_RESIZE) != 0;
    if (params->flags & UVM_MOE_STATIC_ALIAS_REBIND_FLAG_ABORT) {
        if (params->flags != UVM_MOE_STATIC_ALIAS_REBIND_FLAG_ABORT ||
            params->count != 0 || params->rebind_id == 0) {
            return NV_ERR_INVALID_ARGUMENT;
        }
    }
    else if (params->count == 0 ||
             params->count > UVM_MOE_STATIC_ALIAS_MAX_ENTRIES ||
             params->expected_consumer_alias_count == 0 ||
             params->expected_consumer_alias_count >
                 UVM_MOE_STATIC_ALIAS_MAX_REBIND_CONSUMERS ||
             params->expected_consumer_alias_bytes == 0 ||
             ((params->flags & UVM_MOE_STATIC_ALIAS_REBIND_FLAG_BEGIN) != 0) !=
                 (params->rebind_id == 0)) {
        return NV_ERR_INVALID_ARGUMENT;
    }

    for (i = 0; i < params->count; ++i) {
        UVM_MOE_STATIC_ALIAS_ENTRY *entry = &params->entries[i];
        NvU32 j;

        entry->status = UVM_MOE_STATIC_ALIAS_ENTRY_OK;
        if (uvm_moe_static_alias_range_invalid(entry) ||
            (!resize_requested &&
             (entry->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW)) ||
            call_bytes + entry->bytes < call_bytes) {
            entry->status = UVM_MOE_STATIC_ALIAS_ENTRY_INVALID_ARGUMENT;
            status = NV_ERR_INVALID_ARGUMENT;
            continue;
        }
        call_bytes += entry->bytes;
        for (j = 0; j < i; ++j) {
            const UVM_MOE_STATIC_ALIAS_ENTRY *previous = &params->entries[j];

            if (previous->status != UVM_MOE_STATIC_ALIAS_ENTRY_OK)
                continue;
            if (uvm_moe_static_alias_ranges_overlap(entry->va_base,
                                                     entry->bytes,
                                                     previous->va_base,
                                                     previous->bytes)) {
                entry->status = UVM_MOE_STATIC_ALIAS_ENTRY_ALIAS_CONFLICT;
                status = NV_ERR_INVALID_ARGUMENT;
                break;
            }
        }
    }
    if (status != NV_OK) {
        uvm_moe_static_alias_rebind_fail_entries(
            params, UVM_MOE_STATIC_ALIAS_ENTRY_ROLLED_BACK);
        return status;
    }

    if (!(params->flags & UVM_MOE_STATIC_ALIAS_REBIND_FLAG_ABORT)) {
        legacy = uvm_kvmalloc_zero(sizeof(*legacy));
        state = uvm_kvmalloc_zero(sizeof(*state));
        pending_per_pool = uvm_kvmalloc_zero(
            UVM_MOE_SLOT_POOL_MAX_SLOTS * sizeof(*pending_per_pool));
        if (!legacy || !state || !pending_per_pool) {
            status = NV_ERR_NO_MEMORY;
            goto out;
        }
    }

    mutex_lock(&g_uvm_moe_slot_plan_lock);
    plan_locked = true;
    session = uvm_moe_static_alias_find_locked(va_space, params->session_id);
    if (!session) {
        status = NV_ERR_OBJECT_NOT_FOUND;
        goto out;
    }
    upload_count = uvm_moe_static_alias_upload_count(session, &upload_bytes);
    params->upload_alias_count = upload_count;
    params->backing_bytes = session->backing_bytes;
    params->total_alias_bytes = session->alias_bytes;
    params->poisoned = session->poisoned;

    if (!session->sealed || !session->parked || session->poisoned ||
        upload_count == 0) {
        status = session->poisoned ? NV_ERR_INVALID_STATE : NV_ERR_IN_USE;
        goto out;
    }

    if (params->flags & UVM_MOE_STATIC_ALIAS_REBIND_FLAG_ABORT) {
        if (session->rebind_id != params->rebind_id) {
            status = NV_ERR_OBJECT_NOT_FOUND;
            goto out;
        }
        uvm_moe_static_alias_rebind_reset_locked(session);
        params->rebind_id = 0;
        goto out;
    }

    if (params->flags & UVM_MOE_STATIC_ALIAS_REBIND_FLAG_BEGIN) {
        if (session->rebind_id != 0 || session->rebind_aliases != NULL) {
            status = NV_ERR_IN_USE;
            goto out;
        }
        session->rebind_aliases = uvm_kvmalloc_zero(
            (size_t)params->expected_consumer_alias_count *
            sizeof(*session->rebind_aliases));
        if (!session->rebind_aliases) {
            status = NV_ERR_NO_MEMORY;
            goto out;
        }
        ++g_uvm_moe_static_alias_rebind_counter;
        if (g_uvm_moe_static_alias_rebind_counter == 0)
            ++g_uvm_moe_static_alias_rebind_counter;
        session->rebind_id = g_uvm_moe_static_alias_rebind_counter;
        session->rebind_expected_alias_count =
            params->expected_consumer_alias_count;
        session->rebind_expected_alias_bytes =
            params->expected_consumer_alias_bytes;
        session->rebind_resize = resize_requested ? NV_TRUE : NV_FALSE;
        params->rebind_id = session->rebind_id;
        transaction_started = true;
    }
    else if (session->rebind_id == 0 ||
             session->rebind_id != params->rebind_id ||
             session->rebind_expected_alias_count !=
                 params->expected_consumer_alias_count ||
             session->rebind_expected_alias_bytes !=
                 params->expected_consumer_alias_bytes ||
             session->rebind_resize !=
                 (resize_requested ? NV_TRUE : NV_FALSE)) {
        status = NV_ERR_INVALID_STATE;
        goto out;
    }

    if (session->rebind_alias_count > session->rebind_expected_alias_count ||
        params->count > session->rebind_expected_alias_count -
                            session->rebind_alias_count ||
        session->rebind_alias_bytes > session->rebind_expected_alias_bytes ||
        call_bytes > session->rebind_expected_alias_bytes -
                         session->rebind_alias_bytes) {
        status = NV_ERR_OUT_OF_RANGE;
        goto abort_transaction;
    }

    legacy->count = params->count;
    legacy->mode = UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE;
    uvm_uuid_copy(&legacy->gpu_uuid, &session->gpu_uuid);
    gpu = uvm_va_space_retain_gpu_by_uuid(va_space, &session->gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto abort_transaction;
    }
    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    mm_locked = true;
    uvm_va_space_down_write(va_space);
    va_locked = true;
    mutex_lock(&g_uvm_moe_slot_pool_lock);
    pool_locked = true;

    status = uvm_moe_static_alias_validate_transition_locked(session,
                                                             va_space,
                                                             NV_TRUE);
    if (status != NV_OK)
        goto abort_transaction;

    for (i = 0; i < params->count; ++i) {
        UVM_MOE_STATIC_ALIAS_ENTRY *entry = &params->entries[i];
        UVM_MOE_SLOT_REMAP_ENTRY *legacy_entry = &legacy->entries[i];
        uvm_moe_slot_pool_slot_t *slot;
        NvU32 pool_index;
        NvU32 j;

        slot = uvm_moe_slot_pool_find_locked(va_space,
                                             entry->backing_layer_id,
                                             entry->backing_slot_id);
        if (!slot || slot->owner_expert_id != entry->backing_expert_id) {
            entry->status = UVM_MOE_STATIC_ALIAS_ENTRY_BACKING_NOT_FOUND;
            status = NV_ERR_OBJECT_NOT_FOUND;
            break;
        }
        pool_index = (NvU32)(slot - g_uvm_moe_slot_pool);
        if (!uvm_moe_slot_pool_slot_has_vidmem_backing(slot, gpu) ||
            slot->owner_va_space != va_space ||
            slot->static_alias_poisoned ||
            slot->async_copy_pending || slot->async_copy_staging ||
            entry->slot_offset + entry->bytes > slot->slot_bytes ||
            (!resize_requested &&
             (slot->static_alias_session_id != session->session_id ||
              !slot->static_alias_sealed ||
              !uvm_moe_static_alias_has_matching_upload(session,
                                                        pool_index,
                                                        entry))) ||
            (resize_requested &&
             slot->static_alias_session_id != 0 &&
             slot->static_alias_session_id != session->session_id)) {
            entry->status = UVM_MOE_STATIC_ALIAS_ENTRY_ALIAS_CONFLICT;
            status = NV_ERR_INVALID_STATE;
            break;
        }

        for (j = 0; j < session->rebind_alias_count; ++j) {
            const uvm_moe_static_alias_record_t *pending =
                &session->rebind_aliases[j];

            if (uvm_moe_static_alias_ranges_overlap(entry->va_base,
                                                     entry->bytes,
                                                     pending->va_base,
                                                     pending->bytes)) {
                entry->status = UVM_MOE_STATIC_ALIAS_ENTRY_ALIAS_CONFLICT;
                status = NV_ERR_INVALID_ARGUMENT;
                break;
            }
        }
        if (status != NV_OK)
            break;
        for (j = 0; !resize_requested && j < session->alias_count; ++j) {
            const uvm_moe_static_alias_record_t *active = &session->aliases[j];

            if (!active->active ||
                !(active->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW)) {
                continue;
            }
            if (uvm_moe_static_alias_ranges_overlap(entry->va_base,
                                                     entry->bytes,
                                                     active->va_base,
                                                     active->bytes)) {
                entry->status = UVM_MOE_STATIC_ALIAS_ENTRY_ALIAS_CONFLICT;
                status = NV_ERR_INVALID_ARGUMENT;
                break;
            }
        }
        if (status != NV_OK)
            break;

        if (!resize_requested) {
            legacy_entry->layer_id = entry->backing_layer_id;
            legacy_entry->expert_id = entry->backing_expert_id;
            legacy_entry->slot_id = entry->backing_slot_id;
            legacy_entry->role = entry->flags;
            legacy_entry->va_base = entry->va_base;
            legacy_entry->bytes = entry->bytes;
            legacy_entry->slot_offset = entry->slot_offset;
            legacy_entry->flags = UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT;
            legacy_entry->status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
        }
        state->slots[i] = slot;
    }
    if (status != NV_OK)
        goto abort_transaction;

    if (!resize_requested) {
        status = uvm_moe_slot_direct_flatten_blocks_locked(legacy,
                                                            va_space,
                                                            state,
                                                            NV_TRUE);
        if (status != NV_OK)
            goto abort_transaction;
        for (i = 0; i < state->block_count; ++i) {
            NvU32 entry_index = state->block_entry_indices[i];

            state->blocks[i].pte_mode =
                (params->entries[entry_index].flags &
                 UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K) ?
                    UVM_SLOT_DIRECT_PTE_MODE_64K : UVM_SLOT_DIRECT_PTE_MODE_4K;
            state->blocks[i].writable = false;
            state->blocks[i].slot_tracker = NULL;
        }
        status = uvm_va_block_slot_direct_preflight_targets_not_slot_mapped(
            va_space, gpu, state->blocks, state->block_count);
        if (status != NV_OK)
            goto abort_transaction;
    }

    for (i = 0; i < params->count; ++i) {
        const UVM_MOE_STATIC_ALIAS_ENTRY *entry = &params->entries[i];
        uvm_moe_static_alias_record_t *pending =
            &session->rebind_aliases[session->rebind_alias_count + i];

        pending->va_base = entry->va_base;
        pending->bytes = entry->bytes;
        pending->slot_offset = entry->slot_offset;
        pending->backing_layer_id = entry->backing_layer_id;
        pending->backing_expert_id = entry->backing_expert_id;
        pending->backing_slot_id = entry->backing_slot_id;
        pending->pool_index =
            (NvU32)(state->slots[i] - g_uvm_moe_slot_pool);
        pending->flags = entry->flags;
        pending->active = NV_TRUE;
    }
    session->rebind_alias_count += params->count;
    session->rebind_alias_bytes += call_bytes;
    params->accepted_count = params->count;

    if (params->flags & UVM_MOE_STATIC_ALIAS_REBIND_FLAG_COMMIT) {
        uvm_moe_static_alias_record_t *old_aliases;
        NvU32 old_alias_count;
        NvU64 old_alias_bytes;
        NvU32 replacement_index = 0;

        if (session->rebind_alias_count !=
                session->rebind_expected_alias_count ||
            session->rebind_alias_bytes !=
                session->rebind_expected_alias_bytes) {
            status = NV_ERR_INVALID_ARGUMENT;
            goto abort_transaction;
        }

        sort(session->rebind_aliases,
             session->rebind_alias_count,
             sizeof(session->rebind_aliases[0]),
             uvm_moe_static_alias_record_va_cmp,
             NULL);
        for (i = 1; i < session->rebind_alias_count; ++i) {
            const uvm_moe_static_alias_record_t *previous =
                &session->rebind_aliases[i - 1];
            const uvm_moe_static_alias_record_t *candidate =
                &session->rebind_aliases[i];

            if (uvm_moe_static_alias_ranges_overlap(previous->va_base,
                                                     previous->bytes,
                                                     candidate->va_base,
                                                     candidate->bytes)) {
                status = NV_ERR_INVALID_ARGUMENT;
                goto abort_transaction;
            }
        }
        if (resize_requested) {
            status = uvm_moe_static_alias_resize_commit_locked(
                session,
                va_space,
                gpu,
                mm,
                legacy,
                state,
                params,
                &resize_mutation_started);
            if (status != NV_OK)
                goto abort_transaction;
            uvm_moe_static_alias_rebind_reset_locked(session);
            params->rebind_id = 0;
            goto out;
        }
        for (i = 0; i < session->rebind_alias_count; ++i) {
            NvU32 pool_index = session->rebind_aliases[i].pool_index;
            uvm_moe_slot_pool_slot_t *slot;

            if (pool_index >= UVM_MOE_SLOT_POOL_MAX_SLOTS) {
                status = NV_ERR_INVALID_STATE;
                goto abort_transaction;
            }
            slot = &g_uvm_moe_slot_pool[pool_index];
            ++pending_per_pool[pool_index];
            if (slot->pte_range_count + pending_per_pool[pool_index] >
                UVM_MOE_SLOT_POOL_MAX_PTE_RANGES_PER_SLOT) {
                status = NV_ERR_INSUFFICIENT_RESOURCES;
                goto abort_transaction;
            }
        }

        replacement_aliases = uvm_kvmalloc_zero(
            (size_t)(upload_count + session->rebind_alias_count) *
            sizeof(*replacement_aliases));
        if (!replacement_aliases) {
            status = NV_ERR_NO_MEMORY;
            goto abort_transaction;
        }
        for (i = 0; i < session->alias_count; ++i) {
            const uvm_moe_static_alias_record_t *record = &session->aliases[i];

            if (record->active &&
                (record->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW)) {
                replacement_aliases[replacement_index++] = *record;
            }
        }
        if (replacement_index != upload_count) {
            status = NV_ERR_INVALID_STATE;
            goto abort_transaction;
        }
        memcpy(&replacement_aliases[replacement_index],
               session->rebind_aliases,
               (size_t)session->rebind_alias_count *
                   sizeof(*replacement_aliases));
        sort(replacement_aliases,
             upload_count + session->rebind_alias_count,
             sizeof(replacement_aliases[0]),
             uvm_moe_static_alias_record_va_cmp,
             NULL);

        // Non-resizing REBIND also replaces the complete parked consumer
        // directory. Drop the old dual-view recipe before publishing the new
        // topology so RESUME cannot switch a stale hidden GPU child back in.
        status = uvm_moe_static_alias_dual_discard_locked(session,
                                                           va_space,
                                                           mm,
                                                           gpu);
        if (status != NV_OK) {
            session->certificate_valid = NV_FALSE;
            session->poisoned = NV_TRUE;
            params->poisoned = 1;
            for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
                if (g_uvm_moe_slot_pool[i].static_alias_session_id ==
                    session->session_id) {
                    g_uvm_moe_slot_pool[i].static_alias_poisoned = NV_TRUE;
                }
            }
            goto abort_transaction;
        }

        old_aliases = session->aliases;
        old_alias_count = session->alias_count;
        old_alias_bytes = session->alias_bytes;
        session->aliases = replacement_aliases;
        session->alias_count = upload_count + session->rebind_alias_count;
        session->alias_bytes = upload_bytes + session->rebind_alias_bytes;
        status = uvm_moe_static_alias_validate_transition_locked(session,
                                                                 va_space,
                                                                 NV_TRUE);
        if (status == NV_OK)
            status = uvm_moe_static_alias_refresh_certificate_locked(session);
        if (status != NV_OK) {
            session->aliases = old_aliases;
            session->alias_count = old_alias_count;
            session->alias_bytes = old_alias_bytes;
            goto abort_transaction;
        }
        replacement_aliases = NULL;
        uvm_kvfree(old_aliases);
        params->total_alias_bytes = session->alias_bytes;
        params->committed = 1;
        uvm_moe_static_alias_rebind_reset_locked(session);
        params->rebind_id = 0;
    }

    goto out;

abort_transaction:
    if (session && session->rebind_id != 0) {
        uvm_moe_static_alias_rebind_reset_locked(session);
        params->rebind_id = 0;
    }

out:
    if (session) {
        params->staged_consumer_alias_bytes = session->rebind_alias_bytes;
        params->staged_consumer_alias_count = session->rebind_alias_count;
        params->backing_bytes = session->backing_bytes;
        params->total_alias_bytes = session->alias_bytes;
        params->poisoned = session->poisoned;
    }
    if (status != NV_OK) {
        uvm_moe_static_alias_rebind_fail_entries(
            params, UVM_MOE_STATIC_ALIAS_ENTRY_ROLLED_BACK);
        if (transaction_started)
            params->rebind_id = 0;
    }
    if (pool_locked)
        mutex_unlock(&g_uvm_moe_slot_pool_lock);
    if (va_locked)
        uvm_va_space_up_write(va_space);
    if (mm_locked)
        uvm_va_space_mm_or_current_release_unlock(va_space, mm);
    if (gpu)
        uvm_gpu_release(gpu);
    if (plan_locked)
        mutex_unlock(&g_uvm_moe_slot_plan_lock);
    uvm_kvfree(pending_per_pool);
    uvm_kvfree(replacement_aliases);
    uvm_kvfree(state);
    uvm_kvfree(legacy);
    return status;
}

static NV_STATUS uvm_moe_static_alias_resume_chunk_locked(
    uvm_moe_static_alias_session_t *session,
    uvm_va_space_t *va_space,
    uvm_gpu_t *gpu,
    struct mm_struct *mm,
    NvU32 *record_cursor,
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *legacy,
    uvm_moe_slot_direct_batch_state_t *state,
    UVM_MOE_STATIC_ALIAS_TRANSITION_PARAMS *params,
    NvBool force_sequential)
{
    uvm_tracker_t batch_tracker = UVM_TRACKER_INIT();
    uvm_va_block_slot_direct_result_t map_result = {0};
    NV_STATUS status = NV_OK;
    NV_STATUS tracker_status = NV_OK;
    NvU64 chunk_bytes = 0;
    NvU64 stage_start = NV_GETTIME();
    NvU32 i;
    NvBool tracker_initialized = NV_FALSE;

    memset(legacy, 0, sizeof(*legacy));
    memset(state, 0, sizeof(*state));
    legacy->mode = UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE;
    uvm_uuid_copy(&legacy->gpu_uuid, &session->gpu_uuid);

    while (*record_cursor < session->alias_count &&
           legacy->count < UVM_MOE_STATIC_ALIAS_MAX_ENTRIES) {
        const uvm_moe_static_alias_record_t *record =
            &session->aliases[*record_cursor];
        uvm_moe_slot_pool_slot_t *slot;
        UVM_MOE_SLOT_REMAP_ENTRY *entry;

        ++*record_cursor;
        if (record->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW)
            continue;

        slot = &g_uvm_moe_slot_pool[record->pool_index];
        entry = &legacy->entries[legacy->count];
        entry->layer_id = slot->layer_id;
        entry->expert_id = slot->owner_expert_id;
        entry->slot_id = slot->slot_id;
        // role is otherwise diagnostic-only in this control path. Retain the
        // static-alias PTE-mode bits here so flattened block entries can be
        // configured without an O(session-size) record lookup per block.
        entry->role = record->flags;
        entry->va_base = record->va_base;
        entry->bytes = record->bytes;
        entry->slot_offset = record->slot_offset;
        entry->flags = UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT;
        entry->status = UVM_MOE_SLOT_REMAP_ENTRY_OK;
        state->slots[legacy->count] = slot;
        chunk_bytes += record->bytes;
        ++legacy->count;
        if (force_sequential)
            break;
    }

    if (legacy->count == 0)
        return NV_OK;

    status = uvm_moe_slot_direct_flatten_blocks_locked(legacy,
                                                        va_space,
                                                        state,
                                                        NV_TRUE);
    params->build_sort_ns += NV_GETTIME() - stage_start;
    if (status != NV_OK)
        goto out;

    for (i = 0; i < state->block_count; ++i) {
        NvU32 entry_index = state->block_entry_indices[i];
        NvU32 flags = legacy->entries[entry_index].role;

        state->blocks[i].pte_mode =
            (flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K) ?
                UVM_SLOT_DIRECT_PTE_MODE_64K : UVM_SLOT_DIRECT_PTE_MODE_4K;
        state->blocks[i].writable = false;
        state->blocks[i].slot_tracker = NULL;
    }
    uvm_moe_static_alias_dual_filter_blocks(session, state);
    // Likewise, a certified RESUME must not install a single consumer leaf.
    // Metadata may still be restored below for teardown ownership, but all
    // page-table work has already completed through the parent switch.
    if (session->dual_view_full_coverage && state->block_count != 0) {
        status = NV_ERR_INVALID_STATE;
        goto out;
    }

    stage_start = NV_GETTIME();
    status = state->block_count == 0 ? NV_OK :
        uvm_va_block_slot_direct_preflight_targets_not_slot_mapped(
            va_space,
            gpu,
            state->blocks,
            state->block_count);
    params->preflight_ns += NV_GETTIME() - stage_start;
    if (status != NV_OK)
        goto out;

    stage_start = NV_GETTIME();
    status = uvm_moe_slot_direct_record_all_pte_metadata_locked(legacy,
                                                                va_space,
                                                                gpu,
                                                                state);
    params->metadata_ns += NV_GETTIME() - stage_start;
    if (status != NV_OK)
        goto out;

    stage_start = NV_GETTIME();
    status = state->block_count == 0 ? NV_OK :
        uvm_va_block_slot_alias_revoke_cpu_mappings_batch(
            va_space,
            mm,
            state->blocks,
            state->block_count);
    params->cpu_unmap_ns += NV_GETTIME() - stage_start;
    if (status != NV_OK)
        goto out;

    uvm_tracker_init(&batch_tracker);
    tracker_initialized = NV_TRUE;
    stage_start = NV_GETTIME();
    status = state->block_count == 0 ? NV_OK :
        uvm_va_block_slot_alias_map_batch(va_space,
                                          mm,
                                          gpu,
                                          state->blocks,
                                          state->block_count,
                                          false,
                                          true,
                                          &g_uvm_moe_slot_pool_lock,
                                          &batch_tracker,
                                          &map_result);
    params->pte_map_ns += NV_GETTIME() - stage_start;
    stage_start = NV_GETTIME();
    if (status != NV_OK) {
        // A successful map already contributes every push (including its
        // prepare dependencies) to batch_tracker. Only a partial failure
        // needs the conservative all-block tracker sweep.
        for (i = 0; i < state->block_count; ++i) {
            uvm_va_block_t *block = state->blocks[i].va_block;
            NV_STATUS add_status;

            if (i > 0 && state->blocks[i - 1].va_block == block)
                continue;
            uvm_mutex_lock(&block->lock);
            add_status = uvm_tracker_add_tracker_safe(&batch_tracker,
                                                       &block->tracker);
            uvm_mutex_unlock(&block->lock);
            if (tracker_status == NV_OK)
                tracker_status = add_status;
        }
    }
    params->bookkeeping_ns += NV_GETTIME() - stage_start;
    {
        NvU64 wait_start = NV_GETTIME();
        NV_STATUS wait_status = uvm_tracker_wait(&batch_tracker);

        params->wait_ns += NV_GETTIME() - wait_start;
        if (tracker_status == NV_OK)
            tracker_status = wait_status;
    }
    if (status == NV_OK && tracker_status != NV_OK)
        status = tracker_status;
    if (status == NV_OK) {
        stage_start = NV_GETTIME();
        params->processed_alias_count += legacy->count;
        params->processed_alias_bytes += chunk_bytes;
        params->tlb_invalidate_count = (NvU32)min_t(
            NvU64,
            (NvU64)params->tlb_invalidate_count + map_result.mapped_blocks,
            (NvU64)NV_U32_MAX);
        ++params->reserved;
        params->bookkeeping_ns += NV_GETTIME() - stage_start;
    }

out:
    if (tracker_initialized)
        uvm_tracker_deinit(&batch_tracker);
    return status;
}

static NV_STATUS uvm_api_moe_static_alias_resume(
    UVM_MOE_STATIC_ALIAS_TRANSITION_PARAMS *params,
    struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_moe_static_alias_session_t *session;
    UVM_MOE_SLOT_REMAP_BATCH_PARAMS *legacy = NULL;
    uvm_moe_slot_direct_batch_state_t *state = NULL;
    uvm_va_block_slot_alias_dual_result_t dual_result = {0};
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    NV_STATUS status = NV_OK;
    NvU64 stage_start;
    NvU32 record_cursor = 0;
    NvU32 i;

    BUILD_BUG_ON(sizeof(*params) != 152);

    params->processed_alias_bytes = 0;
    params->processed_alias_count = 0;
    params->tlb_invalidate_count = 0;
    params->validation_ns = 0;
    params->build_sort_ns = 0;
    params->preflight_ns = 0;
    params->metadata_ns = 0;
    params->cpu_unmap_ns = 0;
    params->pte_map_ns = 0;
    params->wait_ns = 0;
    params->bookkeeping_ns = 0;
    uvm_moe_static_alias_transition_snapshot(NULL, params);

    if (params->abi_version != UVM_MOE_STATIC_ALIAS_ABI_VERSION ||
        params->struct_size != sizeof(*params) ||
        (params->flags &
         ~UVM_MOE_STATIC_ALIAS_TRANSITION_FLAG_FORCE_SEQUENTIAL) != 0 ||
        params->reserved != 0 ||
        params->session_id == 0) {
        return NV_ERR_INVALID_ARGUMENT;
    }
    params->reserved = 0;

    legacy = uvm_kvmalloc_zero(sizeof(*legacy));
    state = uvm_kvmalloc_zero(sizeof(*state));
    if (!legacy || !state) {
        status = NV_ERR_NO_MEMORY;
        goto out_alloc;
    }

    mutex_lock(&g_uvm_moe_slot_plan_lock);
    session = uvm_moe_static_alias_find_locked(va_space, params->session_id);
    if (!session) {
        status = NV_ERR_OBJECT_NOT_FOUND;
        goto out_plan;
    }
    stage_start = NV_GETTIME();
    uvm_moe_static_alias_transition_snapshot(session, params);
    if (!session->sealed || !session->parked || session->poisoned ||
        session->rebind_id != 0 || session->rebind_aliases != NULL ||
        params->consumer_alias_count == 0 || params->upload_alias_count == 0 ||
        !session->certificate_valid ||
        session->certificate_generation != session->topology_generation ||
        !session->certificate_parked) {
        status = session->poisoned ? NV_ERR_INVALID_STATE : NV_ERR_IN_USE;
        params->validation_ns += NV_GETTIME() - stage_start;
        goto out_plan;
    }
    params->validation_ns += NV_GETTIME() - stage_start;

    gpu = uvm_va_space_retain_gpu_by_uuid(va_space, &session->gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto out_plan;
    }
    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_write(va_space);
    mutex_lock(&g_uvm_moe_slot_pool_lock);

    // PARK committed this exact immutable topology under the same plan lock.
    // Static-alias guards reject external residency/PTE mutations, so a
    // generation-matched certificate replaces the former full 36K-record
    // preflight scan.
    status = NV_OK;
    if (session->dual_view_block_count != 0) {
        NvU64 dual_start = NV_GETTIME();

        if ((params->flags &
             UVM_MOE_STATIC_ALIAS_TRANSITION_FLAG_FORCE_SEQUENTIAL) ||
            session->dual_view_generation != session->topology_generation) {
            status = NV_ERR_INVALID_STATE;
        }
        else {
            status = uvm_va_block_slot_alias_dual_switch_batch(
                va_space,
                gpu,
                session->dual_view_block_starts,
                session->dual_view_block_count,
                false,
                &dual_result);
        }
        params->pte_map_ns += NV_GETTIME() - dual_start;
        params->tlb_invalidate_count = (NvU32)min_t(
            NvU64,
            dual_result.tlb_batches,
            (NvU64)NV_U32_MAX);
        if (status == NV_OK) {
            uvm_phase_f_route_log(
                "moe_static_alias_dual_resume session=%llu generation=%llu blocks=%u pdes=%llu tlb_batches=%llu ns=%llu\n",
                session->session_id,
                session->dual_view_generation,
                session->dual_view_block_count,
                dual_result.pde_writes,
                dual_result.tlb_batches,
                dual_result.elapsed_ns);
        }
    }
    while (status == NV_OK && record_cursor < session->alias_count) {
        status = uvm_moe_static_alias_resume_chunk_locked(session,
                                                          va_space,
                                                          gpu,
                                                          mm,
                                                          &record_cursor,
                                                          legacy,
                                                          state,
                                                          params,
                                                          (params->flags &
                                                           UVM_MOE_STATIC_ALIAS_TRANSITION_FLAG_FORCE_SEQUENTIAL) != 0);
    }

    stage_start = NV_GETTIME();
    if (status == NV_OK &&
        params->processed_alias_count == params->consumer_alias_count &&
        params->processed_alias_bytes == params->consumer_alias_bytes) {
        if (session->dual_view_block_count == 0)
            status = uvm_moe_static_alias_dual_prepare_locked(session,
                                                               va_space,
                                                               mm,
                                                               gpu,
                                                               NV_FALSE);
    }
    if (status == NV_OK &&
        params->processed_alias_count == params->consumer_alias_count &&
        params->processed_alias_bytes == params->consumer_alias_bytes) {
        session->parked = NV_FALSE;
        session->certificate_generation = session->topology_generation;
        session->certificate_valid = NV_TRUE;
        session->certificate_parked = NV_FALSE;
    }
    if (status != NV_OK) {
        session->parked = NV_FALSE;
        session->certificate_valid = NV_FALSE;
        session->poisoned = NV_TRUE;
        for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
            if (g_uvm_moe_slot_pool[i].static_alias_session_id == session->session_id)
                g_uvm_moe_slot_pool[i].static_alias_poisoned = NV_TRUE;
        }
    }
    params->validation_ns += NV_GETTIME() - stage_start;

    stage_start = NV_GETTIME();
    uvm_moe_static_alias_transition_snapshot(session, params);
    params->bookkeeping_ns += NV_GETTIME() - stage_start;
    mutex_unlock(&g_uvm_moe_slot_pool_lock);
    uvm_va_space_up_write(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);
    mm = NULL;

out_plan:
    if (gpu)
        uvm_gpu_release(gpu);
    mutex_unlock(&g_uvm_moe_slot_plan_lock);

out_alloc:
    uvm_kvfree(state);
    uvm_kvfree(legacy);
    return status;
}

static NvBool uvm_moe_static_alias_record_matches_entry(
    const uvm_moe_static_alias_record_t *record,
    const UVM_MOE_STATIC_ALIAS_ENTRY *entry)
{
    return record->active &&
           record->backing_layer_id == entry->backing_layer_id &&
           record->backing_expert_id == entry->backing_expert_id &&
           record->backing_slot_id == entry->backing_slot_id &&
           record->flags == entry->flags &&
           record->va_base == entry->va_base &&
           record->bytes == entry->bytes &&
           record->slot_offset == entry->slot_offset;
}

static NvBool uvm_moe_static_alias_entries_equal(
    const UVM_MOE_STATIC_ALIAS_ENTRY *left,
    const UVM_MOE_STATIC_ALIAS_ENTRY *right)
{
    return left->backing_layer_id == right->backing_layer_id &&
           left->backing_expert_id == right->backing_expert_id &&
           left->backing_slot_id == right->backing_slot_id &&
           left->flags == right->flags && left->va_base == right->va_base &&
           left->bytes == right->bytes &&
           left->slot_offset == right->slot_offset;
}

/*
 * Remove only the complete backing identities named by userspace. The caller
 * has already synchronized the sole serving stream at the scheduler boundary;
 * this command therefore never performs the all-session PARK/RESUME cycle.
 * Exact-record and whole-backing validation completes before the first PTE or
 * allocation changes. A later failure is fail-closed: successfully released
 * identities are removed from the session and every retained identity is
 * poisoned for authoritative DESTROY cleanup.
 */
static NV_STATUS uvm_api_moe_static_alias_quiesced_remove(
    UVM_MOE_STATIC_ALIAS_QUIESCED_REMOVE_PARAMS *params,
    struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_moe_static_alias_session_t *session = NULL;
    uvm_moe_static_alias_record_t *replacement_aliases = NULL;
    NvBool *remove_records = NULL;
    NvBool *remove_pools = NULL;
    NvBool *released_pools = NULL;
    NvU32 *entry_pools = NULL;
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    NV_STATUS status = NV_OK;
    NvU64 requested_bytes = 0;
    NvU64 released_backing_bytes = 0;
    NvU32 remaining_count = 0;
    NvU32 i;
    NvBool mutation_started = NV_FALSE;
    bool plan_locked = false;
    bool mm_locked = false;
    bool va_locked = false;
    bool pool_locked = false;

    BUILD_BUG_ON(sizeof(*params) != 221288);
    BUILD_BUG_ON(offsetof(UVM_MOE_STATIC_ALIAS_QUIESCED_REMOVE_PARAMS,
                          entries) != 96);
    BUILD_BUG_ON(offsetof(UVM_MOE_STATIC_ALIAS_QUIESCED_REMOVE_PARAMS,
                          rmStatus) != 221280);

    params->requested_bytes = 0;
    params->released_alias_bytes = 0;
    params->total_alias_bytes = 0;
    params->backing_bytes = 0;
    params->success_count = 0;
    params->failure_count = 0;
    params->remaining_alias_count = 0;
    params->tlb_invalidate_count = 0;
    params->poisoned = 0;

    if (params->abi_version != UVM_MOE_STATIC_ALIAS_ABI_VERSION ||
        params->struct_size != sizeof(*params) ||
        params->flags !=
            UVM_MOE_STATIC_ALIAS_QUIESCED_REMOVE_FLAG_RELEASE_BACKING ||
        params->reserved != 0 || params->session_id == 0 ||
        params->count == 0 ||
        params->count > UVM_MOE_STATIC_ALIAS_MAX_ENTRIES) {
        return NV_ERR_INVALID_ARGUMENT;
    }

    for (i = 0; i < params->count; ++i) {
        UVM_MOE_STATIC_ALIAS_ENTRY *entry = &params->entries[i];
        NvU32 j;

        entry->status = UVM_MOE_STATIC_ALIAS_ENTRY_OK;
        if (uvm_moe_static_alias_range_invalid(entry) ||
            requested_bytes + entry->bytes < requested_bytes) {
            entry->status = UVM_MOE_STATIC_ALIAS_ENTRY_INVALID_ARGUMENT;
            status = NV_ERR_INVALID_ARGUMENT;
            continue;
        }
        requested_bytes += entry->bytes;
        for (j = 0; j < i; ++j) {
            if (uvm_moe_static_alias_entries_equal(
                    &params->entries[j], entry)) {
                entry->status = UVM_MOE_STATIC_ALIAS_ENTRY_ALIAS_CONFLICT;
                status = NV_ERR_INVALID_ARGUMENT;
                break;
            }
        }
    }
    params->requested_bytes = requested_bytes;
    if (status != NV_OK)
        goto out;

    mutex_lock(&g_uvm_moe_slot_plan_lock);
    plan_locked = true;
    session = uvm_moe_static_alias_find_locked(va_space, params->session_id);
    if (!session) {
        status = NV_ERR_OBJECT_NOT_FOUND;
        goto out;
    }
    params->total_alias_bytes = session->alias_bytes;
    params->backing_bytes = session->backing_bytes;
    params->remaining_alias_count = session->alias_count;
    params->poisoned = session->poisoned;
    /* A parked session has already removed its consumer alias PTEs, but it
     * deliberately retains the complete alias/backing registry for the next
     * RESUME.  A scheduler-quiesced remove may therefore shrink that
     * registry in either sealed state.  release_backing_locked() also clears
     * the removed slot's upload PTE and any current ordinary remote/local PTE
     * for its recorded consumer ranges; retained slots are untouched. */
    if (!session->sealed || session->poisoned ||
        session->rebind_id != 0 || session->rebind_aliases != NULL ||
        !uvm_uuid_eq(&session->gpu_uuid, &params->gpu_uuid) ||
        params->count >= session->alias_count) {
        status = session->poisoned ? NV_ERR_INVALID_STATE : NV_ERR_IN_USE;
        goto out;
    }

    remove_records = uvm_kvmalloc_zero(
        session->alias_count * sizeof(*remove_records));
    remove_pools = uvm_kvmalloc_zero(
        UVM_MOE_SLOT_POOL_MAX_SLOTS * sizeof(*remove_pools));
    released_pools = uvm_kvmalloc_zero(
        UVM_MOE_SLOT_POOL_MAX_SLOTS * sizeof(*released_pools));
    entry_pools = uvm_kvmalloc_zero(params->count * sizeof(*entry_pools));
    replacement_aliases = uvm_kvmalloc_zero(
        (session->alias_count - params->count) * sizeof(*replacement_aliases));
    if (!remove_records || !remove_pools || !released_pools || !entry_pools ||
        !replacement_aliases) {
        status = NV_ERR_NO_MEMORY;
        goto out;
    }

    for (i = 0; i < params->count; ++i) {
        NvU32 record_index;
        NvBool found = NV_FALSE;

        for (record_index = 0; record_index < session->alias_count;
             ++record_index) {
            const uvm_moe_static_alias_record_t *record =
                &session->aliases[record_index];

            if (remove_records[record_index] ||
                !uvm_moe_static_alias_record_matches_entry(
                    record, &params->entries[i])) {
                continue;
            }
            if (record->pool_index >= UVM_MOE_SLOT_POOL_MAX_SLOTS) {
                status = NV_ERR_INVALID_STATE;
                goto out;
            }
            remove_records[record_index] = NV_TRUE;
            remove_pools[record->pool_index] = NV_TRUE;
            entry_pools[i] = record->pool_index;
            found = NV_TRUE;
            break;
        }
        if (!found) {
            params->entries[i].status =
                UVM_MOE_STATIC_ALIAS_ENTRY_BACKING_NOT_FOUND;
            status = NV_ERR_OBJECT_NOT_FOUND;
            goto out;
        }
    }

    /* A backing identity is indivisible: removing only its consumer or only
     * its writable upload alias would leave an unauditable slot lifetime. */
    for (i = 0; i < session->alias_count; ++i) {
        const uvm_moe_static_alias_record_t *record = &session->aliases[i];

        if (record->pool_index >= UVM_MOE_SLOT_POOL_MAX_SLOTS) {
            status = NV_ERR_INVALID_STATE;
            goto out;
        }
        if (record->active && remove_pools[record->pool_index] &&
            !remove_records[i]) {
            status = NV_ERR_INVALID_ARGUMENT;
            goto out;
        }
    }

    gpu = uvm_va_space_retain_gpu_by_uuid(va_space, &session->gpu_uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto out;
    }
    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    mm_locked = true;
    uvm_va_space_down_write(va_space);
    va_locked = true;
    mutex_lock(&g_uvm_moe_slot_pool_lock);
    pool_locked = true;

    if (session->dual_view_block_count != 0)
        mutation_started = NV_TRUE;
    status = uvm_moe_static_alias_dual_discard_locked(session,
                                                       va_space,
                                                       mm,
                                                       gpu);
    if (status != NV_OK)
        goto out;

    for (i = 0; i < session->alias_count; ++i) {
        const uvm_moe_static_alias_record_t *record = &session->aliases[i];
        const uvm_moe_slot_pool_slot_t *slot;
        NvBool mapped;

        if (!remove_records[i])
            continue;
        slot = &g_uvm_moe_slot_pool[record->pool_index];
        mapped = uvm_moe_static_alias_slot_has_exact_range(
            slot, record->va_base, record->bytes);
        if (!slot->valid || slot->owner_va_space != va_space ||
            slot->static_alias_session_id != session->session_id ||
            !slot->static_alias_sealed || slot->static_alias_poisoned ||
            slot->pte_va_space != va_space || slot->pte_gpu != gpu ||
            slot->pte_mode != UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP ||
            ((record->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW) && !mapped) ||
            (!(record->flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW) &&
             (session->parked ? mapped : !mapped))) {
            status = NV_ERR_INVALID_STATE;
            goto out;
        }
    }

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot;
        NvU64 slot_bytes;
        NvU32 pte_range_count;

        if (!remove_pools[i])
            continue;
        slot = &g_uvm_moe_slot_pool[i];
        slot_bytes = slot->slot_bytes;
        pte_range_count = slot->pte_range_count;
        mutation_started = NV_TRUE;
        status = uvm_moe_slot_pool_release_backing_locked(slot, va_space, mm);
        if (status != NV_OK)
            break;
        released_pools[i] = NV_TRUE;
        released_backing_bytes += slot_bytes;
        params->tlb_invalidate_count += pte_range_count;
        memset(slot, 0, sizeof(*slot));
    }

    /* Commit each successfully released identity even on a later failure so
     * DESTROY can converge from the remaining, poisoned session. */
    for (i = 0; i < session->alias_count; ++i) {
        uvm_moe_static_alias_record_t *record = &session->aliases[i];

        if (record->active &&
            record->pool_index < UVM_MOE_SLOT_POOL_MAX_SLOTS &&
            released_pools[record->pool_index]) {
            record->active = NV_FALSE;
            ++params->success_count;
            params->released_alias_bytes += record->bytes;
        }
    }
    for (i = 0; i < session->alias_count; ++i) {
        const uvm_moe_static_alias_record_t *record = &session->aliases[i];

        if (!record->active)
            continue;
        replacement_aliases[remaining_count++] = *record;
    }
    if (params->success_count != 0) {
        uvm_kvfree(session->aliases);
        session->aliases = replacement_aliases;
        replacement_aliases = NULL;
        session->alias_count = remaining_count;
        session->alias_bytes -= min(session->alias_bytes,
                                    params->released_alias_bytes);
        session->backing_bytes -= min(session->backing_bytes,
                                      released_backing_bytes);
        if (status == NV_OK)
            status = uvm_moe_static_alias_refresh_certificate_locked(session);
        if (status == NV_OK && !session->parked)
            status = uvm_moe_static_alias_dual_prepare_locked(session,
                                                               va_space,
                                                               mm,
                                                               gpu,
                                                               NV_FALSE);
    }

    if (status == NV_OK && params->success_count != params->count)
        status = NV_ERR_INVALID_STATE;
    if (status != NV_OK && mutation_started) {
        session->certificate_valid = NV_FALSE;
        session->poisoned = NV_TRUE;
        for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
            if (g_uvm_moe_slot_pool[i].static_alias_session_id ==
                session->session_id) {
                g_uvm_moe_slot_pool[i].static_alias_poisoned = NV_TRUE;
            }
        }
    }

    for (i = 0; i < params->count; ++i) {
        NvBool removed =
            entry_pools[i] < UVM_MOE_SLOT_POOL_MAX_SLOTS &&
            released_pools[entry_pools[i]];

        params->entries[i].status = removed ?
            UVM_MOE_STATIC_ALIAS_ENTRY_OK :
            (mutation_started ? UVM_MOE_STATIC_ALIAS_ENTRY_MAP_FAILED :
                                UVM_MOE_STATIC_ALIAS_ENTRY_ROLLED_BACK);
    }

    params->failure_count = params->count - params->success_count;
    params->total_alias_bytes = session->alias_bytes;
    params->backing_bytes = session->backing_bytes;
    params->remaining_alias_count = session->alias_count;
    params->poisoned = session->poisoned;
    uvm_phase_f_route_log(
        "moe_static_alias_quiesced_remove session=%llu requested_aliases=%u released_aliases=%u released_bytes=%llu remaining_aliases=%u remaining_bytes=%llu backing=%llu poisoned=%u status=0x%x\n",
        session->session_id,
        params->count,
        params->success_count,
        params->released_alias_bytes,
        session->alias_count,
        session->alias_bytes,
        session->backing_bytes,
        session->poisoned,
        status);

out:
    if (session) {
        params->total_alias_bytes = session->alias_bytes;
        params->backing_bytes = session->backing_bytes;
        params->remaining_alias_count = session->alias_count;
        params->poisoned = session->poisoned;
    }
    if (status != NV_OK && !mutation_started) {
        params->success_count = 0;
        params->failure_count = params->count;
        for (i = 0; i < params->count; ++i) {
            if (params->entries[i].status == UVM_MOE_STATIC_ALIAS_ENTRY_OK)
                params->entries[i].status =
                    UVM_MOE_STATIC_ALIAS_ENTRY_ROLLED_BACK;
        }
    }
    if (pool_locked)
        mutex_unlock(&g_uvm_moe_slot_pool_lock);
    if (va_locked)
        uvm_va_space_up_write(va_space);
    if (mm_locked)
        uvm_va_space_mm_or_current_release_unlock(va_space, mm);
    if (gpu)
        uvm_gpu_release(gpu);
    if (plan_locked)
        mutex_unlock(&g_uvm_moe_slot_plan_lock);
    uvm_kvfree(replacement_aliases);
    uvm_kvfree(entry_pools);
    uvm_kvfree(released_pools);
    uvm_kvfree(remove_pools);
    uvm_kvfree(remove_records);
    return status;
}

static void uvm_moe_static_alias_deactivate_pool_records(
    uvm_moe_static_alias_session_t *session,
    NvU32 pool_index,
    UVM_MOE_STATIC_ALIAS_DESTROY_PARAMS *params)
{
    NvU32 i;

    for (i = 0; i < session->alias_count; ++i) {
        uvm_moe_static_alias_record_t *record = &session->aliases[i];

        if (!record->active || record->pool_index != pool_index)
            continue;

        record->active = NV_FALSE;
        ++params->released_alias_count;
        params->released_alias_bytes += record->bytes;
    }
}

static NV_STATUS uvm_api_moe_static_alias_destroy(UVM_MOE_STATIC_ALIAS_DESTROY_PARAMS *params,
                                                  struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_moe_static_alias_session_t *session;
    uvm_gpu_t *gpu = NULL;
    struct mm_struct *mm = NULL;
    NV_STATUS first_error = NV_OK;
    NvU32 i;
    bool release_backing;
    bool defer_cpu_restore;

    BUILD_BUG_ON(sizeof(*params) != 80);

    params->released_alias_bytes = 0;
    params->remaining_alias_bytes = 0;
    params->released_backing_bytes = 0;
    params->remaining_backing_bytes = 0;
    params->released_alias_count = 0;
    params->remaining_alias_count = 0;
    params->session_cleared = 0;
    params->poisoned = 0;

    if (params->abi_version != UVM_MOE_STATIC_ALIAS_ABI_VERSION ||
        params->struct_size != sizeof(*params) ||
        (params->flags & ~UVM_MOE_STATIC_ALIAS_DESTROY_FLAGS_ALL) ||
        params->reserved != 0 ||
        params->session_id == 0) {
        return NV_ERR_INVALID_ARGUMENT;
    }
    release_backing =
        (params->flags & UVM_MOE_STATIC_ALIAS_DESTROY_FLAG_RELEASE_BACKING) != 0;
    defer_cpu_restore =
        (params->flags & UVM_MOE_STATIC_ALIAS_DESTROY_FLAG_DEFER_CPU_RESTORE) != 0;
    if (defer_cpu_restore && !release_backing)
        return NV_ERR_INVALID_ARGUMENT;

    // Userspace must synchronize all streams and destroy every graph which
    // captured an upload/alias operation before this command. A certified
    // remote-only graph may remain live when it contains no alias upload node
    // and references only original allocation VAs: replay observes the PTEs
    // installed after this teardown. The kernel cannot distinguish those graph
    // topologies, so userspace owns and audits that stronger contract.
    mutex_lock(&g_uvm_moe_slot_plan_lock);
    session = uvm_moe_static_alias_find_locked(va_space, params->session_id);
    if (!session) {
        first_error = NV_ERR_OBJECT_NOT_FOUND;
        goto out_plan;
    }

    // Freeze MAP/SEAL immediately. A successful alias-only destroy retains an
    // unpoisoned, destroy-only backing handle so a later RELEASE_BACKING call
    // can reclaim HBM without rebuilding alias metadata.
    session->sealed = NV_TRUE;
    // DESTROY is authoritative teardown. A candidate REBIND set has never
    // owned PTEs, so discard it before walking committed slot metadata.
    uvm_moe_static_alias_rebind_reset_locked(session);
    gpu = uvm_va_space_retain_gpu_by_uuid(va_space, &session->gpu_uuid);
    if (!gpu) {
        first_error = NV_ERR_INVALID_DEVICE;
        goto out_plan;
    }
    mm = uvm_va_space_mm_or_current_retain_lock(va_space);
    uvm_va_space_down_write(va_space);
    mutex_lock(&g_uvm_moe_slot_pool_lock);

    first_error = uvm_moe_static_alias_dual_discard_locked(session,
                                                            va_space,
                                                            mm,
                                                            gpu);
    if (first_error != NV_OK) {
        session->poisoned = NV_TRUE;
        params->poisoned = 1;
        goto out_locked;
    }

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];
        NvU64 slot_bytes;
        NV_STATUS status;

        if (slot->static_alias_session_id != session->session_id)
            continue;

        slot_bytes = slot->slot_bytes;
        status = release_backing ?
                     uvm_moe_slot_pool_release_backing_mode_locked(slot,
                                                                   va_space,
                                                                   mm,
                                                                   defer_cpu_restore ? NV_FALSE : NV_TRUE) :
                     uvm_moe_slot_pool_release_pte_mapping_locked(slot, va_space, mm);
        if (status != NV_OK) {
            session->poisoned = NV_TRUE;
            slot->static_alias_poisoned = NV_TRUE;
            slot->state = UVM_MOE_SLOT_POOL_STATE_ERROR;
            uvm_moe_slot_pool_note_pte_failure_locked(slot);
            if (first_error == NV_OK)
                first_error = status;
            continue;
        }

        uvm_moe_static_alias_deactivate_pool_records(session, i, params);
        if (release_backing) {
            params->released_backing_bytes += slot_bytes;
            memset(slot, 0, sizeof(*slot));
        }
        else {
            // Keep only backing ownership. All GPU aliases and their cleanup
            // metadata are gone, but the slot stays unavailable to cmd91/96
            // until a later RELEASE_BACKING destroy completes.
            slot->static_alias_sealed = NV_TRUE;
            slot->static_alias_poisoned = NV_FALSE;
            if (slot->valid)
                slot->state = UVM_MOE_SLOT_POOL_STATE_READY;
        }
    }

    for (i = 0; i < session->alias_count; ++i) {
        const uvm_moe_static_alias_record_t *record = &session->aliases[i];

        if (!record->active)
            continue;
        ++params->remaining_alias_count;
        params->remaining_alias_bytes += record->bytes;
    }

    for (i = 0; i < UVM_MOE_SLOT_POOL_MAX_SLOTS; ++i) {
        const uvm_moe_slot_pool_slot_t *slot = &g_uvm_moe_slot_pool[i];

        if (slot->static_alias_session_id == session->session_id)
            params->remaining_backing_bytes += slot->slot_bytes;
    }

    if (params->remaining_alias_count == 0 && params->remaining_backing_bytes == 0) {
        params->session_cleared = 1;
        params->poisoned = 0;
        uvm_moe_static_alias_session_free_locked(session);
    }
    else if (!release_backing &&
             first_error == NV_OK &&
             params->remaining_alias_count == 0 &&
             params->remaining_backing_bytes != 0) {
        // Alias-only destruction completed exactly as requested. Compact the
        // session to a backing-only handle; the next valid operation is
        // DESTROY with RELEASE_BACKING.
        uvm_kvfree(session->aliases);
        session->aliases = NULL;
        session->alias_count = 0;
        session->alias_bytes = 0;
        session->backing_bytes = params->remaining_backing_bytes;
        session->poisoned = NV_FALSE;
        params->session_cleared = 0;
        params->poisoned = 0;
    }
    else {
        session->poisoned = NV_TRUE;
        params->poisoned = 1;
        if (first_error == NV_OK)
            first_error = NV_ERR_INVALID_STATE;
    }

out_locked:
    mutex_unlock(&g_uvm_moe_slot_pool_lock);
    uvm_va_space_up_write(va_space);
    uvm_va_space_mm_or_current_release_unlock(va_space, mm);

out_plan:
    if (gpu)
        uvm_gpu_release(gpu);
    mutex_unlock(&g_uvm_moe_slot_plan_lock);
    return first_error;
}

/*
 * A resizing REBIND adopts backing prepared by mode 8 and, on a failed
 * preparation, userspace releases that uncommitted backing with mode 19.
 * RELEASE is safe alongside a static-alias session only while all consumers
 * are parked. PREALLOC touches neither PTEs nor committed session metadata, so
 * it is also safe for a sealed non-parked session; this lets a scheduler-idle
 * quiesced append prepare only its new logical slots without a global PARK.
 * Both operations remain restricted to logical slots outside the committed
 * session. Keep every other legacy slot operation excluded exactly as before.
 *
 * The caller holds the plan lock, so the committed alias set is immutable
 * until the legacy operation finishes. The slot-pool path subsequently
 * validates the concrete owner and backing while holding the pool lock.
 */
static NvBool uvm_moe_static_alias_parked_backing_op_allowed_locked(
    const uvm_moe_static_alias_session_t *session,
    const UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params)
{
    NvU32 entry_index;

    UVM_ASSERT(mutex_is_locked(&g_uvm_moe_slot_plan_lock));

    if (!session || !session->sealed ||
        session->poisoned || session->rebind_id != 0 ||
        session->rebind_aliases != NULL ||
        (params->mode != UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC &&
         params->mode != UVM_MOE_SLOT_REMAP_MODE_RELEASE)) {
        return NV_FALSE;
    }

    if (params->mode == UVM_MOE_SLOT_REMAP_MODE_RELEASE && !session->parked)
        return NV_FALSE;

    if (params->mode == UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC &&
        !uvm_uuid_eq(&session->gpu_uuid, &params->gpu_uuid)) {
        return NV_FALSE;
    }

    for (entry_index = 0; entry_index < params->count; ++entry_index) {
        const UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[entry_index];
        NvU32 alias_index;

        for (alias_index = 0; alias_index < session->alias_count;
             ++alias_index) {
            const uvm_moe_static_alias_record_t *record =
                &session->aliases[alias_index];

            if (record->active &&
                record->backing_layer_id == entry->layer_id &&
                record->backing_slot_id == entry->slot_id) {
                return NV_FALSE;
            }
        }
    }

    return NV_TRUE;
}

static NV_STATUS uvm_api_moe_slot_remap_batch(UVM_MOE_SLOT_REMAP_BATCH_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_moe_static_alias_session_t *static_alias_session;
    uvm_gpu_t *gpu = NULL;
    NV_STATUS first_error = NV_OK;
    NvU32 i;

    uvm_va_block_note_slot_legacy_ioctl(
        params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION);

    if (params->count > UVM_MOE_SLOT_REMAP_MAX_ENTRIES ||
        !uvm_moe_slot_remap_mode_valid(params->mode))
        return NV_ERR_INVALID_ARGUMENT;

    if (params->mode == UVM_MOE_SLOT_REMAP_MODE_RELEASE && params->validate_managed != 0)
        return NV_ERR_INVALID_ARGUMENT;

    params->success_count = 0;
    params->failure_count = 0;
    params->conflict_count = 0;
    params->remap_count = 0;
    params->tlb_invalidate_count = 0;
    params->reserved = 0;
    params->requested_bytes = 0;
    params->accepted_bytes = 0;

    // A registered plan owns the fixed-bank generation. Hold this lock for
    // the complete legacy handler, not just the initial check: otherwise a
    // concurrent register could publish a plan after the check while this
    // ioctl later changes the same slot owners without advancing generation.
    mutex_lock(&g_uvm_moe_slot_plan_lock);
    static_alias_session = uvm_moe_static_alias_find_va_space_locked(va_space);
    if (g_uvm_moe_slot_plan_owner_va_space == va_space ||
        (static_alias_session &&
         !uvm_moe_static_alias_parked_backing_op_allowed_locked(
             static_alias_session, params))) {
        mutex_unlock(&g_uvm_moe_slot_plan_lock);
        return NV_ERR_IN_USE;
    }

    uvm_phase_f_route_log("moe_slot_remap_batch begin mode=%u count=%u validate_managed=%u first_layer=%u first_expert=%u first_slot=%u first_va=0x%llx first_bytes=%llu\n",
                          params->mode,
                          params->count,
                          params->validate_managed,
                          params->count > 0 ? params->entries[0].layer_id : 0,
                          params->count > 0 ? params->entries[0].expert_id : 0,
                          params->count > 0 ? params->entries[0].slot_id : 0,
                          params->count > 0 ? params->entries[0].va_base : 0,
                          params->count > 0 ? params->entries[0].bytes : 0);

    for (i = 0; i < params->count; ++i) {
        UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[i];
        NV_STATUS status = NV_OK;

        entry->status = UVM_MOE_SLOT_REMAP_ENTRY_OK;

        if (entry->bytes > 0)
            params->requested_bytes += entry->bytes;

        if (params->mode == UVM_MOE_SLOT_REMAP_MODE_RELEASE &&
            uvm_moe_slot_release_entry_invalid(entry)) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ARGUMENT;
            status = NV_ERR_INVALID_ARGUMENT;
        }
        else if (params->mode == UVM_MOE_SLOT_REMAP_MODE_RELEASE &&
                 uvm_moe_slot_release_has_duplicate(params, i)) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_SLOT_CONFLICT;
            ++params->conflict_count;
            status = NV_ERR_INVALID_ARGUMENT;
        }
        else if (params->mode != UVM_MOE_SLOT_REMAP_MODE_RELEASE &&
                 uvm_moe_slot_remap_entry_range_invalid(entry)) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ARGUMENT;
            status = NV_ERR_INVALID_ARGUMENT;
        }
        else if (params->mode != UVM_MOE_SLOT_REMAP_MODE_RELEASE &&
                 !uvm_moe_slot_transition_fast_index_enabled(params->mode) &&
                 uvm_moe_slot_remap_has_slot_conflict(params, i)) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_SLOT_CONFLICT;
            ++params->conflict_count;
            status = NV_ERR_INVALID_ARGUMENT;
        }
        else if (params->mode == UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC &&
                 (entry->flags & ~(UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT |
                                   UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED))) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ARGUMENT;
            status = NV_ERR_INVALID_ARGUMENT;
        }
        else if (uvm_moe_slot_remap_is_layer_transition_mode(params->mode) &&
                 !uvm_moe_slot_transition_entry_strategy_valid(params, i)) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ARGUMENT;
            status = NV_ERR_INVALID_ARGUMENT;
        }
        else if (!uvm_moe_slot_remap_is_layer_transition_mode(params->mode) &&
                 uvm_moe_slot_remap_is_slot_direct_mode(params->mode) &&
                 (entry->flags & ~UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT)) {
            entry->status = UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ARGUMENT;
            status = NV_ERR_INVALID_ARGUMENT;
        }
        else if (params->validate_managed &&
                 params->mode != UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP &&
                 !uvm_moe_slot_remap_is_slot_direct_mode(params->mode)) {
            status = uvm_moe_slot_remap_validate_managed_range(va_space, entry->va_base, entry->bytes);
            if (status != NV_OK)
                entry->status = UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ADDRESS;
        }

        if (status == NV_OK && params->mode == UVM_MOE_SLOT_REMAP_MODE_DRY_RUN) {
            ++params->success_count;
            params->accepted_bytes += entry->bytes;
        }
        else if (status != NV_OK) {
            ++params->failure_count;
            if (first_error == NV_OK)
                first_error = status;
        }

        uvm_phase_f_route_log("moe_slot_remap_entry index=%u mode=%u layer=%u expert=%u slot=%u role=%u va=0x%llx bytes=%llu slot_offset=%llu flags=0x%x status=%u nv_status=0x%x\n",
                              i,
                              params->mode,
                              entry->layer_id,
                              entry->expert_id,
                              entry->slot_id,
                              entry->role,
                              entry->va_base,
                              entry->bytes,
                              entry->slot_offset,
                              entry->flags,
                              entry->status,
                              status);
    }

    if (params->mode == UVM_MOE_SLOT_REMAP_MODE_RELEASE)
        first_error = uvm_moe_slot_pool_release_batch(params, va_space, first_error);

    if (uvm_moe_slot_remap_is_slot_direct_mode(params->mode)) {
        first_error = uvm_moe_slot_pool_apply_slot_direct_batch(
            params,
            va_space,
            first_error,
            NULL,
            0,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NV_FALSE,
            NV_FALSE,
            NV_FALSE,
            NV_FALSE,
            NV_FALSE,
            NULL);
    }

    if (params->mode == UVM_MOE_SLOT_REMAP_MODE_REMAP) {
        NV_STATUS apply_status = NV_OK;

        for (i = 0; i < params->count; ++i) {
            UVM_MOE_SLOT_REMAP_ENTRY *entry = &params->entries[i];
            NV_STATUS status;

            if (entry->status != UVM_MOE_SLOT_REMAP_ENTRY_OK)
                continue;

            status = uvm_moe_slot_remap_migrate_entry(params, entry, filp);
            if (status == NV_OK) {
                ++params->remap_count;
                ++params->tlb_invalidate_count;
            }
            else {
                entry->status = UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED;
                ++params->failure_count;
                if (first_error == NV_OK)
                    first_error = status;
            }

            uvm_phase_f_route_log("moe_slot_remap_apply index=%u layer=%u expert=%u slot=%u va=0x%llx bytes=%llu status=%u nv_status=0x%x remap_count=%u tlb_invalidate_count=%u\n",
                                  i,
                                  entry->layer_id,
                                  entry->expert_id,
                                  entry->slot_id,
                                  entry->va_base,
                                  entry->bytes,
                                  entry->status,
                                  status,
                                  params->remap_count,
                                  params->tlb_invalidate_count);
        }

        if (uvm_moe_slot_remap_has_ok_entries(params))
            apply_status = uvm_moe_slot_pool_apply_batch(params,
                                                         UVM_MOE_SLOT_POOL_COPY_NONE,
                                                         NULL,
                                                         va_space);

        if (first_error == NV_OK && apply_status != NV_OK)
            first_error = apply_status;
    }

    if (params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_PTE_REMAP ||
        params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP ||
        params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_USER_BACKING_REMAP) {
        NV_STATUS apply_status = NV_OK;
        NvBool use_va_block_path = params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP ||
                                   params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_USER_BACKING_REMAP;
        NvBool use_user_backing_path =
            params->mode == UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_USER_BACKING_REMAP;

        if (uvm_moe_slot_remap_has_ok_entries(params)) {
            apply_status = uvm_moe_slot_pool_retain_gpu_by_uuid(&params->gpu_uuid, &gpu);
            if (apply_status != NV_OK) {
                uvm_moe_slot_remap_fail_ok_entries(params, UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED);
            }
        }

        if (apply_status == NV_OK &&
            use_va_block_path &&
            first_error != NV_OK) {
            uvm_moe_slot_remap_fail_ok_entries(params, UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED);
            apply_status = first_error;
        }

        if (apply_status == NV_OK)
            apply_status = uvm_moe_slot_pool_apply_slot_pte_batch(
                params,
                va_space,
                gpu,
                use_va_block_path,
                use_user_backing_path);

        if (gpu) {
            uvm_gpu_release(gpu);
            gpu = NULL;
        }

        if (first_error == NV_OK && apply_status != NV_OK)
            first_error = apply_status;
    }

    if (params->mode == UVM_MOE_SLOT_REMAP_MODE_VALIDATE ||
        params->mode == UVM_MOE_SLOT_REMAP_MODE_COPY_ONLY ||
        params->mode == UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ONLY ||
        params->mode == UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC ||
        params->mode == UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ASYNC_SUBMIT) {
        NV_STATUS apply_status = NV_OK;
        uvm_moe_slot_pool_copy_mode_t copy_mode = UVM_MOE_SLOT_POOL_COPY_NONE;

        if (params->mode == UVM_MOE_SLOT_REMAP_MODE_COPY_ONLY)
            copy_mode = UVM_MOE_SLOT_POOL_COPY_SYSMEM;
        else if (params->mode == UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ONLY)
            copy_mode = UVM_MOE_SLOT_POOL_COPY_GPU;
        else if (params->mode == UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC)
            copy_mode = UVM_MOE_SLOT_POOL_PREALLOC_VIDMEM;
        else if (params->mode == UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ASYNC_SUBMIT)
            copy_mode = UVM_MOE_SLOT_POOL_COPY_GPU_ASYNC;

        if ((copy_mode == UVM_MOE_SLOT_POOL_COPY_GPU ||
             copy_mode == UVM_MOE_SLOT_POOL_PREALLOC_VIDMEM ||
             copy_mode == UVM_MOE_SLOT_POOL_COPY_GPU_ASYNC) &&
            uvm_moe_slot_remap_has_ok_entries(params)) {
            NvU64 retain_start = NV_GETTIME();

            if (copy_mode == UVM_MOE_SLOT_POOL_PREALLOC_VIDMEM) {
                // The slot belongs to this already-registered VA space. Avoid
                // the full RM GetGpuInfo path used by the legacy copy modes.
                gpu = uvm_va_space_retain_gpu_by_uuid(va_space, &params->gpu_uuid);
                apply_status = gpu ? NV_OK : NV_ERR_INVALID_DEVICE;
                uvm_phase_f_route_log("moe_slot_pool_fast_gpu_retain mode=%u owner=%px ns=%llu status=0x%x\n",
                                      params->mode,
                                      va_space,
                                      NV_GETTIME() - retain_start,
                                      apply_status);
            }
            else {
                apply_status = uvm_moe_slot_pool_retain_gpu_by_uuid(&params->gpu_uuid, &gpu);
            }
            if (apply_status != NV_OK) {
                uvm_moe_slot_remap_fail_ok_entries(params, UVM_MOE_SLOT_REMAP_ENTRY_COPY_FAILED);
            }
        }

        if (apply_status == NV_OK) {
            apply_status = uvm_moe_slot_pool_apply_batch(params,
                                                         copy_mode,
                                                         gpu,
                                                         va_space);
        }

        if (gpu) {
            uvm_gpu_release(gpu);
            gpu = NULL;
        }

        if (first_error == NV_OK && apply_status != NV_OK)
            first_error = apply_status;
    }

    uvm_phase_f_route_log("moe_slot_remap_batch end mode=%u count=%u success=%u failure=%u conflicts=%u requested_bytes=%llu accepted_bytes=%llu remap_count=%u tlb_invalidate_count=%u first_error=0x%x\n",
                          params->mode,
                          params->count,
                          params->success_count,
                          params->failure_count,
                          params->conflict_count,
                          params->requested_bytes,
                          params->accepted_bytes,
                          params->remap_count,
                          params->tlb_invalidate_count,
                          first_error);

    mutex_unlock(&g_uvm_moe_slot_plan_lock);
    return first_error;
}

static NV_STATUS uvm_api_update_region_state(UVM_UPDATE_REGION_STATE_PARAMS *params,
                                             struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_phase_f_region_t *region;

    if (params->region_id == 0 || params->state > UVM_REGION_STATE_PREFETCHING_GPU)
        return NV_ERR_INVALID_ARGUMENT;

    uvm_va_space_down_write(va_space);
    region = radix_tree_lookup(&va_space->phase_f_regions, params->region_id);
    if (region)
        region->state = params->state;
    uvm_va_space_up_write(va_space);

    return region ? NV_OK : NV_ERR_OBJECT_NOT_FOUND;
}

static NV_STATUS uvm_api_query_region(UVM_QUERY_REGION_PARAMS *params,
                                      struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_phase_f_region_t *region;

    if (params->region_id == 0)
        return NV_ERR_INVALID_ARGUMENT;

    params->base = 0;
    params->length = 0;
    params->state = UVM_REGION_STATE_UNKNOWN;
    params->found_by_va = 0;

    uvm_va_space_down_read(va_space);
    region = radix_tree_lookup(&va_space->phase_f_regions, params->region_id);
    if (region)
        uvm_copy_region_to_query(region, params, 0);
    uvm_va_space_up_read(va_space);

    return region ? NV_OK : NV_ERR_OBJECT_NOT_FOUND;
}

static NV_STATUS uvm_api_query_region_by_va(UVM_QUERY_REGION_BY_VA_PARAMS *params,
                                            struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_phase_f_region_t *region;
    uvm_range_tree_node_t *node;

    if (params->address == 0)
        return NV_ERR_INVALID_ARGUMENT;

    params->region_id = 0;
    params->base = 0;
    params->length = 0;
    params->state = UVM_REGION_STATE_UNKNOWN;
    params->found_by_va = 0;

    uvm_va_space_down_read(va_space);
    node = uvm_range_tree_find(&va_space->phase_f_region_ranges, params->address);
    region = uvm_phase_f_region_from_node(node);
    if (region)
        uvm_copy_region_to_query_by_va(region, params);
    uvm_va_space_up_read(va_space);

    return region ? NV_OK : NV_ERR_OBJECT_NOT_FOUND;
}

static long uvm_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    switch (cmd)
    {
        case UVM_DEINITIALIZE:
            return 0;

        UVM_ROUTE_CMD_STACK_NO_INIT_CHECK(UVM_INITIALIZE,                  uvm_api_initialize);
        UVM_ROUTE_CMD_STACK_NO_INIT_CHECK(UVM_MM_INITIALIZE,               uvm_api_mm_initialize);

        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_PAGEABLE_MEM_ACCESS,            uvm_api_pageable_mem_access);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_PAGEABLE_MEM_ACCESS_ON_GPU,     uvm_api_pageable_mem_access_on_gpu);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_REGISTER_GPU,                   uvm_api_register_gpu);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_UNREGISTER_GPU,                 uvm_api_unregister_gpu);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_CREATE_RANGE_GROUP,             uvm_api_create_range_group);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_DESTROY_RANGE_GROUP,            uvm_api_destroy_range_group);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_ENABLE_PEER_ACCESS,             uvm_api_enable_peer_access);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_DISABLE_PEER_ACCESS,            uvm_api_disable_peer_access);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_SET_RANGE_GROUP,                uvm_api_set_range_group);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_CREATE_EXTERNAL_RANGE,          uvm_api_create_external_range);
        UVM_ROUTE_CMD_ALLOC_INIT_CHECK(UVM_MAP_EXTERNAL_ALLOCATION,        uvm_api_map_external_allocation);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_MAP_EXTERNAL_SPARSE,            uvm_api_map_external_sparse);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_FREE,                           uvm_api_free);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_DISCARD,                        uvm_api_discard);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_PREVENT_MIGRATION_RANGE_GROUPS, uvm_api_prevent_migration_range_groups);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_ALLOW_MIGRATION_RANGE_GROUPS,   uvm_api_allow_migration_range_groups);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_SET_PREFERRED_LOCATION,         uvm_api_set_preferred_location);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_UNSET_PREFERRED_LOCATION,       uvm_api_unset_preferred_location);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_SET_ACCESSED_BY,                uvm_api_set_accessed_by);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_UNSET_ACCESSED_BY,              uvm_api_unset_accessed_by);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_REGISTER_GPU_VASPACE,           uvm_api_register_gpu_va_space);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_UNREGISTER_GPU_VASPACE,         uvm_api_unregister_gpu_va_space);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_REGISTER_CHANNEL,               uvm_api_register_channel);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_UNREGISTER_CHANNEL,             uvm_api_unregister_channel);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_ENABLE_READ_DUPLICATION,        uvm_api_enable_read_duplication);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_DISABLE_READ_DUPLICATION,       uvm_api_disable_read_duplication);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_MIGRATE,                        uvm_api_migrate);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TOOLS_READ_PROCESS_MEMORY,      uvm_api_tools_read_process_memory);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TOOLS_WRITE_PROCESS_MEMORY,     uvm_api_tools_write_process_memory);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TOOLS_GET_PROCESSOR_UUID_TABLE, uvm_api_tools_get_processor_uuid_table);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_MAP_DYNAMIC_PARALLELISM_REGION, uvm_api_map_dynamic_parallelism_region);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_UNMAP_EXTERNAL,                 uvm_api_unmap_external);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_MIGRATE_RANGE_GROUP,            uvm_api_migrate_range_group);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TOOLS_FLUSH_EVENTS,             uvm_api_tools_flush_events);
        UVM_ROUTE_CMD_ALLOC_INIT_CHECK(UVM_ALLOC_SEMAPHORE_POOL,           uvm_api_alloc_semaphore_pool);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_CLEAN_UP_ZOMBIE_RESOURCES,      uvm_api_clean_up_zombie_resources);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_POPULATE_PAGEABLE,              uvm_api_populate_pageable);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_VALIDATE_VA_RANGE,              uvm_api_validate_va_range);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_V2,uvm_api_tools_get_processor_uuid_table_v2);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_ALLOC_DEVICE_P2P,               uvm_api_alloc_device_p2p);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_CLEAR_ALL_ACCESS_COUNTERS,      uvm_api_clear_all_access_counters);
        UVM_ROUTE_CMD_ALLOC_INIT_CHECK(UVM_PHASE_F_BATCH_EXECUTOR,         uvm_api_phase_f_batch_executor);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_PHASE_F_REGISTER_REGION,        uvm_api_phase_f_register_region);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_PHASE_F_UNREGISTER_REGION,      uvm_api_phase_f_unregister_region);
        UVM_ROUTE_CMD_ALLOC_INIT_CHECK(UVM_PHASE_F_BATCH_REGIONS,          uvm_api_phase_f_batch_regions);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_UPDATE_REGION_STATE,             uvm_api_update_region_state);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_QUERY_REGION,                    uvm_api_query_region);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_QUERY_REGION_BY_VA,              uvm_api_query_region_by_va);
        UVM_ROUTE_CMD_ALLOC_INIT_CHECK(UVM_PHASE_F_PREFETCH_GPU_BATCH_V2,    uvm_api_phase_f_prefetch_gpu_batch_v2);
        UVM_ROUTE_CMD_HEAP_INIT_CHECK(UVM_EXPERT_OFFLOAD_RELEASE,            uvm_api_expert_offload_release);
        UVM_ROUTE_CMD_ALLOC_INIT_CHECK(UVM_MOE_SLOT_REMAP_BATCH,            uvm_api_moe_slot_remap_batch);
        UVM_ROUTE_CMD_ALLOC_INIT_CHECK(UVM_MOE_SLOT_PLAN_REGISTER,           uvm_api_moe_slot_plan_register);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_MOE_SLOT_PLAN_EXECUTE,            uvm_api_moe_slot_plan_execute);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_MOE_SLOT_PLAN_EXECUTE_FENCED,     uvm_api_moe_slot_plan_execute_fenced);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_MOE_SLOT_PLAN_DESTROY,            uvm_api_moe_slot_plan_destroy);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_MOE_SLOT_SESSION_ABORT,           uvm_api_moe_slot_session_abort);
        UVM_ROUTE_CMD_ALLOC_INIT_CHECK(UVM_MOE_STATIC_ALIAS_MAP,             uvm_api_moe_static_alias_map);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_MOE_STATIC_ALIAS_SEAL,            uvm_api_moe_static_alias_seal);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_MOE_STATIC_ALIAS_DESTROY,         uvm_api_moe_static_alias_destroy);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_MOE_STATIC_ALIAS_PARK,            uvm_api_moe_static_alias_park);
        UVM_ROUTE_CMD_STACK_INIT_CHECK(UVM_MOE_STATIC_ALIAS_RESUME,          uvm_api_moe_static_alias_resume);
        UVM_ROUTE_CMD_ALLOC_INIT_CHECK(UVM_MOE_STATIC_ALIAS_REBIND,           uvm_api_moe_static_alias_rebind);
        UVM_ROUTE_CMD_ALLOC_INIT_CHECK(UVM_MOE_STATIC_ALIAS_QUIESCED_REMOVE,  uvm_api_moe_static_alias_quiesced_remove);
    }

    // Try the test ioctls if none of the above matched
    return uvm_test_ioctl(filp, cmd, arg);
}

static long uvm_unlocked_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    long ret;

    if (!uvm_down_read_trylock(&g_uvm_global.pm.lock))
        return -EAGAIN;

    ret = uvm_ioctl(filp, cmd, arg);

    uvm_up_read(&g_uvm_global.pm.lock);

    uvm_thread_assert_all_unlocked();

    return ret;
}

static long uvm_unlocked_ioctl_entry(struct file *filp, unsigned int cmd, unsigned long arg)
{
   UVM_ENTRY_RET(uvm_unlocked_ioctl(filp, cmd, arg));
}

static const struct file_operations uvm_fops =
{
    .open            = uvm_open_entry,
    .release         = uvm_release_entry,
    .mmap            = uvm_mmap_entry,
    .unlocked_ioctl  = uvm_unlocked_ioctl_entry,
#if NVCPU_IS_X86_64
    .compat_ioctl    = uvm_unlocked_ioctl_entry,
#endif
    .owner           = THIS_MODULE,
};

NV_STATUS uvm_test_register_unload_state_buffer(UVM_TEST_REGISTER_UNLOAD_STATE_BUFFER_PARAMS *params, struct file *filp)
{
    long ret;
    struct page *page;
    NV_STATUS status = NV_OK;

    if (!IS_ALIGNED(params->unload_state_buf, sizeof(NvU64)))
        return NV_ERR_INVALID_ADDRESS;

    // Hold mmap_lock to call get_user_pages(), the UVM locking helper functions
    // are not used because unload_state_buf may be a managed memory pointer and
    // therefore a locking assertion from the CPU fault handler could be fired.
    nv_mmap_read_lock(current->mm);
    ret = NV_PIN_USER_PAGES(params->unload_state_buf, 1, FOLL_WRITE, &page);
    nv_mmap_read_unlock(current->mm);

    if (ret < 0)
        return errno_to_nv_status(ret);
    UVM_ASSERT(ret == 1);

    uvm_mutex_lock(&g_uvm_global.global_lock);

    if (g_uvm_global.unload_state.ptr) {
        NV_UNPIN_USER_PAGE(page);
        status = NV_ERR_IN_USE;
        goto error;
    }

    g_uvm_global.unload_state.page = page;
    g_uvm_global.unload_state.ptr = (NvU64 *)((char *)kmap(page) + (params->unload_state_buf & ~PAGE_MASK));
    *g_uvm_global.unload_state.ptr = 0;

error:
    uvm_mutex_unlock(&g_uvm_global.global_lock);

    return status;
}

static void uvm_test_unload_state_exit(void)
{
    if (g_uvm_global.unload_state.ptr) {
        kunmap(g_uvm_global.unload_state.page);
        NV_UNPIN_USER_PAGE(g_uvm_global.unload_state.page);
    }
}

static int uvm_chardev_create(void)
{
    dev_t uvm_dev;

    int ret = alloc_chrdev_region(&g_uvm_base_dev,
                                  0,
                                  NVIDIA_UVM_NUM_MINOR_DEVICES,
                                  NVIDIA_UVM_DEVICE_NAME);
    if (ret != 0) {
        UVM_ERR_PRINT("alloc_chrdev_region failed: %d\n", ret);
        return ret;
    }
    uvm_dev = MKDEV(MAJOR(g_uvm_base_dev), NVIDIA_UVM_PRIMARY_MINOR_NUMBER);

    uvm_init_character_device(&g_uvm_cdev, &uvm_fops);
    ret = cdev_add(&g_uvm_cdev, uvm_dev, 1);
    if (ret != 0) {
        UVM_ERR_PRINT("cdev_add (major %u, minor %u) failed: %d\n", MAJOR(uvm_dev), MINOR(uvm_dev), ret);
        unregister_chrdev_region(g_uvm_base_dev, NVIDIA_UVM_NUM_MINOR_DEVICES);
        return ret;
    }

    return 0;
}

static void uvm_chardev_exit(void)
{
    cdev_del(&g_uvm_cdev);
    unregister_chrdev_region(g_uvm_base_dev, NVIDIA_UVM_NUM_MINOR_DEVICES);
}

static int uvm_init(void)
{
    bool initialized_globals = false;
    bool added_device = false;
    int ret;

    NV_STATUS status = uvm_global_init();
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_global_init() failed: %s\n", nvstatusToString(status));
        ret = -ENODEV;
        goto error;
    }
    initialized_globals = true;

    ret = uvm_chardev_create();
    if (ret != 0) {
        UVM_ERR_PRINT("uvm_chardev_create failed: %d\n", ret);
        goto error;
    }
    added_device = true;

    ret = uvm_tools_init(g_uvm_base_dev);
    if (ret != 0) {
        UVM_ERR_PRINT("uvm_tools_init() failed: %d\n", ret);
        goto error;
    }

    if (uvm_enable_builtin_tests)
        UVM_INFO_PRINT("Built-in UVM tests are enabled. This is a security risk.\n");

    return 0;

error:
    if (added_device)
        uvm_chardev_exit();

    if (initialized_globals)
        uvm_global_exit();

    UVM_ERR_PRINT("uvm init failed: %d\n", ret);

    return ret;
}

static int __init uvm_init_entry(void)
{
   UVM_ENTRY_RET(uvm_init());
}

static void uvm_exit(void)
{
    uvm_tools_exit();
    uvm_chardev_exit();

    uvm_moe_slot_pool_reset();
    uvm_global_exit();

    uvm_test_unload_state_exit();
}

static void __exit uvm_exit_entry(void)
{
   UVM_ENTRY_VOID(uvm_exit());
}

module_init(uvm_init_entry);
module_exit(uvm_exit_entry);

MODULE_LICENSE("Dual MIT/GPL");
MODULE_INFO(supported, "external");
MODULE_VERSION(NV_VERSION_STRING);
MODULE_DESCRIPTION("NVIDIA Unified Virtual Memory kernel module");
