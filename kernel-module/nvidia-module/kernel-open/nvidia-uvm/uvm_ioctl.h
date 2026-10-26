/*******************************************************************************
    Copyright (c) 2013-2025 NVidia Corporation

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

#ifndef _UVM_IOCTL_H
#define _UVM_IOCTL_H

#include "uvm_types.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// Please see the header file (uvm.h) for detailed documentation on each of the
// associated API calls.
//

#if defined(WIN32) || defined(WIN64)
#   define UVM_IOCTL_BASE(i)       CTL_CODE(FILE_DEVICE_UNKNOWN, 0x800+i, METHOD_BUFFERED, FILE_READ_DATA | FILE_WRITE_DATA)
#else
#   define UVM_IOCTL_BASE(i) i
#endif

//
// UvmReserveVa
//
#define UVM_RESERVE_VA                                                UVM_IOCTL_BASE(1)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_RESERVE_VA_PARAMS;

//
// UvmReleaseVa
//
#define UVM_RELEASE_VA                                                UVM_IOCTL_BASE(2)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_RELEASE_VA_PARAMS;

//
// UvmRegionCommit
//
#define UVM_REGION_COMMIT                                             UVM_IOCTL_BASE(3)

typedef struct
{
    NvU64           requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64           length        NV_ALIGN_BYTES(8); // IN
    UvmStream       streamId      NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid gpuUuid;                         // IN
    NV_STATUS       rmStatus;                        // OUT
} UVM_REGION_COMMIT_PARAMS;

//
// UvmRegionDecommit
//
#define UVM_REGION_DECOMMIT                                           UVM_IOCTL_BASE(4)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_REGION_DECOMMIT_PARAMS;

//
// UvmRegionSetStream
//
#define UVM_REGION_SET_STREAM                                         UVM_IOCTL_BASE(5)

typedef struct
{
    NvU64           requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64           length        NV_ALIGN_BYTES(8); // IN
    UvmStream       newStreamId   NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid gpuUuid;                         // IN
    NV_STATUS       rmStatus;                        // OUT
} UVM_REGION_SET_STREAM_PARAMS;

//
// UvmSetStreamRunning
//
#define UVM_SET_STREAM_RUNNING                                        UVM_IOCTL_BASE(6)

typedef struct
{
    UvmStream  streamId NV_ALIGN_BYTES(8);  // IN
    NV_STATUS  rmStatus;                    // OUT
} UVM_SET_STREAM_RUNNING_PARAMS;


//
// Due to limitations in how much we want to send per ioctl call, the nStreams
// member must be less than or equal to about 250. That's an upper limit.
//
// However, from a typical user-space driver's point of view (for example, the
// CUDA driver), a vast majority of the time, we expect there to be only one
// stream passed in. The second most common case is something like atmost 32
// streams being passed in. The cases where there are more than 32 streams are
// the most rare. So we might want to optimize the ioctls accordingly so that we
// don't always copy a 250 * sizeof(streamID) sized array when there's only one
// or a few streams.
//
// For that reason, UVM_MAX_STREAMS_PER_IOCTL_CALL is set to 32.
//
// If the higher-level (uvm.h) call requires more streams to be stopped than
// this value, then multiple ioctl calls should be made.
//
#define UVM_MAX_STREAMS_PER_IOCTL_CALL 32

//
// UvmSetStreamStopped
//
#define UVM_SET_STREAM_STOPPED                                        UVM_IOCTL_BASE(7)

typedef struct
{
    UvmStream streamIdArray[UVM_MAX_STREAMS_PER_IOCTL_CALL] NV_ALIGN_BYTES(8); // IN
    NvU64     nStreams                                      NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                                        // OUT
} UVM_SET_STREAM_STOPPED_PARAMS;

//
// UvmAddSession
//
#define UVM_ADD_SESSION                                               UVM_IOCTL_BASE(10)

typedef struct
{
    NvU32        pidTarget;                             // IN
    NV_STATUS    rmStatus;                              // OUT
} UVM_ADD_SESSION_PARAMS;

#define UVM_MAX_COUNTERS_PER_IOCTL_CALL 32

//
// UvmEnableCounters
//
#define UVM_ENABLE_COUNTERS                                           UVM_IOCTL_BASE(12)

typedef struct
{
    UvmCounterConfig config[UVM_MAX_COUNTERS_PER_IOCTL_CALL]; // IN
    NvU32            count;                                   // IN
    NV_STATUS        rmStatus;                                // OUT
} UVM_ENABLE_COUNTERS_PARAMS;

//
// UvmMapCounter
//
#define UVM_MAP_COUNTER                                               UVM_IOCTL_BASE(13)

typedef struct
{
    NvU32           scope;                          // IN (UvmCounterScope)
    NvU32           counterName;                    // IN (UvmCounterName)
    NvProcessorUuid gpuUuid;                        // IN
    NvP64           addr         NV_ALIGN_BYTES(8); // OUT
    NV_STATUS       rmStatus;                       // OUT
} UVM_MAP_COUNTER_PARAMS;

//
// UvmCreateEventQueue
//
#define UVM_CREATE_EVENT_QUEUE                                        UVM_IOCTL_BASE(14)

typedef struct
{
    NvU32                 eventQueueIndex;                      // OUT
    NvU64                 queueSize          NV_ALIGN_BYTES(8); // IN
    NvU64                 notificationCount  NV_ALIGN_BYTES(8); // IN
    NvU64                 notificationHandle NV_ALIGN_BYTES(8); // IN
    NvU32                 timeStampType;                        // IN (UvmEventTimeStampType)
    NV_STATUS             rmStatus;                             // OUT
} UVM_CREATE_EVENT_QUEUE_PARAMS;

//
// UvmRemoveEventQueue
//
#define UVM_REMOVE_EVENT_QUEUE                                        UVM_IOCTL_BASE(15)

typedef struct
{
    NvU32         eventQueueIndex;    // IN
    NV_STATUS     rmStatus;           // OUT
} UVM_REMOVE_EVENT_QUEUE_PARAMS;

//
// UvmMapEventQueue
//
#define UVM_MAP_EVENT_QUEUE                                           UVM_IOCTL_BASE(16)

typedef struct
{
    NvU32         eventQueueIndex;                    // IN
    NvP64         userRODataAddr   NV_ALIGN_BYTES(8); // IN
    NvP64         userRWDataAddr   NV_ALIGN_BYTES(8); // IN
    NvP64         readIndexAddr    NV_ALIGN_BYTES(8); // OUT
    NvP64         writeIndexAddr   NV_ALIGN_BYTES(8); // OUT
    NvP64         queueBufferAddr  NV_ALIGN_BYTES(8); // OUT
    NV_STATUS     rmStatus;                           // OUT
} UVM_MAP_EVENT_QUEUE_PARAMS;

//
// UvmEnableEvent
//
#define UVM_EVENT_CTRL                                                UVM_IOCTL_BASE(17)

typedef struct
{
    NvU32        eventQueueIndex;   // IN
    NvS32        eventType;         // IN
    NvU32        enable;            // IN
    NV_STATUS    rmStatus;          // OUT
} UVM_EVENT_CTRL_PARAMS;

//
// UvmEventGetGpuUuidTable
//
#define UVM_GET_GPU_UUID_TABLE                                        UVM_IOCTL_BASE(20)

typedef struct
{
    NvProcessorUuid gpuUuidArray[UVM_MAX_GPUS_V1]; // OUT
    NvU32           validCount;                    // OUT
    NV_STATUS       rmStatus;                      // OUT
} UVM_GET_GPU_UUID_TABLE_PARAMS;

#if defined(WIN32) || defined(WIN64)
//
// UvmRegionSetBacking
//
#define UVM_REGION_SET_BACKING                                        UVM_IOCTL_BASE(21)

typedef struct
{
    NvProcessorUuid gpuUuid;                        // IN
    NvU32           hAllocation;                    // IN
    NvP64           vaAddr       NV_ALIGN_BYTES(8); // IN
    NvU64           regionLength NV_ALIGN_BYTES(8); // IN
    NV_STATUS       rmStatus;                       // OUT
} UVM_REGION_SET_BACKING_PARAMS;

//
// UvmRegionUnsetBacking
//
#define UVM_REGION_UNSET_BACKING                                      UVM_IOCTL_BASE(22)

typedef struct
{
    NvP64     vaAddr       NV_ALIGN_BYTES(8); // IN
    NvU64     regionLength NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                       // OUT
} UVM_REGION_UNSET_BACKING_PARAMS;

#endif

#define UVM_CREATE_RANGE_GROUP                                        UVM_IOCTL_BASE(23)

typedef struct
{
    NvU64     rangeGroupId NV_ALIGN_BYTES(8); // OUT
    NV_STATUS rmStatus;                       // OUT
} UVM_CREATE_RANGE_GROUP_PARAMS;

#define UVM_DESTROY_RANGE_GROUP                                       UVM_IOCTL_BASE(24)

typedef struct
{
    NvU64     rangeGroupId NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                      // OUT
} UVM_DESTROY_RANGE_GROUP_PARAMS;

//
// UvmRegisterGpuVaSpace
//
#define UVM_REGISTER_GPU_VASPACE                                      UVM_IOCTL_BASE(25)

typedef struct
{
    NvProcessorUuid gpuUuid;  // IN
    NvS32           rmCtrlFd; // IN
    NvHandle        hClient;  // IN
    NvHandle        hVaSpace; // IN
    NV_STATUS       rmStatus; // OUT
} UVM_REGISTER_GPU_VASPACE_PARAMS;

//
// UvmUnregisterGpuVaSpace
//
#define UVM_UNREGISTER_GPU_VASPACE                                    UVM_IOCTL_BASE(26)

typedef struct
{
    NvProcessorUuid gpuUuid;  // IN
    NV_STATUS       rmStatus; // OUT
} UVM_UNREGISTER_GPU_VASPACE_PARAMS;

//
// UvmRegisterChannel
//
#define UVM_REGISTER_CHANNEL                                          UVM_IOCTL_BASE(27)

typedef struct
{
    NvProcessorUuid gpuUuid;                     // IN
    NvS32           rmCtrlFd;                    // IN
    NvHandle        hClient;                     // IN
    NvHandle        hChannel;                    // IN
    NvU64           base      NV_ALIGN_BYTES(8); // IN
    NvU64           length    NV_ALIGN_BYTES(8); // IN
    NV_STATUS       rmStatus;                    // OUT
} UVM_REGISTER_CHANNEL_PARAMS;

//
// UvmUnregisterChannel
//
#define UVM_UNREGISTER_CHANNEL                                        UVM_IOCTL_BASE(28)

typedef struct
{
    NvHandle        hClient;  // IN
    NvHandle        hChannel; // IN
    NV_STATUS       rmStatus; // OUT
} UVM_UNREGISTER_CHANNEL_PARAMS;

//
// UvmEnablePeerAccess
//
#define UVM_ENABLE_PEER_ACCESS                                       UVM_IOCTL_BASE(29)

typedef struct
{
    NvProcessorUuid gpuUuidA; // IN
    NvProcessorUuid gpuUuidB; // IN
    NV_STATUS  rmStatus; // OUT
} UVM_ENABLE_PEER_ACCESS_PARAMS;

//
// UvmDisablePeerAccess
//
#define UVM_DISABLE_PEER_ACCESS                                      UVM_IOCTL_BASE(30)

typedef struct
{
    NvProcessorUuid gpuUuidA; // IN
    NvProcessorUuid gpuUuidB; // IN
    NV_STATUS  rmStatus; // OUT
} UVM_DISABLE_PEER_ACCESS_PARAMS;

//
// UvmSetRangeGroup
//
#define UVM_SET_RANGE_GROUP                                           UVM_IOCTL_BASE(31)

typedef struct
{
    NvU64     rangeGroupId  NV_ALIGN_BYTES(8); // IN
    NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_SET_RANGE_GROUP_PARAMS;

//
// UvmMapExternalAllocation
//
#define UVM_MAP_EXTERNAL_ALLOCATION                                   UVM_IOCTL_BASE(33)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8); // IN
    NvU64                   length                          NV_ALIGN_BYTES(8); // IN
    NvU64                   offset                          NV_ALIGN_BYTES(8); // IN
    UvmGpuMappingAttributes perGpuAttributes[UVM_MAX_GPUS];                    // IN
    NvU64                   gpuAttributesCount              NV_ALIGN_BYTES(8); // IN
    NvS32                   rmCtrlFd;                                          // IN
    NvU32                   hClient;                                           // IN
    NvU32                   hMemory;                                           // IN

    NV_STATUS               rmStatus;                                          // OUT
} UVM_MAP_EXTERNAL_ALLOCATION_PARAMS;

//
// UvmFree
//
#define UVM_FREE                                                      UVM_IOCTL_BASE(34)
typedef struct
{
    NvU64     base      NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                    // OUT
} UVM_FREE_PARAMS;

//
// UvmMemMap
//
#define UVM_MEM_MAP                                                   UVM_IOCTL_BASE(35)

typedef struct
{
    NvP64     regionBase   NV_ALIGN_BYTES(8); // IN
    NvU64     regionLength NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                       // OUT
} UVM_MEM_MAP_PARAMS;

//
// UvmRegisterGpu
//
#define UVM_REGISTER_GPU                                              UVM_IOCTL_BASE(37)

typedef struct
{
    NvProcessorUuid gpu_uuid;    // IN/OUT
    NvBool          numaEnabled; // OUT
    NvS32           numaNodeId;  // OUT
    NvS32           rmCtrlFd;    // IN
    NvHandle        hClient;     // IN
    NvHandle        hSmcPartRef; // IN
    NV_STATUS       rmStatus;    // OUT
} UVM_REGISTER_GPU_PARAMS;

//
// UvmUnregisterGpu
//
#define UVM_UNREGISTER_GPU                                            UVM_IOCTL_BASE(38)

typedef struct
{
    NvProcessorUuid gpu_uuid; // IN
    NV_STATUS       rmStatus; // OUT
} UVM_UNREGISTER_GPU_PARAMS;

#define UVM_PAGEABLE_MEM_ACCESS                                       UVM_IOCTL_BASE(39)

typedef struct
{
    NvBool    pageableMemAccess; // OUT
    NV_STATUS rmStatus;          // OUT
} UVM_PAGEABLE_MEM_ACCESS_PARAMS;

//
// Due to limitations in how much we want to send per ioctl call, the numGroupIds
// member must be less than or equal to about 250. That's an upper limit.
//
// However, from a typical user-space driver's point of view (for example, the
// CUDA driver), a vast majority of the time, we expect there to be only one
// range group passed in. The second most common case is something like atmost 32
// range groups being passed in. The cases where there are more than 32 range
// groups are the most rare. So we might want to optimize the ioctls accordingly
// so that we don't always copy a 250 * sizeof(NvU64) sized array when there's
// only one or a few range groups.
//
// For that reason, UVM_MAX_RANGE_GROUPS_PER_IOCTL_CALL is set to 32.
//
// If the higher-level (uvm.h) call requires more range groups than
// this value, then multiple ioctl calls should be made.
//
#define UVM_MAX_RANGE_GROUPS_PER_IOCTL_CALL 32

//
// UvmPreventMigrationRangeGroups
//
#define UVM_PREVENT_MIGRATION_RANGE_GROUPS                            UVM_IOCTL_BASE(40)

typedef struct
{
    NvU64     rangeGroupIds[UVM_MAX_RANGE_GROUPS_PER_IOCTL_CALL] NV_ALIGN_BYTES(8); // IN
    NvU64     numGroupIds                                        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                                             // OUT
} UVM_PREVENT_MIGRATION_RANGE_GROUPS_PARAMS;

//
// UvmAllowMigrationRangeGroups
//
#define UVM_ALLOW_MIGRATION_RANGE_GROUPS                              UVM_IOCTL_BASE(41)

typedef struct
{
    NvU64     rangeGroupIds[UVM_MAX_RANGE_GROUPS_PER_IOCTL_CALL] NV_ALIGN_BYTES(8); // IN
    NvU64     numGroupIds                                        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                                             // OUT
} UVM_ALLOW_MIGRATION_RANGE_GROUPS_PARAMS;

//
// UvmSetPreferredLocation
//
#define UVM_SET_PREFERRED_LOCATION                                    UVM_IOCTL_BASE(42)

typedef struct
{
    NvU64           requestedBase      NV_ALIGN_BYTES(8); // IN
    NvU64           length             NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid preferredLocation;                    // IN
    NvS32           preferredCpuNumaNode;                 // IN
    NV_STATUS       rmStatus;                             // OUT
} UVM_SET_PREFERRED_LOCATION_PARAMS;

//
// UvmUnsetPreferredLocation
//
#define UVM_UNSET_PREFERRED_LOCATION                                  UVM_IOCTL_BASE(43)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_UNSET_PREFERRED_LOCATION_PARAMS;

//
// UvmEnableReadDuplication
//
#define UVM_ENABLE_READ_DUPLICATION                                   UVM_IOCTL_BASE(44)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_ENABLE_READ_DUPLICATION_PARAMS;

//
// UvmDisableReadDuplication
//
#define UVM_DISABLE_READ_DUPLICATION                                  UVM_IOCTL_BASE(45)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_DISABLE_READ_DUPLICATION_PARAMS;

//
// UvmSetAccessedBy
//
#define UVM_SET_ACCESSED_BY                                           UVM_IOCTL_BASE(46)

typedef struct
{
    NvU64           requestedBase   NV_ALIGN_BYTES(8); // IN
    NvU64           length          NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid accessedByUuid;                    // IN
    NV_STATUS       rmStatus;                          // OUT
} UVM_SET_ACCESSED_BY_PARAMS;

//
// UvmUnsetAccessedBy
//
#define UVM_UNSET_ACCESSED_BY                                         UVM_IOCTL_BASE(47)

typedef struct
{
    NvU64           requestedBase   NV_ALIGN_BYTES(8); // IN
    NvU64           length          NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid accessedByUuid;                    // IN
    NV_STATUS       rmStatus;                          // OUT
} UVM_UNSET_ACCESSED_BY_PARAMS;

// For managed allocations, UVM_MIGRATE implements the behavior described in
// UvmMigrate. If the input virtual range corresponds to system-allocated
// pageable memory, and the GPUs in the system support transparent access to
// pageable memory, the scheme is a bit more elaborate, potentially with
// several transitions betwen user and kernel spaces:
//
// 1) UVM_MIGRATE with the range base address and size. This will migrate
// anonymous vmas until:
//   a) It finds a file-backed vma or no GPUs are registered in the VA space
//      so no GPU can drive the copy. It will try to populate the vma using
//      get_user_pages and return NV_WARN_NOTHING_TO_DO.
//   b) It fails to allocate memory on the destination CPU node. It will return
//       NV_ERR_MORE_PROCESSING_REQUIRED.
//   c) It fails to populate pages directly on the destination GPU. It will try
//      to populate the vma using get_user_pages and return.
//   d) The full input range is migrated (or empty), this call will release
//      the semaphore before returning.
// 2) The user-mode needs to handle the following error codes:
//   a) NV_WARN_NOTHING_TO_DO: use move_pages to migrate pages for the VA
//      range corresponding to the vma that couldn't be migrated in kernel
//      mode. Then, it processes the remainder of the range, starting after
//      that vma.
//   b) NV_ERR_MORE_PROCESSING_REQUIRED: choose a different CPU NUMA node,
//      trying to enforce the NUMA policies of the thread and retry the
//      ioctl. If there are no more CPU NUMA nodes to try, try to populate
//      the remainder of the range anywhere using the UVM_POPULATE_PAGEABLE
//      ioctl.
//   c) NV_OK: success. This only guarantees that pages were populated, not
//      that they moved to the requested destination.
// 3) For cases 2.a) and 2.b) Goto 1
//
// If UVM_MIGRATE_FLAG_ASYNC is 0, the ioctl won't return until the migration is
// done and all mappings are updated, subject to the special rules for pageable
// memory described above. semaphoreAddress must be 0. semaphorePayload is
// ignored.
//
// If UVM_MIGRATE_FLAG_ASYNC is 1, the ioctl may return before the migration is
// complete. If semaphoreAddress is 0, semaphorePayload is ignored and no
// notification will be given on completion. If semaphoreAddress is non-zero
// and the returned error code is NV_OK, semaphorePayload will be written to
// semaphoreAddress once the migration is complete.
#define UVM_MIGRATE_FLAG_ASYNC              0x00000001

// When the migration destination is the CPU, skip the step which creates new
// virtual mappings on the CPU. Creating CPU mappings must wait for the
// migration to complete, so skipping this step allows the migration to be
// fully asynchronous. This flag is ignored for pageable migrations if the GPUs
// in the system support transparent access to pageable memory.
//
// The UVM driver must have builtin tests enabled for the API to use this flag.
#define UVM_MIGRATE_FLAG_SKIP_CPU_MAP       0x00000002

// By default UVM_MIGRATE returns an error if the destination UUID is a GPU
// without a registered GPU VA space. Setting this flag skips that check, so the
// destination GPU only needs to have been registered.
//
// This can be used in tests to trigger migrations of physical memory without
// the overhead of GPU PTE mappings.
//
// The UVM driver must have builtin tests enabled for the API to use this flag.
#define UVM_MIGRATE_FLAG_NO_GPU_VA_SPACE    0x00000004

#define UVM_MIGRATE_FLAGS_TEST_ALL              (UVM_MIGRATE_FLAG_SKIP_CPU_MAP      | \
                                                 UVM_MIGRATE_FLAG_NO_GPU_VA_SPACE)

#define UVM_MIGRATE_FLAGS_ALL                   (UVM_MIGRATE_FLAG_ASYNC | \
                                                 UVM_MIGRATE_FLAGS_TEST_ALL)

// If NV_ERR_INVALID_ARGUMENT is returned it is because cpuMemoryNode is not
// valid and the destination processor is the CPU. cpuMemoryNode is considered
// invalid if:
//      * it is less than -1,
//      * it is equal to or larger than the maximum number of nodes, or
//      * it corresponds to a registered GPU.
//      * it is not in the node_possible_map set of nodes,
//      * it does not have onlined memory
//
// For pageable migrations:
//
// In addition to the above, in the case of pageable memory, the
// cpuMemoryNode is considered invalid if it's -1.
//
// If NV_WARN_NOTHING_TO_DO is returned, user-space is responsible for
// completing the migration of the VA range described by userSpaceStart and
// userSpaceLength using move_pages.
//
// If NV_ERR_MORE_PROCESSING_REQUIRED is returned, user-space is responsible
// for re-trying with a different cpuNumaNode, starting at userSpaceStart.
//
#define UVM_MIGRATE                                                   UVM_IOCTL_BASE(51)
typedef struct
{
    NvU64           base               NV_ALIGN_BYTES(8); // IN
    NvU64           length             NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid destinationUuid;                      // IN
    NvU32           flags;                                // IN
    NvU64           semaphoreAddress   NV_ALIGN_BYTES(8); // IN
    NvU32           semaphorePayload;                     // IN
    NvS32           cpuNumaNode;                          // IN
    NvU64           userSpaceStart     NV_ALIGN_BYTES(8); // OUT
    NvU64           userSpaceLength    NV_ALIGN_BYTES(8); // OUT
    NV_STATUS       rmStatus;                             // OUT
} UVM_MIGRATE_PARAMS;

#define UVM_MIGRATE_RANGE_GROUP                                       UVM_IOCTL_BASE(53)
typedef struct
{
    NvU64           rangeGroupId       NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid destinationUuid;                      // IN
    NV_STATUS       rmStatus;                             // OUT
} UVM_MIGRATE_RANGE_GROUP_PARAMS;

//
// Initialize any tracker object such as a queue or counter
// UvmToolsCreateEventQueue, UvmToolsCreateProcessAggregateCounters,
// UvmToolsCreateProcessorCounters.
//
#define UVM_TOOLS_INIT_EVENT_TRACKER                                  UVM_IOCTL_BASE(56)
typedef struct
{
    NvU64           queueBuffer        NV_ALIGN_BYTES(8); // IN
    NvU64           queueBufferSize    NV_ALIGN_BYTES(8); // IN
    NvU64           controlBuffer      NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid processor;                            // IN
    NvU32           allProcessors;                        // IN
    NvU32           uvmFd;                                // IN
    NV_STATUS       rmStatus;                             // OUT
} UVM_TOOLS_INIT_EVENT_TRACKER_PARAMS;

//
// UvmToolsSetNotificationThreshold
//
#define UVM_TOOLS_SET_NOTIFICATION_THRESHOLD                          UVM_IOCTL_BASE(57)
typedef struct
{
    NvU32     notificationThreshold;                       // IN
    NV_STATUS rmStatus;                                    // OUT
} UVM_TOOLS_SET_NOTIFICATION_THRESHOLD_PARAMS;

//
// UvmToolsEventQueueEnableEvents
//
#define UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS                           UVM_IOCTL_BASE(58)
typedef struct
{
    NvU64     eventTypeFlags            NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                    // OUT
} UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS_PARAMS;

//
// UvmToolsEventQueueDisableEvents
//
#define UVM_TOOLS_EVENT_QUEUE_DISABLE_EVENTS                          UVM_IOCTL_BASE(59)
typedef struct
{
    NvU64     eventTypeFlags            NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                    // OUT
} UVM_TOOLS_EVENT_QUEUE_DISABLE_EVENTS_PARAMS;

//
// UvmToolsEnableCounters
//
#define UVM_TOOLS_ENABLE_COUNTERS                                     UVM_IOCTL_BASE(60)
typedef struct
{
    NvU64     counterTypeFlags          NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                    // OUT
} UVM_TOOLS_ENABLE_COUNTERS_PARAMS;

//
// UvmToolsDisableCounters
//
#define UVM_TOOLS_DISABLE_COUNTERS                                    UVM_IOCTL_BASE(61)
typedef struct
{
    NvU64     counterTypeFlags          NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                    // OUT
} UVM_TOOLS_DISABLE_COUNTERS_PARAMS;

//
// UvmToolsReadProcessMemory
//
#define UVM_TOOLS_READ_PROCESS_MEMORY                                 UVM_IOCTL_BASE(62)
typedef struct
{
    NvU64     buffer                    NV_ALIGN_BYTES(8); // IN
    NvU64     size                      NV_ALIGN_BYTES(8); // IN
    NvU64     targetVa                  NV_ALIGN_BYTES(8); // IN
    NvU64     bytesRead                 NV_ALIGN_BYTES(8); // OUT
    NV_STATUS rmStatus;                                    // OUT
} UVM_TOOLS_READ_PROCESS_MEMORY_PARAMS;

//
// UvmToolsWriteProcessMemory
//
#define UVM_TOOLS_WRITE_PROCESS_MEMORY                                UVM_IOCTL_BASE(63)
typedef struct
{
    NvU64     buffer                    NV_ALIGN_BYTES(8); // IN
    NvU64     size                      NV_ALIGN_BYTES(8); // IN
    NvU64     targetVa                  NV_ALIGN_BYTES(8); // IN
    NvU64     bytesWritten              NV_ALIGN_BYTES(8); // OUT
    NV_STATUS rmStatus;                                    // OUT
} UVM_TOOLS_WRITE_PROCESS_MEMORY_PARAMS;

//
// UvmToolsGetProcessorUuidTable
//
#define UVM_TOOLS_GET_PROCESSOR_UUID_TABLE                            UVM_IOCTL_BASE(64)
typedef struct
{
    NvU64     tablePtr                 NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                   // OUT
} UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_PARAMS;

//
// UvmMapDynamicParallelismRegion
//
#define UVM_MAP_DYNAMIC_PARALLELISM_REGION                            UVM_IOCTL_BASE(65)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8); // IN
    NvU64                   length                          NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid         gpuUuid;                                           // IN
    NV_STATUS               rmStatus;                                          // OUT
} UVM_MAP_DYNAMIC_PARALLELISM_REGION_PARAMS;

//
// UvmUnmapExternal
//
#define UVM_UNMAP_EXTERNAL                                            UVM_IOCTL_BASE(66)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8); // IN
    NvU64                   length                          NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid         gpuUuid;                                           // IN
    NV_STATUS               rmStatus;                                          // OUT
} UVM_UNMAP_EXTERNAL_PARAMS;


//
// UvmToolsFlushEvents
//
#define UVM_TOOLS_FLUSH_EVENTS                                        UVM_IOCTL_BASE(67)
typedef struct
{
    NV_STATUS rmStatus;                                   // OUT
} UVM_TOOLS_FLUSH_EVENTS_PARAMS;

//
// UvmAllocSemaphorePool
//
#define UVM_ALLOC_SEMAPHORE_POOL                                     UVM_IOCTL_BASE(68)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8); // IN
    NvU64                   length                          NV_ALIGN_BYTES(8); // IN
    UvmGpuMappingAttributes perGpuAttributes[UVM_MAX_GPUS];                    // IN
    NvU64                   gpuAttributesCount              NV_ALIGN_BYTES(8); // IN
    NV_STATUS               rmStatus;                                          // OUT
} UVM_ALLOC_SEMAPHORE_POOL_PARAMS;

//
// UvmCleanUpZombieResources
//
#define UVM_CLEAN_UP_ZOMBIE_RESOURCES                                 UVM_IOCTL_BASE(69)
typedef struct
{
    NV_STATUS rmStatus;                    // OUT
} UVM_CLEAN_UP_ZOMBIE_RESOURCES_PARAMS;

//
// UvmIsPageableMemoryAccessSupportedOnGpu
//
#define UVM_PAGEABLE_MEM_ACCESS_ON_GPU                                UVM_IOCTL_BASE(70)

typedef struct
{
    NvProcessorUuid gpu_uuid;          // IN
    NvBool          pageableMemAccess; // OUT
    NV_STATUS       rmStatus;          // OUT
} UVM_PAGEABLE_MEM_ACCESS_ON_GPU_PARAMS;

//
// UvmPopulatePageable
//
#define UVM_POPULATE_PAGEABLE                                         UVM_IOCTL_BASE(71)

// Allow population of managed ranges. The goal is to validate that it is
// possible to populate pageable ranges backed by VMAs with the VM_MIXEDMAP or
// VM_DONTEXPAND special flags set. But since there is no portable way to force
// allocation of such memory from user space, and it is not safe to change the
// flags of an already-created VMA from kernel space, we take advantage of the
// fact that managed ranges have both special flags set at creation time (see
// uvm_mmap).
#define UVM_POPULATE_PAGEABLE_FLAG_ALLOW_MANAGED              0x00000001

// By default UVM_POPULATE_PAGEABLE returns an error if the destination vma
// does not have read permission. This flag skips that check.
#define UVM_POPULATE_PAGEABLE_FLAG_SKIP_PROT_CHECK            0x00000002

// By default UVM_POPULATE_PAGEABLE returns an error if the destination vma
// is VM_IO or VM_PFNMAP. This flag skips that check.
#define UVM_POPULATE_PAGEABLE_FLAG_ALLOW_SPECIAL              0x00000004

// These flags are used internally within the driver and are not allowed from
// user space.
#define UVM_POPULATE_PAGEABLE_FLAGS_INTERNAL    UVM_POPULATE_PAGEABLE_FLAG_ALLOW_SPECIAL

// These flags are allowed from user space only when builtin tests are enabled.
// Some of them may also be used internally within the driver in non-test use
// cases.
#define UVM_POPULATE_PAGEABLE_FLAGS_TEST        (UVM_POPULATE_PAGEABLE_FLAG_ALLOW_MANAGED | \
                                                 UVM_POPULATE_PAGEABLE_FLAG_SKIP_PROT_CHECK)

#define UVM_POPULATE_PAGEABLE_FLAGS_ALL         (UVM_POPULATE_PAGEABLE_FLAGS_INTERNAL | \
                                                 UVM_POPULATE_PAGEABLE_FLAGS_TEST)

typedef struct
{
    NvU64           base      NV_ALIGN_BYTES(8); // IN
    NvU64           length    NV_ALIGN_BYTES(8); // IN
    NvU32           flags;                       // IN
    NV_STATUS       rmStatus;                    // OUT
} UVM_POPULATE_PAGEABLE_PARAMS;

//
// UvmValidateVaRange
//
#define UVM_VALIDATE_VA_RANGE                                         UVM_IOCTL_BASE(72)
typedef struct
{
    NvU64           base      NV_ALIGN_BYTES(8); // IN
    NvU64           length    NV_ALIGN_BYTES(8); // IN
    NV_STATUS       rmStatus;                    // OUT
} UVM_VALIDATE_VA_RANGE_PARAMS;

#define UVM_CREATE_EXTERNAL_RANGE                                     UVM_IOCTL_BASE(73)
typedef struct
{
    NvU64                  base                             NV_ALIGN_BYTES(8); // IN
    NvU64                  length                           NV_ALIGN_BYTES(8); // IN
    NV_STATUS              rmStatus;                                           // OUT
} UVM_CREATE_EXTERNAL_RANGE_PARAMS;

#define UVM_MAP_EXTERNAL_SPARSE                                       UVM_IOCTL_BASE(74)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8); // IN
    NvU64                   length                          NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid         gpuUuid;                                           // IN
    NV_STATUS               rmStatus;                                          // OUT
} UVM_MAP_EXTERNAL_SPARSE_PARAMS;

//
// Used to initialise a secondary UVM file-descriptor which holds a
// reference on the memory map to prevent it being torn down without
// first notifying UVM. This is achieved by preventing mmap() calls on
// the secondary file-descriptor so that on process exit
// uvm_mm_release() will be called while the memory map is present
// such that UVM can cleanly shutdown the GPU by handling faults
// instead of cancelling them.
//
// This ioctl must be called after the primary file-descriptor has
// been initialised with the UVM_INITIALIZE ioctl. The primary FD
// should be passed in the uvmFd field and the UVM_MM_INITIALIZE ioctl
// will hold a reference on the primary FD. Therefore uvm_release() is
// guaranteed to be called after uvm_mm_release().
//
// Once this file-descriptor has been closed the UVM context is
// effectively dead and subsequent operations requiring a memory map
// will fail. Calling UVM_MM_INITIALIZE on a context that has already
// been initialized via any FD will return NV_ERR_INVALID_STATE.
//
// Calling this with a non-UVM file-descriptor in uvmFd will return
// NV_ERR_INVALID_ARGUMENT. Calling this on the same file-descriptor
// as UVM_INITIALIZE or more than once on the same FD will return
// NV_ERR_IN_USE.
//
// Not all platforms require this secondary file-descriptor. On those
// platforms NV_WARN_NOTHING_TO_DO will be returned and users may
// close the file-descriptor at anytime.
#define UVM_MM_INITIALIZE                                             UVM_IOCTL_BASE(75)
typedef struct
{
    NvS32                   uvmFd;    // IN
    NV_STATUS               rmStatus; // OUT
} UVM_MM_INITIALIZE_PARAMS;

#define UVM_TOOLS_INIT_EVENT_TRACKER_V2                               UVM_IOCTL_BASE(76)
typedef UVM_TOOLS_INIT_EVENT_TRACKER_PARAMS UVM_TOOLS_INIT_EVENT_TRACKER_V2_PARAMS;

#define UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_V2                         UVM_IOCTL_BASE(77)
typedef UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_PARAMS UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_V2_PARAMS;

//
// UvmAllocDeviceP2P
//
#define UVM_ALLOC_DEVICE_P2P                                          UVM_IOCTL_BASE(78)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8); // IN
    NvU64                   length                          NV_ALIGN_BYTES(8); // IN
    NvU64                   offset                          NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid         gpuUuid;                                           // IN
    NvS32                   rmCtrlFd;                                          // IN
    NvU32                   hClient;                                           // IN
    NvU32                   hMemory;                                           // IN

    NV_STATUS               rmStatus;                                          // OUT
} UVM_ALLOC_DEVICE_P2P_PARAMS;

#define UVM_CLEAR_ALL_ACCESS_COUNTERS                                 UVM_IOCTL_BASE(79)

typedef struct
{
    NV_STATUS       rmStatus; // OUT
} UVM_CLEAR_ALL_ACCESS_COUNTERS_PARAMS;

//
// UvmPhaseFBatchExecutor
//
// Minimal production-facing Phase F batch executor. User space still owns
// policy decisions and submits explicit VA ranges plus an action. The driver
// only batches the existing UVM operations.
//
#define UVM_PHASE_F_MAX_RANGES                                         64

typedef enum
{
    UVM_PHASE_F_ACTION_PREFETCH_CPU = 1,
    UVM_PHASE_F_ACTION_PREFETCH_GPU = 2,
    UVM_PHASE_F_ACTION_ADVISE_CPU = 3,
    UVM_PHASE_F_ACTION_QUERY_RESIDENCY = 4,
    UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ = 5,
    UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG = 6,
    UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG_DEFERRED = 7,
    UVM_PHASE_F_ACTION_START_BG_PROMOTION = 8,
    UVM_PHASE_F_ACTION_GPU_ONLY_READ_SOFT = 9,
    UVM_PHASE_F_ACTION_GPU_ONLY_RW_SOFT = 10,
    UVM_PHASE_F_ACTION_CLEAR_GPU_ONLY = 13,
    UVM_PHASE_F_ACTION_GPU_ONLY_READ_DUP_SOFT = 14,
    UVM_PHASE_F_ACTION_PREFETCH_CPU_NO_MAP = 15,
    UVM_PHASE_F_ACTION_PAUSE_BG_PROMOTION = 16,
    UVM_PHASE_F_ACTION_RESUME_BG_PROMOTION = 17,
    // Expert-only read-duplicate promotion. This bypasses the generic migrate
    // API path and only accepts managed ranges tagged as MoE expert weights.
    UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST = 18,
    // Region-id executor variant which loops over registered expert regions
    // and invokes the native EXPERT_READ_DUP_FAST range path for each region.
    UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST_NATIVE_REGIONS = 19,
    // Setup-only diagnostic action. Every complete, aligned 2 MiB block must
    // already be CPU-sole-resident and remotely mapped READ-only on the GPU.
    // Existing 2 MiB PTEs are split in place into 32 ordinary 64 KiB PTEs;
    // blocks already using all-64 KiB PTEs are only validated. This action is
    // synchronous regardless of the batch async flag.
    UVM_PHASE_F_ACTION_FORCE_REMOTE_READ_64K = 20,
    // P3-to-remote cutover action. Install any missing CPU remote READ PTEs,
    // then synchronously prove every page is CPU-resident, GPU-nonresident,
    // and GPU READ-mapped. Result byte fields describe the final audited
    // state, not only PTEs newly installed by this invocation.
    UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_AUDITED = 21,
    // Scheduler-quiesced immutable-shadow cutover. The entire range must
    // already be read-duplicated on CPU and the target GPU. No data is copied:
    // the GPU duplicate is revoked and freed, while the existing CPU copy is
    // retained and mapped read-only for remote GPU access.
    UVM_PHASE_F_ACTION_IMMUTABLE_CPU_SHADOW_CUTOVER = 22,
    // Scheduler-quiesced repair for a complete managed Parameter allocation.
    // Rebuild every GPU READ mapping from UVM's authoritative residency masks
    // without migrating or copying data: GPU-resident pages map locally and
    // CPU-only pages map remotely. Slot-backed aliases are rejected because
    // their physical ownership is not represented by the residency masks.
    UVM_PHASE_F_ACTION_REBUILD_GPU_READ_MAPPINGS = 23,
    // Scheduler-quiesced synchronous cutover. Unlike action 22 this does not
    // require a pre-existing immutable read duplicate: it migrates any
    // GPU-only pages to CPU, releases local GPU residency, and installs a
    // remote READ mapping. Intended for infrequent adaptive Plan Class changes.
    UVM_PHASE_F_ACTION_SYNC_CPU_REMOTE_CUTOVER = 24,
} UVM_PHASE_F_ACTION;

typedef struct
{
    NvU64 base   NV_ALIGN_BYTES(8); // IN
    NvU64 length NV_ALIGN_BYTES(8); // IN
} UVM_PHASE_F_RANGE;

typedef struct
{
    NvU64 region_id NV_ALIGN_BYTES(8); // IN
    NvU64 base      NV_ALIGN_BYTES(8); // IN
    NvU64 length    NV_ALIGN_BYTES(8); // IN
} UVM_PHASE_F_REGION;

typedef struct
{
    NV_STATUS status;              // OUT
    NvU32     resident_on_count;   // OUT, query only
    NvBool    resident_on_cpu;     // OUT, query only
    NvBool    resident_on_gpu;     // OUT, query only
    NvU64     queried_bytes        NV_ALIGN_BYTES(8); // OUT, query/setup input bytes
    NvU64     resident_gpu_bytes   NV_ALIGN_BYTES(8); // OUT, query GPU bytes; action 20 split; action 21 remote-map
    NvU64     resident_cpu_bytes   NV_ALIGN_BYTES(8); // OUT, query CPU bytes; action 20 already-64K; action 21 CPU-resident
    NvU64     resident_any_bytes   NV_ALIGN_BYTES(8); // OUT, query any bytes; action 20 validated bytes
    NvU64     resident_unknown_bytes NV_ALIGN_BYTES(8); // OUT, query/setup unvalidated bytes
} UVM_PHASE_F_RANGE_RESULT;

#define UVM_PHASE_F_BATCH_EXECUTOR                                     UVM_IOCTL_BASE(81)
typedef struct
{
    NvU32                    action;                                // IN
    NvU32                    count;                                 // IN
    NvProcessorUuid          gpu_uuid;                              // IN
    NvBool                   async;                                 // IN
    NvU32                    success_count;                         // OUT
    NvU32                    failure_count;                         // OUT
    UVM_PHASE_F_RANGE        ranges[UVM_PHASE_F_MAX_RANGES];        // IN
    UVM_PHASE_F_RANGE_RESULT results[UVM_PHASE_F_MAX_RANGES];       // OUT
    NV_STATUS                rmStatus;                              // OUT
} UVM_PHASE_F_BATCH_EXECUTOR_PARAMS;

#define UVM_PHASE_F_REGISTER_REGION                                   UVM_IOCTL_BASE(82)
typedef struct
{
    UVM_PHASE_F_REGION region;     // IN
    NV_STATUS          rmStatus;   // OUT
} UVM_PHASE_F_REGISTER_REGION_PARAMS;

#define UVM_PHASE_F_UNREGISTER_REGION                                 UVM_IOCTL_BASE(83)
typedef struct
{
    NvU64     region_id NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                    // OUT
} UVM_PHASE_F_UNREGISTER_REGION_PARAMS;

#define UVM_PHASE_F_BATCH_REGIONS                                      UVM_IOCTL_BASE(84)
typedef struct
{
    NvU32                    action;                                // IN
    NvU32                    count;                                 // IN
    NvProcessorUuid          gpu_uuid;                              // IN
    NvBool                   async;                                 // IN
    NvU32                    success_count;                         // OUT
    NvU32                    failure_count;                         // OUT
    NvU64                    region_ids[UVM_PHASE_F_MAX_RANGES];    // IN
    UVM_PHASE_F_RANGE_RESULT results[UVM_PHASE_F_MAX_RANGES];       // OUT
    NV_STATUS                rmStatus;                              // OUT
} UVM_PHASE_F_BATCH_REGIONS_PARAMS;

//
// UvmMoeSlotRemapBatch
//
// Experimental batch interface for MoE cold slot remap planning. User space
// submits explicit (layer, expert, slot, VA range) assignments after router/topk
// and batch slot allocation. The initial driver implementation validates and
// records the batch. COPY_ONLY mode allocates driver-owned slot backing pages
// and copies the submitted user VA ranges into those pages without replacing
// user PTEs. GPU_COPY_ONLY mode allocates driver-owned vidmem slot backing and
// uses CE copies from a DMA-capable sysmem staging allocation. PTE replacement
// and TLB invalidation are intentionally guarded. REMAP mode currently uses the
// native UVM migrate/map machinery. SLOT_PTE_REMAP is a smoke-only mode for
// explicitly-enabled experiments: it writes the target GPU page-table PTEs to
// point at driver-owned VIDMEM slot backing without updating the full VA block
// software state. SLOT_VA_BLOCK_REMAP first lets the target managed VA block
// unmap the GPU range and wait its tracker, then installs the slot-backed PTE.
//
// A Qwen3 MoE layer contributes 128 experts with 9 independently-backed
// 512 KiB weight ranges per expert.  Runtime KV tier changes can add ten
// experts per layer across the retained graph consumers, producing 4,500
// static-alias records in one scheduler-quiesced transaction.  Keep the
// shared slot-remap scratch ABI large enough for that atomic append.
#define UVM_MOE_SLOT_REMAP_MAX_ENTRIES                                4608

typedef enum
{
    UVM_MOE_SLOT_REMAP_MODE_DRY_RUN = 0,
    UVM_MOE_SLOT_REMAP_MODE_VALIDATE = 1,
    UVM_MOE_SLOT_REMAP_MODE_REMAP = 2,
    UVM_MOE_SLOT_REMAP_MODE_COPY_ONLY = 3,
    UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ONLY = 4,
    UVM_MOE_SLOT_REMAP_MODE_SLOT_PTE_REMAP = 5,
    UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_REMAP = 6,
    // Experimental stage-boundary path. entry->va_base is the target managed
    // VA to remap. entry->slot_offset is a managed user backing VA which must
    // already be GPU-resident on gpu_uuid, typically after cudaMemcpyAsync and
    // copy_done_event. No slot copy is performed by this mode.
    UVM_MOE_SLOT_REMAP_MODE_SLOT_VA_BLOCK_USER_BACKING_REMAP = 7,
    // Preallocate driver-owned VIDMEM slot backing for the submitted logical
    // slots. This does not copy data and does not install PTEs. While a
    // static-alias session exists this is accepted only for new, uncommitted
    // slots in a healthy parked session, for a later resizing REBIND.
    UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC = 8,
    // Submit a CPU-to-GPU copy into an existing or newly allocated
    // driver-owned VIDMEM slot and return without waiting for the copy engine.
    // Completion is finalized when the slot is next reused or released.
    UVM_MOE_SLOT_REMAP_MODE_GPU_COPY_ASYNC_SUBMIT = 9,
    // Copy directly from managed CPU chunks into an already-preallocated
    // driver-owned VIDMEM slot, then map the original managed VA read-only to
    // that slot. One CE push is submitted per VA block. This mode is
    // synchronous and never allocates or evicts slot backing.
    UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_PER_BLOCK_PUSH = 10,
    // Identical source, slot backing, and PTE mapping semantics to mode 10,
    // but packs physically-contiguous CPU runs from multiple VA blocks into
    // shared CE pushes. Pushes are flushed dynamically when they approach the
    // hardware pushbuffer limit. This mode is synchronous and never allocates
    // or evicts slot backing.
    UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH = 11,
    // Per-block slot-direct copy with the same semantics as mode 10, but map
    // each eligible full VA block with 64 KiB PTEs.
    UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_PTE = 12,
    // Per-block slot-direct copy with the same semantics as mode 10, but map
    // each eligible full VA block with one 2 MiB PTE.
    UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_2M_PTE = 13,
    // Preserve an existing read-only CPU mapping until the slot copy has
    // completed, then overwrite it with an equal-granularity slot PTE and a
    // SYS-scoped TLB invalidate. This avoids a separate unmap push.
    UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_OVERWRITE = 14,
    // Diagnostic 4 KiB path which synchronizes preparation, copy, and map at
    // their boundaries so copy-complete and map-complete time can be measured
    // independently. It is intentionally not a throughput pipeline mode.
    UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_STAGE_SYNC = 15,
    // Diagnostic equivalent of mode 12 with a synchronization boundary after
    // prepare, 64 KiB slot copy, and map. This measures completion rather than
    // only host submission time.
    UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_STAGE_SYNC = 16,
    // Like mode 14, but require every existing ordinary remote READ mapping to
    // be uniformly 64 KiB. Any 2 MiB, 4 KiB, or mixed block is rejected rather
    // than auto-detected.
    UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_OVERWRITE = 17,
    // Stage-synchronized diagnostic form of mode 17.
    UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_OVERWRITE_STAGE_SYNC = 18,
    // Release driver-owned backing and any installed slot PTEs for the
    // submitted logical slots. Release entries identify a slot only with
    // layer_id, expert_id, and slot_id; all other entry input fields must be 0.
    // A healthy parked static-alias session may use this only to roll back
    // uncommitted slots outside its committed topology.
    UVM_MOE_SLOT_REMAP_MODE_RELEASE = 19,
    // Restricted fixed-bank layer transition. Every entry must name an
    // existing NO_EVICT VIDMEM slot and carry REUSE_BACKING_REQUIRED. The old
    // slot-backed PTEs are removed first, while the old logical owners remain
    // unchanged. Copy and replacement-PTE preparation may overlap, but the
    // replacement PTEs, TLB invalidation, and new owners are published only
    // after all tracked work succeeds. Optional per-entry strategy bits select
    // shared pushes and/or sliced 64 KiB PTEs, and must be uniform across the
    // batch. No backing allocation, eviction, or large-PTE fallback is
    // permitted. 64K eligibility is fully preflighted before old-PTE teardown.
    // Before entering this ioctl, the caller must ensure that ordinary CUDA
    // work has stopped accessing every old slot VA. UVM trackers order the
    // driver's copy/map pushes, but do not track arbitrary CUDA kernels.
    // Validation/resource rejection before TRANSITIONING touches no PTE and
    // leaves the active session intact. Once the execution-stage transition
    // begins, a failure retains the old owners, reports remap_count == 0, and
    // leaves the affected GPU VAs unmapped; registered-plan execution also keeps
    // its active generation unchanged and poisons the session.
    UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION = 20,
    // Ordinary (non-owner-transition) slot-direct path combining the mode-11
    // shared CE push strategy with the mode-12 sliced 64 KiB PTE strategy.
    UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE = 21,
} UVM_MOE_SLOT_REMAP_MODE;

// Keep this slot under explicit userspace lifetime control. Allocations with
// this flag fail when the configured budget or physical VIDMEM is exhausted,
// and the slot cannot be selected as an implicit budget-eviction victim.
#define UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT                              0x00000001U

// Mode-8 only: the submitted logical slot must already have sufficiently
// large VIDMEM backing. The old owner/PTE metadata is replaced, but allocating
// a new uvm_mem_t (or growing the existing one) is forbidden. This makes a
// fixed physical slot bank rebind auditable and fail-closed.
#define UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED                0x00000002U

// Mode-20-only request strategy bits. These describe how the transition's
// copy/map phase is executed; they are never persisted in slot->flags. Every
// entry in one transition must carry the same strategy bits.
#define UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_SHARED_PUSH          0x00000004U
#define UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_64K_PTE              0x00000008U

typedef enum
{
    UVM_MOE_SLOT_REMAP_ENTRY_OK = 0,
    UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ARGUMENT = 1,
    UVM_MOE_SLOT_REMAP_ENTRY_INVALID_ADDRESS = 2,
    UVM_MOE_SLOT_REMAP_ENTRY_SLOT_CONFLICT = 3,
    UVM_MOE_SLOT_REMAP_ENTRY_REMAP_NOT_SUPPORTED = 4,
    UVM_MOE_SLOT_REMAP_ENTRY_BACKING_ALLOC_FAILED = 5,
    UVM_MOE_SLOT_REMAP_ENTRY_COPY_FAILED = 6,
    UVM_MOE_SLOT_REMAP_ENTRY_REMAP_FAILED = 7,
    UVM_MOE_SLOT_REMAP_ENTRY_ROLLED_BACK = 8,
} UVM_MOE_SLOT_REMAP_ENTRY_STATUS;

typedef struct
{
    NvU32 layer_id;                    // IN
    NvU32 expert_id;                   // IN
    NvU32 slot_id;                     // IN
    NvU32 role;                        // IN
    NvU64 va_base NV_ALIGN_BYTES(8);   // IN
    NvU64 bytes NV_ALIGN_BYTES(8);     // IN
    NvU64 slot_offset NV_ALIGN_BYTES(8); // IN, byte offset into the physical slot backing
    NvU32 flags;                       // IN
    NvU32 status;                      // OUT, UVM_MOE_SLOT_REMAP_ENTRY_STATUS
} UVM_MOE_SLOT_REMAP_ENTRY;

#define UVM_MOE_SLOT_REMAP_BATCH                                       UVM_IOCTL_BASE(91)
typedef struct
{
    NvU32 mode;                                      // IN
    NvU32 count;                                     // IN
    NvProcessorUuid gpu_uuid;                        // IN, reserved for REMAP mode
    NvU32 validate_managed;                          // IN
    NvU32 success_count;                             // OUT
    NvU32 failure_count;                             // OUT
    NvU32 conflict_count;                            // OUT
    NvU32 remap_count;                               // OUT
    NvU32 tlb_invalidate_count;                      // OUT
    // OUT; for SLOT_DIRECT_* modes, number of CE copy pushes submitted. For
    // VIDMEM_PREALLOC, number of entries which reused existing VIDMEM backing.
    NvU32 reserved;
    NvU64 requested_bytes NV_ALIGN_BYTES(8);         // OUT
    NvU64 accepted_bytes NV_ALIGN_BYTES(8);          // OUT
    UVM_MOE_SLOT_REMAP_ENTRY entries[UVM_MOE_SLOT_REMAP_MAX_ENTRIES]; // IN/OUT
    NV_STATUS rmStatus;                              // OUT
} UVM_MOE_SLOT_REMAP_BATCH_PARAMS;

// Register one immutable layer placement plan. Registration is deliberately
// separate from execution: the large descriptor array crosses the ioctl
// boundary once during model initialization, while the serving hot path uses
// the compact execute request below. A session must register exactly one mode
// 21 bootstrap descriptor first; every later descriptor must be a mode 20
// target-layer transition referencing a preallocated fixed physical slot bank.
// A dual-bank session may alternate between two disjoint banks; entries from
// the two banks must never share the same physical slot backing.
#define UVM_MOE_SLOT_PLAN_REGISTER                                    UVM_IOCTL_BASE(92)
typedef struct
{
    NvU64 plan_id NV_ALIGN_BYTES(8);                 // IN, non-zero userspace id
    // OUT. The active generation is monotonic for the module lifetime and is
    // never reused when the last plan is destroyed and a new session starts.
    NvU64 generation NV_ALIGN_BYTES(8);
    NvU32 layer_id;                                  // IN, logical model layer
    NvU32 mode;                                      // IN, mode 20 or mode 21
    NvU32 count;                                     // IN
    NvU32 reserved;                                  // must be zero
    NvProcessorUuid gpu_uuid;                        // IN
    NvU64 requested_bytes NV_ALIGN_BYTES(8);         // OUT
    UVM_MOE_SLOT_REMAP_ENTRY entries[UVM_MOE_SLOT_REMAP_MAX_ENTRIES]; // IN
    NV_STATUS rmStatus;                              // OUT
} UVM_MOE_SLOT_PLAN_REGISTER_PARAMS;

// Execute a previously registered immutable plan. No range descriptors are
// copied from userspace on this path. generation serializes owner changes and
// makes stale layer transitions fail before any PTE or slot state is touched.
// Each successful execution advances the module-lifetime generation only
// after copy, PTE publication, and TLB invalidation have completed. Identity,
// device, slot, target, large-leaf, and cleanup-resource preflight completes
// before TRANSITIONING; rejection there touches no PTE and leaves the active
// generation/session intact. Once an execution-stage transition fails,
// owner_cutover and remap_count are zero, new_generation remains equal to the
// input generation, poisoned is set, and the affected GPU VAs remain unmapped.
// Such a session is abort-only. The caller must complete all CUDA work using the
// old slot VAs before execute; the internal tracker covers UVM pushes, not
// ordinary CUDA kernel lifetime.
#define UVM_MOE_SLOT_PLAN_EXECUTE                                     UVM_IOCTL_BASE(93)
typedef struct
{
    NvU64 plan_id NV_ALIGN_BYTES(8);                 // IN
    NvU64 generation NV_ALIGN_BYTES(8);              // IN
    NvU64 new_generation NV_ALIGN_BYTES(8);          // OUT
    NvU64 requested_bytes NV_ALIGN_BYTES(8);         // OUT
    NvU64 accepted_bytes NV_ALIGN_BYTES(8);          // OUT
    NvU32 layer_id;                                  // IN
    NvU32 mode;                                      // OUT
    NvU32 count;                                     // OUT
    NvU32 success_count;                             // OUT
    NvU32 failure_count;                             // OUT
    NvU32 conflict_count;                            // OUT
    NvU32 remap_count;                               // OUT
    NvU32 tlb_invalidate_count;                      // OUT
    NvU32 copy_push_count;                           // OUT
    NvU32 owner_cutover;                             // OUT, 1 only after successful owner/generation commit
    NvU32 poisoned;                                  // OUT, failure requires generation-free session abort
    NV_STATUS rmStatus;                              // OUT
} UVM_MOE_SLOT_PLAN_EXECUTE_PARAMS;

// Execute a registered plan with a kernel-verified CUDA stream hand-off.
// Both addresses must name distinct, naturally aligned 32-bit cells in a UVM
// semaphore-pool VA range owned by this file's VA space. Before entering the
// transition, the UVM MEMOPS channel waits for compute_payload at
// compute_semaphore_address. Only after copy, PTE/TLB publication, metadata,
// owner, and generation commit have all completed does the driver publish
// completion_payload at completion_semaphore_address. CUDA user space must
// enqueue the matching stream write before this ioctl and the matching stream
// wait before submitting kernels which consume the new mapping.
//
// This is a separate command instead of extending command 93 so an older
// userspace structure can never be over-read by a newer experimental driver.
#define UVM_MOE_SLOT_PLAN_EXECUTE_FENCED                              UVM_IOCTL_BASE(96)

#define UVM_MOE_SLOT_PLAN_FENCE_FLAG_COMPUTE_ACQUIRE                 0x00000001U
#define UVM_MOE_SLOT_PLAN_FENCE_FLAG_COMPLETION_RELEASE              0x00000002U
#define UVM_MOE_SLOT_PLAN_FENCE_FLAGS_ALL                            \
    (UVM_MOE_SLOT_PLAN_FENCE_FLAG_COMPUTE_ACQUIRE |                  \
     UVM_MOE_SLOT_PLAN_FENCE_FLAG_COMPLETION_RELEASE)

typedef struct
{
    // Common execute inputs and outputs. Keep this prefix field-for-field
    // compatible with command 93 through poisoned; rmStatus is intentionally
    // last so the two command sizes remain unambiguous.
    NvU64 plan_id NV_ALIGN_BYTES(8);                 // IN
    NvU64 generation NV_ALIGN_BYTES(8);              // IN
    NvU64 new_generation NV_ALIGN_BYTES(8);          // OUT
    NvU64 requested_bytes NV_ALIGN_BYTES(8);         // OUT
    NvU64 accepted_bytes NV_ALIGN_BYTES(8);          // OUT
    NvU32 layer_id;                                  // IN
    NvU32 mode;                                      // OUT
    NvU32 count;                                     // OUT
    NvU32 success_count;                             // OUT
    NvU32 failure_count;                             // OUT
    NvU32 conflict_count;                            // OUT
    NvU32 remap_count;                               // OUT
    NvU32 tlb_invalidate_count;                      // OUT
    NvU32 copy_push_count;                           // OUT
    NvU32 owner_cutover;                             // OUT
    NvU32 poisoned;                                  // OUT

    NvU32 fence_flags;                               // IN, exactly FLAGS_ALL
    NvU64 compute_semaphore_address NV_ALIGN_BYTES(8);    // IN
    NvU64 completion_semaphore_address NV_ALIGN_BYTES(8); // IN
    NvU32 compute_payload;                           // IN, non-zero timeline value
    NvU32 completion_payload;                        // IN, equal to compute_payload
    NvU32 fence_accepted;                            // OUT, kernel validated and waited
    NvU32 completion_signaled;                       // OUT, success payload published
    NvU32 reserved;                                  // IN, must be zero
    NV_STATUS rmStatus;                              // OUT
} UVM_MOE_SLOT_PLAN_EXECUTE_FENCED_PARAMS;

// Remove one cached descriptor plan. Physical slot backing remains under the
// existing mode-19/session-close lifetime contract. VA-space teardown also
// destroys all remaining plans without relying on userspace. Destroying the
// last plan reports current_generation == 0 to preserve the ABI's inactive-
// session sentinel, but the internal module-lifetime counter is not reset.
#define UVM_MOE_SLOT_PLAN_DESTROY                                     UVM_IOCTL_BASE(94)
typedef struct
{
    NvU64 plan_id NV_ALIGN_BYTES(8);                 // IN
    NvU64 generation NV_ALIGN_BYTES(8);              // IN
    NvU64 current_generation NV_ALIGN_BYTES(8);      // OUT
    NvU32 destroyed;                                 // OUT
    NvU32 reserved;                                  // must be zero
    NV_STATUS rmStatus;                              // OUT
} UVM_MOE_SLOT_PLAN_DESTROY_PARAMS;

// Tear down every MoE slot resource owned by the VA space associated with
// this file. Unlike closing an os.dup() of the UVM fd, this is an explicit,
// live-session operation. It is intentionally independent of userspace's
// cached plan ids and slot list so it can recover after a partial register or
// an ioctl result/contract mismatch.
//
// Release is convergent: successfully released slots stay released. If any
// slot fails, its owner/backing and all registered plans remain available for
// a retry and the registered session is poisoned. Plans and session metadata
// are destroyed only after no owned slot remains.
#define UVM_MOE_SLOT_SESSION_ABORT                                   UVM_IOCTL_BASE(95)
typedef struct
{
    NvU32 flags;                                      // IN, must be zero
    NvU32 reserved;                                   // IN, must be zero
    NvU64 generation_before NV_ALIGN_BYTES(8);        // OUT, zero without an owned registered session
    NvU64 released_bytes NV_ALIGN_BYTES(8);           // OUT, bytes released by this call
    NvU64 remaining_bytes NV_ALIGN_BYTES(8);          // OUT, backing/mapping bytes requiring retry
    NvU32 destroyed_plan_count;                       // OUT, plans destroyed by this call
    NvU32 released_slot_count;                        // OUT, slots released by this call
    NvU32 remaining_slot_count;                       // OUT, slot records which still require retry
    NvU32 had_registered_session;                     // OUT, this VA space owned plans/session on entry
    NvU32 session_cleared;                            // OUT, this VA space now owns no slot or plan
    NvU32 poisoned;                                   // OUT, retry-only state remains after this call
    NV_STATUS rmStatus;                               // OUT
} UVM_MOE_SLOT_SESSION_ABORT_PARAMS;

// Initialization-time static aliases for CUDA Graph weight streaming. Unlike
// the dynamic MoE slot transition commands above, these commands never perform
// a serving-time owner cutover. Userspace builds an immutable set of aliases,
// seals it before CUDA Graph capture, and destroys it only after every Graph and
// CUDA kernel which can reference an alias has completed.
//
// MAP creates a session when session_id is zero and returns its non-zero id. A
// later MAP with that id appends aliases to the same unsealed session. A
// QUIESCED_APPEND is the serving-time exception: it appends a disjoint set to
// an already sealed, non-parked session after userspace has synchronized the
// sole compute stream at a scheduler boundary. It never scans, parks, or
// republishes the existing consumer set. DEFER_DUAL_REFRESH is valid only with
// QUIESCED_APPEND. It invalidates the prebuilt dual-view recipe without
// rebuilding it; userspace must issue SEAL(REFRESH_DUAL) after its associated
// physical-layout transaction and before PARK. Entries
// with the same backing identity and overlapping backing offsets intentionally
// alias the same physical bytes. Target VA ranges within a session must not
// overlap. The VA ranges must belong to the UVM VA space associated with this
// file. A successful MAP installs every requested alias or none; a failure
// which reports poisoned leaves the session destroy-only. Before MAP,
// userspace must synchronize every stream which could access a target VA; UVM
// trackers cannot observe arbitrary user kernels or an existing Graph replay.
// All entries in one MAP use the same PTE granularity in ABI v1.
#define UVM_MOE_STATIC_ALIAS_ABI_VERSION                              1U
#define UVM_MOE_STATIC_ALIAS_MAX_ENTRIES                              4608U

#define UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW                            0x00000001U
#define UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K                       0x00000002U
#define UVM_MOE_STATIC_ALIAS_ENTRY_FLAGS_ALL                          \
    (UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW |                             \
     UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K)

typedef enum
{
    UVM_MOE_STATIC_ALIAS_ENTRY_OK = 0,
    UVM_MOE_STATIC_ALIAS_ENTRY_INVALID_ARGUMENT = 1,
    UVM_MOE_STATIC_ALIAS_ENTRY_INVALID_ADDRESS = 2,
    UVM_MOE_STATIC_ALIAS_ENTRY_BACKING_NOT_FOUND = 3,
    UVM_MOE_STATIC_ALIAS_ENTRY_ALIAS_CONFLICT = 4,
    UVM_MOE_STATIC_ALIAS_ENTRY_MAP_FAILED = 5,
    UVM_MOE_STATIC_ALIAS_ENTRY_ROLLED_BACK = 6,
} UVM_MOE_STATIC_ALIAS_ENTRY_STATUS;

typedef struct
{
    NvU32 backing_layer_id;                           // IN
    NvU32 backing_expert_id;                          // IN
    NvU32 backing_slot_id;                            // IN
    NvU32 flags;                                      // IN, ENTRY_FLAG_*
    NvU64 va_base NV_ALIGN_BYTES(8);                  // IN, target alias VA
    NvU64 bytes NV_ALIGN_BYTES(8);                    // IN
    NvU64 slot_offset NV_ALIGN_BYTES(8);              // IN, physical backing offset
    NvU32 status;                                     // OUT, ENTRY_STATUS
} UVM_MOE_STATIC_ALIAS_ENTRY;

#define UVM_MOE_STATIC_ALIAS_MAP                                      UVM_IOCTL_BASE(97)
#define UVM_MOE_STATIC_ALIAS_MAP_FLAG_QUIESCED_APPEND                 0x00000001U
#define UVM_MOE_STATIC_ALIAS_MAP_FLAG_DEFER_DUAL_REFRESH              0x00000002U
#define UVM_MOE_STATIC_ALIAS_MAP_FLAGS_ALL                            \
    (UVM_MOE_STATIC_ALIAS_MAP_FLAG_QUIESCED_APPEND |                  \
     UVM_MOE_STATIC_ALIAS_MAP_FLAG_DEFER_DUAL_REFRESH)
typedef struct
{
    NvU32 abi_version;                                // IN, ABI_VERSION
    NvU32 struct_size;                                // IN, sizeof(params)
    NvU32 flags;                                      // IN, MAP_FLAG_*
    NvU32 reserved;                                   // IN, must be zero
    NvU64 session_id NV_ALIGN_BYTES(8);               // IN/OUT, zero creates
    NvProcessorUuid gpu_uuid;                         // IN
    NvU64 requested_bytes NV_ALIGN_BYTES(8);          // OUT, this call
    NvU64 mapped_bytes NV_ALIGN_BYTES(8);             // OUT, this call
    NvU64 total_alias_bytes NV_ALIGN_BYTES(8);        // OUT, whole session
    NvU64 backing_bytes NV_ALIGN_BYTES(8);            // OUT, unique backing
    NvU32 count;                                      // IN
    NvU32 success_count;                              // OUT, this call
    NvU32 failure_count;                              // OUT, this call
    NvU32 alias_count;                                // OUT, this call
    NvU32 tlb_invalidate_count;                       // OUT, this call
    NvU32 poisoned;                                   // OUT, boolean
    UVM_MOE_STATIC_ALIAS_ENTRY entries[UVM_MOE_STATIC_ALIAS_MAX_ENTRIES]; // IN/OUT
    NV_STATUS rmStatus;                               // OUT
} UVM_MOE_STATIC_ALIAS_MAP_PARAMS;

// Freeze an alias session before CUDA Graph capture. The expected fields make
// a userspace/kernel descriptor mismatch fail before the session is published
// as immutable. REFRESH_DUAL keeps the frozen directory and rebuilds only the
// two detached 2 MiB-parent children from current UVM residency. It is used at
// a quiescent initialization boundary after KV/Graph allocations have settled.
// REFRESH_DUAL_NESTED may be combined with REFRESH_DUAL when the immutable
// cold-page recipe only grows. Existing leaves stay frozen and only newly
// selected leaves are copied into the GPU child and written into the CPU child.
#define UVM_MOE_STATIC_ALIAS_SEAL                                     UVM_IOCTL_BASE(98)
#define UVM_MOE_STATIC_ALIAS_SEAL_FLAG_REFRESH_DUAL                    0x00000001U
#define UVM_MOE_STATIC_ALIAS_SEAL_FLAG_REFRESH_DUAL_NESTED             0x00000002U
#define UVM_MOE_STATIC_ALIAS_SEAL_FLAG_DISCARD_DUAL                    0x00000004U
#define UVM_MOE_STATIC_ALIAS_SEAL_FLAGS_ALL                            \
    (UVM_MOE_STATIC_ALIAS_SEAL_FLAG_REFRESH_DUAL |                     \
     UVM_MOE_STATIC_ALIAS_SEAL_FLAG_REFRESH_DUAL_NESTED |              \
     UVM_MOE_STATIC_ALIAS_SEAL_FLAG_DISCARD_DUAL)
typedef struct
{
    NvU32 abi_version;                                // IN, ABI_VERSION
    NvU32 struct_size;                                // IN, sizeof(params)
    NvU32 flags;                                      // IN, *_FLAG_*
    NvU32 reserved;                                   // IN, must be zero
    NvU64 session_id NV_ALIGN_BYTES(8);               // IN, non-zero
    NvU64 expected_alias_bytes NV_ALIGN_BYTES(8);     // IN
    NvU64 alias_bytes NV_ALIGN_BYTES(8);              // OUT
    NvU64 backing_bytes NV_ALIGN_BYTES(8);            // OUT
    NvU32 expected_alias_count;                       // IN
    NvU32 sealed;                                     // OUT, boolean
    NvU32 alias_count;                                // OUT
    NvU32 poisoned;                                   // OUT, boolean
    NV_STATUS rmStatus;                               // OUT
} UVM_MOE_STATIC_ALIAS_SEAL_PARAMS;

// Remove every mapping owned by one session. RELEASE_BACKING additionally
// releases the session's physical slot backing. Destruction is convergent:
// released resources stay released and remaining_* reports what a retry must
// reclaim. session_cleared is set only when no alias or backing remains.
#define UVM_MOE_STATIC_ALIAS_DESTROY                                  UVM_IOCTL_BASE(99)
#define UVM_MOE_STATIC_ALIAS_DESTROY_FLAG_RELEASE_BACKING              0x00000001U
#define UVM_MOE_STATIC_ALIAS_DESTROY_FLAG_DEFER_CPU_RESTORE             0x00000002U
#define UVM_MOE_STATIC_ALIAS_DESTROY_FLAGS_ALL                         \
    (UVM_MOE_STATIC_ALIAS_DESTROY_FLAG_RELEASE_BACKING |               \
     UVM_MOE_STATIC_ALIAS_DESTROY_FLAG_DEFER_CPU_RESTORE)
typedef struct
{
    NvU32 abi_version;                                // IN, ABI_VERSION
    NvU32 struct_size;                                // IN, sizeof(params)
    NvU32 flags;                                      // IN, DESTROY_FLAG_*
    NvU32 reserved;                                   // IN, must be zero
    NvU64 session_id NV_ALIGN_BYTES(8);               // IN, non-zero
    NvU64 released_alias_bytes NV_ALIGN_BYTES(8);     // OUT, this call
    NvU64 remaining_alias_bytes NV_ALIGN_BYTES(8);    // OUT
    NvU64 released_backing_bytes NV_ALIGN_BYTES(8);   // OUT, this call
    NvU64 remaining_backing_bytes NV_ALIGN_BYTES(8);  // OUT
    NvU32 released_alias_count;                       // OUT, this call
    NvU32 remaining_alias_count;                      // OUT
    NvU32 session_cleared;                            // OUT, boolean
    NvU32 poisoned;                                   // OUT, boolean
    NV_STATUS rmStatus;                               // OUT
} UVM_MOE_STATIC_ALIAS_DESTROY_PARAMS;

// Reversible serving-time topology switch for a sealed static-alias session.
// PARK atomically replaces read-only consumer aliases with ordinary CPU
// remote-read PTEs in grouped VA-block batches. Writable upload aliases,
// physical backing, the session descriptor, and all alias records remain
// live. RESUME replaces those ordinary UVM PTEs with the original read-only
// slot aliases. Both operations require a userspace GPU/scheduler barrier;
// the kernel cannot observe arbitrary CUDA Graph replays.
//
// Command numbers 101 and 102 are used because command 100 is already the
// expert offload/release ABI below.
#define UVM_MOE_STATIC_ALIAS_PARK                                     UVM_IOCTL_BASE(101)
#define UVM_MOE_STATIC_ALIAS_RESUME                                   UVM_IOCTL_BASE(102)
#define UVM_MOE_STATIC_ALIAS_TRANSITION_FLAG_FORCE_SEQUENTIAL         (1U << 0)
#define UVM_MOE_STATIC_ALIAS_TRANSITION_FLAG_DEFER_CPU_RESTORE         (1U << 1)
#define UVM_MOE_STATIC_ALIAS_TRANSITION_FLAGS_ALL                      \
    (UVM_MOE_STATIC_ALIAS_TRANSITION_FLAG_FORCE_SEQUENTIAL |          \
     UVM_MOE_STATIC_ALIAS_TRANSITION_FLAG_DEFER_CPU_RESTORE)
typedef struct
{
    NvU32 abi_version;                                // IN, ABI_VERSION
    NvU32 struct_size;                                // IN, sizeof(params)
    NvU32 flags;                                      // IN, TRANSITION_FLAG_*
    NvU32 reserved;                                   // IN zero; OUT completed transactions
    NvU64 session_id NV_ALIGN_BYTES(8);               // IN, sealed non-zero session
    NvU64 processed_alias_bytes NV_ALIGN_BYTES(8);    // OUT, consumer bytes changed
    NvU64 consumer_alias_bytes NV_ALIGN_BYTES(8);     // OUT, complete consumer set
    NvU64 backing_bytes NV_ALIGN_BYTES(8);            // OUT, retained physical backing
    NvU32 processed_alias_count;                      // OUT, consumers changed this call
    NvU32 consumer_alias_count;                       // OUT, complete consumer set
    NvU32 upload_alias_count;                         // OUT, retained writable aliases
    NvU32 tlb_invalidate_count;                       // OUT, affected VA-block groups/slices
    NvU32 parked;                                     // OUT, consumer aliases absent
    NvU32 sealed;                                     // OUT, session remains sealed
    NvU32 poisoned;                                   // OUT, destroy-only after failure
    NV_STATUS rmStatus;                               // OUT
    // Non-overlapping driver phases. Ioctl wall time not represented here is
    // allocation, outer lock acquisition/release, and ABI bookkeeping.
    NvU64 validation_ns NV_ALIGN_BYTES(8);            // OUT
    NvU64 build_sort_ns NV_ALIGN_BYTES(8);            // OUT
    NvU64 preflight_ns NV_ALIGN_BYTES(8);             // OUT
    NvU64 metadata_ns NV_ALIGN_BYTES(8);              // OUT
    NvU64 cpu_unmap_ns NV_ALIGN_BYTES(8);              // OUT
    NvU64 pte_map_ns NV_ALIGN_BYTES(8);                // OUT
    NvU64 wait_ns NV_ALIGN_BYTES(8);                   // OUT
    NvU64 bookkeeping_ns NV_ALIGN_BYTES(8);            // OUT
    // A true value certifies that every read-only consumer byte belongs to a
    // prebuilt CPU/GPU dual child. In that mode PARK/RESUME only publish 2 MiB
    // parent entries; no consumer leaf PTE has to be created at transition
    // time. Userspace may therefore skip its redundant remote-map pass.
    NvU32 dual_view_full_coverage;                      // OUT, boolean
    NvU32 dual_view_block_count;                        // OUT, switched parents
} UVM_MOE_STATIC_ALIAS_TRANSITION_PARAMS;
typedef UVM_MOE_STATIC_ALIAS_TRANSITION_PARAMS UVM_MOE_STATIC_ALIAS_PARK_PARAMS;
typedef UVM_MOE_STATIC_ALIAS_TRANSITION_PARAMS UVM_MOE_STATIC_ALIAS_RESUME_PARAMS;

// Replace aliases in a parked session. The default form replaces only the
// read-only consumer set without changing writable upload aliases or physical
// backing. RESIZE submits the complete upload + consumer set: backing shared
// with the old set is retained, preallocated additions are adopted, and
// removed backing is released at COMMIT. Large model topologies are submitted
// as a BEGIN/append/COMMIT transaction. No consumer GPU PTE is installed by
// REBIND; a later RESUME publishes the new consumer aliases. Failed or aborted
// transactions before a mutating RESIZE commit leave the old set intact.
#define UVM_MOE_STATIC_ALIAS_REBIND                                   UVM_IOCTL_BASE(103)
#define UVM_MOE_STATIC_ALIAS_REBIND_FLAG_BEGIN                        0x00000001U
#define UVM_MOE_STATIC_ALIAS_REBIND_FLAG_COMMIT                       0x00000002U
#define UVM_MOE_STATIC_ALIAS_REBIND_FLAG_ABORT                        0x00000004U
#define UVM_MOE_STATIC_ALIAS_REBIND_FLAG_RESIZE                       0x00000008U
#define UVM_MOE_STATIC_ALIAS_REBIND_FLAGS_ALL                         \
    (UVM_MOE_STATIC_ALIAS_REBIND_FLAG_BEGIN |                         \
     UVM_MOE_STATIC_ALIAS_REBIND_FLAG_COMMIT |                        \
     UVM_MOE_STATIC_ALIAS_REBIND_FLAG_ABORT |                         \
     UVM_MOE_STATIC_ALIAS_REBIND_FLAG_RESIZE)

typedef struct
{
    NvU32 abi_version;                                // IN, ABI_VERSION
    NvU32 struct_size;                                // IN, sizeof(params)
    NvU32 flags;                                      // IN, REBIND_FLAG_*
    NvU32 reserved;                                   // IN, must be zero in v1
    NvU64 session_id NV_ALIGN_BYTES(8);               // IN, sealed parked session
    NvU64 rebind_id NV_ALIGN_BYTES(8);                // IN/OUT, zero on BEGIN
    NvU64 expected_consumer_alias_bytes NV_ALIGN_BYTES(8); // IN, complete txn/set
    NvU64 staged_consumer_alias_bytes NV_ALIGN_BYTES(8);   // OUT, staged txn/set
    NvU64 backing_bytes NV_ALIGN_BYTES(8);            // OUT, changed only by RESIZE
    NvU64 total_alias_bytes NV_ALIGN_BYTES(8);        // OUT, committed session
    NvU32 count;                                      // IN, entries in this call
    NvU32 accepted_count;                             // OUT, staged this call
    NvU32 expected_consumer_alias_count;              // IN, complete txn/set
    NvU32 staged_consumer_alias_count;                // OUT
    NvU32 upload_alias_count;                         // OUT, changed only by RESIZE
    NvU32 committed;                                  // OUT, boolean
    NvU32 poisoned;                                   // OUT, session state
    NvU32 reserved2;                                  // IN, must be zero in v1
    UVM_MOE_STATIC_ALIAS_ENTRY entries[UVM_MOE_STATIC_ALIAS_MAX_ENTRIES]; // IN/OUT
    NV_STATUS rmStatus;                               // OUT
} UVM_MOE_STATIC_ALIAS_REBIND_PARAMS;

// Remove complete backing identities from a sealed, non-parked static-alias
// session at a userspace-certified scheduler quiescence point. Every active
// upload and consumer alias belonging to each named backing identity must be
// supplied, so the driver can reject an incomplete delta before changing a
// PTE. Unmentioned aliases are neither scanned for publication nor remapped.
// Successful removal restores the submitted target VAs to ordinary UVM
// mappings, releases the corresponding slot backing, and leaves the remaining
// session sealed. A failure after mutation poisons the remaining session.
#define UVM_MOE_STATIC_ALIAS_QUIESCED_REMOVE                           UVM_IOCTL_BASE(104)
#define UVM_MOE_STATIC_ALIAS_QUIESCED_REMOVE_FLAG_RELEASE_BACKING      0x00000001U
typedef struct
{
    NvU32 abi_version;                                // IN, ABI_VERSION
    NvU32 struct_size;                                // IN, sizeof(params)
    NvU32 flags;                                      // IN, exactly RELEASE_BACKING
    NvU32 reserved;                                   // IN, must be zero
    NvU64 session_id NV_ALIGN_BYTES(8);               // IN, sealed non-zero session
    NvProcessorUuid gpu_uuid;                         // IN, session GPU
    NvU64 requested_bytes NV_ALIGN_BYTES(8);          // OUT, requested alias bytes
    NvU64 released_alias_bytes NV_ALIGN_BYTES(8);     // OUT, this call
    NvU64 total_alias_bytes NV_ALIGN_BYTES(8);        // OUT, remaining session
    NvU64 backing_bytes NV_ALIGN_BYTES(8);            // OUT, remaining backing
    NvU32 count;                                      // IN, exact records to remove
    NvU32 success_count;                              // OUT, removed records
    NvU32 failure_count;                              // OUT
    NvU32 remaining_alias_count;                      // OUT
    NvU32 tlb_invalidate_count;                       // OUT, removed PTE ranges
    NvU32 poisoned;                                   // OUT, remaining session state
    UVM_MOE_STATIC_ALIAS_ENTRY entries[UVM_MOE_STATIC_ALIAS_MAX_ENTRIES]; // IN/OUT
    NV_STATUS rmStatus;                               // OUT
} UVM_MOE_STATIC_ALIAS_QUIESCED_REMOVE_PARAMS;

typedef enum
{
    UVM_MOE_SLOT_POOL_STATE_EMPTY = 0,
    UVM_MOE_SLOT_POOL_STATE_RESERVED = 1,
    UVM_MOE_SLOT_POOL_STATE_LOADING = 2,
    UVM_MOE_SLOT_POOL_STATE_READY = 3,
    UVM_MOE_SLOT_POOL_STATE_IN_USE = 4,
    UVM_MOE_SLOT_POOL_STATE_RELEASED = 5,
    UVM_MOE_SLOT_POOL_STATE_ERROR = 6,
} UVM_MOE_SLOT_POOL_STATE;

typedef enum
{
    UVM_MOE_SLOT_POOL_BACKING_NONE = 0,
    UVM_MOE_SLOT_POOL_BACKING_SYSMEM = 1,
    UVM_MOE_SLOT_POOL_BACKING_SYSMEM_DMA = 2,
    UVM_MOE_SLOT_POOL_BACKING_VIDMEM = 3,
} UVM_MOE_SLOT_POOL_BACKING_KIND;

typedef enum
{
    UVM_REGION_STATE_UNKNOWN = 0,
    UVM_REGION_STATE_CPU = 1,
    UVM_REGION_STATE_PREFETCHING_CPU = 2,
    UVM_REGION_STATE_GPU = 3,
    UVM_REGION_STATE_PREFETCHING_GPU = 4,
} UVM_REGION_STATE;

#define UVM_UPDATE_REGION_STATE                                      UVM_IOCTL_BASE(87)
typedef struct
{
    NvU64     region_id NV_ALIGN_BYTES(8);  // IN
    NvU32     state;                        // IN, UVM_REGION_STATE_*
    NV_STATUS rmStatus;                     // OUT
} UVM_UPDATE_REGION_STATE_PARAMS;

#define UVM_QUERY_REGION                                             UVM_IOCTL_BASE(88)
typedef struct
{
    NvU64     region_id NV_ALIGN_BYTES(8);  // IN
    NvU64     base NV_ALIGN_BYTES(8);       // OUT
    NvU64     length NV_ALIGN_BYTES(8);     // OUT
    NvU32     state;                        // OUT, UVM_REGION_STATE_*
    NvU32     found_by_va;                  // OUT
    NV_STATUS rmStatus;                     // OUT
} UVM_QUERY_REGION_PARAMS;

#define UVM_QUERY_REGION_BY_VA                                       UVM_IOCTL_BASE(89)
typedef struct
{
    NvU64     address NV_ALIGN_BYTES(8);    // IN
    NvU64     region_id NV_ALIGN_BYTES(8);  // OUT
    NvU64     base NV_ALIGN_BYTES(8);       // OUT
    NvU64     length NV_ALIGN_BYTES(8);     // OUT
    NvU32     state;                        // OUT, UVM_REGION_STATE_*
    NvU32     found_by_va;                  // OUT
    NV_STATUS rmStatus;                     // OUT
} UVM_QUERY_REGION_BY_VA_PARAMS;

//
// UvmPhaseFPrefetchGpuBatchV2
//
// Synchronous managed-memory GPU prefetch with a single driver batch. This is
// a versioned ABI so the legacy 64-range Phase F executor remains unchanged.
// flags is reserved and must be zero.
//
#define UVM_PHASE_F_PREFETCH_GPU_BATCH_V2_MAX_RANGES                    128
#define UVM_PHASE_F_PREFETCH_GPU_BATCH_V2                               UVM_IOCTL_BASE(90)
typedef struct
{
    NvU32                    count;                                      // IN
    NvU32                    flags;                                      // IN, must be zero
    NvProcessorUuid          gpu_uuid;                                   // IN
    NvU32                    success_count;                              // OUT
    NvU32                    failure_count;                              // OUT
    NvU32                    core_calls;                                 // OUT, batch-core entries (0 or 1)
    NvU32                    tracker_waits;                              // OUT, aggregate migration waits (0 or 1)
    NvU64                    requested_bytes NV_ALIGN_BYTES(8);          // OUT, sum of accepted request lengths
    NvU64                    completed_bytes NV_ALIGN_BYTES(8);          // OUT, sum of successful request lengths
    UVM_PHASE_F_RANGE        ranges[UVM_PHASE_F_PREFETCH_GPU_BATCH_V2_MAX_RANGES];  // IN
    // OUT; status is authoritative. Residency processor-count fields are not queried.
    UVM_PHASE_F_RANGE_RESULT results[UVM_PHASE_F_PREFETCH_GPU_BATCH_V2_MAX_RANGES];
    NV_STATUS                rmStatus;                                   // OUT
} UVM_PHASE_F_PREFETCH_GPU_BATCH_V2_PARAMS;

//
// UvmExpertOffloadRelease
//
// Experimental targeted offload+release path for MoE experts. The managed VA
// range remains valid, but GPU PMM chunks backing the registered region are
// evicted through normal VA-block eviction and returned to PMM free lists.
//
#define UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REQUIRE_EXPERT                (1U << 0)
#define UVM_EXPERT_OFFLOAD_RELEASE_FLAG_SUBCHUNK_64K                  (1U << 2) // region must be 64KiB aligned
#define UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REMOTE_READ                   (1U << 3)
#define UVM_EXPERT_OFFLOAD_RELEASE_FLAG_IMMUTABLE_CPU_SHADOW          (1U << 4)

#define UVM_EXPERT_OFFLOAD_RELEASE                                   UVM_IOCTL_BASE(100)
typedef struct
{
    NvU64 region_id NV_ALIGN_BYTES(8);       // IN, phase_f registered region id
    NvProcessorUuid gpu_uuid;                // IN, target GPU
    NvU32 flags;                             // IN, UVM_EXPERT_OFFLOAD_RELEASE_FLAG_*
    NvU64 base NV_ALIGN_BYTES(8);            // OUT
    NvU64 length NV_ALIGN_BYTES(8);          // OUT
    NvU64 block_count NV_ALIGN_BYTES(8);     // OUT, existing VA blocks scanned
    NvU32 attempted_blocks;                  // OUT
    NvU32 success_blocks;                    // OUT
    NvU32 failure_blocks;                    // OUT
    NvU32 skipped_blocks;                    // OUT
    NvU64 attempted_bytes NV_ALIGN_BYTES(8); // OUT
    NvU64 released_bytes NV_ALIGN_BYTES(8);  // OUT, PMM root chunk bytes freed
    NvU64 failed_bytes NV_ALIGN_BYTES(8);    // OUT
    NvU32 remote_map_blocks;                 // OUT, blocks fully CPU-backed and remotely mapped
    NvU32 remote_map_failure_blocks;         // OUT
    NvU64 cpu_resident_bytes NV_ALIGN_BYTES(8); // OUT, requested bytes resident on CPU after release
    NvU64 remote_mapped_bytes NV_ALIGN_BYTES(8); // OUT, requested bytes readable through GPU remote PTEs
    NvU64 newly_remote_mapped_bytes NV_ALIGN_BYTES(8); // OUT, PTE bytes installed by this call
    NvU64 first_failed_block NV_ALIGN_BYTES(8); // OUT
    NV_STATUS first_error;                   // OUT
    NV_STATUS rmStatus;                      // OUT
} UVM_EXPERT_OFFLOAD_RELEASE_PARAMS;

//
// UvmDiscard
//
#define UVM_DISCARD                                                   UVM_IOCTL_BASE(80)
typedef struct
{
    NvU64           base                                    NV_ALIGN_BYTES(8); // IN
    NvU64           length                                  NV_ALIGN_BYTES(8); // IN
    NvU64           flags                                   NV_ALIGN_BYTES(8); // IN
    NV_STATUS       rmStatus;                                                  // OUT
} UVM_DISCARD_PARAMS;

//
// Temporary ioctls which should be removed before UVM 8 release
// Number backwards from 2047 - highest custom ioctl function number
// windows can handle.
//

//
// UvmIs8Supported
//
#define UVM_IS_8_SUPPORTED                                            UVM_IOCTL_BASE(2047)

typedef struct
{
    NvU32     is8Supported; // OUT
    NV_STATUS rmStatus;     // OUT
} UVM_IS_8_SUPPORTED_PARAMS;

#ifdef __cplusplus
}
#endif

#endif // _UVM_IOCTL_H
