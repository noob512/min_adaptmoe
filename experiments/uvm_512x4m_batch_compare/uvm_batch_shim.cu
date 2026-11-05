// SPDX-License-Identifier: Apache-2.0
// Minimal CUDA C ABI used by benchmark.py. The "batch" entry point is
// intentionally only a C++ loop over cudaMemPrefetchAsync, matching the
// repository's current cpp_batch semantics.

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <limits>
#include <new>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

struct DeviceRange {
    const uint32_t *pointer;
    uint64_t words;
};

struct Dataset {
    size_t allocation_bytes = 0;
    size_t range_bytes = 0;
    size_t guard_bytes = 0;
    std::vector<void *> test_allocations;
    std::vector<size_t> range_offsets;
    // Public logical ranges returned by uvm512_dataset_pointer.  Historical
    // datasets expose one range per allocation.  The Qwen packed-sparse
    // dataset exposes only the selected 512 KiB cold regions from two larger
    // contiguous tensor allocations.
    std::vector<void *> range_pointers;
    // Every range initialized by reseed and covered by the checksum.  This is
    // equal to range_pointers for historical datasets; Qwen uses all 1,152
    // tensor envelopes so hot-expert leaves are integrity checked too.
    std::vector<uint32_t *> seed_ranges;
    size_t checksum_range_count = 0;
    std::vector<void *> guard_allocations;
    DeviceRange *device_ranges = nullptr;
    unsigned long long *device_checksum = nullptr;
    uint64_t expected_checksum = 0;
};

uint64_t g_prefetch_one_outer_calls = 0;
uint64_t g_prefetch_batch_outer_calls = 0;
uint64_t g_runtime_prefetch_calls = 0;

// Production UVM ioctl ABI copied from uvm_ioctl.h. Keep this versioned
// definition local so this userspace shim does not depend on kernel headers.
constexpr unsigned int UVM_INITIALIZE_IOCTL = 0x30000001;
constexpr unsigned int UVM_PHASE_F_BATCH_EXECUTOR_IOCTL = 81;
constexpr unsigned int UVM_PHASE_F_PREFETCH_GPU_BATCH_V2_IOCTL = 90;
constexpr size_t UVM_PHASE_F_MAX_RANGES = 64;
constexpr size_t UVM_PHASE_F_PREFETCH_GPU_BATCH_V2_MAX_RANGES = 128;
constexpr uint32_t UVM_PHASE_F_ACTION_QUERY_RESIDENCY = 4;

struct UvmInitializeParams {
    uint64_t flags;
    uint32_t rm_status;
};

struct PhaseFUuid {
    uint8_t bytes[16];
};

struct PhaseFRange {
    uint64_t base;
    uint64_t length;
};

struct PhaseFRangeResult {
    uint32_t status;
    uint32_t resident_on_count;
    uint8_t resident_on_cpu;
    uint8_t resident_on_gpu;
    uint64_t queried_bytes;
    uint64_t resident_gpu_bytes;
    uint64_t resident_cpu_bytes;
    uint64_t resident_any_bytes;
    uint64_t resident_unknown_bytes;
};

struct PhaseFBatchExecutorParams {
    uint32_t action;
    uint32_t count;
    PhaseFUuid gpu_uuid;
    uint8_t async;
    uint32_t success_count;
    uint32_t failure_count;
    PhaseFRange ranges[UVM_PHASE_F_MAX_RANGES];
    PhaseFRangeResult results[UVM_PHASE_F_MAX_RANGES];
    uint32_t rm_status;
};

struct PhaseFPrefetchGpuBatchV2Params {
    uint32_t count;
    uint32_t flags;
    PhaseFUuid gpu_uuid;
    uint32_t success_count;
    uint32_t failure_count;
    uint32_t core_calls;
    uint32_t tracker_waits;
    uint64_t requested_bytes;
    uint64_t completed_bytes;
    PhaseFRange ranges[UVM_PHASE_F_PREFETCH_GPU_BATCH_V2_MAX_RANGES];
    PhaseFRangeResult results[UVM_PHASE_F_PREFETCH_GPU_BATCH_V2_MAX_RANGES];
    uint32_t rm_status;
};

static_assert(sizeof(PhaseFUuid) == 16, "PhaseFUuid ABI mismatch");
static_assert(sizeof(PhaseFRange) == 16, "PhaseFRange ABI mismatch");
static_assert(sizeof(PhaseFRangeResult) == 56,
              "PhaseFRangeResult ABI mismatch");
static_assert(offsetof(PhaseFRangeResult, queried_bytes) == 16,
              "PhaseFRangeResult alignment mismatch");
static_assert(offsetof(PhaseFBatchExecutorParams, ranges) == 40,
              "legacy batch ranges offset mismatch");
static_assert(offsetof(PhaseFBatchExecutorParams, results) == 1064,
              "legacy batch results offset mismatch");
static_assert(sizeof(PhaseFBatchExecutorParams) == 4656,
              "legacy batch size mismatch");
static_assert(offsetof(PhaseFPrefetchGpuBatchV2Params, ranges) == 56,
              "batch V2 ranges offset mismatch");
static_assert(offsetof(PhaseFPrefetchGpuBatchV2Params, results) == 2104,
              "batch V2 results offset mismatch");
static_assert(offsetof(PhaseFPrefetchGpuBatchV2Params, rm_status) == 9272,
              "batch V2 rmStatus offset mismatch");
static_assert(sizeof(PhaseFPrefetchGpuBatchV2Params) == 9280,
              "batch V2 size mismatch");

struct UvmFdCandidate {
    int source_fd;
    std::string path;
};

struct DriverBatchContext {
    int fd = -1;
    PhaseFUuid gpu_uuid{};
};

thread_local std::string g_driver_last_error;

void report_cuda_error(const char *operation, cudaError_t status);

void set_driver_error(const std::string &message)
{
    g_driver_last_error = message;
}

bool fd_link_is_uvm(const char *target)
{
    if (!target)
        return false;
    const char *base = std::strrchr(target, '/');
    base = base ? base + 1 : target;
    return std::strcmp(base, "nvidia-uvm") == 0 ||
           std::strncmp(base, "nvidia-uvm ", std::strlen("nvidia-uvm ")) == 0;
}

std::vector<UvmFdCandidate> collect_uvm_fd_candidates()
{
    std::vector<UvmFdCandidate> candidates;
    DIR *directory = opendir("/proc/self/fd");
    if (!directory) {
        set_driver_error("cannot open /proc/self/fd");
        return candidates;
    }

    const int directory_fd = dirfd(directory);
    while (dirent *entry = readdir(directory)) {
        if (entry->d_name[0] == '.')
            continue;
        char *end = nullptr;
        errno = 0;
        const long parsed = std::strtol(entry->d_name, &end, 10);
        if (errno != 0 || end == entry->d_name || *end != '\0' ||
            parsed < 0 || parsed > std::numeric_limits<int>::max()) {
            continue;
        }
        const int source_fd = static_cast<int>(parsed);
        if (source_fd == directory_fd)
            continue;

        char fd_path[64];
        char target[4096];
        std::snprintf(fd_path, sizeof(fd_path), "/proc/self/fd/%d", source_fd);
        const ssize_t length = readlink(fd_path, target, sizeof(target) - 1);
        if (length < 0)
            continue;
        target[length] = '\0';
        if (fd_link_is_uvm(target))
            candidates.push_back({source_fd, target});
    }
    closedir(directory);
    std::sort(candidates.begin(), candidates.end(),
              [](const UvmFdCandidate &left, const UvmFdCandidate &right) {
                  return left.source_fd < right.source_fd;
              });
    return candidates;
}

int duplicate_initialized_uvm_fd(const UvmFdCandidate &candidate)
{
    const int fd = fcntl(candidate.source_fd, F_DUPFD_CLOEXEC, 3);
    if (fd < 0)
        return -1;
    UvmInitializeParams params{};
    if (ioctl(fd, UVM_INITIALIZE_IOCTL, &params) != 0 ||
        params.rm_status != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

bool get_device_uuid(int device, PhaseFUuid *uuid)
{
    if (!uuid || device < 0)
        return false;
    cudaDeviceProp properties{};
    const cudaError_t status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        report_cuda_error("cudaGetDeviceProperties", status);
        return false;
    }
#if CUDART_VERSION >= 10000
    static_assert(sizeof(properties.uuid.bytes) == sizeof(uuid->bytes),
                  "CUDA UUID size mismatch");
    std::memcpy(uuid->bytes, properties.uuid.bytes, sizeof(uuid->bytes));
    return true;
#else
    return false;
#endif
}

uint64_t sum_sizes(const size_t *sizes, size_t count)
{
    uint64_t total = 0;
    for (size_t index = 0; index < count; ++index) {
        if (sizes[index] > std::numeric_limits<uint64_t>::max() - total)
            return std::numeric_limits<uint64_t>::max();
        total += sizes[index];
    }
    return total;
}

void report_cuda_error(const char *operation, cudaError_t status)
{
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(status));
}

void destroy_dataset_impl(Dataset *dataset)
{
    if (!dataset)
        return;
    if (dataset->device_ranges)
        cudaFree(dataset->device_ranges);
    if (dataset->device_checksum)
        cudaFree(dataset->device_checksum);
    for (void *pointer : dataset->test_allocations)
        cudaFree(pointer);
    for (void *pointer : dataset->guard_allocations)
        cudaFree(pointer);
    delete dataset;
}

__global__ void checksum_kernel(const DeviceRange *ranges, size_t range_count,
                                int blocks_per_range,
                                unsigned long long *output)
{
    const size_t range_index = blockIdx.x / blocks_per_range;
    const int range_part = blockIdx.x % blocks_per_range;
    if (range_index >= range_count)
        return;
    const DeviceRange range = ranges[range_index];
    unsigned long long local = 0;
    for (uint64_t index =
             static_cast<uint64_t>(range_part) * blockDim.x + threadIdx.x;
         index < range.words;
         index += static_cast<uint64_t>(blocks_per_range) * blockDim.x) {
        local += range.pointer[index];
    }

    __shared__ unsigned long long sums[256];
    sums[threadIdx.x] = local;
    __syncthreads();
    for (int width = blockDim.x / 2; width > 0; width /= 2) {
        if (threadIdx.x < width)
            sums[threadIdx.x] += sums[threadIdx.x + width];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(output, sums[0]);
}

Dataset *create_dataset_impl(size_t allocation_count,
                             size_t allocation_bytes,
                             size_t range_bytes,
                             size_t guard_bytes,
                             size_t range_offset_step,
                             size_t range_offset_period)
{
    if (allocation_count == 0 || allocation_bytes == 0 || range_bytes == 0 ||
        range_bytes > allocation_bytes ||
        range_offset_period == 0 ||
        allocation_bytes % sizeof(uint32_t) != 0 ||
        range_bytes % sizeof(uint32_t) != 0 ||
        range_offset_step % sizeof(uint32_t) != 0) {
        return nullptr;
    }

    Dataset *dataset = new (std::nothrow) Dataset();
    if (!dataset)
        return nullptr;
    dataset->allocation_bytes = allocation_bytes;
    dataset->range_bytes = range_bytes;
    dataset->guard_bytes = guard_bytes;

    std::vector<DeviceRange> host_ranges;
    host_ranges.reserve(allocation_count);
    for (size_t index = 0; index < allocation_count; ++index) {
        const size_t offset_index = index % range_offset_period;
        size_t range_offset;
        void *test_pointer = nullptr;

        if (range_offset_step != 0 &&
            offset_index > std::numeric_limits<size_t>::max() / range_offset_step) {
            destroy_dataset_impl(dataset);
            return nullptr;
        }
        range_offset = offset_index * range_offset_step;
        if (range_offset > allocation_bytes ||
            range_bytes > allocation_bytes - range_offset) {
            destroy_dataset_impl(dataset);
            return nullptr;
        }

        cudaError_t status = cudaMallocManaged(
            &test_pointer, allocation_bytes, cudaMemAttachGlobal);
        if (status != cudaSuccess) {
            report_cuda_error("cudaMallocManaged test allocation", status);
            destroy_dataset_impl(dataset);
            return nullptr;
        }
        dataset->test_allocations.push_back(test_pointer);
        dataset->range_offsets.push_back(range_offset);

        const uint32_t value = static_cast<uint32_t>(257 + index * 13);
        auto *allocation = static_cast<uint8_t *>(test_pointer);
        uint32_t *words = reinterpret_cast<uint32_t *>(allocation + range_offset);
        dataset->range_pointers.push_back(words);
        dataset->seed_ranges.push_back(words);
        const size_t word_count = range_bytes / sizeof(uint32_t);
        std::fill(words, words + word_count, value);
        dataset->expected_checksum +=
            static_cast<uint64_t>(value) * word_count;
        host_ranges.push_back({words, static_cast<uint64_t>(word_count)});

        if (guard_bytes > 0 && index + 1 < allocation_count) {
            void *guard_pointer = nullptr;
            status = cudaMallocManaged(&guard_pointer, guard_bytes,
                                       cudaMemAttachGlobal);
            if (status != cudaSuccess) {
                report_cuda_error("cudaMallocManaged guard allocation", status);
                destroy_dataset_impl(dataset);
                return nullptr;
            }
            dataset->guard_allocations.push_back(guard_pointer);
        }
    }

    cudaError_t status = cudaMalloc(
        &dataset->device_ranges, host_ranges.size() * sizeof(DeviceRange));
    if (status != cudaSuccess) {
        report_cuda_error("cudaMalloc device ranges", status);
        destroy_dataset_impl(dataset);
        return nullptr;
    }
    status = cudaMemcpy(dataset->device_ranges, host_ranges.data(),
                        host_ranges.size() * sizeof(DeviceRange),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        report_cuda_error("cudaMemcpy device ranges", status);
        destroy_dataset_impl(dataset);
        return nullptr;
    }
    status = cudaMalloc(&dataset->device_checksum,
                        sizeof(*dataset->device_checksum));
    if (status != cudaSuccess) {
        report_cuda_error("cudaMalloc device checksum", status);
        destroy_dataset_impl(dataset);
        return nullptr;
    }
    dataset->checksum_range_count = host_ranges.size();
    return dataset;
}

bool allocate_qwen_tensor(Dataset *dataset,
                          size_t tensor_bytes,
                          size_t desired_block_offset,
                          uint8_t **tensor_base_out)
{
    constexpr size_t block_bytes = 2 * 1024 * 1024;
    if (!dataset || !tensor_base_out || tensor_bytes == 0 ||
        desired_block_offset >= block_bytes ||
        desired_block_offset % dataset->range_bytes != 0 ||
        tensor_bytes > std::numeric_limits<size_t>::max() - block_bytes) {
        return false;
    }

    void *allocation = nullptr;
    const size_t allocation_bytes = tensor_bytes + block_bytes;
    cudaError_t status = cudaMallocManaged(
        &allocation, allocation_bytes, cudaMemAttachGlobal);
    if (status != cudaSuccess) {
        report_cuda_error("cudaMallocManaged Qwen tensor", status);
        return false;
    }
    dataset->test_allocations.push_back(allocation);

    const uintptr_t raw = reinterpret_cast<uintptr_t>(allocation);
    const size_t raw_offset = raw % block_bytes;
    const size_t padding =
        (desired_block_offset + block_bytes - raw_offset) % block_bytes;
    if (padding > allocation_bytes || tensor_bytes > allocation_bytes - padding) {
        return false;
    }
    *tensor_base_out = static_cast<uint8_t *>(allocation) + padding;
    return true;
}

Dataset *create_qwen_packed_sparse_impl(const uint32_t *expert_ids,
                                        size_t expert_count,
                                        size_t total_experts,
                                        size_t envelope_bytes,
                                        size_t w13_envelopes_per_expert,
                                        size_t w2_envelopes_per_expert,
                                        size_t w13_block_offset,
                                        size_t w2_block_offset)
{
    constexpr size_t block_bytes = 2 * 1024 * 1024;
    if (!expert_ids || expert_count == 0 || total_experts == 0 ||
        expert_count > total_experts ||
        envelope_bytes == 0 || envelope_bytes % sizeof(uint32_t) != 0 ||
        block_bytes % envelope_bytes != 0 ||
        w13_envelopes_per_expert == 0 || w2_envelopes_per_expert == 0 ||
        total_experts > std::numeric_limits<uint32_t>::max()) {
        return nullptr;
    }
    if (w13_envelopes_per_expert >
            std::numeric_limits<size_t>::max() / total_experts ||
        w2_envelopes_per_expert >
            std::numeric_limits<size_t>::max() / total_experts) {
        return nullptr;
    }
    const size_t w13_envelopes = total_experts * w13_envelopes_per_expert;
    const size_t w2_envelopes = total_experts * w2_envelopes_per_expert;
    if (w13_envelopes > std::numeric_limits<size_t>::max() / envelope_bytes ||
        w2_envelopes > std::numeric_limits<size_t>::max() / envelope_bytes) {
        return nullptr;
    }
    const size_t w13_bytes = w13_envelopes * envelope_bytes;
    const size_t w2_bytes = w2_envelopes * envelope_bytes;

    std::vector<bool> selected(total_experts, false);
    for (size_t index = 0; index < expert_count; ++index) {
        const uint32_t expert_id = expert_ids[index];
        if (expert_id >= total_experts || selected[expert_id])
            return nullptr;
        selected[expert_id] = true;
    }

    Dataset *dataset = new (std::nothrow) Dataset();
    if (!dataset)
        return nullptr;
    dataset->range_bytes = envelope_bytes;
    dataset->allocation_bytes = 0;

    uint8_t *w13_base = nullptr;
    uint8_t *w2_base = nullptr;
    if (!allocate_qwen_tensor(dataset, w13_bytes, w13_block_offset,
                              &w13_base) ||
        !allocate_qwen_tensor(dataset, w2_bytes, w2_block_offset,
                              &w2_base)) {
        destroy_dataset_impl(dataset);
        return nullptr;
    }

    std::vector<DeviceRange> host_ranges;
    host_ranges.reserve(w13_envelopes + w2_envelopes);
    dataset->seed_ranges.reserve(w13_envelopes + w2_envelopes);
    for (size_t index = 0; index < w13_envelopes; ++index) {
        auto *words = reinterpret_cast<uint32_t *>(
            w13_base + index * envelope_bytes);
        dataset->seed_ranges.push_back(words);
        host_ranges.push_back(
            {words, static_cast<uint64_t>(envelope_bytes / sizeof(uint32_t))});
    }
    for (size_t index = 0; index < w2_envelopes; ++index) {
        auto *words = reinterpret_cast<uint32_t *>(
            w2_base + index * envelope_bytes);
        dataset->seed_ranges.push_back(words);
        host_ranges.push_back(
            {words, static_cast<uint64_t>(envelope_bytes / sizeof(uint32_t))});
    }

    dataset->range_pointers.reserve(
        expert_count *
        (w13_envelopes_per_expert + w2_envelopes_per_expert));
    for (size_t selection_index = 0; selection_index < expert_count;
         ++selection_index) {
        const size_t expert_id = expert_ids[selection_index];
        for (size_t envelope = 0; envelope < w13_envelopes_per_expert;
             ++envelope) {
            dataset->range_pointers.push_back(
                w13_base +
                (expert_id * w13_envelopes_per_expert + envelope) *
                    envelope_bytes);
        }
        for (size_t envelope = 0; envelope < w2_envelopes_per_expert;
             ++envelope) {
            dataset->range_pointers.push_back(
                w2_base +
                (expert_id * w2_envelopes_per_expert + envelope) *
                    envelope_bytes);
        }
    }

    cudaError_t status = cudaMalloc(
        &dataset->device_ranges, host_ranges.size() * sizeof(DeviceRange));
    if (status == cudaSuccess) {
        status = cudaMemcpy(dataset->device_ranges, host_ranges.data(),
                            host_ranges.size() * sizeof(DeviceRange),
                            cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess) {
        status = cudaMalloc(&dataset->device_checksum,
                            sizeof(*dataset->device_checksum));
    }
    if (status != cudaSuccess) {
        report_cuda_error("initialize Qwen packed checksum", status);
        destroy_dataset_impl(dataset);
        return nullptr;
    }
    dataset->checksum_range_count = host_ranges.size();

    const size_t word_count = envelope_bytes / sizeof(uint32_t);
    for (size_t index = 0; index < dataset->seed_ranges.size(); ++index) {
        const uint32_t value = static_cast<uint32_t>(257u + index * 13u);
        std::fill(dataset->seed_ranges[index],
                  dataset->seed_ranges[index] + word_count, value);
        dataset->expected_checksum +=
            static_cast<uint64_t>(value) * word_count;
    }
    return dataset;
}

}  // namespace

extern "C" {

int uvm512_set_device(int device)
{
    return static_cast<int>(cudaSetDevice(device));
}

const char *uvm512_cuda_error_string(int status)
{
    return cudaGetErrorString(static_cast<cudaError_t>(status));
}

void *uvm512_dataset_create(size_t allocation_count,
                            size_t allocation_bytes,
                            size_t guard_bytes)
{
    return create_dataset_impl(allocation_count,
                               allocation_bytes,
                               allocation_bytes,
                               guard_bytes,
                               0,
                               1);
}

void *uvm512_dataset_create_view(size_t allocation_count,
                                 size_t allocation_bytes,
                                 size_t range_bytes,
                                 size_t guard_bytes)
{
    return create_dataset_impl(allocation_count,
                               allocation_bytes,
                               range_bytes,
                               guard_bytes,
                               0,
                               1);
}

void *uvm512_dataset_create_offset_cycle(size_t allocation_count,
                                         size_t allocation_bytes,
                                         size_t range_bytes,
                                         size_t range_offset_step,
                                         size_t range_offset_period,
                                         size_t guard_bytes)
{
    return create_dataset_impl(allocation_count,
                               allocation_bytes,
                               range_bytes,
                               guard_bytes,
                               range_offset_step,
                               range_offset_period);
}

void *uvm512_dataset_create_qwen_packed_sparse(
    const uint32_t *expert_ids,
    size_t expert_count,
    size_t total_experts,
    size_t envelope_bytes,
    size_t w13_envelopes_per_expert,
    size_t w2_envelopes_per_expert,
    size_t w13_block_offset,
    size_t w2_block_offset)
{
    return create_qwen_packed_sparse_impl(expert_ids,
                                          expert_count,
                                          total_experts,
                                          envelope_bytes,
                                          w13_envelopes_per_expert,
                                          w2_envelopes_per_expert,
                                          w13_block_offset,
                                          w2_block_offset);
}

void uvm512_dataset_destroy(void *opaque_dataset)
{
    destroy_dataset_impl(static_cast<Dataset *>(opaque_dataset));
}

size_t uvm512_dataset_count(void *opaque_dataset)
{
    Dataset *dataset = static_cast<Dataset *>(opaque_dataset);
    return dataset ? dataset->range_pointers.size() : 0;
}

size_t uvm512_dataset_allocation_bytes(void *opaque_dataset)
{
    Dataset *dataset = static_cast<Dataset *>(opaque_dataset);
    return dataset ? dataset->allocation_bytes : 0;
}

uintptr_t uvm512_dataset_pointer(void *opaque_dataset, size_t index)
{
    Dataset *dataset = static_cast<Dataset *>(opaque_dataset);
    if (!dataset || index >= dataset->range_pointers.size()) {
        return 0;
    }
    return reinterpret_cast<uintptr_t>(dataset->range_pointers[index]);
}

uint64_t uvm512_dataset_expected_checksum(void *opaque_dataset)
{
    Dataset *dataset = static_cast<Dataset *>(opaque_dataset);
    return dataset ? dataset->expected_checksum : 0;
}

int uvm512_dataset_reseed(void *opaque_dataset, uint32_t seed)
{
    Dataset *dataset = static_cast<Dataset *>(opaque_dataset);
    if (!dataset)
        return static_cast<int>(cudaErrorInvalidValue);

    dataset->expected_checksum = 0;
    const size_t word_count = dataset->range_bytes / sizeof(uint32_t);
    for (size_t index = 0; index < dataset->seed_ranges.size(); ++index) {
        // Include both a caller-controlled dataset seed and the range index.
        // Slot-rebind probes use different seeds for the old and new virtual
        // address sets, so a stale mapping can no longer pass merely because
        // both datasets were initialized with identical bytes.
        const uint32_t value =
            static_cast<uint32_t>(seed + 257u + static_cast<uint32_t>(index) * 13u);
        uint32_t *words = dataset->seed_ranges[index];
        std::fill(words, words + word_count, value);
        dataset->expected_checksum +=
            static_cast<uint64_t>(value) * word_count;
    }

    return static_cast<int>(cudaSuccess);
}

void *uvm512_stream_create()
{
    cudaStream_t stream = nullptr;
    const cudaError_t status =
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) {
        report_cuda_error("cudaStreamCreateWithFlags", status);
        return nullptr;
    }
    return reinterpret_cast<void *>(stream);
}

int uvm512_stream_synchronize(void *opaque_stream)
{
    return static_cast<int>(cudaStreamSynchronize(
        reinterpret_cast<cudaStream_t>(opaque_stream)));
}

void uvm512_stream_destroy(void *opaque_stream)
{
    if (opaque_stream)
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(opaque_stream));
}

void *uvm512_event_create()
{
    cudaEvent_t event = nullptr;
    const cudaError_t status = cudaEventCreate(&event);
    if (status != cudaSuccess) {
        report_cuda_error("cudaEventCreate", status);
        return nullptr;
    }
    return reinterpret_cast<void *>(event);
}

int uvm512_event_record(void *opaque_event, void *opaque_stream)
{
    return static_cast<int>(cudaEventRecord(
        reinterpret_cast<cudaEvent_t>(opaque_event),
        reinterpret_cast<cudaStream_t>(opaque_stream)));
}

int uvm512_event_synchronize(void *opaque_event)
{
    return static_cast<int>(
        cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(opaque_event)));
}

int uvm512_event_elapsed_ms(void *opaque_begin, void *opaque_end,
                            float *milliseconds)
{
    return static_cast<int>(cudaEventElapsedTime(
        milliseconds, reinterpret_cast<cudaEvent_t>(opaque_begin),
        reinterpret_cast<cudaEvent_t>(opaque_end)));
}

void uvm512_event_destroy(void *opaque_event)
{
    if (opaque_event)
        cudaEventDestroy(reinterpret_cast<cudaEvent_t>(opaque_event));
}

int uvm512_dataset_prefetch_cpu_and_sync(void *opaque_dataset,
                                         void *opaque_stream)
{
    Dataset *dataset = static_cast<Dataset *>(opaque_dataset);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(opaque_stream);
    if (!dataset || !stream)
        return -1;
    int accepted = 0;
    for (void *pointer : dataset->test_allocations) {
        const cudaError_t status = cudaMemPrefetchAsync(
            pointer, dataset->allocation_bytes, cudaCpuDeviceId, stream);
        if (status == cudaSuccess)
            ++accepted;
        else
            report_cuda_error("cudaMemPrefetchAsync reset CPU", status);
    }
    const cudaError_t sync_status = cudaStreamSynchronize(stream);
    if (sync_status != cudaSuccess) {
        report_cuda_error("reset CPU stream synchronize", sync_status);
        return -1;
    }
    return accepted;
}

void uvm512_prefetch_counter_reset()
{
    g_prefetch_one_outer_calls = 0;
    g_prefetch_batch_outer_calls = 0;
    g_runtime_prefetch_calls = 0;
}

int uvm512_prefetch_one(uintptr_t pointer, size_t bytes, int device,
                        void *opaque_stream)
{
    ++g_prefetch_one_outer_calls;
    ++g_runtime_prefetch_calls;
    const cudaError_t status = cudaMemPrefetchAsync(
        reinterpret_cast<void *>(pointer), bytes, device,
        reinterpret_cast<cudaStream_t>(opaque_stream));
    if (status != cudaSuccess) {
        report_cuda_error("cudaMemPrefetchAsync direct", status);
        return 0;
    }
    return 1;
}

int uvm512_prefetch_batch(const uintptr_t *pointers, const size_t *sizes,
                          size_t count, int device, void *opaque_stream)
{
    ++g_prefetch_batch_outer_calls;
    if (!pointers || !sizes || count == 0)
        return 0;
    int accepted = 0;
    for (size_t index = 0; index < count; ++index) {
        ++g_runtime_prefetch_calls;
        const cudaError_t status = cudaMemPrefetchAsync(
            reinterpret_cast<void *>(pointers[index]), sizes[index], device,
            reinterpret_cast<cudaStream_t>(opaque_stream));
        if (status == cudaSuccess)
            ++accepted;
        else
            report_cuda_error("cudaMemPrefetchAsync cpp_batch", status);
    }
    return accepted;
}

uint64_t uvm512_prefetch_one_outer_calls()
{
    return g_prefetch_one_outer_calls;
}

uint64_t uvm512_prefetch_batch_outer_calls()
{
    return g_prefetch_batch_outer_calls;
}

uint64_t uvm512_runtime_prefetch_calls()
{
    return g_runtime_prefetch_calls;
}

const char *uvm512_driver_last_error()
{
    return g_driver_last_error.c_str();
}

uint32_t uvm512_driver_batch_ioctl_command()
{
    return UVM_PHASE_F_PREFETCH_GPU_BATCH_V2_IOCTL;
}

size_t uvm512_driver_batch_max_ranges()
{
    return UVM_PHASE_F_PREFETCH_GPU_BATCH_V2_MAX_RANGES;
}

size_t uvm512_driver_batch_params_size()
{
    return sizeof(PhaseFPrefetchGpuBatchV2Params);
}

void *uvm512_driver_batch_context_create(int device, uintptr_t probe_pointer,
                                         size_t probe_bytes)
{
    g_driver_last_error.clear();
    if (device < 0 || probe_pointer == 0 || probe_bytes == 0) {
        set_driver_error("invalid driver context probe arguments");
        return nullptr;
    }

    PhaseFUuid gpu_uuid{};
    if (!get_device_uuid(device, &gpu_uuid)) {
        set_driver_error("failed to resolve CUDA device UUID");
        return nullptr;
    }
    const std::vector<UvmFdCandidate> candidates = collect_uvm_fd_candidates();
    if (candidates.empty()) {
        set_driver_error("no initialized CUDA /dev/nvidia-uvm fd found");
        return nullptr;
    }

    std::string last_attempt = "all CUDA UVM fd candidates rejected ioctl 90";
    for (const UvmFdCandidate &candidate : candidates) {
        const int fd = duplicate_initialized_uvm_fd(candidate);
        if (fd < 0)
            continue;

        PhaseFPrefetchGpuBatchV2Params params{};
        params.count = 1;
        params.flags = 0;
        params.gpu_uuid = gpu_uuid;
        params.ranges[0].base = probe_pointer;
        params.ranges[0].length = probe_bytes;
        errno = 0;
        const int result = ioctl(
            fd, UVM_PHASE_F_PREFETCH_GPU_BATCH_V2_IOCTL, &params);
        const int saved_errno = errno;
        const bool accepted =
            result == 0 && params.rm_status == 0 &&
            params.success_count == 1 && params.failure_count == 0 &&
            params.core_calls == 1 && params.tracker_waits == 1 &&
            params.requested_bytes == probe_bytes &&
            params.completed_bytes == probe_bytes &&
            params.results[0].status == 0;
        if (accepted) {
            DriverBatchContext *context =
                new (std::nothrow) DriverBatchContext();
            if (!context) {
                close(fd);
                set_driver_error("failed to allocate driver batch context");
                return nullptr;
            }
            context->fd = fd;
            context->gpu_uuid = gpu_uuid;
            return context;
        }

        char message[512];
        std::snprintf(
            message,
            sizeof(message),
            "ioctl90 probe rejected fd=%d path=%s ret=%d errno=%d "
            "rmStatus=%u success=%u failure=%u core=%u waits=%u "
            "requested=%llu completed=%llu result0=%u",
            candidate.source_fd,
            candidate.path.c_str(),
            result,
            saved_errno,
            params.rm_status,
            params.success_count,
            params.failure_count,
            params.core_calls,
            params.tracker_waits,
            static_cast<unsigned long long>(params.requested_bytes),
            static_cast<unsigned long long>(params.completed_bytes),
            params.results[0].status);
        last_attempt = message;
        close(fd);
    }
    set_driver_error(last_attempt);
    return nullptr;
}

void uvm512_driver_batch_context_destroy(void *opaque_context)
{
    DriverBatchContext *context =
        static_cast<DriverBatchContext *>(opaque_context);
    if (!context)
        return;
    if (context->fd >= 0)
        close(context->fd);
    delete context;
}

int uvm512_driver_prefetch_batch_fast(
    void *opaque_context,
    const uintptr_t *pointers,
    const size_t *sizes,
    size_t count,
    uint32_t *success_count,
    uint32_t *failure_count,
    uint64_t *requested_bytes,
    uint64_t *completed_bytes,
    uint32_t *rm_status)
{
    if (success_count)
        *success_count = 0;
    if (failure_count)
        *failure_count = 0;
    if (requested_bytes)
        *requested_bytes = 0;
    if (completed_bytes)
        *completed_bytes = 0;
    if (rm_status)
        *rm_status = 0;

    DriverBatchContext *context =
        static_cast<DriverBatchContext *>(opaque_context);
    if (!context || context->fd < 0 || !pointers || !sizes ||
        count == 0 || count > UVM_PHASE_F_PREFETCH_GPU_BATCH_V2_MAX_RANGES ||
        !success_count || !failure_count || !requested_bytes ||
        !completed_bytes || !rm_status) {
        set_driver_error("invalid driver batch arguments");
        return -1;
    }

    PhaseFPrefetchGpuBatchV2Params params{};
    params.count = static_cast<uint32_t>(count);
    params.flags = 0;
    params.gpu_uuid = context->gpu_uuid;
    for (size_t index = 0; index < count; ++index) {
        if (pointers[index] == 0 || sizes[index] == 0) {
            set_driver_error("driver batch contains an empty range");
            return -2;
        }
        params.ranges[index].base = pointers[index];
        params.ranges[index].length = sizes[index];
    }
    const uint64_t input_bytes = sum_sizes(sizes, count);

    errno = 0;
    const int result = ioctl(
        context->fd, UVM_PHASE_F_PREFETCH_GPU_BATCH_V2_IOCTL, &params);
    const int saved_errno = errno;
    *success_count = params.success_count;
    *failure_count = params.failure_count;
    *requested_bytes = params.requested_bytes;
    *completed_bytes = params.completed_bytes;
    *rm_status = params.rm_status;
    if (result != 0) {
        char message[256];
        std::snprintf(message, sizeof(message),
                      "ioctl90 syscall failed ret=%d errno=%d", result,
                      saved_errno);
        set_driver_error(message);
        return -3;
    }

    bool results_ok = true;
    for (size_t index = 0; index < count; ++index)
        results_ok = results_ok && params.results[index].status == 0;
    if (params.rm_status != 0 || params.success_count != count ||
        params.failure_count != 0 || params.core_calls != 1 ||
        params.tracker_waits != 1 || params.requested_bytes != input_bytes ||
        params.completed_bytes != input_bytes || !results_ok) {
        char message[384];
        std::snprintf(
            message,
            sizeof(message),
            "ioctl90 result contract failed count=%zu rmStatus=%u "
            "success=%u failure=%u core=%u waits=%u "
            "requested=%llu/%llu completed=%llu/%llu results_ok=%u",
            count,
            params.rm_status,
            params.success_count,
            params.failure_count,
            params.core_calls,
            params.tracker_waits,
            static_cast<unsigned long long>(params.requested_bytes),
            static_cast<unsigned long long>(input_bytes),
            static_cast<unsigned long long>(params.completed_bytes),
            static_cast<unsigned long long>(input_bytes),
            results_ok);
        set_driver_error(message);
        return -4;
    }
    g_driver_last_error.clear();
    return 0;
}

int uvm512_driver_query_residency(
    void *opaque_context,
    const uintptr_t *pointers,
    const size_t *sizes,
    size_t count,
    uint32_t *success_count,
    uint64_t *resident_gpu_bytes,
    uint64_t *resident_unknown_bytes)
{
    if (success_count)
        *success_count = 0;
    if (resident_gpu_bytes)
        *resident_gpu_bytes = 0;
    if (resident_unknown_bytes)
        *resident_unknown_bytes = 0;

    DriverBatchContext *context =
        static_cast<DriverBatchContext *>(opaque_context);
    if (!context || context->fd < 0 || !pointers || !sizes || count == 0 ||
        !success_count || !resident_gpu_bytes || !resident_unknown_bytes) {
        set_driver_error("invalid residency-query arguments");
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
            if (pointers[begin + index] == 0 || sizes[begin + index] == 0) {
                set_driver_error("residency query contains an empty range");
                return -2;
            }
            params.ranges[index].base = pointers[begin + index];
            params.ranges[index].length = sizes[begin + index];
        }

        errno = 0;
        const int result = ioctl(
            context->fd, UVM_PHASE_F_BATCH_EXECUTOR_IOCTL, &params);
        const int saved_errno = errno;
        if (result != 0 || params.rm_status != 0 ||
            params.success_count != chunk_count || params.failure_count != 0) {
            char message[320];
            std::snprintf(
                message,
                sizeof(message),
                "cmd81 residency query failed offset=%zu count=%zu ret=%d "
                "errno=%d rmStatus=%u success=%u failure=%u",
                begin,
                chunk_count,
                result,
                saved_errno,
                params.rm_status,
                params.success_count,
                params.failure_count);
            set_driver_error(message);
            return -3;
        }
        for (size_t index = 0; index < chunk_count; ++index) {
            const PhaseFRangeResult &entry = params.results[index];
            if (entry.status != 0 ||
                entry.queried_bytes != sizes[begin + index]) {
                char message[256];
                std::snprintf(
                    message,
                    sizeof(message),
                    "cmd81 residency result failed index=%zu status=%u "
                    "queried=%llu/%zu",
                    begin + index,
                    entry.status,
                    static_cast<unsigned long long>(entry.queried_bytes),
                    sizes[begin + index]);
                set_driver_error(message);
                return -4;
            }
            *resident_gpu_bytes += entry.resident_gpu_bytes;
            *resident_unknown_bytes += entry.resident_unknown_bytes;
            ++*success_count;
        }
    }
    g_driver_last_error.clear();
    return 0;
}

int uvm512_dataset_launch_checksum(void *opaque_dataset, void *opaque_stream)
{
    Dataset *dataset = static_cast<Dataset *>(opaque_dataset);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(opaque_stream);
    if (!dataset || !stream)
        return static_cast<int>(cudaErrorInvalidValue);
    cudaError_t status = cudaMemsetAsync(
        dataset->device_checksum, 0, sizeof(*dataset->device_checksum), stream);
    if (status != cudaSuccess)
        return static_cast<int>(status);
    constexpr int blocks_per_range = 4;
    if (dataset->checksum_range_count == 0)
        return static_cast<int>(cudaErrorInvalidValue);
    checksum_kernel<<<dataset->checksum_range_count * blocks_per_range,
                      256, 0, stream>>>(
        dataset->device_ranges, dataset->checksum_range_count,
        blocks_per_range, dataset->device_checksum);
    return static_cast<int>(cudaGetLastError());
}

int uvm512_dataset_read_checksum(void *opaque_dataset, uint64_t *checksum)
{
    Dataset *dataset = static_cast<Dataset *>(opaque_dataset);
    if (!dataset || !checksum)
        return static_cast<int>(cudaErrorInvalidValue);
    return static_cast<int>(cudaMemcpy(
        checksum, dataset->device_checksum, sizeof(*checksum),
        cudaMemcpyDeviceToHost));
}

int uvm512_dataset_count_last_prefetch_location(void *opaque_dataset,
                                                int location,
                                                size_t *matched)
{
    Dataset *dataset = static_cast<Dataset *>(opaque_dataset);
    if (!dataset || !matched)
        return static_cast<int>(cudaErrorInvalidValue);
    *matched = 0;
    for (void *pointer : dataset->test_allocations) {
        int observed = std::numeric_limits<int>::min();
        const cudaError_t status = cudaMemRangeGetAttribute(
            &observed, sizeof(observed),
            cudaMemRangeAttributeLastPrefetchLocation, pointer,
            dataset->allocation_bytes);
        if (status != cudaSuccess)
            return static_cast<int>(status);
        if (observed == location)
            ++*matched;
    }
    return static_cast<int>(cudaSuccess);
}

}  // extern "C"
