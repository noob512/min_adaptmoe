#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

namespace {

struct DeviceRange
{
    std::uint32_t *pointer;
    std::uint64_t words;
    std::uint32_t value;
};

struct GuardContext
{
    DeviceRange *device_ranges = nullptr;
    unsigned long long *device_checksum = nullptr;
    std::vector<void *> active_pointers;
    std::size_t count = 0;
    std::size_t complement_range_count = 0;
    std::size_t active_bytes = 0;
    std::uint64_t expected = 0;
};

__global__ void seed_complements(DeviceRange *ranges, std::size_t count)
{
    const std::size_t range_index = blockIdx.x;
    if (range_index >= count)
        return;
    const DeviceRange range = ranges[range_index];
    for (std::uint64_t index = threadIdx.x; index < range.words; index += blockDim.x)
        range.pointer[index] = range.value;
}

__global__ void checksum_complements(const DeviceRange *ranges,
                                     std::size_t count,
                                     unsigned long long *output)
{
    const std::size_t range_index = blockIdx.x;
    if (range_index >= count)
        return;
    const DeviceRange range = ranges[range_index];
    unsigned long long local = 0;
    for (std::uint64_t index = threadIdx.x; index < range.words; index += blockDim.x)
        local += range.pointer[index];

    __shared__ unsigned long long sums[256];
    sums[threadIdx.x] = local;
    __syncthreads();
    for (unsigned int width = blockDim.x / 2; width > 0; width /= 2) {
        if (threadIdx.x < width)
            sums[threadIdx.x] += sums[threadIdx.x + width];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(output, sums[0]);
}

void destroy_context(GuardContext *context)
{
    if (!context)
        return;
    if (context->device_checksum)
        cudaFree(context->device_checksum);
    if (context->device_ranges)
        cudaFree(context->device_ranges);
    delete context;
}

GuardContext *create_context(const std::uint64_t *active_pointers,
                             const std::size_t *active_offsets,
                             std::size_t count,
                             std::size_t allocation_bytes,
                             std::size_t active_bytes,
                             std::uint32_t seed,
                             void *opaque_stream)
{
    if (!active_pointers || count == 0 || active_bytes >= allocation_bytes ||
        active_bytes % sizeof(std::uint32_t) != 0 ||
        allocation_bytes % sizeof(std::uint32_t) != 0) {
        return nullptr;
    }

    GuardContext *context = new (std::nothrow) GuardContext();
    if (!context)
        return nullptr;
    context->count = count;
    context->active_bytes = active_bytes;
    context->active_pointers.reserve(count);

    std::vector<DeviceRange> ranges;
    ranges.reserve(2 * count);
    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t active_offset = active_offsets ? active_offsets[index] : 0;
        const std::uint32_t value =
            seed + 0x9e3779b9U + static_cast<std::uint32_t>(index * 17U);
        const std::uintptr_t active_address =
            static_cast<std::uintptr_t>(active_pointers[index]);
        auto *active = reinterpret_cast<std::uint8_t *>(active_address);

        if (!active || active_address < active_offset ||
            active_offset % sizeof(std::uint32_t) != 0 ||
            active_offset > allocation_bytes ||
            active_bytes > allocation_bytes - active_offset) {
            destroy_context(context);
            return nullptr;
        }

        auto *allocation = active - active_offset;
        context->active_pointers.push_back(active);
        if (active_offset != 0) {
            const std::uint64_t prefix_words =
                active_offset / sizeof(std::uint32_t);
            ranges.push_back(
                {reinterpret_cast<std::uint32_t *>(allocation),
                 prefix_words,
                 value});
            context->expected +=
                static_cast<std::uint64_t>(value) * prefix_words;
        }

        const std::size_t suffix_bytes =
            allocation_bytes - active_offset - active_bytes;
        if (suffix_bytes != 0) {
            const std::uint64_t suffix_words =
                suffix_bytes / sizeof(std::uint32_t);
            ranges.push_back(
                {reinterpret_cast<std::uint32_t *>(active + active_bytes),
                 suffix_words,
                 value});
            context->expected +=
                static_cast<std::uint64_t>(value) * suffix_words;
        }
    }
    context->complement_range_count = ranges.size();
    if (ranges.empty()) {
        destroy_context(context);
        return nullptr;
    }

    cudaError_t status = cudaMalloc(
        &context->device_ranges, ranges.size() * sizeof(DeviceRange));
    if (status == cudaSuccess) {
        status = cudaMemcpy(context->device_ranges,
                            ranges.data(),
                            ranges.size() * sizeof(DeviceRange),
                            cudaMemcpyHostToDevice);
    }
    if (status == cudaSuccess)
        status = cudaMalloc(&context->device_checksum, sizeof(unsigned long long));
    if (status != cudaSuccess) {
        destroy_context(context);
        return nullptr;
    }

    cudaStream_t stream = static_cast<cudaStream_t>(opaque_stream);
    seed_complements<<<static_cast<unsigned int>(ranges.size()), 256, 0, stream>>>(
        context->device_ranges, ranges.size());
    status = cudaGetLastError();
    if (status == cudaSuccess)
        status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
        destroy_context(context);
        return nullptr;
    }

    return context;
}

}  // namespace

extern "C" {

void *uvm_fragment_guard_create(const std::uint64_t *pointers,
                                std::size_t count,
                                std::size_t allocation_bytes,
                                std::size_t active_bytes,
                                std::uint32_t seed,
                                void *opaque_stream)
{
    return create_context(pointers,
                          nullptr,
                          count,
                          allocation_bytes,
                          active_bytes,
                          seed,
                          opaque_stream);
}

void *uvm_fragment_guard_create_offset(const std::uint64_t *active_pointers,
                                       const std::size_t *active_offsets,
                                       std::size_t count,
                                       std::size_t allocation_bytes,
                                       std::size_t active_bytes,
                                       std::uint32_t seed,
                                       void *opaque_stream)
{
    if (!active_offsets)
        return nullptr;
    return create_context(active_pointers,
                          active_offsets,
                          count,
                          allocation_bytes,
                          active_bytes,
                          seed,
                          opaque_stream);
}

int uvm_fragment_guard_restore_active_cpu(void *opaque_context,
                                          void *opaque_stream)
{
    GuardContext *context = static_cast<GuardContext *>(opaque_context);
    if (!context || context->active_pointers.size() != context->count ||
        context->active_bytes == 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    // The complement seed kernel can fault-prefetch neighbouring pages from
    // the same 2 MiB managed allocation. Registered copy plans require the
    // active 512 KiB source pages to be CPU-resident and not GPU-resident.
    // Prefetch only that aligned subrange: the remaining 1.5 MiB complement
    // stays GPU-accessible and is checked independently after every execute.
    cudaStream_t stream = static_cast<cudaStream_t>(opaque_stream);
    for (void *pointer : context->active_pointers) {
        const cudaError_t status = cudaMemPrefetchAsync(pointer,
                                                        context->active_bytes,
                                                        cudaCpuDeviceId,
                                                        stream);
        if (status != cudaSuccess)
            return static_cast<int>(status);
    }

    return static_cast<int>(cudaStreamSynchronize(stream));
}

void uvm_fragment_guard_destroy(void *opaque_context)
{
    destroy_context(static_cast<GuardContext *>(opaque_context));
}

std::uint64_t uvm_fragment_guard_expected_checksum(void *opaque_context)
{
    GuardContext *context = static_cast<GuardContext *>(opaque_context);
    return context ? context->expected : 0;
}

int uvm_fragment_guard_launch_checksum(void *opaque_context, void *opaque_stream)
{
    GuardContext *context = static_cast<GuardContext *>(opaque_context);
    if (!context)
        return static_cast<int>(cudaErrorInvalidValue);
    cudaStream_t stream = static_cast<cudaStream_t>(opaque_stream);
    cudaError_t status = cudaMemsetAsync(
        context->device_checksum, 0, sizeof(unsigned long long), stream);
    if (status != cudaSuccess)
        return static_cast<int>(status);
    checksum_complements<<<static_cast<unsigned int>(context->complement_range_count),
                           256,
                           0,
                           stream>>>(context->device_ranges,
                                    context->complement_range_count,
                                    context->device_checksum);
    return static_cast<int>(cudaGetLastError());
}

int uvm_fragment_guard_read_checksum(void *opaque_context, std::uint64_t *checksum)
{
    GuardContext *context = static_cast<GuardContext *>(opaque_context);
    if (!context || !checksum)
        return static_cast<int>(cudaErrorInvalidValue);
    return static_cast<int>(cudaMemcpy(checksum,
                                       context->device_checksum,
                                       sizeof(*checksum),
                                       cudaMemcpyDeviceToHost));
}

}  // extern "C"
