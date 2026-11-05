// SPDX-License-Identifier: Apache-2.0
// A standalone pinned-copy / managed-prefetch / mapped-UVA microbenchmark.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr size_t KiB = 1024ULL;
constexpr size_t MiB = 1024ULL * KiB;
constexpr double GiB = 1024.0 * 1024.0 * 1024.0;

struct Args {
    int device = 0;
    size_t cold_bytes = 64 * MiB;
    int tensors = 16;
    int slots = 2;
    int warmups = 1;
    int repeats = 3;
    double equivalent_ratio = 0.80;
    std::string only_layout;
    std::string only_backend;
    std::string only_mode;
};

struct DeviceRange {
    const uint32_t *ptr;
    uint64_t words;
};

struct LogicalRange {
    uint8_t *host = nullptr;
    uint8_t *mapped = nullptr;
    uint8_t *managed = nullptr;
    size_t bytes = 0;
    int host_allocation = -1;
    int managed_allocation = -1;
    uint8_t pattern = 0;
};

struct DeviceLogicalRange {
    uint8_t *ptr = nullptr;
    size_t bytes = 0;
    int allocation = -1;
};

struct CopyOp {
    uint8_t *source = nullptr;
    uint8_t *destination = nullptr;
    size_t bytes = 0;
    int source_allocation = -1;
    int destination_allocation = -1;
};

struct ManagedOp {
    uint8_t *ptr = nullptr;
    size_t bytes = 0;
    int allocation = -1;
};

struct Slot {
    std::vector<void *> allocations;
    std::vector<DeviceLogicalRange> cold;
    DeviceRange *device_descriptors = nullptr;
};

struct PinnedDataset {
    std::string layout;
    size_t cold_bytes = 0;
    size_t tensor_bytes = 0;
    std::vector<void *> host_allocations;
    std::vector<LogicalRange> cold;
    std::vector<LogicalRange> hot;
    std::vector<Slot> slots;
    DeviceRange *uva_descriptors = nullptr;
    uint64_t expected_checksum = 0;
    bool mapped_address_equal = true;
};

struct ManagedDataset {
    std::string layout;
    size_t cold_bytes = 0;
    size_t tensor_bytes = 0;
    std::vector<void *> allocations;
    std::vector<LogicalRange> cold;
    std::vector<LogicalRange> hot;
    DeviceRange *device_descriptors = nullptr;
    uint64_t expected_checksum = 0;
    std::vector<uintptr_t> original_addresses;
};

enum class Backend { PinnedCopy, ManagedPrefetch, UvaDirect };
enum class SubmitMode { PerTensorSync, TensorBatch, CoalescedBatch, Direct };

struct CaseDef {
    Backend backend;
    SubmitMode mode;
};

struct Sample {
    double submit_us = 0.0;
    double enqueue_span_us = 0.0;
    double migration_ms = 0.0;
    double exposed_wait_ms = 0.0;
    double read_ms = 0.0;
    double e2e_cuda_ms = 0.0;
    double host_block_ms = 0.0;
    uint64_t checksum = 0;
    bool checksum_ok = false;
    bool placement_ok = true;
    uintptr_t consumer_fingerprint = 0;
    int submitted_ranges = 0;
};

struct CaseResult {
    std::string layout;
    Backend backend;
    SubmitMode mode;
    int logical_ranges = 0;
    int submitted_ranges = 0;
    size_t bytes = 0;
    std::vector<Sample> samples;
    bool functional_ok = false;
    bool same_consumer_va = true;
    bool in_place_va = false;
    double submit_us = 0.0;
    double api_call_us = 0.0;
    double migration_ms = 0.0;
    double exposed_wait_ms = 0.0;
    double read_ms = 0.0;
    double e2e_ms = 0.0;
    double host_block_ms = 0.0;
    double migration_gib_s = std::numeric_limits<double>::quiet_NaN();
    double effective_gib_s = 0.0;
};

[[noreturn]] void fail(const char *message)
{
    std::fprintf(stderr, "error: %s\n", message);
    std::exit(2);
}

void check(cudaError_t status, const char *operation)
{
    if (status == cudaSuccess)
        return;
    std::fprintf(stderr, "%s failed: %s\n", operation,
                 cudaGetErrorString(status));
    std::exit(1);
}

long parse_long(const char *value, const char *name)
{
    char *end = nullptr;
    const long result = std::strtol(value, &end, 10);
    if (end == value || *end != '\0') {
        std::fprintf(stderr, "invalid %s: %s\n", name, value);
        std::exit(2);
    }
    return result;
}

double parse_double(const char *value, const char *name)
{
    char *end = nullptr;
    const double result = std::strtod(value, &end);
    if (end == value || *end != '\0' || !std::isfinite(result)) {
        std::fprintf(stderr, "invalid %s: %s\n", name, value);
        std::exit(2);
    }
    return result;
}

Args parse_args(int argc, char **argv)
{
    Args args;
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--device") == 0 && index + 1 < argc)
            args.device = static_cast<int>(parse_long(argv[++index], "device"));
        else if (std::strcmp(argv[index], "--cold-mib") == 0 && index + 1 < argc)
            args.cold_bytes = static_cast<size_t>(
                parse_double(argv[++index], "cold-mib") * MiB);
        else if (std::strcmp(argv[index], "--tensors") == 0 && index + 1 < argc)
            args.tensors = static_cast<int>(parse_long(argv[++index], "tensors"));
        else if (std::strcmp(argv[index], "--slots") == 0 && index + 1 < argc)
            args.slots = static_cast<int>(parse_long(argv[++index], "slots"));
        else if (std::strcmp(argv[index], "--warmups") == 0 && index + 1 < argc)
            args.warmups = static_cast<int>(parse_long(argv[++index], "warmups"));
        else if (std::strcmp(argv[index], "--repeats") == 0 && index + 1 < argc)
            args.repeats = static_cast<int>(parse_long(argv[++index], "repeats"));
        else if (std::strcmp(argv[index], "--equivalent-ratio") == 0 &&
                 index + 1 < argc)
            args.equivalent_ratio =
                parse_double(argv[++index], "equivalent-ratio");
        else if (std::strcmp(argv[index], "--only-layout") == 0 &&
                 index + 1 < argc)
            args.only_layout = argv[++index];
        else if (std::strcmp(argv[index], "--only-backend") == 0 &&
                 index + 1 < argc)
            args.only_backend = argv[++index];
        else if (std::strcmp(argv[index], "--only-mode") == 0 &&
                 index + 1 < argc)
            args.only_mode = argv[++index];
        else {
            std::fprintf(
                stderr,
                "usage: %s [--device N] [--cold-mib N] [--tensors N] "
                "[--slots N] [--warmups N] [--repeats N] "
                "[--equivalent-ratio R] "
                "[--only-layout NAME] [--only-backend NAME] "
                "[--only-mode NAME]\n",
                argv[0]);
            std::exit(2);
        }
    }

    if (args.device < 0 || args.tensors < 1 || args.tensors > 512 ||
        args.slots < 1 || args.slots > 8 || args.warmups < 0 ||
        args.warmups > 20 || args.repeats < 1 || args.repeats > 100 ||
        args.equivalent_ratio <= 0.0 || args.equivalent_ratio > 2.0 ||
        args.cold_bytes == 0 || args.cold_bytes % args.tensors != 0) {
        fail("invalid benchmark arguments");
    }
    const size_t tensor_bytes = args.cold_bytes / args.tensors;
    if (tensor_bytes < 64 * KiB || tensor_bytes % sizeof(uint32_t) != 0)
        fail("each tensor must be >=64 KiB and uint32 aligned");
    const std::vector<std::string> layouts = {
        "multi_alloc", "arena_interleaved", "arena_contiguous"};
    const std::vector<std::string> backends = {
        "pinned_copy", "managed_prefetch", "uva_direct"};
    const std::vector<std::string> modes = {
        "per_tensor_sync", "tensor_batch", "coalesced_batch", "direct"};
    if ((!args.only_layout.empty() &&
         std::find(layouts.begin(), layouts.end(), args.only_layout) ==
             layouts.end()) ||
        (!args.only_backend.empty() &&
         std::find(backends.begin(), backends.end(), args.only_backend) ==
             backends.end()) ||
        (!args.only_mode.empty() &&
         std::find(modes.begin(), modes.end(), args.only_mode) == modes.end())) {
        fail("invalid --only-layout/--only-backend/--only-mode value");
    }
    return args;
}

__global__ void checksum_ranges_kernel(const DeviceRange *ranges,
                                       int range_count,
                                       int blocks_per_range,
                                       unsigned long long *output)
{
    const int range_index = blockIdx.x / blocks_per_range;
    const int range_part = blockIdx.x % blocks_per_range;
    if (range_index >= range_count)
        return;
    const DeviceRange range = ranges[range_index];
    unsigned long long local = 0;
    for (uint64_t index =
             static_cast<uint64_t>(range_part) * blockDim.x + threadIdx.x;
         index < range.words;
         index += static_cast<uint64_t>(blocks_per_range) * blockDim.x) {
        local += range.ptr[index];
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

uint8_t cold_pattern(int index)
{
    return static_cast<uint8_t>(17 + (index * 29) % 211);
}

uint8_t hot_pattern(int index)
{
    return static_cast<uint8_t>(23 + (index * 17) % 197);
}

uint64_t pattern_checksum(uint8_t pattern, size_t bytes)
{
    const uint64_t word = static_cast<uint64_t>(pattern) * 0x01010101ULL;
    return word * static_cast<uint64_t>(bytes / sizeof(uint32_t));
}

std::vector<ManagedOp> merge_managed_ops(std::vector<ManagedOp> ops)
{
    if (ops.empty())
        return ops;
    std::vector<ManagedOp> result;
    result.push_back(ops.front());
    for (size_t index = 1; index < ops.size(); ++index) {
        ManagedOp &last = result.back();
        const ManagedOp &next = ops[index];
        if (last.allocation == next.allocation &&
            last.ptr + last.bytes == next.ptr) {
            last.bytes += next.bytes;
        }
        else {
            result.push_back(next);
        }
    }
    return result;
}

std::vector<CopyOp> merge_copy_ops(std::vector<CopyOp> ops)
{
    if (ops.empty())
        return ops;
    std::vector<CopyOp> result;
    result.push_back(ops.front());
    for (size_t index = 1; index < ops.size(); ++index) {
        CopyOp &last = result.back();
        const CopyOp &next = ops[index];
        if (last.source_allocation == next.source_allocation &&
            last.destination_allocation == next.destination_allocation &&
            last.source + last.bytes == next.source &&
            last.destination + last.bytes == next.destination) {
            last.bytes += next.bytes;
        }
        else {
            result.push_back(next);
        }
    }
    return result;
}

DeviceRange *make_device_descriptors(
    const std::vector<DeviceLogicalRange> &ranges)
{
    std::vector<DeviceRange> descriptors;
    descriptors.reserve(ranges.size());
    for (const DeviceLogicalRange &range : ranges) {
        descriptors.push_back(
            {reinterpret_cast<const uint32_t *>(range.ptr),
             static_cast<uint64_t>(range.bytes / sizeof(uint32_t))});
    }
    DeviceRange *device = nullptr;
    check(cudaMalloc(&device, descriptors.size() * sizeof(DeviceRange)),
          "cudaMalloc descriptors");
    check(cudaMemcpy(device, descriptors.data(),
                     descriptors.size() * sizeof(DeviceRange),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy descriptors");
    return device;
}

DeviceRange *make_uva_descriptors(const std::vector<LogicalRange> &ranges)
{
    std::vector<DeviceLogicalRange> mapped;
    mapped.reserve(ranges.size());
    for (const LogicalRange &range : ranges)
        mapped.push_back({range.mapped, range.bytes, range.host_allocation});
    return make_device_descriptors(mapped);
}

void allocate_pinned_range(size_t bytes, void **host, void **mapped)
{
    check(cudaHostAlloc(host, bytes, cudaHostAllocMapped), "cudaHostAllocMapped");
    check(cudaHostGetDevicePointer(mapped, *host, 0),
          "cudaHostGetDevicePointer");
}

PinnedDataset make_pinned_dataset(const std::string &layout,
                                  const Args &args)
{
    PinnedDataset data;
    data.layout = layout;
    data.cold_bytes = args.cold_bytes;
    data.tensor_bytes = args.cold_bytes / args.tensors;
    const size_t arena_bytes = 2 * args.cold_bytes;

    if (layout == "multi_alloc") {
        for (int index = 0; index < 2 * args.tensors; ++index) {
            void *host = nullptr;
            void *mapped = nullptr;
            allocate_pinned_range(data.tensor_bytes, &host, &mapped);
            data.host_allocations.push_back(host);
            LogicalRange range;
            range.host = static_cast<uint8_t *>(host);
            range.mapped = static_cast<uint8_t *>(mapped);
            range.bytes = data.tensor_bytes;
            range.host_allocation = index;
            if (index < args.tensors) {
                range.pattern = cold_pattern(index);
                data.cold.push_back(range);
                data.expected_checksum +=
                    pattern_checksum(range.pattern, range.bytes);
            }
            else {
                range.pattern = hot_pattern(index - args.tensors);
                data.hot.push_back(range);
            }
            std::memset(range.host, range.pattern, range.bytes);
            data.mapped_address_equal &= host == mapped;
        }
    }
    else {
        void *host = nullptr;
        void *mapped = nullptr;
        allocate_pinned_range(arena_bytes, &host, &mapped);
        data.host_allocations.push_back(host);
        data.mapped_address_equal = host == mapped;
        for (int index = 0; index < args.tensors; ++index) {
            const size_t cold_offset =
                layout == "arena_contiguous"
                    ? static_cast<size_t>(index) * data.tensor_bytes
                    : static_cast<size_t>(2 * index) * data.tensor_bytes;
            const size_t hot_offset =
                layout == "arena_contiguous"
                    ? args.cold_bytes +
                          static_cast<size_t>(index) * data.tensor_bytes
                    : static_cast<size_t>(2 * index + 1) * data.tensor_bytes;
            LogicalRange cold;
            cold.host = static_cast<uint8_t *>(host) + cold_offset;
            cold.mapped = static_cast<uint8_t *>(mapped) + cold_offset;
            cold.bytes = data.tensor_bytes;
            cold.host_allocation = 0;
            cold.pattern = cold_pattern(index);
            std::memset(cold.host, cold.pattern, cold.bytes);
            data.expected_checksum +=
                pattern_checksum(cold.pattern, cold.bytes);
            data.cold.push_back(cold);

            LogicalRange hot;
            hot.host = static_cast<uint8_t *>(host) + hot_offset;
            hot.mapped = static_cast<uint8_t *>(mapped) + hot_offset;
            hot.bytes = data.tensor_bytes;
            hot.host_allocation = 0;
            hot.pattern = hot_pattern(index);
            std::memset(hot.host, hot.pattern, hot.bytes);
            data.hot.push_back(hot);
        }
    }

    data.uva_descriptors = make_uva_descriptors(data.cold);
    for (int slot_index = 0; slot_index < args.slots; ++slot_index) {
        Slot slot;
        if (layout == "multi_alloc") {
            for (int index = 0; index < 2 * args.tensors; ++index) {
                void *device = nullptr;
                check(cudaMalloc(&device, data.tensor_bytes),
                      "cudaMalloc split slot");
                slot.allocations.push_back(device);
                if (index < args.tensors) {
                    slot.cold.push_back(
                        {static_cast<uint8_t *>(device), data.tensor_bytes, index});
                }
                else {
                    check(cudaMemset(device, hot_pattern(index - args.tensors),
                                     data.tensor_bytes),
                          "cudaMemset hot split slot");
                }
            }
        }
        else {
            void *device = nullptr;
            check(cudaMalloc(&device, arena_bytes), "cudaMalloc arena slot");
            slot.allocations.push_back(device);
            for (int index = 0; index < args.tensors; ++index) {
                const size_t cold_offset =
                    layout == "arena_contiguous"
                        ? static_cast<size_t>(index) * data.tensor_bytes
                        : static_cast<size_t>(2 * index) * data.tensor_bytes;
                const size_t hot_offset =
                    layout == "arena_contiguous"
                        ? args.cold_bytes +
                              static_cast<size_t>(index) * data.tensor_bytes
                        : static_cast<size_t>(2 * index + 1) * data.tensor_bytes;
                slot.cold.push_back(
                    {static_cast<uint8_t *>(device) + cold_offset,
                     data.tensor_bytes, 0});
                check(cudaMemset(static_cast<uint8_t *>(device) + hot_offset,
                                 hot_pattern(index), data.tensor_bytes),
                      "cudaMemset hot arena slot");
            }
        }
        slot.device_descriptors = make_device_descriptors(slot.cold);
        data.slots.push_back(std::move(slot));
    }
    check(cudaDeviceSynchronize(), "pinned dataset setup synchronize");
    return data;
}

ManagedDataset make_managed_dataset(const std::string &layout,
                                    const Args &args,
                                    cudaStream_t stream)
{
    ManagedDataset data;
    data.layout = layout;
    data.cold_bytes = args.cold_bytes;
    data.tensor_bytes = args.cold_bytes / args.tensors;
    const size_t arena_bytes = 2 * args.cold_bytes;

    if (layout == "multi_alloc") {
        for (int index = 0; index < 2 * args.tensors; ++index) {
            void *managed = nullptr;
            check(cudaMallocManaged(&managed, data.tensor_bytes,
                                    cudaMemAttachGlobal),
                  "cudaMallocManaged split");
            data.allocations.push_back(managed);
            data.original_addresses.push_back(
                reinterpret_cast<uintptr_t>(managed));
            LogicalRange range;
            range.managed = static_cast<uint8_t *>(managed);
            range.bytes = data.tensor_bytes;
            range.managed_allocation = index;
            if (index < args.tensors) {
                range.pattern = cold_pattern(index);
                data.cold.push_back(range);
                data.expected_checksum +=
                    pattern_checksum(range.pattern, range.bytes);
            }
            else {
                range.pattern = hot_pattern(index - args.tensors);
                data.hot.push_back(range);
            }
            std::memset(range.managed, range.pattern, range.bytes);
        }
    }
    else {
        void *managed = nullptr;
        check(cudaMallocManaged(&managed, arena_bytes, cudaMemAttachGlobal),
              "cudaMallocManaged arena");
        data.allocations.push_back(managed);
        data.original_addresses.push_back(reinterpret_cast<uintptr_t>(managed));
        for (int index = 0; index < args.tensors; ++index) {
            const size_t cold_offset =
                layout == "arena_contiguous"
                    ? static_cast<size_t>(index) * data.tensor_bytes
                    : static_cast<size_t>(2 * index) * data.tensor_bytes;
            const size_t hot_offset =
                layout == "arena_contiguous"
                    ? args.cold_bytes +
                          static_cast<size_t>(index) * data.tensor_bytes
                    : static_cast<size_t>(2 * index + 1) * data.tensor_bytes;
            LogicalRange cold;
            cold.managed = static_cast<uint8_t *>(managed) + cold_offset;
            cold.bytes = data.tensor_bytes;
            cold.managed_allocation = 0;
            cold.pattern = cold_pattern(index);
            std::memset(cold.managed, cold.pattern, cold.bytes);
            data.expected_checksum +=
                pattern_checksum(cold.pattern, cold.bytes);
            data.cold.push_back(cold);

            LogicalRange hot;
            hot.managed = static_cast<uint8_t *>(managed) + hot_offset;
            hot.bytes = data.tensor_bytes;
            hot.managed_allocation = 0;
            hot.pattern = hot_pattern(index);
            std::memset(hot.managed, hot.pattern, hot.bytes);
            data.hot.push_back(hot);
        }
    }

    std::vector<ManagedOp> hot_ops;
    for (const LogicalRange &range : data.hot)
        hot_ops.push_back(
            {range.managed, range.bytes, range.managed_allocation});
    hot_ops = merge_managed_ops(std::move(hot_ops));
    for (const ManagedOp &op : hot_ops)
        check(cudaMemPrefetchAsync(op.ptr, op.bytes, args.device, stream),
              "initial hot cudaMemPrefetchAsync");
    check(cudaStreamSynchronize(stream), "initial hot prefetch synchronize");

    std::vector<DeviceLogicalRange> cold_device;
    for (const LogicalRange &range : data.cold)
        cold_device.push_back(
            {range.managed, range.bytes, range.managed_allocation});
    data.device_descriptors = make_device_descriptors(cold_device);
    return data;
}

void destroy_pinned_dataset(PinnedDataset &data)
{
    if (data.uva_descriptors)
        cudaFree(data.uva_descriptors);
    for (Slot &slot : data.slots) {
        if (slot.device_descriptors)
            cudaFree(slot.device_descriptors);
        for (void *ptr : slot.allocations)
            cudaFree(ptr);
    }
    for (void *ptr : data.host_allocations)
        cudaFreeHost(ptr);
}

void destroy_managed_dataset(ManagedDataset &data)
{
    if (data.device_descriptors)
        cudaFree(data.device_descriptors);
    for (void *ptr : data.allocations)
        cudaFree(ptr);
}

std::vector<CopyOp> copy_ops(const PinnedDataset &data, int slot_index,
                             SubmitMode mode)
{
    std::vector<CopyOp> result;
    const Slot &slot = data.slots[slot_index];
    for (size_t index = 0; index < data.cold.size(); ++index) {
        result.push_back({data.cold[index].host, slot.cold[index].ptr,
                          data.cold[index].bytes,
                          data.cold[index].host_allocation,
                          slot.cold[index].allocation});
    }
    if (mode == SubmitMode::CoalescedBatch)
        result = merge_copy_ops(std::move(result));
    return result;
}

std::vector<ManagedOp> managed_ops(const ManagedDataset &data,
                                   SubmitMode mode)
{
    std::vector<ManagedOp> result;
    for (const LogicalRange &range : data.cold)
        result.push_back(
            {range.managed, range.bytes, range.managed_allocation});
    if (mode == SubmitMode::CoalescedBatch)
        result = merge_managed_ops(std::move(result));
    return result;
}

void reset_copy_destination(const PinnedDataset &data, int slot_index,
                            cudaStream_t stream)
{
    for (const DeviceLogicalRange &range : data.slots[slot_index].cold)
        check(cudaMemsetAsync(range.ptr, 0, range.bytes, stream),
              "reset copy destination");
    check(cudaStreamSynchronize(stream), "reset copy synchronize");
}

void reset_managed_to_cpu(const ManagedDataset &data, cudaStream_t stream)
{
    std::vector<ManagedOp> ops =
        merge_managed_ops(managed_ops(data, SubmitMode::TensorBatch));
    for (const ManagedOp &op : ops)
        check(cudaMemPrefetchAsync(op.ptr, op.bytes, cudaCpuDeviceId, stream),
              "reset managed to CPU");
    check(cudaStreamSynchronize(stream), "reset managed synchronize");
}

bool managed_placement_ok(const ManagedDataset &data, int device)
{
    for (const LogicalRange &range : data.cold) {
        int location = std::numeric_limits<int>::min();
        const cudaError_t status = cudaMemRangeGetAttribute(
            &location, sizeof(location),
            cudaMemRangeAttributeLastPrefetchLocation,
            range.managed, range.bytes);
        if (status != cudaSuccess || location != device)
            return false;
    }
    return true;
}

uintptr_t fingerprint(const std::vector<DeviceLogicalRange> &ranges)
{
    uintptr_t value = 1469598103934665603ULL;
    for (const DeviceLogicalRange &range : ranges) {
        value ^= reinterpret_cast<uintptr_t>(range.ptr);
        value *= 1099511628211ULL;
    }
    return value;
}

uintptr_t managed_fingerprint(const ManagedDataset &data)
{
    uintptr_t value = 1469598103934665603ULL;
    for (const LogicalRange &range : data.cold) {
        value ^= reinterpret_cast<uintptr_t>(range.managed);
        value *= 1099511628211ULL;
    }
    return value;
}

uintptr_t uva_fingerprint(const PinnedDataset &data)
{
    uintptr_t value = 1469598103934665603ULL;
    for (const LogicalRange &range : data.cold) {
        value ^= reinterpret_cast<uintptr_t>(range.mapped);
        value *= 1099511628211ULL;
    }
    return value;
}

float event_ms(cudaEvent_t begin, cudaEvent_t end)
{
    float result = 0.0F;
    check(cudaEventElapsedTime(&result, begin, end),
          "cudaEventElapsedTime");
    return result;
}

Sample run_sample(const Args &args, const CaseDef &definition,
                  PinnedDataset &pinned, ManagedDataset &managed,
                  int sample_index, cudaStream_t copy_stream,
                  cudaStream_t consume_stream, unsigned long long *checksum_device)
{
    const int slot_index = pinned.slots.empty()
                               ? 0
                               : sample_index %
                                     static_cast<int>(pinned.slots.size());
    if (definition.backend == Backend::PinnedCopy)
        reset_copy_destination(pinned, slot_index, copy_stream);
    else if (definition.backend == Backend::ManagedPrefetch)
        reset_managed_to_cpu(managed, copy_stream);

    check(cudaMemsetAsync(checksum_device, 0, sizeof(*checksum_device),
                          consume_stream),
          "reset checksum");
    check(cudaStreamSynchronize(consume_stream), "checksum reset synchronize");

    cudaEvent_t epoch = nullptr;
    cudaEvent_t ready = nullptr;
    cudaEvent_t wait_start = nullptr;
    cudaEvent_t wait_end = nullptr;
    cudaEvent_t read_end = nullptr;
    check(cudaEventCreate(&epoch), "create epoch event");
    check(cudaEventCreate(&ready), "create ready event");
    check(cudaEventCreate(&wait_start), "create wait start event");
    check(cudaEventCreate(&wait_end), "create wait end event");
    check(cudaEventCreate(&read_end), "create read end event");

    check(cudaEventRecord(epoch, consume_stream), "record epoch");
    check(cudaStreamWaitEvent(copy_stream, epoch, 0), "copy waits epoch");
    check(cudaEventRecord(wait_start, consume_stream), "record wait start");

    Sample sample;
    DeviceRange *descriptors = nullptr;
    uint64_t expected = 0;
    const auto host_begin = std::chrono::steady_clock::now();
    const auto enqueue_begin = host_begin;
    if (definition.backend == Backend::PinnedCopy) {
        std::vector<CopyOp> ops =
            copy_ops(pinned, slot_index, definition.mode);
        sample.submitted_ranges = static_cast<int>(ops.size());
        for (const CopyOp &op : ops) {
            const auto api_begin = std::chrono::steady_clock::now();
            check(cudaMemcpyAsync(op.destination, op.source, op.bytes,
                                  cudaMemcpyHostToDevice, copy_stream),
                  "pinned cudaMemcpyAsync");
            const auto api_end = std::chrono::steady_clock::now();
            sample.submit_us +=
                std::chrono::duration<double, std::micro>(api_end - api_begin)
                    .count();
            if (definition.mode == SubmitMode::PerTensorSync)
                check(cudaStreamSynchronize(copy_stream),
                      "per tensor copy synchronize");
        }
        descriptors = pinned.slots[slot_index].device_descriptors;
        expected = pinned.expected_checksum;
        sample.consumer_fingerprint = fingerprint(pinned.slots[slot_index].cold);
    }
    else if (definition.backend == Backend::ManagedPrefetch) {
        std::vector<ManagedOp> ops = managed_ops(managed, definition.mode);
        sample.submitted_ranges = static_cast<int>(ops.size());
        for (const ManagedOp &op : ops) {
            const auto api_begin = std::chrono::steady_clock::now();
            check(cudaMemPrefetchAsync(op.ptr, op.bytes, args.device,
                                       copy_stream),
                  "cudaMemPrefetchAsync to GPU");
            const auto api_end = std::chrono::steady_clock::now();
            sample.submit_us +=
                std::chrono::duration<double, std::micro>(api_end - api_begin)
                    .count();
            if (definition.mode == SubmitMode::PerTensorSync)
                check(cudaStreamSynchronize(copy_stream),
                      "per tensor prefetch synchronize");
        }
        descriptors = managed.device_descriptors;
        expected = managed.expected_checksum;
        sample.consumer_fingerprint = managed_fingerprint(managed);
    }
    else {
        sample.submitted_ranges = 0;
        descriptors = pinned.uva_descriptors;
        expected = pinned.expected_checksum;
        sample.consumer_fingerprint = uva_fingerprint(pinned);
    }
    const auto enqueue_end = std::chrono::steady_clock::now();
    sample.enqueue_span_us =
        std::chrono::duration<double, std::micro>(enqueue_end - enqueue_begin)
            .count();

    check(cudaEventRecord(ready, copy_stream), "record ready");
    check(cudaStreamWaitEvent(consume_stream, ready, 0), "consume waits ready");
    check(cudaEventRecord(wait_end, consume_stream), "record wait end");
    const int blocks_per_range =
        std::max(1, std::min(64, 256 / std::max(1, args.tensors)));
    checksum_ranges_kernel<<<args.tensors * blocks_per_range, 256, 0,
                             consume_stream>>>(
        descriptors, args.tensors, blocks_per_range, checksum_device);
    check(cudaGetLastError(), "launch checksum_ranges_kernel");
    check(cudaEventRecord(read_end, consume_stream), "record read end");
    check(cudaEventSynchronize(read_end), "synchronize read end");
    const auto host_end = std::chrono::steady_clock::now();

    sample.migration_ms = event_ms(epoch, ready);
    sample.exposed_wait_ms = event_ms(wait_start, wait_end);
    sample.read_ms = event_ms(wait_end, read_end);
    sample.e2e_cuda_ms = event_ms(epoch, read_end);
    sample.host_block_ms =
        std::chrono::duration<double, std::milli>(host_end - host_begin).count();
    check(cudaMemcpy(&sample.checksum, checksum_device, sizeof(sample.checksum),
                     cudaMemcpyDeviceToHost),
          "copy checksum to host");
    sample.checksum_ok = sample.checksum == expected;
    if (definition.backend == Backend::ManagedPrefetch)
        sample.placement_ok = managed_placement_ok(managed, args.device);

    cudaEventDestroy(epoch);
    cudaEventDestroy(ready);
    cudaEventDestroy(wait_start);
    cudaEventDestroy(wait_end);
    cudaEventDestroy(read_end);
    return sample;
}

double percentile(std::vector<double> values, double fraction)
{
    if (values.empty())
        return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    if (lower == upper)
        return values[lower];
    return values[lower] * (static_cast<double>(upper) - position) +
           values[upper] * (position - static_cast<double>(lower));
}

const char *backend_name(Backend backend)
{
    switch (backend) {
        case Backend::PinnedCopy:
            return "pinned_copy";
        case Backend::ManagedPrefetch:
            return "managed_prefetch";
        case Backend::UvaDirect:
            return "uva_direct";
    }
    return "unknown";
}

const char *mode_name(SubmitMode mode)
{
    switch (mode) {
        case SubmitMode::PerTensorSync:
            return "per_tensor_sync";
        case SubmitMode::TensorBatch:
            return "tensor_batch";
        case SubmitMode::CoalescedBatch:
            return "coalesced_batch";
        case SubmitMode::Direct:
            return "direct";
    }
    return "unknown";
}

CaseResult summarize_case(const std::string &layout, const CaseDef &definition,
                          size_t bytes, int logical_ranges,
                          std::vector<Sample> samples)
{
    CaseResult result;
    result.layout = layout;
    result.backend = definition.backend;
    result.mode = definition.mode;
    result.bytes = bytes;
    result.logical_ranges = logical_ranges;
    result.samples = std::move(samples);
    result.submitted_ranges = result.samples.front().submitted_ranges;
    std::vector<double> submit;
    std::vector<double> api_call;
    std::vector<double> migration;
    std::vector<double> wait;
    std::vector<double> read;
    std::vector<double> e2e;
    std::vector<double> host_block;
    for (const Sample &sample : result.samples) {
        submit.push_back(sample.enqueue_span_us);
        api_call.push_back(sample.submit_us);
        migration.push_back(sample.migration_ms);
        wait.push_back(sample.exposed_wait_ms);
        read.push_back(sample.read_ms);
        e2e.push_back(sample.e2e_cuda_ms);
        host_block.push_back(sample.host_block_ms);
    }
    result.functional_ok = std::all_of(
        result.samples.begin(), result.samples.end(),
        [](const Sample &sample) {
            return sample.checksum_ok && sample.placement_ok;
        });
    const uintptr_t first_fingerprint =
        result.samples.front().consumer_fingerprint;
    result.same_consumer_va = std::all_of(
        result.samples.begin(), result.samples.end(),
        [first_fingerprint](const Sample &sample) {
            return sample.consumer_fingerprint == first_fingerprint;
        });
    result.in_place_va = definition.backend != Backend::PinnedCopy;
    result.submit_us = percentile(submit, 0.5);
    result.api_call_us = percentile(api_call, 0.5);
    result.migration_ms = percentile(migration, 0.5);
    result.exposed_wait_ms = percentile(wait, 0.5);
    result.read_ms = percentile(read, 0.5);
    result.e2e_ms = percentile(e2e, 0.5);
    result.host_block_ms = percentile(host_block, 0.5);
    if (definition.backend != Backend::UvaDirect && result.migration_ms > 0.0)
        result.migration_gib_s =
            (static_cast<double>(bytes) / GiB) /
            (result.migration_ms / 1000.0);
    result.effective_gib_s =
        (static_cast<double>(bytes) / GiB) / (result.e2e_ms / 1000.0);
    return result;
}

void print_case(const CaseResult &result)
{
    std::vector<double> e2e;
    for (const Sample &sample : result.samples)
        e2e.push_back(sample.e2e_cuda_ms);
    std::printf(
        "RESULT layout=%s backend=%s mode=%s bytes=%zu logical_ranges=%d "
        "submitted_ranges=%d submit_us_p50=%.3f api_call_us_p50=%.3f "
        "migration_ms_p50=%.3f "
        "wait_ms_p50=%.3f read_ms_p50=%.3f e2e_ms_p50=%.3f "
        "e2e_ms_p95=%.3f host_block_ms_p50=%.3f "
        "migration_gib_s=%.3f effective_gib_s=%.3f "
        "in_place_va=%d same_consumer_va=%d checksum=%llu "
        "functional_ok=%d\n",
        result.layout.c_str(), backend_name(result.backend),
        mode_name(result.mode), result.bytes, result.logical_ranges,
        result.submitted_ranges, result.submit_us, result.api_call_us,
        result.migration_ms,
        result.exposed_wait_ms, result.read_ms, result.e2e_ms,
        percentile(e2e, 0.95), result.host_block_ms,
        result.migration_gib_s,
        result.effective_gib_s, result.in_place_va ? 1 : 0,
        result.same_consumer_va ? 1 : 0,
        static_cast<unsigned long long>(result.samples.front().checksum),
        result.functional_ok ? 1 : 0);
}

const CaseResult *find_case(const std::vector<CaseResult> &results,
                            const std::string &layout, Backend backend,
                            SubmitMode mode)
{
    for (const CaseResult &result : results) {
        if (result.layout == layout && result.backend == backend &&
            result.mode == mode)
            return &result;
    }
    return nullptr;
}

double safe_ratio(double numerator, double denominator)
{
    return denominator > 0.0 ? numerator / denominator
                             : std::numeric_limits<double>::quiet_NaN();
}

void run_layout(const Args &args, const std::string &layout,
                cudaStream_t copy_stream, cudaStream_t consume_stream,
                unsigned long long *checksum_device,
                std::vector<CaseResult> &results)
{
    std::vector<CaseDef> definitions = {
        {Backend::PinnedCopy, SubmitMode::PerTensorSync},
        {Backend::PinnedCopy, SubmitMode::TensorBatch},
        {Backend::ManagedPrefetch, SubmitMode::PerTensorSync},
        {Backend::ManagedPrefetch, SubmitMode::TensorBatch},
        {Backend::UvaDirect, SubmitMode::Direct},
    };
    if (layout == "arena_contiguous") {
        definitions.insert(definitions.begin() + 2,
                           {Backend::PinnedCopy,
                            SubmitMode::CoalescedBatch});
        definitions.insert(definitions.begin() + 5,
                           {Backend::ManagedPrefetch,
                            SubmitMode::CoalescedBatch});
    }
    definitions.erase(
        std::remove_if(
            definitions.begin(), definitions.end(),
            [&args](const CaseDef &definition) {
                return (!args.only_backend.empty() &&
                        args.only_backend != backend_name(definition.backend)) ||
                       (!args.only_mode.empty() &&
                        args.only_mode != mode_name(definition.mode));
            }),
        definitions.end());
    if (definitions.empty())
        return;

    const bool needs_pinned = std::any_of(
        definitions.begin(), definitions.end(), [](const CaseDef &definition) {
            return definition.backend == Backend::PinnedCopy ||
                   definition.backend == Backend::UvaDirect;
        });
    const bool needs_managed = std::any_of(
        definitions.begin(), definitions.end(), [](const CaseDef &definition) {
            return definition.backend == Backend::ManagedPrefetch;
        });
    PinnedDataset pinned;
    ManagedDataset managed;
    if (needs_pinned)
        pinned = make_pinned_dataset(layout, args);
    if (needs_managed)
        managed = make_managed_dataset(layout, args, copy_stream);

    std::vector<std::vector<Sample>> collected(definitions.size());
    for (int warmup = 0; warmup < args.warmups; ++warmup) {
        for (size_t index = 0; index < definitions.size(); ++index) {
            (void)run_sample(args, definitions[index], pinned, managed,
                             warmup * static_cast<int>(definitions.size()) +
                                 static_cast<int>(index),
                             copy_stream, consume_stream, checksum_device);
        }
    }
    for (int repeat = 0; repeat < args.repeats; ++repeat) {
        for (size_t order = 0; order < definitions.size(); ++order) {
            const size_t index = repeat % 2 == 0
                                     ? order
                                     : definitions.size() - 1 - order;
            collected[index].push_back(run_sample(
                args, definitions[index], pinned, managed,
                args.warmups * static_cast<int>(definitions.size()) +
                    repeat,
                copy_stream, consume_stream, checksum_device));
        }
    }

    bool managed_addresses_stable = needs_managed;
    if (needs_managed) {
        managed_addresses_stable =
            managed.original_addresses.size() == managed.allocations.size();
        for (size_t index = 0; index < managed.allocations.size(); ++index) {
            managed_addresses_stable &=
                managed.original_addresses[index] ==
                reinterpret_cast<uintptr_t>(managed.allocations[index]);
        }
    }
    std::printf(
        "LAYOUT name=%s cpu_va_contiguous=%d cpu_physical_contiguity="
        "not_guaranteed managed_allocation_count=%zu managed_va_stable=%d "
        "mapped_host_device_address_equal=%d\n",
        layout.c_str(), layout == "multi_alloc" ? 0 : 1,
        managed.allocations.size(), needs_managed
                                        ? (managed_addresses_stable ? 1 : 0)
                                        : -1,
        needs_pinned ? (pinned.mapped_address_equal ? 1 : 0) : -1);

    for (size_t index = 0; index < definitions.size(); ++index) {
        CaseResult result = summarize_case(
            layout, definitions[index], args.cold_bytes, args.tensors,
            std::move(collected[index]));
        print_case(result);
        results.push_back(std::move(result));
    }
    if (needs_managed)
        destroy_managed_dataset(managed);
    if (needs_pinned)
        destroy_pinned_dataset(pinned);
}

}  // namespace

int main(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const Args args = parse_args(argc, argv);
    check(cudaSetDevice(args.device), "cudaSetDevice");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, args.device),
          "cudaGetDeviceProperties");
    if (!properties.managedMemory)
        fail("selected GPU does not support managed memory");
    if (!properties.canMapHostMemory)
        fail("selected GPU does not support mapped host memory");

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    std::printf(
        "benchmark_begin device=%d gpu=%s cold_bytes=%zu tensors=%d slots=%d "
        "warmups=%d repeats=%d equivalent_ratio=%.3f managed_memory=%d "
        "concurrent_managed_access=%d pageable_memory_access=%d "
        "free_bytes=%zu total_bytes=%zu only_layout=%s only_backend=%s "
        "only_mode=%s\n",
        args.device, properties.name, args.cold_bytes, args.tensors,
        args.slots, args.warmups, args.repeats, args.equivalent_ratio,
        properties.managedMemory, properties.concurrentManagedAccess,
        properties.pageableMemoryAccess, free_bytes, total_bytes,
        args.only_layout.empty() ? "all" : args.only_layout.c_str(),
        args.only_backend.empty() ? "all" : args.only_backend.c_str(),
        args.only_mode.empty() ? "all" : args.only_mode.c_str());
    std::printf(
        "SEMANTICS pinned_copy=pinned_per_tensor_backing+static_round_robin_"
        "gpu_slots+single_copy_stream+one_ready_event "
        "tensor_batch=N_async_calls+one_event_wait "
        "coalesced_batch=adjacent_ranges_merged "
        "uva_direct=mapped_pinned_host_gpu_read\n");

    cudaStream_t copy_stream = nullptr;
    cudaStream_t consume_stream = nullptr;
    check(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking),
          "create copy stream");
    check(cudaStreamCreateWithFlags(&consume_stream, cudaStreamNonBlocking),
          "create consume stream");
    unsigned long long *checksum_device = nullptr;
    check(cudaMalloc(&checksum_device, sizeof(*checksum_device)),
          "cudaMalloc checksum");

    std::vector<CaseResult> results;
    const std::vector<std::string> layouts = {
        "multi_alloc", "arena_interleaved", "arena_contiguous"};
    for (const std::string &layout : layouts) {
        if (args.only_layout.empty() || args.only_layout == layout)
            run_layout(args, layout, copy_stream, consume_stream,
                       checksum_device, results);
    }
    if (results.empty())
        fail("filters selected no benchmark cases");

    const bool filtered = !args.only_layout.empty() ||
                          !args.only_backend.empty() ||
                          !args.only_mode.empty();
    const bool functional_success = std::all_of(
        results.begin(), results.end(),
        [](const CaseResult &result) { return result.functional_ok; });
    if (filtered) {
        std::printf(
            "benchmark_result functional_success=%d filtered=1 "
            "cpu_physical_contiguity_controlled=0 "
            "cpu_virtual_layout_controlled=1 total_cases=%zu\n",
            functional_success ? 1 : 0, results.size());
        cudaFree(checksum_device);
        cudaStreamDestroy(copy_stream);
        cudaStreamDestroy(consume_stream);
        return functional_success ? 0 : 1;
    }

    const CaseResult *copy_reference = find_case(
        results, "multi_alloc", Backend::PinnedCopy,
        SubmitMode::TensorBatch);
    const CaseResult *uvm_multi = find_case(
        results, "multi_alloc", Backend::ManagedPrefetch,
        SubmitMode::TensorBatch);
    const CaseResult *uvm_interleaved = find_case(
        results, "arena_interleaved", Backend::ManagedPrefetch,
        SubmitMode::TensorBatch);
    const CaseResult *uvm_contiguous = find_case(
        results, "arena_contiguous", Backend::ManagedPrefetch,
        SubmitMode::CoalescedBatch);
    const CaseResult *uvm_contiguous_tensor = find_case(
        results, "arena_contiguous", Backend::ManagedPrefetch,
        SubmitMode::TensorBatch);
    const CaseResult *uvm_contiguous_sync = find_case(
        results, "arena_contiguous", Backend::ManagedPrefetch,
        SubmitMode::PerTensorSync);
    const CaseResult *copy_interleaved = find_case(
        results, "arena_interleaved", Backend::PinnedCopy,
        SubmitMode::TensorBatch);
    const CaseResult *copy_contiguous = find_case(
        results, "arena_contiguous", Backend::PinnedCopy,
        SubmitMode::CoalescedBatch);

    if (!copy_reference || !uvm_multi || !uvm_interleaved ||
        !uvm_contiguous || !uvm_contiguous_tensor ||
        !uvm_contiguous_sync || !copy_interleaved || !copy_contiguous)
        fail("internal result matrix is incomplete");

    const double uvm_multi_vs_vllm = safe_ratio(
        uvm_multi->migration_gib_s, copy_reference->migration_gib_s);
    const double uvm_best_vs_vllm = safe_ratio(
        uvm_contiguous->migration_gib_s, copy_reference->migration_gib_s);
    const double uvm_batch_speedup = safe_ratio(
        uvm_contiguous_sync->submit_us,
        uvm_contiguous_tensor->submit_us);
    const double uvm_batch_e2e_speedup = safe_ratio(
        uvm_contiguous_sync->e2e_ms,
        uvm_contiguous_tensor->e2e_ms);
    const double uvm_coalesce_speedup = safe_ratio(
        uvm_contiguous_tensor->migration_ms,
        uvm_contiguous->migration_ms);
    const double uvm_interleave_fraction = safe_ratio(
        uvm_interleaved->migration_gib_s,
        uvm_contiguous->migration_gib_s);
    const double uvm_gap_fraction = safe_ratio(
        uvm_interleaved->migration_gib_s,
        uvm_contiguous_tensor->migration_gib_s);
    const double uvm_allocation_fraction = safe_ratio(
        uvm_multi->migration_gib_s,
        uvm_interleaved->migration_gib_s);
    const double copy_interleave_fraction = safe_ratio(
        copy_interleaved->migration_gib_s,
        copy_contiguous->migration_gib_s);
    const double uvm_specific_fragmentation = safe_ratio(
        uvm_interleave_fraction, copy_interleave_fraction);

    const bool grouped_equivalent =
        uvm_multi_vs_vllm >= args.equivalent_ratio;
    const bool best_equivalent =
        uvm_best_vs_vllm >= args.equivalent_ratio;
    const char *diagnosis = "no_large_uvm_fragmentation_signal";
    if (!best_equivalent)
        diagnosis = "packed_uvm_still_slow_not_explained_by_fragmentation";
    else if (uvm_coalesce_speedup > 1.20 &&
             uvm_gap_fraction >= 0.85 &&
             uvm_allocation_fraction >= 0.85)
        diagnosis = "per_range_uvm_driver_orchestration_likely";
    else if (uvm_gap_fraction < 0.80)
        diagnosis = "hot_cold_va_gap_fragmentation_likely";
    else if (uvm_allocation_fraction < 0.80)
        diagnosis = "multi_allocation_metadata_fragmentation_likely";
    else if (uvm_interleave_fraction < 0.80 &&
             copy_interleave_fraction < 0.80)
        diagnosis = "shared_per_tensor_submission_fragmentation_likely";

    std::printf(
        "DIAGNOSIS uvm_multi_vs_vllm_ratio=%.3f "
        "uvm_best_vs_vllm_ratio=%.3f uvm_batch_submit_speedup=%.3f "
        "uvm_batch_e2e_speedup=%.3f "
        "uvm_coalesce_speedup=%.3f uvm_interleave_fraction=%.3f "
        "uvm_gap_fraction=%.3f uvm_allocation_fraction=%.3f "
        "copy_interleave_fraction=%.3f "
        "uvm_specific_fragmentation=%.3f conclusion=%s\n",
        uvm_multi_vs_vllm, uvm_best_vs_vllm, uvm_batch_speedup,
        uvm_batch_e2e_speedup, uvm_coalesce_speedup,
        uvm_interleave_fraction, uvm_gap_fraction,
        uvm_allocation_fraction,
        copy_interleave_fraction, uvm_specific_fragmentation, diagnosis);
    std::printf(
        "benchmark_result functional_success=%d "
        "uvm_grouped_equivalent=%d uvm_best_equivalent=%d "
        "cpu_physical_contiguity_controlled=0 "
        "cpu_virtual_layout_controlled=1 total_cases=%zu\n",
        functional_success ? 1 : 0, grouped_equivalent ? 1 : 0,
        best_equivalent ? 1 : 0, results.size());

    cudaFree(checksum_device);
    cudaStreamDestroy(copy_stream);
    cudaStreamDestroy(consume_stream);
    return functional_success ? 0 : 1;
}
