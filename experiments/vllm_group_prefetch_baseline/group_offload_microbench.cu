// SPDX-License-Identifier: Apache-2.0
// Standalone reproduction of vLLM's grouped prefetch scheduling topology.

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
#include <vector>

namespace {

constexpr size_t MiB = 1024ULL * 1024ULL;
constexpr double GiB = 1024.0 * 1024.0 * 1024.0;

struct Args {
    int device = 0;
    int total_layers = 16;
    int group_size = 8;
    int num_in_group = 2;
    int prefetch_step = 2;
    size_t layer_bytes = 16 * MiB;
    int tensors = 4;
    double compute_us = 200.0;
    int warmups = 1;
    int repeats = 5;
    double equivalent_ratio = 0.90;
    std::string only_backend;
};

enum class Backend { Pinned, Managed, Uva };

struct DeviceRange {
    const uint32_t *ptr;
    uint64_t words;
};

struct TensorStorage {
    uint8_t *host = nullptr;
    uint8_t *mapped = nullptr;
    uint8_t *managed = nullptr;
    size_t bytes = 0;
    uint32_t value = 0;
};

struct Dataset {
    size_t tensor_bytes = 0;
    std::vector<int> selected_layers;
    std::vector<std::vector<TensorStorage>> modules;
    std::vector<std::vector<uint8_t *>> slots;
    std::vector<DeviceRange *> slot_descriptors;
    std::vector<DeviceRange *> managed_descriptors;
    std::vector<DeviceRange *> uva_descriptors;
    std::vector<void *> pinned_allocations;
    std::vector<void *> managed_allocations;
    std::vector<void *> device_allocations;
    std::vector<void *> descriptor_allocations;
    uint64_t expected_checksum = 0;
    bool mapped_addresses_equal = true;
};

struct Sample {
    double host_submit_ms = 0.0;
    double host_block_ms = 0.0;
    double transfer_work_ms = 0.0;
    double transfer_window_ms = 0.0;
    double exposed_wait_ms = 0.0;
    double read_ms = 0.0;
    double e2e_ms = 0.0;
    uint64_t checksum = 0;
    uintptr_t consumer_fingerprint = 0;
    int submitted_ranges = 0;
    bool checksum_ok = false;
    bool placement_ok = true;
};

struct Result {
    Backend backend = Backend::Pinned;
    std::vector<Sample> samples;
    bool functional_ok = false;
    bool consumer_va_stable = false;
    double host_submit_ms = 0.0;
    double host_block_ms = 0.0;
    double transfer_work_ms = 0.0;
    double transfer_window_ms = 0.0;
    double exposed_wait_ms = 0.0;
    double read_ms = 0.0;
    double e2e_ms = 0.0;
    double transfer_gib_s = 0.0;
    double read_gib_s = 0.0;
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

long parse_long(const char *text, const char *name)
{
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') {
        std::fprintf(stderr, "invalid %s: %s\n", name, text);
        std::exit(2);
    }
    return value;
}

double parse_double(const char *text, const char *name)
{
    char *end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(value)) {
        std::fprintf(stderr, "invalid %s: %s\n", name, text);
        std::exit(2);
    }
    return value;
}

Args parse_args(int argc, char **argv)
{
    Args args;
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--device") == 0 && index + 1 < argc)
            args.device = static_cast<int>(parse_long(argv[++index], "device"));
        else if (std::strcmp(argv[index], "--total-layers") == 0 &&
                 index + 1 < argc)
            args.total_layers = static_cast<int>(
                parse_long(argv[++index], "total-layers"));
        else if (std::strcmp(argv[index], "--group-size") == 0 &&
                 index + 1 < argc)
            args.group_size = static_cast<int>(
                parse_long(argv[++index], "group-size"));
        else if (std::strcmp(argv[index], "--num-in-group") == 0 &&
                 index + 1 < argc)
            args.num_in_group = static_cast<int>(
                parse_long(argv[++index], "num-in-group"));
        else if (std::strcmp(argv[index], "--prefetch-step") == 0 &&
                 index + 1 < argc)
            args.prefetch_step = static_cast<int>(
                parse_long(argv[++index], "prefetch-step"));
        else if (std::strcmp(argv[index], "--layer-mib") == 0 &&
                 index + 1 < argc)
            args.layer_bytes = static_cast<size_t>(
                parse_double(argv[++index], "layer-mib") * MiB);
        else if (std::strcmp(argv[index], "--tensors") == 0 &&
                 index + 1 < argc)
            args.tensors = static_cast<int>(
                parse_long(argv[++index], "tensors"));
        else if (std::strcmp(argv[index], "--compute-us") == 0 &&
                 index + 1 < argc)
            args.compute_us = parse_double(argv[++index], "compute-us");
        else if (std::strcmp(argv[index], "--warmups") == 0 &&
                 index + 1 < argc)
            args.warmups = static_cast<int>(
                parse_long(argv[++index], "warmups"));
        else if (std::strcmp(argv[index], "--repeats") == 0 &&
                 index + 1 < argc)
            args.repeats = static_cast<int>(
                parse_long(argv[++index], "repeats"));
        else if (std::strcmp(argv[index], "--equivalent-ratio") == 0 &&
                 index + 1 < argc)
            args.equivalent_ratio =
                parse_double(argv[++index], "equivalent-ratio");
        else if (std::strcmp(argv[index], "--only-backend") == 0 &&
                 index + 1 < argc)
            args.only_backend = argv[++index];
        else {
            std::fprintf(
                stderr,
                "usage: %s [--device N] [--total-layers N] "
                "[--group-size N] [--num-in-group N] "
                "[--prefetch-step N] [--layer-mib N] [--tensors N] "
                "[--compute-us N] [--warmups N] [--repeats N] "
                "[--equivalent-ratio R] "
                "[--only-backend pinned|managed|uva]\n",
                argv[0]);
            std::exit(2);
        }
    }

    if (args.device < 0 || args.total_layers < 1 || args.total_layers > 4096 ||
        args.group_size < 1 || args.num_in_group < 1 ||
        args.num_in_group > args.group_size || args.prefetch_step < 1 ||
        args.tensors < 1 || args.tensors > 256 || args.layer_bytes == 0 ||
        args.layer_bytes % args.tensors != 0 ||
        (args.layer_bytes / args.tensors) % sizeof(uint32_t) != 0 ||
        args.compute_us < 0.0 || args.compute_us > 100000.0 ||
        args.warmups < 0 || args.warmups > 20 || args.repeats < 1 ||
        args.repeats > 100 || args.equivalent_ratio <= 0.0 ||
        args.equivalent_ratio > 2.0 ||
        (!args.only_backend.empty() && args.only_backend != "pinned" &&
         args.only_backend != "managed" && args.only_backend != "uva")) {
        fail("invalid benchmark arguments");
    }
    return args;
}

std::vector<int> select_offloaded_layers(const Args &args)
{
    std::vector<int> selected;
    const int first_in_group = args.group_size - args.num_in_group;
    for (int layer = 0; layer < args.total_layers; ++layer) {
        if (layer % args.group_size >= first_in_group)
            selected.push_back(layer);
    }
    return selected;
}

const char *backend_name(Backend backend)
{
    switch (backend) {
        case Backend::Pinned:
            return "pinned";
        case Backend::Managed:
            return "managed";
        case Backend::Uva:
            return "uva";
    }
    return "unknown";
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

__global__ void delay_kernel(unsigned long long cycles)
{
    const unsigned long long begin = clock64();
    while (clock64() - begin < cycles) {
    }
}

DeviceRange *make_descriptors(const std::vector<uint8_t *> &pointers,
                              size_t tensor_bytes,
                              std::vector<void *> &owned)
{
    std::vector<DeviceRange> host;
    host.reserve(pointers.size());
    for (uint8_t *pointer : pointers) {
        host.push_back({reinterpret_cast<const uint32_t *>(pointer),
                        static_cast<uint64_t>(tensor_bytes / sizeof(uint32_t))});
    }
    DeviceRange *device = nullptr;
    check(cudaMalloc(&device, host.size() * sizeof(DeviceRange)),
          "cudaMalloc descriptors");
    check(cudaMemcpy(device, host.data(), host.size() * sizeof(DeviceRange),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy descriptors");
    owned.push_back(device);
    return device;
}

uint64_t tensor_checksum(uint32_t value, size_t bytes)
{
    return static_cast<uint64_t>(value) *
           static_cast<uint64_t>(bytes / sizeof(uint32_t));
}

Dataset make_dataset(const Args &args, const std::vector<int> &selected)
{
    Dataset data;
    data.tensor_bytes = args.layer_bytes / args.tensors;
    data.selected_layers = selected;
    data.modules.resize(selected.size());
    data.managed_descriptors.resize(selected.size());
    data.uva_descriptors.resize(selected.size());

    for (size_t position = 0; position < selected.size(); ++position) {
        std::vector<uint8_t *> managed_pointers;
        std::vector<uint8_t *> mapped_pointers;
        for (int tensor = 0; tensor < args.tensors; ++tensor) {
            TensorStorage storage;
            storage.bytes = data.tensor_bytes;
            storage.value = static_cast<uint32_t>(
                1 + selected[position] * 257 + tensor * 17);

            void *host = nullptr;
            check(cudaHostAlloc(&host, storage.bytes, cudaHostAllocMapped),
                  "cudaHostAllocMapped");
            void *mapped = nullptr;
            check(cudaHostGetDevicePointer(&mapped, host, 0),
                  "cudaHostGetDevicePointer");
            storage.host = static_cast<uint8_t *>(host);
            storage.mapped = static_cast<uint8_t *>(mapped);
            data.pinned_allocations.push_back(host);
            data.mapped_addresses_equal &= host == mapped;

            void *managed = nullptr;
            check(cudaMallocManaged(&managed, storage.bytes,
                                    cudaMemAttachGlobal),
                  "cudaMallocManaged tensor");
            storage.managed = static_cast<uint8_t *>(managed);
            data.managed_allocations.push_back(managed);

            uint32_t *host_words = reinterpret_cast<uint32_t *>(storage.host);
            uint32_t *managed_words =
                reinterpret_cast<uint32_t *>(storage.managed);
            const size_t words = storage.bytes / sizeof(uint32_t);
            std::fill(host_words, host_words + words, storage.value);
            std::fill(managed_words, managed_words + words, storage.value);
            data.expected_checksum +=
                tensor_checksum(storage.value, storage.bytes);
            managed_pointers.push_back(storage.managed);
            mapped_pointers.push_back(storage.mapped);
            data.modules[position].push_back(storage);
        }
        data.managed_descriptors[position] = make_descriptors(
            managed_pointers, data.tensor_bytes, data.descriptor_allocations);
        data.uva_descriptors[position] = make_descriptors(
            mapped_pointers, data.tensor_bytes, data.descriptor_allocations);
    }

    data.slots.resize(args.prefetch_step);
    data.slot_descriptors.resize(args.prefetch_step);
    for (int slot = 0; slot < args.prefetch_step; ++slot) {
        std::vector<uint8_t *> pointers;
        for (int tensor = 0; tensor < args.tensors; ++tensor) {
            void *device = nullptr;
            check(cudaMalloc(&device, data.tensor_bytes),
                  "cudaMalloc static slot tensor");
            data.device_allocations.push_back(device);
            data.slots[slot].push_back(static_cast<uint8_t *>(device));
            pointers.push_back(static_cast<uint8_t *>(device));
        }
        data.slot_descriptors[slot] = make_descriptors(
            pointers, data.tensor_bytes, data.descriptor_allocations);
    }
    check(cudaDeviceSynchronize(), "dataset setup synchronize");
    return data;
}

void destroy_dataset(Dataset &data)
{
    for (void *pointer : data.descriptor_allocations)
        cudaFree(pointer);
    for (void *pointer : data.device_allocations)
        cudaFree(pointer);
    for (void *pointer : data.managed_allocations)
        cudaFree(pointer);
    for (void *pointer : data.pinned_allocations)
        cudaFreeHost(pointer);
}

float event_ms(cudaEvent_t begin, cudaEvent_t end)
{
    float value = 0.0F;
    check(cudaEventElapsedTime(&value, begin, end),
          "cudaEventElapsedTime");
    return value;
}

void create_events(std::vector<cudaEvent_t> &events, size_t count,
                   const char *operation)
{
    events.resize(count);
    for (cudaEvent_t &event : events)
        check(cudaEventCreate(&event), operation);
}

void destroy_events(std::vector<cudaEvent_t> &events)
{
    for (cudaEvent_t event : events)
        cudaEventDestroy(event);
}

uintptr_t consumer_fingerprint(Backend backend, const Dataset &data,
                               const Args &args)
{
    uintptr_t fingerprint = static_cast<uintptr_t>(1469598103934665603ULL);
    for (size_t position = 0; position < data.modules.size(); ++position) {
        for (int tensor = 0; tensor < args.tensors; ++tensor) {
            const void *pointer = nullptr;
            if (backend == Backend::Pinned) {
                pointer = data.slots[position % args.prefetch_step][tensor];
            }
            else if (backend == Backend::Managed) {
                pointer = data.modules[position][tensor].managed;
            }
            else {
                pointer = data.modules[position][tensor].mapped;
            }
            fingerprint ^= reinterpret_cast<uintptr_t>(pointer);
            fingerprint *= static_cast<uintptr_t>(1099511628211ULL);
        }
    }
    return fingerprint;
}

bool managed_placement_ok(const Dataset &data, int device)
{
    for (const auto &module : data.modules) {
        for (const TensorStorage &tensor : module) {
            int location = std::numeric_limits<int>::min();
            const cudaError_t status = cudaMemRangeGetAttribute(
                &location, sizeof(location),
                cudaMemRangeAttributeLastPrefetchLocation,
                tensor.managed, tensor.bytes);
            if (status != cudaSuccess || location != device)
                return false;
        }
    }
    return true;
}

void reset_managed_to_cpu(const Dataset &data, cudaStream_t stream)
{
    for (const auto &module : data.modules) {
        for (const TensorStorage &tensor : module) {
            check(cudaMemPrefetchAsync(tensor.managed, tensor.bytes,
                                       cudaCpuDeviceId, stream),
                  "cudaMemPrefetchAsync reset to CPU");
        }
    }
    check(cudaStreamSynchronize(stream), "managed CPU reset synchronize");
}

void launch_layer_delay(const Args &args, unsigned long long delay_cycles,
                        cudaStream_t stream)
{
    if (args.compute_us <= 0.0)
        return;
    delay_kernel<<<1, 1, 0, stream>>>(delay_cycles);
    check(cudaGetLastError(), "launch delay_kernel");
}

Sample run_sample(const Args &args, Backend backend, Dataset &data,
                  const std::vector<int> &layer_to_position,
                  unsigned long long delay_cycles,
                  cudaStream_t copy_stream, cudaStream_t compute_stream,
                  unsigned long long *checksum_device)
{
    const size_t count = data.selected_layers.size();
    if (backend == Backend::Managed)
        reset_managed_to_cpu(data, copy_stream);

    check(cudaMemsetAsync(checksum_device, 0, sizeof(*checksum_device),
                          compute_stream),
          "reset checksum");
    check(cudaStreamSynchronize(compute_stream), "checksum reset synchronize");

    std::vector<cudaEvent_t> ready;
    std::vector<cudaEvent_t> transfer_start;
    std::vector<cudaEvent_t> layer_done;
    std::vector<cudaEvent_t> wait_begin;
    std::vector<cudaEvent_t> wait_end;
    std::vector<cudaEvent_t> read_begin;
    std::vector<cudaEvent_t> read_end;
    create_events(ready, count, "create ready event");
    create_events(transfer_start, count, "create transfer start event");
    create_events(layer_done, count, "create layer done event");
    create_events(wait_begin, count, "create wait begin event");
    create_events(wait_end, count, "create wait end event");
    create_events(read_begin, count, "create read begin event");
    create_events(read_end, count, "create read end event");
    cudaEvent_t epoch = nullptr;
    cudaEvent_t finish = nullptr;
    check(cudaEventCreate(&epoch), "create epoch event");
    check(cudaEventCreate(&finish), "create finish event");

    Sample sample;
    sample.consumer_fingerprint = consumer_fingerprint(backend, data, args);
    check(cudaEventRecord(epoch, compute_stream), "record epoch event");
    if (backend != Backend::Uva)
        check(cudaStreamWaitEvent(copy_stream, epoch, 0),
              "copy stream waits epoch");

    const auto host_begin = std::chrono::steady_clock::now();
    auto enqueue_transfer = [&](size_t position, cudaEvent_t trigger) {
        if (trigger != nullptr) {
            check(cudaStreamWaitEvent(copy_stream, trigger, 0),
                  "copy stream waits layer trigger");
        }
        check(cudaEventRecord(transfer_start[position], copy_stream),
              "record transfer start");
        for (int tensor = 0; tensor < args.tensors; ++tensor) {
            const TensorStorage &storage = data.modules[position][tensor];
            if (backend == Backend::Pinned) {
                check(cudaMemcpyAsync(
                          data.slots[position % args.prefetch_step][tensor],
                          storage.host, storage.bytes, cudaMemcpyHostToDevice,
                          copy_stream),
                      "cudaMemcpyAsync pinned to static slot");
            }
            else {
                check(cudaMemPrefetchAsync(storage.managed, storage.bytes,
                                           args.device, copy_stream),
                      "cudaMemPrefetchAsync managed to GPU");
            }
            ++sample.submitted_ranges;
        }
        check(cudaEventRecord(ready[position], copy_stream),
              "record transfer ready");
    };

    if (backend != Backend::Uva) {
        const size_t initial =
            std::min(count, static_cast<size_t>(args.prefetch_step));
        for (size_t position = 0; position < initial; ++position)
            enqueue_transfer(position, nullptr);
    }

    const int blocks_per_range =
        std::max(1, std::min(64, 256 / std::max(1, args.tensors)));
    for (int layer = 0; layer < args.total_layers; ++layer) {
        const int position = layer_to_position[layer];
        if (position < 0) {
            launch_layer_delay(args, delay_cycles, compute_stream);
            continue;
        }

        check(cudaEventRecord(wait_begin[position], compute_stream),
              "record exposed wait begin");
        if (backend != Backend::Uva) {
            check(cudaStreamWaitEvent(compute_stream, ready[position], 0),
                  "compute waits prefetch ready");
        }
        check(cudaEventRecord(wait_end[position], compute_stream),
              "record exposed wait end");
        check(cudaEventRecord(read_begin[position], compute_stream),
              "record read begin");

        DeviceRange *descriptors = nullptr;
        if (backend == Backend::Pinned) {
            descriptors =
                data.slot_descriptors[position % args.prefetch_step];
        }
        else if (backend == Backend::Managed) {
            descriptors = data.managed_descriptors[position];
        }
        else {
            descriptors = data.uva_descriptors[position];
        }
        checksum_ranges_kernel<<<args.tensors * blocks_per_range, 256, 0,
                                 compute_stream>>>(
            descriptors, args.tensors, blocks_per_range, checksum_device);
        check(cudaGetLastError(), "launch checksum_ranges_kernel");
        check(cudaEventRecord(read_end[position], compute_stream),
              "record read end");
        launch_layer_delay(args, delay_cycles, compute_stream);
        check(cudaEventRecord(layer_done[position], compute_stream),
              "record layer done trigger");

        const size_t target =
            static_cast<size_t>(position + args.prefetch_step);
        if (backend != Backend::Uva && target < count)
            enqueue_transfer(target, layer_done[position]);
    }
    check(cudaEventRecord(finish, compute_stream), "record finish event");
    const auto host_submit_end = std::chrono::steady_clock::now();
    check(cudaEventSynchronize(finish), "synchronize finish event");
    const auto host_end = std::chrono::steady_clock::now();

    sample.host_submit_ms =
        std::chrono::duration<double, std::milli>(host_submit_end - host_begin)
            .count();
    sample.host_block_ms =
        std::chrono::duration<double, std::milli>(host_end - host_begin).count();
    sample.e2e_ms = event_ms(epoch, finish);
    for (size_t position = 0; position < count; ++position) {
        sample.exposed_wait_ms +=
            event_ms(wait_begin[position], wait_end[position]);
        sample.read_ms += event_ms(read_begin[position], read_end[position]);
        if (backend != Backend::Uva) {
            sample.transfer_work_ms +=
                event_ms(transfer_start[position], ready[position]);
        }
    }
    if (backend != Backend::Uva && count > 0) {
        sample.transfer_window_ms =
            event_ms(transfer_start.front(), ready.back());
    }
    check(cudaMemcpy(&sample.checksum, checksum_device,
                     sizeof(sample.checksum), cudaMemcpyDeviceToHost),
          "copy checksum to host");
    sample.checksum_ok = sample.checksum == data.expected_checksum;
    if (backend == Backend::Managed)
        sample.placement_ok = managed_placement_ok(data, args.device);

    destroy_events(ready);
    destroy_events(transfer_start);
    destroy_events(layer_done);
    destroy_events(wait_begin);
    destroy_events(wait_end);
    destroy_events(read_begin);
    destroy_events(read_end);
    cudaEventDestroy(epoch);
    cudaEventDestroy(finish);
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

Result summarize(Backend backend, std::vector<Sample> samples,
                 size_t logical_bytes)
{
    Result result;
    result.backend = backend;
    result.samples = std::move(samples);
    std::vector<double> host_submit;
    std::vector<double> host_block;
    std::vector<double> transfer_work;
    std::vector<double> transfer_window;
    std::vector<double> exposed_wait;
    std::vector<double> read;
    std::vector<double> e2e;
    for (const Sample &sample : result.samples) {
        host_submit.push_back(sample.host_submit_ms);
        host_block.push_back(sample.host_block_ms);
        transfer_work.push_back(sample.transfer_work_ms);
        transfer_window.push_back(sample.transfer_window_ms);
        exposed_wait.push_back(sample.exposed_wait_ms);
        read.push_back(sample.read_ms);
        e2e.push_back(sample.e2e_ms);
    }
    result.functional_ok = std::all_of(
        result.samples.begin(), result.samples.end(), [](const Sample &sample) {
            return sample.checksum_ok && sample.placement_ok;
        });
    const uintptr_t fingerprint = result.samples.front().consumer_fingerprint;
    result.consumer_va_stable = std::all_of(
        result.samples.begin(), result.samples.end(),
        [fingerprint](const Sample &sample) {
            return sample.consumer_fingerprint == fingerprint;
        });
    result.host_submit_ms = percentile(host_submit, 0.5);
    result.host_block_ms = percentile(host_block, 0.5);
    result.transfer_work_ms = percentile(transfer_work, 0.5);
    result.transfer_window_ms = percentile(transfer_window, 0.5);
    result.exposed_wait_ms = percentile(exposed_wait, 0.5);
    result.read_ms = percentile(read, 0.5);
    result.e2e_ms = percentile(e2e, 0.5);
    if (backend != Backend::Uva && result.transfer_work_ms > 0.0) {
        result.transfer_gib_s =
            (static_cast<double>(logical_bytes) / GiB) /
            (result.transfer_work_ms / 1000.0);
    }
    if (result.read_ms > 0.0) {
        result.read_gib_s =
            (static_cast<double>(logical_bytes) / GiB) /
            (result.read_ms / 1000.0);
    }
    return result;
}

void print_integer_list(const char *key, const std::vector<int> &values)
{
    std::printf("%s=", key);
    for (size_t index = 0; index < values.size(); ++index) {
        if (index)
            std::printf(",");
        std::printf("%d", values[index]);
    }
}

void print_schedule(const Args &args, const std::vector<int> &selected)
{
    std::printf("SCHEDULE ");
    print_integer_list("selected_layers", selected);
    std::printf(" slots=");
    for (size_t position = 0; position < selected.size(); ++position) {
        if (position)
            std::printf(",");
        std::printf("%zu", position % args.prefetch_step);
    }
    std::printf(" initial=");
    const size_t initial =
        std::min(selected.size(), static_cast<size_t>(args.prefetch_step));
    for (size_t position = 0; position < initial; ++position) {
        if (position)
            std::printf(",");
        std::printf("%d", selected[position]);
    }
    std::printf(" triggers=");
    bool first = true;
    for (size_t position = 0;
         position + static_cast<size_t>(args.prefetch_step) < selected.size();
         ++position) {
        if (!first)
            std::printf(",");
        first = false;
        std::printf("%d->%d", selected[position],
                    selected[position + args.prefetch_step]);
    }
    if (first)
        std::printf("none");
    std::printf("\n");
}

void print_result(const Result &result, const Args &args,
                  size_t logical_bytes)
{
    std::vector<double> e2e;
    for (const Sample &sample : result.samples)
        e2e.push_back(sample.e2e_ms);
    const Sample &first = result.samples.front();
    std::printf(
        "RESULT backend=%s logical_bytes=%zu transfer_bytes=%zu "
        "offloaded_modules=%zu tensors_per_module=%d submitted_ranges=%d "
        "host_submit_ms_p50=%.3f host_block_ms_p50=%.3f "
        "transfer_work_ms_p50=%.3f "
        "transfer_window_ms_p50=%.3f exposed_wait_ms_p50=%.3f "
        "read_ms_p50=%.3f e2e_ms_p50=%.3f e2e_ms_p95=%.3f "
        "transfer_gib_s=%.3f read_gib_s=%.3f in_place_va=%d "
        "consumer_va_stable=%d checksum=%llu functional_ok=%d\n",
        backend_name(result.backend), logical_bytes,
        result.backend == Backend::Uva ? 0 : logical_bytes,
        result.samples.empty() ? 0 : logical_bytes / args.layer_bytes,
        args.tensors, first.submitted_ranges, result.host_submit_ms,
        result.host_block_ms, result.transfer_work_ms, result.transfer_window_ms,
        result.exposed_wait_ms, result.read_ms, result.e2e_ms,
        percentile(e2e, 0.95), result.transfer_gib_s, result.read_gib_s,
        result.backend == Backend::Pinned ? 0 : 1,
        result.consumer_va_stable ? 1 : 0,
        static_cast<unsigned long long>(first.checksum),
        result.functional_ok ? 1 : 0);
}

const Result *find_result(const std::vector<Result> &results, Backend backend)
{
    for (const Result &result : results) {
        if (result.backend == backend)
            return &result;
    }
    return nullptr;
}

double safe_ratio(double numerator, double denominator)
{
    return denominator > 0.0 ? numerator / denominator
                             : std::numeric_limits<double>::quiet_NaN();
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
    if (!properties.managedMemory || !properties.canMapHostMemory)
        fail("selected device lacks managed or mapped-host memory support");

    const std::vector<int> selected = select_offloaded_layers(args);
    if (selected.empty())
        fail("group policy selected no offloaded layers");
    if (args.prefetch_step > static_cast<int>(selected.size()))
        fail("prefetch-step cannot exceed selected offloaded module count");
    std::vector<int> layer_to_position(args.total_layers, -1);
    for (size_t position = 0; position < selected.size(); ++position)
        layer_to_position[selected[position]] = static_cast<int>(position);

    std::printf(
        "benchmark_begin device=%d gpu=%s total_layers=%d group_size=%d "
        "num_in_group=%d prefetch_step=%d layer_bytes=%zu tensors=%d "
        "compute_us=%.3f warmups=%d repeats=%d concurrent_managed_access=%d "
        "only_backend=%s\n",
        args.device, properties.name, args.total_layers, args.group_size,
        args.num_in_group, args.prefetch_step, args.layer_bytes, args.tensors,
        args.compute_us, args.warmups, args.repeats,
        properties.concurrentManagedAccess,
        args.only_backend.empty() ? "all" : args.only_backend.c_str());
    std::printf(
        "SEMANTICS selector=module_index%%group_size>=group_size-num_in_group "
        "pinned=pinned_cpu+K_static_gpu_slots+single_copy_stream+per_tensor_"
        "async_copy managed=fixed_va+same_trigger_topology "
        "uva=mapped_pinned_direct_read managed_reset=outside_timed_pass\n");
    print_schedule(args, selected);

    Dataset data = make_dataset(args, selected);
    cudaStream_t copy_stream = nullptr;
    cudaStream_t compute_stream = nullptr;
    check(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking),
          "create copy stream");
    check(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking),
          "create compute stream");
    unsigned long long *checksum_device = nullptr;
    check(cudaMalloc(&checksum_device, sizeof(*checksum_device)),
          "cudaMalloc checksum");
    const unsigned long long delay_cycles = static_cast<unsigned long long>(
        args.compute_us * static_cast<double>(properties.clockRate) / 1000.0);

    std::vector<Backend> backends = {
        Backend::Pinned, Backend::Managed, Backend::Uva};
    if (!args.only_backend.empty()) {
        backends.erase(
            std::remove_if(backends.begin(), backends.end(),
                           [&args](Backend backend) {
                               return args.only_backend != backend_name(backend);
                           }),
            backends.end());
    }
    std::vector<Result> results;
    const size_t logical_bytes = selected.size() * args.layer_bytes;
    for (Backend backend : backends) {
        for (int warmup = 0; warmup < args.warmups; ++warmup) {
            (void)run_sample(args, backend, data, layer_to_position,
                             delay_cycles, copy_stream, compute_stream,
                             checksum_device);
        }
        std::vector<Sample> samples;
        for (int repeat = 0; repeat < args.repeats; ++repeat) {
            samples.push_back(run_sample(
                args, backend, data, layer_to_position, delay_cycles,
                copy_stream, compute_stream, checksum_device));
        }
        Result result = summarize(backend, std::move(samples), logical_bytes);
        print_result(result, args, logical_bytes);
        results.push_back(std::move(result));
    }

    const bool functional_success = std::all_of(
        results.begin(), results.end(),
        [](const Result &result) {
            return result.functional_ok && result.consumer_va_stable;
        });
    const Result *pinned = find_result(results, Backend::Pinned);
    const Result *managed = find_result(results, Backend::Managed);
    const Result *uva = find_result(results, Backend::Uva);
    if (pinned && managed && uva) {
        const double managed_transfer_ratio = safe_ratio(
            managed->transfer_gib_s, pinned->transfer_gib_s);
        const double managed_e2e_ratio =
            safe_ratio(pinned->e2e_ms, managed->e2e_ms);
        const double uva_read_ratio =
            safe_ratio(uva->read_gib_s, pinned->read_gib_s);
        std::printf(
            "COMPARISON managed_transfer_vs_pinned=%.3f "
            "managed_e2e_speed_vs_pinned=%.3f "
            "uva_read_vs_pinned_read=%.3f "
            "managed_equivalent=%d threshold=%.3f\n",
            managed_transfer_ratio, managed_e2e_ratio, uva_read_ratio,
            managed_transfer_ratio >= args.equivalent_ratio ? 1 : 0,
            args.equivalent_ratio);
    }
    std::printf(
        "benchmark_result functional_success=%d selected_count=%zu "
        "total_cases=%zu managed_va_preserved=1 "
        "cpu_physical_contiguity_controlled=0\n",
        functional_success ? 1 : 0, selected.size(), results.size());

    cudaFree(checksum_device);
    cudaStreamDestroy(copy_stream);
    cudaStreamDestroy(compute_stream);
    destroy_dataset(data);
    return functional_success ? 0 : 1;
}
