// SPDX-License-Identifier: Apache-2.0
// Standalone CUDA VMM/static-slot CUDA Graph comparison.

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr uint64_t KiB = 1024ULL;
constexpr double GiB = 1024.0 * 1024.0 * 1024.0;
constexpr uint64_t LogicalRegionBytes = 512ULL * KiB;
constexpr uint32_t CanaryWordsPerRegion = 3;

class Error : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

std::string location(const char *file, int line)
{
    std::ostringstream out;
    out << file << ':' << line;
    return out.str();
}

void check_cuda(cudaError_t status, const char *operation, const char *file,
                int line)
{
    if (status == cudaSuccess)
        return;
    std::ostringstream out;
    out << operation << " failed at " << location(file, line) << ": "
        << cudaGetErrorName(status) << " (" << cudaGetErrorString(status)
        << ')';
    throw Error(out.str());
}

void check_driver(CUresult status, const char *operation, const char *file,
                  int line)
{
    if (status == CUDA_SUCCESS)
        return;
    const char *name = nullptr;
    const char *description = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &description);
    std::ostringstream out;
    out << operation << " failed at " << location(file, line) << ": "
        << (name == nullptr ? "CUDA_ERROR_UNKNOWN" : name) << " ("
        << (description == nullptr ? "no description" : description) << ')';
    throw Error(out.str());
}

#define CUDA_CHECK(operation)                                                   \
    check_cuda((operation), #operation, __FILE__, __LINE__)
#define CU_CHECK(operation)                                                     \
    check_driver((operation), #operation, __FILE__, __LINE__)

uint64_t checked_multiply(uint64_t left, uint64_t right, const char *name)
{
    if (right != 0 && left > std::numeric_limits<uint64_t>::max() / right)
        throw Error(std::string(name) + " overflows uint64_t");
    return left * right;
}

uint64_t round_up(uint64_t value, uint64_t alignment)
{
    if (alignment == 0)
        throw Error("zero alignment");
    if (value > std::numeric_limits<uint64_t>::max() - (alignment - 1))
        throw Error("rounded allocation size overflows uint64_t");
    return ((value + alignment - 1) / alignment) * alignment;
}

uint64_t divide_round_up(uint64_t value, uint64_t divisor)
{
    if (divisor == 0)
        throw Error("division by zero");
    return value / divisor + (value % divisor != 0);
}

uint64_t parse_u64(const char *text, const char *name)
{
    if (text == nullptr || *text == '\0' || *text == '-')
        throw Error(std::string("invalid ") + name);
    char *end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0')
        throw Error(std::string("invalid ") + name + ": " + text);
    return static_cast<uint64_t>(value);
}

uint32_t parse_u32(const char *text, const char *name)
{
    const uint64_t value = parse_u64(text, name);
    if (value > std::numeric_limits<uint32_t>::max())
        throw Error(std::string(name) + " exceeds uint32_t");
    return static_cast<uint32_t>(value);
}

std::string json_quote(const std::string &input)
{
    std::ostringstream out;
    out << '"';
    for (const unsigned char character : input) {
        switch (character) {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (character < 0x20) {
                    out << "\\u" << std::hex << std::setw(4)
                        << std::setfill('0') << static_cast<int>(character)
                        << std::dec << std::setfill(' ');
                }
                else {
                    out << static_cast<char>(character);
                }
        }
    }
    out << '"';
    return out.str();
}

enum class Mode {
    GraphRegionAlias,
    GraphRegionDirect,
    EagerRegionAlias,
    GraphCoalescedAlias,
};

enum class VerifyMode {
    Canary,
    FullScan,
};

const char *verify_mode_name(VerifyMode mode)
{
    return mode == VerifyMode::Canary ? "canary" : "full_scan";
}

VerifyMode parse_verify_mode(const std::string &name)
{
    if (name == "canary")
        return VerifyMode::Canary;
    if (name == "full" || name == "full_scan" || name == "full-scan")
        return VerifyMode::FullScan;
    throw Error("unknown --verify-mode: " + name);
}

const char *mode_name(Mode mode)
{
    switch (mode) {
        case Mode::GraphRegionAlias:
            return "graph_region_alias";
        case Mode::GraphRegionDirect:
            return "graph_region_direct";
        case Mode::EagerRegionAlias:
            return "eager_region_alias";
        case Mode::GraphCoalescedAlias:
            return "graph_coalesced_alias";
    }
    return "unknown";
}

Mode parse_mode(const std::string &name)
{
    if (name == "graph_region_alias")
        return Mode::GraphRegionAlias;
    if (name == "graph_region_direct")
        return Mode::GraphRegionDirect;
    if (name == "eager_region_alias")
        return Mode::EagerRegionAlias;
    if (name == "graph_coalesced_alias")
        return Mode::GraphCoalescedAlias;
    throw Error("unknown mode: " + name);
}

std::vector<Mode> all_modes()
{
    return {Mode::GraphRegionAlias, Mode::GraphRegionDirect,
            Mode::EagerRegionAlias, Mode::GraphCoalescedAlias};
}

std::vector<Mode> parse_modes(const std::string &text)
{
    if (text == "all")
        return all_modes();
    std::vector<Mode> modes;
    std::unordered_set<std::string> seen;
    size_t begin = 0;
    while (begin <= text.size()) {
        const size_t comma = text.find(',', begin);
        const std::string token = text.substr(
            begin, comma == std::string::npos ? std::string::npos
                                               : comma - begin);
        if (token.empty())
            throw Error("empty entry in --modes");
        if (seen.insert(token).second)
            modes.push_back(parse_mode(token));
        if (comma == std::string::npos)
            break;
        begin = comma + 1;
    }
    return modes;
}

struct Args {
    int device = 0;
    uint32_t layers = 4;
    uint32_t banks = 2;
    uint32_t regions_per_layer = 1152;
    uint64_t region_bytes = LogicalRegionBytes;
    uint64_t copy_chunk_bytes = LogicalRegionBytes;
    uint32_t verify_blocks = 256;
    uint32_t warmups = 1;
    uint32_t repeats = 3;
    VerifyMode verify_mode = VerifyMode::Canary;
    std::vector<Mode> modes = all_modes();
};

void usage(const char *program)
{
    std::cerr
        << "usage: " << program
        << " [--device N] [--layers N] [--banks N]"
           " [--regions-per-layer N] [--copy-chunk-kib N]"
           " [--verify-mode canary|full]"
           " [--verify-blocks N] [--warmups N] [--repeats N]"
           " [--modes all|NAME[,NAME...]]\n"
           "logical region size is fixed at 512 KiB; copy chunks must be a"
           " multiple of 512 KiB\n"
           "modes: graph_region_alias, graph_region_direct,"
           " eager_region_alias, graph_coalesced_alias\n";
}

Args parse_args(int argc, char **argv)
{
    Args args;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        auto value = [&]() -> const char * {
            if (index + 1 >= argc)
                throw Error("missing value for " + option);
            return argv[++index];
        };
        if (option == "--device")
            args.device = static_cast<int>(parse_u32(value(), "device"));
        else if (option == "--layers")
            args.layers = parse_u32(value(), "layers");
        else if (option == "--banks")
            args.banks = parse_u32(value(), "banks");
        else if (option == "--regions-per-layer")
            args.regions_per_layer =
                parse_u32(value(), "regions-per-layer");
        else if (option == "--copy-chunk-kib")
            args.copy_chunk_bytes = checked_multiply(
                parse_u64(value(), "copy-chunk-kib"), KiB,
                "copy chunk bytes");
        else if (option == "--region-kib") {
            const uint64_t compatibility_value =
                parse_u64(value(), "region-kib");
            if (compatibility_value != LogicalRegionBytes / KiB)
                throw Error("logical --region-kib is fixed at 512");
        }
        else if (option == "--verify-mode")
            args.verify_mode = parse_verify_mode(value());
        else if (option == "--verify-blocks")
            args.verify_blocks = parse_u32(value(), "verify-blocks");
        else if (option == "--warmups")
            args.warmups = parse_u32(value(), "warmups");
        else if (option == "--repeats")
            args.repeats = parse_u32(value(), "repeats");
        else if (option == "--modes")
            args.modes = parse_modes(value());
        else if (option == "--help" || option == "-h") {
            usage(argv[0]);
            std::exit(0);
        }
        else {
            throw Error("unknown argument: " + option);
        }
    }

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (args.device < 0 || args.device >= device_count)
        throw Error("--device is outside the visible device range");
    if (args.layers == 0 || args.layers > 1024)
        throw Error("--layers must be in [1, 1024]");
    if (args.banks == 0 || args.banks > args.layers)
        throw Error("--banks must be in [1, layers]");
    if (args.regions_per_layer == 0 || args.regions_per_layer > 1000000)
        throw Error("--regions-per-layer must be in [1, 1000000]");
    if (args.region_bytes != LogicalRegionBytes)
        throw Error("logical region size must remain 512 KiB");
    if (args.copy_chunk_bytes == 0 ||
        args.copy_chunk_bytes % args.region_bytes != 0)
        throw Error("--copy-chunk-kib must be a positive multiple of 512");
    if (args.verify_blocks == 0 || args.verify_blocks > 65535)
        throw Error("--verify-blocks must be in [1, 65535]");
    if (args.warmups > 100 || args.repeats == 0 || args.repeats > 1000)
        throw Error("--warmups must be <= 100 and --repeats in [1, 1000]");
    if (args.modes.empty())
        throw Error("at least one mode is required");

    const uint64_t layer_bytes = checked_multiply(
        args.region_bytes, args.regions_per_layer, "layer bytes");
    const uint64_t source_bytes =
        checked_multiply(layer_bytes, args.layers, "source bytes");
    if (layer_bytes > std::numeric_limits<size_t>::max() ||
        source_bytes > std::numeric_limits<size_t>::max())
        throw Error("benchmark allocation does not fit size_t");
    return args;
}

__host__ __device__ uint32_t expected_pattern(uint32_t layer,
                                              uint32_t region)
{
    uint32_t value = 0x9e3779b9U * (layer + 1U);
    value ^= 0x85ebca6bU * (region + 1U);
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    return value | 1U;
}

__global__ void verify_full_scan_kernel(const uint32_t *data, uint64_t words,
                                        uint64_t region_words,
                                        uint32_t /*region_count*/,
                                        uint32_t layer,
                                        unsigned long long *layer_errors)
{
    uint64_t local_errors = 0;
    const uint64_t stride =
        static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index =
             static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < words; index += stride) {
        const uint32_t region =
            static_cast<uint32_t>(index / region_words);
        local_errors += data[index] != expected_pattern(layer, region);
    }
    if (local_errors != 0)
        atomicAdd(&layer_errors[layer],
                  static_cast<unsigned long long>(local_errors));
}

__global__ void verify_canary_kernel(const uint32_t *data,
                                     uint64_t /*words*/,
                                     uint64_t region_words,
                                     uint32_t region_count, uint32_t layer,
                                     unsigned long long *layer_errors)
{
    uint64_t local_errors = 0;
    const uint64_t stride =
        static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t region =
             static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         region < region_count; region += stride) {
        const uint64_t base = region * region_words;
        const uint32_t expected =
            expected_pattern(layer, static_cast<uint32_t>(region));
        local_errors += data[base] != expected;
        local_errors += data[base + region_words / 2] != expected;
        local_errors += data[base + region_words - 1] != expected;
    }
    if (local_errors != 0)
        atomicAdd(&layer_errors[layer],
                  static_cast<unsigned long long>(local_errors));
}

uint32_t verification_blocks(const Args &args)
{
    if (args.verify_mode == VerifyMode::FullScan)
        return args.verify_blocks;
    const uint64_t needed =
        (static_cast<uint64_t>(args.regions_per_layer) + 255) / 256;
    return static_cast<uint32_t>(
        std::max<uint64_t>(1, std::min<uint64_t>(args.verify_blocks, needed)));
}

class PinnedSources {
  public:
    PinnedSources(const Args &args, uint64_t layer_bytes)
        : layer_bytes_(layer_bytes), pointers_(args.layers, nullptr)
    {
        try {
            for (uint32_t layer = 0; layer < args.layers; ++layer) {
                CUDA_CHECK(cudaHostAlloc(&pointers_[layer], layer_bytes_,
                                         cudaHostAllocPortable));
                auto *words = static_cast<uint32_t *>(pointers_[layer]);
                const uint64_t region_words =
                    args.region_bytes / sizeof(uint32_t);
                for (uint32_t region = 0;
                     region < args.regions_per_layer; ++region) {
                    std::fill_n(words +
                                    static_cast<uint64_t>(region) *
                                        region_words,
                                region_words,
                                expected_pattern(layer, region));
                }
            }
        }
        catch (...) {
            cleanup();
            throw;
        }
    }

    PinnedSources(const PinnedSources &) = delete;
    PinnedSources &operator=(const PinnedSources &) = delete;

    ~PinnedSources() { cleanup(); }

    const uint8_t *layer(uint32_t index) const
    {
        return static_cast<const uint8_t *>(pointers_.at(index));
    }

    uint64_t bytes() const
    {
        return checked_multiply(layer_bytes_, pointers_.size(),
                                "pinned source bytes");
    }

  private:
    void cleanup() noexcept
    {
        for (void *&pointer : pointers_) {
            if (pointer != nullptr) {
                cudaFreeHost(pointer);
                pointer = nullptr;
            }
        }
    }

    uint64_t layer_bytes_ = 0;
    std::vector<void *> pointers_;
};

struct AllocationInfo {
    CUmemAllocationProp properties{};
    uint64_t minimum_granularity = 0;
    uint64_t recommended_granularity = 0;
    uint64_t physical_bank_bytes = 0;
};

AllocationInfo query_allocation_info(int device, uint64_t logical_bank_bytes)
{
    AllocationInfo info;
    info.properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    info.properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    info.properties.location.id = device;
    info.properties.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    size_t minimum = 0;
    size_t recommended = 0;
    CU_CHECK(cuMemGetAllocationGranularity(
        &minimum, &info.properties,
        CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    CU_CHECK(cuMemGetAllocationGranularity(
        &recommended, &info.properties,
        CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    info.minimum_granularity = minimum;
    info.recommended_granularity = recommended;
    info.physical_bank_bytes = round_up(logical_bank_bytes, minimum);
    return info;
}

void set_access(CUdeviceptr address, size_t bytes, int device,
                CUmemAccess_flags flags)
{
    CUmemAccessDesc descriptor{};
    descriptor.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    descriptor.location.id = device;
    descriptor.flags = flags;
    CU_CHECK(cuMemSetAccess(address, bytes, &descriptor, 1));
}

class VmmBanks {
  public:
    VmmBanks(const Args &args, const AllocationInfo &info)
        : device_(args.device), bank_bytes_(info.physical_bank_bytes),
          handles_(args.banks), upload_(args.banks),
          consumers_(args.layers), upload_mapped_(args.banks, false),
          consumer_mapped_(args.layers, false)
    {
        try {
            for (uint32_t bank = 0; bank < args.banks; ++bank) {
                CU_CHECK(cuMemCreate(&handles_[bank], bank_bytes_,
                                     &info.properties, 0));
                CU_CHECK(cuMemAddressReserve(&upload_[bank], bank_bytes_,
                                             info.minimum_granularity, 0, 0));
                CU_CHECK(cuMemMap(upload_[bank], bank_bytes_, 0,
                                  handles_[bank], 0));
                upload_mapped_[bank] = true;
                set_access(upload_[bank], bank_bytes_, device_,
                           CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
            }

            for (uint32_t layer = 0; layer < args.layers; ++layer) {
                const uint32_t bank = layer % args.banks;
                CU_CHECK(cuMemAddressReserve(&consumers_[layer], bank_bytes_,
                                             info.minimum_granularity, 0, 0));
                CU_CHECK(cuMemMap(consumers_[layer], bank_bytes_, 0,
                                  handles_[bank], 0));
                consumer_mapped_[layer] = true;
                set_access(consumers_[layer], bank_bytes_, device_,
                           CU_MEM_ACCESS_FLAGS_PROT_READ);
            }

            access_ok_ = validate_access();
            if (!access_ok_)
                throw Error("VMM alias access flags did not validate");
        }
        catch (...) {
            cleanup();
            throw;
        }
    }

    VmmBanks(const VmmBanks &) = delete;
    VmmBanks &operator=(const VmmBanks &) = delete;

    ~VmmBanks() { cleanup(); }

    std::vector<uint8_t *> upload_pointers() const
    {
        std::vector<uint8_t *> result;
        result.reserve(upload_.size());
        for (CUdeviceptr pointer : upload_)
            result.push_back(reinterpret_cast<uint8_t *>(pointer));
        return result;
    }

    std::vector<const uint8_t *> consumer_pointers() const
    {
        std::vector<const uint8_t *> result;
        result.reserve(consumers_.size());
        for (CUdeviceptr pointer : consumers_)
            result.push_back(reinterpret_cast<const uint8_t *>(pointer));
        return result;
    }

    bool access_ok() const { return access_ok_; }

  private:
    bool validate_access() const
    {
        CUmemLocation query_location{};
        query_location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        query_location.id = device_;
        for (CUdeviceptr pointer : upload_) {
            unsigned long long flags = 0;
            CU_CHECK(cuMemGetAccess(&flags, &query_location, pointer));
            if (flags != CU_MEM_ACCESS_FLAGS_PROT_READWRITE)
                return false;
        }
        for (CUdeviceptr pointer : consumers_) {
            unsigned long long flags = 0;
            CU_CHECK(cuMemGetAccess(&flags, &query_location, pointer));
            if (flags != CU_MEM_ACCESS_FLAGS_PROT_READ)
                return false;
        }
        return true;
    }

    void cleanup() noexcept
    {
        for (size_t index = 0; index < consumers_.size(); ++index) {
            if (consumer_mapped_[index]) {
                cuMemUnmap(consumers_[index], bank_bytes_);
                consumer_mapped_[index] = false;
            }
            if (consumers_[index] != 0) {
                cuMemAddressFree(consumers_[index], bank_bytes_);
                consumers_[index] = 0;
            }
        }
        for (size_t index = 0; index < upload_.size(); ++index) {
            if (upload_mapped_[index]) {
                cuMemUnmap(upload_[index], bank_bytes_);
                upload_mapped_[index] = false;
            }
            if (upload_[index] != 0) {
                cuMemAddressFree(upload_[index], bank_bytes_);
                upload_[index] = 0;
            }
        }
        for (CUmemGenericAllocationHandle &handle : handles_) {
            if (handle != 0) {
                cuMemRelease(handle);
                handle = 0;
            }
        }
    }

    int device_ = 0;
    size_t bank_bytes_ = 0;
    std::vector<CUmemGenericAllocationHandle> handles_;
    std::vector<CUdeviceptr> upload_;
    std::vector<CUdeviceptr> consumers_;
    std::vector<bool> upload_mapped_;
    std::vector<bool> consumer_mapped_;
    bool access_ok_ = false;
};

class DirectBanks {
  public:
    DirectBanks(uint32_t banks, uint32_t layers, uint64_t bank_bytes)
        : slots_(banks, nullptr), consumers_(layers, nullptr)
    {
        try {
            for (void *&slot : slots_)
                CUDA_CHECK(cudaMalloc(&slot, bank_bytes));
            for (uint32_t layer = 0; layer < layers; ++layer)
                consumers_[layer] =
                    static_cast<const uint8_t *>(slots_[layer % banks]);
        }
        catch (...) {
            cleanup();
            throw;
        }
    }

    DirectBanks(const DirectBanks &) = delete;
    DirectBanks &operator=(const DirectBanks &) = delete;

    ~DirectBanks() { cleanup(); }

    std::vector<uint8_t *> upload_pointers() const
    {
        std::vector<uint8_t *> result;
        result.reserve(slots_.size());
        for (void *pointer : slots_)
            result.push_back(static_cast<uint8_t *>(pointer));
        return result;
    }

    const std::vector<const uint8_t *> &consumer_pointers() const
    {
        return consumers_;
    }

  private:
    void cleanup() noexcept
    {
        for (void *&pointer : slots_) {
            if (pointer != nullptr) {
                cudaFree(pointer);
                pointer = nullptr;
            }
        }
    }

    std::vector<void *> slots_;
    std::vector<const uint8_t *> consumers_;
};

struct Sample {
    uint32_t index = 0;
    bool correct = false;
    uint64_t error_words = 0;
    double host_submit_ms = 0.0;
    double gpu_latency_ms = 0.0;
    double e2e_latency_ms = 0.0;
    double gib_per_second = 0.0;
};

struct RunnerMetadata {
    size_t graph_nodes = 0;
    size_t graph_edges = 0;
    uint64_t memcpy_operations = 0;
    uint64_t graph_memcpy_nodes = 0;
    uint64_t graph_verify_nodes = 0;
    uint64_t effective_copy_chunk_bytes = 0;
    uint64_t copy_operations_per_layer = 0;
    uint64_t tail_copy_bytes = 0;
    uint64_t verified_words_per_layer = 0;
    uint64_t verified_words_per_replay = 0;
    uint32_t effective_verify_blocks = 0;
    VerifyMode verify_mode = VerifyMode::Canary;
    double graph_build_ms = 0.0;
    double graph_instantiate_ms = 0.0;
};

struct KernelArguments {
    const uint32_t *data = nullptr;
    uint64_t words = 0;
    uint64_t region_words = 0;
    uint32_t region_count = 0;
    uint32_t layer = 0;
    unsigned long long *errors = nullptr;
    void *parameters[6]{};

    void bind()
    {
        parameters[0] = &data;
        parameters[1] = &words;
        parameters[2] = &region_words;
        parameters[3] = &region_count;
        parameters[4] = &layer;
        parameters[5] = &errors;
    }
};

class GraphRunner {
  public:
    GraphRunner(const Args &args, const PinnedSources &sources,
                std::vector<uint8_t *> upload,
                std::vector<const uint8_t *> consumers, bool coalesced)
        : args_(args), sources_(sources), upload_(std::move(upload)),
          consumers_(std::move(consumers)), coalesced_(coalesced),
          kernel_arguments_(args.layers)
    {
        try {
            CUDA_CHECK(cudaMalloc(&errors_,
                                  args.layers * sizeof(*errors_)));
            CUDA_CHECK(cudaStreamCreateWithFlags(&stream_,
                                                 cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreate(&begin_));
            CUDA_CHECK(cudaEventCreate(&end_));
            build_graph();
        }
        catch (...) {
            cleanup();
            throw;
        }
    }

    GraphRunner(const GraphRunner &) = delete;
    GraphRunner &operator=(const GraphRunner &) = delete;

    ~GraphRunner() { cleanup(); }

    const RunnerMetadata &metadata() const { return metadata_; }

    Sample run(uint32_t index, uint64_t transfer_bytes)
    {
        const auto wall_begin = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventRecord(begin_, stream_));
        CUDA_CHECK(cudaGraphLaunch(executable_, stream_));
        CUDA_CHECK(cudaEventRecord(end_, stream_));
        const auto submit_end = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventSynchronize(end_));
        const auto wall_end = std::chrono::steady_clock::now();

        std::vector<unsigned long long> errors(args_.layers);
        CUDA_CHECK(cudaMemcpy(errors.data(), errors_,
                              errors.size() * sizeof(errors[0]),
                              cudaMemcpyDeviceToHost));
        const uint64_t error_words =
            std::accumulate(errors.begin(), errors.end(), uint64_t{0});
        float gpu_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, begin_, end_));

        Sample sample;
        sample.index = index;
        sample.correct = error_words == 0;
        sample.error_words = error_words;
        sample.host_submit_ms =
            std::chrono::duration<double, std::milli>(submit_end - wall_begin)
                .count();
        sample.gpu_latency_ms = gpu_ms;
        sample.e2e_latency_ms =
            std::chrono::duration<double, std::milli>(wall_end - wall_begin)
                .count();
        if (gpu_ms > 0.0)
            sample.gib_per_second =
                (static_cast<double>(transfer_bytes) / GiB) /
                (static_cast<double>(gpu_ms) / 1000.0);
        return sample;
    }

  private:
    void build_graph()
    {
        const auto build_begin = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaGraphCreate(&graph_, 0));

        cudaMemsetParams memset_parameters{};
        memset_parameters.dst = errors_;
        memset_parameters.pitch = 0;
        memset_parameters.value = 0;
        memset_parameters.elementSize = 1;
        memset_parameters.width = args_.layers * sizeof(*errors_);
        memset_parameters.height = 1;
        cudaGraphNode_t reset_node = nullptr;
        CUDA_CHECK(cudaGraphAddMemsetNode(&reset_node, graph_, nullptr, 0,
                                          &memset_parameters));

        const uint64_t layer_bytes = checked_multiply(
            args_.region_bytes, args_.regions_per_layer, "layer bytes");
        const uint64_t layer_words = layer_bytes / sizeof(uint32_t);
        const uint64_t region_words = args_.region_bytes / sizeof(uint32_t);
        const uint64_t copy_chunk_bytes =
            coalesced_ ? layer_bytes : args_.copy_chunk_bytes;
        const uint64_t copies_per_layer =
            divide_round_up(layer_bytes, copy_chunk_bytes);
        if (copies_per_layer > std::numeric_limits<size_t>::max())
            throw Error("copy-node count does not fit size_t");
        std::vector<cudaGraphNode_t> verify_nodes(args_.layers, nullptr);

        for (uint32_t layer = 0; layer < args_.layers; ++layer) {
            const uint32_t bank = layer % args_.banks;
            const cudaGraphNode_t predecessor =
                layer >= args_.banks ? verify_nodes[layer - args_.banks]
                                     : reset_node;
            std::vector<cudaGraphNode_t> copy_nodes;
            copy_nodes.reserve(static_cast<size_t>(copies_per_layer));
            for (uint64_t copy = 0; copy < copies_per_layer; ++copy) {
                const uint64_t offset = copy * copy_chunk_bytes;
                const uint64_t bytes =
                    std::min(copy_chunk_bytes, layer_bytes - offset);
                cudaGraphNode_t copy_node = nullptr;
                CUDA_CHECK(cudaGraphAddMemcpyNode1D(
                    &copy_node, graph_, &predecessor, 1,
                    upload_[bank] + offset, sources_.layer(layer) + offset,
                    bytes, cudaMemcpyHostToDevice));
                copy_nodes.push_back(copy_node);
            }

            KernelArguments &arguments = kernel_arguments_[layer];
            arguments.data = reinterpret_cast<const uint32_t *>(
                consumers_[layer]);
            arguments.words = layer_words;
            arguments.region_words = region_words;
            arguments.region_count = args_.regions_per_layer;
            arguments.layer = layer;
            arguments.errors = errors_;
            arguments.bind();

            cudaKernelNodeParams kernel_parameters{};
            kernel_parameters.func = reinterpret_cast<void *>(
                args_.verify_mode == VerifyMode::Canary
                    ? verify_canary_kernel
                    : verify_full_scan_kernel);
            kernel_parameters.gridDim =
                dim3(verification_blocks(args_), 1, 1);
            kernel_parameters.blockDim = dim3(256, 1, 1);
            kernel_parameters.sharedMemBytes = 0;
            kernel_parameters.kernelParams = arguments.parameters;
            kernel_parameters.extra = nullptr;
            if (layer != 0)
                copy_nodes.push_back(verify_nodes[layer - 1]);
            CUDA_CHECK(cudaGraphAddKernelNode(
                &verify_nodes[layer], graph_, copy_nodes.data(),
                copy_nodes.size(), &kernel_parameters));
        }
        const auto build_end = std::chrono::steady_clock::now();

        const auto instantiate_begin = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaGraphInstantiate(&executable_, graph_, nullptr, nullptr,
                                        0));
        const auto instantiate_end = std::chrono::steady_clock::now();

        CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr,
                                     &metadata_.graph_nodes));
        CUDA_CHECK(cudaGraphGetEdges(graph_, nullptr, nullptr,
                                     &metadata_.graph_edges));
        metadata_.graph_memcpy_nodes = checked_multiply(
            args_.layers, copies_per_layer,
            "graph memcpy nodes");
        metadata_.memcpy_operations = metadata_.graph_memcpy_nodes;
        metadata_.graph_verify_nodes = args_.layers;
        metadata_.effective_copy_chunk_bytes = copy_chunk_bytes;
        metadata_.copy_operations_per_layer = copies_per_layer;
        metadata_.tail_copy_bytes =
            layer_bytes - (copies_per_layer - 1) * copy_chunk_bytes;
        metadata_.verified_words_per_layer =
            args_.verify_mode == VerifyMode::Canary
                ? checked_multiply(args_.regions_per_layer,
                                   CanaryWordsPerRegion,
                                   "canary words per layer")
                : layer_words;
        metadata_.verified_words_per_replay = checked_multiply(
            metadata_.verified_words_per_layer, args_.layers,
            "verified words per replay");
        metadata_.effective_verify_blocks = verification_blocks(args_);
        metadata_.verify_mode = args_.verify_mode;
        metadata_.graph_build_ms =
            std::chrono::duration<double, std::milli>(build_end - build_begin)
                .count();
        metadata_.graph_instantiate_ms =
            std::chrono::duration<double, std::milli>(instantiate_end -
                                                       instantiate_begin)
                .count();
    }

    void cleanup() noexcept
    {
        if (stream_ != nullptr)
            cudaStreamSynchronize(stream_);
        if (executable_ != nullptr) {
            cudaGraphExecDestroy(executable_);
            executable_ = nullptr;
        }
        if (graph_ != nullptr) {
            cudaGraphDestroy(graph_);
            graph_ = nullptr;
        }
        if (begin_ != nullptr) {
            cudaEventDestroy(begin_);
            begin_ = nullptr;
        }
        if (end_ != nullptr) {
            cudaEventDestroy(end_);
            end_ = nullptr;
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        if (errors_ != nullptr) {
            cudaFree(errors_);
            errors_ = nullptr;
        }
    }

    const Args &args_;
    const PinnedSources &sources_;
    std::vector<uint8_t *> upload_;
    std::vector<const uint8_t *> consumers_;
    bool coalesced_ = false;
    std::vector<KernelArguments> kernel_arguments_;
    unsigned long long *errors_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t begin_ = nullptr;
    cudaEvent_t end_ = nullptr;
    RunnerMetadata metadata_;
};

class EagerRunner {
  public:
    EagerRunner(const Args &args, const PinnedSources &sources,
                std::vector<uint8_t *> upload,
                std::vector<const uint8_t *> consumers)
        : args_(args), sources_(sources), upload_(std::move(upload)),
          consumers_(std::move(consumers)), copy_streams_(args.banks),
          verify_streams_(args.banks), copy_done_(args.layers),
          verify_done_(args.layers)
    {
        try {
            CUDA_CHECK(cudaMalloc(&errors_,
                                  args.layers * sizeof(*errors_)));
            CUDA_CHECK(cudaStreamCreateWithFlags(&control_stream_,
                                                 cudaStreamNonBlocking));
            for (uint32_t bank = 0; bank < args.banks; ++bank) {
                CUDA_CHECK(cudaStreamCreateWithFlags(
                    &copy_streams_[bank], cudaStreamNonBlocking));
                CUDA_CHECK(cudaStreamCreateWithFlags(
                    &verify_streams_[bank], cudaStreamNonBlocking));
            }
            CUDA_CHECK(cudaEventCreate(&begin_));
            CUDA_CHECK(cudaEventCreate(&reset_done_));
            CUDA_CHECK(cudaEventCreate(&end_));
            for (uint32_t layer = 0; layer < args.layers; ++layer) {
                CUDA_CHECK(cudaEventCreateWithFlags(
                    &copy_done_[layer], cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreateWithFlags(
                    &verify_done_[layer], cudaEventDisableTiming));
            }
            const uint64_t layer_bytes = checked_multiply(
                args.region_bytes, args.regions_per_layer, "layer bytes");
            const uint64_t copies_per_layer =
                divide_round_up(layer_bytes, args.copy_chunk_bytes);
            metadata_.memcpy_operations = checked_multiply(
                args.layers, copies_per_layer,
                "eager memcpy operations");
            metadata_.graph_nodes = 0;
            metadata_.graph_edges = 0;
            metadata_.graph_memcpy_nodes = 0;
            metadata_.graph_verify_nodes = 0;
            metadata_.effective_copy_chunk_bytes = args.copy_chunk_bytes;
            metadata_.copy_operations_per_layer = copies_per_layer;
            metadata_.tail_copy_bytes =
                layer_bytes -
                (copies_per_layer - 1) * args.copy_chunk_bytes;
            metadata_.verified_words_per_layer =
                args.verify_mode == VerifyMode::Canary
                    ? checked_multiply(args.regions_per_layer,
                                       CanaryWordsPerRegion,
                                       "canary words per layer")
                    : layer_bytes / sizeof(uint32_t);
            metadata_.verified_words_per_replay = checked_multiply(
                metadata_.verified_words_per_layer, args.layers,
                "verified words per replay");
            metadata_.effective_verify_blocks = verification_blocks(args);
            metadata_.verify_mode = args.verify_mode;
        }
        catch (...) {
            cleanup();
            throw;
        }
    }

    EagerRunner(const EagerRunner &) = delete;
    EagerRunner &operator=(const EagerRunner &) = delete;

    ~EagerRunner() { cleanup(); }

    const RunnerMetadata &metadata() const { return metadata_; }

    Sample run(uint32_t index, uint64_t transfer_bytes)
    {
        const uint64_t layer_bytes = checked_multiply(
            args_.region_bytes, args_.regions_per_layer, "layer bytes");
        const uint64_t layer_words = layer_bytes / sizeof(uint32_t);
        const uint64_t region_words = args_.region_bytes / sizeof(uint32_t);

        const auto wall_begin = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventRecord(begin_, control_stream_));
        CUDA_CHECK(cudaMemsetAsync(errors_, 0,
                                   args_.layers * sizeof(*errors_),
                                   control_stream_));
        CUDA_CHECK(cudaEventRecord(reset_done_, control_stream_));

        for (uint32_t layer = 0; layer < args_.layers; ++layer) {
            const uint32_t bank = layer % args_.banks;
            const cudaEvent_t predecessor =
                layer >= args_.banks ? verify_done_[layer - args_.banks]
                                     : reset_done_;
            CUDA_CHECK(cudaStreamWaitEvent(copy_streams_[bank], predecessor,
                                           0));
            const uint64_t copies_per_layer =
                divide_round_up(layer_bytes, args_.copy_chunk_bytes);
            for (uint64_t copy = 0; copy < copies_per_layer; ++copy) {
                const uint64_t offset = copy * args_.copy_chunk_bytes;
                const uint64_t bytes =
                    std::min(args_.copy_chunk_bytes, layer_bytes - offset);
                CUDA_CHECK(cudaMemcpyAsync(
                    upload_[bank] + offset, sources_.layer(layer) + offset,
                    bytes, cudaMemcpyHostToDevice,
                    copy_streams_[bank]));
            }
            CUDA_CHECK(cudaEventRecord(copy_done_[layer],
                                       copy_streams_[bank]));
            CUDA_CHECK(cudaStreamWaitEvent(verify_streams_[bank],
                                           copy_done_[layer], 0));
            if (layer != 0) {
                CUDA_CHECK(cudaStreamWaitEvent(verify_streams_[bank],
                                               verify_done_[layer - 1], 0));
            }
            if (args_.verify_mode == VerifyMode::Canary) {
                verify_canary_kernel<<<verification_blocks(args_), 256, 0,
                                       verify_streams_[bank]>>>(
                    reinterpret_cast<const uint32_t *>(consumers_[layer]),
                    layer_words, region_words, args_.regions_per_layer, layer,
                    errors_);
            }
            else {
                verify_full_scan_kernel<<<verification_blocks(args_), 256, 0,
                                          verify_streams_[bank]>>>(
                    reinterpret_cast<const uint32_t *>(consumers_[layer]),
                    layer_words, region_words, args_.regions_per_layer, layer,
                    errors_);
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(verify_done_[layer],
                                       verify_streams_[bank]));
        }
        for (cudaEvent_t event : verify_done_)
            CUDA_CHECK(cudaStreamWaitEvent(control_stream_, event, 0));
        CUDA_CHECK(cudaEventRecord(end_, control_stream_));
        const auto submit_end = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventSynchronize(end_));
        const auto wall_end = std::chrono::steady_clock::now();

        std::vector<unsigned long long> errors(args_.layers);
        CUDA_CHECK(cudaMemcpy(errors.data(), errors_,
                              errors.size() * sizeof(errors[0]),
                              cudaMemcpyDeviceToHost));
        const uint64_t error_words =
            std::accumulate(errors.begin(), errors.end(), uint64_t{0});
        float gpu_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, begin_, end_));

        Sample sample;
        sample.index = index;
        sample.correct = error_words == 0;
        sample.error_words = error_words;
        sample.host_submit_ms =
            std::chrono::duration<double, std::milli>(submit_end - wall_begin)
                .count();
        sample.gpu_latency_ms = gpu_ms;
        sample.e2e_latency_ms =
            std::chrono::duration<double, std::milli>(wall_end - wall_begin)
                .count();
        if (gpu_ms > 0.0)
            sample.gib_per_second =
                (static_cast<double>(transfer_bytes) / GiB) /
                (static_cast<double>(gpu_ms) / 1000.0);
        return sample;
    }

  private:
    void cleanup() noexcept
    {
        if (control_stream_ != nullptr)
            cudaStreamSynchronize(control_stream_);
        for (cudaStream_t stream : copy_streams_) {
            if (stream != nullptr)
                cudaStreamSynchronize(stream);
        }
        for (cudaStream_t stream : verify_streams_) {
            if (stream != nullptr)
                cudaStreamSynchronize(stream);
        }
        for (cudaEvent_t &event : copy_done_) {
            if (event != nullptr) {
                cudaEventDestroy(event);
                event = nullptr;
            }
        }
        for (cudaEvent_t &event : verify_done_) {
            if (event != nullptr) {
                cudaEventDestroy(event);
                event = nullptr;
            }
        }
        if (begin_ != nullptr) {
            cudaEventDestroy(begin_);
            begin_ = nullptr;
        }
        if (reset_done_ != nullptr) {
            cudaEventDestroy(reset_done_);
            reset_done_ = nullptr;
        }
        if (end_ != nullptr) {
            cudaEventDestroy(end_);
            end_ = nullptr;
        }
        for (cudaStream_t &stream : copy_streams_) {
            if (stream != nullptr) {
                cudaStreamDestroy(stream);
                stream = nullptr;
            }
        }
        for (cudaStream_t &stream : verify_streams_) {
            if (stream != nullptr) {
                cudaStreamDestroy(stream);
                stream = nullptr;
            }
        }
        if (control_stream_ != nullptr) {
            cudaStreamDestroy(control_stream_);
            control_stream_ = nullptr;
        }
        if (errors_ != nullptr) {
            cudaFree(errors_);
            errors_ = nullptr;
        }
    }

    const Args &args_;
    const PinnedSources &sources_;
    std::vector<uint8_t *> upload_;
    std::vector<const uint8_t *> consumers_;
    std::vector<cudaStream_t> copy_streams_;
    std::vector<cudaStream_t> verify_streams_;
    std::vector<cudaEvent_t> copy_done_;
    std::vector<cudaEvent_t> verify_done_;
    unsigned long long *errors_ = nullptr;
    cudaStream_t control_stream_ = nullptr;
    cudaEvent_t begin_ = nullptr;
    cudaEvent_t reset_done_ = nullptr;
    cudaEvent_t end_ = nullptr;
    RunnerMetadata metadata_;
};

uint64_t pointer_fingerprint(const std::vector<uint8_t *> &upload,
                             const std::vector<const uint8_t *> &consumers)
{
    uint64_t fingerprint = 1469598103934665603ULL;
    auto mix = [&](const void *pointer) {
        fingerprint ^= static_cast<uint64_t>(
            reinterpret_cast<uintptr_t>(pointer));
        fingerprint *= 1099511628211ULL;
    };
    for (const void *pointer : upload)
        mix(pointer);
    for (const void *pointer : consumers)
        mix(pointer);
    return fingerprint;
}

bool unique_consumers(const std::vector<const uint8_t *> &consumers)
{
    std::unordered_set<const void *> addresses;
    for (const void *pointer : consumers)
        addresses.insert(pointer);
    return addresses.size() == consumers.size();
}

struct StaticMetadata {
    bool uses_vmm_alias = false;
    bool consumer_read_only = false;
    bool vmm_access_valid = true;
    bool consumer_va_unique = false;
    bool va_stable = true;
    uint64_t hbm_backing_bytes = 0;
    uint64_t mapped_va_bytes = 0;
    uint64_t consumer_alias_bytes = 0;
};

double percentile(std::vector<double> values, double fraction)
{
    if (values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

double mean(const std::vector<double> &values)
{
    if (values.empty())
        return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

void emit_config(const Args &args, const cudaDeviceProp &device,
                 const AllocationInfo &allocation, uint64_t layer_bytes,
                 uint64_t transfer_bytes, uint64_t source_bytes)
{
    std::cout << std::setprecision(9)
              << "{\"record\":\"config\",\"schema_version\":1"
              << ",\"cuda_runtime_version\":" << CUDART_VERSION
              << ",\"device\":" << args.device
              << ",\"device_name\":" << json_quote(device.name)
              << ",\"layers\":" << args.layers
              << ",\"banks\":" << args.banks
              << ",\"concurrency\":1"
              << ",\"regions_per_layer\":" << args.regions_per_layer
              << ",\"logical_region_bytes\":" << args.region_bytes
              << ",\"copy_chunk_bytes_requested\":"
              << args.copy_chunk_bytes
              << ",\"layer_bytes\":" << layer_bytes
              << ",\"logical_transfer_bytes\":" << transfer_bytes
              << ",\"pinned_source_bytes\":" << source_bytes
              << ",\"fixed_pinned_sources\":true"
              << ",\"verify_mode\":"
              << json_quote(verify_mode_name(args.verify_mode))
              << ",\"canary_words_per_region\":"
              << CanaryWordsPerRegion
              << ",\"timing_scope\":"
              << json_quote("replay_pipeline_reset_copy_consumer_check")
              << ",\"warmups\":" << args.warmups
              << ",\"repeats\":" << args.repeats
              << ",\"verify_blocks\":" << args.verify_blocks
              << ",\"vmm_minimum_granularity_bytes\":"
              << allocation.minimum_granularity
              << ",\"vmm_recommended_granularity_bytes\":"
              << allocation.recommended_granularity
              << ",\"physical_bank_bytes\":"
              << allocation.physical_bank_bytes << ",\"modes\":[";
    for (size_t index = 0; index < args.modes.size(); ++index) {
        if (index != 0)
            std::cout << ',';
        std::cout << json_quote(mode_name(args.modes[index]));
    }
    std::cout << "]}\n";
}

void emit_sample(Mode mode, const Sample &sample,
                 const RunnerMetadata &runner, const StaticMetadata &metadata)
{
    std::cout << std::setprecision(9)
              << "{\"record\":\"sample\",\"schema_version\":1"
              << ",\"mode\":" << json_quote(mode_name(mode))
              << ",\"sample\":" << sample.index
              << ",\"correct\":" << (sample.correct ? "true" : "false")
              << ",\"error_words\":" << sample.error_words
              << ",\"memcpy_operations\":" << runner.memcpy_operations
              << ",\"graph_memcpy_nodes\":"
              << runner.graph_memcpy_nodes
              << ",\"graph_verify_nodes\":"
              << runner.graph_verify_nodes
              << ",\"graph_total_nodes\":" << runner.graph_nodes
              << ",\"graph_edges\":" << runner.graph_edges
              << ",\"verify_mode\":"
              << json_quote(verify_mode_name(runner.verify_mode))
              << ",\"effective_copy_chunk_bytes\":"
              << runner.effective_copy_chunk_bytes
              << ",\"copy_operations_per_layer\":"
              << runner.copy_operations_per_layer
              << ",\"tail_copy_bytes\":" << runner.tail_copy_bytes
              << ",\"verified_words_per_layer\":"
              << runner.verified_words_per_layer
              << ",\"verified_words_per_replay\":"
              << runner.verified_words_per_replay
              << ",\"effective_verify_blocks\":"
              << runner.effective_verify_blocks
              << ",\"timing_scope\":"
              << json_quote("replay_pipeline_reset_copy_consumer_check")
              << ",\"host_submit_ms\":" << sample.host_submit_ms
              << ",\"gpu_latency_ms\":" << sample.gpu_latency_ms
              << ",\"e2e_latency_ms\":" << sample.e2e_latency_ms
              << ",\"pipeline_gib_s\":" << sample.gib_per_second
              << ",\"uses_vmm_alias\":"
              << (metadata.uses_vmm_alias ? "true" : "false")
              << ",\"consumer_read_only\":"
              << (metadata.consumer_read_only ? "true" : "false")
              << ",\"vmm_access_valid\":"
              << (metadata.vmm_access_valid ? "true" : "false")
              << ",\"consumer_va_unique_by_layer\":"
              << (metadata.consumer_va_unique ? "true" : "false")
              << ",\"va_stable\":"
              << (metadata.va_stable ? "true" : "false")
              << ",\"hbm_backing_bytes\":" << metadata.hbm_backing_bytes
              << ",\"mapped_va_bytes\":" << metadata.mapped_va_bytes
              << ",\"consumer_alias_bytes\":"
              << metadata.consumer_alias_bytes << "}\n";
}

void emit_summary(Mode mode, const std::vector<Sample> &samples,
                  const RunnerMetadata &runner,
                  const StaticMetadata &metadata)
{
    std::vector<double> submit;
    std::vector<double> gpu;
    std::vector<double> e2e;
    std::vector<double> bandwidth;
    uint64_t errors = 0;
    bool correct = true;
    for (const Sample &sample : samples) {
        submit.push_back(sample.host_submit_ms);
        gpu.push_back(sample.gpu_latency_ms);
        e2e.push_back(sample.e2e_latency_ms);
        bandwidth.push_back(sample.gib_per_second);
        errors += sample.error_words;
        correct &= sample.correct;
    }

    std::cout << std::setprecision(9)
              << "{\"record\":\"summary\",\"schema_version\":1"
              << ",\"mode\":" << json_quote(mode_name(mode))
              << ",\"correct\":" << (correct ? "true" : "false")
              << ",\"total_error_words\":" << errors
              << ",\"samples\":" << samples.size()
              << ",\"memcpy_operations_per_replay\":"
              << runner.memcpy_operations
              << ",\"graph_memcpy_nodes\":"
              << runner.graph_memcpy_nodes
              << ",\"graph_verify_nodes\":"
              << runner.graph_verify_nodes
              << ",\"graph_total_nodes\":" << runner.graph_nodes
              << ",\"graph_edges\":" << runner.graph_edges
              << ",\"verify_mode\":"
              << json_quote(verify_mode_name(runner.verify_mode))
              << ",\"effective_copy_chunk_bytes\":"
              << runner.effective_copy_chunk_bytes
              << ",\"copy_operations_per_layer\":"
              << runner.copy_operations_per_layer
              << ",\"tail_copy_bytes\":" << runner.tail_copy_bytes
              << ",\"verified_words_per_layer\":"
              << runner.verified_words_per_layer
              << ",\"verified_words_per_replay\":"
              << runner.verified_words_per_replay
              << ",\"effective_verify_blocks\":"
              << runner.effective_verify_blocks
              << ",\"timing_scope\":"
              << json_quote("replay_pipeline_reset_copy_consumer_check")
              << ",\"graph_build_ms\":" << runner.graph_build_ms
              << ",\"graph_instantiate_ms\":"
              << runner.graph_instantiate_ms
              << ",\"host_submit_ms_mean\":" << mean(submit)
              << ",\"host_submit_ms_p50\":" << percentile(submit, 0.50)
              << ",\"gpu_latency_ms_mean\":" << mean(gpu)
              << ",\"gpu_latency_ms_p50\":" << percentile(gpu, 0.50)
              << ",\"e2e_latency_ms_mean\":" << mean(e2e)
              << ",\"e2e_latency_ms_p50\":" << percentile(e2e, 0.50)
              << ",\"pipeline_gib_s_mean\":" << mean(bandwidth)
              << ",\"pipeline_gib_s_p50\":"
              << percentile(bandwidth, 0.50)
              << ",\"uses_vmm_alias\":"
              << (metadata.uses_vmm_alias ? "true" : "false")
              << ",\"consumer_read_only\":"
              << (metadata.consumer_read_only ? "true" : "false")
              << ",\"vmm_access_valid\":"
              << (metadata.vmm_access_valid ? "true" : "false")
              << ",\"consumer_va_unique_by_layer\":"
              << (metadata.consumer_va_unique ? "true" : "false")
              << ",\"va_stable\":"
              << (metadata.va_stable ? "true" : "false")
              << ",\"hbm_backing_bytes\":" << metadata.hbm_backing_bytes
              << ",\"mapped_va_bytes\":" << metadata.mapped_va_bytes
              << ",\"consumer_alias_bytes\":"
              << metadata.consumer_alias_bytes << "}\n";
}

template <typename Runner>
void execute_runner(Mode mode, Runner &runner, uint64_t transfer_bytes,
                    StaticMetadata metadata, uint64_t initial_fingerprint,
                    const std::vector<uint8_t *> &upload,
                    const std::vector<const uint8_t *> &consumers,
                    uint32_t warmups, uint32_t repeats)
{
    for (uint32_t warmup = 0; warmup < warmups; ++warmup) {
        const Sample sample = runner.run(warmup, transfer_bytes);
        if (!sample.correct)
            throw Error(std::string(mode_name(mode)) +
                        " failed correctness during warmup");
    }

    std::vector<Sample> samples;
    samples.reserve(repeats);
    for (uint32_t index = 0; index < repeats; ++index) {
        Sample sample = runner.run(index, transfer_bytes);
        metadata.va_stable &=
            pointer_fingerprint(upload, consumers) == initial_fingerprint;
        emit_sample(mode, sample, runner.metadata(), metadata);
        samples.push_back(sample);
    }
    emit_summary(mode, samples, runner.metadata(), metadata);
    if (!metadata.va_stable)
        throw Error(std::string(mode_name(mode)) + " changed a benchmark VA");
    for (const Sample &sample : samples) {
        if (!sample.correct)
            throw Error(std::string(mode_name(mode)) +
                        " failed correctness");
    }
}

void run_alias_mode(Mode mode, const Args &args,
                    const AllocationInfo &allocation,
                    const PinnedSources &sources,
                    uint64_t transfer_bytes)
{
    VmmBanks banks(args, allocation);
    const std::vector<uint8_t *> upload = banks.upload_pointers();
    const std::vector<const uint8_t *> consumers = banks.consumer_pointers();
    const uint64_t fingerprint = pointer_fingerprint(upload, consumers);

    StaticMetadata metadata;
    metadata.uses_vmm_alias = true;
    metadata.consumer_read_only = true;
    metadata.vmm_access_valid = banks.access_ok();
    metadata.consumer_va_unique = unique_consumers(consumers);
    metadata.va_stable = true;
    metadata.hbm_backing_bytes = checked_multiply(
        allocation.physical_bank_bytes, args.banks, "HBM backing bytes");
    metadata.mapped_va_bytes = checked_multiply(
        allocation.physical_bank_bytes, args.banks + args.layers,
        "mapped VA bytes");
    metadata.consumer_alias_bytes = checked_multiply(
        allocation.physical_bank_bytes, args.layers,
        "consumer alias bytes");

    if (mode == Mode::EagerRegionAlias) {
        EagerRunner runner(args, sources, upload, consumers);
        execute_runner(mode, runner, transfer_bytes, metadata, fingerprint,
                       upload, consumers, args.warmups, args.repeats);
    }
    else {
        const bool coalesced = mode == Mode::GraphCoalescedAlias;
        GraphRunner runner(args, sources, upload, consumers, coalesced);
        execute_runner(mode, runner, transfer_bytes, metadata, fingerprint,
                       upload, consumers, args.warmups, args.repeats);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

void run_direct_mode(Mode mode, const Args &args,
                     const AllocationInfo &allocation,
                     const PinnedSources &sources, uint64_t transfer_bytes)
{
    DirectBanks banks(args.banks, args.layers,
                      allocation.physical_bank_bytes);
    const std::vector<uint8_t *> upload = banks.upload_pointers();
    const std::vector<const uint8_t *> consumers = banks.consumer_pointers();
    const uint64_t fingerprint = pointer_fingerprint(upload, consumers);

    StaticMetadata metadata;
    metadata.uses_vmm_alias = false;
    metadata.consumer_read_only = false;
    metadata.vmm_access_valid = true;
    metadata.consumer_va_unique = unique_consumers(consumers);
    metadata.va_stable = true;
    metadata.hbm_backing_bytes = checked_multiply(
        allocation.physical_bank_bytes, args.banks, "HBM backing bytes");
    metadata.mapped_va_bytes = metadata.hbm_backing_bytes;
    metadata.consumer_alias_bytes = 0;

    GraphRunner runner(args, sources, upload, consumers, false);
    execute_runner(mode, runner, transfer_bytes, metadata, fingerprint, upload,
                   consumers, args.warmups, args.repeats);
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace

int main(int argc, char **argv)
{
    try {
        CU_CHECK(cuInit(0));
        const Args args = parse_args(argc, argv);
        CUDA_CHECK(cudaSetDevice(args.device));
        CUDA_CHECK(cudaFree(nullptr));

        int vmm_supported = 0;
        CUdevice driver_device = 0;
        CU_CHECK(cuDeviceGet(&driver_device, args.device));
        CU_CHECK(cuDeviceGetAttribute(
            &vmm_supported,
            CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
            driver_device));
        if (vmm_supported == 0)
            throw Error("selected CUDA device does not support VMM");

        cudaDeviceProp device{};
        CUDA_CHECK(cudaGetDeviceProperties(&device, args.device));
        const uint64_t layer_bytes = checked_multiply(
            args.region_bytes, args.regions_per_layer, "layer bytes");
        const uint64_t transfer_bytes = checked_multiply(
            layer_bytes, args.layers, "logical transfer bytes");
        const AllocationInfo allocation =
            query_allocation_info(args.device, layer_bytes);

        {
            PinnedSources sources(args, layer_bytes);
            emit_config(args, device, allocation, layer_bytes, transfer_bytes,
                        sources.bytes());

            for (Mode mode : args.modes) {
                if (mode == Mode::GraphRegionDirect) {
                    run_direct_mode(mode, args, allocation, sources,
                                    transfer_bytes);
                }
                else {
                    run_alias_mode(mode, args, allocation, sources,
                                   transfer_bytes);
                }
            }
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaDeviceReset());
        return 0;
    }
    catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        std::cout << "{\"record\":\"error\",\"schema_version\":1,"
                     "\"message\":"
                  << json_quote(error.what()) << "}\n";
        return 1;
    }
}
