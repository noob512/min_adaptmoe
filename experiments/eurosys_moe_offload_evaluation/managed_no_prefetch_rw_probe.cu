#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr size_t MiB = 1024ULL * 1024ULL;

struct Args {
    int device = 0;
    size_t bytes = 64 * MiB;
};

void print_cuda_error(const char *stage, cudaError_t status)
{
    std::fprintf(stderr,
                 "stage=%s status=error cuda_status=%d cuda_error=%s\n",
                 stage,
                 static_cast<int>(status),
                 cudaGetErrorString(status));
    std::fflush(stderr);
}

bool check(cudaError_t status, const char *stage)
{
    if (status == cudaSuccess)
        return true;
    print_cuda_error(stage, status);
    return false;
}

Args parse_args(int argc, char **argv)
{
    Args args;
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--device") == 0 && index + 1 < argc) {
            args.device = std::atoi(argv[++index]);
        }
        else if (std::strcmp(argv[index], "--mib") == 0 && index + 1 < argc) {
            char *end = nullptr;
            const unsigned long long mib = std::strtoull(argv[++index], &end, 10);
            if (end == argv[index] || *end != '\0' || mib == 0) {
                std::fprintf(stderr, "invalid --mib value: %s\n", argv[index]);
                std::exit(2);
            }
            args.bytes = static_cast<size_t>(mib) * MiB;
        }
        else {
            std::fprintf(stderr,
                         "usage: %s [--device N] [--mib N]\n",
                         argv[0]);
            std::exit(2);
        }
    }
    if (args.bytes % sizeof(uint32_t) != 0) {
        std::fprintf(stderr, "allocation size must be uint32 aligned\n");
        std::exit(2);
    }
    return args;
}

int read_fault_mode()
{
    constexpr const char *path =
        "/sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode";
    FILE *handle = std::fopen(path, "r");
    if (!handle)
        return -1;
    int mode = -1;
    if (std::fscanf(handle, "%d", &mode) != 1)
        mode = -1;
    std::fclose(handle);
    return mode;
}

__device__ __forceinline__ uint32_t expected_value(size_t index)
{
    return 0xA5A50000U ^ static_cast<uint32_t>(index);
}

__global__ void write_kernel(uint32_t *data, size_t words)
{
    const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index = start; index < words; index += stride)
        data[index] = expected_value(index);
}

__global__ void read_verify_kernel(const uint32_t *data,
                                   size_t words,
                                   unsigned long long *mismatches)
{
    const size_t start = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned long long local = 0;
    for (size_t index = start; index < words; index += stride) {
        if (data[index] != expected_value(index))
            ++local;
    }
    if (local != 0)
        atomicAdd(mismatches, local);
}

}  // namespace

int main(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::setvbuf(stderr, nullptr, _IOLBF, 0);
    const Args args = parse_args(argc, argv);
    const int fault_mode = read_fault_mode();
    std::printf("stage=preflight fault_mode=%d prefetch_used=0 bytes=%zu\n",
                fault_mode,
                args.bytes);
    if (fault_mode != 2) {
        std::fprintf(stderr,
                     "stage=preflight status=error reason=fault_mode_not_2\n");
        return 2;
    }

    if (!check(cudaSetDevice(args.device), "cudaSetDevice"))
        return 3;

    size_t free_before = 0;
    size_t total = 0;
    if (!check(cudaMemGetInfo(&free_before, &total), "cudaMemGetInfo_before"))
        return 3;

    unsigned long long *device_mismatches = nullptr;
    if (!check(cudaMalloc(&device_mismatches, sizeof(*device_mismatches)),
               "cudaMalloc_verify_counter"))
        return 3;
    if (!check(cudaMemset(device_mismatches, 0, sizeof(*device_mismatches)),
               "cudaMemset_verify_counter"))
        return 3;

    uint32_t *data = nullptr;
    if (!check(cudaMallocManaged(&data, args.bytes, cudaMemAttachGlobal),
               "cudaMallocManaged"))
        return 4;

    size_t free_after_allocation = 0;
    if (!check(cudaMemGetInfo(&free_after_allocation, &total),
               "cudaMemGetInfo_after_allocation"))
        return 4;
    std::printf(
        "stage=allocated status=ok ptr=0x%llx bytes=%zu free_before=%zu "
        "free_after=%zu free_drop=%zu prefetch_used=0\n",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(data)),
        args.bytes,
        free_before,
        free_after_allocation,
        free_before > free_after_allocation ? free_before - free_after_allocation : 0);

    constexpr int threads = 256;
    constexpr int blocks = 256;
    const size_t words = args.bytes / sizeof(uint32_t);

    std::printf("stage=gpu_write_launch status=begin words=%zu\n", words);
    write_kernel<<<blocks, threads>>>(data, words);
    if (!check(cudaGetLastError(), "gpu_write_launch"))
        return 10;
    const cudaError_t write_sync = cudaDeviceSynchronize();
    if (!check(write_sync, "gpu_write_sync")) {
        std::printf("probe_result success=0 failed_stage=gpu_write_sync "
                    "read_attempted=0\n");
        return 10;
    }
    std::printf("stage=gpu_write_sync status=ok\n");

    std::printf("stage=gpu_read_launch status=begin words=%zu\n", words);
    read_verify_kernel<<<blocks, threads>>>(data, words, device_mismatches);
    if (!check(cudaGetLastError(), "gpu_read_launch"))
        return 11;
    const cudaError_t read_sync = cudaDeviceSynchronize();
    if (!check(read_sync, "gpu_read_sync")) {
        std::printf("probe_result success=0 failed_stage=gpu_read_sync "
                    "read_attempted=1\n");
        return 11;
    }

    unsigned long long mismatches = ~0ULL;
    if (!check(cudaMemcpy(&mismatches,
                          device_mismatches,
                          sizeof(mismatches),
                          cudaMemcpyDeviceToHost),
               "copy_verify_counter"))
        return 12;
    const bool success = mismatches == 0;
    std::printf("stage=gpu_read_sync status=ok mismatches=%llu\n", mismatches);
    std::printf("probe_result success=%u failed_stage=none read_attempted=1 "
                "mismatches=%llu\n",
                success ? 1U : 0U,
                mismatches);

    if (!check(cudaFree(data), "cudaFree_managed"))
        return 13;
    if (!check(cudaFree(device_mismatches), "cudaFree_verify_counter"))
        return 13;
    return success ? 0 : 14;
}
