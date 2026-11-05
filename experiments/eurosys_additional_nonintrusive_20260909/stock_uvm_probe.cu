#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr size_t MiB = 1024ULL * 1024ULL;

__device__ __forceinline__ uint32_t expected(size_t index) {
    return 0x6A5A0000U ^ static_cast<uint32_t>(index);
}

__global__ void write_kernel(uint32_t *data, size_t words) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (; index < words; index += stride) data[index] = expected(index);
}

__global__ void verify_kernel(const uint32_t *data, size_t words,
                              unsigned long long *mismatches) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned long long local = 0;
    for (; index < words; index += stride) local += data[index] != expected(index);
    if (local) atomicAdd(mismatches, local);
}

double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start)
        .count();
}

void check(cudaError_t status, const char *stage) {
    if (status == cudaSuccess) return;
    std::fprintf(stderr, "stage=%s cuda_status=%d error=%s\n", stage,
                 static_cast<int>(status), cudaGetErrorString(status));
    std::exit(10);
}

}  // namespace

int main(int argc, char **argv) {
    int device = 0;
    size_t mib = 64;
    bool prefetch = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--mib") == 0 && i + 1 < argc) {
            mib = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "--prefetch-advice") == 0) {
            prefetch = true;
        } else {
            std::fprintf(stderr,
                         "usage: %s [--device N] [--mib N] [--prefetch-advice]\n",
                         argv[0]);
            return 2;
        }
    }
    if (mib == 0) return 2;
    check(cudaSetDevice(device), "set_device");
    const size_t bytes = mib * MiB;
    const size_t words = bytes / sizeof(uint32_t);
    uint32_t *data = nullptr;
    unsigned long long *mismatches_device = nullptr;
    auto allocation_start = std::chrono::steady_clock::now();
    check(cudaMallocManaged(&data, bytes, cudaMemAttachGlobal), "malloc_managed");
    const double allocation_ms = elapsed_ms(allocation_start);
    double advice_ms = 0.0;
    double prefetch_ms = 0.0;
    if (prefetch) {
        auto advice_start = std::chrono::steady_clock::now();
        check(cudaMemAdvise(data, bytes, cudaMemAdviseSetPreferredLocation, device),
              "advise_preferred_device");
        check(cudaMemAdvise(data, bytes, cudaMemAdviseSetAccessedBy, device),
              "advise_accessed_by_device");
        advice_ms = elapsed_ms(advice_start);
        auto prefetch_start = std::chrono::steady_clock::now();
        check(cudaMemPrefetchAsync(data, bytes, device, nullptr), "prefetch_device");
        check(cudaDeviceSynchronize(), "prefetch_sync");
        prefetch_ms = elapsed_ms(prefetch_start);
    }
    check(cudaMalloc(&mismatches_device, sizeof(*mismatches_device)), "malloc_counter");
    check(cudaMemset(mismatches_device, 0, sizeof(*mismatches_device)), "clear_counter");
    auto write_start = std::chrono::steady_clock::now();
    write_kernel<<<256, 256>>>(data, words);
    check(cudaGetLastError(), "write_launch");
    check(cudaDeviceSynchronize(), "write_sync");
    const double write_ms = elapsed_ms(write_start);
    auto read_start = std::chrono::steady_clock::now();
    verify_kernel<<<256, 256>>>(data, words, mismatches_device);
    check(cudaGetLastError(), "verify_launch");
    check(cudaDeviceSynchronize(), "verify_sync");
    const double read_ms = elapsed_ms(read_start);
    unsigned long long mismatches = ~0ULL;
    check(cudaMemcpy(&mismatches, mismatches_device, sizeof(mismatches),
                     cudaMemcpyDeviceToHost),
          "copy_counter");
    std::printf(
        "{\"status\":\"%s\",\"bytes\":%zu,\"prefetch_advice\":%s,"
        "\"allocation_ms\":%.6f,\"advice_ms\":%.6f,\"prefetch_ms\":%.6f,"
        "\"first_write_ms\":%.6f,\"read_verify_ms\":%.6f,"
        "\"mismatches\":%llu}\n",
        mismatches == 0 ? "ok" : "mismatch", bytes,
        prefetch ? "true" : "false", allocation_ms, advice_ms, prefetch_ms,
        write_ms, read_ms, mismatches);
    check(cudaFree(data), "free_managed");
    check(cudaFree(mismatches_device), "free_counter");
    return mismatches == 0 ? 0 : 11;
}
