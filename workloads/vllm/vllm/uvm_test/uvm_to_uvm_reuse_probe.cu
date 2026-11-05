#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" int uvm_driver_batch_query_residency_bytes(
    const uintptr_t *ptrs,
    const size_t *sizes,
    size_t count,
    int device,
    uint64_t *gpu_bytes_out,
    uint64_t *cpu_bytes_out,
    uint64_t *any_bytes_out,
    uint64_t *unknown_bytes_out,
    uint64_t *queried_bytes_out);

namespace {

constexpr size_t GiB = 1024ULL * 1024ULL * 1024ULL;

struct Args {
    int device = 0;
    size_t hot_bytes = 2 * GiB;
    size_t kv_bytes = 2 * GiB;
    size_t expert_bytes = 2 * GiB;
};

struct Residency {
    int status = -1;
    uint64_t gpu_bytes = 0;
    uint64_t cpu_bytes = 0;
    uint64_t any_bytes = 0;
    uint64_t unknown_bytes = 0;
    uint64_t queried_bytes = 0;
};

struct MemoryInfo {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
};

void check(cudaError_t status, const char *operation)
{
    if (status == cudaSuccess)
        return;
    std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
    std::exit(1);
}

size_t parse_gib(const char *value)
{
    char *end = nullptr;
    const double gib = std::strtod(value, &end);
    if (end == value || *end != '\0' || gib <= 0.0) {
        std::fprintf(stderr, "invalid GiB value: %s\n", value);
        std::exit(2);
    }
    return static_cast<size_t>(gib * static_cast<double>(GiB));
}

Args parse_args(int argc, char **argv)
{
    Args args;
    for (int index = 1; index < argc; ++index) {
        if (std::strcmp(argv[index], "--device") == 0 && index + 1 < argc)
            args.device = std::atoi(argv[++index]);
        else if (std::strcmp(argv[index], "--hot-gib") == 0 && index + 1 < argc)
            args.hot_bytes = parse_gib(argv[++index]);
        else if (std::strcmp(argv[index], "--kv-gib") == 0 && index + 1 < argc)
            args.kv_bytes = parse_gib(argv[++index]);
        else if (std::strcmp(argv[index], "--expert-gib") == 0 && index + 1 < argc)
            args.expert_bytes = parse_gib(argv[++index]);
        else {
            std::fprintf(stderr,
                         "usage: %s [--device N] [--hot-gib N] [--kv-gib N] "
                         "[--expert-gib N]\n",
                         argv[0]);
            std::exit(2);
        }
    }
    return args;
}

MemoryInfo print_memory(const char *stage)
{
    MemoryInfo result;
    check(cudaMemGetInfo(&result.free_bytes, &result.total_bytes),
          "cudaMemGetInfo");
    std::printf("memory stage=%s free_bytes=%zu used_bytes=%zu total_bytes=%zu\n",
                stage,
                result.free_bytes,
                result.total_bytes - result.free_bytes,
                result.total_bytes);
    return result;
}

Residency query_residency(void *ptr, size_t bytes, int device)
{
    const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    Residency result;
    result.status = uvm_driver_batch_query_residency_bytes(
        &address,
        &bytes,
        1,
        device,
        &result.gpu_bytes,
        &result.cpu_bytes,
        &result.any_bytes,
        &result.unknown_bytes,
        &result.queried_bytes);
    return result;
}

Residency print_residency(const char *stage,
                          const char *name,
                          void *ptr,
                          size_t bytes,
                          int device)
{
    const Residency result = query_residency(ptr, bytes, device);
    std::printf(
        "residency stage=%s name=%s status=%d requested_bytes=%zu "
        "queried_bytes=%llu gpu_bytes=%llu cpu_bytes=%llu any_bytes=%llu "
        "unknown_bytes=%llu\n",
        stage,
        name,
        result.status,
        bytes,
        static_cast<unsigned long long>(result.queried_bytes),
        static_cast<unsigned long long>(result.gpu_bytes),
        static_cast<unsigned long long>(result.cpu_bytes),
        static_cast<unsigned long long>(result.any_bytes),
        static_cast<unsigned long long>(result.unknown_bytes));
    return result;
}

bool mostly_resident(uint64_t resident_bytes, size_t requested_bytes)
{
    const size_t tolerance = 2 * 1024 * 1024;
    return resident_bytes + tolerance >= requested_bytes;
}

template <typename Operation>
double time_operation(const char *name, size_t bytes, Operation operation)
{
    const auto begin = std::chrono::steady_clock::now();
    operation();
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - begin).count();
    std::printf("timing operation=%s bytes=%zu elapsed_ms=%.3f\n",
                name,
                bytes,
                elapsed_ms);
    return elapsed_ms;
}

}  // namespace

int main(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const Args args = parse_args(argc, argv);
    check(cudaSetDevice(args.device), "cudaSetDevice");

    std::printf(
        "probe_begin device=%d hot_bytes=%zu kv_bytes=%zu expert_bytes=%zu\n",
        args.device,
        args.hot_bytes,
        args.kv_bytes,
        args.expert_bytes);
    print_memory("context_ready");

    void *hot = nullptr;
    void *kv = nullptr;
    void *expert = nullptr;
    check(cudaMallocManaged(&hot, args.hot_bytes, cudaMemAttachGlobal),
          "cudaMallocManaged hot");
    check(cudaMallocManaged(&kv, args.kv_bytes, cudaMemAttachGlobal),
          "cudaMallocManaged kv");
    check(cudaMallocManaged(&expert, args.expert_bytes, cudaMemAttachGlobal),
          "cudaMallocManaged expert");

    time_operation("initial_placement", args.hot_bytes + args.kv_bytes,
                   [&]() {
                       check(cudaMemPrefetchAsync(
                                 expert, args.expert_bytes, cudaCpuDeviceId),
                             "prefetch expert to CPU");
                       check(cudaMemPrefetchAsync(hot, args.hot_bytes, args.device),
                             "prefetch hot to GPU");
                       check(cudaMemPrefetchAsync(kv, args.kv_bytes, args.device),
                             "prefetch KV to GPU");
                       check(cudaDeviceSynchronize(),
                             "initial residency synchronization");
                   });

    const MemoryInfo initial_memory = print_memory("hot_and_kv_gpu");
    const Residency initial_hot = print_residency(
        "hot_and_kv_gpu", "hot", hot, args.hot_bytes, args.device);
    const Residency initial_kv = print_residency(
        "hot_and_kv_gpu", "kv", kv, args.kv_bytes, args.device);
    const Residency initial_expert = print_residency(
        "hot_and_kv_gpu", "expert", expert, args.expert_bytes, args.device);

    time_operation("kv_gpu_to_cpu", args.kv_bytes, [&]() {
        check(cudaMemPrefetchAsync(kv, args.kv_bytes, cudaCpuDeviceId),
              "prefetch KV to CPU");
        check(cudaDeviceSynchronize(), "KV CPU synchronization");
    });

    const MemoryInfo offload_memory = print_memory("kv_cpu_before_expert_gpu");
    const Residency offload_hot = print_residency(
        "kv_cpu_before_expert_gpu", "hot", hot, args.hot_bytes, args.device);
    const Residency offload_kv = print_residency(
        "kv_cpu_before_expert_gpu", "kv", kv, args.kv_bytes, args.device);
    const Residency offload_expert = print_residency(
        "kv_cpu_before_expert_gpu", "expert", expert, args.expert_bytes, args.device);

    time_operation("expert_cpu_to_gpu", args.expert_bytes, [&]() {
        check(cudaMemPrefetchAsync(expert, args.expert_bytes, args.device),
              "prefetch expert to GPU");
        check(cudaDeviceSynchronize(), "expert GPU synchronization");
    });

    const MemoryInfo final_memory = print_memory("expert_gpu_after_kv_cpu");
    const Residency final_hot = print_residency(
        "expert_gpu_after_kv_cpu", "hot", hot, args.hot_bytes, args.device);
    const Residency final_kv = print_residency(
        "expert_gpu_after_kv_cpu", "kv", kv, args.kv_bytes, args.device);
    const Residency final_expert = print_residency(
        "expert_gpu_after_kv_cpu", "expert", expert, args.expert_bytes, args.device);

    const bool query_ok = initial_hot.status >= 0 && initial_kv.status >= 0 &&
                          initial_expert.status >= 0 && offload_hot.status >= 0 &&
                          offload_kv.status >= 0 && offload_expert.status >= 0 &&
                          final_hot.status >= 0 && final_kv.status >= 0 &&
                          final_expert.status >= 0;
    const bool initial_placement_ok =
        mostly_resident(initial_hot.gpu_bytes, args.hot_bytes) &&
        mostly_resident(initial_kv.gpu_bytes, args.kv_bytes) &&
        mostly_resident(initial_expert.cpu_bytes, args.expert_bytes);
    const bool hot_preserved = mostly_resident(final_hot.gpu_bytes, args.hot_bytes);
    const bool kv_offloaded = mostly_resident(final_kv.cpu_bytes, args.kv_bytes) &&
                              final_kv.gpu_bytes <= 2 * 1024 * 1024;
    const bool expert_promoted =
        mostly_resident(final_expert.gpu_bytes, args.expert_bytes);
    const uint64_t hot_evicted_bytes =
        initial_hot.gpu_bytes > final_hot.gpu_bytes
            ? initial_hot.gpu_bytes - final_hot.gpu_bytes
            : 0;
    const uint64_t kv_gpu_reclaimed_bytes =
        initial_kv.gpu_bytes > offload_kv.gpu_bytes
            ? initial_kv.gpu_bytes - offload_kv.gpu_bytes
            : 0;
    const uint64_t expert_gpu_growth_bytes =
        final_expert.gpu_bytes > offload_expert.gpu_bytes
            ? final_expert.gpu_bytes - offload_expert.gpu_bytes
            : 0;
    const uint64_t reported_free_gain_after_kv =
        offload_memory.free_bytes > initial_memory.free_bytes
            ? offload_memory.free_bytes - initial_memory.free_bytes
            : 0;
    const uint64_t nonreported_capacity_required_bytes =
        expert_gpu_growth_bytes > offload_memory.free_bytes
            ? expert_gpu_growth_bytes - offload_memory.free_bytes
            : 0;
    const bool expert_exceeds_reported_free =
        nonreported_capacity_required_bytes > 2 * 1024 * 1024;
    const bool reuse_without_hot_eviction =
        expert_exceeds_reported_free && hot_evicted_bytes <= 2 * 1024 * 1024 &&
        kv_offloaded && expert_promoted;
    const bool success = query_ok && initial_placement_ok && hot_preserved &&
                         kv_offloaded && expert_promoted;

    std::printf(
        "reuse_evidence initial_free_bytes=%zu offload_free_bytes=%zu "
        "final_free_bytes=%zu reported_free_gain_after_kv=%llu "
        "kv_gpu_reclaimed_bytes=%llu expert_gpu_growth_bytes=%llu "
        "nonreported_capacity_required_bytes=%llu hot_evicted_bytes=%llu "
        "expert_exceeds_reported_free=%u reuse_without_hot_eviction=%u\n",
        initial_memory.free_bytes,
        offload_memory.free_bytes,
        final_memory.free_bytes,
        static_cast<unsigned long long>(reported_free_gain_after_kv),
        static_cast<unsigned long long>(kv_gpu_reclaimed_bytes),
        static_cast<unsigned long long>(expert_gpu_growth_bytes),
        static_cast<unsigned long long>(nonreported_capacity_required_bytes),
        static_cast<unsigned long long>(hot_evicted_bytes),
        expert_exceeds_reported_free ? 1U : 0U,
        reuse_without_hot_eviction ? 1U : 0U);

    std::printf(
        "probe_result success=%u query_ok=%u initial_placement_ok=%u "
        "hot_preserved=%u kv_offloaded=%u expert_promoted=%u\n",
        success ? 1U : 0U,
        query_ok ? 1U : 0U,
        initial_placement_ok ? 1U : 0U,
        hot_preserved ? 1U : 0U,
        kv_offloaded ? 1U : 0U,
        expert_promoted ? 1U : 0U);

    check(cudaFree(expert), "cudaFree expert");
    check(cudaFree(kv), "cudaFree kv");
    check(cudaFree(hot), "cudaFree hot");
    check(cudaDeviceSynchronize(), "cleanup synchronization");
    print_memory("cleanup");
    return success ? 0 : 1;
}
