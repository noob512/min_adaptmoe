#include <cuda_runtime.h>

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

constexpr size_t MiB = 1024ULL * 1024ULL;
constexpr size_t GiB = 1024ULL * MiB;

struct Args {
    int device = 0;
    size_t hot_bytes = 20 * GiB;
    size_t expert_bytes = 8 * GiB;
    size_t native_bytes = 6 * GiB;
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
        else if (std::strcmp(argv[index], "--expert-gib") == 0 && index + 1 < argc)
            args.expert_bytes = parse_gib(argv[++index]);
        else if (std::strcmp(argv[index], "--native-gib") == 0 && index + 1 < argc)
            args.native_bytes = parse_gib(argv[++index]);
        else {
            std::fprintf(stderr,
                         "usage: %s [--device N] [--hot-gib N] "
                         "[--expert-gib N] [--native-gib N]\n",
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
    return resident_bytes + 2 * MiB >= requested_bytes;
}

}  // namespace

int main(int argc, char **argv)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const Args args = parse_args(argc, argv);
    check(cudaSetDevice(args.device), "cudaSetDevice");

    std::printf(
        "probe_begin device=%d hot_bytes=%zu expert_bytes=%zu native_bytes=%zu\n",
        args.device,
        args.hot_bytes,
        args.expert_bytes,
        args.native_bytes);
    print_memory("context_ready");

    void *hot = nullptr;
    void *expert = nullptr;
    void *native = nullptr;
    check(cudaMallocManaged(&hot, args.hot_bytes, cudaMemAttachGlobal),
          "cudaMallocManaged hot");
    check(cudaMallocManaged(&expert, args.expert_bytes, cudaMemAttachGlobal),
          "cudaMallocManaged expert");
    check(cudaMemPrefetchAsync(hot, args.hot_bytes, args.device),
          "prefetch hot to GPU");
    check(cudaMemPrefetchAsync(expert, args.expert_bytes, args.device),
          "prefetch expert to GPU");
    check(cudaDeviceSynchronize(), "initial GPU placement");

    const MemoryInfo before_offload = print_memory("before_expert_offload");
    const Residency hot_before = print_residency(
        "before_expert_offload", "hot", hot, args.hot_bytes, args.device);
    const Residency expert_before = print_residency(
        "before_expert_offload", "expert", expert, args.expert_bytes, args.device);

    check(cudaMemPrefetchAsync(expert, args.expert_bytes, cudaCpuDeviceId),
          "offload expert to CPU");
    check(cudaDeviceSynchronize(), "expert CPU placement");

    const MemoryInfo after_offload = print_memory("after_expert_offload");
    const Residency hot_after_offload = print_residency(
        "after_expert_offload", "hot", hot, args.hot_bytes, args.device);
    const Residency expert_after_offload = print_residency(
        "after_expert_offload", "expert", expert, args.expert_bytes, args.device);

    const cudaError_t native_status = cudaMalloc(&native, args.native_bytes);
    if (native_status != cudaSuccess) {
        std::printf("native_allocation status=%d error=%s\n",
                    static_cast<int>(native_status),
                    cudaGetErrorString(native_status));
        return 1;
    }
    cudaPointerAttributes attributes{};
    check(cudaPointerGetAttributes(&attributes, native),
          "cudaPointerGetAttributes native");
    check(cudaMemset(native, 0x5a, args.native_bytes), "touch native allocation");
    check(cudaDeviceSynchronize(), "native allocation synchronization");

    const MemoryInfo during_native = print_memory("during_native_allocation");
    const Residency hot_during = print_residency(
        "during_native_allocation", "hot", hot, args.hot_bytes, args.device);
    const Residency expert_during = print_residency(
        "during_native_allocation", "expert", expert, args.expert_bytes, args.device);

    const uint64_t offload_free_gain =
        after_offload.free_bytes > before_offload.free_bytes
            ? after_offload.free_bytes - before_offload.free_bytes
            : 0;
    const uint64_t native_free_drop =
        after_offload.free_bytes > during_native.free_bytes
            ? after_offload.free_bytes - during_native.free_bytes
            : 0;
    const uint64_t hot_evicted =
        hot_after_offload.gpu_bytes > hot_during.gpu_bytes
            ? hot_after_offload.gpu_bytes - hot_during.gpu_bytes
            : 0;
    const bool query_ok = hot_before.status >= 0 && expert_before.status >= 0 &&
                          hot_after_offload.status >= 0 &&
                          expert_after_offload.status >= 0 && hot_during.status >= 0 &&
                          expert_during.status >= 0;
    const bool initial_gpu_ok = mostly_resident(hot_before.gpu_bytes, args.hot_bytes) &&
                                mostly_resident(expert_before.gpu_bytes,
                                                args.expert_bytes);
    const bool offload_ok =
        mostly_resident(hot_after_offload.gpu_bytes, args.hot_bytes) &&
        mostly_resident(expert_after_offload.cpu_bytes, args.expert_bytes) &&
        expert_after_offload.gpu_bytes <= 2 * MiB;
    const bool native_is_device = attributes.type == cudaMemoryTypeDevice;
    const bool released_space_required = args.native_bytes > before_offload.free_bytes;
    const bool free_accounting_ok =
        offload_free_gain + 64 * MiB >= args.expert_bytes &&
        native_free_drop + 64 * MiB >= args.native_bytes;
    const bool post_native_ok =
        mostly_resident(hot_during.gpu_bytes, args.hot_bytes) &&
        mostly_resident(expert_during.cpu_bytes, args.expert_bytes) &&
        expert_during.gpu_bytes <= 2 * MiB && hot_evicted <= 2 * MiB;
    const bool success = query_ok && initial_gpu_ok && offload_ok &&
                         native_is_device && released_space_required &&
                         free_accounting_ok && post_native_ok;

    std::printf(
        "reuse_evidence initial_free_bytes=%zu offload_free_bytes=%zu "
        "native_free_bytes=%zu offload_free_gain=%llu native_free_drop=%llu "
        "native_exceeds_pre_offload_free=%u native_pointer_type=%d "
        "native_is_device=%u hot_evicted_bytes=%llu\n",
        before_offload.free_bytes,
        after_offload.free_bytes,
        during_native.free_bytes,
        static_cast<unsigned long long>(offload_free_gain),
        static_cast<unsigned long long>(native_free_drop),
        released_space_required ? 1U : 0U,
        static_cast<int>(attributes.type),
        native_is_device ? 1U : 0U,
        static_cast<unsigned long long>(hot_evicted));
    std::printf(
        "probe_result success=%u query_ok=%u initial_gpu_ok=%u offload_ok=%u "
        "free_accounting_ok=%u post_native_ok=%u\n",
        success ? 1U : 0U,
        query_ok ? 1U : 0U,
        initial_gpu_ok ? 1U : 0U,
        offload_ok ? 1U : 0U,
        free_accounting_ok ? 1U : 0U,
        post_native_ok ? 1U : 0U);

    check(cudaFree(native), "cudaFree native");
    check(cudaFree(expert), "cudaFree expert");
    check(cudaFree(hot), "cudaFree hot");
    check(cudaDeviceSynchronize(), "cleanup synchronization");
    print_memory("cleanup");
    return success ? 0 : 1;
}
