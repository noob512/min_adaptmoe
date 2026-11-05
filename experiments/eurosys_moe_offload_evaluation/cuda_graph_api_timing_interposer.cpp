#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

namespace {

uint64_t monotonic_ns()
{
    struct timespec value {};
    clock_gettime(CLOCK_MONOTONIC, &value);
    return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL
        + static_cast<uint64_t>(value.tv_nsec);
}

void record(const char *name, uint64_t start_ns, int result)
{
    const char *path = getenv("E3_CUDA_API_TIMING_LOG");
    if (path == nullptr || path[0] == '\0')
        return;
    const uint64_t end_ns = monotonic_ns();
    const int fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (fd < 0)
        return;
    dprintf(fd,
            "%llu\t%llu\t%llu\t%d\t%d\t%s\n",
            static_cast<unsigned long long>(start_ns),
            static_cast<unsigned long long>(end_ns),
            static_cast<unsigned long long>(end_ns - start_ns),
            static_cast<int>(getpid()),
            result,
            name);
    close(fd);
}

template <typename Function>
Function next_symbol(const char *name)
{
    return reinterpret_cast<Function>(dlsym(RTLD_NEXT, name));
}

}  // namespace

extern "C" CUresult cuGraphLaunch(CUgraphExec graph_exec, CUstream stream)
{
    using Function = CUresult (*)(CUgraphExec, CUstream);
    static Function function = next_symbol<Function>("cuGraphLaunch");
    const uint64_t start_ns = monotonic_ns();
    const CUresult result = function(graph_exec, stream);
    record("cuGraphLaunch", start_ns, static_cast<int>(result));
    return result;
}

extern "C" CUresult cuGraphLaunch_ptsz(CUgraphExec graph_exec, CUstream stream)
{
    using Function = CUresult (*)(CUgraphExec, CUstream);
    static Function function = next_symbol<Function>("cuGraphLaunch_ptsz");
    const uint64_t start_ns = monotonic_ns();
    const CUresult result = function(graph_exec, stream);
    record("cuGraphLaunch_ptsz", start_ns, static_cast<int>(result));
    return result;
}

extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t graph_exec,
                                         cudaStream_t stream)
{
    using Function = cudaError_t (*)(cudaGraphExec_t, cudaStream_t);
    static Function function = next_symbol<Function>("cudaGraphLaunch");
    const uint64_t start_ns = monotonic_ns();
    const cudaError_t result = function(graph_exec, stream);
    record("cudaGraphLaunch", start_ns, static_cast<int>(result));
    return result;
}

extern "C" cudaError_t cudaDeviceSynchronize()
{
    using Function = cudaError_t (*)();
    static Function function = next_symbol<Function>("cudaDeviceSynchronize");
    const uint64_t start_ns = monotonic_ns();
    const cudaError_t result = function();
    record("cudaDeviceSynchronize", start_ns, static_cast<int>(result));
    return result;
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    using Function = cudaError_t (*)(cudaStream_t);
    static Function function = next_symbol<Function>("cudaStreamSynchronize");
    const uint64_t start_ns = monotonic_ns();
    const cudaError_t result = function(stream);
    record("cudaStreamSynchronize", start_ns, static_cast<int>(result));
    return result;
}

extern "C" CUresult cuCtxSynchronize()
{
    using Function = CUresult (*)();
    static Function function = next_symbol<Function>("cuCtxSynchronize");
    const uint64_t start_ns = monotonic_ns();
    const CUresult result = function();
    record("cuCtxSynchronize", start_ns, static_cast<int>(result));
    return result;
}

extern "C" CUresult cuStreamSynchronize(CUstream stream)
{
    using Function = CUresult (*)(CUstream);
    static Function function = next_symbol<Function>("cuStreamSynchronize");
    const uint64_t start_ns = monotonic_ns();
    const CUresult result = function(stream);
    record("cuStreamSynchronize", start_ns, static_cast<int>(result));
    return result;
}
