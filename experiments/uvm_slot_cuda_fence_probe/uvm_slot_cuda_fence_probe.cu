// SPDX-License-Identifier: Apache-2.0
//
// Standalone proof that a stock UVM semaphore-pool allocation can be used by
// CUDA stream memory operations. This intentionally does not depend on any of
// the experimental MoE ioctls.

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

__global__ void stream_ordered_signal_kernel(uint32_t *address, uint32_t value)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        __threadfence_system();
        atomicExch_system(address, value);
    }
}

__global__ void stream_ordered_wait_and_signal_kernel(uint32_t *wait_address,
                                                       uint32_t wait_value,
                                                       uint32_t *signal_address,
                                                       uint32_t signal_value)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        while (atomicAdd_system(wait_address, 0) < wait_value)
            __nanosleep(128);
        __threadfence_system();
        atomicExch_system(signal_address, signal_value);
    }
}

constexpr unsigned long UVM_ALLOC_SEMAPHORE_POOL_IOCTL = 68;
constexpr unsigned long UVM_FREE_IOCTL = 34;
constexpr unsigned long UVM_INITIALIZE_IOCTL = 0x30000001UL;
constexpr size_t UVM_MAX_GPUS_LOCAL = 32 * 8;

struct NvProcessorUuidLocal {
    uint8_t uuid[16];
};

struct UvmGpuMappingAttributesLocal {
    NvProcessorUuidLocal gpu_uuid;
    uint32_t gpu_mapping_type;
    uint32_t gpu_caching_type;
    uint32_t gpu_format_type;
    uint32_t gpu_element_bits;
    uint32_t gpu_compression_type;
};

struct UvmAllocSemaphorePoolParamsLocal {
    uint64_t base;
    uint64_t length;
    UvmGpuMappingAttributesLocal per_gpu_attributes[UVM_MAX_GPUS_LOCAL];
    uint64_t gpu_attributes_count;
    uint32_t rm_status;
};

struct UvmFreeParamsLocal {
    uint64_t base;
    uint32_t rm_status;
};

struct UvmInitializeParamsLocal {
    uint64_t flags;
    uint32_t rm_status;
};

static_assert(sizeof(NvProcessorUuidLocal) == 16, "UUID ABI mismatch");
static_assert(sizeof(UvmGpuMappingAttributesLocal) == 36,
              "GPU mapping attribute ABI mismatch");
static_assert(offsetof(UvmAllocSemaphorePoolParamsLocal, gpu_attributes_count) ==
                  9232,
              "semaphore-pool ABI offset mismatch");
static_assert(offsetof(UvmAllocSemaphorePoolParamsLocal, rm_status) == 9240,
              "semaphore-pool rmStatus offset mismatch");
static_assert(sizeof(UvmAllocSemaphorePoolParamsLocal) == 9248,
              "semaphore-pool ABI size mismatch");
static_assert(sizeof(UvmFreeParamsLocal) == 16, "UVM free ABI mismatch");

std::string errno_string(const char *what)
{
    std::ostringstream out;
    out << what << ": errno=" << errno << " (" << std::strerror(errno) << ")";
    return out.str();
}

void check_cuda(cudaError_t status, const char *what)
{
    if (status == cudaSuccess)
        return;

    std::ostringstream out;
    out << what << ": " << cudaGetErrorName(status) << " ("
        << cudaGetErrorString(status) << ")";
    throw std::runtime_error(out.str());
}

void check_cu(CUresult status, const char *what)
{
    if (status == CUDA_SUCCESS)
        return;

    const char *name = nullptr;
    const char *description = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &description);
    std::ostringstream out;
    out << what << ": " << (name ? name : "unknown") << " ("
        << (description ? description : "no description") << ")";
    throw std::runtime_error(out.str());
}

bool is_uvm_fd_target(const std::string &target)
{
    if (target.find("nvidia-uvm-tools") != std::string::npos)
        return false;

    const size_t slash = target.find_last_of('/');
    const std::string basename =
        slash == std::string::npos ? target : target.substr(slash + 1);
    return basename == "nvidia-uvm" || basename.rfind("nvidia-uvm ", 0) == 0;
}

int duplicate_current_uvm_fd()
{
    DIR *dir = opendir("/proc/self/fd");
    if (!dir)
        throw std::runtime_error(errno_string("opendir(/proc/self/fd)"));

    int duplicated = -1;
    while (dirent *entry = readdir(dir)) {
        char *end = nullptr;
        const long candidate_long = std::strtol(entry->d_name, &end, 10);
        if (!end || *end != '\0' || candidate_long < 0 ||
            candidate_long == dirfd(dir)) {
            continue;
        }

        const int candidate = static_cast<int>(candidate_long);
        const std::string fd_path = std::string("/proc/self/fd/") + entry->d_name;
        char target[512];
        const ssize_t length = readlink(fd_path.c_str(), target, sizeof(target) - 1);
        if (length < 0)
            continue;
        target[length] = '\0';
        if (!is_uvm_fd_target(target))
            continue;

        const int probe_fd = fcntl(candidate, F_DUPFD_CLOEXEC, 3);
        if (probe_fd < 0)
            continue;

        UvmInitializeParamsLocal initialize = {};
        const int initialize_ret =
            ioctl(probe_fd, UVM_INITIALIZE_IOCTL, &initialize);
        if (initialize_ret != 0 || initialize.rm_status != 0) {
            close(probe_fd);
            continue;
        }

        duplicated = probe_fd;
        break;
    }

    closedir(dir);
    if (duplicated < 0) {
        throw std::runtime_error(
            "no initialized CUDA /dev/nvidia-uvm fd was found");
    }
    return duplicated;
}

uint32_t load_cell(uint32_t *cell)
{
    return __atomic_load_n(cell, __ATOMIC_ACQUIRE);
}

void store_cell(uint32_t *cell, uint32_t value)
{
    __atomic_store_n(cell, value, __ATOMIC_RELEASE);
}

class ProbeResources {
  public:
    ~ProbeResources()
    {
        cleanup();
    }

    void cleanup()
    {
        if (producer_) {
            cudaStreamSynchronize(producer_);
            cudaStreamDestroy(producer_);
            producer_ = nullptr;
        }
        if (consumer_) {
            cudaStreamSynchronize(consumer_);
            cudaStreamDestroy(consumer_);
            consumer_ = nullptr;
        }

        if (mapping_ != MAP_FAILED) {
            if (host_registered_) {
                const CUresult status = cuMemHostUnregister(mapping_);
                if (status != CUDA_SUCCESS) {
                    const char *name = nullptr;
                    cuGetErrorName(status, &name);
                    std::cerr << "warning: cuMemHostUnregister cleanup failed: "
                              << (name ? name : "unknown") << '\n';
                }
                host_registered_ = false;
            }
            munmap(mapping_, mapping_bytes_);
            mapping_ = MAP_FAILED;
        }

        if (pool_allocated_ && uvm_fd_ >= 0) {
            UvmFreeParamsLocal params = {};
            params.base = pool_base_;
            const int ret = ioctl(uvm_fd_, UVM_FREE_IOCTL, &params);
            if (ret != 0 || params.rm_status != 0) {
                std::cerr << "warning: UVM_FREE cleanup failed: ret=" << ret
                          << " errno=" << errno
                          << " rmStatus=" << params.rm_status << '\n';
            }
            pool_allocated_ = false;
        }

        if (uvm_fd_ >= 0) {
            close(uvm_fd_);
            uvm_fd_ = -1;
        }
        if (managed_dummy_) {
            cudaFree(managed_dummy_);
            managed_dummy_ = nullptr;
        }
    }

    void *managed_dummy_ = nullptr;
    int uvm_fd_ = -1;
    void *mapping_ = MAP_FAILED;
    size_t mapping_bytes_ = 0;
    uint64_t pool_base_ = 0;
    bool pool_allocated_ = false;
    bool host_registered_ = false;
    cudaStream_t producer_ = nullptr;
    cudaStream_t consumer_ = nullptr;
};

void require_value(const char *name, uint32_t actual, uint32_t expected)
{
    if (actual == expected)
        return;

    std::ostringstream out;
    out << name << " mismatch: expected=0x" << std::hex << expected
        << " actual=0x" << actual;
    throw std::runtime_error(out.str());
}

}  // namespace

int main(int argc, char **argv)
{
    ProbeResources resources;
    try {
        bool raw_kernel_mode = false;
        if (argc == 2 && std::strcmp(argv[1], "--raw-kernel") == 0) {
            raw_kernel_mode = true;
        }
        else if (argc != 1) {
            throw std::runtime_error(
                "usage: uvm_slot_cuda_fence_probe [--raw-kernel]");
        }

        int device = 0;
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");

        // Match the vLLM/PyTorch serving process: create a primary CUDA
        // context and ordinary device allocation, then adopt the first UVM fd
        // for which repeated UVM_INITIALIZE succeeds. Creating a managed
        // allocation here changes the runtime's UVM setup and is unnecessary.
        check_cuda(cudaMalloc(&resources.managed_dummy_, 4096),
                   "cudaMalloc(context bootstrap)");
        resources.uvm_fd_ = duplicate_current_uvm_fd();

        CUdevice cu_device = 0;
        check_cu(cuDeviceGet(&cu_device, device), "cuDeviceGet");
        CUcontext current_context = nullptr;
        check_cu(cuCtxGetCurrent(&current_context), "cuCtxGetCurrent");
        if (!current_context)
            throw std::runtime_error("CUDA primary context is not current");
        int stream_mem_ops = 0;
        check_cu(cuDeviceGetAttribute(
                     &stream_mem_ops,
                     CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1,
                     cu_device),
                 "query stream memory operations support");
        // Attribute 92 describes the deprecated v1 batch-mem-op ABI. Newer
        // drivers can report 0 here while accepting the v2 WriteValue/WaitValue
        // entry points, so the calls below are the authoritative capability
        // check. Keep the attribute only as diagnostic output.

        const long page_size = sysconf(_SC_PAGESIZE);
        if (page_size <= 0)
            throw std::runtime_error("sysconf(_SC_PAGESIZE) failed");
        resources.mapping_bytes_ = static_cast<size_t>(page_size);

        void *reservation = mmap(nullptr,
                                 resources.mapping_bytes_,
                                 PROT_NONE,
                                 MAP_PRIVATE | MAP_ANONYMOUS,
                                 -1,
                                 0);
        if (reservation == MAP_FAILED)
            throw std::runtime_error(errno_string("reserve semaphore-pool VA"));
        resources.mapping_ = reservation;
        resources.pool_base_ = reinterpret_cast<uint64_t>(reservation);

        UvmAllocSemaphorePoolParamsLocal alloc_params = {};
        alloc_params.base = resources.pool_base_;
        alloc_params.length = resources.mapping_bytes_;
        alloc_params.gpu_attributes_count = 0;
        errno = 0;
        const int alloc_ret = ioctl(resources.uvm_fd_,
                                    UVM_ALLOC_SEMAPHORE_POOL_IOCTL,
                                    &alloc_params);
        if (alloc_ret != 0 || alloc_params.rm_status != 0) {
            std::ostringstream out;
            out << "UVM_ALLOC_SEMAPHORE_POOL failed: ret=" << alloc_ret
                << " errno=" << errno
                << " rmStatus=" << alloc_params.rm_status;
            throw std::runtime_error(out.str());
        }
        resources.pool_allocated_ = true;

        void *mapped = mmap(reservation,
                            resources.mapping_bytes_,
                            PROT_READ | PROT_WRITE,
                            MAP_SHARED | MAP_FIXED,
                            resources.uvm_fd_,
                            static_cast<off_t>(resources.pool_base_));
        if (mapped == MAP_FAILED)
            throw std::runtime_error(errno_string("MAP_FIXED semaphore pool"));
        if (mapped != reservation)
            throw std::runtime_error("semaphore-pool mmap returned the wrong VA");
        resources.mapping_ = mapped;

        if (!raw_kernel_mode) {
            check_cu(cuMemHostRegister(mapped,
                                       resources.mapping_bytes_,
                                       CU_MEMHOSTREGISTER_DEVICEMAP),
                     "cuMemHostRegister_v2(semaphore pool)");
            resources.host_registered_ = true;
        }

        auto *cells = static_cast<uint32_t *>(mapped);
        for (size_t index = 0; index < 8; ++index)
            store_cell(&cells[index], 0);

        check_cuda(cudaStreamCreateWithFlags(&resources.producer_,
                                             cudaStreamNonBlocking),
                   "create producer stream");
        check_cuda(cudaStreamCreateWithFlags(&resources.consumer_,
                                             cudaStreamNonBlocking),
                   "create consumer stream");

        const CUstream producer = reinterpret_cast<CUstream>(resources.producer_);
        const CUstream consumer = reinterpret_cast<CUstream>(resources.consumer_);
        CUdeviceptr gpu_base = reinterpret_cast<CUdeviceptr>(mapped);
        if (!raw_kernel_mode) {
            check_cu(cuMemHostGetDevicePointer(&gpu_base, mapped, 0),
                     "cuMemHostGetDevicePointer_v2(semaphore pool)");
        }

        if (raw_kernel_mode) {
            // Do not ask libcuda to register or classify the pointer. The UVM
            // semaphore pool is already mapped at this VA in the adopted UVM
            // VA space; a kernel launch passes the UVA value through without
            // the StreamMemOp allocation-metadata validation.
            constexpr uint32_t raw_gpu_done = 0x4101;
            constexpr uint32_t raw_gpu_ack = 0x4102;
            stream_ordered_signal_kernel<<<1, 1, 0, resources.producer_>>>(
                &cells[0], raw_gpu_done);
            check_cuda(cudaGetLastError(), "launch raw-UVA signal kernel");
            stream_ordered_wait_and_signal_kernel<<<1, 1, 0,
                                                    resources.consumer_>>>(
                &cells[0], raw_gpu_done, &cells[1], raw_gpu_ack);
            check_cuda(cudaGetLastError(), "launch raw-UVA wait kernel");
            check_cuda(cudaStreamSynchronize(resources.consumer_),
                       "synchronize raw-UVA GPU-to-GPU test");
            require_value("raw GPU done CPU visibility",
                          load_cell(&cells[0]), raw_gpu_done);
            require_value("raw GPU acknowledgement CPU visibility",
                          load_cell(&cells[1]), raw_gpu_ack);

            constexpr uint32_t raw_cpu_release = 0x4201;
            constexpr uint32_t raw_cpu_ack = 0x4202;
            store_cell(&cells[2], 0);
            store_cell(&cells[3], 0);
            stream_ordered_wait_and_signal_kernel<<<1, 1, 0,
                                                    resources.consumer_>>>(
                &cells[2], raw_cpu_release, &cells[3], raw_cpu_ack);
            check_cuda(cudaGetLastError(), "launch raw-UVA CPU wait kernel");
            const cudaError_t before_release =
                cudaStreamQuery(resources.consumer_);
            if (before_release != cudaErrorNotReady) {
                check_cuda(before_release,
                           "raw consumer unexpectedly completed before release");
                throw std::runtime_error(
                    "raw-UVA GPU wait did not remain pending before CPU release");
            }
            cudaGetLastError();
            store_cell(&cells[2], raw_cpu_release);
            check_cuda(cudaStreamSynchronize(resources.consumer_),
                       "synchronize raw-UVA CPU-to-GPU test");
            require_value("raw CPU release",
                          load_cell(&cells[2]), raw_cpu_release);
            require_value("raw CPU acknowledgement",
                          load_cell(&cells[3]), raw_cpu_ack);

            std::cout << "{\n"
                      << "  \"status\": \"pass\",\n"
                      << "  \"mode\": \"raw_kernel\",\n"
                      << "  \"device\": " << device << ",\n"
                      << "  \"uvm_fd\": " << resources.uvm_fd_ << ",\n"
                      << "  \"semaphore_pool_va\": \"0x" << std::hex
                      << resources.pool_base_ << std::dec << "\",\n"
                      << "  \"cuda_host_registered\": false,\n"
                      << "  \"gpu_write_cpu_visible\": true,\n"
                      << "  \"cpu_release_gpu_wait\": true\n"
                      << "}\n";
            resources.cleanup();
            return 0;
        }

        // GPU producer -> GPU consumer, with CPU observing the same uncached
        // semaphore page after the consumer completes.
        constexpr uint32_t gpu_done = 0x1001;
        constexpr uint32_t gpu_ack = 0x1002;
        const CUresult first_pool_write =
            cuStreamWriteValue32(producer,
                                 gpu_base + 0,
                                 gpu_done,
                                 CU_STREAM_WRITE_VALUE_DEFAULT);
        if (first_pool_write != CUDA_SUCCESS) {
            // Distinguish an address-registration problem from a device-wide
            // lack of stream-mem-op support with CUDA's canonical mapped-host
            // allocation path.
            void *control_host = nullptr;
            void *control_device = nullptr;
            CUresult control_write = CUDA_ERROR_UNKNOWN;
            cudaError_t host_alloc_status =
                cudaHostAlloc(&control_host, 4096, cudaHostAllocMapped);
            cudaError_t host_pointer_status = cudaSuccess;
            if (host_alloc_status == cudaSuccess) {
                host_pointer_status =
                    cudaHostGetDevicePointer(&control_device, control_host, 0);
                if (host_pointer_status == cudaSuccess) {
                    control_write = cuStreamWriteValue32(
                        producer,
                        reinterpret_cast<CUdeviceptr>(control_device),
                        gpu_done,
                        CU_STREAM_WRITE_VALUE_DEFAULT);
                    if (control_write == CUDA_SUCCESS)
                        cudaStreamSynchronize(resources.producer_);
                }
                cudaFreeHost(control_host);
            }

            // Also determine whether the pool is genuinely mapped in the
            // CUDA user VA space. Kernel launches pass raw UVA values and do
            // not require libcuda pointer-allocation bookkeeping.
            store_cell(&cells[0], 0);
            store_cell(&cells[1], 0);
            stream_ordered_signal_kernel<<<1, 1, 0, resources.producer_>>>(
                &cells[0], gpu_done);
            cudaError_t kernel_signal_status = cudaGetLastError();
            cudaError_t kernel_wait_status = cudaSuccess;
            cudaError_t kernel_sync_status = cudaSuccess;
            if (kernel_signal_status == cudaSuccess) {
                stream_ordered_wait_and_signal_kernel<<<1, 1, 0,
                                                        resources.consumer_>>>(
                    &cells[0], gpu_done, &cells[1], gpu_ack);
                kernel_wait_status = cudaGetLastError();
                if (kernel_wait_status == cudaSuccess)
                    kernel_sync_status = cudaStreamSynchronize(resources.consumer_);
            }
            const bool kernel_fallback_ok =
                kernel_signal_status == cudaSuccess &&
                kernel_wait_status == cudaSuccess &&
                kernel_sync_status == cudaSuccess &&
                load_cell(&cells[0]) == gpu_done &&
                load_cell(&cells[1]) == gpu_ack;

            const char *pool_name = nullptr;
            const char *control_name = nullptr;
            cuGetErrorName(first_pool_write, &pool_name);
            cuGetErrorName(control_write, &control_name);
            std::ostringstream out;
            out << "semaphore-pool stream write rejected: pool="
                << (pool_name ? pool_name : "unknown")
                << "; mapped-host control="
                << (control_name ? control_name : "not-run")
                << "; cudaHostAlloc=" << cudaGetErrorName(host_alloc_status)
                << "; cudaHostGetDevicePointer="
                << cudaGetErrorName(host_pointer_status)
                << "; raw-UVA kernel fallback="
                << (kernel_fallback_ok ? "success" : "failure")
                << "(signal=" << cudaGetErrorName(kernel_signal_status)
                << ",wait=" << cudaGetErrorName(kernel_wait_status)
                << ",sync=" << cudaGetErrorName(kernel_sync_status) << ")"
                << "; stream_mem_ops_v1_attribute=" << stream_mem_ops;
            throw std::runtime_error(out.str());
        }
        check_cu(cuStreamWaitValue32(consumer,
                                     gpu_base + 0,
                                     gpu_done,
                                     CU_STREAM_WAIT_VALUE_GEQ),
                 "consumer wait compute-done");
        check_cu(cuStreamWriteValue32(consumer,
                                      gpu_base + 4,
                                      gpu_ack,
                                      CU_STREAM_WRITE_VALUE_DEFAULT),
                 "consumer write acknowledgement");
        check_cuda(cudaStreamSynchronize(resources.consumer_),
                   "synchronize GPU-to-GPU fence test");
        require_value("GPU done CPU visibility", load_cell(&cells[0]), gpu_done);
        require_value("GPU acknowledgement CPU visibility",
                      load_cell(&cells[1]),
                      gpu_ack);

        // CPU release -> GPU stream wait -> GPU acknowledgement. Query first
        // to prove that the wait is actually pending while the payload is 0.
        constexpr uint32_t cpu_release = 0x2001;
        constexpr uint32_t cpu_ack = 0x2002;
        check_cu(cuStreamWaitValue32(consumer,
                                     gpu_base + 8,
                                     cpu_release,
                                     CU_STREAM_WAIT_VALUE_GEQ),
                 "consumer wait CPU completion");
        check_cu(cuStreamWriteValue32(consumer,
                                      gpu_base + 12,
                                      cpu_ack,
                                      CU_STREAM_WRITE_VALUE_DEFAULT),
                 "consumer write CPU-completion acknowledgement");
        const cudaError_t before_release = cudaStreamQuery(resources.consumer_);
        if (before_release != cudaErrorNotReady) {
            check_cuda(before_release, "consumer stream unexpectedly completed before CPU release");
            throw std::runtime_error("GPU wait did not remain pending before CPU release");
        }
        // Clear the runtime's expected NotReady status before later checks.
        cudaGetLastError();
        store_cell(&cells[2], cpu_release);
        check_cuda(cudaStreamSynchronize(resources.consumer_),
                   "synchronize CPU-to-GPU fence test");
        require_value("CPU release", load_cell(&cells[2]), cpu_release);
        require_value("CPU release GPU acknowledgement",
                      load_cell(&cells[3]),
                      cpu_ack);

        // Finally prove that a CPU can observe a stream-ordered GPU write
        // without first calling cudaStreamSynchronize. This is the userspace
        // analogue of the kernel waiting on a compute-done payload.
        constexpr uint32_t polled_done = 0x3001;
        store_cell(&cells[4], 0);
        const auto poll_start = std::chrono::steady_clock::now();
        check_cu(cuStreamWriteValue32(producer,
                                      gpu_base + 16,
                                      polled_done,
                                      CU_STREAM_WRITE_VALUE_DEFAULT),
                 "producer write CPU-polled payload");
        const auto deadline = poll_start + std::chrono::seconds(2);
        while (load_cell(&cells[4]) != polled_done) {
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("CPU timed out polling GPU stream write");
            std::this_thread::yield();
        }
        const auto poll_end = std::chrono::steady_clock::now();
        check_cuda(cudaStreamSynchronize(resources.producer_),
                   "synchronize CPU-poll test producer");

        const auto poll_us =
            std::chrono::duration_cast<std::chrono::microseconds>(poll_end - poll_start)
                .count();
        std::cout << "{\n"
                  << "  \"status\": \"pass\",\n"
                  << "  \"device\": " << device << ",\n"
                  << "  \"stream_mem_ops_v1_attribute\": "
                  << (stream_mem_ops ? "true" : "false") << ",\n"
                  << "  \"stream_mem_ops_v2_calls\": true,\n"
                  << "  \"uvm_fd\": " << resources.uvm_fd_ << ",\n"
                  << "  \"semaphore_pool_va\": \"0x" << std::hex
                  << resources.pool_base_ << std::dec << "\",\n"
                  << "  \"gpu_to_gpu_wait\": true,\n"
                  << "  \"gpu_write_cpu_visible\": true,\n"
                  << "  \"cpu_release_gpu_wait\": true,\n"
                  << "  \"gpu_write_cpu_poll_us\": " << poll_us << "\n"
                  << "}\n";

        resources.cleanup();
        return 0;
    }
    catch (const std::exception &error) {
        std::cerr << "probe failed: " << error.what() << '\n';
        return 1;
    }
}
