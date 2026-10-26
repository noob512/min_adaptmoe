/*
 * CUDA UVM Allocator for vLLM
 *
 * This is a custom CUDA allocator that uses cudaMallocManaged
 * to enable Unified Virtual Memory (UVM) in vLLM.
 *
 * UVM allows memory oversubscription - allocating more GPU memory
 * than physically available by using CPU memory as backing store.
 *
 * Usage:
 *   Set environment variable VLLM_USE_UVM=1 before starting vLLM
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <ctime>
#include <chrono>
#include <mutex>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <cstdarg>
#include <cerrno>
#include <dirent.h>
#include <limits.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

constexpr unsigned int UVM_INITIALIZE_IOCTL = 0x30000001;
constexpr unsigned int UVM_TEST_PHASE_F_PING_IOCTL =
    200 + 115;
constexpr unsigned int UVM_TEST_PHASE_F_BATCH_MINI_IOCTL =
    200 + 116;
constexpr unsigned int UVM_TEST_PHASE_F_BATCH_EXECUTOR_IOCTL =
    200 + 117;
constexpr unsigned int UVM_PHASE_F_BATCH_EXECUTOR_IOCTL = 81;
constexpr unsigned int UVM_PHASE_F_REGISTER_REGION_IOCTL = 82;
constexpr unsigned int UVM_PHASE_F_UNREGISTER_REGION_IOCTL = 83;
constexpr unsigned int UVM_UPDATE_REGION_STATE_IOCTL = 87;
constexpr unsigned int UVM_QUERY_REGION_IOCTL = 88;
constexpr unsigned int UVM_QUERY_REGION_BY_VA_IOCTL = 89;
constexpr size_t UVM_TEST_PHASE_F_MAX_RANGES = 64;
constexpr uint32_t UVM_TEST_PHASE_F_PING_MAGIC = 0x50464e47;  // "PFNG"

constexpr uint32_t UVM_TEST_PHASE_F_ACTION_PREFETCH_CPU = 1;
constexpr uint32_t UVM_TEST_PHASE_F_ACTION_PREFETCH_GPU = 2;
constexpr uint32_t UVM_TEST_PHASE_F_ACTION_ADVISE_CPU = 3;
constexpr uint32_t UVM_TEST_PHASE_F_ACTION_QUERY_RESIDENCY = 4;
constexpr uint32_t UVM_TEST_PHASE_F_ACTION_PREPARE_REMOTE_READ = 5;
constexpr uint32_t UVM_TEST_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG = 6;
constexpr uint32_t UVM_TEST_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG_DEFERRED = 7;
constexpr uint32_t UVM_TEST_PHASE_F_ACTION_START_BG_PROMOTION = 8;
constexpr uint32_t UVM_TEST_PHASE_F_ACTION_GPU_ONLY_READ_DUP_SOFT = 14;
constexpr uint32_t UVM_TEST_PHASE_F_ACTION_PAUSE_BG_PROMOTION = 16;
constexpr uint32_t UVM_TEST_PHASE_F_ACTION_RESUME_BG_PROMOTION = 17;
constexpr uint32_t UVM_TEST_PHASE_F_ACTION_EXPERT_READ_DUP_FAST = 18;
constexpr uint32_t UVM_TEST_PHASE_F_ACTION_EXPERT_READ_DUP_FAST_NATIVE_REGIONS = 19;
struct PhaseFUuid {
    uint8_t uuid[16];
};

struct UvmInitializeParams {
    uint64_t flags;
    uint32_t rmStatus;
};

struct PhaseFRange {
    uint64_t base;
    uint64_t length;
};

struct PhaseFRegion {
    uint64_t region_id;
    uint64_t base;
    uint64_t length;
};

struct PhaseFRangeResult {
    uint32_t status;
    uint32_t resident_on_count;
    uint8_t resident_on_cpu;
    uint8_t resident_on_gpu;
    uint64_t queried_bytes;
    uint64_t resident_gpu_bytes;
    uint64_t resident_cpu_bytes;
    uint64_t resident_any_bytes;
    uint64_t resident_unknown_bytes;
};

struct PhaseFBatchExecutorParams {
    uint32_t action;
    uint32_t count;
    PhaseFUuid gpu_uuid;
    uint8_t async;
    uint32_t success_count;
    uint32_t failure_count;
    PhaseFRange ranges[UVM_TEST_PHASE_F_MAX_RANGES];
    PhaseFRangeResult results[UVM_TEST_PHASE_F_MAX_RANGES];
    uint32_t rmStatus;
};

struct PhaseFRegisterRegionParams {
    PhaseFRegion region;
    uint32_t rmStatus;
};

struct PhaseFUnregisterRegionParams {
    uint64_t region_id;
    uint32_t rmStatus;
};

struct UpdateRegionStateParams {
    uint64_t region_id;
    uint32_t state;
    uint32_t rmStatus;
};

struct QueryRegionParams {
    uint64_t region_id;
    uint64_t base;
    uint64_t length;
    uint32_t state;
    uint32_t found_by_va;
    uint32_t rmStatus;
};

struct QueryRegionByVaParams {
    uint64_t address;
    uint64_t region_id;
    uint64_t base;
    uint64_t length;
    uint32_t state;
    uint32_t found_by_va;
    uint32_t rmStatus;
};

struct PhaseFPingParams {
    uint32_t magic;
    uint32_t rmStatus;
};

struct PhaseFBatchMiniParams {
    uint32_t action;
    uint32_t count;
    uint64_t base;
    uint64_t length;
    uint32_t success_count;
    uint32_t failure_count;
    uint32_t rmStatus;
};

constexpr PhaseFUuid CPU_UUID = {{
    0xa6, 0x5e, 0x0f, 0x4e, 0xd7, 0xd4, 0x7b, 0xa2,
    0x50, 0x47, 0x41, 0x2c, 0x14, 0x2a, 0x77, 0x73
}};

std::mutex phase_f_log_mutex;

void phase_f_debug_log(const char* fmt, ...);

struct UvmFdCandidate {
    int source_fd;
    std::string path;
};

std::mutex g_region_persistent_fd_mutex;
int g_region_persistent_fd = -1;
std::mutex g_phase_f_region_cache_mutex;
std::unordered_map<uint64_t, PhaseFRange> g_phase_f_region_cache;
std::mutex g_cuda_device_uuid_cache_mutex;
std::unordered_map<int, PhaseFUuid> g_cuda_device_uuid_cache;

void cache_phase_f_region(uint64_t region_id, uintptr_t base, size_t length) {
    if (region_id == 0 || base == 0 || length == 0)
        return;

    std::lock_guard<std::mutex> lock(g_phase_f_region_cache_mutex);
    g_phase_f_region_cache[region_id] = PhaseFRange{
        static_cast<uint64_t>(base),
        static_cast<uint64_t>(length),
    };
}

void uncache_phase_f_region(uint64_t region_id) {
    std::lock_guard<std::mutex> lock(g_phase_f_region_cache_mutex);
    g_phase_f_region_cache.erase(region_id);
}

bool phase_f_use_test_ioctl() {
    const char* mode = std::getenv("VLLM_UVM_PHASE_F_IOCTL_MODE");
    return mode != nullptr &&
           (std::strcmp(mode, "test") == 0 ||
            std::strcmp(mode, "builtin_test") == 0);
}

unsigned int phase_f_batch_ioctl_cmd() {
    return phase_f_use_test_ioctl() ?
        UVM_TEST_PHASE_F_BATCH_EXECUTOR_IOCTL :
        UVM_PHASE_F_BATCH_EXECUTOR_IOCTL;
}

const char* phase_f_batch_ioctl_name() {
    return phase_f_use_test_ioctl() ? "test" : "policy";
}

bool phase_f_adopt_cuda_uvm_fd_enabled() {
    const char* mode = std::getenv("VLLM_UVM_PHASE_F_FD_MODE");
    if (mode == nullptr || mode[0] == '\0') return true;

    return std::strcmp(mode, "new") != 0 &&
           std::strcmp(mode, "fresh") != 0 &&
           std::strcmp(mode, "open") != 0;
}

bool phase_f_fd_link_is_uvm(const char* link_target) {
    if (link_target == nullptr) return false;

    const char* base = std::strrchr(link_target, '/');
    base = base == nullptr ? link_target : base + 1;

    return std::strcmp(base, "nvidia-uvm") == 0 ||
           std::strncmp(base, "nvidia-uvm ", strlen("nvidia-uvm ")) == 0;
}

std::vector<UvmFdCandidate> collect_existing_uvm_fd_candidates(const char* op_name) {
    std::vector<UvmFdCandidate> candidates;
    DIR* dir = opendir("/proc/self/fd");
    if (dir == nullptr) {
        phase_f_debug_log("%s adopt_fd opendir_failed errno=%d", op_name, errno);
        return candidates;
    }

    int dir_fd = dirfd(dir);
    struct dirent* entry = nullptr;

    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_name[0] == '.') continue;

        char* end = nullptr;
        long fd_long = std::strtol(entry->d_name, &end, 10);
        if (end == entry->d_name || *end != '\0') continue;
        if (fd_long < 0 || fd_long > INT_MAX) continue;

        int candidate_fd = static_cast<int>(fd_long);
        if (candidate_fd == dir_fd) continue;

        char fd_path[64];
        char link_target[PATH_MAX];
        std::snprintf(fd_path, sizeof(fd_path), "/proc/self/fd/%d", candidate_fd);

        ssize_t link_len = readlink(fd_path, link_target, sizeof(link_target) - 1);
        if (link_len < 0) continue;
        link_target[link_len] = '\0';

        if (!phase_f_fd_link_is_uvm(link_target)) continue;

        candidates.push_back(UvmFdCandidate{candidate_fd, link_target});
    }

    closedir(dir);

    std::sort(candidates.begin(), candidates.end(), [](const UvmFdCandidate& a, const UvmFdCandidate& b) {
        return a.source_fd < b.source_fd;
    });

    if (candidates.empty())
        phase_f_debug_log("%s adopt_fd no_existing_cuda_uvm_fd", op_name);
    else
        phase_f_debug_log("%s adopt_fd candidates=%zu", op_name, candidates.size());

    return candidates;
}

int duplicate_uvm_fd_candidate(const char* op_name, const UvmFdCandidate& candidate) {
    int dup_fd = fcntl(candidate.source_fd, F_DUPFD_CLOEXEC, 3);
    if (dup_fd < 0) {
        phase_f_debug_log("%s adopt_fd duplicate_failed candidate_fd=%d path=%s errno=%d",
                          op_name, candidate.source_fd, candidate.path.c_str(), errno);
        return -1;
    }

    return dup_fd;
}

bool initialize_duplicated_uvm_fd(const char* op_name, int fd, const UvmFdCandidate& candidate) {
    UvmInitializeParams init_params{};
    if (ioctl(fd, UVM_INITIALIZE_IOCTL, &init_params) == 0 &&
        init_params.rmStatus == 0)
        return true;

    int saved_errno = errno;
    phase_f_debug_log("%s adopt_fd init_rejected candidate_fd=%d dup_fd=%d path=%s errno=%d rmStatus=%u",
                      op_name,
                      candidate.source_fd,
                      fd,
                      candidate.path.c_str(),
                      saved_errno,
                      init_params.rmStatus);
    return false;
}

int adopt_existing_uvm_fd(const char* op_name) {
    std::vector<UvmFdCandidate> candidates = collect_existing_uvm_fd_candidates(op_name);
    if (candidates.empty())
        return -1;

    for (const UvmFdCandidate& candidate : candidates) {
        int dup_fd = duplicate_uvm_fd_candidate(op_name, candidate);
        if (dup_fd < 0)
            continue;

        if (initialize_duplicated_uvm_fd(op_name, dup_fd, candidate)) {
            phase_f_debug_log("%s adopt_fd adopted candidate_fd=%d dup_fd=%d path=%s",
                              op_name, candidate.source_fd, dup_fd, candidate.path.c_str());
            return dup_fd;
        }

        close(dup_fd);
    }

    return -1;
}

int open_initialized_uvm_fd(const char* op_name, bool prefer_existing = false) {
    if (prefer_existing && phase_f_adopt_cuda_uvm_fd_enabled()) {
        int adopted_fd = adopt_existing_uvm_fd(op_name);
        if (adopted_fd >= 0)
            return adopted_fd;
    }

    int fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        phase_f_debug_log("%s open_failed errno=%d", op_name, errno);
        return -1;
    }

    UvmInitializeParams init_params{};
    if (ioctl(fd, UVM_INITIALIZE_IOCTL, &init_params) != 0 ||
        init_params.rmStatus != 0) {
        int saved_errno = errno;
        phase_f_debug_log("%s init_failed errno=%d rmStatus=%u",
                          op_name, saved_errno, init_params.rmStatus);
        close(fd);
        return -1;
    }

    phase_f_debug_log("%s open_new_initialized fd=%d", op_name, fd);
    return fd;
}

int get_region_persistent_fd(const char* op_name) {
    std::lock_guard<std::mutex> lock(g_region_persistent_fd_mutex);
    if (g_region_persistent_fd >= 0)
        return g_region_persistent_fd;

    g_region_persistent_fd = open_initialized_uvm_fd(op_name, false);
    if (g_region_persistent_fd >= 0) {
        phase_f_debug_log("%s persistent_fd_opened fd=%d",
                          op_name,
                          g_region_persistent_fd);
    }
    return g_region_persistent_fd;
}

template <typename Fn>
int apply_to_existing_uvm_fds(const char* op_name, Fn fn) {
    if (!phase_f_adopt_cuda_uvm_fd_enabled())
        return -4;

    std::vector<UvmFdCandidate> candidates = collect_existing_uvm_fd_candidates(op_name);
    int successes = 0;
    int last_error = -4;

    for (const UvmFdCandidate& candidate : candidates) {
        int fd = duplicate_uvm_fd_candidate(op_name, candidate);
        int result;

        if (fd < 0)
            continue;

        if (!initialize_duplicated_uvm_fd(op_name, fd, candidate)) {
            close(fd);
            continue;
        }

        result = fn(fd, candidate.source_fd, candidate.path.c_str());
        close(fd);

        if (result >= 0)
            ++successes;
        else
            last_error = result;
    }

    if (successes > 0) {
        phase_f_debug_log("%s apply_existing successes=%d candidates=%zu",
                          op_name,
                          successes,
                          candidates.size());
        return 1;
    }

    return last_error;
}

const char* phase_f_action_name(uint32_t action) {
    switch (action) {
        case UVM_TEST_PHASE_F_ACTION_PREFETCH_CPU:
            return "PREFETCH_CPU";
        case UVM_TEST_PHASE_F_ACTION_PREFETCH_GPU:
            return "PREFETCH_GPU";
        case UVM_TEST_PHASE_F_ACTION_ADVISE_CPU:
            return "ADVISE_CPU";
        case UVM_TEST_PHASE_F_ACTION_QUERY_RESIDENCY:
            return "QUERY_RESIDENCY";
        case UVM_TEST_PHASE_F_ACTION_PREPARE_REMOTE_READ:
            return "PREPARE_REMOTE_READ";
        case UVM_TEST_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG:
            return "PREPARE_REMOTE_READ_BG";
        case UVM_TEST_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG_DEFERRED:
            return "PREPARE_REMOTE_READ_BG_DEFERRED";
        case UVM_TEST_PHASE_F_ACTION_START_BG_PROMOTION:
            return "START_BG_PROMOTION";
        case UVM_TEST_PHASE_F_ACTION_GPU_ONLY_READ_DUP_SOFT:
            return "GPU_ONLY_READ_DUP_SOFT";
        case UVM_TEST_PHASE_F_ACTION_PAUSE_BG_PROMOTION:
            return "PAUSE_BG_PROMOTION";
        case UVM_TEST_PHASE_F_ACTION_RESUME_BG_PROMOTION:
            return "RESUME_BG_PROMOTION";
        case UVM_TEST_PHASE_F_ACTION_EXPERT_READ_DUP_FAST:
            return "EXPERT_READ_DUP_FAST";
        case UVM_TEST_PHASE_F_ACTION_EXPERT_READ_DUP_FAST_NATIVE_REGIONS:
            return "EXPERT_READ_DUP_FAST_NATIVE_REGIONS";
        default:
            return "UNKNOWN";
    }
}

bool phase_f_action_needs_gpu_uuid(uint32_t action) {
    switch (action) {
        case UVM_TEST_PHASE_F_ACTION_PREFETCH_GPU:
        case UVM_TEST_PHASE_F_ACTION_QUERY_RESIDENCY:
        case UVM_TEST_PHASE_F_ACTION_PREPARE_REMOTE_READ:
        case UVM_TEST_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG:
        case UVM_TEST_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG_DEFERRED:
        case UVM_TEST_PHASE_F_ACTION_START_BG_PROMOTION:
        case UVM_TEST_PHASE_F_ACTION_GPU_ONLY_READ_DUP_SOFT:
        case UVM_TEST_PHASE_F_ACTION_EXPERT_READ_DUP_FAST:
        case UVM_TEST_PHASE_F_ACTION_EXPERT_READ_DUP_FAST_NATIVE_REGIONS:
            return true;
        default:
            return false;
    }
}



void phase_f_debug_log(const char* fmt, ...) {
    const char* path = std::getenv("VLLM_UVM_PHASE_F_DEBUG_LOG");
    if (path == nullptr || path[0] == '\0') return;

    std::lock_guard<std::mutex> guard(phase_f_log_mutex);
    FILE* fp = std::fopen(path, "a");
    if (fp == nullptr) return;

    auto now = std::chrono::system_clock::now();
    auto now_time = std::chrono::system_clock::to_time_t(now);
    struct tm tm_buf;
    gmtime_r(&now_time, &tm_buf);
    char ts[64];
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);

    std::fprintf(fp, "[%s] ", ts);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(fp, fmt, args);
    va_end(args);
    std::fprintf(fp, "\n");

    std::fflush(fp);
    fsync(fileno(fp));
    std::fclose(fp);
}


bool get_cuda_device_uuid(int device, PhaseFUuid* uuid) {
    if (uuid == nullptr) return false;
    if (device < 0) {
        cudaError_t current_err = cudaGetDevice(&device);
        if (current_err != cudaSuccess) return false;
    }

    {
        std::lock_guard<std::mutex> lock(g_cuda_device_uuid_cache_mutex);
        auto it = g_cuda_device_uuid_cache.find(device);
        if (it != g_cuda_device_uuid_cache.end()) {
            *uuid = it->second;
            return true;
        }
    }

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) return false;

#if CUDART_VERSION >= 10000
    static_assert(sizeof(prop.uuid.bytes) == sizeof(uuid->uuid),
                  "Unexpected CUDA UUID size");
    memcpy(uuid->uuid, prop.uuid.bytes, sizeof(uuid->uuid));
    {
        std::lock_guard<std::mutex> lock(g_cuda_device_uuid_cache_mutex);
        g_cuda_device_uuid_cache[device] = *uuid;
    }
    return true;
#else
    (void)prop;
    return false;
#endif
}

int uvm_driver_batch_execute_internal(const uintptr_t* ptrs,
                                      const size_t* sizes,
                                      size_t count,
                                      uint32_t action,
                                      int device,
                                      int async,
                                      uint32_t* failure_count_out,
                                      PhaseFRangeResult* results_out = nullptr) {
    if (failure_count_out != nullptr) *failure_count_out = 0;
    if (ptrs == nullptr || sizes == nullptr || count == 0) return 0;
    if (count > UVM_TEST_PHASE_F_MAX_RANGES) return -2;

    PhaseFBatchExecutorParams params{};
    params.action = action;
    params.count = static_cast<uint32_t>(count);
    params.async = async ? 1 : 0;
    params.gpu_uuid = CPU_UUID;
    if (phase_f_action_needs_gpu_uuid(action) &&
        !get_cuda_device_uuid(device, &params.gpu_uuid)) {
        phase_f_debug_log("driver_batch action=%s count=%zu device=%d uuid_lookup_failed",
                          phase_f_action_name(action), count, device);
        return -3;
    }

    for (size_t i = 0; i < count; ++i) {
        params.ranges[i].base = static_cast<uint64_t>(ptrs[i]);
        params.ranges[i].length = static_cast<uint64_t>(sizes[i]);
    }

    unsigned int ioctl_cmd = phase_f_batch_ioctl_cmd();
    const char* ioctl_name = phase_f_batch_ioctl_name();
    int final_result = -4;
    uint32_t final_failure_count = 0;
    bool attempted_existing_fd = false;
    std::vector<UvmFdCandidate> fd_candidates;

    if (phase_f_adopt_cuda_uvm_fd_enabled())
        fd_candidates = collect_existing_uvm_fd_candidates("driver_batch");

    auto try_ioctl = [&](int fd, int source_fd, const char* fd_source, const char* path) {
        PhaseFBatchExecutorParams attempt = params;
        phase_f_debug_log("driver_batch ioctl=%s cmd=%u fd_source=%s source_fd=%d fd=%d path=%s action=%s count=%zu device=%d async=%d ioctl_begin first_base=0x%llx first_len=%llu",
                          ioctl_name, ioctl_cmd, fd_source, source_fd, fd, path ? path : "",
                          phase_f_action_name(action), count, device, async,
                          count > 0 ? (unsigned long long)attempt.ranges[0].base : 0ULL,
                          count > 0 ? (unsigned long long)attempt.ranges[0].length : 0ULL);
        int ret = ioctl(fd, ioctl_cmd, &attempt);
        int saved_errno = errno;
        phase_f_debug_log("driver_batch ioctl=%s cmd=%u fd_source=%s source_fd=%d fd=%d action=%s count=%zu ioctl_done ret=%d errno=%d rmStatus=%u success=%u failure=%u first_status=%u first_queried=%llu first_gpu_bytes=%llu first_cpu_bytes=%llu first_any_bytes=%llu first_unknown_bytes=%llu",
                          ioctl_name, ioctl_cmd, fd_source, source_fd, fd,
                          phase_f_action_name(action), count, ret, saved_errno,
                          attempt.rmStatus, attempt.success_count, attempt.failure_count,
                          count > 0 ? attempt.results[0].status : 0,
                          count > 0 ? static_cast<unsigned long long>(attempt.results[0].queried_bytes) : 0ULL,
                          count > 0 ? static_cast<unsigned long long>(attempt.results[0].resident_gpu_bytes) : 0ULL,
                          count > 0 ? static_cast<unsigned long long>(attempt.results[0].resident_cpu_bytes) : 0ULL,
                          count > 0 ? static_cast<unsigned long long>(attempt.results[0].resident_any_bytes) : 0ULL,
                          count > 0 ? static_cast<unsigned long long>(attempt.results[0].resident_unknown_bytes) : 0ULL);

        final_failure_count = attempt.failure_count;
        if (ret != 0) {
            final_result = -6;
            return false;
        }

        if (attempt.rmStatus != 0 && attempt.success_count == 0) {
            final_result = -static_cast<int>(attempt.rmStatus);
            return false;
        }

        if (results_out != nullptr)
            std::copy(attempt.results, attempt.results + count, results_out);

        final_result = static_cast<int>(attempt.success_count);
        return true;
    };

    for (const UvmFdCandidate& candidate : fd_candidates) {
        int adopted_fd = duplicate_uvm_fd_candidate("driver_batch", candidate);
        if (adopted_fd < 0)
            continue;

        attempted_existing_fd = true;
        bool success = try_ioctl(
            adopted_fd, candidate.source_fd, "adopted", candidate.path.c_str());
        close(adopted_fd);
        if (success) {
            if (failure_count_out != nullptr)
                *failure_count_out = final_failure_count;
            return final_result;
        }
    }

    int fd = open_initialized_uvm_fd("driver_batch", false);
    if (fd >= 0) {
        bool success = try_ioctl(fd, -1, "new", "/dev/nvidia-uvm");
        close(fd);
        if (success || !attempted_existing_fd) {
            if (failure_count_out != nullptr)
                *failure_count_out = final_failure_count;
            return final_result;
        }
    }

    if (failure_count_out != nullptr)
        *failure_count_out = final_failure_count;
    return final_result;
}

int uvm_driver_phase_f_register_region_fd(int fd,
                                          uint64_t region_id,
                                          uintptr_t base,
                                          size_t length) {
    PhaseFRegisterRegionParams params{};
    params.region.region_id = region_id;
    params.region.base = static_cast<uint64_t>(base);
    params.region.length = static_cast<uint64_t>(length);
    // The wire contract contains only identity and VA geometry.

    phase_f_debug_log("driver_register_region cmd=%u id=%llu base=0x%llx length=%llu ioctl_begin",
                      UVM_PHASE_F_REGISTER_REGION_IOCTL,
                      static_cast<unsigned long long>(params.region.region_id),
                      static_cast<unsigned long long>(params.region.base),
                      static_cast<unsigned long long>(params.region.length));
    int ret = ioctl(fd, UVM_PHASE_F_REGISTER_REGION_IOCTL, &params);
    int saved_errno = errno;
    phase_f_debug_log("driver_register_region cmd=%u id=%llu ioctl_done ret=%d errno=%d rmStatus=%u",
                      UVM_PHASE_F_REGISTER_REGION_IOCTL,
                      static_cast<unsigned long long>(region_id),
                      ret,
                      saved_errno,
                      params.rmStatus);

    if (ret != 0) return -6;
    if (params.rmStatus != 0) return -static_cast<int>(params.rmStatus);
    return 1;
}

int uvm_driver_phase_f_register_region_internal(uint64_t region_id,
                                                uintptr_t base,
                                                size_t length) {
    int existing_result = apply_to_existing_uvm_fds(
        "driver_register_region",
        [&](int fd, int source_fd, const char* path) {
            phase_f_debug_log("driver_register_region fd_source=adopted source_fd=%d fd=%d path=%s id=%llu base=0x%llx length=%zu",
                              source_fd,
                              fd,
                              path ? path : "",
                              static_cast<unsigned long long>(region_id),
                              static_cast<unsigned long long>(base),
                              length);
            return uvm_driver_phase_f_register_region_fd(fd, region_id, base, length);
        });

    int fd = get_region_persistent_fd("driver_register_region_persistent");
    int persistent_result = -4;

    if (fd >= 0) {
        phase_f_debug_log("driver_register_region fd_source=persistent fd=%d id=%llu base=0x%llx length=%zu",
                          fd,
                          static_cast<unsigned long long>(region_id),
                          static_cast<unsigned long long>(base),
                          length);
        persistent_result = uvm_driver_phase_f_register_region_fd(fd, region_id, base, length);
    }

    if (existing_result > 0 || persistent_result > 0) {
        cache_phase_f_region(region_id, base, length);
        return 1;
    }
    if (persistent_result < 0)
        return persistent_result;
    return existing_result;
}

int uvm_driver_phase_f_unregister_region_fd(int fd, uint64_t region_id) {
    PhaseFUnregisterRegionParams params{};
    params.region_id = region_id;

    phase_f_debug_log("driver_unregister_region cmd=%u id=%llu ioctl_begin",
                      UVM_PHASE_F_UNREGISTER_REGION_IOCTL,
                      static_cast<unsigned long long>(region_id));
    int ret = ioctl(fd, UVM_PHASE_F_UNREGISTER_REGION_IOCTL, &params);
    int saved_errno = errno;
    phase_f_debug_log("driver_unregister_region cmd=%u id=%llu ioctl_done ret=%d errno=%d rmStatus=%u",
                      UVM_PHASE_F_UNREGISTER_REGION_IOCTL,
                      static_cast<unsigned long long>(region_id),
                      ret,
                      saved_errno,
                      params.rmStatus);

    if (ret != 0) return -6;
    if (params.rmStatus != 0) return -static_cast<int>(params.rmStatus);
    return 1;
}

int uvm_driver_phase_f_unregister_region_internal(uint64_t region_id) {
    int existing_result = apply_to_existing_uvm_fds(
        "driver_unregister_region",
        [&](int fd, int source_fd, const char* path) {
            phase_f_debug_log("driver_unregister_region fd_source=adopted source_fd=%d fd=%d path=%s id=%llu",
                              source_fd,
                              fd,
                              path ? path : "",
                              static_cast<unsigned long long>(region_id));
            return uvm_driver_phase_f_unregister_region_fd(fd, region_id);
        });
    if (existing_result > 0) {
        uncache_phase_f_region(region_id);
        return existing_result;
    }

    int fd = open_initialized_uvm_fd("driver_unregister_region", false);
    int result;

    if (fd < 0) {
        uncache_phase_f_region(region_id);
        return -4;
    }
    result = uvm_driver_phase_f_unregister_region_fd(fd, region_id);
    close(fd);
    uncache_phase_f_region(region_id);
    return result;
}

int uvm_driver_update_region_state_internal(uint64_t region_id,
                                            uint32_t state) {
    auto update_state_on_fd = [&](int fd, const char* fd_source, int source_fd, const char* path) {
        UpdateRegionStateParams params{};
        params.region_id = region_id;
        params.state = state;

        phase_f_debug_log("driver_update_region_state cmd=%u fd_source=%s source_fd=%d fd=%d path=%s id=%llu state=%u ioctl_begin",
                          UVM_UPDATE_REGION_STATE_IOCTL,
                          fd_source,
                          source_fd,
                          fd,
                          path ? path : "",
                          static_cast<unsigned long long>(region_id),
                          state);
        int ret = ioctl(fd, UVM_UPDATE_REGION_STATE_IOCTL, &params);
        int saved_errno = errno;
        phase_f_debug_log("driver_update_region_state cmd=%u fd_source=%s source_fd=%d fd=%d id=%llu ioctl_done ret=%d errno=%d rmStatus=%u",
                          UVM_UPDATE_REGION_STATE_IOCTL,
                          fd_source,
                          source_fd,
                          fd,
                          static_cast<unsigned long long>(region_id),
                          ret,
                          saved_errno,
                          params.rmStatus);

        if (ret != 0) return -6;
        if (params.rmStatus != 0) return -static_cast<int>(params.rmStatus);
        return 1;
    };

    int existing_result = apply_to_existing_uvm_fds(
        "driver_update_region_state",
        [&](int fd, int source_fd, const char* path) {
            return update_state_on_fd(fd, "adopted", source_fd, path);
        });

    int fd = get_region_persistent_fd("driver_update_region_state_persistent");
    int persistent_result = -4;
    if (fd >= 0)
        persistent_result = update_state_on_fd(fd, "persistent", -1, "/dev/nvidia-uvm:persistent");

    if (existing_result > 0 || persistent_result > 0)
        return 1;
    if (persistent_result < 0)
        return persistent_result;
    return existing_result;
}

int uvm_driver_query_region_internal(uint64_t region_id,
                                     uint64_t* values_out,
                                     size_t values_count) {
    if (values_out == nullptr || values_count < 4) return -2;

    QueryRegionParams params{};
    params.region_id = region_id;
    int final_ret = -1;
    int final_errno = 0;

    auto query_on_fd = [&](int fd, const char* fd_source, int source_fd, const char* path) {
        QueryRegionParams attempt{};
        attempt.region_id = region_id;

        phase_f_debug_log("driver_query_region cmd=%u fd_source=%s source_fd=%d fd=%d path=%s id=%llu ioctl_begin",
                          UVM_QUERY_REGION_IOCTL,
                          fd_source,
                          source_fd,
                          fd,
                          path ? path : "",
                          static_cast<unsigned long long>(region_id));
        int ret = ioctl(fd, UVM_QUERY_REGION_IOCTL, &attempt);
        int saved_errno = errno;
        phase_f_debug_log("driver_query_region cmd=%u fd_source=%s source_fd=%d fd=%d id=%llu ioctl_done ret=%d errno=%d rmStatus=%u base=0x%llx length=%llu state=%u",
                          UVM_QUERY_REGION_IOCTL,
                          fd_source,
                          source_fd,
                          fd,
                          static_cast<unsigned long long>(region_id),
                          ret,
                          saved_errno,
                          attempt.rmStatus,
                          static_cast<unsigned long long>(attempt.base),
                          static_cast<unsigned long long>(attempt.length),
                          attempt.state);
        final_ret = ret;
        final_errno = saved_errno;
        if (ret == 0 && attempt.rmStatus == 0) {
            params = attempt;
            return 1;
        }
        if (ret != 0)
            return -6;
        return -static_cast<int>(attempt.rmStatus);
    };

    int existing_result = apply_to_existing_uvm_fds(
        "driver_query_region",
        [&](int fd, int source_fd, const char* path) {
            return query_on_fd(fd, "adopted", source_fd, path);
        });

    int persistent_result = -4;
    if (existing_result <= 0) {
        int fd = get_region_persistent_fd("driver_query_region_persistent");
        if (fd >= 0)
            persistent_result = query_on_fd(fd, "persistent", -1, "/dev/nvidia-uvm:persistent");
    }

    phase_f_debug_log("driver_query_region cmd=%u id=%llu ioctl_done ret=%d errno=%d rmStatus=%u base=0x%llx length=%llu state=%u",
                      UVM_QUERY_REGION_IOCTL,
                      static_cast<unsigned long long>(region_id),
                      final_ret,
                      final_errno,
                      params.rmStatus,
                      static_cast<unsigned long long>(params.base),
                      static_cast<unsigned long long>(params.length),
                      params.state);

    if (existing_result <= 0 && persistent_result <= 0) {
        if (persistent_result < 0)
            return persistent_result;
        return existing_result;
    }

    values_out[0] = params.region_id;
    values_out[1] = params.base;
    values_out[2] = params.length;
    values_out[3] = params.state;
    return 1;
}

int uvm_driver_query_region_by_va_internal(uintptr_t address,
                                           uint64_t* values_out,
                                           size_t values_count) {
    if (values_out == nullptr || values_count < 4) return -2;
    int fd = open_initialized_uvm_fd("driver_query_region_by_va", true);
    if (fd < 0) return -4;

    QueryRegionByVaParams params{};
    params.address = static_cast<uint64_t>(address);

    phase_f_debug_log("driver_query_region_by_va cmd=%u address=0x%llx ioctl_begin",
                      UVM_QUERY_REGION_BY_VA_IOCTL,
                      static_cast<unsigned long long>(params.address));
    int ret = ioctl(fd, UVM_QUERY_REGION_BY_VA_IOCTL, &params);
    int saved_errno = errno;
    close(fd);
    phase_f_debug_log("driver_query_region_by_va cmd=%u address=0x%llx ioctl_done ret=%d errno=%d rmStatus=%u id=%llu state=%u",
                      UVM_QUERY_REGION_BY_VA_IOCTL,
                      static_cast<unsigned long long>(params.address),
                      ret,
                      saved_errno,
                      params.rmStatus,
                      static_cast<unsigned long long>(params.region_id),
                      params.state);

    if (ret != 0) return -6;
    if (params.rmStatus != 0) return -static_cast<int>(params.rmStatus);

    values_out[0] = params.region_id;
    values_out[1] = params.base;
    values_out[2] = params.length;
    values_out[3] = params.state;
    return 1;
}

int uvm_driver_phase_f_probe_existing_uvm_fd_internal() {
    int fd = adopt_existing_uvm_fd("driver_probe_existing_uvm_fd");

    if (fd < 0)
        return 0;

    close(fd);
    return 1;
}

int uvm_driver_phase_f_ping_internal() {
    phase_f_debug_log("driver_ping before_params_init");
    PhaseFBatchExecutorParams params{};
    params.action = UVM_TEST_PHASE_F_ACTION_QUERY_RESIDENCY;
    params.count = 0;
    params.async = 0;
    params.gpu_uuid = CPU_UUID;
    phase_f_debug_log("driver_ping after_gpu_uuid_init action=%u count=%u async=%u",
                      params.action, params.count, params.async);

    int fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        phase_f_debug_log("driver_ping open_failed errno=%d", errno);
        return -4;
    }

    UvmInitializeParams init_params{};
    if (ioctl(fd, UVM_INITIALIZE_IOCTL, &init_params) != 0 ||
        init_params.rmStatus != 0) {
        int saved_errno = errno;
        phase_f_debug_log("driver_ping init_failed errno=%d rmStatus=%u",
                          saved_errno, init_params.rmStatus);
        close(fd);
        return -5;
    }

    unsigned int ioctl_cmd = phase_f_batch_ioctl_cmd();
    const char* ioctl_name = phase_f_batch_ioctl_name();
    phase_f_debug_log("driver_ping ioctl=%s cmd=%u ioctl_begin", ioctl_name, ioctl_cmd);

    int ret = ioctl(fd, ioctl_cmd, &params);
    int saved_errno = errno;
    close(fd);
    phase_f_debug_log("driver_ping ioctl=%s cmd=%u ioctl_done ret=%d errno=%d rmStatus=%u success=%u failure=%u",
                      ioctl_name, ioctl_cmd, ret, saved_errno, params.rmStatus,
                      params.success_count, params.failure_count);

    if (ret != 0) return -6;
    if (params.rmStatus != 0) return -static_cast<int>(params.rmStatus);
    return 0;
}

int uvm_driver_phase_f_small_ping_internal() {
    PhaseFPingParams params{};
    params.magic = UVM_TEST_PHASE_F_PING_MAGIC;

    int fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        phase_f_debug_log("driver_small_ping open_failed errno=%d", errno);
        return -4;
    }

    UvmInitializeParams init_params{};
    if (ioctl(fd, UVM_INITIALIZE_IOCTL, &init_params) != 0 ||
        init_params.rmStatus != 0) {
        int saved_errno = errno;
        phase_f_debug_log("driver_small_ping init_failed errno=%d rmStatus=%u",
                          saved_errno, init_params.rmStatus);
        close(fd);
        return -5;
    }

    phase_f_debug_log("driver_small_ping ioctl_begin magic=0x%x", params.magic);
    int ret = ioctl(fd, UVM_TEST_PHASE_F_PING_IOCTL, &params);
    int saved_errno = errno;
    close(fd);
    phase_f_debug_log("driver_small_ping ioctl_done ret=%d errno=%d rmStatus=%u magic=0x%x",
                      ret, saved_errno, params.rmStatus, params.magic);
    if (ret != 0) return -6;
    if (params.rmStatus != 0) return -static_cast<int>(params.rmStatus);
    return 0;
}

int uvm_driver_phase_f_batch_mini_internal(uint32_t count, uintptr_t base, size_t length) {
    PhaseFBatchMiniParams params{};
    params.action = UVM_TEST_PHASE_F_ACTION_QUERY_RESIDENCY;
    params.count = count;
    params.base = static_cast<uint64_t>(base);
    params.length = static_cast<uint64_t>(length);

    int fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        phase_f_debug_log("driver_batch_mini count=%u open_failed errno=%d", count, errno);
        return -4;
    }

    UvmInitializeParams init_params{};
    if (ioctl(fd, UVM_INITIALIZE_IOCTL, &init_params) != 0 ||
        init_params.rmStatus != 0) {
        int saved_errno = errno;
        phase_f_debug_log("driver_batch_mini count=%u init_failed errno=%d rmStatus=%u",
                          count, saved_errno, init_params.rmStatus);
        close(fd);
        return -5;
    }

    phase_f_debug_log("driver_batch_mini count=%u ioctl_begin base=0x%llx length=%llu",
                      count,
                      static_cast<unsigned long long>(params.base),
                      static_cast<unsigned long long>(params.length));
    int ret = ioctl(fd, UVM_TEST_PHASE_F_BATCH_MINI_IOCTL, &params);
    int saved_errno = errno;
    close(fd);
    phase_f_debug_log("driver_batch_mini count=%u ioctl_done ret=%d errno=%d rmStatus=%u success=%u failure=%u",
                      count, ret, saved_errno, params.rmStatus,
                      params.success_count, params.failure_count);
    if (ret != 0) return -6;
    if (params.rmStatus != 0) return -static_cast<int>(params.rmStatus);
    return 0;
}

int uvm_driver_phase_f_open_only_internal() {
    int fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        phase_f_debug_log("driver_open open_failed errno=%d", errno);
        return -4;
    }

    phase_f_debug_log("driver_open open_done fd=%d", fd);
    close(fd);
    phase_f_debug_log("driver_open close_done");
    return 0;
}

int uvm_driver_phase_f_init_only_internal() {
    UvmInitializeParams init_params{};

    int fd = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        phase_f_debug_log("driver_init open_failed errno=%d", errno);
        return -4;
    }

    phase_f_debug_log("driver_init open_done fd=%d", fd);
    if (ioctl(fd, UVM_INITIALIZE_IOCTL, &init_params) != 0 || init_params.rmStatus != 0) {
        int saved_errno = errno;
        phase_f_debug_log("driver_init init_failed errno=%d rmStatus=%u",
                          saved_errno, init_params.rmStatus);
        close(fd);
        return -5;
    }

    phase_f_debug_log("driver_init init_done rmStatus=%u", init_params.rmStatus);
    close(fd);
    phase_f_debug_log("driver_init close_done");
    return 0;
}

int uvm_driver_phase_f_ioctl_only_internal() {
    return uvm_driver_phase_f_ping_internal();
}

}  // namespace

extern "C" {

// Global memory statistics (using C++ std::atomic)
static std::atomic<size_t> total_allocated{0};
static std::atomic<size_t> peak_allocated{0};
static std::atomic<size_t> num_allocs{0};
static std::atomic<size_t> num_frees{0};

// Configuration
static int enable_prefetch = 0;  // Whether to prefetch to device after allocation
static int verbose_logging = 0;  // Whether to log allocations
static size_t trace_min_bytes = 1 * 1024 * 1024;  // Trace allocations >= 1 MiB
static int policy_enabled = 1;  // Trace-only policy classification is enabled by default
static int policy_warmup_prefetch_enabled = 0;
static int policy_warmup_advise_gpu = 0;
static size_t policy_warmup_prefetch_min_bytes = 1 * 1024 * 1024;
static std::string policy_mode = "trace_only";
static int unknown_detail_enabled = 0;
static size_t unknown_detail_min_bytes = 0;
static int gap_watch_enabled = 0;
static uintptr_t gap_watch_start = 0;
static uintptr_t gap_watch_end = 0;
static std::string gap_watch_name = "unnamed_gap";
static int gap_watch_all_classes = 1;
static size_t gap_watch_min_bytes = 0;
static std::string gap_watch_target_class = "any";
static std::string gap_watch_policy_action = "observe";
static std::string gap_watch_control_file = "";
static size_t gap_watch_refresh_ms = 250;
static int device_direct_enable = 0;
static size_t device_direct_min_bytes = 4096;
static size_t device_direct_max_bytes = 1 * 1024 * 1024;
static size_t device_direct_max_total_bytes = 256 * 1024 * 1024;
static std::string device_direct_backend = "cuda_malloc";
static bool device_direct_pool_release_threshold_set = false;
static size_t device_direct_pool_release_threshold = 0;
static bool device_direct_pool_config_attempted = false;
static bool device_direct_pool_config_success = false;
static int device_direct_pool_config_device = -1;
static std::string device_direct_pool_config_error = "not_configured";
static std::string device_direct_target_phases =
    "enabled:attention,enabled:moe,enabled:model_forward";
static int kv_device_direct_enable = 0;
static size_t kv_device_direct_min_bytes = 0;
static size_t kv_device_direct_max_bytes = 0;
static size_t kv_device_direct_max_total_bytes = 0;
static int fixed_weight_device_direct_enable = 0;
static size_t fixed_weight_device_direct_min_bytes = 0;
static size_t fixed_weight_device_direct_max_bytes = 0;
static size_t fixed_weight_device_direct_max_total_bytes = 0;
static size_t kv_budget_bytes = 0;
static std::string kv_budget_mode = "trace_only";
static size_t weight_budget_bytes = 0;
static std::string weight_budget_mode = "trace_only";
static int weight_budget_cpu_offload_enable = 0;
static int pool_registry_enabled = 0;
static int driver_scratch_tag_enable = 0;
static int driver_scratch_budget_enforce = 0;
static int driver_scratch_budget_enforce_warmup = 0;
static size_t driver_scratch_budget_bytes = 0;
static int scratch_pool_enable = 0;
static size_t scratch_pool_budget_bytes = 1 * 1024 * 1024;
static std::string scratch_pool_mode = "trace_only";
static std::string scratch_pool_target_phases = "enabled:attention";
static int scratch_pool_reuse_cache_enable = 0;
static size_t scratch_pool_reuse_cache_max_bytes = 64 * 1024 * 1024;
static size_t scratch_pool_reuse_cache_max_entries_per_bucket = 8;
static bool gap_watch_control_seen = false;
static uint64_t gap_watch_control_mtime_ns = 0;
static off_t gap_watch_control_size = -1;
static std::chrono::steady_clock::time_point gap_watch_last_refresh_check =
    std::chrono::steady_clock::time_point::min();

// Log file handling
static FILE* log_file = nullptr;
static std::mutex log_mutex;
static std::mutex device_direct_pool_mutex;
static std::mutex scratch_pool_reuse_cache_mutex;
static std::chrono::steady_clock::time_point start_time;
static bool log_initialized = false;
static std::string current_phase = "unscoped";

struct ScratchPoolReuseCacheEntry {
    void* ptr = nullptr;
    size_t size = 0;
    int device = -1;
    std::string size_bucket;
    cudaEvent_t ready_event = nullptr;
    bool has_ready_event = false;
};

static std::unordered_map<std::string, std::vector<ScratchPoolReuseCacheEntry>>
    scratch_pool_reuse_cache;
static size_t scratch_pool_reuse_cache_current_bytes = 0;
static size_t scratch_pool_reuse_cache_current_entries = 0;

// =============================================================================
// 全局策略遥测与效能评估指标 (Global Policy Telemetry Metrics)
// 采用无锁 (Lock-free) 的 std::atomic 实现，确保在多流、多线程高并发推理场景下
// 收集指标时不会引入额外的锁竞争与长尾延迟。这些数据通常在 Session 结束时汇总打印。
// =============================================================================

// -----------------------------------------------------------------------------
// 1. Gap Watch 基础观测指标 (Gap Watch Observation Metrics)
// 用于量化目标虚拟/物理地址空间（Gap）的内存活动密集度（即：热点有多“热”）
// -----------------------------------------------------------------------------

/// @brief 触发 Gap 重叠的分配总次数。只要申请的内存块与监控区间有交集即 +1。
static std::atomic<size_t> gap_watch_overlap_allocs{0};

/// @brief 落入监控区间的实际重叠字节总量。用于计算“热点区域空间利用率/重叠率”。
static std::atomic<size_t> gap_watch_overlap_bytes_total{0};

/// @brief 既与监控区间重叠，又完全符合预设意图类别（Target Class，如 KV Cache）的分配次数。
/// @note 该指标与 gap_watch_overlap_allocs 的差值，反映了目标热点区域内的“内存噪声”
/// （即有多少非目标类型的零碎内存恰好也分配在了这个区间）。
static std::atomic<size_t> gap_watch_target_class_match_allocs{0};


// -----------------------------------------------------------------------------
// 2. Gap Watch 策略执行指标 (Gap Watch Policy Execution Metrics)
// 用于评估主动干预动作（如强制 Advise 或 Prefetch）的覆盖面与系统兼容性
// -----------------------------------------------------------------------------

/// @brief 成功触发并尝试执行主动策略干预（提权动作）的分配次数。
static std::atomic<size_t> gap_watch_policy_applied_allocs{0};

/// @brief 被主动策略干预所覆盖的热点字节总量。
/// @note 这是评估“预取/迁移带宽开销”的核心基准数据。
static std::atomic<size_t> gap_watch_policy_applied_overlap_bytes{0};

/// @brief 底层 CUDA API（如 cudaMemPrefetchAsync）调用成功的次数。
static std::atomic<size_t> gap_watch_policy_success_allocs{0};

/// @brief 策略执行失败的次数（例如：设备不支持并发预取、CUDA 流错误等）。
/// @note 如果此数值异常升高，提示底层驱动或环境配置存在瓶颈，需结合 trace 日志排查。
static std::atomic<size_t> gap_watch_policy_failed_allocs{0};


// -----------------------------------------------------------------------------
// 3. Device Direct 演进指标 (Device Direct Evolution Metrics)
// 用于支撑系统从“纯 UVM 托管”向“混合显存池（Bypass UVM）”演进的决策依据。
// 这些是前瞻性评估指标，量化了如果切断 CPU 侧的 Page Fault 关联，能拯救多少内存。
// -----------------------------------------------------------------------------

/// @brief 参与 Device Direct 旁路分配评估的分配总次数（即经过 Device Direct Trace 策略判定的次数）。
static std::atomic<size_t> device_direct_trace_allocs{0};

/// @brief 经策略引擎判定，完全具备“安全绕过 UVM 直接分配纯显存”资格的分配次数。
/// @note 达标条件通常包括：无 CPU 访问风险、属于明确的计算临时空间 (RuntimeWorkspace)、大小在设定阈值内。
static std::atomic<size_t> device_direct_eligible_allocs{0};

/// @brief Stage C 真正路由到 GPU-only backend 的分配次数。
static std::atomic<size_t> device_direct_actual_allocs{0};

/// @brief Stage C 真正路由到 GPU-only backend 的累计字节数。
static std::atomic<size_t> device_direct_actual_bytes{0};

/// @brief Stage C GPU-only backend 当前仍然存活的真实显存字节数。
static std::atomic<size_t> device_direct_live_bytes{0};

/// @brief Stage C GPU-only backend 存活显存高水位，用于验证 C1 总预算是否生效。
static std::atomic<size_t> device_direct_peak_live_bytes{0};

/// @brief Stage C 因总预算不足而拒绝 device_direct、回退 managed 的次数。
static std::atomic<size_t> device_direct_budget_rejects{0};

/// @brief Stage C 试图启用 GPU-only backend 但 cudaMalloc 失败、回退到 managed 的次数。
static std::atomic<size_t> device_direct_fallback_allocs{0};

/// @brief Stage C GPU-only backend 分配在 free 阶段成功释放的次数。
static std::atomic<size_t> device_direct_free_success_allocs{0};

/// @brief 所有参与评估的内存请求累计字节量。
/// @note 结合 eligible_allocs 的大小，可以量化出下一阶段（Phase C）真正开启 Device Direct 后，
/// 能为系统节省多少 UVM 缺页追踪开销。
static std::atomic<size_t> device_direct_requested_bytes{0};

/// @brief Stage D：被识别为 KV cache 的分配次数。
static std::atomic<size_t> kv_trace_allocs{0};

/// @brief Stage D：KV cache 分配请求累计字节数。
static std::atomic<size_t> kv_requested_bytes{0};

/// @brief Stage D：当前仍存活的 KV cache 字节数。
static std::atomic<size_t> kv_live_bytes{0};

/// @brief Stage D：KV cache 存活字节高水位。
static std::atomic<size_t> kv_peak_live_bytes{0};

/// @brief Stage D：KV cache 分配后超过独立预算的次数。
static std::atomic<size_t> kv_budget_over_allocs{0};

/// @brief Stage D：KV budget enforce 模式下产生的软拒绝信号次数。
static std::atomic<size_t> kv_budget_reject_allocs{0};

/// @brief Stage D：KV cache 分配在 free 阶段成功释放的次数。
static std::atomic<size_t> kv_free_success_allocs{0};

/// @brief Stage E：被识别为模型权重的分配次数。
static std::atomic<size_t> weight_trace_allocs{0};

/// @brief Stage E：模型权重分配请求累计字节数。
static std::atomic<size_t> weight_requested_bytes{0};

/// @brief Stage E：当前仍存活的模型权重字节数。
static std::atomic<size_t> weight_live_bytes{0};

/// @brief Stage E：模型权重存活字节高水位。
static std::atomic<size_t> weight_peak_live_bytes{0};

/// @brief Stage E：模型权重分配后超过独立预算的次数。
static std::atomic<size_t> weight_budget_over_allocs{0};

/// @brief Stage E：weight budget enforce 模式下产生的软拒绝信号次数。
static std::atomic<size_t> weight_budget_reject_allocs{0};

/// @brief Stage G：weight budget enforce 模式下真实触发 CPU offload 的次数。
static std::atomic<size_t> weight_budget_cpu_offload_allocs{0};

/// @brief Stage G：weight budget enforce 模式下真实触发 CPU offload 的累计字节数。
static std::atomic<size_t> weight_budget_cpu_offload_bytes{0};

/// @brief Stage G：weight budget enforce CPU offload 执行失败次数。
static std::atomic<size_t> weight_budget_cpu_offload_failures{0};

/// @brief Stage E：模型权重分配在 free 阶段成功释放的次数。
static std::atomic<size_t> weight_free_success_allocs{0};

/// @brief Stage F：统一 Pool Registry 记录到对象索引的分配次数。
static std::atomic<size_t> pool_registry_tracked_allocs{0};

/// @brief Stage F：统一 Pool Registry 已成功闭环释放的对象次数。
static std::atomic<size_t> pool_registry_free_success_allocs{0};

/// @brief Stage F：统一 Pool Registry 当前存活对象数。
static std::atomic<size_t> pool_registry_live_objects{0};

/// @brief Stage F：统一 Pool Registry 存活对象数高水位。
static std::atomic<size_t> pool_registry_peak_live_objects{0};

/// @brief Stage F：KV pool 分配次数。
static std::atomic<size_t> pool_kv_allocs{0};
static std::atomic<size_t> pool_kv_requested_bytes{0};
static std::atomic<size_t> pool_kv_live_bytes{0};
static std::atomic<size_t> pool_kv_peak_live_bytes{0};
static std::atomic<size_t> pool_kv_free_success_allocs{0};

/// @brief Stage F：Weights pool 分配次数。
static std::atomic<size_t> pool_weight_allocs{0};
static std::atomic<size_t> pool_weight_requested_bytes{0};
static std::atomic<size_t> pool_weight_live_bytes{0};
static std::atomic<size_t> pool_weight_peak_live_bytes{0};
static std::atomic<size_t> pool_weight_free_success_allocs{0};

/// @brief Stage F：Runtime scratch/workspace pool 分配次数。
static std::atomic<size_t> pool_runtime_scratch_allocs{0};
static std::atomic<size_t> pool_runtime_scratch_requested_bytes{0};
static std::atomic<size_t> pool_runtime_scratch_live_bytes{0};
static std::atomic<size_t> pool_runtime_scratch_peak_live_bytes{0};
static std::atomic<size_t> pool_runtime_scratch_free_success_allocs{0};

/// @brief Stage N：driver scratch pool tag admission live bytes.
static std::atomic<size_t> driver_scratch_live_bytes{0};

/// @brief Stage N：driver scratch pool tag admission peak live bytes.
static std::atomic<size_t> driver_scratch_peak_live_bytes{0};

/// @brief Stage N：driver scratch tag admissions rejected by hard budget.
static std::atomic<size_t> driver_scratch_budget_reject_allocs{0};

/// @brief Stage N：driver scratch tag admissions that would exceed budget.
static std::atomic<size_t> driver_scratch_budget_over_allocs{0};

/// @brief Stage N：driver scratch tag budget reservations released on free.
static std::atomic<size_t> driver_scratch_budget_release_allocs{0};

/// @brief Stage G：runtime scratch pool 准入控制参与评估的分配次数。
static std::atomic<size_t> scratch_pool_trace_allocs{0};

/// @brief Stage G：runtime scratch pool 满足独立准入条件的分配次数。
static std::atomic<size_t> scratch_pool_eligible_allocs{0};

/// @brief Stage G：runtime scratch pool 实际进入 device-direct backend 的分配次数。
static std::atomic<size_t> scratch_pool_device_direct_allocs{0};

/// @brief Stage G：runtime scratch pool 实际进入 device-direct backend 的累计字节数。
static std::atomic<size_t> scratch_pool_device_direct_bytes{0};

/// @brief Stage G：runtime scratch pool device-direct 当前存活字节数。
static std::atomic<size_t> scratch_pool_device_direct_live_bytes{0};

/// @brief Stage G：runtime scratch pool device-direct 存活字节高水位。
static std::atomic<size_t> scratch_pool_device_direct_peak_live_bytes{0};

/// @brief Stage G：runtime scratch pool 准入预算超限记录数。
static std::atomic<size_t> scratch_pool_budget_over_allocs{0};

/// @brief Stage G：enforce 模式下因 scratch pool 预算拒绝 device-direct 的次数。
static std::atomic<size_t> scratch_pool_budget_reject_allocs{0};

/// @brief Stage G：runtime scratch pool device-direct free 成功次数。
static std::atomic<size_t> scratch_pool_device_direct_free_success_allocs{0};

/// @brief Phase 7：scratch device-direct reuse cache 命中次数。
static std::atomic<size_t> scratch_pool_reuse_cache_hit_allocs{0};

/// @brief Phase 7：scratch device-direct reuse cache 未命中次数。
static std::atomic<size_t> scratch_pool_reuse_cache_miss_allocs{0};

/// @brief Phase 7：未命中后走新 cudaMallocAsync backend 的次数。
static std::atomic<size_t> scratch_pool_reuse_cache_fresh_backend_allocs{0};

/// @brief Phase 7：device-direct scratch 释放时被保留进 reuse cache 的次数。
static std::atomic<size_t> scratch_pool_reuse_cache_insert_allocs{0};

/// @brief Phase 7：device-direct scratch 释放时因容量/事件失败未保留的次数。
static std::atomic<size_t> scratch_pool_reuse_cache_insert_reject_allocs{0};

/// @brief Phase 7：reuse cache 命中累计复用字节数。
static std::atomic<size_t> scratch_pool_reuse_cache_hit_bytes{0};

/// @brief Phase 7：reuse cache 成功保留累计字节数。
static std::atomic<size_t> scratch_pool_reuse_cache_insert_bytes{0};

/// @brief Phase 7：当前 cache 中保留的 device-direct 字节数。
static std::atomic<size_t> scratch_pool_reuse_cache_cached_bytes{0};

/// @brief Phase 7：cache 保留字节高水位。
static std::atomic<size_t> scratch_pool_reuse_cache_peak_cached_bytes{0};

/// @brief Phase 7：当前 cache 中保留的条目数。
static std::atomic<size_t> scratch_pool_reuse_cache_cached_entries{0};

/// @brief Phase 7：cache 条目事件尚未完成导致不能复用的次数。
static std::atomic<size_t> scratch_pool_reuse_cache_event_not_ready{0};

/// @brief Phase 7：cache 条目事件查询/记录失败次数。
static std::atomic<size_t> scratch_pool_reuse_cache_event_errors{0};

/**
 * @struct AllocationInfo
 * @brief UVM 显存分配的元数据生命周期快照 (Metadata Lifecycle Snapshot)
 * * 此结构体充当每个 UVM 显存块的“数字档案”。它在内存分配 (cudaMallocManaged) 时被创建，
 * 记录了从底层物理属性到上层业务意图（如 KV Cache、Weights）的完整映射关系。
 * 这些详尽的追踪数据不仅用于运行时的动态策略干预（Gap Watch/Device Direct），
 * 更是离线分析 CPU-GPU 耦合瓶颈和生成性能重放日志的核心数据源。
 */
struct AllocationInfo {
    // ========================================================================
    // 1. 基础物理与生命周期元数据 (Base Physical & Lifecycle Metadata)
    // ========================================================================

    size_t size;                    ///< 分配的内存字节数。
    int device;                     ///< 目标设备的 ID（>=0 为特定 GPU，-1 通常代表 CPU/主机内存）。
    size_t alloc_id;                ///< 全局单调递增的分配序号，用于在海量日志中精确定位单次申请。
    std::string phase;              ///< 分配发生时上层系统（如 vLLM）所处的业务阶段（例如 "load_model", "initialize_kv_cache"）。
    double alloc_elapsed_seconds;   ///< 相对于调度器/拦截器启动时间的时间戳。配合 free 时的相对时间，可精确计算内存生命周期。
    std::string size_bucket;        ///< 将离散的 size 映射到预定义的区间（如 "1MiB-2MiB"），方便进行显存碎片率或分配频率的直方图统计。

    // ========================================================================
    // 2. 启发式意图与策略引擎状态 (Heuristic Intent & Policy Engine State)
    // ========================================================================

    std::string alloc_class_name;   ///< 启发式分类器推断出的内存用途类别（如 "WeightPersistent", "RuntimeScratch"）。
    std::string policy_action_name; ///< 最终决定执行的主动干预动作（如 "ManagedPrefetchGpu", "ManagedDefault"）。
    std::string policy_source_name; ///< 触发该动作的策略来源（"base_policy" 默认启发式，或 "gap_watch_policy" 动态热点干预）。
    bool policy_action_success;     ///< 记录 cudaMemAdvise 或 cudaMemPrefetchAsync 是否调用成功。
    std::string policy_action_error;///< 如果策略执行失败（如流冲突或设备不支持），记录具体的 CUDA 错误字符串。

    // ========================================================================
    // 3. 动态热点观测与隔离 (Dynamic Gap Watch & Hotspot Tracking)
    // ========================================================================

    bool unknown_detail_logged;     ///< 标志位：是否已将此未知内存的详细信息落盘，防止日志风暴。
    bool gap_watch_logged;          ///< 标志位：是否已被 Gap Watch 机制捕获并记录。
    uintptr_t gap_overlap_start;    ///< 该内存块与目标观测区间 (Gap) 发生重叠的起始绝对虚拟地址。
    uintptr_t gap_overlap_end;      ///< 重叠区域的结束地址。
    size_t gap_overlap_bytes;       ///< 实际落在观测热点区间内的字节数，用于计算重叠率 (Overlap Ratio)。
    std::string gap_watch_target_class_name;  ///< 触发监控时，Gap Watch 所关注的目标内存类别。
    std::string gap_watch_policy_action_name; ///< Gap Watch 机制为此热点区域尝试执行的提权动作（如强制预取）。
    bool hot_gap_match;             ///< 核心高频标志：该分配是否完美命中了当前配置的活跃热点区域。

    // ========================================================================
    // 4. 后端路由与直接内存分配演进 (Backend Routing & Device Direct Stage)
    // 用于量化 CPU-GPU 耦合问题，并在必要时绕过 UVM 直接分配纯显存
    // ========================================================================

    std::string placement_backend_name; ///< 实际使用的分配后端（当前大多为 "managed"，为演进到 "device_only" 预留）。
    std::string device_direct_backend_name; ///< device_direct 实际使用的 CUDA 分配后端，如 cuda_malloc 或 cuda_malloc_async。
    bool device_direct_eligible;    ///< 评估此内存块是否具备绕过 UVM、直接使用 cudaMalloc 的资格（取决于大小和生命周期）。
    std::string device_direct_reason; ///< 判定是否具备绕过资格的具体原因（如 "below_min_bytes", "target_class_mismatch"）。
    std::string cpu_access_risk;    ///< 风险评估标签：如果将其强制放置在 GPU 上，未来被 CPU 访问触发严重 Page Fault 的风险等级。
    bool kv_budget_tracked;         ///< Stage D：此分配是否纳入 KV 独立预算遥测。
    bool kv_budget_over_budget;     ///< Stage D：记录分配完成后 KV live bytes 是否超过预算。
    std::string kv_budget_reason;   ///< Stage D：KV budget 判定原因。
    bool weight_budget_tracked;      ///< Stage E：此分配是否纳入 weights 独立预算遥测。
    bool weight_budget_over_budget;  ///< Stage E：记录分配完成后 weight live bytes 是否超过预算。
    std::string weight_budget_reason;///< Stage E：weight budget 判定原因。
    bool weight_budget_cpu_offload_issued; ///< Stage G：越界 weight 是否已请求迁到 CPU。
    bool weight_budget_cpu_offload_success; ///< Stage G：CPU offload CUDA API 是否成功。
    std::string weight_budget_cpu_offload_error; ///< Stage G：CPU offload 失败原因。
    bool pool_registry_tracked;      ///< Stage F：此分配是否纳入统一 pool registry。
    std::string pool_kind_name;      ///< Stage F：统一 pool 名称，如 kv_cache/weights/runtime_scratch。
    std::string pool_object_state;   ///< Stage F：对象状态，当前仅 telemetry active/freed。
    bool driver_scratch_region_registered; ///< Stage N：是否已向 UVM driver 注册 scratch tag。
    uint64_t driver_scratch_region_id; ///< Stage N：scratch driver region id。
    int driver_scratch_register_status; ///< Stage N：driver register 返回值。
    bool driver_scratch_budget_reserved; ///< Stage N：是否占用了 driver scratch tag 预算。
    bool scratch_pool_tracked;       ///< Stage G：此分配是否参与 scratch pool admission。
    bool scratch_pool_eligible;      ///< Stage G：是否满足 scratch pool 独立准入。
    bool scratch_pool_device_direct; ///< Stage G：是否实际进入 scratch pool device-direct。
    bool scratch_pool_budget_over_budget; ///< Stage G：准入时是否超过 scratch pool 预算。
    std::string scratch_pool_reason; ///< Stage G：scratch pool 准入/回退原因。
};
static std::unordered_map<void*, AllocationInfo> active_allocations;

enum class AllocationClass {
    /**
     * 1. 模型权重 (Model Weights)
     * 特点：只读、生命周期极长（随模型加载而存在，随程序结束而释放）。
     * 策略：由于每一轮推理（Forward Pass）都要读取全部权重，这类内存绝对不能被换出到 CPU，
     * 否则会因为 PCIe 带宽瓶颈导致吞吐量断崖式下跌。
     */
    WeightPersistent,

    /**
     * 2. KV 缓存 (Key-Value Cache)
     * 特点：读写频繁、空间占用最大、生命周期长。
     * 策略：这是 vLLM 的核心显存占用者。
     */
    KvPersistent,

    /**
     * 3. 预热期工作区 (Warmup/Profile Workspace)
     * 特点：在模型启动阶段（Warmup）大量产生，用于算子自动调优 (Autotune) 或初始状态设置。
     */
    WarmupWorkspace,

    /**
     * 4. 运行时临时碎片 (Runtime Scratch)
     * 特点：极小（通常 < 1MB）、存活时间极短（可能一个算子执行完就释放）。
     * 策略：这类内存分配非常频繁。由于其尺寸小且释放快，通常不需要进行 UVM 预取，
     */
    RuntimeScratch,

    /**
     * 5. 运行时计算空间 (Runtime Workspace)
     * 特点：中等或大尺寸（如中间激活值 Activations、Logits）、存活于单次推理迭代中。
     * 策略：这些是计算时的“草稿纸”。它们在推理时需要极高的读写速度，但在两次推理请求之间
     */
    RuntimeWorkspace,

    /**
     * 6. 未知托管内存 (Unknown Managed)
     * 特点：未能命中任何启发式规则的分配请求。
     * 策略：作为兜底类别。如果日志中大量出现此类别，说明现有的 classify_allocation
     * 匹配规则（如 Phase 名称、Size 阈值）需要更新。
     */
    UnknownManaged,
};

enum class PolicyAction {
    ManagedDefault,//默认动作
    ManagedPrefetchGpu,//为阶段 1 预留,阶段 0 默认不会触发
    ManagedAdvisePrefetchGpu,//gap watch 的更激进版本：advise + prefetch
    DeviceDirectTrace,//阶段 B：只记录 would-use-device-direct，不改变分配行为
    DeviceDirect,//阶段 B 默认仍 trace-only，阶段 C 才会真正切换 GPU-only backend
};

static const char* allocation_class_to_string(AllocationClass alloc_class);

enum class PoolKind {
    KvCache,
    Weights,
    RuntimeScratch,
    OtherManaged,
};

static const char* pool_kind_to_string(PoolKind pool_kind);

/**
 * Get current timestamp string
 */
static void get_timestamp(char* buffer, size_t size) {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    struct tm* tm_info = localtime(&time_t_now);
    size_t len = strftime(buffer, size, "%Y-%m-%d %H:%M:%S", tm_info);
    snprintf(buffer + len, size - len, ".%03ld", (long)ms.count());
}

/**
 * Get elapsed time since start in seconds
 */
static double get_elapsed_seconds() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start_time).count();
}

static size_t read_trace_min_bytes_from_env() {
    const char* raw = getenv("VLLM_UVM_TRACE_MIN_BYTES");
    if (!raw || !*raw) {
        return trace_min_bytes;
    }

    char* end_ptr = nullptr;
    unsigned long long parsed = strtoull(raw, &end_ptr, 10);
    if (end_ptr == raw || parsed == 0) {
        return trace_min_bytes;
    }
    return static_cast<size_t>(parsed);
}

static bool read_bool_from_env(const char* name, bool default_value) {
    const char* raw = getenv(name);
    if (!raw || !*raw) {
        return default_value;
    }
    return strcmp(raw, "1") == 0 || strcasecmp(raw, "true") == 0 ||
           strcasecmp(raw, "yes") == 0 || strcasecmp(raw, "on") == 0;
}

static size_t read_size_from_env(const char* name, size_t default_value) {
    const char* raw = getenv(name);
    if (!raw || !*raw) {
        return default_value;
    }

    char* end_ptr = nullptr;
    unsigned long long parsed = strtoull(raw, &end_ptr, 10);
    if (end_ptr == raw || parsed == 0) {
        return default_value;
    }
    return static_cast<size_t>(parsed);
}

static size_t read_size_from_env_allow_zero(const char* name, size_t default_value) {
    const char* raw = getenv(name);
    if (!raw || !*raw) {
        return default_value;
    }

    char* end_ptr = nullptr;
    unsigned long long parsed = strtoull(raw, &end_ptr, 10);
    if (end_ptr == raw || *end_ptr != '\0') {
        return default_value;
    }
    return static_cast<size_t>(parsed);
}

static bool read_hex_u64_from_env(const char* name, uintptr_t* value_out) {
    const char* raw = getenv(name);
    if (!raw || !*raw || !value_out) {
        return false;
    }

    char* end_ptr = nullptr;
    unsigned long long parsed = strtoull(raw, &end_ptr, 0);
    if (end_ptr == raw || *end_ptr != '\0') {
        return false;
    }

    *value_out = static_cast<uintptr_t>(parsed);
    return true;
}

static uint64_t stat_mtime_ns(const struct stat& st) {
    return static_cast<uint64_t>(st.st_mtim.tv_sec) * 1000000000ULL +
           static_cast<uint64_t>(st.st_mtim.tv_nsec);
}

static std::string read_string_from_env(const char* name,
                                        const char* default_value) {
    const char* raw = getenv(name);
    if (!raw || !*raw) {
        return default_value;
    }
    return raw;
}

static std::string lower_copy(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

static bool string_equals_ignore_case(const std::string& left,
                                      const std::string& right) {
    return lower_copy(left) == lower_copy(right);
}

static bool phase_contains(const std::string& phase, const char* needle) {
    return phase.find(needle) != std::string::npos;
}

static bool phase_starts_with(const std::string& phase,
                              const std::string& prefix) {
    return phase.rfind(prefix, 0) == 0;
}

static std::string trim_copy(const std::string& input) {
    size_t start = 0;
    while (start < input.size() &&
           (input[start] == ' ' || input[start] == '\t' ||
            input[start] == '\r' || input[start] == '\n')) {
        ++start;
    }

    size_t end = input.size();
    while (end > start &&
           (input[end - 1] == ' ' || input[end - 1] == '\t' ||
            input[end - 1] == '\r' || input[end - 1] == '\n')) {
        --end;
    }

    return input.substr(start, end - start);
}

static std::string normalize_device_direct_backend(const std::string& value) {
    std::string lowered = lower_copy(trim_copy(value));
    if (lowered == "cuda_malloc_async" || lowered == "cuda_async") {
        return "cuda_malloc_async";
    }
    return "cuda_malloc";
}

static std::string normalize_kv_budget_mode(const std::string& value) {
    std::string lowered = lower_copy(trim_copy(value));
    if (lowered == "enforce" || lowered == "soft_enforce") {
        return "enforce";
    }
    return "trace_only";
}

static std::string normalize_weight_budget_mode(const std::string& value) {
    std::string lowered = lower_copy(trim_copy(value));
    if (lowered == "enforce" || lowered == "soft_enforce") {
        return "enforce";
    }
    return "trace_only";
}

static std::string normalize_scratch_pool_mode(const std::string& value) {
    std::string lowered = lower_copy(trim_copy(value));
    if (lowered == "enforce" || lowered == "soft_enforce") {
        return "enforce";
    }
    return "trace_only";
}

static bool configure_device_direct_async_pool_if_needed(int device) {
    if (!device_direct_pool_release_threshold_set ||
        device_direct_backend != "cuda_malloc_async") {
        return true;
    }

    std::lock_guard<std::mutex> lock(device_direct_pool_mutex);
    if (device_direct_pool_config_attempted &&
        device_direct_pool_config_device == device) {
        return device_direct_pool_config_success;
    }

    device_direct_pool_config_attempted = true;
    device_direct_pool_config_success = false;
    device_direct_pool_config_device = device;

    cudaMemPool_t pool = nullptr;
    cudaError_t err = cudaDeviceGetDefaultMemPool(&pool, device);
    if (err != cudaSuccess) {
        device_direct_pool_config_error = cudaGetErrorString(err);
        return false;
    }

    uint64_t threshold =
        static_cast<uint64_t>(device_direct_pool_release_threshold);
    err = cudaMemPoolSetAttribute(
        pool,
        cudaMemPoolAttrReleaseThreshold,
        &threshold
    );
    if (err != cudaSuccess) {
        device_direct_pool_config_error = cudaGetErrorString(err);
        return false;
    }

    device_direct_pool_config_success = true;
    device_direct_pool_config_error = "none";
    return true;
}

static bool comma_list_contains_phase_prefix(const std::string& csv,
                                             const std::string& phase) {
    if (csv.empty() || string_equals_ignore_case(csv, "any")) {
        return true;
    }

    size_t start = 0;
    while (start <= csv.size()) {
        size_t comma = csv.find(',', start);
        std::string token = trim_copy(
            csv.substr(
                start,
                comma == std::string::npos ? std::string::npos : comma - start
            )
        );
        if (!token.empty() && phase_starts_with(phase, token)) {
            return true;
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return false;
}

static bool is_device_direct_target_phase(const std::string& phase) {
    return comma_list_contains_phase_prefix(device_direct_target_phases, phase);
}

static bool is_scratch_pool_target_phase(const std::string& phase) {
    return comma_list_contains_phase_prefix(scratch_pool_target_phases, phase);
}

static bool parse_bool_string(const std::string& value, bool* out) {
    if (!out) {
        return false;
    }
    if (value == "1" || value == "true" || value == "True" ||
        value == "TRUE" || value == "yes" || value == "on") {
        *out = true;
        return true;
    }
    if (value == "0" || value == "false" || value == "False" ||
        value == "FALSE" || value == "no" || value == "off") {
        *out = false;
        return true;
    }
    return false;
}

static bool parse_policy_action_string(const std::string& value,
                                       PolicyAction* out) {
    if (!out) {
        return false;
    }
    std::string lowered = lower_copy(value);
    if (lowered == "observe" || lowered == "managed_default") {
        *out = PolicyAction::ManagedDefault;
        return true;
    }
    if (lowered == "prefetch" || lowered == "managed_prefetch_gpu") {
        *out = PolicyAction::ManagedPrefetchGpu;
        return true;
    }
    if (lowered == "advise_prefetch" ||
        lowered == "managed_advise_prefetch_gpu") {
        *out = PolicyAction::ManagedAdvisePrefetchGpu;
        return true;
    }
    if (lowered == "device_direct_trace") {
        *out = PolicyAction::DeviceDirectTrace;
        return true;
    }
    if (lowered == "device_direct") {
        *out = PolicyAction::DeviceDirect;
        return true;
    }
    return false;
}

static bool gap_watch_target_class_matches(AllocationClass alloc_class,
                                           const std::string& phase) {
    const char* alloc_class_name = allocation_class_to_string(alloc_class);
    if (!alloc_class_name || alloc_class_name[0] == '\0') {
        return false;
    }
    if (gap_watch_target_class.empty() ||
        string_equals_ignore_case(gap_watch_target_class, "any")) {
        return true;
    }
    if (string_equals_ignore_case(gap_watch_target_class, "gap_hot_runtime_scratch")) {
        return is_device_direct_target_phase(phase) &&
               (alloc_class == AllocationClass::UnknownManaged ||
                alloc_class == AllocationClass::RuntimeScratch ||
                alloc_class == AllocationClass::RuntimeWorkspace);
    }
    return string_equals_ignore_case(gap_watch_target_class, alloc_class_name);
}

static bool parse_size_string(const std::string& value, size_t* out) {
    if (!out) {
        return false;
    }

    char* end_ptr = nullptr;
    unsigned long long parsed = strtoull(value.c_str(), &end_ptr, 10);
    if (end_ptr == value.c_str() || *end_ptr != '\0') {
        return false;
    }

    *out = static_cast<size_t>(parsed);
    return true;
}

static bool parse_hex_string(const std::string& value, uintptr_t* out) {
    if (!out) {
        return false;
    }

    char* end_ptr = nullptr;
    unsigned long long parsed = strtoull(value.c_str(), &end_ptr, 0);
    if (end_ptr == value.c_str() || *end_ptr != '\0') {
        return false;
    }

    *out = static_cast<uintptr_t>(parsed);
    return true;
}

static const char* allocation_class_to_string(AllocationClass alloc_class) {
    switch (alloc_class) {
        case AllocationClass::WeightPersistent:
            return "weight_persistent";
        case AllocationClass::KvPersistent:
            return "kv_persistent";
        case AllocationClass::WarmupWorkspace:
            return "warmup_workspace";
        case AllocationClass::RuntimeScratch:
            return "runtime_scratch";
        case AllocationClass::RuntimeWorkspace:
            return "runtime_workspace";
        case AllocationClass::UnknownManaged:
        default:
            return "unknown_managed";
    }
}

static const char* policy_action_to_string(PolicyAction action) {
    switch (action) {
        case PolicyAction::DeviceDirect:
            return "device_direct";
        case PolicyAction::DeviceDirectTrace:
            return "device_direct_trace";
        case PolicyAction::ManagedAdvisePrefetchGpu:
            return "managed_advise_prefetch_gpu";
        case PolicyAction::ManagedPrefetchGpu:
            return "managed_prefetch_gpu";
        case PolicyAction::ManagedDefault:
        default:
            return "managed_default";
    }
}

static bool is_device_direct_action(PolicyAction action) {
    return action == PolicyAction::DeviceDirectTrace ||
           action == PolicyAction::DeviceDirect;
}

static bool is_kv_allocation(AllocationClass alloc_class) {
    return alloc_class == AllocationClass::KvPersistent;
}

static bool is_weight_allocation(AllocationClass alloc_class) {
    return alloc_class == AllocationClass::WeightPersistent;
}

static bool is_runtime_scratch_pool_allocation(AllocationClass alloc_class) {
    return alloc_class == AllocationClass::RuntimeScratch ||
           alloc_class == AllocationClass::RuntimeWorkspace ||
           alloc_class == AllocationClass::WarmupWorkspace;
}

static PoolKind pool_kind_for_allocation(AllocationClass alloc_class) {
    switch (alloc_class) {
        case AllocationClass::KvPersistent:
            return PoolKind::KvCache;
        case AllocationClass::WeightPersistent:
            return PoolKind::Weights;
        case AllocationClass::RuntimeScratch:
        case AllocationClass::RuntimeWorkspace:
        case AllocationClass::WarmupWorkspace:
            return PoolKind::RuntimeScratch;
        case AllocationClass::UnknownManaged:
        default:
            return PoolKind::OtherManaged;
    }
}

static const char* pool_kind_to_string(PoolKind pool_kind) {
    switch (pool_kind) {
        case PoolKind::KvCache:
            return "kv_cache";
        case PoolKind::Weights:
            return "weights";
        case PoolKind::RuntimeScratch:
            return "runtime_scratch";
        case PoolKind::OtherManaged:
        default:
            return "other_managed";
    }
}

static PoolKind pool_kind_from_string(const std::string& value) {
    if (value == "kv_cache") {
        return PoolKind::KvCache;
    }
    if (value == "weights") {
        return PoolKind::Weights;
    }
    if (value == "runtime_scratch") {
        return PoolKind::RuntimeScratch;
    }
    return PoolKind::OtherManaged;
}

static uint64_t allocator_region_id(void* ptr, size_t alloc_id) {
    uint64_t value = reinterpret_cast<uintptr_t>(ptr);
    value ^= (static_cast<uint64_t>(alloc_id) + 0x9e3779b97f4a7c15ULL +
              (value << 6) + (value >> 2));
    value &= 0x7FFFFFFFFFFFFFFFULL;
    return value == 0 ? 1 : value;
}

static void update_device_direct_peak_live(size_t current_live) {
    size_t peak = device_direct_peak_live_bytes.load();
    while (current_live > peak) {
        if (device_direct_peak_live_bytes.compare_exchange_weak(peak, current_live)) {
            break;
        }
    }
}

static bool reserve_device_direct_budget_with_limit(size_t size,
                                                    size_t max_total_bytes) {
    size_t live = device_direct_live_bytes.load();
    while (true) {
        if (max_total_bytes > 0 &&
            live > max_total_bytes - std::min(size, max_total_bytes)) {
            return false;
        }
        size_t next_live = live + size;
        if (next_live < live) {
            return false;
        }
        if (max_total_bytes > 0 && next_live > max_total_bytes) {
            return false;
        }
        if (device_direct_live_bytes.compare_exchange_weak(live, next_live)) {
            update_device_direct_peak_live(next_live);
            return true;
        }
    }
}

static bool reserve_device_direct_budget(size_t size) {
    return reserve_device_direct_budget_with_limit(
        size,
        device_direct_max_total_bytes
    );
}

static void release_device_direct_budget(size_t size) {
    size_t live = device_direct_live_bytes.load();
    while (true) {
        size_t next_live = live >= size ? live - size : 0;
        if (device_direct_live_bytes.compare_exchange_weak(live, next_live)) {
            return;
        }
    }
}

static size_t device_direct_budget_remaining_snapshot(size_t live) {
    if (device_direct_max_total_bytes == 0) {
        return 0;
    }
    return live >= device_direct_max_total_bytes
        ? 0
        : device_direct_max_total_bytes - live;
}

static void update_driver_scratch_peak_live(size_t current_live) {
    size_t peak = driver_scratch_peak_live_bytes.load();
    while (current_live > peak) {
        if (driver_scratch_peak_live_bytes.compare_exchange_weak(
                peak,
                current_live
            )) {
            break;
        }
    }
}

static bool reserve_driver_scratch_budget(size_t size,
                                          bool enforce_budget,
                                          bool* over_budget_out,
                                          size_t* live_before_out,
                                          size_t* live_after_out) {
    size_t live = driver_scratch_live_bytes.load();
    while (true) {
        if (live_before_out) {
            *live_before_out = live;
        }

        size_t next_live = live + size;
        if (next_live < live) {
            if (over_budget_out) {
                *over_budget_out = true;
            }
            if (live_after_out) {
                *live_after_out = live;
            }
            return false;
        }

        bool over_budget =
            driver_scratch_budget_bytes > 0 &&
            next_live > driver_scratch_budget_bytes;
        if (over_budget_out) {
            *over_budget_out = over_budget;
        }
        if (live_after_out) {
            *live_after_out = next_live;
        }

        if (over_budget) {
            driver_scratch_budget_over_allocs.fetch_add(1);
            if (enforce_budget) {
                driver_scratch_budget_reject_allocs.fetch_add(1);
                return false;
            }
        }

        if (driver_scratch_live_bytes.compare_exchange_weak(live, next_live)) {
            update_driver_scratch_peak_live(next_live);
            return true;
        }
    }
}

static void release_driver_scratch_budget(size_t size) {
    size_t live = driver_scratch_live_bytes.load();
    while (true) {
        size_t next_live = live >= size ? live - size : 0;
        if (driver_scratch_live_bytes.compare_exchange_weak(live, next_live)) {
            driver_scratch_budget_release_allocs.fetch_add(1);
            return;
        }
    }
}

static void update_kv_peak_live(size_t current_live) {
    size_t peak = kv_peak_live_bytes.load();
    while (current_live > peak) {
        if (kv_peak_live_bytes.compare_exchange_weak(peak, current_live)) {
            break;
        }
    }
}

static size_t kv_budget_remaining_snapshot(size_t live) {
    if (kv_budget_bytes == 0) {
        return 0;
    }
    return live >= kv_budget_bytes ? 0 : kv_budget_bytes - live;
}

static void record_kv_allocation(size_t size,
                                 bool* over_budget_out,
                                 const char** reason_out) {
    kv_trace_allocs.fetch_add(1);
    kv_requested_bytes.fetch_add(size);

    size_t current_live = kv_live_bytes.fetch_add(size) + size;
    update_kv_peak_live(current_live);

    bool over_budget = kv_budget_bytes > 0 && current_live > kv_budget_bytes;
    if (over_budget) {
        kv_budget_over_allocs.fetch_add(1);
        if (kv_budget_mode == "enforce") {
            kv_budget_reject_allocs.fetch_add(1);
        }
    }

    if (over_budget_out) {
        *over_budget_out = over_budget;
    }
    if (reason_out) {
        if (kv_budget_bytes == 0) {
            *reason_out = "kv_budget_unlimited";
        } else if (!over_budget) {
            *reason_out = "kv_budget_within_budget";
        } else if (kv_budget_mode == "enforce") {
            // Stage D enforce is intentionally soft at allocator level. The
            // block manager must own real KV eviction/swap/recompute decisions.
            *reason_out = "kv_budget_exceeded_soft_enforce";
        } else {
            *reason_out = "kv_budget_exceeded_trace_only";
        }
    }
}

static void release_kv_budget(size_t size) {
    size_t live = kv_live_bytes.load();
    while (true) {
        size_t next_live = live >= size ? live - size : 0;
        if (kv_live_bytes.compare_exchange_weak(live, next_live)) {
            return;
        }
    }
}

static void update_weight_peak_live(size_t current_live) {
    size_t peak = weight_peak_live_bytes.load();
    while (current_live > peak) {
        if (weight_peak_live_bytes.compare_exchange_weak(peak, current_live)) {
            break;
        }
    }
}

static size_t weight_budget_remaining_snapshot(size_t live) {
    if (weight_budget_bytes == 0) {
        return 0;
    }
    return live >= weight_budget_bytes ? 0 : weight_budget_bytes - live;
}

static void record_weight_allocation(size_t size,
                                     bool* over_budget_out,
                                     const char** reason_out) {
    weight_trace_allocs.fetch_add(1);
    weight_requested_bytes.fetch_add(size);

    size_t current_live = weight_live_bytes.fetch_add(size) + size;
    update_weight_peak_live(current_live);

    bool over_budget =
        weight_budget_bytes > 0 && current_live > weight_budget_bytes;
    if (over_budget) {
        weight_budget_over_allocs.fetch_add(1);
        if (weight_budget_mode == "enforce") {
            weight_budget_reject_allocs.fetch_add(1);
        }
    }

    if (over_budget_out) {
        *over_budget_out = over_budget;
    }
    if (reason_out) {
        if (weight_budget_bytes == 0) {
            *reason_out = "weight_budget_unlimited";
        } else if (!over_budget) {
            *reason_out = "weight_budget_within_budget";
        } else if (weight_budget_mode == "enforce") {
            // Stage E only emits an allocator-side soft signal. Real weight
            // offload/eviction must be owned by later model-loader/runtime work.
            *reason_out = "weight_budget_exceeded_soft_enforce";
        } else {
            *reason_out = "weight_budget_exceeded_trace_only";
        }
    }
}

static void release_weight_budget(size_t size) {
    size_t live = weight_live_bytes.load();
    while (true) {
        size_t next_live = live >= size ? live - size : 0;
        if (weight_live_bytes.compare_exchange_weak(live, next_live)) {
            return;
        }
    }
}

static bool offload_managed_range_to_cpu(void* ptr,
                                         size_t size,
                                         cudaStream_t stream,
                                         const char** error_out) {
    if (error_out) {
        *error_out = "none";
    }
    if (ptr == nullptr || size == 0) {
        if (error_out) {
            *error_out = "empty_range";
        }
        return false;
    }

    cudaError_t advise_err = cudaMemAdvise(
        ptr,
        size,
        cudaMemAdviseSetPreferredLocation,
        cudaCpuDeviceId
    );
    if (advise_err != cudaSuccess) {
        if (error_out) {
            *error_out = cudaGetErrorString(advise_err);
        }
        return false;
    }

    cudaError_t prefetch_err = cudaMemPrefetchAsync(
        ptr,
        size,
        cudaCpuDeviceId,
        stream
    );
    if (prefetch_err != cudaSuccess) {
        if (error_out) {
            *error_out = cudaGetErrorString(prefetch_err);
        }
        return false;
    }

    return true;
}

static void update_pool_peak_live(std::atomic<size_t>& peak_counter,
                                  size_t current_live) {
    size_t peak = peak_counter.load();
    while (current_live > peak) {
        if (peak_counter.compare_exchange_weak(peak, current_live)) {
            break;
        }
    }
}

static void release_pool_live_bytes(std::atomic<size_t>& live_counter,
                                    size_t size) {
    size_t live = live_counter.load();
    while (true) {
        size_t next_live = live >= size ? live - size : 0;
        if (live_counter.compare_exchange_weak(live, next_live)) {
            return;
        }
    }
}

static void record_pool_registry_allocation(PoolKind pool_kind, size_t size) {
    pool_registry_tracked_allocs.fetch_add(1);
    size_t live_objects = pool_registry_live_objects.fetch_add(1) + 1;
    update_pool_peak_live(pool_registry_peak_live_objects, live_objects);

    switch (pool_kind) {
        case PoolKind::KvCache: {
            pool_kv_allocs.fetch_add(1);
            pool_kv_requested_bytes.fetch_add(size);
            size_t current_live = pool_kv_live_bytes.fetch_add(size) + size;
            update_pool_peak_live(pool_kv_peak_live_bytes, current_live);
            break;
        }
        case PoolKind::Weights: {
            pool_weight_allocs.fetch_add(1);
            pool_weight_requested_bytes.fetch_add(size);
            size_t current_live = pool_weight_live_bytes.fetch_add(size) + size;
            update_pool_peak_live(pool_weight_peak_live_bytes, current_live);
            break;
        }
        case PoolKind::RuntimeScratch: {
            pool_runtime_scratch_allocs.fetch_add(1);
            pool_runtime_scratch_requested_bytes.fetch_add(size);
            size_t current_live =
                pool_runtime_scratch_live_bytes.fetch_add(size) + size;
            update_pool_peak_live(
                pool_runtime_scratch_peak_live_bytes,
                current_live
            );
            break;
        }
        case PoolKind::OtherManaged:
        default:
            break;
    }
}

static void release_pool_registry_allocation(PoolKind pool_kind, size_t size) {
    size_t live_objects = pool_registry_live_objects.load();
    while (true) {
        size_t next_live = live_objects > 0 ? live_objects - 1 : 0;
        if (pool_registry_live_objects.compare_exchange_weak(
                live_objects,
                next_live
            )) {
            break;
        }
    }
    pool_registry_free_success_allocs.fetch_add(1);

    switch (pool_kind) {
        case PoolKind::KvCache:
            release_pool_live_bytes(pool_kv_live_bytes, size);
            pool_kv_free_success_allocs.fetch_add(1);
            break;
        case PoolKind::Weights:
            release_pool_live_bytes(pool_weight_live_bytes, size);
            pool_weight_free_success_allocs.fetch_add(1);
            break;
        case PoolKind::RuntimeScratch:
            release_pool_live_bytes(pool_runtime_scratch_live_bytes, size);
            pool_runtime_scratch_free_success_allocs.fetch_add(1);
            break;
        case PoolKind::OtherManaged:
        default:
            break;
    }
}

static void update_scratch_pool_device_direct_peak_live(size_t current_live) {
    size_t peak = scratch_pool_device_direct_peak_live_bytes.load();
    while (current_live > peak) {
        if (scratch_pool_device_direct_peak_live_bytes.compare_exchange_weak(
                peak,
                current_live
            )) {
            break;
        }
    }
}

static bool reserve_scratch_pool_device_direct_budget(size_t size,
                                                      bool* over_budget_out) {
    size_t live = scratch_pool_device_direct_live_bytes.load();
    while (true) {
        size_t next_live = live + size;
        if (next_live < live) {
            if (over_budget_out) {
                *over_budget_out = true;
            }
            return false;
        }
        bool over_budget =
            scratch_pool_budget_bytes > 0 && next_live > scratch_pool_budget_bytes;
        if (over_budget_out) {
            *over_budget_out = over_budget;
        }
        if (over_budget && scratch_pool_mode == "enforce") {
            return false;
        }
        if (scratch_pool_device_direct_live_bytes.compare_exchange_weak(
                live,
                next_live
            )) {
            update_scratch_pool_device_direct_peak_live(next_live);
            return true;
        }
    }
}

static void release_scratch_pool_device_direct_budget(size_t size) {
    size_t live = scratch_pool_device_direct_live_bytes.load();
    while (true) {
        size_t next_live = live >= size ? live - size : 0;
        if (scratch_pool_device_direct_live_bytes.compare_exchange_weak(
                live,
                next_live
            )) {
            return;
        }
    }
}

static size_t scratch_pool_budget_remaining_snapshot(size_t live) {
    if (scratch_pool_budget_bytes == 0) {
        return 0;
    }
    return live >= scratch_pool_budget_bytes
        ? 0
        : scratch_pool_budget_bytes - live;
}

static void update_scratch_pool_reuse_cache_peak_cached(size_t current_cached) {
    size_t peak = scratch_pool_reuse_cache_peak_cached_bytes.load();
    while (current_cached > peak) {
        if (scratch_pool_reuse_cache_peak_cached_bytes.compare_exchange_weak(
                peak,
                current_cached
            )) {
            break;
        }
    }
}

static bool scratch_pool_reuse_cache_available(void) {
    return scratch_pool_reuse_cache_enable != 0 &&
           device_direct_backend == "cuda_malloc_async";
}

static bool try_pop_scratch_pool_reuse_cache(const char* size_bucket,
                                             size_t size,
                                             int device,
                                             ScratchPoolReuseCacheEntry* out_entry) {
    if (!out_entry ||
        !scratch_pool_reuse_cache_available() ||
        !size_bucket ||
        size == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(scratch_pool_reuse_cache_mutex);
    auto bucket_it = scratch_pool_reuse_cache.find(size_bucket);
    if (bucket_it == scratch_pool_reuse_cache.end()) {
        scratch_pool_reuse_cache_miss_allocs.fetch_add(1);
        return false;
    }

    auto& entries = bucket_it->second;
    for (size_t i = entries.size(); i > 0; --i) {
        size_t index = i - 1;
        ScratchPoolReuseCacheEntry& candidate = entries[index];
        if (candidate.size != size || candidate.device != device) {
            continue;
        }

        if (candidate.has_ready_event && candidate.ready_event != nullptr) {
            cudaError_t event_status = cudaEventQuery(candidate.ready_event);
            if (event_status == cudaErrorNotReady) {
                scratch_pool_reuse_cache_event_not_ready.fetch_add(1);
                continue;
            }
            if (event_status != cudaSuccess) {
                scratch_pool_reuse_cache_event_errors.fetch_add(1);
                continue;
            }
            cudaEventDestroy(candidate.ready_event);
            candidate.ready_event = nullptr;
            candidate.has_ready_event = false;
        }

        *out_entry = candidate;
        entries.erase(entries.begin() + index);
        if (entries.empty()) {
            scratch_pool_reuse_cache.erase(bucket_it);
        }

        scratch_pool_reuse_cache_current_bytes =
            scratch_pool_reuse_cache_current_bytes >= size
                ? scratch_pool_reuse_cache_current_bytes - size
                : 0;
        scratch_pool_reuse_cache_current_entries =
            scratch_pool_reuse_cache_current_entries > 0
                ? scratch_pool_reuse_cache_current_entries - 1
                : 0;
        scratch_pool_reuse_cache_cached_bytes.store(
            scratch_pool_reuse_cache_current_bytes
        );
        scratch_pool_reuse_cache_cached_entries.store(
            scratch_pool_reuse_cache_current_entries
        );
        scratch_pool_reuse_cache_hit_allocs.fetch_add(1);
        scratch_pool_reuse_cache_hit_bytes.fetch_add(size);
        return true;
    }

    scratch_pool_reuse_cache_miss_allocs.fetch_add(1);
    return false;
}

static bool try_insert_scratch_pool_reuse_cache(void* ptr,
                                                size_t size,
                                                int device,
                                                const std::string& size_bucket,
                                                cudaStream_t stream) {
    if (!ptr ||
        !scratch_pool_reuse_cache_available() ||
        size == 0 ||
        size_bucket.empty() ||
        scratch_pool_reuse_cache_max_bytes == 0 ||
        scratch_pool_reuse_cache_max_entries_per_bucket == 0) {
        return false;
    }

    cudaEvent_t ready_event = nullptr;
    cudaError_t event_err = cudaEventCreateWithFlags(
        &ready_event,
        cudaEventDisableTiming
    );
    if (event_err != cudaSuccess) {
        scratch_pool_reuse_cache_event_errors.fetch_add(1);
        scratch_pool_reuse_cache_insert_reject_allocs.fetch_add(1);
        return false;
    }
    event_err = cudaEventRecord(ready_event, stream);
    if (event_err != cudaSuccess) {
        cudaEventDestroy(ready_event);
        scratch_pool_reuse_cache_event_errors.fetch_add(1);
        scratch_pool_reuse_cache_insert_reject_allocs.fetch_add(1);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(scratch_pool_reuse_cache_mutex);
        auto& entries = scratch_pool_reuse_cache[size_bucket];
        bool bucket_full =
            entries.size() >= scratch_pool_reuse_cache_max_entries_per_bucket;
        bool byte_full =
            size > scratch_pool_reuse_cache_max_bytes ||
            scratch_pool_reuse_cache_current_bytes >
                scratch_pool_reuse_cache_max_bytes - size;
        if (bucket_full || byte_full) {
            cudaEventDestroy(ready_event);
            scratch_pool_reuse_cache_insert_reject_allocs.fetch_add(1);
            return false;
        }

        entries.push_back(ScratchPoolReuseCacheEntry{
            ptr,
            size,
            device,
            size_bucket,
            ready_event,
            true,
        });
        scratch_pool_reuse_cache_current_bytes += size;
        scratch_pool_reuse_cache_current_entries += 1;
        scratch_pool_reuse_cache_cached_bytes.store(
            scratch_pool_reuse_cache_current_bytes
        );
        scratch_pool_reuse_cache_cached_entries.store(
            scratch_pool_reuse_cache_current_entries
        );
        update_scratch_pool_reuse_cache_peak_cached(
            scratch_pool_reuse_cache_current_bytes
        );
    }

    scratch_pool_reuse_cache_insert_allocs.fetch_add(1);
    scratch_pool_reuse_cache_insert_bytes.fetch_add(size);
    return true;
}

static void clear_scratch_pool_reuse_cache(void) {
    std::vector<ScratchPoolReuseCacheEntry> entries_to_free;
    {
        std::lock_guard<std::mutex> lock(scratch_pool_reuse_cache_mutex);
        for (auto& bucket : scratch_pool_reuse_cache) {
            for (auto& entry : bucket.second) {
                entries_to_free.push_back(std::move(entry));
            }
        }
        scratch_pool_reuse_cache.clear();
        scratch_pool_reuse_cache_current_bytes = 0;
        scratch_pool_reuse_cache_current_entries = 0;
        scratch_pool_reuse_cache_cached_bytes.store(0);
        scratch_pool_reuse_cache_cached_entries.store(0);
    }

    int previous_device = -1;
    cudaError_t get_device_err = cudaGetDevice(&previous_device);
    for (auto& entry : entries_to_free) {
        if (entry.has_ready_event && entry.ready_event != nullptr) {
            cudaEventSynchronize(entry.ready_event);
            cudaEventDestroy(entry.ready_event);
            entry.ready_event = nullptr;
            entry.has_ready_event = false;
        }
        if (entry.device >= 0) {
            cudaSetDevice(entry.device);
        }
        cudaError_t free_err = cudaFreeAsync(entry.ptr, 0);
        if (free_err == cudaSuccess) {
            cudaStreamSynchronize(0);
        } else {
            cudaFree(entry.ptr);
        }
    }
    if (get_device_err == cudaSuccess && previous_device >= 0) {
        cudaSetDevice(previous_device);
    }
}

static const char* size_bucket_for(size_t size) {
    if (size < 64 * 1024) return "<64KiB";
    if (size < 256 * 1024) return "64KiB-256KiB";
    if (size < 1 * 1024 * 1024) return "256KiB-1MiB";
    if (size < 2 * 1024 * 1024) return "1MiB-2MiB";
    if (size < 4 * 1024 * 1024) return "2MiB-4MiB";
    if (size < 8 * 1024 * 1024) return "4MiB-8MiB";
    if (size < 16 * 1024 * 1024) return "8MiB-16MiB";
    return ">=16MiB";
}

static uintptr_t region_end_for(uintptr_t start, size_t size) {
    if (size == 0) {
        return start;
    }
    return start + size - 1;
}

/**
 * @struct GapWatchConfigSnapshot
 * @brief 动态显存热点观测配置的“不可变快照” (Immutable Snapshot of Gap Watch Configuration)
 * * @details
 * 该结构体充当 Gap Watch 机制的控制平面 (Control Plane) 与数据平面 (Data Plane) 之间的契约。
 * 系统通过轮询控制文件 (Control File)，在后台将最新的监控意图打包成此快照。
 * 底层的 `uvm_malloc` / `uvm_free` 会直接基于此快照进行 O(1) 复杂度的重叠计算与策略路由，
 * 从而在实现“零侵入式策略热更新”的同时，避免在 CUDA 分配关键路径上引入严重的锁竞争。
 */
struct GapWatchConfigSnapshot {
    // ========================================================================
    // 1. 空间定位与全局开关 (Spatial Targeting & Master Switch)
    // ========================================================================

    /// @brief 全局拨测开关。若为 false，分配器将完全跳过地址重叠计算，以最低开销放行。
    bool enabled;

    /// @brief 监控区间的起始绝对地址（通常为虚拟地址）。
    /// @note 在 xCoord 等软硬协同架构中，这可能对应一段被频繁触发 CPU-GPU UVM 缺页抖动的热点物理内存映射区。
    uintptr_t start;

    /// @brief 监控区间的结束绝对地址。与 start 共同划定了一个“楚河汉界”。
    uintptr_t end;

    /// @brief 该观测任务的可读标识符（如 "layer_0_kv_cache_hotspot"）。
    /// @note 极具工程价值：它会被直接透传到详细日志中，方便与内核级 page fault
    /// 堆栈或 SysAI-Agent 的时序图谱进行无缝联合关联 (Join)。
    std::string name;

    // ========================================================================
    // 2. 启发式意图过滤器 (Heuristic Intent Filters)
    // 决定哪些内存请求有资格触发干预，避免“误杀”或陷入细碎内存的追踪泥潭。
    // ========================================================================

    /// @brief 泛类型匹配标志位。若为 true，则无视内存的具体业务语义（Class），拦截区间内所有分配。
    bool all_classes;

    /// @brief 尺寸噪声过滤器。只有单次申请大小 >= 此阈值（如 1MB）的内存块才会被监控。
    /// @note 过滤掉极高频但无足轻重的 RuntimeScratch 碎片，是保障分析信噪比的关键。
    size_t min_bytes;

    /// @brief 精准打击目标类别（当 all_classes = false 时生效）。
    /// @note 例如仅针对 "KvPersistent" (KV Cache) 执行特定的监控或迁移操作，
    /// 从而保护 "WeightPersistent" (只读权重) 不受影响。
    std::string target_class;

    // ========================================================================
    // 3. 提权干预动作 (Privilege Escalation Action)
    // ========================================================================

    /// @brief 命中观测区间且通过过滤器后，分配器需要强制重载（Override）的策略动作名称。
    /// @note 例如，原启发式规则认为这块内存该走 "ManagedDefault"，但快照指示
    /// 这里是高危抖动区，强制执行 "ManagedAdvisePrefetchGpu" 或尝试 "DeviceDirect"。
    std::string policy_action_name;
};

static GapWatchConfigSnapshot current_gap_watch_config_snapshot() {
    return GapWatchConfigSnapshot{
        gap_watch_enabled != 0,
        gap_watch_start,
        gap_watch_end,
        gap_watch_name,
        gap_watch_all_classes != 0,
        gap_watch_min_bytes,
        gap_watch_target_class,
        gap_watch_policy_action,
    };
}

static bool gap_watch_configs_equal(const GapWatchConfigSnapshot& left,
                                    const GapWatchConfigSnapshot& right) {
    return left.enabled == right.enabled &&
           left.start == right.start &&
           left.end == right.end &&
           left.name == right.name &&
           left.all_classes == right.all_classes &&
           left.min_bytes == right.min_bytes &&
           left.target_class == right.target_class &&
           left.policy_action_name == right.policy_action_name;
}

static void trace_gap_watch_config_event_locked(const char* source,
                                                const char* event,
                                                const GapWatchConfigSnapshot& config,
                                                const char* reason) {
    if (!log_file) {
        return;
    }

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();
    const char* safe_source =
        (source && source[0] != '\0') ? source : "unknown_source";
    const char* safe_event =
        (event && event[0] != '\0') ? event : "applied";
    const char* safe_reason =
        (reason && reason[0] != '\0') ? reason : "none";

    fprintf(
        log_file,
        "[%s] [+%.3fs] TRACE_GAP_WATCH_CONFIG source=%s event=%s "
        "enabled=%d name=%s start=0x%llx end=0x%llx all_classes=%d "
        "min_bytes=%zu target_class=%s policy_action=%s reason=%s\n",
        timestamp,
        elapsed,
        safe_source,
        safe_event,
        config.enabled ? 1 : 0,
        config.name.c_str(),
        static_cast<unsigned long long>(config.start),
        static_cast<unsigned long long>(config.end),
        config.all_classes ? 1 : 0,
        config.min_bytes,
        config.target_class.c_str(),
        config.policy_action_name.c_str(),
        safe_reason
    );
    fflush(log_file);
}

static void apply_gap_watch_config_locked(const GapWatchConfigSnapshot& config,
                                          const char* source,
                                          const char* reason) {
    GapWatchConfigSnapshot previous = current_gap_watch_config_snapshot();
    if (gap_watch_configs_equal(previous, config)) {
        return;
    }

    gap_watch_enabled = config.enabled ? 1 : 0;
    gap_watch_start = config.start;
    gap_watch_end = config.end;
    gap_watch_name = config.name.empty() ? "unnamed_gap" : config.name;
    gap_watch_all_classes = config.all_classes ? 1 : 0;
    gap_watch_min_bytes = config.min_bytes;
    gap_watch_target_class =
        config.target_class.empty() ? "any" : config.target_class;
    gap_watch_policy_action =
        config.policy_action_name.empty() ? "observe" : config.policy_action_name;

    GapWatchConfigSnapshot updated = current_gap_watch_config_snapshot();
    trace_gap_watch_config_event_locked(source, "applied", updated, reason);
}

static bool compute_overlap(uintptr_t start,
                            size_t size,
                            uintptr_t other_start,
                            uintptr_t other_end,
                            uintptr_t* overlap_start_out,
                            uintptr_t* overlap_end_out,
                            size_t* overlap_bytes_out) {
    if (size == 0 || other_end < other_start) {
        return false;
    }

    uintptr_t end = region_end_for(start, size);
    uintptr_t overlap_start = start > other_start ? start : other_start;
    uintptr_t overlap_end = end < other_end ? end : other_end;
    if (overlap_start > overlap_end) {
        return false;
    }

    if (overlap_start_out) *overlap_start_out = overlap_start;
    if (overlap_end_out) *overlap_end_out = overlap_end;
    if (overlap_bytes_out) *overlap_bytes_out = overlap_end - overlap_start + 1;
    return true;
}

static bool should_trace_unknown_detail(AllocationClass alloc_class,
                                        size_t size) {
    if (!unknown_detail_enabled) {
        return false;
    }
    if (alloc_class != AllocationClass::UnknownManaged) {
        return false;
    }
    return size >= unknown_detail_min_bytes;
}

static bool should_trace_gap_watch(AllocationClass alloc_class,
                                   size_t size,
                                   size_t overlap_bytes) {
    if (!gap_watch_enabled || overlap_bytes == 0) {
        return false;
    }
    if (size < gap_watch_min_bytes) {
        return false;
    }
    return gap_watch_all_classes ||
           alloc_class == AllocationClass::UnknownManaged;
}

/**
 * @brief 从外部控制文件动态刷新 Gap Watch 配置 (Dynamic Configuration Hot-reload)
 * * @details
 * 这是一个为了在不重启 vLLM/推理服务的前提下，动态调整 UVM 观测热点而设计的轮询引擎。
 * 由于此函数在 `uvm_malloc` 的关键路径中被调用，必须极力避免长尾延迟。
 * 设计上采用了“三级短路机制”来降低 IO 与系统调用开销：
 * 1. 软限流 (Time-based Throttling)：基于时钟，避免 ms 级别内的频繁 stat 调用。
 * 2. 元数据比对 (Stat Mtime Check)：对比文件修改时间和大小，内容未变则跳过解析。
 * 3. 内存快照 (Atomic Snapshot)：仅在配置真正变更时，才加锁更新全局状态。
 * * @param force 是否无视时间节流阀，强制执行一次文件系统探测（通常在初始化阶段置为 true）。
 */
static void refresh_gap_watch_from_control_file_if_needed(bool force) {
    // 短路检查 0：如果没有配置控制文件路径，直接退回默认状态
    if (gap_watch_control_file.empty()) {
        return;
    }

    auto now = std::chrono::steady_clock::now();

    // ========================================================================
    // 阶段 1：软限流机制 (Time-based Throttling)
    // 防止高并发显存申请时，每个线程都疯狂发起文件系统调用，导致内核态开销激增。
    // ========================================================================
    if (!force &&
        gap_watch_last_refresh_check != std::chrono::steady_clock::time_point::min()) {
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - gap_watch_last_refresh_check
        ).count();
        // 只有当距离上次检查的时间超过 gap_watch_refresh_ms (如 250ms) 时，才放行
        if (elapsed_ms >= 0 &&
            static_cast<size_t>(elapsed_ms) < gap_watch_refresh_ms) {
            return;
        }
    }
    // 更新最后检查时间戳（即使后续 stat 失败也需更新，避免密集重试）
    gap_watch_last_refresh_check = now;

    // ========================================================================
    // 阶段 2：文件元数据探测 (Filesystem Stat Fast-path)
    // ========================================================================
    struct stat st {};
    if (stat(gap_watch_control_file.c_str(), &st) != 0) {
        return; // 文件不存在或无权限，静默返回，保持现有配置
    }

    uint64_t mtime_ns = stat_mtime_ns(st);
    // 比对文件的 MTime 和 Size。只有当文件被实际修改过时，才进入昂贵的 fopen 和解析逻辑。
    if (!force && gap_watch_control_seen &&
        mtime_ns == gap_watch_control_mtime_ns &&
        st.st_size == gap_watch_control_size) {
        return;
    }

    // ========================================================================
    // 阶段 3：解析控制平面指令 (Control Plane Parsing)
    // ========================================================================
    FILE* control = fopen(gap_watch_control_file.c_str(), "r");
    if (!control) {
        return;
    }

    // 基于当前活跃的快照作为基准进行修改，保证未在文件中显式声明的字段继承旧值
    GapWatchConfigSnapshot next = current_gap_watch_config_snapshot();
    next.enabled = false;
    next.start = 0;
    next.end = 0;

    // 解析状态机变量
    bool saw_enabled = false;
    bool parsed_enabled = false;
    bool parsed_all_classes = false;
    bool parsed_name = false;
    bool parsed_start = false;
    bool parsed_end = false;
    bool parsed_min_bytes = false;
    bool parsed_target_class = false;
    bool parsed_policy_action = false;

    // 解析缓冲变量
    bool enabled_value = false;
    bool all_classes_value = gap_watch_all_classes != 0;
    uintptr_t start_value = 0;
    uintptr_t end_value = 0;
    size_t min_bytes_value = gap_watch_min_bytes;
    std::string name_value = gap_watch_name;
    std::string target_class_value = gap_watch_target_class;
    std::string policy_action_value = gap_watch_policy_action;
    char buffer[1024];

    // 逐行解析 K-V 格式的配置文件，支持 # 注释和空行
    while (fgets(buffer, sizeof(buffer), control) != nullptr) {
        std::string line = trim_copy(buffer);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            continue;
        }

        std::string key = trim_copy(line.substr(0, eq_pos));
        std::string value = trim_copy(line.substr(eq_pos + 1));

        // 路由解析指令到具体的转换函数 (支持 hex 地址、bool、size 等)
        if (key == "enabled") {
            saw_enabled = true;
            parsed_enabled = parse_bool_string(value, &enabled_value);
        } else if (key == "all_classes") {
            parsed_all_classes = parse_bool_string(value, &all_classes_value);
        } else if (key == "name") {
            name_value = value;
            parsed_name = true;
        } else if (key == "start") {
            parsed_start = parse_hex_string(value, &start_value);
        } else if (key == "end") {
            parsed_end = parse_hex_string(value, &end_value);
        } else if (key == "min_bytes") {
            parsed_min_bytes = parse_size_string(value, &min_bytes_value);
        } else if (key == "target_class") {
            target_class_value = value;
            parsed_target_class = true;
        } else if (key == "policy_action") {
            PolicyAction action{};
            parsed_policy_action = parse_policy_action_string(value, &action);
            if (parsed_policy_action) {
                policy_action_value = policy_action_to_string(action);
            }
        }
    }
    fclose(control);

    // 记录本次成功解析的元数据特征，用于下一次的短路检查
    gap_watch_control_seen = true;
    gap_watch_control_mtime_ns = mtime_ns;
    gap_watch_control_size = st.st_size;

    const char* reason = "control_file";
    // 强制校验：控制文件必须至少包含清晰的 enabled 字段，否则视为无效修改，丢弃变更
    if (!saw_enabled || !parsed_enabled) {
        return;
    }

    // ========================================================================
    // 阶段 4：配置快照构建与应用 (Snapshot Construction & Commit)
    // ========================================================================

    // 合并解析到的新值与系统旧值
    next.name = parsed_name ? name_value : gap_watch_name;
    next.all_classes = parsed_all_classes ? all_classes_value :
        (gap_watch_all_classes != 0);
    next.min_bytes = parsed_min_bytes ? min_bytes_value : gap_watch_min_bytes;
    next.target_class = parsed_target_class ? target_class_value :
        gap_watch_target_class;
    next.policy_action_name = parsed_policy_action ? policy_action_value :
        gap_watch_policy_action;

    // 场景 A：被用户手动置为 disabled，清空监控区间并提交
    if (!enabled_value) {
        next.enabled = false;
        next.start = 0;
        next.end = 0;
        std::lock_guard<std::mutex> lock(log_mutex);
        apply_gap_watch_config_locked(next, "control_file", "disabled");
        return;
    }

    // 场景 B：非法地址区间（如 end < start），自动执行熔断并降级为 disabled
    if (!parsed_start || !parsed_end || end_value < start_value) {
        next.enabled = false;
        next.start = 0;
        next.end = 0;
        std::lock_guard<std::mutex> lock(log_mutex);
        apply_gap_watch_config_locked(next, "control_file", "invalid_range");
        return;
    }

    // 场景 C：合法配置，正式装载新的监控边界
    next.enabled = true;
    next.start = start_value;
    next.end = end_value;

    // 获取日志与配置更新锁，原子性地提交给全局数据平面
    std::lock_guard<std::mutex> lock(log_mutex);
    apply_gap_watch_config_locked(next, "control_file", reason);
}

/**
 * 启发式分类器：将显存申请映射到具体的业务类别
 * * 这个函数实现了从“通用内存申请”到“业务意图”的转换。
 * 它依赖于上层 Python 代码通过 uvm_set_phase() 注入的上下文。
 * * @param phase 当前运行阶段的字符串标识（由 Python 侧设置）
 * @param size  申请的内存大小（字节）
 * @return AllocationClass 推断出的内存类别
 */
static AllocationClass classify_allocation(const std::string& phase,
                                           size_t size) {
    // 1. 模型加载阶段：判定为【持久权重】
    // 这里的内存通常包含 Transformer 的各层参数（Linear/Embedding 等）。
    // 这些内存一旦分配，在整个服务生命周期内都不会释放，且每一轮推理都要全量读取。
    if (phase == "load_model" || phase == "fixed_weight_device_direct") {
        return AllocationClass::WeightPersistent;
    }

    // 2. KV Cache 初始化阶段：判定为【持久缓存】
    // vLLM 通常会预先申请一个巨大的显存池（Block Manager 管理的空间）。
    // 这是 UVM 调优的重灾区，决定了最大并发量。
    if (phase == "initialize_kv_cache") {
        return AllocationClass::KvPersistent;
    }

    // 3. 预热/预处理阶段：判定为【启动工作区】
    // 包含 Autotuning（寻找算子最优配置）、Profile（试运行）等产生的临时内存。
    // 特点是：在启动时很忙，但模型正式跑起来（Inference）后基本就没用了。
    if (phase_contains(phase, "warmup") ||
        phase_contains(phase, "autotune") ||
        phase_contains(phase, "preinit") ||
        phase == "profile_run") {
        return AllocationClass::WarmupWorkspace;
    }

    // 4. 正式推理阶段：根据大小进一步细分
    // 当 phase 被设置为 "enabled" 时，代表模型已经开始处理用户请求。
    // 此时无法仅凭阶段名区分意图，必须引入【尺寸启发式 (Size Heuristics)】。
    if (phase == "enabled" || phase.rfind("enabled:", 0) == 0) {
        // 4.1 运行时临时碎片 (1MB ~ 16MB)
        // 通常是一些算子的中间极短命输出，或者很小的临时变量。
        if (size >= 1 * 1024 * 1024 && size <= 16 * 1024 * 1024) {
            return AllocationClass::RuntimeScratch;
        }

        // 4.2 运行时工作空间 (16MB ~ 128MB)
        // 可能是中间层的激活值 (Activations) 或 Logits。
        // 这些内存会在每一轮推理迭代中循环申请和释放。
        if (size > 16 * 1024 * 1024 && size <= 128 * 1024 * 1024) {
            return AllocationClass::RuntimeWorkspace;
        }
    }

    // 5. 兜底逻辑：无法识别的托管内存
    // 如果日志中出现大量 UnknownManaged，说明你可能发现了一个新的 vLLM 内部阶段，
    // 需要回来更新这个“分拣规则”。
    return AllocationClass::UnknownManaged;
}

/**
 * 策略选择器：决定对当前内存块执行何种管理动作
 * * 这是将“分类结果”转化为“具体指令”的转换逻辑。
 * * @param alloc_class 之前由 classify_allocation 识别出的内存类别
 * @param size        内存块大小
 * @param device      目标设备 ID（>=0 为 GPU，<0 通常指 CPU）
 * @return PolicyAction 最终决定的动作类型
 */
static PolicyAction choose_policy_action(AllocationClass alloc_class,
                                         size_t size,
                                         int device) {
    // 1. 准入检查：全局开关与设备合法性
    // 如果全局策略开关 policy_enabled 被关闭（阶段 0 常驻状态），
    // 或者目标设备是 CPU (device < 0)，则不做任何干预，直接返回默认处理。
    if (!policy_enabled || device < 0) {
        return PolicyAction::ManagedDefault;
    }

    // 2. 预热期策略触发 (Warmup Optimization)
    // 这是目前代码中唯一的“干预点”。
    // 触发条件需要同时满足以下三点：
    //   - 类别匹配：必须被识别为 WarmupWorkspace（预热期工作区）。
    //   - 功能开启：环境变量 VLLM_UVM_POLICY_WARMUP_PREFETCH 为 true。
    //   - 尺寸达标：大小必须超过阈值（policy_warmup_prefetch_min_bytes），防止频繁预取碎小内存。
    if (alloc_class == AllocationClass::WarmupWorkspace &&
        policy_warmup_prefetch_enabled &&
        size >= policy_warmup_prefetch_min_bytes) {

        // 如果满足，下达“预取到 GPU”的指令
        return PolicyAction::ManagedPrefetchGpu;
    }

    // 3. 默认回退
    // 对于其他类别（如 WeightPersistent 或 RuntimeScratch），
    // 目前版本的代码选择“相信”驱动程序的默认缺页调度，不主动干预。
    return PolicyAction::ManagedDefault;
}

static PolicyAction choose_gap_watch_policy_action(AllocationClass alloc_class,
                                                   const std::string& phase,
                                                   size_t size,
                                                   int device,
                                                   size_t gap_overlap_bytes,
                                                   bool* class_match_out,
                                                   const char** source_out) {
    if (class_match_out) {
        *class_match_out = false;
    }
    if (source_out) {
        *source_out = "base_policy";
    }

    if (!policy_enabled || device < 0 || gap_overlap_bytes == 0 || !gap_watch_enabled) {
        return PolicyAction::ManagedDefault;
    }

    bool class_match = gap_watch_target_class_matches(alloc_class, phase);
    if (class_match_out) {
        *class_match_out = class_match;
    }
    if (!class_match) {
        return PolicyAction::ManagedDefault;
    }

    if (size < gap_watch_min_bytes) {
        return PolicyAction::ManagedDefault;
    }

    PolicyAction action = PolicyAction::ManagedDefault;
    if (!parse_policy_action_string(gap_watch_policy_action, &action)) {
        return PolicyAction::ManagedDefault;
    }

    if (source_out) {
        *source_out = "gap_watch_policy";
    }
    return action;
}

/**
 * Initialize log file (called once on first allocation)
 */
/**
 * 初始化日志文件与全局策略配置
 * * 该函数采用延迟加载（Lazy Initialization）模式，在第一次调用 uvm_malloc 时触发。
 * 它负责从环境变量中读取用户的调优意图，并打印本次 Session 的配置清单。
 */
static void init_log_file() {
    // 1. 快速检查：如果已经初始化过，直接返回。
    // 这是为了避免在高性能路径中每次分配都要竞争锁。
    if (log_initialized) return;

    {
        // 2. 双重检查锁定模式 (Double-Checked Locking)
        // 加锁防止多个 CPU 线程同时执行初始化逻辑。
        std::lock_guard<std::mutex> lock(log_mutex);

        // 获取锁后再次检查，确保在等待锁期间没有其他线程完成初始化。
        if (log_initialized) return;

        // 3. 时间基准建立
        // 记录程序启动/首次分配的时间点，后续日志中的 [+1.234s] 相对时间戳都以此为原点。
        start_time = std::chrono::steady_clock::now();

        // 4. 从环境变量读取策略配置 (Control Panel)
        // 这种设计允许研究员在不重新编译 C++ 代码的情况下，通过 Shell 直接调整 UVM 行为。
        trace_min_bytes = read_trace_min_bytes_from_env();

        // 默认开启策略层
        policy_enabled = read_bool_from_env("VLLM_UVM_POLICY_ENABLE", true) ? 1 : 0;

        // 读取策略模式（trace_only 代表演习模式，prefetch 代表实战模式）
        policy_mode = read_string_from_env("VLLM_UVM_POLICY_MODE", "trace_only");

        // 级联逻辑：如果模式被设为 prefetch 或 warmup_prefetch，
        // 则自动强制开启 policy_warmup_prefetch_enabled 开关。
        policy_warmup_prefetch_enabled =
            policy_mode == "prefetch" ||
            policy_mode == "warmup_prefetch" ||
            read_bool_from_env("VLLM_UVM_POLICY_WARMUP_PREFETCH", false);

        // 是否开启 Preferred Location (建议位置) 提示
        policy_warmup_advise_gpu =
            read_bool_from_env("VLLM_UVM_POLICY_WARMUP_ADVISE_GPU", false) ? 1 : 0;

        // 读取预取的尺寸下限阈值
        policy_warmup_prefetch_min_bytes = read_size_from_env(
            "VLLM_UVM_POLICY_WARMUP_PREFETCH_MIN_BYTES",
            policy_warmup_prefetch_min_bytes
        );
        unknown_detail_enabled =
            read_bool_from_env("VLLM_UVM_UNKNOWN_DETAIL_ENABLE", false) ? 1 : 0;
        unknown_detail_min_bytes = read_size_from_env(
            "VLLM_UVM_UNKNOWN_DETAIL_MIN_BYTES",
            unknown_detail_min_bytes
        );
        gap_watch_enabled =
            read_bool_from_env("VLLM_UVM_GAP_WATCH_ENABLE", false) ? 1 : 0;
        gap_watch_all_classes =
            read_bool_from_env("VLLM_UVM_GAP_WATCH_ALL_CLASSES", true) ? 1 : 0;
        gap_watch_min_bytes = read_size_from_env(
            "VLLM_UVM_GAP_WATCH_MIN_BYTES",
            gap_watch_min_bytes
        );
        gap_watch_name = read_string_from_env(
            "VLLM_UVM_GAP_WATCH_NAME",
            gap_watch_name.c_str()
        );
        gap_watch_target_class = read_string_from_env(
            "VLLM_UVM_GAP_WATCH_TARGET_CLASS",
            gap_watch_target_class.c_str()
        );
        gap_watch_policy_action = read_string_from_env(
            "VLLM_UVM_GAP_WATCH_POLICY_ACTION",
            gap_watch_policy_action.c_str()
        );
        gap_watch_control_file = read_string_from_env(
            "VLLM_UVM_GAP_WATCH_CONTROL_FILE",
            ""
        );
        gap_watch_refresh_ms = read_size_from_env(
            "VLLM_UVM_GAP_WATCH_REFRESH_MS",
            gap_watch_refresh_ms
        );
        device_direct_enable =
            read_bool_from_env("VLLM_UVM_DEVICE_DIRECT_ENABLE", false) ? 1 : 0;
        device_direct_min_bytes = read_size_from_env(
            "VLLM_UVM_DEVICE_DIRECT_MIN_BYTES",
            device_direct_min_bytes
        );
        device_direct_max_bytes = read_size_from_env(
            "VLLM_UVM_DEVICE_DIRECT_MAX_BYTES",
            device_direct_max_bytes
        );
        device_direct_max_total_bytes = read_size_from_env(
            "VLLM_UVM_DEVICE_DIRECT_MAX_TOTAL_BYTES",
            device_direct_max_total_bytes
        );
        device_direct_max_total_bytes = read_size_from_env_allow_zero(
            "VLLM_UVM_DEVICE_DIRECT_MAX_TOTAL_BYTES",
            device_direct_max_total_bytes
        );
        device_direct_backend = normalize_device_direct_backend(
            read_string_from_env(
                "VLLM_UVM_DEVICE_DIRECT_BACKEND",
                device_direct_backend.c_str()
            )
        );
        const char* pool_release_threshold_env =
            getenv("VLLM_UVM_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD");
        device_direct_pool_release_threshold_set =
            pool_release_threshold_env && pool_release_threshold_env[0] != '\0';
        device_direct_pool_release_threshold = read_size_from_env_allow_zero(
            "VLLM_UVM_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD",
            device_direct_pool_release_threshold
        );
        device_direct_target_phases = read_string_from_env(
            "VLLM_UVM_DEVICE_DIRECT_TARGET_PHASES",
            device_direct_target_phases.c_str()
        );
        kv_device_direct_enable =
            read_bool_from_env(
                "VLLM_UVM_KV_DEVICE_DIRECT_ENABLE",
                false
            ) ? 1 : 0;
        kv_device_direct_min_bytes = read_size_from_env(
            "VLLM_UVM_KV_DEVICE_DIRECT_MIN_BYTES",
            kv_device_direct_min_bytes
        );
        kv_device_direct_max_bytes = read_size_from_env_allow_zero(
            "VLLM_UVM_KV_DEVICE_DIRECT_MAX_BYTES",
            kv_device_direct_max_bytes
        );
        kv_device_direct_max_total_bytes = read_size_from_env_allow_zero(
            "VLLM_UVM_KV_DEVICE_DIRECT_MAX_TOTAL_BYTES",
            kv_device_direct_max_total_bytes
        );
        fixed_weight_device_direct_enable =
            read_bool_from_env(
                "VLLM_UVM_FIXED_WEIGHT_DEVICE_DIRECT_ENABLE",
                false
            ) ? 1 : 0;
        fixed_weight_device_direct_min_bytes = read_size_from_env(
            "VLLM_UVM_FIXED_WEIGHT_DEVICE_DIRECT_MIN_BYTES",
            fixed_weight_device_direct_min_bytes
        );
        fixed_weight_device_direct_max_bytes = read_size_from_env_allow_zero(
            "VLLM_UVM_FIXED_WEIGHT_DEVICE_DIRECT_MAX_BYTES",
            fixed_weight_device_direct_max_bytes
        );
        fixed_weight_device_direct_max_total_bytes = read_size_from_env_allow_zero(
            "VLLM_UVM_FIXED_WEIGHT_DEVICE_DIRECT_MAX_TOTAL_BYTES",
            fixed_weight_device_direct_max_total_bytes
        );
        kv_budget_bytes = read_size_from_env_allow_zero(
            "VLLM_UVM_KV_BUDGET_BYTES",
            kv_budget_bytes
        );
        kv_budget_mode = normalize_kv_budget_mode(
            read_string_from_env(
                "VLLM_UVM_KV_BUDGET_MODE",
                kv_budget_mode.c_str()
            )
        );
        weight_budget_bytes = read_size_from_env_allow_zero(
            "VLLM_UVM_WEIGHT_BUDGET_BYTES",
            weight_budget_bytes
        );
        weight_budget_mode = normalize_weight_budget_mode(
            read_string_from_env(
                "VLLM_UVM_WEIGHT_BUDGET_MODE",
                weight_budget_mode.c_str()
            )
        );
        weight_budget_cpu_offload_enable =
            read_bool_from_env(
                "VLLM_UVM_WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE",
                false
            ) ? 1 : 0;
        pool_registry_enabled =
            read_bool_from_env("VLLM_UVM_POOL_REGISTRY_ENABLE", false) ? 1 : 0;
        driver_scratch_tag_enable =
            read_bool_from_env(
                "VLLM_UVM_DRIVER_SCRATCH_TAG_ENABLE",
                read_bool_from_env("VLLM_UVM_DRIVER_TAG_ENABLE", false)
            ) ? 1 : 0;
        driver_scratch_budget_bytes = read_size_from_env_allow_zero(
            "VLLM_UVM_DRIVER_TEMP_BUFFER_BYTES",
            driver_scratch_budget_bytes
        );
        driver_scratch_budget_enforce =
            read_bool_from_env(
                "VLLM_UVM_DRIVER_SCRATCH_BUDGET_ENFORCE",
                false
            ) ? 1 : 0;
        driver_scratch_budget_enforce_warmup =
            read_bool_from_env(
                "VLLM_UVM_DRIVER_SCRATCH_BUDGET_ENFORCE_WARMUP",
                false
            ) ? 1 : 0;
        scratch_pool_enable =
            read_bool_from_env("VLLM_UVM_SCRATCH_POOL_ENABLE", false) ? 1 : 0;
        scratch_pool_budget_bytes = read_size_from_env_allow_zero(
            "VLLM_UVM_SCRATCH_POOL_BUDGET_BYTES",
            scratch_pool_budget_bytes
        );
        scratch_pool_mode = normalize_scratch_pool_mode(
            read_string_from_env(
                "VLLM_UVM_SCRATCH_POOL_MODE",
                scratch_pool_mode.c_str()
            )
        );
        scratch_pool_target_phases = read_string_from_env(
            "VLLM_UVM_SCRATCH_POOL_TARGET_PHASES",
            scratch_pool_target_phases.c_str()
        );
        scratch_pool_reuse_cache_enable =
            read_bool_from_env(
                "VLLM_UVM_SCRATCH_POOL_REUSE_CACHE_ENABLE",
                false
            ) ? 1 : 0;
        scratch_pool_reuse_cache_max_bytes = read_size_from_env_allow_zero(
            "VLLM_UVM_SCRATCH_POOL_REUSE_CACHE_MAX_BYTES",
            scratch_pool_reuse_cache_max_bytes
        );
        scratch_pool_reuse_cache_max_entries_per_bucket =
            read_size_from_env_allow_zero(
                "VLLM_UVM_SCRATCH_POOL_REUSE_CACHE_MAX_ENTRIES_PER_BUCKET",
                scratch_pool_reuse_cache_max_entries_per_bucket
            );
        uintptr_t parsed_gap_start = 0;
        uintptr_t parsed_gap_end = 0;
        if (read_hex_u64_from_env("VLLM_UVM_GAP_WATCH_START", &parsed_gap_start) &&
            read_hex_u64_from_env("VLLM_UVM_GAP_WATCH_END", &parsed_gap_end) &&
            parsed_gap_end >= parsed_gap_start) {
            gap_watch_start = parsed_gap_start;
            gap_watch_end = parsed_gap_end;
            gap_watch_enabled = 1;
        }

        // 5. 日志文件路径确定
        // 默认文件名为 vllm_uvm_allocations.log，可通过 VLLM_UVM_LOG_FILE 修改。
        const char* log_path = getenv("VLLM_UVM_LOG_FILE");
        if (!log_path) {
            log_path = "vllm_uvm_allocations.log";
        }

        // 6. 以追加模式 ("a") 打开文件
        log_file = fopen(log_path, "a");
        if (log_file) {
            char timestamp[64];
            get_timestamp(timestamp, sizeof(timestamp));

            // 7. 打印 Session 头部信息 (Session Header)
            // 这一步在科研中极其重要，因为它记录了本次实验的所有超参数。
            // 以后你回顾几个月前的日志时，一眼就能看出当时有没有开预取。
            fprintf(log_file, "\n========================================\n");
            fprintf(log_file, "[%s] vLLM UVM Allocator Session Started\n", timestamp);
            fprintf(log_file, "[%s] trace_min_bytes=%zu current_phase=%s\n",
                    timestamp, trace_min_bytes, current_phase.c_str());
            fprintf(
                log_file,
                "[%s] policy_enabled=%d policy_mode=%s "
                "policy_warmup_prefetch_enabled=%d "
                "policy_warmup_prefetch_min_bytes=%zu "
                "policy_warmup_advise_gpu=%d "
                "unknown_detail_enabled=%d unknown_detail_min_bytes=%zu "
                "gap_watch_enabled=%d gap_watch_name=%s "
                "gap_watch_start=0x%llx gap_watch_end=0x%llx "
                "gap_watch_all_classes=%d gap_watch_min_bytes=%zu "
                "gap_watch_target_class=%s gap_watch_policy_action=%s "
                "gap_watch_control_file=%s gap_watch_refresh_ms=%zu "
                "device_direct_enable=%d device_direct_min_bytes=%zu "
                "device_direct_max_bytes=%zu "
                "device_direct_max_total_bytes=%zu "
                "device_direct_backend=%s "
                "device_direct_pool_release_threshold_set=%d "
                "device_direct_pool_release_threshold=%zu "
                "device_direct_target_phases=%s "
                "kv_device_direct_enable=%d "
                "kv_device_direct_min_bytes=%zu "
                "kv_device_direct_max_bytes=%zu "
                "kv_device_direct_max_total_bytes=%zu "
                "fixed_weight_device_direct_enable=%d "
                "fixed_weight_device_direct_min_bytes=%zu "
                "fixed_weight_device_direct_max_bytes=%zu "
                "fixed_weight_device_direct_max_total_bytes=%zu "
                "kv_budget_bytes=%zu kv_budget_mode=%s "
                "weight_budget_bytes=%zu weight_budget_mode=%s "
                "weight_budget_cpu_offload_enable=%d "
                "pool_registry_enabled=%d "
                "driver_scratch_tag_enable=%d "
                "driver_scratch_budget_enforce=%d "
                "driver_scratch_budget_enforce_warmup=%d "
                "driver_scratch_budget_bytes=%zu "
                "scratch_pool_enable=%d scratch_pool_budget_bytes=%zu "
                "scratch_pool_mode=%s scratch_pool_target_phases=%s "
                "scratch_pool_reuse_cache_enable=%d "
                "scratch_pool_reuse_cache_max_bytes=%zu "
                "scratch_pool_reuse_cache_max_entries_per_bucket=%zu\n",
                timestamp, policy_enabled, policy_mode.c_str(),
                policy_warmup_prefetch_enabled, policy_warmup_prefetch_min_bytes,
                policy_warmup_advise_gpu,
                unknown_detail_enabled,
                unknown_detail_min_bytes,
                gap_watch_enabled,
                gap_watch_name.c_str(),
                static_cast<unsigned long long>(gap_watch_start),
                static_cast<unsigned long long>(gap_watch_end),
                gap_watch_all_classes,
                gap_watch_min_bytes,
                gap_watch_target_class.c_str(),
                gap_watch_policy_action.c_str(),
                gap_watch_control_file.empty() ? "none" : gap_watch_control_file.c_str(),
                gap_watch_refresh_ms,
                device_direct_enable,
                device_direct_min_bytes,
                device_direct_max_bytes,
                device_direct_max_total_bytes,
                device_direct_backend.c_str(),
                device_direct_pool_release_threshold_set ? 1 : 0,
                device_direct_pool_release_threshold,
                device_direct_target_phases.c_str(),
                kv_device_direct_enable,
                kv_device_direct_min_bytes,
                kv_device_direct_max_bytes,
                kv_device_direct_max_total_bytes,
                fixed_weight_device_direct_enable,
                fixed_weight_device_direct_min_bytes,
                fixed_weight_device_direct_max_bytes,
                fixed_weight_device_direct_max_total_bytes,
                kv_budget_bytes,
                kv_budget_mode.c_str(),
                weight_budget_bytes,
                weight_budget_mode.c_str(),
                weight_budget_cpu_offload_enable,
                pool_registry_enabled,
                driver_scratch_tag_enable,
                driver_scratch_budget_enforce,
                driver_scratch_budget_enforce_warmup,
                driver_scratch_budget_bytes,
                scratch_pool_enable,
                scratch_pool_budget_bytes,
                scratch_pool_mode.c_str(),
                scratch_pool_target_phases.c_str(),
                scratch_pool_reuse_cache_enable,
                scratch_pool_reuse_cache_max_bytes,
                scratch_pool_reuse_cache_max_entries_per_bucket
            );
            fprintf(log_file, "========================================\n");

            // 强制刷新缓冲区，确保头部信息立即落盘，防止程序崩溃导致日志丢失。
            fflush(log_file);
        } else {
            // 如果日志打开失败（如目录无权限），降级到 stderr 报警，但不阻塞分配流程。
            fprintf(stderr, "[vLLM UVM] Warning: Could not open log file: %s\n", log_path);
        }

        // 8. 标记初始化完成
        log_initialized = true;
    }

    // 注意：这里必须在释放 log_mutex 后再去 refresh 控制文件。
    // 否则 refresh_gap_watch_from_control_file_if_needed() 内部再次尝试获取
    // log_mutex 时会发生自锁，导致服务卡在 "UVM allocator enabled successfully" 之后。
    refresh_gap_watch_from_control_file_if_needed(true);
}

/**
 * Log allocation to file
 */
static void log_allocation(const char* type, size_t size, size_t alloc_num,
                           size_t current_total, size_t peak, int device) {
    if (!log_file) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();

    fprintf(log_file, "[%s] [+%.3fs] %s #%zu: %.2f MB | device: %d | total: %.2f GB | peak: %.2f GB\n",
            timestamp, elapsed, type, alloc_num, size / 1e6, device,
            current_total / 1e9, peak / 1e9);
    fflush(log_file);
}

/**
 * Log free to file
 */
static void log_free(size_t size, size_t free_num, size_t current_total, int device) {
    if (!log_file) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();

    fprintf(log_file, "[%s] [+%.3fs] FREE #%zu: %.2f MB | device: %d | total: %.2f GB\n",
            timestamp, elapsed, free_num, size / 1e6, device, current_total / 1e9);
    fflush(log_file);
}

static void log_phase_event(const char* event, const char* phase) {
    if (!log_file) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();
    const char* safe_phase = (phase && phase[0] != '\0') ? phase : "unscoped";
    const char* safe_event = (event && event[0] != '\0') ? event : "marker";

    fprintf(log_file, "[%s] [+%.3fs] TRACE_PHASE event=%s phase=%s\n",
            timestamp, elapsed, safe_event, safe_phase);
    fflush(log_file);
}

static void trace_allocation_event(void* ptr,
                                   size_t size,
                                   size_t alloc_num,
                                   size_t current_total,
                                   size_t peak,
                                   int device,
                                   const std::string& phase) {
    if (!log_file) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();
    uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t end = region_end_for(start, size);

    fprintf(
        log_file,
        "[%s] [+%.3fs] TRACE_ALLOC alloc_id=%zu ptr=0x%llx end=0x%llx "
        "size_bytes=%zu size_mb=%.6f device=%d phase=%s total_bytes=%zu peak_bytes=%zu\n",
        timestamp,
        elapsed,
        alloc_num,
        static_cast<unsigned long long>(start),
        static_cast<unsigned long long>(end),
        size,
        size / 1e6,
        device,
        phase.c_str(),
        current_total,
        peak
    );
    fflush(log_file);
}

static void trace_free_event(void* ptr,
                             size_t size,
                             size_t free_num,
                             size_t current_total,
                             int device,
                             const AllocationInfo* info,
                             const std::string& current_phase_snapshot) {
    if (!log_file) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();
    uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t end = region_end_for(start, size);
    double lifetime_seconds =
        info ? (elapsed - info->alloc_elapsed_seconds) : -1.0;

    fprintf(
        log_file,
        "[%s] [+%.3fs] TRACE_FREE free_id=%zu ptr=0x%llx end=0x%llx "
        "size_bytes=%zu size_mb=%.6f device=%d phase=%s alloc_id=%zu "
        "alloc_phase=%s lifetime_s=%.6f total_bytes=%zu\n",
        timestamp,
        elapsed,
        free_num,
        static_cast<unsigned long long>(start),
        static_cast<unsigned long long>(end),
        size,
        size / 1e6,
        device,
        current_phase_snapshot.c_str(),
        info ? info->alloc_id : 0,
        info ? info->phase.c_str() : "unknown",
        lifetime_seconds,
        current_total
    );
    fflush(log_file);
}

static void trace_policy_event(void* ptr,
                               size_t size,
                               size_t alloc_num,
                               int device,
                               const std::string& phase,
                               AllocationClass alloc_class,
                               PolicyAction action,
                               const char* policy_source,
                               bool gap_watch_class_match,
                               size_t gap_overlap_bytes,
                               const char* size_bucket,
                               const char* placement_backend,
                               const char* device_direct_backend_used,
                               bool device_direct_eligible,
                               const char* device_direct_reason,
                               const char* cpu_access_risk,
                               bool kv_budget_tracked,
                               bool kv_budget_over_budget,
                               const char* kv_budget_reason,
                               bool weight_budget_tracked,
                               bool weight_budget_over_budget,
                               const char* weight_budget_reason,
                               bool scratch_pool_tracked,
                               bool scratch_pool_eligible,
                               bool scratch_pool_device_direct,
                               bool scratch_pool_budget_over_budget,
                               const char* scratch_pool_reason,
                               bool hot_gap_match,
                               bool action_success,
                               const char* action_error) {
    if (!log_file || !policy_enabled) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();
    uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t end = region_end_for(start, size);
    const char* safe_error = (action_error && action_error[0] != '\0') ? action_error : "none";
    size_t device_direct_live_snapshot = device_direct_live_bytes.load();
    size_t device_direct_budget_remaining =
        device_direct_budget_remaining_snapshot(device_direct_live_snapshot);
    size_t kv_live_snapshot = kv_live_bytes.load();
    size_t kv_budget_remaining = kv_budget_remaining_snapshot(kv_live_snapshot);
    size_t weight_live_snapshot = weight_live_bytes.load();
    size_t weight_budget_remaining =
        weight_budget_remaining_snapshot(weight_live_snapshot);
    size_t scratch_pool_live_snapshot =
        scratch_pool_device_direct_live_bytes.load();
    size_t scratch_pool_budget_remaining =
        scratch_pool_budget_remaining_snapshot(scratch_pool_live_snapshot);
    int pool_config_attempted_snapshot = 0;
    int pool_config_success_snapshot = 0;
    int pool_config_device_snapshot = -1;
    {
        std::lock_guard<std::mutex> pool_lock(device_direct_pool_mutex);
        pool_config_attempted_snapshot =
            device_direct_pool_config_attempted ? 1 : 0;
        pool_config_success_snapshot =
            device_direct_pool_config_success ? 1 : 0;
        pool_config_device_snapshot = device_direct_pool_config_device;
    }

    fprintf(
        log_file,
        "[%s] [+%.3fs] TRACE_POLICY alloc_id=%zu ptr=0x%llx end=0x%llx "
        "size_bytes=%zu size_bucket=%s device=%d phase=%s "
        "predicted_class=%s action=%s policy_source=%s "
        "gap_watch_class_match=%d gap_overlap_bytes=%zu "
        "action_success=%d action_error=%s placement_backend=%s "
        "device_direct_backend=%s "
        "device_direct_eligible=%d device_direct_reason=%s "
        "device_direct_live_bytes=%zu device_direct_max_total_bytes=%zu "
        "device_direct_budget_remaining=%zu "
        "device_direct_pool_release_threshold_set=%d "
        "device_direct_pool_release_threshold=%zu "
        "device_direct_pool_config_attempted=%d "
        "device_direct_pool_config_success=%d "
        "device_direct_pool_config_device=%d "
        "kv_budget_tracked=%d kv_budget_over_budget=%d "
        "kv_budget_reason=%s kv_live_bytes=%zu kv_budget_bytes=%zu "
        "kv_budget_remaining=%zu kv_budget_mode=%s "
        "weight_budget_tracked=%d weight_budget_over_budget=%d "
        "weight_budget_reason=%s weight_live_bytes=%zu "
        "weight_budget_bytes=%zu weight_budget_remaining=%zu "
        "weight_budget_mode=%s "
        "scratch_pool_tracked=%d scratch_pool_eligible=%d "
        "scratch_pool_device_direct=%d scratch_pool_over_budget=%d "
        "scratch_pool_reason=%s scratch_pool_live_bytes=%zu "
        "scratch_pool_budget_bytes=%zu scratch_pool_budget_remaining=%zu "
        "scratch_pool_mode=%s scratch_pool_enable=%d "
        "cpu_access_risk=%s hot_gap_match=%d\n",
        timestamp,
        elapsed,
        alloc_num,
        static_cast<unsigned long long>(start),
        static_cast<unsigned long long>(end),
        size,
        size_bucket,
        device,
        phase.c_str(),
        allocation_class_to_string(alloc_class),
        policy_action_to_string(action),
        (policy_source && policy_source[0] != '\0') ? policy_source : "unknown",
        gap_watch_class_match ? 1 : 0,
        gap_overlap_bytes,
        action_success ? 1 : 0,
        safe_error,
        placement_backend ? placement_backend : "managed",
        device_direct_backend_used ? device_direct_backend_used : "none",
        device_direct_eligible ? 1 : 0,
        device_direct_reason ? device_direct_reason : "not_requested",
        device_direct_live_snapshot,
        device_direct_max_total_bytes,
        device_direct_budget_remaining,
        device_direct_pool_release_threshold_set ? 1 : 0,
        device_direct_pool_release_threshold,
        pool_config_attempted_snapshot,
        pool_config_success_snapshot,
        pool_config_device_snapshot,
        kv_budget_tracked ? 1 : 0,
        kv_budget_over_budget ? 1 : 0,
        kv_budget_reason ? kv_budget_reason : "not_kv",
        kv_live_snapshot,
        kv_budget_bytes,
        kv_budget_remaining,
        kv_budget_mode.c_str(),
        weight_budget_tracked ? 1 : 0,
        weight_budget_over_budget ? 1 : 0,
        weight_budget_reason ? weight_budget_reason : "not_weight",
        weight_live_snapshot,
        weight_budget_bytes,
        weight_budget_remaining,
        weight_budget_mode.c_str(),
        scratch_pool_tracked ? 1 : 0,
        scratch_pool_eligible ? 1 : 0,
        scratch_pool_device_direct ? 1 : 0,
        scratch_pool_budget_over_budget ? 1 : 0,
        scratch_pool_reason ? scratch_pool_reason : "not_scratch_pool",
        scratch_pool_live_snapshot,
        scratch_pool_budget_bytes,
        scratch_pool_budget_remaining,
        scratch_pool_mode.c_str(),
        scratch_pool_enable,
        cpu_access_risk ? cpu_access_risk : "unknown",
        hot_gap_match ? 1 : 0
    );
    fflush(log_file);
}

static void trace_pool_alloc_event(void* ptr,
                                   size_t size,
                                   size_t alloc_num,
                                   int device,
                                   const std::string& phase,
                                   AllocationClass alloc_class,
                                   PoolKind pool_kind,
                                   const char* size_bucket,
                                   const char* placement_backend,
                                   const char* device_direct_backend_used,
                                   bool scratch_pool_tracked,
                                   bool scratch_pool_eligible,
                                   bool scratch_pool_device_direct,
                                   bool scratch_pool_budget_over_budget,
                                   const char* scratch_pool_reason,
                                   const char* pool_object_state) {
    if (!log_file || !pool_registry_enabled ||
        pool_kind == PoolKind::OtherManaged) {
        return;
    }

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();
    uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t end = region_end_for(start, size);

    fprintf(
        log_file,
        "[%s] [+%.3fs] TRACE_POOL_ALLOC alloc_id=%zu ptr=0x%llx "
        "end=0x%llx size_bytes=%zu size_bucket=%s device=%d phase=%s "
        "predicted_class=%s pool_kind=%s pool_object_state=%s "
        "pool_registry_enabled=%d placement_backend=%s "
        "device_direct_backend=%s "
        "scratch_pool_tracked=%d scratch_pool_eligible=%d "
        "scratch_pool_device_direct=%d scratch_pool_over_budget=%d "
        "scratch_pool_reason=%s "
        "pool_registry_live_objects=%zu pool_registry_peak_live_objects=%zu "
        "pool_kv_live_bytes=%zu pool_weight_live_bytes=%zu "
        "pool_runtime_scratch_live_bytes=%zu\n",
        timestamp,
        elapsed,
        alloc_num,
        static_cast<unsigned long long>(start),
        static_cast<unsigned long long>(end),
        size,
        size_bucket,
        device,
        phase.c_str(),
        allocation_class_to_string(alloc_class),
        pool_kind_to_string(pool_kind),
        (pool_object_state && pool_object_state[0] != '\0')
            ? pool_object_state
            : "active",
        pool_registry_enabled,
        placement_backend ? placement_backend : "managed",
        device_direct_backend_used ? device_direct_backend_used : "none",
        scratch_pool_tracked ? 1 : 0,
        scratch_pool_eligible ? 1 : 0,
        scratch_pool_device_direct ? 1 : 0,
        scratch_pool_budget_over_budget ? 1 : 0,
        scratch_pool_reason ? scratch_pool_reason : "not_scratch_pool",
        pool_registry_live_objects.load(),
        pool_registry_peak_live_objects.load(),
        pool_kv_live_bytes.load(),
        pool_weight_live_bytes.load(),
        pool_runtime_scratch_live_bytes.load()
    );
    fflush(log_file);
}

static void trace_pool_free_event(void* ptr,
                                  size_t size,
                                  size_t free_num,
                                  int device,
                                  const AllocationInfo* info,
                                  const std::string& current_phase_snapshot) {
    if (!log_file || !pool_registry_enabled || !info ||
        !info->pool_registry_tracked) {
        return;
    }

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();
    uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t end = region_end_for(start, size);
    double lifetime_seconds = elapsed - info->alloc_elapsed_seconds;

    fprintf(
        log_file,
        "[%s] [+%.3fs] TRACE_POOL_FREE free_id=%zu ptr=0x%llx "
        "end=0x%llx size_bytes=%zu device=%d phase=%s alloc_id=%zu "
        "alloc_phase=%s lifetime_s=%.6f predicted_class=%s pool_kind=%s "
        "pool_object_state=freed placement_backend=%s "
        "device_direct_backend=%s pool_registry_live_objects=%zu "
        "pool_registry_peak_live_objects=%zu pool_kv_live_bytes=%zu "
        "pool_weight_live_bytes=%zu pool_runtime_scratch_live_bytes=%zu "
        "scratch_pool_tracked=%d scratch_pool_eligible=%d "
        "scratch_pool_device_direct=%d scratch_pool_reason=%s\n",
        timestamp,
        elapsed,
        free_num,
        static_cast<unsigned long long>(start),
        static_cast<unsigned long long>(end),
        size,
        device,
        current_phase_snapshot.c_str(),
        info->alloc_id,
        info->phase.c_str(),
        lifetime_seconds,
        info->alloc_class_name.c_str(),
        info->pool_kind_name.c_str(),
        info->placement_backend_name.c_str(),
        info->device_direct_backend_name.c_str(),
        pool_registry_live_objects.load(),
        pool_registry_peak_live_objects.load(),
        pool_kv_live_bytes.load(),
        pool_weight_live_bytes.load(),
        pool_runtime_scratch_live_bytes.load(),
        info->scratch_pool_tracked ? 1 : 0,
        info->scratch_pool_eligible ? 1 : 0,
        info->scratch_pool_device_direct ? 1 : 0,
        info->scratch_pool_reason.c_str()
    );
    fflush(log_file);
}

static void trace_unknown_detail_event(void* ptr,
                                       size_t size,
                                       size_t alloc_num,
                                       int device,
                                       const std::string& phase,
                                       const char* size_bucket,
                                       PolicyAction action,
                                       cudaStream_t stream,
                                       bool overlaps_gap_watch,
                                       uintptr_t gap_overlap_start,
                                       uintptr_t gap_overlap_end,
                                       size_t gap_overlap_bytes) {
    if (!log_file) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();
    uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t end = region_end_for(start, size);

    fprintf(
        log_file,
        "[%s] [+%.3fs] TRACE_UNKNOWN_DETAIL alloc_id=%zu ptr=0x%llx end=0x%llx "
        "size_bytes=%zu size_bucket=%s device=%d phase=%s predicted_class=%s "
        "action=%s stream=0x%llx gap_watch_name=%s overlaps_gap_watch=%d "
        "gap_overlap_start=0x%llx gap_overlap_end=0x%llx gap_overlap_bytes=%zu\n",
        timestamp,
        elapsed,
        alloc_num,
        static_cast<unsigned long long>(start),
        static_cast<unsigned long long>(end),
        size,
        size_bucket,
        device,
        phase.c_str(),
        allocation_class_to_string(AllocationClass::UnknownManaged),
        policy_action_to_string(action),
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(stream)),
        gap_watch_name.c_str(),
        overlaps_gap_watch ? 1 : 0,
        static_cast<unsigned long long>(gap_overlap_start),
        static_cast<unsigned long long>(gap_overlap_end),
        gap_overlap_bytes
    );
    fflush(log_file);
}

static void trace_gap_watch_alloc_event(void* ptr,
                                        size_t size,
                                        size_t alloc_num,
                                        int device,
                                        const std::string& phase,
                                        AllocationClass alloc_class,
                                        PolicyAction action,
                                        const char* policy_source,
                                        bool gap_watch_class_match,
                                        const char* size_bucket,
                                        cudaStream_t stream,
                                        uintptr_t overlap_start,
                                        uintptr_t overlap_end,
                                        size_t overlap_bytes,
                                        const char* placement_backend,
                                        const char* device_direct_backend_used,
                                        bool device_direct_eligible,
                                        const char* device_direct_reason,
                                        const char* cpu_access_risk,
                                        bool kv_budget_tracked,
                                        bool kv_budget_over_budget,
                                        const char* kv_budget_reason,
                                        bool weight_budget_tracked,
                                        bool weight_budget_over_budget,
                                        const char* weight_budget_reason,
                                        bool scratch_pool_tracked,
                                        bool scratch_pool_eligible,
                                        bool scratch_pool_device_direct,
                                        bool scratch_pool_budget_over_budget,
                                        const char* scratch_pool_reason,
                                        bool hot_gap_match) {
    if (!log_file) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();
    uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t end = region_end_for(start, size);
    double gap_ratio = 0.0;
    if (gap_watch_enabled && gap_watch_end >= gap_watch_start) {
        size_t gap_size = gap_watch_end - gap_watch_start + 1;
        if (gap_size > 0) {
            gap_ratio = static_cast<double>(overlap_bytes) /
                        static_cast<double>(gap_size);
        }
    }
    size_t device_direct_live_snapshot = device_direct_live_bytes.load();
    size_t device_direct_budget_remaining =
        device_direct_budget_remaining_snapshot(device_direct_live_snapshot);
    size_t kv_live_snapshot = kv_live_bytes.load();
    size_t kv_budget_remaining = kv_budget_remaining_snapshot(kv_live_snapshot);
    size_t weight_live_snapshot = weight_live_bytes.load();
    size_t weight_budget_remaining =
        weight_budget_remaining_snapshot(weight_live_snapshot);
    size_t scratch_pool_live_snapshot =
        scratch_pool_device_direct_live_bytes.load();
    size_t scratch_pool_budget_remaining =
        scratch_pool_budget_remaining_snapshot(scratch_pool_live_snapshot);
    int pool_config_attempted_snapshot = 0;
    int pool_config_success_snapshot = 0;
    int pool_config_device_snapshot = -1;
    {
        std::lock_guard<std::mutex> pool_lock(device_direct_pool_mutex);
        pool_config_attempted_snapshot =
            device_direct_pool_config_attempted ? 1 : 0;
        pool_config_success_snapshot =
            device_direct_pool_config_success ? 1 : 0;
        pool_config_device_snapshot = device_direct_pool_config_device;
    }

    fprintf(
        log_file,
        "[%s] [+%.3fs] TRACE_GAP_WATCH_ALLOC alloc_id=%zu watch_name=%s "
        "ptr=0x%llx end=0x%llx size_bytes=%zu size_bucket=%s device=%d phase=%s "
        "predicted_class=%s action=%s policy_source=%s "
        "gap_watch_target_class=%s gap_watch_class_match=%d stream=0x%llx overlap_start=0x%llx "
        "overlap_end=0x%llx overlap_bytes=%zu overlap_ratio_of_watch=%.6f "
        "placement_backend=%s device_direct_backend=%s "
        "device_direct_eligible=%d device_direct_reason=%s "
        "device_direct_live_bytes=%zu device_direct_max_total_bytes=%zu "
        "device_direct_budget_remaining=%zu "
        "device_direct_pool_release_threshold_set=%d "
        "device_direct_pool_release_threshold=%zu "
        "device_direct_pool_config_attempted=%d "
        "device_direct_pool_config_success=%d "
        "device_direct_pool_config_device=%d "
        "kv_budget_tracked=%d kv_budget_over_budget=%d "
        "kv_budget_reason=%s kv_live_bytes=%zu kv_budget_bytes=%zu "
        "kv_budget_remaining=%zu kv_budget_mode=%s "
        "weight_budget_tracked=%d weight_budget_over_budget=%d "
        "weight_budget_reason=%s weight_live_bytes=%zu "
        "weight_budget_bytes=%zu weight_budget_remaining=%zu "
        "weight_budget_mode=%s "
        "scratch_pool_tracked=%d scratch_pool_eligible=%d "
        "scratch_pool_device_direct=%d scratch_pool_over_budget=%d "
        "scratch_pool_reason=%s scratch_pool_live_bytes=%zu "
        "scratch_pool_budget_bytes=%zu scratch_pool_budget_remaining=%zu "
        "scratch_pool_mode=%s scratch_pool_enable=%d "
        "cpu_access_risk=%s hot_gap_match=%d\n",
        timestamp,
        elapsed,
        alloc_num,
        gap_watch_name.c_str(),
        static_cast<unsigned long long>(start),
        static_cast<unsigned long long>(end),
        size,
        size_bucket,
        device,
        phase.c_str(),
        allocation_class_to_string(alloc_class),
        policy_action_to_string(action),
        (policy_source && policy_source[0] != '\0') ? policy_source : "unknown",
        gap_watch_target_class.c_str(),
        gap_watch_class_match ? 1 : 0,
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(stream)),
        static_cast<unsigned long long>(overlap_start),
        static_cast<unsigned long long>(overlap_end),
        overlap_bytes,
        gap_ratio,
        placement_backend ? placement_backend : "managed",
        device_direct_backend_used ? device_direct_backend_used : "none",
        device_direct_eligible ? 1 : 0,
        device_direct_reason ? device_direct_reason : "not_requested",
        device_direct_live_snapshot,
        device_direct_max_total_bytes,
        device_direct_budget_remaining,
        device_direct_pool_release_threshold_set ? 1 : 0,
        device_direct_pool_release_threshold,
        pool_config_attempted_snapshot,
        pool_config_success_snapshot,
        pool_config_device_snapshot,
        kv_budget_tracked ? 1 : 0,
        kv_budget_over_budget ? 1 : 0,
        kv_budget_reason ? kv_budget_reason : "not_kv",
        kv_live_snapshot,
        kv_budget_bytes,
        kv_budget_remaining,
        kv_budget_mode.c_str(),
        weight_budget_tracked ? 1 : 0,
        weight_budget_over_budget ? 1 : 0,
        weight_budget_reason ? weight_budget_reason : "not_weight",
        weight_live_snapshot,
        weight_budget_bytes,
        weight_budget_remaining,
        weight_budget_mode.c_str(),
        scratch_pool_tracked ? 1 : 0,
        scratch_pool_eligible ? 1 : 0,
        scratch_pool_device_direct ? 1 : 0,
        scratch_pool_budget_over_budget ? 1 : 0,
        scratch_pool_reason ? scratch_pool_reason : "not_scratch_pool",
        scratch_pool_live_snapshot,
        scratch_pool_budget_bytes,
        scratch_pool_budget_remaining,
        scratch_pool_mode.c_str(),
        scratch_pool_enable,
        cpu_access_risk ? cpu_access_risk : "unknown",
        hot_gap_match ? 1 : 0
    );
    fflush(log_file);
}

static void trace_gap_watch_free_event(void* ptr,
                                       size_t size,
                                       size_t free_num,
                                       size_t current_total,
                                       int device,
                                       const AllocationInfo* info,
                                       const std::string& current_phase_snapshot) {
    if (!log_file || !info || !info->gap_watch_logged) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();
    uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t end = region_end_for(start, size);
    double lifetime_seconds = elapsed - info->alloc_elapsed_seconds;
    double gap_ratio = 0.0;
    if (gap_watch_enabled && gap_watch_end >= gap_watch_start) {
        size_t gap_size = gap_watch_end - gap_watch_start + 1;
        if (gap_size > 0) {
            gap_ratio = static_cast<double>(info->gap_overlap_bytes) /
                        static_cast<double>(gap_size);
        }
    }

    fprintf(
        log_file,
        "[%s] [+%.3fs] TRACE_GAP_WATCH_FREE free_id=%zu watch_name=%s "
        "ptr=0x%llx end=0x%llx size_bytes=%zu device=%d phase=%s alloc_id=%zu "
        "alloc_phase=%s alloc_predicted_class=%s alloc_action=%s "
        "alloc_policy_source=%s alloc_policy_success=%d alloc_policy_error=%s "
        "gap_watch_target_class=%s gap_watch_policy_action=%s "
        "overlap_start=0x%llx overlap_end=0x%llx overlap_bytes=%zu "
        "overlap_ratio_of_watch=%.6f lifetime_s=%.6f total_bytes=%zu "
        "placement_backend=%s device_direct_backend=%s "
        "device_direct_eligible=%d device_direct_reason=%s "
        "kv_budget_tracked=%d kv_budget_over_budget=%d kv_budget_reason=%s "
        "weight_budget_tracked=%d weight_budget_over_budget=%d "
        "weight_budget_reason=%s "
        "scratch_pool_tracked=%d scratch_pool_eligible=%d "
        "scratch_pool_device_direct=%d scratch_pool_reason=%s "
        "cpu_access_risk=%s hot_gap_match=%d\n",
        timestamp,
        elapsed,
        free_num,
        gap_watch_name.c_str(),
        static_cast<unsigned long long>(start),
        static_cast<unsigned long long>(end),
        size,
        device,
        current_phase_snapshot.c_str(),
        info->alloc_id,
        info->phase.c_str(),
        info->alloc_class_name.c_str(),
        info->policy_action_name.c_str(),
        info->policy_source_name.c_str(),
        info->policy_action_success ? 1 : 0,
        info->policy_action_error.c_str(),
        info->gap_watch_target_class_name.c_str(),
        info->gap_watch_policy_action_name.c_str(),
        static_cast<unsigned long long>(info->gap_overlap_start),
        static_cast<unsigned long long>(info->gap_overlap_end),
        info->gap_overlap_bytes,
        gap_ratio,
        lifetime_seconds,
        current_total,
        info->placement_backend_name.c_str(),
        info->device_direct_backend_name.c_str(),
        info->device_direct_eligible ? 1 : 0,
        info->device_direct_reason.c_str(),
        info->kv_budget_tracked ? 1 : 0,
        info->kv_budget_over_budget ? 1 : 0,
        info->kv_budget_reason.c_str(),
        info->weight_budget_tracked ? 1 : 0,
        info->weight_budget_over_budget ? 1 : 0,
        info->weight_budget_reason.c_str(),
        info->scratch_pool_tracked ? 1 : 0,
        info->scratch_pool_eligible ? 1 : 0,
        info->scratch_pool_device_direct ? 1 : 0,
        info->scratch_pool_reason.c_str(),
        info->cpu_access_risk.c_str(),
        info->hot_gap_match ? 1 : 0
    );
    fflush(log_file);
}

static void trace_unknown_detail_free_event(void* ptr,
                                            size_t size,
                                            size_t free_num,
                                            size_t current_total,
                                            int device,
                                            const AllocationInfo* info,
                                            const std::string& current_phase_snapshot) {
    if (!log_file || !info || !info->unknown_detail_logged) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();
    uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t end = region_end_for(start, size);
    double lifetime_seconds = elapsed - info->alloc_elapsed_seconds;

    fprintf(
        log_file,
        "[%s] [+%.3fs] TRACE_UNKNOWN_DETAIL_FREE free_id=%zu ptr=0x%llx end=0x%llx "
        "size_bytes=%zu device=%d phase=%s alloc_id=%zu alloc_phase=%s "
        "alloc_predicted_class=%s alloc_action=%s gap_watch_name=%s "
        "gap_overlap_bytes=%zu lifetime_s=%.6f total_bytes=%zu\n",
        timestamp,
        elapsed,
        free_num,
        static_cast<unsigned long long>(start),
        static_cast<unsigned long long>(end),
        size,
        device,
        current_phase_snapshot.c_str(),
        info->alloc_id,
        info->phase.c_str(),
        info->alloc_class_name.c_str(),
        info->policy_action_name.c_str(),
        gap_watch_name.c_str(),
        info->gap_overlap_bytes,
        lifetime_seconds,
        current_total
    );
    fflush(log_file);
}

/**
 * Allocate CUDA managed (UVM) memory
 *
 * 该函数是 vLLM/PyTorch 底层显存分配器的 Pluggable Allocator 核心入口。
 * 它的主要使命是拦截常规的 cudaMalloc，将其替换为 UVM 分配，
 * 从而允许大模型（权重+KV Cache）的显存占用突破物理 GPU 显存上限。
 *
 * @param size 请求分配的字节数
 * @param device 目标 CUDA 设备 ID
 * @param stream CUDA 流（对于纯 Managed Memory 分配本身不生效，但在后续的 Prefetch 中会用到）
 * @return 成功返回内存指针，失败返回 NULL
 */
void* uvm_malloc(ssize_t size, int device, cudaStream_t stream) {
    // 1. 延迟初始化：确保在进程生命周期内首次分配时加载环境变量和日志配置
    if (!log_initialized) {
        init_log_file();
    }

    void* ptr = NULL;
    cudaError_t err;

    // 2. 核心分配逻辑：使用 cudaMallocManaged 申请 UVM
    // cudaMemAttachGlobal 参数至关重要，它向 CUDA 驱动声明这块内存对主机 CPU 和所有 GPU 均可访问。
    // 这使得底层驱动可以通过 Page Fault (缺页中断) 在主机 RAM 和 GPU VRAM 之间自动换页。
    err = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);

    if (err != cudaSuccess) {
        fprintf(stderr, "[vLLM UVM] cudaMallocManaged failed for %zd bytes: %s\n",
                size, cudaGetErrorString(err));
        return NULL;
    }

    // 3. 维护全局原子统计信息（高并发友好）
    // fetch_add 返回的是相加之前的值，所以加上 size 才是当前实际值
    size_t current = total_allocated.fetch_add(size) + size;
    size_t alloc_count = num_allocs.fetch_add(1) + 1;

    // 4. 无锁 (Lock-free) 更新显存高水位线 (Peak Memory)
    // 使用 Compare-And-Swap (CAS) 循环来处理多线程并发分配时的竞争状态，
    // 确保 peak_allocated 准确记录历史最高分配量。
    size_t peak = peak_allocated.load();
    while (current > peak) {
        if (peak_allocated.compare_exchange_weak(peak, current)) {
            break;
        }
    }

    std::string phase_snapshot;
    double alloc_elapsed = get_elapsed_seconds();

    // 5. 捕获当前 phase 快照
    {
        std::lock_guard<std::mutex> lock(log_mutex);
        phase_snapshot = current_phase.empty() ? "unscoped" : current_phase;
    }

    // 6. 分级日志记录
    // 对于超大块内存 (> 100MB，通常是模型权重加载或大规模 KV Cache 初始化)，强制记录简报
    if (size > 100 * 1024 * 1024) {
        log_allocation("ALLOC", size, alloc_count, current, peak_allocated.load(), device);
    }

    // 标准错误输出 (stderr) 回显，通常用于本地调试
    if (verbose_logging && size > 100 * 1024 * 1024) {
        fprintf(stderr, "[vLLM UVM] Alloc #%zu: %.2f MB (total: %.2f GB, peak: %.2f GB)\n",
                alloc_count, size / 1e6, current / 1e9,
                peak_allocated.load() / 1e9);
    }

    // 7. 启发式内存策略推断 (Heuristic Policy Engine)
    // 结合当前运行阶段 (如 warmup) 和申请大小，推测这块内存的用途
    AllocationClass alloc_class = classify_allocation(
        phase_snapshot,
        static_cast<size_t>(size)
    );
    // 决定是否要对该内存块采取特殊动作（如主动 Advise 或 Prefetch）
    PolicyAction base_policy_action = choose_policy_action(
        alloc_class,
        static_cast<size_t>(size),
        device
    );

    /* ============================================================================
    * 第一阶段：动态配置加载与地址空间重叠分析 (Gap-Watch Detection)
    * ============================================================================ */

    // 1.1 动态同步监控策略：
    // 从外部控制文件（通常是 JSON）刷新热点地址区间（Gap）。
    // 这允许我们在不重启 vLLM 进程的情况下，通过外部工具动态修正要优化的内存地址段。
    refresh_gap_watch_from_control_file_if_needed(false);

    const char* policy_action_error = "none";
    bool policy_action_success = true;

    // 1.2 尺寸分桶：将字节大小转换为可读标签（如 "1MB-4MB"），便于日志聚合分析。
    const char* size_bucket = size_bucket_for(static_cast<size_t>(size));

    // 1.3 核心逻辑：计算地址重叠 (Compute Overlap)
    // 检查当前分配的 UVM 地址 (ptr) 及其范围是否落在了我们标记为“频繁缺页中断”的 Hot Gap 区间内。
    // 如果重叠，gap_overlap_bytes 将记录具体命中的字节数，这是 Stage C 优化的物理前提。
    uintptr_t gap_overlap_start = 0;
    uintptr_t gap_overlap_end = 0;
    size_t gap_overlap_bytes = 0;
    bool overlaps_gap_watch = compute_overlap(
        reinterpret_cast<uintptr_t>(ptr),
        static_cast<size_t>(size),
        gap_watch_start,
        gap_watch_end,
        &gap_overlap_start,
        &gap_overlap_end,
        &gap_overlap_bytes
    );

    /* ============================================================================
    * 第二阶段：策略仲裁与追踪配置 (Policy Arbitration)
    * ============================================================================ */

    // 2.1 追踪详情判定：决定是否记录该次分配的原始细节（用于诊断未知内存块）。
    bool log_unknown_detail = should_trace_unknown_detail(alloc_class, static_cast<size_t>(size));
    bool log_gap_watch = should_trace_gap_watch(alloc_class, static_cast<size_t>(size), gap_overlap_bytes);

    // 2.2 获取针对 Gap 区间的特定策略：
    // 比如：如果命中 Gap，且类别是 runtime_scratch，则策略可能是 PolicyAction::DeviceDirect。
    bool gap_watch_class_match = false;
    const char* policy_source = "base_policy";
    PolicyAction gap_watch_policy_action = choose_gap_watch_policy_action(
        alloc_class, phase_snapshot, static_cast<size_t>(size), device,
        gap_overlap_bytes, &gap_watch_class_match, &policy_source
    );

    // 2.3 策略覆盖：
    // 优先级：Gap-Watch 策略 > 基础启发式策略。
    // 如果 Gap-Watch 返回了有效动作，则覆盖默认行为，实现“精确打击”。
    PolicyAction policy_action =
        (gap_watch_policy_action != PolicyAction::ManagedDefault)
            ? gap_watch_policy_action
            : base_policy_action;

    /* ============================================================================
    * 第三阶段：安全边界检查与预算审计 (Safety & Budgeting)
    * ============================================================================ */

    // 3.1 基础状态标志位：标记是否命中热点区间、是否请求了直连模式、当前执行相位是否匹配。
    bool hot_gap_match = gap_watch_enabled && gap_overlap_bytes > 0;
    bool device_direct_requested = is_device_direct_action(policy_action);
    bool device_direct_phase_match = is_device_direct_target_phase(phase_snapshot);
    bool kv_device_direct_requested =
        kv_device_direct_enable != 0 && is_kv_allocation(alloc_class);
    bool fixed_weight_device_direct_requested =
        fixed_weight_device_direct_enable != 0 &&
        phase_snapshot == "fixed_weight_device_direct" &&
        is_weight_allocation(alloc_class);

    // 3.2 初始环境回显：记录后端默认为 managed。
    const char* placement_backend = "managed";
    const char* device_direct_backend_used = "none";

    // 3.3 CPU 访问风险评估：
    // 如果在允许的计算相位（如 attention）开启直连，CPU 触碰风险极低。
    // 反之，若在未知相位强开直连，则存在被 CPU 访问导致 Segment Fault 的风险。
    const char* cpu_access_risk = device_direct_phase_match
        ? "low_no_cpu_access_evidence_runtime_phase"
        : "unknown";

    const char* device_direct_reason = "not_requested";
    bool device_direct_eligible = false;

    if (kv_device_direct_requested) {
        policy_action = PolicyAction::DeviceDirect;
        policy_source = "kv_device_direct_policy";
        device_direct_requested = true;
        device_direct_reason = "kv_device_direct_requested";
        cpu_access_risk = "low_gpu_only_kv_cache";
    } else if (fixed_weight_device_direct_requested) {
        policy_action = PolicyAction::DeviceDirect;
        policy_source = "fixed_weight_device_direct_policy";
        device_direct_requested = true;
        device_direct_reason = "fixed_weight_device_direct_requested";
        cpu_access_risk = "low_read_only_fixed_weight";
    }

    // 3.4 核心组件显存记账 (KV & Weights)：
    // 这里的 record_* 调用会进行原子加减，确保我们永远知道 KV Cache 和模型权重占用了多少“真显存”。
    // 这是 Stage E 成功的关键——对每一字节权重进行身份追踪。
    bool kv_budget_tracked = is_kv_allocation(alloc_class);
    bool kv_budget_over_budget = false;
    const char* kv_budget_reason = "not_kv";
    bool weight_budget_tracked = is_weight_allocation(alloc_class);
    bool weight_budget_over_budget = false;
    const char* weight_budget_reason = "not_weight";
    bool weight_budget_cpu_offload_issued = false;
    bool weight_budget_cpu_offload_success = false;
    const char* weight_budget_cpu_offload_error = "not_issued";

    if (kv_budget_tracked) {
        record_kv_allocation(static_cast<size_t>(size), &kv_budget_over_budget, &kv_budget_reason);
    }
    if (weight_budget_tracked) {
        record_weight_allocation(static_cast<size_t>(size), &weight_budget_over_budget, &weight_budget_reason);
    }

    PoolKind pool_kind = pool_kind_for_allocation(alloc_class);
    bool pool_registry_tracked =
        pool_registry_enabled &&
        pool_kind != PoolKind::OtherManaged &&
        device >= 0;
    const char* pool_object_state =
        pool_registry_tracked ? "active" : "untracked";
    if (pool_registry_tracked) {
        record_pool_registry_allocation(pool_kind, static_cast<size_t>(size));
    }

    bool driver_scratch_region_registered = false;
    uint64_t driver_scratch_region_id = 0;
    int driver_scratch_register_status = 0;
    bool driver_scratch_budget_reserved = false;
    bool driver_scratch_hard_reject = false;
    if (driver_scratch_tag_enable &&
            pool_registry_tracked &&
            pool_kind == PoolKind::RuntimeScratch &&
            strcmp(placement_backend, "managed") == 0 &&
            ptr != NULL &&
            static_cast<size_t>(size) > 0) {
        bool driver_scratch_over_budget = false;
        size_t driver_scratch_live_before = 0;
        size_t driver_scratch_live_after = 0;
        size_t driver_scratch_over_bytes = 0;
        bool driver_scratch_enforce_this_alloc =
            driver_scratch_budget_enforce != 0 &&
            (alloc_class != AllocationClass::WarmupWorkspace ||
             driver_scratch_budget_enforce_warmup != 0);

        driver_scratch_region_id = allocator_region_id(ptr, alloc_count);
        driver_scratch_budget_reserved = reserve_driver_scratch_budget(
            static_cast<size_t>(size),
            driver_scratch_enforce_this_alloc,
            &driver_scratch_over_budget,
            &driver_scratch_live_before,
            &driver_scratch_live_after
        );
        if (driver_scratch_budget_bytes > 0 &&
                driver_scratch_live_after > driver_scratch_budget_bytes) {
            driver_scratch_over_bytes =
                driver_scratch_live_after - driver_scratch_budget_bytes;
        }

        if (driver_scratch_budget_reserved) {
            driver_scratch_register_status = uvm_driver_phase_f_register_region_internal(
                driver_scratch_region_id,
                reinterpret_cast<uintptr_t>(ptr),
                static_cast<size_t>(size)
            );
            driver_scratch_region_registered = driver_scratch_register_status >= 0;
            if (!driver_scratch_region_registered) {
                release_driver_scratch_budget(static_cast<size_t>(size));
                driver_scratch_budget_reserved = false;
                driver_scratch_live_after = driver_scratch_live_bytes.load();
            }
        } else {
            driver_scratch_register_status = driver_scratch_over_budget ? -12 : -1;
            driver_scratch_hard_reject = driver_scratch_enforce_this_alloc;
        }

        if (log_file) {
            std::lock_guard<std::mutex> lock(log_mutex);
            if (driver_scratch_budget_reserved || driver_scratch_region_registered) {
                fprintf(log_file,
                        "[%.3fs] TRACE_DRIVER_SCRATCH_TAG alloc_id=%zu ptr=0x%llx "
                        "size_bytes=%zu region_id=%llu status=%d registered=%d "
                        "budget_reserved=%d budget_bytes=%zu live_before_bytes=%zu "
                        "live_after_bytes=%zu over_budget=%d over_bytes=%zu "
                        "enforce=%d enforce_warmup=%d phase=%s predicted_class=%s\n",
                        get_elapsed_seconds(),
                        alloc_count,
                        static_cast<unsigned long long>(
                            reinterpret_cast<uintptr_t>(ptr)),
                        static_cast<size_t>(size),
                        static_cast<unsigned long long>(driver_scratch_region_id),
                        driver_scratch_register_status,
                        driver_scratch_region_registered ? 1 : 0,
                        driver_scratch_budget_reserved ? 1 : 0,
                        driver_scratch_budget_bytes,
                        driver_scratch_live_before,
                        driver_scratch_live_after,
                        driver_scratch_over_budget ? 1 : 0,
                        driver_scratch_over_bytes,
                        driver_scratch_enforce_this_alloc ? 1 : 0,
                        driver_scratch_budget_enforce_warmup,
                        phase_snapshot.c_str(),
                        allocation_class_to_string(alloc_class));
            } else {
                fprintf(log_file,
                        "[%.3fs] TRACE_DRIVER_SCRATCH_REJECT alloc_id=%zu ptr=0x%llx "
                        "size_bytes=%zu region_id=%llu status=%d registered=0 "
                        "budget_reserved=0 budget_bytes=%zu live_before_bytes=%zu "
                        "live_after_bytes=%zu over_budget=%d over_bytes=%zu "
                        "enforce=%d enforce_warmup=%d phase=%s predicted_class=%s "
                        "reason=%s\n",
                        get_elapsed_seconds(),
                        alloc_count,
                        static_cast<unsigned long long>(
                            reinterpret_cast<uintptr_t>(ptr)),
                        static_cast<size_t>(size),
                        static_cast<unsigned long long>(driver_scratch_region_id),
                        driver_scratch_register_status,
                        driver_scratch_budget_bytes,
                        driver_scratch_live_before,
                        driver_scratch_live_after,
                        driver_scratch_over_budget ? 1 : 0,
                        driver_scratch_over_bytes,
                        driver_scratch_enforce_this_alloc ? 1 : 0,
                        driver_scratch_budget_enforce_warmup,
                        phase_snapshot.c_str(),
                        allocation_class_to_string(alloc_class),
                        driver_scratch_over_budget ?
                            "driver_scratch_budget_exceeded" :
                            "driver_scratch_budget_reserve_failed");
            }
            fflush(log_file);
        }
    }

    if (driver_scratch_hard_reject) {
        if (pool_registry_tracked) {
            release_pool_registry_allocation(pool_kind, static_cast<size_t>(size));
        }

        total_allocated.fetch_sub(static_cast<size_t>(size));
        cudaError_t free_rejected_err = cudaFree(ptr);
        if (free_rejected_err != cudaSuccess) {
            fprintf(stderr,
                    "[vLLM UVM] cudaFree failed after driver scratch hard reject "
                    "for %zd bytes: %s\n",
                    size,
                    cudaGetErrorString(free_rejected_err));
        }
        return NULL;
    }

    bool scratch_pool_tracked =
        scratch_pool_enable &&
        is_runtime_scratch_pool_allocation(alloc_class) &&
        device >= 0;
    bool scratch_pool_phase_match = is_scratch_pool_target_phase(phase_snapshot);
    bool scratch_pool_eligible = false;
    bool scratch_pool_device_direct = false;
    bool scratch_pool_budget_over_budget = false;
    bool scratch_pool_budget_reserved = false;
    const char* scratch_pool_reason = scratch_pool_tracked
        ? "scratch_pool_not_evaluated"
        : "not_scratch_pool";

    if (scratch_pool_tracked) {
        scratch_pool_trace_allocs.fetch_add(1);
        if (!scratch_pool_phase_match) {
            scratch_pool_reason = "scratch_pool_phase_not_allowed";
        } else if (static_cast<size_t>(size) < device_direct_min_bytes) {
            scratch_pool_reason = "scratch_pool_below_min_bytes";
        } else if (static_cast<size_t>(size) > device_direct_max_bytes) {
            scratch_pool_reason = "scratch_pool_above_max_bytes";
        } else if (!device_direct_enable) {
            scratch_pool_reason = "scratch_pool_device_direct_disabled";
        } else {
            scratch_pool_eligible = true;
            scratch_pool_eligible_allocs.fetch_add(1);
            bool over_budget = false;
            scratch_pool_budget_reserved =
                reserve_scratch_pool_device_direct_budget(
                    static_cast<size_t>(size),
                    &over_budget
                );
            scratch_pool_budget_over_budget = over_budget;
            if (over_budget) {
                scratch_pool_budget_over_allocs.fetch_add(1);
            }
            if (!scratch_pool_budget_reserved) {
                scratch_pool_budget_reject_allocs.fetch_add(1);
                scratch_pool_reason = "scratch_pool_budget_exceeded_fallback_managed";
            } else if (over_budget) {
                scratch_pool_reason = "scratch_pool_trace_only_over_budget_device_direct";
            } else {
                scratch_pool_reason = "scratch_pool_device_direct_enabled";
            }
        }
    }

    if (scratch_pool_eligible && scratch_pool_budget_reserved) {
        policy_action = PolicyAction::DeviceDirect;
        policy_source = "scratch_pool_policy";
        device_direct_requested = true;
        device_direct_reason = scratch_pool_reason;
    }

    /* ============================================================================
    * 第四阶段：Stage C 准入八要素检查 (Gatekeeping)
    * ============================================================================ */

    // 4.1 深度过滤：即使策略引擎想要开启 Device-Direct，也必须通过以下所有物理约束：
    if (device_direct_requested) {
        if (device < 0) {
            device_direct_reason = "invalid_device";
        } else if (kv_device_direct_requested) {
            if (static_cast<size_t>(size) < kv_device_direct_min_bytes) {
                device_direct_reason = "kv_device_direct_below_min_bytes";
            } else if (kv_device_direct_max_bytes > 0 &&
                       static_cast<size_t>(size) > kv_device_direct_max_bytes) {
                device_direct_reason = "kv_device_direct_above_max_bytes";
            } else {
                device_direct_eligible = true;
                device_direct_reason = "kv_device_direct_enabled";
            }
        } else if (fixed_weight_device_direct_requested) {
            if (static_cast<size_t>(size) < fixed_weight_device_direct_min_bytes) {
                device_direct_reason = "fixed_weight_device_direct_below_min_bytes";
            } else if (fixed_weight_device_direct_max_bytes > 0 &&
                       static_cast<size_t>(size) >
                           fixed_weight_device_direct_max_bytes) {
                device_direct_reason = "fixed_weight_device_direct_above_max_bytes";
            } else {
                device_direct_eligible = true;
                device_direct_reason = "fixed_weight_device_direct_enabled";
            }
        } else if (scratch_pool_budget_reserved) {
            device_direct_eligible = true;
        } else if (!hot_gap_match) {
            // 约束 1：必须在监测到的热点 Gap 区间内（防止盲目加速）。
            device_direct_reason = "no_hot_gap_match";
        } else if (!gap_watch_class_match) {
            // 约束 2：内存类别必须匹配目标（如 runtime_scratch）。
            device_direct_reason = "target_class_mismatch";
        } else if (!device_direct_phase_match) {
            // 约束 3：执行相位必须在白名单内（如 enabled:attention）。
            device_direct_reason = "phase_not_allowed";
        } else if (static_cast<size_t>(size) < device_direct_min_bytes) {
            // 约束 4：申请尺寸不能太小（避免碎块显存池碎片化）。
            device_direct_reason = "below_min_bytes";
        } else if (static_cast<size_t>(size) > device_direct_max_bytes) {
            // 约束 5：申请尺寸不能太大（防止单块内存瞬间耗尽显存预算）。
            device_direct_reason = "above_max_bytes";
        } else {
            // 最终准入：通过物理边界检查，具备进入 Device-Direct 路径的资格。
            device_direct_eligible = true;

            // 判定具体开启模式：是真实迁移还是仅在基准测试阶段做追踪 (Trace)。
            if (policy_action == PolicyAction::DeviceDirect && device_direct_enable) {
                device_direct_reason = "device_direct_enabled";
            } else if (policy_action == PolicyAction::DeviceDirectTrace) {
                device_direct_reason = "trace_action_only";
            } else {
                device_direct_reason = "trace_only_not_enabled";
            }
        }

        // 4.2 统计更新：记录该次请求的踪迹、请求的总字节数以及最终符合条件的合规计数。
        device_direct_trace_allocs.fetch_add(1);
        device_direct_requested_bytes.fetch_add(static_cast<size_t>(size));
        if (device_direct_eligible) {
            device_direct_eligible_allocs.fetch_add(1);
        }
    }

    /* ============================================================================
    * 第五阶段：执行物理分配替换 (Actual Device-Direct Allocation & Swap)
    * ============================================================================ */

    // 5.1 终极准入检查：
    // 只有同时满足：1. 符合八要素准入；2. 策略动作为真实分配；3. 全局开关开启；4. 初始 UVM 指针有效。
    if (device_direct_eligible &&
            policy_action == PolicyAction::DeviceDirect &&
            (device_direct_enable ||
             kv_device_direct_requested ||
             fixed_weight_device_direct_requested) &&
            ptr != NULL) {

        void* device_ptr = NULL;
        ScratchPoolReuseCacheEntry reuse_cache_entry{};
        bool scratch_pool_reuse_cache_hit = false;
        bool device_direct_budget_reserved_for_request = false;

        int previous_device = -1;
        cudaError_t get_device_err = cudaGetDevice(&previous_device);
        if (device >= 0) {
            cudaSetDevice(device);
        }

        // 5.2 预算预留 (Stage C1 核心逻辑)：
        // 在进行任何物理分配前，先尝试在账本中扣除预算（max_total_bytes）。
        // 这确保了纯显存的占用始终在安全范围内，防止因 Direct 分配导致 GPU 整体 OOM。
        bool use_async_backend = device_direct_backend == "cuda_malloc_async";
        if (scratch_pool_budget_reserved && use_async_backend) {
            scratch_pool_reuse_cache_hit = try_pop_scratch_pool_reuse_cache(
                size_bucket,
                static_cast<size_t>(size),
                device,
                &reuse_cache_entry
            );
            if (scratch_pool_reuse_cache_hit) {
                device_ptr = reuse_cache_entry.ptr;
                release_scratch_pool_device_direct_budget(
                    static_cast<size_t>(size)
                );
                scratch_pool_budget_reserved = false;
                scratch_pool_reason = "scratch_pool_reuse_cache_hit";
            }
        }

        bool budget_reserved = scratch_pool_reuse_cache_hit;
        if (!budget_reserved) {
            if (kv_device_direct_requested) {
                budget_reserved = reserve_device_direct_budget_with_limit(
                    static_cast<size_t>(size),
                    kv_device_direct_max_total_bytes
                );
            } else if (fixed_weight_device_direct_requested) {
                budget_reserved = reserve_device_direct_budget_with_limit(
                    static_cast<size_t>(size),
                    fixed_weight_device_direct_max_total_bytes
                );
            } else {
                budget_reserved = reserve_device_direct_budget(
                    static_cast<size_t>(size)
                );
            }
            device_direct_budget_reserved_for_request = budget_reserved;
        }

        if (!budget_reserved) {
            // 如果预算不足，记录原因并增加拒绝计数，随后程序将直接使用原本的 managed (UVM) 指针。
            device_direct_reason = kv_device_direct_requested
                ? "kv_device_direct_budget_exceeded"
                : (fixed_weight_device_direct_requested
                    ? "fixed_weight_device_direct_budget_exceeded"
                    : "device_direct_budget_exceeded");
            device_direct_budget_rejects.fetch_add(1);
            if (scratch_pool_budget_reserved) {
                release_scratch_pool_device_direct_budget(static_cast<size_t>(size));
                scratch_pool_budget_reserved = false;
                scratch_pool_reason = "scratch_pool_global_device_direct_budget_exceeded";
            }
        } else {
            // 5.4 后端选择与配置 (Stage C2 核心逻辑)：
            // 判断是使用同步的 cudaMalloc 还是异步的 cudaMallocAsync。
            bool pool_config_ready = true;
            std::string pool_config_error_snapshot = "none";

            if (!scratch_pool_reuse_cache_hit && use_async_backend) {
                // 如果使用异步后端，尝试配置 CUDA 默认内存池（如设置 Release Threshold）。
                pool_config_ready = configure_device_direct_async_pool_if_needed(device);
                if (!pool_config_ready) {
                    // 如果内存池配置失败（可能是 CUDA 版本或硬件不支持），记录错误快照。
                    std::lock_guard<std::mutex> pool_lock(device_direct_pool_mutex);
                    pool_config_error_snapshot = device_direct_pool_config_error;
                }
            }

            // 5.5 执行物理显存申请：
            cudaError_t device_alloc_err = cudaSuccess;
            if (scratch_pool_reuse_cache_hit) {
                device_alloc_err = cudaSuccess;
            } else if (pool_config_ready) {
                // 根据 backend 配置，调用对应的 CUDA API 申请纯显存（GPU-only）。
                device_alloc_err = use_async_backend
                    ? cudaMallocAsync(&device_ptr, static_cast<size_t>(size), stream)
                    : cudaMalloc(&device_ptr, static_cast<size_t>(size));
            } else {
                device_alloc_err = cudaErrorInvalidValue;
            }

            /* ====================================================================
            * 第六阶段：指针交换与清理 (The Swap Operation)
            * ==================================================================== */

            // 判定申请是否成功且获得了有效的纯显存指针
            if (device_alloc_err == cudaSuccess && device_ptr != NULL) {

                // 6.1 撤销原 Managed 内存：
                // 这是 Stage C 设计最精妙的地方：为了保持内存地址的唯一性，
                // 必须立刻释放掉最初通过 cudaMallocManaged 申请的“候选”指针。
                cudaError_t free_candidate_err = cudaFree(ptr);

                if (free_candidate_err == cudaSuccess) {
                    // 6.2 成功替换：
                    // 将返回指针 ptr 更新为纯显存指针 device_ptr。
                    // 自此，vLLM 的计算 Kernel 将直接访问物理显存，完全绕过 UVM 缺页机制。
                    ptr = device_ptr;
                    placement_backend = "device_direct";
                    device_direct_backend_used = device_direct_backend.c_str();
                    policy_action_success = true;
                    policy_action_error = "none";
                    device_direct_reason = scratch_pool_reuse_cache_hit
                        ? "device_direct_reuse_cache_hit"
                        : (kv_device_direct_requested
                            ? "kv_device_direct_enabled"
                            : (fixed_weight_device_direct_requested
                                ? "fixed_weight_device_direct_enabled"
                                : "device_direct_enabled"));
                    if (scratch_pool_budget_reserved ||
                            scratch_pool_reuse_cache_hit) {
                        scratch_pool_device_direct = true;
                        if (scratch_pool_reuse_cache_hit) {
                            scratch_pool_reason = "scratch_pool_reuse_cache_hit";
                        } else {
                            scratch_pool_reason =
                                scratch_pool_budget_over_budget
                                    ? "scratch_pool_trace_only_over_budget_device_direct"
                                    : "scratch_pool_device_direct_enabled";
                        }
                        scratch_pool_device_direct_allocs.fetch_add(1);
                        scratch_pool_device_direct_bytes.fetch_add(
                            static_cast<size_t>(size)
                        );
                        if (!scratch_pool_reuse_cache_hit &&
                                scratch_pool_reuse_cache_available()) {
                            scratch_pool_reuse_cache_fresh_backend_allocs
                                .fetch_add(1);
                        }
                    }

                    // 更新统计指标
                    device_direct_actual_allocs.fetch_add(1);
                    device_direct_actual_bytes.fetch_add(static_cast<size_t>(size));
                } else {
                    // 6.3 异常分支 A：Managed 内存释放失败
                    // 这是一个极其罕见的异常。为了防止内存泄露，必须立刻释放刚申请的 device_ptr。
                    cudaError_t cleanup_device_err =
                        (use_async_backend || scratch_pool_reuse_cache_hit)
                            ? cudaFreeAsync(device_ptr, stream)
                            : cudaFree(device_ptr);

                    // 释放刚才预留的预算，因为这次替换失败了。
                    if (cleanup_device_err == cudaSuccess) {
                        if (scratch_pool_reuse_cache_hit) {
                            release_device_direct_budget(static_cast<size_t>(size));
                            release_scratch_pool_device_direct_budget(
                                static_cast<size_t>(size)
                            );
                        } else if (device_direct_budget_reserved_for_request) {
                            release_device_direct_budget(static_cast<size_t>(size));
                        }
                        if (!scratch_pool_reuse_cache_hit &&
                                scratch_pool_budget_reserved) {
                            release_scratch_pool_device_direct_budget(
                                static_cast<size_t>(size)
                            );
                            scratch_pool_budget_reserved = false;
                        }
                    }
                    policy_action_success = false;
                    policy_action_error = cudaGetErrorString(free_candidate_err);
                    device_direct_reason = "managed_candidate_free_failed";
                    if (scratch_pool_tracked) {
                        scratch_pool_reason =
                            "scratch_pool_managed_candidate_free_failed";
                    }

                    // 回退（Fallback）：程序将继续使用原本的 managed 指针，虽然性能没提升但不会崩。
                    device_direct_fallback_allocs.fetch_add(1);
                }
            } else {
                // 6.4 异常分支 B：纯显存申请失败
                // 如果因为显存不足等原因导致 cudaMalloc(Async) 失败：

                // 释放预留预算
                if (device_direct_budget_reserved_for_request) {
                    release_device_direct_budget(static_cast<size_t>(size));
                }
                if (scratch_pool_budget_reserved) {
                    release_scratch_pool_device_direct_budget(
                        static_cast<size_t>(size)
                    );
                    scratch_pool_budget_reserved = false;
                }
                policy_action_success = false;

                // 记录详细的失败原因（是 Backend 报错还是池配置报错）
                policy_action_error = pool_config_ready
                    ? cudaGetErrorString(device_alloc_err)
                    : pool_config_error_snapshot.c_str();

                if (!pool_config_ready) {
                    device_direct_reason = "device_direct_pool_config_failed";
                    if (scratch_pool_tracked) {
                        scratch_pool_reason =
                            "scratch_pool_device_direct_pool_config_failed";
                    }
                } else {
                    device_direct_reason = use_async_backend
                        ? "device_malloc_async_failed_fallback_managed"
                        : "device_malloc_failed_fallback_managed";
                    if (scratch_pool_tracked) {
                        scratch_pool_reason = use_async_backend
                            ? "scratch_pool_malloc_async_failed_fallback_managed"
                            : "scratch_pool_malloc_failed_fallback_managed";
                    }
                }

                // 回退（Fallback）：记录回退计数。
                device_direct_fallback_allocs.fetch_add(1);
            }
        }

        // 5.6 环境恢复：切回之前的 CUDA 设备
        if (get_device_err == cudaSuccess && previous_device >= 0) {
            cudaSetDevice(previous_device);
        }
    }

    if (weight_budget_tracked &&
            weight_budget_over_budget &&
            weight_budget_mode == "enforce" &&
            weight_budget_cpu_offload_enable != 0 &&
            ptr != NULL) {
        weight_budget_cpu_offload_issued = true;
        weight_budget_cpu_offload_success = offload_managed_range_to_cpu(
            ptr,
            static_cast<size_t>(size),
            stream,
            &weight_budget_cpu_offload_error
        );
        if (weight_budget_cpu_offload_success) {
            weight_budget_cpu_offload_allocs.fetch_add(1);
            weight_budget_cpu_offload_bytes.fetch_add(static_cast<size_t>(size));
            weight_budget_reason = "weight_budget_exceeded_offloaded_cpu";
        } else {
            weight_budget_cpu_offload_failures.fetch_add(1);
            weight_budget_reason = "weight_budget_exceeded_offload_cpu_failed";
        }
    }

    // 结构化的详细追踪日志，用于后续的离线分析或性能重放关联
    if (size >= trace_min_bytes) {
        trace_allocation_event(
            ptr, static_cast<size_t>(size), alloc_count, current,
            peak_allocated.load(), device, phase_snapshot
        );
    }

    if (gap_overlap_bytes > 0) {
        gap_watch_overlap_allocs.fetch_add(1);
        gap_watch_overlap_bytes_total.fetch_add(gap_overlap_bytes);
        if (gap_watch_class_match) {
            gap_watch_target_class_match_allocs.fetch_add(1);
        }
    }
    bool store_active_info =
        static_cast<size_t>(size) >= trace_min_bytes ||
        log_unknown_detail || log_gap_watch ||
        strcmp(placement_backend, "device_direct") == 0 ||
        kv_budget_tracked ||
        weight_budget_tracked ||
        pool_registry_tracked;

    // 7.1 记录活跃分配元数据，供 free / gap watch / unknown detail 使用
    if (store_active_info) {
        std::lock_guard<std::mutex> lock(log_mutex);
        active_allocations[ptr] = AllocationInfo{
            static_cast<size_t>(size),
            device,
            alloc_count,
            phase_snapshot,
            alloc_elapsed,
            size_bucket,
            allocation_class_to_string(alloc_class),
            policy_action_to_string(policy_action),
            policy_source,
            true,
            "none",
            log_unknown_detail,
            log_gap_watch,
            gap_overlap_start,
            gap_overlap_end,
            gap_overlap_bytes,
            gap_watch_target_class,
            policy_action_to_string(gap_watch_policy_action),
            hot_gap_match,
            placement_backend,
            device_direct_backend_used,
            device_direct_eligible,
            device_direct_reason,
            cpu_access_risk,
            kv_budget_tracked,
            kv_budget_over_budget,
            kv_budget_reason,
            weight_budget_tracked,
            weight_budget_over_budget,
            weight_budget_reason,
            weight_budget_cpu_offload_issued,
            weight_budget_cpu_offload_success,
            weight_budget_cpu_offload_error,
            pool_registry_tracked,
            pool_kind_to_string(pool_kind),
            pool_object_state,
            driver_scratch_region_registered,
            driver_scratch_region_id,
            driver_scratch_register_status,
            driver_scratch_budget_reserved,
            scratch_pool_tracked,
            scratch_pool_eligible,
            scratch_pool_device_direct,
            scratch_pool_budget_over_budget,
            scratch_pool_reason,
        };
    }

    // 8. 策略执行阶段：主动显存调优
    if ((policy_action == PolicyAction::ManagedPrefetchGpu ||
         policy_action == PolicyAction::ManagedAdvisePrefetchGpu) &&
        ptr != NULL) {

        // 8.1 内存建议 (MemAdvise)：提示 CUDA 驱动该内存的首选物理位置在 GPU 上
        // 这可以在真正发生缺页中断时，降低驱动寻找最佳页存放地的开销。
        bool should_advise =
            policy_action == PolicyAction::ManagedAdvisePrefetchGpu ||
            policy_warmup_advise_gpu;
        if (should_advise) {
            cudaError_t advise_err = cudaMemAdvise(
                ptr,
                static_cast<size_t>(size),
                cudaMemAdviseSetPreferredLocation,
                device
            );
            if (advise_err != cudaSuccess) {
                policy_action_success = false;
                policy_action_error = cudaGetErrorString(advise_err);
            }
        }

        // 8.2 异步预取 (Prefetch Async)
        // 主动将内存页迁移到 GPU 上，避免首次访问时触发昂贵的 Page Fault。
        // 工程考量：大规模的 GPU 跨块预取 (Cross-block prefetch) 极易引发 PCIe 带宽被严重挤占，
        // 反而会导致其他并发的推理任务因总线竞争而性能劣化。
        // 因此这里被严格约束在特定 policy（如专门的 warmup 阶段或极特定的内存类）下才触发。
        cudaError_t prefetch_err = cudaMemPrefetchAsync(
            ptr,
            static_cast<size_t>(size),
            device,
            stream // 绑定到特定流，允许与该流上的计算重叠
        );
        if (prefetch_err != cudaSuccess) {
            policy_action_success = false;
            policy_action_error = cudaGetErrorString(prefetch_err);
        }
    }

    /* ============================================================================
    * 第七阶段：Gap-Watch 策略效能统计 (Policy Effectiveness Tracking)
    * ============================================================================ */

    // 7.1 判定是否成功应用了针对性优化：
    // 如果本次分配命中了 Gap 区间，且决策确实来源于 gap_watch_policy，
    // 并且动作不是默认的 Managed 模式，则进入统计逻辑。
    if (gap_overlap_bytes > 0 &&
            policy_source != nullptr &&
            strcmp(policy_source, "gap_watch_policy") == 0 &&
            policy_action != PolicyAction::ManagedDefault) {

        // 记录受该策略影响的分配次数及重叠的总字节数
        gap_watch_policy_applied_allocs.fetch_add(1);
        gap_watch_policy_applied_overlap_bytes.fetch_add(gap_overlap_bytes);

        // 细分统计：策略执行是成功（例如成功替换为纯显存）还是失败（由于 OOM 或配置错误回退）
        if (policy_action_success) {
            gap_watch_policy_success_allocs.fetch_add(1);
        } else {
            gap_watch_policy_failed_allocs.fetch_add(1);
        }
    }

    /* ============================================================================
    * 第八阶段：更新活跃分配元数据 (Metadata State Synchronization)
    * ============================================================================ */

    // 8.1 检查是否需要持久化本次分配的信息：
    // store_active_info 是在函数前期判定的标志位，通常对于大内存、命中 Gap
    // 或被标记为 KV/Weight 的分配为真。
    if (store_active_info) {

        // 使用互斥锁保护全局 map (active_allocations)，确保多线程环境下元数据写入的安全。
        std::lock_guard<std::mutex> lock(log_mutex);

        // 在映射表中查找刚刚分配成功的指针 (ptr)
        auto it = active_allocations.find(ptr);
        if (it != active_allocations.end()) {

            // --- 核心元数据注入 ---

            // 记录最终采取的策略名称（如 "ManagedPrefetchGpu" 或 "DeviceDirect"）
            it->second.policy_action_name = policy_action_to_string(policy_action);
            // 记录决策来源（是基础启发式逻辑还是精确的 Gap-Watch 逻辑）
            it->second.policy_source_name = policy_source ? policy_source : "unknown";

            // 记录策略执行的具体结果及错误信息（若有）
            it->second.policy_action_success = policy_action_success;
            it->second.policy_action_error = policy_action_error;

            // --- Stage C 相关记录 ---

            it->second.gap_watch_target_class_name = gap_watch_target_class;
            it->second.gap_watch_policy_action_name =
                policy_action_to_string(gap_watch_policy_action);

            // 标记最终内存驻留的后端（"managed" 或 "device_direct"）
            it->second.placement_backend_name = placement_backend;
            // 记录具体使用的 Backend API（如 "cuda_malloc_async"）
            it->second.device_direct_backend_name = device_direct_backend_used;

            // 记录 Device-Direct 的准入详情，用于离线分析为何某些分配没能成功提速
            it->second.device_direct_eligible = device_direct_eligible;
            it->second.device_direct_reason = device_direct_reason;

            // 记录风险等级与热点命中状态
            it->second.cpu_access_risk = cpu_access_risk;
            it->second.hot_gap_match = hot_gap_match;

            // --- Stage E 相关记录：组件预算快照 ---

            // 记录 KV Cache 和模型权重（Weight）的预算追踪状态，
            // 包括它们是否超预算以及具体的拒绝理由。
            it->second.kv_budget_tracked = kv_budget_tracked;
            it->second.kv_budget_over_budget = kv_budget_over_budget;
            it->second.kv_budget_reason = kv_budget_reason;

            it->second.weight_budget_tracked = weight_budget_tracked;
            it->second.weight_budget_over_budget = weight_budget_over_budget;
            it->second.weight_budget_reason = weight_budget_reason;
            it->second.weight_budget_cpu_offload_issued =
                weight_budget_cpu_offload_issued;
            it->second.weight_budget_cpu_offload_success =
                weight_budget_cpu_offload_success;
            it->second.weight_budget_cpu_offload_error =
                weight_budget_cpu_offload_error;
            it->second.pool_registry_tracked = pool_registry_tracked;
            it->second.pool_kind_name = pool_kind_to_string(pool_kind);
            it->second.pool_object_state = pool_object_state;
            it->second.driver_scratch_region_registered =
                driver_scratch_region_registered;
            it->second.driver_scratch_region_id = driver_scratch_region_id;
            it->second.driver_scratch_register_status =
                driver_scratch_register_status;
            it->second.driver_scratch_budget_reserved =
                driver_scratch_budget_reserved;
            it->second.scratch_pool_tracked = scratch_pool_tracked;
            it->second.scratch_pool_eligible = scratch_pool_eligible;
            it->second.scratch_pool_device_direct = scratch_pool_device_direct;
            it->second.scratch_pool_budget_over_budget =
                scratch_pool_budget_over_budget;
            it->second.scratch_pool_reason = scratch_pool_reason;
        }
    }

    trace_pool_alloc_event(
        ptr,
        static_cast<size_t>(size),
        alloc_count,
        device,
        phase_snapshot,
        alloc_class,
        pool_kind,
        size_bucket,
        placement_backend,
        device_direct_backend_used,
        scratch_pool_tracked,
        scratch_pool_eligible,
        scratch_pool_device_direct,
        scratch_pool_budget_over_budget,
        scratch_pool_reason,
        pool_object_state
    );

    // 记录策略引擎的判定与执行结果
    if (policy_enabled) {
        trace_policy_event(
            ptr, static_cast<size_t>(size), alloc_count, device,
            phase_snapshot, alloc_class, policy_action,
            policy_source,
            gap_watch_class_match,
            gap_overlap_bytes,
            size_bucket,
            placement_backend,
            device_direct_backend_used,
            device_direct_eligible,
            device_direct_reason,
            cpu_access_risk,
            kv_budget_tracked,
            kv_budget_over_budget,
            kv_budget_reason,
            weight_budget_tracked,
            weight_budget_over_budget,
            weight_budget_reason,
            scratch_pool_tracked,
            scratch_pool_eligible,
            scratch_pool_device_direct,
            scratch_pool_budget_over_budget,
            scratch_pool_reason,
            hot_gap_match,
            policy_action_success, policy_action_error
        );
    }

    if (log_unknown_detail) {
        trace_unknown_detail_event(
            ptr,
            static_cast<size_t>(size),
            alloc_count,
            device,
            phase_snapshot,
            size_bucket,
            policy_action,
            stream,
            overlaps_gap_watch,
            gap_overlap_start,
            gap_overlap_end,
            gap_overlap_bytes
        );
    }
    if (log_gap_watch) {
        trace_gap_watch_alloc_event(
            ptr,
            static_cast<size_t>(size),
            alloc_count,
            device,
            phase_snapshot,
            alloc_class,
            policy_action,
            policy_source,
            gap_watch_class_match,
            size_bucket,
            stream,
            gap_overlap_start,
            gap_overlap_end,
            gap_overlap_bytes,
            placement_backend,
            device_direct_backend_used,
            device_direct_eligible,
            device_direct_reason,
            cpu_access_risk,
            kv_budget_tracked,
            kv_budget_over_budget,
            kv_budget_reason,
            weight_budget_tracked,
            weight_budget_over_budget,
            weight_budget_reason,
            scratch_pool_tracked,
            scratch_pool_eligible,
            scratch_pool_device_direct,
            scratch_pool_budget_over_budget,
            scratch_pool_reason,
            hot_gap_match
        );
    }

    // 9. 全局暴力预取 Fallback (通常默认关闭)
    // 如果系统开启了全局预取 (enable_prefetch=1)，且策略引擎没有覆盖该内存块，则执行普适性预取。
    // 在生产环境的推理服务中，由于 PCIe 带宽的脆弱性，此选项通常不建议保持常开。
    if (enable_prefetch && device >= 0 && ptr != NULL &&
        policy_action != PolicyAction::ManagedPrefetchGpu &&
        policy_action != PolicyAction::ManagedAdvisePrefetchGpu) {
        cudaMemPrefetchAsync(ptr, size, device, stream);
    }

    return ptr;
}

/**
 * 释放 CUDA 托管内存或纯显存 (Device-Direct)
 *
 * @param ptr 指向待释放内存的指针
 * @param size 分配的大小（用于统计更新）
 * @param device CUDA 设备 ID（内部追踪使用）
 * @param stream CUDA 流（用于异步释放）
 */
void uvm_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
    if (ptr != NULL) {
        // 用于存储从映射表中找出的元数据快照
        AllocationInfo info{};
        bool has_info = false;
        std::string phase_snapshot;

        /* ============================================================================
         * 第一阶段：元数据检索与状态清理 (Metadata Retrieval)
         * ============================================================================ */
        {
            // 使用互斥锁保护全局 active_allocations 映射表
            std::lock_guard<std::mutex> lock(log_mutex);

            // 捕获当前的运行相位（如 "attention"、"prefill" 等）
            phase_snapshot = current_phase.empty() ? "unscoped" : current_phase;

            // 在全局字典中查找该指针的“身世信息”
            auto it = active_allocations.find(ptr);
            if (it != active_allocations.end()) {
                info = it->second; // 拷贝元数据快照
                has_info = true;

                // 【关键操作】从活跃表中移除，防止重复释放或过时的统计
                active_allocations.erase(it);
            }
        }

        /* ============================================================================
         * 第二阶段：确定释放后端 (Backend Symmetry Selection)
         * ============================================================================ */

        // 判断这块内存当初是驻留在 UVM (managed) 还是纯显存 (device_direct)
        bool is_device_direct =
            has_info && info.placement_backend_name == "device_direct";

        // 判断是否需要使用异步释放。
        // 注意：分配与释放必须对称。如果当初是用 cudaMallocAsync 分配的，现在必须用 cudaFreeAsync。
        bool use_async_backend =
            is_device_direct &&
            info.device_direct_backend_name == "cuda_malloc_async";

        bool retained_in_scratch_reuse_cache = false;
        if (is_device_direct &&
                info.scratch_pool_device_direct &&
                use_async_backend &&
                scratch_pool_reuse_cache_available()) {
            retained_in_scratch_reuse_cache = try_insert_scratch_pool_reuse_cache(
                ptr,
                info.size,
                info.device,
                info.size_bucket,
                stream
            );
        }

        // 执行真实的 CUDA 释放操作；Phase 7 cache 命中时只做逻辑 free，物理指针保留复用。
        cudaError_t err = cudaSuccess;
        if (!retained_in_scratch_reuse_cache) {
            err = use_async_backend
                ? cudaFreeAsync(ptr, stream)
                : cudaFree(ptr);
        }

        if (err != cudaSuccess) {
            fprintf(stderr, "[vLLM UVM] %s failed: %s\n",
                    use_async_backend ? "cudaFreeAsync" : "cudaFree",
                    cudaGetErrorString(err));
        }

        /* ============================================================================
         * 第三阶段：全局统计与预算回收 (Budget Reconciliation)
         * ============================================================================ */

        // 3.1 更新全局水位线：原子减去当前释放的大小
        size_t current = total_allocated.fetch_sub(size) - size;
        // 增加释放次数统计
        size_t free_count = num_frees.fetch_add(1) + 1;

        // 3.2 Stage C1 预算回收：
        // 如果是直连显存，释放后必须归还之前在 reserve_device_direct_budget 中扣除的额度
        if (has_info &&
            info.placement_backend_name == "device_direct" &&
            err == cudaSuccess) {
            if (!retained_in_scratch_reuse_cache) {
                release_device_direct_budget(info.size);
                device_direct_free_success_allocs.fetch_add(1);
                if (info.scratch_pool_device_direct) {
                    release_scratch_pool_device_direct_budget(info.size);
                    scratch_pool_device_direct_free_success_allocs.fetch_add(1);
                }
            }
        }

        // 3.3 KV Cache 预算回收：
        // 归还 KV Cache 专用的显存额度，确保下一轮 Prefill 有足够的空间
        if (has_info && info.kv_budget_tracked && err == cudaSuccess) {
            release_kv_budget(info.size);
            kv_free_success_allocs.fetch_add(1);
        }

        // 3.4 模型权重预算回收 (Stage E)：
        // 如果释放的是模型权重（常见于多模型切换或专家卸载场景），归还权重预算
        if (has_info && info.weight_budget_tracked && err == cudaSuccess) {
            release_weight_budget(info.size);
            weight_free_success_allocs.fetch_add(1);
        }

        // 3.5 Stage F pool registry 回收：
        // 只更新 telemetry，不触发任何驱逐、迁移或预取动作。
        if (has_info && info.pool_registry_tracked && err == cudaSuccess) {
            release_pool_registry_allocation(
                pool_kind_from_string(info.pool_kind_name),
                info.size
            );
            info.pool_object_state = "freed";
        }

        if (has_info &&
                info.driver_scratch_region_registered &&
                info.driver_scratch_region_id != 0) {
            int unregister_status = uvm_driver_phase_f_unregister_region_internal(
                info.driver_scratch_region_id
            );
            bool scratch_budget_released = false;
            size_t scratch_live_after_release = driver_scratch_live_bytes.load();
            if (info.driver_scratch_budget_reserved) {
                release_driver_scratch_budget(info.size);
                scratch_budget_released = true;
                scratch_live_after_release = driver_scratch_live_bytes.load();
            }
            if (log_file) {
                std::lock_guard<std::mutex> lock(log_mutex);
                fprintf(log_file,
                        "[%.3fs] TRACE_DRIVER_SCRATCH_UNTAG free_id=%zu "
                        "ptr=0x%llx size_bytes=%zu region_id=%llu status=%d "
                        "alloc_id=%zu budget_reserved=%d budget_released=%d "
                        "budget_bytes=%zu live_after_bytes=%zu phase=%s\n",
                        get_elapsed_seconds(),
                        free_count,
                        static_cast<unsigned long long>(
                            reinterpret_cast<uintptr_t>(ptr)),
                        info.size,
                        static_cast<unsigned long long>(
                            info.driver_scratch_region_id),
                        unregister_status,
                        info.alloc_id,
                        info.driver_scratch_budget_reserved ? 1 : 0,
                        scratch_budget_released ? 1 : 0,
                        driver_scratch_budget_bytes,
                        scratch_live_after_release,
                        phase_snapshot.c_str());
                fflush(log_file);
            }
        }

        /* ============================================================================
         * 第四阶段：遥测与追踪记录 (Telemetry & Trace)
         * ============================================================================ */

        // 4.1 大块内存审计：
        // 对于超过 100MB 的内存释放进行专项文件记录，用于分析显存剧烈波动的诱因
        if (size > 100 * 1024 * 1024) {
            log_free(size, free_count, current, device);
        }

        // 4.2 结构化事件追踪：
        // 将释放事件记录到 trace log 中。这些日志后续会被分析脚本用来计算“内存生命周期”。
        if (static_cast<size_t>(size) >= trace_min_bytes || has_info) {
            trace_free_event(
                ptr,
                static_cast<size_t>(size),
                free_count,
                current,
                device,
                has_info ? &info : nullptr,
                phase_snapshot
            );
        }

        // 4.3 专项分析事件：
        // 记录关于“未知内存详情”和“Gap 监控”的释放事件，
        // 帮助开发者确认之前应用的 Gap 策略是否在释放时能够完美闭环。
        if (has_info) {
            trace_unknown_detail_free_event(
                ptr,
                static_cast<size_t>(size),
                free_count,
                current,
                device,
                &info,
                phase_snapshot
            );
            trace_gap_watch_free_event(
                ptr,
                static_cast<size_t>(size),
                free_count,
                current,
                device,
                &info,
                phase_snapshot
            );
            trace_pool_free_event(
                ptr,
                static_cast<size_t>(size),
                free_count,
                device,
                &info,
                phase_snapshot
            );
        }
    }
}

void uvm_set_phase(const char* phase) {
    if (!log_initialized) {
        init_log_file();
    }

    const char* safe_phase = (phase && phase[0] != '\0') ? phase : "unscoped";
    {
        std::lock_guard<std::mutex> lock(log_mutex);
        current_phase = safe_phase;
    }
    log_phase_event("set", safe_phase);
}

void uvm_mark_phase_event(const char* event, const char* phase) {
    if (!log_initialized) {
        init_log_file();
    }

    if (phase && phase[0] != '\0') {
        log_phase_event(event, phase);
        return;
    }

    std::string phase_snapshot;
    {
        std::lock_guard<std::mutex> lock(log_mutex);
        phase_snapshot = current_phase.empty() ? "unscoped" : current_phase;
    }
    log_phase_event(event, phase_snapshot.c_str());
}

// =============================================================================
// Statistics API (can be called from Python via ctypes)
// =============================================================================

/**
 * Get current allocated bytes
 */
size_t uvm_get_allocated_bytes(void) {
    return total_allocated.load();
}

/**
 * Get peak allocated bytes
 */
size_t uvm_get_peak_allocated_bytes(void) {
    return peak_allocated.load();
}

/**
 * Get total number of allocations
 */
size_t uvm_get_num_allocs(void) {
    return num_allocs.load();
}

/**
 * Get total number of frees
 */
size_t uvm_get_num_frees(void) {
    return num_frees.load();
}

/**
 * Reset peak statistics to current allocation
 */
void uvm_reset_peak_stats(void) {
    peak_allocated.store(total_allocated.load());
}

/**
 * Reset all statistics
 */
void uvm_reset_all_stats(void) {
    clear_scratch_pool_reuse_cache();
    total_allocated.store(0);
    peak_allocated.store(0);
    num_allocs.store(0);
    num_frees.store(0);
    device_direct_live_bytes.store(0);
    device_direct_peak_live_bytes.store(0);
    device_direct_budget_rejects.store(0);
    kv_trace_allocs.store(0);
    kv_requested_bytes.store(0);
    kv_live_bytes.store(0);
    kv_peak_live_bytes.store(0);
    kv_budget_over_allocs.store(0);
    kv_budget_reject_allocs.store(0);
    kv_free_success_allocs.store(0);
    weight_trace_allocs.store(0);
    weight_requested_bytes.store(0);
    weight_live_bytes.store(0);
    weight_peak_live_bytes.store(0);
    weight_budget_over_allocs.store(0);
    weight_budget_reject_allocs.store(0);
    weight_budget_cpu_offload_allocs.store(0);
    weight_budget_cpu_offload_bytes.store(0);
    weight_budget_cpu_offload_failures.store(0);
    weight_free_success_allocs.store(0);
    pool_registry_tracked_allocs.store(0);
    pool_registry_free_success_allocs.store(0);
    pool_registry_live_objects.store(0);
    pool_registry_peak_live_objects.store(0);
    pool_kv_allocs.store(0);
    pool_kv_requested_bytes.store(0);
    pool_kv_live_bytes.store(0);
    pool_kv_peak_live_bytes.store(0);
    pool_kv_free_success_allocs.store(0);
    pool_weight_allocs.store(0);
    pool_weight_requested_bytes.store(0);
    pool_weight_live_bytes.store(0);
    pool_weight_peak_live_bytes.store(0);
    pool_weight_free_success_allocs.store(0);
    pool_runtime_scratch_allocs.store(0);
    pool_runtime_scratch_requested_bytes.store(0);
    pool_runtime_scratch_live_bytes.store(0);
    pool_runtime_scratch_peak_live_bytes.store(0);
    pool_runtime_scratch_free_success_allocs.store(0);
    driver_scratch_live_bytes.store(0);
    driver_scratch_peak_live_bytes.store(0);
    driver_scratch_budget_reject_allocs.store(0);
    driver_scratch_budget_over_allocs.store(0);
    driver_scratch_budget_release_allocs.store(0);
    scratch_pool_trace_allocs.store(0);
    scratch_pool_eligible_allocs.store(0);
    scratch_pool_device_direct_allocs.store(0);
    scratch_pool_device_direct_bytes.store(0);
    scratch_pool_device_direct_live_bytes.store(0);
    scratch_pool_device_direct_peak_live_bytes.store(0);
    scratch_pool_budget_over_allocs.store(0);
    scratch_pool_budget_reject_allocs.store(0);
    scratch_pool_device_direct_free_success_allocs.store(0);
    scratch_pool_reuse_cache_hit_allocs.store(0);
    scratch_pool_reuse_cache_miss_allocs.store(0);
    scratch_pool_reuse_cache_fresh_backend_allocs.store(0);
    scratch_pool_reuse_cache_insert_allocs.store(0);
    scratch_pool_reuse_cache_insert_reject_allocs.store(0);
    scratch_pool_reuse_cache_hit_bytes.store(0);
    scratch_pool_reuse_cache_insert_bytes.store(0);
    scratch_pool_reuse_cache_cached_bytes.store(0);
    scratch_pool_reuse_cache_peak_cached_bytes.store(0);
    scratch_pool_reuse_cache_cached_entries.store(0);
    scratch_pool_reuse_cache_event_not_ready.store(0);
    scratch_pool_reuse_cache_event_errors.store(0);
}

/**
 * Enable/disable prefetching
 */
void uvm_set_prefetch(int enabled) {
    enable_prefetch = enabled;
}

/**
 * Enable/disable verbose logging
 */
void uvm_set_verbose(int enabled) {
    verbose_logging = enabled;
}

/**
 * Flush and close the log file, writing a summary
 */
void uvm_close_log(void) {
    if (!log_file) return;

    std::lock_guard<std::mutex> lock(log_mutex);

    char timestamp[64];
    get_timestamp(timestamp, sizeof(timestamp));
    double elapsed = get_elapsed_seconds();

    fprintf(log_file, "========================================\n");
    fprintf(log_file, "[%s] Session Summary (duration: %.2fs)\n", timestamp, elapsed);
    fprintf(log_file, "  Total allocations: %zu\n", num_allocs.load());
    fprintf(log_file, "  Total frees: %zu\n", num_frees.load());
    fprintf(log_file, "  Current allocated: %.2f GB\n", total_allocated.load() / 1e9);
    fprintf(log_file, "  Peak allocated: %.2f GB\n", peak_allocated.load() / 1e9);
    fprintf(log_file, "  Gap-watch overlap allocations: %zu\n", gap_watch_overlap_allocs.load());
    fprintf(log_file, "  Gap-watch overlap bytes total: %zu\n", gap_watch_overlap_bytes_total.load());
    fprintf(log_file, "  Gap-watch target-class matches: %zu\n", gap_watch_target_class_match_allocs.load());
    fprintf(log_file, "  Gap-watch policy applied: %zu\n", gap_watch_policy_applied_allocs.load());
    fprintf(log_file, "  Gap-watch policy applied overlap bytes: %zu\n", gap_watch_policy_applied_overlap_bytes.load());
    fprintf(log_file, "  Gap-watch policy success: %zu\n", gap_watch_policy_success_allocs.load());
    fprintf(log_file, "  Gap-watch policy failed: %zu\n", gap_watch_policy_failed_allocs.load());
    fprintf(log_file, "  Device-direct trace allocations: %zu\n", device_direct_trace_allocs.load());
    fprintf(log_file, "  Device-direct eligible allocations: %zu\n", device_direct_eligible_allocs.load());
    fprintf(log_file, "  Device-direct requested bytes: %zu\n", device_direct_requested_bytes.load());
    fprintf(log_file, "  Device-direct backend: %s\n", device_direct_backend.c_str());
    fprintf(log_file, "  Device-direct max total bytes: %zu\n", device_direct_max_total_bytes);
    {
        std::lock_guard<std::mutex> pool_lock(device_direct_pool_mutex);
        fprintf(log_file, "  Device-direct pool release threshold set: %d\n",
                device_direct_pool_release_threshold_set ? 1 : 0);
        fprintf(log_file, "  Device-direct pool release threshold: %zu\n",
                device_direct_pool_release_threshold);
        fprintf(log_file, "  Device-direct pool config attempted: %d\n",
                device_direct_pool_config_attempted ? 1 : 0);
        fprintf(log_file, "  Device-direct pool config success: %d\n",
                device_direct_pool_config_success ? 1 : 0);
        fprintf(log_file, "  Device-direct pool config device: %d\n",
                device_direct_pool_config_device);
        fprintf(log_file, "  Device-direct pool config error: %s\n",
                device_direct_pool_config_error.c_str());
    }
    fprintf(log_file, "  Device-direct actual allocations: %zu\n", device_direct_actual_allocs.load());
    fprintf(log_file, "  Device-direct actual bytes: %zu\n", device_direct_actual_bytes.load());
    fprintf(log_file, "  Device-direct live bytes: %zu\n", device_direct_live_bytes.load());
    fprintf(log_file, "  Device-direct peak live bytes: %zu\n", device_direct_peak_live_bytes.load());
    fprintf(log_file, "  Device-direct budget rejects: %zu\n", device_direct_budget_rejects.load());
    fprintf(log_file, "  Device-direct fallback allocations: %zu\n", device_direct_fallback_allocs.load());
    fprintf(log_file, "  Device-direct free success: %zu\n", device_direct_free_success_allocs.load());
    fprintf(log_file, "  KV budget bytes: %zu\n", kv_budget_bytes);
    fprintf(log_file, "  KV budget mode: %s\n", kv_budget_mode.c_str());
    fprintf(log_file, "  KV trace allocations: %zu\n", kv_trace_allocs.load());
    fprintf(log_file, "  KV requested bytes: %zu\n", kv_requested_bytes.load());
    fprintf(log_file, "  KV live bytes: %zu\n", kv_live_bytes.load());
    fprintf(log_file, "  KV peak live bytes: %zu\n", kv_peak_live_bytes.load());
    fprintf(log_file, "  KV budget remaining: %zu\n",
            kv_budget_remaining_snapshot(kv_live_bytes.load()));
    fprintf(log_file, "  KV budget over allocations: %zu\n", kv_budget_over_allocs.load());
    fprintf(log_file, "  KV budget reject allocations: %zu\n", kv_budget_reject_allocs.load());
    fprintf(log_file, "  KV free success: %zu\n", kv_free_success_allocs.load());
    fprintf(log_file, "  Weight budget bytes: %zu\n", weight_budget_bytes);
    fprintf(log_file, "  Weight budget mode: %s\n", weight_budget_mode.c_str());
    fprintf(log_file, "  Weight budget CPU offload enable: %d\n",
            weight_budget_cpu_offload_enable);
    fprintf(log_file, "  Weight trace allocations: %zu\n", weight_trace_allocs.load());
    fprintf(log_file, "  Weight requested bytes: %zu\n", weight_requested_bytes.load());
    fprintf(log_file, "  Weight live bytes: %zu\n", weight_live_bytes.load());
    fprintf(log_file, "  Weight peak live bytes: %zu\n", weight_peak_live_bytes.load());
    fprintf(log_file, "  Weight budget remaining: %zu\n",
            weight_budget_remaining_snapshot(weight_live_bytes.load()));
    fprintf(log_file, "  Weight budget over allocations: %zu\n", weight_budget_over_allocs.load());
    fprintf(log_file, "  Weight budget reject allocations: %zu\n", weight_budget_reject_allocs.load());
    fprintf(log_file, "  Weight budget CPU offload allocations: %zu\n",
            weight_budget_cpu_offload_allocs.load());
    fprintf(log_file, "  Weight budget CPU offload bytes: %zu\n",
            weight_budget_cpu_offload_bytes.load());
    fprintf(log_file, "  Weight budget CPU offload failures: %zu\n",
            weight_budget_cpu_offload_failures.load());
    fprintf(log_file, "  Weight free success: %zu\n", weight_free_success_allocs.load());
    fprintf(log_file, "  Pool registry enabled: %d\n", pool_registry_enabled);
    fprintf(log_file, "  Driver scratch budget enforce: %d\n",
            driver_scratch_budget_enforce);
    fprintf(log_file, "  Driver scratch budget enforce warmup: %d\n",
            driver_scratch_budget_enforce_warmup);
    fprintf(log_file, "  Driver scratch budget bytes: %zu\n",
            driver_scratch_budget_bytes);
    fprintf(log_file, "  Driver scratch live bytes: %zu\n",
            driver_scratch_live_bytes.load());
    fprintf(log_file, "  Driver scratch peak live bytes: %zu\n",
            driver_scratch_peak_live_bytes.load());
    fprintf(log_file, "  Driver scratch budget over allocations: %zu\n",
            driver_scratch_budget_over_allocs.load());
    fprintf(log_file, "  Driver scratch budget reject allocations: %zu\n",
            driver_scratch_budget_reject_allocs.load());
    fprintf(log_file, "  Driver scratch budget release allocations: %zu\n",
            driver_scratch_budget_release_allocs.load());
    fprintf(log_file, "  Pool registry tracked allocations: %zu\n",
            pool_registry_tracked_allocs.load());
    fprintf(log_file, "  Pool registry free success: %zu\n",
            pool_registry_free_success_allocs.load());
    fprintf(log_file, "  Pool registry live objects: %zu\n",
            pool_registry_live_objects.load());
    fprintf(log_file, "  Pool registry peak live objects: %zu\n",
            pool_registry_peak_live_objects.load());
    fprintf(log_file, "  Pool kv allocations: %zu\n", pool_kv_allocs.load());
    fprintf(log_file, "  Pool kv requested bytes: %zu\n",
            pool_kv_requested_bytes.load());
    fprintf(log_file, "  Pool kv live bytes: %zu\n", pool_kv_live_bytes.load());
    fprintf(log_file, "  Pool kv peak live bytes: %zu\n",
            pool_kv_peak_live_bytes.load());
    fprintf(log_file, "  Pool kv free success: %zu\n",
            pool_kv_free_success_allocs.load());
    fprintf(log_file, "  Pool weight allocations: %zu\n",
            pool_weight_allocs.load());
    fprintf(log_file, "  Pool weight requested bytes: %zu\n",
            pool_weight_requested_bytes.load());
    fprintf(log_file, "  Pool weight live bytes: %zu\n",
            pool_weight_live_bytes.load());
    fprintf(log_file, "  Pool weight peak live bytes: %zu\n",
            pool_weight_peak_live_bytes.load());
    fprintf(log_file, "  Pool weight free success: %zu\n",
            pool_weight_free_success_allocs.load());
    fprintf(log_file, "  Pool runtime scratch allocations: %zu\n",
            pool_runtime_scratch_allocs.load());
    fprintf(log_file, "  Pool runtime scratch requested bytes: %zu\n",
            pool_runtime_scratch_requested_bytes.load());
    fprintf(log_file, "  Pool runtime scratch live bytes: %zu\n",
            pool_runtime_scratch_live_bytes.load());
    fprintf(log_file, "  Pool runtime scratch peak live bytes: %zu\n",
            pool_runtime_scratch_peak_live_bytes.load());
    fprintf(log_file, "  Pool runtime scratch free success: %zu\n",
            pool_runtime_scratch_free_success_allocs.load());
    fprintf(log_file, "  Scratch pool enabled: %d\n", scratch_pool_enable);
    fprintf(log_file, "  Scratch pool budget bytes: %zu\n",
            scratch_pool_budget_bytes);
    fprintf(log_file, "  Scratch pool mode: %s\n",
            scratch_pool_mode.c_str());
    fprintf(log_file, "  Scratch pool target phases: %s\n",
            scratch_pool_target_phases.c_str());
    fprintf(log_file, "  Scratch pool reuse cache enabled: %d\n",
            scratch_pool_reuse_cache_enable);
    fprintf(log_file, "  Scratch pool reuse cache max bytes: %zu\n",
            scratch_pool_reuse_cache_max_bytes);
    fprintf(log_file, "  Scratch pool reuse cache max entries per bucket: %zu\n",
            scratch_pool_reuse_cache_max_entries_per_bucket);
    fprintf(log_file, "  Scratch pool trace allocations: %zu\n",
            scratch_pool_trace_allocs.load());
    fprintf(log_file, "  Scratch pool eligible allocations: %zu\n",
            scratch_pool_eligible_allocs.load());
    fprintf(log_file, "  Scratch pool device-direct allocations: %zu\n",
            scratch_pool_device_direct_allocs.load());
    fprintf(log_file, "  Scratch pool device-direct bytes: %zu\n",
            scratch_pool_device_direct_bytes.load());
    fprintf(log_file, "  Scratch pool device-direct live bytes: %zu\n",
            scratch_pool_device_direct_live_bytes.load());
    fprintf(log_file, "  Scratch pool device-direct peak live bytes: %zu\n",
            scratch_pool_device_direct_peak_live_bytes.load());
    fprintf(log_file, "  Scratch pool budget remaining: %zu\n",
            scratch_pool_budget_remaining_snapshot(
                scratch_pool_device_direct_live_bytes.load()
            ));
    fprintf(log_file, "  Scratch pool budget over allocations: %zu\n",
            scratch_pool_budget_over_allocs.load());
    fprintf(log_file, "  Scratch pool budget reject allocations: %zu\n",
            scratch_pool_budget_reject_allocs.load());
    fprintf(log_file, "  Scratch pool device-direct free success: %zu\n",
            scratch_pool_device_direct_free_success_allocs.load());
    fprintf(log_file, "  Scratch pool reuse cache hits: %zu\n",
            scratch_pool_reuse_cache_hit_allocs.load());
    fprintf(log_file, "  Scratch pool reuse cache misses: %zu\n",
            scratch_pool_reuse_cache_miss_allocs.load());
    fprintf(log_file, "  Scratch pool reuse cache fresh backend allocations: %zu\n",
            scratch_pool_reuse_cache_fresh_backend_allocs.load());
    fprintf(log_file, "  Scratch pool reuse cache inserts: %zu\n",
            scratch_pool_reuse_cache_insert_allocs.load());
    fprintf(log_file, "  Scratch pool reuse cache insert rejects: %zu\n",
            scratch_pool_reuse_cache_insert_reject_allocs.load());
    fprintf(log_file, "  Scratch pool reuse cache hit bytes: %zu\n",
            scratch_pool_reuse_cache_hit_bytes.load());
    fprintf(log_file, "  Scratch pool reuse cache insert bytes: %zu\n",
            scratch_pool_reuse_cache_insert_bytes.load());
    fprintf(log_file, "  Scratch pool reuse cache cached bytes: %zu\n",
            scratch_pool_reuse_cache_cached_bytes.load());
    fprintf(log_file, "  Scratch pool reuse cache peak cached bytes: %zu\n",
            scratch_pool_reuse_cache_peak_cached_bytes.load());
    fprintf(log_file, "  Scratch pool reuse cache cached entries: %zu\n",
            scratch_pool_reuse_cache_cached_entries.load());
    fprintf(log_file, "  Scratch pool reuse cache event not ready: %zu\n",
            scratch_pool_reuse_cache_event_not_ready.load());
    fprintf(log_file, "  Scratch pool reuse cache event errors: %zu\n",
            scratch_pool_reuse_cache_event_errors.load());
    fprintf(log_file, "========================================\n\n");

    fflush(log_file);
    fclose(log_file);
    log_file = nullptr;
    log_initialized = false;
}

/**
 * Set custom log file path (must be called before first allocation)
 */
void uvm_set_log_file(const char* path) {
    if (log_initialized) {
        fprintf(stderr, "[vLLM UVM] Warning: Cannot change log file after initialization\n");
        return;
    }
    // The log path is read from VLLM_UVM_LOG_FILE environment variable in init_log_file()
    // This function is provided for programmatic control if needed
    if (path) {
        setenv("VLLM_UVM_LOG_FILE", path, 1);
    }
}

/**
 * Prefetch memory region to a specific device
 *
 * @param ptr Pointer to memory region
 * @param size Size of region in bytes
 * @param device Target device (-1 for CPU, >= 0 for GPU)
 * @param stream CUDA stream for async prefetch
 */
void uvm_prefetch(void* ptr, size_t size, int device, cudaStream_t stream) {
    if (ptr != NULL && size > 0) {
        cudaMemPrefetchAsync(ptr, size, device, stream);
    }
}

/**
 * Batch prefetch memory regions to a specific device.
 *
 * Returns the number of ranges whose CUDA prefetch call was accepted. The
 * caller still owns stream synchronization and end-to-end migration checks.
 */
int uvm_batch_prefetch(const uintptr_t* ptrs,
                       const size_t* sizes,
                       size_t count,
                       int device,
                       cudaStream_t stream) {
    if (ptrs == NULL || sizes == NULL || count == 0) return 0;

    int successes = 0;
    for (size_t i = 0; i < count; ++i) {
        if (ptrs[i] == 0 || sizes[i] == 0) continue;
        cudaError_t err = cudaMemPrefetchAsync(
            reinterpret_cast<void*>(ptrs[i]), sizes[i], device, stream);
        if (err == cudaSuccess) {
            ++successes;
        } else {
            fprintf(stderr, "[vLLM UVM] cudaMemPrefetchAsync batch[%zu] failed: %s\n",
                    i, cudaGetErrorString(err));
        }
    }
    return successes;
}

/**
 * Set memory advice for a region
 *
 * @param ptr Pointer to memory region
 * @param size Size of region in bytes
 * @param advice One of: 1=ReadMostly, 2=PreferredLocation, 3=AccessedBy
 * @param device Device ID for the advice
 */
void uvm_advise(void* ptr, size_t size, int advice, int device) {
    if (ptr == NULL || size == 0) return;

    cudaMemoryAdvise cuda_advice;
    switch (advice) {
        case 1:
            cuda_advice = cudaMemAdviseSetReadMostly;
            break;
        case 2:
            cuda_advice = cudaMemAdviseSetPreferredLocation;
            break;
        case 3:
            cuda_advice = cudaMemAdviseSetAccessedBy;
            break;
        default:
            fprintf(stderr, "[vLLM UVM] Unknown advice type: %d\n", advice);
            return;
    }

    cudaError_t err = cudaMemAdvise(ptr, size, cuda_advice, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "[vLLM UVM] cudaMemAdvise failed: %s\n",
                cudaGetErrorString(err));
    }
}

/**
 * Batch set memory advice for multiple regions.
 *
 * Returns the number of ranges whose CUDA advise call succeeded.
 */
int uvm_batch_advise(const uintptr_t* ptrs,
                     const size_t* sizes,
                     size_t count,
                     int advice,
                     int device) {
    if (ptrs == NULL || sizes == NULL || count == 0) return 0;

    cudaMemoryAdvise cuda_advice;
    switch (advice) {
        case 1:
            cuda_advice = cudaMemAdviseSetReadMostly;
            break;
        case 2:
            cuda_advice = cudaMemAdviseSetPreferredLocation;
            break;
        case 3:
            cuda_advice = cudaMemAdviseSetAccessedBy;
            break;
        default:
            fprintf(stderr, "[vLLM UVM] Unknown batch advice type: %d\n", advice);
            return 0;
    }

    int successes = 0;
    for (size_t i = 0; i < count; ++i) {
        if (ptrs[i] == 0 || sizes[i] == 0) continue;
        cudaError_t err = cudaMemAdvise(
            reinterpret_cast<void*>(ptrs[i]), sizes[i], cuda_advice, device);
        if (err == cudaSuccess) {
            ++successes;
        } else {
            fprintf(stderr, "[vLLM UVM] cudaMemAdvise batch[%zu] failed: %s\n",
                    i, cudaGetErrorString(err));
        }
    }
    return successes;
}

int uvm_driver_batch_prefetch_cpu(const uintptr_t* ptrs,
                                  const size_t* sizes,
                                  size_t count,
                                  int async) {
    return uvm_driver_batch_execute_internal(
        ptrs,
        sizes,
        count,
        UVM_TEST_PHASE_F_ACTION_PREFETCH_CPU,
        -1,
        async,
        nullptr);
}

int uvm_driver_phase_f_ping() {
    return uvm_driver_phase_f_ping_internal();
}

int uvm_driver_phase_f_small_ping() {
    return uvm_driver_phase_f_small_ping_internal();
}

int uvm_driver_phase_f_batch_mini0() {
    return uvm_driver_phase_f_batch_mini_internal(0, 0, 0);
}

int uvm_driver_phase_f_batch_mini1(uintptr_t base, size_t length) {
    return uvm_driver_phase_f_batch_mini_internal(1, base, length);
}

int uvm_driver_phase_f_open_only() {
    return uvm_driver_phase_f_open_only_internal();
}

int uvm_driver_phase_f_init_only() {
    return uvm_driver_phase_f_init_only_internal();
}

int uvm_driver_phase_f_ioctl_only() {
    return uvm_driver_phase_f_ioctl_only_internal();
}

int uvm_driver_phase_f_register_region(uint64_t region_id,
                                       uintptr_t base,
                                       size_t length) {
    return uvm_driver_phase_f_register_region_internal(region_id, base, length);
}

int uvm_driver_phase_f_unregister_region(uint64_t region_id) {
    return uvm_driver_phase_f_unregister_region_internal(region_id);
}

int uvm_driver_batch_prefetch_gpu(const uintptr_t* ptrs,
                                  const size_t* sizes,
                                  size_t count,
                                  int device,
                                  int async) {
    return uvm_driver_batch_execute_internal(
        ptrs,
        sizes,
        count,
        UVM_TEST_PHASE_F_ACTION_PREFETCH_GPU,
        device,
        async,
        nullptr);
}

int uvm_driver_batch_advise_cpu(const uintptr_t* ptrs,
                                const size_t* sizes,
                                size_t count) {
    return uvm_driver_batch_execute_internal(
        ptrs,
        sizes,
        count,
        UVM_TEST_PHASE_F_ACTION_ADVISE_CPU,
        -1,
        0,
        nullptr);
}

int uvm_driver_batch_query_residency(const uintptr_t* ptrs,
                                     const size_t* sizes,
                                     size_t count,
                                     int device) {
    return uvm_driver_batch_execute_internal(
        ptrs,
        sizes,
        count,
        UVM_TEST_PHASE_F_ACTION_QUERY_RESIDENCY,
        device,
        0,
        nullptr);
}

int uvm_driver_batch_prepare_remote_read_bg_deferred(const uintptr_t* ptrs,
                                                     const size_t* sizes,
                                                     size_t count,
                                                     int device,
                                                     int async) {
    return uvm_driver_batch_execute_internal(
        ptrs,
        sizes,
        count,
        UVM_TEST_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG_DEFERRED,
        device,
        async,
        nullptr);
}

int uvm_driver_batch_prepare_remote_read(const uintptr_t* ptrs,
                                         const size_t* sizes,
                                         size_t count,
                                         int device,
                                         int async) {
    return uvm_driver_batch_execute_internal(
        ptrs,
        sizes,
        count,
        UVM_TEST_PHASE_F_ACTION_PREPARE_REMOTE_READ,
        device,
        async,
        nullptr);
}

int uvm_driver_batch_start_bg_promotion(const uintptr_t* ptrs,
                                        const size_t* sizes,
                                        size_t count,
                                        int device,
                                        int async) {
    return uvm_driver_batch_execute_internal(
        ptrs,
        sizes,
        count,
        UVM_TEST_PHASE_F_ACTION_START_BG_PROMOTION,
        device,
        async,
        nullptr);
}





int uvm_driver_batch_pause_bg_promotion(const uintptr_t* ptrs,
                                        const size_t* sizes,
                                        size_t count,
                                        int device,
                                        int async) {
    return uvm_driver_batch_execute_internal(
        ptrs,
        sizes,
        count,
        UVM_TEST_PHASE_F_ACTION_PAUSE_BG_PROMOTION,
        device,
        async,
        nullptr);
}

int uvm_driver_batch_resume_bg_promotion(const uintptr_t* ptrs,
                                         const size_t* sizes,
                                         size_t count,
                                         int device,
                                         int async) {
    return uvm_driver_batch_execute_internal(
        ptrs,
        sizes,
        count,
        UVM_TEST_PHASE_F_ACTION_RESUME_BG_PROMOTION,
        device,
        async,
        nullptr);
}

static int uvm_driver_batch_query_residency_bytes_internal(
    const uintptr_t* ptrs,
    const size_t* sizes,
    size_t count,
    int device,
    int async,
    uint64_t* gpu_bytes_out,
    uint64_t* cpu_bytes_out,
    uint64_t* any_bytes_out,
    uint64_t* unknown_bytes_out,
    uint64_t* queried_bytes_out) {
    uint64_t gpu_bytes = 0;
    uint64_t cpu_bytes = 0;
    uint64_t any_bytes = 0;
    uint64_t unknown_bytes = 0;
    uint64_t queried_bytes = 0;

    if (gpu_bytes_out != nullptr) *gpu_bytes_out = 0;
    if (cpu_bytes_out != nullptr) *cpu_bytes_out = 0;
    if (any_bytes_out != nullptr) *any_bytes_out = 0;
    if (unknown_bytes_out != nullptr) *unknown_bytes_out = 0;
    if (queried_bytes_out != nullptr) *queried_bytes_out = 0;

    if (count > UVM_TEST_PHASE_F_MAX_RANGES)
        return -2;

    std::vector<PhaseFRangeResult> results(count);
    int ret = uvm_driver_batch_execute_internal(
        ptrs,
        sizes,
        count,
        UVM_TEST_PHASE_F_ACTION_QUERY_RESIDENCY,
        device,
        async,
        nullptr,
        results.empty() ? nullptr : results.data());

    if (ret < 0)
        return ret;

    for (size_t i = 0; i < count; ++i) {
        if (results[i].status != 0)
            continue;

        gpu_bytes += results[i].resident_gpu_bytes;
        cpu_bytes += results[i].resident_cpu_bytes;
        any_bytes += results[i].resident_any_bytes;
        unknown_bytes += results[i].resident_unknown_bytes;
        queried_bytes += results[i].queried_bytes;
    }

    if (gpu_bytes_out != nullptr) *gpu_bytes_out = gpu_bytes;
    if (cpu_bytes_out != nullptr) *cpu_bytes_out = cpu_bytes;
    if (any_bytes_out != nullptr) *any_bytes_out = any_bytes;
    if (unknown_bytes_out != nullptr) *unknown_bytes_out = unknown_bytes;
    if (queried_bytes_out != nullptr) *queried_bytes_out = queried_bytes;
    return ret;
}

int uvm_driver_batch_query_residency_bytes(const uintptr_t* ptrs,
                                           const size_t* sizes,
                                           size_t count,
                                           int device,
                                           uint64_t* gpu_bytes_out,
                                           uint64_t* cpu_bytes_out,
                                           uint64_t* any_bytes_out,
                                           uint64_t* unknown_bytes_out,
                                           uint64_t* queried_bytes_out) {
    return uvm_driver_batch_query_residency_bytes_internal(
        ptrs,
        sizes,
        count,
        device,
        0,
        gpu_bytes_out,
        cpu_bytes_out,
        any_bytes_out,
        unknown_bytes_out,
        queried_bytes_out);
}

int uvm_driver_batch_query_residency_bytes_nowait(
    const uintptr_t* ptrs,
    const size_t* sizes,
    size_t count,
    int device,
    uint64_t* gpu_bytes_out,
    uint64_t* cpu_bytes_out,
    uint64_t* any_bytes_out,
    uint64_t* unknown_bytes_out,
    uint64_t* queried_bytes_out) {
    return uvm_driver_batch_query_residency_bytes_internal(
        ptrs,
        sizes,
        count,
        device,
        1,
        gpu_bytes_out,
        cpu_bytes_out,
        any_bytes_out,
        unknown_bytes_out,
        queried_bytes_out);
}

int uvm_is_managed_pointer(uintptr_t ptr, size_t size) {
    if (ptr == 0 || size == 0) return -1;

    cudaPointerAttributes attrs{};
    cudaError_t err = cudaPointerGetAttributes(
        &attrs,
        reinterpret_cast<const void*>(ptr));
    if (err != cudaSuccess) {
        phase_f_debug_log("managed_probe ptr=0x%llx size=%zu cuda_error=%d error=%s",
                          static_cast<unsigned long long>(ptr),
                          size,
                          static_cast<int>(err),
                          cudaGetErrorString(err));
        // Clear the sticky runtime error so this diagnostic probe does not
        // poison later CUDA calls in the serving process.
        (void)cudaGetLastError();
        return -static_cast<int>(err);
    }

    phase_f_debug_log("managed_probe ptr=0x%llx size=%zu type=%d device=%d device_ptr=%p host_ptr=%p",
                      static_cast<unsigned long long>(ptr),
                      size,
                      static_cast<int>(attrs.type),
                      attrs.device,
                      attrs.devicePointer,
                      attrs.hostPointer);

    return attrs.type == cudaMemoryTypeManaged ? 1 : 0;
}




int uvm_driver_update_region_state(uint64_t region_id,
                                   uint32_t state) {
    return uvm_driver_update_region_state_internal(
        region_id,
        state);
}

int uvm_driver_query_region(uint64_t region_id,
                            uint64_t* values_out,
                            size_t values_count) {
    return uvm_driver_query_region_internal(
        region_id,
        values_out,
        values_count);
}

int uvm_driver_query_region_by_va(uintptr_t address,
                                  uint64_t* values_out,
                                  size_t values_count) {
    return uvm_driver_query_region_by_va_internal(
        address,
        values_out,
        values_count);
}

int uvm_driver_phase_f_probe_existing_uvm_fd() {
    return uvm_driver_phase_f_probe_existing_uvm_fd_internal();
}

}  // extern "C"
