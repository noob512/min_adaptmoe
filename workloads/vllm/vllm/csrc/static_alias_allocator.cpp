#include "static_alias_allocator.h"

#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#include "cumem_allocator_compat.h"

#ifdef USE_ROCM

namespace {

PyObject* rocm_not_supported() {
  PyErr_SetString(
      PyExc_NotImplementedError,
      "CUDA VMM static aliases are currently implemented only for CUDA");
  return nullptr;
}

}  // namespace

PyObject* python_static_alias_create(PyObject*, PyObject*) {
  return rocm_not_supported();
}

PyObject* python_static_alias_info(PyObject*, PyObject*) {
  return rocm_not_supported();
}

PyObject* python_static_alias_dlpack(PyObject*, PyObject*) {
  return rocm_not_supported();
}

PyObject* python_static_alias_synchronize(PyObject*, PyObject*) {
  return rocm_not_supported();
}

PyObject* python_static_alias_destroy(PyObject*, PyObject*) {
  return rocm_not_supported();
}

#else

namespace {

// Minimal legacy DLPack declarations. Keeping these local avoids adding a
// build-time dependency on a particular DLPack header version. Their layout is
// the stable ABI consumed by torch.from_dlpack.
enum SADLDeviceType : int32_t {
  kSADLCUDA = 2,
};

struct SADLDevice {
  SADLDeviceType device_type;
  int32_t device_id;
};

struct SADLDataType {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
};

struct SADLTensor {
  void* data;
  SADLDevice device;
  int32_t ndim;
  SADLDataType dtype;
  int64_t* shape;
  int64_t* strides;
  uint64_t byte_offset;
};

struct SADLManagedTensor {
  SADLTensor dl_tensor;
  void* manager_ctx;
  void (*deleter)(SADLManagedTensor* self);
};

struct Mapping {
  CUdeviceptr address = 0;
  bool reserved = false;
  bool mapped = false;
};

struct StaticAliasAllocation {
  uint64_t token = 0;
  int device_ordinal = -1;
  CUdevice device = 0;
  CUcontext primary_context = nullptr;
  bool context_retained = false;

  size_t requested_size = 0;
  size_t aligned_size = 0;
  size_t granularity = 0;

  CUmemGenericAllocationHandle handle = 0;
  bool handle_created = false;
  Mapping upload;
  std::vector<Mapping> consumers;

  size_t live_tensors = 0;
  // Raw VAs are returned to Python, so a newly created allocation is treated
  // as potentially in use. Teardown must either explicitly synchronize or be
  // passed synchronize=true.
  bool synchronized_for_destroy = false;
  bool destroying = false;
};

struct Registry {
  std::mutex mutex;
  uint64_t next_token = 1;
  std::unordered_map<uint64_t, StaticAliasAllocation*> allocations;
};

// Deliberately leaked process-lifetime registry: CUDA/Python teardown order is
// not defined at interpreter exit. Explicit static_alias_destroy owns device
// cleanup; the CUDA context cleans leaked resources if a process aborts.
Registry& registry() {
  static Registry* instance = new Registry();
  return *instance;
}

struct ManagedTensorContext {
  SADLManagedTensor managed = {};
  uint64_t token = 0;
  int64_t* shape = nullptr;
  int64_t* strides = nullptr;
};

std::string cuda_error_message(CUresult result, const char* operation) {
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  std::string message(operation);
  message += " failed";
  if (name != nullptr) {
    message += " (";
    message += name;
    message += ")";
  }
  if (description != nullptr) {
    message += ": ";
    message += description;
  }
  return message;
}

bool cuda_ok(CUresult result, const char* operation, std::string* error) {
  if (result == CUDA_SUCCESS) {
    return true;
  }
  if (error != nullptr && error->empty()) {
    *error = cuda_error_message(result, operation);
  }
  return false;
}

void set_runtime_error(const std::string& message) {
  PyErr_SetString(PyExc_RuntimeError, message.c_str());
}

class CurrentContextGuard {
 public:
  explicit CurrentContextGuard(CUcontext target) : target_(target) {}

  bool Enter(std::string* error) {
    if (!cuda_ok(cuCtxGetCurrent(&previous_), "cuCtxGetCurrent", error)) {
      return false;
    }
    if (previous_ != target_) {
      if (!cuda_ok(cuCtxSetCurrent(target_), "cuCtxSetCurrent", error)) {
        return false;
      }
      changed_ = true;
    }
    entered_ = true;
    return true;
  }

  ~CurrentContextGuard() {
    if (entered_ && changed_) {
      // A restoration failure cannot be surfaced safely from a destructor.
      (void)cuCtxSetCurrent(previous_);
    }
  }

 private:
  CUcontext target_ = nullptr;
  CUcontext previous_ = nullptr;
  bool changed_ = false;
  bool entered_ = false;
};

bool retain_primary_context(int device_ordinal, StaticAliasAllocation* alloc,
                            std::string* error) {
  if (!cuda_ok(cuInit(0), "cuInit", error)) {
    return false;
  }
  if (!cuda_ok(cuDeviceGet(&alloc->device, device_ordinal), "cuDeviceGet",
               error)) {
    return false;
  }
  if (!cuda_ok(cuDevicePrimaryCtxRetain(&alloc->primary_context, alloc->device),
               "cuDevicePrimaryCtxRetain", error)) {
    return false;
  }
  alloc->context_retained = true;
  return true;
}

bool reserve_map_and_set_access(StaticAliasAllocation* alloc, Mapping* mapping,
                                CUmemAccess_flags access,
                                std::string* error) {
  if (!cuda_ok(cuMemAddressReserve(&mapping->address, alloc->aligned_size,
                                   alloc->granularity, 0, 0),
               "cuMemAddressReserve", error)) {
    return false;
  }
  mapping->reserved = true;

  if (!cuda_ok(cuMemMap(mapping->address, alloc->aligned_size, 0,
                        alloc->handle, 0),
               "cuMemMap", error)) {
    return false;
  }
  mapping->mapped = true;

  CUmemAccessDesc access_desc = {};
  access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access_desc.location.id = alloc->device_ordinal;
  access_desc.flags = access;
  return cuda_ok(cuMemSetAccess(mapping->address, alloc->aligned_size,
                                &access_desc, 1),
                 "cuMemSetAccess", error);
}

bool release_mapping(StaticAliasAllocation* alloc, Mapping* mapping,
                     std::string* error) {
  bool ok = true;
  if (mapping->mapped) {
    CUresult result = cuMemUnmap(mapping->address, alloc->aligned_size);
    if (cuda_ok(result, "cuMemUnmap", error)) {
      mapping->mapped = false;
    } else {
      ok = false;
    }
  }
  if (!mapping->mapped && mapping->reserved) {
    CUresult result =
        cuMemAddressFree(mapping->address, alloc->aligned_size);
    if (cuda_ok(result, "cuMemAddressFree", error)) {
      mapping->reserved = false;
      mapping->address = 0;
    } else {
      ok = false;
    }
  }
  return ok;
}

bool has_live_mapping(const StaticAliasAllocation& alloc) {
  if (alloc.upload.mapped || alloc.upload.reserved) {
    return true;
  }
  for (const Mapping& mapping : alloc.consumers) {
    if (mapping.mapped || mapping.reserved) {
      return true;
    }
  }
  return false;
}

bool cleanup_allocation(StaticAliasAllocation* alloc, std::string* error) {
  bool ok = true;
  for (Mapping& mapping : alloc->consumers) {
    if (!release_mapping(alloc, &mapping, error)) {
      ok = false;
    }
  }
  if (!release_mapping(alloc, &alloc->upload, error)) {
    ok = false;
  }

  if (!has_live_mapping(*alloc) && alloc->handle_created) {
    CUresult result = cuMemRelease(alloc->handle);
    if (cuda_ok(result, "cuMemRelease", error)) {
      alloc->handle_created = false;
      alloc->handle = 0;
    } else {
      ok = false;
    }
  }
  return ok && !has_live_mapping(*alloc) && !alloc->handle_created;
}

bool release_primary_context(StaticAliasAllocation* alloc,
                             std::string* error) {
  if (!alloc->context_retained) {
    return true;
  }
  if (!cuda_ok(cuDevicePrimaryCtxRelease(alloc->device),
               "cuDevicePrimaryCtxRelease", error)) {
    return false;
  }
  alloc->context_retained = false;
  alloc->primary_context = nullptr;
  return true;
}

bool checked_round_up(size_t size, size_t alignment, size_t* result) {
  if (alignment == 0 || size > std::numeric_limits<size_t>::max() -
                                   (alignment - 1)) {
    return false;
  }
  *result = ((size + alignment - 1) / alignment) * alignment;
  return *result >= size;
}

bool dict_set_unsigned(PyObject* dict, const char* key, uint64_t value) {
  PyObject* object = PyLong_FromUnsignedLongLong(value);
  if (object == nullptr) {
    return false;
  }
  int result = PyDict_SetItemString(dict, key, object);
  Py_DECREF(object);
  return result == 0;
}

bool dict_set_signed(PyObject* dict, const char* key, int64_t value) {
  PyObject* object = PyLong_FromLongLong(value);
  if (object == nullptr) {
    return false;
  }
  int result = PyDict_SetItemString(dict, key, object);
  Py_DECREF(object);
  return result == 0;
}

bool dict_set_bool(PyObject* dict, const char* key, bool value) {
  PyObject* object = value ? Py_True : Py_False;
  return PyDict_SetItemString(dict, key, object) == 0;
}

bool dict_set_string(PyObject* dict, const char* key, const char* value) {
  PyObject* object = PyUnicode_FromString(value);
  if (object == nullptr) {
    return false;
  }
  int result = PyDict_SetItemString(dict, key, object);
  Py_DECREF(object);
  return result == 0;
}

PyObject* allocation_info_locked(const StaticAliasAllocation& alloc) {
  PyObject* dict = PyDict_New();
  if (dict == nullptr) {
    return nullptr;
  }

  PyObject* consumer_vas = PyTuple_New(alloc.consumers.size());
  if (consumer_vas == nullptr) {
    Py_DECREF(dict);
    return nullptr;
  }
  for (size_t i = 0; i < alloc.consumers.size(); ++i) {
    PyObject* address = PyLong_FromUnsignedLongLong(
        static_cast<uint64_t>(alloc.consumers[i].address));
    if (address == nullptr) {
      Py_DECREF(consumer_vas);
      Py_DECREF(dict);
      return nullptr;
    }
    PyTuple_SetItem(consumer_vas, static_cast<Py_ssize_t>(i), address);
  }

  bool ok =
      dict_set_unsigned(dict, "token", alloc.token) &&
      dict_set_signed(dict, "device", alloc.device_ordinal) &&
      dict_set_unsigned(dict, "requested_size", alloc.requested_size) &&
      dict_set_unsigned(dict, "aligned_size", alloc.aligned_size) &&
      dict_set_unsigned(dict, "granularity", alloc.granularity) &&
      dict_set_unsigned(dict, "backing_bytes", alloc.aligned_size) &&
      dict_set_unsigned(dict, "upload_va", alloc.upload.address) &&
      PyDict_SetItemString(dict, "consumer_vas", consumer_vas) == 0 &&
      dict_set_string(dict, "upload_access", "rw") &&
      dict_set_string(dict, "consumer_access", "ro") &&
      dict_set_unsigned(dict, "live_tensors", alloc.live_tensors) &&
      dict_set_bool(dict, "synchronized_for_destroy",
                    alloc.synchronized_for_destroy) &&
      dict_set_bool(dict, "destroying", alloc.destroying);
  Py_DECREF(consumer_vas);
  if (!ok) {
    Py_DECREF(dict);
    return nullptr;
  }
  return dict;
}

StaticAliasAllocation* lookup_locked(uint64_t token) {
  Registry& state = registry();
  auto iterator = state.allocations.find(token);
  return iterator == state.allocations.end() ? nullptr : iterator->second;
}

bool parse_int64_tuple(PyObject* object, const char* name,
                       std::vector<int64_t>* values) {
  if (!PyTuple_Check(object)) {
    std::string message(name);
    message += " must be a tuple";
    PyErr_SetString(PyExc_TypeError, message.c_str());
    return false;
  }
  Py_ssize_t count = PyTuple_Size(object);
  if (count < 0 || count > std::numeric_limits<int32_t>::max()) {
    PyErr_SetString(PyExc_ValueError, "invalid tensor rank");
    return false;
  }
  values->reserve(static_cast<size_t>(count));
  for (Py_ssize_t i = 0; i < count; ++i) {
    PyObject* item = PyTuple_GetItem(object, i);
    if (item == nullptr) {
      return false;
    }
    long long value = PyLong_AsLongLong(item);
    if (value == -1 && PyErr_Occurred()) {
      std::string message(name);
      message += " entries must be integers";
      PyErr_SetString(PyExc_TypeError, message.c_str());
      return false;
    }
    values->push_back(static_cast<int64_t>(value));
  }
  return true;
}

bool checked_add_u64(uint64_t left, uint64_t right, uint64_t* result) {
  if (left > std::numeric_limits<uint64_t>::max() - right) {
    return false;
  }
  *result = left + right;
  return true;
}

bool checked_mul_u64(uint64_t left, uint64_t right, uint64_t* result) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

bool validate_tensor_view(const StaticAliasAllocation& alloc,
                          uint64_t byte_offset,
                          const std::vector<int64_t>& shape,
                          const std::vector<int64_t>* strides, int dtype_code,
                          int dtype_bits, int dtype_lanes,
                          uint64_t* byte_span) {
  // DLPack 1.2 extends the original [0, 6] scalar codes with the FP8
  // encodings [7, 14]. PyTorch 2.9 uses these codes for FP8 MoE weights.
  if (dtype_code < 0 || dtype_code > 14) {
    PyErr_SetString(PyExc_ValueError,
                    "dtype_code must be a standard DLPack code in [0, 14]");
    return false;
  }
  if (dtype_bits <= 0 || dtype_bits > 255 || dtype_lanes != 1 ||
      dtype_bits % 8 != 0) {
    PyErr_SetString(
        PyExc_ValueError,
        "dtype must have 1 lane and a positive byte-aligned bit width <= 255");
    return false;
  }
  uint64_t element_size = static_cast<uint64_t>(dtype_bits / 8);
  if (byte_offset % element_size != 0) {
    PyErr_SetString(PyExc_ValueError,
                    "byte_offset must be aligned to the dtype element size");
    return false;
  }
  if (strides != nullptr && strides->size() != shape.size()) {
    PyErr_SetString(PyExc_ValueError,
                    "strides must have the same length as shape");
    return false;
  }

  bool empty = false;
  uint64_t max_element_offset = 0;
  uint64_t contiguous_stride = 1;
  for (size_t reverse_i = 0; reverse_i < shape.size(); ++reverse_i) {
    size_t i = shape.size() - reverse_i - 1;
    if (shape[i] < 0) {
      PyErr_SetString(PyExc_ValueError, "shape entries must be non-negative");
      return false;
    }
    if (shape[i] == 0) {
      empty = true;
    }
    uint64_t stride = contiguous_stride;
    if (strides != nullptr) {
      if ((*strides)[i] < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "negative strides are not supported");
        return false;
      }
      stride = static_cast<uint64_t>((*strides)[i]);
    }
    if (!empty && shape[i] > 0) {
      uint64_t contribution = 0;
      if (!checked_mul_u64(static_cast<uint64_t>(shape[i] - 1), stride,
                           &contribution) ||
          !checked_add_u64(max_element_offset, contribution,
                           &max_element_offset)) {
        PyErr_SetString(PyExc_OverflowError, "tensor view span overflow");
        return false;
      }
    }
    if (strides == nullptr) {
      if (!checked_mul_u64(contiguous_stride,
                           static_cast<uint64_t>(shape[i]),
                           &contiguous_stride)) {
        PyErr_SetString(PyExc_OverflowError, "tensor shape overflow");
        return false;
      }
    }
  }

  if (empty) {
    *byte_span = 0;
  } else {
    uint64_t elements = 0;
    if (!checked_add_u64(max_element_offset, 1, &elements) ||
        !checked_mul_u64(elements, element_size, byte_span)) {
      PyErr_SetString(PyExc_OverflowError, "tensor byte span overflow");
      return false;
    }
  }

  uint64_t end = 0;
  if (!checked_add_u64(byte_offset, *byte_span, &end) ||
      end > alloc.requested_size) {
    PyErr_SetString(
        PyExc_ValueError,
        "tensor view exceeds requested_size (VMM alignment padding is hidden)");
    return false;
  }
  return true;
}

void managed_tensor_deleter(SADLManagedTensor* managed) {
  if (managed == nullptr || managed->manager_ctx == nullptr) {
    return;
  }
  ManagedTensorContext* context =
      static_cast<ManagedTensorContext*>(managed->manager_ctx);
  {
    Registry& state = registry();
    std::lock_guard<std::mutex> lock(state.mutex);
    StaticAliasAllocation* alloc = lookup_locked(context->token);
    if (alloc != nullptr && alloc->live_tensors > 0) {
      --alloc->live_tensors;
    }
  }
  delete[] context->shape;
  delete[] context->strides;
  delete context;
}

void dlpack_capsule_destructor(PyObject* capsule) {
  if (!PyCapsule_IsValid(capsule, "dltensor")) {
    return;
  }
  void* pointer = PyCapsule_GetPointer(capsule, "dltensor");
  if (pointer == nullptr) {
    PyErr_Clear();
    return;
  }
  SADLManagedTensor* managed = static_cast<SADLManagedTensor*>(pointer);
  if (managed->deleter != nullptr) {
    managed->deleter(managed);
  }
}

bool initialize_allocation(StaticAliasAllocation* alloc, size_t consumer_count,
                           std::string* error) {
  if (!retain_primary_context(alloc->device_ordinal, alloc, error)) {
    return false;
  }
  CurrentContextGuard context_guard(alloc->primary_context);
  if (!context_guard.Enter(error)) {
    return false;
  }

  CUmemAllocationProp property = {};
  property.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  property.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  property.location.id = alloc->device_ordinal;
  property.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_NONE;

  if (!cuda_ok(cuMemGetAllocationGranularity(
                   &alloc->granularity, &property,
                   CU_MEM_ALLOC_GRANULARITY_MINIMUM),
               "cuMemGetAllocationGranularity", error)) {
    return false;
  }
  if (!checked_round_up(alloc->requested_size, alloc->granularity,
                        &alloc->aligned_size)) {
    *error = "requested size cannot be aligned without overflow";
    return false;
  }
  if (!cuda_ok(cuMemCreate(&alloc->handle, alloc->aligned_size, &property, 0),
               "cuMemCreate", error)) {
    return false;
  }
  alloc->handle_created = true;

  if (!reserve_map_and_set_access(alloc, &alloc->upload,
                                  CU_MEM_ACCESS_FLAGS_PROT_READWRITE, error)) {
    return false;
  }
  alloc->consumers.resize(consumer_count);
  for (Mapping& consumer : alloc->consumers) {
    if (!reserve_map_and_set_access(alloc, &consumer,
                                    CU_MEM_ACCESS_FLAGS_PROT_READ, error)) {
      return false;
    }
  }
  return true;
}

bool synchronize_locked(StaticAliasAllocation* alloc, std::string* error) {
  if (alloc->live_tensors != 0) {
    *error = "cannot synchronize for teardown while DLPack tensors are live";
    return false;
  }
  CurrentContextGuard context_guard(alloc->primary_context);
  if (!context_guard.Enter(error)) {
    return false;
  }
  if (!cuda_ok(cuCtxSynchronize(), "cuCtxSynchronize", error)) {
    return false;
  }
  alloc->synchronized_for_destroy = true;
  return true;
}

}  // namespace

PyObject* python_static_alias_create(PyObject*, PyObject* args) {
  int device_ordinal = -1;
  unsigned long long requested_size_value = 0;
  Py_ssize_t consumer_count_value = 0;
  if (!PyArg_ParseTuple(args, "iKn", &device_ordinal, &requested_size_value,
                        &consumer_count_value)) {
    return nullptr;
  }
  if (device_ordinal < 0) {
    PyErr_SetString(PyExc_ValueError, "device must be non-negative");
    return nullptr;
  }
  if (requested_size_value == 0 ||
      requested_size_value > std::numeric_limits<size_t>::max()) {
    PyErr_SetString(PyExc_ValueError,
                    "size must be in the host size_t range and non-zero");
    return nullptr;
  }
  if (consumer_count_value <= 0 || consumer_count_value > 4096) {
    PyErr_SetString(PyExc_ValueError,
                    "consumer_count must be between 1 and 4096");
    return nullptr;
  }

  StaticAliasAllocation* alloc =
      new (std::nothrow) StaticAliasAllocation();
  if (alloc == nullptr) {
    return PyErr_NoMemory();
  }
  alloc->device_ordinal = device_ordinal;
  alloc->requested_size = static_cast<size_t>(requested_size_value);

  std::string error;
  if (!initialize_allocation(alloc,
                             static_cast<size_t>(consumer_count_value),
                             &error)) {
    if (alloc->context_retained) {
      std::string cleanup_error;
      {
        CurrentContextGuard context_guard(alloc->primary_context);
        if (context_guard.Enter(&cleanup_error)) {
          (void)cleanup_allocation(alloc, &cleanup_error);
        }
      }
      (void)release_primary_context(alloc, &cleanup_error);
      if (!cleanup_error.empty()) {
        error += "; cleanup: ";
        error += cleanup_error;
      }
    }
    delete alloc;
    set_runtime_error(error);
    return nullptr;
  }

  Registry& state = registry();
  PyObject* result = nullptr;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    uint64_t token = state.next_token++;
    if (token == 0) {
      token = state.next_token++;
    }
    alloc->token = token;
    try {
      state.allocations.emplace(token, alloc);
    } catch (const std::bad_alloc&) {
      PyErr_NoMemory();
    }
    if (!PyErr_Occurred()) {
      result = allocation_info_locked(*alloc);
    }
  }

  if (result == nullptr) {
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.allocations.erase(alloc->token);
    }
    std::string cleanup_error;
    {
      CurrentContextGuard context_guard(alloc->primary_context);
      if (context_guard.Enter(&cleanup_error)) {
        (void)cleanup_allocation(alloc, &cleanup_error);
      }
    }
    (void)release_primary_context(alloc, &cleanup_error);
    delete alloc;
    return nullptr;
  }
  return result;
}

PyObject* python_static_alias_info(PyObject*, PyObject* args) {
  unsigned long long token_value = 0;
  if (!PyArg_ParseTuple(args, "K", &token_value)) {
    return nullptr;
  }
  Registry& state = registry();
  std::lock_guard<std::mutex> lock(state.mutex);
  StaticAliasAllocation* alloc = lookup_locked(token_value);
  if (alloc == nullptr) {
    PyErr_SetString(PyExc_KeyError, "unknown static-alias token");
    return nullptr;
  }
  return allocation_info_locked(*alloc);
}

PyObject* python_static_alias_dlpack(PyObject*, PyObject* args) {
  unsigned long long token_value = 0;
  int role = -1;
  Py_ssize_t consumer_index = 0;
  unsigned long long byte_offset = 0;
  PyObject* shape_object = nullptr;
  PyObject* strides_object = nullptr;
  int dtype_code = 0;
  int dtype_bits = 0;
  int dtype_lanes = 0;
  if (!PyArg_ParseTuple(args, "KinKOOiii", &token_value, &role,
                        &consumer_index, &byte_offset, &shape_object,
                        &strides_object, &dtype_code, &dtype_bits,
                        &dtype_lanes)) {
    return nullptr;
  }
  if (role != 0 && role != 1) {
    PyErr_SetString(PyExc_ValueError,
                    "role must be 0 (upload) or 1 (consumer)");
    return nullptr;
  }

  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  if (!parse_int64_tuple(shape_object, "shape", &shape)) {
    return nullptr;
  }
  const std::vector<int64_t>* strides_pointer = nullptr;
  if (strides_object != Py_None) {
    if (!parse_int64_tuple(strides_object, "strides", &strides)) {
      return nullptr;
    }
    strides_pointer = &strides;
  }

  Registry& state = registry();
  std::lock_guard<std::mutex> lock(state.mutex);
  StaticAliasAllocation* alloc = lookup_locked(token_value);
  if (alloc == nullptr) {
    PyErr_SetString(PyExc_KeyError, "unknown static-alias token");
    return nullptr;
  }
  if (alloc->destroying) {
    PyErr_SetString(PyExc_RuntimeError,
                    "static-alias allocation is being destroyed");
    return nullptr;
  }
  CUdeviceptr base_address = alloc->upload.address;
  if (role == 1) {
    if (consumer_index < 0 ||
        static_cast<size_t>(consumer_index) >= alloc->consumers.size()) {
      PyErr_SetString(PyExc_IndexError, "consumer_index out of range");
      return nullptr;
    }
    base_address = alloc->consumers[consumer_index].address;
  }
  uint64_t byte_span = 0;
  if (!validate_tensor_view(*alloc, byte_offset, shape, strides_pointer,
                            dtype_code, dtype_bits, dtype_lanes, &byte_span)) {
    return nullptr;
  }

  ManagedTensorContext* context =
      new (std::nothrow) ManagedTensorContext();
  if (context == nullptr) {
    return PyErr_NoMemory();
  }
  context->token = token_value;
  if (!shape.empty()) {
    context->shape = new (std::nothrow) int64_t[shape.size()];
    if (context->shape == nullptr) {
      delete context;
      return PyErr_NoMemory();
    }
    for (size_t i = 0; i < shape.size(); ++i) {
      context->shape[i] = shape[i];
    }
  }
  if (strides_pointer != nullptr && !strides.empty()) {
    context->strides = new (std::nothrow) int64_t[strides.size()];
    if (context->strides == nullptr) {
      delete[] context->shape;
      delete context;
      return PyErr_NoMemory();
    }
    for (size_t i = 0; i < strides.size(); ++i) {
      context->strides[i] = strides[i];
    }
  }

  // PyTorch's legacy capsule importer does not consistently apply the DLPack
  // byte_offset field. Fold it into data so torch.Tensor.data_ptr() is the
  // requested alias address on every supported PyTorch version.
  context->managed.dl_tensor.data = reinterpret_cast<void*>(
      static_cast<uintptr_t>(base_address) +
      static_cast<uintptr_t>(byte_offset));
  context->managed.dl_tensor.device = {
      kSADLCUDA, static_cast<int32_t>(alloc->device_ordinal)};
  context->managed.dl_tensor.ndim = static_cast<int32_t>(shape.size());
  context->managed.dl_tensor.dtype = {
      static_cast<uint8_t>(dtype_code), static_cast<uint8_t>(dtype_bits),
      static_cast<uint16_t>(dtype_lanes)};
  context->managed.dl_tensor.shape = context->shape;
  context->managed.dl_tensor.strides = context->strides;
  context->managed.dl_tensor.byte_offset = 0;
  context->managed.manager_ctx = context;
  context->managed.deleter = managed_tensor_deleter;

  ++alloc->live_tensors;
  alloc->synchronized_for_destroy = false;
  PyObject* capsule = PyCapsule_New(&context->managed, "dltensor",
                                    dlpack_capsule_destructor);
  if (capsule == nullptr) {
    --alloc->live_tensors;
    delete[] context->shape;
    delete[] context->strides;
    delete context;
    return nullptr;
  }
  return capsule;
}

PyObject* python_static_alias_synchronize(PyObject*, PyObject* args) {
  unsigned long long token_value = 0;
  if (!PyArg_ParseTuple(args, "K", &token_value)) {
    return nullptr;
  }
  Registry& state = registry();
  std::lock_guard<std::mutex> lock(state.mutex);
  StaticAliasAllocation* alloc = lookup_locked(token_value);
  if (alloc == nullptr) {
    PyErr_SetString(PyExc_KeyError, "unknown static-alias token");
    return nullptr;
  }
  std::string error;
  if (!synchronize_locked(alloc, &error)) {
    set_runtime_error(error);
    return nullptr;
  }
  Py_RETURN_NONE;
}

PyObject* python_static_alias_destroy(PyObject*, PyObject* args) {
  unsigned long long token_value = 0;
  int synchronize = 0;
  if (!PyArg_ParseTuple(args, "Kp", &token_value, &synchronize)) {
    return nullptr;
  }

  Registry& state = registry();
  std::lock_guard<std::mutex> lock(state.mutex);
  StaticAliasAllocation* alloc = lookup_locked(token_value);
  if (alloc == nullptr) {
    PyErr_SetString(PyExc_KeyError, "unknown static-alias token");
    return nullptr;
  }
  if (alloc->destroying) {
    PyErr_SetString(PyExc_RuntimeError,
                    "static-alias allocation is already being destroyed");
    return nullptr;
  }
  if (alloc->live_tensors != 0) {
    PyErr_Format(PyExc_RuntimeError,
                 "cannot destroy static-alias allocation with %llu live "
                 "DLPack tensor(s)",
                 static_cast<unsigned long long>(alloc->live_tensors));
    return nullptr;
  }

  std::string error;
  if (synchronize) {
    if (!synchronize_locked(alloc, &error)) {
      set_runtime_error(error);
      return nullptr;
    }
  } else if (!alloc->synchronized_for_destroy) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "static-alias allocation is not synchronized; destroy CUDA Graphs "
        "and tensors, then call synchronize() or close(synchronize=True)");
    return nullptr;
  }

  alloc->destroying = true;
  bool cleaned = false;
  {
    CurrentContextGuard context_guard(alloc->primary_context);
    if (!context_guard.Enter(&error)) {
      alloc->destroying = false;
      set_runtime_error(error);
      return nullptr;
    }
    cleaned = cleanup_allocation(alloc, &error);
  }
  // The caller context has been restored before releasing our retained
  // primary-context reference. The release is safe here because all mappings
  // and the handle are gone.
  if (cleaned && !release_primary_context(alloc, &error)) {
    cleaned = false;
  }
  if (!cleaned) {
    alloc->destroying = false;
    set_runtime_error(error.empty() ? "static-alias cleanup incomplete" : error);
    return nullptr;
  }

  state.allocations.erase(token_value);
  delete alloc;
  Py_RETURN_NONE;
}

#endif
