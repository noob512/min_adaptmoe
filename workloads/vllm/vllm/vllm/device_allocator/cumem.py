# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# cumem-based pytorch pluggable allocator to implement sleep mode.
# other approaches tried but failed:
# - cuda-python package binding
# - custom libcuda driver ctypes wrapper
# both of them failed because of cuda context mismatch.
# not sure why, they are created from a different context.
# the only successful approach is to call cuda driver API in C.
import dataclasses
import gc
import os
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available

logger = init_logger(__name__)


def find_loaded_library(lib_name) -> str | None:
    """
    According to according to https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html,
    the file `/proc/self/maps` contains the memory maps of the process, which includes the
    shared libraries loaded by the process. We can use this file to find the path of the
    a loaded library.
    """  # noqa
    found_line = None
    with open("/proc/self/maps") as f:
        for line in f:
            if lib_name in line:
                found_line = line
                break
    if found_line is None:
        # the library is not loaded in the current process
        return None
    # if lib_name is libcudart, we need to match a line with:
    # address /path/to/libcudart-hash.so.11.0
    start = found_line.index("/")
    path = found_line[start:].strip()
    filename = path.split("/")[-1]
    assert filename.rpartition(".so")[0].startswith(lib_name), (
        f"Unexpected filename: {filename} for library {lib_name}"
    )
    return path


cumem_available = False
static_alias_available = False
try:
    from vllm.cumem_allocator import (
        init_module,
        python_create_and_map,
        python_unmap_and_release,
    )
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    lib_name = find_loaded_library("cumem_allocator")
    libcudart = CudaRTLibrary()
    cumem_available = True
    try:
        from vllm.cumem_allocator import (
            static_alias_create as _static_alias_create,
        )
        from vllm.cumem_allocator import (
            static_alias_destroy as _static_alias_destroy,
        )
        from vllm.cumem_allocator import (
            static_alias_dlpack as _static_alias_dlpack,
        )
        from vllm.cumem_allocator import static_alias_info as _static_alias_info
        from vllm.cumem_allocator import (
            static_alias_synchronize as _static_alias_synchronize,
        )

        static_alias_available = torch.version.hip is None
    except ImportError:
        # Keep source checkouts compatible with an older, already-built
        # cumem_allocator extension. Static aliases remain explicitly
        # unavailable until the extension is rebuilt.
        _static_alias_create = None
        _static_alias_destroy = None
        _static_alias_dlpack = None
        _static_alias_info = None
        _static_alias_synchronize = None
except ModuleNotFoundError:
    # only cuda and rocm platforms support cumem allocator
    init_module = None
    python_create_and_map = None
    python_unmap_and_release = None
    CudaRTLibrary = None
    lib_name = None
    libcudart = None
    _static_alias_create = None
    _static_alias_destroy = None
    _static_alias_dlpack = None
    _static_alias_info = None
    _static_alias_synchronize = None

# py_device, py_alignedSize, py_d_mem, py_p_memHandle
HandleType = tuple[int, int, int, int]


@dataclasses.dataclass
class AllocationData:
    handle: HandleType
    tag: str
    cpu_backup_tensor: torch.Tensor | None = None


_DLPACK_DTYPES: dict[torch.dtype, tuple[int, int, int]] = {
    torch.int8: (0, 8, 1),
    torch.int16: (0, 16, 1),
    torch.int32: (0, 32, 1),
    torch.int64: (0, 64, 1),
    torch.uint8: (1, 8, 1),
    torch.float16: (2, 16, 1),
    torch.float32: (2, 32, 1),
    torch.float64: (2, 64, 1),
    torch.bfloat16: (4, 16, 1),
    torch.complex64: (5, 64, 1),
    torch.complex128: (5, 128, 1),
    torch.bool: (6, 8, 1),
}

# DLPack 1.2 FP8 codes. Some PyTorch releases expose only a subset, so add
# them dynamically instead of referencing missing torch.dtype attributes at
# import time.
for _dtype_name, _dtype_code in (
    ("float8_e3m4", 7),
    ("float8_e4m3", 8),
    ("float8_e4m3b11fnuz", 9),
    ("float8_e4m3fn", 10),
    ("float8_e4m3fnuz", 11),
    ("float8_e5m2", 12),
    ("float8_e5m2fnuz", 13),
    ("float8_e8m0fnu", 14),
):
    _dtype = getattr(torch, _dtype_name, None)
    if _dtype is not None:
        _DLPACK_DTYPES[_dtype] = (_dtype_code, 8, 1)


@dataclasses.dataclass(frozen=True)
class StaticAliasMetadata:
    """Immutable CUDA VMM layout for one physical streaming bank."""

    token: int
    device: int
    requested_size: int
    aligned_size: int
    granularity: int
    backing_bytes: int
    upload_va: int
    consumer_vas: tuple[int, ...]
    upload_access: str
    consumer_access: str


class StaticAliasAllocation:
    """An explicit CUDA VMM bank with stable upload and consumer aliases.

    One physical allocation is mapped once at a read-write upload VA and at
    ``consumer_count`` read-only consumer VAs. The returned CUDA tensors share
    that physical allocation and keep a native live-reference until they are
    destroyed.

    This object is intentionally independent of :class:`CuMemAllocator` and
    does not alter vLLM's normal CUDAPluggableAllocator behavior. It is meant
    for initialization-time construction of static streaming layouts.

    CUDA Graphs retain raw addresses without participating in the native
    reference count. Callers must destroy every Graph that captured these VAs,
    release every alias tensor, and then call ``close(synchronize=True)``.
    Consumer tensors are protected read-only by CUDA VMM; PyTorch cannot
    represent that property, so an in-place write may poison the CUDA context.
    """

    def __init__(self, metadata: StaticAliasMetadata):
        self._metadata = metadata
        self._closed = False

    @classmethod
    def create(
        cls, device: int | torch.device, size: int, consumer_count: int
    ) -> "StaticAliasAllocation":
        if not static_alias_available or _static_alias_create is None:
            raise RuntimeError(
                "CUDA VMM static aliases are unavailable; rebuild "
                "vllm.cumem_allocator from this source tree"
            )
        if isinstance(device, torch.device):
            if device.type != "cuda":
                raise ValueError("StaticAliasAllocation requires a CUDA device")
            device_index = device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
        else:
            device_index = int(device)
        raw = _static_alias_create(device_index, int(size), int(consumer_count))
        metadata = StaticAliasMetadata(
            token=int(raw["token"]),
            device=int(raw["device"]),
            requested_size=int(raw["requested_size"]),
            aligned_size=int(raw["aligned_size"]),
            granularity=int(raw["granularity"]),
            backing_bytes=int(raw["backing_bytes"]),
            upload_va=int(raw["upload_va"]),
            consumer_vas=tuple(int(value) for value in raw["consumer_vas"]),
            upload_access=str(raw["upload_access"]),
            consumer_access=str(raw["consumer_access"]),
        )
        return cls(metadata)

    @property
    def metadata(self) -> StaticAliasMetadata:
        return self._metadata

    @property
    def token(self) -> int:
        return self._metadata.token

    @property
    def device(self) -> int:
        return self._metadata.device

    @property
    def requested_size(self) -> int:
        return self._metadata.requested_size

    @property
    def aligned_size(self) -> int:
        return self._metadata.aligned_size

    @property
    def granularity(self) -> int:
        return self._metadata.granularity

    @property
    def backing_bytes(self) -> int:
        return self._metadata.backing_bytes

    @property
    def upload_va(self) -> int:
        return self._metadata.upload_va

    @property
    def consumer_vas(self) -> tuple[int, ...]:
        return self._metadata.consumer_vas

    @property
    def upload_access(self) -> str:
        return self._metadata.upload_access

    @property
    def consumer_access(self) -> str:
        return self._metadata.consumer_access

    @property
    def closed(self) -> bool:
        return self._closed

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("static-alias allocation is closed")

    @staticmethod
    def _normalize_view(
        shape: tuple[int, ...] | list[int] | torch.Size,
        dtype: torch.dtype,
        strides: tuple[int, ...] | list[int] | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...] | None, tuple[int, int, int]]:
        try:
            dlpack_dtype = _DLPACK_DTYPES[dtype]
        except KeyError as error:
            raise ValueError(f"unsupported DLPack dtype: {dtype}") from error
        shape_tuple = tuple(int(value) for value in shape)
        stride_tuple = (
            None if strides is None else tuple(int(value) for value in strides)
        )
        return shape_tuple, stride_tuple, dlpack_dtype

    def _tensor(
        self,
        role: int,
        consumer_index: int,
        shape: tuple[int, ...] | list[int] | torch.Size,
        dtype: torch.dtype,
        *,
        strides: tuple[int, ...] | list[int] | None,
        byte_offset: int,
    ) -> torch.Tensor:
        self._check_open()
        if _static_alias_dlpack is None:
            raise RuntimeError("static-alias DLPack support is unavailable")
        shape_tuple, stride_tuple, dlpack_dtype = self._normalize_view(
            shape, dtype, strides
        )
        export_dtype = dlpack_dtype
        reinterpret_dtype = None
        if dlpack_dtype[0] >= 7:
            # PyTorch 2.9 defines FP8 dtypes but its legacy DLPack capsule
            # importer still rejects the DLPack 1.2 FP8 codes (7--14). All
            # current FP8 formats occupy one byte, so export the identical
            # storage as uint8 and reinterpret it without a copy afterward.
            export_dtype = _DLPACK_DTYPES[torch.uint8]
            reinterpret_dtype = dtype
        capsule = _static_alias_dlpack(
            self.token,
            role,
            consumer_index,
            int(byte_offset),
            shape_tuple,
            stride_tuple,
            *export_dtype,
        )
        tensor = torch.from_dlpack(capsule)
        if reinterpret_dtype is not None:
            tensor = tensor.view(reinterpret_dtype)
        return tensor

    def upload_tensor(
        self,
        shape: tuple[int, ...] | list[int] | torch.Size,
        dtype: torch.dtype = torch.uint8,
        *,
        strides: tuple[int, ...] | list[int] | None = None,
        byte_offset: int = 0,
    ) -> torch.Tensor:
        """Return a writable CUDA tensor view at the stable upload VA."""
        return self._tensor(
            0,
            0,
            shape,
            dtype,
            strides=strides,
            byte_offset=byte_offset,
        )

    def consumer_tensor(
        self,
        index: int,
        shape: tuple[int, ...] | list[int] | torch.Size,
        dtype: torch.dtype = torch.uint8,
        *,
        strides: tuple[int, ...] | list[int] | None = None,
        byte_offset: int = 0,
    ) -> torch.Tensor:
        """Return a read-only CUDA tensor view at one stable consumer VA."""
        return self._tensor(
            1,
            int(index),
            shape,
            dtype,
            strides=strides,
            byte_offset=byte_offset,
        )

    def info(self) -> dict[str, Any]:
        """Return current native lifecycle state and immutable VA metadata."""
        self._check_open()
        if _static_alias_info is None:
            raise RuntimeError("static-alias info support is unavailable")
        return dict(_static_alias_info(self.token))

    def synchronize(self) -> None:
        """Synchronize the CUDA context after all alias tensors are released.

        This is a teardown barrier, not a general execution barrier. It fails
        closed while any DLPack-backed tensor remains live.
        """
        self._check_open()
        if _static_alias_synchronize is None:
            raise RuntimeError("static-alias synchronization is unavailable")
        _static_alias_synchronize(self.token)

    def close(self, *, synchronize: bool = False) -> None:
        """Unmap every alias and release the single physical backing handle.

        With the default ``synchronize=False``, native code rejects an
        allocation that has not passed the explicit teardown barrier. Passing
        ``synchronize=True`` performs that barrier as part of close. Both modes
        reject live DLPack tensors; CUDA Graphs must be destroyed by the caller
        before this method because CUDA does not expose their pointer owners.
        """
        if self._closed:
            return
        if _static_alias_destroy is None:
            raise RuntimeError("static-alias teardown support is unavailable")
        _static_alias_destroy(self.token, bool(synchronize))
        self._closed = True


def create_and_map(allocation_handle: HandleType) -> None:
    python_create_and_map(*allocation_handle)


def unmap_and_release(allocation_handle: HandleType) -> None:
    python_unmap_and_release(*allocation_handle)


def get_pluggable_allocator(
    python_malloc_fn: Callable[[int], int], python_free_func: Callable[[int, int], None]
) -> torch.cuda.memory.CUDAPluggableAllocator:
    init_module(python_malloc_fn, python_free_func)
    new_alloc = torch.cuda.memory.CUDAPluggableAllocator(
        lib_name, "my_malloc", "my_free"
    )
    return new_alloc


@contextmanager
def use_memory_pool_with_allocator(
    python_malloc_fn: Callable[[int], int], python_free_func: Callable[[int, int], None]
) -> None:
    new_alloc = get_pluggable_allocator(python_malloc_fn, python_free_func)
    mem_pool = torch.cuda.memory.MemPool(new_alloc._allocator)
    with torch.cuda.memory.use_mem_pool(mem_pool):
        yield mem_pool, new_alloc


class CuMemAllocator:
    """
    A singleton class that manages a memory pool for CUDA tensors.
    The memory in this pool can be offloaded or discarded when the
    allocator sleeps.

    Inside the `use_memory_pool(tag)` context, all tensors created will
    be allocated in the memory pool, and has the same tag as the
    tag passed to the context.

    When we call `sleep`, all tensors with the specified tag will be
    offloaded to CPU memory, and the rest of the tensors will be discarded.
    When we call `wake_up`, all tensors that are previously offloaded
    will be loaded back to GPU memory, and the rest of the tensors will
    have empty memory.

    Why it needs to be a singleton?
    When allocated tensors are garbage collected, PyTorch will call
    the free callback, which will call the `python_free_callback` method.
    The C-extension uses a global variable to store the function of an
    instance of this class. If we create multiple instances of this class,
    the global variable will be overwritten and the free callback will
    not work as expected.
    """

    instance: "CuMemAllocator" = None
    default_tag: str = "default"

    @staticmethod
    def get_instance() -> "CuMemAllocator":
        """
        CuMemAllocator is a singleton class.
        We cannot call the constructor directly.
        Call this method to get the instance.
        """
        assert cumem_available, "cumem allocator is not available"
        if CuMemAllocator.instance is None:
            CuMemAllocator.instance = CuMemAllocator()
        return CuMemAllocator.instance

    def __init__(self):
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        assert "expandable_segments:True" not in conf, (
            "Expandable segments are not compatible with memory pool. "
            "Please track https://github.com/pytorch/pytorch/issues/147851 "
            "for the latest updates."
        )

        self.pointer_to_data: dict[int, AllocationData] = {}
        self.current_tag: str = CuMemAllocator.default_tag
        self.allocator_and_pools: dict[str, Any] = {}
        # Creating strong references to the two callbacks here to prevent
        # these ephemeral bound-method objects being garbage collected.
        # See discussions in https://github.com/vllm-project/vllm/pull/22724
        self.python_malloc_callback = self._python_malloc_callback
        self.python_free_callback = self._python_free_callback

    def _python_malloc_callback(self, allocation_handle: HandleType) -> None:
        """
        Internal method to store the allocation data
        when memory is allocated in the memory pool."""
        py_d_mem = allocation_handle[2]
        self.pointer_to_data[py_d_mem] = AllocationData(
            allocation_handle, self.current_tag
        )
        logger.debug(
            "Allocated %s bytes for %s with address %s from cumem allocator",
            allocation_handle[1],
            self.current_tag,
            py_d_mem,
        )
        return

    def _python_free_callback(self, ptr: int) -> HandleType:
        """
        Internal method to look up the allocation data
        when memory is freed in the memory pool."""
        data = self.pointer_to_data.pop(ptr)
        if data.cpu_backup_tensor is not None:
            data.cpu_backup_tensor = None
        logger.debug(
            "Freed %s bytes for %s with address %s from cumem allocator",
            data.handle[1],
            data.tag,
            ptr,
        )
        return data.handle

    def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None:
        """
        Put the allocator in sleep mode.
        All data in the memory allocation with the specified tag will be
        offloaded to CPU memory, and others will be discarded.

        :param offload_tags: The tags of the memory allocation that will be
            offloaded. The rest of the memory allocation will be discarded.
        """
        if offload_tags is None:
            # by default, allocated tensors are offloaded
            # when the allocator sleeps
            offload_tags = (CuMemAllocator.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)

        assert isinstance(offload_tags, tuple)

        total_bytes = 0
        backup_bytes = 0

        for ptr, data in self.pointer_to_data.items():
            handle = data.handle
            total_bytes += handle[1]
            if data.tag in offload_tags:
                backup_bytes += handle[1]
                size_in_bytes = handle[1]
                cpu_backup_tensor = torch.empty(
                    size_in_bytes,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=is_pin_memory_available(),
                )
                cpu_ptr = cpu_backup_tensor.data_ptr()
                libcudart.cudaMemcpy(cpu_ptr, ptr, size_in_bytes)
                data.cpu_backup_tensor = cpu_backup_tensor
            unmap_and_release(handle)

        logger.info(
            "CuMemAllocator: sleep freed %.2f GiB memory in total, of which "
            "%.2f GiB is backed up in CPU and the rest %.2f GiB is discarded "
            "directly.",
            total_bytes / 1024**3,
            backup_bytes / 1024**3,
            (total_bytes - backup_bytes) / 1024**3,
        )

        gc.collect()
        torch.cuda.empty_cache()

    def wake_up(self, tags: list[str] | None = None) -> None:
        """
        Wake up the allocator from sleep mode.
        All data that is previously offloaded will be loaded back to GPU
        memory, and the rest of the data will have empty memory.

        :param tags: The tags of the memory allocation that will be loaded
            back to GPU memory. If None, all memory allocation will be loaded
            back to GPU memory.
        """
        for ptr, data in self.pointer_to_data.items():
            if tags is None or data.tag in tags:
                handle = data.handle
                create_and_map(handle)
                if data.cpu_backup_tensor is not None:
                    cpu_backup_tensor = data.cpu_backup_tensor
                    if cpu_backup_tensor is not None:
                        size_in_bytes = (
                            cpu_backup_tensor.numel() * cpu_backup_tensor.element_size()
                        )
                        cpu_ptr = cpu_backup_tensor.data_ptr()
                        libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)
                        data.cpu_backup_tensor = None

    @contextmanager
    def use_memory_pool(self, tag: str | None = None):
        """
        A context manager to use the memory pool.
        All memory allocation created inside the context will be allocated
        in the memory pool, and has the specified tag.

        :param tag: The tag of the memory allocation. If None, the default tag
            will be used.
        """
        if tag is None:
            tag = CuMemAllocator.default_tag

        assert isinstance(tag, str)

        old_tag = self.current_tag
        self.current_tag = tag
        with use_memory_pool_with_allocator(
            self.python_malloc_callback, self.python_free_callback
        ) as data:
            # start to hit another PyTorch bug in PyTorch 2.6,
            # possibly because of gc-related issue w.r.t. the allocator and
            # the memory pool.
            # to avoid the issue, we keep a reference of the data.
            # see https://github.com/pytorch/pytorch/issues/146431 .
            self.allocator_and_pools[tag] = data
            yield
            # PyTorch's bug, calling torch.cuda.empty_cache() will error
            # when using pluggable allocator, see
            # https://github.com/pytorch/pytorch/issues/145168 .
            # if we have some memory allocated and then freed,
            # the memory will not be released, e.g. in online quantization,
            # where the model is created in higher precision, and then
            # quantized in lower precision.
            # Find all unused allocations and manually release them.
            # TODO: we should expose `empty_cache` method in the memory pool.
            # TODO: ask for help from PyTorch team to expose this method.
            allocations = data[0].snapshot()
            for allocation in allocations:
                if allocation["allocated_size"] == 0:
                    handle = self._python_free_callback(allocation["address"])
                    unmap_and_release(handle)
            self.current_tag = old_tag

    def get_current_usage(self) -> int:
        """
        Get the total number of bytes allocated in the memory pool.
        """
        sum_bytes: int = 0
        for ptr, data in self.pointer_to_data.items():
            handle = data.handle
            sum_bytes += handle[1]
        return sum_bytes
