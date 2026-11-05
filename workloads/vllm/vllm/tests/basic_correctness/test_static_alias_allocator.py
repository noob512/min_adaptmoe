# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc

import pytest
import torch

from vllm.device_allocator.cumem import (
    StaticAliasAllocation,
    static_alias_available,
)
from vllm.platforms import current_platform

from ..utils import create_new_process_for_each_test


def _run_graph_replay(
    dtype: torch.dtype, first_value: float, second_value: float
) -> None:
    size = 2 * 1024 * 1024
    allocation = StaticAliasAllocation.create(0, size, consumer_count=2)
    upload = allocation.upload_tensor((size,), dtype)
    consumer = allocation.consumer_tensor(0, (size,), dtype)
    source = torch.empty(size, dtype=dtype, device="cpu", pin_memory=True)
    source.fill_(first_value)
    output = torch.empty_like(consumer)

    assert upload.dtype == dtype
    assert consumer.dtype == dtype
    assert upload.data_ptr() == allocation.upload_va
    assert consumer.data_ptr() == allocation.consumer_vas[0]
    assert allocation.aligned_size % allocation.granularity == 0
    assert allocation.backing_bytes == allocation.aligned_size
    assert allocation.upload_access == "rw"
    assert allocation.consumer_access == "ro"

    # Warm the copy and consumer paths before capture.
    upload.copy_(source, non_blocking=True)
    output.copy_(consumer)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        upload.copy_(source, non_blocking=True)
        output.copy_(consumer)

    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output.cpu().view(torch.uint8), source.view(torch.uint8))

    # A replay must execute H2D again without changing either CUDA VA.
    source.fill_(second_value)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output.cpu().view(torch.uint8), source.view(torch.uint8))
    assert upload.data_ptr() == allocation.upload_va
    assert consumer.data_ptr() == allocation.consumer_vas[0]

    with pytest.raises(RuntimeError, match="live DLPack tensor"):
        allocation.close(synchronize=True)

    del graph, upload, consumer, output, source
    gc.collect()
    allocation.close(synchronize=True)
    assert allocation.closed


@pytest.mark.skipif(
    not static_alias_available or not current_platform.is_cuda(),
    reason="CUDA VMM static aliases are unavailable",
)
@create_new_process_for_each_test("spawn")
def test_static_alias_cuda_graph_replay():
    _run_graph_replay(torch.uint8, 17, 93)
    _run_graph_replay(torch.float8_e4m3fn, 1.5, 3.0)
    _run_graph_replay(torch.float8_e5m2, 1.5, 3.0)


@pytest.mark.skipif(
    not static_alias_available or not current_platform.is_cuda(),
    reason="CUDA VMM static aliases are unavailable",
)
@create_new_process_for_each_test("spawn")
def test_static_alias_view_bounds_and_teardown_barrier():
    allocation = StaticAliasAllocation.create(0, 4103, consumer_count=1)
    assert allocation.requested_size == 4103
    assert allocation.aligned_size >= allocation.requested_size

    view = allocation.consumer_tensor(
        0, (10, 5), torch.uint8, strides=(5, 1), byte_offset=3
    )
    assert view.data_ptr() == allocation.consumer_vas[0] + 3
    with pytest.raises(ValueError, match="exceeds requested_size"):
        allocation.consumer_tensor(0, (4104,), torch.uint8)
    with pytest.raises(ValueError, match="negative strides"):
        allocation.consumer_tensor(0, (2, 2), torch.uint8, strides=(-2, 1))

    del view
    gc.collect()
    with pytest.raises(RuntimeError, match="not synchronized"):
        allocation.close()
    allocation.synchronize()
    allocation.close()
    assert allocation.closed
