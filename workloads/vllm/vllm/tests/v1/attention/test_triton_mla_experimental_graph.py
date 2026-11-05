from __future__ import annotations

from typing import Any

import pytest

from vllm.v1.attention.backends.mla.triton_mla import TritonMLAMetadataBuilder
from vllm.v1.attention.backends.utils import AttentionCGSupport


def test_experimental_full_graph_uses_generic_mla_prefill_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_TRITON_MLA_EXPERIMENTAL_FULL_CUDAGRAPH", "1")
    builder = object.__new__(TritonMLAMetadataBuilder)
    metadata = object()
    expected = object()
    calls: list[tuple[int, Any]] = []

    def build(common_prefix_len: int, common_attn_metadata: Any) -> object:
        calls.append((common_prefix_len, common_attn_metadata))
        return expected

    monkeypatch.setattr(builder, "build", build)

    assert (
        builder.get_cudagraph_support(None, None)
        is AttentionCGSupport.ALWAYS
    )
    assert builder.build_for_cudagraph_capture(metadata) is expected
    assert calls == [(0, metadata)]


def test_experimental_full_graph_rejects_invalid_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_TRITON_MLA_EXPERIMENTAL_FULL_CUDAGRAPH", "maybe")

    with pytest.raises(ValueError, match="must be 0 or 1"):
        TritonMLAMetadataBuilder.get_cudagraph_support(None, None)
