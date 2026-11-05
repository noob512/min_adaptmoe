# SPDX-License-Identifier: Apache-2.0

from vllm.compilation.cuda_graph import CUDAGraphEntry, CUDAGraphWrapper
from vllm.forward_context import BatchDescriptor


class _FakeGraph:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


def _wrapper() -> CUDAGraphWrapper:
    wrapper = object.__new__(CUDAGraphWrapper)
    wrapper.concrete_cudagraph_entries = {}
    wrapper._cudagraph_class_entries = {
        "default": wrapper.concrete_cudagraph_entries
    }
    wrapper._active_cudagraph_class = "default"
    return wrapper


def test_cudagraph_classes_preserve_entries_and_activate_without_reset() -> None:
    wrapper = _wrapper()
    c2_graph = _FakeGraph()
    c2_descriptor = BatchDescriptor(2, uniform_decode=False)
    wrapper.begin_cudagraph_class_capture("c2")
    wrapper.concrete_cudagraph_entries[c2_descriptor] = CUDAGraphEntry(
        c2_descriptor, cudagraph=c2_graph
    )

    c4_graph = _FakeGraph()
    c4_descriptor = BatchDescriptor(4, uniform_decode=False)
    wrapper.begin_cudagraph_class_capture("c4")
    wrapper.concrete_cudagraph_entries[c4_descriptor] = CUDAGraphEntry(
        c4_descriptor, cudagraph=c4_graph
    )

    result = wrapper.activate_cudagraph_class("c2")

    assert result == {
        "ok": True,
        "reason": "class_activated",
        "old_class": "c4",
        "new_class": "c2",
        "entry_count": 1,
    }
    assert wrapper.concrete_cudagraph_entries[c2_descriptor].cudagraph is c2_graph
    assert c2_graph.reset_count == 0
    assert c4_graph.reset_count == 0


def test_cudagraph_class_activation_rejects_missing_or_partial_class() -> None:
    wrapper = _wrapper()
    assert wrapper.activate_cudagraph_class("c2")["reason"] == (
        "class_not_registered"
    )

    wrapper.begin_cudagraph_class_capture("c2")
    descriptor = BatchDescriptor(2)
    wrapper.concrete_cudagraph_entries[descriptor] = CUDAGraphEntry(descriptor)

    result = wrapper.activate_cudagraph_class("c2")
    assert result["reason"] == "class_not_fully_captured"
    assert result["uncaptured_count"] == 1


def test_replacing_cudagraph_class_resets_only_that_class() -> None:
    wrapper = _wrapper()
    graph = _FakeGraph()
    descriptor = BatchDescriptor(2)
    wrapper.begin_cudagraph_class_capture("c2")
    wrapper.concrete_cudagraph_entries[descriptor] = CUDAGraphEntry(
        descriptor, cudagraph=graph
    )

    wrapper.begin_cudagraph_class_capture("c2", replace=True)

    assert graph.reset_count == 1
    assert wrapper.concrete_cudagraph_entries == {}


def test_discard_temporary_class_reactivates_live_default() -> None:
    wrapper = _wrapper()
    default_graph = _FakeGraph()
    default_descriptor = BatchDescriptor(1, uniform_decode=True)
    wrapper.concrete_cudagraph_entries[default_descriptor] = CUDAGraphEntry(
        default_descriptor,
        cudagraph=default_graph,
    )

    seed_graph = _FakeGraph()
    seed_descriptor = BatchDescriptor(16_384, uniform_decode=False)
    wrapper.begin_cudagraph_class_capture("prefill_pool_seed")
    wrapper.concrete_cudagraph_entries[seed_descriptor] = CUDAGraphEntry(
        seed_descriptor,
        cudagraph=seed_graph,
        output=object(),
        input_addresses=[1, 2],
    )

    result = wrapper.discard_cudagraph_class("prefill_pool_seed")

    assert result == {
        "ok": True,
        "reason": "class_discarded_pool_retained",
        "discarded_class": "prefill_pool_seed",
        "old_class": "prefill_pool_seed",
        "active_class": "default",
        "reset_count": 1,
        "active_entry_count": 1,
    }
    assert seed_graph.reset_count == 1
    assert default_graph.reset_count == 0
    assert wrapper.concrete_cudagraph_entries is wrapper._cudagraph_class_entries[
        "default"
    ]
    assert "prefill_pool_seed" not in wrapper._cudagraph_class_entries


def test_four_cudagraph_classes_switch_without_recapture_or_reset() -> None:
    wrapper = _wrapper()
    graphs: dict[str, _FakeGraph] = {}
    descriptors: dict[str, BatchDescriptor] = {}

    for width in (2, 4, 6, 8):
        class_name = f"c{width}"
        graph = _FakeGraph()
        descriptor = BatchDescriptor(width, uniform_decode=False)
        wrapper.begin_cudagraph_class_capture(class_name)
        wrapper.concrete_cudagraph_entries[descriptor] = CUDAGraphEntry(
            descriptor,
            cudagraph=graph,
        )
        graphs[class_name] = graph
        descriptors[class_name] = descriptor

    identities = {
        class_name: id(wrapper._cudagraph_class_entries[class_name])
        for class_name in graphs
    }
    for class_name in ("c2", "c4", "c6", "c8", "c2"):
        result = wrapper.activate_cudagraph_class(class_name)
        assert result["ok"] is True
        assert wrapper.concrete_cudagraph_entries[
            descriptors[class_name]
        ].cudagraph is graphs[class_name]

    assert {
        class_name: id(wrapper._cudagraph_class_entries[class_name])
        for class_name in graphs
    } == identities
    assert all(graph.reset_count == 0 for graph in graphs.values())
    diagnostics = wrapper.cudagraph_class_diagnostics()
    assert diagnostics["active_class"] == "c2"
    assert diagnostics["class_count"] == 5
    assert all(
        diagnostics["classes"][class_name]["fully_captured"]
        for class_name in graphs
    )


def test_startup_graph_table_can_be_adopted_as_initial_class() -> None:
    wrapper = _wrapper()
    graph = _FakeGraph()
    descriptor = BatchDescriptor(8, uniform_decode=False)
    wrapper.concrete_cudagraph_entries[descriptor] = CUDAGraphEntry(
        descriptor,
        cudagraph=graph,
    )
    original_entries = wrapper.concrete_cudagraph_entries

    result = wrapper.register_active_cudagraph_class("c8")

    assert result["ok"] is True
    assert result["reason"] == "active_class_registered"
    assert wrapper._active_cudagraph_class == "c8"
    assert wrapper._cudagraph_class_entries == {"c8": original_entries}
    assert graph.reset_count == 0


def test_equal_shape_classes_share_graph_table_and_reset_once() -> None:
    wrapper = _wrapper()
    graph = _FakeGraph()
    descriptor = BatchDescriptor(512, uniform_decode=False)
    wrapper.begin_cudagraph_class_capture("base")
    wrapper.concrete_cudagraph_entries[descriptor] = CUDAGraphEntry(
        descriptor,
        cudagraph=graph,
    )

    result = wrapper.register_cudagraph_class_alias("d005", "base")

    assert result["ok"] is True
    assert result["reason"] == "equal_shape_class_aliased"
    assert wrapper._cudagraph_class_entries["d005"] is (
        wrapper._cudagraph_class_entries["base"]
    )
    assert wrapper.activate_cudagraph_class("base")["ok"] is True
    assert wrapper.activate_cudagraph_class("d005")["ok"] is True
    assert wrapper.reset_all_cudagraph_classes() == 1
    assert graph.reset_count == 1


def test_reset_all_cudagraph_classes_releases_each_graph_once() -> None:
    wrapper = _wrapper()
    graphs = []
    for width in (2, 4, 6, 8):
        graph = _FakeGraph()
        graphs.append(graph)
        descriptor = BatchDescriptor(width, uniform_decode=False)
        wrapper.begin_cudagraph_class_capture(f"c{width}")
        wrapper.concrete_cudagraph_entries[descriptor] = CUDAGraphEntry(
            descriptor,
            cudagraph=graph,
        )

    reset_count = wrapper.reset_all_cudagraph_classes()

    assert reset_count == 4
    assert [graph.reset_count for graph in graphs] == [1, 1, 1, 1]
    assert wrapper.cudagraph_class_diagnostics() == {
        "active_class": "default",
        "class_count": 1,
        "classes": {
            "default": {
                "entry_count": 0,
                "captured_count": 0,
                "fully_captured": False,
            }
        },
    }
