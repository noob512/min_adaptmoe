from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmark_full_graph_steps as bench  # noqa: E402
import verify_integration as verify  # noqa: E402


def test_graph_tracker_delta_omits_zeroes() -> None:
    before = {"full_capture": 2, "full_replay": 5}
    after = {"full_capture": 2, "full_replay": 6, "none_passthrough": 1}
    assert bench.GraphCallTracker.delta(before, after) == {
        "full_replay": 1,
        "none_passthrough": 1,
    }


def test_decode_graph_delta_contract() -> None:
    verify.validate_graph_delta(
        {"full_replay": 1}, graph_arm=True, path="graph/step"
    )
    verify.validate_graph_delta({}, graph_arm=False, path="eager/step")
    with pytest.raises(verify.VerificationError, match="one FULL replay"):
        verify.validate_graph_delta({}, graph_arm=True, path="graph/step")
    with pytest.raises(verify.VerificationError, match="unexpectedly captured"):
        verify.validate_graph_delta(
            {"full_capture": 1}, graph_arm=True, path="graph/step"
        )


def test_alias_manifest_deduplicates_physical_backing(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    records = []
    sizes = {"w13_weight": 384 << 20, "w2_weight": 192 << 20}
    upload_va = {
        (bank, weight): 0x100000000 + bank * 0x40000000 + index * 0x20000000
        for bank in (0, 1)
        for index, weight in enumerate(sizes)
    }
    for layer in range(4):
        bank = layer % 2
        for weight_index, (weight, size) in enumerate(sizes.items()):
            records.append(
                {
                    "event": "cpu_offload_parameter",
                    "full_parameter_name": (
                        f"model.layers.{layer}.mlp.experts.{weight}"
                    ),
                    "bytes": size,
                    "graph_static_alias": True,
                    "graph_static_alias_layer_index": layer,
                    "graph_static_alias_bank_id": bank,
                    "graph_static_alias_upload_va": upload_va[(bank, weight)],
                    "graph_static_alias_source_ptr": 0x800000000 + len(records),
                    "graph_static_alias_consumer_va": (
                        0x400000000 + layer * 0x40000000 + weight_index * 0x20000000
                    ),
                    "graph_static_alias_backing_bytes": size,
                    "graph_static_alias_total_backing_bytes": (
                        verify.DUAL_BANK_BYTES
                    ),
                    "graph_static_alias_upload_access": "rw",
                    "graph_static_alias_consumer_access": "ro",
                }
            )
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    evidence = bench.read_manifest_evidence(path)
    assert evidence["parameter_record_count"] == 8
    assert evidence["parameter_logical_bytes"] == verify.LOGICAL_BYTES
    assert evidence["unique_physical_backing_bytes"] == verify.DUAL_BANK_BYTES
    assert evidence["bank_ids"] == [0, 1]
    assert evidence["layer_indices"] == [0, 1, 2, 3]
    assert len(evidence["upload_vas"]) == 4
    assert len(evidence["consumer_vas"]) == 8
    assert len(evidence["pinned_source_vas"]) == 8
    assert evidence["upload_access_values"] == ["rw"]
    assert evidence["consumer_access_values"] == ["ro"]


def test_alias_manifest_ignores_unprefixed_decoys(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        json.dumps(
            {
                "event": "cpu_offload_parameter",
                "bytes": 1,
                "graph_static_alias": True,
                "upload_va": 0xDEAD,
                "consumer_va": 0xBEEF,
                "backing_bytes": 123,
                "bank_id": 7,
                "layer_index": 9,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    evidence = bench.read_manifest_evidence(path)
    assert evidence["upload_vas"] == []
    assert evidence["consumer_vas"] == []
    assert evidence["unique_physical_backing_bytes"] == 0
    assert evidence["bank_ids"] == []
    assert evidence["layer_indices"] == []


def test_manifest_rejects_inconsistent_backing_size(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        json.dumps(
            {
                "event": "cpu_offload_parameter",
                "graph_static_alias": True,
                "graph_static_alias_upload_va": 0x1000,
                "graph_static_alias_backing_bytes": 4096,
            }
        )
        + "\n"
        + json.dumps(
            {
                "event": "cpu_offload_parameter",
                "graph_static_alias": True,
                "graph_static_alias_upload_va": 0x1000,
                "graph_static_alias_backing_bytes": 8192,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="inconsistent backing"):
        bench.read_manifest_evidence(path)
