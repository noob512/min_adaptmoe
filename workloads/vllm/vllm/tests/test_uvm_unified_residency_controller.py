# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any

import pytest

from vllm.device_allocator.uvm_unified_residency_controller import (
    UnifiedResidencyController,
)


class _FakeKvRuntimePolicy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, object, str, dict[str, Any]]] = []

    def apply_unified_kv_target_blocks(
        self,
        target_blocks: int,
        *,
        block_pool: object,
        reason: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(("target", target_blocks, block_pool, reason, metadata))
        return {"ok": True, "target_blocks": target_blocks}

    def prefetch_kv_blocks_from_unified(
        self,
        block_specs: list[dict[str, int | None]],
        *,
        block_pool: object,
        reason: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls.append(("prefetch", block_specs, block_pool, reason, metadata))
        return {"ok": True, "block_specs": block_specs}


def test_control_file_keeps_real_actions_without_trace_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_file = tmp_path / "control.json"
    control_file.write_text(
        json.dumps(
            {
                "commands": [
                    {"kv_target_blocks": 12},
                    {"prefetch_kv_blocks": "0:1-2,3"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VLLM_UVM_UNIFIED_RESIDENCY_ENABLE", "1")
    monkeypatch.setenv(
        "VLLM_UVM_UNIFIED_RESIDENCY_CONTROL_FILE",
        str(control_file),
    )
    monkeypatch.setenv("VLLM_UVM_UNIFIED_RESIDENCY_CONTROL_ENABLE", "1")
    # This removed debug setting must not gate control-file execution.
    monkeypatch.setenv(
        "VLLM_UVM_UNIFIED_RESIDENCY_TRACE_FILE",
        str(tmp_path / "trace.jsonl"),
    )

    controller = UnifiedResidencyController()
    policy = _FakeKvRuntimePolicy()
    block_pool = object()

    result = controller.apply_control_file(
        kv_runtime_policy=policy,
        block_pool=block_pool,
        reason="test",
    )

    assert result["ok"] is True
    assert [call[0] for call in policy.calls] == ["target", "prefetch"]
    assert policy.calls[0][1:4] == (12, block_pool, "unified_control_file")
    assert policy.calls[1][1] == [
        {"group_id": 0, "block_id": 1},
        {"group_id": 0, "block_id": 2},
        {"group_id": None, "block_id": 3},
    ]
    assert (
        controller.apply_control_file(
            kv_runtime_policy=policy,
            block_pool=block_pool,
        )
        == {}
    )
    assert not (tmp_path / "trace.jsonl").exists()


def test_legacy_pool_control_is_not_reported_as_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_file = tmp_path / "control.json"
    control_file.write_text(
        json.dumps({"reclaim_pool": "weights", "required_bytes": 4096}),
        encoding="utf-8",
    )
    monkeypatch.setenv("VLLM_UVM_UNIFIED_RESIDENCY_ENABLE", "1")
    monkeypatch.setenv(
        "VLLM_UVM_UNIFIED_RESIDENCY_CONTROL_FILE",
        str(control_file),
    )

    result = UnifiedResidencyController().apply_control_file()

    assert result["ok"] is False
    command_result = result["results"][0]
    assert command_result["ok"] is False
    assert command_result["results"] == [
        {
            "action": "unsupported_control_command",
            "ok": False,
            "keys": ["reclaim_pool", "required_bytes"],
        }
    ]
