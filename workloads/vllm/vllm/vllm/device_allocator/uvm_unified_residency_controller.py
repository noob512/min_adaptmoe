# SPDX-License-Identifier: Apache-2.0
"""Manual control-file bridge for application-level KV placement."""

from __future__ import annotations

import json
import os
import shlex
import threading
from typing import Any


def _read_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, str) and value.startswith("0x"):
            return int(value, 16)
        return int(value)
    except (TypeError, ValueError):
        return default


class UnifiedResidencyController:
    """Apply functional KV target and prefetch commands from a control file."""

    def __init__(self) -> None:
        self.enabled = _read_bool_env(
            "VLLM_UVM_UNIFIED_RESIDENCY_ENABLE",
            False,
        )
        self.control_file = os.environ.get(
            "VLLM_UVM_UNIFIED_RESIDENCY_CONTROL_FILE",
            "",
        )
        self.control_enabled = _read_bool_env(
            "VLLM_UVM_UNIFIED_RESIDENCY_CONTROL_ENABLE",
            bool(self.control_file),
        )
        self._lock = threading.Lock()
        self._control_mtime_ns: int | None = None
        self._control_size: int | None = None
        self._control_generation = 0

    @property
    def active(self) -> bool:
        return self.enabled

    def apply_control_file(
        self,
        *,
        kv_runtime_policy: Any | None = None,
        block_pool: Any | None = None,
        reason: str = "poll",
    ) -> dict[str, Any]:
        """Apply supported commands once whenever the control file changes."""
        if not self.active or not self.control_enabled or not self.control_file:
            return {}
        try:
            stat = os.stat(self.control_file)
        except FileNotFoundError:
            return {}
        except OSError as exc:  # pragma: no cover - optional control file.
            return {
                "ok": False,
                "action": "unified_control_file_failed",
                "control_file": self.control_file,
                "reason": reason,
                "error": repr(exc),
            }

        if (
            stat.st_mtime_ns == self._control_mtime_ns
            and stat.st_size == self._control_size
        ):
            return {}
        self._control_mtime_ns = stat.st_mtime_ns
        self._control_size = stat.st_size

        try:
            with open(
                self.control_file,
                encoding="utf-8",
                errors="replace",
            ) as handle:
                raw = handle.read()
        except OSError as exc:  # pragma: no cover - optional control file.
            return {
                "ok": False,
                "action": "unified_control_file_failed",
                "control_file": self.control_file,
                "reason": reason,
                "error": repr(exc),
            }

        commands = self._parse_control_text(raw)
        with self._lock:
            self._control_generation += 1
            generation = self._control_generation
        command_results = [
            self._execute_control_command(
                command,
                kv_runtime_policy=kv_runtime_policy,
                block_pool=block_pool,
                generation=generation,
                reason=reason,
            )
            for command in commands
        ]
        return {
            "ok": all(bool(result.get("ok")) for result in command_results),
            "action": "unified_control_file_applied",
            "control_file": self.control_file,
            "reason": reason,
            "generation": generation,
            "command_count": len(commands),
            "commands": commands,
            "results": command_results,
        }

    def _parse_control_text(self, raw: str) -> list[dict[str, Any]]:
        text = (raw or "").strip()
        if not text:
            return []
        if text.startswith("{") or text.startswith("["):
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                return [{"invalid_control": text, "error": str(exc)}]
            if isinstance(data, list):
                return [dict(item) for item in data if isinstance(item, dict)]
            if isinstance(data, dict):
                commands = data.get("commands")
                if isinstance(commands, list):
                    return [dict(item) for item in commands if isinstance(item, dict)]
                return [dict(data)]
            return [{"invalid_control": text, "error": "json_root_not_object"}]

        commands: list[dict[str, Any]] = []
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            command: dict[str, Any] = {}
            for token in shlex.split(line):
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                command[key.strip()] = value.strip()
            if command:
                commands.append(command)
        return commands

    def _execute_control_command(
        self,
        command: dict[str, Any],
        *,
        kv_runtime_policy: Any | None,
        block_pool: Any | None,
        generation: int,
        reason: str,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "command": command,
            "generation": generation,
            "reason": reason,
            "results": [],
        }
        if "invalid_control" in command:
            result["ok"] = False
            result["results"].append(
                {
                    "action": "invalid_control",
                    "ok": False,
                    "error": command.get("error"),
                }
            )
            return result

        if "kv_target_blocks" in command:
            target_blocks = _as_int(command.get("kv_target_blocks"), 0)
            if kv_runtime_policy is None or not hasattr(
                kv_runtime_policy,
                "apply_unified_kv_target_blocks",
            ):
                target_result = {
                    "ok": False,
                    "reason": "kv_runtime_policy_unavailable",
                    "target_blocks": target_blocks,
                }
            else:
                target_result = kv_runtime_policy.apply_unified_kv_target_blocks(
                    target_blocks,
                    block_pool=block_pool,
                    reason="unified_control_file",
                    metadata={
                        "generation": generation,
                        "command": command,
                    },
                )
            result["results"].append(
                {
                    "action": "kv_target_blocks",
                    **target_result,
                }
            )

        if "prefetch_kv_blocks" in command:
            block_specs = self._parse_kv_block_specs(command.get("prefetch_kv_blocks"))
            if kv_runtime_policy is None or not hasattr(
                kv_runtime_policy,
                "prefetch_kv_blocks_from_unified",
            ):
                prefetch_result = {
                    "ok": False,
                    "reason": "kv_runtime_policy_unavailable",
                    "block_specs": block_specs,
                }
            else:
                prefetch_result = kv_runtime_policy.prefetch_kv_blocks_from_unified(
                    block_specs,
                    block_pool=block_pool,
                    reason="unified_control_file",
                    metadata={
                        "generation": generation,
                        "command": command,
                    },
                )
            result["results"].append(
                {
                    "action": "prefetch_kv_blocks",
                    **prefetch_result,
                }
            )

        if not result["results"]:
            result["results"].append(
                {
                    "action": "unsupported_control_command",
                    "ok": False,
                    "keys": sorted(command),
                }
            )
        result["ok"] = all(bool(item.get("ok")) for item in result["results"])
        return result

    def _parse_kv_block_specs(self, raw: Any) -> list[dict[str, int | None]]:
        if raw is None:
            return []
        if isinstance(raw, list):
            specs: list[dict[str, int | None]] = []
            for item in raw:
                if isinstance(item, dict):
                    specs.append(
                        {
                            "group_id": (
                                _as_int(item.get("group_id"))
                                if item.get("group_id") is not None
                                else None
                            ),
                            "block_id": _as_int(item.get("block_id"), -1),
                        }
                    )
                else:
                    specs.extend(self._parse_kv_block_specs(item))
            return [
                spec
                for spec in specs
                if spec.get("block_id") is not None and int(spec["block_id"]) >= 0
            ]
        text = str(raw).strip()
        if not text:
            return []
        specs = []
        for item in text.split(","):
            part = item.strip()
            if not part:
                continue
            group_id: int | None = None
            block_part = part
            if ":" in part:
                group_text, block_part = part.split(":", 1)
                group_id = _as_int(group_text, -1)
                if group_id < 0:
                    continue
            for block_id in self._expand_block_ids(block_part):
                specs.append({"group_id": group_id, "block_id": block_id})
        return specs

    @staticmethod
    def _expand_block_ids(text: str) -> list[int]:
        value = text.strip()
        if ".." in value:
            start_text, end_text = value.split("..", 1)
        elif "-" in value:
            start_text, end_text = value.split("-", 1)
        else:
            block_id = _as_int(value, -1)
            return [block_id] if block_id >= 0 else []
        start = _as_int(start_text, -1)
        end = _as_int(end_text, -1)
        if start < 0 or end < start:
            return []
        return list(range(start, end + 1))


_UNIFIED_CONTROLLER: UnifiedResidencyController | None = None
_UNIFIED_CONTROLLER_LOCK = threading.Lock()


def get_uvm_unified_residency_controller() -> UnifiedResidencyController:
    global _UNIFIED_CONTROLLER
    if _UNIFIED_CONTROLLER is None:
        with _UNIFIED_CONTROLLER_LOCK:
            if _UNIFIED_CONTROLLER is None:
                _UNIFIED_CONTROLLER = UnifiedResidencyController()
    return _UNIFIED_CONTROLLER
