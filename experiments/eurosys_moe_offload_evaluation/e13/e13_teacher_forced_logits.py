#!/usr/bin/env python3
"""Teacher-forced raw-logit recorder used by the E13 numerical audit.

The processor records the unmodified float32 logits seen by the sampler and
then forces the next token from a frozen continuation.  It is intentionally
enabled only for correctness diagnostics; its CPU copies invalidate timing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


_RECORDS: dict[str, dict[str, Any]] = {}


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reset_records() -> None:
    _RECORDS.clear()


class _RequestTeacherForce:
    def __init__(self, logical_request: str, target: list[int], record: bool):
        self.logical_request = logical_request
        self.target = target
        self.record = record
        if record:
            existing = _RECORDS.get(logical_request)
            if existing is None:
                _RECORDS[logical_request] = {
                    "target_token_ids": list(target),
                    "target_token_ids_sha256": sha256_json(target),
                    "logits_by_position": {},
                    "recomputed_positions": set(),
                }
            elif existing["target_token_ids"] != target:
                raise RuntimeError(
                    "teacher target changed while rebuilding batch state: "
                    f"{logical_request}"
                )

    def __call__(
        self, output_token_ids: list[int], logits: torch.Tensor
    ) -> torch.Tensor:
        position = len(output_token_ids)
        if position >= len(self.target):
            raise RuntimeError(
                "teacher-forced request exceeded its frozen continuation: "
                f"{self.logical_request} position={position} length={len(self.target)}"
            )
        if self.record:
            # The sampler has converted the model logits to float32 but has not
            # yet applied this non-argmax-invariant processor.
            record = _RECORDS[self.logical_request]
            by_position = record["logits_by_position"]
            if position in by_position:
                record["recomputed_positions"].add(position)
            by_position[position] = (
                logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
            )
        target = int(self.target[position])
        logits.fill_(float("-inf"))
        logits[target] = 0.0
        return logits


class E13TeacherForcedLogitsProcessor(AdapterLogitsProcessor):
    """Adapt frozen per-request continuations to the vLLM V1 batch API."""

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        payload = (sampling_params.extra_args or {}).get("e13_teacher_forced")
        if payload is None:
            return
        if not isinstance(payload, dict):
            raise ValueError("e13_teacher_forced must be an object")
        logical = payload.get("logical_request")
        target = payload.get("target_token_ids")
        if not isinstance(logical, str) or not logical:
            raise ValueError("teacher-forced logical_request is missing")
        if not isinstance(target, list) or not target:
            raise ValueError("teacher-forced target_token_ids is empty")
        if any(not isinstance(value, int) or value < 0 for value in target):
            raise ValueError("teacher-forced target_token_ids are invalid")
        if sampling_params.max_tokens != len(target):
            raise ValueError(
                "teacher-forced target length must equal max_tokens: "
                f"{len(target)} != {sampling_params.max_tokens}"
            )

    def new_req_logits_processor(self, params: SamplingParams):
        payload = (params.extra_args or {}).get("e13_teacher_forced")
        if payload is None:
            return None
        return _RequestTeacherForce(
            str(payload["logical_request"]),
            [int(value) for value in payload["target_token_ids"]],
            bool(payload.get("record_logits", False)),
        )

    def is_argmax_invariant(self) -> bool:
        return False


def export_records(path: Path) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for logical, record in sorted(_RECORDS.items()):
        by_position = record["logits_by_position"]
        target = record["target_token_ids"]
        expected_positions = set(range(len(target)))
        if set(by_position) != expected_positions:
            raise RuntimeError(
                f"incomplete logits for {logical}: "
                f"observed={sorted(by_position)} expected={sorted(expected_positions)}"
            )
        serialized[logical] = {
            "target_token_ids": target,
            "target_token_ids_sha256": record["target_token_ids_sha256"],
            "logits": torch.stack(
                [by_position[position] for position in range(len(target))], dim=0
            ),
            "recomputed_positions": sorted(record["recomputed_positions"]),
        }
    payload = {
        "schema": "eurosys_e13_teacher_forced_logits_v1",
        "records": serialized,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite teacher logits: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "recorded_requests": len(serialized),
        "recorded_token_positions": sum(
            int(record["logits"].shape[0]) for record in serialized.values()
        ),
        "vocab_width": (
            int(next(iter(serialized.values()))["logits"].shape[1])
            if serialized
            else 0
        ),
    }
