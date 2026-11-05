#!/usr/bin/env python3
"""Freeze Adaptive continuations and known-divergent E13 requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["logical_request"]): item
        for item in result.get("requests") or []
        if item.get("success")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=("formal", "capacity"), required=True)
    args = parser.parse_args()

    source = args.source_run_root.resolve()
    output = args.output_root.resolve()
    protocol = load_json(args.config)
    seeds = [int(value) for value in protocol["profiles"][args.profile]["seeds"]]
    fixed_arms = [
        str(item["name"])
        for item in protocol["arms"]
        if item["strategy"] == "fixed"
    ]
    repeats: list[dict[str, Any]] = []
    for repeat_index, seed in enumerate(seeds):
        repeat_name = f"repeat-{repeat_index:02d}-seed-{seed}"
        source_repeat = source / repeat_name
        adaptive_path = source_repeat / "adaptive" / "result.json"
        adaptive_result = load_json(adaptive_path)
        if adaptive_result.get("status") != "ok":
            raise RuntimeError(f"Adaptive source is not verified: {adaptive_path}")
        adaptive = request_map(adaptive_result)
        targets = {
            logical: [int(value) for value in record["output_token_ids"]]
            for logical, record in adaptive.items()
        }
        affected: dict[str, list[str]] = {}
        record_logicals: set[str] = set()
        source_hashes = {"adaptive": file_sha256(adaptive_path)}
        for arm in fixed_arms:
            arm_path = source_repeat / arm / "result.json"
            result = load_json(arm_path)
            if result.get("status") != "ok":
                raise RuntimeError(f"Fixed source is not verified: {arm_path}")
            source_hashes[arm] = file_sha256(arm_path)
            candidate = request_map(result)
            if set(candidate) != set(adaptive):
                raise RuntimeError(f"logical request set differs for {repeat_name}/{arm}")
            mismatches = sorted(
                logical
                for logical in adaptive
                if candidate[logical].get("output_token_ids")
                != adaptive[logical].get("output_token_ids")
            )
            if mismatches:
                affected[arm] = mismatches
                record_logicals.update(mismatches)

        target_path = output / "teacher_targets" / f"{repeat_name}.json"
        target_payload = {
            "schema": "eurosys_e13_teacher_targets_v1",
            "profile": args.profile,
            "repeat_index": repeat_index,
            "seed": seed,
            "source_run_root": str(source),
            "source_result_sha256": source_hashes,
            "trace_sha256": adaptive_result.get("trace_sha256"),
            "affected_fixed_arms": affected,
            "record_logical_requests": sorted(record_logicals),
            "targets": targets,
        }
        atomic_json(target_path, target_payload)
        cases = ["adaptive", *affected]
        repeats.append(
            {
                "repeat_index": repeat_index,
                "seed": seed,
                "repeat_name": repeat_name,
                "target_path": str(target_path),
                "record_logical_requests": sorted(record_logicals),
                "affected_fixed_arms": affected,
                "cases": cases if affected else [],
            }
        )

    manifest = {
        "schema": "eurosys_e13_numerical_manifest_v1",
        "profile": args.profile,
        "source_run_root": str(source),
        "source_verification_sha256": file_sha256(source / "verification.json"),
        "config_path": str(args.config.resolve()),
        "config_sha256": file_sha256(args.config),
        "repeats": repeats,
        "case_count": sum(len(item["cases"]) for item in repeats),
        "divergent_request_count": sum(
            len(item["record_logical_requests"]) for item in repeats
        ),
    }
    atomic_json(output / "numerical_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
