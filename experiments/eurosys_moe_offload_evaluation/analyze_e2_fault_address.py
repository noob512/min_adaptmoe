#!/usr/bin/env python3
"""Attribute an E2 MMU fault address to a captured CUDA allocation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FAULT_PATTERN = re.compile(r"faulted @ 0x([0-9a-fA-F_]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def contains(record: dict[str, Any], address: int) -> bool:
    return int(record["address"]) <= address < int(record["end"])


def main() -> int:
    args = parse_args()
    kernel_log = (args.run_root / "kernel.since.log").read_text(
        encoding="utf-8", errors="replace"
    )
    match = FAULT_PATTERN.search(kernel_log)
    if match is None:
        result = {"status": "no_fault_address"}
    else:
        fault_address = int(match.group(1).replace("_", ""), 16)
        snapshots = json.loads(
            (args.run_root / "cuda_address_snapshots.json").read_text(
                encoding="utf-8"
            )
        )
        matches: list[dict[str, Any]] = []
        for snapshot in snapshots:
            segment_matches = []
            for segment in snapshot.get("segments", []):
                if not contains(segment, fault_address):
                    continue
                block_matches = [
                    block
                    for block in segment.get("blocks", [])
                    if contains(block, fault_address)
                ]
                segment_matches.append(
                    {
                        **{key: value for key, value in segment.items() if key != "blocks"},
                        "fault_offset": fault_address - int(segment["address"]),
                        "matching_blocks": block_matches,
                    }
                )
            tensor_matches = [
                record
                for record in snapshot.get("named_tensor_ranges", [])
                if int(record["base"]) <= fault_address < int(record["end"])
            ]
            graph_matches = []
            for record in snapshot.get("graph_entries", []):
                matching_outputs = [
                    output
                    for output in record.get("output_tensors", [])
                    if int(output["base"])
                    <= fault_address
                    < int(output["end"])
                ]
                matching_inputs = [
                    address
                    for address in record.get("input_addresses", [])
                    if int(address) == fault_address
                ]
                if matching_outputs or matching_inputs:
                    graph_matches.append(
                        {
                            **{
                                key: value
                                for key, value in record.items()
                                if key != "output_tensors"
                            },
                            "matching_input_addresses": matching_inputs,
                            "matching_output_tensors": matching_outputs,
                        }
                    )
            matches.append(
                {
                    "stage": snapshot.get("stage"),
                    "segments": segment_matches,
                    "named_tensors": tensor_matches,
                    "graph_entries": graph_matches,
                }
            )
        result = {
            "status": "ok",
            "fault_address": fault_address,
            "fault_address_hex": f"0x{fault_address:x}",
            "snapshots": matches,
        }
    output = args.output or args.run_root / "fault_address_attribution.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
