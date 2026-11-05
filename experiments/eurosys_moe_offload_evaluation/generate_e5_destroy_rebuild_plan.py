#!/usr/bin/env python3
"""Generate the frozen E5 all-experts-in-layers-0..3 matched plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected = {str(layer): list(range(128)) for layer in range(4)}
    payload = {
        "schema": "eurosys_e5_qwen_layers_0_3_full_expert_plan_v1",
        "selection_rule": "all 128 experts in each of layers 0,1,2,3",
        "selected_by_layer": selected,
        "metadata": {
            "num_layers": 4,
            "num_experts_per_layer": 128,
            "selected_expert_slots": 512,
            "selected_parameter_count": 8,
            "selected_main_bytes": 2_415_919_104,
            "logical_bytes_per_forward": 2_415_919_104,
            "dual_bank_backing_bytes": 1_207_959_552,
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["metadata"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
