#!/usr/bin/env python3
"""Compare one cold auto-pressure E10 arm with one cold direct-tier arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise RuntimeError(f"invalid {label}: {value!r}")
    return number


def delta(direct: float, pressure: float) -> dict[str, Any]:
    absolute = direct - pressure
    return {
        "auto_pressure_ms": pressure,
        "direct_tier_ms": direct,
        "direct_minus_auto_ms": absolute,
        "direct_minus_auto_percent": (
            absolute * 100.0 / pressure if pressure else None
        ),
        "observed_absolute_difference_within_10_percent": (
            abs(absolute) <= pressure * 0.10 if pressure else direct == 0
        ),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    root = args.run_root.resolve()
    arms: dict[str, dict[str, Any]] = {}
    for name, expected_mode in (
        ("auto_pressure", "pressure"),
        ("direct_tier", "direct"),
    ):
        arm_root = root / name
        verification_path = arm_root / "verification.json"
        result_path = arm_root / "sequence" / "result.json"
        verification = load(verification_path)
        result = load(result_path)
        if verification.get("status") != "ok":
            raise RuntimeError(f"{name} verifier did not pass")
        if verification.get("trigger_mode") != expected_mode:
            raise RuntimeError(f"{name} trigger mode differs")
        qualification = result.get("kv_multilevel_pressure") or {}
        tiers = qualification.get("tiers") or []
        if len(tiers) != 3:
            raise RuntimeError(f"{name} does not contain three tiers")
        arms[name] = {
            "verification": verification,
            "qualification": qualification,
            "tiers": tiers,
            "verification_sha256": sha256(verification_path),
            "result_sha256": sha256(result_path),
        }

    pressure = arms["auto_pressure"]
    direct = arms["direct_tier"]
    if (
        pressure["verification"].get("output_token_ids_sha256")
        != direct["verification"].get("output_token_ids_sha256")
    ):
        raise RuntimeError("generated output differs between trigger modes")

    tiers: dict[str, Any] = {}
    for pressure_tier, direct_tier in zip(
        pressure["tiers"], direct["tiers"], strict=True
    ):
        tier_gib = int(pressure_tier["tier_gib"])
        if (
            int(direct_tier["tier_gib"]) != tier_gib
            or int(direct_tier["target_blocks"])
            != int(pressure_tier["target_blocks"])
            or direct_tier["plan_class"] != pressure_tier["plan_class"]
        ):
            raise RuntimeError(f"tier identity differs at {tier_gib} GiB")
        pt = pressure_tier["timing"]
        dt = direct_tier["timing"]
        pressure_prefill = finite(
            pt["prefill_compute_excluding_rebalance_and_resume_ms"],
            f"pressure {tier_gib} GiB pure Prefill",
        )
        direct_prefill = finite(
            dt["prefill_compute_excluding_rebalance_and_resume_ms"],
            f"direct {tier_gib} GiB pure Prefill",
        )
        tiers[f"{tier_gib}g"] = {
            "target_blocks": int(pressure_tier["target_blocks"]),
            "plan_class": pressure_tier["plan_class"],
            "tier_transition": delta(
                finite(dt["tier_transition_wall_ms"], "direct transition"),
                finite(pt["tier_transition_wall_ms"], "pressure transition"),
            ),
            "shared_tier_transaction_rpc": delta(
                finite(
                    dt.get("rebalance_rpc_including_prefill_resume_ms", 0.0),
                    "direct shared tier transaction",
                ),
                finite(
                    pt.get("rebalance_rpc_including_prefill_resume_ms", 0.0),
                    "pressure shared tier transaction",
                ),
            ),
            "decode_to_prefill_resume": delta(
                finite(dt["prefill_topology_resume_ms"], "direct D-to-P"),
                finite(pt["prefill_topology_resume_ms"], "pressure D-to-P"),
            ),
            "pure_prefill": delta(direct_prefill, pressure_prefill),
            "prefill_to_decode": delta(
                finite(dt["first_decode_topology_park_ms"], "direct P-to-D"),
                finite(pt["first_decode_topology_park_ms"], "pressure P-to-D"),
            ),
            "steady_decode_median": delta(
                finite(dt["steady_decode_step_ms"]["median"], "direct Decode"),
                finite(pt["steady_decode_step_ms"]["median"], "pressure Decode"),
            ),
            "request_excluding_pre_request_transition": delta(
                finite(dt["request_end_to_end_ms"], "direct request"),
                finite(pt["request_end_to_end_ms"], "pressure request"),
            ),
            "request_including_transition": delta(
                finite(
                    dt["request_plus_tier_transition_ms"],
                    "direct request plus transition",
                ),
                finite(
                    pt["request_plus_tier_transition_ms"],
                    "pressure request plus transition",
                ),
            ),
        }

    comparison = {
        "schema": "eurosys_e10_trigger_mode_comparison_v1",
        "status": "ok",
        "qualification_only": True,
        "statistical_repetitions": 0,
        "completed_sequences_per_arm": 1,
        "independent_cold_processes": True,
        "arms": {
            name: {
                "trigger_mode": data["verification"]["trigger_mode"],
                "result_sha256": data["result_sha256"],
                "verification_sha256": data["verification_sha256"],
            }
            for name, data in arms.items()
        },
        "output_token_ids_sha256": pressure["verification"][
            "output_token_ids_sha256"
        ],
        "tiers": tiers,
        "interpretation": (
            "Observed differences come from one sample per arm; 10% flags are "
            "descriptive engineering checks, not statistical equivalence tests."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
