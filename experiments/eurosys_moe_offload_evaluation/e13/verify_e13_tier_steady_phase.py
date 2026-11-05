#!/usr/bin/env python3
"""Verify and summarize the one-shot E13 same-tier phase comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


TIERS = ("k2", "k4", "k6")
BATCHES = (1, 8)
EQUIVALENCE_LOWER = 0.90
EQUIVALENCE_UPPER = 1.10


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    checks: list[dict[str, Any]] = []
    cases: dict[str, dict[str, Any]] = {}

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    for tier in TIERS:
        for mechanism in ("direct", "switched"):
            arm = f"{mechanism}_{tier}"
            case_dir = root / arm
            result_path = case_dir / "result.json"
            returncode_path = case_dir / "returncode.txt"
            present = result_path.is_file() and returncode_path.is_file()
            check(f"{arm}_artifacts", present, str(case_dir))
            if not present:
                continue
            result = load_json(result_path)
            cases[arm] = result
            returncode = int(returncode_path.read_text(encoding="utf-8").strip())
            check(f"{arm}_returncode", returncode == 0, returncode)
            check(f"{arm}_status", result.get("status") == "ok", result.get("status"))
            check(
                f"{arm}_identity",
                result.get("arm") == arm
                and result.get("target_tier") == tier
                and result.get("mechanism") == mechanism,
                {
                    "arm": result.get("arm"),
                    "target_tier": result.get("target_tier"),
                    "mechanism": result.get("mechanism"),
                },
            )
            phases = result.get("phases") or []
            check(
                f"{arm}_phase_coverage",
                [item.get("batch_size") for item in phases] == [1, 8],
                [item.get("batch_size") for item in phases],
            )
            check(
                f"{arm}_measurement_contract",
                bool(result.get("configuration", {}).get("switch_transaction_excluded"))
                and bool(
                    result.get("configuration", {}).get(
                        "source_conditioning_excluded"
                    )
                )
                and bool(
                    result.get("configuration", {}).get(
                        "controlled_decode_tokens"
                    )
                )
                and not bool(
                    result.get("configuration", {}).get(
                        "controlled_token_logits_recording", True
                    )
                )
                and bool(result.get("configuration", {}).get("target_tier_warmup_excluded"))
                and (
                    mechanism != "switched"
                    or (
                        result.get("source_conditioning", {}).get("tier")
                        == result.get("source_tier")
                        and int(
                            result.get("source_conditioning", {}).get(
                                "output_tokens_per_request", -1
                            )
                        )
                        == 32
                    )
                )
                and all(
                    int(item.get("prefill_scheduled_tokens", -1))
                    == 2048 * int(item.get("batch_size", 0))
                    and int(item.get("steady_decode_step_ms", {}).get("count", -1)) == 16
                    for item in phases
                ),
                [
                    {
                        "batch": item.get("batch_size"),
                        "prefill_tokens": item.get("prefill_scheduled_tokens"),
                        "decode_steps": item.get("steady_decode_step_ms", {}).get("count"),
                    }
                    for item in phases
                ],
            )
            before = result.get("measurement_audit", {}).get("before", {})
            after = result.get("measurement_audit", {}).get("after", {})
            check(
                f"{arm}_tier_stable",
                before.get("kv", {}).get("current_target_blocks")
                == after.get("kv", {}).get("current_target_blocks")
                and before.get("plan", {}).get("active_plan_class")
                == after.get("plan", {}).get("active_plan_class"),
                {"before": before, "after": after},
            )
            gpu_after_path = case_dir / "gpu.after.json"
            gpu_after = load_json(gpu_after_path) if gpu_after_path.is_file() else {}
            check(f"{arm}_gpu_clean", gpu_after.get("gpu_clean") is True, gpu_after)

    rows: list[dict[str, Any]] = []
    for tier in TIERS:
        direct = cases.get(f"direct_{tier}")
        switched = cases.get(f"switched_{tier}")
        if direct is None or switched is None:
            continue
        direct_phases = {int(item["batch_size"]): item for item in direct["phases"]}
        switched_phases = {int(item["batch_size"]): item for item in switched["phases"]}
        for batch in BATCHES:
            d = direct_phases[batch]
            s = switched_phases[batch]
            output_equal = d["output_token_ids_sha256"] == s["output_token_ids_sha256"]
            check(f"{tier}_b{batch}_output_equal", output_equal, {
                "direct": d["output_token_ids_sha256"],
                "switched": s["output_token_ids_sha256"],
            })
            prefill_latency_ratio = float(s["prefill_ms"]) / float(d["prefill_ms"])
            decode_latency_ratio = (
                float(s["steady_decode_step_ms"]["median"])
                / float(d["steady_decode_step_ms"]["median"])
            )
            rows.append(
                {
                    "tier": tier,
                    "batch_size": batch,
                    "direct_prefill_ms": float(d["prefill_ms"]),
                    "switched_prefill_ms": float(s["prefill_ms"]),
                    "switched_over_direct_prefill_latency": prefill_latency_ratio,
                    "prefill_equivalent_within_10pct": (
                        EQUIVALENCE_LOWER <= prefill_latency_ratio <= EQUIVALENCE_UPPER
                    ),
                    "direct_decode_median_ms": float(d["steady_decode_step_ms"]["median"]),
                    "switched_decode_median_ms": float(s["steady_decode_step_ms"]["median"]),
                    "switched_over_direct_decode_latency": decode_latency_ratio,
                    "decode_equivalent_within_10pct": (
                        EQUIVALENCE_LOWER <= decode_latency_ratio <= EQUIVALENCE_UPPER
                    ),
                    "output_hash_equal": output_equal,
                }
            )

    execution_ok = len(cases) == 6 and all(item["ok"] for item in checks)
    all_prefill_equivalent = len(rows) == 6 and all(
        item["prefill_equivalent_within_10pct"] for item in rows
    )
    all_decode_equivalent = len(rows) == 6 and all(
        item["decode_equivalent_within_10pct"] for item in rows
    )
    summary = {
        "schema": "eurosys_e13_tier_steady_phase_verification_v1",
        "status": "ok" if execution_ok else "error",
        "single_round_no_repeats": True,
        "expected_case_count": 6,
        "observed_case_count": len(cases),
        "equivalence_band": {
            "metric": "switched latency / direct latency",
            "lower": EQUIVALENCE_LOWER,
            "upper": EQUIVALENCE_UPPER,
        },
        "all_prefill_cells_equivalent_within_10pct": all_prefill_equivalent,
        "all_decode_cells_equivalent_within_10pct": all_decode_equivalent,
        "all_phase_cells_equivalent_within_10pct": (
            all_prefill_equivalent and all_decode_equivalent
        ),
        "rows": rows,
        "checks": checks,
    }
    output = args.output or (root / "verification.json")
    atomic_text(output, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    markdown = [
        "# E13 switched-tier vs direct-tier single-round phase result",
        "",
        "Switch transactions and the common target-tier warmup are outside the timing window.",
        "All arms use the same frozen 32-token continuation without logit recording.",
        "",
        "| Tier | B | Direct Prefill ms | Switched Prefill ms | S/D | Direct Decode ms | Switched Decode ms | S/D |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        markdown.append(
            f"| {row['tier'].upper()} | {row['batch_size']} | "
            f"{row['direct_prefill_ms']:.2f} | {row['switched_prefill_ms']:.2f} | "
            f"{row['switched_over_direct_prefill_latency']:.3f}× | "
            f"{row['direct_decode_median_ms']:.2f} | "
            f"{row['switched_decode_median_ms']:.2f} | "
            f"{row['switched_over_direct_decode_latency']:.3f}× |"
        )
    markdown.extend(
        [
            "",
            f"Prefill all cells within ±10%: `{all_prefill_equivalent}`.",
            f"Decode all cells within ±10%: `{all_decode_equivalent}`.",
            "This is a single-round qualification, not a statistical equivalence test.",
            "",
        ]
    )
    atomic_text(root / "summary.md", "\n".join(markdown))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if execution_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
