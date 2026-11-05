#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


CASES = {
    "fixed_4gb": {
        "engine_case": "static_custom",
        "experts_per_layer": 9,
        "target_blocks": 2528,
    },
    "fixed_7gb": {
        "engine_case": "static_custom",
        "experts_per_layer": 16,
        "target_blocks": 4096,
    },
    "fixed_10gb": {
        "engine_case": "static_custom",
        "experts_per_layer": 23,
        "target_blocks": 5664,
    },
    "adaptive_4_7_10gb": {
        "engine_case": "adaptive",
        "experts_per_layer": 23,
        "target_blocks": 5664,
    },
}
PHASES = ("c64", "c96", "c128", "c160")
AUTO_EXPERTS = {"c64": 9, "c96": 9, "c128": 16, "c160": 23}
EXPERT_EST_BYTES = 17_301_504
MOE_LAYERS = 26


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def external_peak(path: Path) -> int | None:
    if not path.exists():
        return None
    values: list[int] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                values.append(int(row["used_mib"].strip()))
            except (KeyError, TypeError, ValueError):
                continue
    return max(values) if values else None


def offload_bytes(experts_per_layer: int) -> int:
    return experts_per_layer * MOE_LAYERS * EXPERT_EST_BYTES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root

    case_results: dict[str, Any] = {}
    fixture_hashes: dict[str, str | None] = {}
    rows: list[dict[str, Any]] = []

    for case_name, config in CASES.items():
        case_root = root / case_name
        engine_case = str(config["engine_case"])
        result = read_json(case_root / engine_case / "result.json")
        phases = {
            str(phase.get("phase")): phase
            for phase in result.get("phases", [])
            if isinstance(phase, dict)
        }
        plan = read_json(case_root / "hot_cold_plan_k45.json")
        plan_metadata = plan.get("metadata", {})
        fixture = case_root / "sharegpt_256_prompt320_360.json"
        fixture_hashes[case_name] = file_sha256(fixture)

        phase_results: dict[str, Any] = {}
        for phase_name in PHASES:
            phase = phases.get(phase_name, {})
            experts = (
                AUTO_EXPERTS[phase_name]
                if case_name == "adaptive_4_7_10gb"
                else int(config["experts_per_layer"])
            )
            record = {
                "phase": phase_name,
                "concurrency": int(phase_name[1:]),
                "experts_per_layer_offloaded": experts,
                "offload_bytes": offload_bytes(experts),
                "offload_gb_decimal": offload_bytes(experts) / 1e9,
                "offload_gib": offload_bytes(experts) / 1024**3,
                "elapsed_seconds": phase.get("elapsed_seconds"),
                "output_tokens": phase.get("output_tokens"),
                "output_tokens_per_second": phase.get(
                    "output_tokens_per_second"
                ),
                "mean_ttft_ms": phase.get("mean_ttft_ms"),
                "p99_ttft_ms": phase.get("p99_ttft_ms"),
                "mean_tpot_ms": phase.get("mean_tpot_ms"),
                "p99_tpot_ms": phase.get("p99_tpot_ms"),
                "completed_requests": phase.get("completed_requests"),
                "gpu_after_phase_mib": (
                    phase.get("gpu_after_phase") or {}
                ).get("used_mib"),
            }
            phase_results[phase_name] = record
            rows.append({"case": case_name, **record})

        maintenance = result.get("maintenance", [])
        adjustments = []
        for after_phase, label in (
            (0, "initial_to_4gb"),
            (3, "scale_4gb_to_7gb"),
            (4, "scale_7gb_to_10gb"),
        ):
            records = [
                item
                for item in maintenance
                if isinstance(item, dict)
                and item.get("after_phase") == after_phase
            ]
            if records:
                adjustments.append(
                    {
                        "label": label,
                        "after_phase_index": after_phase,
                        "waves": len(records),
                        "elapsed_seconds": sum(
                            float(item.get("elapsed_seconds") or 0.0)
                            for item in records
                        ),
                        "first_target_blocks": records[0].get("target_blocks"),
                        "final_target_blocks": records[-1].get("target_blocks"),
                        "all_ok": all(
                            bool((item.get("result") or {}).get("ok"))
                            for item in records
                        ),
                    }
                )

        case_results[case_name] = {
            "status": result.get("status"),
            "error": result.get("error"),
            "load_seconds": result.get("load_seconds"),
            "external_peak_gpu_used_mib": external_peak(case_root / "gpu.csv"),
            "returncode": (
                (case_root / "launcher_returncode.txt").read_text().strip()
                if (case_root / "launcher_returncode.txt").exists()
                else None
            ),
            "configured_experts_per_layer": config["experts_per_layer"],
            "configured_target_blocks": config["target_blocks"],
            "plan_estimated_cold_bytes": plan_metadata.get(
                "estimated_cold_bytes"
            ),
            "plan_cold_experts_per_layer": plan_metadata.get(
                "cold_experts_per_layer"
            ),
            "phases": phase_results,
            "adjustments": adjustments,
            "result_path": str(case_root / engine_case / "result.json"),
        }

    equivalent_comparison: dict[str, Any] = {}
    fixed_by_auto_phase = {
        "c64": "fixed_4gb",
        "c96": "fixed_4gb",
        "c128": "fixed_7gb",
        "c160": "fixed_10gb",
    }
    auto_phases = case_results["adaptive_4_7_10gb"]["phases"]
    for phase_name, fixed_case in fixed_by_auto_phase.items():
        fixed_tps = case_results[fixed_case]["phases"][phase_name].get(
            "output_tokens_per_second"
        )
        auto_tps = auto_phases[phase_name].get("output_tokens_per_second")
        equivalent_comparison[phase_name] = {
            "concurrency": int(phase_name[1:]),
            "fixed_case": fixed_case,
            "fixed_tps": fixed_tps,
            "adaptive_tps": auto_tps,
            "adaptive_over_fixed_ratio": (
                auto_tps / fixed_tps if fixed_tps and auto_tps else None
            ),
            "adaptive_vs_fixed_percent": (
                (auto_tps / fixed_tps - 1.0) * 100.0
                if fixed_tps and auto_tps
                else None
            ),
        }

    valid_hashes = [value for value in fixture_hashes.values() if value]
    summary = {
        "run_root": str(root),
        "scope": {
            "model": "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "request_counts": [64, 96, 128, 160],
            "output_tokens_per_request": 32,
            "prompt_token_range": "320-360",
            "adjustment_time_included_in_phase_metrics": False,
            "fixed_offload_experts_per_layer": [9, 16, 23],
            "fixed_offload_bytes": [
                offload_bytes(value) for value in (9, 16, 23)
            ],
            "adaptive_schedule_experts_per_layer": {
                "64": 9,
                "96": 9,
                "128": 16,
                "160": 23,
            },
            "kv_target_blocks": {
                "4gb": 2528,
                "7gb": 4096,
                "10gb": 5664,
            },
        },
        "fixture_hashes": fixture_hashes,
        "fixtures_identical": bool(valid_hashes)
        and len(set(valid_hashes)) == 1
        and len(valid_hashes) == len(CASES),
        "cases": case_results,
        "adaptive_vs_equivalent_fixed": equivalent_comparison,
    }
    (root / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (root / "phase_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report_lines = [
        "# DeepSeek-V2-Lite 卸载量与自动扩缩容实验",
        "",
        "## 吞吐结果",
        "",
        "| 并发 | 固定 4GB | 固定 7GB | 固定 10GB | 自动 4→7→10GB |",
        "|---:|---:|---:|---:|---:|",
    ]
    for phase_name in PHASES:
        values = []
        for case_name in (
            "fixed_4gb",
            "fixed_7gb",
            "fixed_10gb",
            "adaptive_4_7_10gb",
        ):
            value = case_results[case_name]["phases"][phase_name].get(
                "output_tokens_per_second"
            )
            values.append(f"{value:.4f}" if isinstance(value, (int, float)) else "N/A")
        report_lines.append(
            f"| {int(phase_name[1:])} | " + " | ".join(values) + " |"
        )
    report_lines.extend(
        [
            "",
            "## 自动扩缩容与同卸载量固定方案",
            "",
            "| 并发 | 对应固定方案 | 固定吞吐 | 自动吞吐 | 自动相对差异 |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for phase_name in PHASES:
        item = equivalent_comparison[phase_name]
        percent = item.get("adaptive_vs_fixed_percent")
        report_lines.append(
            "| {concurrency} | {fixed_case} | {fixed:.4f} | {adaptive:.4f} | {percent:+.2f}% |".format(
                concurrency=item["concurrency"],
                fixed_case=item["fixed_case"],
                fixed=item["fixed_tps"] or 0.0,
                adaptive=item["adaptive_tps"] or 0.0,
                percent=percent or 0.0,
            )
        )
    adaptive_adjustments = case_results["adaptive_4_7_10gb"]["adjustments"]
    report_lines.extend(
        [
            "",
            "## 调整耗时（未计入吞吐）",
            "",
            "| 调整 | waves | 目标 blocks | 调整时间 | 成功 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for item in adaptive_adjustments:
        report_lines.append(
            f"| {item['label']} | {item['waves']} | {item['final_target_blocks']} | "
            f"{item['elapsed_seconds']:.4f} s | {item['all_ok']} |"
        )
    report_lines.extend(
        [
            "",
            "## 校验",
            "",
            f"- 四个 case 的 fixture 完全一致：`{summary['fixtures_identical']}`",
            "- 主阶段只统计 `llm.generate()`，模型加载、预热和扩缩容维护时间均未计入。",
            "- 4/7/10GB 为按专家估算的十进制近似值，实际为 4.049/7.197/10.346 GB。",
            "",
        ]
    )
    (root / "RESULTS.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
