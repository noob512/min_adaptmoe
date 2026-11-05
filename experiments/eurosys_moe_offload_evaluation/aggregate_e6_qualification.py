#!/usr/bin/env python3
"""Validate and aggregate the three successful E6 qualification arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


CLASSES = ("d005", "d010", "d025", "d050", "d100")
EXPECTED_SEQUENCE = [
    "base", "d005", "base", "d010", "base", "d025", "base",
    "d050", "base", "d100", "base",
]
MECHANISMS = ("full_rebuild", "serial_wave", "batched_difference")
EXPECTED_OUTPUT_SHA256 = (
    "a365db717e87663b5f807b34fc6013b44f98cd925bea9480cece8e65eac690b9"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_leaf(root: Path) -> Path:
    block = root / "block_01"
    return block if (block / "result.json").is_file() else root


def validate_artifact(root: Path, mechanism: str) -> tuple[dict[str, Any], dict[str, Any]]:
    leaf = run_leaf(root)
    result_path = leaf / "result.json"
    verification_path = root / "verification.json"
    result = load(result_path)
    verification = load(verification_path)
    require(result.get("status") == "ok", f"{mechanism}: runner status")
    require(verification.get("status") == "ok", f"{mechanism}: verifier status")
    require(result.get("sequence") == EXPECTED_SEQUENCE, f"{mechanism}: sequence")
    require(result.get("prompt_tokens") == 512, f"{mechanism}: prompt tokens")
    require(result.get("max_tokens") == 8, f"{mechanism}: output tokens")
    require(result.get("cleanup_errors") == [], f"{mechanism}: cleanup errors")
    require(
        result.get("reference_release", {}).get("ok") is True,
        f"{mechanism}: reference release",
    )
    require(
        result.get("output_sha256") == verification.get("output_sha256")
        == EXPECTED_OUTPUT_SHA256,
        f"{mechanism}: output hash",
    )
    records = result.get("records")
    require(isinstance(records, list) and len(records) == 11, f"{mechanism}: records")
    require(
        {record.get("output_sha256") for record in records}
        == {EXPECTED_OUTPUT_SHA256},
        f"{mechanism}: per-request output hash",
    )
    if mechanism == "full_rebuild":
        require(
            result.get("capture_count_after_load") == 2
            and result.get("capture_count_final") == 22,
            "full_rebuild: Graph rebuild count",
        )
        require(result.get("distributed_cleanup") is True, "full_rebuild: process group")
        require(
            result.get("controlled_hard_exit_after_cleanup") is True,
            "full_rebuild: controlled hard-exit contract",
        )
    else:
        require(
            result.get("capture_count_after_load")
            == result.get("capture_count_final") == 2,
            f"{mechanism}: unexpected Graph recapture",
        )
    require((leaf / "returncode.txt").read_text().strip() == "0", f"{mechanism}: rc")

    gpu_after = load(leaf / "gpu.after.json")
    module_after = load(leaf / "module.after.json")
    require(gpu_after.get("gpu_clean") is True, f"{mechanism}: GPU not clean")
    require(gpu_after.get("memory_used_mib") == 0, f"{mechanism}: GPU memory")
    require(gpu_after.get("compute_pids") == [], f"{mechanism}: compute clients")
    require(gpu_after.get("uvm_client_pids") == [], f"{mechanism}: UVM clients")
    require(module_after.get("gpu_clean") is True, f"{mechanism}: module cleanup")
    require(module_after.get("memory_used_mib") == 0, f"{mechanism}: module GPU memory")
    require(module_after.get("uvm_client_pids") == [], f"{mechanism}: module clients")
    require(
        str(module_after.get("module_path", "")).startswith("/usr/lib/modules/"),
        f"{mechanism}: stock UVM was not restored",
    )

    kernel_text = "\n".join(
        (leaf / name).read_text(encoding="utf-8", errors="replace")
        for name in ("kernel.since.log", "kernel.since.err")
    )
    fatal_signatures = (
        "NVRM: Xid", "FAULT_PTE", "FAULT_PDE", "CUDA illegal access",
        "GPU has fallen off the bus", "BUG:", "Kernel panic",
    )
    hits = [signature for signature in fatal_signatures if signature in kernel_text]
    require(not hits, f"{mechanism}: fatal kernel signatures: {hits}")

    return result, {
        "root": str(root),
        "result_sha256": sha256(result_path),
        "verification_sha256": sha256(verification_path),
        "returncode": 0,
        "verifier_status": "ok",
        "artifact_gpu_after_clean": True,
        "artifact_stock_uvm_restored": True,
        "stock_uvm_module_path": str(module_after["module_path"]),
        "stock_uvm_module_sha256": str(module_after["module_sha256"]),
        "fatal_kernel_signatures": [],
    }


def classify_transition(old_class: str, new_class: str) -> tuple[str, str]:
    if old_class == "base" and new_class in CLASSES:
        return new_class, "outbound"
    if new_class == "base" and old_class in CLASSES:
        return old_class, "return"
    raise RuntimeError(f"unexpected transition: {old_class}->{new_class}")


def transition_ms(mechanism: str, record: dict[str, Any]) -> float:
    if mechanism == "full_rebuild":
        rebuild = record.get("rebuild")
        require(isinstance(rebuild, dict) and rebuild.get("ok") is True, "full rebuild")
        value = float(rebuild.get("wall_ns", 0)) / 1e6
    elif mechanism == "serial_wave":
        require(record.get("wave_activations"), "serial wave activations")
        value = float(record.get("switch_ns", 0)) / 1e6
    else:
        activation = record.get("activation")
        switch = activation.get("plan_switch") if isinstance(activation, dict) else None
        require(isinstance(switch, dict) and switch.get("ok") is True, "batched switch")
        value = float(switch.get("wall_ns", 0)) / 1e6
    require(math.isfinite(value) and value > 0, f"{mechanism}: transition timing")
    return value


def summarize_mechanism(
    mechanism: str,
    result: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    records = result["records"]
    summary: dict[str, dict[str, Any]] = {class_name: {} for class_name in CLASSES}
    for previous, current in zip(records, records[1:]):
        class_name, direction = classify_transition(
            str(previous["class_name"]), str(current["class_name"])
        )
        previous_ms = float(previous["request_ns"]) / 1e6
        current_ms = float(current["request_ns"]) / 1e6
        denominator_ms = previous_ms - current_ms
        cost_ms = transition_ms(mechanism, current)

        verified_class = verification["aggregate"][class_name]
        verified = (
            verified_class["directions"][direction]
            if mechanism == "batched_difference"
            else verified_class[direction]
        )
        if mechanism == "full_rebuild":
            verified_ms = float(verified["total_ms"])
        elif mechanism == "serial_wave":
            verified_ms = float(verified["activation_wall_ms"])
        else:
            verified_ms = float(verified["median_ms"])
        require(
            math.isclose(cost_ms, verified_ms, rel_tol=0.0, abs_tol=1e-6),
            f"{mechanism}/{class_name}/{direction}: verifier timing mismatch",
        )

        if denominator_ms > 0:
            requests = cost_ms / denominator_ms
            status = "break_even"
            first_integer_request = math.ceil(requests)
        else:
            requests = None
            status = "no_break_even"
            first_integer_request = None
        item: dict[str, Any] = {
            "old_class": str(previous["class_name"]),
            "new_class": str(current["class_name"]),
            "transition_ms": cost_ms,
            "previous_request_ms": previous_ms,
            "current_request_ms": current_ms,
            "denominator_ms": denominator_ms,
            "break_even_status": status,
            "requests_to_amortize": requests,
            "first_integer_request": first_integer_request,
        }
        if mechanism == "serial_wave":
            item["wave_count"] = len(current["transition_path"])
            item["summed_plan_switch_ms"] = (
                float(current["transition_path_plan_wall_ns"]) / 1e6
            )
        elif mechanism == "full_rebuild":
            item["destroyed_graph_count"] = int(current["rebuild"]["destroyed_graph_count"])
        summary[class_name][direction] = item
    require(
        all(set(directions) == {"outbound", "return"} for directions in summary.values()),
        f"{mechanism}: incomplete direction coverage",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--difference-root", required=True, type=Path)
    parser.add_argument("--full-rebuild-root", required=True, type=Path)
    parser.add_argument("--serial-wave-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    roots = {
        "batched_difference": args.difference_root.resolve(),
        "full_rebuild": args.full_rebuild_root.resolve(),
        "serial_wave": args.serial_wave_root.resolve(),
    }
    results: dict[str, dict[str, Any]] = {}
    verifications: dict[str, dict[str, Any]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for mechanism in MECHANISMS:
        result, source = validate_artifact(roots[mechanism], mechanism)
        results[mechanism] = result
        verifications[mechanism] = load(roots[mechanism] / "verification.json")
        sources[mechanism] = source

    configs = {mechanism: load(root / "run_config.json") for mechanism, root in roots.items()}
    require(
        {config.get("model") for config in configs.values()}
        == {"DeepSeek-V2-Lite-Chat BF16"},
        "model mismatch",
    )
    for field, expected in (
        ("process_blocks", 1),
        ("kv_cache_bytes", 268435456),
        ("slot_budget_bytes", 10737418240),
        ("max_tokens", 8),
    ):
        require(
            {config.get(field) for config in configs.values()} == {expected},
            f"run config mismatch: {field}",
        )

    manifest = load(roots["batched_difference"] / "inputs" / "plan_classes.json")
    class_records = manifest["classes"]
    require(set(CLASSES) < set(class_records), "difference classes missing")
    require(int(manifest["selected_main_bytes"]) == 8546942976, "selected bytes")
    selected_bytes = {
        int(record["selected_bytes"]) for result in results.values()
        for record in result["records"]
    }
    backing_bytes = {
        int(record["backing_bytes"]) for result in results.values()
        for record in result["records"]
    }
    require(selected_bytes == {8546942976}, "cross-arm selected bytes")
    require(backing_bytes == {657457152}, "cross-arm backing bytes")

    mechanisms = {
        mechanism: summarize_mechanism(
            mechanism, results[mechanism], verifications[mechanism]
        )
        for mechanism in MECHANISMS
    }
    comparisons: dict[str, dict[str, Any]] = {}
    for class_name in CLASSES:
        comparisons[class_name] = {}
        for direction in ("outbound", "return"):
            costs = {
                mechanism: mechanisms[mechanism][class_name][direction]["transition_ms"]
                for mechanism in MECHANISMS
            }
            comparisons[class_name][direction] = {
                "transition_ms": costs,
                "batched_vs_full_speedup": (
                    costs["full_rebuild"] / costs["batched_difference"]
                ),
                "batched_vs_serial_speedup": (
                    costs["serial_wave"] / costs["batched_difference"]
                ),
                "serial_vs_full_speedup": (
                    costs["full_rebuild"] / costs["serial_wave"]
                ),
            }

    output = args.output.resolve()
    require(not output.exists(), f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "eurosys_e6_qualification_summary_v1",
        "status": "ok",
        "qualification_scope": "single_process_single_sample_per_transition",
        "output_sha256": EXPECTED_OUTPUT_SHA256,
        "statistical_claims": {
            "median": False,
            "p95": False,
            "confidence_interval": False,
            "note": "Verifier median fields have count=1 and are source checks only.",
        },
        "break_even_formula": (
            "transition_ms / (previous_request_ms - current_request_ms)"
        ),
        "frozen_parameters": {
            "model": "DeepSeek-V2-Lite-Chat BF16",
            "batch": 1,
            "prompt_tokens": 512,
            "output_tokens": 8,
            "kv_cache_bytes": 268435456,
            "slot_budget_bytes": 10737418240,
            "capacity_class": "C8",
            "expert_layers": 26,
            "experts_per_layer": 19,
            "expert_slots": 494,
            "selected_bytes": 8546942976,
            "retained_backing_bytes": 657457152,
            "graph_mode": "FULL_SEPARATE",
            "graph_capture_sizes": [1, 512],
            "seed": 20260828,
            "difference_percentages": [5, 10, 25, 50, 100],
        },
        "classes": {
            class_name: {
                "requested_difference_percent": int(
                    class_records[class_name]["difference_percent"]
                ),
                "actual_difference_percent": float(
                    class_records[class_name]["actual_difference_percent"]
                ),
                "changed_expert_slots": int(
                    class_records[class_name]["changed_expert_slots"]
                ),
            }
            for class_name in CLASSES
        },
        "sources": sources,
        "mechanisms": mechanisms,
        "comparisons": comparisons,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
