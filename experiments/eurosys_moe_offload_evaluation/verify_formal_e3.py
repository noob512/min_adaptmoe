#!/usr/bin/env python3
"""Verify and summarize a frozen exact-coordinate Track-A E3 experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_SELECTED_BYTES = 2_264_924_160
EXPECTED_SELECTED_EXPERTS = 480
EXPECTED_PARAMETER_COUNT = 96
EXPECTED_COORDINATE_SHA256 = (
    "c8754744311aebcdc1a5d768c849a346b33c51027ce3d0998b225e0107dde898"
)
EXPECTED_PLAN_SHA256 = (
    "69d625693e7bd31ce00b9847bcfa0c80ccb487e5c0da03f1c985276ebf1f99af"
)
EXPECTED_EXACT_NAMES_SHA256 = (
    "6142b29ba13671bd2c0c1e18525a64e1bd3f676ad369add9d3d2323086ac4141"
)
EXPECTED_COLD_EXPERTS_PER_LAYER = 10
EXPECTED_KT_GPU_EXPERTS = 118
EXPECTED_MOE_LAYERS = 48
VERIFICATION_SCHEMA = "eurosys_qwen_e3_verification_v1"
PROFILES = {
    "qwen3": {
        "selected_bytes": 2_264_924_160,
        "selected_experts": 480,
        "parameter_count": 96,
        "coordinate_sha256": "c8754744311aebcdc1a5d768c849a346b33c51027ce3d0998b225e0107dde898",
        "plan_sha256": "69d625693e7bd31ce00b9847bcfa0c80ccb487e5c0da03f1c985276ebf1f99af",
        "exact_names_sha256": "6142b29ba13671bd2c0c1e18525a64e1bd3f676ad369add9d3d2323086ac4141",
        "cold_experts_per_layer": 10,
        "kt_gpu_experts": 118,
        "moe_layers": 48,
        "schema": "eurosys_qwen_e3_verification_v1",
    },
    "deepseek_v2_lite": {
        "selected_bytes": 2_249_195_520,
        "selected_experts": 130,
        "parameter_count": 52,
        "coordinate_sha256": "6e69cd376842366eb29c147b131789ff6416c1e9cf1df14575be8740c7284dba",
        "plan_sha256": "282170b4b8f7cb0ec81938f91accdfa0396705774534f2569499202369611ad1",
        "exact_names_sha256": "e23f367bb3d9855cbcf55d5a5d70a30ba5f884bcf6fffb76c9c5da421ae493b0",
        "cold_experts_per_layer": 5,
        "kt_gpu_experts": 59,
        "moe_layers": 26,
        "schema": "eurosys_deepseek_v2_lite_e3_verification_v1",
    },
}
ARMS = ("H", "A", "R")
ARM_NAMES = {
    "H": "hybrid_prefill_copy_decode_remote",
    "A": "always_copy",
    "R": "always_remote",
}
WORKLOADS = {
    "short_dialogue": (128, 32),
    "ordinary_chat": (512, 128),
    "medium_context": (2048, 128),
    "rag_summary": (8192, 128),
    "long_generation": (512, 512),
}
BATCHES = (1, 4, 8)
ACTIVE_BATCHES = BATCHES


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_coordinate_hash(expert_plan: list[Any]) -> str:
    by_layer = {str(int(layer)): [int(value) for value in experts]
                for layer, experts in expert_plan}
    raw = json.dumps(by_layer, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def nvml_peak_mib(path: Path) -> float | None:
    values: list[float] = []
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) > 1:
            try:
                values.append(float(fields[1]))
            except ValueError:
                pass
    return max(values) if values else None


def preemption_count(result: dict[str, Any]) -> int:
    steps = result["steps"]
    records = [steps["prefill"], steps["first_decode"]]
    records.extend(steps["warmup_decode"])
    records.extend(steps["measured_decode"])
    return sum(len(record["schedule"].get("preempted_request_ids", []))
               for record in records)


def check_custom_case(
    root: Path, row: dict[str, str], plan: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    case = root / "custom" / row["case_dir"]
    status = row["status"]
    if status == "N/A":
        na_path = case / "na.json"
        if not na_path.is_file():
            errors.append("missing na.json")
        else:
            na = load_json(na_path)
            if na.get("status") != "N/A" or not na.get("reason"):
                errors.append("invalid N/A record")
        if (case / "result.json").exists():
            errors.append("N/A case unexpectedly has result.json")
        return None, errors
    if status != "ok":
        return None, [f"execution status is {status!r}"]
    result_path = case / "result.json"
    if not result_path.is_file():
        return None, ["missing result.json"]
    result = load_json(result_path)
    arm = row["arm"]
    batch = int(row["batch"])
    prompt = int(row["prompt"])
    output = int(row["output"])
    model = result.get("diagnostics", {}).get("after_first_decode", {}).get("model", {})
    shutdown = result.get("shutdown_contract", {}).get(
        "diagnostics_after_shutdown", {}
    ).get("driver", {})
    outputs = result.get("outputs", [])

    if result.get("status") != "ok" or result.get("arm") != arm:
        errors.append("result status/arm mismatch")
    if len(outputs) != batch or any(
        len(item.get("output_token_ids", [])) != output for item in outputs
    ):
        errors.append("output batch or length mismatch")
    if result.get("fixture", {}).get("metadata", {}).get("prompt_tokens") != prompt:
        errors.append("fixture prompt length mismatch")
    if result.get("fixture", {}).get("metadata", {}).get("output_tokens") != output:
        errors.append("fixture output length mismatch")
    if result.get("exact_names_sha256") != EXPECTED_EXACT_NAMES_SHA256:
        errors.append("exact-name hash mismatch")
    if model.get("selected_expert_count") != EXPECTED_SELECTED_EXPERTS:
        errors.append("selected expert count mismatch")
    if model.get("selected_parameter_count") != EXPECTED_PARAMETER_COUNT:
        errors.append("selected parameter count mismatch")
    if model.get("selected_parameter_bytes_per_forward") != EXPECTED_SELECTED_BYTES:
        errors.append("selected bytes mismatch")
    if canonical_coordinate_hash(model.get("expert_plan", [])) != EXPECTED_COORDINATE_SHA256:
        errors.append("runtime coordinate plan mismatch")
    if model.get("expert_plan") != [
        [int(layer), experts]
        for layer, experts in sorted(
            plan["selected_by_layer"].items(), key=lambda item: int(item[0])
        )
    ]:
        errors.append("runtime coordinates differ from frozen plan")
    cold = model.get("cold_residency_result") or {}
    hot = model.get("hot_residency_result") or {}
    if not cold.get("ok") or cold.get("cpu_bytes") != EXPECTED_SELECTED_BYTES:
        errors.append("cold CPU residency incomplete")
    if cold.get("unknown_bytes") != 0 or hot.get("unknown_bytes") != 0:
        errors.append("UNKNOWN residency is non-zero")
    if shutdown.get("session_count") != 0:
        errors.append("driver session leaked at shutdown")
    if result.get("shutdown_contract", {}).get("cleanup_errors"):
        errors.append("shutdown cleanup errors present")
    if preemption_count(result) != 0:
        errors.append("scheduler preemption observed")

    transition = result.get("cutover_transition")
    remote = model.get("remote_map_result") or {}
    if arm == "H":
        if result.get("config", {}).get("phase_mode") != "prefill_remote_decode":
            errors.append("hybrid phase mode mismatch")
        if not transition or not transition.get("ok"):
            errors.append("hybrid cutover missing/failed")
        else:
            residence = transition.get("physical_residency") or {}
            if (residence.get("cpu_bytes") != EXPECTED_SELECTED_BYTES
                    or residence.get("unknown_bytes") != 0):
                errors.append("hybrid cutover residency mismatch")
        if not model.get("remote_map_active") or not remote.get("ok"):
            errors.append("hybrid Decode remote map inactive")
    elif arm == "A":
        if result.get("config", {}).get("phase_mode") != "off":
            errors.append("always-copy phase mode mismatch")
        if transition is not None or not model.get("sealed"):
            errors.append("always-copy is not sealed alias mode")
        if model.get("remote_map_active"):
            errors.append("always-copy unexpectedly remote-mapped")
    elif arm == "R":
        if result.get("config", {}).get("phase_mode") != "remote_only":
            errors.append("always-remote phase mode mismatch")
        if transition is not None:
            errors.append("always-remote unexpectedly cut over")
        if not model.get("remote_map_active") or not remote.get("ok"):
            errors.append("always-remote map inactive")
    if arm in ("H", "R"):
        if (remote.get("covered_bytes") != EXPECTED_SELECTED_BYTES
                or remote.get("remote_mapped_bytes") != EXPECTED_SELECTED_BYTES
                or remote.get("unknown_bytes") != 0):
            errors.append("remote-map byte coverage mismatch")

    timing = result.get("timing", {})
    metric = {
        "block": int(row["block"]),
        "workload": row["workload"],
        "prompt_tokens": prompt,
        "output_tokens": output,
        "batch": batch,
        "arm": arm,
        "arm_name": ARM_NAMES[arm],
        "output_token_ids_sha256": result.get("output_token_ids_sha256"),
        "prefill_ms": timing.get("prefill_step_ms"),
        "cutover_ms": timing.get("cutover_wall_ms"),
        "first_decode_excluding_cutover_ms": timing.get(
            "first_decode_excluding_cutover_ms"
        ),
        "steady_decode_median_ms": timing.get("steady_decode_step_ms", {}).get("median"),
        "steady_decode_mean_ms": timing.get("steady_decode_step_ms", {}).get("mean"),
        "steady_decode_p90_ms": timing.get("steady_decode_step_ms", {}).get("p90"),
        "steady_decode_tokens_per_second": timing.get("steady_decode_tokens_per_second"),
        "cycle_e2e_ms": timing.get("cycle_e2e_ms"),
        "load_seconds": result.get("load_seconds"),
        "peak_hbm_mib": nvml_peak_mib(case / "nvml.csv"),
        "selected_bytes": EXPECTED_SELECTED_BYTES,
        "coordinate_sha256": EXPECTED_COORDINATE_SHA256,
        "audit_status": "ok" if not errors else "error",
    }
    for key in (
        "prefill_ms", "cutover_ms", "first_decode_excluding_cutover_ms",
        "steady_decode_median_ms", "steady_decode_mean_ms", "cycle_e2e_ms",
    ):
        value = metric[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            errors.append(f"invalid timing metric {key}")
    metric["audit_status"] = "ok" if not errors else "error"
    return metric, errors


def extrapolated_e2e(row: dict[str, Any], output_tokens: int) -> float:
    """Locally extrapolate E2E from the observed point using Decode median."""
    return (
        float(row["cycle_e2e_ms"])
        + (output_tokens - int(row["output_tokens"]))
        * float(row["steady_decode_median_ms"])
    )


def pairwise_break_even(
    cell: tuple[str, int], left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any]:
    # Anchor fixed costs to the measured E2E point, then vary only Decode length.
    li, ri = float(left["cycle_e2e_ms"]), float(right["cycle_e2e_ms"])
    ld, rd = float(left["steady_decode_median_ms"]), float(right["steady_decode_median_ms"])
    anchor_output = int(left["output_tokens"])
    denominator = rd - ld
    if abs(denominator) < 1e-12:
        threshold: float | None = None
        meaning = f"{left['arm']} always" if li <= ri else f"{right['arm']} always"
    else:
        threshold = anchor_output + (li - ri) / denominator
        if denominator > 0:
            meaning = (
                f"{left['arm']} wins for output_tokens >= {max(1, math.ceil(threshold))}"
                if threshold > 1 else f"{left['arm']} wins for all output lengths"
            )
        else:
            meaning = (
                f"{right['arm']} wins for output_tokens >= {max(1, math.ceil(threshold))}"
                if threshold > 1 else f"{right['arm']} wins for all output lengths"
            )
    return {
        "workload": cell[0],
        "batch": cell[1],
        "prompt_tokens": left["prompt_tokens"],
        "left_arm": left["arm"],
        "right_arm": right["arm"],
        "anchor_output_tokens": anchor_output,
        "left_anchor_e2e_ms": li,
        "right_anchor_e2e_ms": ri,
        "left_decode_median_ms": ld,
        "right_decode_median_ms": rd,
        "break_even_output_tokens": threshold,
        "interpretation": meaning,
        "model": (
            "measured_cycle_e2e_ms + (output_tokens-anchor_output_tokens)"
            "*steady_decode_median_ms"
        ),
    }


def strategy_regions(cell_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    # Enumerate an ample output range, then coalesce exact integer winner regions.
    winners: list[str] = []
    for output in range(1, 4097):
        winners.append(min(
            ARMS,
            key=lambda arm: extrapolated_e2e(cell_rows[arm], output),
        ))
    regions: list[dict[str, Any]] = []
    start = 1
    for output in range(2, 4098):
        changed = output == 4097 or winners[output - 1] != winners[start - 1]
        if changed:
            arm = winners[start - 1]
            sample = cell_rows[arm]
            regions.append({
                "workload": sample["workload"],
                "prompt_tokens": sample["prompt_tokens"],
                "batch": sample["batch"],
                "output_start_inclusive": start,
                "output_end_inclusive": 4096 if output == 4097 else output - 1,
                "best_arm": arm,
                "best_arm_name": ARM_NAMES[arm],
                "model": "measured-E2E-anchored affine model; descriptive single-block estimate",
            })
            start = output
    return regions


def parse_kt(root: Path, rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[str]]:
    metrics: list[dict[str, Any]] = []
    errors: list[str] = []
    for row in rows:
        case = root / "kt" / row["case_dir"]
        prefix = row["case_dir"]
        if row["status"] != "ok":
            errors.append(f"{prefix}: KT execution status is {row['status']!r}")
            continue
        result_path = case / "result.json"
        if not result_path.is_file():
            errors.append(f"{prefix}: missing result.json")
            continue
        result = load_json(result_path)
        expected_output = int(row["output"])
        if result.get("status") != "ok":
            errors.append(f"{prefix}: result not ok")
            continue
        fixture_meta = result.get("fixture", {}).get("metadata", {})
        if (fixture_meta.get("prompt_tokens") != int(row["prompt"])
                or fixture_meta.get("output_tokens") != expected_output):
            errors.append(f"{prefix}: frozen fixture lengths differ")
        log = (case / "server.log").read_text(encoding="utf-8", errors="replace")
        audit_marker = f"has {EXPECTED_KT_GPU_EXPERTS} GPU experts"
        if log.count(audit_marker) != EXPECTED_MOE_LAYERS:
            errors.append(
                f"{prefix}: exact-coordinate {EXPECTED_KT_GPU_EXPERTS}-hot "
                f"audit is not {EXPECTED_MOE_LAYERS}/{EXPECTED_MOE_LAYERS}"
            )
        measurement = [cycle for cycle in result.get("cycles", [])
                       if cycle.get("role") == "measurement"]
        expected_batches = [int(value) for value in row["batches"].split(",")]
        if sorted(cycle.get("batch_size") for cycle in measurement) != expected_batches:
            errors.append(f"{prefix}: KT measurement batches differ")
        all_cycles = result.get("cycles", [])
        for batch in expected_batches:
            hashes = {
                cycle.get("output_token_ids_sha256")
                for cycle in all_cycles if cycle.get("batch_size") == batch
            }
            if len(hashes) != 1 or None in hashes:
                errors.append(
                    f"{prefix}/B{batch}: conditioning/measurement output hash differs"
                )
        for cycle in measurement:
            ids = cycle.get("output_token_ids", [])
            batch = int(cycle["batch_size"])
            if len(ids) != batch or any(len(item) != expected_output for item in ids):
                errors.append(f"{prefix}/B{batch}: output length mismatch")
            metrics.append({
                "block": int(row["block"]),
                "workload": row["workload"],
                "prompt_tokens": int(row["prompt"]),
                "output_tokens": expected_output,
                "batch": batch,
                "client_ttft_batch_ms": cycle.get("client_ttft_batch_ms"),
                "server_prefill_forward_median_ms": (
                    (cycle.get("server_prefill_forward_ms") or {}).get("median")
                ),
                "steady_decode_median_ms": cycle.get("measured_decode_step_ms", {}).get("median"),
                "steady_decode_mean_ms": cycle.get("measured_decode_step_ms", {}).get("mean"),
                "decode_tokens_per_second": cycle.get("decode_tokens_per_second"),
                "cycle_e2e_ms": cycle.get("cycle_client_wall_ms"),
                "output_token_ids_sha256": cycle.get("output_token_ids_sha256"),
                "peak_hbm_mib": nvml_peak_mib(case / "nvml.csv"),
                "gpu_experts_per_layer": EXPECTED_KT_GPU_EXPERTS,
                "cpu_experts_per_layer": EXPECTED_COLD_EXPERTS_PER_LAYER,
                "coordinate_sha256": EXPECTED_COORDINATE_SHA256,
                "audit_status": "ok",
            })
    return metrics, errors


def parse_kt_direct(
    root: Path, rows: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Verify no-HTTP, one-generate-call KTransformers B1/B8 cells."""

    metrics: list[dict[str, Any]] = []
    errors: list[str] = []
    for row in rows:
        case = root / "kt" / row["case_dir"]
        prefix = row["case_dir"]
        if row["status"] != "ok":
            errors.append(f"{prefix}: KT direct execution status is {row['status']!r}")
            continue
        result_path = case / "result.json"
        if not result_path.is_file():
            errors.append(f"{prefix}: missing direct result.json")
            continue
        result = load_json(result_path)
        config = result.get("configuration", {})
        if result.get("status") != "ok":
            errors.append(f"{prefix}: direct result not ok")
            continue
        if (
            config.get("transport") != "none"
            or config.get("http_server_started") is not False
            or config.get("client_used") is not False
        ):
            errors.append(f"{prefix}: direct transport contract failed")
        if config.get("mixed_chunk_enabled") is not False:
            errors.append(f"{prefix}: mixed chunk is not disabled")
        if config.get("prefill_covers_full_batch") is not True:
            errors.append(f"{prefix}: prefill budget does not cover full batch")
        fixture_meta = result.get("fixture", {}).get("metadata", {})
        expected_output = int(row["output"])
        if (
            fixture_meta.get("prompt_tokens") != int(row["prompt"])
            or fixture_meta.get("output_tokens") != expected_output
        ):
            errors.append(f"{prefix}: frozen fixture lengths differ")
        log = (case / "run.log").read_text(encoding="utf-8", errors="replace")
        if any(marker in log for marker in ("Uvicorn running", "POST /generate", "http://127.0.0.1")):
            errors.append(f"{prefix}: HTTP/server marker observed")
        audit_marker = f"has {EXPECTED_KT_GPU_EXPERTS} GPU experts"
        if log.count(audit_marker) != EXPECTED_MOE_LAYERS:
            errors.append(
                f"{prefix}: exact-coordinate {EXPECTED_KT_GPU_EXPERTS}-hot "
                f"audit is not {EXPECTED_MOE_LAYERS}/{EXPECTED_MOE_LAYERS}"
            )
        expected_batches = [int(value) for value in row["batches"].split(",")]
        cells = result.get("cells", [])
        if sorted(cell.get("batch_size") for cell in cells) != sorted(expected_batches):
            errors.append(f"{prefix}: KT direct measurement batches differ")
        for cell in cells:
            batch = int(cell.get("batch_size", 0))
            ids = cell.get("output_token_ids", [])
            if (
                batch not in expected_batches
                or cell.get("single_generate_call") is not True
                or cell.get("all_requests_submitted_together") is not True
                or cell.get("transport") != "none"
                or cell.get("mixed_chunk_enabled") is not False
            ):
                errors.append(f"{prefix}/B{batch}: direct batch contract failed")
            if (
                cell.get("successful_requests") != batch
                or len(ids) != batch
                or any(len(item) != expected_output for item in ids)
            ):
                errors.append(f"{prefix}/B{batch}: output length/success mismatch")
            kernel = (case / "kernel.since.log").read_text(
                encoding="utf-8", errors="replace"
            ).lower() if (case / "kernel.since.log").is_file() else ""
            if "nvrm: xid" in kernel or "fault_pde" in kernel:
                errors.append(f"{prefix}/B{batch}: kernel fault observed")
            metrics.append({
                "block": int(row["block"]),
                "workload": row["workload"],
                "prompt_tokens": int(row["prompt"]),
                "output_tokens": expected_output,
                "batch": batch,
                "client_ttft_batch_ms": cell.get("client_ttft_batch_ms"),
                "server_prefill_forward_median_ms": None,
                "steady_decode_median_ms": cell.get("steady_decode_median_ms"),
                "steady_decode_mean_ms": cell.get("steady_decode_mean_ms"),
                "decode_tokens_per_second": cell.get("decode_tokens_per_second"),
                "cycle_e2e_ms": cell.get("cycle_e2e_ms"),
                "output_token_ids_sha256": cell.get("output_token_ids_sha256"),
                "peak_hbm_mib": nvml_peak_mib(case / "nvml.csv"),
                "gpu_experts_per_layer": EXPECTED_KT_GPU_EXPERTS,
                "cpu_experts_per_layer": EXPECTED_COLD_EXPERTS_PER_LAYER,
                "coordinate_sha256": EXPECTED_COORDINATE_SHA256,
                "audit_status": "ok",
                "transport": "none",
            })
    return metrics, errors


def main() -> int:
    global EXPECTED_SELECTED_BYTES
    global EXPECTED_SELECTED_EXPERTS
    global EXPECTED_PARAMETER_COUNT
    global EXPECTED_COORDINATE_SHA256
    global EXPECTED_PLAN_SHA256
    global EXPECTED_EXACT_NAMES_SHA256
    global EXPECTED_COLD_EXPERTS_PER_LAYER
    global EXPECTED_KT_GPU_EXPERTS
    global EXPECTED_MOE_LAYERS
    global VERIFICATION_SCHEMA
    global ACTIVE_BATCHES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--profile", choices=sorted(PROFILES), default="qwen3",
        help="Frozen model/plan contract to verify (default: qwen3)",
    )
    parser.add_argument(
        "--batches", default="1,4,8",
        help="Comma-separated subset of the frozen B1/B4/B8 matrix",
    )
    parser.add_argument(
        "--kt-mode", choices=("legacy", "direct"), default="legacy",
        help="KTransformers result transport/schema",
    )
    args = parser.parse_args()
    try:
        parsed_batches = tuple(int(value) for value in args.batches.split(","))
    except ValueError as exc:
        raise SystemExit("--batches must contain integers") from exc
    if (
        not parsed_batches
        or len(parsed_batches) != len(set(parsed_batches))
        or any(value not in BATCHES for value in parsed_batches)
    ):
        raise SystemExit("--batches must be a unique non-empty subset of 1,4,8")
    ACTIVE_BATCHES = parsed_batches
    profile = PROFILES[args.profile]
    EXPECTED_SELECTED_BYTES = profile["selected_bytes"]
    EXPECTED_SELECTED_EXPERTS = profile["selected_experts"]
    EXPECTED_PARAMETER_COUNT = profile["parameter_count"]
    EXPECTED_COORDINATE_SHA256 = profile["coordinate_sha256"]
    EXPECTED_PLAN_SHA256 = profile["plan_sha256"]
    EXPECTED_EXACT_NAMES_SHA256 = profile["exact_names_sha256"]
    EXPECTED_COLD_EXPERTS_PER_LAYER = profile["cold_experts_per_layer"]
    EXPECTED_KT_GPU_EXPERTS = profile["kt_gpu_experts"]
    EXPECTED_MOE_LAYERS = profile["moe_layers"]
    VERIFICATION_SCHEMA = profile["schema"]
    root = args.run_root.resolve()
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    errors: list[str] = []

    manifest = load_json(root / "provenance/calibration_manifest.json")
    plan_path = root / "provenance/cold_decode_plan_2g.json"
    plan = load_json(plan_path)
    if sha256_file(plan_path) != EXPECTED_PLAN_SHA256:
        errors.append("frozen plan SHA-256 mismatch")
    manifest_plan_sha = (
        manifest.get("plan_sha256")
        or manifest.get("tiers", {}).get("2g", {}).get("plan_sha256")
    )
    if manifest_plan_sha != EXPECTED_PLAN_SHA256:
        errors.append("calibration manifest plan SHA-256 mismatch")
    manifest_coordinate_sha = manifest.get("metadata", {}).get(
        "selected_by_layer_canonical_sha256"
    )
    if (manifest_coordinate_sha is not None
            and manifest_coordinate_sha != EXPECTED_COORDINATE_SHA256):
        errors.append("calibration coordinate SHA-256 mismatch")
    if any(
        len(values) != EXPECTED_COLD_EXPERTS_PER_LAYER
        for values in plan.get("selected_by_layer", {}).values()
    ):
        errors.append(
            "frozen plan is not "
            f"{EXPECTED_COLD_EXPERTS_PER_LAYER} cold experts per layer"
        )

    custom_rows = read_tsv(root / "custom/execution.tsv")
    blocks = sorted({int(row["block"]) for row in custom_rows})
    expected_rows = len(blocks) * len(WORKLOADS) * len(ACTIVE_BATCHES) * len(ARMS)
    if len(custom_rows) != expected_rows:
        errors.append(f"custom execution rows {len(custom_rows)} != {expected_rows}")
    if len({row["case_dir"] for row in custom_rows}) != len(custom_rows):
        errors.append("duplicate custom case labels")

    custom_metrics: list[dict[str, Any]] = []
    na_rows: list[dict[str, Any]] = []
    case_errors: dict[str, list[str]] = {}
    for row in custom_rows:
        metric, found = check_custom_case(root, row, plan)
        if found:
            case_errors[row["case_dir"]] = found
        if metric:
            custom_metrics.append(metric)
        else:
            case = root / "custom" / row["case_dir"]
            reason = load_json(case / "na.json").get("reason") if (case / "na.json").is_file() else None
            na_rows.append({**row, "reason": reason})
    if case_errors:
        errors.append(f"{len(case_errors)} custom cases failed detailed audit")

    by_cell: dict[tuple[int, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in custom_metrics:
        by_cell[(row["block"], row["workload"], row["batch"])][row["arm"]] = row
    hash_checks: dict[str, Any] = {}
    for cell, arms in by_cell.items():
        if set(arms) != set(ARMS):
            errors.append(f"incomplete successful arm set for {cell}")
            continue
        hashes = {item["output_token_ids_sha256"] for item in arms.values()}
        hash_checks[str(cell)] = {"hashes": sorted(hashes), "equal": len(hashes) == 1}
        if len(hashes) != 1:
            errors.append(f"H/A/R output hash mismatch for {cell}")

    heatmap: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    regions: list[dict[str, Any]] = []
    for workload, (prompt, output) in WORKLOADS.items():
        for batch in ACTIVE_BATCHES:
            arms = by_cell.get((blocks[0], workload, batch), {}) if blocks else {}
            if set(arms) == set(ARMS):
                best = min(ARMS, key=lambda arm: arms[arm]["cycle_e2e_ms"])
                heatmap.append({
                    "workload": workload, "prompt_tokens": prompt,
                    "output_tokens": output, "batch": batch,
                    "best_arm": best, "best_arm_name": ARM_NAMES[best],
                    "best_cycle_e2e_ms": arms[best]["cycle_e2e_ms"],
                    "hybrid_cycle_e2e_ms": arms["H"]["cycle_e2e_ms"],
                    "always_copy_cycle_e2e_ms": arms["A"]["cycle_e2e_ms"],
                    "always_remote_cycle_e2e_ms": arms["R"]["cycle_e2e_ms"],
                    "status": "ok",
                })
                for left, right in (("H", "A"), ("H", "R"), ("A", "R")):
                    pairwise.append(pairwise_break_even((workload, batch), arms[left], arms[right]))
                regions.extend(strategy_regions(arms))
            else:
                reason_rows = [item for item in na_rows
                               if item["workload"] == workload and int(item["batch"]) == batch]
                heatmap.append({
                    "workload": workload, "prompt_tokens": prompt,
                    "output_tokens": output, "batch": batch,
                    "best_arm": "N/A", "best_arm_name": "N/A",
                    "best_cycle_e2e_ms": None,
                    "hybrid_cycle_e2e_ms": None,
                    "always_copy_cycle_e2e_ms": None,
                    "always_remote_cycle_e2e_ms": None,
                    "status": "N/A",
                    "reason": reason_rows[0].get("reason") if reason_rows else "incomplete cell",
                })

    kt_rows = read_tsv(root / "kt/execution.tsv")
    if len(kt_rows) != len(blocks) * len(WORKLOADS):
        errors.append(f"KT execution rows {len(kt_rows)} != {len(blocks) * len(WORKLOADS)}")
    kt_metrics, kt_errors = (
        parse_kt_direct(root, kt_rows)
        if args.kt_mode == "direct"
        else parse_kt(root, kt_rows)
    )
    errors.extend(kt_errors)

    native = load_json(root / "capability/vllm_native.json")
    if native.get("pinned_uva", {}).get("status") != "N/A" or native.get("copy_on_forward", {}).get("status") != "N/A":
        errors.append("native vLLM capability must be explicit N/A")

    custom_fields = [
        "block", "workload", "prompt_tokens", "output_tokens", "batch", "arm",
        "arm_name", "output_token_ids_sha256", "prefill_ms", "cutover_ms",
        "first_decode_excluding_cutover_ms", "steady_decode_median_ms",
        "steady_decode_mean_ms", "steady_decode_p90_ms",
        "steady_decode_tokens_per_second", "cycle_e2e_ms", "load_seconds",
        "peak_hbm_mib", "selected_bytes", "coordinate_sha256", "audit_status",
    ]
    heatmap_fields = [
        "workload", "prompt_tokens", "output_tokens", "batch", "best_arm",
        "best_arm_name", "best_cycle_e2e_ms", "hybrid_cycle_e2e_ms",
        "always_copy_cycle_e2e_ms", "always_remote_cycle_e2e_ms", "status", "reason",
    ]
    break_fields = [
        "workload", "batch", "prompt_tokens", "left_arm", "right_arm",
        "anchor_output_tokens", "left_anchor_e2e_ms", "right_anchor_e2e_ms",
        "left_decode_median_ms",
        "right_decode_median_ms", "break_even_output_tokens", "interpretation", "model",
    ]
    region_fields = [
        "workload", "prompt_tokens", "batch", "output_start_inclusive",
        "output_end_inclusive", "best_arm", "best_arm_name", "model",
    ]
    kt_fields = [
        "block", "workload", "prompt_tokens", "output_tokens", "batch",
        "client_ttft_batch_ms", "server_prefill_forward_median_ms",
        "steady_decode_median_ms", "steady_decode_mean_ms", "decode_tokens_per_second",
        "cycle_e2e_ms", "output_token_ids_sha256", "peak_hbm_mib",
        "gpu_experts_per_layer", "cpu_experts_per_layer", "coordinate_sha256", "audit_status",
    ]
    write_csv(analysis / "custom_metrics.csv", custom_metrics, custom_fields)
    write_csv(analysis / "best_strategy_heatmap.csv", heatmap, heatmap_fields)
    write_csv(analysis / "pairwise_break_even.csv", pairwise, break_fields)
    write_csv(analysis / "strategy_regions.csv", regions, region_fields)
    write_csv(analysis / "kt_exact_coordinate_metrics.csv", kt_metrics, kt_fields)
    write_csv(analysis / "na_matrix.csv", na_rows, list(na_rows[0]) if na_rows else ["status"])

    status_counts = Counter(row["status"] for row in custom_rows)
    winners = Counter(row["best_arm"] for row in heatmap if row["status"] == "ok")
    verification = {
        "schema": VERIFICATION_SCHEMA,
        "profile": args.profile,
        "status": "ok" if not errors else "error",
        "run_root": str(root),
        "protocol_id": manifest.get("protocol_id"),
        "process_blocks": len(blocks),
        "requested_batches": list(ACTIVE_BATCHES),
        "kt_mode": args.kt_mode,
        "publication_repetition_requirement_met": len(blocks) >= 10,
        "checks": {
            "custom_execution_rows": len(custom_rows),
            "custom_status_counts": dict(status_counts),
            "custom_detailed_audit_failures": case_errors,
            "successful_complete_cells": len(by_cell),
            "h_a_r_output_hash_checks": hash_checks,
            "kt_execution_rows": len(kt_rows),
            "kt_measurement_cells": len(kt_metrics),
            "vllm_native_capability": native,
            "frozen_plan_sha256": sha256_file(plan_path),
            "coordinate_sha256": EXPECTED_COORDINATE_SHA256,
            "selected_bytes": EXPECTED_SELECTED_BYTES,
        },
        "descriptive_results": {
            "observed_winner_counts": dict(winners),
            "heatmap_cells_ok": sum(row["status"] == "ok" for row in heatmap),
            "heatmap_cells_na": sum(row["status"] == "N/A" for row in heatmap),
            "break_even_model": (
                "measured_cycle_e2e_ms + (output_tokens-anchor_output_tokens) * "
                "steady_decode_median_ms (local descriptive extrapolation)"
            ),
        },
        "statistical_scope": (
            "One independent process block: descriptive point estimates only; "
            "no CI or publication-grade repeated-cell claim."
            if len(blocks) == 1 else
            f"{len(blocks)} independent process blocks."
        ),
        "errors": errors,
        "outputs": {
            "custom_metrics_csv": str(analysis / "custom_metrics.csv"),
            "best_strategy_heatmap_csv": str(analysis / "best_strategy_heatmap.csv"),
            "best_strategy_heatmap_svg": str(analysis / "best_strategy_heatmap.svg"),
            "pairwise_break_even_csv": str(analysis / "pairwise_break_even.csv"),
            "strategy_regions_csv": str(analysis / "strategy_regions.csv"),
            "kt_exact_coordinate_metrics_csv": str(analysis / "kt_exact_coordinate_metrics.csv"),
            "na_matrix_csv": str(analysis / "na_matrix.csv"),
        },
    }
    (analysis / "verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
