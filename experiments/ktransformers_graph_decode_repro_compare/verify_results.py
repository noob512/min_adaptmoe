#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ARTIFACTS = HERE / "artifacts"


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    assert isinstance(value, dict), path
    return value


def main() -> int:
    checked: list[str] = []
    historical = (
        REPO
        / "experiments/broad_offload_compare/artifacts/kt_formal_frozen_20260813/cases"
    )
    for concurrency in (1, 4, 8):
        current_root = (
            ARTIFACTS
            / f"kt_old_protocol_graph_20260821/c{concurrency}/cases/graph_c{concurrency}"
        )
        current = load(current_root / "benchmark.json")
        previous = load(historical / f"c{concurrency}/benchmark.json")
        gate = load(current_root / "runtime_verification.json")
        assert current["status"] == "ok"
        assert current["phases"][0]["output_fingerprints_sha256"] == previous["phases"][0]["output_fingerprints_sha256"]
        assert gate["status"] == "ok" and gate["cuda_graph"] == "enabled"
        checked.append(f"kt_historical_c{concurrency}")

    steady = load(ARTIFACTS / "kt_steady_graph_64tok_20260821/result.json")
    steady_gate = load(ARTIFACTS / "kt_steady_graph_64tok_20260821/run_summary.json")
    assert steady["status"] == "ok" and steady_gate["cuda_graph_verified"] is True
    assert steady["config"]["output_tokens_per_request"] == 64
    for batch in ("1", "4"):
        summary = steady["phase_summary"][batch]
        assert summary["measurement_repeats"] == 3
        assert summary["cuda_graph_pass_delta"]["true"] > 0
        assert summary["cuda_graph_pass_delta"]["false"] == 0
        assert len(summary["output_token_ids_sha256"]) == 1
        checked.append(f"kt_steady_b{batch}")

    custom_roots = {
        "2g": ARTIFACTS / "custom_graph_2g_64tok_20260821_v2",
        "4g": ARTIFACTS / "custom_graph_4g_64tok_20260821",
    }
    hashes: dict[int, set[str]] = {1: set(), 4: set()}
    expected_bytes = {"2g": 2_264_924_160, "4g": 4_303_355_904}
    for tier, root in custom_roots.items():
        for batch in (1, 4):
            result = load(root / f"b{batch}/result.json")
            assert result["status"] == "ok"
            assert result["config"]["total_output_tokens"] == 64
            assert result["graph_wrapper_calls_final"]["full_capture"] == 2
            assert result["graph_wrapper_calls_final"]["full_replay"] == 63
            model = result["diagnostics"]["final"]["model"]
            assert model["selected_parameter_bytes_per_forward"] == expected_bytes[tier]
            assert model["remote_map_result"]["ok"] is True
            assert model["remote_map_result"]["remote_mapped_bytes"] == expected_bytes[tier]
            for segment in result["manifest_segments"].values():
                assert segment["actual_h2d_bytes"] == 0
            snapshots = result["parameter_va_snapshots"]
            assert len({tuple(snapshots[key]["vas"]) for key in ("after_load", "after_prefill", "after_first_decode", "final")}) == 1
            shutdown = result["shutdown_contract"]
            assert shutdown["cleanup_errors"] == []
            assert shutdown["diagnostics_after_shutdown"]["driver"]["session_count"] == 0
            hashes[batch].add(result["output_token_ids_sha256"])
            checked.append(f"custom_{tier}_b{batch}")
    assert all(len(values) == 1 for values in hashes.values())

    exact_6g = load(ARTIFACTS / "custom_graph_exact_20260821/b1/result.json")
    balanced_6g = load(ARTIFACTS / "custom_graph_balanced6g_20260821_v2/b1/result.json")
    assert exact_6g["status"] == "exception" and "same number of experts" in exact_6g["error"]
    assert balanced_6g["status"] == "exception" and "illegal memory access" in balanced_6g["error"]
    remote_6g = balanced_6g["shutdown_contract"]["diagnostics_after_shutdown"]["model"]["remote_map_result"]
    assert remote_6g["ok"] is True and remote_6g["remote_mapped_bytes"] == 6_568_280_064
    checked.extend(("custom_exact_6g_fail_closed", "custom_balanced_6g_fault_before_measurement"))

    summary = load(HERE / "RESULTS_20260821.json")
    assert summary["status"] == "ok_with_unsupported_custom_6g"
    output = {
        "schema": "ktransformers_graph_decode_repro_compare_verification_v1",
        "status": "ok_with_unsupported_custom_6g",
        "checked": checked,
        "checked_count": len(checked),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
