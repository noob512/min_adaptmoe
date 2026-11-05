#!/usr/bin/env python3
"""Compare E13 teacher-forced raw logits against the Adaptive arm."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import torch


# These are diagnostic acceptance bounds, fixed before the formal rerun.  The
# exact top-1 gate is reported separately and is never relaxed by these bounds.
MAX_TOKEN_RELATIVE_L1 = 5e-3
MAX_TARGET_LOGPROB_ABS_DELTA = 1e-1
MAX_REQUEST_PERPLEXITY_RELATIVE_DELTA = 5e-2


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


def load_logits(path: Path) -> dict[str, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "eurosys_e13_teacher_forced_logits_v1":
        raise ValueError(f"unsupported logits schema: {path}")
    return payload["records"]


def compare_request(
    *,
    repeat_index: int,
    seed: int,
    arm: str,
    logical: str,
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ref = reference["logits"].float()
    cand = candidate["logits"].float()
    target = torch.tensor(reference["target_token_ids"], dtype=torch.long)
    if reference["target_token_ids"] != candidate["target_token_ids"]:
        raise RuntimeError(f"teacher target differs for {arm}/{logical}")
    if ref.shape != cand.shape or ref.shape[0] != target.numel():
        raise RuntimeError(
            f"logit shape differs for {arm}/{logical}: {ref.shape} vs {cand.shape}"
        )
    finite = torch.isfinite(ref).all(dim=1) & torch.isfinite(cand).all(dim=1)
    diff = (cand - ref).abs()
    denominator = ref.abs().sum(dim=1).clamp_min(torch.finfo(torch.float32).tiny)
    relative_l1 = diff.sum(dim=1) / denominator
    max_abs = diff.max(dim=1).values
    ref_top2 = ref.topk(2, dim=1)
    cand_top2 = cand.topk(2, dim=1)
    ref_top1 = ref_top2.indices[:, 0]
    cand_top1 = cand_top2.indices[:, 0]
    top1_equal = ref_top1 == cand_top1
    rows = torch.arange(target.numel())
    ref_target_logprob = ref[rows, target] - torch.logsumexp(ref, dim=1)
    cand_target_logprob = cand[rows, target] - torch.logsumexp(cand, dim=1)
    target_delta = (cand_target_logprob - ref_target_logprob).abs()
    ref_perplexity = math.exp(-float(ref_target_logprob.mean()))
    cand_perplexity = math.exp(-float(cand_target_logprob.mean()))
    perplexity_relative_delta = abs(cand_perplexity - ref_perplexity) / max(
        abs(ref_perplexity), torch.finfo(torch.float32).tiny
    )
    token_rows: list[dict[str, Any]] = []
    for position in range(target.numel()):
        token_rows.append(
            {
                "repeat_index": repeat_index,
                "seed": seed,
                "arm": arm,
                "logical_request": logical,
                "position": position,
                "target_token_id": int(target[position]),
                "reference_top1_token_id": int(ref_top1[position]),
                "candidate_top1_token_id": int(cand_top1[position]),
                "top1_equal": bool(top1_equal[position]),
                "reference_top1_margin": float(
                    ref_top2.values[position, 0] - ref_top2.values[position, 1]
                ),
                "candidate_top1_margin": float(
                    cand_top2.values[position, 0] - cand_top2.values[position, 1]
                ),
                "relative_l1": float(relative_l1[position]),
                "max_abs_logit_delta": float(max_abs[position]),
                "reference_target_logprob": float(ref_target_logprob[position]),
                "candidate_target_logprob": float(cand_target_logprob[position]),
                "target_logprob_abs_delta": float(target_delta[position]),
                "finite": bool(finite[position]),
            }
        )
    request = {
        "repeat_index": repeat_index,
        "seed": seed,
        "arm": arm,
        "logical_request": logical,
        "token_positions": target.numel(),
        "finite_positions": int(finite.sum()),
        "top1_equal_positions": int(top1_equal.sum()),
        "top1_agreement_fraction": float(top1_equal.float().mean()),
        "maximum_relative_l1": float(relative_l1.max()),
        "maximum_abs_logit_delta": float(max_abs.max()),
        "maximum_target_logprob_abs_delta": float(target_delta.max()),
        "reference_perplexity": ref_perplexity,
        "candidate_perplexity": cand_perplexity,
        "perplexity_relative_delta": perplexity_relative_delta,
    }
    return request, token_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    manifest = load_json(root / "numerical_manifest.json")
    requests: list[dict[str, Any]] = []
    tokens: list[dict[str, Any]] = []
    execution_errors: list[str] = []
    for repeat in manifest["repeats"]:
        cases = list(repeat["cases"])
        if not cases:
            continue
        repeat_dir = root / repeat["repeat_name"]
        by_arm: dict[str, dict[str, dict[str, Any]]] = {}
        for arm in cases:
            case_dir = repeat_dir / arm
            result_path = case_dir / "result.json"
            returncode_path = case_dir / "returncode.txt"
            logits_path = case_dir / "teacher_logits.pt"
            if not result_path.is_file() or not returncode_path.is_file() or not logits_path.is_file():
                execution_errors.append(f"missing artifact: {repeat['repeat_name']}/{arm}")
                continue
            result = load_json(result_path)
            returncode = int(returncode_path.read_text(encoding="utf-8").strip())
            if returncode != 0 or result.get("status") != "ok":
                execution_errors.append(f"failed case: {repeat['repeat_name']}/{arm}")
                continue
            by_arm[arm] = load_logits(logits_path)
        if "adaptive" not in by_arm:
            execution_errors.append(f"missing Adaptive reference: {repeat['repeat_name']}")
            continue
        reference = by_arm["adaptive"]
        for arm, logicals in repeat["affected_fixed_arms"].items():
            if arm not in by_arm:
                continue
            for logical in logicals:
                if logical not in reference or logical not in by_arm[arm]:
                    execution_errors.append(
                        f"missing recorded request: {repeat['repeat_name']}/{arm}/{logical}"
                    )
                    continue
                request, token_rows = compare_request(
                    repeat_index=int(repeat["repeat_index"]),
                    seed=int(repeat["seed"]),
                    arm=arm,
                    logical=logical,
                    reference=reference[logical],
                    candidate=by_arm[arm][logical],
                )
                requests.append(request)
                tokens.extend(token_rows)

    all_finite = bool(tokens) and all(item["finite"] for item in tokens)
    all_top1 = bool(tokens) and all(item["top1_equal"] for item in tokens)
    tolerance_gate = (
        all_finite
        and bool(tokens)
        and max(item["relative_l1"] for item in tokens) <= MAX_TOKEN_RELATIVE_L1
        and max(item["target_logprob_abs_delta"] for item in tokens)
        <= MAX_TARGET_LOGPROB_ABS_DELTA
        and max(item["perplexity_relative_delta"] for item in requests)
        <= MAX_REQUEST_PERPLEXITY_RELATIVE_DELTA
    )
    expected_requests = int(manifest["divergent_request_count"])
    execution_complete = not execution_errors and len(requests) == expected_requests
    summary = {
        "schema": "eurosys_e13_teacher_forced_analysis_v1",
        "status": "ok" if execution_complete else "error",
        "profile": manifest["profile"],
        "source_run_root": manifest["source_run_root"],
        "execution_complete": execution_complete,
        "compared_requests": len(requests),
        "expected_requests": expected_requests,
        "compared_token_positions": len(tokens),
        "finite_token_positions": sum(item["finite"] for item in tokens),
        "top1_equal_token_positions": sum(item["top1_equal"] for item in tokens),
        "top1_agreement_fraction": (
            sum(item["top1_equal"] for item in tokens) / len(tokens) if tokens else None
        ),
        "exact_top1_gate_passed": all_top1,
        "numerical_tolerance_gate_passed": tolerance_gate,
        "maximum_relative_l1": max(
            (item["relative_l1"] for item in tokens), default=None
        ),
        "maximum_abs_logit_delta": max(
            (item["max_abs_logit_delta"] for item in tokens), default=None
        ),
        "maximum_target_logprob_abs_delta": max(
            (item["target_logprob_abs_delta"] for item in tokens), default=None
        ),
        "maximum_request_perplexity_relative_delta": max(
            (item["perplexity_relative_delta"] for item in requests), default=None
        ),
        "thresholds": {
            "maximum_token_relative_l1": MAX_TOKEN_RELATIVE_L1,
            "maximum_target_logprob_abs_delta": MAX_TARGET_LOGPROB_ABS_DELTA,
            "maximum_request_perplexity_relative_delta": (
                MAX_REQUEST_PERPLEXITY_RELATIVE_DELTA
            ),
            "exact_top1_required_for_algorithmic_equivalence": True,
        },
        "publication_disposition": {
            "teacher_forced_execution_valid": execution_complete,
            "algorithmic_equivalence_established": execution_complete and all_top1,
            "bounded_numerical_drift_established": execution_complete and tolerance_gate,
        },
        "execution_errors": execution_errors,
        "requests": requests,
    }
    output = args.output or (root / "teacher_forced_analysis.json")
    atomic_json(output, summary)
    csv_path = output.with_suffix(".tokens.csv")
    if csv_path.exists():
        raise FileExistsError(f"refusing to overwrite: {csv_path}")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tokens[0]) if tokens else ["empty"])
        writer.writeheader()
        writer.writerows(tokens)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if execution_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
