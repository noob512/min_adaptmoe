#!/usr/bin/env python3
"""Compute arrival-cohort latency and wall-clock throughput without dropping tails."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ARMS = ("adaptmoe", "native_copy", "native_uva")


def mean(values):
    return statistics.fmean(values) if values else None


def p95(values):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * 0.95
    low = int(position)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)


def request_tpot(row):
    times = row.get("token_times_perf", [])
    count = row.get("completion_tokens", 0)
    if len(times) > 1 and count > 1:
        return (times[-1] - times[0]) * 1000 / (count - 1)
    return row.get("tpot_ms")


def interval_metrics(replay, start, end):
    origin = replay["trace_start_perf"]
    rows = replay["requests"]
    arrivals = [r for r in rows if start <= r["sent_at_perf"] - origin < end]
    good = [r for r in arrivals if r.get("success")]
    finished = [r for r in rows if r.get("success") and start <= r["finished_at_perf"] - origin < end]
    ttft = [r["ttft_ms"] for r in good]
    tpot = [request_tpot(r) for r in good if request_tpot(r) is not None]
    token_gaps = []
    token_count = 0
    for row in rows:
        times = row.get("token_times_perf", [])
        token_count += sum(start <= t - origin < end for t in times)
        token_gaps.extend((b - a) * 1000 for a, b in zip(times, times[1:])
                          if start <= b - origin < end)
    return dict(start_s=start, end_s=end, arrival_count=len(arrivals),
                successful_arrivals=len(good), completed_count=len(finished),
                arrival_qps=len(arrivals) / (end - start),
                completion_qps=len(finished) / (end - start),
                ttft_mean_ms=mean(ttft), ttft_p95_ms=p95(ttft),
                tpot_request_mean_ms=mean(tpot), tpot_request_p95_ms=p95(tpot),
                tpot_observed_mean_ms=mean(token_gaps), observed_token_gaps=len(token_gaps),
                output_tok_s=token_count / (end - start),
                outstanding_at_end=sum(r["sent_at_perf"] - origin < end <= r["finished_at_perf"] - origin for r in rows),
                max_client_send_lag_ms=max(((r["sent_at_perf"] - r["scheduled_at_perf"]) * 1000 for r in arrivals), default=None))


def tier_events(arm_root, replay):
    events = []
    # Core commits are the point capacity is published, after physical audits.
    year = datetime.fromtimestamp(replay["trace_start_epoch"], ZoneInfo("Asia/Shanghai")).year
    pattern = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d).*UVM KV autoscale committed target=(\d+) at step=(\d+)")
    for line in (arm_root / "server.log").read_text().split("\n"):
        match = pattern.search(line)
        if match:
            epoch = datetime.strptime(f"{year}-{match[1]}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
            events.append(dict(time_s=epoch - replay["trace_start_epoch"], blocks=int(match[2]), step=int(match[3])))
    return events


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def server_samples(arm_root, replay):
    year = datetime.fromtimestamp(replay["trace_start_epoch"], ZoneInfo("Asia/Shanghai")).year
    pattern = re.compile(r"(\d\d-\d\d \d\d:\d\d:\d\d).*Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%")
    result = []
    for line in (arm_root / "server.log").read_text().split("\n"):
        match = pattern.search(line)
        if match:
            epoch = datetime.strptime(f"{year}-{match[1]}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
            result.append(dict(time_s=epoch - replay["trace_start_epoch"], running=int(match[2]), waiting=int(match[3]), kv_usage_pct=float(match[4])))
    return result


def analyze(root, window):
    manifest = json.loads((root / "pressure_wave.manifest.json").read_text())
    replays = {arm: json.loads((root / arm / "replay.json").read_text()) for arm in ARMS}
    expected = manifest["request_count"]
    for arm, replay in replays.items():
        if replay["status"] != "ok" or replay["successful_requests"] != expected:
            raise ValueError(f"{arm} is incomplete")
        if replay["trace_sha256"] != manifest["trace_sha256"] or replay["arrival_scale"] != 1:
            raise ValueError(f"{arm} used a different arrival trace")
        if replay["max_connections"] < expected:
            raise ValueError(f"{arm} may be client-connection limited")
    horizon = manifest["arrival_horizon_s"]
    # Windows are half-open. Extend beyond the final timestamp by one float
    # step so the last completion is included in the final drain window.
    end = math.nextafter(max(horizon, *(max(r["finished_at_perf"] for r in replay["requests"]) - replay["trace_start_perf"] for replay in replays.values())), math.inf)
    windows, phases, samples = [], [], []
    for arm, replay in replays.items():
        samples.extend(dict(mechanism=arm, **row) for row in server_samples(root / arm, replay))
        for index in range(math.ceil(end / window)):
            windows.append(dict(mechanism=arm, **interval_metrics(replay, index * window, min((index + 1) * window, end))))
        for phase in manifest["phases"]:
            phases.append(dict(mechanism=arm, phase=phase["name"], target_qps=phase["offered_qps"],
                               **interval_metrics(replay, phase["start_s"], phase["end_s"])))
        phases.append(dict(mechanism=arm, phase="recovery_last_128s", target_qps=0.25,
                           **interval_metrics(replay, horizon - 128, horizon)))
        if end > horizon:
            phases.append(dict(mechanism=arm, phase="final_drain", target_qps=0,
                               **interval_metrics(replay, horizon, end)))
    audit = dict(trace=manifest, window_seconds=window,
                 tier_commits=tier_events(root / "adaptmoe", replays["adaptmoe"]),
                 overall={arm: {key: replay[key] for key in ("qps", "successful_requests", "ttft_ms", "tpot_ms", "e2e_ms", "measurement_window_seconds")} for arm, replay in replays.items()},
                 phase_metrics=phases)
    audit["native_offload"] = {}
    audit["pressure_recovery"] = {}
    recovery_start = manifest["phases"][2]["start_s"]
    high_start = manifest["phases"][1]["start_s"]
    for arm in ARMS:
        arm_samples = [row for row in samples if row["mechanism"] == arm]
        recovered = next((row["time_s"] for row in arm_samples
                          if row["time_s"] >= recovery_start and row["waiting"] == 0), None)
        audit["pressure_recovery"][arm] = dict(
            first_sample_without_waiting_after_high_s=recovered,
            seconds_after_load_drop=None if recovered is None else recovered - recovery_start,
            max_waiting_after_high_start=max((row["waiting"] for row in arm_samples if row["time_s"] >= high_start), default=0),
            peak_running=max((row["running"] for row in arm_samples), default=0),
        )
    for arm in ARMS[1:]:
        entries = [json.loads(line) for line in (root / arm / "offload_manifest.jsonl").read_text().split("\n") if line]
        parameters = [entry for entry in entries if entry["event"] == "cpu_offload_parameter"]
        audit["native_offload"][arm] = dict(parameter_count=len(parameters), bytes=sum(entry["bytes"] for entry in parameters),
                                           selection=[(entry["full_parameter_name"], entry["bytes"]) for entry in parameters])
    adapt_events = [json.loads(line) for line in (root / "adaptmoe/offload_manifest.jsonl").read_text().split("\n") if line]
    origin_epoch = replays["adaptmoe"]["trace_start_epoch"]
    last_epoch = origin_epoch + replays["adaptmoe"]["measurement_window_seconds"]
    formal_events = [entry for entry in adapt_events if origin_epoch <= entry.get("timestamp", 0) <= last_epoch]
    audit["adaptmoe_transitions"] = {}
    for event in ("cpu_offload_uvm_original_va_remote_transition", "cpu_offload_uvm_original_va_alias_prefill_activated"):
        selected = [entry for entry in formal_events if entry["event"] == event]
        audit["adaptmoe_transitions"][event] = dict(count=len(selected), total_ms=sum(entry.get("wall_ns", 0) for entry in selected) / 1e6)
    audit["plan_class_switches"] = [dict(time_s=entry["timestamp"] - origin_epoch,
                                              old_class=entry.get("old_class"), new_class=entry.get("new_class"), ok=entry.get("ok"))
                                   for entry in formal_events if entry["event"] == "cpu_offload_uvm_stable_plan_class_switched"]
    reference = replays["adaptmoe"]["requests"]
    audit["stream_capture_audit"] = {}
    for arm, replay in replays.items():
        incomplete = [row for row in replay["requests"] if len(row["output_token_ids"]) != row["completion_tokens"]]
        audit["stream_capture_audit"][arm] = dict(
            incomplete_token_id_records=[dict(request_id=row["request_id"], captured=len(row["output_token_ids"]), service_count=row["completion_tokens"]) for row in incomplete],
            corrected_tpot_mean_ms=mean([request_tpot(row) for row in replay["requests"]]),
            method="(last received token time - first received token time) / (service completion_tokens - 1); original replay unchanged",
        )
    audit["output_comparison"] = {}
    for arm in ARMS[1:]:
        candidate = replays[arm]["requests"]
        if [(r["request_id"], r["prompt_tokens"], r["expected_output_tokens"]) for r in reference] != [(r["request_id"], r["prompt_tokens"], r["expected_output_tokens"]) for r in candidate]:
            raise ValueError("request identity or lengths differ")
        comparable = [(a, b) for a, b in zip(reference, candidate)
                      if len(a["output_token_ids"]) == a["completion_tokens"] and len(b["output_token_ids"]) == b["completion_tokens"]]
        audit["output_comparison"][arm] = dict(first_token_matches=sum(a["output_token_ids"][:1] == b["output_token_ids"][:1] for a, b in zip(reference, candidate)),
                                                complete_sequence_comparable=len(comparable),
                                                full_sequence_matches=sum(a["output_token_ids"] == b["output_token_ids"] for a, b in comparable))
    write_csv(root / "time_windows.csv", windows)
    write_csv(root / "phase_summary.csv", phases)
    if samples:
        write_csv(root / "server_samples.csv", samples)
    (root / "analysis.json").write_text(json.dumps(audit, indent=2) + "\n")
    plot(root, manifest, windows, audit["tier_commits"])
    return audit


def plot(root, manifest, windows, commits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True, constrained_layout=True)
    colors = dict(adaptmoe="#237a57", native_copy="#c46a1b", native_uva="#3265b0")
    labels = dict(adaptmoe="AdaptMoE K2/K4", native_copy="Native Copy K4", native_uva="Native UVA K4")
    for arm in ARMS:
        rows = [r for r in windows if r["mechanism"] == arm]
        times = [(r["start_s"] + r["end_s"]) / 2 for r in rows]
        for ax, field, divisor in zip(axes, ("completion_qps", "ttft_mean_ms", "tpot_observed_mean_ms"), (1, 1000, 1)):
            values = [r[field] / divisor if r[field] is not None else float("nan") for r in rows]
            ax.plot(times, values, label=labels[arm], color=colors[arm], marker=".", linewidth=1.5)
    for phase in manifest["phases"]:
        axes[0].plot([phase["start_s"], phase["end_s"]], [phase["offered_qps"]] * 2, "k--", linewidth=1)
        for ax in axes:
            ax.axvspan(phase["start_s"], phase["end_s"], color="#e5a747" if phase["name"] == "high" else "#7ebde0", alpha=0.10)
    xs, ys = [0], [2]
    for event in commits:
        if event["time_s"] >= 0:
            xs.append(event["time_s"])
            ys.append(2 if event["blocks"] == 1364 else 4)
    xs.append(max(r["end_s"] for r in windows))
    ys.append(ys[-1])
    axes[3].step(xs, ys, where="post", color=colors["adaptmoe"])
    axes[3].set_yticks([2, 4], labels=["K2 / cold14", "K4 / cold28"])
    axes[3].set_ylim(1.5, 4.5)
    for ax, label in zip(axes, ("Completed requests/s", "TTFT mean (s)\nby request arrival", "Token interval mean (ms)\nby observation time", "Committed KV tier")):
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper right", ncol=3)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("TTFT mean (s, log scale)\nby request arrival")
    axes[-1].set_xlabel("Seconds since formal trace start (warmup excluded)")
    fig.suptitle("ShareGPT pressure wave: 0.25 → 2 → 0.25 req/s; C16; eager; one run per arm")
    fig.savefig(root / "pressure_wave.png", dpi=170)
    fig.savefig(root / "pressure_wave.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--window", type=float, default=30)
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("window must be positive")
    print(json.dumps(analyze(args.root, args.window), indent=2))
