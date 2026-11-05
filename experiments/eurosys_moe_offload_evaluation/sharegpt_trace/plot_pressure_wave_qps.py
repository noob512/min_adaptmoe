#!/usr/bin/env python3
"""Plot completion QPS for the three pressure-wave arms from measured windows."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(root: Path) -> None:
    with (root / "time_windows.csv").open(newline="") as stream:
        windows = list(csv.DictReader(stream))
    manifest = json.loads((root / "pressure_wave.manifest.json").read_text())
    arms = (
        ("adaptmoe", "AdaptMoE K2/K4", "#16805d", "o"),
        ("native_copy", "Native Copy", "#d47a24", "s"),
        ("native_uva", "Native UVA", "#386bb4", "^"),
    )
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.19, top=0.77)
    exported = []
    finish_times = []
    for arm, label, color, marker in arms:
        replay = json.loads((root / arm / "replay.json").read_text())
        last_completion = max(row["finished_at_perf"] for row in replay["requests"]) - replay["trace_start_perf"]
        finish_times.append(last_completion)
        # Retain real zero-completion intervals during the run; omit padding
        # intervals inserted only to align an already completed arm with Copy.
        rows = [row for row in windows if row["mechanism"] == arm
                and float(row["start_s"]) <= last_completion]
        assert sum(int(row["completed_count"]) for row in rows) == replay["successful_requests"]
        xs = [(float(row["start_s"]) + float(row["end_s"])) / 2 for row in rows]
        ys = [float(row["completion_qps"]) for row in rows]
        ax.plot(xs, ys, color=color, marker=marker, markersize=4.3,
                linewidth=2, label=label, zorder=4)
        exported.extend({key: row[key] for key in ("mechanism", "start_s", "end_s", "completed_count", "completion_qps")}
                        for row in rows)

    for phase, name in zip(manifest["phases"], ("Initial low", "High", "Recovery low")):
        start, end = phase["start_s"], phase["end_s"]
        ax.axvspan(start, end, color="#efb458" if name == "High" else "#bad9ec",
                   alpha=0.19 if name == "High" else 0.14, zorder=0)
        ax.text((start + end) / 2, 1.045, f"{name}\n{phase['offered_qps']:g} req/s",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=10)
    horizon = manifest["arrival_horizon_s"]
    end = max(finish_times)
    for boundary in (512, 576, horizon):
        ax.axvline(boundary, color="#a5adb4", linestyle=":", linewidth=1, zorder=1)
    ax.axvspan(horizon, end, color="#eeeeee", alpha=0.55, zorder=0)
    ax.text((horizon + end) / 2, 1.045, "Drain\n0 req/s",
            transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=10)
    ax.step([0, 512, 576, horizon, end], [0.25, 2, 0.25, 0, 0], where="post",
            color="#60666d", linestyle="--", linewidth=1.4,
            label="Input QPS (reference)", zorder=2)

    ax.set(xlabel="Time since measurement start (s)", ylabel="QPS (requests/s)",
           xlim=(0, end + 12), ylim=(0, 2.2))
    ax.set_xticks([0, 200, 400, 512, 576, 800, 1000, 1088])
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="y", color="#dce1e5", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.suptitle("QPS over time: AdaptMoE vs Native Copy vs Native UVA", y=0.985, fontsize=16)
    fig.text(0.085, 0.915, "ShareGPT | 30-second windows | C16 | eager, CUDA Graph disabled", fontsize=10, color="#58616a")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.895),
               ncol=4, frameon=False, fontsize=10)
    fig.text(0.085, 0.025,
             "Solid lines: completed requests / window duration. Dashed line: target input rate.\n"
             "Curves stop after each arm's final completion window; post-completion zero padding is omitted.",
             fontsize=9, color="#58616a")
    for extension in ("png", "svg", "pdf"):
        path = root / f"qps_over_time.{extension}"
        fig.savefig(path, dpi=220, facecolor="white")
        print(path.resolve())
    plt.close(fig)
    with (root / "qps_over_time.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    plot(parser.parse_args().root)
