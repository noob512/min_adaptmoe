#!/usr/bin/env python3
"""Render the verified E3 best-strategy heatmap as dependency-free SVG."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


def text(x: float, y: float, value: str, *, size: int = 14,
         weight: str = "normal", anchor: str = "middle") -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="DejaVu Sans, sans-serif" font-size="{size}" '
        f'font-weight="{weight}">{html.escape(value)}</text>'
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("heatmap_csv", type=Path)
    parser.add_argument("output_svg", type=Path)
    args = parser.parse_args()
    with args.heatmap_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    prompts = [128, 512, 2048, 8192]
    outputs = [32, 128, 512]
    batches = [1, 4, 8]
    colors = {"H": "#377eb8", "A": "#ff9f1c", "R": "#2ca25f", "N/A": "#bdbdbd"}
    lookup = {
        (int(row["batch"]), int(row["prompt_tokens"]), int(row["output_tokens"])): row
        for row in rows
    }
    width, height = 1220, 560
    cell_w, cell_h = 90, 72
    panel_gap, first_x, grid_y = 75, 95, 115
    panel_w = cell_w * len(outputs)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<defs><pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" '
        'patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="8" '
        'stroke="#dddddd" stroke-width="2"/></pattern></defs>',
        text(width / 2, 34, "Qwen Track-A E3: observed best internal path", size=20, weight="bold"),
        text(width / 2, 58, "one independent process block; values are E2E seconds", size=13),
    ]
    for panel, batch in enumerate(batches):
        x0 = first_x + panel * (panel_w + panel_gap)
        parts.append(text(x0 + panel_w / 2, 91, f"Strict batch B{batch}", size=16, weight="bold"))
        for x_index, output in enumerate(outputs):
            parts.append(text(x0 + (x_index + 0.5) * cell_w, grid_y - 12, str(output), size=13))
        for y_index, prompt in enumerate(prompts):
            if panel == 0:
                parts.append(text(x0 - 12, grid_y + (y_index + 0.55) * cell_h,
                                  str(prompt), size=13, anchor="end"))
            for x_index, output in enumerate(outputs):
                x = x0 + x_index * cell_w
                y = grid_y + y_index * cell_h
                row = lookup.get((batch, prompt, output))
                fill = "url(#hatch)" if row is None else colors[row["best_arm"]]
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                    f'fill="{fill}" stroke="white" stroke-width="3"/>'
                )
                if row is not None:
                    arm = row["best_arm"]
                    label = "N/A" if arm == "N/A" else arm
                    parts.append(text(x + cell_w / 2, y + 30, label, size=16, weight="bold"))
                    if arm != "N/A":
                        seconds = float(row["best_cycle_e2e_ms"]) / 1000.0
                        parts.append(text(x + cell_w / 2, y + 52, f"{seconds:.2f}s", size=12))
        parts.append(text(x0 + panel_w / 2, grid_y + len(prompts) * cell_h + 27,
                          "Output tokens", size=13))
    parts.append(
        f'<text x="25" y="{grid_y + len(prompts) * cell_h / 2}" '
        'text-anchor="middle" font-family="DejaVu Sans, sans-serif" font-size="13" '
        'transform="rotate(-90 25 259)">Prompt tokens</text>'
    )
    legend = [
        ("H", "H: Prefill-copy / Decode-remote"),
        ("A", "A: always-copy"),
        ("R", "R: always-remote"),
        ("N/A", "N/A: backend unsupported"),
    ]
    legend_y = 475
    for index, (arm, label) in enumerate(legend):
        x = 120 + index * 265
        parts.append(f'<rect x="{x}" y="{legend_y}" width="22" height="18" fill="{colors[arm]}"/>')
        parts.append(text(x + 30, legend_y + 14, label, size=12, anchor="start"))
    parts.append('<rect x="470" y="515" width="22" height="18" fill="url(#hatch)" stroke="#ccc"/>')
    parts.append(text(500, 529, "not in controlled matrix", size=12, anchor="start"))
    parts.append('</svg>')
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
