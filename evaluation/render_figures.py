#!/usr/bin/env python3
"""Render the stored measurement CSVs without models, GPUs or historical runs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import pandas as pd

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "figures/scripts"
DATA = ROOT / "figures/data"
sys.path.insert(0, str(SCRIPTS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / ".build/figures")
    parser.add_argument("--only", nargs="+", help="CSV stems; default: all 21 CSVs")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    available = {path.stem for path in DATA.glob("*.csv")}
    chosen = set(args.only or available)
    if chosen - available:
        parser.error(f"Unknown CSV stems: {sorted(chosen - available)}")
    receipts = []
    for stem in sorted(chosen):
        source = DATA / f"{stem}.csv"
        data = pd.read_csv(source)
        if data.empty:
            raise ValueError(f"No measurement rows: {source}")
        pdf, png = output / f"{stem}.pdf", output / f"{stem}.png"
        if stem.startswith("fig3"):
            module = importlib.import_module("fig3_performance_bars")
            model = next(key for key, value in module.MODELS.items() if value["stem"] == stem)
            module.render_model(data, model, pdf, png)
        elif stem == "fig_e14_placement_scaling":
            module = importlib.import_module("plot_fig_e14_placement_scaling")
            module.configure_matplotlib()
            module.render_figure(module.aggregate_tidy(module.read_tidy_csv(source)), output)
        elif stem == "tab_pd_lifecycle_invariants":
            module = importlib.import_module(stem)
            module.write_table(data, output / f"{stem}.tex")
        elif stem == "fig_kv_expert_partition_crossover":
            module = importlib.import_module(stem)
            claims = {}
            for workload in module.WORKLOAD_LABELS:
                values = data.loc[data["workload"].eq(workload), "required_kv_gib"].dropna().unique()
                if len(values) != 1:
                    raise ValueError(f"Inconsistent KV demand for {workload}: {values}")
                claims[f"{workload}_requires_gib"] = float(values[0])
            module.render({"claims": claims}, data, pdf, png)
        else:
            module = importlib.import_module(stem)
            renderer = getattr(module, "render_figure", None) or module.render
            renderer(data, pdf, png)
        receipts.append({"input": str(source.relative_to(ROOT)), "rows": len(data),
                         "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
        print(f"Rendered {stem}: {len(data)} input rows")
    (output / "render_manifest.json").write_text(json.dumps(receipts, indent=2) + "\n")


if __name__ == "__main__":
    main()
