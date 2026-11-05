# Evaluation scripts

This directory contains the plotting and table extraction code used by the
project's measured evaluation. It is deliberately separate from the runtime:
the scripts consume serialized CSV inputs, validate their schemas, and write
new output files without loading a model or touching the GPU.

Install the pinned plotting dependencies and render all stored tables:

```bash
python3 -m pip install -r evaluation/requirements.txt
make FIGURE_PYTHON=python3 figures
```

`evaluation/figures/scripts/` retains the per-figure parsers and renderers from
the experiment analysis. `evaluation/figures/data/` contains the derived,
validated tables used for the current figures. To regenerate a table from a
new structured experiment result, pass that result to the corresponding
script's `--report`, `--result`, or `--verification` option, then render from
the resulting CSV. Generated PDF/PNG/TEX files belong in `.build/` or another
ignored output directory.

The original measurements and model weights are not required for this static
path. They remain external inputs for a fresh GPU experiment and are referenced
by the provenance columns in the tables.
