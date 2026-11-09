# Figure reproduction status

The source-side entry point for the paper figure experiments is
`run_figure_experiments.py`; the complete mapping and environment contract are
documented in [FIGURE_REPRODUCTION.md](FIGURE_REPRODUCTION.md). Every run must
use a new ignored output directory, for example:

```bash
ALLOW_GPU_RUN=1 python3 -u experiments/reproduce/run_figure_experiments.py \
  --output .build/figure-suite-once
python3 experiments/reproduce/figure_run_report.py \
  .build/figure-suite-once --output .build/figure-run-report.json
```

The 2026-11-09 single-round validation used the stored `my_paper` figure
inputs and completed the placement, E6, E5 stable/non-stable, E13 KV,
stock-UVM, pressure, DeepSeek C16, Qwen AdaptMoE, and native Qwen protocols.
DeepSeek C1 and the additional E5 paths were also rerun in a separate output
directory. The report preserves unsuccessful cells: stock-UVM and Qwen KV
high-concurrency cells observed capacity OOM, while several C1 Graph cells
were rejected by the available GPU headroom. These observations are retained
as execution records and are not presented as successful performance claims.

Fresh measurements and logs stay under `.build/` and are intentionally not
part of the repository. The repository contains the runners, configuration,
frozen small inputs, verifiers, and report generator needed to repeat the
experiments after supplying local model and runtime paths.
