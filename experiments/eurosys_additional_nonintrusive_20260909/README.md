# Non-intrusive EuroSys additional experiments

This directory contains only wrappers, frozen configuration, logs, and
summaries added for the 2026-09-09 additional-experiment pass. Existing
runtime, driver, model, paper, experiment, and configuration files are treated
as immutable inputs.

`guard.py` records the pre-existing dirty worktree (including pre-existing
untracked files) and verifies before and after every case that it has not
changed. Files below this directory are the only allowed repository changes.

The pass intentionally uses qualification-sized repeats. Its results establish
availability, correctness, and first-order trends; they are not substitutes
for the five-seed/independent-process formal matrix.

