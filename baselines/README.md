# Optional baseline sources

The four-mechanism vLLM experiment uses only the bundled engine. The older
cross-system experiments additionally use KTransformers or MoE-Infinity.
`versions.json` pins their Git revisions and KTransformers' nested SGLang revision.

```bash
# Inspect versions without network or filesystem changes.
python3 scripts/fetch_baselines.py --dry-run

# Fetch both, or pass only one of these names.
python3 scripts/fetch_baselines.py KTransformers MoE-Infinity
```

The helper checks out the exact revisions, initializes their nested submodules
and applies `patches/sglang-scheduler-timestamps.patch`. This patch records the
schedule timestamps consumed by the measurement harness. An existing checkout
at a different revision is rejected. Repeating the command recognizes the
already applied patch.

For anonymous review copies, provide mirrors containing the pinned commits
before fetching:

```bash
export ADAPTMOE_KTRANSFORMERS_URL=https://example.invalid/ktransformers.git
export ADAPTMOE_MOE_INFINITY_URL=https://example.invalid/moe-infinity.git
```

Build and install each system according to its upstream README in its own
environment. Historical runners expect `KTransformers/.venv/bin/python`; some
MoE-Infinity experiments require a fresh extension build receipt as specified
by their protocol. Downloaded checkouts and their build outputs are ignored by
this repository. Expert placement generation and external model/data inputs are
listed in [the experiment index](../docs/EXPERIMENTS.md).
