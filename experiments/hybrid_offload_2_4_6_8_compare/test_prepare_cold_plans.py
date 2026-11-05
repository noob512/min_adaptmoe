from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("prepare_cold_plans.py")
SPEC = importlib.util.spec_from_file_location("prepare_cold_plans", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_batched_decode_is_not_misclassified_as_prefill(tmp_path: Path) -> None:
    trace = tmp_path / "routing.jsonl"
    records = []
    for step, num_tokens in ((0, 32768), (1, 32768), (2, 32), (3, 512), (4, 8)):
        for layer_id in MODULE.LAYER_IDS:
            records.append(
                {
                    "step": step,
                    "layer_name": f"model.layers.{layer_id}.mlp.experts",
                    "num_tokens": num_tokens,
                    "top_k": 6,
                    "expert_token_counts": {"0": num_tokens * 6},
                }
            )
    trace.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    counts, coverage = MODULE.load_counts(
        trace, serving_start_step=3, prefill_step_count=1
    )

    assert coverage["record_counts"] == {"prefill": 26, "decode": 26}
    assert coverage["scheduler_steps"] == {"prefill": 1, "decode": 1, "total": 2}
    assert coverage["prefill_num_tokens"] == [512]
    assert coverage["decode_num_tokens_min"] == 8
    assert coverage["decode_num_tokens_max"] == 8
    assert all(layer_counts[0] == (512 + 8) * 6 for layer_counts in counts.values())
