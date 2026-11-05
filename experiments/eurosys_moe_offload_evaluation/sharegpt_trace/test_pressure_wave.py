import json
import tempfile
import unittest
from pathlib import Path

from analyze_pressure_wave import interval_metrics, request_tpot
from build_pressure_wave import build
from replay_sharegpt_trace import load_trace, merge_token_ids


class PressureWaveTests(unittest.TestCase):
    def test_repeated_token_delta_and_cumulative_streams(self):
        self.assertEqual(merge_token_ids([4710], [4710], 2), [4710, 4710])
        self.assertEqual(merge_token_ids([4710], [4710, 4710], 2), [4710, 4710])
        self.assertEqual(merge_token_ids([1, 2], [3, 4], 4), [1, 2, 3, 4])
        self.assertEqual(merge_token_ids([1, 2], [1, 2, 3], 3), [1, 2, 3])
        row = dict(token_times_perf=[100, 102, 103], completion_tokens=4)
        self.assertEqual(request_tpot(row), 1000)

    def test_full_content_and_burst_shape_are_identical_in_each_phase(self):
        source = Path(__file__).resolve().parent.parent / "inputs/sharegpt_trace_20261013/sharegpt_qwen_trace_low.jsonl"
        original = load_trace(source, None)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "wave.jsonl"
            manifest = build(source, output)
            rows = load_trace(output, None)
            self.assertEqual(len(rows), 384)
            self.assertEqual(len({r["request_id"] for r in rows}), 384)
            for index, phase in enumerate(manifest["phases"]):
                subset = rows[index * 128:(index + 1) * 128]
                self.assertEqual([r["source_request_id"] for r in subset], [r["request_id"] for r in original])
                self.assertEqual([r["prompt_token_ids"] for r in subset], [r["prompt_token_ids"] for r in original])
                self.assertEqual([r["max_tokens"] for r in subset], [r["max_tokens"] for r in original])
                self.assertTrue(all(phase["start_s"] <= r["arrival_ms"] / 1000 < phase["end_s"] for r in subset))
            low = [r["arrival_ms"] for r in rows[:128]]
            high = [r["arrival_ms"] - 512000 for r in rows[128:256]]
            for left, right in zip(low, high):
                self.assertAlmostEqual(left, right * 8, places=7)

    def test_window_keeps_late_completion_and_does_not_mix_latency_cohorts(self):
        replay = dict(trace_start_perf=100, requests=[
            dict(sent_at_perf=105, scheduled_at_perf=105, finished_at_perf=135,
                 success=True, ttft_ms=2000, tpot_ms=14000,
                 token_times_perf=[107, 121, 135]),
            dict(sent_at_perf=130, scheduled_at_perf=129.9, finished_at_perf=140,
                 success=True, ttft_ms=1000, tpot_ms=9000,
                 token_times_perf=[131, 140]),
        ])
        before = interval_metrics(replay, 0, 30)
        after = interval_metrics(replay, 30, 60)
        self.assertEqual(before["arrival_count"], 1)
        self.assertEqual(before["completed_count"], 0)
        self.assertEqual(before["outstanding_at_end"], 1)
        self.assertEqual(before["ttft_mean_ms"], 2000)
        self.assertEqual(after["arrival_count"], 1)
        self.assertEqual(after["completed_count"], 2)
        self.assertEqual(after["ttft_mean_ms"], 1000)
        self.assertEqual(after["observed_token_gaps"], 2)
        self.assertEqual(after["tpot_observed_mean_ms"], 11500)
        self.assertAlmostEqual(after["max_client_send_lag_ms"], 100)
        empty = interval_metrics(replay, 60, 90)
        self.assertIsNone(empty["ttft_mean_ms"])
        self.assertIsNone(empty["tpot_observed_mean_ms"])


if __name__ == "__main__":
    unittest.main()
