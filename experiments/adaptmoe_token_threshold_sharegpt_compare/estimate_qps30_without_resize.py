#!/usr/bin/env python3
"""Account for measured resize intervals on the existing execution trace."""
import argparse
import csv
import json
import shutil
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    source = json.loads((root / 'summary.json').read_text())
    samples, requests = [], []
    for case in source['cases']:
        if case['arm'] != 'adaptive_dynamic':
            continue
        raw = json.loads((Path(case['directory']) / 'result.json').read_text())
        origin = case['experiment_started_perf_counter']
        intervals = sorted((t['started_perf_counter'], t['finished_perf_counter']) for t in case['transactions'])
        assert all(origin <= left < right <= origin + case['elapsed_s'] for left, right in intervals)
        assert all(right <= next_left for (_, right), (next_left, _) in zip(intervals, intervals[1:]))
        resize_s = sum(right - left for left, right in intervals)
        assert abs(resize_s * 1000 - case['resize_wall_ms']) < 1e-6

        def overlap(left, right):
            return sum(max(0.0, min(right, end) - max(left, start)) for start, end in intervals)

        selected = []
        for req in raw['requests']:
            arrival = origin + req['planned_arrival_offset_seconds']
            first = arrival + req['ttft_ms'] / 1000
            last = arrival + req['e2e_ms'] / 1000
            count = req['completion_tokens'] - 1
            assert count == len(req['itl_ms']) == 15
            assert abs((last - first) * 1000 - count * req['tpot_ms']) < 1e-5
            before_ms, after_ms = overlap(arrival, first) * 1000, overlap(first, last) * 1000
            assert abs(before_ms + after_ms - overlap(arrival, last) * 1000) < 1e-5
            adjusted = {'sample': case['sample'], 'request_id': req['request_id'],
                        'original_ttft_ms': req['ttft_ms'], 'original_tpot_ms': req['tpot_ms'],
                        'resize_overlap_before_first_ms': before_ms,
                        'resize_overlap_after_first_ms': after_ms,
                        'adjusted_ttft_ms': req['ttft_ms'] - before_ms,
                        'adjusted_tpot_ms': req['tpot_ms'] - after_ms / count,
                        'adjusted_e2e_ms': req['e2e_ms'] - before_ms - after_ms}
            assert adjusted['adjusted_ttft_ms'] >= 0 and adjusted['adjusted_tpot_ms'] >= 0
            requests.append(adjusted)
            selected.append(adjusted)
        elapsed = case['elapsed_s'] - resize_s
        samples.append({'sample': case['sample'], 'requests': len(selected),
                        'original_qps': case['actual_qps'], 'original_elapsed_s': case['elapsed_s'],
                        'resize_s': resize_s, 'adjusted_elapsed_s': elapsed,
                        'adjusted_qps': len(selected) / elapsed,
                        **{key: statistics.fmean(r[key] for r in selected) for key in (
                            'adjusted_ttft_ms', 'adjusted_tpot_ms', 'adjusted_e2e_ms')},
                        'qps_minus_resize_and_controller': len(selected) / (
                            elapsed - case['controller_outside_resize_ms'] / 1000)})
    assert len(samples) == 2
    adjusted = {key: statistics.fmean(s[key] for s in samples) for key in (
        'adjusted_qps', 'adjusted_elapsed_s', 'adjusted_ttft_ms', 'adjusted_tpot_ms', 'adjusted_e2e_ms')}
    fixed, dynamic = source['rows']
    comparisons = {'qps_vs_fixed_percent': (adjusted['adjusted_qps'] / fixed['actual_qps'] - 1) * 100,
                   'ttft_vs_fixed_percent': (adjusted['adjusted_ttft_ms'] / fixed['ttft_mean_ms'] - 1) * 100,
                   'tpot_vs_fixed_percent': (adjusted['adjusted_tpot_ms'] / fixed['tpot_mean_ms'] - 1) * 100,
                   'qps_vs_original_dynamic_percent': (adjusted['adjusted_qps'] / dynamic['actual_qps'] - 1) * 100,
                   'resize_share_of_elapsed_gap_percent': dynamic['resize_wall_ms'] / 1000 / (
                       dynamic['elapsed_s'] - fixed['elapsed_s']) * 100}
    result = {'status': 'accounting_estimate_only', 'new_gpu_measurements': False,
              'method': 'Preserve the recorded execution. Subtract the serial resize intervals from elapsed time. For request latencies, pause accounting only during intersection of resize with arrival-to-first-token and first-to-last-token intervals. Arrival and scheduling are not replayed, so this is not a zero-cost counterfactual, a throughput upper bound, or a valid replacement for the primary measurements.',
              'source': str(root / 'summary.json'), 'samples': samples,
              'adjusted': adjusted, 'comparisons': comparisons}
    (root / 'without_resize_estimate.json').write_text(json.dumps(result, indent=2) + '\n')
    with (root / 'without_resize_requests.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(requests[0]))
        writer.writeheader()
        writer.writerows(requests)
    lines = ['# QPS30 K2/K4：排除实测扩容区间的记账估算', '',
             '**本页为原始轨迹上的事后扣时，没有重新运行GPU，也不替代原始实测。** 保留原合批、访问路径、专家卸载、KV档位及所有非扩容开销。', '',
             '总吞吐按128/(原总时长−实测扩容秒数)计算。TTFT仅扣除计划到达至首token之间与扩容的重叠；TPOT仅扣除首token至末token之间的扩容重叠，再除以15个生成间隔。不能从每个请求TTFT或每个生成间隔直接减去整段575.7ms。', '',
             '| 配置/口径 | QPS | TTFT s | TPOT ms |', '|---|---:|---:|---:|',
             f"| 固定K2实测 | {fixed['actual_qps']:.3f} | {fixed['ttft_mean_ms']/1000:.3f} | {fixed['tpot_mean_ms']:.2f} |",
             f"| 动态K2/K4实测 | {dynamic['actual_qps']:.3f} | {dynamic['ttft_mean_ms']/1000:.3f} | {dynamic['tpot_mean_ms']:.2f} |",
             f"| 动态原轨迹扣除扩容区间 | {adjusted['adjusted_qps']:.3f} | {adjusted['adjusted_ttft_ms']/1000:.3f} | {adjusted['adjusted_tpot_ms']:.2f} |", '',
             f"平均总时长{dynamic['elapsed_s']:.6f}s，扩容{dynamic['resize_wall_ms']/1000:.6f}s，扣除后{adjusted['adjusted_elapsed_s']:.6f}s。吞吐比原动态提高{comparisons['qps_vs_original_dynamic_percent']:.2f}%，仍比固定低{-comparisons['qps_vs_fixed_percent']:.2f}%。扩容约占本轮两组总时长差的{comparisons['resize_share_of_elapsed_gap_percent']:.2f}%。", '',
             f"扣时后的TTFT相对固定变化{comparisons['ttft_vs_fixed_percent']:+.2f}%，TPOT变化{comparisons['tpot_vs_fixed_percent']:+.2f}%。专家H2D仍是203.188GiB对179.728GiB，且保留约202ms控制器其他开销与不同计算形状。", '',
             '去掉扩容实际会改变到达积压、合批、后续step耗时及控制器决策；此处没有重建这些因果变化。延迟结果仅表示原轨迹中与扩容直接重叠部分被排除后的记账值，不代表真实零成本系统必然得到相同TTFT/TPOT。吞吐估算也不是一般意义的性能上限。', '',
             '逐进程计算见without_resize_estimate.json，逐请求扣除量见without_resize_requests.csv。原始性能报告与数据保持不变。', '']
    (root / 'WITHOUT_RESIZE_ESTIMATE.md').write_text('\n'.join(lines))
    shutil.copy2(Path(__file__), root / Path(__file__).name)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
