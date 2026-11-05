#!/usr/bin/env python3
"""Read existing forward manifests to explain the fixed/dynamic KV results."""
import argparse
import collections
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
    summary = json.loads((root / 'summary.json').read_text())
    expert_bytes = {'cold11': 4948230144, 'cold17': 7647264768, 'cold23': 10346299392}
    cases, steps = [], []
    for case in summary['cases']:
        directory = Path(case['directory'])
        result = json.loads((directory / 'result.json').read_text())
        window = result['measurement_window']
        copies = []
        for line in (directory / 'cpu_offload_manifest.jsonl').read_text().splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event['event'] == 'cpu_offload_copy_forward' and window['started_epoch'] <= event['timestamp'] <= window['finished_epoch']:
                copies.append(event)
        layers = collections.defaultdict(list)
        for event in copies:
            layers[event['module_prefix']].append(event)
        count = result['summary']['engine_steps']
        assert len(layers) == 26 and all(len(events) == count for events in layers.values())
        stage_counts = collections.Counter()
        h2d_total, copy_steps = 0, 0
        assert len(result['scheduler_steps']) == len(result['dynamic_kv']['controller_steps']) == count
        for index, (scheduled, controller) in enumerate(zip(result['scheduler_steps'], result['dynamic_kv']['controller_steps'])):
            events = [events[index] for events in layers.values()]
            submits = {e['graph_static_alias_copy_submitted'] for e in events}
            assert len(submits) == 1
            submitted = submits.pop()
            assert isinstance(submitted, bool)
            actual_h2d = sum(e['graph_static_alias_actual_h2d_bytes'] for e in events)
            stage = controller['active_plan_class']
            assert actual_h2d == (expert_bytes[stage] if submitted else 0)
            stage_counts[stage] += 1
            h2d_total += actual_h2d
            copy_steps += int(submitted)
            steps.append({'offered_qps': case['offered_qps'], 'arm': case['arm'], 'sample': case['sample'],
                          'step': index + 1, 'scheduled_tokens': scheduled['total_num_scheduled_tokens'],
                          'expert_class': stage, 'copy_submitted': submitted,
                          'actual_expert_h2d_bytes': actual_h2d,
                          'engine_step_duration_ms': scheduled['engine_step_duration_ms']})
        cases.append({'offered_qps': case['offered_qps'], 'arm': case['arm'], 'sample': case['sample'],
                      'engine_steps': count, 'copy_steps': copy_steps, 'remote_steps': count - copy_steps,
                      'expert_h2d_bytes': h2d_total, 'expert_h2d_gib': h2d_total / 2**30,
                      'resize_wall_ms': case['resize_wall_ms'], 'stage_step_counts': dict(stage_counts)})
    rows = []
    for rate in (4, 8, 10, 20, 30):
        for arm in ('adaptive_fixed', 'adaptive_dynamic'):
            selected = [c for c in cases if c['offered_qps'] == rate and c['arm'] == arm]
            assert len(selected) == 2
            rows.append({'offered_qps': rate, 'arm': arm,
                         **{key: statistics.fmean(c[key] for c in selected) for key in (
                             'engine_steps', 'copy_steps', 'remote_steps', 'expert_h2d_gib', 'resize_wall_ms')}})
    analysis = {'status': 'ok', 'rows': rows, 'cases': cases, 'expert_bytes_by_class': expert_bytes,
                'scope': 'Expert H2D explicitly submitted in forward; excludes KV migration and plan-class migration. No additional profiling or timed-path instrumentation.'}
    (root / 'transport_analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')
    with (root / 'transport_steps.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(steps[0]))
        writer.writeheader()
        writer.writerows(steps)
    marker = '\n## 搬运量与性能差异证据\n'
    report = (root / 'REPORT.md').read_text().split(marker)[0]
    lines = [marker,
             '依据已采集的forward manifest逐层核算显式专家H2D，不追加GPU剖析或改变计时路径。下表为每个配置两次独立进程的均值；KV迁移与专家档位迁移另计，未包含在专家forward H2D列。', '',
             '| QPS | 配置 | Engine steps | Copy steps | Remote steps | 专家forward H2D GiB | resize总耗时ms |',
             '|---:|---|---:|---:|---:|---:|---:|']
    for row in rows:
        label = '固定K2' if row['arm'] == 'adaptive_fixed' else '动态'
        lines.append(f"| {row['offered_qps']} | {label} | {row['engine_steps']:.1f} | {row['copy_steps']:.1f} | {row['remote_steps']:.1f} | {row['expert_h2d_gib']:.3f} | {row['resize_wall_ms']:.2f} |")
    lines += ['',
              '每个copy step的专家搬运量随cold11/17/23分别为4.608/7.122/9.636GiB。扩大KV允许更多请求同时运行，同时更多专家需走远程/copy路径；更高并发不自动等于更高吞吐。应将总H2D和resize时间与实际吞吐、TTFT、TPOT同时评估。', '',
              '这些记录能够直接证实搬运量和扩容开销的变化，不能将全部延迟差精确分摊到DMA、GPU计算和控制面；若需各项毫秒贡献，仍需单独的对称剖析。逐step调度token和实际H2D见transport_steps.csv。', '']
    lines += ['## 本轮结论与解释边界', '',
              '本轮各档QPS中，固定K2的吞吐、TTFT和TPOT均优于动态组；自动扩容功能实际生效，但未取得性能收益。QPS=4/8/10两组均保持K2；QPS=20/30动态组两次重复均执行K2→K4→K6，无降档、无抢占。', '']
    for rate in (20, 30):
        fixed = next(r for r in rows if r['offered_qps'] == rate and r['arm'] == 'adaptive_fixed')
        dynamic = next(r for r in rows if r['offered_qps'] == rate and r['arm'] == 'adaptive_dynamic')
        comparison = next(c for c in summary['comparisons'] if c['offered_qps'] == rate)
        lines.append(
            f"QPS={rate}：动态组专家forward H2D为{dynamic['expert_h2d_gib']:.3f}GiB，"
            f"固定组为{fixed['expert_h2d_gib']:.3f}GiB，增加"
            f"{(dynamic['expert_h2d_gib'] / fixed['expert_h2d_gib'] - 1) * 100:.2f}%；"
            f"另有{dynamic['resize_wall_ms']/1000:.3f}s扩容总耗时，"
            f"实际吞吐变化{comparison['dynamic_qps_change_percent']:+.2f}%。"
        )
        lines.append('')
    lines += [
        '固定K2在高压力下限制同时运行请求为66，所有请求仍能完成且没有抢占；动态组放大并发和专家卸载，额外搬运和扩容成本超过本轮容量收益。动态组活跃KV峰值约3.20/3.74GiB，触发K6不等于实测活跃KV已超过4GiB。不能据此假定容量越大吞吐越高，也未单独测量限制在K4的反事实性能。', '',
        '低压力时没有扩容，专家H2D相同或更少，但动态组延迟仍略高。源码中自动控制器每step仍会调用队列snapshot，遍历全池并检查不变量；同时实际合批和remote步数也有变化。此处没有对控制器单独计时，不能把低压力延迟差全部归因于控制面。', '',
        '输出校验未达到严格等价：高压力两组重复内一致，但固定/动态之间QPS=20有22/128、QPS=30有27/128完整输出不同，首token相同；低压力还有重复间差异，QPS=8/10跨配置分别有2/128、1/128首token不同。已保留所有差异，不将成功返回等同于逐token正确性通过，也没有证实差异均来自数值或合批。性能数字可报告，但不应作为已通过严格输出等价验证的最终排名。', '',
        '每配置只有两个独立冷启动，全部测量包含排队、排空和实际扩容；这组数据不用于宣称统计显著性或纯稳态性能。', '']
    (root / 'REPORT.md').write_text(report + '\n'.join(lines))
    for name in ('analyze_p1024_kv_transport.py',):
        shutil.copy2(Path(__file__).with_name(name), root / 'source_snapshots' / name)
    shutil.copy2(root / 'REPORT.md', Path(__file__).with_name('P1024_N128_C128_FIXED_VS_DYNAMIC_KV_20261024.md'))
    print(json.dumps({'status': 'ok', 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
