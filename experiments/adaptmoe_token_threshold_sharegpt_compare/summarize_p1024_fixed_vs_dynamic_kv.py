#!/usr/bin/env python3
"""Summarize freshly interleaved P1024 locked-K2 and automatic-KV runs."""
import argparse
import csv
import hashlib
import json
import shutil
import statistics
from pathlib import Path

from verify_p1024_fixed_vs_dynamic_kv import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    rows, comparisons, cases, checks = [], [], [], []
    for rate in (4, 8, 10, 20, 30):
        value = verify(root / f'qps{rate}')
        cases.extend(value['cases'])
        checks.extend(dict(c, offered_qps=rate) for c in value['output_checks'])
        selected = []
        for arm, label in (('adaptive_fixed', '固定K2'), ('adaptive_dynamic', '动态K2→K4→K6')):
            samples = [c for c in value['cases'] if c['arm'] == arm]
            assert len(samples) == 2
            row = {'offered_qps': rate, 'arm': arm, 'label': label, 'samples': 2,
                   **{key: statistics.fmean(c[key] for c in samples)
                      for key in ('actual_qps', 'ttft_mean_ms', 'tpot_mean_ms', 'e2e_mean_ms')},
                   'actual_qps_min': min(c['actual_qps'] for c in samples),
                   'actual_qps_max': max(c['actual_qps'] for c in samples),
                   'used_tiers': sorted({t for c in samples for t in c['used_tiers']}),
                   'growth_counts': [c['growth_count'] for c in samples],
                   'shrink_counts': [c['shrink_count'] for c in samples],
                   'resize_wall_ms': statistics.fmean(c['resize_wall_ms'] for c in samples),
                   'max_running_requests': max(c['max_running_requests'] for c in samples),
                   'max_waiting_requests': max(c['max_waiting_requests'] for c in samples),
                   'max_active_blocks': max(c['scheduler_max_active_blocks'] for c in samples),
                   'max_active_kv_gib': max(c['max_active_kv_gib'] for c in samples),
                   'preemptions': sum(c['preemptions'] for c in samples)}
            rows.append(row)
            selected.append(row)
        fixed, dynamic = selected
        comparisons.append({'offered_qps': rate,
                            'dynamic_qps_change_percent': (dynamic['actual_qps'] / fixed['actual_qps'] - 1) * 100,
                            'dynamic_ttft_change_percent': (dynamic['ttft_mean_ms'] / fixed['ttft_mean_ms'] - 1) * 100,
                            'dynamic_tpot_change_percent': (dynamic['tpot_mean_ms'] / fixed['tpot_mean_ms'] - 1) * 100,
                            'dynamic_e2e_change_percent': (dynamic['e2e_mean_ms'] / fixed['e2e_mean_ms'] - 1) * 100})
    result = {'status': 'ok_with_output_drift' if any(c['full_output_differences'] for c in checks) else 'ok',
              'errors': [], 'rows': rows, 'comparisons': comparisons, 'cases': cases, 'output_checks': checks,
              'valid_processes': 20, 'successful_requests': 2560,
              'fixture_manifest': json.loads((root / 'fixture_manifest.json').read_text()),
              'tier_manifest': json.loads((root / 'inputs/manifest.json').read_text())}
    assert len(cases) == 20
    (root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    with (root / 'summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# P1024/C128：AdaptMoE固定K2与动态KV对照', '',
             '128条相同ShareGPT请求，1024输入/16输出，DeepSeek-V2-Lite-Chat BF16，TRITON_MLA，C128，8192-token预算，混合调度，PARK≤16/RESUME≥128，eager，无CUDA Graph，关闭prefix cache。每档QPS两组各两个独立进程，固定→动态→动态→固定交错；共20个新进程、2560次请求，无复用历史数据。无额外conditioning，不宣称纯稳态；QPS含排空，TTFT包含计划到达后的排队，TPOT包含mixed step与扩容。', '',
             '固定组为容量锁定的managed K2（GPU target4313 blocks，约2GiB）/cold11；动态组从相同K2/cold11启动，可自动K2/K4/K6并配对cold11/17/23。两组都预留6GiB虚拟KV地址池，启动时仅K2前缀GPU驻留，其余CPU驻留。CLI的6GiB表示VA预留上限，不表示固定组有6GiB GPU KV。', '',
             '两组运行设置仅AUTO_SCALE_ENABLE和观测标签不同；初始化、分配器、初始专家坐标、KV几何、16个backup、token访问切换、全部物理迁移/驻留检查和fault mode2一致。固定组并非历史native分配器固定KV基线。', '',
             '| 提供QPS | 配置 | 实际QPS | TTFT mean s | TPOT mean ms | E2E mean s | 实际档位 |',
             '|---:|---|---:|---:|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['offered_qps']} | {row['label']} | {row['actual_qps']:.3f} | {row['ttft_mean_ms']/1000:.3f} | {row['tpot_mean_ms']:.2f} | {row['e2e_mean_ms']/1000:.3f} | {'→'.join(row['used_tiers'])} |")
    lines += ['', '## 动态相对固定K2', '',
              '正的QPS变化表示吞吐提高；负的延迟变化表示延迟降低。', '',
              '| 提供QPS | QPS变化 | TTFT变化 | TPOT变化 | E2E变化 |',
              '|---:|---:|---:|---:|---:|']
    for item in comparisons:
        lines.append(f"| {item['offered_qps']} | {item['dynamic_qps_change_percent']:+.2f}% | {item['dynamic_ttft_change_percent']:+.2f}% | {item['dynamic_tpot_change_percent']:+.2f}% | {item['dynamic_e2e_change_percent']:+.2f}% |")
    lines += ['', '## 容量、并发与实际扩容', '',
              '| QPS | 配置 | 峰值运行请求 | 峰值等待 | 活跃KV峰值GiB | 两次升档数 | 两次降档数 | 平均resize总耗时ms | 抢占次数 |',
              '|---:|---|---:|---:|---:|---|---|---:|---:|']
    for row in rows:
        lines.append(f"| {row['offered_qps']} | {row['label']} | {row['max_running_requests']} | {row['max_waiting_requests']} | {row['max_active_kv_gib']:.3f} | {row['growth_counts']} | {row['shrink_counts']} | {row['resize_wall_ms']:.2f} | {row['preemptions']} |")
    lines += ['', '## 逐事务记录', '',
              '| QPS | 样本 | Step | target blocks | 专家档位 | resize wall ms |',
              '|---:|---:|---:|---|---|---:|']
    for case in cases:
        for transaction in case['transactions']:
            lines.append(f"| {case['offered_qps']} | {case['sample']} | {transaction['controller_step']} | {transaction['old_target_blocks']}→{transaction['applied_target_blocks']} | {transaction['active_plan_class_before']}→{transaction['active_plan_class_after']} | {transaction['wall_ms']:.3f} |")
    lines += ['', '动态吞吐/TTFT收益需与更高专家卸载量、实际KV扩容成本和TPOT一起判断；本轮测量整个联动配置，不能宣称每token计算更快。两次重复不足以建立统计显著性。逐请求输出差异均保留，不能仅凭请求成功率宣称严格输出等价。', '',
              '## 输出与重复性', '',
              '| QPS | 对比 | 请求对数 | 完整输出不同 | 首token不同 |',
              '|---:|---|---:|---:|---:|']
    for check in checks:
        lines.append(f"| {check['offered_qps']} | {check['comparison']} | {check['request_pairs']} | {check['full_output_differences']} | {check['first_token_differences']} |")
    lines += ['', '## 逐进程测量', '',
              '| QPS | 配置 | 样本 | 实际QPS | TTFT s | TPOT ms |',
              '|---:|---|---:|---:|---:|---:|']
    for case in cases:
        lines.append(f"| {case['offered_qps']} | {case['arm']} | {case['sample']} | {case['actual_qps']:.6f} | {case['ttft_mean_ms']/1000:.6f} | {case['tpot_mean_ms']:.6f} |")
    lines += ['', f'原始artifact：{root}', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))
    hashes = {str(p.relative_to(root / 'source_snapshots')): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (root / 'source_snapshots').iterdir() if p.is_file()}
    (root / 'source_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    for suffix, source in (('md', 'REPORT.md'), ('json', 'summary.json'), ('csv', 'summary.csv')):
        shutil.copy2(root / source, Path(__file__).with_name(f'P1024_N128_C128_FIXED_VS_DYNAMIC_KV_20261024.{suffix}'))
    print(json.dumps({'status': result['status'], 'comparisons': comparisons,
                      'rows': rows, 'report': str(root / 'REPORT.md')}, indent=2))


if __name__ == '__main__':
    main()
