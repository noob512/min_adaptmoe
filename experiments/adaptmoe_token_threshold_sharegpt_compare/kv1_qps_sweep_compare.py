#!/usr/bin/env python3
"""Verify and summarize the four-arm P1024 KV1 / dynamic KV1-2 QPS sweep."""
import argparse
import hashlib
import itertools
import json
import os
import shutil
import statistics
from pathlib import Path

from analyze_qps30_kv_pressure import analyze_case, write_csv
from five_arm_k1_compare import verify_vllm
from verify_p1024_dynamic_kv import FIXTURE_SHA

HERE = Path(__file__).resolve().parent
LABELS = {'native_uva': 'Native UVA / fixed KV1',
          'native_copy': 'Native Copy / fixed KV1',
          'adaptive_fixed_1g': 'AdaptMoE / fixed KV1',
          'adaptive_dynamic': 'AdaptMoE / dynamic KV1→2'}
METRICS = ('actual_qps', 'ttft_mean_ms', 'tpot_mean_ms', 'e2e_mean_ms',
           'elapsed_s', 'resize_wall_ms', 'expert_h2d_gib')


def load(path):
    return json.loads(path.read_text())


def verify_rate(root, rate):
    samples=load(root/'run_config.json')['samples_per_arm']
    assert samples in (1,2)
    checked = verify_vllm(root, expected_qps=rate, expected_samples=samples)
    cases, outputs = [], {}
    for case in checked['cases']:
        if case['arm'].startswith('adaptive_'):
            case, _, _, _ = analyze_case(case, require_decode_allocations_success=False)
        cases.append({**case, 'offered_qps': rate})
        outputs[case['arm'], case['sample']] = load(Path(case['directory'])/'result.json')['requests']
    pairs = [(arm+'_repeat', (arm, 1), (arm, 2)) for arm in LABELS] if samples==2 else []
    pairs += [(f'{a}_vs_{b}_s{s}', (a, s), (b, s))
              for s in range(1,samples+1) for a, b in itertools.combinations(LABELS, 2)]
    checks = []
    for label, a, b in pairs:
        left, right = outputs[a], outputs[b]
        checks.append({'offered_qps': rate, 'comparison': label, 'request_pairs': 128,
                       'full_output_differences': sum(x['output_token_ids'] != y['output_token_ids'] for x, y in zip(left, right)),
                       'first_token_differences': sum(x['output_token_ids'][0] != y['output_token_ids'][0] for x, y in zip(left, right))})
    return {**checked, 'cases': cases, 'output_checks': checks,
            'status': 'ok_with_output_drift' if any(c['full_output_differences'] for c in checks) else 'ok'}


def plot(root, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for arm, label in LABELS.items():
        selected = [r for r in rows if r['arm'] == arm]
        x = [r['offered_qps'] for r in selected]
        for ax, key, divisor in zip(axes, ('actual_qps', 'ttft_mean_ms', 'tpot_mean_ms'), (1, 1000, 1)):
            ax.plot(x, [r[key]/divisor for r in selected], marker='o', label=label)
    rates=sorted({r['offered_qps'] for r in rows})
    axes[0].plot([0, max(rates)], [0, max(rates)], color='gray', linestyle=':', label='Offered rate')
    for ax, title, ylabel in zip(axes, ('Achieved throughput', 'Mean TTFT', 'Mean TPOT'), ('req/s', 'seconds', 'ms/token')):
        ax.set(title=title, xlabel='Offered QPS', ylabel=ylabel)
        ax.set_xticks(rates)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('ShareGPT P1024/O16; N128; C128; eager; fixed KV1 vs dynamic KV1-2')
    fig.savefig(root/'qps_sweep.png', dpi=180)
    fig.savefig(root/'qps_sweep.pdf')
    plt.close(fig)


def analyze(root, draw=False):
    protocol = load(root/'sweep_protocol.json')
    rates=protocol['new_qps']+protocol['reused_qps']
    assert rates and len(rates)==len(set(rates))
    assert protocol['qps']==sorted(rates)
    assert all(q in (1,2,4,8,10,20,30) for q in protocol['new_qps'])
    assert set(protocol['reused_qps']) <= {30}
    assert protocol['fixture_sha256'] == FIXTURE_SHA
    rows, cases, checks = [], [], []
    for rate in protocol['qps']:
        origin = Path(protocol['roots'][str(rate)])
        checked = verify_rate(origin/'vllm', rate)
        samples=checked['protocol']['samples_per_arm']
        assert samples==(protocol.get('samples_per_new_arm',2) if rate in protocol['new_qps'] else 2)
        assert load(origin/'inputs/capacity_certificate.json') == protocol['capacity_certificate']
        cases.extend(checked['cases']); checks.extend(checked['output_checks'])
        for arm, label in LABELS.items():
            selected = [c for c in checked['cases'] if c['arm'] == arm]
            assert len(selected) == samples
            row = {'offered_qps': rate, 'arm': arm, 'label': label,
                   **{k: statistics.fmean(c[k] for c in selected) for k in METRICS},
                   'samples': samples, 'source': 'reused' if rate in protocol['reused_qps'] else 'new',
                   'qps_sample1': selected[0]['actual_qps'], 'qps_sample2': selected[1]['actual_qps'] if samples==2 else None}
            if arm.startswith('adaptive_'):
                row.update({'max_active_kv_gib': max(c['max_active_kv_gib'] for c in selected),
                            'max_running_requests': max(c['max_running_requests'] for c in selected),
                            'max_waiting_requests': max(c['max_waiting_requests'] for c in selected),
                            'growth_counts': [c['growth_count'] for c in selected],
                            'shrink_counts': [c['shrink_count'] for c in selected],
                            'used_tiers': sorted(set(t for c in selected for t in c['used_tiers'])),
                            **{k: statistics.fmean(c[k] for c in selected) for k in
                               ('allocation_failures', 'preemptions', 'copy_steps', 'remote_steps')}})
            rows.append(row)
    status = 'ok_with_output_drift' if any(c['full_output_differences'] for c in checks) else 'ok'
    samples=protocol.get('samples_per_new_arm',2)
    new_processes=len(protocol['new_qps'])*4*samples
    reused_processes=len(protocol['reused_qps'])*8
    result = {'status': status, 'new_processes': new_processes, 'reused_processes': reused_processes,
              'new_successful_requests': new_processes*128, 'total_successful_requests': len(cases)*128,
              'fixture_sha256': FIXTURE_SHA, 'protocol': protocol,
              'rows': rows, 'cases': cases, 'output_checks': checks,
              'strict_output_equivalence_verified': status == 'ok'}
    (root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    # Include all columns for Native rows, which have no dynamic-policy telemetry.
    fields = list(dict.fromkeys(k for row in rows for k in row))
    write_csv(root/'summary.csv', [{k: row.get(k, '') for k in fields} for row in rows])
    write_csv(root/'output_checks.csv', checks)
    new_label='/'.join(map(str,protocol['new_qps']))
    reuse_label=('，QPS='+ '/'.join(map(str,protocol['reused_qps']))+'复用已核验的上一轮相同配置结果') if protocol['reused_qps'] else ''
    lines = ['# P1024：固定KV1与动态KV1→2的四机制QPS扫描', '',
             f'QPS={new_label}为本轮新测{reuse_label}；未重新运行KTransformers。每个新测机制每档{samples}个独立进程。新增{new_processes}进程、{new_processes*128}请求；含复用数据共{len(cases)}进程、{len(cases)*128}请求。', '',
             'DeepSeek-V2-Lite-Chat BF16；同一128条ShareGPT、1024输入/16 greedy输出、ignore EOS；C128、8192-token调度预算、max_model_len=1088、TRITON_MLA、混合调度、关闭prefix cache、eager、无CUDA Graph。仅更改到达速率，到达时间i/QPS；没有额外真实conditioning，计时包含排队、排空、真实扩缩容，不是纯稳态。', '',
             '三个固定组均为34,480可用KV tokens（2155可用块，约1GiB）。AdaptMoE固定和动态均预留相同2GiB managed虚拟池；固定仅K1驻留/可分配，动态允许K1=2155与K2=4313可用块，保留80%高水位、4步最小驻留、3步低压确认和物理迁移/驻留审计。按token选择路径仍为PARK≤16、RESUME≥128；不强制任何QPS发生扩容或降档。', '',
             'AdaptMoE启动cold8：26层各卸载8个专家、3,598,712,832B；升K2配套cold11：4,948,230,144B。Native保持整Parameter机制，UVA/Copy同7个Parameter、3,690,987,520B，比cold8多2.56%。这是配套KV与专家驻留配置的系统比较。', '',
             ('每档执行UVA→Copy→固定→动态。每组仅一次进程，未重复。' if samples==1 else '每档执行UVA→Copy→固定→动态→动态→固定→Copy→UVA。表中各指标为两进程指标的算术均值。')+'单进程实际QPS=128/首个计划到达到最后请求完成的时间；不排除尾部排空时间。', '',
             '| 提供QPS | 机制 | 实际QPS | TTFT mean s | TPOT mean ms | E2E mean s |',
             '|---:|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['offered_qps']} | {r['label']} | {r['actual_qps']:.3f} | {r['ttft_mean_ms']/1000:.3f} | {r['tpot_mean_ms']:.2f} | {r['e2e_mean_ms']/1000:.3f} |")
    lines += ['', '## 动态相对固定AdaptMoE', '',
              '| 提供QPS | QPS变化 | TTFT变化 | TPOT变化 | E2E变化 |',
              '|---:|---:|---:|---:|---:|']
    for rate in protocol['qps']:
        f = next(r for r in rows if r['offered_qps'] == rate and r['arm'] == 'adaptive_fixed_1g')
        d = next(r for r in rows if r['offered_qps'] == rate and r['arm'] == 'adaptive_dynamic')
        deltas = [(d[k]/f[k]-1)*100 for k in ('actual_qps', 'ttft_mean_ms', 'tpot_mean_ms', 'e2e_mean_ms')]
        lines.append(f"| {rate} | "+' | '.join(f'{v:+.2f}%' for v in deltas)+' |')
    lines += ['', '## 容量与访问路径', '',
              '| QPS | 机制 | KV峰值GiB | 最大运行/等待 | 各次升/降档数 | resize mean ms | KV分配失败 mean | copy/remote step mean | forward H2D GiB mean |',
              '|---:|---|---:|---:|---|---:|---:|---:|---:|']
    for r in rows:
        if not r['arm'].startswith('adaptive_'): continue
        lines.append(f"| {r['offered_qps']} | {r['label']} | {r['max_active_kv_gib']:.3f} | {r['max_running_requests']}/{r['max_waiting_requests']} | {r['growth_counts']}/{r['shrink_counts']} | {r['resize_wall_ms']:.2f} | {r['allocation_failures']:.1f} | {r['copy_steps']:.1f}/{r['remote_steps']:.1f} | {r['expert_h2d_gib']:.3f} |")
    lines += ['', 'KV分配失败为尝试次数，不代表请求最终失败。forward H2D不含KV/专家档位迁移；所有真实迁移耗时均保留。未做CUDA分时剖析，不将延迟变化单独归因于搬运或控制面。', '',
              '## 逐进程指标', '',
              '| QPS | 机制 | 样本 | 实际QPS | TTFT s | TPOT ms | E2E s |',
              '|---:|---|---:|---:|---:|---:|---:|']
    for c in sorted(cases, key=lambda c: (c['offered_qps'], list(LABELS).index(c['arm']), c['sample'])):
        lines.append(f"| {c['offered_qps']} | {LABELS[c['arm']]} | {c['sample']} | {c['actual_qps']:.6f} | {c['ttft_mean_ms']/1000:.6f} | {c['tpot_mean_ms']:.6f} | {c['e2e_mean_ms']/1000:.6f} |")
    lines += ['', '## 输出一致性', '',
              '所有完整输出和首token均比较，不因请求成功而假定数值等价。存在输出差异时，结果属于性能诊断，不能作为通过严格数值等价验证的排名。完整逐对比计数见output_checks.csv与summary.json；每档一或两个进程样本，不建立统计显著性。', '',
              '| QPS | 对比 | 完整输出不同/128 | 首token不同/128 |', '|---:|---|---:|---:|']
    for c in checks:
        lines.append(f"| {c['offered_qps']} | {c['comparison']} | {c['full_output_differences']} | {c['first_token_differences']} |")
    lines += ['', f'原始artifact：{root}', '']
    (root/'REPORT.md').write_text('\n'.join(lines))
    if draw:
        try:
            plot(root, rows)
        except ModuleNotFoundError as exc:
            if exc.name != 'matplotlib':
                raise
            (root/'plot_status.json').write_text(json.dumps({
                'status': 'skipped', 'reason': 'Optional matplotlib dependency is unavailable; all measured data are saved in CSV/JSON/Markdown.'
            }, indent=2)+'\n')
    report_stem=protocol.get('report_stem','P1024_KV1_QPS4_8_10_20_30_FOUR_MECHANISMS_20261024')
    assert Path(report_stem).name==report_stem and report_stem not in ('','.', '..')
    for suffix, source in (('md', 'REPORT.md'), ('json', 'summary.json'), ('csv', 'summary.csv')):
        shutil.copy2(root/source, HERE/f'{report_stem}.{suffix}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--expected-qps', type=float)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    try:
        if args.output:
            rate = args.expected_qps or float(os.environ['KV1_SWEEP_EXPECTED_QPS'])
            result = verify_rate(args.root.resolve(), rate)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            args.report.write_text('# Four-arm KV1 verification\n\n'+json.dumps(result, indent=2)+'\n')
        else:
            result = analyze(args.root.resolve(), args.plot)
    except BaseException as exc:
        if args.output:
            args.output.write_text(json.dumps({'status': 'error', 'errors': [f'{type(exc).__name__}: {exc}']}, indent=2)+'\n')
        raise
    print(json.dumps({'status': result['status'], 'cases': len(result['cases']),
                      'rows': result.get('rows')}, indent=2))


if __name__ == '__main__':
    main()
