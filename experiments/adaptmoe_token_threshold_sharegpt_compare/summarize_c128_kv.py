#!/usr/bin/env python3
"""Summarize matched C128 fixed/dynamic runs and distinguish C16 history."""
import argparse
import csv
import json
import statistics
from pathlib import Path

from verify_c128_kv import verify


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    root=a.root.resolve();here=Path(__file__).resolve().parent
    old_fixed=json.loads((here/'P512_N128_QPS4_8_10_20_30_SUMMARY_20261023.json').read_text())['rows']
    old_dynamic=json.loads((here/'P512_N128_DYNAMIC_KV_K2_K4_K6_20261024.json').read_text())['rows']
    cases=[];checks=[];rows=[];comparisons=[]
    for rate in (4,8,10,20,30):
        v=verify(root/f'qps{rate}');cases+=v['cases']
        checks+=[dict(c,offered_qps=rate) for c in v['output_checks']]
        per_rate=[]
        for variant in ('fixed','dynamic'):
            selected=[c for c in v['cases'] if c['variant']==variant]
            historical=next(r for r in (old_fixed if variant=='fixed' else old_dynamic)
                            if r['offered_qps']==rate and (variant=='dynamic' or r['arm']=='adaptive'))
            row={'offered_qps':rate,'variant':variant,'samples':2,
                 **{k:statistics.fmean(c[k] for c in selected) for k in ('actual_qps','ttft_mean_ms','tpot_mean_ms','e2e_mean_ms')},
                 'max_running_requests':max(c['max_running_requests'] for c in selected),
                 'max_waiting_requests':max(c['max_waiting_requests'] for c in selected),
                 'max_active_blocks':max(c['max_active_blocks'] for c in selected),
                 'max_active_kv_gib':max(c['max_active_blocks'] for c in selected)*497664/2**30,
                 'preemptions':sum(c['preemptions'] for c in selected),
                 'used_targets':sorted({t for c in selected for t in c['used_targets']}),
                 'resize_count':sum(len(c['changes']) for c in selected),
                 'historical_c16_actual_qps':historical['actual_qps'],
                 'c128_vs_c16_qps_percent':(statistics.fmean(c['actual_qps'] for c in selected)/historical['actual_qps']-1)*100}
            per_rate.append(row);rows.append(row)
        fixed,dynamic=per_rate
        comparisons.append({'offered_qps':rate,'dynamic_vs_fixed_qps_percent':(dynamic['actual_qps']/fixed['actual_qps']-1)*100,
                            'dynamic_vs_fixed_ttft_percent':(dynamic['ttft_mean_ms']/fixed['ttft_mean_ms']-1)*100,
                            'dynamic_vs_fixed_tpot_percent':(dynamic['tpot_mean_ms']/fixed['tpot_mean_ms']-1)*100})
    status='ok_with_output_drift' if any(c['full_output_differences'] for c in checks) else 'ok'
    result={'status':status,'errors':[],'rows':rows,'comparisons':comparisons,'cases':cases,'output_checks':checks,
            'tier_manifest':json.loads((root/'inputs/manifest.json').read_text())}
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (root/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# P512/N128：放开至C128的固定KV/动态KV对照','',
           '同一份128条ShareGPT，512输入/16输出，DeepSeek-V2-Lite-Chat BF16，TRITON_MLA，8192-token调度预算，混合调度，PARK≤16/RESUME≥128，无CUDA Graph。C128等于全trace请求数；每个QPS两臂均为两个独立进程，按固定→动态→动态→固定交错，无额外conditioning。共20进程、2560次请求。','',
           '固定512MiB KV为本轮C128重测；动态为初始K2、可自动K4/K6的managed KV，沿用此前参数和专家配对。固定与动态起始cold11坐标相同，主专家卸载4,948,230,144 B。','',
           '| 提供QPS | 机制 | 实际QPS | TTFT mean s | TPOT mean ms | E2E mean s | 峰值运行并发 | 活跃KV峰值GiB |',
           '|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['offered_qps']} | {r['variant']} | {r['actual_qps']:.3f} | {r['ttft_mean_ms']/1000:.3f} | {r['tpot_mean_ms']:.2f} | {r['e2e_mean_ms']/1000:.3f} | {r['max_running_requests']} | {r['max_active_kv_gib']:.3f} |")
    lines+=['','## 同C128动态相对固定','', '| QPS | QPS变化 | TTFT变化 | TPOT变化 |', '|---:|---:|---:|---:|']
    for c in comparisons:
        lines.append(f"| {c['offered_qps']} | {c['dynamic_vs_fixed_qps_percent']:+.2f}% | {c['dynamic_vs_fixed_ttft_percent']:+.2f}% | {c['dynamic_vs_fixed_tpot_percent']:+.2f}% |")
    lines+=['','## 实际KV调节','', '| QPS | 机制 | 实际target blocks | 升降档总次数 | 抢占次数 | 峰值等待数 |', '|---:|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['offered_qps']} | {r['variant']} | {r['used_targets']} | {r['resize_count']} | {r['preemptions']} | {r['max_waiting_requests']} |")
    dynamic_rows=[r for r in rows if r['variant']=='dynamic']
    if not any(r['resize_count'] for r in dynamic_rows):
        lines+=['','本轮动态KV未实际升降档，始终K2/cold11。因此本轮即使优于固定512MiB，也只证明这组较大初始KV与运行框架的配置收益，不能证明自动扩容带来的收益。']
    else:
        lines+=['','实际升降档逐步记录在summary.json的cases[].changes与每进程result.json的dynamic_kv.controller_steps；依据记录区分K2→K4与后续回落。']
    lines+=['','该trace在C128的理论KV上限为4224 blocks≈1.958GiB。达到K2高水位3451 blocks才会主动增长；K4高水位超出该trace全部请求工作集。因此本轮不能覆盖K4→K6。','',
            '## 与C16历史值比较','', '| QPS | 机制 | C16历史实际QPS | 本轮C128实际QPS | 变化 |', '|---:|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['offered_qps']} | {r['variant']} | {r['historical_c16_actual_qps']:.3f} | {r['actual_qps']:.3f} | {r['c128_vs_c16_qps_percent']:+.2f}% |")
    lines+=['','C16为历史参考；C128固定/动态为本轮交错实测。两臂KV容量及分配器/控制器不同，不能把所有差异都归因于自动控制器。正常初始化后测量，未额外做完整请求预热；不宣称纯稳态或严格统计显著性。实际QPS含尾部排空；TTFT从计划到达起算，包含所有排队；TPOT包含mixed step及控制面时间。','',
            '## 输出检查','',f"共1280组跨配置请求对比，完整输出不同{sum(c['full_output_differences'] for c in checks)}组，首token不同{sum(c['first_token_differences'] for c in checks)}组。输入token hash与输出长度已校验，完整输出差异保留，不假定为正常数值误差或宣称严格输出等价。逐cell校验见verification.json。",'',
            '## 逐进程结果','', '| QPS | 顺序 | 机制 | 实际QPS | TTFT s | TPOT ms | 峰值运行并发 | 观测CPU ms |', '|---:|---:|---|---:|---:|---:|---:|---:|']
    for c in cases:
        lines.append(f"| {c['offered_qps']} | {c['ordinal']} | {c['variant']} | {c['actual_qps']:.6f} | {c['ttft_mean_ms']/1000:.6f} | {c['tpot_mean_ms']:.6f} | {c['max_running_requests']} | {c['observation_cpu_ms']:.3f} |")
    lines+=['',f'原始artifact：{root}','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'status':status,'rows':rows,'comparisons':comparisons,'report':str(root/'REPORT.md')},indent=2))


if __name__=='__main__':
    main()
