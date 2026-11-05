#!/usr/bin/env python3
"""Summarize actual tier transitions for the P1024/C128 ShareGPT replay."""
import argparse
import csv
import json
import statistics
from pathlib import Path
from verify_p1024_dynamic_kv import verify


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    root=a.root.resolve();rows=[];cases=[];checks=[]
    for rate in (4,8,10,20,30):
        v=verify(root/f'qps{rate}');selected=v['cases'];cases+=selected
        checks.append(dict(v['repeat_output_check'],offered_qps=rate))
        rows.append({'offered_qps':rate,'samples':2,
                     **{k:statistics.fmean(c[k] for c in selected) for k in ('actual_qps','ttft_mean_ms','tpot_mean_ms','e2e_mean_ms')},
                     'actual_qps_min':min(c['actual_qps'] for c in selected),'actual_qps_max':max(c['actual_qps'] for c in selected),
                     'used_tiers':sorted({t for c in selected for t in c['used_tiers']}),
                     'growth_counts':[c['growth_count'] for c in selected],'shrink_counts':[c['shrink_count'] for c in selected],
                     'max_running_requests':max(c['max_running_requests'] for c in selected),
                     'max_waiting_requests':max(c['max_waiting_requests'] for c in selected),
                     'max_active_kv_gib':max(c['scheduler_max_active_kv_gib'] for c in selected),
                     'max_active_blocks':max(c['scheduler_max_active_blocks'] for c in selected),
                     'preemptions':sum(c['preemptions'] for c in selected)})
    result={'status':'ok_with_output_drift' if any(c['full_output_differences'] for c in checks) else 'ok',
            'errors':[],'rows':rows,'cases':cases,'repeat_output_checks':checks,
            'fixture_manifest':json.loads((root/'fixture_manifest.json').read_text()),
            'tier_manifest':json.loads((root/'inputs/manifest.json').read_text())}
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (root/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# P1024/N128/C128：自动KV是否升档','',
           '同一批128条ShareGPT请求，逐请求保留原前512个token并用ShareGPT原文扩展为1024输入，输出仍16 token；C128，DeepSeek-V2-Lite-Chat BF16，TRITON_MLA，混合调度，8192-token预算，PARK≤16/RESUME≥128，关闭CUDA Graph。仅测动态KV，每档QPS两个独立进程，共10进程、1280请求。未额外conditioning，不宣称纯稳态；实际QPS含排空，TTFT含排队。','',
           '保持K2/K4/K6、cold11/17/23配对、80%高水位、物理迁移与原控制器安全检查；没有按时间强制升档。每次实际resize的结果与起止时间保存在各进程resize_transactions.json。','',
           '| 提供QPS | 实际QPS | TTFT mean s | TPOT mean ms | E2E mean s | 实际使用档位 | 两次升档数 | 两次降档数 | 活跃KV峰值GiB |',
           '|---:|---:|---:|---:|---:|---|---|---|---:|']
    for r in rows:
        lines.append(f"| {r['offered_qps']} | {r['actual_qps']:.3f} | {r['ttft_mean_ms']/1000:.3f} | {r['tpot_mean_ms']:.2f} | {r['e2e_mean_ms']/1000:.3f} | {','.join(r['used_tiers'])} | {r['growth_counts']} | {r['shrink_counts']} | {r['max_active_kv_gib']:.3f} |")
    lines+=['','## 实际升降档记录','',
            '| QPS | 进程 | Step | 转换 | 观测时间s | resize wall ms | 转换后active blocks |',
            '|---:|---:|---:|---|---:|---:|---:|']
    for c in cases:
        for x in c['changes']:
            wall=f"{x['transaction_wall_ms']:.2f}" if x['transaction_wall_ms'] is not None else '见transactions'
            lines.append(f"| {c['offered_qps']} | {c['sample']} | {x['step']} | {x['old_tier']}→{x['new_tier']} | {x['observed_seconds']:.3f} | {wall} | {x['active_blocks_after']} |")
    if not any(c['growth_count'] for c in cases):
        lines+=['','本轮未观察到成功升档。配置允许K2/K4/K6不能当作实际使用了三档；应根据实际活跃blocks与控制器理由解释。']
    lines+=['','工作集理论上限8320 blocks≈3.856GiB；K2/K4高水位分别3451/6904 blocks，均按ceil(target×0.8)计算。实际是否达到高水位由运行并发、合批及完成速率共同决定。','',
            '## 运行并发与重复性','', '| QPS | 实际QPS范围 | 峰值运行请求 | 峰值等待 | 抢占次数 |', '|---:|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['offered_qps']} | {r['actual_qps_min']:.3f}–{r['actual_qps_max']:.3f} | {r['max_running_requests']} | {r['max_waiting_requests']} | {r['preemptions']} |")
    lines+=['','每个QPS只有两个独立进程，不宣称统计显著性。P512历史结果使用不同长度的输入；本轮未重测固定KV，不能据此声称动态KV优于相同容量固定KV。','',
            f"两次重复共640组请求，完整输出不同{sum(c['full_output_differences'] for c in checks)}组、首token不同{sum(c['first_token_differences'] for c in checks)}组。输入hash、请求长度及到达顺序已校验；输出差异保留，不假定严格数值等价。",'',f'原始artifact：{root}','']
    if (root/'reused_samples.json').exists():
        provenance=json.loads((root/'reused_samples.json').read_text())
        lines+=['## 错误修复与样本来源','',
                '首次QPS20升档失败于驻留审计：18KiB MLA block从block1开始的范围未按4KiB对齐，驱动返回NV_ERR_INVALID_ADDRESS(-30)。修正为与初始化相同的含null block0完整GPU前缀；仍保留物理迁移、专家协调及完整驻留检查。失败样本不计入性能结果。','',
                f"QPS4/8/10沿用{provenance['source_root']}的6个有效进程，经重验确认无resize事务；这些样本不经过被修复的分支。QPS20/30使用修复后的代码各重跑2个进程。当前目录reused_samples.json及qps目录链接记录来源。",'']
    (root/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'status':result['status'],'rows':rows,'report':str(root/'REPORT.md')},indent=2))


if __name__=='__main__':main()
