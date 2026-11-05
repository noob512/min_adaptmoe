#!/usr/bin/env python3
"""Summarize fresh C128 fixed 1/2 GiB KV measurements."""
import argparse
import csv
import json
import statistics
from pathlib import Path

from verify_c128_fixed_kv import verify, BLOCK_BYTES


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    root=a.root.resolve();here=Path(__file__).resolve().parent
    historical_path=here/'P512_N128_C128_FIXED_VS_DYNAMIC_KV_20261024.json'
    historical=json.loads(historical_path.read_text())['rows']
    rows=[];cases=[];checks=[];comparisons=[]
    for rate in (4,8,10,20,30):
        v=verify(root/f'qps{rate}');cases+=v['cases']
        checks.extend(dict(x,offered_qps=rate) for x in v['output_checks'])
        pair=[]
        for gib in (1,2):
            selected=[c for c in v['cases'] if c['kv_gib']==gib]
            r={'offered_qps':rate,'kv_gib':gib,'samples':len(selected),
               **{k:statistics.fmean(c[k] for c in selected) for k in ('actual_qps','ttft_mean_ms','tpot_mean_ms','e2e_mean_ms')},
               'actual_qps_min':min(c['actual_qps'] for c in selected),
               'actual_qps_max':max(c['actual_qps'] for c in selected),
               'pool_blocks_including_null':selected[0]['pool_blocks_including_null'],
               'max_running_requests':max(c['max_running_requests'] for c in selected),
               'max_waiting_requests':max(c['max_waiting_requests'] for c in selected),
               'max_active_blocks':max(c['max_active_blocks'] for c in selected),
               'max_active_kv_gib':max(c['max_active_blocks'] for c in selected)*BLOCK_BYTES/2**30,
               'preemptions':sum(c['preemptions'] for c in selected)}
            rows.append(r);pair.append(r)
        first,second=pair
        comparisons.append({'offered_qps':rate,
                            'fixed2_vs_fixed1_qps_percent':(second['actual_qps']/first['actual_qps']-1)*100,
                            'fixed2_vs_fixed1_ttft_percent':(second['ttft_mean_ms']/first['ttft_mean_ms']-1)*100,
                            'fixed2_vs_fixed1_tpot_percent':(second['tpot_mean_ms']/first['tpot_mean_ms']-1)*100})
    status='ok_with_output_drift' if any(x['full_output_differences'] for x in checks) else 'ok'
    result={'status':status,'errors':[],'rows':rows,'comparisons':comparisons,'cases':cases,
            'output_checks':checks,'historical_c128_reference':{'path':str(historical_path),'rows':historical},
            'initialization_note':'Both fresh fixed arms temporarily use the existing all-remote startup topology for kernel autotuning, then restore all hot GPU weights and dual page tables before sampler warmup and measurement. Manual fault mode2 is retained. Failed attempts01/02 and mode1 diagnostic are excluded.'}
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (root/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# C128：固定 1 GiB 与 2 GiB KV 实测','',
           '与上一轮相同的128条ShareGPT、512输入/16输出，DeepSeek-V2-Lite-Chat BF16，C128，TRITON_MLA，8192-token调度预算，混合调度，PARK≤16/RESUME≥128，无CUDA Graph。两组均为固定非managed KV，关闭动态池、自动档位与独立专家换入换出，专家卸载均为cold11、4,948,230,144 B。','',
           '每个QPS两组各2个独立进程，1G→2G→2G→1G顺序，表中为两次均值。共20进程、2560请求、40960输出token。正常初始化后直接测量，无额外conditioning，不宣称纯稳态。QPS含排空，TTFT含排队与准入延迟。','',
           '| 提供QPS | 固定KV GiB | 实际QPS | TTFT mean s | TPOT mean ms | E2E mean s | 峰值运行并发 | 峰值等待 | 活跃KV峰值GiB |',
           '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['offered_qps']} | {r['kv_gib']} | {r['actual_qps']:.3f} | {r['ttft_mean_ms']/1000:.3f} | {r['tpot_mean_ms']:.2f} | {r['e2e_mean_ms']/1000:.3f} | {r['max_running_requests']} | {r['max_waiting_requests']} | {r['max_active_kv_gib']:.3f} |")
    lines+=['','## 固定2G相对固定1G','', '| 提供QPS | QPS变化 | TTFT变化 | TPOT变化 |','|---:|---:|---:|---:|']
    for c in comparisons:
        lines.append(f"| {c['offered_qps']} | {c['fixed2_vs_fixed1_qps_percent']:+.2f}% | {c['fixed2_vs_fixed1_ttft_percent']:+.2f}% | {c['fixed2_vs_fixed1_tpot_percent']:+.2f}% |")
    high=[r for r in rows if r['offered_qps']==30]
    delta=comparisons[-1]
    lines+=['',
            f"QPS=30下，固定2G相对1G的实际吞吐变化{delta['fixed2_vs_fixed1_qps_percent']:+.2f}%，TTFT变化{delta['fixed2_vs_fixed1_ttft_percent']:+.2f}%，TPOT变化{delta['fixed2_vs_fixed1_tpot_percent']:+.2f}%。1G峰值运行{high[0]['max_running_requests']}个请求、等待{high[0]['max_waiting_requests']}个，活跃KV {high[0]['max_active_blocks']}/{high[0]['pool_blocks_including_null']-1} blocks；2G峰值运行{high[1]['max_running_requests']}个、等待{high[1]['max_waiting_requests']}个。该高负载结果支持KV容量影响准入并发、排队和吞吐；不能把它解释为单token执行全面加速。"]
    lines+=['','## 与上一轮C128参考对照（历史值，未重测）','',
            '| 提供QPS | 历史固定512MiB QPS | 本轮固定1G QPS | 本轮固定2G QPS | 历史dynamic初始K2 QPS |',
            '|---:|---:|---:|---:|---:|']
    for rate in (4,8,10,20,30):
        old1=next(r for r in historical if r['offered_qps']==rate and r['variant']=='fixed')
        old2=next(r for r in historical if r['offered_qps']==rate and r['variant']=='dynamic')
        fresh=[r for r in rows if r['offered_qps']==rate]
        lines.append(f"| {rate} | {old1['actual_qps']:.3f} | {fresh[0]['actual_qps']:.3f} | {fresh[1]['actual_qps']:.3f} | {old2['actual_qps']:.3f} |")
    lines+=['','历史dynamic始终停留在K2/cold11，未发生扩缩容。固定2G使用4315个物理blocks（含null），历史dynamic受页对齐约束，K2 target4313且有backup/safety预留、managed分配器与控制器，因此不能当作只开关控制器的严格消融。','',
            '## 初始化故障与测量边界','',
            'attempt01在启动dummy warmup前后触发非法地址，无真实请求。独立mode1诊断显示：kernel warmup前热专家GPU驻留23,841,472,512 B，之后减少403,177,472 B（384.5 MiB），说明最大token数autotune会驱逐热专家。attempt02尝试在kernel warmup后修复，但同步时已发现该forward内发生的异步CUDA错误，因此同样不纳入结果。','',
            '正式结果来自attempt03：两组均仅在kernel warmup期间调用项目已有的临时全远程拓扑；保留native workspace cache，再恢复热专家、刷新双页表及RESUME。逐进程startup_residency_audit.json证明进入sampler dummy128前所有热专家重新GPU驻留。正式运行保持manual_migration_fault_mode=2，不依赖隐式迁移。初始化操作不在性能计时内；与历史dynamic比较仍涉及初始化流程差异。','',
            '## 输入、输出与运行检查','',
            f"全部输入hash、计划到达时间、输出长度及每进程GPU/内核状态已校验。共1280组跨KV配置请求对比，完整输出不同{sum(x['full_output_differences'] for x in checks)}组，首token不同{sum(x['first_token_differences'] for x in checks)}组。保留原始token序列，不假定差异原因或宣称严格输出等价。每组仅两次进程，不宣称统计显著性。",'',
            '## 逐进程结果','', '| QPS | 顺序 | 固定KV GiB | 实际QPS | TTFT s | TPOT ms | 峰值运行并发 | 峰值等待 | 抢占次数 |',
            '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in cases:
        lines.append(f"| {c['offered_qps']} | {c['ordinal']} | {c['kv_gib']} | {c['actual_qps']:.6f} | {c['ttft_mean_ms']/1000:.6f} | {c['tpot_mean_ms']:.6f} | {c['max_running_requests']} | {c['max_waiting_requests']} | {c['preemptions']} |")
    lines+=['',f'原始artifact：{root}','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'status':status,'rows':rows,'comparisons':comparisons,'report':str(root/'REPORT.md')},indent=2))


if __name__=='__main__':
    main()
