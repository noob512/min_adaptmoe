#!/usr/bin/env python3
"""Join automatic K2/4/6 measurements with the frozen fixed-KV comparison."""
import argparse
import csv
import json
import statistics
from pathlib import Path

from verify_dynamic_kv_p512 import verify
from summarize_ktransformers_p512_scan import baseline_dirs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    here=Path(__file__).resolve().parent
    baseline=json.loads((here/'P512_N128_QPS4_8_10_20_30_SUMMARY_20261023.json').read_text())['rows']
    rows=[]
    cases=[]
    outputs=[]
    for rate in (4,8,10,20,30):
        check=verify(root/f'qps{rate}')
        selected=check['cases']
        cases+=selected
        fixed=next(r for r in baseline if r['offered_qps']==rate and r['arm']=='adaptive')
        row={'offered_qps':rate,'samples':2,'fixed_actual_qps':fixed['actual_qps'],
             'fixed_ttft_mean_ms':fixed['ttft_mean_ms'],'fixed_tpot_mean_ms':fixed['tpot_mean_ms'],
             'fixed_e2e_mean_ms':fixed['e2e_mean_ms'],
             **{k:statistics.fmean(c[k] for c in selected) for k in ('actual_qps','ttft_mean_ms','tpot_mean_ms','e2e_mean_ms')},
             'used_targets':sorted({t for c in selected for t in c['used_targets']}),
             'resize_count':sum(len(c['changes']) for c in selected),
             'max_active_blocks':max(c['max_active_blocks'] for c in selected),
             'max_required_blocks':max(c['max_required_blocks'] for c in selected),
             'preemptions':sum(c['preemptions'] for c in selected),
             'observation_cpu_ms_mean':statistics.fmean(c['observation_cpu_ms'] for c in selected)}
        row['qps_delta_percent']=(row['actual_qps']/fixed['actual_qps']-1)*100
        row['ttft_delta_percent']=(row['ttft_mean_ms']/fixed['ttft_mean_ms']-1)*100
        row['tpot_delta_percent']=(row['tpot_mean_ms']/fixed['tpot_mean_ms']-1)*100
        rows.append(row)
        for case,ref in zip(selected,baseline_dirs(fixed)):
            new=json.loads((Path(case['directory'])/'result.json').read_text())['requests']
            old=json.loads((ref/'result.json').read_text())['requests']
            assert [r['prompt_token_ids_sha256'] for r in new]==[r['prompt_token_ids_sha256'] for r in old]
            outputs.append({'offered_qps':rate,'sample':case['sample'],'requests':128,
                            'full_output_differences':sum(a['output_token_ids']!=b['output_token_ids'] for a,b in zip(new,old)),
                            'first_token_differences':sum(a['output_token_ids'][0]!=b['output_token_ids'][0] for a,b in zip(new,old)),
                            'reference':str(ref/'result.json')})
    result={'status':'ok_with_output_drift' if any(o['full_output_differences'] for o in outputs) else 'ok',
            'errors':[],'rows':rows,'cases':cases,'output_comparisons':outputs,
            'tier_manifest':json.loads((root/'inputs/manifest.json').read_text()),
            'historical_baseline':str(here/'P512_N128_QPS4_8_10_20_30_SUMMARY_20261023.json')}
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (root/'summary.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text=['# 相同P512/N128 trace：AdaptMoE自动K2/K4/K6补测','',
          '同一份冻结128条ShareGPT、512输入/16输出、DeepSeek-V2-Lite-Chat BF16、C16、8192-token预算、TRITON_MLA、混合调度、PARK≤16/RESUME≥128、无Graph。每档两个独立进程，无额外conditioning，计时与原实验一致。','',
          '固定KV参考为已有512 MiB结果，动态KV为6 GiB managed虚拟池、初始约2 GiB物理驻留，可自动调整至4/6 GiB。动态使用现有80%高水位、相邻档事务和安全缩容；不强制升档。K2/cold11与历史固定臂使用完全相同的专家坐标和4,948,230,144 B主权重卸载。','',
          '| 提供QPS | 固定KV实际QPS | 动态KV实际QPS | 吞吐变化 | 固定TTFT s | 动态TTFT s | 固定TPOT ms | 动态TPOT ms |',
          '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['offered_qps']} | {r['fixed_actual_qps']:.3f} | {r['actual_qps']:.3f} | {r['qps_delta_percent']:+.2f}% | {r['fixed_ttft_mean_ms']/1000:.3f} | {r['ttft_mean_ms']/1000:.3f} | {r['fixed_tpot_mean_ms']:.2f} | {r['tpot_mean_ms']:.2f} |")
    text+=['','## 实际档位与压力','',
           '| QPS | 实际使用target blocks | 升降档次数（两进程合计） | 活跃KV峰值blocks | 控制器所需峰值blocks | 抢占次数 |',
           '|---:|---|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['offered_qps']} | {r['used_targets']} | {r['resize_count']} | {r['max_active_blocks']} | {r['max_required_blocks']} | {r['preemptions']} |")
    if all(r['used_targets']==[4313] and not r['resize_count'] for r in rows):
        text+=['','所有正式请求期间均保持K2/cold11，没有K2→K4或K4→K6，也没有降档。已注册三档不等于实际执行三档。本轮没有验证到动态扩容相对固定KV的性能收益。']
    faster=[r['offered_qps'] for r in rows if r['actual_qps']>r['fixed_actual_qps']]
    text+=['',f"动态配置实际QPS高于历史固定值的负载点：{faster}；各点吞吐变化范围为{min(r['qps_delta_percent'] for r in rows):+.2f}%至{max(r['qps_delta_percent'] for r in rows):+.2f}%。这只是本轮与历史值的数值差，不能凭两个进程推断统计显著性。"]
    text+=['','该负载最多16*(512+16)=8,448个同时活跃token，理论KV上界528*497,664=262,766,592 B（250.59 MiB）。历史512 MiB固定KV已能容纳；K2更不易达到约3,451 blocks的80%阈值。提高提供QPS增加等待请求，不能突破C16的运行请求上限。因此不能期待仅增大到达率就自然触发K4/K6或得到动态扩容收益。','',
           'K2/K4/K6计划分别为每层cold11/17/23，主权重卸载4,948,230,144 / 7,647,264,768 / 10,346,299,392 B。未触发档位的GPU切换正确性和性能不算本轮验证。','',
           '历史固定512 MiB使用原生固定KV分配，动态使用managed KV和控制器、plan-class注册等额外路径，且不是本轮同日交错重测的固定对照。若测得差异，应按完整配置解释，不能仅归因于KV容量或控制器。逐步记录直接读取已生成的控制器decision，不重复全池审计；其额外CPU记录时间保留在summary.json。','',
           '正常初始化后直接测量；实际QPS=128/首个计划到达到最后请求完成的时间，含排队和尾部排空。TTFT包含admission及engine内部排队；TPOT包含mixed step与控制面的影响。两独立进程均值不提供严格统计置信。','',
           '## 逐进程结果','', '| QPS | sample | 实际QPS | TTFT s | TPOT ms | 观测记录CPU总ms |', '|---:|---:|---:|---:|---:|---:|']
    for c in cases:
        text.append(f"| {c['offered_qps']} | {c['sample']} | {c['actual_qps']:.6f} | {c['ttft_mean_ms']/1000:.6f} | {c['tpot_mean_ms']:.6f} | {c['observation_cpu_ms']:.3f} |")
    text+=['', 'QPS=4/sample1最后请求有一次约2664.74ms的token间隔，拉低含尾部排空的实际吞吐；相邻控制器decision仍为hold，未发生KV升降档。该样本完整保留，不据此推断动态扩容性能。尚未剖析该间隔的CPU/GPU根因，不将其直接归因于控制器或KV迁移。']
    text+=['','## 与历史固定KV的输出检查','', '| QPS | sample | 完整输出不同/128 | 首token不同/128 |', '|---:|---:|---:|---:|']
    text += [f"| {o['offered_qps']} | {o['sample']} | {o['full_output_differences']} | {o['first_token_differences']} |" for o in outputs]
    text+=['', f"共1280条跨配置请求对比中，{sum(o['full_output_differences'] for o in outputs)}条完整输出不同，{sum(o['first_token_differences'] for o in outputs)}条首token不同。1280次请求均成功完成且输出长度一致，但不宣称严格输出等价，也不把差异未经剖析就认定为正常数值漂移。该限制与历史对照限制一起保留；本轮不能据此建立动态优于固定的机制性结论。输入hash、输出长度和逐请求时刻均保留。每档verification.json及summary.json列出校验结果。",
           '','## 启动修复与失败留档','',
           'attempt-01因未显式enforce_eager=True而误入Graph类初始化；修复后attempt-02在KV物理驻留审计失败。attempt-03记录原有审计，确认27层KV的head/tail边界未按4KiB对齐，驱动查询返回status=-30、queried_bytes=0。每层MLA block为18,432B，单个block不是4KiB整数倍。',
           '', '正式attempt-04将包含null的物理block数向下对齐为偶数：4314/8630/12944；policy targets为4313/8629/12943。6GiB池通过num_gpu_blocks_override=12944对齐，名义预算不变，每档最多少一个block（约0.475MiB）。全部启动审计通过，GPU前缀与CPU尾部独占驻留，未关闭物理迁移或校验。前三次失败均在正式请求前，不计入性能平均。',
           '',f'原始artifact：[目录]({root})。','']
    (root/'REPORT.md').write_text('\n'.join(text))
    print(json.dumps({'status':result['status'],'rows':rows,'report':str(root/'REPORT.md')},indent=2))


if __name__=='__main__':
    main()
