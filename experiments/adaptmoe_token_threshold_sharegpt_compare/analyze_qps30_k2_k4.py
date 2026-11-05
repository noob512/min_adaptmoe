#!/usr/bin/env python3
"""Summarize a fresh fixed-K2 versus capped K2/K4 QPS30 experiment."""
import argparse
import hashlib
import json
import shutil
import statistics
from pathlib import Path

from analyze_qps30_kv_pressure import analyze_case, plot, write_csv
from verify_qps30_k2_k4 import verify

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE / 'artifacts/p1024-qps30-fixed-vs-dynamic-kv-pressure-detail-20261024-attempt-01/summary.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    checked = verify(root / 'qps30')
    cap = json.loads((root / 'inputs/tier_cap.json').read_text())
    assert cap['max_gpu_kv_gib'] == 4 and cap['allowed_target_blocks'] == [4313,8629]
    cases, timeline, failures, requests = [], [], [], []
    for case in checked['cases']:
        detailed, steps, rejects, reqs = analyze_case(case)
        cases.append(detailed)
        timeline.extend(steps)
        failures.extend(rejects)
        requests.extend(reqs)
    rows = []
    for arm in ('adaptive_fixed', 'adaptive_dynamic'):
        selected = [c for c in cases if c['arm'] == arm]
        assert len(selected) == 2
        fields = ('actual_qps','ttft_mean_ms','tpot_mean_ms','e2e_mean_ms','elapsed_s',
                  'allocation_failures','waiting_allocation_failures','running_allocation_failures',
                  'preemptions','waiting_steps','deferred_requests','expert_h2d_gib','copy_steps','remote_steps',
                  'controller_including_resize_ms','controller_outside_resize_ms','resize_wall_ms',
                  'admission_delay_ms','engine_queueing_ms','queueing_ms','scheduled_to_first_token_ms')
        rows.append({'arm':arm,**{key:statistics.fmean(c[key] for c in selected) for key in fields},
                     'max_running_requests':max(c['max_running_requests'] for c in selected),
                     'max_waiting_requests':max(c['max_waiting_requests'] for c in selected),
                     'max_active_kv_gib':max(c['max_active_kv_gib'] for c in selected),
                     'used_tiers':sorted({tier for c in selected for tier in c['used_tiers']}),
                     'growth_counts':[c['growth_count'] for c in selected],
                     'shrink_counts':[c['shrink_count'] for c in selected],
                     'waiting_steps_by_cause':{key:statistics.fmean(c['waiting_steps_by_cause'].get(key,0) for c in selected)
                                               for key in ('kv_allocation_failure','token_budget_exhausted','max_num_seqs','preemption','other')}})
    previous = json.loads(PREVIOUS.read_text()) if PREVIOUS.exists() else None
    previous_dynamic = next(r for r in previous['rows'] if r['arm']=='adaptive_dynamic') if previous else None
    fixed, dynamic = rows
    changes = {key:(dynamic[key]/fixed[key]-1)*100 for key in ('actual_qps','ttft_mean_ms','tpot_mean_ms','e2e_mean_ms')}
    vs_previous = {key:(dynamic[key]/previous_dynamic[key]-1)*100 for key in (
        'actual_qps','ttft_mean_ms','tpot_mean_ms','e2e_mean_ms','expert_h2d_gib')} if previous_dynamic else None
    result = {'status':checked['status'],'valid_processes':4,'successful_requests':512,
              'rows':rows,'cases':cases,'output_checks':checked['output_checks'],'tier_cap':cap,
              'dynamic_vs_fresh_fixed_percent':changes,'dynamic_vs_previous_k6_percent':vs_previous,
              'previous_k6_reference':{'path':str(PREVIOUS),'row':previous_dynamic} if previous_dynamic else None,
              'scope':'Only QPS30; capped K2/K4; two fresh processes per arm. Prior K6 shown only as a historical reference.'}
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    write_csv(root/'summary.csv',rows)
    write_csv(root/'scheduler_timeline.csv',timeline)
    write_csv(root/'request_latency.csv',requests)
    if failures:
        write_csv(root/'allocation_failures.csv',failures)
    if args.plot:
        plot(root,timeline,cases)
    lines = ['# QPS30：固定K2与动态K2/K4（封顶4GiB）', '',
             '同一ShareGPT P1024/O16、128请求；DeepSeek-V2-Lite-Chat BF16、TRITON_MLA、C128、8192-token预算、PARK≤16/RESUME≥128、混合调度、eager、无Graph、无prefix cache。提供QPS=30，计划到达i/30；固定→动态→动态→固定，共4个新进程、512次请求。无额外conditioning；指标包含排队、排空和真实扩容，不是纯稳态。', '',
             '本轮只将控制器allowed targets改为4313/8629（K2/K4），maximum改为8629；80%高水位、16backup+16safety、4-step驻留、3-step低压计数、物理迁移/驻留检查及token访问路径均保持。固定与动态运行配置仍仅相差自动控制开关和观测标签。保留相同6GiB VA预留与专家类初始化，固定组GPU KV锁定K2，动态组GPU最高K4。cold23只保留在初始化注册表，不允许成为运行档位；已逐事务及逐forward核对。', '',
             '## 本轮测量（两次均值）', '',
             '| 配置 | 实际QPS | TTFT s | TPOT ms | E2E s | 总时长s |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        label = '固定K2' if row['arm']=='adaptive_fixed' else '动态K2→K4'
        lines.append(f"| {label} | {row['actual_qps']:.3f} | {row['ttft_mean_ms']/1000:.3f} | {row['tpot_mean_ms']:.2f} | {row['e2e_mean_ms']/1000:.3f} | {row['elapsed_s']:.3f} |")
    lines += ['',f"动态相对本轮固定K2：吞吐{changes['actual_qps']:+.2f}%，TTFT{changes['ttft_mean_ms']:+.2f}%，TPOT{changes['tpot_mean_ms']:+.2f}%。", '',
              '## KV与排队', '',
              '| 配置 | 实际档位 | 峰值活跃KV GiB | 最大运行/等待 | 抢占 | KV分配失败/进程 | KV阻塞step | token预算阻塞step | 升档/降档次数（两次） |',
              '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        label = '固定K2' if row['arm']=='adaptive_fixed' else '动态K2/K4'
        reasons = row['waiting_steps_by_cause']
        lines.append(f"| {label} | {'→'.join(row['used_tiers'])} | {row['max_active_kv_gib']:.3f} | {row['max_running_requests']}/{row['max_waiting_requests']} | {row['preemptions']:.0f} | {row['allocation_failures']:.0f} | {reasons['kv_allocation_failure']:.0f} | {reasons['token_budget_exhausted']:.0f} | {row['growth_counts']} / {row['shrink_counts']} |")
    lines += ['', '峰值取两个进程最大；分配失败是尝试次数，全部请求最终成功。容量不足使新请求等待，只有RUNNING追加KV失败才触发抢占。P1024/O16每请求最多65块，固定K2可容纳66个完整活跃请求，Decode追加0/1块全部成功。扩容后的残留等待按逐step分配结果、剩余token预算及C限制核对，原始分类见scheduler_timeline.csv。', '',
              '## 实际事务', '',
              '| 样本 | Step | KV target | 专家类 | resize ms | 原因 |',
              '|---:|---:|---|---|---:|---|']
    for case in cases:
        for transaction in case['transactions']:
            lines.append(f"| {case['sample']} | {transaction['controller_step']} | {transaction['old_target_blocks']}→{transaction['applied_target_blocks']} | {transaction['active_plan_class_before']}→{transaction['active_plan_class_after']} | {transaction['wall_ms']:.3f} | {transaction['reason']} |")
    lines += ['', '## 搬运与控制开销', '',
              '| 配置 | copy/remote steps | 专家forward H2D GiB | resize ms | 控制器wall扣除resize ms |',
              '|---|---:|---:|---:|---:|']
    for row in rows:
        label = '固定K2' if row['arm']=='adaptive_fixed' else '动态K2/K4'
        lines.append(f"| {label} | {row['copy_steps']:.0f}/{row['remote_steps']:.0f} | {row['expert_h2d_gib']:.3f} | {row['resize_wall_ms']:.2f} | {row['controller_outside_resize_ms']:.2f} |")
    lines += ['', 'cold11/cold17每个copy step分别4.608/7.122GiB；H2D仅统计显式forward专家copy，未包含KV迁移和专家类切换迁移。控制器wall包含其同步等待，扣除嵌套resize不等于全部系统控制面或纯CPU成本。没有CUDA剖析，不能给出DMA与计算的精确毫秒分摊。', '',
              '## TTFT分解（ms）', '',
              '| 配置 | 计划到达→引擎接收 | 引擎接收→首次调度 | 首次调度→首token |',
              '|---|---:|---:|---:|']
    for row in rows:
        label = '固定K2' if row['arm']=='adaptive_fixed' else '动态K2/K4'
        lines.append(f"| {label} | {row['admission_delay_ms']:.2f} | {row['engine_queueing_ms']:.2f} | {row['scheduled_to_first_token_ms']:.2f} |")
    lines += ['', '第一项包含同步离线replay在engine.step返回后接收已到时请求的延迟，与前轮口径一致；不能全部视为vLLM内部waiting。', '']
    if previous_dynamic:
        lines += ['## 与上一轮K2/K4/K6的历史参考对照', '',
                  '| 配置 | 实际QPS | TTFT s | TPOT ms | H2D GiB | resize ms |',
                  '|---|---:|---:|---:|---:|---:|']
        for label, row in (('上一轮K2/K4/K6',previous_dynamic),('本轮K2/K4',dynamic)):
            lines.append(f"| {label} | {row['actual_qps']:.3f} | {row['ttft_mean_ms']/1000:.3f} | {row['tpot_mean_ms']:.2f} | {row['expert_h2d_gib']:.3f} | {row['resize_wall_ms']:.2f} |")
        lines += ['',f"相对上一轮动态K6参考：吞吐{vs_previous['actual_qps']:+.2f}%，TTFT{vs_previous['ttft_mean_ms']:+.2f}%，TPOT{vs_previous['tpot_mean_ms']:+.2f}%，专家H2D{vs_previous['expert_h2d_gib']:+.2f}%。上一轮K6没有在本轮重新测量，不作为新的对称多臂统计结论。", '']
    lines += ['## 输出校验', '', '| 对比 | 请求对 | 完整输出不同 | 首token不同 |', '|---|---:|---:|---:|']
    for check in checked['output_checks']:
        lines.append(f"| {check['comparison']} | {check['request_pairs']} | {check['full_output_differences']} | {check['first_token_differences']} |")
    lines += ['', '成功返回不等于严格输出等价；差异均保留，未将其原因预先归为数值/合批。每配置仅两个冷启动，不建立统计显著性。原始逐请求及逐step数据与runtime/source快照均保留。', '']
    if args.plot:
        lines += ['![KV and queue timeline](kv_pressure_timeline.png)', '']
    (root/'REPORT.md').write_text('\n'.join(lines))
    shutil.copy2(Path(__file__),root/'source_snapshots'/Path(__file__).name)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'source_snapshots').iterdir() if p.is_file()}
    (root/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    for suffix,name in (('md','REPORT.md'),('json','summary.json'),('csv','summary.csv')):
        shutil.copy2(root/name,HERE/f'P1024_QPS30_K2_K4_20261024.{suffix}')
    easy=HERE/'P1024_QPS30_K2_K4_20261024.md'
    easy.write_text(easy.read_text().replace('](kv_pressure_timeline.png)',f"]({root/'kv_pressure_timeline.png'})"))
    print(json.dumps({'status':result['status'],'rows':rows,'changes':changes,
                      'vs_previous_k6':vs_previous,'report':str(root/'REPORT.md')},indent=2))


if __name__=='__main__':
    main()
