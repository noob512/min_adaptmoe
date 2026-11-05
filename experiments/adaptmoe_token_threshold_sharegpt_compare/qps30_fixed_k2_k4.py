#!/usr/bin/env python3
"""Verify and summarize fresh fixed GPU-KV K2/cold11 and K4/cold17 runs."""
import argparse
import ast
import hashlib
import json
import re
import shutil
import statistics
from pathlib import Path

from analyze_qps30_kv_pressure import analyze_case, write_csv
from verify_dynamic_kv_p512 import verify as verify_base
from verify_p1024_dynamic_kv import FIXTURE_SHA

HERE = Path(__file__).resolve().parent
ARMS = {'adaptive_fixed': (4313, 'cold11', 4948230144),
        'adaptive_fixed_4g': (8629, 'cold17', 7647264768)}


def verify(root):
    config = json.loads((root/'run_config.json').read_text())
    assert config['arms'] == list(ARMS) and config['arrival_qps'] == 30
    assert config['samples_per_arm'] == 2 and config['max_num_seqs'] == 128
    assert config['arm_kv_cache_memory_bytes'] == {arm:6442450944 for arm in ARMS}
    assert config['fixture_sha256'] == FIXTURE_SHA
    assert json.loads((root/'module.custom.json').read_text())['manual_migration_fault_mode'] == 2
    cases, outputs, settings = [], {}, {}
    for arm, (target, plan_class, expert_bytes) in ARMS.items():
        value = verify_base(root, selected_arm=arm, expected_fixture_sha=FIXTURE_SHA,
                            expected_prompt_tokens=1024, expected_max_model_len=1088,
                            expected_autoscale=False, expected_target_blocks=(4313,8629),
                            expected_initial_target=target, expected_bootstrap_target=4313)
        for case in value['cases']:
            directory = Path(case['directory'])
            raw = json.loads((directory/'result.json').read_text())
            detail = json.loads((directory/'kv_pressure_detail.json').read_text())
            runtime = json.loads((directory/'runtime_settings.json').read_text())
            assert runtime['VLLM_EXPERIMENT_KV_VARIANT'] == 'fixed_managed'
            assert runtime['VLLM_UVM_KV_AUTO_SCALE_ENABLE'] == '0'
            assert runtime['VLLM_UVM_KV_DYNAMIC_POOL_CONTROL_FILE'] == ''
            assert runtime['VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS'] == '4313'
            assert runtime['VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS'] == '4313'
            assert runtime['VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_GROW_TARGET_BLOCKS'] == str(target)
            assert runtime['VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS'] == '4313,8629'
            assert runtime['VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS'] == '8629'
            assert runtime['VLLM_UVM_KV_BACKUP_PRESSURE_STANDALONE_GROW_ENABLE'] == '0'
            assert detail['controller_calls'] == []
            assert detail['initial_target_blocks'] == target
            assert raw['dynamic_kv']['resize_transactions'] == []
            assert case['used_targets'] == [target]
            plan = raw['dynamic_kv']['initial_plan']
            assert plan['initial_class'] == 'cold11' and plan['active_class'] == plan_class
            assert Path(runtime['VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE']).resolve() == Path(plan['classes']['cold11']['plan_file']).resolve()
            startup = json.loads((directory/'kv_startup_residency.json').read_text())
            assert startup['target_blocks'] == 4313
            gpu, cpu = [a['residency'] for a in startup['audits']]
            assert gpu['ok'] and gpu['gpu_bytes'] == gpu['queried_bytes'] == (4313+1)*497664
            assert gpu['cpu_bytes'] == gpu['unknown_bytes'] == 0
            assert cpu['ok'] and cpu['cpu_bytes'] == cpu['queried_bytes'] == (12943-4313)*497664
            assert cpu['gpu_bytes'] == cpu['unknown_bytes'] == 0
            log = (directory/'run.log').read_text()
            growth_audits = re.findall(r'Post-growth active KV residency audit: target=(\d+) ok=(\w+) residency=(\{[^\n]+\})',log)
            if target == 8629:
                assert len(growth_audits) == 1, growth_audits
                audited_target, ok, residency = growth_audits[0]
                gpu = ast.literal_eval(residency)
                assert int(audited_target) == target and ok == 'True'
                assert gpu['ok'] and gpu['gpu_bytes'] == gpu['queried_bytes'] == (target+1)*497664
                assert gpu['cpu_bytes'] == gpu['unknown_bytes'] == 0
            else:
                assert growth_audits == []
            workspace = json.loads((directory/'mla_workspace_startup_residency.json').read_text())
            assert workspace['expected_bytes'] == 10027008
            audit = workspace['residency']
            assert audit['ok'] and audit['gpu_bytes'] == audit['queried_bytes'] == 10027008
            assert audit['cpu_bytes'] == audit['unknown_bytes'] == 0
            steps = raw['kv_concurrency']['scheduler_steps']
            assert raw['kv_concurrency']['variant'] == 'fixed_managed'
            assert raw['kv_concurrency']['initial_pool_blocks_including_null'] == 12944
            for step in steps:
                assert 0 <= step['active_blocks'] <= min(target,8320)
                assert step['active_blocks'] + sum(step['free_counts'].values()) == 12943
                assert step['free_counts']['offline_block_queue'] == 12943-target
                assert 0 <= step['running_requests'] <= 128
            assert all(step['before']['target_blocks'] == step['after']['target_blocks'] == target
                       for step in detail['scheduler_steps'])
            modes = re.findall(r'CUDA_GRAPH_RUNTIME_AUDIT[^\n]*runtime_mode=(\w+)',(directory/'run.log').read_text())
            assert modes and set(modes) == {'NONE'}
            settings[arm] = runtime
            outputs[arm,case['sample']] = raw['requests']
            cases.append({**case,'arm':arm,'fixed_target_blocks':target,'expert_class':plan_class,
                          'expert_bytes_per_copy_step':expert_bytes,'used_tiers':['K2' if target==4313 else 'K4'],
                          'max_running_requests':max(s['running_requests'] for s in steps),
                          'max_waiting_requests':max(s['waiting_requests'] for s in steps),
                          'max_active_kv_gib':max(s['active_blocks'] for s in steps)*497664/2**30,
                          'transactions':[],'growth_count':0,'shrink_count':0,'resize_wall_ms':0})
    a,b = (settings[arm] for arm in ARMS)
    changed = {key for key in a.keys()|b.keys() if a.get(key)!=b.get(key)}
    assert changed == {'VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_GROW_TARGET_BLOCKS'},changed
    checks = []
    for label,left,right in (
        ('K2_repeat',('adaptive_fixed',1),('adaptive_fixed',2)),
        ('K4_repeat',('adaptive_fixed_4g',1),('adaptive_fixed_4g',2)),
        ('K2_vs_K4_s1',('adaptive_fixed',1),('adaptive_fixed_4g',1)),
        ('K2_vs_K4_s2',('adaptive_fixed',2),('adaptive_fixed_4g',2))):
        pairs = list(zip(outputs[left],outputs[right]))
        assert len(pairs) == 128
        checks.append({'comparison':label,'request_pairs':128,
                       'full_output_differences':sum(a['output_token_ids']!=b['output_token_ids'] for a,b in pairs),
                       'first_token_differences':sum(a['output_token_ids'][0]!=b['output_token_ids'][0] for a,b in pairs)})
    return {'status':'ok_with_output_drift' if any(c['full_output_differences'] for c in checks) else 'ok',
            'errors':[],'cases':cases,'output_checks':checks,'protocol':config,
            'runtime_settings_differences':sorted(changed)}


def make_plot(root,timeline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,constrained_layout=True)
    for column,(arm,(target,_,_)) in enumerate(ARMS.items()):
        rows=[r for r in timeline if r['arm']==arm and r['sample']==1]
        x=[r['start_s'] for r in rows]+[rows[-1]['finish_s']]
        active=[r['active_kv_gib'] for r in rows]+[0]
        axes[0,column].step(x,active,where='post',label='Active KV',color='#2563eb',linewidth=2)
        axes[0,column].hlines(target*497664/2**30,0,x[-1],label='GPU KV target',color='#dc2626',linestyle='--')
        axes[0,column].set(title='Fixed K2 / cold11' if target==4313 else 'Fixed K4 / cold17',ylabel='GiB',ylim=(0,4.4))
        for key,label,color in (('running_requests','Running','#2563eb'),('waiting_requests','Waiting','#ea580c')):
            axes[1,column].step(x,[r[key] for r in rows]+[0],where='post',label=label,color=color)
        axes[1,column].set(xlabel='Seconds since first planned arrival',ylabel='Requests',ylim=(0,135))
        for ax in axes[:,column]:
            ax.axvline(127/30,color='#777777',linestyle=':',label='Last planned arrival')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[1,0].set_xlim(0,max(r['finish_s'] for r in timeline)*1.03)
    fig.suptitle('QPS30; P1024 / O16; N128; C128; autoscaling disabled; sample 1')
    fig.savefig(root/'fixed_kv_timeline.png',dpi=180)
    fig.savefig(root/'fixed_kv_timeline.pdf')
    plt.close(fig)


def analyze(root,draw_plot):
    checked=verify(root/'qps30')
    cases,timeline,failures,requests=[],[],[],[]
    for case in checked['cases']:
        summary,steps,rejected,reqs=analyze_case(case)
        cases.append(summary);timeline.extend(steps);failures.extend(rejected);requests.extend(reqs)
    rows=[]
    for arm in ARMS:
        selected=[c for c in cases if c['arm']==arm]
        fields=('actual_qps','ttft_mean_ms','tpot_mean_ms','e2e_mean_ms','elapsed_s','allocation_failures',
                'preemptions','copy_steps','remote_steps','expert_h2d_gib','resize_wall_ms',
                'controller_including_resize_ms','admission_delay_ms','engine_queueing_ms','scheduled_to_first_token_ms')
        rows.append({'arm':arm,'label':'固定K2/cold11' if arm=='adaptive_fixed' else '固定K4/cold17',
                     **{key:statistics.fmean(c[key] for c in selected) for key in fields},
                     'max_active_kv_gib':max(c['max_active_kv_gib'] for c in selected),
                     'max_running_requests':max(c['max_running_requests'] for c in selected),
                     'max_waiting_requests':max(c['max_waiting_requests'] for c in selected),
                     'waiting_steps_by_cause':{reason:statistics.fmean(c['waiting_steps_by_cause'].get(reason,0) for c in selected)
                                               for reason in ('kv_allocation_failure','token_budget_exhausted','max_num_seqs','other')}})
    k2,k4=rows
    change={key:(k4[key]/k2[key]-1)*100 for key in ('actual_qps','ttft_mean_ms','tpot_mean_ms','e2e_mean_ms','expert_h2d_gib')}
    result={'status':checked['status'],'processes':4,'successful_requests':512,'rows':rows,'cases':cases,
            'k4_vs_k2_percent':change,'output_checks':checked['output_checks'],
            'scope':'Fresh fixed K2/cold11 versus fixed K4/cold17; no timed resize and no autoscaling in either arm.'}
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    write_csv(root/'summary.csv',rows);write_csv(root/'scheduler_timeline.csv',timeline)
    write_csv(root/'request_latency.csv',requests)
    if failures:write_csv(root/'allocation_failures.csv',failures)
    if draw_plot:make_plot(root,timeline)
    lines=['# QPS30：固定K2与固定K4的真实模型对照','',
           '同一ShareGPT P1024/O16、128请求，DeepSeek-V2-Lite-Chat BF16、TRITON_MLA、C128、8192-token预算、PARK≤16/RESUME≥128、混合调度、eager、无CUDA Graph、无prefix cache；提供QPS=30，计划到达i/30。每组两个新进程，K2→K4→K4→K2交错，共512请求。','',
           '两组自动KV扩缩容均关闭，在计时前完成相应GPU容量驻留与专家档位初始化。固定K2为4313块/cold11，固定K4为8629块/cold17；共享相同6GiB managed VA预留，GPU驻留和离线尾部已分别核验，不能将VA预留当作实际GPU KV容量。保留token-threshold访问切换及原有物理驻留检查，运行期不切换KV或专家档位。','',
           '两组都从K2/cold11初始化专家源，唯一不同的运行设置是pre-service目标。K4在LLM构造完成前通过现有事务扩到K4/cold17，并核验扩容后的GPU KV驻留；该准备发生在全部请求计时前。两组外部容量控制文件均关闭，避免bootstrap K2文件覆盖已准备好的K4。运行期固定为各自目标，resize和自动控制器调用均为0，逐step核验目标及offline队列大小。这是KV与配套专家驻留的系统配置比较，不能单独归因为KV容量本身。专家实际卸载量分别4.608GiB和7.122GiB；legacy adaptive_selected_bytes和源finalized事件描述bootstrap cold11，测量期实际量以服务就绪时active_class与逐step forward manifest为准。','',
           '## 性能（两个独立进程均值）','','| 配置 | 实际QPS | TTFT s | TPOT ms | E2E s | 总时长s |','|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['label']} | {row['actual_qps']:.3f} | {row['ttft_mean_ms']/1000:.3f} | {row['tpot_mean_ms']:.2f} | {row['e2e_mean_ms']/1000:.3f} | {row['elapsed_s']:.3f} |")
    lines+=['',f"K4相对K2：吞吐{change['actual_qps']:+.2f}%，TTFT{change['ttft_mean_ms']:+.2f}%，TPOT{change['tpot_mean_ms']:+.2f}%，专家H2D{change['expert_h2d_gib']:+.2f}%。",'',
            '## 容量、排队与实际搬运','','| 配置 | KV峰值GiB | 最大运行/等待请求 | 抢占 | KV分配失败/进程 | KV/token预算阻塞step | copy/remote step | 专家H2D GiB |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        reasons=row['waiting_steps_by_cause']
        lines.append(f"| {row['label']} | {row['max_active_kv_gib']:.3f} | {row['max_running_requests']}/{row['max_waiting_requests']} | {row['preemptions']:.0f} | {row['allocation_failures']:.0f} | {reasons['kv_allocation_failure']:.0f}/{reasons['token_budget_exhausted']:.0f} | {row['copy_steps']:.0f}/{row['remote_steps']:.0f} | {row['expert_h2d_gib']:.3f} |")
    lines+=['','峰值为两次最大值；分配失败为尝试次数，不等于最终请求失败。两组计时期resize事务数均为0，自动控制器调用均为0。较大KV允许更多并发，但配套专家卸载更多；H2D列仅统计forward显式专家copy，不含初始化迁移。','',
            '## TTFT分解（ms）','','| 配置 | 计划到达→引擎接收 | 引擎内等待 | 首次调度→首token |','|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['label']} | {row['admission_delay_ms']:.2f} | {row['engine_queueing_ms']:.2f} | {row['scheduled_to_first_token_ms']:.2f} |")
    lines+=['','第一项包含同步离线replay在engine.step返回后接收已到时请求的延迟，与之前口径一致。全部指标包括排队及排空，没有额外真实conditioning；不是纯稳态测量。','',
            '## 输出校验','','| 对比 | 请求对 | 完整输出不同 | 首token不同 |','|---|---:|---:|---:|']
    for c in checked['output_checks']:
        lines.append(f"| {c['comparison']} | {c['request_pairs']} | {c['full_output_differences']} | {c['first_token_differences']} |")
    lines+=['','请求返回成功不等于严格逐token等价，差异原样保留；每组两个独立冷启动，不建立统计显著性。逐step/请求/分配失败CSV、逐arm原始JSON与源码快照均保留。','',f'原始artifact：{root}','']
    if draw_plot:lines+=['![Fixed KV timeline](fixed_kv_timeline.png)','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'source_snapshots').iterdir() if p.is_file()}
    (root/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    for suffix,name in (('md','REPORT.md'),('json','summary.json'),('csv','summary.csv')):
        shutil.copy2(root/name,HERE/f'P1024_QPS30_FIXED_K2_VS_K4_20261024.{suffix}')
    easy=HERE/'P1024_QPS30_FIXED_K2_VS_K4_20261024.md'
    easy.write_text(easy.read_text().replace('](fixed_kv_timeline.png)',f"]({root/'fixed_kv_timeline.png'})"))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--report',type=Path);parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    try:
        if args.output or args.report:
            assert args.output and args.report
            value=verify(args.root.resolve())
            args.output.write_text(json.dumps(value,indent=2)+'\n')
            args.report.write_text('# Fixed K2/K4 verification\n\n'+json.dumps(value,indent=2)+'\n')
        else:
            value=analyze(args.root.resolve(),args.plot)
    except BaseException as exc:
        if args.output:args.output.write_text(json.dumps({'status':'error','errors':[f'{type(exc).__name__}: {exc}']},indent=2)+'\n')
        raise
    print(json.dumps({'status':value['status'],'rows':value.get('rows'),
                      'cases':[{k:c[k] for k in ('arm','sample','actual_qps','ttft_mean_ms','tpot_mean_ms')} for c in value['cases']]},indent=2))
