#!/usr/bin/env python3
"""Prepare, validate and summarize fixed K1, fixed K2, automatic K1/K2."""
import argparse
import ast
import hashlib
import json
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

from analyze_qps30_kv_pressure import analyze_case, write_csv
from verify_dynamic_kv_p512 import verify as verify_base
from verify_p1024_dynamic_kv import FIXTURE_SHA

HERE = Path(__file__).resolve().parent
ARMS = {'adaptive_fixed_1g': ('固定 K1/cold8', False, 2155),
        'adaptive_fixed_2g': ('固定 K2/cold11', False, 4313),
        'adaptive_dynamic': ('动态 K1↔K2', True, 2155)}
CLASSES = {2155: 'cold8', 4313: 'cold11'}


def prepare(out):
    subprocess.run([sys.executable, str(HERE/'prepare_dynamic_kv_p512.py'), str(out),
                    '--small-kv-ladder', '--max-num-seqs', '128', '--prompt-tokens', '1024'], check=True)
    base = json.loads((out/'runtime_settings.json').read_text())
    tiers = json.loads((out/'manifest.json').read_text())['tiers']
    assert [t['target_blocks'] for t in tiers] == [2155,4313]
    base.update({'VLLM_UVM_KV_DYNAMIC_POOL_CONTROL_FILE': '',
                 'VLLM_EXPERIMENT_KV_POOL_BLOCKS': '4314',
                 'VLLM_EXPERIMENT_RESIZE_TELEMETRY': '1',
                 'VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE': tiers[0]['plan_file']})
    for arm, (_, automatic, ready_target) in ARMS.items():
        settings = {**base, 'VLLM_UVM_KV_AUTO_SCALE_ENABLE': '1' if automatic else '0',
                    'VLLM_EXPERIMENT_KV_VARIANT': 'dynamic' if automatic else 'fixed_managed',
                    'VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_GROW_TARGET_BLOCKS': str(ready_target)}
        (out/f'{arm}_runtime_settings.json').write_text(json.dumps(settings, indent=2)+'\n')


def verify(root, selected_arms=None, *, expected_qps=30, expected_samples=2):
    active_arms = {a:ARMS[a] for a in (selected_arms or ARMS)}
    cfg = json.loads((root/'run_config.json').read_text())
    assert (set(active_arms)<=set(cfg['arms']) if selected_arms else cfg['arms']==list(ARMS))
    assert cfg['arrival_qps'] == expected_qps
    assert expected_samples in (1,2)
    assert cfg['samples_per_arm'] == expected_samples and cfg['max_num_seqs'] == 128
    assert cfg['fixture_sha256'] == FIXTURE_SHA
    assert all(cfg['arm_kv_cache_memory_bytes'][arm]==2**31 for arm in active_arms)
    assert json.loads((root/'module.custom.json').read_text())['manual_migration_fault_mode'] == 2
    cases, outputs, settings = [], {}, {}
    for arm, (_, automatic, ready_target) in active_arms.items():
        checked = verify_base(root, selected_arm=arm, expected_fixture_sha=FIXTURE_SHA,
                    expected_prompt_tokens=1024, expected_max_model_len=1088,
                    expected_autoscale=automatic, expected_target_blocks=(2155,4313),
                    expected_initial_target=ready_target, expected_bootstrap_target=2155,
                    expected_pool_blocks=4313, expected_kv_reservation_bytes=2**31,
                    expected_plan_classes=('cold8','cold11'), expected_samples=expected_samples)
        for case in checked['cases']:
            directory = Path(case['directory'])
            raw = json.loads((directory/'result.json').read_text())
            detail = json.loads((directory/'kv_pressure_detail.json').read_text())
            runtime = json.loads((directory/'runtime_settings.json').read_text())
            assert runtime['VLLM_EXPERIMENT_KV_VARIANT'] == ('dynamic' if automatic else 'fixed_managed')
            assert runtime['VLLM_UVM_KV_AUTO_SCALE_ENABLE'] == ('1' if automatic else '0')
            assert runtime['VLLM_UVM_KV_DYNAMIC_POOL_CONTROL_FILE'] == ''
            assert runtime['VLLM_UVM_KV_BACKUP_PRESSURE_STANDALONE_GROW_ENABLE'] == '0'
            assert runtime['VLLM_EXPERIMENT_KV_POOL_BLOCKS'] == '4314'
            assert runtime['VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS'] == '2155'
            assert runtime['VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS'] == '2155'
            assert runtime['VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_GROW_TARGET_BLOCKS'] == str(ready_target)
            assert runtime['VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS'] == '2155,4313'
            plan = raw['dynamic_kv']['initial_plan']
            assert plan['initial_class'] == 'cold8' and plan['active_class'] == CLASSES[ready_target]
            assert detail['initial_target_blocks'] == ready_target
            startup = json.loads((directory/'kv_startup_residency.json').read_text())
            assert startup['target_blocks'] == 2155
            gpu, cpu = [a['residency'] for a in startup['audits']]
            assert gpu['ok'] and gpu['gpu_bytes'] == gpu['queried_bytes'] == 2156*497664
            assert gpu['cpu_bytes'] == gpu['unknown_bytes'] == 0
            assert cpu['ok'] and cpu['cpu_bytes'] == cpu['queried_bytes'] == 2158*497664
            assert cpu['gpu_bytes'] == cpu['unknown_bytes'] == 0
            workspace = json.loads((directory/'mla_workspace_startup_residency.json').read_text())
            assert workspace['expected_bytes'] == 10027008
            audit = workspace['residency']
            assert audit['ok'] and audit['gpu_bytes'] == audit['queried_bytes'] == 10027008
            assert audit['cpu_bytes'] == audit['unknown_bytes'] == 0
            telemetry = raw['kv_concurrency']
            assert telemetry['initial_pool_blocks_including_null'] == 4314
            assert telemetry['variant'] == ('dynamic' if automatic else 'fixed_managed')
            for observed, controller, step in zip(detail['scheduler_steps'], raw['dynamic_kv']['controller_steps'], telemetry['scheduler_steps']):
                assert step['active_blocks']+sum(step['free_counts'].values()) == 4313
                for state in (observed['before'],observed['after'],observed['after_engine']):
                    target = state['target_blocks']
                    assert target in CLASSES and (automatic or target == ready_target)
                    assert 0 <= state['active_blocks'] <= target
                    assert state['free_counts']['offline_block_queue'] == 4313-target
                assert controller['active_plan_class'] == CLASSES[controller['applied_target_blocks']]
            transactions = raw['dynamic_kv']['resize_transactions']
            for transaction in transactions:
                assert not transaction.get('exception') and transaction['result']['ok'] is True
                assert transaction['requested_target_blocks'] == transaction['applied_target_blocks']
                assert transaction['old_target_blocks'] in CLASSES and transaction['applied_target_blocks'] in CLASSES
            if not automatic:
                assert detail['controller_calls'] == [] and transactions == []
                assert case['used_targets'] == [ready_target]
            log = (directory/'run.log').read_text()
            modes = re.findall(r'CUDA_GRAPH_RUNTIME_AUDIT[^\n]*runtime_mode=(\w+)',log)
            assert modes and set(modes) == {'NONE'}
            audits = re.findall(r'Post-growth active KV residency audit: target=(\d+) ok=(\w+) residency=(\{[^\n]+\})',log)
            expected_grows = (ready_target != 2155)+sum(t['applied_target_blocks']>t['old_target_blocks'] for t in transactions)
            assert len(audits) == expected_grows
            for target, ok, payload in audits:
                residency = ast.literal_eval(payload)
                assert int(target) == 4313 and ok == 'True' and residency['ok']
                assert residency['gpu_bytes'] == residency['queried_bytes'] == 4314*497664
                assert residency['cpu_bytes'] == residency['unknown_bytes'] == 0
            steps = telemetry['scheduler_steps']
            cases.append({**case, 'arm':arm, 'used_tiers':['K1' if t==2155 else 'K2' for t in case['used_targets']],
                          'max_running_requests':max(s['running_requests'] for s in steps),
                          'max_waiting_requests':max(s['waiting_requests'] for s in steps),
                          'max_active_kv_gib':max(s['active_blocks'] for s in steps)*497664/2**30,
                          'transactions':transactions, 'growth_count':sum(t['applied_target_blocks']>t['old_target_blocks'] for t in transactions),
                          'shrink_count':sum(t['applied_target_blocks']<t['old_target_blocks'] for t in transactions),
                          'resize_wall_ms':sum(t['wall_ms'] for t in transactions)})
            outputs[arm,case['sample']] = raw['requests']
            settings[arm] = runtime
    allowed = {'VLLM_UVM_KV_AUTO_SCALE_ENABLE','VLLM_EXPERIMENT_KV_VARIANT',
               'VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_GROW_TARGET_BLOCKS'}
    first = settings['adaptive_fixed_1g']
    for other in settings.values():
        changed = {k for k in first.keys()|other.keys() if first.get(k) != other.get(k)}
        assert changed <= allowed, changed
    pairs = [(arm+'_repeat',(arm,1),(arm,2)) for arm in active_arms] if expected_samples==2 else []
    for sample in range(1,expected_samples+1):
        for left, right in (('adaptive_fixed_1g','adaptive_fixed_2g'),
                            ('adaptive_fixed_1g','adaptive_dynamic'),('adaptive_fixed_2g','adaptive_dynamic')):
            if left in active_arms and right in active_arms:
                pairs.append((f'{left}_vs_{right}_s{sample}',(left,sample),(right,sample)))
    checks = []
    for label, left, right in pairs:
        a,b = outputs[left],outputs[right]
        assert len(a) == len(b) == 128
        checks.append({'comparison':label,'request_pairs':128,
                       'full_output_differences':sum(x['output_token_ids']!=y['output_token_ids'] for x,y in zip(a,b)),
                       'first_token_differences':sum(x['output_token_ids'][0]!=y['output_token_ids'][0] for x,y in zip(a,b))})
    return {'status':'ok_with_output_drift' if any(x['full_output_differences'] for x in checks) else 'ok',
            'errors':[],'cases':cases,'output_checks':checks,'protocol':cfg}


def plot(root, timeline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,3,figsize=(15,7),sharex=True,constrained_layout=True)
    for col,(arm,(_,automatic,initial)) in enumerate(ARMS.items()):
        rows = [r for r in timeline if r['arm']==arm and r['sample']==1]
        x = [r['start_s'] for r in rows]+[rows[-1]['finish_s']]
        axes[0,col].step(x,[r['active_kv_gib'] for r in rows]+[0],where='post',label='Active KV')
        axes[0,col].step(x,[r['target_gib'] for r in rows]+[rows[-1]['target_gib']],where='post',label='GPU KV target',linestyle='--')
        axes[0,col].set(title=['Fixed K1 / cold8','Fixed K2 / cold11','Automatic K1 / K2'][col],ylabel='GiB',ylim=(0,2.2))
        for key,label in (('running_requests','Running'),('waiting_requests','Waiting')):
            axes[1,col].step(x,[r[key] for r in rows]+[0],where='post',label=label)
        axes[1,col].set(xlabel='Seconds since first planned arrival',ylabel='Requests',ylim=(0,135))
        for ax in axes[:,col]:
            ax.axvline(127/30,color='gray',linestyle=':',label='Last planned arrival')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[1,0].set_xlim(0,max(r['finish_s'] for r in timeline)*1.03)
    fig.suptitle('QPS30; ShareGPT P1024/O16; N128; C128; eager; sample 1')
    fig.savefig(root/'kv_and_queue_timeline.png',dpi=180)
    fig.savefig(root/'kv_and_queue_timeline.pdf')
    plt.close(fig)


def analyze(root, draw):
    checked = verify(root/'qps30')
    cases,timeline,failures,requests = [],[],[],[]
    for case in checked['cases']:
        summary,steps,rejected,reqs = analyze_case(case,require_decode_allocations_success=False)
        cases.append(summary);timeline.extend(steps);failures.extend(rejected);requests.extend(reqs)
    rows = []
    fields = ('actual_qps','ttft_mean_ms','tpot_mean_ms','e2e_mean_ms','elapsed_s',
              'allocation_failures','preemptions','copy_steps','remote_steps','expert_h2d_gib',
              'resize_wall_ms','controller_including_resize_ms','controller_outside_resize_ms',
              'admission_delay_ms','engine_queueing_ms','scheduled_to_first_token_ms')
    for arm,(label,_,_) in ARMS.items():
        selected = [c for c in cases if c['arm']==arm]
        assert len(selected)==2
        rows.append({'arm':arm,'label':label,**{k:statistics.fmean(c[k] for c in selected) for k in fields},
                     'max_active_kv_gib':max(c['max_active_kv_gib'] for c in selected),
                     'max_running_requests':max(c['max_running_requests'] for c in selected),
                     'max_waiting_requests':max(c['max_waiting_requests'] for c in selected),
                     'used_tiers':sorted(set(t for c in selected for t in c['used_tiers'])),
                     'growth_counts':[c['growth_count'] for c in selected],
                     'shrink_counts':[c['shrink_count'] for c in selected],
                     'waiting_steps_by_cause':{reason:statistics.fmean(c['waiting_steps_by_cause'].get(reason,0) for c in selected)
                                               for reason in ('kv_allocation_failure','token_budget_exhausted','max_num_seqs','other')}})
    changes = {rows[i]['arm']:{k:(rows[2][k]/rows[i][k]-1)*100 for k in ('actual_qps','ttft_mean_ms','tpot_mean_ms','e2e_mean_ms')} for i in (0,1)}
    dynamic_raw = [json.loads((Path(c['directory'])/'result.json').read_text())
                   for c in cases if c['arm']=='adaptive_dynamic']
    drift = []
    for left,right in zip(dynamic_raw[0]['requests'],dynamic_raw[1]['requests']):
        if left['output_token_ids'][0] == right['output_token_ids'][0]:
            continue
        fragments = []
        for raw in dynamic_raw:
            remaining,parts = 1024,[]
            for step in raw['scheduler_steps']:
                count = step['num_scheduled_tokens'].get(left['request_id'],0)
                if count and remaining:
                    take = min(count,remaining);remaining -= take
                    parts.append({'step':step['scheduler_step'],'prefill_tokens':take})
            fragments.append(parts)
        drift.append({'request_id':left['request_id'],'fixture_index':left['fixture_index'],
                      'sample1_prefill':fragments[0],'sample2_prefill':fragments[1],
                      'first_token_ids':[left['output_token_ids'][0],right['output_token_ids'][0]]})
    drift_diagnostic = {'observation':'Dynamic repeats have different chunked-prefill scheduling; split versus unsplit prefill is recorded for first-token differences. This association alone does not prove the root cause of output drift.',
                        'examples':drift}
    (root/'output_drift_diagnostics.json').write_text(json.dumps(drift_diagnostic,indent=2)+'\n')
    result = {'status':checked['status'],'processes':6,'successful_requests':768,
              'rows':rows,'cases':cases,'dynamic_vs_fixed_percent':changes,'output_checks':checked['output_checks'],
              'output_drift_diagnostics':drift_diagnostic,
              'strict_output_equivalence_verified':not any(c['full_output_differences'] for c in checked['output_checks'])}
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    write_csv(root/'summary.csv',rows);write_csv(root/'scheduler_timeline.csv',timeline)
    write_csv(root/'request_latency.csv',requests)
    if failures:write_csv(root/'allocation_failures.csv',failures)
    if draw:plot(root,timeline)
    lines=['# QPS30：AdaptMoE 固定 K1、固定 K2 与动态 K1↔K2','',
           '同一128条ShareGPT trace，1024输入/16 greedy输出，DeepSeek-V2-Lite-Chat BF16、TRITON_MLA、C128、8192-token预算、PARK≤16/RESUME≥128、混合调度、eager、无CUDA Graph、无prefix cache。提供QPS=30，到达时间i/30。每组两个独立新进程，执行顺序K1→K2→动态→动态→K2→K1，共768请求。无额外真实conditioning，指标包含排队、排空与动态组真实扩缩容；不是纯稳态。','',
           '三组使用相同2GiB managed虚拟池（4314块含null），K1=2155可用GPU块，K2=4313块。K1/cold8每层卸载8个专家（3.3515625GiB），K2/cold11每层11个（4.6083984375GiB）；二者严格嵌套，都是既有校准计划。升档仅增加每层3个卸载专家，为约1GiB新增KV提供空间。这是KV和配套专家驻留配置的系统比较。','',
           '两组固定容量均关闭自动控制器；K2通过现有pre-service事务在全部请求计时前准备完成，运行期逐步验证容量恒定。动态组启动于K1，仅允许K1/K2、80%高水位、4步最小驻留、3步低压确认，并保留尾部空闲证明与物理迁移/驻留审计。外部容量控制文件均关闭，避免bootstrap文件覆盖服务容量。','',
           '## 性能：两个独立进程均值','','| 配置 | 实际QPS | TTFT s | TPOT ms | E2E s |','|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['label']} | {r['actual_qps']:.3f} | {r['ttft_mean_ms']/1000:.3f} | {r['tpot_mean_ms']:.2f} | {r['e2e_mean_ms']/1000:.3f} |")
    lines+=['','## 容量、调度和搬运','','| 配置 | KV峰值GiB | 最大运行/等待 | KV分配失败 | 抢占 | copy/remote step | forward专家H2D GiB | resize ms |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['label']} | {r['max_active_kv_gib']:.3f} | {r['max_running_requests']}/{r['max_waiting_requests']} | {r['allocation_failures']:.0f} | {r['preemptions']:.0f} | {r['copy_steps']:.0f}/{r['remote_steps']:.0f} | {r['expert_h2d_gib']:.3f} | {r['resize_wall_ms']:.2f} |")
    lines+=['','峰值取两次最大值，其余为两个进程均值。KV分配失败是分配尝试次数，不等于最终请求失败；WAITING分配失败导致排队，RUNNING追加分配失败才可能抢占。H2D仅统计forward显式专家copy，不含KV和专家档位迁移。未做CUDA分时剖析，不把字节增量直接等同于全部延迟差异。','',
            '## 动态组相对固定组','','| 对照 | QPS变化 | TTFT变化 | TPOT变化 | E2E变化 |','|---|---:|---:|---:|---:|']
    for arm,c in changes.items():lines.append(f"| 相对{ARMS[arm][0]} | {c['actual_qps']:+.2f}% | {c['ttft_mean_ms']:+.2f}% | {c['tpot_mean_ms']:+.2f}% | {c['e2e_mean_ms']:+.2f}% |")
    lines+=['','本轮动态组相对固定1GiB减少了KV容量阻塞并提高吞吐，但TPOT和平均E2E没有改善。相对固定2GiB，动态组未取得性能优势。固定1GiB最多33个活跃请求，固定2GiB及升档后的动态组最多66个；固定1GiB虽然每步卸载专家少，但copy step更多，总forward H2D也更多。三组都没有RUNNING追加KV失败或抢占，排队是实测的主要容量压力表现。','',
            f"动态组forward专家H2D比固定2GiB少{(1-rows[2]['expert_h2d_gib']/rows[1]['expert_h2d_gib'])*100:.2f}%，但新增控制器wall约{rows[2]['controller_including_resize_ms']:.2f}ms，其中resize约{rows[2]['resize_wall_ms']:.2f}ms，其余约{rows[2]['controller_outside_resize_ms']:.2f}ms。控制器wall包含同步等待，不能解释为纯CPU耗时。整个trace使用恒定高到达率，不能据此证明低高低压力下动态策略的价值。",'']
    lines+=['','## 实际档位事件','','| 组/进程 | 实际档位 | 升/降档 | resize ms | 控制器其余wall ms |','|---|---|---:|---:|---:|']
    for c in cases:lines.append(f"| {c['arm']}/s{c['sample']} | {'→'.join(c['used_tiers'])} | {c['growth_count']}/{c['shrink_count']} | {c['resize_wall_ms']:.2f} | {c['controller_outside_resize_ms']:.2f} |")
    lines+=['','逐次事务时间、controller_step、旧/新容量与专家类见summary.json和各进程resize_transactions.json；升档与降档仅按真实事务记录报告。','',
            '## 输出校验','','| 对照 | 请求对 | 完整输出不同 | 首token不同 |','|---|---:|---:|---:|']
    for c in checked['output_checks']:lines.append(f"| {c['comparison']} | 128 | {c['full_output_differences']} | {c['first_token_differences']} |")
    if drift:
        lines+=['','动态组两次首token不同的请求，其Prefill分块如下：','',
                '| 请求 | 第一次Prefill分块 | 第二次Prefill分块 |','|---|---|---|']
        for entry in drift:
            a='+'.join(str(p['prefill_tokens']) for p in entry['sample1_prefill'])
            b='+'.join(str(p['prefill_tokens']) for p in entry['sample2_prefill'])
            lines.append(f"| {entry['request_id']} | {a} | {b} |")
        lines+=['','已确认两次实际合批与chunked prefill分块不同；这支持数值路径差异的可能，但尚未单独证明输出变化的根因，也不能排除映射问题。保留output_drift_diagnostics.json与原始输出。这些性能数值作为诊断结果，不能作为已通过严格输出等价校验的排名。','']
    lines+=['','成功返回不等于逐token完全等价，输出差异原样保留；每组仅两次新进程，不建立统计置信结论。固定2/4GiB的前一轮是历史参考，本表全部六个进程均为本轮新测。','',f'原始artifact：{root}','']
    if draw:lines+=['![KV and queue timeline](kv_and_queue_timeline.png)','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'source_snapshots').iterdir() if p.is_file()}
    (root/'source_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    for suffix,name in (('md','REPORT.md'),('json','summary.json'),('csv','summary.csv')):
        shutil.copy2(root/name,HERE/f'P1024_QPS30_FIXED_K1_K2_DYNAMIC_K1_K2_20261024.{suffix}')
    easy=HERE/'P1024_QPS30_FIXED_K1_K2_DYNAMIC_K1_K2_20261024.md'
    easy.write_text(easy.read_text().replace('](kv_and_queue_timeline.png)',f"]({root/'kv_and_queue_timeline.png'})"))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--output',type=Path);parser.add_argument('--report',type=Path)
    parser.add_argument('--plot',action='store_true');args=parser.parse_args()
    if args.prepare:
        prepare(args.root.resolve());raise SystemExit(0)
    try:
        if args.output or args.report:
            assert args.output and args.report
            value=verify(args.root.resolve())
            args.output.write_text(json.dumps(value,indent=2)+'\n')
            args.report.write_text('# K1/K2 verification\n\n'+json.dumps(value,indent=2)+'\n')
        else:value=analyze(args.root.resolve(),args.plot)
    except BaseException as exc:
        if args.output:args.output.write_text(json.dumps({'status':'error','errors':[f'{type(exc).__name__}: {exc}']},indent=2)+'\n')
        raise
    print(json.dumps({'status':value['status'],'rows':value.get('rows'),
                      'cases':[{k:c[k] for k in ('arm','sample','actual_qps','ttft_mean_ms','tpot_mean_ms','used_tiers')} for c in value['cases']]},indent=2))
