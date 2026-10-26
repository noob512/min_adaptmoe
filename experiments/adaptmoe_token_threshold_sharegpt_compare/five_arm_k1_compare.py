#!/usr/bin/env python3
"""Validate fresh fixed-K1 Native/KT/AdaptMoE and automatic K1/K2 results."""
import argparse
import hashlib
import itertools
import json
import re
import shutil
import statistics
from pathlib import Path

from analyze_qps30_kv_pressure import analyze_case, write_csv
from qps30_k1_k2_compare import prepare as prepare_small, verify as verify_small
from verify_dynamic_kv_p512 import verify_clean_snapshot
from verify_p1024_dynamic_kv import FIXTURE_SHA

HERE=Path(__file__).resolve().parent
NATIVE_BYTES=3690987520
LABELS={'native_uva':'Native UVA / KV1', 'native_copy':'Native Copy / KV1',
        'ktransformers':'KTransformers / KV1', 'adaptive_fixed_1g':'AdaptMoE fixed KV1',
        'adaptive_dynamic':'AdaptMoE dynamic KV1→2'}


def load(path):return json.loads(path.read_text())


def prepare(root):
    prepare_small(root)
    names=[f'model.layers.{layer}.mlp.experts.{parameter}_weight'
           for layer in range(1,4) for parameter in ('w13','w2')]
    names.append('model.layers.4.mlp.experts.w2_weight')
    (root/'native_cold8_near_equal_names.txt').write_text('\n'.join(names)+'\n')
    plan=load(Path(load(root/'adaptive_fixed_1g_runtime_settings.json')['VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE']))
    certificate={'adapt_fixed_expert_bytes':plan['selected_main_bytes'],
                 'kt_expert_bytes':plan['selected_main_bytes'],'native_expert_bytes':NATIVE_BYTES,
                 'native_vs_adapt_byte_difference_percent':(NATIVE_BYTES/plan['selected_main_bytes']-1)*100,
                 'native_parameter_names':names,'fixed_effective_kv_tokens':34480,
                 'vllm_fixed_resident_kv_tensor_bytes':2156*497664,
                 'kt_resident_kv_tensor_bytes':34481*31104,
                 'dynamic_max_effective_kv_tokens':4313*16}
    (root/'capacity_certificate.json').write_text(json.dumps(certificate,indent=2)+'\n')


def validate_requests(requests,fixture,rate):
    assert len(requests)==128
    for index,(r,source) in enumerate(zip(requests,fixture['requests'])):
        assert r['fixture_index']==index and r['success']
        assert r['prompt_token_ids_sha256']==source['prompt_token_ids_sha256']
        assert len(r['output_token_ids'])==r['completion_tokens']==16
        assert abs(r['planned_arrival_offset_seconds']-index/rate)<1e-6
        assert abs(r['tpot_ms']-statistics.fmean(r['itl_ms']))<1e-6


def verify_vllm(root, *, expected_qps=30, expected_samples=2):
    checked=verify_small(root,selected_arms=('adaptive_fixed_1g','adaptive_dynamic'),expected_qps=expected_qps,expected_samples=expected_samples)
    config=load(root/'run_config.json')
    assert config['arms']==['native_uva','native_copy','adaptive_fixed_1g','adaptive_dynamic']
    assert config['fixture_sha256']==FIXTURE_SHA and config['native_expected_bytes']==NATIVE_BYTES
    fixture=load(Path(config['fixture']))
    cases=checked['cases'][:]
    for line in (root/'execution_status.tsv').read_text().splitlines()[1:]:
        _,sample,arm,directory,rc=line.split('\t')
        if not arm.startswith('native_'):continue
        path=root/'arms'/directory;r=load(path/'result.json');s=r['summary'];c=r['configuration']
        assert rc=='0' and r['status']=='ok'
        assert c['kv_cache_memory_bytes']==2**30 and config['arm_kv_cache_memory_bytes'][arm]==2**30
        assert c['max_num_seqs']==128 and c['max_num_batched_tokens']==8192 and c['max_model_len']==1088
        assert c['arrival_qps']==expected_qps and c['output_tokens']==16
        assert c['compilation_config']=={'mode':'NONE','cudagraph_mode':'NONE'}
        assert c['phase_mode']=='off' and c['phase_batched_scheduling'] is False
        assert r['fixed_kv_capacity']=={'pool_blocks_including_null':2156,'usable_blocks':2155,
                                      'usable_tokens':34480,'kv_tensor_bytes':2156*497664,
                                      'managed_kv':False,'autoscaling':False}
        validate_requests(r['requests'],fixture,expected_qps)
        assert s['successful_requests']==128 and s['generated_tokens']==2048
        assert abs(s['request_throughput_per_second']-128/s['elapsed_seconds'])<1e-9
        events=[json.loads(line) for line in (path/'cpu_offload_manifest.jsonl').read_text().splitlines() if line.strip()]
        parameters=[e for e in events if e['event']=='cpu_offload_parameter']
        assert len(parameters)==7 and sum(e['bytes'] for e in parameters)==NATIVE_BYTES
        expected=load(root.parent/'inputs/capacity_certificate.json')['native_parameter_names']
        assert sorted(e['full_parameter_name'] for e in parameters)==sorted(expected)
        assert all(e['offload_backend']==('uva' if arm=='native_uva' else 'copy') for e in parameters)
        config_event=next(e for e in events if e['event']=='cpu_offload_config')
        assert config_event['graph_static_alias'] is False and config_event['copy_exact_name_count']==7
        window=r['measurement_window']
        forward=[e for e in events if e['event']=='cpu_offload_copy_forward' and window['started_epoch']<=e['timestamp']<=window['finished_epoch']]
        copied=sum(e.get('copy_bytes',0) for e in forward)
        assert copied==(0 if arm=='native_uva' else NATIVE_BYTES*s['engine_steps'])
        log=(path/'run.log').read_text()
        modes=re.findall(r'CUDA_GRAPH_RUNTIME_AUDIT[^\n]*runtime_mode=(\w+)',log)
        assert modes and set(modes)=={'NONE'}
        kernel=(path/'kernel.since.log').read_text()
        assert not any(token in kernel for token in ('Xid ','FAULT_PDE','FAULT_PTE','BUG:','Oops:'))
        verify_clean_snapshot(path/'gpu.after.json')
        cases.append({'arm':arm,'sample':int(sample),'directory':str(path),
                      'actual_qps':s['request_throughput_per_second'],
                      **{f'{m}_mean_ms':s[f'{m}_ms']['mean'] for m in ('ttft','tpot','e2e')},
                      'elapsed_s':s['elapsed_seconds'],'engine_steps':s['engine_steps'],
                      'expert_h2d_gib':copied/2**30,'expert_offload_bytes':NATIVE_BYTES,
                      'kv_capacity_tokens':34480,'resize_wall_ms':0})
    assert len(cases)==4*expected_samples
    return {**checked,'cases':cases}


def verify_kt(directory):
    r=load(directory/'result.json');c=r['configuration'];m=r['measurement']
    assert (directory/'returncode.txt').read_text().strip()=='0' and r['status']=='ok'
    fixture_path=Path(c['fixture']);assert hashlib.sha256(fixture_path.read_bytes()).hexdigest()==FIXTURE_SHA
    fixture=load(fixture_path)
    for key,value in {'arrival_qps':30,'prompt_tokens':1024,'output_tokens':16,'max_num_seqs':128,
                      'max_num_batched_tokens':8192,'max_model_len':1088,
                      'kv_budget_reference_bytes':2**30,'kv_effective_tokens':34480,
                      'kv_buffer_bytes_with_dummy_slot':34481*31104,'cuda_graph':False,'mixed_chunk':True}.items():
        assert c[key]==value,(key,c[key])
    actual=r['server_info_before']
    assert actual['max_total_num_tokens']==34480
    assert actual['max_running_requests']==128 and actual['disable_cuda_graph'] and actual['enable_mixed_chunk']
    assert actual['disable_radix_cache'] and actual['attention_backend']=='triton'
    assert c['placement']['cpu_experts_per_moe_layer']==8 and c['placement']['gpu_experts_per_moe_layer']==56
    assert c['placement']['selected_main_bytes']==3598712832
    assert c['placement']['exact_mask_verified_with_kt_kernel'] is True
    validate_requests(m['requests'],fixture,30)
    assert m['successful_requests']==128 and m['failed_requests']==0
    assert abs(m['actual_qps']-128/m['elapsed_seconds'])<1e-9
    for metric in ('ttft','tpot','e2e'):
        assert abs(m[f'{metric}_ms']['mean']-statistics.fmean(q[f'{metric}_ms'] for q in m['requests']))<1e-6
    log=(directory/'run.log').read_text()
    placements=re.findall(r'KT GPU experts: layer (\d+) \(MoE\) has (\d+) GPU experts',log)
    assert sorted((int(a),int(b)) for a,b in placements)==[(layer,56) for layer in range(1,27)]
    assert 'TRACE_END' in log
    kernel=(directory/'kernel.log').read_text()
    assert not re.search(r'NVRM: Xid|\bBUG:|\bOops:|GPU has fallen off|out of memory: Killed process',kernel,re.I)
    return {'arm':'ktransformers','sample':int(directory.name.removeprefix('sample')),'directory':str(directory),
            'actual_qps':m['actual_qps'],**{f'{x}_mean_ms':m[f'{x}_ms']['mean'] for x in ('ttft','tpot','e2e')},
            'elapsed_s':m['elapsed_seconds'],'expert_offload_bytes':3598712832,'kv_capacity_tokens':34480,
            'resize_wall_ms':0,'grouped_stream_events':m['grouped_stream_events'],
            'max_logged_running_requests':max(map(int,re.findall(r'#running-req: (\d+)',log)),default=0),
            'max_logged_queued_requests':max(map(int,re.findall(r'#queue-req: (\d+)',log)),default=0)}


def analyze(root):
    checked=verify_vllm(root/'vllm');cases=[];outputs={}
    for case in checked['cases']:
        if case['arm'].startswith('adaptive_'):
            summary,_,_,_=analyze_case(case,require_decode_allocations_success=False)
            summary['expert_offload_bytes']=3598712832
            cases.append(summary)
        else:cases.append(case)
        outputs[case['arm'],case['sample']]=load(Path(case['directory'])/'result.json')['requests']
    for sample in (1,2):
        path=root/f'ktransformers/sample{sample}';cases.append(verify_kt(path))
        outputs['ktransformers',sample]=load(path/'result.json')['measurement']['requests']
    rows=[]
    for arm,label in LABELS.items():
        selected=[c for c in cases if c['arm']==arm];assert len(selected)==2
        fields=('actual_qps','ttft_mean_ms','tpot_mean_ms','e2e_mean_ms','elapsed_s','resize_wall_ms')
        rows.append({'arm':arm,'label':label,**{k:statistics.fmean(c[k] for c in selected) for k in fields},
                     'expert_offload_bytes_at_start':selected[0]['expert_offload_bytes']})
    comparisons=[]
    pairs=[(f'{arm}_repeat',(arm,1),(arm,2)) for arm in LABELS]
    pairs += [(f'{a}_vs_{b}_s{s}',(a,s),(b,s)) for s in (1,2) for a,b in itertools.combinations(LABELS,2)]
    for label,a,b in pairs:
        left,right=outputs[a],outputs[b]
        comparisons.append({'comparison':label,'request_pairs':128,
                            'full_output_differences':sum(x['output_token_ids']!=y['output_token_ids'] for x,y in zip(left,right)),
                            'first_token_differences':sum(x['output_token_ids'][0]!=y['output_token_ids'][0] for x,y in zip(left,right))})
    status='ok_with_output_drift' if any(x['full_output_differences'] for x in comparisons) else 'ok'
    result={'status':status,'processes':10,'successful_requests':1280,'fixture_sha256':FIXTURE_SHA,
            'rows':rows,'cases':cases,'output_checks':comparisons,
            'capacity_certificate':load(root/'inputs/capacity_certificate.json')}
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');write_csv(root/'summary.csv',rows)
    lines=['# QPS30：固定1GiB四机制与AdaptMoE动态1→2GiB','',
           '全部重新采集：每机制两个独立进程，共10进程、1280请求。汇总表各指标为两进程指标的算术均值；单进程QPS=128/从首个计划到达到最后完成的时间。同一128条ShareGPT P1024/O16，QPS=30、到达i/30、C128、8192-token预算、上下文1088、BF16、混合调度、无CUDA Graph、关闭prefix cache。模型DeepSeek-V2-Lite-Chat。无额外真实conditioning，计时包含排队、排空和动态扩容，不是纯稳态。','',
           '固定组可用KV均为34,480 tokens。vLLM固定KV物理tensor为1,072,963,584B（2156块含null）；KTransformers为1,072,497,024B（page_size=1、含1个dummy token），可用token容量相同，dummy开销差异不足0.05%。AdaptMoE固定与动态均预留同一2GiB managed虚拟池，但固定组始终仅K1驻留/可分配；动态允许K1=2155块与K2=4313块，逐步校验真实目标、offline队列与物理驻留。','',
           'AdaptMoE固定与KTransformers采用相同cold8专家坐标，26层各8个CPU专家，共3,598,712,832B（3.3515625GiB）。Native保留整Parameter机制：3层完整专家参数加下一层w2，共7个Parameter、3,690,987,520B（3.4375GiB），比cold8多2.56%；UVA与Copy坐标和字节完全相同。动态升到K2时切cold11，每层11个专家、4,948,230,144B。','',
           'vLLM使用TRITON_MLA；KTransformers使用SGLang triton、56 GPU/8 CPU专家、BF16 CPU算子、14线程、单线程池、NUMA0，关闭动态专家更新；两框架attention内核与scheduler并不完全相同。vLLM为嵌入式同步replay，KTransformers为direct Engine async streaming，包含各自调度/IPC成本，属于系统配置比较。','',
           'vLLM执行顺序UVA→Copy→固定→动态→动态→固定→Copy→UVA，随后执行两次KTransformers；不声称五组完全交错。','',
           '| 机制 | 实际QPS | TTFT mean s | TPOT mean ms | E2E mean s |','|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['label']} | {r['actual_qps']:.3f} | {r['ttft_mean_ms']/1000:.3f} | {r['tpot_mean_ms']:.2f} | {r['e2e_mean_ms']/1000:.3f} |")
    by_arm={r['arm']:r for r in rows};fixed=by_arm['adaptive_fixed_1g'];dynamic=by_arm['adaptive_dynamic']
    lines+=['',f"动态相对固定K1：QPS提高{(dynamic['actual_qps']/fixed['actual_qps']-1)*100:.2f}%，TTFT降低{(1-dynamic['ttft_mean_ms']/fixed['ttft_mean_ms'])*100:.2f}%，但TPOT增加{(dynamic['tpot_mean_ms']/fixed['tpot_mean_ms']-1)*100:.2f}%，E2E增加{(dynamic['e2e_mean_ms']/fixed['e2e_mean_ms']-1)*100:.2f}%。吞吐和首token改善并不代表所有延迟指标都改善。两次均真实扩容一次，平均扩容{dynamic['resize_wall_ms']:.2f}ms；最大同时运行数由33增至66。KV增大、专家卸载增多和实际合批共同变化，不能单独归因于KV控制器。",'',
            'KTransformers两次QPS为0.965939/1.142690，波动明显。两次日志均提示缺少RTX5090/E=56的MoE调优配置，使用默认kernel配置；没有额外真实conditioning，结果包含首用初始化影响，不能视为充分预热或充分调优后的KTransformers上限。','']
    lines+=['','## 每个进程','','| 机制 | 样本 | QPS | TTFT s | TPOT ms | E2E s |','|---|---:|---:|---:|---:|---:|']
    for c in sorted(cases,key=lambda c:(list(LABELS).index(c['arm']),c['sample'])):
        lines.append(f"| {LABELS[c['arm']]} | {c['sample']} | {c['actual_qps']:.6f} | {c['ttft_mean_ms']/1000:.6f} | {c['tpot_mean_ms']:.6f} | {c['e2e_mean_ms']/1000:.6f} |")
    lines+=['','## AdaptMoE实际容量事件','','| 组/样本 | KV峰值GiB | 最大运行/等待 | KV分配失败 | 抢占 | 升/降档 | resize ms | forward H2D GiB |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in cases:
        if not c['arm'].startswith('adaptive_'):continue
        lines.append(f"| {c['arm']}/s{c['sample']} | {c['max_active_kv_gib']:.3f} | {c['max_running_requests']}/{c['max_waiting_requests']} | {c['allocation_failures']} | {c['preemptions']} | {c['growth_count']}/{c['shrink_count']} | {c['resize_wall_ms']:.2f} | {c['expert_h2d_gib']:.3f} |")
    lines+=['','分配失败是尝试次数，不等于最终请求失败。动态开销与真实专家/KV迁移均纳入计时，不做算术扣除。未做CUDA分时剖析，不能把全部差距归因于单一搬运或控制面成本。','',
            '## 输出差异','','| 对比 | 完整输出不同/128 | 首token不同/128 |','|---|---:|---:|']
    for c in comparisons:lines.append(f"| {c['comparison']} | {c['full_output_differences']} | {c['first_token_differences']} |")
    lines+=['','全部请求返回成功不等于严格逐token等价。保留所有输出差异，若输出不一致，本表仅作性能诊断，不能作为已通过数值等价验证的排名。每组两次新进程，不建立统计显著性。','',f'原始artifact：{root}','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    (root/'source_sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'source_snapshots').iterdir() if p.is_file()},indent=2)+'\n')
    for suffix,name in (('md','REPORT.md'),('json','summary.json'),('csv','summary.csv')):
        shutil.copy2(root/name,HERE/f'P1024_QPS30_KV1_FIVE_MECHANISMS_20261024.{suffix}')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--verify-kt-case',action='store_true')
    parser.add_argument('--output',type=Path);parser.add_argument('--report',type=Path);args=parser.parse_args()
    if args.prepare:prepare(args.root.resolve());raise SystemExit(0)
    try:
        if args.verify_kt_case:value=verify_kt(args.root.resolve())
        elif args.output:
            assert args.report
            value=verify_vllm(args.root.resolve());args.output.write_text(json.dumps(value,indent=2)+'\n')
            args.report.write_text('# vLLM fixed1/dynamic1-2 verification\n\n'+json.dumps(value,indent=2)+'\n')
        else:value=analyze(args.root.resolve())
    except BaseException as exc:
        if args.output:args.output.write_text(json.dumps({'status':'error','errors':[f'{type(exc).__name__}: {exc}']},indent=2)+'\n')
        raise
    print(json.dumps({'status':value.get('status'),'rows':value.get('rows'),'cases':len(value.get('cases',[]))},indent=2))
