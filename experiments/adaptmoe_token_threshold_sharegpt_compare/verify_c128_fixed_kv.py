#!/usr/bin/env python3
"""Validate the fixed 1/2 GiB C128 replay, including measured token paths."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from verify_dynamic_kv_p512 import verify_clean_snapshot

ARMS={'adaptive_fixed_1g':1073741824,'adaptive_fixed_2g':2147483648}
FIXTURE_SHA='febf1babf6981abb9f1d65cfd600e33e7a8833ad799b87e09e66cc913b8edef7'
BLOCK_BYTES=497664


def verify(root):
    config=json.loads((root/'run_config.json').read_text())
    selected=config['arms'];samples=config['samples_per_arm']
    assert selected and all(arm in ARMS for arm in selected)
    assert config['max_num_seqs']==128 and samples in (1,2)
    assert config['arm_kv_cache_memory_bytes']=={arm:ARMS[arm] for arm in selected}
    fixture=Path(config['fixture'])
    assert hashlib.sha256(fixture.read_bytes()).hexdigest()==FIXTURE_SHA
    source=json.loads(fixture.read_text())['requests']
    cases=[];raw={}
    for line in (root/'execution_status.tsv').read_text().splitlines()[1:]:
        ordinal,sample,arm,directory,rc=line.split('\t')
        case=root/'arms'/directory
        value=json.loads((case/'result.json').read_text())
        assert arm in ARMS and rc=='0' and value['status']=='ok'
        assert value['mode']=='adaptmoe_copy_eager_continuous'
        budget=ARMS[arm];blocks=budget//BLOCK_BYTES
        c=value['configuration'];s=value['summary'];requests=value['requests']
        for key,expected in {'arrival_qps':config['arrival_qps'],'output_tokens':16,
                             'max_model_len':576,'max_num_batched_tokens':8192,'max_num_seqs':128,
                             'phase_mode':'token_threshold','phase_batched_scheduling':False,
                             'kv_cache_memory_bytes':budget,'latency_origin':'planned_arrival'}.items():
            assert c[key]==expected,(key,c[key],expected)
        assert c['compilation_config']=={'mode':'NONE','cudagraph_mode':'NONE'}
        assert len(requests)==s['successful_requests']==128 and s['generated_tokens']==2048
        assert [r['prompt_token_ids_sha256'] for r in requests]==config['prompt_token_hashes']
        for i,(r,src) in enumerate(zip(requests,source)):
            assert r['success'] and r['fixture_index']==i and len(r['output_token_ids'])==r['completion_tokens']==16
            assert r['prompt_token_ids_sha256']==src['prompt_token_ids_sha256']
            assert abs(r['planned_arrival_offset_seconds']-i/config['arrival_qps'])<1e-6
            assert len(r['itl_ms'])==15 and abs(r['tpot_ms']-statistics.fmean(r['itl_ms']))<1e-6
        assert abs(s['request_throughput_per_second']-128/s['elapsed_seconds'])<1e-9
        for metric in ('ttft','tpot','e2e'):
            assert abs(s[f'{metric}_ms']['mean']-statistics.fmean(r[f'{metric}_ms'] for r in requests))<1e-6
        t=value['kv_concurrency'];steps=t['scheduler_steps']
        assert t['variant']=='fixed' and 'dynamic_kv' not in value
        assert len(steps)==s['engine_steps'] and t['initial_pool_blocks_including_null']==blocks
        assert all(0<=x['active_blocks']<=min(blocks-1,128*33) and 0<=x['running_requests']<=128
                   and 0<=x['scheduled_tokens']<=8192 for x in steps)
        assert [x['step'] for x in steps]==list(range(1,len(steps)+1))
        schedule=value['scheduler_steps']
        assert [x['scheduled_tokens'] for x in steps]==[x['total_num_scheduled_tokens'] for x in schedule]
        env=dict(line.split('=',1) for line in (case/'environment.txt').read_text().splitlines() if '=' in line)
        assert env['VLLM_EXPERIMENT_KV_VARIANT']=='fixed'
        assert all(env[k]=='0' for k in ('VLLM_UVM_KV_AUTO_SCALE_ENABLE','VLLM_UVM_KV_DYNAMIC_POOL_ENABLE',
                                       'VLLM_UVM_KV_CACHE_MANAGED','VLLM_UVM_KV_RUNTIME_ENABLE',
                                       'VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE','VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE',
                                       'VLLM_UVM_PHASE_BATCHED_SCHEDULING'))
        assert env['VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_BYTES']=='2147483648'
        if env.get('VLLM_EXPERIMENT_FIXED_REMOTE_KERNEL_WARMUP')=='1':
            audits=json.loads((case/'startup_residency_audit.json').read_text())
            remote=next(x for x in audits if x['stage']=='kernel_warmup_all_remote_prepared')
            assert remote['residency']['ok'] and remote['residency']['gpu_bytes']==0
            repair=next(x for x in audits if x['stage']=='after_remote_kernel_warmup_restore')
            assert repair['residency']['ok'] and repair['residency']['gpu_bytes']==repair['hot_bytes']
            assert audits[-1]['stage']=='before_sampler_dummy_128'
            assert audits[-1]['residency']['gpu_bytes']==audits[-1]['hot_bytes']
            assert json.loads((root/'module.custom.json').read_text())['manual_migration_fault_mode']==2
        manifest=[json.loads(l) for l in (case/'cpu_offload_manifest.jsonl').read_text().splitlines() if l.strip()]
        offload=[e for e in manifest if e['event']=='cpu_offload_config']
        assert len(offload)==1
        offload=offload[0]
        assert offload['uvm_original_va_phase_mode']=='token_threshold'
        assert offload['uvm_original_va_token_threshold']==16 and offload['uvm_original_va_resume_token_threshold']==128
        assert offload['uvm_token_threshold_fast_transition'] is True
        finalized=[e for e in manifest if e['event']=='cpu_offload_uvm_static_alias_finalized']
        assert len(finalized)==1 and finalized[0]['cold_residency_bytes']==4948230144
        window=value['measurement_window']
        events=[e for e in manifest if window['started_epoch']<=e.get('timestamp',-1)<=window['finished_epoch']]
        assert all(e.get('graph_static_alias_capture_active') is not True for e in events)
        runtime_modes=[l.split('runtime_mode=',1)[1].split()[0] for l in (case/'run.log').read_text().splitlines()
                       if 'CUDA_GRAPH_RUNTIME_AUDIT' in l and 'runtime_mode=' in l]
        assert runtime_modes and set(runtime_modes)=={'NONE'}
        parks=[e for e in events if e['event']=='cpu_offload_uvm_original_va_remote_transition'
               and e.get('reason')=='token_threshold_remote']
        resumes=[e for e in events if e['event']=='cpu_offload_uvm_original_va_alias_prefill_activated'
                 and e.get('pre_state')=='remote_parked']
        assert all(e.get('ok') is True for e in parks+resumes)
        assert all(e.get('fast_runtime_transition') is True and e.get('post_map_physical_audit_enabled') is False for e in parks)
        assert not any(e['event']=='cpu_offload_uvm_hot_cuda_read_gate_begin' and
                       str(e.get('gate_stage','')).startswith(('runtime_remote_','runtime_prefill_')) for e in events)
        verify_clean_snapshot(case/'gpu.after.json')
        assert not any(k in (case/'kernel.since.log').read_text() for k in ('Xid ','FAULT_PDE','FAULT_PTE','BUG:','Oops:'))
        cases.append({'ordinal':int(ordinal),'sample':int(sample),'arm':arm,'kv_gib':budget//2**30,
                      'kv_cache_memory_bytes':budget,'pool_blocks_including_null':blocks,'directory':str(case),
                      'offered_qps':config['arrival_qps'],'actual_qps':s['request_throughput_per_second'],
                      'ttft_mean_ms':s['ttft_ms']['mean'],'tpot_mean_ms':s['tpot_ms']['mean'],
                      'e2e_mean_ms':s['e2e_ms']['mean'],'engine_steps':s['engine_steps'],
                      'max_running_requests':max(x['running_requests'] for x in steps),
                      'max_waiting_requests':max(x['waiting_requests'] for x in steps),
                      'max_active_blocks':max(x['active_blocks'] for x in steps),
                      'preemptions':sum(x['preempted_requests'] for x in steps),
                      'observation_cpu_ms':sum(x['observation_cpu_ms'] for x in steps),
                      'park_count':len(parks),'resume_count':len(resumes),'errors':[]})
        raw[int(sample),arm]=requests
    expected=[(s,arm) for s in range(1,samples+1) for arm in (selected if s%2 else selected[::-1])]
    assert [(c['sample'],c['arm']) for c in cases]==expected
    checks=[]
    for sample in range(1,samples+1) if len(selected)==2 else ():
        left,right=raw[sample,'adaptive_fixed_1g'],raw[sample,'adaptive_fixed_2g']
        checks.append({'sample':sample,'requests':128,
                       'full_output_differences':sum(x['output_token_ids']!=y['output_token_ids'] for x,y in zip(left,right)),
                       'first_token_differences':sum(x['output_token_ids'][0]!=y['output_token_ids'][0] for x,y in zip(left,right))})
    verify_clean_snapshot(root/'module.after.json')
    return {'status':'ok_with_output_drift' if any(x['full_output_differences'] for x in checks) else 'ok',
            'errors':[],'cases':cases,'output_checks':checks,'protocol':config}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    try:
        result=verify(a.root.resolve())
    except BaseException as exc:
        a.output.write_text(json.dumps({'status':'error','errors':[f'{type(exc).__name__}: {exc}']},indent=2)+'\n')
        raise
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    a.report.write_text('# C128 fixed 1/2 GiB KV verification\n\n'+json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'cases':result['cases']},indent=2))
