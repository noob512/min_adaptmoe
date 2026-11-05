#!/usr/bin/env python3
"""Validate fixed and automatic KV with identical C128 arrivals."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

from verify_dynamic_kv_p512 import verify as verify_dynamic, verify_clean_snapshot


def verify(root):
    config=json.loads((root/'run_config.json').read_text())
    assert config['arms']==['adaptive_fixed','adaptive_dynamic']
    assert config['max_num_seqs']==128 and config['samples_per_arm']==2
    assert config['arm_kv_cache_memory_bytes']=={'adaptive_fixed':536870912,'adaptive_dynamic':6442450944}
    fixture=Path(config['fixture'])
    assert hashlib.sha256(fixture.read_bytes()).hexdigest()=='febf1babf6981abb9f1d65cfd600e33e7a8833ad799b87e09e66cc913b8edef7'
    dynamic=verify_dynamic(root,selected_arm='adaptive_dynamic')
    dynamic_by_dir={Path(c['directory']).name:c for c in dynamic['cases']}
    cases=[]
    raw={}
    for line in (root/'execution_status.tsv').read_text().splitlines()[1:]:
        ordinal,sample,arm,directory,rc=line.split('\t')
        case=root/'arms'/directory
        value=json.loads((case/'result.json').read_text())
        assert rc=='0' and value['status']=='ok'
        expected_kv=config['arm_kv_cache_memory_bytes'][arm]
        c=value['configuration']
        for key,expected in {'arrival_qps':config['arrival_qps'],'output_tokens':16,
                             'max_model_len':576,'max_num_batched_tokens':8192,'max_num_seqs':128,
                             'phase_mode':'token_threshold','phase_batched_scheduling':False,
                             'kv_cache_memory_bytes':expected_kv}.items():
            assert c[key]==expected,(key,c[key])
        assert c['compilation_config']=={'mode':'NONE','cudagraph_mode':'NONE'}
        requests=value['requests'];s=value['summary']
        assert len(requests)==s['successful_requests']==128 and s['generated_tokens']==2048
        assert [r['prompt_token_ids_sha256'] for r in requests]==config['prompt_token_hashes']
        for i,r in enumerate(requests):
            assert r['success'] and r['fixture_index']==i and len(r['output_token_ids'])==r['completion_tokens']==16
            assert abs(r['planned_arrival_offset_seconds']-i/config['arrival_qps'])<1e-6
            assert len(r['itl_ms'])==15 and abs(r['tpot_ms']-statistics.fmean(r['itl_ms']))<1e-6
        assert abs(s['request_throughput_per_second']-128/s['elapsed_seconds'])<1e-9
        for metric in ('ttft','tpot','e2e'):
            assert abs(s[f'{metric}_ms']['mean']-statistics.fmean(r[f'{metric}_ms'] for r in requests))<1e-6
        t=value['kv_concurrency'];steps=t['scheduler_steps']
        variant='fixed' if arm=='adaptive_fixed' else 'dynamic'
        assert t['variant']==variant and len(steps)==s['engine_steps']
        assert t['initial_pool_blocks_including_null']==(1078 if variant=='fixed' else 12944)
        assert all(0<=x['active_blocks']<=128*33 and 0<=x['running_requests']<=128 and 0<=x['scheduled_tokens']<=8192 for x in steps)
        assert [x['step'] for x in steps]==list(range(1,len(steps)+1))
        manifest=[json.loads(l) for l in (case/'cpu_offload_manifest.jsonl').read_text().splitlines() if l.strip()]
        finalized=[e for e in manifest if e['event']=='cpu_offload_uvm_static_alias_finalized']
        assert len(finalized)==1 and finalized[0]['cold_residency_bytes']==4948230144
        env=dict(line.split('=',1) for line in (case/'environment.txt').read_text().splitlines() if '=' in line)
        assert env['VLLM_EXPERIMENT_KV_VARIANT']==variant
        if variant=='fixed':
            assert 'dynamic_kv' not in value
            assert all(env[k]=='0' for k in ('VLLM_UVM_KV_AUTO_SCALE_ENABLE','VLLM_UVM_KV_DYNAMIC_POOL_ENABLE','VLLM_UVM_KV_CACHE_MANAGED','VLLM_UVM_KV_RUNTIME_ENABLE'))
        window=value['measurement_window']
        events=[e for e in manifest if window['started_epoch']<=e.get('timestamp',-1)<=window['finished_epoch']]
        transitions=[e for e in events if e['event'] in ('cpu_offload_uvm_original_va_remote_transition','cpu_offload_uvm_original_va_alias_prefill_activated')]
        assert all(e.get('ok') is True for e in transitions)
        assert not any(e['event']=='cpu_offload_uvm_hot_cuda_read_gate_begin' and str(e.get('gate_stage','')).startswith(('runtime_remote_','runtime_prefill_')) for e in events)
        verify_clean_snapshot(case/'gpu.after.json')
        assert not any(k in (case/'kernel.since.log').read_text() for k in ('Xid ','FAULT_PDE','FAULT_PTE','BUG:','Oops:'))
        extra=dynamic_by_dir.get(directory,{})
        cases.append({'ordinal':int(ordinal),'sample':int(sample),'arm':arm,'variant':variant,'directory':str(case),
                      'offered_qps':config['arrival_qps'],'actual_qps':s['request_throughput_per_second'],
                      'ttft_mean_ms':s['ttft_ms']['mean'],'tpot_mean_ms':s['tpot_ms']['mean'],
                      'e2e_mean_ms':s['e2e_ms']['mean'],'engine_steps':s['engine_steps'],
                      'max_running_requests':max(x['running_requests'] for x in steps),
                      'max_waiting_requests':max(x['waiting_requests'] for x in steps),
                      'max_active_blocks':max(x['active_blocks'] for x in steps),
                      'preemptions':sum(x['preempted_requests'] for x in steps),
                      'used_targets':extra.get('used_targets',[1077]),'changes':extra.get('changes',[]),
                      'controller_max_required_blocks':extra.get('max_required_blocks'),
                      'observation_cpu_ms':sum(x['observation_cpu_ms'] for x in steps)+extra.get('observation_cpu_ms',0),
                      'errors':[]})
        raw[(int(sample),variant)]=requests
    assert len(cases)==4
    assert [c['arm'] for c in cases]==['adaptive_fixed','adaptive_dynamic','adaptive_dynamic','adaptive_fixed']
    output_checks=[]
    for sample in (1,2):
        a,b=raw[sample,'fixed'],raw[sample,'dynamic']
        output_checks.append({'sample':sample,'requests':128,
                              'full_output_differences':sum(x['output_token_ids']!=y['output_token_ids'] for x,y in zip(a,b)),
                              'first_token_differences':sum(x['output_token_ids'][0]!=y['output_token_ids'][0] for x,y in zip(a,b))})
    verify_clean_snapshot(root/'module.after.json')
    return {'status':'ok_with_output_drift' if any(x['full_output_differences'] for x in output_checks) else 'ok',
            'errors':[],'cases':cases,'output_checks':output_checks,'protocol':config}


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
    a.report.write_text('# C128 fixed/dynamic KV verification\n\n'+json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'cases':result['cases']},indent=2))
