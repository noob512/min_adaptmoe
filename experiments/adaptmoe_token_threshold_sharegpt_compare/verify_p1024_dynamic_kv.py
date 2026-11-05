#!/usr/bin/env python3
"""Verify P1024 dynamic KV and distinguish configured from observed tiers."""
import argparse
import json
import statistics
from pathlib import Path
from verify_dynamic_kv_p512 import verify as base_verify

FIXTURE_SHA='a4cc14060c94efb8c86f0524f2c49fbf85c0ff284eb12636e1e5c133d85bb9f2'
TARGETS={4313:'K2',8629:'K4',12943:'K6'}


def verify(root):
    config=json.loads((root/'run_config.json').read_text())
    assert config['arms']==['adaptive_dynamic'] and config['max_num_seqs']==128
    value=base_verify(root,selected_arm='adaptive_dynamic',expected_fixture_sha=FIXTURE_SHA,
                      expected_prompt_tokens=1024,expected_max_model_len=1088)
    cases=[];raw=[]
    for c in value['cases']:
        path=Path(c['directory']);r=json.loads((path/'result.json').read_text())
        t=r['kv_concurrency'];steps=t['scheduler_steps'];controller=r['dynamic_kv']['controller_steps']
        assert t['variant']=='dynamic' and t['initial_pool_blocks_including_null']==12944
        assert len(steps)==len(controller)==r['summary']['engine_steps']
        assert all(0<=x['active_blocks']<=8320 and 0<=x['running_requests']<=128 for x in steps)
        env=dict(line.split('=',1) for line in (path/'environment.txt').read_text().splitlines() if '=' in line)
        assert env['VLLM_EXPERIMENT_RESIZE_TELEMETRY']=='1'
        assert json.loads((root/'module.custom.json').read_text())['manual_migration_fault_mode']==2
        log=(path/'run.log').read_text()
        modes=[l.split('runtime_mode=',1)[1].split()[0] for l in log.splitlines()
               if 'CUDA_GRAPH_RUNTIME_AUDIT' in l and 'runtime_mode=' in l]
        assert modes and set(modes)=={'NONE'}
        transactions=r['dynamic_kv'].get('resize_transactions',[])
        for x in transactions:
            assert not x.get('exception') and x['result']['ok'] is True,x
            assert x['applied_target_blocks']==x['requested_target_blocks']
            assert x['applied_target_blocks'] in TARGETS and x['wall_ms']>=0
        start=r['measurement_window']['started_epoch']
        changes=[]
        for change in c['changes']:
            step=controller[change['step']-controller[0]['step']]
            match=[x for x in transactions if x['old_target_blocks']==change['old'] and x['applied_target_blocks']==change['new']
                   and x['controller_step']==change['step']-controller[0]['step']+1]
            changes.append({**change,'old_tier':TARGETS[change['old']],'new_tier':TARGETS[change['new']],
                            'observed_seconds':step['observed_epoch']-start,'active_blocks_after':step['active_blocks'],
                            'required_blocks_after':step['required_blocks'],
                            'transaction_wall_ms':sum(x['wall_ms'] for x in match) if match else None})
        confirmed={4313}|{x['applied_target_blocks'] for x in transactions}|set(c['used_targets'])
        cases.append({**c,'changes':changes,'transactions':transactions,'used_tiers':[TARGETS[x] for x in sorted(confirmed)],
                      'max_running_requests':max(x['running_requests'] for x in steps),
                      'max_waiting_requests':max(x['waiting_requests'] for x in steps),
                      'scheduler_max_active_blocks':max(x['active_blocks'] for x in steps),
                      'scheduler_max_active_kv_gib':max(x['active_blocks'] for x in steps)*497664/2**30,
                      'growth_count':sum(x['applied_target_blocks']>x['old_target_blocks'] for x in transactions),
                      'shrink_count':sum(x['applied_target_blocks']<x['old_target_blocks'] for x in transactions)})
        raw.append(r['requests'])
    assert len(cases)==2
    checks={'request_pairs':128,'full_output_differences':sum(a['output_token_ids']!=b['output_token_ids'] for a,b in zip(*raw)),
            'first_token_differences':sum(a['output_token_ids'][0]!=b['output_token_ids'][0] for a,b in zip(*raw))}
    return {**value,'status':'ok_with_output_drift' if checks['full_output_differences'] else 'ok',
            'cases':cases,'repeat_output_check':checks}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    try:result=verify(a.root.resolve())
    except BaseException as exc:
        a.output.write_text(json.dumps({'status':'error','errors':[f'{type(exc).__name__}: {exc}']},indent=2)+'\n');raise
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    a.report.write_text('# P1024/C128 automatic KV verification\n\n'+json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'cases':[{k:v for k,v in x.items() if k!='transactions'} for x in result['cases']]},indent=2))
