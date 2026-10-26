#!/usr/bin/env python3
"""Verify actual automatic KV decisions as well as the original replay metrics."""
import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


def verify_clean_snapshot(path):
    original=json.loads(path.read_text())
    if original['gpu_clean'] is True:
        return
    # Keep the original snapshot. Accept only a separate, later clean proof
    # for a zero-memory/no-compute transient client, as in verify_compare.py.
    recheck=json.loads(path.with_name(path.stem+'.recheck.json').read_text())
    assert original['memory_used_mib']==0 and original['compute_pids']==[]
    assert original['uvm_client_pids']
    assert recheck['gpu_clean'] is True
    assert recheck['memory_used_mib']==0 and recheck['compute_pids']==[] and recheck['uvm_client_pids']==[]
    assert recheck['timestamp']>=original['timestamp']
    assert recheck['loaded_srcversion']==original['loaded_srcversion']


def verify(root, selected_arm='adaptive', *,
           expected_fixture_sha='febf1babf6981abb9f1d65cfd600e33e7a8833ad799b87e09e66cc913b8edef7',
           expected_prompt_tokens=512, expected_max_model_len=576,
           expected_autoscale=True, expected_target_blocks=(4313,8629,12943),
           expected_initial_target=4313, expected_bootstrap_target=None,
           expected_pool_blocks=12943, expected_kv_reservation_bytes=6442450944,
           expected_plan_classes=('cold11','cold17','cold23'), expected_samples=2):
    assert tuple(expected_target_blocks) in ((2155,4313), (4313,8629), (4313,8629,12943))
    assert expected_initial_target in expected_target_blocks
    if expected_bootstrap_target is None:
        expected_bootstrap_target = expected_initial_target
    assert expected_bootstrap_target in expected_target_blocks
    all_classes={2155:'cold8',4313:'cold11',8629:'cold17',12943:'cold23'}
    initial_class=all_classes[expected_initial_target]
    config = json.loads((root/'run_config.json').read_text())
    fixture = json.loads(Path(config['fixture']).read_text())
    assert hashlib.sha256(Path(config['fixture']).read_bytes()).hexdigest() == expected_fixture_sha
    assert fixture['metadata']['prompt_tokens']==expected_prompt_tokens
    assert all(r['prompt_tokens']==len(r['prompt_token_ids'])==expected_prompt_tokens for r in fixture['requests'])
    cases = []
    for line in (root/'execution_status.tsv').read_text().splitlines()[1:]:
        ordinal,sample,arm,directory,rc = line.split('\t')
        if arm != selected_arm:
            continue
        case = root/'arms'/directory
        value = json.loads((case/'result.json').read_text())
        assert rc == '0' and value['status'] == 'ok', value.get('error')
        assert selected_arm in config['arms']
        c = value['configuration']
        for key,expected in {'arrival_qps':config['arrival_qps'], 'output_tokens':16,
                             'max_model_len':expected_max_model_len,'max_num_batched_tokens':8192,'max_num_seqs':config['max_num_seqs'],
                             'phase_mode':'token_threshold','phase_batched_scheduling':False,
                             'kv_cache_memory_bytes':expected_kv_reservation_bytes}.items():
            assert c[key] == expected,(key,c[key])
        assert c['compilation_config'] == {'mode':'NONE','cudagraph_mode':'NONE'}
        requests = value['requests']
        summary = value['summary']
        assert summary['successful_requests'] == len(requests) == 128
        assert summary['generated_tokens'] == 2048
        for i,(request,source) in enumerate(zip(requests,fixture['requests'])):
            assert request['fixture_index'] == i and request['success']
            assert request['prompt_token_ids_sha256'] == source['prompt_token_ids_sha256']
            assert len(request['output_token_ids']) == request['completion_tokens'] == 16
            assert abs(request['planned_arrival_offset_seconds']-i/config['arrival_qps']) < 1e-6
            assert abs(request['tpot_ms']-statistics.fmean(request['itl_ms'])) < 1e-6
            assert len(request['itl_ms']) == 15
        assert abs(summary['request_throughput_per_second']-128/summary['elapsed_seconds']) < 1e-9
        for metric in ('ttft','tpot','e2e'):
            assert abs(summary[f'{metric}_ms']['mean']-statistics.fmean(r[f'{metric}_ms'] for r in requests)) < 1e-6
        dynamic = value['dynamic_kv']
        assert dynamic['policy_total_blocks'] == expected_pool_blocks
        startup=json.loads((case/'kv_startup_residency.json').read_text())
        assert startup['result']['offload_complete'] is True
        assert all(a['unaligned_start_count']==a['unaligned_end_count']==0 for a in startup['audits'])
        initial = dynamic['initial_controller']
        assert initial['enabled'] is expected_autoscale and initial['scheduler_boundary_resize_enabled']
        assert initial['current_target_blocks'] == expected_initial_target
        assert initial['config']['allowed_target_blocks'] == list(expected_target_blocks)
        assert initial['config']['maximum_blocks'] == expected_target_blocks[-1]
        assert initial['config']['active_high_watermark_growth']
        assert initial['config']['high_watermark'] == .8
        assert dynamic['initial_plan']['active_class'] == initial_class
        assert set(dynamic['initial_plan']['classes']) == set(expected_plan_classes)
        records = dynamic['controller_steps']
        assert len(records) == summary['engine_steps']
        if expected_autoscale:
            assert all(r['ok'] is True and r['queue_invariant_ok'] is True for r in records)
            assert [r['step'] for r in records] == list(range(records[0]['step'],records[0]['step']+len(records)))
        else:
            assert initial['step'] == 0 and initial['last_decision'] == {}
            assert all(r['step'] is None and r['action'] is None for r in records)
            assert all(r['applied_target_blocks'] == expected_initial_target and r['active_plan_class'] == initial_class for r in records)
            assert dynamic['resize_transactions'] == []
        targets = [expected_initial_target]+[r['applied_target_blocks'] for r in records]
        mapping={target:all_classes[target] for target in expected_target_blocks}
        assert all(r['applied_target_blocks'] in mapping and r['active_plan_class']==mapping[r['applied_target_blocks']] for r in records)
        changes = [{'step':records[i]['step'],'old':old,'new':new} for i,(old,new) in enumerate(zip(targets,targets[1:])) if old != new]
        assert all(abs(list(mapping).index(x['old'])-list(mapping).index(x['new']))==1 for x in changes)
        schedule=value['scheduler_steps']
        assert all(0 <= s['total_num_scheduled_tokens'] <= 8192 for s in schedule)
        assert max(len(s['scheduled_new_request_ids'])+len(s['scheduled_cached_request_ids']) for s in schedule) <= config['max_num_seqs']
        manifest=[json.loads(l) for l in (case/'cpu_offload_manifest.jsonl').read_text().splitlines() if l.strip()]
        offload=next(x for x in manifest if x['event']=='cpu_offload_config')
        assert offload['uvm_original_va_phase_mode']=='token_threshold'
        assert offload['uvm_original_va_token_threshold']==16 and offload['uvm_original_va_resume_token_threshold']==128
        assert offload['uvm_token_threshold_fast_transition'] is True
        finalized=[x for x in manifest if x['event']=='cpu_offload_uvm_static_alias_finalized']
        assert len(finalized)==1 and finalized[0]['cold_residency_bytes']=={2155:3598712832,4313:4948230144,8629:7647264768,12943:10346299392}[expected_bootstrap_target]
        window=value['measurement_window']
        events=[x for x in manifest if window['started_epoch'] <= x.get('timestamp',-1) <= window['finished_epoch']]
        parks=[x for x in events if x['event']=='cpu_offload_uvm_original_va_remote_transition' and x.get('reason')=='token_threshold_remote']
        resumes=[x for x in events if x['event']=='cpu_offload_uvm_original_va_alias_prefill_activated' and x.get('pre_state')=='remote_parked']
        assert all(x.get('ok') is True for x in parks+resumes)
        assert all(x.get('fast_runtime_transition') is True and x.get('post_map_physical_audit_enabled') is False for x in parks)
        assert not any(x['event']=='cpu_offload_uvm_hot_cuda_read_gate_begin' and str(x.get('gate_stage','')).startswith(('runtime_remote_','runtime_prefill_')) for x in events)
        kernel=(case/'kernel.since.log').read_text()
        assert not any(x in kernel for x in ('Xid ','FAULT_PDE','FAULT_PTE','BUG:','Oops:'))
        verify_clean_snapshot(case/'gpu.after.json')
        cases.append({'sample':int(sample),'directory':str(case),'offered_qps':config['arrival_qps'],
                      'actual_qps':summary['request_throughput_per_second'],
                      'ttft_mean_ms':summary['ttft_ms']['mean'],'tpot_mean_ms':summary['tpot_ms']['mean'],
                      'e2e_mean_ms':summary['e2e_ms']['mean'],'engine_steps':summary['engine_steps'],
                      'used_targets':sorted(set(targets)),'changes':changes,
                      'max_active_blocks':max(r['active_blocks'] for r in records) if expected_autoscale else max(r['active_blocks'] for r in value['kv_concurrency']['scheduler_steps']),
                      'max_required_blocks':max(r['required_blocks'] for r in records) if expected_autoscale else None,
                      'min_high_threshold_blocks':min(r['high_threshold_blocks'] for r in records) if expected_autoscale else None,
                      'max_k2_active_fraction':max(r['active_blocks']/r['applied_target_blocks'] for r in records) if expected_autoscale else None,
                      'preemptions':sum(r['preemption_count'] for r in records) if expected_autoscale else sum(r['preempted_requests'] for r in value['kv_concurrency']['scheduler_steps']),
                      'observation_cpu_ms':sum(r['observation_cpu_ms'] for r in records),
                      'max_request_itl_ms':max(t for r in requests for t in r['itl_ms']),
                      'actions':dict(Counter(r['action'] for r in records)),
                      'park_count':len(parks),'resume_count':len(resumes),
                      'errors':[]})
    assert len(cases) == config['samples_per_arm'] == expected_samples
    assert sorted(c['sample'] for c in cases) == list(range(1, expected_samples+1))
    verify_clean_snapshot(root/'module.after.json')
    return {'status':'ok','errors':[],'cases':cases,'protocol':config}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=verify(args.run_root.resolve())
    except BaseException as exc:
        args.output.write_text(json.dumps({'status':'error','errors':[f'{type(exc).__name__}: {exc}']},indent=2)+'\n')
        raise
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    args.report.write_text('# Dynamic K2/K4/K6 replay verification\n\n'+json.dumps(result['cases'],indent=2)+'\n')
    print(json.dumps(result['cases'],indent=2))
