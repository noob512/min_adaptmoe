#!/usr/bin/env python3
"""Summarize the completed first QPS2 samples after user-cancelled repeats."""
import argparse
import hashlib
import itertools
import json
import re
import shutil
import statistics
from pathlib import Path

from analyze_qps30_kv_pressure import analyze_case, write_csv
from five_arm_k1_compare import NATIVE_BYTES, validate_requests
from kv1_qps_sweep_compare import LABELS
from verify_dynamic_kv_p512 import verify_clean_snapshot
from verify_p1024_dynamic_kv import FIXTURE_SHA

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text())


def main(root):
    run = root/'qps2/vllm'
    config = load(run/'run_config.json')
    assert config['arrival_qps'] == 2 and config['fixture_sha256'] == FIXTURE_SHA
    assert config['arms'] == list(LABELS)
    fixture_path = Path(config['fixture'])
    assert hashlib.sha256(fixture_path.read_bytes()).hexdigest() == FIXTURE_SHA
    fixture = load(fixture_path)
    cert = load(root/'qps2/inputs/capacity_certificate.json')
    rows, cases, outputs, excluded = [], [], {}, []
    for line in (run/'execution_status.tsv').read_text().splitlines()[1:]:
        _, sample, arm, directory, rc = line.split('\t')
        path = run/'arms'/directory
        if sample != '1':
            excluded.append({'arm': arm, 'sample': int(sample), 'returncode': int(rc), 'directory': str(path)})
            continue
        raw = load(path/'result.json'); c = raw['configuration']; s = raw['summary']
        assert rc == '0' and raw['status'] == 'ok'
        for key, expected in {'arrival_qps': 2, 'max_num_seqs': 128, 'max_model_len': 1088,
                              'max_num_batched_tokens': 8192, 'output_tokens': 16,
                              'phase_batched_scheduling': False}.items():
            assert c[key] == expected, (key, c[key])
        assert c['compilation_config'] == {'mode': 'NONE', 'cudagraph_mode': 'NONE'}
        validate_requests(raw['requests'], fixture, 2)
        assert s['successful_requests'] == 128 and s['generated_tokens'] == 2048
        assert abs(s['request_throughput_per_second']-128/s['elapsed_seconds']) < 1e-9
        for m in ('ttft', 'tpot', 'e2e'):
            assert abs(s[f'{m}_ms']['mean']-statistics.fmean(r[f'{m}_ms'] for r in raw['requests'])) < 1e-6
        log = (path/'run.log').read_text()
        assert set(re.findall(r'CUDA_GRAPH_RUNTIME_AUDIT[^\n]*runtime_mode=(\w+)', log)) == {'NONE'}
        assert not any(t in (path/'kernel.since.log').read_text() for t in ('Xid ', 'FAULT_PDE', 'FAULT_PTE', 'BUG:', 'Oops:'))
        verify_clean_snapshot(path/'gpu.after.json')
        row = {'offered_qps': 2, 'arm': arm, 'label': LABELS[arm],
               'actual_qps': s['request_throughput_per_second'],
               **{f'{m}_mean_ms': s[f'{m}_ms']['mean'] for m in ('ttft', 'tpot', 'e2e')},
               'elapsed_s': s['elapsed_seconds'], 'resize_wall_ms': 0,
               'samples': 1, 'source': 'new_first_sample', 'directory': str(path)}
        events = [json.loads(t) for t in (path/'cpu_offload_manifest.jsonl').read_text().splitlines() if t.strip()]
        if arm.startswith('native_'):
            assert c['kv_cache_memory_bytes'] == 2**30 and c['phase_mode'] == 'off'
            assert raw['fixed_kv_capacity'] == {'pool_blocks_including_null': 2156, 'usable_blocks': 2155,
                    'usable_tokens': 34480, 'kv_tensor_bytes': 2156*497664, 'managed_kv': False, 'autoscaling': False}
            parameters = [e for e in events if e['event'] == 'cpu_offload_parameter']
            assert len(parameters) == 7 and sum(e['bytes'] for e in parameters) == NATIVE_BYTES
            assert sorted(e['full_parameter_name'] for e in parameters) == sorted(cert['native_parameter_names'])
            assert all(e['offload_backend'] == ('uva' if arm == 'native_uva' else 'copy') for e in parameters)
            w = raw['measurement_window']
            copied = sum(e.get('copy_bytes', 0) for e in events if e['event'] == 'cpu_offload_copy_forward'
                         and w['started_epoch'] <= e['timestamp'] <= w['finished_epoch'])
            assert copied == (0 if arm == 'native_uva' else NATIVE_BYTES*s['engine_steps'])
            row['expert_h2d_gib'] = copied/2**30
        else:
            assert c['kv_cache_memory_bytes'] == 2**31 and c['phase_mode'] == 'token_threshold'
            d = raw['dynamic_kv']; automatic = arm == 'adaptive_dynamic'
            assert d['initial_controller']['enabled'] is automatic
            assert d['initial_controller']['current_target_blocks'] == 2155
            assert d['initial_plan']['active_class'] == 'cold8' and d['resize_transactions'] == []
            assert all(r['applied_target_blocks'] == 2155 and r['active_plan_class'] == 'cold8' for r in d['controller_steps'])
            assert raw['kv_concurrency']['initial_pool_blocks_including_null'] == 4314
            t = raw['kv_concurrency']['scheduler_steps']
            assert all(x['active_blocks']+sum(x['free_counts'].values()) == 4313 and
                       x['free_counts']['offline_block_queue'] == 2158 for x in t)
            startup = load(path/'kv_startup_residency.json')
            assert startup['target_blocks'] == 2155 and startup['result']['offload_complete']
            gpu, cpu = [a['residency'] for a in startup['audits']]
            assert gpu['ok'] and gpu['gpu_bytes'] == gpu['queried_bytes'] == 2156*497664
            assert cpu['ok'] and cpu['cpu_bytes'] == cpu['queried_bytes'] == 2158*497664
            finalized = [e for e in events if e['event'] == 'cpu_offload_uvm_static_alias_finalized']
            assert len(finalized) == 1 and finalized[0]['cold_residency_bytes'] == 3598712832
            case, _, _, _ = analyze_case({**row, 'sample': 1, 'engine_steps': s['engine_steps']}, require_decode_allocations_success=False)
            row.update({k: case[k] for k in ('expert_h2d_gib', 'copy_steps', 'remote_steps', 'allocation_failures', 'preemptions')})
            row.update({'max_active_kv_gib': max(x['active_blocks'] for x in t)*497664/2**30,
                        'max_running_requests': max(x['running_requests'] for x in t),
                        'max_waiting_requests': max(x['waiting_requests'] for x in t),
                        'growth_counts': [0], 'shrink_counts': [0], 'used_tiers': ['K1']})
            cases.append(case)
        rows.append(row); outputs[arm] = raw['requests']
    assert len(rows) == 4 and set(outputs) == set(LABELS)
    rows.sort(key=lambda r: list(LABELS).index(r['arm']))
    checks = [{'comparison': f'{a}_vs_{b}', 'request_pairs': 128,
               'full_output_differences': sum(x['output_token_ids'] != y['output_token_ids'] for x,y in zip(outputs[a],outputs[b])),
               'first_token_differences': sum(x['output_token_ids'][0] != y['output_token_ids'][0] for x,y in zip(outputs[a],outputs[b]))}
              for a,b in itertools.combinations(LABELS,2)]
    previous = load(HERE/'P1024_KV1_QPS4_8_10_20_30_FOUR_MECHANISMS_20261024.json')
    assert previous['fixture_sha256'] == FIXTURE_SHA and len(previous['rows']) == 20
    assert all(r['samples'] == 2 for r in previous['rows'])
    combined = rows+previous['rows']
    result = {'status': 'ok_with_output_drift' if any(c['full_output_differences'] for c in checks) else 'ok',
              'qps2_successful_requests': 512, 'qps2_included_sample': 1,
              'qps2_repeat_stop_reason': 'User requested no repeats; only the symmetric first sample is included.',
              'qps2_excluded_repeats': excluded, 'qps2_rows': rows, 'rows': combined,
              'qps2_output_checks': checks, 'qps2_adapt_cases': cases,
              'previous_summary': str(HERE/'P1024_KV1_QPS4_8_10_20_30_FOUR_MECHANISMS_20261024.json'),
              'previous_output_status': previous['status'], 'fixture_sha256': FIXTURE_SHA,
              'total_included_requests': 512+previous['total_successful_requests']}
    fields = list(dict.fromkeys(k for r in combined for k in r))
    (root/'summary_single.json').write_text(json.dumps(result, indent=2)+'\n')
    write_csv(root/'summary_single.csv', [{k:r.get(k,'') for k in fields} for r in combined])
    lines = ['# P1024：QPS2补测与既有QPS4/8/10/20/30汇总', '',
             'QPS=2统一取四机制首轮单次进程，各128请求，共512条成功；按用户要求停止第二轮。第二轮已完成的动态样本保留但不纳入汇总，中断的固定样本也不纳入。其他QPS保持上一轮两次进程均值，数值未更改。', '',
             '同一128条ShareGPT、1024输入/16 greedy输出、DeepSeek-V2-Lite-Chat BF16、C128、8192-token预算、TRITON_MLA、混合调度、无prefix cache、无CUDA Graph；固定KV1GiB，动态允许KV1→2GiB。PARK≤16、RESUME≥128。计时包含从首个计划到达到尾部排空，无额外真实conditioning，不是纯稳态。', '',
             '固定AdaptMoE卸载cold8（3,598,712,832B）；Native整Parameter卸载3,690,987,520B，多2.56%。动态升K2时配套cold11；本次QPS2未发生升档。', '',
             '| 提供QPS | 机制 | 样本数 | 实际QPS | TTFT mean s | TPOT mean ms | E2E mean s |',
             '|---:|---|---:|---:|---:|---:|---:|']
    for r in combined:
        lines.append(f"| {r['offered_qps']} | {r['label']} | {r['samples']} | {r['actual_qps']:.3f} | {r['ttft_mean_ms']/1000:.3f} | {r['tpot_mean_ms']:.2f} | {r['e2e_mean_ms']/1000:.3f} |")
    lines += ['', 'QPS2下固定与动态AdaptMoE最大同时运行数均为3，等待请求峰值为0，KV占用峰值均为'+f"{rows[2]['max_active_kv_gib']:.6f}GiB。动态组始终保持K1，未扩容。", '',
              '## QPS2首轮输出比较', '', '| 对比 | 完整输出不同/128 | 首token不同/128 |', '|---|---:|---:|']
    for c in checks: lines.append(f"| {c['comparison']} | {c['full_output_differences']} | {c['first_token_differences']} |")
    lines += ['', '全部请求返回成功不等于严格逐token等价。输出存在差异，数据属于性能诊断；QPS2为单次样本，其他档位为两次均值，不建立统计显著性。', '', f'QPS2原始artifact：{root}', '']
    (root/'REPORT_SINGLE.md').write_text('\n'.join(lines))
    stem = 'P1024_KV1_QPS2_4_8_10_20_30_FOUR_MECHANISMS_20261024'
    for suffix, source in (('md','REPORT_SINGLE.md'),('json','summary_single.json'),('csv','summary_single.csv')):
        shutil.copy2(root/source, HERE/f'{stem}.{suffix}')
    print(json.dumps({'status': result['status'], 'qps2_rows': rows, 'qps2_output_checks': checks},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root.resolve())
