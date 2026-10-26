#!/usr/bin/env python3
"""Validate locked K2 and automatic K2/K4/K6 on the identical P1024 replay."""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

from verify_dynamic_kv_p512 import verify as base_verify
from verify_p1024_dynamic_kv import FIXTURE_SHA, TARGETS


def verify(root, *, expected_target_blocks=(4313,8629,12943)):
    config = json.loads((root / 'run_config.json').read_text())
    assert config['arms'] == ['adaptive_fixed', 'adaptive_dynamic']
    assert config['max_num_seqs'] == 128 and config['samples_per_arm'] == 2
    assert config['arm_kv_cache_memory_bytes'] == {
        'adaptive_fixed': 6442450944, 'adaptive_dynamic': 6442450944}
    fixture = Path(config['fixture'])
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == FIXTURE_SHA
    assert json.loads((root / 'module.custom.json').read_text())['manual_migration_fault_mode'] == 2
    cases, raw, settings = [], {}, {}
    for arm, variant in (('adaptive_fixed', 'fixed_managed'), ('adaptive_dynamic', 'dynamic')):
        automatic = variant == 'dynamic'
        value = base_verify(root, selected_arm=arm, expected_fixture_sha=FIXTURE_SHA,
                            expected_prompt_tokens=1024, expected_max_model_len=1088,
                            expected_autoscale=automatic, expected_target_blocks=expected_target_blocks)
        for case in value['cases']:
            directory = Path(case['directory'])
            result = json.loads((directory / 'result.json').read_text())
            telemetry = result['kv_concurrency']
            steps = telemetry['scheduler_steps']
            assert telemetry['variant'] == variant
            assert telemetry['initial_pool_blocks_including_null'] == 12944
            assert len(steps) == result['summary']['engine_steps']
            assert all(0 <= s['active_blocks'] <= (8320 if automatic else 4313)
                       and 0 <= s['running_requests'] <= 128 for s in steps)
            for step in steps:
                assert step['active_blocks'] + sum(step['free_counts'].values()) == 12943
            runtime = json.loads((directory / 'runtime_settings.json').read_text())
            assert runtime['VLLM_UVM_KV_AUTO_SCALE_ENABLE'] == ('1' if automatic else '0')
            assert runtime['VLLM_EXPERIMENT_KV_VARIANT'] == variant
            assert runtime['VLLM_EXPERIMENT_RESIZE_TELEMETRY'] == '1'
            assert runtime['VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS'] == ','.join(map(str, expected_target_blocks))
            assert runtime['VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS'] == str(expected_target_blocks[-1])
            settings[arm] = runtime
            workspace = json.loads((directory / 'mla_workspace_startup_residency.json').read_text())
            audit = workspace['residency']
            assert workspace['expected_bytes'] == 10027008
            assert audit['ok'] and audit['gpu_bytes'] == audit['queried_bytes'] == workspace['expected_bytes']
            assert audit['cpu_bytes'] == audit['unknown_bytes'] == 0
            transactions = result['dynamic_kv']['resize_transactions']
            for transaction in transactions:
                assert not transaction.get('exception') and transaction['result']['ok'] is True
                assert transaction['requested_target_blocks'] == transaction['applied_target_blocks']
                assert transaction['applied_target_blocks'] in expected_target_blocks
            log = (directory / 'run.log').read_text()
            modes = [line.split('runtime_mode=', 1)[1].split()[0] for line in log.splitlines()
                     if 'CUDA_GRAPH_RUNTIME_AUDIT' in line and 'runtime_mode=' in line]
            assert modes and set(modes) == {'NONE'}
            growth_audits = []
            for target, ok, payload in re.findall(
                    r'Post-growth active KV residency audit: target=(\d+) ok=(\w+) residency=(\{[^\n]+\})', log):
                physical = ast.literal_eval(payload)
                assert ok == 'True' and physical['ok']
                assert physical['queried_bytes'] == physical['gpu_bytes'] == (int(target) + 1) * 497664
                assert physical['cpu_bytes'] == physical['unknown_bytes'] == 0
                growth_audits.append({'target_blocks': int(target), 'residency': physical})
            growth_count = sum(t['applied_target_blocks'] > t['old_target_blocks'] for t in transactions)
            assert len(growth_audits) == growth_count
            if not automatic:
                assert transactions == [] and case['used_targets'] == [4313]
            cases.append({**case, 'arm': arm, 'variant': variant,
                          'used_tiers': [TARGETS[t] for t in case['used_targets']],
                          'max_running_requests': max(s['running_requests'] for s in steps),
                          'max_waiting_requests': max(s['waiting_requests'] for s in steps),
                          'scheduler_max_active_blocks': max(s['active_blocks'] for s in steps),
                          'max_active_kv_gib': max(s['active_blocks'] for s in steps) * 497664 / 2**30,
                          'growth_count': growth_count,
                          'shrink_count': sum(t['applied_target_blocks'] < t['old_target_blocks'] for t in transactions),
                          'resize_wall_ms': sum(t['wall_ms'] for t in transactions),
                          'transactions': transactions, 'growth_audits': growth_audits})
            raw[arm, case['sample']] = result['requests']
    changed = {k for k in settings['adaptive_fixed'].keys() | settings['adaptive_dynamic'].keys()
               if settings['adaptive_fixed'].get(k) != settings['adaptive_dynamic'].get(k)}
    assert changed == {'VLLM_UVM_KV_AUTO_SCALE_ENABLE', 'VLLM_EXPERIMENT_KV_VARIANT'}, changed
    checks = []
    for label, left, right in (
            ('fixed_repeat', ('adaptive_fixed', 1), ('adaptive_fixed', 2)),
            ('dynamic_repeat', ('adaptive_dynamic', 1), ('adaptive_dynamic', 2)),
            ('fixed_vs_dynamic_s1', ('adaptive_fixed', 1), ('adaptive_dynamic', 1)),
            ('fixed_vs_dynamic_s2', ('adaptive_fixed', 2), ('adaptive_dynamic', 2))):
        pairs = list(zip(raw[left], raw[right]))
        checks.append({'comparison': label, 'request_pairs': 128,
                       'full_output_differences': sum(a['output_token_ids'] != b['output_token_ids'] for a, b in pairs),
                       'first_token_differences': sum(a['output_token_ids'][0] != b['output_token_ids'][0] for a, b in pairs)})
    return {'status': 'ok_with_output_drift' if any(c['full_output_differences'] for c in checks) else 'ok',
            'errors': [], 'cases': cases, 'output_checks': checks,
            'runtime_settings_differences': sorted(changed), 'protocol': config}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    try:
        value = verify(args.root.resolve())
    except BaseException as exc:
        args.output.write_text(json.dumps({'status': 'error', 'errors': [f'{type(exc).__name__}: {exc}']}, indent=2) + '\n')
        raise
    args.output.write_text(json.dumps(value, indent=2) + '\n')
    args.report.write_text('# P1024 fixed K2 versus automatic K2/K4/K6\n\n' + json.dumps(value, indent=2) + '\n')
    print(json.dumps({'status': value['status'], 'cases': [
        {k: c[k] for k in ('arm', 'sample', 'actual_qps', 'ttft_mean_ms', 'tpot_mean_ms', 'used_tiers', 'growth_count')}
        for c in value['cases']]}, indent=2))
