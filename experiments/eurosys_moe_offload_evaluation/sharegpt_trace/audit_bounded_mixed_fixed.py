#!/usr/bin/env python3
"""Validate the complete fixed-KV, four-mechanism mixed-scheduler sweep."""
import argparse
import hashlib
import json
import shlex
from collections import Counter
from pathlib import Path

from analyze_pressure_wave import tier_events


def jsonlines(path):
    return [json.loads(s) for s in path.read_text().split('\n') if s]


def audit(root):
    manifest = json.loads((root / 'traces/manifest.json').read_text())
    cells = {}
    for trace, spec in manifest['traces'].items():
        source = root / 'traces' / f'{trace}.jsonl'
        assert hashlib.sha256(source.read_bytes()).hexdigest() == spec['sha256']
        rows = jsonlines(source)
        replays = {}
        for arm in ('fixed_k4', 'native_uva', 'native_copy', 'ktransformers'):
            directory = root / trace / arm
            replay = json.loads((directory / 'replay.json').read_text())
            replays[arm] = replay
            assert replay['status'] == 'ok' and replay['successful_requests'] == 64, directory
            assert replay['request_count'] == 64 and replay['failed_requests'] == 0, directory
            assert replay['trace_sha256'] == spec['sha256'], directory
            assert replay['arrival_scale'] == 1 and replay['max_connections'] >= 64, directory
            requests = replay['requests']
            assert [r['request_id'] for r in requests] == [r['request_id'] for r in rows], directory
            assert [r['source_request_id'] for r in requests] == [r['source_request_id'] for r in rows], directory
            assert [r['trace_arrival_ms'] for r in requests] == [r['arrival_ms'] for r in rows], directory
            assert all(r['success'] and r['prompt_tokens'] == spec['prompt_tokens'] and
                       r['completion_tokens'] == 512 and r['usage_completion_tokens'] == 512 and
                       len(r['output_token_ids']) == 512 for r in requests), directory
            command = shlex.split((directory / 'server_command.txt').read_text())
            cell = {'successful_requests': 64, 'graph_disabled': True,
                    'max_send_lag_ms': max((r['sent_at_perf'] - r['scheduled_at_perf']) * 1000 for r in requests)}
            if arm == 'ktransformers':
                for flag in ('--disable-cuda-graph', '--enable-mixed-chunk', '--disable-radix-cache'):
                    assert flag in command, (directory, flag)
                for flag, value in (('--kt-num-gpu-experts', '100'), ('--max-total-tokens', '43696'),
                                    ('--kt-gpu-prefill-token-threshold', '0')):
                    assert command[command.index(flag) + 1] == value, (directory, flag)
            else:
                assert '--enforce-eager' in command, directory
                compilation = json.loads(command[command.index('--compilation-config') + 1])
                assert compilation['cudagraph_mode'] == 'NONE', directory
                steps = [s for s in jsonlines(directory / 'scheduler_steps.jsonl')
                         if s['epoch'] >= replay['trace_start_epoch']]
                counts = Counter(s['phase'] for s in steps)
                assert counts['mixed'] > 0, directory
                cell['scheduler_phase_counts'] = dict(counts)
                offload = jsonlines(directory / 'offload_manifest.jsonl')
                if arm == 'fixed_k4':
                    assert 'VLLM_UVM_ORIGINAL_VA_PHASE_MODE=token_threshold' in command, directory
                    assert 'VLLM_UVM_PHASE_BATCHED_SCHEDULING=0' in command, directory
                    assert 'VLLM_UVM_ORIGINAL_VA_TOKEN_THRESHOLD=16' in command, directory
                    assert 'VLLM_UVM_ORIGINAL_VA_RESUME_TOKEN_THRESHOLD=128' in command, directory
                    events = tier_events(directory, replay)
                    assert len(events) == 1 and events[0]['time_s'] < 0 and events[0]['blocks'] == 2730, events
                    formal = [e for e in offload if e.get('timestamp', 0) >= replay['trace_start_epoch']]
                    assert not any(e.get('event') == 'cpu_offload_uvm_stable_plan_class_switched' for e in formal), directory
                    cell['initialization_tier_events'] = events
                    cell['formal_tier_changes'] = 0
                    cell['scheduled_token_buckets'] = dict(Counter(
                        '<=16' if s['tokens'] <= 16 else '>=128' if s['tokens'] >= 128 else '17..127'
                        for s in steps if s['tokens'] > 0))
                    for key, event in (('park', 'cpu_offload_uvm_original_va_remote_transition'),
                                       ('resume', 'cpu_offload_uvm_original_va_alias_prefill_activated')):
                        transitions = [e for e in formal if e.get('event') == event]
                        assert transitions and all(e.get('ok') is True for e in transitions), (directory, key)
                        if key == 'park':
                            assert all(e.get('reason') == 'token_threshold_remote' for e in transitions), directory
                        cell[f'formal_{key}_count'] = len(transitions)
                        cell[f'formal_{key}_wall_ms'] = sum(e['wall_ns'] for e in transitions) / 1e6
                else:
                    params = [e for e in offload if e.get('event') == 'cpu_offload_parameter']
                    cell['native_parameter_count'] = len(params)
                    cell['native_parameter_bytes'] = sum(e['bytes'] for e in params)
            cells[f'{trace}/{arm}'] = cell
        reference = replays['fixed_k4']['requests']
        for arm, replay in replays.items():
            if arm == 'fixed_k4':
                continue
            cells[f'{trace}/{arm}']['first_token_agreement_with_adaptmoe'] = sum(
                a['output_token_ids'][0] == b['output_token_ids'][0]
                for a, b in zip(reference, replay['requests']))
            cells[f'{trace}/{arm}']['full_output_agreement_with_adaptmoe'] = sum(
                a['output_token_ids'] == b['output_token_ids']
                for a, b in zip(reference, replay['requests']))
    assert len(cells) == 12, len(cells)
    result = {'status': 'ok', 'cells': 12, 'successful_requests': 768, 'details': cells}
    (root / 'audit_fixed.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
