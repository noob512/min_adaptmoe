#!/usr/bin/env python3
"""Audit an equal-byte UVA/AdaptMoE sweep at a uniform arrival rate."""
import argparse
import collections
import csv
import json
import math
import shlex
import statistics
from pathlib import Path

from analyze_pressure_wave import request_tpot, tier_events

EXPECTED_OFFLOAD_BYTES = 2264924160
KV_BYTES = 1610612736


def jsonlines(path):
    return [json.loads(line) for line in path.read_text().split('\n') if line]


def analyze(root, partial=False):
    manifest = json.loads((root / 'traces/manifest.json').read_text())
    offered_qps = manifest['offered_qps']
    assert math.isfinite(offered_qps) and offered_qps > 0
    expected_cells = len(manifest['traces']) * 2
    results, details = [], {}
    for trace, spec in manifest['traces'].items():
        inputs = jsonlines(root / 'traces' / f'{trace}.jsonl')
        assert all(math.isclose(r['arrival_ms'], i * 1000 / offered_qps)
                   for i, r in enumerate(inputs))
        replays = {}
        for arm in ('native_uva', 'adaptmoe'):
            directory = root / trace / arm
            if partial and not (directory / 'replay.json').exists():
                continue
            replay = json.loads((directory / 'replay.json').read_text())
            replays[arm] = replay
            assert replay['status'] == 'ok' and replay['successful_requests'] == 64, directory
            assert replay['failed_requests'] == 0 and replay['request_count'] == 64, directory
            assert replay['trace_sha256'] == spec['sha256'], directory
            assert replay['arrival_scale'] == 1 and replay['max_connections'] >= 64, directory
            requests = replay['requests']
            assert [r['request_id'] for r in requests] == [r['request_id'] for r in inputs], directory
            assert [r['source_request_id'] for r in requests] == [r['source_request_id'] for r in inputs], directory
            assert [r['trace_arrival_ms'] for r in requests] == [r['arrival_ms'] for r in inputs], directory
            assert all(r['success'] and r['prompt_tokens'] == spec['prompt_tokens'] and
                       r['completion_tokens'] == 512 and r['usage_completion_tokens'] == 512 and
                       len(r['output_token_ids']) == 512 for r in requests), directory
            command = shlex.split((directory / 'server_command.txt').read_text())
            assert '--enforce-eager' in command, directory
            assert json.loads(command[command.index('--compilation-config') + 1])['cudagraph_mode'] == 'NONE', directory
            assert int(command[command.index('--kv-cache-memory-bytes') + 1]) == KV_BYTES, directory
            assert 'VLLM_ATTENTION_BACKEND=TRITON_ATTN' in command, directory
            steps = [s for s in jsonlines(directory / 'scheduler_steps.jsonl')
                     if s['epoch'] >= replay['trace_start_epoch'] and s['tokens'] > 0]
            phase_counts = collections.Counter(s['phase'] for s in steps)
            assert phase_counts['mixed'] > 0, directory
            offload = jsonlines(directory / 'offload_manifest.jsonl')
            params = [e for e in offload if e.get('event') == 'cpu_offload_parameter']
            selected_bytes = sum(e['bytes'] for e in params)
            assert selected_bytes == EXPECTED_OFFLOAD_BYTES, (directory, selected_bytes)
            detail = {'scheduler_phase_counts': dict(phase_counts),
                      'selected_bytes': selected_bytes, 'parameter_count': len(params),
                      'max_send_lag_ms': max((r['sent_at_perf'] - r['scheduled_at_perf']) * 1000 for r in requests),
                      'max_running': max(s['running'] for s in steps),
                      'max_waiting': max(s['waiting'] for s in steps),
                      'max_kv_usage': max(s['kv_usage'] for s in steps),
                      'preemptions': sum(s['preemptions'] for s in steps)}
            if arm == 'adaptmoe':
                for value in ('VLLM_UVM_ORIGINAL_VA_PHASE_MODE=token_threshold',
                              'VLLM_UVM_PHASE_BATCHED_SCHEDULING=0',
                              'VLLM_UVM_ORIGINAL_VA_TOKEN_THRESHOLD=16',
                              'VLLM_UVM_ORIGINAL_VA_RESUME_TOKEN_THRESHOLD=128',
                              'VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS=1023',
                              'VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS=1023'):
                    assert value in command, (directory, value)
                commits = tier_events(directory, replay)
                assert not commits, commits
                assert not any(e.get('event') == 'cpu_offload_uvm_stable_plan_class_switched' for e in offload), directory
                detail['tier_commits'] = commits
                formal = [e for e in offload if replay['trace_start_epoch'] <= e.get('timestamp', 0) <= replay['finished_at_epoch']]
                for key, event in (('park', 'cpu_offload_uvm_original_va_remote_transition'),
                                   ('resume', 'cpu_offload_uvm_original_va_alias_prefill_activated')):
                    transitions = [e for e in formal if e.get('event') == event]
                    assert transitions and all(e.get('ok') is True for e in transitions), (directory, key)
                    if key == 'park':
                        assert all(e.get('reason') == 'token_threshold_remote' for e in transitions), directory
                    detail[f'{key}_count'] = len(transitions)
                    detail[f'{key}_total_ms'] = sum(e['wall_ns'] for e in transitions) / 1e6
                detail['scheduled_token_buckets'] = dict(collections.Counter(
                    '<=16' if s['tokens'] <= 16 else '>=128' if s['tokens'] >= 128 else '17..127'
                    for s in steps))
            else:
                assert all(e.get('uva_offloading') is True and e.get('copy_on_forward') is False for e in params), directory
            details[f'{trace}/{arm}'] = detail
            results.append(dict(trace=trace, arm=arm, requests=64, offered_qps=offered_qps,
                                qps=replay['qps'], ttft_ms=replay['ttft_ms']['mean'],
                                tpot_ms=statistics.fmean(request_tpot(r) for r in requests),
                                e2e_ms=replay['e2e_ms']['mean'],
                                duration_s=replay['measurement_window_seconds'],
                                selected_bytes=selected_bytes, kv_bytes=KV_BYTES,
                                mixed_steps=phase_counts['mixed'], preemptions=detail['preemptions']))
        if len(replays) == 2:
            a = replays['adaptmoe']['requests']
            b = replays['native_uva']['requests']
            details[f'{trace}/output_comparison'] = dict(
                first_token_agreement=sum(x['output_token_ids'][0] == y['output_token_ids'][0] for x, y in zip(a, b)),
                full_output_agreement=sum(x['output_token_ids'] == y['output_token_ids'] for x, y in zip(a, b)))
    if not partial:
        assert len(results) == expected_cells, len(results)
    result = {'status': 'partial' if partial and len(results) < expected_cells else 'ok',
              'cells': len(results), 'successful_requests': len(results) * 64,
              'protocol': manifest, 'overall': results, 'details': details}
    (root / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    if results:
        with (root / 'overall.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    analyze(args.root, args.partial)
