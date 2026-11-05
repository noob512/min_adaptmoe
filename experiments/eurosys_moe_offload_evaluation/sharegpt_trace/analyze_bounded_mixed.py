#!/usr/bin/env python3
"""Audit a single mixed scheduler sweep, including fixed-capacity controls."""
import argparse
import collections
import csv
import json
import re
import statistics
from pathlib import Path

from analyze_pressure_wave import interval_metrics, request_tpot, tier_events

ARMS = ('dynamic', 'fixed_k2', 'fixed_k4', 'native_uva', 'native_copy', 'ktransformers')


def jsonlines(path):
    return [json.loads(s) for s in path.read_text().split('\n') if s] if path.exists() else []


def analyze(root):
    manifest = json.loads((root / 'traces/manifest.json').read_text())
    rows, phases, details = [], [], {}
    for trace in manifest['traces']:
        for arm in ARMS:
            directory = root / trace / arm
            if not (directory / 'replay.json').exists():
                continue
            replay = json.loads((directory / 'replay.json').read_text())
            assert replay['status'] == 'ok', directory
            assert replay['successful_requests'] == 64, directory
            assert replay['trace_sha256'] == manifest['traces'][trace]['sha256'], directory
            assert replay['arrival_scale'] == 1, directory
            assert replay['max_connections'] >= 64, directory
            steps = jsonlines(directory / 'scheduler_steps.jsonl')
            formal_steps = [s for s in steps if s['epoch'] >= replay['trace_start_epoch']]
            row = dict(trace=trace, arm=arm, requests=replay['successful_requests'],
                       qps=replay['qps'], ttft_ms=replay['ttft_ms']['mean'],
                       tpot_ms=statistics.fmean(request_tpot(r) for r in replay['requests']),
                       e2e_ms=replay['e2e_ms']['mean'],
                       duration_s=replay['measurement_window_seconds'],
                       mixed_steps=sum(s['phase'] == 'mixed' for s in formal_steps) if steps else None,
                       preemptions=sum(s['preemptions'] for s in formal_steps) if steps else None,
                       max_running=max((s['running'] for s in formal_steps), default=None),
                       max_waiting=max((s['waiting'] for s in formal_steps), default=None),
                       max_kv_usage=max((s['kv_usage'] for s in formal_steps), default=None))
            rows.append(row)
            for phase in manifest['phases']:
                phases.append(dict(trace=trace, arm=arm, phase=phase['name'],
                                   **interval_metrics(replay, phase['start_s'], phase['end_s'])))
            events = tier_events(directory, replay) if arm in ARMS[:3] else []
            token_issues = [r['request_id'] for r in replay['requests']
                            if len(r.get('output_token_ids', [])) != 512]
            offload = jsonlines(directory / 'offload_manifest.jsonl')
            parameters = [e for e in offload if e.get('event') == 'cpu_offload_parameter']
            formal_offload = [e for e in offload if e.get('timestamp', 0) >= replay['trace_start_epoch']]
            kt_mixed = []
            if arm == 'ktransformers':
                for line in (directory / 'server.log').read_text().split('\n'):
                    match = re.search(r'Prefill batch,.*#running-req: (\d+)', line)
                    if match and int(match[1]) > 0:
                        kt_mixed.append(line)
            details[f'{trace}/{arm}'] = dict(tier_events=events,
                 scheduler_phase_counts=dict(collections.Counter(s['phase'] for s in formal_steps)),
                 output_id_count_mismatches=token_issues,
                 native_parameter_bytes=sum(e.get('bytes', 0) for e in parameters) if arm.startswith('native') else None,
                 native_parameter_count=len(parameters) if arm.startswith('native') else None,
                 kt_logged_prefill_with_running=len(kt_mixed) if arm == 'ktransformers' else None,
                 formal_access_transitions=dict(collections.Counter(e['event'] for e in formal_offload
                      if e.get('event') in ('cpu_offload_uvm_original_va_remote_transition',
                                           'cpu_offload_uvm_original_va_alias_prefill_activated'))),
                 offload_event_counts=dict(collections.Counter(r.get('event', r.get('kind', 'unknown')) for r in offload)))
    if rows:
        for name, data in [('overall', rows), ('phases', phases)]:
            with (root / f'{name}.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    report = {'protocol': manifest, 'overall': rows, 'details': details, 'phases': phases}
    (root / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    analyze(parser.parse_args().root)
