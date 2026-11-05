#!/usr/bin/env python3
"""Freeze bounded ShareGPT prefixes; only arrival rate varies within a trace."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--qps', type=float, help='Uniform arrival rate; default preserves the pressure wave')
    parser.add_argument('--traces', nargs='+', choices=('short', 'medium', 'long'),
                        default=['short', 'medium', 'long'])
    args = parser.parse_args()
    if args.qps is not None and (not math.isfinite(args.qps) or args.qps <= 0):
        parser.error('--qps must be finite and positive')
    # The frozen stratified source contains exactly 64 unique prompts >=1024.
    source = [json.loads(s) for s in args.source.read_text().split('\n') if s]
    selected = [r for r in source if r['prompt_tokens'] >= 1024][:64]
    assert len(selected) == 64
    assert len({r['request_id'] for r in selected}) == 64
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {'source': str(args.source.resolve()),
                'source_sha256': digest(args.source.read_bytes()),
                'output_tokens': 512, 'max_context_tokens': 1536,
                'max_num_seqs': 64, 'repeats': 1,
                'phases': [{'name': 'low', 'count': 8, 'rate': 0.25,
                            'start_s': 0, 'end_s': 32},
                           {'name': 'high', 'count': 48, 'rate': 4,
                            'start_s': 32, 'end_s': 44},
                           {'name': 'recovery', 'count': 8, 'rate': 0.25,
                            'start_s': 44, 'end_s': 76}],
                'transformation': 'Deterministic prompt token prefix; fixed '
                                  '512-token generation with ignore_eos. '
                                  'Not original ShareGPT answer lengths.',
                'traces': {}}
    if args.qps is not None:
        manifest['offered_qps'] = args.qps
        manifest['phases'] = [{'name': 'constant', 'count': 64, 'rate': args.qps,
                               'start_s': 0, 'end_s': 64 / args.qps}]
    for name, length in [('short', 128), ('medium', 512), ('long', 1024)]:
        if name not in args.traces:
            continue
        rows = []
        for i, original in enumerate(selected):
            if args.qps is not None:
                phase, arrival = 'constant', i * 1000 / args.qps
            elif i < 8:
                phase, arrival = 'low', i * 4000
            elif i < 56:
                phase, arrival = 'high', 32000 + (i - 8) * 250
            else:
                phase, arrival = 'recovery', 44000 + (i - 56) * 4000
            tokens = original['prompt_token_ids'][:length]
            rows.append({'request_id': f'{name}-{i:03d}',
                         'source_request_id': original['request_id'],
                         'prompt_token_ids': tokens,
                         'prompt_token_ids_sha256': digest(','.join(map(str, tokens)).encode('ascii')),
                         'prompt_tokens': length, 'max_tokens': 512,
                         'arrival_ms': arrival, 'pressure_phase': phase})
        path = args.output_dir / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        manifest['traces'][name] = {'path': str(path.resolve()),
                                    'sha256': digest(path.read_bytes()),
                                    'request_count': len(rows), 'prompt_tokens': length}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
