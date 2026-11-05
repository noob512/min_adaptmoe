#!/usr/bin/env python3
"""Verify the identical QPS30 replay with automatic GPU KV capped at K4."""
import argparse
import json
from pathlib import Path

from verify_p1024_fixed_vs_dynamic_kv import verify as verify_base


def verify(root):
    value = verify_base(root, expected_target_blocks=(4313,8629))
    assert value['protocol']['arrival_qps'] == 30
    for case in value['cases']:
        assert set(case['used_targets']) <= {4313,8629}
        assert all(t['requested_target_blocks'] <= 8629 for t in case['transactions'])
        assert case['growth_count'] <= 1
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    try:
        value = verify(args.root.resolve())
    except BaseException as exc:
        args.output.write_text(json.dumps({'status':'error','errors':[f'{type(exc).__name__}: {exc}']},indent=2)+'\n')
        raise
    args.output.write_text(json.dumps(value,indent=2)+'\n')
    args.report.write_text('# QPS30 fixed K2 versus dynamic K2/K4\n\n'+json.dumps(value,indent=2)+'\n')
    print(json.dumps({'status':value['status'],'cases':[{k:c[k] for k in (
        'arm','sample','actual_qps','ttft_mean_ms','tpot_mean_ms','used_tiers','growth_count')}
        for c in value['cases']]},indent=2))
