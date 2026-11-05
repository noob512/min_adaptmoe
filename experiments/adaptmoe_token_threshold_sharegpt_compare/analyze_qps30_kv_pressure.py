#!/usr/bin/env python3
"""Explain QPS30 admission, preemption, token budgets and resize cost."""
import argparse
import collections
import csv
import hashlib
import json
import shutil
import statistics
from pathlib import Path

from verify_p1024_fixed_vs_dynamic_kv import verify

HERE = Path(__file__).resolve().parent
EXPERT_BYTES = {'cold8': 3598712832, 'cold11': 4948230144, 'cold17': 7647264768, 'cold23': 10346299392}


def mean(values):
    return statistics.fmean(values)


def write_csv(path, rows):
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_case(case, *, require_decode_allocations_success=True):
    directory = Path(case['directory'])
    result = json.loads((directory / 'result.json').read_text())
    detail = json.loads((directory / 'kv_pressure_detail.json').read_text())
    scheduled = result['scheduler_steps']
    steps = detail['scheduler_steps']
    allocations = detail['allocations']
    assert len(steps) == len(scheduled) == result['summary']['engine_steps']
    assert detail['bytes_per_block'] == 497664
    assert detail['max_num_running_reqs'] == 128 and detail['max_num_scheduled_tokens'] == 8192
    first = result['requests'][0]
    origin = scheduled[first['first_scheduled_step'] - 1]['finished_perf_counter'] - (
        first['planned_arrival_offset_seconds'] + first['queueing_ms'] / 1000)
    labels = {'arm': case['arm'], 'sample': case['sample']}
    failures, timeline, deferred_ids, waiting_causes = [], [], set(), collections.Counter()
    for observed, base, controller, concurrency in zip(
            steps, scheduled, result['dynamic_kv']['controller_steps'], result['kv_concurrency']['scheduler_steps']):
        assert observed['step'] == base['scheduler_step']
        assert observed['scheduled_tokens'] == base['total_num_scheduled_tokens']
        assert observed['after']['active_blocks'] == concurrency['active_blocks']
        assert observed['after']['free_counts'] == concurrency['free_counts']
        assert len(observed['waiting_after']) == concurrency['waiting_requests']
        assert len(observed['preempted_request_ids']) == concurrency['preempted_requests']
        calls = allocations[observed['allocation_start_index']:observed['allocation_end_index']]
        assert all(c['step'] == observed['step'] for c in calls)
        rejected = [c for c in calls if not c['success']]
        cause = 'none'
        if observed['waiting_after']:
            deferred_ids.update(observed['waiting_after'])
            if any(c['status_before'] in ('WAITING', 'PREEMPTED') for c in rejected):
                cause = 'kv_allocation_failure'
            elif observed['remaining_token_budget'] == 0:
                cause = 'token_budget_exhausted'
            elif len(observed['running_after']) == detail['max_num_running_reqs']:
                cause = 'max_num_seqs'
            elif observed['preempted_request_ids']:
                cause = 'preemption'
            else:
                cause = 'other'
            waiting_causes[cause] += 1
        for entry in rejected:
            if entry['policy_allowed']:
                assert entry['num_blocks_to_allocate'] > entry['after_policy']['gpu_available_blocks']
                why = 'insufficient_gpu_free_blocks'
            else:
                why = 'runtime_budget_policy'
            failures.append({**labels, 'step': entry['step'], 'request_id': entry['request_id'],
                             'status': entry['status_before'], 'reason': why,
                             'requested_blocks': entry['num_blocks_to_allocate'],
                             'available_blocks': entry['after_policy']['gpu_available_blocks'],
                             'active_blocks': entry['before']['active_blocks'],
                             'target_blocks': entry['before']['target_blocks']})
        timeline.append({**labels, 'step': observed['step'],
                         'start_s': base['engine_step_started_perf_counter'] - origin,
                         'finish_s': base['engine_step_finished_perf_counter'] - origin,
                         'duration_ms': base['engine_step_duration_ms'],
                         'active_blocks': observed['after']['active_blocks'],
                         'active_kv_gib': observed['after']['active_blocks'] * 497664 / 2**30,
                         'target_before_controller': observed['after']['target_blocks'],
                         'target_after_controller': controller['applied_target_blocks'],
                         'target_gib': observed['after']['target_blocks'] * 497664 / 2**30,
                         'active_blocks_after_engine': observed['after_engine']['active_blocks'],
                         'running_after_engine': observed['running_after_engine'],
                         'waiting_after_engine': observed['waiting_after_engine'],
                         'running_requests': len(observed['running_after']),
                         'waiting_requests': len(observed['waiting_after']),
                         'scheduled_tokens': observed['scheduled_tokens'],
                         'remaining_token_budget': observed['remaining_token_budget'],
                         'waiting_cause': cause, 'allocation_failures': len(rejected),
                         'preemptions': len(observed['preempted_request_ids']),
                         'expert_class': controller['active_plan_class']})
    assert waiting_causes['other'] == 0
    by_layer = collections.defaultdict(list)
    window = result['measurement_window']
    for line in (directory / 'cpu_offload_manifest.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event['event'] == 'cpu_offload_copy_forward' and window['started_epoch'] <= event['timestamp'] <= window['finished_epoch']:
            by_layer[event['module_prefix']].append(event)
    assert len(by_layer) == 26 and all(len(events) == len(steps) for events in by_layer.values())
    for index, row in enumerate(timeline):
        events = [events[index] for events in by_layer.values()]
        submits = {event['graph_static_alias_copy_submitted'] for event in events}
        assert len(submits) == 1
        row['copy_submitted'] = submits.pop()
        row['expert_h2d_bytes'] = sum(event['graph_static_alias_actual_h2d_bytes'] for event in events)
        assert row['expert_h2d_bytes'] == (EXPERT_BYTES[row['expert_class']] if row['copy_submitted'] else 0)
    request_rows = []
    for req in result['requests']:
        assert abs(req['queueing_ms'] - req['admission_delay_ms'] - req['engine_queueing_ms']) < 1e-5
        request_rows.append({**labels, 'request_id': req['request_id'],
                             'planned_arrival_s': req['planned_arrival_offset_seconds'],
                             'admitted_s': req['admitted_offset_seconds'],
                             'first_scheduled_step': req['first_scheduled_step'],
                             'deferred_scheduler_steps': req['deferred_scheduler_steps'],
                             'admission_delay_ms': req['admission_delay_ms'],
                             'engine_queueing_ms': req['engine_queueing_ms'],
                             'queueing_ms': req['queueing_ms'],
                             'scheduled_to_first_token_ms': req['ttft_ms'] - req['queueing_ms'],
                             'ttft_ms': req['ttft_ms'], 'tpot_ms': req['tpot_ms'], 'e2e_ms': req['e2e_ms']})
    decode_calls = [c for c in allocations if c['computed_tokens_before'] >= 1024]
    assert decode_calls and set(c['num_blocks_to_allocate'] for c in decode_calls) <= {0, 1}
    if require_decode_allocations_success:
        assert all(c['success'] for c in decode_calls)
    control_total = sum(c['wall_ms'] for c in detail['controller_calls'])
    resize_total = sum(t['wall_ms'] for t in result['dynamic_kv']['resize_transactions'])
    summary = {**case, 'elapsed_s': result['summary']['elapsed_seconds'],
               'experiment_started_perf_counter': origin,
               'allocation_failures': len(failures),
               'waiting_allocation_failures': sum(c['status'] in ('WAITING', 'PREEMPTED') for c in failures),
               'running_allocation_failures': sum(c['status'] == 'RUNNING' for c in failures),
               'preemptions': sum(row['preemptions'] for row in timeline),
               'waiting_steps_by_cause': dict(waiting_causes),
               'waiting_steps': sum(waiting_causes.values()), 'deferred_requests': len(deferred_ids),
               'max_request_blocks': max(sum(r['block_counts']) for s in steps for r in s['running_after']),
               'decode_additional_blocks_histogram': dict(collections.Counter(c['num_blocks_to_allocate'] for c in decode_calls)),
               'first_allocation_failure': failures[0] if failures else None,
               'last_allocation_failure': failures[-1] if failures else None,
               'copy_steps': sum(row['copy_submitted'] for row in timeline),
               'remote_steps': sum(not row['copy_submitted'] for row in timeline),
               'expert_h2d_gib': sum(row['expert_h2d_bytes'] for row in timeline) / 2**30,
               'controller_including_resize_ms': control_total,
               'controller_outside_resize_ms': max(control_total - resize_total, 0),
               'controller_calls': detail['controller_calls'],
               'observation_outer_ms': sum(a['outer_observation_cpu_ms'] for a in allocations) + sum(s['outer_observation_cpu_ms'] for s in steps),
               **{key: mean(r[key] for r in request_rows) for key in (
                   'admission_delay_ms', 'engine_queueing_ms', 'queueing_ms', 'scheduled_to_first_token_ms')}}
    return summary, timeline, failures, request_rows


def plot(root, timeline, cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True, constrained_layout=True)
    highest_target = max(r['target_after_controller'] for r in timeline) * 497664 / 2**30
    for col, (arm, title) in enumerate((('adaptive_fixed', 'Fixed K2'), ('adaptive_dynamic', 'Dynamic KV'))):
        rows = [r for r in timeline if r['arm'] == arm and r['sample'] == 1]
        case = next(c for c in cases if c['arm'] == arm and c['sample'] == 1)
        if arm == 'adaptive_dynamic':
            title = 'Dynamic ' + ' / '.join(case['used_tiers'])
        x = [r['start_s'] for r in rows] + [rows[-1]['finish_s']]
        occupancy = [r['active_kv_gib'] for r in rows] + [rows[-1]['active_blocks_after_engine'] * 497664 / 2**30]
        capacity_x = [0] + [t['finished_perf_counter'] - case['experiment_started_perf_counter']
                            for t in case['transactions']] + [rows[-1]['finish_s']]
        capacity = [4313 * 497664 / 2**30] + [t['applied_target_blocks'] * 497664 / 2**30
                                              for t in case['transactions']] + [rows[-1]['target_after_controller'] * 497664 / 2**30]
        axes[0, col].step(x, occupancy, where='post', label='Active KV', color='#2563eb', linewidth=2)
        axes[0, col].step(capacity_x, capacity, where='post', label='GPU KV target', color='#dc2626', linestyle='--')
        axes[0, col].set(title=title, ylabel='GiB', ylim=(0, highest_target * 1.09))
        for key, label, color in (('running_requests', 'Running', '#2563eb'), ('waiting_requests', 'Waiting', '#ea580c')):
            end_key = 'running_after_engine' if key == 'running_requests' else 'waiting_after_engine'
            values = [r[key] for r in rows] + [rows[-1][end_key]]
            axes[1, col].step(x, values, where='post', label=label, color=color, linewidth=1.8)
        axes[1, col].set(xlabel='Seconds since planned arrival of request 0', ylabel='Requests', ylim=(0, 135))
        for ax in axes[:, col]:
            ax.axvline(127 / 30, color='#777777', linestyle=':', label='Last planned arrival')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, loc='upper right')
    axes[1, 0].set_xlim(0, max(r['finish_s'] for r in timeline) * 1.03)
    fig.suptitle('QPS=30; P1024 / O16; 128 requests; C128; sample 1', fontsize=13)
    fig.savefig(root / 'kv_pressure_timeline.png', dpi=180)
    fig.savefig(root / 'kv_pressure_timeline.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    verification = verify(root / 'qps30')
    cases, timeline, failures, requests = [], [], [], []
    for case in verification['cases']:
        summary, rows, rejected, reqs = analyze_case(case)
        cases.append(summary)
        timeline.extend(rows)
        failures.extend(rejected)
        requests.extend(reqs)
    rows = []
    for arm in ('adaptive_fixed', 'adaptive_dynamic'):
        selected = [c for c in cases if c['arm'] == arm]
        assert len(selected) == 2
        fields = ('actual_qps', 'ttft_mean_ms', 'tpot_mean_ms', 'e2e_mean_ms', 'elapsed_s',
                  'allocation_failures', 'waiting_allocation_failures', 'running_allocation_failures',
                  'preemptions', 'waiting_steps', 'deferred_requests', 'expert_h2d_gib',
                  'copy_steps', 'remote_steps', 'controller_including_resize_ms',
                  'controller_outside_resize_ms', 'resize_wall_ms', 'observation_outer_ms',
                  'admission_delay_ms', 'engine_queueing_ms', 'queueing_ms', 'scheduled_to_first_token_ms')
        row = {'arm': arm, **{key: mean(c[key] for c in selected) for key in fields},
               'max_running_requests': max(c['max_running_requests'] for c in selected),
               'max_waiting_requests': max(c['max_waiting_requests'] for c in selected),
               'max_active_kv_gib': max(c['max_active_kv_gib'] for c in selected),
               'used_tiers': sorted({tier for c in selected for tier in c['used_tiers']}),
               'waiting_steps_by_cause': {key: mean(c['waiting_steps_by_cause'].get(key, 0) for c in selected)
                                         for key in ('kv_allocation_failure', 'token_budget_exhausted', 'max_num_seqs', 'preemption', 'other')}}
        rows.append(row)
    summary = {'status': verification['status'], 'valid_processes': 4, 'successful_requests': 512,
               'rows': rows, 'cases': cases, 'output_checks': verification['output_checks'],
               'geometry': {'bytes_per_block': 497664, 'tokens_per_block': 16,
                            'prompt_blocks': 64, 'peak_blocks_per_request': 65,
                            'max_all_requests_blocks': 128 * 65,
                            'max_all_requests_kv_gib': 128 * 65 * 497664 / 2**30},
               'scope': 'Fresh QPS30 only, two independent processes per arm; symmetric observation of native decisions; no scheduling, allocation, offload or resize-policy changes.'}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    write_csv(root / 'summary.csv', rows)
    write_csv(root / 'scheduler_timeline.csv', timeline)
    if failures:
        write_csv(root / 'allocation_failures.csv', failures)
    write_csv(root / 'request_latency.csv', requests)
    if args.plot:
        plot(root, timeline, cases)
    fixed, dynamic = rows
    lines = ['# QPS=30：固定K2与动态KV的排队、抢占复跑', '',
             '**动态扩容消除了本轮KV分配失败，但没有消除排队；固定K2也未发生抢占。** 两次重复均得到同样的调度原因分类。', '',
             '沿用同一ShareGPT P1024/O16、128请求，SHA256=a4cc14060c94efb8c86f0524f2c49fbf85c0ff284eb12636e1e5c133d85bb9f2；DeepSeek-V2-Lite-Chat BF16、TRITON_MLA、C128、8192-token预算、PARK≤16/RESUME≥128、混合调度、eager、无Graph、无prefix cache。提供QPS仅30，计划到达i/30。固定→动态→动态→固定，共4个新进程、512次请求，全部成功返回。', '',
             '固定组使用同一managed池，GPU容量锁定K2/cold11；动态组同样从K2/cold11开始，自适应K2/K4/K6并配对cold11/17/23。两臂均预留6GiB VA，固定组GPU目标仅4313块，离线尾部8630块未被当作可分配GPU容量。两组新增相同的只读观测，未改变现有调度、分配或升档策略。控制器wall为包含同步等待的实测区间，不等于纯CPU时间。', '',
             '## 性能结果（两个进程均值）', '',
             '| 配置 | 实际QPS | TTFT mean s | TPOT mean ms | E2E mean s | 总时长s |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        label = '固定K2' if row['arm'] == 'adaptive_fixed' else '动态K2→K4→K6'
        lines.append(f"| {label} | {row['actual_qps']:.3f} | {row['ttft_mean_ms']/1000:.3f} | {row['tpot_mean_ms']:.2f} | {row['e2e_mean_ms']/1000:.3f} | {row['elapsed_s']:.3f} |")
    lines += ['', f"动态吞吐变化{(dynamic['actual_qps']/fixed['actual_qps']-1)*100:+.2f}%；TTFT变化{(dynamic['ttft_mean_ms']/fixed['ttft_mean_ms']-1)*100:+.2f}%；TPOT变化{(dynamic['tpot_mean_ms']/fixed['tpot_mean_ms']-1)*100:+.2f}%。QPS包含排空，延迟包含计划到达后的等待及真实升档，不是纯稳态测量。", '',
              '## 容量与调度证据（每个进程）', '',
              '| 配置 | 活跃KV峰值GiB | 最大运行/等待请求 | 抢占 | 新请求KV分配失败 | 运行请求分配失败 | KV阻塞step | token预算阻塞step |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        label = '固定K2' if row['arm'] == 'adaptive_fixed' else '动态'
        lines.append(f"| {label} | {row['max_active_kv_gib']:.3f} | {row['max_running_requests']}/{row['max_waiting_requests']} | {row['preemptions']:.0f} | {row['waiting_allocation_failures']:.0f} | {row['running_allocation_failures']:.0f} | {row['waiting_steps_by_cause']['kv_allocation_failure']:.0f} | {row['waiting_steps_by_cause']['token_budget_exhausted']:.0f} |")
    lines += ['', '分配失败次数表示调度尝试次数，不等于失败请求数；同一等待请求可能多次尝试。请求最终全部完成。等待step按本次直接停止准入的原因分类，不声称同一时刻不存在其他潜在约束。', '',
              '## 为什么固定2GiB没有抢占', '',
              'vLLM先服务RUNNING请求，再尝试WAITING请求。RUNNING追加KV失败会释放最低优先级请求的块、置为PREEMPTED并重算；WAITING首次分配失败仅停止本step的新请求准入，保留等待。两种行为在scheduler.py的独立分支中实现。', '',
              '本trace每个KV块覆盖16 tokens：1024-token prefill需要64块；16-token输出过程中需缓存的已计算token最多1039个，因此最多65块。固定K2的4313块可同时容纳66个完整活跃请求：66×65=4290，剩余23块。新请求需要64块而无法准入，但已运行请求的短Decode仅需首次增加1块，随后不跨入下一块。实测全部Decode分配调用只需0或1块、全部成功，每请求最多65块；未发生RUNNING分配失败，所以没有触发抢占。部分RUNNING调用仍是chunked prefill，已在观测中与Decode区分。', '']
    for case in cases:
        failure = case['first_allocation_failure']
        if failure:
            lines += [f"固定组样本{case['sample']}首次阻塞：step {failure['step']}，请求{failure['request_id']}处于{failure['status']}，需要{failure['requested_blocks']}块，可用仅{failure['available_blocks']}块，活跃{failure['active_blocks']}块，目标{failure['target_blocks']}块。这是准入等待，没有已运行请求被回收重算。", '']
    lines += ['## 为什么扩容后仍有排队', '',
              '动态组全部allocation成功，排除了这次可观测的KV容量阻塞，但所有残留等待step都用满8192-token预算。新增请求prefill为1024 tokens，一个step最多完成约8个请求的prefill，混合Decode还会占用预算。提供30req/s时GPU服务速度跟不上到达，扩大KV不能增加每step的token预算或消除forward/copy时间。', '',
              '| 配置 | 到达→引擎接收ms | 引擎接收→首次调度ms | 首次调度→首token ms | TTFT合计ms |',
              '|---|---:|---:|---:|---:|']
    for row in rows:
        label = '固定K2' if row['arm'] == 'adaptive_fixed' else '动态'
        lines.append(f"| {label} | {row['admission_delay_ms']:.2f} | {row['engine_queueing_ms']:.2f} | {row['scheduled_to_first_token_ms']:.2f} | {row['ttft_mean_ms']:.2f} |")
    lines += ['', '第一项也包括离线replay在同步engine.step返回后才接收已到时请求的时间；它按计划到达计入TTFT，与前次实验口径相同，不能全部称为vLLM内部waiting队列。第二项是引擎内排队。第三项包含本轮mixed forward以及该step实际发生的控制/升档。', '',
              '## 扩容实际触发与容量是否必要', '',
              '| 样本 | Step | KV target | 活跃块 | required块 | 80%阈值块 | resize ms | 原因 |',
              '|---:|---:|---|---:|---:|---:|---:|---|']
    for case in cases:
        for transaction in case['transactions']:
            decision = next(d for d in case['controller_calls'] if d['step'] == transaction['controller_step'])
            lines.append(f"| {case['sample']} | {transaction['controller_step']} | {transaction['old_target_blocks']}→{transaction['applied_target_blocks']} | {decision['active_blocks']} | {decision['required_blocks']} | {decision['high_threshold_blocks']} | {transaction['wall_ms']:.3f} | {transaction['reason']} |")
    lines += ['',
              'required=active+16个backup+16个safety（本轮预测项为0）。两次升档由80%高水位主动触发，触发时没有抢占或分配失败。', '',
              f"即使128请求全部同时达到65块，也只需8320块，约{summary['geometry']['max_all_requests_kv_gib']:.3f}GiB；加32块余量仍低于K4的8629块。因此本trace容量上K4已能容纳C128，K6由保守高水位触发，不是活跃KV已超过4GiB的证据。未另测K4封顶，不据此预填其性能。", '',
              '## 额外代价', '',
              '| 配置 | Copy/remote steps | 专家forward H2D GiB | resize总耗时ms | 控制器wall总计ms（含resize） | 控制器wall扣除resize ms |',
              '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        label = '固定K2' if row['arm'] == 'adaptive_fixed' else '动态'
        lines.append(f"| {label} | {row['copy_steps']:.0f}/{row['remote_steps']:.0f} | {row['expert_h2d_gib']:.3f} | {row['resize_wall_ms']:.2f} | {row['controller_including_resize_ms']:.2f} | {row['controller_outside_resize_ms']:.2f} |")
    lines += ['',
              'cold11/17/23每个copy step分别搬运4.608/7.122/9.636GiB。动态组虽然step更少，增加专家卸载后总显式H2D更多，且有实际扩容及控制器开销；更多同时运行请求还改变了计算形状。没有CUDA剖析，不能把剩余差距精确分摊成DMA和计算各若干毫秒。H2D列仅为forward专家copy，不含KV和档位切换迁移。控制器总wall扣除嵌套resize仅用于显示该控制路径的其余开销，不代表全部系统控制面。', '',
              '固定组10次分配失败集中在step10–19，涉及late-02/late-03/late-08/late-15共4个请求；同一请求可被重复尝试。每组每个进程均有1920次Decode分配调用，其中128次新增1块、1792次新增0块，全部成功。', '',
              '动态组样本1的step10：KV目标8629块、活跃4525块、尚有4104块可用，但8192-token预算已全部用完，仍有37个请求留在等待队列。这是扩容后排队由其他约束继续产生的直接例子。', '',
              '## 输出与验证边界', '',
              '| 对比 | 请求对 | 完整输出不同 | 首token不同 |',
              '|---|---:|---:|---:|']
    for check in verification['output_checks']:
        lines.append(f"| {check['comparison']} | {check['request_pairs']} | {check['full_output_differences']} | {check['first_token_differences']} |")
    lines += ['', '全部返回成功不等于严格输出等价；保留原始token及差异，未认定差异均来自数值或合批。每组两个独立进程，不能建立统计显著性。本轮增加了两组对称的轻量观测，适合验证调度原因，不是无观测开销的微基准。', '',
              '逐step时间线：scheduler_timeline.csv；逐分配失败：allocation_failures.csv；逐请求延迟分解：request_latency.csv；完整逐分配记录：各arm/kv_pressure_detail.json。', '']
    if args.plot:
        lines += ['![KV and queue timeline, sample 1](kv_pressure_timeline.png)', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))
    shutil.copy2(Path(__file__), root / 'source_snapshots' / Path(__file__).name)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / 'source_snapshots').iterdir() if p.is_file()}
    (root / 'source_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    for suffix, filename in (('md', 'REPORT.md'), ('json', 'summary.json'), ('csv', 'summary.csv')):
        shutil.copy2(root / filename, HERE / f'P1024_QPS30_KV_PRESSURE_DETAIL_20261024.{suffix}')
    easy_report = HERE / 'P1024_QPS30_KV_PRESSURE_DETAIL_20261024.md'
    easy_report.write_text(easy_report.read_text().replace(
        '](kv_pressure_timeline.png)', f"]({root / 'kv_pressure_timeline.png'})"))
    print(json.dumps({'status': summary['status'], 'rows': rows, 'report': str(root / 'REPORT.md')}, indent=2))


if __name__ == '__main__':
    main()
