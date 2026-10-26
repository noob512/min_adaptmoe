#!/usr/bin/env python3
"""Observe real allocation decisions without changing scheduler/KV policy."""
import json
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    import vllm

    output = Path(sys.argv[sys.argv.index('--output') + 1])
    telemetry = {'schema': 'qps30_kv_pressure_v1', 'scheduler_steps': [],
                 'allocations': [], 'controller_calls': []}
    original_llm = vllm.LLM

    class PressureLLM(original_llm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            core = self.llm_engine.engine_core.engine_core
            scheduler = core.scheduler
            manager = scheduler.kv_cache_manager
            pool = manager.block_pool
            policy = manager.uvm_kv_runtime_policy
            telemetry['bytes_per_block'] = int(policy.bytes_per_block)
            telemetry['max_num_running_reqs'] = int(scheduler.max_num_running_reqs)
            telemetry['max_num_scheduled_tokens'] = int(scheduler.max_num_scheduled_tokens)
            telemetry['initial_target_blocks'] = int(policy._dynamic_pool_applied_target_blocks)
            context = {'step': 0, 'allocation': None}

            def counts():
                # Read O(1) queue counters. Calling get_num_free_blocks would
                # itself refresh policy state, so avoid extra calls to it.
                free = {name: int(getattr(pool, name).num_free_blocks) for name in (
                    'free_block_queue', 'backup_block_queue', 'offline_block_queue')}
                return {'active_blocks': int(pool.num_gpu_blocks) - 1 - sum(free.values()),
                        'free_counts': free,
                        'gpu_available_blocks': free['free_block_queue'] + free['backup_block_queue'],
                        'target_blocks': int(policy._dynamic_pool_applied_target_blocks)}

            def requests_state(requests):
                return [{'request_id': request.request_id,
                         'computed_tokens': int(request.num_computed_tokens),
                         'output_tokens': int(request.num_output_tokens),
                         'num_preemptions': int(request.num_preemptions),
                         'block_counts': [len(group) for group in manager.coordinator.get_blocks(request.request_id)]}
                        for request in requests]

            original_allow = policy.should_allow_allocation

            def observed_allow(*allow_args, **allow_kwargs):
                value = original_allow(*allow_args, **allow_kwargs)
                entry = context['allocation']
                if entry is not None:
                    entry.update({'num_blocks_to_allocate': int(allow_kwargs['num_blocks_to_allocate']),
                                  'policy_allowed': bool(value),
                                  'after_policy': counts()})
                return value

            policy.should_allow_allocation = observed_allow
            original_allocate = manager.allocate_slots

            def observed_allocate(request, num_new_tokens, *allocate_args, **allocate_kwargs):
                observed_started = time.perf_counter()
                entry = {'step': context['step'], 'request_id': request.request_id,
                         'status_before': request.status.name,
                         'computed_tokens_before': int(request.num_computed_tokens),
                         'output_tokens_before': int(request.num_output_tokens),
                         'num_new_tokens': int(num_new_tokens),
                         'existing_block_counts': [len(group) for group in manager.coordinator.get_blocks(request.request_id)],
                         'before': counts()}
                before_cpu = time.perf_counter() - observed_started
                context['allocation'] = entry
                try:
                    value = original_allocate(request, num_new_tokens, *allocate_args, **allocate_kwargs)
                finally:
                    context['allocation'] = None
                observed_started = time.perf_counter()
                entry.update({'success': value is not None, 'after': counts(),
                              'allocated_block_counts': [len(group) for group in value.blocks] if value is not None else None})
                telemetry['allocations'].append(entry)
                entry['outer_observation_cpu_ms'] = (before_cpu + time.perf_counter() - observed_started) * 1000
                return value

            manager.allocate_slots = observed_allocate
            original_schedule = scheduler.schedule

            def observed_schedule():
                context['step'] += 1
                observed_started = time.perf_counter()
                entry = {'step': context['step'], 'started_perf_counter': observed_started,
                         'before': counts(), 'running_before': requests_state(scheduler.running),
                         'waiting_before': [request.request_id for request in scheduler.waiting],
                         'allocation_start_index': len(telemetry['allocations'])}
                before_cpu = time.perf_counter() - observed_started
                value = original_schedule()
                observed_started = time.perf_counter()
                entry.update({'finished_perf_counter': observed_started,
                              'after': counts(), 'running_after': requests_state(scheduler.running),
                              'waiting_after': [request.request_id for request in scheduler.waiting],
                              'scheduled_tokens': int(value.total_num_scheduled_tokens),
                              'remaining_token_budget': int(scheduler.max_num_scheduled_tokens - value.total_num_scheduled_tokens),
                              'preempted_request_ids': sorted(value.preempted_req_ids or ()),
                              'new_request_ids': [request.req_id for request in value.scheduled_new_reqs],
                              'allocation_end_index': len(telemetry['allocations'])})
                telemetry['scheduler_steps'].append(entry)
                entry['outer_observation_cpu_ms'] = (before_cpu + time.perf_counter() - observed_started) * 1000
                return value

            scheduler.schedule = observed_schedule
            original_controller = core._observe_uvm_kv_autoscale_pressure

            def observed_controller(*controller_args, **controller_kwargs):
                started = time.perf_counter()
                value = original_controller(*controller_args, **controller_kwargs)
                record = {key: value.get(key) for key in (
                    'step', 'action', 'reason', 'current_target_blocks', 'active_blocks',
                    'required_blocks', 'high_threshold_blocks', 'predicted_blocks',
                    'backup_activated_blocks', 'preemption_count', 'eligible_round')}
                record.update({'wall_ms': (time.perf_counter() - started) * 1000,
                               'growth': value.get('growth')})
                telemetry['controller_calls'].append(record)
                return value

            core._observe_uvm_kv_autoscale_pressure = observed_controller
            original_step = self.llm_engine.step

            def observed_step(*step_args, **step_kwargs):
                started = time.perf_counter()
                value = original_step(*step_args, **step_kwargs)
                finished = time.perf_counter()
                entry = telemetry['scheduler_steps'][-1]
                entry.update({'engine_started_perf_counter': started,
                              'engine_finished_perf_counter': finished,
                              'after_engine': counts(),
                              'running_after_engine': len(scheduler.running),
                              'waiting_after_engine': len(scheduler.waiting)})
                return value

            self.llm_engine.step = observed_step

    vllm.LLM = PressureLLM
    try:
        runpy.run_path(str(HERE / 'c128_kv_runner.py'), run_name='__main__')
    finally:
        vllm.LLM = original_llm
        (output.parent / 'kv_pressure_detail.json').write_text(json.dumps(telemetry, indent=2) + '\n')


if __name__ == '__main__':
    main()
