"""Opt-in, CPU-only per-step scheduler evidence for the mixed trace sweep."""
import atexit
import json
import os
import time

if os.environ.get('MIXED_SCHEDULER_TRACE'):
    from vllm.v1.core.sched.scheduler import Scheduler

    original_schedule = Scheduler.schedule
    stream = None
    sequence = 0

    def traced_schedule(self, *args, **kwargs):
        global stream, sequence
        result = original_schedule(self, *args, **kwargs)
        if stream is None:
            stream = open(os.environ['MIXED_SCHEDULER_TRACE'], 'a', buffering=65536)
            atexit.register(stream.close)
        new_ids = {r.req_id for r in result.scheduled_new_reqs}
        counts = result.num_scheduled_tokens
        prefill = bool(new_ids) or any(n > 1 for n in counts.values())
        decode = any(r not in new_ids and n == 1 for r, n in counts.items())
        pool = self.kv_cache_manager.block_pool
        record = {'epoch': time.time(), 'step': sequence,
                  'tokens': result.total_num_scheduled_tokens,
                  'phase': 'mixed' if prefill and decode else 'prefill' if prefill else 'decode',
                  'scheduled_requests': len(counts),
                  'running': len(self.running), 'waiting': len(self.waiting),
                  'preemptions': len(result.preempted_req_ids or []),
                  'kv_usage': pool.get_usage()}
        stream.write(json.dumps(record, separators=(',', ':')) + '\n')
        sequence += 1
        if sequence % 64 == 0:
            stream.flush()
        return result

    Scheduler.schedule = traced_schedule
