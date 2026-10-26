#!/usr/bin/env python3
"""Replay the frozen trace, observing scheduler pressure without scanning KV."""
import json
import os
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'eurosys_moe_offload_evaluation/e11_continuous_arrival_runner.py'


def main():
    import vllm

    output=Path(sys.argv[sys.argv.index('--output')+1])
    variant=os.environ['VLLM_EXPERIMENT_KV_VARIANT']
    assert variant in ('fixed','dynamic','fixed_managed')
    telemetry={'variant':variant,'scheduler_steps':[]}
    original_llm=vllm.LLM
    startup_audit=os.environ.get('VLLM_EXPERIMENT_STARTUP_RESIDENCY_AUDIT')=='1'
    startup_reconcile=os.environ.get('VLLM_EXPERIMENT_FIXED_REMOTE_KERNEL_WARMUP')=='1'
    if startup_reconcile:
        assert variant=='fixed'
    if startup_audit or startup_reconcile:
        from vllm.device_allocator import uvm
        from vllm.model_executor.models import utils
        from vllm.v1.worker import gpu_worker, gpu_model_runner
        audits=[]

        def audit(stage):
            ranges=[r for _,b in sorted(utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items())
                    for r in utils._uvm_original_alias_binding_hot_ranges(b)]
            value=uvm.driver_batch_query_ranges_residency_bytes(ranges,device=0,wait_for_trackers=False)
            audits.append({'stage':stage,'residency':value,'hot_bytes':sum(n for _,n in ranges)})
            (output.parent/'startup_residency_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
        original_kernel_warmup=gpu_worker.kernel_warmup
        def observed_kernel_warmup(worker):
            audit('before_kernel_warmup')
            if startup_reconcile:
                # Autotune executes a maximum-token model forward. Its
                # workspace growth can evict hot weights while that forward
                # is already reading them. Use the existing temporary remote
                # startup topology, then restore before sampler/measurement.
                old=os.environ.get('VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE')
                os.environ['VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE']='1'
                try:
                    prepared=utils.prepare_cpu_offload_uvm_original_va_profile_topology(
                        transition_reason='fixed_kv_kernel_warmup_all_remote')
                    assert prepared['ok'] and utils.cpu_offload_uvm_original_va_profile_all_remote_active()
                finally:
                    if old is None:
                        os.environ.pop('VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE',None)
                    else:
                        os.environ['VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE']=old
                audit('kernel_warmup_all_remote_prepared')
            result=original_kernel_warmup(worker)
            audit('after_kernel_warmup')
            if startup_reconcile:
                utils.reconcile_cpu_offload_uvm_original_va_remote_graph_residency(preserve_allocator_cache=True)
                utils.refresh_cpu_offload_uvm_original_va_full_dual_view()
                utils.activate_cpu_offload_uvm_original_va_prefill_topology()
                assert not utils.cpu_offload_uvm_original_va_profile_all_remote_active()
                audit('after_remote_kernel_warmup_restore')
            return result
        gpu_worker.kernel_warmup=observed_kernel_warmup
        original_dummy=gpu_model_runner.GPUModelRunner._dummy_run
        def observed_dummy(self,*args,**kwargs):
            if kwargs.get('num_tokens')==128:
                audit('before_sampler_dummy_128')
            return original_dummy(self,*args,**kwargs)
        gpu_model_runner.GPUModelRunner._dummy_run=observed_dummy

    class ObservedLLM(original_llm):
        def __init__(self,*args,**kwargs):
            kwargs['enforce_eager']=True
            assert kwargs['max_num_seqs']==128
            if variant=='fixed':
                assert kwargs['kv_cache_memory_bytes'] in (536870912,1073741824,2147483648)
                assert os.environ['VLLM_UVM_KV_AUTO_SCALE_ENABLE']=='0'
                assert os.environ['VLLM_UVM_KV_CACHE_MANAGED']=='0'
            super().__init__(*args,**kwargs)
            scheduler=self.llm_engine.engine_core.engine_core.scheduler
            pool=scheduler.kv_cache_manager.block_pool
            telemetry['initial_pool_blocks_including_null']=pool.num_gpu_blocks
            expected_blocks=int(os.environ.get('VLLM_EXPERIMENT_KV_POOL_BLOCKS','12944')) if variant in ('dynamic','fixed_managed') else kwargs['kv_cache_memory_bytes']//497664
            assert pool.num_gpu_blocks==expected_blocks,(pool.num_gpu_blocks,expected_blocks)
            original_schedule=scheduler.schedule

            def observed_schedule():
                value=original_schedule()
                before=time.perf_counter()
                free_counts={name:int(getattr(getattr(pool,name,None),'num_free_blocks',0))
                             for name in ('free_block_queue','backup_block_queue','offline_block_queue')}
                telemetry['scheduler_steps'].append({
                    'step':len(telemetry['scheduler_steps'])+1,
                    'active_blocks':int(pool.num_gpu_blocks)-1-sum(free_counts.values()),
                    'free_counts':free_counts,
                    'running_requests':len(scheduler.running),
                    'waiting_requests':len(scheduler.waiting),
                    'preempted_requests':len(value.preempted_req_ids or ()),
                    'scheduled_tokens':int(value.total_num_scheduled_tokens),
                    'observation_cpu_ms':(time.perf_counter()-before)*1000,
                })
                return value

            scheduler.schedule=observed_schedule

    vllm.LLM=ObservedLLM
    rc=0
    try:
        target=HERE/'dynamic_kv_p512_runner.py' if variant in ('dynamic','fixed_managed') else BASE
        runpy.run_path(str(target),run_name='__main__')
    except SystemExit as exc:
        rc=int(exc.code or 0)
    finally:
        vllm.LLM=original_llm
        if output.exists():
            value=json.loads(output.read_text())
            value['kv_concurrency']=telemetry
            temporary=output.with_suffix('.tmp.json')
            temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
            temporary.replace(output)
    raise SystemExit(rc)


if __name__=='__main__':
    main()
