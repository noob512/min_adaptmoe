#!/usr/bin/env python3
"""Keep the existing replay unchanged and record the dynamic controller state."""
import json
import os
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'experiments/eurosys_moe_offload_evaluation/e11_continuous_arrival_runner.py'


def main():
    import vllm
    from vllm.model_executor.models import utils
    from vllm.v1.worker import gpu_model_runner as model_runner
    from vllm.v1.core.uvm_kv_runtime_policy import snapshot_uvm_kv_cache_physical_spans

    output = Path(sys.argv[sys.argv.index('--output')+1])
    result = {'controller_steps':[], 'environment':{
        k:v for k,v in os.environ.items() if k.startswith(('VLLM_UVM_KV_', 'VLLM_CPU_OFFLOAD_PLAN_CLASS_'))}}
    original_llm = vllm.LLM
    original_rebalance = model_runner.GPUModelRunner.rebalance_uvm_kv_cache_after_warmup

    def observed_rebalance(self, target_blocks):
        # Capture the existing startup audits; do not issue extra residency
        # queries or bypass the physical placement check.
        diagnostic = {'target_blocks':target_blocks,
                      'spans':snapshot_uvm_kv_cache_physical_spans(), 'audits':[]}
        original_query = model_runner.driver_batch_query_ranges_residency_bytes

        def observed_query(ranges, *args, **kwargs):
            value = original_query(ranges, *args, **kwargs)
            diagnostic['audits'].append({'ranges':ranges, 'residency':value,
                'requested_bytes':sum(size for _,size in ranges),
                'unaligned_start_count':sum(ptr % 4096 != 0 for ptr,_ in ranges),
                'unaligned_end_count':sum((ptr+size) % 4096 != 0 for ptr,size in ranges)})
            return value

        model_runner.driver_batch_query_ranges_residency_bytes = observed_query
        try:
            value = original_rebalance(self, target_blocks)
            diagnostic['result'] = value
            return value
        finally:
            model_runner.driver_batch_query_ranges_residency_bytes = original_query
            (output.parent/'kv_startup_residency.json').write_text(json.dumps(diagnostic,indent=2)+'\n')

    model_runner.GPUModelRunner.rebalance_uvm_kv_cache_after_warmup = observed_rebalance

    class ObservedLLM(original_llm):
        def __init__(self, *args, **kwargs):
            # Plan-class eager initialization requires enforce_eager in
            # addition to compilation/cudagraph NONE (the E13 protocol).
            kwargs['enforce_eager'] = True
            # 18-KiB MLA blocks must be paired to keep every per-layer
            # migration/query boundary aligned to the 4-KiB UVM page.
            expected_pool_blocks = int(os.environ.get('VLLM_EXPERIMENT_KV_POOL_BLOCKS', '12944'))
            kwargs['num_gpu_blocks_override'] = expected_pool_blocks
            super().__init__(*args, **kwargs)
            core = self.llm_engine.engine_core.engine_core
            policy = core.scheduler.kv_cache_manager.uvm_kv_runtime_policy
            if os.environ.get('VLLM_EXPERIMENT_RESIZE_TELEMETRY')=='1':
                runner=core.model_executor.driver_worker.worker.model_runner
                workspaces={}
                for group in runner._attn_group_iterator():
                    for builder in group.metadata_builders:
                        workspace=getattr(builder,'chunked_prefill_workspace',None)
                        if workspace is not None:
                            workspaces[int(workspace.data_ptr())]=int(workspace.numel()*workspace.element_size())
                workspace_ranges=sorted(workspaces.items())
                workspace_residency=model_runner.driver_batch_query_ranges_residency_bytes(
                    workspace_ranges,device=0,wait_for_trackers=False)
                diagnostic={'ranges':workspace_ranges,'residency':workspace_residency,
                            'expected_bytes':sum(workspaces.values())}
                result['startup_attention_workspaces']=diagnostic
                (output.parent/'mla_workspace_startup_residency.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
                assert workspace_ranges and model_runner._uvm_kv_ranges_have_exclusive_residency(
                    workspace_ranges,workspace_residency,location='gpu'),diagnostic
            assert policy.dynamic_pool_active(), 'dynamic KV pool did not activate'
            assert policy.bytes_per_block == 497664, policy.bytes_per_block
            result['initial_controller'] = core.get_uvm_kv_autoscale_snapshot()
            result['initial_plan'] = utils.cpu_offload_uvm_plan_class_registry()
            expected_autoscale = os.environ.get('VLLM_EXPERIMENT_KV_VARIANT') != 'fixed_managed'
            assert result['initial_controller']['enabled'] is expected_autoscale
            initial_target = int(os.environ.get('VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_GROW_TARGET_BLOCKS')
                                 or os.environ['VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS'])
            target_classes = {int(k):v for k,v in json.loads(os.environ['VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_MAP']).items()}
            assert initial_target in target_classes
            assert result['initial_controller']['current_target_blocks'] == initial_target
            expected_targets = [int(item) for item in os.environ['VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS'].split(',')]
            assert expected_targets in ([2155,4313], [4313,8629], [4313,8629,12943])
            assert result['initial_controller']['config']['allowed_target_blocks'] == expected_targets
            assert result['initial_controller']['config']['maximum_blocks'] == expected_targets[-1]
            assert result['initial_plan']['active_class'] == target_classes[initial_target]
            if os.environ.get('VLLM_EXPERIMENT_RESIZE_TELEMETRY')=='1':
                transactions=[]
                result['resize_transactions']=transactions
                original_resize=core._resize_uvm_kv_target
                def observed_resize(target_blocks,*resize_args,**resize_kwargs):
                    entry={'index':len(transactions)+1,'controller_step':len(result['controller_steps'])+1,
                           'old_target_blocks':int(policy._dynamic_pool_applied_target_blocks),
                           'requested_target_blocks':int(target_blocks),'reason':resize_kwargs.get('reason'),
                           'started_epoch':time.time(),'started_perf_counter':time.perf_counter(),
                           'active_plan_class_before':utils.cpu_offload_uvm_plan_class_registry()['active_class']}
                    transactions.append(entry)
                    path=output.parent/'resize_transactions.json'
                    path.write_text(json.dumps(transactions,indent=2)+'\n')
                    try:
                        value=original_resize(target_blocks,*resize_args,**resize_kwargs)
                        entry['result']=value
                        return value
                    except BaseException as exc:
                        entry['exception']=repr(exc)
                        raise
                    finally:
                        entry.update({'finished_epoch':time.time(),'finished_perf_counter':time.perf_counter(),
                                      'applied_target_blocks':int(policy._dynamic_pool_applied_target_blocks),
                                      'active_plan_class_after':utils.cpu_offload_uvm_plan_class_registry()['active_class']})
                        entry['wall_ms']=(entry['finished_perf_counter']-entry['started_perf_counter'])*1000
                        path.write_text(json.dumps(transactions,indent=2)+'\n')
                core._resize_uvm_kv_target=observed_resize
            original_step = self.llm_engine.step

            def observed_step(*args, **kwargs):
                if not expected_autoscale:
                    assert int(policy._dynamic_pool_applied_target_blocks) == initial_target, 'fixed KV target changed before step'
                value = original_step(*args, **kwargs)
                if not expected_autoscale:
                    assert int(policy._dynamic_pool_applied_target_blocks) == initial_target, 'fixed KV target changed during step'
                # Read the existing decision; no extra whole-pool scan per step.
                started = time.perf_counter()
                decision = getattr(core, '_uvm_kv_autoscale_last_decision', {})
                queue = decision.get('queue_snapshot', {})
                record = {key:decision.get(key) for key in (
                    'step','ok','action','reason','active_blocks','required_blocks','predicted_blocks',
                    'current_target_blocks','high_threshold_blocks','preemption_count',
                    'backup_activated_blocks','low_streak','eligible_round')}
                record.update({'applied_target_blocks':int(policy._dynamic_pool_applied_target_blocks),
                               'queue_invariant_ok':queue.get('invariant_ok'),
                               'active_plan_class':utils.cpu_offload_uvm_plan_class_registry()['active_class'],
                               'observed_epoch':time.time()})
                record['observation_cpu_ms'] = (time.perf_counter()-started)*1000
                result['controller_steps'].append(record)
                return value

            self.llm_engine.step = observed_step
            result['policy_bytes_per_block'] = int(policy.bytes_per_block)
            result['policy_total_blocks'] = int(policy.total_blocks)
            assert policy.total_blocks == expected_pool_blocks-1

    vllm.LLM = ObservedLLM
    rc = 0
    try:
        runpy.run_path(str(BASE), run_name='__main__')
    except SystemExit as exc:
        rc = int(exc.code or 0)
    finally:
        vllm.LLM = original_llm
        model_runner.GPUModelRunner.rebalance_uvm_kv_cache_after_warmup = original_rebalance
        result['base_runner_path'] = str(BASE)
        if output.exists():
            value = json.loads(output.read_text())
            value['dynamic_kv'] = result
            temporary = output.with_suffix('.tmp.json')
            temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
            temporary.replace(output)
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
