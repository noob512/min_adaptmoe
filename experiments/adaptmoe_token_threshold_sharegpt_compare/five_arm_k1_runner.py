#!/usr/bin/env python3
"""Keep common replay timing and align native KV to the managed K1 capacity."""
import json
import runpy
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    mode=sys.argv[sys.argv.index('--mode')+1]
    if not mode.startswith('native_'):
        runpy.run_path(str(HERE/'qps30_kv_pressure_runner.py'),run_name='__main__')
        return
    import vllm
    output=Path(sys.argv[sys.argv.index('--output')+1])
    original=vllm.LLM
    observation={}

    class NativeLLM(original):
        def __init__(self,*args,**kwargs):
            kwargs['enforce_eager']=True
            kwargs['num_gpu_blocks_override']=2156
            super().__init__(*args,**kwargs)
            manager=self.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager
            pool=manager.block_pool
            assert pool.num_gpu_blocks==2156
            assert not manager.uvm_kv_runtime_policy.enabled
            observation.update({'pool_blocks_including_null':pool.num_gpu_blocks,
                                'usable_blocks':pool.num_gpu_blocks-1,'usable_tokens':(pool.num_gpu_blocks-1)*16,
                                'kv_tensor_bytes':pool.num_gpu_blocks*497664,
                                'managed_kv':False,'autoscaling':False})

    vllm.LLM=NativeLLM
    try:
        runpy.run_path(str(HERE.parent/'eurosys_moe_offload_evaluation/e11_continuous_arrival_runner.py'),run_name='__main__')
    finally:
        vllm.LLM=original
        if output.exists():
            value=json.loads(output.read_text());value['fixed_kv_capacity']=observation
            temporary=output.with_suffix('.tmp.json')
            temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temporary.replace(output)


if __name__=='__main__':main()
