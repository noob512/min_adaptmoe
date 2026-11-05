# Physical KV pool request-boundary resize

This qualification reserves a fixed 1.5 GiB managed KV virtual-address range
and changes only its physical residency at idle request boundaries. The low
tier keeps blocks 0..105 on GPU (159 MiB including the null block) and migrates
the remaining 1,377 MiB to CPU. The high tier restores the complete 1.5 GiB
range to GPU without replacing a KV tensor or recapturing CUDA Graphs.

All three Qwen3-30B-A3B-FP8 processes use the same 6.1172 GiB decode-cold
expert plan:

1. dynamic low -> high -> low;
2. direct low, which remains at low;
3. direct-to-high, which starts from the common low initialization state and
   grows once at the first legal scheduler-idle serving boundary.

The direct-to-high definition is intentional. A complete 1.5 GiB KV residency
cannot coexist with the sealed prefill-alias topology on this 32 GiB GPU.
After the mandatory first request places experts in `remote_parked`, the same
high target is fully resident and stable. This is the deployable request-boundary
baseline, not a claim that sealed startup can hold the high tier.

Each measured tier contains three requests. Every request has five decode
warmup steps and sixteen measured stable decode steps. The historical result
table was cleared; new runs must publish their limitations and artifact paths
under the fresh EuroSys protocol.

```bash
./experiments/vllm_uvm_physical_kv_pool_resize/run_experiment.sh static-preflight

ALLOW_GPU_RUN=1 \
  ./experiments/vllm_uvm_physical_kv_pool_resize/run_experiment.sh qualify
```
