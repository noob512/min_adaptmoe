# 复现 E12 Graph replay与双页表PARK/RESUME qualification

最后核验：2026-09-02。当前仍无更新的E12运行；最新有效证据为：

```text
Graph replay  artifacts/e12-graph-soak-1000-REPRO-20260827-attempt-01.json
双页表往返   artifacts/e12-dual-view-soak-20260831T130345Z-attempt-01-qualification
```

## 1,000次Graph replay smoke

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm
env CUDA_VISIBLE_DEVICES=0 taskset -c 0-13 .venv/bin/python \
  ../../../experiments/vllm_p3_static_alias_graph_integration/smoke_static_alias_graph.py \
  --output ../../../experiments/eurosys_moe_offload_evaluation/artifacts/\
e12-graph-soak-1000-NEW-attempt-01.json \
  --manifest ../../../experiments/eurosys_moe_offload_evaluation/artifacts/\
e12-graph-soak-1000-NEW-attempt-01.manifest.jsonl \
  --layer-mib 576 --warmup-replays 3 --measured-replays 1000
```

## 1,000次完整双页表往返

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e12-dual-view-soak-NEW-attempt-01-qualification \
CYCLES=1000 VERIFY_EVERY=1 CHECKPOINT_EVERY=100 \
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/run_e12_dual_view_soak_qualification.sh
```

验收要求1000/1000次PARK和RESUME逐轮数据检查通过、双页表父块覆盖100%、无range-map
回退、Xid、PDE/PTE fault或资源泄漏，退出后恢复stock UVM和GPU 0 MiB。

该循环只满足机制次数门，不是6/24小时整模型服务soak，也不包含KV峰谷和barrier故障注入。

参考结果：Graph 1,000/1,000 replay且full-byte error为0；PARK/RESUME 1,000/1,000 cycle
通过，PARK p50/p95为0.769/0.838 ms，RESUME为0.567/0.611 ms。两份结果来自历史模块，
若用当前源码重跑必须保存新模块指纹并重新验收，不能直接替换旧轮。
