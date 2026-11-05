# 复现 E7 六臂独立冷进程workload-shift qualification

最后核验：2026-09-02。当前权威artifact：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
e7-independent-cold-20260902-attempt-25-targeted-canonical-pte-qualification
```

六个arm分别使用独立冷进程，每arm运行一次32-request shift trace。冻结target来自同一
calibration，不得在测量后重新选择。该轮使用全覆盖双页表和canonical target PTE修复；
verifier为`status=ok`，六臂输出及路由计数hash一致，Graph capture count保持2。

## 前置检查与运行

```bash
cd /home/ubuntu/nvidia-uvm-gpu
bash experiments/eurosys_moe_offload_evaluation/run_e7_independent_cold_qualification.sh preflight

nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
sudo fuser -v /dev/nvidia-uvm /dev/nvidia-uvm-tools 2>/dev/null || true

RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e7-independent-cold-NEW-attempt-01-qualification \
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/run_e7_independent_cold_qualification.sh qualify

bash experiments/eurosys_moe_offload_evaluation/run_e7_independent_cold_qualification.sh \
  verify "$RUN_ROOT"
```

开始前必须是stock UVM、GPU 0 MiB且无compute/UVM client。失败attempt原样保留。

## 当前结果

| Arm | 请求中位延迟 | cycle tok/s | 动态remote率 | Oracle hit | 同步维护 |
|---|---:|---:|---:|---:|---:|
| frequency-cold | 838.62 ms | 35.15 | 1.427% | 36.98% | 90.75 ms |
| random | 868.77 ms | 34.14 | 3.186% | 4.17% | 74.37 ms |
| frozen static | 804.58 ms | 36.30 | 1.688% | 18.23% | 72.10 ms |
| GPU heat + CPU EWMA | 851.22 ms | 20.97 | 0.712% | 88.54% | 19765.65 ms |
| oracle | 811.88 ms | 35.98 | 0.178% | 100% | 73.36 ms |
| hot-adversarial | 1002.28 ms | 31.54 | 12.484% | 0% | 72.12 ms |

Adaptive仍把动态remote assignment降低50.12%，但请求中位延迟比frequency-cold高1.50%，
同步维护使cycle throughput下降40.35%。因此本轮只证明放置方向和机制链路正确，不能声称
包含维护成本后的端到端性能提高。这是每arm一次的qualification，无方差或置信区间。
