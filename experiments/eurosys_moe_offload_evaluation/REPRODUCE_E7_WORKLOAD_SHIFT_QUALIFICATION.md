# 复现 E7 workload-shift qualification

## 范围

该流程在一个模型进程内顺序运行一次 calibration 和六个完整 shift arm，不做统计重复：

```text
evaluation calibration
  -> frequency-cold -> random -> frozen static
  -> GPU heat + CPU EWMA -> oracle -> hot-adversarial
```

每个 arm 使用同一批 32×128-token shift prompts、每请求固定输出 32 tokens。输出目录必须
不存在；失败 attempt 必须保留，不能覆盖或从 partial result 补写成功记录。

## 冻结配置

| 参数 | 值 |
| --- | --- |
| 模型 | Qwen3-30B-A3B-FP8，snapshot `d206ba732169f29bb77fbf80fc2c4b81d4d30782` |
| pre-shift / shift | evaluation short-dialogue / code-shift short-dialogue，各 32 请求 |
| 请求 | B1，Prompt 128，Output 32 |
| window | 4 requests，8 windows/arm |
| base / superset | cold-6G / cold-7G |
| 动态 slots | superset − base = 48 层×4 experts |
| static / dynamic bytes | 6,568,280,064 / 905,969,664 B |
| EWMA | alpha 0.25，1 swap/layer/window，min delta 1 |
| KV | 268,435,456 B |
| Graph | `FULL_SEPARATE`，`[1,128]` |
| seed | 20260829 |

## 前置检查

```bash
cd /home/ubuntu/nvidia-uvm-gpu
bash experiments/eurosys_moe_offload_evaluation/run_e7_workload_shift_qualification.sh preflight
nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
sudo fuser -v /dev/nvidia-uvm /dev/nvidia-uvm-tools 2>/dev/null || true
printf 'loaded=%s\nstock=%s\n' \
  "$(tr -d '[:space:]' </sys/module/nvidia_uvm/srcversion)" \
  "$(modinfo -F srcversion nvidia-uvm | tr -d '[:space:]')"
```

GPU 工作开始前必须为 0 MiB、无 compute/UVM client，并加载系统 stock UVM。harness 会暂停
已知 exporter、独占 phase-switch lock、加载 custom UVM；退出路径无论成功失败都恢复 stock
module 和 exporter。不得自动终止未知 GPU 进程。

## 单样本完整运行

```bash
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/e7-workload-shift-REPRO-<tag>-qualification \
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/run_e7_workload_shift_qualification.sh qualify
```

已完成记录使用：

```text
artifacts/e7-workload-shift-REPRO-20260829-attempt-03-qualification
```

一次成功运行应执行 32 个 calibration 请求和 6×32 个 shift 请求，共 224 个请求。arm setup
transition 位于各 arm 计时前并单独记录；固定 arm 每 window 的 heat snapshot 是实验观测开销，
adaptive arm 的 snapshot、EWMA、plan 和真实 swap 均计入 maintenance。

## 验收

成功 harness 会自动运行 verifier，也可离线再次验证同一 raw artifact：

```bash
bash experiments/eurosys_moe_offload_evaluation/run_e7_workload_shift_qualification.sh \
  verify experiments/eurosys_moe_offload_evaluation/artifacts/e7-workload-shift-REPRO-<tag>-qualification

jq '{status,arm_count,request_count_total,graph_capture_count,
  ewma_convergence_requests,actual_remote_demand_bytes_available}' \
  experiments/eurosys_moe_offload_evaluation/artifacts/e7-workload-shift-REPRO-<tag>-qualification/verification.json
```

必须同时满足：

```text
runner status=ok
verifier status=ok
arm order=frequency_cold,random,frozen_static,gpu_heat_cpu_ewma,oracle,hot_adversarial
calibration=32 requests；每 arm=32 requests/8 windows
六臂 output hash 和 route-count hash 完全一致
每层动态 remote slots 始终为 4，总数 192
oracle 是固定计划中 dynamic remote assignments 最小值
hot-adversarial 是固定计划中 dynamic remote assignments 最大值
adaptive 至少执行一个真实 swap
Graph capture count 2 -> 2
distributed cleanup=true；cleanup_errors=[]
module.after 为 stock UVM；gpu.after 为 0 MiB/clean
```

本机成功 result/verification SHA256 应为：

```text
3190d36d37a0ca9eaccb5bbeb35d4dcfc3b9f553815e0ea7240df1ade917187c  result.json
77e1d258e7f236ab35db00acd88d00ba8b08f98f19286003526b8b2d929ffb29  verification.json
```

## 解释边界

- `oracle` 从完整 shift route 离线构造，使用未来信息，只是上界；不得称为在线方法。
- 当前 heat 路径在 CUDA Graph 下为 `graph_stable_synchronous`。异步 pinned D2H 未完成时，
  所有维护数字必须标注 synchronous。
- 当前 driver counters 不提供逐 arm demand-read bytes；该字段必须保持 unavailable，不得用
  assignment count 乘专家大小替代。
- `ewma_convergence_requests=null` 表示 32 requests 内未达到 95% plan-hit 门，不得改写为
  “32 requests 收敛”。
- 本流程是单样本 qualification。正式 E7 至少需要 5 seeds、平衡 arm order/独立冷进程以及
  相同重复数后再报告 median、p95 和置信区间。
