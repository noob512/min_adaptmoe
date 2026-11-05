# GPU/free、GPU/backup、CPU/offline 双向 E2E qualification

该实验使用真实 Qwen3-30B-A3B-FP8 模型，在同一实验组进程中验证一次完整的
三队列升压与自动降压往返，而不是只验证 Python 队列单测。

当前 v2 协议不使用请求预测。启动时按冻结的 cold-frequency 计划预捕获两个嵌套档位：
`cold29`（每层 29 个冷专家）和 `cold30`（每层 30 个冷专家）。第一轮请求使用
314-token Prefill 和 6 个输出 token。实验组从 24-block GPU KV prefix 启动，其中
8 blocks 是 GPU backup；请求先耗尽 16 个普通 free blocks，再消费 4 个 backup
blocks。达到冻结的 50% backup 压力门后，在同一调度步触发：

```text
GPU/free -> GPU/backup -> pressure event
         -> cold29 批量切换到 cold30 -> KV 24 -> 40
         -> CPU/offline blocks 迁入 GPU -> backup 补回 8
```

第二轮请求使用 16-token Prefill 和 5 个输出 token。统一 autoscale 控制器在前三个
有效低压调度轮依次产生：

```text
observe_low_pressure -> observe_low_pressure -> shrink_pending
```

第三轮只生成缩容决定；活动请求尚未到安全边界时不会迁移其 KV。最终 Decode 完成后，
控制器在空闲边界执行 `40 -> 20`，把 KV tail 迁回 CPU/offline，再把专家档位从
`cold30` 批量恢复到 `cold29`。

baseline 从固定 40-block target 启动，并关闭 backup 和 autoscale。两个 case 都预捕获
相同的两个 Plan Class，以避免 Graph 配置不对称。
两组都顺序执行：

```text
314-token Prefill Graph -> Decode Graph
16-token Prefill Graph  -> Decode Graph
```

CUDA Graph 使用 `FULL_SEPARATE`，预捕获大小为 `1,16,314`。验证器检查两轮输出 token
逐项一致、Prefill/Decode replay、`remote_parked -> sealed -> remote_parked` 往返、
三队列守恒/互斥、GPU/CPU 物理驻留、KV 与 Parameter VA 稳定、双向档位差量、
Plan Class session/Graph 保留、Graph capture 计数不增加、内核日志以及 shutdown 清理。

验证器仍兼容历史 v1 逐层 incremental-wave artifact。

历史 qualification artifact 已按清理策略移除。本目录保留 runner、验证器和复现方法；
任何新结果必须使用新的 run ID 重新生成，不再引用旧数值。

```bash
./experiments/vllm_uvm_three_queue_e2e/run_experiment.sh static-preflight

ALLOW_GPU_RUN=1 \
  ./experiments/vllm_uvm_three_queue_e2e/run_experiment.sh qualify
```

新结果保存在 `artifacts/<UTC>_qualification/verification.json`。可对已有产物重新验证：

```bash
./experiments/vllm_uvm_three_queue_e2e/run_experiment.sh verify <RUN_ROOT>
```
