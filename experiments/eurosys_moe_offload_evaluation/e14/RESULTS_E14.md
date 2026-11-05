# E14：同请求集冷专家与固定整层卸载的 eager UVA 并发扩展性

当前权威结果为 2026-09-03 request-specific direct-forward formal。先记录 E14 的 32 个冻结请求在 Qwen3
上的实际路由频率，再按每层综合 Prefill+Decode 命中数最低的 29 个专家生成新计划；随后
关闭路由跟踪和逐 forward copy trace，启用 remote-only direct forward，按 `DWWDDW`
完成两臂各 3 个独立冷进程、每点 3 轮测量。六个进程、机制门、清理门和全部四项趋势门
均通过。

权威 artifact：
`artifacts/e14-uva-eager-request-cold29-direct-formal-20260903-attempt-01/`。

路由采集 artifact：`artifacts/e14-request-routes-b32-20260903-attempt-01/`。

## 请求路由频率与冷专家计划

路由采集只统计一个完整 B32 measurement 时间窗，启动和模型 warmup 记录均被排除：

- 32 个不同冻结请求，每请求 8-token Prefill，共 256 个 Prefill token；
- 5 个 warmup Decode step 加 16 个 measured Decode step，共 672 个 Decode token；
- 48 层各 22 条记录，共 1,056 条层级记录、356,352 个 Top-8 路由槽；
- 每层从 128 个专家中选综合命中数最低的 29 个，命中数相同则按 expert ID；
- 选中的 1,392 个层-专家坐标承载 3,680 个槽，占 1.033%；其中 Prefill 为 1.244%，
  Decode 为 0.952%，541 个坐标零命中；
- 新旧 cold-29 计划重叠 739/1,392 个坐标，653 个坐标被重新选择。

冻结计划为
[request_specific_cold29_plan.json](inputs/request_specific_cold29_plan.json)，其 SHA256 为
`52c98cd00883ee4bf33532b0151a5b5d414f62b8529d719fe026358ad911c1fd`。三个 D 性能进程的
manifest 均逐层匹配该计划，且性能进程中
`VLLM_MOE_TOKEN_EXPERT_TRACE_ENABLE=0`。

这是“同请求集选冷、同请求集评测”的 in-distribution oracle placement。它直接回答当前
请求集上的效果，但不能证明该名单能泛化到其他请求分布。

## Formal 协议

- 模型：Qwen3-30B-A3B-FP8；B1/B4/B16/B32 分别使用前 1/4/16/32 个不同冻结请求；
- `D`：48 个 MoE 层各卸载 request-specific cold-29，共 1,392 个专家实例、6.117 GiB；
- `W`：固定完整卸载 L0--L10，共 1,408 个专家实例、6.188 GiB；
- 两臂权重字节差 1.15%，均为 original-VA managed UVA `remote_read`，服务期无 H2D；
- 两臂均设置 `VLLM_UVM_REMOTE_ONLY_DIRECT_FORWARD=1` 和
  `VLLM_CPU_OFFLOAD_COPY_TRACE=0`；初始化完成并审计远端映射后直接执行原始
  `module.forward`，服务期不进入通用 copy/static-alias wrapper；
- Prefill、Decode 均为 eager：`enforce_eager=True`、`CompilationMode.NONE`、
  `CUDAGraphMode.NONE`，没有 CUDA Graph capture/replay；
- 每个 B 先 conditioning 1 次，再正式测量 3 次；先取进程内中位数，再取三个冷进程
  中位数的跨进程中位数。

## Formal 主结果

| B | placement | Prefill ms | Prefill tok/s | Decode ms/step | Decode tok/s |
|---:|---|---:|---:|---:|---:|
| 1 | D：逐层 request-cold29 | 61.20 | 130.71 | 62.46 | 16.01 |
| 1 | W：固定整层 L0--L10 | 58.65 | 136.41 | 62.83 | 15.91 |
| 4 | D：逐层 request-cold29 | 64.19 | 498.50 | 65.29 | 61.27 |
| 4 | W：固定整层 L0--L10 | 96.14 | 332.86 | 62.33 | 64.17 |
| 16 | D：逐层 request-cold29 | 61.38 | 2085.27 | 64.83 | 246.81 |
| 16 | W：固定整层 L0--L10 | 127.88 | 1000.97 | 81.43 | 196.48 |
| 32 | D：逐层 request-cold29 | 61.68 | 4150.22 | 62.25 | 514.10 |
| 32 | W：固定整层 L0--L10 | 143.34 | 1785.98 | 101.89 | 314.07 |

| B | D/W Prefill throughput | D/W Decode throughput |
|---:|---:|---:|
| 1 | 0.958× | 1.006× |
| 4 | 1.498× | 0.955× |
| 16 | 2.083× | 1.256× |
| 32 | 2.324× | 1.637× |

B1 时 D/W 为 0.958×/1.006×，即 Prefill 低 4.2%、Decode 高 0.6%，符合“低并发基本
一致”的目标；这是三冷进程中位数结果，不是预注册统计等价检验。B32 时 D 已分别快
132.4% 和 63.7%。从 B1 到 B32，D/W 吞吐倍率在 Prefill 和 Decode 分别增加 136.6 和
63.1 个百分点；Decode 在 B4 仍基本一致，到 B16/B32 出现明显优势。

## 扩展效率与验收

| B | D Prefill | W Prefill | D Decode | W Decode |
|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | 0.953 | 0.610 | 0.957 | 1.008 |
| 16 | 0.997 | 0.459 | 0.964 | 0.772 |
| 32 | 0.992 | 0.409 | 1.003 | 0.617 |

全部趋势门通过：B1 两阶段均在 ±25%；B32 D 两阶段均至少快 15%；B32 D 两阶段线性
效率均不低于 90%；B32 W 两阶段线性效率均不高于 75%。固定整层的绝对吞吐仍然增长，
但到 B32 的扩展效率只有 40.9%/61.7%，逐层冷专家为 99.2%/100.3%。

旧的 request-specific formal（`e14-uva-eager-request-cold29-formal-20260903-attempt-02`）
仍启用了逐 forward 通用 wrapper 和 JSON trace。移除这两项后，D 的 B1 Prefill/Decode
吞吐从 100.47/11.85 提升到 130.71/16.01 tok/s，而 W 从 121.44/14.68 提升到
136.41/15.91 tok/s；B1 的 D/W 因而从 0.827×/0.807×收敛到 0.958×/1.006×。
这个前后对照同时移除了 wrapper 检查和文件 trace，不能进一步拆分二者各自的贡献。

两臂完整输出 hash 在四个 B 上均一致；96 次 Prefill 和 2,016 次 Decode 均为未捕获
eager 执行。六个进程均无 `cpu_offload_copy_forward` 事件；D/W 每进程分别有 48/11 条
仅初始化时写入的 direct-forward 证书。没有服务期 H2D、CUDA Graph capture、
Xid/MMU/PTE/PDE fault。运行结束后 stock UVM 恢复，GPU 为 0 MiB。

复现命令见 [README.md](README.md)。
