# 512×32、PARK=16：逐 scheduler step 差异与根因

日期：2026-09-15。

## 结论

旧的 9.833 vs 11.124 req/s 结果被一个只在 8192-token Prefill
触发的驻留诊断污染，不能用于评价正常 AdaptMoE 数据面。冻结环境中的
`VLLM_UVM_FUSED_MOE_POST_WORKSPACE_RESIDENCY_AUDIT=1` 同时开启了
DeepSeek 第 1 层的四个完整内存/驻留快照。每个快照查询 518 个热区间，
每个 8192-token step 增加约 250 ms 控制面开销。

关闭该诊断后，两进程复跑为 Adaptive 10.966 req/s、Native copy
10.928 req/s，整体吞吐仅相差 +0.35%。剩余 copy-mode 单步差距主要来自
52 次切片 H2D 相对 9 次完整 Parameter H2D 的碎片化；不是 CUDA kernel
计算变慢。

## 协议与有效性

- DeepSeek-V2-Lite-Chat BF16，TRITON_MLA，eager，无 CUDA Graph。
- 32 个相同冻结 ShareGPT prompt，每个 512 token；16 个初始请求，
  engine step 4 后加入另外 16 个；每请求生成 16 token。
- `max_num_batched_tokens=8192`，`PARK=16`，`RESUME=128`。
- Adaptive 搬运 4,948,230,144 B、286 个 expert slices；Native copy
  搬运 4,798,283,776 B、9 个完整 Parameter。
- 无剖析性能为两个独立进程，ABC/CBA 反序；全部输出 hash 跨臂一致。
- Nsight Systems 各臂一次，只用于组件归因，不使用其受扰动 wall time排名。

## 无剖析逐 step 结果

单位为 ms；同 token 值汇总两个独立进程中的所有对应 step。

| 调度 token | Adaptive 模式 | 样本/臂 | Adaptive | Native copy | Adaptive − Native |
|---:|---|---:|---:|---:|---:|
| 8192 | copy（step 5 含一次 RESUME） | 4 | 342.472 | 325.192 | +17.280 (+5.31%) |
| 16 | UVA/PARK | 14 | 81.994 | 108.046 | −26.053 (−24.11%) |
| 47 | 保持 copy | 2 | 147.347 | 133.341 | +14.005 (+10.50%) |
| 32 | 保持 copy | 20 | 122.497 | 109.671 | +12.826 (+11.69%) |

两个 8192-token step 应分开解释：

| Step | 含义 | Adaptive | Native copy | 差值 |
|---:|---|---:|---:|---:|
| 1 | 初始 Prefill，已在 copy topology | 351.532 | 337.440 | +14.092 |
| 5 | 后续请求 Prefill，先 RESUME 再 copy | 333.412 | 312.945 | +20.467 |

step 5 的 Adaptive RESUME 实测约 7.7 ms；因此它比 step 1 多出的机制差距
符合“约 13 ms copy 数据面差距 + 约 7.7 ms topology 切换”。

总体结果：

| 机制 | req/s | TTFT ms | TPOT ms | E2E ms |
|---|---:|---:|---:|---:|
| Adaptive PARK/RESUME | 10.966 | 348.36 | 122.16 | 2180.82 |
| Native copy-on-forward | 10.928 | 330.33 | 118.06 | 2101.28 |

吞吐基本相同并不表示每个阶段相同：Adaptive 的 copy step 较慢，但 16-token
UVA step 无需每轮搬 4.798 GB，平均快约 26 ms，两者在整条 trace 中抵消。

## Nsight 数据面归因（诊断关闭后）

GPU H2D 使用中位数以避开一个 profiler 扰动样本；kernel 为均值。

| token | Adaptive H2D | Native H2D | H2D差值 | Adaptive kernel | Native kernel |
|---:|---:|---:|---:|---:|---:|
| 8192 | 96.36 ms / 4.948 GB | 85.01 ms / 4.798 GB | +11.35 ms | 222.87 ms | 221.74 ms |
| 47 | 96.31 ms / 4.948 GB | 84.99 ms / 4.798 GB | +11.33 ms | 24.67 ms | 24.22 ms |
| 32 | 96.32 ms / 4.948 GB | 84.97 ms / 4.798 GB | +11.35 ms | 21.33 ms | 20.70 ms |
| 16 | 约 0（UVA） | 84.96 ms / 4.798 GB | −84.96 ms | 89.59 ms | 18.29 ms |

稳定 32-token copy step 中：

- Adaptive 有 52 次选中权重 H2D，Native 有 9 次；加上框架小 copy 后，
  Nsight 分别看到约 59 和 16 次 H2D。
- Adaptive 有效 H2D 约 51.4 GB/s，Native 约 56.5 GB/s。
- 4.948/4.798 GB 的 3.13% 字节差只能解释约 2.7 ms；按 Native 带宽归一，
  仍有约 8.7 ms 是切片碎片化造成的传输效率损失。
- manifest 中每个 copy step 的 CPU copy 提交约为 Adaptive 1.95 ms、
  Native 0.36 ms，额外约 1.6 ms；kernel 差仅约 0.6 ms。

因此正常 copy-mode 的约 13 ms 单步差距主要是碎片化 H2D与提交开销，
而非 forward kernel。16-token 时 Adaptive 选择 UVA，虽然远程访问令 MoE
kernel 变慢，但省去约 85 ms 整体 H2D，净结果仍快约 26 ms。

## 原 260 ms Prefill 差距为什么出现

在诊断未关闭的两进程复跑中，8192-token step 为 Adaptive 586.085 ms、
Native 325.977 ms，差 260.108 ms。Adaptive 每个请求窗口执行 8 次
`cpu_offload_uvm_prefill_memory_trace`（每个 Prefill 四次），两进程分别累计
498.737 和 499.127 ms，即每个 Prefill 约 249.5 ms。

Nsight callchain 将这些调用定位为：

`ioctl → uvm_driver_batch_execute_internal → uvm_allocator.so`

每个 Prefill 约 112–113 次 ioctl。关闭诊断后，这些 Prefill 驻留查询降为0，
8192-token 差距从 260.108 ms 降至 17.280 ms，移除了 93.4%的原差距。
源码上，`deepseek_v2.py` 直接用
`VLLM_UVM_FUSED_MOE_POST_WORKSPACE_RESIDENCY_AUDIT` 控制第1层内存 trace，
因此原本意在诊断的开关污染了真实性能路径。

## Artifacts

- 诊断开启、两进程逐步计时：
  `artifacts/pressure-p512-c32-park16-step-timing-20260915-attempt-01/`
- 诊断开启、Nsight：
  `artifacts/pressure-p512-c32-park16-nsys-20260915-attempt-01/`
- 诊断关闭、两进程正式结果：
  `artifacts/pressure-p512-c32-park16-audit-off-step-timing-20260915-attempt-01/`
- 诊断关闭、Nsight：
  `artifacts/pressure-p512-c32-park16-audit-off-nsys-20260915-attempt-01/`

