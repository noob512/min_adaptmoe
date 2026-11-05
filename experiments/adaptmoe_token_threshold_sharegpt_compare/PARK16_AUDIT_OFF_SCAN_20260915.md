# PARK=16、驻留诊断关闭后的 ShareGPT 三臂复跑

日期：2026-09-15。

## 结论

固定 `PARK=16`、`RESUME=128` 并关闭会遍历热区、发起 UVM 查询的
Prefill/运行时驻留诊断后，三档结果如下（单位 `req/s`，每个数为两个独立
进程的均值）：

| 负载 | Adaptive | Native UVA | Native copy | Adaptive / UVA | Adaptive / copy |
|---|---:|---:|---:|---:|---:|
| 128×8 | 5.017 | 5.356 | 3.833 | -6.33% | +30.88% |
| 512×16 | 8.681 | 4.457 | 6.838 | +94.78% | +26.95% |
| 512×32 | 10.895 | 4.840 | 10.934 | +125.09% | -0.35% |

因此，关闭诊断后原先 `512×32, PARK=16` 的 9.833 vs 11.124 req/s
大差距不再存在；Adaptive 与 native copy 仅差 0.35%，在本轮两进程精度下
应视为整体持平。原差距主要来自 8192-token Prefill 中每次约 249.5 ms 的
全量驻留扫描，而不是正常 PARK/RESUME 或 copy 数据面。

## 延迟指标

| 负载 / 机制 | TTFT ms | TPOT ms | E2E ms |
|---|---:|---:|---:|
| 128×8 Adaptive | 158.25 | 73.31 | 1257.92 |
| 128×8 UVA | 186.50 | 66.17 | 1179.04 |
| 128×8 copy | 140.30 | 101.40 | 1661.27 |
| 512×16 Adaptive | 233.97 | 82.16 | 1466.38 |
| 512×16 UVA | 1038.71 | 119.42 | 2830.02 |
| 512×16 copy | 217.66 | 109.47 | 1859.71 |
| 512×32 Adaptive | 348.51 | 122.08 | 2179.78 |
| 512×32 UVA | 2216.15 | 178.87 | 4899.12 |
| 512×32 copy | 329.83 | 118.20 | 2102.83 |

## 为什么三档的相对排名不同

- `128×8` 的 Decode 压力很低。Adaptive 只有两个 Prefill 使用 copy，其余
  小 Decode step 使用 UVA；切换和别名机制没有足够负载来摊薄，因此 UVA
  吞吐最高。Native copy 每轮搬 4.798 GB，明显不合算。
- `512×16` 中，Adaptive 只在两个大 Prefill 搬运，16-token Decode 在
  `PARK=16` 时进入 UVA，总 H2D 仅 9.896 GB；native copy 累计 95.966 GB，
  native UVA 又被较高并发的随机远程读取拖慢，因此 Adaptive 最快。
- `512×32` 的轨迹为 `8192,16,16,16,8192,47,32...,16...,1`。
  47/32 token 位于 16–128 的滞回区并保持 copy，Adaptive 累计 H2D
  64.327 GB；native copy 为 100.764 GB。Adaptive 的低搬运量被其 52 个
  layer-local binding/286 个 expert slice 的碎片化 H2D及切换成本抵消，
  最终两者整体持平。

## 协议与有效性

- DeepSeek-V2-Lite-Chat BF16，TRITON_MLA，vLLM v1 mixed continuous
  scheduler，eager，无 CUDA Graph。
- 三臂使用相同冻结 ShareGPT prompt、相同输出长度和调度/KV 配置。
- Adaptive 卸载 4,948,230,144 B；两个 native arm 卸载同一组
  4,798,283,776 B 参数。两者字节相差 3.13%，所以不是严格的同坐标微基准。
- 每档三臂各两个独立进程，执行顺序为 ABC/CBA，共 18 个进程；全部成功，
  所有跨臂输出 hash 一致，无 Xid/PTE 错误。
- 明确关闭 `VLLM_UVM_FUSED_MOE_POST_WORKSPACE_RESIDENCY_AUDIT`、
  `VLLM_UVM_FUSED_MOE_POST_WORKSPACE_AUDIT_CAPTURE_WARMUP`、
  `VLLM_UVM_RUNTIME_POST_MAP_RESIDENCY_AUDIT` 和
  `VLLM_UVM_RUNTIME_HOT_RESIDENCY_AUDIT`。保留轻量 copy trace 以核对 H2D
  字节；它不执行此前污染 Prefill 的 518 区间全量驻留扫描。
- 两个进程足以确认上述大效应和诊断污染是否消失，但不足以给出严格置信区间。

## Artifacts

- `artifacts/park16-audit-off-p128-c8-20260915-attempt-01/`
- `artifacts/park16-audit-off-p512-c16-20260915-attempt-01/`
- `artifacts/park16-audit-off-p512-c32-20260915-attempt-01/`

每个目录包含 `REPORT.md`、`verification.json`、逐进程命令、环境、原始结果和
内核日志。
