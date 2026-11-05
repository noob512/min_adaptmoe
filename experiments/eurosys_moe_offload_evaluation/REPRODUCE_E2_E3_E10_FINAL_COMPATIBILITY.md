# 历史兼容性入口：当前协议索引

日期：2026-08-31（Asia/Shanghai）

本文档原先混合了不同模块二进制、C4/C8、B8、cold63 和早期 PARK/RESUME 结果，不能再
作为当前复现说明。

请统一使用以下文档：

```text
复现  REPRODUCE_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md
结果  RESULTS_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md
```

当前协议只接纳同一 custom UVM 二进制
`c8fe77c8ecc87f6d724d94ddf6535c934150d4e12bec384bd44562097aec043a` 下采集的最新
ours 数据。E2 为 C1/C16，E3 为 B1/B16，E10 为 KV 2→4→6 GiB；ours 必须达到全覆盖
双页表且 CPU range-map 回退为0。

pinned-UVA 基线只从已冻结的 attempt-01 读取，不重跑，也不与当前 ours artifact 合并成
同一轮。历史 artifact 保留用于根因分析，不得覆盖当前结果。
