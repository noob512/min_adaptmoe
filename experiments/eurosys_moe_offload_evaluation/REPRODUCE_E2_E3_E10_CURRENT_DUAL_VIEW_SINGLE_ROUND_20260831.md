# 历史复现入口：已由 C1/C16 全覆盖双页表协议取代

日期：2026-08-31（Asia/Shanghai）

本文档原先描述 C1/C4/C8/C16、B1/B8 和部分非双页表路径，已经不再代表当前实验。
请勿按旧命令补点，也不要将旧 artifact 与当前结果拼接。

当前唯一权威复现文档：

```text
REPRODUCE_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md
```

当前协议为：

- E2 只运行 C1/C16 的 `ours`，要求全覆盖 Prefill/Decode 双页表；
- E3 只运行 B1/B16 的 D/ours，小场景 cold10、中等场景 cold38、RAG/C16 cold89；
- E10 运行 cold70→cold80→cold89、KV 2→4→6 GiB 压力切档；
- 每个配置一次冷进程运行，不静默重试；
- 不重复 pinned-UVA、copy-on-forward，MoE-Infinity 跳过；
- 任一子进程、receipt 或 verifier 失败都必须 fail-fast 并报告。

权威结果：

```text
RESULTS_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md
```

历史目录仍可用于故障追溯，但不得用于当前性能结论。
