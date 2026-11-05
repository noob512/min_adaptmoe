# 两模型 C1/C16 六机制完整结果

本文档扩展 `RESULTS_E0_E14_CONCISE.md` 的 E2/E3 数据，复用已有合法结果并仅补测原矩阵缺口。所有延迟单位均为 ms；每个新增 cell 为一个冷进程样本。

指标口径：TTFT/TPOT 为请求级 P50；Prefill E2E 为完整 batch 从 EngineCore step 开始到全批 Prefill 完成并产生首 token 的墙钟时间；Decode tok/s 优先使用全批 Prefill 完成后直接累计的稳态 step。新测 Native UVA 两种 Graph 协议均使用该 phase counter；仅没有保存该 counter 的既有 DeepSeek AdaptMoE/KTransformers phase-batched 结果按 `C × 1000 / TPOT` 换算。E2E 为 batch 中最后一个请求完成的时间（Qwen 使用 cycle E2E）。`—` 表示源运行没有独立记录该指标，N/A/OOM 不填性能值。

## DeepSeek-V2-Lite BF16

请求为 Short、Ordinary、RAG、Long，共 4 × C1/C16 × 6 = 48 cells。按用户指定，Native UVA 不改为真正的 NONE/NONE；这里用新版 direct runner 重测 eager Prefill + Decode Graph 和双阶段 Graph，以补齐独立 Prefill E2E。KTransformers 已按与 AdaptMoE 相同的 cold2/cold6/cold20 档位重测，使用 7 个 CPU 专家线程和 Decode CUDA Graph；RAG/C16 继续使用 `kt_gpu_prefill_token_threshold=4096` 的 full-GPU Prefill fallback。

| 场景 | C | 机制 | 实际卸载 GiB | TTFT (ms) | TPOT (ms) | Prefill E2E (ms) | Decode tok/s | E2E (ms) | 状态 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Short | 1 | AdaptMoE 双阶段 Graph | 0.838 | 62.35 | 8.50 | 50.63 | 117.58 | 326.00 | ok（复用完整运行 + 新测 Prefill） |
| Short | 1 | Native UVA（eager Prefill + Decode Graph） | 0.896 | 67.11 | 15.51 | 67.11 | 64.52 | 547.88 | ok（direct runner 重测） |
| Short | 1 | Native Copy eager | 0.813 | 65.85 | 65.85 | 65.85 | 15.19 | 2,107.12 | ok（新测） |
| Short | 1 | KTransformers 7T + Decode Graph | 0.838 | 77.36 | 7.20 | 77.36 | 138.90 | 300.54 | ok（等量卸载新测 cold2） |
| Short | 1 | Native UVA 双阶段 Graph | 0.896 | 67.48 | 15.53 | 67.48 | 64.43 | 548.88 | ok（direct runner 重测） |
| Short | 1 | Native Copy 双阶段 Graph | 0.813 | 45.60 | 21.58 | 45.60 | 46.35 | 714.64 | ok（修正参数后重测） |
| Short | 16 | AdaptMoE 双阶段 Graph | 0.838 | 85.65 | 22.26 | 86.72 | 718.63 | 775.85 | ok（复用完整运行 + 新测 Prefill） |
| Short | 16 | Native UVA（eager Prefill + Decode Graph） | 0.896 | 368.26 | 33.52 | 368.26 | 477.66 | 1,407.25 | ok（direct runner 重测） |
| Short | 16 | Native Copy eager | 0.813 | 80.68 | 66.65 | 80.68 | 240.12 | 2,146.97 | ok（新测） |
| Short | 16 | KTransformers 7T + Decode Graph | 0.838 | 1,003.29 | 22.40 | 1,004.52 | 714.38 | 1,698.42 | ok（等量卸载新测 cold2） |
| Short | 16 | Native UVA 双阶段 Graph | 0.896 | 376.11 | 33.47 | 376.11 | 478.32 | 1,413.67 | ok（direct runner 重测） |
| Short | 16 | Native Copy 双阶段 Graph | 0.813 | 81.43 | 33.89 | 81.43 | 472.35 | 1,132.10 | ok（修正参数后重测） |
| Ordinary | 1 | AdaptMoE 双阶段 Graph | 0.838 | 64.60 | 7.98 | 59.17 | 125.27 | 1,078.44 | ok（复用完整运行 + 新测 Prefill） |
| Ordinary | 1 | Native UVA（eager Prefill + Decode Graph） | 0.896 | 136.39 | 15.69 | 136.39 | 63.77 | 2,129.17 | ok（direct runner 重测） |
| Ordinary | 1 | Native Copy eager | 0.813 | 60.21 | 64.36 | 60.21 | 15.54 | 8,233.99 | ok（新测） |
| Ordinary | 1 | KTransformers 7T + Decode Graph | 0.838 | 131.21 | 8.09 | 131.21 | 123.56 | 1,159.04 | ok（等量卸载新测 cold2） |
| Ordinary | 1 | Native UVA 双阶段 Graph | 0.896 | 138.98 | 15.79 | 138.98 | 63.38 | 2,143.93 | ok（direct runner 重测） |
| Ordinary | 1 | Native Copy 双阶段 Graph | 0.813 | 54.07 | 21.75 | 54.07 | 46.00 | 2,816.00 | ok（修正参数后重测） |
| Ordinary | 16 | AdaptMoE 双阶段 Graph | 2.514 | 280.94 | 40.74 | 279.62 | 392.70 | 5,455.32 | ok（复用完整运行 + 新测 Prefill） |
| Ordinary | 16 | Native UVA（eager Prefill + Decode Graph） | 2.330 | 2,292.06 | 56.98 | 2,292.06 | 280.92 | 9,528.08 | ok（direct runner 重测） |
| Ordinary | 16 | Native Copy eager | 2.406 | 265.93 | 69.34 | 265.93 | 230.81 | 9,072.34 | ok（新测） |
| Ordinary | 16 | KTransformers 7T + Decode Graph | 2.514 | 6,542.67 | 44.62 | 6,543.66 | 358.58 | 12,209.98 | ok（等量卸载新测 cold6） |
| Ordinary | 16 | Native UVA 双阶段 Graph | 2.330 | 2,278.42 | 57.05 | 2,278.42 | 280.54 | 9,524.22 | ok（direct runner 重测） |
| Ordinary | 16 | Native Copy 双阶段 Graph | 2.406 | 266.91 | 64.67 | 266.91 | 247.50 | 8,479.49 | ok（修正参数后重测） |
| RAG | 1 | AdaptMoE 双阶段 Graph | 0.838 | 370.75 | 12.09 | 350.99 | 82.75 | 1,905.58 | ok（复用完整运行 + 新测 Prefill） |
| RAG | 1 | Native UVA（eager Prefill + Decode Graph） | 0.896 | 1,443.51 | 19.57 | 1,443.51 | 51.13 | 3,928.60 | ok（direct runner 重测） |
| RAG | 1 | Native Copy eager | 0.813 | 399.28 | 63.96 | 399.28 | 15.64 | 8,522.24 | ok（新测） |
| RAG | 1 | KTransformers 7T + Decode Graph | 0.838 | 2,419.43 | 10.66 | 2,419.43 | 93.81 | 3,773.28 | ok（等量卸载新测 cold2） |
| RAG | 1 | Native UVA 双阶段 Graph | 0.896 | 1,448.61 | 19.52 | 1,448.61 | 51.26 | 3,927.27 | ok（direct runner 重测） |
| RAG | 1 | Native Copy 双阶段 Graph | 1.844 | 418.28 | 44.82 | 418.28 | 22.31 | 6,110.83 | ok（增加卸载量后重测） |
| RAG | 16 | AdaptMoE 双阶段 Graph | 8.379 | 2,795.17 | 150.92 | 4,817.28 | 106.02 | 21,962.26 | ok（复用完整运行 + 新测 Prefill） |
| RAG | 16 | Native UVA（eager Prefill + Decode Graph） | 8.522 | 58,521.37 | 518.04 | 104,051.10 | 100.30 | 124,312.60 | ok（direct runner 重测） |
| RAG | 16 | Native Copy eager | 8.250 | 3,072.39 | 199.43 | 5,462.83 | 88.60 | 28,400.04 | ok（新测） |
| RAG | 16 | KTransformers 7T + Decode Graph | 8.379 | 5,896.03 | 160.98 | 8,710.50 | 99.39 | 26,341.82 | ok（等量卸载新测 cold20） |
| RAG | 16 | Native UVA 双阶段 Graph | 8.522 | 58,555.52 | 518.11 | 104,089.84 | 100.28 | 124,356.09 | ok（direct runner 重测） |
| RAG | 16 | Native Copy 双阶段 Graph | 8.250 | 3,070.77 | 198.43 | 5,459.84 | 89.09 | 28,271.33 | ok（修正参数后重测） |
| Long | 1 | AdaptMoE 双阶段 Graph | 0.838 | 74.51 | 7.98 | 59.31 | 125.27 | 4,153.72 | ok（复用完整运行 + 新测 Prefill） |
| Long | 1 | Native UVA（eager Prefill + Decode Graph） | 0.896 | 138.44 | 15.82 | 138.44 | 63.23 | 8,224.53 | ok（direct runner 重测） |
| Long | 1 | Native Copy eager | 0.813 | 59.87 | 57.18 | 59.87 | 17.49 | 29,281.28 | ok（新测） |
| Long | 1 | KTransformers 7T + Decode Graph | 0.838 | 216.92 | 9.39 | 216.92 | 106.48 | 5,015.81 | ok（等量卸载新测 cold2） |
| Long | 1 | Native UVA 双阶段 Graph | 0.896 | 139.22 | 15.82 | 139.22 | 63.27 | 8,220.84 | ok（direct runner 重测） |
| Long | 1 | Native Copy 双阶段 Graph | 0.813 | 54.16 | 21.82 | 54.16 | 45.84 | 11,205.14 | ok（修正参数后重测） |
| Long | 16 | AdaptMoE 双阶段 Graph | 2.514 | 281.54 | 43.29 | 274.34 | 369.58 | 22,403.69 | ok（复用完整运行 + 新测 Prefill） |
| Long | 16 | Native UVA（eager Prefill + Decode Graph） | 2.330 | 2,270.56 | 57.02 | 2,270.56 | 280.67 | 31,409.70 | ok（direct runner 重测） |
| Long | 16 | Native Copy eager | 2.406 | 266.60 | 67.38 | 266.60 | 237.52 | 34,698.22 | ok（新测） |
| Long | 16 | KTransformers 7T + Decode Graph | 2.514 | 7,135.71 | 45.67 | 7,136.91 | 350.30 | 30,476.23 | ok（等量卸载新测 cold6） |
| Long | 16 | Native UVA 双阶段 Graph | 2.330 | 2,273.94 | 56.95 | 2,273.94 | 281.03 | 31,375.42 | ok（direct runner 重测） |
| Long | 16 | Native Copy 双阶段 Graph | 2.406 | 266.97 | 64.45 | 266.97 | 248.33 | 33,199.69 | ok（修正参数后重测） |

DeepSeek Native Copy 的安全 exact-name 三档实际卸载量通常为 0.812744/2.406/8.250 GiB。low 档是当前 Copy wrapper 可执行的最近组合，相对 Native UVA 0.896 GiB 差 10.24%；尝试补入 shared expert、attention 或 router 小张量均因 DeepSeek 自定义权重加载后参数回到 GPU 而失败。RAG/C1 双阶段 Graph 为满足 capture 峰值容量，额外卸载异层专家张量，实际卸载量提高至 1.843994 GiB；mid/high 档相对 2.330/8.522 GiB 分别差 3.3%/3.3%。

## Qwen3-30B-A3B FP8

请求为 Short、Ordinary、Medium、RAG、Long，共 5 × C1/C16 × 6 = 60 cells。AdaptMoE、KTransformers 与两种双 Graph 路径复用已有合法结果；Native UVA/Copy eager 为本轮 TRITON_ATTN、C1/C16、Graph NONE/NONE 新测。按当前要求，本轮未重测 Qwen KTransformers，其 cold10/cold38/cold89 行保持不变，尚不是与 AdaptMoE 完全相同的 cold 档位。

| 场景 | C | 机制 | 实际卸载 GiB | TTFT (ms) | TPOT (ms) | Prefill E2E (ms) | Decode tok/s | E2E (ms) | 状态 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| Short | 1 | AdaptMoE 双阶段 Graph | 2.320 | 72.58 | 12.05 | 71.93 | 105.24 | 446.70 | ok（已有 cold11） |
| Short | 1 | Native UVA eager | 2.322 | 68.42 | 71.26 | 68.19 | 14.06 | 2,277.76 | ok（新测 NONE/NONE） |
| Short | 1 | Native Copy eager | 2.322 | 71.42 | 74.17 | 71.17 | 13.50 | 2,370.96 | ok（新测 NONE/NONE） |
| Short | 1 | KTransformers 7T + Decode Graph | 2.109 | 118.74 | 17.34 | 109.75 | 61.04 | 656.59 | ok（已有最佳 cold10） |
| Short | 1 | Native UVA 双阶段 Graph | 2.322 | 61.19 | 12.34 | 60.97 | 82.47 | 443.86 | ok（已有） |
| Short | 1 | Native Copy 双阶段 Graph | 2.322 | 64.39 | 52.40 | 64.11 | 19.13 | 1,689.10 | ok（已有） |
| Short | 16 | AdaptMoE 双阶段 Graph | 2.320 | 108.49 | 20.03 | 107.78 | 896.05 | 731.25 | ok（已有 cold11） |
| Short | 16 | Native UVA eager | 2.322 | 406.88 | 73.24 | 406.13 | 218.30 | 2,678.79 | ok（新测 NONE/NONE） |
| Short | 16 | Native Copy eager | 2.322 | 94.33 | 74.35 | 93.48 | 217.13 | 2,400.71 | ok（新测 NONE/NONE） |
| Short | 16 | KTransformers 7T + Decode Graph | 2.109 | 1,475.05 | 28.10 | 246.35 | 581.91 | 2,348.34 | ok（已有最佳 cold10） |
| Short | 16 | Native UVA 双阶段 Graph | 2.322 | 417.95 | 41.71 | 417.16 | 379.07 | 1,712.35 | ok（已有） |
| Short | 16 | Native Copy 双阶段 Graph | 2.322 | 94.10 | 60.68 | 93.38 | 263.60 | 1,976.66 | ok（已有） |
| Ordinary | 1 | AdaptMoE 双阶段 Graph | 2.320 | 78.16 | 9.70 | 77.64 | 110.10 | 1,311.02 | ok（已有 cold11） |
| Ordinary | 1 | Native UVA eager | 2.322 | 121.07 | 67.71 | 120.83 | 14.36 | 8,719.86 | ok（新测 NONE/NONE） |
| Ordinary | 1 | Native Copy eager | 2.322 | 70.75 | 60.55 | 70.49 | 17.66 | 7,761.25 | ok（新测 NONE/NONE） |
| Ordinary | 1 | KTransformers 7T + Decode Graph | 2.109 | 284.67 | 19.11 | 142.45 | 52.78 | 2,712.41 | ok（已有最佳 cold10） |
| Ordinary | 1 | Native UVA 双阶段 Graph | 2.322 | 121.14 | 12.33 | 120.82 | 81.76 | 1,687.11 | ok（已有） |
| Ordinary | 1 | Native Copy 双阶段 Graph | 2.322 | 70.15 | 52.49 | 69.94 | 19.08 | 6,736.05 | ok（已有） |
| Ordinary | 16 | AdaptMoE 双阶段 Graph | 8.227 | 372.33 | 56.66 | 371.49 | 286.87 | 7,570.46 | ok（已有 cold39） |
| Ordinary | 16 | Native UVA eager | 7.942 | 3,705.07 | 111.30 | 3,704.42 | 143.53 | 17,841.77 | ok（新测 NONE/NONE） |
| Ordinary | 16 | Native Copy eager | 7.942 | 331.66 | 169.43 | 331.00 | 94.44 | 21,850.64 | ok（新测 NONE/NONE） |
| Ordinary | 16 | KTransformers 7T + Decode Graph | 8.016 | 5,394.00 | 118.24 | 5,226.31 | 134.51 | 20,413.97 | ok（已有最佳 cold38） |
| Ordinary | 16 | Native UVA 双阶段 Graph | 7.942 | 3,701.24 | 110.82 | 3,700.62 | 144.21 | 17,777.27 | ok（已有） |
| Ordinary | 16 | Native Copy 双阶段 Graph | 7.942 | 335.07 | 169.62 | 334.21 | 94.42 | 21,878.26 | ok（已有） |
| Medium | 1 | AdaptMoE 双阶段 Graph | 2.320 | 117.12 | 14.21 | 116.66 | 71.80 | 1,921.92 | ok（已有 cold11） |
| Medium | 1 | Native UVA eager | 2.322 | 435.89 | 67.98 | 435.62 | 14.27 | 9,069.50 | ok（新测 NONE/NONE） |
| Medium | 1 | Native Copy eager | 2.322 | 108.13 | 69.09 | 107.85 | 14.00 | 8,882.22 | ok（新测 NONE/NONE） |
| Medium | 1 | KTransformers 7T + Decode Graph | 2.109 | 7,390.93 | 30.23 | 1,045.06 | 33.32 | 11,231.03 | ok（已有最佳 cold10） |
| Medium | 1 | Native UVA 双阶段 Graph | 2.322 | 451.75 | 12.61 | 451.50 | 79.93 | 2,053.30 | ok（已有） |
| Medium | 1 | Native Copy 双阶段 Graph | 2.322 | 108.32 | 52.75 | 108.06 | 18.99 | 6,808.37 | ok（已有） |
| Medium | 16 | AdaptMoE 双阶段 Graph | 8.227 | 1,144.10 | 58.73 | 1,143.10 | 276.87 | 8,604.90 | ok（已有 cold39） |
| Medium | 16 | Native UVA eager | 7.942 | 14,701.46 | 108.50 | 14,700.67 | 146.87 | 28,481.90 | ok（新测 NONE/NONE） |
| Medium | 16 | Native Copy eager | 7.942 | 1,081.11 | 170.99 | 1,080.24 | 93.57 | 22,798.84 | ok（新测 NONE/NONE） |
| Medium | 16 | KTransformers 7T + Decode Graph | 8.016 | 3,339.90 | 131.91 | 3,177.86 | 121.23 | 20,093.51 | ok（已有最佳 cold38） |
| Medium | 16 | Native UVA 双阶段 Graph | 7.942 | 14,662.69 | 108.04 | 14,661.36 | 147.44 | 28,385.61 | ok（已有） |
| Medium | 16 | Native Copy 双阶段 Graph | 7.942 | 1,104.14 | 170.90 | 1,103.21 | 93.66 | 22,810.14 | ok（已有） |
| RAG | 1 | AdaptMoE 双阶段 Graph | 8.227 | 593.55 | 17.26 | 593.01 | 64.60 | 2,786.45 | ok（已有 cold39） |
| RAG | 1 | Native UVA eager | 7.942 | 4,061.69 | 55.10 | 4,061.37 | 18.19 | 11,059.48 | ok（新测 NONE/NONE） |
| RAG | 1 | Native Copy eager | 7.942 | 562.86 | 160.93 | 562.53 | 6.21 | 21,001.17 | ok（新测 NONE/NONE） |
| RAG | 1 | KTransformers 7T + Decode Graph | 8.016 | 3,913.21 | 21.88 | 3,886.95 | 46.07 | 6,692.05 | ok（已有最佳 cold38） |
| RAG | 1 | Native UVA 双阶段 Graph | 7.942 | 3,918.03 | 24.02 | 3,917.74 | 41.85 | 6,968.59 | ok（已有） |
| RAG | 1 | Native Copy 双阶段 Graph | 7.942 | 572.21 | 161.03 | 571.80 | 6.21 | 21,023.05 | ok（已有） |
| RAG | 16 | AdaptMoE 双阶段 Graph | 18.984 | 7,397.25 | 164.30 | 7,394.70 | 98.06 | 28,268.10 | ok（已有 cold90） |
| RAG | 16 | Native UVA eager | 20.324 | 143,692.87 | 240.27 | 143,691.19 | 66.26 | 174,210.65 | ok（新测 NONE/NONE） |
| RAG | 16 | Native Copy eager | 20.324 | 7,134.20 | 412.92 | 7,132.54 | 38.75 | 59,577.68 | ok（新测 NONE/NONE） |
| RAG | 16 | KTransformers 7T + Decode Graph | 18.773 | 10,340.32 | 330.92 | 10,135.03 | 48.47 | 52,370.06 | ok（已有最佳 cold89） |
| RAG | 16 | Native UVA 双阶段 Graph | 20.324 | 140,317.38 | 239.95 | 140,315.46 | 66.41 | 170,794.72 | ok（已有） |
| RAG | 16 | Native Copy 双阶段 Graph | 21.298 | 7,376.60 | 430.98 | 7,375.45 | 37.12 | 62,113.55 | ok（增加卸载量后重测） |
| Long | 1 | AdaptMoE 双阶段 Graph | 2.320 | 79.11 | 10.50 | 78.57 | 98.92 | 5,445.50 | ok（已有 cold11） |
| Long | 1 | Native UVA eager | 2.322 | 118.50 | 55.85 | 118.28 | 17.97 | 28,659.51 | ok（新测 NONE/NONE） |
| Long | 1 | Native Copy eager | 2.322 | 71.17 | 56.85 | 70.95 | 17.61 | 29,121.09 | ok（新测 NONE/NONE） |
| Long | 1 | KTransformers 7T + Decode Graph | 2.109 | 583.40 | 20.17 | 166.47 | 49.62 | 10,893.06 | ok（已有最佳 cold10） |
| Long | 1 | Native UVA 双阶段 Graph | 2.322 | 122.35 | 12.40 | 122.14 | 81.48 | 6,457.72 | ok（已有） |
| Long | 1 | Native Copy 双阶段 Graph | 2.322 | 71.07 | 52.47 | 70.79 | 19.09 | 26,883.10 | ok（已有） |
| Long | 16 | AdaptMoE 双阶段 Graph | 8.227 | 372.07 | 61.59 | 371.16 | 257.44 | 31,847.70 | ok（已有 cold39） |
| Long | 16 | Native UVA eager | 7.942 | 3,751.63 | 104.67 | 3,750.95 | 153.08 | 57,241.82 | ok（新测 NONE/NONE） |
| Long | 16 | Native Copy eager | 7.942 | 332.23 | 169.11 | 331.56 | 94.61 | 86,749.61 | ok（新测 NONE/NONE） |
| Long | 16 | KTransformers 7T + Decode Graph | 8.016 | 26,958.86 | 163.01 | 6,327.14 | 97.72 | 110,262.31 | ok（已有最佳 cold38） |
| Long | 16 | Native UVA 双阶段 Graph | 7.942 | 3,644.16 | 104.19 | 3,643.19 | 153.88 | 56,887.80 | ok（已有） |
| Long | 16 | Native Copy 双阶段 Graph | 7.942 | 335.43 | 169.08 | 334.40 | 94.69 | 86,734.68 | ok（已有） |

## 覆盖与失败

本轮 Qwen Native UVA/Copy eager 20/20 点完成。DeepSeek 的 AdaptMoE 独立 Prefill以及等量卸载 KTransformers cold2/cold6/cold20 均完成 8/8；Native Copy eager、两种 Native UVA Graph 协议也均为 8/8。修正 `STATIC_BANK` 参数并采用当前 MLA Graph 修复后，DeepSeek Native Copy 双阶段 Graph 完成 8/8：RAG/C1 将实际卸载量从 0.812744 GiB 提高至 1.843994 GiB 后通过。Qwen Native Copy 双阶段 Graph 的 RAG/C16 将实际卸载量从 20.324 GiB 提高至 21.298 GiB 后也通过。两个补测点均确认 Prefill/Decode Full Graph runtime dispatch、完整输出和冷进程退出后 GPU 显存恢复为 0 MiB。

此前 Native UVA 双阶段 Graph 的 N/A 与 128-token 分块产生的全零输出均已废弃。根因有两处：MLA 纯 Prefill metadata 在 capture 时通过减法生成临时 GPU tensor，capture 返回后其 storage 被释放，但 GraphExec 仍保留旧地址；同时 Prefill Graph key 只区分总 token 数，未区分请求布局，使 16×8 capture 可能 replay 到 1×128。修复后纯 Prefill 使用 runner 持久输入 buffer 的 view，并按正式 step 的请求数 capture（RAG/C16 为 2×8192，其余 C16 为 16-request 布局）。8 对同行 direct-runner 结果均确认相应 Graph replay且完整输出一致；按纯 Decode phase counter，双阶段 Graph 相对 eager Prefill + Decode Graph 的吞吐差异绝对值不超过 0.61%。

总矩阵为 108 cells；有效完成 108，无保留的 N/A/OOM 点。此前两个 OOM 均为固定卸载量下的 Graph capture 峰值容量不足，而非 Native Copy 的全局卸载上限。

## 原始结果索引

| 模型 | 场景 | C | 机制 | 原始结果 | 说明 |
|---|---|---:|---|---|---|
| DeepSeek-V2-Lite BF16 | Short | 1 | AdaptMoE 双阶段 Graph | `artifacts/e2-near-equal-offload-plus-one-expert-20260903-attempt-01/c1/cases/ours/short_dialogue/cold_02/short_dialogue_c1/client.json; artifacts/full-matrix-deepseek-gaps-20260911-attempt-01/adaptmoe_prefill_c1/cases/ours/short_dialogue/cold_02/result.json` | Decode tok/s 由同步 phase-batched TPOT 计算 |
| DeepSeek-V2-Lite BF16 | Short | 1 | Native UVA（eager Prefill + Decode Graph） | `artifacts/e2-native-uva-direct-phase-metrics-20260911-attempt-01-full-decode-only-c1/cases/native_pinned_uva/short_dialogue/offload_0p4189453125g/result.json` | FULL_DECODE_ONLY：Prefill eager、Decode FULL Graph replay |
| DeepSeek-V2-Lite BF16 | Short | 1 | Native Copy eager | `artifacts/full-matrix-deepseek-native-copy-eager-c1-20260911-attempt-04/cases/native_copy_on_forward/short_dialogue/offload_exact_names/result.json` | 低档为最近可执行 exact-name 组合，0.812744 GiB；相对 0.896 GiB 差 10.24% |
| DeepSeek-V2-Lite BF16 | Short | 1 | KTransformers 7T + Decode Graph | `artifacts/e2-ktransformers-equal-offload-c1-20260911-attempt-02/cases/ktransformers/short_dialogue/cold_02/result.json` | 与 AdaptMoE 使用相同 cold 档位；Prefill E2E 为全批请求产生首 token 的时刻；Decode tok/s 由同步 TPOT 计算 |
| DeepSeek-V2-Lite BF16 | Short | 1 | Native UVA 双阶段 Graph | `artifacts/e2-native-uva-direct-phase-metrics-20260911-attempt-01-full-separate-c1/cases/native_pinned_uva/short_dialogue/offload_0p4189453125g/result.json` | Prefill/Decode 均为 FULL Graph replay；capture 使用与正式请求一致的 1-request 布局；Decode 相对 Decode-only 基线 -0.13%；完整输出一致 |
| DeepSeek-V2-Lite BF16 | Short | 1 | Native Copy 双阶段 Graph | `artifacts/e2-native-copy-full-separate-direct-c1-20260911-attempt-02/cases/native_copy_on_forward/short_dialogue/offload_exact_names/result.json` | — |
| DeepSeek-V2-Lite BF16 | Short | 16 | AdaptMoE 双阶段 Graph | `artifacts/e2-near-equal-offload-plus-one-expert-20260903-attempt-01/c16/cases/ours/short_dialogue/cold_02/result.json; artifacts/full-matrix-deepseek-gaps-20260911-attempt-01/adaptmoe_prefill_c16/cases/ours/short_dialogue/cold_02/result.json` | Decode tok/s 由同步 phase-batched TPOT 计算 |
| DeepSeek-V2-Lite BF16 | Short | 16 | Native UVA（eager Prefill + Decode Graph） | `artifacts/e2-native-uva-direct-phase-metrics-20260911-attempt-01-full-decode-only-c16/cases/native_pinned_uva/short_dialogue/offload_0p4189453125g/result.json` | FULL_DECODE_ONLY：Prefill eager、Decode FULL Graph replay |
| DeepSeek-V2-Lite BF16 | Short | 16 | Native Copy eager | `artifacts/full-matrix-deepseek-native-copy-eager-short-c16-20260911-attempt-04/cases/native_copy_on_forward/short_dialogue/offload_exact_names/result.json` | 低档为最近可执行 exact-name 组合，0.812744 GiB；相对 0.896 GiB 差 10.24% |
| DeepSeek-V2-Lite BF16 | Short | 16 | KTransformers 7T + Decode Graph | `artifacts/e2-ktransformers-equal-offload-c16-20260911-attempt-01/cases/ktransformers/short_dialogue/cold_02/result.json` | 与 AdaptMoE 使用相同 cold 档位；Prefill E2E 为全批请求产生首 token 的时刻；Decode tok/s 由同步 TPOT 计算 |
| DeepSeek-V2-Lite BF16 | Short | 16 | Native UVA 双阶段 Graph | `artifacts/e2-native-uva-full-separate-fixed-probe-20260911-short-c16/cases/native_pinned_uva/short_dialogue/offload_0p4189453125g/result.json` | Prefill/Decode 均为 FULL Graph replay；capture 使用与正式请求一致的 16-request 布局；Decode 相对 Decode-only 基线 +0.14%；完整输出一致 |
| DeepSeek-V2-Lite BF16 | Short | 16 | Native Copy 双阶段 Graph | `artifacts/e2-native-copy-full-separate-direct-c16-20260911-attempt-01/cases/native_copy_on_forward/short_dialogue/offload_exact_names/result.json` | — |
| DeepSeek-V2-Lite BF16 | Ordinary | 1 | AdaptMoE 双阶段 Graph | `artifacts/e2-near-equal-offload-plus-one-expert-20260903-attempt-01/c1/cases/ours/ordinary_chat/cold_02/ordinary_chat_c1/client.json; artifacts/full-matrix-deepseek-gaps-20260911-attempt-01/adaptmoe_prefill_c1/cases/ours/ordinary_chat/cold_02/result.json` | Decode tok/s 由同步 phase-batched TPOT 计算 |
| DeepSeek-V2-Lite BF16 | Ordinary | 1 | Native UVA（eager Prefill + Decode Graph） | `artifacts/e2-native-uva-direct-phase-metrics-20260911-attempt-01-full-decode-only-c1/cases/native_pinned_uva/ordinary_chat/offload_0p4189453125g/result.json` | FULL_DECODE_ONLY：Prefill eager、Decode FULL Graph replay |
| DeepSeek-V2-Lite BF16 | Ordinary | 1 | Native Copy eager | `artifacts/full-matrix-deepseek-native-copy-eager-c1-20260911-attempt-04/cases/native_copy_on_forward/ordinary_chat/offload_exact_names/result.json` | 低档为最近可执行 exact-name 组合，0.812744 GiB；相对 0.896 GiB 差 10.24% |
| DeepSeek-V2-Lite BF16 | Ordinary | 1 | KTransformers 7T + Decode Graph | `artifacts/e2-ktransformers-equal-offload-c1-20260911-attempt-02/cases/ktransformers/ordinary_chat/cold_02/result.json` | 与 AdaptMoE 使用相同 cold 档位；Prefill E2E 为全批请求产生首 token 的时刻；Decode tok/s 由同步 TPOT 计算 |
| DeepSeek-V2-Lite BF16 | Ordinary | 1 | Native UVA 双阶段 Graph | `artifacts/e2-native-uva-direct-phase-metrics-20260911-attempt-01-full-separate-c1/cases/native_pinned_uva/ordinary_chat/offload_0p4189453125g/result.json` | Prefill/Decode 均为 FULL Graph replay；capture 使用与正式请求一致的 1-request 布局；Decode 相对 Decode-only 基线 -0.61%；完整输出一致 |
| DeepSeek-V2-Lite BF16 | Ordinary | 1 | Native Copy 双阶段 Graph | `artifacts/e2-native-copy-full-separate-direct-c1-20260911-attempt-02/cases/native_copy_on_forward/ordinary_chat/offload_exact_names/result.json` | — |
| DeepSeek-V2-Lite BF16 | Ordinary | 16 | AdaptMoE 双阶段 Graph | `artifacts/e2-near-equal-offload-plus-one-expert-20260903-attempt-01/c16/cases/ours/ordinary_chat/cold_06/result.json; artifacts/e2-adaptmoe-prefill-ordinary-c16-fixed-20260911-attempt-01/cases/ours/ordinary_chat/cold_06/result.json` | Decode tok/s 由同步 phase-batched TPOT 计算 |
| DeepSeek-V2-Lite BF16 | Ordinary | 16 | Native UVA（eager Prefill + Decode Graph） | `artifacts/e2-native-uva-direct-phase-metrics-20260911-attempt-01-full-decode-only-c16/cases/native_pinned_uva/ordinary_chat/offload_2p0947265625g/result.json` | FULL_DECODE_ONLY：Prefill eager、Decode FULL Graph replay |
| DeepSeek-V2-Lite BF16 | Ordinary | 16 | Native Copy eager | `artifacts/full-matrix-deepseek-gaps-20260911-attempt-01/native_copy_eager_c16/cases/native_copy_on_forward/ordinary_chat/offload_exact_names/result.json` | — |
| DeepSeek-V2-Lite BF16 | Ordinary | 16 | KTransformers 7T + Decode Graph | `artifacts/e2-ktransformers-equal-offload-c16-20260911-attempt-01/cases/ktransformers/ordinary_chat/cold_06/result.json` | 与 AdaptMoE 使用相同 cold 档位；Prefill E2E 为全批请求产生首 token 的时刻；Decode tok/s 由同步 TPOT 计算 |
| DeepSeek-V2-Lite BF16 | Ordinary | 16 | Native UVA 双阶段 Graph | `artifacts/e2-native-uva-full-separate-fixed-20260911-c16-rest/cases/native_pinned_uva/ordinary_chat/offload_2p0947265625g/result.json` | Prefill/Decode 均为 FULL Graph replay；capture 使用与正式请求一致的 16-request 布局；Decode 相对 Decode-only 基线 -0.13%；完整输出一致 |
| DeepSeek-V2-Lite BF16 | Ordinary | 16 | Native Copy 双阶段 Graph | `artifacts/e2-native-copy-full-separate-direct-c16-20260911-attempt-01/cases/native_copy_on_forward/ordinary_chat/offload_exact_names/result.json` | — |
| DeepSeek-V2-Lite BF16 | RAG | 1 | AdaptMoE 双阶段 Graph | `artifacts/e2-near-equal-offload-plus-one-expert-20260903-attempt-01/c1/cases/ours/rag_summary/cold_02_chunk_2048/rag_summary_c1/client.json; artifacts/full-matrix-deepseek-retries-20260911-attempt-02/adaptmoe_prefill_rag_c1/cases/ours/rag_summary/cold_02/result.json` | Decode tok/s 由同步 phase-batched TPOT 计算 |
| DeepSeek-V2-Lite BF16 | RAG | 1 | Native UVA（eager Prefill + Decode Graph） | `artifacts/e2-native-uva-direct-phase-metrics-20260911-attempt-01-full-decode-only-c1/cases/native_pinned_uva/rag_summary/offload_0p4189453125g/result.json` | FULL_DECODE_ONLY：Prefill eager、Decode FULL Graph replay |
| DeepSeek-V2-Lite BF16 | RAG | 1 | Native Copy eager | `artifacts/full-matrix-deepseek-native-copy-eager-rag-c1-20260911-attempt-05/cases/native_copy_on_forward/rag_summary/offload_exact_names/result.json` | 低档为最近可执行 exact-name 组合，0.812744 GiB；相对 0.896 GiB 差 10.24% |
| DeepSeek-V2-Lite BF16 | RAG | 1 | KTransformers 7T + Decode Graph | `artifacts/e2-ktransformers-equal-offload-c1-20260911-attempt-02/cases/ktransformers/rag_summary/cold_02/result.json` | 与 AdaptMoE 使用相同 cold 档位；Prefill E2E 为全批请求产生首 token 的时刻；Decode tok/s 由同步 TPOT 计算 |
| DeepSeek-V2-Lite BF16 | RAG | 1 | Native UVA 双阶段 Graph | `artifacts/e2-native-uva-direct-phase-metrics-20260911-attempt-01-full-separate-c1/cases/native_pinned_uva/rag_summary/offload_0p4189453125g/result.json` | Prefill/Decode 均为 FULL Graph replay；capture 使用与正式请求一致的 1-request 布局；Decode 相对 Decode-only 基线 +0.26%；完整输出一致 |
| DeepSeek-V2-Lite BF16 | RAG | 1 | Native Copy 双阶段 Graph | `artifacts/e2-native-copy-full-separate-rag-c1-extra-offload-20260913-attempt-02/cases/native_copy_on_forward/rag_summary/offload_exact_names/result.json` | 实际卸载 1.843994 GiB；Prefill/Decode Graph 均 replay，完整生成 128 tokens；峰值显存 32,061 MiB |
| DeepSeek-V2-Lite BF16 | RAG | 16 | AdaptMoE 双阶段 Graph | `artifacts/e2-near-equal-offload-plus-one-expert-20260903-attempt-01/c16/cases/ours/rag_summary/cold_20/result.json; artifacts/full-matrix-deepseek-gaps-20260911-attempt-01/adaptmoe_prefill_c16/cases/ours/rag_summary/cold_20/result.json` | Decode tok/s 由同步 phase-batched TPOT 计算 |
| DeepSeek-V2-Lite BF16 | RAG | 16 | Native UVA（eager Prefill + Decode Graph） | `artifacts/e2-native-uva-direct-phase-metrics-20260911-attempt-01-full-decode-only-c16/cases/native_pinned_uva/rag_summary/offload_7p9599609375g/result.json` | FULL_DECODE_ONLY：Prefill eager、Decode FULL Graph replay |
| DeepSeek-V2-Lite BF16 | RAG | 16 | Native Copy eager | `artifacts/full-matrix-deepseek-gaps-20260911-attempt-01/native_copy_eager_c16/cases/native_copy_on_forward/rag_summary/offload_exact_names/result.json` | — |
| DeepSeek-V2-Lite BF16 | RAG | 16 | KTransformers 7T + Decode Graph | `artifacts/e2-ktransformers-equal-offload-rag-c16-20260911-attempt-02/cases/ktransformers/rag_summary/cold_20/result.json` | 与 AdaptMoE 使用相同 cold 档位；Prefill E2E 为全批请求产生首 token 的时刻；Decode tok/s 由同步 TPOT 计算 |
| DeepSeek-V2-Lite BF16 | RAG | 16 | Native UVA 双阶段 Graph | `artifacts/e2-native-uva-full-separate-fixed-20260911-c16-rest/cases/native_pinned_uva/rag_summary/offload_7p9599609375g/result.json` | Prefill/Decode 均为 FULL Graph replay；capture 使用2×8192 请求布局；Decode 相对 Decode-only 基线 -0.02%；完整输出一致 |
| DeepSeek-V2-Lite BF16 | RAG | 16 | Native Copy 双阶段 Graph | `artifacts/e2-native-copy-full-separate-direct-c16-20260911-attempt-01/cases/native_copy_on_forward/rag_summary/offload_exact_names/result.json` | — |
| DeepSeek-V2-Lite BF16 | Long | 1 | AdaptMoE 双阶段 Graph | `artifacts/e2-near-equal-offload-plus-one-expert-20260903-attempt-01/c1/cases/ours/long_generation/cold_02/long_generation_c1/client.json; artifacts/full-matrix-deepseek-gaps-20260911-attempt-01/adaptmoe_prefill_c1/cases/ours/long_generation/cold_02/result.json` | Decode tok/s 由同步 phase-batched TPOT 计算 |
| DeepSeek-V2-Lite BF16 | Long | 1 | Native UVA（eager Prefill + Decode Graph） | `artifacts/e2-native-uva-direct-phase-metrics-20260911-attempt-01-full-decode-only-c1/cases/native_pinned_uva/long_generation/offload_0p4189453125g/result.json` | FULL_DECODE_ONLY：Prefill eager、Decode FULL Graph replay |
| DeepSeek-V2-Lite BF16 | Long | 1 | Native Copy eager | `artifacts/full-matrix-deepseek-native-copy-eager-c1-20260911-attempt-04/cases/native_copy_on_forward/long_generation/offload_exact_names/result.json` | 低档为最近可执行 exact-name 组合，0.812744 GiB；相对 0.896 GiB 差 10.24% |
| DeepSeek-V2-Lite BF16 | Long | 1 | KTransformers 7T + Decode Graph | `artifacts/e2-ktransformers-equal-offload-c1-20260911-attempt-02/cases/ktransformers/long_generation/cold_02/result.json` | 与 AdaptMoE 使用相同 cold 档位；Prefill E2E 为全批请求产生首 token 的时刻；Decode tok/s 由同步 TPOT 计算 |
| DeepSeek-V2-Lite BF16 | Long | 1 | Native UVA 双阶段 Graph | `artifacts/e2-native-uva-direct-phase-metrics-20260911-attempt-01-full-separate-c1/cases/native_pinned_uva/long_generation/offload_0p4189453125g/result.json` | Prefill/Decode 均为 FULL Graph replay；capture 使用与正式请求一致的 1-request 布局；Decode 相对 Decode-only 基线 +0.05%；完整输出一致 |
| DeepSeek-V2-Lite BF16 | Long | 1 | Native Copy 双阶段 Graph | `artifacts/e2-native-copy-full-separate-direct-c1-20260911-attempt-02/cases/native_copy_on_forward/long_generation/offload_exact_names/result.json` | — |
| DeepSeek-V2-Lite BF16 | Long | 16 | AdaptMoE 双阶段 Graph | `artifacts/e2-near-equal-offload-plus-one-expert-20260903-attempt-01/c16/cases/ours/long_generation/cold_06/result.json; artifacts/full-matrix-deepseek-gaps-20260911-attempt-01/adaptmoe_prefill_c16/cases/ours/long_generation/cold_06/result.json` | Decode tok/s 由同步 phase-batched TPOT 计算 |
| DeepSeek-V2-Lite BF16 | Long | 16 | Native UVA（eager Prefill + Decode Graph） | `artifacts/e2-native-uva-direct-phase-metrics-20260911-attempt-01-full-decode-only-c16/cases/native_pinned_uva/long_generation/offload_2p0947265625g/result.json` | FULL_DECODE_ONLY：Prefill eager、Decode FULL Graph replay |
| DeepSeek-V2-Lite BF16 | Long | 16 | Native Copy eager | `artifacts/full-matrix-deepseek-gaps-20260911-attempt-01/native_copy_eager_c16/cases/native_copy_on_forward/long_generation/offload_exact_names/result.json` | — |
| DeepSeek-V2-Lite BF16 | Long | 16 | KTransformers 7T + Decode Graph | `artifacts/e2-ktransformers-equal-offload-c16-20260911-attempt-01/cases/ktransformers/long_generation/cold_06/result.json` | 与 AdaptMoE 使用相同 cold 档位；Prefill E2E 为全批请求产生首 token 的时刻；Decode tok/s 由同步 TPOT 计算 |
| DeepSeek-V2-Lite BF16 | Long | 16 | Native UVA 双阶段 Graph | `artifacts/e2-native-uva-full-separate-fixed-20260911-c16-rest/cases/native_pinned_uva/long_generation/offload_2p0947265625g/result.json` | Prefill/Decode 均为 FULL Graph replay；capture 使用与正式请求一致的 16-request 布局；Decode 相对 Decode-only 基线 +0.13%；完整输出一致 |
| DeepSeek-V2-Lite BF16 | Long | 16 | Native Copy 双阶段 Graph | `artifacts/e2-native-copy-full-separate-direct-c16-20260911-attempt-01/cases/native_copy_on_forward/long_generation/offload_exact_names/result.json` | — |
| Qwen3-30B-A3B FP8 | Short | 1 | AdaptMoE 双阶段 Graph | `artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/ours/ours/short_dialogue_b1_D/custom/b0_short_dialogue_b1_D/result.json` | — |
| Qwen3-30B-A3B FP8 | Short | 1 | Native UVA eager | `artifacts/full-matrix-qwen-native-eager-20260911-attempt-01/cases/short_dialogue_b1_uva/result.json` | — |
| Qwen3-30B-A3B FP8 | Short | 1 | Native Copy eager | `artifacts/full-matrix-qwen-native-eager-20260911-attempt-01/cases/short_dialogue_b1_copy/result.json` | — |
| Qwen3-30B-A3B FP8 | Short | 1 | KTransformers 7T + Decode Graph | `artifacts/e3-ktransformers-tuning-7t-allcores-flashinfer-overlap-20260903-attempt-01/cases/short_dialogue_b1_cold10/result.json` | — |
| Qwen3-30B-A3B FP8 | Short | 1 | Native UVA 双阶段 Graph | `artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/native/cases/short_dialogue_b1_uva/result.json` | — |
| Qwen3-30B-A3B FP8 | Short | 1 | Native Copy 双阶段 Graph | `artifacts/e3-near-equal-native-copy-graph-20260903-attempt-01/native_copy/cases/short_dialogue_b1_copy/result.json` | — |
| Qwen3-30B-A3B FP8 | Short | 16 | AdaptMoE 双阶段 Graph | `artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/ours/ours/short_dialogue_b16_D/custom/b0_short_dialogue_b16_D/result.json` | — |
| Qwen3-30B-A3B FP8 | Short | 16 | Native UVA eager | `artifacts/full-matrix-qwen-native-eager-20260911-attempt-01/cases/short_dialogue_b16_uva/result.json` | — |
| Qwen3-30B-A3B FP8 | Short | 16 | Native Copy eager | `artifacts/full-matrix-qwen-native-eager-20260911-attempt-01/cases/short_dialogue_b16_copy/result.json` | — |
| Qwen3-30B-A3B FP8 | Short | 16 | KTransformers 7T + Decode Graph | `artifacts/e3-ktransformers-tuning-7t-allcores-split-attn-overlap-20260903-attempt-01/cases/short_dialogue_b16_cold10/result.json` | — |
| Qwen3-30B-A3B FP8 | Short | 16 | Native UVA 双阶段 Graph | `artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/native/cases/short_dialogue_b16_uva/result.json` | — |
| Qwen3-30B-A3B FP8 | Short | 16 | Native Copy 双阶段 Graph | `artifacts/e3-near-equal-native-copy-graph-20260903-attempt-01/native_copy/cases/short_dialogue_b16_copy/result.json` | — |
| Qwen3-30B-A3B FP8 | Ordinary | 1 | AdaptMoE 双阶段 Graph | `artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/ours/ours/ordinary_chat_b1_D/custom/b0_ordinary_chat_b1_D/result.json` | — |
| Qwen3-30B-A3B FP8 | Ordinary | 1 | Native UVA eager | `artifacts/full-matrix-qwen-native-eager-20260911-attempt-01/cases/ordinary_chat_b1_uva/result.json` | — |
| Qwen3-30B-A3B FP8 | Ordinary | 1 | Native Copy eager | `artifacts/full-matrix-qwen-native-eager-20260911-attempt-01/cases/ordinary_chat_b1_copy/result.json` | — |
| Qwen3-30B-A3B FP8 | Ordinary | 1 | KTransformers 7T + Decode Graph | `artifacts/e3-ktransformers-tuned-capacity-aware-graph-20260903-repeat-1/cases/ordinary_chat_b1_cold10/result.json` | — |
| Qwen3-30B-A3B FP8 | Ordinary | 1 | Native UVA 双阶段 Graph | `artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/native/cases/ordinary_chat_b1_uva/result.json` | — |
| Qwen3-30B-A3B FP8 | Ordinary | 1 | Native Copy 双阶段 Graph | `artifacts/e3-near-equal-native-copy-graph-20260903-attempt-01/native_copy/cases/ordinary_chat_b1_copy/result.json` | — |
| Qwen3-30B-A3B FP8 | Ordinary | 16 | AdaptMoE 双阶段 Graph | `artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/ours/ours/ordinary_chat_b16_D/custom/b0_ordinary_chat_b16_D/result.json` | — |
| Qwen3-30B-A3B FP8 | Ordinary | 16 | Native UVA eager | `artifacts/full-matrix-qwen-native-eager-20260911-attempt-01/cases/ordinary_chat_b16_uva/result.json` | — |
| Qwen3-30B-A3B FP8 | Ordinary | 16 | Native Copy eager | `artifacts/full-matrix-qwen-native-eager-20260911-attempt-01/cases/ordinary_chat_b16_copy/result.json` | — |
| Qwen3-30B-A3B FP8 | Ordinary | 16 | KTransformers 7T + Decode Graph | `artifacts/e3-ktransformers-tuning-prefill-fallback4096-20260903-attempt-01/cases/ordinary_chat_b16_cold38/result.json` | — |
| Qwen3-30B-A3B FP8 | Ordinary | 16 | Native UVA 双阶段 Graph | `artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/native/cases/ordinary_chat_b16_uva/result.json` | — |
| Qwen3-30B-A3B FP8 | Ordinary | 16 | Native Copy 双阶段 Graph | `artifacts/e3-near-equal-native-copy-graph-20260903-attempt-01/native_copy/cases/ordinary_chat_b16_copy/result.json` | — |
| Qwen3-30B-A3B FP8 | Medium | 1 | AdaptMoE 双阶段 Graph | `artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/ours/ours/medium_context_b1_D/custom/b0_medium_context_b1_D/result.json` | — |
| Qwen3-30B-A3B FP8 | Medium | 1 | Native UVA eager | `artifacts/full-matrix-qwen-native-eager-20260911-attempt-01/cases/medium_context_b1_uva/result.json` | — |
| Qwen3-30B-A3B FP8 | Medium | 1 | Native Copy eager | `artifacts/full-matrix-qwen-native-eager-20260911-attempt-01/cases/medium_context_b1_copy/result.json` | — |
| Qwen3-30B-A3B FP8 | Medium | 1 | KTransformers 7T + Decode Graph | `artifacts/e3-ktransformers-tuned-7t-graph-20260903-repeat-1/cases/medium_context_b1_cold10/result.json` | — |
| Qwen3-30B-A3B FP8 | Medium | 1 | Native UVA 双阶段 Graph | `artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/native/cases/medium_context_b1_uva/result.json` | — |
| Qwen3-30B-A3B FP8 | Medium | 1 | Native Copy 双阶段 Graph | `artifacts/e3-near-equal-native-copy-graph-20260903-attempt-01/native_copy/cases/medium_context_b1_copy/result.json` | — |
| Qwen3-30B-A3B FP8 | Medium | 16 | AdaptMoE 双阶段 Graph | `artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/ours/ours/medium_context_b16_D/custom/b0_medium_context_b16_D/result.json` | — |
| Qwen3-30B-A3B FP8 | Medium | 16 | Native UVA eager | `artifacts/full-matrix-qwen-native-eager-20260911-attempt-01/cases/medium_context_b16_uva/result.json` | — |
| Qwen3-30B-A3B FP8 | Medium | 16 | Native Copy eager | `artifacts/full-matrix-qwen-native-eager-20260911-attempt-01/cases/medium_context_b16_copy/result.json` | — |
| Qwen3-30B-A3B FP8 | Medium | 16 | KTransformers 7T + Decode Graph | `artifacts/e3-ktransformers-tuning-prefill-fallback1024-20260903-attempt-01/cases/medium_context_b16_cold38/result.json` | — |
| Qwen3-30B-A3B FP8 | Medium | 16 | Native UVA 双阶段 Graph | `artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/native/cases/medium_context_b16_uva/result.json` | — |
| Qwen3-30B-A3B FP8 | Medium | 16 | Native Copy 双阶段 Graph | `artifacts/e3-near-equal-native-copy-graph-20260903-attempt-01/native_copy/cases/medium_context_b16_copy/result.json` | — |
| Qwen3-30B-A3B FP8 | RAG | 1 | AdaptMoE 双阶段 Graph | `artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/ours/ours/rag_summary_b1_D/custom/b0_rag_summary_b1_D/result.json` | — |
| Qwen3-30B-A3B FP8 | RAG | 1 | Native UVA eager | `artifacts/full-matrix-qwen-native-eager-20260911-attempt-01/cases/rag_summary_b1_uva/result.json` | — |
| Qwen3-30B-A3B FP8 | RAG | 1 | Native Copy eager | `artifacts/full-matrix-qwen-native-eager-20260911-attempt-01/cases/rag_summary_b1_copy/result.json` | — |
| Qwen3-30B-A3B FP8 | RAG | 1 | KTransformers 7T + Decode Graph | `artifacts/e3-ktransformers-tuning-prefill-fallback1024-20260903-attempt-01/cases/rag_summary_b1_cold38/result.json` | — |
| Qwen3-30B-A3B FP8 | RAG | 1 | Native UVA 双阶段 Graph | `artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/native/cases/rag_summary_b1_uva/result.json` | — |
| Qwen3-30B-A3B FP8 | RAG | 1 | Native Copy 双阶段 Graph | `artifacts/e3-near-equal-native-copy-graph-20260903-attempt-01/native_copy/cases/rag_summary_b1_copy/result.json` | — |
| Qwen3-30B-A3B FP8 | RAG | 16 | AdaptMoE 双阶段 Graph | `artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/ours/ours/rag_summary_b16_D/custom/b0_rag_summary_b16_D/result.json` | — |
| Qwen3-30B-A3B FP8 | RAG | 16 | Native UVA eager | `artifacts/full-matrix-qwen-native-eager-20260911-attempt-01/cases/rag_summary_b16_uva/result.json` | — |
| Qwen3-30B-A3B FP8 | RAG | 16 | Native Copy eager | `artifacts/full-matrix-qwen-native-eager-20260911-attempt-01/cases/rag_summary_b16_copy/result.json` | — |
| Qwen3-30B-A3B FP8 | RAG | 16 | KTransformers 7T + Decode Graph | `artifacts/e3-ktransformers-tuned-final-20260903-repeat-1/cases/rag_summary_b16_cold89/result.json` | — |
| Qwen3-30B-A3B FP8 | RAG | 16 | Native UVA 双阶段 Graph | `artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/native/cases/rag_summary_b16_uva/result.json` | — |
| Qwen3-30B-A3B FP8 | RAG | 16 | Native Copy 双阶段 Graph | `artifacts/e3-native-copy-full-separate-rag-c16-extra-offload-20260913-attempt-02/cases/rag_summary_b16_copy/result.json` | 实际卸载 21.298364 GiB；Prefill/Decode runtime mode 均为 FULL，16/16 请求各完成 128 tokens；峰值显存 32,071 MiB |
| Qwen3-30B-A3B FP8 | Long | 1 | AdaptMoE 双阶段 Graph | `artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/ours/ours/long_generation_b1_D/custom/b0_long_generation_b1_D/result.json` | — |
| Qwen3-30B-A3B FP8 | Long | 1 | Native UVA eager | `artifacts/full-matrix-qwen-native-eager-20260911-attempt-01/cases/long_generation_b1_uva/result.json` | — |
| Qwen3-30B-A3B FP8 | Long | 1 | Native Copy eager | `artifacts/full-matrix-qwen-native-eager-20260911-attempt-01/cases/long_generation_b1_copy/result.json` | — |
| Qwen3-30B-A3B FP8 | Long | 1 | KTransformers 7T + Decode Graph | `artifacts/e3-ktransformers-tuned-capacity-aware-graph-20260903-repeat-1/cases/long_generation_b1_cold10/result.json` | — |
| Qwen3-30B-A3B FP8 | Long | 1 | Native UVA 双阶段 Graph | `artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/native/cases/long_generation_b1_uva/result.json` | — |
| Qwen3-30B-A3B FP8 | Long | 1 | Native Copy 双阶段 Graph | `artifacts/e3-near-equal-native-copy-graph-20260903-attempt-01/native_copy/cases/long_generation_b1_copy/result.json` | — |
| Qwen3-30B-A3B FP8 | Long | 16 | AdaptMoE 双阶段 Graph | `artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/ours/ours/long_generation_b16_D/custom/b0_long_generation_b16_D/result.json` | — |
| Qwen3-30B-A3B FP8 | Long | 16 | Native UVA eager | `artifacts/full-matrix-qwen-native-eager-20260911-attempt-01/cases/long_generation_b16_uva/result.json` | — |
| Qwen3-30B-A3B FP8 | Long | 16 | Native Copy eager | `artifacts/full-matrix-qwen-native-eager-20260911-attempt-01/cases/long_generation_b16_copy/result.json` | — |
| Qwen3-30B-A3B FP8 | Long | 16 | KTransformers 7T + Decode Graph | `artifacts/e3-ktransformers-tuned-final-20260903-repeat-1/cases/long_generation_b16_cold38/result.json` | — |
| Qwen3-30B-A3B FP8 | Long | 16 | Native UVA 双阶段 Graph | `artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/native/cases/long_generation_b16_uva/result.json` | — |
| Qwen3-30B-A3B FP8 | Long | 16 | Native Copy 双阶段 Graph | `artifacts/e3-near-equal-native-copy-graph-20260903-attempt-01/native_copy/cases/long_generation_b16_copy/result.json` | — |
