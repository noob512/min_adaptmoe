# ShareGPT Low / C16 / Eager 单轮结果（2026-10-13）

## 1. 实验范围

本轮只运行冻结 ShareGPT low trace 的四种机制，每种机制一次独立冷启动，不使用 CUDA Graph：

- 模型：Qwen3-30B-A3B-FP8
- 请求：128 条，16 个输入/输出长度格各 8 条
- 到达过程：Gamma(shape=0.5)，名义速率 0.25 request/s
- 并发上限：16
- `max_num_batched_tokens=8192`
- KV cache：1.75 GiB（1,879,048,192 B）
- 每臂先执行 4 个独立 warmup 请求，warmup 不计入正式窗口
- 正式请求统一使用 greedy、固定 `max_tokens` 和 `ignore_eos=True`
- Trace SHA-256：`416fe6e9bd464fec30259bdc375c462a1486e252b5a2555176452ab7253d98a2`

有限 trace 的最后一个到达时刻对应 offered QPS 0.263708。Achieved QPS 的口径是 128 个成功请求除以正式首请求发出至最后请求完成的测量窗口。

CUDA Graph 确实关闭：三条 vLLM 路径均使用 `--enforce-eager` 和 `cudagraph_mode=NONE`，日志显示 `Cudagraph is disabled under eager mode`；KTransformers 使用 `--disable-cuda-graph`，运行日志显示 `cuda graph: False`。

## 2. 单轮结果

所有机制均成功完成 128/128 个正式请求，无请求错误。

| 机制 | 实际卸载量 | 测量窗口 (s) | QPS | TTFT mean (ms) | TTFT p50 (ms) | TTFT p95 (ms) | TPOT mean (ms) | TPOT p50 (ms) | TPOT p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AdaptMoE | 2.320 GiB | 520.729 | 0.245809 | 26,707.59 | 21,071.00 | 60,719.58 | 58.107 | 55.213 | 72.909 |
| KTransformers | 2.110 GiB | 724.303 | 0.176722 | 112,107.51 | 137,656.10 | 215,012.52 | 170.067 | 137.733 | 312.503 |
| Native UVA | 2.322 GiB | 498.767 | **0.256633** | 916.89 | 271.15 | 4,473.88 | **53.719** | **51.031** | 69.424 |
| Native Copy | 2.322 GiB | 501.723 | 0.255121 | **667.09** | **148.56** | **3,823.07** | 59.880 | 59.839 | **63.885** |

相对 QPS：Native UVA 比 AdaptMoE 高 4.40%，Native Copy 比 AdaptMoE 高 3.79%，AdaptMoE 比 KTransformers 高 39.09%。Native UVA 与 Native Copy 只差 0.59%。

## 3. 结果解释

Native UVA 和 Native Copy 的完成速率都接近本 trace 的到达上限，因此二者 TTFT 主要是服务时间和短时 burst 排队。Native Copy 的 TTFT 更低，但 Native UVA 的平均 TPOT 更低；在 low offered load 下，两者 QPS 基本相当。

AdaptMoE 的平均 TPOT 只比 Native UVA 高 4.39 ms，且比 Native Copy 低 1.77 ms，但平均 TTFT 达 26.7 s。这不是“单次 Prefill 计算需要 26.7 s”，而是 open-loop 请求积压后的到达至首 token 延迟。运行期间记录到 26 次 Prefill→Decode PARK 和 26 次 Decode→Prefill RESUME：26 次 PARK 的 transition wall 合计约 9.70 s，26 次 RESUME wall 合计约 7.78 s。频繁阶段切换是本轮 AdaptMoE 排队的重要额外开销，但端到端单轮数据不能把全部 TTFT 差距严格分解到该控制面。

KTransformers 在这档配置下已经明显低于 offered load，队列持续增长，因此 TTFT 和 QPS 都受积压支配。其 TPOT 也显著更高。该臂使用 KTransformers/SGLang 运行时和 2.110 GiB 卸载量，不应把与 vLLM 三臂的全部差距仅解释为权重访问机制差异。

## 4. 公平性和限制

- 四臂读取相同 trace，请求顺序、输入 token、期望输出长度和到达时间一致；usage 口径的正式输出 token 总数均为 24,254。
- 实际卸载量并非完全相等：AdaptMoE 为 2,491,416,576 B，Native UVA/Copy 为 2,493,476,896 B，KTransformers 为 2,265,477,120 B（含 scale）。
- 四臂完整输出 token 序列没有保持一致。128 条中，AdaptMoE/KTransformers 完全一致 3 条，AdaptMoE/UVA 21 条，AdaptMoE/Copy 20 条，UVA/Copy 52 条。不同批处理、执行路径及 FP8 数值差异都可能改变后续 greedy token，因此本轮只验证协议输入和输出长度，不能声称逐 token 输出一致。
- 请求 `sharegpt-0092` 在三条 vLLM 路径上的 HTTP usage 为 110 completion tokens，但客户端 tokenizer 得到 109 个 token ID；KTransformers 得到 110 个。聚合 TPOT 使用服务返回与流式时间戳，不受该 tokenizer 计数差异影响。
- 每种机制只有一次独立冷启动，没有方差或置信区间。本表是功能性单轮结果，不能作为统计显著性结论，也不能外推到 medium/high offered load。

## 5. 产物

原始产物位于：

`experiments/eurosys_moe_offload_evaluation/sharegpt_trace/artifacts/eager-low-c16-once-20261013/`

每个机制目录均保存 `configuration.txt`、`server_command.txt`、`server.log`、`replay.json`、输入 hash 和运行后 GPU 状态。AdaptMoE、Native UVA、Native Copy 另有实际 offload manifest；KTransformers 保存 cold-plan 转换记录。

为使 AdaptMoE eager 路径合法运行，本轮补齐了 all-remote 启动 profile 后的 hot-expert 恢复，并避免在 eager cold11 配置中误启用仅 full-dual-view 需要的内核强制项。相关回归测试结果为 `3 passed, 102 deselected`。
