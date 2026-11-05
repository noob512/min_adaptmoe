# 短上下文、变到达率、混合调度单轮实验（2026-10-23）

实验已完成：18/18 个配置，每个配置均为 64/64 请求成功，合计 1152/1152。所有请求均返回并捕获完整 512 个输出 token；未出现 GPU 运行错误，实验结束后 GPU 显存回到 0 MiB。

本轮检验两件事：无 CUDA Graph 时四种机制的混合调度性能，以及仅靠短上下文并发能否触发动态 KV 并改善性能。实测支持“动态 KV 可以减轻固定 K2 的首 token 排队”，但不支持“动态 KV 在所有指标上全面优于固定 KV”。固定 K2 的 TPOT 和平均 E2E 均更好；动态与固定 K4 在 Medium/Long 已很接近。

使用 Qwen3-30B-A3B-FP8、单张 RTX 5090 32 GB。vLLM 三种访问路径均为 TRITON_ATTN、eager、chunked mixed scheduling；AdaptMoE 为 token_threshold，关闭 phase cohort，PARK≤16、RESUME≥128，中间区域保持现有访问状态。KTransformers/SGLang 开启 mixed chunk、关闭 CUDA Graph。所有机制并发上限 C64、step token budget 8192、client connections 96。每个配置独立启动一次，4 条克隆预热请求各生成 8 tokens，再完整回放一次；无重复采样或置信区间。

从之前冻结的 ShareGPT low trace 选取 64 条原 prompt 至少 1024 tokens 的请求，保留来源顺序，分别取 128/512/1024-token 前缀，统一生成 512 tokens、ignore_eos。Short/Medium/Long 最大上下文分别为 640/1024/1536 tokens，没有 8K 长上下文。同一 trace 内长度固定，压力变化仅由到达率变化产生；三种 trace 共用来源请求、到达顺序和时刻。64 个来源身份截断后产生 63 个不同 prompt hash；所有引擎关闭 prefix/radix cache，并读取同一请求集合。这是受控 ShareGPT 前缀实验，输出长度、到达过程和截断文本经过人工构造。

| 到达阶段 | 请求数 | 到达率 | 时间窗口 |
|---|---:|---:|---:|
| low | 8 | 0.25 req/s | [0,32) s |
| high | 48 | 4 req/s | [32,44) s |
| recovery | 8 | 0.25 req/s | [44,76) s |

最后一次到达在 72 s，统计继续覆盖全部请求排空。QPS=成功请求数/正式回放总时长，包含 drain；TTFT、TPOT、E2E 均为请求均值。TPOT 使用首末输出 token 时间差除以服务端输出数减一；每请求输出数为 512。按到达阶段划分的延迟和按墙钟窗口计算的完成 QPS 保存在 phases.csv，二者不可混为同一种“阶段性能”。本轮是预热后一次变压回放，包含瞬态和排队。

| Trace | 配置 | QPS | TTFT mean (s) | TPOT mean (ms) | E2E mean (s) | vLLM 抢占次数 |
|---|---|---:|---:|---:|---:|---:|
| Short 128 | AdaptMoE dynamic K2→K4 | 0.4981 | 0.154 | 126.46 | 64.778 | 0 |
| Short 128 | AdaptMoE fixed K2 | 0.5513 | 0.750 | 90.91 | 47.206 | 23 |
| Short 128 | AdaptMoE fixed K4 | 0.4820 | 0.230 | 143.95 | 73.787 | 0 |
| Short 128 | Native UVA | 0.5135 | 0.223 | 130.84 | 67.081 | 0 |
| Short 128 | Native Copy | 0.4486 | 0.225 | 142.34 | 72.962 | 0 |
| Short 128 | KTransformers | 0.2413 | 8.840 | 404.45 | 215.524 | — |
| Medium 512 | AdaptMoE dynamic K2→K4 | 0.4276 | 0.232 | 148.21 | 75.967 | 17 |
| Medium 512 | AdaptMoE fixed K2 | 0.4213 | 15.644 | 97.37 | 65.398 | 35 |
| Medium 512 | AdaptMoE fixed K4 | 0.4239 | 0.260 | 150.49 | 77.158 | 18 |
| Medium 512 | Native UVA | 0.4180 | 0.620 | 156.93 | 80.811 | 19 |
| Medium 512 | Native Copy | 0.3623 | 2.099 | 151.00 | 79.260 | 21 |
| Medium 512 | KTransformers | 0.1681 | 108.746 | 348.80 | 286.992 | — |
| Long 1024 | AdaptMoE dynamic K2→K4 | 0.3343 | 26.253 | 151.08 | 103.457 | 14 |
| Long 1024 | AdaptMoE fixed K2 | 0.3444 | 39.349 | 79.46 | 79.953 | 20 |
| Long 1024 | AdaptMoE fixed K4 | 0.3346 | 26.164 | 149.76 | 102.691 | 15 |
| Long 1024 | Native UVA | 0.3468 | 30.527 | 145.50 | 104.880 | 15 |
| Long 1024 | Native Copy | 0.3507 | 23.096 | 151.42 | 100.470 | 13 |
| Long 1024 | KTransformers | 0.1152 | 213.448 | 327.97 | 381.047 | — |

动态相对固定 KV 的变化如下。TTFT 降低为正数表示改善；QPS/TPOT 为 dynamic/control−1。

| Trace | 固定对照 | QPS 变化 | TTFT 降低 | TPOT 变化 |
|---|---|---:|---:|---:|
| Short | K2/cold14 | −9.65% | 79.46% | +39.11% |
| Medium | K2/cold14 | +1.51% | 98.52% | +52.22% |
| Long | K2/cold14 | −2.94% | 33.28% | +90.14% |
| Short | K4/cold28 | +3.34% | 33.15% | −12.14% |
| Medium | K4/cold28 | +0.88% | 10.70% | −1.51% |
| Long | K4/cold28 | −0.08% | −0.34% | +0.89% |

固定 K2 为 1364 policy blocks、约 1.998 GiB，配 cold14（每层卸载 14 个专家，主权重 2.953125 GiB）；K4 为 2730 blocks、约 3.999 GiB，配 cold28（每层 28 个，5.90625 GiB）。block size 为 16 tokens。三个 AdaptMoE 对照共用 K4 虚拟池预约、双视图和 plan-class 框架；fixed K2 将 min=max=1364，fixed K4 经预热的已验证事务扩至 K4 后保持，正式窗口内容量恒定。动态使用 80% 主动增长阈值和既有安全缩容策略。物理热专家/KV 驻留检查和 192 MiB headroom gate 保留；每轮全量诊断和 read gate 关闭。

| Dynamic trace | 正式开始后 K2→K4 | 峰值运行请求 | 抢占 | K4→K2 |
|---|---:|---:|---:|---|
| Short | 52.943 s | 58 | 0 | 未观测到 |
| Medium | 37.234 s | 59 | 17 | 未观测到 |
| Long | 34.235 s | 39 | 14 | 未观测到 |

high 在 32 s 开始，三个增长事件均在其后；低压段没有单请求长上下文引起的提前增长。正式窗口内 copy/remote 转换分别为 17/17、17/17、15/15 次，证明 token-threshold 访问选择确实在运行。vLLM CPU scheduler 记录了真实 mixed steps；KTransformers 日志和 mixed-chunk 代码路径也显示新 Prefill 与运行中的 Decode 合批。当前回放只验证到增长，没有验证到完整的 K4→K2 恢复周期。

到 76 s 到达阶段结束时，三个动态臂仍分别有 58/59/60 个未完成请求。降低到达率后仍存在高压积压，不能把 recovery 到达阶段直接解释成低 KV 压力稳态，也不能用本轮未观察到降档判断控制器的完整降档能力。

Medium 的固定 K2 在 high/recovery 到达组中 TTFT 分别为 16.861/23.835 s，动态则为 0.244/0.251 s，说明更大的可运行容量明显减少了 Prefill 前排队。但扩容同时绑定 cold14→cold28，运行并发和专家访问成本一同变化；TPOT 上升后，TTFT 的下降没有转化为更低的平均 E2E。仅提高到达率并不能保证动态 KV 吞吐全面领先，尤其在 prompt 较短、抢占重算成本受限时。本轮没有做独立剖析，不能把差异进一步量化归因于 KV 控制面或某一种搬运成本。

动态相对固定 K4 的主要改善出现在 Short：QPS +3.34%、TTFT −33.15%、TPOT −12.14%；Medium 差异较小，Long 基本相同。动态相对 Native Copy 的 Short/Medium 吞吐更高，Long 则更低；相对 UVA 也没有统一的吞吐优势。这些都是单次观察，百分之几的差异不构成统计显著性结论。

公平性边界：Native 两臂均使用 5.907 GiB 预算选择完整 Parameter，实际 181 个、6,656,924,760 B（6.199744 GiB），比 AdaptMoE K4 主权重多约 4.97%，还包含 attention 等非专家参数。KTransformers 使用同一 cold28 坐标、每层 100 个 GPU 专家，max-total-tokens=43696，与 vLLM K4 的 43680 个有效 token 接近；其 page size/调度器与 vLLM 不同，冷专家在 CPU 计算，full-GPU Prefill fallback 关闭，使用当前默认 MoE 配置且日志提示缺少 E=100 的专用调优配置。本轮衡量这些现有系统配置，无法隔离成相同专家坐标和相同算子的纯访问路径消融。

输入 token hash、请求顺序、输出长度和计划到达时刻跨臂一致；已完成请求均捕获完整 512 个输出 token。完整输出存在差异：例如 Short 动态与 fixed K2/fixed K4/UVA/Copy 的首 token 一致数为 63/64、64/64、64/64、64/64，完整 512-token 序列一致数分别为 11/64、14/64、13/64、11/64；与 KTransformers 为 54/64 和 3/64。因此这些结果只提供性能证据，精度/数值等价需要独立验证。

最终审计通过：18 个 cell 的 trace SHA-256 均匹配，各臂 arrival_scale=1、connections≥64；命令行和日志确认 Graph 关闭，vLLM 每个 cell 都有真实 mixed steps；fixed K2 没有档位提交，fixed K4 的唯一增长均在正式窗口前完成，三个 dynamic 的增长均在 high 开始后。客户端最大发送偏差为 6.46 ms。

原始数据和复现入口：

- [实验方案](PLAN_BOUNDED_MIXED_SINGLE_SWEEP_20261023.md)
- [矩阵脚本](run_bounded_mixed_matrix.sh)、[trace 生成器](build_bounded_mixed_trace.py)、[分析脚本](analyze_bounded_mixed.py)
- [Trace manifest](artifacts/bounded-mixed-six-arm-20261023-attempt-02/traces/manifest.json)
- [完整结果 CSV](artifacts/bounded-mixed-six-arm-20261023-attempt-02/overall.csv)、[阶段结果 CSV](artifacts/bounded-mixed-six-arm-20261023-attempt-02/phases.csv)、[分析 JSON](artifacts/bounded-mixed-six-arm-20261023-attempt-02/analysis.json)
- [执行进度](artifacts/bounded-mixed-six-arm-20261023-attempt-02/progress.log)，各 cell 目录保留 replay.json、逐请求时间与输出 token、server.log、configuration.txt、server_command.txt、scheduler_steps.jsonl（vLLM）和 offload manifest。

首个启动 attempt-01 在模型加载和正式计时之前因运行期间编辑 Bash 脚本而中止，没有产生测量样本；修复后 attempt-02 中每个 cell 只进行一次有效回放。失败启动保留，未混入结果。
