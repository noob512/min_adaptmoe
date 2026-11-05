# ShareGPT Trace 对照实验方案

本文档定义 AdaptMoE、KTransformers、Native UVA 和 Native Copy 的真实请求内容对照实验。当前只生成并冻结 trace，不包含任何性能结果；性能结果必须由四个机制读取同一份 trace 后独立运行得到。

## 1. 实验边界

主模型选择 `Qwen3-30B-A3B-FP8`。四个机制在本仓库已有可运行路径，使用同一 tokenizer、同一模型 snapshot 和同一请求协议，避免把模型差异混入机制比较。DeepSeek 作为后续复核模型，不和本轮 trace 结果合并。

ShareGPT 没有请求到达时间戳，因此本实验的准确表述是：

> ShareGPT 真实请求内容 + 固定、可复现的合成 open-loop 到达过程。

不能把合成的 `arrival_ms` 称为生产流量时间戳，也不能让四个机制分别使用各自的随机 `--request-rate`，否则它们看到的请求时序不一致。

## 2. Trace 来源与选样

源文件为：

`workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json`

SHA-256：`35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`。

使用 Qwen snapshot
`/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782` 的 tokenizer。每条候选请求取第一轮 `human/user` 和紧随其后的第一轮 `gpt/assistant`，只把第一轮用户文本作为模型输入；后续多轮内容不拼接，避免不同机制的会话状态和缓存行为影响本轮比较。

过滤规则如下：

- 去掉少于两个 turn、角色非法、空文本、控制字符和重复的首轮请求；重复键为 `SHA256(prompt + NUL + reference_assistant)`。
- 输入长度为 32--8192 tokenizer tokens。
- 参考 assistant 输出长度为 16--512 tokens；`max_tokens = min(reference_length, 512)`。
- `prompt_tokens + max_tokens <= 8192`。
- 使用二维输入/输出长度分层，每个格子 8 条，共 16 个格子、128 条请求；不复制请求补齐稀疏格。

| 输入 tokens | 输出 tokens | 每格请求数 |
| --- | ---: | ---: |
| 32--255 | 16--63、64--127、128--255、256--512 | 8 |
| 256--1023 | 16--63、64--127、128--255、256--512 | 8 |
| 1024--4095 | 16--63、64--127、128--255、256--512 | 8 |
| 4096--8192 | 16--63、64--127、128--255、256--512 | 8 |

严格的 256 条等额 4×4 设计不可行：源数据在 `4096--8192 / 16--63` 格只有 8 条合格记录。选择 128 条是该源文件上不复制请求的最大等额矩阵。实际筛选后有 22,922 条候选记录。

生成器和冻结产物：

- `sharegpt_trace/build_sharegpt_trace.py`
- `inputs/sharegpt_trace_20261013/sharegpt_qwen_trace_base.jsonl`
- `inputs/sharegpt_trace_20261013/sharegpt_qwen_trace_low.jsonl`
- `inputs/sharegpt_trace_20261013/sharegpt_qwen_trace_medium.jsonl`
- `inputs/sharegpt_trace_20261013/sharegpt_qwen_trace_high.jsonl`
- `inputs/sharegpt_trace_20261013/length_summary.csv`
- `inputs/sharegpt_trace_20261013/trace_manifest.json`

每条记录同时保存 `prompt`、OpenAI `messages`、冻结 `prompt_token_ids`、token ID hash、参考输出长度和来源索引。assistant 原文只用于确定长度分层，不作为模型输出目标。直接 engine adapter 使用 token IDs；HTTP adapter 使用 `messages`，并在启动前校验 token ID hash 与 manifest 一致。若要严格固定输出长度，四个 adapter 必须统一启用 `ignore_eos` 或等价 fixed-length 设置，并记录实际生成长度。

## 3. 到达过程

四个机制共用同一条请求顺序和同一条归一化 burst 序列。间隔服从 Gamma(shape=0.5)，先以 1 request/s 生成归一化累计时间，再按速率缩放：

| trace | 名义到达率 | 用途 |
| --- | ---: | --- |
| low | 0.25 request/s | 低压力探索点 |
| medium | 0.50 request/s | 中压力探索点 |
| high | 0.75 request/s | 高压力探索点 |

这三档是名义 offered load，不是已经验证的 GPU 利用率档位。正式实验前先用 16 条 pilot trace 在最慢机制上校准：若 high 档长期排队或无法在预算内完成，只按相同归一化间隔序列重新缩放速率，并在结果中报告实际 offered rate。重新生成速率不会改变选中的请求，只会改变 trace 文件 hash。

当前生成文件的到达跨度约为 low 485.385 s、medium 242.692 s、high 161.795 s；四档文件的请求顺序和 token IDs 完全一致，只有 `arrival_ms` 不同。

## 4. 四种机制与公平配置

主实验使用各机制当前已验证的最佳合法路径，但必须把 Graph 状态、实际卸载量和失败原因单独记录，不能把不同 Graph 能力隐含到结论中。

| 机制 | 主路径 | 固定要求 |
| --- | --- | --- |
| AdaptMoE | Prefill/Decode `FULL_SEPARATE`、双页表和 remote Decode Graph | 使用预先冻结的 Qwen 冷专家 plan；不根据本 trace 重新选冷；记录页表覆盖率和 P→D 时间 |
| KTransformers | 当前调优路径和 Decode CUDA Graph | 7 个 CPU 专家线程、进程保留 14 核 affinity；使用预先冻结的 cold plan；不把切档事务从动态实验中删除 |
| Native UVA | Native pinned-UVA，Prefill/Decode 双阶段 Graph（若该配置对某 shape 非法访存则记 N/A） | 与 Copy 使用同一 tokenizer、scheduler、KV 容量和 capture shape；记录 UVA/pinned 状态 |
| Native Copy | Native copy-on-forward，Prefill/Decode 双阶段 Graph | 与 UVA 使用同一 scheduler/KV/capture 配置；Graph capture OOM 记录为 OOM，不改写成成功 |

建议采用 Qwen 现有近等量静态计划，并逐 run 从 manifest 读取实际卸载量：低档约 AdaptMoE/UVA/Copy/KT = 2.320/2.322/2.322/2.109 GiB，中档约 8.227/7.942/7.942/8.016 GiB，高档约 18.984/20.324/20.324/18.773 GiB。若新 trace 只允许某个机制使用略有不同的合法计划，表中保留实际 GiB，不用名义预算替代。

所有机制固定：greedy（temperature=0）、相同 `max_tokens`、相同 stop/EOS 处理、相同 `max_model_len`、`max_num_batched_tokens`、KV cache 容量和 `max_num_seqs`。建议主实验为 `max_num_seqs=16`，并增加 `max_num_seqs=1` 作为无排队控制组。

## 5. 运行矩阵

最小可发表主矩阵为：

`3 loads × 4 mechanisms × C16 × 3 independent cold repeats = 36 runs`。

再做 C1 控制组：

`3 loads × 4 mechanisms × C1 × 3 repeats = 36 runs`。

每个 cold process 的顺序为：加载模型和冻结 offload plan，运行独立 warmup（不使用正式 trace 的请求），等待 GPU/CPU/stock UVM 状态清理，再按 `arrival_ms` 绝对时间 replay 128 条正式请求。四个机制不得跨运行拼接 TTFT、TPOT、Prefill 或 E2E；每一行都来自同一个完整 replay。

如果完整 128 条 trace 的时间成本过高，先用同一文件前 16 条做 pilot，只用于校准速率和超时，不得将 pilot 数值填入正式结果。

## 6. 采集指标

每个请求记录：到达时间、实际开始服务时间、首 token 时间、完成时间、输入/输出 token 数、TTFT、TPOT（排除首 token）、E2E、错误类型和 request ID。聚合报告：

- 完成率、失败/OOM/非法访存/Graph fallback 数量；
- TTFT、TPOT、E2E 的 mean、p50、p95、p99；
- achieved request/s 和 output tok/s；
- offered arrival rate、队列等待时间、goodput；
- Prefill wall、P→D 和 steady Decode tok/s（只能来自完成 Prefill 后的稳态 Decode step）；
- GPU HBM peak、CPU RSS、PCIe RX/TX、UVM fault/Xid/PTE/PDE/MMU fault；
- 机制配置、实际卸载 GiB、trace/模型/tokenizer hash。

其中 output tok/s 同时报告两种口径：完成输出 token 数除以首个 arrival 到最后完成的 wall time，以及仅对完成请求的 service interval 吞吐。不能用不同机制的单请求闭环时间替代 open-loop trace 的 goodput。

## 7. 结果判定与审计

只有以下条件同时满足时才进入机制比较：四臂 trace hash、请求顺序、prompt token ID hash 和 `max_tokens` 一致；正式请求完成率和 verifier 通过；无 GPU fault；运行结束 GPU 显存清零并恢复 stock UVM。某臂因组件错误或容量限制未完成时，保留 N/A/OOM 和原始日志，不补造数值。

主结论应分别回答：

1. 在相同 offered load 下，哪种机制的完成率、goodput、p95 TTFT/E2E 和 output tok/s 更好；
2. 随压力从 low 到 high，哪种机制首先出现排队、Graph fallback 或容量失败；
3. C1 与 C16 的差异来自远程权重访问、KV/调度排队还是 Graph capture；
4. AdaptMoE 的优势是否在未参与 plan 选择的 ShareGPT 请求上仍然存在。

不得把 ShareGPT 首轮内容和合成 arrival 过程描述为“真实生产 trace”，也不得把一个 trace 的单轮结果写成统计显著性结论。正式稿至少报告三次独立冷运行的中位数和离散度。
