# 面向 EuroSys 的阶段感知 MoE 卸载评测方案（v2）

日期：2026-08-24  
状态：v2 协议已冻结；E0 已通过基础兼容性 qualification，E1 已完成五重复但处于
跨框架数值正确性 hold  
协议 ID：`eurosys-pa-v2-20260824`

## 1. 评测目标与论文范围

本项目要回答的问题不是“把一部分专家放到 CPU 后某个 kernel 快了多少”，而是：

> 在 GPU HBM 同时被 MoE 专家权重和 KV cache 竞争时，系统能否保持 Parameter VA、
> KV VA 和 CUDA Graph 稳定，根据 Prefill/Decode 阶段选择不同的数据路径，并在安全
> 调度边界动态调整 KV 物理驻留与专家驻留，从而提高可服务容量和 SLO goodput？

当前可诚实支撑的系统范围是：

- 单机、单 GPU；
- phase-batched serving，即一个 scheduler step 不混合新的 Prefill 和旧请求 Decode；
- Qwen3-30B-A3B-FP8 使用 `TRITON_ATTN` 和 Prefill/Decode `FULL_SEPARATE` CUDA Graph；
- 专家计划按层对称：每个 MoE 层卸载相同数量的冷专家；
- 拓扑变化只在 scheduler idle 或等价安全 epoch 执行；
- 未支持状态 fail-closed，不依赖隐式 UVM fault 修复。

除非 E11 完成 phase-epoch continuous batching，否则论文不得声称支持通用 vLLM
continuous batching。除非新增多 GPU 结果，否则不得外推到 TP/EP 多卡部署。

## 2. 当前受测系统

### 2.1 阶段专用权重路径

系统为同一组专家维护稳定的原始 Parameter VA：

```text
Prefill
  pinned CPU source -> upload VA -> GPU backing
                                  -> Parameter VA 读取

Decode
  Parameter VA -> CPU-backed expert pages remote-read
```

Prefill 的 upload VA 和模型读取的 Parameter VA 虚拟地址不同，但在 `sealed` 状态映射
到相同 GPU backing。进入 Decode 时，PARK 将 Parameter VA 的 consumer alias 切换到
CPU-backed 权重页；下一轮 Prefill 再 RESUME。Graph 始终持有稳定 VA。

### 2.2 双 Graph 与可逆拓扑

当前实现分别捕获 Prefill alias Graph 和 Decode remote Graph，并维护：

```text
sealed <-> remote_parked
```

切换使用定制 UVM 的 MAP/SEAL/PARK/RESUME/REBIND/RESIZE ABI。Graph replay 期间不执行
页表修改；Graph、PTE、session、backing 和 residency generation 必须一致，否则拒绝
发布新状态。

基本理解接近，但更准确地说：

  > Graph replay 期间，虚拟地址和 VA → 物理页 的映射关系都不变；Graph 可以读取或修改这些物理页中的数据。

  这里要区分三件事：

  - 虚拟地址 VA：例如 Parameter.data_ptr()，只是 kernel 使用的地址值。
  - 页表项 PTE：决定这个 VA 当前映射到哪个 GPU/CPU 物理页。
  - 物理页内容：真正存放权重、激活和输出数据的内存。

  因此，“不执行页表修改”不是说“不通过虚拟地址修改内存”。GPU kernel 和 cudaMemcpyAsync 仍然通过虚拟地址访问内存，只是地址翻译关系在 replay 期间被冻结。

  以 Prefill Graph 为例：

  固定 Upload VA ──PTE──> 固定 GPU 物理页 <──PTE── 固定 Parameter VA
         │                       │
         │ H2D 写入权重           │ MoE kernel 读取权重
         └───────────────────────┘

  Replay 期间：

  1. 捕获的 H2D 节点通过 Upload VA 修改 GPU 物理页中的权重数据；
  2. MoE kernel 通过原始 Parameter VA 读取同一物理页；
  3. 两个 VA 不变；
  4. 两边的 PTE 不变；
  5. 变化的是物理页中存储的数据，而不是“物理地址本身”。

  这正是当前 static-alias 路径的实现：docs/cudagraph_static_alias_transport.md:19。稳态 replay 只执行 H2D 和计算，不允许 mapping ioctl：docs/
  cudagraph_static_alias_transport.md:43。

  而在 Prefill/Decode 阶段切换方案中，PTE 确实可能改变，但发生在 Graph replay 之外：

  Prefill Graph replay
          ↓ 完成并同步
  PARK：Parameter VA 的 PTE
        GPU 物理页 → CPU-backed 物理页
          ↓
  Decode Graph replay

  具体流程见 docs/moe_phase_aware_offload_technical_design.md:287。Decode replay 时权重主要是只读：GPU 通过不变的 Parameter VA 远程读取 CPU 权重页，Graph 内
  没有权重 H2D。

  所以一句话总结：

  > Graph replay 期间不是“不修改虚拟地址中的内容”，而是“不修改虚拟地址值和页表映射”；内存数据仍然可以正常读写。PTE 切换只放在两次 replay 之间，并在下一次
  > replay 前完成 TLB/tracker 同步。

### 2.3 固定 Plan Class 与差集切换

系统支持少量固定容量档位，例如 C2/C4/C6/C8。启动时建立全生命周期稳定的 upload、
pinned source、Parameter VA 和逻辑 slot envelope，并为各 class 保留独立 GraphExec
表和地址证书。正常切换只处理：

```text
retained = old ∩ new
added    = new - old
removed  = old - new
```

retained backing/session 原位复用；只 refill 发生 `slot -> expert` 变化的 source；切换
不执行 DESTROY、全量 finalize 或 CUDA Graph recapture。`STATIC_ALIAS_REBIND` 与
`RESIZE` 具有 BEGIN/COMMIT/ABORT 和反向回滚。

C2/C4/C6/C8 是 manifest 中的 Graph/slot 容量 class，不应仅凭名称推断其 GPU HBM
含义。KV target 到 class 的映射必须由实测预算生成并冻结。

### 2.4 三队列 KV pool 与统一控制器

KV tensor 和最大 VA 一次建立，运行时只改变物理驻留：

- GPU/free：GPU resident，可立即分配；
- GPU/backup：GPU resident，普通 free 不足时才开放；
- CPU/offline：CPU resident，grow 成功后进入 GPU 队列。

高压路径：

```text
backup 被使用 / preemption / 高水位
  -> 取消 pending shrink
  -> 每层对称地释放一轮或多轮冷专家
  -> KV target 按倍增或预测需求增长
  -> CPU/offline KV 迁入 GPU
  -> 补回 backup reserve
```

低压路径：

```text
R = max(实际峰值 A, 下一轮预测 P) + backup + 预测误差保护 E

连续三轮满足 R <= T/2
且无 backup 使用、preemption、UNKNOWN block
  -> 生成 shrink_pending
  -> 等待安全边界
  -> quarantine 空闲 tail
  -> GPU -> CPU migration
  -> target 减半但不低于下界
  -> 每层对称恢复此前由运行时卸载的专家
```

控制器统一处理预测、backup、preemption、实际峰值、上下界、最短驻留时间、迟滞和
prediction error，避免旧 low/high 控制器与 backup 控制器相互覆盖。

### 2.5 GPU 热度统计与 CPU 计划更新

GPU 复用 router assignment，按层累计专家 token 命中数；两个 heat buffer 轮换，CPU
维护 EWMA，并周期性生成 class 内候选集合。当前已实现同步 maintenance RPC 读取已轮换
buffer，以及同容量 class 内的冷热专家交换。

尚未完成的是将 pinned D2H 热度摘要完全异步接入请求结果通道。因此论文可以测量现有
同步维护开销，但在该路径完成前不能宣称热度返回与推理完全重叠。

## 3. 当前证据台账

### 3.1 已完成的 qualification

| 能力 | 当前证据 | 已证明 | 未证明 |
|---|---|---|---|
| 三系统兼容性 E0 | DeepSeek-V2-Lite BF16，ours/KT/MoE-Infinity 输出 hash 一致 | 三方能通过统一 OpenAI 接口执行冻结请求 | 三方正式性能排名 |
| 双 Graph 与 PARK/RESUME | Qwen B1/B4、2/4/6 GiB remote Decode | Prefill/Decode Graph、VA 与输出正确 | mixed batch、长期在线稳定性 |
| 固定 C2/C4/C6/C8 class | DeepSeek 单进程 `C2 -> C4 -> C6 -> C8 -> C2` | session/backing 差集复用、Graph capture `8 -> 8`、输出一致 | Qwen 四 class 正式 E2E、在线自动选 class |
| 物理 KV pool | Qwen 105 <-> 1,023 blocks | 固定 VA、真实 GPU/CPU residency、无 recapture、HBM 可复用 | 长时间控制器稳定性 |
| 三队列双向闭环 | Qwen `24 -> 40 -> 20` | backup 触发 grow、三轮低压 shrink、每层卸载/恢复 1 个专家 | 大并发和重复峰谷 |
| 小规模路径等价 | Qwen dynamic 与 direct high/low | 到达同状态后无显著持续损失 | 论文级重复和置信区间 |
| 真实大容量路径等价 | Qwen `2 -> 4 -> 8 -> 10 -> 8 -> 4 -> 2 GiB` | 物理 KV、多轮逐层专家差集、双 Graph、输出/VA/清理 | 自动预测、跨 class、低切换延迟 |
| 热度控制面 | 128 experts、双 buffer、CPU EWMA | 计数正确和 class 安全选择 | 异步 pinned D2H、长期 workload 收益 |

主要证据入口见第 18 节。所有这些结果都是功能或单次路径 qualification；除非其 run
root 属于新冻结协议、重复数和统计方法满足本文要求，否则不能直接进入论文主表。

### 3.2 Qwen 2--10 GiB qualification 的准确含义

该实验使用 Qwen3-30B-A3B-FP8、`TRITON_ATTN` 和 FULL Prefill/Decode Graph：

| KV 档位 | GPU KV target | 每层 remote experts | 相对 base70 的增量 waves |
|---|---:|---:|---:|
| K2 | 2,146,959,360 B | 70 | 0 |
| K4 | 4,295,491,584 B | 80 | 10 |
| K8 | 8,589,410,304 B | 99 | 29 |
| K10 | 10,737,942,528 B | 108 | 38 |

stepped 与 direct 到达相同状态后的 Prefill、Decode median 和 Decode p95 差异绝对值
均小于 0.5%，Graph capture 在每个进程内保持 `2 -> 2`。这证明没有检测到路径历史导致
的持续性能损失。

它不证明切换已经足够快。增长阶段 38 个串行 expert waves 累计约 95.9 s，而三次纯
KV grow 合计约 2.21 s；下降阶段 expert restore 合计约 25.7 s，KV shrink 合计约
9.93 s。正式动态性能实验前，应把多轮 wave 合并为单次全层差集事务。

### 3.3 正式评测基础设施现状

`experiments/eurosys_moe_offload_evaluation/` 已具备：

- 不可覆盖 run root 和 provenance 收集；
- 预 tokenized streaming client；
- 统一 localhost OpenAI 请求边界；
- ours、KTransformers、MoE-Infinity server launcher；
- E0、E1、E2 独立 verifier；
- DeepSeek 与 Qwen 的 calibration/validation/evaluation/shift fixtures；
- E2 的 workload/concurrency cell 描述和 equal-HBM 检查框架。

当前正式状态是：

- E0 已在 v2 run root 完成 provenance 补登记；三系统共同 validation 请求的 32-token
  hash 一致，KT eager/Graph 和 placement、MoE-Infinity counters、ours 的
  VA/PTE/residency/fault/cleanup 证书均已登记；
- E1 已完成三系统各 5 个独立 process blocks，共 15 blocks 和 150 条 measurement
  请求，balanced cyclic order、C1 和每分钟最多 10 次请求启动均通过；
- E1 的 10-prompt broad-generation gate 未通过：ours 和 KT 各自五次稳定，
  MoE-Infinity 仅 2/10 prompts 在五次间稳定，且 0/10 prompts 在全部 15 blocks 上
  一致；因此现有数字只能作为描述性 anchor，不能进入跨系统 speedup 表；
- E2 只有脚本、配置和 verifier，尚无完整正式矩阵；
- `eurosys-pa-v1-20260823` 已冻结，不得在文档更新后悄悄复用；v2 正式 run root 为
  `eurosys-pa-v2-20260824T142031Z-e0-e1-formal`。

### 3.4 当前 P0 缺口

1. 将多轮 expert release/restore 合并为单次全层差集事务；
2. 完成长期 workload 中 KV target、backup、plan class 和热度计划的同控制器闭环；
3. 完成或明确放弃 mixed Prefill/Decode phase-epoch scheduler；
4. 为 Qwen 固定 Graph class 做正式四档整模型 qualification；
5. 完成 E1 teacher-forced 数值门和 E2 equal-HBM 主矩阵；
6. 增加 6--24 小时 soak、故障注入和重复峰谷；
7. 若要声称热度采集与请求完全重叠，完成 pinned asynchronous D2H；
8. 多 GPU 仍未实现，保持 limitation。

## 4. 论文主张与 Research Questions

### 4.1 建议主张

| 主张 | 必需证据 | 当前状态 |
|---|---|---|
| C1：阶段专用 copy/remote 路径优于单一路径 | 同模型、同专家坐标的 hybrid/always-copy/always-remote crossover | 待正式实验 |
| C2：稳定 VA 支持可逆拓扑和无 recapture Graph | VA fingerprint、Graph counter、PTE/residency、输出 | qualification 已有，需重复统计 |
| C3：固定 class 差集切换显著低于全量重建 | full rebuild、差集 class switch、batched wave 对比 | 已有 pilot，需 Qwen/重复实验 |
| C4：三队列与统一控制器提高 KV burst goodput | fixed/oracle/automatic trace、OOM、SLO、HBM timeline | 单次双向已过，长期待做 |
| C5：热度驱动计划能适应 workload shift | static、EWMA、oracle、adversarial shift | 控制面已实现，E2E 待做 |
| C6：同等完整系统 HBM 下优于外部基线 | 统一 client、equal measured HBM、5 process blocks、CI | 尚无正式数据 |

这里的 C1--C6 是论文候选主张的编号，不是已经得到证明的结论；其中 C1 也不表示
concurrency=1。每项主张只有在对应证据通过正式 verifier 后才能进入论文结论。

#### C1：阶段专用路径优于单一路径

目标方案在 Prefill 中把选定权重复制到 GPU 后计算，在 Decode 中通过稳定 Parameter VA
直接 remote-read CPU 权重。正式 crossover 必须在同一模型、同一专家坐标和同一请求下
比较 hybrid、always-copy 和 always-remote，回答不同 prompt/output/batch 和权重复用程度下
哪条路径更合适。当前尚无正式 crossover，因此只能说明设计动机，不能宣称 hybrid 已经
优于两种单一路径。

#### C2：稳定 VA 支持可逆拓扑和无 recapture Graph

该主张要求 Parameter VA 在 GPU backing 和 CPU remote-read 拓扑之间往返时保持不变，
并继续 replay 已捕获的 CUDA Graph。证据必须同时覆盖 VA fingerprint、Graph capture
counter、PTE、物理 residency、PARK/RESUME 往返输出、UVM fault 和资源清理。现有
qualification 已证明机制可行，是当前最接近被支持的主张，但仍需多个独立进程和重复切换，
报告失败率、延迟分布和长期稳定性。

#### C3：固定 class 差集切换低于全量重建

固定 class 切换应复用 `retained = old intersect new`，只处理 `added = new - old` 和
`removed = old - new`。正式实验必须比较 full rebuild、固定 class 差集切换、多轮串行
expert wave 和合并后的 batched wave，并报告 median、p95、失败及回滚。当前 pilot 只能
说明差集机制有潜力，仍需在真实 Qwen 大容量计划上重复验证，不能直接作为论文结论。

#### C4：三队列和统一控制器提高 KV burst goodput

三队列控制器管理 GPU free、GPU backup 和 CPU offline：突发到来时先吸收 backup，必要时
卸载专家并扩大 GPU-resident KV；压力下降后缩小 KV 并恢复专家。正式证据必须比较固定
策略、提前知道未来压力的 oracle 和在线 automatic 控制器，并报告 goodput、SLO 违约、
OOM、HBM/KV 时间线以及专家切换次数和成本。当前仅完成一次升压和一次降压，证明双向
路径可用；在长期重复 burst、多 seed 和高低压力交替完成前，不能宣称在线 goodput 提升。

#### C5：热度计划适应 workload shift

目标是在请求分布变化后识别新的热门专家并更新 GPU/CPU 放置。实验必须比较 static、
EWMA、oracle 和 adversarial shift，测量热点识别与 EWMA 收敛时间、plan hit rate、
remote-read/H2D 流量、切换净收益和计划抖动。当前已经实现 GPU heat、CPU EWMA 和同步
专家交换控制面，但异步 shadow plan 和真实 workload shift E2E 尚未完成，因此只能声称
控制逻辑存在，不能声称它已经带来性能收益。

#### C6：相同完整 HBM 预算下优于外部系统

这是要求最高的条件性主张。公平比较必须统一模型、输入、client 和计时边界，输出通过
正确性门，并按完整 measured HBM 对齐 GPU 权重、KV cache、retained backing、CUDA Graph
pool/workspace、allocator reserve 和其他框架显存。每个主点至少需要 5 个独立 server
process blocks，报告均值、方差和置信区间，并保留 OOM、失败和 unsupported case。当前
没有满足这些条件的完整正式数据，因此不能宣称优于 KTransformers、MoE-Infinity 或
其他外部基线。

C6 不属于必须坚持的核心主张。如果完整 E2 无法满足输出、预算或计时公平性，应删除
“优于外部系统”和总 speedup 排名，转而保留有充分证据的机制与容量结论，包括稳定 VA
无 recapture Graph、阶段拓扑切换、差集事务、KV/专家共享 HBM 和可服务容量。整体上，
C2 已有机制 qualification 但缺正式重复；C3 有 pilot 但缺真实规模统计；C4 只有单次闭环；
C5 只有控制面；C1 和 C6 尚缺正式实验。论文应优先补齐 C1--C5，只有 E2 通过完整公平性
检查后才加入 C6 的跨系统排名。

### 4.2 Research Questions

这些 RQ 是论文需要通过实验回答的问题，不是预设为正面的结论。实验可能证明某项机制
只在有限范围内有效，或其成本超过收益；这种负面结果也必须如实报告。

#### RQ1：Prefill-copy/Decode-remote 在什么条件下优于单一路径？

目标方案在 Prefill 中先把选定权重复制到 GPU，再执行高复用计算；Decode 则让 GPU 直接
remote-read CPU 权重。实验需要在相同模型、相同专家坐标和相同请求下比较 hybrid、
always-copy 与 always-remote，并扫描三个主要变量：prompt 长度决定 Prefill 的权重复用，
output 长度决定 Decode 重复步数，batch 大小决定同一权重能被多少 token 共同使用。

预期短 prompt、长 output 和小 batch 更可能适合 remote，长 prompt、大 batch 和高复用
更可能适合 copy，但实际 crossover 必须由测量决定。最终结果应给出可操作的路径选择边界，
而不是只报告单一 workload 上的最好数字。

#### RQ2：稳定 VA、双 Graph、grouped PTE 和 PARK/RESUME 分别贡献多少？

完整系统同时使用四类机制：稳定 VA 保持 `Parameter.data_ptr()` 不变；双 Graph 分别服务
Prefill 和 Decode；grouped PTE 批量更新映射；PARK/RESUME 在 CPU remote-read 与 GPU
backing 之间执行可逆切换。RQ2 应通过消融逐项替换这些机制，例如稳定 VA 对比重新绑定与
recapture、双 Graph 对比单 Graph/eager、grouped PTE 对比 sequential PTE、PARK/RESUME
对比 destroy/rebuild。

每组实验必须报告 Graph recapture 次数、切换 median/p95、PTE ioctl 数量、HBM 占用、
输出正确性、fault 和资源清理。目标是把收益归因到具体机制，不能把所有改善笼统归因于
“固定 VA”。

#### RQ3：固定 Plan Class 差集切换相对全量重建节省多少？

固定 class 切换只处理集合差异：

```text
retained = old intersect new
added    = new - old
removed  = old - new
```

实验应比较 full rebuild、固定 class 差集切换、多轮串行 expert wave 和一次 batched 全层
差集事务。除总切换时间外，还要拆分 CPU 计划生成、source refill、backing 分配、PTE 修改、
TLB/tracker 同步、residency audit 与 Graph/certificate 激活。RQ3 不仅要回答“快了多少”，
还要定位批量化之后剩余的主要开销。

#### RQ4：专家和 KV 共享 HBM 能否提高服务能力？

GPU HBM 同时承载模型与专家权重、KV cache、retained backing、CUDA Graph/workspace 和
安全余量。系统在 KV 压力上升时卸载部分专家，把 HBM 转给 KV；压力下降时缩小 KV 并
恢复专家。实验要判断这种共享是否提高可服务并发，是否减少 KV 抢占与 OOM，以及专家
remote-read 的损失是否小于 KV 扩容收益。

核心指标是 SLO goodput，即单位时间内满足 TTFT、TPOT 或 E2E 延迟目标的请求数，而不是
单纯吞吐。如果完成的请求更多，但大量请求超过延迟目标，就不能认为服务能力提高。

#### RQ5：GPU backup reserve 应保留多少？

GPU backup 是为突然出现的 KV 需求保留的 GPU block。reserve 太小会被突发迅速耗尽，
控制器来不及卸载专家和扩容 KV，导致抢占、SLO 违约或 OOM；reserve 太大则长期占用
HBM，减少 GPU expert 容量并增加 remote-read 流量。

实验需要扫描 reserve 大小、突发规模和持续时间，报告可吸收的请求/KV blocks、控制器
反应时间、突发 goodput、稳态闲置 HBM 和专家切换次数。目标是找到足以覆盖控制器反应
延迟、但不会过度浪费 HBM 的水位。

#### RQ6：workload shift 后热度计划多久恢复有效？

当请求分布变化时，原来的 hot experts 可能变冷，新的专家开始变热。系统通过 GPU heat
统计、CPU EWMA、candidate plan 和专家放置更新跟踪这种变化。实验应以墙钟时间、请求数、
token 数和 heat windows 衡量恢复时间，并报告 class 内专家集合的 Jaccard overlap、plan
hit rate、remote-read/H2D 流量和切换净收益。

还要检查 EWMA 过慢是否长期跟不上变化、EWMA 过快是否被短期噪声驱动，以及 adversarial
shift 下是否发生反复换入换出。只有节省的远程流量与计算时间超过计划切换成本，热度控制
才具有端到端价值。

#### RQ7：remote-read 在高并发下首先遇到什么瓶颈？

remote-read 的主要数据路径是 `CPU DRAM -> PCIe -> GPU kernel`。它可能先受到 PCIe 带宽、
CPU 内存带宽或 GPU memory stall 限制：即使 PCIe 尚未完全饱和，GPU 也可能因为等待远程
数据而降低 SM 利用率。

实验应从 concurrency C1/C4/C8/C16 等档位逐步增加并发，联合采集 PCIe RX/TX、CPU DRAM
bandwidth、GPU utilization/memory stall、Decode TPOT 和每 token remote bytes。最终需要
确定 remote-read 的可扩展范围，以及超过哪个并发或流量后应切换到 copy/cache。

#### RQ8：系统在长期运行和故障下是否保持安全？

短时 qualification 不能代表长期可靠性。RQ8 需要覆盖长时间高低压力交替、热点漂移、
重复 PARK/RESUME 和 class 切换，以及 ioctl、D2H、PTE 提交、审计、请求取消、shutdown
等故障注入。

检查项包括 VA/PTE/session/generation 一致性、是否出现部分提交、回滚是否成功、是否出现
UVM fault、Xid、OOM 或 kernel error，以及 backing、pinned memory、Graph 和 GPU client
是否全部释放。还要检查多轮运行后性能和 HBM 是否持续恶化。正式证据应包含数小时 soak、
重复峰谷、故障注入和退出后的清理证书。

#### RQ9：相同真实 HBM 下是否优于外部基线？

跨系统比较不能只设置相同的名义 offload GiB，而要对齐实际 measured peak HBM，并完整
计算 GPU 权重、GPU expert cache、KV cache、retained backing、Graph/workspace、allocator
和框架额外显存。比较对象包括 KTransformers、MoE-Infinity、vLLM UVA/copy 和本方案。

主要指标为 TTFT（首 token 延迟）、TPOT（Decode 每 token 时间）、throughput（单位时间
完成的 token/请求）和 goodput（满足 SLO 的有效吞吐）。所有系统必须使用统一 client、
输入和计时边界，通过正确性门；每个主点至少运行 5 个独立 server process blocks 并报告
置信区间，同时展示相同绝对 arrival rate 和相同相对饱和度。若输出、HBM 或计时边界无法
公平对齐，RQ9 就不能得出“优于外部系统”的结论。

九个问题可以归为四类：RQ1/RQ7 研究路径选择与瓶颈，RQ2/RQ3 研究机制贡献，RQ4/RQ5/RQ6
研究在线控制，RQ8/RQ9 研究长期可靠性与公平比较。它们共同回答系统为什么有效、适用范围、
成本和安全边界，而不只是回答单一配置下是否更快。

## 5. 模型、平台与 claim 分工

### 5.1 核心机制主模型：Qwen

核心机制、动态 KV/专家协同和扩展性主模型固定为：

```text
Qwen3-30B-A3B-FP8
48 MoE layers
128 experts per layer
TRITON_ATTN
Prefill/Decode FULL_SEPARATE CUDA Graph
```

正式专家计划必须在每层选择相同数量 `k` 的冷专家：

```text
|cold_experts(layer_i)| = k, for every MoE layer i
```

专家 ID 可因每层热度不同而不同，但不得用“只卸载前几层”作为主实验。层不均衡方案只
可作为明确标注的诊断消融，不进入主结论。

### 5.2 三系统共同模型：DeepSeek

MoE-Infinity 当前不支持 Qwen block-FP8，因此三系统外部比较使用
DeepSeek-V2-Lite-Chat BF16：

| 模型 | ours | KTransformers | MoE-Infinity | 用途 |
|---|---:|---:|---:|---|
| Qwen3-30B-A3B-FP8 | 支持 | 支持 | 不支持 | 核心机制、动态策略、ours/KT 扩展 |
| DeepSeek-V2-Lite BF16 | 支持 | 支持 | public-demand 支持 | 三系统 equal-HBM 表 |

DeepSeek 外部表不能替代 Qwen 核心机制实验；Qwen 表也不能给 MoE-Infinity 填入推测值。
unsupported 必须写 N/A 和原因。

### 5.3 硬件平台

Platform A 为当前必需平台：

- NVIDIA GeForce RTX 5090 32 GiB；
- Intel Xeon Gold 6530，当前可见 14 physical cores；
- 单 NUMA node；
- 固定 CPU affinity、driver、kernel、CUDA、PyTorch 和功耗采样。

Platform B 数据中心 PCIe GPU 为加分项，不是当前硬门槛。若可用，选择 A100 40/80 GiB、
L40S 48 GiB 或等价设备，只重复关键 crossover、E2 子集和动态 trace。

## 6. 基线与公平性协议

### 6.1 本方案内部变体

- `ours-full`：阶段专用双 Graph + 稳定 VA + 固定 class + 三队列控制器；
- `ours-static`：固定 KV target 和固定专家计划；
- `ours-no-backup`：保留自动预测但关闭 GPU backup；
- `ours-no-heat`：容量动态，但 class 内专家集合冻结；
- `ours-eager`：相同 topology，关闭 CUDA Graph；
- `ours-always-remote`：Prefill/Decode 都 remote-read；
- `ours-always-copy`：每阶段按相同专家集合搬运；
- `ours-full-rebuild`：旧的全量 plan/session/Graph 重建路径，仅用于切换消融；
- `ours-diff-class`：固定 Graph class 的 added/removed/retained 差集路径。

### 6.2 外部与辅助基线

外部与辅助基线用于区分完整系统收益、数据路径收益和控制策略收益。它们不一定都是可部署
方案，但必须使用真实实现和可审计配置，不能只根据论文名称推测其行为。

#### all-GPU / maximum-resident

all-GPU 将模型权重全部放在 GPU；显存不足时，maximum-resident 则尽可能多地保留 GPU
权重。它们既是减少 CPU 访问和权重搬运后的性能上界，也是其他方案的正确性 reference。
这类配置不一定还能容纳大 KV 或高并发，因此不能直接代表实际服务容量。

#### vLLM native pinned UVA

权重驻留在 pinned CPU 内存，GPU 通过 UVA 经 PCIe 直接读取：

```text
GPU kernel -> PCIe -> pinned CPU weight
```

该基线不使用本项目的定制 PTE、alias 或 PARK/RESUME，用于衡量原生 remote-read 路径。

#### vLLM copy-on-forward

每次 forward 前将需要的权重从 CPU 显式复制到 GPU，再执行 kernel：

```text
CPU weight -> H2D copy -> GPU buffer -> kernel
```

它是普通复制基线，用于判断稳定 alias、Graph 内复制和阶段专用路径是否带来额外收益。

#### KTransformers frequency placement

KTransformers 使用 calibration 中的专家访问频率选择 GPU-resident experts，并分别测量
eager 和 CUDA Graph 模式。这是接近 KTransformers 正常使用方式的系统级基线。

#### KTransformers exact-coordinate placement

该变体强制 KTransformers 与本方案使用相同的层号和 expert ID，而不是各自选择专家。
它主要用于 Track A，使差异能够归因于数据路径与执行机制，而不是某一系统恰好选择了
更热门的专家。

#### MoE-Infinity public-demand / public-speculative

public-demand 是公开代码中可以验证的按需加载路径，专家被访问时根据 cache 状态加载或
读取。只有 prediction、correction 和 enqueue 计数均非零，证明预测、提前入队和纠错路径
实际发生时，才允许把对应运行标为 public-speculative；否则即使配置名称包含 speculative，
也仍按 demand-only 报告。

不得把公开 demand-only 结果称为 paper EAM/EAMC。如果为兼容当前环境修改了
MoE-Infinity，必须公开 compatibility patch、extension 源码或二进制 hash、cache/store
配置，以及超过 256 token 后是否进入 fallback。否则无法判断测量对象是原始算法、兼容
补丁还是另一条性能不同的退化路径。

#### oracle KV/expert controller

Oracle 提前知道未来请求和资源需求，可以直接选择理想的 KV 容量与专家放置。它不是可部署
系统，只作为策略上界：自动控制器越接近 oracle，说明预测与响应越有效；差距越大，则说明
控制策略仍有优化空间。

### 6.3 三条评测 Track

三条 Track 回答不同问题，必须分别报告，不能把资源约束、专家选择和控制策略不同的结果
混为同一张性能表。

#### Track A：相同专家坐标

目的：在处理相同专家和相同数据量时，隔离数据路径与执行机制的差异。

所有系统必须使用相同模型、精度、输入 token IDs、逐层冷专家坐标、KV bytes、batch 和
output length。每个 case 还必须报告：

- `selected bytes`：参与卸载或传输的专家参数字节；
- `GPU hot bytes`：仍驻留 GPU 的权重；
- `CPU bytes`：驻留 CPU 的权重；
- `alias backing`：本方案仍保留在 GPU 的物理 backing。

Qwen 主要比较 ours、原生 vLLM 和 KTransformers；DeepSeek 可以加入由相同专家坐标生成的
MoE-Infinity cache manifest。Track A 容易把差异归因于 copy、remote、Graph 或 PTE，
但强制相同专家坐标可能不是每个系统各自的最佳配置，因此不能替代系统级比较。

#### Track B：相同完整系统 HBM

目的：给各系统相同的完整 GPU 显存预算，比较它们采用最佳合法配置时的整体服务能力。

Track B 不要求专家坐标相同，但必须对齐实际 measured peak HBM，而不是名义 offload GiB。
完整预算至少包括：

```text
GPU weights
+ GPU expert cache
+ KV cache
+ retained backing
+ Graph/workspace
+ allocator and framework memory
```

各系统的 placement、threads、cache ratio 和 Graph sizes 只能使用 calibration split 调节；
进入 evaluation 后必须全部冻结，不能看到最终性能后重新选配置。

主结果是 TTFT、TPOT 和 goodput 的 Pareto frontier，而不是强行选择单个“最快”数字。例如
一个配置可能具有更低 TTFT，另一个配置可能在高并发下具有更高 goodput，两者都可能是
非支配点。Track B 衡量完整系统能力，不能把总差距直接归因于单个 PTE 机制。

#### Track C：动态策略

目的：在相同请求压力变化下，比较 KV/专家控制策略，而不是生成外部系统总排名。

需要比较 fixed-low、fixed-high、oracle、auto、auto+backup 和 full joint controller：

- `fixed-low`：始终使用较小 KV target；
- `fixed-high`：始终保留较大 KV target；
- `oracle`：提前知道未来请求；
- `auto`：在线预测并调整；
- `auto+backup`：自动控制并加入 GPU backup；
- `full joint controller`：联合控制 KV、专家容量和热度计划。

所有策略必须使用相同 arrival trace、总 HBM 上界、模型和服务条件。重点报告：

- OOM：是否发生显存不足；
- admission：能够接纳多少请求；
- queueing：请求等待时间；
- migration：迁移的 KV 与专家字节；
- 抖动：是否频繁来回切换；
- SLO goodput：在延迟目标内完成的有效请求数。

Track C 只评价动态控制器，不能用于宣称本方案整体优于某个外部系统。

三条 Track 的控制变量不同：Track A 固定专家坐标以隔离机制，Track B 固定完整 HBM 以
比较系统能力，Track C 固定请求轨迹和 HBM 上界以比较控制策略。例如 Track B 中本方案与
KTransformers 的差距可能同时来自专家选择、KV 容量、Graph、cache、线程、kernel 和 PTE，
因此不能解释为单一 PTE 机制收益；PTE 的独立贡献必须由 Track A 或内部消融证明。

## 7. HBM 预算与档位命名

### 7.1 避免三类“GiB”混淆

正式 artifact 必须区分：

- `Kx`：GPU-resident KV prefix，例如 K2/K4/K8/K10；
- `Cx`：固定 Graph/slot Plan Class，由 class manifest 定义；
- `Hx`：完整系统 measured peak HBM cap；
- `selected_expert_bytes`：计划选择的逻辑专家参数字节；
- `physical_backing_bytes`：去重后真实 GPU backing；
- `remote_expert_bytes`：CPU resident 且被 GPU remote-read 的专家字节。

不得把 C8、8 GiB remote experts 和 8 GiB KV 当作同一个量。

### 7.2 完整预算账本

每个 case 必须保存：

```text
usable HBM
  = device/Torch usable bytes
  - dense and always-hot weights
  - current GPU expert bytes
  - retained alias backing
  - active KV prefix
  - CUDA Graph pool/workspace
  - allocator fragmentation reserve
  - safety headroom
```

同时记录 NVML peak、PyTorch allocator、driver residency 和 native CUDA allocation。
只报告“卸载 4 GiB”而不报告 retained backing 和 Graph pool，不具备 equal-HBM 资格。

### 7.3 预算容差

沿用预注册规则：

```text
tolerance = max(3% * target_peak_HBM, 256 MiB)
```

超过容差的 case 保留 artifact，但从 equal-HBM speedup 中排除。不能看到性能后修改
容差或重新选择 comparison target。

## 8. Workload 设计

### 8.1 冻结 split

split 是把全部实验请求划分为若干互不重叠的数据子集，每个子集只承担一种用途，避免
提前查看正式测试请求后再调参。比如 1,000 条请求可以冻结为：

```text
calibration: requests 1--400
validation:  requests 401--600
evaluation:  requests 601--900
shift:       requests 901--1000
```

#### calibration：只用于调参

calibration 是 cold/hot expert、KV target 到 Plan Class 映射、baseline threads/cache ratio、
Graph sizes 和控制器参数的唯一选择来源，相当于机器学习中的训练集。所有会影响最终性能的
配置都必须在查看 evaluation 结果之前由 calibration 生成并冻结。

#### validation：正式运行前检查

validation 用于确认模型与精度一致、输出通过正确性门、Graph 能 capture/replay、placement
与 manifest 一致、HBM 预算合法，并且没有 UVM fault 或资源泄漏。它可以暴露实现或配置
错误，但不能被反复用于选择最终性能最好的专家和参数。

#### evaluation：只产生论文结果

evaluation 是正式测试集。进入该阶段后，专家坐标、Plan Class 映射、threads、cache ratio、
Graph sizes 和控制器阈值必须保持冻结。不能看到 evaluation 中某个专家更热后再把它放入
GPU 并重跑，否则相当于用正式考试题调参。

例如 calibration 发现专家 3、7、12 最热，于是将其放在 GPU；即使 evaluation 中专家 20
更热，也不能据此修改当前方案。专家 20 是否应被识别，只能作为在线自适应机制的观测，
不能反向改变静态基线。

#### shift：专门测试工作负载变化

shift 的请求领域或长度分布与 calibration 不同。例如从编程请求切换到医学请求，用于测试
原热点是否变冷、新热点多久被识别、EWMA 多久收敛、专家计划是否抖动，以及动态更新是否
真正减少 remote-read。

每个 fixture 必须保存完整 token IDs、来源索引和 SHA-256。token IDs 排除 tokenizer/chat
template 漂移，来源索引记录原数据位置，SHA-256 用于发现输入被修改。现有 v1 DeepSeek/Qwen
fixtures 可以作为新输入的生成材料或在新 manifest 中按 hash 登记，但不得根据已经看过的
qualification 结果重新挑选 evaluation 专家。

### 8.2 受控长度矩阵

| Workload | Prompt | Output | 主要问题 |
|---|---:|---:|---|
| short dialogue | 128 | 32 | TTFT 与切换固定成本 |
| ordinary chat | 512 | 128 | 常见 serving |
| medium context | 2,048 | 128 | Prefill-copy/remote crossover |
| RAG/summary | 8,192 | 128 | 长 Prefill 与 Graph/headroom |
| long generation | 512 | 512 | Decode remote 摊销与 PCIe |

不同 prompt/output 长度突出不同瓶颈：

- short dialogue 的计算量小，TTFT、PARK/RESUME、Graph launch、计划切换、HTTP 和
  scheduler 等固定成本占比最高；
- ordinary chat 代表常见服务请求，用于观察综合性能；
- medium context 的 Prefill 已足够长，适合寻找 copy 与 remote-read 的 crossover：如果
  一次 H2D 后权重被大量 token 复用，copy 可能开始占优；
- RAG/summary 测试长 Prefill、Graph headroom、workspace/KV OOM 风险及 H2D 摊销；
- long generation 包含大量 Decode steps，用于观察 remote-read 累计流量、PCIe/CPU DRAM
  瓶颈，以及 copy-on-forward 是否因重复搬运而恶化。

每个长度点必须先做 backend/Graph support preflight，确认 attention backend 支持该形状、
Graph capture 成功、replay shape 匹配、HBM 预算合法且没有静默降级到 eager。RTX 5090 上
若 8K FULL Graph 不满足 HBM/headroom，应明确报告 `Graph: N/A` 和原因，并把 eager 结果
单独列出；不能内部已经退化为 eager，结果名称仍写 FULL Graph。

### 8.3 并发和到达过程

#### strict batch：B1/B4/B8

strict batch 一次明确组成固定大小的 batch：B1/B4/B8 分别表示一次处理 1/4/8 条请求。
它主要用于研究权重复用、kernel 效率和 Prefill/Decode 路径，不负责模拟真实在线排队。

#### closed loop：C1/C4/C8/C16/C32

closed loop 始终维持固定数量的未完成请求，一个请求完成后客户端立即补充一个。例如 C4
表示保持约 4 个并发请求。它适合测量固定并发下的饱和吞吐、TPOT 和资源占用；C32 仅在
资源允许时运行。

#### open loop：Poisson 与冻结 trace replay

open loop 按外部时间表发送请求，不等待前一个请求完成。Poisson 使用随机到达间隔模拟
在线流量，冻结 trace replay 则按预先保存的时间线重放。如果系统处理不过来，请求会自然
排队，因此 open loop 适合测量 queueing、SLO 违约和 goodput。

负载点使用各系统独立饱和吞吐的 20/40/60/80/90%，同时还要报告相同绝对 arrival rate。
例如系统 A/B 的最大吞吐分别为 100/50 req/s：相同 40 req/s 对应 A 的 40% 和 B 的 80%；
相同 80% 饱和度则分别对应 80/40 req/s。前者回答相同真实流量下谁更能服务，后者回答
接近各自容量上限时谁的退化更合理，两种结果都必须报告。

每个 open-loop 点运行 20--30 分钟并至少使用 5 个 seed，以覆盖随机到达、排队与突发波动。
不能只选择让弱系统 OOM 的绝对负载，也不能只展示让各系统承担不同流量的相对负载。

### 8.4 动态压力 trace

至少包含：

```text
low steady
-> short burst consuming backup
-> sustained high pressure
-> three valid low-pressure rounds
-> idle shrink
-> repeated high/low cycles
-> workload route shift
```

该 trace 模拟一轮完整在线压力周期：

1. `low steady`：低负载稳定运行，建立延迟、HBM、KV 和专家放置基线；
2. `short burst consuming backup`：短突发先消耗 GPU backup，验证 reserve 能否覆盖控制器
   反应时间；
3. `sustained high pressure`：持续高压迫使控制器卸载专家、扩大 GPU-resident KV、调整
   Plan Class，并避免抢占和 OOM；
4. `three valid low-pressure rounds`：连续三轮确认低压不是瞬时波动，以迟滞避免计划反复
   升降；
5. `idle shrink`：到达安全的 scheduler idle 边界后缩小 KV、恢复专家和回退 Plan Class；
6. `repeated high/low cycles`：重复峰谷，检查控制器抖动、HBM 泄漏、延迟退化、状态恢复和
   意外 Graph recapture；
7. `workload route shift`：改变请求领域或路由分布，测试 GPU heat、CPU EWMA 和候选计划
   能否识别新的热门专家。

Qwen 大容量 trace 使用 K2/K4/K8/K10 作为 GPU-resident KV 档位，其名称只近似表示
2/4/8/10 GiB KV，不等于 Plan Class 或 remote expert GiB。每个 tier 的专家数量和具体 ID
必须根据 calibration 与完整 HBM 预算重新生成：

```text
KV target
-> available expert HBM
-> experts retained/offloaded per layer
-> concrete expert IDs
```

不得机械复用 qualification 的 70/80/99/108，因为那是旧路径等价实验人为配对的状态，
不是各 KV tier 的最小卸载量或最优容量策略。旧映射仍可作为固定复现实验，用于验证路径
历史、release/restore 和 Graph 无 recapture，但不得替代正式控制器的 calibration 映射。

## 9. 统一计时边界与指标

### 9.1 用户可见边界

所有系统由同一 localhost、预 tokenized streaming client 测量：

- TTFT：request body 发送完成到首个生成 token 到达；
- ITL：相邻生成 token 到达间隔；
- TPOT：排除首 token 后的平均时间；
- E2E：request body 完成到最终 token；
- throughput：成功输出 tokens / wall time；
- goodput：满足 workload SLO 的请求吞吐。

主表不计 tokenizer 和连接建立；真实文本 API 开销作为附录。不能用 custom Engine step
与 KT HTTP token interval 直接相除。

### 9.2 内部 breakdown

至少分解：

```text
TTFT
  = queue
  + RESUME
  + Prefill H2D
  + Prefill compute
  + PARK / remote map / residency audit
  + first Decode / sampling / serialization
```

动态切换还要单独记录：

- expert candidate selection；
- source refill；
- driver REBIND/RESIZE；
- expert release/restore residency；
- KV CPU->GPU 或 GPU->CPU migration；
- Graph class activation；
- rollback/reconciliation。

稳态结果可以扣除 RESUME/PARK 做机制分析，但论文主 TTFT/E2E 必须包含全部用户可见开销。

### 9.3 必报指标

用户指标：TTFT、TPOT/ITL、E2E、throughput、goodput、timeout/OOM/failure。  
资源指标：peak/steady HBM、CPU DRAM、pinned bytes、PCIe RX/TX、GPU/CPU utilization、
power 和 energy/token。  
机制指标：H2D/remote bytes、Graph capture/replay、PTE groups、PARK/RESUME、KV migrated
bytes、backup hit、prediction error、plan hit rate、wave count、class switches、rollback、
UNKNOWN residency 和 UVM fault。

## 10. 正式实验矩阵

E0--E2 逐级推进：E0 检查系统能否正确、合法地参加比较；E1 验证统一测量流程并提供
maximum-resident 描述性参考；E2 才在相同完整 HBM 预算下承担正式跨系统比较。E0/E1
通过执行检查不代表可以发布 speedup，主排名必须由通过公平性与正确性门的 E2 承担。

### E0：兼容性与正确性

状态：v2 基础兼容性 qualification 已完成并通过；论文级数值正确性门仍待完成。

E0 相当于正式比较前的资格检查，重点不是性能排名，而是确认模型 snapshot、权重、精度、
tokenizer、冻结 token IDs 和生成配置一致，并检查以下项目：

- 三系统在同一 validation 输入上的输出；
- eager/Graph 路径是否真实执行，是否发生静默 fallback；
- KTransformers 是否按冻结的 frequency placement 加载指定专家；
- MoE-Infinity 的 request/token、pending/cancelled 和 KV cleanup counters；
- ours 的稳定 Parameter VA、PTE、物理 residency、UVM fault、session/backing cleanup。

2026-08-24 结果：三系统同一 validation 请求得到相同 32-token hash；KT eager/Graph
各 10/10 成功且逐条一致，26 层均加载 54 个 frequency experts；MoE-Infinity request、
token 和 KV cleanup counters 一致；ours 的 UVM 机制证据按内容 hash 重新登记。DeepSeek
maximum-resident 下 ours CUDA Graph 因 32-GiB HBM headroom 不足写为 N/A，
MoE-Infinity 公开 server 未暴露 Graph 模式也写为 N/A。

一条 32-token 请求一致只说明基础配置对齐，不能证明所有输入和 kernel 都数值等价。“按
内容 hash 重新登记”表示已有机制 artifact 以 SHA-256 纳入 v2 provenance，证明引用内容
未被修改，不等于这些机制在本次 E0 中全部重新运行。Unsupported 必须写 N/A 和原因；
N/A 不表示 0 ms 或实验失败，也不能静默换成 eager 后仍标为 Graph。

E0 尚缺论文级数值正确性门，需要在更广输入上完成 teacher-forced logits、top-1 agreement
和 perplexity 等检查。

### E1：框架开销与 maximum-resident anchor

使用统一 client，在不发生动态切换的 maximum-resident 配置下测量 ours、KT 和
MoE-Infinity。每系统 5 个独立 server blocks，balanced order。该实验给出框架、HTTP、
scheduler 和 sampling 的差异下界。

maximum-resident 表示各系统在合法配置下尽可能多地保留 GPU 权重，并在测量期间冻结 KV
和专家计划；它不要求三个系统使用完全相同的 placement。E1 的差异可能同时来自 HTTP、
scheduler、sampling、Graph/eager、专家 placement、kernel、KV 和框架额外开销，因此不是
单独的 PTE/offload 机制比较。

每个 block 都重新启动独立 server 进程，用于观察初始化、allocator、cache 和 Graph 状态的
进程间波动。balanced order 轮换三系统的执行先后，避免冷机、热机或时间漂移持续偏向同一
系统。15/15 通过表示所有进程完成执行、限速、顺序和退出清理检查。

状态：执行完成。三系统各 5 个独立 blocks，15/15 通过执行、限速、清理和 balanced
order 检查。观测均值为：

| 系统 | TTFT | TPOT | peak HBM |
|---|---:|---:|---:|
| ours | 129.47 ms | 54.22 ms | 30,997 MiB |
| KTransformers | 408.22 ms | 9.96 ms | 31,559 MiB |
| MoE-Infinity | 1,348.69 ms | 306.98 ms | 29,379 MiB |

例如 KTransformers 的 TTFT 较高而 TPOT 较低，说明其首 token 前成本更高、后续 Decode
更快；这只是当前配置的描述，不能直接换算成系统间 speedup。

这些数字暂不进入论文性能表。虽然所有 15 blocks 使用相同冻结输入和请求体，E1 的
exact generation gate 未通过，且 MoE-Infinity 存在进程间输出漂移。必须先完成协议中
预登记的 teacher-forced logits relative-L1、top-1 agreement 和 perplexity gate；此外
E1 本身不是 equal-HBM 比较，主排名仍由 E2 承担。

具体来说，10 个 evaluation prompts 没有任何一个在三系统全部 15 个进程上得到完全相同
的生成序列；ours 和 KT 各自在五次进程间稳定，MoE-Infinity 多数 prompt 存在漂移。自回归
生成中，一个 token 分歧会改变后续全部输入，因此不同输出的延迟可能对应不同计算路径。
Teacher forcing 为每一步提供相同真实 token 前缀，再比较 relative-L1、top-1 agreement 和
perplexity，可以避免早期 token 分歧不断放大，更适合判断底层数值差异是否可接受。

E1 的 peak HBM 也未完全对齐，且执行模式和 placement 不同，所以它只作为框架级
maximum-resident anchor 和 E2 的基础，不承担正式排名。

### E2：equal-HBM 主性能矩阵

E2 是 Track B 的正式跨系统主矩阵。它使用固定策略，不启用在线 KV、热度计划或专家交换，
避免不同请求时刻触发的控制决策混入系统基线。KV target、placement、cache ratio、threads
和 Graph sizes 必须在 calibration 后冻结；动态控制器单独由 Track C 评价。

比较对象和矩阵为：

- DeepSeek：ours、KTransformers、MoE-Infinity public-demand；
- Qwen：ours、KTransformers、vLLM UVA/copy；
- HBM tiers：根据实际 peak 对齐，名义候选 0/2/4/6/8 GiB offload；
- workloads：short、ordinary、RAG、long generation；
- concurrency：C1/C4/C8/C16；
- 每个 cell 5 个独立 process blocks。

每个 cell 是一个固定组合，例如 `DeepSeek x 4-GiB nominal tier x ordinary x C8 x KT`。
五个独立进程用于统计均值、运行间波动和置信区间。不支持的系统/模型组合必须写 N/A，
不能填入推测结果。

名义 0/2/4/6/8 GiB offload 只是候选标签，正式配平必须使用完整 measured peak HBM，包括
GPU 权重、GPU expert cache、KV、retained alias backing、Graph/workspace、allocator 和
框架额外显存。只有实际 peak HBM 落入预注册容差的 case 才能进入 equal-HBM 主表。

E2 输出 TTFT、TPOT 和 goodput 的 Pareto frontier，因为同一个系统可能分别在首 token、
Decode 或高并发有效吞吐上占优，不存在所有指标都最好的唯一配置。HBM 预算不匹配的点仍
保留 artifact，但只能进入附录或诊断表，不能进入 equal-HBM speedup 主表。

### E3：Prefill/Decode 路径 crossover

E3 回答 Prefill 和 Decode 分别应该使用 copy 还是 remote。Qwen Track A 固定相同模型、
精度、KV、请求和逐层 expert IDs，避免专家冷热差异干扰数据路径比较。实验包括：

- hybrid Prefill-copy/Decode-remote；
- always-copy；
- always-remote；
- vLLM pinned UVA/copy；
- KT exact-coordinate CPU experts。

hybrid 在 Prefill 先把权重搬到 GPU，以高本地带宽服务大量 token；Decode 则直接读取
CPU 权重，避免每步重复 H2D。always-copy 两个阶段都复制，计算阶段快但 Decode 可能产生
大量重复搬运；always-remote 两个阶段都远程读取，节省 GPU 权重空间，但长 Prefill 和大
batch 可能反复通过 PCIe 读取同一权重。

vLLM pinned UVA/copy 提供不使用定制 PTE 的原生 remote/copy 对照。KT exact-coordinate
强制 KTransformers 使用与本方案相同的层号和 expert ID，避免 placement 不同导致比较偏差。

实验扫描 prompt、output 和 B1/B4/B8：prompt 越长，Prefill 权重复用通常越高；output 越长，
Decode 重复步数越多；batch 越大，同一权重可服务的 token 越多。crossover 是最佳策略发生
变化的边界，break-even 是两种路径总成本相等的点：

```text
H2D copy cost + GPU compute cost
approximately equals
CPU remote-read cost
```

最终输出以 prompt/output 为轴、B1/B4/B8 为面板的最优策略热力图和 break-even。该实验直接
支撑阶段专用路径的适用范围，而不是依赖整系统总排名间接推断。

### E4：并发与 PCIe 饱和

E4 扫描 concurrency C1--C32，寻找系统从正常扩展转为持续退化的第一个并发点。这里的
C1--C32 表示并发请求数，不是论文 Claim 编号。

必须联合记录：

- remote bytes：每个请求/token 实际读取的 CPU 权重字节；
- PCIe bandwidth：PCIe RX/TX 是否接近平台上限；
- GPU stall：kernel 是否因等待远程数据而降低 SM 利用率；
- CPU bandwidth：CPU DRAM 是否先于 PCIe 饱和；
- TPOT：用户可见的 Decode 退化。

典型 PCIe 饱和表现为 remote 需求继续增加、实际带宽不再增长且 TPOT 持续恶化。但即使
PCIe 尚未达到理论峰值，小粒度访问或远程延迟也可能先造成 GPU stall；CPU DRAM 也可能
更早成为瓶颈。因此不能仅凭单个带宽计数判断。

不同系统的退化来源不同：ours remote-read 可能受 PCIe、CPU DRAM 或 GPU stall 限制；
KTransformers CPU compute 可能受 CPU 核数、内存带宽或线程调度限制；MoE-Infinity 可能
受 cache miss、专家 fetch、存储读取或搬运队列限制。“首个持续退化点”必须在重复实验和
连续并发档位中出现，不能由单次慢请求定义。最终结果应给出 remote-read 的可扩展范围，
以及超过哪个并发或流量后应改用 copy/cache。

### E5：稳定 VA、Graph 与 PTE 消融

状态（2026-08-31）：已在当前 custom UVM 上补跑 stable-VA DESTROY/rebuild 与
non-stable-VA/reallocate+recapture 两臂，各10个独立冷进程、20个plan generation，均通过
输出、VA contract、Graph、fault scan与清理门。B4/B1 rebuild median分别为
8,603.378/9,167.878 ms与11,525.182/13,884.723 ms，non-stable/stable为
1.340x/1.514x。其余eager/Graph、PARK/RESUME和sequential/grouped PTE数值仍来自较早
模块，当前模块整组复验尚未完成。

| 变体 | 隔离因素 |
|---|---|
| eager remote | Decode Graph 收益 |
| remote Graph | stable Parameter VA |
| eager alias Prefill | Prefill Graph 收益 |
| dual Graph | 完整阶段路径 |
| DESTROY/rebuild | retained session/backing 价值 |
| PARK/RESUME | 可逆拓扑 |
| sequential PTE | grouped replace 收益 |
| grouped PTE | 当前驱动路径 |

E5 保持模型、专家集合、请求和 HBM 尽量一致，每次只替换一个机制，以测量独立贡献。

#### eager remote 与 remote Graph

两者都让 GPU 读取 CPU 权重。eager remote 使用普通 kernel launch，remote Graph 使用
Decode CUDA Graph replay；二者比较用于测量 Graph 减少的 launch、scheduler 和 CPU 提交
开销。remote Graph 还验证 Parameter VA 不变时，即使 PTE 指向 CPU-backed 权重，已捕获
Graph 仍能正确 replay。

#### eager alias Prefill 与 Graph Prefill

两者都使用 `Upload VA -> GPU backing <- Parameter VA`，区别是 H2D 和计算是否通过 CUDA
Graph replay。比较结果用于衡量 Prefill Graph 对 H2D 节点提交、kernel launch、CPU 调度
和节点间空隙的改善。

#### dual Graph

dual Graph 分别保留包含 H2D 与 GPU 计算的 Prefill Graph，以及使用 CPU remote-read 的
Decode Graph。它代表完整阶段专用路径，用于验证两个 Graph 能否在稳定 VA、不同 PTE 拓扑
之间反复切换而不 recapture 或错误派发。

#### DESTROY/rebuild 与 PARK/RESUME

DESTROY/rebuild 销毁旧 alias/session/backing，再重建映射、填充数据并在需要时重新捕获
Graph。PARK/RESUME 保留 session、upload alias 和 backing，只把 consumer PTE 在 CPU-backed
权重与 retained GPU backing 之间切换。比较二者用于量化资源复用与可逆拓扑的延迟收益，
同时必须报告为快速恢复而保留的 HBM 成本。

#### sequential PTE 与 grouped PTE

sequential PTE 逐 range 修改并同步；grouped PTE 按 VA block 合并 page mask，以少量批次
提交并统一 tracker/TLB 同步。比较二者用于测量 ioctl、PTE 操作、TLB invalidate、tracker
同步以及切换 median/p95 的减少量。

每个消融必须报告：

- output：确认优化未改变模型结果；
- VA fingerprint：确认 Parameter/upload/KV VA 没有意外变化；
- capture counter：确认切换没有隐式 Graph recapture；
- launch：解释 Graph 对 kernel/节点提交的收益；
- PTE 和 remote map：确认映射实际切到目标物理页；
- residency audit：确认权重真实驻留在声明的 GPU/CPU 位置；
- retained backing：记录 PARK 后仍占用的 GPU backing；
- fault/cleanup：确认没有依赖 UVM fault 修复且资源完整释放。

核心配对关系为：

```text
eager remote     vs remote Graph  -> Decode Graph benefit
eager Prefill    vs Graph Prefill -> Prefill Graph benefit
DESTROY/rebuild  vs PARK/RESUME    -> reversible topology/resource reuse
sequential PTE   vs grouped PTE    -> PTE batching benefit
single-phase     vs dual Graph     -> phase-specific full-path benefit
```

只有输出、VA、PTE、residency、fault 和 cleanup 同时通过，才能把性能变化归因于对应机制。

### E6：Plan Class、expert wave 与切换摊销

比较：

1. 旧全量 plan/session/Graph rebuild；
2. 固定 C2/C4/C6/C8 差集切换；
3. 当前串行逐 wave release/restore；
4. 合并后的全层 batched difference transaction。

分别测量 retained/added/removed、source refill、backing bytes、REBIND、residency、Graph
activation 和 rollback。扫描 plan difference 5/10/25/50/100%。

定义真实摊销点：

```text
requests_to_amortize
  = transition_cost
    / (per-request old-plan cost - per-request new-plan cost)
```

分母非正时报告“无 break-even”。95.9 s 的现有 Qwen 串行 wave 必须作为优化前基线保留。

### E7：专家热度、选择与 workload shift

所有方法保持逐层专家数量相同：

- per-layer frequency-cold；
- per-layer random；
- frozen static plan；
- GPU heat + CPU EWMA；
- oracle evaluation route，仅作上界；
- hot-adversarial shift。

报告 plan hit rate、每层 Jaccard overlap、remote/H2D bytes、切换数、EWMA 收敛时间、
性能恢复和维护 RPC/D2H 开销。异步 D2H 未完成时必须标注 synchronous maintenance。

### E8：MoE-Infinity cache/prefetch

仅在 DeepSeek 共同模型上运行：cold cache、standard warm cache、public-demand、合法的
public-speculative、可选 SSD tier。记录 cache hit、fetch bytes、prediction/correction、
wasted prefetch 和 store I/O。paper EAM 数据只引用原论文，不与本机结果混算 speedup。

### E9：KTransformers placement 与 CPU 消融

比较 per-layer frequency/exact coordinates、eager/Graph、7/14/28 threads、NUMA、overlap
和 mixed chunk。所有调参在 calibration split 完成并冻结。长 Prefill Graph 不支持或
崩溃时保留错误 artifact，报告 eager，不删除失败点。

### E10：三队列与联合 HBM 控制器

这是当前实现更新后的动态核心实验。比较：

- fixed K2；
- fixed K10；
- oracle tier；
- auto grow/shrink，无 backup；
- auto + fixed backup；
- auto + backup + expert waves；
- full joint：KV target + class + heat-based expert set。

使用 K2/K4/K8/K10 重复峰谷和 open-loop trace，验证：

- backup 是否降低 burst admission/TTFT；
- 三个有效低压轮后才 shrink；
- 任意高压事件取消 pending shrink；
- target 上下界和最短驻留时间有效；
- 每层专家数量始终相同；
- Graph 不 recapture；
- stepped 与 direct 到达同状态后的稳态一致；
- 迁移失败时 quarantine/UNKNOWN/reconciliation 正确；
- 最终 goodput、OOM 和 HBM timeline。

显式 2--10 GiB stepped/direct qualification 作为 E10 preflight；正式 E10 必须由自动
控制器根据 trace 决策，不能直接传入人工 target 后称为自动策略。

### E11：Continuous batching / phase epoch

构造新请求持续 Prefill、旧请求持续 Decode 的 trace，比较：

- vLLM 默认 continuous batching；
- ours phase-epoch batching（若实现）；
- ours remote-only fallback；
- KT/MoE-Infinity 原生 scheduler。

报告 queueing、batch formation、GPU idle、starvation 和 goodput。若系统仍 fail-closed，
E11 的合法结论是“scope 限定为 phase-batched serving”，而不是伪造支持结果。

### E12：鲁棒性、soak 与故障注入

状态（2026-08-31）：当前模块的机制级1,000次PARK/RESUME qualification已通过，4/4
parent-block full dual-view coverage和逐轮数据校验均保持成立，退出后成功恢复stock UVM、
GPU 0 MiB。该循环仅3.700 s，不是6小时整模型服务soak；KV峰谷、barrier故障注入和长期
性能/HBM退化检查仍未完成。

- 6 小时为最低，投稿前目标 24 小时；
- 至少 1,000 次 PARK/RESUME；
- 重复 K2/K4/K8/K10 峰谷；
- 合法 barrier 注入 MAP/PARK/RESUME/REBIND/KV migration 失败；
- 路由 shift 和 Graph headroom 压力；
- 检查输出、Xid、PDE fault、UNKNOWN、poison、session/backing 和 GPU 泄漏；
- 验证 custom -> stock module 生命周期。

### E13：KV Cache 自适应档位与初始化固定档位

状态（2026-09-01）：K2/K4/K6 离散自动切换、late-cohort pending grow、全局安全
边界、Plan Class 联动、迁移计时、NVML 监控、三重复聚合和时序图已实现。动态矩阵
与容量包络各有三个 seed × 四个 arm，24/24 case 均通过机制 verifier，合计
840/840 请求成功、无拒绝、OOM、preemption 或 kernel fault。

对比 arm：

- Adaptive：从 K4 启动，根据预测和运行期 KV 压力切换 K2/K4/K6；
- Fixed-K2/K4/K6：只在初始化时选择档位，运行期 target 不变；
- K2/K4/K6 同时对应 cold70/cold80/cold89，KV GPU target 增大时增加专家卸载以
  保持 HBM 预算。

workload 包含低压 Short/B1、高压 RAG-8192/B6、Short/Ordinary/Medium/RAG 动态混合
以及低压→RAG burst→恢复。每个实验记录 req/output throughput、TTFT/TPOT/E2E 及
P95/P99、成功/拒绝/失败、最大观测并发和上下文、三队列占用、Plan Class、
requested migration bytes、resize 墙钟/响应时间和 NVML 利用率。

当前结论是功能 qualification 通过，性能假设未通过：Adaptive 每轮稳定执行
`K4→K2→K6→K4→K2→K4→K2`，六次同步切换平均耗时 17.169 s。完整 trace 吞吐为
0.2163 req/s，只有 global best Fixed-K6 的 0.714×。四臂均完成最大 8,192-token
Prompt/8,208-token context；Adaptive 只证明对低固定档提高同时运行数，没有
观察到独有的 OOM/拒绝避免。修复后的 burst 会登记 pending grow，在 K2→K4 后
准入 RAG/B5 并恢复 K2；但等待已有请求排空使 arrival→admission 平均为 26.519 s。
这些负结果要求后续优先实现异步迁移/后台准备、细粒度 quiescence 和
deadline/admission 容量门，然后再运行更长的 5-seed open-loop trace。

独立 capacity profile 用 RAG-8192/Output-16 扫描 B2/B5/B6/B7/B8/B10，三重复
fully-resident 上限为 Adaptive/Fixed-K2/Fixed-K4/Fixed-K6 = 7/2/5/7，最大 live
context 为 57,456/16,416/41,040/57,456 tokens。Adaptive 相对低固定档能承载更多
同时长上下文请求，但没有超过 Fixed-K6；capacity 吞吐为 global best Fixed-K4 的
0.702×。B8/B10 超额请求均排队完成，因此这不是物理 OOM/拒绝上限。

本轮另有 18/288 跨臂完整输出 hash 分叉，虽然 288/288 首 token 一致且 verifier
按预注册诊断规则为 `status=ok`，但投稿前仍必须补 teacher-forced 数值门。权威结果为
[`../experiments/eurosys_moe_offload_evaluation/RESULTS_E13_KV_ADAPTIVE_20260901.md`](../experiments/eurosys_moe_offload_evaluation/RESULTS_E13_KV_ADAPTIVE_20260901.md)，
复现入口为
[`../experiments/reproduce/REPRODUCE_E13_KV_ADAPTIVE.md`](../experiments/reproduce/REPRODUCE_E13_KV_ADAPTIVE.md)。

## 11. 正确性与 fail-closed 门

每个正式 case 至少检查：

- greedy token IDs 与同框架 reference 一致；
- 跨框架使用 teacher-forced logits/top-1/perplexity 门，不能只看文本相似；
- Parameter、upload、source 和 KV VA fingerprint 稳定；
- Graph certificate、capture/replay counter 与 active class 一致；
- selected、remote、GPU backing、KV GPU/CPU bytes 精确；
- 三队列互斥且总 block 守恒；
- 每层 remote expert 数量相同，ID 属于冻结计划；
- removed expert 回收前不再被旧 epoch 使用；
- unknown residency 和 manual migration fault 为 0；
- 回滚后 driver、residency、lifecycle 和 generation 完整恢复；
- shutdown 后 session/backing/Graph/GPU/UVM client 全部释放；
- kernel log 无 Xid、`FAULT_PDE`、Oops、BUG 或 panic。

任何 correctness、fault、dirty GPU 或 budget gate 失败的 case 均不得进入 speedup，
但 artifact 必须保留。

## 12. 统计方法与环境控制

### 12.1 重复

- 机制 microbenchmark：每点至少 10 个独立冷进程；
- 系统主矩阵：每个 model/system/HBM tier 至少 5 个独立 server process；
- open-loop/dynamic trace：至少 5 个 seed；
- 固定 warmup 不计入性能，但计入正确性。

同一 server 内 48 个 Decode step 是时间样本，不是 48 个独立系统重复。

### 12.2 顺序与报告

- 使用 balanced block order 或 Latin square；
- tier 在奇偶 repeat 中正序/逆序；
- 系统启动顺序轮换；
- 报告 median、p95、p99 和 bootstrap 95% CI；
- speedup 使用 paired process-block bootstrap；
- 跨 workload 几何平均，同时保留原值；
- 不删除性能 outlier；只按预注册 invalid reason 排除；
- 同时报告成功率。

### 12.3 环境

每 100--200 ms 记录 clock、power、temperature、throttle reason、PCIe、NVML HBM 和
GPU utilization。记录 CPU affinity、NUMA、page-locked bytes、driver/kernel/CUDA、
所有 commits、dirty patches 和二进制 SHA-256。

## 13. SLO 与 goodput

沿用 v1 的初始 SLO，允许在 v2 冻结前只根据 calibration 调整一次：

| Workload | TTFT SLO | TPOT SLO |
|---|---:|---:|
| short dialogue | 2,000 ms | 100 ms |
| ordinary chat | 5,000 ms | 150 ms |
| medium context | 10,000 ms | 175 ms |
| RAG/summary | 30,000 ms | 200 ms |
| long generation | 5,000 ms | 150 ms |

SLO 冻结后不得根据 evaluation 结果修改。goodput 同时要求请求成功、输出正确并满足
TTFT/TPOT 两个门。

## 14. Artifact 与新协议

### 14.1 新鲜度规则

`eurosys-pa-v1-20260823` 的 protocol 和 provenance 已冻结，保存其历史意义。由于本文、
实现和实验矩阵已经更新，正式新结果必须：

1. 生成新的 protocol ID；
2. 重新计算本文和技术设计 SHA-256；
3. 重新登记或生成 Qwen/DeepSeek fixtures；
4. 冻结 plan-class manifests、逐层 expert plans 和 KV/class mapping；
5. 使用新的不可覆盖 run root；
6. 只从顶层 `verification.json: status=ok` 生成论文表图。

现有 qualification artifact 可以作为 preflight input，但不能复制进新 run root 冒充
正式重复。

### 14.2 目录

```text
experiments/eurosys_moe_offload_evaluation/artifacts/<paper-run-id>/
  protocol.json
  run_status.json
  provenance/
    git_commits.json
    dirty_patches/
    binaries.sha256
    environment.txt
    model_manifest.json
    fixture_manifest.json
    plan_manifests/
  cases/<experiment>/<track>/<system>/<model>/<budget>/<repeat>/<attempt>/
    command.txt
    env.json
    client.json
    server.log
    nvml.csv
    pcie.csv
    residency.json
    graph_certificate.json
    controller_trace.jsonl
    kernel.log
    returncode.txt
    verification.json
  verification.json
  summary.json
```

失败重试必须创建新 attempt。顶层 verifier 检查完整 cell 数、budget eligibility、输出、
fault、provenance 和重复结构后才发布 `status=ok`。

## 15. 执行优先级

### P0：让动态路径可测

1. 合并串行 expert waves；
2. 为 Qwen 完成固定 class 四档 qualification；
3. 跑重复峰谷和最短 1 小时 joint-controller smoke；
4. 决定 E11：实现 phase epoch，或冻结 phase-batched claim；
5. 保持已冻结的 v2 protocol/fixtures 不变，所有后续 reporting amendment 单独登记 hash。

### P1：核心机制

1. E3 phase crossover；
2. E5 Graph/VA/PTE 消融；
3. E6 full rebuild/difference/batched wave；
4. E7 heat/shift；
5. 每点 10 个冷进程。

### P2：系统主矩阵

1. E1 五重复已完成；收尾 teacher-forced 数值门；
2. E2 DeepSeek 三系统 equal-HBM；
3. E2 Qwen ours/KT/vLLM；
4. E4 concurrency/PCIe；
5. 生成 Pareto 和 CI，不提前选择赢家。

### P3：动态与鲁棒性

1. E10 自动 K2/K4/K8/K10 长 trace；
2. backup sizing 与预测误差；
3. E11 scheduler；
4. E12 soak/fault injection；
5. E13 异步迁移、细粒度quiescence、deadline/admission容量门与5-seed open-loop复验；
6. 可选 Platform B。

## 16. 最小可投稿门槛

至少满足：

1. Qwen 核心机制 E3/E5/E6，逐层对称计划；
2. 一个完整三系统共同模型 equal-HBM 主表；
3. Qwen ours/KT/vLLM 扩展表；
4. E1/E2 每主点 5 个 process blocks；
5. TTFT/TPOT/goodput、HBM、PCIe、CPU/GPU utilization 和 95% CI；
6. all-GPU、UVA、copy、always-remote、always-copy 和 eager/Graph 消融；
7. 固定 class 差集与 batched wave 切换结果；
8. 自动三队列至少 5-seed 重复峰谷 trace；
9. heat/shift 结果，或从主 claim 删除自适应专家；
10. mixed batching 实现，或在标题、摘要和结论中明确 phase-batched 范围；
11. 6 小时以上 soak、故障注入、fault scan 和清理验证；
12. 完整、可独立复验的 artifact。

若 E2 三系统公平性或正确性不满足，论文仍可作为机制论文投稿，但必须删除跨系统总
speedup 主张，改为稳定 VA、阶段专用路径和动态 HBM 协同的机制评测。

## 17. 结果解释规则

- 不把 qualification 的单次数字写成论文 speedup；
- 不把 stepped/direct 稳态等价解释为切换免费；
- 不把 K2/K4/K8/K10 与 C2/C4/C6/C8 混淆；
- 不把逻辑 selected bytes 当作真实 HBM freed；
- 不把不同计时边界相除；
- 不把 unsupported 当作 0 ms；
- 不把 MoE-Infinity public-demand 称为 paper EAM；
- 不在 evaluation split 上选择 cold experts 或 baseline 参数；
- 不隐藏 OOM、timeout、Xid、rollback 和 dirty-GPU attempt；
- 不把同 server 内 token steps 当作独立重复；
- 主 TTFT/E2E 必须包含 RESUME/PARK 和在线控制开销；
- 稳态 compute 和一次性 transition 必须分列；
- mixed batching、多 GPU、异步 heat D2H 未完成时保持对应 limitation。

## 18. 当前入口与证据

技术设计：

- [`moe_phase_aware_offload_technical_design.md`](moe_phase_aware_offload_technical_design.md)
- [`cudagraph_static_alias_transport.md`](cudagraph_static_alias_transport.md)

正式评测框架：

- [`../experiments/eurosys_moe_offload_evaluation/README.md`](../experiments/eurosys_moe_offload_evaluation/README.md)
- [`../experiments/eurosys_moe_offload_evaluation/protocol_template.json`](../experiments/eurosys_moe_offload_evaluation/protocol_template.json)
- [`../experiments/eurosys_moe_offload_evaluation/run_formal_e1.sh`](../experiments/eurosys_moe_offload_evaluation/run_formal_e1.sh)
- [`../experiments/eurosys_moe_offload_evaluation/run_formal_e2.sh`](../experiments/eurosys_moe_offload_evaluation/run_formal_e2.sh)

当前模块E5/E12补充qualification：

- [`../experiments/eurosys_moe_offload_evaluation/RESULTS_E5_E12_CURRENT_MODULE_QUALIFICATION_20260831.md`](../experiments/eurosys_moe_offload_evaluation/RESULTS_E5_E12_CURRENT_MODULE_QUALIFICATION_20260831.md)
- [`../experiments/eurosys_moe_offload_evaluation/artifacts/e5-destroy-rebuild-current-module-20260831T131000Z-attempt-01-formal/verification.json`](../experiments/eurosys_moe_offload_evaluation/artifacts/e5-destroy-rebuild-current-module-20260831T131000Z-attempt-01-formal/verification.json)
- [`../experiments/eurosys_moe_offload_evaluation/artifacts/e5-nonstable-va-current-module-20260831T133200Z-attempt-01-formal/verification.json`](../experiments/eurosys_moe_offload_evaluation/artifacts/e5-nonstable-va-current-module-20260831T133200Z-attempt-01-formal/verification.json)
- [`../experiments/eurosys_moe_offload_evaluation/artifacts/e12-dual-view-soak-20260831T130345Z-attempt-01-qualification/verification.json`](../experiments/eurosys_moe_offload_evaluation/artifacts/e12-dual-view-soak-20260831T130345Z-attempt-01-qualification/verification.json)

E13 KV 自适应档位动态与容量三重复 qualification：

- [`../experiments/eurosys_moe_offload_evaluation/RESULTS_E13_KV_ADAPTIVE_20260901.md`](../experiments/eurosys_moe_offload_evaluation/RESULTS_E13_KV_ADAPTIVE_20260901.md)
- [`../experiments/eurosys_moe_offload_evaluation/artifacts/e13-kv-adaptive-20260901-attempt-10-formal-burst-complete/verification.json`](../experiments/eurosys_moe_offload_evaluation/artifacts/e13-kv-adaptive-20260901-attempt-10-formal-burst-complete/verification.json)
- [`../experiments/eurosys_moe_offload_evaluation/artifacts/e13-kv-adaptive-20260901-attempt-10-formal-burst-complete/analysis/e13_summary.json`](../experiments/eurosys_moe_offload_evaluation/artifacts/e13-kv-adaptive-20260901-attempt-10-formal-burst-complete/analysis/e13_summary.json)
- [`../experiments/eurosys_moe_offload_evaluation/artifacts/e13-kv-adaptive-20260901-attempt-13-capacity-formal/verification.json`](../experiments/eurosys_moe_offload_evaluation/artifacts/e13-kv-adaptive-20260901-attempt-13-capacity-formal/verification.json)
- [`../experiments/eurosys_moe_offload_evaluation/artifacts/e13-kv-adaptive-20260901-attempt-13-capacity-formal/analysis/e13_summary.json`](../experiments/eurosys_moe_offload_evaluation/artifacts/e13-kv-adaptive-20260901-attempt-13-capacity-formal/analysis/e13_summary.json)

v2 E0/E1 正式结果：

- [`../experiments/eurosys_moe_offload_evaluation/RESULTS_E0_E1_20260824.md`](../experiments/eurosys_moe_offload_evaluation/RESULTS_E0_E1_20260824.md)
- [`../experiments/eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/summary.json`](../experiments/eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/summary.json)
- [`../experiments/eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/cases/E0/verification.json`](../experiments/eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/cases/E0/verification.json)
- [`../experiments/eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/cases/E1/output_divergence_diagnostic.json`](../experiments/eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/cases/E1/output_divergence_diagnostic.json)

历史 qualification 的独立 artifact 已按用户要求清理。当前 E0 引用的四份机制证书已按
原 SHA256 收编进当前 v2 run；它们只用于保持当前结果的 provenance，不恢复为独立
历史实验：

- [`../experiments/eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/provenance/retained_evidence/manifest.json`](../experiments/eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/provenance/retained_evidence/manifest.json)
- [`../experiments/eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/provenance/historical_results_cleanup.json`](../experiments/eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/provenance/historical_results_cleanup.json)

## 19. 最终论文叙事

最合适的 EuroSys 主线是：

> 稳定虚拟地址使同一 MoE 模型能够在不替换 Parameter/KV tensor、也不重新捕获 CUDA
> Graph 的前提下切换阶段专用权重路径；固定 Plan Class 和差集事务降低专家拓扑更新
> 成本，三队列 KV pool 则把 GPU 上的短时 reserve、长期 KV 容量和专家 cache 纳入同一
> HBM 控制循环。

论文需要用 Qwen 证明机制和真实 2--10 GiB 动态范围，用 DeepSeek 共同模型完成合法的
三系统公平比较，并用长期 trace 证明控制器价值。现有结果已经证明关键机制可行，但在
batched wave、长期联合闭环、正式 equal-HBM 矩阵和 mixed batching 完成前，不应宣称
已经得到最终在线 serving speedup。

## 20. 后续优化方案

当前热度控制面采用同步维护路径：仅在 scheduler idle 且 batch queue 排空后同步 GPU，
通过 RPC 将 heat buffer 搬到 CPU，由 EngineCore 计算 EWMA、临时生成逐层 swap，并立即
执行 `demote -> promote`。该实现结构简单、内存开销低，适合作为功能验证、故障定位和
同步性能基线，但 maintenance 停顿包含 GPU 同步、D2H、CPU 规划、多次 RPC、逐专家
release/restore 及审计开销；连续高负载下还可能长期没有可用的维护窗口。逐专家执行也
不是完整计划的原子发布，失败时可能留下保守但部分完成的 remote 状态。

长期目标是异步 CPU shadow-plan 控制面：GPU 双 heat buffer 轮换后，通过 pinned D2H
把已完成窗口异步送入请求结果通道；独立 CPU planner 在后台维护 EWMA，并生成带
generation 的 candidate plan。控制器使用整体预期收益、Jaccard 差异、最短驻留时间和
最大交换数抑制计划抖动，提前计算 `retained/added/removed` 差集和事务描述，在安全边界
通过 grouped `RESIZE/REBIND` 一次提交目标拓扑，并同步切换对应 GraphExec 与 certificate。
COMMIT 前 active plan 始终有效；失败执行 ABORT，提交后审计失败则反向回滚。

不应为 active plan 和 shadow plan 完整复制两份 GPU 物理 backing。CPU 侧可以双缓冲完整
计划元数据，GPU 侧应原位复用 retained backing，只为 added 部分预分配资源，并使用
inactive bank 或少量 staging slot 准备新增数据。即：双缓冲计划状态，增量准备物理资源，
原子发布最终拓扑。现有同步路径保留为调试、降级和对照基线。

候选计划只有同时满足以下条件才允许发布：候选计划和所需资源已经准备完成；新旧计划
差异或预期收益达到阈值；候选计划依赖的 active generation、Plan Class、HBM 预算和
Graph certificate 仍然有效；系统已经到达经过审计的安全切换边界。在异步 D2H、后台
planner 和事务化整计划切换全部完成并通过长期 workload qualification 前，论文不得宣称
热度返回、计划生成或专家放置更新与推理完全重叠。
