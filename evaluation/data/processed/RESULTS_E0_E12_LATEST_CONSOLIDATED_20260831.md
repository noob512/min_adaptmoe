# E0--E14 最新实验结果统一汇总

日期：2026-08-31--2026-09-13（Asia/Shanghai）
最后更新：2026-09-13（Asia/Shanghai）

## 1. 汇总规则

本文按 E0、E1、E2……E14 的编号顺序汇总截至 2026-09-03 已取得的最新可用结果。

- E2、E3、E10 使用最新 C1/C16、全覆盖双页表单轮结果；E2 的较早 C1/C16 六机制以及
  E3 的较早同坐标 H/A/R、G/B/C 和 native copy/UVA qualification 另表保留，只作辅助
  上下文，不与最新主表混成同一轮；
- E2/E3另保留2026-09-03原生vLLM pinned Native UVA CUDA Graph补充轮：E2只对Decode
  启用Graph，E3启用Prefill/Decode双Graph；该轮不与ours拼成同一进程或equal-HBM实验；
- E3最新补做Native/Ours近等量卸载双Graph对照：十个B1/B16坐标的实际卸载字节差不超过
  10%，两臂Prefill/Decode均为`FULL_SEPARATE`，完整输出hash 10/10一致；
- E3另补做Graph-enabled Native copy/Ours近等量对照：唯一单轮中Native copy 9/10、
  Ours 10/10成功；九个有效配对点的两阶段均为Full Graph，Native的RAG/B16在完整
  Prefill Graph捕获时达到容量边界，保留为N/A；
- E3 KTransformers补做请求形状感知调优：最终只采用一个10/10完整轮次，使用7个expert
  线程但保留14核进程affinity、Decode Graph、FlashInfer/TRITON容量选择和两个大B16
  Prefill fallback；用户要求停止重复实验，`repeat-2/repeat-2b`部分artifact不纳入；
- E2还补做2026-09-03提高Native卸载量的同轮Ours/Native对照，并在最新敏感性轮中把
  Ours每层再多卸载1个专家；最新cold2/6/20与Native实际卸载量相差不超过10%；
- E4、E6、E7和E10使用2026-09-02最近一次成功复验；
- E5 使用2026-09-03当前模块的四项核心机制消融；各性能臂3个冷进程，Graph/eager性能
  臂关闭逐forward trace；
- E12 同时保留较早的 1,000-replay smoke 和当前模块的 1,000-cycle PARK/RESUME；
- E13 使用按当前复现文档重跑且verifier通过的双阶段CUDA Graph K2/K6/Adaptive回环
  `e13-short-b64-b128-full-separate-roundtrip-REPRO-20260902-attempt-02`；
- E13另补充同一RAG-8192/B6高压trace下的Fixed-K2、Fixed-K6和Adaptive-K6 eager三臂
  容量实验；该轮只用于验证K2容量不足时的批次吞吐收益，不与双Graph低压轮混算；
- E13进一步在完全相同的RAG-8192→16 eager协议下补齐B1/B5/B6、Fixed-K2/K4/K6与
  Adaptive四臂压力矩阵；切档与目标档warmup排除，该矩阵作为三档crossover曲线的数据源；
- E13最新把同一四臂压力矩阵扩展到B1--B8；B8揭示K6达到自身容量边界后吞吐回落，
  因此完整曲线不强制单调K2→K4→K6，也不把预设档位写成事后最优；
- E14 先记录同一32请求集的Prefill+Decode路由频率，按每层最低命中数重选cold-29，再以
  无逐forward wrapper/trace的纯eager UVA按`DWWDDW`完成两臂各3个冷进程、每点3轮；
  两臂卸载6.117/6.188 GiB；
- E6、E7、E11 使用各自后续专项 qualification，而不是 2026-08-27 的早期状态；
- E9使用2026-09-02唯一一次5-block formal结果；E8没有新的通过结果；E11已按用户要求
  终止，只保留离线审计；
- 失败、阻塞和仅 smoke 的实验明确保留原状态，不写成通过；
- 不把不同日期、模块或协议的 artifact 拼成同一轮统计实验。

当前成功artifact横跨三个主要custom UVM代际，不能直接混算跨实验性能变化：

```text
E2/E3历史D/E4
           srcversion 68A3CA3F7E2D57082767C59
           SHA256 6965f8ee1a22029c5f83d602093318b3cca5241fa6c2ecb3839552d08e974547
E7/E10     srcversion 18E6563D21D9D2937D99EDF
           SHA256 90fb5a7d9852a754ecf0358700d36dbcd04abd19886753ee5486401a937e5e15
E3近等量ours/E5/E13/E14
           srcversion D158DB985980F4A580DB740
           SHA256 4168020f14fc3217f8d217e4fa8898299e4d81c0c18653c98cce9d73346afdbe
```

E6、E11、E12及历史消融仍以各自artifact中的模块和环境为准。相同日期也不代表
相同进程、协议或统计重复。

## 2. 总体状态

| 实验 | 最新状态 | 当前可支持的核心结论 | 证据边界 |
|---|---|---|---|
| E0 | 基础 qualification 通过 | 三系统可执行同一冻结请求并得到相同32-token hash | 不是性能排名 |
| E1 | 执行完成，publication gate 暂停 | 15/15 process blocks 完成 | 跨系统完整输出不一致，不能发布speedup |
| E2 | ours C1/C16、Native UVA Decode Graph及近等量卸载对照均通过；Native双Graph全矩阵不可用 | Ours改为cold2/6/20后，在实际字节相差不超过10%的8/8点TPOT仍低1.33×--3.42× | 有效近等量表仍为Native eager Prefill；双Graph复核在Long/C1 Prefill replay失败；RAG两点hash不同；每cell单样本 |
| E3 | 最新D臂10/10、Native UVA/Ours近等量双Graph 20/20通过；Native copy/Ours为19/20；调优KT单轮10/10 | 全覆盖双页表与双Graph正确；调优KT的RAG/B16相对旧KT的P/D/E2E快102.70×/3.60×/26.53×；相对copy的Decode在8/9有效点更低 | KT与vLLM是CPU expert对GPU whole-Parameter offload的系统级比较；Native copy RAG/B16容量N/A；均为单样本 |
| E4 | C1--C32各5个process block复验通过 | 未观察到预注册TPOT持续退化；峰值PCIe RX为4.698 GiB/s | 只覆盖固定remote计划 |
| E5 | 四项核心机制、每性能臂3冷进程，统一verifier通过 | P/D直接切换达2,015×--2,260×；Graph使Prefill/Decode快1.24×--1.76×/2.13×--2.42×；stable VA快1.35×--1.49×；prebuild使RESUME快26.87×--29.76× | P/D直接切换依赖prebuilt双视图；Graph/eager的B4跨臂hash不同；三冷进程不是跨硬件普遍结论 |
| E6 | 三臂单轮qualification通过 | batched difference明显快于full rebuild和多wave串行切换 | 无统计重复 |
| E7 | 六臂独立冷进程qualification通过 | 自适应降低remote assignment，但同步维护使cycle吞吐下降40.35% | 每arm仅一次 |
| E8 | 阻塞 | provenance门正确阻止无fresh receipt运行 | 无性能数据 |
| E9 | 5-block formal execution qualification通过 | 35/35冷进程、70/70 phase完成；7线程和frequency placement最优 | uniform/no-mixed改变数值路径，不具备跨臂exact或teacher-forced质量等价证据 |
| E10 | scheduler-boundary压力序列单轮qualification通过 | 完成2→4→6 GiB KV/专家同步切档，三档双页表100%覆盖 | 不是自动联合控制器formal；切档仍需543--565 ms |
| E11 | verifier `status=error`，已终止 | phase epoch能避免mixed step，hybrid fail-closed有效 | phase-specific与remote-only输出不一致；只允许离线审计 |
| E12 | 1,000-replay smoke和1,000-cycle PARK/RESUME通过 | Graph回放、完整双页表往返和生命周期清理通过 | 不是6小时整模型soak或故障注入 |
| E13 | 双阶段Graph回环及同请求B1--B8四臂压力矩阵通过 | RAG-8192中B1--2最佳为K2、B3--5为K4、B6--7为K6；B8因K6也超容量而回到K4 | 压力矩阵为eager且切档/warmup被排除；Adaptive预设B8仍为K6，比当点最佳K4低4.20%；单冷进程 |
| E14 | direct-forward六冷进程formal与全部趋势门通过 | B1两臂相差-4.2%/+0.6%；B32逐层卸载相对整层卸载的Prefill/Decode吞吐高132.4%/63.7%，线性扩展效率为99.2%/100.3%对40.9%/61.7% | 同请求集选冷和评测是in-distribution oracle；B1不是预注册统计等价检验；Decode优势从B16出现 |

按实验编号计，当前有12项至少取得一条通过验收的功能或机制证据：E0、E2、E3、E4、E5、
E6、E7、E9、E10、E12、E13、E14。E1完成了执行但publication gate保持暂停；E8在provenance
门前阻塞；E11完成了诊断运行但正确性验收未通过。这里的“12项有证据”不等于12项
论文级完整完成：E2/E3/E6/E7/E10/E13仍主要是单轮qualification；E14已完成3+3个冷进程
并通过趋势门，但同请求集选冷不能证明跨workload泛化；E5已补齐四项核心机制但每臂仅
3个冷进程，E9的两个numerical-path臂还没有teacher-forced质量等价证据，E12也只
满足机制次数门。完整论文主矩阵、动态长trace和鲁棒性实验仍未完成。

## 3. E0：兼容性与基础正确性

DeepSeek-V2-Lite-Chat BF16 的同一冻结 validation 请求在 ours、KTransformers 和
MoE-Infinity 上得到相同的32-token输出：

```text
69cd56b10352d7b1b4478134f1455d6cfbd6509a7e08d9fc100b4ac433bf9aa6
```

三方基础执行、接口和输出门通过。ours 的 DeepSeek maximum-resident Graph 因32-GiB卡
缺少 headroom 合法记为 N/A；这不是失败，也不能以 eager 冒充 Graph。

状态：基础 compatibility qualification 通过，但 E0 不承担性能排名；更广输入上的
teacher-forced logits、top-1 agreement和perplexity论文级数值门仍未完成。

来源：[RESULTS_E0_E1_20260824.md](RESULTS_E0_E1_20260824.md)

## 4. E1：maximum-resident anchor

三系统各完成5个独立 server process blocks，共15 blocks、150条 measurement 请求。

| 系统 | TTFT | TPOT | E2E | output tok/s | peak HBM |
|---|---:|---:|---:|---:|---:|
| ours | 129.47 ms | 54.22 ms | 1810.24 ms | 17.66 | 30,997 MiB |
| KTransformers | 408.22 ms | 9.96 ms | 717.09 ms | 44.50 | 31,559 MiB |
| MoE-Infinity | 1348.69 ms | 306.98 ms | 10865.13 ms | 2.95 | 29,379 MiB |

这些数值只是描述性 anchor。10个 evaluation prompts 中，没有一个在全部15 blocks间得到
相同完整输出；此外三系统的 HBM、执行模式和 placement 也未配平。因此 E1 的执行状态为
completed，但跨系统 correctness/publication gate 为 held，不能计算正式speedup。

来源：[RESULTS_E0_E1_20260824.md](RESULTS_E0_E1_20260824.md)

## 5. E2：C1/C16 全覆盖双页表与多机制对照

### 5.1 最新全覆盖双页表主结果

最新ours使用DeepSeek-V2-Lite-Chat BF16，只运行C1、C16，每个场景一次独立冷进程。
C1和C16均为4/4运行通过；Long/C1的`grouped_token_events=0`，旧轮的流式协议问题已消失。
pinned-UVA仍复用2026-08-31独立冷进程基线，因此倍率是跨轮描述性比较。

| 场景 | ours E2E | eager pinned-UVA remote E2E | 加速 | ours单臂验收 |
|---|---:|---:|---:|---|
| Short/C1 | 309.88 ms | 3012.17 ms | 9.72× | 通过 |
| Ordinary/C1 | 1052.93 ms | 9089.28 ms | 8.63× | 通过 |
| RAG/C1 | 1851.91 ms | 9401.91 ms | 5.08× | 通过；跨臂hash不同 |
| Long/C1 | 4116.30 ms | 47566.72 ms | 11.56× | 通过 |
| Short/C16 | 724.68 ms | 2352.90 ms | 3.25× | 通过 |
| Ordinary/C16 | 4841.16 ms | 11387.91 ms | 2.35× | 通过 |
| RAG/C16 | 21198.94 ms | 86402.18 ms | 4.08× | 通过；跨臂hash不同 |
| Long/C16 | 19719.94 ms | 41269.70 ms | 2.09× | 通过 |

这里的“严格通过”是每个 arm 自身的运行、请求长度、流式事件、故障和清理验收，不自动
等于跨 arm 输出一致。逐 token hash 审计结果是：all-GPU、pinned-UVA、copy-on-forward
三种原生 vLLM 路径在四类 workload 上相互一致；最新 ours 的 Short、Ordinary、Long 与
原生路径一致，但 RAG/C1 和 RAG/C16 不一致。因此应分两层报告：

- 按单臂运行、流式、故障和清理门，8/8点通过，跨轮ours/pinned-UVA E2E比为
  `2.09×–11.56×`；
- 再要求跨臂exact-token一致后排除两个RAG点，剩余6个有效点；这6点的范围仍为
  `2.09×–11.56×`。

### 5.2 同一冻结请求口径下的其他机制数据

下表汇总 2026-08-31 attempt 01 的参考臂和 attempt 02 的最新 ours；单位为 ms，均为单个
冷进程样本。`—` 表示本次批次没有完成该点，不代表机制普遍不可运行。

- all-GPU：原生 vLLM eager，全部权重常驻GPU，不卸载；
- pinned-UVA：原生 vLLM eager，卸载参数始终由GPU远程读取；
- copy-on-forward：原生 vLLM eager，每次forward把所需CPU参数复制到GPU；
- KTransformers：按冻结冷专家计划进行GPU/CPU混合放置；
- MoE-Infinity：使用全局按需专家缓存与offload store。

| 场景 | ours，全覆盖双表+Graph | all-GPU eager | pinned-UVA eager | copy-on-forward eager | KTransformers | MoE-Infinity |
|---|---:|---:|---:|---:|---:|---:|
| Short/C1 | 309.88 | 2796.42 | 3012.17 | 2263.21 | 744.47 | 11736.57 |
| Ordinary/C1 | 1052.93 | 10670.23 | 9089.28 | 9287.99 | 1700.49 | — |
| RAG/C1 | 1851.91† | 9241.32 | 9401.91 | 9307.59 | 3022.12 | — |
| Long/C1 | 4116.30 | 35879.16 | 47566.72 | 36868.29 | 5510.82 | — |
| Short/C16 | 724.68 | 3185.41 | 2352.90 | 2423.91 | 13312.23 | — |
| Ordinary/C16 | 4841.16 | OOM | 11387.91 | 10039.89 | 53875.96 | — |
| RAG/C16 | 21198.94† | OOM | 86402.18 | 21757.48 | 183044.19§ | — |
| Long/C16 | 19719.94 | OOM | 41269.70 | 39709.86 | — | — |

说明：

- `†`：ours 的 RAG 输出长度和单臂 verifier 通过，但 token hash 与三种原生 vLLM
  路径不一致，不能进入 exact-token 性能排名；
- `§`：KTransformers RAG/C16 留有完整 runner 性能结果，但本批次没有生成 case-level
  verification receipt，因此只列作 raw diagnostic；KTransformers 其他已完成点也尚未
  通过完整跨框架 generation gate；
- all-GPU 的 Ordinary/RAG/Long C16 分别在正式固定请求下遇到172/684/342 MiB临时分配
  OOM，属于该固定容量下的 N/A；
- MoE-Infinity 本批次只完成 Short/C1，其他点未完成，不用早期结果填入本轮空格。

在同时满足 ours 单臂严格验收、原生路径完成且跨臂 exact-token 一致的点上，描述性
E2E 比值为：

| 对照机制 | 有效配对点 | ours相对对照的E2E加速 | 缺口 |
|---|---:|---:|---|
| pinned-UVA eager remote | 6 | 2.09×–11.56× | 跨轮、非equal-HBM，卸载集合不同 |
| copy-on-forward eager | 6 | 2.01×–8.96× | 跨轮、非equal-HBM，按完整Parameter复制 |
| all-GPU eager | 4 | 4.40×–10.13× | 跨轮；C16除Short外均容量N/A |

这些比值是相同冻结请求上的单样本描述，不是单项机制消融。KTransformers 和
MoE-Infinity 因跨框架 generation gate 或本批次覆盖不完整，不计算正式加速比。

### 5.5 Ours每层多卸载一个专家后的近等量对照

为把5.4节中0.419 GiB对0.896 GiB的粗粒度差距进一步收紧，本轮把Ours每层多卸载
1个专家：C1全部和Short/C16使用cold2，Ordinary/Long C16使用cold6，RAG/C16使用
cold20。Native保持5.4节的请求预算。最终实际字节分别为0.838/0.896、2.514/2.330和
8.379/8.522 GiB，三组相对差值为7.0%、7.9%和1.7%，均小于10%。

本表的有效协议是Native eager Prefill加`FULL_DECODE_ONLY`，Ours为Prefill/Decode
`FULL_SEPARATE`。因此TPOT可用于近等量Decode路径诊断，TTFT和E2E不能解释为双Graph
单变量消融。后续尝试把Native也改为Prefill/Decode双Graph，但未形成完整合法矩阵，详见
5.5.1。

单位均为ms，倍率为`Native/Ours`；`†`表示跨机制完整输出hash不一致。

| 场景 | C | Ours TTFT | Native TTFT | Ours TPOT | Native TPOT | Ours E2E | Native E2E | Native/Ours：TTFT/TPOT/E2E | 输出hash |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Short | 1 | 62.35 | 85.37 | 8.50 | 15.72 | 326.00 | 572.60 | 1.37×/1.85×/1.76× | 一致 |
| Ordinary | 1 | 64.60 | 141.99 | 7.98 | 15.91 | 1,078.44 | 2,162.13 | 2.20×/1.99×/2.00× | 一致 |
| RAG | 1 | 370.75 | 1,620.00 | 12.09 | 19.83 | 1,905.58 | 4,137.94 | 4.37×/1.64×/2.17× | 不一致† |
| Long | 1 | 74.51 | 146.64 | 7.98 | 16.10 | 4,153.72 | 8,374.10 | 1.97×/2.02×/2.02× | 一致 |
| Short | 16 | 85.65 | 363.52 | 22.26 | 33.82 | 775.85 | 1,411.84 | 4.24×/1.52×/1.82× | 一致 |
| Ordinary | 16 | 280.94 | 2,280.32 | 40.74 | 57.03 | 5,455.32 | 9,523.66 | 8.12×/1.40×/1.75× | 一致 |
| RAG | 16 | 2,795.51 | 58,124.07 | 150.92 | 516.11 | 21,962.26 | 123,670.07 | 20.79×/3.42×/5.63× | 不一致† |
| Long | 16 | 281.54 | 2,267.45 | 43.29 | 57.58 | 22,403.69 | 31,690.72 | 8.05×/1.33×/1.41× | 一致 |

Ours在8/8点的TTFT、TPOT和E2E均更低；TPOT优势为1.33×--3.42×，6个完整输出一致
点仍为1.33×--2.02×。与5.4节的Ours cold1/5/19相比，每层多卸载一个专家后Ours
TPOT在8/8点增加5.18%--18.30%，说明增加remote专家确实带来Decode成本，但未改变本轮
相对排名。跨轮敏感性比较不是同进程ABBA，只作描述性诊断。

16/16正式cell和136/136 measured请求完成，聚合verifier为`status=ok`，83/83项检查
通过；无GPU fault，最终0 MiB并恢复stock UVM。该轮仍不是纯机制消融：两臂卸载对象不同，
且RAG两点不能进入exact-generation排名。

来源：

- `artifacts/e2-near-equal-offload-plus-one-expert-20260903-attempt-01/summary.json`
- `artifacts/e2-near-equal-offload-plus-one-expert-20260903-attempt-01/verification.json`
- [RESULTS_E2_NEAR_EQUAL_OFFLOAD_PLUS_ONE_EXPERT_20260903.md](RESULTS_E2_NEAR_EQUAL_OFFLOAD_PLUS_ONE_EXPERT_20260903.md)

#### 5.5.1 Native UVA Prefill/Decode双Graph可行性复核

2026-09-03按相同近等量卸载和冻结请求尝试将Native改为`FULL_SEPARATE`。128-token
Prefill Graph下，Native的Short/C1、Ordinary/C1和RAG/C1均完成，运行时审计确认
Prefill为`runtime_mode=FULL`；Long/C1则在第一次真实Prefill replay后触发CUDA
`illegal memory access`，没有产生请求结果。把Long/C1的Graph块从128降到64后，捕获仍
成功，但第一次真实64-token Prefill replay发生相同故障。因此问题不是通过缩小Graph块即可
消除的容量问题。

由于C1已不能全覆盖，本轮在进入C16和Ours正式臂前终止；没有生成16-cell聚合
`verification.json`，也没有把前三个可运行点当作性能表发布。当前不能声称DeepSeek E2的
Native UVA Prefill/Decode双Graph全矩阵可行，也不能用这次不完整尝试替换上面的有效近等量
结果。正式复现入口和两个冻结profile已恢复为最近一次成功协议，即Native eager Prefill加
`FULL_DECODE_ONLY`。

失败证据：

- `artifacts/e2-near-equal-plus-one-full-separate-graph128-20260903-attempt-01/`
- `artifacts/e2-native-prefill-graph64-feasibility-long-c1-20260903-attempt-01/`
- [RESULTS_E2_NATIVE_UVA_FULL_SEPARATE_FEASIBILITY_20260903.md](RESULTS_E2_NATIVE_UVA_FULL_SEPARATE_FEASIBILITY_20260903.md)


完整E2尚未完成：当前没有按完整measured peak HBM配平的DeepSeek三系统主表，也没有
Qwen ours/KT/vLLM所有tier、workload、C1/C4/C8/C16和每点5个独立进程的正式矩阵。

来源：

- D/ours B1/B16：`artifacts/e3-c1-c16-full-dual-20260901-reproduce-all-attempt-02/`
- Native FLASH copy/UVA：`artifacts/e3-native-copy-uva-reproduce-all-20260901-attempt-01/`
- Native TRITON大B8：`artifacts/e3-native-triton-large-b8-reproduce-all-20260901-attempt-01/`
- Prefill Graph-copy：`artifacts/e3-prefill-graph-copy-reproduce-all-20260901-attempt-01/`
- Decode完整汇总：`artifacts/e3-decode-complete-reproduce-all-20260901-attempt-01/`
- [RESULTS_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md](RESULTS_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md)
- [RESULTS_C1_MECHANISM_RUNTIME_HOT_AUDIT_OFF_MATRIX_20260827.md](RESULTS_C1_MECHANISM_RUNTIME_HOT_AUDIT_OFF_MATRIX_20260827.md)
- [RESULTS_E2_C16_DIRECT_BATCH_MATRIX_20260828.md](RESULTS_E2_C16_DIRECT_BATCH_MATRIX_20260828.md)
- reference arms：`artifacts/e2-e3-e10-full-c1-c16-uva-single-round-20260831-attempt-01/e2/`
- ours C1：`artifacts/e2-e3-e10-full-c1-c16-dual-20260901-reproduce-all-attempt-01/e2/c1/`
- ours C16：`artifacts/e2-c16-frozen-20260901-reproduce-all-attempt-02/e2/c16/`

## 6. E3：Prefill/Decode当前档双页表与完整路径对照

### 6.1 最新 D/ours 全覆盖双页表

最新 D 定义为“GPU-local/copy Prefill Graph → 当前档双页表 P→D 切换 → remote Decode
Graph”。本轮只运行 B1/B16：Short/B1、Short/B16、Ordinary/B1、Medium/B1、Long/B1
使用 cold10（2.109 GiB）；Ordinary/B16、Medium/B16、RAG/B1、Long/B16 使用 cold38
（8.016 GiB）；RAG/B16 因容量约束使用 cold89（18.773 GiB）。十个场景均达到双页表
100%覆盖，没有CPU range-map回退或Graph重捕获。

| 场景 | Batch | Prefill | P→D | steady Decode | E2E |
|---|---:|---:|---:|---:|---:|
| Short | 1 | 68.11 ms | 6.60 ms | 9.97 ms | 455.45 ms |
| Short | 16 | 105.95 ms | 6.47 ms | 18.30 ms | 754.39 ms |
| Ordinary | 1 | 74.13 ms | 6.41 ms | 9.55 ms | 1379.46 ms |
| Ordinary | 16 | 368.50 ms | 23.54 ms | 54.98 ms | 7503.39 ms |
| Medium | 1 | 112.85 ms | 6.44 ms | 13.90 ms | 1940.10 ms |
| Medium | 16 | 1123.14 ms | 24.20 ms | 57.05 ms | 8528.73 ms |
| RAG | 1 | 587.35 ms | 27.40 ms | 15.97 ms | 2894.29 ms |
| RAG | 16 | 7272.86 ms | 63.68 ms | 161.56 ms | 27996.95 ms |
| Long | 1 | 75.09 ms | 6.50 ms | 10.48 ms | 5618.18 ms |
| Long | 16 | 369.47 ms | 23.20 ms | 61.37 ms | 31558.98 ms |

RAG/C16的早期cold63配置需要14.72 GiB热专家，但工作区建立后仅剩10.17 GiB可用，因此
合法改为cold89；这是容量约束，不是页表故障。

最新结果来自`e3-c1-c16-full-dual-20260901-reproduce-all-attempt-02`，证明当前档双页表能
正确覆盖和切换，但这一轮没有成功配对纯remote或纯copy arm，
因此不能单独由上表计算 D/remote 或 D/copy 加速比。下面把已经完成的较早机制矩阵分协议
列出；这些数据补足机制行为，但不回填成2026-08-31最新D的同轮对照。

### 6.2 相同B1/B16请求的KTransformers完整框架基线

2026-09-02按与最新D相同的10个请求坐标补做KTransformers。模型snapshot、冻结输入
token、Prompt/输出长度、batch、`TRITON_ATTN`和逐cell专家坐标相同；KTransformers固定
14 CPU线程。每个cell是独立冷进程、完整batch、完整Prefill，未拆batch或启用mixed
chunk。本轮启用KTransformers原生Decode CUDA Graph：B1进程捕获B1，B16进程捕获
B1/B16；Prefill仍为eager。KTransformers没有D的双页表或P→D切换，因此P→D为N/A。

下表单位为ms，倍率为`KTransformers/D`：

| 场景 | B | 档位 | KT Prefill | KT TTFT | KT steady Decode | KT E2E | Prefill比 | Decode比 | E2E比 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Short | 1 | cold10 | 374.42 | 886.92 | 61.80 | 3,417.84 | 5.50× | 6.20× | 7.50× |
| Short | 16 | cold10 | 442.99 | 4,496.14 | 778.29 | 28,475.71 | 4.18× | 42.54× | 37.75× |
| Ordinary | 1 | cold10 | 378.11 | 1,371.48 | 63.37 | 11,175.02 | 5.10× | 6.64× | 8.10× |
| Ordinary | 16 | cold38 | 4,459.57 | 23,540.25 | 855.79 | 130,617.89 | 12.10× | 15.56× | 17.41× |
| Medium | 1 | cold10 | 1,248.43 | 11,748.19 | 295.75 | 52,385.63 | 11.06× | 21.28× | 27.00× |
| Medium | 16 | cold38 | 18,681.38 | 94,799.06 | 873.48 | 212,822.33 | 16.63× | 15.31× | 24.95× |
| RAG | 1 | cold38 | 3,743.08 | 19,197.41 | 171.57 | 43,580.07 | 6.37× | 10.74× | 15.06× |
| RAG | 16 | cold89 | 1,040,867.90 | 1,239,060.09 | 1,181.66 | 1,389,293.19 | 143.12× | 7.31× | 49.62× |
| Long | 1 | cold10 | 365.92 | 1,248.06 | 139.35 | 63,269.63 | 4.87× | 13.30× | 11.26× |
| Long | 16 | cold38 | 5,232.82 | 27,084.41 | 973.06 | 668,035.45 | 14.16× | 15.86× | 21.17× |

为避免只看倍率而看不到自定义机制的绝对值，下表把最新D/ours与KTransformers逐cell直接
并列。`D P→D`是D额外支付的双页表快速切换；KTransformers没有对应阶段。最后一列仍为
`KTransformers/D`的Prefill/Decode/E2E延迟比。

| 场景 | B | Ours D Prefill | KT Prefill | Ours D P→D | Ours D Decode | KT Decode | Ours D E2E | KT E2E | KT/Ours D：P/D/E2E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 1 | 68.11 | 374.42 | 6.60 | 9.97 | 61.80 | 455.45 | 3,417.84 | 5.50×/6.20×/7.50× |
| Short | 16 | 105.95 | 442.99 | 6.47 | 18.30 | 778.29 | 754.39 | 28,475.71 | 4.18×/42.54×/37.75× |
| Ordinary | 1 | 74.13 | 378.11 | 6.41 | 9.55 | 63.37 | 1,379.46 | 11,175.02 | 5.10×/6.64×/8.10× |
| Ordinary | 16 | 368.50 | 4,459.57 | 23.54 | 54.98 | 855.79 | 7,503.39 | 130,617.89 | 12.10×/15.56×/17.41× |
| Medium | 1 | 112.85 | 1,248.43 | 6.44 | 13.90 | 295.75 | 1,940.10 | 52,385.63 | 11.06×/21.28×/27.00× |
| Medium | 16 | 1,123.14 | 18,681.38 | 24.20 | 57.05 | 873.48 | 8,528.73 | 212,822.33 | 16.63×/15.31×/24.95× |
| RAG | 1 | 587.35 | 3,743.08 | 27.40 | 15.97 | 171.57 | 2,894.29 | 43,580.07 | 6.37×/10.74×/15.06× |
| RAG | 16 | 7,272.86 | 1,040,867.90 | 63.68 | 161.56 | 1,181.66 | 27,996.95 | 1,389,293.19 | 143.12×/7.31×/49.62× |
| Long | 1 | 75.09 | 365.92 | 6.50 | 10.48 | 139.35 | 5,618.18 | 63,269.63 | 4.87×/13.30×/11.26× |
| Long | 16 | 369.47 | 5,232.82 | 23.20 | 61.37 | 973.06 | 31,558.98 | 668,035.45 | 14.16×/15.86×/21.17× |

从绝对值看，D在10/10 cell的Prefill、steady Decode和E2E均更低。即使D额外执行
6.41--63.68 ms的P→D切换，该阶段只占D E2E的0.07%--1.45%，没有抵消其稳态收益。
Prefill差距最大的是RAG/B16（143.12×），Decode差距为6.20×--42.54×；完整请求E2E
差距为7.50×--49.62×。D没有采集与KT客户端定义完全相同的TTFT，因此不构造跨系统
TTFT倍率。

10/10 cell、85/85请求均在单轮中完成，verifier为`status=ok`；所有日志均包含Graph捕获
和`cuda graph: True` replay标记，没有Xid/MMU fault，GPU最终为0 MiB。RAG/B16完整
131,072-token服务端Prefill计时约17.35分钟，E2E约23.15分钟；7,200秒watchdog未触发。

跨框架完整输出hash为0/10 cell一致；要求cell内所有请求首token均一致时为3/10。因此
这些倍率只能作为相同冻结输入坐标下的描述性性能诊断，不能进入exact-generation系统
排名。两边均启用各自的CUDA Graph，但D为双阶段Graph+GPU remote-read，KT仅Decode
Graph+CPU expert执行；倍率还同时包含内存布局、调度器和P→D机制差异，不能归因于单项
机制。相对已取代的eager单样本，本轮B1 E2E改善1.64×--3.30×，B16为0.95×--1.45×；
这是跨轮点估计，不是重复统计或纯Graph消融。每个cell只有一个成功冷进程样本。

来源：

- `artifacts/e3-ktransformers-c1-c16-cudagraph-20260902-attempt-01/verification.json`
- [RESULTS_E3_KTRANSFORMERS_C1_C16_20260902.md](RESULTS_E3_KTRANSFORMERS_C1_C16_20260902.md)

#### 6.2.1 KTransformers请求形状感知调优后的当前结果

2026-09-03在相同五类冻结请求和B1/B16坐标上重新调优KTransformers。最终配置把
expert CPU线程从14降为7，但保留整个进程使用0--13共14个物理核，避免scheduler、
Graph和专家线程互相争抢；默认使用FlashInfer，RAG/B16因容量改用TRITON；Decode
启用CUDA Graph，Prefill为eager；Medium/B16和RAG/B16分别以16384/4096阈值进入
layerwise full-GPU Prefill fallback。

下表单位为whole-batch ms，Decode为稳态单token step的p50。UVA和copy使用近等量
whole-Parameter卸载及`TRITON_ATTN + FULL_SEPARATE`；KT按expert放置且只有Decode
Graph，因此只作系统级同请求描述性比较。

| 场景 | B | KT Prefill | UVA Prefill | Copy Prefill | KT Decode | UVA Decode | Copy Decode |
|---|---:|---:|---:|---:|---:|---:|---:|
| Short | 1 | 119.67 | 60.97 | 64.11 | 23.93 | 12.13 | 52.27 |
| Short | 16 | 271.96 | 417.16 | 93.38 | 39.55 | 42.21 | 60.70 |
| Ordinary | 1 | 114.73 | 120.82 | 69.94 | 20.70 | 12.23 | 52.40 |
| Ordinary | 16 | 5,438.10 | 3,700.62 | 334.21 | 116.78 | 110.95 | 169.46 |
| Medium | 1 | 1,034.87 | 451.50 | 108.06 | 33.21 | 12.51 | 52.67 |
| Medium | 16 | 3,894.07 | 14,661.36 | 1,103.21 | 132.00 | 108.52 | 170.84 |
| RAG | 1 | 4,694.37 | 3,917.74 | 571.80 | 30.66 | 23.89 | 160.91 |
| RAG | 16 | 10,135.03 | 140,315.46 | N/A | 328.20 | 240.93 | N/A |
| Long | 1 | 151.15 | 122.14 | 70.79 | 24.54 | 12.27 | 52.38 |
| Long | 16 | 6,327.14 | 3,643.19 | 334.40 | 169.53 | 103.98 | 168.98 |

相对UVA，KT的Prefill/Decode/E2E分别在4/10、1/10、2/10点更低；相对copy，分别在
0/9、8/9、5/9有效点更低。最严重的旧KT异常已消除：RAG/B16的Prefill、Decode、E2E
由1,040,867.90/1,181.66/1,389,293.19 ms降到
10,135.03/328.20/52,370.06 ms，即快102.70×/3.60×/26.53×。相对旧KT，调优后的
Decode和E2E分别在10/10点改善2.58×--19.68×和2.24×--26.53×。

剩余差距来自CPU冷专家实际计算和CPU/GPU同步不能被CUDA Graph覆盖，以及RTX 5090
缺少专用W8A8 Block FP8与E=118/90/39 fused-MoE Triton配置；日志明确回退到可能非最优
的默认kernel。RAG/B16的FlashInfer fallback和hybrid还分别因仅余36/182 MiB而无法满足
768 MiB workspace，最终TRITON+fallback才完成。

正式只采用`e3-ktransformers-tuned-final-20260903-repeat-1`的10/10完整单轮。聚合
verifier为`status=ok`、`errors=[]`；输出长度/本臂hash、Graph capture/replay、48层
placement、fault和cleanup门全部通过，最终GPU为0 MiB。用户要求停止重复实验后，
第二轮及独立续跑已终止且`repeat-2/repeat-2b`部分artifact不纳入，第三轮未运行。因此
这些数值都是单冷进程点估计，没有方差、置信区间或显著性结论；Medium/B16 fallback
与hybrid hash不同，也没有建立KT/vLLM跨框架exact-generation门。

来源：

- `artifacts/e3-ktransformers-tuned-final-20260903-verification.json`
- [RESULTS_E3_KTRANSFORMERS_TUNED_20260903.md](RESULTS_E3_KTRANSFORMERS_TUNED_20260903.md)
- [REPRODUCE_E3_KTRANSFORMERS_TUNED_SINGLE_ROUND_20260903.md](../reproduce/REPRODUCE_E3_KTRANSFORMERS_TUNED_SINGLE_ROUND_20260903.md)

### 6.3 原生Native UVA的Prefill/Decode双Graph

2026-09-03按最新D相同的十个B1/B16冻结请求坐标，仅运行原生vLLM pinned Native UVA。
Qwen `TRITON_ATTN`路径使用真正的`FULL_SEPARATE`：pure-Prefill和uniform-Decode分别
捕获。10/10点的两阶段运行时dispatch均为`runtime_mode=FULL`，没有eager fallback；
10/10完整输出hash与D/ours一致。

| 场景 | B | Native Prefill | Native Decode | Native TTFT | Native TPOT | Native E2E |
|---|---:|---:|---:|---:|---:|---:|
| Short | 1 | 60.68 | 11.93 | 60.88 | 11.96 | 431.65 |
| Short | 16 | 428.90 | 41.90 | 429.50 | 41.57 | 1,718.13 |
| Ordinary | 1 | 119.80 | 12.09 | 120.01 | 12.12 | 1,658.99 |
| Ordinary | 16 | 1,174.14 | 46.25 | 1,174.75 | 46.08 | 7,027.44 |
| Medium | 1 | 426.77 | 12.31 | 426.99 | 12.35 | 1,994.89 |
| Medium | 16 | 14,676.97 | 108.26 | 14,677.78 | 107.80 | 28,368.51 |
| RAG | 1 | 3,932.51 | 23.69 | 3,932.80 | 23.72 | 6,945.26 |
| RAG | 16 | 140,164.81 | 240.82 | 140,166.50 | 239.78 | 170,618.24 |
| Long | 1 | 121.06 | 12.06 | 121.28 | 12.08 | 6,296.24 |
| Long | 16 | 1,157.09 | 45.34 | 1,157.89 | 45.27 | 24,288.81 |

下表将最新D/ours与Native Graph的实际延迟逐cell并列，单位均为ms。Native没有双页表
P→D切换，因此只列D实际支付的P→D成本；最后一列为`Native/D`的
Prefill/steady Decode/E2E倍率，小于1表示Native更快。

| 场景 | B | D Prefill | Native Prefill | D P→D | D Decode | Native Decode | D E2E | Native E2E | Native/D：P/D/E2E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 1 | 68.11 | 60.68 | 6.60 | 9.97 | 11.93 | 455.45 | 431.65 | 0.89×/1.20×/0.95× |
| Short | 16 | 105.95 | 428.90 | 6.47 | 18.30 | 41.90 | 754.39 | 1,718.13 | 4.05×/2.29×/2.28× |
| Ordinary | 1 | 74.13 | 119.80 | 6.41 | 9.55 | 12.09 | 1,379.46 | 1,658.99 | 1.62×/1.27×/1.20× |
| Ordinary | 16 | 368.50 | 1,174.14 | 23.54 | 54.98 | 46.25 | 7,503.39 | 7,027.44 | 3.19×/0.84×/0.94× |
| Medium | 1 | 112.85 | 426.77 | 6.44 | 13.90 | 12.31 | 1,940.10 | 1,994.89 | 3.78×/0.89×/1.03× |
| Medium | 16 | 1,123.14 | 14,676.97 | 24.20 | 57.05 | 108.26 | 8,528.73 | 28,368.51 | 13.07×/1.90×/3.33× |
| RAG | 1 | 587.35 | 3,932.51 | 27.40 | 15.97 | 23.69 | 2,894.29 | 6,945.26 | 6.70×/1.48×/2.40× |
| RAG | 16 | 7,272.86 | 140,164.81 | 63.68 | 161.56 | 240.82 | 27,996.95 | 170,618.24 | 19.27×/1.49×/6.09× |
| Long | 1 | 75.09 | 121.06 | 6.50 | 10.48 | 12.06 | 5,618.18 | 6,296.24 | 1.61×/1.15×/1.12× |
| Long | 16 | 369.47 | 1,157.09 | 23.20 | 61.37 | 45.34 | 31,558.98 | 24,288.81 | 3.13×/0.74×/0.77× |

Native/D范围为Prefill 0.89×--19.27×、Decode 0.74×--2.29×、E2E
0.77×--6.09×。Ordinary/B16和Long/B16中Native保留更多GPU权重，因此其Decode或E2E
可以低于D；RAG/B16则由Native的remote Prefill主导，Native Prefill和E2E分别为D的
19.27×和6.09×。

按cell计，D的Prefill在9/10点更低，steady Decode在7/10点更低，完整E2E也在7/10点
更低。Native完整E2E更低的三个点是Short/B1、Ordinary/B16和Long/B16。D表中的E2E已经
包含6.41--63.68 ms的P→D切换，因此并列表没有隐去ours的阶段切换成本。

该对照证明Native UVA可以合法使用CUDA Graph，但不是Graph单变量消融。Native按whole
Parameters卸载，D按exact expert slices放置并额外使用双页表/P→D；实际卸载集合和HBM
分配不同。所有cell只有一个冷进程样本。

来源：

- `artifacts/e2-e3-native-uva-cudagraph-20260903-attempt-04/verification.json`
- [RESULTS_E2_E3_NATIVE_UVA_CUDAGRAPH_20260903.md](RESULTS_E2_E3_NATIVE_UVA_CUDAGRAPH_20260903.md)

### 6.4 近等量卸载下Native UVA与Ours双阶段Graph对照

2026-09-03进一步消除6.3节的卸载字节不配平：Ours改用cold11/39/90，Native使用
2.25/7.875/20.25 GiB请求预算。三档实际卸载量分别为2.320/2.322、8.227/7.942和
18.984/20.324 GiB（Ours/Native），相对差值0.08%、3.58%和7.06%，均不超过10%。

两臂均使用`TRITON_ATTN`和`FULL_SEPARATE`；下表的Prefill与Decode均由运行时审计确认
为`runtime_mode=FULL`。单位为ms，倍率为`Native/Ours`，大于1表示Ours更快。

| 场景 | B | Ours/Native卸载GiB | Ours Prefill | Native Prefill | Ours P→D | Ours Decode | Native Decode | Ours E2E | Native E2E | Native/Ours：P/D/E2E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 1 | 2.320/2.322 | 71.93 | 60.97 | 6.99 | 9.50 | 12.13 | 446.70 | 443.86 | 0.85×/1.28×/0.99× |
| Short | 16 | 2.320/2.322 | 107.78 | 417.16 | 6.16 | 17.86 | 42.21 | 731.25 | 1,712.35 | 3.87×/2.36×/2.34× |
| Ordinary | 1 | 2.320/2.322 | 77.64 | 120.82 | 6.30 | 9.08 | 12.23 | 1,311.02 | 1,687.11 | 1.56×/1.35×/1.29× |
| Ordinary | 16 | 8.227/7.942 | 371.49 | 3,700.62 | 21.03 | 55.78 | 110.95 | 7,570.46 | 17,777.27 | 9.96×/1.99×/2.35× |
| Medium | 1 | 2.320/2.322 | 116.66 | 451.50 | 6.28 | 13.93 | 12.51 | 1,921.92 | 2,053.30 | 3.87×/0.90×/1.07× |
| Medium | 16 | 8.227/7.942 | 1,143.10 | 14,661.36 | 21.58 | 57.79 | 108.52 | 8,604.90 | 28,385.61 | 12.83×/1.88×/3.30× |
| RAG | 1 | 8.227/7.942 | 593.01 | 3,917.74 | 20.87 | 15.48 | 23.89 | 2,786.45 | 6,968.59 | 6.61×/1.54×/2.50× |
| RAG | 16 | 18.984/20.324 | 7,394.70 | 140,315.46 | 58.39 | 163.17 | 240.93 | 28,268.10 | 170,794.72 | 18.98×/1.48×/6.04× |
| Long | 1 | 2.320/2.322 | 78.57 | 122.14 | 6.31 | 10.11 | 12.27 | 5,445.50 | 6,457.72 | 1.55×/1.21×/1.19× |
| Long | 16 | 8.227/7.942 | 371.16 | 3,643.19 | 21.28 | 62.15 | 103.98 | 31,847.70 | 56,887.80 | 9.82×/1.67×/1.79× |

Ours的Prefill和steady Decode均在9/10点更低；计入6.16--58.39 ms的P→D后，E2E也在
9/10点更低1.07×--6.04×。Short/B1 E2E近似持平，Native低0.64%；Medium/B1的Native
Decode低10.17%。因此不能写成Ours逐cell全部更快，但在近等量卸载和双阶段Graph条件下，
整体趋势已不再由Native卸载量明显偏少解释。

20/20 cell完成，10/10完整输出hash一致，聚合verifier为`status=ok`；Ours十点均使用
预建完整双页表执行P→D，无eager fallback、Xid、MMU/PDE/PTE或CUDA fault，最终GPU
0 MiB并恢复stock UVM。该轮仍只有每cell一个冷进程样本，且两臂卸载对象不同，不能作为
单项机制或统计显著性结论。

来源：

- `artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/summary.json`
- `artifacts/e3-near-equal-offload-plus-one-expert-20260903-attempt-02/verification.json`
- [RESULTS_E3_NEAR_EQUAL_OFFLOAD_DUAL_GRAPH_20260903.md](RESULTS_E3_NEAR_EQUAL_OFFLOAD_DUAL_GRAPH_20260903.md)

### 6.5 近等量卸载下Graph-enabled Native copy与Ours对照

2026-09-03按与6.4相同的cold11/39/90和Native 2.25/7.875/20.25 GiB预算，补做
Graph-enabled Native copy并重新运行Ours。两臂仍为`TRITON_ATTN`和`FULL_SEPARATE`，
完整batch、完整Prefill且每cell只有一个冷进程样本。单位为ms，倍率为
`Native copy/Ours`。

| 场景 | B | Ours/Native卸载GiB | Ours Prefill | Native copy Prefill | Ours P→D | Ours Decode | Native copy Decode | Ours E2E | Native copy E2E | Native/Ours：P/D/E2E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 1 | 2.320/2.322 | 71.15 | 64.11 | 6.13 | 9.41 | 52.27 | 440.77 | 1,689.10 | 0.90×/5.55×/3.83× |
| Short | 16 | 2.320/2.322 | 108.18 | 93.38 | 6.23 | 18.00 | 60.70 | 737.67 | 1,976.66 | 0.86×/3.37×/2.68× |
| Ordinary | 1 | 2.320/2.322 | 78.06 | 69.94 | 6.15 | 9.24 | 52.40 | 1,321.47 | 6,736.05 | 0.90×/5.67×/5.10× |
| Ordinary | 16 | 8.227/7.942 | 371.96 | 334.21 | 21.19 | 55.77 | 169.46 | 7,569.74 | 21,878.26 | 0.90×/3.04×/2.89× |
| Medium | 1 | 2.320/2.322 | 116.89 | 108.06 | 6.21 | 13.75 | 52.67 | 1,925.80 | 6,808.37 | 0.92×/3.83×/3.54× |
| Medium | 16 | 8.227/7.942 | 1,140.93 | 1,103.21 | 20.70 | 57.92 | 170.84 | 8,598.77 | 22,810.14 | 0.97×/2.95×/2.65× |
| RAG | 1 | 8.227/7.942 | 593.42 | 571.80 | 21.90 | 15.46 | 160.91 | 2,792.33 | 21,023.05 | 0.96×/10.41×/7.53× |
| Long | 1 | 2.320/2.322 | 78.85 | 70.79 | 6.88 | 10.08 | 52.38 | 5,396.46 | 26,883.10 | 0.90×/5.20×/4.98× |
| Long | 16 | 8.227/7.942 | 370.55 | 334.40 | 21.05 | 62.01 | 168.98 | 31,829.13 | 86,734.68 | 0.90×/2.72×/2.73× |

九个有效配对点中，Native copy Prefill均低3.31%--13.69%；Ours steady Decode均低
2.72×--10.41×，含P→D的E2E均低2.65×--7.53×。Graph降低了dispatch开销，但没有
消除Native copy每个forward回拷卸载参数的数据移动。

RAG/B16中，Ours以18.984 GiB实际卸载成功完成：Prefill 7,389.36 ms、P→D 56.19 ms、
Decode 163.26 ms、E2E 28,277.22 ms，Prefill/Decode均为运行时Full Graph。Native copy
实际卸载20.324 GiB，在完整131,072-token Prefill Graph捕获时尝试再申请1,024 MiB，
当时仅余942.75 MiB，峰值31,979 MiB，因此该点为容量N/A，不能用拆batch、chunked
Prefill或第二次attempt替换。

Native成功9点与Ours对应点完整输出hash为9/9一致；Native成功点18/18阶段、Ours十点
20/20阶段均实际`runtime_mode=FULL`。Ours十点均为`prebuilt_full_dual_view`。strict
verifier因Native launcher非零及RAG/B16无result保留`status=error`，209/211检查为true；
这表示矩阵不完整，不表示其余九点机制失败。无Xid、MMU/PDE/PTE fault或illegal access，
GPU最终0 MiB并恢复stock UVM。

来源：

- `artifacts/e3-near-equal-native-copy-graph-20260903-attempt-01/verification.json`
- [RESULTS_E3_NEAR_EQUAL_NATIVE_COPY_GRAPH_20260903.md](RESULTS_E3_NEAR_EQUAL_NATIVE_COPY_GRAPH_20260903.md)

### 6.6 自定义 Graph 与原生 copy/UVA 的阶段对照（完整数据复跑）

2026-09-01/02重新采集native FLASH copy/UVA 20/20、Custom Prefill Graph-copy 10/10和
原协议`FLASH_ATTN` Custom Decode 8/8。旧表的Medium/B8、RAG/B8由当前档全覆盖双页表
D/TRITON路径补齐，并对称重跑native TRITON copy/UVA 4/4。最终Decode表10/10逐cell
同backend且完整输出hash一致；矩阵仍分为FLASH和TRITON两个协议层。

Prefill custom为`TRITON_ATTN`+exact expert slices，native为`FLASH_ATTN`+whole
Parameters。倍率定义为`native/custom`，只作数值诊断。单位均为ms。

| 请求 | B | Custom Graph-copy | Native copy | Native UVA | Graph/copy | Graph/UVA |
|---|---:|---:|---:|---:|---:|---:|
| Short 128 | 1 | 67.93 | 68.19 | 70.18 | 1.004× | 1.033× |
| Short 128 | 8 | 92.14 | 76.82 | 196.81 | 0.834× | 2.136× |
| Ordinary 512 | 1 | 73.96 | 70.62 | 121.89 | 0.955× | 1.648× |
| Ordinary 512 | 8 | 151.54 | 137.93 | 826.48 | 0.910× | 5.454× |
| Medium 2048 | 1 | 113.44 | 103.48 | 418.44 | 0.912× | 3.689× |
| Medium 2048 | 8 | 651.61 | 563.05 | 7508.83 | 0.864× | 11.524× |
| RAG 8192 | 1 | 588.59 | 461.11 | 3754.00 | 0.783× | 6.378× |
| RAG 8192 | 8 | 3696.11 | 2728.73 | 48291.27 | 0.738× | 13.065× |
| Long 512 | 1 | 75.03 | 71.55 | 122.66 | 0.954× | 1.635× |
| Long 512 | 8 | 151.16 | 138.14 | 834.24 | 0.914× | 5.519× |

Custom Graph-copy相对UVA为10/10更快1.033×–13.065×；相对native copy仅Short/B1
快1.004×，其余9/10更慢。首token仍为9/10 cell一致，Medium/B8有1/8请求不同。

Decode延迟是whole-batch单token稳态step中位数。8个C点三臂统一`FLASH_ATTN`；两个旧
缺失点三臂统一`TRITON_ATTN`，并单列协议和完整hash验收。

| 请求 | B | 三臂协议 | Custom Graph remote | Native copy | Native UVA | Custom/copy | Custom/UVA | 完整hash |
|---|---:|---|---:|---:|---:|---:|---:|---|
| Short 128→32 | 1 | C/FLASH | 10.20 | 70.56 | 68.80 | 6.92× | 6.75× | 一致 |
| Short 128→32 | 8 | C/FLASH | 15.37 | 70.64 | 69.48 | 4.60× | 4.52× | 一致 |
| Ordinary 512→128 | 1 | C/FLASH | 10.13 | 69.90 | 67.36 | 6.90× | 6.65× | 一致 |
| Ordinary 512→128 | 8 | C/FLASH | 17.95 | 71.87 | 69.41 | 4.00× | 3.87× | 一致 |
| Medium 2048→128 | 1 | C/FLASH | 13.98 | 69.32 | 68.97 | 4.96× | 4.93× | 一致 |
| Medium 2048→128 | 8 | D/TRITON | 42.01 | 167.77 | 82.78 | 3.99× | 1.97× | 一致 |
| RAG 8192→128 | 1 | C/FLASH | 24.01 | 161.39 | 68.01 | 6.72× | 2.83× | 一致 |
| RAG 8192→128 | 8 | D/TRITON | 64.21 | 280.88 | 125.06 | 4.37× | 1.95× | 一致 |
| Long 512→512 | 1 | C/FLASH | 11.06 | 69.65 | 67.34 | 6.30× | 6.09× | 一致 |
| Long 512→512 | 8 | C/FLASH | 31.51 | 72.38 | 70.49 | 2.30× | 2.24× | 一致 |

逐cell同backend的10点中，Custom全部最快：对native copy快2.30×–6.92×、对native
UVA快1.95×–6.75×，完整输出hash 10/10一致。这些范围合并了FLASH和TRITON两个协议层，
不能作为纯remote访问、纯Graph或attention backend的单项排名。

D补充点保持原B8和完整16K/65K Prefill，未拆batch或启用chunked Prefill；两点均完整生成，
Prefill/Decode Graph replay、双页表100%覆盖、CPU range-map回退为0，并与对称native
TRITON arms完整hash一致。另做的三次
Medium/B8-G/FLASH定向尝试在cold38/cold63下仍触发原original-VA大Prefill非法访问，
证明旧N/A不能通过简单扩容合法消除。完整记录见
`RESULTS_E3_SECTION65_COMPLETE_RERUN_20260831.md`。


### 6.7 E3综合结论与边界

当前证据可以支持：

1. 最新D在B1/B16十点实现当前档双页表100%覆盖并正确完成请求；
2. 较早同坐标Decode-Graph矩阵中，hybrid G相对always-copy B为7/7更快3.014×–5.270×，
   但always-remote C在6/7点更快，说明极冷计划下不切换通常更好；
3. 原生native矩阵中copy降低Prefill成本、UVA降低Decode成本，RAG/B8已经实测出现完整
   请求crossover；
4. 自定义remote Decode Graph在10个逐cell同backend点都快于两个native eager路径，但
   这不是equal-HBM或单机制隔离结论；
5. 调优KTransformers在相同B1/B16十坐标完成10/10；RAG/B16相对旧KT的
   Prefill/Decode/E2E快102.70×/3.60×/26.53×，相对copy的Decode在8/9有效点更低，
   但相对Native UVA的Decode仅1/10更低；跨框架未建立exact-generation门，只作描述性
   诊断；
6. 原生Native UVA双Graph已在相同B1/B16十坐标完成10/10点，并与D完整输出hash一致；
   Native/D的结果随placement和请求变化，不能解释为单一Graph收益。
7. 近等量卸载双Graph轮中，两臂实际字节差不超过10%且输出hash 10/10一致；Ours的
   Prefill、Decode和含P→D的E2E分别在9/10点更低。
8. 近等量Native copy双Graph轮中，九个合法配对点的Native Prefill略快，但Ours的
   Decode和E2E分别快2.72×--10.41×和2.65×--7.53×；Native RAG/B16因Prefill Graph
   捕获HBM不足为容量N/A。

仍不能支持：把较早G/B/C、H/A/R数值填入最新D/C/F空格；把不同whole-Parameter与
exact-slice placement写成equal-HBM；发布跨backend、跨placement或跨模块的统一系统
排名。所有补充矩阵均只有一个冷进程样本，没有方差或置信区间。

来源：

- [RESULTS_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md](RESULTS_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md)
- [RESULTS_E3_GRAPH_SINGLE_ROUND_B1_B8_NO_HTTP_20260828.md](RESULTS_E3_GRAPH_SINGLE_ROUND_B1_B8_NO_HTTP_20260828.md)
- [RESULTS_E3_RERUN_20260825.md](RESULTS_E3_RERUN_20260825.md)
- [RESULTS_E3_NATIVE_COPY_VS_UVA_PHASE_20260828.md](RESULTS_E3_NATIVE_COPY_VS_UVA_PHASE_20260828.md)
- [RESULTS_E3_PREFILL_GRAPH_COPY_VS_NATIVE_COPY_UVA_B1_B8_20260828.md](RESULTS_E3_PREFILL_GRAPH_COPY_VS_NATIVE_COPY_UVA_B1_B8_20260828.md)
- [RESULTS_E3_DECODE_THREE_MECHANISM_B1_B8_20260828.md](RESULTS_E3_DECODE_THREE_MECHANISM_B1_B8_20260828.md)
- [RESULTS_E3_D_VS_C_PAIRED_ATTEMPT02_20260831.md](RESULTS_E3_D_VS_C_PAIRED_ATTEMPT02_20260831.md)
- [RESULTS_E3_KTRANSFORMERS_TUNED_20260903.md](RESULTS_E3_KTRANSFORMERS_TUNED_20260903.md)
- [RESULTS_E3_D_VS_C_FULL_GRAPH_PAIRED_ATTEMPT02_20260831.md](RESULTS_E3_D_VS_C_FULL_GRAPH_PAIRED_ATTEMPT02_20260831.md)
- [RESULTS_E3_KTRANSFORMERS_C1_C16_20260902.md](RESULTS_E3_KTRANSFORMERS_C1_C16_20260902.md)
- [RESULTS_E2_E3_NATIVE_UVA_CUDAGRAPH_20260903.md](RESULTS_E2_E3_NATIVE_UVA_CUDAGRAPH_20260903.md)
- [RESULTS_E3_NEAR_EQUAL_OFFLOAD_DUAL_GRAPH_20260903.md](RESULTS_E3_NEAR_EQUAL_OFFLOAD_DUAL_GRAPH_20260903.md)
- [RESULTS_E3_NEAR_EQUAL_NATIVE_COPY_GRAPH_20260903.md](RESULTS_E3_NEAR_EQUAL_NATIVE_COPY_GRAPH_20260903.md)

### 6.8 等字节同协议Prefill复跑（稳定快态）

2026-09-09汇总Short 128-token、Ordinary 512-token、Medium 2048-token、RAG
8192-token和Long 512-token的B1/B8稳定快态。两臂统一使用`TRITON_ATTN`和
`FULL_SEPARATE` Graph；AdaptMoE按48层exact expert slices搬运，Native
copy-on-forward按whole Parameters搬运。除Short/B8外，逐cell两臂实际搬运字节完全
相等；Short/B8的AdaptMoE/Native分别为2,491,416,576/2,493,476,896 B，差0.083%，
仍在预设0.1%门内。Medium/B8和RAG/B1两臂均为8,606,711,808 B，RAG/B8两臂均为
14,722,007,040 B，其余点约为2.491 GiB。

每个机制/Batch纳入2个独立进程块，按AB/BA顺序；每进程先执行1次完整真实
conditioning，再执行10次measurement。表中steady值先取每个进程10次measurement的
中位数，再对两个独立进程块取中位数。单位为ms；倍率为`Native/AdaptMoE`，小于1表示
Native更快。

| 请求 | B | AdaptMoE首请求 | Native首请求 | AdaptMoE steady（两块） | Native steady（两块） | steady Native-AdaptMoE | Native/AdaptMoE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Short 128 | 1 | 72.659 | 72.109 | 68.691（68.719/68.662） | 63.732（63.731/63.733） | -4.959 | 0.928× |
| Short 128 | 8 | 88.859 | 86.664 | 80.419（80.487/80.351） | 75.177（75.154/75.200） | -5.242 | 0.935× |
| Ordinary 512 | 1 | 85.596 | 72.750 | 74.807（74.782/74.832） | 69.639（69.661/69.617） | -5.168 | 0.931× |
| Ordinary 512 | 8 | 154.134 | 144.267 | 144.973（144.849/145.097） | 138.861（138.799/138.922） | -6.112 | 0.958× |
| Medium 2048 | 1 | 117.552 | 108.874 | 113.065（113.085/113.046） | 107.584（107.603/107.566） | -5.481 | 0.952× |
| Medium 2048 | 8 | 649.080 | 639.155 | 641.009（640.441/641.577） | 659.977（659.896/660.058） | +18.968 | 1.030× |
| RAG 8192 | 1 | 584.815 | 575.166 | 585.759（585.003/586.515） | 577.298（577.372/577.225） | -8.461 | 0.986× |
| RAG 8192 | 8 | 3754.223 | 3737.814 | 3780.758（3777.779/3783.737） | 3773.847（3773.871/3773.823） | -6.911 | 0.998× |
| Long 512 | 1 | 77.143 | 71.061 | 75.337（75.345/75.329） | 70.127（70.131/70.124） | -5.209 | 0.931× |
| Long 512 | 8 | 150.465 | 143.937 | 144.821（144.782/144.859） | 138.792（138.827/138.757） | -6.029 | 0.958× |

稳定快态下，Native在Short/B1、Short/B8、Ordinary/B1、Ordinary/B8和Medium/B1的
steady延迟分别低4.959/5.242/5.168/6.112/5.481 ms，相对AdaptMoE低
7.22%/6.52%/6.91%/4.22%/4.85%。Medium/B8则由AdaptMoE领先18.968 ms，相对Native
低2.87%。Native在RAG/B1、RAG/B8、Long/B1和Long/B8还分别领先
8.461/6.911/5.209/6.029 ms，相对AdaptMoE低1.44%/0.18%/6.91%/4.16%。第一个真实请求
仍是Native在十点均更快，低0.550--16.409 ms。因此本轮支持的结论是：相同backend、
Graph模式和近等搬运字节下，Native copy-on-forward在9/10个稳定快态点略快；唯一反转
为Medium/B8。优劣仍随Batch和搬运规模变化，不能写成Native逐cell必胜。

表中纳入的40个case进程均为`status=ok`，逐cell跨机制完整输出hash一致；每进程的全部
11次请求均实际FULL Graph replay，shutdown后的driver session为0。Medium定向复跑的
聚合检查还确认8/8 case完成、字节误差为0、无Xid/MMU/PDE/PTE/CUDA fault，最终GPU为
0 MiB并恢复stock UVM；RAG/Long主矩阵及RAG/B1补充轮也分别为16/16和4/4 case完成，
聚合`status=ok`、无kernel fault且恢复stock UVM。Short/Ordinary来自整体未完成的
Attempt-01/03，本节只引用其中已完成且通过逐case契约的进程，不把两个artifact根标记为
完整矩阵成功。

本节明确限定为AdaptMoE稳定快态，并没有删除不利样本：未纳入的同协议独立进程曾出现
Short/B8 127.096 ms、Ordinary/B8 202.836 ms、Medium/B1约172 ms及Medium/B8
840.102 ms的steady慢态；RAG/B1主矩阵也出现AdaptMoE 783.313 ms慢态，对应快态为
585.925 ms。RAG/B1的Native在主矩阵约603.2--603.5 ms、补充快态轮约577.2--577.4 ms，
说明该大搬运点并非只有AdaptMoE存在跨进程状态变化。慢态触发概率和根因尚未确定，因此
表中结果不能代表全部冷启动分布，也不能用于总体期望延迟排名。每点两个独立进程块只够
诊断复现，没有稳定置信区间。此外，两臂虽等字节或误差小于0.1%，但expert slices与
whole Parameters的坐标和DMA粒度仍不同，不能解释为纯Graph或纯copy单变量消融。

来源：

- `artifacts/e3-prefill-transport-fair-rerun-medium-20260909-attempt-01/summary.json`
- `artifacts/e3-prefill-transport-fair-rerun-medium-20260909-attempt-01/execution.tsv`
- `artifacts/e3-prefill-transport-fair-rerun-20260909-attempt-03/execution.tsv`
- `artifacts/e3-prefill-transport-fair-rerun-20260909-attempt-01/execution.tsv`
- `artifacts/e3-prefill-transport-fair-rerun-rag-long-20260909-attempt-01/summary.json`
- `artifacts/e3-prefill-transport-fair-rerun-rag-b1-supplement-20260909-attempt-01/summary.json`

## 7. E4：并发与PCIe饱和

最近一次5个独立进程复验使用固定always-remote计划，扫描C1--C32。预注册的持续退化判据
在所有并发档位均未触发。

| C | request TPOT | Decode tok/s | PCIe RX |
|---:|---:|---:|---:|
| 1 | 87.41 ms | 11.60 | 0.166 GiB/s |
| 4 | 87.96 ms | 45.89 | 0.397 GiB/s |
| 8 | 89.80 ms | 89.87 | 1.387 GiB/s |
| 16 | 89.75 ms | 180.33 | 2.603 GiB/s |
| 24 | 85.02 ms | 284.77 | 3.841 GiB/s |
| 32 | 86.56 ms | 372.94 | 4.698 GiB/s |

截至C32，没有观察到TPOT持续恶化；PCIe RX峰值约为Gen5 x16理论单向带宽的8.00%。
CPU DRAM带宽因KVM不暴露IMC而为N/A。结论只覆盖该模型、请求和2.2649-GB remote计划。

来源：`artifacts/e4-repro-20260902-attempt-01/analysis/`

## 8. E5：四项核心机制消融

2026-09-03在当前custom UVM上重新完成四项核心机制。模型为Qwen3-30B-A3B-FP8，主要
坐标为B1/B4，每个性能臂3个独立冷进程，统一verifier为`status=ok`。

| 核心机制 | B1 | B4 |
|---|---:|---:|
| P/D双视图PDE直接切换：DESTROY/rebuild ÷ RESUME | 2,259.59× | 2,014.76× |
| CUDA Graph：Prefill eager ÷ Graph | 1.76× | 1.24× |
| CUDA Graph：Decode eager ÷ Graph | 2.42× | 2.13× |
| 地址稳定：non-stable ÷ stable重建路径 | 1.49× | 1.35× |
| 页表提前构建：runtime-install ÷ prebuilt RESUME | 29.76× | 26.87× |

P/D直接切换与页表预构建存在依赖关系，不能把表中倍率相乘。完整快速切换臂使用已预构建
的双视图，RESUME为B1/B4 3.399/3.471 ms；DESTROY/rebuild为
7,679.761/6,993.427 ms。关闭prebuild后，runtime range-install RESUME为
101.136/93.255 ms，仍比rebuild快75.93×/74.99×；on/off对照再隔离出prebuild的
29.76×/26.87×增益。

Graph/eager使用相同48层cold-5 placement和1.055 GiB选中权重。所有A/G性能进程都设置
`VLLM_CPU_OFFLOAD_COPY_TRACE=0`，逐forward记录数为0；初始化参数证书保留。B1/B4的
Prefill从75.732/72.394 ms降至42.943/58.187 ms，Decode从
74.049/73.501 ms降至30.656/34.465 ms。契约确认Graph臂的Prefill和Decode全部实际
replay，eager臂均未replay。

Graph/eager在各自执行类和三进程内输出稳定，B1跨臂hash一致；B4 eager与Graph的hash不同。
因此Graph倍率用于固定step调度的执行开销消融，不作为B4 exact-generation或数值质量等价
证据。

地址稳定性臂使用相同2.416 GB逻辑alias和128 MiB KV。stable VA的B1/B4重建中位数为
7,679.761/6,993.427 ms；每代8/8地址变化并重捕获Graph时为
11,474.982/9,411.040 ms，分别增加3,795.220/2,417.613 ms。

prebuild on/off只切换dual-view预构建标志并固定4 KiB PTE。on臂为1,152个2 MiB parent、
100% coverage和`prebuilt_full_dual_view`，off臂为`audited_range_install`。预构建使
Cutover快1.50×--1.65×、Re-PARK快1.71×--1.86×，稳态Prefill/Decode均约1.00×，说明
它主要优化控制面切换而不改变稳态数据路径。

所有正式根各自适用的输出稳定性、fault scan、资源释放和模块恢复门通过；最终恢复stock UVM
`85A79790636BBD99BA3E43B`，GPU 0 MiB且无compute/UVM client。三冷进程结果不能外推为
跨模型或跨硬件的普遍倍率。

来源：

- [RESULTS_E5_CORE_FOUR_MECHANISM_ABLATION_20260903.md](RESULTS_E5_CORE_FOUR_MECHANISM_ABLATION_20260903.md)
- `artifacts/e5-core-four-mechanism-verification-r3-20260903-v3/verification.json`
- `artifacts/e5-core-graph-eager-no-trace-formal-r3-20260903/verification.json`
- [RESULTS_E5_E12_CURRENT_MODULE_QUALIFICATION_20260831.md](RESULTS_E5_E12_CURRENT_MODULE_QUALIFICATION_20260831.md)

### 8.1 单冷进程端到端递进消融补充（2026-09-16）

为把控制面倍率与完整请求周期联系起来，补充一轮统一配置的端到端递进消融。四臂均使用
Qwen3-30B-A3B-FP8、同一冻结fixture、同一组8个完整参数（逻辑字节
2,415,919,104 B）、128 MiB KV、B1/B4和16个measured Decode step。每个cell只运行
一个独立冷进程，不报告置信区间；按用户要求，跨臂语义一致性不作为本轮验收门。为兼容
当前模块新增的headroom guard，所有臂统一使用32 MiB资格阈值；运行期间没有实际CUDA OOM。

| B | 路径 | Prefill | P→D | steady Decode | Decode tok/s | cycle E2E |
|---:|---|---:|---:|---:|---:|---:|
| 1 | H：eager Prefill + eager remote Decode | 206.77 ms | 110.69 ms | 63.49 ms | 15.75 | 1,967.50 ms |
| 1 | G：eager Prefill + remote Decode Graph | 162.67 ms | 110.86 ms | 11.39 ms | 87.78 | 544.37 ms |
| 1 | D：Prefill/Decode Graph + prebuilt双页表 | 70.29 ms | 15.40 ms | 11.44 ms | 87.39 | 357.81 ms |
| 1 | D-no-prebuild：Prefill/Decode Graph + runtime install | 69.93 ms | 26.75 ms | 11.10 ms | 90.06 | 361.92 ms |
| 4 | H：eager Prefill + eager remote Decode | 164.84 ms | 110.40 ms | 65.26 ms | 61.30 | 1,950.68 ms |
| 4 | G：eager Prefill + remote Decode Graph | 147.03 ms | 111.49 ms | 22.50 ms | 177.76 | 751.20 ms |
| 4 | D：Prefill/Decode Graph + prebuilt双页表 | 84.95 ms | 15.40 ms | 22.10 ms | 180.96 | 590.99 ms |
| 4 | D-no-prebuild：Prefill/Decode Graph + runtime install | 84.75 ms | 26.83 ms | 22.04 ms | 181.45 | 600.42 ms |

H→G只改变Decode Graph路径，steady Decode在B1/B4分别提高5.57×/2.90×，cycle E2E提高
3.61×/2.60×。G→D同时加入Prefill Graph和prebuilt快速P→D，cycle E2E再提高
1.52×/1.27×；该倍率是两个机制的组合效果，不能写成纯Prefill Graph消融。

关闭prebuild后，B1/B4的RESUME从2.71/3.11 ms增加到54.68/53.63 ms，即慢
20.16×/17.23×；P→D慢1.74×/1.74×，但稳态Prefill/Decode保持0.97×--1.00×，最终
cycle E2E只增加1.15%/1.60%。这说明prebuild的主要收益位于控制面，单次切换在当前短周期
中的端到端贡献有限。

同一轮还各运行一个stable-VA rebuild和non-stable reallocate/recapture冷进程。B1/B4的
stable重建为3,590.90/3,219.29 ms，non-stable为7,976.31/5,988.53 ms；non-stable慢
2.22×/1.86×。以prebuilt D的单次P→D为参照，stable DESTROY/rebuild仍慢233.16×/209.00×。
这些均为单次点估计，只用于补足递进消融，不替代上面的三冷进程正式E5结果。

补充轮来源：

- `artifacts/e5-e2e-graph-ablation-once-20260916-attempt-04/verification.json`
- `artifacts/e5-e2e-prebuild-on-once-20260916-attempt-04/verification.json`
- `artifacts/e5-e2e-prebuild-off-once-20260916-attempt-01/verification.json`
- `artifacts/e5-stable-va-rebuild-once-20260916-attempt-01/verification.json`
- `artifacts/e5-nonstable-va-recapture-once-20260916-attempt-01/verification.json`

## 9. E6：Plan Class与切换摊销

full rebuild、serial wave和batched difference三臂按相同冻结几何各完整运行一次，三个
verifier均为`status=ok`，输出hash一致。

| Difference | Full rebuild | Serial wave | Batched difference |
|---|---:|---:|---:|
| 5% | 17.761/18.437 s | 2.005/2.156 s | 2.010/2.101 s |
| 10% | 18.968/19.024 s | 4.018/4.130 s | 2.677/2.758 s |
| 25% | 20.068/19.238 s | 10.278/10.409 s | 4.795/4.793 s |
| 50% | 21.299/21.187 s | 21.070/20.360 s | 8.506/8.129 s |
| 100% | 24.312/23.957 s | 39.554/39.698 s | 14.969/14.987 s |

表中为outbound/return。batched difference相对full rebuild快1.60×--8.84×；10%--100%
相对serial wave快1.50×–2.65×，5%两者近似相同。full rebuild把Graph capture count从
2增加到22，serial和batched均保持2。

这是单进程、单样本qualification，不能把点估计当作median或置信区间。

来源：`artifacts/e6-summary-20260902-reproduce-all-attempt-01-latest-dual/summary.json`

## 10. E7：专家热度与workload shift

最新实验让六个arm分别使用独立冷进程，每arm执行一次32-request shift trace，六臂输出和
路由计数hash一致，Graph count保持2。

| Arm | 请求中位延迟 | cycle tok/s | 动态remote命中率 | Oracle plan hit | 同步维护 |
|---|---:|---:|---:|---:|---:|
| frequency-cold | 838.62 ms | 35.15 | 1.427% | 36.98% | 90.75 ms |
| random | 868.77 ms | 34.14 | 3.186% | 4.17% | 74.37 ms |
| frozen static | 804.58 ms | 36.30 | 1.688% | 18.23% | 72.10 ms |
| GPU heat + CPU EWMA | 851.22 ms | 20.97 | 0.712% | 88.54% | 19,765.65 ms |
| oracle | 811.88 ms | 35.98 | 0.178% | 100% | 73.36 ms |
| hot-adversarial | 1002.28 ms | 31.54 | 12.484% | 0% | 72.12 ms |

Adaptive将plan hit从36.98%提高到88.54%，动态remote assignments减少50.12%。但请求
中位延迟仍比frequency-cold高1.50%，同步维护累计19.766 s，使cycle throughput从35.15
降至20.97 tok/s，即下降40.35%，且32请求内未达到95%收敛门。

因此E7证明放置方向和机制链路正确，尚不能证明包含维护成本后的端到端性能无回退。

来源：`artifacts/e7-independent-cold-20260902-attempt-25-targeted-canonical-pte-qualification/verification.json`

## 11. E8：MoE-Infinity cache/prefetch

E8仍阻塞在provenance门：缺少与当前活动扩展、commit和约30-GiB store匹配的fresh build
receipt。preflight返回码1，实验没有绕过commit、patch、ELF或receipt hash检查。

状态：无性能数字，不能声称E8通过或失败于算法性能。

来源：[REPORT_E2_E12_REPRO_20260827.md](REPORT_E2_E12_REPRO_20260827.md)

## 12. E9：KTransformers placement/CPU消融

2026-09-02仅启动一次正式矩阵。baseline、threads7、threads28、eager、uniform、
no-overlap和no-mixed七臂按轮转正反顺序各运行5个独立冷进程，共35/35 case、70/70个
C1/C8 phase成功。每个phase生成32 tokens/request，共完成10,080/10,080个正式输出token；
顶层verifier为`status=ok`。下表为5个冷进程的中位数，括号内为min--max：

| Arm | C1吞吐 tok/s | C1 TTFT | C1 TPOT | C8吞吐 tok/s | C8 TTFT | C8 TPOT |
|---|---:|---:|---:|---:|---:|---:|
| baseline：14线程、Graph、frequency、overlap、mixed | 7.067 (5.977--12.560) | 2701.39 ms | 57.71 ms | 6.552 (5.959--6.679) | 8114.47 ms | 1036.41 ms |
| threads7 | 32.706 (29.012--35.385) | 266.36 ms | 22.86 ms | 39.924 (39.271--42.317) | 4256.57 ms | 67.00 ms |
| threads28 | 4.124 (3.649--4.564) | 5427.46 ms | 76.54 ms | 3.141 (2.917--4.355) | 13227.44 ms | 2195.18 ms |
| eager | 3.228 (2.746--3.643) | 2181.47 ms | 247.44 ms | 8.030 (5.681--12.211) | 6947.83 ms | 792.70 ms |
| uniform placement | 1.661 (0.907--2.436) | 5396.92 ms | 445.17 ms | 4.541 (4.036--6.659) | 19939.26 ms | 1209.07 ms |
| no-overlap | 11.562 (7.556--12.537) | 986.24 ms | 57.45 ms | 10.405 (8.868--12.703) | 7024.86 ms | 552.44 ms |
| no-mixed | 12.198 (11.460--12.727) | 935.99 ms | 53.04 ms | 6.746 (6.035--15.093) | 7208.17 ms | 998.79 ms |

CPU线程数是本轮最强主效应。相对14线程baseline，7线程的中位吞吐在C1/C8分别为
4.63×和6.09×，而28线程仅为0.584×和0.479×。该机器只有14个物理核并固定
`taskset -c 0-13`，因此28线程会过度订阅CPU；7线程反而减少线程调度和共享内存带宽
争用。frequency相对uniform placement的中位吞吐为C1 4.26×、C8 1.44×，说明冻结的
请求热度放置明显优于逐层均匀预算。

其余消融不能简化成“Graph、overlap或mixed必然更快”。Graph baseline在C1比eager快
2.19×，但C8 eager中位数反而为1.23×，且eager C8范围很宽；关闭overlap的中位吞吐为
baseline的1.64×/1.59×，关闭mixed为1.73×/1.03×。这说明当前300--320-token短Prompt、
32-token输出和CPU专家路径下，默认overlap/mixed配置没有形成稳定收益，部分点还落入
明显的运行态快慢区间。5点bootstrap 95% CI和全部原始样本保存在`verification.json`，
不把这些点估计写成统计显著性证明。

正确性门按数值路径分层：baseline、threads7、threads28、eager和no-overlap不改变专家
放置或batch formation，五臂在所有5个block的C1/C8均与baseline逐token exact；全部七臂
各自在5个冷进程间也都bitwise稳定。uniform改变CPU/GPU专家归约路径，C1/C8相对baseline
的exact request比例为0/5和10/40，token-position agreement为90.625%和49.609%；no-mixed
改变batch formation，C1为5/5 exact，C8为35/40 exact、88.281% token-position agreement。
因此本轮通过的是完整E9 execution/performance qualification，不是uniform/no-mixed的
teacher-forced logits、top-1或perplexity质量等价证明；全七臂跨臂exact仍为false。

所有case-level runtime verifier和kernel日志通过，峰值HBM为26,385--26,667 MiB；无Xid、
PDE/PTE fault或清理泄漏，最终GPU为0 MiB。跨NUMA臂因本机只有一个NUMA node，按协议
记为N/A。

状态：完整5-block E9 formal execution qualification通过；数值路径变化臂保留质量边界。

来源：`artifacts/e9-formal-20260902-attempt-01/verification.json`

## 13. E10：2→4→6 GiB联合HBM压力切档

最新E10完整执行一条cold70/2 GiB→cold80/4 GiB→cold89/6 GiB压力触发序列，verifier
`status=ok`。三档全覆盖双页表均为100%，没有CPU range-map回退、Graph重捕获或内核错误。

| KV | 纯Prefill | 切档 | P→D | steady Decode | 请求E2E |
|---:|---:|---:|---:|---:|---:|
| 2 GiB | 340.74 ms | 0 | 108.40 ms | 20.11 ms | 1007.96 ms |
| 4 GiB | 399.37 ms | 542.80 ms | 117.82 ms | 24.25 ms | 1672.99 ms |
| 6 GiB | 447.60 ms | 565.27 ms | 138.35 ms | 26.78 ms | 1831.87 ms |

结果证明当前档双页表能在scheduler quiescent point按相邻约2 GiB同步切换KV容量与专家
Plan Class，并保持三档100%覆盖。完整档位事务仍需约543--565 ms，P→D为108--138 ms，
不能称为已经消除切档成本。

本轮由冻结的压力序列触发三档转换，用于验证当前档双页表和切档路径。它尚未比较fixed、
oracle、auto无backup、auto+backup和full-joint等正式E10策略，也没有K2/K4/K8/K10重复
峰谷、5 seeds、open-loop到达或goodput/SLO统计。因此只能写作E10 preflight/qualification，
不能写成自动联合HBM控制器已经完成。

来源：`artifacts/e10-kv-2-4-6-pressure-20260902-attempt-01-scheduler-quiescent-sync-tier/verification.json`

## 14. E11：continuous arrival与phase epoch

最近一次完整attempt为
`e11-continuous-arrival-20260902-attempt-09-remote-park-full-dual-strict-qualification`，
证明了调度行为，但顶层verifier保持`status=error`：

| 指标 | phase epoch + phase-specific | phase epoch + remote-only | continuous + remote-only |
|---|---:|---:|---:|
| 成功请求 | 8/8 | 8/8 | 8/8 |
| mixed steps | 0 | 0 | 1 |
| TTFT p50 | 406.31 ms | 613.60 ms | 153.10 ms |
| TPOT p50 | 54.58 ms | 77.55 ms | 81.88 ms |
| E2E p50 | 1225.00 ms | 1776.83 ms | 1381.30 ms |
| generated-token throughput | 61.02 tok/s | 48.75 tok/s | 72.97 tok/s |

phase epoch把新Prefill和旧Decode完全分开；另一个hybrid fail-closed arm在mixed request
上按预期拒绝执行。但phase-specific Full Prefill Graph与两条remote-only路径输出token IDs
不一致，不能进入性能排名。

状态：E11已按用户要求终止并撤销为完成它而做的后续修改。attempt 10--16只保留失败诊断，
不得晋升；当前只允许离线审计，不再启动GPU复现。

来源：`artifacts/e11-continuous-arrival-20260902-attempt-09-remote-park-full-dual-strict-qualification/verification.json`

## 15. E12：Graph replay与PARK/RESUME qualification

最近可用结果为3次warmup加1,000次measured Graph replays：

- full-byte compare错误数：0；
- GPU时间min/median/mean/max：45.8621/45.9045/45.9051/45.9544 ms；
- host submit median：77.664 µs；
- 每次逻辑H2D：2,415,919,104 B，对应49.015 GiB/s；
- consumer/upload VA保持稳定，allocator正常关闭。

当前模块还完成了1,000/1,000次完整双页表PARK/RESUME，逐轮验证Decode CPU-shadow和
Prefill GPU-backing视图。PARK p50/p95为0.769/0.838 ms，RESUME为0.567/0.611 ms；
共记录2,000次dual-view switch、8,000次PDE写入和2,000个TLB batches。无Xid、PDE/PTE
fault或资源泄漏，并成功恢复stock UVM和GPU 0 MiB。

状态：Graph replay smoke与1,000-cycle机制qualification通过，但尚未完成6/24小时整模型
soak、重复KV峰谷、barrier故障注入或长期性能/HBM退化检查。1,000轮循环仅3.700 s，不能
把次数门写成时间soak通过。

来源：

- [RESULTS_E5_E12_CURRENT_MODULE_QUALIFICATION_20260831.md](RESULTS_E5_E12_CURRENT_MODULE_QUALIFICATION_20260831.md)
- [REPORT_E2_E12_REPRO_20260827.md](REPORT_E2_E12_REPRO_20260827.md)

## 16. E13：双阶段 CUDA Graph 下的 KV 档位与 Adaptive 回环

### 16.1 协议与验收

当前权威轮为
`artifacts/e13-short-b64-b128-full-separate-roundtrip-REPRO-20260902-attempt-02/`。
它在Qwen3-30B-A3B-FP8、Short-128/Output-32、B64/B128上依次运行Fixed-K2、Fixed-K6和
Adaptive三个独立冷进程。每个cell先运行一次排除warmup，再取三个进程内样本的中位数。

该轮启用`TRITON_ATTN`、`CUDAGraphMode.FULL_SEPARATE`、
`prefill_graph_remote_decode`、original-VA双页表以及PARK/RESUME快速P→D。Fixed每臂只
捕获当前Plan Class的3个Graph shape；Adaptive预捕获cold70/cold80/cold89三档共9个
Graph entry，并严格按`K2→K4→K6→K4→K2`执行四次相邻约2 GB的KV/专家同步切档。
初始化、四个档位事务和每个目标档的一次warmup均不进入性能计时。

三臂3/3返回0，正式测量2,880个请求，加排除warmup共3,840/3,840请求完成。正式服务的
975/975个step均为FULL Graph replay，捕获后没有Graph重捕获或eager fallback；84/84项
协议、输出、档位、fault和清理检查通过。输出hash一致，无Xid、PTE/PDE或MMU fault，
最终GPU为0 MiB并恢复stock UVM。

### 16.2 Fixed-K2与Fixed-K6

| B | 档位 | req/s | TTFT | TPOT | E2E | Prefill | P→D | steady Decode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | Fixed-K2 | 29.1489 | 726.68 ms | 46.59 ms | 2189.23 ms | 722.16 ms | 489.74 ms | 24.89 ms |
| 64 | Fixed-K6 | 25.3277 | 768.05 ms | 56.65 ms | 2519.78 ms | 763.26 ms | 470.04 ms | 32.67 ms |
| 128 | Fixed-K2 | 39.6909 | 972.30 ms | 72.17 ms | 3212.92 ms | 1208.64 ms | 492.87 ms | 41.12 ms |
| 128 | Fixed-K6 | 32.5717 | 1066.51 ms | 91.25 ms | 3917.92 ms | 1344.89 ms | 501.92 ms | 59.54 ms |

Fixed-K2在B64/B128的吞吐分别比K6高15.09%/21.86%，E2E分别低13.12%/17.99%。两个
Short workload都远未达到K2的KV容量限制，因此K6的额外KV容量没有收益；更多专家权重
卸载反而增加Prefill和remote Decode开销。该结论只覆盖当前低KV压力坐标，不能外推为
高压力时K2仍优于K6。

### 16.3 Adaptive切档后的稳态等价

下表均为`Adaptive/对应Fixed`；档位事务和目标档warmup已经排除。

| Adaptive状态 | B | req/s | Prefill | steady Decode | E2E |
|---|---:|---:|---:|---:|---:|
| 初始K2 | 64 | 0.988× | 1.005× | 0.982× | 1.012× |
| 增长后K6 | 64 | 0.965× | 1.010× | 1.019× | 1.037× |
| 回缩后K2 | 64 | 0.988× | 1.001× | 1.003× | 1.012× |
| 初始K2 | 128 | 0.979× | 1.004× | 1.001× | 1.021× |
| 增长后K6 | 128 | 0.971× | 1.004× | 1.020× | 1.030× |
| 回缩后K2 | 128 | 0.968× | 1.007× | 1.001× | 1.034× |

用户指定的吞吐、Prefill、steady Decode和E2E共24/24个A/F倍率均位于预注册的±10%
带内。因此可以声称：忽略档位事务及warmup后，Adaptive在初始K2、增长后K6和回缩后K2
均恢复到对应Fixed档的稳态水平，回缩没有留下稳态性能回退。

若把output tok/s、batch wall、P→D、TTFT、TPOT也纳入，则只有49/54在带内。5个超带点
全部是P→D：初始K2/B128为`1.117×`，增长后K6的B64/B128为`1.142×/1.240×`，回缩后
K2的B64/B128为`1.109×/1.138×`。P→D是每个batch首个Decode step中的PARK和
remote-map提交，不是已排除的K2↔K6档位事务；初始K2/B128在切档前也超带，所以现有数据
不能把差异唯一归因于切档历史，也不能声称P→D瞬时停顿完全等价。

四个排除的档位事务依次耗时7.272/6.992/7.671/6.800 s。本轮是每臂一个冷进程的单轮
qualification，三个cell样本是进程内重复；不能把±10%门写成统计等价检验，也不能声称
包含同步档位迁移成本后的Adaptive端到端性能已经与Fixed等价。

### 16.4 同一高压 trace 的 Fixed-K2、Fixed-K6 与 Adaptive-K6 补充

为直接检验KV容量不足时K6能否优于K2，补充轮将请求从低压Short-128/B1→32提高为
RAG-8192/B6→16，并在完全相同的高压trace上分别运行Fixed-K2、Fixed-K6和Adaptive-K6。
三臂均为独立冷进程；高压段先运行一次排除warmup，再取两次正式进程内测量的中位数。
初始化、Adaptive的`K2→K4→K6`切档事务和目标档warmup均不计时。本轮使用eager Prefill
和remote Decode，不是16.1--16.3的`FULL_SEPARATE`双Graph协议，因而只作为容量压力的
补充证据，不替换前述低压双Graph结果。

有效output吞吐按`req/s × 16 output tokens/request`换算。

| 高压臂 | output tok/s | TTFT | TPOT | 请求E2E | Prefill | Batch wall | steady Decode | 最大同时运行请求 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed-K2 | 14.7059 | 3287.71 ms | 71.25 ms | 4351.90 ms | 4411.44 ms | 6527.98 ms | 44.22 ms | 2/6 |
| Fixed-K6 | 17.0059 | 2860.51 ms | 185.58 ms | 5644.25 ms | 4887.45 ms | 5645.10 ms | 44.39 ms | 6/6 |
| Adaptive-K6 | 16.9321 | 2865.19 ms | 186.91 ms | 5668.87 ms | 4893.61 ms | 5669.75 ms | 45.29 ms | 6/6 |

在该高压trace上，Fixed-K2最多只能同时运行2/6个请求，而Fixed-K6可同时运行6/6个请求。
有效output吞吐由14.7059提高到17.0059 tok/s，即Fixed-K6相对Fixed-K2高15.64%；同时
Batch wall低13.52%，TTFT低12.99%。
这是容量和减少cohort分批带来的吞吐crossover；K6的单请求TPOT和请求E2E并不更低，不能
据此声称K6在所有延迟指标上都优于K2。

进入K6并完成warmup后，Adaptive-K6相对Fixed-K6的output tok/s为0.996×，Prefill、steady
Decode、请求E2E和Batch wall分别为1.001×、1.020×、1.004×和1.004×，说明目标档
稳态与Fixed-K6接近。三臂返回码均为0，verifier为`status=ok`且38/38项检查通过；输出
hash一致，无Xid、PTE/PDE或MMU fault，结束后GPU显存为0 MiB并恢复stock UVM。该轮每臂
只有一个冷进程、两个正式进程内样本，不能作为置信区间或统计等价证明。

### 16.5 同一请求下的 K2/K4/K6/Adaptive 三档压力矩阵

为补齐Fixed-K4并消除16.2与16.4混用请求类型的问题，最新补充轮在四个独立冷进程中
统一使用Qwen3-30B-A3B-FP8、RAG-8192→16、eager Prefill和remote Decode，只改变并发量：
B1为低压，B5为中压，B6为高压。Fixed三臂始终固定在K2/K4/K6；Adaptive依次进入
K2→K4→K6。K2/K4/K6的KV target分别为1364/2730/4095 blocks，对应cold70/cold80/cold89。
初始化、两次相邻切档事务及每个目标档的一次warmup均不计时；下表取每cell两次正式
进程内测量的中位数，有效吞吐按`B × 16 × 1000 / Batch wall`计算。

| 压力 | B | 机制 | output tok/s | TTFT | TPOT | 请求E2E | Prefill | Batch wall | steady Decode | 最大同时运行 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 低 | 1 | Fixed-K2 | 9.2729 | 744.64 ms | 65.37 ms | 1725.25 ms | 744.36 ms | 1725.46 ms | 61.38 ms | 1/1 |
| 低 | 1 | Fixed-K4 | 8.8915 | 793.18 ms | 67.07 ms | 1799.29 ms | 792.85 ms | 1799.48 ms | 62.48 ms | 1/1 |
| 低 | 1 | Fixed-K6 | 8.6103 | 835.48 ms | 68.17 ms | 1858.07 ms | 835.18 ms | 1858.25 ms | 62.78 ms | 1/1 |
| 低 | 1 | Adaptive | 9.2388 | 746.46 ms | 65.68 ms | 1731.65 ms | 746.15 ms | 1731.83 ms | 61.76 ms | 1/1 |
| 中 | 5 | Fixed-K2 | 12.0440 | 3200.62 ms | 66.61 ms | 4920.67 ms | 3684.44 ms | 6642.33 ms | 61.12 ms | 2/5 |
| 中 | 5 | Fixed-K4 | 16.2643 | 2333.30 ms | 172.30 ms | 4917.75 ms | 3872.37 ms | 4918.77 ms | 64.32 ms | 5/5 |
| 中 | 5 | Fixed-K6 | 15.6618 | 2456.02 ms | 176.73 ms | 5106.94 ms | 4074.63 ms | 5107.98 ms | 63.07 ms | 5/5 |
| 中 | 5 | Adaptive | 16.3645 | 2334.98 ms | 170.17 ms | 4887.60 ms | 3875.02 ms | 4888.63 ms | 62.41 ms | 5/5 |
| 高 | 6 | Fixed-K2 | 13.0513 | 3560.98 ms | 89.48 ms | 4900.84 ms | 4413.24 ms | 7355.62 ms | 61.48 ms | 2/6 |
| 高 | 6 | Fixed-K4 | 14.3325 | 2721.00 ms | 145.24 ms | 4899.84 ms | 4670.77 ms | 6698.06 ms | 62.18 ms | 5/6 |
| 高 | 6 | Fixed-K6 | 16.2004 | 2863.26 ms | 204.09 ms | 5924.58 ms | 4889.71 ms | 5925.78 ms | 63.10 ms | 6/6 |
| 高 | 6 | Adaptive | 16.1300 | 2865.88 ms | 205.64 ms | 5950.41 ms | 4893.38 ms | 5951.64 ms | 64.44 ms | 6/6 |

固定档实测排序为：低压`K2 > K4 > K6`，中压`K4 > K6 > K2`，高压`K6 > K4 > K2`。
中压Fixed-K4吞吐比次优Fixed-K6高3.85%、比容量不足的Fixed-K2高35.04%；高压
Fixed-K6比K4/K2高13.03%/24.13%。Adaptive相对各压力下最佳Fixed档的吞吐分别为
0.996×、1.006×和0.996×。因此数据支持“最佳档随压力K2→K4→K6切换，Adaptive跟随
对应档”的crossover，但不支持把中压次序写成`K4 > K2 > K6`。

![E13三档压力吞吐曲线](../../../experiments/eurosys_moe_offload_evaluation/artifacts/e13-rag-pressure-k2-k4-k6-adaptive-20260913-attempt-01/analysis/e13_rag_pressure_throughput.svg)

四臂4/4返回0，verifier为`status=ok`且56/56项检查通过；同一压力下输出hash跨四臂一致，
Adaptive切档路径为1364→2730→4095，无Xid、PTE/PDE、MMU fault或illegal address，最终
GPU显存为0 MiB并恢复stock UVM。这仍是每臂一个冷进程、每cell两个进程内样本的单轮
qualification，不是独立重复统计或置信区间证明。

### 16.6 RAG-8192 并发 B1--B8 四臂 sweep

在16.5完全相同的模型、冻结请求、16-token输出、eager Prefill、remote Decode和KV/专家
档位下，将并发量逐点扩展为B1--B8。Fixed-K2/K4/K6在整条曲线上保持固定；Adaptive按
预注册容量规则在B1--2使用K2、B3--5使用K4、B6--8使用K6。每个cell先运行一次排除
warmup，再取两次正式进程内样本的中位数；切档事务不计时。

| B | Fixed-K2 output tok/s | Fixed-K4 | Fixed-K6 | Adaptive | 最佳Fixed | Adaptive/目标Fixed | 最大同时运行 K2/K4/K6/A |
|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | 9.1461 | 8.7913 | 8.5824 | 9.1761 | K2 | 1.003× | 1/1/1/1 |
| 2 | 12.9134 | 12.2700 | 11.9021 | 12.9308 | K2 | 1.001× | 2/2/2/2 |
| 3 | 11.3806 | 14.2625 | 13.7235 | 14.2703 | K4 | 1.001× | 2/3/3/3 |
| 4 | 12.8793 | 15.4177 | 14.8252 | 15.4286 | K4 | 1.001× | 2/4/4/4 |
| 5 | 11.9269 | 16.1904 | 15.5564 | 16.2205 | K4 | 1.002× | 2/5/5/5 |
| 6 | 12.8616 | 14.1677 | 16.1145 | 16.1142 | K6 | 1.000× | 2/5/6/6 |
| 7 | 12.1133 | 14.8406 | 16.5002 | 16.4843 | K6 | 0.999× | 2/5/7/7 |
| 8 | 12.8005 | 15.3800 | 14.7491 | 14.7335 | K4 | 0.999× | 2/5/7/7 |

B1--2、B3--5和B6--7分别由K2、K4和K6获得最高Fixed吞吐，Adaptive在这些点与目标Fixed
相差不超过0.33%。B8时K6和Adaptive最多只能同时运行7/8个请求；Fixed-K4虽然只能运行
5/8，但更少的远端专家开销使其吞吐达到15.3800 tok/s，比K6高4.28%。Adaptive仍按容量
规则选择K6，吞吐14.7335 tok/s，比该点全局最佳Fixed-K4低4.20%。因此完整曲线支持
K2→K4→K6的前两次crossover，但也表明“最大可驻留档”在超过最高档容量后不必然是吞吐
最优，需要在控制器中同时考虑分批数和远端权重成本。

![E13 B1--B8四臂吞吐曲线](../../../experiments/eurosys_moe_offload_evaluation/artifacts/e13-rag-b1-b8-k2-k4-k6-adaptive-20260913-attempt-02/analysis/e13_rag_b1_b8_throughput.svg)

四臂4/4返回0，verifier为`status=ok`且101/101项检查通过；B1--B8各点的输出hash均跨
四臂一致，Adaptive切档路径为1364→2730→4095。无Xid、PTE/PDE、MMU fault或illegal
address，结束后显存为0 MiB并恢复运行前stock UVM。完整TTFT、TPOT、Prefill、请求E2E、
Batch wall和steady Decode见本轮`summary.md`；统计边界仍为每臂一个冷进程、每cell两个
进程内样本。

来源：

- [RESULTS_E13_KV_ADAPTIVE_20260901.md](RESULTS_E13_KV_ADAPTIVE_20260901.md)
- `artifacts/e13-short-b64-b128-full-separate-roundtrip-REPRO-20260902-attempt-02/verification.json`
- `artifacts/e13-k2-k6-high-pressure-roundtrip-20260913-attempt-01/{verification.json,summary.md}`
- `artifacts/e13-rag-pressure-k2-k4-k6-adaptive-20260913-attempt-01/{verification.json,summary.md,analysis/e13_rag_pressure_throughput.{csv,svg}}`
- `artifacts/e13-rag-b1-b8-k2-k4-k6-adaptive-20260913-attempt-02/{verification.json,summary.md,analysis/e13_rag_b1_b8_throughput.{csv,svg}}`
- [详细复现文档](../reproduce/REPRODUCE_E13_FULL_SEPARATE_K2_K6_ADAPTIVE_ROUNDTRIP.md)

## 17. E14：同请求集冷专家与固定整层卸载的 eager UVA 扩展性

当前权威轮为
`artifacts/e14-uva-eager-request-cold29-direct-formal-20260903-attempt-01/`。两臂均为
Qwen3-30B-A3B-FP8、original-VA managed UVA `remote_read`、`enforce_eager=True`、
`CompilationMode.NONE`和`CUDAGraphMode.NONE`；Prefill与Decode都没有CUDA Graph。
B1/B4/B16/B32分别使用1/4/16/32个不同冻结请求，每请求取8-token Prefill前缀。

在正式运行前，`artifacts/e14-request-routes-b32-20260903-attempt-01/`只统计一个完整
B32 measurement窗口：48层各22条记录，共1,056条、356,352个Top-8路由槽。每层按综合
Prefill+Decode命中数最低选择29个专家；最终1,392个坐标承载3,680个槽（1.033%），
其中Prefill占1.244%、Decode占0.952%，541个坐标零命中。新旧计划仅重叠739个坐标，
653个坐标被替换。正式性能进程关闭路由跟踪，三个D进程的实际专家ID均逐层匹配新计划。

逐层臂D在48个MoE层各卸载request-specific cold-29，共6.117 GiB；固定整层臂W完整卸载
L0--L10，共6.188 GiB，权重字节差1.15%。两臂都启用remote-only direct forward：初始化
完成并审计远端映射后直接执行原始`module.forward`，关闭逐forward通用wrapper和JSON
trace。按`DWWDDW`运行两臂各3个独立冷进程，每个B先conditioning一次、正式测量三次；
先取进程内中位数，再取跨进程中位数。

| B | D Prefill tok/s | W Prefill tok/s | D/W | D Decode tok/s | W Decode tok/s | D/W |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 130.71 | 136.41 | 0.958× | 16.01 | 15.91 | 1.006× |
| 4 | 498.50 | 332.86 | 1.498× | 61.27 | 64.17 | 0.955× |
| 16 | 2085.27 | 1000.97 | 2.083× | 246.81 | 196.48 | 1.256× |
| 32 | 4150.22 | 1785.98 | 2.324× | 514.10 | 314.07 | 1.637× |

B1时D的Prefill低4.2%、Decode高0.6%，符合低并发基本一致的目标，但不是预注册统计等价
检验。B32时D分别快132.4%/63.7%；D/W倍率从B1到B32分别增加136.6/63.1个百分点。
相对B1的B32线性扩展效率，D为Prefill 99.2%、Decode 100.3%，W仅为40.9%/61.7%。因此数据支持
“整层臂随并发增长仍增加绝对吞吐，但扩展效率明显下降，而逐层冷专家臂基本保持线性”这一
formal结果；不能声称W吞吐完全停止增长。Decode在B4仍基本一致，到B16/B32出现逐层臂
优势。四项趋势门全部通过。

旧formal仍启用逐forward wrapper和文件trace，B1 D/W只有0.827×/0.807×；新轮收敛到
0.958×/1.006×，说明旧低并发差距主要混入了管理/审计路径开销。新旧对照同时去掉wrapper
检查与trace，不能拆分两者各自贡献。

verifier的机制与清理门通过：六个进程均无`cpu_offload_copy_forward`事件；D/W每进程
分别只有48/11条初始化direct-forward证书。四个B的完整输出hash跨臂一致，96次Prefill和
2,016次Decode均为未捕获eager执行，无服务期H2D、CUDA Graph capture、
Xid/MMU/PTE/PDE fault，结束后恢复stock UVM且GPU为0 MiB。

来源：[E14详细结果与复现入口](e14/RESULTS_E14.md)

## 18. 当前可以与不可以得出的结论

当前可以支持：

1. 全覆盖双页表在最新E2/E3/E10中功能正确，覆盖率100%，没有range-map回退或页表故障。
2. 最新E2近等量轮的16/16 cell通过；Ours cold2/6/20与Native实际卸载量相差不超过
   10%，Ours TPOT在8/8点低1.33×--3.42×。其中6点完整输出hash一致，对应优势为
   1.33×--2.02×。
3. E5当前模块的同placement、无逐forward trace消融表明CUDA Graph使Prefill和Decode
   分别快1.24×--1.76×和2.13×--2.42×；stable VA使重建路径快1.35×--1.49×。
4. E5的P/D直接切换比DESTROY/rebuild快2,015×--2,260×；单独关闭页表预构建后，
   RESUME慢26.87×--29.76×，而稳态Prefill/Decode基本不变。
5. E3较早同坐标Decode-Graph矩阵中hybrid相对always-copy为7/7更快，但always-remote在
   6/7点胜出；native whole-Parameter矩阵则实测到RAG/B8由copy胜出的phase crossover。
6. 原生Native UVA可启用CUDA Graph：E2的8/8 Decode点和E3的10/10 Prefill/Decode点
   均实际Full Graph replay；E3与D/ours完整输出hash为10/10一致。
7. E3近等量Native copy对照的九个合法点均使用Prefill/Decode Full Graph；Ours的
   Decode和E2E分别快2.72×--10.41×和2.65×--7.53×，而Native Prefill低
   3.31%--13.69%。
8. 固定remote计划在E4的C1--C32范围内没有出现持续TPOT退化或PCIe饱和。
9. E13同一RAG-8192→16 eager B1--B8压力矩阵中，最佳Fixed依次为B1--2的K2、B3--5的
   K4、B6--7的K6；Adaptive在这些点与目标Fixed相差不超过0.33%。B8时最佳Fixed回到K4，
   Adaptive-K6比其低4.20%。原双Graph回环中Adaptive的24/24项核心稳态指标仍均在对应
   Fixed的±10%内。
10. E14同请求集选冷的eager UVA formal中，B32逐层cold-29相对固定整层L0--L10的
    Prefill和Decode吞吐高132.4%/63.7%；B1相差-4.2%/+0.6%；B32线性扩展效率为
    99.2%/100.3%对40.9%/61.7%。

当前不能支持：

1. 不能声称ours在所有模型、请求和系统上全局最优；当前E2只有单轮且placement不完全相同。
2. 不能把E2的RAG/C1和RAG/C16写入跨系统exact-token性能排名；两点虽完成请求，但ours与
   原生vLLM输出hash不同。
3. 不能从E2组合对比中单独归因CUDA Graph、GPU缓存或双页表的贡献。
4. 不能发布E1跨系统排名或E11正式性能结果，因为正确性门未通过；E9可报告同框架系统
   消融性能，但不能把uniform/no-mixed写成teacher-forced数值质量等价。
5. 不能声称E8已运行，也不能把E12的1,000-replay或1,000-cycle机制测试称为完整鲁棒性soak。
6. 不能声称E7在线自适应已经无开销；当前同步维护使cycle吞吐下降40.35%。
7. 不能把E3 Native UVA双Graph与D的差异写成纯Graph或equal-HBM crossover；两者虽有
   相同请求和完整输出hash，但whole-Parameter与exact-slice placement、卸载量不同。
8. 不能声称Graph-enabled Native copy已完成E3十点矩阵；RAG/B16在Prefill Graph捕获时
   因HBM不足为容量N/A。九点倍率也不是纯copy/remote单变量结论，因为卸载对象不同。
9. 不能把E10人工压力切档写成automatic/full-joint控制器结果。
10. 不能由E13声称档位切换无开销或快速P→D完全等价；四个7秒级双Graph档位事务未计时，
   且5个P→D点比对应Fixed高10.92%--24.03%。三档压力矩阵也排除了切档和warmup，只证明
   当前RAG-8192的B1--B8吞吐crossover；B8已反例说明Adaptive的容量档位不保证全局吞吐
   最优。不能外推到任意请求，或把单冷进程臂写成统计等价证明。
11. 不能由E14声称固定整层吞吐不再增长，也不能把B1的三冷进程点估计写成统计等价证明；
    Decode的逐层优势从B16出现。同请求集选冷和评测属于in-distribution oracle，不能
    由此声称该专家名单能泛化到未参与选冷的请求或workload。
12. 不能由E5 Graph/eager的B4固定step延迟声称exact-generation或数值质量等价；两臂在
    各自执行类内输出稳定，但B4跨执行类hash不同。

## 19. 剩余工作与执行优先级

当前最关键的未完成项按依赖关系排序如下：

1. 正确性门：隔离E2的RAG Prefill Graph输出分叉，并补齐E1/E11以及E9 numerical-path
   两臂的teacher-forced logits、top-1 agreement和perplexity；在对应门通过前不发布相关
   跨系统、scheduler或质量等价结论。
2. E2主矩阵：按完整measured peak HBM配平DeepSeek三系统与Qwen ours/KT/vLLM，每个主点
   5个独立process blocks并报告95% CI。
3. E3 crossover：在同模型、同expert coordinates、同实际卸载集合和同Graph能力下完成
   hybrid/always-copy/always-remote配对，消除当前Native whole-Parameter与D exact-slice
   placement差异。
4. E5/E6统计：把E5四机制从每臂3个冷进程扩展到至少5个并增加B8/B16；把E6三臂从
   单样本扩展为重复统计。
5. E7/E10动态实验：异步化heat maintenance或明确同步开销；运行5-seed重复峰谷、backup
   sizing、预测误差、open-loop trace和full-joint goodput/SLO对照。
6. E8：取得与当前MoE-Infinity扩展、commit和store一致的fresh build receipt后再运行，
   不绕过provenance门。
7. E11范围：先隔离pure-Prefill Full Graph与eager输出差异，再决定继续phase epoch还是
   冻结phase-batched claim。
8. E12：完成至少6小时整模型soak、重复KV峰谷、barrier故障注入、长期HBM/性能退化检查
   和完整custom→stock生命周期验证。
9. E13：用多个独立冷进程和动态长trace复验B1--B8 crossover，将压力矩阵迁移到双Graph
   协议并纳入档位事务成本；为超过K6容量的点加入“分批数+远端权重成本”联合选档，继续
   隔离5个P→D超带点。
10. E14：用与选冷集不相交的请求、更多请求长度、更高并发和独立seed复验，检查
    request-specific cold-29的B32 crossover能否泛化。

## 20. 权威复现入口

所有实验的最新可用复现状态和入口统一见：

[experiments/reproduce/README.md](../reproduce/README.md)

其中E2/E3/E10联合协议见：

[REPRODUCE_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md](../reproduce/REPRODUCE_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md)

原生vLLM Native UVA CUDA Graph补充协议见：

[REPRODUCE_E2_E3_NATIVE_UVA_CUDAGRAPH.md](../reproduce/REPRODUCE_E2_E3_NATIVE_UVA_CUDAGRAPH.md)

E2最新近等量卸载敏感性轮见：

[REPRODUCE_E2_NEAR_EQUAL_OFFLOAD_PLUS_ONE_EXPERT.md](../reproduce/REPRODUCE_E2_NEAR_EQUAL_OFFLOAD_PLUS_ONE_EXPERT.md)

E3近等量Graph-enabled Native copy对照见：

[REPRODUCE_E3_NEAR_EQUAL_NATIVE_COPY_GRAPH.md](../reproduce/REPRODUCE_E3_NEAR_EQUAL_NATIVE_COPY_GRAPH.md)

E3 KTransformers调优后单轮和离线验证见：

[REPRODUCE_E3_KTRANSFORMERS_TUNED_SINGLE_ROUND_20260903.md](../reproduce/REPRODUCE_E3_KTRANSFORMERS_TUNED_SINGLE_ROUND_20260903.md)

E13完整协议总入口和本轮双Graph回环专用入口见：

- [REPRODUCE_E13_KV_ADAPTIVE.md](../reproduce/REPRODUCE_E13_KV_ADAPTIVE.md)
- [REPRODUCE_E13_FULL_SEPARATE_K2_K6_ADAPTIVE_ROUNDTRIP.md](../reproduce/REPRODUCE_E13_FULL_SEPARATE_K2_K6_ADAPTIVE_ROUNDTRIP.md)

E14 eager UVA并发扩展qualification与formal入口见：

[e14/README.md](e14/README.md)

当前模块E5/E12补充qualification见：

- [RESULTS_E5_CORE_FOUR_MECHANISM_ABLATION_20260903.md](RESULTS_E5_CORE_FOUR_MECHANISM_ABLATION_20260903.md)
- [RESULTS_E5_E12_CURRENT_MODULE_QUALIFICATION_20260831.md](RESULTS_E5_E12_CURRENT_MODULE_QUALIFICATION_20260831.md)

本文只统一结果与证据级别；它不替代各实验原始artifact、verifier和冻结协议。
