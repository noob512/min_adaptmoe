# AdaptMoe EuroSys 新增实验完整结果、结论与分析

## 1. 文档定位

本文汇总 2026-09-09 在当前单机平台完成的 AdaptMoe EuroSys 新增实验，覆盖实验 A–H。本文包括实验范围、执行配置、实际命令入口、完成状态、全部可用指标、失败与重试、机制关闭情况、跨实验结论、论文可使用范围，以及 git/GPU 清理证据。

原始汇总生成时间为 2026-09-09T15:13:03Z；本文于 2026-09-10 扩充。扩充仅修改本新增实验目录中的报告和校验清单，没有修改运行时、驱动、模型、既有实验入口或论文。

本轮采用减少重复次数的 qualification 协议，目标是尽快确认机制可运行、正确性边界和一阶趋势：

- 系统实验通常只有 1 个独立进程或 1 个 seed；
- 在线实验只使用 1 条冻结 trace；
- B/C 页表微基准执行 1,000 个 P/D 循环；
- profiler 和代表点通常只运行 1 次；
- 没有第二硬件平台，也没有长时间随机压力测试；
- 除 B/C 的 1,000 次微基准外，当前结果不能替代正式多冷进程、误差条和置信区间。

因此，本文使用“资格完成”表示机制和当前成功判定已通过，不表示已经达到论文最终统计强度。

## 2. 平台、模型与非侵入约束

### 2.1 平台

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| 可见显存 | 32,607 MiB |
| GPU index | 0 |
| NVIDIA driver | 595.71.05 |
| CPU set | 0–13 |
| DeepSeek | DeepSeek-V2-Lite-Chat，BF16 |
| Qwen | Qwen3-30B-A3B-FP8 |

模型使用本机冻结 snapshot。具体绝对路径、环境覆盖和每个 wrapper 的完整 argv 见 [summary.json](summary.json) 的 `executed_wrappers`。

### 2.2 非侵入执行规则

- 既有代码、接口、默认参数和文件结构均视为只读输入；
- 新增内容只位于 `experiments/eurosys_additional_nonintrusive_20260909/`；
- 每个实验由新增 `run_case.py` 包装，在运行前后调用 git guard；
- 既有 dirty worktree 在实验开始时按路径和内容 hash 冻结；
- 新增配置或参数错误允许在新增层修正并重试；
- 原组件错误只记录，不修改原组件；
- 无法确认归属时先隔离或最小化复现。

通用执行形式为：

```bash
python3 experiments/eurosys_additional_nonintrusive_20260909/run_case.py \
  --case <case-name> \
  --output-dir <wrapper-output> \
  --timeout <seconds> \
  -- <existing-entry-and-arguments>
```

30 个 wrapper 的完整命令、环境变量、返回码、前后 guard 状态均保存在 [summary.json](summary.json)。

## 3. 实验矩阵和最终状态

| 实验 | 状态 | 当前证据 | 不能据此声称的内容 |
|---|---|---|---|
| A 在线阶段分离 | 部分完成 | 获得 HOL blocking 趋势 | 因输出不一致，不能发布 speedup/slowdown |
| B 页表快切 | 资格完成 | 8 MiB、1,000 循环 PDE 数据和 cold80 全规模叶级 PTE 点 | 不能代替完整 25%–100%、B=1/8/16 配对矩阵 |
| C 生命周期 | 部分完成 | P/D 切换期间 allocation/free/copy/fault 均为 0 | Plan 扩缩容尚未闭合 runtime/workspace 守恒 |
| D 动态切档 | 资格完成 | Auto、K2/K4/K6、No-Hysteresis 均运行；Auto 完成双向切档 | 不能声称 Auto 优于 Best-Static |
| E 资源开销 | 部分完成 | HBM、RSS、初始化和 shared backing 包络 | 缺 GraphExec、页表和 allocator reserve 独立归因 |
| F 机制消融 | 部分完成 | 阶段专用 D arm 和 stable-VA/full-rebuild 对照成功 | all-remote FULL_SEPARATE 不受原运行时支持 |
| G PCIe | 部分完成 | graph-copy 证书和 remote-decode NVML 代表点 | 不能作为 CUPTI 级逐字节归因 |
| H Stock UVM | 单轮完成 | 64 MiB 微基准；两模型、C1/C16、36 个 vLLM Device/Stock-Demand 执行 | 只有单轮 eager direct-batch，不能替代 tuned-UVM 或 AdaptMoe 配对复测 |

## 4. 实验 A：阶段分离调度的在线代价

### 4.1 实际配置与成功条件

实际执行入口：

```bash
bash experiments/eurosys_moe_offload_evaluation/run_e11_continuous_arrival_qualification.sh
```

有效重试使用 `ALLOW_GPU_RUN=1`、`TIMEOUT_SEC=2400`，结果目录为 `runs/A_e11_payload_retry2/`。本次是单条 8 请求确定性冲突 trace，不是原计划的 ShareGPT Poisson open-loop 20%–110% offered-load 扫描。

三个 arm 为：

1. `remote_phase_epoch`：不混合 P/D；
2. `remote_continuous`：允许 continuous scheduler 形成混合 step；
3. `hybrid_fail_closed`：验证不受支持的 mixed original-VA 路径会显式拒绝。

### 4.2 结果

| 指标 | Phase-epoch | Continuous | 观察 |
|---|---:|---:|---|
| 成功请求 | 8/8 | 8/8 | 两臂均完成 |
| 请求吞吐 | 1.682 req/s | 6.016 req/s | Phase-epoch 为 0.280× |
| token 吞吐 | 26.91 token/s | 96.25 token/s | Phase-epoch 明显更低 |
| TTFT P50 | 449.05 ms | 174.84 ms | +274.22 ms |
| TTFT P99 | 604.05 ms | 201.43 ms | +402.62 ms |
| TPOT P50 | 144.46 ms | 55.10 ms | +89.36 ms |
| TPOT P99 | 249.78 ms | 68.12 ms | +181.66 ms |
| E2E P50 | 2615.95 ms | 1001.39 ms | +1614.56 ms |
| E2E P99 | 4041.00 ms | 1170.26 ms | +2870.74 ms |
| Queueing P50 | 237.93 ms | 0.29 ms | 阶段隔离产生等待 |
| Queueing P99 | 475.61 ms | 0.44 ms | 尾部等待明显 |
| 平均 Queueing | 237.93 ms | 0.30 ms | 平均额外 237.63 ms |
| SLO good requests | 4/8 | 8/8 | Phase-epoch goodput 0.841 req/s |
| scheduler steps | 32 | 20 | Phase-epoch 需要更多 step |
| mixed scheduler steps | 0 | 1 | 调度契约符合预期 |

`hybrid_fail_closed` 返回：

```text
RuntimeError: Original-VA alias-prefill/remote-decode requires an unmixed request
```

这是预期的运行时契约拒绝，不是静默执行错误。

### 4.3 结论与限制

资格数据表明，强制阶段隔离会让已经活跃的 decode 等待长 prefill，产生明显 HOL blocking。当前构造负载中吞吐下降约 72.0%，但这个幅度不能进入论文主结果，因为 `remote_phase_epoch` 与 `remote_continuous` 产生了不同 token IDs，正确性门未通过。

当前 A 只能支持：

> 阶段分离存在真实且可能显著的排队风险，必须在完整 offered-load 曲线中报告适用区间。

当前 A 不能支持：

- AdaptMoe 相对 vLLM-Mixed 的正式净收益；
- 阶段分离固定造成 72% 吞吐下降；
- 接近饱和时的 P99 或 SLO goodput 结论。

原始证据：[verification.json](../runs/A_e11_payload_retry2/verification.json)。

## 5. 实验 B：P/D 快速页表切换

### 5.1 8 MiB parent/PDE 1,000 循环

入口：

```bash
CYCLES=1000 MINIMUM_SECONDS=0 \
bash experiments/eurosys_moe_offload_evaluation/run_e12_dual_view_soak_qualification.sh
```

配置为 4 个 2 MiB block、8 MiB 目标区域，执行 1,000 个双向映射视图切换循环。代码接口历史上把两个方向命名为 `park` 和 `resume`，但这两个词容易被误解为暂停/恢复执行。它们的实际含义是：

- **Prefill→Decode：解除 Prefill 只读 consumer VA 到 GPU backing 的映射。** 只断开 Prefill Graph 使用的只读 consumer alias；upload VA、consumer VA 地址范围和已经分配的 GPU 物理 backing 都保留，Decode 随后使用 remote-access 路径。代码接口名为 `park`。
- **Decode→Prefill：重新建立 Prefill consumer VA 到保留 GPU backing 的映射。** 让 Prefill Graph 能再次通过原来的稳定 VA 访问同一批 backing；不重新申请 VA，也不重新分配物理页。代码接口名为 `resume`。

因此，一个循环包含一次“解除 Prefill consumer 映射”和一次“恢复 Prefill consumer 映射”，测量的是页表视图转换，不是请求暂停时间，也不是专家权重复制时间。

| 操作 | 样本 | Min | Mean | P50 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| Prefill→Decode：解除 consumer 映射 | 1,000 | 0.294 ms | 0.312 ms | 0.306 ms | 0.342 ms | 0.972 ms |
| Decode→Prefill：恢复 consumer→backing 映射 | 1,000 | 0.162 ms | 0.171 ms | 0.166 ms | 0.192 ms | 0.548 ms |

1,000 个循环总耗时 1.439 s，10 个进度 checkpoint 全部完成。解除映射的 P50/P95 为 0.306/0.342 ms，恢复映射为 0.166/0.192 ms；二者尾部仍低于 1 ms。总耗时还包含每轮状态/覆盖率检查、Python 调用和 checkpoint，并不等于两列平均时间简单相加。

### 5.2 cold80 全规模 4 KiB 叶级 PTE 点

全规模 synthetic leaf-PTE 点使用：

- consumer VA 总覆盖：18,119,393,280 bytes，即 18.119 GB/16.875 GiB；这是虚拟地址覆盖量，不是额外占用 16.875 GiB 物理显存；
- upload/shared backing：754,974,720 bytes，即 720 MiB；
- slot 数：1,440；
- consumer records：34,560；每个 512 KiB backing slot 被 24 个 consumer VA range 复用，因此 1,440×24=34,560；
- PTE mode：4K；
- dual-parent switch：关闭；
- 单次运行，无统计重复；
- 模型未加载，属于同几何 synthetic probe。

| 操作 | Outer wall | Driver wall | PTE map | TLB invalidates | Batch |
|---|---:|---:|---:|---:|---:|
| Decode→Prefill：逐 4 KiB PTE 恢复 34,560 个 consumer 映射 | 292.824 ms | 292.806 ms | 223.722 ms | 34,560 | 8 |
| Prefill→Decode：逐 4 KiB PTE 解除 consumer 映射 | 89.885 ms | 89.856 ms | 35.356 ms | 8,640 | 8 |

恢复 consumer→backing 映射还包含 65.186 ms CPU unmap、1.618 ms preflight、1.483 ms build/sort 和 0.462 ms metadata；kernel fault 为 0，低、中、高三个 backing slot 的抽样 consumer 均读到预期值。

该结果表示，在关闭预构建 parent/PDE 切换后，驱动必须处理 34,560 个 consumer alias record 并更新叶级 4 KiB PTE。恢复映射耗时 292.824 ms，其中 PTE map 为 223.722 ms，占外层时间约 76%；解除映射为 89.885 ms。恢复映射约为解除映射的 3.26×，与恢复方向报告的 PTE 更新和 TLB invalidate 数量更多相符。测量前已经完成 720 MiB backing 预分配和全目录构造，所以表中时间不包含 backing 分配，也不是 18.119 GB 数据复制时间。

### 5.3 同几何 parent 路径参考

从 D 的 cold80 adaptive trace 中抽取到 7 个“Prefill→Decode 解除 consumer 映射”和 6 个“Decode→Prefill 恢复 consumer 映射”的 parent/PDE 路径记录：

| 操作 | Count | Min | Mean | Median | Max |
|---|---:|---:|---:|---:|---:|
| Prefill→Decode：解除 consumer 映射 | 7 | 29.41 ms | 31.11 ms | 31.74 ms | 32.44 ms |
| Decode→Prefill：恢复 consumer→backing 映射 | 6 | 17.97 ms | 26.27 ms | 26.63 ms | 33.40 ms |

相对该中位数，叶级 4K PTE 路径：

- Decode→Prefill 恢复映射慢约 11.0×；
- Prefill→Decode 解除映射慢约 2.8×。

这支持“预构建双页表并切换 parent/PDE”能够避免逐叶 PTE 更新的规模相关成本。不过 parent 和 leaf 来自不同 qualification 运行，不能把上述倍率当作正式配对统计结果。

### 5.4 Full-model Qwen 失败点

- parent/full-model：Qwen 2.25 GiB dual-view SEAL、6,912 aliases、1,207,959,552 backing bytes 时出现 `rm_status=86`；session poisoned，后续发生 abort/segfault；
- leaf/full-model：KV 从 160 MiB 调到 128 MiB，并把既有 headroom gate 从 192 MiB 调到 128 MiB 后，post-residency/hot-restore 仍只剩 172.75/40.63 MiB；按规则停止，没有修改原组件；
- synthetic 全规模测量本身成功，但原 shell 在 source common harness 后覆盖了 `VERIFY`；外部调用目标 verifier 后恢复为 `status=ok`。

### 5.5 B 的结论

当前最强结论是：

> 通过预构建 parent/PDE 完成“解除 Prefill consumer 映射”和“恢复 Prefill consumer→backing 映射”能够稳定循环；在 cold80 同几何资格点上，逐个更新 4 KiB PTE 明显更慢，其中 Decode→Prefill 恢复 34,560 个 consumer 映射的差异最大。

尚未完成：25%/50%/75%/100%、P→D/D→P、B=1/8/16 的同进程配对矩阵和正式误差条，因此还没有完整摊销点曲线。

原始证据：[B/C verification](../runs/BC_e12_payload/verification.json)、[full-scale recovered verification](../runs/B_full_scale_pte_payload/verification_recovered.json)。

## 6. 实验 C：物理显存生命周期

### 6.1 固定 Plan 下的 P/D 循环

1,000 个 P/D 循环产生：

| 计数 | 结果 |
|---|---:|
| Parent switch calls | 2,000 |
| PDE writes | 8,000 |
| TLB batches | 2,000 |
| Vidmem allocations during switch | 0 |
| Vidmem free/evictions during switch | 0 |
| Copy pushes during switch | 0 |
| Faults | 0 |

这说明 P/D 快切没有进入物理页分配、释放、数据复制或缺页迁移路径。它改变的是已存在物理 backing 的映射视图，而不是临时搬运专家权重。

### 6.2 Plan Class 扩缩容守恒

按现有配置和计数估算：

| 转换 | ΔResident estimate | ΔShared | ΔKV | 未归因 residual |
|---|---:|---:|---:|---:|
| K2→K4 | −2,264,924,160 B | +94,371,840 B | +2,148,532,224 B | +22,020,096 B |
| K4→K6 | −2,038,431,744 B | +84,934,656 B | +2,146,959,360 B | −193,462,272 B |

Shared backing 会随 Plan 改变，且现有接口不能独立给出 runtime/workspace/allocator 页。因此论文不能写成：

```text
-ΔResident = ΔKV
```

正确表述应至少为：

```text
ΔResident + ΔShared + ΔKV + ΔRuntime/Workspace/Allocator = 0
```

目前 C1 的 P/D 不变量已经通过；C2 的 Plan 级物理页守恒只能部分验证，不能声称精确闭合。

## 7. 实验 D：动态 KV–专家自动切档

### 7.1 实际矩阵

主入口：

```bash
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh qualification
```

使用固定 seed `20260901` 和相同 frozen trace，执行：

- `adaptive`：完整控制器，初始 K4，允许 K2/K4/K6 双向切换；
- `fixed_k2`：1364 KV blocks，cold70，运行期 autoscale 关闭；
- `fixed_k4`：2730 KV blocks，cold80，运行期 autoscale 关闭；
- `fixed_k6`：4095 KV blocks，cold89，运行期 autoscale 关闭；
- `auto_no_hysteresis`：通过新增外部 adapter 将 low-pressure streak 设为 1、minimum residency 设为 0，不修改原控制器。

所有 arm 都完成 32/32 请求，failed/rejected/preempted/OOM 均为 0，queue invariant failure 为 0。

### 7.2 完整 trace 指标

| Arm | Req/s | Tok/s | TTFT P50/P99 | TPOT Mean/P99 | E2E P50/P99 | Queue P50/P99 | Resize |
|---|---:|---:|---:|---:|---:|---:|---:|
| Auto | 0.4168 | 20.633 | 3396.7/18593.8 ms | 111.74/349.92 ms | 10624.2/23518.3 ms | 2592.4/17858.4 ms | 3 grow + 4 shrink |
| Fixed-K2 | 0.4824 | 23.879 | 3519.4/18278.0 ms | 64.21/108.46 ms | 9580.6/19262.0 ms | 2790.3/17550.4 ms | 0 |
| Fixed-K4 | 0.4804 | 23.779 | 2706.5/16372.3 ms | 109.26/276.85 ms | 9405.7/21015.1 ms | 1942.2/15077.6 ms | 0 |
| Fixed-K6 | 0.4567 | 22.605 | 2849.5/17963.7 ms | 130.29/340.48 ms | 9377.9/23367.3 ms | 2045.8/16441.0 ms | 0 |
| Auto-No-Hysteresis | 0.3884 | 19.225 | 3400.4/23195.3 ms | 97.23/348.60 ms | 12227.4/23984.2 ms | 2595.5/22461.4 ms | 3 grow + 4 shrink |

Auto 的 7 次实际切换全部成功：

| # | 方向 | KV blocks | 场景/epoch | Wall |
|---:|---|---|---|---:|
| 1 | Grow | 2730→4095 | high-pressure/high-rag-b6 | 542.33 ms |
| 2 | Shrink | 4095→2730 | dynamic-mixed/mixed-overlap | 90.47 ms |
| 3 | Shrink | 2730→1364 | dynamic-mixed/mixed-overlap | 92.25 ms |
| 4 | Grow | 1364→2730 | dynamic-mixed/mixed-overlap | 434.55 ms |
| 5 | Shrink | 2730→1364 | recovery-low-01 | 92.07 ms |
| 6 | Grow | 1364→2730 | burst-overlap | 443.15 ms |
| 7 | Shrink | 2730→1364 | burst-overlap | 91.75 ms |

累计 resize wall 为 1,786.58 ms；单次 response P50/P95/P99 为 152.20/512.58/536.38 ms。

### 7.3 Auto 与输出一致的 Fixed-K6

Auto 与 Fixed-K6 的 32/32 输出 hash 完全一致，因此这是当前唯一满足完整输出配对的 Auto/Fixed 性能比较：

- Auto 请求吞吐低 8.72%；
- Auto token 吞吐低 8.72%；
- Auto 平均 TPOT 改善 14.24%，但 P99 TPOT 高 2.78%；
- Auto 平均 queueing 高 512.27 ms，P99 queueing 高 1,417.41 ms；
- Auto 平均 TTFT 高 468.32 ms，P99 TTFT 高 630.09 ms；
- Auto 平均 E2E 高 343.50 ms，P99 E2E 高 150.95 ms。

按 `32/throughput` 估算，Auto 服务窗口约 76.77 s，Fixed-K6 约 70.07 s，总差约 6.70 s。1.787 s 显式 resize wall 最多只能直接解释约 27% 的总时间差；仅去掉这部分，Auto 理论吞吐约 0.427 req/s，仍低于 Fixed-K6 的 0.457 req/s。

### 7.4 缩容后的首个 decode step 尖峰

为解释剩余差距，将每个 `resize_events.elapsed_s` 与 Auto `time_series` 中其后的第一条 engine step 对齐，并与 Fixed-K6 中相同 epoch、相同 engine-step index 比较。四次 shrink 均出现首个 cached/decode step 尖峰：

| Shrink | 显式事务 | Auto 切换后首步 | Auto 后续稳态 | Fixed 对应步 | 首步额外时间 |
|---|---:|---:|---:|---:|---:|
| 4095→2730 | 90.47 ms | 1527.23 ms | 约46 ms | 58.6 ms | 约1468.6 ms |
| 2730→1364 | 92.25 ms | 1565.16 ms | 约46 ms | 57.6 ms | 约1507.6 ms |
| 2730→1364 | 92.07 ms | 1417.28 ms | 约44 ms | 43.5 ms | 约1373.8 ms |
| 2730→1364 | 91.75 ms | 1447.60 ms | 约45 ms | 44.1 ms | 约1403.5 ms |

现象为：

- 四次 shrink、四次首步尖峰，相关性为 4/4；
- shrink 本身仅约 90–92 ms，但首个 post-shrink decode step 达 1.42–1.57 s；
- 第二步起立即恢复到约 44–47 ms；
- Fixed-K6 对应位置没有尖峰；
- 四个首步相对 Fixed 的额外时间合计约 5.75 s；
- 两个可直接比较的 grow 后 cached/decode step 为约 46–47 ms，没有同类尖峰；第一次 grow 后进入新 prefill cohort，不能作为 decode 首步比较。

显式 resize 1.79 s 与 post-shrink 首步额外约 5.75 s 和观察到的 6.70 s 总差处于同一量级。它们不能机械相加，因为 Auto 后续稳态 decode 更快，且计时边界可能相关；但当前 trace 强烈支持：

> 完整 trace 的吞吐损失主要来自显式切档暂停和缩容后的首次执行冷启动，而不是目标 Plan 的长期稳态数据路径变慢。

可能涉及首次映射访问、TLB/GMMU cache、remote-read 状态或执行路径重新热身。当前计数器不能进一步唯一归因，因此上述机制原因仍是解释性假设，不是已经定因的硬件事实。

### 7.5 与既有稳态等价实验的关系

既有 E13 FULL_SEPARATE K2/K6/Adaptive roundtrip qualification（不是本轮新增运行）已经比较：Auto-K2 对 Fixed-K2、Auto-K6 对 Fixed-K6、以及 Auto 回缩 K2 对 Fixed-K2。排除 Plan 事务和目标档 warmup 后：

- Adaptive/Fixed 请求吞吐倍率为 0.965–0.988×；
- Prefill 倍率为 1.001–1.010×；
- steady Decode 倍率为 0.982–1.020×；
- E2E 倍率为 1.012–1.037×；
- 核心 24/24 比较全部位于预注册 ±10% 区间；
- 输出 hash 一致，975/975 serving steps 为 FULL Graph replay；
- 回缩后没有持续稳态回退。

但全部 9 类指标只有 49/54 位于 ±10%，5 个超带点全部是每批自身的 P→D 首步，高 10.92%–24.03%。该实验明确排除了 K2↔K6 事务，因此只能证明“切档并 warmup 后恢复对应 Fixed 的稳态水平”，不能证明包含切换的端到端性能等价。

既有证据见 [E13 roundtrip 复现文档](../../reproduce/REPRODUCE_E13_FULL_SEPARATE_K2_K6_ADAPTIVE_ROUNDTRIP.md)。

### 7.6 Best-Static 和 No-Hysteresis 限制

Raw 吞吐最高的是 Fixed-K2，其次是 Fixed-K4，但输出正确性矩阵为：

| 比较 | 不同请求 hash 数 |
|---|---:|
| Auto vs Fixed-K2 | 6/32 |
| Auto vs Fixed-K4 | 1/32 |
| Auto vs Fixed-K6 | 0/32 |
| Auto vs No-Hysteresis | 6/32 |

因此：

- 不能把 Fixed-K2 的 raw 0.4824 req/s 发布为严格可比的 Best-Static；
- No-Hysteresis 确认有效使用 streak=1、minimum-residency=0，但仍只发生 7 次切换，与 Auto 相同；
- No-Hysteresis 吞吐为 0.3884 req/s、累计 resize 为 1.987 s，但因输出不同，不能形成正式性能消融；
- 当前 trace 没有证明 hysteresis 显著减少切换次数。

### 7.7 D 的最终结论

可支持：

1. 控制器能够在 K2/K4/K6 间双向切换并最终缩回 K2；
2. 32/32 请求完成，无 OOM、抢占、拒绝或队列守恒错误；
3. 目标 Plan 稳态路径与对应 Fixed 基本一致；
4. 本 trace 的端到端损失与显式切换及 shrink 后首步尖峰高度相关。

不可支持：

1. Auto 在当前 trace 上优于可比较的 Fixed-K6；
2. Auto 优于 Best-Static；
3. hysteresis 已经被该 trace 证明有效降低振荡；
4. shrink 后首步尖峰已经完成硬件根因归因。

原始结果：[Auto](../runs/D_e13_payload/repeat-00-seed-20260901/adaptive/result.json)、[Fixed-K6](../runs/D_e13_payload/repeat-00-seed-20260901/fixed_k6/result.json)、[Fixed-K2/K4](../runs/D_e13_fixed_k2_k4_payload/repeat-00-seed-20260901/)、[No-Hysteresis](../runs/D_e13_no_hysteresis_payload_retry1/repeat-00-seed-20260901/adaptive/result.json)。

## 8. 实验 E：显存、Host 内存和初始化开销

### 8.1 三个 Plan Class 的内存构成

完整参数估算为 28,991,029,248 bytes，即 27.0 GiB。

| Tier | Plan | KV blocks | KV physical | Expert offload | Resident estimate | Prefill shared | Shared/GPU |
|---|---|---:|---:|---:|---:|---:|---:|
| K2 | cold70 | 1364 | 2,145,386,496 B | 15,854,469,120 B | 13,136,560,128 B | 630 MiB | 1.93% |
| K4 | cold80 | 2730 | 4,293,918,720 B | 18,119,393,280 B | 10,871,635,968 B | 720 MiB | 2.21% |
| K6 | cold89 | 4095 | 6,440,878,080 B | 20,157,825,024 B | 8,833,204,224 B | 801 MiB | 2.46% |

固定资源开销：Prefill shared pool 在 Decode 阶段不参与当前计算，但其 801 MiB GPU backing 仍被保留，占 32,607 MiB GPU 显存的 2.46%。这是一项可见但总体有限的固定资源成本。

### 8.2 进程资源包络

| Arm | Peak HBM | Max host RSS | `/usr/bin/time` elapsed | Load | Tier setup |
|---|---:|---:|---:|---:|---:|
| Auto | 31,134.25 MiB | 64.75 GiB | 130.95 s | 39.82 s | 335.31 ms |
| Fixed-K2 | 31,134.25 MiB | 65.14 GiB | 116.11 s | 34.07 s | 2.99 ms |
| Fixed-K4 | 31,134.25 MiB | 65.14 GiB | 208.75 s | 104.45 s | 349.83 ms |
| Fixed-K6 | 31,134.25 MiB | 65.14 GiB | 157.37 s | 33.00 s | 659.99 ms |
| No-Hysteresis | 31,134.25 MiB | 65.14 GiB | 201.05 s | 104.83 s | 485.06 ms |

这些是各自单次冷进程的资源包络，不应把 load 或 elapsed 差异解释为机制因果效应。代表性 Qwen D arm 的 load 为 16.09 s，Graph capture count 从加载后 2 到最终 2，没有运行期重捕获；该 arm 的 prefill shared backing 为 90 MiB。

### 8.3 E 的结论和缺口

当前可以报告“Decode 阶段保留但暂不参与计算的 Prefill shared backing”最大占比为 2.46%，以及进程级 HBM/RSS/初始化包络。当前接口没有提供：

- CUDA GraphExec 独立字节数；
- GPU/Host 页表独立字节数；
- allocator reserved-but-unused 精确归因；
- Graph 内存和 capture 时间随 Plan/shape 数量增长的完整曲线。

因此 E 为部分完成，不能把 2.46% 误写成 AdaptMoe 的全部元数据开销。

## 9. 实验 F：核心机制消融

### 9.1 Qwen 阶段专用完整 D arm

配置：Ordinary、B=1、TRITON_ATTN、per-batch Graph capture、`RUN_KT=0`。成功 arm 为 `alias_graph_prefill_remote_graph_decode`。

| 指标 | 结果 |
|---|---:|
| Prefill step | 72.03 ms |
| P→D cutover wall | 55.77 ms |
| First decode step（含 cutover） | 74.71 ms |
| First decode excluding cutover | 18.94 ms |
| Warmup decode mean | 8.30 ms |
| Steady decode mean/median | 8.42/8.35 ms |
| Steady decode P90/max | 8.73/9.51 ms |
| Steady decode throughput | 119.75 token/s |
| Cycle E2E | 1220.32 ms |
| Graph capture | 2→2 |
| Parameter VA | 全程稳定 |

121 个 steady decode 样本完成，输出 SHA256 为 `59b45c91e03db942b026da2dc39598b4973a0924f743690b10d497460ca09037`。

### 9.2 all-remote F arm

`remote_only + FULL_SEPARATE` 被原运行时契约拒绝：

```text
FULL_SEPARATE is reserved for prefill_graph_remote_decode
```

清理阶段还报告 lifecycle 仍为 `collecting`，无法释放 original-VA references。按约束没有修改原运行时，因此 UVA-for-Both 的该精确组合跳过。

### 9.3 Stable VA 与 Full-Rebuild

DeepSeek stable-VA 与 full-rebuild 输出 SHA256 完全相同：

```text
a365db717e87663b5f807b34fc6013b44f98cd925bea9480cece8e65eac690b9
```

| 规模 | Stable outbound | Rebuild outbound | 倍率 | Stable return | Rebuild return | 倍率 |
|---:|---:|---:|---:|---:|---:|---:|
| 5% | 521.54 ms | 3431.83 ms | 6.58× | 552.33 ms | 3836.91 ms | 6.95× |
| 10% | 725.72 ms | 3708.13 ms | 5.11× | 722.87 ms | 3674.36 ms | 5.08× |
| 25% | 925.09 ms | 4193.32 ms | 4.53× | 922.20 ms | 4217.81 ms | 4.57× |
| 50% | 1250.73 ms | 5097.33 ms | 4.08× | 1262.63 ms | 4847.94 ms | 3.84× |
| 100% | 1889.41 ms | 6458.97 ms | 3.42× | 1932.97 ms | 6125.61 ms | 3.17× |

Stable arm capture count 保持 2；Full-Rebuild 从 2 增加到 22。100% outbound 的 6458.97 ms 中，Plan rebuild 为 5166.00 ms，Graph capture 为 1121.07 ms；100% return 的对应值为 4831.16/1121.22 ms。

这证明稳定 VA 的收益不只是页表操作更快，还避免了地址空间重新建立和 CUDA Graph recapture。F1 的完整四机制稳态面板仍未完成，但 F2 的 stable-VA 机制证据很强。

原始结果：[Qwen D/F](../runs/F_qwen_DF_payload_retry1/)、[Stable VA](../runs/F2_e6_stable_payload/verification.json)、[Full-Rebuild](../runs/F2_e6_full_rebuild_payload/verification.json)。

## 10. 实验 G：PCIe 流量与阶段路径解释

### 10.1 Prefill graph-copy

代表性 512-token Qwen Prefill：

| 指标 | 结果 |
|---|---:|
| Certified H2D bytes | 2,264,924,160 B |
| 单点 bytes/input-token | 4,423,680 B/token |
| Prefill step | 72.03 ms |

该 bytes/token 是单次固定传输量除以 512-token 得到的单点比例，不能替代长度扫描；只有在补充 128/512/2048/8192 后才能判断随输入长度的摊销趋势。

### 10.2 Remote decode

代表点为 C1、prompt=128、output=16：

| 指标 | 结果 |
|---|---:|
| Prefill ramp | 48.12 ms |
| First full decode | 49.84 ms |
| Steady decode count | 9 |
| Steady decode mean/median | 40.57/40.64 ms |
| Steady decode min/max | 40.18/40.89 ms |
| Decode throughput | 24.61 token/s |
| Request TTFT/TPOT/E2E | 48.24/41.24/666.82 ms |
| Preemptions | 0 |
| Selected remote bytes/forward | 2,264,924,160 B |
| Manifest H2D copy bytes | 0 |

NVML 在 0.365 s decode 窗口只取得 2 个样本：平均 RX 162,539.5 KiB/s，观察到约 60.79 MB RX，即约 6.75 MB/measured-token；平均 GPU utilization 为 20%。

### 10.3 G 的结论和限制

数据与设计解释一致：Prefill graph-copy 明确产生大块 H2D，而 remote decode manifest 没有显式 H2D copy。可是 NVML 是设备级低频聚合计数，无法证明全部 RX 字节都来自 GPU 对 pinned host expert 的直接读取，也不能排除其他 PCIe 流量。因此当前结果只能作为代表性机制证据或下界，不能作为 CUPTI/Nsight 级精确归因。

原始结果：[G result](../runs/G_e4_remote_payload/blocks/b0_asc/result.json)、[NVML samples](../runs/G_e4_remote_payload/blocks/b0_asc/nvml.csv)。

## 11. 实验 H：原生 UVM 对照

### 11.1 64 MiB Stock UVM 微基准

新增 `stock_uvm_probe.cu` 在相同 stock driver 下比较 demand paging 与 prefetch/advice：

| 指标 | Demand | Prefetch/advice |
|---|---:|---:|
| Allocation | 0.062 ms | 0.093 ms |
| Advice | 0 | 0.015 ms |
| Prefetch | 0 | 0.400 ms |
| First GPU write | 450.819 ms | 0.301 ms |
| Read/verify | 0.103 ms | 0.059 ms |
| Mismatches | 0 | 0 |

Demand 首次写包含隐式缺页/迁移成本；显式 advice+prefetch 将迁移前移，prefetch 加首次写约 0.700 ms。该结果说明论文若批评未经管理的 demand paging，有受控证据；同时也必须承认 stock UVM prefetch/advice 能显著改善首次访问。

### 11.2 自定义 no-migration 模式

现有 `managed_no_prefetch` runner 使用 `manual_migration_fault_mode=2` 时，64 MiB managed allocation 在第一次 GPU 写同步返回 CUDA 700 illegal memory access，probe return code 为 10。该错误来自原组件/模式组合，未修改或修复。

### 11.3 vLLM Device-resident 与 Stock UVM Demand 单轮对照

补充一轮与完整结果文档使用相同模型、冻结请求及 C1/C16 并发量的端到端对照。DeepSeek-V2-Lite BF16 使用 Short 128/32、Ordinary 512/128、RAG 8192/128、Long 512/512；Qwen3-30B-A3B FP8 额外包含 Medium 2048/128。每个 cell 使用独立冷进程、3 个单请求预热、eager direct-batch、固定 greedy 输出长度、`cpu_offload_gb=0`，不经过 HTTP。

两个 arm 除分配器外使用相同 runner 和配置：Device-resident 使用普通 CUDA device allocator；Stock UVM Demand 将 PyTorch CUDA allocation（包括权重、KV cache 和 workspace）改为 `cudaMallocManaged`，显式设置 `VLLM_UVM_PREFETCH=0`、`VLLM_UVM_POLICY_ENABLE=0`，并关闭 warmup advice、gap-watch、KV/fixed-weight device-direct 等策略。18 份 UVM allocator 日志均包含上述关闭状态及 managed `TRACE_ALLOC`，没有出现 `cudaMemPrefetchAsync`、`cudaMemAdvise` 或对应主动策略。加载模块和磁盘 stock module 的 srcversion 均为 `85A79790636BBD99BA3E43B`。

以下 TTFT、TPOT 和 E2E 均为请求级 P50，单位为 ms；`OOM` 表示 Device-resident 在模型初始化或 KV/workspace 分配时容量不足，因此该 cell 不计算 slowdown。

#### 11.3.1 DeepSeek-V2-Lite BF16

| 请求 | C | Device TTFT | UVM TTFT | Device TPOT | UVM TPOT | Device E2E | UVM E2E | UVM/Device TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 1 | 63.78 | 240.87 | 64.19 | 96.77 | 2,053.67 | 3,240.88 | 3.78× |
| Short | 16 | 65.37 | 1,692.12 | 64.86 | 132.82 | 2,075.97 | 5,809.61 | 25.88× |
| Ordinary | 1 | 59.31 | 596.74 | 61.93 | 90.98 | 7,923.92 | 12,150.72 | 10.06× |
| Ordinary | 16 | OOM | 4,770.25 | OOM | 124.64 | OOM | 20,599.70 | — |
| RAG | 1 | OOM | 3,127.12 | OOM | 82.08 | OOM | 13,551.57 | — |
| RAG | 16 | OOM | 41,245.00 | OOM | 1,103.84 | OOM | 181,432.14 | — |
| Long | 1 | 59.31 | 558.89 | 55.79 | 81.59 | 28,567.69 | 42,248.95 | 9.42× |
| Long | 16 | OOM | 4,397.54 | OOM | 122.34 | OOM | 66,912.71 | — |

#### 11.3.2 Qwen3-30B-A3B FP8

| 请求 | C | Device TTFT | UVM TTFT | Device TPOT | UVM TPOT | Device E2E | UVM E2E | UVM/Device TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Short | 1 | 68.63 | 243.05 | 70.81 | 125.12 | 2,263.64 | 4,121.72 | 3.54× |
| Short | 16 | 73.08 | 1,812.39 | 74.54 | 164.07 | 2,383.89 | 6,898.40 | 24.80× |
| Ordinary | 1 | 69.80 | 665.07 | 72.31 | 98.85 | 9,253.24 | 13,218.63 | 9.53× |
| Ordinary | 16 | OOM | 5,332.73 | OOM | 158.14 | OOM | 25,416.76 | — |
| Medium | 1 | 70.53 | 1,044.19 | 71.30 | 99.79 | 9,126.13 | 13,717.90 | 14.80× |
| Medium | 16 | OOM | 16,435.05 | OOM | 208.70 | OOM | 42,939.85 | — |
| RAG | 1 | 417.55 | 6,209.60 | 71.60 | 115.55 | 9,511.16 | 20,884.31 | 14.87× |
| RAG | 16 | OOM | 44,831.71 | OOM | 1,889.95 | OOM | 284,855.17 | — |
| Long | 1 | 69.25 | 649.89 | 65.33 | 104.18 | 33,452.31 | 53,883.90 | 9.38× |
| Long | 16 | OOM | 5,494.60 | OOM | 157.63 | OOM | 86,041.56 | — |

36 个执行中，Stock UVM Demand 完成 18/18；Device-resident 完成 10/18，另外 8 个为真实 OOM。在 10 个可直接比较的 cell 中，Stock UVM 的 TTFT 是 Device-resident 的 3.54×–25.88×，TPOT 是 1.37×–2.20×，E2E 是 1.43×–2.89×；10/10 个 cell 的逐请求输出 token 哈希完全一致。并发效应在两模型 Short/C16 上最清晰：UVM TTFT 分别放大 25.88× 和 24.80×，TPOT 放大 2.05× 和 2.20×。容量压力最大的 RAG/C16 中，Device-resident 均 OOM；UVM 虽能完成，但 DeepSeek/Qwen 的 E2E 已分别达到 181.43 s/284.86 s。DeepSeek RAG/C16 因分块 prefill，请求级 TTFT P50 为 41.25 s，而全批产生首 token 的 Prefill E2E 为 70.73 s。

### 11.4 H 的结论和限制

新增结果把原先的 64 MiB 机制观察扩展到了真实 vLLM 模型执行：在显存能够容纳、语义和输出一致的 cell 中，未经 advice/prefetch 的 Stock UVM demand paging 同时显著恶化 TTFT、TPOT 和 E2E；在不能完整驻留的高压力 cell 中，它可以依靠主存完成请求，但服务延迟很高。该结果支持限定性结论：**本机 32 GiB RTX 5090、这两种模型和冻结请求下，Stock UVM 的无提示自动 demand migration 不适合作为低延迟 vLLM serving 路径。**

这仍然只是单轮 eager direct-batch，不含 HTTP 到达过程、fault/migration 的 CUPTI 计数，也没有与 Stock UVM advice/prefetch 或 AdaptMoe 做同协议配对。因此不能据此声称所有 UVM 使用方式都性能差，也不能单独据此声称 AdaptMoe 已端到端优于 tuned Stock UVM。Device OOM 的 8 个 cell 只提供容量边界，不是性能 speedup/slowdown 数据。

原始证据：[vLLM protocol](../runs/H_vllm_stock_uvm_20260911_single_round/protocol.json)、[execution ledger](../runs/H_vllm_stock_uvm_20260911_single_round/execution.tsv)、[machine-readable summary](../runs/H_vllm_stock_uvm_20260911_single_round/summary.json)、[summary TSV](../runs/H_vllm_stock_uvm_20260911_single_round/summary.tsv)。原 64 MiB 证据：[Demand stdout](../runs/H_stock_demand_wrapper/stdout.log)、[Prefetch stdout](../runs/H_stock_prefetch_wrapper/stdout.log)、[custom no-migration result](../runs/H_managed_no_prefetch_payload/result.json)。

## 12. 跨实验综合结论

### 12.1 当前证据最强的贡献

1. **PDE 映射视图切换有效。** 8 MiB parent path 在 1,000 次循环中，解除 Prefill consumer 映射和恢复 consumer→backing 映射的 P50 均低于 0.31 ms；cold80 同几何下逐个修改 4K PTE 明显更慢。
2. **P/D 快切不进入物理分配路径。** 2,000 次 switch 中 allocation、eviction/free、copy、fault 全为 0。
3. **稳定 VA 避免重建和重捕获。** Full-Rebuild 比 stable VA 慢 3.17×–6.95×，Graph capture 2→22，而 stable 保持 2；输出一致。
4. **切档后稳态没有持续回退。** 既有回环资格实验中，Auto 对对应 Fixed 的核心 24/24 指标位于 ±10%。
5. **动态控制功能正确。** 本轮 Auto 完成 3 grow+4 shrink，无 OOM、抢占、拒绝或队列错误。
6. **Decode 阶段保留但暂不使用的 Prefill shared backing 成本可见但有限。** 最大为 801 MiB，占 GPU 可见显存 2.46%。

### 12.2 当前暴露的主要代价

1. 阶段分离会造成 HOL blocking；当前资格 trace 的额外平均排队为 237.63 ms。
2. Auto 的显式切换累计 1.787 s。
3. 四次 shrink 后首个 decode step 均从约 44–58 ms 升到 1.42–1.57 s，合计额外约 5.75 s，是本轮 Auto 吞吐损失的主要相关因素。
4. P→D 首步并非总能与 Fixed 保持 ±10%，既有 roundtrip 中 5 个点高 10.92%–24.03%。
5. Prefill shared pool 在 Decode 阶段虽然不参与当前计算，但物理 backing 仍保留，最坏占用 2.46% HBM。

### 12.3 当前不能发布的主张

- AdaptMoe 在所有 offered load 下优于 mixed scheduler；
- 本次阶段分离损失的 72% 是一般性结果；
- Auto 优于 Best-Static；
- No-Hysteresis 一定产生更多切换；
- `-ΔResident=ΔKV` 在所有 Plan 转换中成立；
- PCIe 字节已经得到精确硬件归因；
- AdaptMoe 端到端优于经过 advice/prefetch 优化的 Stock UVM；
- GraphExec、页表和 allocator 总开销已经完整测量。

### 12.4 建议的论文表述

可以写：

> AdaptMoe 使用预构建稳定 VA 映射在 P/D 阶段间切换；1,000 个资格循环中未观察到物理分配、释放、复制或 fault。全规模资格点显示逐 4K PTE 更新明显慢于 parent/PDE 路径，稳定 VA 还避免了地址重建和 Graph recapture。动态控制能够安全双向调整 KV–专家档位，且目标档 warmup 后恢复对应 Fixed 的稳态性能。

必须同时写：

> 阶段隔离可能引入 HOL blocking；完整动态 trace 还包含显式 resize 和 post-shrink 首步冷启动。本轮 Auto 吞吐低于输出一致的 Fixed-K6，因此当前证据支持动态容量安全和稳态恢复，不支持 Auto 总体性能优于最佳静态配置。

## 13. 失败、重试和错误归属

### 13.1 原组件错误或不支持组合

| 实验 | 触发条件 | 错误 | 处理 |
|---|---|---|---|
| A | 同拓扑 scheduler arms | token IDs 不一致 | 保留趋势，禁止正式性能结论 |
| B parent full-model | Qwen dual-view SEAL | `rm_status=86`，session poisoned，后续 abort/segfault | 不修改驱动，跳过完整 arm |
| B leaf full-model | Qwen hot restore | HBM 只余 172.75/40.63 MiB | 参数重试后仍不足，停止 |
| B synthetic harness | source common harness | `VERIFY` 变量被覆盖 | 不改脚本，外部调用目标 verifier 并通过 |
| F all-remote | `remote_only + FULL_SEPARATE` | 原运行时契约拒绝 | 记录并跳过该组合 |
| H custom | no-migration 首次 GPU 写 | CUDA 700 | 记录原组件错误，保留 stock baseline |

### 13.2 新增层调整与重试

- `run_case.py` 改用 `sys.executable` 调用新增 guard，解决 guard 文件没有 executable bit；
- A 首次 preflight 缺 `ALLOW_GPU_RUN=1`，补充新增 wrapper 参数后重试；
- B leaf 将 KV 160→128 MiB，并将既有 headroom gate 192→128 MiB；仍不足后停止；
- 模块切换 F 中移除外部 pynvml wrapper，避免其持有 UVM client；随后 D arm 成功；
- 两个误走默认路径的输出目录完整移动到允许的新目录，不改内容；
- No-Hysteresis 自定义 arm 名被原 shell 硬编码表拒绝，改由新增 case-runner adapter 以合法 `adaptive` arm 注入外部配置后成功；
- B full-scale 原始测量成功但 shell 调错 verifier，外部运行目标 verifier 后通过。

### 13.3 未完成或不具正式性的部分

- A：Poisson/open-loop 20%–110% offered-load 扫描和 2×3 图；
- B：full-model 25%/50%/75%/100%、两个方向、B=1/8/16；
- D：Oracle arm；
- D：可发布 Best-Static，因为 K2/K4 与 Auto 输出 hash 不同；
- E：GraphExec、页表、allocator reserve 独立归因和 Plan/shape 扩展曲线；
- F：Copy-for-Both 精确专家粒度，以及 all-remote FULL_SEPARATE；
- G：完整长度/batch/offload grid 和 CUPTI/Nsight 归因；
- H：同模型语义的端到端 Stock-UVM baseline。

## 14. 被关闭的机制及影响

| 场景 | 关闭方式 | 影响 |
|---|---|---|
| Custom dual-view probes | 使用现有 module switch 关闭 access-counter migration、UVM prefetch、thrashing | 隔离显式映射路径；不能代表默认 UVM 行为 |
| B leaf-PTE | 使用现有 runner 关闭 dual-parent switch | 测得逐叶 PTE 路径；不是完整 AdaptMoe 快切 |
| Fixed K2/K4/K6 | 使用现有 arm 禁用 autoscale | 提供静态档位对照 |
| Auto-No-Hysteresis | 外部配置 streak=1、minimum residency=0 | 验证控制参数生效；输出差异限制性能结论 |
| F representative | `RUN_KT=0` | 未重复 KTransformers 外部 baseline |
| 全部新增实验 | 不使用第二平台和长随机 soak | 结果只具单平台资格性质 |

## 15. 复现命令和结果索引

以下为关键入口；每个实际 argv、环境覆盖、timeout、return code 和 git guard 均见 [summary.json](summary.json) 的 `executed_wrappers`。

| 实验 | 关键入口 | 主要结果目录 |
|---|---|---|
| A | `run_e11_continuous_arrival_qualification.sh` | `runs/A_e11_payload_retry2/` |
| B/C parent | `run_e12_dual_view_soak_qualification.sh` | `runs/BC_e12_payload/` |
| B leaf | `run_full_scale_page_table_probe.sh` + external verifier | `runs/B_full_scale_pte_payload/` |
| D Auto/K6 | `run_e13_kv_adaptive_matrix.sh qualification` | `runs/D_e13_payload/` |
| D K2/K4 | 同上，`E13_ARMS="fixed_k2 fixed_k4"` | `runs/D_e13_fixed_k2_k4_payload/` |
| D No-Hysteresis | 同上 + 新增 `E13_CASE_RUNNER` | `runs/D_e13_no_hysteresis_payload_retry1/` |
| F Qwen D/F | `run_formal_e3.sh` | `runs/F_qwen_DF_payload_retry1/` |
| F stable VA | `run_formal_e6_difference.sh` | `runs/F2_e6_stable_payload/` |
| F rebuild | `run_e6_full_rebuild_qualification.sh` | `runs/F2_e6_full_rebuild_payload/` |
| G | `run_formal_e4.sh` | `runs/G_e4_remote_payload/` |
| H | `stock_uvm_probe.cu`；`run_h_vllm_stock_uvm.sh`；`summarize_h_vllm_stock_uvm.py` | `runs/H_*`；`runs/H_vllm_stock_uvm_20260911_single_round/` |

机器可读输出：

- [summary.json](summary.json)：全部结构化结果、限制、命令和环境；
- [d_metrics.csv](d_metrics.csv)：D 五个 arm 的主要指标；
- [manifest.json](manifest.json)：summary、report 和 CSV 的 SHA256；
- `../runs/*_wrapper/{command.json,status.json,stdout.log,stderr.log}`：逐 case 命令、状态与日志；
- `../runs/*_payload/`：既有 runner 产生的原始结果、模块快照、kernel log 和 verification。

## 16. Git、GPU 和驱动清理审计

### 16.1 2026-09-09 实验执行审计

实验开始时冻结状态为：

| 项目 | 值 |
|---|---|
| HEAD | `7b6f3a5722305510a9d314d258c3846bd122220f` |
| 原有 dirty paths | 78 |
| Cached diff SHA256 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Worktree diff SHA256 | `53d5aa22c7ebc7757a41b43a16c7849309078d3b0d53f29cc77ff11af0b63d58` |
| 受保护 wrapper | 30 |
| Wrapper 前后 guard | 全部通过 |

实验结束时：

- RTX 5090 memory.used 为 0 MiB；
- GPU utilization 为 0%；
- compute client 列表为空；
- loaded UVM srcversion 与系统 stock UVM 均为 `85A79790636BBD99BA3E43B`；
- 原有 78 个 dirty path 的内容 hash 未变化。

证据：[最终实验 guard](../final_audit/git_guard_final.json)、[GPU state](../final_audit/gpu_state.csv)、[compute clients](../final_audit/compute_clients.csv)、[loaded UVM](../final_audit/loaded_uvm_srcversion.txt)、[stock UVM](../final_audit/stock_uvm_srcversion.txt)。

### 16.2 2026-09-10 报告扩充审计

在本次报告扩充开始前，旧 2026-09-09 baseline 检测到工作区外部后来出现：

- `experiments/eurosys_moe_offload_evaluation/e11_continuous_arrival_runner.py` 新变化；
- 新目录 `experiments/adaptmoe_mixed_schedule_probe/`。

这些变化发生在昨日最终实验审计之后，不属于本报告扩充操作；没有修改、回退，也没有把它们当作本轮 A 的新数据。旧 baseline 的差异记录保存在 `../final_audit/git_guard_before_report_expansion.json`。

为避免覆盖并继续安全编辑，报告扩充以当时工作区建立独立只读基线 `../report_edit_baseline_20260910.json`。扩充前 guard 为 `ok=true`，扩充后 guard 见 `../final_audit/git_guard_report_edit_after.json`；它验证除允许的新实验目录外没有进一步变化。

## 17. 正式补跑优先级

1. 先解决或解释 A、D 跨调度/跨 Plan 的输出 hash 差异；否则不能形成正式 Mixed/Separate 和 Best-Static 结论。
2. 对 D 做受控 `0/1/3/7` 次切换实验，固定每档驻留时间；将 resize、post-switch step 0–16、TTFT/TPOT 尖峰放在共享时间轴上。
3. 专门隔离 shrink 后首步：分别测 page-table/TLB、Graph state、remote-read/GMMU cache 和首次 kernel submission；首步不能从端到端统计中静默删除。
4. 补 A 的 Poisson open-loop 20%–110% offered-load 曲线、真实 ShareGPT trace 和 P/D 冲突 trace。
5. 在同一冷进程中配对 full-scale parent/PTE，补 25%/50%/75%/100%、两个方向和 B=1/8/16。
6. 增加 CUPTI/Nsight 字节归因，并单独测 GraphExec、页表和 allocator reserve 内存。
7. 对现有 full-allocation Stock UVM 单轮结果增加独立重复、fault/migration 计数及 advice/prefetch arm；在完成与 AdaptMoe 的同协议配对前不作端到端优胜声明。

## 18. 最终结论

本轮新增实验最有力地验证了 AdaptMoe 的三个底层机制：PDE 级快切、P/D 映射切换不触发物理显存生命周期、稳定 VA 避免地址重建和 Graph recapture。动态控制器可以无 OOM/抢占地完成双向 KV–专家切档，并在 warmup 后恢复到对应 Fixed 的稳态性能。

当前主要系统代价也已经显现：阶段分离可能造成 HOL blocking；完整 Auto trace 还受到显式 resize 和 shrink 后首个 decode step 冷启动影响。本轮 Auto 吞吐比输出一致的 Fixed-K6 低 8.7%，因此现阶段应把动态控制的贡献定位为容量安全、避免错误静态配置和稳态可恢复，而不是已经证明的总体吞吐优势。

所有正式主张都必须保留单平台、减少重复次数、部分 arm 输出不一致和硬件计数器归因不足这些边界。当前结果足以指导论文重写和下一轮正式矩阵，但尚不足以直接替代最终 EuroSys 图表与统计验证。
