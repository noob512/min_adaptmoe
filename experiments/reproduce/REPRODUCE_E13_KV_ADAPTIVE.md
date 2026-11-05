# 复现 E13 KV Cache 自适应档位与固定档位对照

## 实验范围

E13 在相同 Qwen3-30B-A3B-FP8 模型、fixture、请求序列和 seed 上比较四个独立冷进程 arm：

- `adaptive`：从 K4 启动，根据预测和实时 KV 压力在 K2/K4/K6 间切换；
- `fixed_k2`、`fixed_k4`、`fixed_k6`：只在初始化时选择档位，运行期间不切换。

KV 档位与专家 Plan Class 是联动的：KV GPU target 增大时同时增加专家权重
卸载量，以保持 HBM 预算。

| 档位 | GPU KV target | CPU/offline KV | Plan Class | 专家卸载 |
| --- | ---: | ---: | --- | ---: |
| K2 | 2.145 GB | 6.442 GB | cold70 | 15.854 GB |
| K4 | 4.294 GB | 4.294 GB | cold80 | 18.119 GB |
| K6 | 6.441 GB | 2.147 GB | cold89 | 20.158 GB |

表中 GB 为十进制单位。精确几何为 5,461 个 1,572,864-byte KV block，其中一个是
null block；每档保留 16 个 GPU backup block。

当前协议启用 `VLLM_UVM_KV_SCHEDULER_BOUNDARY_RESIZE_ENABLE=1`。普通非流水
`EngineCore.step()` 在上一轮 model future 完成后、下一轮 `schedule()` 前执行在线切档；
`scheduler.running` 可以非空，但不得有 CUDA kernel/future 在途。pipeline batch queue
路径仍保守地等待全局 idle。

同时启用 `VLLM_UVM_KV_AUTO_SCALE_SYNCHRONIZED_TIER_STEPS=1`。K2/K4/K6 是绑定
专家 Plan Class 的同步档位，一次底层事务只允许移动到相邻档：K2↔K4 迁移 1,366 个
KV block（2.148 GB），K4↔K6 迁移 1,365 个 block（2.147 GB）。若预测需要 K2→K6，
控制器会在同一个 scheduler quiescent point 依次提交 K2→K4、K4→K6；每一步都先
完成对应专家卸载再 H2D KV，禁止一次 4 GB 跳档。缩档逐级执行 D2H KV 后再恢复对应
专家档，整个链路完成前不会进入下一次 `schedule()`。

缩档保持连续 GPU 前缀。控制器使用 `max_active_block_id` 推导最大的连续空闲尾部，只有
候选新 target 以上全部 `ref_cnt==0` 且容量、backup、safety 条件连续三轮满足时才提交；
提交前再次逐块校验。`min(free_block_id)` 仅记录为诊断，不能作为安全条件。扩档只迁移
原 offline 后缀，迁移成功前不向 allocator 公开新增 block。

## 冻结 workload

`config.json` 定义四类连续 trace：

1. `low_pressure`：4 个 Short/Output-32、B1 低速请求；
2. `high_pressure`：6 个并发 RAG-8192/Output-16 请求；
3. `dynamic_mixed`：Short、Ordinary、Medium 和 RAG cohort 按不同 engine-step 注入；
4. `burst_recovery`：运行中注入 RAG/B5；Adaptive 在当前 token step 结束后的 scheduler
   quiescent point 同步扩档，事务完成后立即准入，再通过连续低压恢复。

qualification 使用 seed `20260901`；formal 和 capacity 使用
`20260901/20260917/20260929`，即每个 arm 三个独立冷进程。capacity 额外用
RAG-8192/Output-16 扫描 B2/B5/B6/B7/B8/B10，测量同时 resident 上限；四个 arm
在同一 repeat 内的 `trace_sha256` 必须相同。

## 前置条件

```bash
cd /home/ubuntu/nvidia-uvm-gpu
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
sudo fuser -v /dev/nvidia-uvm /dev/nvidia-uvm-tools 2>/dev/null || true
modinfo nvidia_uvm | rg 'filename|version|srcversion'
```

开始前必须是 GPU 0 MiB、无 compute/UVM client，且已加载 stock `nvidia_uvm`。runner 会
临时切换到定制模块，每个 case 后验证 GPU clean，整个矩阵结束后恢复 stock 模块。

首先运行不触发 GPU 正式矩阵的静态检查：

```bash
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  static-preflight
```

该命令包含 shell/Python 语法检查、E13 工具测试和 autoscale/Plan Class 单元测试。

## 单重复 qualification

`RUN_ROOT` 必须指向不存在的新目录：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e13-kv-adaptive-REPRO-<tag>-attempt-01-qualification \
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  qualification
```

## 三重复正式矩阵

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e13-kv-adaptive-REPRO-<tag>-attempt-01-formal \
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  formal
```

默认顺序为每个 seed 内 `adaptive -> fixed_k2 -> fixed_k4 -> fixed_k6`。完整 formal 共 12 个
case，不要并发运行或在运行期间占用 GPU。失败 attempt 必须原样保留。

## 三重复容量包络

容量矩阵必须使用新的目录，不能覆盖主 formal：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e13-kv-adaptive-REPRO-<tag>-attempt-01-capacity \
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  capacity
```

它同样运行 12 个独立冷进程。Output-16 是测量协议的一部分：Output-1 会在首次
Prefill step 结束，post-step 快照无法观测 resident 集合，不得用于 max-running 结论。

## 不计切换开销的单轮稳态阶段对照

此协议回答“经历运行期切换后，目标档的 Prefill/Decode 是否回到直接建立该档的
性能水平”。只运行一轮，共 6 个独立冷进程：
`direct_k2/switched_k2/direct_k4/switched_k4/direct_k6/switched_k6`。

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e13-tier-phase-once-REPRO-<tag>-attempt-01 \
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  phase-once
```

`RUN_ROOT` 必须不存在，不得覆盖成功或失败 attempt。协议固定为：

- Medium-2048，B1/B8，32-token 冻结 continuation，不记录 logits；
- `direct` 在服务前建立目标档；`switched` 先完整运行源档 32-token conditioning，
  再切换 K4→K2、K6→K4 或 K4→K6；
- 源档 conditioning、切换事务以及两臂共有的目标档 B8 warmup 都在计时窗外；
- 正式 Prefill 记录完整墙钟；Decode 先跑 5 个 warmup step，再取 16 个
  whole-batch 单 token step 的中位数；
- 执行路径为 eager，目标 KV target 和 Plan Class 在测量期间必须不变，两臂输出 hash
  必须一致。

验证：

```bash
ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/<phase-root>
jq '{status,expected_case_count,observed_case_count,\
all_prefill_cells_equivalent_within_10pct,\
all_decode_cells_equivalent_within_10pct,rows,\
failed_checks:[.checks[]|select(.ok==false)]}' "$ROOT/verification.json"
cat "$ROOT/summary.md"
```

`status=ok` 表示六臂的功能、输出、档位稳定性和资源清理通过；性能等价需要单独查看
两个 `*_equivalent_within_10pct` 字段。这是单轮 qualification，不是统计等价检验。
冻结 continuation 不保存 logits、hidden state 或专家路由 hash，所以输出 hash 门只是
受控 token 序列与完整执行检查，不得解释为自然生成的数值等价。
若某点超出 ±10% 带，仍应原样报告，不得重复运行直到得到正面结果。

注意：上述固定 5-step warmup 已被后续诊断证明不足。attempt-06 的原始数据保留，
但不得再用其 Decode 超带点归因切档机制。新的 Decode 因果比较必须使用下面的
同进程 ABBA，或另行实现滚动窗口收敛门。

## K2/K6 固定档与 Adaptive 回环单轮对照

此入口同时回答三件事：低压时 K2 是否优于 K6；高压时 K6 是否有吞吐/并发容量
优势；忽略切换与 warmup 后，Adaptive 的 K2→K6→K2 三个稳态是否分别恢复到
对应 Fixed 档。只执行一次顶层实验，内部使用三个独立冷进程。

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e13-k2-k6-roundtrip-REPRO-<tag>-attempt-01 \
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  k2-k6-roundtrip-once
```

三个冷进程依次为 `fixed_k2`、`fixed_k6`、`roundtrip`。协议固定为：

- 低压 Short-128/B1→32：1 次排除 warmup，3 次进程内测量；
- 高压 RAG-8192/B6→16：1 次排除 warmup，2 次进程内测量；
- 稳态阶段 Medium-2048、B1/B8→32：目标档 B8 warmup 排除，每点 3 次进程内测量；
- `roundtrip` 从 K2 开始，严格按 K2→K4→K6→K4→K2 相邻约 2 GB 事务切换；
- 切换、conditioning 和目标档 warmup 均不计时；正式比较使用冻结 continuation；
- “相同水平”预注册为 Adaptive/Fixed 延迟落在 `[0.90,1.10]`，超带也必须原样
  报告，不能通过重跑筛选结果。

查看结果：

```bash
ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/<roundtrip-root>
jq '{status,fixed_comparison,hypotheses,
all_equivalence_cells_within_10pct,equivalence_rows,
failed_checks:[.checks[]|select(.ok==false)]}' "$ROOT/verification.json"
cat "$ROOT/summary.md"
```

`status=ok` 只表示三臂执行、输出 hash、档位稳定性、相邻切换和资源清理通过；三项
性能假设必须查看 `hypotheses`。高压容量以 `max_scheduler_running_requests` 判断；
调度器允许超额请求排队，所以 K2 的 `max_running<B` 表示不能让整个 batch 同时驻留，
不应写成 OOM 或请求无法最终完成。进程内多个样本不是独立冷进程重复，该入口仍是
单轮 qualification。

只补测 Short-128/B6→32、Short-128/B64→32 或 Short-128/B128→32 的
Fixed-K2/Fixed-K6 两臂时，使用
独立入口，不能重跑完整回环矩阵：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e13-fixed-k2-k6-short-b6-REPRO-<tag>-attempt-01 \
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  short-b6-once

# B64 使用新的 RUN_ROOT，并将最后一行改为：
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  short-b64-once

# B128 使用新的 RUN_ROOT，并将最后一行改为：
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  short-b128-once
```

该入口运行两个独立冷进程，每臂一次排除 warmup和三个进程内稳态样本，输出 req/s、
tok/s、TTFT、TPOT、单请求 E2E、整批墙钟、Prefill 与 steady Decode step。两臂必须
`max-running=6`、输出 hash 一致、无 preemption；三个进程内样本仍不构成三个独立
冷进程重复。

B64 入口会把有效 `max_num_seqs` 提升为 64；32 条冻结 Short fixture 按相同顺序复制
一轮，prefix cache 保持关闭。验收改为 `max-running=64`、`max-waiting=0`，且每个
正式 batch 必须严格包含 1 个 Prefill step 和 31 个 Decode step。

B128 入口把有效 `max_num_seqs` 提升为 128；冻结 Short fixture 循环四轮，prefix
cache 保持关闭。每请求需要约 10 个 KV blocks，B128 的约 1,280 blocks 小于 K2
的 1,364-block target，因此两档都必须达到 `max-running=128`，不能因 KV 容量拆分
cohort。冻结的 `max_num_batched_tokens=8192` 使 16,384 个 Prompt tokens 分成两个
Prefill step；首步的 `max-waiting=64` 是 token budget 限制，不是 KV 容量不足。
每个正式 batch 必须为 2 个 Prefill step 加 31 个 Decode step。

若要判断B128 Decode差异是否来自档位本身，必须使用同进程ABBA入口，不能再比较
两个独立冷进程的三点中位数：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e13-short-b128-decode-abba-REPRO-<tag>-attempt-01 \
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  short-b128-decode-diagnose
```

该入口在一个冷进程内严格执行`K2→K6→K6→K2`。每段仍有一个排除warmup和三个
测量batch，并保存31个Decode step的原始延迟序列。切换按K2↔K4↔K6相邻边执行且
不计入性能窗口；验收要求四段均`max-running=128`、档位稳定、输出hash一致和资源
清理通过。ABBA用于识别历史相关快慢态，单轮结果不是统计显著性证明。

按同一档内 B1→B16→B64 顺序测量，并在完成整个 K2 sweep 后再运行 K6 sweep：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e13-fixed-k2-k6-short-b1-b16-b64-REPRO-<tag>-attempt-01 \
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  short-batch-matrix-once
```

该入口只有两个冷进程，固定进程顺序为 `fixed_k2`、`fixed_k6`；每个进程的 segment
顺序必须为 B1、B16、B64。所有 cell 统一 `max_num_seqs=64`，每个 batch 先做一次
排除 warmup，再取三个进程内样本。verifier 对每个正式 batch 强制检查1个Prefill
step、31个Decode step、`max-running=B`、`max-waiting=0`和跨档输出hash一致。

## B64/B128 双阶段 CUDA Graph 与快速 P→D 切换

前述Short入口默认使用eager，只能用于隔离档位性能。需要测量完整的双阶段Graph机制
时使用以下专用入口：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e13-fixed-k2-k6-short-b64-b128-full-separate-REPRO-<tag>-attempt-01 \
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  short-graph-matrix-once
```

该命令只执行一次顶层实验，依次启动Fixed-K2和Fixed-K6两个冷进程。每个进程按
B64→B128运行，每个cell包含一次排除warmup和三个进程内样本。固定协议为：

- `TRITON_ATTN`与`CUDAGraphMode.FULL_SEPARATE`；
- `VLLM_UVM_ORIGINAL_VA_PHASE_MODE=prefill_graph_remote_decode`；
- original-VA双页表与park/resume快速P→D切换；
- Graph shape为Decode 64/128和Prefill 8192 tokens；
- K2/K6分别只注册cold70/cold89，避免捕获不参与本臂的Plan Class；
- K2为1,364 blocks，K6为4,095 blocks，运行期间固定，不计档位迁移。

验收不是只检查配置。`verify_e13_short_graph_matrix.py`要求每个正式B64 batch的32个
step、每个B128 batch的33个step全部各产生一次FULL replay，禁止eager fallback和
运行时Graph重捕获；还要求每个请求周期确实发生alias Prefill恢复和remote Decode
cutover，结束于`remote_parked`。同时检查完整并发、输出hash、Xid/PTE/PDE/MMU fault、
GPU清理及stock UVM恢复。

查看结果：

```bash
ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/<full-separate-root>
jq '{status,comparisons,failed_checks:[.checks[]|select(.ok==false)]}' \
  "$ROOT/verification.json"
cat "$ROOT/summary.md"
```

P→D列为首个Decode step，包含park/remap提交；steady Decode取最后16个Decode step
的中位数。三个样本属于同一进程，不是三个独立冷进程重复。

### Adaptive K2→K6→K2 完整Graph回环

在相同B64/B128完整机制下同时比较Fixed与Adaptive，使用新的`RUN_ROOT`执行：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e13-short-b64-b128-full-separate-roundtrip-REPRO-<tag>-attempt-01 \
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  short-graph-roundtrip-once
```

不要设置`E13_GRAPH_ARMS`；该变量只供单臂故障诊断，设置后不会生成三臂正式
verification。正式顺序为`fixed_k2→fixed_k6→roundtrip`。Adaptive预捕获K2/K4/K6
三档Graph，回环严格为`K2→K4→K6→K4→K2`；四个切档事务和每个目标档的一次
warmup均排除。每个正式B64 batch必须有32次FULL replay，B128必须有33次，运行期
禁止Graph capture与eager passthrough。

查看结果：

```bash
ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/<graph-roundtrip-root>
jq '{status,all_equivalence_metrics_within_10pct,fixed_comparisons,\
outside_band:[.equivalence[]|select(.within_10pct==false)],\
failed_checks:[.checks[]|select(.ok==false)]}' "$ROOT/verification.json"
cat "$ROOT/summary.md"
```

`status=ok`表示三臂、逐step Graph、快速P→D、输出、档位回环和资源清理全部通过。
`all_equivalence_metrics_within_10pct`还包含P→D step等9项性能指标，必须与核心
Prefill/Decode/E2E/吞吐比较分开解释；任一超带点都应原样报告，不得通过重跑筛选。
单次顶层实验仍只有每臂一个冷进程，cell内三个样本不是独立重复。

## Decode 差异定因

同进程 ABBA 同时记录逐层逐 step 路由 hash、remote assignment、KV block 顺序 hash、
页表/TLB 事务、MoE/attention kernel 时间和 NVML PCIe RX：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
ALLOW_GPU_RUN=1 \
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e13-decode-abba-diagnostic-REPRO-<tag>-attempt-01 \
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  decode-diagnose
```

CPU backing 4K/64K/2M 构成、冷进程离散度、时钟、P-state 和 PCIe link 诊断：

```bash
ALLOW_GPU_RUN=1 \
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e13-decode-backing-diagnostic-REPRO-<tag>-attempt-01 \
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  decode-backing-diagnose
```

Nsight Systems 单例 kernel/span/gap 诊断：

```bash
ALLOW_GPU_RUN=1 \
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e13-decode-nsys-diagnostic-REPRO-<tag>-attempt-01 \
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  decode-nsys-diagnose
```

Nsight 入口用 `cudaProfilerStart/Stop` 限定 condition window，NVTX 标注每个 step。
成功条件包括 `timeline.nsys-rep`、`timeline.sqlite` 和 `analysis.json` 均存在且
`analysis.status=ok`。Nsight 会显著放大 wall-time；只用其 kernel duration、类别和
span/gap 诊断，不把被剖析 wall-time 写入性能表。

当前权威诊断结果为：K4/K6 四个 ABBA 点 switched/direct 为
`0.983×/0.986×/1.013×/1.000×`；原 K4/B8 `1.324×` 与 K6 约 `0.78×` 的偏离消失。
错误原因是独立冷进程的固定 5-step warmup 没有保证 remote-read 路径收敛，不是切档
改变了稳态 Decode 路径。微观变化集中在 MoE/remote-read 侧，但不得在没有专用硬件
计数器时进一步声称是某一级 TLB。

## 离线验证、聚合与绘图

顶层命令成功时已自动执行以下步骤。需要重建分析产物时，先选择新的
`--output-dir`，不覆盖已发布证据：

```bash
ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/<successful-formal-root>
CFG=experiments/eurosys_moe_offload_evaluation/e13/config.json
PY=workloads/vllm/vllm/.venv/bin/python

"$PY" experiments/eurosys_moe_offload_evaluation/e13/verify_e13_kv_adaptive.py \
  "$ROOT" --config "$CFG" --profile formal \
  --output "$ROOT/verification.recheck.json"

"$PY" experiments/eurosys_moe_offload_evaluation/e13/aggregate_e13_kv_adaptive.py \
  "$ROOT" --config "$CFG" --profile formal \
  --output-dir "$ROOT/analysis-recheck"

"$PY" experiments/eurosys_moe_offload_evaluation/e13/plot_e13_kv_adaptive.py \
  "$ROOT" --config "$CFG" --profile formal \
  --output-dir "$ROOT/analysis-recheck"
```

主要产物为：

- `verification.json`：12 个 case 的 fail-closed 验收；
- `analysis/e13_summary.{json,csv,md}`：分场景均值、样本标准差与 min/max；
- `analysis/e13_cases.csv`：每个冷进程的原子统计；
- `analysis/e13_global.csv`：完整 trace 的 arm 聚合；
- `analysis/e13_timeseries.csv`：KV 压力、档位、Plan Class、队列和吞吐时序；
- `analysis/e13_capacity_{cases,envelope}.csv`：逐进程与三重复容量包络；
- `analysis/e13_kv_pressure_tier_throughput_latency_timeseries.{png,svg}`：可视化。

## 验收条件

```bash
jq '{status,expected_case_count,observed_case_count}' "$ROOT/verification.json"
jq '{global_best_fixed,scenario_comparisons,global_comparison}' \
  "$ROOT/analysis/e13_summary.json"
```

必须满足：

- `verification.status=ok` 且 12/12 case 存在、返回码为 0；
- 每个 repeat 的 prompt/trace hash 一致，成功请求有完整 TTFT/TPOT/E2E 指标；
- Fixed 档位不变且运行期 resize 为 0；
- Adaptive 覆盖 K2/K4/K6，同时观测到 grow 和 shrink，且 resize 尝试无失败；
- Adaptive 的全部 4 个 late cohort 都有统一控制器记录；burst 必须出现 grow 及其后的
  recovery shrink；RAG/B5 必须在扩档提交后准入，但不得等待已有请求完成；
- 每次在线 shrink 的候选边界以上均为连续空闲尾部；运行中 grow 必须记录
  `committed_with_running_requests=true`；
- 每个 resize event 只能连接相邻的 K2/K4 或 K4/K6，KV target 与专家 Plan Class
  在恢复调度前必须一致；
- 三队列守恒、KV target 与 Plan Class 一致；
- 无 Xid/MMU fault、GPU 或 UVM client 泄漏，最终恢复 stock `nvidia_uvm`。

capacity profile 还必须满足：Adaptive/Fixed-K2/Fixed-K4/Fixed-K6 的
`maximum_fully_resident_offered_concurrency` 分别为 7/2/5/7，B8/B10 超额请求计入
`requests_not_simultaneously_resident`，不能标成 OOM 或失败。

完整输出 token hash 作为数值敏感性诊断，必须报告分叉；当前验收的硬门是同一
prompt/trace、请求完整生成与共同首 token 一致。不得把该门表述为逐 token 完全一致。

## 限制

E13 是单机单卡、phase-batched、eager 路径的三 seed qualification，不是开环生产服务
soak。capacity profile 给出当前 B≤10、context≤8,208 的 fully-resident 上限；请求仍可
排队后完成，所以这不是物理 OOM/拒绝上限。per-scenario best-fixed 是事后 oracle；
整条 trace 的 global best-fixed 才是长期固定策略的主要对照。

在线切换仍为同步事务，因此当前请求会出现一次覆盖 Plan Class 和 KV 迁移的 TPOT 停顿；
它消除了“等待请求生成结束”，没有消除迁移本身。pipeline batch queue、异步后台迁移和
多GPU仍不在本协议范围内。

## 当前成功证据

最新 scheduler-boundary、同步约 2 GB 档 qualification：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e13-kv-online-synchronized-2gb-20260901-attempt-08-qualification/
```

该目录 4/4 冷进程返回 0，`verification.json` 为 `status=ok`，128/128 请求成功。
Adaptive 完成 7/7 次实际 resize，所有边均为 K2↔K4 或 K4↔K6；4 次 shrink 都有
连续空闲尾部证明，burst 的 K2→K4 记录
`committed_with_running_requests=true`，没有 deferred admission。burst late cohort
到达至准入为 743.345 ms，不再等待已有 256-token 请求结束。恢复调度后的所有时序点
均满足 KV target/Plan Class 一致，三队列错误为 0，无 Xid/MMU fault，最终恢复 stock
UVM 且 GPU 为 0 MiB。burst 的 12 个请求相对三个 Fixed 臂共 36/36 完整输出 hash
一致，其中包括切换时仍在生成的 256-token 请求。

Fixed-K6 的初始化记录为 `1364→2730→4095`，用于证明固定高档也遵守相邻约 2 GB
事务。该 qualification 只有一个 seed；其中 Adaptive 为 0.3090 req/s，best
Fixed-K4 为 0.3854 req/s，因此只能作为在线机制正确性证据，不能作为 Adaptive
性能领先或统计显著性的证据。

2026-09-01 三重复目录（历史 drain-at-idle 协议）：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e13-kv-adaptive-20260901-attempt-10-formal-burst-complete/
```

该目录 12/12 case 返回 0，`verification.json` 为 `status=ok`，合计 384/384
请求成功。修复后的 burst 三轮均为 `K2→K4→K2`，且 5 个 RAG 请求在扩档后
准入。Adaptive 完整 trace 吞吐为 0.2163 req/s，只有 global best Fixed-K6 的
0.714×；因此该成功 artifact 是机制 qualification 和负性能结果，
不是“Adaptive 性能更好”的证据。

该历史 artifact 未启用 scheduler-boundary resize，其中 burst arrival→admission 平均
26.519 s。当前在线协议必须写入新的 attempt 目录；不得用下列历史数值代替在线结果。

容量包络目录：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e13-kv-adaptive-20260901-attempt-13-capacity-formal/
```

该目录 12/12 case、456/456 请求通过。Adaptive/Fixed-K2/Fixed-K4/Fixed-K6 的
fully-resident 上限为 B7/B2/B5/B7；Adaptive capacity 吞吐为 best Fixed-K4 的
0.702×。

完整解释见：

```text
experiments/eurosys_moe_offload_evaluation/
  RESULTS_E13_KV_ADAPTIVE_20260901.md
```

不计切换开销的单轮稳态阶段证据：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e13-tier-phase-once-20260901-attempt-06/
```

该目录 6/6 arm 返回 0，`verification.status=ok`，冻结输出 hash 6/6 一致。
Prefill 6/6 cell 在 switched/direct ±10% 内。Decode 原始表只有 3/6，但该解释已被
同进程 ABBA 定因结果取代：超带来自 warmup 协议错误，不能归因切档。该 attempt 是
正式受控序列原始结果；自由 greedy 的 attempt-05 因 K4/B1、K4/B8 输出 hash 分叉而
只作辅助诊断。

K2/K6 固定档与 Adaptive 回环单轮证据：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e13-k2-k6-roundtrip-20260901-attempt-01/
```

该目录 3/3 冷进程返回 0、245/245 请求完成，`verification.status=ok`。低压 K2/K6
吞吐比为 1.065×；高压 K6/K2 吞吐比为 1.257×，max-running 为 6/2。
Adaptive 初始 K2、增长后 K6、回切后 K2 相对对应 Fixed 档的 Prefill/Decode/batch-wall
共 18/18 个 cell 位于 ±10% 内，受控输出 hash 全部一致。四次实际切换均为相邻约
2 GB 事务；无 Xid/MMU fault，最终恢复 stock UVM 且 GPU 为 0 MiB。

Short/B6 固定档补充证据：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e13-fixed-k2-k6-short-b6-20260901-attempt-01/
```

该目录 2/2 冷进程返回 0、48/48 请求完成，verifier 为 `status=ok`。两档均可同时
驻留 B6；Fixed-K2/Fixed-K6 吞吐比为 1.049×，TTFT 和 E2E 比为 0.810×、0.953×，
受控输出 hash 一致，无 Xid/MMU fault，最终 GPU 0 MiB 并恢复 stock UVM。

Short/B64 固定档补充证据：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e13-fixed-k2-k6-short-b64-20260901-attempt-01/
```

该目录 2/2 冷进程返回 0、512/512 请求完成，verifier 为 `status=ok`。两档都完整
驻留 B64，Fixed-K6/Fixed-K2 吞吐比为 1.123×；输出 hash 一致，无 Xid/MMU fault，
最终 GPU 0 MiB 并恢复 stock UVM。

Short B1/B16/B64 同进程档内矩阵证据：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e13-fixed-k2-k6-short-b1-b16-b64-20260901-attempt-01/
```

该目录 2/2 冷进程返回0、648/648请求完成，verifier为`status=ok`。B16/B64中
Fixed-K2吞吐分别高6.67%/7.57%；K6/B1直接出现Decode慢态到快态的进程内转换，
因此孤立B64的K6表观优势不再作为机制结论。六个cell输出hash一致，无Xid/MMU
fault，最终GPU 0 MiB并恢复stock UVM。

Short/B128 固定档补充证据：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e13-fixed-k2-k6-short-b128-20260901-attempt-01/
```

该目录2/2冷进程返回0、1,024/1,024请求完成，verifier为`status=ok`。两档每个
正式样本都达到`max-running=128`；K2/K6中位吞吐为34.8089/35.1977 req/s，差异
只有1.12%，且两臂Decode样本分别出现快慢态，不能解释为稳定档位优势。K2 Prefill
和TTFT分别低14.57%/14.51%。跨档输出hash一致，无Xid/MMU fault，最终GPU 0 MiB
并恢复stock UVM。

Short/B128 Decode同进程ABBA定因：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e13-short-b128-decode-abba-20260901-attempt-01/
```

该目录1/1冷进程返回0、2,048/2,048请求完成，verifier为`status=ok`。初始K2为
65.22/65.59/65.73 ms，K6两段稳定在67.10–68.40 ms，但回切K2为
84.92/85.43/84.44 ms。相同K2档位相差29.48%，证明独立冷进程表的K2慢态不是
档位固有性能。逻辑Plan、KV target、输出和PCIe Gen5×16一致；底层差异收敛到
remote-read的GMMU cache/PCIe request状态，当前计数器不能进一步唯一拆分。

辅助的`e13-short-b128-decode-shrink-isolation-20260901-attempt-01`只在Xid前完成四段：
初始K2从84.37/83.66自行转为66.63 ms，K4→K2后K2稳定在约67.11 ms，再次排除
K2固有慢态；随后K6→K4触发Xid 31/PTE read fault，故该目录为`status=error`，
不得当作完整通过或使用故障后的数据。

双阶段CUDA Graph B64/B128固定档证据：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e13-fixed-k2-k6-short-b64-b128-full-separate-20260901-attempt-01/
```

该目录2/2冷进程返回0、1,536/1,536请求完成，verifier为`status=ok`。正式测量的
390/390个serving step均为FULL Graph replay，每个请求周期都通过park/resume完成
alias Prefill→remote Decode切换，捕获后无重捕获。B64/B128上K2吞吐分别比K6高
11.11%/11.21%，E2E分别低10.01%/10.09%；输出hash一致，无GPU/UVM故障，最终GPU
0 MiB并恢复stock UVM。

双阶段CUDA Graph B64/B128 Fixed/Adaptive回环证据：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e13-short-b64-b128-full-separate-roundtrip-REPRO-20260902-attempt-02/
```

该目录3/3冷进程返回0，verifier为`status=ok`，84/84协议检查通过。正式测量
975/975个serving step均为FULL Graph replay；Adaptive完成
`K2→K4→K6→K4→K2`且capture count保持9。Fixed-K2在B64/B128的吞吐分别比
Fixed-K6高15.09%/21.86%。排除档位事务和warmup后，Adaptive三个状态相对对应
Fixed的吞吐、Prefill、steady Decode和E2E共24/24个比较位于±10%；全部9类指标
为49/54，5个超带点均为P→D step（高10.92%–24.03%）。输出hash一致，无
Xid/MMU fault，最终GPU 0 MiB并恢复stock UVM。该轮加载的当前源码模块srcversion
为`D158DB985980F4A580DB740`，SHA-256为
`4168020f14fc3217f8d217e4fa8898299e4d81c0c18653c98cce9d73346afdbe`。
该目录是在同步cutover/ReadMostly CPU duplicate修复后生成的当前权威完整机制证据；
上一成功轮`e13-short-b64-b128-full-separate-roundtrip-20260901-attempt-03-source-validated`
及更早固定臂目录只保留作历史记录。

第一次按本文档复现的`REPRO-20260902-attempt-01`在Adaptive预捕获时暴露big-PTE有效
下级映射被误判为缺少4K页表的问题，返回134并触发Xid 31。修复驱动后从三个arm完整
重跑得到上述attempt-02；两个attempt都必须保留。详细定因、补丁验收和新旧轮差异见
专用入口`REPRODUCE_E13_FULL_SEPARATE_K2_K6_ADAPTIVE_ROUNDTRIP.md`。
