# 复现 E10：当前档位双页表、2→4→6 GiB KV 压力切换

## 2026-09-02当前权威轮

最新成功artifact为：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
e10-kv-2-4-6-pressure-20260902-attempt-01-scheduler-quiescent-sync-tier
```

该轮使用scheduler quiescent point、KV与专家Plan Class同步相邻切档、全覆盖双页表和
快速P→D，verifier为`status=ok`。custom UVM为：

```text
srcversion  18E6563D21D9D2937D99EDF
SHA-256     90fb5a7d9852a754ecf0358700d36dbcd04abd19886753ee5486401a937e5e15
```

| KV档 | 纯Prefill | 档位事务 | P→D | steady Decode | 请求E2E |
|---:|---:|---:|---:|---:|---:|
| 2 GiB/cold70 | 340.74 ms | 0 | 108.40 ms | 20.11 ms | 1007.96 ms |
| 4 GiB/cold80 | 399.37 ms | 542.80 ms | 117.82 ms | 24.25 ms | 1672.99 ms |
| 6 GiB/cold89 | 447.60 ms | 565.27 ms | 138.35 ms | 26.78 ms | 1831.87 ms |

2→4与4→6 GiB分别移动相邻约2 GiB KV容量并同步切换专家档。三档输出hash一致、双页表
覆盖100%、无range-map fallback、Graph重捕获或GPU/UVM故障；结束后恢复stock UVM。
这仍是一个冻结压力序列、一个冷进程样本，不是自动联合控制器formal或统计结论。

推荐运行命令：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e10-kv-2-4-6-pressure-NEW-attempt-01-scheduler-quiescent-sync-tier \
E10_TRIGGER_MODE=pressure ALLOW_GPU_RUN=1 \
bash experiments/vllm_uvm_e10_multilevel_pressure/run_experiment.sh qualify
```

以下章节保留较早协议和pressure/direct诊断；与本节冲突时以本节为准。

## 1. 实验目标和边界

E10 验证在固定 8 GiB KV 虚拟地址池中，系统能否按 backup 队列的真实压力把 GPU 上的
KV 物理容量从 2 GiB 扩到 4 GiB、再扩到 6 GiB，同时把专家冷集合从 cold70 切到
cold80、再切到 cold89，并保持 CUDA Graph、Parameter 虚拟地址和输出不变。

每次档位变化后只保留当前档位的一对专家页表：Prefill 视图指向 GPU 专家缓存，Decode
视图指向固定的 CPU shadow。两种视图必须覆盖全部专家相关的 2 MiB 父入口，运行时以父入口
切换，不能退回逐个 CPU range-map。

默认协议是单轮 qualification：每档一个请求、完整 2→4→6 GiB 序列一次、统计重复为 0、
失败后不自动重试。因此结果只能说明该轮功能和观测性能，不能作为统计显著性结论。

## 2. 冻结配置

| 项目 | 固定值 |
|---|---|
| 模型 | Qwen3-30B-A3B-FP8，snapshot `d206ba732169f29bb77fbf80fc2c4b81d4d30782` |
| GPU | 单卡 RTX 5090；已完成记录驱动为 595.71.05 |
| 专家策略 | cold-frequency；cold70→cold80→cold89，三档逐层严格嵌套 |
| 专家主权重 | cold70 15,854,469,120 B；cold80 18,119,393,280 B；cold89 20,157,825,024 B |
| KV 虚拟地址预留 | 8,589,410,304 B，实验全程不变 |
| KV GPU 目标 | 1364、2730、4095 blocks，对应约 2、4、6 GiB |
| backup 目标 | 1348、2714、4079 blocks；每档保留 16 个普通 free block |
| 请求 | B1；同一 fixture 请求；每档固定输出 22 tokens（1 次首次 Decode + 5 次 warmup + 16 次 measured），另有 1 次 Prefill |
| Graph | Prefill/Decode FULL Graph；capture sizes 1、314 |
| 双页表 | 完整 Parameter CPU shadow；2 MiB 父入口；每批 64 个；覆盖率必须为 100% |
| 自动迁移 | manual fault mode 2；access-counter、prefetch、thrashing 均关闭 |
| 预测 | 关闭；增长只允许由 backup 队列阈值触发 |
| seed | 0 |

重要输入位于：

```text
experiments/vllm_uvm_e10_multilevel_pressure/inputs/cold70.json
experiments/vllm_uvm_e10_multilevel_pressure/inputs/cold80.json
experiments/vllm_uvm_e10_multilevel_pressure/inputs/cold89.json
experiments/vllm_uvm_e10_multilevel_pressure/inputs/plan_classes.json
experiments/vllm_uvm_large_kv_multilevel/inputs/exact_parameter_names.txt
experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json
```

如需从冻结的 cold70 基线和专家排序波次重新生成三档计划，执行：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
workloads/vllm/vllm/.venv/bin/python \
  experiments/vllm_uvm_e10_multilevel_pressure/prepare_plan_classes.py \
  --base-plan experiments/vllm_uvm_large_kv_multilevel/inputs/cold_decode_plan_base70.json \
  --wave-plan experiments/vllm_uvm_large_kv_multilevel/inputs/incremental_expert_waves.json \
  --output-dir experiments/vllm_uvm_e10_multilevel_pressure/inputs
```

该命令会改写冻结输入；正式复现实验通常应直接使用仓库中已经审核的 JSON，并由 harness
记录 SHA256。

## 3. 构建和静态检查

先构建自定义 UVM 模块和 vLLM UVM helper。若当前工作区已经生成与源码对应的文件，可跳过
重复构建，但不能使用来源不明的旧模块。

```bash
cd /home/ubuntu/nvidia-uvm-gpu
make -C kernel-module/nvidia-module -j"$(nproc)" modules

make -C workloads/vllm/vllm/uvm_test

cd /home/ubuntu/nvidia-uvm-gpu
E10_TRIGGER_MODE=pressure \
bash experiments/vllm_uvm_e10_multilevel_pressure/run_experiment.sh static-preflight
```

静态检查会验证 96 个 Parameter 名称、三档计划严格嵌套、目标 block 映射、Python/Bash
语法及与 E10 相关的单元测试。它分别校验8 GiB虚拟地址池为5,461 blocks、最终6 GiB
GPU物理档位为4,095 blocks，不能把两者当成同一目标。它不会加载模块或运行模型。

## 4. GPU 运行前检查

运行必须从系统 stock `nvidia_uvm`、0 MiB GPU 占用且无 UVM client 开始：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
sudo fuser -v /dev/nvidia-uvm /dev/nvidia-uvm-tools 2>/dev/null || true
printf 'loaded=%s\nstock=%s\n' \
  "$(tr -d '[:space:]' </sys/module/nvidia_uvm/srcversion)" \
  "$(modinfo -F srcversion nvidia-uvm | tr -d '[:space:]')"
```

不要手工终止未知 GPU 进程。harness 会暂停已知 exporter、取得独占锁、加载 custom UVM，
并在正常或失败退出时恢复 stock UVM 和 exporter。模块无法安全卸载时，保留产物并按错误处理
文档判断是否需要重启，不能把 partial run 当作成功数据。

## 5. 唯一一轮完整运行

输出目录必须事先不存在。为避免覆盖失败记录，每次给出新的、带 attempt 的路径：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e10-kv-2-4-6-pressure-REPRO-YYYYMMDD-attempt-NN-current-tier-dual \
E10_TRIGGER_MODE=pressure \
ALLOW_GPU_RUN=1 \
bash experiments/vllm_uvm_e10_multilevel_pressure/run_experiment.sh qualify
```

完整顺序为：

```text
加载模型并捕获 Graph
→ 建立 27 GiB Parameter CPU shadow
→ cold70 + 2 GiB KV
→ 运行2 GiB请求
→ 下一请求从backup取4 blocks，压力触发cold70→cold80和2→4 GiB
→ KV布局完成后重建cold80 Prefill/Decode双表
→ 运行4 GiB请求
→ 下一请求再次从backup取4 blocks，触发cold80→cold89和4→6 GiB
→ KV布局完成后重建cold89双表
→ 运行6 GiB请求
→ 校验、清理、恢复stock UVM
```

每档运行一次 Prefill、一次首次 Decode、5 次 Decode warmup 和 16 次 measured Decode。
4/6 GiB 的 `prefill_end_to_end_ms` 包含当轮压力触发的档位转换；纯 Prefill 必须读取
`prefill_compute_excluding_rebalance_and_resume_ms`，不能把两者混用。

## 6. 验收和离线复核

harness 成功后自动执行 verifier。也可在不重新运行 GPU 的情况下复核：

```bash
ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e10-kv-2-4-6-pressure-REPRO-YYYYMMDD-attempt-NN-current-tier-dual
E10_TRIGGER_MODE=pressure \
bash experiments/vllm_uvm_e10_multilevel_pressure/run_experiment.sh verify "$ROOT"

jq '{status,trigger_mode,kv_target_blocks,expert_plan_classes,
  output_token_ids_sha256,dual_view_parent_switch,timing}' \
  "$ROOT/verification.json"
```

成功必须同时满足：

- runner 与 verifier 均为 `status=ok`；三档和两个增长事务完整。
- 1364/2730/4095 block 的 GPU 前缀及 CPU 尾部与目标完全一致，8 GiB VA hash 不变。
- active plan 依次为 cold70/cold80/cold89，Parameter VA 全程不变。
- 两次增长均为专家档先切、KV 后扩，并在 KV 物理布局完成后刷新当前档位双表。
- 每档双表覆盖率 100%，Decode CPU range-map ioctl 为 0。
- 三档输出 hash 相同，Graph capture 数不增加。
- 内核日志无 Xid、MMU Fault、NULL pointer、kernel BUG 或 general protection fault。
- `module.after.json` 为 stock UVM，`sequence/gpu.after.json` 为 0 MiB/clean。

主要产物：

```text
protocol.json                       冻结实验协议
input_sha256.txt                    输入、runner、驱动及模块哈希
module.before/custom/after.json     模块和GPU状态
latency_stats.before/after.txt      驱动累计计数
sequence/result.json                原始逐步结果
sequence/run.log                    完整模型日志
verification.json                  机器可读验收结果
kernel.since.log                    本轮内核日志
```

当前已完成的参考运行是：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
e2-e3-e10-current-dual-view-REPRO-20260831-attempt-01/e10/
```

其解释报告为 `RESULTS_E2_E3_E10_CURRENT_DUAL_VIEW_REPRO_20260831.md`；该轮的联合协议是
历史材料，当前统一入口为本文件及
`experiments/reproduce/REPRODUCE_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md`。

## 7. 压力触发与直接切档对照

对照实验复用同一 runner、请求、档位事务和 verifier，只改变调用时机：

- `auto_pressure`：scheduler 从 backup 队列取到阈值后，在 Prefill schedule 内调用切档。
- `direct_tier`：在请求加入 scheduler 前、GPU 和 scheduler 均空闲时，显式调用完全相同的
  cold/KV 目标事务；请求仍会真实使用 backup 队列，但不会由该压力再次切档。

两个 arm 必须是独立冷模型进程，各跑一个完整序列且不重试：

### 7.1 实现入口

对照实现由以下四个入口组成：

| 文件 | 作用 |
|---|---|
| `experiments/vllm_uvm_prefill_alias_decode_remote_graph/benchmark_phase_switch_steps.py` | 接受 `--kv-multilevel-trigger-mode pressure/direct`，在同一请求循环中只改变切档调用时机 |
| `experiments/vllm_uvm_e10_multilevel_pressure/run_experiment.sh` | 将 `E10_TRIGGER_MODE` 转换为 runner 参数，并为单个 arm 完成模块加载、冷进程运行、校验和恢复 |
| `experiments/vllm_uvm_e10_multilevel_pressure/verify_experiment.py` | 分别验证 pressure/direct 的触发证据，同时要求物理布局、专家档、双表、输出和 Graph 正确 |
| `experiments/vllm_uvm_e10_multilevel_pressure/compare_trigger_modes.py` | 在不重新运行 GPU 的情况下读取两个 arm，生成统一的 `comparison.json` |

两条路径最终都进入 EngineCore 的同一个 `_resize_uvm_kv_target()` 事务：

```text
pressure：scheduler消耗backup
  → 压力达到4 blocks
  → 自动控制器选择下一档
  → _resize_uvm_kv_target(target)

direct：请求尚未加入scheduler、系统空闲
  → apply_uvm_kv_dynamic_pool_target(target)
  → _resize_uvm_kv_target(target)
```

`VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_MAP` 固定为：

```json
{"1364":"cold70","2730":"cold80","4095":"cold89"}
```

所以两条路径内部都执行同一顺序：

```text
恢复上一档Prefill双表
→ 根据target查找cold档位
→ 追加4,500或4,050条专家差量
→ 使用CPU shadow直接释放专家显存
→ 调整KV的GPU物理前缀
→ 重建当前cold档位的Prefill/Decode双表
```

direct arm 在调用完成后才加入请求，并保持 autoscale 关闭；该请求仍必须从 backup 队列取得
4 blocks，用来证明请求负载与 pressure arm 相同，但这个压力不会重复触发第二次切档。

### 7.2 静态预检

```bash
cd /home/ubuntu/nvidia-uvm-gpu
bash experiments/vllm_uvm_e10_multilevel_pressure/run_trigger_comparison.sh \
  static-preflight
```

预检会分别检查 `pressure` 和 `direct` 参数，并执行 E10 相关单元测试。通过标志为：

```text
E10 multilevel pressure static preflight: OK
E10 trigger comparison static preflight: OK
```

### 7.3 一条命令运行两个独立冷 arm

输出根目录及其 `auto_pressure`、`direct_tier` 子目录必须都不存在：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e10-trigger-mode-comparison-REPRO-20260830-attempt-01 \
ALLOW_GPU_RUN=1 \
bash experiments/vllm_uvm_e10_multilevel_pressure/run_trigger_comparison.sh qualify
```

固定运行顺序是 auto-pressure 后 direct-tier。每个单 arm harness 都会独立执行：

```text
确认stock UVM和GPU 0 MiB
→ 加载custom UVM
→ 启动一个全新模型进程
→ 完整运行2→4→6 GiB一次
→ verifier
→ 进程退出并恢复stock UVM、GPU 0 MiB
```

第一个 arm 完成清理后才会启动第二个 arm，因此模型权重、Graph、CPU shadow、KV 和页表都
不会跨 arm 复用。脚本没有重试循环；任一 arm 的 `returncode.txt` 非 0 都会保留失败产物并
终止汇总，不能把失败 arm 再跑一次后拼入同一个 qualification。

如果需要逐条执行而不是使用包装脚本，应保持完全相同的顺序：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e10-trigger-mode-comparison-REPRO-20260830-attempt-01
mkdir -p "$ROOT"

RUN_ROOT="$ROOT/auto_pressure" \
E10_TRIGGER_MODE=pressure \
ALLOW_GPU_RUN=1 \
bash experiments/vllm_uvm_e10_multilevel_pressure/run_experiment.sh qualify

RUN_ROOT="$ROOT/direct_tier" \
E10_TRIGGER_MODE=direct \
ALLOW_GPU_RUN=1 \
bash experiments/vllm_uvm_e10_multilevel_pressure/run_experiment.sh qualify

workloads/vllm/vllm/.venv/bin/python \
  experiments/vllm_uvm_e10_multilevel_pressure/compare_trigger_modes.py \
  "$ROOT" --output "$ROOT/comparison.json"
```

这种手动方式不会自动生成包装脚本的顶层 `protocol.json`，正式归档优先使用上一条
`run_trigger_comparison.sh qualify` 命令。

### 7.4 计时口径

输出中的 `comparison.json` 按三类数字分别比较：

1. `tier_transition`：完整切档本身。
2. `shared_tier_transaction_rpc`：两组共有的专家切档 RPC 与 KV resize RPC，用于排除 direct
   外层 API 包装差异。
3. `pure_prefill`：自动组扣除请求内切档后的 Prefill，与 direct 组请求内 Prefill。
4. `decode_to_prefill_resume`：切下一档前，把上一请求的 Decode 表恢复为 Prefill 表。
5. `prefill_to_decode`：当前请求首次 Decode 前切换为 CPU 读取表。
6. `steady_decode_median`：16 个 measured Decode step 的中位数。

`request_excluding_pre_request_transition` 的统计边界在两个 arm 中不同，不应直接用来判断机制
快慢；公平的总体代价是 `request_including_transition`。每 arm 只有一个样本，脚本给出的 10%
标志仅为描述性工程检查，不是统计等价检验。

auto 组的纯 Prefill 计算方式为：

```text
prefill_end_to_end
− activate_uvm_expert_graph_class RPC
− resize_uvm_kv_cache_target RPC
```

direct 组已经在请求前完成这两个 RPC，因此 `prefill_end_to_end` 就是其纯 Prefill。不能直接
比较 auto 的原始 Prefill E2E 与 direct 的原始 Prefill E2E，否则会把“切档在请求内还是请求
外”误写成计算性能差异。

### 7.5 离线复核

复核不会运行模型或重复 GPU 实验：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e10-trigger-mode-comparison-REPRO-20260830-attempt-01

E10_TRIGGER_MODE=pressure \
bash experiments/vllm_uvm_e10_multilevel_pressure/run_experiment.sh \
  verify "$ROOT/auto_pressure"

E10_TRIGGER_MODE=direct \
bash experiments/vllm_uvm_e10_multilevel_pressure/run_experiment.sh \
  verify "$ROOT/direct_tier"

bash experiments/vllm_uvm_e10_multilevel_pressure/run_trigger_comparison.sh \
  compare "$ROOT"

jq '{status,completed_sequences_per_arm,independent_cold_processes,
  statistical_repetitions,tiers}' "$ROOT/comparison.json"
```

### 7.6 对照验收条件

除第 6 节的单 arm 条件外，还必须满足：

- `auto_pressure/verification.json` 为 `trigger_mode=pressure`，4/6 GiB 请求中 backup 压力为
  4 blocks，decision 明确为 `backup_threshold_only`，prediction 为 0。
- `direct_tier/verification.json` 为 `trigger_mode=direct`，切档位置为
  `before_request_while_idle`；请求本身仍产生相同的 4-block backup 压力。
- 两组均只出现一次 cold70→cold80 和一次 cold80→cold89；均先切专家、再扩 KV。
- 两组三档 KV 物理位置、专家字节数、双表覆盖率和父入口操作数量一致。
- 两组输出 hash 完全相同，Graph capture 数不增加，内核日志无故障。
- 两个 arm 的返回码均为 0，并分别完成 stock UVM/0 MiB GPU 清理。

### 7.7 本次参考结果

已完成的单轮对照位于：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
e10-trigger-mode-comparison-20260830-attempt-01-qualification
```

其 `comparison.json` 为 `status=ok`，两个 arm 各一轮、统计重复 0。观察到的最大差异为：

```text
纯Prefill                 2.70%
Prefill→Decode            1.77%
steady Decode中位数       1.29%
相同核心切档事务           1.47%
包含完整切档的请求总时间    0.94%
```

完整解释见 `RESULTS_E10_PRESSURE_VS_DIRECT_TIER_QUALIFICATION_20260830.md`。这些百分比用于
复核同一轮产物，不应设为正式性能回归阈值；单样本只能支持 qualification 结论。

## 8. 2026-08-31当前同模块复验

实际artifact：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
e2-e3-e10-current-dual-view-REPRO-20260831-attempt-01/e10/
```

该轮和配套E2/E3使用相同custom UVM：

```text
SHA256     c8fe77c8ecc87f6d724d94ddf6535c934150d4e12bec384bd44562097aec043a
srcversion CD4A27464CD3E907C4EF493
```

运行命令：

```bash
REPRO_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e2-e3-e10-current-dual-view-REPRO-20260831-attempt-01

RUN_ROOT="$REPRO_ROOT/e10" \
E10_TRIGGER_MODE=pressure \
ALLOW_GPU_RUN=1 \
bash experiments/vllm_uvm_e10_multilevel_pressure/run_experiment.sh \
  qualify
```

上述目录已经存在；再次运行必须使用新的attempt目录。

| KV | 纯Prefill | Prefill E2E | P→D | steady Decode | 请求E2E |
|---:|---:|---:|---:|---:|---:|
| 2 GiB | 340.65 ms | 340.65 ms | 51.54 ms | 22.45 ms | 1030.31 ms |
| 4 GiB | 400.54 ms | 945.49 ms | 54.49 ms | 24.53 ms | 1683.42 ms |
| 6 GiB | 435.46 ms | 925.27 ms | 67.27 ms | 27.34 ms | 1739.66 ms |

完整压力切档为2→4 GiB 544.95 ms、4→6 GiB 489.81 ms。相对最近同协议参考，纯
Prefill最大绝对差异1.04%，请求E2E最大绝对差异1.15%，steady Decode最大差异2.40%。
4→6切档高4.49%，属于单轮内部阶段波动，没有形成核心计算或请求E2E的一致回退。

三档双页表覆盖率100%，CPU range-map fallback为0，输出hash均为：

```text
d9510d414d87cc5aaca79755e0e35473b443ec1d4fdbad30fdab7462093de739
```

`verification.json`为`status=ok`，Graph无重捕获，无Xid/MMU fault/内核错误；结束后
stock UVM恢复，GPU为0 MiB。本轮仅一个完整序列，不能作为统计显著性结论。
