# 复现 E13 双阶段 CUDA Graph 下的 K2/K6 与 Adaptive 回环

## 1. 实验回答的问题

本协议在相同模型、输入、batch 和执行机制下完成两个比较：

1. 比较固定 K2 与固定 K6 在 Short-128、B64/B128 下的 Prefill、P→D、稳态 Decode、
   TTFT、TPOT、E2E 和吞吐；
2. 将 Adaptive 依次置于初始 K2、增长后 K6、回缩后 K2，在排除档位事务和目标档
   warmup 后，验证三个稳态是否恢复到对应 Fixed 档的性能水平。

该实验使用完整的双阶段机制，而非 eager 近似：Prefill 与 Decode 都使用
`CUDAGraphMode.FULL_SEPARATE`，Prefill 使用 GPU alias 视图，Decode 使用 original-VA
remote 视图，两阶段通过 PARK/RESUME 快速切换。

当前权威成功轮为：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e13-short-b64-b128-full-separate-roundtrip-REPRO-20260902-attempt-02/
```

2026-09-02 的 `REPRO-...-attempt-01` 在 Adaptive 预捕获多档 Graph 时暴露出当前源码
对有效 big-PTE 下级映射的错误验收，返回134并触发fail-closed；该失败目录必须保留且
不得用来补表。修复后从三个arm完整重跑得到上述 `attempt-02`。此前
`20260901-attempt-03-source-validated` 仍保留为上一成功参考轮。

## 2. 冻结协议

### 2.1 模型、输入和请求

- 模型：`Qwen3-30B-A3B-FP8`，snapshot
  `d206ba732169f29bb77fbf80fc2c4b81d4d30782`；
- workload：Short，单请求 Prompt 128 token、生成 32 token；
- batch：B64、B128，所有请求必须在同一 batch 中同时运行，不允许拆 cohort；
- seed：`20260901`；
- attention backend：`TRITON_ATTN`；
- `max_num_batched_tokens=8192`，因此 B64 Prefill 为1个step，B128 Prefill为2个step；
- 每个 cell 先执行1次不计入结果的 warmup，再执行3次进程内测量，结果取中位数。

三个样本共享同一冷进程，不能当作三个独立冷进程重复，也不能据此计算置信区间。

### 2.2 档位与同步迁移

| 档位 | KV target blocks | GPU KV容量 | Plan Class | 专家卸载量 |
| --- | ---: | ---: | --- | ---: |
| K2 | 1,364 | 2.145 GB | cold70 | 15.854 GB |
| K4 | 2,730 | 4.294 GB | cold80 | 18.119 GB |
| K6 | 4,095 | 6.441 GB | cold89 | 20.158 GB |

每个 KV block 为 1,572,864 bytes、16 tokens。KV 和专家权重必须同步按相邻约2 GB
档移动；Adaptive 的实际路径固定为：

```text
K2 -> K4 -> K6 -> K4 -> K2
```

四条边分别迁移1,366、1,365、1,365、1,366个KV block，并同步切换对应专家
Plan Class。固定臂运行期不迁移：Fixed-K2只注册cold70的Graph，Fixed-K6只注册
cold89；Adaptive预捕获cold70/cold80/cold89三档Graph。

### 2.3 CUDA Graph 与 P→D 协议

- 有效编译配置必须为`mode=NONE`、`cudagraph_mode=FULL_SEPARATE`；
- Graph capture sizes固定为`64,128,8192`；
- Fixed每臂捕获3个Graph entry，Adaptive捕获3档×3 shape，共9个；
- `VLLM_UVM_ORIGINAL_VA_PHASE_MODE=prefill_graph_remote_decode`；
- original-VA双页表和`PARK/RESUME`必须启用；
- 每个B64正式样本必须有32个FULL Graph replay；
- 每个B128正式样本必须有33个FULL Graph replay；
- 正式计时期间禁止Graph重捕获、eager/none passthrough或档位漂移。

`P→D step`是首个Decode scheduler step，包含本batch的PARK和remote-map提交，不是
纯Decode kernel时间。`steady Decode`取最后16个单token Decode step的中位数。

## 3. 前置环境

权威轮硬件为单张NVIDIA GeForce RTX 5090（32,607 MiB），驱动`595.71.05`。复现时
需要仓库内的vLLM虚拟环境、模型snapshot、精确参数名、三档Plan Class和已编译的定制
`nvidia-uvm.ko`。

```bash
cd /home/ubuntu/nvidia-uvm-gpu

test -x workloads/vllm/vllm/.venv/bin/python
test -f /home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/\
snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782/config.json
test -f experiments/vllm_uvm_large_kv_multilevel/inputs/exact_parameter_names.txt
test -f experiments/vllm_uvm_e10_multilevel_pressure/inputs/\
formal_k2_k4_k6_k8_k10/cold70.json
test -f experiments/vllm_uvm_e10_multilevel_pressure/inputs/\
formal_k2_k4_k6_k8_k10/cold89.json
test -f experiments/eurosys_moe_offload_evaluation/e13/plan_classes_k2_k4_k6.json
test -f experiments/eurosys_moe_offload_evaluation/e13/plan_classes_graph_fixed_k2.json
test -f experiments/eurosys_moe_offload_evaluation/e13/plan_classes_graph_fixed_k6.json
test -f kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko
test -f workloads/vllm/vllm/uvm_test/uvm_allocator.so
```

如果模块或allocator不是当前源码构建产物，先编译：

```bash
make -C kernel-module/nvidia-module -j"$(nproc)" modules
make -C workloads/vllm/vllm/uvm_test
```

权威模块指纹为：

```text
srcversion  D158DB985980F4A580DB740
SHA-256    4168020f14fc3217f8d217e4fa8898299e4d81c0c18653c98cce9d73346afdbe
```

若源码有意更新，可以产生新指纹，但必须把新值保存在artifact并重新通过全部验收，不能
把新模块运行结果混入旧轮。核对当前待加载模块：

```bash
modinfo -F srcversion kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko
sha256sum kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko
```

开始前必须加载stock UVM、GPU为0 MiB、没有compute或UVM client：

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
sudo fuser -v /dev/nvidia-uvm /dev/nvidia-uvm-tools 2>/dev/null || true
modinfo nvidia_uvm | rg 'filename|version|srcversion'
```

不要与其他GPU任务并发。runner会保存stock模块快照，临时加载定制模块，并在三个arm
结束后恢复stock模块。

## 4. 静态预检

先运行不会启动正式GPU矩阵的检查：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  static-preflight
```

该入口检查必需文件、96个精确参数名、shell/Python语法、E13工具测试，以及vLLM中的
autoscale、next-request、Plan Class和同步切档单元测试。输出必须以
`E13 static preflight: OK`结束。

## 5. 运行一次正式矩阵

`RUN_ROOT`必须是不存在的新目录。不要设置`E13_GRAPH_ARMS`；该变量是故障诊断过滤器，
设置后不会执行或验证完整三臂矩阵。

```bash
cd /home/ubuntu/nvidia-uvm-gpu

RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e13-short-b64-b128-full-separate-roundtrip-REPRO-<tag>-attempt-01 \
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/e13/run_e13_kv_adaptive_matrix.sh \
  short-graph-roundtrip-once
```

顶层顺序固定为`fixed_k2 -> fixed_k6 -> roundtrip`，每个arm是独立冷进程。Adaptive
内部顺序为初始K2的B64/B128、扩到K6后的B64/B128、回到K2后的B64/B128。四个档位
事务及每次到达目标档后的1次warmup均在计时窗外，正式结果不含档位迁移时间。

失败或中断的attempt必须原样保留；修复后使用新的`attempt-02`目录，不得覆盖旧目录，
也不得只重跑产生有利结果的arm。

## 6. 自动产物

成功目录至少包含：

```text
config.snapshot.json
execution_status.tsv
git_status.txt
input_sha256.txt
module.before.json
module.custom.json
module.after.json
fixed_k2/{command.txt,environment.txt,result.json,returncode.txt,uvm.log,...}
fixed_k6/{command.txt,environment.txt,result.json,returncode.txt,uvm.log,...}
roundtrip/{command.txt,environment.txt,result.json,returncode.txt,uvm.log,...}
verification.json
summary.md
verify.log
verify.err
```

`input_sha256.txt`冻结配置、runner、verifier、模型配置、参数名、placement计划、Plan
Class和模块二进制。`execution_status.tsv`的三个returncode都必须为0。

## 7. 验证与结果提取

顶层命令会自动运行verifier。查看核心状态：

```bash
ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/<new-root>

cat "$ROOT/execution_status.tsv"
jq '{status,check_count:(.checks|length),passed:([.checks[]|select(.ok)]|length),
  fixed_comparisons,
  outside_band:[.equivalence[]|select(.within_10pct==false)]}' \
  "$ROOT/verification.json"
cat "$ROOT/summary.md"
```

如需独立复核，应写入新文件；verifier为防止污染证据而拒绝覆盖已有文件。由于verifier
还会生成`summary.md`，最安全的离线复核方式是复制完整artifact到新的审计目录后执行：

```bash
AUDIT_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e13-short-b64-b128-full-separate-roundtrip-AUDIT-<tag>
cp -a "$ROOT" "$AUDIT_ROOT"
rm "$AUDIT_ROOT/verification.json" "$AUDIT_ROOT/summary.md"
workloads/vllm/vllm/.venv/bin/python \
  experiments/eurosys_moe_offload_evaluation/e13/verify_e13_short_graph_roundtrip.py \
  "$AUDIT_ROOT" --output "$AUDIT_ROOT/verification.json"
```

验收必须同时满足：

- 三个arm均返回0且`verification.status=ok`；
- 84/84功能和协议检查通过；
- Fixed每臂3个、Adaptive 9个Graph entry完整捕获；
- B64每样本32/32、B128每样本33/33正式step均为一次FULL replay；
- 计时期间无Graph capture、eager fallback、档位或Plan Class漂移；
- 回环严格为`1364 -> 2730 -> 4095 -> 2730 -> 1364` blocks；
- 每个请求周期执行alias Prefill恢复和remote Decode cutover，并结束于
  `remote_parked`；
- Fixed/Adaptive对应cell输出token hash一致；
- 无Xid、PTE/PDE、MMU fault或illegal address；
- 最终GPU为0 MiB、没有UVM client并恢复原stock模块。

性能等价单独判断：核心指标定义为req/s、Prefill、steady Decode和E2E。三个状态×两个
batch×四指标共24个A/F倍率均须落在`[0.90, 1.10]`。总字段
`all_equivalence_metrics_within_10pct`还包含output tok/s、batch wall、P→D、TTFT和
TPOT，共54点；它为false不等于功能verifier失败，必须列出具体超带点。

## 8. 当前权威结果

### 8.1 Fixed-K2与Fixed-K6

| B | 档位 | req/s | TTFT | TPOT | E2E | Prefill | P→D | steady Decode |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | Fixed-K2 | 29.1489 | 726.68 ms | 46.59 ms | 2189.23 ms | 722.16 ms | 489.74 ms | 24.89 ms |
| 64 | Fixed-K6 | 25.3277 | 768.05 ms | 56.65 ms | 2519.78 ms | 763.26 ms | 470.04 ms | 32.67 ms |
| 128 | Fixed-K2 | 39.6909 | 972.30 ms | 72.17 ms | 3212.92 ms | 1208.64 ms | 492.87 ms | 41.12 ms |
| 128 | Fixed-K6 | 32.5717 | 1066.51 ms | 91.25 ms | 3917.92 ms | 1344.89 ms | 501.92 ms | 59.54 ms |

K2的吞吐在B64/B128分别高15.09%/21.86%，E2E分别低13.12%/17.99%。这两个Short
场景都没有触及K2的KV容量上限，因此K6增加KV容量没有带来收益，反而因卸载更多专家而
增加Prefill和remote Decode开销。该结论不能外推到K2容量不足的长上下文或更高KV压力。

### 8.2 Adaptive相对对应Fixed

表内延迟倍率为`Adaptive/Fixed`；req/s倍率同样为`Adaptive/Fixed`，越接近1越等价。

| Adaptive状态 | B | req/s | Prefill | steady Decode | E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| 初始K2 | 64 | 0.988× | 1.005× | 0.982× | 1.012× |
| 增长后K6 | 64 | 0.965× | 1.010× | 1.019× | 1.037× |
| 回缩后K2 | 64 | 0.988× | 1.001× | 1.003× | 1.012× |
| 初始K2 | 128 | 0.979× | 1.004× | 1.001× | 1.021× |
| 增长后K6 | 128 | 0.971× | 1.004× | 1.020× | 1.030× |
| 回缩后K2 | 128 | 0.968× | 1.007× | 1.001× | 1.034× |

核心24/24点均在±10%带内，所以排除档位事务和warmup后，Adaptive在K2、K6以及回到
K2后的稳态均恢复到对应Fixed水平，回缩没有留下稳态性能回退。

### 8.3 P→D限制

把9类指标全部纳入后只有49/54在带内，5个超带点全部是P→D：

| 状态 | B | Adaptive/Fixed P→D |
| --- | ---: | ---: |
| 增长后K6 | 64 | 1.142× |
| 回缩后K2 | 64 | 1.109× |
| 初始K2 | 128 | 1.117× |
| 增长后K6 | 128 | 1.240× |
| 回缩后K2 | 128 | 1.138× |

这表示Adaptive进程中某些batch的首个Decode step比对应Fixed高10.92%–24.03%。这里的
P→D是每个请求batch自身的Prefill→Decode视图切换，和已排除计时的K2↔K6档位迁移不是
同一事件。初始K2/B128在任何档位迁移前也超带，因此当前数据不能把差异唯一归因于增长
或回缩历史，也不能声称P→D瞬时停顿完全等价；但后续steady Decode与整体核心24点通过。

四个被排除的档位事务耗时依次为7.272、6.992、7.671、6.800秒。它们不应计入上面的
稳态等价结论，也不能因此声称Adaptive包含切换成本后的端到端性能与Fixed等价。

## 9. 证据范围

权威轮3/3冷进程成功；正式测量2,880个请求，加上排除warmup共3,840/3,840请求完成。
975/975个正式serving step均为FULL Graph replay，运行时无重捕获或eager fallback；
84/84检查通过，输出hash一致，无GPU/UVM故障，最终恢复stock UVM。

这是一轮qualification。它支持“切档后的稳态路径恢复对应Fixed水平”，不支持统计显著性、
切换事务零开销、P→D停顿完全等价，或Adaptive在所有KV压力下优于Fixed。

## 10. 2026-09-02复现核验与修复记录

严格复制第4、5节命令执行。`REPRO-20260902-attempt-01`的Fixed-K2和Fixed-K6均返回0，
Adaptive在预捕获cold80时失败。驱动日志表明两个半block range已经合法映射为active
big-PTE下级页表，但`uvm_va_block_premap_readmostly_remote_read()`的实验后置检查只接受
active 4K table，错误返回`NV_ERR_INVALID_STATE`；部分cutover随后进入回滚并触发Xid 31。

修复位于`kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c`：后置检查
现在按映射页逐项验证，big-PTE覆盖页要求active big table，其余页要求active 4K table，
同时继续拒绝2 MiB PTE或覆盖不完整状态。重新编译模块并再次通过7项E13工具测试和31项
vLLM相关测试后，从三个arm完整重跑为`REPRO-20260902-attempt-02`。

新轮84/84门通过，核心24/24等价点通过；相对上一成功参考轮，10个cell的吞吐、Prefill、
steady Decode和E2E最大变化4.30%，因此结果基本一致。失败attempt和成功attempt必须同时
保留，不能删除失败证据或用成功臂拼接两个目录。
