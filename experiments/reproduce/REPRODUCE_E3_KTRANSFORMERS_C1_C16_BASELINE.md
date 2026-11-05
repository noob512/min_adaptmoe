# 复现E3 KTransformers B1/B16同请求基线

最后核验：2026-09-02（Asia/Shanghai）

## 1. 实验范围

本实验把最新D/ours的10个E3坐标原样送入KTransformers：

```text
Short     128→32    B1/B16
Ordinary  512→128   B1/B16
Medium    2048→128  B1/B16
RAG       8192→128  B1/B16
Long      512→512   B1/B16
```

它冻结模型snapshot、fixture token IDs、输出长度、batch、TRITON attention和逐cell专家
坐标。KTransformers启用原生Decode CUDA Graph，但没有D的Prefill Graph、双页表或
P→D切换。本实验用于系统级描述性比较，不是单机制消融。

## 2. 环境和输入

```text
repo   /home/ubuntu/nvidia-uvm-gpu
GPU    NVIDIA GeForce RTX 5090 32 GiB，GPU 0，TP=1
CPU    固定CPU 0--13，共14个逻辑CPU，单NUMA节点
model  /home/ubuntu/.cache/huggingface/hub/
       models--Qwen--Qwen3-30B-A3B-FP8/snapshots/
       d206ba732169f29bb77fbf80fc2c4b81d4d30782
```

输入和计划：

```text
experiments/eurosys_moe_offload_evaluation/inputs/generated/
  eurosys-pa-v2-20260824/qwen3_30b_a3b_fp8/evaluation_*.json
experiments/qwen_decode_six_mechanism_compare/inputs/cold_decode_plan_2g.json
experiments/qwen_decode_six_mechanism_compare/inputs/cold_decode_plan_8g.json
experiments/vllm_uvm_e10_multilevel_pressure/inputs/cold89.json
```

placement为：

| 坐标 | 档位 | 每层GPU/CPU专家 |
|---|---|---:|
| Short B1/B16、Ordinary B1、Medium B1、Long B1 | cold10 | 118/10 |
| Ordinary B16、Medium B16、RAG B1、Long B16 | cold38 | 90/38 |
| RAG B16 | cold89 | 39/89 |

runner会把D的冷专家计划转换为KTransformers frequency mask，并验证每层exact top-k专家
坐标相同。CPU参数固定为：

```text
kt_cpuinfer=14
OMP_NUM_THREADS=14
MKL_NUM_THREADS=14
taskset -c 0-13
kt_threadpool_count=1
kt_numa_nodes=[0]
```

## 3. 预检查

```bash
cd /home/ubuntu/nvidia-uvm-gpu

test -x KTransformers/.venv/bin/python
test -f experiments/eurosys_moe_offload_evaluation/\
direct_batch_e3_ktransformers_runner.py
test -f experiments/eurosys_moe_offload_evaluation/\
verify_e3_ktransformers_c1_c16.py

bash -n experiments/eurosys_moe_offload_evaluation/\
run_e3_ktransformers_c1_c16_baseline.sh
KTransformers/.venv/bin/python -m py_compile \
  experiments/eurosys_moe_offload_evaluation/\
direct_batch_e3_ktransformers_runner.py \
  experiments/eurosys_moe_offload_evaluation/\
verify_e3_ktransformers_c1_c16.py

nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits
```

要求开始前GPU为0 MiB且没有compute process。KTransformers使用stock驱动路径，不加载
custom UVM。

## 4. 完整单轮运行

输出目录必须不存在：

```bash
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e3-ktransformers-c1-c16-$(date -u +%Y%m%dT%H%M%SZ)

ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/\
run_e3_ktransformers_c1_c16_baseline.sh "$RUN_ROOT"
```

脚本按10个独立冷进程串行运行。每个cell只调用一次`Engine.generate`提交整个batch，关闭
mixed chunk并让Prefill budget覆盖完整batch。B1进程捕获B1 Decode Graph，B16进程捕获
B1/B16；Prefill仍为eager。RAG/B16的完整Prefill可能超过默认300秒，因此runner明确设置
`watchdog_timeout=7200`；外层进程超时同样为7,200秒。放宽watchdog不改变模型、请求、
placement或计时区间。

完整运行成功时，脚本自动生成`$RUN_ROOT/verification.json`。验收：

```bash
jq -e '
  .status == "ok" and
  .completed_cells == 10 and
  .expected_cells == 10 and
  (.errors | length) == 0 and
  .protocol.same_frozen_prompt_token_ids == true and
  .protocol.same_exact_expert_coordinates_per_cell == true and
  .protocol.ktransformers_cuda_graph == true
' "$RUN_ROOT/verification.json"

awk -F '\t' 'NR > 1 && ($8 != "ok" || $9 != 0) {bad=1} END {exit bad}' \
  "$RUN_ROOT/execution.tsv"

! find "$RUN_ROOT/cases" -name kernel.since.log -print0 | xargs -0 \
  rg -n -i 'Xid|MMU fault|FAULT_PDE|FAULT_PTE|GPU page fault|kernel BUG'

nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
```

## 5. 2026-09-02权威证据复核

当前单轮10个有效点保存在：

```bash
BASE=experiments/eurosys_moe_offload_evaluation/artifacts/\
e3-ktransformers-c1-c16-cudagraph-20260902-attempt-01
D_REF=experiments/eurosys_moe_offload_evaluation/artifacts/\
e3-c1-c16-full-dual-20260901-reproduce-all-attempt-02
```

不重新运行GPU，只复核该单轮证据：

```bash
KTransformers/.venv/bin/python \
  experiments/eurosys_moe_offload_evaluation/\
verify_e3_ktransformers_c1_c16.py \
  --root "$BASE" \
  --d-reference "$D_REF" \
  --output "$BASE/verification.local-audit.json"
```

不要覆盖已经发布的`verification.json`。复核期望为10/10 cell通过、Graph协议门为true、
完整输出hash与D为0/10一致、要求cell内所有请求首token一致时为3/10。

## 6. 结果和解释边界

完整结果见：

```text
experiments/eurosys_moe_offload_evaluation/
  RESULTS_E3_KTRANSFORMERS_C1_C16_20260902.md
```

复核时应得到下列逐阶段绝对值；单位均为ms。`D P→D`没有KTransformers对应项，倍率为
`KTransformers/D`的Prefill/Decode/E2E：

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

KTransformers相对D的观测延迟比分别为：Prefill 4.18×--143.12×、steady Decode
6.20×--42.54×、E2E 7.50×--49.62×。跨框架完整输出hash不一致，因此只能报告为相同
冻结输入坐标的数值诊断；不得写成exact-generation排名。两边均启用各自Graph，但不能
把倍率单独归因于CPU专家、remote-read、双阶段Graph、页表或调度器。

D在10/10 cell的Prefill、steady Decode和E2E绝对值均更低。其6.41--63.68 ms的P→D
开销已包含在D E2E内，只占D E2E的0.07%--1.45%。没有同定义D客户端TTFT，因此不得用
KT TTFT除以D Prefill构造伪TTFT倍率。

本轮每个cell只有一个成功冷进程样本，没有均值、波动范围或置信区间。若调整CPU核心数、
NUMA、关闭CUDA Graph、attention backend、mixed chunk、专家数量或batch拆分，必须建立
新协议，不能回填本表。
