# 复现E2 Ours每层多卸载一个专家后的近等量单轮对照

日期：2026-09-03（Asia/Shanghai）

## 1. 最新成功证据

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e2-near-equal-offload-plus-one-expert-20260903-attempt-01/
```

该轮16/16个正式cell成功，聚合`verification.json`为`status=ok`，83/83项检查通过。
每个cell只有一个冷进程样本，不得写成median、置信区间或重复实验。

2026-09-03后续尝试让Native的Prefill也使用`FULL_SEPARATE`，但Long/C1在128-token和
64-token两种Graph块下均于第一次真实Prefill replay触发CUDA非法访问。因此最新可用、
可完整复现的协议仍是本文记录的Native eager Prefill加`FULL_DECODE_ONLY`；不得把失败的
双Graph尝试混入本表。详细边界见
[RESULTS_E2_NATIVE_UVA_FULL_SEPARATE_FEASIBILITY_20260903.md](../eurosys_moe_offload_evaluation/RESULTS_E2_NATIVE_UVA_FULL_SEPARATE_FEASIBILITY_20260903.md)。

## 2. 固定坐标

| 场景组 | Ours计划 | Ours实际 | Native实际 |
|---|---:|---:|---:|
| C1全部、Short/C16 | cold2 | 0.838 GiB | 0.896 GiB |
| Ordinary/Long C16 | cold6 | 2.514 GiB | 2.330 GiB |
| RAG/C16 | cold20 | 8.379 GiB | 8.522 GiB |

8个cell的两臂实际卸载量相差均不超过10%。Ours选择跨层冷专家切片，Native选择注册顺序
whole-Parameter前缀；这不是相同专家ID或相同参数名的单变量实验。

其他固定条件：DeepSeek-V2-Lite-Chat BF16 snapshot、`eurosys-pa-v2-20260824`冻结请求、
TRITON_MLA、C1/C16、种子20260823。Native为eager Prefill加`FULL_DECODE_ONLY`；Ours为
`FULL_SEPARATE`加完整双页表P→D。

## 3. 入口文件

```text
experiments/eurosys_moe_offload_evaluation/
  run_e2_near_equal_offload_plus_one_expert.sh
  run_e2_equal_offload_native_uva_graph.sh
  verify_e2_equal_offload_native_uva_graph.py
  configs/e2_c1_near_equal_offload_plus_one_expert.json
  configs/e2_c16_near_equal_offload_plus_one_expert.json
```

## 4. 运行前与静态检查

```bash
cd /home/ubuntu/nvidia-uvm-gpu

nvidia-smi --query-gpu=index,name,memory.used,memory.free \
  --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits
sudo fuser -v /dev/nvidia-uvm /dev/nvidia-uvm-tools 2>/dev/null || true
printf 'loaded=%s\nstock=%s\n' \
  "$(tr -d '[:space:]' </sys/module/nvidia_uvm/srcversion)" \
  "$(modinfo -F srcversion nvidia-uvm | tr -d '[:space:]')"

bash -n \
  experiments/eurosys_moe_offload_evaluation/run_e2_near_equal_offload_plus_one_expert.sh \
  experiments/eurosys_moe_offload_evaluation/run_e2_equal_offload_native_uva_graph.sh \
  experiments/eurosys_moe_offload_evaluation/run_c1_mechanism_runtime_hot_audit_off_matrix.sh \
  experiments/eurosys_moe_offload_evaluation/run_e2_c16_direct_batch_matrix.sh

workloads/vllm/vllm/.venv/bin/python -m py_compile \
  experiments/eurosys_moe_offload_evaluation/verify_e2_equal_offload_native_uva_graph.py

jq empty \
  experiments/eurosys_moe_offload_evaluation/configs/e2_c1_near_equal_offload_plus_one_expert.json \
  experiments/eurosys_moe_offload_evaluation/configs/e2_c16_near_equal_offload_plus_one_expert.json
```

要求GPU为0 MiB、无未知compute/UVM client，并且loaded与stock srcversion一致。

## 5. 唯一一轮GPU命令

输出目录必须不存在：

```bash
cd /home/ubuntu/nvidia-uvm-gpu

RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e2-near-equal-offload-plus-one-expert-$(date -u +%Y%m%dT%H%M%SZ)

ALLOW_GPU_RUN=1 bash \
  experiments/eurosys_moe_offload_evaluation/run_e2_near_equal_offload_plus_one_expert.sh \
  "$RUN_ROOT"
```

入口先运行C1的Native四点和Ours四点，再运行C16的相同八点；Ours阶段自动加载custom
UVM并在结束后恢复stock。`MATRIX_SINGLE_ATTEMPT=1`禁止为同一cell追加正式重跑。

## 6. 验收

```bash
jq -e '
  .status == "ok" and
  .failed_checks == [] and
  ([.checks[]] | all) and
  (.cells | length == 8)
' "$RUN_ROOT/verification.json"

jq -r '.cells[] |
  [.concurrency,.workload,
   (.ours.manifest.realized_bytes/1073741824),
   (.native.manifest.realized_bytes/1073741824),
   .ours.metrics.tpot_ms,.native.metrics.tpot_ms,
   .native_over_ours.tpot_ms,.outputs_match] | @tsv
' "$RUN_ROOT/summary.json" | column -t

! rg -n -i \
  'Xid [0-9]|illegal memory access|FAULT_PDE|FAULT_PTE|GPU page fault|kernel BUG' \
  "$RUN_ROOT/kernel.since.log"

nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
```

RAG两点`outputs_match=false`延续既有跨机制数值路径差异，不等于单臂运行失败；其余6点应
完整输出一致。最终还必须确认GPU为0 MiB并恢复stock UVM。

## 7. 当前结果

Ours的TPOT在8/8点低于Native，优势为1.33×--3.42×；6个完整输出一致点为
1.33×--2.02×。完整结果见：

```text
experiments/eurosys_moe_offload_evaluation/
  RESULTS_E2_NEAR_EQUAL_OFFLOAD_PLUS_ONE_EXPERT_20260903.md
```

## 8. Native Prefill Graph失败复核

以下目录只用于失败诊断，不是新的正式结果：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e2-near-equal-plus-one-full-separate-graph128-20260903-attempt-01/
  e2-native-prefill-graph64-feasibility-long-c1-20260903-attempt-01/
```

128-token轮只有Native C1前三点成功，Long/C1失败；64-token轮只验证Long/C1并得到相同
故障。没有C16、Ours或聚合成功verifier，不能从中报告系统性能排名。当前入口脚本和profile
已恢复为第5节所示的最近成功协议。
