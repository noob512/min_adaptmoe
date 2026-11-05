# 复现E2 Ours与Native UVA提高卸载量后的单轮对照

日期：2026-09-03（Asia/Shanghai）

> 该文档保留cold1/5/19与Native whole-Parameter前缀的首轮提高卸载量证据。实际字节
> 配平更紧的最新cold2/6/20轮见
> [REPRODUCE_E2_NEAR_EQUAL_OFFLOAD_PLUS_ONE_EXPERT.md](REPRODUCE_E2_NEAR_EQUAL_OFFLOAD_PLUS_ONE_EXPERT.md)。

## 1. 目的与最新证据

本实验比较Ours和原生vLLM pinned Native UVA，并把Native请求卸载预算提高到Ours实际
卸载字节。两臂保持各自放置策略，不要求相同专家ID：Ours选择跨层冷专家切片，Native选择
模型注册顺序的whole-Parameter前缀。

最新成功证据：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e2-equal-offload-native-uva-graph-20260903-attempt-01/
```

16/16个正式cell完成，最终`verification.json`为`status=ok`。每个cell只有一个冷进程
样本，不得写成median、置信区间或重复实验。

## 2. 配平定义

Ours的正式坐标为：

| 场景组 | Ours计划 | Ours实际字节 | Native请求GiB |
|---|---:|---:|---:|
| C1全部、Short/C16 | cold1 | 449,839,104 | 0.4189453125 |
| Ordinary/Long C16 | cold5 | 2,249,195,520 | 2.0947265625 |
| RAG/C16 | cold19 | 8,546,942,976 | 7.9599609375 |

原生Native无法拆分Parameter，最后一个Parameter可以跨过预算。因此验收必须同时检查：

```text
Native requested_bytes == Ours realized_bytes
Native previous_prefix_bytes < Ours realized_bytes <= Native realized_bytes
```

即Native使用“不低于Ours的最小whole-Parameter前缀”。不要只根据`--cpu-offload-gb`
声称实际字节严格相等。当前实际Native档为0.896、2.330和8.522 GiB。

## 3. 固定机制

- 模型：DeepSeek-V2-Lite-Chat BF16固定snapshot；
- 请求：`eurosys-pa-v2-20260824`冻结Short/Ordinary/RAG/Long；
- C1和C16各四点，随机种子20260823；
- Native：TRITON_MLA、eager Prefill、`FULL_DECODE_ONLY` Decode Graph；
- Ours：TRITON_MLA、`FULL_SEPARATE` Prefill/Decode Graph、完整双页表、P→D快速切换；
- C16无HTTP，16个请求在第一个engine step前全部调用`add_request`；
- 每个cell独立冷进程，只运行一遍。

## 4. 入口

```text
experiments/eurosys_moe_offload_evaluation/
  run_e2_equal_offload_native_uva_graph.sh
  verify_e2_equal_offload_native_uva_graph.py
  configs/e2_c1_equal_offload_native_uva_graph.json
  configs/e2_c16_equal_offload_native_uva_graph.json
```

## 5. 运行前检查

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
```

要求GPU为0 MiB、没有未知compute/UVM client，且loaded与stock srcversion相同。

## 6. 静态检查

```bash
bash -n \
  experiments/eurosys_moe_offload_evaluation/run_e2_equal_offload_native_uva_graph.sh \
  experiments/eurosys_moe_offload_evaluation/run_c1_mechanism_runtime_hot_audit_off_matrix.sh \
  experiments/eurosys_moe_offload_evaluation/run_e2_c16_direct_batch_matrix.sh

workloads/vllm/vllm/.venv/bin/python -m py_compile \
  experiments/eurosys_moe_offload_evaluation/verify_e2_equal_offload_native_uva_graph.py

jq empty \
  experiments/eurosys_moe_offload_evaluation/configs/e2_c1_equal_offload_native_uva_graph.json \
  experiments/eurosys_moe_offload_evaluation/configs/e2_c16_equal_offload_native_uva_graph.json
```

## 7. 单轮GPU运行

输出目录必须不存在：

```bash
cd /home/ubuntu/nvidia-uvm-gpu

RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e2-equal-offload-native-uva-graph-$(date -u +%Y%m%dT%H%M%SZ)

ALLOW_GPU_RUN=1 bash \
  experiments/eurosys_moe_offload_evaluation/run_e2_equal_offload_native_uva_graph.sh \
  "$RUN_ROOT"
```

runner先执行C1的Native四点和Ours四点，恢复stock UVM后再按相同顺序执行C16。Ours阶段
自动加载custom UVM并在结束后恢复stock。warmup只用于建立运行时状态，不是正式重复样本。

## 8. 验收

```bash
jq -e '
  .status == "ok" and
  .failed_checks == [] and
  ([.checks[]] | all) and
  (.cells | length == 8)
' "$RUN_ROOT/verification.json"

jq -r '.cells[] |
  [.concurrency,.workload,.offload.target_bytes,
   .offload.native_realized_bytes,
   .offload.native_realized_over_target_ratio,
   .native_over_ours.tpot_ms,.outputs_match] | @tsv
' "$RUN_ROOT/summary.json" | column -t

! rg -n -i \
  'Xid [0-9]|illegal memory access|FAULT_PDE|FAULT_PTE|GPU page fault|kernel BUG' \
  "$RUN_ROOT/kernel.since.log"

nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
```

`outputs_match=false`不等于单臂运行失败；当前RAG两点延续既有的跨机制token hash差异，
必须标为描述性诊断。其他6点要求完整输出一致。最终还必须确认GPU为0 MiB并恢复stock UVM。

## 9. 当前单轮结果

Native/Ours TPOT为1.55×--3.59×，即Ours的稳态Decode在8/8点更低；6个完整输出一致点
为1.55×--2.25×。完整数值见：

```text
experiments/eurosys_moe_offload_evaluation/
  RESULTS_E2_EQUAL_OFFLOAD_NATIVE_UVA_GRAPH_20260903.md
```

不得把该结果写成严格equal-realized-byte或机制单变量消融；Native实际字节仍因
whole-Parameter粒度比Ours高7.1%--114.0%，且卸载对象不同。
