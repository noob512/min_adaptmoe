# 复现E3近等量卸载下Graph-enabled Native copy与Ours对照

日期：2026-09-03（Asia/Shanghai）

## 1. 最近一次且唯一的正式轮

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e3-near-equal-native-copy-graph-20260903-attempt-01/
```

本轮Native copy为9/10成功，Ours为10/10成功。Native copy唯一未完成点是RAG/B16：在
完整131,072-token pure-Prefill Graph捕获时OOM，应记为容量N/A。未进行第二轮或定向
重跑。

严格verifier保留`status=error`，209/211项检查为true；两个false项只对应Native launcher
非零和缺少RAG/B16运行结果。这一状态用于防止把9/10矩阵误报为全面通过。

## 2. 冻结协议

```text
模型       Qwen3-30B-A3B-FP8
snapshot   d206ba732169f29bb77fbf80fc2c4b81d4d30782
输入       eurosys-pa-v2-20260824冻结请求
场景       Short/Ordinary/Medium/RAG/Long
batch      B1、B16
attention  TRITON_ATTN
Graph      两臂均为FULL_SEPARATE
Prefill    pure-Prefill FULL Graph，完整batch，不启用chunked Prefill
Decode     uniform-Decode FULL Graph
重复       每个cell一个独立冷进程、一次measurement
```

卸载映射：

| 场景组 | Ours | Ours实际 | Native请求 | Native实际 |
|---|---:|---:|---:|---:|
| Short B1/B16；Ordinary、Medium、Long B1 | cold11 | 2.320 GiB | 2.25 GiB | 2.322 GiB |
| Ordinary、Medium、Long B16；RAG B1 | cold39 | 8.227 GiB | 7.875 GiB | 7.942 GiB |
| RAG B16 | cold90 | 18.984 GiB | 20.25 GiB | 20.324 GiB |

实际卸载字节的对称比值为1.0008×、1.0358×和1.0706×，均不超过1.10×。Native按
whole-Parameter前缀卸载；Ours按跨层冷专家切片卸载，不要求相同参数ID。

## 3. 实现入口

```text
experiments/eurosys_moe_offload_evaluation/
  run_e3_near_equal_native_copy_graph.sh
  prepare_e3_near_equal_offload_plans.py
  run_e3_native_copy_vs_uva_phase.sh
  run_e3_c1_c16_uva_single_round.sh
  verify_e3_near_equal_native_copy_graph.py
```

入口先校验冻结路由并生成cold11/39/90计划，顺序运行Native copy十点和Ours十点，最后
聚合实际卸载字节、Graph运行时dispatch、双页表、输出hash、故障和清理状态。Ours按cell
加载custom UVM，结束后恢复stock UVM。

## 4. 运行前检查

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

要求GPU为0 MiB、没有compute process或未知UVM client，且loaded与stock srcversion一致。

## 5. 静态检查

```bash
cd /home/ubuntu/nvidia-uvm-gpu

bash -n \
  experiments/eurosys_moe_offload_evaluation/run_e3_near_equal_native_copy_graph.sh \
  experiments/eurosys_moe_offload_evaluation/run_e3_native_copy_vs_uva_phase.sh \
  experiments/eurosys_moe_offload_evaluation/run_e3_c1_c16_uva_single_round.sh \
  experiments/eurosys_moe_offload_evaluation/run_formal_e3.sh

workloads/vllm/vllm/.venv/bin/python -m py_compile \
  experiments/eurosys_moe_offload_evaluation/prepare_e3_near_equal_offload_plans.py \
  experiments/eurosys_moe_offload_evaluation/verify_e3_near_equal_native_copy_graph.py \
  experiments/vllm_p3_pinned_uva_phase_compare/benchmark_phase_steps.py \
  experiments/vllm_uvm_prefill_alias_decode_remote_graph/benchmark_phase_switch_steps.py

sha256sum \
  experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/routing/expert_distribution.pt
```

冻结路由源的期望SHA256为：

```text
912732708a46291b73e70c93291758747f285b364437ce77f58d53501d072f81
```

## 6. 单轮GPU命令

输出目录必须不存在；每次复现只运行一次，不得覆盖artifact或自动追加attempt：

```bash
cd /home/ubuntu/nvidia-uvm-gpu

RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e3-near-equal-native-copy-graph-$(date -u +%Y%m%dT%H%M%SZ)

set +e
ALLOW_GPU_RUN=1 bash \
  experiments/eurosys_moe_offload_evaluation/run_e3_near_equal_native_copy_graph.sh \
  "$RUN_ROOT"
runner_rc=$?
set -e
printf 'runner_rc=%s\n' "$runner_rc"
```

在32-GiB RTX 5090和当前冻结协议下，RAG/B16的Native copy预计在Prefill Graph捕获处
达到容量边界，因此总入口可能返回1。不得为得到返回0而缩小输入、拆batch、启用chunked
Prefill或补跑第二次。应按下一节区分9个有效配对点与1个容量N/A。

## 7. 验收

```bash
# 只允许两个由RAG/B16容量N/A产生的strict失败项。
jq -e '
  .status == "error" and
  (.cells | length == 9) and
  (.failed_checks == [
    "launchers_succeeded",
    "rag_summary_b16_cell_processed"
  ]) and
  (.errors | keys == ["rag_summary_b16"]) and
  ([.cells[].output_hash_match] | all) and
  ([.cells[].offload.symmetric_realized_ratio <= 1.10] | all)
' "$RUN_ROOT/verification.json"

# Native只有RAG/B16为error；其余9点必须为ok。
awk -F '\t' '
  NR == 1 {next}
  $1 == "rag_summary" && $4 == 16 {rag=($8=="error" && $9==1); next}
  $8 != "ok" || $9 != 0 {bad=1}
  END {exit !(rag && !bad)}
' "$RUN_ROOT/native_copy/execution.tsv"

# Ours十点均必须返回0。
awk -F '\t' '
  NR > 1 && $1 == "ours" && $6 != 0 {bad=1}
  END {exit bad}
' "$RUN_ROOT/ours/execution.tsv"

# Native RAG/B16必须是明确的Prefill Graph捕获容量失败。
rg -n 'Capturing CUDA graphs \(pure prefill, FULL\)' \
  "$RUN_ROOT/native_copy/cases/rag_summary_b16_copy/run.log"
rg -n 'Tried to allocate 1024.00 MiB.*942.75 MiB is free' \
  "$RUN_ROOT/native_copy/cases/rag_summary_b16_copy/run.log"

# 无内核故障、GPU清空并恢复stock UVM。
! rg -n -i \
  'Xid [0-9]|MMU fault|PDE fault|PTE fault|illegal memory access|kernel BUG' \
  "$RUN_ROOT/kernel.since.log"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
printf 'loaded=%s\nstock=%s\n' \
  "$(tr -d '[:space:]' </sys/module/nvidia_uvm/srcversion)" \
  "$(modinfo -F srcversion nvidia-uvm | tr -d '[:space:]')"
```

有效配对点的结果表：

```bash
jq -r '.cells[] |
  [.workload,.batch,.offload.ours_gib,.offload.native_copy_gib,
   .ours.metrics.prefill_ms,.native_copy.metrics.prefill_ms,
   .ours.metrics.decode_ms,.native_copy.metrics.decode_ms,
   .ours.metrics.e2e_ms,.native_copy.metrics.e2e_ms,
   .native_copy_over_ours.prefill_ms,
   .native_copy_over_ours.decode_ms,
   .native_copy_over_ours.e2e_ms,.ours.metrics.cutover_ms] | @tsv
' "$RUN_ROOT/summary.json" | column -t
```

## 8. 结果与边界

本轮9个合法配对点中，Native copy Prefill低3.31%--13.69%；Ours steady Decode低
2.72×--10.41×，E2E低2.65×--7.53×。完整表和RAG/B16容量记录见：

- [结果文档](../eurosys_moe_offload_evaluation/RESULTS_E3_NEAR_EQUAL_NATIVE_COPY_GRAPH_20260903.md)
- [统一结果汇总](../eurosys_moe_offload_evaluation/RESULTS_E0_E12_LATEST_CONSOLIDATED_20260831.md)

本协议是单轮qualification，不是统计重复或同进程ABBA。两臂只配平卸载字节范围，卸载
对象和数据路径不同，不能把倍率归因于任一单项机制。
