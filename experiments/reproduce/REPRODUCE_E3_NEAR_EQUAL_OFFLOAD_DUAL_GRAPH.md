# 复现E3近等量卸载下Native UVA与Ours双阶段CUDA Graph对照

日期：2026-09-03（Asia/Shanghai）

## 1. 最新成功证据

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e3-near-equal-offload-plus-one-expert-20260903-attempt-02/
```

该轮只运行一次。Native与Ours各完成Short、Ordinary、Medium、RAG、Long的B1/B16十个
cell，共20/20成功；聚合`verification.json`为`status=ok`且`failed_checks=[]`，
213/213项检查通过。

attempt-01的Ordinary/B16和Long/B16错误沿用了2.25 GiB Native预算，导致字节门失败；
修正后的正式attempt-02对这两点使用7.875 GiB请求预算。不得发布attempt-01性能表。

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

实际字节差为0.08%、3.58%和7.06%，均不超过10%。Native按whole-Parameter前缀卸载，
Ours按每层校准Decode路由频次最低的11/39/90个专家切片卸载；不要求相同参数ID。
这对应在原D的cold10/38/89基础上，每层再多卸载一个专家。

## 3. 实现入口

```text
experiments/eurosys_moe_offload_evaluation/
  run_e3_near_equal_offload_plus_one_expert.sh
  prepare_e3_near_equal_offload_plans.py
  run_e3_native_copy_vs_uva_phase.sh
  run_e3_c1_c16_uva_single_round.sh
  verify_e3_near_equal_offload_plus_one_expert.py
```

总入口先校验冻结路由源SHA256并生成cold11/39/90计划，然后运行Native十点，再运行Ours
十点，最后聚合Graph、双页表、字节、hash、故障和清理门。Ours runner按cell加载custom
UVM，结束后恢复stock UVM。

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
  experiments/eurosys_moe_offload_evaluation/run_e3_near_equal_offload_plus_one_expert.sh \
  experiments/eurosys_moe_offload_evaluation/run_e3_native_copy_vs_uva_phase.sh \
  experiments/eurosys_moe_offload_evaluation/run_e3_c1_c16_uva_single_round.sh \
  experiments/eurosys_moe_offload_evaluation/run_formal_e3.sh

workloads/vllm/vllm/.venv/bin/python -m py_compile \
  experiments/eurosys_moe_offload_evaluation/prepare_e3_near_equal_offload_plans.py \
  experiments/eurosys_moe_offload_evaluation/verify_e3_near_equal_offload_plus_one_expert.py \
  experiments/vllm_p3_pinned_uva_phase_compare/benchmark_phase_steps.py \
  experiments/vllm_uvm_prefill_alias_decode_remote_graph/benchmark_phase_switch_steps.py

sha256sum \
  experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/routing/expert_distribution.pt
```

冻结路由源的期望SHA256为：

```text
912732708a46291b73e70c93291758747f285b364437ce77f58d53501d072f81
```

## 6. 唯一一轮GPU命令

输出目录必须不存在，禁止覆盖任何成功或失败attempt：

```bash
cd /home/ubuntu/nvidia-uvm-gpu

RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e3-near-equal-offload-plus-one-expert-$(date -u +%Y%m%dT%H%M%SZ)

ALLOW_GPU_RUN=1 bash \
  experiments/eurosys_moe_offload_evaluation/run_e3_near_equal_offload_plus_one_expert.sh \
  "$RUN_ROOT"
```

RAG/B16会捕获并回放完整131,072-token pure-Prefill Graph，初始化明显长于小cell；不得因
准备双页表或Graph耗时较长而把它改成chunked Prefill、拆batch或追加第二轮。

## 7. 验收

```bash
jq -e '
  .status == "ok" and
  .failed_checks == [] and
  ([.checks[]] | all) and
  (.cells | length == 10) and
  ([.cells[].output_hash_match] | all) and
  ([.cells[].offload.symmetric_realized_ratio <= 1.10] | all)
' "$RUN_ROOT/verification.json"

awk -F '\t' 'NR > 1 && $1 != "verification" && $2 != 0 {bad=1}
  END {exit bad}' "$RUN_ROOT/execution.tsv"

jq -r '.cells[] |
  [.workload,.batch,.offload.ours_gib,.offload.native_gib,
   .ours.metrics.prefill_ms,.native.metrics.prefill_ms,
   .ours.metrics.decode_ms,.native.metrics.decode_ms,
   .ours.metrics.e2e_ms,.native.metrics.e2e_ms,
   .native_over_ours.prefill_ms,.native_over_ours.decode_ms,
   .native_over_ours.e2e_ms,.ours.metrics.cutover_ms] | @tsv
' "$RUN_ROOT/summary.json" | column -t

! rg -n -i \
  'Xid [0-9]|MMU fault|PDE fault|PTE fault|illegal memory access|CUDA error|out of memory|kernel BUG' \
  "$RUN_ROOT/kernel.since.log"

nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
printf 'loaded=%s\nstock=%s\n' \
  "$(tr -d '[:space:]' </sys/module/nvidia_uvm/srcversion)" \
  "$(modinfo -F srcversion nvidia-uvm | tr -d '[:space:]')"
```

期望：20/20 arm-cell完成；Native和Ours各10个Prefill及10个Decode，共40/40个阶段的
运行时审计全部为`FULL`；Ours全部使用`prebuilt_full_dual_view`；10/10输出hash一致；
无故障；GPU最终0 MiB并恢复stock UVM。

## 8. 结果与边界

正式单轮中，Ours的Prefill、稳态Decode和含P→D的E2E均在9/10点更低；完整结果见：

- [RESULTS_E3_NEAR_EQUAL_OFFLOAD_DUAL_GRAPH_20260903.md](../eurosys_moe_offload_evaluation/RESULTS_E3_NEAR_EQUAL_OFFLOAD_DUAL_GRAPH_20260903.md)
- [统一结果汇总](../eurosys_moe_offload_evaluation/RESULTS_E0_E12_LATEST_CONSOLIDATED_20260831.md)

本协议是单轮qualification，不是统计重复或同进程ABBA。两臂卸载字节近等量，但卸载对象
不同，因此不能把倍率归因于CUDA Graph、UVA、双页表或专家选择中的任一单项机制。
