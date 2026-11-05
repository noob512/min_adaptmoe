# 复现 E2、E3、E10 C1/C16 全覆盖双页表单轮实验

日期：2026-08-31（Asia/Shanghai）

## 1. 范围

本文档只采集最新 `ours`：

- E2：C1、C16；
- E3：B1、B16；
- E10：B1，KV 2→4→6 GiB 压力切档；
- 不运行 MoE-Infinity；
- 不重复 pinned-UVA 和 copy-on-forward，它们可复用此前独立冷进程产物；
- 每个配置运行一次，失败时保留原产物，不覆盖或静默重试。

## 2. 冻结环境

```text
repo       /home/ubuntu/nvidia-uvm-gpu
GPU        NVIDIA GeForce RTX 5090 32 GiB，GPU 0，TP=1
driver     595.71.05
kernel     6.13.12-061312-generic
timezone   Asia/Shanghai
```

所需模型：

```text
/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/
  snapshots/85864749cd611b4353ce1decdb286193298f64c7/
/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/
  snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782/
```

本次已验证模块：

```text
nvidia-uvm.ko SHA256  c8fe77c8ecc87f6d724d94ddf6535c934150d4e12bec384bd44562097aec043a
srcversion            CD4A27464CD3E907C4EF493
stock srcversion      85A79790636BBD99BA3E43B
```

## 3. 机制门禁

`ours` 必须同时满足：

```text
VLLM_CPU_OFFLOAD_PLAN_CLASS_IMMUTABLE_CPU_SHADOW=1
VLLM_CPU_OFFLOAD_DUAL_VIEW_FULL_PARAMETER_SHADOW=1
VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW=1
dual_view_full_coverage=true
dual_view_block_count>0
remote_map_operation=prebuilt_full_dual_view
remote range-map ioctl calls=0
```

E2 模块加载时还强制：

```text
STATIC_ALIAS_DUAL_VIEW_ENABLE=1
STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL=1
MANUAL_MIGRATION_FAULT_MODE=2
ACCESS_COUNTER_MIGRATION_ENABLE=0
UVM_PREFETCH_ENABLE=0
UVM_THRASHING_ENABLE=0
```

E3 的模块允许不兼容块返回错误，但进程级
`VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW=1` 会在覆盖不足时 fail-fast；验收仍要求实际覆盖
100%。E10 同时在模块和进程级要求全覆盖。

Prefill 工作区预留复用由以下入口完成：

```text
VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_BYTES=<该场景KV字节数>
VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS=<该场景预留token数>
VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=1
```

## 4. 构建与静态检查

```bash
cd /home/ubuntu/nvidia-uvm-gpu

make -C kernel-module/nvidia-module -j"$(nproc)" modules
make -C workloads/vllm/vllm/uvm_test

sha256sum kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko
modinfo -F srcversion kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko

bash -n \
  experiments/eurosys_moe_offload_evaluation/run_c1_mechanism_runtime_hot_audit_off_matrix.sh \
  experiments/eurosys_moe_offload_evaluation/run_e2_c16_direct_batch_matrix.sh \
  experiments/eurosys_moe_offload_evaluation/run_e3_c1_c16_uva_single_round.sh

E10_TRIGGER_MODE=pressure \
bash experiments/vllm_uvm_e10_multilevel_pressure/run_experiment.sh static-preflight
```

本轮 E10 专项门禁为13项，公共门禁为8项。还应运行当前工作树中的定向 vLLM/UVM
测试；本轮合计101项通过。

## 5. 每次 GPU 运行前检查

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

要求 GPU 0 MiB、无 compute process、无未知 UVM client，并且 loaded 与 stock
srcversion 相同。

```bash
REPRO_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e2-e3-e10-full-c1-c16-dual-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$REPRO_ROOT/e2"
```

不要预先创建 `c1`、`c16`、`e3` 或 `e10` 子目录；runner 会拒绝覆盖已有目录。

## 6. E2：只运行 ours

### C1

```bash
ALLOW_GPU_RUN=1 \
MATRIX_SYSTEMS=ours \
MATRIX_WORKLOADS=short_dialogue,ordinary_chat,rag_summary,long_generation \
MATRIX_SINGLE_ATTEMPT=1 \
bash experiments/eurosys_moe_offload_evaluation/\
run_c1_mechanism_runtime_hot_audit_off_matrix.sh \
  "$REPRO_ROOT/e2/c1"
```

### C16

```bash
ALLOW_GPU_RUN=1 \
E2_C16_SYSTEMS=ours \
E2_C16_WORKLOADS=short_dialogue,ordinary_chat,rag_summary,long_generation \
bash experiments/eurosys_moe_offload_evaluation/\
run_e2_c16_direct_batch_matrix.sh \
  "$REPRO_ROOT/e2/c16"
```

验收：

```bash
find "$REPRO_ROOT/e2" -type f -name dual_view_verification.json -print0 |
  xargs -0 -n1 jq -e '.status == "ok" and ([.checks[]] | all)'

find "$REPRO_ROOT/e2" -type f -name kernel.since.log -print0 |
  xargs -0 rg -n -i 'Xid|FAULT_PDE|FAULT_PTE|GPU page fault|kernel BUG' && exit 1 || true
```

流式 C1 verifier 还要求一个 SSE event 只携带一个 token。如果模型完整输出但仅有
`grouped_token_events>0`，必须报告为协议不通过，不能把它写成功能或 GPU 故障，也不能
为获得绿灯而重跑。

## 7. E3：只运行 D/ours

当前脚本使用 frozen 当前档：小场景 cold10，中等场景 cold38，RAG/C16 cold89。
RAG/C16 必须用 cold89，因为其12.21 GiB KV和131,072-token Prefill Graph/工作区无法
与 cold63 留下的14.72 GiB热专家同时驻留。

```bash
ALLOW_GPU_RUN=1 \
E3_RUN_NATIVE_BASELINES=0 \
E3_RUN_KTRANSFORMERS=0 \
bash experiments/eurosys_moe_offload_evaluation/\
run_e3_c1_c16_uva_single_round.sh \
  "$REPRO_ROOT/e3"
```

该命令只运行10个 D 单元，不执行 UVA、copy、KTransformers 或 MoE-Infinity。runner
同时检查子单元的 `status/returncode` 并 fail-fast，不再只依赖外层 launcher 的退出码。
验收：

```bash
awk -F '\t' 'NR > 1 && $1 == "ours" && $6 != "0" {bad=1} END {exit bad}' \
  "$REPRO_ROOT/e3/execution.tsv"

! find "$REPRO_ROOT/e3" -type f -name run.log -print0 |
  xargs -0 rg -n -i 'Xid|FAULT_PDE|FAULT_PTE|GPU page fault|CUDA error|Traceback'
```

每个 `manifest.jsonl` 的首次 Decode transition 都必须包含
`dual_view_full_coverage=true`、正数 `dual_view_block_count`、
`remote_map_operation=prebuilt_full_dual_view` 和 `ioctl_calls=0`。

若只需复核 RAG/C16 容量修正，使用全新目录且只运行一次：

```bash
ALLOW_GPU_RUN=1 \
E3_RUN_NATIVE_BASELINES=0 \
E3_RUN_KTRANSFORMERS=0 \
E3_OURS_CELL_FILTER=rag_summary_b16 \
bash experiments/eurosys_moe_offload_evaluation/\
run_e3_c1_c16_uva_single_round.sh \
  "$REPRO_ROOT/e3-rag-c16-cold89"
```

## 8. E10：压力触发2→4→6 GiB

```bash
RUN_ROOT="$REPRO_ROOT/e10" \
E10_TRIGGER_MODE=pressure \
ALLOW_GPU_RUN=1 \
bash experiments/vllm_uvm_e10_multilevel_pressure/run_experiment.sh qualify
```

验收：

```bash
jq -e '
  .status == "ok" and
  .completed_sequences == 1 and
  .statistical_repetitions == 0 and
  .trigger_mode == "pressure" and
  .kv_target_blocks == [1364,2730,4095] and
  .expert_plan_classes == ["cold70","cold80","cold89"] and
  .dual_view_parent_switch.enabled == true and
  .dual_view_parent_switch.latency_stats_delta.slot_alias_dual_skipped_blocks == 0
' "$REPRO_ROOT/e10/verification.json"
```

## 9. 最终清理与故障审计

runner 正常退出时会恢复 stock UVM。最后再次执行：

```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
modinfo -F filename nvidia_uvm

find "$REPRO_ROOT" -type f \
  \( -name '*.log' -o -name '*.err' -o -name '*.txt' \) -print0 |
  xargs -0 rg -n -i \
  'Xid [0-9]|FAULT_PDE|FAULT_PTE|GPU page fault|kernel BUG|general protection fault'
```

期望为 GPU 0 MiB、stock `nvidia-uvm.ko`，故障扫描无输出。任何跳过、超时、协议失败、
容量调整或被 fail-fast 阻止的单元都必须单独列出，不能并入成功计数。

## 10. 复核 ours + Graph 与 pinned-UVA remote

按本协议不重复运行 UVA/copy。E2 的 pinned-UVA remote 基线来自此前已经完成的独立冷
进程单轮：

```bash
BASELINE_E2=experiments/eurosys_moe_offload_evaluation/artifacts/\
e2-e3-e10-full-c1-c16-uva-single-round-20260831-attempt-01/e2
OURS_E2=experiments/eurosys_moe_offload_evaluation/artifacts/\
e2-e3-e10-full-c1-c16-uva-single-round-20260831-attempt-02-full-dual/e2
```

先确认基线确实是 eager pinned-UVA，而不是另一种 native arm：

```bash
rg -n -- '--enforce-eager|--cpu-offload-gb' \
  "$BASELINE_E2"/c1/cases/native_pinned_uva/*/*/command.txt

jq -e '.configuration.enforce_eager == true' \
  "$BASELINE_E2"/c16/cases/native_pinned_uva/short_dialogue/offload_0p01g/result.json
```

从冻结产物提取请求 E2E。C1 的结果位于 `client.json`，C16 direct-batch 的结果位于
`result.json`：

```bash
for workload_name in short_dialogue ordinary_chat rag_summary long_generation; do
  baseline_json=$(find "$BASELINE_E2/c1/cases/native_pinned_uva/$workload_name" \
    -name client.json -print -quit)
  ours_json=$(find "$OURS_E2/c1/cases/ours/$workload_name" \
    -name client.json -print -quit)
  jq -nr \
    --arg workload "$workload_name" \
    --argjson ours "$(jq '.phases[0].e2e_ms.p50' "$ours_json")" \
    --argjson remote "$(jq '.phases[0].e2e_ms.p50' "$baseline_json")" \
    '[$workload, $ours, $remote, ($remote / $ours)] | @tsv'
done

for workload_name in short_dialogue ordinary_chat rag_summary long_generation; do
  baseline_json=$(find "$BASELINE_E2/c16/cases/native_pinned_uva/$workload_name" \
    -name result.json -print -quit)
  ours_json=$(find "$OURS_E2/c16/cases/ours/$workload_name" \
    -name result.json -print -quit)
  jq -nr \
    --arg workload "$workload_name" \
    --argjson ours "$(jq '.measurement.e2e_ms.p50' "$ours_json")" \
    --argjson remote "$(jq '.measurement.e2e_ms.p50' "$baseline_json")" \
    '[$workload, $ours, $remote, ($remote / $ours)] | @tsv'
done
```

正式汇总只接纳 `ours` verifier 通过的配对点。Long/C1 虽然完整生成512 token，但
`grouped_token_events=1`，必须从正式加速范围中排除。其余7点的预期范围为
`2.08×–8.47×`。

该对比验证的是完整系统组合，不是消融实验。不得据此把收益单独归因于 CUDA Graph、双页表
或 GPU 缓存中的任一项；也不得把不同的实际卸载权重集合表述成完全相同的数据布局。E3 本轮
没有配对 remote arm，因此 E3 只能验证 ours 的功能、覆盖率与切换开销，不能独立提供
ours/remote 加速比。

## 11. 本轮权威结果

结果、失败边界、E2配对表和所有 artifact 路径统一记录在：

```text
RESULTS_E2_E3_E10_C1_C16_FULL_DUAL_SINGLE_ROUND_20260831.md
```

不要与早期 C1/C4/C8/C16、B1/B8 或 RAG/C16 cold63 协议拼接；这些旧协议来自不同运行
边界，不能作为本轮结果的补点。
