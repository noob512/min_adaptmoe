# 复现 E3 B1/B8 Prefill 与 Decode 三机制实验

本文档复现 E3 6.5 phase-only qualification。协议最初冻结于 2026-08-31，当前权威证据
核验于 2026-09-02，覆盖：

1. 自定义 CUDA Graph copy 的 Prefill；
2. 自定义 CUDA Graph remote-read 的 Decode；
3. vLLM 原生 copy-on-forward；
4. vLLM 原生 pinned-UVA remote-read。

实验只测试 B1/B8，不经过 HTTP。每个 B8 在第一次 EngineCore step 前一次性提交 8 条请求，禁止改成 `1+7`、两个 B4 或 continuous arrival。Prefill 与 Decode 的 scheduler step 必须完全分离。

2026-08-28 的原协议在 Decode Medium/B8 和 RAG/B8 留有两个 N/A。当前协议不拆 batch、
不启用 chunked Prefill，而是让这两个大 Prefill cell 使用已经通过验收的 D/TRITON
全覆盖双页表路径，并对称补跑 native TRITON copy/UVA。最终每个 Decode cell 内的三个
机制使用相同 attention backend，10/10 cell 完整输出 hash 一致。

## 0. 2026-09-02 当前权威证据

本协议最近一次完整复验由四组不可覆盖 artifact 组成：

```text
Native FLASH copy/UVA
  artifacts/e3-native-copy-uva-reproduce-all-20260901-attempt-01
Native TRITON Medium/B8、RAG/B8
  artifacts/e3-native-triton-large-b8-reproduce-all-20260901-attempt-01
Prefill TRITON Graph-copy
  artifacts/e3-prefill-graph-copy-reproduce-all-20260901-attempt-01
Decode 当前档完整汇总
  artifacts/e3-decode-complete-reproduce-all-20260901-attempt-01
```

前三组分别为20/20、4/4和10/10运行成功；Decode verifier为`status=ok`，覆盖10/10
logical cell和24/24 run。本文12节中的参考表已按这些artifact更新。它们是分臂复验，
不得写成一个共同冷进程，也不能跨attention backend解释成单变量Graph消融。

## 1. 可以复现的结论

当前证据分为三个层次，报告时不得混淆：

- Prefill，TRITON Graph-copy 对 native UVA：10/10 cell 更快，为 1.033–13.065x。
- Prefill，TRITON Graph-copy 对 FLASH native copy：Short/B1 基本持平并快 1.004x，
  其余 9/10 cell 更慢；该表不是同 backend、同卸载几何的单变量消融。
- Decode，8 个 C/FLASH cell 加 2 个 D/TRITON cell 均为逐 cell 同 backend；Graph remote
  对 native copy 为 2.30–6.92x，对 native UVA 为 1.95–6.75x，10/10完整输出 hash 一致。

不能声称“Graph-copy Prefill 对 FLASH vLLM copy 全部大幅领先”。Prefill 主表混入
attention backend、expert-slice/whole-Parameter 卸载粒度和进程 conditioning 差异。
Prefill 的 Medium/B8 还有一条请求首 token 不同，因此该 cell 不能进入跨三臂
exact-generation 排名。Decode 的10个cell虽然在各自cell内同backend，但整张表由
FLASH和TRITON两个协议层组成，也不能解释为全局统一backend的纯Graph消融。

## 2. 实验组成

推荐按顺序串行执行五个主矩阵；它们共用一张 GPU，不能并行：

| 矩阵 | 机制 | 作用 |
| --- | --- | --- |
| N-FLASH | vLLM copy/UVA，FLASH_ATTN eager | 提供全部10个Prefill点和8个Decode同backend基线 |
| P | custom Graph-copy，TRITON_ATTN FULL_SEPARATE | 提供10个Prefill Graph cells |
| C-FLASH | custom Graph remote，FLASH_ATTN FULL_DECODE_ONLY | 提供8个常规Decode cell |
| D-TRITON | custom Prefill+Decode Graph，TRITON_ATTN FULL_SEPARATE | 补齐Medium/B8和RAG/B8 |
| N-TRITON | vLLM copy/UVA，TRITON_ATTN eager | 为两个D-TRITON cell提供对称native基线 |

如需隔离 Prefill Graph 相对 eager copy 的贡献，再运行可选矩阵 T：双方均使用 TRITON，custom 使用 exact expert slices，native 使用近似字节数的完整 Parameter。

## 3. 冻结环境

| 项目 | 设置 |
| --- | --- |
| 工作目录 | `/home/ubuntu/nvidia-uvm-gpu` |
| GPU | NVIDIA GeForce RTX 5090 32 GiB，GPU 0，TP=1 |
| 模型 | Qwen3-30B-A3B-FP8 |
| snapshot | `d206ba732169f29bb77fbf80fc2c4b81d4d30782` |
| vLLM | `0.1.dev11683+gd9d342d21.d20260330` |
| PyTorch/CUDA | `2.9.0+cu128` / CUDA 12.8 |
| CPU | `taskset -c 0-13`，OMP/MKL 14 threads |
| seed | 0 |
| 采样 | greedy，temperature 0，ignore EOS |
| prefix cache | 关闭 |
| chunked Prefill | 关闭 |
| EngineCore multiprocessing | 关闭 |
| 请求入口 | in-process EngineCore；无 server、HTTP 或 client |

请求和 KV 必须保持如下设置：

| Workload | Prompt | Output | B | Prefill tokens | KV bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short dialogue | 128 | 32 | 1 / 8 | 128 / 1,024 | 167,772,160 / 167,772,160 |
| Ordinary chat | 512 | 128 | 1 / 8 | 512 / 4,096 | 167,772,160 / 515,899,392 |
| Medium context | 2,048 | 128 | 1 / 8 | 2,048 / 16,384 | 215,482,368 / 1,723,858,944 |
| RAG summary | 8,192 | 128 | 1 / 8 | 8,192 / 65,536 | 819,462,144 / 6,555,697,152 |
| Long generation | 512 | 512 | 1 / 8 | 512 / 4,096 | 167,772,160 / 817,889,280 |

KV 公式为：

```text
tokens_per_request = round_up_16(prompt + output + 16)
KV bytes = tokens_per_request * batch * 98,304
actual KV = max(KV bytes, 160 MiB)
```

即使 Prefill-only custom arm 只采样一个 token，仍使用 `max_model_len=prompt+原始 output` 和上表 KV。禁止缩成 `prompt+1`，否则 warmup、KV 和内部 shape envelope 已改变。

## 4. 文件与环境预检

在仓库根目录执行：

```bash
cd /home/ubuntu/nvidia-uvm-gpu

nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
sudo fuser /dev/nvidia-uvm /dev/nvidia-uvm-tools 2>/dev/null || true

test -x workloads/vllm/vllm/.venv/bin/python
test -f workloads/vllm/vllm/uvm_test/uvm_allocator.so
test -f kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko
test -f experiments/eurosys_moe_offload_evaluation/e3_prefill_only_runner.py
test -f experiments/eurosys_moe_offload_evaluation/run_e3_prefill_only_four_mechanism.sh
test -f experiments/eurosys_moe_offload_evaluation/run_e3_native_copy_vs_uva_phase.sh
test -f experiments/eurosys_moe_offload_evaluation/run_e3_decode_graph_remote_b1_b8.sh
test -f experiments/eurosys_moe_offload_evaluation/run_e3_graph_single_round_b1_b8_no_http.sh
test -f experiments/eurosys_moe_offload_evaluation/summarize_e3_graph_single_round.py
test -f experiments/eurosys_moe_offload_evaluation/inputs/frozen/e3_qwen_cold63_20260828/exact_parameter_names.txt
test -f experiments/qwen_decode_six_mechanism_compare/inputs/cold_decode_plan_2g.json
test -f experiments/qwen_decode_six_mechanism_compare/inputs/cold_decode_plan_8g.json
test -f experiments/eurosys_moe_offload_evaluation/inputs/frozen/e3_qwen_cold63_20260828/cold_decode_plan_2g.json
test -f /home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782/config.json
```

开始前必须满足：

- GPU 为 0 MiB；
- 无 compute process；
- `/dev/nvidia-uvm*` 无客户端；
- 当前加载的是系统 stock UVM，而不是上次异常退出遗留的 custom UVM。

检查 stock 与 custom module：

```bash
modinfo -F filename nvidia-uvm
modinfo -F srcversion nvidia-uvm
cat /sys/module/nvidia_uvm/srcversion

modinfo -p kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko \
  | rg 'uvm_moe_slot_pte_remap_enable|uvm_manual_migration_fault_mode|uvm_phase_f_remote_read_full_tlb_invalidate'
```

custom module 必须包含上述三个参数。若源码变更后 `.ko` 过期，重新构建本地模块；不要安装覆盖系统模块，runner 会临时加载本地 `.ko` 并在结束后恢复 stock：

```bash
make -C kernel-module/nvidia-module modules -j"$(nproc)"
make -C workloads/vllm/vllm/uvm_test
```

静态检查：

```bash
bash -n \
  experiments/eurosys_moe_offload_evaluation/run_e3_prefill_only_four_mechanism.sh \
  experiments/eurosys_moe_offload_evaluation/run_e3_native_copy_vs_uva_phase.sh \
  experiments/eurosys_moe_offload_evaluation/run_e3_decode_graph_remote_b1_b8.sh \
  experiments/eurosys_moe_offload_evaluation/run_e3_graph_single_round_b1_b8_no_http.sh

workloads/vllm/vllm/.venv/bin/python -m py_compile \
  experiments/eurosys_moe_offload_evaluation/e3_prefill_only_runner.py \
  experiments/vllm_uvm_prefill_alias_decode_remote_graph/benchmark_phase_switch_steps.py \
  workloads/vllm/vllm/vllm/model_executor/models/utils.py \
  workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py
```

## 5. Artifact 命名规则

所有 runner 都拒绝覆盖已有目录。为本次复现指定一个新标签：

```bash
E3_REPRO_TAG=manual-01
NATIVE_ART=experiments/eurosys_moe_offload_evaluation/artifacts/e3-native-copy-uva-${E3_REPRO_TAG}
PREFILL_ART=experiments/eurosys_moe_offload_evaluation/artifacts/e3-prefill-graph-copy-${E3_REPRO_TAG}
DECODE_ART=experiments/eurosys_moe_offload_evaluation/artifacts/e3-decode-graph-remote-${E3_REPRO_TAG}
NATIVE_TRITON_ART=experiments/eurosys_moe_offload_evaluation/artifacts/e3-native-triton-large-b8-${E3_REPRO_TAG}
DECODE_COMPLETE_ART=experiments/eurosys_moe_offload_evaluation/artifacts/e3-decode-complete-${E3_REPRO_TAG}
TRITON_PAIR_ART=experiments/eurosys_moe_offload_evaluation/artifacts/e3-prefill-triton-pair-${E3_REPRO_TAG}

test ! -e "${NATIVE_ART}"
test ! -e "${PREFILL_ART}"
test ! -e "${DECODE_ART}"
test ! -e "${NATIVE_TRITON_ART}"
test ! -e "${DECODE_COMPLETE_ART}"
test ! -e "${TRITON_PAIR_ART}"
```

artifact 中的 `command.txt`、`environment.txt`、`run.log`、`returncode.txt`、`result.json`、`manifest.jsonl`、`nvml.csv` 和 `provenance/` 均应保留，不要只复制汇总表。

## 6. 矩阵 N：原生 vLLM copy 与 UVA

运行一次即可同时获得完整 Prefill 与 Decode 数据：

```bash
ALLOW_GPU_RUN=1 \
ATTENTION_BACKEND=FLASH_ATTN \
CONDITIONING_REPEATS=1 \
MEASUREMENT_REPEATS=1 \
bash experiments/eurosys_moe_offload_evaluation/run_e3_native_copy_vs_uva_phase.sh \
  "${NATIVE_ART}"
```

配置语义：

- stock UVM；
- copy 和 UVA 均为 eager，无 CUDA Graph；
- 同一 workload/B 的两个 arm 使用相同完整 Parameter 集合；
- 每个进程先运行 1 个 conditioning cycle，再运行 1 个 measurement cycle；
- 一个完整 Prefill step 后只允许 cached-only、每请求一个 token 的 Decode steps；
- 完整生成 32/128/512 output tokens。

原生 vLLM 不能按 expert slice 卸载，因此使用完整 Parameter：

| Cell | 请求 budget | 典型实际 Parameters | 典型实际 bytes |
| --- | ---: | ---: | ---: |
| 常规 cell | 2.25 GiB | 63 | 2,493,476,896 |
| Medium/B8、RAG/B1 | 7.875 GiB | 232 | 8,527,542,384 |
| RAG/B8 | 13.5 GiB | 402 | 14,762,934,464 |

不要降低大 cell 的 budget，否则可能在模型加载、KV 或 Prefill workspace 分配阶段 OOM。

### 6.1 两个大B8 cell的TRITON native基线

Decode 的 Medium/B8 和 RAG/B8 使用 D/TRITON 补齐，因此必须另外运行相同
TRITON attention backend 的 native copy/UVA；不能用原 N-FLASH 数值与 D/TRITON
直接组成 exact-generation 三臂比较：

```bash
ALLOW_GPU_RUN=1 \
ATTENTION_BACKEND=TRITON_ATTN \
WORKLOAD_FILTER=medium_context,rag_summary \
BATCH_FILTER=8 \
CONDITIONING_REPEATS=1 \
MEASUREMENT_REPEATS=1 \
bash experiments/eurosys_moe_offload_evaluation/run_e3_native_copy_vs_uva_phase.sh \
  "${NATIVE_TRITON_ART}"
```

该命令应得到4/4成功进程。两个arm在各自cell中必须使用相同的realized whole-Parameter
signature，完整生成128 tokens，并与对应D/TRITON输出hash一致。

## 7. 矩阵 P：自定义 Prefill Graph-copy

```bash
ALLOW_GPU_RUN=1 \
RUN_CUSTOM=1 \
RUN_NATIVE=0 \
CUSTOM_ARM_FILTER=graph_copy \
ATTENTION_BACKEND=TRITON_ATTN \
bash experiments/eurosys_moe_offload_evaluation/run_e3_prefill_only_four_mechanism.sh \
  "${PREFILL_ART}"
```

配置语义：

- custom UVM original-VA static alias；
- `CUDAGraphMode=FULL_SEPARATE`；
- Graph capture sizes 为去重后的 `{batch, prompt*batch}`；
- expert ID 使用冻结 exact coordinates；
- `max_tokens=min_tokens=1`，只执行一次 Prefill/首 token step，Decode step 为 0；
- 计时边界为 `torch.cuda.synchronize(); llm_engine.step(); torch.cuda.synchronize()`；
- 模型加载、warmup、Graph capture、allocator reserve、rebalance 和 cleanup 不计入 Prefill 时间。

容量计划：

| Cell | cold/层 | selected bytes | alias backing | reserve warmup tokens |
| --- | ---: | ---: | ---: | ---: |
| 常规 | 10 | 2,264,924,160 | 94,371,840 | 0 |
| Medium/B8 | 38 | 8,606,711,808 | 358,612,992 | 16,384 |
| RAG/B1 | 38 | 8,606,711,808 | 358,612,992 | 8,192 |
| RAG/B8 | 63 | 14,269,022,208 | 594,542,592 | 20,480 |

关键启动顺序：

1. 模型权重进入 managed original VA；
2. 建立等于该 cell KV 大小的 pre-KV native hole；
3. 建立 KV 和 kernel/workspace warmup；
4. 专家暂时保持 all-remote，完成正式 Graph capture 和 allocator reserve；
5. 释放 hole；
6. 恢复热门专家、保留 Graph pool 与 allocator cache；
7. 正式请求只 replay，不 recapture；
8. 销毁 session/backing，恢复 stock UVM。

当前 FlashAttention 不支持该机制要求的 `FULL_SEPARATE`。不要把本命令改为 `FLASH_ATTN` 后接受自动降级的 `FULL_DECODE_ONLY` 数据；契约应在测量前 fail closed。

## 8. 矩阵 C-FLASH与D-TRITON：自定义Decode CUDA Graph remote-read

### 8.1 八个C/FLASH cell

```bash
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/run_e3_decode_graph_remote_b1_b8.sh \
  "${DECODE_ART}"
```

配置语义：

- arm C：conditioning Prefill 为 remote eager；
- Decode 为 original-VA CPU remote-read `FULL_DECODE_ONLY` CUDA Graph；
- FLASH_ATTN；
- Graph capture size 为当前 B1 或 B8；
- 首个完整 Prefill 结束后才进入 Decode；
- 第一个 Decode、warmup Decode 和 steady Decode 分开记录；
- runtime 三轮热门专家内容审计关闭，但结构、residency、PTE/PDE、Graph 和输出契约仍保留。

有效 cell 的容量配置：

| Cell | cold/层 | MoE chunk | FlashInfer autotune cap |
| --- | ---: | ---: | ---: |
| Short/Ordinary B1/B8 | 10 | 16,384 | 默认 |
| Medium/B1 | 10 | 16,384 | 默认 |
| Long/B1 | 10 | 16,384 | 默认 |
| Long/B8 | 38 | 1,024 | 2,048 |
| RAG/B1 | 63 | 1,024 | 2,048 |

该C/FLASH命令仍会为Medium/B8和RAG/B8生成两个历史N/A receipt；它们只表示C/FLASH
original-VA大Prefill路径的能力边界，不再表示完整三机制矩阵缺数据。

### 8.2 两个D/TRITON补缺cell和完整双页表复核

使用当前档全覆盖双页表runner。该runner会同时重跑其余cell以形成自身10-cell、24-run
结构验收；最终三机制性能表只从中取Medium/B8和RAG/B8两个D/TRITON结果，RAG/B1仍取
上述C/FLASH结果：

```bash
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/\
run_e3_graph_single_round_b1_b8_no_http.sh \
  "${DECODE_COMPLETE_ART}"

python3 experiments/eurosys_moe_offload_evaluation/\
summarize_e3_graph_single_round.py \
  --require-current-tier-dual-view \
  --output-dir "${DECODE_COMPLETE_ART}/summary" \
  "${DECODE_COMPLETE_ART}/groups/tier2_main" \
  "${DECODE_COMPLETE_ART}/groups/tier2_medium_b1" \
  "${DECODE_COMPLETE_ART}/groups/dual_tier8_medium_b8" \
  "${DECODE_COMPLETE_ART}/groups/dual_tier8_rag_b1" \
  "${DECODE_COMPLETE_ART}/groups/dual_cold63_rag_b8"
```

Medium/B8使用cold38、8.607 GB selected weights和16,384-token单步Prefill；RAG/B8
使用cold63、14.269 GB selected weights和65,536-token单步Prefill。两点均保持B8整批
入队、不拆batch、不启用chunked Prefill。D的Prefill和Decode均为Graph replay，P→D
通过全覆盖双页表切换，`remote_range_map_ioctl_calls=0`，服务期不得recapture。

### 8.3 必须保留的故障修复

Decode custom runner 固定：

```text
REMOTE_READ_FULL_TLB_INVALIDATE=1
VLLM_UVM_POST_RECONCILE_FULL_TLB_INVALIDATE=1
uvm_phase_f_remote_read_full_tlb_invalidate=1
```

全 GPU TLB flush 单独不能修复原 Long/B8 fault。真正修复是在所有 kernel、Graph、sampler 和 allocator warmup 完成后，再执行一次最终 cold-remote/hot-GPU residency reconciliation；随后进行同步 all-GPC TLB invalidate。

不要用 `torch.cuda.empty_cache()` 替代该顺序。它既不建立 GPU MMU 完成边界，又可能破坏已建立的 Graph/allocator pool。

对Medium/B8和RAG/B8，C/FLASH曾在大Prefill conditioning中失败。D/TRITON的
FULL_SEPARATE Graph和全覆盖双页表是当前通过验收的协议路径；不能把这一补缺解释为
FLASH故障已经修复，也不能把两个backend层合并成统一FLASH实验。

## 9. 可选矩阵 T：同为 TRITON 的 Prefill Graph 与 eager copy

该矩阵用于复现 7 个共同有效 cell 上 1.023–1.383x 的 Graph-copy 单轮优势：

```bash
ALLOW_GPU_RUN=1 \
RUN_CUSTOM=1 \
RUN_NATIVE=1 \
CUSTOM_ARM_FILTER=graph_copy \
ATTENTION_BACKEND=TRITON_ATTN \
bash experiments/eurosys_moe_offload_evaluation/run_e3_prefill_only_four_mechanism.sh \
  "${TRITON_PAIR_ART}"
```

该可选矩阵T仍只复现2026-08-28冻结的7个共同cell。它当时记录的TRITON native N/A
不是当前平台的全局能力声明：最新N-TRITON已经让Medium/B8和RAG/B8完整运行。不要把
不同runner、不同日期的点静默拼进旧T矩阵；如需扩展Prefill同backend消融，应重新注册
并整组运行，而不是只补有利cell。

这仍不是完全相同的卸载几何：custom 是 exact expert slices，native 是最接近字节数的完整 packed Parameters。

## 10. 结构验收

### 10.1 执行状态

```bash
awk 'NR>1 {count[$8]++} END {for (key in count) print key,count[key]}' \
  "${NATIVE_ART}/execution.tsv"

awk 'NR>1 {count[$8]++} END {for (key in count) print key,count[key]}' \
  "${PREFILL_ART}/execution.tsv"

awk 'NR>1 {count[$8]++} END {for (key in count) print key,count[key]}' \
  "${NATIVE_TRITON_ART}/execution.tsv"

find "${DECODE_ART}/groups" -path '*/custom/execution.tsv' -type f \
  -exec awk 'NR>1 {count[$9]++} END {for (key in count) print key,count[key]}' {} \;

find "${DECODE_ART}/na" -name '*.json' -type f -print

jq -e '
  .status == "ok" and
  .expected_cells == 10 and .successful_cells == 10 and
  .expected_runs == 24 and .successful_runs == 24 and
  .current_tier_dual_view_required == true and
  (.errors | length) == 0
' "${DECODE_COMPLETE_ART}/summary/verification.json"
```

预期：

- native FLASH：20 `ok`；
- native TRITON补缺：4 `ok`；
- Prefill Graph-copy：10 `ok`；
- C/FLASH Decode Graph remote：8 `ok`，另有2个历史N/A receipt；
- 当前档双页表矩阵：10/10 cell、24/24 run，聚合verifier为`status=ok`。

### 10.2 Prefill Graph 验收

```bash
for result in "${PREFILL_ART}"/cases/*_graph_copy/result.json; do
  jq -e '
    .status == "ok" and
    .configuration.graph_prefill == true and
    .configuration.decode_steps == 0 and
    (.outputs | length) == .configuration.batch_size and
    all(.outputs[]; .finished and (.output_token_ids | length) == 1) and
    (.schedule.new_request_ids | length) == .configuration.batch_size and
    (.schedule.cached_request_ids | length) == 0 and
    (.schedule.resumed_request_ids | length) == 0 and
    (.schedule.preempted_request_ids | length) == 0 and
    .schedule.total_num_scheduled_tokens ==
      (.configuration.prompt_tokens * .configuration.batch_size) and
    .graph_wrapper_delta_prefill.full_replay == 1 and
    (.compilation_counter_after_request.num_cudagraph_captured ==
     .compilation_counter_before_request.num_cudagraph_captured)
  ' "${result}" >/dev/null || exit 1
done
```

### 10.3 Native pure-phase 与 B8 验收

```bash
for result in "${NATIVE_ART}"/cases/*/result.json; do
  jq -e '
    .status == "ok" and
    all(.cycles[];
      . as $cycle |
      (($cycle.prefill_schedule.new_request_ids | length) == $cycle.batch_size) and
      (($cycle.prefill_schedule.cached_request_ids | length) == 0) and
      all($cycle.measured_decode_steps[];
        ((.schedule.new_request_ids | length) == 0) and
        ((.schedule.cached_request_ids | length) == $cycle.batch_size) and
        (.schedule.total_num_scheduled_tokens == $cycle.batch_size)))
  ' "${result}" >/dev/null || exit 1
done

for result in "${NATIVE_TRITON_ART}"/cases/*/result.json; do
  jq -e '
    .status == "ok" and
    all(.cycles[];
      . as $cycle |
      (($cycle.prefill_schedule.new_request_ids | length) == $cycle.batch_size) and
      (($cycle.prefill_schedule.cached_request_ids | length) == 0) and
      all($cycle.measured_decode_steps[];
        ((.schedule.new_request_ids | length) == 0) and
        ((.schedule.cached_request_ids | length) == $cycle.batch_size) and
        (.schedule.total_num_scheduled_tokens == $cycle.batch_size)))
  ' "${result}" >/dev/null || exit 1
done
```

### 10.4 Custom Decode pure-phase、Graph 与 cleanup 验收

```bash
while IFS= read -r result; do
  jq -e '
    . as $root |
    .status == "ok" and
    (($root.steps.prefill.schedule.new_request_ids | length) == $root.config.batch_size) and
    (($root.steps.prefill.schedule.cached_request_ids | length) == 0) and
    (($root.steps.first_decode.schedule.new_request_ids | length) == 0) and
    (($root.steps.first_decode.schedule.cached_request_ids | length) == $root.config.batch_size) and
    ($root.steps.first_decode.schedule.total_num_scheduled_tokens == $root.config.batch_size) and
    all($root.steps.measured_decode[];
      ((.schedule.new_request_ids | length) == 0) and
      ((.schedule.cached_request_ids | length) == $root.config.batch_size) and
      (.schedule.total_num_scheduled_tokens == $root.config.batch_size) and
      (.graph_wrapper_delta.full_replay == 1)) and
    ($root.graph_wrapper_calls_after_load.full_capture == 1) and
    ($root.shutdown_contract.cleanup_errors | length) == 0 and
    ($root.shutdown_contract.diagnostics_after_shutdown.driver.session_count == 0)
  ' "${result}" >/dev/null || exit 1
done < <(find "${DECODE_ART}/groups" -path '*/custom/*/result.json' -type f)
```

### 10.5 输出一致性

对每个有效 custom Decode cell，copy、UVA 和 custom 的完整输出 hash 必须一致：

```bash
while IFS= read -r result; do
  jq -e '.status == "ok"' "${result}" >/dev/null || continue
  workload=$(jq -r '.config.fixture | split("/")[-1] | sub("^evaluation_";"") | sub("[.]json$";"")' "${result}")
  batch=$(jq -r '.config.batch_size' "${result}")
  custom_hash=$(jq -r '.output_token_ids_sha256' "${result}")
  copy_hash=$(jq -r '.cycles[-1].output_token_ids_sha256' \
    "${NATIVE_ART}/cases/${workload}_b${batch}_copy/result.json")
  uva_hash=$(jq -r '.cycles[-1].output_token_ids_sha256' \
    "${NATIVE_ART}/cases/${workload}_b${batch}_uva/result.json")
  test "${custom_hash}" = "${copy_hash}"
  test "${custom_hash}" = "${uva_hash}"
done < <(find "${DECODE_ART}/groups" -path '*/custom/*/result.json' -type f)
```

两个D/TRITON补缺cell必须与N-TRITON的copy/UVA完整hash一致：

```bash
for workload in medium_context rag_summary; do
  if [[ "${workload}" == medium_context ]]; then
    group=dual_tier8_medium_b8
  else
    group=dual_cold63_rag_b8
  fi
  result=$(find "${DECODE_COMPLETE_ART}/groups/${group}/custom" \
    -path "*_${workload}_b8_D/result.json" -print -quit)
  test -n "${result}"
  custom_hash=$(jq -r '.output_token_ids_sha256' "${result}")
  for arm in copy uva; do
    native_hash=$(jq -r '.cycles[-1].output_token_ids_sha256' \
      "${NATIVE_TRITON_ART}/cases/${workload}_b8_${arm}/result.json")
    test "${custom_hash}" = "${native_hash}"
  done
done
```

Prefill-only首token另行验收：10个logical cell中9个三臂一致；Medium/B8的8条请求中
有1条不同。该Prefill cell保留运行和性能数据，但不得标为跨三臂exact-generation通过。

## 11. 性能数据提取

### 11.1 Prefill

Custom Graph-copy：

```bash
for result in "${PREFILL_ART}"/cases/*_graph_copy/result.json; do
  jq -r '[.configuration.workload,.configuration.batch_size,
          .timing.prefill_step_ms,.timing.prompt_tokens_per_second] | @tsv' \
    "${result}"
done
```

Native copy/UVA：

```bash
for result in "${NATIVE_ART}"/cases/*/result.json; do
  jq -r '
    .config.batch_sizes[0] as $batch |
    [(.profile_label | capture("e3_native_(?<arm>copy|uva)_(?<workload>.*)_b").workload),
     $batch,.environment.VLLM_CPU_OFFLOAD_BACKEND,
     .phase_summary[($batch|tostring)].prefill_step_ms.median] | @tsv' \
    "${result}"
done
```

### 11.2 Decode

Native 直接读取 measurement cycle 的 batch-step median：

```bash
for result in "${NATIVE_ART}"/cases/*/result.json; do
  jq -r '
    .config.batch_sizes[0] as $batch |
    [.environment.VLLM_CPU_OFFLOAD_BACKEND,$batch,
     .phase_summary[($batch|tostring)].decode_all_step_ms.median,
     .phase_summary[($batch|tostring)].decode_all_step_ms.p90,
     .phase_summary[($batch|tostring)].decode_tokens_per_second] | @tsv' \
    "${result}"
done

for result in "${NATIVE_TRITON_ART}"/cases/*/result.json; do
  jq -r '
    .config.batch_sizes[0] as $batch |
    [.environment.VLLM_CPU_OFFLOAD_BACKEND,$batch,
     .phase_summary[($batch|tostring)].decode_all_step_ms.median,
     .phase_summary[($batch|tostring)].decode_all_step_ms.p90,
     .phase_summary[($batch|tostring)].decode_tokens_per_second] | @tsv' \
    "${result}"
done
```

Custom runner 单独保存第一个 Decode，然后保存 5 个额外 warmup。为与 native 的“排除前 5 个物理 Decode steps”对齐，使用 `warmup_decode[-1] + measured_decode` 重新计算 median：

```bash
while IFS= read -r result; do
  jq -r '
    select(.status == "ok") |
    ([.steps.warmup_decode[-1].elapsed_ms] +
     [.steps.measured_decode[].elapsed_ms] | sort) as $samples |
    ($samples|length) as $count |
    [(.config.fixture|split("/")[-1]|sub("^evaluation_";"")|sub("[.]json$";"")),
     .config.batch_size,$count,
     (if ($count % 2) == 0
      then (($samples[$count/2-1] + $samples[$count/2]) / 2)
      else $samples[(($count-1)/2)] end)] | @tsv' "${result}"
done < <(find "${DECODE_ART}/groups" -path '*/custom/*/result.json' -type f)
```

对两个D/TRITON cell使用相同口径，从当前档双页表artifact提取：

```bash
for result in \
  "${DECODE_COMPLETE_ART}"/groups/dual_tier8_medium_b8/custom/*_medium_context_b8_D/result.json \
  "${DECODE_COMPLETE_ART}"/groups/dual_cold63_rag_b8/custom/*_rag_summary_b8_D/result.json; do
  jq -r '
    ([.steps.warmup_decode[-1].elapsed_ms] +
     [.steps.measured_decode[].elapsed_ms] | sort) as $samples |
    ($samples|length) as $count |
    [(.config.fixture|split("/")[-1]|sub("^evaluation_";"")|sub("[.]json$";"")),
     .config.batch_size,$count,
     (if ($count % 2) == 0
      then (($samples[$count/2-1] + $samples[$count/2]) / 2)
      else $samples[(($count-1)/2)] end)] | @tsv' "${result}"
done
```

Decode 延迟是“整个 B1/B8 批次各生成一个 token”的 step latency，不是单请求 latency。吞吐为 `batch * 1000 / step_ms`。

## 12. 参考结果

### 12.1 Prefill 跨配置数值对照

单位 ms：

| 请求 | B | Custom Graph-copy | Native copy | Native UVA | Graph/copy | Graph/UVA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Short 128 | 1 | 67.93 | 68.19 | 70.18 | 1.004x | 1.033x |
| Short 128 | 8 | 92.14 | 76.82 | 196.81 | 0.834x | 2.136x |
| Ordinary 512 | 1 | 73.96 | 70.62 | 121.89 | 0.955x | 1.648x |
| Ordinary 512 | 8 | 151.54 | 137.93 | 826.48 | 0.910x | 5.454x |
| Medium 2048 | 1 | 113.44 | 103.48 | 418.44 | 0.912x | 3.689x |
| Medium 2048 | 8 | 651.61 | 563.05 | 7,508.83 | 0.864x | 11.524x |
| RAG 8192 | 1 | 588.59 | 461.11 | 3,754.00 | 0.783x | 6.378x |
| RAG 8192 | 8 | 3,696.11 | 2,728.73 | 48,291.27 | 0.738x | 13.065x |
| Long 512 | 1 | 75.03 | 71.55 | 122.66 | 0.954x | 1.635x |
| Long 512 | 8 | 151.16 | 138.14 | 834.24 | 0.914x | 5.519x |

倍率定义为`native latency / Custom latency`。Custom相对UVA为10/10更快；相对
native copy仅Short/B1基本持平并快1.004x，其余9点更慢。Custom使用TRITON和exact
expert slices，native使用FLASH和whole Parameters，因此这里只作数值诊断。

### 12.2 Decode 三机制

单位为 ms/whole-batch one-token step：

| 请求 | B | 协议层 | Custom Graph remote | Native copy | Native UVA | Custom/copy | Custom/UVA |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Short 128->32 | 1 | C/FLASH | 10.20 | 70.56 | 68.80 | 6.92x | 6.75x |
| Short 128->32 | 8 | C/FLASH | 15.37 | 70.64 | 69.48 | 4.60x | 4.52x |
| Ordinary 512->128 | 1 | C/FLASH | 10.13 | 69.90 | 67.36 | 6.90x | 6.65x |
| Ordinary 512->128 | 8 | C/FLASH | 17.95 | 71.87 | 69.41 | 4.00x | 3.87x |
| Medium 2048->128 | 1 | C/FLASH | 13.98 | 69.32 | 68.97 | 4.96x | 4.93x |
| Medium 2048->128 | 8 | D/TRITON | 42.01 | 167.77 | 82.78 | 3.99x | 1.97x |
| RAG 8192->128 | 1 | C/FLASH | 24.01 | 161.39 | 68.01 | 6.72x | 2.83x |
| RAG 8192->128 | 8 | D/TRITON | 64.21 | 280.88 | 125.06 | 4.37x | 1.95x |
| Long 512->512 | 1 | C/FLASH | 11.06 | 69.65 | 67.34 | 6.30x | 6.09x |
| Long 512->512 | 8 | C/FLASH | 31.51 | 72.38 | 70.49 | 2.30x | 2.24x |

倍率定义为`native latency / Custom latency`。逐cell同backend的10个点中Custom全部
最快，完整输出hash为10/10一致。该范围合并8个FLASH点和2个TRITON点，只是矩阵概览；
不能将跨协议层的范围解释为统一attention backend下的纯remote或纯Graph排名。

## 13. 故障处理

### GPU 或 UVM 不干净

不要在存在 UVM client 时直接卸载模块。先确认是否为本实验进程：

```bash
nvidia-smi
sudo fuser -v /dev/nvidia-uvm /dev/nvidia-uvm-tools 2>/dev/null || true
pgrep -af 'e3_prefill_only_runner|benchmark_phase_switch_steps|benchmark_phase_steps|run_formal_e3'
```

等待 runner 自行 cleanup。只有确认是本轮异常遗留进程后才终止对应 PID；不要使用影响无关任务的宽泛 kill 命令。

### custom module 缺少参数

现象为 preflight 报缺少 `uvm_phase_f_remote_read_full_tlb_invalidate` 或 srcversion 不一致。重新构建本地 module，确认 `modinfo -p`，并确保实验开始时加载的仍是 stock module。不要手工安装 local `.ko` 覆盖系统文件。

### Graph Prefill 被降级

`FLASH_ATTN + FULL_SEPARATE` 会被降到 `FULL_DECODE_ONLY`。这是 backend 不兼容，不是 OOM。custom Prefill 必须使用 TRITON；不得接受降级后生成的非 Graph-Prefill 数据。

### KV/workspace 挤占热门专家

不得只依赖模型加载后的 residency：

- KV 前建立与 KV 大小一致的 native hole；
- warmup 阶段完成最大必要 workspace/allocator reserve；
- capture 后、正式请求前恢复热门专家；
- remote-only Decode 在所有 startup warmup 后再次 final reconcile；
- reconcile 后同步全 GPU TLB invalidate。

不要在正式请求前随意减少 cold 数，也不要用 `empty_cache()` 破坏 Graph pool。

### `FAULT_PDE` / `FAULT_PTE`

先区分故障发生在 warmup、conditioning Prefill 还是 Decode。全 TLB flush 后仍 fault 表明问题不是旧 TLB，而是 residency/PTE/PDE 建立时序或不支持的大 shape。保存：

```bash
sudo journalctl -k --since '10 minutes ago' --no-pager \
  | rg 'Xid|MMU Fault|FAULT_PDE|FAULT_PTE'
```

Medium/B8、RAG/B8的C/FLASH conditioning仍属于已知能力边界：三次Medium/B8-G/FLASH
定向尝试均在startup或正式16K Prefill触发CUDA illegal-address。完整矩阵应改走本文规定的
D/TRITON路径并使用对称N-TRITON基线；不得拆B8、启用chunked Prefill，或把跨backend
输出hash当成同协议通过。

### 输出 hash 不一致

同一有效 cell 的完整输出 hash 不一致时，该 pair 不能参与性能结论。先检查 fixture SHA256、seed、sampling、模型 snapshot、Parameter/expert placement 和 attention backend；不要只比较请求是否返回成功。

### 清理失败

每个 custom result 必须满足 `session_count=0`，下一 cell 前 GPU 必须回到 0 MiB。若 runner 未恢复 stock UVM，本轮后续数据全部暂停，先恢复模块和确认无客户端。

## 14. 最终解释边界

- 这是单轮 qualification，只提供点估计；正式论文结果需要多个独立冷进程、median/p95 和置信区间。
- Prefill Graph 不包含一次性 capture 成本；它测量稳定 replay。
- custom exact expert slices 与 native whole Parameters 不等价。
- Prefill 主表的 custom 使用 TRITON，native 使用 FLASH；不能把全部差异归因于 CUDA Graph。
- Decode的8个C cell共享FLASH，两个大B8 D cell共享TRITON；每个cell内部backend一致，
  但整张10-cell表不是全局统一backend。
- custom仍同时包含Graph、expert-slice几何和HBM布局差异，native为eager与whole-Parameter；
  Decode倍率不能作为纯remote访问内核或纯Graph的单项排名。
- Medium/B8、RAG/B8已由D/TRITON补齐；C/FLASH的两个N/A必须保留为能力边界，不能删除
  或改写成FLASH成功。

复现成功的标准是：性能趋势、调度契约、输出一致性、Graph replay 和 cleanup 同时通过，而不是只得到一张延迟表。

## 15. 已验证 artifact

2026-09-01/02当前复验：

- Native FLASH copy/UVA：`artifacts/e3-native-copy-uva-reproduce-all-20260901-attempt-01/`
- Native TRITON大B8补缺：`artifacts/e3-native-triton-large-b8-reproduce-all-20260901-attempt-01/`
- Prefill Graph-copy：`artifacts/e3-prefill-graph-copy-reproduce-all-20260901-attempt-01/`
- Decode 10-cell/24-run完整汇总：`artifacts/e3-decode-complete-reproduce-all-20260901-attempt-01/`

2026-08-31历史完整复跑：

- Native FLASH copy/UVA：`artifacts/e3-section65-native-20260831-attempt-01/`
- Native TRITON大B8补缺：`artifacts/e3-section65-native-triton-large-b8-20260831-attempt-01/`
- Prefill Graph-copy：`artifacts/e3-section65-prefill-20260831-attempt-01/`
- C/FLASH Decode：`artifacts/e3-section65-decode-flash-20260831-attempt-01/`
- D/TRITON补缺和双页表verifier：`artifacts/e3-section65-decode-complete-20260831-attempt-01/`
- G/FLASH失败诊断：`artifacts/e3-section65-flash-g-medium-b8-20260831-attempt-01/`、
  `artifacts/e3-section65-flash-g-medium-b8-20260831-attempt-02/`、
  `artifacts/e3-section65-flash-g-medium-b8-20260831-attempt-03-cold63/`

2026-08-28历史补充消融：

- 同TRITON Graph/copy pair：`artifacts/e3-prefill-only-graph-vs-native-copy-20260828-attempt-01`
- Long/B8 TLB修复定向验证：`artifacts/e3-decode-eager-remote-long-b8-post-warmup-reconcile-tlb-20260828-attempt-01`

配套结果与问题记录：

- `RESULTS_E3_SECTION65_COMPLETE_RERUN_20260831.md`（历史完整结果）
- `RESULTS_E0_E12_LATEST_CONSOLIDATED_20260831.md`（当前权威汇总）
- `RESULTS_E3_PREFILL_GRAPH_COPY_VS_NATIVE_COPY_UVA_B1_B8_20260828.md`（历史）
- `RESULTS_E3_PREFILL_GRAPH_VS_NATIVE_COPY_20260828.md`（历史）
- `RESULTS_E3_DECODE_THREE_MECHANISM_B1_B8_20260828.md`（历史）
- `ERROR_FIX_LOG_E3_DECODE_TLB_RECONCILE_20260828.md`
- `UNRESOLVED_E3_DECODE_B1_B8_20260828.md`
