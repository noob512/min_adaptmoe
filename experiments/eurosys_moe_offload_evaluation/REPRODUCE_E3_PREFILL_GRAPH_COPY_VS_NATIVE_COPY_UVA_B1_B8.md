# 复现 E3 Prefill：Graph-copy、vLLM copy-on-forward 与 pinned-UVA

## 1. 实验目标

在五种冻结请求上分别测量 B1 和 B8 的完整 Prefill step，对齐展示：

1. 自定义 exact expert-slice Graph-copy；
2. 原生 vLLM copy-on-forward；
3. 原生 vLLM pinned-UVA remote-read。

本实验不通过 HTTP。B8 必须在第一次 EngineCore step 前一次性加入 8 条请求，禁止拆成 `1+7`、两个 B4 或 continuous arrival。Prefill step 内不得混入 cached Decode request。

这次结果由两个独立矩阵组成。当前 FlashAttention 不支持自定义机制要求的 `FULL_SEPARATE` Graph，因此 Graph 与两个原生 arm 使用不同 attention backend。复现后只能做带明确限制的数值对齐，不能称为严格同配置机制排名。

## 2. 已验证 artifact

- Graph-copy：`artifacts/e3-prefill-only-graph-copy-triton-20260828-attempt-02`
- 原生 copy/UVA：`artifacts/e3-native-copy-vs-uva-phase-20260828-attempt-01`
- 被排除的 Flash Graph 尝试：`artifacts/e3-prefill-only-graph-copy-flashattn-20260828-attempt-01`

所有新实验必须使用不存在的新目录，禁止覆盖上述 artifact。

## 3. 公共环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 32 GiB，GPU 0 |
| 模型 | Qwen3-30B-A3B-FP8 |
| snapshot | `d206ba732169f29bb77fbf80fc2c4b81d4d30782` |
| vLLM | `0.1.dev11683+gd9d342d21.d20260330` |
| PyTorch/CUDA | `2.9.0+cu128` / CUDA 12.8 |
| CPU | `taskset -c 0-13`，OMP/MKL 14 threads |
| TP | 1 |
| seed/采样 | 0，temperature 0，ignore EOS |
| fixture | `inputs/generated/eurosys-pa-v2-20260824/qwen3_30b_a3b_fp8/evaluation_*.json` |
| 请求传输 | in-process EngineCore，无 HTTP/client |
| prefix cache/chunked Prefill | 均关闭 |

开始前检查：

```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
modinfo -n nvidia_uvm
test -x workloads/vllm/vllm/.venv/bin/python
test -f experiments/eurosys_moe_offload_evaluation/e3_prefill_only_runner.py
test -f experiments/eurosys_moe_offload_evaluation/run_e3_native_copy_vs_uva_phase.sh
```

GPU 使用应为 0 MiB，无 compute process；开始 Graph 矩阵前和最终结束后都应是 stock `nvidia-uvm.ko`。

## 4. 请求和容量矩阵

| Workload | Prompt | 原始 output | B | 单步 Prefill tokens | KV bytes |
|---|---:|---:|---:|---:|---:|
| Short dialogue | 128 | 32 | 1/8 | 128 / 1,024 | 167,772,160 |
| Ordinary chat | 512 | 128 | 1/8 | 512 / 4,096 | 167,772,160 / 515,899,392 |
| Medium context | 2,048 | 128 | 1/8 | 2,048 / 16,384 | 215,482,368 / 1,723,858,944 |
| RAG summary | 8,192 | 128 | 1/8 | 8,192 / 65,536 | 819,462,144 / 6,555,697,152 |
| Long generation | 512 | 512 | 1/8 | 512 / 4,096 | 167,772,160 / 817,889,280 |

保留 `max_model_len=prompt+原始 output`，即使 Graph Prefill-only arm 只采样一个 token，也不得缩成 `prompt+1`。这能维持原 E3 的 KV 和内部 shape envelope。

## 5. Graph-copy 矩阵

### 5.1 配置

- `TRITON_ATTN`；
- `CUDAGraphMode=FULL_SEPARATE`；
- capture sizes 为去重后的 `{batch, prompt*batch}`；
- 自定义 UVM original-VA static alias；
- exact expert-slice coordinates；
- `max_tokens=min_tokens=1`，只执行一个 Prefill/首 token step，Decode steps 为 0；
- 计时为 `torch.cuda.synchronize()` 包围的一次 `llm_engine.step()`；
- 模型加载、warmup、Graph capture、rebalance、热门专家恢复和 cleanup 不计入 Prefill 时间。

容量计划：

| Cell | cold/层 | selected bytes | alias backing bytes |
|---|---:|---:|---:|
| 常规 cell | 10 | 2,264,924,160 | 94,371,840 |
| Medium/B8、RAG/B1 | 38 | 8,606,711,808 | 358,612,992 |
| RAG/B8 | 63 | 14,269,022,208 | 594,542,592 |

### 5.2 命令

```bash
ALLOW_GPU_RUN=1 \
RUN_CUSTOM=1 \
RUN_NATIVE=0 \
CUSTOM_ARM_FILTER=graph_copy \
ATTENTION_BACKEND=TRITON_ATTN \
bash experiments/eurosys_moe_offload_evaluation/run_e3_prefill_only_four_mechanism.sh \
  experiments/eurosys_moe_offload_evaluation/artifacts/e3-prefill-only-graph-copy-triton-NEW
```

每个 `cases/*/result.json` 必须满足：

```bash
jq -e '
  .status == "ok" and
  .configuration.graph_prefill == true and
  .configuration.decode_steps == 0 and
  (.outputs | length) == .configuration.batch_size and
  all(.outputs[]; .finished and (.output_token_ids | length) == 1) and
  (.schedule.new_request_ids | length) == .configuration.batch_size and
  (.schedule.cached_request_ids | length) == 0 and
  .graph_wrapper_delta_prefill.full_replay == 1 and
  (.compilation_counter_after_request.num_cudagraph_captured ==
   .compilation_counter_before_request.num_cudagraph_captured)
' CASE/result.json
```

## 6. 原生 copy/UVA 矩阵

### 6.1 配置

- stock UVM；
- `FLASH_ATTN`；
- eager，CUDA Graph 关闭；
- `VLLM_CPU_OFFLOAD_BACKEND=copy` 或 `uva`；
- native vLLM 只能卸载完整 Parameter，不能选择 expert slice；
- 每个进程包含 1 个 conditioning cycle 和 1 个 measurement cycle；
- 完整请求执行到 32/128/512 个输出 token；报告中只抽取 measurement cycle 的同步 Prefill step；
- Prefill 后的 Decode 每步仍由 runner 验证为严格 cached-only B1/B8，但 Decode 数据不进入本次 Prefill 表。

实际成对卸载集合：

| Cell | 请求 budget | 实际 Parameters | 实际 bytes |
|---|---:|---:|---:|
| 常规 cell | 2.25 GiB | 63 | 2,493,476,896 |
| Medium/B8、RAG/B1 | 7.875 GiB | 232 | 8,527,542,384 |
| RAG/B8 | 13.5 GiB | 402 | 14,762,934,464 |

同一 cell 的 copy/UVA 必须具有完全相同的 Parameter 名称、顺序、shape、dtype 和 bytes。

### 6.2 命令

```bash
ALLOW_GPU_RUN=1 \
ATTENTION_BACKEND=FLASH_ATTN \
bash experiments/eurosys_moe_offload_evaluation/run_e3_native_copy_vs_uva_phase.sh \
  experiments/eurosys_moe_offload_evaluation/artifacts/e3-native-copy-vs-uva-phase-NEW
```

## 7. 汇总三组 Prefill

令 Graph artifact 为 `G`，native artifact 为 `N`：

```bash
G=experiments/eurosys_moe_offload_evaluation/artifacts/e3-prefill-only-graph-copy-triton-NEW/cases
N=experiments/eurosys_moe_offload_evaluation/artifacts/e3-native-copy-vs-uva-phase-NEW/cases

for workload in short_dialogue ordinary_chat medium_context rag_summary long_generation; do
  for batch in 1 8; do
    jq -nr \
      --slurpfile g "$G/${workload}_b${batch}_graph_copy/result.json" \
      --slurpfile c "$N/${workload}_b${batch}_copy/result.json" \
      --slurpfile u "$N/${workload}_b${batch}_uva/result.json" \
      --arg workload "$workload" --argjson batch "$batch" '
        ($g[0].timing.prefill_step_ms) as $graph |
        ($c[0].phase_summary[($batch|tostring)].prefill_step_ms.median) as $copy |
        ($u[0].phase_summary[($batch|tostring)].prefill_step_ms.median) as $uva |
        [$workload,$batch,$graph,$copy,$uva,$copy/$graph,$uva/$graph] | @tsv'
  done
done
```

`copy/graph` 或 `uva/graph` 大于 1 表示 Graph 的延迟更低。但由于 backend、卸载粒度和 conditioning 不完全相同，该倍率必须标注为跨配置数值对照。

## 8. 已知错误和处理

不要把 Graph arm 改成 `FLASH_ATTN` 后继续接受结果。当前运行时会打印：

```text
FULL_SEPARATE is not supported with FlashAttentionBackend
setting cudagraph_mode=FULL_DECODE_ONLY
```

自定义契约随后应拒绝 `prefill_graph_remote_decode`。这是正确的 fail-closed 行为，不是 OOM。不得通过静默 eager fallback、拆 B8、启用 chunked Prefill 或删除契约来制造数据。

最终检查所有日志无 OOM、CUDA illegal address、Xid、PTE/PDE fault 或 traceback；每个 Graph result 的 `shutdown_contract.cleanup_errors=[]`、driver `session_count=0`；GPU 为 0 MiB，stock UVM 已恢复。

## 9. 解释边界

- Graph 表示完成 capture 和拓扑准备后的稳态 replay，不包含一次性 Graph 建立成本。
- Graph exact expert-slice 与 native whole-Parameter 不同，不能把差异全部归因于 CUDA Graph。
- Graph 使用 TRITON attention，native 使用 FLASH attention，尤其长 Prompt 的差异会受到 attention kernel 明显影响。
- Graph 只采首 token，native measurement cycle 继续完成完整输出；同步 Prefill step 的边界相近，但进程 conditioning 历史不同。
- 正式论文结论需要同 backend、同卸载几何和多个独立冷进程重复。
