# 复现 E3 Prefill-only Graph-copy vs vLLM copy-on-forward

## 目标

在相同冻结 fixture 和 B1/B8 请求量下，只运行 Graph-copy 与原生 vLLM copy-on-forward。每个 batch 在第一次 EngineCore step 前一次性提交，不使用 HTTP，不拆分请求，不运行 Decode。

## 前置检查

在仓库根目录执行：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
test -x workloads/vllm/vllm/.venv/bin/python
test -f experiments/eurosys_moe_offload_evaluation/e3_prefill_only_runner.py
test -f experiments/eurosys_moe_offload_evaluation/inputs/frozen/e3_qwen_cold63_20260828/exact_parameter_names.txt
test -f experiments/qwen_decode_six_mechanism_compare/inputs/cold_decode_plan_2g.json
test -f experiments/qwen_decode_six_mechanism_compare/inputs/cold_decode_plan_8g.json
test -f experiments/eurosys_moe_offload_evaluation/inputs/frozen/e3_qwen_cold63_20260828/cold_decode_plan_2g.json
```

开始前 GPU 应无 compute process。不要复用已存在的 artifact 目录；runner 会 fail closed。

## 一条命令运行完整矩阵

```bash
ALLOW_GPU_RUN=1 \
CUSTOM_ARM_FILTER=graph_copy \
RUN_CUSTOM=1 \
RUN_NATIVE=1 \
bash experiments/eurosys_moe_offload_evaluation/run_e3_prefill_only_four_mechanism.sh \
  experiments/eurosys_moe_offload_evaluation/artifacts/e3-prefill-only-graph-vs-native-copy-NEW
```

脚本固定运行五种 workload、B1/B8 和一轮独立进程。先加载 custom UVM 运行 10 个 Graph-copy cell，再恢复 stock UVM 运行 native cells，最后再次验证 stock UVM 与 GPU cleanup。

## 请求与计时语义

- fixture：Short 128、Ordinary 512、Medium 2048、RAG 8192、Long 512；
- 保留原 E3 的 `max_model_len=prompt+original_output`；
- KV 按原请求的 prompt/output/batch 计算，最少 160 MiB；
- sampling 为 greedy、`max_tokens=min_tokens=1`、seed 0；
- runner 先把 B1/B8 全部 request 加到 in-process EngineCore，然后只执行一次 `llm_engine.step()`；
- `enable_chunked_prefill=false`、`enable_dbo=false`、无 HTTP；
- `prefill_step_ms` 的边界是 step 前后两次 `torch.cuda.synchronize()`，包括一次完整 Prompt Prefill 和首 token sampling。

因此本实验只能报告 Prefill/TTFT，不报告 TPOT 或完整请求 E2E。

## 两个 arm

### Graph-copy

- `CUDAGraphMode=FULL_SEPARATE`；
- capture sizes 为当前 batch 和总 Prefill token 数；
- exact expert-slice coordinates；
- cold10/cold38/cold63 随容量要求冻结选择；
- warmup/capture 后恢复热门专家，测量时必须恰好一次 `full_replay`；
- runtime 热门专家三轮审计关闭，但保留结构与 residency contract。

### Native copy-on-forward

- stock UVM、`enforce_eager=true`、CUDA Graph 关闭；
- `VLLM_CPU_OFFLOAD_BACKEND=copy`；
- 选择 4/14/24 层的完整 `w13_weight` 与 `w2_weight` Parameter，使卸载字节数尽量接近 custom cold plan；
- manifest 中每个选中 Parameter 必须是 `copy_on_forward=true`。

## 容量与大 Prefill规则

| Cell | Graph plan | Native完整层 | Native状态 |
|---|---|---:|---|
| 普通 cell | cold10 | 4 | 运行 |
| Medium/B8、RAG/B1 | cold38 | 14 | N/A |
| RAG/B8 | cold63 | 24 | N/A |

Native eager 在本平台单步 Prefill `>=8192` tokens 时 fail closed。不要为了得到数值而拆成两批、启用 chunking 或改 backend；这些操作会改变实验问题。

## 验收

对每个 `result.json` 检查：

```bash
jq -e '
  .status == "ok" and
  (.outputs | length) == .configuration.batch_size and
  all(.outputs[]; .finished and (.output_token_ids | length) == 1) and
  (.schedule.new_request_ids | length) == .configuration.batch_size and
  .schedule.total_num_scheduled_tokens ==
    (.configuration.prompt_tokens * .configuration.batch_size) and
  (.schedule.cached_request_ids | length) == 0 and
  (.schedule.resumed_request_ids | length) == 0 and
  (.schedule.preempted_request_ids | length) == 0
' PATH/TO/result.json
```

Graph-copy 还必须满足：

```bash
jq -e '.graph_wrapper_delta_prefill.full_replay == 1' PATH/TO/result.json
```

结束后检查：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

应无 compute process，memory.used 为 0 MiB；`provenance/module.after.json` 应指向 stock `nvidia-uvm.ko`。

## 易错点

1. 不要把 `max_model_len` 改成 `prompt+1`。即使只采一个 token，也会改变原 E3 的内部 envelope、warmup shape 和 KV 条件。
2. 不要用逐条 `generate` 或 HTTP 并发替代 direct batch；否则 B8 可能变成 `1+7`。
3. 不要把 Graph capture 成功当作 replay 成功；必须检查测量期 `full_replay=1` 且没有 recapture。
4. 大 Prefill 需要 cold38/cold63；cold10 会让 KV、workspace 或 Graph pool 挤占热门专家。
5. native 是完整 Parameter footprint match，不是 exact expert-coordinate；报告时必须保留该限制。
6. 单轮差异尤其是 Long/B1 的大差距必须重复后才能做稳定性结论。
