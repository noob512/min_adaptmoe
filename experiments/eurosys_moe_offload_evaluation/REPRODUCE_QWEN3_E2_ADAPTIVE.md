# 复现 Qwen3 E2 自适应并发实验

本文档复现 `e2-qwen3-adaptive-concurrency-20260827-attempt-01`：在 2 GiB KV 下，按并发档位选择 cold19/cold25 专家计划，在热门专家迁回前捕获并永久保留正式 pure-Prefill Graph，然后完成 Qwen3 E2 的 16 个 cell。

## 1. 复现范围

运行矩阵：

```text
workloads = short_dialogue, ordinary_chat, rag_summary, long_generation
concurrency = C1, C4, C8, C16
cells = 16
```

进程块映射：

```text
C1          → cold19 → Prefill Graph 最大 8192 tokens
C4/C8/C16  → cold25 → Prefill Graph 最大 16384 tokens
```

每个进程块内 placement 冻结；不会在性能测量期间在线增减 cold experts。

## 2. 必需文件

核心实现：

- `workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py`
  - `capture_pure_prefill_graphs()`；
  - `VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=1` 路径；
  - final Prefill Graph capture 后以 `preserve_allocator_cache=True` 恢复热门专家。
- `workloads/vllm/vllm/vllm/v1/core/sched/scheduler.py`
  - pure Prefill/Decode epoch；
  - KV-aware FCFS cohort partition。
- `experiments/eurosys_moe_offload_evaluation/formal_openai_block.py`
  - model-specific fixture root；
  - workload filter；
  - per-block verifier 和 cleanup。

实验入口与配置：

- `experiments/eurosys_moe_offload_evaluation/run_qwen_e2_adaptive_concurrency.sh`
- `experiments/eurosys_moe_offload_evaluation/configs/e2_qwen_adaptive_concurrency_env.json`
- `experiments/eurosys_moe_offload_evaluation/configs/e2_matrix_cells.json`
- `experiments/eurosys_moe_offload_evaluation/streaming_client.py`

冻结输入：

- `experiments/eurosys_moe_offload_evaluation/inputs/generated/eurosys-pa-v2-20260824/qwen3_30b_a3b_fp8/`
- `.../e2_adaptive_plans/cold19/`
- `.../e2_adaptive_plans/cold25/`

汇总：

- `experiments/eurosys_moe_offload_evaluation/summarize_qwen_e2_adaptive.py`

## 3. 环境前提

参考成功环境：

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090，32 GiB |
| Kernel | `6.13.12-061312-generic` |
| 模型 | 本地 Qwen3-30B-A3B-FP8 snapshot `d206ba732169f29bb77fbf80fc2c4b81d4d30782` |
| vLLM Python | `workloads/vllm/vllm/.venv/bin/python` |
| custom UVM fault mode | 2 |
| custom UVM slot budget | 1,207,959,552 B |
| custom UVM PTE remap | enabled |

运行者必须能够加载/卸载本仓库 custom `nvidia_uvm`，并在实验后恢复 stock 模块。开始前必须满足：

- 没有 GPU compute process；
- `nvidia-smi` 正常；
- GPU memory used 为 0 MiB；
- stock UVM 已加载；
- model snapshot、fixture、cold19/cold25 plan 均存在；
- 当前 shell 没有另一个实验占用 UVM control slot。

launcher 会执行 preflight、custom module load、进程块清理和最终 stock UVM restore。若 preflight 不满足，停止并先清理环境，不要并行启动第二轮 GPU 实验。

## 4. 先运行定向测试

在仓库根目录执行：

```bash
workloads/vllm/vllm/.venv/bin/python -m pytest \
  workloads/vllm/vllm/tests/v1/core/test_scheduler.py \
  -k 'uvm_phase_batched_scheduler'
```

预期结果为 3 passed，覆盖：

- chunked Prefill 与 Decode 分离；
- waiting cohort Prefill 完成后才 cutover；
- 按 KV capacity 拆分 FCFS cohort。

静态检查：

```bash
bash -n experiments/eurosys_moe_offload_evaluation/run_qwen_e2_adaptive_concurrency.sh
workloads/vllm/vllm/.venv/bin/python -m py_compile \
  experiments/eurosys_moe_offload_evaluation/formal_openai_block.py \
  experiments/eurosys_moe_offload_evaluation/summarize_qwen_e2_adaptive.py
```

## 5. 启动完整实验

选择一个不存在的新 artifact 路径；runner 会拒绝覆盖已有目录：

```bash
bash experiments/eurosys_moe_offload_evaluation/run_qwen_e2_adaptive_concurrency.sh \
  experiments/eurosys_moe_offload_evaluation/artifacts/e2-qwen3-adaptive-concurrency-REPRO-attempt-01
```

如模型位于其他本地路径：

```bash
FORMAL_MODEL=/absolute/path/to/Qwen3-30B-A3B-FP8 \
bash experiments/eurosys_moe_offload_evaluation/run_qwen_e2_adaptive_concurrency.sh \
  experiments/eurosys_moe_offload_evaluation/artifacts/e2-qwen3-adaptive-concurrency-REPRO-attempt-01
```

不要在实验进程运行时调用 `torch.cuda.empty_cache()`，也不要销毁 final pure-Prefill Graph。两个操作都会破坏正式请求需要复用的 allocator/Graph pool。

## 6. launcher 的精确行为

### 6.1 C1/cold19 进程块

```text
workloads: short_dialogue_c1, ordinary_chat_c1,
           rag_summary_c1, long_generation_c1
cold experts/layer: 19
Prefill Graph tokens: 128,512,1024,2048,4096,8192
Decode Graph batches: 1,4,8,16
```

### 6.2 C4/C8/C16/cold25 进程块

```text
workloads: 每个 workload 的 C4/C8/C16
cold experts/layer: 25
Prefill Graph tokens: 128,512,1024,2048,4096,8192,16384
Decode Graph batches: 1,4,8,16
```

两个进程块共同使用：

```text
--kv-cache-memory-bytes 2147483648
--max-model-len 8320
--max-num-batched-tokens 16384
--max-num-seqs 16
--gpu-memory-utilization 0.97
--no-enable-prefix-caching
CompilationMode.NONE
CUDAGraphMode.FULL_SEPARATE
```

关键环境变量：

```text
VLLM_UVM_ORIGINAL_VA_PHASE_MODE=prefill_graph_remote_decode
VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE=1
VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=1
VLLM_UVM_PHASE_BATCHED_SCHEDULING=1
VLLM_UVM_PHASE_KV_AWARE_COHORTS=1
VLLM_UVM_PHASE_COHORT_KV_RESERVE_BLOCKS=1
VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES=201326592
VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE=1
```

`run_qwen_e2_adaptive_concurrency.sh` 中的 `workload_filter` 决定并发→计划映射，且 `formal_openai_block.py --fixture-root` 将通用 E2 cell 名称解析到 Qwen3 的冻结 fixture。

## 7. 生成机器可读汇总

实验成功后执行：

```bash
python3 experiments/eurosys_moe_offload_evaluation/summarize_qwen_e2_adaptive.py \
  experiments/eurosys_moe_offload_evaluation/artifacts/e2-qwen3-adaptive-concurrency-REPRO-attempt-01
```

输出：

```text
summary.json
metrics.csv
```

若确实要重新生成已有汇总，显式添加 `--overwrite`。汇总脚本会：

- 验证 16 个 cell 和全部请求成功；
- 汇总 TTFT、客户端 TPOT、E2E、goodput 和 SLO；
- 对每个请求删除首个 post-cutover ITL，计算 steady Decode TPOT；
- 提取 Graph partition、allocator reserved、hot/cold residency、NVML peak；
- 提取 KV partition 序列；
- 检查 Xid/PTE/PDE/illegal access/CUDA error/eager fallback；
- 检查 alias/backing cleanup 和 stock UVM restore。

## 8. 成功判据

设 artifact 根目录为 `$RUN_ROOT`，应满足：

```bash
jq -r '.status, .qualification.observed_cells, .qualification.successful_requests, .qualification.stock_uvm_restored' \
  "$RUN_ROOT/summary.json"
```

期望：

```text
ok
16
116
true
```

还应满足：

```bash
jq -e '
  .qualification.expected_cells == 16 and
  .qualification.observed_cells == 16 and
  .qualification.successful_cells == 16 and
  .qualification.successful_requests == .qualification.total_requests and
  .qualification.fatal_log_patterns_absent and
  .qualification.stock_uvm_restored and
  ([.blocks[].verifier_status] | all(. == "ok")) and
  ([.blocks[].cleanup.ok] | all)
' "$RUN_ROOT/summary.json"
```

## 9. 关键诊断检查

若启动或请求失败，按以下顺序检查：

1. `offload_manifest.jsonl` 中 final Graph capture 是否成功；
2. `cpu_offload_uvm_original_va_sealed_hot_residency_reconciled` 的 `allocator_reserved_before == allocator_reserved_after`；
3. serving gate 是否满足 hot GPU bytes = hot bytes，cold CPU bytes = cold bytes，unknown = 0；
4. `free_bytes >= minimum_free_bytes`；
5. server log 中正式请求前后是否出现新的大 allocation、hot CPU residency 或 fault；
6. 对 RAG，检查 `KV partition`，不要把 `Running/Waiting` 排队误判为 GPU hang；
7. 退出后检查 `remaining_alias_bytes=0`、`remaining_backing_bytes=0`、GPU 0 MiB 和 stock UVM。

如果失败进程在清理后仍占用 `/dev/nvidia-uvm`，不要立即启动下一轮；先终止进程并确认模块可以恢复。只有进程处于不可中断 D state 且 custom UVM 无法卸载时才需要重启。

## 10. 成功参考 artifact

本次参考结果位于：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e2-qwen3-adaptive-concurrency-20260827-attempt-01/
```

其中：

- `c1_cold19/`：C1 进程块；
- `c4_c8_c16_cold25/`：C4/C8/C16 进程块；
- `summary.json`：完整资格、Graph、驻留、KV partition 与清理摘要；
- `metrics.csv`：16 个 cell 的性能数据；
- `module.before/custom/after.json`：stock→custom→stock 模块证据。
