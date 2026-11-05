# 复现 E2-C16 direct-batch 单轮矩阵

本文复现 2026-08-28 的 DeepSeek-V2-Lite C16 单轮 qualification。它不是 equal-HBM 正式 E2。

## 1. 实验语义

每个 cell 必须同时满足：

1. 16 条请求在首次 `EngineCore.step()` 前全部加入引擎；
2. 不启动 HTTP server/client/socket；
3. 运行完整 prompt 和文档规定的完整 output；
4. pure-phase：该 cohort 全部 Prefill 后才 Decode；
5. 不因 OOM 缩短请求，只能增加卸载量；
6. 每个候选独立进程，3 条 warmup 后测 1 轮。

“一次性入队”不等于“所有 prompt token 必须在一个 GPU kernel/engine step 内计算”。RAG 的 token budget 是 16,384，而 16×8,192=131,072，因此 scheduler 会用多个 Prefill step，但 16 条 admission 仍在第一次 step 前完成，且不会与 Decode 混合。

## 2. 文件和数据

```text
experiments/eurosys_moe_offload_evaluation/
  configs/e2_c16_direct_batch_matrix.json
  configs/c1_native_common_env.json
  configs/c1_ours_runtime_hot_audit_off_env.json
  configs/e2_ktransformers_env.json
  configs/e2_moe_infinity_env.json
  direct_batch_c16_vllm_runner.py
  direct_batch_c16_ktransformers_runner.py
  direct_batch_c16_moe_infinity_runner.py
  direct_batch_phase_runner.py
  verify_direct_batch_c16_case.py
  run_e2_c16_direct_batch_matrix.sh
```

模型必须存在：

```text
/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/
snapshots/85864749cd611b4353ce1decdb286193298f64c7
```

fixtures 位于：

```text
experiments/eurosys_moe_offload_evaluation/inputs/generated/
eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/
```

MoE-Infinity 还需要冻结的 offload store 和 receipt；路径由 launcher 顶部的 `MOEI_OFFLOAD`、`MOEI_RECEIPT` 固定。

## 3. 执行前检查

```bash
cd /home/ubuntu/nvidia-uvm-gpu
nvidia-smi
test -f workloads/vllm/vllm/.venv/bin/python
test -f KTransformers/.venv/bin/python
jq empty experiments/eurosys_moe_offload_evaluation/configs/e2_c16_direct_batch_matrix.json
bash -n experiments/eurosys_moe_offload_evaluation/run_e2_c16_direct_batch_matrix.sh
workloads/vllm/vllm/.venv/bin/python -m py_compile \
  experiments/eurosys_moe_offload_evaluation/direct_batch_c16_vllm_runner.py
```

确认没有遗留实验、端口和 GPU 占用。launcher 会额外执行 static preflight、模型/fixture/path 检查，并同时持有：

```text
/tmp/eurosys_c16_direct_batch_matrix.lock
/tmp/eurosys_moe_offload_e2_gpu0.lock
```

不要并行启动另一个 GPU 实验。

## 4. 完整运行

```bash
bash experiments/eurosys_moe_offload_evaluation/run_e2_c16_direct_batch_matrix.sh \
  experiments/eurosys_moe_offload_evaluation/artifacts/e2-c16-direct-batch-<tag>
```

定向运行：

```bash
E2_C16_SYSTEMS=ours \
E2_C16_WORKLOADS=rag_summary \
bash experiments/eurosys_moe_offload_evaluation/run_e2_c16_direct_batch_matrix.sh \
  experiments/eurosys_moe_offload_evaluation/artifacts/e2-c16-direct-batch-ours-rag-<tag>
```

可用 system 名称：

```text
native_all_gpu,native_pinned_uva,native_copy_on_forward,
ours,ktransformers,moe_infinity
```

可用 workload 名称：

```text
short_dialogue,ordinary_chat,rag_summary,long_generation
```

只允许对同一个未完成 artifact 显式恢复：

```bash
E2_C16_RESUME=1 E2_C16_SYSTEMS=ours E2_C16_WORKLOADS=rag_summary \
bash experiments/eurosys_moe_offload_evaluation/run_e2_c16_direct_batch_matrix.sh \
  <existing-artifact>
```

launcher 不覆盖已有 case；成功 case 会跳过，失败候选会保留。

## 5. 空间预留与 ours 的易错顺序

顺序不能改变：

1. 模型权重加载，按 frequency plan 标记 cold experts；
2. KV 初始化前分配与该 workload `kv_cache_bytes` 相同的 native hole sentinel；
3. 专家保持 all-remote，完成 kernel warmup 和 Decode Graph 捕获；
4. 释放 hole，捕获仅包含 `{1,16,prefill_tokens}` 的 Graph；
5. 在专家仍 remote 时，用正式 shape 建立 Prefill allocator reserve；
6. 恢复热点专家，检查所有热点页 GPU resident；
7. 强制保留至少 201,326,592 bytes headroom；
8. 三条 warmup；每次 Prefill→Decode 执行 PARK/remote map；下一个 Prefill 前 RESUME；
9. 正式 16-request admission 后运行。

hole 就是“warmup 阶段为后续 KV/Graph/临时缓冲区预留连续空间”的实现。它解决分配顺序和碎片问题，不会减少模型、KV、Graph 和 workspace 的总物理占用。若恢复热点专家后容量仍不足，必须增加 cold 数；不要反复调整 sentinel 来掩盖真实容量冲突。

每类 workload 的候选在 `e2_c16_direct_batch_matrix.json` 中冻结。RAG 当前最小成功为 cold19/层。

## 6. copy-on-forward 的精确卸载

Ordinary/RAG/Long 不得再用通用参数遍历顺序。launcher 为每个候选生成：

```text
cases/native_copy_on_forward/<workload>/offload_w2_layers_N/
exact_w2_parameter_names.json
```

内容只允许：

```text
model.layers.1.mlp.experts.w2_weight
...
model.layers.N.mlp.experts.w2_weight
```

每个参数 369,098,752 bytes（352 MiB），`cpu_offload_gb=N×0.34375`。不要选择完整 `w13_weight`：其单层约 704 MiB，forward 临时复制容易 OOM。不要卸载 attention 参数，否则派生缓存（例如 `W_UK_T`）可能继续引用 CPU storage。

## 7. MoE-Infinity 特殊要求

DeepSeek non-paged 路径使用 HF `past_key_values`，因此本实验设置：

```text
materialize_paged_kv_cache=False
paged_kv_storage_materialized=false
kv_cache_storage_bytes=0
```

逻辑 KV block 数不能减少；只是不物化从未消费的第二份 GPU tensor。serving 只投影每条序列最后 hidden state 的 logits。若日志仍显示全序列 logits 或物理 PagedKV，则配置/代码版本不匹配，结果无效。

## 8. 监控

```bash
tail -f <artifact>/execution.tsv
ps -eo pid,stat,etime,cmd | rg 'run_e2_c16|direct_batch_c16'
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```

不要在某个 launcher 正在运行时修改 runner、launcher 或配置；artifact 已记录输入 hash，运行中修改会破坏可追溯性。

## 9. 验收

每个成功 case 的 `verification.json` 必须为 `status=ok`，且以下字段全部 true：

```text
runner_returncode_zero
result_status_ok
three_warmups_succeeded
measurement_16_of_16_succeeded
no_http_server
no_client
one_c16_measurement_admission
scheduler_prefilled_all_16_before_decode
no_mixed_prefill_decode_step
full_configured_generation_completed
no_fatal_log_pattern
```

另外手工检查：

```bash
jq . <case>/verification.json
rg -n 'CUDA error|illegal memory access|Xid|PTE.*fault|PDE.*fault|Traceback' \
  <case>/run.log <case>/kernel.since.log
nvidia-smi
lsmod | rg '^nvidia_uvm'
```

退出后必须 GPU 0 MiB、无 compute process、stock UVM 恢复。

## 10. 常见失败与处理

- `Hot experts were not exclusively GPU resident`：当前 cold 数不足或临时 reserve 抢占热点；确认 hole/reserve 顺序后增加 cold 数。
- `Insufficient GPU headroom`：热点虽全部恢复，但没有正式请求余量；增加 cold 数，不删除 192 MiB 门禁。
- copy 路径在 attention/MLA 中访问 CPU tensor：检查 exact-name 文件，禁止通用遍历卸载。
- copy 路径临时 OOM：优先只卸载 `w2_weight`，增加层数；不要改成 704 MiB `w13_weight`。
- RAG 首 step 只有 2 条：这是 16,384 token budget 的内部 chunk，不是两次 admission；检查 `admitted_request_count=16` 和 cohort cutover 日志。
- MoE-Infinity `GPUFetchFunc0 ... cudaMalloc ... out of memory`：属于 expert-cache 容量问题，保留完整日志并继续其他机制，不缩短 8K prompt。
- exact-generation 不一致：结构成功不等于正确性成功；保留逐请求 token/hash，进入未解决报告。

复现 stock pinned-UVA 的 FULL Graph 诊断（预期在当前 Triton MLA backend 初始化时被拒绝）：

```bash
E2_C16_FORCE_CUDAGRAPH=1 \
E2_C16_SYSTEMS=native_pinned_uva \
E2_C16_WORKLOADS=rag_summary \
bash experiments/eurosys_moe_offload_evaluation/run_e2_c16_direct_batch_matrix.sh \
  experiments/eurosys_moe_offload_evaluation/artifacts/e2-c16-rag-graph-diag-<tag>
```

该开关只用于定位 RAG exact-generation 分叉，不属于正式机制配置。

## 11. 结果文件

每个 case 至少包含：

```text
command.txt              实际命令
environment.json         冻结环境变量
run.log                  stdout/stderr
result.json              请求级 token、TTFT、TPOT、E2E、SLO
verification.json        结构验收
nvml.csv                 200 ms GPU 采样
resource_time.txt        /usr/bin/time -v
kernel.since.log         本 cell 内核日志
gpu.before.json/after    清理证据
offload_manifest.jsonl   支持该机制时的卸载记录
```

本次参考结果和问题分别见：

```text
RESULTS_E2_C16_DIRECT_BATCH_MATRIX_20260828.md
UNRESOLVED_E2_C16_DIRECT_BATCH_MATRIX_20260828.md
ERROR_FIX_LOG_E2_C16_DIRECT_BATCH_MATRIX_20260828.md
```
