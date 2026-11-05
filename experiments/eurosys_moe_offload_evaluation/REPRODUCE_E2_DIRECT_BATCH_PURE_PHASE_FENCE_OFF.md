# 复现 E2 direct-batch pure-phase fence-off 四 cell 实验

2026-08-31 当前工作树的最新同模块复验产物为：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
e2-e3-e10-current-dual-view-REPRO-20260831-attempt-01/e2/
```

该轮使用 custom UVM SHA256
`c8fe77c8ecc87f6d724d94ddf6535c934150d4e12bec384bd44562097aec043a`，29/29请求成功，
`verification.json`为`status=ok`。E3、E10在相同模块二进制上的配套复现见
`REPRODUCE_E2_E3_E10_CURRENT_DUAL_VIEW_SINGLE_ROUND_20260831.md`。

本文档复现 2026-08-27 完成的 E2 定向资格实验：在 DeepSeek-V2-Lite-Chat、
2 GiB KV cache、每层 cold11 专家计划下，关闭 post-RESUME 只读 PTE fence，
不启动 OpenAI HTTP server、不使用 streaming client，直接在进程内将
C1/C4/C8/C16 的完整请求批次加入 EngineCore。实验要求每个测量 cell 在首次
EngineCore `step()` 前已经加入全部请求，以排除 HTTP 请求到达偏斜造成的
`1+3`、`1+7` 和 `1+15` cohort。

最初的历史参考 artifact：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
e2-direct-batch-pure-phase-fence-off-20260827-attempt-02/
```

这是一项单轮 qualification，不是跨系统 equal-HBM 正式结果，也不能单独证明
fence-off 在多次独立进程中的稳定性。

## 1. 实验回答的问题

此前 HTTP/SSE client 虽然使用异步 barrier 近乎同时发送 C 个请求，但各连接仍有
约 0.5 ms 的实际提交间隔。调度器可能在第一个请求到达后立即开始一个 cohort，
导致 C4/C8/C16 实际变成 `1+3`、`1+7`、`1+15`。pure-phase scheduler 禁止下一
cohort 的 Prefill 与当前 cohort 的 Decode 重叠，因此其余请求必须等待第一个请求
完成，TTFT 被排队时间抬高到约 2.2--2.5 秒。

本实验要验证：

1. 在没有 HTTP/client 到达偏斜时，调度器能否形成精确的 C1/C4/C8/C16 单一
   measurement cohort；
2. pure-Prefill 后统一切换 pure-Decode 的执行是否成功；
3. 关闭 post-RESUME 只读 PTE fence 后，本轮是否仍能完成所有请求且无 CUDA、
   Xid、PTE/PDE fault；
4. 去掉 `1+(N-1)` 排队后，engine-level TTFT、TPOT 和 E2E 的大致趋势。

## 2. direct batch 的精确定义

本实验不是把 16 个请求从客户端“同时 POST”。执行路径为：

```text
加载冻结的 prompt_token_ids
  -> 为整个 cell 设置同一个 arrival_time
  -> 逐个调用 Processor.process_inputs
  -> 逐个调用 LLMEngine.add_request
  -> 确认 C 个请求全部加入
  -> 记录计时起点
  -> 第一次调用 LLMEngine.step
  -> pure Prefill cohort
  -> 统一 cutover 到 pure Decode
  -> 生成固定 32 token
```

`add_request` 在这里是向同一进程内 EngineCore 的直接 admission，不涉及 socket、
HTTP、JSON body、SSE 或独立客户端进程。虽然 C 个请求的 Python input processing
仍按顺序进行，但 engine 在全部请求加入前不会执行任何一步；因此 scheduler 首次
看到这些请求时，C 个请求已经全部 ready。

实现还检查首次 `step()` 返回的 request ID 集合必须恰好等于本 cell 的全部 request
ID。相关结果字段为：

```text
all_requests_added_before_first_step = true
first_step_request_count             = C
single_first_step_batch              = true
```

不要用普通 `LLM.generate(list_of_prompts)` 替换当前实现来采集本实验指标。
`LLM.generate` 会把输出模式改为 `FINAL_ONLY`，无法逐 token 记录 TTFT/TPOT。
当前 runner 显式使用 `RequestOutputKind.DELTA` 并逐步调用 engine。

## 3. 指标边界

本实验的计时起点是：完整批次 admission 完成、首次 EngineCore `step()` 开始之前。

- TTFT：计时起点到 engine 返回该批请求首 token；
- TPOT：首 token 后相邻 engine token step 时间的平均值；
- E2E：计时起点到第 32 个 token；
- admission duration：`process_inputs + add_request` 填充整个批次所需时间，单独记录，
  不计入 TTFT/E2E；
- SLO：short dialogue 的 TTFT <= 2000 ms 且 TPOT <= 100 ms。

由于同一 batch 的 token 在同一 engine step 返回，同一个 cell 内所有请求的
TTFT/TPOT/E2E 相同是正常现象。这些数字排除了客户端、HTTP、序列化、SSE 与 input
processing 开销，不能直接替换论文中使用统一 client 测得的用户可见 E2E 数据；
它们适合解释 scheduler/cohort 机制。

## 4. 冻结配置

| 项目 | 值 |
|---|---|
| 系统 | ours / in-process vLLM V1 EngineCore |
| 模型 | DeepSeek-V2-Lite-Chat BF16 |
| snapshot | `85864749cd611b4353ce1decdb286193298f64c7` |
| GPU | NVIDIA GeForce RTX 5090 32 GiB，GPU 0，TP=1 |
| CPU | `taskset -c 0-13`，OMP/MKL 14 threads |
| KV cache | 2,147,483,648 B；本次实际容量 69,040 tokens |
| cold plan | 每个 MoE 层 11 个 cold experts，固定 calibration placement |
| cold bytes | 4,948,230,144 B |
| hot bytes | 23,841,472,512 B |
| max model length | 8,320 |
| max batched tokens | 16,384 |
| max sequences | 16 |
| workload | short dialogue，128 prompt tokens / 32 output tokens |
| cells | C1、C4、C8、C16，各运行一次 |
| warmup | 每个 cell 前 3 个 C1 请求，每个输出 8 tokens |
| sampling | temperature=0，top_p=1，ignore_eos，min=max=32，seed=20260823 |
| attention | TRITON_MLA |
| CUDA Graph | `CompilationMode.NONE` + `CUDAGraphMode.FULL_SEPARATE` |
| Decode Graph | batch 1/4/8/16 |
| pure-Prefill Graph | tokens 128/512/1024/2048/4096/8192/16384 |
| prefix cache | disabled |
| phase scheduler | enabled，KV-aware FCFS cohorts |
| post-RESUME fence | disabled |

冻结环境文件：

```text
experiments/eurosys_moe_offload_evaluation/configs/e2_ours_2g_env.json
```

冻结 cold11 plan：

```text
experiments/eurosys_moe_offload_evaluation/inputs/generated/
eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/calibration_assets/
plans_adjusted_11_17_23_29/cold_plan_2g.json
```

关键环境变量应包括：

```text
VLLM_ENABLE_V1_MULTIPROCESSING=0
VLLM_UVM_ORIGINAL_VA_PHASE_MODE=prefill_graph_remote_decode
VLLM_UVM_PHASE_BATCHED_SCHEDULING=1
VLLM_UVM_PHASE_KV_AWARE_COHORTS=1
VLLM_UVM_PHASE_COHORT_KV_RESERVE_BLOCKS=1
VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1
VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=1
VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_BYTES=2147483648
VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_EMPTY_CACHE=1
VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES=201326592
```

必须不存在：

```text
VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE
```

注意：fence-off 只表示关闭 `VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE`。
`VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE` 和
`VLLM_UVM_FUSED_MOE_PRE_KERNEL_ALIAS_READ_GATE` 是初始化/捕获阶段的 fail-closed
正确性检查，不是本实验要关闭的 post-RESUME 只读 PTE fence，不要误删。

## 5. 为什么仍要保留 2 GiB native hole

2 GiB native-hole sentinel 是 warmup/初始化阶段的容量保护，不是 PTE fence。
launcher 在 KV 初始化前分配并触碰 2 GiB hole，迫使后续内存布局提前为 native
临时缓冲区与 Graph pool 留出连续空间；Decode Graph 捕获完成后释放 hole，再捕获
pure-Prefill Graph。

参考成功日志应包含：

```text
Allocated native UVM hole sentinel ... bytes=2147483648
Released native UVM hole sentinel ... free_delta=2147483648
Capturing CUDA graphs (decode, FULL) ... 4/4
Capturing CUDA graphs (pure prefill, FULL) ... 7/7
Restored original-VA serving residency ...
  cold_bytes=4948230144 hot_bytes=23841472512 free_bytes=376176640
```

最后一条 `Restored original-VA serving residency after sampler warmup` 就是正式请求前
的 residency reconciliation/rebalance：Graph 和 sampler warmup 结束后重新恢复
cold11/hot expert 布局，再允许第一个 `DIRECT_BATCH_CELL_START`。如果这条记录缺失，
或它出现在正式 measurement 之后，不能验收；否则 KV/Graph warmup 可能已经挤占热门
专家。

这个保护解决的是“正式请求期间首次创建临时/native 分配时驱逐热门专家”的问题。
它不能弥补总物理容量不相容；cold experts 数量、KV、Graph pool 与 hot experts 总和
如果超过 HBM，改变 sentinel 顺序或反复 rebalance 都不会减少总占用。本配置使用
cold11 是已经验证可容纳 2 GiB KV 与完整 Graph 集的组合。

## 6. 必需文件

执行入口：

```text
experiments/eurosys_moe_offload_evaluation/
run_direct_batch_phase_four_cells.sh
```

进程内 benchmark：

```text
experiments/eurosys_moe_offload_evaluation/
direct_batch_phase_runner.py
```

底层 UVM 生命周期 harness：

```text
experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh
```

冻结输入：

```text
experiments/eurosys_moe_offload_evaluation/inputs/generated/
eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/
evaluation_short_dialogue.json
calibration_short_dialogue.json
```

参考成功运行时，上述两个新脚本与主要输入的 SHA-256 为：

```text
72e2db64d330a432402271d3afc691fec268fe6f0a267a48cfd4554db8c5e58f  direct_batch_phase_runner.py
2623fe1ffdc6cd7173bf980ba6bc7e031eb7c3d01c9cb809616fd098a2494c1a  run_direct_batch_phase_four_cells.sh
e2aade89990115f977858fd736bd502a066e5b7023bbdd9fedada67767821ddb  configs/e2_ours_2g_env.json
3aec45dda0fe5a5e8b8aa99cdd216714206c25975f007a508616b79bca8cfe2e  evaluation_short_dialogue.json
337404dd625f5ca91748c5ace76ceb6adb6810de8d08959e83baddd1fba0e21c  calibration_short_dialogue.json
```

如果代码本来就有后续受控修改，hash 不同不等于错误；但必须在新 artifact 中保留
`input_sha256.txt` 与 `git_status.txt`，不能把不同代码版本的数据混为同一重复。

## 7. 运行前检查

从仓库根目录开始：

```bash
cd /home/ubuntu/nvidia-uvm-gpu

workloads/vllm/vllm/.venv/bin/python -m py_compile \
  experiments/eurosys_moe_offload_evaluation/direct_batch_phase_runner.py

bash -n \
  experiments/eurosys_moe_offload_evaluation/run_direct_batch_phase_four_cells.sh

nvidia-smi --query-gpu=index,name,memory.used,memory.free \
  --format=csv,noheader,nounits

nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits

sudo fuser /dev/nvidia-uvm /dev/nvidia-uvm-tools 2>/dev/null || true

modinfo -F srcversion nvidia-uvm
cat /sys/module/nvidia_uvm/srcversion
```

必须满足：

- GPU memory used 精确为 0 MiB；
- 无 compute process；
- `/dev/nvidia-uvm*` 无客户端；
- 当前加载的是 stock `nvidia_uvm`；
- vLLM Python、模型 snapshot、fixture、cold11 plan 和 custom UVM `.ko` 存在；
- 没有其他 E2/phase-switch 实验持有 GPU/UVM lock；
- `sudo`、`modprobe`、`insmod`、`journalctl`、`nvidia-smi`、`jq`、`flock`、
  `timeout`、`taskset` 可用。

launcher 自己还会重复检查这些条件、运行底层 8 个单测、停止可能占用 UVM 的
exporter、加载 custom UVM，并在退出时恢复 stock UVM。

## 8. 正式运行命令

必须选择一个尚不存在的新 artifact 目录。runner 拒绝覆盖已有结果：

```bash
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/\
run_direct_batch_phase_four_cells.sh \
  experiments/eurosys_moe_offload_evaluation/artifacts/\
e2-direct-batch-pure-phase-fence-off-REPRO-attempt-01
```

不要并行启动第二个 GPU 实验。参考机器上耗时约 4--8 分钟，绝大部分时间用于模型
加载、4 个 Decode Graph 和 7 个 pure-Prefill Graph 的捕获。benchmark stdout/stderr
定向写到 `run.log`，所以长时间没有终端输出不表示进程挂起。

launcher 在导入 vLLM 之前以两层方式关闭 fence：

```text
unset VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE
env -u VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE ... direct_batch_phase_runner.py
```

同时清除父 shell 中可能残留的
`FORMAL_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE`。不要只在启动命令前执行一次
`unset` 后就假定 fence 已关闭；环境 JSON 或父 launcher 仍可能重新注入变量。

## 9. 自动清理行为

脚本持有两个互斥锁：

```text
/tmp/eurosys_moe_offload_e2_gpu0.lock
/tmp/vllm_uvm_prefill_alias_decode_remote_graph.lock
```

正常或异常退出时，底层 trap 会：

1. 等待所有 GPU/UVM client 退出；
2. 卸载 custom `nvidia_uvm`；
3. 恢复 stock NVIDIA/UVM module；
4. 验证 GPU 回到 0 MiB；
5. 重启运行前被暂停的 exporter。

不要在 GPU 或 UVM client 尚存活时手工 `modprobe -r nvidia_uvm`。如果 timeout 后仍有
进程，先保存日志并精确终止 artifact 记录的进程，再等待 GPU clean，最后恢复 stock
module。

## 10. 结果文件

每次 artifact 至少包含：

| 文件 | 内容 |
|---|---|
| `command.txt` | 实际 direct runner 命令 |
| `environment.json` | 冻结运行环境；fence key 必须不存在 |
| `input_sha256.txt` | 代码、配置、fixture、model config、UVM module hash |
| `git_status.txt` | 工作区版本状态 |
| `result.json` | cell、逐请求指标、首次 step request IDs 和 SLO |
| `verification.json` | 自动验收结论 |
| `run.log` | vLLM、Graph、scheduler、PARK/RESUME 日志 |
| `phase_transition.json` | original-VA phase transition 与 `runtime_cold_read_gate` |
| `uvm.log` | UVM allocation/alias 记录 |
| `offload_manifest.jsonl` | 实际 offload manifest |
| `nvml.csv` | 200 ms NVML 采样 |
| `kernel.since.log` | 本轮内核日志 |
| `gpu.before.json` / `gpu.after.json` | GPU 生命周期证据 |
| `module.before.json` / `module.custom.json` / `module.after.json` | UVM module 生命周期证据 |
| `returncode.txt` | direct runner 返回码 |

## 11. 自动验收

设：

```bash
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e2-direct-batch-pure-phase-fence-off-REPRO-attempt-01
```

总体验收：

```bash
jq '.' "$RUN_ROOT/verification.json"
```

必须得到 `status=ok`，且所有 check 为 `true`：

```text
runner_returncode_zero
kernel_log_returncode_zero
result_status_ok
fence_absent
no_client
no_http_server
requested_cells
all_requests_succeeded
all_cells_single_first_step_batch
no_fatal_log_pattern
```

检查四个 cell 是否确实一次成批进入首次 step：

```bash
jq -e '
  .status == "ok" and
  [.cells[].concurrency] == [1,4,8,16] and
  ([.cells[] |
    .all_requests_added_before_first_step == true and
    .single_first_step_batch == true and
    .first_step_request_count == .concurrency and
    .successful_requests == .request_count] | all)
' "$RUN_ROOT/result.json"
```

检查没有 HTTP/client：

```bash
jq -e '
  .configuration.client_used == false and
  .configuration.http_server_started == false
' "$RUN_ROOT/result.json"
```

检查 fence 未注入：

```bash
jq -e '
  has("VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE") | not
' "$RUN_ROOT/environment.json"

jq -e '
  .configuration.runtime_post_resume_cold_read_gate_present == false
' "$RUN_ROOT/result.json"

jq -e '.transition.runtime_cold_read_gate == null' \
  "$RUN_ROOT/phase_transition.json"
```

`phase_transition.json` 中每次 transition 的 `runtime_cold_read_gate` 应为 `null`；
不能只凭没有打印 fence 文本判定关闭成功。

检查 scheduler measurement cohort：

```bash
rg -n \
  'DIRECT_BATCH_CELL_(START|DONE)|UVM phase-batched cohort (started|cutover|completed)' \
  "$RUN_ROOT/run.log"
```

在每个 `DIRECT_BATCH_CELL_START concurrency=C` 后，先出现 3 个 warmup
`requests=1`，随后 measurement 必须出现：

```text
C1  -> UVM phase-batched cohort started: phase=prefill requests=1
C4  -> UVM phase-batched cohort started: phase=prefill requests=4
C8  -> UVM phase-batched cohort started: phase=prefill requests=8
C16 -> UVM phase-batched cohort started: phase=prefill requests=16
```

不得把 warmup 的 `requests=1` 误认为 measurement 被拆批。应使用
`DIRECT_BATCH_CELL_START/DONE` 标记确定边界。

检查 fatal 日志：

```bash
rg -n -i \
  'CUDA error|illegal memory access|\bXid\b|PTE.*fault|PDE.*fault|FAULT_PDE|FAULT_PTE|Traceback' \
  "$RUN_ROOT/run.log" "$RUN_ROOT/kernel.since.log"
```

预期没有输出。再检查 mixed phase：

```bash
! rg -n -i 'mixed.*step|mixed prefill|prefill.*decode.*mixed' \
  "$RUN_ROOT/run.log"
```

上式应返回成功且不打印匹配，表示没有 mixed phase step。然后检查清理：

```bash
jq -e '
  .gpu_clean == true and
  .memory_used_mib == 0 and
  (.compute_pids | length) == 0 and
  (.uvm_client_pids | length) == 0
' "$RUN_ROOT/gpu.after.json"

jq -e '.gpu_clean == true' "$RUN_ROOT/module.after.json"
```

## 12. 成功结果

### 12.1 2026-08-31当前同模块复验

本轮实际artifact：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
e2-e3-e10-current-dual-view-REPRO-20260831-attempt-01/e2/
```

| Cell | 首次step | 成功 | TTFT p50 | TPOT p50 | E2E p50 | 相对最近参考E2E |
|---:|---:|---:|---:|---:|---:|---:|
| C1 | 1 | 1/1 | 542.88 ms | 26.40 ms | 1361.36 ms | -16.01% |
| C4 | 4 | 4/4 | 553.28 ms | 47.93 ms | 2038.99 ms | -9.60% |
| C8 | 8 | 8/8 | 562.55 ms | 65.96 ms | 2607.29 ms | -6.49% |
| C16 | 16 | 16/16 | 586.58 ms | 76.78 ms | 2966.62 ms | -5.81% |

这里的最近参考为
`e2-direct-batch-pure-phase-fence-off-REPRO-20260831-attempt-07-hybrid-deferred-shared`。
本轮无HTTP、无地址诊断、无`CUDA_LAUNCH_BLOCKING`，四个cell均为一次首步批量入队；
输出、Graph、内核日志和清理验收全部通过。实验结束后stock UVM恢复，GPU为0 MiB。

### 12.2 2026-08-27历史参考

参考有效 artifact 为 `e2-direct-batch-pure-phase-fence-off-20260827-attempt-02`：

| Cell | admission | 首次 step | 成功/SLO | TTFT | TPOT | E2E |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 0.249 ms | 1 | 1/1 | 539.21 ms | 33.76 ms | 1585.91 ms |
| C4 | 0.601 ms | 4 | 4/4 | 549.19 ms | 54.56 ms | 2240.64 ms |
| C8 | 1.067 ms | 8 | 8/8 | 562.66 ms | 72.45 ms | 2808.75 ms |
| C16 | 2.406 ms | 16 | 16/16 | 571.87 ms | 83.05 ms | 3146.47 ms |

合计 29/29 请求成功且满足 TTFT/TPOT SLO。NVML 峰值 HBM 为 31,753 MiB，
结束后 GPU 为 0 MiB，stock UVM 已恢复。

读取 NVML peak 时，当前 CSV 字段顺序为：

```text
timestamp, temperature, utilization.gpu, utilization.memory,
memory.used, memory.free, power, sm clock, memory clock, pstate
```

所以 `memory.used` 是第 5 列，不是第 4 列：

```bash
awk -F, '
  BEGIN { max=0 }
  { gsub(/ /,"",$5); if ($5+0>max) max=$5+0 }
  END { print max }
' "$RUN_ROOT/nvml.csv"
```

## 13. 容易出错的情况、原因与处理方法

### 13.1 实际运行了旧 benchmark，而不是 direct runner

症状：

```text
usage: benchmark_phase_switch_steps.py ...
error: the following arguments are required:
  --exact-names, --offload-manifest, --arm, --batch-size
```

原因：底层 `run_experiment.sh` 被 source 时定义自己的 `RUNNER` shell 变量。如果上层
launcher 也把 direct runner 保存为 `RUNNER`，source 后变量会被静默覆盖。2026-08-27
的无效 attempt-01 就是这一错误；它没有加载模型，也没有执行任何请求。

解决：上层必须使用隔离名称 `DIRECT_RUNNER`，并在 source 后仍引用
`${DIRECT_RUNNER}`。当前 launcher 已修复。不要重新把它改回通用 `RUNNER`。遇到该
错误时保留无效 artifact，选择新目录重跑，不能在原目录覆盖。

### 13.2 又出现 `1+3`、`1+7` 或 `1+15`

症状：scheduler 在 measurement 区间先记录 `requests=1`，随后才记录剩余请求；
高并发请求 TTFT 出现一快多慢两组。

原因通常是：

- 又使用了 HTTP streaming client；
- 在 admission 循环完成前调用了 `step()`；
- 改成多个 asyncio task，engine 在其他 task 尚未 add 时已经调度；
- 误读了每个 cell 前 3 个 C1 warmup 的日志。

解决：确认 `client_used=false`、`http_server_started=false`，检查
`first_step_request_count==concurrency` 和 `single_first_step_batch=true`。不要通过
增大人为 sleep 或放宽 scheduler window 来伪造同时到达；本实验要求所有请求在首次
step 前直接加入。

### 13.3 fence 实际没有关闭

症状：`environment.json` 含有
`VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE`，或 transition 中
`runtime_cold_read_gate` 不是 `null`。

原因：父 shell、formal launcher 或其他环境文件重新导出了变量；只在启动前手工
`unset` 一次不够可靠。vLLM 的环境常量还可能在 import 时缓存。

解决：保持 launcher 的“双重关闭”：载入环境后 `unset`，创建 Python 进程时再次
`env -u`，并保证 vLLM 在此后才 import。不得在 Python import vLLM 后再修改环境。

### 13.4 误把其他正确性 gate 当作 fence 一起关闭

症状：初始化阶段 hot residency 或 fused-MoE alias 错误没有 fail-closed，直到 kernel
异步报错才退出。

原因：把所有名称中含 `READ_GATE` 的变量都当成只读 PTE fence。

解决：只关闭 `VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE`。保留 hot CUDA read
gate、post-workspace residency audit 与 pre-kernel alias read gate。它们不在正式
post-RESUME 请求路径中承担本次 fence 的作用。

### 13.5 Graph 捕获或 hot restore 时显存不足

症状：Graph capture OOM、hot GPU bytes 小于 23,841,472,512、free HBM 不足，或在
首个 MoE kernel 前 fail-closed。

原因可能是：

- cold plan 不是 cold11，卸载专家太少；
- 2 GiB pre-KV hole 未分配/触碰/按正确时机释放；
- KV 大于 2,147,483,648 B；
- 捕获了额外 Graph shape；
- GPU 启动时不是 0 MiB；
- allocator/Graph cache 被提前清空，破坏了预留布局。

解决顺序：

1. 对照 `environment.json` 检查 cold11 plan、KV、sentinel 与 Graph sizes；
2. 检查 GPU/UVM client 和 exporter；
3. 在 `run.log` 中确认 2 GiB hole 分配/释放以及 Decode 4/4、Prefill 7/7；
4. 检查 capture 后 cold/hot/free 三项账目；
5. 清理并用新 artifact 重跑。

不要通过反复调 sentinel 顺序、删除 Graph、启用 eager、减少 KV 或临时增加 cold
experts 来把失败 run 计入本配置；这些都会改变实验定义。如果确需调整 cold/KV，
应冻结为新的实验 cell 并重新标注配置。

### 13.6 CUDA illegal access、Xid 或 PTE/PDE fault

症状可能包括：

```text
CUDA error: an illegal memory access was encountered
Xid 31
FAULT_PDE / FAULT_PTE
```

原因要区分：

- hot experts 被 KV/临时分配驱逐且未恢复；
- PARK/RESUME 后 consumer PTE 不可读；
- custom UVM module、slot budget 或 PTE remap 配置漂移；
- 进程退出后的异步 CUDA 错误只是更早 kernel fault 的后果。

处理：

1. 立即保留 `run.log`、`kernel.since.log`、`phase_transition.json`、`uvm.log`、
   module snapshots 和 manifest；
2. 先找第一个 CUDA/kernel fault，不要把随后出现的 traceback 当根因；
3. 核对 custom UVM 的 srcversion、slot budget 1,207,959,552 B、PTE remap=1 和
   manual fault mode=2；
4. 核对 hot restore、cold bytes、free bytes 和 PARK/RESUME 返回值；
5. 恢复 stock UVM，确认 GPU 0 MiB 后再用新 artifact 重跑；
6. 可以另跑 fence-on 对照来定位 PTE 可见性，但 fence-on 结果不能算作本
   fence-off arm 的成功结果。

不要简单删除 CUDA/PTE 检查或切换 eager，这会掩盖真实错误。此前 fence-off 路径
出现过非确定性异步 illegal access，因此一次成功只能说明本轮 qualification 通过；
至少需要多个独立进程重复才能声称稳定。

### 13.7 日志显示 16 个 started/cutover，但只有 15 个 completed

这不一定是失败。phase scheduler 的 `cohort completed` 信息在下一次 schedule 调用
开始时打印；最后一个 C16 完成后进程直接关闭，没有下一次调度，因此参考成功日志为
16 个 started、16 个 cutover、15 个显式 completed。

最终判断应以 C16 的 16/16 token 输出、runner return code、
`single_first_step_batch=true` 和无 fatal log 为准，不应仅按 completed 日志行数判错。

### 13.8 cell 内所有请求时间完全相同

这是 direct engine-level 计时的预期结果。一个 batch 的所有 token 在相同 engine step
返回，runner 给同一步内的请求记录同一时间戳。它不意味着每条请求在 HTTP 层拥有
完全相同的用户可见延迟。

如果需要连接建立、序列化、请求排队和 SSE 交付的差异，应使用统一 streaming client
运行正式 E2；但那是另一种计时边界，不能用它验证“首次 step 前 C 个请求全部 ready”。

### 13.9 运行结束后 GPU/UVM 未清理

症状：GPU memory used 非 0、compute PID/UVM client 仍在、module.after 仍是 custom
srcversion。

原因：Python 被强杀、子进程未退出、exporter 重新连接或手工中断发生在 module
切换边界。

解决：根据 artifact PID 与 `nvidia-smi`/`fuser` 精确定位残留进程，正常终止后等待
GPU clean，再恢复 stock module。不要删除 artifact；该 run 应标为 cleanup failure，
即使 29 个请求都成功也不能验收。

### 13.10 NVML peak 解析错误

当前 `nvml.csv` 的第 4 列是 `utilization.memory`，第 5 列才是 `memory.used`。列索引
错误会得到几十这样的不合理“MiB 峰值”。使用第 5 列，并核对结果应接近 32 GiB
而不是 GPU utilization 百分比。

## 14. 如何解释复现结果

如果新运行也满足 29/29、首次 step 为 1/4/8/16、无 mixed/fatal fault，可得出：

- direct admission 消除了 `1+(N-1)` cohort 拆分；
- 在这一轮 2 GiB/cold11 配置下，关闭 post-RESUME fence 仍能正确完成；
- TPOT 随并发增加而上升，而 TTFT 不再包含第二 cohort 等待第一个请求完成的时间。

不能据此得出：

- fence 在所有运行中都不需要；
- fence-off 已通过多进程稳定性验证；
- 这些 engine-level 指标可以直接替代正式 client-visible E2 指标；
- ours 已在 equal-HBM 跨系统矩阵中获胜。

若要把该结果提升为稳定性结论，应保持所有冻结参数不变，以多个独立 Python/GPU/UVM
生命周期重复运行，并同时报告成功率、分布和任何异步 PTE/CUDA fault。
