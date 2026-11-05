# 复现原生 vLLM Native UVA CUDA Graph 的 E2/E3 单轮实验

日期：2026-09-03（Asia/Shanghai）

## 1. 实验范围与当前状态

本协议只运行原生 vLLM 的 pinned Native UVA arm，不运行 ours、copy-on-forward、
all-GPU、KTransformers 或 MoE-Infinity。每个配置只测量一次，所有请求均使用既有 E2/E3
冻结输入、随机种子、模型 snapshot 和容量坐标。

当前权威成功轮为：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e2-e3-native-uva-cudagraph-20260903-attempt-04/
```

聚合 `verification.json` 为 `status=ok`：E2 C1 4/4、E2 C16 4/4、E3 B1/B16 10/10
通过；无 Xid、MMU、PDE/PTE fault，结束时 GPU 为 0 MiB。E2 的8点与旧 eager Native
UVA 输出 hash 全部一致，E3 的10点与最新 D/ours 输出 hash 全部一致。

CUDA Graph 覆盖必须按模型分别解释：

- E2/DeepSeek-V2-Lite-Chat 使用 `FULL_DECODE_ONLY`：Prefill 为 eager，Decode 为 Full
  Graph replay；8/8点均由运行时审计确认 Decode `runtime_mode=FULL`，没有 Decode eager
  fallback。
- E3/Qwen3-30B-A3B-FP8 使用 `FULL_SEPARATE`：pure-Prefill Graph 与 uniform-Decode
  Graph 分开捕获和回放；10/10点的 Prefill、Decode 均由运行时审计确认
  `runtime_mode=FULL`。

不得把 E2 写成 Prefill/Decode 双 Graph。Native UVA 下的 DeepSeek Triton MLA 只有
Short/128 的 Prefill Graph可运行；Ordinary/512及以上在真实 Prefill replay发生
`illegal memory access`，详见第8节。

## 2. 冻结环境

```text
repo       /home/ubuntu/nvidia-uvm-gpu
GPU        NVIDIA GeForce RTX 5090 32 GiB，GPU 0，TP=1
driver     595.71.05
kernel     6.13.12-061312-generic
CPU        taskset 0-13，14个CPU核心
timezone   Asia/Shanghai
```

模型 snapshot：

```text
/home/ubuntu/.cache/huggingface/hub/
  models--deepseek-ai--DeepSeek-V2-Lite-Chat/
  snapshots/85864749cd611b4353ce1decdb286193298f64c7/

/home/ubuntu/.cache/huggingface/hub/
  models--Qwen--Qwen3-30B-A3B-FP8/
  snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782/
```

本实验不使用 custom UVM 的双页表、PARK/RESUME 或 P→D 切换。运行前应加载系统 stock
`nvidia-uvm.ko`，并确保没有遗留的 compute/UVM client。

## 3. 实现入口

一键入口：

```text
experiments/eurosys_moe_offload_evaluation/
  run_e2_e3_native_uva_cudagraph.sh
```

该入口依次调用：

1. `run_c1_mechanism_runtime_hot_audit_off_matrix.sh`，只运行 E2 C1 Native UVA；
2. `run_e2_c16_direct_batch_matrix.sh`，只运行 E2 C16 Native UVA；
3. `run_e3_native_copy_vs_uva_phase.sh`，只运行 E3 B1/B16 Native UVA；
4. `verify_e2_e3_native_uva_cudagraph.py`，聚合功能、Graph dispatch、故障和清理门。

新增的 `VLLM_CUDAGRAPH_DISPATCH_AUDIT=1` 会在首次观察到每类调度时记录
`requested_tokens`、`uniform_decode`、实际 `runtime_mode` 和 `dispatched_tokens`。验收依赖
实际 dispatch 记录，不只检查命令行配置。

## 4. 静态检查

```bash
cd /home/ubuntu/nvidia-uvm-gpu

bash -n \
  experiments/eurosys_moe_offload_evaluation/run_e2_e3_native_uva_cudagraph.sh \
  experiments/eurosys_moe_offload_evaluation/run_c1_mechanism_runtime_hot_audit_off_matrix.sh \
  experiments/eurosys_moe_offload_evaluation/run_e2_c16_direct_batch_matrix.sh \
  experiments/eurosys_moe_offload_evaluation/run_e3_native_copy_vs_uva_phase.sh

workloads/vllm/vllm/.venv/bin/python - <<'PY'
import ast
from pathlib import Path

for name in (
    "experiments/eurosys_moe_offload_evaluation/direct_batch_c16_vllm_runner.py",
    "experiments/eurosys_moe_offload_evaluation/verify_direct_batch_c16_case.py",
    "experiments/eurosys_moe_offload_evaluation/verify_e2_e3_native_uva_cudagraph.py",
    "experiments/vllm_p3_pinned_uva_phase_compare/benchmark_phase_steps.py",
    "workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py",
):
    ast.parse(Path(name).read_text(encoding="utf-8"), filename=name)
print("syntax: ok")
PY
```

实现时的相关 CUDA Graph 静态回归共13项通过。重新运行源码测试时，应使用 vLLM 子模块
自己的虚拟环境，并保存完整 pytest 输出到新的 provenance 目录。

## 5. GPU运行前检查

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

要求 GPU 为0 MiB、无 compute process、无未知 UVM client，且 loaded srcversion 与
stock srcversion相同。

## 6. 单轮正式运行

输出目录必须不存在，禁止覆盖成功或失败的 attempt：

```bash
cd /home/ubuntu/nvidia-uvm-gpu

RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e2-e3-native-uva-cudagraph-$(date -u +%Y%m%dT%H%M%SZ)

ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/\
run_e2_e3_native_uva_cudagraph.sh "$RUN_ROOT"
```

该命令内部冻结为：

```text
E2 C1/C16   arm=native_pinned_uva
             graph=FULL_DECODE_ONLY
             Prefill=eager，Decode=Full Graph

E3 B1/B16   arm=uva
             attention=TRITON_ATTN
             graph=FULL_SEPARATE
             conditioning_repeats=1
             measurement_repeats=1
             Prefill=Full Graph，Decode=Full Graph
```

E2 C1为Short/Ordinary/RAG/Long四点；E2 C16使用冻结的
`e2_c16_single_attempt_matrix.json`。E3为Short、Ordinary、Medium、RAG、Long的
B1/B16十点，完整batch一次提交，不拆batch，不启用mixed chunk。

## 7. 验收

```bash
jq -e '
  .status == "ok" and
  .checks.e2_c1_4_of_4_ok == true and
  .checks.e2_c16_4_of_4_ok == true and
  .checks.e3_10_of_10_ok == true and
  .checks.no_kernel_fault == true and
  ([.records[].checks[]] | all)
' "$RUN_ROOT/verification.json"

awk -F '\t' 'NR > 1 && $2 != 0 {bad=1} END {exit bad}' \
  "$RUN_ROOT/execution.tsv"

! rg -n -i \
  'Xid [0-9]|illegal memory access|FAULT_PDE|FAULT_PTE|GPU page fault|kernel BUG' \
  "$RUN_ROOT/kernel.since.log"

nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
```

E2 每个 record 必须同时满足 `prefill_eager_by_protocol=true`、
`decode_graph_captured=true`、`decode_graph_replayed=true` 和
`no_decode_eager_runtime_dispatch=true`。E3 每个 record 还必须满足
`prefill_graph_captured=true`、`prefill_graph_replayed=true` 和
`no_prefill_eager_runtime_dispatch=true`。

期望为18/18 record通过、所有 section returncode为0、故障扫描无输出、GPU最终0 MiB。

## 8. 已知失败路径与边界

必须保留以下失败证据，不能删除或并入成功计数：

```text
artifacts/e2-e3-native-uva-cudagraph-20260903-attempt-01/
artifacts/e2-e3-native-uva-cudagraph-20260903-attempt-02/
artifacts/e2-e3-native-uva-cudagraph-20260903-attempt-03/
```

- attempt-01暴露 `FULL_SEPARATE` 捕获尺寸分区错误：Prefill dummy run预期 eager，却被
  dispatch为Full Graph。当前实现已将 uniform-Decode 与 Prefill capture key分开。
- attempt-02/03修正捕获拓扑并使用真正 pure-Prefill capture 后，DeepSeek Short/128
  可以运行，但 Ordinary/512及更长的 Triton MLA Prefill在Graph replay发生非法访问。
- 因而正式attempt-04对E2使用`FULL_DECODE_ONLY`，不是用 eager fallback掩盖失败；E3
  仍合法使用`FULL_SEPARATE`。

当前结果是每点一个冷进程、一次测量的qualification。不得把点估计写成均值分布、p95、
置信区间或统计等价。E2与旧eager的速度比较是跨轮比较；E3与D/ours虽然冻结输入和完整
输出hash一致，但实际whole-Parameter卸载量/placement不同，不是equal-HBM或纯Graph消融。

## 9. 结果入口

- [独立结果文档](../eurosys_moe_offload_evaluation/RESULTS_E2_E3_NATIVE_UVA_CUDAGRAPH_20260903.md)
- [统一结果汇总](../eurosys_moe_offload_evaluation/RESULTS_E0_E12_LATEST_CONSOLIDATED_20260831.md)
- `artifacts/e2-e3-native-uva-cudagraph-20260903-attempt-04/verification.json`

