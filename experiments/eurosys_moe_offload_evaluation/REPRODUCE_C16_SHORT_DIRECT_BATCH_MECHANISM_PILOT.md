# 复现 Short-dialogue C16 无 HTTP 单批跨机制 Pilot

## 1. 目标

在相同的 16 条 Short-dialogue 请求上比较六个机制。正式请求不经过 HTTP/client；
所有 16 条请求先完成内存内 admission，再开始首个 scheduler step，从而排除
`1+15` 到达偏斜。

这是单轮可行性 pilot。默认 launcher 不执行重复，也不声称 equal-HBM。

## 2. 关键文件

```text
experiments/eurosys_moe_offload_evaluation/
  run_c16_short_direct_batch_pilot.sh
  direct_batch_c16_vllm_runner.py
  direct_batch_c16_ktransformers_runner.py
  direct_batch_c16_moe_infinity_runner.py
  verify_direct_batch_c16_case.py
  configs/c16_short_mechanism_runtime_hot_audit_off_pilot.json
  configs/c1_native_common_env.json
  configs/c1_ours_runtime_hot_audit_off_env.json
  configs/e2_ktransformers_env.json
  configs/e2_moe_infinity_env.json
```

MoE-Infinity 还依赖修复后的：

```text
MoE-Infinity/moe_infinity/serving/model_runner.py
```

launcher 会把这些关键输入的 SHA256 写到 artifact provenance。

## 3. 冻结配置

| 项目 | 值 |
|---|---|
| model | DeepSeek-V2-Lite-Chat BF16 snapshot `858647...` |
| evaluation | Short-dialogue 前 16 条 |
| prompt/output | 每条 128/32 tokens |
| admission | 一次完整 C16 cohort |
| warmup | 3 条单请求，每条 output=8 |
| CPU | cores 0--13，14 threads |
| seed | 20260823 |
| vLLM KV | 100,663,296 B / 3,232 tokens |
| max model len | 192（KT 为 194，保留内部槽） |
| max batched tokens | 2,048 |
| ours cold plan | 每层 1 expert，冻结 frequency placement |
| ours Graph | Decode 1/4/8/16；Prefill 2,048 |
| MoE KV | 192 blocks / 3,072 tokens |
| MoE ratios | device 0.80；KV-within-device 0.0145625 |

## 4. 启动前检查

从仓库根目录执行：

```bash
cd /home/ubuntu/nvidia-uvm-gpu

nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
nvidia-smi --query-gpu=memory.used --format=csv,noheader
lsmod | rg '^nvidia_uvm'
flock -n /tmp/eurosys_c1_mechanism_runtime_hot_audit_off.lock true
flock -n /tmp/eurosys_moe_offload_e2_gpu0.lock true
```

预期：无 compute process、GPU used memory 为 0 MiB、加载 stock UVM、两把 lock
均可获取。不要在已有 GPU 实验运行时启动本 launcher。

检查模型、fixture、虚拟环境、MoE offload store 和 receipt：

```bash
test -f /home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7/config.json
test -f experiments/eurosys_moe_offload_evaluation/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/evaluation_short_dialogue.json
test -x workloads/vllm/vllm/.venv/bin/python
test -x KTransformers/.venv/bin/python
test -d /home/ubuntu/moe_infinity_offload/deepseek_v2_lite_chat_85864749cd611b4353ce1decdb286193298f64c7
```

## 5. 先运行定向单测

```bash
workloads/vllm/vllm/.venv/bin/python -m py_compile \
  MoE-Infinity/moe_infinity/serving/model_runner.py \
  experiments/eurosys_moe_offload_evaluation/direct_batch_c16_ktransformers_runner.py \
  experiments/eurosys_moe_offload_evaluation/verify_direct_batch_c16_case.py

PYTHONPATH=/home/ubuntu/nvidia-uvm-gpu/MoE-Infinity \
  workloads/vllm/vllm/.venv/bin/python -m pytest -q \
  MoE-Infinity/tests/python/serving/test_model_runner.py
```

预期为 `5 passed`。

## 6. 一键运行

必须使用一个不存在的新 artifact 目录，launcher 会拒绝覆盖：

```bash
bash experiments/eurosys_moe_offload_evaluation/run_c16_short_direct_batch_pilot.sh \
  experiments/eurosys_moe_offload_evaluation/artifacts/c16-short-direct-batch-REPRO-attempt-01
```

顺序为 all-GPU、pinned-UVA、copy、ours、KTransformers、MoE-Infinity。同一时刻
只运行一个 GPU arm。ours 前加载 custom UVM，结束后立即恢复 stock UVM；其余
机制使用 stock UVM。

不要在 launcher 正在运行时编辑 launcher/runner。Bash 可能在后续读取脚本时混入
新版本，产生不可解释的半旧半新 attempt。

## 7. 如何证明是真正的 1 x 16

vLLM/ours/MoE-Infinity 的 `result.json` 必须满足：

```text
measurement.request_count = 16
measurement.successful_requests = 16
measurement.all_requests_added_before_first_step = true
measurement.first_step_request_count = 16
measurement.single_first_step_batch = true
```

KTransformers 必须在 measurement dispatch marker 之后出现：

```text
Prefill batch, #new-seq: 16, #new-token: 2048, ... #running-req: 0
```

`running-req=0` 是正确值：它表示本步前的旧请求数。若出现
`new-seq=1` 后再出现 15，则是 `1+15`，必须判失败。

每个 case 的 `verification.json` 八项检查都应为 true。

## 8. 各机制的空间预留

### 8.1 native vLLM

all-GPU 使用 0 offload。pinned-UVA/copy 使用最小正预算
`cpu-offload-gb=0.01`。96 MiB KV 按最大 `16 x 192` token 上界预留。

### 8.2 ours

1. 按 routing trace 生成每层 cold1 plan。
2. 权重加载期间建立 original-VA alias。
3. KV 初始化前建立 96 MiB native hole sentinel。
4. 保持专家 all-remote 完成 Decode/Prefill Graph 和 workspace warmup。
5. pure-Prefill capture 前释放 hole。
6. 恢复 hot experts，正式 Prefill 后 PARK，映射 cold experts 到 CPU，进入 Decode。

必须保留：

```text
VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_BYTES=100663296
VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS=2048
VLLM_UVM_FUSED_MOE_POST_WORKSPACE_AUDIT_MIN_TOKENS=2048
VLLM_FUSED_MOE_CHUNK_SIZE=2048
VLLM_UVM_RUNTIME_HOT_RESIDENCY_AUDIT=0
```

必须删除/不设置 runtime PTE fence：

```text
VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE
FORMAL_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE
```

关闭的是 28.34 GiB hot 全量审计；cold residency、stable VA、PARK/RESUME 和
错误检查不能删除。

### 8.3 KTransformers

直接构造 SGLang args 时必须传 `kt_numa_nodes=[0]`，不能传整数 0。CLI 的类型
规范化不会自动作用于 Python direct API。

### 8.4 MoE-Infinity

绝对预算公式为：

```text
absolute KV ratio     = device_ratio * kv_cache_ratio
absolute expert ratio = device_ratio * (1 - kv_cache_ratio)
```

本轮目标 KV 为 192 blocks：

```text
device_ratio = 0.80
kv_cache_ratio = 0.0145625
absolute KV ratio = 0.01165
```

若调整 device ratio，必须反算 KV ratio，不能让 KV blocks 降到 191。启动后以
`initial_runtime_stats.kv_cache_num_blocks == 192` 为准。

non-paged runner 必须使用预分配 packed logits 修复。若代码仍为
`packed_logits.append(logits)` 加最终 `torch.cat`，C16 会额外产生 800 MiB copy，
即使 ratio 0.80 也会 OOM。

## 9. 常见故障

### KTransformers 启动时报 int has no len

检查 `kt_numa_nodes`，应为 `[0]`。修复后重新启动整个 KT 冷进程，不复用半初始化
状态。

### 验证器说 KT 首批失败但 new-seq=16

检查 verifier 是否错误要求 `running-req=16`。当前日志语义下，全新 C16 cohort 为
`new-seq=16,running-req=0`。

### MoE KV 只有 191 blocks

不要降低 expected gate，也不要减少请求。按绝对 KV 公式提高内部 KV ratio，直到
实际 blocks 恢复 192。

### MoE 在 50/800 MiB logits 处分配失败

先检查是否包含预分配修复。已有边界证明：0.83/0.829/0.827 不足；修复后的
0.80 成功。不要通过分成 16 个独立 measurement 批次绕过，因为那会改变本实验的
`1 x 16` 目标。

### GPU 为 0 MiB 但 UVM 不能切换

```bash
sudo fuser -v /dev/nvidia-uvm /dev/nvidia-uvm-tools
ps -fp <pid>
readlink /proc/<pid>/fd/1
```

只有在 cmdline 和 FD 明确证明进程是本次失败 artifact 的孤儿 NVML sampler 时，
才定向 `SIGTERM`。不要杀死来源不明的 UVM client，不要跳过 stock-module gate。

## 10. 验收和清理

```bash
find experiments/eurosys_moe_offload_evaluation/artifacts/c16-short-direct-batch-REPRO-attempt-01 \
  -name verification.json -print -exec jq -r '.status' {} \;

nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
nvidia-smi --query-gpu=memory.used --format=csv,noheader
lsmod | rg '^nvidia_uvm'
flock -n /tmp/eurosys_c1_mechanism_runtime_hot_audit_off.lock true
flock -n /tmp/eurosys_moe_offload_e2_gpu0.lock true
```

六个最终 verification 均须为 `ok`。结束后必须无 compute process、GPU 0 MiB、
stock UVM、两把 lock 可获取。任何失败 attempt 都保留完整 command、run.log、
result、NVML、kernel log 和 verification，不覆盖、不计入最终均值。
