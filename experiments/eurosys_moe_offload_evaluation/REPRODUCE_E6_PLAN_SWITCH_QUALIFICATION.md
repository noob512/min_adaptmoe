# 复现 E6 Plan Class、serial wave 与切换摊销 qualification

## 范围

本流程复现三个独立冷进程 arm，每臂只运行一次完整 sequence，不做统计重复：

```text
base -> d005 -> base -> d010 -> base -> d025 -> base
     -> d050 -> base -> d100 -> base
```

三臂分别为 full session/Graph rebuild、serial wave，以及保留 Graph/session 的全层 batched
difference transaction。输出目录必须不存在，失败 attempt 必须原样保留。

## 冻结配置

| 参数 | 值 |
| --- | --- |
| 模型 | DeepSeek-V2-Lite-Chat BF16 |
| 请求 | B1，Prompt 512，Output 8 |
| KV | 268435456 B |
| slot budget | 10737418240 B |
| C8 | 26 层 × 19 experts = 494 slots |
| selected/backing | 8546942976 / 657457152 B |
| Graph | `FULL_SEPARATE`，`[1,512]` |
| seed/difference | 20260828；5/10/25/50/100% |

base plan SHA256 必须为：

```text
2800b757a7093a2f5214ae30bb05f4a00ec2a95e2501b127047ee835e36c2a42
```

## 前置检查

```bash
cd /home/ubuntu/nvidia-uvm-gpu
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
sudo fuser -v /dev/nvidia-uvm /dev/nvidia-uvm-tools 2>/dev/null || true
modinfo nvidia_uvm | rg 'filename|version'
sha256sum experiments/hybrid_offload_2_4_6_8_compare/inputs/cold_plan_8g.json
```

开始前要求 GPU 为 0 MiB、无 compute/UVM client，且 `nvidia_uvm` 指向系统 stock module。
三臂应顺序运行；每臂结束先确认 stock UVM 和 GPU clean，再启动下一臂。

## 单样本完整运行

### 1. Batched difference

```bash
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/e6-difference-REPRO-<tag>-qualification \
MODE=qualification PROCESS_BLOCKS=1 ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/run_formal_e6_difference.sh
```

### 2. Full rebuild

```bash
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/e6-full-rebuild-REPRO-<tag>-qualification \
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/run_e6_full_rebuild_qualification.sh
```

full rebuild 必须保留 `VLLM_PHASE_SWITCH_CONTROLLED_HARD_EXIT_AFTER_CLEANUP=1`。verifier 同时
要求 reference、distributed、Graph/UVM cleanup 已完成；不能用 hard exit 掩盖清理失败。

### 3. Serial wave

```bash
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/e6-serial-wave-REPRO-<tag>-qualification \
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/run_e6_serial_wave_qualification.sh
```

生成的 wave 数必须为 1/2/5/10/19。不要从 `block_01` 重写父 `RUN_ROOT` 下的 plan path。

## 离线聚合与 break-even

聚合步骤不触发 GPU，可以对三个成功目录重复执行，但 output 文件必须不存在：

```bash
python3 experiments/eurosys_moe_offload_evaluation/aggregate_e6_qualification.py \
  --difference-root <successful-difference-root> \
  --full-rebuild-root <successful-full-rebuild-root> \
  --serial-wave-root <successful-serial-wave-root> \
  --output <new-summary-root>/summary.json
```

汇总器从相邻 raw records 计算：

```text
denominator_ms = previous_request_ms - current_request_ms
requests_to_amortize = transition_ms / denominator_ms
```

分母不大于 0 时写入 `break_even_status=no_break_even`，且 requests 为 `null`。不得跨独立
进程拼接 previous/current request，也不得把 qualification verifier 的 count=1 median
解释为统计 median。

## 验收

```bash
jq '{status,qualification_scope,output_sha256,statistical_claims}' \
  <new-summary-root>/summary.json
jq '.sources | map_values({returncode,verifier_status,artifact_gpu_after_clean,
  artifact_stock_uvm_restored,fatal_kernel_signatures})' \
  <new-summary-root>/summary.json
```

必须满足：

```text
summary status=ok
qualification_scope=single_process_single_sample_per_transition
三臂 returncode=0、verifier_status=ok
output_sha256=a365db717e87663b5f807b34fc6013b44f98cd925bea9480cece8e65eac690b9
artifact_gpu_after_clean=true
artifact_stock_uvm_restored=true
fatal_kernel_signatures=[]
full capture count 2 -> 22
serial/batched capture count 2 -> 2
```

最后再次执行前置检查中的 GPU、client 和 module 查询。只有三臂都闭合以上契约，才能将
参数视为已冻结；正式重复应另建新目录和单独协议，不能覆盖 qualification artifacts。
