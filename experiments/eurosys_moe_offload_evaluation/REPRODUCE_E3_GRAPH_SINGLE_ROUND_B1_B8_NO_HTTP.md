# 复现 E3 CUDA Graph B1/B8 单轮 direct-batch

## 范围

本流程只跑 B1/B8、一个独立冷进程块，不经过 HTTP。每个 batch 在第一个
EngineCore step 前一次性入队，Prefill 完成后才进入 Decode，不允许 mixed step。

常规 cell 跑 G/B/C Decode-Graph 三臂；`prompt×batch>=8192` 的大 Prefill
cell 跑已认证的 D Prefill+Decode 双 Graph qualification。当前入口只为 G/D
建立“当前固定 cold 档位”的 Prefill/Decode 双页表；C 始终远程读取，B 始终
使用缓存路径，不建立双页表，也不执行阶段页表切换。本流程不预建多档页表。

## 构建

自定义 UVM 必须由当前源码重新构建。实际 Makefile 位于
`kernel-module/nvidia-module`：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module
make -j"$(nproc)" modules

cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test
make
```

## 前置检查

```bash
cd /home/ubuntu/nvidia-uvm-gpu
nvidia-smi --query-gpu=memory.used --format=csv,noheader
ps -eo pid,cmd | rg 'run_formal_e3|benchmark_phase_switch_steps'
modinfo nvidia_uvm | rg 'filename|version'
test -f /home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782/config.json
```

启动前要求 GPU=0 MiB、没有旧 runner，且 UVM filename 指向系统 stock module。

## 冻结 cold63 计划

仓库已包含运行所需的冻结计划：

```text
experiments/eurosys_moe_offload_evaluation/inputs/frozen/e3_qwen_cold63_20260828/cold_decode_plan_2g.json
experiments/eurosys_moe_offload_evaluation/inputs/frozen/e3_qwen_cold63_20260828/exact_parameter_names.txt
```

如需从相同校准输入重新派生并核对，执行：

```bash
workloads/vllm/vllm/.venv/bin/python \
  experiments/eurosys_moe_offload_evaluation/freeze_qwen_e3_plan.py \
  --source experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/routing/expert_distribution.pt \
  --calibration-fixture experiments/eurosys_moe_offload_evaluation/inputs/generated/eurosys-pa-v2-20260824/qwen3_30b_a3b_fp8/calibration_rag_summary.json \
  --output-dir experiments/eurosys_moe_offload_evaluation/artifacts/e3-derived-plan-cold63-20260828 \
  --calibration-batch 32 --expected-decode-tokens 224 --experts-per-layer 63
```

必须验证：

```text
routing source SHA256 = 912732708a46291b73e70c93291758747f285b364437ce77f58d53501d072f81
plan SHA256           = 05686bd2c4df0f5c052c90902e5945bee9d921a8d36b7b1009712dd314cc8620
selected main bytes   = 14269022208
backing bytes          = 594542592
```

## 一键执行

输出目录必须不存在：

```bash
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/run_e3_graph_single_round_b1_b8_no_http.sh \
  experiments/eurosys_moe_offload_evaluation/artifacts/e3-graph-repro-$(date -u +%Y%m%dT%H%M%SZ)
```

launcher为每组写入`provenance/*.command.txt`。核心配置：

| Cells | Arm | cold/层 | KV | Graph sizes | reserve tokens |
|---|---|---:|---:|---|---:|
| Short/Ordinary/Long B1/B8 | G/B/C | 10 | 精确计算，最低160 MiB | B1或B8 Decode | 0 |
| Medium/B1 | G/B/C | 10 | 205.5 MiB | B1 Decode | 0 |
| Medium/B8 | D | 38 | 1,723,858,944 B | 8,16384 | 16384 |
| RAG/B1 | D | 38 | 819,462,144 B | 1,8192 | 8192 |
| RAG/B8 | D | 63 | 6,555,697,152 B | 8,65536 | 20480 |

launcher 默认传入 `E3_CURRENT_TIER_DUAL_VIEW=1`，并只对 G/D 增加：

```text
VLLM_CPU_OFFLOAD_PLAN_CLASS_IMMUTABLE_CPU_SHADOW=1
VLLM_CPU_OFFLOAD_DUAL_VIEW_FULL_PARAMETER_SHADOW=1
VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW=1
VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1
VLLM_UVM_RUNTIME_POST_MAP_RESIDENCY_AUDIT=0
```

验收时 G/D 必须满足 `dual_view_full_coverage=true` 且
`remote_range_map_ioctl_calls=0`。B/C 的 `dual_view_environment.txt` 必须为空，
以证明其原执行路径未改变。

大 Prefill D 必须同时设置：

```text
VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1
VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE=1
VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=1
VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES=201326592
VLLM_FUSED_MOE_CHUNK_SIZE=4096
VLLM_FLASHINFER_AUTOTUNE_MAX_TOKENS=4096
PRE_KV_NATIVE_HOLE_SENTINEL=auto
```

不要删除 PARK/RESUME 或 capture-before-hot-restore；不要把65K reserve重新当成
安全默认值。RAG/B8 的20,480-token warmup实际约1.020 GiB，正式65K Prefill由
Graph replay覆盖。

## 汇总验证

将五个成功根目录传给：

```bash
python3 experiments/eurosys_moe_offload_evaluation/summarize_e3_graph_single_round.py \
  --require-current-tier-dual-view \
  --output-dir <summary-dir> \
  <tier2-main-root> <medium-b1-root> <medium-b8-D-root> <rag-b1-D-root> <rag-b8-D-root>
```

验收必须为：

```text
status=ok
expected_cells=10, successful_cells=10
expected_runs=24, successful_runs=24
successful_compare_cells=7
successful_dual_graph_only_cells=3
errors=[]
current_tier_dual_view_required=true
```

并人工确认 GPU=0 MiB、stock UVM恢复、无 CUDA illegal access/Xid/PTE/PDE
fault。大 Prefill缺E/F对称arms和多冷进程重复，不能把本流程描述为论文级正式E3。

## 2026-08-31当前同模块复验

实际采集目录：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
e2-e3-e10-current-dual-view-REPRO-20260831-attempt-01/e3/
```

该轮使用custom UVM SHA256
`c8fe77c8ecc87f6d724d94ddf6535c934150d4e12bec384bd44562097aec043a`。24/24 run、
10/10 cell成功，聚合`verification.json`为`status=ok`，winner为C×6、G×1、B×0。

若复用共同父目录，实际命令为：

```bash
REPRO_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
e2-e3-e10-current-dual-view-REPRO-20260831-attempt-01

ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/\
run_e3_graph_single_round_b1_b8_no_http.sh \
  "$REPRO_ROOT/e3"

python3 experiments/eurosys_moe_offload_evaluation/\
summarize_e3_graph_single_round.py \
  --require-current-tier-dual-view \
  --output-dir "$REPRO_ROOT/e3/summary" \
  "$REPRO_ROOT/e3/groups/tier2_main" \
  "$REPRO_ROOT/e3/groups/tier2_medium_b1" \
  "$REPRO_ROOT/e3/groups/dual_tier8_medium_b8" \
  "$REPRO_ROOT/e3/groups/dual_tier8_rag_b1" \
  "$REPRO_ROOT/e3/groups/dual_cold63_rag_b8"
```

上述目录已经存在；新复现必须换用不存在的新attempt目录，不能覆盖本轮数据。

D大Prefill结果：

| 场景 | Prefill | P→D | steady Decode中位 | E2E |
|---|---:|---:|---:|---:|
| Medium/B8 | 650.94 ms | 24.43 ms | 42.07 ms | 6232.72 ms |
| RAG/B1 | 586.80 ms | 23.61 ms | 16.00 ms | 2879.94 ms |
| RAG/B8 | 3692.33 ms | 45.91 ms | 64.28 ms | 12094.85 ms |

相对最近同协议双页表参考，24点请求E2E差异中位数为-0.16%，steady Decode差异中位数
为-0.01%，Prefill差异中位数为-0.76%，没有观察到实质回退。G/D双表覆盖率100%，
CPU range-map ioctl为0，Graph无重捕获，输出完全一致；五个group均恢复stock UVM和
GPU 0 MiB。
