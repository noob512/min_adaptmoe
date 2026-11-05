# Qwen3 ours/KTransformers 原结果复现方法

最后核验：2026-09-02。本文仍只复现2026-08-26历史结果。当前KTransformers工作区HEAD为
`5a9c7d4cfaaed0136930e4b629ce5d55cec71a54`，父仓库gitlink为
`8d9325af331d916721dfe7e2104f93f9b268678b`；在更新gitlink并重新运行前，不得把本文数字
表述为当前main上的新结果。

本文记录 2026-08-26 在单张 RTX 5090 上复现 Qwen3 ours/KTransformers
Decode 对比的方法。该实验用于复现旧结果和检查机制可运行性；输入专家路由来自
`legacy_pilot`，不能替代 EuroSys E3 的 v2 frozen-plan 正式结果。

## 1. 软件与模型

- 仓库：`/home/ubuntu/nvidia-uvm-gpu`
- ours Python：`workloads/vllm/vllm/.venv/bin/python`
- KT Python：`KTransformers/.venv/bin/python`
- 模型：`Qwen/Qwen3-30B-A3B-FP8`
- snapshot：`d206ba732169f29bb77fbf80fc2c4b81d4d30782`
- CPU：固定 `taskset -c 0-13`，`OMP_NUM_THREADS=14`，`MKL_NUM_THREADS=14`
- GPU：`CUDA_VISIBLE_DEVICES=0`
- 离线模式：`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`

运行前要求 GPU 无计算进程且显存使用为 0 MiB。ours 会加载 custom UVM，结束后必须恢复
stock UVM；KT 只在 stock UVM 下运行。

## 2. 冻结输入与专家计划

准备脚本：`prepare_inputs.py`

路由源：

`experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/routing/expert_distribution.pt`

路由源 SHA256：

`912732708a46291b73e70c93291758747f285b364437ce77f58d53501d072f81`

计划按每层 calibration Decode route count 从低到高选择专家，expert id 用于 tie-break：

| tier | 每层 cold experts | 总 cold 坐标 | main weight bytes |
|---|---:|---:|---:|
| 2 GiB | 10 | 480 | 2,264,924,160 |
| 4 GiB | 19 | 912 | 4,303,355,904 |

模型有 48 个 MoE 层、每层 128 个专家。exact-name 文件包含 96 个 fused expert Parameter：
每层 `w13_weight` 和 `w2_weight`。

准备命令由两个 matrix 脚本自动执行，也可单独执行：

```bash
workloads/vllm/vllm/.venv/bin/python \
  experiments/qwen_decode_six_mechanism_compare/prepare_inputs.py \
  --source experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/routing/expert_distribution.pt \
  --output-dir experiments/qwen_decode_six_mechanism_compare/inputs
```

生成后检查：

```bash
jq '.tiers."2g", .tiers."4g"' \
  experiments/qwen_decode_six_mechanism_compare/inputs/manifest.json
wc -l experiments/qwen_decode_six_mechanism_compare/inputs/exact_parameter_names.txt
```

## 3. ours 实验设置

矩阵：2/4 GiB × eager/Graph × B1/B4，共 8 个独立进程 case。

- eager arm：`R`，remote Prefill + remote eager Decode；
- Graph arm：`C`，remote Prefill + remote CUDA Graph Decode；
- warmup Decode：5 steps；
- measured Decode：16 steps；
- KV cache：167,772,160 B；
- `max_model_len=512`；
- `max_num_seqs=4`；
- `max_num_batched_tokens=2048`；
- Graph capture sizes：1、4；
- expert alias region：512 KiB；
- PTE mode：4 KiB；
- phase mode：`remote_only`；
- dynamic init/incremental/swap/warm-offload 全部关闭。

执行命令：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
ALLOW_GPU_RUN=1 \
RUN_ROOT=experiments/qwen_decode_six_mechanism_compare/artifacts/repro_ours_$(date -u +%Y%m%dT%H%M%SZ) \
TIERS=2g,4g BATCHES=1,4 ARMS=R,C \
WARMUP_DECODE_STEPS=5 MEASURED_DECODE_STEPS=16 \
bash experiments/qwen_decode_six_mechanism_compare/run_custom_matrix.sh
```

脚本逐 case 检查：返回码为 0、`result.json.status=ok`、GPU 清理成功。矩阵结束后恢复 stock
UVM 并写出 `module.after.json`。

## 4. KTransformers 实验设置

矩阵：2/4 GiB × eager/Graph，共 4 个 server case；每个 server 内测 B1/B4。

- KT method：FP8；
- CPU cores：14；threadpool count：1；NUMA node：0；
- cold coordinates 由同一个 JSON plan 转换成 KT logical-count placement；
- 2 GiB：118 GPU experts/layer、10 CPU experts/layer；
- 4 GiB：109 GPU experts/layer、19 CPU experts/layer；
- `max_deferred_experts_per_token=0`；
- `gpu_prefill_token_threshold=0`；
- attention：FlashInfer；FP8 GEMM：Triton；
- conditioning repeats：1；measurement repeats：3；
- warmup/measured Decode：5/16 steps；
- eager 使用 `--disable-cuda-graph`；Graph 使用 `--cuda-graph-bs 1 4`。

执行命令：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
ALLOW_GPU_RUN=1 \
RUN_ROOT=experiments/qwen_decode_six_mechanism_compare/artifacts/repro_kt_$(date -u +%Y%m%dT%H%M%SZ) \
TIERS=2g,4g MODES=eager,graph \
WARMUP_DECODE_STEPS=5 MEASURED_DECODE_STEPS=16 \
CONDITIONING_REPEATS=1 MEASUREMENT_REPEATS=3 \
bash experiments/qwen_decode_six_mechanism_compare/run_ktransformers_matrix.sh
```

## 5. 结果验证与汇总

参考成功产物：

- ours：`artifacts/repro_ours_20260826_attempt_01`
- KT：`artifacts/repro_kt_20260826_attempt_01`

ours 的 `execution.tsv` 应为 8 行且所有 return code 为 0；KT 应为 4 行且所有 return code
为 0。进一步检查：

```bash
awk -F '\t' 'NR>1 && $NF != 0 {bad++} END {exit bad!=0}' \
  experiments/qwen_decode_six_mechanism_compare/artifacts/repro_ours_20260826_attempt_01/execution.tsv

jq '.gpu_clean, .memory_used_mib, .module_path' \
  experiments/qwen_decode_six_mechanism_compare/artifacts/repro_ours_20260826_attempt_01/module.after.json
```

2026-08-26 复现的 Decode median（ms）：

| 配置 | B1 | B4 |
|---|---:|---:|
| ours 2 GiB eager | 59.23 | 59.30 |
| ours 2 GiB Graph | 9.26 | 14.50 |
| ours 4 GiB eager | 57.77 | 57.17 |
| ours 4 GiB Graph | 9.98 | 17.83 |
| KT 2 GiB eager | 132.96 | 294.58 |
| KT 2 GiB Graph | 31.78 | 131.97 |
| KT 4 GiB eager | 157.06 | 805.89 |
| KT 4 GiB Graph | 38.79 | 701.09 |

## 6. 解释限制

ours 数字来自同步 EngineCore step 边界；KT 数字来自 localhost HTTP/SSE server 路径。
因此可以据此确认旧结果可复现、Graph 命中和固定坐标生效，但不能直接将两列比值声明为
严格跨框架 speedup。论文级 E3 应使用 v2 fixture、评价前冻结计划、随机化顺序和多个独立
process block。
