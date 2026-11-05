# 复现 E3 KTransformers 调优后单轮 Prefill/Decode 对照

最后核验：2026-09-03（Asia/Shanghai）

## 范围与证据等级

模型为 Qwen3-30B-A3B-FP8，使用 E3 冻结的五类 token-ID 请求：Short
128→32、Ordinary 512→128、Medium 2048→128、RAG 8192→128、Long
512→512；每类测试 B1/B16。每个 cell 是独立冷进程并一次提交完整 batch，Prefill
和 Decode 分别计时。

当前正式结果只采用一个完整轮次，共 10 个冷进程。它是单轮点估计，没有重复样本、
方差、置信区间或显著性结论。被终止的 `repeat-2/repeat-2b` 部分 artifact 不进入结果。

## 固定配置

- `kt_cpuinfer=7`，整个进程的 CPU affinity 为物理核 `0-13`；
- frequency placement，逐 cell 使用 cold10/cold38/cold89；
- 默认 FlashInfer attention；RAG/B16 因 HBM 容量使用 TRITON；
- Decode CUDA Graph 开启，B1 捕获 B1，B16 捕获 B1/B16；Prefill 为 eager；
- overlap 开启，mixed chunk 关闭，threadpool count 为 1；
- 默认 `kt_gpu_prefill_token_threshold=0`；Medium/B16 为 16384，RAG/B16 为
  4096，这两个点使用 layerwise full-GPU Prefill fallback。

Medium/B16 的 fallback 与早期 hybrid qualification 的完整输出 hash 不同。因此这是
请求特定的系统调优，不是 exact-generation 单机制消融。

## 当前结果的离线复核（不会启动 GPU 实验）

```bash
cd /home/ubuntu/nvidia-uvm-gpu

python3 experiments/eurosys_moe_offload_evaluation/\
verify_e3_ktransformers_tuned_single_round.py \
  --kt-root experiments/eurosys_moe_offload_evaluation/artifacts/\
e3-ktransformers-tuned-final-20260903-repeat-1 \
  --native-uva-summary experiments/eurosys_moe_offload_evaluation/artifacts/\
e3-near-equal-offload-plus-one-expert-20260903-attempt-02/summary.json \
  --native-copy-summary experiments/eurosys_moe_offload_evaluation/artifacts/\
e3-near-equal-native-copy-graph-20260903-attempt-01/summary.json \
  --old-kt-verification experiments/eurosys_moe_offload_evaluation/artifacts/\
e3-ktransformers-c1-c16-cudagraph-20260902-attempt-01/verification.json \
  --output experiments/eurosys_moe_offload_evaluation/artifacts/\
e3-ktransformers-tuned-final-20260903-verification.json

jq -e '
  .status == "ok" and
  (.errors | length) == 0 and
  (.cells | length) == 10 and
  (.cells | all(.ktransformers.prefill_ms.count == 1 and
                .ktransformers.decode_ms.count == 1 and
                .ktransformers.output_hash_valid == true))
' experiments/eurosys_moe_offload_evaluation/artifacts/\
e3-ktransformers-tuned-final-20260903-verification.json
```

底层历史文件名 `verify_e3_ktransformers_tuned_three_round.py` 只为保持已有 artifact
命令兼容而保留；当前入口和底层验证器都强制只接受一个 `--kt-root`。

## 从头运行一个新单轮（当前结果不需要执行）

```bash
cd /home/ubuntu/nvidia-uvm-gpu

PREFIX=experiments/eurosys_moe_offload_evaluation/artifacts/\
e3-ktransformers-tuned-single-round-$(date -u +%Y%m%dT%H%M%SZ)

ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/\
run_e3_ktransformers_tuned_single_round.sh "$PREFIX"
```

历史 runner 文件名同样仅为兼容而保留；当前脚本只串行运行 10 个冷进程和一次聚合验证，
不会运行三轮。

## 验收与比较边界

每个 cell 必须通过：固定请求形状、完整输出长度和 hash、Decode Graph 捕获/回放、48 层
专家 placement、无 Xid/MMU/PDE/PTE fault、无 cleanup error。当前完整轮次 10/10 通过，
运行结束后 GPU 为 0 MiB。

vLLM Native UVA 与 copy 对照是同模型、同冻结请求和同 B1/B16 坐标，均为单冷进程；
但 vLLM 使用 `TRITON_ATTN + FULL_SEPARATE` 和 whole-Parameter offload，KTransformers
默认使用 FlashInfer、按 expert 放置并在 CPU 执行冷专家。三者实际卸载字节近似配平，
但并非同一机制、同一 attention backend 或同一数值路径。因此只允许做系统级同请求
描述性比较，不得称作单变量消融；Native copy 的 RAG/B16 容量失败必须保留为 N/A。
