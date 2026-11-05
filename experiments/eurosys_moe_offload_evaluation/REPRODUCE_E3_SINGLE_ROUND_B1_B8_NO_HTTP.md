# 复现 E3 B1/B8 单轮无 HTTP 实验

## 目标

在固定 Qwen3 30B FP8 模型、480 个专家坐标和 B1/B8 下比较 H/A/R，并运行 KTransformers direct-batch 对照。所有请求在进程内提交，不启动 HTTP server；完整 batch 在第一次执行前入队，Prefill 与 Decode 不混合。

## 前置检查

```bash
cd /home/ubuntu/nvidia-uvm-gpu
nvidia-smi --query-gpu=index,name,memory.used,memory.total,driver_version --format=csv
modinfo -F filename nvidia_uvm
test -f experiments/eurosys_moe_offload_evaluation/artifacts/e3-calibration-20260825T091247Z/calibration_assets/cold_decode_plan_2g.json
test -f /home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782/config.json
bash -n experiments/eurosys_moe_offload_evaluation/run_e3_single_round_no_http.sh
python3 -m py_compile \
  experiments/eurosys_moe_offload_evaluation/direct_batch_e3_ktransformers_runner.py \
  experiments/eurosys_moe_offload_evaluation/verify_formal_e3.py
```

启动前必须满足 GPU 无 compute process、显存接近 0 MiB。不要复用已有输出目录。

## 执行命令

```bash
ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/run_e3_single_round_no_http.sh \
  experiments/eurosys_moe_offload_evaluation/artifacts/e3-repro-YYYYMMDD-attempt-N-single-round-no-http
```

launcher 固定传入：

```text
PROCESS_BLOCKS=1
BATCH_FILTER=1,8
RUN_KT=0   # 基础 runner 的 HTTP KT 路径关闭；外层 launcher 随后运行 direct KT
```

不得把 B8 拆成 `1+7`，不得以 B4 等较小 batch 代替容量失败。custom runner 的 scheduler 断言要求第一个 step 为完整 Prefill，后续 step 为完整 Decode；KT 设置 `enable_mixed_chunk=False`，并对每个 B1/B8 cell 只调用一次 `Engine.generate`。

## 固定参数

- H：`VLLM_UVM_ORIGINAL_VA_PHASE_MODE=prefill_remote_decode`；
- A：phase off，eager alias/copy；
- R：`VLLM_UVM_ORIGINAL_VA_PHASE_MODE=remote_only`；
- custom attention：TRITON_ATTN；PTE：4k；
- launcher 会传入候选 Graph sizes 1、4、8，但当前 H/A/R contract 均为 eager：
  `enforce_eager=true`、`cudagraph_mode=NONE`，这些 sizes 不会被实际捕获；
- warmup Decode：5 steps；
- H/A/R 的专家坐标和 selected bytes 必须完全一致；
- KTransformers：118 GPU experts/layer、10 CPU experts/layer、FlashInfer attention、Triton FP8 GEMM、Graph off、mixed chunk off。

注意 Python Engine API 字段为：

```python
fp8_gemm_runner_backend="triton"
```

`fp8_gemm_backend` 是错误的 Python keyword，会在模型加载前触发 `TypeError`。

## N/A 规则

- custom：`prompt * batch >= 8192` 时 fail-closed；
- custom KV 上限：2,400,000,000 B；
- KT KV 上限：2,800,000,000 B；
- 不允许为绕过 N/A 而改变 batch。

## 验证

```bash
RUN=experiments/eurosys_moe_offload_evaluation/artifacts/e3-repro-YYYYMMDD-attempt-N-single-round-no-http
KTransformers/.venv/bin/python \
  experiments/eurosys_moe_offload_evaluation/verify_formal_e3.py \
  --profile qwen3 --batches 1,8 --kt-mode direct "$RUN"
jq . "$RUN/analysis/verification.json"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
modinfo -F filename nvidia_uvm
```

合格条件：verifier 为 `status=ok`、`errors=[]`，所有非 N/A 请求输出完整，H/A/R 同 cell hash 相同，无 HTTP marker、CUDA/Xid/PTE/PDE fault，session/backing 全部释放，stock UVM 恢复且 GPU 为 0 MiB。

若目标是与开启 CUDA Graph 的 E2 做直接比较，不能复用这组 H/A/R eager 结果。
现有 runner 中 G/C/D 是 Graph arm；还需为 always-copy 定义等价 Graph arm，并统一
capture、warmup 和计时边界后重新运行。
