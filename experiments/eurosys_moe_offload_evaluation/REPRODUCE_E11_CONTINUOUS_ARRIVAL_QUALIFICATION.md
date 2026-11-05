# 复现 E11 continuous-arrival qualification

## 前提

仓库根目录为 `/home/ubuntu/nvidia-uvm-gpu`，模型和冻结输入必须存在。运行前必须满足：

```text
loaded nvidia_uvm srcversion == modinfo -F srcversion nvidia-uvm
GPU memory.used == 0 MiB
无 compute process
无 /dev/nvidia-uvm client
```

构建当前模块和用户态库：

```bash
make -C kernel-module/nvidia-module -j"$(nproc)" modules
make -C workloads/vllm/vllm/uvm_test
```

## 静态检查

```bash
workloads/vllm/vllm/.venv/bin/python -m py_compile \
  experiments/eurosys_moe_offload_evaluation/e11_continuous_arrival_runner.py \
  experiments/eurosys_moe_offload_evaluation/verify_e11_continuous_arrival.py

bash -n \
  experiments/eurosys_moe_offload_evaluation/run_e11_continuous_arrival_qualification.sh

workloads/vllm/vllm/.venv/bin/python -m pytest -q \
  workloads/vllm/vllm/tests/v1/core/test_scheduler.py \
  -k phase_batched_scheduler
```

## GPU 运行

必须使用新的、不存在的 attempt 目录：

```bash
ALLOW_GPU_RUN=1 \
RUN_ROOT=/home/ubuntu/nvidia-uvm-gpu/experiments/eurosys_moe_offload_evaluation/artifacts/\
e11-continuous-arrival-NEW-attempt-01-qualification \
  experiments/eurosys_moe_offload_evaluation/\
run_e11_continuous_arrival_qualification.sh
```

launcher 顺序运行三个独立冷进程：

- `phase_epoch`：phase-specific + phase epoch，必须 8/8 成功且 mixed step 为 0；
- `remote_continuous`：remote-only eager + default scheduler，必须 8/8 成功且至少一个 mixed step；
- `hybrid_fail_closed`：phase-specific + default scheduler，必须在 mixed step 明确失败。

每个 arm 都保存命令、环境、结果、内核日志、GPU before/after 和资源使用。最后 launcher
恢复 stock UVM；任一 Xid、页表 fault、脏 GPU 或无法恢复模块都算失败。

## 验证

```bash
workloads/vllm/vllm/.venv/bin/python \
  experiments/eurosys_moe_offload_evaluation/verify_e11_continuous_arrival.py \
  <RUN_ROOT> --output <RUN_ROOT>/verification.json
```

当前 verifier 还要求两个成功 arm 的完整 token IDs 相同。2026-08-31 attempt-02 没有通过
该 correctness 门，因此只能引用调度行为和 fail-closed 结果，不能声称正式 E11 已完成。
