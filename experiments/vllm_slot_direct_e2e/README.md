# vLLM 原 VA slot-direct PyTorch E2E

这个最小 harness 验证 vLLM 接入前最关键的一条真实路径，而不是重复完整 2 MiB
微基准：

- 通过 vLLM 的 process-local UVM `MemPool` 创建一个真实 PyTorch managed fused
  allocation；
- 从同一 allocation 切出相邻的 512 KiB expert envelope，每个都是非完整 2 MiB
  VA block；
- mode 8 在计时区外预分配 VIDMEM slot；
- 将 envelope 普通 residency 放到 CPU；
- mode 10 从 CPU backing 复制到 slot，并安装 4 KiB slot PTE；
- 继续使用原来的 `torch.Tensor` 和原 `data_ptr()` 执行 CUDA GEMM 与全量求和；
- 输出 JSON，并严格验证 mode 10 timed path 中 allocation、eviction、fallback、
  error 均为零。

相邻 envelope 会共享 2 MiB VA block。驱动当前拒绝在同一个 direct ioctl 中重复
同一 VA block，因此 runner 会按 VA block 自动着色分批；JSON 保存精确 batch
indices 和每块 envelope 数量。这也是 vLLM 集成层必须遵守的提交规则。

## 安全边界

脚本不会加载、卸载或替换内核模块。它要求调用者已经加载当前 custom UVM，且：

```text
loaded srcversion == kernel-open/nvidia-uvm.ko srcversion
uvm_moe_slot_pte_remap_enable == 1
```

driver context 只采用 CUDA 进程已经打开并初始化的 `/dev/nvidia-uvm` fd，并先用
ioctl 90 对真实 managed VA 做归属验证；没有创建无关 VA-space 的 fallback。

slot backing 是 module-global 实验状态。完成整组实验后应按项目既有安全流程恢复
stock UVM；不要并行运行其他 UVM workload。

## 运行

custom 模块已由上层安全切换后：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
PROFILE=smoke \
  bash experiments/vllm_slot_direct_e2e/run_harness.sh
```

正式 128×512 KiB（64 MiB）torch 路径：

```bash
PROFILE=full \
  bash experiments/vllm_slot_direct_e2e/run_harness.sh
```

结果位于新的时间戳 `artifacts/<run-id>/result.json`。独立复核：

```bash
workloads/vllm/vllm/.venv/bin/python \
  experiments/vllm_slot_direct_e2e/verify_result.py \
  experiments/vllm_slot_direct_e2e/artifacts/<run-id>/result.json
```

## 验收合同

成功结果必须同时满足：

- 所有 envelope 为 512 KiB、原 VA 前后完全一致；
- mode 8 和 mode 10 每条 entry 均成功，accepted bytes 等于 logical bytes；
- mode 10 profile 的 allocation/eviction/fallback/error 全为 0；
- slot 映射前后普通 UVM residency 保持 CPU（slot PTE 不伪装成普通 GPU
  residency）；
- 原 Tensor 的 GEMM 输出逐 bit 等于 GPU-resident baseline；
- 所有 weight checksum 相等；
- slot GEMM 后仍没有触发普通 UVM migration。

完整 2 MiB existing-64K overwrite 仍由
`experiments/uvm_slot_64k_overwrite_stage_sync_compare` 覆盖，它是性能对照，不是
当前 vLLM 512 KiB envelope 接入的前置门槛。

## mode 19 生命周期与同进程复用

仅验证一次 mode 10 不足以证明 slot 可以长期使用。mode 19 成功后，内核 owner
和 userspace policy 必须同时提交释放；否则 `_SLOT_DIRECT_READY` 残留会让下一次
restore 错误返回 `already_slot_ready`，跳过真正的 mode 8/mode 10。

CPU-only 合同测试会 mock 全部 ioctl，并使用同一 2 MiB VA block 内的两个
512 KiB region 执行：

1. policy mode 8 + mode 10 restore 两个 region，确认 mode 10 被着色为两个 ioctl；
2. mode 19 只释放第一个 region，第二个 slot PTE 继续存活；
3. 检查第一个 region 已从 `READY/PREALLOCATED/BINDINGS/SLOT_OWNERS` 清除，而
   第二个完全保留；
4. 同进程再次 restore 第一个 region，必须重新出现 mode 8 + mode 10，不能返回
   `already_slot_ready`；
5. 最终释放全部 region，并确认 policy 与低层 owner table 均为空。

运行该静态测试不会启动 CUDA workload：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
PYTHONPATH=workloads/vllm/vllm \
  workloads/vllm/vllm/.venv/bin/python \
  experiments/vllm_slot_direct_e2e/test_slot_policy_lifecycle.py
```

真实 GPU lifecycle runner 另外验证 mode 19/PTE 行为。`first-then-rest` 是关键
场景：先释放同 block 的第一个 512 KiB slot，在相邻 slot 仍映射时把该 range
prefetch 回 CPU，再于同一进程重新 preallocate/restore，最后释放全部 slot：

```bash
workloads/vllm/vllm/.venv/bin/python \
  experiments/vllm_slot_direct_e2e/slot_python_lifecycle.py \
  --regions 8 --release-pattern first-then-rest \
  --output /tmp/vllm_slot_lifecycle_partial.json

workloads/vllm/vllm/.venv/bin/python \
  experiments/vllm_slot_direct_e2e/slot_python_lifecycle.py \
  --regions 8 --release-pattern all \
  --output /tmp/vllm_slot_lifecycle_all.json
```

这两个命令同样不会切换内核模块；必须由外层先确认 custom UVM 已加载，且不要与
其他 UVM workload 并发。成功 artifact 必须包含
`same_process_reuse=true`、`low_level_owner_state_clean=true`、稳定 VA 和相同
checksum。完成后仍由外层恢复 stock UVM。
