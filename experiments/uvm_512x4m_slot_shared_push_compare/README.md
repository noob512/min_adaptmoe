# 512×4 MiB slot-backed shared-push 对照

这个独立微基准不启动或导入 vLLM。正式配置对完全相同的 512 个独立
`cudaMallocManaged` range（每个 4 MiB、总计 2 GiB）比较三条同步路径：

- A `driver_4x128_sync`：现有 ioctl 90，每次 128 ranges，共四次；
- B `slot_per_block_push`：ioctl 91 mode 10，使用预分配 slot，但每个 VA block
  单独提交 CE push；
- C `slot_shared_push`：ioctl 91 mode 11，使用同一组预分配 slot，并在 pushbuffer
  实际容量允许的范围内让多个 block 共享 CE push。

B/C 都先在计时区外用 ioctl 91 mode 8 为每个 4 MiB range 创建一个独立 VIDMEM
slot。计时路径不允许重新分配、驱逐或退回普通 migrate。mode 10/11 直接从
managed CPU page/chunk 复制到 slot，不经过 `copy_from_user` staging，然后为原
managed VA 安装只读 slot PTE。因此消费地址在整个进程中不变。

## 严格边界

每个正式样本都是一个新 Python 进程，runner 使用三组逐对 ABBA：
`A-B-B-A`、`B-C-C-B`、`A-C-C-A`；下一 block 反向为 BAAB。每种模式每 block
获得四个正式样本。默认 full 跑两个 block，即每种模式八个独立进程样本。

每个 worker 在任何计时和 profile 基线之后立即执行以下前置条件：

1. 将全部 512 ranges prefetch 到 CPU 并同步；
2. 检查 512/512 `LastPrefetchLocation=CPU`；
3. 用 cmd 81/action 4 检查完整 2 GiB 都是 CPU residency、GPU/unknown 为零；
4. 再次核对所有 managed VA；
5. 连续读取 driver profile，以第二次快照作为计时基线。

四次 ioctl 都是同步完成语义；统一 e2e 时间是主机单调时钟从第一次调用前到
第四次调用返回。B/C 还从 `slot_direct_profile` 分别输出 copy 与 slot-map 时间；
这些 profile stage 可能嵌套，不能与 e2e 直接相加。A 只保留现有
`migrate_profile.copy_between_total/block_map/api_tracker_wait` 作为非同口径诊断，
不会把 CE 命令生成时间伪装成可与 B/C 比较的 copy 完成时间。

迁移后必须同时通过：

- 512/512 ranges、2 GiB requested/completed；
- 原 VA 上由 GPU 读取完整 2 GiB 的 checksum；
- 所有 VA 与初始快照一致；
- A 的普通 UVM residency 为 GPU 2 GiB；
- B/C 的普通 UVM residency仍为 CPU 2 GiB（slot PTE 不伪装成 UVM resident
  mask），但原 VA 的 GPU checksum 正确；
- B 为一 block 一 copy push，C 按真实 128 KiB pushbuffer 容量 flush，并报告
  ioctl 返回的实际 push 数；验收不把它硬编码成四次，是否真正减少由结果单独
  给出；两者复制的逻辑 bytes 相同；
- 计时区内 slot allocation、eviction、fallback、error 全为零，slot map bytes
  恰好 2 GiB。

`latency_stats` 是 module-global，所以运行期间不能有其他 UVM workload。runner
自身持有 advisory lock，并用 profile 的精确增量捕获外部污染。slot pool 的
VIDMEM budget 必须为 0（不限）或至少覆盖逻辑工作集；full 默认要求至少 2 GiB。

## 运行

只做小规模 ABI/正确性 smoke：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
PROFILE=smoke bash experiments/uvm_512x4m_slot_shared_push_compare/run_benchmark.sh
```

正式 512×4 MiB、每模式八个独立进程样本：

```bash
bash experiments/uvm_512x4m_slot_shared_push_compare/run_benchmark.sh
```

结果写入全新的时间戳目录，不覆盖历史结果。只复核一个已保存结果：

```bash
python3 experiments/uvm_512x4m_slot_shared_push_compare/verify_results.py \
  --require-full \
  experiments/uvm_512x4m_slot_shared_push_compare/artifacts/<run-id>
```

复核多个独立 run 并输出跨 run speedup 中位数时，把多个 artifact 目录依次传给
同一个 `verify_results.py` 命令。

## 驱动要求

需要配套实验模块提供：ioctl 90 原 batch、ioctl 91 mode 8/10/11，以及
`latency_stats` 中 versioned slot-direct profile/counters。旧模块会因为缺失
mode/counter 明确失败，不会静默生成可误读的性能数字。这个目录只包含
userspace benchmark，不负责构建、加载或切换 UVM 模块。
