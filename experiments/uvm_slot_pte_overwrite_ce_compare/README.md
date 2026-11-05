# Slot PTE 粒度、直接覆盖与 CE 完成时间对照

已完成的三个 clean 2 GiB 正式 run、被排除的干扰轮次及跨轮结论见
`RESULTS_FULL_20260816.md`。

这是一个不启动、不导入 vLLM 的独立 UVM 微基准。正式实验固定使用 512 个彼此
不连续的 4 MiB `cudaMallocManaged` region（相邻测试 region 之间插入 4 MiB guard，
总逻辑数据 2 GiB），每次 ioctl 传 128 个 region，共四次。五条路径使用完全相同
的数据、slot backing 和原始虚拟地址：

- `slot_4k_baseline`：ioctl 91 mode 10，4 KiB PTE 基线；
- `slot_64k`：mode 12，以 64 KiB PTE 写入相同 slot backing；
- `slot_2m`：mode 13，以 2 MiB PTE 写入相同 slot backing；
- `direct_overwrite_existing_grain`：mode 14，不先独立 unmap，按当前 PTE 粒度直接
  覆盖；实际采用的 4 KiB/64 KiB/2 MiB block 数由 telemetry 报告，不预设；
- `stage_sync_4k_ce`：mode 15，仍为 4 KiB，但在 prepare、copy、map 三个阶段边界
  分别同步，用来区分 CPU 命令提交时间与设备完成等待时间。

所有模式都先在计时区外用 mode 8 为每个 region 分配 4 MiB VIDMEM slot。正式
计时区禁止 slot allocation、eviction、fallback 和普通 migrate；每个 4 MiB
region 必须 2 MiB 对齐并完整覆盖两个 VA block。因此 full 样本严格对应 1024 个
VA block，smoke 样本对应 8 个 VA block。

## 时间口径

统一性能指标 `wall_ms` 是主机单调时钟从第一次 ioctl 前到第四次同步 ioctl 返回。
`prepare_ms`、`unmap_ms`、`copy_ms`、`map_ms` 是各阶段的 CPU/命令提交开销；
它们不是 CE 完成时间。mode 14 跳过独立 unmap，因此 `unmap_ms=0`，其他模式
必须记录非零 unmap 时间。`wait_ms` 是通常路径末尾的 tracker 等待。

只有 mode 15 会产生 `prepare_complete_ms`、`copy_complete_ms` 和
`map_complete_ms`：每个阶段提交后立即同步，再记录完成边界。这条诊断路径会把
每个 ioctl 的 tracker wait 从 1 次提高到 4 次，所以不把 mode 15 的 e2e 性能
当作优化路径；它用于定位等待究竟发生在 prepare/unmap、CE copy 还是 PTE
write/map/TLB invalidate。其他四个模式的三个 completion profile stage 必须全为
零，结果 JSON 中对应值为 `null`。

PTE 写入量通过以下精确计数验收：4 KiB block 每块 512 个 entry，64 KiB block
每块 32 个 entry，2 MiB block 每块 1 个 entry。mode 14 的
`slot_direct_pte_write_entries` 必须等于三个实际粒度 block 数的加权和。

## 正确性与隔离

每个测量样本都在新的 Python 进程中运行。五种模式组成十个无序 pair，每个 pair
执行一次 ABBA 或 BAAB，所以一个 schedule block 有 40 个独立进程样本、每种模式
恰好 8 个样本。默认 full 和 smoke 都运行一个 schedule block。

mode 8 prealloc 后，所有模式还会在计时区外统一执行一次
`CPU reset → 全量 GPU checksum → CPU reset → ioctl-81 action-5 remote READ map`。
最后一步以 `async=0` 同步补齐缺失的 GPU READ 映射，同时保持 CPU residency；
不能依赖 CPU prefetch 偶然保留旧 GPU PTE。这样 mode 14 即使在 smoke（零
warmup）中也有可覆盖的旧 PTE；其他四种模式使用完全相同的起点。每次 warmup 和
正式计时前也重复 `CPU reset → action-5`。action-5 本身不会清理已有
WRITE/ATOMIC PTE，且其 `resident_cpu_bytes` 字段是候选字节而非真实 residency，
所以 worker 仍独立查询完整 CPU residency，mode 14 还会在
内核中硬性检查 1024 个 block 的 READ 全覆盖、CPU sole-resident 和 same-grain，
并由本实验要求 1024 个粒度命中、零 fallback，因此前置映射缺失时样本会失败，
不会被当成有效性能结果。

每个 worker 在 profile 基线和计时之前立即执行：

1. 将全部 managed source prefetch 回 CPU 并同步；
2. 验证所有 `LastPrefetchLocation=CPU`；
3. 通过 residency query 验证全部字节在 CPU、GPU/unknown 为零；
4. 再次核对原始 VA；
5. 连续读取两次全局 profile，确认边界没有变化。

计时后必须同时满足：requested/completed bytes 完整；copy push 和 memcopy method
均为 VA block 数；原 VA 上的全量 GPU checksum 正确；VA 不变；普通 UVM
residency 仍为 CPU（slot PTE 不伪装成正常 GPU residency）；allocation、eviction、
fallback、error 全为零；`migrate_profile.api_total.calls=0`，且 ioctl-90 的所有
counter 增量为零。

`latency_stats` 是 module-global。实验期间不能并行运行其他 UVM workload；runner
会持有本实验的 advisory lock，并用精确 counter 增量拒绝受污染样本。

## 运行

先运行只含 4 个 4 MiB region、仍完整覆盖 2 MiB VA block 的 smoke：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
PROFILE=smoke bash experiments/uvm_slot_pte_overwrite_ce_compare/run_benchmark.sh
```

正式 512×4 MiB、每模式 8 个独立进程样本：

```bash
bash experiments/uvm_slot_pte_overwrite_ce_compare/run_benchmark.sh
```

结果只写入本目录下新的时间戳 artifact，不覆盖旧实验。复核正式结果：

```bash
python3 experiments/uvm_slot_pte_overwrite_ce_compare/verify_results.py \
  --require-full \
  experiments/uvm_slot_pte_overwrite_ce_compare/artifacts/<run-id>
```

可一次传入多个 artifact；verifier 会从 raw samples 重算所有中位数、P95、speedup
和 PTE entry reduction，再输出跨 run 的 speedup 中位数及 CE 完成带宽。若同一
artifact 还保存了匹配的 2 GiB、512-range pinned-copy 结果，可加入：

```bash
python3 experiments/uvm_slot_pte_overwrite_ce_compare/verify_results.py \
  --require-full \
  --pinned-result experiments/uvm_slot_pte_overwrite_ce_compare/artifacts/<run-id>/pinned_copy_2g_512x4m.txt \
  experiments/uvm_slot_pte_overwrite_ce_compare/artifacts/<run-id>
```

verifier 会额外验收 pinned-copy 的布局、字节数、range 数、checksum 状态和带宽
计算，并输出 `ce_vs_pinned`。

## 驱动要求

需要配套实验模块提供 ioctl 91 mode 8/10/12/13/14/15，以及
`slot_direct_profile` 的 `prepare_complete`、`copy_complete`、`map_complete` 和以下
flat counters：

```text
slot_direct_pte_4k_blocks
slot_direct_pte_64k_blocks
slot_direct_pte_2m_blocks
slot_direct_pte_write_entries
slot_direct_skipped_unmap_blocks
slot_direct_stage_sync_calls
```

旧模块或字段不完整的模块会明确失败，不会静默生成可误读结果。本目录只构建
userspace shim，不负责构建、加载、卸载或切换内核模块。
