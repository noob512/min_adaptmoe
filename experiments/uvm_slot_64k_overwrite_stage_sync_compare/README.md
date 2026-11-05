# Slot 预分配下的 existing-64K overwrite / stage-sync 对照

这是一个不导入、不启动 vLLM 的独立 UVM 微基准。它在 slot backing 已经分配好、
计时路径不允许 allocation 或 eviction 的条件下，回答两个更窄的问题：

1. 已有 CPU remote READ PTE 严格为 64 KiB 时，跳过单独 unmap 并直接覆盖 PTE，
   能否降低 2 GiB 离散搬运的端到端时间；
2. 主机提交很快但端到端仍慢时，等待主要暴露在 prepare、copy 还是 map 的完成边界。

跨 VA block 的 shared-push 合批由相邻实验
`experiments/uvm_512x4m_slot_shared_push_compare` 验证。本目录有意保留“一块一个
copy push”，专门隔离 slot 预分配、64 KiB PTE 元数据以及独立 unmap 的代价，不能
把这里的结果解释成 shared-push 本身的收益。

## 固定路径与数据集

正式配置是 512 个互不相邻的 4 MiB `cudaMallocManaged` region，共 2 GiB；每两个
测试 region 之间至少有 4 MiB guard。每次 ioctl 91 传 128 个 region，共四次。
smoke 配置是 4 个 4 MiB region，每次传一个，仍固定四次 ioctl。每个 region 均按
2 MiB VA block 对齐。

四个正式模式只改变 PTE 更新和同步边界：

| 名称 | ioctl 91 mode | 独立 unmap | 同步边界 |
|---|---:|---|---|
| `normal_64k_e2e` | 12 | 有 | 每个 ioctl 末尾一次 |
| `overwrite_64k_e2e` | 17 | 无，严格覆盖 existing-64K | 每个 ioctl 末尾一次 |
| `normal_64k_stage_sync` | 16 | 有 | prepare/copy/map 后各一次，末尾一次 |
| `overwrite_64k_stage_sync` | 18 | 无，严格覆盖 existing-64K | prepare/copy/map 后各一次，末尾一次 |

所有模式先在计时区外用 mode 8 为每个逻辑 slot 预分配 4 MiB VIDMEM backing。
正式计时必须观察到：

- full 为 1024 个 2 MiB VA block，smoke 为 8 个；
- 每个 block 一个 copy push 和一个 memcopy method；
- 全部 block 使用 64 KiB PTE，full 写 32768 个 PTE entry；
- slot allocation、eviction、fallback、error 均为零；
- 普通 migrate 和 ioctl 90 fast-prefetch 计数均为零。

这组条件把 UVM 自动显存分配/驱逐排除在 timed path 外。如果仍然缓慢，结果才能
进一步用于区分 copy 完成等待与 PTE/锁/metadata 路径，而不是把时间误归因给临时
分配或 eviction retry。

## 严格 existing-64K 前置状态

每次 warmup 和每个正式样本之前都严格执行：

```text
CPU reset → ioctl 81 action 5 → ioctl 81 action 20
```

- CPU reset 将全部普通 UVM residency 放回 CPU，同步并复核
  `LastPrefetchLocation=CPU`、原 VA 不变；
- action 5 补齐 GPU 对 CPU backing 的 remote READ 映射；
- action 20 把完整 2 MiB remote PTE 原地拆为 32 个普通 64 KiB PTE；已经全为
  64 KiB 的 block 只验证。4 KiB、混合粒度、非 READ-only 或非 CPU
  sole-resident 状态都会失败；严格 overwrite 也会拒绝任何拆分后仍非 64 KiB
  的状态，不允许静默 fallback。

action 20 的 userspace 合同是：`split_2m_to_64k_bytes + already_64k_bytes ==
logical_bytes`，`validated_64k_bytes == logical_bytes`，`unknown_bytes == 0`。full
配置因 ioctl 81 每次最多 64 range，action 5 和 action 20 各调用八次。profile
基线只在 action 20 完成以后读取，所以前置 PTE 拆分不进入正式计时。

## 时间口径

主指标 `wall_ms` 是主机单调时钟从第一次 ioctl 91 前到第四次同步 ioctl 返回。
`prepare_ms`、`copy_ms`、`map_ms` 和 `wait_ms` 来自内核 profile。
`unmap_ms` 是 `prepare_ms` 内部的子区间，不能把两者相加；normal 模式必须非零，
overwrite 模式必须为零。

普通 e2e 模式中的 `copy_ms` 只是 copy push 的主机提交时间，设备完成通常暴露在
最后的 `wait_ms`，因此不能用 `copy_ms` 计算 CE 带宽。stage-sync 模式额外给出：

- `prepare_complete_ms`；
- `copy_complete_ms` 和由其计算的 `copy_complete_gib_s`；
- `map_complete_ms`；
- `copy_complete_minus_submit_ms`。

这些 completion 指标包含该阶段的提交、tracker 聚合和同步等待，是“驱动可见的
阶段完成时间”，不是纯硬件 CE 时间。stage-sync 每个四批样本应有 16 次 tracker
wait；非 stage-sync 应为 4 次，且 completion 字段必须为 `null`、对应 profile
计数为零。stage-sync 会主动破坏流水重叠，只用于诊断，不作为最终吞吐优化路径。

### 细粒度驱动闭合（schema v2）

新版 profile 合同 `slot_direct_detailed_closure_v2` 另外发布三个互斥父区间：
`pre_helper_total`、`helper_total` 和 `post_helper_total`。它们必须按 raw integer ns
精确满足：

```text
total = pre_helper_total + helper_total + post_helper_total
```

原来没有细分的编排时间被拆成 21 个互斥叶子：

- helper 前：`gate`、`state_alloc`、`gpu_retain`、`outer_lock_acquire`、
  `batch_bookkeeping`、`slot_validate`、`flatten_sort`、`slot_tracker_init`；
- helper 内、但不属于 prepare/copy/map 完成区间：`helper_context`、
  `block_lock_acquire`、`locked_revalidate`、`copy_to_map`、`block_unlock`、
  `helper_context_free`；
- helper 后：`result_check`、`pte_metadata`、`status_resolve`、
  `slot_state_commit`、`telemetry_cleanup`、`outer_lock_release`、
  `resource_teardown`。

stage-sync 样本的互斥闭合式是：

```text
total = prepare_complete + copy_complete + map_complete + wait
      + sum(21 leaves) + driver_residual
```

普通 e2e 样本把三个 `*_complete` 换成 `prepare + copy + map`。`unmap` 仍是
`prepare` 的子区间，submit 计时也仍嵌在 completion 计时中，二者都不能重复相加。
verifier 从保存的 raw ns 独立重算每个闭合量，并要求未解释的
`driver_residual <= max(0.1 ms, total 的 0.5%)`。`host_outer = wall - total`
单独表示 ioctl/userspace/profile 发布边界，不混入驱动内部阶段。

## 正确性与实验调度

每个样本都在新的 Python 进程中创建数据、预分配 slot、建立前置状态并完成全量
GPU checksum。计时后还必须验证：

- requested/completed range 和 byte 完整；
- 原 managed VA 不变；
- checksum 与预期一致；
- slot PTE 映射不伪装成普通 UVM GPU residency，普通 residency 仍为 CPU；
- overwrite 模式恰好跳过全部 block 的独立 unmap，normal 模式一个也不跳过。

runner 固定运行四个 schedule block。每个 block 含两个 matched pair：
`normal_64k_e2e` 对 `overwrite_64k_e2e`，以及
`normal_64k_stage_sync` 对 `overwrite_64k_stage_sync`。每个 pair 按 ABBA 或 BAAB
运行，下一 block 同时反转 pair 顺序和 ABBA 方向。总计 32 个独立进程，每种模式
八个样本。汇总从 raw sample 计算所有时间字段的 P50/P95，并报告：

- normal overwrite 的 wall speedup；
- stage-sync overwrite 的 wall 和 copy-complete speedup；
- normal 与 overwrite 两条路径各自的 stage-sync wall penalty。

`latency_stats` 是 module-global。运行期间不能并行执行其他 UVM workload；runner
会持有独立 advisory lock，worker 也会用精确 counter delta 拒绝受污染样本。

## 构建、运行和复核

只做 userspace 静态构建：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/experiments/uvm_slot_64k_overwrite_stage_sync_compare
make
```

先运行 smoke：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
PROFILE=smoke \
  bash experiments/uvm_slot_64k_overwrite_stage_sync_compare/run_benchmark.sh
```

再运行正式 512×4 MiB：

```bash
bash experiments/uvm_slot_64k_overwrite_stage_sync_compare/run_benchmark.sh
```

结果只写入本目录新的时间戳 artifact。复核正式结果：

```bash
python3 experiments/uvm_slot_64k_overwrite_stage_sync_compare/verify_results.py \
  --require-full \
  experiments/uvm_slot_64k_overwrite_stage_sync_compare/artifacts/<run-id>
```

verifier 从 `samples.jsonl` 和各 sample artifact 复核 32 个独立 PID、精确 ABBA
调度、ABI、action 5/20 前置合同、profile 计数、checksum/VA/residency，并从 raw
samples 重算 aggregate；summary 中预填的汇总值不会被直接信任。

2026-08-16 的历史结果已清理；新数据必须按统一 EuroSys 协议重新生成。

## pinned-copy 对照的边界

vLLM 风格 pinned-copy control 继续复用
`experiments/uvm_transfer_microbench` 的独立 sidecar（`backend=pinned_copy`），不新增
一个假的 ioctl 91 mode。sidecar 使用预分配 GPU slot 和 `cudaMemcpyAsync`，可作为
搬运带宽参考，但它不保持 managed VA，也没有本实验的 PTE overwrite 语义；因此
必须把它标为独立 control，不能声称与这四个模式具有同 VA 或同 metadata 合同。

生成匹配的 2 GiB、512×4 MiB filtered sidecar：

```bash
PROFILE=full COLD_MIB=2048 TENSORS=512 \
ONLY_LAYOUT=multi_alloc ONLY_BACKEND=pinned_copy ONLY_MODE=tensor_batch \
RESULT_FILE=/tmp/pinned_copy_2g_512x4m.txt \
  bash experiments/uvm_transfer_microbench/run_benchmark.sh
```

把它作为独立 control 加入正式 artifact 的复核：

```bash
python3 experiments/uvm_slot_64k_overwrite_stage_sync_compare/verify_results.py \
  --require-full \
  --pinned-result /tmp/pinned_copy_2g_512x4m.txt \
  experiments/uvm_slot_64k_overwrite_stage_sync_compare/artifacts/<run-id>
```

## 驱动 ABI 要求

需要配套实验模块提供：

- ioctl 81，max ranges 64，params size 4656，action 5 和 action 20；
- ioctl 91，max entries 128，params size 6224，mode 8/12/16/17/18；
- `slot_direct_profile` 的 submit 与 completion stages；
- strict 64 KiB PTE、skipped-unmap、stage-sync、allocation/eviction 等 flat counters。

本目录只构建 userspace shim、runner 和 verifier，不负责构建、加载、卸载或切换
内核模块。ABI 或 telemetry 不完整时实验会明确失败，不会生成可误读的性能结论。
