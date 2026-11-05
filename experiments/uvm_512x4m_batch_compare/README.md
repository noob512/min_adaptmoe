# 512×4 MiB direct 与 cpp_batch 对照

这个实验只比较 Python/ctypes 的外层调用形状，不启动或导入 vLLM，也不切换
驱动模块。正式配置使用同一组 512 个独立 `cudaMallocManaged` 测试 allocation，
每个 4 MiB，总逻辑数据 2 GiB：

- `direct`：Python 调用 `prefetch_one` 512 次。
- `batch`：Python 调用 `prefetch_batch` 4 次，每批 128 个 range；shared shim
  内部仍严格循环 128 次 `cudaMemPrefetchAsync`。

因此两条路径的 CUDA Runtime/UVM 调用数都为 512。batch 只把 Python→C 外层
调用从 512 降到 4，不是 driver 级的多 range migrate，也没有合并 UVM 元数据。

## VA 隔离与公平性

每两个 4 MiB 测试 allocation 之间还保留一个默认 4 MiB 的 managed guard
allocation。guard 不初始化、不预取、不计入逻辑字节；程序读取所有测试指针，
排序后验证不存在相邻测试 range，并输出 `adjacent_test_ranges` 和
`min_gap_bytes`。

direct 与 batch 在整个进程中复用完全相同的测试指针、顺序、CUDA stream 和
checksum kernel。每个样本开始前都会：

1. 把全部测试 range 迁到 CPU 并同步；
2. 验证 512 个 `LastPrefetchLocation` 都是 CPU；
3. 在计时区外再次核对所有 VA。

测量采用 `A-B-B-A` 顺序。默认每个模式 2 次 warmup、8 次正式样本。每个样本
记录：

- Python 外层调用的 `host_submit_us`；
- 同一 stream 上从 begin event 到最后一次 prefetch 后 ready event 的
  `migration_ms`；
- GPU 读取完整 2 GiB 的 `checksum_read_ms`；
- begin 到 checksum 完成的 `e2e_ms`；
- 以 2 GiB / migration envelope 计算的 GiB/s。

每个样本必须同时满足：accepted 数为 512、底层 runtime 调用数为 512、外层
调用数为 512/4、VA 不变、全量 checksum 正确，并且 checksum 完成后 512 个
range 的 `LastPrefetchLocation` 都是目标 GPU。

`LastPrefetchLocation` 是请求位置属性，不单独证明物理驻留；这里还通过同一
stream 的 completion event 和随后完整 GPU checksum 形成消费完成条件。实验
不会把它表述为 PFN 或物理连续性证明。

## 运行

小规模功能验证：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
PROFILE=smoke bash experiments/uvm_512x4m_batch_compare/run_benchmark.sh
```

正式 2 GiB 对照（默认配置）：

```bash
bash experiments/uvm_512x4m_batch_compare/run_benchmark.sh
```

验收本轮三次正式运行并输出跨进程/ABBA 汇总：

```bash
python3 experiments/uvm_512x4m_batch_compare/verify_results.py
```

历史性能结果已清理。新的输出必须写入带 protocol ID 的 EuroSys artifact 目录，
不能继续使用旧日期结果文件名。所有配置都可用同名环境变量覆盖，但程序要求
`ALLOCATIONS == 4 * BATCH_SIZE`，从而保证 batch 始终恰好四次外层调用。

## 真正的 128-range driver batch（独立实验）

`true_batch_benchmark.py` 是独立的三模式实验，不改变上面的两模式输出和历史
结果：

- `native_async`：512 次 `cudaMemPrefetchAsync`，最后同步一次；
- `native_4x128_sync`：每批提交 128 次 native prefetch 并立即同步，共四批；
- `driver_4x128_sync`：调用四次 production ioctl 90，每次 128 ranges，ioctl
  返回即代表本批 copy、PTE 更新和 tracker wait 全部完成。

三条路径统一用主机单调时钟从提交开始量到明确完成点。driver ioctl 不属于
CUDA stream，因此本实验不会用 CUDA event 冒充它的迁移时间；纯 submit 时间
也只对 `native_async` 单独报告。

每个样本先把全部 range reset 到 CPU 并同步，然后连续读取两次
`/proc/driver/nvidia-uvm/latency_stats`，确认边界稳定并用第二次作为基线；迁移完成
后立即再读一次并计算增量，所以 reset 工作不进入被测统计，也不要求 root 写全局
reset 节点。driver 模式必须满足：4 次 ioctl、4 次
batch core、512 requested/completed ranges、2 GiB requested/completed bytes、4 次
tracker wait、0 fallback、0 error，并且旧 `uvm_api_migrate` 的
`migrate_profile.api_total.calls` 为 0。两个 native 模式则必须恰好产生 512 次
legacy API 调用且所有 driver-fast 计数为 0。

统计快照之后，程序再通过 production cmd 81/action 4 分成 8×64 独立查询全部
range 的 GPU residency，然后由 GPU 读取完整 2 GiB checksum，并再次核对 VA。
自定义 ioctl 不更新 `cudaMemRangeAttributeLastPrefetchLocation`，因此该属性仅在
两个 native 模式中作为硬断言；driver 模式由逐 range completed 结果、独立
residency 查询和全量 checksum 共同验收。

需要加载包含 versioned ioctl 90 的配套 UVM 模块。运行期间 `latency_stats` 是
module-global，必须保证没有其他 UVM workload 并发污染计数。

```bash
# 小规模功能验证
PROFILE=smoke bash experiments/uvm_512x4m_batch_compare/run_true_batch_benchmark.sh

# 正式 512×4 MiB / 4×128 对照并自动验收结果
bash experiments/uvm_512x4m_batch_compare/run_true_batch_benchmark.sh
```

也可以只复核已保存结果：

```bash
python3 experiments/uvm_512x4m_batch_compare/verify_true_batch_results.py \
  --require-full experiments/uvm_512x4m_batch_compare/RESULTS_TRUE_BATCH_FULL_20260815.txt
```

最终 512×4 MiB 三轮结果中，driver batch 相对具有相同四个完成边界的
`native_4x128_sync` 分别加速 `1.0710x / 1.1310x / 1.1338x`，中位数
`1.1310x`；全部 VA、residency、checksum 和 driver profile 契约通过。完整结果与
边界说明见 `RESULTS_TRUE_BATCH_20260815.md`。
