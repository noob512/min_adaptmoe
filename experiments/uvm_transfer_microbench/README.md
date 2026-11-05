# 不启动 vLLM 的搬运微基准

这个基准只依赖 CUDA Runtime，不导入 PyTorch，也不启动 vLLM。它把三条路径放在同一个 tensor 切分、数据量、stream/event 拓扑和 checksum 消费 kernel 下比较：

- `pinned_copy`：每 tensor 的 mapped pinned CPU backing；预先分配并循环复用固定 GPU slots；单 copy stream 上逐 tensor `cudaMemcpyAsync`，一次 ready event 交给消费 stream。这对应 vLLM `PrefetchOffloader` 的搬运语义，slot 分配不计入搬运时间。
- `managed_prefetch`：相同逻辑 tensor 使用 `cudaMallocManaged`，用 `cudaMemPrefetchAsync` 搬到 GPU；消费地址始终是原 managed 地址。
- `uva_direct`：不搬运，GPU checksum kernel 直接读取 mapped pinned CPU backing，作为 UVA remote-read 对照。

布局包含：

- `multi_alloc`：每个冷/热 tensor 都是独立 allocation，模拟参数碎片和 allocation metadata。
- `arena_interleaved`：单 arena 中冷、热 tensor 交错，冷区间不能合并。
- `arena_contiguous`：单 arena 中冷区间连续、热区间连续；`coalesced_batch` 可把相邻冷 tensor 合成一个调用。

提交方式包含逐 tensor 同步、逐 tensor 异步提交后一次 event/wait（`tensor_batch`），以及相邻区间合并后一次 event/wait。这里的 batch 是 vLLM 风格的“N 个 async API 调用 + 一次 ready event”，不是 CUDA 不存在的 vector-prefetch API。

每条 `RESULT` 都输出提交函数墙钟 `submit_us`、纯 API 调用累计 `api_call_us`、迁移/暴露等待/读取/e2e/host-block 时延、GiB/s、checksum、是否原址访问和跨重复消费 VA 是否相同。`DIAGNOSIS` 分开比较“相同 range 数但有无冷热间隙”、“相同 range 数但单/多 allocation”以及“同 arena 的 N range/1 coalesced range”，给出启发式归因；它只能筛查 host submit、range/allocation 或 driver orchestration 开销，不能把全部差距独立定因为 metadata，也不能替代 Nsight/驱动插桩。

## 快速验证

```bash
cd /home/ubuntu/nvidia-uvm-gpu
bash experiments/uvm_transfer_microbench/run_benchmark.sh
```

完整配置：

```bash
PROFILE=full RESULT_FILE=/tmp/uvm_transfer_full.txt \
  bash experiments/uvm_transfer_microbench/run_benchmark.sh
```

可以覆盖 `COLD_MIB`、`TENSORS`、`SLOTS`、`WARMUPS`、`REPEATS` 和
`EQUIVALENT_RATIO`（默认 `0.90`）。性能不达标仍会正常输出判定；只有 CUDA 错误、
checksum/placement 失败或结果字段缺失才返回失败。

对仓库内保存的正式结果和驱动 profile 做一键验收：

```bash
python3 experiments/uvm_transfer_microbench/verify_results.py
```

它同时检查功能正确性、managed VA 稳定、coalesced UVM 与 pinned-copy 的性能门槛、
range 合并收益、`DIAGNOSIS` 归因、profile 命令成功状态、精确的默认 2 MiB copy
method，以及同配置下强制 4 KiB chunk 的反证。

如需同时确认实验后当前模块仍是被 profile 的构建且 chunk mask 已恢复默认，可运行：

```bash
python3 experiments/uvm_transfer_microbench/verify_results.py --check-live-state
```

正式报告和验收器只引用小写文件名的 canonical 产物：
`result_20260814_default_chunks.txt`、`result_contiguous_*.txt` 和
`driver_profile_*.json`。目录中的大写 `RESULTS_*` / `PROFILE_*` 是独立重复采样，
不作为报告数字或自动验收输入；其中
`PROFILE_ARENA_CONTIGUOUS_SPLIT_128_20260814.json` 采于强制 4 KiB chunk 期间，
不能当作默认 chunk profile。

驱动计数器或 Nsight 应一次只包裹一个 case，避免全矩阵聚合污染归因：

```bash
ONLY_LAYOUT=arena_contiguous \
ONLY_BACKEND=managed_prefetch \
ONLY_MODE=coalesced_batch \
RESULT_FILE=/tmp/uvm_one_case.txt \
  bash experiments/uvm_transfer_microbench/run_benchmark.sh
```

二进制也直接支持 `--only-layout`、`--only-backend`、`--only-mode`；三项可单独或组合使用。过滤执行只创建所选 backend 必需的存储，默认不带过滤器仍运行 17 个 case。

## “物理连续”的边界

`arena_contiguous` 保证的是 CPU/managed **虚拟地址**和 CUDA allocation 的连续布局。`cudaHostAlloc`、`cudaMallocManaged` 不提供“CPU PFN 必须连续”的 API，普通用户态也不能可靠控制或审计所有 PFN；而 CUDA DMA 本身支持 scatter/gather。因此输出明确标记 `cpu_physical_contiguity_controlled=0`，不会把单 arena 结果误称为物理连续内存实验。若必须验证 PFN 连续，需要另行使用有权限的 hugetlb/CMA/内核分配路径，再接入 CUDA 注册与驱动级审计。
