# vLLM 分组卸载调度独立微基准

这个目录用一个不导入、不启动 vLLM 的 CUDA 程序复现上游
`PrefetchOffloader` 的调度拓扑，并在相同层大小、tensor 切分、计算间隔和
checksum 消费下比较三种数据路径：

- `pinned`：每个 tensor 使用 pinned CPU backing；按 tensor 执行
  `cudaMemcpyAsync`；所有模块只复用 `prefetch_step` 组静态 GPU slot；使用
  单 copy stream、layer-done trigger 和 ready event。
- `managed`：每个 tensor 使用 managed allocation；消费地址始终不变；以和
  pinned 路径相同的初始 K 个预取及 `i + K` trigger 拓扑执行
  `cudaMemPrefetchAsync`。每次计时前迁回 CPU，迁回时间不计入结果。
- `uva`：checksum kernel 直接读取 mapped pinned CPU 地址，不执行 H2D。

层选择严格使用上游条件：

```text
module_index % group_size >= group_size - num_in_group
```

默认 `total_layers=16, group_size=8, num_in_group=2, prefetch_step=2`
会选择 `6,7,14,15`，绑定 slot `0,1,0,1`；先预取 6、7，层 6/7
执行完后分别触发 14/15。

## 冒烟验证

```bash
cd /home/ubuntu/nvidia-uvm-gpu
bash experiments/vllm_group_prefetch_baseline/run_group_benchmark.sh
```

脚本默认把带时间戳的输出保存到 `/tmp`，并校验选择层、slot、trigger、三条路径
checksum、managed VA 稳定性和结果字段。仓库中的
`RESULTS_GROUP_SMOKE_20260814.txt` 是本轮保存的默认 smoke 证据。
参数都可由同名大写环境变量覆盖，例如：

```bash
LAYER_MIB=64 TENSORS=16 COMPUTE_US=500 REPEATS=7 \
  RESULT_FILE=/tmp/group-offload.txt \
  bash experiments/vllm_group_prefetch_baseline/run_group_benchmark.sh
```

无需使用 GPU 即可验收本轮保存的 smoke 与 256 MiB 正式结果：

```bash
python3 experiments/vllm_group_prefetch_baseline/verify_group_results.py
```

正式结果使用 64 MiB/layer、16 tensors/layer、2 warmup + 7 repeats。pinned、
managed 和 UVA 分别得到 `51.727 GiB/s` 搬入、`21.185 GiB/s` 搬入和
`47.643 GiB/s` 直接读取；含 500 us/layer 合成计算的 e2e 为
`7.014/12.316/11.834 ms`。原始输出保存在 `RESULTS_GROUP_FULL_20260814.txt`。

## 指标边界

- `host_submit_ms` 只到所有 stream work 入队完成为止；`host_block_ms` 还包含样本
  末尾等待完成的时间。`transfer_work_ms` 是各模块 ready ticket 的传输时间之和；
  `transfer_window_ms` 包括单 copy stream 等待后续 layer trigger 的空隙。
- `exposed_wait_ms` 是计算 stream 到达已卸载层后真正等待 ready event 的时间。
- `read_ms` 是相同 checksum kernel 的累计读取时间；`e2e_ms` 包含所有层的合成
  计算、暴露等待和读取。
- pinned 的静态 slot 地址和 managed/UVA 的消费地址在重复间都必须稳定。
- managed 基准为了比较一次完整 CPU→GPU onload，每轮结束后允许所有已选择层
  驻留 GPU，下一轮计时前再迁回 CPU。因此它没有声称与 pinned 静态 slot 的
  峰值 HBM 预算相同。
- pinned/managed 都仍按 tensor 提交；“group”只控制层选择和预取时机，不代表
  把一组权重合成一次 memcpy，这与上游实现一致。
- 程序采用单 pass：只触发存在的 `i + K`，pass 末尾不环回重填前 K 个。这样对
  任意被选层数都不会在最后消费者完成前覆盖其 slot；跨 forward 的稳态 rearm
  不计入本基准。
- `cudaHostAlloc`/`cudaMallocManaged` 不保证 CPU PFN 连续，本基准不把连续虚拟
  地址误称为物理连续内存。
