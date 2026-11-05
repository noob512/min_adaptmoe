# vLLM P3-derived static-alias CUDA Graph integration

本目录验证两件彼此独立的事情：

1. CUDA VMM 的双 bank、RW upload VA、RO consumer alias VA 能否在同一个
   CUDA Graph 中按 `H2D -> consume -> reuse` 正确 replay；
2. Qwen3-30B-A3B-FP8 的 L0-L3 `w13_weight/w2_weight` 接入后，vLLM 的
   FULL decode Graph 是否真的包含这些 H2D，并与 eager 输出一致、性能如何。

它不切换 `nvidia_uvm`，也不使用 cmd96。运行时要求当前加载的是系统安装的
stock UVM；脚本只使用 CUDA Driver VMM API。

旧 Qwen 结果已清理；本目录只保留实现与运行入口，正式性能需要按统一 EuroSys
协议重新测量。

## 精确定义

Qwen fixture 与
`experiments/vllm_p3_pinned_uva_phase_compare` 相同：

- layer 0-3，共 8 个 packed 参数；
- 每层 576 MiB，每次 forward 逻辑 H2D 为 2.25 GiB；
- layer 0/2 使用 bank 0，layer 1/3 使用 bank 1；
- 两个 physical bank 的 backing 合计 1.125 GiB；
- pinned source、upload VA、consumer VA 和 region topology 在 Graph
  生命周期内固定；consumer mapping 为只读。

Qwen Graph 配置固定为：

```text
cudagraph_mode             FULL_DECODE_ONLY
cudagraph_capture_sizes    1,4
compilation_mode           NONE (no Dynamo/Inductor)
TP / PP / DP               1 / 1 / 1
DBO                         disabled
V1 multiprocessing          disabled
```

因此 314/1217-token prefill 仍走 eager；batch 1/4 的 strict one-token
decode 必须各命中一个 FULL Graph。这里不把 piecewise Graph 算作通过。

## 三个测试臂

| 符号 | 模式 | 后备显存语义 | 默认运行 |
|---|---|---:|---|
| `E` | 已发表的 eager pinned `value.to(device)` | 无持久 staging bank | 是 |
| `A` | FULL decode Graph + dual-bank VMM static alias | 1.125 GiB | 是 |
| `D` | FULL decode Graph + direct staging bank | 由 manifest 报告 | 否，诊断用 |

`D` 只有在 direct static-bank hook 已能跨 eager/profile stream 和 capture
stream 安全重绑定时才应启用。默认验收只需要 `E` 和 `A`。

## 1. 静态门禁

不会初始化 CUDA 或加载模型：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
bash experiments/vllm_p3_static_alias_graph_integration/run_integration.sh \
  static-preflight
```

它检查 Python 语法、allocator API 可导入，并运行 CPU-only verifier tests。

## 2. allocator/tensor Graph smoke

默认每个 logical layer 仅 8 MiB，适合先验证 API 和 teardown：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
ALLOW_GPU_RUN=1 \
bash experiments/vllm_p3_static_alias_graph_integration/run_integration.sh smoke
```

smoke 建立：

```text
bank0: RW upload0 <-> physical backing <-> RO consumer L0, L2
bank1: RW upload1 <-> physical backing <-> RO consumer L1, L3
```

Graph 内依次执行四组 H2D 和 alias-to-snapshot D2D。随后修改四个 pinned
source 的内容但不重新 capture，再 replay 并做全字节比较及 SHA-256；这能
区分“Graph 真正重放 H2D”和“只是读到 capture 时残留的数据”。它还检查所有
VA 在 replay 前后不变，并按 `Graph -> tensor wrappers -> allocation.close()`
顺序 teardown。

用正式 576-MiB/layer 几何做 qualification：

```bash
SMOKE_LAYER_MIB=576 SMOKE_WARMUPS=3 SMOKE_REPLAYS=20 \
ALLOW_GPU_RUN=1 \
bash experiments/vllm_p3_static_alias_graph_integration/run_integration.sh smoke
```

smoke 的 `gpu_ms` 包含 4 次 H2D 和 4 次完整 D2D snapshot，不能当作纯 H2D
带宽；它的用途是 allocator/Graph 正确性和 soak。

## 3. Qwen 一进程资格测试

先各运行一个独立的 `E/A` 进程。资格测试打开 copy manifest trace，主要看
正确性和 capture 生命周期，不作为正式性能样本：

```bash
RUN_ROOT=/home/ubuntu/nvidia-uvm-gpu/experiments/\
vllm_p3_static_alias_graph_integration/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_qualify \
ALLOW_GPU_RUN=1 \
bash experiments/vllm_p3_static_alias_graph_integration/run_integration.sh qualify
```

通过条件包括：

- 每个结果状态为 `ok`，scheduler 的 prefill/decode 形状严格成立；
- `E/A` 每个 batch、repeat 的 token IDs 和 SHA-256 完全一致；
- `A` 实际模式仍为 `FULL_DECODE_ONLY`，load 期至少捕获 batch 1/4 两个图；
- `A` 的每个 warmup/measured decode step 恰好有一次 `full_replay`；
- prefill 没有 FULL replay，运行期没有 lazy capture；
- manifest 精确覆盖 8 个参数、2.25 GiB logical bytes、bank 0/1、L0-L3；
- `A` 有 4 个 upload mapping、8 个 consumer mapping，去重 physical backing
  恰好 1.125 GiB；
- trace 中存在 capture-time forward；Graph replay 本身不再次进入 Python
  forward hook。

## 4. 正式性能运行

正式默认顺序是 `EA / AE / EA`，每个机制三个独立进程，并在 block 间反转
batch 顺序。copy trace 默认关闭，避免文件 I/O 污染 eager 数据：

```bash
RUN_ROOT=/home/ubuntu/nvidia-uvm-gpu/experiments/\
vllm_p3_static_alias_graph_integration/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_formal \
ALLOW_GPU_RUN=1 \
bash experiments/vllm_p3_static_alias_graph_integration/run_integration.sh run
```

默认每进程、每 batch：1 个 conditioning cycle、3 个 measured cycles；每个
cycle 有 1 个完整 prefill、5 个 decode warmup 和 16 个 measured decode。
每个 engine step 前后都有 `torch.cuda.synchronize()`。

主要指标为：

- `decode_step_ms`：FULL Graph 的主要端到端指标，包含 scheduler submit、
  captured 2.25-GiB H2D、MoE compute 和完成等待；
- `prefill_step_ms`：本配置下仍是 eager，不可解释为 Graph 性能；
- `cycle_e2e_ms`：完整请求 cycle，单独报告；
- `load_seconds`：包含模型加载、compile 和 Graph capture，只作启动成本参考；
- `cuda_memory_after_load` 与 alias manifest 的去重 backing bytes。

verifier 以“每进程内部 repeat median，再对三个独立进程 median”汇总，报告
`A/E` ratio 和 speedup；首次集成阶段性能数值不是 fail gate，正确性、Graph
命中和 backing 大小才是 fail gate。

如需加入诊断臂 `D`，每个三进程 block 必须是 `EAD` 的排列，例如：

```bash
ORDER=EADADEDEA REQUIRED_ARMS=EAD ALLOW_GPU_RUN=1 \
bash experiments/vllm_p3_static_alias_graph_integration/run_integration.sh run
```

## 安全与复现要求

- 独占 GPU；开始和每个子进程退出后均要求目标 GPU 为 0 MiB 且全机无
  compute PID。
- wrapper 使用全局 `flock`，但不会停止其他服务；运行前仍需人工确认无人
  使用 GPU。
- 不卸载、不插入、不覆盖 kernel module；若加载的 UVM `srcversion` 与系统
  installed module 不同，运行会 fail closed。
- 模型和 fixture 只允许离线读取，不自动下载。
- Graph/source/tensor 的生命周期由进程隔离；allocator smoke 显式验证释放
  顺序。
- 性能运行建议外部锁定 GPU/显存频率，结束后恢复；脚本本身不修改时钟。

离线重跑 verifier：

```bash
bash experiments/vllm_p3_static_alias_graph_integration/run_integration.sh \
  verify /absolute/path/to/run_root
```

artifact 保存命令、环境、输入 SHA-256、GPU 前后快照、resource usage、完整
result、offload manifest、token hash、Graph capture/replay 计数及汇总结果。
