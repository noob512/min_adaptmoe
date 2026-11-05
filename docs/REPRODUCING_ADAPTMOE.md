# 构建、验证与复现 AdaptMoE

本页描述当前代码的入口，不自动运行模型、不下载模型，也不重新生成论文中的全部图表。
[项目机制与边界](ADAPTMOE_PROJECT_GUIDE.md) 给出了论文到源码的对应关系。

## 环境与源码

已使用的环境是单张 RTX 5090（32,607 MiB）、NVIDIA 595.71.05、Linux
6.13.12、Python 3.12。vLLM 构建依赖见
[`pyproject.toml`](../workloads/vllm/vllm/pyproject.toml) 与
[`requirements/`](../workloads/vllm/vllm/requirements/)，其中当前 build-system
固定 PyTorch 2.9.0。CUDA toolkit、PyTorch 和 vLLM 扩展需相互兼容。
这不是对其他 GPU/驱动/多 GPU 配置的兼容承诺。

```bash
git clone https://github.com/noob512/min_adaptmoe.git
cd min_adaptmoe
```

vLLM 和驱动源码已放在树内。四机制比较不需要额外 baseline。
KTransformers/MoE-Infinity 实验先获取固定版本及嵌套依赖，脚本会应用记录的测量补丁：

```bash
python3 scripts/fetch_baselines.py --dry-run
python3 scripts/fetch_baselines.py KTransformers MoE-Infinity
```

随后按各 baseline 上游说明构建；版本和补丁说明见 [baselines](../baselines/README.md)。

在 `workloads/vllm/vllm/` 建立虚拟环境，并按该目录上游构建说明安装兼容 CUDA/PyTorch
和 build dependencies。使用此树的 editable/source build，例如在依赖齐备后：

```bash
cd workloads/vllm/vllm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/build.txt
python -m pip install --no-build-isolation -e .
cd ../../..
```

不要让另一份 PyPI vLLM 遮蔽修改过的模块。可用下面的 CPU import 检查路径：

```bash
workloads/vllm/vllm/.venv/bin/python -c 'import vllm; print(vllm.__file__)'
```

## 驱动与 allocator 构建

须安装匹配的 NVIDIA 595.71.05 基础驱动、运行内核的 headers 和 build tools。
本仓库修改的 UVM 与基础驱动之间存在 ABI 依赖。

```bash
make driver
make allocator
```

`make driver` 将校验固定版本的 NVIDIA 上游源码并叠加本仓库 UVM 修改，在 `.build/`
生成独立构建树。可设置 `NVIDIA_SOURCE_ARCHIVE` 使用本地 checksum 匹配的源码包，
或设置 `DRIVER_BUILD_DIR` 选择新的构建位置。
内核 headers 不在默认位置时，执行
`bash scripts/build_driver.sh SYSSRC=/path/to/kernel/headers`。
CUDA 不在默认位置时，为 allocator 构建设置 `CUDA_HOME` 与 `NVCC`。
模型 wrapper 默认查找源码树内模块；使用上述构建树时应设置
`CUSTOM_UVM="$PWD/.build/nvidia-595.71.05/kernel-open/nvidia-uvm.ko"`。
完整模块构建会生成 NVIDIA 导出符号；直接只构建 `nvidia-uvm` 而不提供基础驱动的
符号表可能在 modpost 阶段出现未定义的 `nvUvmInterface*`。

驱动和 allocator 的构建记录见 [VALIDATION.md](VALIDATION.md)。模型实验 wrapper 会在 GPU 无
compute/UVM client 时临时加载 custom UVM，结束时恢复 stock UVM 和原先运行的
exporter 服务。该操作需要 sudo 和独占 GPU，不能与其他 GPU 服务并行运行。

## 不运行模型的检查

```bash
make check
make test
```

聚焦回归集合结果为 **464 passed**，包括控制器、队列、计划与 ABI 检查。测试使用仓库内 vLLM 源码与已安装运行时依赖，可用 `make test
PYTHON=/path/to/compatible/python` 指定环境。这些 mock/contract 测试不能替代真实模型
的输出一致性与性能验证。

四机制输入可以在无模型、无数据集、无 GPU 的情况下准备：

```bash
PREPARE_ONLY=1 bash experiments/adaptmoe_token_threshold_sharegpt_compare/run_kv1_compare.sh \
  /absolute/new/prepared-inputs 1 20
```

`PREPARE_ONLY=1` 的目录用于检查输入，不是一次完成的实验。正式运行须另选新目录。

## 当前 mixed-scheduling 四机制实验

准备本地 DeepSeek-V2-Lite-Chat snapshot 与 ShareGPT V3 数据集。
冻结 trace 已包含 prompt token IDs，因此四臂使用同一请求内容和顺序；完整数据集
留在本地，用于 runner 的来源一致性检查。

```bash
export MODEL=/absolute/path/to/DeepSeek-V2-Lite-Chat
export SHAREGPT_DATASET=/absolute/path/to/ShareGPT_V3_unfiltered_cleaned_split.json

ALLOW_GPU_RUN=1 SAMPLES=1 \
bash experiments/adaptmoe_token_threshold_sharegpt_compare/run_kv1_compare.sh \
  /absolute/new/artifacts/kv1-qps1 1
```

四个进程依次为 Native UVA、Native Copy、AdaptMoE fixed K1、AdaptMoE dynamic
K1↔K2。默认 `SAMPLES=1`，每臂只运行一次，不自动复测。
需要扫多个到达率时，在同一命令末尾传 `1 2 4 8 10 20 30`；这些 rate 都是全新运行，
不复用旧结果。`SAMPLES=2` 才会显式请求第二遍反序运行。

| 参数 | 固定值 |
|---|---|
| 请求 | 128 条 ShareGPT，1024 输入 / 16 输出 |
| 到达 | 第 i 个请求在 `i / QPS` 到达；QPS 是 offered rate |
| 模型 | DeepSeek-V2-Lite-Chat BF16 |
| 调度 | C128，8192-token budget，max_model_len=1088 |
| Attention / Graph | TRITON_MLA，eager，CUDA Graph `NONE` |
| Prefix cache / 采样 | 关闭；greedy，ignore EOS |
| Access thresholds | PARK≤16，RESUME≥128，区间内保持原状态 |
| 固定 KV | 2155 usable blocks，34,480 tokens，约 1 GiB |
| 动态 KV | 2155↔4313 usable blocks，K1↔K2 |
| AdaptMoE 专家 | K1/cold8，K2/cold11，逐层 cold slices |
| Native 专家 | UVA/Copy 使用同 7 个 whole Parameters，较 cold8 多卸载 2.56% |

`CPUSET`、`VLLM_PY`、`CUSTOM_UVM`、`TIMEOUT_SEC` 可按环境覆盖；默认 CPUSET 为
0–13。模型和 dataset 路径必须显式设置。不要照抄历史 frozen env 中的绝对路径直接
启动服务；wrapper 会覆盖当前实验需要的 plan、phase、Graph 与路径配置。

## 结果与正确性

新输出目录保留 `sweep_protocol.json`、每档输入/执行状态、逐臂 `result.json`、
manifest、日志、内核快照、清理证明，以及汇总 `summary.json`、`summary.csv`、
`output_checks.csv` 和 `REPORT.md`。不覆写已有成功或失败目录。

actual QPS = 128 / 从首个计划到达到最后请求完成的时长，包含尾部 drain。
TTFT 包含到达、admission 和排队等待；TPOT 来自后续 token 的间隔。
计时包含真实 resize，未减去扩容时间；无额外真实 conditioning，请勿标为纯稳态。

校验器检查完整请求数、trace hash、到达时间、卸载字节、Graph 禁用、copy/remote
路径、KV 队列/驻留、容量事件、内核 fault 和最终清理，并比较完整输出与首 token。
出现输出差异时保留计数并输出 `ok_with_output_drift`；这不是严格数值等价通过。

该入口使用共享底层 runner，输入准备不依赖历史 artifact。源码与 CPU 检查的范围见
[VALIDATION.md](VALIDATION.md)。新机器上的性能与输出一致性需实际运行后重新验收。

## 实验索引与输入

[实验与图表索引](EXPERIMENTS.md) 对应 placement、E2/E3、E5/E6、E13、Stock UVM
和 mixed scheduling。较早的双 Graph、placement、KV round-trip 协议从
[`experiments/reproduce/README.md`](../experiments/reproduce/README.md) 进入，
其数据口径与本页的 mixed/eager/1024-token trace 不同。

冻结 trace、JSON 专家计划和来源清单保存在树内。模型与完整 ShareGPT 是外部输入；
某些历史 baseline 协议还需要先运行其 calibration/placement 生成脚本，详见对应协议。
旧协议中的 artifact 路径指向历史测量，不随源码分发，也不保证脱离它们后单条旧命令
即可重跑。新的四机制比较使用 `run_kv1_compare.sh`，不要使用依赖历史结果的
`run_kv1_qps_sweep.sh` 来启动一次独立实验。

绘图重建无需上述外部输入：

```bash
python3 -m pip install -r evaluation/requirements.txt
make figures
```
