# 复现 C1 跨机制、按请求精确预留、关闭运行时热门专家审计的矩阵

本文档复现 2026-08-27 的单轮 C1 可行性矩阵。实验保持同一模型、冻结请求、
并发数和生成参数，依次比较三种原生 vLLM 路径、本文机制、KTransformers 和
MoE-Infinity；并为每类请求单独设置 KV、Prefill Graph 和 workspace 预算，以搜索
能够成功服务该请求的最小卸载配置。

本矩阵是 feasibility pilot，不是论文最终的 equal-HBM 排名。不同框架的权重布局、
执行内核和实际峰值 HBM 尚未严格配平，且每个 cell 只有一个独立冷进程样本。

## 1. 实验问题与边界

固定 C1，测试四种请求：

| workload | prompt/output | 正式请求数 | SLO |
|---|---:|---:|---|
| Short dialogue | 128/32 | 1 | TTFT <= 2,000 ms，TPOT <= 100 ms |
| Ordinary chat | 512/128 | 1 | 同上 |
| RAG summary | 8,192/128 | 1 | 同上 |
| Long generation | 512/512 | 1 | 同上 |

每个冷进程先发送 3 个冻结 calibration warmup，请求输出 8 tokens，再发送 1 个正式
evaluation 请求。seed 固定为 `20260823`，prefix cache 关闭。

六个 arm：

| arm | 机制和最小配置搜索 |
|---|---|
| `native_all_gpu` | vLLM eager，全部权重在 GPU，不卸载 |
| `native_pinned_uva` | vLLM eager，从 `cpu-offload-gb=0.01` 开始，直接读 pinned-UVA |
| `native_copy_on_forward` | vLLM eager，从 `cpu-offload-gb=0.01` 开始，forward 时 copy |
| `ours` | 每层从 1 个 calibration-cold expert 开始，不足时逐层增加到 16 个 |
| `ktransformers` | 与 ours 相同 cold-set；从每层 63 GPU/1 CPU expert 开始 |
| `moe_infinity` | 全局 demand cache；在各 workload 已冻结的容量 bracket 内以 0.01 搜索最大成功 ratio |

原生 vLLM 的 `cpu-offload-gb=0.01` 按完整 Parameter 粒度选择，实际卸载量不等于
“每层一个专家”。MoE-Infinity 也是全局 cache ratio，不能换算成逐层 cold 数。

## 2. 运行时热门专家审计开关

本实验显式设置：

```text
VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE=0
VLLM_UVM_RUNTIME_HOT_RESIDENCY_AUDIT=0
```

第一项关闭 PARK 前、PARK 后、remote-map 后的三轮热门专家 CUDA read/checksum；
第二项关闭 transition 尾部对全部热门专家页的 physical-residency 查询。默认值仍为
开启，所以只有显式实验配置会采用此策略。

关闭的只是正式请求首个 Decode 前的高开销诊断。以下保护仍保留：

- warmup 建立 KV、正式 Prefill/Decode Graph pool 和 workspace；
- 正式服务前恢复/再平衡热门专家；
- cold expert CPU-only residency 检查；
- stable original VA、PARK 和 remote-map 返回值检查；
- CUDA illegal access、Xid、PTE/PDE fault 和输出完整性检查；
- 每个 arm 后恢复 stock UVM 并确认 GPU 清空。

实现和回归测试：

```text
workloads/vllm/vllm/vllm/model_executor/models/utils.py
workloads/vllm/vllm/tests/test_cpu_offload_exact_static.py
```

运行最小单测：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm
.venv/bin/python -m pytest -q \
  tests/test_cpu_offload_exact_static.py \
  -k 'uvm_original_transition_to_remote_audits_metadata'
```

验收为 `2 passed`。

## 3. 硬件、模型和版本

参考环境：

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090，32,607 MiB |
| driver | 595.71.05 |
| kernel | 6.13.12-061312-generic |
| CPU | Intel Xeon Gold 6530，`taskset -c 0-13`，14 threads，NUMA 0 |
| model | DeepSeek-V2-Lite-Chat BF16 |
| snapshot | `85864749cd611b4353ce1decdb286193298f64c7` |
| root/vLLM commit | `6fe5033aa9645d1c4eda61a0d1bc52e189c3c864` |
| KTransformers commit | `8d9325af331d916721dfe7e2104f93f9b268678b` |
| KT SGLang commit | `a1b31959e60c7bc989650aef507a74e41e567f56` |
| MoE-Infinity commit | `92c2bc9266e33bbed22405356ca7ca7e71ae8974` |

工作树含尚未提交的实验修改，只有 commit ID 不足以复现。runner 会把实际
`git status --short` 写入 artifact；还应保存相关 diff 和 SHA-256。

### 3.1 本轮关键文件指纹

以下 SHA-256 对应 2026-08-27 最终 78-attempt 汇总所使用的冻结版本。复现前应逐项
比对；任一值变化都必须把新文件、`git diff --binary` 和新哈希保存到新 artifact，
不能仍把运行结果标为本轮的直接复现。

| 文件 | SHA-256 |
|---|---|
| `configs/c1_mechanism_runtime_hot_audit_off_matrix.json` | `39a74a08e7a437f6de886e7e5b2165de8666c5a08cd98de746b8d3d06d8b59f1` |
| `configs/c1_native_common_env.json` | `bfd338b1955daf781cb7ef5596b427ea1d3d6220ce9451fbb6da10ed5726b0bf` |
| `configs/c1_ours_runtime_hot_audit_off_env.json` | `8d90560b32126cfeb91643160fff2a67d5480359ccc9b294b18046fee9757ebf` |
| `run_c1_mechanism_runtime_hot_audit_off_matrix.sh` | `4858f787145518f2c887b138099cb0b37d7dd02b22dd72bfb7224bc7e7767ac1` |
| `summarize_c1_mechanism_runtime_hot_audit_off.py` | `bf3e9eec0fef59d32ca099b57dc218bba2db39fdcf95882f76123bfb670a0a2b` |
| `launch_moe_infinity_server.py` | `441c5bda91e31784f4fb9dbf0d7abd5726fdddeaf1cfa13cee7b089392c0b1ad` |

从仓库根目录验证：

```bash
sha256sum \
  experiments/eurosys_moe_offload_evaluation/configs/c1_mechanism_runtime_hot_audit_off_matrix.json \
  experiments/eurosys_moe_offload_evaluation/configs/c1_native_common_env.json \
  experiments/eurosys_moe_offload_evaluation/configs/c1_ours_runtime_hot_audit_off_env.json \
  experiments/eurosys_moe_offload_evaluation/run_c1_mechanism_runtime_hot_audit_off_matrix.sh \
  experiments/eurosys_moe_offload_evaluation/summarize_c1_mechanism_runtime_hot_audit_off.py \
  experiments/eurosys_moe_offload_evaluation/launch_moe_infinity_server.py
```

KTransformers 和 MoE-Infinity 环境 JSON 可能同时服务其他实验，因此以新运行保存到
case `env.json` 的合并后环境为最终依据；不要只依赖其当前工作树文件。

上表 runner 哈希是最终 78-attempt 运行时的原始值。该运行结束后，runner 新增了一个
向后兼容的 eager/Graph 诊断开关，当前工作树哈希为
`f75f9cd2f7d2b2b248f060dd9cace9873309612441faca95321b3cd08d012497`。
新增分支的默认值仍为 `FULL_SEPARATE`，但它不再是字节级相同的原始 runner；用当前
版本做功能复现时必须显式设置 `MATRIX_OURS_CUDAGRAPH_MODE=FULL_SEPARATE`，并把当前
runner 哈希记入新 artifact。需要宣称“字节级直接复现”时，必须使用上表原始版本。

## 4. 按请求精确预留

冻结配置：

```text
experiments/eurosys_moe_offload_evaluation/configs/
  c1_mechanism_runtime_hot_audit_off_matrix.json
```

| workload | KV tokens/bytes | ours Prefill Graph | ours workspace reserve | MoE step/blocks |
|---|---:|---:|---:|---:|
| Short | 192 / 6 MiB | 128 | 128 tokens | 128 / 12 |
| Ordinary | 672 / 21 MiB | 512 | 512 tokens | 512 / 42 |
| RAG | 8,320 / 260 MiB | 2,048 | 2,048 tokens | 8,192 / 520 |
| Long | 1,056 / 33 MiB | 512 | 512 tokens | 512 / 66 |

KV 每 token 为 32,768 B。ours 的 native-hole sentinel 精确等于上表 KV bytes；
`VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS` 等于该请求实际 Prefill chunk；CUDA
Graph 只捕获 Decode-C1 和该请求的一个 Prefill chunk，不捕获 C4/C8/C16 或其他长度。
RAG 请求仍是完整 8,192 tokens，只是与原生 vLLM 一样分成 2,048 x 4；这样无需为
单步 8K Graph/workspace 多卸载三名专家/层，cold1 即可运行。

RAG 有两个已记录的框架差异：

1. 原生 vLLM 的单步 8K 和 4K chunk 会申请 512 MiB MLA 临时 workspace 并 OOM；
   runner 保持同一 8,192/128 请求，最终以 2,048-token chunk 执行四个 Prefill step。
2. SGLang 内部需要两个 token 槽，所以 KTransformers RAG 使用
   `max-total-tokens=8322`。8320 只能生成 126/128 tokens，增加卸载专家数无效。

MoE-Infinity 的 KV 参数是 `device_memory_ratio` 内的相对比例，不能对四种请求固定使用
同一个小数。runner 按 16 tokens/block、1,769,472 B/block 和参考机可用预算，对每个
device-ratio 向上取整到 `1e-6`，从而保持上表绝对 block 数不变。若 Ordinary 使用
`kv_ratio=0.001`，只能得到 13–14 blocks，请求会永久停在 waiting，而不是进入 Prefill。

这些修正不改变输入 token 序列和目标输出长度。RAG 的原生 vLLM 与 ours 均为 2K x 4；
KTransformers 和 MoE-Infinity 的内部调度仍不同，因此结果只能作为可行性与系统端到端
观察，不能当作纯内核同形比较。

## 5. 环境与依赖检查

从仓库根目录执行：

```bash
cd /home/ubuntu/nvidia-uvm-gpu

MODEL=/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7
test -f "${MODEL}/config.json"
test -x workloads/vllm/vllm/.venv/bin/python
test -x KTransformers/.venv/bin/python
test -f MoE-Infinity/moe_infinity/_store*.so

nvidia-smi --query-gpu=index,name,memory.used,memory.free \
  --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits

for port in 30401 30402 30403 30404 30405 30406; do
  ss -ltn "sport = :${port}"
done
```

开始矩阵前必须满足：GPU 无 compute process、端口均空闲、模型和两个 venv 存在、
MoE-Infinity fresh build receipt 与 `_store*.so` SHA-256 一致。不要与其他 GPU/UVM
实验并行。

参考运行使用的 receipt 为：

```bash
MOEI_RECEIPT=experiments/eurosys_moe_offload_evaluation/artifacts/\
c1-mechanism-min-offload-pilot-20260827-attempt-01/provenance/\
moe_infinity_store_build/store_build_receipt.json

test "$(jq -r '.status' "${MOEI_RECEIPT}")" = complete
MOEI_EXTENSION=$(find MoE-Infinity/moe_infinity -maxdepth 1 -type f \
  -name '_store*.so' -print -quit)
test -n "${MOEI_EXTENSION}"
test "$(sha256sum "${MOEI_EXTENSION}" | awk '{print $1}')" = \
  "$(jq -r '.extension_sha256' "${MOEI_RECEIPT}")"
```

若 receipt 或扩展缺失，必须先 fresh rebuild 到一个不存在的证据目录，再把新 receipt
显式传给矩阵；不要用旧 receipt 为新二进制背书：

```bash
MOEI_BUILD_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
c1-mechanism-runtime-hot-audit-off-matrix-REPRO-moe-build-01

test ! -e "${MOEI_BUILD_ROOT}"
bash experiments/eurosys_moe_offload_evaluation/rebuild_moe_infinity_store.sh \
  "${MOEI_BUILD_ROOT}"
MOEI_RECEIPT="${MOEI_BUILD_ROOT}/store_build_receipt.json"
```

公共环境为离线 Hugging Face、GPU 0、OMP/MKL 14 threads、Asia/Shanghai。具体环境
以以下文件为准：

```text
configs/c1_native_common_env.json
configs/c1_ours_runtime_hot_audit_off_env.json
configs/e2_ktransformers_env.json
configs/e2_moe_infinity_env.json
```

## 6. 一键运行和断点续跑

新运行必须使用不存在的 artifact 目录：

```bash
cd /home/ubuntu/nvidia-uvm-gpu

RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
c1-mechanism-runtime-hot-audit-off-matrix-REPRO-attempt-01

test ! -e "${RUN_ROOT}"
MATRIX_OURS_CUDAGRAPH_MODE=FULL_SEPARATE \
MATRIX_MOEI_RECEIPT="${MOEI_RECEIPT}" \
  bash experiments/eurosys_moe_offload_evaluation/\
  run_c1_mechanism_runtime_hot_audit_off_matrix.sh "${RUN_ROOT}"
```

runner 用独占 `flock` 串行执行六个机制，自动：

- 生成 cold1...cold16 calibration-frequency plans；
- 为 KTransformers 生成相同 cold-set placement；
- 对每个 workload 精确设置 KV、Graph、workspace 和最大长度；
- 搜索 ours/KT 的最小 cold 数和 MoE-Infinity 最大可行 ratio；
- 每次失败保留完整 case，不覆盖失败证据；
- 每个成功/失败 attempt 追加到 `execution.tsv`。

若 shell 或主机被人工中断，先确认旧 server 已退出和 GPU 清空，再原目录续跑：

```bash
MATRIX_RESUME=1 \
MATRIX_OURS_CUDAGRAPH_MODE=FULL_SEPARATE \
MATRIX_MOEI_RECEIPT="${MOEI_RECEIPT}" \
  bash experiments/eurosys_moe_offload_evaluation/\
  run_c1_mechanism_runtime_hot_audit_off_matrix.sh "${RUN_ROOT}"
```

若 NVML 显示 0 MiB、无 compute process，但 custom-UVM preflight 仍报告 client，检查
是否为人工中断遗留的 NVML sampler：

```bash
sudo fuser /dev/nvidia-uvm /dev/nvidia-uvm-tools
ps -fp <pid>
readlink -f /proc/<pid>/fd/1
```

只有当命令确认为本次 artifact 下、父进程已退出的 `nvidia-smi --loop-ms=200` sampler
时才可 `kill -TERM <pid>`；不得终止来源不明的 UVM client。

可用过滤器只补独立项，例如：

```bash
MATRIX_RESUME=1 \
MATRIX_SYSTEMS=moe_infinity \
MATRIX_WORKLOADS=rag_summary,long_generation \
MATRIX_MOEI_RECEIPT="${MOEI_RECEIPT}" \
  bash experiments/eurosys_moe_offload_evaluation/\
  run_c1_mechanism_runtime_hot_audit_off_matrix.sh "${RUN_ROOT}"
```

定位 MoE-Infinity 的无进展问题时，可限制本次只探测一个 ratio，并把单请求超时缩短
为 300 秒；模型启动仍使用独立的 1,200 秒 ready timeout：

```bash
MATRIX_RESUME=1 \
MATRIX_SYSTEMS=moe_infinity \
MATRIX_WORKLOADS=ordinary_chat \
MATRIX_MOEI_CANDIDATES=83 \
MATRIX_REQUEST_TIMEOUT=300 \
MATRIX_MOEI_RECEIPT="${MOEI_RECEIPT}" \
  bash experiments/eurosys_moe_offload_evaluation/\
  run_c1_mechanism_runtime_hot_audit_off_matrix.sh "${RUN_ROOT}"
```

这些变量只改变 watchdog 和本次搜索集合，不改变请求、模型或 ratio 的含义。每个被
跳过的 ratio 仍需在报告中明确为“未测”，不能推断为失败。

若修改硬件、模型或 KV calibration，使冻结 bracket 不再适用，可显式强制搜索其他
ratio。例如：

```bash
MATRIX_RESUME=1 \
MATRIX_SYSTEMS=moe_infinity \
MATRIX_WORKLOADS=ordinary_chat,long_generation \
MATRIX_MOEI_FORCE_SEARCH=1 \
MATRIX_MOEI_CANDIDATES=90,89,88,87,86 \
MATRIX_REQUEST_TIMEOUT=300 \
MATRIX_MOEI_RECEIPT="${MOEI_RECEIPT}" \
  bash experiments/eurosys_moe_offload_evaluation/\
  run_c1_mechanism_runtime_hot_audit_off_matrix.sh "${RUN_ROOT}"
```

当前冻结 bracket 为 Short 0.85→0.83、Ordinary/Long 0.90→0.85、RAG 0.40→0.35；
runner 从高到低，在首个成功点停止。参考 RAG 的先验粗搜证据为 0.40 失败、0.30 成功，
随后按 0.01 封闭到 0.36 失败、0.35 成功。

不要删除失败目录后重用相同路径。成功 case 会被 resume 自动跳过。

## 7. 单个 case 的证据

每个 attempt 至少包含：

```text
command.txt          实际 server 命令
env.json             合并后的环境
server.log           server stdout/stderr
nvml.csv             200 ms GPU 采样
verification.json    harness 验收结果
<cell>/client.json   请求、token hash、TTFT/ITL/E2E
cleanup/             退出和 GPU/UVM 清理证据
```

ours 另有：

```text
offload_manifest.jsonl
phase_transition.json
uvm.log
```

`execution.tsv` 是所有尝试的不可变索引；`returncode=0` 仍需同时满足
`verification.status=ok`、client status、1/1 请求成功和输出 token 数完整。

## 8. 汇总与验收

矩阵结束后执行：

```bash
workloads/vllm/vllm/.venv/bin/python \
  experiments/eurosys_moe_offload_evaluation/\
summarize_c1_mechanism_runtime_hot_audit_off.py \
  "${RUN_ROOT}"
```

应生成 `summary.json` 和 `metrics.csv`。完整矩阵要求：

- 6 mechanisms x 4 workloads = 24 个最终成功 cell；
- 每 cell 1/1 正式请求成功，token 数完整；
- TTFT、TPOT、E2E、peak HBM 均存在；
- ours 的 `hot_physical_audit_enabled=false` 且不出现 runtime hot-read gate；
- 不出现 CUDA illegal access、Xid、PTE/PDE fault、segfault；
- vLLM all-GPU/UVA/copy/ours 进行 exact-generation hash 比对；
- KTransformers/MoE-Infinity 若 hash 不同必须显式记录，不能静默视作一致；
- 所有进程退出、GPU 回到 0 MiB、stock UVM 恢复。

用以下只读命令检查矩阵完整性和最终选择的配置：

```bash
jq '{
  status,
  successful_final_cells,
  attempt_count,
  failed_attempt_count: (.failed_attempts | length),
  missing
}' "${RUN_ROOT}/summary.json"

jq -r '.final_results[] | [
  .arm,.workload,.setting,.ttft_ms,.tpot_ms,.e2e_ms,.peak_hbm_mib,
  .slo_pass,.output_token_ids_sha256
] | @tsv' "${RUN_ROOT}/summary.json" | column -t -s $'\t'
```

参考运行的结构性结果是 `status=ok`、24 个 final cells、78 个总 attempts、52 个失败
边界 attempts、`missing=[]`。总 attempt 数不是跨机器的硬性验收值：如果 frozen bracket
内更早或更晚达到容量边界，它会变化；24 个最终成功 cell、完整证据和配置选择才是硬门槛。
参考 artifact 的不可变汇总指纹为：

| 文件 | SHA-256 |
|---|---|
| `summary.json` | `24505266ae8b93e6340d20462837d3fdb59a236e8e7ab4104e16ac74218a8a8b` |
| `metrics.csv` | `fc225c0585fc5fe1423412f40ae3acf1b591ab325746a6d5cf18caee74f3d7a0` |
| `execution.tsv` | `cde6c1cbbadb82198b13af03402e3208c05bae044e92f4d527f26c6b89297292` |

延迟和 peak HBM 会随硬件状态波动，不能用上述 artifact 哈希验收一次新运行；这些哈希
只用于确认引用的是原始证据。新运行应依据本节的结构、SLO、安全和 generation gate
重新验收。

参考运行最终选中的容量边界如下。新运行若不同，应保留失败点并说明硬件空闲显存、
allocator、依赖或二进制差异，不要人工改写成参考值：

| arm | Short | Ordinary | RAG | Long |
|---|---|---|---|---|
| all-GPU | no offload | no offload | 2K x 4 | no offload |
| pinned-UVA | 0.01 GiB | 0.01 GiB | 0.01 GiB，2K x 4 | 0.01 GiB |
| copy | 0.01 GiB | 0.01 GiB | 0.01 GiB，2K x 4 | 0.01 GiB |
| ours | cold1/layer | cold1/layer | cold1/layer，2K x 4 | cold1/layer |
| KTransformers | 63 GPU/1 CPU | 63 GPU/1 CPU | 同左，`max-total-tokens=8322` | 63 GPU/1 CPU |
| MoE-Infinity | ratio 0.83，12 blocks | 0.85，42 blocks | 0.35，520 blocks | 0.85，66 blocks |

参考 SLO 结果为 20/24：只有 MoE-Infinity 四个 cell 未通过，其中 RAG 同时超过 TTFT
和 TPOT 阈值，其余三个超过 TPOT。请求完成与 SLO 是两个 gate，不能因为请求返回成功
就把该 cell 标为 SLO pass。

fatal 检查：

```bash
rg -n -i \
  'CUDA error|illegal memory access|out of memory|\\bXid\\b|PTE.*fault|PDE.*fault|segmentation fault|Traceback' \
  "${RUN_ROOT}/cases" -g server.log
```

失败搜索点中的 OOM 是边界证据，不应计入最终成功 cell。KTransformers 在请求已经
完整成功后的正常 shutdown 可能打印 `KeyboardInterrupt` traceback；只能忽略这一种
已确认的关停 traceback，不能忽略运行期 traceback。

## 9. 常见故障和处理

### ours 初始化或 Graph capture OOM

先确认使用的是该 workload 的精确 KV/Graph/workspace，而非完整 E2 的 2 GiB KV 和
C1/C4/C8/C16 全 Graph 集。RAG 必须使用 2K x 4，不能把完整请求长度误当作单个 Graph
shape。若配置无误仍不足，再逐层增加一个 cold expert并重启冷进程；不要反复 rebalance
试图修复真实总容量不足。

### warmup 后热门专家被临时分配挤占

正式 Graph pool 和 workspace 必须在热门专家最终恢复前建立，并在恢复时保留 allocator
cache。native-hole sentinel 大小必须等于该 workload KV bytes。正式请求中不得首次
创建更大 shape 的 Graph/workspace。

### RAG 原生 vLLM 的 512 MiB MLA workspace OOM

先试 `PYTORCH_ALLOC_CONF=expandable_segments:True`；仍失败则在保持 8192-token 请求不变
的前提下依次尝试 4096、2048 chunk。记录最终 chunk，不能把它描述为单步 8K Prefill。

### KTransformers RAG 少生成两个 token

若 8192/128 只完成 126 tokens，把 `max-total-tokens` 从 8320 改为 8322。该问题是
SGLang 两个内部槽，不是 GPU 专家数量不足；增加 cold 数不会修复。

### MoE-Infinity warmup OOM

保留首次 `cudaMalloc` OOM 日志，以 0.01 降低 `device-memory-ratio`，每次使用全新冷
进程，直到请求完整成功。不得缩短 warmup、减少输出或复用已经污染的 cache 进程。

### MoE-Infinity 请求长期停在 waiting

读取 `/admin/stats`。若 `pending_requests=1`、`num_steps=0`、GPU 利用率为 0，比较
`kv_cache_num_blocks` 与 `ceil(max_model_len/16)`。这是 KV admission 失败，不是推理慢；
按矩阵 calibration 重新计算 KV ratio，并用新 case 路径重跑，旧 attempt 保留为失败证据。

若 KV blocks 已足够但 8K RAG 仍为 `waiting,num_steps=0`，检查 `/v1/config` 的
`max_tokens_per_step`。参考版本默认固定为 2048，无法接纳 8192-token Prefill；本实验
通过 `MoE.serve` 和 launcher 的显式参数传入 8192。不要用增加 KV 或继续降低 expert
ratio 掩盖这个 scheduler admission 问题。

### HTTP stream 截断或 Connection refused

它们通常是 server 异常退出的结果。优先定位 `server.log` 中最早的 CUDA/OOM/PTE
错误，不要把客户端网络异常当作根因。

### 输出 hash 不同

核对 frozen token IDs、dtype、temperature、seed、EOS 策略和目标输出长度。跨框架
hash 未一致时，本轮只证明“请求可完成”，不能据此给出最终质量等价或机制排名。

### ours/RAG Prefill 与 Decode Graph 的最小差分诊断

runner 支持三个显式模式；默认值仍是主矩阵使用的 `FULL_SEPARATE`：

| `MATRIX_OURS_CUDAGRAPH_MODE` | Prefill | Decode | 用途 |
|---|---|---|---|
| `FULL_SEPARATE` | Graph | Graph | 主矩阵路径 |
| `NONE` | eager | eager | 去除全部 Graph 的 ground truth |
| `FULL_DECODE_ONLY` | eager | Decode-C1 Graph | 单独验证 Decode Graph |

`NONE` 和 `FULL_DECODE_ONLY` 都保留 cold1、stable VA、alias、PARK/RESUME、remote map、
260 MiB KV 和 2K x 4 请求；三次真实 calibration warmup 会先建立 eager Prefill 的
MLA/MoE allocator cache。诊断中显式把
`VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS` 置 0，因为该 hook 只服务 Graph
capture 后的 reserve；这里由真实 warmup 建 cache。不得删除三次 warmup，否则会把
首次 workspace 分配重新带入正式请求，并改变容量与延迟条件。

每个模式必须使用新的 artifact 根目录，且串行执行：

```bash
EAGER_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
c1-rag-exact-generation-eager-diagnostic-REPRO-attempt-01

test ! -e "${EAGER_ROOT}"
MATRIX_SYSTEMS=ours \
MATRIX_WORKLOADS=rag_summary \
MATRIX_OURS_CUDAGRAPH_MODE=NONE \
  bash experiments/eurosys_moe_offload_evaluation/\
  run_c1_mechanism_runtime_hot_audit_off_matrix.sh "${EAGER_ROOT}"

DECODE_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
c1-rag-exact-generation-decode-graph-diagnostic-REPRO-attempt-01

test ! -e "${DECODE_ROOT}"
MATRIX_SYSTEMS=ours \
MATRIX_WORKLOADS=rag_summary \
MATRIX_OURS_CUDAGRAPH_MODE=FULL_DECODE_ONLY \
  bash experiments/eurosys_moe_offload_evaluation/\
  run_c1_mechanism_runtime_hot_audit_off_matrix.sh "${DECODE_ROOT}"
```

两次运行都必须满足 1/1 请求、128/128 输出 token、无 fatal、GPU cleanup 和 stock-UVM
恢复。取 hash：

```bash
find "${EAGER_ROOT}" "${DECODE_ROOT}" \
  -path '*/rag_summary_c1/verification.json' -print -exec \
  jq -r '.checks.phases[0].request_output_hashes[]' {} \;
```

参考结果中两者都必须为：

```text
d471ac81b4a0631724f49fd8bb60be5f0f19930960ba2aee1ec795adb9e9ef6b
```

主矩阵 `FULL_SEPARATE` 为 `db6f9300...`，所以差分定位到 Prefill Graph 路径。这里只
能宣称“定位到路径”，不能宣称具体 kernel 已修复。`NONE` 的 TPOT 和
`FULL_DECODE_ONLY` 的 TTFT 也不是主矩阵性能结果。

若启动时出现 `pure virtual method called`，保留整个失败目录，核对
`/sys/module/nvidia_uvm/srcversion`，并确认失败 case 的 NVML sampler 已退出；只有
恢复 stock `85A797...`、GPU 0 MiB 和端口空闲后才能用新 attempt 重试。一次重试成功
不能关闭该问题；再次出现时必须收集 native core/backtrace。

## 10. 结果解释限制

本实验固定的是请求，不是所有系统内部执行路径，也没有 equal-HBM：

- 三种原生 vLLM 只卸载约一个完整 Parameter，ours/KT 按每层 cold expert；
- MoE-Infinity 使用全局 demand cache；
- ours 使用请求形状的 FULL_SEPARATE CUDA Graph，原生 vLLM 为 eager；
- RAG 原生 vLLM 与 ours 均使用 2K x 4；MoE-Infinity 需要 8K admission；
- 只有一轮，无法给出方差或置信区间。

因此可报告最小可行配置、请求成功、SLO、延迟分解和容量边界；正式论文性能结论仍需
equal-HBM 配平、跨框架 generation gate 和多个独立冷进程重复。

## 11. 对应报告与原始证据

- 最终指标：`RESULTS_C1_MECHANISM_RUNTIME_HOT_AUDIT_OFF_MATRIX_20260827.md`；
- 已解决错误：`ERROR_FIX_LOG_C1_MECHANISM_RUNTIME_HOT_AUDIT_OFF_20260827.md`；
- 尚未解决问题：`UNRESOLVED_C1_MECHANISM_RUNTIME_HOT_AUDIT_OFF_20260827.md`；
- 机器证据：`artifacts/c1-mechanism-runtime-hot-audit-off-matrix-20260827-attempt-01/`。

最终 ours/RAG ground truth 是第 78 个 attempt：`cold_per_layer=1`、
`prefill_chunk_tokens=2048`、TTFT 385.00 ms、TPOT 11.74 ms、E2E 1,875.39 ms。
旧 `cold4 + 单步8K` 只表示过度预留配置的容量边界，不能作为最终复现配置。
