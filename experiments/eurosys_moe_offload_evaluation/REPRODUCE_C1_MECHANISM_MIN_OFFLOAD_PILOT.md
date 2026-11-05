# 复现 Short-dialogue C1 跨机制最小卸载 Pilot

本文档复现 2026-08-27 完成的 Short-dialogue C1 单轮可行性实验，并说明
`ours` 首个 Decode token 中约 426.84 ms phase-transition 开销的来源。

参考结果与原始证据位于：

```text
experiments/eurosys_moe_offload_evaluation/
  RESULTS_C1_MECHANISM_MIN_OFFLOAD_PILOT_20260827.md
experiments/eurosys_moe_offload_evaluation/artifacts/
  c1-mechanism-min-offload-pilot-20260827-attempt-01/
```

本实验是单轮、非 equal-HBM feasibility pilot。它回答“在精确 C1 空间预留下，
各机制能否运行及最小卸载粒度是多少”，不能作为论文中的最终性能排名。

## 1. 冻结实验定义

| 项目 | 冻结值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090，32,607 MiB，GPU 0 |
| driver | 595.71.05 |
| kernel | 6.13.12-061312-generic |
| CPU | `taskset -c 0-13`，OMP/MKL 14 threads |
| 模型 | DeepSeek-V2-Lite-Chat BF16 |
| snapshot | `85864749cd611b4353ce1decdb286193298f64c7` |
| workload | Short dialogue，冻结 evaluation request 0 |
| prompt/output | 128/32 tokens |
| concurrency | C1 |
| warmup | 3 个冻结 calibration 请求，每个输出 8 tokens |
| measurement | 1 个正式请求 |
| seed | 20260823 |
| SLO | TTFT <= 2000 ms 且 TPOT <= 100 ms |
| prefix cache | disabled |

六个最终 arm：

| arm | 配置 |
|---|---|
| `native_all_gpu` | vLLM eager，不卸载 |
| `native_pinned_uva` | vLLM eager，`cpu-offload-gb=0.01`，UVA backend |
| `native_copy_on_forward` | vLLM eager，`cpu-offload-gb=0.01`，copy backend |
| `ours_cold1` | 每个 MoE 层各卸载一个 calibration-cold expert；Prefill/Decode FULL_SEPARATE Graph |
| `ktransformers_cold1` | 每层 63 GPU experts + 1 CPU expert；Prefill eager，Decode-C1 Graph |
| `moe_infinity_ratio_083` | global demand cache，`device-memory-ratio=0.83` |

原生 vLLM 的 `cpu-offload-gb=0.01` 按完整 Parameter tensor 选择，实际只卸载
14,942,208 B，不能称为“每层 cold1”。`ours` 与 KTransformers 才使用相同的
26 个 cold experts，共 449,839,104 B。

## 2. 输入与版本固定

从仓库根目录执行：

```bash
cd /home/ubuntu/nvidia-uvm-gpu

sha256sum \
  experiments/eurosys_moe_offload_evaluation/configs/c1_short_mechanism_pilot.json \
  experiments/eurosys_moe_offload_evaluation/streaming_client.py \
  experiments/eurosys_moe_offload_evaluation/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/evaluation_short_dialogue.json \
  experiments/eurosys_moe_offload_evaluation/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/calibration_short_dialogue.json
```

参考 SHA-256：

```text
4e270a2ed571013c642359833689ea727d5abefcdace8968ee0f4127c3c983ac  c1_short_mechanism_pilot.json
27e96df05ed06420c99b14a5f92e0a34a2543f89a3de0d4c9ee478e5afcc9a9e  streaming_client.py
3aec45dda0fe5a5e8b8aa99cdd216714206c25975f007a508616b79bca8cfe2e  evaluation_short_dialogue.json
337404dd625f5ca91748c5ace76ceb6adb6810de8d08959e83baddd1fba0e21c  calibration_short_dialogue.json
```

正式请求的 prompt-token-ID SHA-256 必须为：

```text
eb9a77425f9afefa503200bd45a694762c265707a3255a0c6fcbaceb15c73d5e
```

代码基线：

```text
root/vLLM commit:       6fe5033aa9645d1c4eda61a0d1bc52e189c3c864
KTransformers commit:   8d9325af331d916721dfe7e2104f93f9b268678b
KT SGLang commit:       a1b31959e60c7bc989650aef507a74e41e567f56
MoE-Infinity commit:    92c2bc9266e33bbed22405356ca7ca7e71ae8974
```

注意：参考运行包含尚未提交的 vLLM 和 KT SGLang 修改，只有 commit ID 不足以
复现。运行前必须保存 `git status --short`、相关文件 SHA-256 和二进制 SHA-256。
参考 custom UVM：

```text
srcversion:  5BA0A4BFB3AD96A55780A4C
ko SHA-256: e0fd8ea7a714c36f6759d6de00f0d93a93f1f471a7674e6cc87ee297e9a630ac
```

参考运行关键文件 SHA-256：

```text
6a25f05de70c0eb66d2fa6ef6382f00c4d582b8fb2e9d1c0df1ea2dfc753c7ba  vllm/model_executor/models/utils.py
e0ad04f1d337b8eb2a8f5f80a81cc4b2391f793ea7e7972d9573b95541969fab  vllm/v1/worker/gpu_model_runner.py
63da70d1ba43fc7161015f7be7ec7979b30cfa38aaa874b9620e7a1dbaf32c25  vllm/v1/worker/gpu_worker.py
0473bd01ec596832572fcd69bc48f2a3bf50418f5aa36a42e6cf496b8f86f9d9  vllm/model_executor/layers/fused_moe/layer.py
e7e25e6424c68d7e551100ee66f7f6a2b448702ad57d53a970782717d0edfe43  vllm/model_executor/models/deepseek_v2.py
000343de25bf789448272a54fdbba355b2cb22bc37de5b47ca357bd756c9301f  vllm/v1/attention/backends/mla/triton_mla.py
5f89b6d224bc6b2ee6a2382ab7196313ffe5ec32e180498fe47f6032bfbb9550  vllm/device_allocator/uvm.py
210be2e3d3cfe864b475e1f2e08fbef6f281f00c0dfdc6e221509006d8b0592f  vllm/device_allocator/uvm_static_alias.py
d82332a679101edba0739e798b3c20608724bca0cb9dd371458c62e62b88e112  KT SGLang model_runner_kv_cache_mixin.py
```

原始 arm 的完整环境和命令以 artifact 内的 `env.json`、`command.txt` 为准；不要
用后来修改过的通用 E2 配置覆盖它们。

## 3. 创建新的 artifact 与冷专家计划

必须使用尚不存在的新目录，禁止覆盖参考结果：

```bash
REPO=/home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=${REPO}/experiments/eurosys_moe_offload_evaluation/artifacts/c1-mechanism-min-offload-pilot-REPRO-attempt-01
REFERENCE_ARTIFACT=${REPO}/experiments/eurosys_moe_offload_evaluation/artifacts/c1-mechanism-min-offload-pilot-20260827-attempt-01
MODEL=/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7
TRACE=${REPO}/experiments/eurosys_moe_offload_evaluation/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/calibration_assets/routing/routing_trace.jsonl

test ! -e "${RUN_ROOT}"
test -f "${MODEL}/config.json"
test -f "${TRACE}"
mkdir -p "${RUN_ROOT}/provenance/cold1_plan"

git diff --binary -- \
  workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py \
  workloads/vllm/vllm/vllm/model_executor/models/deepseek_v2.py \
  workloads/vllm/vllm/vllm/model_executor/models/utils.py \
  workloads/vllm/vllm/vllm/v1/attention/backends/mla/triton_mla.py \
  >"${RUN_ROOT}/provenance/root_relevant.patch"
git -C KTransformers/third_party/sglang diff --binary -- \
  python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py \
  >"${RUN_ROOT}/provenance/kt_sglang_relevant.patch"

${REPO}/workloads/vllm/vllm/.venv/bin/python \
  ${REPO}/experiments/hybrid_offload_2_4_6_8_compare/prepare_cold_plans.py \
  --trace "${TRACE}" \
  --output-dir "${RUN_ROOT}/provenance/cold1_plan" \
  --serving-start-step 3 \
  --prefill-step-count 1 \
  --tier-widths c1=1 \
  >"${RUN_ROOT}/provenance/prepare_cold1.stdout.json"
```

验收 cold1：

```bash
jq -e '
  .tiers.c1.cold_experts_per_layer == 1 and
  .tiers.c1.selected_expert_count == 26 and
  .tiers.c1.selected_main_bytes == 449839104
' "${RUN_ROOT}/provenance/cold1_plan/manifest.json"

sha256sum "${RUN_ROOT}/provenance/cold1_plan/cold_plan_c1.json"
```

计划 SHA-256 应为：

```text
78f726d4907b662d143c78c304cc680a8790363b7ccfd472fd49414ed46f6d00
```

为 KTransformers 生成相同 cold-set 的 frequency placement。转换器要求四个 tier
文件，因此本 pilot 把相同 cold1 plan 复制成四个输入；最终只使用 `2g` 输出：

```bash
mkdir -p "${RUN_ROOT}/provenance/kt_cold1_input" \
         "${RUN_ROOT}/provenance/kt_cold1_placement"

for tier in 2g 4g 6g 8g; do
  cp "${RUN_ROOT}/provenance/cold1_plan/cold_plan_c1.json" \
     "${RUN_ROOT}/provenance/kt_cold1_input/cold_plan_${tier}.json"
done

${REPO}/workloads/vllm/vllm/.venv/bin/python \
  ${REPO}/experiments/hybrid_offload_2_4_6_8_compare/prepare_kt_frequency_plans.py \
  --input-dir "${RUN_ROOT}/provenance/kt_cold1_input" \
  --output-dir "${RUN_ROOT}/provenance/kt_cold1_placement"

jq -e '
  .tiers["2g"].gpu_experts_per_layer == 63 and
  .tiers["2g"].total_cpu_experts == 26 and
  .tiers["2g"].selected_main_bytes == 449839104
' "${RUN_ROOT}/provenance/kt_cold1_placement/kt_frequency_manifest.json"
```

## 4. 每个 arm 前必须执行的环境检查

六个 arm 必须使用六个独立冷进程，顺序执行，不能共享模型进程或 CUDA allocator。

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free \
  --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits
sudo fuser /dev/nvidia-uvm /dev/nvidia-uvm-tools 2>/dev/null || true
for port in 30301 30302 30303 30304 30305 30306; do
  ss -ltn "sport = :${port}"
done
```

开始每个 arm 前必须满足：

- GPU `memory.used` 为 0 MiB；
- 无 compute process；
- 无遗留 UVM client；
- 对应端口未监听；
- 模型、fixture、Python venv 和机制依赖存在；
- 没有其他 GPU 实验持有 E2/UVM lock。

公共环境：

```text
CUDA_VISIBLE_DEVICES=0
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=14
MKL_NUM_THREADS=14
PYTHONUNBUFFERED=1
TZ=Asia/Shanghai
```

每个 server 启动后等待 `/health` 返回成功，再启动 NVML 200 ms 采样，然后运行
3 个 warmup 和 1 个正式请求。客户端的公共命令模板为：

```bash
${CLIENT_PY} ${REPO}/experiments/eurosys_moe_offload_evaluation/streaming_client.py \
  --system "${CLIENT_SYSTEM}" \
  --model "${MODEL}" \
  --fixture "${REPO}/experiments/eurosys_moe_offload_evaluation/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/evaluation_short_dialogue.json" \
  --warmup-fixture "${REPO}/experiments/eurosys_moe_offload_evaluation/inputs/generated/eurosys-pa-v2-20260824/deepseek_v2_lite_bf16/calibration_short_dialogue.json" \
  --output "${ARM_DIR}/short_dialogue/client.json" \
  --base-url "http://127.0.0.1:${PORT}" \
  --concurrencies 1 \
  --warmup-requests 3 \
  --warmup-output-tokens 8 \
  --measurement-requests 1 \
  --minimum-request-start-interval 0 \
  --request-timeout 1800 \
  --barrier-timeout 60 \
  --seed 20260823
```

vLLM arms 的 `CLIENT_SYSTEM=ours`；KTransformers 使用 `ktransformers`；
MoE-Infinity 使用 `moe_infinity`。这里的名称只选择客户端协议适配，不改变服务端机制。

每条 server 启动命令执行后必须立即记录 PID，并使用统一生命周期：

```bash
SERVER_PID=$!
printf '%s\n' "${SERVER_PID}" >"${ARM_DIR}/server.pid"

ready=0
for attempt in $(seq 1 1800); do
  if curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null; then
    ready=1
    break
  fi
  kill -0 "${SERVER_PID}" 2>/dev/null || break
  sleep 1
done
test "${ready}" = 1

(
  while kill -0 "${SERVER_PID}" 2>/dev/null; do
    date --iso-8601=ns
    nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,clocks.sm,clocks.mem,pstate \
      --format=csv,noheader,nounits
    sleep 0.2
  done
) >"${ARM_DIR}/nvml.csv" 2>"${ARM_DIR}/nvml.err" &
NVML_PID=$!
```

客户端 Python：vLLM 与 MoE-Infinity arm 使用
`${REPO}/workloads/vllm/vllm/.venv/bin/python`；KTransformers arm 使用
`${REPO}/KTransformers/.venv/bin/python`。客户端结束后先保存 return code，再正常停止
server 和采样器：

```bash
kill -INT "${SERVER_PID}"
wait "${SERVER_PID}"
SERVER_RC=$?
printf '%s\n' "${SERVER_RC}" >"${ARM_DIR}/server_returncode.txt"
kill "${NVML_PID}" 2>/dev/null || true
wait "${NVML_PID}" 2>/dev/null || true
```

如 client 失败，仍应完成日志和 GPU/module 清理；不能直接跳过清理进入下一 arm。

## 5. 三个原生 vLLM arm

三个 arm 使用 stock UVM、eager execution、6,291,456 B KV、192-token 上限。

### 5.1 all-GPU

```bash
PORT=30301
ARM_DIR=${RUN_ROOT}/native_all_gpu
mkdir -p "${ARM_DIR}/short_dialogue"

env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 \
  PYTHONUNBUFFERED=1 TZ=Asia/Shanghai VLLM_DISABLE_OPENAI_METRICS=1 \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  taskset -c 0-13 ${REPO}/workloads/vllm/vllm/.venv/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 --port 30301 --disable-frontend-multiprocessing \
  --model "${MODEL}" --trust-remote-code --dtype bfloat16 --enforce-eager \
  --max-model-len 192 --gpu-memory-utilization 0.97 \
  --kv-cache-memory-bytes 6291456 --cpu-offload-gb 0 \
  --max-num-batched-tokens 192 --max-num-seqs 1 \
  --no-enable-prefix-caching --generation-config vllm \
  >"${ARM_DIR}/server.log" 2>&1 &
```

### 5.2 pinned UVA

除输出目录和端口外，server 参数与 all-GPU 相同，仅修改：

```text
port=30302
--cpu-offload-gb 0.01
VLLM_CPU_OFFLOAD_BACKEND=uva
VLLM_CPU_OFFLOAD_MANIFEST_FILE=$ARM_DIR/offload_manifest.jsonl
VLLM_CPU_OFFLOAD_MANIFEST_RESET=1
```

### 5.3 copy-on-forward

除输出目录和端口外，server 参数与 pinned UVA 相同，仅修改：

```text
port=30303
VLLM_CPU_OFFLOAD_BACKEND=copy
VLLM_CPU_OFFLOAD_COPY_SYNC=0
VLLM_CPU_OFFLOAD_COPY_TRACE=0
```

两种 `0.01 GiB` arm 均应在 manifest 中得到实际卸载 14,942,208 B。若卸载量不同，
说明 Parameter 遍历顺序或代码版本漂移，该次结果不能与参考表直接比较。

## 6. ours cold1 arm

### 6.1 加载 custom UVM

不要在存在 GPU/UVM client 时卸载模块。优先复用已有 harness 的 fail-closed module
lifecycle，而不是手写不带清理 trap 的 `insmod`：

```bash
export ALLOW_GPU_RUN=1
export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
export SLOT_BUDGET_BYTES=1207959552
export MANUAL_MIGRATION_FAULT_MODE=2
source ${REPO}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh

runtime_preflight
module_snapshot "${RUN_ROOT}/provenance/module.before.json" "${STOCK_UVM}"
load_custom_module
module_snapshot "${RUN_ROOT}/provenance/module.custom.json" "${CUSTOM_UVM}"
```

验收 custom module：

```bash
test "$(cat /sys/module/nvidia_uvm/srcversion)" = 5BA0A4BFB3AD96A55780A4C
test "$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes)" = 1207959552
test "$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable)" = 1
test "$(cat /sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode)" = 2
```

### 6.2 精确空间与 Graph 配置

`ours` 的关键参数：

```text
KV/native hole:             6,291,456 B（192 tokens）
workspace warmup:           128 tokens
fused-MoE chunk:            128 tokens
Graph mode:                 FULL_SEPARATE
Decode Graph:               batch 1
pure-Prefill Graph:         128 tokens
minimum post-Graph free:    201,326,592 B
phase mode:                 prefill_graph_remote_decode
PARK/RESUME:                enabled
```

启动命令：

```bash
ARM_DIR=${RUN_ROOT}/ours_cold1
mkdir -p "${ARM_DIR}/short_dialogue"

env \
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 \
  PYTHONUNBUFFERED=1 TZ=Asia/Shanghai \
  VLLM_DISABLE_OPENAI_METRICS=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  VLLM_ATTENTION_BACKEND=TRITON_MLA \
  VLLM_CPU_OFFLOAD_BACKEND=copy \
  VLLM_CPU_OFFLOAD_COPY_SYNC=0 VLLM_CPU_OFFLOAD_COPY_TRACE=0 \
  VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE="${RUN_ROOT}/provenance/cold1_plan/exact_parameter_names.txt" \
  VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1 \
  VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va \
  VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE="${RUN_ROOT}/provenance/cold1_plan/cold_plan_c1.json" \
  VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=auto \
  VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288 \
  VLLM_CPU_OFFLOAD_MANIFEST_FILE="${ARM_DIR}/offload_manifest.jsonl" \
  VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 \
  VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE=1 \
  VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1 \
  VLLM_FUSED_MOE_CHUNK_SIZE=128 \
  VLLM_TRITON_MLA_EXPERIMENTAL_FULL_CUDAGRAPH=1 \
  VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0 \
  VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
  VLLM_UVM_FUSED_MOE_POST_WORKSPACE_AUDIT_CAPTURE_WARMUP=1 \
  VLLM_UVM_FUSED_MOE_POST_WORKSPACE_AUDIT_MIN_TOKENS=128 \
  VLLM_UVM_FUSED_MOE_POST_WORKSPACE_RESIDENCY_AUDIT=1 \
  VLLM_UVM_FUSED_MOE_PRE_KERNEL_ALIAS_READ_GATE=1 \
  VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
  VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0 \
  VLLM_UVM_LOG_FILE="${ARM_DIR}/uvm.log" \
  VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES=201326592 \
  VLLM_UVM_ORIGINAL_VA_PHASE_MODE=prefill_graph_remote_decode \
  VLLM_UVM_ORIGINAL_VA_PHASE_RESULT_FILE="${ARM_DIR}/phase_transition.json" \
  VLLM_UVM_PHASE_BATCHED_SCHEDULING=1 \
  VLLM_UVM_PHASE_COHORT_KV_RESERVE_BLOCKS=1 \
  VLLM_UVM_PHASE_KV_AWARE_COHORTS=1 \
  VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS=128 \
  VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE=1 \
  VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ=0 \
  VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_BYTES=6291456 \
  VLLM_UVM_PRE_KV_NATIVE_HOLE_SENTINEL_EMPTY_CACHE=1 \
  VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE=1 \
  taskset -c 0-13 ${REPO}/workloads/vllm/vllm/.venv/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 --port 30304 --disable-frontend-multiprocessing \
  --model "${MODEL}" --trust-remote-code --dtype bfloat16 \
  --max-model-len 192 --gpu-memory-utilization 0.97 \
  --kv-cache-memory-bytes 6291456 --cpu-offload-gb 30 \
  --max-num-batched-tokens 128 --max-num-seqs 1 \
  --no-enable-prefix-caching --generation-config vllm \
  --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_SEPARATE","cudagraph_capture_sizes":[1,128],"max_cudagraph_capture_size":128}' \
  >"${ARM_DIR}/server.log" 2>&1 &
```

这里 `--cpu-offload-gb 30` 只是让 exact-name/plan 路径覆盖完整候选 Parameter；实际
卸载量由 cold1 plan 限定为 449,839,104 B，不是卸载 30 GiB。

完成请求并停止 server 后，必须恢复 stock UVM：

```bash
wait_gpu_clean
restore_stock
module_snapshot "${RUN_ROOT}/provenance/module.after.json" "${STOCK_UVM}"
restart_exporters
```

不得在 server 尚存活时执行 `restore_stock`。若退出 trap 已经恢复模块，重复调用是
幂等的。参考 stock srcversion 为 `85A79790636BBD99BA3E43B`。

### 6.3 ours 验收日志

必须确认：

- Decode-C1 和 pure-Prefill-128 Graph 均捕获成功；
- 6 MiB native hole 已分配、触碰并在 Prefill Graph 前释放；
- allocator 预热 128 tokens，预留 12,582,912 B；
- 正式请求前 cold 449,839,104 B 全在 CPU；
- hot 28,339,863,552 B 全在 GPU；
- free HBM >= 201,326,592 B；
- PARK、remote map、RESUME 全部 `ok=true`；
- 无 unknown residency、CUDA illegal access、Xid、PTE/PDE fault。

## 7. KTransformers cold1 arm

必须先确认 stock UVM 已恢复。启动命令：

```bash
ARM_DIR=${RUN_ROOT}/ktransformers_cold1
mkdir -p "${ARM_DIR}/short_dialogue"

env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 \
  PYTHONUNBUFFERED=1 TZ=Asia/Shanghai \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  SGLANG_APPLY_CONFIG_BACKUP=none \
  PATH="${REPO}/KTransformers/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  taskset -c 0-13 ${REPO}/KTransformers/.venv/bin/python \
  -m sglang.launch_server \
  --model-path "${MODEL}" --host 127.0.0.1 --port 30305 --tp-size 1 \
  --context-length 194 --max-running-requests 1 --mem-fraction-static 0.97 \
  --max-total-tokens 192 --chunked-prefill-size 128 --max-prefill-tokens 128 \
  --schedule-policy fcfs --stream-interval 1 --random-seed 20260823 \
  --trust-remote-code --kt-weight-path "${MODEL}" --kt-method BF16 \
  --kt-cpuinfer 14 --kt-threadpool-count 1 --kt-numa-nodes 0 \
  --kt-num-gpu-experts 63 --kt-expert-placement-strategy frequency \
  --init-expert-location "${RUN_ROOT}/provenance/kt_cold1_placement/kt_frequency_2g.pt" \
  --kt-max-deferred-experts-per-token 0 --kt-gpu-prefill-token-threshold 0 \
  --attention-backend flashinfer --cuda-graph-max-bs 1 --cuda-graph-bs 1 \
  --enable-mixed-chunk --disable-shared-experts-fusion \
  --disable-radix-cache --enable-metrics \
  >"${ARM_DIR}/server.log" 2>&1 &
```

日志必须明确给出 26 层均为 63 GPU + 1 CPU expert，总计 1,638 GPU experts 和
26 CPU experts。不要仅根据命令行的 `--kt-num-gpu-experts 63` 推断 placement 成功。

## 8. MoE-Infinity arm 与边界搜索

### 8.1 fresh build receipt

先验证 `_store` receipt：

```bash
jq -e '
  .status == "complete" and
  .returncode == 0 and
  .moe_infinity_commit == "92c2bc9266e33bbed22405356ca7ca7e71ae8974" and
  .extension_sha256 == "3946666dde229528870cde08989e0681ce2f9f35becf4d8a2e5ca8dfa0a30f3b"
' "${REPO}/experiments/eurosys_moe_offload_evaluation/artifacts/c1-mechanism-min-offload-pilot-20260827-attempt-01/provenance/moe_infinity_store_build/store_build_receipt.json"
```

若当前 tree 或 extension 与 receipt 不匹配，使用新目录重建，不能修改旧 receipt：

```bash
bash ${REPO}/experiments/eurosys_moe_offload_evaluation/rebuild_moe_infinity_store.sh \
  "${RUN_ROOT}/provenance/moe_infinity_store_build"
```

### 8.2 ratio 运行

参考 offload store：

```text
/home/ubuntu/moe_infinity_offload/
  deepseek_v2_lite_chat_85864749cd611b4353ce1decdb286193298f64c7
```

最终成功 arm：

```bash
ARM_DIR=${RUN_ROOT}/moe_infinity_ratio_083
mkdir -p "${ARM_DIR}/short_dialogue"

env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 \
  PYTHONUNBUFFERED=1 TZ=Asia/Shanghai \
  taskset -c 0-13 ${REPO}/workloads/vllm/vllm/.venv/bin/python \
  ${REPO}/experiments/eurosys_moe_offload_evaluation/launch_moe_infinity_server.py \
  --model "${MODEL}" \
  --offload-path /home/ubuntu/moe_infinity_offload/deepseek_v2_lite_chat_85864749cd611b4353ce1decdb286193298f64c7 \
  --host 127.0.0.1 --port 30306 \
  --device-memory-ratio 0.83 --kv-cache-ratio 0.01 \
  --max-batch-size 1 --num-threads 1 \
  >"${ARM_DIR}/server.log" 2>&1 &
```

要复现最大成功 ratio，还需在独立冷进程中依次验证：

```text
0.80 -> success
0.83 -> success
0.84 -> expected OOM at warmup-1
0.85 -> expected OOM at warmup-1
0.90 -> expected OOM at warmup-1
```

0.84/0.85/0.90 的 server 能达到 healthy，warmup-0 也可能成功；第二个 warmup 引入
新专家集合时，demand cache 在 `device_caching_allocator.cpp:28` 的 `cudaMalloc`
OOM 才是边界失败。每个 ratio 必须重启进程并确认 GPU 回到 0 MiB，不能在同一进程
中连续调 ratio。

## 9. 统一采集、停止与验收

每个 arm 都应保存：

```text
command.txt                 实际 server 命令
env.json                    实际环境覆盖
gpu.before/ready/after.json GPU 生命周期
server.log                  server stdout/stderr
nvml.csv                    200 ms GPU 采样
short_dialogue/command.txt  实际 client 命令
short_dialogue/client.json  逐 token 指标
short_dialogue/client.log   client stdout/stderr
verification.json           arm 验收
```

`ours` 还必须保存 `offload_manifest.jsonl`、`uvm.log`、`phase_transition.json` 和
custom-module snapshots；MoE-Infinity 保存 `/admin/stats` 结果。

先向 server 发送正常终止信号并等待退出；不要直接卸载 UVM。若 server 未退出，保存
日志后只终止 artifact 中记录的 PID。结束后重新执行第 4 节 GPU/UVM 检查。

客户端验收模板：

```bash
jq -e '
  .status == "ok" and
  .phases[0].concurrency == 1 and
  .phases[0].request_count == 1 and
  .phases[0].failed_requests == 0 and
  .phases[0].completed_output_tokens == 32 and
  .phases[0].requests[0].prompt_token_ids_sha256 ==
    "eb9a77425f9afefa503200bd45a694762c265707a3255a0c6fcbaceb15c73d5e"
' "${ARM_DIR}/short_dialogue/client.json"
```

fatal 日志检查：

```bash
rg -n -i \
  'CUDA error|illegal memory access|out of memory|\bXid\b|PTE.*fault|PDE.*fault|FAULT_PDE|FAULT_PTE|Traceback' \
  "${ARM_DIR}/server.log"
```

成功 arm 应无匹配；MoE-Infinity 的 0.84/0.85/0.90 边界尝试应保留预期 OOM，不能
把它们标为成功。KTransformers 正常 shutdown 的已知 KeyboardInterrupt traceback
可单独标注，不能用它忽略运行期间的其他 traceback。

vLLM all-GPU/UVA/copy/ours 的正式 output-token-ID SHA-256 应同为：

```text
bc80295f4b1f6afcb032c4d3e7709d3049f987fc6a4b7b0f3df054c786a62547
```

KTransformers 和 MoE-Infinity 在参考运行中输出 hash 不同。因此它们只通过请求成功、
token 数与指标完整性检查，没有通过跨框架 exact-generation gate。

## 10. 参考结果

| arm | TTFT ms | TPOT ms | E2E ms | Peak HBM MiB |
|---|---:|---:|---:|---:|
| vLLM all-GPU | 100.43 | 56.95 | 1865.97 | 30,943 |
| vLLM pinned UVA | 100.80 | 56.00 | 1836.76 | 30,931 |
| vLLM copy-on-forward | 97.43 | 59.45 | 1940.35 | 30,963 |
| ours cold1 | 279.53 | 21.38 | 942.24 | 30,799 |
| KTransformers cold1 | 370.55 | 9.58 | 667.66 | 31,051 |
| MoE-Infinity ratio 0.83 | 938.07 | 289.99 | 9927.76 | 31,925 |

结果允许因机器状态产生波动，但配置、请求 hash、成功数、卸载量、Graph 集、驻留检查
和清理状态必须完全一致。单次延迟接近参考值不等于完整复现成功。

## 11. ours 426.84 ms 切换开销根因

### 11.1 粗粒度计时字段容易误读

参考 `ours_cold1/phase_transition.json` 的正式请求 transition：

| transition 字段 | 时间 | 含义 |
|---|---:|---|
| `sync_ns` | 0.109 ms | cutover 前 CUDA synchronize |
| `alias_detach_ns` | 289.434 ms | 两次 hot read gate + PARK；不是纯 PARK 时间 |
| `remote_map_ns` | 2.201 ms | 为 52 个 cold ranges 建立 remote-read map |
| `physical_audit_ns` | 135.097 ms | post-map gate + cold/hot physical-residency audit |
| `wall_ns` | 426.842 ms | 完整 cutover |

`alias_detach_ns=289.434 ms` 过去被简称为“PARK/alias detach”，但 PARK ioctl 自己在
同一结果中明确记录：

```text
park.wall_ns       = 38.446 ms
park.ioctl_wall_ns = 38.309 ms
processed aliases  = 858
TLB invalidations  = 858
```

所以真正已测到的拓扑控制操作为：

```text
PARK ioctl + remote map = 38.446 + 2.201 = 40.647 ms
```

其余安全门和审计为：

```text
(289.434 - 38.446) + 135.097 = 386.085 ms
386.085 / 426.842 = 90.5%
```

也就是说，约 90.5% 并非在搬运 449.84 MB cold expert，也不是 CUDA Graph kernel
本身，而是诊断级正确性证明位于首个 Decode 的同步关键路径。

可直接复核：

```bash
jq '
  .transition as $t |
  {
    wall_ms: ($t.wall_ns / 1000000),
    park_ms: ($t.park.wall_ns / 1000000),
    remote_map_ms: ($t.remote_map_ns / 1000000),
    audit_ms: ((
      $t.alias_detach_ns - $t.park.wall_ns + $t.physical_audit_ns
    ) / 1000000),
    audit_percent: ((
      $t.alias_detach_ns - $t.park.wall_ns + $t.physical_audit_ns
    ) * 100 / $t.wall_ns)
  }
' "${REFERENCE_ARTIFACT}/ours_cold1/phase_transition.json"
```

### 11.2 为什么审计如此昂贵

对应实现位置：

```text
workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:3336
  首个 Decode step 触发 first_decode_cutover
workloads/vllm/vllm/vllm/model_executor/models/utils.py:6980
  transition_cpu_offload_uvm_original_va_to_remote
workloads/vllm/vllm/vllm/model_executor/models/utils.py:3372
  _uvm_original_alias_hot_cuda_read_gate
```

`VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE=1` 时，runtime cutover 会在三个位置调用
hot CUDA-read gate：

1. PARK 前；
2. PARK 后；
3. remote map 后。

每次 runtime gate 都先对 104 个 hot ranges、共 28,339,863,552 B 执行页粒度物理
驻留查询，再按每 2 MiB 一个样本，从 52 个 Parameter 中建立 13,546 个 GPU index，
执行真实 CUDA read 并与基线 checksum 比较。transition 最后还再次查询 52 个 cold
ranges 和 104 个 hot ranges，验证：

```text
cold: 449,839,104 B CPU-only
hot:  28,339,863,552 B GPU-only
unknown: 0 B
```

这些检查用于捕获 missing PDE/PTE、热门专家被 KV/临时分配驱逐、stable VA 内容变化
等真实故障；但页元数据遍历、GPU checksum、同步和重复查询都在首个 Decode 前串行
完成，因此放大了第一个 ITL。

执行链如下：

```text
首个 Decode step
  -> hot residency query + sampled CUDA-read（PARK 前）
  -> PARK 858 consumer aliases + TLB invalidation
  -> hot residency query + sampled CUDA-read（PARK 后）
  -> remote-map 52 cold ranges
  -> hot residency query + sampled CUDA-read（map 后）
  -> cold/hot 全量 physical-residency audit
  -> 执行第一个 Decode Graph
```

KTransformers 使用静态 63-GPU/1-CPU layout，不在请求中途执行上述 PTE/alias 切换和
三重审计，因此没有这次一次性停顿。

### 11.3 对 TPOT/E2E 的影响

参考正式请求：

```text
first ITL:              446.551 ms
phase transition:       426.842 ms
first-ITL residual:      19.710 ms
later 30 ITL mean:        7.21 ms
reported TPOT:           21.38 ms
```

因此 `ours` 稳态 Decode 并不慢；TPOT 是被一次性 transition 拉高。去掉 transition
得到的 515.40 ms E2E 是后处理估算，不是另一次实测，正式报告必须同时给出原始
端到端结果、steady TPOT 和 transition 时间。

### 11.4 可优化项与验证顺序

复现本轮结果时必须保持 gate 开启，否则不是同一 arm。后续优化应另建配置并依次验证：

1. 把全量 hot residency/checksum 证明移到 warmup 完成、measurement 开始前；
2. runtime 只保留 fail-closed 的 PARK、remote-map、stable-VA 和必要轻量抽样合同；
3. 缓存 hot-range 元数据，消除 PARK 前/后/map 后三次重复全量查询；
4. 在多个同类请求间保持 remote-parked Decode topology，按 cohort 摊薄一次切换；
5. 再单独优化 PARK 的 858 次 PTE/TLB 工作，例如 grouped PTE/batched invalidate。

不能直接删除断言或所有 gate 后宣称获得加速。最低验证要求是同一 cold1、KV、Graph、
请求和 custom UVM 下做 gate-on/off 配对，多次独立冷进程运行，并同时检查 output hash、
hot/cold residency、CUDA/Xid/PTE/PDE fault 和 stock-UVM cleanup。当前证据可以确认
“诊断审计主导延迟”，但不能仅凭后处理数据断言关闭审计后的最终实测 TPOT。

## 12. 常见失败、原因与处理

### Graph 或 KV 初始化 OOM

确认只捕获 Decode-C1 和 Prefill-128，KV 为 6 MiB，native hole 也是 6 MiB。若误用
完整 E2 的 2 GiB KV 或 C1/C4/C8/C16 全 Graph 集，cold1 不再具有相同容量条件。
不要只调换 sentinel 顺序；总容量不相容时必须把它记作新配置。

### warmup 后热门专家不再全在 GPU

检查正式请求前的 serving-residency reconciliation。必须在 KV、workspace、sampler
和 Graph pool 建立后恢复 hot experts；否则后续分配可能已经挤占热点专家。重复
`rebalance` 不能弥补真实容量不足。

### ours 出现 CUDA illegal access 或 PTE/PDE fault

保留第一个 kernel fault、`phase_transition.json`、manifest、UVM log 和 kernel log。
先核对 custom module srcversion、slot budget、PTE remap、manual fault mode、PARK/
RESUME 返回值和 unknown bytes。`Connection refused` 或 HTTP stream 截断通常只是
server 已崩溃的后果。

### MoE-Infinity warmup-1 OOM

若 ratio >= 0.84，这正是本轮边界结果；不要缩短 warmup 或换请求规避。ratio=0.83
也失败时，检查 fresh receipt、offload store、GPU 是否真正清空，以及 activation/KV
ratio 是否漂移。

### 输出 hash 不一致

先确认相同 frozen token IDs、temperature、EOS 策略、输出长度和 seed。参考运行中
KT/MoE-Infinity 仍未通过 exact-generation gate，因此不能把跨框架 hash 不同静默
标成通过，也不能据此直接做最终性能排名。

### Peak HBM 看似相等

参考 peak 相差最多约 1.1 GiB，明确不是 equal-HBM。NVML 离散采样值相近不代表静态
预算或可用 HBM 相同。若要论文排名，必须另做等实际峰值 HBM 的配平矩阵和独立重复。
