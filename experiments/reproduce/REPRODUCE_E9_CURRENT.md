# 复现 E9 KTransformers placement/CPU消融正式矩阵

最后核验：2026-09-02。当前通过的正式artifact为
`experiments/eurosys_moe_offload_evaluation/artifacts/e9-formal-20260902-attempt-01/`，
顶层`verification.json`为`status=ok`。活动KTransformers HEAD和父仓库gitlink均为
`5a9c7d4cfaaed0136930e4b629ce5d55cec71a54`，其SGLang子模块为
`f3ef6723f61ecbfa3cd28b76eea08af6da49a7d0`且工作树无patch。

## 1. 实验范围

固定Qwen3-30B-A3B-FP8、同一ShareGPT evaluation fixture、32输出token、C1/C8和同一
GPU expert budget，比较七臂：

- baseline：14 CPU threads、CUDA Graph、frequency placement、overlap、mixed chunk；
- threads7和threads28：只改变CPUInfer线程数；
- eager：只关闭CUDA Graph；
- uniform：只把frequency placement改为相同预算的逐层uniform placement；
- no-overlap：只关闭overlap scheduler；
- no-mixed：只关闭mixed chunk。

本机只有一个NUMA node，因此跨NUMA臂合法记为N/A。正式模式包含5个process blocks、
每臂5个独立冷进程，使用轮转正反顺序，共35个case和70个C1/C8 phase。

## 2. 前置检查

```bash
cd /home/ubuntu/nvidia-uvm-gpu

git -C KTransformers rev-parse HEAD
git -C KTransformers/third_party/sglang rev-parse HEAD
git -C KTransformers/third_party/sglang diff --binary | sha256sum

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits
ss -ltnp | rg ':30000\b' || true
```

期望依次为上述两个commit、空patch SHA256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`、GPU约0 MiB、
无compute process且30000端口未占用。runner还会核验14个物理核、单NUMA、模型snapshot、
数据集hash、`kt_kernel 0.6.4/avx512_bf16`和冻结calibration输入。

## 3. 正式运行

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT="$PWD/experiments/eurosys_moe_offload_evaluation/artifacts/\
e9-formal-NEW-attempt-01" \
ALLOW_GPU_RUN=1 MODE=formal PROCESS_BLOCKS=5 \
bash experiments/eurosys_moe_offload_evaluation/run_formal_e9.sh
```

`RUN_ROOT`必须是全新且不存在的路径；runner不会覆盖成功或失败artifact。一个正式命令只
启动这一遍矩阵，但矩阵本身按协议包含每臂5个独立冷进程。不要用
`MODE=qualification PROCESS_BLOCKS=1`冒充正式结果。

## 4. 验收与结果读取

```bash
ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/e9-formal-NEW-attempt-01

jq '{status, mode, process_blocks, case_count, correctness_gate,
     exact_outputs_equal_across_all_arms}' "$ROOT/verification.json"

jq -r '
  ["arm","phase","tps_median","tps_min","tps_max",
   "ttft_median_ms","tpot_median_ms"],
  (.aggregate | to_entries[] as $arm | $arm.value | to_entries[] |
   [$arm.key,.key,.value.throughput_tps.median,
    .value.throughput_tps.min,.value.throughput_tps.max,
    .value.mean_ttft_ms.median,.value.mean_tpot_ms.median]) | @tsv
' "$ROOT/verification.json"

rg -il 'NVRM: Xid|FAULT_PDE|PTE fault|kernel panic' \
  "$ROOT"/cases/*/kernel.since.log
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
```

通过条件：35个case全部返回0；每个case的benchmark、runtime verifier和kernel log通过；
所有七臂分别在5个冷进程间bitwise稳定；baseline、threads7、threads28、eager、no-overlap
在相同block内与baseline逐token exact。uniform会改变CPU/GPU专家计算和归约路径，
no-mixed会改变batch formation；二者的跨臂token差异必须量化报告，但不以autoregressive
后缀exact作为execution gate。

参考正式结果为35/35 case和10,080/10,080正式输出token完成。7线程相对14线程baseline
的C1/C8中位吞吐为4.63×/6.09×；frequency相对uniform为4.26×/1.44×。完整数值、
min--max和bootstrap 95% CI均在`verification.json`。

## 5. 正确性边界

该协议验证执行正确性、同臂可重复性和不改变数值路径的控制臂exact一致性。它没有生成
teacher-forced logits、top-1 agreement或perplexity，因此不能把uniform/no-mixed的性能
表述为数值质量等价。参考结果中uniform的C1/C8 exact request比例为0/5和10/40，
no-mixed为5/5和35/40；这些差异必须保留，不能删除或写成全七臂exact。

冻结selection reference为
`experiments/eurosys_moe_offload_evaluation/inputs/e9_sharegpt_selection_reference_20260828.json`。
