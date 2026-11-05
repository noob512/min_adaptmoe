# 复现 E4 并发与PCIe饱和qualification

最后核验：2026-09-02。当前权威artifact为：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/e4-repro-20260902-attempt-01
```

该轮使用固定always-remote exact-coordinate计划，5个独立进程按
`asc/desc/asc/desc/asc`扫描C1--C32，每档输出128 tokens。分析verifier为`status=ok`，
5个process block全部完成，输出hash已记录，关闭后GPU为0 MiB并恢复stock UVM。

## 运行

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT="$PWD/experiments/eurosys_moe_offload_evaluation/artifacts/e4-repro-NEW-attempt-01" \
PROCESS_BLOCKS=5 ALLOW_GPU_RUN=1 \
bash experiments/eurosys_moe_offload_evaluation/run_formal_e4.sh

workloads/vllm/vllm/.venv/bin/python \
  experiments/eurosys_moe_offload_evaluation/verify_formal_e4.py "$RUN_ROOT"
```

输出目录必须不存在。开始前要求GPU 0 MiB、无compute/UVM client；失败attempt必须保留。

## 当前结果

| C | request TPOT | Decode tok/s | PCIe RX |
|---:|---:|---:|---:|
| 1 | 87.41 ms | 11.60 | 0.166 GiB/s |
| 4 | 87.96 ms | 45.89 | 0.397 GiB/s |
| 8 | 89.80 ms | 89.87 | 1.387 GiB/s |
| 16 | 89.75 ms | 180.33 | 2.603 GiB/s |
| 24 | 85.02 ms | 284.77 | 3.841 GiB/s |
| 32 | 86.56 ms | 372.94 | 4.698 GiB/s |

预注册TPOT持续退化判据在C1--C32均未触发，峰值PCIe RX为Gen5 x16理论单向带宽的
8.00%。CPU DRAM带宽在当前KVM环境为N/A，不能用PCIe RX替代。该结论只覆盖本计划和
请求；本轮不是不同offload机制之间的比较。

离线复核：

```bash
jq '.status,.process_blocks,.concurrency_levels,.decision' \
  experiments/eurosys_moe_offload_evaluation/artifacts/e4-repro-20260902-attempt-01/analysis/verification.json
```
