# 复现 E0/E1 当前证据

最后核验：2026-09-02。没有比`eurosys-pa-v2-20260824`更新且通过完整验收的E0/E1
artifact；因此本文件仍是最新可用的离线复核入口，但不是完整的一键GPU复现协议。

当前证据冻结于 `eurosys-pa-v2-20260824`。E0 是三系统基础兼容性qualification；E1是
maximum-resident描述性anchor。E1虽然完成15个server process blocks，但跨系统完整输出
不一致，因此publication gate仍为held。

权威结果和冻结参数见：

- `../eurosys_moe_offload_evaluation/RESULTS_E0_E1_20260824.md`
- `../eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/protocol.json`

## 离线复核

```bash
cd /home/ubuntu/nvidia-uvm-gpu
RUN_ROOT=experiments/eurosys_moe_offload_evaluation/artifacts/\
eurosys-pa-v2-20260824T142031Z-e0-e1-formal

jq '.status' "$RUN_ROOT/cases/E0/verification.json"
jq '.status,.checks.observed_verified_process_blocks' \
  "$RUN_ROOT/cases/E1/verification.json"

workloads/vllm/vllm/.venv/bin/python \
  experiments/eurosys_moe_offload_evaluation/diagnose_formal_e1_outputs.py \
  "$RUN_ROOT/cases/E1" --protocol-id eurosys-pa-v2-20260824 \
  --output /tmp/e1-output-diagnostic-new.json
```

## 重跑边界

E1执行器为`run_formal_e1.sh`，但它要求先建立带相同protocol和fixture manifest的新run
root。当前没有封装三系统E0加E1的单命令复现器；不得把旧retained E0证书冒充新运行。
