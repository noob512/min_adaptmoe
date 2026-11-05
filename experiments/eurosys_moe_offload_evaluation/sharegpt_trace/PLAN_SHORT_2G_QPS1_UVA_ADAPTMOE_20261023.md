# Short、QPS=1、等卸载量 UVA/AdaptMoE 复跑

本轮按用户最新要求，只测 Short 的 Native UVA 和 AdaptMoE，提供速率从 0.5 改为 1 req/s。前一轮已按要求停止，未完成的 Long/AdaptMoE 不纳入结果。

保留同一组 64 条 ShareGPT 请求、128-token prompt 前缀、512-token 输出、ignore_eos，计划到达时刻改为 0,1,...,63 秒。每臂独立冷启动一次，先执行 4 条并发预热，各输出 8 tokens；正式回放一次，不额外重复。

Qwen3-30B-A3B-FP8、RTX 5090；TRITON_ATTN，eager，无 CUDA Graph，混合调度，C64、8192-token bucket、96 client connections。两臂固定 1.5 GiB KV（1023 个有效 blocks、16368 tokens），不作 KV/专家档位调整。

沿用前一轮匹配好的“2 GiB 档”：**实际均为 2,264,924,160 B = 2.109375 GiB**，字节差为 0。AdaptMoE 为 cold10（48 层各 10 个专家，96 个 weight bindings），UVA 使用冻结的 12 个完整 Parameter；量相同、坐标不同，不能当作相同权重坐标的纯传输消融。现有逐层等宽静态计划不能精确表示 2.000 GiB。

AdaptMoE 采用固定 cold10 单 plan class、KV initial/min/max/allowed target 全部为 1023，并关闭主动增长；仍按每个 scheduler step 总 token 数在 ≤16 remote、≥128 copy 间切换，滞回区保持状态。关闭运行期全量诊断和 read gate，保留安全初始化。

指标为请求均值 TTFT/TPOT/E2E，以及含排空时间的完成 QPS。QPS=1 是提供速率，实际完成 QPS 不保证达到 1。审核 2 个配置、128/128 成功、完整 512-token 输出、trace hash 和到达时刻一致、卸载字节相同、Graph 关闭、真实 mixed steps、没有 KV/expert tier commits。

复现：

```bash
bash experiments/eurosys_moe_offload_evaluation/sharegpt_trace/run_short_2g_qps1_uva_adaptmoe.sh \
 /home/ubuntu/nvidia-uvm-gpu/experiments/eurosys_moe_offload_evaluation/sharegpt_trace/artifacts/short-2g-qps1-uva-adaptmoe-20261023-attempt-01
```

矩阵脚本 `run_mixed_2g_qps05_uva_adaptmoe.sh` 现支持 `OFFERED_QPS` 和 `TRACE_NAMES`，默认仍是前一轮 0.5、三种 trace。新 wrapper 将其设为 1 和 short。分析使用 `analyze_mixed_2g.py`，预期配置数由冻结 manifest 确定。
