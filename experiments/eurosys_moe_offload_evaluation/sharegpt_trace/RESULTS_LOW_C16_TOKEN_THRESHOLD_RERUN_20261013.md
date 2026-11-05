# ShareGPT low / C16 / token-threshold 复跑结果（2026-10-13）

## 1. 实验协议

- 模型：Qwen3-30B-A3B-FP8
- 输入：冻结 ShareGPT low trace，128 条请求，输入/输出长度格各 8 条
- 并发上限：16；`max_num_batched_tokens=8192`
- KV cache：1.75 GiB（`1879048192` B）
- CUDA Graph：关闭，vLLM `--enforce-eager` / `cudagraph_mode=NONE`
- warmup：4 条独立请求，不计入正式窗口
- AdaptMoE：`phase_mode=token_threshold`，`PARK=16`，`RESUME=128`
- phase-batched scheduling、KV cohort 和运行期驻留审计均关闭
- `VLLM_UVM_TOKEN_THRESHOLD_FAST_TRANSITION=1`
- Trace SHA-256：`416fe6e9bd464fec30259bdc375c462a1486e252b5a2555176452ab7253d98a2`

完整 artifact：
`artifacts/eager-low-c16-token-threshold-20261013-rerun02/`

## 2. 结果

| 指标 | AdaptMoE token-threshold |
|---|---:|
| 成功请求 | 128/128 |
| 测量窗口 | 500.514 s |
| offered QPS | 0.263708 |
| achieved QPS | 0.255737 |
| TTFT mean / p50 / p95 | 679.49 / 251.62 / 3768.60 ms |
| TPOT mean / p50 / p95 | 57.37 / 56.43 / 67.29 ms |
| E2E mean / p50 / p95 | 11307.01 / 8182.56 / 29557.08 ms |
| output token/s | 48.458 |

## 3. 机制是否实际生效

manifest 共记录 89 次 `alias_prefill_activated`（RESUME，remote -> copy）、
90 次 `remote_transition`（其中 89 次正式 PARK，另 1 次服务结束清理）。
所有正式 PARK 的 `reason` 均为 `token_threshold_remote`，说明不是旧的
`prefill_remote_decode` cohort 调度。PARK 的 wall time 合计 4.387 s，均值
48.74 ms；89 次 RESUME wall time 合计 10.765 s，均值 120.96 ms。
阈值切换事件均为 `ok=true`、`poisoned=false`。

本次使用的滞回规则是：调度 token 数 `<=16` 时切到 remote/UVA，
`>=128` 时切到 copy，中间区间保持当前拓扑。因此请求可以在 Decode 期间
继续接收新的 Prefill，不再等待整个 Decode cohort 结束。

## 4. 与此前结果的区别

此前 `RESULTS_LOW_C16_EAGER_ONCE_20261013.md` 的 AdaptMoE 使用严格
`prefill_remote_decode`，只有 26 次阶段切换，QPS 为 0.245809，TTFT mean
为 26,707.59 ms。token-threshold 复跑将 TTFT mean 降到 679.49 ms，p50
降到 251.62 ms；这证明此前的 20 秒级 TTFT 主要来自阶段 cohort 的队头等待，
而不是一次 Prefill forward 本身需要几十秒。

同一 low trace 的已有原生 eager 参考为：

| 机制 | QPS | TTFT mean | TPOT mean |
|---|---:|---:|---:|
| AdaptMoE token-threshold | 0.255737 | 679.49 ms | 57.37 ms |
| Native UVA | 0.256633 | 916.89 ms | 53.72 ms |
| Native Copy | 0.255121 | 667.09 ms | 59.88 ms |

这三行不是多次重复的统计结论：每臂目前只有一次独立冷启动，且 Native
参考来自同一 low trace 的前一轮 artifact；它们适合说明量级和机制方向，
不应当作置信区间意义上的胜负排名。

## 5. 异常记录

第一次完整复跑的 `rerun01` 忘记传入 1.75 GiB KV 配置，退回默认 2 GiB，
在请求前的 dummy warmup 出现 CUDA illegal memory access，未计入结果。
恢复 `KV_CACHE_BYTES=1879048192` 后，`rerun02` 启动和 128 条正式请求均成功，
server log 无 Xid、PTE/PDE、poison 或断连错误。
