# AdaptMoE 固定搬运拓扑的 mixed-schedule 资格结果

运行日期：2026-09-10（Asia/Shanghai）。模型为 DeepSeek-V2-Lite-Chat，单张
RTX 5090。4 个请求先到达，在其 Decode 期间再注入 4 个 128-token 请求；每个请求
固定生成 16 token。

## 结论

固定搬运拓扑在 continuous mixed 调度下可以完整运行。第 5 个 scheduler step 确实
同时包含 `4 x 128` Prefill token 和 `4 x 1` Decode token；8/8 请求均完成，未出现
remote transition、UVM poison、GPU Xid 或清理泄漏。

实现全程保持 sealed Parameter alias：Prefill/mixed step 使用 eager H2D，纯 Decode
使用包含 H2D 节点的 `FULL_DECODE_ONLY` Graph。运行记录了 130 条 serving/eager H2D
证据和 78 条 Graph-capture H2D 证据。冻结计划每次完整 forward 搬运
4,948,230,144 bytes；20 个 engine step 的逻辑搬运量约 98.965 GB。

## 初步性能

| 指标 | Always-copy mixed | Continuous remote 参考 | 变化 |
|---|---:|---:|---:|
| 请求吞吐 | 3.327 req/s | 6.016 req/s | -44.7% |
| token 吞吐 | 53.229 token/s | 96.253 token/s | -44.7% |
| 平均 TTFT | 219.38 ms | 174.84 ms | +25.5% |
| 平均 TPOT | 111.65 ms | 55.10 ms | +102.6% |
| 平均 E2E | 1894.13 ms | 1001.38 ms | +89.2% |
| 平均排队 | 0.332 ms | 0.298 ms | +0.033 ms |

排队时间基本相同，差异主要出现在执行阶段；这与 Decode 每一步重复搬运约 4.95 GB
权重的机制相符。就这个短请求资格点看，固定搬运消除了 phase-epoch HOL blocking，
但仍明显慢于固定 remote-read continuous arm。

## 正确性与适用范围

所有请求都生成了规定的 16 token，但与 continuous-remote 参考相比只有 6/8 个请求
的完整 token hash 一致；另外两个请求发生生成分叉。因此本结果证明的是机制可执行和
资源生命周期安全，尚不能作为严格的跨拓扑输出等价证明。

本轮只有一个 always-copy 冷进程，并复用了此前同轨迹 remote arm 作为参考；没有做
Poisson offered-load 扫描、ABBA、多 seed 或同轮配对。表中数字只能作为第一阶性能
趋势，不能直接用于论文正式定量结论。

原始结果见
[`artifacts/adaptmoe-always-copy-mixed-20260909T163532Z-qualification/verification.json`](artifacts/adaptmoe-always-copy-mixed-20260909T163532Z-qualification/verification.json)
和
[`artifacts/adaptmoe-always-copy-mixed-20260909T163532Z-qualification/result.json`](artifacts/adaptmoe-always-copy-mixed-20260909T163532Z-qualification/result.json)。
