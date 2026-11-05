# 固定虚拟地址的阶段感知 MoE 卸载：完整技术方案

> 本文是早期阶段调度实现快照。当前新增的 mixed-step `token_threshold` 与动态 KV
> 配置请先阅读 [2026-10-26 项目梳理](ADAPTMOE_PROJECT_GUIDE.md)。

日期：2026-08-24
状态：实现与实验现状说明，不等同于最终论文设计

## 1. 一句话说明

本方案让同一批 MoE 专家权重始终保留原来的 `Parameter` 对象和虚拟地址，但在
不同推理阶段切换这些地址背后的物理位置：

- Prefill：把选定权重搬到可复用的 GPU 工作区，用 GPU 高吞吐计算；
- Decode：让同一个权重地址直接远程读取 CPU 内存，避免每个 token 重搬权重；
- 请求之间：根据下一请求的 KV 压力调整 KV cache 的 GPU 物理驻留量，并可在
  scheduler idle 时重建专家计划；
- CUDA Graph：分别为 Prefill 和 Decode 捕获 Graph，页表切换只发生在 Graph
  replay 之外。

可以把它理解成：`Parameter VA` 是一个始终不变的门牌号，PTE 是门牌号背后的
导航表。Prefill 时导航到 GPU 仓库，Decode 时导航到 CPU 仓库；模型和 CUDA
Graph 始终拿着同一个门牌号，不需要替换权重对象。

## 2. 要解决的问题

MoE 模型的总专家权重很大，但一次计算只命中少数专家。简单地把所有权重常驻
GPU 会挤压 KV cache；简单地把权重卸载到 CPU，又会产生两类问题：

1. 每层或每个 token 搬权重，H2D 传输可能成为主瓶颈；
2. 改变 tensor storage 或指针会破坏 fused MoE kernel 和 CUDA Graph 对稳定地址
   的要求。

Prefill 和 Decode 的访存特征也不同：

| 阶段 | 典型特征 | 更合适的默认策略 |
|---|---|---|
| Prefill | 同时处理很多 token，同一权重可能被大量复用 | 搬到 GPU 后计算 |
| Decode | 每步 token 少、专家访问稀疏、步数多 | 从 CPU 远程读取实际命中的权重 |

因此本方案不是单纯的“CPU offload”，而是一个阶段感知的物理内存拓扑切换系统。

## 3. 设计目标与非目标

### 3.1 设计目标

- 原始 `Parameter.data_ptr()` 在初始化、Prefill、Decode 和恢复过程中保持不变；
- Prefill 使用显式 H2D 搬运和 GPU MoE kernel；
- Decode 不复制完整专家权重，由 GPU kernel 远程读取 CPU-backed 权重页；
- Prefill 和 Decode 都能 replay 已捕获的 CUDA Graph；
- PTE 切换可逆，下一请求不必重建 backing 或重新捕获 Graph；
- KV cache 使用固定最大 VA，并在请求间改变 GPU/CPU 物理驻留；
- 所有未支持状态均 fail-closed，不依赖隐式 UVM fault 来“碰运气”修复。

### 3.2 当前非目标

- 尚未支持同一 scheduler step 中新请求 Prefill 与旧请求 Decode 混跑；
- 尚未验证多 GPU、tensor parallel 下的跨卡 PTE/ownership 协议；
- 尚未验证多个 CUDA Graph 实例并发 replay 同一组可变 PTE；
- 三队列压力事件、动态专家计划和物理 KV pool 已通过一次单 GPU 双向往返 E2E，
  但尚未验证长期高低压力交替 workload；
- 尚未证明任意模型、任意量化格式和任意驱动版本都可直接使用。

## 4. 当前能力状态

| 能力 | 当前状态 | 证据 |
|---|---|---|
| 原始 Parameter VA 不变 | 已验证 | 双 Graph、6 GiB Decode、物理 KV qualification |
| Prefill alias Graph | 已验证 | Qwen3 B1/B4 FULL replay |
| Decode remote-read Graph | 已验证 | Qwen3 2/4/6 GiB，B1/B4 |
| `sealed -> remote_parked -> sealed` 往返 | 已验证 | PARK/RESUME qualification |
| 批量 PTE replace | 已验证 | PARK ioctl 约 42 ms，完整 cutover 约 57--58 ms，2.25 GiB 集合 |
| 请求边界动态重建专家计划 | 已验证功能正确 | 2 GiB -> 4 GiB -> 2 GiB，但需 13--19 s |
| C2/C4/C6/C8 固定 Graph class | 已通过增量整模型 qualification | DeepSeek-V2-Lite 同进程 C2 -> C4 -> C6 -> C8 -> C2；session/backing 复用、差集 refill/residency、Graph capture 8 -> 8、三类 VA 不变、输出一致 |
| 固定 VA 的物理 KV 自动扩缩 | 已验证 | 105 <-> 1,023 blocks，Graph 无重捕获 |
| GPU/free、GPU/backup、CPU/offline 三队列 | 单次双向 E2E 已通过 | Qwen3 在同一实验组进程中完成 KV 24 -> 40 -> 20；每层卸载并恢复 1 个专家；两轮 Prefill/Decode Graph 输出与 baseline 一致 |
| 缩容后的 HBM 可复用 | 已验证 | 512 MiB native、non-managed CUDA allocation |
| mixed Prefill/Decode continuous batching | 未支持 | 当前明确 fail-closed |
| 联合调整专家计划和物理 KV pool | 单次双向 E2E 已通过 | backup 压力触发逐层卸载与 KV grow；连续三轮低压产生 shrink pending，在安全边界缩容并逐层恢复专家；长期交替 workload 尚未 qualification |
| 动态路径与直接落档的稳态性能 | 已通过单次路径等价 qualification | 相同 KV/逐层专家状态下，Prefill 差异 0.37%/0.48%，Decode 差异 3.10%/5.17%；输出一致且 Graph 无 recapture |
| Qwen 2--10 GiB 真实 HBM 多级路径 | 已通过单次大容量 qualification | `2 -> 4 -> 8 -> 10 -> 8 -> 4 -> 2 GiB` 与四组 direct 基线；每层 70/80/99/108 个 remote expert，六组稳态 Prefill/Decode 差异均小于 0.5%，Graph capture `2 -> 2` |

## 5. 核心术语

### 5.1 Parameter VA

模型权重 tensor 的 GPU 虚拟地址，也就是 kernel 参数中看到的地址。这里的
“地址不变”指 `Parameter.data_ptr()` 不变，并不表示其背后的物理内存永远不变。

### 5.2 PTE

页表项，负责把虚拟地址翻译到实际物理页。本方案通过定制 UVM 驱动修改 PTE，
让同一个 Parameter VA 在不同时刻指向 GPU backing 或 CPU-backed 权重页。

### 5.3 Upload VA

专门用于 H2D 写入的可写虚拟地址。Prefill 的搬运节点向 upload VA 写数据，模型
kernel 则从原始 Parameter VA 读取；二者在 alias 状态下指向同一 GPU backing。
因此它们的虚拟地址不同，但在 `sealed` 状态下映射到同一组 GPU 物理页；进入
remote 状态后 Parameter VA 改为指向 CPU-backed 页，upload VA 仍属于稳定 envelope。

### 5.4 Consumer alias

映射在原始 Parameter VA 上的只读 GPU alias。它是 MoE kernel 的读取入口。

### 5.5 Backing

当前 class 内的 PARK/RESUME 在 Decode 时保留 GPU backing，以换取快速恢复。跨容量
class 时会释放旧 class 的活动 backing 并按新宽度重建；长期不释放的是 Graph 可见的
upload/source/Parameter VA envelope，而不是始终占满 C8 大小的物理 HBM。

### 5.6 Remote-read

原始 Parameter VA 映射到 CPU-resident 权重页，GPU kernel 经 PCIe 直接读取实际
访问的数据。它不等同于在每个 forward 前把完整参数复制到 GPU。

## 6. 总体架构

```text
                 控制面
  冷热校准/冻结计划 ----> 请求边界策略 ----> Graph/拓扑状态机
          |                    |                    |
          |                    +----> KV 物理档位  |
          |                                         |
          v                                         v
                 数据面

  pinned CPU source --H2D--> upload VA ----+
                                            | 同一 GPU backing
                           Parameter VA <---+  Prefill: sealed alias
                                |
                                +------------> Decode: CPU remote-read PTE

  固定最大 KV VA：active prefix 在 GPU，inactive tail 在 CPU
```

系统分为四层：

1. 计划层：决定哪些专家是 hot、哪些专家由本机制管理；
2. vLLM 接入层：保持原始 forward，管理阶段、Graph key 和 scheduler barrier；
3. 用户态 UVM 层：构造 session、调用 MAP/SEAL/PARK/RESUME/DESTROY；
4. 驱动层：建立 alias、替换 PTE、同步 tracker/TLB 并维护 fail-closed 状态。

## 7. 内存关系

### 7.1 Prefill 状态

```text
CPU 权重源
    |
    | cudaMemcpyAsync，可被 CUDA Graph 捕获
    v
Upload VA（RW） ----+
                    | 共享 GPU backing
Parameter VA（RO）--+
    |
    v
原始 fused MoE forward
```

H2D 写入和模型读取使用不同虚拟地址，避免计算 kernel 获得写权限，同时允许搬运
节点稳定地写入同一块 GPU backing。

### 7.2 Decode 状态

```text
Upload VA ---------> 保留的 GPU backing（不再连接 consumer）

Parameter VA ------> CPU-backed 权重页
    |
    v
fused MoE kernel 经 PCIe remote-read
```

Parameter 对象和 VA 不变，变化的只是它的 PTE。Decode kernel 只读取路由实际命中
的权重区域，不执行每步全量 H2D。

### 7.3 2.25 GiB L0--L3 实例

Qwen3 阶段切换基础计划管理：

- L0--L3；
- 每层 `w13_weight` 和 `w2_weight`；
- 8 个 Parameter；
- 512 个专家；
- 2,415,919,104 B，即 2.25 GiB 主权重；
- 1,207,959,552 B，即 1.125 GiB retained backing。

在该固定几何中，1.125 GiB backing 被切成 2,304 个区域；每个区域对应一个可写
upload alias 和两个只读 consumer alias，因此共有 `2,304 x 3 = 6,912` 条 alias。
这是该实验几何的计数，不是所有专家计划都固定使用三倍 alias 或一半 backing。

## 8. 驱动 ABI 与状态机

当前定制 UVM ABI 使用以下 session-scoped ioctl：

| 命令 | 名称 | 作用 |
|---:|---|---|
| 97 | `UVM_MOE_STATIC_ALIAS_MAP` | 分批建立 upload/consumer alias |
| 98 | `UVM_MOE_STATIC_ALIAS_SEAL` | 校验计数并冻结拓扑 |
| 99 | `UVM_MOE_STATIC_ALIAS_DESTROY` | 最终销毁 alias，可释放 backing |
| 101 | `UVM_MOE_STATIC_ALIAS_PARK` | 将 consumer PTE 切到 CPU remote-read |
| 102 | `UVM_MOE_STATIC_ALIAS_RESUME` | 从不可变记录恢复 GPU consumer alias |
| 103 | `UVM_MOE_STATIC_ALIAS_REBIND` | parked 状态下原子替换只读 consumer 集合，保留 upload/backing |

用户态生命周期为：

```text
building
   |
   | MAP + SEAL
   v
sealed（Prefill alias）
   |
   | PARK
   v
remote_parked（Decode remote-read）
   |
   | REBIND（同容量，仅替换 consumer 集合）
   v
remote_parked
   |
   | RESUME
   +---------------------------> sealed

任何不可恢复错误 -> poisoned -> 只允许 DESTROY
```

MAP 失败会回滚；部分提交、计数不一致或布尔状态异常会使 session 进入 destroy-only。
驱动禁止 sealed alias range 被普通 UVM migration、discard 或 fault servicing 静默
改变。

## 9. 初始化与双 Graph 捕获

### 9.1 专家计划

计划来源可以是：

- 手工指定层/Parameter，例如 L0--L3；
- 独立 calibration 请求产生的每层路由频率；
- 冻结的 cold-expert JSON 计划。

正式实验要求 calibration 与 evaluation 输入隔离，并让 2/4/6/8 GiB 计划形成
嵌套集合。计划记录精确专家坐标和实际字节数，不能只保存“约 4 GiB”标签。

### 9.2 物理驻留校准

初始化在 KV 分配和 Graph capture 前执行冷热校准：

1. 为 cold ranges 安装 CPU remote-read；
2. 对 hot ranges 设置 GPU preferred location 并批量 prefetch；
3. 同步并检查 cold=CPU-only、hot=GPU-only、unknown=0；
4. 检查 Parameter VA 不变和最小 Graph headroom；
5. 任一条件不满足则在 warmup 前拒绝运行。

这是 6 GiB 配置修复的关键。只检查“某物理页仍存在”不够，还必须验证页表可读、
hot 权重实际在 GPU，并通过真实 warmup/Graph replay。

### 9.3 捕获顺序

双 Graph 初始化使用如下顺序：

```text
建立 sealed alias
  -> PARK 到 remote_parked
  -> 审计 remote PTE 和 CPU residency
  -> 捕获 uniform Decode Graph，例如 B1/B4
  -> RESUME 到 sealed
  -> 捕获 non-uniform Prefill Graph，例如 314/1217 tokens
  -> 以 sealed 状态开始服务
```

`FULL_SEPARATE` 为两类 Graph 使用不同的 `BatchDescriptor` key，避免 Decode 错派到
包含 H2D 的 Prefill Graph，或 Prefill 错派到 remote topology 的 Decode Graph。

## 10. 请求执行流程

### 10.1 Prefill

1. scheduler 确认当前是独立 Prefill epoch；
2. 如果上一请求结束在 `remote_parked`，先在 Graph 外执行 RESUME；
3. replay Prefill Graph；
4. Graph 内的 H2D 节点把冻结计划中的选定权重写到 upload VA；
5. fused MoE kernel 从原始 Parameter VA 读取 GPU backing。

当前静态 Graph 会在每次 Prefill 搬运计划中的完整选定集合，即使某些 cold expert
没有被本次路由命中。冷度能降低 Decode remote-read 流量，但不会自动跳过
Prefill Graph 中未命中专家的 H2D。

### 10.2 Prefill -> Decode cutover

```text
Prefill Graph 完成
  -> GPU barrier
  -> PARK consumer PTE
  -> remote-map/驻留审计
  -> Decode Graph replay
```

当前 grouped PARK 不再逐条撤销 4,608 个 consumer。它按 managed VA block 合并
page mask，并以少量批次完成 `GPU alias PTE -> CPU remote-read PTE` 替换。2.25 GiB
实例从 4,608 条 consumer 归并为 1,152 个 VA-block group、4 个批次。

### 10.3 Decode

- 原始 forward 和 fused MoE kernel 不变；
- Graph 内没有权重 H2D；
- GPU 只远程读取路由实际访问的 CPU 权重；
- 第一次 Decode 单独报告 cutover，稳定 Decode 不混入这项一次性成本。

### 10.4 下一请求

请求结束且 scheduler idle 后执行 RESUME，重新连接 Parameter VA 与 retained GPU
backing，再 replay 同一 Prefill Graph。Graph capture 计数不应增加。

### 10.5 Shutdown

顺序必须是：

1. 销毁并同步所有引用 alias/remote VA 的 Graph；
2. cmd99 删除全部 alias 并释放 backing；
3. 关闭 session；
4. 释放模型、KV 和 runtime；
5. 恢复 stock UVM。

不能在 Graph 仍可能 replay 时销毁 PTE 或 backing。

## 11. 为什么 CUDA Graph 可以工作

CUDA Graph 节点保存的是 kernel、参数和虚拟地址，不要求同一虚拟地址永远连接
同一物理页。本方案满足三个条件：

1. Parameter VA 始终不变；
2. 每次 replay 前已完成合法拓扑切换和 TLB/tracker 同步；
3. Prefill 和 Decode 使用不同 Graph 分区。

PTE ioctl、PARK/RESUME、remote map 和 residency audit 一律位于 Graph 外。
用户态 wrapper 在 capture 活跃时会拒绝这些 ioctl。

## 12. Retained backing 的取舍

PARK/RESUME 不销毁 upload alias 和 GPU backing，因此 Decode 时仍占用一部分 HBM。
对于 2.25 GiB L0--L3 实例：

- 选定主权重：2.25 GiB；
- retained backing：1.125 GiB；
- Decode 净释放：约 1.125 GiB。

收益是：

- RESUME 不重新分配 backing；
- Prefill Graph 可直接复用；
- session 和不可变 alias 记录保持有效。

代价是不能获得“销毁所有 backing”时的最大显存释放。不同冷专家计划的 backing
取决于 slot/bank 几何，必须从实际 manifest 计算，不能总按 50% 估算。

## 13. 动态专家计划

当前已经验证请求间重建：

```text
请求结束、scheduler idle
  -> 读取下一请求逻辑 KV 压力
  -> 选择新的 frozen expert plan
  -> 恢复新 hot complement
  -> 重建 Prefill binding/alias/backing
  -> 重建 Decode remote ranges
  -> 重新捕获 Graph
  -> 执行下一请求
```

2 GiB -> 4 GiB -> 2 GiB qualification 保持了 96 个 Parameter VA 和输出一致，但
当前完整重建约需 13--19 s，其中计划/拓扑重建约 8.6--14.5 s，Graph capture 约
4.2--4.4 s。因此它目前适合初始化、长空闲窗口或低频档位变化，不适合每请求
频繁换专家。

这仍是任意、未预捕获专家计划的全量路径。固定 C2/C4/C6/C8 class 已在第 15.2 节
实现差集迁移、session/backing 复用和已有 GraphExec 切换；任意动态计划若不能落入
预捕获 class，仍需使用本节的全量重建路径。

## 14. 固定 VA 的物理 KV pool

### 14.1 基本思路

初始化一次预留最大 KV VA，但不要求全部 KV 页长期占用 HBM：

```text
固定 1.5 GiB KV VA
  ├─ active prefix：GPU resident
  └─ inactive tail：CPU resident
```

这里“释放物理内存”指释放 GPU/HBM 物理页。inactive tail 仍保留 CPU backing，
不是完全未提交的 VA hole，这样扩容时才能安全搬回 GPU，且 `unknown_bytes=0`。

### 14.2 自动压力决策

`EngineCore.apply_uvm_kv_next_request_pressure()` 根据下一批请求的 prompt tokens、
最大输出 tokens 和 KV block size 计算预测 blocks，并在 scheduler idle 时选择
low/high 档位。实验序列不直接传 `target_blocks`，避免把人工答案伪装成自动策略。

当前 Qwen3 qualification：

| 请求 | 预测 KV blocks | 自动目标 | 物理放置 |
|---|---:|---:|---|
| B8/high | 170 | 1,023 | 1.5 GiB 全部在 GPU |
| B1/low | 22 | 105 | 159 MiB GPU + 1,377 MiB CPU |

### 14.3 扩缩流程

1. 确认无未完成请求和 queued batch；
2. 对将变化的 block range 做 quarantine；
3. Grow：设置 GPU preferred location 并批量 CPU -> GPU prefetch；
4. Shrink：批量 GPU -> CPU migration；
5. 同步并核对完整物理驻留；
6. 只有迁移成功才更新 block manager 可用容量；
7. 失败 block 保持 quarantine/UNKNOWN，不能提前重新分配。

KV tensor 和 VA 从不替换，因此现有 Graph 无需重新捕获。v16 中 Graph capture
计数保持 4 -> 4，缩容后还成功分配并写满 512 MiB native、non-managed CUDA
内存，证明 HBM 能被其他分配器消费者复用。

### 14.4 当前性能

- low -> high：约 303 ms；
- high -> low：约 1.94 s；
- 到达相同物理档位后，Prefill 计算差异不超过 0.91%，稳定 Decode 差异不超过
  1.94%；
- 端到端 Prefill 仍受独立 alias RESUME 抖动影响，v16 中约有 16.8%--21.2%
  差异，不能归因于 KV resize。

### 14.5 GPU/backup/CPU 三队列原型

动态物理 pool 可选启用一个逻辑 backup reserve。三个队列的物理含义为：

```text
GPU free       GPU resident，可直接分配
GPU backup     GPU resident，free 不足时才分配
CPU offline    CPU resident，扩容完成后进入 GPU 队列
```

设置 `VLLM_UVM_KV_BACKUP_RESERVE_BLOCKS` 后，block pool 从当前 GPU free
集合末尾保留相应数量的 block。调度器统计总可用量时包含 backup，但实际分配严格
先取 free；任何一次取用 backup 都会累计压力事件。EngineCore 在该调度步触发现有的
增量 expert offload，再把物理 KV target 至少增加已消费的 backup block 数；可用
`VLLM_UVM_KV_BACKUP_GROW_BLOCKS` 指定更大的单次补充量。请求释放 GPU block 后，
优先把 backup 补回配置水位。

如果 pipeline batch queue 中还有 in-flight batch，扩容会延迟到队列排空后执行。
2026-08-24 的最新真实模型 artifact 已在同一实验组进程中 qualification 一次升压和
一次降压。它证明受控单轮双向路径可用，但仍不能外推为长期交替 workload。使用
expert 回收时仍需启用并配置 `VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE`
对应的增量卸载与反向恢复能力。

当前原型配置入口：

```bash
VLLM_UVM_KV_BACKUP_RESERVE_BLOCKS=128
VLLM_UVM_KV_BACKUP_GROW_BLOCKS=128
VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=1
```

`BACKUP_RESERVE_BLOCKS=0` 保持旧的 GPU/offline 双队列行为。backup 是逻辑保留池，
不是额外物理容量；其中的 block 已占用 HBM。

2026-08-24 的真实 Qwen3-30B-A3B-FP8 qualification 已验证一次完整双向周期。baseline
和三队列组使用同一 6 GiB 静态专家 Graph；增量 expert wave 从 7 GiB−6 GiB
差集中每层选择一个原 hot 专家，避免把已被静态 alias 接管的 Parameter VA 再交给
普通 managed-region release。第一轮是 314-token Prefill 加 6-token Decode，第二轮
是 16-token Prefill 加 5-token Decode。实测结果如下：

| 项目 | 结果 |
|---|---:|
| 初始/扩容后/缩容后物理 KV target | 24 / 40 / 20 blocks |
| GPU backup 配置/实际消费/最终恢复 | 8 / 4 / 8 blocks |
| CPU offline -> GPU | 16 blocks |
| GPU -> CPU offline | 20 blocks |
| 增量卸载/恢复专家 | 48 / 48 个，每层各 1 个 |
| 专家差集字节数 | 226,492,416 bytes（216 MiB） |
| cutover 识别的 runtime-remote | 96 个合并 range，226,492,416 bytes |
| Graph 模式与 capture sizes | `FULL_SEPARATE`，`1,16,314` |
| Graph capture 计数 | 3 -> 3，无 recapture |
| 第一轮输出 | 与 baseline token 完全一致，SHA-256 `c254ed7336d394573c3d99a76ae37307ace43fb3954d5c7836ebd274ce0dd44e` |
| 第二轮输出 | 与 baseline token 完全一致，SHA-256 `6cac54099bf9ad91133c048c33ae7dd4ce44167d793ceca3a2729abb1f4d5137` |
| 内核与清理 | 0 个 Xid/FAULT_PDE；GPU client、UVM session 和 backing 全部清理 |

首次 Decode cutover 的 hot 驻留审计也覆盖了运行时差集：22,196,256,768 bytes
仍独占 GPU，216 MiB 动态 remote experts 位于 CPU，两者之和与完整 hot-range 查询
字节数一致。该检查防止为了通过实验而简单放宽 hot audit。

该轮只做功能 qualification，暂不优化同步延迟。三队列组第一轮 Prefill step 为
2,059.33 ms，baseline 为 152.02 ms；差值包含同步 expert wave、物理 KV grow 和
审计，不应解释为最终 serving 性能。稳定 Decode 中位数分别为 10.41 ms 和
9.54 ms。第二轮受控往返耗时分别为 2,020.83 ms 和 1,170.10 ms，其中还包含
RESUME/PARK、物理 shrink、专家恢复和审计。

### 14.6 Readahead 式自动倍增/减半

三队列运行时已经加入统一的双向 autoscale 控制器。它复用既有物理迁移、quarantine
和驻留审计，不再让“下一请求 low/high 策略”和“backup 压力策略”分别写 KV target。
启用新控制器后，下一请求接口只提供预测值，backup 和 preemption 只提供紧急压力
事件，最终 target 由同一个状态机决定。该路径默认关闭，必须显式设置
`VLLM_UVM_KV_AUTO_SCALE_ENABLE=1`。

控制器使用以下量：

```text
T = 当前 GPU-resident KV target blocks
L = 配置与最小服务需求共同决定的下界
U = HBM 安全预算允许的上界
B = GPU backup reserve
A = 当前调度轮结束前观察到的 active KV blocks
P = 下一请求预测 blocks
E = safety blocks + 正预测误差 EWMA

R = max(A, P) + B + E
```

增长是非对称的快速路径。backup 被使用、发生 request preemption，或者 `R` 到达
高水位时，控制器取消所有 pending shrink，并计算：

```text
grow_target = min(U, align_up(max(T * grow_factor, R)))
```

默认 `grow_factor=2`。backup/preemption 和 `R > T` 可以绕过最短驻留时间；普通
高水位预测受最短驻留时间保护，避免同一预测在迁移后的下一轮再次把 target 翻倍。
pipeline batch queue 未排空时只记录最大 pending grow，排空后再提交。

缩小使用保守的连续确认。一个调度轮只有同时满足以下条件才累计 `low_streak`：

- 该轮确实调度了 token，不把空轮当作低压力；
- `R` 能完整放进 `ceil(T / shrink_divisor)`；
- 没有 backup activation、preemption、pending grow；
- 没有 UNKNOWN block 或外部 reconciliation；
- 当前 target 已满足最短驻留轮数。

默认连续三轮后生成：

```text
shrink_target = max(L, align_up(ceil(T / 2)))
```

任一有效轮不再满足条件就清零 streak 并取消 pending shrink。一次 shrink 成功后重新
从零观察；持续低压时因此形成 `1024 -> 512 -> 256 -> 128`，但不会越过下界。

pending shrink 在 scheduler 边界提交。普通单流允许仍有未完成请求，但必须没有
in-flight pipeline batch，而且 `(shrink_target, T]` 的全部 tail block 都满足
`ref_cnt == 0`。活动请求只要完全位于保留前缀就可以继续存在；任一 tail block 仍被
引用时，既有 resize planner 会拒绝迁移并保留 pending，不移动有效 KV。提交流程仍是：

```text
quarantine tail
  -> GPU 到 CPU 批量迁移
  -> 同步并审计实际驻留
  -> 原子更新 block manager target
  -> tail 进入 CPU/offline queue
```

缩容成功后，控制器执行一轮反向 expert wave。每个 MoE 层只选择一个此前由 KV
压力增量卸载的专家；静态基础计划中的 cold expert 不在可恢复集合中。开启 GPU 热度
计数时优先恢复该集合中最热的专家，否则使用确定性顺序。这样 grow 和 shrink 都按
层对称地增加或减少一个专家，不会只处理前几层。若请求尚未结束，恢复操作延迟到
idle。候选选择采用全层事务：任一层缺少同类候选时整轮不执行，不能只修改其余层；
若恢复过程中只有部分层成功，则把本轮已经恢复的专家重新卸载，恢复原有逐层对称
状态并保留重试标记。恢复失败不回滚已经正确提交的 KV shrink，专家继续
remote-read。

最新 E2E 中，低压轮以 0 为起点记录：step 0 和 step 1 分别得到
`observe_low_pressure`，step 2 达到阈值并得到 `shrink_pending`。这时活动请求仍在
Decode，因此 target 保持 40，48 个动态 remote expert 也保持不变；最终 Decode
完成的 step 4 到达安全边界后，物理 target 才提交为 20，随后恢复每层 1 个专家。
第二轮 Prefill 确认 `remote_parked -> sealed` RESUME，首个 Decode 再确认
`sealed -> remote_parked` PARK，整个往返的 Graph capture 计数保持 3。该结果同时
区分了“第三轮作出缩容决定”和“安全边界真正执行迁移”，不能要求二者发生在同一个
调度步。

固定 Graph class 的机器/模型预算关系通过显式映射配置，避免运行时猜测 C2/C4/C6/C8
名称的物理含义。例如：

```bash
VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_MAP='{
  "128": "C8",
  "256": "C6",
  "512": "C4",
  "1024": "C2"
}'
```

JSON key 是 KV target 上界，value 是已经完整捕获并持有证书的 class。每次 target
提交后记录目标 class，只在 scheduler idle 且 batch queue 排空后调用已有的原子
Graph-class activation；失败时旧 class 继续服务并保留 pending。部署应根据真实 HBM
预算生成该映射，不应机械复用示例数值。

主要配置为：

```bash
VLLM_UVM_KV_AUTO_SCALE_ENABLE=1
VLLM_UVM_KV_AUTO_SCALE_LOW_PRESSURE_STREAK=3
VLLM_UVM_KV_AUTO_SCALE_GROW_FACTOR=2
VLLM_UVM_KV_AUTO_SCALE_SHRINK_DIVISOR=2
VLLM_UVM_KV_AUTO_SCALE_HIGH_WATERMARK=0.80
VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS=128
VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS=1024
VLLM_UVM_KV_AUTO_SCALE_SAFETY_BLOCKS=16
VLLM_UVM_KV_AUTO_SCALE_MIN_RESIDENCY_STEPS=3
VLLM_UVM_KV_AUTO_SCALE_RESIZE_GRANULARITY_BLOCKS=1
VLLM_UVM_KV_AUTO_SCALE_PREDICTION_ERROR_ALPHA=0.25
VLLM_UVM_KV_AUTO_SCALE_EXPERT_RESTORE_ENABLE=1
VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=1
```

上下界会按 `RESIZE_GRANULARITY_BLOCKS` 对齐：下界向上取整以保住 backup 与 safety，
上界向下取整以避免越过 HBM 预算；如果对齐后下界大于上界，控制器 fail-closed，
不会以突破 HBM 上界为代价继续扩缩。部署时该粒度应同时满足 KV group 布局和物理
迁移批次要求。

控制器提供 `get_uvm_kv_autoscale_snapshot()`，报告当前 target、上下界、低压连续计数、
预测误差、pending grow/shrink、pending expert restore、pending plan class、最近决策和
三队列完整快照。增长前已经完成的 expert wave 如果在物理迁移提交前失败，会按相同
逐层集合反向恢复；物理迁移一旦可能发生部分进度，则不冒险恢复专家，相关 KV block
继续保持 UNKNOWN/quarantine，等待既有 reconciliation。

### 14.7 逐步切换与直接落档的稳态性能等价性

2026-08-24 在 RTX 5090 上增加了独立路径等价 qualification，用于回答“经压力控制器
逐步到达某一状态后，是否会比进程启动时直接进入该状态更慢”。模型为
Qwen3-30B-A3B-FP8，Attention backend 为 `TRITON_ATTN`，CUDA Graph 模式为
`FULL_SEPARATE`。三个隔离进程分别执行：

1. dynamic：从 24 KV blocks 开始，B8、每请求 33-token Prefill 实际消费全部 8 个
   backup blocks，触发 `24 -> 40` 和每层卸载 1 个专家；随后经低压路径缩到 20 并
   恢复同一组 48 个专家；
2. direct-high：启动时直接使用 40 KV blocks，并直接卸载与 dynamic-high 完全相同的
   48 个 expert IDs；
3. direct-low：启动时直接使用 20 KV blocks，48 层均没有增量 remote expert。

进入目标状态后冻结 autoscale。每个状态运行 3 个性能请求，每个请求先做 5 个 Decode
warmup，再记录 16 个 Decode step，因此每组 Decode 有 48 个样本。主 Prefill 指标扣除
alias RESUME，Decode 稳态样本不包含 resize、expert wave 或 PARK；这些切换开销仍单独
保留在原始 artifact 中，不能据此宣称切换本身无成本。

| 最终状态与阶段 | 逐步路径 | 直接路径 | 逐步相对直接 |
|---|---:|---:|---:|
| 40 blocks + 每层 1 个 remote expert，Prefill（扣除 RESUME） | 148.673 ms | 148.119 ms | +0.37% |
| 同上，Decode median | 10.946 ms | 10.617 ms | +3.10% |
| 20 blocks + 增量专家全部恢复，Prefill（扣除 RESUME） | 145.992 ms | 145.300 ms | +0.48% |
| 同上，Decode median | 10.553 ms | 10.034 ms | +5.17% |

四项均通过预设的 Prefill 15%、Decode 10% qualification 阈值。验证器还确认两条路径
到达的队列计数、KV 物理驻留、48 层 expert ID 集合、生命周期状态完全一致；输出 token
hash 一致，进程内 Parameter/KV VA 不变，Graph capture 计数保持 `5 -> 5`，运行窗口无
Xid、`FAULT_PDE`、Oops 或 BUG，退出后 stock UVM 和 GPU clean 状态恢复。因此当前证据
支持：**切换过程有一次性成本，但到达相同 KV/专家状态后，没有检测到由历史路径造成的
持续性能退化。**该结论仍限于一次受控单卡运行；论文级数据需要多轮交错顺序和置信区间。

### 14.8 Qwen 2--10 GiB 真实 HBM 多级 qualification

为避免把 24/40/20 blocks 的控制路径结果外推到 GiB 级容量，2026-08-24 又在 RTX
5090 上完成了独立的大容量实验。模型固定为 Qwen3-30B-A3B-FP8，Attention backend
为 `TRITON_ATTN`，Prefill 和 Decode 均使用 `FULL_SEPARATE` CUDA Graph；实验没有改用
DeepSeek 或 MLA/FlashInfer attention backend。

KV tensor 在每个进程中一次性保留 10,737,942,528 B VA，运行时只改变 GPU-resident
前缀。四个物理档位为：

| 档位 | target blocks | GPU-resident KV（含 null block） | 每层 remote experts | 相对 base70 的增量 wave/字节 |
|---|---:|---:|---:|---:|
| 2 GiB | 1,364 | 2,146,959,360 B | 70 | 0 / 0 B |
| 4 GiB | 2,730 | 4,295,491,584 B | 80 | 10 / 2,264,924,160 B |
| 8 GiB | 5,460 | 8,589,410,304 B | 99 | 29 / 6,568,280,064 B |
| 10 GiB | 6,826 | 10,737,942,528 B | 108 | 38 / 8,606,711,808 B |

32 GiB GPU 无法先保留全部专家再建立 10 GiB KV 和 Prefill alias，因此启动时先按每层
相同数量卸载 70 个冷专家，共 15,854,469,120 B；之后每个增量 wave 再从 48 层各卸载
或恢复 1 个专家。该设计始终保持逐层对称，不使用“只卸载前几层”的特殊实验。
增长时先完成所需 expert release，再执行 CPU -> GPU KV migration；下降时先执行
GPU -> CPU KV migration，再恢复相同的 expert wave。

实验组由一个 stepped 进程和四个隔离 direct 进程组成：

- stepped：`2 -> 4 -> 8 -> 10 -> 8 -> 4 -> 2 GiB`，除初始 2 GiB 外每个到达状态执行
  3 个请求，共 18 条记录；
- direct：分别从 2 GiB 初始状态直接到达 2/4/8/10 GiB，每个进程执行 3 个请求；
- 每个请求先 replay Prefill Graph 和首个 Decode Graph，再执行 5 个 Decode warmup，
  最后采集 16 个 Decode step，因此每个状态有 3 个 Prefill 和 48 个 Decode 稳态样本。

direct 基线的绝对性能为：

| 最终状态 | Prefill median，扣除 RESUME | Decode median | Decode p95 | Decode token/s |
|---|---:|---:|---:|---:|
| 2 GiB / 每层 70 remote | 333.852 ms | 15.734 ms | 23.087 ms | 63.558 |
| 4 GiB / 每层 80 remote | 358.778 ms | 19.212 ms | 25.106 ms | 52.052 |
| 8 GiB / 每层 99 remote | 444.170 ms | 27.140 ms | 32.871 ms | 36.846 |
| 10 GiB / 每层 108 remote | 491.192 ms | 30.669 ms | 36.796 ms | 32.607 |

这里绝对延迟随档位升高，是因为为 KV 腾出更多 HBM 后，更多专家改为 CPU remote-read；
它不是逐步切换历史造成的额外损失。用同一最终 KV/专家集合与 direct 进程比较，结果为：

| stepped 到达状态 | Prefill 相对 direct | Decode median 相对 direct | Decode p95 相对 direct |
|---|---:|---:|---:|
| ascending 4 GiB | +0.086% | -0.154% | -0.255% |
| ascending 8 GiB | -0.012% | -0.048% | -0.466% |
| ascending 10 GiB | +0.156% | -0.235% | -0.404% |
| descending 8 GiB | +0.092% | +0.284% | -0.289% |
| descending 4 GiB | +0.070% | -0.270% | +0.020% |
| descending 2 GiB | +0.069% | -0.057% | -0.385% |

六组 Prefill、Decode median 和 Decode p95 的绝对差异均小于 0.5%，远低于预设的
15%/10%/15% qualification 阈值。因此在本次真实 2--10 GiB 范围内，**逐步到达相同
物理 KV 和逐层专家状态后，没有检测到持续性的路径性能损失。**

一次性切换开销没有包含在上述稳态表中。实测 expert wave 与纯 KV migration 为：

| 转换 | expert 操作 | expert wave 总时间 | KV migration 时间 |
|---|---:|---:|---:|
| 2 -> 4 GiB | release 10 waves | 29.388 s | 0.695 s |
| 4 -> 8 GiB | release 19 waves | 49.652 s | 0.896 s |
| 8 -> 10 GiB | release 9 waves | 16.871 s | 0.616 s |
| 10 -> 8 GiB | restore 9 waves | 6.079 s | 2.558 s |
| 8 -> 4 GiB | restore 19 waves | 12.935 s | 4.798 s |
| 4 -> 2 GiB | restore 10 waves | 6.718 s | 2.578 s |

当前 wave 仍逐个串行提交，因而专家调整而非 KV migration 是增长路径的主要一次性
开销；该结果不能解释为 2--10 GiB 在线切换已经足够快。

独立 verifier 还确认：五个进程全部返回 0；所有输出 token hash 一致；KV、Parameter
和 pinned source VA 在各自进程内不变；Prefill/Decode FULL Graph capture counter 在
每个进程中均保持 `2 -> 2`；无 Xid、`FAULT_PDE`、Oops、BUG 或 UNKNOWN residency；
退出后 GPU 为 0 MiB，alias session/backing 全部释放，stock `nvidia_uvm` 的 srcversion
和模块哈希恢复。10 GiB 保守预算在扣除 resident model、alias backing 与 512 MiB
Graph/workspace 后仍有 5,145,018,368 B raw headroom，扣除 2 GiB safety 后剩余约
2.79 GiB。

本实验使用显式 target 保证 stepped/direct 的最终状态严格相同，验证的是大容量物理
迁移、逐层专家差集和 Graph 路径等价性；它不是自动压力预测器的长期 workload
qualification，也没有在一个进程中切换 C2/C4/C6/C8 Graph class。后两项仍应分别用
长时间峰谷 workload 和固定 class registry 实验验证。

## 15. 当前内存协同策略的准确边界

系统现在具备四种能力：

1. 根据逻辑 KV 压力重建专家计划；
2. 根据下一请求压力扩缩物理 KV pool；
3. 以 GPU backup reserve 吸收突发，并在 backup 被使用后触发增量 expert 回收与
   物理 KV grow；
4. 由单一控制器按高水位即时倍增，并在连续三轮低压后逐级减半，同时反向恢复每层
   的动态卸载专家。

前两种能力已经分别通过 qualification。第三和第四种能力现已在一次真实模型受控
双向周期中完成 E2E qualification：backup pressure 在同一调度周期触发 48-layer
expert wave，把物理 KV target 从 24 增长到 40，并恢复 8-block backup 水位；随后
连续三个有效低压轮生成 shrink pending，在请求完成边界把 target 从 40 缩到 20，
再恢复同一组 48-layer expert。两轮请求均覆盖 Prefill Graph、Decode Graph 和阶段
PARK/RESUME，输出与固定 40-block baseline 一致，Graph capture 未增加。

这里的“已通过”边界是单进程、单 GPU、一次高压加一次低压的确定性功能往返，
不是长期在线控制器结论。重复峰谷、预测误差、多个并发请求与跨 class 交替仍需独立
qualification。

当前运行时状态机为：

```text
GPU free 不足
  -> 分配 GPU backup
  -> 发布 backup pressure event
  -> 增量卸载 expert 到 CPU
  -> KV target 倍增并把 CPU/offline block 迁入 GPU

连续三轮 R 能放入 T/2
  -> quarantine 空闲 KV tail
  -> KV tail 迁回 CPU/offline
  -> 每层恢复一个此前动态卸载的热门 expert
  -> 可选激活映射后的固定 Graph class
```

尚未 qualification 或仍需扩展的边界包括：

- 重复高低压力和长时间 workload 下的真实模型 E2E；
- backup 大小的在线调整；
- 物理迁移已有部分进度时的自动跨组件恢复；当前保持 UNKNOWN/quarantine 并要求
  reconciliation；
- mixed Prefill/Decode、pipeline batch queue 和长期突发 workload 下的 E2E 行为；
- 多 GPU 下每张卡独立 reserve 与全局 admission 的协调。

最终联合策略应以 HBM 预算为约束：

```text
可用 HBM
  = 总 HBM
  - hot weights
  - retained alias backing
  - Graph/workspace reserve
  - safety headroom

剩余预算在 active KV prefix 与 promotion/expert cache 之间分配
```

为了避免 2 GiB/4 GiB 计划来回抖动，应使用高低水位、最短驻留时间和预测误差
保护，而不是每次请求都立即重建。

### 15.1 热度驱动的周期性专家交换

当前第一阶段实现已经复用 MoE router 的 `topk_ids`，在 GPU 上累计每层逻辑专家
的 token 命中数。每层维护两个固定 heat buffer；统计窗口结束时切换 active buffer，
读取刚完成的 buffer，并递增 `heat_epoch`。CPU EngineCore 对连续窗口应用 EWMA：

```text
H_t(e) = alpha * observed_t(e) + (1 - alpha) * H_(t-1)(e)
```

控制器比较当前 remote 专家和 EWMA 目标集合，只交换收益超过阈值的专家，并限制
每层单次交换数量。自动维护仅在 scheduler idle 且 pipeline batch queue 已排空时
执行，并受最短时间间隔约束。主要配置为：

```bash
VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=1
VLLM_UVM_DYNAMIC_EXPERT_AUTO_REBALANCE_ENABLE=1
VLLM_UVM_DYNAMIC_EXPERT_HEAT_EWMA_ALPHA=0.25
VLLM_UVM_DYNAMIC_EXPERT_REBALANCE_INTERVAL_MS=60000
VLLM_UVM_DYNAMIC_EXPERT_MAX_SWAPS_PER_LAYER=1
VLLM_UVM_DYNAMIC_EXPERT_MIN_COUNT_DELTA=1
```

该路径保持当前 remote-expert 数量不变，属于同一容量 class 内的热专家/冷专家差集
交换。交换复用显式 expert record 和原始 Parameter VA，不执行完整 expert-plan
销毁与 Graph recapture。

### 15.2 固定 Plan Class 的当前实现

最终系统预先准备 C2/C4/C6/C8 等少量固定容量档位。HBM/KV 控制器选择不超过安全
预算的最大 class，热度控制器选择 class 内的具体专家：

> 实施结论：固定 class 切换已经改为只处理 added/removed 专家，retained backing
> 与 alias session 全程复用，并且只 refill `slot -> expert` 发生变化的 source slot；
> 正常切换不再执行 DESTROY、finalize 或 CUDA Graph recapture。

```text
HBM/KV 压力 -> 选择 C2/C4/C6/C8
GPU heat + CPU EWMA -> 选择 class 内专家集合
安全 epoch -> retained/added/removed 差集迁移
稳定 slot 重绑定 -> 切换已捕获 GraphExec
```

固定 class 真正免 recapture 的必要条件是 Graph 捕获稳定 slot，而不是生命周期短于
Graph 的 arena。当前实现以最大 class（默认 C8）建立一次性的稳定 envelope：

- upload managed allocation 按最大专家宽度预留，VA 全生命周期不变；
- pinned CPU source 按相同最大宽度创建，VA 全生命周期不变；
- 原始 Parameter 对象和 VA 不变；
- 每个专家切片在 envelope 前缀中具有稳定的逻辑 slot/region identity；
- 活动物理 GPU backing 只按当前 class 宽度分配，不因稳定 VA 而永久占满 C8 HBM。

这里的稳定 slot 是逻辑 slot；多个 slot 可以位于同一连续 upload arena 中，不要求每个
专家单独执行一次 CUDA allocation。Graph 只记录稳定 source/upload/Parameter VA 和该
class 的 H2D 长度。物理 backing 在 Graph 外按差集扩缩，共同 slot 的 backing 不重建。

每个 class 拥有独立的 `BatchDescriptor -> CUDAGraphEntry` 表和双拓扑证书。启动时先
把初始 Graph 表命名为 initial class，再依次捕获其余配置 class，最后恢复 initial
class。之后激活只切换已捕获的 GraphExec 表，不调用 capture/reset；未注册、空、未
完整捕获、证书 VA 不匹配或 Graph/plan 当前 class 不一致都会 fail-closed。

#### 15.2.1 差集切换与回滚流程

class 切换只允许在 scheduler idle 边界执行：

```text
sealed 时先 PARK
  -> 保留 retained 专家的 packed slot
  -> 只重填 slot -> expert 发生变化的 pinned source
  -> 只把 added cold 迁到 CPU、removed cold 恢复到 GPU
  -> mode 8 只预分配 added backing
  -> ioctl 103 RESIZE 提交完整目标拓扑
       retained backing 原位保留
       added upload alias 映射到新增 backing
       removed backing 在提交末尾释放
  -> 恢复切换前的 sealed 或 remote_parked 生命周期状态
  -> 激活对应 Graph certificate 和 GraphExec 表
```

`STATIC_ALIAS_REBIND` 使用 `BEGIN / append / COMMIT / ABORT` 事务，最多接受 65,536
个 consumer alias；COMMIT 前候选集合不拥有 PTE，失败或 ABORT 保留旧 consumer
集合。`RESIZE` 在同一 ABI 中提交完整 upload + consumer 集合，session id 不变；共同
pool slot 原位保留，只接纳预分配的新增 backing，并仅释放目标集合中已经删除的
backing。legacy mode 8/19 只在 session 已 sealed、已 PARK、未 poisoned、没有进行中
REBIND，且目标逻辑 slot 不属于当前 committed session 时放行，其他 legacy 操作继续
fail-closed。

用户态切换也具有反向事务：若 target REBIND 已提交但后续审计失败，系统在 parked
状态反向 RESIZE 到完整旧拓扑，并只预分配需要恢复的 removed backing；提交前失败会
释放尚未接纳的 added backing。residency 回滚使用 `target -> old` 的反向专家差集，
不会再次扫描和迁移整份计划。只有驱动拓扑、residency、backing bytes、生命周期
状态以及三类 VA 全部恢复后才返回可继续服务；回滚本身失败会把进程置为
`poisoned`。

RTX 5090 Graph-class 微实验让 C2/C4 两张真实 CUDA Graph 共用一个稳定 GPU slot，
交替切换 1,000 次，2,000 次 replay/结果检查全部通过；slot VA 和 Graph 对象身份保持
不变，切换期间 reset/recapture 次数为 0。class 切换中位数为 1.141 us、p95 为
1.774 us。

仓库中的 C2/C4/C6/C8 计划已经接入上述稳定 envelope 与 Graph registry。代码级测试
覆盖 C2 -> C4 -> C6 -> C8 -> C2 的 active view/backing 宽度变化、VA 恒定、启动
Graph 表收编、四档捕获、证书恢复、激活不 reset/recapture，以及同容量/跨容量失败
回滚。

#### 15.2.2 增量整模型实验结果

2026-08-23 的 RTX 5090 32 GiB 增量整模型 qualification 已在同一
DeepSeek-V2-Lite 进程中完成 `C2 -> C4 -> C6 -> C8 -> C2`：

| class | 逻辑计划 | 物理 backing | retained/added/removed slots | source refill | 切换时间 |
|---|---:|---:|---:|---:|---:|
| C2 | 2,145 MiB | 165 MiB | 330 / 0 / 924 | 0 MiB | 2.848 s |
| C4 | 4,290 MiB | 330 MiB | 330 / 330 / 0 | 2,145 MiB | 2.665 s |
| C6 | 6,006 MiB | 462 MiB | 660 / 264 / 0 | 1,716 MiB | 2.371 s |
| C8 | 8,151 MiB | 627 MiB | 924 / 330 / 0 | 2,145 MiB | 2.852 s |
| C2 | 2,145 MiB | 165 MiB | 330 / 0 / 924 | 0 MiB | 2.202 s |

逻辑计划字节表示每次 forward 选择的专家参数范围；物理 backing 是按稳定 arena
几何去重后实际占用的 HBM，两者不是同一审计量。五轮中 52 个 Parameter VA、52 个
pinned-source VA 和 4 个 upload VA 全部不变，Graph capture counter 从模型加载后的
8 保持到最终的 8，五次 greedy 输出 hash 完全相同。五轮 driver audit 的 session
id 均为 1，`destroy` 和 `finalize` 均为 `null`，session 始终未 poisoned。与同机同
序列的旧全量路径相比，五次切换总时间从 52.546 s 降到 12.938 s，缩短 75.4%，总计
4.06x；单次切换由 6.901--17.448 s 降到 2.202--2.852 s。shutdown 后 GPU 回到
0 MiB，stock UVM 和 exporter 均恢复，运行时窗无 Xid、`FAULT_PDE`、Oops 或 BUG。

#### 15.2.3 实验边界与剩余开销

本机原生 `TRITON_MLA` 声明不支持 `FULL_SEPARATE`，因此该 qualification 设置
`VLLM_MLA_DISABLE=1` 并使用 `VLLM_ATTENTION_BACKEND=TRITON_ATTN`。这证明固定
Graph class、稳定 envelope 和跨容量切换成立，不等价于证明 RTX 5090 上原生 MLA
backend 已支持 full Graph。上述秒级切换已消除全量 backing 重建和 retained source
refill，但仍包含 GiB 级 added/removed 专家 residency 迁移与必要 source refill；本
阶段不把单次 qualification 解释为最终在线切换性能。

RTX 5090 控制面微实验使用 128 个专家、每轮 65,536 个路由 assignment，500 轮的
heat update 中位数为 166.528 us、p95 为 206.784 us；两个统计 epoch 的计数均精确，
5.3 GiB 安全预算选择 C4。该历史独立结果文件已按清理策略移除；后续论文实验需要在
当前冻结协议下重新生成，不再引用旧结果文件。

## 16. 正确性与 fail-closed 机制

每个正式 case 至少检查：

- Parameter VA fingerprint 全阶段相同；
- alias、consumer、upload、backing 的计数和字节数精确；
- remote mapped/covered/CPU resident 字节等于计划字节；
- hot ranges 为 GPU-only，cold ranges 为 CPU-only，unknown=0；
- 普通分配严格先消耗 GPU free，只有不足部分才能消耗 GPU backup；
- `free + backup + offline + active + null` 的 block 数守恒且队列互斥；
- backup block 必须保持 GPU resident，offline block 必须保持 CPU resident；
- backup 激活事件只按实际取用数量累计，消费后清零；
- 每个阶段 replay 正确的 Graph，capture 计数不意外增加；
- H2D 字节符合阶段证书；
- 输出 token hash 一致；
- 无 Xid、`FAULT_PDE`、`manual_migration_fault`、Oops 或 BUG；
- shutdown 后 Graph、session、alias、backing、GPU memory 全部释放。

模块参数 `uvm_manual_migration_fault_mode=2` 会记录并取消不应发生的 replayable
GPU fault，而不是让 UVM 自动迁移掩盖缺失映射。

## 17. 已发现并修复的典型故障

6 GiB cold plan 曾在 1.2 GiB KV + 4,096-token capture 压力下触发 Xid 31
`FAULT_PDE`。根因不是 remote map 删除相邻 PTE，而是 hot expert 被 UVM 自动逐出
GPU，Graph 仍按 hot=GPU 访问。

修复是在所有大额 KV/Graph allocation 后、warmup/capture 前重新执行冷热驻留
校准，并设置最小 headroom gate。这个案例说明：

- preferred location 不是物理 pin；
- residency 元数据不能代替真实读取验证；
- Graph capture 前必须重新审计最终内存压力下的状态。

## 18. 已知限制

1. mixed Prefill/Decode batch 当前会拒绝执行；一般 continuous batching 需要
   scheduler epoch 或 phase-batched 调度；
2. Prefill Graph 每次搬完整 frozen set，尚未按实际 route 跳过 H2D；
3. Decode remote-read 受 PCIe 带宽影响，高并发或热点集中时会饱和；
4. retained backing 降低了净 HBM 释放量；
5. 任意、未预捕获的 expert-plan 仍需走 13--19 s 全量重建/recapture；固定 class
   切换不再走该路径；
6. KV shrink 明显慢于 grow；
7. alias RESUME 存在跨进程/请求延迟抖动；
8. 当前只正式验证单 GPU；
9. 依赖定制 NVIDIA UVM ABI，驱动升级需要重新移植和 qualification；
10. 冷专家来自有限 calibration，遇到 workload shift 可能变热；
11. RTX 5090 的整模型四档 Graph qualification 使用关闭 MLA 后的 `TRITON_ATTN`；
    原生 `TRITON_MLA` full-Graph 路径仍受 vLLM backend 能力限制。

## 19. 实现位置

驱动：

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h`：ABI；
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c`：MAP/SEAL/PARK/
  RESUME/DESTROY 和 grouped PTE replace；
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c`：VA block/PTE
  操作。

vLLM：

- `workloads/vllm/vllm/vllm/device_allocator/uvm_static_alias.py`：ioctl wrapper 和
  userspace session cache，包括分批 REBIND/ABORT；
- `workloads/vllm/vllm/vllm/model_executor/models/utils.py`：计划、binding、阶段
  状态、remote map、诊断和清理；
- `workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py`：阶段检测、Graph capture、
  多 class Graph registry、cutover、冷热校准、逐层 expert wave 和 KV worker 操作；
- `workloads/vllm/vllm/vllm/v1/engine/core.py`：统一 KV autoscale 状态机、请求预测、
  backup/preemption 升压、三轮降压、pending 事务和 Graph class 激活；
- `workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py`：记录只属于运行时
  KV 压力的增量卸载集合，并按层选择和恢复热门专家；
- `workloads/vllm/vllm/vllm/v1/core/block_pool.py`：GPU free、GPU backup 和
  CPU offline 三队列及分配优先级；
- `workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py`：KV block quarantine、
  backup 压力事件、物理迁移和提交协议。

Qualification harness：

- `experiments/vllm_uvm_prefill_alias_decode_remote_graph/benchmark_phase_switch_steps.py`：
  在同一进程内执行两轮 Prefill/Decode，并逐步记录 autoscale、队列、专家、Graph、
  VA 和物理驻留；
- `experiments/vllm_uvm_three_queue_e2e/run_experiment.sh`：运行固定 40-block baseline
  与 `24 -> 40 -> 20` 三队列实验组，并负责定制/stock UVM 模块切换；
- `experiments/vllm_uvm_three_queue_e2e/verify_experiment.py`：独立验证两轮输出、三轮
  低压迟滞、安全边界 shrink、逐层专家差集、Graph replay/recapture、驻留、守恒、
  内核日志和清理状态。
- `experiments/vllm_uvm_kv_expert_path_parity/`：用隔离的 dynamic、direct-high 和
  direct-low 进程验证到达相同联合 KV/专家状态后的稳态性能等价性。
- `experiments/vllm_uvm_large_kv_multilevel/`：生成 Qwen 逐层对称 base70/增量 wave
  计划，执行真实 `2/4/8/10 GiB` KV stepped/direct qualification，并独立审计大容量
  物理驻留、专家差集、双 Graph、VA、性能和清理状态。

## 20. 复现入口与结果

- 面向 EuroSys 的配套实验方案：
  [`eurosys_moe_offload_evaluation_plan.md`](eurosys_moe_offload_evaluation_plan.md)
- 静态 alias 基础：
  [`cudagraph_static_alias_transport.md`](cudagraph_static_alias_transport.md)
- 双 Graph、PARK/RESUME 实验入口：
  [`../experiments/vllm_uvm_dual_topology_graph/README.md`](../experiments/vllm_uvm_dual_topology_graph/README.md)
- 动态专家计划实验入口：
  [`../experiments/vllm_uvm_dynamic_kv_plan_rebuild/README.md`](../experiments/vllm_uvm_dynamic_kv_plan_rebuild/README.md)
- 物理 KV 自动扩缩实验入口：
  [`../experiments/vllm_uvm_physical_kv_pool_resize/README.md`](../experiments/vllm_uvm_physical_kv_pool_resize/README.md)
- GPU/free、GPU/backup、CPU/offline 三队列 E2E：
  [`../experiments/vllm_uvm_three_queue_e2e/README.md`](../experiments/vllm_uvm_three_queue_e2e/README.md)
- C2/C4/C6/C8 同进程 Graph class 序列：
  [`../experiments/hybrid_offload_2_4_6_8_compare/README.md`](../experiments/hybrid_offload_2_4_6_8_compare/README.md)
- 当前保留的正式 E0/E1 结果：
  [`../experiments/eurosys_moe_offload_evaluation/RESULTS_E0_E1_20260824.md`](../experiments/eurosys_moe_offload_evaluation/RESULTS_E0_E1_20260824.md)
- 当前 E0 所需的历史机制证书快照：
  [`../experiments/eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/provenance/retained_evidence/manifest.json`](../experiments/eurosys_moe_offload_evaluation/artifacts/eurosys-pa-v2-20260824T142031Z-e0-e1-formal/provenance/retained_evidence/manifest.json)

旧性能报告和历史运行 artifact 已清理，不再作为论文数据源。后续正式性能数据必须按
EuroSys 实验方案重新生成，使用新的 run ID、统一 E2E harness 和独立 verifier。

## 21. 推荐的后续工程顺序

1. 给 scheduler 增加 phase epoch，支持 continuous batching 下的安全分阶段执行；
2. 缓存 PARK/RESUME 的 VA-block grouping，降低 RESUME 抖动；
3. 让固定 class 的 Prefill Graph 只搬 route superset，减少未命中专家 H2D；
4. 把多轮逐层 expert release/restore 合并为单次全层差集事务，消除每个 wave 重复的
   RPC、同步和完整 residency audit；
5. 对三队列双向闭环增加真实模型故障注入、重复突发和长期 workload qualification；
6. 在现有固定 backup 水位之上增加基于消费速率和 grow 延迟的在线定容；
7. 用实测 HBM 预算自动生成 KV target -> Graph class 映射，替代部署时手工配置；
8. 将 class 内热度计划接入已完成的 ioctl 103 RESIZE 差集迁移控制器；
9. 扩展到多 GPU，并定义每张卡的 Parameter VA ownership 与 barrier；
10. 完成统一 HTTP serving harness 后再做跨系统论文性能比较。
