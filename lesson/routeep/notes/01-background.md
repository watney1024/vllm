# L1 背景调研：MoE 专家并行与 DeepEP 基线

> 对应课程 Module 0（第 1 课）。源码锚点基于 `main@8c17e46`。

## 1. 本节要解决什么

1. MoE 层由哪些部分组成，路由/专家/共享专家/汇合各自做什么；
2. TP、DP、EP 三种并行范式的切分对象与通信模式，什么时候选 EP；
3. DeepEP 的架构分层、HT/LL 两种模式的定位、优势与局限；
4. 本项目（RouteEP）的目标指标，以及在 vLLM 源码里从哪条链路切入。

**一句话结论**：EP 把"专家"分到不同 rank，token 先 dispatch 到持有目标专家的 rank，算完再 combine 回来；通信从 TP 的逐层 all-reduce 变成 token 维度的 all-to-all，难点在变长调度、通信-计算重叠和负载均衡。

## 2. MoE 层结构

一个 MoE 层的前向：

```text
hidden_states (T, H)
   │
   ├──► router: logits = x @ W_gate          # (T, E)
   │       softmax/sigmoid → TopK → (weights, ids)
   │
   ├──► routed experts: 每个被选中的 expert_i 是一个 SwiGLU MLP
   │       out_i = expert_i(x)
   │
   ├──► shared expert（可选）: 所有 token 都走的公共 dense MLP
   │
   └──► combine: y = Σ_k weight_{i,k} · out_{id_{i,k}}  (+ shared_out)
```

- **router**：只产生 `(topk_weights, topk_ids)`，不做专家计算；
- **routed experts**：`E` 个独立 MLP，每个 token 只走其中 `top_k` 个；
- **shared expert**：DeepSeek 系模型常见，每个 rank 都持有，不参与 EP 分发；
- **combine**：按路由权重把多个专家输出加权求和，还原成 `(T, H)`。

工程上的核心矛盾是**负载不均衡**：热门 expert 被大量 token 选中，冷门 expert 空闲，EP 下表现为 rank 间长尾。

## 3. TP / DP / EP 对比

| 维度 | TP（张量并行） | DP（数据并行） | EP（专家并行） |
|---|---|---|---|
| 切分对象 | 每个 expert 权重的列/行 | 模型副本，各管一批 token | 不同 expert 整体放到不同 rank |
| 每 rank 持有的专家 | 所有 expert 的权重切片 | 完整模型（含全部专家） | 一部分专家（`E / EP` 个本地专家） |
| 通信模式 | 每层 all-reduce / all-gather + reduce-scatter | 梯度同步（推理时无） | token all-to-all（dispatch / combine） |
| 专家内计算 | 被切分，依赖通信拼回完整维度 | 各副本独立算全部专家 | 本地专家独立 GEMM，无专家内通信 |
| 适合场景 | 小规模、单节点、hidden 维大 | 吞吐扩展、请求并行 | 大规模、跨节点、专家数多 |

**选型直觉**：

- 专家数少、单节点：TP 下 MoE 用 all-reduce 即可，EP 的 all-to-all 反而更贵；
- 专家数多、跨节点：EP 把专家权重散开，避免每个 rank 都存全量专家，通信换成 token 路由；
- EP 常与 TP/DP 组合：`TP × DP × EP` 三维并行，专家在 EP 组内分布，token 在 DP 组间划分。

## 4. Dispatch-Compute-Combine 三段式

```text
Router 输出 (T, top_k) 的 (weight, expert_id)
   │
   ├─ Dispatch: 按目标 rank/专家把 token 分组重排，记录原 token 位置
   │            all-to-all：每个 rank 把属于其他 rank 的 token 发出去
   │
   ├─ Compute:  本地持有的专家做 grouped GEMM（变长分组）+ 激活函数
   │
   └─ Combine:  all-to-all 把输出发回原 token 所在 rank，按原顺序写回 × 路由权重
```

三段的工程难点（后续课程逐一攻克）：

1. **dispatch 排列**：token 按目标 rank 分组，必须保留"来自哪个原 token、第几个路由槽位"的映射，否则 combine 无法还原；
2. **grouped GEMM**：各专家 token 数不同，kernel 要支持变长分组，且 batch 维度每步都变；
3. **combine 还原**：多份输出可能回到同一 token（top_k 个专家），需要按位置累加并乘权重；
4. **变长通信**：每轮各 rank 收到的 token 数不固定，需要前缀元数据同步（计数/偏移）才能对齐收发。

## 5. DeepEP 架构拆解

DeepEP 是 DeepSeek 开源的 MoE 通信库，vLLM 在 `All2AllManagerBase` 之下以 manager 形式接入。本仓库可见的封装锚点：

| 组件 | 源码锚点 | 作用 |
|---|---|---|
| 抽象基类 | `vllm/distributed/device_communicators/base_device_communicator.py:34` `All2AllManagerBase` | 定义 `dispatch_router_logits` / `dispatch` / `combine` 接口 |
| DeepEP 基类 | `vllm/distributed/device_communicators/all2all.py:205` `DeepEPAll2AllManagerBase` | handle 缓存、`num_sms=20`、`destroy()` |
| 高吞吐模式 HT | `all2all.py:257` `DeepEPHTAll2AllManager` | `low_latency_mode=False`，NVLink + RDMA 双缓冲，prefill 场景 |
| 低延迟模式 LL | `all2all.py:316` `DeepEPLLAll2AllManager` | 低延迟 kernel，decode 场景，带 mask/容错状态 |
| 后端选择 | `vllm/config/parallel.py:128` `_resolve_all2all_backend` | `auto` 按硬件/DP/EP 规模/dtype 选后端 |

### 5.1 HT（High-Throughput）与 LL（Low-Latency）

从 `all2all.py:263` `_make_all2all_kwargs` 可见 HT 的关键参数：

- `num_nvl_bytes`：节点内 NVLink 通信缓冲；
- `num_rdma_bytes`：跨节点 RDMA 缓冲（节点内置 0）；
- `num_qps_per_rank`：每 rank Queue Pair 数，跨节点时 `num_sms // 2`，节点内为 1；
- `low_latency_mode=False`：HT 走批量吞吐优化。

定位区分：

| | HT | LL |
|---|---|---|
| 目标 | 大 batch、prefill 阶段的吞吐 | decode 阶段的延迟 |
| 消息特征 | token 多、消息块大 | token 少、频次高、对尾延迟敏感 |
| 通信优化 | 批量打包、RDMA 大消息 | 轻量同步、mask 快速路径 |

### 5.2 优势与局限（调研结论，待 GPU 实测复核）

**优势**：

- 通信与计算深度重叠，dispatch/combine 用独立 kernel 与缓冲管理；
- 同时覆盖节点内 NVLink 与跨节点 RDMA；
- HT/LL 分模式适配 prefill/decode 两类负载。

**局限**：

- 绑定 NVIDIA + 特定 kernel 栈，安装依赖 `deep_ep` 预编译 kernel（`all2all.py:209` `has_deep_ep()` 断言）；
- 缓冲按启动时配置 sizing，专家容量变化可能需要重配；
- 黑盒程度高，出问题时难以从外部观测分组/计数细节——这正是手写一套透明 EP 库的价值。

## 6. vLLM 现有 MoE 调用链（源码走读锚点）

`MoERunner._forward_impl`（`vllm/model_executor/layers/fused_moe/runner/moe_runner.py:950`）的实际顺序：

```text
forward()  [moe_runner.py:737]
 └─ _forward_entry → _forward_impl()  [:950]
      ├─ shared expert 异步前置（可选）        [:980]
      ├─ gate: router_logits = x @ W_gate    [:988]
      ├─ _maybe_dispatch()                    [:878 / :1000]
      │    └─ naive DP/EP: dispatch_router_logits；PCP: all_gather
      ├─ _apply_quant_method()                [:1005]
      │    └─ router TopK → fused experts kernel（grouped GEMM）
      └─ _maybe_combine()                     [:914 / :1014]
           └─ naive EP: combine；PCP: reduce_scatter；shared 合并
```

关键组件：

| 组件 | 锚点 | 职责 |
|---|---|---|
| `BaseRouter` | `router/base_router.py:147` | 模板方法：`_select_experts`（:248）预处理/后处理，子类实现 `_compute_routing`（:222） |
| `FusedTopKRouter` | `router/fused_topk_router.py:210` | 默认 softmax/sigmoid TopK；`fused_topk`（:160）分配 weights/ids/`token_expert_indices` |
| `MoERunner` | `runner/moe_runner.py:240` | 组织 router / experts / shared / dispatch / combine |
| `ExpertMapManager` | `expert_map_manager.py:152` | 全局 expert id ↔ 本地 expert id 映射；`map_global_to_local`（:338）、`use_ep`（:278） |
| `All2AllManagerBase` | `base_device_communicator.py:34` | 通信抽象：`dispatch`（:95）、`combine`（:161）、`dispatch_router_logits`（:80） |

其他后端对照（`all2all.py`）：`PassThrough:43`、`AgRs:57`、`NixlEP:441`、`FlashInferNVLinkTwoSided:680`、`FlashInferNVLinkOneSided:804`、`Mori:969`、`MoriEPv2:1123`、`DeepEPV2:1222`、`MoonEP:1316`。

## 7. 本项目目标

- 在 4×H100 上实现一套 RouteEP：Router → Dispatch → All-to-All → Grouped GEMM → Combine；
- 正确性对齐：固定输入逐层对比 eager 参考 → vLLM 原生 MoE → DeepEP；
- 性能目标（教程基线，待压测复核）：Decode 约 3.1×、Prefill 约 23% 提升；
- 交付能力：手写路由/重排、变长 A2A、Triton 融合算子、Runner 替换、CUDA Graph 分桶、EP 拓扑扩展与稳定性压测。

## 8. 自检 5 问

1. TP、DP、EP 各自切分什么？EP 下每 rank 存多少专家？
2. Dispatch-Compute-Combine 三段各自的输入输出是什么，哪一步需要保留原 token 位置？
3. HT 和 LL 分别服务 prefill 还是 decode？关键缓冲参数有哪些？
4. `ExpertMapManager` 解决什么问题？为什么 EP 下必须有全局→本地映射？
5. vLLM `_forward_impl` 里 dispatch 和 combine 分别对应哪个方法？中间的专家计算走哪一层？

<details>
<summary>参考答案</summary>

1. TP 切专家权重；DP 切 token 批次；EP 切专家整体。EP 下每 rank 存 `E / EP` 个本地专家。
2. Dispatch：`(T,H)` + 路由表 → 按 rank 重排后的 token 块；Compute：本地专家 grouped GEMM；Combine：输出发回原 token 并加权求和。dispatch 必须保留原 token id 与槽位号。
3. HT 服务 prefill（大 batch 吞吐），LL 服务 decode（低延迟）；参数含 NVLink/RDMA 缓冲字节数、每 rank QP 数、`low_latency_mode`。
4. 路由输出是全局 expert id，EP 下只有部分专家在本地，需要查表得到本地权重下标（或判定不在本 rank）。
5. dispatch 对应 `_maybe_dispatch`（:878），combine 对应 `_maybe_combine`（:914）；中间专家计算走 `_apply_quant_method`（:1005）内的 fused experts kernel。
</details>

## 9. 延伸阅读

- `vllm/model_executor/layers/fused_moe/layer.py`：`FusedMoEFactory` 组装入口；
- `vllm/model_executor/layers/fused_moe/all2all_utils.py`：上层如何取 all2all manager；
- DeepEP 仓库（HT/LL kernel 设计文档）；教程 Module 2/3 会回到本文件的通信与算子细节。
