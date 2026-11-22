# L2 Router 原理与 Top-K 路由实现

> 对应课程 Module 1（第 2 课）。代码：`routeep/router.py`；测试：`tests/test_router.py`。

## 1. 本节要解决什么

1. Router 如何从 hidden state 得到每个 token 的专家打分；
2. softmax / sigmoid 两种打分路径的区别；
3. TopK 选择、权重归一化、路由偏差、缩放因子各自作用在哪一步；
4. 输出张量的形状与 dtype 约定，供下游 dispatch 使用。

**一句话结论**：Router 只做"打分 → 选 K 个专家 → 算权重"，不碰专家计算；输出 `(T, K)` 的 `topk_weights` 与 `topk_ids`，是后续所有重排与通信的元数据来源。

## 2. 计算流程

```text
hidden_states (T, H)
   │  logits = x @ W_gateᵀ            # (T, E)，W_gate: (E, H)
   │  scores = softmax(logits) 或 sigmoid(logits)
   │  scores += e_score_correction_bias（可选，广播到 E）
   │  topk_weights, topk_ids = scores.topk(K, dim=-1)
   │  renormalize: w ← w / Σw          # 可选
   │  routed_scaling_factor: w ← w × s  # 可选后置缩放
   └─► (topk_weights: (T,K) fp32, topk_ids: (T,K) int64)
```

## 3. 两种打分函数

| | softmax | sigmoid |
|---|---|---|
| 公式 | `exp(l_i) / Σ_j exp(l_j)` | `1 / (1 + exp(-l_i))` |
| 专家间关系 | 互斥，概率和为 1 | 独立打分，不互相压制 |
| 典型模型 | 多数 MoE（Qwen/Mixtral 系） | DeepSeek 系部分配置 |
| 归一化 | 先 softmax 再 TopK，选中后可再 renorm | sigmoid 后 TopK，renorm 按需开启 |

vLLM 对应实现：`router/fused_topk_router.py:160` `fused_topk` 按 `scoring_func` 分派 `topk_softmax` / `topk_sigmoid`；`router/grouped_topk_router.py:31` `fused_grouped_topk` 额外支持专家组筛选（`num_expert_group` / `topk_group`）与偏差。

## 4. 关键细节

### 4.1 路由偏差（e_score_correction_bias）

- 形状 `(E,)`，逐专家加在打分上，用于负载均衡校正（如 aux-loss-free 路由：给热门专家减分、冷门专家加分）；
- 本实现加在 softmax/sigmoid **之后**、TopK 之前，与 vLLM grouped topk 路径一致；
- 偏差只改变选择倾向，不参与最终专家输出加权。

### 4.2 重归一化（renormalize）

TopK 选中的 K 个权重原本是全量 softmax 的一部分，和小于 1。开启 renormalize 后：

```text
w_k ← w_k / Σ_{j∈TopK} w_j
```

使单个 token 的路由权重和为 1，combine 时输出幅度与 token 数解耦。

### 4.3 routed_scaling_factor

归一化之后再乘的常数因子（对齐 vLLM 同名参数），常用于配合量化/激活缩放；默认 1.0 不生效。

### 4.4 无效槽位约定

padding / CUDA Graph 补位的 token，其槽位专家 id 置 `-1`；Router 本身不过滤，由 `metadata.py` 在构造 dispatch 表时丢弃。

## 5. 与 vLLM 源码对照

| 本项目 | vLLM 锚点 | 说明 |
|---|---|---|
| `compute_router_logits` | `runner/moe_runner.py:988` gate 线性层 | logits 由 hidden 过门控权重得到 |
| `route` | `router/fused_topk_router.py:210` `FusedTopKRouter` | 模板方法见 `base_router.py:248` `_select_experts` |
| `Router` 模块 | `router/base_router.py:147` `BaseRouter` | 持有 gate 权重，forward 返回 logits/weights/ids |
| 偏差/缩放 | `router/grouped_topk_router.py:36-40` | `e_score_correction_bias`、`routed_scaling_factor` |

## 6. 常见误区

1. **以为 TopK 在 softmax 之前**：多数实现先全量 softmax/sigmoid 再选 TopK；先 TopK 再在选中项内 softmax 是另一种语义，权重不可混用。
2. **忽略 renormalize**：不开归一化时，路由权重和随专家数、打分分布漂移，combine 输出幅度不稳。
3. **bias 加错位置**：偏差应作用于打分阶段；加到专家输出权重上会改变 combine 幅度。
4. **dtype 混用**：路由权重统一 fp32 累加，专家 id 用整数类型，避免 GEMM 索引类型错误。

## 7. 自检 5 问

1. Router 的输入输出分别是什么形状？Router 是否调用专家 MLP？
2. softmax 与 sigmoid 打分的本质区别是什么？
3. renormalize 在哪一步执行？关掉它权重和是多少？
4. 路由偏差作用在哪个张量上？典型用途是什么？
5. 无效 padding 槽位用什么标记？在哪一层被过滤？

<details>
<summary>参考答案</summary>

1. 输入 `(T, H)` hidden，输出 logits `(T, E)`、weights `(T, K)`、ids `(T, K)`；Router 不调用专家 MLP。
2. softmax 专家间互斥（概率归一），sigmoid 专家独立打分。
3. TopK 之后、combine 之前；关闭时权重和等于全量 softmax 下选中项之和，小于 1。
4. 加在 `(E,)` 专家打分子上，用于负载均衡校正冷热专家。
5. id 置 -1；在 `metadata.build_dispatch_metadata` 构造 dispatch 表时过滤。
</details>
