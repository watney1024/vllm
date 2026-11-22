# L10 接入 RouteEP：自定义 MoE Runner 替换

> 对应课程 Module 4（第 10 课）。代码：`routeep/moe_runner.py`、`routeep/local_plan.py`。

## 1. 本节要解决什么

1. 把 Router、Dispatch、All-to-All、Expert、Combine 封装成一个 Runner；
2. 明确 Runner 的输入输出契约；
3. 单机单 rank 跑通一轮 prefill/decode 形状与数值；
4. 预留多 rank / vLLM method 注册的接入点。

**一句话结论**：Runner 是组合根——内部持有 router、experts、backend 三个子模块，前向按"路由 → 元数据 → 分发 → 本地计算 → 汇合"串联；换通信后端只改注入的 `All2AllBackend`，换 kernel 只改 experts 内部实现。

## 2. 前向数据流

```text
hidden (T,H)
  │  router（或外部 router_logits）
  ├─► topk_weights (T,K), topk_ids (T,K)
  ├─► DispatchMetadata（排序/去重/计数/偏移）
  ├─► gather_tokens → permuted (M,H)
  ├─► backend.dispatch → DispatchBundle（本 rank 收到的条目）
  ├─► LocalPlan：bundle → 本地专家排序 + 逆置换
  ├─► GroupedExpertMLP(local_tokens, local_start_loc) → local_out
  ├─► 逆置换回 bundle 顺序
  ├─► backend.combine → CombineBundle（源 rank 视角）
  └─► combine_scatter_add → out (T,H)
```

## 3. 接口契约

### 输入

- `hidden_states: (T, H)`；
- `router_logits: (T, E) | None`：外部已算 logits 时跳过内部 gate。

### 输出

- `(T, H)` routed 输出；不含 shared expert（后续叠加）。

### 子模块注入

| 子模块 | 类型 | 默认 |
|---|---|---|
| `router` | `Router` | `build_default_runner` 内部构造 |
| `experts` | `GroupedExpertMLP` | 内部构造，权重可外部注入 |
| `backend` | `All2AllBackend` | `IdentityAll2All`（单 rank）；GPU 多 rank 换 `NcclAll2All` |

## 4. 本地排序与逆置换

接收侧 bundle 的条目按全局专家到达，本地 GEMM 需要按本地专家分段：

1. `expert_map[global_id] → local_id`；
2. 按 `local_id` 稳定排序得 `order`，同时记录 `inv_order`；
3. `expert_start_loc` 供 grouped GEMM 分段；
4. 专家输出经 `inv_order` 还原回 bundle 顺序，再走 combine A2A。

## 5. 验证步骤（GPU 机器）

1. **形状验证**：随机 hidden，前向不报错，输出 `(T, H)`；
2. **数值对齐**：与 eager 参考 MoE 逐 token 对比（`tests/test_runner_e2e.py`）；
3. **prefill**：长 prompt（大 T）跑一轮；
4. **decode**：单 token（T=1）连续多步，检查空段与计数表；
5. **多 rank**：注入 `NcclAll2All`，`torchrun` 两 rank 起验证 dispatch/combine 往返。

## 6. 常见误区

1. 本地计算后忘记逆置换，combine 输出与溯源信息错位；
2. 把全局专家 id 直接当本地权重下标；
3. 空 batch（M=0）未单独处理，GEMM 启动空张量报错；
4. Runner 内硬编码通信后端，无法在单机模拟与多 rank 间切换。

## 7. 自检

1. Runner 持有哪三个子模块？各自职责？
2. `LocalPlan.order` 与 `inv_order` 分别用于什么？
3. 外部传入 `router_logits` 时跳过哪一步？
4. 单 rank 与多 rank 分别用什么 backend？如何切换？
