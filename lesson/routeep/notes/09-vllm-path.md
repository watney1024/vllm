# L9 vLLM MoE 执行路径源码走读

> 对应课程 Module 4（第 9 课）。锚点基于 `main@8c17e46`。

## 1. 本节要解决什么

1. vLLM 原有 MoE Runner / MoE 层的调用链路；
2. 输入输出 tensor 布局与元数据格式；
3. 通信抽象与专家映射在哪一层；
4. 我们的 Runner 对接时需要满足哪些接口约定。

**一句话结论**：vLLM 的 MoE 前向是 `gate → dispatch → 路由+专家 kernel → combine` 四段，Runner 负责组织 router、experts、shared expert 与 all2all manager；通信被抽象成 `All2AllManagerBase`，上层不感知后端是 DeepEP/NIXL/FlashInfer。

## 2. 调用链（`MoERunner._forward_impl`）

锚点：`vllm/model_executor/layers/fused_moe/runner/moe_runner.py:950`

```text
forward()                                  moe_runner.py:737
 └─ _forward_entry → _forward_impl()        :950
      ├─ shared expert 异步前置（可选）      :980
      ├─ gate: router_logits = x @ W_gate  :988
      ├─ _sequence_parallel_context()       :996
      ├─ _maybe_dispatch()                 :878 / :1000
      │    ├─ naive DP/EP: dispatch_router_logits
      │    └─ PCP: all_gather hidden + logits
      ├─ _apply_quant_method()             :1005
      │    └─ router TopK → fused experts grouped GEMM
      └─ _maybe_combine()                  :914 / :1014
           ├─ naive EP: combine
           ├─ PCP: reduce_scatter
           └─ shared + routed 合并
```

## 3. 关键组件与锚点

| 组件 | 锚点 | 职责 |
|---|---|---|
| `MoERunner` | `runner/moe_runner.py:240` | 持有 gate、routed experts、shared experts、quant method |
| `BaseRouter` | `router/base_router.py:147` | 模板方法 `_select_experts`（:248）；子类实现 `_compute_routing`（:222） |
| `FusedTopKRouter` | `router/fused_topk_router.py:210` | softmax/sigmoid TopK；`fused_topk`（:160） |
| `ExpertMapManager` | `expert_map_manager.py:152` | 全局↔本地专家映射；`map_global_to_local`（:338）、`use_ep`（:278） |
| `All2AllManagerBase` | `base_device_communicator.py:34` | `dispatch_router_logits`（:80）、`dispatch`（:95）、`combine`（:161） |
| 后端实现 | `device_communicators/all2all.py` | PassThrough:43、AgRs:57、DeepEPHT:257、DeepEPLL:316、NixlEP:441、FlashInfer:680/804、Mori:969/1123、DeepEPV2:1222、MoonEP:1316 |

## 4. Tensor 布局与元数据

- **hidden_states**：`(total_tokens, hidden_size)`，prefill/decode 混合 batch，第 0 维是 token 维；
- **router_logits**：`(total_tokens, num_experts)`；
- **topk_weights / topk_ids**：`(total_tokens, top_k)`；
- **dispatch 后**：按专家分组的 token 块 + 溯源信息（原 token id、槽位、权重）；
- **专家权重**：`w1 (E, 2I, H)` 融合 gate/up、`w2 (E, H, I)`（本项目沿用此布局）；
- **combine 后**：`(total_tokens, hidden_size)`，按原 token 顺序加权累加。

## 5. EP 下的映射问题

路由输出是**全局专家 id**；EP 下每个 rank 只持有部分专家。`ExpertMapManager` / 本项目 `placement.build_expert_map` 提供：

- `expert_map[global_id] = local_id`，非本地为 -1；
- 收到的条目先查映射再取本地权重；
- 放置策略：`linear` 连续分片 / `round_robin` 轮转。

## 6. 对接要点（本项目 Runner）

| vLLM 概念 | RouteEP 对应 |
|---|---|
| `MoERunner.forward(hidden, router_logits, ...)` | `RouteEPRunner.forward(hidden_states, router_logits=None)` |
| router 输出 `(weights, ids)` | `router.route` / `Router.forward` |
| dispatch 元数据 | `metadata.build_dispatch_metadata` |
| all2all manager | `comm.base.All2AllBackend`（`NcclAll2All` / `IdentityAll2All`） |
| 本地专家分组表 | `local_plan.build_local_plan` |
| fused experts kernel | `kernels.grouped_gemm` + `experts.mlp` |
| combine 还原 | `kernels.dispatch_combine.combine_scatter_add` |

## 7. 尚未接入的 vLLM 能力（后续阶段）

- shared expert 前向与合并；
- 量化输入/输出（fp8、NVFP4 等）与 scale 传递；
- sequence parallel / PCP 的 all-gather / reduce-scatter；
- EPLB 冗余专家与逻辑→物理映射；
- 作为 vLLM 可注册 MoE method 的封装（对齐 `FusedMoEMethodBase`）。
