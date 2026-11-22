"""L10：RouteEP 自定义 MoE Runner。

组装完整 EP 前向链路：

```text
hidden_states (T, H)
  ├─ Router: logits → topk_weights / topk_ids
  ├─ build_dispatch_metadata: 重排 / 去重 / 计数表
  ├─ gather_tokens: 按 dispatch 顺序取 token
  ├─ backend.dispatch: All-to-All 发到专家所在 rank
  ├─ build_local_plan: 接收侧按本地专家排序
  ├─ GroupedExpertMLP: 本地专家 SwiGLU 分组 GEMM
  ├─ backend.combine: All-to-All 发回来源 rank
  └─ combine_scatter_add: 按原 token 加权累加 → (T, H)
```

与 vLLM `MoERunner`（`runner/moe_runner.py:240`）的接口对齐思路：
输入 hidden_states（可选外部 router_logits），输出 routed hidden；
shared expert、量化、PCP/SP 等扩展在后续阶段接入。
"""

import torch
from torch import nn

from routeep.comm.base import All2AllBackend
from routeep.comm.placement import build_expert_map
from routeep.comm.sim_a2a import IdentityAll2All
from routeep.config import RouteEPConfig
from routeep.experts.mlp import GroupedExpertMLP
from routeep.kernels.dispatch_combine import combine_scatter_add, gather_tokens
from routeep.local_plan import build_local_plan
from routeep.metadata import build_dispatch_metadata
from routeep.router import Router, route


class RouteEPRunner(nn.Module):
    """RouteEP MoE Runner：串联路由、通信、专家计算与汇合。"""

    def __init__(
        self,
        config: RouteEPConfig,
        router: Router | None = None,
        experts: GroupedExpertMLP | None = None,
        backend: All2AllBackend | None = None,
        strategy: str = "linear",
        ep_rank: int = 0,
    ):
        super().__init__()
        self.config = config
        self.ep_rank = ep_rank

        self.router = router
        self.experts = experts
        self.backend = backend if backend is not None else IdentityAll2All(
            config.ep_size, ep_rank
        )

        _, self.expert_map = build_expert_map(
            config.ep_size, ep_rank, config.num_experts, strategy
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行 MoE 前向。

        Args:
            hidden_states: ``(T, H)``。
            router_logits: 可选外部已算好的 ``(T, E)`` 路由 logits；
                为 None 时由内部 Router 过门控权重计算。

        Returns:
            ``(T, H)`` routed 输出（不含 shared expert）。
        """
        num_tokens = hidden_states.shape[0]

        # 1. 路由
        if router_logits is None:
            if self.router is None:
                raise ValueError("router_logits 为 None 时必须提供内部 Router")
            _, topk_weights, topk_ids = self.router(hidden_states)
        else:
            topk_weights, topk_ids = route(
                router_logits,
                top_k=self.config.top_k,
                scoring_func=self.config.scoring_func,
                renormalize=self.config.renormalize,
                routed_scaling_factor=self.config.routed_scaling_factor,
            )

        # 2. 调度元数据 + dispatch 重排
        meta = build_dispatch_metadata(
            topk_weights, topk_ids, self.config.num_experts
        )
        permuted = gather_tokens(hidden_states, meta)

        # 3. All-to-All dispatch
        bundle = self.backend.dispatch(permuted, meta)

        # 4. 接收侧本地专家分组与 Grouped GEMM
        plan = build_local_plan(
            bundle.expert_ids, self.expert_map, self.config.local_num_experts
        )
        if plan.order.numel() > 0:
            local_tokens = bundle.tokens[plan.order]
            local_out = self.experts(local_tokens, plan.expert_start_loc)
            bundle_out = local_out[plan.inv_order]
        else:
            bundle_out = bundle.tokens.new_empty(
                (0, bundle.tokens.shape[1])
            )

        # 5. All-to-All combine + 加权还原
        combined = self.backend.combine(bundle, bundle_out)
        return combine_scatter_add(
            combined.outputs,
            combined.token_ids,
            combined.weights,
            num_tokens,
        )


def build_default_runner(
    config: RouteEPConfig,
    ep_rank: int = 0,
    strategy: str = "linear",
    backend: All2AllBackend | None = None,
) -> RouteEPRunner:
    """构造一套默认 RouteEPRunner：内部 Router + 分组专家 MLP。

    Args:
        config: MoE 配置。
        ep_rank: 当前 rank。
        strategy: 专家放置策略。
        backend: 通信后端；默认单 rank 恒等后端，GPU 多 rank 时注入
            ``NcclAll2All``。
    """
    router = Router(config)
    experts = GroupedExpertMLP(config)
    return RouteEPRunner(
        config,
        router=router,
        experts=experts,
        backend=backend,
        strategy=strategy,
        ep_rank=ep_rank,
    )
