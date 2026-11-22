"""L2：MoE Router —— logits 打分、TopK 选择、权重归一化。

对齐 vLLM ``FusedTopKRouter`` / ``fused_grouped_topk`` 的语义：

- softmax：先对全部专家 logits 做 softmax，再取 TopK；
- sigmoid：先对 logits 做 sigmoid，再取 TopK；
- ``e_score_correction_bias``：加到打分上（路由偏差，如 aux-loss-free bias）；
- ``renormalize``：选中的 K 个权重重新归一化；
- ``routed_scaling_factor``：归一化之后再乘的缩放因子。

torch 在函数内使用，模块顶层不 import torch，保证无 torch 环境下
``import routeep.router`` 不会直接崩（调用时才报依赖缺失）。
"""

import math

import torch
from torch import nn

from routeep.config import RouteEPConfig


def compute_router_logits(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算路由 logits。

    Args:
        hidden_states: ``(T, H)``。
        gate_weight: ``(E, H)`` 门控权重。
        gate_bias: 可选 ``(E,)`` 门控偏置。

    Returns:
        ``(T, E)`` 的 router logits。
    """
    logits = hidden_states @ gate_weight.t()
    if gate_bias is not None:
        logits = logits + gate_bias
    return logits


def route(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    scoring_func: str = "softmax",
    renormalize: bool = True,
    e_score_correction_bias: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从 router logits 选出 TopK 专家并计算路由权重。

    Args:
        router_logits: ``(T, E)``。
        top_k: 选中专家数 K。
        scoring_func: ``"softmax"`` / ``"sigmoid"``。
        renormalize: 选中后是否归一化权重。
        e_score_correction_bias: ``(E,)`` 路由偏差，加到打分上。
        routed_scaling_factor: 权重后置缩放。

    Returns:
        ``(topk_weights, topk_ids)``：
        ``topk_weights`` 为 ``(T, K)`` float32；``topk_ids`` 为 ``(T, K)`` int64。
        被 mask 的无效槽位在调用侧应把 id 置为 -1（见 metadata 过滤）。
    """
    if router_logits.ndim != 2:
        raise ValueError(f"router_logits 必须是二维 (T, E)，得到 {router_logits.shape}")
    num_tokens, num_experts = router_logits.shape
    if top_k > num_experts:
        raise ValueError(f"top_k={top_k} 大于专家数 {num_experts}")

    if scoring_func == "softmax":
        scores = torch.softmax(router_logits, dim=-1)
    elif scoring_func == "sigmoid":
        scores = torch.sigmoid(router_logits)
    else:
        raise ValueError(f"不支持的 scoring_func: {scoring_func!r}")

    if e_score_correction_bias is not None:
        if e_score_correction_bias.shape != (num_experts,):
            raise ValueError(
                f"bias 形状 {e_score_correction_bias.shape} 与专家数 "
                f"{num_experts} 不匹配"
            )
        scores = scores + e_score_correction_bias

    topk_weights, topk_ids = torch.topk(scores, k=top_k, dim=-1, sorted=False)

    if renormalize:
        denom = topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        topk_weights = topk_weights / denom

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor

    return topk_weights.to(torch.float32), topk_ids.to(torch.int64)


class Router(nn.Module):
    """持有门控权重的 Router 模块。

    用法::

        router = Router(config)
        logits, weights, ids = router(hidden_states)
    """

    def __init__(self, config: RouteEPConfig, gate_weight: torch.Tensor | None = None):
        super().__init__()
        self.config = config
        if gate_weight is None:
            gate_weight = torch.empty(config.num_experts, config.hidden_size)
            nn.init.kaiming_uniform_(gate_weight, a=math.sqrt(5))
        self.gate_weight = nn.Parameter(gate_weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        e_score_correction_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = compute_router_logits(hidden_states, self.gate_weight)
        topk_weights, topk_ids = route(
            logits,
            top_k=self.config.top_k,
            scoring_func=self.config.scoring_func,
            renormalize=self.config.renormalize,
            e_score_correction_bias=e_score_correction_bias,
            routed_scaling_factor=self.config.routed_scaling_factor,
        )
        return logits, topk_weights, topk_ids
