"""专家 MLP：SwiGLU 分组前向（eager 参考实现 + 模块封装）。

权重布局约定：

- ``w1``：``(E, 2I, H)``，融合 gate/up；第 1 维前 ``I`` 行为 gate，后 ``I`` 行为 up；
- ``w2``：``(E, H, I)``，down projection。

单个专家计算：

```text
gate_up = x @ w1[e]ᵀ            # (n, 2I)
h = silu(gate) * up             # SwiGLU
y = h @ w2[e]ᵀ                  # (n, H)
```

eager 路径是数值对齐基线；Triton 分组 GEMM 实现在
`routeep/kernels/grouped_gemm.py`，通过设备/可用性选择。
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from routeep.config import RouteEPConfig


def grouped_expert_forward_eager(
    x: torch.Tensor,
    expert_start_loc: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """逐专家执行 SwiGLU MLP（eager 参考实现）。

    Args:
        x: ``(M, H)``，按专家排序后的 token。
        expert_start_loc: ``(E+1,)`` 专家条目前缀偏移。
        w1: ``(E, 2I, H)`` 融合 gate/up 权重。
        w2: ``(E, H, I)`` down 权重。

    Returns:
        ``(M, H)`` 专家输出，顺序与 ``x`` 一致。
    """
    num_experts = w1.shape[0]
    intermediate = w2.shape[2]
    out = torch.zeros_like(x)

    for e in range(num_experts):
        start = int(expert_start_loc[e])
        end = int(expert_start_loc[e + 1])
        if end == start:
            continue
        xe = x[start:end]

        gate_up = xe @ w1[e].t()
        gate, up = gate_up[:, :intermediate], gate_up[:, intermediate:]
        hidden = F.silu(gate) * up
        out[start:end] = hidden @ w2[e].t()

    return out


class GroupedExpertMLP(nn.Module):
    """持有全部专家权重的分组 MLP。

    权重可外部注入（加载 checkpoint），否则按形状新建并初始化。
    """

    def __init__(
        self,
        config: RouteEPConfig,
        w1: torch.Tensor | None = None,
        w2: torch.Tensor | None = None,
    ):
        super().__init__()
        self.config = config
        E, H, I = config.num_experts, config.hidden_size, config.intermediate_size

        if w1 is None:
            w1 = torch.empty(E, 2 * I, H)
            nn.init.kaiming_uniform_(w1, a=math.sqrt(5))
        if w2 is None:
            w2 = torch.empty(E, H, I)
            nn.init.kaiming_uniform_(w2, a=math.sqrt(5))

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)

    def forward(
        self,
        x: torch.Tensor,
        expert_start_loc: torch.Tensor,
        *,
        use_triton: bool = True,
    ) -> torch.Tensor:
        """分组前向。

        Args:
            x: ``(M, H)`` 按专家排序后的 token。
            expert_start_loc: ``(E+1,)`` 前缀偏移。
            use_triton: 设备支持时是否走 Triton 分组 GEMM；否则 eager。
        """
        if use_triton and x.is_cuda:
            from routeep.kernels.grouped_gemm import grouped_expert_forward_triton

            return grouped_expert_forward_triton(x, expert_start_loc, self.w1, self.w2)
        return grouped_expert_forward_eager(x, expert_start_loc, self.w1, self.w2)
