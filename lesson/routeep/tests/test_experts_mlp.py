"""专家 MLP eager 前向单测：与逐专家手工计算对齐。"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from routeep.config import RouteEPConfig  # noqa: E402
from routeep.experts.mlp import (  # noqa: E402
    GroupedExpertMLP,
    grouped_expert_forward_eager,
)


def _weights(E, H, I):
    torch.manual_seed(0)
    w1 = torch.randn(E, 2 * I, H, dtype=torch.float32)
    w2 = torch.randn(E, H, I, dtype=torch.float32)
    return w1, w2


def test_eager_forward_matches_manual() -> None:
    E, H, I, per_e = 3, 8, 4, 2
    M = E * per_e
    x = torch.randn(M, H)
    start_loc = torch.tensor([0, 2, 4, 6])
    w1, w2 = _weights(E, H, I)

    out = grouped_expert_forward_eager(x, start_loc, w1, w2)

    expected = torch.zeros_like(x)
    for e in range(E):
        xe = x[e * per_e : (e + 1) * per_e]
        gate_up = xe @ w1[e].t()
        gate, up = gate_up[:, :I], gate_up[:, I:]
        hidden = F.silu(gate) * up
        expected[e * per_e : (e + 1) * per_e] = hidden @ w2[e].t()

    assert out.shape == (M, H)
    assert torch.allclose(out, expected, atol=1e-6)


def test_empty_expert_skipped() -> None:
    E, H, I = 3, 8, 4
    # 专家 1 无 token。
    x = torch.randn(4, H)
    start_loc = torch.tensor([0, 2, 2, 4])
    w1, w2 = _weights(E, H, I)
    out = grouped_expert_forward_eager(x, start_loc, w1, w2)
    assert out.shape == (4, H)
    assert torch.isfinite(out).all()


def test_module_forward_eager_path() -> None:
    config = RouteEPConfig(
        num_experts=3, top_k=2, hidden_size=8, intermediate_size=4
    )
    model = GroupedExpertMLP(config)
    x = torch.randn(6, 8)
    start_loc = torch.tensor([0, 2, 4, 6])
    out = model(x, start_loc, use_triton=False)
    assert out.shape == (6, 8)
    assert torch.isfinite(out).all()
