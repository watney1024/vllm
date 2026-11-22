"""Stage 4 端到端单机模拟：Runner 全链路与 eager 参考 MoE 对齐。"""

import pytest

torch = pytest.importorskip("torch")

from routeep.config import RouteEPConfig  # noqa: E402
from routeep.experts.mlp import grouped_expert_forward_eager  # noqa: E402
from routeep.metadata import build_dispatch_metadata, combine_outputs  # noqa: E402
from routeep.moe_runner import build_default_runner  # noqa: E402
from routeep.router import route  # noqa: E402


def _reference_moe(x, gate, w1, w2, config):
    logits = x @ gate.t()
    weights, ids = route(
        logits,
        top_k=config.top_k,
        scoring_func=config.scoring_func,
        renormalize=config.renormalize,
    )
    meta = build_dispatch_metadata(weights, ids, config.num_experts)
    permuted = x[meta.sorted_token_ids]
    expert_out = grouped_expert_forward_eager(
        permuted, meta.expert_start_loc, w1, w2
    )
    return combine_outputs(expert_out, meta)


def test_runner_matches_eager_reference() -> None:
    torch.manual_seed(0)
    E, H, I, T, K = 4, 8, 6, 7, 2
    config = RouteEPConfig(
        num_experts=E, top_k=K, hidden_size=H, intermediate_size=I
    )
    runner = build_default_runner(config)

    # 注入同一组权重，保证两边计算一致。
    gate = runner.router.gate_weight.detach().clone()
    w1 = runner.experts.w1.detach().clone()
    w2 = runner.experts.w2.detach().clone()

    x = torch.randn(T, H)
    expected = _reference_moe(x, gate, w1, w2, config)
    out = runner(x)

    assert out.shape == (T, H)
    assert torch.allclose(out, expected, atol=1e-6)


def test_runner_with_external_router_logits() -> None:
    torch.manual_seed(1)
    E, H, I, T, K = 4, 8, 6, 5, 2
    config = RouteEPConfig(
        num_experts=E, top_k=K, hidden_size=H, intermediate_size=I
    )
    runner = build_default_runner(config)

    x = torch.randn(T, H)
    router_logits = torch.randn(T, E)
    out = runner(x, router_logits=router_logits)
    assert out.shape == (T, H)
    assert torch.isfinite(out).all()
