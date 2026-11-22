"""L2 Router 单测：打分、TopK、归一化、偏差与缩放因子。

无 torch 环境自动 skip；在 GPU/CPU torch 环境直接 ``pytest tests/test_router.py``。
"""

import pytest

torch = pytest.importorskip("torch")

from routeep.config import RouteEPConfig  # noqa: E402
from routeep.router import Router, route  # noqa: E402


def test_softmax_topk_ids_and_renormalize() -> None:
    logits = torch.tensor(
        [[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]]
    )
    weights, ids = route(logits, top_k=2, scoring_func="softmax", renormalize=True)

    assert ids.tolist() == [[0, 1], [3, 2]]
    assert weights.shape == (2, 2)
    # 归一化后每行权重和为 1。
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-6)
    # 权重非负。
    assert (weights >= 0).all()


def test_sigmoid_path_shape_and_valid_ids() -> None:
    logits = torch.randn(4, 5)
    weights, ids = route(logits, top_k=3, scoring_func="sigmoid", renormalize=True)
    assert weights.shape == (4, 3)
    assert ids.shape == (4, 3)
    assert ids.min() >= 0
    assert ids.max() < 5
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-6)


def test_routed_scaling_factor_multiplies_weights() -> None:
    logits = torch.randn(3, 4)
    base_w, base_ids = route(logits, top_k=2, renormalize=True)
    scaled_w, scaled_ids = route(
        logits, top_k=2, renormalize=True, routed_scaling_factor=2.0
    )
    assert torch.equal(base_ids, scaled_ids)
    assert torch.allclose(scaled_w, base_w * 2.0)


def test_e_score_bias_can_flip_selection() -> None:
    # 两个专家 logits 相同，偏差把选择推向专家 1。
    logits = torch.zeros(1, 2)
    bias = torch.tensor([-10.0, 10.0])
    _, ids = route(
        logits, top_k=1, scoring_func="softmax", e_score_correction_bias=bias
    )
    assert ids.item() == 1


def test_router_module_forward_shapes() -> None:
    config = RouteEPConfig(
        num_experts=4, top_k=2, hidden_size=8, intermediate_size=16
    )
    router = Router(config)
    hidden = torch.randn(5, 8)
    logits, weights, ids = router(hidden)
    assert logits.shape == (5, 4)
    assert weights.shape == (5, 2)
    assert ids.shape == (5, 2)


def test_topk_greater_than_experts_raises() -> None:
    logits = torch.randn(2, 3)
    with pytest.raises(ValueError):
        route(logits, top_k=4)
