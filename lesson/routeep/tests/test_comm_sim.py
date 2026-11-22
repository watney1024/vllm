"""通信层单机模拟单测：IdentityAll2All 的 dispatch/combine 往返。"""

import pytest

torch = pytest.importorskip("torch")

from routeep.comm.sim_a2a import IdentityAll2All  # noqa: E402
from routeep.metadata import (  # noqa: E402
    build_dispatch_metadata,
    permute_tokens,
)


def _make_meta():
    ids = [[0, 1], [1, 2], [0, 2], [2, 0]]
    weights = [[0.6, 0.4], [0.7, 0.3], [0.55, 0.45], [0.8, 0.2]]
    return build_dispatch_metadata(
        torch.tensor(weights, dtype=torch.float32),
        torch.tensor(ids, dtype=torch.int64),
        num_experts=3,
    )


def test_identity_dispatch_keeps_entries() -> None:
    meta = _make_meta()
    hidden = torch.randn(4, 8)
    permuted = permute_tokens(hidden, meta)
    backend = IdentityAll2All(ep_size=1, ep_rank=0)

    bundle = backend.dispatch(permuted, meta)
    assert bundle.tokens.shape == (meta.num_routed, 8)
    assert bundle.token_ids.tolist() == meta.sorted_token_ids.tolist()
    assert bundle.expert_ids.tolist() == meta.sorted_expert_ids.tolist()
    assert bundle.recv_counts == [meta.num_routed]


def test_identity_combine_roundtrip() -> None:
    meta = _make_meta()
    torch.manual_seed(1)
    hidden = torch.randn(4, 8)
    permuted = permute_tokens(hidden, meta)
    backend = IdentityAll2All(ep_size=1, ep_rank=0)

    bundle = backend.dispatch(permuted, meta)
    # 恒等专家：输出 = 来源 token 本身。
    combined = backend.combine(bundle, bundle.tokens)

    result = torch.zeros(4, 8, dtype=hidden.dtype)
    result.index_add_(
        0,
        combined.token_ids,
        combined.outputs * combined.weights.unsqueeze(-1),
    )
    assert torch.allclose(result, hidden, atol=1e-6)
