"""Dispatch/Combine gather 与 scatter-add 单测。"""

import pytest

torch = pytest.importorskip("torch")

from routeep.kernels.dispatch_combine import (  # noqa: E402
    combine_scatter_add,
    gather_tokens,
)


def test_gather_matches_indexing() -> None:
    x = torch.randn(5, 8)
    token_ids = torch.tensor([2, 0, 4, 1])
    out = gather_tokens(x, token_ids)
    assert torch.allclose(out, x[token_ids])


def test_gather_empty() -> None:
    x = torch.randn(5, 8)
    out = gather_tokens(x, torch.empty(0, dtype=torch.long))
    assert out.shape == (0, 8)


def test_scatter_add_matches_index_add() -> None:
    torch.manual_seed(1)
    M, T, H = 6, 4, 8
    outputs = torch.randn(M, H)
    token_ids = torch.tensor([0, 1, 2, 0, 3, 1])
    weights = torch.rand(M)

    out = combine_scatter_add(outputs, token_ids, weights, num_tokens=T)

    ref = torch.zeros(T, H)
    ref.index_add_(0, token_ids, outputs * weights.unsqueeze(-1))
    assert torch.allclose(out, ref, atol=1e-6)


def test_scatter_add_empty() -> None:
    out = combine_scatter_add(
        torch.empty(0, 8),
        torch.empty(0, dtype=torch.long),
        torch.empty(0),
        num_tokens=3,
    )
    assert out.shape == (3, 8)
    assert (out == 0).all()
