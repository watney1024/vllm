"""分组 GEMM eager 基线单测；Triton 路径需 CUDA 环境验证。"""

import pytest

torch = pytest.importorskip("torch")

from routeep.kernels.grouped_gemm import _grouped_gemm_eager  # noqa: E402


def test_grouped_gemm_eager_matches_manual() -> None:
    E, M, N, K = 3, 6, 5, 4
    x = torch.randn(M, K)
    start_loc = torch.tensor([0, 2, 4, 6])
    W = torch.randn(E, N, K)

    out = _grouped_gemm_eager(x, W, start_loc)

    expected = torch.zeros(M, N)
    for e in range(E):
        s, t = e * 2, e * 2 + 2
        expected[s:t] = x[s:t] @ W[e].t()

    assert out.shape == (M, N)
    assert torch.allclose(out, expected, atol=1e-6)


def test_grouped_gemm_empty_segment() -> None:
    E, M, N, K = 3, 4, 5, 4
    x = torch.randn(M, K)
    start_loc = torch.tensor([0, 2, 2, 4])
    W = torch.randn(E, N, K)
    out = _grouped_gemm_eager(x, W, start_loc)
    assert out.shape == (M, N)
    assert torch.isfinite(out).all()
