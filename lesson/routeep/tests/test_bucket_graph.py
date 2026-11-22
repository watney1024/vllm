"""CUDA Graph 分桶选择与 eager 回退单测（不依赖真实捕获）。"""

import pytest

torch = pytest.importorskip("torch")

from routeep.graph.bucket_graph import (  # noqa: E402
    BucketedGraphManager,
    GraphNotCapturedError,
)


def test_bucket_id_selection() -> None:
    mgr = BucketedGraphManager([1, 8, 32, 128])
    assert mgr.bucket_id(1) == 0
    assert mgr.bucket_id(2) == 1
    assert mgr.bucket_id(8) == 1
    assert mgr.bucket_id(9) == 2
    assert mgr.bucket_id(128) == 3
    assert mgr.bucket_id(500) == 3  # 最后一桶无上限


def test_run_falls_back_to_eager_without_cuda(monkeypatch) -> None:
    mgr = BucketedGraphManager([1, 8, 32])
    monkeypatch.setattr(mgr, "is_available", staticmethod(lambda: False))

    calls = []

    def forward_fn(x):
        calls.append(x.shape[0])
        return x * 2

    x = torch.randn(5, 4)
    out = mgr.run(x, forward_fn)
    assert calls == [5]
    assert torch.allclose(out, x * 2)


def test_replay_uncaptured_bucket_raises() -> None:
    mgr = BucketedGraphManager([1, 8, 32])
    with pytest.raises(GraphNotCapturedError):
        mgr.replay(0, torch.randn(1, 4))
