"""分组辅助函数单测：按 rank 排序、计数与区间。"""

import pytest

torch = pytest.importorskip("torch")

from routeep.comm.grouping import group_by_rank  # noqa: E402


def test_group_by_rank_counts_and_bounds() -> None:
    target = torch.tensor([0, 2, 0, 1, 2, 2])
    order, counts, bounds = group_by_rank(target, world_size=3)

    assert counts == [2, 1, 3]
    assert bounds == [(0, 2), (2, 3), (3, 6)]
    sorted_ranks = target[order]
    assert sorted_ranks.tolist() == [0, 0, 1, 2, 2, 2]


def test_group_by_rank_empty() -> None:
    target = torch.empty(0, dtype=torch.long)
    order, counts, bounds = group_by_rank(target, world_size=3)
    assert counts == [0, 0, 0]
    assert bounds == [(0, 0), (0, 0), (0, 0)]
    assert order.numel() == 0


def test_group_by_rank_stable_order() -> None:
    target = torch.tensor([1, 0, 1, 0])
    order, _, bounds = group_by_rank(target, world_size=2)
    # 同 rank 内保留原始相对顺序。
    first_rank0 = order[:2].tolist()
    assert first_rank0 == [1, 3]
