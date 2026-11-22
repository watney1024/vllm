"""专家放置策略单测：linear / round_robin 的映射与归属。"""

import pytest

torch = pytest.importorskip("torch")

from routeep.comm.placement import (  # noqa: E402
    build_expert_map,
    expert_to_rank,
    rank_of_experts,
)


def test_ep1_identity_map() -> None:
    local_num, emap = build_expert_map(1, 0, 6, "linear")
    assert local_num == 6
    assert emap.tolist() == list(range(6))


def test_linear_segments_even() -> None:
    # 8 专家 / 4 rank，每 rank 2 个。
    local_num, emap = build_expert_map(4, 1, 8, "linear")
    assert local_num == 2
    assert emap.tolist() == [-1, -1, 0, 1, -1, -1, -1, -1]
    assert expert_to_rank(2, 8, 4, "linear") == 1
    assert expert_to_rank(7, 8, 4, "linear") == 3


def test_linear_segments_remainder() -> None:
    # 10 专家 / 4 rank：余数 2，前两个 rank 各 3 个。
    rank0_num, map0 = build_expert_map(4, 0, 10, "linear")
    rank3_num, map3 = build_expert_map(4, 3, 10, "linear")
    assert rank0_num == 3
    assert rank3_num == 2
    assert (map0 >= 0).sum() == 3
    assert (map3 >= 0).sum() == 2
    assert expert_to_rank(9, 10, 4, "linear") == 3


def test_round_robin_map() -> None:
    local_num, emap = build_expert_map(4, 1, 8, "round_robin")
    assert local_num == 2
    # rank 1 持有专家 1, 5。
    assert emap[1] == 0
    assert emap[5] == 1
    assert expert_to_rank(5, 8, 4, "round_robin") == 1


def test_rank_of_experts_matches() -> None:
    for strategy in ("linear", "round_robin"):
        ranks = rank_of_experts(10, 4, strategy)
        for e in range(10):
            assert ranks[e].item() == expert_to_rank(e, 10, 4, strategy)
