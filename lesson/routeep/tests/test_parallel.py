"""拓扑配置单测：TP×DP×EP 一致性校验与 EP 组划分。"""

import pytest

from routeep.parallel import ParallelTopology


def test_world_size_is_tp_times_dp() -> None:
    topo = ParallelTopology(tp_size=2, dp_size=4, ep_size=2)
    assert topo.world_size == 8


def test_validate_expert_divisibility() -> None:
    topo = ParallelTopology(ep_size=4)
    topo.validate(num_experts=16)  # 不抛异常即通过


def test_validate_rejects_undivisible_experts() -> None:
    topo = ParallelTopology(ep_size=4)
    with pytest.raises(ValueError):
        topo.validate(num_experts=10)


def test_validate_rejects_ep_greater_than_dp() -> None:
    topo = ParallelTopology(dp_size=2, ep_size=4)
    with pytest.raises(ValueError):
        topo.validate(num_experts=16)


def test_ep_group_dp_ranks() -> None:
    topo = ParallelTopology(dp_size=8, ep_size=2)
    assert topo.ep_group_dp_ranks(0) == [0, 1]
    assert topo.ep_group_dp_ranks(3) == [2, 3]
    assert topo.ep_group_dp_ranks(7) == [6, 7]
