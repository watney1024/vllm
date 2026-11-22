"""L3 调度元数据单测：重排计数、前缀偏移、去重、无效过滤、往返还原。"""

import pytest

torch = pytest.importorskip("torch")

from routeep.metadata import (  # noqa: E402
    build_dispatch_metadata,
    combine_outputs,
    permute_tokens,
)


def _meta_from(ids: list[list[int]], weights: list[list[float]], E: int):
    return build_dispatch_metadata(
        torch.tensor(weights, dtype=torch.float32),
        torch.tensor(ids, dtype=torch.int64),
        num_experts=E,
    )


def test_expert_counts_and_start_loc() -> None:
    ids = [[0, 1], [1, 2], [0, 2]]
    weights = [[0.6, 0.4], [0.7, 0.3], [0.55, 0.45]]
    meta = _meta_from(ids, weights, E=3)

    assert meta.expert_counts.tolist() == [2, 2, 2]
    assert meta.expert_start_loc.tolist() == [0, 2, 4, 6]
    assert meta.num_routed == 6


def test_entries_grouped_sorted_by_expert() -> None:
    ids = [[2, 0], [1, 2], [0, 1]]
    weights = [[1.0, 1.0]] * 3
    meta = _meta_from(ids, weights, E=3)

    experts = meta.sorted_expert_ids.tolist()
    assert experts == sorted(experts)
    # 每个专家区间内 token 下标升序。
    for e in range(3):
        sl = meta.expert_slice(e)
        toks = meta.sorted_token_ids[sl].tolist()
        assert toks == sorted(toks)


def test_round_trip_restore_weighted_sum() -> None:
    """专家输出取对应 token 本身时，combine 应还原为 hidden × 权重和（=1）。"""
    torch.manual_seed(0)
    T, H, E = 3, 8, 3
    ids = [[0, 1], [1, 2], [0, 2]]
    weights = [[0.6, 0.4], [0.7, 0.3], [0.55, 0.45]]
    meta = _meta_from(ids, weights, E=E)

    hidden = torch.randn(T, H)
    permuted = permute_tokens(hidden, meta)
    # 每条目的“专家输出”就是它来源 token 的 hidden（等价于恒等专家）。
    out = combine_outputs(permuted, meta)
    assert out.shape == (T, H)
    assert torch.allclose(out, hidden, atol=1e-6)


def test_dedup_aggregates_same_token_expert() -> None:
    # token 0 两次命中专家 0，权重应聚合为一条。
    ids = [[0, 0], [1, 2]]
    weights = [[0.6, 0.4], [0.7, 0.3]]
    meta = _meta_from(ids, weights, E=3)

    assert meta.num_routed == 3
    assert meta.expert_counts.tolist() == [1, 1, 1]
    e0_slice = meta.expert_slice(0)
    agg_weight = meta.sorted_weights[e0_slice].item()
    assert torch.allclose(torch.tensor(agg_weight), torch.tensor(1.0))


def test_invalid_slot_dropped() -> None:
    ids = [[0, -1], [1, 2]]
    weights = [[0.6, 0.4], [0.7, 0.3]]
    meta = _meta_from(ids, weights, E=3)

    assert meta.num_routed == 3
    assert (meta.sorted_expert_ids >= 0).all()


def test_empty_routing_when_all_invalid() -> None:
    ids = [[-1, -1], [-1, -1]]
    weights = [[1.0, 1.0], [1.0, 1.0]]
    meta = _meta_from(ids, weights, E=3)

    assert meta.num_routed == 0
    assert meta.expert_counts.tolist() == [0, 0, 0]
    assert meta.expert_start_loc.tolist() == [0, 0, 0, 0]
