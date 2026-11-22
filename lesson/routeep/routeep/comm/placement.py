"""专家放置策略：全局专家 id ↔ rank / 本地专家 id 映射。

对齐 vLLM ``determine_expert_map``（`expert_map_manager.py:22`）的两种策略：

- ``linear``：连续分片，rank r 持有一段连续专家；
- ``round_robin``：轮转，专家 e 放在 rank ``e % ep_size``。
"""

import torch


def _linear_segments(
    num_experts: int, ep_size: int
) -> list[tuple[int, int]]:
    """返回每个 rank 在 linear 策略下的 ``[start, end)`` 专家区间。"""
    base, remainder = divmod(num_experts, ep_size)
    segments: list[tuple[int, int]] = []
    start = 0
    for r in range(ep_size):
        count = base + (1 if r < remainder else 0)
        segments.append((start, start + count))
        start += count
    return segments


def expert_to_rank(
    expert_id: int, num_experts: int, ep_size: int, strategy: str = "linear"
) -> int:
    """全局专家 id 所属的 rank。"""
    if strategy == "round_robin":
        return expert_id % ep_size
    if strategy == "linear":
        for r, (start, end) in enumerate(_linear_segments(num_experts, ep_size)):
            if start <= expert_id < end:
                return r
        raise ValueError(f"expert_id {expert_id} 越界")
    raise ValueError(f"不支持的放置策略: {strategy!r}")


def build_expert_map(
    ep_size: int,
    ep_rank: int,
    num_experts: int,
    strategy: str = "linear",
) -> tuple[int, torch.Tensor]:
    """构造全局→本地专家映射表。

    Args:
        ep_size: EP 组大小。
        ep_rank: 当前 rank。
        num_experts: 全局专家数 E。
        strategy: ``linear`` / ``round_robin``。

    Returns:
        ``(local_num_experts, expert_map)``：
        ``expert_map`` 形状 ``(E,)``，全局 id → 本地下标；非本 rank 专家为 -1。
        ``ep_size == 1`` 时 map 为恒等映射。
    """
    if ep_size == 1:
        return num_experts, torch.arange(num_experts, dtype=torch.int64)

    base, remainder = divmod(num_experts, ep_size)
    local_num = base + (1 if ep_rank < remainder else 0)
    expert_map = torch.full((num_experts,), -1, dtype=torch.int64)

    if strategy == "linear":
        start, end = _linear_segments(num_experts, ep_size)[ep_rank]
        expert_map[start:end] = torch.arange(end - start)
    elif strategy == "round_robin":
        local_ids = torch.arange(ep_rank, num_experts, ep_size)
        expert_map[local_ids] = torch.arange(local_num)
    else:
        raise ValueError(f"不支持的放置策略: {strategy!r}")

    return local_num, expert_map


def rank_of_experts(
    num_experts: int, ep_size: int, strategy: str = "linear"
) -> torch.Tensor:
    """返回每个全局专家所在 rank 的张量 ``(E,)``。"""
    if strategy == "round_robin":
        return torch.arange(num_experts) % ep_size
    if strategy == "linear":
        ranks = torch.empty(num_experts, dtype=torch.int64)
        for r, (start, end) in enumerate(_linear_segments(num_experts, ep_size)):
            ranks[start:end] = r
        return ranks
    raise ValueError(f"不支持的放置策略: {strategy!r}")
