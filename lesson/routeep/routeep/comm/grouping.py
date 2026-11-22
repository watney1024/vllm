"""按目标 rank 分组的纯张量辅助函数（不依赖分布式运行时）。"""

import torch


def group_by_rank(
    target_ranks: torch.Tensor, world_size: int
) -> tuple[torch.Tensor, list[int], list[tuple[int, int]]]:
    """把条目按目标 rank 稳定排序。

    Args:
        target_ranks: ``(M,)`` 每条目的目标 rank。
        world_size: rank 数。

    Returns:
        ``(order, send_counts, bounds)``：
        - ``order``：排序后的下标；
        - ``send_counts[r]``：发往 rank r 的条目数；
        - ``bounds[r]``：rank r 条目在排序表中的 ``(start, end)`` 切片。
    """
    order = torch.argsort(target_ranks, stable=True)
    counts = torch.bincount(target_ranks, minlength=world_size).tolist()
    bounds: list[tuple[int, int]] = []
    start = 0
    for count in counts:
        bounds.append((start, start + count))
        start += count
    return order, counts, bounds
