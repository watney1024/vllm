"""接收侧本地专家排序：bundle 条目 → 本地专家分组顺序。

dispatch 到达本 rank 的条目只持有本地专家；把全局专家 id 经
``expert_map`` 映射成本地下标后，按本地专家稳定排序，得到专家 GEMM
需要的分段表，并保留逆置换以便把输出还原回 bundle 顺序。
"""

from dataclasses import dataclass

import torch


@dataclass
class LocalPlan:
    """本地排序计划。

    Attributes:
        order: ``(M,)`` bundle 下标 → 本地排序位置。
        inv_order: ``(M,)`` 本地排序位置 → bundle 下标。
        local_expert_ids: ``(M,)`` 本地专家下标。
        expert_counts: ``(local_E,)`` 每个本地专家条目数。
        expert_start_loc: ``(local_E+1,)`` 本地专家前缀偏移。
    """

    order: torch.Tensor
    inv_order: torch.Tensor
    local_expert_ids: torch.Tensor
    expert_counts: torch.Tensor
    expert_start_loc: torch.Tensor


def build_local_plan(
    expert_ids: torch.Tensor,
    expert_map: torch.Tensor,
    local_num_experts: int,
) -> LocalPlan:
    """由 bundle 的全局专家 id 构造本地排序计划。

    Args:
        expert_ids: ``(M,)`` 全局专家 id。
        expert_map: ``(E,)`` 全局 → 本地映射，非本地为 -1。
        local_num_experts: 本 rank 专家数。

    Returns:
        :class:`LocalPlan`。
    """
    local_ids = expert_map[expert_ids]
    if (local_ids < 0).any():
        raise ValueError("bundle 中存在不属于本 rank 的专家条目")

    num_entries = expert_ids.numel()
    if num_entries > 0:
        order = torch.argsort(local_ids, stable=True)
        inv_order = torch.empty_like(order)
        inv_order[order] = torch.arange(num_entries, device=order.device)
    else:
        order = torch.empty(0, dtype=torch.long)
        inv_order = torch.empty(0, dtype=torch.long)

    counts = torch.bincount(local_ids, minlength=local_num_experts).to(torch.long)
    start_loc = torch.zeros(local_num_experts + 1, dtype=torch.long)
    start_loc[1:] = torch.cumsum(counts, dim=0)

    return LocalPlan(
        order=order,
        inv_order=inv_order,
        local_expert_ids=local_ids,
        expert_counts=counts,
        expert_start_loc=start_loc,
    )
