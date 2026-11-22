"""L3：Token 重排、去重、Dispatch 元数据构造。

输入 Router 的输出 ``topk_weights (T, K)`` / ``topk_ids (T, K)``，产出：

1. 扁平路由表：每条目记录 *原 token 下标*、*槽位 k*、*全局专家 id*、*权重；
2. 无效条目过滤：padding 槽位的 ``expert_id == -1`` 直接丢弃；
3. 去重：同一 token 重复命中同一专家时，权重聚合为一条；
4. 按专家稳定排序：得到 dispatch 顺序、每个专家的条目数与前缀偏移；
5. 反向还原：combine 阶段按原 token 下标把专家输出加权累加回去。

本模块只依赖 torch 张量操作，不涉及通信；EP 分片（全局 id → 本地 rank）
在 Stage 2 通信层基于本表扩展。
"""

from dataclasses import dataclass

import torch


@dataclass
class DispatchMetadata:
    """dispatch/combine 所需的全部索引与计数表。

    排序约定：条目按 ``(expert_id, token_id)`` 升序稳定排列，同一专家的
    条目落在 ``[expert_start_loc[e], expert_start_loc[e+1])`` 区间。

    Attributes:
        num_tokens: 原 token 数 T。
        top_k: 路由槽位数 K。
        num_experts: 全局专家数 E。
        sorted_token_ids: ``(M,)`` 排序后条目的原 token 下标。
        sorted_slot_ids: ``(M,)`` 排序后条目的槽位号 k。
        sorted_expert_ids: ``(M,)`` 排序后条目的全局专家 id。
        sorted_weights: ``(M,)`` 排序后条目的路由权重。
        expert_counts: ``(E,)`` 每个专家的条目数。
        expert_start_loc: ``(E+1,)`` 专家条目前缀偏移。
        flat_token_ids: ``(M0,)`` 去重过滤前的扁平 token 下标（调试/通信用）。
        flat_slot_ids: ``(M0,)`` 扁平槽位号。
        flat_expert_ids: ``(M0,)`` 扁平专家 id。
        flat_weights: ``(M0,)`` 扁平权重。
    """

    num_tokens: int
    top_k: int
    num_experts: int
    sorted_token_ids: torch.Tensor
    sorted_slot_ids: torch.Tensor
    sorted_expert_ids: torch.Tensor
    sorted_weights: torch.Tensor
    expert_counts: torch.Tensor
    expert_start_loc: torch.Tensor
    flat_token_ids: torch.Tensor
    flat_slot_ids: torch.Tensor
    flat_expert_ids: torch.Tensor
    flat_weights: torch.Tensor

    @property
    def num_routed(self) -> int:
        """去重过滤后的实际路由条目数 M。"""
        return self.sorted_token_ids.numel()

    def expert_slice(self, expert_id: int) -> slice:
        """返回某专家条目在排序表中的切片区间。"""
        return slice(
            int(self.expert_start_loc[expert_id]),
            int(self.expert_start_loc[expert_id + 1]),
        )


def _filter_and_dedup(
    token_ids: torch.Tensor,
    slot_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    num_experts: int,
    *,
    drop_invalid: bool,
    dedup: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """过滤无效槽位并按 (token, expert) 去重聚合权重。"""
    valid = torch.ones_like(expert_ids, dtype=torch.bool)
    if drop_invalid:
        valid = expert_ids >= 0

    token_ids = token_ids[valid]
    slot_ids = slot_ids[valid]
    expert_ids = expert_ids[valid]
    weights = weights[valid]

    if not dedup or token_ids.numel() == 0:
        return token_ids, slot_ids, expert_ids, weights

    # 复合 key：同一 token 的同一专家视为同一条目。
    key = token_ids * num_experts + expert_ids
    order_key = torch.argsort(key, stable=True)
    sorted_key = key[order_key]
    is_new = torch.ones(sorted_key.shape[0], dtype=torch.bool, device=key.device)
    is_new[1:] = sorted_key[1:] != sorted_key[:-1]

    rep_pos = order_key[is_new]
    group_id = torch.cumsum(is_new, dim=0) - 1

    sorted_weights = weights[order_key]
    agg_weights = torch.zeros(
        int(is_new.sum()), dtype=weights.dtype, device=weights.device
    ).scatter_add(0, group_id, sorted_weights)

    rep_weights = agg_weights  # 聚合结果按代表条目顺序排列
    return (
        token_ids[rep_pos],
        slot_ids[rep_pos],
        expert_ids[rep_pos],
        rep_weights,
    )


def build_dispatch_metadata(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    *,
    drop_invalid: bool = True,
    dedup: bool = True,
) -> DispatchMetadata:
    """由 Router 输出构造 dispatch 元数据。

    Args:
        topk_weights: ``(T, K)`` 路由权重。
        topk_ids: ``(T, K)`` 全局专家 id；无效槽位用 -1 标记。
        num_experts: 全局专家数 E。
        drop_invalid: 是否丢弃 ``id < 0`` 的 padding 槽位。
        dedup: 是否按 (token, expert) 去重聚合。

    Returns:
        填充完成的 :class:`DispatchMetadata`。
    """
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            f"weights 形状 {topk_weights.shape} 与 ids 形状 {topk_ids.shape} 不一致"
        )
    num_tokens, top_k = topk_weights.shape
    if num_experts <= 0:
        raise ValueError(f"num_experts 必须为正整数，得到 {num_experts}")

    device = topk_weights.device
    # 扁平展开：条目顺序为 token 主序、槽位次序。
    token_ids = torch.arange(num_tokens, device=device).repeat_interleave(top_k)
    slot_ids = torch.arange(top_k, device=device).repeat(num_tokens)
    flat_expert_ids = topk_ids.reshape(-1)
    flat_weights = topk_weights.reshape(-1)

    token_ids, slot_ids, expert_ids, weights = _filter_and_dedup(
        token_ids,
        slot_ids,
        flat_expert_ids,
        flat_weights,
        num_experts,
        drop_invalid=drop_invalid,
        dedup=dedup,
    )

    # 按 (expert, token) 稳定排序，得到 dispatch 顺序。
    if token_ids.numel() > 0:
        sort_key = expert_ids * num_tokens + token_ids
        order = torch.argsort(sort_key, stable=True)
        sorted_token_ids = token_ids[order]
        sorted_slot_ids = slot_ids[order]
        sorted_expert_ids = expert_ids[order]
        sorted_weights = weights[order]
    else:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        sorted_token_ids = empty
        sorted_slot_ids = empty
        sorted_expert_ids = empty
        sorted_weights = torch.empty(0, dtype=flat_weights.dtype, device=device)

    expert_counts = torch.bincount(
        sorted_expert_ids, minlength=num_experts
    ).to(torch.int64)
    expert_start_loc = torch.zeros(
        num_experts + 1, dtype=torch.int64, device=device
    )
    expert_start_loc[1:] = torch.cumsum(expert_counts, dim=0)

    return DispatchMetadata(
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_experts,
        sorted_token_ids=sorted_token_ids,
        sorted_slot_ids=sorted_slot_ids,
        sorted_expert_ids=sorted_expert_ids,
        sorted_weights=sorted_weights,
        expert_counts=expert_counts,
        expert_start_loc=expert_start_loc,
        flat_token_ids=token_ids,
        flat_slot_ids=slot_ids,
        flat_expert_ids=expert_ids,
        flat_weights=weights,
    )


def permute_tokens(
    hidden_states: torch.Tensor, meta: DispatchMetadata
) -> torch.Tensor:
    """按 dispatch 顺序重排 token。

    Args:
        hidden_states: ``(T, H)``。
        meta: :class:`DispatchMetadata`。

    Returns:
        ``(M, H)``，第 i 行对应 ``meta.sorted_token_ids[i]`` 的原 token。
    """
    if hidden_states.shape[0] != meta.num_tokens:
        raise ValueError(
            f"hidden_states 行数 {hidden_states.shape[0]} 与元数据 token 数 "
            f"{meta.num_tokens} 不一致"
        )
    return hidden_states[meta.sorted_token_ids]


def combine_outputs(
    sorted_out: torch.Tensor,
    meta: DispatchMetadata,
    *,
    num_tokens: int | None = None,
) -> torch.Tensor:
    """combine：把专家输出按路由权重累加回原 token 顺序。

    Args:
        sorted_out: ``(M, H)``，与 ``meta`` 排序顺序一致的专家输出。
        meta: :class:`DispatchMetadata`。
        num_tokens: 输出 token 数；默认取 ``meta.num_tokens``。

    Returns:
        ``(T, H)`` 加权求和后的 routed 输出（不含 shared expert）。
    """
    if sorted_out.shape[0] != meta.num_routed:
        raise ValueError(
            f"sorted_out 行数 {sorted_out.shape[0]} 与元数据条目数 "
            f"{meta.num_routed} 不一致"
        )
    T = meta.num_tokens if num_tokens is None else num_tokens
    H = sorted_out.shape[1]
    out = torch.zeros(T, H, dtype=sorted_out.dtype, device=sorted_out.device)
    weighted = sorted_out * meta.sorted_weights.to(sorted_out.dtype).unsqueeze(-1)
    out.index_add_(0, meta.sorted_token_ids, weighted)
    return out
