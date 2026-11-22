"""torch.distributed（NCCL/Gloo）后端 All-to-All。

包含两种实现：

- 固定容量：各 rank 发送条目数相同，直接 ``all_to_all``；
- 变长：先 ``all_gather`` 同步各 rank 发送计数，padding 到最大条目数后
  ``all_to_all``，接收侧按计数裁掉 padding。

分布式运行时在方法内惰性初始化/引用，无 dist 环境构造对象不报错；
``ep_size == 1`` 时走恒等短路，不触发集合通信。
"""

import torch

from routeep.comm.base import All2AllBackend, CombineBundle, DispatchBundle
from routeep.comm.grouping import group_by_rank
from routeep.comm.placement import rank_of_experts


def _pad_to(chunk: torch.Tensor, max_count: int) -> torch.Tensor:
    if chunk.shape[0] == max_count:
        return chunk
    pad_shape = (max_count - chunk.shape[0], *chunk.shape[1:])
    pad = torch.zeros(pad_shape, dtype=chunk.dtype, device=chunk.device)
    return torch.cat([chunk, pad], dim=0)


def _all_to_all_variable(
    tensors: list[torch.Tensor],
    send_counts: list[int],
    group: object | None,
    ep_rank: int,
) -> tuple[list[torch.Tensor], list[int]]:
    """变长 all-to-all：多张张量按相同计数一起交换。

    Args:
        tensors: 每张张量第 0 维为条目维，长度一致。
        send_counts: 发往每个 rank 的条目数。
        group: 分布式进程组（CPU/Gloo 组用于计数与元数据）。
        ep_rank: 当前 rank。

    Returns:
        ``(received_tensors, recv_counts)``：接收侧按源 rank 顺序拼接的张量，
        以及来自每个 rank 的条目数。
    """
    import torch.distributed as dist

    world = dist.get_world_size(group)
    counts_t = torch.tensor(send_counts, dtype=torch.long)
    gathered = [torch.zeros(world, dtype=torch.long) for _ in range(world)]
    dist.all_gather(gathered, counts_t, group=group)
    # gathered[s] 是 rank s 发出的计数；本 rank 从 s 收到的是其第 ep_rank 项。
    recv_counts = [int(gathered[s][ep_rank].item()) for s in range(world)]
    max_count = max(max(send_counts, default=0), max(recv_counts, default=0))

    received: list[torch.Tensor] = []
    for tensor in tensors:
        chunks = list(tensor.split(send_counts, dim=0))
        padded = [_pad_to(chunk, max_count) for chunk in chunks]
        recv_chunks = [torch.zeros_like(padded[r]) for r in range(world)]
        dist.all_to_all(recv_chunks, padded, group=group)
        trimmed = [
            recv_chunks[r][: recv_counts[r]] for r in range(world)
        ]
        if trimmed:
            received.append(torch.cat(trimmed, dim=0))
        else:
            received.append(
                tensor.new_empty((0, *tensor.shape[1:]))
            )
    return received, recv_counts


class NcclAll2All(All2AllBackend):
    """基于 ``torch.distributed`` 的 EP all-to-all 后端。"""

    def __init__(
        self,
        ep_size: int,
        ep_rank: int,
        num_experts: int,
        strategy: str = "linear",
        group: object | None = None,
    ):
        super().__init__(ep_size, ep_rank, group)
        self.num_experts = num_experts
        self.strategy = strategy
        self._expert_rank = rank_of_experts(num_experts, ep_size, strategy)

    def dispatch(
        self, tokens: torch.Tensor, meta: object
    ) -> DispatchBundle:
        if self.ep_size == 1:
            n = meta.num_routed
            return DispatchBundle(
                tokens=tokens,
                token_ids=meta.sorted_token_ids,
                slot_ids=meta.sorted_slot_ids,
                expert_ids=meta.sorted_expert_ids,
                weights=meta.sorted_weights,
                src_ranks=torch.zeros(n, dtype=torch.long),
                recv_counts=[n],
            )

        target_ranks = self._expert_rank[meta.sorted_expert_ids]
        order, send_counts, _ = group_by_rank(target_ranks, self.ep_size)

        ordered_tokens = tokens[order]
        token_ids = meta.sorted_token_ids[order]
        slot_ids = meta.sorted_slot_ids[order]
        expert_ids = meta.sorted_expert_ids[order]
        weights = meta.sorted_weights[order]
        src_ranks = torch.full(
            (ordered_tokens.shape[0],), self.ep_rank, dtype=torch.long
        )

        received, recv_counts = _all_to_all_variable(
            [ordered_tokens, token_ids, slot_ids, expert_ids, weights, src_ranks],
            send_counts,
            self.group,
            self.ep_rank,
        )
        (
            recv_tokens,
            recv_token_ids,
            recv_slot_ids,
            recv_expert_ids,
            recv_weights,
            recv_src_ranks,
        ) = received

        return DispatchBundle(
            tokens=recv_tokens,
            token_ids=recv_token_ids,
            slot_ids=recv_slot_ids,
            expert_ids=recv_expert_ids,
            weights=recv_weights,
            src_ranks=recv_src_ranks,
            recv_counts=recv_counts,
        )

    def combine(
        self, bundle: DispatchBundle, outputs: torch.Tensor
    ) -> CombineBundle:
        if self.ep_size == 1:
            return CombineBundle(
                outputs=outputs,
                token_ids=bundle.token_ids,
                slot_ids=bundle.slot_ids,
                weights=bundle.weights,
            )

        order, send_counts, _ = group_by_rank(bundle.src_ranks, self.ep_size)
        received, _ = _all_to_all_variable(
            [
                outputs[order],
                bundle.token_ids[order],
                bundle.slot_ids[order],
                bundle.weights[order],
            ],
            send_counts,
            self.group,
            self.ep_rank,
        )
        recv_outputs, recv_token_ids, recv_slot_ids, recv_weights = received
        return CombineBundle(
            outputs=recv_outputs,
            token_ids=recv_token_ids,
            slot_ids=recv_slot_ids,
            weights=recv_weights,
        )
