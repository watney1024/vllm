"""All-to-All 通信层抽象接口与数据结构。

对齐 vLLM ``All2AllManagerBase``（`base_device_communicator.py:34`）的两段式：

- ``dispatch``：把按专家重排后的 token 发到持有目标专家的 rank；
- ``combine``：把专家输出发回来源 rank，供上层按原 token 位置加权累加。

本模块不直接依赖 torch.distributed；具体后端（NCCL / 模拟）在子类实现。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class DispatchBundle:
    """dispatch 之后本 rank 收到的条目与 token。

    Attributes:
        tokens: ``(M, H)`` 收到的隐状态。
        token_ids: ``(M,)`` 条目在源 rank 上的原 token 下标。
        slot_ids: ``(M,)`` 路由槽位号。
        expert_ids: ``(M,)`` 全局专家 id。
        weights: ``(M,)`` 路由权重。
        src_ranks: ``(M,)`` 条目来源 rank。
        recv_counts: 来自每个 rank 的条目数，长度 ``ep_size``。
    """

    tokens: torch.Tensor
    token_ids: torch.Tensor
    slot_ids: torch.Tensor
    expert_ids: torch.Tensor
    weights: torch.Tensor
    src_ranks: torch.Tensor
    recv_counts: list[int]


@dataclass
class CombineBundle:
    """combine 回到源 rank 后的输出与溯源信息。

    Attributes:
        outputs: ``(M, H)`` 专家输出。
        token_ids: ``(M,)`` 原 token 下标。
        slot_ids: ``(M,)`` 路由槽位号。
        weights: ``(M,)`` 路由权重。
    """

    outputs: torch.Tensor
    token_ids: torch.Tensor
    slot_ids: torch.Tensor
    weights: torch.Tensor


class All2AllBackend(ABC):
    """EP all-to-all 后端抽象基类。"""

    def __init__(self, ep_size: int, ep_rank: int, group: object | None = None):
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.group = group

    @abstractmethod
    def dispatch(
        self, tokens: torch.Tensor, meta: object
    ) -> DispatchBundle:
        """把本地路由后的 token 发到目标专家所在 rank。

        Args:
            tokens: ``(M, H)``，与 ``meta`` 排序顺序一致的 token。
            meta: :class:`routeep.metadata.DispatchMetadata`。

        Returns:
            :class:`DispatchBundle`，本 rank 收到的条目。
        """

    @abstractmethod
    def combine(
        self, bundle: DispatchBundle, outputs: torch.Tensor
    ) -> CombineBundle:
        """把专家输出发回来源 rank。

        Args:
            bundle: dispatch 阶段收到的 :class:`DispatchBundle`。
            outputs: ``(M, H)``，与 bundle 顺序一致的专家输出。

        Returns:
            :class:`CombineBundle`，源 rank 上可直接累加的输出。
        """
