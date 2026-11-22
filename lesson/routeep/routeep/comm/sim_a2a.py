"""单机模拟后端：无分布式运行时的恒等/调试实现。

- ``IdentityAll2All``：``ep_size == 1`` 的恒等后端，dispatch/combine 都不做
  跨进程交换，用于 CPU 单测与单 rank 前向验证；
- 多 rank 模拟交换需要共享注册表，当前不在此实现；多 rank 变长交换逻辑
  以 :class:`NcclAll2All` 为准，在 GPU 集群上用 ``torchrun`` 验证。
"""

import torch

from routeep.comm.base import All2AllBackend, CombineBundle, DispatchBundle


class IdentityAll2All(All2AllBackend):
    """单 rank 恒等 all-to-all：条目原样返回。"""

    def dispatch(
        self, tokens: torch.Tensor, meta: object
    ) -> DispatchBundle:
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

    def combine(
        self, bundle: DispatchBundle, outputs: torch.Tensor
    ) -> CombineBundle:
        return CombineBundle(
            outputs=outputs,
            token_ids=bundle.token_ids,
            slot_ids=bundle.slot_ids,
            weights=bundle.weights,
        )
