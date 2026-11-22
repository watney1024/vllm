"""TP × DP × EP 拓扑描述与一致性校验。

并行关系（对齐 vLLM 并行配置口径）：

- ``tp_size``：张量并行组大小；
- ``dp_size``：数据并行组大小；``world_size = tp_size × dp_size``；
- ``ep_size``：专家并行组大小，在 DP 组内划分，``ep_size <= dp_size``；
- 专家数必须能被 ``ep_size`` 整除。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelTopology:
    tp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1

    @property
    def world_size(self) -> int:
        """总进程数 = TP × DP。"""
        return self.tp_size * self.dp_size

    def validate(self, num_experts: int) -> None:
        """校验拓扑与专家数的一致性。"""
        if self.tp_size < 1 or self.dp_size < 1 or self.ep_size < 1:
            raise ValueError("并行度必须 >= 1")
        if self.ep_size > self.dp_size:
            raise ValueError(
                f"ep_size ({self.ep_size}) 不能大于 dp_size ({self.dp_size})"
            )
        if num_experts % self.ep_size != 0:
            raise ValueError(
                f"专家数 {num_experts} 不能被 ep_size {self.ep_size} 整除"
            )

    def ep_group_dp_ranks(self, dp_rank: int) -> list[int]:
        """返回某 DP rank 所在 EP 组的 DP rank 列表。

        EP 组在 DP 维内按 ``ep_size`` 切分：每 ``ep_size`` 个 DP rank 共享
        同一组专家。映射到全局 rank 时再叠加 TP 偏移。
        """
        base = (dp_rank // self.ep_size) * self.ep_size
        return [base + i for i in range(self.ep_size)]
