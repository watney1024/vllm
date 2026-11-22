"""L11：CUDA Graph 分桶捕获与回放。

decode 阶段每步 token 数变化，但相邻步容量接近；按 token 容量分桶，
每个桶独立 warmup + capture，使用静态输入/输出缓冲回放，避免每步重新
捕获 Graph。容量超出当前桶时落到更大的桶，规避动态 shape 触发重捕获。

无 CUDA 环境自动回退 eager 前向，对象构造不报错。
"""

from bisect import bisect_left

import torch


class GraphNotCapturedError(RuntimeError):
    """请求回放尚未捕获的桶。"""


class BucketedGraphManager:
    """按容量分桶的 CUDA Graph 管理器。

    Args:
        bucket_capacities: 各桶 token 容量上界（升序），如
            ``[1, 8, 32, 128, 512]``；最后一桶视为无上限。
        device: 设备名，仅用于记录。
    """

    def __init__(self, bucket_capacities: list[int], device: str = "cuda"):
        self.capacities = sorted(bucket_capacities)
        self.device = device
        self._graphs: dict[int, object] = {}
        self._static_input: dict[int, torch.Tensor] = {}
        self._static_output: dict[int, torch.Tensor] = {}

    @staticmethod
    def is_available() -> bool:
        """CUDA Graph 是否可用。"""
        return torch.cuda.is_available()

    def bucket_id(self, num_tokens: int) -> int:
        """根据 token 数定位桶：第一个容量 >= num_tokens 的桶。"""
        idx = bisect_left(self.capacities, num_tokens)
        return min(idx, len(self.capacities) - 1)

    def capture(
        self,
        bucket_id: int,
        sample_input: torch.Tensor,
        forward_fn,
        warmup_runs: int = 2,
    ) -> None:
        """warmup 后捕获指定桶的 Graph。

        Args:
            bucket_id: 桶编号。
            sample_input: 该桶容量形状的示例输入（第 0 维 = 桶容量）。
            forward_fn: 单参数前向函数，``out = forward_fn(in)``。
            warmup_runs: 捕获前 eager 预热次数。
        """
        for _ in range(warmup_runs):
            forward_fn(sample_input)
        torch.cuda.synchronize()

        static_in = sample_input.detach().clone()
        static_in.requires_grad_(False)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = forward_fn(static_in)

        self._graphs[bucket_id] = graph
        self._static_input[bucket_id] = static_in
        self._static_output[bucket_id] = static_out

    def replay(self, bucket_id: int, inputs: torch.Tensor) -> torch.Tensor:
        """把当前输入拷进静态缓冲并回放 Graph。"""
        if bucket_id not in self._graphs:
            raise GraphNotCapturedError(f"桶 {bucket_id} 尚未捕获 Graph")

        static_in = self._static_input[bucket_id]
        num_tokens = inputs.shape[0]
        static_in.zero_()
        static_in[:num_tokens].copy_(inputs)
        self._graphs[bucket_id].replay()
        return self._static_output[bucket_id][:num_tokens].clone()

    def run(self, inputs: torch.Tensor, forward_fn) -> torch.Tensor:
        """按输入容量选桶：已捕获则回放，否则 eager 执行。"""
        if not self.is_available():
            return forward_fn(inputs)

        bucket = self.bucket_id(inputs.shape[0])
        if bucket not in self._graphs:
            return forward_fn(inputs)
        return self.replay(bucket, inputs)
