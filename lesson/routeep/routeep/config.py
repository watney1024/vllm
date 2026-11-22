"""RouteEP 配置：MoE 结构参数与 EP 拓扑参数。

只存放纯 Python 标量/字符串，避免在无 GPU 环境导入本包时依赖 torch。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RouteEPConfig:
    """单个 MoE 层的路由与并行配置。

    Attributes:
        num_experts: 全局专家数 E。
        top_k: 每个 token 选中的专家数 K。
        hidden_size: 输入/输出隐藏维 H。
        intermediate_size: 专家 MLP 中间维（SwiGLU 的 hidden）。
        scoring_func: 打分函数，``"softmax"`` 或 ``"sigmoid"``。
        renormalize: TopK 选中后是否把权重重新归一化为和为 1。
        routed_scaling_factor: 路由权重的后置缩放因子（对齐 vLLM 同名参数）。
        ep_size: EP 组大小；1 表示不做专家并行（单 rank 持有全部专家）。
        dtype_name: 计算精度名，``"float32"`` / ``"bfloat16"`` / ``"float16"``。
    """

    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    scoring_func: str = "softmax"
    renormalize: bool = True
    routed_scaling_factor: float = 1.0
    ep_size: int = 1
    dtype_name: str = "float32"

    def __post_init__(self) -> None:
        if self.top_k > self.num_experts:
            raise ValueError(
                f"top_k ({self.top_k}) 不能大于 num_experts ({self.num_experts})"
            )
        if self.scoring_func not in ("softmax", "sigmoid"):
            raise ValueError(
                f"不支持的 scoring_func: {self.scoring_func!r}，"
                '可选 "softmax" / "sigmoid"'
            )
        if self.ep_size < 1:
            raise ValueError(f"ep_size 必须 >= 1，得到 {self.ep_size}")
        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) 必须能被 ep_size "
                f"({self.ep_size}) 整除"
            )

    @property
    def local_num_experts(self) -> int:
        """本 rank 持有的专家数（EP 分片后）。"""
        return self.num_experts // self.ep_size
