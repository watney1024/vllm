"""RouteEP：从零手写的 MoE 专家并行后端（vLLM 实战课程）。

包根只暴露无 torch 依赖的配置类；router / metadata / comm / kernels / runner
在各自模块内惰性使用 torch，保证无 GPU 环境导入根包不报错。
"""

from routeep.config import RouteEPConfig

__all__ = ["RouteEPConfig"]
