"""单机模拟冒烟：构造小尺寸 RouteEP Runner，跑一轮前向并打印形状/误差。

用法（装好 torch 后）::

    python lesson/routeep/examples/smoke_sim.py

单 rank 恒等后端 + eager 专家路径，不依赖 CUDA/NCCL。
"""

import pathlib
import sys

import torch

# 允许直接脚本运行时导入 routeep。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from routeep.alignment import compare_tensor  # noqa: E402
from routeep.config import RouteEPConfig  # noqa: E402
from routeep.experts.mlp import grouped_expert_forward_eager  # noqa: E402
from routeep.metadata import build_dispatch_metadata, combine_outputs  # noqa: E402
from routeep.moe_runner import build_default_runner  # noqa: E402
from routeep.router import route  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    config = RouteEPConfig(
        num_experts=4,
        top_k=2,
        hidden_size=8,
        intermediate_size=6,
    )
    runner = build_default_runner(config)

    x = torch.randn(7, config.hidden_size)
    out = runner(x)

    # eager 参考对齐。
    gate = runner.router.gate_weight.detach()
    w1 = runner.experts.w1.detach()
    w2 = runner.experts.w2.detach()
    logits = x @ gate.t()
    weights, ids = route(logits, top_k=config.top_k, renormalize=True)
    meta = build_dispatch_metadata(weights, ids, config.num_experts)
    permuted = x[meta.sorted_token_ids]
    expert_out = grouped_expert_forward_eager(permuted, meta.expert_start_loc, w1, w2)
    ref = combine_outputs(expert_out, meta)

    diff = compare_tensor(ref, out)
    print(f"output shape: {tuple(out.shape)}")
    print(f"max_abs={diff.max_abs:.3e} cosine={diff.cosine:.6f} "
          f"mismatch={diff.mismatch_frac:.3e}")
    assert diff.mismatch_frac == 0.0, "Runner 输出与 eager 参考不一致"
    print("OK: 单机模拟前向与 eager 参考对齐")


if __name__ == "__main__":
    main()
