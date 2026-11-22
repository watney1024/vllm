"""L6：Dispatch/Combine 的 gather 与加权 scatter-add 算子。

- ``gather_tokens``：按 ``token_ids`` 把原 token 拷贝到 dispatch 顺序（scatter/gather）；
- ``combine_scatter_add``：把专家输出乘路由权重后，按原 token 下标累加回输出。

CUDA + Triton 可用时走 Triton kernel；否则回退 PyTorch 索引/``index_add_``，
两者数值语义一致，便于在 CPU 上先验证逻辑。
"""

import torch


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


def gather_tokens(x: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """按下标 gather：``out[i] = x[token_ids[i]]``。

    Args:
        x: ``(T, H)``。
        token_ids: ``(M,)`` 原 token 下标。

    Returns:
        ``(M, H)``。
    """
    if token_ids.numel() == 0:
        return x.new_empty((0, x.shape[1]))
    if x.is_cuda and _has_triton():
        return _gather_triton(x, token_ids)
    return x[token_ids]


def combine_scatter_add(
    outputs: torch.Tensor,
    token_ids: torch.Tensor,
    weights: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """加权 scatter-add：``out[t] += Σ weight[i] · outputs[i]``。

    Args:
        outputs: ``(M, H)`` 专家输出。
        token_ids: ``(M,)`` 原 token 下标。
        weights: ``(M,)`` 路由权重。
        num_tokens: 输出 token 数 T。

    Returns:
        ``(T, H)`` 累加结果。
    """
    H = outputs.shape[1]
    out = torch.zeros(num_tokens, H, dtype=outputs.dtype, device=outputs.device)
    if token_ids.numel() == 0:
        return out
    if outputs.is_cuda and _has_triton():
        _scatter_add_triton(out, outputs, token_ids, weights)
    else:
        weighted = outputs * weights.to(outputs.dtype).unsqueeze(-1)
        out.index_add_(0, token_ids, weighted)
    return out


def _gather_triton(x: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    import triton
    import triton.language as tl

    M, H = x.shape
    BLOCK_H = triton.next_power_of_2(H)
    out = torch.empty_like(x, shape=(M, H))

    @triton.jit
    def _kernel(X, token_ids, Out, H, BLOCK_H: tl.constexpr):
        pid_m = tl.program_id(0)
        src = tl.load(token_ids + pid_m)
        offs_h = tl.arange(0, BLOCK_H)
        mask = offs_h < H
        vals = tl.load(X + src * H + offs_h, mask=mask)
        tl.store(Out + pid_m * H + offs_h, vals, mask=mask)

    _kernel[(M,)](x, token_ids, out, H, BLOCK_H=BLOCK_H)
    return out


def _scatter_add_triton(
    out: torch.Tensor,
    outputs: torch.Tensor,
    token_ids: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    import triton
    import triton.language as tl

    M, H = outputs.shape
    BLOCK_H = triton.next_power_of_2(H)

    @triton.jit
    def _kernel(Out, outputs, token_ids, weights, H, BLOCK_H: tl.constexpr):
        pid_m = tl.program_id(0)
        dst = tl.load(token_ids + pid_m)
        w = tl.load(weights + pid_m)
        offs_h = tl.arange(0, BLOCK_H)
        mask = offs_h < H
        vals = tl.load(outputs + pid_m * H + offs_h, mask=mask) * w
        tl.atomic_add(Out + dst * H + offs_h, vals, mask=mask)

    _kernel[(M,)](out, outputs, token_ids, weights, H, BLOCK_H=BLOCK_H)
