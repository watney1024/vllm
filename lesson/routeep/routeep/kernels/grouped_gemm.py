"""L7：Grouped GEMM + SwiGLU 融合算子。

分组 GEMM：不同专家的 token 数不同，kernel 按专家分段，每段独立做矩阵乘。
本模块提供：

- eager 参考实现（数值对齐基线）；
- Triton 分组 GEMM kernel（CUDA 可用时启用）；
- 两段式前向：GEMM1 出 gate/up → SwiGLU 融合激活 → GEMM2 出专家输出。

权重布局与 `routeep/experts/mlp.py` 一致：
``w1 (E, 2I, H)``、``w2 (E, H, I)``。
"""

import torch
import torch.nn.functional as F


def grouped_expert_forward_triton(
    x: torch.Tensor,
    expert_start_loc: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """分组 SwiGLU 前向：Triton GEMM + 融合激活。

    Args:
        x: ``(M, H)`` 按专家排序后的 token。
        expert_start_loc: ``(E+1,)`` 前缀偏移。
        w1: ``(E, 2I, H)``。
        w2: ``(E, H, I)``。

    Returns:
        ``(M, H)`` 专家输出。
    """
    intermediate = w2.shape[2]
    gate_up = _grouped_gemm(x, w1, expert_start_loc)
    gate, up = gate_up[:, :intermediate], gate_up[:, intermediate:]
    hidden = F.silu(gate) * up
    return _grouped_gemm(hidden, w2, expert_start_loc)


def _grouped_gemm_eager(
    A: torch.Tensor, W: torch.Tensor, start_loc: torch.Tensor
) -> torch.Tensor:
    """逐专家 ``C[e] = A[e] @ W[e]ᵀ``（eager 基线）。"""
    num_experts = W.shape[0]
    N = W.shape[1]
    C = torch.zeros(A.shape[0], N, dtype=A.dtype, device=A.device)
    for e in range(num_experts):
        s = int(start_loc[e])
        t = int(start_loc[e + 1])
        if t > s:
            C[s:t] = A[s:t] @ W[e].t()
    return C


def _grouped_gemm(
    A: torch.Tensor, W: torch.Tensor, start_loc: torch.Tensor
) -> torch.Tensor:
    """单段分组 GEMM，按设备可用性选择 Triton / eager。"""
    if A.is_cuda and _has_triton():
        return _grouped_gemm_triton(A, W, start_loc)
    return _grouped_gemm_eager(A, W, start_loc)


def _has_triton() -> bool:
    try:
        import triton  # noqa: F401

        return True
    except ImportError:
        return False


def _grouped_gemm_triton(
    A: torch.Tensor, W: torch.Tensor, start_loc: torch.Tensor
) -> torch.Tensor:
    import triton
    import triton.language as tl

    M, K = A.shape
    E, N, _ = W.shape
    C = torch.zeros(M, N, dtype=A.dtype, device=A.device)
    if M == 0:
        return C

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    @triton.jit
    def _kernel(
        A,
        W,
        C,
        start_loc,
        N,
        K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_e = tl.program_id(0)
        pid_m = tl.program_id(1)

        s = tl.load(start_loc + pid_e)
        e = tl.load(start_loc + pid_e + 1)
        rows = s + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < e
        if not tl.any(row_mask):
            return

        n_offs = tl.arange(0, BLOCK_N)
        n_mask = n_offs < N
        k_offs = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + k_offs
            k_mask = k < K
            a = tl.load(
                A + rows[:, None] * K + k[None, :],
                mask=row_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            # W[e] 行优先 (N, K)；B[k, n] = W[e][n, k]，再转置给 tl.dot。
            w = tl.load(
                W + pid_e * N * K + n_offs[:, None] * K + k[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            acc += tl.dot(a, w.T)

        c_mask = row_mask[:, None] & n_mask[None, :]
        tl.store(
            C + rows[:, None] * N + n_offs[None, :],
            acc.to(C.dtype.element_type),
            mask=c_mask,
        )

    grid = (E, triton.cdiv(int(start_loc[-1].item()), BLOCK_M))
    _kernel[grid](A, W, C, start_loc, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
    return C
