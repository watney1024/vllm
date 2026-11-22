"""L12：逐层数值对齐工具。

固定输入下，把 RouteEP 各层 hidden state 与参考实现（eager / vLLM 原生 /
DeepEP）逐张量对比，输出最大/平均绝对误差、余弦相似度、不一致比例与
NaN 计数，用于定位路由、通信、GEMM 引入的差异。
"""

from dataclasses import dataclass

import torch


@dataclass
class TensorDiff:
    """单张量对齐结果。"""

    max_abs: float
    mean_abs: float
    cosine: float
    mismatch_frac: float
    num_nan: int
    num_elements: int

    def is_close(self, atol: float, rtol: float) -> bool:
        """按 ``atol + rtol·|ref|`` 口径判断是否全部一致。"""
        return self.mismatch_frac == 0.0


def compare_tensor(
    ref: torch.Tensor,
    out: torch.Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> TensorDiff:
    """对比两个同形状张量。

    Args:
        ref: 参考张量。
        out: 待验证张量。
        atol: 绝对误差容限。
        rtol: 相对误差容限。

    Returns:
        :class:`TensorDiff`。
    """
    if ref.shape != out.shape:
        raise ValueError(f"形状不一致：{ref.shape} vs {out.shape}")

    ref_f = ref.detach().float().flatten()
    out_f = out.detach().float().flatten()

    abs_diff = (ref_f - out_f).abs()
    denom = atol + rtol * ref_f.abs()
    mismatch = abs_diff > denom

    num_nan = int(torch.isnan(out_f).sum().item())
    num_elements = ref_f.numel()

    if num_elements == 0:
        return TensorDiff(0.0, 0.0, 1.0, 0.0, 0, 0)

    cosine = _cosine(ref_f, out_f)
    return TensorDiff(
        max_abs=float(abs_diff.max().item()),
        mean_abs=float(abs_diff.mean().item()),
        cosine=float(cosine),
        mismatch_frac=float(mismatch.float().mean().item()),
        num_nan=num_nan,
        num_elements=num_elements,
    )


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float(torch.dot(a, b).item() / denom.item())


def layerwise_compare(
    refs: dict[str, torch.Tensor],
    outs: dict[str, torch.Tensor],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, TensorDiff]:
    """逐层对比同名张量。

    Args:
        refs: 参考层 ``{name: tensor}``。
        outs: 待验证层 ``{name: tensor}``。
        atol / rtol: 误差容限。

    Returns:
        ``{name: TensorDiff}``；两边缺名的层跳过并在结果中标记。
    """
    result: dict[str, TensorDiff] = {}
    for name in refs:
        if name not in outs:
            continue
        result[name] = compare_tensor(refs[name], outs[name], atol=atol, rtol=rtol)
    return result


def format_report(diffs: dict[str, TensorDiff]) -> str:
    """把对齐结果格式化成可读表格文本。"""
    lines = [
        f"{'layer':<28} {'max_abs':>10} {'mean_abs':>10} {'cosine':>8} "
        f"{'mismatch':>10} {'nan':>5}"
    ]
    for name, d in diffs.items():
        lines.append(
            f"{name:<28} {d.max_abs:>10.3e} {d.mean_abs:>10.3e} "
            f"{d.cosine:>8.5f} {d.mismatch_frac:>10.3e} {d.num_nan:>5}"
        )
    return "\n".join(lines)
