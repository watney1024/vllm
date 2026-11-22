"""L13：服务压测指标采集与稳定性统计。

提供请求级计时、聚合指标（Prefill/Decode 延迟、吞吐、失败率、P99）与
稳定性压测循环。GPU 上计时可改用 CUDA Event；本模块默认用
``perf_counter``，CPU/无设备环境也能跑通统计逻辑。
"""

import time
from dataclasses import dataclass, field

import torch


@dataclass
class BenchResult:
    """压测聚合结果。"""

    total_requests: int = 0
    completed: int = 0
    failed: int = 0
    tokens: int = 0
    prefill_latency_ms: list[float] = field(default_factory=list)
    decode_step_ms: list[float] = field(default_factory=list)

    def add_request(
        self,
        *,
        ok: bool,
        prefill_ms: float,
        decode_step_ms: list[float],
        tokens: int,
    ) -> None:
        self.total_requests += 1
        self.tokens += tokens
        if ok:
            self.completed += 1
            self.prefill_latency_ms.append(prefill_ms)
            self.decode_step_ms.extend(decode_step_ms)
        else:
            self.failed += 1

    @property
    def failure_rate(self) -> float:
        return self.failed / self.total_requests if self.total_requests else 0.0

    @property
    def avg_prefill_ms(self) -> float:
        vals = self.prefill_latency_ms
        return sum(vals) / len(vals) if vals else float("nan")

    @property
    def avg_decode_ms(self) -> float:
        vals = self.decode_step_ms
        return sum(vals) / len(vals) if vals else float("nan")

    def decode_p99_ms(self) -> float:
        if not self.decode_step_ms:
            return float("nan")
        ordered = sorted(self.decode_step_ms)
        idx = min(len(ordered) - 1, int(0.99 * len(ordered)))
        return ordered[idx]

    @property
    def decode_throughput_tok_s(self) -> float:
        """decode 阶段每 token 耗时换算的吞吐（步均毫秒 → token/s）。"""
        avg = self.avg_decode_ms
        if not avg or avg != avg:
            return float("nan")
        return 1000.0 / avg


def _perf_counter_ms(fn) -> tuple[float, object]:
    start = time.perf_counter()
    out = fn()
    elapsed = (time.perf_counter() - start) * 1000.0
    return elapsed, out


def time_prefill(forward_fn, hidden: torch.Tensor) -> tuple[float, torch.Tensor]:
    """计时一次 prefill 前向。"""
    return _perf_counter_ms(lambda: forward_fn(hidden))


def time_decode_steps(
    forward_fn, hidden: torch.Tensor, steps: int
) -> tuple[list[float], torch.Tensor]:
    """计时连续 decode 步，返回每步耗时（毫秒）与最后输出。"""
    step_ms: list[float] = []
    out = hidden
    for _ in range(steps):
        ms, out = _perf_counter_ms(lambda: forward_fn(out))
        step_ms.append(ms)
    return step_ms, out


def run_stability(
    run_request_fn,
    num_requests: int,
) -> BenchResult:
    """稳定性压测循环：逐条执行请求，异常计入失败。

    Args:
        run_request_fn: 无参可调用，返回
            ``(prefill_ms, decode_step_ms, tokens, ok)``。
        num_requests: 请求总数（教程基线 17038）。

    Returns:
        聚合后的 :class:`BenchResult`。
    """
    result = BenchResult()
    for _ in range(num_requests):
        try:
            prefill_ms, decode_steps, tokens, ok = run_request_fn()
            result.add_request(
                ok=ok,
                prefill_ms=prefill_ms,
                decode_step_ms=decode_steps,
                tokens=tokens,
            )
        except Exception:  # noqa: BLE001 - 压测需记录任意异常为失败
            result.add_request(
                ok=False, prefill_ms=0.0, decode_step_ms=[], tokens=0
            )
    return result


def format_report(result: BenchResult) -> str:
    """格式化压测报告文本。"""
    return (
        f"requests: {result.total_requests} | completed: {result.completed} | "
        f"failed: {result.failed} | failure_rate: {result.failure_rate:.4f}\n"
        f"prefill avg: {result.avg_prefill_ms:.2f} ms | "
        f"decode avg: {result.avg_decode_ms:.2f} ms | "
        f"decode P99: {result.decode_p99_ms():.2f} ms | "
        f"decode throughput: {result.decode_throughput_tok_s:.1f} tok/s"
    )
