"""压测聚合与稳定性循环单测。"""

import pytest

from routeep.bench import BenchResult, format_report, run_stability


def test_bench_result_aggregation() -> None:
    result = BenchResult()
    result.add_request(ok=True, prefill_ms=10.0, decode_step_ms=[2.0, 2.0], tokens=8)
    result.add_request(ok=True, prefill_ms=12.0, decode_step_ms=[3.0], tokens=4)
    result.add_request(ok=False, prefill_ms=0.0, decode_step_ms=[], tokens=0)

    assert result.total_requests == 3
    assert result.completed == 2
    assert result.failed == 1
    assert abs(result.failure_rate - 1 / 3) < 1e-9
    assert abs(result.avg_prefill_ms - 11.0) < 1e-9
    assert result.tokens == 12
    assert result.decode_step_ms == [2.0, 2.0, 3.0]


def test_p99_decode() -> None:
    result = BenchResult()
    for ms in [1.0, 1.0, 1.0, 5.0]:
        result.add_request(ok=True, prefill_ms=0.0, decode_step_ms=[ms], tokens=1)
    assert result.decode_p99_ms() == 5.0


def test_run_stability_counts_exceptions_as_failure() -> None:
    calls = iter([(10.0, [2.0], 4, True), RuntimeError("boom")])

    def run_request():
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    result = run_stability(run_request, num_requests=2)
    assert result.total_requests == 2
    assert result.completed == 1
    assert result.failed == 1


def test_format_report_contains_metrics() -> None:
    result = BenchResult()
    result.add_request(ok=True, prefill_ms=10.0, decode_step_ms=[2.0], tokens=4)
    report = format_report(result)
    assert "failure_rate" in report
    assert "decode" in report
