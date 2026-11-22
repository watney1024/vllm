"""逐层对齐工具单测。"""

import pytest

torch = pytest.importorskip("torch")

from routeep.alignment import compare_tensor, format_report, layerwise_compare  # noqa: E402


def test_identical_tensor_is_close() -> None:
    x = torch.randn(4, 8)
    diff = compare_tensor(x, x.clone())
    assert diff.max_abs == 0.0
    assert diff.cosine == 1.0
    assert diff.mismatch_frac == 0.0
    assert diff.num_nan == 0


def test_known_difference() -> None:
    ref = torch.zeros(3, 4)
    out = torch.ones(3, 4)
    diff = compare_tensor(ref, out, atol=1e-5)
    assert diff.max_abs == 1.0
    assert diff.mean_abs == 1.0
    assert diff.mismatch_frac == 1.0


def test_cosine_orthogonal() -> None:
    ref = torch.tensor([1.0, 0.0])
    out = torch.tensor([0.0, 1.0])
    diff = compare_tensor(ref, out)
    assert abs(diff.cosine) < 1e-6


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        compare_tensor(torch.randn(3, 4), torch.randn(3, 5))


def test_layerwise_compare_and_report() -> None:
    refs = {"h1": torch.randn(2, 4), "h2": torch.randn(2, 4)}
    outs = {"h1": refs["h1"].clone(), "h2": refs["h2"] * 1.01}
    diffs = layerwise_compare(refs, outs, atol=1e-3)
    assert set(diffs) == {"h1", "h2"}
    report = format_report(diffs)
    assert "h1" in report and "cosine" in report
