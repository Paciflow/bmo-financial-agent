import pytest

from bmo_agent.agent.calculations import compute


def test_percent_change_increase():
    out = compute("percent_change", a=36274, b=32795)
    assert out["unit"] == "%"
    assert out["result"] == pytest.approx(10.61, abs=0.01)


def test_percent_change_decrease():
    out = compute("percent_change", a=90, b=100)
    assert out["result"] == pytest.approx(-10.0)


def test_percent_change_division_by_zero():
    out = compute("percent_change", a=10, b=0)
    assert out["result"] is None
    assert "error" in out


def test_percentage_point_change_is_simple_subtraction_not_relative():
    # 45% -> 50% is a 5 percentage-point change, NOT a 11.1% relative change.
    out = compute("percentage_point_change", a=50, b=45)
    assert out["result"] == 5
    assert "percentage points" in out["unit"]


def test_difference():
    assert compute("difference", a=10, b=3)["result"] == 7


def test_ratio():
    out = compute("ratio", a=10, b=4)
    assert out["result"] == 2.5


def test_sum():
    assert compute("sum", a=2, b=3)["result"] == 5


def test_unsupported_operation_raises():
    with pytest.raises(ValueError):
        compute("logarithm", a=1, b=2)
