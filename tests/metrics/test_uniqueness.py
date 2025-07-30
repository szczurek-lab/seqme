import pytest

from seqme.metrics import Uniqueness


def test_name_and_objective():
    metric = Uniqueness()
    assert metric.name == "Uniqueness"
    assert metric.objective == "maximize"


def test_compute_metric():
    metric = Uniqueness()

    result = metric(["A", "B", "A", "C"])
    assert result.value == 0.75
    assert result.deviation is None

    result = metric(["X", "X", "X"])
    assert result.value == pytest.approx(1 / 3)
    assert result.deviation is None


def test_empty_sequences():
    metric = Uniqueness()
    result = metric([])
    assert result.value == 0.0
    assert result.deviation is None
