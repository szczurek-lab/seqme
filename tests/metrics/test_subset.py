import pytest

from seqme.metrics import Count, Length, Subset


def test_n_samples():
    metric = Subset(metric=Count(), n_samples=2)
    assert metric.name == "Count"
    assert metric.objective == "maximize"

    sequences = ["AA", "AAA", "AAAA", "AAA", "AAAAAA"]
    result = metric(sequences)
    assert result.value == 2
    assert result.deviation is None


def test_uses_fixed_seed():
    metric = Subset(metric=Length(), n_samples=2)
    assert metric.name == "Length"
    assert metric.objective == "minimize"

    sequences = ["A", "A" * 10, "A" * 100, "A" * 1000]
    result1 = metric(sequences)
    assert result1.value == 550
    assert result1.deviation == 450

    result2 = metric(sequences)
    assert result2.value == 550
    assert result2.deviation == 450


def test_too_few_samples():
    metric = Subset(metric=Length(), n_samples=10)
    assert metric.name == "Length"
    assert metric.objective == "minimize"

    sequences = ["A"]
    with pytest.raises(ValueError):
        metric(sequences)
