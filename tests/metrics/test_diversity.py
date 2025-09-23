import pytest

from seqme.metrics import Diversity


def test_multiple_sequences():
    metric = Diversity()

    # Name and objective
    assert metric.name == "Diversity"
    assert metric.objective == "maximize"

    # Compute on a sample set
    result = metric(["AA", "BB", "CCCCD"])

    # Compare value and deviation
    assert result.value == 1.0
    assert result.deviation is None


def test_reference_with_min():
    metric = Diversity(reference=["AB", "BA", "CCCC"], k=2, seed=42, aggregate="min")

    # Name and objective
    assert metric.name == "Diversity"
    assert metric.objective == "maximize"

    # Compute on a sample set
    result = metric(["AA"])

    # Compare value and deviation
    assert result.value == 0.5
    assert result.deviation is None
