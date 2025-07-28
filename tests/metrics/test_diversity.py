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
    assert result.value == pytest.approx(4 / 3)
    assert result.deviation is None


def test_single_sequence():
    metric = Diversity()

    # Name and objective
    assert metric.name == "Diversity"
    assert metric.objective == "maximize"

    # Only one input sequence → zero diversity
    with pytest.raises(ValueError, match=r"^Expected at least 2 sequences.$"):
        metric(["AAA"])
