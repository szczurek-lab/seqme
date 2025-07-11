import numpy as np
import pytest

from pepme.metrics import Diversity


def test_compute_metric():
    reference = ["AAA", "BBBB", "CCCCC"]
    metric = Diversity(reference=reference, reference_name="Random")

    # Name and objective
    assert metric.name == "Diversity (Random)"
    assert metric.objective == "maximize"

    # Compute on a sample set
    result = metric(["AA", "BB", "CCCCD"])

    # Manually compute expected min-distances
    minimum_distances = np.array([1, 2, 1])

    # Compare value and deviation
    assert result.value == pytest.approx(minimum_distances.mean())
    assert result.deviation == pytest.approx(minimum_distances.std())


def test_single_sequence():
    reference = ["AAA", "BBBB", "CCCCC"]
    metric = Diversity(reference=reference, reference_name="Random2")

    # Name and objective
    assert metric.name == "Diversity (Random2)"
    assert metric.objective == "maximize"

    # Only one input sequence → zero diversity
    result = metric(["AAA"])
    assert result.value == pytest.approx(0.0)
    assert result.deviation == pytest.approx(0.0)


def test_empty_reference_raises():
    # Must provide at least one reference sequence
    with pytest.raises(Exception, match=r"References must contain at least one sample\."):
        Diversity(reference=[])
