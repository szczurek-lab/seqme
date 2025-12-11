import numpy as np
import pytest

from seqme.metrics import Threshold


def discriminator(sequences: list[str]) -> np.ndarray:
    lengths = [len(sequence) for sequence in sequences]
    return np.array(lengths)


def test_min_value():
    metric = Threshold(predictor=discriminator, name="Sequence length", min_value=2)

    assert metric.name == "Sequence length"
    assert metric.objective == "maximize"

    result = metric(["A", "AA", "AAAA"])

    assert result.value == pytest.approx(2 / 3)
    assert result.deviation is None


def test_max_value():
    metric = Threshold(predictor=discriminator, name="Sequence length2", max_value=2)

    assert metric.name == "Sequence length2"
    assert metric.objective == "maximize"

    result = metric(["A", "AA", "AAAA"])

    assert result.value == pytest.approx(2 / 3)
    assert result.deviation is None


def test_between():
    metric = Threshold(predictor=discriminator, name="Sequence length2", min_value=1, max_value=1)

    assert metric.name == "Sequence length2"
    assert metric.objective == "maximize"

    result = metric(["A", "AA", "AAAA"])

    assert result.value == pytest.approx(1 / 3)
    assert result.deviation is None
