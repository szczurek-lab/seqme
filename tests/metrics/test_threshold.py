import numpy as np
import pytest

from seqme.metrics import Threshold


def discriminator(sequences: list[str]) -> np.ndarray:
    lengths = [len(sequence) for sequence in sequences]
    return np.array(lengths)


def test_above_threshold_inclusive():
    metric = Threshold(
        predictor=discriminator,
        name="Sequence length",
        threshold=2,
    )

    assert metric.name == "Sequence length"
    assert metric.objective == "maximize"

    result = metric(["A", "AA", "AAAA"])

    assert result.value == pytest.approx(2 / 3)
    assert result.deviation is None


def test_above_threshold_exclusive():
    metric = Threshold(
        predictor=discriminator,
        name="Sequence length",
        threshold=2,
        inclusive=False,
    )

    assert metric.name == "Sequence length"
    assert metric.objective == "maximize"

    result = metric(["A", "AA", "AAAA"])

    assert result.value == pytest.approx(1 / 3)
    assert result.deviation is None


def test_below_threshold_inclusive():
    metric = Threshold(
        predictor=discriminator,
        name="Sequence length2",
        threshold=2,
        inclusive=False,
        objective="minimize",
    )

    assert metric.name == "Sequence length2"
    assert metric.objective == "minimize"

    result = metric(["A", "AA", "AAAA"])

    assert result.value == pytest.approx(1 / 3)
    assert result.deviation is None
