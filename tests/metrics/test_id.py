import numpy as np
import pytest

from seqme.metrics import ID


def discriminator(sequences: list[str]) -> np.ndarray:
    lengths = [len(sequence) for sequence in sequences]
    return np.array(lengths)


def test_basic():
    metric = ID(predictor=discriminator, name="Sequence length", objective="maximize")

    assert metric.name == "Sequence length"
    assert metric.objective == "maximize"

    result = metric(["A", "AA", "AAA"])

    assert result.value == 2.0
    assert result.deviation == pytest.approx(0.5773502)
