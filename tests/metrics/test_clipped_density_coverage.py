import numpy as np
import pytest

from seqme.metrics import ClippedCoverage, ClippedDensity


def test_density():
    reference = ["A" * 1, "A" * 2, "A" * 3]
    metric = ClippedDensity(
        n_neighbors=1,
        reference=reference,
        embedder=length_mock_embedder,
        batch_size=128,
        strict=False,
    )

    assert metric.name == "Clipped density"
    assert metric.objective == "maximize"

    result = metric(["A" * 2, "A" * 4, "A" * 16])
    assert result.value == pytest.approx(2 / 3)
    assert result.deviation is None


def test_clipped_density():
    reference = ["A" * 1, "A" * 2, "A" * 3, "A" * 10, "A" * 11, "A" * 100]
    metric = ClippedDensity(
        n_neighbors=2,
        reference=reference,
        embedder=length_mock_embedder,
        batch_size=128,
        strict=False,
    )

    assert metric.name == "Clipped density"
    assert metric.objective == "maximize"

    result = metric(["A" * 2, "A" * 12, "A" * 80])
    assert result.value == 1.0
    assert result.deviation is None


def test_coverage():
    reference = ["A" * 1, "A" * 2, "A" * 3]
    metric = ClippedCoverage(
        n_neighbors=1,
        reference=reference,
        embedder=length_mock_embedder,
        batch_size=128,
        strict=False,
    )

    assert metric.name == "Clipped coverage"
    assert metric.objective == "maximize"

    result = metric(["A" * 2, "A" * 4, "A" * 16])
    assert result.value == 1.0
    assert result.deviation is None


def test_clipped_coverage():
    reference = ["A" * 1, "A" * 2, "A" * 3, "A" * 10, "A" * 11, "A" * 100]
    metric = ClippedCoverage(
        n_neighbors=2,
        reference=reference,
        embedder=length_mock_embedder,
        batch_size=128,
        strict=False,
    )

    assert metric.name == "Clipped coverage"
    assert metric.objective == "maximize"

    result = metric(["A" * 1, "A" * 1, "A" * 1, "A" * 1, "A" * 2, "A" * 2])
    assert result.value == pytest.approx(0.6666666)
    assert result.deviation is None


def length_mock_embedder(sequences: list[str]) -> np.ndarray:
    lengths = [len(sequence) for sequence in sequences]
    return np.array(lengths).reshape(-1, 1).astype(np.float64)
