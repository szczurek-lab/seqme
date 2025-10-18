import numpy as np
import pytest

from seqme.metrics import AuthPct


def mock_embedder(sequences: list[str]) -> np.ndarray:
    lengths = [len(sequence) for sequence in sequences]
    return np.array(lengths)[:, None]


def test_authenticity():
    train_set = ["A" * 3, "A" * 10, "A" * 12]
    metric = AuthPct(
        train_set=train_set,
        embedder=mock_embedder,
    )

    result = metric(["A" * 15, "A" * 4, "A" * 13])
    assert result.value == pytest.approx(0.333, abs=1e-3)


def test_the_same():
    train_set = ["KKAA", "KKAA"]
    metric = AuthPct(train_set=train_set, embedder=mock_embedder)
    result = metric(["KKAA", "KKAA"])
    assert result.value == 0.0


def test_empty_reference_raises():
    # initializing with no training data should fail immediately
    with pytest.raises(ValueError):
        AuthPct(train_set=[], embedder=mock_embedder)


def test_empty_sequences_raises():
    train_set = ["KKAA", "KKAA"]
    metric = AuthPct(train_set=train_set, embedder=mock_embedder)
    # calling the metric on an empty list should error
    with pytest.raises(ValueError):
        metric([])
