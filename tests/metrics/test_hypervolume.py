import numpy as np
import pytest

from seqme.metrics import Hypervolume


def p_count_aa(sequences: list[str], aa: str) -> np.ndarray:
    return np.array([sequence.count(aa) for sequence in sequences])


def test_hvi():
    metric = Hypervolume(
        predictors=[
            lambda seqs: p_count_aa(seqs, aa="K"),
            lambda seqs: p_count_aa(seqs, aa="R"),
        ],
        method="hvi",
        nadir=np.zeros(2),
    )

    # Name and objective properties
    assert metric.name == "Hypervolume"
    assert metric.objective == "maximize"

    result = metric(["KKKK", "RRR", "KKKKRRR"])
    assert result.value == pytest.approx(12)
    assert result.deviation is None


def test_convex_hull_hv():
    metric = Hypervolume(
        predictors=[
            lambda seqs: p_count_aa(seqs, aa="K"),
            lambda seqs: p_count_aa(seqs, aa="R"),
        ],
        method="convex-hull",
        nadir=np.zeros(2),
    )

    # Name and objective properties
    assert metric.name == "Hypervolume"
    assert metric.objective == "maximize"

    result = metric(["KKKK", "RRR"])
    assert result.value == pytest.approx(6)
    assert result.deviation is None


def test_hvi_with_ideal_hv():
    metric = Hypervolume(
        predictors=[
            lambda seqs: p_count_aa(seqs, aa="K"),
            lambda seqs: p_count_aa(seqs, aa="R"),
        ],
        method="hvi",
        nadir=np.zeros(2),
        ideal=np.array([10, 10]),
    )

    # Name and objective properties
    assert metric.name == "Hypervolume"
    assert metric.objective == "maximize"

    result = metric(["RRR", "KKKK", "KKKKRRR"])
    assert result.value == pytest.approx(0.12)
    assert result.deviation is None


def test_strict():
    metric = Hypervolume(
        predictors=[
            lambda seqs: p_count_aa(seqs, aa="K"),
            lambda seqs: p_count_aa(seqs, aa="R"),
        ],
        method="hvi",
        nadir=np.ones(2),
    )

    # Name and objective properties
    assert metric.name == "Hypervolume"
    assert metric.objective == "maximize"

    with pytest.raises(ValueError):
        metric(["KKKK", "RRR", "KKKKRRR"])


def test_not_strict():
    metric = Hypervolume(
        predictors=[
            lambda seqs: p_count_aa(seqs, aa="K"),
            lambda seqs: p_count_aa(seqs, aa="R"),
        ],
        method="hvi",
        nadir=np.zeros(2),
        ideal=np.ones(2) * 3,
        strict=False,
    )

    # Name and objective properties
    assert metric.name == "Hypervolume"
    assert metric.objective == "maximize"

    result = metric(["KKKK", "RRR", "KKKKRRR"])
    assert result.value == 1.0
    assert result.deviation is None
