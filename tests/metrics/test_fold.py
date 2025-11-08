import pytest

from seqme.metrics import Count, Fold


def test_k_fold():
    metric = Fold(metric=Count(), n_splits=2, deviation="se", estimate="biased")
    assert metric.name == "Count"
    assert metric.objective == "maximize"

    sequences = ["AA", "AAA", "AAAA", "AAA", "AAAAAA"]
    result = metric(sequences)
    assert result.value == 2.5
    assert result.deviation == 0.5 / (2**0.5)


def test_std():
    metric = Fold(metric=Count(), n_splits=2, deviation="std", estimate="biased")
    assert metric.name == "Count"
    assert metric.objective == "maximize"

    sequences = ["AA", "AAA", "AAAA", "AAA", "AAAAAA"]
    result = metric(sequences)
    assert result.value == 2.5
    assert result.deviation == 0.25 ** (1 / 2)


def test_variance():
    metric = Fold(metric=Count(), n_splits=2, deviation="var", estimate="biased")
    assert metric.name == "Count"
    assert metric.objective == "maximize"

    sequences = ["AA", "AAA", "AAAA", "AAA", "AAAAAA"]
    result = metric(sequences)
    assert result.value == 2.5
    assert result.deviation == 0.25


def test_one_fold():
    with pytest.raises(ValueError):
        Fold(metric=Count(), n_splits=1, estimate="biased")


def test_k_larger_than_sequence_count():
    metric = Fold(metric=Count(), n_splits=20, estimate="biased")
    assert metric.name == "Count"
    assert metric.objective == "maximize"

    sequences = ["AA", "AAA", "AAAA", "AAA", "AAAAAA"]
    with pytest.raises(ValueError, match=r"^Cannot split into 20 folds with only 5 sequences\.$"):
        metric(sequences)


def test_split_size():
    metric = Fold(metric=Count(), split_size=2, deviation="se", estimate="biased")
    assert metric.name == "Count"
    assert metric.objective == "maximize"

    sequences = ["AA", "AAA", "AAAA", "AAA", "AAAAAA"]
    result = metric(sequences)
    # total count is 5, split into chunks of 2 → [2,2,1] → mean=5/3, std≈0.471405
    assert result.value == pytest.approx(5 / 3)
    assert result.deviation == pytest.approx(0.471405 / (3**0.5), abs=1e-6)


def test_large_split_size():
    metric = Fold(metric=Count(), split_size=20)
    assert metric.name == "Count"
    assert metric.objective == "maximize"

    sequences = ["AA", "AAA", "AAAA", "AAA", "AAAAAA"]
    result = metric(sequences)
    # split_size > n_seqs → single fold of all sequences
    assert result.value == 5
    assert result.deviation is None


def test_large_split_size_drop_last():
    metric = Fold(metric=Count(), split_size=20, drop_last=True)
    assert metric.name == "Count"
    assert metric.objective == "maximize"

    sequences = ["AA", "AAA", "AAAA", "AAA", "AAAAAA"]
    with pytest.raises(ValueError, match=r"^With drop_last=True, cannot form any fold of size 20 from 5 sequences\.$"):
        metric(sequences)


def test_split_size_drop_last():
    metric = Fold(metric=Count(), split_size=2, drop_last=True)
    assert metric.name == "Count"
    assert metric.objective == "maximize"

    sequences = ["AA", "AAA", "AAAA", "AAA", "AAAAAA"]
    result = metric(sequences)
    # drop_last=True, sizes: [2,2] (1 leftover dropped) → mean=2, std=0
    assert result.value == 2
    assert result.deviation == 0
