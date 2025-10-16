import pytest

from seqme.metrics import Novelty


def test_simple():
    reference = ["KRQS", "KKPRA", "KKKR"]
    metric = Novelty(reference=reference)

    assert metric.name == "Novelty"
    assert metric.objective == "maximize"

    result = metric(["KRQS", "KA"])  # "KRQS" seen, "KA" novel → 1 novel out of 2
    assert result.value == pytest.approx(0.5)
    assert result.deviation is None


def test_empty_sequences():
    metric = Novelty(reference=["A", "B"])
    result = metric([])
    assert result.value == 0.0
    assert result.deviation is None
