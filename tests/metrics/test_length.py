import pytest

import seqme as sm


def test_single_sequence():
    metric = sm.metrics.Length()

    assert metric.name == "Length"
    assert metric.objective == "minimize"

    sequences = ["AAA"]
    result = metric(sequences)

    assert result.value == 3
    assert result.deviation is None


def test_multiple_sequences():
    metric = sm.metrics.Length(objective="maximize")
    sequences = ["AAA", "BBBB", "CCCCC"]

    assert metric.name == "Length"
    assert metric.objective == "maximize"

    result = metric(sequences)
    assert result.value == 4
    assert pytest.approx(result.deviation, abs=1e-3) == 0.816
