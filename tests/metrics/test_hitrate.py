import numpy as np
import pytest

from seqme.metrics import HitRate
from seqme.models import Charge, Gravy


def condition_fn(sequences: list[str]) -> np.ndarray:
    charges = Charge()(sequences)
    gravys = Gravy()(sequences)
    return (charges > 2.0) & (charges < 3.0) & (gravys > -1.0) & (gravys < 0.0)


def test_hitrate():
    metric = HitRate(condition_fn=condition_fn)

    # Name and objective properties
    assert metric.name == "Hit-rate"
    assert metric.objective == "maximize"

    result = metric(["KKKPVAAA", "KARA"])
    assert result.value == 0.5
    assert result.deviation == pytest.approx(0.3535533)


def test_hitrate_custom_name():
    metric = HitRate(condition_fn=condition_fn, name="hitrate (physico)")

    # Name and objective properties
    assert metric.name == "hitrate (physico)"
    assert metric.objective == "maximize"

    result = metric(["KKKPVAAA", "KKKPVAAA"])
    assert result.value == 1.0
    assert result.deviation == 0.0
