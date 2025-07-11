import numpy as np

from pepme.metrics import HitRate
from pepme.models.properties import Charge, Gravy


def test_hitrate():
    def condition_fn(sequences: list[str]) -> np.ndarray:
        charges = Charge()(sequences)
        gravys = Gravy()(sequences)
        return (charges > 2.0) & (charges < 3.0) & (gravys > -1.0) & (gravys < 0.0)

    metric = HitRate(condition_fn=condition_fn)

    # Name and objective properties
    assert metric.name == "Hit-rate"
    assert metric.objective == "maximize"

    result = metric(["KKKPVAAA", "KARA"])
    assert result.value == 0.5
