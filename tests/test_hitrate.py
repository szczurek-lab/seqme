import unittest

import numpy as np

from pepme.metrics.hitrate import HitRate
from pepme.properties.physicochemical import Charge, Gravy


class TestHitRate(unittest.TestCase):
    def test_hitrate(self):
        def filter_fn(sequences: list[str]) -> np.ndarray:
            charges = Charge()(sequences)
            gravys = Gravy()(sequences)
            return (charges > 2.0) & (charges < 3.0) & (gravys > -1.0) & (gravys < 0.0)

        metric = HitRate(filter_fn=filter_fn)

        # Name and objective properties
        self.assertEqual(metric.name, "Hit-rate")
        self.assertEqual(metric.objective, "maximize")

        result = metric(["KKKPVAAA", "KARA"])
        self.assertEqual(result.value, 0.5)


if __name__ == "__main__":
    unittest.main()
