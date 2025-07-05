import unittest

import numpy as np

from pepme.metrics.hypervolume import HV


def p_count_aa(sequences: list[str], aa: str) -> np.ndarray:
    return np.array([sequence.count(aa) for sequence in sequences])


class TestHypervolume(unittest.TestCase):
    def test_standard(self):
        metric = HV(
            predictors=[
                lambda seqs: p_count_aa(seqs, aa="K"),
                lambda seqs: p_count_aa(seqs, aa="R"),
            ],
            method="standard",
            nadir=np.zeros(2),
        )

        # Name and objective properties
        self.assertEqual(metric.name, "HV")
        self.assertEqual(metric.objective, "maximize")

        result = metric(["KKKK", "RRR", "KKKKRRR"])
        self.assertAlmostEqual(result.value, 12)
        self.assertIsNone(result.deviation)

    def test_convex_hull(self):
        metric = HV(
            predictors=[
                lambda seqs: p_count_aa(seqs, aa="K"),
                lambda seqs: p_count_aa(seqs, aa="R"),
            ],
            method="convex-hull",
            nadir=np.zeros(2),
        )

        # Name and objective properties
        self.assertEqual(metric.name, "HV (convex-hull)")
        self.assertEqual(metric.objective, "maximize")

        result = metric(["KKKK", "RRR"])
        self.assertAlmostEqual(result.value, 6)
        self.assertIsNone(result.deviation)

    def test_standard_with_ideal(self):
        metric = HV(
            predictors=[
                lambda seqs: p_count_aa(seqs, aa="K"),
                lambda seqs: p_count_aa(seqs, aa="R"),
            ],
            method="standard",
            nadir=np.zeros(2),
            ideal=np.array([10, 10]),
        )

        # Name and objective properties
        self.assertEqual(metric.name, "HV")
        self.assertEqual(metric.objective, "maximize")

        result = metric(["RRR", "KKKK", "KKKKRRR"])
        self.assertAlmostEqual(result.value, 0.12)
        self.assertIsNone(result.deviation)


if __name__ == "__main__":
    unittest.main()
