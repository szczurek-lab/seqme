import unittest

import numpy as np

from pepme.metrics import Diversity


class TestDiversity(unittest.TestCase):
    def test_compute_metric(self):
        reference = ["AAA", "BBBB", "CCCCC"]
        metric = Diversity(reference=reference, reference_name="Random")

        # Name and objective properties
        self.assertEqual(metric.name, "Diversity (Random)")
        self.assertEqual(metric.objective, "maximize")

        result = metric(["AA", "BB", "CCCCD"])

        minimum_distances = np.array([1, 2, 1])

        self.assertAlmostEqual(result.value, minimum_distances.mean())
        self.assertAlmostEqual(result.deviation, minimum_distances.std())

    def test_single_sequence(self):
        reference = ["AAA", "BBBB", "CCCCC"]
        metric = Diversity(reference=reference, reference_name="Random2")

        # Name and objective properties
        self.assertEqual(metric.name, "Diversity (Random2)")
        self.assertEqual(metric.objective, "maximize")

        result = metric(["AAA"])

        self.assertAlmostEqual(result.value, 0.0)
        self.assertAlmostEqual(result.deviation, 0.0)

    def test_empty_reference(self):
        with self.assertRaises(Exception) as context:
            Diversity(reference=[])
            self.assertTrue(
                "References must contain at least one sample." in context.exception,
            )


if __name__ == "__main__":
    unittest.main()
