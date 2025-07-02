import unittest

from pepme.metrics.uniqueness import Uniqueness


class TestUniqueness(unittest.TestCase):
    def test_name_and_objective(self):
        # With a custom name
        metric = Uniqueness(name="Sample")
        self.assertEqual(metric.name, "Uniqueness (Sample)")
        self.assertEqual(metric.objective, "maximize")

        # Default name
        default_metric = Uniqueness()
        self.assertEqual(default_metric.name, "Uniqueness")
        self.assertEqual(default_metric.objective, "maximize")

    def test_compute_metric(self):
        metric = Uniqueness()

        # Sequences: ["A","B","A","C"] → 3 unique out of 4 → 0.75
        result = metric(["A", "B", "A", "C"])
        self.assertAlmostEqual(result.value, 0.75)
        self.assertIsNone(result.deviation)

        # All duplicates: ["X","X","X"] → 1 unique out of 3 → ~0.3333
        result = metric(["X", "X", "X"])
        self.assertAlmostEqual(result.value, 1 / 3)
        self.assertIsNone(result.deviation)

    def test_empty_sequences(self):
        # When no sequences are provided, uniqueness should be 0.0
        metric = Uniqueness()
        result = metric([])
        self.assertEqual(result.value, 0.0)
        self.assertIsNone(result.deviation)


if __name__ == "__main__":
    unittest.main()
