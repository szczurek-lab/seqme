import unittest

from pepme.metrics import Novelty


class TestNovelty(unittest.TestCase):
    def test_compute_metric(self):
        # Given a small reference set
        reference = ["KRQS", "KKPRA", "KKKR"]
        metric = Novelty(reference=reference, reference_name="Random")

        # Name and objective properties
        self.assertEqual(metric.name, "Novelty (Random)")
        self.assertEqual(metric.objective, "maximize")

        # Compute novelty: one seen ("KRQS"), one novel ("KA") => 1/2
        result = metric(["KRQS", "KA"])
        self.assertAlmostEqual(result.value, 0.5)
        # No defined value range
        self.assertIsNone(result.deviation)

    def test_empty_sequences(self):
        # When no sequences are provided, novelty should be 0.0
        metric = Novelty(reference=["A", "B"])
        result = metric([])
        self.assertEqual(result.value, 0.0)
        self.assertIsNone(result.deviation)


if __name__ == "__main__":
    unittest.main()
