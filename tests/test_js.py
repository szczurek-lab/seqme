import unittest

from pepme.metrics.js import JaccardSimilarity
from pepme.core import MetricResult


class TestJaccardSimilarity(unittest.TestCase):
    def test_name_and_objective_default(self):
        # Default name (no reference_name) and default objective ("minimize")
        reference = ["ABC"]
        metric = JaccardSimilarity(reference=reference, n=3)
        self.assertEqual(metric.name, "Jaccard-3")
        self.assertEqual(metric.objective, "minimize")

    def test_name_with_reference_name_and_objective_override(self):
        # Custom reference_name and overriding objective to "maximize"
        reference = ["ABC"]
        metric = JaccardSimilarity(
            reference=reference, n=3, objective="maximize", reference_name="SampleRef"
        )
        self.assertEqual(metric.name, "Jaccard-3 (SampleRef)")
        self.assertEqual(metric.objective, "maximize")

    def test_compute_metric_basic(self):
        # reference 2-grams: {"AB","BC","CD"}, sequences ["ABC","XYZ"]
        # J("ABC") = 2/3≈0.6667, J("XYZ") = 0/5=0.0 → average = (0.6667+0)/2 ≈0.3333
        reference = ["ABC", "BCD"]
        metric = JaccardSimilarity(reference=reference, n=2)
        result = metric(["ABC", "XYZ"])
        self.assertAlmostEqual(result.value, 1 / 3)
        self.assertIsNone(result.deviation)

    def test_perfect_overlap(self):
        # sequence identical to reference yields J=1.0
        reference = ["HELLO"]
        metric = JaccardSimilarity(reference=reference, n=3)
        result = metric(["HELLO"])
        self.assertAlmostEqual(result.value, 1.0)
        self.assertIsNone(result.deviation)

    def test_sequences_shorter_than_n(self):
        # reference_ngrams empty (all refs length < n), sequences all too short:
        # both unions empty → similarity treated as 0 each → average=0
        reference = ["A", "B"]
        metric = JaccardSimilarity(reference=reference, n=2)
        result = metric(["A", "B"])
        self.assertEqual(result.value, 0.0)
        self.assertIsNone(result.deviation)

    def test_empty_sequences(self):
        # no sequences → defined to return 0.0
        reference = ["ABC"]
        metric = JaccardSimilarity(reference=reference, n=3)
        result = metric([])
        self.assertEqual(result.value, 0.0)
        self.assertIsNone(result.deviation)


if __name__ == "__main__":
    unittest.main()
