import unittest

from pepme.metrics import Count, Fold


class TestFold(unittest.TestCase):
    def test_k_fold(self):
        metric = Fold(metric=Count(), k=2)
        self.assertEqual(metric.name, "Count")
        self.assertEqual(metric.objective, "maximize")

        sequences = ["AA", "AAA", "AAAA", "AAA", "AAAAAA"]
        result = metric(sequences)
        self.assertEqual(result.value, 2.5)
        self.assertEqual(result.deviation, 0.5)

    def test_k_larger_than_sequence_count(self):
        metric = Fold(metric=Count(), k=20)
        self.assertEqual(metric.name, "Count")
        self.assertEqual(metric.objective, "maximize")

        sequences = ["AA", "AAA", "AAAA", "AAA", "AAAAAA"]

        with self.assertRaisesRegex(ValueError, "^Cannot split into 20 folds with only 5 sequences.$"):
            metric(sequences)

    def test_split_size(self):
        metric = Fold(metric=Count(), split_size=2)
        self.assertEqual(metric.name, "Count")
        self.assertEqual(metric.objective, "maximize")

        sequences = ["AA", "AAA", "AAAA", "AAA", "AAAAAA"]
        result = metric(sequences)
        self.assertEqual(result.value, 5 / 3)
        self.assertAlmostEqual(result.deviation, 0.471405, places=6)


    def test_large_split_size(self):
        metric = Fold(metric=Count(), split_size=20)
        self.assertEqual(metric.name, "Count")
        self.assertEqual(metric.objective, "maximize")

        sequences = ["AA", "AAA", "AAAA", "AAA", "AAAAAA"]
        result = metric(sequences)
        self.assertEqual(result.value,5)
        self.assertEqual(result.deviation, 0)

    def test_large_split_size_drop_last(self):
        metric = Fold(metric=Count(), split_size=20, drop_last=True)
        self.assertEqual(metric.name, "Count")
        self.assertEqual(metric.objective, "maximize")

        sequences = ["AA", "AAA", "AAAA", "AAA", "AAAAAA"]
        with self.assertRaisesRegex(ValueError, "^With drop_last=True, cannot form any fold of size 20 from 5 sequences.$"):
            metric(sequences)

    def test_split_size_drop_last(self):
        metric = Fold(metric=Count(), split_size=2, drop_last=True)
        self.assertEqual(metric.name, "Count")
        self.assertEqual(metric.objective, "maximize")

        sequences = ["AA", "AAA", "AAAA", "AAA", "AAAAAA"]
        result = metric(sequences)
        self.assertEqual(result.value, 2)
        self.assertEqual(result.deviation, 0)


if __name__ == "__main__":
    unittest.main()
