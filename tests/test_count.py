import unittest

from pepme.metrics import Count


class TestFold(unittest.TestCase):
    def test_count(self):
        metric = Count()
        self.assertEqual(metric.name, "Count")
        self.assertEqual(metric.objective, "maximize")

        sequences = ["AAA", "BBBB", "CCCCC"]
        result = metric(sequences)

        self.assertEqual(result.value, 3)


if __name__ == "__main__":
    unittest.main()
