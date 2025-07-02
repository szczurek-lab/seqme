import unittest
import numpy as np
from pepme.metrics.improved_precision_recall import ImprovedPrecisionRecall


class TestImprovedPrecisionRecall(unittest.TestCase):
    def test_basic_precision(self):
        reference = ["KKAA", "KKAA", "KKKA", "KKAK"]
        metric = ImprovedPrecisionRecall(
            reference=reference, embedder=embedder, metric="precision"
        )

        self.assertEqual(metric.name, "improved precision")
        self.assertEqual(metric.objective, "maximize")

        result = metric(["KKMMM", "KKZAC", "KCJSNL", "KKANJ"])
        self.assertTrue(0 <= result.value <= 1)  # precision should be between 0 and 1

    def test_basic_recall(self):
        reference = ["KKAA", "KKAA", "KKKA", "KKAK"]
        metric = ImprovedPrecisionRecall(
            reference=reference, embedder=embedder, metric="recall"
        )

        self.assertEqual(metric.name, "improved recall")
        self.assertEqual(metric.objective, "maximize")

        result = metric(["KKMMM", "KKZAC", "KCJSNL", "KKANJ"])
        self.assertTrue(0 <= result.value <= 1)  # recall should be between 0 and 1

    def test_identical_sequences_precision(self):
        reference = ["KKAA", "KKAA", "KKKA", "KKAK"]
        metric = ImprovedPrecisionRecall(
            reference=reference, embedder=embedder, metric="precision"
        )

        result = metric(sequences=reference)
        self.assertEqual(result.value, 1.0)

    def test_identical_sequences_recall(self):
        reference = ["KKAA", "KKAA", "KKKA", "KKAK"]
        metric = ImprovedPrecisionRecall(
            reference=reference, embedder=embedder, metric="recall"
        )

        result = metric(sequences=reference)
        self.assertEqual(result.value, 1.0)


if __name__ == "__main__":
    unittest.main()


def embedder(seqs: list[str]) -> np.ndarray:
    # This is a mock embedder function that counts the number of 'K's in each sequence.
    n_ks = [seq.count("K") for seq in seqs]
    zeros = [0] * len(seqs)
    return np.array(list(zip(n_ks, zeros, strict=True)))
