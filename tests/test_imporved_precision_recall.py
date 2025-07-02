import unittest
import numpy as np
from pepme.metrics.improved_precision_recall import ImprovedPrecisionRecall


class TestImprovedPrecisionRecall(unittest.TestCase):
    def test_basic_precision(self):
        reference = ["A" * 15, "A" * 17]
        metric = ImprovedPrecisionRecall(
            reference=reference,
            embedder=length_mock_embedder,
            metric="precision",
            nhood_size=1,
        )

        self.assertEqual(metric.name, "improved precision")
        self.assertEqual(metric.objective, "maximize")

        result = metric(["A" * 2, "A" * 16])
        self.assertTrue(result.value == 0.5)

    def test_basic_recall(self):
        reference = ["A" * 15, "A" * 17]
        metric = ImprovedPrecisionRecall(
            reference=reference,
            embedder=length_mock_embedder,
            metric="recall",
            nhood_size=1,
        )

        self.assertEqual(metric.name, "improved recall")
        self.assertEqual(metric.objective, "maximize")

        result = metric(["A" * 2, "A" * 16])
        self.assertTrue(result.value == 1.0)

    def test_identical_sequences_precision(self):
        reference = ["KKAA", "KKAA", "KKKA", "KKAK"]
        metric = ImprovedPrecisionRecall(
            reference=reference, embedder=mock_embedder, metric="precision"
        )

        result = metric(sequences=reference)
        self.assertEqual(result.value, 1.0)

    def test_identical_sequences_recall(self):
        reference = ["KKAA", "KKAA", "KKKA", "KKAK"]
        metric = ImprovedPrecisionRecall(
            reference=reference, embedder=mock_embedder, metric="recall"
        )

        result = metric(sequences=reference)
        self.assertEqual(result.value, 1.0)

    def test_empty_reference(self):
        reference = []
        with self.assertRaises(ValueError):
            metric = ImprovedPrecisionRecall(
                reference=reference, embedder=mock_embedder, metric="precision"
            )
            metric(sequences=["KKAA", "KKAA"])

    def test_empty_sequences(self):
        reference = ["KKAA", "KKAA"]
        metric = ImprovedPrecisionRecall(
            reference=reference, embedder=mock_embedder, metric="precision"
        )

        with self.assertRaises(ValueError):
            metric(sequences=[])


if __name__ == "__main__":
    unittest.main()


def length_mock_embedder(sequences: list[str]) -> np.ndarray:
    lengths = [len(sequence) for sequence in sequences]
    return np.array(lengths).reshape(-1, 1)


def mock_embedder(seqs: list[str]) -> np.ndarray:
    n_ks = [seq.count("K") for seq in seqs]
    zeros = [0] * len(seqs)
    return np.array(list(zip(n_ks, zeros, strict=True)))
