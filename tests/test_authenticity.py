import unittest

import numpy as np

from pepme.metrics.authenticity import AuthPCT


class TestAuthPCT(unittest.TestCase):
    def test_authenticity(self):
        reference = ["ABCD", "ABKK", "KKKK"]
        metric = AuthPCT(
            reference=reference,
            embedder=mock_embedder,
            reference_name="Random",
            embedder_name="embedder",
        )

        result = metric(["ABCDE", "ABCDK", "ABCKK"])
        self.assertAlmostEqual(result.value, 0.333, places=3)

    def test_the_same(self):
        reference = ["KKAA", "KKAA"]
        metric = AuthPCT(reference=reference, embedder=mock_embedder)
        result = metric(["KKAA", "KKAA"])
        self.assertEqual(result.value, 0.0)

    def test_empty_reference(self):
        with self.assertRaises(ValueError):
            AuthPCT(reference=[], embedder=mock_embedder)

    def test_empty_sequences(self):
        reference = ["KKAA", "KKAA"]
        metric = AuthPCT(reference=reference, embedder=mock_embedder)
        with self.assertRaises(ValueError):
            metric([])

    def test_strict_mode(self):
        reference = ["KKAA", "KKAA"]
        metric = AuthPCT(reference=reference, embedder=mock_embedder, strict=True)

        with self.assertRaises(ValueError):
            metric(["KAAA"])


def mock_embedder(sequences: list[str]) -> np.ndarray:
    lengths = [len(sequence) for sequence in sequences]
    counts = [sequence.count("K") for sequence in sequences]
    return np.array([lengths, counts]).T
