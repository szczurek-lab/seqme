import unittest

import numpy as np

from pepme.metrics.authenticity import AuthPct


class TestAuthPCT(unittest.TestCase):
    def test_authenticity(self):
        train_set = ["A" * 3, "A" * 10, "A" * 12]
        metric = AuthPct(
            train_set=train_set,
            embedder=mock_embedder,
            embedder_name="embedder",
        )

        result = metric(["A" * 15, "A" * 4, "A" * 13])
        self.assertAlmostEqual(result.value, 0.333, places=3)

    def test_the_same(self):
        train_set = ["KKAA", "KKAA"]
        metric = AuthPct(train_set=train_set, embedder=mock_embedder)
        result = metric(["KKAA", "KKAA"])
        self.assertEqual(result.value, 0.0)

    def test_empty_reference(self):
        with self.assertRaises(ValueError):
            AuthPct(train_set=[], embedder=mock_embedder)

    def test_empty_sequences(self):
        train_set = ["KKAA", "KKAA"]
        metric = AuthPct(train_set=train_set, embedder=mock_embedder)
        with self.assertRaises(ValueError):
            metric([])

    def test_strict_mode(self):
        train_set = ["KKAA", "KKAA"]
        metric = AuthPct(train_set=train_set, embedder=mock_embedder, strict=True)

        with self.assertRaises(ValueError):
            metric(["KAAA"])


def mock_embedder(sequences: list[str]) -> np.ndarray:
    lengths = [len(sequence) for sequence in sequences]
    return np.array(lengths)[:, None]


if __name__ == "__main__":
    unittest.main()
