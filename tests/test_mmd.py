import unittest

import numpy as np

from pepme.metrics.mmd import MMD


class TestMMD(unittest.TestCase):
    def test_shifted(self):
        reference = ["KKAA", "KKAA"]
        metric = MMD(reference=reference, embedder=embedder)

        self.assertEqual(metric.name, "MMD")
        self.assertEqual(metric.objective, "minimize")

        result = metric(["KAAA", "KAAA"])
        self.assertAlmostEqual(result.value, 9.975075721740723, places=7)

    def test_the_same(self):
        reference = ["KKAA", "KKAA"]
        metric = MMD(reference=reference, embedder=embedder)
        result = metric(["KKAA", "KKAA"])
        self.assertEqual(result.value, 0.0)

    def test_empty_reference(self):
        with self.assertRaises(ValueError):
            MMD(reference=[], embedder=embedder)

    def test_empty_sequences(self):
        reference = ["KKAA", "KKAA"]
        metric = MMD(reference=reference, embedder=embedder)
        with self.assertRaises(ValueError):
            metric([])

    def test_strict_mode(self):
        reference = ["KKAA", "KKAA"]
        metric = MMD(reference=reference, embedder=embedder, strict=True)

        with self.assertRaises(ValueError):
            metric(["KAAA"])


def embedder(seqs: list[str]) -> np.ndarray:
    n_ks = [seq.count("K") for seq in seqs]
    zeros = [0] * len(seqs)
    return np.array(list(zip(n_ks, zeros, strict=True)))
