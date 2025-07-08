import math
import unittest

import numpy as np

from pepme.metrics import FID


class TestFid(unittest.TestCase):
    def test_shifted(self):
        def embedder(seqs: list[str]) -> np.ndarray:
            n_ks = [seq.count("K") for seq in seqs]
            zeros = [0] * len(seqs)
            return np.array(list(zip(n_ks, zeros, strict=True)))

        reference = ["KKAA", "KKAA"]
        metric = FID(
            reference=reference,
            embedder=embedder,
            reference_name="ref",
            embedder_name="seq",
        )

        # Name and objective properties
        self.assertEqual(metric.name, "FID@seq (ref)")
        self.assertEqual(metric.objective, "minimize")

        result = metric(["KAAA", "KAAA"])
        self.assertEqual(result.value, 1.0)

    def test_overlapping(self):
        def embedder(seq: list[str], n_dim: int = 3) -> np.ndarray:
            return np.zeros((len(seq), n_dim))

        reference = ["KRQS", "AAAA"]
        metric = FID(reference=reference, embedder=embedder)

        # Name and objective properties
        self.assertEqual(metric.name, "FID")
        self.assertEqual(metric.objective, "minimize")

        result = metric(["KA", "BBBB"])
        self.assertEqual(result.value, 0.0)

    def test_single_sequence(self):
        def embedder(seq: list[str], n_dim: int = 3) -> np.ndarray:
            return np.zeros((len(seq), n_dim))

        reference = ["KRQS", "AAA"]
        metric = FID(reference=reference, embedder=embedder)

        # Name and objective properties
        self.assertEqual(metric.name, "FID")
        self.assertEqual(metric.objective, "minimize")

        result = metric(["KA"])
        assert math.isnan(result.value)


if __name__ == "__main__":
    unittest.main()
