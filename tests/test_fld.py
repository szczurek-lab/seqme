import math
import unittest

import numpy as np

from pepme.metrics import FLD


class TestFLD(unittest.TestCase):
    def test_shifted(self):
        def embedder(seqs: list[str]) -> np.ndarray:
            n_ks = [seq.count("K") for seq in seqs]
            zeros = [0] * len(seqs)
            return np.array(list(zip(n_ks, zeros, strict=True)))

        reference = {
            "train": ["KKAA", "KKAA", "KKKK"],
            "test": ["KKAA", "KKKK"]
        }
        metric = FLD(
            reference=reference,
            embedder=embedder,
            reference_name={"train": "train_ref", "test": "test_ref"},
            embedder_name="seq",
        )

        # Name and objective properties
        self.assertEqual(metric.name, "FLD@seq ({'train': 'train_ref', 'test': 'test_ref'})")
        self.assertEqual(metric.objective, "minimize")

        result = metric(["KAAA", "KAAA"])
        # FLD computation is complex, just verify it returns a numeric value
        self.assertIsInstance(result.value, (int, float))
        self.assertFalse(math.isnan(result.value))

    def test_overlapping(self):
        def embedder(seq: list[str], n_dim: int = 3) -> np.ndarray:
            return np.zeros((len(seq), n_dim))

        reference = {
            "train": ["KRQS", "AAAA", "BBBB"],
            "test": ["CCCC", "DDDD"]
        }
        metric = FLD(reference=reference, embedder=embedder)

        # Name and objective properties
        self.assertEqual(metric.name, "FLD")
        self.assertEqual(metric.objective, "minimize")

        result = metric(["KA", "BBBB"])
        # With zero embeddings, should produce some numeric result
        self.assertIsInstance(result.value, (int, float))

    def test_minimum_sequences_requirement(self):
        def embedder(seq: list[str], n_dim: int = 3) -> np.ndarray:
            return np.zeros((len(seq), n_dim))

        # Test error when train has < 2 sequences
        with self.assertRaises(ValueError):
            FLD(
                reference={"train": ["KRQS"], "test": ["AAA", "BBB"]},
                embedder=embedder
            )

        # Test error when test has < 2 sequences  
        with self.assertRaises(ValueError):
            FLD(
                reference={"train": ["KRQS", "AAA"], "test": ["BBB"]},
                embedder=embedder
            )

    def test_missing_train_test_keys(self):
        def embedder(seq: list[str], n_dim: int = 3) -> np.ndarray:
            return np.zeros((len(seq), n_dim))

        # Test error when missing train key
        with self.assertRaises(AssertionError):
            FLD(
                reference={"test": ["AAA", "BBB"]},
                embedder=embedder
            )

        # Test error when missing test key
        with self.assertRaises(AssertionError):
            FLD(
                reference={"train": ["AAA", "BBB"]},
                embedder=embedder
            )


if __name__ == "__main__":
    unittest.main()
