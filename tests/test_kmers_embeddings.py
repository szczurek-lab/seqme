import unittest

import numpy as np

from pepme.models.embeddings import KmerFrequencyEmbedding


class TestJaccardSimilarity(unittest.TestCase):
    def test_single_sequence_shape(self):
        sequences = ["AKKAASL"]

        kmers = ["AKK", "KKA", "AAS", "ASL", "SLL"]
        model = KmerFrequencyEmbedding(kmers=kmers)

        embeddings = model(sequences)
        assert embeddings.shape == (1, 5)

    def test_batch_embedding_shape(self):
        sequences = ["AKKAASL", "LLKK", "KLVFF"]

        kmers = ["AKK", "KKA", "AAS", "ASL", "SLL"]
        model = KmerFrequencyEmbedding(kmers=kmers)

        embeddings = model(sequences)
        assert embeddings.shape == (3, 5)

    def test_known_kmers_counted_correctly(self):
        sequences = ["AKKAASL"]

        kmers = ["AKK", "KKA", "AAS", "ASL", "SLL"]
        model = KmerFrequencyEmbedding(kmers=kmers)

        embeddings = model(sequences)

        # AKK → 1, KKA → 1, AAS → 1, ASL → 1, total = 5 → all get 1/5
        expected_vector = np.array([1 / 5, 1 / 5, 1 / 5, 1 / 5, 0.0]).reshape(1, -1)
        assert np.allclose(embeddings, expected_vector)


if __name__ == "__main__":
    unittest.main()
