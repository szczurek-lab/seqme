import numpy as np

from seqme.models import KmerFrequencyEmbedding


def test_single_sequence_shape():
    sequences = ["AKKAASL"]
    kmers = ["AKK", "KKA", "AAS", "ASL", "SLL"]
    model = KmerFrequencyEmbedding(kmers=kmers)

    embeddings = model(sequences)
    assert embeddings.shape == (1, 5)


def test_batch_embedding_shape():
    sequences = ["AKKAASL", "LLKK", "KLVFF"]
    kmers = ["AKK", "KKA", "AAS", "ASL", "SLL"]
    model = KmerFrequencyEmbedding(kmers=kmers)

    embeddings = model(sequences)
    assert embeddings.shape == (3, 5)


def test_known_kmers_counted_correctly():
    sequences = ["AKKAASL"]
    kmers = ["AKK", "KKA", "AAS", "ASL", "SLL"]
    model = KmerFrequencyEmbedding(kmers=kmers)
    embeddings = model(sequences)
    # AKK → 1, KKA → 1, AAS → 1, ASL → 1, SLL → 0; normalized by total 5 occurrences
    expected_vector = np.array([[1 / 5, 1 / 5, 1 / 5, 1 / 5, 0.0]])
    assert np.allclose(embeddings, expected_vector)
