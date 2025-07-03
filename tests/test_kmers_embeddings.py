import numpy as np
import pytest
from pepme.embeddings.kmers_embeddings import KmerFrequencyEmbedding  # replace with actual module name


@pytest.fixture
def fake_kmer_data():
    # Simulated AMP-specific top k-mers (as if loaded from pickle)
    return {
        3: ['AKK', 'KKA', 'AAS', 'ASL', 'SLL', 'LLK', 'KKL', 'KKK', 'KLG', 'LGG']
    }


@pytest.fixture
def model(fake_kmer_data):
    return KmerFrequencyEmbedding(k=3, m=5, embeddings=fake_kmer_data)


def test_single_sequence_shape(model):
    sequence = "AKKAASL"
    embedding = model.embed(sequence)
    assert embedding.shape == (5,)
    assert np.isclose(np.sum(embedding), 1.0)


def test_batch_embedding_shape(model):
    sequences = ["AKKAASL", "LLKK", "KLVFF"]
    embeddings = model.embed(sequences)
    assert embeddings.shape == (3, 5)
    for vec in embeddings:
        assert np.isclose(np.sum(vec), 1.0)


def test_known_kmers_counted_correctly(fake_kmer_data):
    model = KmerFrequencyEmbedding(k=3, m=5, embeddings=fake_kmer_data)
    sequence = "AKKAASL"  # contains: AKK, KKA, KAA, AAS, ASL → AKK, KKA, AAS, ASL in vocab
    embedding = model.embed(sequence)

    # AKK → 1, KKA → 1, AAS → 1, ASL → 1, total = 5 → all get 1/5
    expected_vector = np.array([1/5, 1/5, 1/5, 1/5, 0.0])
    assert np.allclose(embedding, expected_vector)
