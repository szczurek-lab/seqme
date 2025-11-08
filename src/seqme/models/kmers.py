import numpy as np


class KmerFrequencyEmbedding:
    """Computes normalized k-mer frequency embeddings for sequences."""

    def __init__(self, kmers: list[str]):
        """Initialize model.

        Args:
            kmers: List of valid k-mers (same length).
        """
        ks = {len(s) for s in kmers}
        if len(ks) > 1:
            raise ValueError("Not all kmers have the same length")

        self.k = list(ks)[0]
        self.kmer_to_idx = {kmer: idx for idx, kmer in enumerate(kmers)}

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """Embed a list of sequences as k-mer frequency vectors.

        Args:
            sequences: Sequences to embed.

        Returns:
            A NumPy array of shape (n_sequences, total_kmers) containing the embeddings.
        """
        return np.array([self._embed(seq, self.kmer_to_idx, self.k) for seq in sequences])

    def _embed(self, sequence: str, kmer_to_idx: dict[str, int], k: int) -> np.ndarray:
        """Embed one sequence as a k-mer frequency vector."""
        embedding = np.zeros(len(kmer_to_idx))
        total = max(len(sequence) - k + 1, 0)

        for i in range(total):
            kmer = sequence[i : i + k]
            idx = kmer_to_idx.get(kmer)
            if idx is not None:
                embedding[idx] += 1

        return embedding / total if total > 0 else embedding
