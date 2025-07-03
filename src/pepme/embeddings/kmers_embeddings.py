import pickle
from typing import List, Union

import numpy as np


class KmerFrequencyEmbedding:
    """
    Generate k-mer–based frequency embeddings for peptide sequences using AMP-specific vocabulary.

    The vocabulary is based on the most frequent k-mers in antimicrobial peptides (AMPs),
    with globally frequent k-mers (from UniProt proteins) removed to retain specificity.

    This model supports embedding a single sequence or a batch of sequences.

    Parameters
    ----------
    k : int
        Length of k-mers to extract.
    m : int
        Number of top AMP-specific k-mers to use.
    embeddings : str or dict
        Path to pickle file or in-memory dict with structure:
            {1: [k1, k2, ...], 2: [...], 3: [...], ...}
        where the values are prefiltered lists of top AMP-specific k-mers.

    Example
    -------
    >>> model = KmerFrequencyEmbedding(k=3, m=10, embeddings="kmers.pkl")
    >>> model.embed("AKKAASL")
    array([...])

    >>> model.embed(["AKKAASL", "LLKK"])
    array([[...], [...]])
    """

    def __init__(self, k: int, m: int, embeddings: Union[str, dict]):
        self.k = k
        self.m = m

        # Load AMP-specific k-mer vocabulary
        if isinstance(embeddings, str):
            with open(embeddings, "rb") as f:
                loaded = pickle.load(f)
        else:
            loaded = embeddings

        # Just take top-m strings from list of k-mers
        self.vocab = loaded[k][:m]
        self.kmer_to_idx = {kmer: idx for idx, kmer in enumerate(self.vocab)}

    def _embed_one(self, sequence: str) -> np.ndarray:
        """Embed one peptide into normalized AMP k-mer frequency vector."""
        vector = np.zeros(len(self.vocab))
        n_possible = max(len(sequence) - self.k + 1, 1)

        for i in range(n_possible):
            kmer = sequence[i : i + self.k]
            if kmer in self.kmer_to_idx:
                vector[self.kmer_to_idx[kmer]] += 1

        return vector / n_possible

    def embed(
        self, sequences: Union[str, List[str]]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Embed one or multiple peptide sequences into AMP-specific k-mer space.

        Parameters
        ----------
        sequences : str or list of str

        Returns
        -------
        np.ndarray or 2D np.ndarray
        """
        if isinstance(sequences, str):
            return self._embed_one(sequences)
        return np.array([self._embed_one(seq) for seq in sequences])
    
    __call__ = embed
