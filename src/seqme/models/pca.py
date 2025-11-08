from collections.abc import Callable

import numpy as np
import sklearn.decomposition


class PCA:
    """Principle component analysis."""

    def __init__(self, embedder: Callable[[list[str]], np.ndarray], reference: list[str], n_components: int):
        """Initialize principle component analysis.

        Args:
            embedder: Embedding function.
            reference: Reference sequences to fit PCA on.
            n_components: Number of principle components.
        """
        self.embedder = embedder
        reference_embeddings = self.embedder(reference)

        if n_components > reference_embeddings.shape[-1]:
            raise ValueError("n_components cannot exceed embedding dimensionality")

        self.pca = sklearn.decomposition.PCA(n_components=n_components, random_state=0).fit(reference_embeddings)

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """Project sequences into PCA space.

        Args:
            sequences: Sequences to project their embeddings.

        Returns:
            A NumPy array of shape (n_sequences, n_components) containing the embeddings.
        """
        return self.pca.transform(self.embedder(sequences))

    @property
    def variance_explained(self) -> np.ndarray:
        """Per-component explained variance ratio."""
        return self.pca.explained_variance_ratio_
