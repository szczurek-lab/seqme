import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


def pca(embeddings: np.ndarray | list[np.ndarray], seed: int = 42) -> np.ndarray | list[np.ndarray]:
    """Project embeddings into 2D using PCA.

    Args:
        embeddings: 2D array where each row is a data point.
        seed: Seed for reproducibility in PCA.

    Returns:
        2D array of shape (n_samples, 2) or list.

    Notes:
        PCA is a linear dimensionality reduction that preserves global structure by projecting data into directions of maximal variance.
    """

    def _pca(embeds: np.ndarray) -> np.ndarray:
        return PCA(n_components=2, random_state=seed).fit_transform(embeds)

    if isinstance(embeddings, list):
        embeddings, splits = _prepare_data_groups(embeddings)
        zs = _pca(embeddings)
        zs_split = np.split(zs, splits)
        return zs_split

    return _pca(embeddings)


def tsne(embeddings: np.ndarray | list[np.ndarray], seed: int = 42) -> np.ndarray | list[np.ndarray]:
    """Project embeddings into 2D using t-SNE.

    Args:
        embeddings: 2D array where each row is a data point or list.
        seed: Seed for reproducibility in t-SNE.

    Returns:
        2D array of shape (n_samples, 2) or list.

    Notes:
        t-SNE is a nonlinear technique that preserves local neighborhood structure by minimizing KL-divergence between high-dimensional and low-dim similarity distributions.
    """

    def _tsne(embeds: np.ndarray) -> np.ndarray:
        return TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto").fit_transform(embeds)

    if isinstance(embeddings, list):
        embeddings, splits = _prepare_data_groups(embeddings)
        zs = _tsne(embeddings)
        zs_split = np.split(zs, splits)
        return zs_split

    return _tsne(embeddings)


def umap(embeddings: np.ndarray | list[np.ndarray], seed: int = 42) -> np.ndarray | list[np.ndarray]:
    """Project embeddings into 2D using UMAP.

    Args:
        embeddings: 2D array where each row is a data point.
        seed: Seed for reproducibility in UMAP.

    Returns:
        2D array of shape (n_samples, 2) or list.

    Notes:
        UMAP is a nonlinear manifold learning method that preserves both local and some global structure, offering speed and scalability comparable to or better than t-SNE.
    """

    def _umap(embeds: np.ndarray) -> np.ndarray:
        return UMAP(n_components=2, random_state=seed).fit_transform(embeds)

    if isinstance(embeddings, list):
        embeddings, splits = _prepare_data_groups(embeddings)
        zs = _umap(embeddings)
        zs_split = np.split(zs, splits)
        return zs_split

    return _umap(embeddings)


def _prepare_data_groups(data_groups: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    """Stacks a list of 2D arrays and returns the combined array and group split indices."""
    processed: list[np.ndarray] = []
    lengths: list[int] = []
    for arr in data_groups:
        X = np.asarray(arr)
        if X.ndim != 2:
            raise ValueError("Each group must be a 2D array of shape (n_samples, n_features).")
        processed.append(X)
        lengths.append(X.shape[0])
    combined = np.vstack(processed)
    split_indices: list[int] = np.cumsum(lengths)[:-1].tolist()
    return combined, split_indices
