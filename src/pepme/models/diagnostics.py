import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


def feature_alignment_score(xs: np.ndarray, labels: np.ndarray, n_neighbors: int = 5) -> float:
    """
    Compute the feature alignment score of an embedding model.

    Args:
        xs: Sequence embeddings.
        labels: Group label for each sequence.
        n_neighbors: Number of neighbors used by k-NN.

    Returns:
        Feature alignment score between [0, 1].

    """
    nearest_neighbour = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1).fit(xs)
    closest_indices = nearest_neighbour.kneighbors(return_distance=False)
    matches = (labels[closest_indices] == labels[:, None]).sum()
    total = labels.shape[0] * n_neighbors
    score = matches / total
    return score.item()


def spearman_correlation_coefficient(xs_a: np.ndarray, xs_b: np.ndarray) -> float:
    """
    Compute the Spearman correlation coefficient using the pairwise distance between embedding spaces.

    Args:
        xs_a: Sequence embeddings for space A.
        xs_b: Sequence embeddings for space B.

    Returns:
        Spearman correlation coefficient between [-1, 1].

    """
    dists_a = cdist(xs_a, xs_a).ravel()
    dists_b = cdist(xs_b, xs_b).ravel()
    res = spearmanr(dists_a, dists_b)
    return float(res.statistic)


def plot_feature_alignment_score(
    xs: np.ndarray,
    labels: np.ndarray,
    n_neighbors: list[int],
    label: str = None,
    ax: Axes = None,
):
    """
    Plot the feature alignment score of an embedding model using variable-number of neighbors.

    Args:
        xs: Sequence embeddings.
        labels: Group label for each sequence.
        n_neighbors: Number of neighbors used by k-NN.
        label: Model name.
        ax: Optional Axes.

    """
    scores = [feature_alignment_score(xs, labels, k) for k in n_neighbors]

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    ax.plot(n_neighbors, scores, label=label)
    ax.legend()
    ax.set_xlabel("N_neighbors")
    ax.set_ylabel("Score")
    ax.set_title("Feature alignment score")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_ylim(0, 1)
