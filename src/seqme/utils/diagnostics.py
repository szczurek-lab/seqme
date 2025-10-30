from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors


def knn_alignment_score(xs: np.ndarray, labels: np.ndarray, n_neighbors: int = 5) -> float:
    """Compute the k-NN feature alignment score of an embedding model.

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


def spearman_alignment_score(xs_a: np.ndarray, xs_b: np.ndarray) -> float:
    """Compute the Spearman correlation coefficient using the pairwise distance between embedding spaces.

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


def plot_knn_alignment_score(
    xs: np.ndarray,
    labels: np.ndarray,
    n_neighbors: list[int],
    label: str | None = None,
    legend_loc: Literal["right margin"] | str | None = "right margin",
    figsize: tuple[int, int] = (4, 3),
    ax: Axes | None = None,
):
    """
    Plot the k-NN feature alignment score of an embedding model using variable-number of neighbors.

    Args:
        xs: Sequence embeddings.
        labels: Label for each sequence.
        n_neighbors: Number of neighbors used by k-NN.
        label: Model name.
        legend_loc: Legend location.
        figsize: Size of the figure.
        ax: Optional Axes.

    """
    scores = [knn_alignment_score(xs, labels, k) for k in n_neighbors]

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        created_fig = True

    ax.plot(n_neighbors, scores, label=label)
    ax.set_xlabel("N_neighbors")
    ax.set_ylabel("Score")
    ax.set_title("Feature alignment score")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_ylim(0, 1)

    if legend_loc == "right margin":
        n_labels = len(np.unique(labels))
        ax.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            ncol=(1 if n_labels <= 14 else 2 if n_labels <= 30 else 3),
        )
    else:
        ax.legend(loc=legend_loc)

    if created_fig:
        plt.show()
