from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


def plot_embeddings(
    embeddings: np.ndarray | list[np.ndarray],
    *,
    values: (str | np.ndarray) | (list[str] | list[np.ndarray]) | None = None,
    colors: str | list[str] | None = None,
    cmap: str | None = None,
    title: str | None = None,
    xlabel: str = "dim1",
    ylabel: str = "dim2",
    outline_width: float = 0,
    point_size: float = 20,
    show_legend: bool = True,
    legend_point_size: float | None = 20,
    alpha: float = 0.6,
    show_ticks: bool = False,
    legend_loc: Literal["right margin"] | str | None = "right margin",
    figsize: tuple[int, int] = (4, 3),
    ax: Axes | None = None,
):
    """Plot projections for one or more groups.

    Args:
        embeddings: Groups of arrays, each containing 2d embeddings.
        values: Either group names or values for each individual embedding.
        colors: Colors for each group of points.
        cmap: Colors used for values.
        title: Optional plot title.
        xlabel: x-axis label.
        ylabel: y-axis label.
        outline_width: Width of the outline around points.
        point_size: Size of scatter points.
        show_legend: Whether to show legend (only for categorical data).
        legend_point_size: Size of scatter points in the legend.
        alpha: Transparency of points.
        show_ticks: Whether to show axis ticks.
        legend_loc: Legend location.
        figsize: Size of the figure (if no Axes provided).
        ax: Optional matplotlib Axes to plot on.
    """
    # try making the parameters lists then parse those normally.

    if isinstance(embeddings, np.ndarray):
        embeddings = [embeddings]

    if isinstance(values, str) or isinstance(values, np.ndarray):
        values = [values]  # type: ignore

    if isinstance(colors, str):
        colors = [colors]

    embeddings = list(embeddings)
    values = list(values) if values else None  # type: ignore
    colors = list(colors) if colors else None

    for projection in embeddings:
        if projection.ndim != 2:
            raise ValueError(
                f"All projection groups should have two dimensions [embeddings, 2], but a group has {projection.ndim} dimensions."
            )
        if projection.shape[-1] != 2:
            raise ValueError(f"Only 2D embeddings can be plotted, but got {projection.shape[-1]}D embeddings.")

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        created_fig = True

    if values:
        if isinstance(values[0], np.ndarray):
            group = np.vstack(embeddings)
            c = np.vstack(values)
            sc = ax.scatter(
                group[:, 0],
                group[:, 1],
                c=c,
                s=point_size,
                alpha=alpha,
                edgecolor="black",
                linewidth=outline_width,
                cmap=cmap,
            )
            ax.figure.colorbar(sc, ax=ax)
        else:
            if len(values) != len(embeddings):
                raise ValueError(
                    f"'group_or_values' has {len(values)} groups (elements). 'projections' has {len(embeddings)} list elements. Required the same sizes."
                )

            if colors:
                if len(colors) != len(values):
                    raise ValueError(
                        f"'group_colors' has {len(colors)} list elements. 'group_or_values' has {len(values)} list elements. Required the same sizes."
                    )

            for i, group in enumerate(embeddings):
                ax.scatter(
                    group[:, 0],
                    group[:, 1],
                    label=values[i],
                    c=colors[i] if colors else None,
                    s=point_size,
                    alpha=alpha,
                    edgecolor="black",
                    linewidth=outline_width,
                )

            if show_legend:
                if legend_loc == "right margin":
                    leg = ax.legend(
                        frameon=False,
                        loc="center left",
                        bbox_to_anchor=(1, 0.5),
                        ncol=(1 if len(embeddings) <= 14 else 2 if len(embeddings) <= 30 else 3),
                    )
                else:
                    leg = ax.legend(loc=legend_loc)

                for lh in leg.legend_handles:
                    lh.set_alpha(1.0)

                    if legend_point_size is not None:
                        lh.set_sizes([legend_point_size])  # type: ignore

    else:
        for i, group in enumerate(embeddings):
            ax.scatter(
                group[:, 0],
                group[:, 1],
                c=colors[i] if colors else None,
                s=point_size,
                alpha=alpha,
                edgecolor="black",
                linewidth=outline_width,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    if title is not None:
        ax.set_title(title)

    if created_fig:
        plt.show()


def pca(embeddings: np.ndarray | list[np.ndarray], seed: int | None = 0) -> np.ndarray | list[np.ndarray]:
    """Project embeddings into 2D using PCA.

    Args:
        embeddings: 2D array where each row is a data point.
        seed: Seed for deterministic computation of PCA.

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


def tsne(embeddings: np.ndarray | list[np.ndarray], seed: int | None = 0) -> np.ndarray | list[np.ndarray]:
    """Project embeddings into 2D using t-SNE.

    Args:
        embeddings: 2D array where each row is a data point or list.
        seed: Seed for deterministic computation of t-SNE.

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


def umap(embeddings: np.ndarray | list[np.ndarray], seed: int | None = 0) -> np.ndarray | list[np.ndarray]:
    """Project embeddings into 2D using UMAP.

    Args:
        embeddings: 2D array where each row is a data point.
        seed: Seed for deterministic computation of UMAP.

    Returns:
        2D array of shape (n_samples, 2) or list.

    Notes:
        UMAP is a nonlinear manifold learning method that preserves both local and some global structure, offering speed and scalability comparable to or better than t-SNE.
    """

    def _umap(embeds: np.ndarray) -> np.ndarray:
        return UMAP(n_components=2, n_jobs=1 if seed is not None else None, random_state=seed).fit_transform(embeds)

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
