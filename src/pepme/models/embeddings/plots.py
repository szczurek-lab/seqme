from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from sklearn.decomposition import PCA  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from umap import UMAP  # type: ignore


def plot_pca(
    data: list[np.ndarray],
    names: list[str],
    colors: list[str],
    ax: Optional[Axes] = None,
    figsize: tuple[int, int] = (4, 3),
    point_size: int = 20,
    alpha: float = 0.6,
    seed: int = 42,
):
    """
    Plots PCA of one or multiple groups on the same axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    reducer = PCA(n_components=2, random_state=seed)

    _plot_reduction(
        reducer,
        data,
        names,
        colors,
        xlabel="PC 1",
        ylabel="PC 2",
        ax=ax,
        point_size=point_size,
        alpha=alpha,
    )
    if ax.figure:
        ax.figure.tight_layout()  # type: ignore


def plot_tsne(
    data: list[np.ndarray],
    names: list[str],
    colors: list[str],
    ax: Optional[Axes] = None,
    figsize: tuple[int, int] = (4, 3),
    point_size: int = 20,
    alpha: float = 0.6,
    seed: int = 42,
):
    """
    Plots t-SNE of one or multiple groups on the same axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    reducer = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto")
    _plot_reduction(
        reducer,
        data,
        names,
        colors,
        xlabel="t-SNE 1",
        ylabel="t-SNE 2",
        ax=ax,
        point_size=point_size,
        alpha=alpha,
    )
    if ax.figure:
        ax.figure.tight_layout()  # type: ignore


def plot_umap(
    data: list[np.ndarray],
    names: list[str],
    colors: list[str],
    ax: Optional[Axes] = None,
    figsize: tuple[int, int] = (4, 3),
    point_size: int = 20,
    alpha: float = 0.6,
    seed: int = 42,
):
    """
    Plots UMAP of one or multiple groups on the same axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    reducer = UMAP(n_components=2, random_state=seed)
    _plot_reduction(
        reducer,
        data,
        names,
        colors,
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        ax=ax,
        point_size=point_size,
        alpha=alpha,
    )
    if ax.figure:
        ax.figure.tight_layout()  # type: ignore


def _prepare_data_groups(data_groups: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    """
    Stacks a list of 2D arrays and returns the combined array and group split indices.
    """
    processed: list[np.ndarray] = []
    lengths: list[int] = []
    for arr in data_groups:
        X = np.asarray(arr)
        if X.ndim != 2:
            raise ValueError(
                "Each group must be a 2D array of shape (n_samples, n_features)."
            )
        processed.append(X)
        lengths.append(X.shape[0])
    combined = np.vstack(processed)
    split_indices: list[int] = np.cumsum(lengths)[:-1].tolist()
    return combined, split_indices


def _plot_reduction(
    reducer,
    data: list[np.ndarray],
    names: list[str],
    colors: list[str],
    xlabel: str,
    ylabel: str,
    ax: Axes,
    point_size: int,
    alpha: float,
):
    # Normalize data_groups to list
    data_groups = data
    if len(names) != len(data_groups) or len(colors) != len(data_groups):
        raise ValueError("`names` and `colors` must match number of groups.")

    X_all, splits = _prepare_data_groups(data_groups)
    X2 = reducer.fit_transform(X_all)
    segments = np.split(X2, splits)

    for seg, name, color in zip(segments, names, colors):
        ax.scatter(
            seg[:, 0],
            seg[:, 1],
            label=name,
            c=color,
            s=point_size,
            alpha=alpha,
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
