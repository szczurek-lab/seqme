from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


def plot_hist(
    data: np.ndarray,
    xlabel: str,
    color: str = "#68d6bc",
    ytype: Literal["frequency", "density"] = "density",
    figsize: tuple[int, int] = (4, 3),
    bins: int = 10,
    alpha: float = 1.0,
    edgecolor: str = "black",
    linewidth: float = 1.0,
    label: str | None = None,
    ax: Axes | None = None,
):
    """Plot a histogram from array-like input.

    Args:
        data: Input values to plot.
        xlabel: Label for the x-axis.
        color: Fill color for the bars.
        ytype: Whether to show 'frequency' or 'density'.
        figsize: Size of the figure (if no Axes provided).
        bins: Number of histogram bins.
        alpha: Transparency level of the bars.
        edgecolor: Color of the bar edges.
        linewidth: Width of the bar edges.
        label: Label for the plot legend.
        ax: Optional matplotlib Axes to plot on.

    Raises:
        ValueError: If the input array is empty.
    """
    arr = np.asarray(data).ravel()
    if arr.size == 0:
        raise ValueError("Input array is empty.")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    ylabels = {"density": "Density", "frequency": "Frequency"}

    ax.hist(
        arr,
        bins=bins,
        color=color,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
        density=(ytype == "density"),
        label=label,
    )

    if label is not None:
        ax.legend()

    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabels[ytype])

    if created_fig:
        plt.tight_layout()
        plt.show()


def plot_kde(
    data: np.ndarray,
    xlabel: str,
    color: str = "#68d6bc",
    figsize: tuple[int, int] = (4, 3),
    bandwidth: float | Literal["scott", "silverman"] | None = None,
    num_points: int = 200,
    linewidth: float = 1.0,
    alpha: float = 0.8,
    label: str | None = None,
    xlim: tuple[float, float] | None = None,
    ax: Axes | None = None,
):
    """Plot a kernel density estimate (KDE) from array-like input.

    Args:
        data: Input values to estimate density from.
        xlabel: Label for the x-axis.
        color: Fill color under the KDE curve.
        figsize: Size of the figure (if no Axes provided).
        bandwidth: Bandwidth method for the KDE.
        num_points: Number of points to evaluate the KDE on.
        linewidth: Width of the KDE curve line.
        alpha: Transparency level for the curve and fill.
        label: Label for the plot legend.
        xlim: Optional x-axis limits as a tuple [min, max].
        ax: Optional matplotlib Axes to plot on.

    Raises:
        ValueError: If the input array is empty.
    """
    arr = np.asarray(data).ravel()
    if arr.size == 0:
        raise ValueError("Input array is empty.")

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        created_fig = True

    kde_est = gaussian_kde(arr, bw_method=bandwidth)

    x_min, x_max = arr.min(), arr.max()
    x_vals = np.linspace(x_min, x_max, num_points)
    y_vals = kde_est(x_vals)

    ax.plot(x_vals, y_vals, color="black", linewidth=linewidth, alpha=alpha)
    ax.fill_between(x_vals, y_vals, color=color, alpha=alpha, label=label)

    if label is not None:
        ax.legend()

    # ax.set_axisbelow(True)
    # ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])

    if created_fig:
        plt.tight_layout()
        plt.show()


def plot_violin(
    data: np.ndarray,
    xlabel: str,
    color: str = "#68d6bc",
    figsize: tuple[int, int] = (4, 3),
    bandwidth: float | Literal["scott", "silverman"] | None = None,
    widths: float = 0.7,
    edge_color: str = "black",
    linewidth: float = 1.0,
    alpha: float = 0.8,
    show_means: bool = False,
    show_medians: bool = True,
    ax: Axes | None = None,
):
    """Plot a violin plot from array-like input.

    Args:
        data: Input values to visualize.
        xlabel: Label for the x-axis.
        color: Fill color for the violin body.
        figsize: Size of the figure (if no Axes provided).
        bandwidth: Bandwidth method for the KDE.
        widths: Width of the violin body.
        edge_color: Outline color of the violin.
        linewidth: Width of the violin edge lines.
        alpha: Transparency of the fill color.
        show_means: Whether to show the mean marker.
        show_medians: Whether to show the median marker.
        ax: Optional matplotlib Axes to plot on.

    Raises:
        ValueError: If the input array is empty.
    """
    arr = np.asarray(data).ravel()
    if arr.size == 0:
        raise ValueError("Input array is empty.")

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        created_fig = True

    parts = ax.violinplot(
        arr,
        vert=True,
        widths=widths,
        bw_method=bandwidth,
        showmeans=show_means,
        showmedians=show_medians,
    )

    for body in parts["bodies"]:  # type: ignore
        body.set_facecolor(color)
        body.set_edgecolor(edge_color)
        body.set_alpha(alpha)
        body.set_linewidth(linewidth)

    for key in ("cmeans", "cmedians", "cbars"):
        if key in parts:
            parts[key].set_edgecolor(edge_color)
            parts[key].set_linewidth(linewidth)

    ax.set_ylabel(xlabel)
    ax.set_xticks([])  # no category axis

    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

    if created_fig:
        plt.tight_layout()
        plt.show()


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


def plot_embeddings(
    projections: list[np.ndarray],
    *,
    colors: list[str] | None = None,
    labels: list[str] | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    xlabel: str = "dim1",
    ylabel: str = "dim2",
    figsize: tuple[int, int] = (4, 3),
    outline_width: float = 0,
    point_size: float = 20,
    legend_point_size: float | None = None,
    alpha: float = 0.6,
    show_ticks: bool = False,
):
    """Plot projections for one or more groups.

    Args:
        projections: List of arrays, each containing vectors to embed.
        labels: Labels corresponding to each set in data.
        colors: Colors for each group of points.
        title: Optional plot title.
        ax: Optional matplotlib Axes to plot on.
        xlabel: x-axis label.
        ylabel: y-axis label.
        figsize: Size of the figure (if no Axes provided).
        outline_width: Width of the outline around points.
        point_size: Size of scatter points.
        legend_point_size: Size of scatter points in the legend.
        alpha: Transparency of points.
        show_ticks: Whether to show axis ticks.
    """
    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        created_fig = True

    for i, seg in enumerate(projections):
        color = colors[i] if colors is not None else None
        label = labels[i] if labels is not None else None
        ax.scatter(
            seg[:, 0],
            seg[:, 1],
            label=label,
            c=color,
            s=point_size,
            alpha=alpha,
            edgecolor="black",
            linewidth=outline_width,
        )

        if labels is not None:
            leg = ax.legend(frameon=True)

            if legend_point_size is not None:
                for lh in leg.legend_handles:
                    lh.set_sizes([legend_point_size])  # type: ignore
                    lh.set_alpha(1.0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if title is not None:
        ax.set_title(title)

    if created_fig:
        plt.tight_layout()
        plt.show()


def plot_embedding_with_value(
    projections: np.ndarray,
    *,
    values: np.ndarray | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    xlabel: str = "dim1",
    ylabel: str = "dim2",
    figsize: tuple[int, int] = (4, 3),
    point_size: int = 20,
    alpha: float = 0.6,
    outline_width: float = 0.4,
    show_ticks: bool = False,
):
    """Plot projections and color by value.

    Args:
        projections: Arrays of projected embeddings.
        values: Attribute values.
        title: Optional plot title.
        ax: Optional matplotlib Axes to plot on.
        figsize: Size of the figure (if no Axes provided).
        xlabel: x-axis label.
        ylabel: y-axis label.
        figsize: Size of figure.
        point_size: Size of scatter points.
        alpha: Transparency of points.
        outline_width: Width of the outline around points.
        show_ticks: Whether to show axis ticks.
    """
    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        created_fig = True

    sc = ax.scatter(
        projections[:, 0],
        projections[:, 1],
        label=None,
        c=values,
        s=point_size,
        alpha=alpha,
        edgecolor="black",
        linewidth=outline_width,
    )
    ax.figure.colorbar(sc, ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if title is not None:
        ax.set_title(title)

    if created_fig:
        plt.tight_layout()
        plt.show()
