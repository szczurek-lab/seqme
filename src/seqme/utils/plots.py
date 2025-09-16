from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde


def plot_hist(
    data: np.ndarray,
    xlabel: str,
    *,
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
    *,
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

    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

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
    *,
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


def plot_2d_embeddings(
    embeddings: np.ndarray | list[np.ndarray],
    *,
    values: (str | np.ndarray) | (list[str] | list[np.ndarray]) | None = None,
    colors: str | list[str] | None = None,
    cmap: str | None = None,
    title: str | None = None,
    xlabel: str = "dim1",
    ylabel: str = "dim2",
    figsize: tuple[int, int] = (4, 3),
    outline_width: float = 0,
    point_size: float = 20,
    show_legend: bool = True,
    legend_point_size: float | None = None,
    alpha: float = 0.6,
    show_ticks: bool = False,
    ax: Axes | None = None,
):
    """Plot projections for one or more groups.

    Args:
        embeddings: Groups of arrays, each containing 2d embeddings.
        values: Either group names or values for each individual embedding.
        colors: Colors for each group of points.
        cmap: Colors used for values.
        title: Optional plot title.
        ax: Optional matplotlib Axes to plot on.
        xlabel: x-axis label.
        ylabel: y-axis label.
        figsize: Size of the figure (if no Axes provided).
        outline_width: Width of the outline around points.
        point_size: Size of scatter points.
        show_legend: Whether to show legend (only for categorical data)
        legend_point_size: Size of scatter points in the legend.
        alpha: Transparency of points.
        show_ticks: Whether to show axis ticks.
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
                    leg = ax.legend(frameon=True)

                    if legend_point_size is not None:
                        for lh in leg.legend_handles:
                            lh.set_sizes([legend_point_size])  # type: ignore
                            lh.set_alpha(1.0)
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
        plt.tight_layout()
        plt.show()
