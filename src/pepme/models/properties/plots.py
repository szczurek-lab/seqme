from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde


def hist(
    data: np.ndarray,
    xlabel: str,
    color: str = "#68d6bc",
    ytype: Literal["frequency", "density"] = "density",
    figsize: tuple[int, int] = (4, 3),
    bins: int = 10,
    alpha: float = 1.0,
    edgecolor: str = "black",
    linewidth: float = 1.0,
    ax: Axes | None = None,
):
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
    )

    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabels[ytype])

    if created_fig:
        plt.tight_layout()
        plt.show()


def kde(
    data: np.ndarray,
    xlabel: str,
    color: str = "#68d6bc",
    figsize: tuple[int, int] = (4, 3),
    bw_method=None,
    num_points: int = 200,
    linewidth: float = 1.0,
    alpha: float = 0.8,
    ax: Axes | None = None,
):
    arr = np.asarray(data).ravel()
    if arr.size == 0:
        raise ValueError("Input array is empty.")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    kde_est = gaussian_kde(arr, bw_method=bw_method)

    x_min, x_max = arr.min(), arr.max()
    x_vals = np.linspace(x_min, x_max, num_points)
    y_vals = kde_est(x_vals)

    ax.plot(x_vals, y_vals, color="black", linewidth=linewidth, alpha=alpha)
    ax.fill_between(x_vals, y_vals, color=color, alpha=alpha)

    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")

    if created_fig:
        plt.tight_layout()
        plt.show()


def violin(
    data: np.ndarray,
    xlabel: str,
    color: str = "#68d6bc",
    figsize: tuple[int, int] = (4, 3),
    bw_method=None,
    widths: float = 0.7,
    edge_color: str = "black",
    linewidth: float = 1.0,
    alpha: float = 0.8,
    show_means: bool = False,
    show_medians: bool = True,
    ax: Axes | None = None,
):
    """
    Draws a vertical violin plot (boxplot-like but keeps the full KDE shape).

    If `ax` is provided, draws into that Axes; otherwise creates its own figure.
    """
    arr = np.asarray(data).ravel()
    if arr.size == 0:
        raise ValueError("Input array is empty.")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    parts = ax.violinplot(
        arr,
        vert=True,
        widths=widths,
        bw_method=bw_method,
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
