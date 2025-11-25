from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes


def plot_bar(
    df: pd.DataFrame,
    metric: str | None = None,
    *,
    color: str = "#68d6bc",
    xticks_rotation: float = 45,
    ylim: tuple[float, float] | None = None,
    show_arrow: bool = True,
    show_deviation: bool = True,
    figsize: tuple[int, int] = (4, 3),
    ax: Axes | None = None,
):
    """Plot a bar chart for a given metric with optional error bars.

    Args:
        df: A DataFrame with a MultiIndex column [metric, {"value", "deviation"}].
        metric: The name of the metric to plot. If ``None``, plot all metrics in ``df``, assumes one metric is in the dataframe.
        color: Bar color. Default is teal.
        xticks_rotation: Rotation angle for x-axis labels.
        ylim: y-axis limits (optional).
        show_arrow: Whether to show an arrow indicating maximize/minimize in the x-labels.
        show_deviation: Whether to plot the deviation if available.
        figsize: Size of the figure.
        ax: Optional matplotlib Axes to plot on.
    """
    available_metrics = list(df.columns.get_level_values(0).unique())
    if metric is None:
        if len(available_metrics) > 1:
            raise ValueError("Specify the metric to use.")

        metric = available_metrics[0]

    if metric not in available_metrics:
        raise ValueError(f"'{metric}' is not a column in the DataFrame.")

    values = df[(metric, "value")]
    deviations = df[(metric, "deviation")]

    # filter NaN values
    valid_mask = values.notna()
    values = values[valid_mask]
    deviations = deviations[valid_mask]

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        created_fig = True

    bar_names = (
        [" ".join(map(str, row_index)) for row_index in values.index.to_flat_index()]
        if df.index.nlevels > 1
        else values.index
    )

    ax.bar(bar_names, values, color=color, edgecolor="black")

    if show_deviation:
        ax.errorbar(bar_names, values, yerr=deviations, fmt="none", ecolor="black", capsize=4, lw=1)

    arrows = {"maximize": "↑", "minimize": "↓"}
    arrow = arrows[df.attrs["objective"][metric]]

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(bar_names, rotation=xticks_rotation, ha="center")

    ax.set_ylabel(f"{metric}{arrow}" if show_arrow else metric)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    if created_fig:
        plt.show()


def plot_parallel(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    *,
    n_decimals: int | list[int] = 2,
    xticks_fontsize: float | None = None,
    xticks_rotation: float = 90,
    yticks_fontsize: float = 8,
    show_yticks: bool = True,
    show_arrow: bool = True,
    arrow_size: float | None = None,
    zero_width: float | None = 0.25,
    xpad: float = 0.25,
    legend_loc: Literal["right margin"] | str | None = "right margin",
    figsize: tuple[int, int] = (5, 3),
    ax: Axes | None = None,
):
    """Plot a parallel coordinates plot where each coordinate is a metric.

    Args:
        df: A DataFrame with a MultiIndex column [metric, {"value", "deviation"}].
        metrics: Which metrics to plot. If ``None``, plot all metrics in ``df``.
        n_decimals: Decimal precision for formatting metric values.
        xticks_fontsize: Font size of x-ticks. If ``None``, selects default fontsize.
        xticks_rotation: Rotation angle for x-axis tick labels.
        yticks_fontsize: Font size of y-ticks.
        show_yticks: Whether to you show the minimum and maximum value on the y-axis for each metric.
        show_arrow: Whether to show an arrow indicating maximize/minimize in the x-labels.
        arrow_size: Size of arrows displayed in the plot. If ``None``, do not show.
        zero_width: Width of the zero value indicator. If ``None``, do not show.
        xpad: Left and right padding of axes.
        legend_loc: Legend location.
        figsize: Size of the figure.
        ax: Optional matplotlib Axes to plot on.
    """
    if metrics is None:
        metrics = list(df.columns.get_level_values(0).unique())

    if len(metrics) < 2:
        raise ValueError("Expected at least two metrics.")

    for metric in metrics:
        if df[(metric, "value")].isna().any():
            raise ValueError(f"'{metric}' has NaN values.")

    n_metrics = len(metrics)
    n_decimals = [n_decimals] * n_metrics if isinstance(n_decimals, int) else n_decimals

    if len(n_decimals) != n_metrics:
        raise ValueError(
            f"Expected {n_metrics} decimals, got {len(n_decimals)}. Provide a single int or a list matching the number of metrics."
        )

    names = (
        [" ".join(map(str, row_index)) for row_index in df.index.to_flat_index()]
        if df.index.nlevels > 1
        else list(df.index)
    )

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        created_fig = True

    ax.set_xlim(0 - xpad, n_metrics - 1 + xpad)
    ax.set_xticks(range(n_metrics))

    ax.set_yticklabels([])

    ax.grid(True, axis="x", linewidth=1.0, color="black", linestyle="-", alpha=0.3)
    ax.grid(True, axis="y", linewidth=0.8, color="gray", linestyle="--", alpha=0.2)

    objectives = df.attrs["objective"]

    # Normalize each metric separately
    normalized = {}
    ranges = {}
    for m in metrics:
        vals = df[(m, "value")].values
        vmin, vmax = vals.min(), vals.max()
        ranges[m] = (vmin, vmax)

        if vmax > vmin:
            normalized[m] = (vals - vmin) / (vmax - vmin)
        else:
            normalized[m] = np.ones_like(vals) if objectives[m] == "maximize" else np.zeros_like(vals)

    for i, name in enumerate(names):
        values = [normalized[m][i] for m in metrics]
        ax.plot(values, label=name)

    if zero_width:
        for i, m in enumerate(metrics):
            vmin, vmax = ranges[m]
            if vmin <= 0 <= vmax:
                ax.hlines(
                    y=-vmin / (vmax - vmin) if vmax > vmin else 1.0 if objectives[m] == "maximize" else 0.0,
                    xmin=i - zero_width / 2,
                    xmax=i + zero_width / 2,
                    color="gray",
                    linewidth=1.1,
                    alpha=0.8,
                )

    for i, m in enumerate(metrics):
        vmin, vmax = ranges[m]
        ax2 = ax.twinx()
        ax2.set_ylim(0, 1)
        ax2.yaxis.set_ticks_position("left")
        ax2.set_yticks([])
        ax2.spines["left"].set_position(("data", i))
        ax2.spines["left"].set_visible(False)  # Hide duplicate spines

    if legend_loc == "right margin":
        ax.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            ncol=(1 if len(names) <= 14 else 2 if len(names) <= 30 else 3),
        )
    else:
        ax.legend(loc=legend_loc)

    arrows = {"maximize": "↑", "minimize": "↓"}
    xlabels = [f"{m}{arrows[objectives[m]]}" if show_arrow else m for m in metrics]
    ax.set_xticklabels(xlabels, rotation=xticks_rotation, ha="center", va="top", fontsize=xticks_fontsize)

    if arrow_size is not None:
        for i, m in enumerate(metrics):
            vmin, vmax = ranges[m]
            y_min, y_max = ax.get_ylim()

            ax.text(
                i,
                1.0 if objectives[m] == "maximize" else 0.0,
                f"{arrows[objectives[m]]}",
                ha="center",
                va="top" if objectives[m] == "maximize" else "bottom",
                fontsize=arrow_size,
                color="black",
                clip_on=False,
            )

    if show_yticks:
        y_min, y_max = ax.get_ylim()

        y_offset_top = 0.05 * (y_max - y_min) / figsize[1]
        y_offset_bottom = 0.1 * (y_max - y_min) / figsize[1]

        x_label_y_pad = 0.5
        auto_pad = yticks_fontsize + y_offset_bottom + x_label_y_pad
        ax.tick_params(axis="x", pad=auto_pad)

        for i, m in enumerate(metrics):
            vmin, vmax = ranges[m]

            ax.text(
                i,
                y_max + y_offset_top,
                f"{vmax:.{n_decimals[i]}f}",
                ha="center",
                va="bottom",
                fontsize=yticks_fontsize,
                color="black",
                clip_on=False,
                fontweight="bold" if objectives[m] == "maximize" else None,
            )

            ax.text(
                i,
                y_min - y_offset_bottom,
                f"{vmin:.{n_decimals[i]}f}",
                ha="center",
                va="top",
                fontsize=yticks_fontsize,
                color="black",
                clip_on=False,
                fontweight="bold" if objectives[m] == "minimize" else None,
            )

    if created_fig:
        plt.show()


def plot_line(
    df: pd.DataFrame,
    metric: str | None = None,
    *,
    color: list[str] | None = None,
    xlabel: str = "Iteration",
    linestyle: str | list[str] = "-",
    show_arrow: bool = True,
    marker: str | None = "x",
    marker_size: float | None = None,
    show_deviation: bool = True,
    deviation_alpha: float = 0.4,
    legend_loc: Literal["right margin"] | str | None = "right margin",
    figsize: tuple[int, int] = (4, 3),
    ax: Axes | None = None,
):
    """Plot a series for a given metric across multiple iterations/steps with optional error bars.

    Args:
        df: A DataFrame with a MultiIndex column [metric, {"value", "deviation"}].
        metric: The name of the metric to plot. If ``None``, plot all metrics in ``df``, assumes one metric is in the dataframe.
        color: Color for each series.
        xlabel: Name of x-label.
        linestyle: Series linestyle.
        show_arrow: Whether to show an arrow indicating maximize/minimize.
        marker: Marker type for serie values. If ``None``, no marker is shown.
        marker_size: Size of marker. If ``None``, auto-selects size.
        show_deviation: Whether to the plot deviation if available.
        deviation_alpha: opacity level of deviation intervals.
        legend_loc: Legend location.
        figsize: Size of the figure.
        ax: Optional matplotlib Axes to plot on.
    """
    available_metrics = list(df.columns.get_level_values(0).unique())
    if metric is None:
        if len(available_metrics) > 1:
            raise ValueError("Specify the metric to use.")

        metric = available_metrics[0]

    if metric not in df.columns.get_level_values(0):
        raise ValueError(f"'{metric}' is not a column in the DataFrame.")

    if df.index.nlevels != 2:
        raise ValueError("sequences should have tuple names: (model name, iteration).")

    for model_name, iteration in df.index:
        if not isinstance(iteration, int | float):
            raise ValueError(
                "Expected a tuple of type (Any, int | float), "
                f"but got ({model_name}, {iteration}) "
                f"with types ({type(model_name).__name__}, {type(iteration).__name__})."
            )

    model_names = list(df.index.get_level_values(0).unique())
    linestyle = [linestyle] * len(model_names) if isinstance(linestyle, str) else linestyle

    if len(linestyle) != len(model_names):
        f"Expected {len(model_names)} linestyles, got {len(linestyle)}. Provide a single linestyle or a list matching the number of entries."

    if color:
        if len(color) != len(model_names):
            raise ValueError(f"Expected a color for each entry. Got {len(color)}, expected {len(model_names)}.")

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        created_fig = True

    for i, model_name in enumerate(model_names):
        df_model = df.loc[model_name]
        df_model = df_model.sort_index()

        xs = df_model.index
        vs = df_model[(metric, "value")]
        dev = df_model[(metric, "deviation")]

        lines = ax.plot(
            xs,
            vs,
            marker=marker,
            markersize=marker_size,
            label=model_name,
            color=color[i] if color else None,
            linestyle=linestyle[i],
        )

        if show_deviation:
            c = lines[0].get_color()
            ax.fill_between(xs, vs - dev, vs + dev, alpha=deviation_alpha, color=c)

    objective = df.attrs["objective"][metric]
    arrows = {"maximize": "↑", "minimize": "↓"}

    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{metric}{arrows[objective]}" if show_arrow else metric)

    ax.grid(True, linestyle="--", alpha=0.4)

    iterations = df.index.get_level_values(1)
    if pd.api.types.is_integer_dtype(iterations.dtype):
        from matplotlib.ticker import FuncFormatter, MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(round(x))}"))

    if legend_loc == "right margin":
        ax.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            ncol=(1 if len(model_names) <= 14 else 2 if len(model_names) <= 30 else 3),
        )
    else:
        ax.legend(loc=legend_loc)

    if created_fig:
        plt.show()


def plot_scatter(
    df: pd.DataFrame,
    metrics: list[str] | tuple[str, str] | None = None,
    *,
    color: list[str] | None = None,
    show_arrow: bool = True,
    marker: str | None = "o",
    marker_size: float | None = None,
    linestyle: str | None = "--",
    linewidth: float = 1.0,
    show_deviation: bool = True,
    deviation_alpha: float = 0.5,
    deviation_linewidth: float = 1.0,
    legend_loc: Literal["right margin"] | str | None = "right margin",
    figsize: tuple[int, int] = (4, 3),
    ax: Axes | None = None,
):
    """Plot a scatter plot for two metrics with optional error rectangles or bars.

    Args:
        df: A DataFrame with a MultiIndex column [metric, {"value", "deviation"}].
        metrics: The name of the metrics to plot. If ``None``, plot all metrics in ``df``, assumes two metrics are in the dataframe.
        color: Circle color.
        show_arrow: Whether to show an arrow indicating maximize/minimize in the x- and y-labels.
        marker: Marker type for serie values. If ``None``, no marker is shown.
        marker_size: Size of marker. If ``None``, auto-selects size.
        linestyle: Series linestyle. If ``None``, dont show connecting lines.
        linewidth: Line width of connected points.
        show_deviation: Whether to plot the deviation if available.
        deviation_alpha: opacity level of deviation intervals.
        deviation_linewidth: Deviation line width.
        legend_loc: Legend location.
        figsize: Size of the figure.
        ax: Optional matplotlib Axes to plot on.
    """
    if metrics is None:
        metrics = list(df.columns.get_level_values(0).unique())

    if len(metrics) != 2:
        raise ValueError(f"Expected two metrics, got {len(metrics)}.")

    for metric in metrics:
        if metric not in df.columns.get_level_values(0):
            raise ValueError(f"'{metric}' is not a column in the DataFrame.")

    for metric in metrics:
        if df[(metric, "value")].isna().any():
            raise ValueError(f"'{metric}' has NaN values.")

    model_names = list(df.index.get_level_values(0).unique())

    if color:
        if len(color) != len(model_names):
            raise ValueError(f"Expected a color for each entry. Got {len(color)}, expected {len(model_names)}.")

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        created_fig = True

    metric1, metric2 = metrics

    for i, model_name in enumerate(model_names):
        df_model = df.loc[model_name]
        df_model = df_model.sort_index()

        xs = np.array(df_model[(metric1, "value")])
        ys = np.array(df_model[(metric2, "value")])

        sc = ax.scatter(
            xs,
            ys,
            marker=marker,
            s=marker_size,
            color=color[i] if color else None,
            label=model_name,
            zorder=2,
        )

        c = color[i] if color else sc.get_facecolors()  # type: ignore

        if show_deviation:
            xs_dev = np.array(df_model[(metric1, "deviation")])
            ys_dev = np.array(df_model[(metric2, "deviation")])

            nan_xs_dev = np.isnan(xs_dev).any()
            nan_ys_dev = np.isnan(ys_dev).any()

            if (not nan_xs_dev) and (not nan_ys_dev):
                fill_alpha = deviation_alpha * 0.5

                for i in range(xs.size):
                    w, h = xs_dev[i], ys_dev[i]
                    x, y = xs[i] - w / 2, ys[i] - h / 2
                    face_rect = mpl.patches.Rectangle(
                        (x, y), w, h, linewidth=deviation_linewidth, facecolor=c, alpha=fill_alpha, zorder=0
                    )
                    edge_rect = mpl.patches.Rectangle(
                        (x, y),
                        w,
                        h,
                        linewidth=deviation_linewidth,
                        edgecolor=c,
                        facecolor="none",
                        zorder=1,
                        alpha=deviation_alpha,
                    )
                    ax.add_patch(face_rect)
                    ax.add_patch(edge_rect)
            elif (not nan_xs_dev) or (not nan_ys_dev):
                if not nan_xs_dev:
                    p1 = np.stack([xs - xs_dev, ys], axis=1)
                    p2 = np.stack([xs + xs_dev, ys], axis=1)
                else:
                    p1 = np.stack([xs, ys - ys_dev], axis=1)
                    p2 = np.stack([xs, ys + ys_dev], axis=1)

                segments = np.stack([p1, p2], axis=1)
                lc = mpl.collections.LineCollection(
                    segments,  # type: ignore
                    colors=c,
                    linewidths=deviation_linewidth,
                    zorder=1,
                    alpha=deviation_alpha,
                )
                ax.add_collection(lc)

        if linestyle and xs.size > 1:
            ax.plot(xs, ys, color=c, linestyle=linestyle, linewidth=linewidth, zorder=0)  # type: ignore

    arrows = {"maximize": "↑", "minimize": "↓"}

    arrow1 = arrows[df.attrs["objective"][metric1]]
    ax.set_xlabel(f"{metric1}{arrow1}" if show_arrow else metric1)

    arrow2 = arrows[df.attrs["objective"][metric2]]
    ax.set_ylabel(f"{metric2}{arrow2}" if show_arrow else metric2)

    if legend_loc == "right margin":
        ax.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            ncol=(1 if len(model_names) <= 14 else 2 if len(model_names) <= 30 else 3),
        )
    else:
        ax.legend(loc=legend_loc)

    ax.grid(axis="both", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    if created_fig:
        plt.show()
