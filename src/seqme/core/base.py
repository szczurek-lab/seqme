import abc
import pickle
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from pandas.io.formats.style import Styler
from pylatex import NoEscape, Table, Tabular
from tqdm import tqdm


@dataclass
class MetricResult:
    """Data structure to store a metric result."""

    value: float | int
    deviation: float | None = None


class Metric(abc.ABC):
    """Abstract base class for defining a metric.

    Subclasses implement a callable interface to compute a score and
    specify a name and optimization direction.
    """

    @abc.abstractmethod
    def __call__(self, sequences: list[str]) -> MetricResult:
        """Calculate the metric for the provided sequences.

        Args:
            sequences: Text inputs to evaluate.

        Returns:
            An object containing the score and optional details.
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """A short identifier for this metric, used in reporting.

        Returns:
            The metric name.
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def objective(self) -> Literal["minimize", "maximize"]:
        """Whether lower or higher scores indicate better performance.

        Returns:
            The optimization goal ('minimize' or 'maximize').
        """
        raise NotImplementedError()


def evaluate(
    sequences: dict[str, list[str]] | dict[tuple[str, ...], list[str]],
    metrics: list[Metric],
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute a set of metrics for multiple sequence groups.

    Args:
        sequences: A dict mapping group name to lists of sequences.
        metrics: Metrics to compute per sequence group.
        verbose: Whether to show a progress-bar.

    Returns:
        A DataFrame where each row corresponds to a sequence group (dict key)
        and columns are a MultiIndex [metric_name, {"value", "deviation"}].
    """
    if len(metrics) == 0:
        raise ValueError("No metrics provided")

    metric_names = [m.name for m in metrics]
    metric_duplicates = [name for name, count in Counter(metric_names).items() if count > 1]
    if len(metric_duplicates) > 0:
        duplicate_names = ", ".join(metric_duplicates)
        raise ValueError(f"Metrics must have unique names. Found duplicates: {duplicate_names}.")

    for group_name, seqs in sequences.items():
        if len(seqs) == 0:
            raise ValueError(f"'{group_name}' has no sequences.")

    # Prepare nested results: model -> metric -> {value, deviation}
    nested = {}
    total = len(sequences) * len(metrics)
    with tqdm(total=total, disable=(not verbose)) as pbar:
        for group_name, seqs in sequences.items():
            group_results = {}
            for metric in metrics:
                pbar.set_postfix(data=group_name, metric=metric.name)
                result = metric(seqs)
                group_results[metric.name] = {"value": result.value, "deviation": result.deviation}
                pbar.update()

            nested[group_name] = group_results

    # Create a DataFrame with MultiIndex columns
    df_parts = []
    for metric in metrics:
        data = {
            (metric.name, key): {group_name: nested[group_name][metric.name][key] for group_name in nested}
            for key in ("value", "deviation")
        }
        df_metric = pd.DataFrame(data, dtype=float)
        df_parts.append(df_metric)

    df = pd.concat(df_parts, axis=1)
    # Ensure order matches input metrics
    df = df.reindex(
        columns=pd.MultiIndex.from_product([[m.name for m in metrics], ["value", "deviation"]]),
    )
    df.attrs["objective"] = {m.name: m.objective for m in metrics}
    return df


def combine(
    dfs: list[pd.DataFrame],
    *,
    on_overlap: Literal["fail", "mean,std"] = "fail",
) -> pd.DataFrame:
    """Combine multiple DataFrames with metric results into a single DataFrame.

    Args:
        dfs: Metric dataframes, each with MultiIndex columns [(metric, 'value'), (metric, 'deviation')], and an 'objective' attribute.
        on_overlap: How to handle cells with multiple values.

            - ``'fail'``: raises an exception on overlap.
            - ``'mean,std'``: sets the cell value to the mean and the deviation to the std of the values.

    Returns:
        DataFrame: A single DataFrame combining multiple metric dataframes.

    Raises:
        ValueError: If ``dfs`` is empty, any DataFrame lacks 'objective', objectives conflict, or potentially overlapping non-null cells.
    """
    if not dfs:
        raise ValueError("The list of DataFrames is empty.")

    for df in dfs:
        if "objective" not in df.attrs:
            raise ValueError("Each DataFrame must have an 'objective' attribute.")

    # preserve column and row order
    combined_rows = []
    seen_rows = set()

    combined_cols = []
    seen_cols = set()

    combined_objectives: dict[str, str] = {}

    for df in dfs:
        objectives = df.attrs["objective"]
        metrics = pd.unique(df.columns.get_level_values(0)).tolist()
        for metric in metrics:
            objective = objectives[metric]
            if metric in combined_objectives and combined_objectives[metric] != objective:
                raise ValueError(
                    f"Conflicting objective for metric '{metric}': '{combined_objectives[metric]}' vs '{objective}'"
                )
            combined_objectives[metric] = objective

        for idx in df.index:
            if idx not in seen_rows:
                seen_rows.add(idx)
                combined_rows.append(idx)

        for col in df.columns:
            if col not in seen_cols:
                seen_cols.add(col)
                combined_cols.append(col)

    # extract cell values
    values: dict[tuple[Any, tuple[str, str]], list[float]] = defaultdict(list)
    for df in dfs:
        for col in df.columns:
            for idx, val in df[col].items():
                if pd.isna(val):
                    continue
                values[(idx, col)].append(val)  # type: ignore

    # handle on_overlap
    res: dict[tuple[Any, tuple[str, str]], float] = {}
    if on_overlap == "fail":
        for cell_name, vs in values.items():
            if len(vs) > 1:
                raise ValueError(f"Multiple values in cell: [{cell_name[0]}, {cell_name[1]}]")
            res[cell_name] = vs[0]
    elif on_overlap == "mean,std":
        for cell_name, vs in values.items():
            col_subname = cell_name[1][1]  # either "value" or "deviation"
            if col_subname == "value":
                res[cell_name] = np.mean(vs).item()
                if len(vs) > 1:
                    dev_cell = (cell_name[0], (cell_name[1][0], "deviation"))
                    res[dev_cell] = np.std(vs).item()
    else:
        raise ValueError(f"'{on_overlap}' not supported.")

    # construct combined dataframe
    row_index = (
        pd.MultiIndex.from_tuples(combined_rows)
        if len(combined_rows) > 0 and isinstance(combined_rows[0], tuple)
        else pd.Index(combined_rows)
    )
    col_index = pd.MultiIndex.from_tuples(combined_cols)  # type: ignore

    combined_df = pd.DataFrame(index=row_index, columns=col_index, dtype=float)
    combined_df.attrs["objective"] = combined_objectives

    for (row, col), val in res.items():  # type: ignore
        combined_df.at[row, col] = val

    return combined_df


def rename(df: pd.DataFrame, metrics: dict[str, str]) -> pd.DataFrame:
    """Rename one or more metrics.

    Args:
        df: Metric Dataframe.
        metrics: Metrics to rename. Format: {old: new, ...}.

    Returns:
        A copy of the original dataframe with the metrics (columns) renamed.

    Raises:
        ValueError: If an `old` metric name is not present in the ``df``, or if a `new` name would create a duplicate objective key.
    """
    old_metrics = pd.unique(df.columns.get_level_values(0)).tolist()
    old_objectives = {metric: df.attrs["objective"][metric] for metric in old_metrics}
    new_objectives = {metric: obj for metric, obj in old_objectives.items() if metric not in metrics}

    for old, new in metrics.items():
        if old not in old_objectives:
            raise ValueError(f"Metric '{old}' does not exist.")

        if new in new_objectives:
            raise ValueError(f"Duplicate metric name '{new}'.")

        new_objectives[new] = old_objectives[old]

    new_df = df.rename(columns=metrics)
    new_df.attrs["objective"] = new_objectives

    return new_df


def sort(df: pd.DataFrame, metric: str, *, level: int = 0, order: Literal["best", "worst"] = "best") -> pd.DataFrame:
    """Sort metric dataframe by a metrics values.

    Args:
        df: Metric Dataframe.
        metric: Metric to consider when sorting.
        level: The tuple index-names level to consider as a group.
        order: Which sequences to be first after sorting.

    Returns:
        Sorted metric dataframe.
    """

    def sort_df(df: pd.DataFrame, metric: str, order: str) -> pd.DataFrame:
        ascending = df.attrs["objective"][metric] == "minimize"
        if order == "worst":
            ascending = not ascending
        return df.sort_values(by=(metric, "value"), ascending=ascending)

    if metric not in df.columns.get_level_values(0):
        raise ValueError(f"'{metric}' is not a column in the DataFrame.")

    if level >= df.index.nlevels or level < 0:
        raise ValueError(f"Level should be in range [0;{df.index.nlevels - 1}].")

    if "objective" not in df.attrs:
        raise ValueError("The DataFrame must have an 'objective' attribute.")

    groups = defaultdict(list)
    for index in df.index:
        level_index = index[:level]
        groups[level_index].append(index)

    dfs_sorted = []
    for group in groups.values():
        df_sub = df.loc[group]
        df_sub_sorted = sort_df(df_sub, metric, order)
        dfs_sorted.append(df_sub_sorted)

    return pd.concat(dfs_sorted)


def top_k(
    df: pd.DataFrame,
    metric: str,
    k: int,
    *,
    level: int = 0,
    keep: Literal["first", "last", "all"] = "all",
) -> pd.DataFrame:
    """Extract top-k rows of the metric dataframe based on a metrics values.

    Args:
        df: Metric Dataframe.
        metric: Metric to consider when selecting top-k rows.
        k: Number of rows to extract.
        level: The tuple index-names level to consider as a group.
        keep: Which entry to keep if multiple are equally good.

    Returns:
        DataFrame: A subset of the metric dataframe with the top-k rows.
    """

    def get_best(df: pd.DataFrame, metric: str, k: int, keep: str) -> pd.DataFrame:
        if df.attrs["objective"][metric] == "minimize":
            return df.nsmallest(k, columns=(metric, "value"), keep=keep)  # type: ignore
        return df.nlargest(k, columns=(metric, "value"), keep=keep)  # type: ignore

    if metric not in df.columns.get_level_values(0):
        raise ValueError(f"'{metric}' is not a column in the DataFrame.")

    if level >= df.index.nlevels or level < 0:
        raise ValueError(f"Level should be in range [0;{df.index.nlevels - 1}].")

    if "objective" not in df.attrs:
        raise ValueError("The DataFrame must have an 'objective' attribute.")

    groups = defaultdict(list)
    for index in df.index:
        level_index = index[:level]
        groups[level_index].append(index)

    dfs_bests = []
    for group in groups.values():
        df_sub = df.loc[group]
        df_sub_best = get_best(df_sub, metric, k, keep)
        dfs_bests.append(df_sub_best)

    df_combined = pd.concat(dfs_bests)

    # keep the original index order
    top_k_indices = set(df_combined.index)
    ordered_index = [index for index in df.index if index in top_k_indices]

    return df_combined.loc[ordered_index]


def show(
    df: pd.DataFrame,
    *,
    color: str | None = "#68d6bc",
    color_style: Literal["solid", "gradient", "bar"] = "solid",
    n_decimals: int | list[int] = 2,
    notation: Literal["decimals", "exponent"] | list[Literal["decimals", "exponent"]] = "decimals",
    na_value: str = "-",
    show_arrow: bool = True,
    level: int = 0,
    hline_level: int | None = None,
    caption: str | None = None,
) -> Styler:
    """Display a metric dataframe as a styled table.

    Render a styled DataFrame that:
        - Combines 'value' and 'deviation' into "value ± deviation".
        - Highlights the best metric per column with color.
        - Underlines the second-best metric per column.
        - Arrows indicate maximize (↑) or minimize (↓).
        - Vertical divider between columns.

    Args:
        df: DataFrame with MultiIndex columns [(metric, 'value'), (metric, 'deviation')], attributed with 'objective'.
        color: Color (hex) for highlighting best scores. If ``None``, no coloring.
        color_style: Style of the coloring. Ignored if color is ``None``.
        n_decimals: Decimal precision for formatting.
        notation: Whether to use scientific notation (exponent) or fixed-point notation (decimals).
        na_value: string to show for cells with no metric value, i.e., cells with NaN values.
        show_arrow: Whether to include the objective arrow in the column names.
        level: The tuple index-names level to consider as a group.
        hline_level: When to add horizontal lines seperaing model names. If ``None``, add horizontal lines at the first level if more than 1 level.
        caption: Bottom caption text.

    Returns:
        Styler: pandas Styler object.
    """

    def format_cell(
        val: float,
        dev: float,
        n_decimals: int,
        notation: Literal["decimals", "exponent"],
        no_value: str,
    ) -> str:
        notation_formatters = {"decimals": "f", "exponent": "e"}
        suffix_notation = notation_formatters[notation]

        if pd.isna(val):
            return no_value
        if pd.isna(dev):
            return f"{val:.{n_decimals}{suffix_notation}}"
        return f"{val:.{n_decimals}{suffix_notation}}±{dev:.{n_decimals}{suffix_notation}}"

    def _fraction(val: float, min_value: float, max_value: float, objective: str) -> float:
        if pd.isna(val):
            return 0.0
        if max_value <= min_value:
            return 1.0
        t = (val - min_value) / (max_value - min_value)
        return 1 - t if objective == "minimize" else t

    def decorate_solid(idx: int, metric: str, df: pd.DataFrame, is_best: bool, is_second_best: bool) -> str:
        fmts = []
        if is_best:
            if color:
                fmts += [f"background-color:{color}"]
            fmts += ["font-weight:bold"]
        if is_second_best:
            fmts += ["text-decoration:underline"]
        return "; ".join(fmts)

    def decorate_gradient(idx: int, metric: str, df: pd.DataFrame, is_best: bool, is_second_best: bool) -> str:
        def gradient_lerp(hex_color1: str, hex_color2: str, t: float) -> str:
            cmap = mpl.colors.LinearSegmentedColormap.from_list(None, [hex_color1, hex_color2])
            return mpl.colors.to_hex(cmap(t), keep_alpha=True)

        fmts = []
        if color:
            objective = df.attrs["objective"][metric]
            values = df[(metric, "value")]
            frac = _fraction(values.at[idx], values.min(), values.max(), objective)
            gradient = gradient_lerp(f"{color}00", f"{color}ff", frac)
            fmts += [f"background-color:{gradient}"]

        if is_best:
            fmts += ["font-weight:bold"]
        if is_second_best:
            fmts += ["text-decoration:underline"]
        return "; ".join(fmts)

    def decorate_bar(idx: int, metric: str, df: pd.DataFrame, is_best: bool, is_second_best: bool) -> str:
        fmts = []
        if color:
            objective = df.attrs["objective"][metric]
            values = df[(metric, "value")]
            frac = _fraction(values.at[idx], values.min(), values.max(), objective)
            if frac > 0:
                width = f"{frac * 100:.1f}%"
                fmts += [f"background: linear-gradient(90deg, {color} {width}, transparent {width})"]

        if is_best:
            fmts += ["font-weight:bold"]
        if is_second_best:
            fmts += ["text-decoration:underline"]
        return "; ".join(fmts)

    def decorate_col(col_series: pd.Series, metric: str, fn: Callable, df: pd.DataFrame) -> list[str]:
        best_indices, second_best_indices = _get_top_indices(df, metric)
        return [fn(idx, metric, df, idx in best_indices, idx in second_best_indices) for idx in col_series.index]

    def get_changing_rows_iloc(indices: pd.Index, hline_level: int) -> list[int]:
        level_names = [idx[:hline_level] for idx in indices]
        changing_rows = []
        prev = None
        for i, v in enumerate(level_names):
            if i != 0 and v != prev:
                changing_rows.append(i)  # i is 0-based index into dataframe rows
            prev = v
        return changing_rows

    n_metrics = df.shape[1] // 2
    n_decimals = [n_decimals] * n_metrics if isinstance(n_decimals, int) else n_decimals
    notation = [notation] * n_metrics if isinstance(notation, str) else notation

    if len(n_decimals) != n_metrics:
        raise ValueError(
            f"Expected {n_metrics} decimals, got {len(n_decimals)}. Provide a single int or a list matching the number of metrics."
        )

    if len(notation) != n_metrics:
        raise ValueError(
            f"Expected {n_metrics} notations, got {len(notation)}. Provide a single int or a list matching the number of metrics."
        )

    if level >= df.index.nlevels or level < 0:
        raise ValueError(f"Level should be in range [0;{df.index.nlevels - 1}].")

    df = df.round(dict(zip(df.columns, [d for d in n_decimals for _ in range(2)], strict=True)))

    arrows = {"maximize": "↑", "minimize": "↓"}
    metrics = pd.unique(df.columns.get_level_values(0)).tolist()

    objectives = df.attrs["objective"]
    df_styled = pd.DataFrame(index=df.index)
    for i, m in enumerate(metrics):
        vals, devs = df[(m, "value")], df[(m, "deviation")]
        arrow = arrows[objectives[m]]
        col_name = f"{m}{arrow}" if show_arrow else m
        df_styled[col_name] = [
            format_cell(val, dev, n_decimals[i], notation[i], na_value) for val, dev in zip(vals, devs, strict=True)
        ]

    decorators = {"solid": decorate_solid, "gradient": decorate_gradient, "bar": decorate_bar}
    decorator = decorators[color_style]

    styler = df_styled.style

    # Decorate columns based on a levels groups
    groups = defaultdict(list)
    for index in df.index:
        level_index = index[:level]
        groups[level_index].append(index)

    for group in groups.values():
        df_sub = df.loc[group]

        for col, metric in zip(styler.columns, metrics, strict=True):
            styler = styler.apply(
                partial(decorate_col, metric=metric, fn=decorator, df=df_sub),
                axis=0,
                subset=(df_sub.index, [col]),  # type: ignore
            )

    table_styles = [
        {"selector": "th.col_heading", "props": [("text-align", "center")]},
        {"selector": "td", "props": [("border-right", "1px solid #ccc")]},
        {"selector": "th.row_heading", "props": [("border-right", "1px solid #ccc")]},
    ]

    if hline_level is None:
        hline_level = 1 if df.index.nlevels > 1 else 0

    if hline_level > df.index.nlevels or hline_level < 0:
        raise ValueError(f"Level should be in range [0;{df.index.nlevels}].")

    if hline_level > 0:
        rows_iloc = get_changing_rows_iloc(df.index, hline_level)
        for row_idx in rows_iloc:
            nth_child = row_idx + 1  # add CSS using tbody nth-child (nth-child is 1-based, so add 1)
            selector = f"tbody tr:nth-child({nth_child}) td, tbody tr:nth-child({nth_child}) th"
            table_styles += [({"selector": selector, "props": [("border-top", "1px solid #ccc")]})]

    if caption:
        styler = styler.set_caption(caption)
        table_styles += [{"selector": "caption", "props": [("caption-side", "bottom"), ("margin-top", "0.75em")]}]

    styler = styler.set_table_styles(table_styles, overwrite=False)  # type: ignore

    return styler


def to_latex(
    df: pd.DataFrame,
    path: str | Path,
    *,
    color: str | None = None,
    n_decimals: int | list[int] = 2,
    notation: Literal["decimals", "exponent"] | list[Literal["decimals", "exponent"]] = "decimals",
    na_value: str = "-",
    show_arrow: bool = True,
    caption: str = None,
):
    """Export a metric dataframe to a LaTeX table.

    Args:
        df: DataFrame with MultiIndex columns [(metric, 'value'), (metric, 'deviation')], attributed with 'objective'.
        path: Output filename, e.g., ``"./path/table.tex"``.
        color: Color (hex) for highlighting best scores. If ``None``, no coloring.
        n_decimals: Decimal precision for formatting.
        notation: Whether to use scientific notation (exponent) or fixed-point notation (decimals).
        na_value: str to show for cells with no metric value, i.e., cells with NaN values.
        show_arrow: Whether to include the objective arrow in the column names.
        caption: Bottom caption text.
    """

    def no_escapes(values: list[Any]) -> list[NoEscape]:
        return [NoEscape(v) for v in values]

    def macro(command: str, *args) -> str:
        vs = "".join(["{" + str(v) + "}" for v in args])
        return f"\\{command}{vs}"

    def imath(value: str) -> str:
        return f"${value}$"

    def format_notation(value: str, notation: Literal["decimals", "exponent"], n_decimal: int) -> str:
        if notation == "decimals":
            return value
        elif notation == "exponent":
            base, exp = f"{value:.{n_decimal}e}".split("e")
            exp_abs = abs(int(exp))
            f_exp_sign = macro("text", exp[0])
            return f"{base} e{f_exp_sign}{exp_abs}" if exp_abs != 0 else base
        raise ValueError(f"Unsupported notation '{notation}'.")

    def format_column_header(metric: str) -> str:
        arrow = arrows[objectives[metric]]
        text = f"{metric} ({arrow})" if show_arrow else metric
        return macro("textbf", text)

    # @TODO: support multi-index rows + levels
    if df.index.nlevels != 1:
        raise ValueError("to_latex() does not support tuple sequence names.")

    n_metrics = df.shape[1] // 2
    n_decimals = [n_decimals] * n_metrics if isinstance(n_decimals, int) else n_decimals
    notation = [notation] * n_metrics if isinstance(notation, str) else notation

    if len(n_decimals) != n_metrics:
        raise ValueError(
            f"Expected {n_metrics} decimals, got {len(n_decimals)}. Provide a single int or a list matching the number of metrics."
        )

    if len(notation) != n_metrics:
        raise ValueError(
            f"Expected {n_metrics} notations, got {len(notation)}. Provide a single int or a list matching the number of metrics."
        )

    df = df.round(dict(zip(df.columns, [d for d in n_decimals for _ in range(2)], strict=True)))

    best_indices, second_best_indices = {}, {}
    metrics = pd.unique(df.columns.get_level_values(0)).tolist()
    for m in metrics:
        best_indices[m], second_best_indices[m] = _get_top_indices(df, m)

    objectives = df.attrs["objective"]
    arrows = {"maximize": "↑", "minimize": "↓"}

    col_names = list(df.columns.get_level_values(0).unique())
    n_cols = len(col_names)
    n_row_levels = df.index.nlevels
    n_cols_and_row_levels = n_row_levels + n_cols

    # LaTeX formatting

    table = Table()
    table.append(NoEscape(macro("centering")))

    col_header = "c" * n_cols_and_row_levels
    tabular = Tabular(col_header)

    tabular.append(NoEscape(macro("toprule")))

    t_col_names = [format_column_header(metric) for metric in col_names]
    tabular.add_row(no_escapes([macro("textbf", "Method")] + t_col_names))

    tabular.append(NoEscape(macro("midrule")))

    for row_name, row in df.iterrows():
        values = [row_name]
        for i, (col_name, val, dev) in enumerate(zip(col_names, row[::2], row[1::2], strict=True)):
            if pd.isna(val):
                values.append(na_value)
                continue

            best = row_name in best_indices[col_name]
            second_best = row_name in second_best_indices[col_name]

            fval = format_notation(val, notation[i], n_decimals[i])
            if not pd.isna(dev):
                fdev = format_notation(dev, notation[i], n_decimals[i])

            if best:
                fvalue = macro("mathbf", fval)
                if not pd.isna(dev):
                    fvalue += " \\pm " + macro("mathbf", fdev)
                if color:
                    fvalue = macro("cellcolor[HTML]", color[1:], fvalue)
            elif second_best:
                fvalue = f"{fval}" if pd.isna(dev) else f"{fval} \\pm {fdev}"
                fvalue = macro("underline", fvalue)
            else:
                fvalue = f"{fval}" if pd.isna(dev) else f"{fval} \\pm {fdev}"

            values.append(imath(fvalue))

        tabular.add_row(no_escapes(values))

    tabular.append(NoEscape(macro("bottomrule")))
    table.append(tabular)

    if caption:
        table.add_caption(caption)

    latex_code = table.dumps()

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_code)


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
        if not isinstance(model_name, str) or not isinstance(iteration, int | float):
            raise ValueError(
                "Expected a tuple of type (str, int | float), "
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
                    linewidths=linewidth,
                    zorder=1,
                    alpha=deviation_alpha,
                )
                ax.add_collection(lc)

        if linestyle and xs.size > 1:
            ax.plot(xs, ys, color=c, linestyle=linestyle, zorder=0)  # type: ignore

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


class Cache:
    """Caches model-generated feature representations of sequences.

    Allows storing and retrieving representations per model to avoid
    recomputation, with support for adding models and precomputed values.
    """

    def __init__(
        self,
        models: dict[str, Callable[[list[str]], list[Any] | np.ndarray]] | None = None,
        init_cache: dict[str, dict[str, np.ndarray]] | None = None,
    ):
        """Initialize the cache with optional models and precomputed representations.

        Args:
            models: Mapping from model name to callable for generating representations.
            init_cache: Initial cache of sequence feature representations by model and sequence.
        """
        self.model_to_callable = models.copy() if models else {}
        self.model_to_cache = init_cache.copy() if init_cache else {}

        for name in self.model_to_callable:
            if name not in self.model_to_cache:
                self.model_to_cache[name] = {}

    def __call__(self, sequences: list[str], model_name: str, stack: bool) -> list[Any] | np.ndarray:
        """Return feature representations for the given sequences using the specified model.

        Uncached sequences are computed and stored.

        Args:
            sequences: List of text sequences.
            model_name: Name of the model to use.
            stack: Whether the feature representations should be stacked as a numpy array. If ``True`` then stack as a numpy array else return a list of representations.

        Returns:
            Feature representations in the same order as the input sequences.
        """
        sequence_to_rep = self.model_to_cache[model_name]

        new_sequences = [seq for seq in set(sequences) if seq not in sequence_to_rep]
        if len(new_sequences) > 0:
            model = self.model_to_callable.get(model_name)
            if model is None:
                raise ValueError(f"New sequences found, but '{model_name}' is not callable.")

            new_reps = model(new_sequences)

            for sequence, rep in zip(new_sequences, new_reps, strict=True):
                sequence_to_rep[sequence] = rep

        reps = [sequence_to_rep[seq] for seq in sequences]
        return np.stack(reps) if stack else reps

    def model(
        self,
        model_name: str,
        *,
        stack: bool = True,
    ) -> Callable[[list[str]], list[Any] | np.ndarray]:
        """Return a callable interface for a given model name.

        Args:
            model_name: Name of the model to use.
            stack: Whether the feature representations should be stacked as a numpy array. If ``True`` then stack as a numpy array else return a list of feature representations.

        Raises:
            ValueError: If the model is unknown.
        """
        if model_name not in self.model_to_cache:
            raise ValueError(f"'{model_name}' is not callable nor has any pre-cached sequences.")
        return lambda sequence: self(sequence, model_name, stack)

    def add(
        self,
        model_name: str,
        element: Callable[[list[str]], list[Any] | np.ndarray] | dict[str, Any],
    ):
        """Add a new model or precomputed representations to the cache.

        Args:
            model_name: Name of the model to use.
            element: A callable representations function or pre-computed (sequence, representation) pairs.

        Raises:
            ValueError: If the model already exists.
        """
        if callable(element):
            if model_name in self.model_to_callable:
                raise ValueError("Model already exists.")

            self.model_to_callable[model_name] = element

            if model_name not in self.model_to_cache:
                self.model_to_cache[model_name] = {}

        elif isinstance(element, dict):
            if model_name not in self.model_to_cache:
                self.model_to_cache[model_name] = {}

            sequence_to_rep = self.model_to_cache[model_name]
            for sequence, reps in element.items():
                sequence_to_rep[sequence] = reps
        else:
            raise TypeError(
                f"element must be either dict[str, np.ndarray] or Callable[[list[str]], np.ndarray], "
                f"but got {type(element).__name__}"
            )

    def remove(self, model_name: str):
        """Remove the cache of a model and the model callable if defined.

        Args:
            model_name: Name of the model to use.
        """
        del self.model_to_cache[model_name]

        if model_name in self.model_to_callable:
            del self.model_to_callable[model_name]

    def get(self) -> dict[str, dict[str, Any]]:
        """Return a copy of the current cache.

        Returns:
            A nested dictionary of cached sequence representations.
        """
        return self.model_to_cache.copy()


def _get_top_indices(df: pd.DataFrame, metric: str) -> tuple[set[int], set[int]]:
    def top_indices_helper(top_two: pd.Series) -> tuple[set[int], set[int]]:
        if pd.isna(top_two.values[0]):
            return set(), set()

        # get all indices with the same value as the best value
        value1 = top_two.values[0]
        indices1 = top_two.index[top_two == value1].tolist()
        if len(indices1) >= 2:
            return set(indices1), set()

        if len(top_two) < 2 or pd.isna(top_two.values[1]):
            return set(indices1), set()

        # get all indices with the same value as the second best value
        value2 = top_two.values[1]
        indices2 = top_two.index[top_two == value2].tolist()

        return set(indices1), set(indices2)

    if "objective" not in df.attrs:
        raise ValueError("DataFrame must have an 'objective' attribute. Use 'sm.evaluate' to create the DataFrame.")

    objective = df.attrs["objective"][metric]
    vals = df[(metric, "value")]

    if objective == "maximize":
        best_cells = vals.nlargest(2, keep="all")
    elif objective == "minimize":
        best_cells = vals.nsmallest(2, keep="all")
    else:
        raise ValueError(f"Unknown objective '{objective}' for metric '{metric}'.")

    return top_indices_helper(best_cells)


def read_fasta(path: str | Path) -> list[str]:
    """Retrieve sequences from a FASTA file.

    Args:
        path: Path to FASTA file.

    Returns:
        The list of sequences.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    sequences: list[str] = []
    current_seq: list[str] = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            if line.startswith(">"):
                if current_seq:
                    sequence = "".join(current_seq)
                    if sequence:
                        sequences.append(sequence)
                    current_seq = []
            else:
                current_seq.append(line)

        # Add the last sequence if present
        if current_seq:
            sequence = "".join(current_seq)
            if sequence:
                sequences.append(sequence)

    return sequences


def to_fasta(sequences: list[str], path: str | Path, *, headers: list[str] | None = None):
    """Write sequences to a FASTA file.

    Args:
       sequences: List of text sequences.
       path: Output filepath, e.g., ``"/path/seqs.fasta"``.
       headers: Optional sequence names.
    """
    if headers is not None and len(headers) != len(sequences):
        raise ValueError("headers length must match sequences length")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        for i, seq in enumerate(sequences):
            header = headers[i] if headers else f">seq_{i + 1}"

            if not header.startswith(">"):
                header = ">" + header

            f.write(f"{header}\n")
            f.write(f"{seq}\n")


def read_pickle(path: str | Path) -> Any:
    """Load and return an object from a pickle file.

    Args:
        path: Path to pickle file.

    Returns:
        The deserialized Python object.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("rb") as f:
        return pickle.load(f)


def to_pickle(content: Any, path: str | Path):
    """Serialize an object and write it to a pickle file.

    Args:
       content: Pickable object.
       path: Output filepath, e.g., ``"/path/cache.pkl"``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        pickle.dump(content, f)
