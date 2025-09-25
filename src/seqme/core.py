import abc
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
    """Data structure to store metric result."""

    value: float | int
    deviation: float | None = None


class Metric(abc.ABC):
    """Abstract base for metrics evaluating lists of text sequences.

    Subclasses implement a callable interface to compute a score and
    specify a name and optimization direction.
    """

    @abc.abstractmethod
    def __call__(self, sequences: list[str]) -> MetricResult:
        """Calculate the metric over provided sequences.

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
    """Compute a set of metrics on multiple sequence groups.

    Args:
        sequences: A dict mapping group names to lists of sequences.
        metrics: A list of Metric instances to apply.
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
    """Combine multiple DataFrames with metrics results into a single DataFrame.

    Args:
        dfs: list of DataFrames, each with MultiIndex columns [(metric, 'value'), (metric, 'deviation')], and an 'objective' attribute.
        on_overlap: How to handle cells with multiple values.

            - "fail": raises an exception on overlap.
            - "mean,std": sets the cell value to the mean and the deviation to the std of the values.

    Returns:
        A single DataFrame combining multiple metric dataframes.

    Raises:
        ValueError: If dfs is empty, any DataFrame lacks 'objective', objectives conflict, or potentially overlapping non-null cells.
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
    """Rename metrics.

    Args:
        df: Metric Dataframe.
        metrics: Metrics to rename. Format: {old: new, ...}.

    Returns:
        A subset of the metric dataframe with the top-k rows.

    Raises:
        ValueError: If an `old` metric name is not present in the `df`, or if a `new` name would create a duplicate objective key.
    """
    old_metrics = pd.unique(df.columns.get_level_values(0)).tolist()
    old_objectives = {metric: df.attrs["objective"][metric] for metric in old_metrics}
    new_objectives = {mt: obj for mt, obj in old_objectives.items() if mt not in metrics}

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
    """Sort metric dataframe by metric value.

    Args:
        df: Metric Dataframe.
        metric: Metric to consider when sorting.
        level: The tuple index-names level to consider as a group.
        order: Which sequences to be first after sorting.

    Returns:
        Sorted metric dataframe.
    """

    def sort_df(df: pd.DataFrame, metric: str) -> pd.DataFrame:
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
        df_sub_sorted = sort_df(df_sub, metric)
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
        A subset of the metric dataframe with the top-k rows.
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


def display(
    df: pd.DataFrame,
    *,
    n_decimals: int | list[int] = 2,
    color: str | None = "#68d6bc",
    color_style: Literal["solid", "gradient", "bar"] = "solid",
    notation: Literal["decimals", "exponent"] | list[Literal["decimals", "exponent"]] = "decimals",
    missing_value: str = "-",
    show_arrow: bool = True,
    level: int = 0,
    hline_level: int | None = None,
    caption: str | None = None,
) -> Styler:
    """Visualize a table of a metric dataframe.

    Render a styled DataFrame that:
        - Combines 'value' and 'deviation' into "value ± deviation".
        - Highlights the best metric per column with color.
        - Underlines the second-best metric per column.
        - Arrows indicate maximize (↑) or minimize (↓).
        - Vertical divider between columns.

    Args:
        df: DataFrame with MultiIndex columns [(metric, 'value'), (metric, 'deviation')], attributed with 'objective'.
        n_decimals: Decimal precision for formatting.
        color: Color (hex) for highlighting best scores.
        color_style: Style of the coloring. Ignored if color is None.
        notation: Whether to use scientific notation (exponent) or fixed-point notation (decimals).
        missing_value: str to show for cells with no metric value, i.e., cells with NaN values.
        show_arrow: Whether to include the objective arrow in the column names.
        level: The tuple index-names level to consider as a group.
        hline_level: When to add horizontal lines seperaing model names. If None, add horizontal lines at the first level if more than 1 level.
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
            values = df[metric]["value"]
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
            values = df[metric]["value"]
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
            format_cell(val, dev, n_decimals[i], notation[i], missing_value)
            for val, dev in zip(vals, devs, strict=True)
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
    fname: str | Path,
    df: pd.DataFrame,
    *,
    n_decimals: int | list[int] = 2,
    color: str | None = None,
    notation: Literal["decimals", "exponent"] | list[Literal["decimals", "exponent"]] = "decimals",
    missing_value: str = "-",
    show_arrow: bool = True,
    caption: str = None,
):
    """Convert a metric dataframe to a LaTeX table.

    Args:
        fname: Output filename, e.g., "table.tex".
        df: DataFrame with MultiIndex columns [(metric, 'value'), (metric, 'deviation')], attributed with 'objective'.
        n_decimals: Decimal precision for formatting.
        color: Color (hex) for highlighting best scores.
        notation: Whether to use scientific notation (exponent) or fixed-point notation (decimals).
        missing_value: str to show for cells with no metric value, i.e., cells with NaN values.
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
                values.append(missing_value)
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

    output_path = Path(fname)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_code)


def barplot(
    df: pd.DataFrame,
    metric: str,
    *,
    show_deviation: bool = True,
    color: str = "#14c299",
    x_ticks_label_rotation: float = 45,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[int, int] = (4, 3),
    show_arrow: bool = True,
    ax: Axes | None = None,
):
    """Plot a bar chart for a given metric, optionally with error bars.

    Args:
        df: A DataFrame with a MultiIndex column [metric, {"value", "deviation"}].
        metric: The name of the metric to plot.
        show_deviation: Whether to plot the deviation if available.
        color: Bar color (optional, default is teal).
        x_ticks_label_rotation: Rotation angle for x-axis tick labels.
        ylim: Y-axis limits (optional).
        figsize: Size of the figure.
        show_arrow: Whether to show an arrow indicating maximize/minimize (default is True).
        ax: Optional matplotlib Axes to plot on.
    """
    if metric not in df.columns.get_level_values(0):
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
    ax.set_xticklabels(bar_names, rotation=x_ticks_label_rotation, ha="center")

    ax.set_ylabel(f"{metric}{arrow}" if show_arrow else metric)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    if created_fig:
        plt.tight_layout()
        plt.show()


def plot_series(
    df: pd.DataFrame,
    metric: str,
    *,
    show_deviation: bool = True,
    figsize: tuple[int, int] = (4, 3),
    marker: str | None = "x",
    xlabel: str = "Iteration",
    alpha: float = 0.4,
    show_arrow: bool = True,
    ax: Axes | None = None,
):
    """Plot a graph for a given metric across multiple iterations/steps, optionally with error bars.

    Args:
        df: A DataFrame with a MultiIndex column [metric, {"value", "deviation"}].
        metric: The name of the metric to plot.
        show_deviation: Whether to the plot deviation if available.
        marker: Marker type for graphs.
        xlabel: Name of x-label.
        alpha: opacity level of deviation intervals.
        figsize: Size of the figure.
        show_arrow: Whether to show an arrow indicating maximize/minimize (default is True).
        ax: Optional matplotlib Axes to plot on.
    """
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

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        created_fig = True

    model_names = list(df.index.get_level_values(0).unique())
    for model_name in model_names:
        df_model = df.loc[model_name]
        df_model = df_model.sort_index()

        xs = df_model.index
        vs = df_model[metric]["value"]
        dev = df_model[metric]["deviation"]

        if show_deviation:
            ax.fill_between(xs, vs - dev, vs + dev, alpha=alpha)

        ax.plot(xs, vs, marker=marker, label=model_name)

    objective = df.attrs["objective"][metric]
    arrows = {"maximize": "↑", "minimize": "↓"}

    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{metric}{arrows[objective]}" if show_arrow else metric)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    if created_fig:
        plt.tight_layout()
        plt.show()


class ModelCache:
    """Caches model-generated feature representations of sequences.

    Allows storing and retrieving embeddings per model to avoid
    recomputation, with support for adding models and precomputed values.
    """

    def __init__(
        self,
        models: dict[str, Callable[[list[str]], list[Any] | np.ndarray]] | None = None,
        init_cache: dict[str, dict[str, np.ndarray]] | None = None,
    ):
        """Initialize the cache with optional models and precomputed representations.

        Args:
            models: Mapping from model name to callable for generating embeddings.
            init_cache: Initial cache of embeddings by model and sequence.
        """
        self.model_to_callable = models.copy() if models else {}
        self.model_to_cache = init_cache.copy() if init_cache else {}

        for name in self.model_to_callable:
            if name not in self.model_to_cache:
                self.model_to_cache[name] = {}

    def __call__(self, sequences: list[str], model_name: str, variable_length: bool) -> list[Any] | np.ndarray:
        """Return embeddings for the given sequences using the specified model.

        Uncached sequences are computed and stored automatically.

        Args:
            sequences: List of input texts.
            model_name: Name of the model to use.
            variable_length: Whether the embeddings are variable size. If true then return a list of embeddings else stack as a numpy array.

        Returns:
            Embeddings in the same order as input.
        """
        sequence_to_rep = self.model_to_cache[model_name]

        new_sequences = [seq for seq in sequences if seq not in sequence_to_rep]
        if len(new_sequences) > 0:
            model = self.model_to_callable.get(model_name)
            if model is None:
                raise ValueError(f"New sequences found, but '{model_name}' is not callable.")

            new_reps = model(new_sequences)

            for sequence, rep in zip(new_sequences, new_reps, strict=True):
                sequence_to_rep[sequence] = rep

        reps = [sequence_to_rep[seq] for seq in sequences]
        return reps if variable_length else np.stack(reps)

    def model(
        self,
        model_name: str,
        *,
        variable_length: bool = False,
    ) -> Callable[[list[str]], list[Any] | np.ndarray]:
        """Return a callable interface for a given model name.

        Args:
            model_name: Name of the model to use.
            variable_length: Whether the embeddings are variable size. If true then return a list of embeddings else stack as a numpy array.

        Raises:
            ValueError: If the model is unknown.
        """
        if model_name not in self.model_to_cache:
            raise ValueError(f"'{model_name}' is not callable nor has any pre-cached sequences.")
        return lambda sequence: self(sequence, model_name, variable_length)

    def add(
        self,
        model_name: str,
        element: Callable[[list[str]], list[Any] | np.ndarray] | dict[str, Any],
    ):
        """Add a new model or precomputed embeddings to the cache.

        Args:
            model_name: Name of the model to use.
            element: A callable embedding function or pre-computed (sequence, embedding) pairs.

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
            A nested dictionary of cached embeddings.
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
        raise ValueError("DataFrame must have an 'objective' attribute. Use compute_metrics to create the DataFrame.")

    objective = df.attrs["objective"][metric]
    vals = df[(metric, "value")]

    if objective == "maximize":
        best_cells = vals.nlargest(2, keep="all")
    elif objective == "minimize":
        best_cells = vals.nsmallest(2, keep="all")
    else:
        raise ValueError(f"Unknown objective '{objective}' for metric '{metric}'.")

    return top_indices_helper(best_cells)
