import abc
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from pandas.io.formats.style import Styler
from pylatex import NoEscape, Table, Tabular
from tqdm import tqdm


@dataclass
class MetricResult:
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


def compute_metrics(
    sequences: dict[str, list[str]] | dict[tuple[str, ...], list[str]],
    metrics: list[Metric],
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

    # ensure all metrics have unique names
    metric_names = [m.name for m in metrics]
    if len(metric_names) != len(set(metric_names)):
        raise ValueError(
            "Metrics must have unique names. Found duplicates: "
            + ", ".join(name for name in metric_names if metric_names.count(name) > 1)
        )

    # Prepare nested results: group -> metric -> {value, deviation}
    nested: dict[str | tuple[str, ...], dict[str, dict[str, float | None]]] = {}
    total = len(sequences) * len(metrics)
    with tqdm(total=total, disable=(not verbose)) as pbar:
        for group_name, seqs in sequences.items():
            group_results: dict[str, dict[str, float | None]] = {}
            for metric in metrics:
                pbar.set_postfix(data=group_name, metric=metric.name)
                result: MetricResult = metric(seqs)
                group_results[metric.name] = {
                    "value": result.value,
                    "deviation": result.deviation,
                }
                pbar.update()

            nested[group_name] = group_results

    # Convert to a DataFrame with MultiIndex columns
    # First, create a dict of DataFrames per metric
    df_parts: list[pd.DataFrame] = []
    for metric in metrics:
        name = metric.name
        # DataFrame with two columns: (name, 'value') and (name, 'deviation')
        df_metric = pd.DataFrame(
            {
                (name, "value"): {g: nested[g][name]["value"] for g in nested},
                (name, "deviation"): {g: nested[g][name]["deviation"] for g in nested},
            },
            dtype=float,
        )
        df_parts.append(df_metric)

    df = pd.concat(df_parts, axis=1)
    # Ensure order matches input metrics
    df = df.reindex(
        columns=pd.MultiIndex.from_product([[m.name for m in metrics], ["value", "deviation"]]),
    )
    df.attrs["objective"] = {m.name: m.objective for m in metrics}
    return df


def combine_metric_dataframes(dfs: list[pd.DataFrame], on_overlap: Literal["fail", "mean"] = "fail") -> pd.DataFrame:
    """Combine multiple DataFrames with metrics results into a single DataFrame.

    Args:
        dfs: list of DataFrames, each with MultiIndex columns [(metric, 'value'), (metric, 'deviation')], and an 'objective' attribute.
        on_overlap: How to handle cells with multiple values.

            - "fail": raises an exception on overlap.
            - "mean": sets the cell value to the mean and the deviation to the std of the values.

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
        for key, value in df.attrs["objective"].items():
            if key in combined_objectives and combined_objectives[key] != value:
                raise ValueError(f"Conflicting objective for metric '{key}': {combined_objectives[key]} vs {value}")
            combined_objectives[key] = value

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
    elif on_overlap == "mean":
        for cell_name, vs in values.items():
            col_subname = cell_name[1][1]  # either "value" or "deviation"
            if col_subname == "value":
                res[cell_name] = np.mean(vs).item()
                if len(vs) > 1:
                    dev_cell = (cell_name[0], (cell_name[1][0], "deviation"))
                    res[dev_cell] = np.std(vs).item()
    else:
        raise ValueError(f"{on_overlap} not supported.")

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


def show_table(
    df: pd.DataFrame,
    n_decimals: int | list[int] = 2,
    color: str | None = "#68d6bc",
    notation: Literal["decimals", "exponent"] | list[Literal["decimals", "exponent"]] = "decimals",
    missing_value: str = "-",
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
        color: Color for highlighting best scores.
        notation: Whether to use scientific notation (exponent) or fixed-point notation (decimals).
        missing_value: str to show for cells with no metric value, i.e., cells with NaN values.

    Returns:
        Styler: pandas Styler object.
    """
    if "objective" not in df.attrs:
        raise ValueError("DataFrame must have an 'objective' attribute. Use compute_metrics to create the DataFrame.")

    objectives = df.attrs["objective"]

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

    df_rounded = _prepare_for_visualization(df, n_decimals)

    best_indices = df_rounded.attrs["best_indices"]
    second_best_indices = df_rounded.attrs["second_best_indices"]

    ## Formatting
    arrows = {"maximize": "↑", "minimize": "↓"}
    metrics = pd.unique(df.columns.get_level_values(0)).tolist()

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

    df_styled = pd.DataFrame(index=df.index)
    for i, m in enumerate(metrics):
        vals, devs = df_rounded[(m, "value")], df_rounded[(m, "deviation")]
        arrow = arrows[objectives[m]]
        col_name = f"{m}{arrow}"
        df_styled[col_name] = [
            format_cell(val, dev, n_decimals[i], notation[i], missing_value)
            for val, dev in zip(vals, devs, strict=True)
        ]

    def decorate_cell(idx: int, metric: str) -> str:
        if idx in best_indices[metric]:
            fmts = ["font-weight:bold"]
            if color:
                fmts.append(f"background-color:{color}")
            return "; ".join(fmts)
        if idx in second_best_indices[metric]:
            return "text-decoration:underline"
        return ""

    def decorate_col(col_series, metric):
        return [decorate_cell(idx, metric) for idx in col_series.index]

    styler = df_styled.style
    for col, metric in zip(df_styled.columns, metrics, strict=True):
        styler = styler.apply(partial(decorate_col, metric=metric), axis=0, subset=[col])

    table_styles = [
        {"selector": "th.col_heading", "props": [("text-align", "center")]},
        {"selector": "td", "props": [("border-right", "1px solid #ccc")]},
        {"selector": "th.row_heading", "props": [("border-right", "1px solid #ccc")]},
    ]
    styler = styler.set_table_styles(table_styles, overwrite=False)  # type: ignore

    return styler


def to_latex(
    df: pd.DataFrame,
    n_decimals: int | list[int] = 2,
    color: str | None = "#68d6bc",
    notation: Literal["decimals", "exponent"] | list[Literal["decimals", "exponent"]] = "decimals",  # TODO: unused
    missing_value: str = "-",
    caption: str = None,
):
    """Convert a metric table to latex.

    Args:
        df: DataFrame with MultiIndex columns [(metric, 'value'), (metric, 'deviation')], attributed with 'objective'.
        n_decimals: Decimal precision for formatting.
        color: Color for highlighting best scores.
        notation: Whether to use scientific notation (exponent) or fixed-point notation (decimals).
        missing_value: str to show for cells with no metric value, i.e., cells with NaN values.
        caption: Latex table caption.
    """
    # @TODO: support multi-index rows
    if df.index.nlevels != 1:
        raise ValueError("to_latex() does not support tuple sequence names.")

    if "objective" not in df.attrs:
        raise ValueError("DataFrame must have an 'objective' attribute. Use compute_metrics to create the DataFrame.")

    objectives = df.attrs["objective"]

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

    df_rounded = _prepare_for_visualization(df, n_decimals)

    best_indices = df_rounded.attrs["best_indices"]
    second_best_indices = df_rounded.attrs["second_best_indices"]

    arrows = {"maximize": "↑", "minimize": "↓"}

    def no_escapes(vs):
        return [NoEscape(v) for v in vs]

    col_names = list(df_rounded.columns.get_level_values(0).unique())
    n_cols = len(col_names)
    n_row_levels = df_rounded.index.nlevels
    n_cols_and_row_levels = n_row_levels + n_cols

    table = Table()
    table.append(NoEscape(r"\centering"))

    # col_header = "|" + "c|" * n_cols_and_row_levels
    col_header = "c" * n_cols_and_row_levels

    tabular = Tabular(col_header)

    # tabular.add_hline()
    tabular.append(NoEscape("\\toprule"))

    t_col_names = [f"{m}{arrows[objectives[m]]}" for m in col_names]
    tabular.add_row(no_escapes(["Method"] + t_col_names))

    # tabular.add_hline()
    tabular.append(NoEscape("\\midrule"))

    for row_name, row in df_rounded.iterrows():
        values = []
        for col_name, val, dev in zip(col_names, row[::2], row[1::2], strict=True):
            if pd.isna(val):
                values.append(NoEscape(missing_value))
                continue

            value = f"{val}" if pd.isna(dev) else f"{val} \\pm {dev}"

            best = row_name in best_indices[col_name]
            second_best = row_name in second_best_indices[col_name]
            if best:
                value = f"\\mathbf{{{value}}}"
                if color:
                    value = f"\\cellcolor[HTML]{{{color[1:]}}}{{{value}}}"
            elif second_best:
                value = f"\\underline{{{value}}}"

            value = f"${value}$"
            values.append(value)

        tabular.add_row(no_escapes([row_name] + values))

    # tabular.add_hline()
    tabular.append(NoEscape("\\bottomrule"))

    table.append(tabular)

    if caption:
        table.add_caption(caption)

    return table.dumps()


def _prepare_for_visualization(df: pd.DataFrame, n_decimals: list[int]) -> pd.DataFrame:
    if "objective" not in df.attrs:
        raise ValueError("DataFrame must have an 'objective' attribute. Use compute_metrics to create the DataFrame.")

    objectives = df.attrs["objective"]
    df_rounded = df.round(dict(zip(df.columns, [d for d in n_decimals for _ in range(2)], strict=True)))

    ## Extract top sequence indices
    def get_top_indices(top_two: pd.Series) -> tuple[set[int], set[int]]:
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

    best_indices = {}
    second_best_indices = {}

    metrics = pd.unique(df.columns.get_level_values(0)).tolist()

    for m in metrics:
        vals = df_rounded[(m, "value")]
        if objectives[m] == "maximize":
            best_cells = vals.nlargest(2, keep="all")
            best_indices[m], second_best_indices[m] = get_top_indices(best_cells)
        elif objectives[m] == "minimize":
            best_cells = vals.nsmallest(2, keep="all")
            best_indices[m], second_best_indices[m] = get_top_indices(best_cells)
        else:
            raise ValueError(f"Unknown objective '{objectives[m]}' for metric '{m}")

    df_rounded.attrs["best_indices"] = best_indices
    df_rounded.attrs["second_best_indices"] = best_indices

    return df_rounded


def barplot(
    df: pd.DataFrame,
    metric: str,
    show_deviation: bool = True,
    color: str = "#68d6bc",
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
                f"but got ({model_name!r}, {iteration!r}) "
                f"with types ({type(model_name).__name__}, {type(iteration).__name__})."
            )

    created_fig = False
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        created_fig = True

    model_names = {v[0] for v in df.index}
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

    ax.grid(True, linestyle="--", alpha=0.7)
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
        models: dict[str, Callable[[list[str]], np.ndarray]] | None = None,
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

    def __call__(self, sequences: list[str], model_name: str) -> np.ndarray:
        """Return embeddings for the given sequences using the specified model.

        Uncached sequences are computed and stored automatically.

        Args:
            sequences: List of input texts.
            model_name: Name of the model to use.

        Returns:
            Array of embeddings in the same order as input.
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

        return np.stack([sequence_to_rep[seq] for seq in sequences])

    def model(self, model_name: str) -> Callable[[list[str]], np.ndarray]:
        """Return a callable interface for a given model name.

        Args:
            model_name: Name of the model to use.

        Raises:
            ValueError: If the model is unknown.
        """
        if model_name not in self.model_to_cache:
            raise ValueError(f"{model_name} is not callable nor has any pre-cached sequences.")
        return lambda sequence: self(sequence, model_name)

    def add(self, model_name: str, element: Callable[[list[str]], np.ndarray] | dict[str, np.ndarray]):
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

    def get(self) -> dict[str, dict[str, np.ndarray]]:
        """Return a copy of the current cache.

        Returns:
            A nested dictionary of cached embeddings.
        """
        return self.model_to_cache.copy()
