import abc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler
from tqdm import tqdm


@dataclass
class MetricResult:
    value: float | int
    deviation: float | None = None


class Metric(abc.ABC):
    """
    Abstract base for metrics evaluating lists of text sequences.

    Subclasses implement a callable interface to compute a score and
    specify a name and optimization direction.
    """

    @abc.abstractmethod
    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Calculate the metric over provided sequences.

        Args:
            sequences: Text inputs to evaluate.

        Returns:
            An object containing the score and optional details.
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        A short identifier for this metric, used in reporting.

        Returns:
            The metric name.
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def objective(self) -> Literal["minimize", "maximize"]:
        """
        Whether lower or higher scores indicate better performance.

        Returns:
            The optimization goal ('minimize' or 'maximize').
        """
        raise NotImplementedError()


def compute_metrics(
    sequences: dict[str, list[str]],
    metrics: list[Metric],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute a set of metrics on multiple sequence groups.

    Args:
        sequences: A dict mapping group names to lists of sequences.
        metrics: A list of Metric instances to apply.
        verbose: Whether to show progressbar.

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
    nested: dict[str, dict[str, dict[str, float | None]]] = {}
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


def combine_metric_dataframes(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine multiple DataFrames with metrics results into a single DataFrame.

    Args:
        dfs: list of DataFrames, each with MultiIndex columns [(metric, 'value'), (metric, 'deviation')], and an 'objective' attribute.

    Returns:
        A single DataFrame with combined metrics, ensuring no overlapping cells and converting all values to float.

    Raises:
        ValueError: If dfs is empty, any DataFrame lacks 'objective', objectives conflict, or overlapping non-null cells.
    """
    if not dfs:
        raise ValueError("The list of DataFrames is empty.")

    # Prepare lists for ordered index and columns
    combined_index = []  # maintain order of first occurrence
    seen_idx = set()

    combined_columns = []
    seen_cols = set()

    combined_objectives: dict[str, str] = {}

    # Collect indices, columns, and objectives preserving order
    for df in dfs:
        if "objective" not in df.attrs:
            raise ValueError("Each DataFrame must have an 'objective' attribute.")

        # Merge objectives
        for key, value in df.attrs["objective"].items():
            if key in combined_objectives and combined_objectives[key] != value:
                raise ValueError(f"Conflicting objective for metric '{key}': {combined_objectives[key]} vs {value}")
            combined_objectives[key] = value

        # Index order
        for idx in df.index:
            if idx not in seen_idx:
                seen_idx.add(idx)
                combined_index.append(idx)

        # Column order
        for col in df.columns:
            if col not in seen_cols:
                seen_cols.add(col)
                combined_columns.append(col)

    # Reconstruct MultiIndex with original level names
    col_index = pd.MultiIndex.from_tuples(combined_columns, names=dfs[0].columns.names)  # type: ignore

    combined_df = pd.DataFrame(index=combined_index, columns=col_index, dtype=float)
    combined_df.attrs["objective"] = combined_objectives

    # Populate and check for overlapping values
    for df in dfs:
        for col in df.columns:
            for idx, val in df[col].items():
                if pd.isna(val):
                    continue
                if pd.notna(combined_df.at[idx, col]):
                    raise ValueError(
                        f"Overlap detected at index '{idx}', column '{col}' (value: {combined_df.at[idx, col]} vs {val})."
                    )
                combined_df.at[idx, col] = float(val)

    return combined_df


def show_table(
    df: pd.DataFrame,
    decimals: int | list[int] = 2,
    notations: Literal["e", "f"] | list[Literal["e", "f"]] = "f",
    color: str = "#68d6bc",
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
        decimals: Decimal precision for formatting.
        color: Color for highlighting best scores.
        notations: Whether to use scientific notation (e) or not (f).
        missing_value: str to show for cells with no metric value, i.e., cells with NaN values.

    Returns:
        Styler: pandas Styler object.
    """
    if "objective" not in df.attrs:
        raise ValueError("DataFrame must have an 'objective' attribute. Use compute_metrics to create the DataFrame.")

    objectives = df.attrs["objective"]

    n_metrics = df.shape[1] // 2
    decimals = [decimals] * n_metrics if isinstance(decimals, int) else decimals
    notations = [notations] * n_metrics if isinstance(notations, str) else notations

    if len(decimals) != n_metrics:
        raise ValueError(
            f"Expected {n_metrics} decimals, got {len(decimals)}. Provide a single int or a list matching the number of metrics."
        )

    metrics = pd.unique(df.columns.get_level_values(0)).tolist()
    arrows = {"maximize": "↑", "minimize": "↓"}

    df_display = pd.DataFrame(index=df.index)
    df_rounded = df.round(dict(zip(df.columns, [d for d in decimals for _ in range(2)], strict=True)))

    ## Extract top sequence indices
    def get_top_indices(top_two: pd.Series) -> tuple[list[int], list[int]]:
        if pd.isna(top_two.values[0]):
            return [], []

        # get all indices with the same value as the best value
        value1 = top_two.values[0]
        indices1 = top_two.index[top_two == value1].tolist()
        if len(indices1) > 2:
            return indices1, []

        if len(top_two) < 2 or pd.isna(top_two.values[1]):
            return indices1, []

        # get all indices with the same value as the second best value
        value2 = top_two.values[1]
        indices2 = top_two.index[top_two == value2].tolist()

        return indices1, indices2

    best_indices = {}
    second_best_indices = {}

    for m in metrics:
        vals, devs = df_rounded[(m, "value")], df_rounded[(m, "deviation")]
        if objectives[m] == "maximize":
            best_cells = vals.nlargest(2, keep="all")
            best_indices[m], second_best_indices[m] = get_top_indices(best_cells)
        elif objectives[m] == "minimize":
            best_cells = vals.nsmallest(2, keep="all")
            best_indices[m], second_best_indices[m] = get_top_indices(best_cells)
        else:
            raise ValueError(f"Unknown objective '{objectives[m]}' for metric '{m}")

    ## Format cell values
    def format_cell(val: float, dev: float, n_decimals: int, notation: str, no_value: str = missing_value) -> str:
        if pd.isna(val):
            return no_value
        if pd.isna(dev):
            return f"{val:.{n_decimals}{notation}}"
        return f"{val:.{n_decimals}{notation}}±{dev:.{n_decimals}{notation}}"

    for i, m in enumerate(metrics):
        vals, devs = df_rounded[(m, "value")], df_rounded[(m, "deviation")]
        decimal, notation = decimals[i], notations[i]

        arrow = arrows[objectives[m]]
        df_display[f"{m}{arrow}"] = [
            format_cell(val, dev, decimal, notation) for val, dev in zip(vals, devs, strict=True)
        ]

    ## Apply cell styles per column
    styler = df_display.style

    for col, metric in zip(df_display.columns, metrics, strict=True):

        def highlight_column(col_series: pd.Series, metric: str = metric) -> list[str]:
            return [
                f"background-color:{color}; font-weight:bold"
                if idx in best_indices[metric]
                else "text-decoration:underline; font-weight:bold"
                if idx in second_best_indices[metric]
                else ""
                for idx in col_series.index
            ]

        styler = styler.apply(highlight_column, axis=0, subset=[col])

    table_styles = [
        {"selector": "th.col_heading", "props": [("text-align", "center")]},
        {"selector": "td", "props": [("border-right", "1px solid #ccc")]},
        {"selector": "th.row_heading", "props": [("border-right", "1px solid #ccc")]},
    ]
    styler = styler.set_table_styles(table_styles, overwrite=False)  # type: ignore
    return styler


def barplot(
    df: pd.DataFrame,
    metric: str,
    color: str = "#68d6bc",
    x_ticks_label_rotation: float = 45,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[int, int] = (4, 3),
    show_arrow: bool = True,
):
    """
    Plot a bar chart for a given metric, optionally with error bars.

    Args:
        df: A DataFrame with a MultiIndex column [metric, {"value", "deviation"}].
        metric: The name of the metric to plot.
        color: Bar color (optional, default is teal).
        x_ticks_label_rotation: Rotation angle for x-axis tick labels.
        ylim: Y-axis limits (optional).
        figsize: Size of the figure.
        show_arrow: Whether to show an arrow indicating maximize/minimize (default is True).
    """
    if metric not in df.columns.get_level_values(0):
        raise ValueError(f"'{metric}' is not a column in the DataFrame.")

    values = df[(metric, "value")]
    deviations = df[(metric, "deviation")]

    # filter NaN values
    valid_mask = values.notna()
    values = values[valid_mask]
    deviations = deviations[valid_mask]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(values.index, values, color=color, edgecolor="black")

    if deviations.notna().all():
        ax.errorbar(
            values.index,
            values,
            yerr=deviations,
            fmt="none",
            ecolor="black",
            capsize=4,
            lw=1,
        )

    arrows = {"maximize": "↑", "minimize": "↓"}
    arrow = arrows[df.attrs["objective"][metric]]

    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(values.index, rotation=x_ticks_label_rotation, ha="center")

    ax.set_ylabel(f"{metric}{arrow}" if show_arrow else metric)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout()


class FeatureCache:
    """
    Caches model-generated feature representations for sequences.

    Allows storing and retrieving embeddings per model to avoid
    recomputation, with support for adding models and precomputed values.
    """

    def __init__(
        self,
        models: dict[str, Callable[[list[str]], np.ndarray]] | None = None,
        init_cache: dict[str, dict[str, np.ndarray]] | None = None,
    ):
        """
        Initialize the cache with optional models and precomputed representations.

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
        """
        Return embeddings for the given sequences using the specified model.

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
        """
        Return a callable interface for a given model name.

        Args:
            model_name: Name of the model to use.

        Raises:
            ValueError: If the model is unknown.
        """
        if model_name not in self.model_to_cache:
            raise ValueError(f"{model_name} is not callable nor has any pre-cached sequences.")
        return lambda sequence: self(sequence, model_name)

    def add(self, model_name: str, element: Callable[[list[str]], np.ndarray] | dict[str, np.ndarray]):
        """
        Add a new model or precomputed embeddings to the cache.

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
        """
        Removes the cache of a model and the model callable if defined.

        Args:
            model_name: Name of the model to use.
        """
        del self.model_to_cache[model_name]

        if model_name in self.model_to_callable:
            del self.model_to_callable[model_name]

    def get(self) -> dict[str, dict[str, np.ndarray]]:
        """
        Return a copy of the current cache.

        Returns:
            A nested dictionary of cached embeddings.
        """
        return self.model_to_cache.copy()
