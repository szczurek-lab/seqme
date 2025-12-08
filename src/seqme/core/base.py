import abc
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
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
        raise ValueError("No metrics provided.")

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
