from collections import defaultdict
from typing import Literal

import moocore
import numpy as np
import pandas as pd
import scipy.stats


def rank(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    *,
    tiebreak: Literal["crowding-distance", "mean-rank"] | None = None,
    ties: Literal["min", "max", "mean", "dense", "auto"] = "auto",
    name: str = "Rank",
) -> pd.DataFrame:
    """Calculate the non-dominated rank of each entry using one or more metrics.

    If the column already exists, then don't use it to compute the rank unless explicitly selected in ``metrics``. Rank overrides the column ``name`` if it already exists.

    Reference:
        - David Come and Joshua Knowles, *Techniques for Highly Multiobjective Optimisation: Some Nondominated Points are Better than Others* (https://arxiv.org/pdf/0908.3025.pdf)
        - K. Deb et al., *A fast and elitist multiobjective genetic algorithm: NSGA-II*

    Note:
        Deviations are ignored.

    Args:
        df: Metric dataframe.
        metrics: Metrics for dominance-based comparison. If ``None``, use all metrics in dataframe (except the column with the same name if it exists).
        tiebreak: How to break ties when rows have same rank. If ``None``, ranks correspond to each "peeled" Pareto set.

            - ``'crowding-distance'``: Crowding distance.
            - ``'mean-rank'``: Mean rank.

        ties: How to do rank numbering when there are ties.

            - ``'min'``: ``[1, 2, 2, 4]``-ranking
            - ``'max'``: ``[1, 3, 3, 4]``-ranking
            - ``'mean'``: ``[1, 2.5, 2.5, 4]``-ranking
            - ``'dense'``: ``[1, 2, 2, 3]``-ranking
            - ``'auto'``: ``'dense'`` if ``tiebreak`` is ``None`` else ``'min'``

        name: Name of metric.

    Returns:
        A copy of the original dataframe with an extra column indicating the non-dominated rank of each entry.
    """
    if "objective" not in df.attrs:
        raise ValueError("The DataFrame must have an 'objective' attribute.")

    if metrics is None:
        metrics = df.columns.get_level_values(0).unique().tolist()
        if name in metrics:
            metrics.remove(name)

    if len(metrics) == 0:
        raise ValueError("Empty list of metrics specified.")

    for metric in metrics:
        if metric not in df.columns.get_level_values(0):
            raise ValueError(f"'{metric}' is not a column in the DataFrame.")

    for metric in metrics:
        if df[metric]["value"].isna().any():
            raise ValueError(f"Metric {metric} contains NaN values which cannot be compared.")

    df = df.copy()

    objs = df.attrs["objective"]
    signs = {"maximize": -1, "minimize": 1}
    costs = np.column_stack([df[metric]["value"] * signs[objs[metric]] for metric in metrics])

    ranks = _non_dominated_rank(costs, tie_break=tiebreak, ties=ties)

    df[(name, "value")] = pd.Series(ranks, index=df.index)
    df[(name, "deviation")] = float("nan")

    df.attrs["objective"][name] = "minimize"

    return df


def extract_non_dominated(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    *,
    level: int = 0,
) -> pd.DataFrame:
    """Extract the non-dominated rows using one or more metrics.

    Same behavior as calling ``seqme.rank`` followed by ``seqme.top_k``, but improves the performance.

    Args:
        df: Metric dataframe.
        metrics: Metrics for dominance-based comparison. If ``None``, use all metrics in dataframe.
        level: The tuple index-names level to consider as a group.

    Returns:
        DataFrame: A subset of the metric dataframe with the non-dominated rows.
    """
    if level >= df.index.nlevels or level < 0:
        raise ValueError(f"Level should be in range [0;{df.index.nlevels - 1}].")

    if "objective" not in df.attrs:
        raise ValueError("The DataFrame must have an 'objective' attribute.")

    if metrics is None:
        metrics = df.columns.get_level_values(0).unique().tolist()

    if len(metrics) == 0:
        raise ValueError("Empty list of metrics specified.")

    for metric in metrics:
        if metric not in df.columns.get_level_values(0):
            raise ValueError(f"'{metric}' is not a column in the DataFrame.")

    for metric in metrics:
        if df[metric]["value"].isna().any():
            raise ValueError(f"Metric {metric} contains NaN values which cannot be compared.")

    df = df.copy()

    objs = df.attrs["objective"]
    signs = {"maximize": -1, "minimize": 1}

    def _extract_non_dominated(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
        costs = np.column_stack([df[metric]["value"] * signs[objs[metric]] for metric in metrics])
        non_dominated = moocore.is_nondominated(costs)
        return df[non_dominated]

    groups = defaultdict(list)
    for index in df.index:
        level_index = index[:level]
        groups[level_index].append(index)

    dfs_nd = []
    for group in groups.values():
        df_sub = df.loc[group]
        df_sub_best = _extract_non_dominated(df_sub, metrics)
        dfs_nd.append(df_sub_best)

    return pd.concat(dfs_nd)


def _non_dominated_rank(
    costs: np.ndarray,
    tie_break: Literal["crowding-distance", "mean-rank"] | None = None,
    ties: Literal["min", "max", "mean", "dense", "auto"] = "auto",
) -> np.ndarray:
    """
    Calculate the non-dominated rank of each observation.

    Args:
        costs: An array of costs (or objectives). The shape is (n_observations, n_objectives).
        tie_break: Whether we apply tie-break or not.
        ties: How to do rank numbering when there are ties.

    Returns:
        ranks:
            If not tie_break:
                The non-dominated rank of each observation. The shape is (n_observations, ). The rank starts from one and lower rank is better.
            else:
                The each non-dominated rank will be tie-broken so that we can sort identically.
                The shape is (n_observations, ) and the array is a permutation of zero to n_observations - 1.
    """
    ranks = moocore.pareto_rank(costs)

    if tie_break is None:
        pass
    elif tie_break == "crowding-distance":
        ranks = _crowding_distance_tie_break(costs, ranks)
    elif tie_break == "mean-rank":
        ranks = _mean_rank_tie_break(costs, ranks)
    else:
        raise ValueError(f"Unsupported tie-break: {tie_break}.")

    if ties == "auto":
        ranks = _dense_tied_ranks(ranks) if tie_break is None else _min_tied_ranks(ranks)
    elif ties == "min":
        ranks = _min_tied_ranks(ranks)
    elif ties == "max":
        ranks = _max_tied_ranks(ranks)
    elif ties == "mean":
        ranks = _mean_tied_ranks(ranks)
    elif ties == "dense":
        ranks = _dense_tied_ranks(ranks)
    else:
        raise ValueError(f"Invalid ties: {ties}.")

    return ranks + 1


def _crowding_distance_tie_break(costs: np.ndarray, nd_ranks: np.ndarray) -> np.ndarray:
    # K. Deb et al., "A fast and elitist multiobjective genetic algorithm: NSGA-II"

    ranks = scipy.stats.rankdata(costs, axis=0)
    masks: list[list[int]] = [[] for _ in range(nd_ranks.max() + 1)]

    for idx, nd_rank in enumerate(nd_ranks):
        masks[nd_rank].append(idx)

    n_checked = 0
    tie_broken_nd_ranks = np.zeros(ranks.shape[0], dtype=int)

    for mask in masks:
        tie_break_ranks = _compute_rank_based_crowding_distance(ranks=ranks[mask])
        tie_broken_nd_ranks[mask] = tie_break_ranks + n_checked - 1
        n_checked += len(mask)

    return tie_broken_nd_ranks


def _mean_rank_tie_break(costs: np.ndarray, nd_ranks: np.ndarray) -> np.ndarray:
    # David Come and Joshua Knowles, "Techniques for Highly Multiobjective Optimisation: Some Nondominated Points are Better than Others"
    # (https://arxiv.org/pdf/0908.3025.pdf)

    ranks = scipy.stats.rankdata(costs, axis=0)
    masks: list[list[int]] = [[] for _ in range(nd_ranks.max() + 1)]

    for idx, nd_rank in enumerate(nd_ranks):
        masks[nd_rank].append(idx)

    # min_ranks_factor plays a role when we tie-break same average ranks
    min_ranks_factor = np.min(ranks, axis=-1) / (nd_ranks.size**2 + 1)
    avg_ranks = np.mean(ranks, axis=-1) + min_ranks_factor

    n_checked = 0
    tie_broken_nd_ranks = np.zeros(ranks.shape[0], dtype=int)

    for mask in masks:
        tie_break_ranks = scipy.stats.rankdata(avg_ranks[mask], method="min").astype(int)
        tie_broken_nd_ranks[mask] = tie_break_ranks + n_checked - 1
        n_checked += len(mask)

    return tie_broken_nd_ranks


def _compute_rank_based_crowding_distance(ranks: np.ndarray) -> np.ndarray:
    n_observations, n_obj = ranks.shape
    order = np.argsort(ranks, axis=0)
    order_inv = np.zeros(order.shape[0], dtype=int)
    dists = np.zeros(n_observations)

    for i in range(n_obj):
        sorted_ranks = ranks[:, i][order[:, i]]
        order_inv[order[:, i]] = np.arange(n_observations)
        scale = sorted_ranks[-1] - sorted_ranks[0]
        crowding_dists = (
            np.hstack([np.inf, sorted_ranks[2:] - sorted_ranks[:-2], np.inf]) / scale
            if scale != 0
            else np.zeros(n_observations)
        )
        dists += crowding_dists[order_inv]
    return scipy.stats.rankdata(-dists, method="min").astype(int)


def _min_tied_ranks(ranks: np.ndarray) -> np.ndarray:
    """Rank [0, 1, 2, 3]."""
    return np.searchsorted(np.sort(ranks), ranks, side="left")


def _max_tied_ranks(ranks: np.ndarray) -> np.ndarray:
    """Rank [0, 2, 2, 3]."""
    return np.searchsorted(np.sort(ranks), ranks, side="right") - 1


def _mean_tied_ranks(ranks: np.ndarray) -> np.ndarray:
    """Rank [0, 1.5, 1.5, 3]."""
    idx = np.argsort(ranks)
    sorted_x = ranks[idx]

    _, inv_sorted = np.unique(sorted_x, return_inverse=True)
    sum_pos = np.bincount(inv_sorted, weights=np.arange(ranks.size))
    counts = np.bincount(inv_sorted)
    mean_pos_per_group = sum_pos / counts

    mean_sorted = mean_pos_per_group[inv_sorted]
    out = np.empty(ranks.size, dtype=float)
    out[idx] = mean_sorted
    return out


def _dense_tied_ranks(ranks: np.ndarray) -> np.ndarray:
    """Rank [0, 1, 1, 2]."""
    return np.unique(ranks, return_inverse=True)[1]
