from typing import Literal

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

    ranks = non_dominated_rank(costs, tie_break=tiebreak, ties=ties)

    df[(name, "value")] = pd.Series(ranks, index=df.index)
    df[(name, "deviation")] = float("nan")

    df.attrs["objective"][name] = "minimize"

    return df


# Adapted from https://github.com/nabenabe0928/fast-pareto
# Distributed under Apache-2.0: https://github.com/nabenabe0928/fast-pareto/blob/main/LICENSE


def non_dominated_rank(
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
    unique_costs, order_inv = np.unique(costs, axis=0, return_inverse=True)
    ranks = _non_dominated_rank(costs=unique_costs)[order_inv.flatten()]

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


def is_pareto_front(costs: np.ndarray, assume_unique_lexsorted: bool = False) -> np.ndarray:
    """Determine the Pareto front from a provided set of costs.

    The time complexity is O(N (log N)^(M - 2)) for M > 3 and O(N log N) for M = 2, 3 where
    N is n_observations and M is n_objectives. (Kung's algorithm).

    Args:
        costs: An array of costs (or objectives). The shape is (n_observations, n_objectives).
        assume_unique_lexsorted: Whether to assume the unique lexsorted costs or not. Basically, we omit np.unique(costs, axis=0) if True.

    Returns:
        on_front: Whether the solution is on the Pareto front. Each element is True or False and the shape is (n_observations, ).

    Note:
        f dominates g if and only if:
            1. f[i] <= g[i] for all i, and 2. f[i] < g[i] for some i
        g is not dominated by f if and only if:
            1. f[i] > g[i] for some i, or 2. f[i] == g[i] for all i
    """
    if assume_unique_lexsorted:
        costs, order_inv = np.unique(costs, axis=0, return_inverse=True)

    on_front = _is_pareto_front_2d(costs) if costs.shape[-1] == 2 else _is_pareto_front_nd(costs)
    return on_front[order_inv] if assume_unique_lexsorted else on_front


def _is_pareto_front_2d(costs: np.ndarray) -> np.ndarray:
    n_observations = costs.shape[0]
    cummin_value1 = np.minimum.accumulate(costs[:, 1])
    on_front = np.ones(n_observations, dtype=bool)
    on_front[1:] = cummin_value1[1:] < cummin_value1[:-1]  # True if cummin value1 is new minimum.
    return on_front


def _is_pareto_front_nd(costs: np.ndarray) -> np.ndarray:
    n_observations = costs.shape[0]
    on_front = np.zeros(n_observations, dtype=bool)
    nondominated_indices = np.arange(n_observations)
    while len(costs) > 0:
        # The following judges `np.any(costs[i] < costs[0])` for each `i`.
        nondominated_and_not_top = np.any(costs < costs[0], axis=1)
        # NOTE: trials[j] cannot dominate trials[i] for i < j because of lexsort. Therefore, nondominated_indices[0] is always non-dominated.
        on_front[nondominated_indices[0]] = True
        costs = costs[nondominated_and_not_top]
        nondominated_indices = nondominated_indices[nondominated_and_not_top]

    return on_front


def _non_dominated_rank(costs: np.ndarray) -> np.ndarray:
    n_observations, n_obj = costs.shape
    if n_obj == 1:
        return np.unique(costs[:, 0], return_inverse=True)[1]

    ranks = np.zeros(n_observations, dtype=int)
    rank = 0
    indices = np.arange(n_observations)
    while indices.size > 0:
        on_front = is_pareto_front(costs, assume_unique_lexsorted=True)
        ranks[indices[on_front]] = rank
        indices, costs = indices[~on_front], costs[~on_front]
        rank += 1

    return ranks


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
