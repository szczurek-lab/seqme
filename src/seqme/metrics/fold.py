from typing import Literal

import numpy as np

from seqme.core.base import Metric, MetricResult


class Fold(Metric):
    """A cross-validation wrapper for any Metric.

    Fold splits the data into k-folds or fixed-size splits, with optional shuffling, and then aggregates the results.
    """

    def __init__(
        self,
        metric: Metric,
        *,
        deviation: Literal["se", "std", "var"] = "se",
        estimate: Literal["biased", "unbiased"] = "unbiased",
        n_splits: int | None = None,
        split_size: int | None = None,
        drop_last: bool = False,
        strict: bool = True,
        shuffle: bool = False,
        seed: int = 0,
    ):
        """
        Initialize the Fold wrapper.

        Args:
            metric: The underlying metric to evaluate per fold.
            deviation: Type of deviation to compute:

                - ``'se'``: Standard error
                - ``'std'``: Standard deviation
                - ``'var'``: Variance

            estimate: How to estimate the deviation.
            n_splits: Number of folds to create (exclusive with ``split_size``).
            split_size: Fixed size for each fold (exclusive with ``n_splits``).
            drop_last: Drop final fold if smaller than ``split_size``.
            strict: Error on any non-null fold deviation.
            shuffle: Shuffle data before splitting.
            seed: Seed for deterministic shuffling of sequences when creating folds.
        """
        self.metric = metric
        self.deviation = deviation
        self.estimate = estimate
        self.n_splits = n_splits
        self.split_size = split_size
        self.drop_last = drop_last
        self.strict = strict
        self.shuffle = shuffle
        self.seed = seed

        if (self.n_splits is not None) and (self.split_size is not None):
            raise ValueError("Only one of n_splits or split_size may be specified.")
        if (self.n_splits is None) and (self.split_size is None):
            raise ValueError("One of n_splits or split_size must be specified.")

        if (self.n_splits is not None) and (self.n_splits < 2):
            raise ValueError("Expected n_splits >= 2.")
        if (self.split_size is not None) and (self.split_size <= 0):
            raise ValueError("Expected split_size > 0.")

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Call the wrapped metric on each fold of ``sequences`` and aggregate the results.

        Args:
            sequences: Sequences to split into folds.

        Returns:
            MetricResult: Aggregated mean value and standard error, standard error or variance across folds.
        """
        n = len(sequences)
        indices = np.arange(n)

        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        # Determine folds
        if self.n_splits is not None:
            if self.n_splits > n:
                raise ValueError(f"Cannot split into {self.n_splits} folds with only {n} sequences.")
            raw_folds = np.array_split(indices, self.n_splits)
        else:
            raw_folds = [indices[i : i + self.split_size] for i in range(0, n, self.split_size)]
            if self.drop_last and raw_folds and len(raw_folds[-1]) < self.split_size:
                raw_folds = raw_folds[:-1]

            if self.drop_last and len(raw_folds) == 0:
                raise ValueError(
                    f"With drop_last=True, cannot form any fold of size {self.split_size} from {n} sequences."
                )

        results = []
        for fold_idx in raw_folds:
            idx_list = fold_idx.tolist()
            result = self.metric([sequences[i] for i in idx_list])

            if self.strict and (result.deviation is not None):
                raise ValueError("Fold result has non-null deviation in strict mode.")

            results.append(result)

        values = np.array([result.value for result in results], float)

        if len(results) > 1:
            if self.estimate == "biased":
                ddof = 0
            elif self.estimate == "unbiased":
                ddof = 1
            else:
                raise ValueError(f"Invalid estimate: {self.estimate}")

            if self.deviation == "std":
                deviation = float(values.std(ddof=ddof))
            elif self.deviation == "var":
                deviation = float(values.var(ddof=ddof))
            elif self.deviation == "se":
                deviation = float(values.std(ddof=ddof)) / (len(values) ** 0.5)
            else:
                raise ValueError(f"Invalid deviation: {self.deviation}")
        else:
            assert len(results) == 1
            deviation = results[0].deviation

        return MetricResult(value=values.mean().item(), deviation=deviation)

    @property
    def name(self) -> str:
        return self.metric.name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return self.metric.objective
