from collections.abc import Callable
from typing import Literal

import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity

from seqme.core.base import Metric, MetricResult


class ConformityScore(Metric):
    """
    Distributional conformity score.

    Reference:
        Frey et al., "Protein Discovery With Discrete Walk-jump Sampling" (2024).
        (https://arxiv.org/abs/2306.12360)
    """

    def __init__(
        self,
        reference: list[str],
        predictors: list[Callable[[list[str]], np.ndarray]],
        *,
        n_splits: int = 5,
        kde_bandwidth: float | Literal["scott", "silverman"] = "silverman",
        seed: int = 0,
        name: str = "Conformity score",
    ):
        """
        Initialize the metric.

        Args:
            reference: Reference sequences assumed to represent the target distribution.
            predictors: A list of predictor functions. Each should take a list of sequences and return a 1D NumPy array of features.
            n_splits: Number of cross-validation folds for KDE.
            kde_bandwidth: Bandwidth parameter for the Gaussian KDE.
            seed: Seed for deterministic k-fold shuffling.
            name: Metric name.
        """
        if n_splits < 2:
            raise ValueError("Number of cross-validation folds for KDE (n_splits) must be at least 2.")

        self.reference = reference
        self.predictors = predictors
        self._name = name

        reference_arr = self._sequences_to_predictors(self.reference)  # (n_ref, n_descs)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        self.ref_log_prob_per_split = [
            self._fit_and_score_reference(
                train=reference_arr[train_idx], val=reference_arr[val_idx], kde_bandwidth=kde_bandwidth
            )
            for train_idx, val_idx in kf.split(reference_arr)
        ]  # list of (validation log probabilities, fitted KDE) per split

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Compute the conformity score for the given sequences.

        Args:
            sequences: Sequences to evaluate.

        Returns:
            MetricResult: Mean and standard error of the conformity scores across all folds.
        """
        seqs_predictors = self._sequences_to_predictors(sequences)  # (n_gen, n_descs)
        conformity_scores = self._compute_conformity_score(seqs_predictors)
        return MetricResult(
            float(np.mean(conformity_scores)),
            float(np.std(conformity_scores)) / (len(conformity_scores) ** 0.5),
        )

    def _sequences_to_predictors(self, sequences: list[str]) -> np.ndarray:
        return np.stack([desc_func(sequences) for desc_func in self.predictors], axis=1)

    def _fit_and_score_reference(
        self,
        train: np.ndarray,
        val: np.ndarray,
        kde_bandwidth: float | Literal["scott", "silverman"],
    ) -> tuple[np.ndarray, KernelDensity]:
        kde = KernelDensity(kernel="gaussian", bandwidth=kde_bandwidth)
        kde.fit(train)
        log_prob_val = kde.score_samples(val)
        return log_prob_val, kde

    def _compute_conformity_score(self, generated_arr: np.ndarray) -> list[float]:
        scores = []
        for log_prob_val, kde in self.ref_log_prob_per_split:
            n_val = log_prob_val.shape[0]
            log_prob_generated = kde.score_samples(generated_arr)
            comp_matrix = log_prob_generated[:, None] >= log_prob_val[None, :]
            counts = comp_matrix.sum(axis=1)
            scores.append((counts / (n_val + 1)).mean())
        return scores

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"
